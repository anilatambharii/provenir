"""The ``import provenir`` viral substrate: a 3-line wrapper.

Drop provenance, trustworthy eval, reward-hacking detection, and a signed
Model Passport into *any* existing training loop (TRL / verl / Unsloth /
custom) with almost no code::

    import provenir
    from provenir.integrations import track

    with track("my-grpo-run", dataset=ds, base_model="Qwen2.5-7B") as run:
        for step, batch in enumerate(training_loop()):
            run.log_step({"kl": batch.kl, "entropy": batch.entropy})
        run.record_eval("gsm8k", score=0.71)

    run.manifest       # a content-addressed RunManifest, written to disk
    run.hacking_report # reward-hacking findings across buffered trajectories
    run.passport       # an optionally-signed ModelPassport

The :class:`ProvenirRun` context manager composes the existing Provenir
building blocks — environment fingerprinting, the RL flight recorder, the
reward-hacking detector, a lineage DAG, a Model Bill-of-Materials, and the
passport signer — and persists every artifact under ``output_dir`` on exit.

Everything is deterministic when a ``timestamp`` is injected: no clocks or
randomness are consulted for content that must hash stably.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, TypeVar

from provenir.core.abstractions import RunManifest
from provenir.core.manifest import RunManifestStore
from provenir.data.dataset import JsonlDataset
from provenir.eval.harness import MultiMetricEvaluator
from provenir.governance.bom import (
    CodeComponent,
    DataComponent,
    EvalComponent,
    ModelBOM,
)
from provenir.governance.passport import ModelPassport, PassportSigner
from provenir.observability import (
    FlightRecorder,
    HackingReport,
    RewardHackingDetector,
    RLStepMetrics,
)
from provenir.provenance import (
    LineageEdge,
    LineageGraph,
    LineageNode,
    capture_fingerprint,
)

_F = TypeVar("_F", bound=Callable[..., Any])

#: Fields of :class:`RLStepMetrics` we accept from a plain ``dict`` step log.
_STEP_FIELDS: frozenset[str] = frozenset(
    {
        "step",
        "kl",
        "entropy",
        "reward_mean",
        "reward_std",
        "response_length_mean",
        "advantage_std",
        "grad_norm",
        "learning_rate",
    }
)

_EMPTY_HASH = hashlib.sha256(b"").hexdigest()


@dataclass(frozen=True)
class TrackingConfig:
    """Immutable configuration for a :class:`ProvenirRun`.

    Args:
        name: Human-readable run name; also the BOM ``model_id``. Required.
        base_model: The base model the run fine-tunes.
        output_dir: Directory root under which every artifact is written.
        seed: The run's random seed, recorded in the manifest.
        sign_passport: When True, a signed :class:`ModelPassport` is issued.
        signing_key: HMAC key used when ``sign_passport`` is True.
        capture_env: When True, an environment fingerprint is captured.

    Example:
        >>> TrackingConfig(name="run-1").base_model
        'unknown'
        >>> TrackingConfig(name="run-1", sign_passport=True, signing_key=b"k").seed
        0
    """

    name: str
    base_model: str = "unknown"
    output_dir: str = "artifacts/provenir"
    seed: int = 0
    sign_passport: bool = False
    signing_key: bytes = b""
    capture_env: bool = True

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name must be non-empty")
        if self.sign_passport and not self.signing_key:
            raise ValueError("signing_key must be non-empty when sign_passport is True")


@dataclass
class _RecordedEval:
    """A single benchmark result buffered for the BOM and lineage graph."""

    benchmark: str
    score: float
    contaminated: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


class ProvenirRun:
    """A context manager that instruments any training run.

    Enter the context to capture the environment, hash the config and dataset,
    and spin up a flight recorder plus a reward-hacking detector. Inside the
    ``with`` block, forward per-step metrics via :meth:`log_step`, buffer
    trajectories via :meth:`log_trajectory`, and record benchmark results via
    :meth:`record_eval`. On a clean exit the run is finalized: a
    :class:`RunManifest`, a :class:`LineageGraph`, the flight-recorder history,
    a :class:`HackingReport`, a :class:`ModelBOM`, and an optionally-signed
    :class:`ModelPassport` are assembled and written under ``output_dir``.

    If an exception propagates through the block a *partial* manifest is still
    written (best-effort) and the exception is re-raised.

    Example:
        >>> import tempfile
        >>> cfg = TrackingConfig(name="demo", output_dir=tempfile.mkdtemp())
        >>> with ProvenirRun(cfg, config_payload={"lr": 0.001}) as run:
        ...     run.log_step({"step": 0, "kl": 0.02, "entropy": 1.4})
        >>> run.manifest is not None
        True
        >>> run.passport is None
        True
    """

    def __init__(
        self,
        config: TrackingConfig,
        config_payload: dict[str, Any] | None = None,
        dataset: JsonlDataset | None = None,
        timestamp: str = "",
    ) -> None:
        self.config = config
        self.config_payload: dict[str, Any] = dict(config_payload or {})
        self.dataset = dataset
        self.timestamp = timestamp

        self._output_root = Path(config.output_dir)
        self._config_hash = _hash_payload(self.config_payload)
        self._dataset_hash = dataset.hash() if dataset is not None else _EMPTY_HASH
        self._dataset_records = len(dataset.records) if dataset is not None else 0

        self._flight_recorder = FlightRecorder()
        self._detector = RewardHackingDetector()
        self._trajectories: list[Mapping[str, Any]] = []
        self._evals: list[_RecordedEval] = []
        self._env_dict: dict[str, str] = {}
        self._hardware_fingerprint = ""

        # Populated on finalize.
        self._manifest: RunManifest | None = None
        self._hacking_report: HackingReport | None = None
        self._lineage: LineageGraph | None = None
        self._bom: ModelBOM | None = None
        self._passport: ModelPassport | None = None
        self._finalized = False

    # -- context management ------------------------------------------------

    def __enter__(self) -> ProvenirRun:
        if self.config.capture_env:
            fingerprint = capture_fingerprint()
            self._env_dict = fingerprint.to_dict()
            self._hardware_fingerprint = fingerprint.packages_hash
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: Any,
    ) -> None:
        if exc_type is None:
            self._finalize(partial=False)
        else:
            # Best-effort partial manifest; returning None re-raises the error.
            try:
                self._finalize(partial=True)
            except Exception:  # pragma: no cover - never mask the original error
                pass

    # -- instrumentation ---------------------------------------------------

    def log_step(self, metrics: dict[str, float] | RLStepMetrics) -> None:
        """Forward one training step's metrics to the flight recorder.

        Accepts either an :class:`RLStepMetrics` or a plain ``dict`` (unknown
        keys are ignored; a missing ``step`` defaults to the running index).
        """
        if isinstance(metrics, RLStepMetrics):
            record = metrics
        else:
            known: dict[str, Any] = {
                key: value for key, value in metrics.items() if key in _STEP_FIELDS
            }
            known.setdefault("step", len(self._flight_recorder.history))
            record = RLStepMetrics(**known)
        self._flight_recorder.log_step(record)

    def log_trajectory(self, trajectory: dict[str, Any]) -> None:
        """Buffer a rollout ``trajectory`` for batch reward-hacking detection."""
        self._trajectories.append(dict(trajectory))

    def record_eval(
        self,
        benchmark: str,
        predictions: list[str] | None = None,
        eval_dataset: JsonlDataset | None = None,
        score: float | None = None,
        contaminated: bool = False,
    ) -> None:
        """Record a benchmark result, by explicit score or by evaluation.

        Provide either an explicit ``score`` or both ``predictions`` and an
        ``eval_dataset`` (scored with :class:`MultiMetricEvaluator`, using the
        ``exact_match``/accuracy summary). Raises ``ValueError`` if neither is
        supplied.
        """
        if not benchmark:
            raise ValueError("benchmark must be non-empty")
        metadata: dict[str, Any] = {}
        if predictions is not None and eval_dataset is not None:
            result = MultiMetricEvaluator().evaluate(eval_dataset, predictions)
            summary = result.accuracy
            resolved = summary.mean if summary is not None else 0.0
            metadata = result.metadata
        elif score is not None:
            resolved = float(score)
        else:
            raise ValueError(
                "record_eval requires either an explicit score or "
                "predictions plus eval_dataset"
            )
        self._evals.append(
            _RecordedEval(
                benchmark=benchmark,
                score=resolved,
                contaminated=contaminated,
                metadata=metadata,
            )
        )

    # -- finalization ------------------------------------------------------

    def _finalize(self, partial: bool) -> None:
        if self._finalized:
            return

        run_id = self.config.name
        self._hacking_report = self._detector.detect_batch(self._trajectories)
        self._lineage = self._build_lineage(run_id)
        self._bom = self._build_bom(run_id)
        self._manifest = self._build_manifest(run_id, partial=partial)

        manifests_dir = self._output_root / "manifests"
        RunManifestStore(manifests_dir).save(self._manifest)

        self._output_root.mkdir(parents=True, exist_ok=True)
        _write_json(self._output_root / "lineage.json", self._lineage.to_dict())
        _write_json(
            self._output_root / "flight_recorder.json", self._flight_recorder.to_dict()
        )
        _write_json(
            self._output_root / "hacking_report.json", self._hacking_report.to_dict()
        )
        _write_json(self._output_root / "bom.json", self._bom.to_dict())

        if self.config.sign_passport:
            signer = PassportSigner(self.config.signing_key)
            self._passport = signer.sign(self._bom, signed_at=self.timestamp)
            (self._output_root / "passport.md").write_text(
                self._passport.to_markdown(), encoding="utf-8"
            )
            (self._output_root / "passport.json").write_text(
                self._passport.to_json(), encoding="utf-8"
            )

        self._finalized = True

    def _build_lineage(self, run_id: str) -> LineageGraph:
        graph = LineageGraph()
        dataset_node = f"dataset:{self._dataset_hash}"
        run_node = f"run:{run_id}"
        graph.add_node(
            LineageNode(
                node_id=dataset_node,
                node_type="dataset",
                content_hash=self._dataset_hash,
                attributes={"num_records": self._dataset_records},
            )
        )
        graph.add_node(
            LineageNode(
                node_id=run_node,
                node_type="run",
                content_hash=self._config_hash,
                attributes={
                    "seed": self.config.seed,
                    "base_model": self.config.base_model,
                    "config_hash": self._config_hash,
                },
            )
        )
        graph.add_edge(
            LineageEdge(parent_id=dataset_node, child_id=run_node, relation="trained_on")
        )
        for index, recorded in enumerate(self._evals):
            eval_node = f"eval:{run_id}:{index}"
            graph.add_node(
                LineageNode(
                    node_id=eval_node,
                    node_type="eval",
                    content_hash=_hash_payload(
                        {"benchmark": recorded.benchmark, "score": recorded.score}
                    ),
                    attributes={
                        "benchmark": recorded.benchmark,
                        "score": recorded.score,
                    },
                )
            )
            graph.add_edge(
                LineageEdge(
                    parent_id=run_node, child_id=eval_node, relation="evaluated_by"
                )
            )
        return graph

    def _build_bom(self, run_id: str) -> ModelBOM:
        data = [
            DataComponent(
                name="train",
                content_hash=self._dataset_hash,
                num_records=self._dataset_records,
            )
        ]
        code = CodeComponent(
            git_sha=self._env_dict.get("platform") or "unknown",
            dependencies_hash=self._hardware_fingerprint or _EMPTY_HASH,
            framework="provenir",
        )
        evals = [
            EvalComponent(
                benchmark=recorded.benchmark,
                score=recorded.score,
                contaminated=recorded.contaminated,
            )
            for recorded in self._evals
        ]
        return ModelBOM(
            model_id=self.config.name,
            base_model=self.config.base_model,
            run_id=run_id,
            data=data,
            code=code,
            evals=evals,
            hyperparameters=dict(self.config_payload),
            created_at=self.timestamp,
        )

    def _build_manifest(self, run_id: str, partial: bool) -> RunManifest:
        provenance: dict[str, Any] = {
            "environment": self._env_dict,
            "base_model": self.config.base_model,
            "flight_recorder": self._flight_recorder.summary(),
            "hacking_rate": (
                self._hacking_report.hacking_rate if self._hacking_report else 0.0
            ),
            "num_evals": len(self._evals),
            "partial": partial,
        }
        return RunManifest(
            run_id=run_id,
            config_hash=self._config_hash,
            dataset_hash=self._dataset_hash,
            seed=self.config.seed,
            hardware_fingerprint=self._hardware_fingerprint,
            metrics_history=[m.to_dict() for m in self._flight_recorder.history],
            provenance=provenance,
        )

    # -- results (available after exit) ------------------------------------

    @property
    def manifest(self) -> RunManifest | None:
        """The assembled :class:`RunManifest` (``None`` before finalization)."""
        return self._manifest

    @property
    def flight_recorder(self) -> FlightRecorder:
        """The live :class:`FlightRecorder` fed by :meth:`log_step`."""
        return self._flight_recorder

    @property
    def hacking_report(self) -> HackingReport | None:
        """The :class:`HackingReport` over buffered trajectories (after exit)."""
        return self._hacking_report

    @property
    def lineage(self) -> LineageGraph | None:
        """The :class:`LineageGraph` linking dataset -> run -> evals."""
        return self._lineage

    @property
    def bom(self) -> ModelBOM | None:
        """The assembled :class:`ModelBOM` (``None`` before finalization)."""
        return self._bom

    @property
    def passport(self) -> ModelPassport | None:
        """The signed :class:`ModelPassport`, or ``None`` when unsigned."""
        return self._passport

    @property
    def anomalies(self) -> list[Any]:
        """Every anomaly the flight recorder detected so far."""
        return self._flight_recorder.anomalies


def track(name: str, **kwargs: Any) -> ProvenirRun:
    """Open a :class:`ProvenirRun` with a one-liner.

    Any :class:`TrackingConfig` field can be passed as a keyword; the special
    keywords ``config_payload``, ``dataset``, and ``timestamp`` are forwarded
    to the :class:`ProvenirRun` constructor.

    Example:
        >>> import tempfile
        >>> with track("run-1", output_dir=tempfile.mkdtemp()) as run:
        ...     run.log_step({"kl": 0.01})
        >>> run.manifest is not None
        True
    """
    config_payload = kwargs.pop("config_payload", None)
    dataset = kwargs.pop("dataset", None)
    timestamp = kwargs.pop("timestamp", "")
    config = TrackingConfig(name=name, **kwargs)
    return ProvenirRun(
        config,
        config_payload=config_payload,
        dataset=dataset,
        timestamp=timestamp,
    )


def provenance_tracked(name: str, **cfg_kwargs: Any) -> Callable[[_F], _F]:
    """Decorate ``fn(run, *args, **kwargs)`` to run inside a :class:`ProvenirRun`.

    The decorated function receives the open :class:`ProvenirRun` as its first
    positional argument and its return value is passed straight through.

    Example:
        >>> import tempfile
        >>> @provenance_tracked("run-1", output_dir=tempfile.mkdtemp())
        ... def train(run, epochs):
        ...     run.log_step({"kl": 0.02})
        ...     return epochs
        >>> train(3)
        3
    """

    def decorator(fn: _F) -> _F:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with track(name, **cfg_kwargs) as run:
                return fn(run, *args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator


def _hash_payload(payload: dict[str, Any]) -> str:
    """Return a stable sha256 over ``payload`` serialized as sorted JSON."""
    serialized = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _write_json(path: Path, data: dict[str, Any]) -> None:
    """Write ``data`` to ``path`` as indented JSON (creating parents)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


__all__ = [
    "TrackingConfig",
    "ProvenirRun",
    "track",
    "provenance_tracked",
]
