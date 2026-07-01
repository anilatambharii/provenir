from __future__ import annotations

import math
import statistics
from dataclasses import asdict, dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Per-step metrics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RLStepMetrics:
    """A single training step's telemetry from an RL run.

    All fields except ``step`` are optional and default to ``0.0`` (or ``None``
    for a learning rate that was not recorded).  Only populate what your trainer
    exposes; the recorder tolerates missing signals.

    Example::

        m = RLStepMetrics(step=1, kl=0.02, entropy=1.4, reward_mean=0.3)
        assert m.to_dict()["step"] == 1
    """

    step: int
    kl: float = 0.0
    entropy: float = 0.0
    reward_mean: float = 0.0
    reward_std: float = 0.0
    response_length_mean: float = 0.0
    advantage_std: float = 0.0
    grad_norm: float = 0.0
    learning_rate: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain ``dict`` (JSON-friendly)."""
        return asdict(self)


# ---------------------------------------------------------------------------
# Anomaly record
# ---------------------------------------------------------------------------

#: Valid anomaly ``kind`` values emitted by :class:`FlightRecorder`.
#:
#: * ``kl_blowup`` — KL divergence exceeded ``kl_max`` (policy diverging).
#: * ``kl_collapse`` — KL fell below ``kl_min`` (policy frozen / no learning).
#: * ``entropy_collapse`` — entropy below floor or sharply dropped vs its max.
#: * ``length_explosion`` — mean response length blew up vs the baseline step.
#: * ``advantage_collapse`` — GRPO advantage std ~ 0 (degenerate group).
#: * ``reward_std_collapse`` — reward std ~ 0 (mode collapse).
#: * ``reward_spike`` — reward mean is a z-score outlier (possible hacking).
#: * ``grad_explosion`` — gradient norm exceeded ``grad_norm_max``.
ANOMALY_KINDS: tuple[str, ...] = (
    "kl_blowup",
    "kl_collapse",
    "entropy_collapse",
    "length_explosion",
    "advantage_collapse",
    "reward_std_collapse",
    "reward_spike",
    "grad_explosion",
)

_VALID_SEVERITIES: frozenset[str] = frozenset({"warn", "critical"})


@dataclass(frozen=True)
class Anomaly:
    """A single detected anomaly at a training step.

    Example::

        a = Anomaly(step=5, kind="kl_blowup", severity="critical",
                    detail="KL 0.9 > 0.5", value=0.9)
        assert a.severity == "critical"
    """

    step: int
    kind: str
    severity: str
    detail: str
    value: float

    def __post_init__(self) -> None:
        if self.severity not in _VALID_SEVERITIES:
            raise ValueError(
                f"severity must be one of {sorted(_VALID_SEVERITIES)}, got {self.severity!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain ``dict`` (JSON-friendly)."""
        return asdict(self)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FlightRecorderConfig:
    """Detector thresholds for the :class:`FlightRecorder`.

    Every threshold is documented on the corresponding detector.  All values
    must be strictly positive (validated in ``__post_init__``).

    Example::

        cfg = FlightRecorderConfig(kl_max=0.3, window=50)
        assert cfg.window == 50
    """

    kl_max: float = 0.5
    kl_min: float = 1e-6
    entropy_min: float = 0.1
    entropy_drop_frac: float = 0.5
    length_explosion_ratio: float = 3.0
    advantage_std_min: float = 1e-4
    reward_std_min: float = 1e-4
    grad_norm_max: float = 100.0
    reward_spike_z: float = 4.0
    window: int = 20

    def __post_init__(self) -> None:
        positives: dict[str, float] = {
            "kl_max": self.kl_max,
            "kl_min": self.kl_min,
            "entropy_min": self.entropy_min,
            "entropy_drop_frac": self.entropy_drop_frac,
            "length_explosion_ratio": self.length_explosion_ratio,
            "advantage_std_min": self.advantage_std_min,
            "reward_std_min": self.reward_std_min,
            "grad_norm_max": self.grad_norm_max,
            "reward_spike_z": self.reward_spike_z,
        }
        for name, value in positives.items():
            if value <= 0.0:
                raise ValueError(f"{name} must be > 0, got {value}")
        if self.window < 2:
            raise ValueError(f"window must be >= 2, got {self.window}")


# ---------------------------------------------------------------------------
# The black box
# ---------------------------------------------------------------------------


@dataclass
class _RunningState:
    """Mutable running aggregates the detectors need across steps."""

    baseline_length: float | None = None
    entropy_max: float = 0.0
    reward_history: list[float] = field(default_factory=list)


class FlightRecorder:
    """The "black box" flight recorder for an RL run.

    Feed one :class:`RLStepMetrics` per training step to :meth:`log_step`; it
    appends to the run history, evaluates every detector against the configured
    thresholds plus rolling history, and returns the anomalies detected *this*
    step.  Attach :meth:`to_dict` to a run manifest for post-mortem analysis.

    Usage inside a training loop::

        rec = FlightRecorder()
        for step, batch in enumerate(training_loop()):
            new = rec.log_step(RLStepMetrics(step=step, kl=batch.kl, ...))
            if any(a.severity == "critical" for a in new):
                print(rec.health_report())

    Detectors (see :data:`ANOMALY_KINDS`): KL blowup/collapse, entropy collapse
    (floor or sharp drop vs running max), response-length explosion vs baseline,
    GRPO advantage collapse (``advantage_std`` ~ 0), reward-std collapse, reward
    spike (z-score over the rolling window), and gradient explosion.
    """

    def __init__(self, config: FlightRecorderConfig | None = None) -> None:
        self.config = config or FlightRecorderConfig()
        self._history: list[RLStepMetrics] = []
        self._anomalies: list[Anomaly] = []
        self._state = _RunningState()

    # -- public API --------------------------------------------------------

    def log_step(self, metrics: RLStepMetrics) -> list[Anomaly]:
        """Record *metrics*, run all detectors, and return this step's anomalies."""
        self._history.append(metrics)
        found: list[Anomaly] = []

        self._check_kl(metrics, found)
        self._check_entropy(metrics, found)
        self._check_length(metrics, found)
        self._check_advantage(metrics, found)
        self._check_reward_std(metrics, found)
        self._check_reward_spike(metrics, found)
        self._check_grad(metrics, found)

        # Update rolling reward history *after* the spike check so the current
        # step is compared against prior steps only.
        self._state.reward_history.append(metrics.reward_mean)
        if len(self._state.reward_history) > self.config.window:
            self._state.reward_history = self._state.reward_history[-self.config.window :]

        self._anomalies.extend(found)
        return found

    @property
    def history(self) -> list[RLStepMetrics]:
        """A copy of the recorded per-step metrics, in order."""
        return list(self._history)

    @property
    def anomalies(self) -> list[Anomaly]:
        """A copy of every anomaly detected so far, in order."""
        return list(self._anomalies)

    def summary(self) -> dict[str, Any]:
        """Counts by kind/severity, final metrics, and a health verdict."""
        by_kind: dict[str, int] = {}
        by_severity: dict[str, int] = {}
        for a in self._anomalies:
            by_kind[a.kind] = by_kind.get(a.kind, 0) + 1
            by_severity[a.severity] = by_severity.get(a.severity, 0) + 1
        final = self._history[-1].to_dict() if self._history else None
        return {
            "num_steps": len(self._history),
            "num_anomalies": len(self._anomalies),
            "by_kind": by_kind,
            "by_severity": by_severity,
            "final_metrics": final,
            "verdict": self._verdict(),
        }

    def health_report(self) -> str:
        """Human-readable multiline health assessment."""
        verdict = self._verdict()
        lines = [f"Flight Recorder: {verdict}"]
        lines.append(f"  steps recorded : {len(self._history)}")
        lines.append(f"  anomalies      : {len(self._anomalies)}")
        if not self._anomalies:
            lines.append("  no detectors fired; run looks healthy.")
            return "\n".join(lines)

        by_kind: dict[str, int] = {}
        for a in self._anomalies:
            by_kind[a.kind] = by_kind.get(a.kind, 0) + 1
        lines.append("  detectors fired:")
        for kind in sorted(by_kind):
            lines.append(f"    - {kind}: {by_kind[kind]}")
        critical = [a for a in self._anomalies if a.severity == "critical"]
        if critical:
            last = critical[-1]
            lines.append(f"  latest critical: step {last.step} {last.kind} — {last.detail}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Full serialization: config, history, anomalies, and summary."""
        return {
            "config": asdict(self.config),
            "history": [m.to_dict() for m in self._history],
            "anomalies": [a.to_dict() for a in self._anomalies],
            "summary": self.summary(),
        }

    # -- verdict -----------------------------------------------------------

    def _verdict(self) -> str:
        if any(a.severity == "critical" for a in self._anomalies):
            return "CRITICAL"
        if self._anomalies:
            return "DEGRADED"
        return "HEALTHY"

    # -- detectors ---------------------------------------------------------

    def _check_kl(self, m: RLStepMetrics, out: list[Anomaly]) -> None:
        cfg = self.config
        if m.kl > cfg.kl_max:
            out.append(
                Anomaly(
                    step=m.step,
                    kind="kl_blowup",
                    severity="critical",
                    detail=f"KL {m.kl:.4g} > kl_max {cfg.kl_max:.4g}; policy diverging",
                    value=m.kl,
                )
            )
        elif 0.0 < m.kl < cfg.kl_min:
            out.append(
                Anomaly(
                    step=m.step,
                    kind="kl_collapse",
                    severity="warn",
                    detail=f"KL {m.kl:.4g} < kl_min {cfg.kl_min:.4g}; policy barely moving",
                    value=m.kl,
                )
            )

    def _check_entropy(self, m: RLStepMetrics, out: list[Anomaly]) -> None:
        cfg = self.config
        state = self._state
        if m.entropy > state.entropy_max:
            state.entropy_max = m.entropy

        # Sharp drop relative to the running max is the classic collapse.
        if state.entropy_max > 0.0 and m.entropy < cfg.entropy_drop_frac * state.entropy_max:
            out.append(
                Anomaly(
                    step=m.step,
                    kind="entropy_collapse",
                    severity="critical",
                    detail=(
                        f"entropy {m.entropy:.4g} < {cfg.entropy_drop_frac:.2g} * "
                        f"running max {state.entropy_max:.4g}; distribution collapsing"
                    ),
                    value=m.entropy,
                )
            )
        elif 0.0 < m.entropy < cfg.entropy_min:
            out.append(
                Anomaly(
                    step=m.step,
                    kind="entropy_collapse",
                    severity="warn",
                    detail=f"entropy {m.entropy:.4g} < entropy_min {cfg.entropy_min:.4g}",
                    value=m.entropy,
                )
            )

    def _check_length(self, m: RLStepMetrics, out: list[Anomaly]) -> None:
        cfg = self.config
        state = self._state
        if state.baseline_length is None:
            if m.response_length_mean > 0.0:
                state.baseline_length = m.response_length_mean
            return
        limit = state.baseline_length * cfg.length_explosion_ratio
        if m.response_length_mean > limit:
            out.append(
                Anomaly(
                    step=m.step,
                    kind="length_explosion",
                    severity="warn",
                    detail=(
                        f"mean length {m.response_length_mean:.4g} > "
                        f"{cfg.length_explosion_ratio:.2g}x baseline {state.baseline_length:.4g}"
                    ),
                    value=m.response_length_mean,
                )
            )

    def _check_advantage(self, m: RLStepMetrics, out: list[Anomaly]) -> None:
        cfg = self.config
        if 0.0 <= m.advantage_std < cfg.advantage_std_min:
            out.append(
                Anomaly(
                    step=m.step,
                    kind="advantage_collapse",
                    severity="critical",
                    detail=(
                        f"advantage_std {m.advantage_std:.4g} < {cfg.advantage_std_min:.4g}; "
                        "GRPO group degenerate — only the KL term updates, policy drifts"
                    ),
                    value=m.advantage_std,
                )
            )

    def _check_reward_std(self, m: RLStepMetrics, out: list[Anomaly]) -> None:
        cfg = self.config
        if 0.0 <= m.reward_std < cfg.reward_std_min:
            out.append(
                Anomaly(
                    step=m.step,
                    kind="reward_std_collapse",
                    severity="warn",
                    detail=(
                        f"reward_std {m.reward_std:.4g} < {cfg.reward_std_min:.4g}; "
                        "possible mode collapse"
                    ),
                    value=m.reward_std,
                )
            )

    def _check_reward_spike(self, m: RLStepMetrics, out: list[Anomaly]) -> None:
        cfg = self.config
        window = self._state.reward_history
        if len(window) < 2:
            return
        mean = statistics.fmean(window)
        std = statistics.pstdev(window)
        if std <= 0.0:
            return
        z = (m.reward_mean - mean) / std
        if abs(z) >= cfg.reward_spike_z:
            out.append(
                Anomaly(
                    step=m.step,
                    kind="reward_spike",
                    severity="warn",
                    detail=(
                        f"reward_mean {m.reward_mean:.4g} is z={z:.2f} vs rolling window "
                        f"(mean {mean:.4g}, std {std:.4g}); possible reward hacking"
                    ),
                    value=m.reward_mean,
                )
            )

    def _check_grad(self, m: RLStepMetrics, out: list[Anomaly]) -> None:
        cfg = self.config
        if m.grad_norm > cfg.grad_norm_max or math.isinf(m.grad_norm) or math.isnan(m.grad_norm):
            value = m.grad_norm if math.isfinite(m.grad_norm) else float("inf")
            out.append(
                Anomaly(
                    step=m.step,
                    kind="grad_explosion",
                    severity="critical",
                    detail=f"grad_norm {m.grad_norm:.4g} > grad_norm_max {cfg.grad_norm_max:.4g}",
                    value=value,
                )
            )
