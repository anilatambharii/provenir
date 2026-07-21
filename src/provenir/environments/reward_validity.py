"""Spurious-reward ablation harness: is your reward signal valid or just amplifying priors?

Shao et al. 2026 (*Spurious Rewards*, arXiv:2506.10947) showed that GRPO on Qwen2.5-Math-7B
gains +21.4 pp on MATH-500 with **entirely random rewards** (vs +29.1 pp with ground truth).
This means benchmark gains alone cannot distinguish a valid reward from a spurious one that
merely unlocks pre-trained knowledge. Provenir's spurious-reward ablation harness exposes the
methodology: run short diagnostic trainings under a battery of **degenerate reward controls**
and measure whether improvement survives.

A valid reward produces measurable gain that degenerate signals (random, constant, format-only,
length-only, shuffled) do **not** replicate. A spurious reward produces gain that degenerate
signals match — the optimizer is amplifying pretrained priors, not learning from the reward.

**Backend-agnostic design**: the :data:`TrainEval` callable lets the harness stay decoupled
from any training engine. In tests, inject a synthetic thunk; in production, wire a real
``RLOrchestrator.run``. The harness exposes the *methodology*; the quality of the numbers
depends on what backend you provide.

Example::

    from provenir.environments.reward_validity import (
        Ablation,
        RewardValidityHarness,
        gate_reward_validity,
        RewardValidityBlocked,
    )

    # Synthetic: real reward wins, degenerate rewards gain nothing.
    def train_eval(reward_fn):
        base = 0.40
        gain = 0.30 if reward_fn.kind == "real" else 0.01
        return base, base + gain, 0.8, {}

    harness = RewardValidityHarness()
    report = harness.evaluate(real_reward, train_eval)
    print(report.summary())
    gate_reward_validity(report, "production")   # passes
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping, Sequence

from provenir.core.abstractions import RewardFn

# ---------------------------------------------------------------------------
# Ablation enum
# ---------------------------------------------------------------------------


class Ablation(str, Enum):
    """The six ablation controls in the battery.

    ``REAL`` is the baseline (the actual reward under test). The remaining five
    are degenerate controls designed to reveal spurious reward dynamics.
    """

    REAL = "real"
    RANDOM = "random"
    CONSTANT = "constant"
    FORMAT_ONLY = "format_only"
    LENGTH_ONLY = "length_only"
    SHUFFLED = "shuffled"


# ---------------------------------------------------------------------------
# Degenerate RewardFn implementations
# ---------------------------------------------------------------------------

#: Small epsilon used in validity denominator to avoid division by zero.
_EPS: float = 1e-9

#: Marker attribute used by the harness to identify the real reward.
_REAL_KIND: str = "real"


class _RandomReward(RewardFn):
    """Uniform-random reward in [0, 1], seeded and independent of the response.

    This is Shao et al.'s core control: if gain under this reward matches gain
    under the real reward, the signal is spurious.
    """

    kind: str = "random"

    def __init__(self, seed: int) -> None:
        self._rng = random.Random(seed)

    def score(self, trajectory: Mapping[str, Any]) -> float:  # noqa: ARG002
        return self._rng.random()

    def gameability_check(self, trajectory: Mapping[str, Any]) -> list[str]:  # noqa: ARG002
        return []


class _ConstantReward(RewardFn):
    """Fixed constant reward for every response.

    Diagnoses advantage-collapse / prior amplification: if a constant reward
    still yields improvement, the model was already learning from other signals.
    """

    kind: str = "constant"

    def __init__(self, value: float = 1.0) -> None:
        self._value = value

    def score(self, trajectory: Mapping[str, Any]) -> float:  # noqa: ARG002
        return self._value

    def gameability_check(self, trajectory: Mapping[str, Any]) -> list[str]:  # noqa: ARG002
        return []


class _FormatOnlyReward(RewardFn):
    """1.0 iff the response matches a shallow format regex; content is ignored.

    Diagnoses format-exploit gaming: if gain under format-matching alone matches
    the real reward, the model learned to format, not to be correct.
    """

    kind: str = "format_only"

    def __init__(self, pattern: str) -> None:
        self._re = re.compile(pattern, re.DOTALL)

    def score(self, trajectory: Mapping[str, Any]) -> float:
        response = str(trajectory.get("prediction", trajectory.get("response", "")))
        return 1.0 if self._re.search(response) else 0.0

    def gameability_check(self, trajectory: Mapping[str, Any]) -> list[str]:  # noqa: ARG002
        return []


class _LengthOnlyReward(RewardFn):
    """Monotone function of response length (normalised to [0, 1]).

    Diagnoses length-inflation reward hacking: a model that just generates
    longer responses should not appear to improve on this metric.
    """

    kind: str = "length_only"

    #: Cap at this character count before normalising (avoids extreme values).
    _MAX_LEN: int = 4096

    def score(self, trajectory: Mapping[str, Any]) -> float:
        response = str(trajectory.get("prediction", trajectory.get("response", "")))
        return min(len(response), self._MAX_LEN) / self._MAX_LEN

    def gameability_check(self, trajectory: Mapping[str, Any]) -> list[str]:  # noqa: ARG002
        return []


class _ShuffledReward(RewardFn):
    """Real reward evaluated against a **mismatched** reference (shuffled references).

    Tests reference-binding validity: if the reward scores well even when the
    reference is wrong, the verifier is not actually checking the answer.
    """

    kind: str = "shuffled"

    def __init__(self, real_reward: RewardFn, seed: int) -> None:
        self._real = real_reward
        self._rng = random.Random(seed)
        self._pool: list[Any] = []

    def _mismatched_reference(self, trajectory: Mapping[str, Any]) -> dict[str, Any]:
        """Return a copy of *trajectory* whose reference is replaced by a pooled one."""
        ref = trajectory.get("reference")
        self._pool.append(ref)
        # Pick a reference from the pool that is different (best-effort).
        candidates = [r for r in self._pool if r != ref]
        if not candidates:
            # Only one unique reference seen so far; use a fixed sentinel.
            mismatched: Any = "__SHUFFLED_SENTINEL__"
        else:
            mismatched = self._rng.choice(candidates)
        patched = dict(trajectory)
        patched["reference"] = mismatched
        return patched

    def score(self, trajectory: Mapping[str, Any]) -> float:
        return self._real.score(self._mismatched_reference(trajectory))

    def gameability_check(self, trajectory: Mapping[str, Any]) -> list[str]:  # noqa: ARG002
        return []


def _make_degenerate(
    ablation: Ablation,
    real_reward: RewardFn,
    seed: int,
    format_pattern: str,
) -> RewardFn:
    """Factory: return a degenerate :class:`RewardFn` for *ablation*.

    The *real* ablation just returns *real_reward* itself unchanged.
    """
    if ablation is Ablation.REAL:
        return real_reward
    if ablation is Ablation.RANDOM:
        return _RandomReward(seed)
    if ablation is Ablation.CONSTANT:
        return _ConstantReward(1.0)
    if ablation is Ablation.FORMAT_ONLY:
        return _FormatOnlyReward(format_pattern)
    if ablation is Ablation.LENGTH_ONLY:
        return _LengthOnlyReward()
    if ablation is Ablation.SHUFFLED:
        return _ShuffledReward(real_reward, seed)
    raise ValueError(f"Unknown ablation: {ablation!r}")  # pragma: no cover


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AblationRun:
    """Recorded outcome for one ablation control run.

    ``base_score`` and ``final_score`` are task-metric scores (e.g. accuracy
    on a held-out probe set) **before** and **after** the short diagnostic
    training under the ablation reward. ``mean_reward`` is the mean reward
    signal observed during training. ``anomalies`` carries flight-recorder
    counts (e.g. ``{"reward_spike": 3}``).
    """

    ablation: Ablation
    base_score: float
    final_score: float
    mean_reward: float
    anomalies: dict[str, int] = field(default_factory=dict)

    @property
    def gain(self) -> float:
        """Post-train minus pre-train task-metric score (``final_score − base_score``)."""
        return self.final_score - self.base_score

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain, JSON-friendly ``dict``."""
        return {
            "ablation": self.ablation.value,
            "base_score": self.base_score,
            "final_score": self.final_score,
            "mean_reward": self.mean_reward,
            "gain": self.gain,
            "anomalies": dict(self.anomalies),
        }


@dataclass(frozen=True)
class RewardValidityReport:
    """Aggregate spurious-reward ablation findings for one reward function.

    ``validity`` is a 0–1 score: 1.0 means the real reward drives improvement
    that no degenerate signal replicates; 0.0 means degenerate signals match or
    beat the real reward. ``spurious`` is a hard-fail flag that trips when *any*
    degenerate gain is within ``tolerance`` of the real gain.
    """

    reward_name: str
    runs: dict[Ablation, AblationRun]
    validity: float  # 0–1
    spurious: bool  # True → degenerate gain ≈ real gain

    def summary(self) -> str:
        """One-line human summary."""
        flag = "SPURIOUS" if self.spurious else "ok"
        real_run = self.runs.get(Ablation.REAL)
        real_gain = real_run.gain if real_run is not None else float("nan")
        degen_gains = ", ".join(
            f"{a.value}={r.gain:+.3f}"
            for a, r in sorted(self.runs.items(), key=lambda kv: kv[0].value)
            if a is not Ablation.REAL
        )
        return (
            f"reward {self.reward_name!r}: validity={self.validity:.3f} [{flag}] "
            f"real_gain={real_gain:+.3f} ({degen_gains})"
        )

    def content_hash(self) -> str:
        """SHA-256 of the canonical serialisation — suitable for signing into a Passport."""
        serialised = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialised.encode()).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain, JSON-friendly ``dict`` (embeds in a Passport)."""
        return {
            "reward_name": self.reward_name,
            "validity": self.validity,
            "spurious": self.spurious,
            "runs": {a.value: r.to_dict() for a, r in self.runs.items()},
        }


# ---------------------------------------------------------------------------
# TrainEval thunk type alias
# ---------------------------------------------------------------------------

#: A callable that runs a short diagnostic training+evaluation under one reward.
#:
#: Signature: ``(reward: RewardFn) -> (base_score, final_score, mean_reward, anomaly_counts)``
#:
#: The harness calls this once per ablation. In tests supply a synthetic closure;
#: in production wire ``RLOrchestrator.run`` (or similar).
TrainEval = Callable[[RewardFn], tuple[float, float, float, dict[str, int]]]


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


class RewardValidityHarness:
    """Run the spurious-reward ablation battery against a candidate reward function.

    The harness is fully deterministic given ``seed``: two calls with the same
    reward, ``train_eval``, and seed produce identical reports.

    Parameters
    ----------
    ablations:
        Subset of :class:`Ablation` controls to run (default: all six).
    tolerance:
        How close a degenerate gain must be to the real gain to trip the
        ``spurious`` flag. Default 0.1 (within 10 pp of real gain).
    seed:
        Integer seed for all degenerate RNG-based rewards.
    format_pattern:
        Regex used by the ``FORMAT_ONLY`` ablation. Default: ``\\boxed{...}``.

    Example::

        harness = RewardValidityHarness(tolerance=0.05, seed=42)
        report = harness.evaluate(my_reward, my_train_eval)
        assert not report.spurious
    """

    def __init__(
        self,
        ablations: Sequence[Ablation] | None = None,
        tolerance: float = 0.1,
        seed: int = 0,
        format_pattern: str = r"\\boxed\{.*\}",
    ) -> None:
        self.ablations: tuple[Ablation, ...] = (
            tuple(ablations) if ablations is not None else tuple(Ablation)
        )
        if tolerance < 0:
            raise ValueError(f"tolerance must be >= 0, got {tolerance}")
        self.tolerance = tolerance
        self.seed = seed
        self.format_pattern = format_pattern

    def evaluate(self, reward: RewardFn, train_eval: TrainEval) -> RewardValidityReport:
        """Run the ablation battery and return a :class:`RewardValidityReport`.

        For each ablation in ``self.ablations``, the harness:

        1. Constructs the appropriate degenerate :class:`RewardFn`.
        2. Calls ``train_eval(degenerate_reward)`` to obtain
           ``(base_score, final_score, mean_reward, anomalies)``.
        3. Records an :class:`AblationRun`.

        Then computes ``validity`` and the ``spurious`` flag from §4 of the spec.
        """
        reward_name = getattr(reward, "name", type(reward).__name__)

        runs: dict[Ablation, AblationRun] = {}
        for ablation in self.ablations:
            degenerate = _make_degenerate(ablation, reward, self.seed, self.format_pattern)
            base_score, final_score, mean_reward, anomalies = train_eval(degenerate)
            runs[ablation] = AblationRun(
                ablation=ablation,
                base_score=base_score,
                final_score=final_score,
                mean_reward=mean_reward,
                anomalies=dict(anomalies),
            )

        validity, spurious = self._compute_validity(runs)
        return RewardValidityReport(
            reward_name=str(reward_name),
            runs=runs,
            validity=validity,
            spurious=spurious,
        )

    def _compute_validity(
        self, runs: dict[Ablation, AblationRun]
    ) -> tuple[float, bool]:
        """Compute validity score and spurious flag from the run results.

        Formula (§4):
            validity = clamp(
                (Δ_real − max(Δ_random, Δ_constant, Δ_format, Δ_length)) / max(Δ_real, ε),
                0, 1
            )
        ``spurious`` trips when any degenerate Δ ≥ Δ_real − tolerance.
        """
        real_run = runs.get(Ablation.REAL)
        real_gain = real_run.gain if real_run is not None else 0.0

        degenerate_ablations = (
            Ablation.RANDOM,
            Ablation.CONSTANT,
            Ablation.FORMAT_ONLY,
            Ablation.LENGTH_ONLY,
            Ablation.SHUFFLED,
        )

        degen_gains: list[float] = []
        for abl in degenerate_ablations:
            if abl in runs:
                degen_gains.append(runs[abl].gain)

        if not degen_gains:
            # No degenerate controls ran — can't assess; return max validity.
            return 1.0, False

        max_degen = max(degen_gains)

        # Validity: how much of the real gain is *exclusive* to the real reward.
        numerator = real_gain - max_degen
        denominator = max(real_gain, _EPS)
        validity = max(0.0, min(1.0, numerator / denominator))

        # Spurious: any degenerate gain is within tolerance of the real gain.
        spurious = any(g >= real_gain - self.tolerance for g in degen_gains)

        return validity, spurious


# ---------------------------------------------------------------------------
# Promotion gate
# ---------------------------------------------------------------------------


class RewardValidityBlocked(RuntimeError):
    """Raised when a reward is too spurious to promote a run to a protected stage."""


#: Stage names that require a valid (non-spurious) reward before promotion.
_DEFAULT_PROTECTED_STAGES: frozenset[str] = frozenset({"production", "prod", "release"})


def gate_reward_validity(
    report: RewardValidityReport,
    stage: str,
    *,
    protected_stages: frozenset[str] = _DEFAULT_PROTECTED_STAGES,
    min_validity: float | None = None,
) -> None:
    """Block promotion to a protected *stage* when the reward is spurious or invalid.

    A spurious reward (any degenerate signal matches or beats the real reward) always
    blocks a protected stage. Optionally also require ``validity`` to meet
    ``min_validity``. Promotion to non-protected stages (e.g. ``"staging"``) is never
    blocked. Mirrors :func:`~provenir.environments.reliability.gate_promotion`.

    Raises:
        RewardValidityBlocked: if *stage* is protected and the report fails the bar.

    Example::

        gate_reward_validity(report, "staging")                      # never blocks
        gate_reward_validity(report, "production")                   # blocks if spurious
        gate_reward_validity(report, "production", min_validity=0.7) # also enforces floor
    """
    if stage.lower() not in protected_stages:
        return
    if report.spurious:
        raise RewardValidityBlocked(
            f"reward {report.reward_name!r} is SPURIOUS (degenerate controls match real gain); "
            f"refusing promotion to {stage!r}. {report.summary()}"
        )
    if min_validity is not None and report.validity < min_validity:
        raise RewardValidityBlocked(
            f"reward {report.reward_name!r} validity {report.validity:.3f} "
            f"< required {min_validity:.3f}; refusing promotion to {stage!r}."
        )
