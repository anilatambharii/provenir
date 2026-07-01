"""The Loop Doctor: differential diagnosis for a stalled training loop.

When an RL or fine-tuning loop stops improving, "it's not working" is useless.
The Loop Doctor answers *why* by reasoning over Provenir's trust signals and
attributing the stall to one of four causes:

* **eval** — the evaluation is contaminated, so the metrics are lying;
* **reward** — the reward is being gamed (reward hacking);
* **algorithm** — the optimization is unstable (entropy/advantage/KL collapse);
* **data** — the model has plateaued for lack of sufficient or fresh data.

Only Provenir can do this differential diagnosis, because it already produces
all four signals (contamination firewall, reward-hacking detector, RL flight
recorder, and slice analysis). When the verdict is *data*, the doctor emits a
concrete, human-facing :class:`DataRequest` — which slices, how many examples,
and how recent — turning "give me more data" into an actionable ask.

Example
-------
>>> signals = LoopSignals(
...     reward_history=[0.4, 0.41, 0.4, 0.41, 0.4],   # plateaued
...     anomaly_kinds=[],
...     slice_failures={"tool_use": 0.8, "math": 0.1},
... )
>>> diagnosis = LoopDoctor().diagnose(signals)
>>> diagnosis.primary_category
'data'
>>> diagnosis.data_request is not None
True
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

_CATEGORIES: frozenset[str] = frozenset({"data", "reward", "eval", "algorithm"})
_SEVERITIES: frozenset[str] = frozenset({"warn", "critical"})

#: Flight-recorder anomaly kinds that indicate an optimization/algorithm fault.
_ALGORITHM_ANOMALIES: frozenset[str] = frozenset(
    {
        "entropy_collapse",
        "advantage_collapse",
        "kl_blowup",
        "kl_collapse",
        "grad_explosion",
        "reward_std_collapse",
    }
)

#: Concrete remediation per algorithm anomaly kind.
_ALGORITHM_ACTIONS: dict[str, str] = {
    "advantage_collapse": "enable DAPO dynamic sampling or raise group_size — the "
    "group has no reward variance, so only the KL term is updating",
    "entropy_collapse": "add an entropy bonus or lower the learning rate — the "
    "policy is collapsing to a single mode",
    "kl_blowup": "lower the learning rate or add/raise the KL penalty — the policy "
    "is drifting too far from the reference",
    "kl_collapse": "the policy has stopped moving; raise the learning rate or relax "
    "the KL penalty",
    "grad_explosion": "clip gradients or lower the learning rate — gradients are "
    "exploding",
    "reward_std_collapse": "diversify rollouts (higher temperature / larger groups) "
    "— reward variance has collapsed (mode collapse)",
}


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DataRequest:
    """A concrete, human-facing request for more (or fresher) training data.

    Emitted when the Loop Doctor attributes a plateau to a data problem.

    Example::

        req = DataRequest(slices=["tool_use"], num_examples=50,
                          freshness_days=180, rationale="plateaued on tool_use")
        assert req.to_dict()["num_examples"] == 50
    """

    slices: list[str]
    num_examples: int
    freshness_days: int | None
    rationale: str

    def __post_init__(self) -> None:
        if self.num_examples < 1:
            raise ValueError(f"num_examples must be >= 1, got {self.num_examples}")
        if self.freshness_days is not None and self.freshness_days < 1:
            raise ValueError(
                f"freshness_days must be >= 1 or None, got {self.freshness_days}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "slices": list(self.slices),
            "num_examples": self.num_examples,
            "freshness_days": self.freshness_days,
            "rationale": self.rationale,
        }


@dataclass(frozen=True)
class Finding:
    """A single diagnosed cause with evidence and a recommended action."""

    category: str
    severity: str
    confidence: float
    evidence: str
    recommended_action: str

    def __post_init__(self) -> None:
        if self.category not in _CATEGORIES:
            raise ValueError(
                f"category must be one of {sorted(_CATEGORIES)}, got {self.category!r}"
            )
        if self.severity not in _SEVERITIES:
            raise ValueError(
                f"severity must be one of {sorted(_SEVERITIES)}, got {self.severity!r}"
            )
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"confidence must be in [0.0, 1.0], got {self.confidence}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "severity": self.severity,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "recommended_action": self.recommended_action,
        }


@dataclass(frozen=True)
class Diagnosis:
    """The Loop Doctor's verdict: ranked findings plus an optional data request."""

    findings: list[Finding]
    data_request: DataRequest | None = None

    @property
    def is_healthy(self) -> bool:
        """True when no fault was found."""
        return not self.findings

    @property
    def primary_category(self) -> str:
        """Category of the highest-confidence finding, or ``"none"`` if healthy."""
        if not self.findings:
            return "none"
        return self.findings[0].category

    def to_dict(self) -> dict[str, Any]:
        return {
            "primary_category": self.primary_category,
            "is_healthy": self.is_healthy,
            "findings": [f.to_dict() for f in self.findings],
            "data_request": self.data_request.to_dict() if self.data_request else None,
        }

    def to_markdown(self) -> str:
        """Render a human-readable diagnosis card."""
        if self.is_healthy:
            return "# Loop Diagnosis\n\n**Healthy** — no fault detected; keep training.\n"
        lines = ["# Loop Diagnosis", ""]
        lines.append(f"**Primary cause: {self.primary_category}**")
        lines.append("")
        for i, f in enumerate(self.findings, 1):
            lines.append(
                f"{i}. **{f.category}** ({f.severity}, confidence "
                f"{f.confidence:.0%}) — {f.evidence}"
            )
            lines.append(f"   - Action: {f.recommended_action}")
        if self.data_request is not None:
            req = self.data_request
            fresh = (
                f", from the last {req.freshness_days} days"
                if req.freshness_days is not None
                else ""
            )
            lines.append("")
            lines.append(
                f"**Data request:** ~{req.num_examples} more examples of "
                f"{', '.join(req.slices)}{fresh}. {req.rationale}"
            )
        return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Evidence bundle
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LoopSignals:
    """The evidence the Loop Doctor reasons over.

    Bundle the trust-layer scalars directly, or build one from live trust-layer
    objects with :meth:`from_reports`.
    """

    reward_history: list[float]
    anomaly_kinds: list[str] = field(default_factory=list)
    hacking_rate: float = 0.0
    hacking_kinds: list[str] = field(default_factory=list)
    contamination_rate: float = 0.0
    slice_failures: dict[str, float] = field(default_factory=dict)
    data_age_days: float | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.hacking_rate <= 1.0:
            raise ValueError(f"hacking_rate must be in [0.0, 1.0], got {self.hacking_rate}")
        if not 0.0 <= self.contamination_rate <= 1.0:
            raise ValueError(
                f"contamination_rate must be in [0.0, 1.0], got {self.contamination_rate}"
            )

    @classmethod
    def from_reports(
        cls,
        reward_history: list[float],
        flight_recorder: Any = None,
        hacking_report: Any = None,
        contamination_report: Any = None,
        slice_report: Any = None,
        data_age_days: float | None = None,
    ) -> LoopSignals:
        """Extract signals from live trust-layer objects (any may be ``None``).

        Accepts a :class:`~provenir.observability.FlightRecorder`,
        :class:`~provenir.observability.HackingReport`,
        :class:`~provenir.eval.contamination.ContaminationReport`, and
        :class:`~provenir.loop.slices.SliceReport`. Attribute access is
        defensive so partially-populated objects are tolerated.
        """
        anomaly_kinds: list[str] = []
        if flight_recorder is not None:
            anomaly_kinds = [a.kind for a in getattr(flight_recorder, "anomalies", [])]

        hacking_rate = 0.0
        hacking_kinds: list[str] = []
        if hacking_report is not None:
            hacking_rate = float(getattr(hacking_report, "hacking_rate", 0.0))
            by_kind = getattr(hacking_report, "by_kind", None)
            if callable(by_kind):
                hacking_kinds = list(by_kind().keys())

        contamination_rate = 0.0
        if contamination_report is not None:
            contamination_rate = float(
                getattr(contamination_report, "contamination_rate", 0.0)
            )

        slice_failures: dict[str, float] = {}
        if slice_report is not None:
            slice_failures = dict(getattr(slice_report, "slice_failures", {}))

        return cls(
            reward_history=list(reward_history),
            anomaly_kinds=anomaly_kinds,
            hacking_rate=hacking_rate,
            hacking_kinds=hacking_kinds,
            contamination_rate=contamination_rate,
            slice_failures=slice_failures,
            data_age_days=data_age_days,
        )


# ---------------------------------------------------------------------------
# Doctor
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DoctorConfig:
    """Thresholds governing the differential diagnosis."""

    plateau_window: int = 5
    plateau_slope_eps: float = 1e-3
    hacking_rate_max: float = 0.1
    contamination_rate_max: float = 0.0
    stale_data_days: float = 180.0
    slice_failure_max: float = 0.5
    min_examples_per_slice: int = 50

    def __post_init__(self) -> None:
        if self.plateau_window < 2:
            raise ValueError(f"plateau_window must be >= 2, got {self.plateau_window}")
        if self.plateau_slope_eps < 0.0:
            raise ValueError(
                f"plateau_slope_eps must be >= 0, got {self.plateau_slope_eps}"
            )
        if not 0.0 <= self.hacking_rate_max <= 1.0:
            raise ValueError(
                f"hacking_rate_max must be in [0.0, 1.0], got {self.hacking_rate_max}"
            )
        if not 0.0 <= self.contamination_rate_max <= 1.0:
            raise ValueError(
                f"contamination_rate_max must be in [0.0, 1.0], got "
                f"{self.contamination_rate_max}"
            )
        if self.stale_data_days <= 0.0:
            raise ValueError(f"stale_data_days must be > 0, got {self.stale_data_days}")
        if not 0.0 <= self.slice_failure_max <= 1.0:
            raise ValueError(
                f"slice_failure_max must be in [0.0, 1.0], got {self.slice_failure_max}"
            )
        if self.min_examples_per_slice < 1:
            raise ValueError(
                f"min_examples_per_slice must be >= 1, got {self.min_examples_per_slice}"
            )


def _slope(values: list[float]) -> float:
    """Least-squares slope of ``values`` against their index (0 for < 2 points)."""
    n = len(values)
    if n < 2:
        return 0.0
    mean_x = (n - 1) / 2.0
    mean_y = sum(values) / n
    num = sum((i - mean_x) * (v - mean_y) for i, v in enumerate(values))
    den = sum((i - mean_x) ** 2 for i in range(n))
    return num / den if den else 0.0


class LoopDoctor:
    """Diagnose why a training loop has stalled and recommend the fix.

    The diagnosis is ordered by certainty: a contaminated eval is checked first
    (it invalidates every other metric), then reward hacking, then optimization
    instability, and finally — as the residual explanation for a plateau — a
    data problem, which yields a concrete :class:`DataRequest`.
    """

    def __init__(self, config: DoctorConfig | None = None) -> None:
        self.config = config or DoctorConfig()

    def diagnose(self, signals: LoopSignals) -> Diagnosis:
        cfg = self.config
        window = signals.reward_history[-cfg.plateau_window :]
        slope = _slope(window)
        plateaued = abs(slope) < cfg.plateau_slope_eps
        declining = slope <= -cfg.plateau_slope_eps

        findings: list[Finding] = []
        data_request: DataRequest | None = None

        # 1. EVAL — a contaminated eval makes every other signal untrustworthy.
        if signals.contamination_rate > cfg.contamination_rate_max:
            findings.append(
                Finding(
                    category="eval",
                    severity="critical",
                    confidence=min(1.0, 0.7 + signals.contamination_rate),
                    evidence=(
                        f"eval is {signals.contamination_rate:.1%} contaminated by "
                        f"training data — reported metrics are inflated"
                    ),
                    recommended_action=(
                        "decontaminate train vs eval (contamination firewall) and "
                        "re-measure before trusting any reward curve"
                    ),
                )
            )

        # 2. REWARD — the reward signal is being gamed.
        if signals.hacking_rate > cfg.hacking_rate_max or signals.hacking_kinds:
            kinds = ", ".join(signals.hacking_kinds) if signals.hacking_kinds else "detected"
            findings.append(
                Finding(
                    category="reward",
                    severity="critical" if signals.hacking_rate >= 0.25 else "warn",
                    confidence=min(1.0, 0.5 + signals.hacking_rate),
                    evidence=(
                        f"reward hacking at {signals.hacking_rate:.1%} ({kinds}) — "
                        f"reward is rising without real task success"
                    ),
                    recommended_action=(
                        "tighten the verifier, add held-out / sandboxed checks, and "
                        "gate on the reward-hacking detector"
                    ),
                )
            )

        # 3. ALGORITHM — the optimization itself is unstable.
        algo_hits = [k for k in signals.anomaly_kinds if k in _ALGORITHM_ANOMALIES]
        if algo_hits:
            worst = algo_hits[0]
            findings.append(
                Finding(
                    category="algorithm",
                    severity="critical",
                    confidence=0.8,
                    evidence=(
                        f"optimization instability: {', '.join(sorted(set(algo_hits)))}"
                    ),
                    recommended_action=_ALGORITHM_ACTIONS.get(
                        worst, "inspect the flight recorder and stabilize training"
                    ),
                )
            )

        # 4. DATA — the residual cause of a plateau when nothing above fired.
        strong_other = any(f.category in {"eval", "reward", "algorithm"} for f in findings)
        if (plateaued or declining) and not strong_other:
            bad_slices = sorted(
                (name for name, rate in signals.slice_failures.items()
                 if rate > cfg.slice_failure_max)
            )
            stale = (
                signals.data_age_days is not None
                and signals.data_age_days > cfg.stale_data_days
            )
            confidence = 0.6
            evidence_bits = [
                "reward has " + ("declined" if declining else "plateaued")
                + f" (slope {slope:+.4f}) with no reward/eval/algorithm fault"
            ]
            if bad_slices:
                confidence += 0.15
                evidence_bits.append(
                    f"concentrated failures on: {', '.join(bad_slices)}"
                )
            if stale:
                confidence += 0.15
                evidence_bits.append(
                    f"training data is ~{signals.data_age_days:.0f} days old (stale)"
                )
            findings.append(
                Finding(
                    category="data",
                    severity="warn",
                    confidence=min(1.0, confidence),
                    evidence="; ".join(evidence_bits),
                    recommended_action=(
                        "collect more" + (" recent" if stale else "") + " data for the "
                        "failing slices (see the attached DataRequest)"
                    ),
                )
            )
            target_slices = bad_slices or ["overall"]
            data_request = DataRequest(
                slices=target_slices,
                num_examples=cfg.min_examples_per_slice * max(1, len(target_slices)),
                freshness_days=int(cfg.stale_data_days) if stale else None,
                rationale=evidence_bits[0]
                + (
                    " — prioritize the failing slices above"
                    if bad_slices
                    else " — broaden coverage"
                ),
            )

        findings.sort(key=lambda f: f.confidence, reverse=True)
        return Diagnosis(findings=findings, data_request=data_request)


# ---------------------------------------------------------------------------
# Controller — turn a diagnosis into the next loop action
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LoopAction:
    """The recommended next action for an adaptive training loop."""

    action: str
    reason: str
    data_request: DataRequest | None = None

    _ACTIONS: ClassVar[frozenset[str]] = frozenset(
        {"continue", "collect_data", "fix_reward", "clean_eval", "stabilize", "stop"}
    )

    def __post_init__(self) -> None:
        if self.action not in self._ACTIONS:
            raise ValueError(
                f"action must be one of {sorted(self._ACTIONS)}, got {self.action!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "reason": self.reason,
            "data_request": self.data_request.to_dict() if self.data_request else None,
        }


class LoopController:
    """Map a :class:`Diagnosis` to the next :class:`LoopAction`.

    This is the "intelligent loop" step: rather than blindly continuing or
    halting on a threshold, the controller acts on the *cause* — cleaning the
    eval, tightening the reward, stabilizing the optimizer, or pausing to
    request data.
    """

    #: Cause -> action, in priority order (eval invalidates everything else).
    _PRIORITY: tuple[str, ...] = ("eval", "reward", "algorithm", "data")
    _ACTION_FOR: dict[str, str] = {
        "eval": "clean_eval",
        "reward": "fix_reward",
        "algorithm": "stabilize",
        "data": "collect_data",
    }

    def decide(self, diagnosis: Diagnosis) -> LoopAction:
        if diagnosis.is_healthy:
            return LoopAction(action="continue", reason="no fault detected")

        categories = {f.category for f in diagnosis.findings}
        for category in self._PRIORITY:
            if category in categories:
                finding = next(f for f in diagnosis.findings if f.category == category)
                return LoopAction(
                    action=self._ACTION_FOR[category],
                    reason=finding.evidence,
                    data_request=diagnosis.data_request if category == "data" else None,
                )
        return LoopAction(action="continue", reason="no actionable finding")
