"""Training-Inference Mismatch (TIM) Detector.

After RLVR training, the model's learned policy may diverge from inference-time
behaviour due to different sampling, batching, or context handling (Zhong 2026,
OpenRLHF #1108). This module measures KL divergence between training-time and
inference-time log-probability distributions on held-out probes.

High KL divergence between the two distributions signals that training
improvements may not transfer to the deployment setting.

**Backend-agnostic design**: the :data:`TIMProbeFn` callable lets the detector
stay decoupled from any training or inference engine. In tests, inject a
synthetic closure; in production, wire real model log-probs from your rollout
loop and inference server. The detector exposes the *methodology*; the quality of
the numbers depends on the backend you provide.

Example::

    from provenir.environments.tim import TIMDetector, gate_tim

    def probe_fn(prompt: str) -> tuple[list[float], list[float]]:
        # In production: call training engine + inference server.
        train_lp = [-0.1, -0.2, -0.3]
        inf_lp   = [-0.1, -0.2, -0.3]
        return train_lp, inf_lp

    detector = TIMDetector(threshold=0.1)
    report = detector.detect(probe_fn, ["prompt1", "prompt2"])
    print(report.summary())
    gate_tim(report, "production")   # passes when no mismatch detected
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------

#: Given a prompt string, returns ``(train_log_probs, inference_log_probs)`` —
#: two parallel lists of per-token log-probs of the *same* length.
#:
#: In tests supply a synthetic closure; in production wire the training engine
#: rollout log-probs and the inference server log-probs.
TIMProbeFn = Callable[[str], tuple[list[float], list[float]]]


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TIMResult:
    """Per-prompt KL-divergence measurement.

    ``kl_divergence`` is ``KL(train || inference)`` — the information-theoretic
    distance from the training distribution to the inference distribution.
    ``mismatch`` is ``True`` when ``kl_divergence`` exceeds the detector's
    configured threshold.
    """

    prompt: str
    kl_divergence: float
    mismatch: bool

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain, JSON-friendly ``dict``."""
        return {
            "prompt": self.prompt,
            "kl_divergence": self.kl_divergence,
            "mismatch": self.mismatch,
        }


@dataclass(frozen=True)
class TIMReport:
    """Aggregate TIM findings over a set of probes.

    ``mismatch_detected`` is ``True`` when the mean KL across all probes exceeds
    the detector threshold — the training-time policy has materially diverged from
    the inference-time policy and training gains may not transfer.
    """

    probe_count: int
    mean_kl: float
    max_kl: float
    mismatch_rate: float
    mismatch_detected: bool
    results: list[TIMResult] = field(default_factory=list)

    def summary(self) -> str:
        """One-line human summary."""
        flag = "MISMATCH" if self.mismatch_detected else "ok"
        return (
            f"TIM [{flag}]: probes={self.probe_count} "
            f"mean_kl={self.mean_kl:.4f} max_kl={self.max_kl:.4f} "
            f"mismatch_rate={self.mismatch_rate:.3f}"
        )

    def content_hash(self) -> str:
        """SHA-256 of the canonical serialisation — suitable for signing into a Passport."""
        serialised = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialised.encode()).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain, JSON-friendly ``dict`` (embeds in a Passport)."""
        return {
            "probe_count": self.probe_count,
            "mean_kl": self.mean_kl,
            "max_kl": self.max_kl,
            "mismatch_rate": self.mismatch_rate,
            "mismatch_detected": self.mismatch_detected,
            "results": [r.to_dict() for r in self.results],
        }


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


class TIMDetector:
    """Measure KL divergence between training-time and inference-time distributions.

    For each prompt in the probe set, the detector calls ``probe_fn(prompt)`` to
    obtain per-token log-probabilities from both the training rollout and the
    inference server, then computes ``KL(train || inference)``.

    Parameters
    ----------
    threshold:
        Per-probe KL threshold used to set :attr:`TIMResult.mismatch` and,
        when the *mean* KL exceeds it, :attr:`TIMReport.mismatch_detected`.
    min_prob:
        Lower-bound clamp applied to every probability before computing KL, to
        avoid ``log(0)`` crashes.

    Example::

        detector = TIMDetector(threshold=0.05)
        report   = detector.detect(my_probe_fn, prompts)
        assert not report.mismatch_detected
    """

    def __init__(self, threshold: float = 0.1, min_prob: float = 1e-9) -> None:
        self.threshold = threshold
        self.min_prob = min_prob

    def detect(self, probe_fn: TIMProbeFn, prompts: Sequence[str]) -> TIMReport:
        """Run the detector over *prompts* and return a :class:`TIMReport`.

        For each prompt the detector:

        1. Calls ``probe_fn(prompt)`` → ``(train_lp, inf_lp)``.
        2. Normalises each list to a probability distribution via log-sum-exp.
        3. Clamps each probability to ``min_prob``.
        4. Computes ``KL(train || inference)``.
        5. Marks the result as a mismatch when KL > ``threshold``.

        The report's ``mismatch_detected`` flag is set when ``mean_kl > threshold``.
        """
        results: list[TIMResult] = []
        for prompt in prompts:
            train_lp, inf_lp = probe_fn(prompt)
            kl = self._kl_divergence(train_lp, inf_lp)
            results.append(
                TIMResult(
                    prompt=prompt,
                    kl_divergence=kl,
                    mismatch=kl > self.threshold,
                )
            )

        probe_count = len(results)
        if probe_count == 0:
            return TIMReport(
                probe_count=0,
                mean_kl=0.0,
                max_kl=0.0,
                mismatch_rate=0.0,
                mismatch_detected=False,
                results=[],
            )

        kl_values = [r.kl_divergence for r in results]
        mean_kl = sum(kl_values) / probe_count
        max_kl = max(kl_values)
        mismatch_rate = sum(1 for r in results if r.mismatch) / probe_count
        mismatch_detected = mean_kl > self.threshold

        return TIMReport(
            probe_count=probe_count,
            mean_kl=mean_kl,
            max_kl=max_kl,
            mismatch_rate=mismatch_rate,
            mismatch_detected=mismatch_detected,
            results=results,
        )

    def _kl_divergence(self, train_lp: list[float], inf_lp: list[float]) -> float:
        """Compute ``KL(train || inference)`` from parallel log-prob lists.

        Handles length mismatch by truncating to the shorter list.  Uses
        log-sum-exp normalisation for numerical stability and clamps probabilities
        to ``min_prob`` before computing the divergence.

        Returns a non-negative float (clamped to ``[0, ∞)``).
        """
        n = min(len(train_lp), len(inf_lp))
        if n == 0:
            return 0.0

        t_lp = train_lp[:n]
        i_lp = inf_lp[:n]

        # Normalise via log-sum-exp to get proper probability distributions.
        p_train = _log_softmax_to_probs(t_lp, self.min_prob)
        p_inf = _log_softmax_to_probs(i_lp, self.min_prob)

        # KL(P || Q) = sum_i p_i * log(p_i / q_i)
        kl = sum(
            p_t * math.log(p_t / p_i)
            for p_t, p_i in zip(p_train, p_inf)
        )
        return max(0.0, kl)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _log_softmax_to_probs(log_probs: list[float], min_prob: float) -> list[float]:
    """Convert a list of log-probs to a clamped probability distribution.

    Uses the log-sum-exp trick for numerical stability.
    """
    if not log_probs:
        return []
    log_max = max(log_probs)
    shifted = [lp - log_max for lp in log_probs]
    exps = [math.exp(s) for s in shifted]
    total = sum(exps)
    if total == 0.0:
        # Degenerate: all -inf; return uniform distribution clamped to min_prob.
        return [min_prob] * len(log_probs)
    raw = [e / total for e in exps]
    return [max(p, min_prob) for p in raw]


# ---------------------------------------------------------------------------
# Promotion gate
# ---------------------------------------------------------------------------


class TIMBlocked(RuntimeError):
    """Raised by :func:`gate_tim` when training-inference mismatch blocks promotion."""


#: Stage names that require a clean TIM check before promotion.
_DEFAULT_PROTECTED_STAGES: frozenset[str] = frozenset({"production", "prod", "release"})


def gate_tim(
    report: TIMReport,
    stage: str,
    *,
    protected_stages: frozenset[str] = _DEFAULT_PROTECTED_STAGES,
    max_mean_kl: float | None = None,
) -> None:
    """Block promotion to a protected *stage* when training-inference mismatch is detected.

    A detected mismatch (``report.mismatch_detected is True``) always blocks a
    protected stage. Optionally also require ``mean_kl`` to stay below
    ``max_mean_kl``. Promotion to non-protected stages (e.g. ``"staging"``) is
    never blocked. Mirrors :func:`~provenir.environments.reliability.gate_promotion`.

    Raises:
        TIMBlocked: if *stage* is protected and the report fails the bar.

    Example::

        gate_tim(report, "staging")                         # never blocks
        gate_tim(report, "production")                      # blocks if mismatch_detected
        gate_tim(report, "production", max_mean_kl=0.05)    # also enforces KL ceiling
    """
    if stage not in protected_stages:
        return
    if report.mismatch_detected:
        raise TIMBlocked(
            f"training-inference mismatch detected (mean_kl={report.mean_kl:.4f} "
            f"> threshold); refusing promotion to {stage!r}. {report.summary()}"
        )
    if max_mean_kl is not None and report.mean_kl > max_mean_kl:
        raise TIMBlocked(
            f"mean KL divergence {report.mean_kl:.4f} > max_mean_kl={max_mean_kl:.4f}; "
            f"refusing promotion to {stage!r}."
        )
