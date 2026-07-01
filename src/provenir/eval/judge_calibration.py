from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from statistics import mean
from typing import Any, Literal

from provenir.eval.judge import LLMJudge, Preference, RubricScore

__all__ = [
    "BiasReport",
    "DebiasedJudge",
    "EnsembleJudge",
    "JudgeCalibrator",
]


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BiasReport:
    """Calibration diagnostics for an :class:`LLMJudge`.

    LLM judges suffer from position bias (preferring a fixed slot), low
    self-consistency (disagreeing with themselves on repeats) and verdict
    flips when the two responses are swapped.  A reliable judge scores low on
    all three.

    Attributes:
        position_bias: Tendency to pick the first slot regardless of content,
            measured as ``|P(A) - 0.5| * 2`` over swapped pairs with ties
            excluded.  ``0`` = balanced, ``1`` = always the first slot.
        self_consistency: Agreement rate across repeated / swapped judgments,
            in ``0..1``.
        flip_rate: Fraction of pairs whose content-level verdict flips when the
            two responses are swapped.  High means unreliable.
        samples: Number of cases the report was computed over.

    Example:
        >>> report = BiasReport(0.0, 1.0, 0.0, 4)
        >>> report.is_reliable
        True
    """

    position_bias: float
    self_consistency: float
    flip_rate: float
    samples: int

    def __post_init__(self) -> None:
        if not 0.0 <= self.position_bias <= 1.0:
            raise ValueError("position_bias must be in [0, 1]")
        if not 0.0 <= self.self_consistency <= 1.0:
            raise ValueError("self_consistency must be in [0, 1]")
        if not 0.0 <= self.flip_rate <= 1.0:
            raise ValueError("flip_rate must be in [0, 1]")
        if self.samples < 0:
            raise ValueError("samples must be non-negative")

    @property
    def is_reliable(self) -> bool:
        """True when the judge is consistent, stable and low-bias."""
        return (
            self.self_consistency >= 0.8
            and self.flip_rate <= 0.2
            and self.position_bias <= 0.2
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "position_bias": self.position_bias,
            "self_consistency": self.self_consistency,
            "flip_rate": self.flip_rate,
            "samples": self.samples,
            "is_reliable": self.is_reliable,
        }


# ---------------------------------------------------------------------------
# Wilson interval helper (optional, mirrors harness.py)
# ---------------------------------------------------------------------------


def _wilson_ci(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """95 % Wilson score interval for a proportion — exact at small n, unlike Wald."""
    if n == 0:
        return (0.0, 0.0)
    z2 = z * z
    centre = (p + z2 / (2 * n)) / (1 + z2 / n)
    margin = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return (max(0.0, centre - margin), min(1.0, centre + margin))


# ---------------------------------------------------------------------------
# Calibrator
# ---------------------------------------------------------------------------


def _swap(preferred: Literal["a", "b", "tie"]) -> Literal["a", "b", "tie"]:
    """Map a verdict from swapped-order slots back to original content slots."""
    if preferred == "a":
        return "b"
    if preferred == "b":
        return "a"
    return "tie"


class JudgeCalibrator:
    """Measures position bias and self-consistency of an :class:`LLMJudge`.

    Example:
        >>> from provenir.eval.judge import StubJudge
        >>> calibrator = JudgeCalibrator(StubJudge())
        >>> report = calibrator.measure_position_bias(
        ...     [("q", "short", "a much longer response")]
        ... )
        >>> report.flip_rate
        0.0
    """

    def __init__(self, judge: LLMJudge) -> None:
        self._judge = judge

    def measure_position_bias(
        self, cases: list[tuple[str, str, str]]
    ) -> BiasReport:
        """Probe position bias by judging each case in both orderings.

        Each case is ``(prompt, response_a, response_b)``.  The judge is asked
        both ``(a, b)`` and ``(b, a)``.  A position-unbiased judge picks the
        same *content* winner both times; a flip signals unreliability.  The
        first-slot pick counts feed the ``position_bias`` estimate.
        """
        flips = 0
        first_slot_picks = 0
        decisive = 0
        for prompt, response_a, response_b in cases:
            forward = self._judge.score_pairwise(prompt, response_a, response_b)
            reverse = self._judge.score_pairwise(prompt, response_b, response_a)
            # Content-level verdict of the reverse call (undo the slot swap).
            reverse_content = _swap(reverse.preferred)

            if forward.preferred != "tie":
                decisive += 1
                if forward.preferred == "a":
                    first_slot_picks += 1
            if reverse.preferred != "tie":
                decisive += 1
                if reverse.preferred == "a":
                    first_slot_picks += 1

            if forward.preferred != reverse_content:
                flips += 1

        n = len(cases)
        flip_rate = flips / n if n else 0.0
        if decisive:
            p_first = first_slot_picks / decisive
            position_bias = abs(p_first - 0.5) * 2.0
        else:
            position_bias = 0.0
        self_consistency = 1.0 - flip_rate
        return BiasReport(
            position_bias=position_bias,
            self_consistency=self_consistency,
            flip_rate=flip_rate,
            samples=n,
        )

    def measure_self_consistency(
        self, cases: list[tuple[str, str, str]], repeats: int = 3
    ) -> float:
        """Mean pairwise agreement across ``repeats`` judgments of each case.

        Calls the judge ``repeats`` times on the identical input and returns
        the average fraction of agreeing judgment pairs, in ``0..1``.  A fully
        deterministic judge returns ``1.0``.
        """
        if repeats < 2:
            raise ValueError("repeats must be at least 2")
        agreements: list[float] = []
        for prompt, response_a, response_b in cases:
            verdicts = [
                self._judge.score_pairwise(prompt, response_a, response_b).preferred
                for _ in range(repeats)
            ]
            agree = 0
            total = 0
            for i in range(len(verdicts)):
                for j in range(i + 1, len(verdicts)):
                    total += 1
                    if verdicts[i] == verdicts[j]:
                        agree += 1
            if total:
                agreements.append(agree / total)
        return mean(agreements) if agreements else 1.0


# ---------------------------------------------------------------------------
# Debiased judge
# ---------------------------------------------------------------------------


class DebiasedJudge:
    """Neutralises position bias by requiring both orderings to agree.

    Implements :class:`LLMJudge`.  ``score_pairwise`` evaluates ``(a, b)`` and
    ``(b, a)``; only when both orderings agree on the same *content* winner is a
    decisive preference returned (with averaged confidence).  Otherwise the
    disagreement is reported as a low-confidence tie.

    Example:
        >>> from provenir.eval.judge import StubJudge
        >>> judge = DebiasedJudge(StubJudge())
        >>> judge.score_pairwise("q", "short", "longer response").preferred
        'b'
    """

    def __init__(self, judge: LLMJudge, repeats: int = 1) -> None:
        if repeats < 1:
            raise ValueError("repeats must be at least 1")
        self._judge = judge
        self._repeats = repeats

    def score_pairwise(
        self, prompt: str, response_a: str, response_b: str
    ) -> Preference:
        forward = self._judge.score_pairwise(prompt, response_a, response_b)
        reverse = self._judge.score_pairwise(prompt, response_b, response_a)
        reverse_content = _swap(reverse.preferred)

        if forward.preferred != "tie" and forward.preferred == reverse_content:
            confidence = (forward.confidence + reverse.confidence) / 2.0
            return Preference(
                preferred=forward.preferred,
                confidence=confidence,
                rationale="both orderings agree",
            )
        return Preference(
            preferred="tie",
            confidence=min(forward.confidence, reverse.confidence) / 2.0,
            rationale="orderings disagree; position bias suspected",
        )

    def score_rubric(
        self, prompt: str, response: str, criteria: list[str]
    ) -> list[RubricScore]:
        """Delegate rubric scoring, averaging over ``repeats`` runs.

        Rubric scoring has no position bias, so this simply averages the score
        of each criterion across repeated judgments for noise reduction.
        """
        runs = [
            self._judge.score_rubric(prompt, response, criteria)
            for _ in range(self._repeats)
        ]
        first = runs[0]
        return [
            RubricScore(
                criterion=first[idx].criterion,
                score=mean(run[idx].score for run in runs),
                rationale=first[idx].rationale,
            )
            for idx in range(len(criteria))
        ]


# ---------------------------------------------------------------------------
# Ensemble judge
# ---------------------------------------------------------------------------


class EnsembleJudge:
    """Aggregates several judges by majority vote / mean score.

    Implements :class:`LLMJudge`.  ``score_pairwise`` takes a majority vote
    across the member judges (ties broken to ``"tie"``) with confidence equal
    to the winning agreement fraction.  ``score_rubric`` averages each
    criterion's score across judges.

    Example:
        >>> from provenir.eval.judge import StubJudge
        >>> ensemble = EnsembleJudge([StubJudge(), StubJudge()])
        >>> ensemble.score_pairwise("q", "aa", "b").preferred
        'a'
    """

    def __init__(self, judges: list[LLMJudge]) -> None:
        if not judges:
            raise ValueError("judges list must be non-empty")
        self._judges = list(judges)

    def score_pairwise(
        self, prompt: str, response_a: str, response_b: str
    ) -> Preference:
        verdicts = [
            j.score_pairwise(prompt, response_a, response_b) for j in self._judges
        ]
        counts = Counter(v.preferred for v in verdicts)
        top = max(counts.values())
        winners = [pref for pref, count in counts.items() if count == top]
        preferred: Literal["a", "b", "tie"] = winners[0] if len(winners) == 1 else "tie"
        confidence = counts[preferred] / len(verdicts)
        return Preference(
            preferred=preferred,
            confidence=confidence,
            rationale=f"majority vote of {len(self._judges)} judges",
        )

    def score_rubric(
        self, prompt: str, response: str, criteria: list[str]
    ) -> list[RubricScore]:
        per_judge = [
            j.score_rubric(prompt, response, criteria) for j in self._judges
        ]
        return [
            RubricScore(
                criterion=per_judge[0][idx].criterion,
                score=mean(run[idx].score for run in per_judge),
                rationale=f"mean of {len(self._judges)} judges",
            )
            for idx in range(len(criteria))
        ]
