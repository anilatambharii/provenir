from __future__ import annotations

from typing import Literal

import pytest

from provenir.eval.judge import LLMJudge, Preference, RubricScore, StubJudge
from provenir.eval.judge_calibration import (
    BiasReport,
    DebiasedJudge,
    EnsembleJudge,
    JudgeCalibrator,
)

# ---------------------------------------------------------------------------
# Deterministic test judges
# ---------------------------------------------------------------------------


class PositionBiasedJudge:
    """Always prefers whatever is in the first slot ("a")."""

    def score_pairwise(
        self, prompt: str, response_a: str, response_b: str
    ) -> Preference:
        return Preference(preferred="a", confidence=0.9, rationale="first slot")

    def score_rubric(
        self, prompt: str, response: str, criteria: list[str]
    ) -> list[RubricScore]:
        return [RubricScore(criterion=c, score=1.0, rationale="r") for c in criteria]


class ContentJudge:
    """Prefers the response equal to a fixed target string, regardless of slot."""

    def __init__(self, target: str, score: float = 1.0) -> None:
        self._target = target
        self._score = score

    def score_pairwise(
        self, prompt: str, response_a: str, response_b: str
    ) -> Preference:
        if response_a == self._target:
            return Preference(preferred="a", confidence=0.8, rationale="content")
        if response_b == self._target:
            return Preference(preferred="b", confidence=0.8, rationale="content")
        return Preference(preferred="tie", confidence=0.5, rationale="neither")

    def score_rubric(
        self, prompt: str, response: str, criteria: list[str]
    ) -> list[RubricScore]:
        return [
            RubricScore(criterion=c, score=self._score, rationale="content")
            for c in criteria
        ]


class ConstantJudge:
    """Always returns a fixed verdict — used for majority-vote tests."""

    def __init__(self, preferred: Literal["a", "b", "tie"]) -> None:
        self._preferred = preferred

    def score_pairwise(
        self, prompt: str, response_a: str, response_b: str
    ) -> Preference:
        return Preference(preferred=self._preferred, confidence=1.0, rationale="const")

    def score_rubric(
        self, prompt: str, response: str, criteria: list[str]
    ) -> list[RubricScore]:
        score = 1.0 if self._preferred == "a" else 0.0
        return [RubricScore(criterion=c, score=score, rationale="const") for c in criteria]


CASES = [
    ("q1", "alpha", "beta"),
    ("q2", "gamma", "delta"),
    ("q3", "one", "two"),
    ("q4", "left", "right"),
]


# ---------------------------------------------------------------------------
# BiasReport
# ---------------------------------------------------------------------------


class TestBiasReport:
    def test_fields(self) -> None:
        r = BiasReport(position_bias=0.1, self_consistency=0.9, flip_rate=0.1, samples=4)
        assert r.position_bias == 0.1
        assert r.samples == 4

    def test_is_frozen(self) -> None:
        r = BiasReport(0.0, 1.0, 0.0, 1)
        with pytest.raises((AttributeError, TypeError)):
            r.flip_rate = 0.5  # type: ignore[misc]

    def test_is_reliable_true(self) -> None:
        assert BiasReport(0.0, 1.0, 0.0, 4).is_reliable is True

    def test_is_reliable_false_on_flip(self) -> None:
        assert BiasReport(0.0, 0.5, 0.5, 4).is_reliable is False

    def test_is_reliable_false_on_position_bias(self) -> None:
        assert BiasReport(1.0, 1.0, 0.0, 4).is_reliable is False

    def test_is_reliable_boundary(self) -> None:
        assert BiasReport(0.2, 0.8, 0.2, 4).is_reliable is True

    def test_to_dict(self) -> None:
        d = BiasReport(0.0, 1.0, 0.0, 4).to_dict()
        assert d["is_reliable"] is True
        assert d["samples"] == 4
        assert set(d) == {
            "position_bias",
            "self_consistency",
            "flip_rate",
            "samples",
            "is_reliable",
        }

    def test_post_init_position_bias_range(self) -> None:
        with pytest.raises(ValueError):
            BiasReport(1.5, 1.0, 0.0, 4)

    def test_post_init_self_consistency_range(self) -> None:
        with pytest.raises(ValueError):
            BiasReport(0.0, -0.1, 0.0, 4)

    def test_post_init_flip_rate_range(self) -> None:
        with pytest.raises(ValueError):
            BiasReport(0.0, 1.0, 2.0, 4)

    def test_post_init_samples_non_negative(self) -> None:
        with pytest.raises(ValueError):
            BiasReport(0.0, 1.0, 0.0, -1)


# ---------------------------------------------------------------------------
# JudgeCalibrator.measure_position_bias
# ---------------------------------------------------------------------------


class TestMeasurePositionBias:
    def test_position_biased_judge_is_unreliable(self) -> None:
        calibrator = JudgeCalibrator(PositionBiasedJudge())
        report = calibrator.measure_position_bias(CASES)
        assert report.flip_rate == 1.0
        assert report.position_bias == 1.0
        assert report.self_consistency == 0.0
        assert report.is_reliable is False

    def test_content_judge_has_no_flips(self) -> None:
        # Judge always prefers whichever slot holds "alpha"/"gamma"/... target.
        # Use a judge that prefers the lexicographically smaller content.
        calibrator = JudgeCalibrator(ContentJudge(target="alpha"))
        report = calibrator.measure_position_bias([("q1", "alpha", "beta")])
        assert report.flip_rate == 0.0
        assert report.position_bias == 0.0
        assert report.self_consistency == 1.0
        assert report.is_reliable is True

    def test_content_judge_reliable_over_many(self) -> None:
        # A judge preferring a fixed target present as slot-a in every case:
        # forward picks a, reverse (target now in slot b) picks b -> content agrees.
        judge = ContentJudge(target="alpha")
        report = JudgeCalibrator(judge).measure_position_bias([("q1", "alpha", "beta")])
        assert report.is_reliable

    def test_samples_recorded(self) -> None:
        report = JudgeCalibrator(StubJudge()).measure_position_bias(CASES)
        assert report.samples == len(CASES)

    def test_empty_cases(self) -> None:
        report = JudgeCalibrator(StubJudge()).measure_position_bias([])
        assert report.samples == 0
        assert report.flip_rate == 0.0
        assert report.position_bias == 0.0

    def test_stub_judge_length_bias_no_flip(self) -> None:
        # StubJudge prefers the longer response — content-stable across swaps.
        report = JudgeCalibrator(StubJudge()).measure_position_bias(
            [("q", "short", "a longer response here")]
        )
        assert report.flip_rate == 0.0
        assert report.is_reliable is True

    def test_all_ties_zero_position_bias(self) -> None:
        # Equal-length responses -> StubJudge ties both ways; no decisive picks.
        report = JudgeCalibrator(StubJudge()).measure_position_bias(
            [("q", "abcd", "wxyz")]
        )
        assert report.position_bias == 0.0
        assert report.flip_rate == 0.0


# ---------------------------------------------------------------------------
# JudgeCalibrator.measure_self_consistency
# ---------------------------------------------------------------------------


class TestMeasureSelfConsistency:
    def test_deterministic_judge_is_one(self) -> None:
        sc = JudgeCalibrator(StubJudge()).measure_self_consistency(CASES, repeats=3)
        assert sc == 1.0

    def test_position_biased_deterministic_is_one(self) -> None:
        sc = JudgeCalibrator(PositionBiasedJudge()).measure_self_consistency(CASES)
        assert sc == 1.0

    def test_repeats_below_two_raises(self) -> None:
        with pytest.raises(ValueError):
            JudgeCalibrator(StubJudge()).measure_self_consistency(CASES, repeats=1)

    def test_empty_cases_returns_one(self) -> None:
        assert JudgeCalibrator(StubJudge()).measure_self_consistency([]) == 1.0


# ---------------------------------------------------------------------------
# DebiasedJudge
# ---------------------------------------------------------------------------


class TestDebiasedJudge:
    def test_protocol_compliance(self) -> None:
        assert isinstance(DebiasedJudge(StubJudge()), LLMJudge)

    def test_biased_verdicts_become_tie(self) -> None:
        # PositionBiasedJudge always picks slot a -> orderings disagree on content.
        judge = DebiasedJudge(PositionBiasedJudge())
        result = judge.score_pairwise("q", "alpha", "beta")
        assert result.preferred == "tie"
        assert result.confidence < 0.5

    def test_agreeing_content_judge_stays_decisive(self) -> None:
        judge = DebiasedJudge(ContentJudge(target="alpha"))
        result = judge.score_pairwise("q", "alpha", "beta")
        assert result.preferred == "a"

    def test_averaged_confidence(self) -> None:
        judge = DebiasedJudge(ContentJudge(target="alpha"))
        result = judge.score_pairwise("q", "alpha", "beta")
        assert result.confidence == pytest.approx(0.8)

    def test_rubric_delegates(self) -> None:
        judge = DebiasedJudge(ContentJudge(target="alpha", score=0.6))
        scores = judge.score_rubric("q", "resp", ["clarity", "accuracy"])
        assert [s.criterion for s in scores] == ["clarity", "accuracy"]
        assert all(s.score == pytest.approx(0.6) for s in scores)

    def test_rubric_averages_over_repeats(self) -> None:
        judge = DebiasedJudge(ContentJudge(target="x", score=0.4), repeats=3)
        scores = judge.score_rubric("q", "resp", ["c"])
        assert scores[0].score == pytest.approx(0.4)

    def test_invalid_repeats(self) -> None:
        with pytest.raises(ValueError):
            DebiasedJudge(StubJudge(), repeats=0)

    def test_debiased_stub_prefers_longer(self) -> None:
        judge = DebiasedJudge(StubJudge())
        assert judge.score_pairwise("q", "short", "much longer").preferred == "b"


# ---------------------------------------------------------------------------
# EnsembleJudge
# ---------------------------------------------------------------------------


class TestEnsembleJudge:
    def test_protocol_compliance(self) -> None:
        assert isinstance(EnsembleJudge([StubJudge()]), LLMJudge)

    def test_empty_ensemble_raises(self) -> None:
        with pytest.raises(ValueError):
            EnsembleJudge([])

    def test_majority_vote(self) -> None:
        ensemble = EnsembleJudge(
            [ConstantJudge("a"), ConstantJudge("a"), ConstantJudge("b")]
        )
        result = ensemble.score_pairwise("q", "x", "y")
        assert result.preferred == "a"
        assert result.confidence == pytest.approx(2 / 3)

    def test_tie_breaks_to_tie(self) -> None:
        ensemble = EnsembleJudge([ConstantJudge("a"), ConstantJudge("b")])
        assert ensemble.score_pairwise("q", "x", "y").preferred == "tie"

    def test_unanimous_confidence_one(self) -> None:
        ensemble = EnsembleJudge([ConstantJudge("b"), ConstantJudge("b")])
        result = ensemble.score_pairwise("q", "x", "y")
        assert result.preferred == "b"
        assert result.confidence == pytest.approx(1.0)

    def test_rubric_mean_score(self) -> None:
        ensemble = EnsembleJudge([ConstantJudge("a"), ConstantJudge("b")])
        scores = ensemble.score_rubric("q", "resp", ["clarity"])
        # ConstantJudge("a") -> 1.0, ConstantJudge("b") -> 0.0, mean 0.5
        assert scores[0].score == pytest.approx(0.5)
        assert scores[0].criterion == "clarity"

    def test_rubric_multiple_criteria(self) -> None:
        ensemble = EnsembleJudge([StubJudge(), StubJudge()])
        scores = ensemble.score_rubric("q", "resp", ["a", "b", "c"])
        assert len(scores) == 3
        assert all(s.score == pytest.approx(0.5) for s in scores)
