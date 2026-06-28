from __future__ import annotations

import pytest

from provenir.data.dataset import JsonlDataset
from provenir.eval.harness import MultiMetricEvaluator, _wilson_ci
from provenir.eval.metrics import (
    BLEUMetric,
    ExactMatchMetric,
    ROUGELMetric,
    TokenF1Metric,
    _lcs_length,
    _tokenize,
)

# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------


def test_tokenize_lowercases_and_splits() -> None:
    assert _tokenize("Hello World") == ["hello", "world"]


def test_tokenize_empty() -> None:
    assert _tokenize("") == []


# ---------------------------------------------------------------------------
# LCS helper
# ---------------------------------------------------------------------------


def test_lcs_identical() -> None:
    tokens = ["a", "b", "c"]
    assert _lcs_length(tokens, tokens) == 3


def test_lcs_disjoint() -> None:
    assert _lcs_length(["a", "b"], ["c", "d"]) == 0


def test_lcs_subsequence() -> None:
    assert _lcs_length(["a", "b", "c"], ["a", "c"]) == 2


# ---------------------------------------------------------------------------
# ExactMatchMetric
# ---------------------------------------------------------------------------


class TestExactMatch:
    metric = ExactMatchMetric()

    def test_perfect(self) -> None:
        assert self.metric.score("hello", "hello") == 1.0

    def test_no_match(self) -> None:
        assert self.metric.score("hello", "world") == 0.0

    def test_strip_normalised(self) -> None:
        assert self.metric.score("  hello  ", "hello") == 1.0

    def test_case_sensitive(self) -> None:
        assert self.metric.score("Hello", "hello") == 0.0

    def test_empty_both(self) -> None:
        assert self.metric.score("", "") == 1.0

    def test_name(self) -> None:
        assert self.metric.name == "exact_match"


# ---------------------------------------------------------------------------
# TokenF1Metric
# ---------------------------------------------------------------------------


class TestTokenF1:
    metric = TokenF1Metric()

    def test_perfect(self) -> None:
        assert self.metric.score("the cat sat", "the cat sat") == pytest.approx(1.0)

    def test_no_overlap(self) -> None:
        assert self.metric.score("dog runs", "cat sits") == pytest.approx(0.0)

    def test_partial_overlap(self) -> None:
        # pred: [the, cat], ref: [the, dog] → overlap=1, P=0.5, R=0.5, F1=0.5
        assert self.metric.score("the cat", "the dog") == pytest.approx(0.5)

    def test_empty_prediction(self) -> None:
        assert self.metric.score("", "the cat") == 0.0

    def test_empty_reference(self) -> None:
        assert self.metric.score("the cat", "") == 0.0

    def test_name(self) -> None:
        assert self.metric.name == "token_f1"


# ---------------------------------------------------------------------------
# BLEUMetric
# ---------------------------------------------------------------------------


class TestBLEU:
    metric = BLEUMetric()

    def test_perfect(self) -> None:
        score = self.metric.score("the cat sat on the mat", "the cat sat on the mat")
        assert score == pytest.approx(1.0)

    def test_no_overlap(self) -> None:
        assert self.metric.score("dog runs fast", "cat sits still") == 0.0

    def test_empty_prediction(self) -> None:
        assert self.metric.score("", "the cat") == 0.0

    def test_empty_reference(self) -> None:
        assert self.metric.score("the cat", "") == 0.0

    def test_brevity_penalty_applied(self) -> None:
        # Prediction shares all n-grams with reference but is shorter → BP < 1 → score < 1.0
        score = self.metric.score(
            "the quick brown cat",
            "the quick brown cat sat on the mat",
        )
        assert 0.0 < score < 1.0

    def test_unigram_only(self) -> None:
        metric = BLEUMetric(max_ngram=1)
        # Perfect unigram overlap
        assert metric.score("the cat", "the cat") == pytest.approx(1.0)

    def test_score_in_range(self) -> None:
        score = self.metric.score("the cat sat", "the dog ran")
        assert 0.0 <= score <= 1.0

    def test_name(self) -> None:
        assert self.metric.name == "bleu"


# ---------------------------------------------------------------------------
# ROUGELMetric
# ---------------------------------------------------------------------------


class TestROUGEL:
    metric = ROUGELMetric()

    def test_perfect(self) -> None:
        assert self.metric.score("the cat sat", "the cat sat") == pytest.approx(1.0)

    def test_no_overlap(self) -> None:
        assert self.metric.score("dog runs", "cat sits") == pytest.approx(0.0)

    def test_subsequence(self) -> None:
        # pred=[the, cat, sat], ref=[the, sat] → LCS=2
        # P=2/3, R=2/2=1.0, F1=2*(2/3)/(2/3+1) = 4/3 / (5/3) = 4/5 = 0.8
        score = self.metric.score("the cat sat", "the sat")
        assert score == pytest.approx(0.8)

    def test_empty_prediction(self) -> None:
        assert self.metric.score("", "the cat") == 0.0

    def test_empty_reference(self) -> None:
        assert self.metric.score("the cat", "") == 0.0

    def test_score_in_range(self) -> None:
        score = self.metric.score("the quick brown fox", "a lazy dog")
        assert 0.0 <= score <= 1.0

    def test_name(self) -> None:
        assert self.metric.name == "rouge_l"


# ---------------------------------------------------------------------------
# Wilson CI helper
# ---------------------------------------------------------------------------


def test_wilson_ci_perfect_accuracy() -> None:
    lo, hi = _wilson_ci(1.0, 100)
    assert lo > 0.9
    assert hi == pytest.approx(1.0)


def test_wilson_ci_zero_accuracy() -> None:
    lo, hi = _wilson_ci(0.0, 100)
    assert lo == pytest.approx(0.0)
    assert hi < 0.1


def test_wilson_ci_zero_samples() -> None:
    assert _wilson_ci(0.5, 0) == (0.0, 0.0)


def test_wilson_ci_bounds_valid() -> None:
    lo, hi = _wilson_ci(0.6, 50)
    assert 0.0 <= lo <= hi <= 1.0


# ---------------------------------------------------------------------------
# MultiMetricEvaluator
# ---------------------------------------------------------------------------


def _make_dataset(pairs: list[tuple[str, str]]) -> JsonlDataset:
    return JsonlDataset.from_records(
        [{"prompt": p, "response": r} for p, r in pairs]
    )


class TestMultiMetricEvaluator:
    def test_default_metrics_present(self) -> None:
        ds = _make_dataset([("q", "answer")])
        result = MultiMetricEvaluator().evaluate(ds, ["answer"])
        assert set(result.metrics or {}) >= {"exact_match", "token_f1", "bleu", "rouge_l"}

    def test_perfect_predictions(self) -> None:
        ds = _make_dataset([("q1", "yes"), ("q2", "no")])
        result = MultiMetricEvaluator().evaluate(ds, ["yes", "no"])
        assert result.accuracy is not None
        assert result.accuracy.mean == pytest.approx(1.0)

    def test_zero_predictions(self) -> None:
        ds = _make_dataset([("q1", "yes"), ("q2", "no")])
        result = MultiMetricEvaluator().evaluate(ds, ["X", "Y"])
        assert result.accuracy is not None
        assert result.accuracy.mean == pytest.approx(0.0)

    def test_mismatch_raises(self) -> None:
        ds = _make_dataset([("q", "a")])
        with pytest.raises(ValueError, match="prediction count"):
            MultiMetricEvaluator().evaluate(ds, ["a", "extra"])

    def test_custom_metrics(self) -> None:
        ds = _make_dataset([("q", "hello world")])
        evaluator = MultiMetricEvaluator(metrics=[TokenF1Metric()])
        result = evaluator.evaluate(ds, ["hello world"])
        assert "token_f1" in (result.metrics or {})
        assert "bleu" not in (result.metrics or {})

    def test_samples_count_correct(self) -> None:
        pairs = [("q", "a"), ("q", "b"), ("q", "c")]
        ds = _make_dataset(pairs)
        result = MultiMetricEvaluator().evaluate(ds, ["a", "b", "c"])
        for summary in (result.metrics or {}).values():
            assert summary.samples == 3

    def test_confidence_interval_ordered(self) -> None:
        ds = _make_dataset([("q", "a")] * 10)
        result = MultiMetricEvaluator().evaluate(ds, ["a"] * 10)
        for summary in (result.metrics or {}).values():
            lo, hi = summary.confidence_interval
            assert lo <= summary.mean <= hi
