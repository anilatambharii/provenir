from __future__ import annotations

import pytest

from provenir.eval.rag_metrics import (
    AnswerRelevanceMetric,
    ContextPrecisionMetric,
    FaithfulnessMetric,
    RAGEvaluator,
    RAGMetricSummary,
)


class TestRAGMetricSummary:
    def test_combined_zero_when_any_zero(self) -> None:
        s = RAGMetricSummary(faithfulness=0.0, context_precision=0.8, answer_relevance=0.9)
        assert s.combined == 0.0

    def test_combined_harmonic_mean(self) -> None:
        s = RAGMetricSummary(faithfulness=1.0, context_precision=1.0, answer_relevance=1.0)
        assert s.combined == pytest.approx(1.0)

    def test_combined_in_range(self) -> None:
        s = RAGMetricSummary(faithfulness=0.5, context_precision=0.6, answer_relevance=0.7)
        assert 0.0 <= s.combined <= 1.0

    def test_is_frozen(self) -> None:
        s = RAGMetricSummary(faithfulness=0.5, context_precision=0.5, answer_relevance=0.5)
        with pytest.raises((AttributeError, TypeError)):
            s.faithfulness = 0.9  # type: ignore[misc]


class TestFaithfulnessMetric:
    m = FaithfulnessMetric()

    def test_name(self) -> None:
        assert self.m.name == "faithfulness"

    def test_empty_answer_returns_zero(self) -> None:
        assert self.m.score("", "some context") == 0.0

    def test_fully_grounded_returns_one(self) -> None:
        score = self.m.score("the cat sat", "the cat sat on the mat")
        assert score == pytest.approx(1.0)

    def test_ungrounded_answer_returns_zero(self) -> None:
        score = self.m.score("quantum entanglement physics", "cats dogs birds")
        assert score == pytest.approx(0.0)

    def test_partial_grounding(self) -> None:
        # "the cat" → 2 tokens in context; "quantum" → not in context
        score = self.m.score("the cat quantum", "the cat sat")
        assert 0.0 < score < 1.0

    def test_empty_context_gives_zero(self) -> None:
        assert self.m.score("words here", "") == 0.0


class TestContextPrecisionMetric:
    m = ContextPrecisionMetric()

    def test_name(self) -> None:
        assert self.m.name == "context_precision"

    def test_empty_context_returns_zero(self) -> None:
        assert self.m.score("what is a cat?", "") == 0.0

    def test_fully_relevant_context(self) -> None:
        score = self.m.score("what is a cat", "what is a cat")
        assert score == pytest.approx(1.0)

    def test_irrelevant_context_low_score(self) -> None:
        score = self.m.score("what is a cat?", "dogs birds fish")
        assert score < 0.1


class TestAnswerRelevanceMetric:
    m = AnswerRelevanceMetric()

    def test_name(self) -> None:
        assert self.m.name == "answer_relevance"

    def test_empty_question_returns_zero(self) -> None:
        assert self.m.score("", "some answer") == 0.0

    def test_full_coverage_returns_one(self) -> None:
        # All question tokens ("alpha", "beta") appear in the answer.
        score = self.m.score("alpha beta", "answer about alpha beta indeed")
        assert score == pytest.approx(1.0)

    def test_no_coverage_returns_zero(self) -> None:
        # Question tokens ("xyzzy", "plugh") never appear in the answer.
        score = self.m.score("xyzzy plugh", "hello world foo bar baz")
        assert score == pytest.approx(0.0)


class TestRAGEvaluator:
    eval_r = RAGEvaluator()

    _TRIPLE = {
        "question": "what is the capital of France",
        "answer": "Paris is the capital of France",
        "context": "France is a country. Paris is its capital city.",
    }

    def test_evaluate_returns_one_per_triple(self) -> None:
        results = self.eval_r.evaluate([self._TRIPLE])
        assert len(results) == 1

    def test_evaluate_scores_in_range(self) -> None:
        results = self.eval_r.evaluate([self._TRIPLE])
        for result in results:
            assert 0.0 <= result.faithfulness <= 1.0
            assert 0.0 <= result.context_precision <= 1.0
            assert 0.0 <= result.answer_relevance <= 1.0

    def test_aggregate_means_in_range(self) -> None:
        summaries = self.eval_r.evaluate([self._TRIPLE, self._TRIPLE])
        agg = self.eval_r.aggregate(summaries)
        for v in agg.values():
            assert 0.0 <= v <= 1.0

    def test_aggregate_empty_returns_zeros(self) -> None:
        agg = self.eval_r.aggregate([])
        assert all(v == 0.0 for v in agg.values())

    def test_evaluate_empty_returns_empty(self) -> None:
        assert self.eval_r.evaluate([]) == []

    def test_aggregate_keys(self) -> None:
        summaries = self.eval_r.evaluate([self._TRIPLE])
        agg = self.eval_r.aggregate(summaries)
        assert set(agg) == {"faithfulness", "context_precision", "answer_relevance"}
