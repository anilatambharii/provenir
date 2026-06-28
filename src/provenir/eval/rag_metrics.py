from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


@dataclass(frozen=True)
class RAGMetricSummary:
    """Per-example scores for a single (question, answer, context) triple."""

    faithfulness: float
    context_precision: float
    answer_relevance: float

    @property
    def combined(self) -> float:
        """Harmonic mean of all three metrics; 0.0 if any score is zero."""
        scores = [self.faithfulness, self.context_precision, self.answer_relevance]
        if any(s == 0.0 for s in scores):
            return 0.0
        return 3.0 / sum(1.0 / s for s in scores)


class FaithfulnessMetric:
    """Token-overlap proxy: fraction of answer tokens grounded in the retrieved context.

    A grounded token is one that appears verbatim in the context.  This is a
    retrieval-free approximation — plug in an LLMJudge for higher fidelity.
    """

    name: str = "faithfulness"

    def score(self, answer: str, context: str) -> float:
        answer_tokens = _tokenize(answer)
        if not answer_tokens:
            return 0.0
        context_tokens = set(_tokenize(context))
        grounded = sum(1 for t in answer_tokens if t in context_tokens)
        return grounded / len(answer_tokens)


class ContextPrecisionMetric:
    """Token-overlap proxy: fraction of context tokens relevant to the question."""

    name: str = "context_precision"

    def score(self, question: str, context: str) -> float:
        context_tokens = _tokenize(context)
        if not context_tokens:
            return 0.0
        question_tokens = set(_tokenize(question))
        overlap = sum(1 for t in context_tokens if t in question_tokens)
        return overlap / len(context_tokens)


class AnswerRelevanceMetric:
    """Token-overlap proxy: fraction of question tokens covered by the answer."""

    name: str = "answer_relevance"

    def score(self, question: str, answer: str) -> float:
        question_tokens = _tokenize(question)
        if not question_tokens:
            return 0.0
        answer_tokens = set(_tokenize(answer))
        covered = sum(1 for t in question_tokens if t in answer_tokens)
        return covered / len(question_tokens)


@dataclass
class RAGEvaluator:
    """Runs all three RAG metrics over a list of (question, answer, context) triples.

    Each triple is a dict with keys ``question``, ``answer``, and ``context``.
    Use :meth:`evaluate` for per-example scores and :meth:`aggregate` for
    dataset-level means.
    """

    faithfulness: FaithfulnessMetric = field(default_factory=FaithfulnessMetric)
    context_precision: ContextPrecisionMetric = field(default_factory=ContextPrecisionMetric)
    answer_relevance: AnswerRelevanceMetric = field(default_factory=AnswerRelevanceMetric)

    def evaluate(self, triples: list[dict[str, Any]]) -> list[RAGMetricSummary]:
        results: list[RAGMetricSummary] = []
        for triple in triples:
            q = str(triple.get("question", ""))
            a = str(triple.get("answer", ""))
            ctx = str(triple.get("context", ""))
            results.append(
                RAGMetricSummary(
                    faithfulness=self.faithfulness.score(a, ctx),
                    context_precision=self.context_precision.score(q, ctx),
                    answer_relevance=self.answer_relevance.score(q, a),
                )
            )
        return results

    def aggregate(self, summaries: list[RAGMetricSummary]) -> dict[str, float]:
        """Dataset-level mean for each metric."""
        if not summaries:
            return {"faithfulness": 0.0, "context_precision": 0.0, "answer_relevance": 0.0}
        n = len(summaries)
        return {
            "faithfulness": sum(s.faithfulness for s in summaries) / n,
            "context_precision": sum(s.context_precision for s in summaries) / n,
            "answer_relevance": sum(s.answer_relevance for s in summaries) / n,
        }
