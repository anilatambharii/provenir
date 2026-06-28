from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


def _ngrams(tokens: list[str], n: int) -> Counter[tuple[str, ...]]:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def _lcs_length(x: list[str], y: list[str]) -> int:
    """O(m*n) dynamic programming LCS using rolling array."""
    m, n = len(x), len(y)
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    return prev[n]


@runtime_checkable
class MetricFn(Protocol):
    """Protocol for a single-example scorer: (prediction, reference) -> [0, 1]."""

    @property
    def name(self) -> str: ...

    def score(self, prediction: str, reference: str) -> float: ...


@dataclass(frozen=True)
class ExactMatchMetric:
    """Strip-normalised exact match."""

    name: str = field(default="exact_match")

    def score(self, prediction: str, reference: str) -> float:
        return 1.0 if prediction.strip() == reference.strip() else 0.0


@dataclass(frozen=True)
class TokenF1Metric:
    """Token-overlap F1 (SQuAD-style, lowercased whitespace tokenization)."""

    name: str = field(default="token_f1")

    def score(self, prediction: str, reference: str) -> float:
        pred_tokens = _tokenize(prediction)
        ref_tokens = _tokenize(reference)
        if not pred_tokens or not ref_tokens:
            return 0.0
        pred_counts = Counter(pred_tokens)
        ref_counts = Counter(ref_tokens)
        overlap = sum((pred_counts & ref_counts).values())
        precision = overlap / len(pred_tokens)
        recall = overlap / len(ref_tokens)
        if precision + recall == 0.0:
            return 0.0
        return 2.0 * precision * recall / (precision + recall)


@dataclass(frozen=True)
class BLEUMetric:
    """Corpus-free sentence BLEU with brevity penalty (max_ngram = 1..4)."""

    max_ngram: int = field(default=4)
    name: str = field(default="bleu")

    def score(self, prediction: str, reference: str) -> float:
        pred_tokens = _tokenize(prediction)
        ref_tokens = _tokenize(reference)
        if not pred_tokens or not ref_tokens:
            return 0.0
        bp = (
            1.0
            if len(pred_tokens) >= len(ref_tokens)
            else math.exp(1.0 - len(ref_tokens) / len(pred_tokens))
        )
        log_avg = 0.0
        for n in range(1, self.max_ngram + 1):
            pred_ngrams = _ngrams(pred_tokens, n)
            ref_ngrams = _ngrams(ref_tokens, n)
            if not pred_ngrams:
                return 0.0
            clipped = sum(min(c, ref_ngrams[g]) for g, c in pred_ngrams.items())
            total = sum(pred_ngrams.values())
            p_n = clipped / total if total else 0.0
            if p_n == 0.0:
                return 0.0
            log_avg += math.log(p_n) / self.max_ngram
        return bp * math.exp(log_avg)


@dataclass(frozen=True)
class ROUGELMetric:
    """ROUGE-L F1 based on longest common subsequence."""

    name: str = field(default="rouge_l")

    def score(self, prediction: str, reference: str) -> float:
        pred_tokens = _tokenize(prediction)
        ref_tokens = _tokenize(reference)
        if not pred_tokens or not ref_tokens:
            return 0.0
        lcs = _lcs_length(pred_tokens, ref_tokens)
        precision = lcs / len(pred_tokens)
        recall = lcs / len(ref_tokens)
        if precision + recall == 0.0:
            return 0.0
        return 2.0 * precision * recall / (precision + recall)


DEFAULT_METRICS: list[MetricFn] = [
    ExactMatchMetric(),
    TokenF1Metric(),
    BLEUMetric(),
    ROUGELMetric(),
]
