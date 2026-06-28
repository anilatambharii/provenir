from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any

from provenir.data.dataset import JsonlDataset
from provenir.eval.metrics import DEFAULT_METRICS, MetricFn


@dataclass(frozen=True)
class MetricSummary:
    mean: float
    confidence_interval: tuple[float, float]
    samples: int


@dataclass
class EvaluationResult:
    metrics: dict[str, MetricSummary] | None = None
    accuracy: MetricSummary | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"metadata": self.metadata}
        if self.metrics is not None:
            payload["metrics"] = {
                name: {
                    "mean": metric.mean,
                    "confidence_interval": list(metric.confidence_interval),
                    "samples": metric.samples,
                }
                for name, metric in self.metrics.items()
            }
        if self.accuracy is not None:
            payload["accuracy"] = {
                "mean": self.accuracy.mean,
                "confidence_interval": list(self.accuracy.confidence_interval),
                "samples": self.accuracy.samples,
            }
        return payload

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


def _wilson_ci(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """95 % Wilson score interval for a proportion — exact at small n, unlike Wald."""
    if n == 0:
        return (0.0, 0.0)
    z2 = z * z
    centre = (p + z2 / (2 * n)) / (1 + z2 / n)
    margin = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return (max(0.0, centre - margin), min(1.0, centre + margin))


class Evaluator:
    """Accuracy evaluator with Wilson 95 % confidence intervals."""

    def evaluate(self, dataset: JsonlDataset, predictions: list[str]) -> EvaluationResult:
        if len(predictions) != len(dataset.records):
            raise ValueError("prediction count does not match dataset size")

        labels = [str(record.get("response", "")) for record in dataset.records]
        matches = [int(pred == label) for pred, label in zip(predictions, labels, strict=True)]
        n = len(matches)
        sample_mean = mean(matches) if matches else 0.0
        ci = _wilson_ci(sample_mean, n)
        summary = MetricSummary(mean=sample_mean, confidence_interval=ci, samples=n)
        return EvaluationResult(
            accuracy=summary,
            metrics={"accuracy": summary},
            metadata={"dataset_size": n},
        )


class MultiMetricEvaluator:
    """Variance-aware evaluator with pluggable metric functions.

    Defaults to [ExactMatch, TokenF1, BLEU, ROUGE-L] — pass *metrics* to override.
    """

    def __init__(self, metrics: list[MetricFn] | None = None) -> None:
        self.metrics: list[MetricFn] = (
            list(metrics) if metrics is not None else list(DEFAULT_METRICS)
        )

    def evaluate(self, dataset: JsonlDataset, predictions: list[str]) -> EvaluationResult:
        if len(predictions) != len(dataset.records):
            raise ValueError("prediction count does not match dataset size")

        labels = [str(record.get("response", "")) for record in dataset.records]
        n = len(predictions)
        summaries: dict[str, MetricSummary] = {}

        for metric in self.metrics:
            scores = [
                metric.score(pred, label)
                for pred, label in zip(predictions, labels, strict=True)
            ]
            m = mean(scores) if scores else 0.0
            ci = _wilson_ci(m, n)
            summaries[metric.name] = MetricSummary(mean=m, confidence_interval=ci, samples=n)

        accuracy = summaries.get("exact_match") or (
            next(iter(summaries.values())) if summaries else None
        )
        return EvaluationResult(
            accuracy=accuracy,
            metrics=summaries,
            metadata={"dataset_size": n, "metrics": [m.name for m in self.metrics]},
        )


class RegressionGate:
    """Fails a run when the current result is materially below a baseline."""

    def __init__(self, baseline: float, threshold: float) -> None:
        self.baseline = baseline
        self.threshold = threshold

    def check(self, result: EvaluationResult) -> bool:
        accuracy = result.accuracy or (
            next(iter(result.metrics.values()), None) if result.metrics else None
        )
        if accuracy is None:
            return False
        return (accuracy.mean - self.baseline) >= -self.threshold
