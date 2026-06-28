from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from provenir.data.dataset import JsonlDataset
from provenir.eval.harness import EvaluationResult, MultiMetricEvaluator, RegressionGate
from provenir.eval.metrics import MetricFn


@dataclass(frozen=True)
class EvalCallbackConfig:
    """Configuration for mid-training evaluation callbacks."""

    eval_steps: int = 100
    primary_metric: str = "exact_match"
    early_stopping_patience: int | None = None
    regression_baseline: float | None = None
    regression_threshold: float = 0.05

    def __post_init__(self) -> None:
        if self.eval_steps < 1:
            raise ValueError(f"eval_steps must be >= 1, got {self.eval_steps}")
        if self.regression_threshold < 0.0:
            raise ValueError(
                f"regression_threshold must be >= 0, got {self.regression_threshold}"
            )


class EvalCallback:
    """Runs :class:`MultiMetricEvaluator` every N training steps.

    Bridges the training loop and the evaluation pipeline.  Each call to
    :meth:`on_step` returns a full :class:`EvaluationResult` on eval steps
    and ``None`` otherwise.  Tracks per-step history and early-stopping
    patience automatically.

    Usage inside a training loop::

        callback = EvalCallback(EvalCallbackConfig(eval_steps=50), eval_dataset)
        for step, preds in enumerate(training_loop()):
            result = callback.on_step(step, preds)
            if callback.should_stop_early():
                break
    """

    def __init__(
        self,
        config: EvalCallbackConfig,
        dataset: JsonlDataset,
        metrics: list[MetricFn] | None = None,
    ) -> None:
        self.config = config
        self.dataset = dataset
        self.evaluator = MultiMetricEvaluator(metrics=metrics)
        self._history: list[dict[str, Any]] = []
        self._no_improve_count = 0
        self._best_score: float | None = None

    def on_step(self, step: int, predictions: list[str]) -> EvaluationResult | None:
        """Evaluate when *step* is a multiple of *eval_steps*; else return None."""
        if step % self.config.eval_steps != 0:
            return None

        result = self.evaluator.evaluate(self.dataset, predictions)

        if result.metrics:
            metric = result.metrics.get(self.config.primary_metric)
            if metric is None:
                metric = next(iter(result.metrics.values()))
            current = metric.mean
            if self._best_score is None or current > self._best_score:
                self._best_score = current
                self._no_improve_count = 0
            else:
                self._no_improve_count += 1

        self._history.append(
            {
                "step": step,
                "result": result.to_dict(),
                "best_score": self._best_score,
            }
        )
        return result

    def should_stop_early(self) -> bool:
        """True when patience budget is exhausted."""
        if self.config.early_stopping_patience is None:
            return False
        return self._no_improve_count >= self.config.early_stopping_patience

    def check_regression(self, result: EvaluationResult) -> bool:
        """True when *result* passes the configured regression gate."""
        if self.config.regression_baseline is None:
            return True
        gate = RegressionGate(
            baseline=self.config.regression_baseline,
            threshold=self.config.regression_threshold,
        )
        return gate.check(result)

    def best_step(self) -> int | None:
        """Step index with the highest primary-metric score, or None."""
        if not self._history:
            return None
        key = self.config.primary_metric

        def _score(entry: dict[str, Any]) -> float:
            return float(
                entry.get("result", {})
                .get("metrics", {})
                .get(key, {})
                .get("mean", 0.0)
            )

        return int(max(self._history, key=_score)["step"])

    @property
    def history(self) -> list[dict[str, Any]]:
        return list(self._history)
