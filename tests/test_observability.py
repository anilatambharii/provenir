from __future__ import annotations

import pytest

from provenir.core.abstractions import RunManifest
from provenir.eval.harness import EvaluationResult, MetricSummary
from provenir.train.observability import ObservabilityConfig, TrainingObserver


def _manifest() -> RunManifest:
    return RunManifest(
        config_hash="abc",
        dataset_hash="def",
        seed=0,
        provenance={"backend": "stub"},
    )


class TestObservabilityConfig:
    def test_default_backend_is_none(self) -> None:
        assert ObservabilityConfig().backend == "none"

    def test_default_project(self) -> None:
        assert ObservabilityConfig().project == "provenir"

    def test_log_every_n_steps_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="log_every_n_steps"):
            ObservabilityConfig(log_every_n_steps=0)

    def test_is_frozen(self) -> None:
        cfg = ObservabilityConfig()
        with pytest.raises((AttributeError, TypeError)):
            cfg.backend = "wandb"  # type: ignore[misc]


class TestTrainingObserverNoneBackend:
    """With backend='none' the observer stores events in memory."""

    def setup_method(self) -> None:
        self.observer = TrainingObserver(ObservabilityConfig(), _manifest())

    def test_history_empty_before_start(self) -> None:
        assert self.observer.history == []

    def test_start_sets_active(self) -> None:
        self.observer.start()
        assert self.observer._active

    def test_finish_clears_active(self) -> None:
        self.observer.start()
        self.observer.finish()
        assert not self.observer._active

    def test_log_step_appends_to_history(self) -> None:
        self.observer.start()
        self.observer.log_step(0, {"loss": 2.5})
        assert len(self.observer.history) == 1
        assert self.observer.history[0]["loss"] == 2.5
        assert self.observer.history[0]["step"] == 0

    def test_multiple_steps_all_recorded(self) -> None:
        self.observer.start()
        for i in range(5):
            self.observer.log_step(i, {"loss": float(i)})
        assert len(self.observer.history) == 5

    def test_log_eval_adds_eval_prefix(self) -> None:
        summary = MetricSummary(mean=0.8, confidence_interval=(0.7, 0.9), samples=10)
        result = EvaluationResult(
            metrics={"exact_match": summary},
            accuracy=summary,
        )
        self.observer.start()
        self.observer.log_eval(step=10, result=result)
        entry = self.observer.history[0]
        assert "eval/exact_match" in entry

    def test_log_eval_with_no_metrics_is_noop(self) -> None:
        result = EvaluationResult()
        self.observer.start()
        self.observer.log_eval(step=0, result=result)
        assert self.observer.history == []

    def test_history_is_copy(self) -> None:
        self.observer.start()
        self.observer.log_step(0, {"x": 1.0})
        h = self.observer.history
        h.clear()
        assert len(self.observer.history) == 1
