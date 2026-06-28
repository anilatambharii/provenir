from __future__ import annotations

import pytest

from provenir.data.dataset import JsonlDataset
from provenir.eval.harness import EvaluationResult
from provenir.train.eval_callback import EvalCallback, EvalCallbackConfig


def _ds() -> JsonlDataset:
    return JsonlDataset.from_records([
        {"prompt": "q1", "response": "answer1"},
        {"prompt": "q2", "response": "answer2"},
        {"prompt": "q3", "response": "answer3"},
    ])


def _preds_correct() -> list[str]:
    return ["answer1", "answer2", "answer3"]


def _preds_wrong() -> list[str]:
    return ["wrong", "wrong", "wrong"]


class TestEvalCallbackConfig:
    def test_default_eval_steps(self) -> None:
        assert EvalCallbackConfig().eval_steps == 100

    def test_zero_eval_steps_raises(self) -> None:
        with pytest.raises(ValueError, match="eval_steps"):
            EvalCallbackConfig(eval_steps=0)

    def test_negative_threshold_raises(self) -> None:
        with pytest.raises(ValueError, match="regression_threshold"):
            EvalCallbackConfig(regression_threshold=-0.1)

    def test_is_frozen(self) -> None:
        cfg = EvalCallbackConfig()
        with pytest.raises((AttributeError, TypeError)):
            cfg.eval_steps = 50  # type: ignore[misc]


class TestEvalCallbackOnStep:
    def test_returns_none_between_eval_steps(self) -> None:
        cb = EvalCallback(EvalCallbackConfig(eval_steps=10), _ds())
        assert cb.on_step(1, _preds_correct()) is None

    def test_returns_result_on_eval_step(self) -> None:
        cb = EvalCallback(EvalCallbackConfig(eval_steps=10), _ds())
        result = cb.on_step(10, _preds_correct())
        assert result is not None

    def test_step_zero_triggers_eval(self) -> None:
        cb = EvalCallback(EvalCallbackConfig(eval_steps=1), _ds())
        result = cb.on_step(0, _preds_correct())
        assert result is not None

    def test_eval_result_has_metrics(self) -> None:
        cb = EvalCallback(EvalCallbackConfig(eval_steps=1), _ds())
        result = cb.on_step(0, _preds_correct())
        assert result is not None
        assert result.metrics is not None

    def test_history_grows_with_evals(self) -> None:
        cb = EvalCallback(EvalCallbackConfig(eval_steps=1), _ds())
        for i in range(3):
            cb.on_step(i, _preds_correct())
        assert len(cb.history) == 3

    def test_history_is_copy(self) -> None:
        cb = EvalCallback(EvalCallbackConfig(eval_steps=1), _ds())
        cb.on_step(0, _preds_correct())
        h = cb.history
        h.clear()
        assert len(cb.history) == 1


class TestEvalCallbackEarlyStopping:
    def test_no_stop_without_patience(self) -> None:
        cb = EvalCallback(EvalCallbackConfig(eval_steps=1), _ds())
        for i in range(10):
            cb.on_step(i, _preds_wrong())
        assert not cb.should_stop_early()

    def test_stops_after_patience_exhausted(self) -> None:
        cb = EvalCallback(
            EvalCallbackConfig(eval_steps=1, early_stopping_patience=2), _ds()
        )
        # First eval establishes best; next two don't improve → stop
        cb.on_step(0, _preds_correct())
        cb.on_step(1, _preds_wrong())
        cb.on_step(2, _preds_wrong())
        assert cb.should_stop_early()

    def test_reset_on_improvement(self) -> None:
        cb = EvalCallback(
            EvalCallbackConfig(eval_steps=1, early_stopping_patience=2), _ds()
        )
        cb.on_step(0, _preds_wrong())
        cb.on_step(1, _preds_wrong())
        # Improve — counter resets
        cb.on_step(2, _preds_correct())
        cb.on_step(3, _preds_wrong())
        assert not cb.should_stop_early()


class TestEvalCallbackRegressionGate:
    def test_no_baseline_always_passes(self) -> None:
        cb = EvalCallback(EvalCallbackConfig(), _ds())
        result = EvaluationResult()
        assert cb.check_regression(result) is True

    def test_baseline_pass(self) -> None:
        cb = EvalCallback(
            EvalCallbackConfig(
                regression_baseline=0.3,
                regression_threshold=0.1,
            ),
            _ds(),
        )
        cb2 = EvalCallback(EvalCallbackConfig(eval_steps=1), _ds())
        result = cb2.on_step(0, _preds_correct())
        assert result is not None
        assert cb.check_regression(result) is True


class TestEvalCallbackBestStep:
    def test_best_step_none_before_any_eval(self) -> None:
        cb = EvalCallback(EvalCallbackConfig(eval_steps=100), _ds())
        assert cb.best_step() is None

    def test_best_step_single_eval(self) -> None:
        cb = EvalCallback(EvalCallbackConfig(eval_steps=1), _ds())
        cb.on_step(0, _preds_correct())
        assert cb.best_step() == 0

    def test_best_step_selects_higher_score(self) -> None:
        cb = EvalCallback(EvalCallbackConfig(eval_steps=1), _ds())
        cb.on_step(0, _preds_wrong())
        cb.on_step(1, _preds_correct())
        assert cb.best_step() == 1
