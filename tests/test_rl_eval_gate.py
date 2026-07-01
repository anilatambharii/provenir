from __future__ import annotations

from dataclasses import dataclass

import pytest

from provenir.data.dataset import JsonlDataset
from provenir.train.rl_eval_gate import GateDecision, RLEvalGate, RLGateConfig


def _eval_ds() -> JsonlDataset:
    return JsonlDataset.from_records([
        {"prompt": "q1", "response": "answer1"},
        {"prompt": "q2", "response": "answer2"},
        {"prompt": "q3", "response": "answer3"},
    ])


def _preds_correct() -> list[str]:
    return ["answer1", "answer2", "answer3"]


def _preds_wrong() -> list[str]:
    return ["wrong", "wrong", "wrong"]


@dataclass(frozen=True)
class _Signal:
    """Object-shaped hacking signal for duck-typing tests."""

    severity: str


# --------------------------------------------------------------------------
# RLGateConfig validation
# --------------------------------------------------------------------------


class TestRLGateConfig:
    def test_defaults(self) -> None:
        cfg = RLGateConfig()
        assert cfg.eval_every == 1
        assert cfg.primary_metric == "exact_match"
        assert cfg.max_contamination_rate == 0.0
        assert cfg.halt_on_hacking is True

    def test_is_frozen(self) -> None:
        cfg = RLGateConfig()
        with pytest.raises((AttributeError, TypeError)):
            cfg.eval_every = 5  # type: ignore[misc]

    def test_zero_eval_every_raises(self) -> None:
        with pytest.raises(ValueError, match="eval_every"):
            RLGateConfig(eval_every=0)

    def test_empty_primary_metric_raises(self) -> None:
        with pytest.raises(ValueError, match="primary_metric"):
            RLGateConfig(primary_metric="")

    def test_regression_baseline_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError, match="regression_baseline"):
            RLGateConfig(regression_baseline=1.5)

    def test_negative_regression_threshold_raises(self) -> None:
        with pytest.raises(ValueError, match="regression_threshold"):
            RLGateConfig(regression_threshold=-0.01)

    def test_max_contamination_rate_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError, match="max_contamination_rate"):
            RLGateConfig(max_contamination_rate=2.0)

    def test_min_primary_score_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError, match="min_primary_score"):
            RLGateConfig(min_primary_score=-0.1)

    def test_valid_boundary_values(self) -> None:
        cfg = RLGateConfig(
            regression_baseline=1.0,
            max_contamination_rate=1.0,
            min_primary_score=0.0,
        )
        assert cfg.regression_baseline == 1.0


# --------------------------------------------------------------------------
# eval scheduling
# --------------------------------------------------------------------------


class TestScheduling:
    def test_no_eval_between_steps(self) -> None:
        gate = RLEvalGate(RLGateConfig(eval_every=5), _eval_ds())
        decision = gate.on_iteration(1, _preds_correct())
        assert decision.should_halt is False
        assert decision.reasons == []
        assert gate.history == []

    def test_eval_runs_on_multiple(self) -> None:
        gate = RLEvalGate(RLGateConfig(eval_every=5), _eval_ds())
        gate.on_iteration(5, _preds_correct())
        assert len(gate.history) == 1

    def test_step_zero_triggers_eval(self) -> None:
        gate = RLEvalGate(RLGateConfig(eval_every=5), _eval_ds())
        gate.on_iteration(0, _preds_correct())
        assert len(gate.history) == 1

    def test_off_eval_step_not_recorded(self) -> None:
        gate = RLEvalGate(RLGateConfig(eval_every=3), _eval_ds())
        gate.on_iteration(1, _preds_correct())
        gate.on_iteration(2, _preds_correct())
        gate.on_iteration(3, _preds_correct())
        assert [e["step"] for e in gate.history] == [3]


# --------------------------------------------------------------------------
# clean path
# --------------------------------------------------------------------------


class TestCleanPath:
    def test_clean_no_halt(self) -> None:
        gate = RLEvalGate(RLGateConfig(), _eval_ds())
        decision = gate.on_iteration(0, _preds_correct())
        assert decision.should_halt is False
        assert decision.should_warn is False
        assert decision.reasons == []

    def test_clean_with_baseline_no_halt(self) -> None:
        gate = RLEvalGate(
            RLGateConfig(regression_baseline=0.5, regression_threshold=0.1),
            _eval_ds(),
        )
        decision = gate.on_iteration(0, _preds_correct())
        assert decision.should_halt is False


# --------------------------------------------------------------------------
# regression
# --------------------------------------------------------------------------


class TestRegression:
    def test_regression_halt_below_baseline(self) -> None:
        gate = RLEvalGate(
            RLGateConfig(regression_baseline=0.9, regression_threshold=0.05),
            _eval_ds(),
        )
        decision = gate.on_iteration(0, _preds_wrong())
        assert decision.should_halt is True
        assert any("regression" in r for r in decision.reasons)

    def test_no_baseline_no_regression_check(self) -> None:
        gate = RLEvalGate(RLGateConfig(), _eval_ds())
        decision = gate.on_iteration(0, _preds_wrong())
        assert not any("regression" in r for r in decision.reasons)

    def test_within_threshold_no_halt(self) -> None:
        # score 1.0, baseline 1.0, drop 0 → passes.
        gate = RLEvalGate(
            RLGateConfig(regression_baseline=1.0, regression_threshold=0.05),
            _eval_ds(),
        )
        decision = gate.on_iteration(0, _preds_correct())
        assert decision.should_halt is False


# --------------------------------------------------------------------------
# score floor
# --------------------------------------------------------------------------


class TestScoreFloor:
    def test_floor_halt(self) -> None:
        gate = RLEvalGate(RLGateConfig(min_primary_score=0.5), _eval_ds())
        decision = gate.on_iteration(0, _preds_wrong())
        assert decision.should_halt is True
        assert any("score floor" in r for r in decision.reasons)

    def test_floor_pass(self) -> None:
        gate = RLEvalGate(RLGateConfig(min_primary_score=0.5), _eval_ds())
        decision = gate.on_iteration(0, _preds_correct())
        assert not any("score floor" in r for r in decision.reasons)

    def test_no_floor_configured(self) -> None:
        gate = RLEvalGate(RLGateConfig(), _eval_ds())
        decision = gate.on_iteration(0, _preds_wrong())
        assert not any("score floor" in r for r in decision.reasons)


# --------------------------------------------------------------------------
# contamination (guard_eval_set)
# --------------------------------------------------------------------------


class TestContamination:
    def test_leaked_eval_row_halts(self) -> None:
        # Train set contains an exact copy of an eval prompt → contaminated.
        train = JsonlDataset.from_records([
            {"prompt": "q1", "response": "answer1"},
            {"prompt": "totally different training row", "response": "x"},
        ])
        gate = RLEvalGate(RLGateConfig(max_contamination_rate=0.0), _eval_ds())
        report = gate.guard_eval_set(train)
        assert not report.is_clean
        decision = gate.on_iteration(0, _preds_correct())
        assert decision.should_halt is True
        assert any("contamination" in r for r in decision.reasons)

    def test_clean_train_no_contamination_halt(self) -> None:
        train = JsonlDataset.from_records([
            {"prompt": "some unrelated training prompt", "response": "x"},
            {"prompt": "another unrelated training prompt", "response": "y"},
        ])
        gate = RLEvalGate(RLGateConfig(), _eval_ds())
        report = gate.guard_eval_set(train)
        assert report.is_clean
        decision = gate.on_iteration(0, _preds_correct())
        assert not any("contamination" in r for r in decision.reasons)

    def test_contamination_within_tolerance_no_halt(self) -> None:
        # One of three eval rows leaks → rate 1/2 train rows contaminated.
        train = JsonlDataset.from_records([
            {"prompt": "q1", "response": "answer1"},
            {"prompt": "unrelated", "response": "x"},
        ])
        gate = RLEvalGate(RLGateConfig(max_contamination_rate=0.9), _eval_ds())
        report = gate.guard_eval_set(train)
        assert report.contamination_rate <= 0.9
        decision = gate.on_iteration(0, _preds_correct())
        assert not any("contamination" in r for r in decision.reasons)

    def test_guard_eval_set_returns_report(self) -> None:
        train = JsonlDataset.from_records([{"prompt": "x", "response": "y"}])
        gate = RLEvalGate(RLGateConfig(), _eval_ds())
        report = gate.guard_eval_set(train)
        assert report.train_size == 1
        assert report.eval_size == 3


# --------------------------------------------------------------------------
# reward hacking (duck-typed)
# --------------------------------------------------------------------------


class TestHacking:
    def test_critical_dict_signal_halts(self) -> None:
        gate = RLEvalGate(RLGateConfig(), _eval_ds())
        decision = gate.on_iteration(
            0, _preds_correct(), hacking_signals=[{"severity": "critical"}]
        )
        assert decision.should_halt is True
        assert any("reward hacking" in r for r in decision.reasons)

    def test_critical_object_signal_halts(self) -> None:
        gate = RLEvalGate(RLGateConfig(), _eval_ds())
        decision = gate.on_iteration(
            0, _preds_correct(), hacking_signals=[_Signal("critical")]
        )
        assert decision.should_halt is True
        assert any("reward hacking" in r for r in decision.reasons)

    def test_non_critical_signal_no_halt(self) -> None:
        gate = RLEvalGate(RLGateConfig(), _eval_ds())
        decision = gate.on_iteration(
            0,
            _preds_correct(),
            hacking_signals=[{"severity": "warning"}, _Signal("info")],
        )
        assert decision.should_halt is False
        assert not any("reward hacking" in r for r in decision.reasons)

    def test_halt_on_hacking_disabled(self) -> None:
        gate = RLEvalGate(RLGateConfig(halt_on_hacking=False), _eval_ds())
        decision = gate.on_iteration(
            0, _preds_correct(), hacking_signals=[{"severity": "critical"}]
        )
        assert decision.should_halt is False

    def test_no_signals_no_halt(self) -> None:
        gate = RLEvalGate(RLGateConfig(), _eval_ds())
        decision = gate.on_iteration(0, _preds_correct(), hacking_signals=None)
        assert decision.should_halt is False

    def test_mixed_signals_with_one_critical_halts(self) -> None:
        gate = RLEvalGate(RLGateConfig(), _eval_ds())
        decision = gate.on_iteration(
            0,
            _preds_correct(),
            hacking_signals=[{"severity": "info"}, _Signal("critical")],
        )
        assert decision.should_halt is True


# --------------------------------------------------------------------------
# combined reasons
# --------------------------------------------------------------------------


class TestCombined:
    def test_multiple_reasons_accumulate(self) -> None:
        gate = RLEvalGate(
            RLGateConfig(
                regression_baseline=0.9,
                regression_threshold=0.05,
                min_primary_score=0.5,
            ),
            _eval_ds(),
        )
        decision = gate.on_iteration(
            0, _preds_wrong(), hacking_signals=[{"severity": "critical"}]
        )
        assert decision.should_halt is True
        assert len(decision.reasons) >= 3

    def test_warn_when_reasons_but_no_halt(self) -> None:
        # halt_on_hacking disabled: a critical signal is ignored, so no reason
        # is recorded and there is nothing to warn about here — assert clean.
        gate = RLEvalGate(RLGateConfig(halt_on_hacking=False), _eval_ds())
        decision = gate.on_iteration(
            0, _preds_correct(), hacking_signals=[{"severity": "critical"}]
        )
        assert decision.should_warn is False
        assert decision.reasons == []


# --------------------------------------------------------------------------
# GateDecision
# --------------------------------------------------------------------------


class TestGateDecision:
    def test_to_dict(self) -> None:
        decision = GateDecision(
            should_halt=True, should_warn=False, reasons=["r1"], step=7
        )
        payload = decision.to_dict()
        assert payload == {
            "should_halt": True,
            "should_warn": False,
            "reasons": ["r1"],
            "step": 7,
        }

    def test_to_dict_reasons_is_copy(self) -> None:
        decision = GateDecision(
            should_halt=False, should_warn=False, reasons=["r1"], step=0
        )
        payload = decision.to_dict()
        payload["reasons"].append("r2")
        assert decision.reasons == ["r1"]

    def test_is_frozen(self) -> None:
        decision = GateDecision(
            should_halt=False, should_warn=False, reasons=[], step=0
        )
        with pytest.raises((AttributeError, TypeError)):
            decision.should_halt = True  # type: ignore[misc]


# --------------------------------------------------------------------------
# introspection
# --------------------------------------------------------------------------


class TestIntrospection:
    def test_best_step_none_before_eval(self) -> None:
        gate = RLEvalGate(RLGateConfig(eval_every=100), _eval_ds())
        assert gate.best_step() is None

    def test_best_step_single(self) -> None:
        gate = RLEvalGate(RLGateConfig(), _eval_ds())
        gate.on_iteration(0, _preds_correct())
        assert gate.best_step() == 0

    def test_best_step_selects_highest(self) -> None:
        gate = RLEvalGate(RLGateConfig(), _eval_ds())
        gate.on_iteration(0, _preds_wrong())
        gate.on_iteration(1, _preds_correct())
        assert gate.best_step() == 1

    def test_history_grows(self) -> None:
        gate = RLEvalGate(RLGateConfig(), _eval_ds())
        for i in range(3):
            gate.on_iteration(i, _preds_correct())
        assert len(gate.history) == 3

    def test_history_is_copy(self) -> None:
        gate = RLEvalGate(RLGateConfig(), _eval_ds())
        gate.on_iteration(0, _preds_correct())
        h = gate.history
        h.clear()
        assert len(gate.history) == 1

    def test_summary_shape(self) -> None:
        gate = RLEvalGate(RLGateConfig(min_primary_score=0.5), _eval_ds())
        gate.on_iteration(0, _preds_correct())
        gate.on_iteration(1, _preds_wrong())
        summary = gate.summary()
        assert summary["evaluations"] == 2
        assert summary["best_step"] == 0
        assert summary["halt_steps"] == [1]

    def test_summary_reflects_contamination(self) -> None:
        train = JsonlDataset.from_records([{"prompt": "q1", "response": "answer1"}])
        gate = RLEvalGate(RLGateConfig(), _eval_ds())
        gate.guard_eval_set(train)
        summary = gate.summary()
        assert summary["eval_compromised"] is True
        assert summary["contamination_rate"] == pytest.approx(1.0)
