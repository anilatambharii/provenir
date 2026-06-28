from __future__ import annotations

import pytest

from provenir.data.dataset import JsonlDataset
from provenir.data.flywheel import DataFlywheel, FlywheelConfig
from provenir.eval.harness import EvaluationResult, MetricSummary
from provenir.eval.judge import StubJudge


def _ds() -> JsonlDataset:
    return JsonlDataset.from_records([
        {"prompt": "q1", "response": "a short answer"},
        {"prompt": "q2", "response": "another short answer"},
    ])


def _result(mean: float) -> EvaluationResult:
    summary = MetricSummary(
        mean=mean, confidence_interval=(max(0, mean - 0.1), min(1, mean + 0.1)), samples=2
    )
    return EvaluationResult(accuracy=summary, metrics={"exact_match": summary})


class TestFlywheelConfig:
    def test_defaults(self) -> None:
        cfg = FlywheelConfig()
        assert cfg.min_score_threshold == 0.5
        assert cfg.max_variants_per_failure == 3
        assert cfg.quality_filter_threshold == 0.6

    def test_invalid_threshold_raises(self) -> None:
        with pytest.raises(ValueError, match="min_score_threshold"):
            FlywheelConfig(min_score_threshold=1.5)

    def test_invalid_quality_threshold_raises(self) -> None:
        with pytest.raises(ValueError, match="quality_filter_threshold"):
            FlywheelConfig(quality_filter_threshold=-0.1)

    def test_zero_variants_raises(self) -> None:
        with pytest.raises(ValueError, match="max_variants_per_failure"):
            FlywheelConfig(max_variants_per_failure=0)

    def test_is_frozen(self) -> None:
        cfg = FlywheelConfig()
        with pytest.raises((AttributeError, TypeError)):
            cfg.max_variants_per_failure = 10  # type: ignore[misc]


class TestDataFlywheelMineFailures:
    fw = DataFlywheel(StubJudge())

    def test_no_failures_when_above_threshold(self) -> None:
        result = _result(mean=0.9)
        failures = self.fw.mine_failures(_ds(), result)
        assert failures == []

    def test_all_records_returned_when_below_threshold(self) -> None:
        result = _result(mean=0.1)
        failures = self.fw.mine_failures(_ds(), result)
        assert len(failures) == len(_ds().records)

    def test_no_accuracy_returns_empty(self) -> None:
        result = EvaluationResult()
        assert self.fw.mine_failures(_ds(), result) == []

    def test_at_threshold_no_failures(self) -> None:
        result = _result(mean=0.5)
        assert self.fw.mine_failures(_ds(), result) == []


class TestDataFlywheelGenerateVariants:
    # StubJudge returns score=0.5 on rubric → below default threshold 0.6
    # So by default no variants pass the quality filter.
    fw_loose = DataFlywheel(
        StubJudge(),
        config=FlywheelConfig(quality_filter_threshold=0.3),
    )
    fw_strict = DataFlywheel(StubJudge())

    _FAILURE = {"prompt": "q", "response": "original answer"}

    def test_variants_within_max(self) -> None:
        variants = self.fw_loose.generate_variants(self._FAILURE, n=3)
        assert len(variants) <= 3

    def test_variants_contain_flywheel_meta(self) -> None:
        variants = self.fw_loose.generate_variants(self._FAILURE, n=1)
        if variants:
            assert "_flywheel" in variants[0]

    def test_strict_quality_filter_may_drop_all(self) -> None:
        # StubJudge returns 0.5; threshold is 0.6 → all dropped
        variants = self.fw_strict.generate_variants(self._FAILURE, n=3)
        assert variants == []


class TestDataFlywheelRun:
    def test_run_returns_dataset(self) -> None:
        fw = DataFlywheel(StubJudge())
        ds = _ds()
        result = _result(mean=0.9)
        augmented = fw.run(ds, result)
        assert isinstance(augmented, JsonlDataset)

    def test_run_preserves_original_records_when_no_failures(self) -> None:
        fw = DataFlywheel(StubJudge())
        ds = _ds()
        result = _result(mean=0.9)  # above threshold → no failures
        augmented = fw.run(ds, result)
        assert len(augmented.records) == len(ds.records)

    def test_run_augmented_dataset_is_superset(self) -> None:
        fw = DataFlywheel(
            StubJudge(),
            config=FlywheelConfig(quality_filter_threshold=0.3, max_variants_per_failure=2),
        )
        ds = _ds()
        result = _result(mean=0.1)  # below threshold → failures found
        augmented = fw.run(ds, result)
        assert len(augmented.records) >= len(ds.records)
