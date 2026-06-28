from __future__ import annotations

from typing import Any, Mapping

import pytest

from provenir.core.abstractions import RewardFn
from provenir.rewards.composition import (
    ClampedReward,
    MaxReward,
    MinReward,
    ThresholdGatedReward,
    WeightedSumReward,
)

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


class _Fixed(RewardFn):
    """Returns a constant score. Optionally emits a gameability warning."""

    def __init__(self, value: float, warning: str = "") -> None:
        self._value = value
        self._warning = warning

    def score(self, trajectory: Mapping[str, Any]) -> float:
        return self._value

    def gameability_check(self, trajectory: Mapping[str, Any]) -> list[str]:
        return [self._warning] if self._warning else []


_ZERO = _Fixed(0.0)
_HALF = _Fixed(0.5)
_ONE = _Fixed(1.0)
_WARN = _Fixed(0.8, warning="length_inflation")


# ---------------------------------------------------------------------------
# WeightedSumReward
# ---------------------------------------------------------------------------


class TestWeightedSumReward:
    def test_single_reward_returns_its_score(self) -> None:
        r = WeightedSumReward([(_ONE, 1.0)])
        assert r.score({}) == pytest.approx(1.0)

    def test_equal_weights_is_simple_average(self) -> None:
        r = WeightedSumReward([(_ZERO, 1.0), (_ONE, 1.0)])
        assert r.score({}) == pytest.approx(0.5)

    def test_unequal_weights_normalised(self) -> None:
        # 0.8 * (2/3) + 0.2 * (1/3) = (1.6 + 0.2) / 3 = 0.6
        r = WeightedSumReward([(_Fixed(0.8), 2.0), (_Fixed(0.2), 1.0)])
        assert r.score({}) == pytest.approx(0.6)

    def test_weights_already_summing_to_one(self) -> None:
        r = WeightedSumReward([(_Fixed(0.6), 0.7), (_Fixed(0.2), 0.3)])
        expected = 0.6 * 0.7 + 0.2 * 0.3
        assert r.score({}) == pytest.approx(expected)

    def test_zero_weight_component_ignored(self) -> None:
        r = WeightedSumReward([(_ONE, 1.0), (_ZERO, 0.0)])
        assert r.score({}) == pytest.approx(1.0)

    def test_all_zero_weights_returns_zero(self) -> None:
        r = WeightedSumReward([(_ONE, 0.0), (_ONE, 0.0)])
        assert r.score({}) == pytest.approx(0.0)

    def test_kind_is_composite(self) -> None:
        assert WeightedSumReward([(_ONE, 1.0)]).kind == "composite"

    def test_empty_list_raises(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            WeightedSumReward([])

    def test_gameability_aggregates_all_warnings(self) -> None:
        r = WeightedSumReward([(_WARN, 1.0), (_Fixed(0.5, warning="format_exploit"), 1.0)])
        issues = r.gameability_check({})
        assert "length_inflation" in issues
        assert "format_exploit" in issues

    def test_gameability_empty_when_no_warnings(self) -> None:
        r = WeightedSumReward([(_ONE, 1.0), (_ZERO, 1.0)])
        assert r.gameability_check({}) == []

    def test_is_reward_fn_subclass(self) -> None:
        r = WeightedSumReward([(_ONE, 1.0)])
        assert isinstance(r, RewardFn)


# ---------------------------------------------------------------------------
# MinReward
# ---------------------------------------------------------------------------


class TestMinReward:
    def test_returns_minimum(self) -> None:
        r = MinReward([_Fixed(0.9), _Fixed(0.3), _Fixed(0.7)])
        assert r.score({}) == pytest.approx(0.3)

    def test_single_reward_passes_through(self) -> None:
        r = MinReward([_Fixed(0.4)])
        assert r.score({}) == pytest.approx(0.4)

    def test_all_equal_returns_that_value(self) -> None:
        r = MinReward([_HALF, _HALF, _HALF])
        assert r.score({}) == pytest.approx(0.5)

    def test_zero_drags_result_to_zero(self) -> None:
        r = MinReward([_ONE, _ZERO, _HALF])
        assert r.score({}) == pytest.approx(0.0)

    def test_kind_is_composite(self) -> None:
        assert MinReward([_ONE]).kind == "composite"

    def test_empty_list_raises(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            MinReward([])

    def test_gameability_aggregates(self) -> None:
        r = MinReward([_WARN, _Fixed(0.5, warning="format_exploit")])
        issues = r.gameability_check({})
        assert "length_inflation" in issues
        assert "format_exploit" in issues

    def test_is_reward_fn_subclass(self) -> None:
        assert isinstance(MinReward([_ONE]), RewardFn)


# ---------------------------------------------------------------------------
# MaxReward
# ---------------------------------------------------------------------------


class TestMaxReward:
    def test_returns_maximum(self) -> None:
        r = MaxReward([_Fixed(0.1), _Fixed(0.9), _Fixed(0.5)])
        assert r.score({}) == pytest.approx(0.9)

    def test_single_reward_passes_through(self) -> None:
        r = MaxReward([_Fixed(0.7)])
        assert r.score({}) == pytest.approx(0.7)

    def test_all_equal_returns_that_value(self) -> None:
        r = MaxReward([_HALF, _HALF])
        assert r.score({}) == pytest.approx(0.5)

    def test_one_lifts_result(self) -> None:
        r = MaxReward([_ZERO, _ZERO, _ONE])
        assert r.score({}) == pytest.approx(1.0)

    def test_kind_is_composite(self) -> None:
        assert MaxReward([_ONE]).kind == "composite"

    def test_empty_list_raises(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            MaxReward([])

    def test_gameability_aggregates(self) -> None:
        r = MaxReward([_WARN, _ONE])
        assert "length_inflation" in r.gameability_check({})

    def test_is_reward_fn_subclass(self) -> None:
        assert isinstance(MaxReward([_ONE]), RewardFn)


# ---------------------------------------------------------------------------
# ThresholdGatedReward
# ---------------------------------------------------------------------------


class TestThresholdGatedReward:
    def test_passes_score_when_above_threshold(self) -> None:
        r = ThresholdGatedReward(_Fixed(0.9), threshold=0.5)
        assert r.score({}) == pytest.approx(0.9)

    def test_passes_score_when_equal_to_threshold(self) -> None:
        r = ThresholdGatedReward(_Fixed(0.5), threshold=0.5)
        assert r.score({}) == pytest.approx(0.5)

    def test_gates_to_zero_when_below_threshold(self) -> None:
        r = ThresholdGatedReward(_Fixed(0.4), threshold=0.5)
        assert r.score({}) == pytest.approx(0.0)

    def test_default_threshold_is_half(self) -> None:
        r = ThresholdGatedReward(_Fixed(0.3))
        assert r.score({}) == pytest.approx(0.0)
        r2 = ThresholdGatedReward(_Fixed(0.7))
        assert r2.score({}) == pytest.approx(0.7)

    def test_threshold_zero_always_passes(self) -> None:
        r = ThresholdGatedReward(_ZERO, threshold=0.0)
        assert r.score({}) == pytest.approx(0.0)

    def test_threshold_one_only_passes_perfect(self) -> None:
        r = ThresholdGatedReward(_Fixed(0.99), threshold=1.0)
        assert r.score({}) == pytest.approx(0.0)
        r2 = ThresholdGatedReward(_ONE, threshold=1.0)
        assert r2.score({}) == pytest.approx(1.0)

    def test_invalid_threshold_raises(self) -> None:
        with pytest.raises(ValueError, match="threshold must be in"):
            ThresholdGatedReward(_ONE, threshold=1.5)
        with pytest.raises(ValueError, match="threshold must be in"):
            ThresholdGatedReward(_ONE, threshold=-0.1)

    def test_kind_is_gated(self) -> None:
        assert ThresholdGatedReward(_ONE).kind == "gated"

    def test_gameability_delegates_to_inner(self) -> None:
        r = ThresholdGatedReward(_WARN)
        assert "length_inflation" in r.gameability_check({})

    def test_is_reward_fn_subclass(self) -> None:
        assert isinstance(ThresholdGatedReward(_ONE), RewardFn)


# ---------------------------------------------------------------------------
# ClampedReward
# ---------------------------------------------------------------------------


class TestClampedReward:
    def test_clamps_above_hi(self) -> None:
        r = ClampedReward(_Fixed(1.5), lo=0.0, hi=1.0)
        assert r.score({}) == pytest.approx(1.0)

    def test_clamps_below_lo(self) -> None:
        r = ClampedReward(_Fixed(-0.5), lo=0.0, hi=1.0)
        assert r.score({}) == pytest.approx(0.0)

    def test_in_range_unchanged(self) -> None:
        r = ClampedReward(_Fixed(0.7), lo=0.0, hi=1.0)
        assert r.score({}) == pytest.approx(0.7)

    def test_floor_clamp(self) -> None:
        r = ClampedReward(_ZERO, lo=0.1, hi=1.0)
        assert r.score({}) == pytest.approx(0.1)

    def test_ceiling_clamp(self) -> None:
        r = ClampedReward(_ONE, lo=0.0, hi=0.9)
        assert r.score({}) == pytest.approx(0.9)

    def test_lo_equals_hi_pins_to_constant(self) -> None:
        r = ClampedReward(_Fixed(0.3), lo=0.5, hi=0.5)
        assert r.score({}) == pytest.approx(0.5)

    def test_invalid_range_raises(self) -> None:
        with pytest.raises(ValueError, match="lo .* must be <= hi"):
            ClampedReward(_ONE, lo=0.9, hi=0.1)

    def test_kind_is_clamped(self) -> None:
        assert ClampedReward(_ONE).kind == "clamped"

    def test_gameability_delegates_to_inner(self) -> None:
        r = ClampedReward(_WARN)
        assert "length_inflation" in r.gameability_check({})

    def test_is_reward_fn_subclass(self) -> None:
        assert isinstance(ClampedReward(_ONE), RewardFn)


# ---------------------------------------------------------------------------
# Composition: nesting and integration
# ---------------------------------------------------------------------------


class TestComposition:
    def test_weighted_sum_of_min_and_max(self) -> None:
        # min([0.2, 0.8]) = 0.2 ; max([0.2, 0.8]) = 0.8
        # weighted_sum: 0.5 * 0.2 + 0.5 * 0.8 = 0.5
        inner = WeightedSumReward([
            (MinReward([_Fixed(0.2), _Fixed(0.8)]), 1.0),
            (MaxReward([_Fixed(0.2), _Fixed(0.8)]), 1.0),
        ])
        assert inner.score({}) == pytest.approx(0.5)

    def test_gated_inside_weighted_sum(self) -> None:
        # gate fails (0.3 < 0.5) → 0.0; other component is 1.0
        # weighted sum = (0.0 + 1.0) / 2 = 0.5
        r = WeightedSumReward([
            (ThresholdGatedReward(_Fixed(0.3), threshold=0.5), 1.0),
            (_ONE, 1.0),
        ])
        assert r.score({}) == pytest.approx(0.5)

    def test_clamped_inside_weighted_sum(self) -> None:
        r = WeightedSumReward([
            (ClampedReward(_ONE, hi=0.6), 1.0),
            (_ZERO, 1.0),
        ])
        assert r.score({}) == pytest.approx(0.3)

    def test_gameability_propagates_through_nesting(self) -> None:
        inner = MinReward([_WARN, _ONE])
        outer = WeightedSumReward([(inner, 1.0)])
        assert "length_inflation" in outer.gameability_check({})

    def test_three_level_nesting_score_correct(self) -> None:
        # ClampedReward(ThresholdGatedReward(Fixed(0.9), 0.5)) → clamp(0.9, 0, 0.8) = 0.8
        r = ClampedReward(ThresholdGatedReward(_Fixed(0.9), threshold=0.5), lo=0.0, hi=0.8)
        assert r.score({}) == pytest.approx(0.8)
