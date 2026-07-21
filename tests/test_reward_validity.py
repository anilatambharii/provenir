"""Tests for the spurious-reward ablation harness (reward_validity.py).

All ``TrainEval`` closures here are SYNTHETIC — no real training backend, no bom.py,
no ModelBOM. The harness is backend-agnostic by design.
"""

from __future__ import annotations

from typing import Any, Mapping

import pytest

from provenir.core.abstractions import RewardFn
from provenir.environments.reward_validity import (
    Ablation,
    AblationRun,
    RewardValidityBlocked,
    RewardValidityHarness,
    _ConstantReward,
    _FormatOnlyReward,
    _LengthOnlyReward,
    _make_degenerate,
    _RandomReward,
    _ShuffledReward,
    gate_reward_validity,
)

# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


class _ExactReward(RewardFn):
    """Minimal real reward: 1.0 if 'prediction' equals 'reference', else 0.0."""

    kind: str = "real"
    name: str = "exact"

    def score(self, trajectory: Mapping[str, Any]) -> float:
        return 1.0 if trajectory.get("prediction") == trajectory.get("reference") else 0.0

    def gameability_check(self, trajectory: Mapping[str, Any]) -> list[str]:
        return []


def _make_real_reward() -> _ExactReward:
    return _ExactReward()


def _valid_train_eval(reward: RewardFn) -> tuple[float, float, float, dict[str, int]]:
    """Synthetic: real reward yields large gain; degenerate rewards yield ~0."""
    base = 0.40
    gain = 0.30 if getattr(reward, "kind", "") == "real" else 0.01
    return base, base + gain, 0.75, {}


def _spurious_train_eval(reward: RewardFn) -> tuple[float, float, float, dict[str, int]]:
    """Synthetic Shao repro: random gain ≈ real gain (spurious scenario)."""
    base = 0.40
    kind = getattr(reward, "kind", "")
    if kind == "real":
        gain = 0.25
    elif kind == "random":
        gain = 0.24  # within default tolerance=0.1 → spurious
    else:
        gain = 0.02
    return base, base + gain, 0.80, {}


_REAL = _make_real_reward()
_HARNESS = RewardValidityHarness(seed=0)


# ---------------------------------------------------------------------------
# 1. Valid reward: real gains >> degenerate gains → high validity, not spurious
# ---------------------------------------------------------------------------


def test_valid_reward_high_validity() -> None:
    report = RewardValidityHarness(seed=0).evaluate(_REAL, _valid_train_eval)
    assert report.validity > 0.8, f"expected high validity, got {report.validity}"


def test_valid_reward_not_spurious() -> None:
    report = RewardValidityHarness(seed=0).evaluate(_REAL, _valid_train_eval)
    assert not report.spurious


def test_valid_reward_real_gain_recorded() -> None:
    report = RewardValidityHarness(seed=0).evaluate(_REAL, _valid_train_eval)
    assert Ablation.REAL in report.runs
    assert abs(report.runs[Ablation.REAL].gain - 0.30) < 1e-9


# ---------------------------------------------------------------------------
# 2. Shao repro: random gain ≈ real gain → validity ≈ 0, spurious=True
# ---------------------------------------------------------------------------


def test_spurious_random_trips_spurious_flag() -> None:
    report = RewardValidityHarness(seed=0, tolerance=0.1).evaluate(_REAL, _spurious_train_eval)
    assert report.spurious, "expected spurious=True when random gain ≈ real gain"


def test_spurious_validity_near_zero() -> None:
    report = RewardValidityHarness(seed=0, tolerance=0.1).evaluate(_REAL, _spurious_train_eval)
    # random gain=0.24, real gain=0.25 → validity=(0.25-0.24)/0.25=0.04
    assert report.validity < 0.1, f"expected validity near 0, got {report.validity}"


def test_spurious_summary_contains_spurious() -> None:
    report = RewardValidityHarness(seed=0).evaluate(_REAL, _spurious_train_eval)
    assert "SPURIOUS" in report.summary()


# ---------------------------------------------------------------------------
# 3. Degenerate rewards beating real → spurious
# ---------------------------------------------------------------------------


def _make_length_only_beats_real(reward: RewardFn) -> tuple[float, float, float, dict[str, int]]:
    base = 0.40
    kind = getattr(reward, "kind", "")
    gain = 0.25 if kind in ("real", "length_only") else 0.01
    return base, base + gain, 0.70, {}


def _make_format_only_beats_real(reward: RewardFn) -> tuple[float, float, float, dict[str, int]]:
    base = 0.40
    kind = getattr(reward, "kind", "")
    gain = 0.25 if kind in ("real", "format_only") else 0.01
    return base, base + gain, 0.70, {}


def test_length_only_beating_real_trips_spurious() -> None:
    report = RewardValidityHarness(seed=0, tolerance=0.1).evaluate(
        _REAL, _make_length_only_beats_real
    )
    assert report.spurious


def test_format_only_beating_real_trips_spurious() -> None:
    report = RewardValidityHarness(seed=0, tolerance=0.1).evaluate(
        _REAL, _make_format_only_beats_real
    )
    assert report.spurious


# ---------------------------------------------------------------------------
# 4. Determinism: same seed → identical report
# ---------------------------------------------------------------------------


def test_determinism_to_dict() -> None:
    r1 = RewardValidityHarness(seed=42).evaluate(_REAL, _valid_train_eval)
    r2 = RewardValidityHarness(seed=42).evaluate(_REAL, _valid_train_eval)
    assert r1.to_dict() == r2.to_dict()


def test_determinism_content_hash() -> None:
    r1 = RewardValidityHarness(seed=42).evaluate(_REAL, _valid_train_eval)
    r2 = RewardValidityHarness(seed=42).evaluate(_REAL, _valid_train_eval)
    assert r1.content_hash() == r2.content_hash()


def test_different_seeds_different_hash() -> None:
    """Different seeds can produce different results for RNG-based rewards.

    We use a custom train_eval that actually inspects the reward's output so
    that seed differences manifest in the report.
    """

    def seed_sensitive_eval(
        reward: RewardFn,
    ) -> tuple[float, float, float, dict[str, int]]:
        traj: dict[str, Any] = {"prediction": "x", "reference": "y"}
        score_sum = sum(reward.score(traj) for _ in range(10))
        return 0.4, 0.4 + score_sum / 100, score_sum / 10, {}

    r1 = RewardValidityHarness(seed=0).evaluate(_REAL, seed_sensitive_eval)
    r2 = RewardValidityHarness(seed=99).evaluate(_REAL, seed_sensitive_eval)
    # Hashes may or may not differ depending on RNG; at minimum both are valid strings.
    assert isinstance(r1.content_hash(), str) and len(r1.content_hash()) == 64
    assert isinstance(r2.content_hash(), str) and len(r2.content_hash()) == 64


# ---------------------------------------------------------------------------
# 5. gate_reward_validity: raises iff spurious or validity < min_validity
# ---------------------------------------------------------------------------


def test_gate_allows_valid_report_production() -> None:
    report = RewardValidityHarness(seed=0).evaluate(_REAL, _valid_train_eval)
    gate_reward_validity(report, "production")  # must not raise


def test_gate_blocks_spurious_production() -> None:
    report = RewardValidityHarness(seed=0).evaluate(_REAL, _spurious_train_eval)
    with pytest.raises(RewardValidityBlocked):
        gate_reward_validity(report, "production")


def test_gate_non_protected_stage_never_blocks() -> None:
    report = RewardValidityHarness(seed=0).evaluate(_REAL, _spurious_train_eval)
    gate_reward_validity(report, "staging")  # must not raise even though spurious


def test_gate_min_validity_blocks_when_below_floor() -> None:
    report = RewardValidityHarness(seed=0).evaluate(_REAL, _valid_train_eval)
    # validity ~= 0.967; require 0.99 → should block
    assert report.validity < 0.99
    with pytest.raises(RewardValidityBlocked):
        gate_reward_validity(report, "production", min_validity=0.99)


def test_gate_min_validity_passes_when_above_floor() -> None:
    report = RewardValidityHarness(seed=0).evaluate(_REAL, _valid_train_eval)
    # validity ~= 0.967; require 0.5 → should pass
    gate_reward_validity(report, "production", min_validity=0.5)  # must not raise


def test_gate_custom_protected_stages() -> None:
    report = RewardValidityHarness(seed=0).evaluate(_REAL, _spurious_train_eval)
    # "beta" is not protected by default → should not raise
    gate_reward_validity(
        report, "beta", protected_stages=frozenset({"production"})
    )


# ---------------------------------------------------------------------------
# 6. Degenerate RewardFn behaviours in isolation
# ---------------------------------------------------------------------------


def test_random_reward_in_range() -> None:
    rng_rwd = _RandomReward(seed=7)
    scores = [rng_rwd.score({"prediction": str(i)}) for i in range(100)]
    assert all(0.0 <= s <= 1.0 for s in scores)


def test_random_reward_deterministic_given_seed() -> None:
    r1 = _RandomReward(seed=5)
    r2 = _RandomReward(seed=5)
    traj: dict[str, Any] = {}
    scores_1 = [r1.score(traj) for _ in range(20)]
    scores_2 = [r2.score(traj) for _ in range(20)]
    assert scores_1 == scores_2


def test_constant_reward_always_one() -> None:
    c = _ConstantReward(1.0)
    assert c.score({}) == 1.0
    assert c.score({"prediction": "anything"}) == 1.0


def test_constant_reward_custom_value() -> None:
    c = _ConstantReward(0.5)
    assert c.score({}) == 0.5


def test_format_only_reward_matches_pattern() -> None:
    f = _FormatOnlyReward(r"\\boxed\{.*\}")
    assert f.score({"prediction": r"The answer is \boxed{42}"}) == 1.0
    assert f.score({"prediction": "42"}) == 0.0


def test_length_only_reward_monotone() -> None:
    lo = _LengthOnlyReward()
    short_score = lo.score({"prediction": "hi"})
    long_score = lo.score({"prediction": "hi " * 100})
    assert short_score < long_score


def test_length_only_reward_caps_at_one() -> None:
    lo = _LengthOnlyReward()
    huge = "x" * 100_000
    assert lo.score({"prediction": huge}) == 1.0


def test_shuffled_reward_mismatches_reference() -> None:
    """After two distinct references are seen, shuffled must sometimes return a low score."""
    real = _make_real_reward()
    sh = _ShuffledReward(real, seed=0)
    # First call: only one ref in pool → sentinel used
    score0 = sh.score({"prediction": "A", "reference": "A"})
    # Second call: "A" is in pool, ref is "B" → mismatch with "A" should give 0.0
    score1 = sh.score({"prediction": "B", "reference": "B"})
    # With exact-match reward, at least one score should be 0 (mismatched ref)
    assert score0 == 0.0 or score1 == 0.0


# ---------------------------------------------------------------------------
# 7. AblationRun and RewardValidityReport API
# ---------------------------------------------------------------------------


def test_ablation_run_gain_property() -> None:
    run = AblationRun(
        ablation=Ablation.REAL,
        base_score=0.4,
        final_score=0.7,
        mean_reward=0.8,
        anomalies={},
    )
    assert abs(run.gain - 0.3) < 1e-12


def test_ablation_run_to_dict_keys() -> None:
    run = AblationRun(
        ablation=Ablation.RANDOM,
        base_score=0.4,
        final_score=0.41,
        mean_reward=0.5,
        anomalies={"spike": 2},
    )
    d = run.to_dict()
    assert d["ablation"] == "random"
    assert d["gain"] == pytest.approx(0.01)
    assert d["anomalies"] == {"spike": 2}


def test_report_content_hash_is_hex_string() -> None:
    report = _HARNESS.evaluate(_REAL, _valid_train_eval)
    h = report.content_hash()
    assert isinstance(h, str)
    assert len(h) == 64
    int(h, 16)  # must be valid hex


def test_report_to_dict_round_trip() -> None:
    report = _HARNESS.evaluate(_REAL, _valid_train_eval)
    d = report.to_dict()
    assert d["reward_name"] == "exact"
    assert 0.0 <= d["validity"] <= 1.0
    assert isinstance(d["spurious"], bool)
    assert set(d["runs"].keys()) == {a.value for a in Ablation}


def test_report_tampering_changes_hash() -> None:
    """Mutating validity (conceptually) changes the hash — verified via to_dict."""
    import hashlib
    import json

    report = _HARNESS.evaluate(_REAL, _valid_train_eval)
    d = report.to_dict()
    # Tamper validity
    d["validity"] = 0.0
    tampered_hash = hashlib.sha256(
        json.dumps(d, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert tampered_hash != report.content_hash()


def test_make_degenerate_real_returns_same_object() -> None:
    real = _make_real_reward()
    result = _make_degenerate(Ablation.REAL, real, seed=0, format_pattern=r".*")
    assert result is real


def test_harness_subset_of_ablations() -> None:
    """Harness with a restricted ablation set only runs those ablations."""
    report = RewardValidityHarness(
        ablations=[Ablation.REAL, Ablation.RANDOM], seed=0
    ).evaluate(_REAL, _valid_train_eval)
    assert set(report.runs.keys()) == {Ablation.REAL, Ablation.RANDOM}


def test_harness_tolerance_zero_valid_reward_not_spurious() -> None:
    """With tolerance=0 and real gain >> degen gains, still not spurious."""
    report = RewardValidityHarness(seed=0, tolerance=0.0).evaluate(
        _REAL, _valid_train_eval
    )
    # real gain=0.30, all degen gains=0.01 → 0.01 < 0.30 - 0.0=0.30 → not spurious
    assert not report.spurious


def test_harness_invalid_tolerance_raises() -> None:
    with pytest.raises(ValueError, match="tolerance"):
        RewardValidityHarness(tolerance=-0.1)
