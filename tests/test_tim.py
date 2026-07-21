"""Tests for the Training-Inference Mismatch (TIM) Detector.

Covers:
1.  Identical distributions → KL = 0.0, mismatch_detected=False
2.  Heavily diverged distributions → KL > threshold, mismatch_detected=True
3.  mismatch_rate = fraction of probes above threshold
4.  content_hash() is deterministic (same inputs → same hex, length 64)
5.  content_hash() changes when mean_kl changes
6.  to_dict() has correct keys
7.  summary() is a non-empty string
8.  gate_tim raises TIMBlocked when mismatch on "production"
9.  gate_tim does NOT raise on "staging" even with mismatch
10. gate_tim raises when max_mean_kl exceeded on protected stage
11. gate_tim does NOT raise when max_mean_kl not exceeded
12. Single-token log-prob lists work correctly
13. Zero log-prob clamping (no math.log(0) crash)
14. TIMResult.to_dict() round-trips
15. Mixed mismatch/non-mismatch probes → correct aggregate stats
"""

from __future__ import annotations

import math

import pytest

from provenir.environments.tim import (
    TIMBlocked,
    TIMDetector,
    TIMResult,
    gate_tim,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _identical_probe(prompt: str) -> tuple[list[float], list[float]]:
    """Both sides have the same log-probs → KL = 0."""
    lp = [-1.0, -2.0, -0.5]
    return lp, lp


def _diverged_probe(prompt: str) -> tuple[list[float], list[float]]:
    """Massively diverged: train concentrates on token 0, inf concentrates on token 2."""
    train_lp = [0.0, -100.0, -100.0]
    inf_lp = [-100.0, -100.0, 0.0]
    return train_lp, inf_lp


def _uniform_probe(prompt: str) -> tuple[list[float], list[float]]:
    """Both sides uniform but with slight offset to force non-zero KL."""
    train_lp = [-1.0, -1.0, -1.0]
    inf_lp = [-1.1, -0.9, -1.0]
    return train_lp, inf_lp


# ---------------------------------------------------------------------------
# 1. Identical distributions → KL ≈ 0, no mismatch
# ---------------------------------------------------------------------------


def test_identical_distributions_kl_zero() -> None:
    detector = TIMDetector(threshold=0.1)
    report = detector.detect(_identical_probe, ["p1", "p2"])
    assert math.isclose(report.mean_kl, 0.0, abs_tol=1e-10)
    assert report.mismatch_detected is False
    assert all(math.isclose(r.kl_divergence, 0.0, abs_tol=1e-10) for r in report.results)


# ---------------------------------------------------------------------------
# 2. Heavily diverged distributions → KL > threshold, mismatch_detected=True
# ---------------------------------------------------------------------------


def test_diverged_distributions_mismatch_detected() -> None:
    detector = TIMDetector(threshold=0.1)
    report = detector.detect(_diverged_probe, ["p1"])
    assert report.mean_kl > 0.1
    assert report.mismatch_detected is True
    assert report.results[0].mismatch is True


# ---------------------------------------------------------------------------
# 3. mismatch_rate = fraction of probes above threshold
# ---------------------------------------------------------------------------


def test_mismatch_rate_fraction() -> None:
    """2 probes: one identical (KL=0), one diverged (KL>>0.1)."""
    prompts = ["identical", "diverged"]

    def mixed_probe(prompt: str) -> tuple[list[float], list[float]]:
        if prompt == "identical":
            return _identical_probe(prompt)
        return _diverged_probe(prompt)

    detector = TIMDetector(threshold=0.1)
    report = detector.detect(mixed_probe, prompts)
    assert math.isclose(report.mismatch_rate, 0.5, abs_tol=1e-9)


# ---------------------------------------------------------------------------
# 4. content_hash() is deterministic
# ---------------------------------------------------------------------------


def test_content_hash_deterministic() -> None:
    detector = TIMDetector(threshold=0.1)
    r1 = detector.detect(_identical_probe, ["p1", "p2"])
    r2 = detector.detect(_identical_probe, ["p1", "p2"])
    assert r1.content_hash() == r2.content_hash()
    assert len(r1.content_hash()) == 64  # SHA-256 hex


# ---------------------------------------------------------------------------
# 5. content_hash() changes when mean_kl changes
# ---------------------------------------------------------------------------


def test_content_hash_changes_with_mean_kl() -> None:
    detector = TIMDetector(threshold=0.1)
    report_zero = detector.detect(_identical_probe, ["p1"])
    report_div = detector.detect(_diverged_probe, ["p1"])
    assert report_zero.content_hash() != report_div.content_hash()


# ---------------------------------------------------------------------------
# 6. to_dict() has correct keys
# ---------------------------------------------------------------------------


def test_to_dict_correct_keys() -> None:
    detector = TIMDetector(threshold=0.1)
    report = detector.detect(_identical_probe, ["p1"])
    d = report.to_dict()
    expected_keys = {
        "probe_count", "mean_kl", "max_kl", "mismatch_rate", "mismatch_detected", "results"
    }
    assert set(d.keys()) == expected_keys


# ---------------------------------------------------------------------------
# 7. summary() is a non-empty string
# ---------------------------------------------------------------------------


def test_summary_non_empty() -> None:
    detector = TIMDetector(threshold=0.1)
    report = detector.detect(_identical_probe, ["p1"])
    s = report.summary()
    assert isinstance(s, str)
    assert len(s) > 0


def test_summary_contains_flag_ok() -> None:
    detector = TIMDetector(threshold=0.1)
    report = detector.detect(_identical_probe, ["p1"])
    assert "ok" in report.summary()


def test_summary_contains_flag_mismatch() -> None:
    detector = TIMDetector(threshold=0.1)
    report = detector.detect(_diverged_probe, ["p1"])
    assert "MISMATCH" in report.summary()


# ---------------------------------------------------------------------------
# 8. gate_tim raises TIMBlocked when mismatch on "production"
# ---------------------------------------------------------------------------


def test_gate_tim_blocks_production_on_mismatch() -> None:
    detector = TIMDetector(threshold=0.1)
    report = detector.detect(_diverged_probe, ["p1"])
    assert report.mismatch_detected is True
    with pytest.raises(TIMBlocked):
        gate_tim(report, "production")


def test_gate_tim_blocks_prod_alias() -> None:
    detector = TIMDetector(threshold=0.1)
    report = detector.detect(_diverged_probe, ["p1"])
    with pytest.raises(TIMBlocked):
        gate_tim(report, "prod")


# ---------------------------------------------------------------------------
# 9. gate_tim does NOT raise on "staging" even with mismatch
# ---------------------------------------------------------------------------


def test_gate_tim_allows_staging_despite_mismatch() -> None:
    detector = TIMDetector(threshold=0.1)
    report = detector.detect(_diverged_probe, ["p1"])
    assert report.mismatch_detected is True
    # Should not raise
    gate_tim(report, "staging")


# ---------------------------------------------------------------------------
# 10. gate_tim raises when max_mean_kl exceeded on protected stage
# ---------------------------------------------------------------------------


def test_gate_tim_raises_when_max_mean_kl_exceeded() -> None:
    detector = TIMDetector(threshold=1000.0)  # won't trigger mismatch_detected
    report = detector.detect(_uniform_probe, ["p1", "p2", "p3"])
    assert report.mismatch_detected is False
    # Set max_mean_kl below actual mean_kl
    with pytest.raises(TIMBlocked):
        gate_tim(report, "production", max_mean_kl=0.0)


# ---------------------------------------------------------------------------
# 11. gate_tim does NOT raise when max_mean_kl not exceeded
# ---------------------------------------------------------------------------


def test_gate_tim_no_raise_when_max_mean_kl_ok() -> None:
    detector = TIMDetector(threshold=0.1)
    report = detector.detect(_identical_probe, ["p1"])
    assert report.mismatch_detected is False
    # max_mean_kl well above zero
    gate_tim(report, "production", max_mean_kl=1.0)


# ---------------------------------------------------------------------------
# 12. Single-token log-prob lists work correctly
# ---------------------------------------------------------------------------


def test_single_token_log_probs() -> None:
    """A single token: both sides have the same probability → KL = 0."""

    def single_token_probe(prompt: str) -> tuple[list[float], list[float]]:
        return [-1.0], [-1.0]

    detector = TIMDetector(threshold=0.1)
    report = detector.detect(single_token_probe, ["p1"])
    assert math.isclose(report.mean_kl, 0.0, abs_tol=1e-10)
    assert report.mismatch_detected is False


# ---------------------------------------------------------------------------
# 13. Zero log-prob clamping (no math.log(0) crash)
# ---------------------------------------------------------------------------


def test_zero_log_prob_clamping_no_crash() -> None:
    """Passing -inf log-probs must not crash (clamps to min_prob)."""

    def zero_prob_probe(prompt: str) -> tuple[list[float], list[float]]:
        train_lp = [0.0, float("-inf"), float("-inf")]
        inf_lp = [float("-inf"), 0.0, float("-inf")]
        return train_lp, inf_lp

    detector = TIMDetector(threshold=0.1)
    report = detector.detect(zero_prob_probe, ["p1"])
    # Just must not raise; KL will be large
    assert report.mean_kl >= 0.0


# ---------------------------------------------------------------------------
# 14. TIMResult.to_dict() round-trips
# ---------------------------------------------------------------------------


def test_tim_result_to_dict_round_trips() -> None:
    result = TIMResult(prompt="hello", kl_divergence=0.042, mismatch=False)
    d = result.to_dict()
    assert d["prompt"] == "hello"
    assert math.isclose(d["kl_divergence"], 0.042)
    assert d["mismatch"] is False
    assert set(d.keys()) == {"prompt", "kl_divergence", "mismatch"}


# ---------------------------------------------------------------------------
# 15. Mixed mismatch/non-mismatch probes → correct aggregate stats
# ---------------------------------------------------------------------------


def test_mixed_probes_aggregate_stats() -> None:
    """3 probes: 2 identical (KL≈0) + 1 diverged (KL>>0.1)."""
    prompts = ["a", "b", "c"]
    diverged_prompt = "c"

    def mixed_probe(prompt: str) -> tuple[list[float], list[float]]:
        if prompt == diverged_prompt:
            return _diverged_probe(prompt)
        return _identical_probe(prompt)

    detector = TIMDetector(threshold=0.1)
    report = detector.detect(mixed_probe, prompts)

    assert report.probe_count == 3
    # mismatch_rate: 1 out of 3 probes mismatched
    assert math.isclose(report.mismatch_rate, 1 / 3, abs_tol=1e-9)
    # max_kl must be from the diverged probe
    assert report.max_kl > 1.0
    # mean_kl is average including the near-zero probes
    kl_sum = sum(r.kl_divergence for r in report.results)
    assert math.isclose(report.mean_kl, kl_sum / 3, rel_tol=1e-9)
    # results list has correct length
    assert len(report.results) == 3
