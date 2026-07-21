"""Tests for the verifier-reliability harness."""

from __future__ import annotations

import json
from typing import Any

import pytest

from provenir.environments.base import VerificationResult
from provenir.environments.reliability import (
    DEFAULT_PERTURBATIONS,
    DerivedProbe,
    FailureMode,
    Probe,
    ProbeOutcome,
    PromotionBlocked,
    ReliabilityHarness,
    ReliabilityReport,
    gate_promotion,
)
from provenir.environments.verifiers import (
    ContainsVerifier,
    ExactAnswerVerifier,
    MathVerifier,
)

# ---------------------------------------------------------------------------
# Deliberately-broken stub verifiers (each fails exactly one mode)
# ---------------------------------------------------------------------------


class AlwaysPassVerifier:
    """Passes everything — cannot tell right from wrong (fails sensitivity)."""

    name = "always_pass"

    def verify(self, response: str, reference: Any) -> VerificationResult:
        return VerificationResult(True, 1.0, "always")


class LengthRewardVerifier:
    """Passes any response over 50 chars — trivially gameable by padding."""

    name = "length_reward"

    def verify(self, response: str, reference: Any) -> VerificationResult:
        passed = len(response) > 50
        return VerificationResult(passed, 1.0 if passed else 0.0, f"len={len(response)}")


class NonDeterministicVerifier:
    """Toggles its verdict every call (fails consistency)."""

    name = "flaky"

    def __init__(self) -> None:
        self._n = 0

    def verify(self, response: str, reference: Any) -> VerificationResult:
        self._n += 1
        passed = self._n % 2 == 0
        return VerificationResult(passed, 1.0 if passed else 0.0, f"call {self._n}")


class CrashingVerifier:
    """Raises on empty or very long input (fails boundary)."""

    name = "crashing"

    def verify(self, response: str, reference: Any) -> VerificationResult:
        if response == "" or len(response) > 1000:
            raise ValueError("cannot handle this input")
        return VerificationResult(True, 1.0, "ok")


class ShorterIsBetterVerifier:
    """Rewards shorter responses — violates monotonicity."""

    name = "shorter_better"

    def verify(self, response: str, reference: Any) -> VerificationResult:
        reward = 1.0 / (1 + len(response.split()))
        return VerificationResult(False, reward, "shorter=higher")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def harness() -> ReliabilityHarness:
    return ReliabilityHarness(seed=7)


CAT_PROBE = Probe("The answer is \\boxed{cat}", "cat", should_pass=True, label="cat")
NUM_PROBE = Probe("Final: \\boxed{42}", "42", should_pass=True, label="num42")
CONTAINS_PROBE = Probe("alpha beta gamma extra words here", None, should_pass=True, label="abc")


# ---------------------------------------------------------------------------
# Happy path: real verifiers score well
# ---------------------------------------------------------------------------


def test_exact_answer_no_hard_fail(harness: ReliabilityHarness) -> None:
    report = harness.evaluate(ExactAnswerVerifier(), [CAT_PROBE])
    assert not report.hard_fail
    assert report.overall == pytest.approx(1.0)


def test_exact_answer_sensitivity_perfect(harness: ReliabilityHarness) -> None:
    report = harness.evaluate(ExactAnswerVerifier(), [CAT_PROBE])
    assert report.scores[FailureMode.SENSITIVITY] == 1.0


def test_math_verifier_all_modes_pass(harness: ReliabilityHarness) -> None:
    report = harness.evaluate(MathVerifier(), [NUM_PROBE])
    assert not report.hard_fail
    assert report.scores[FailureMode.GAMEABILITY] == 1.0
    assert report.scores[FailureMode.SENSITIVITY] == 1.0


def test_math_verifier_boundary_no_crash(harness: ReliabilityHarness) -> None:
    report = harness.evaluate(MathVerifier(), [NUM_PROBE])
    assert report.scores[FailureMode.BOUNDARY] == 1.0


def test_contains_verifier_clean(harness: ReliabilityHarness) -> None:
    report = harness.evaluate(ContainsVerifier(["alpha", "beta", "gamma"]), [CONTAINS_PROBE])
    assert not report.hard_fail


# ---------------------------------------------------------------------------
# Each broken verifier fails its designated mode
# ---------------------------------------------------------------------------


def test_always_pass_hard_fails_sensitivity(harness: ReliabilityHarness) -> None:
    report = harness.evaluate(AlwaysPassVerifier(), [CAT_PROBE])
    assert report.hard_fail
    assert report.scores[FailureMode.SENSITIVITY] < 1.0


def test_length_reward_hard_fails_gameability(harness: ReliabilityHarness) -> None:
    report = harness.evaluate(LengthRewardVerifier(), [NUM_PROBE])
    assert report.hard_fail
    assert report.scores[FailureMode.GAMEABILITY] < 1.0


def test_non_deterministic_fails_consistency(harness: ReliabilityHarness) -> None:
    consistency = harness.evaluate(
        NonDeterministicVerifier(), [CAT_PROBE]
    ).scores[FailureMode.CONSISTENCY]
    assert consistency < 1.0


def test_non_deterministic_not_hard_fail_when_only_consistency() -> None:
    h = ReliabilityHarness(modes=[FailureMode.CONSISTENCY])
    report = h.evaluate(NonDeterministicVerifier(), [CAT_PROBE])
    assert report.scores[FailureMode.CONSISTENCY] < 1.0
    assert not report.hard_fail  # consistency is not a hard-fail mode


def test_crashing_fails_boundary_only() -> None:
    h = ReliabilityHarness(modes=[FailureMode.BOUNDARY])
    report = h.evaluate(CrashingVerifier(), [CAT_PROBE])
    assert report.scores[FailureMode.BOUNDARY] < 1.0
    assert not report.hard_fail  # boundary is not a hard-fail mode


def test_shorter_is_better_fails_monotonicity() -> None:
    h = ReliabilityHarness(modes=[FailureMode.MONOTONICITY])
    report = h.evaluate(ShorterIsBetterVerifier(), [CONTAINS_PROBE])
    assert report.scores[FailureMode.MONOTONICITY] < 1.0
    assert not report.hard_fail


def test_contains_monotonicity_holds(harness: ReliabilityHarness) -> None:
    report = harness.evaluate(ContainsVerifier(["alpha", "beta", "gamma"]), [CONTAINS_PROBE])
    assert report.scores[FailureMode.MONOTONICITY] == 1.0


# ---------------------------------------------------------------------------
# hard_fail semantics
# ---------------------------------------------------------------------------


def test_hard_fail_true_iff_sensitivity_or_gameability_violated(
    harness: ReliabilityHarness,
) -> None:
    good = harness.evaluate(ExactAnswerVerifier(), [CAT_PROBE])
    bad = harness.evaluate(AlwaysPassVerifier(), [CAT_PROBE])
    assert not good.hard_fail
    assert bad.hard_fail


def test_failures_returns_only_non_held(harness: ReliabilityHarness) -> None:
    report = harness.evaluate(AlwaysPassVerifier(), [CAT_PROBE])
    assert report.failures()
    assert all(not o.held for o in report.failures())


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_evaluate_is_deterministic() -> None:
    h1 = ReliabilityHarness(seed=3)
    h2 = ReliabilityHarness(seed=3)
    r1 = h1.evaluate(ExactAnswerVerifier(), [CAT_PROBE, NUM_PROBE])
    r2 = h2.evaluate(ExactAnswerVerifier(), [CAT_PROBE, NUM_PROBE])
    assert r1.to_dict() == r2.to_dict()


def test_boundary_never_propagates_crash() -> None:
    # CrashingVerifier raises internally; evaluate must not raise.
    report = ReliabilityHarness().evaluate(CrashingVerifier(), [CAT_PROBE])
    assert isinstance(report, ReliabilityReport)


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def test_to_dict_is_json_serializable(harness: ReliabilityHarness) -> None:
    report = harness.evaluate(MathVerifier(), [NUM_PROBE])
    blob = json.dumps(report.to_dict())
    restored = json.loads(blob)
    assert restored["verifier_name"] == "math"
    assert restored["hard_fail"] is False
    assert "scores" in restored


def test_report_summary_mentions_name_and_flag(harness: ReliabilityHarness) -> None:
    report = harness.evaluate(AlwaysPassVerifier(), [CAT_PROBE])
    text = report.summary()
    assert "always_pass" in text
    assert "HARD-FAIL" in text


def test_probe_outcome_to_dict() -> None:
    outcome = ProbeOutcome(FailureMode.SENSITIVITY, "must_fail", False, "x", "lbl")
    d = outcome.to_dict()
    assert d["mode"] == "sensitivity"
    assert d["held"] is False


# ---------------------------------------------------------------------------
# Mode selection / scope
# ---------------------------------------------------------------------------


def test_modes_subset_only_runs_selected() -> None:
    h = ReliabilityHarness(modes=[FailureMode.SENSITIVITY])
    report = h.evaluate(ExactAnswerVerifier(), [CAT_PROBE])
    assert set(report.scores) == {FailureMode.SENSITIVITY}


def test_boundary_runs_once_not_per_probe() -> None:
    h = ReliabilityHarness(modes=[FailureMode.BOUNDARY])
    report = h.evaluate(ExactAnswerVerifier(), [CAT_PROBE, NUM_PROBE, CONTAINS_PROBE])
    boundary_outcomes = [o for o in report.outcomes if o.mode is FailureMode.BOUNDARY]
    assert len(boundary_outcomes) == 5  # len(_BOUNDARY_INPUTS), independent of probe count


def test_should_pass_false_skips_sensitivity() -> None:
    probe = Probe("wrong", "right", should_pass=False)
    report = ReliabilityHarness().evaluate(ExactAnswerVerifier(), [probe])
    # No known-correct response -> sensitivity/gameability/monotonicity omitted.
    assert FailureMode.SENSITIVITY not in report.scores
    assert FailureMode.INVARIANCE in report.scores  # invariance still applies


def test_empty_probe_list_overall_one() -> None:
    report = ReliabilityHarness().evaluate(ExactAnswerVerifier(), [])
    assert report.n_probes == 0
    assert not report.hard_fail
    # only boundary produces outcomes (uses a None reference)
    assert report.scores.get(FailureMode.BOUNDARY) == 1.0


# ---------------------------------------------------------------------------
# Config validation & customization
# ---------------------------------------------------------------------------


def test_consistency_runs_must_be_at_least_two() -> None:
    with pytest.raises(ValueError, match="consistency_runs"):
        ReliabilityHarness(consistency_runs=1)


def test_custom_perturbation_override() -> None:
    def only_distractor(probe: Probe, rng: Any) -> list[DerivedProbe]:
        if not probe.should_pass:
            return []
        return [
            DerivedProbe("\\boxed{zzz}", probe.reference, FailureMode.SENSITIVITY, "must_fail")
        ]

    h = ReliabilityHarness(
        modes=[FailureMode.SENSITIVITY],
        perturbations={FailureMode.SENSITIVITY: [only_distractor]},
    )
    report = h.evaluate(ExactAnswerVerifier(), [CAT_PROBE])
    sensitivity_outcomes = [o for o in report.outcomes if o.mode is FailureMode.SENSITIVITY]
    assert len(sensitivity_outcomes) == 1
    assert report.scores[FailureMode.SENSITIVITY] == 1.0


def test_default_perturbations_cover_expected_modes() -> None:
    assert set(DEFAULT_PERTURBATIONS) == {
        FailureMode.INVARIANCE,
        FailureMode.SENSITIVITY,
        FailureMode.GAMEABILITY,
        FailureMode.MONOTONICITY,
    }


def test_custom_weights_change_overall() -> None:
    # Weighting sensitivity to zero should raise a bad verifier's overall.
    weights = {m: (0.0 if m is FailureMode.SENSITIVITY else 1.0) for m in FailureMode}
    default = ReliabilityHarness().evaluate(AlwaysPassVerifier(), [CAT_PROBE]).overall
    reweighted = ReliabilityHarness(weights=weights).evaluate(
        AlwaysPassVerifier(), [CAT_PROBE]
    ).overall
    assert reweighted >= default


def test_multiple_probes_aggregate() -> None:
    report = ReliabilityHarness().evaluate(ExactAnswerVerifier(), [CAT_PROBE, NUM_PROBE])
    assert report.n_probes == 2
    assert not report.hard_fail


# ---------------------------------------------------------------------------
# Promotion gate
# ---------------------------------------------------------------------------


def test_gate_allows_reliable_verifier_to_production(harness: ReliabilityHarness) -> None:
    report = harness.evaluate(ExactAnswerVerifier(), [CAT_PROBE])
    gate_promotion(report, "production")  # must not raise


def test_gate_blocks_hard_fail_to_production(harness: ReliabilityHarness) -> None:
    report = harness.evaluate(AlwaysPassVerifier(), [CAT_PROBE])
    with pytest.raises(PromotionBlocked, match="production"):
        gate_promotion(report, "production")


def test_gate_never_blocks_staging(harness: ReliabilityHarness) -> None:
    report = harness.evaluate(AlwaysPassVerifier(), [CAT_PROBE])
    gate_promotion(report, "staging")  # non-protected stage: must not raise


def test_gate_min_overall_enforced(harness: ReliabilityHarness) -> None:
    report = harness.evaluate(ExactAnswerVerifier(), [CAT_PROBE])
    with pytest.raises(PromotionBlocked, match="reliability"):
        gate_promotion(report, "production", min_overall=1.01)
