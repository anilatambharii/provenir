from __future__ import annotations

import json
from pathlib import Path

import pytest

from provenir.governance.bom import (
    CodeComponent,
    DataComponent,
    EvalComponent,
    ModelBOM,
    RewardValidityComponent,
)
from provenir.governance.passport import ModelPassport, PassportSigner
from provenir.governance.promotion_gate import (
    PromotionBlocked,
    PromotionCheck,
    PromotionGate,
    PromotionResult,
    load_and_gate,
)
from provenir.governance.retraction import RetractionComponent
from provenir.governance.scan import ScanComponent

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

KEY = b"gate-test-key"
TS = "2026-01-01T00:00:00Z"


def _code() -> CodeComponent:
    return CodeComponent(git_sha="abc123", dependencies_hash="dephash", framework="trl")


def _data(*, pii_scanned: bool = True, contamination_checked: bool = True) -> DataComponent:
    return DataComponent(
        name="train",
        content_hash="datahash",
        num_records=1000,
        license="apache-2.0",
        pii_scanned=pii_scanned,
        contamination_checked=contamination_checked,
    )


def _bom(
    model_id: str = "m1",
    *,
    pii_scanned: bool = True,
    contamination_checked: bool = True,
    scan: ScanComponent | None = None,
    retraction: RetractionComponent | None = None,
    reward_validity: RewardValidityComponent | None = None,
    contaminated_eval: bool = False,
) -> ModelBOM:
    return ModelBOM(
        model_id=model_id,
        base_model="base",
        run_id="run-1",
        data=[_data(pii_scanned=pii_scanned, contamination_checked=contamination_checked)],
        code=_code(),
        evals=[EvalComponent(benchmark="mmlu", score=0.75, contaminated=contaminated_eval)],
        hyperparameters={},
        created_at=TS,
        scan=scan,
        retraction=retraction,
        reward_validity=reward_validity,
    )


def _unsigned_passport(**kwargs: object) -> ModelPassport:
    """Build an unsigned passport, forwarding kwargs to _bom."""
    return ModelPassport(bom=_bom(**kwargs), attestation=None)  # type: ignore[arg-type]


def _signed_passport(**kwargs: object) -> ModelPassport:
    """Build a signed passport."""
    bom = _bom(**kwargs)  # type: ignore[arg-type]
    return PassportSigner(KEY).sign(bom, signed_at=TS)


def _clean_scan() -> ScanComponent:
    return ScanComponent(
        scanner_version="0.7.0",
        report_hash="reporthash",
        unsafe=False,
        finding_counts={"critical": 0, "high": 0, "medium": 0, "low": 0},
    )


def _unsafe_scan() -> ScanComponent:
    return ScanComponent(
        scanner_version="0.7.0",
        report_hash="reporthash",
        unsafe=True,
        finding_counts={"critical": 1, "high": 0, "medium": 0, "low": 0},
    )


def _clean_retraction() -> RetractionComponent:
    return RetractionComponent(
        checked_count=10,
        retracted_count=0,
        retraction_rate=0.0,
        risk_level="none",
        report_hash="rhash",
    )


def _dirty_retraction() -> RetractionComponent:
    return RetractionComponent(
        checked_count=10,
        retracted_count=2,
        retraction_rate=0.2,
        risk_level="high",
        report_hash="rhash",
    )


def _reward(validity: float = 0.9, spurious: bool = False) -> RewardValidityComponent:
    return RewardValidityComponent(
        reward_name="math",
        report_hash="rvhash",
        validity=validity,
        spurious=spurious,
    )


# ---------------------------------------------------------------------------
# PromotionCheck and PromotionResult unit tests
# ---------------------------------------------------------------------------


def test_promotion_check_to_dict() -> None:
    check = PromotionCheck("no_pii", False, "some reason")
    d = check.to_dict()
    assert d == {"name": "no_pii", "passed": False, "detail": "some reason"}


def test_promotion_result_summary_passed() -> None:
    result = PromotionResult("m1", "production", True, [], [])
    assert result.summary() == "PASSED: m1 -> production"


def test_promotion_result_summary_blocked() -> None:
    checks = [
        PromotionCheck("no_pii", False, "pii not scanned"),
        PromotionCheck("scan_clean", False, "unsafe findings"),
    ]
    result = PromotionResult("m1", "production", False, checks, ["no_pii", "scan_clean"])
    summary = result.summary()
    assert summary.startswith("BLOCKED: m1 -> production")
    assert "no_pii" in summary
    assert "pii not scanned" in summary
    assert "scan_clean" in summary


def test_promotion_result_to_json() -> None:
    result = PromotionResult("m1", "staging", True, [], [])
    data = json.loads(result.to_json())
    assert data["model_id"] == "m1"
    assert data["stage"] == "staging"
    assert data["passed"] is True


# ---------------------------------------------------------------------------
# Default gate (no requirements) — always passes
# ---------------------------------------------------------------------------


def test_default_gate_always_passes() -> None:
    gate = PromotionGate()
    passport = _unsigned_passport(pii_scanned=False)
    result = gate.evaluate(passport)
    assert result.passed is True
    assert result.checks == []
    assert result.failed_checks == []


def test_default_gate_no_exception_on_gate() -> None:
    gate = PromotionGate()
    passport = _unsigned_passport()
    gate.gate(passport)  # must not raise


# ---------------------------------------------------------------------------
# require_no_pii
# ---------------------------------------------------------------------------


def test_require_no_pii_passes_when_pii_scanned() -> None:
    gate = PromotionGate(require_no_pii=True)
    passport = _unsigned_passport(pii_scanned=True)
    result = gate.evaluate(passport)
    assert result.passed is True
    assert result.failed_checks == []


def test_require_no_pii_fails_when_not_scanned() -> None:
    gate = PromotionGate(require_no_pii=True)
    passport = _unsigned_passport(pii_scanned=False)
    result = gate.evaluate(passport)
    assert result.passed is False
    assert "no_pii" in result.failed_checks


# ---------------------------------------------------------------------------
# require_no_contamination
# ---------------------------------------------------------------------------


def test_require_no_contamination_fails_when_unchecked() -> None:
    gate = PromotionGate(require_no_contamination=True)
    passport = _unsigned_passport(contamination_checked=False)
    result = gate.evaluate(passport)
    assert result.passed is False
    assert "no_contamination" in result.failed_checks


def test_require_no_contamination_fails_when_eval_contaminated() -> None:
    gate = PromotionGate(require_no_contamination=True)
    passport = _unsigned_passport(contamination_checked=True, contaminated_eval=True)
    result = gate.evaluate(passport)
    assert result.passed is False
    assert "no_contamination" in result.failed_checks


def test_require_no_contamination_passes_when_clean() -> None:
    gate = PromotionGate(require_no_contamination=True)
    passport = _unsigned_passport(contamination_checked=True, contaminated_eval=False)
    result = gate.evaluate(passport)
    assert result.passed is True


# ---------------------------------------------------------------------------
# require_scan
# ---------------------------------------------------------------------------


def test_require_scan_fails_when_scan_is_none() -> None:
    gate = PromotionGate(require_scan=True)
    passport = _unsigned_passport(scan=None)
    result = gate.evaluate(passport)
    assert result.passed is False
    assert "scan_clean" in result.failed_checks


def test_require_scan_fails_when_scan_unsafe() -> None:
    gate = PromotionGate(require_scan=True)
    passport = _unsigned_passport(scan=_unsafe_scan())
    result = gate.evaluate(passport)
    assert result.passed is False
    assert "scan_clean" in result.failed_checks


def test_require_scan_passes_when_scan_clean() -> None:
    gate = PromotionGate(require_scan=True)
    passport = _unsigned_passport(scan=_clean_scan())
    result = gate.evaluate(passport)
    assert result.passed is True
    assert "scan_clean" not in result.failed_checks


# ---------------------------------------------------------------------------
# require_no_retraction
# ---------------------------------------------------------------------------


def test_require_no_retraction_fails_when_retracted() -> None:
    gate = PromotionGate(require_no_retraction=True)
    passport = _unsigned_passport(retraction=_dirty_retraction())
    result = gate.evaluate(passport)
    assert result.passed is False
    assert "no_retraction" in result.failed_checks


def test_require_no_retraction_passes_when_retraction_none() -> None:
    gate = PromotionGate(require_no_retraction=True)
    passport = _unsigned_passport(retraction=None)
    result = gate.evaluate(passport)
    assert result.passed is True


def test_require_no_retraction_passes_when_retraction_clean() -> None:
    gate = PromotionGate(require_no_retraction=True)
    passport = _unsigned_passport(retraction=_clean_retraction())
    result = gate.evaluate(passport)
    assert result.passed is True


# ---------------------------------------------------------------------------
# min_validity
# ---------------------------------------------------------------------------


def test_min_validity_fails_when_validity_too_low() -> None:
    gate = PromotionGate(min_validity=0.8)
    passport = _unsigned_passport(reward_validity=_reward(validity=0.5))
    result = gate.evaluate(passport)
    assert result.passed is False
    assert "reward_valid" in result.failed_checks


def test_min_validity_passes_when_validity_sufficient() -> None:
    gate = PromotionGate(min_validity=0.8)
    passport = _unsigned_passport(reward_validity=_reward(validity=0.9, spurious=False))
    result = gate.evaluate(passport)
    assert result.passed is True


def test_min_validity_fails_when_spurious_even_if_high() -> None:
    gate = PromotionGate(min_validity=0.8)
    passport = _unsigned_passport(reward_validity=_reward(validity=0.9, spurious=True))
    result = gate.evaluate(passport)
    assert result.passed is False
    assert "reward_valid" in result.failed_checks


def test_min_validity_fails_when_no_reward_validity() -> None:
    gate = PromotionGate(min_validity=0.8)
    passport = _unsigned_passport(reward_validity=None)
    result = gate.evaluate(passport)
    assert result.passed is False
    assert "reward_valid" in result.failed_checks


# ---------------------------------------------------------------------------
# require_signed
# ---------------------------------------------------------------------------


def test_require_signed_fails_when_unsigned() -> None:
    gate = PromotionGate(require_signed=True)
    passport = _unsigned_passport()
    result = gate.evaluate(passport)
    assert result.passed is False
    assert "signed" in result.failed_checks


def test_require_signed_passes_when_signed() -> None:
    gate = PromotionGate(require_signed=True)
    passport = _signed_passport()
    result = gate.evaluate(passport)
    assert result.passed is True


# ---------------------------------------------------------------------------
# Multiple requirements — all must pass
# ---------------------------------------------------------------------------


def test_multiple_requirements_all_pass() -> None:
    gate = PromotionGate(
        require_no_pii=True,
        require_scan=True,
        require_no_retraction=True,
        min_validity=0.8,
        require_signed=True,
    )
    passport = _signed_passport(
        pii_scanned=True,
        scan=_clean_scan(),
        retraction=_clean_retraction(),
        reward_validity=_reward(validity=0.9, spurious=False),
    )
    result = gate.evaluate(passport)
    assert result.passed is True
    assert result.failed_checks == []


def test_multiple_requirements_partial_failure() -> None:
    """PII passes, scan fails (None), reward passes — two checks total, one fails."""
    gate = PromotionGate(
        require_no_pii=True,
        require_scan=True,
        min_validity=0.8,
    )
    passport = _unsigned_passport(
        pii_scanned=True,
        scan=None,  # will fail scan check
        reward_validity=_reward(validity=0.9),
    )
    result = gate.evaluate(passport)
    assert result.passed is False
    assert "scan_clean" in result.failed_checks
    assert "no_pii" not in result.failed_checks
    assert "reward_valid" not in result.failed_checks


# ---------------------------------------------------------------------------
# PromotionBlocked carries result with correct summary
# ---------------------------------------------------------------------------


def test_promotion_blocked_carries_result() -> None:
    gate = PromotionGate(require_no_pii=True, require_signed=True)
    passport = _unsigned_passport(pii_scanned=False)
    with pytest.raises(PromotionBlocked) as exc_info:
        gate.gate(passport)
    exc = exc_info.value
    assert exc.result.passed is False
    summary = exc.result.summary()
    assert "BLOCKED" in summary
    assert "m1" in summary


def test_promotion_blocked_str_matches_summary() -> None:
    gate = PromotionGate(require_no_pii=True)
    passport = _unsigned_passport(pii_scanned=False)
    with pytest.raises(PromotionBlocked) as exc_info:
        gate.gate(passport)
    exc = exc_info.value
    assert str(exc) == exc.result.summary()


# ---------------------------------------------------------------------------
# load_and_gate — writes passport to tmp_path file and loads it
# ---------------------------------------------------------------------------


def test_load_and_gate_passes(tmp_path: Path) -> None:
    passport = _signed_passport(pii_scanned=True)
    passport_file = tmp_path / "passport.json"
    passport_file.write_text(passport.to_json(), encoding="utf-8")

    result = load_and_gate(passport_file, require_no_pii=True, require_signed=True)
    assert result.passed is True
    assert result.model_id == "m1"


def test_load_and_gate_raises_when_blocked(tmp_path: Path) -> None:
    passport = _unsigned_passport(pii_scanned=False)
    passport_file = tmp_path / "passport.json"
    passport_file.write_text(passport.to_json(), encoding="utf-8")

    with pytest.raises(PromotionBlocked) as exc_info:
        load_and_gate(passport_file, require_no_pii=True)
    assert exc_info.value.result.passed is False


def test_load_and_gate_stage_propagated(tmp_path: Path) -> None:
    passport = _unsigned_passport()
    passport_file = tmp_path / "passport.json"
    passport_file.write_text(passport.to_json(), encoding="utf-8")

    result = load_and_gate(passport_file, "staging")
    assert result.stage == "staging"
