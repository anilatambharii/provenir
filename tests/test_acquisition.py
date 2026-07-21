"""Tests for the Acquisition Package Export module.

≥18 tests covering:
- Directory and file creation
- BOM manifest content
- IP risk flags and unknown-license detection
- Security audit with/without ScanComponent
- Regulatory exposure coverage percentages
- Model reliability with/without RewardValidityComponent
- Reproducibility fields
- Retraction risk with/without RetractionComponent
- Lineage (parent hash present / absent)
- SIGNED-SUMMARY structure
- AcquisitionPackage.to_dict() and summary()
- Warnings population
"""

from __future__ import annotations

import json
from pathlib import Path

from provenir.governance.acquisition import (
    SECTION_BOM_MANIFEST,
    SECTION_IP_RISK,
    SECTION_LINEAGE,
    SECTION_MODEL_RELIABILITY,
    SECTION_REGULATORY_EXPOSURE,
    SECTION_REPRODUCIBILITY,
    SECTION_RETRACTION_RISK,
    SECTION_SECURITY_AUDIT,
    SECTION_SIGNED_SUMMARY,
    generate_acquisition_package,
)
from provenir.governance.bom import (
    CodeComponent,
    DataComponent,
    EvalComponent,
    ModelBOM,
    RewardValidityComponent,
)
from provenir.governance.passport import ModelPassport, PassportSigner
from provenir.governance.retraction import RetractionComponent
from provenir.governance.scan import ScanComponent

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_KEY = b"test"
_TS = "2026-01-01T00:00:00Z"
_PARENT_HASH = "deadbeef" * 8  # 64-char hex string


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _make_bom(
    *,
    scan: ScanComponent | None = None,
    reward_validity: RewardValidityComponent | None = None,
    retraction: RetractionComponent | None = None,
    parent_passport_hash: str | None = None,
    unknown_license: bool = False,
) -> ModelBOM:
    license_val = "unknown" if unknown_license else "apache-2.0"
    data = [
        DataComponent(
            name="train",
            content_hash="deadbeefdeadbeef",
            num_records=50_000,
            license=license_val,
            pii_scanned=True,
            contamination_checked=True,
            source_category="web",
            crawl_domains={"example.com": 30_000, "other.org": 20_000},
            optout_respected=True,
            retraction_dois=["10.1234/test.001"],
        ),
        DataComponent(
            name="eval",
            content_hash="cafecafecafecafe",
            num_records=5_000,
            license="cc-by-4.0",
            pii_scanned=True,
            contamination_checked=True,
            source_category="curated",
        ),
    ]
    return ModelBOM(
        model_id="acq-test-model-v1",
        base_model="llama3-8b",
        run_id="run-acq-abc123",
        data=data,
        code=CodeComponent(
            git_sha="abc123deadbeef99",
            dependencies_hash="depshash99aabb",
            framework="trl",
        ),
        evals=[
            EvalComponent(benchmark="mmlu", score=0.71),
            EvalComponent(benchmark="hellaswag", score=0.83),
        ],
        hyperparameters={"lr": 1e-4, "epochs": 3, "batch_size": 8},
        created_at=_TS,
        scan=scan,
        reward_validity=reward_validity,
        retraction=retraction,
        parent_passport_hash=parent_passport_hash,
    )


def _make_scan() -> ScanComponent:
    return ScanComponent(
        scanner_version="0.9.0",
        report_hash="scanhash123abc",
        unsafe=False,
        finding_counts={"critical": 0, "high": 0, "medium": 1, "low": 2},
    )


def _make_reward_validity() -> RewardValidityComponent:
    return RewardValidityComponent(
        reward_name="math_reward",
        report_hash="rewardhash999xyz",
        validity=0.87,
        spurious=False,
    )


def _make_retraction() -> RetractionComponent:
    return RetractionComponent(
        checked_count=42,
        retracted_count=1,
        retraction_rate=0.0238,
        risk_level="high",
        report_hash="retractionhash456",
    )


def _full_passport(parent_hash: str | None = None) -> ModelPassport:
    bom = _make_bom(
        scan=_make_scan(),
        reward_validity=_make_reward_validity(),
        retraction=_make_retraction(),
        parent_passport_hash=parent_hash,
    )
    return PassportSigner(_KEY).sign(bom, signed_at=_TS)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_generate_creates_output_dir(tmp_path: Path) -> None:
    """generate_acquisition_package creates the output directory."""
    out = tmp_path / "new_pkg_dir"
    assert not out.exists()
    generate_acquisition_package(_full_passport(), out, include_regulation=False)
    assert out.is_dir()


def test_bom_manifest_exists(tmp_path: Path) -> None:
    """bom-manifest.json is written into the output directory."""
    pkg = generate_acquisition_package(_full_passport(), tmp_path, include_regulation=False)
    bom_file = Path(pkg.output_dir) / pkg.sections[SECTION_BOM_MANIFEST]
    assert bom_file.exists()


def test_bom_manifest_valid_json(tmp_path: Path) -> None:
    """bom-manifest.json contains valid JSON."""
    pkg = generate_acquisition_package(_full_passport(), tmp_path, include_regulation=False)
    bom_file = Path(pkg.output_dir) / pkg.sections[SECTION_BOM_MANIFEST]
    data = json.loads(bom_file.read_text(encoding="utf-8"))
    assert "bom" in data


def test_bom_manifest_matches_passport_to_json(tmp_path: Path) -> None:
    """bom-manifest.json content matches passport.to_json() exactly."""
    passport = _full_passport()
    pkg = generate_acquisition_package(passport, tmp_path, include_regulation=False)
    bom_file = Path(pkg.output_dir) / pkg.sections[SECTION_BOM_MANIFEST]
    assert bom_file.read_text(encoding="utf-8") == passport.to_json()


def test_ip_risk_exists_and_contains_model_id(tmp_path: Path) -> None:
    """ip-risk.md exists and mentions the model_id."""
    passport = _full_passport()
    pkg = generate_acquisition_package(passport, tmp_path, include_regulation=False)
    ip_file = Path(pkg.output_dir) / pkg.sections[SECTION_IP_RISK]
    assert ip_file.exists()
    content = ip_file.read_text(encoding="utf-8")
    assert "acq-test-model-v1" in content


def test_ip_risk_flags_unknown_license(tmp_path: Path) -> None:
    """ip-risk.md flags an unknown license."""
    bom = _make_bom(
        scan=_make_scan(),
        reward_validity=_make_reward_validity(),
        retraction=_make_retraction(),
        unknown_license=True,
    )
    passport = PassportSigner(_KEY).sign(bom, signed_at=_TS)
    pkg = generate_acquisition_package(passport, tmp_path, include_regulation=False)
    ip_file = Path(pkg.output_dir) / pkg.sections[SECTION_IP_RISK]
    content = ip_file.read_text(encoding="utf-8")
    assert "Unknown license" in content or "unknown" in content.lower()
    assert "verify IP clearance" in content


def test_security_audit_exists_with_scan(tmp_path: Path) -> None:
    """security-audit.md mentions scan details when ScanComponent is present."""
    pkg = generate_acquisition_package(_full_passport(), tmp_path, include_regulation=False)
    audit_file = Path(pkg.output_dir) / pkg.sections[SECTION_SECURITY_AUDIT]
    assert audit_file.exists()
    content = audit_file.read_text(encoding="utf-8")
    assert "0.9.0" in content  # scanner_version
    assert "scanhash123abc" in content  # report_hash


def test_security_audit_no_scan_warning(tmp_path: Path) -> None:
    """security-audit.md shows the 'No supply-chain scan' message when scan is absent."""
    bom = _make_bom()  # no scan
    passport = PassportSigner(_KEY).sign(bom, signed_at=_TS)
    pkg = generate_acquisition_package(passport, tmp_path, include_regulation=False)
    audit_file = Path(pkg.output_dir) / pkg.sections[SECTION_SECURITY_AUDIT]
    content = audit_file.read_text(encoding="utf-8")
    assert "No supply-chain scan" in content


def test_regulatory_exposure_exists_with_coverage(tmp_path: Path) -> None:
    """regulatory-exposure.md exists and contains coverage percentages."""
    pkg = generate_acquisition_package(_full_passport(), tmp_path, include_regulation=True)
    reg_file = Path(pkg.output_dir) / pkg.sections[SECTION_REGULATORY_EXPOSURE]
    assert reg_file.exists()
    content = reg_file.read_text(encoding="utf-8")
    # Coverage percentages from RegulationGenerator should appear
    assert "%" in content


def test_model_reliability_reward_name(tmp_path: Path) -> None:
    """model-reliability.md mentions the reward name when RewardValidityComponent is present."""
    pkg = generate_acquisition_package(_full_passport(), tmp_path, include_regulation=False)
    rel_file = Path(pkg.output_dir) / pkg.sections[SECTION_MODEL_RELIABILITY]
    content = rel_file.read_text(encoding="utf-8")
    assert "math_reward" in content


def test_model_reliability_no_reward_warning(tmp_path: Path) -> None:
    """model-reliability.md warns when RewardValidityComponent is absent."""
    bom = _make_bom(scan=_make_scan())  # no reward_validity
    passport = PassportSigner(_KEY).sign(bom, signed_at=_TS)
    pkg = generate_acquisition_package(passport, tmp_path, include_regulation=False)
    rel_file = Path(pkg.output_dir) / pkg.sections[SECTION_MODEL_RELIABILITY]
    content = rel_file.read_text(encoding="utf-8")
    assert "No reward validity" in content


def test_reproducibility_contains_run_id_and_git_sha(tmp_path: Path) -> None:
    """reproducibility.md contains the BOM run_id and code git_sha."""
    pkg = generate_acquisition_package(_full_passport(), tmp_path, include_regulation=False)
    repro_file = Path(pkg.output_dir) / pkg.sections[SECTION_REPRODUCIBILITY]
    content = repro_file.read_text(encoding="utf-8")
    assert "run-acq-abc123" in content
    assert "abc123deadbeef99" in content


def test_retraction_risk_with_retraction_data(tmp_path: Path) -> None:
    """retraction-risk.md shows retraction data when RetractionComponent is present."""
    pkg = generate_acquisition_package(_full_passport(), tmp_path, include_regulation=False)
    ret_file = Path(pkg.output_dir) / pkg.sections[SECTION_RETRACTION_RISK]
    content = ret_file.read_text(encoding="utf-8")
    assert "42" in content  # checked_count
    assert "retractionhash456" in content


def test_lineage_with_parent_hash(tmp_path: Path) -> None:
    """lineage.md mentions the parent passport hash when set."""
    passport = _full_passport(parent_hash=_PARENT_HASH)
    pkg = generate_acquisition_package(passport, tmp_path, include_regulation=False)
    lin_file = Path(pkg.output_dir) / pkg.sections[SECTION_LINEAGE]
    content = lin_file.read_text(encoding="utf-8")
    assert _PARENT_HASH in content


def test_lineage_base_model_no_parent(tmp_path: Path) -> None:
    """lineage.md says 'Base model' when parent_passport_hash is None."""
    passport = _full_passport(parent_hash=None)
    pkg = generate_acquisition_package(passport, tmp_path, include_regulation=False)
    lin_file = Path(pkg.output_dir) / pkg.sections[SECTION_LINEAGE]
    content = lin_file.read_text(encoding="utf-8")
    assert "Base model" in content


def test_signed_summary_valid_json_with_package_hash(tmp_path: Path) -> None:
    """SIGNED-SUMMARY.json exists, is valid JSON, and contains package_hash."""
    pkg = generate_acquisition_package(_full_passport(), tmp_path, include_regulation=False)
    summary_file = Path(pkg.output_dir) / pkg.sections[SECTION_SIGNED_SUMMARY]
    assert summary_file.exists()
    data = json.loads(summary_file.read_text(encoding="utf-8"))
    assert "package_hash" in data
    assert len(data["package_hash"]) == 64  # SHA-256 hex


def test_acquisition_package_to_dict_keys(tmp_path: Path) -> None:
    """AcquisitionPackage.to_dict() returns the expected top-level keys."""
    pkg = generate_acquisition_package(_full_passport(), tmp_path, include_regulation=False)
    d = pkg.to_dict()
    assert "model_id" in d
    assert "output_dir" in d
    assert "sections" in d
    assert "package_hash" in d
    assert "warnings" in d
    assert d["model_id"] == "acq-test-model-v1"


def test_acquisition_package_summary_is_non_empty(tmp_path: Path) -> None:
    """AcquisitionPackage.summary() returns a non-empty string."""
    pkg = generate_acquisition_package(_full_passport(), tmp_path, include_regulation=False)
    s = pkg.summary()
    assert isinstance(s, str)
    assert len(s) > 0
    assert "acq-test-model-v1" in s


def test_warnings_populated_when_scan_absent(tmp_path: Path) -> None:
    """AcquisitionPackage.warnings is non-empty when ScanComponent is absent."""
    bom = _make_bom()  # no scan, no reward_validity, no retraction
    passport = PassportSigner(_KEY).sign(bom, signed_at=_TS)
    pkg = generate_acquisition_package(passport, tmp_path, include_regulation=False)
    warning_texts = " ".join(pkg.warnings)
    assert "scan" in warning_texts.lower()


def test_package_hash_is_64_char_hex(tmp_path: Path) -> None:
    """AcquisitionPackage.package_hash is a 64-character hex string (SHA-256)."""
    pkg = generate_acquisition_package(_full_passport(), tmp_path, include_regulation=False)
    assert len(pkg.package_hash) == 64
    assert all(c in "0123456789abcdef" for c in pkg.package_hash)


def test_all_expected_sections_present(tmp_path: Path) -> None:
    """All nine section keys are present in AcquisitionPackage.sections."""
    pkg = generate_acquisition_package(_full_passport(), tmp_path, include_regulation=False)
    expected = {
        SECTION_BOM_MANIFEST,
        SECTION_IP_RISK,
        SECTION_SECURITY_AUDIT,
        SECTION_REGULATORY_EXPOSURE,
        SECTION_MODEL_RELIABILITY,
        SECTION_REPRODUCIBILITY,
        SECTION_RETRACTION_RISK,
        SECTION_LINEAGE,
        SECTION_SIGNED_SUMMARY,
    }
    assert expected == set(pkg.sections.keys())


def test_unsigned_passport_adds_warning(tmp_path: Path) -> None:
    """An unsigned passport causes an 'unsigned passport' warning."""
    bom = _make_bom(scan=_make_scan(), reward_validity=_make_reward_validity())
    passport = ModelPassport(bom=bom, attestation=None)
    pkg = generate_acquisition_package(passport, tmp_path, include_regulation=False)
    warning_texts = " ".join(pkg.warnings)
    assert "unsigned" in warning_texts.lower()


def test_include_regulation_false_skips_coverage(tmp_path: Path) -> None:
    """When include_regulation=False, regulatory-exposure.md says 'skipped'."""
    pkg = generate_acquisition_package(_full_passport(), tmp_path, include_regulation=False)
    reg_file = Path(pkg.output_dir) / pkg.sections[SECTION_REGULATORY_EXPOSURE]
    content = reg_file.read_text(encoding="utf-8")
    assert "skipped" in content.lower()
