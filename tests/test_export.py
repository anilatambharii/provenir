"""Tests for CycloneDX 1.5 / SPDX 3.0 export module.

≥14 tests covering:
- CycloneDX format fields and structure
- SPDX 3.0 format fields and structure
- Signature / attestation presence / absence
- Vulnerabilities from ScanComponent
- Determinism
- JSON validity
"""

from __future__ import annotations

import json
from typing import Any

from provenir.governance.bom import (
    CodeComponent,
    DataComponent,
    EvalComponent,
    ModelBOM,
    RewardValidityComponent,
)
from provenir.governance.export import ExportFormat, export_passport, to_cyclonedx, to_spdx3
from provenir.governance.passport import ModelPassport, PassportSigner
from provenir.governance.scan import ScanComponent

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

_KEY = b"test"
_TS = "2026-01-01T00:00:00Z"


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _make_bom(
    *,
    scan: ScanComponent | None = None,
    reward_validity: RewardValidityComponent | None = None,
    extra_data: list[DataComponent] | None = None,
) -> ModelBOM:
    data = [
        DataComponent(
            name="train",
            content_hash="deadbeefdeadbeef",
            num_records=50_000,
            license="apache-2.0",
            pii_scanned=True,
            contamination_checked=True,
            source_category="web",
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
    if extra_data:
        data = data + extra_data
    return ModelBOM(
        model_id="test-model-v1",
        base_model="llama3-8b",
        run_id="run-abc123",
        data=data,
        code=CodeComponent(
            git_sha="abc123deadbeef",
            dependencies_hash="depshash99",
            framework="trl",
        ),
        evals=[
            EvalComponent(benchmark="mmlu", score=0.71),
            EvalComponent(benchmark="hellaswag", score=0.83),
        ],
        hyperparameters={"lr": 2e-5, "epochs": 3},
        created_at=_TS,
        scan=scan,
        reward_validity=reward_validity,
    )


def _signed_passport(
    *,
    scan: ScanComponent | None = None,
    reward_validity: RewardValidityComponent | None = None,
) -> ModelPassport:
    bom = _make_bom(scan=scan, reward_validity=reward_validity)
    return PassportSigner(_KEY).sign(bom, signed_at=_TS)


def _unsigned_passport(
    *,
    scan: ScanComponent | None = None,
) -> ModelPassport:
    bom = _make_bom(scan=scan)
    return ModelPassport(bom=bom, attestation=None)


def _scan_component(*, unsafe: bool = False) -> ScanComponent:
    return ScanComponent(
        scanner_version="0.7.0",
        report_hash="aabbccdd11223344",
        unsafe=unsafe,
        finding_counts={"critical": 1 if unsafe else 0, "high": 0, "medium": 0, "low": 0},
    )


# ---------------------------------------------------------------------------
# CycloneDX tests
# ---------------------------------------------------------------------------


def test_cyclonedx_bom_format() -> None:
    """CycloneDX output must have bomFormat == 'CycloneDX'."""
    cdx = to_cyclonedx(_unsigned_passport())
    assert cdx["bomFormat"] == "CycloneDX"


def test_cyclonedx_spec_version() -> None:
    """CycloneDX output must have specVersion == '1.5'."""
    cdx = to_cyclonedx(_unsigned_passport())
    assert cdx["specVersion"] == "1.5"


def test_cyclonedx_metadata_component_name() -> None:
    """CycloneDX metadata.component.name must equal bom.model_id."""
    passport = _unsigned_passport()
    cdx = to_cyclonedx(passport)
    assert cdx["metadata"]["component"]["name"] == passport.bom.model_id


def test_cyclonedx_components_count_matches_data() -> None:
    """CycloneDX components list must have one entry per DataComponent."""
    passport = _unsigned_passport()
    cdx = to_cyclonedx(passport)
    assert len(cdx["components"]) == len(passport.bom.data)


def test_cyclonedx_component_license() -> None:
    """CycloneDX component license id must match DataComponent.license."""
    passport = _unsigned_passport()
    cdx = to_cyclonedx(passport)
    expected_licenses = [dc.license for dc in passport.bom.data]
    actual_licenses = [
        comp["licenses"][0]["license"]["id"] for comp in cdx["components"]
    ]
    assert actual_licenses == expected_licenses


def test_cyclonedx_no_vulnerabilities_when_no_scan() -> None:
    """CycloneDX must not have a 'vulnerabilities' key when bom.scan is None."""
    cdx = to_cyclonedx(_unsigned_passport(scan=None))
    assert "vulnerabilities" not in cdx


def test_cyclonedx_vulnerabilities_present_when_scan_present() -> None:
    """CycloneDX must include one vulnerability entry when bom.scan is present."""
    sc = _scan_component(unsafe=True)
    cdx = to_cyclonedx(_unsigned_passport(scan=sc))
    assert "vulnerabilities" in cdx
    vulns = cdx["vulnerabilities"]
    assert len(vulns) == 1
    vuln = vulns[0]
    assert vuln["id"].startswith("PROVENIR-SCAN-")
    assert vuln["ratings"][0]["severity"] == "critical"
    assert vuln["analysis"]["state"] == "in_triage"


def test_cyclonedx_vulnerability_not_unsafe() -> None:
    """CycloneDX vulnerability entry for a clean scan has severity 'none'."""
    sc = _scan_component(unsafe=False)
    cdx = to_cyclonedx(_unsigned_passport(scan=sc))
    vuln = cdx["vulnerabilities"][0]
    assert vuln["ratings"][0]["severity"] == "none"
    assert vuln["analysis"]["state"] == "false_positive"


def test_cyclonedx_signature_present_when_attestation_present() -> None:
    """CycloneDX must include a 'signature' block when passport is signed."""
    passport = _signed_passport()
    assert passport.attestation is not None
    cdx = to_cyclonedx(passport)
    assert "signature" in cdx
    sig = cdx["signature"]
    assert sig["algorithm"] == passport.attestation.algorithm
    assert sig["value"] == passport.attestation.signature
    assert sig["keyId"] == passport.attestation.key_id


def test_cyclonedx_no_signature_when_no_attestation() -> None:
    """CycloneDX must NOT include a 'signature' key when passport is unsigned."""
    cdx = to_cyclonedx(_unsigned_passport())
    assert "signature" not in cdx


def test_cyclonedx_reward_validity_property() -> None:
    """CycloneDX metadata properties include reward_validity when present."""
    rv = RewardValidityComponent(
        reward_name="math",
        report_hash="rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr1",
        validity=0.86,
        spurious=False,
    )
    passport = _signed_passport(reward_validity=rv)
    cdx = to_cyclonedx(passport)
    prop_names = [p["name"] for p in cdx["metadata"]["component"]["properties"]]
    assert "provenir:reward_validity" in prop_names
    rv_prop = next(p for p in cdx["metadata"]["component"]["properties"]
                   if p["name"] == "provenir:reward_validity")
    assert "0.860" in rv_prop["value"]
    assert "False" in rv_prop["value"] or "false" in rv_prop["value"].lower()


# ---------------------------------------------------------------------------
# SPDX 3.0 tests
# ---------------------------------------------------------------------------


def test_spdx_version() -> None:
    """SPDX output must have spdxVersion == 'SPDX-3.0'."""
    spdx = to_spdx3(_unsigned_passport())
    assert spdx["spdxVersion"] == "SPDX-3.0"


def test_spdx_packages_count() -> None:
    """SPDX packages count must equal 1 (model) + len(data components)."""
    passport = _unsigned_passport()
    spdx = to_spdx3(passport)
    expected = 1 + len(passport.bom.data)
    assert len(spdx["packages"]) == expected


def test_spdx_relationships_count() -> None:
    """SPDX relationships count must equal len(data components) (HAS_DATA_INPUT)."""
    passport = _unsigned_passport()
    spdx = to_spdx3(passport)
    assert len(spdx["relationships"]) == len(passport.bom.data)
    for rel in spdx["relationships"]:
        assert rel["relationshipType"] == "HAS_DATA_INPUT"
        assert rel["spdxElementId"] == "SPDXRef-Model"


def test_spdx_model_package_fields() -> None:
    """SPDX primary model package has correct name, version, and purpose."""
    passport = _unsigned_passport()
    spdx = to_spdx3(passport)
    model_pkg = next(p for p in spdx["packages"] if p["SPDXID"] == "SPDXRef-Model")
    assert model_pkg["name"] == passport.bom.model_id
    assert model_pkg["versionInfo"] == passport.bom.run_id
    assert model_pkg["primaryPackagePurpose"] == "MACHINE-LEARNING-MODEL"


def test_spdx_data_package_license() -> None:
    """SPDX data packages carry licenseDeclared from DataComponent.license."""
    passport = _unsigned_passport()
    spdx = to_spdx3(passport)
    data_pkgs = [p for p in spdx["packages"] if p["SPDXID"] != "SPDXRef-Model"]
    expected = [dc.license for dc in passport.bom.data]
    actual = [p["licenseDeclared"] for p in data_pkgs]
    assert actual == expected


def test_spdx_annotations_include_evals() -> None:
    """SPDX annotations include one entry per eval."""
    passport = _unsigned_passport()
    spdx = to_spdx3(passport)
    eval_annotations = [
        a for a in spdx["annotations"]
        if a["comment"].startswith("Eval:")
    ]
    assert len(eval_annotations) == len(passport.bom.evals)


def test_spdx_annotations_include_scan_when_present() -> None:
    """SPDX annotations include a scan entry when bom.scan is present."""
    sc = _scan_component(unsafe=False)
    passport = _unsigned_passport(scan=sc)
    spdx = to_spdx3(passport)
    scan_annotations = [
        a for a in spdx["annotations"]
        if "Supply-chain scan" in a["comment"]
    ]
    assert len(scan_annotations) == 1


# ---------------------------------------------------------------------------
# Determinism and JSON validity
# ---------------------------------------------------------------------------


def test_both_formats_deterministic() -> None:
    """Both export formats produce identical output on repeated calls."""
    passport = _signed_passport(scan=_scan_component(unsafe=False))
    cdx1 = export_passport(passport, ExportFormat.CYCLONEDX_JSON)
    cdx2 = export_passport(passport, ExportFormat.CYCLONEDX_JSON)
    assert cdx1 == cdx2

    spdx1 = export_passport(passport, ExportFormat.SPDX3_JSON)
    spdx2 = export_passport(passport, ExportFormat.SPDX3_JSON)
    assert spdx1 == spdx2


def test_export_passport_returns_valid_json_cyclonedx() -> None:
    """export_passport CycloneDX output is parseable as JSON."""
    passport = _signed_passport()
    out = export_passport(passport, ExportFormat.CYCLONEDX_JSON)
    parsed: Any = json.loads(out)
    assert isinstance(parsed, dict)
    assert parsed["bomFormat"] == "CycloneDX"


def test_export_passport_returns_valid_json_spdx() -> None:
    """export_passport SPDX 3.0 output is parseable as JSON."""
    passport = _signed_passport()
    out = export_passport(passport, ExportFormat.SPDX3_JSON)
    parsed: Any = json.loads(out)
    assert isinstance(parsed, dict)
    assert parsed["spdxVersion"] == "SPDX-3.0"


def test_export_format_enum_values() -> None:
    """ExportFormat enum members have the expected string values."""
    assert ExportFormat.CYCLONEDX_JSON == "cyclonedx-json"
    assert ExportFormat.SPDX3_JSON == "spdx3-json"


def test_cyclonedx_serial_number_format() -> None:
    """CycloneDX serialNumber must start with 'urn:uuid:'."""
    cdx = to_cyclonedx(_unsigned_passport())
    assert cdx["serialNumber"].startswith("urn:uuid:")


def test_spdx_document_namespace_format() -> None:
    """SPDX documentNamespace must start with 'https://provenir.ai/sbom/'."""
    spdx = to_spdx3(_unsigned_passport())
    assert spdx["documentNamespace"].startswith("https://provenir.ai/sbom/")


def test_cyclonedx_component_properties_source_category() -> None:
    """CycloneDX data component properties include source_category."""
    passport = _unsigned_passport()
    cdx = to_cyclonedx(passport)
    for i, comp in enumerate(cdx["components"]):
        prop_names = [p["name"] for p in comp["properties"]]
        assert "provenir:source_category" in prop_names
        sc_prop = next(p for p in comp["properties"]
                       if p["name"] == "provenir:source_category")
        assert sc_prop["value"] == passport.bom.data[i].source_category
