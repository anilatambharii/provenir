"""Tests for EU AI Act Art. 53 / Annex IV evidence generator.

≥18 tests covering:
- Golden Art. 53 summary (domain rollup, SME threshold, no-crawl-data)
- Golden Annex IV skeleton (§2c/2d SATISFIED, §4/7-9 MISSING)
- Coverage score / missing() correctness
- Additive BOM fields round-trip (to_dict / from_dict)
- Backward-compat (old passports without new fields)
- Determinism (golden stable across runs)
- --fail-under CLI semantics
- to_dict() round-trips
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from provenir.governance.bom import (
    CodeComponent,
    DataComponent,
    EvalComponent,
    ModelBOM,
    RewardValidityComponent,
)
from provenir.governance.passport import ModelPassport, PassportSigner
from provenir.governance.regulation import (
    _NO_CRAWL_DATA,
    Coverage,
    EvidenceReport,
    FieldCoverage,
    RegulationGenerator,
)
from provenir.governance.scan import ScanComponent

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

_KEY = b"test-signing-key"
_TS = "2026-01-01T00:00:00Z"


def _minimal_bom(*, with_evals: bool = True) -> ModelBOM:
    """Minimal valid BOM — no crawl data, no source categories."""
    return ModelBOM(
        model_id="test-model",
        base_model="llama3-8b",
        run_id="run-abc",
        data=[
            DataComponent(
                name="train",
                content_hash="datahash1",
                num_records=1000,
                license="apache-2.0",
                pii_scanned=True,
                contamination_checked=True,
            )
        ],
        code=CodeComponent(git_sha="deadbeef", dependencies_hash="deps123", framework="trl"),
        evals=[EvalComponent(benchmark="mmlu", score=0.72)] if with_evals else [],
        hyperparameters={"lr": 0.001, "epochs": 3},
        created_at=_TS,
    )


def _rich_bom() -> ModelBOM:
    """Fully-populated BOM with crawl_domains, source_category, optout_respected."""
    crawl_a: dict[str, int] = {
        "wikipedia.org": 50_000,
        "arxiv.org": 30_000,
        "github.com": 15_000,
        "stackoverflow.com": 10_000,
        "reddit.com": 8_000,
        "news.ycombinator.com": 5_000,
        "medium.com": 3_000,
        "quora.com": 2_000,
        "nytimes.com": 1_500,
        "bbc.co.uk": 1_000,
        "cnn.com": 800,
        "theguardian.com": 600,
    }
    return ModelBOM(
        model_id="rich-model-v1",
        base_model="mistral-7b",
        run_id="run-rich",
        data=[
            DataComponent(
                name="web-crawl",
                content_hash="webhash",
                num_records=sum(crawl_a.values()),
                license="cc-by-4.0",
                pii_scanned=True,
                contamination_checked=True,
                source_category="scraped",
                crawl_domains=crawl_a,
                optout_respected=True,
            ),
            DataComponent(
                name="licensed-books",
                content_hash="bookshash",
                num_records=10_000,
                license="licensed",
                pii_scanned=True,
                contamination_checked=True,
                source_category="licensed",
                crawl_domains={},
                optout_respected=None,
            ),
        ],
        code=CodeComponent(
            git_sha="cafebabe", dependencies_hash="depshash", framework="trl"
        ),
        evals=[
            EvalComponent(benchmark="mmlu", score=0.82),
            EvalComponent(benchmark="hellaswag", score=0.78),
        ],
        hyperparameters={"lr": 0.0002, "epochs": 2, "modalities": "text"},
        created_at=_TS,
    )


def _signed_passport(bom: ModelBOM) -> ModelPassport:
    return PassportSigner(_KEY).sign(bom, signed_at=_TS)


def _unsigned_passport(bom: ModelBOM) -> ModelPassport:
    return ModelPassport(bom=bom, attestation=None)


# ---------------------------------------------------------------------------
# 1. FieldCoverage to_dict round-trip
# ---------------------------------------------------------------------------


def test_field_coverage_to_dict() -> None:
    fc = FieldCoverage("annex_iv.2c", Coverage.SATISFIED, "Data BOM present")
    d = fc.to_dict()
    assert d["field_id"] == "annex_iv.2c"
    assert d["coverage"] == "satisfied"
    assert d["note"] == "Data BOM present"


# ---------------------------------------------------------------------------
# 2. EvidenceReport basic API
# ---------------------------------------------------------------------------


def test_evidence_report_coverage_score_all_missing() -> None:
    fields = [
        FieldCoverage("f1", Coverage.MISSING, "n1"),
        FieldCoverage("f2", Coverage.MISSING, "n2"),
    ]
    report = EvidenceReport("test", "# md", {}, fields)
    assert report.coverage_score() == 0.0


def test_evidence_report_coverage_score_all_satisfied() -> None:
    fields = [
        FieldCoverage("f1", Coverage.SATISFIED),
        FieldCoverage("f2", Coverage.SATISFIED),
    ]
    report = EvidenceReport("test", "# md", {}, fields)
    assert report.coverage_score() == 1.0


def test_evidence_report_coverage_score_mixed() -> None:
    fields = [
        FieldCoverage("f1", Coverage.SATISFIED),
        FieldCoverage("f2", Coverage.MISSING),
        FieldCoverage("f3", Coverage.PARTIAL),
        FieldCoverage("f4", Coverage.SATISFIED),
    ]
    report = EvidenceReport("test", "# md", {}, fields)
    assert report.coverage_score() == pytest.approx(0.5)


def test_evidence_report_missing_returns_only_missing() -> None:
    fields = [
        FieldCoverage("f1", Coverage.SATISFIED),
        FieldCoverage("f2", Coverage.MISSING, "fix it"),
        FieldCoverage("f3", Coverage.PARTIAL),
    ]
    report = EvidenceReport("test", "# md", {}, fields)
    missing = report.missing()
    assert len(missing) == 1
    assert missing[0].field_id == "f2"


def test_evidence_report_empty_fields_score_zero() -> None:
    report = EvidenceReport("test", "", {}, [])
    assert report.coverage_score() == 0.0


def test_evidence_report_to_dict_round_trip() -> None:
    fields = [FieldCoverage("f1", Coverage.SATISFIED, "ok")]
    data = {"key": "value"}
    report = EvidenceReport("art53_training_data_summary", "# md", data, fields)
    d = report.to_dict()
    assert d["artifact"] == "art53_training_data_summary"
    assert d["coverage_score"] == 1.0
    assert len(d["fields"]) == 1
    assert d["data"]["key"] == "value"


# ---------------------------------------------------------------------------
# 3. Additive BOM fields round-trip
# ---------------------------------------------------------------------------


def test_data_component_new_fields_default() -> None:
    """Old-style construction (no new args) still works with correct defaults."""
    dc = DataComponent(name="d", content_hash="h", num_records=5)
    assert dc.source_category == "unknown"
    assert dc.crawl_domains == {}
    assert dc.optout_respected is None


def test_data_component_to_dict_emits_new_fields() -> None:
    dc = DataComponent(
        name="d",
        content_hash="h",
        num_records=5,
        source_category="scraped",
        crawl_domains={"example.com": 100},
        optout_respected=True,
    )
    d = dc.to_dict()
    assert d["source_category"] == "scraped"
    assert d["crawl_domains"] == {"example.com": 100}
    assert d["optout_respected"] is True


def test_passport_round_trip_with_new_fields() -> None:
    """New fields survive passport to_dict → from_dict."""
    bom = ModelBOM(
        model_id="m1",
        base_model="b1",
        run_id="r1",
        data=[
            DataComponent(
                name="d",
                content_hash="h",
                num_records=10,
                source_category="public",
                crawl_domains={"foo.com": 50},
                optout_respected=True,
            )
        ],
        code=CodeComponent(git_sha="s", dependencies_hash="dh", framework="trl"),
        evals=[EvalComponent(benchmark="b", score=0.5)],
    )
    passport = _signed_passport(bom)
    d = passport.to_dict()
    restored = ModelPassport.from_dict(d)
    dc = restored.bom.data[0]
    assert dc.source_category == "public"
    assert dc.crawl_domains == {"foo.com": 50}
    assert dc.optout_respected is True


def test_passport_backward_compat_old_fields_missing() -> None:
    """Old passport JSON (no new fields) loads with correct defaults."""
    old_json: dict[str, Any] = {
        "bom": {
            "model_id": "old-model",
            "base_model": "gpt2",
            "run_id": "r-old",
            "data": [
                {
                    "name": "train",
                    "content_hash": "oldhash",
                    "num_records": 500,
                    "license": "mit",
                    "pii_scanned": False,
                    "contamination_checked": False,
                    # NO source_category, crawl_domains, optout_respected
                }
            ],
            "code": {
                "git_sha": "abc",
                "dependencies_hash": "def",
                "framework": "pytorch",
            },
            "evals": [],
            "hyperparameters": {},
            "created_at": "",
            "scan": None,
            "reward_validity": None,
        },
        "attestation": None,
    }
    passport = ModelPassport.from_dict(old_json)
    dc = passport.bom.data[0]
    assert dc.source_category == "unknown"
    assert dc.crawl_domains == {}
    assert dc.optout_respected is None


# ---------------------------------------------------------------------------
# 4. Art. 53 — golden tests
# ---------------------------------------------------------------------------


def test_art53_artifact_identifier() -> None:
    passport = _signed_passport(_minimal_bom())
    report = RegulationGenerator().art53_training_data_summary(passport)
    assert report.artifact == "art53_training_data_summary"


def test_art53_no_crawl_data_uses_sentinel() -> None:
    """Without crawl_domains, domain section uses the sentinel string — no fabrication."""
    passport = _signed_passport(_minimal_bom())
    report = RegulationGenerator().art53_training_data_summary(passport)
    assert _NO_CRAWL_DATA in report.markdown
    # structured data also flags unavailable
    assert report.data["domain_rollup"]["available"] is False


def test_art53_with_crawl_data_top10_pct() -> None:
    """With crawl_domains, top-10% domain rollup appears in markdown and data."""
    passport = _signed_passport(_rich_bom())
    report = RegulationGenerator().art53_training_data_summary(passport)
    assert _NO_CRAWL_DATA not in report.markdown
    rollup = report.data["domain_rollup"]
    assert rollup["available"] is True
    assert rollup["threshold_pct"] == 10
    # wikipedia.org is the largest domain — must appear in the top-10%
    assert "wikipedia.org" in rollup["domains"]


def test_art53_sme_uses_5pct_threshold() -> None:
    """With sme=True the domain threshold is 5% and shows accordingly."""
    passport = _signed_passport(_rich_bom())
    report = RegulationGenerator(sme=True).art53_training_data_summary(passport)
    rollup = report.data["domain_rollup"]
    assert rollup["threshold_pct"] == 5
    assert "5%" in report.markdown


def test_art53_top_domains_deterministic() -> None:
    """Running the generator twice produces identical markdown."""
    passport = _signed_passport(_rich_bom())
    gen = RegulationGenerator()
    r1 = gen.art53_training_data_summary(passport)
    r2 = gen.art53_training_data_summary(passport)
    assert r1.markdown == r2.markdown


def test_art53_domain_rollup_sorted_descending() -> None:
    """Domains in the rollup are ordered by volume descending."""
    passport = _signed_passport(_rich_bom())
    report = RegulationGenerator().art53_training_data_summary(passport)
    rollup = report.data["domain_rollup"]
    counts = list(rollup["domains"].values())
    assert counts == sorted(counts, reverse=True)


def test_art53_old_passport_generates_missing_markers() -> None:
    """Old passport (no crawl data / no source_category) generates MISSING markers."""
    old_json: dict[str, Any] = {
        "bom": {
            "model_id": "old",
            "base_model": "gpt2",
            "run_id": "r-old",
            "data": [
                {
                    "name": "train",
                    "content_hash": "h",
                    "num_records": 100,
                    "license": "mit",
                    "pii_scanned": False,
                    "contamination_checked": False,
                }
            ],
            "code": {"git_sha": "a", "dependencies_hash": "b", "framework": "pt"},
            "evals": [],
            "hyperparameters": {},
            "created_at": "",
            "scan": None,
            "reward_validity": None,
        },
        "attestation": None,
    }
    passport = ModelPassport.from_dict(old_json)
    report = RegulationGenerator().art53_training_data_summary(passport)
    # top_domains field must be MISSING (no crawl data)
    domain_fc = next(f for f in report.fields if f.field_id == "art53.top_domains")
    assert domain_fc.coverage == Coverage.MISSING
    # sentinel string present in markdown
    assert _NO_CRAWL_DATA in report.markdown


def test_art53_coverage_score_increases_with_richer_passport() -> None:
    """A richer BOM raises coverage score vs. minimal BOM."""
    minimal_passport = _signed_passport(_minimal_bom())
    rich_passport = _signed_passport(_rich_bom())
    gen = RegulationGenerator()
    s_minimal = gen.art53_training_data_summary(minimal_passport).coverage_score()
    s_rich = gen.art53_training_data_summary(rich_passport).coverage_score()
    assert s_rich > s_minimal


# ---------------------------------------------------------------------------
# 5. Annex IV — golden tests
# ---------------------------------------------------------------------------


def test_annex_iv_artifact_identifier() -> None:
    passport = _signed_passport(_minimal_bom())
    report = RegulationGenerator().annex_iv_technical_file(passport)
    assert report.artifact == "annex_iv_technical_file"


def test_annex_iv_section2c_satisfied_when_data_present() -> None:
    """§2(c) data provenance is SATISFIED when BOM has data components."""
    passport = _signed_passport(_minimal_bom())
    report = RegulationGenerator().annex_iv_technical_file(passport)
    s2c = next(f for f in report.fields if f.field_id == "annex_iv.2c.data_provenance")
    assert s2c.coverage == Coverage.SATISFIED


def test_annex_iv_section2d_satisfied_when_evals_present() -> None:
    """§2(d) validation/testing is SATISFIED when BOM has eval components."""
    passport = _signed_passport(_minimal_bom())
    report = RegulationGenerator().annex_iv_technical_file(passport)
    s2d = next(f for f in report.fields if f.field_id == "annex_iv.2d.validation")
    assert s2d.coverage == Coverage.SATISFIED


def test_annex_iv_section4_risk_management_missing() -> None:
    """§4 risk management is always MISSING (organisational, not in BOM)."""
    passport = _signed_passport(_minimal_bom())
    report = RegulationGenerator().annex_iv_technical_file(passport)
    s4 = next(f for f in report.fields if f.field_id == "annex_iv.4.risk_management")
    assert s4.coverage == Coverage.MISSING
    assert len(s4.note) > 10  # actionable note present


def test_annex_iv_sections_7_8_9_missing_with_notes() -> None:
    """§7, §8, §9 are always MISSING and have actionable notes."""
    passport = _signed_passport(_minimal_bom())
    report = RegulationGenerator().annex_iv_technical_file(passport)
    for fid in (
        "annex_iv.7.post_market",
        "annex_iv.8.human_oversight",
        "annex_iv.9.conformity",
    ):
        fc = next(f for f in report.fields if f.field_id == fid)
        assert fc.coverage == Coverage.MISSING, f"{fid} should be MISSING"
        assert len(fc.note) > 10, f"{fid} should have an actionable note"


def test_annex_iv_missing_returns_expected_fields() -> None:
    """missing() includes §4, §7, §8, §9 for a minimal passport."""
    passport = _signed_passport(_minimal_bom())
    report = RegulationGenerator().annex_iv_technical_file(passport)
    missing_ids = {f.field_id for f in report.missing()}
    assert "annex_iv.4.risk_management" in missing_ids
    assert "annex_iv.7.post_market" in missing_ids
    assert "annex_iv.8.human_oversight" in missing_ids
    assert "annex_iv.9.conformity" in missing_ids


def test_annex_iv_section2d_missing_when_no_evals() -> None:
    """§2(d) is MISSING when the BOM has no evaluations."""
    passport = _signed_passport(_minimal_bom(with_evals=False))
    report = RegulationGenerator().annex_iv_technical_file(passport)
    s2d = next(f for f in report.fields if f.field_id == "annex_iv.2d.validation")
    assert s2d.coverage == Coverage.MISSING


def test_annex_iv_deterministic() -> None:
    """Running the generator twice produces identical markdown."""
    passport = _signed_passport(_rich_bom())
    gen = RegulationGenerator()
    r1 = gen.annex_iv_technical_file(passport)
    r2 = gen.annex_iv_technical_file(passport)
    assert r1.markdown == r2.markdown


def test_annex_iv_to_dict_has_sections() -> None:
    passport = _signed_passport(_minimal_bom())
    report = RegulationGenerator().annex_iv_technical_file(passport)
    d = report.to_dict()
    assert "sections" in d["data"]
    assert "2c_data_provenance" in d["data"]["sections"]


def test_annex_iv_coverage_score_in_range() -> None:
    passport = _signed_passport(_minimal_bom())
    report = RegulationGenerator().annex_iv_technical_file(passport)
    score = report.coverage_score()
    assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# 6. Coverage.MISSING provider field — always missing (Art. 53)
# ---------------------------------------------------------------------------


def test_art53_provider_always_missing() -> None:
    """Provider / organisation is not stored in the BOM → always MISSING."""
    passport = _signed_passport(_rich_bom())
    report = RegulationGenerator().art53_training_data_summary(passport)
    provider_fc = next(f for f in report.fields if f.field_id == "art53.provider")
    assert provider_fc.coverage == Coverage.MISSING


# ---------------------------------------------------------------------------
# 7. fail-under CLI semantics (via sys.exit)
# ---------------------------------------------------------------------------


def test_cli_regulation_art53_invokes(tmp_path: Path) -> None:
    """CLI round-trip: write passport JSON, call regulation art53, check output."""
    from provenir.cli.main import main

    bom = _minimal_bom()
    passport = _signed_passport(bom)
    pf = tmp_path / "passport.json"
    pf.write_text(json.dumps(passport.to_dict()), encoding="utf-8")

    out_md = tmp_path / "art53.md"
    import sys

    sys.argv = ["provenir", "regulation", "art53", "--passport", str(pf), "--out", str(out_md)]
    main()  # should not raise
    assert out_md.exists()
    content = out_md.read_text(encoding="utf-8")
    assert "Art. 53" in content


def test_cli_regulation_fail_under_passes(tmp_path: Path) -> None:
    """--fail-under below coverage score → exit 0 (no SystemExit)."""
    from provenir.cli.main import main

    bom = _minimal_bom()
    passport = _signed_passport(bom)
    pf = tmp_path / "passport.json"
    pf.write_text(json.dumps(passport.to_dict()), encoding="utf-8")

    import sys

    sys.argv = [
        "provenir",
        "regulation",
        "art53",
        "--passport",
        str(pf),
        "--fail-under",
        "0.0",  # 0.0 always passes
    ]
    main()  # should not raise SystemExit


def test_cli_regulation_fail_under_exits(tmp_path: Path) -> None:
    """--fail-under above coverage score → exit 1."""
    from provenir.cli.main import main

    bom = _minimal_bom()
    passport = _signed_passport(bom)
    pf = tmp_path / "passport.json"
    pf.write_text(json.dumps(passport.to_dict()), encoding="utf-8")

    import sys

    sys.argv = [
        "provenir",
        "regulation",
        "art53",
        "--passport",
        str(pf),
        "--fail-under",
        "1.0",  # 1.0 will always fail unless 100% coverage
    ]
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 1


def test_cli_regulation_annex_iv_invokes(tmp_path: Path) -> None:
    """CLI round-trip: regulation annex-iv writes a markdown file."""
    from provenir.cli.main import main

    bom = _minimal_bom()
    passport = _signed_passport(bom)
    pf = tmp_path / "passport.json"
    pf.write_text(json.dumps(passport.to_dict()), encoding="utf-8")

    out_md = tmp_path / "annex_iv.md"
    import sys

    sys.argv = [
        "provenir",
        "regulation",
        "annex-iv",
        "--passport",
        str(pf),
        "--out",
        str(out_md),
    ]
    main()
    assert out_md.exists()
    content = out_md.read_text(encoding="utf-8")
    assert "Annex IV" in content


# ---------------------------------------------------------------------------
# 8. FDA PCCP — Predetermined Change Control Plan
# ---------------------------------------------------------------------------


def _pccp_full_bom() -> ModelBOM:
    """Full BOM with samd_function, scan, reward_validity, pii_scanned=True."""
    return ModelBOM(
        model_id="samd-model-v1",
        base_model="resnet50",
        run_id="pccp-run-001",
        data=[
            DataComponent(
                name="clinical-train",
                content_hash="clinicalhash",
                num_records=5000,
                license="licensed",
                pii_scanned=True,
                contamination_checked=True,
            ),
        ],
        code=CodeComponent(git_sha="abc1234", dependencies_hash="depshash", framework="pytorch"),
        evals=[EvalComponent(benchmark="auroc", score=0.91)],
        hyperparameters={
            "lr": 0.0001,
            "epochs": 10,
            "samd_function": "diagnostic_support",
        },
        created_at=_TS,
        scan=ScanComponent("0.7.0", "scanhash1", False, {"critical": 0, "high": 0}),
        reward_validity=RewardValidityComponent(
            "clinical_reward", "rephash", 0.88, spurious=False
        ),
    )


class TestFDAPCCP:
    """Tests for RegulationGenerator.fda_pccp_summary."""

    def test_full_bom_device_description_satisfied(self) -> None:
        """Full BOM → pccp.device_description is SATISFIED."""
        passport = _signed_passport(_pccp_full_bom())
        report = RegulationGenerator().fda_pccp_summary(passport)
        fc = next(f for f in report.fields if f.field_id == "pccp.device_description")
        assert fc.coverage == Coverage.SATISFIED

    def test_full_bom_training_data_documentation_satisfied(self) -> None:
        """Full BOM with all pii_scanned → pccp.training_data_documentation is SATISFIED."""
        passport = _signed_passport(_pccp_full_bom())
        report = RegulationGenerator().fda_pccp_summary(passport)
        fc = next(
            f for f in report.fields if f.field_id == "pccp.training_data_documentation"
        )
        assert fc.coverage == Coverage.SATISFIED

    def test_full_bom_performance_testing_satisfied(self) -> None:
        """Full BOM with evals → pccp.performance_testing is SATISFIED."""
        passport = _signed_passport(_pccp_full_bom())
        report = RegulationGenerator().fda_pccp_summary(passport)
        fc = next(f for f in report.fields if f.field_id == "pccp.performance_testing")
        assert fc.coverage == Coverage.SATISFIED

    def test_empty_evals_performance_testing_missing(self) -> None:
        """Empty evals → pccp.performance_testing is MISSING."""
        bom = ModelBOM(
            model_id="samd-v2",
            base_model="resnet50",
            run_id="pccp-run-002",
            data=[DataComponent("train", "h", 100, pii_scanned=True)],
            code=CodeComponent(git_sha="sha1", dependencies_hash="dh", framework="pytorch"),
            evals=[],
            hyperparameters={"samd_function": "triage"},
            created_at=_TS,
        )
        passport = _signed_passport(bom)
        report = RegulationGenerator().fda_pccp_summary(passport)
        fc = next(f for f in report.fields if f.field_id == "pccp.performance_testing")
        assert fc.coverage == Coverage.MISSING

    def test_no_scan_impact_assessment_missing(self) -> None:
        """bom.scan absent → pccp.impact_assessment is MISSING."""
        bom = ModelBOM(
            model_id="samd-v3",
            base_model="resnet50",
            run_id="pccp-run-003",
            data=[DataComponent("train", "h", 100, pii_scanned=True)],
            code=CodeComponent(git_sha="sha1", dependencies_hash="dh", framework="pytorch"),
            evals=[EvalComponent("auroc", 0.85)],
            hyperparameters={"samd_function": "triage"},
            created_at=_TS,
            scan=None,
        )
        passport = _signed_passport(bom)
        report = RegulationGenerator().fda_pccp_summary(passport)
        fc = next(f for f in report.fields if f.field_id == "pccp.impact_assessment")
        assert fc.coverage == Coverage.MISSING

    def test_samd_function_in_hyperparameters_satisfied(self) -> None:
        """samd_function in hyperparameters → pccp.samd_function is SATISFIED."""
        passport = _signed_passport(_pccp_full_bom())
        report = RegulationGenerator().fda_pccp_summary(passport)
        fc = next(f for f in report.fields if f.field_id == "pccp.samd_function")
        assert fc.coverage == Coverage.SATISFIED

    def test_coverage_score_in_range(self) -> None:
        """coverage_score() is between 0 and 1 inclusive."""
        passport = _signed_passport(_pccp_full_bom())
        report = RegulationGenerator().fda_pccp_summary(passport)
        assert 0.0 <= report.coverage_score() <= 1.0

    def test_missing_returns_only_missing_fields(self) -> None:
        """missing() returns only fields with Coverage.MISSING."""
        passport = _signed_passport(_pccp_full_bom())
        report = RegulationGenerator().fda_pccp_summary(passport)
        for fc in report.missing():
            assert fc.coverage == Coverage.MISSING

    def test_deterministic_output(self) -> None:
        """Calling fda_pccp_summary twice produces identical reports."""
        passport = _signed_passport(_pccp_full_bom())
        gen = RegulationGenerator()
        r1 = gen.fda_pccp_summary(passport)
        r2 = gen.fda_pccp_summary(passport)
        assert r1.markdown == r2.markdown
        assert r1.data == r2.data

    def test_artifact_identifier(self) -> None:
        """artifact field equals 'fda_pccp'."""
        passport = _signed_passport(_pccp_full_bom())
        report = RegulationGenerator().fda_pccp_summary(passport)
        assert report.artifact == "fda_pccp"

    def test_missing_samd_function_is_missing(self) -> None:
        """No samd_function in hyperparameters → pccp.samd_function is MISSING."""
        bom = ModelBOM(
            model_id="samd-v4",
            base_model="resnet50",
            run_id="pccp-run-004",
            data=[DataComponent("train", "h", 100, pii_scanned=True)],
            code=CodeComponent(git_sha="sha1", dependencies_hash="dh", framework="pytorch"),
            evals=[EvalComponent("auroc", 0.85)],
            hyperparameters={"lr": 0.001},  # no samd_function
            created_at=_TS,
        )
        passport = _signed_passport(bom)
        report = RegulationGenerator().fda_pccp_summary(passport)
        fc = next(f for f in report.fields if f.field_id == "pccp.samd_function")
        assert fc.coverage == Coverage.MISSING
        assert "samd_function" in fc.note


# ---------------------------------------------------------------------------
# 9. NIST AI RMF 1.0
# ---------------------------------------------------------------------------


def _rmf_full_bom() -> ModelBOM:
    """Full BOM suitable for NIST AI RMF testing."""
    return ModelBOM(
        model_id="rmf-model-v1",
        base_model="llama3-8b",
        run_id="rmf-run-001",
        data=[
            DataComponent(
                name="train",
                content_hash="trainhash",
                num_records=10000,
                license="apache-2.0",
                pii_scanned=True,
                contamination_checked=True,
            ),
        ],
        code=CodeComponent(git_sha="deadbeef", dependencies_hash="deps1", framework="trl"),
        evals=[EvalComponent(benchmark="mmlu", score=0.75)],
        hyperparameters={"lr": 0.001, "fairness_eval": "demographic_parity_0.95"},
        created_at=_TS,
        scan=ScanComponent("0.7.0", "scanhash2", False, {"critical": 0}),
        reward_validity=RewardValidityComponent(
            "rlhf_reward", "rephash2", 0.82, spurious=False
        ),
    )


class TestNISTAIRMF:
    """Tests for RegulationGenerator.nist_ai_rmf_summary."""

    def test_signed_passport_govern1_satisfied(self) -> None:
        """Signed passport → GOVERN.1 is SATISFIED."""
        passport = _signed_passport(_rmf_full_bom())
        report = RegulationGenerator().nist_ai_rmf_summary(passport)
        fc = next(f for f in report.fields if f.field_id == "GOVERN.1")
        assert fc.coverage == Coverage.SATISFIED

    def test_unsigned_passport_govern1_partial(self) -> None:
        """Unsigned passport → GOVERN.1 is PARTIAL."""
        passport = _unsigned_passport(_rmf_full_bom())
        report = RegulationGenerator().nist_ai_rmf_summary(passport)
        fc = next(f for f in report.fields if f.field_id == "GOVERN.1")
        assert fc.coverage == Coverage.PARTIAL

    def test_scan_present_map1_satisfied(self) -> None:
        """Scan present in BOM → MAP.1 is SATISFIED."""
        passport = _signed_passport(_rmf_full_bom())
        report = RegulationGenerator().nist_ai_rmf_summary(passport)
        fc = next(f for f in report.fields if f.field_id == "MAP.1")
        assert fc.coverage == Coverage.SATISFIED

    def test_scan_absent_map1_partial(self) -> None:
        """No scan in BOM → MAP.1 is PARTIAL."""
        bom = ModelBOM(
            model_id="rmf-v2",
            base_model="llama3",
            run_id="rmf-run-002",
            data=[DataComponent("train", "h", 100)],
            code=CodeComponent(git_sha="sha1", dependencies_hash="dh", framework="trl"),
            evals=[EvalComponent("mmlu", 0.7)],
            created_at=_TS,
            scan=None,
        )
        passport = _signed_passport(bom)
        report = RegulationGenerator().nist_ai_rmf_summary(passport)
        fc = next(f for f in report.fields if f.field_id == "MAP.1")
        assert fc.coverage == Coverage.PARTIAL

    def test_non_spurious_reward_measure3_satisfied(self) -> None:
        """Non-spurious reward_validity → MEASURE.3 is SATISFIED."""
        passport = _signed_passport(_rmf_full_bom())
        report = RegulationGenerator().nist_ai_rmf_summary(passport)
        fc = next(f for f in report.fields if f.field_id == "MEASURE.3")
        assert fc.coverage == Coverage.SATISFIED

    def test_spurious_reward_measure3_partial(self) -> None:
        """Spurious reward_validity → MEASURE.3 is PARTIAL."""
        bom = ModelBOM(
            model_id="rmf-v3",
            base_model="llama3",
            run_id="rmf-run-003",
            data=[DataComponent("train", "h", 100)],
            code=CodeComponent(git_sha="sha1", dependencies_hash="dh", framework="trl"),
            evals=[EvalComponent("mmlu", 0.7)],
            created_at=_TS,
            reward_validity=RewardValidityComponent(
                "bad_reward", "rh", 0.3, spurious=True
            ),
        )
        passport = _signed_passport(bom)
        report = RegulationGenerator().nist_ai_rmf_summary(passport)
        fc = next(f for f in report.fields if f.field_id == "MEASURE.3")
        assert fc.coverage == Coverage.PARTIAL

    def test_artifact_identifier(self) -> None:
        """artifact field equals 'nist_ai_rmf'."""
        passport = _signed_passport(_rmf_full_bom())
        report = RegulationGenerator().nist_ai_rmf_summary(passport)
        assert report.artifact == "nist_ai_rmf"

    def test_coverage_score_positive_for_typical_bom(self) -> None:
        """coverage_score() > 0 for a typical full BOM."""
        passport = _signed_passport(_rmf_full_bom())
        report = RegulationGenerator().nist_ai_rmf_summary(passport)
        assert report.coverage_score() > 0.0

    def test_deterministic_output(self) -> None:
        """Calling nist_ai_rmf_summary twice produces identical reports."""
        passport = _signed_passport(_rmf_full_bom())
        gen = RegulationGenerator()
        r1 = gen.nist_ai_rmf_summary(passport)
        r2 = gen.nist_ai_rmf_summary(passport)
        assert r1.markdown == r2.markdown
        assert r1.data == r2.data

    def test_missing_reward_validity_measure3_missing(self) -> None:
        """No reward_validity in BOM → MEASURE.3 is MISSING."""
        bom = ModelBOM(
            model_id="rmf-v4",
            base_model="llama3",
            run_id="rmf-run-004",
            data=[DataComponent("train", "h", 100)],
            code=CodeComponent(git_sha="sha1", dependencies_hash="dh", framework="trl"),
            evals=[EvalComponent("mmlu", 0.7)],
            created_at=_TS,
            reward_validity=None,
        )
        passport = _signed_passport(bom)
        report = RegulationGenerator().nist_ai_rmf_summary(passport)
        fc = next(f for f in report.fields if f.field_id == "MEASURE.3")
        assert fc.coverage == Coverage.MISSING

    def test_govern2_always_partial(self) -> None:
        """GOVERN.2 is always PARTIAL (org process)."""
        passport = _signed_passport(_rmf_full_bom())
        report = RegulationGenerator().nist_ai_rmf_summary(passport)
        fc = next(f for f in report.fields if f.field_id == "GOVERN.2")
        assert fc.coverage == Coverage.PARTIAL

    def test_manage1_always_partial(self) -> None:
        """MANAGE.1 is always PARTIAL (org incident response process)."""
        passport = _signed_passport(_rmf_full_bom())
        report = RegulationGenerator().nist_ai_rmf_summary(passport)
        fc = next(f for f in report.fields if f.field_id == "MANAGE.1")
        assert fc.coverage == Coverage.PARTIAL

    def test_manage2_satisfied_with_run_id_and_created_at(self) -> None:
        """MANAGE.2 is SATISFIED when run_id and created_at are both present."""
        passport = _signed_passport(_rmf_full_bom())
        report = RegulationGenerator().nist_ai_rmf_summary(passport)
        fc = next(f for f in report.fields if f.field_id == "MANAGE.2")
        assert fc.coverage == Coverage.SATISFIED

    def test_fairness_eval_in_hyperparameters_measure2_satisfied(self) -> None:
        """fairness_eval in hyperparameters → MEASURE.2 is SATISFIED."""
        passport = _signed_passport(_rmf_full_bom())
        report = RegulationGenerator().nist_ai_rmf_summary(passport)
        fc = next(f for f in report.fields if f.field_id == "MEASURE.2")
        assert fc.coverage == Coverage.SATISFIED
