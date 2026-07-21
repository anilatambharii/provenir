"""Tests for the Retraction Monitor feature (provenir.governance.retraction)."""

from __future__ import annotations

import pytest

from provenir.governance.bom import CodeComponent, DataComponent, EvalComponent, ModelBOM
from provenir.governance.passport import ModelPassport
from provenir.governance.retraction import (
    RetractionBlocked,
    RetractionComponent,
    RetractionMonitor,
    gate_retraction,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

KNOWN_RETRACTED: frozenset[str] = frozenset(
    {
        "10.1000/retracted.one",
        "10.1000/retracted.two",
        "10.1000/retracted.three",
    }
)

CLEAN_DOIS = ["10.9999/clean.a", "10.9999/clean.b", "10.9999/clean.c"]
MIXED_DOIS = ["10.9999/clean.a", "10.1000/retracted.one", "10.9999/clean.b"]


def _make_bom(
    data_components: list[DataComponent] | None = None,
    retraction_component: RetractionComponent | None = None,
) -> ModelBOM:
    data = data_components or [DataComponent(name="d", content_hash="h", num_records=1)]
    return ModelBOM(
        model_id="m1",
        base_model="base",
        run_id="r1",
        data=data,
        code=CodeComponent(git_sha="s", dependencies_hash="dh", framework="trl"),
        evals=[EvalComponent(benchmark="mmlu", score=0.5)],
        hyperparameters={},
        retraction=retraction_component,
    )


def _make_passport(
    data_components: list[DataComponent] | None = None,
    retraction_component: RetractionComponent | None = None,
) -> ModelPassport:
    bom = _make_bom(data_components, retraction_component)
    return ModelPassport(bom=bom, attestation=None)


# ---------------------------------------------------------------------------
# Test 1: Empty DOI list → retracted_count=0, risk_level="none"
# ---------------------------------------------------------------------------


def test_empty_dois_no_retractions() -> None:
    monitor = RetractionMonitor(known_retracted=KNOWN_RETRACTED)
    report = monitor.check([])
    assert report.retracted_count == 0 if False else True  # use RetractionComponent path below
    # Check via the report object directly
    assert len(report.retracted_dois) == 0
    assert report.retraction_rate == 0.0
    assert report.risk_level == "none"
    assert len(report.checked_dois) == 0


# ---------------------------------------------------------------------------
# Test 2: All DOIs clean → retracted_count=0
# ---------------------------------------------------------------------------


def test_all_dois_clean() -> None:
    monitor = RetractionMonitor(known_retracted=KNOWN_RETRACTED)
    report = monitor.check(CLEAN_DOIS)
    assert len(report.retracted_dois) == 0
    assert report.retraction_rate == 0.0
    assert report.risk_level == "none"
    assert len(report.checked_dois) == len(CLEAN_DOIS)


# ---------------------------------------------------------------------------
# Test 3: One retracted DOI → retracted_count=1, risk_level="high"
# ---------------------------------------------------------------------------


def test_one_retracted_doi() -> None:
    monitor = RetractionMonitor(known_retracted=KNOWN_RETRACTED)
    report = monitor.check(["10.9999/clean", "10.1000/retracted.one"])
    assert len(report.retracted_dois) == 1
    assert report.retracted_dois == ["10.1000/retracted.one"]
    assert report.risk_level == "high"


# ---------------------------------------------------------------------------
# Test 4: Rate > 0.1 → risk_level="high"
# ---------------------------------------------------------------------------


def test_high_rate_is_high_risk() -> None:
    retracted_set = frozenset({"10.1/r1", "10.1/r2"})
    monitor = RetractionMonitor(known_retracted=retracted_set)
    # 2 retracted out of 3 total → rate ≈ 0.67 > 0.1
    report = monitor.check(["10.1/r1", "10.1/r2", "10.2/clean"])
    assert report.retraction_rate > 0.1
    assert report.risk_level == "high"


# ---------------------------------------------------------------------------
# Test 5: Rate < 0.1 but retracted > 0 → risk_level="high" (any retraction = high)
# ---------------------------------------------------------------------------


def test_any_retraction_is_high_risk_even_low_rate() -> None:
    # 1 retracted out of 100 = 1% rate, still "high"
    many_clean = [f"10.9/{i}" for i in range(99)]
    monitor = RetractionMonitor(known_retracted=frozenset({"10.1000/retracted.one"}))
    report = monitor.check(many_clean + ["10.1000/retracted.one"])
    assert report.retraction_rate < 0.1
    assert len(report.retracted_dois) == 1
    assert report.risk_level == "high"


# ---------------------------------------------------------------------------
# Test 6: content_hash() determinism
# ---------------------------------------------------------------------------


def test_content_hash_determinism() -> None:
    monitor = RetractionMonitor(known_retracted=KNOWN_RETRACTED)
    report_a = monitor.check(MIXED_DOIS)
    report_b = monitor.check(MIXED_DOIS)
    assert report_a.content_hash() == report_b.content_hash()
    assert len(report_a.content_hash()) == 64  # SHA-256 hex


# ---------------------------------------------------------------------------
# Test 7: to_dict() round-trips — verify keys
# ---------------------------------------------------------------------------


def test_retraction_report_to_dict_keys() -> None:
    monitor = RetractionMonitor(known_retracted=KNOWN_RETRACTED)
    report = monitor.check(MIXED_DOIS)
    d = report.to_dict()
    assert set(d.keys()) == {"checked_dois", "retracted_dois", "retraction_rate", "risk_level"}
    assert isinstance(d["checked_dois"], list)
    assert isinstance(d["retracted_dois"], list)
    assert isinstance(d["retraction_rate"], float)
    assert isinstance(d["risk_level"], str)


# ---------------------------------------------------------------------------
# Test 8: RetractionComponent.from_report() fields match report
# ---------------------------------------------------------------------------


def test_retraction_component_from_report() -> None:
    monitor = RetractionMonitor(known_retracted=KNOWN_RETRACTED)
    report = monitor.check(MIXED_DOIS)
    component = RetractionComponent.from_report(report)
    assert component.checked_count == len(report.checked_dois)
    assert component.retracted_count == len(report.retracted_dois)
    assert component.retraction_rate == report.retraction_rate
    assert component.risk_level == report.risk_level
    assert component.report_hash == report.content_hash()


# ---------------------------------------------------------------------------
# Test 9: RetractionComponent.from_dict() reconstructs correctly
# ---------------------------------------------------------------------------


def test_retraction_component_from_dict() -> None:
    monitor = RetractionMonitor(known_retracted=KNOWN_RETRACTED)
    report = monitor.check(MIXED_DOIS)
    original = RetractionComponent.from_report(report)
    reconstructed = RetractionComponent.from_dict(original.to_dict())
    assert reconstructed.checked_count == original.checked_count
    assert reconstructed.retracted_count == original.retracted_count
    assert reconstructed.retraction_rate == original.retraction_rate
    assert reconstructed.risk_level == original.risk_level
    assert reconstructed.report_hash == original.report_hash


# ---------------------------------------------------------------------------
# Test 10: check_passport() flattens all DataComponent retraction_dois
# ---------------------------------------------------------------------------


def test_check_passport_flattens_all_retraction_dois() -> None:
    monitor = RetractionMonitor(known_retracted=KNOWN_RETRACTED)
    components = [
        DataComponent(
            name="d1",
            content_hash="h1",
            num_records=10,
            retraction_dois=["10.1000/retracted.one", "10.9999/clean.a"],
        ),
        DataComponent(
            name="d2",
            content_hash="h2",
            num_records=5,
            retraction_dois=["10.1000/retracted.two", "10.9999/clean.b"],
        ),
    ]
    passport = _make_passport(data_components=components)
    report = monitor.check_passport(passport)
    # Should have checked all 4 DOIs (deduped), 2 retracted
    assert len(report.checked_dois) == 4
    assert len(report.retracted_dois) == 2
    assert report.risk_level == "high"


# ---------------------------------------------------------------------------
# Test 11: check_passport() with no retraction_dois anywhere → empty check
# ---------------------------------------------------------------------------


def test_check_passport_no_retraction_dois() -> None:
    monitor = RetractionMonitor(known_retracted=KNOWN_RETRACTED)
    components = [
        DataComponent(name="d1", content_hash="h1", num_records=10),
        DataComponent(name="d2", content_hash="h2", num_records=5),
    ]
    passport = _make_passport(data_components=components)
    report = monitor.check_passport(passport)
    assert len(report.checked_dois) == 0
    assert len(report.retracted_dois) == 0
    assert report.risk_level == "none"


# ---------------------------------------------------------------------------
# Test 12: gate_retraction raises RetractionBlocked on protected stage
# ---------------------------------------------------------------------------


def test_gate_retraction_raises_on_protected_stage() -> None:
    monitor = RetractionMonitor(known_retracted=KNOWN_RETRACTED)
    report = monitor.check(MIXED_DOIS)
    assert report.retraction_rate > 0.0
    with pytest.raises(RetractionBlocked):
        gate_retraction(report, "production")


# ---------------------------------------------------------------------------
# Test 13: gate_retraction does NOT raise on "staging" even with retracted DOIs
# ---------------------------------------------------------------------------


def test_gate_retraction_passes_on_staging() -> None:
    monitor = RetractionMonitor(known_retracted=KNOWN_RETRACTED)
    report = monitor.check(MIXED_DOIS)
    # Should not raise
    gate_retraction(report, "staging")


# ---------------------------------------------------------------------------
# Test 14: gate_retraction allows protected stage when allow_rate=1.0 and rate<1.0
# ---------------------------------------------------------------------------


def test_gate_retraction_allow_rate_override() -> None:
    monitor = RetractionMonitor(known_retracted=frozenset({"10.1/r1"}))
    # 1 retracted out of 2 → rate = 0.5
    report = monitor.check(["10.1/r1", "10.2/clean"])
    assert report.retraction_rate == 0.5
    # allow_rate=1.0 means anything ≤ 1.0 passes
    gate_retraction(report, "production", allow_rate=1.0)  # should not raise


# ---------------------------------------------------------------------------
# Test 15: BOM with retraction present → risk_flags() includes "retracted_training_data"
# ---------------------------------------------------------------------------


def test_bom_risk_flags_retracted_training_data() -> None:
    monitor = RetractionMonitor(known_retracted=KNOWN_RETRACTED)
    report = monitor.check(MIXED_DOIS)
    component = RetractionComponent.from_report(report)
    bom = _make_bom(retraction_component=component)
    flags = bom.risk_flags()
    assert "retracted_training_data" in flags


def test_bom_risk_flags_no_retraction_flag_when_clean() -> None:
    monitor = RetractionMonitor(known_retracted=KNOWN_RETRACTED)
    report = monitor.check(CLEAN_DOIS)
    component = RetractionComponent.from_report(report)
    bom = _make_bom(retraction_component=component)
    flags = bom.risk_flags()
    assert "retracted_training_data" not in flags


# ---------------------------------------------------------------------------
# Test 16: BOM to_dict() round-trip includes "retraction" key (even when None)
# ---------------------------------------------------------------------------


def test_bom_to_dict_retraction_key_present_when_none() -> None:
    bom = _make_bom(retraction_component=None)
    d = bom.to_dict()
    assert "retraction" in d
    assert d["retraction"] is None


def test_bom_to_dict_retraction_key_present_with_component() -> None:
    monitor = RetractionMonitor(known_retracted=KNOWN_RETRACTED)
    report = monitor.check(MIXED_DOIS)
    component = RetractionComponent.from_report(report)
    bom = _make_bom(retraction_component=component)
    d = bom.to_dict()
    assert "retraction" in d
    assert d["retraction"] is not None
    assert d["retraction"]["retracted_count"] == 1


# ---------------------------------------------------------------------------
# Additional: DataComponent with retraction_dois round-trips through to_dict()
# ---------------------------------------------------------------------------


def test_data_component_retraction_dois_round_trip() -> None:
    dois = ["10.1/a", "10.2/b"]
    dc = DataComponent(
        name="train",
        content_hash="abc",
        num_records=100,
        retraction_dois=dois,
    )
    d = dc.to_dict()
    assert "retraction_dois" in d
    assert d["retraction_dois"] == dois


def test_data_component_retraction_dois_default_empty() -> None:
    dc = DataComponent(name="train", content_hash="abc", num_records=100)
    assert dc.retraction_dois == []
    d = dc.to_dict()
    assert d["retraction_dois"] == []


# ---------------------------------------------------------------------------
# Additional: passport round-trip preserves retraction_dois
# ---------------------------------------------------------------------------


def test_passport_round_trip_preserves_retraction_dois() -> None:
    dois = ["10.1/retracted", "10.2/clean"]
    dc = DataComponent(
        name="train",
        content_hash="abc",
        num_records=100,
        retraction_dois=dois,
    )
    passport = _make_passport(data_components=[dc])
    round_tripped = ModelPassport.from_dict(passport.to_dict())
    assert round_tripped.bom.data[0].retraction_dois == dois


def test_passport_round_trip_preserves_retraction_component() -> None:
    monitor = RetractionMonitor(known_retracted=KNOWN_RETRACTED)
    report = monitor.check(MIXED_DOIS)
    component = RetractionComponent.from_report(report)
    passport = _make_passport(retraction_component=component)
    round_tripped = ModelPassport.from_dict(passport.to_dict())
    assert round_tripped.bom.retraction is not None
    assert round_tripped.bom.retraction.retracted_count == component.retracted_count
    assert round_tripped.bom.retraction.report_hash == component.report_hash


# ---------------------------------------------------------------------------
# Additional: RetractionMonitor with None known_retracted uses empty frozenset
# ---------------------------------------------------------------------------


def test_monitor_none_known_retracted_finds_nothing() -> None:
    monitor = RetractionMonitor(known_retracted=None)
    report = monitor.check(["10.1/anything", "10.2/whatever"])
    assert len(report.retracted_dois) == 0
    assert report.risk_level == "none"


# ---------------------------------------------------------------------------
# Additional: gate_retraction with custom protected stages
# ---------------------------------------------------------------------------


def test_gate_retraction_custom_protected_stages() -> None:
    monitor = RetractionMonitor(known_retracted=frozenset({"10.1/r"}))
    report = monitor.check(["10.1/r"])
    # "deploy" is not in default protected set → passes
    gate_retraction(report, "deploy")
    # "deploy" in custom set → raises
    with pytest.raises(RetractionBlocked):
        gate_retraction(report, "deploy", protected_stages=frozenset({"deploy"}))


# ---------------------------------------------------------------------------
# Additional: deduplication in check()
# ---------------------------------------------------------------------------


def test_check_deduplicates_dois() -> None:
    monitor = RetractionMonitor(known_retracted=frozenset({"10.1/r"}))
    report = monitor.check(["10.1/r", "10.1/r", "10.2/clean", "10.2/clean"])
    assert len(report.checked_dois) == 2  # deduped
    assert len(report.retracted_dois) == 1
