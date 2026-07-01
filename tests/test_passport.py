from __future__ import annotations

import json
from pathlib import Path

import pytest

from provenir.core.abstractions import RunManifest
from provenir.governance.audit import AuditLogger
from provenir.governance.bom import (
    CodeComponent,
    DataComponent,
    EvalComponent,
    ModelBOM,
)
from provenir.governance.passport import (
    Attestation,
    ModelPassport,
    PassportSigner,
    PassportStore,
    generate_passport_from_manifest,
)

KEY = b"test-key"
OTHER_KEY = b"other-key"
TS = "2026-01-01T00:00:00Z"


def _clean_bom() -> ModelBOM:
    return ModelBOM(
        model_id="model-1",
        base_model="base-model",
        run_id="run-1",
        data=[
            DataComponent(
                name="train",
                content_hash="datahash",
                num_records=100,
                license="apache-2.0",
                pii_scanned=True,
                contamination_checked=True,
            )
        ],
        code=CodeComponent(git_sha="sha", dependencies_hash="deps", framework="trl"),
        evals=[EvalComponent(benchmark="mmlu", score=0.71)],
        hyperparameters={"lr": 0.001, "epochs": 3},
        created_at=TS,
    )


def _risky_bom() -> ModelBOM:
    return ModelBOM(
        model_id="model-2",
        base_model="base-model",
        run_id="run-2",
        data=[DataComponent(name="train", content_hash="h", num_records=1)],
        code=CodeComponent(git_sha="sha", dependencies_hash="deps", framework="trl"),
        evals=[EvalComponent(benchmark="mmlu", score=0.5, contaminated=True)],
        hyperparameters={},
        created_at=TS,
    )


# --- DataComponent / CodeComponent / EvalComponent validation ---


def test_data_component_rejects_empty_name() -> None:
    with pytest.raises(ValueError):
        DataComponent(name="", content_hash="h", num_records=1)


def test_data_component_rejects_empty_hash() -> None:
    with pytest.raises(ValueError):
        DataComponent(name="d", content_hash="", num_records=1)


def test_data_component_rejects_negative_records() -> None:
    with pytest.raises(ValueError):
        DataComponent(name="d", content_hash="h", num_records=-1)


def test_code_component_rejects_empty_fields() -> None:
    with pytest.raises(ValueError):
        CodeComponent(git_sha="", dependencies_hash="d", framework="f")


def test_eval_component_rejects_empty_benchmark() -> None:
    with pytest.raises(ValueError):
        EvalComponent(benchmark="", score=0.1)


# --- ModelBOM validation ---


def test_bom_rejects_empty_model_id() -> None:
    with pytest.raises(ValueError):
        ModelBOM(
            model_id="",
            base_model="b",
            run_id="r",
            data=[],
            code=CodeComponent(git_sha="s", dependencies_hash="d", framework="f"),
            evals=[],
        )


def test_bom_rejects_empty_base_model() -> None:
    with pytest.raises(ValueError):
        ModelBOM(
            model_id="m",
            base_model="",
            run_id="r",
            data=[],
            code=CodeComponent(git_sha="s", dependencies_hash="d", framework="f"),
            evals=[],
        )


def test_bom_rejects_empty_run_id() -> None:
    with pytest.raises(ValueError):
        ModelBOM(
            model_id="m",
            base_model="b",
            run_id="",
            data=[],
            code=CodeComponent(git_sha="s", dependencies_hash="d", framework="f"),
            evals=[],
        )


# --- canonical_json / content_hash determinism ---


def test_canonical_json_is_sorted_keys() -> None:
    bom = _clean_bom()
    parsed = json.loads(bom.canonical_json())
    assert list(parsed.keys()) == sorted(parsed.keys())


def test_canonical_json_independent_of_hyperparameter_order() -> None:
    bom_a = _clean_bom()
    bom_b = ModelBOM(
        model_id="model-1",
        base_model="base-model",
        run_id="run-1",
        data=list(bom_a.data),
        code=bom_a.code,
        evals=list(bom_a.evals),
        hyperparameters={"epochs": 3, "lr": 0.001},
        created_at=TS,
    )
    assert bom_a.canonical_json() == bom_b.canonical_json()


def test_content_hash_is_stable_across_instances() -> None:
    assert _clean_bom().content_hash() == _clean_bom().content_hash()


def test_content_hash_length_is_sha256_hex() -> None:
    assert len(_clean_bom().content_hash()) == 64


def test_content_hash_changes_when_content_changes() -> None:
    bom_a = _clean_bom()
    bom_b = ModelBOM(
        model_id="model-1",
        base_model="base-model",
        run_id="run-1",
        data=list(bom_a.data),
        code=bom_a.code,
        evals=[EvalComponent(benchmark="mmlu", score=0.99)],
        hyperparameters=dict(bom_a.hyperparameters),
        created_at=TS,
    )
    assert bom_a.content_hash() != bom_b.content_hash()


# --- risk_flags ---


def test_risk_flags_clean_bom_is_empty() -> None:
    assert _clean_bom().risk_flags() == []


def test_risk_flags_unscanned_pii() -> None:
    bom = ModelBOM(
        model_id="m",
        base_model="b",
        run_id="r",
        data=[
            DataComponent(
                name="d",
                content_hash="h",
                num_records=1,
                license="mit",
                pii_scanned=False,
                contamination_checked=True,
            )
        ],
        code=CodeComponent(git_sha="s", dependencies_hash="d", framework="f"),
        evals=[],
    )
    assert "unscanned_pii" in bom.risk_flags()


def test_risk_flags_unchecked_contamination() -> None:
    bom = ModelBOM(
        model_id="m",
        base_model="b",
        run_id="r",
        data=[
            DataComponent(
                name="d",
                content_hash="h",
                num_records=1,
                license="mit",
                pii_scanned=True,
                contamination_checked=False,
            )
        ],
        code=CodeComponent(git_sha="s", dependencies_hash="d", framework="f"),
        evals=[],
    )
    assert "unchecked_contamination" in bom.risk_flags()


def test_risk_flags_unknown_license() -> None:
    bom = ModelBOM(
        model_id="m",
        base_model="b",
        run_id="r",
        data=[
            DataComponent(
                name="d",
                content_hash="h",
                num_records=1,
                pii_scanned=True,
                contamination_checked=True,
            )
        ],
        code=CodeComponent(git_sha="s", dependencies_hash="d", framework="f"),
        evals=[],
    )
    assert "unknown_license" in bom.risk_flags()


def test_risk_flags_contaminated_eval() -> None:
    assert "contaminated_eval" in _risky_bom().risk_flags()


def test_risk_flags_sorted_and_deduplicated() -> None:
    flags = _risky_bom().risk_flags()
    assert flags == sorted(flags)
    assert len(flags) == len(set(flags))


# --- BOM to_dict ---


def test_bom_to_dict_round_trips_through_json() -> None:
    bom = _clean_bom()
    data = bom.to_dict()
    assert json.loads(json.dumps(data)) == data


# --- sign / verify ---


def test_sign_then_verify_true() -> None:
    passport = PassportSigner(KEY).sign(_clean_bom(), signed_at=TS)
    assert passport.verify(KEY) is True


def test_signer_verify_helper_true() -> None:
    signer = PassportSigner(KEY)
    passport = signer.sign(_clean_bom(), signed_at=TS)
    assert signer.verify(passport) is True


def test_signature_is_deterministic() -> None:
    a = PassportSigner(KEY).sign(_clean_bom(), signed_at=TS)
    b = PassportSigner(KEY).sign(_clean_bom(), signed_at=TS)
    assert a.attestation is not None and b.attestation is not None
    assert a.attestation.signature == b.attestation.signature


def test_attestation_records_algorithm_and_key_id() -> None:
    passport = PassportSigner(KEY, key_id="ci").sign(_clean_bom(), signed_at=TS)
    assert passport.attestation is not None
    assert passport.attestation.algorithm == "HMAC-SHA256"
    assert passport.attestation.key_id == "ci"
    assert passport.attestation.signed_at == TS


def test_wrong_key_verify_false() -> None:
    passport = PassportSigner(KEY).sign(_clean_bom(), signed_at=TS)
    assert passport.verify(OTHER_KEY) is False


def test_tamper_detection_verify_false() -> None:
    passport = PassportSigner(KEY).sign(_clean_bom(), signed_at=TS)
    assert passport.attestation is not None
    tampered_bom = ModelBOM(
        model_id="model-1",
        base_model="base-model",
        run_id="run-1",
        data=list(passport.bom.data),
        code=passport.bom.code,
        evals=[EvalComponent(benchmark="mmlu", score=0.99)],
        hyperparameters=dict(passport.bom.hyperparameters),
        created_at=TS,
    )
    tampered = ModelPassport(bom=tampered_bom, attestation=passport.attestation)
    assert tampered.verify(KEY) is False


def test_unsigned_passport_verify_false() -> None:
    passport = ModelPassport(bom=_clean_bom(), attestation=None)
    assert passport.verify(KEY) is False


def test_signer_rejects_empty_key() -> None:
    with pytest.raises(ValueError):
        PassportSigner(b"")


# --- to_markdown ---


def test_markdown_contains_model_id() -> None:
    passport = PassportSigner(KEY).sign(_clean_bom(), signed_at=TS)
    assert "model-1" in passport.to_markdown()


def test_markdown_contains_risk_flags() -> None:
    passport = PassportSigner(KEY).sign(_risky_bom(), signed_at=TS)
    markdown = passport.to_markdown()
    assert "contaminated_eval" in markdown
    assert "unscanned_pii" in markdown


def test_markdown_shows_signed_status() -> None:
    passport = PassportSigner(KEY).sign(_clean_bom(), signed_at=TS)
    assert "SIGNED" in passport.to_markdown()


def test_markdown_shows_unsigned_status() -> None:
    passport = ModelPassport(bom=_clean_bom(), attestation=None)
    assert "UNSIGNED" in passport.to_markdown()


# --- to_dict / from_dict round-trip ---


def test_passport_to_dict_from_dict_round_trip_signed() -> None:
    passport = PassportSigner(KEY).sign(_clean_bom(), signed_at=TS)
    restored = ModelPassport.from_dict(passport.to_dict())
    assert restored.bom.content_hash() == passport.bom.content_hash()
    assert restored.verify(KEY) is True


def test_passport_to_dict_from_dict_round_trip_unsigned() -> None:
    passport = ModelPassport(bom=_clean_bom(), attestation=None)
    restored = ModelPassport.from_dict(passport.to_dict())
    assert restored.attestation is None
    assert restored.bom.content_hash() == passport.bom.content_hash()


def test_passport_to_json_from_dict_round_trip() -> None:
    passport = PassportSigner(KEY).sign(_risky_bom(), signed_at=TS)
    restored = ModelPassport.from_dict(json.loads(passport.to_json()))
    assert restored.bom.risk_flags() == passport.bom.risk_flags()
    assert restored.verify(KEY) is True


def test_attestation_from_dict_round_trip() -> None:
    attestation = Attestation("HMAC-SHA256", "c2ln", "default", TS)
    assert Attestation.from_dict(attestation.to_dict()) == attestation


# --- PassportStore ---


def test_store_save_and_load(tmp_path: Path) -> None:
    store = PassportStore(tmp_path)
    passport = PassportSigner(KEY).sign(_clean_bom(), signed_at=TS)
    path = store.save(passport)
    assert path.exists()
    loaded = store.load("model-1")
    assert loaded.verify(KEY) is True


def test_store_list_ids(tmp_path: Path) -> None:
    store = PassportStore(tmp_path)
    store.save(PassportSigner(KEY).sign(_clean_bom(), signed_at=TS))
    store.save(PassportSigner(KEY).sign(_risky_bom(), signed_at=TS))
    assert store.list_ids() == ["model-1", "model-2"]


def test_store_writes_audit_log_line(tmp_path: Path) -> None:
    audit_dir = tmp_path / "audit"
    logger = AuditLogger(log_dir=audit_dir)
    store = PassportStore(tmp_path / "store", audit_logger=logger)
    store.save(PassportSigner(KEY).sign(_clean_bom(), signed_at=TS))
    log_path = audit_dir / "audit.jsonl"
    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["event"] == "passport_issued"
    assert entry["model_id"] == "model-1"


def test_store_save_without_logger_does_not_error(tmp_path: Path) -> None:
    store = PassportStore(tmp_path)
    store.save(PassportSigner(KEY).sign(_clean_bom(), signed_at=TS))
    assert store.list_ids() == ["model-1"]


# --- generate_passport_from_manifest ---


def test_generate_passport_from_manifest_signs() -> None:
    manifest = RunManifest(run_id="run-1")
    passport = generate_passport_from_manifest(
        manifest, _clean_bom(), PassportSigner(KEY), signed_at=TS
    )
    assert passport.attestation is not None
    assert passport.verify(KEY) is True


def test_generate_passport_from_manifest_rejects_run_id_mismatch() -> None:
    manifest = RunManifest(run_id="different")
    with pytest.raises(ValueError):
        generate_passport_from_manifest(manifest, _clean_bom(), PassportSigner(KEY))
