"""Tests for the DSSE signing module (sigstore_signing.py)."""

from __future__ import annotations

import base64
import json

from provenir.governance.bom import (
    CodeComponent,
    DataComponent,
    EvalComponent,
    ModelBOM,
)
from provenir.governance.passport import ModelPassport, PassportSigner
from provenir.governance.sigstore_signing import (
    DSSEEnvelope,
    DSSESignature,
    _pae,
    envelope_from_dict,
    passport_from_dsse,
    sign_dsse,
    verify_dsse,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _make_bom(model_id: str = "test-model") -> ModelBOM:
    return ModelBOM(
        model_id=model_id,
        base_model="base-llm",
        run_id="run-001",
        data=[
            DataComponent(
                name="train",
                content_hash="abc123",
                num_records=1000,
                license="apache-2.0",
                pii_scanned=True,
                contamination_checked=True,
            )
        ],
        code=CodeComponent(
            git_sha="deadbeef",
            dependencies_hash="pip-hash-sha256",
            framework="trl",
        ),
        evals=[EvalComponent(benchmark="mmlu", score=0.75)],
        hyperparameters={"lr": 2e-5, "epochs": 3},
        created_at="2026-07-12T00:00:00Z",
    )


def _make_passport(model_id: str = "test-model", key: bytes = b"test-key") -> ModelPassport:
    bom = _make_bom(model_id)
    signer = PassportSigner(key, key_id="test")
    return signer.sign(bom, signed_at="2026-07-12T00:00:00Z")


_KEY = b"provenir-dsse-test-key"
_WRONG_KEY = b"wrong-key"


# ---------------------------------------------------------------------------
# Test 1: sign_dsse returns DSSEEnvelope with correct payload_type
# ---------------------------------------------------------------------------


def test_sign_dsse_returns_envelope_with_correct_payload_type() -> None:
    passport = _make_passport()
    envelope = sign_dsse(passport, _KEY)
    assert isinstance(envelope, DSSEEnvelope)
    assert envelope.payload_type == "application/vnd.provenir.passport+json"


# ---------------------------------------------------------------------------
# Test 2: envelope.payload decodes back to passport JSON
# ---------------------------------------------------------------------------


def test_envelope_payload_decodes_to_passport_json() -> None:
    passport = _make_passport()
    envelope = sign_dsse(passport, _KEY)
    decoded = base64.urlsafe_b64decode(envelope.payload.encode()).decode("utf-8")
    parsed = json.loads(decoded)
    assert parsed["bom"]["model_id"] == "test-model"


# ---------------------------------------------------------------------------
# Test 3: verify_dsse returns True with correct key
# ---------------------------------------------------------------------------


def test_verify_dsse_correct_key() -> None:
    passport = _make_passport()
    envelope = sign_dsse(passport, _KEY)
    assert verify_dsse(envelope, _KEY) is True


# ---------------------------------------------------------------------------
# Test 4: verify_dsse returns False with wrong key
# ---------------------------------------------------------------------------


def test_verify_dsse_wrong_key() -> None:
    passport = _make_passport()
    envelope = sign_dsse(passport, _KEY)
    assert verify_dsse(envelope, _WRONG_KEY) is False


# ---------------------------------------------------------------------------
# Test 5: verify_dsse returns False with empty signatures list
# ---------------------------------------------------------------------------


def test_verify_dsse_empty_signatures() -> None:
    passport = _make_passport()
    envelope = sign_dsse(passport, _KEY)
    empty_envelope = DSSEEnvelope(
        payload_type=envelope.payload_type,
        payload=envelope.payload,
        signatures=[],
    )
    assert verify_dsse(empty_envelope, _KEY) is False


# ---------------------------------------------------------------------------
# Test 6: passport_from_dsse recovers the original passport (bom.model_id)
# ---------------------------------------------------------------------------


def test_passport_from_dsse_recovers_model_id() -> None:
    passport = _make_passport(model_id="my-special-model")
    envelope = sign_dsse(passport, _KEY)
    recovered = passport_from_dsse(envelope)
    assert recovered.bom.model_id == "my-special-model"


# ---------------------------------------------------------------------------
# Test 7: passport_from_dsse → verify still passes for embedded attestation
# ---------------------------------------------------------------------------


def test_passport_from_dsse_attestation_still_verifiable() -> None:
    passport = _make_passport(key=_KEY)
    envelope = sign_dsse(passport, _KEY)
    recovered = passport_from_dsse(envelope)
    # The original HMAC-SHA256 attestation is preserved inside the envelope
    assert recovered.verify(_KEY) is True


# ---------------------------------------------------------------------------
# Test 8: DSSEEnvelope.to_dict() has required keys
# ---------------------------------------------------------------------------


def test_envelope_to_dict_has_required_keys() -> None:
    passport = _make_passport()
    envelope = sign_dsse(passport, _KEY)
    d = envelope.to_dict()
    assert "payloadType" in d
    assert "payload" in d
    assert "signatures" in d


# ---------------------------------------------------------------------------
# Test 9: DSSEEnvelope.to_json() is valid JSON
# ---------------------------------------------------------------------------


def test_envelope_to_json_is_valid_json() -> None:
    passport = _make_passport()
    envelope = sign_dsse(passport, _KEY)
    raw = envelope.to_json()
    parsed = json.loads(raw)
    assert isinstance(parsed, dict)
    assert parsed["payloadType"] == "application/vnd.provenir.passport+json"


# ---------------------------------------------------------------------------
# Test 10: DSSEEnvelope.content_hash() is deterministic and 64 hex chars
# ---------------------------------------------------------------------------


def test_envelope_content_hash_is_deterministic_and_64_chars() -> None:
    passport = _make_passport()
    envelope = sign_dsse(passport, _KEY)
    h1 = envelope.content_hash()
    h2 = envelope.content_hash()
    assert h1 == h2
    assert len(h1) == 64
    assert all(c in "0123456789abcdef" for c in h1)


# ---------------------------------------------------------------------------
# Test 11: DSSEEnvelope.content_hash() changes when payload changes
# ---------------------------------------------------------------------------


def test_envelope_content_hash_changes_with_payload() -> None:
    passport_a = _make_passport(model_id="model-a")
    passport_b = _make_passport(model_id="model-b")
    env_a = sign_dsse(passport_a, _KEY)
    env_b = sign_dsse(passport_b, _KEY)
    assert env_a.content_hash() != env_b.content_hash()


# ---------------------------------------------------------------------------
# Test 12: envelope_from_dict round-trips correctly
# ---------------------------------------------------------------------------


def test_envelope_from_dict_round_trips() -> None:
    passport = _make_passport()
    original = sign_dsse(passport, _KEY)
    restored = envelope_from_dict(original.to_dict())
    assert restored.payload_type == original.payload_type
    assert restored.payload == original.payload
    assert len(restored.signatures) == len(original.signatures)
    assert restored.signatures[0].keyid == original.signatures[0].keyid
    assert restored.signatures[0].sig == original.signatures[0].sig
    # Verification must still pass after round-trip
    assert verify_dsse(restored, _KEY) is True


# ---------------------------------------------------------------------------
# Test 13: DSSESignature.to_dict() has keyid and sig keys
# ---------------------------------------------------------------------------


def test_dsse_signature_to_dict_has_required_keys() -> None:
    sig = DSSESignature(keyid="my-key", sig="c2lnbmF0dXJl")
    d = sig.to_dict()
    assert "keyid" in d
    assert "sig" in d
    assert d["keyid"] == "my-key"
    assert d["sig"] == "c2lnbmF0dXJl"


# ---------------------------------------------------------------------------
# Test 14: PAE is correctly computed for a known input
# ---------------------------------------------------------------------------


def test_pae_known_vector() -> None:
    # From the DSSE spec example: PAE("text/plain", "hello")
    # = b"DSSEv1 10 text/plain 5 hello"
    result = _pae("text/plain", "hello")
    assert result == b"DSSEv1 10 text/plain 5 hello"


def test_pae_provenir_payload_type_structure() -> None:
    payload_type = "application/vnd.provenir.passport+json"
    payload = "AAAA"
    result = _pae(payload_type, payload)
    expected = (
        b"DSSEv1 "
        + str(len(payload_type.encode())).encode()
        + b" "
        + payload_type.encode()
        + b" "
        + str(len(payload.encode())).encode()
        + b" "
        + payload.encode()
    )
    assert result == expected


# ---------------------------------------------------------------------------
# Test 15: Two sign_dsse calls with same passport+key produce same envelope
# ---------------------------------------------------------------------------


def test_sign_dsse_is_deterministic() -> None:
    passport = _make_passport()
    env1 = sign_dsse(passport, _KEY)
    env2 = sign_dsse(passport, _KEY)
    assert env1.payload == env2.payload
    assert env1.signatures[0].sig == env2.signatures[0].sig
    assert env1.to_json() == env2.to_json()


# ---------------------------------------------------------------------------
# Test 16: Tampered payload causes verify_dsse to return False
# ---------------------------------------------------------------------------


def test_verify_dsse_tampered_payload_fails() -> None:
    passport = _make_passport()
    envelope = sign_dsse(passport, _KEY)
    # Tamper the payload by replacing it with a different base64url-encoded value
    tampered_json = json.loads(envelope.payload_json())
    tampered_json["bom"]["model_id"] = "hacked-model"
    tampered_payload = base64.urlsafe_b64encode(
        json.dumps(tampered_json).encode("utf-8")
    ).decode("ascii")
    tampered_envelope = DSSEEnvelope(
        payload_type=envelope.payload_type,
        payload=tampered_payload,
        signatures=envelope.signatures,  # original signatures, wrong payload
    )
    assert verify_dsse(tampered_envelope, _KEY) is False


# ---------------------------------------------------------------------------
# Additional: key_id is carried through into the envelope
# ---------------------------------------------------------------------------


def test_sign_dsse_key_id_propagated() -> None:
    passport = _make_passport()
    envelope = sign_dsse(passport, _KEY, key_id="ci-runner")
    assert envelope.signatures[0].keyid == "ci-runner"


# ---------------------------------------------------------------------------
# Additional: envelope_from_dict with no signatures key
# ---------------------------------------------------------------------------


def test_envelope_from_dict_missing_signatures_defaults_to_empty() -> None:
    data: dict[str, object] = {
        "payloadType": "application/vnd.provenir.passport+json",
        "payload": "cGF5bG9hZA==",
    }
    envelope = envelope_from_dict(data)
    assert envelope.signatures == []


# ---------------------------------------------------------------------------
# Additional: payload_json() returns a parseable JSON string
# ---------------------------------------------------------------------------


def test_envelope_payload_json_is_parseable() -> None:
    passport = _make_passport()
    envelope = sign_dsse(passport, _KEY)
    pj = envelope.payload_json()
    parsed = json.loads(pj)
    assert "bom" in parsed
