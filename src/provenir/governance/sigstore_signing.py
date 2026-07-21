"""DSSE (Dead Simple Signing Envelope) signing for Provenir passports.

This module implements the DSSE envelope format used by Sigstore, in-toto, and
the broader software supply-chain security ecosystem (npm, PyPI, Maven, GitHub
Actions).  Passports wrapped in a DSSE envelope can be verified by any
conforming DSSE implementation without requiring Provenir-specific tooling.

Offline / HMAC variant
----------------------
This module implements the offline HMAC-SHA256 variant of DSSE — no network
access or certificate infrastructure is required.  The signature can be
verified by anyone who holds the shared key.

Optional Sigstore / Rekor integration
--------------------------------------
To submit a passport to the Rekor public transparency log (making signatures
publicly verifiable without key distribution), install the ``sigstore`` package
and use ``sigstore.sign()`` with the DSSE envelope payload.  The envelope
format produced here is compatible with Sigstore's DSSE-based signing workflow.
The Rekor log entry gives you a globally-visible, append-only record that a
specific passport was signed at a given time — suitable for PyPI, npm, and
Maven supply-chain compliance audits.

DSSE spec: https://github.com/secure-systems-lab/dsse
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from provenir.governance.passport import ModelPassport

_PAYLOAD_TYPE = "application/vnd.provenir.passport+json"


# ---------------------------------------------------------------------------
# PAE (Pre-Authentication Encoding)
# ---------------------------------------------------------------------------

def _pae(payload_type: str, payload: str) -> bytes:
    """Compute the DSSE Pre-Authentication Encoding for ``payload_type`` and ``payload``.

    PAE(type, body) = "DSSEv1" SP LEN(type) SP type SP LEN(body) SP body

    Args:
        payload_type: The DSSE payload type URI string.
        payload: The base64url-encoded payload string.

    Returns:
        The PAE bytes to be signed.

    Example:
        >>> _pae("text/plain", "hello")
        b'DSSEv1 10 text/plain 5 hello'
    """
    pt = payload_type.encode("utf-8")
    pb = payload.encode("utf-8")
    return (
        b"DSSEv1 "
        + str(len(pt)).encode() + b" " + pt + b" "
        + str(len(pb)).encode() + b" " + pb
    )


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DSSESignature:
    """A single DSSE signature entry.

    Attributes:
        keyid: An opaque identifier for the signing key.
        sig: Base64url-encoded raw signature bytes.

    Example:
        >>> s = DSSESignature(keyid="k1", sig="c2ln")
        >>> s.to_dict()
        {'keyid': 'k1', 'sig': 'c2ln'}
    """

    keyid: str
    sig: str  # base64url-encoded

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-serialisable dict representation."""
        return {"keyid": self.keyid, "sig": self.sig}


@dataclass(frozen=True)
class DSSEEnvelope:
    """A DSSE signing envelope wrapping a Provenir passport payload.

    Conforms to https://github.com/secure-systems-lab/dsse so the envelope
    can be consumed by any DSSE-aware tool (Sigstore, in-toto, cosign …).

    Attributes:
        payload_type: MIME type identifying the payload format.
        payload: Base64url-encoded canonical passport JSON.
        signatures: One or more :class:`DSSESignature` entries.

    Example:
        >>> env = DSSEEnvelope(payload="cGF5bG9hZA==", signatures=[])
        >>> env.payload_type
        'application/vnd.provenir.passport+json'
    """

    payload_type: str = _PAYLOAD_TYPE
    payload: str = ""
    signatures: list[DSSESignature] = field(default_factory=list)

    def payload_json(self) -> str:
        """Decode ``payload`` back to the canonical passport JSON string."""
        return base64.urlsafe_b64decode(self.payload.encode()).decode("utf-8")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict in DSSE wire format."""
        return {
            "payloadType": self.payload_type,
            "payload": self.payload,
            "signatures": [s.to_dict() for s in self.signatures],
        }

    def to_json(self) -> str:
        """Serialise the envelope to a canonical JSON string (sorted keys, 2-space indent)."""
        return json.dumps(self.to_dict(), sort_keys=True, indent=2)

    def content_hash(self) -> str:
        """Return the SHA-256 hex digest of :meth:`to_json` — suitable for audit logs.

        Example:
            >>> env = DSSEEnvelope(payload="abc", signatures=[])
            >>> len(env.content_hash())
            64
        """
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Public signing / verification API
# ---------------------------------------------------------------------------

def sign_dsse(
    passport: ModelPassport,
    key: bytes,
    key_id: str = "default",
) -> DSSEEnvelope:
    """Sign a :class:`~provenir.governance.passport.ModelPassport` into a DSSE envelope.

    The passport is serialised to canonical JSON, base64url-encoded into the
    DSSE ``payload`` field, and a HMAC-SHA256 signature is computed over the
    PAE of ``(payloadType, payload)``.

    Args:
        passport: The passport to sign.
        key: Raw HMAC-SHA256 signing key bytes.
        key_id: Opaque identifier for the key (stored in the envelope).

    Returns:
        A :class:`DSSEEnvelope` containing the signed passport.

    Example:
        >>> from provenir.governance.bom import (
        ...     CodeComponent, DataComponent, EvalComponent, ModelBOM)
        >>> from provenir.governance.passport import ModelPassport
        >>> bom = ModelBOM(
        ...     model_id="m1", base_model="base", run_id="r1",
        ...     data=[DataComponent(name="d", content_hash="h", num_records=1)],
        ...     code=CodeComponent(git_sha="s", dependencies_hash="dh", framework="trl"),
        ...     evals=[EvalComponent(benchmark="mmlu", score=0.5)],
        ...     hyperparameters={},
        ... )
        >>> passport = ModelPassport(bom=bom, attestation=None)
        >>> env = sign_dsse(passport, b"secret")
        >>> env.payload_type
        'application/vnd.provenir.passport+json'
    """
    passport_json = passport.to_json()
    payload = base64.urlsafe_b64encode(passport_json.encode("utf-8")).decode("ascii")
    message = _pae(_PAYLOAD_TYPE, payload)
    sig_bytes = hmac.new(key, message, hashlib.sha256).digest()
    sig = base64.urlsafe_b64encode(sig_bytes).decode("ascii")
    return DSSEEnvelope(
        payload_type=_PAYLOAD_TYPE,
        payload=payload,
        signatures=[DSSESignature(keyid=key_id, sig=sig)],
    )


def verify_dsse(envelope: DSSEEnvelope, key: bytes) -> bool:
    """Verify at least one signature in a DSSE envelope against ``key``.

    Computes the expected HMAC-SHA256 over the PAE of the envelope's
    ``payloadType`` and ``payload`` and compares in constant time.

    Args:
        envelope: The :class:`DSSEEnvelope` to verify.
        key: Raw HMAC-SHA256 key bytes.

    Returns:
        ``True`` if any signature in the envelope verifies; ``False`` otherwise.

    Example:
        >>> from provenir.governance.bom import (
        ...     CodeComponent, DataComponent, EvalComponent, ModelBOM)
        >>> from provenir.governance.passport import ModelPassport
        >>> bom = ModelBOM(
        ...     model_id="m1", base_model="base", run_id="r1",
        ...     data=[DataComponent(name="d", content_hash="h", num_records=1)],
        ...     code=CodeComponent(git_sha="s", dependencies_hash="dh", framework="trl"),
        ...     evals=[EvalComponent(benchmark="mmlu", score=0.5)],
        ...     hyperparameters={},
        ... )
        >>> passport = ModelPassport(bom=bom, attestation=None)
        >>> env = sign_dsse(passport, b"secret")
        >>> verify_dsse(env, b"secret")
        True
        >>> verify_dsse(env, b"wrong-key")
        False
    """
    if not envelope.signatures:
        return False
    message = _pae(envelope.payload_type, envelope.payload)
    expected_bytes = hmac.new(key, message, hashlib.sha256).digest()
    expected_sig = base64.urlsafe_b64encode(expected_bytes).decode("ascii")
    for signature in envelope.signatures:
        if hmac.compare_digest(expected_sig, signature.sig):
            return True
    return False


def envelope_from_dict(data: dict[str, Any]) -> DSSEEnvelope:
    """Reconstruct a :class:`DSSEEnvelope` from a dict (e.g. parsed from JSON).

    Args:
        data: A dict with keys ``payloadType``, ``payload``, and ``signatures``.

    Returns:
        The reconstructed :class:`DSSEEnvelope`.

    Example:
        >>> d = {"payloadType": "application/vnd.provenir.passport+json",
        ...      "payload": "cA==", "signatures": [{"keyid": "k", "sig": "c2ln"}]}
        >>> env = envelope_from_dict(d)
        >>> env.payload
        'cA=='
    """
    return DSSEEnvelope(
        payload_type=data["payloadType"],
        payload=data["payload"],
        signatures=[
            DSSESignature(keyid=s["keyid"], sig=s["sig"])
            for s in data.get("signatures", [])
        ],
    )


def passport_from_dsse(envelope: DSSEEnvelope) -> ModelPassport:
    """Extract and deserialise a ``ModelPassport`` from a :class:`DSSEEnvelope`.

    The runtime import avoids a circular dependency between ``sigstore_signing``
    and ``passport``.

    Args:
        envelope: A :class:`DSSEEnvelope` whose payload is a base64url-encoded
            passport JSON string.

    Returns:
        The :class:`~provenir.governance.passport.ModelPassport` encoded in the envelope.

    Example:
        >>> from provenir.governance.bom import (
        ...     CodeComponent, DataComponent, EvalComponent, ModelBOM)
        >>> from provenir.governance.passport import ModelPassport
        >>> bom = ModelBOM(
        ...     model_id="m1", base_model="base", run_id="r1",
        ...     data=[DataComponent(name="d", content_hash="h", num_records=1)],
        ...     code=CodeComponent(git_sha="s", dependencies_hash="dh", framework="trl"),
        ...     evals=[EvalComponent(benchmark="mmlu", score=0.5)],
        ...     hyperparameters={},
        ... )
        >>> passport = ModelPassport(bom=bom, attestation=None)
        >>> env = sign_dsse(passport, b"secret")
        >>> recovered = passport_from_dsse(env)
        >>> recovered.bom.model_id
        'm1'
    """
    from provenir.governance.passport import ModelPassport  # noqa: PLC0415,I001

    payload_json = base64.urlsafe_b64decode(envelope.payload.encode()).decode("utf-8")
    return ModelPassport.from_dict(json.loads(payload_json))
