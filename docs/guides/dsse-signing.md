# DSSE Signing — Sigstore-compatible Passport Envelopes

## Why DSSE?

Provenir's built-in `PassportSigner` uses HMAC-SHA256 to produce tamper-evident
passports.  HMAC works well inside a single organisation where all verifiers
hold the shared key, but it breaks down the moment you need **cross-organisational
trust**:

- You cannot verify the signature without the secret key.
- Sharing the key means sharing the ability to forge signatures.
- There is no standard wire format — every consumer needs Provenir-specific
  parsing logic.

**DSSE (Dead Simple Signing Envelope)** solves all three problems.  It is the
envelope format used by:

| Ecosystem | Standard / Tool |
|-----------|----------------|
| Sigstore | [`sigstore/sigstore-python`](https://github.com/sigstore/sigstore-python) |
| in-toto | [ITE-6 / DSSE transport](https://github.com/in-toto/ITE/blob/master/ITE/6/) |
| npm | `npm audit signatures` |
| PyPI | Trusted Publishers + Sigstore |
| Maven / Java | Sigstore for Maven |
| GitHub Actions | OIDC-based artifact attestations |

By wrapping Provenir passports in DSSE envelopes, they plug into every
existing SBOM and supply-chain compliance workflow without additional
integration work.

---

## The envelope format

```json
{
  "payloadType": "application/vnd.provenir.passport+json",
  "payload": "<base64url(canonical_passport_json)>",
  "signatures": [
    {
      "keyid": "<key_id>",
      "sig": "<base64url(HMAC-SHA256(PAE(payloadType, payload)))>"
    }
  ]
}
```

The signature is computed not over the raw payload, but over the
**Pre-Authentication Encoding (PAE)**:

```
PAE(type, body) = "DSSEv1" SP LEN(type) SP type SP LEN(body) SP body
```

PAE binds the payload type into the signed content so a signature over
`application/vnd.provenir.passport+json` cannot be replayed against a
different payload type.

---

## How it differs from HMAC-only signing

| Feature | `PassportSigner` (HMAC) | `sign_dsse` (DSSE) |
|---------|------------------------|-------------------|
| Wire format | Provenir-specific JSON | DSSE — industry standard |
| Interoperability | Provenir only | Any DSSE consumer |
| Key distribution | Required for verify | Required (HMAC variant) or optional (Sigstore/OIDC) |
| Replay protection | Hash-of-BOM | PAE type-binding |
| Transparency log | No | Optional via Rekor |

The two mechanisms are **complementary**, not competing.  You can keep the
existing `PassportSigner` attestation inside the passport and additionally
wrap the whole passport in a DSSE envelope for cross-org distribution.

---

## Quick start

```python
from provenir.governance.passport import PassportSigner
from provenir.governance.sigstore_signing import (
    sign_dsse,
    verify_dsse,
    passport_from_dsse,
    envelope_from_dict,
)
import json

# 1. Build a signed passport the usual way
signer = PassportSigner(b"my-secret-key", key_id="ci")
passport = signer.sign(bom, signed_at="2026-07-12T00:00:00Z")

# 2. Wrap it in a DSSE envelope
envelope = sign_dsse(passport, b"my-secret-key", key_id="ci")

# 3. Serialise and ship
envelope_json = envelope.to_json()

# 4. On the receiving side: verify
received = envelope_from_dict(json.loads(envelope_json))
assert verify_dsse(received, b"my-secret-key")

# 5. Extract the full passport
recovered = passport_from_dsse(received)
print(recovered.bom.model_id)
```

---

## API reference

### `sign_dsse(passport, key, key_id="default") -> DSSEEnvelope`

Sign a `ModelPassport` into a DSSE envelope using HMAC-SHA256.

- `passport` — a `ModelPassport` (signed or unsigned).
- `key` — raw signing key bytes.
- `key_id` — opaque string stored in the `signatures[].keyid` field.

### `verify_dsse(envelope, key) -> bool`

Return `True` if any signature in the envelope verifies against `key`.
Returns `False` for an empty `signatures` list.

### `passport_from_dsse(envelope) -> ModelPassport`

Decode the base64url payload and deserialise back to a `ModelPassport`.

### `envelope_from_dict(data) -> DSSEEnvelope`

Reconstruct a `DSSEEnvelope` from a parsed JSON dict — useful after
`json.loads(envelope_json)`.

### `DSSEEnvelope.content_hash() -> str`

SHA-256 hex digest of the serialised envelope.  Use this as a stable
reference in audit logs so you can prove which exact envelope was issued
at a given point in time.

### `DSSEEnvelope.payload_json() -> str`

Decode the base64url `payload` field back to the canonical passport JSON
string without fully deserialising it.

---

## Optional: Sigstore / Rekor transparency log

The offline HMAC variant in this module works without any network access.
For publicly-verifiable signatures (no key distribution required), the
DSSE envelope payload is directly compatible with Sigstore:

```bash
pip install sigstore
```

```python
# Submit the envelope payload to the Rekor public transparency log.
# The resulting log entry is publicly auditable and does not require
# distributing the signing key to verifiers.
import sigstore
# See the sigstore-python documentation for the full workflow.
```

Rekor log entries are suitable for PyPI, npm, and Maven supply-chain
compliance audits and satisfy the SLSA Level 2+ provenance requirements.

---

## Running the demo

```bash
python examples/sigstore_dsse_demo.py
```

The demo signs a passport, verifies the DSSE envelope, round-trips through
JSON, extracts the passport, and prints the full envelope structure.
