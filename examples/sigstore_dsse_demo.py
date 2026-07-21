"""Sigstore-compatible DSSE Signing Demo.

Demonstrates how to sign a Provenir :class:`~provenir.governance.passport.ModelPassport`
into a DSSE (Dead Simple Signing Envelope), verify it, extract the passport
back from the envelope, and inspect the envelope JSON structure.

DSSE is the envelope format used by Sigstore, in-toto, npm, PyPI, Maven, and
GitHub Actions attestations.  By wrapping Provenir passports in DSSE envelopes
they become natively compatible with the entire supply-chain security ecosystem.

Run::

    python examples/sigstore_dsse_demo.py
"""

from __future__ import annotations

import json

from provenir.governance.bom import (
    CodeComponent,
    DataComponent,
    EvalComponent,
    ModelBOM,
)
from provenir.governance.passport import PassportSigner
from provenir.governance.sigstore_signing import (
    envelope_from_dict,
    passport_from_dsse,
    sign_dsse,
    verify_dsse,
)

__all__: list[str] = []

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

SIGNING_KEY = b"provenir-demo-key-32-bytes-secret"
KEY_ID = "provenir-ci-2026"

SIGNER = PassportSigner(SIGNING_KEY, key_id=KEY_ID)


def _build_passport() -> object:
    bom = ModelBOM(
        model_id="llama3-8b-sft-v1",
        base_model="meta-llama/Meta-Llama-3-8B",
        run_id="run-2026-07-12-abc123",
        data=[
            DataComponent(
                name="openhermes-2.5",
                content_hash="sha256-deadbeefcafe0000",
                num_records=1_000_000,
                license="cc-by-4.0",
                pii_scanned=True,
                contamination_checked=True,
                source_category="instruction-following",
            )
        ],
        code=CodeComponent(
            git_sha="a1b2c3d4e5f6",
            dependencies_hash="sha256-pip-lock-hash",
            framework="trl",
        ),
        evals=[
            EvalComponent(benchmark="mmlu", score=0.71),
            EvalComponent(benchmark="hellaswag", score=0.83),
        ],
        hyperparameters={"lr": 2e-5, "epochs": 3, "batch_size": 64},
        created_at="2026-07-12T00:00:00Z",
    )
    return SIGNER.sign(bom, signed_at="2026-07-12T00:00:00Z")


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


def main() -> None:
    sep = "=" * 68

    print(sep)
    print("Provenir  —  Sigstore-compatible DSSE Signing Demo")
    print(sep)

    # ── Step 1: Build and sign a passport ──────────────────────────────────
    print("\n[1] Building and HMAC-signing a ModelPassport…")
    passport = _build_passport()
    print(f"    model_id : {passport.bom.model_id}")  # type: ignore[union-attr]
    print(f"    bom hash : {passport.bom.content_hash()[:24]}…")  # type: ignore[union-attr]
    print(f"    passport HMAC verifies : {passport.verify(SIGNING_KEY)}")  # type: ignore[union-attr]

    # ── Step 2: Wrap in a DSSE envelope ────────────────────────────────────
    print("\n[2] Wrapping passport in a DSSE envelope (sign_dsse)…")
    envelope = sign_dsse(passport, SIGNING_KEY, key_id=KEY_ID)  # type: ignore[arg-type]
    print(f"    payloadType : {envelope.payload_type}")
    print(f"    payload     : {envelope.payload[:40]}… ({len(envelope.payload)} chars)")
    print(f"    signatures  : {len(envelope.signatures)} entry/entries")
    print(f"    key_id      : {envelope.signatures[0].keyid}")
    print(f"    sig         : {envelope.signatures[0].sig[:32]}…")
    print(f"    content_hash: {envelope.content_hash()[:24]}…")

    # ── Step 3: Verify the DSSE envelope ───────────────────────────────────
    print("\n[3] Verifying the DSSE envelope (verify_dsse)…")
    good = verify_dsse(envelope, SIGNING_KEY)
    bad = verify_dsse(envelope, b"wrong-key")
    print(f"    correct key  → {good}   (expected True)")
    print(f"    wrong key    → {bad}  (expected False)")

    # ── Step 4: Extract passport from envelope ─────────────────────────────
    print("\n[4] Extracting passport from envelope (passport_from_dsse)…")
    recovered = passport_from_dsse(envelope)
    print(f"    model_id recovered : {recovered.bom.model_id}")
    print(f"    attestation present: {recovered.attestation is not None}")
    print(f"    HMAC still valid   : {recovered.verify(SIGNING_KEY)}")

    # ── Step 5: JSON round-trip ────────────────────────────────────────────
    print("\n[5] JSON serialisation and round-trip (to_json / envelope_from_dict)…")
    envelope_json = envelope.to_json()
    restored = envelope_from_dict(json.loads(envelope_json))
    print(f"    round-trip verify  : {verify_dsse(restored, SIGNING_KEY)}")
    print(f"    envelope JSON size : {len(envelope_json):,} bytes")

    # ── Step 6: Show the DSSE JSON structure ───────────────────────────────
    print("\n[6] DSSE envelope JSON structure (truncated payload/sig):")
    d = envelope.to_dict()
    preview = dict(d)
    preview["payload"] = preview["payload"][:40] + "…"
    preview["signatures"] = [
        {k: (v[:32] + "…" if k == "sig" else v) for k, v in s.items()}
        for s in preview["signatures"]
    ]
    print(json.dumps(preview, indent=2))

    print(f"\n{sep}")
    print("Optional: Sigstore / Rekor transparency log integration")
    print(sep)
    print(
        "  Install:  pip install sigstore"
        "\n  Then use: sigstore.sign() with the DSSE payload above to submit"
        "\n  the envelope to the Rekor public log for publicly-verifiable,"
        "\n  key-free attestations compatible with npm, PyPI, and Maven."
    )
    print(sep)


if __name__ == "__main__":
    main()
