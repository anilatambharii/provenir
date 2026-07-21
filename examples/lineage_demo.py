"""LoRA / Fine-tune Lineage Signing demo.

Demonstrates a 3-step fine-tuning chain::

    Llama3-70B (base)
        └─ SFT  (supervised fine-tuning)
             └─ RLHF  (reinforcement learning from human feedback)

Each BOM is linked to its parent via ``parent_passport_hash``, signed, then the
full chain is verified and rendered as an ASCII tree.

Run::

    python examples/lineage_demo.py
"""

from __future__ import annotations

from provenir.governance.bom import (
    CodeComponent,
    DataComponent,
    EvalComponent,
    ModelBOM,
)
from provenir.governance.lineage import LineageVerifier, link_parent
from provenir.governance.passport import PassportSigner

__all__: list[str] = []

SIGNER = PassportSigner(b"provenir-demo-key", key_id="demo")


def _bom(model_id: str, run_id: str, lr: float, score: float) -> ModelBOM:
    return ModelBOM(
        model_id=model_id,
        base_model="llama3-70b",
        run_id=run_id,
        data=[
            DataComponent(
                name="train",
                content_hash=f"hash-{model_id}",
                num_records=50_000,
                license="apache-2.0",
                pii_scanned=True,
                contamination_checked=True,
            )
        ],
        code=CodeComponent(
            git_sha="a1b2c3d4",
            dependencies_hash="pip-lock-sha256",
            framework="trl",
        ),
        evals=[EvalComponent(benchmark="mmlu", score=score)],
        hyperparameters={"lr": lr, "epochs": 3},
        created_at="2026-06-28T00:00:00Z",
    )


def main() -> None:
    print("=" * 64)
    print("Provenir  —  LoRA / Fine-tune Lineage Signing Demo")
    print("=" * 64)

    # Step 1: base model — no parent
    base_bom = _bom("llama3-70b-base", "run-base", lr=0.0, score=0.72)
    base_passport = SIGNER.sign(base_bom)
    print(f"\n[base] model_id : {base_bom.model_id}")
    print(f"       bom hash  : {base_bom.content_hash()[:16]}...")
    print("       parent    : (none)")

    # Step 2: SFT fine-tune — parent is base
    sft_bom_draft = _bom("sft-llama3-70b", "run-sft", lr=2e-5, score=0.78)
    sft_bom = link_parent(sft_bom_draft, base_passport)
    sft_passport = SIGNER.sign(sft_bom)
    print(f"\n[ft:1] model_id : {sft_bom.model_id}")
    print(f"       bom hash  : {sft_bom.content_hash()[:16]}...")
    print(f"       parent    : {sft_bom.parent_passport_hash[:16]}...")  # type: ignore[index]

    # Step 3: RLHF fine-tune — parent is SFT
    rlhf_bom_draft = _bom("rlhf-llama3-70b", "run-rlhf", lr=1e-5, score=0.83)
    rlhf_bom = link_parent(rlhf_bom_draft, sft_passport)
    rlhf_passport = SIGNER.sign(rlhf_bom)
    print(f"\n[ft:2] model_id : {rlhf_bom.model_id}")
    print(f"       bom hash  : {rlhf_bom.content_hash()[:16]}...")
    print(f"       parent    : {rlhf_bom.parent_passport_hash[:16]}...")  # type: ignore[index]

    # Verify the chain
    verifier = LineageVerifier()
    chain = verifier.build_chain([base_passport, sft_passport, rlhf_passport])

    print("\n" + "=" * 64)
    print("Chain verification")
    print("=" * 64)
    print(f"  valid : {chain.valid}")
    print(f"  depth : {chain.depth()} fine-tuning steps")
    ok = verifier.verify_hashes([base_passport, sft_passport, rlhf_passport])
    print(f"  hashes consistent : {ok}")

    print("\n" + "=" * 64)
    print("ASCII lineage tree")
    print("=" * 64)
    print(chain.to_ascii_tree())

    # Show that tampering invalidates the chain
    print("\n" + "=" * 64)
    print("Tamper detection demo")
    print("=" * 64)
    print("Modifying the SFT BOM (changing lr) invalidates the RLHF parent hash:")
    import dataclasses

    tampered_sft_bom = dataclasses.replace(sft_bom, hyperparameters={"lr": 9e-9, "epochs": 3})
    tampered_sft_passport = SIGNER.sign(tampered_sft_bom)
    ok_after_tamper = verifier.verify_hashes(
        [base_passport, tampered_sft_passport, rlhf_passport]
    )
    print(f"  hashes consistent after tamper : {ok_after_tamper}  (expected False)")


if __name__ == "__main__":
    main()
