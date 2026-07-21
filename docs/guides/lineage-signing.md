# LoRA / Fine-tune Lineage Signing

## The problem: no traceability for fine-tune chains

A production model is never directly a base model.  The real chain looks like:

```
Llama3-70B → SFT → RLHF → domain-finetune → LoRA-adapter → production
```

Each arrow is a separate fine-tuning pass, with its own dataset, hyperparameters,
and code.  When a production model misbehaves — generates harmful content,
regresses on a benchmark, fails a regulatory audit — nobody can answer:

- *Which step introduced the regression?*
- *Which training dataset was used at step 3?*
- *Was the SFT checkpoint the same one that was audited?*

Provenir's Lineage Signing module closes this gap.

---

## How `parent_passport_hash` works

Every `ModelBOM` has an optional field:

```python
parent_passport_hash: str | None = None
```

For a base model this is `None`.  For every fine-tuned checkpoint it is the
**SHA-256 content hash of the parent BOM** — the exact `ModelBOM.content_hash()`
of the BOM that was signed into the parent's `ModelPassport`.

Because `parent_passport_hash` is part of `ModelBOM.canonical_json()` (the
signed payload), the chain is **tamper-evident**:

- Modify any ancestor's BOM → its `content_hash()` changes.
- Every descendant's `parent_passport_hash` now points to a hash that no longer
  exists in the known-good passport registry.
- The tamper is immediately detected by `LineageVerifier.verify_hashes()`.

---

## Calling `link_parent`

Use `link_parent` to wire a child BOM to its parent passport before signing:

```python
from provenir.governance.lineage import link_parent
from provenir.governance.passport import PassportSigner

signer = PassportSigner(signing_key, key_id="ci-pipeline")

# 1. Sign the base model
base_passport = signer.sign(base_bom)

# 2. Build the SFT BOM and link it to the base passport
sft_bom = link_parent(sft_bom_draft, base_passport)
# sft_bom.parent_passport_hash == base_passport.bom.content_hash()
sft_passport = signer.sign(sft_bom)

# 3. Build the RLHF BOM and link it to the SFT passport
rlhf_bom = link_parent(rlhf_bom_draft, sft_passport)
rlhf_passport = signer.sign(rlhf_bom)
```

`link_parent` returns a **new** frozen `ModelBOM` (uses `dataclasses.replace`)
so the original draft is never mutated.

---

## Verifying a chain

```python
from provenir.governance.lineage import LineageVerifier

verifier = LineageVerifier()

# Build and validate the chain from an unordered list of passports
chain = verifier.build_chain([base_passport, sft_passport, rlhf_passport])
assert chain.valid, chain.error

# Cross-check that every declared parent hash resolves correctly
assert verifier.verify_hashes([base_passport, sft_passport, rlhf_passport])

# Render the ASCII tree for audits / logs
print(chain.to_ascii_tree())
# [base] llama3-70b-base (run: run-base) hash: a1b2c3d4...
#   └─ [ft:1] sft-llama3 (run: run-sft) hash: e5f6a7b8...
#        └─ [ft:2] rlhf-llama3 (run: run-rlhf) hash: c9d0e1f2...
```

`build_chain` validates:
- Exactly one root (passport with `parent_passport_hash=None`).
- No broken links (every declared parent hash resolves to a passport in the list).
- No cycles.
- No disconnected / forked passports.

---

## Acquisition-readiness value

During due diligence, acquirers routinely demand:

> "Show us the exact training data and code for every step that produced your
> production model."

With Provenir lineage signing you can:

1. Hand the acquirer the ordered chain of `ModelPassport` JSON files.
2. They call `LineageVerifier.build_chain()` and `verify_hashes()` — both pass
   iff the chain is intact and untampered.
3. Each passport's BOM records the exact dataset content hashes, git SHA, and
   framework version for that step.

The chain is cryptographically bound: a corrupt or missing step breaks
`verify_hashes`, making fabrication detectable.

---

## Running the demo

```bash
python examples/lineage_demo.py
```

The demo builds a three-step chain (base → SFT → RLHF), signs each passport,
verifies the chain, prints the ASCII tree, and then demonstrates tamper detection
by mutating the SFT BOM's hyperparameters and re-running `verify_hashes`.

---

## API reference

| Symbol | Description |
|--------|-------------|
| `link_parent(child_bom, parent_passport)` | Wire a child BOM to its parent passport; returns a new frozen BOM. |
| `LineageVerifier.build_chain(passports)` | Build a validated `LineageChain` from an unordered list of passports. |
| `LineageVerifier.verify_hashes(passports)` | Cross-check all `parent_passport_hash` values against actual BOM hashes. |
| `LineageChain.to_ascii_tree()` | Render the chain as a human-readable ASCII tree. |
| `LineageChain.depth()` | Number of fine-tuning steps (0 for a single base model). |
| `LineageChain.root()` | First node (the base model). |
| `LineageChain.tip()` | Last node (the production model). |
| `LineageNode.to_dict()` | Serialisable representation of one chain node. |
