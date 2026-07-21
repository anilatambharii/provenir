# Acquisition Package Export

When an AI startup is acquired, the acquirer's technical due-diligence team
asks a standard set of questions.  Provenir's **Acquisition Package Export**
answers every one of those questions in seconds — from a signed passport — and
produces a directory of dated, hash-verified artefacts ready to hand to counsel
or a third-party reviewer.

## Why this matters for M&A

Technical due diligence on AI models fails predictably in two ways:

1. **Undisclosed IP exposure** — training data with unclear or restrictive
   licences, or scraped data where opt-out signals were ignored, can become a
   material liability after close.
2. **Unreproducible provenance** — if the acquirer cannot verify that the model
   came from the declared training run, code commit, and data set, they cannot
   price the IP.

Provenir's BOM already captures everything needed; the Acquisition Package
Export assembles it into acquirer-ready artefacts with a single function call.

## What is generated

Call `generate_acquisition_package(passport, output_dir)` and the following
files are written to `output_dir`:

| File | Section | Due-diligence question answered |
|------|---------|--------------------------------|
| `bom-manifest.json` | BOM Manifest | "What exactly went into this model?" |
| `ip-risk.md` | IP Risk | "Are there licence or opt-out risks in the training data?" |
| `security-audit.md` | Security Audit | "Has the model been scanned for supply-chain threats?" |
| `regulatory-exposure.md` | Regulatory Exposure | "What is the EU AI Act, FDA PCCP, and NIST RMF coverage?" |
| `model-reliability.md` | Model Reliability | "Is the reward signal sound? Have evals been contamination-checked?" |
| `reproducibility.md` | Reproducibility | "Can this training run be reproduced from the BOM?" |
| `retraction-risk.md` | Retraction Risk | "Are any training DOIs retracted from the literature?" |
| `lineage.md` | Lineage | "Is this a fine-tune? Where is the parent passport?" |
| `SIGNED-SUMMARY.json` | Signed Summary | "What is the tamper-evident hash of the whole package?" |

The `SIGNED-SUMMARY.json` contains a `package_hash` (SHA-256 of all written
file contents, sorted by filename) so any third party can verify the package
has not been modified after export.

## Quick start

```python
from provenir.governance.acquisition import generate_acquisition_package
from provenir.governance.passport import PassportSigner

# Sign the BOM (you already have a ModelBOM — see governance guide)
passport = PassportSigner(b"your-signing-key").sign(bom)

# Generate the package
pkg = generate_acquisition_package(passport, "/path/to/output")
print(pkg.summary())
```

See `examples/acquisition_demo.py` for a complete working example with all
optional components populated (scan, reward validity, retraction, lineage).

## Acquisition-signalling value

The Acquisition Package is the **single highest-signal artefact** you can
present to an acquirer's technical team:

- It is **automatically generated** from the passport — no manual document
  assembly, no risk of inconsistency between sections.
- The `SIGNED-SUMMARY.json` `package_hash` means the acquirer can verify
  *nothing was changed after generation* — a level of integrity most companies
  cannot provide.
- Every gap is **honestly flagged** as a warning rather than omitted.  An
  acquirer who receives a report with explicit, addressed gaps trusts it more
  than one that looks artificially complete.
- The regulatory coverage scores (EU AI Act Art. 53, FDA PCCP, NIST AI RMF)
  directly de-risk the post-close compliance roadmap.

## Warnings and gaps

If any component is missing (no scan, no reward validity assessment, unsigned
passport), a human-readable warning is included in the package.  The warnings
list also appears in `SIGNED-SUMMARY.json`.

Typical warnings and their remedies:

| Warning | Remedy |
|---------|--------|
| `no scan component` | Run `provenir scan <model_path>` and attach `ScanComponent` |
| `no reward validity assessment` | Run `provenir reward-validity` before production promotion |
| `unsigned passport` | Sign with `PassportSigner` before generating the package |
| `unknown licenses found` | Set `DataComponent.license` on all training datasets |
| `no evaluations recorded` | Add `EvalComponent` entries to the BOM |

## Section details

### BOM Manifest (`bom-manifest.json`)

The raw `passport.to_json()` output — the full, signed BOM in a format that
can be round-tripped with `ModelPassport.from_dict()`.

### IP Risk (`ip-risk.md`)

Flags unknown licences, lists top crawled domains by record volume, and
checks whether opt-out signals (robots.txt / TDM) were respected.

### Security Audit (`security-audit.md`)

Reports the supply-chain scan verdict (scanner version, report hash, finding
counts by severity) and cross-references retraction risk.  Explicitly notes if
no scan has been performed.

### Regulatory Exposure (`regulatory-exposure.md`)

Uses `RegulationGenerator` to compute coverage scores for:

- **EU AI Act Art. 53 / Annex IV** — training-data transparency and technical
  file requirements for GPAI models.
- **FDA PCCP** — Predetermined Change Control Plan for AI/ML software as a
  medical device.
- **NIST AI RMF 1.0** — the four-function GOVERN / MAP / MEASURE / MANAGE
  framework.

Pass `include_regulation=False` to skip this section in offline environments.

### Model Reliability (`model-reliability.md`)

Reports the reward validity score and spurious-reward verdict, plus all
evaluation benchmarks with contamination flags.

### Reproducibility (`reproducibility.md`)

Records run ID, base model, code provenance (git SHA, dependencies hash,
framework), all hyperparameters, data fingerprints, and attestation status —
everything needed to verify or re-run the training.

### Retraction Risk (`retraction-risk.md`)

Reports DOI check statistics from `RetractionMonitor`.  A high risk level here
is a material red flag: the model may have internalised patterns from retracted
scientific papers.

### Lineage (`lineage.md`)

Records the parent passport hash for fine-tuned models so the acquirer can
request and verify the full provenance chain.

### Signed Summary (`SIGNED-SUMMARY.json`)

Machine-readable index of all sections with the `package_hash` for integrity
verification.
