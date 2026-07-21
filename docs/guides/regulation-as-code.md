# Regulation-as-Code: EU AI Act Art. 53 / Annex IV Evidence Generator

> **Provenir v0.7** — builds on v0.6 BOM/Passport + audit log.

## Why this matters

EU AI Act General-Purpose AI (GPAI) obligations are **in force since 2025-08-02**;
high-risk obligations land **2026-08-02**.  Two artifacts are mandatory but historically
unproduced:

1. **Art. 53 / Annex XII training-data summary** — the AI Office published a mandatory
   template (2025-07-24) requiring, for scraped data, *"the top 10% of internet domains
   by volume crawled (5% for SMEs)"* — granular provenance that was never standard
   practice.

2. **Art. 11 / Annex IV technical file** — model cards cover only ~30-40%; the remaining
   ~60-70% (risk management, change docs, foreseeable-misuse, oversight design) is
   unproduced, and the **contemporaneity requirement means it cannot be reconstructed
   retroactively.**

Provenir already captures the evidence (BOM: data provenance, code, evals,
hyperparameters, timestamps; audit log; signed passport).  The move: **turn provenance
we already hold into the exact artifacts the law demands — and sign them so they are
contemporaneous and tamper-evident.**

---

## Legal references

| Document | Reference | Notes |
|---|---|---|
| EU AI Act | Regulation (EU) 2024/1689 | In force 2024-08-01 |
| GPAI obligations | Art. 53 | In force 2025-08-02 |
| High-risk obligations | Art. 11 + Annex IV | Apply 2026-08-02 |
| Training-data summary | Annex XII (via Art. 53(1)(d)) | AI Office template 2025-07-24 |
| Technical file | Annex IV §1-9 | Cannot be reconstructed retroactively |

---

## Provenir data model extensions

Three additive, defaulted fields were added to `DataComponent` in v0.7 — zero breaking
changes; existing passports still validate:

```python
from provenir.governance.bom import DataComponent

# Old construction still works:
dc_old = DataComponent(name="d", content_hash="h", num_records=100)
assert dc_old.source_category == "unknown"   # default
assert dc_old.crawl_domains == {}            # default
assert dc_old.optout_respected is None       # default

# New construction with Art. 53 provenance:
dc_new = DataComponent(
    name="web-crawl",
    content_hash="sha256:...",
    num_records=500_000,
    license="cc-by-4.0",
    pii_scanned=True,
    contamination_checked=True,
    source_category="scraped",          # NEW: public/licensed/scraped/synthetic/user
    crawl_domains={                     # NEW: {domain: record_count}
        "wikipedia.org": 200_000,
        "arxiv.org": 120_000,
        "github.com":  80_000,
        # ... more domains
    },
    optout_respected=True,              # NEW: was robots.txt/TDM opt-out respected?
)
```

---

## Generating evidence

### Python API

```python
from provenir.governance.passport import PassportSigner
from provenir.governance.regulation import RegulationGenerator

# 1. Sign the passport (establishes contemporaneous tamper-evident record)
passport = PassportSigner(b"your-signing-key").sign(bom, signed_at="2026-07-01T00:00:00Z")

# 2. Generate Art. 53 training-data summary
gen = RegulationGenerator()                 # sme=True for 5% threshold
art53 = gen.art53_training_data_summary(passport)
print(art53.summary())
print(art53.markdown)                       # ready to submit to AI Office

# 3. Generate Annex IV technical file
annexiv = gen.annex_iv_technical_file(passport)
print(annexiv.summary())
# SATISFIED: §2(c) data provenance, §2(d) validation/testing
# MISSING with notes: §4 risk management, §7-9 post-market/oversight/conformity
```

### CLI

```bash
# Art. 53 training-data summary
provenir regulation art53 \
  --passport passport.json \
  --out art53_summary.md

# SME mode (5% domain threshold)
provenir regulation art53 \
  --passport passport.json \
  --sme \
  --out art53_summary_sme.md

# Annex IV technical file
provenir regulation annex-iv \
  --passport passport.json \
  --out annex_iv.md

# CI gate: fail if coverage below 40%
provenir regulation annex-iv \
  --passport passport.json \
  --fail-under 0.4
```

---

## Coverage report

Every `EvidenceReport` carries a per-field coverage verdict:

| Status | Meaning |
|---|---|
| `SATISFIED` | Derived from the BOM — evidence present |
| `PARTIAL` | Some data present but incomplete |
| `MISSING` | Cannot be derived; actionable note explains what to supply |

```python
report = gen.annex_iv_technical_file(passport)
print(f"Score: {report.coverage_score():.1%}")
for fc in report.missing():
    print(f"  [{fc.field_id}] {fc.note}")
```

### Typical Annex IV coverage from a signed Provenir passport

| Section | Auto-filled from | Status |
|---|---|---|
| §1 General description | `model_id`, `base_model` | PARTIAL |
| §2(a) Development methods | `code` (framework, git SHA, deps) | PARTIAL |
| §2(b) Design specifications | `hyperparameters` | PARTIAL |
| **§2(c) Data provenance** | **`data` components + Art. 53 summary** | **SATISFIED** |
| **§2(d) Validation/testing** | **`evals` + scan + reward-validity** | **SATISFIED** |
| §3 Monitoring/logging | Audit log presence | PARTIAL |
| §4 Risk management | — (organisational) | MISSING |
| §5 Changes over lifecycle | Signed passport timestamp | PARTIAL |
| §6 Standards applied | Static (CycloneDX, ISO 42001) | PARTIAL |
| §7 Post-market monitoring | — (organisational) | MISSING |
| §8 Human oversight design | — (organisational) | MISSING |
| §9 Conformity assessment | — (organisational) | MISSING |

Sections 4, 7, 8, 9 are intentionally MISSING — they require organisational input that
cannot be derived from the BOM alone.  **The tool generates evidence and marks gaps
honestly; it does not assert legal compliance.**

---

## Hard honesty constraint

Provenir **never fabricates a value**.  When crawl-domain data is absent the summary
says exactly:

```
domain breakdown unavailable — supply crawl manifest
```

Fabricated compliance evidence is worse than a marked gap.

---

## Structured output (OSCAL-ready)

`EvidenceReport.data` is a structured dict designed as the intermediate model for a
future OSCAL serialiser:

```python
import json
report = gen.art53_training_data_summary(passport)
print(json.dumps(report.to_dict(), indent=2))
# {
#   "artifact": "art53_training_data_summary",
#   "coverage_score": 0.625,
#   "fields": [...],
#   "data": {
#     "domain_rollup": { "available": true, "domains": {...}, ... },
#     ...
#   }
# }
```

---

## See also

- `examples/art53_evidence_demo.py` — full demo with rich BOM
- `src/provenir/governance/regulation.py` — implementation
- `src/provenir/governance/bom.py` — `DataComponent` with Art. 53 fields
- `docs/guides/governance.md` — audit log and passport guide
- [EU AI Office — GPAI Code of Practice](https://digital-strategy.ec.europa.eu/en/policies/ai-office)
- [EU AI Act text — EUR-Lex](https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=OJ:L_202401689)
