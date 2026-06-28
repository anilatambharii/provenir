# Governance & Audit

Provenir ships governance tooling as a first-class part of the core package —
not an afterthought plugin. Every production fine-tuning deployment should be
auditable, PII-free, and accompanied by a model card.

---

## Audit Log

Every significant event in a Provenir run is written to an append-only JSONL
audit trail.

### Writing Events

```python
from provenir.governance.audit import AuditLogger

audit = AuditLogger("artifacts/audit")

# Log any event with any key-value payload
audit.log("training_start",  actor="ci-bot",  run_id="abc123", config_hash="sha256:...")
audit.log("eval_complete",   actor="ci-bot",  run_id="abc123", exact_match=0.81)
audit.log("model_push",      actor="ci-bot",  run_id="abc123", repo="myorg/llama3-v2")
audit.log("training_abort",  actor="ci-bot",  run_id="abc123", reason="regression")
```

The log file is stored as `artifacts/audit/audit.jsonl` — one JSON object per
line, each with an `event`, `actor`, `timestamp`, and any additional fields.

### Reading Events

```python
events = audit.read()
for event in events:
    print(f"[{event['timestamp']}] {event['event']} by {event['actor']}")
```

### REST API

```bash
# Get full audit log
curl http://localhost:8000/audit
```

### CLI

```bash
provenir audit
# Outputs the raw JSONL audit log to stdout
```

---

## PII Scanning & Masking

Scan training datasets for personally identifiable information before training:

```python
from provenir.governance.pii import PIIScanner, PIIMasker

scanner = PIIScanner()
report  = scanner.scan(dataset)

if report.has_pii:
    print(f"PII detected in fields: {report.pii_fields}")
    print(f"Affected records: {report.record_count}")

    # Mask PII before training
    masker       = PIIMasker()
    clean_dataset = masker.mask(dataset)
    clean_dataset.save("data/train_clean.jsonl")
```

The scanner checks for:

- Email addresses
- Phone numbers
- Social Security Numbers (US format)
- Credit card numbers
- IP addresses

The masker replaces detected PII with typed placeholders (e.g., `[EMAIL]`,
`[PHONE]`, `[SSN]`).

---

## Secret Scanning

Detect accidentally included secrets (API keys, tokens, credentials) in
training datasets:

```python
from provenir.governance.scanners import SecretScanner

scanner = SecretScanner()
report  = scanner.scan(dataset)

if report.has_secrets:
    print(f"Secrets found: {report.secret_types}")
    # Do NOT proceed with training — remove secrets from dataset first
```

Secret patterns detected include:

- Generic high-entropy strings resembling API keys
- AWS access key prefixes (`AKIA…`)
- GitHub tokens (`ghp_…`, `github_pat_…`)
- Common bearer token patterns

---

## Model Card Generation

Generate a standardised model card from a `RunManifest` and eval results:

```python
from provenir.governance.model_card import ModelCardGenerator

generator = ModelCardGenerator()
card      = generator.generate(manifest, eval_result)
card.save("MODEL_CARD.md")
```

Or via CLI:

```bash
provenir model-card \
  --manifest artifacts/manifests/<run_id>.json \
  --eval-results artifacts/eval.json \
  --output MODEL_CARD.md
```

The generated card follows the HuggingFace Model Card format and includes:

- Model description and intended uses
- Training data summary (dataset hash, size)
- Evaluation results (metrics + confidence intervals)
- Training configuration summary
- Known limitations
- License

---

## Recommended Pre-Training Checklist

```python
from provenir.governance.pii import PIIScanner, PIIMasker
from provenir.governance.scanners import SecretScanner
from provenir.data.quality import SemanticDecontaminationChecker

dataset = JsonlDataset.from_jsonl("data/train.jsonl")

# 1. Scan for PII
pii = PIIScanner().scan(dataset)
if pii.has_pii:
    dataset = PIIMasker().mask(dataset)

# 2. Scan for secrets
secrets = SecretScanner().scan(dataset)
if secrets.has_secrets:
    raise ValueError(f"Secrets in training data: {secrets.secret_types}")

# 3. Decontaminate against eval set
eval_ds = JsonlDataset.from_jsonl("data/eval.jsonl")
checker = SemanticDecontaminationChecker(threshold=0.90)
contaminated = checker.check(dataset, [r["prompt"] for r in eval_ds.records])
if contaminated:
    print(f"Removing {len(contaminated)} contaminated records")
    ids = {id(r) for r in contaminated}
    dataset = dataset.filter(lambda r: id(r) not in ids)

# 4. Train (now safe)
trainer.fit(config, dataset)
```

---

## Compliance Notes

The audit log, PII masker, and secret scanner are designed to support
compliance workflows but are **not a substitute for legal review**. For
HIPAA, SOC 2, or GDPR compliance, consult your legal team and consider
whether additional controls are needed beyond what Provenir provides.
