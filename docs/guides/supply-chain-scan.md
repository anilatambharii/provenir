# Supply-Chain Model Scanning

Model files downloaded from public registries are not inherently trustworthy.
HuggingFace's "safe" badge has been bypassed by multiple tools; researchers found
over 350 000 unsafe issues across 51 000 public models; and a new class of attack
— chat-template injection via Jinja2 — executes arbitrary code at *inference time*,
long after any static scan would have been run. Provenir's supply-chain scanner
turns a static artifact check into **portable, signed evidence** embedded in a
Model Passport, so the verdict travels with every promoted model.

---

## Threat Classes

| Class | What is checked | Severity |
|---|---|:---:|
| `pickle_reduce` | Pickle/`.pt`/`.bin`/`.ckpt` files scanned for `REDUCE`, `GLOBAL`, `STACK_GLOBAL` opcodes referencing non-allowlisted modules. Truncated streams are fail-closed. | critical/high |
| `scanner_evasion` | File-extension vs magic-bytes mismatch; ZIP CRC errors; truncated pickle streams (no `STOP` opcode). | high |
| `chat_template_exec` | GGUF metadata and `tokenizer_config.json` chat templates scanned for Jinja2 exec constructs (`{% for/if/set/… %}`, dunder access, `cycler`, `namespace`). | critical |
| `keras_lambda` | Keras `Lambda` layers and custom-object code in `.h5`/`.keras` files detected from raw byte markers and JSON config. | high |
| `weight_format` | Pickle-format weight files flagged (prefer safetensors); safetensors headers validated against the 8-byte length prefix. | medium/high |
| `namespace_pin` | Model references in config files checked for floating bare names (e.g. `"meta-llama/Llama-3"` with no content hash pin). | low |
| `embedded_secret` | Config files and READMEs swept for AWS keys, GitHub PATs, OpenAI keys, HuggingFace tokens, and generic API-key patterns. | critical |

Findings with `critical` or `high` severity set `ScanReport.unsafe = True`, which
blocks promotion via `scan_gate`.

---

## Compliance Context

- **OWASP LLM03** (Training Data Poisoning) and **OWASP LLM05** (Supply Chain
  Vulnerabilities): the scanner directly addresses both by detecting malicious
  payloads in model artifacts before they are loaded or deployed.
- **EU AI Act Article 15** (Accuracy, robustness and cybersecurity): requires
  high-risk AI systems to be resilient against attempts by third parties to exploit
  system vulnerabilities. A signed `ScanComponent` embedded in the Model Passport
  provides the auditable evidence trail.

---

## Python Usage

```python
from provenir.governance.scan import ModelScanner, ScanComponent, scan_gate, ScanBlocked

# Scan a local model directory
scanner = ModelScanner()
report = scanner.scan("/path/to/my-model")

print(report.summary())
# scan '/path/to/my-model': clean [0 findings, 4 files scanned]

for finding in report.findings:
    print(f"[{finding.severity.value.upper()}] {finding.threat.value}: {finding.detail}")

# Gate: raise ScanBlocked if unsafe
try:
    scan_gate(report)
    print("Promotion: ALLOWED")
except ScanBlocked as exc:
    print(f"Promotion: BLOCKED — {exc}")
```

### Embedding the Scan in a Signed Passport

```python
from provenir.governance.bom import ModelBOM, CodeComponent, DataComponent, EvalComponent
from provenir.governance.passport import PassportSigner
from provenir.governance.scan import ModelScanner, ScanComponent

report = ModelScanner().scan("/path/to/my-model")
sc = ScanComponent.from_report(report)           # captures verdict + counts

bom = ModelBOM(
    model_id="my-model-v1",
    base_model="base-llm",
    run_id="run-42",
    data=[DataComponent(name="train", content_hash="sha256:…", num_records=50_000)],
    code=CodeComponent(git_sha="abc123", dependencies_hash="dh", framework="trl"),
    evals=[EvalComponent(benchmark="mmlu", score=0.72)],
    scan=sc,                                      # additive, defaulted to None
)

passport = PassportSigner(b"my-signing-key").sign(bom)
print(passport.verify(b"my-signing-key"))        # True

# The scan verdict is inside the signed envelope:
# changing scan.unsafe would break the signature.
print(bom.risk_flags())
# ['unsafe_model_scan'] if scan found critical/high findings, else []
```

---

## Promotion Gate

`scan_gate` mirrors `reliability.gate_promotion`: it raises `ScanBlocked` if
the report is unsafe. Use `allow_severities` to selectively downgrade severity
levels (e.g. allow medium/low findings in a staging environment while still
blocking on critical/high).

```python
from provenir.governance.scan import Severity, scan_gate, ScanBlocked

# Block on critical and high (default)
scan_gate(report)

# Allow high, still block on critical
scan_gate(report, allow_severities=frozenset({Severity.HIGH}))

# Compose with reliability gate at the same promotion chokepoint:
from provenir.environments.reliability import gate_promotion, PromotionBlocked

try:
    scan_gate(report)
    gate_promotion(reliability_report, "production")
    print("Promotion: ALLOWED")
except (ScanBlocked, PromotionBlocked) as exc:
    print(f"Promotion: BLOCKED — {exc}")
```

---

## CLI Usage

Scan a model directory and print a human-readable report:

```bash
provenir scan /path/to/my-model
```

Write the full JSON report to a file:

```bash
provenir scan /path/to/my-model --json artifacts/scan/report.json
```

Exit code is `0` when the model is clean, `1` when unsafe:

```
scan '/path/to/my-model': UNSAFE (1 critical) [1 findings, 4 files scanned]
  [CRITICAL] embedded_secret: potential aws_access_key found: 'AKIAIOSFODNN7EXAMPL'… (config.json)
Result: UNSAFE — promotion blocked
```

---

## Fail-Closed Design

The scanner never executes or deserialises any model content. It only reads
opcodes and bytes. Any parse error — truncated stream, corrupt ZIP, malformed
safetensors header — is itself a `high`-severity finding that sets `unsafe = True`,
so a deliberately obfuscated file cannot silently pass the gate.
