# Verifier Reliability

In Reinforcement Learning from Verifiable Rewards (RLVR), the reward signal
is only as trustworthy as the verifier that produces it. A non-deterministic,
brittle, or gameable verifier silently corrupts the entire training run — the
model learns to exploit the verifier rather than to answer correctly. Provenir's
**verifier-reliability harness** stress-tests any `Verifier` across six
well-defined failure modes before it is ever used in a live RL loop.

---

## The Six Failure Modes

| Mode | What it checks | Weight |
|---|---|:---:|
| **consistency** | Same input → same verdict every time (no hidden RNG / state) | 1× |
| **invariance** | Reward-preserving edits (whitespace, trailing prose) must not flip the verdict | 1× |
| **sensitivity** | A genuinely wrong answer must fail — the verifier can tell right from wrong | 2× |
| **gameability** | Length padding and keyword stuffing must not pass a wrong answer | 2× |
| **monotonicity** | A degraded (truncated) response must never score higher than the full one | 1× |
| **boundary** | Empty / huge / non-UTF-8 / injection / `None` inputs must not crash the verifier | 1× |

Sensitivity and gameability are weighted 2× and are **hard-fail modes**: any
violation sets `ReliabilityReport.hard_fail = True` and blocks promotion to
protected stages regardless of the overall score.

---

## Python Usage

```python
from provenir.environments import (
    ExactAnswerVerifier,
    Probe,
    ReliabilityHarness,
    ReliabilityReport,
    gate_promotion,
    PromotionBlocked,
)

verifier = ExactAnswerVerifier()

probes = [
    Probe(response=r"The answer is \boxed{42}", reference="42", should_pass=True, label="p1"),
    Probe(response=r"\boxed{7}",                reference="7",  should_pass=True, label="p2"),
]

report: ReliabilityReport = ReliabilityHarness().evaluate(verifier, probes)

print(report.summary())
# verifier 'exact_answer': overall=1.000 [ok] (2 probes; ...)

for outcome in report.failures():
    print(f"FAIL [{outcome.mode.value}] {outcome.detail}")
```

The harness is fully deterministic given `seed` (default `0`): two runs on the
same verifier and probe set always produce identical reports.

---

## Promotion Gate

Use `gate_promotion` to block a model from moving to a protected stage when
the verifier is unreliable. Protected stages are `"production"`, `"prod"`, and
`"release"` by default.

```python
from provenir.environments import gate_promotion, PromotionBlocked

try:
    gate_promotion(report, "production")                        # hard-fail always blocks
    gate_promotion(report, "production", min_overall=0.95)      # also enforce a score floor
    print("Promotion: ALLOWED")
except PromotionBlocked as exc:
    print(f"Promotion: BLOCKED — {exc}")
```

`gate_promotion` is a no-op for non-protected stages (e.g. `"staging"`), so
you can call it unconditionally in your CI pipeline.

---

## CLI Usage

Prepare a JSONL file where each line is a labeled probe:

```jsonl
{"response": "The answer is \\boxed{42}", "reference": "42", "should_pass": true, "label": "math-p1"}
{"response": "\\boxed{7}",                "reference": "7",  "should_pass": true, "label": "math-p2"}
```

Run the harness:

```bash
provenir verify-reliability \
    --verifier exact_answer \
    --probes probes.jsonl \
    --stage production \
    --output artifacts/reliability/report.json
```

A verifier that passes all modes exits `0` and prints:

```
verifier 'exact_answer': overall=1.000 [ok] (2 probes; ...)
Promotion to 'production': ALLOWED
```

A blocked verifier exits `1`:

```
verifier 'contains': overall=0.875 [HARD-FAIL] (2 probes; ...)
  FAIL [sensitivity] WRONGLY PASSED a bad response (math-p1)
Promotion to 'production': BLOCKED — verifier 'contains' hard-failed ...
```

Use `--min-overall <float>` to also enforce a minimum overall score:

```bash
provenir verify-reliability \
    --verifier math \
    --probes probes.jsonl \
    --stage production \
    --min-overall 0.95
```

---

## Compliance Context

The verifier-reliability harness directly supports two EU AI Act obligations
for high-risk AI systems:

- **Article 15 (Accuracy and robustness)**: requires that high-risk AI systems
  achieve documented levels of accuracy and robustness against errors, faults,
  and inconsistencies. A `ReliabilityReport` with overall score, per-mode
  scores, and a full outcome log constitutes the required evidence.

- **Article 55(1)(a) (Adversarial testing for GPAI models)**: requires providers
  of general-purpose AI models to carry out documented adversarial testing. The
  harness implements metamorphic and perturbation testing across six dimensions;
  the `--output` JSON artifact provides the audit trail.

The report is designed to embed cleanly inside a
[Model Passport](governance.md) so that adversarial testing results travel
with every promoted artifact.
