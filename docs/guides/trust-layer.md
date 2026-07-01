# Trust Layer

New in **v0.3.0**, the Trust Layer is what turns Provenir from an orchestration
framework into **the trust layer for model post-training**. It adds the
primitives that raw-throughput engines (verl, TRL, Unsloth) leave out:
RL observability, reward-hacking detection, contamination-safe evaluation,
verifiable rewards, deterministic replay, and a signed Model Passport.

This guide covers:

1. The `import provenir` wrapper — drop the trust layer into any run
2. The RL Flight Recorder
3. The reward-hacking detector
4. Verifiable-reward environments (RLVR)
5. The contamination firewall + canary vaults
6. The Model Passport (sign, verify, risk flags)

---

## 1. The `import provenir` wrapper

The fastest way to adopt the trust layer is the 3-line substrate. Wrap **any**
existing training loop — verl, TRL, Unsloth, or your own — in a
`provenir.track(...)` context. Provenir attaches provenance, an RL Flight
Recorder, a reward-hacking report, and a signed Model Passport without you
changing your trainer.

```python
import provenir

with provenir.track("my-run", dataset=train_ds) as run:
    for step, metrics in training_loop():
        run.log_step(metrics)          # feeds the flight recorder
    run.record_eval("mmlu", score=0.71)

# Everything the trust layer produced is available on `run`:
manifest        = run.manifest          # content-addressed RunManifest
flight_recorder = run.flight_recorder   # per-step RL anomalies
hacking_report  = run.hacking_report    # reward-hacking findings
passport        = run.passport          # signed Model Passport / BOM
```

`metrics` can be any dict of training/RL signals (loss, KL, entropy, reward,
reward std, advantages, response length, gradient norm, …). The more you log,
the more the flight recorder and hacking detector can see.

---

## 2. The RL Flight Recorder

The Flight Recorder (`provenir.observability`) is a **black box for RL runs**.
It watches every step and flags the pathologies that quietly wreck RL training
long before the final reward curve looks wrong:

- KL blowup / KL collapse
- Entropy collapse
- Response-length explosion
- GRPO advantage collapse
- Reward-std collapse
- Reward spikes
- Gradient explosion

### Logging step metrics and reading anomalies

```python
from provenir.observability import FlightRecorder

recorder = FlightRecorder()

for step, batch in enumerate(rl_loop()):
    recorder.log_step({
        "step": step,
        "kl": batch.kl,
        "entropy": batch.entropy,
        "reward_mean": batch.reward_mean,
        "reward_std": batch.reward_std,
        "advantage_std": batch.advantage_std,
        "response_length": batch.mean_response_length,
        "grad_norm": batch.grad_norm,
    })

# Inspect what the black box caught
for anomaly in recorder.anomalies():
    print(f"[step {anomaly.step}] {anomaly.kind}: {anomaly.detail}")

# e.g.
# [step 214] entropy_collapse: entropy fell to 0.02 (baseline 1.7)
# [step 217] kl_blowup: KL 48.3 exceeds threshold
```

Because the recorder only needs a dict of per-step signals, it works whether the
gradient step runs on verl, TRL, Unsloth, or a custom loop. When you use the
`import provenir` wrapper or the `RLOrchestrator` (below), the flight recorder
is wired up for you.

---

## 3. The reward-hacking detector

Reward hacking is the **#1 RL bottleneck**: models learn to maximise the reward
signal without doing the task — inflating response length, exploiting output
format, or literally editing the tests to make them pass. The detector
(`provenir.observability`) looks for:

- Length inflation
- Format exploits
- Test tampering (`unittest.skip`, `sys.exit(0)`, monkeypatching)
- Verifier gaming
- Proxy-reward divergence (reward up, real eval flat/down)
- Degenerate repetition
- Advantage collapse

```python
from provenir.observability import RewardHackingDetector

detector = RewardHackingDetector()

report = detector.analyze(rollouts)   # rollouts: prompts, responses, rewards, eval scores

if report.is_hacking:
    print("Reward hacking detected:")
    for finding in report.findings:
        print(f"  - {finding.kind}: {finding.detail}")
    # e.g.
    #   - test_tampering: response calls sys.exit(0) before assertions
    #   - proxy_divergence: reward +18% while held-out eval -3%
```

Pair this with the [Model Passport](#6-the-model-passport) so a run that shows
reward hacking is flagged in its signed Bill-of-Materials.

---

## 4. Verifiable-reward environments (RLVR)

For RL with verifiable rewards, Provenir ships a library of **sandboxed,
hack-resistant reward functions** (`provenir.environments`) behind an
OpenEnv-compatible `Environment` protocol:

| Verifier | Checks |
|---|---|
| `ExactAnswerVerifier` | Exact match against a gold answer |
| `MathVerifier` | Numeric / symbolic math equivalence |
| `RegexFormatVerifier` | Output matches a required format |
| `JSONSchemaVerifier` | Output validates against a JSON schema |
| `ToolCallVerifier` | Correct tool call / arguments |
| `ContainsVerifier` | Output contains required substrings |
| `CompositeVerifier` | Combine multiple verifiers |
| `CodeVerifier` | Runs code in a `PythonSandbox` (subprocess isolation + reward-hacking detection) |

```python
from provenir.environments import MathVerifier, CompositeVerifier, RegexFormatVerifier

verifier = CompositeVerifier([
    MathVerifier(),
    RegexFormatVerifier(pattern=r"\\boxed\{.*\}"),
])

reward = verifier.verify(prompt=problem, response=model_output, target=gold)
```

The `CodeVerifier`'s `PythonSandbox` runs candidate solutions in an isolated
subprocess and runs the reward-hacking detector over the generated code, so a
model that tries to `unittest.skip` or `sys.exit(0)` its way to a passing test
is caught rather than rewarded.

### Orchestrating GRPO / DAPO / GSPO

The `RLOrchestrator` (`provenir.train.rl`) fuses everything into one loop —
rollout → verify → reward → flight recorder → hacking detector → eval gate —
and delegates the gradient step to a backend adapter. It supports GRPO, DAPO
(decoupled clip + dynamic sampling), and GSPO (sequence-level, stabilises MoE):

```python
from provenir.train.rl import RLOrchestrator, RLConfig
from provenir.environments import MathVerifier

orchestrator = RLOrchestrator(
    config=RLConfig(algorithm="grpo"),   # grpo | dapo | gspo
    verifier=MathVerifier(),
)

report = orchestrator.run(train_ds, eval_ds)
print(report.flight_recorder.anomalies())
print(report.hacking_report.findings)
```

Backend selection is automatic: `provenir.train.backends.adapters` wraps
verl / TRL / Unsloth with capability detection and a `BackendSelector` that
routes by scale tier. The `provenir.train.rl_eval_gate` fuses
contamination-safety + regression + reward-hacking into one loop guard that
halts a run before it wastes GPU budget.

On the CLI:

```bash
provenir rl config.yaml --dataset data/train.jsonl \
  --algorithm grpo --verifier math
```

---

## 5. The contamination firewall + canary vaults

Train/eval contamination is the #1 eval-reliability pain: if eval examples leak
into training, your benchmark numbers are inflated and untrustworthy. The
contamination firewall (`provenir.eval.contamination`) detects overlap via
13-gram, embedding, and exact matching, with MinHash for scale.

```python
from provenir.eval.contamination import ContaminationChecker

checker = ContaminationChecker()
report  = checker.check(train_dataset, eval_dataset)

print(f"Overlap: {report.overlap_ratio:.1%} across {report.n_hits} records")
if report.overlap_ratio > 0.0:
    for hit in report.hits:
        print(f"  train#{hit.train_index} ↔ eval#{hit.eval_index} ({hit.method})")
```

On the CLI:

```bash
provenir contamination data/train.jsonl data/eval.jsonl
```

### Canary vaults

A canary vault (`provenir.eval.canary`) is a **private eval set tagged with
canary strings**. If those canaries later show up in training data, you know the
held-out set has leaked — the strongest signal that a benchmark result cannot
be trusted.

```python
from provenir.eval.canary import CanaryVault

# Seal a private eval set with unique canary tags
vault = CanaryVault.create(eval_dataset)
vault.save("vaults/mmlu_private.vault")

# Later, check any training corpus for leaked canaries
leak = vault.scan(train_dataset)
if leak.detected:
    raise RuntimeError(f"Private eval leaked into training: {leak.canaries}")
```

### Judge calibration

When you evaluate with an LLM judge, `provenir.eval.judge_calibration` measures
position bias, self-consistency, and flip-rate, and gives you two debiased
wrappers:

```python
from provenir.eval.judge_calibration import DebiasedJudge, EnsembleJudge
from provenir.eval.judge import AnthropicJudge, OpenAIJudge, StubJudge

# Evaluate both orderings to remove position bias
debiased = DebiasedJudge(AnthropicJudge())

# Majority vote across multiple judges
ensemble = EnsembleJudge([AnthropicJudge(), OpenAIJudge(), StubJudge()])
```

---

## 6. The Model Passport

The Model Passport (`provenir.governance.passport`, built on
`provenir.governance.bom`) is a **signed, portable Bill-of-Materials** of
exactly what data, code, evals, and config produced a model. It is signed with
HMAC-SHA256 so it is tamper-evident, and it carries compliance risk flags. This
is the enterprise/regulatory acquisition wedge and maps directly to **EU AI Act
Article 12** (tamper-proof audit trails + model lineage, enforced Aug 2, 2026).

### Build, sign, save

```python
from provenir.governance.passport import ModelPassport

# Build a signed passport from a run manifest (and its trust-layer artifacts)
passport = ModelPassport.build(run.manifest, key="team-signing-key")
passport.save("passport.json")
```

### Verify

```python
from provenir.governance.passport import ModelPassport

loaded = ModelPassport.load("passport.json")

# Verify the HMAC-SHA256 signature — fails if the BOM was tampered with
assert loaded.verify(key="team-signing-key")
```

On the CLI:

```bash
provenir passport show   passport.json    # print the Bill-of-Materials
provenir passport verify passport.json    # verify the signature
```

### Risk flags

Every passport carries compliance risk flags derived from the trust layer, so a
downstream consumer can gate on them:

```python
for flag in loaded.risk_flags:
    print(flag)
# unscanned_pii      → dataset was not PII-scanned
# contaminated_eval  → contamination firewall found train/eval overlap
# unknown_license    → a data or model dependency has no known license
```

A clean passport (no risk flags, valid signature, no reward hacking) is a
portable proof that a model was produced by a trustworthy, reproducible process.

---

## Deterministic replay + lineage DAG

Underpinning the passport is the deterministic replay subsystem
(`provenir.provenance`): a content-addressed environment fingerprint,
kernel-determinism flags, a lineage DAG (dataset → run → adapter → eval →
merge), and a `ReplayEngine` that can re-execute a run from its fingerprint.

```python
from provenir.provenance import ReplayEngine

engine = ReplayEngine()
replay = engine.replay(run.manifest)   # re-runs from the captured fingerprint
print(replay.matches_original)          # True if deterministic
```

The lineage DAG lets you trace any artifact back to the exact dataset, code,
and config that produced it — the model-lineage requirement behind modern AI
governance regimes.

---

## Putting it together

The trust layer is designed to compose: use verifiable-reward environments to
get a reward you cannot game, run the `RLOrchestrator` so the flight recorder,
hacking detector, and eval gate watch the whole loop, decontaminate against
your eval sets and canary-tag the private ones, then emit a signed Model
Passport as the portable, tamper-evident proof of what produced the model. Or,
for an existing loop, get all of it in three lines with
[`import provenir`](#1-the-import-provenir-wrapper).
