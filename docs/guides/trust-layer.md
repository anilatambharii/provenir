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

All examples are copy-paste accurate against the v0.3.0 API.

---

## 1. The `import provenir` wrapper

The fastest way to adopt the trust layer is the 3-line substrate. Wrap **any**
existing training loop — verl, TRL, Unsloth, or your own — in a
`provenir.track(...)` context. Provenir attaches provenance, an RL Flight
Recorder, a reward-hacking report, and (optionally) a signed Model Passport
without you changing your trainer.

```python
import provenir

with provenir.track("my-run", dataset=train_ds) as run:
    for step, metrics in enumerate(training_loop()):
        run.log_step(metrics)          # feeds the flight recorder
    run.record_eval("mmlu", score=0.71)

# Everything the trust layer produced is available on `run` after exit:
manifest        = run.manifest          # content-addressed RunManifest
flight_recorder = run.flight_recorder   # FlightRecorder with per-step anomalies
hacking_report  = run.hacking_report    # HackingReport over buffered trajectories
lineage         = run.lineage           # LineageGraph (dataset -> run -> eval)
bom             = run.bom               # ModelBOM (bill of materials)
passport        = run.passport          # signed ModelPassport, or None if unsigned
anomalies       = run.anomalies         # list[Anomaly] the flight recorder caught
```

To also emit a **signed** Model Passport, pass a signing key:

```python
with provenir.track(
    "my-run",
    dataset=train_ds,
    base_model="Qwen2.5-7B",
    sign_passport=True,
    signing_key=b"team-signing-key",
    output_dir="artifacts/my-run",
) as run:
    for step, metrics in enumerate(training_loop()):
        run.log_step(metrics)
        run.log_trajectory({"prediction": last_response, "reward": reward})
    run.record_eval("gsm8k", score=0.71)

assert run.passport is not None
assert run.passport.verify(b"team-signing-key")
```

`metrics` can be any dict of training/RL signals — recognised keys are `step`,
`kl`, `entropy`, `reward_mean`, `reward_std`, `response_length_mean`,
`advantage_std`, `grad_norm`, `learning_rate` (unknown keys are ignored). On
exit, every artifact is written under `output_dir`
(`manifests/<run_id>.json`, `lineage.json`, `flight_recorder.json`,
`hacking_report.json`, `bom.json`, and `passport.md` / `passport.json` when
signed).

---

## 2. The RL Flight Recorder

The Flight Recorder (`provenir.observability`) is a **black box for RL runs**.
It watches every step and flags the pathologies that quietly wreck RL training
long before the final reward curve looks wrong:

- `kl_blowup` / `kl_collapse`
- `entropy_collapse`
- `length_explosion`
- `advantage_collapse` (GRPO group degeneracy)
- `reward_std_collapse`
- `reward_spike`
- `grad_explosion`

### Logging step metrics and reading anomalies

```python
from provenir.observability import FlightRecorder, RLStepMetrics

recorder = FlightRecorder()

for step, batch in enumerate(rl_loop()):
    anomalies = recorder.log_step(RLStepMetrics(
        step=step,
        kl=batch.kl,
        entropy=batch.entropy,
        reward_mean=batch.reward_mean,
        reward_std=batch.reward_std,
        advantage_std=batch.advantage_std,
        response_length_mean=batch.mean_response_length,
        grad_norm=batch.grad_norm,
    ))
    for a in anomalies:                 # anomalies detected on THIS step
        print(f"[step {a.step}] {a.severity} {a.kind}: {a.detail}")

# `anomalies` is also a property holding everything caught so far:
for a in recorder.anomalies:
    print(a.kind, a.detail)

print(recorder.health_report())         # HEALTHY / DEGRADED / CRITICAL summary
summary = recorder.summary()            # counts by kind/severity + final metrics
```

`log_step` accepts an `RLStepMetrics`; the `import provenir` wrapper also lets
you pass a plain dict. Because the recorder only needs per-step scalars, it
works whether the gradient step runs on verl, TRL, Unsloth, or a custom loop.

---

## 3. The reward-hacking detector

Reward hacking is the **#1 RL bottleneck**: models learn to maximise the reward
signal without doing the task — inflating response length, exploiting output
format, or literally editing the tests to make them pass. The detector
(`provenir.observability`) looks for `length_inflation`, `format_exploit`,
`test_tampering` (`unittest.skip`, `sys.exit(0)`, monkeypatching),
`verifier_gaming`, `proxy_divergence` (reward up, real eval flat/down),
`degenerate_repetition`, and `advantage_collapse`.

```python
from provenir.observability import RewardHackingDetector

detector = RewardHackingDetector()

# One trajectory:
signals = detector.detect({
    "prediction": "import sys; sys.exit(0)  # skip failing tests",
    "proxy_reward": 1.0,
    "true_reward": 0.0,
})
print([s.kind for s in signals])        # ['test_tampering', 'proxy_divergence']

# A batch of rollouts -> a HackingReport:
report = detector.detect_batch(rollouts)   # rollouts: list[dict] with prediction/reward
if not report.is_clean:
    for signal in report.signals:
        print(f"  - {signal.kind}: {signal.detail}")
    print(f"hacking rate: {report.hacking_rate:.1%}")
    print(report.by_kind())             # {'test_tampering': 3, 'proxy_divergence': 1}

# Group degeneracy (all rewards equal -> zero advantage):
group_signals = detector.detect_group([0.5, 0.5, 0.5, 0.5])   # advantage_collapse
```

Pair this with the [Model Passport](#6-the-model-passport) so a run that shows
reward hacking is surfaced in its `contaminated_eval` / risk-flag story.

---

## 4. Verifiable-reward environments (RLVR)

For RL with verifiable rewards, Provenir ships a library of **sandboxed,
hack-resistant reward functions** (`provenir.environments`) behind an
OpenEnv-compatible `Environment` protocol. Every verifier exposes
`verify(response, reference) -> VerificationResult` (with `.passed`, `.reward`,
`.detail`, `.metadata`):

| Verifier | Checks |
|---|---|
| `ExactAnswerVerifier` | Exact match against a gold answer (`\boxed{}`, `####`, or raw) |
| `MathVerifier` | Numeric equivalence with tolerance |
| `RegexFormatVerifier` | Output matches a required format |
| `JSONSchemaVerifier` | Output parses as JSON with required keys/types |
| `ToolCallVerifier` | Valid tool call with an allowed tool + dict args |
| `ContainsVerifier` | Required substrings present, forbidden absent |
| `CompositeVerifier` | Weighted combination of verifiers |
| `CodeVerifier` | Runs code in a `PythonSandbox` (subprocess isolation + hacking detection) |

```python
from provenir.environments import (
    MathVerifier,
    RegexFormatVerifier,
    CompositeVerifier,
)

verifier = CompositeVerifier([
    (MathVerifier(), 0.7),
    (RegexFormatVerifier(pattern=r"\\boxed\{.*\}", full_match=False), 0.3),
])

result = verifier.verify(model_output, gold_answer)   # (response, reference)
print(result.passed, result.reward, result.detail)
```

The `CodeVerifier`'s `PythonSandbox` runs candidate solutions in an isolated
subprocess and inspects the generated code, so a model that tries to
`unittest.skip` or `sys.exit(0)` its way to a passing test is flagged
(`metadata["suspected_hacking"]`) rather than rewarded:

```python
from provenir.environments import CodeVerifier

code_verifier = CodeVerifier()
result = code_verifier.verify(candidate_solution, {"test_code": unit_tests})
```

### Orchestrating GRPO / DAPO / GSPO

The `RLOrchestrator` (`provenir.train.rl`) fuses everything into one loop —
rollout -> verify -> reward -> flight recorder -> hacking detector -> eval
gate — and delegates the gradient step to a backend adapter. Pass a `GRPOConfig`,
`DAPOConfig` (decoupled clip + dynamic sampling), or `GSPOConfig` (sequence-level,
stabilises MoE):

```python
from provenir.train.rl import RLOrchestrator
from provenir.train.algorithms import GRPOConfig
from provenir.environments import MathVerifier
from provenir.observability import FlightRecorder, RewardHackingDetector

orchestrator = RLOrchestrator(
    algorithm=GRPOConfig(group_size=8, max_steps=100),   # or DAPOConfig / GSPOConfig
    verifier=MathVerifier(),
    flight_recorder=FlightRecorder(),
    hacking_detector=RewardHackingDetector(),
)

result = orchestrator.run(train_ds, eval_ds)
print(result.steps_completed, result.mean_reward)
print(result.anomaly_count, result.hacking_rate, result.halted)
print(result.flight_summary)            # serialised flight-recorder state
```

Backend selection is automatic: `provenir.train.backends.adapters` wraps
verl / TRL / Unsloth with capability detection and a `BackendSelector` that
routes by scale tier. The `provenir.train.rl_eval_gate.RLEvalGate` fuses
contamination-safety + regression + reward-hacking into one loop guard that
halts a run before it wastes GPU budget.

On the CLI:

```bash
provenir rl --dataset data/train.jsonl --algorithm grpo --verifier math
```

---

## 5. The contamination firewall + canary vaults

Train/eval contamination is the #1 eval-reliability pain: if eval examples leak
into training, your benchmark numbers are inflated and untrustworthy. The
contamination firewall (`provenir.eval.contamination`) detects overlap via
13-gram, embedding, or exact matching, with MinHash for scale.

```python
from provenir.eval.contamination import ContaminationChecker, ContaminationConfig

checker = ContaminationChecker(ContaminationConfig(method="ngram"))
report  = checker.check_datasets(train_dataset, eval_dataset)

print(f"{report.contamination_rate:.1%} contaminated across {len(report.hits)} hits")
for hit in report.hits:
    print(f"  train#{hit.train_index} <-> eval#{hit.eval_index} ({hit.method})")

clean_train = checker.filter_contaminated(train_dataset, report)
```

On the CLI:

```bash
provenir contamination data/train.jsonl data/eval.jsonl
```

### Canary vaults

A canary (`provenir.eval.canary`) is a unique token embedded into a **private
eval set**. If that token later shows up in training data, you know the held-out
set has leaked — the strongest signal that a benchmark result cannot be trusted.

```python
from provenir.eval.canary import CanaryGuard

guard  = CanaryGuard()
canary = guard.mint("mmlu-private")            # deterministic, reproducible token

# Embed the canary into each private-eval record before you distribute it:
tagged_eval = guard.tag(eval_dataset, canary)

# Later, scan any training corpus for the leaked canary:
leaked_rows = guard.scan(train_dataset, canary)
if leaked_rows:
    raise RuntimeError(f"Private eval leaked into training at rows {leaked_rows}")
```

### Judge calibration

When you evaluate with an LLM judge, `provenir.eval.judge_calibration` measures
position bias, self-consistency, and flip-rate, and gives you two debiased
wrappers:

```python
from provenir.eval.judge_calibration import (
    JudgeCalibrator,
    DebiasedJudge,
    EnsembleJudge,
)
from provenir.eval.judge import AnthropicJudge, OpenAIJudge, StubJudge

# Measure how reliable a judge is:
report = JudgeCalibrator(AnthropicJudge()).measure_position_bias(cases)
print(report.position_bias, report.flip_rate, report.is_reliable)

# Evaluate both orderings to remove position bias:
debiased = DebiasedJudge(AnthropicJudge())

# Majority vote across multiple judges:
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

### The easy way — via the wrapper

If you used `provenir.track(..., sign_passport=True, signing_key=...)`, the
signed passport is already on `run.passport` and written to disk. Otherwise,
build one explicitly.

### Build, sign, verify

```python
from provenir.governance.bom import (
    ModelBOM,
    DataComponent,
    CodeComponent,
    EvalComponent,
)
from provenir.governance.passport import PassportSigner, PassportStore

bom = ModelBOM(
    model_id="my-model",
    base_model="Qwen2.5-7B",
    run_id=run.manifest.run_id,
    data=[DataComponent(
        name="train",
        content_hash=run.manifest.dataset_hash,
        num_records=1200,
        license="apache-2.0",
        pii_scanned=True,
        contamination_checked=True,
    )],
    code=CodeComponent(
        git_sha=run.manifest.git_sha,
        dependencies_hash=run.manifest.dependencies_lockfile,
        framework="trl",
    ),
    evals=[EvalComponent(benchmark="gsm8k", score=0.71)],
    hyperparameters={"lr": 1e-5},
)

signer   = PassportSigner(key=b"team-signing-key", key_id="team")
passport = signer.sign(bom)

# Verify the HMAC-SHA256 signature — fails if the BOM was tampered with:
assert passport.verify(b"team-signing-key")

# Persist (also appends an immutable audit-log line):
PassportStore("artifacts/passports").save(passport)
```

On the CLI:

```bash
provenir passport show   passport.json                 # print the passport markdown
provenir passport verify passport.json --key team-signing-key
```

### Risk flags

The BOM derives compliance risk flags from the trust layer, so a downstream
consumer can gate on them:

```python
for flag in passport.bom.risk_flags():
    print(flag)
# unscanned_pii          -> a data component was not PII-scanned
# unchecked_contamination -> a data component was not contamination-checked
# contaminated_eval      -> an eval was flagged as contaminated
# unknown_license        -> a data component has no known license
```

A clean passport (no risk flags, valid signature) is a portable proof that a
model was produced by a trustworthy, reproducible process.

---

## Deterministic replay + lineage DAG

Underpinning the passport is the deterministic replay subsystem
(`provenir.provenance`): a content-addressed environment fingerprint,
kernel-determinism flags, a lineage DAG (dataset -> run -> adapter -> eval ->
merge), and a `ReplayEngine` that verifies whether a run can be reproduced.

```python
from provenir.provenance import (
    ReplayEngine,
    capture_fingerprint,
    kernel_determinism_flags,
)
from provenir.core.manifest import RunManifestStore

engine       = ReplayEngine(RunManifestStore("artifacts/manifests"))
verification = engine.verify(
    run.manifest.run_id,
    current_config_hash=run.manifest.config_hash,
    current_dataset_hash=run.manifest.dataset_hash,
    current_fingerprint=capture_fingerprint(),
)
print(verification.reproducible, verification.differences)

# A full reproducibility recipe (seed, hashes, env flags, git SHA):
recipe = engine.replay_command(run.manifest.run_id)
print(kernel_determinism_flags())        # env vars for bitwise reproducibility
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
