# Provenir

[![CI](https://github.com/anilatambharii/provenir/actions/workflows/ci.yml/badge.svg)](https://github.com/anilatambharii/provenir/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/provenir.svg)](https://pypi.org/project/provenir/)
[![Python](https://img.shields.io/pypi/pyversions/provenir.svg)](https://pypi.org/project/provenir/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-1153%20passing-brightgreen.svg)](https://github.com/anilatambharii/provenir/actions)
[![Docs](https://img.shields.io/badge/docs-online-blue.svg)](https://anilatambharii.github.io/provenir)

**The trust layer for model post-training — reproducible, evaluation-first orchestration for LLM fine-tuning and RL.**

Provenir sits above training engines (verl, TRL, Unsloth) and coordinates
every layer of the fine-tuning and reinforcement-learning lifecycle — from raw
data through preference learning, verifiable-reward RL, RAG dataset generation,
model merging, benchmark evaluation, and production serving — in a single,
auditable, reproducible pipeline. It adds the trust primitives that raw
throughput engines leave out: RL observability, reward-hacking detection,
contamination-safe evaluation, deterministic replay, and a signed Model
Passport.

```
pip install provenir
pip install "provenir[train]"          # SFT + DPO + LoRA / QLoRA via TRL
pip install "provenir[all]"            # Everything
```

> **v0.6.0** · Apache-2.0 · Python ≥ 3.11 · zero breaking changes

---

## Why Provenir?

Every major fine-tuning framework optimises for raw throughput.
Provenir optimises for **trust** — the ability to reproduce a run, audit every
decision, catch regressions and reward-hacking before they reach production,
verify rewards that cannot be gamed, and prove exactly what data and code
produced a model.

| | Axolotl | TRL | Unsloth | verl | **Provenir** |
|---|:---:|:---:|:---:|:---:|:---:|
| SFT / DPO / GRPO | ✓ | ✓ | ✓ | ✓ | ✓ |
| LoRA / QLoRA | ✓ | ✓ | ✓ | ✓ | ✓ |
| Reproducible manifests | — | — | — | — | ✓ |
| Eval-in-the-loop | — | partial | — | — | ✓ |
| LLM-as-Judge pipeline | — | — | — | — | ✓ |
| **RLAIF (no human labels)** | — | — | — | — | ✓ |
| Data flywheel (auto-augment) | — | — | — | — | ✓ |
| RAG dataset generation | — | — | — | — | ✓ |
| Adapter merging (SLERP/TIES/DARE) | — | — | — | — | ✓ |
| Standard benchmark suite | — | — | — | — | ✓ |
| Semantic decontamination | — | — | — | — | ✓ |
| REST API server | — | — | — | — | ✓ |
| Full governance & audit log | — | — | — | — | ✓ |
| Plugin architecture | — | — | — | — | ✓ |
| **RL Flight Recorder** | — | — | — | — | ✓ |
| **Reward-hacking detection** | — | — | — | — | ✓ |
| **Verifiable-reward environments (RLVR)** | — | — | — | — | ✓ |
| **Contamination firewall** | — | — | — | — | ✓ |
| **Judge calibration** | — | — | — | — | ✓ |
| **Deterministic replay / lineage DAG** | — | — | — | — | ✓ |
| **Signed Model Passport** | — | — | — | — | ✓ |
| **Loop Doctor (differential diagnosis)** | — | — | — | — | ✓ |
| **Agentic environments (multi-turn tool use)** | — | — | — | — | ✓ |
| **Webhook / Slack alerts** | — | — | — | — | ✓ |
| **Self-contained HTML run reports** | — | — | — | — | ✓ |
| **Passport-signed Hub push** | — | — | — | — | ✓ |

---

## New in v0.3 — The Trust Layer

v0.3.0 turns Provenir into the **trust layer for model post-training**. It
orchestrates the best RL engines (verl, TRL, Unsloth) rather than reimplementing
kernels, and wraps every run with the observability, verification, and
provenance that RL at scale actually needs. Four pillars:

### A. RL Flight Recorder + reward-hacking detection

A **black box for RL runs**. The Flight Recorder (`provenir.observability`)
watches every step and flags KL blowup/collapse, entropy collapse,
response-length explosion, GRPO advantage collapse, reward-std collapse, reward
spikes, and gradient explosion. The **reward-hacking detector** catches the #1
RL failure mode — length inflation, format exploits, test tampering
(`unittest.skip` / `sys.exit(0)` / monkeypatch), verifier gaming, proxy-reward
divergence, degenerate repetition, and advantage collapse. No other OSS RL
framework ships RL-native observability.

### B. Contamination-safe trustworthy eval + judge calibration

A **contamination firewall** (`provenir.eval.contamination`,
`provenir.eval.canary`) detects train/eval overlap via 13-gram, embedding, and
exact matching (with MinHash for scale) and supports **canary-tagged private
eval vaults** that fire if a held-out set leaks into training. **Judge
calibration** (`provenir.eval.judge_calibration`) measures LLM-judge position
bias, self-consistency, and flip-rate, with a `DebiasedJudge` (evaluates both
orderings) and `EnsembleJudge` (majority vote).

### C. Verifiable-reward environments + GRPO/DAPO/GSPO orchestration

A library of **sandboxed, hack-resistant reward functions** for RLVR
(`provenir.environments`): `ExactAnswerVerifier`, `MathVerifier`,
`RegexFormatVerifier`, `JSONSchemaVerifier`, `ToolCallVerifier`,
`ContainsVerifier`, `CompositeVerifier`, and a `CodeVerifier` backed by a
`PythonSandbox` (subprocess isolation + reward-hacking detection) — all behind an
OpenEnv-compatible `Environment` protocol. Provenir's `RLOrchestrator`
(`provenir.train.rl`) runs a real rollout → verify → reward → flight recorder →
hacking detector → eval-gate loop over **GRPO + DAPO + GSPO**, delegating the
gradient step to **backend-agnostic adapters** (`provenir.train.backends.adapters`)
that wrap verl / TRL / Unsloth with capability detection and a `BackendSelector`
that auto-routes by scale tier.

### D. Deterministic replay + lineage DAG + signed Model Passport

**Deterministic replay** (`provenir.provenance`) captures a content-addressed
environment fingerprint, kernel-determinism flags, a lineage DAG
(dataset → run → adapter → eval → merge), and a `ReplayEngine`. The **Model
Passport / BOM** (`provenir.governance.bom`, `provenir.governance.passport`) is
a signed (HMAC-SHA256), portable Bill-of-Materials of exactly what data, code,
evals, and config produced a model — with compliance risk flags
(`unscanned_pii`, `contaminated_eval`, `unknown_license`). Maps directly to
**EU AI Act Article 12** (tamper-proof audit trails + model lineage, enforced
Aug 2, 2026).

### The `import provenir` wrapper — the viral 3-line substrate

Drop provenance, trustworthy eval, reward-hacking detection, and a signed
passport into **any** training loop:

```python
import provenir

with provenir.track("my-run", dataset=train_ds) as run:
    for step, metrics in training_loop():
        run.log_step(metrics)
    run.record_eval("mmlu", score=0.71)

# run.manifest, run.flight_recorder, run.hacking_report, run.passport
```

See the [Trust Layer guide](https://anilatambharii.github.io/provenir/guides/trust-layer/)
for full examples.

---

## New in v0.4 — Loop Doctor + Agentic Environments

### Loop Doctor — differential diagnosis for stalled training loops

When a training loop stops improving, "it's not working" is useless. The Loop Doctor
(`provenir.loop`) does **differential diagnosis** over Provenir's trust signals and
attributes the stall to exactly one cause:

- **eval** — the eval is contaminated; the metrics are lying
- **reward** — the reward is being gamed (reward hacking)
- **algorithm** — the optimiser is unstable; emits a concrete per-anomaly fix
- **data** — the model has plateaued; emits an actionable `DataRequest` (which slices, how many, how recent)

`LoopController` maps the verdict to the next action (`clean_eval` / `fix_reward` /
`stabilize` / `collect_data` / `continue`). New CLI: `provenir diagnose <reward_history...>`.

### Agentic environments — multi-turn, tool-use, verifiable rewards

`provenir.environments.agentic` delivers stateful, **OpenEnv-compatible** multi-turn
environments where the agent emits JSON tool-calls that read/write shared episode state,
then submits a final answer verified by any Provenir `Verifier`.
`EpisodeRunner` + `AgentPolicy` run a policy to a terminal reward; multi-turn
**credit assignment** (`assign_credit`, `CreditConfig`) spreads a sparse terminal reward
across turns (`last_turn` / `uniform` / `discounted`).

---

## New in v0.5 — Observability, Alerts, and Passport Hub Push

### Webhook / Slack alerts (`provenir.alerts`)

```python
with provenir.track(
    "my-run",
    alert_webhook_url="https://hooks.slack.com/...",
    alert_on_anomaly=True,
    alert_on_hacking=True,
) as run:
    ...
```

`AlertConfig` + `Alerter` fire JSON POST payloads to any HTTP endpoint using **stdlib
`urllib.request` only** — no extra dependencies. All network errors are caught so
alerting can never crash a training run.

### Self-contained HTML run reports (`provenir.report`)

```bash
provenir report ./my-run-dir --output report.html
```

`RunReport.from_run_dir(path)` reads the JSON artifacts written by any `ProvenirRun`
and produces a **self-contained HTML report** — health badge, eval table, reward-hacking
signals by category, lineage nodes, and full flight-recorder summary.

### Passport-signed Hub push

```bash
provenir hub push ./my-adapter --repo-id myorg/llama3 --passport passport.json
```

Writes `provenir_passport.json` + `provenir_passport.md` alongside the adapter before
uploading so the signed attestation travels with every model on HuggingFace Hub.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                      provenir CLI / REST API                      │
└──────────────┬───────────────────────────────────┬───────────────┘
               │                                   │
   ┌───────────▼──────────┐           ┌────────────▼──────────────┐
   │      Orchestration    │           │       Data Pipeline        │
   │  ┌─────────────────┐ │           │  ┌──────────────────┐   │
   │  │  RunManifest     │ │           │  │  JsonlDataset      │   │
   │  │  (content-addr.) │ │           │  │  QualityScorer     │   │
   │  ├─────────────────┤ │           │  │  CurriculumSampler │   │
   │  │  RLAIFPipeline  │ │           │  │  RAGDataGenerator  │   │
   │  │  RLOrchestrator │ │           │  │  DataFlywheel      │   │
   │  │  DataFlywheel   │ │           │  │  PromptTemplates   │   │
   │  ├─────────────────┤ │           │  └──────────────────┘   │
   │  │  HyperparmSweep │ │           └───────────────────────────┘
   │  └─────────────────┘ │
   └───────────┬──────────┘
               │
   ┌───────────▼──────────────────────────────────────────────────┐
   │                 Trust Layer (new in v0.3)                      │
   │  RL Flight Recorder · Reward-hacking detector                 │
   │  Verifiable-reward environments (RLVR) · Contamination firewall│
   │  Canary vaults · Judge calibration · Deterministic replay      │
   │  Lineage DAG · Signed Model Passport / BOM                     │
   └───────────┬──────────────────────────────────────────────────┘
               │
   ┌───────────▼──────────────────────────────────────────────────┐
   │                       Training Backends                        │
   │   TRLBackend (SFT · DPO · LoRA · QLoRA)   StubBackend        │
   │   BackendSelector → verl · TRL · Unsloth adapters            │
   │   DistributedConfig (FSDP · DeepSpeed · DDP)                 │
   └───────────┬──────────────────────────────────────────────────┘
               │
   ┌───────────▼──────────────────────────────────────────────────┐
   │                      Evaluation Layer                          │
   │  ExactMatch · TokenF1 · BLEU-4 · ROUGE-L · Wilson CI         │
   │  RAGEvaluator (faithfulness · precision · relevance)          │
   │  BenchmarkEvaluator (MMLU · HellaSwag · ARC · GSM8K · …)     │
   │  LLM-as-Judge (StubJudge · CachedJudge · Anthropic · OpenAI) │
   │  EvalCallback (early stopping · regression gate)              │
   └───────────┬──────────────────────────────────────────────────┘
               │
   ┌───────────▼──────────────────────────────────────────────────┐
   │                      Governance Layer                          │
   │  AuditLogger · PIIScanner · SecretScanner · ModelCardGen      │
   │  SemanticDecontamination · AdapterRegistry · HubClient        │
   │  Model Passport / BOM (signed) · Lineage DAG                  │
   └──────────────────────────────────────────────────────────────┘
```

---

## Installation

```bash
# Minimal — manifests, eval, governance, CLI
pip install provenir

# Full training stack (SFT + DPO + LoRA/QLoRA via TRL)
pip install "provenir[train]"

# Individual feature groups
pip install "provenir[serve]"            # REST API server
pip install "provenir[hub]"             # HuggingFace Hub push / pull
pip install "provenir[merge]"           # Adapter merging
pip install "provenir[semantic]"        # Semantic decontamination
pip install "provenir[benchmarks]"      # MMLU, HellaSwag, ARC, …
pip install "provenir[judge-anthropic]" # LLM-as-judge via Claude
pip install "provenir[judge-openai]"    # LLM-as-judge via GPT-4o

# Everything
pip install "provenir[all]"
```

---

## Quick Start

### 1 — Train from a YAML config

```yaml
# config.yaml
name: my-llama3-run
backend: trl
model_name_or_path: meta-llama/Llama-3.2-1B
seed: 42
max_steps: 500
batch_size: 4
peft:
  rank: 16
  alpha: 32
  target_modules: ["q_proj", "v_proj"]
  dropout: 0.05
observability_backend: wandb
observability_project: my-project
```

```bash
provenir train config.yaml --dataset data/train.jsonl
```

Every run produces a **content-addressed manifest** — a tamper-evident record
of the exact config hash, dataset hash, git SHA, and provenance chain.

### 2 — Evaluate predictions

```bash
provenir eval predictions.jsonl --dataset data/eval.jsonl --metrics all
```

Outputs `exact_match`, `token_f1`, `bleu4`, `rouge_l`, plus Wilson 95%
confidence intervals and a regression gate vs. the baseline.

### 3 — Verifiable-reward RL with a flight recorder

```bash
provenir rl --dataset data/train.jsonl \
  --algorithm grpo --verifier math
```

Runs the `RLOrchestrator` loop (rollout → verify → reward → flight recorder →
hacking detector → eval gate) with a hack-resistant verifiable reward and full
RL observability. `--algorithm` accepts `grpo | dapo | gspo`; `--verifier`
accepts `exact_answer | math | contains`.

### 4 — Wrap any training run with `import provenir`

```python
import provenir

with provenir.track("my-run", dataset=train_ds) as run:
    for step, metrics in training_loop():
        run.log_step(metrics)
    run.record_eval("mmlu", score=0.71)

# run.manifest, run.flight_recorder, run.hacking_report, run.passport
```

### 5 — Run the RLAIF pipeline (no human labels needed)

```python
from provenir.train.rlaif import RLAIFConfig, RLAIFPipeline
from provenir.eval.judge import AnthropicJudge
from provenir.train.backends.trl import TRLBackend
from provenir.core.config import RunConfig
from provenir.data.dataset import JsonlDataset

pipeline = RLAIFPipeline(
    judge=AnthropicJudge(model="claude-haiku-4-5-20251001"),
    backend=TRLBackend(),
    base_config=RunConfig(model_name_or_path="meta-llama/Llama-3.2-1B"),
    rlaif_config=RLAIFConfig(n_iterations=3, responses_per_prompt=4),
)

train_ds = JsonlDataset.from_jsonl("data/train.jsonl")
eval_ds  = JsonlDataset.from_jsonl("data/eval.jsonl")

for it in pipeline.run(train_ds, eval_ds):
    print(f"Iteration {it.iteration}: {it.preference_count} preferences, "
          f"eval={it.eval_result}")
```

### 6 — Generate a RAG training dataset from documents

```python
from provenir.data.rag_generator import RAGDataGenerator, RAGGeneratorConfig
from provenir.eval.judge import OpenAIJudge
from pathlib import Path

generator = RAGDataGenerator(
    config=RAGGeneratorConfig(questions_per_chunk=5, quality_filter_threshold=0.6),
    judge=OpenAIJudge(model="gpt-4o-mini"),
)

docs = [open(p).read() for p in Path("docs/").glob("*.txt")]
dataset = generator.generate_from_documents(docs)
dataset.save("data/rag_train.jsonl")
print(f"Generated {len(dataset.records)} Q&A pairs")
```

### 7 — Data flywheel: auto-augment from eval failures

```python
from provenir.data.flywheel import DataFlywheel, FlywheelConfig
from provenir.eval.judge import AnthropicJudge

flywheel = DataFlywheel(
    config=FlywheelConfig(
        min_score_threshold=0.7,       # mine failures below this
        max_variants_per_failure=3,    # generate N augmented variants
        quality_filter_threshold=0.5,  # keep only quality variants
    ),
    judge=AnthropicJudge(),
)

augmented_dataset = flywheel.run(train_dataset, eval_dataset)
```

### 8 — Merge LoRA adapters

```bash
provenir merge adapter_v1/ adapter_v2/ --strategy slerp --output merged/
```

Or via the Python API:

```python
from provenir.adapters.merging import MergeConfig, ModelMerger
from pathlib import Path

merger = ModelMerger()
result = merger.merge(
    [Path("adapter_v1"), Path("adapter_v2"), Path("adapter_v3")],
    config=MergeConfig(strategy="ties", density=0.5),
    output_dir=Path("merged_adapter"),
)
print(f"Merged to {result.output_path} using {result.strategy}")
```

### 9 — Run standard benchmarks

```bash
provenir benchmark --model-path ./my-adapter --benchmarks mmlu hellaswag arc_easy
```

```python
from provenir.eval.benchmarks import BenchmarkConfig, BenchmarkEvaluator

evaluator = BenchmarkEvaluator()
results = evaluator.run_suite("./my-adapter", [
    BenchmarkConfig(benchmark="mmlu", num_fewshot=5),
    BenchmarkConfig(benchmark="hellaswag", num_fewshot=10),
    BenchmarkConfig(benchmark="gsm8k", num_fewshot=8),
])
for r in results:
    print(f"{r.benchmark}: {r.score:.3f} ({r.num_examples} examples)")
```

### 10 — Push to HuggingFace Hub

```bash
provenir hub push ./my-adapter --repo-id myorg/llama3-finetuned --private
```

### 11 — Start the REST API server

```bash
provenir serve --host 0.0.0.0 --port 8000
```

```
GET  /health              → {"status": "ok"}
POST /jobs/train          → Submit training job, returns manifest
GET  /manifests           → List all run IDs
GET  /manifests/{run_id}  → Retrieve manifest by ID
POST /eval                → Run evaluation on predictions
GET  /adapters            → List registered adapters
GET  /audit               → Full audit log (JSONL)
```

---

## Core Concepts

### Manifests — Reproducibility by Default

Every training run produces a `RunManifest`: a content-addressed record that
captures the SHA-256 hash of the config, dataset, git commit, random seed, and
full provenance chain. You can reproduce any run exactly:

```bash
provenir reproduce manifest_abc123.json --output reproduced_run/
```

### Prompt Templates

Six built-in formats, all swappable at runtime:

| Template | Use Case |
|---|---|
| `alpaca` | Alpaca-style instruction tuning |
| `chatml` | ChatML multi-turn (OpenAI format) |
| `llama3` | Llama 3 / Meta's official format |
| `mistral` | Mistral instruction format |
| `phi3` | Microsoft Phi-3 format |
| `raw_completion` | Plain prompt → completion (legacy) |

```python
from provenir.data.templates import TEMPLATE_REGISTRY

text = TEMPLATE_REGISTRY.format("llama3", {
    "system": "You are a helpful assistant.",
    "prompt": "Explain transformers.",
    "response": "Transformers are...",
})
```

### RLAIF Pipeline — The Unique Moat

Provenir is the **only open-source fine-tuning framework** that combines
automated preference generation, DPO training, evaluation-gated iteration,
and regression detection into a single reproducible loop — without requiring
human labellers.

```
┌─────────────────────────────────────────────────────────────┐
│                    RLAIF Iteration Loop                      │
│                                                             │
│  dataset ──► generate N response variants per prompt        │
│              ──► LLM judge: pairwise ranking                │
│              ──► (chosen, rejected) preference pairs        │
│              ──► DPO training                               │
│              ──► automatic evaluation                       │
│              ──► regression gate                            │
│              ──► iterate (up to n_iterations)               │
└─────────────────────────────────────────────────────────────┘
```

Each iteration is fully logged, manifested, and replayable. The pipeline
terminates early if quality regresses beyond the configured threshold.

### Verifiable-Reward RL — RLVR without gaming

For RL with verifiable rewards (RLVR), Provenir provides an OpenEnv-compatible
`Environment` protocol and a library of hack-resistant verifiers. The
`RLOrchestrator` fuses rollout, verification, reward-hacking detection, RL
observability, and an eval gate into one loop over GRPO / DAPO / GSPO:

```python
from provenir.environments import MathVerifier
from provenir.train.rl import RLOrchestrator, RLConfig

orchestrator = RLOrchestrator(
    config=RLConfig(algorithm="grpo"),   # grpo | dapo | gspo
    verifier=MathVerifier(),
)

report = orchestrator.run(train_ds, eval_ds)
# report.flight_recorder → per-step RL anomalies
# report.hacking_report  → reward-hacking findings
```

See the [Trust Layer guide](https://anilatambharii.github.io/provenir/guides/trust-layer/).

### LLM-as-Judge

```python
from provenir.eval.judge import (
    StubJudge,      # deterministic, no API calls — for CI
    CachedJudge,    # SHA-256 disk cache over any judge
    AnthropicJudge, # Claude
    OpenAIJudge,    # GPT-4o
)

# Wrap any judge with disk caching
judge = CachedJudge(AnthropicJudge(model="claude-haiku-4-5-20251001"))
pref = judge.score_pairwise("Explain gravity", response_a, response_b)
# → Preference(preferred='a', confidence=0.87, rationale='...')

scores = judge.score_rubric("Explain gravity", response, criteria=[
    "factual accuracy", "clarity", "conciseness"
])
```

To measure and remove judge bias, wrap judges with the calibration tooling:

```python
from provenir.eval.judge_calibration import DebiasedJudge, EnsembleJudge

# Evaluate both orderings to remove position bias
debiased = DebiasedJudge(AnthropicJudge())

# Majority vote across multiple judges
ensemble = EnsembleJudge([AnthropicJudge(), OpenAIJudge(), StubJudge()])
```

### Evaluation Layer

```python
from provenir.eval.harness import MultiMetricEvaluator

evaluator = MultiMetricEvaluator()
result = evaluator.evaluate(dataset, predictions)
# result.metrics → {exact_match, token_f1, bleu4, rouge_l}
# result.metrics[*].mean, .ci_lower, .ci_upper (Wilson 95% CI)
```

### RL Observability & Reward-Hacking Detection

```python
from provenir.observability import FlightRecorder, RewardHackingDetector

recorder = FlightRecorder()
detector = RewardHackingDetector()

for step, metrics in rl_loop():
    recorder.log_step(metrics)     # KL, entropy, reward std, advantages, …

for anomaly in recorder.anomalies():
    print(anomaly.kind, anomaly.step, anomaly.detail)

report = detector.analyze(rollouts)
if report.is_hacking:
    print("Reward hacking:", report.findings)
```

### Contamination Firewall

Detect and remove training samples that overlap with your evaluation set —
preventing inflated benchmark numbers:

```python
from provenir.eval.contamination import ContaminationChecker

checker = ContaminationChecker()          # 13-gram + embedding + exact, MinHash at scale
report  = checker.check(train_dataset, eval_dataset)
print(f"Overlap: {report.overlap_ratio:.1%} across {report.n_hits} records")
```

Legacy `SemanticDecontaminationChecker` remains available and falls back to
substring matching when `sentence-transformers` is not installed.

### Model Passport / BOM

```python
from provenir.governance.passport import ModelPassport

passport = ModelPassport.build(run.manifest, key="team-signing-key")
passport.save("passport.json")           # signed HMAC-SHA256 Bill-of-Materials

# Later, verify integrity and inspect compliance risk flags
loaded = ModelPassport.load("passport.json")
assert loaded.verify(key="team-signing-key")
print(loaded.risk_flags)  # e.g. ["unscanned_pii", "contaminated_eval", "unknown_license"]
```

### Semantic Decontamination

```python
from provenir.data.quality import SemanticDecontaminationChecker

checker = SemanticDecontaminationChecker(threshold=0.90)
contaminated = checker.check(train_dataset, eval_references)
clean_dataset = train_dataset.filter(lambda r: r not in contaminated)
```

Falls back to substring matching when `sentence-transformers` is not installed.

### Governance

```python
from provenir.governance.audit import AuditLogger
from provenir.governance.pii import PIIScanner, PIIMasker

# Every training event is logged to an append-only JSONL audit trail
audit = AuditLogger("artifacts/audit")
audit.log("training_complete", actor="ci-bot", run_id=manifest.run_id)

# Scan for PII before training
scanner = PIIScanner()
report = scanner.scan(dataset)
if report.has_pii:
    masker = PIIMasker()
    clean_dataset = masker.mask(dataset)
```

---

## CLI Reference

```
provenir train <config.yaml>              Run a training job
provenir eval <predictions.jsonl>         Evaluate predictions
provenir rl --dataset <data.jsonl>        Verifiable-reward RL + flight recorder
                                          (--algorithm grpo|dapo|gspo,
                                           --verifier exact_answer|math|contains)
provenir diagnose <reward_history...>     Loop Doctor: diagnose a stalled loop
                                          (--anomaly, --hacking-rate,
                                           --contamination-rate, --data-age-days)
provenir contamination <train> <eval>     Train/eval overlap check
provenir passport show|verify <p.json>    Inspect / verify a signed Model Passport
provenir audit                            Inspect the audit log
provenir model-card                       Generate a model card
provenir reproduce <manifest.json>        Reproduce a run exactly
provenir sweep <config.yaml>              Hyperparameter sweep
provenir compare <run_a> <run_b>          Side-by-side manifest diff
provenir benchmark --model-path <path>    Run MMLU / HellaSwag / ARC / …
provenir merge <a/> <b/> --strategy slerp Merge LoRA adapters
provenir hub push <adapter/> --repo-id …  Push to HuggingFace Hub
provenir hub pull <repo_id> --output …    Pull from HuggingFace Hub
provenir serve                            Start REST API server
provenir rlaif <config.yaml>              Run RLAIF pipeline
```

---

## Configuration Reference

`RunConfig` (YAML or Python):

```yaml
# Core
name: my-run
backend: trl              # trl | stub
seed: 42
deterministic: true
output_dir: artifacts/

# Model
model_name_or_path: meta-llama/Llama-3.2-1B
max_steps: 1000
batch_size: 8

# PEFT / LoRA
peft:
  rank: 16
  alpha: 32
  target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"]
  dropout: 0.05
  load_in_4bit: false     # QLoRA
  load_in_8bit: false
  use_rslora: false       # rsLoRA scaling

# Distributed
distributed:
  strategy: fsdp          # fsdp | deepspeed | ddp | none
  num_gpus: 8
  num_nodes: 1
  deepspeed_stage: 3

# Observability
observability_backend: wandb    # wandb | mlflow | tensorboard | none
observability_project: provenir
```

---

## Supported Benchmarks

| Benchmark | Category | Shots |
|---|---|---|
| `mmlu` | Knowledge (57 subjects) | 5-shot |
| `hellaswag` | Commonsense reasoning | 10-shot |
| `arc_easy` | Science QA | 0-shot |
| `arc_challenge` | Science QA (hard) | 25-shot |
| `truthfulqa_mc1` | Truthfulness | 0-shot |
| `truthfulqa_mc2` | Truthfulness | 0-shot |
| `winogrande` | Commonsense NLI | 5-shot |
| `gsm8k` | Grade-school math | 8-shot |
| `humaneval` | Code generation | 0-shot |

Powered by [EleutherAI lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).

---

## Adapter Merging Algorithms

| Strategy | Best For | Description |
|---|---|---|
| `slerp` | Two similar-task adapters | Spherical linear interpolation — preserves weight direction, blends magnitude smoothly |
| `ties` | Dissimilar-task adapters | Magnitude trimming + sign election — reduces task interference |
| `dare` | Many adapters | Random drop + rescale (DARE-TIES) — scales to 5+ adapters without magnitude collapse |

---

## REST API

Start the server:

```bash
provenir serve --host 0.0.0.0 --port 8000
# or
pip install "provenir[serve]"
uvicorn provenir.server.app:create_app --factory --port 8000
```

Example requests:

```bash
# Health check
curl http://localhost:8000/health

# Submit a training job
curl -X POST http://localhost:8000/jobs/train \
  -H "Content-Type: application/json" \
  -d '{"config": {"name": "api-run", "backend": "stub"}, "records": [...]}'

# List manifests
curl http://localhost:8000/manifests

# Get audit log
curl http://localhost:8000/audit
```

Interactive docs available at `http://localhost:8000/docs` (Swagger UI).

---

## Optional Dependency Groups

| Group | Install | Unlocks |
|---|---|---|
| `train` | `pip install "provenir[train]"` | SFT, DPO, LoRA, QLoRA via TRL |
| `distributed` | `pip install "provenir[distributed]"` | FSDP, DeepSpeed multi-GPU |
| `serve` | `pip install "provenir[serve]"` | REST API server (FastAPI + uvicorn) |
| `hub` | `pip install "provenir[hub]"` | HuggingFace Hub push / pull |
| `merge` | `pip install "provenir[merge]"` | SLERP / TIES / DARE adapter merging |
| `semantic` | `pip install "provenir[semantic]"` | Embedding-based decontamination |
| `benchmarks` | `pip install "provenir[benchmarks]"` | MMLU, HellaSwag, ARC, GSM8K, … |
| `judge-anthropic` | `pip install "provenir[judge-anthropic]"` | LLM-as-judge via Claude |
| `judge-openai` | `pip install "provenir[judge-openai]"` | LLM-as-judge via GPT-4o |
| `all` | `pip install "provenir[all]"` | Everything |

All optional features degrade gracefully when their packages are absent —
stub implementations run without error so pipelines still work in CPU-only
or sandboxed environments.

---

## Package Layout

```
src/provenir/
├── core/           RunManifest, RunConfig, DistributedConfig
├── data/           JsonlDataset, QualityScorer, CurriculumSampler,
│                   SemanticDecontaminationChecker, PromptTemplates (6 formats),
│                   DataFlywheel, RAGDataGenerator
├── environments/   Verifiable-reward RLVR verifiers + OpenEnv-compatible
│                   Environment protocol (ExactAnswer, Math, RegexFormat,
│                   JSONSchema, ToolCall, Contains, Composite, Code +
│                   PythonSandbox)
├── observability/  RL Flight Recorder (KL / entropy / length / advantage /
│                   reward-std / gradient anomalies), RewardHackingDetector
├── provenance/     Deterministic replay: environment fingerprint,
│                   kernel-determinism flags, lineage DAG, ReplayEngine
├── integrations/   `import provenir` wrapper — provenir.track(...) run context
├── train/          Trainer, TRLBackend (SFT/DPO/LoRA/QLoRA),
│   ├── rl.py       RLOrchestrator + GRPO / DAPO / GSPO configs
│   ├── rl_eval_gate.py  Fused contamination + regression + hacking loop guard
│   ├── backends/   PEFTConfig, TrainingObserver (W&B/MLflow/TensorBoard),
│   │   └── adapters.py  verl / TRL / Unsloth adapters + BackendSelector
│   └── ...         EvalCallback, DPOTrainer, GRPOTrainer, PPOTrainer,
│                   GridSweep, RandomSweep, RLAIFPipeline
├── eval/           ExactMatch, TokenF1, BLEU-4, ROUGE-L, Wilson CI,
│   ├── contamination.py  13-gram / embedding / exact + MinHash firewall
│   ├── canary.py         Canary-tagged private eval vaults
│   ├── judge_calibration.py  DebiasedJudge, EnsembleJudge, bias metrics
│   └── ...         RAGEvaluator, BenchmarkEvaluator,
│                   LLM-as-Judge (Stub/Cached/Anthropic/OpenAI)
├── adapters/       AdapterRegistry, HubClient, ModelMerger (SLERP/TIES/DARE)
├── rewards/        ExactMatch, Format, WeightedSum, Min, Max, Threshold, Clamped
├── governance/     AuditLogger, PIIScanner, PIIMasker, SecretScanner, ModelCardGen
│   ├── bom.py      Model Bill-of-Materials (data + code + evals + config)
│   └── passport.py Signed (HMAC-SHA256) Model Passport + risk flags
├── orchestrate/    CostEstimator
├── plugins/        PluginRegistry
├── server/         FastAPI REST server
└── cli/            CLI commands (incl. rl, contamination, passport)
```

---

## Development

```bash
git clone https://github.com/anilatambharii/provenir
cd provenir
pip install -e ".[dev,all]"

# Quality gate (must pass before any PR)
python -m ruff check .
python -m mypy src
python -m pytest -q
```

The project enforces strict type-checking (`mypy --strict`), ruff linting
(E, F, I rules, line-length 100), and comprehensive test coverage of all
core logic across 1,153 tests.

---

## Project Status

| Component | Status |
|---|---|
| Core manifests + reproducibility | Production-ready |
| Evaluation pipeline | Production-ready |
| PII / governance / audit | Production-ready |
| RL Flight Recorder + reward-hacking detection | Production-ready |
| Contamination firewall + canary vaults | Production-ready |
| Judge calibration | Production-ready |
| Deterministic replay + lineage DAG | Production-ready |
| Signed Model Passport / BOM | Production-ready |
| Loop Doctor (differential diagnosis) | Production-ready |
| Agentic environments (multi-turn tool-use) | Production-ready |
| Webhook / Slack alerts | Production-ready |
| Self-contained HTML run reports | Production-ready |
| Passport-signed Hub push | Production-ready |
| Verifiable-reward environments (RLVR) | Beta |
| RL orchestration (GRPO / DAPO / GSPO) | Beta |
| Backend adapters (verl / TRL / Unsloth) | Beta |
| TRL backend (SFT + DPO + LoRA) | Beta |
| RLAIF pipeline | Beta |
| REST API server | Beta |
| Data flywheel | Beta |
| RAG dataset generation | Beta |
| Adapter merging | Beta |
| HuggingFace Hub integration | Beta |
| Benchmark evaluation suite | Beta |
| Distributed training (FSDP / DeepSpeed) | Experimental |
| Interactive Streamlit dashboard | Available (`streamlit run dashboard/app.py`) |

---

## Roadmap

- ✅ **v0.4** — Loop Doctor · Agentic Environments · Multi-turn credit assignment
- ✅ **v0.5** — Webhook alerts · HTML run reports · Passport-signed Hub push · Interactive dashboard
- **v0.6** — Streaming training events via WebSocket · live flight-recorder dashboard · reward model training
- **v1.0** — Production SLAs · enterprise governance controls · HIPAA/SOC2 audit trail · multi-tenant serving

---

## Design Principles

**1. Reproducibility is a first-class citizen.**
Every run produces a content-addressed manifest. Every manifest can reproduce
the exact run that created it. There is no "it worked on my machine."

**2. Evaluation before deployment.**
The regression gate, EvalCallback, RL eval gate, and RLAIF loop all prevent
degraded models from advancing through the pipeline without explicit override.

**3. Trust over throughput.**
RL observability, reward-hacking detection, verifiable rewards, contamination
firewalls, and signed passports catch the failures that raw-throughput engines
ignore.

**4. Degrade gracefully.**
Every optional dependency — torch, TRL, anthropic, wandb, huggingface_hub —
is conditionally imported. The full orchestration stack, manifest system, and
evaluation pipeline work without a GPU.

**5. Governance is not optional.**
PII scanning, secret detection, audit logging, model card generation, and the
signed Model Passport are built into the core package, not plugins. Every
production deployment should be auditable.

**6. Orchestrate the winners.**
Provenir wraps verl / TRL / Unsloth via backend-agnostic adapters rather than
reimplementing kernels. It adds the trust layer on top of the best engines.

---

## Non-Goals

- Reimplementing training kernels or competing with Unsloth on raw single-GPU throughput
- Forking or replacing the Hugging Face / verl training stacks — Provenir wraps and orchestrates them
- Providing hosted SaaS, billing, or proprietary cloud services

---

## License

Apache 2.0 — see [LICENSE](LICENSE).

---

## Citation

If you use Provenir in research, please cite:

```bibtex
@software{provenir2026,
  author = {Prasad, Anil},
  title  = {Provenir: An Open-Source Trust Layer for Model Post-Training},
  year   = {2026},
  url    = {https://github.com/anilatambharii/provenir},
  note   = {v0.6.0}
}
```
