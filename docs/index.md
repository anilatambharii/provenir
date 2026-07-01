# Provenir

**The trust layer for model post-training — reproducible, evaluation-first orchestration for LLM fine-tuning and RL.**

> **v0.3.0** · Apache-2.0 · Python ≥ 3.11 · 909 tests · zero breaking changes

Provenir coordinates every layer of the fine-tuning and reinforcement-learning lifecycle — raw data ingestion, quality filtering, prompt templating, SFT/DPO/GRPO training, verifiable-reward RL, eval-in-the-loop early stopping, LLM-as-judge preference generation, RLAIF iteration, adapter merging, benchmark evaluation, and production serving — in a single auditable, reproducible pipeline. On top of that it adds the trust primitives that raw-throughput engines leave out: RL observability, reward-hacking detection, contamination-safe evaluation, deterministic replay, and a signed Model Passport.

---

## Why Provenir?

Every major fine-tuning framework optimises for raw throughput.
Provenir optimises for **trust**:

- **Reproducibility** — every run produces a content-addressed `RunManifest` that captures config hash, dataset hash, git SHA, and the full provenance chain. Any run can be reproduced exactly with `provenir reproduce`.
- **Evaluation-gated iteration** — the `EvalCallback` and RL eval gate evaluate mid-training and halt or raise an alarm when quality regresses, before the model reaches production.
- **RL you can trust** — an RL Flight Recorder detects training pathologies, and a reward-hacking detector catches the #1 RL failure mode before it wastes GPU budget.
- **No human labellers required** — the RLAIF pipeline uses an LLM judge to generate preference pairs, run DPO, evaluate, and iterate autonomously.
- **Governance built-in** — PII scanning, secret detection, append-only audit log, model card generation, and a signed Model Passport ship in the core package, not as plugins.

---

## Feature Matrix

| | Axolotl | TRL | Unsloth | verl | **Provenir** |
|---|:---:|:---:|:---:|:---:|:---:|
| SFT / DPO / GRPO | ✓ | ✓ | ✓ | ✓ | ✓ |
| LoRA / QLoRA | ✓ | ✓ | ✓ | ✓ | ✓ |
| Reproducible manifests | — | — | — | — | ✓ |
| Eval-in-the-loop | — | partial | — | — | ✓ |
| LLM-as-Judge | — | — | — | — | ✓ |
| **RLAIF (no human labels)** | — | — | — | — | ✓ |
| Data flywheel | — | — | — | — | ✓ |
| RAG dataset generation | — | — | — | — | ✓ |
| Adapter merging (SLERP/TIES/DARE) | — | — | — | — | ✓ |
| Standard benchmark suite | — | — | — | — | ✓ |
| Semantic decontamination | — | — | — | — | ✓ |
| REST API | — | — | — | — | ✓ |
| Governance & audit log | — | — | — | — | ✓ |
| **RL Flight Recorder** | — | — | — | — | ✓ |
| **Reward-hacking detection** | — | — | — | — | ✓ |
| **Verifiable-reward environments (RLVR)** | — | — | — | — | ✓ |
| **Contamination firewall** | — | — | — | — | ✓ |
| **Judge calibration** | — | — | — | — | ✓ |
| **Deterministic replay / lineage DAG** | — | — | — | — | ✓ |
| **Signed Model Passport** | — | — | — | — | ✓ |

---

## New in v0.3 — The Trust Layer

v0.3.0 turns Provenir into the **trust layer for model post-training**. Four pillars:

- **A. RL Flight Recorder + reward-hacking detection** — a black box for RL runs (`provenir.observability`) flags KL blowup/collapse, entropy collapse, length explosion, advantage collapse, reward-std collapse, reward spikes, and gradient explosion; the reward-hacking detector catches length inflation, format exploits, test tampering, verifier gaming, and proxy-reward divergence.
- **B. Contamination-safe trustworthy eval + judge calibration** — a contamination firewall (`provenir.eval.contamination`, `provenir.eval.canary`) with 13-gram / embedding / exact overlap detection, MinHash at scale, and canary-tagged private eval vaults, plus `DebiasedJudge` and `EnsembleJudge` calibration.
- **C. Verifiable-reward environments + GRPO/DAPO/GSPO orchestration** — hack-resistant RLVR verifiers (`provenir.environments`) and an `RLOrchestrator` (`provenir.train.rl`) that runs rollout → verify → reward → flight recorder → hacking detector → eval-gate over verl / TRL / Unsloth via backend-agnostic adapters.
- **D. Deterministic replay + lineage DAG + signed Model Passport** — content-addressed environment fingerprint, lineage DAG, and `ReplayEngine` (`provenir.provenance`) plus a signed (HMAC-SHA256) Model Passport / BOM (`provenir.governance.passport`) with compliance risk flags. Maps to **EU AI Act Article 12**.

Drop the whole trust layer into **any** training loop with three lines:

```python
import provenir

with provenir.track("my-run", dataset=train_ds) as run:
    for step, metrics in training_loop():
        run.log_step(metrics)
    run.record_eval("mmlu", score=0.71)

# run.manifest, run.flight_recorder, run.hacking_report, run.passport
```

See the [Trust Layer guide](guides/trust-layer.md) for full examples.

---

## Quick Install

```bash
pip install provenir                        # core
pip install "provenir[train]"               # + SFT/DPO/LoRA via TRL
pip install "provenir[all]"                 # everything
```

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                      provenir CLI / REST API                      │
└──────────────┬───────────────────────────────────┬───────────────┘
               │                                   │
   ┌───────────▼──────────┐           ┌────────────▼──────────────┐
   │      Orchestration    │           │       Data Pipeline        │
   │  RunManifest          │           │  JsonlDataset              │
   │  RLAIFPipeline        │           │  QualityScorer             │
   │  DataFlywheel         │           │  CurriculumSampler         │
   │  HyperparamSweep      │           │  RAGDataGenerator          │
   └───────────┬──────────┘           │  PromptTemplates (6)       │
               │                      └───────────────────────────┘
   ┌───────────▼──────────────────────────────────────────────────┐
   │                       Training Backends                        │
   │         TRLBackend (SFT · DPO · LoRA · QLoRA)                │
   │         DistributedConfig (FSDP · DeepSpeed · DDP)           │
   └───────────┬──────────────────────────────────────────────────┘
               │
   ┌───────────▼──────────────────────────────────────────────────┐
   │                      Evaluation Layer                          │
   │  ExactMatch · TokenF1 · BLEU-4 · ROUGE-L · Wilson CI         │
   │  RAGEvaluator · BenchmarkEvaluator (MMLU/HellaSwag/ARC/…)    │
   │  LLM-as-Judge (Stub / Cached / Anthropic / OpenAI)           │
   │  EvalCallback (early stopping · regression gate)              │
   └───────────┬──────────────────────────────────────────────────┘
               │
   ┌───────────▼──────────────────────────────────────────────────┐
   │                      Governance Layer                          │
   │  AuditLogger · PIIScanner · SecretScanner · ModelCardGen      │
   │  SemanticDecontamination · AdapterRegistry · HubClient        │
   └──────────────────────────────────────────────────────────────┘
```

---

## Navigation

- [Quickstart](quickstart.md) — up and running in 5 minutes
- [Core Concepts](concepts.md) — manifests, RLAIF, judge, governance
- [Trust Layer](guides/trust-layer.md) — `import provenir`, flight recorder, reward-hacking, contamination firewall, Model Passport
- [RLAIF Pipeline](guides/rlaif.md) — AI-feedback loop without human labellers
- [Data Flywheel](guides/flywheel.md) — auto-augment from eval failures
- [RAG Dataset Generation](guides/rag.md) — build RAG training data from documents
- [Adapter Merging](guides/merging.md) — SLERP, TIES, DARE
- [Benchmarks](guides/benchmarks.md) — MMLU, HellaSwag, ARC, GSM8K
- [Governance & Audit](guides/governance.md) — PII, secrets, audit log, model cards
- [API Reference](api.md) — full Python API
