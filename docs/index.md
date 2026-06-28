# Provenir

**Reproducible, evaluation-first orchestration for LLM fine-tuning.**

Provenir coordinates every layer of the fine-tuning lifecycle — raw data ingestion, quality filtering, prompt templating, SFT/DPO/GRPO training, eval-in-the-loop early stopping, LLM-as-judge preference generation, RLAIF iteration, adapter merging, benchmark evaluation, and production serving — in a single auditable, reproducible pipeline.

---

## Why Provenir?

Every major fine-tuning framework optimises for raw throughput.
Provenir optimises for **trust**:

- **Reproducibility** — every run produces a content-addressed `RunManifest` that captures config hash, dataset hash, git SHA, and the full provenance chain. Any run can be reproduced exactly with `provenir reproduce`.
- **Evaluation-gated iteration** — the `EvalCallback` evaluates mid-training and halts or raises an alarm when quality regresses, before the model reaches production.
- **No human labellers required** — the RLAIF pipeline uses an LLM judge to generate preference pairs, run DPO, evaluate, and iterate autonomously.
- **Governance built-in** — PII scanning, secret detection, append-only audit log, and model card generation ship in the core package, not as plugins.

---

## Feature Matrix

| | Axolotl | TRL | Unsloth | **Provenir** |
|---|:---:|:---:|:---:|:---:|
| SFT / DPO / GRPO | ✓ | ✓ | ✓ | ✓ |
| LoRA / QLoRA | ✓ | ✓ | ✓ | ✓ |
| Reproducible manifests | — | — | — | ✓ |
| Eval-in-the-loop | — | partial | — | ✓ |
| LLM-as-Judge | — | — | — | ✓ |
| **RLAIF (no human labels)** | — | — | — | ✓ |
| Data flywheel | — | — | — | ✓ |
| RAG dataset generation | — | — | — | ✓ |
| Adapter merging (SLERP/TIES/DARE) | — | — | — | ✓ |
| Standard benchmark suite | — | — | — | ✓ |
| Semantic decontamination | — | — | — | ✓ |
| REST API | — | — | — | ✓ |
| Governance & audit log | — | — | — | ✓ |

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
- [RLAIF Pipeline](guides/rlaif.md) — AI-feedback loop without human labellers
- [Data Flywheel](guides/flywheel.md) — auto-augment from eval failures
- [RAG Dataset Generation](guides/rag.md) — build RAG training data from documents
- [Adapter Merging](guides/merging.md) — SLERP, TIES, DARE
- [Benchmarks](guides/benchmarks.md) — MMLU, HellaSwag, ARC, GSM8K
- [Governance & Audit](guides/governance.md) — PII, secrets, audit log, model cards
- [API Reference](api.md) — full Python API
