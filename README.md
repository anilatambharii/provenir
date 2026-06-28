# Provenir

[![CI](https://github.com/anilatambharii/provenir/actions/workflows/ci.yml/badge.svg)](https://github.com/anilatambharii/provenir/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/provenir.svg)](https://pypi.org/project/provenir/)
[![Python](https://img.shields.io/pypi/pyversions/provenir.svg)](https://pypi.org/project/provenir/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-456%20passing-brightgreen.svg)](https://github.com/anilatambharii/provenir/actions)
[![Docs](https://img.shields.io/badge/docs-online-blue.svg)](https://anilatambharii.github.io/provenir)

**Reproducible, evaluation-first orchestration for LLM fine-tuning.**

Provenir sits above training engines (TRL, Unsloth, TorchTune) and coordinates
every layer of the fine-tuning lifecycle â€” from raw data through preference
learning, RAG dataset generation, model merging, benchmark evaluation, and
production serving â€” in a single, auditable, reproducible pipeline.

```
pip install provenir
pip install "provenir[train]"          # SFT + DPO + LoRA / QLoRA via TRL
pip install "provenir[all]"            # Everything
```

> **v0.2.0** Â· Apache-2.0 Â· Python â‰¥ 3.11 Â· 456 tests Â· zero breaking changes

---

## Why Provenir?

Every major fine-tuning framework optimises for raw throughput.
Provenir optimises for **trust** â€” the ability to reproduce a run, audit every
decision, catch regressions before they reach production, and iterate with
AI-generated feedback instead of human labellers.

| | Axolotl | TRL | Unsloth | **Provenir** |
|---|:---:|:---:|:---:|:---:|
| SFT / DPO / GRPO | âœ“ | âœ“ | âœ“ | âœ“ |
| LoRA / QLoRA | âœ“ | âœ“ | âœ“ | âœ“ |
| Reproducible manifests | â€” | â€” | â€” | âœ“ |
| Eval-in-the-loop | â€” | partial | â€” | âœ“ |
| LLM-as-Judge pipeline | â€” | â€” | â€” | âœ“ |
| **RLAIF (no human labels)** | â€” | â€” | â€” | âœ“ |
| Data flywheel (auto-augment) | â€” | â€” | â€” | âœ“ |
| RAG dataset generation | â€” | â€” | â€” | âœ“ |
| Adapter merging (SLERP/TIES/DARE) | â€” | â€” | â€” | âœ“ |
| Standard benchmark suite | â€” | â€” | â€” | âœ“ |
| Semantic decontamination | â€” | â€” | â€” | âœ“ |
| REST API server | â€” | â€” | â€” | âœ“ |
| Full governance & audit log | â€” | â€” | â€” | âœ“ |
| Plugin architecture | â€” | â€” | â€” | âœ“ |

---

## Architecture

```
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚                      provenir CLI / REST API                      â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
               â”‚                                   â”‚
   â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â–¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”           â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â–¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
   â”‚      Orchestration    â”‚           â”‚       Data Pipeline        â”‚
   â”‚  â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â” â”‚           â”‚  â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”   â”‚
   â”‚  â”‚  RunManifest     â”‚ â”‚           â”‚  â”‚  JsonlDataset      â”‚   â”‚
   â”‚  â”‚  (content-addr.) â”‚ â”‚           â”‚  â”‚  QualityScorer     â”‚   â”‚
   â”‚  â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤ â”‚           â”‚  â”‚  CurriculumSampler â”‚   â”‚
   â”‚  â”‚  RLAIFPipeline  â”‚ â”‚           â”‚  â”‚  RAGDataGenerator  â”‚   â”‚
   â”‚  â”‚  DataFlywheel   â”‚ â”‚           â”‚  â”‚  DataFlywheel      â”‚   â”‚
   â”‚  â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤ â”‚           â”‚  â”‚  PromptTemplates   â”‚   â”‚
   â”‚  â”‚  HyperparmSweep â”‚ â”‚           â”‚  â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜   â”‚
   â”‚  â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜ â”‚           â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
   â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
               â”‚
   â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â–¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
   â”‚                       Training Backends                        â”‚
   â”‚   TRLBackend (SFT Â· DPO Â· LoRA Â· QLoRA)   StubBackend        â”‚
   â”‚   DistributedConfig (FSDP Â· DeepSpeed Â· DDP)                 â”‚
   â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
               â”‚
   â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â–¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
   â”‚                      Evaluation Layer                          â”‚
   â”‚  ExactMatch Â· TokenF1 Â· BLEU-4 Â· ROUGE-L Â· Wilson CI         â”‚
   â”‚  RAGEvaluator (faithfulness Â· precision Â· relevance)          â”‚
   â”‚  BenchmarkEvaluator (MMLU Â· HellaSwag Â· ARC Â· GSM8K Â· â€¦)     â”‚
   â”‚  LLM-as-Judge (StubJudge Â· CachedJudge Â· Anthropic Â· OpenAI) â”‚
   â”‚  EvalCallback (early stopping Â· regression gate)              â”‚
   â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
               â”‚
   â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â–¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
   â”‚                      Governance Layer                          â”‚
   â”‚  AuditLogger Â· PIIScanner Â· SecretScanner Â· ModelCardGen      â”‚
   â”‚  SemanticDecontamination Â· AdapterRegistry Â· HubClient        â”‚
   â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
```

---

## Installation

```bash
# Minimal â€” manifests, eval, governance, CLI
pip install provenir

# Full training stack (SFT + DPO + LoRA/QLoRA via TRL)
pip install "provenir[train]"

# Individual feature groups
pip install "provenir[serve]"            # REST API server
pip install "provenir[hub]"             # HuggingFace Hub push / pull
pip install "provenir[merge]"           # Adapter merging
pip install "provenir[semantic]"        # Semantic decontamination
pip install "provenir[benchmarks]"      # MMLU, HellaSwag, ARC, â€¦
pip install "provenir[judge-anthropic]" # LLM-as-judge via Claude
pip install "provenir[judge-openai]"    # LLM-as-judge via GPT-4o

# Everything
pip install "provenir[all]"
```

---

## Quick Start

### 1 â€” Train from a YAML config

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

Every run produces a **content-addressed manifest** â€” a tamper-evident record
of the exact config hash, dataset hash, git SHA, and provenance chain.

### 2 â€” Evaluate predictions

```bash
provenir eval predictions.jsonl --dataset data/eval.jsonl --metrics all
```

Outputs `exact_match`, `token_f1`, `bleu4`, `rouge_l`, plus Wilson 95%
confidence intervals and a regression gate vs. the baseline.

### 3 â€” Run the RLAIF pipeline (no human labels needed)

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

### 4 â€” Generate a RAG training dataset from documents

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

### 5 â€” Data flywheel: auto-augment from eval failures

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

### 6 â€” Merge LoRA adapters

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

### 7 â€” Run standard benchmarks

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

### 8 â€” Push to HuggingFace Hub

```bash
provenir hub push ./my-adapter --repo-id myorg/llama3-finetuned --private
```

### 9 â€” Start the REST API server

```bash
provenir serve --host 0.0.0.0 --port 8000
```

```
GET  /health              â†’ {"status": "ok"}
POST /jobs/train          â†’ Submit training job, returns manifest
GET  /manifests           â†’ List all run IDs
GET  /manifests/{run_id}  â†’ Retrieve manifest by ID
POST /eval                â†’ Run evaluation on predictions
GET  /adapters            â†’ List registered adapters
GET  /audit               â†’ Full audit log (JSONL)
```

---

## Core Concepts

### Manifests â€” Reproducibility by Default

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
| `raw_completion` | Plain prompt â†’ completion (legacy) |

```python
from provenir.data.templates import TEMPLATE_REGISTRY

text = TEMPLATE_REGISTRY.format("llama3", {
    "system": "You are a helpful assistant.",
    "prompt": "Explain transformers.",
    "response": "Transformers are...",
})
```

### RLAIF Pipeline â€” The Unique Moat

Provenir is the **only open-source fine-tuning framework** that combines
automated preference generation, DPO training, evaluation-gated iteration,
and regression detection into a single reproducible loop â€” without requiring
human labellers.

```
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚                    RLAIF Iteration Loop                      â”‚
â”‚                                                             â”‚
â”‚  dataset â”€â”€â–º generate N response variants per prompt        â”‚
â”‚              â”€â”€â–º LLM judge: pairwise ranking                â”‚
â”‚              â”€â”€â–º (chosen, rejected) preference pairs        â”‚
â”‚              â”€â”€â–º DPO training                               â”‚
â”‚              â”€â”€â–º automatic evaluation                       â”‚
â”‚              â”€â”€â–º regression gate                            â”‚
â”‚              â”€â”€â–º iterate (up to n_iterations)               â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
```

Each iteration is fully logged, manifested, and replayable. The pipeline
terminates early if quality regresses beyond the configured threshold.

### LLM-as-Judge

```python
from provenir.eval.judge import (
    StubJudge,      # deterministic, no API calls â€” for CI
    CachedJudge,    # SHA-256 disk cache over any judge
    AnthropicJudge, # Claude
    OpenAIJudge,    # GPT-4o
)

# Wrap any judge with disk caching
judge = CachedJudge(AnthropicJudge(model="claude-haiku-4-5-20251001"))
pref = judge.score_pairwise("Explain gravity", response_a, response_b)
# â†’ Preference(preferred='a', confidence=0.87, rationale='...')

scores = judge.score_rubric("Explain gravity", response, criteria=[
    "factual accuracy", "clarity", "conciseness"
])
```

### Evaluation Layer

```python
from provenir.eval.harness import MultiMetricEvaluator

evaluator = MultiMetricEvaluator()
result = evaluator.evaluate(dataset, predictions)
# result.metrics â†’ {exact_match, token_f1, bleu4, rouge_l}
# result.metrics[*].mean, .ci_lower, .ci_upper (Wilson 95% CI)
```

### Training Observability

```python
from provenir.train.observability import ObservabilityConfig, TrainingObserver

observer = TrainingObserver(ObservabilityConfig(
    backend="wandb",          # wandb | mlflow | tensorboard | none
    project="my-project",
    run_name="llama3-v2",
    log_every_n_steps=10,
    tags=("baseline", "lora"),
))

with observer:
    for step, metrics in training_loop():
        observer.log_step(step, metrics)
    observer.log_eval({"exact_match": 0.82, "bleu4": 0.61})
```

### Semantic Decontamination

Detect and remove training samples that semantically overlap with your
evaluation set â€” preventing inflated benchmark numbers:

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
provenir audit                            Inspect the audit log
provenir model-card                       Generate a model card
provenir reproduce <manifest.json>        Reproduce a run exactly
provenir sweep <config.yaml>              Hyperparameter sweep
provenir compare <run_a> <run_b>          Side-by-side manifest diff
provenir benchmark --model-path <path>    Run MMLU / HellaSwag / ARC / â€¦
provenir merge <a/> <b/> --strategy slerp Merge LoRA adapters
provenir hub push <adapter/> --repo-id â€¦  Push to HuggingFace Hub
provenir hub pull <repo_id> --output â€¦    Pull from HuggingFace Hub
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
| `slerp` | Two similar-task adapters | Spherical linear interpolation â€” preserves weight direction, blends magnitude smoothly |
| `ties` | Dissimilar-task adapters | Magnitude trimming + sign election â€” reduces task interference |
| `dare` | Many adapters | Random drop + rescale (DARE-TIES) â€” scales to 5+ adapters without magnitude collapse |

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
| `benchmarks` | `pip install "provenir[benchmarks]"` | MMLU, HellaSwag, ARC, GSM8K, â€¦ |
| `judge-anthropic` | `pip install "provenir[judge-anthropic]"` | LLM-as-judge via Claude |
| `judge-openai` | `pip install "provenir[judge-openai]"` | LLM-as-judge via GPT-4o |
| `all` | `pip install "provenir[all]"` | Everything |

All optional features degrade gracefully when their packages are absent â€”
stub implementations run without error so pipelines still work in CPU-only
or sandboxed environments.

---

## Package Layout

```
src/provenir/
â”œâ”€â”€ core/           RunManifest, RunConfig, DistributedConfig
â”œâ”€â”€ data/           JsonlDataset, QualityScorer, CurriculumSampler,
â”‚                   SemanticDecontaminationChecker, PromptTemplates (6 formats),
â”‚                   DataFlywheel, RAGDataGenerator
â”œâ”€â”€ train/          Trainer, TRLBackend (SFT/DPO/LoRA/QLoRA),
â”‚   â”œâ”€â”€ backends/   PEFTConfig, TrainingObserver (W&B/MLflow/TensorBoard),
â”‚   â””â”€â”€ ...         EvalCallback, DPOTrainer, GRPOTrainer, PPOTrainer,
â”‚                   GridSweep, RandomSweep, RLAIFPipeline
â”œâ”€â”€ eval/           ExactMatch, TokenF1, BLEU-4, ROUGE-L, Wilson CI,
â”‚                   RAGEvaluator, BenchmarkEvaluator,
â”‚                   LLM-as-Judge (Stub/Cached/Anthropic/OpenAI)
â”œâ”€â”€ adapters/       AdapterRegistry, HubClient, ModelMerger (SLERP/TIES/DARE)
â”œâ”€â”€ rewards/        ExactMatch, Format, WeightedSum, Min, Max, Threshold, Clamped
â”œâ”€â”€ governance/     AuditLogger, PIIScanner, PIIMasker, SecretScanner, ModelCardGen
â”œâ”€â”€ orchestrate/    CostEstimator
â”œâ”€â”€ plugins/        PluginRegistry
â”œâ”€â”€ server/         FastAPI REST server
â””â”€â”€ cli/            13 CLI commands
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
core logic across 456 tests.

---

## Project Status

| Component | Status |
|---|---|
| Core manifests + reproducibility | Production-ready |
| Evaluation pipeline | Production-ready |
| PII / governance / audit | Production-ready |
| TRL backend (SFT + DPO + LoRA) | Beta |
| RLAIF pipeline | Beta |
| REST API server | Beta |
| Data flywheel | Beta |
| RAG dataset generation | Beta |
| Adapter merging | Beta |
| HuggingFace Hub integration | Beta |
| Benchmark evaluation suite | Beta |
| Distributed training (FSDP / DeepSpeed) | Experimental |
| Web UI / dashboard | Roadmap |
| Real-time rollout integration | Roadmap |

---

## Roadmap

- **v0.3** â€” vLLM and SGLang backends for online rollout; perplexity metric
- **v0.4** â€” Streaming training events via WebSocket; live eval dashboard
- **v0.5** â€” Multi-agent RLAIF (constitutional AI loop); reward model training
- **v1.0** â€” Production SLAs; enterprise governance controls; HIPAA/SOC2 audit trail

---

## Design Principles

**1. Reproducibility is a first-class citizen.**
Every run produces a content-addressed manifest. Every manifest can reproduce
the exact run that created it. There is no "it worked on my machine."

**2. Evaluation before deployment.**
The regression gate, EvalCallback, and RLAIF loop all prevent degraded models
from advancing through the pipeline without explicit override.

**3. Degrade gracefully.**
Every optional dependency â€” torch, TRL, anthropic, wandb, huggingface_hub â€”
is conditionally imported. The full orchestration stack, manifest system, and
evaluation pipeline work without a GPU.

**4. Governance is not optional.**
PII scanning, secret detection, audit logging, and model card generation are
built into the core package, not plugins. Every production deployment should
be auditable.

**5. Plugin-first architecture.**
New training backends, reward functions, metrics, and judges are registered
via protocol interfaces. No monkey-patching, no subclassing deep hierarchies.

---

## Non-Goals

- Reimplementing training kernels or competing with Unsloth on raw single-GPU throughput
- Forking or replacing the Hugging Face training stack â€” Provenir wraps and orchestrates it
- Providing hosted SaaS, billing, or proprietary cloud services

---

## License

Apache 2.0 â€” see [LICENSE](LICENSE).

---

## Citation

If you use Provenir in research, please cite:

```bibtex
@software{provenir2025,
  title  = {Provenir: Reproducible, Evaluation-First Fine-Tuning Orchestration},
  year   = {2025},
  url    = {https://github.com/anilatambharii/provenir},
  note   = {v0.2.0}
}
```

