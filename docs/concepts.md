# Core Concepts

This page explains the key abstractions in Provenir and how they fit together.

---

## RunManifest — Reproducibility by Default

Every training run in Provenir produces a `RunManifest`: a content-addressed,
tamper-evident record that captures:

| Field | What it captures |
|---|---|
| `run_id` | UUID for this specific run |
| `config_hash` | SHA-256 of the serialised `RunConfig` |
| `dataset_hash` | SHA-256 of the training dataset |
| `git_sha` | Current HEAD commit (if in a git repo) |
| `seed` | Random seed used |
| `timestamp` | ISO-8601 start time |
| `provenance` | Chain of parent run IDs (for RLAIF iterations) |

Any run can be reproduced exactly:

```bash
provenir reproduce artifacts/manifests/<run_id>.json
```

Provenir will verify the config hash and dataset hash before starting and raise
an error if either has changed.

The manifest is stored as a JSON file and surfaced via the REST API at
`GET /manifests/{run_id}`.

---

## RunConfig — Unified Training Configuration

`RunConfig` is a Pydantic model that covers every aspect of a training job:

```yaml
name: my-run
backend: trl          # trl | stub
seed: 42
model_name_or_path: meta-llama/Llama-3.2-1B
max_steps: 1000
batch_size: 8

peft:
  rank: 16
  alpha: 32
  target_modules: ["q_proj", "v_proj"]
  load_in_4bit: false   # QLoRA

distributed:
  strategy: fsdp        # fsdp | deepspeed | ddp | none
  num_gpus: 8

observability_backend: wandb
```

Load from YAML:

```python
from provenir.core.config import RunConfig

config = RunConfig.from_yaml("my_run.yaml")
```

Or construct in Python:

```python
from provenir.core.config import RunConfig, PEFTConfig

config = RunConfig(
    name="my-run",
    model_name_or_path="meta-llama/Llama-3.2-1B",
    peft=PEFTConfig(rank=16, alpha=32),
)
```

---

## Training Backends

Backends implement the `TrainingBackend` protocol and are selected by the
`backend` field in `RunConfig`.

| Backend | When to use |
|---|---|
| `trl` | Production training — SFT, DPO, LoRA, QLoRA via HuggingFace TRL |
| `stub` | Testing and CI — returns immediately with a fake manifest |

The TRL backend supports:

- **SFT** (Supervised Fine-Tuning) — default algorithm
- **DPO** (Direct Preference Optimization) — for preference datasets
- **LoRA** — parameter-efficient fine-tuning via `peft` config
- **QLoRA** — 4-bit or 8-bit quantized LoRA
- **rsLoRA** — rank-stabilised LoRA scaling (`use_rslora: true`)

---

## Prompt Templates

Six built-in formats, switchable at runtime:

| Template | Chat format | Use case |
|---|---|---|
| `alpaca` | `### Instruction: … ### Response:` | Classic instruction tuning |
| `chatml` | `<|im_start|>user … <|im_end|>` | Multi-turn, OpenAI-compatible |
| `llama3` | `<|begin_of_text|> … <|eot_id|>` | Llama 3 official format |
| `mistral` | `[INST] … [/INST]` | Mistral instruction format |
| `phi3` | `<|user|> … <|assistant|>` | Microsoft Phi-3 |
| `raw_completion` | `{prompt}{response}` | Plain completion, no special tokens |

```python
from provenir.data.templates import TEMPLATE_REGISTRY

# List available templates
names = TEMPLATE_REGISTRY.list_names()

# Format a record
text = TEMPLATE_REGISTRY.format("llama3", {
    "system": "You are a helpful assistant.",
    "prompt": "What is LoRA?",
    "response": "LoRA is a parameter-efficient fine-tuning method...",
})
```

---

## Evaluation Layer

### Metrics

All metrics implement the `MetricFn` protocol:

| Metric | Description |
|---|---|
| `ExactMatch` | Binary string equality |
| `TokenF1` | Token-level F1 score |
| `BLEU4` | BLEU-4 (4-gram overlap) |
| `ROUGE_L` | Longest common subsequence recall |

All results include **Wilson 95% confidence intervals** — not just the point
estimate. This catches "improvements" that are just noise.

```python
from provenir.eval.harness import MultiMetricEvaluator

result = MultiMetricEvaluator().evaluate(dataset, predictions)
em = result.metrics["exact_match"]
print(f"{em.mean:.3f}  [{em.ci_lower:.3f}, {em.ci_upper:.3f}]")
```

### Regression Gate

The `RegressionGate` blocks promotion if eval scores drop below the baseline
by more than a configured tolerance:

```python
from provenir.eval.regression import RegressionGate

gate = RegressionGate(tolerance=0.02)   # allow up to 2% regression
gate.check(new_result, baseline_result)  # raises RegressionError if gate trips
```

### EvalCallback

`EvalCallback` hooks into training to run evaluation at configurable intervals
and optionally stop training early if quality stops improving:

```python
from provenir.train.eval_callback import EvalCallback, EvalCallbackConfig

callback = EvalCallback(
    config=EvalCallbackConfig(
        eval_every_n_steps=100,
        early_stopping_patience=3,
        regression_tolerance=0.02,
    ),
    evaluator=MultiMetricEvaluator(),
    eval_dataset=eval_ds,
)
```

### RAG Metrics

For retrieval-augmented generation evaluation:

| Metric | What it measures |
|---|---|
| `faithfulness` | Fraction of answer tokens grounded in context |
| `context_precision` | Overlap between context and question |
| `answer_relevance` | Fraction of question tokens covered by answer |

```python
from provenir.eval.rag_metrics import RAGEvaluator

results = RAGEvaluator().evaluate([{
    "question": "What is the capital of France?",
    "answer": "Paris is the capital of France.",
    "context": "France is a country in Western Europe. Its capital is Paris.",
}])
```

---

## LLM-as-Judge

Provenir ships four judge implementations:

| Judge | Use case |
|---|---|
| `StubJudge` | Deterministic, no API calls — for CI and offline testing |
| `CachedJudge` | SHA-256 disk cache wrapping any other judge |
| `AnthropicJudge` | Pairwise and rubric scoring via Claude |
| `OpenAIJudge` | Pairwise and rubric scoring via GPT-4o |

All judges implement the `LLMJudge` protocol:

```python
from provenir.eval.judge import AnthropicJudge, CachedJudge

judge = CachedJudge(AnthropicJudge(model="claude-haiku-4-5-20251001"))

# Pairwise: which response is better?
pref = judge.score_pairwise("What is LoRA?", response_a, response_b)
# → Preference(preferred='a', confidence=0.91, rationale='Response A is more precise...')

# Rubric: score against specific criteria
scores = judge.score_rubric("What is LoRA?", response, criteria=[
    "factual accuracy", "clarity", "conciseness"
])
```

---

## RLAIF Pipeline

Provenir's unique differentiator: an end-to-end loop that generates preference
data, trains via DPO, evaluates, and iterates — without any human labelling.

```
dataset
  │
  ▼
generate N response variants per prompt
  │
  ▼
LLM judge: pairwise ranking of variants
  │
  ▼
build (chosen, rejected) preference pairs
  │
  ▼
DPO training on preference pairs
  │
  ▼
automatic evaluation against held-out set
  │
  ▼
regression gate (stop if quality regresses)
  │
  ▼
next iteration (up to n_iterations)
```

See the [RLAIF guide](guides/rlaif.md) for full configuration and usage.

---

## Data Flywheel

The `DataFlywheel` automatically mines prediction failures, generates augmented
variants via the judge, filters by quality, and augments the training set:

```
eval failures  ──►  judge: generate variants  ──►  quality filter  ──►  augmented dataset
```

See the [Data Flywheel guide](guides/flywheel.md).

---

## Governance

### Audit Log

Every significant event in Provenir is logged to an append-only JSONL file:

```python
from provenir.governance.audit import AuditLogger

audit = AuditLogger("artifacts/audit")
audit.log("training_complete", actor="ci-bot", run_id=manifest.run_id, steps=500)
```

The audit log is exposed via `GET /audit` in the REST API and `provenir audit`
in the CLI.

### PII & Secret Scanning

```python
from provenir.governance.pii import PIIScanner, PIIMasker
from provenir.governance.scanners import SecretScanner

# Scan before training
pii_report = PIIScanner().scan(dataset)
secret_report = SecretScanner().scan(dataset)

if pii_report.has_pii:
    dataset = PIIMasker().mask(dataset)
```

### Model Cards

```python
from provenir.governance.model_card import ModelCardGenerator

card = ModelCardGenerator().generate(manifest, eval_result)
card.save("MODEL_CARD.md")
```

Or via CLI:

```bash
provenir model-card --manifest artifacts/manifests/<run_id>.json
```

---

## Plugin Architecture

New backends, metrics, judges, and reward functions are registered via protocol
interfaces — no subclassing required:

```python
from provenir.plugins.registry import PluginRegistry
from provenir.core.abstractions import TrainingBackend

@PluginRegistry.register("my-backend")
class MyBackend(TrainingBackend):
    def fit(self, config, dataset):
        ...
```

---

## Reward Functions

Reward primitives for RLHF/GRPO training:

| Reward | Description |
|---|---|
| `ExactMatchReward` | Binary 1.0 if prediction matches reference |
| `FormatReward` | Checks structural constraints (JSON, code block, etc.) |
| `WeightedSumReward` | Linear combination of multiple rewards |
| `MinReward` | Returns the minimum across multiple rewards |
| `MaxReward` | Returns the maximum across multiple rewards |
| `ThresholdGatedReward` | Returns 0 if any component falls below a threshold |
| `ClampedReward` | Clamps the reward to a specified range |

```python
from provenir.rewards.primitives import WeightedSumReward, ExactMatchReward, FormatReward

reward = WeightedSumReward(rewards=[
    (ExactMatchReward(), 0.7),
    (FormatReward(pattern=r"^\[.*\]$"), 0.3),
])
score = reward.score(prediction, reference)
```
