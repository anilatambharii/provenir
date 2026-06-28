# RLAIF Pipeline

**Reinforcement Learning from AI Feedback** — iterate from a raw dataset to a
preference-tuned model without a single human label.

---

## Overview

Provenir's RLAIF pipeline is the framework's primary differentiator. It is the
only open-source fine-tuning pipeline that combines:

- Automated response variant generation
- Pairwise LLM judging
- DPO training on AI-generated preferences
- Evaluation-gated iteration
- Regression detection and automatic early stopping

…into a single reproducible, auditable loop.

```
dataset
  │
  ▼  ┌─────────────────────────────────────┐
  │  │          Iteration N                │
  │  │                                     │
  │  │  1. generate N response variants    │
  │  │        per training prompt          │
  │  │                                     │
  │  │  2. LLM judge: pairwise ranking     │
  │  │        of response variants         │
  │  │                                     │
  │  │  3. build (chosen, rejected) pairs  │
  │  │                                     │
  │  │  4. DPO training                    │
  │  │                                     │
  │  │  5. evaluate on held-out set        │
  │  │                                     │
  │  │  6. regression gate                 │
  │  │     → stop if quality regresses     │
  │  │                                     │
  │  └─────────────────────────────────────┘
  │               │
  └───────────────┘ (up to n_iterations)
```

Each iteration produces a fully logged, content-addressed `RunManifest` with
a `provenance` chain linking it to the previous iteration.

---

## Quick Start

```bash
pip install "provenir[train,judge-anthropic]"

provenir rlaif my_config.yaml \
  --dataset data/train.jsonl \
  --eval-dataset data/eval.jsonl \
  --judge anthropic \
  --iterations 3
```

---

## Python API

```python
from provenir.train.rlaif import RLAIFConfig, RLAIFPipeline
from provenir.eval.judge import AnthropicJudge, CachedJudge
from provenir.train.backends.trl import TRLBackend
from provenir.core.config import RunConfig, PEFTConfig
from provenir.data.dataset import JsonlDataset

# Configure the pipeline
pipeline = RLAIFPipeline(
    judge=CachedJudge(
        AnthropicJudge(model="claude-haiku-4-5-20251001"),
        cache_dir=".judge_cache",
    ),
    backend=TRLBackend(),
    base_config=RunConfig(
        model_name_or_path="meta-llama/Llama-3.2-1B",
        max_steps=200,
        peft=PEFTConfig(rank=16, alpha=32),
    ),
    rlaif_config=RLAIFConfig(
        n_iterations=3,
        responses_per_prompt=4,
        regression_tolerance=0.05,
    ),
)

train_ds = JsonlDataset.from_jsonl("data/train.jsonl")
eval_ds  = JsonlDataset.from_jsonl("data/eval.jsonl")

iterations = pipeline.run(train_ds, eval_ds)

for it in iterations:
    print(f"Iteration {it.iteration}")
    print(f"  Preference pairs: {it.preference_count}")
    print(f"  Eval result: {it.eval_result}")
    print(f"  Manifest: {it.manifest.run_id}")
    if it.regressed:
        print("  [stopped: regression detected]")
        break
```

---

## Configuration

### RLAIFConfig

| Field | Default | Description |
|---|---|---|
| `n_iterations` | `3` | Maximum number of RLAIF iterations |
| `responses_per_prompt` | `4` | Candidate responses generated per prompt |
| `regression_tolerance` | `0.05` | Maximum allowed drop in primary metric before stopping |

### Choosing a Judge

| Judge | Cost | Speed | Best for |
|---|---|---|---|
| `StubJudge` | Free | Instant | CI, testing, offline |
| `CachedJudge(AnthropicJudge(...))` | Low | Fast | Production — caches repeated prompts |
| `AnthropicJudge(model="claude-opus-4-8")` | Higher | Moderate | Highest-quality preferences |
| `OpenAIJudge(model="gpt-4o")` | Higher | Moderate | Alternative judge |

Always wrap production judges with `CachedJudge` to avoid re-judging identical
prompt/response pairs across iterations.

---

## How Preferences Are Generated

In each iteration:

1. For each training prompt, the pipeline generates `responses_per_prompt`
   candidate responses (stub: syntactic variants of the original response).

2. The judge compares candidates pairwise and ranks them.

3. The highest-ranked candidate becomes `chosen`; the lowest-ranked becomes
   `rejected`.

4. The resulting `(prompt, chosen, rejected)` triples form the DPO training set
   for this iteration.

---

## Regression Detection

After each iteration's eval, the pipeline compares the primary metric (default:
`exact_match`) against the best score seen so far. If the score drops by more
than `regression_tolerance`, the pipeline stops and marks the iteration as
`regressed=True`.

The best model checkpoint (from the iteration with the highest eval score) is
the recommended final model.

---

## Manifests and Reproducibility

Each RLAIF iteration produces a `RunManifest` with:

- `run_id` — unique to this iteration
- `provenance` — list of parent run IDs (the full iteration chain)
- `config_hash` — hash of the DPO config used
- `dataset_hash` — hash of the preference pairs used

To reproduce a specific iteration:

```bash
provenir reproduce artifacts/manifests/<iteration_run_id>.json
```

---

## Cost Estimation

Judge API calls are the main cost driver. Rough estimates for 3 iterations,
1000 training prompts, 4 responses per prompt:

| Judge | API calls | Estimated cost |
|---|---|---|
| `AnthropicJudge(claude-haiku-4-5)` | ~6,000 | < $5 |
| `AnthropicJudge(claude-sonnet-4-6)` | ~6,000 | ~$30 |
| `OpenAIJudge(gpt-4o-mini)` | ~6,000 | < $5 |

Use `CachedJudge` to avoid re-judging the same pairs in subsequent runs.
