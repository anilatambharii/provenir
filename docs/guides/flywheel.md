# Data Flywheel

Automatically mine evaluation failures, generate high-quality augmented
variants, and feed them back into the training set — without human labelling.

---

## Overview

The `DataFlywheel` closes the loop between evaluation and training:

```
training set  ──►  model  ──►  eval predictions
                                    │
                           score < threshold?
                                    │
                                    ▼
                           mine failure records
                                    │
                                    ▼
                           judge: generate variants
                                    │
                                    ▼
                           quality filter
                                    │
                                    ▼
                           augmented training set
```

Each cycle produces new training examples targeted exactly at the failure modes
your current model exhibits — no manual annotation required.

---

## Quick Start

```python
from provenir.data.flywheel import DataFlywheel, FlywheelConfig
from provenir.eval.judge import AnthropicJudge
from provenir.data.dataset import JsonlDataset

flywheel = DataFlywheel(
    config=FlywheelConfig(
        min_score_threshold=0.7,       # mine predictions scoring below this
        max_variants_per_failure=3,    # generate N variants per failure
        quality_filter_threshold=0.5,  # keep variants scoring above this
    ),
    judge=AnthropicJudge(),
)

train_ds = JsonlDataset.from_jsonl("data/train.jsonl")
eval_ds  = JsonlDataset.from_jsonl("data/eval.jsonl")

augmented = flywheel.run(train_ds, eval_ds)
augmented.save("data/train_augmented.jsonl")

print(f"Original:  {len(train_ds.records)} records")
print(f"Augmented: {len(augmented.records)} records")
```

---

## Configuration

### FlywheelConfig

| Field | Default | Description |
|---|---|---|
| `min_score_threshold` | `0.7` | Eval predictions below this score are treated as failures |
| `max_variants_per_failure` | `3` | Number of augmented variants to generate per failure |
| `quality_filter_threshold` | `0.5` | Generated variants must score above this to be included |

---

## Using with RLAIF

The flywheel pairs naturally with the RLAIF pipeline. Run the flywheel first
to enrich the training set, then run RLAIF to fine-tune on both original and
augmented data:

```python
# Step 1: Augment
augmented = DataFlywheel(config=FlywheelConfig(), judge=judge).run(train_ds, eval_ds)

# Step 2: RLAIF on augmented set
pipeline = RLAIFPipeline(judge=judge, backend=backend, base_config=config)
iterations = pipeline.run(augmented, eval_ds)
```

---

## Offline Mode (StubJudge)

For CI and testing, use `StubJudge` to run the flywheel without any API calls:

```python
from provenir.eval.judge import StubJudge

flywheel = DataFlywheel(
    config=FlywheelConfig(max_variants_per_failure=1),
    judge=StubJudge(),
)
```

The stub judge generates deterministic synthetic variants — useful for
verifying pipeline logic without incurring API costs.
