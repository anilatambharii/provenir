# Adapter Merging

Combine multiple LoRA adapters into a single adapter using mathematically
principled merging strategies — no additional training required.

---

## Overview

Provenir ships three merging algorithms:

| Strategy | Best For | How It Works |
|---|---|---|
| `slerp` | Two adapters, similar tasks | Spherical linear interpolation in weight space — preserves direction, blends magnitude smoothly |
| `ties` | Adapters from different tasks | Magnitude trimming + sign election — reduces task interference |
| `dare` | Many adapters (3+) | Random drop + rescale (DARE-TIES) — scales to many adapters without magnitude collapse |

---

## Quick Start

### CLI

```bash
provenir merge adapter_a/ adapter_b/ \
  --strategy slerp \
  --output merged_adapter/
```

With TIES:

```bash
provenir merge adapter_a/ adapter_b/ adapter_c/ \
  --strategy ties \
  --density 0.4 \
  --output merged_adapter/
```

### Python

```python
from provenir.adapters.merging import MergeConfig, ModelMerger
from pathlib import Path

merger = ModelMerger()

result = merger.merge(
    adapter_paths=[
        Path("adapters/task_a"),
        Path("adapters/task_b"),
        Path("adapters/task_c"),
    ],
    config=MergeConfig(
        strategy="ties",
        density=0.5,   # keep 50% of weights by magnitude
        scale=1.0,
    ),
    output_dir=Path("merged/"),
)

print(f"Merged to: {result.output_path}")
print(f"Strategy:  {result.strategy}")
print(f"Inputs:    {len(result.adapter_paths)} adapters")
```

---

## Configuration

### MergeConfig

| Field | Default | Description |
|---|---|---|
| `strategy` | `"slerp"` | Merging algorithm: `"slerp"` \| `"ties"` \| `"dare"` |
| `density` | `0.5` | (TIES/DARE) Fraction of weights to retain by magnitude — `0.0`–`1.0` |
| `scale` | `1.0` | Post-merge scale factor applied to the merged adapter |

---

## Algorithm Details

### SLERP

Spherical linear interpolation between two weight tensors at interpolation
factor `t = 0.5`. Preserves the geometric relationship between weights and
avoids the magnitude collapse of naive linear averaging.

Best when merging two adapters trained on related tasks.

**Requires exactly 2 adapters.**

### TIES

Trim, Elect, Merge:

1. **Trim** — for each parameter, keep only the top-`density` fraction of
   weights by magnitude; set the rest to zero.
2. **Elect** — resolve sign conflicts by majority vote across adapters.
3. **Merge** — average the surviving (trimmed, sign-resolved) weights.

Best when merging 2–4 adapters from different tasks. Reduces interference by
eliminating low-magnitude weights that carry task-specific noise.

### DARE

Drop And REscale:

1. Randomly drop `(1 - density)` fraction of each adapter's delta weights.
2. Rescale the surviving weights by `1 / density` to preserve expected magnitude.
3. Average the rescaled weights across all adapters.

Best for merging 4+ adapters. The random drop strategy reduces pairwise
interference without requiring a magnitude ordering, and scales better than
TIES to large numbers of adapters.

---

## Requirements

```bash
pip install "provenir[merge]"
```

Installs `safetensors` and `torch`. When these packages are absent, `merge()`
returns a stub `MergeResult` without writing any files (useful for testing
pipeline logic without GPU dependencies).

---

## Verifying a Merged Adapter

After merging, verify the output on your eval set before deploying:

```bash
provenir eval predictions.jsonl \
  --dataset data/eval.jsonl \
  --model-path merged_adapter/ \
  --metrics all
```

Or benchmark against standard tasks:

```bash
provenir benchmark --model-path merged_adapter/ --benchmarks mmlu hellaswag
```
