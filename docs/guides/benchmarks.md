# Benchmark Evaluation

Evaluate any model or fine-tuned adapter against industry-standard benchmarks
using the EleutherAI `lm-evaluation-harness`.

---

## Supported Benchmarks

| Benchmark | Domain | Default shots | Measures |
|---|---|---|---|
| `mmlu` | Knowledge (57 subjects) | 5-shot | Factual knowledge breadth |
| `hellaswag` | Commonsense reasoning | 10-shot | Situational language understanding |
| `arc_easy` | Science QA | 0-shot | Elementary science knowledge |
| `arc_challenge` | Science QA (hard) | 25-shot | Advanced science reasoning |
| `truthfulqa_mc1` | Truthfulness | 0-shot | Avoiding false beliefs |
| `truthfulqa_mc2` | Truthfulness | 0-shot | Calibrated truthfulness |
| `winogrande` | Commonsense NLI | 5-shot | Pronoun disambiguation |
| `gsm8k` | Grade-school math | 8-shot | Multi-step arithmetic |
| `humaneval` | Code generation | 0-shot | Python programming |

---

## Quick Start

### CLI

```bash
# Install benchmark deps
pip install "provenir[benchmarks]"

# Run a single benchmark
provenir benchmark \
  --model-path ./my-adapter \
  --benchmarks mmlu

# Run a suite
provenir benchmark \
  --model-path ./my-adapter \
  --benchmarks mmlu hellaswag arc_easy arc_challenge gsm8k
```

### Python

```python
from provenir.eval.benchmarks import BenchmarkConfig, BenchmarkEvaluator
from pathlib import Path

evaluator = BenchmarkEvaluator()

results = evaluator.run_suite(
    model_path=Path("./my-adapter"),
    configs=[
        BenchmarkConfig(benchmark="mmlu", num_fewshot=5),
        BenchmarkConfig(benchmark="hellaswag", num_fewshot=10),
        BenchmarkConfig(benchmark="arc_challenge", num_fewshot=25),
        BenchmarkConfig(benchmark="gsm8k", num_fewshot=8),
    ],
)

for r in results:
    print(f"{r.benchmark:20s}  {r.score:.3f}  ({r.num_examples} examples)")
```

Output:

```
mmlu                  0.614  (14042 examples)
hellaswag             0.791  (10042 examples)
arc_challenge         0.531  (1172 examples)
gsm8k                 0.482  (1319 examples)
```

---

## Configuration

### BenchmarkConfig

| Field | Default | Description |
|---|---|---|
| `benchmark` | required | Benchmark name (see table above) |
| `num_fewshot` | `0` | Number of in-context examples |
| `limit` | `None` | Limit number of evaluation examples (for fast testing) |

---

## Comparing Before and After Fine-Tuning

```python
base_results    = evaluator.run_suite("base-model/",    configs)
tuned_results   = evaluator.run_suite("tuned-adapter/", configs)

for base, tuned in zip(base_results, tuned_results):
    delta = tuned.score - base.score
    sign  = "+" if delta >= 0 else ""
    print(f"{base.benchmark:20s}  {base.score:.3f} → {tuned.score:.3f}  ({sign}{delta:.3f})")
```

---

## Using with Manifests

```bash
# Save results alongside the run manifest
provenir benchmark \
  --model-path ./my-adapter \
  --benchmarks mmlu hellaswag \
  --manifest artifacts/manifests/<run_id>.json
```

Benchmark results are appended to the manifest and visible via
`GET /manifests/{run_id}` in the REST API.

---

## Requirements

```bash
pip install "provenir[benchmarks]"
```

Installs `lm-eval ≥ 0.4.0` and its dependencies. When `lm-eval` is not
installed, `BenchmarkEvaluator.run()` raises `ImportError` with a clear
install instruction.
