"""
RLAIF pipeline example — automated preference learning without human labels.

Uses StubJudge (no API key required). For production, swap in:
    AnthropicJudge(model="claude-haiku-4-5-20251001")
    OpenAIJudge(model="gpt-4o-mini")

Requires: pip install "provenir[train]"

Usage:
    python examples/rlaif_pipeline.py
"""

from __future__ import annotations

from provenir.core.config import RunConfig
from provenir.data.dataset import JsonlDataset
from provenir.eval.judge import CachedJudge, StubJudge
from provenir.train.backends.stub import StubBackend
from provenir.train.rlaif import RLAIFConfig, RLAIFPipeline

# ---------------------------------------------------------------------------
# 1. Prepare datasets
# ---------------------------------------------------------------------------

train_records = [
    {"prompt": "Explain gradient descent.",
     "response": "Gradient descent minimises a loss function by iterative parameter updates."},
    {"prompt": "What is overfitting?",
     "response": "Overfitting occurs when a model learns noise in training data."},
    {"prompt": "Define regularisation.",
     "response": "Regularisation adds a penalty term to prevent overfitting."},
    {"prompt": "What is a learning rate?",
     "response": "The learning rate controls how large each gradient step is."},
]

eval_records = [
    {"prompt": "What is backpropagation?",
     "response": "Backpropagation computes gradients by chain rule through the network."},
    {"prompt": "Define batch normalisation.",
     "response": "Batch normalisation normalises layer inputs to stabilise training."},
]

train_ds = JsonlDataset.from_records(train_records)
eval_ds  = JsonlDataset.from_records(eval_records)

# ---------------------------------------------------------------------------
# 2. Configure the pipeline
# ---------------------------------------------------------------------------

# StubJudge = deterministic, no API calls, for offline testing.
# Wrap with CachedJudge so repeated calls are free in subsequent runs.
# Production: CachedJudge(AnthropicJudge(model="claude-haiku-4-5-20251001"))
judge = CachedJudge(StubJudge(), cache_dir=".judge_cache")

base_config = RunConfig(
    name="rlaif-example",
    backend="stub",          # change to "trl" for real training
    seed=42,
    output_dir="artifacts/rlaif",
)

rlaif_config = RLAIFConfig(
    n_iterations=3,
    responses_per_prompt=4,
    regression_threshold=0.05,
)

# ---------------------------------------------------------------------------
# 3. Run the pipeline
# ---------------------------------------------------------------------------

pipeline = RLAIFPipeline(
    judge=judge,
    backend=StubBackend(),
    base_config=base_config,
    rlaif_config=rlaif_config,
)

print("Starting RLAIF pipeline...")
print(f"  Iterations:            {rlaif_config.n_iterations}")
print(f"  Responses per prompt:  {rlaif_config.responses_per_prompt}")
print(f"  Regression threshold:  {rlaif_config.regression_threshold:.0%}")
print()

iterations = pipeline.run(train_ds, eval_ds)

# ---------------------------------------------------------------------------
# 4. Inspect results
# ---------------------------------------------------------------------------

for it in iterations:
    em_score = "n/a"
    if it.eval_result and it.eval_result.metrics:
        em = it.eval_result.metrics.get("exact_match")
        if em:
            em_score = f"{em.mean:.3f}"

    print(f"Iteration {it.iteration}  |  preferences={it.preference_count}"
          f"  |  exact_match={em_score}")
    print(f"  Manifest:   {it.manifest.run_id}")
    print(f"  Provenance: {it.manifest.provenance}")
    print()

print(f"Completed {len(iterations)} iteration(s).")
