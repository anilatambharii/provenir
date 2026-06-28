"""
Basic supervised fine-tuning using the stub backend (no GPU required).

For real training: change backend to "trl" and install provenir[train].

Usage:
    python examples/basic_sft.py
"""

from __future__ import annotations

import json

from provenir.core.config import RunConfig
from provenir.data.dataset import JsonlDataset
from provenir.eval.harness import MultiMetricEvaluator
from provenir.governance.audit import AuditLogger
from provenir.governance.pii import PIIScanner
from provenir.train.backends.stub import StubBackend
from provenir.train.trainer import Trainer

# ---------------------------------------------------------------------------
# 1. Build a small dataset inline (in practice, load from JSONL)
# ---------------------------------------------------------------------------

records = [
    {"prompt": "What is LoRA?",  "response": "LoRA is a parameter-efficient fine-tuning method."},
    {"prompt": "What is DPO?",   "response": "DPO is Direct Preference Optimization."},
    {"prompt": "What is RLHF?",  "response": "RLHF is Reinforcement Learning from Human Feedback."},
    {"prompt": "What is RLAIF?", "response": "RLAIF replaces human feedback with an AI judge."},
]

train_ds = JsonlDataset.from_records(records[:3])
eval_ds  = JsonlDataset.from_records(records[3:])

# ---------------------------------------------------------------------------
# 2. Governance: scan for PII before training
# ---------------------------------------------------------------------------

audit   = AuditLogger("artifacts/audit")
scanner = PIIScanner()

all_text = " ".join(
    json.dumps(r) for r in train_ds.records
)
findings = scanner.scan(all_text)

if findings:
    categories = {f.category.value for f in findings}
    print(f"PII detected: {categories}")
else:
    print("PII scan: clean")

audit.log("pii_scan", actor="example-script", has_pii=bool(findings))

# ---------------------------------------------------------------------------
# 3. Configure the run
# ---------------------------------------------------------------------------

config = RunConfig(
    name="basic-sft-example",
    backend="stub",            # change to "trl" for real training
    seed=42,
    output_dir="artifacts/basic_sft",
)

# ---------------------------------------------------------------------------
# 4. Train
# ---------------------------------------------------------------------------

trainer  = Trainer(backend=StubBackend(), config=config)
manifest = trainer.run(train_ds)

print(f"\nRun ID:       {manifest.run_id}")
print(f"Config hash:  {manifest.config_hash}")
print(f"Dataset hash: {manifest.dataset_hash}")

audit.log("training_complete", actor="example-script", run_id=manifest.run_id)

# ---------------------------------------------------------------------------
# 5. Evaluate
# ---------------------------------------------------------------------------

predictions = ["RLAIF replaces human feedback with an AI judge."]
result = MultiMetricEvaluator().evaluate(eval_ds, predictions)

print("\nEvaluation:")
if result.metrics:
    for name, summary in result.metrics.items():
        lo, hi = summary.confidence_interval
        print(f"  {name:15s} {summary.mean:.3f}  [{lo:.3f}, {hi:.3f}]")

audit.log("eval_complete", actor="example-script", run_id=manifest.run_id)
