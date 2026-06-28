"""
Merge multiple LoRA adapters using SLERP, TIES, or DARE.

This example uses the stub path (no actual weight files required).
For real merging with weights: install provenir[merge] and pass adapter
directories that contain adapter_model.safetensors files.

Usage:
    python examples/adapter_merging.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import provenir.adapters.merging as merge_mod
from provenir.adapters.merging import MergeConfig, ModelMerger

# Force stub mode so the example works without actual safetensors weight files.
# Remove these two lines when running with real adapter directories.
_orig_torch = merge_mod._HAS_TORCH
_orig_st    = merge_mod._HAS_SAFETENSORS
merge_mod._HAS_TORCH       = False
merge_mod._HAS_SAFETENSORS = False

try:
    merger = ModelMerger()

    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)

        adapter_a = base / "adapter_a"
        adapter_b = base / "adapter_b"
        adapter_c = base / "adapter_c"
        adapter_a.mkdir()
        adapter_b.mkdir()
        adapter_c.mkdir()

        # --- SLERP (2 adapters, similar tasks) ---
        print("Strategy: SLERP")
        result = merger.merge(
            [adapter_a, adapter_b],
            config=MergeConfig(strategy="slerp"),
            output_dir=base / "slerp",
        )
        print(f"  Output:   {result.output_path}")
        print(f"  Strategy: {result.strategy}")
        print(f"  Inputs:   {len(result.adapter_paths)} adapters")
        print(f"  Stub:     {result.metadata.get('stub', False)}")

        # --- TIES (2–4 adapters, different tasks) ---
        print("\nStrategy: TIES (density=0.4)")
        result = merger.merge(
            [adapter_a, adapter_b, adapter_c],
            config=MergeConfig(strategy="ties", density=0.4),
            output_dir=base / "ties",
        )
        print(f"  Output:   {result.output_path}")
        print(f"  Strategy: {result.strategy}")
        print(f"  Inputs:   {len(result.adapter_paths)} adapters")

        # --- DARE (4+ adapters) ---
        print("\nStrategy: DARE (density=0.5)")
        result = merger.merge(
            [adapter_a, adapter_b, adapter_c],
            config=MergeConfig(strategy="dare", density=0.5),
            output_dir=base / "dare",
        )
        print(f"  Output:   {result.output_path}")
        print(f"  Strategy: {result.strategy}")
        print(f"  Inputs:   {len(result.adapter_paths)} adapters")

finally:
    # Restore the original flags
    merge_mod._HAS_TORCH       = _orig_torch
    merge_mod._HAS_SAFETENSORS = _orig_st

print("\nDone. Install provenir[merge] and pass real adapter directories")
print("to merge actual model weights.")
