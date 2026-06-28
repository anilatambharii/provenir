from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field


class DistributedConfig(BaseModel):
    """Configuration for multi-GPU / multi-node training."""

    strategy: Literal["fsdp", "deepspeed", "ddp", "none"] = "none"
    num_gpus: int = Field(default=1, ge=1)
    num_nodes: int = Field(default=1, ge=1)
    deepspeed_stage: int = Field(default=2, ge=0, le=3)


class RunConfig(BaseModel):
    """Top-level run configuration for Provenir.

    Extended fields are optional and backward-compatible — existing YAML
    configs that only set *name*, *backend*, *seed*, *deterministic*, and
    *output_dir* continue to work without changes.
    """

    # --- core ---
    name: str = Field(default="default-run")
    backend: str = Field(default="stub")
    seed: int = Field(default=0)
    deterministic: bool = Field(default=True)
    output_dir: str = Field(default="artifacts")

    # --- model ---
    model_name_or_path: str | None = Field(default=None)
    max_steps: int = Field(default=10, ge=1)
    batch_size: int = Field(default=1, ge=1)

    # --- PEFT / LoRA (serialised via PEFTConfig.to_dict()) ---
    peft: dict[str, Any] | None = Field(default=None)

    # --- distributed ---
    distributed: DistributedConfig | None = Field(default=None)

    # --- observability ---
    observability_backend: Literal["wandb", "mlflow", "tensorboard", "none"] = Field(
        default="none"
    )
    observability_project: str = Field(default="provenir")


def load_run_config(path: str | Path) -> RunConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw_config: dict[str, Any] = yaml.safe_load(handle) or {}
    return RunConfig(**raw_config)
