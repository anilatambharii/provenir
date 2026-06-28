from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PEFTConfig:
    """LoRA / QLoRA configuration for parameter-efficient fine-tuning.

    *rank* and *alpha* control the LoRA update matrices:
    - W_update = B @ A * (alpha / rank)   (standard LoRA)
    - W_update = B @ A * (alpha / sqrt(rank))  (rsLoRA, ``use_rslora=True``)
    """

    rank: int = 16
    alpha: int = 32
    target_modules: tuple[str, ...] = ("q_proj", "v_proj")
    dropout: float = 0.05
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    use_rslora: bool = False
    lora_bias: str = "none"  # "none" | "all" | "lora_only"

    def __post_init__(self) -> None:
        if self.rank < 1:
            raise ValueError(f"rank must be >= 1, got {self.rank}")
        if self.alpha <= 0:
            raise ValueError(f"alpha must be > 0, got {self.alpha}")
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if self.load_in_4bit and self.load_in_8bit:
            raise ValueError("load_in_4bit and load_in_8bit are mutually exclusive")
        valid_bias = {"none", "all", "lora_only"}
        if self.lora_bias not in valid_bias:
            raise ValueError(
                f"lora_bias must be one of {sorted(valid_bias)}, got {self.lora_bias!r}"
            )

    @property
    def scaling(self) -> float:
        """Effective LoRA scaling factor (α/r or α/√r for rsLoRA)."""
        if self.use_rslora:
            return float(self.alpha) / math.sqrt(float(self.rank))
        return float(self.alpha) / float(self.rank)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "alpha": self.alpha,
            "target_modules": list(self.target_modules),
            "dropout": self.dropout,
            "load_in_4bit": self.load_in_4bit,
            "load_in_8bit": self.load_in_8bit,
            "use_rslora": self.use_rslora,
            "lora_bias": self.lora_bias,
            "scaling": self.scaling,
        }
