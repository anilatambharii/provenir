from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DPOConfig:
    """Configuration for a minimal DPO-style preference workflow."""

    beta: float = 0.1
    max_steps: int = 10


class DPOTrainer:
    """Lightweight DPO trainer stub — wires to a real backend in a future increment."""

    def __init__(self, config: DPOConfig) -> None:
        self.config = config

    def train(self, preferences: list[dict[str, Any]]) -> dict[str, Any]:
        return {"preference_count": len(preferences), "beta": self.config.beta}


# ---------------------------------------------------------------------------
# GRPO — Group Relative Policy Optimization
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GRPOConfig:
    """Configuration for Group Relative Policy Optimization.

    GRPO scores a *group* of candidate responses per prompt relative to each
    other, removing the need for a separate critic/value network.
    """

    kl_coef: float = 0.05
    group_size: int = 8
    clip_ratio: float = 0.2
    max_steps: int = 100


class GRPOTrainer:
    """Lightweight GRPO stub — interface contract for backend integration."""

    def __init__(self, config: GRPOConfig) -> None:
        self.config = config

    def train(self, groups: list[dict[str, Any]]) -> dict[str, Any]:
        """Each element of *groups* is a dict with keys ``prompt`` and ``responses``."""
        return {
            "group_count": len(groups),
            "kl_coef": self.config.kl_coef,
            "group_size": self.config.group_size,
            "clip_ratio": self.config.clip_ratio,
        }


# ---------------------------------------------------------------------------
# PPO — Proximal Policy Optimization
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PPOConfig:
    """Configuration for Proximal Policy Optimization (classic RLHF).

    Uses a clipped surrogate objective with a GAE advantage estimator.
    """

    kl_coef: float = 0.2
    clip_ratio: float = 0.2
    vf_coef: float = 0.5
    max_steps: int = 100
    gamma: float = 1.0
    lam: float = 0.95


class PPOTrainer:
    """Lightweight PPO stub — interface contract for backend integration."""

    def __init__(self, config: PPOConfig) -> None:
        self.config = config

    def train(self, episodes: list[dict[str, Any]]) -> dict[str, Any]:
        """Each element of *episodes* is a rollout dict with ``states`` and ``rewards``."""
        return {
            "episode_count": len(episodes),
            "clip_ratio": self.config.clip_ratio,
            "kl_coef": self.config.kl_coef,
            "vf_coef": self.config.vf_coef,
        }
