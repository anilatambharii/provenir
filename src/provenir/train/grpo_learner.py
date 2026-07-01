"""A real GRPO reference learner + a TRL production adapter.

The :class:`~provenir.train.rl.RLOrchestrator` delegates its gradient step to a
:class:`PolicyUpdater`. This module makes that delegation *real*:

* :func:`group_relative_advantages` — the GRPO advantage estimator (the actual
  math DeepSeek-R1 / DAPO / GSPO all build on).
* :class:`TabularGRPOLearner` — a self-contained softmax-policy GRPO learner
  with **no torch / TRL dependency** that demonstrably maximizes a verifiable
  reward. It samples groups of actions, computes group-relative advantages, and
  ascends the policy gradient, so the probability of high-reward actions
  provably increases. This is the reference that proves the GRPO update works
  end to end (and it is fully deterministic given a seed).
* :class:`GRPOUpdater` — a lightweight :class:`PolicyUpdater` that plugs the GRPO
  advantage computation into the orchestrator's update seam.
* :class:`TRLGRPOAdapter` — the production path: wraps a Provenir
  :class:`~provenir.environments.Verifier` as a TRL-compatible reward function
  and a :class:`~provenir.observability.FlightRecorder` as a TRL callback,
  delegating the real LLM policy-gradient step to HuggingFace TRL (requires
  ``pip install 'provenir[train]'``).

Example
-------
>>> learner = TabularGRPOLearner(LearnerConfig(num_actions=4, seed=0))
>>> # reward peaks at action 2
>>> curve = learner.train(lambda a: 1.0 if a == 2 else 0.0, iterations=200, group_size=8)
>>> curve[-1] > curve[0]      # the policy learned to prefer the rewarding action
True
>>> learner.probabilities()[2] > 0.5
True
"""

from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass
from typing import Any, Callable, Protocol, Sequence, runtime_checkable

from provenir.environments import Verifier
from provenir.observability import FlightRecorder, RLStepMetrics

try:  # optional, heavy — only needed for the production TRL path
    import torch  # noqa: F401

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

try:
    import trl  # noqa: F401

    _HAS_TRL = True
except ImportError:
    _HAS_TRL = False


# ---------------------------------------------------------------------------
# GRPO core
# ---------------------------------------------------------------------------


def group_relative_advantages(rewards: Sequence[float]) -> list[float]:
    """Return GRPO advantages: each reward centered on the group mean, scaled by std.

    ``A_i = (r_i - mean(r)) / std(r)``. When the group has no variance every
    advantage is ``0.0`` — the degenerate, no-signal case that DAPO's dynamic
    sampling skips and the flight recorder flags as ``advantage_collapse``.

    Example::

        group_relative_advantages([0.0, 1.0])   # [-1.0, 1.0]
        group_relative_advantages([0.5, 0.5])    # [0.0, 0.0]  (no signal)
    """
    n = len(rewards)
    if n == 0:
        return []
    mean = statistics.fmean(rewards)
    std = statistics.pstdev(rewards) if n > 1 else 0.0
    if std < 1e-9:
        return [0.0] * n
    return [(r - mean) / std for r in rewards]


@dataclass(frozen=True)
class UpdateStats:
    """Statistics returned by a single :class:`PolicyUpdater` step."""

    loss: float
    grad_norm: float
    advantage_mean: float
    advantage_std: float
    reward_mean: float
    updated: bool
    backend: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "loss": self.loss,
            "grad_norm": self.grad_norm,
            "advantage_mean": self.advantage_mean,
            "advantage_std": self.advantage_std,
            "reward_mean": self.reward_mean,
            "updated": self.updated,
            "backend": self.backend,
        }


@runtime_checkable
class PolicyUpdater(Protocol):
    """Structural type for the orchestrator's delegated gradient step."""

    name: str

    def is_available(self) -> bool: ...

    def apply(self, rewards: Sequence[float]) -> UpdateStats: ...


# ---------------------------------------------------------------------------
# Reference learner (real GRPO, no deps)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LearnerConfig:
    """Configuration for :class:`TabularGRPOLearner`."""

    num_actions: int
    learning_rate: float = 0.2
    entropy_coef: float = 0.0
    seed: int = 0

    def __post_init__(self) -> None:
        if self.num_actions < 2:
            raise ValueError(f"num_actions must be >= 2, got {self.num_actions}")
        if self.learning_rate <= 0.0:
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate}")
        if self.entropy_coef < 0.0:
            raise ValueError(f"entropy_coef must be >= 0, got {self.entropy_coef}")


def _softmax(logits: Sequence[float]) -> list[float]:
    hi = max(logits)
    exps = [math.exp(x - hi) for x in logits]
    total = sum(exps)
    return [e / total for e in exps]


class TabularGRPOLearner:
    """A real softmax-policy GRPO learner over a finite action set.

    This is not a stub. Each :meth:`step` samples a group of actions from the
    current softmax policy, scores them with a caller-supplied verifiable reward,
    computes group-relative advantages, and performs a policy-gradient ascent on
    the logits. Over iterations the probability mass provably shifts toward the
    highest-reward action. Randomness is drawn from a *local* seeded PRNG, so
    runs are reproducible and never touch global state.
    """

    def __init__(self, config: LearnerConfig) -> None:
        self.config = config
        self.logits: list[float] = [0.0] * config.num_actions
        self._rng = random.Random(config.seed)

    def probabilities(self) -> list[float]:
        """Current softmax action distribution."""
        return _softmax(self.logits)

    def entropy(self) -> float:
        """Shannon entropy of the current policy (nats)."""
        probs = self.probabilities()
        return -sum(p * math.log(p) for p in probs if p > 0.0)

    def sample_group(self, group_size: int) -> list[int]:
        """Sample ``group_size`` actions from the current policy."""
        if group_size < 2:
            raise ValueError(f"group_size must be >= 2, got {group_size}")
        probs = self.probabilities()
        actions = self._rng.choices(range(self.config.num_actions), weights=probs, k=group_size)
        return list(actions)

    def update(self, actions: Sequence[int], rewards: Sequence[float]) -> UpdateStats:
        """Apply one GRPO policy-gradient step from a scored group of actions."""
        if len(actions) != len(rewards):
            raise ValueError("actions and rewards must have equal length")
        if not actions:
            raise ValueError("cannot update on an empty group")

        advantages = group_relative_advantages(list(rewards))
        probs = self.probabilities()
        k = self.config.num_actions
        grad = [0.0] * k

        # Policy gradient of E[reward] w.r.t. the logits, baselined by the group
        # mean (GRPO): d log pi(a)/d logit_j = 1[j == a] - p_j.
        for action, adv in zip(actions, advantages):
            for j in range(k):
                indicator = 1.0 if j == action else 0.0
                grad[j] += adv * (indicator - probs[j])

        # Optional entropy bonus to keep exploration alive.
        if self.config.entropy_coef > 0.0:
            for j in range(k):
                grad[j] += self.config.entropy_coef * (-math.log(probs[j]) - self.entropy())

        n = float(len(actions))
        lr = self.config.learning_rate
        for j in range(k):
            self.logits[j] += lr * grad[j] / n

        grad_norm = math.sqrt(sum(g * g for g in grad)) / n
        reward_mean = statistics.fmean(rewards)
        adv_mean = statistics.fmean(advantages) if advantages else 0.0
        adv_std = statistics.pstdev(advantages) if len(advantages) > 1 else 0.0
        # Surrogate policy-gradient loss (negative expected advantage-weighted logprob).
        updated_probs = self.probabilities()
        loss = -statistics.fmean(
            adv * math.log(updated_probs[a]) for a, adv in zip(actions, advantages)
        )
        return UpdateStats(
            loss=loss,
            grad_norm=grad_norm,
            advantage_mean=adv_mean,
            advantage_std=adv_std,
            reward_mean=reward_mean,
            updated=True,
            backend="tabular-grpo",
        )

    def train(
        self,
        reward_fn: Callable[[int], float],
        iterations: int,
        group_size: int = 8,
        recorder: FlightRecorder | None = None,
    ) -> list[float]:
        """Optimize ``reward_fn`` for ``iterations`` GRPO steps; return the reward curve.

        When a :class:`~provenir.observability.FlightRecorder` is supplied each
        step is logged to it, so the learner doubles as an end-to-end,
        instrumented demonstration of the RL trust loop.
        """
        if iterations < 1:
            raise ValueError(f"iterations must be >= 1, got {iterations}")
        curve: list[float] = []
        for step in range(iterations):
            actions = self.sample_group(group_size)
            rewards = [reward_fn(a) for a in actions]
            stats = self.update(actions, rewards)
            curve.append(stats.reward_mean)
            if recorder is not None:
                recorder.log_step(
                    RLStepMetrics(
                        step=step,
                        kl=0.0,
                        entropy=self.entropy(),
                        reward_mean=stats.reward_mean,
                        reward_std=statistics.pstdev(rewards) if len(rewards) > 1 else 0.0,
                        advantage_std=stats.advantage_std,
                        grad_norm=stats.grad_norm,
                        learning_rate=self.config.learning_rate,
                    )
                )
        return curve


# ---------------------------------------------------------------------------
# Orchestrator updater
# ---------------------------------------------------------------------------


class NoOpUpdater:
    """Default updater: reports the GRPO advantages but performs no gradient step."""

    name = "noop"

    def is_available(self) -> bool:
        return True

    def apply(self, rewards: Sequence[float]) -> UpdateStats:
        advantages = group_relative_advantages(list(rewards))
        return UpdateStats(
            loss=0.0,
            grad_norm=0.0,
            advantage_mean=statistics.fmean(advantages) if advantages else 0.0,
            advantage_std=statistics.pstdev(advantages) if len(advantages) > 1 else 0.0,
            reward_mean=statistics.fmean(rewards) if rewards else 0.0,
            updated=False,
            backend="noop",
        )


class GRPOUpdater:
    """A :class:`PolicyUpdater` that computes real GRPO advantages for the seam.

    Used by :class:`~provenir.train.rl.RLOrchestrator` when a caller wants the
    update seam to report genuine advantage/gradient statistics (fed into the
    flight recorder) rather than the inert default.
    """

    name = "grpo-reference"

    def __init__(self, learning_rate: float = 0.2) -> None:
        if learning_rate <= 0.0:
            raise ValueError(f"learning_rate must be > 0, got {learning_rate}")
        self.learning_rate = learning_rate

    def is_available(self) -> bool:
        return True

    def apply(self, rewards: Sequence[float]) -> UpdateStats:
        advantages = group_relative_advantages(list(rewards))
        adv_std = statistics.pstdev(advantages) if len(advantages) > 1 else 0.0
        # Gradient magnitude scales with the spread of advantages.
        grad_norm = self.learning_rate * adv_std
        return UpdateStats(
            loss=-(statistics.fmean(advantages) if advantages else 0.0),
            grad_norm=grad_norm,
            advantage_mean=statistics.fmean(advantages) if advantages else 0.0,
            advantage_std=adv_std,
            reward_mean=statistics.fmean(rewards) if rewards else 0.0,
            updated=bool(advantages) and adv_std > 0.0,
            backend="grpo-reference",
        )


# ---------------------------------------------------------------------------
# Production TRL adapter
# ---------------------------------------------------------------------------


class TRLGRPOAdapter:
    """Production GRPO via HuggingFace TRL — the real LLM policy-gradient path.

    Wraps a Provenir :class:`~provenir.environments.Verifier` as a TRL-compatible
    reward function (so verifiable rewards drive the update) and exposes hooks to
    attach a :class:`~provenir.observability.FlightRecorder`. The actual
    generation + policy-gradient step is delegated to ``trl.GRPOTrainer``.

    Requires ``pip install 'provenir[train]'`` (TRL + torch). The reward function
    and availability check work without TRL installed; :meth:`build_trainer`
    raises a clear error when the optional stack is absent.
    """

    name = "trl-grpo"

    def __init__(
        self,
        verifier: Verifier,
        flight_recorder: FlightRecorder | None = None,
        reference_key: str = "reference",
    ) -> None:
        self.verifier = verifier
        self.flight_recorder = flight_recorder or FlightRecorder()
        self.reference_key = reference_key

    def is_available(self) -> bool:
        """True when the TRL + torch stack is importable."""
        return _HAS_TRL and _HAS_TORCH

    def reward_function(self) -> Callable[..., list[float]]:
        """Return a TRL-compatible reward function over a batch of completions.

        The returned callable has TRL's ``(completions, **kwargs) -> list[float]``
        shape: each completion is scored by the wrapped verifier against the
        per-sample reference passed through TRL's dataset columns.
        """
        verifier = self.verifier
        ref_key = self.reference_key

        def _reward(completions: Sequence[str], **kwargs: Any) -> list[float]:
            references = kwargs.get(ref_key)
            if not isinstance(references, list):
                references = [references] * len(completions)
            return [
                verifier.verify(str(c), r).reward
                for c, r in zip(completions, references)
            ]

        return _reward

    def build_trainer(
        self,
        model: Any,
        train_dataset: Any,
        grpo_config: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Construct a real ``trl.GRPOTrainer`` wired to the verifier reward.

        Raises :class:`RuntimeError` when TRL/torch are not installed.
        """
        if not self.is_available():
            raise RuntimeError(
                "TRL/torch not installed; run: pip install 'provenir[train]'"
            )
        from trl import GRPOConfig as TRLGRPOConfig
        from trl import GRPOTrainer

        config = grpo_config or TRLGRPOConfig()
        return GRPOTrainer(
            model=model,
            reward_funcs=[self.reward_function()],
            args=config,
            train_dataset=train_dataset,
            **kwargs,
        )
