from __future__ import annotations

from typing import Any, Mapping

from provenir.core.abstractions import RewardFn


class WeightedSumReward(RewardFn):
    """Normalised weighted average of component reward functions.

    Weights need not sum to 1 — they are normalised internally so the result
    stays in [0, 1] as long as each component also returns values in [0, 1].

    Useful for blending orthogonal reward signals::

        reward = WeightedSumReward([
            (ExactMatchReward(), 0.6),
            (FormatReward(min_length=5), 0.4),
        ])
    """

    kind: str = "composite"

    def __init__(self, rewards: list[tuple[RewardFn, float]]) -> None:
        if not rewards:
            raise ValueError("rewards list must not be empty")
        self.rewards = list(rewards)

    def score(self, trajectory: Mapping[str, Any]) -> float:
        total_weight = sum(w for _, w in self.rewards)
        if total_weight == 0.0:
            return 0.0
        return sum(r.score(trajectory) * w for r, w in self.rewards) / total_weight

    def gameability_check(self, trajectory: Mapping[str, Any]) -> list[str]:
        issues: list[str] = []
        for r, _ in self.rewards:
            issues.extend(r.gameability_check(trajectory))
        return issues


class MinReward(RewardFn):
    """Bottleneck reward — returns the lowest score across all components.

    Useful when every sub-reward represents a necessary condition, e.g.
    correctness AND safety must both be high.
    """

    kind: str = "composite"

    def __init__(self, rewards: list[RewardFn]) -> None:
        if not rewards:
            raise ValueError("rewards list must not be empty")
        self.rewards = list(rewards)

    def score(self, trajectory: Mapping[str, Any]) -> float:
        return min(r.score(trajectory) for r in self.rewards)

    def gameability_check(self, trajectory: Mapping[str, Any]) -> list[str]:
        issues: list[str] = []
        for r in self.rewards:
            issues.extend(r.gameability_check(trajectory))
        return issues


class MaxReward(RewardFn):
    """Best-of-N reward — returns the highest score across all components.

    Useful when any one of several equivalent reward signals suffices, e.g.
    matching the reference via exact match OR paraphrase.
    """

    kind: str = "composite"

    def __init__(self, rewards: list[RewardFn]) -> None:
        if not rewards:
            raise ValueError("rewards list must not be empty")
        self.rewards = list(rewards)

    def score(self, trajectory: Mapping[str, Any]) -> float:
        return max(r.score(trajectory) for r in self.rewards)

    def gameability_check(self, trajectory: Mapping[str, Any]) -> list[str]:
        issues: list[str] = []
        for r in self.rewards:
            issues.extend(r.gameability_check(trajectory))
        return issues


class ThresholdGatedReward(RewardFn):
    """Hard-constraint gate — returns the inner score only if it meets *threshold*.

    If the inner score is below the threshold the gate returns 0.0,
    regardless of any outer composition.  Useful for safety checks::

        safe_and_correct = WeightedSumReward([
            (ThresholdGatedReward(safety_reward, threshold=0.8), 1.0),
            (correctness_reward, 1.0),
        ])
    """

    kind: str = "gated"

    def __init__(self, inner: RewardFn, threshold: float = 0.5) -> None:
        if not (0.0 <= threshold <= 1.0):
            raise ValueError(f"threshold must be in [0, 1], got {threshold}")
        self.inner = inner
        self.threshold = threshold

    def score(self, trajectory: Mapping[str, Any]) -> float:
        s = self.inner.score(trajectory)
        return s if s >= self.threshold else 0.0

    def gameability_check(self, trajectory: Mapping[str, Any]) -> list[str]:
        return self.inner.gameability_check(trajectory)


class ClampedReward(RewardFn):
    """Clamps the inner reward's output to [*lo*, *hi*].

    Useful for normalising a reward that may return values outside [0, 1],
    or for putting a floor/ceiling on a trusted component::

        reward = ClampedReward(my_reward, lo=0.1, hi=0.9)
    """

    kind: str = "clamped"

    def __init__(self, inner: RewardFn, lo: float = 0.0, hi: float = 1.0) -> None:
        if lo > hi:
            raise ValueError(f"lo ({lo}) must be <= hi ({hi})")
        self.inner = inner
        self.lo = lo
        self.hi = hi

    def score(self, trajectory: Mapping[str, Any]) -> float:
        return max(self.lo, min(self.hi, self.inner.score(trajectory)))

    def gameability_check(self, trajectory: Mapping[str, Any]) -> list[str]:
        return self.inner.gameability_check(trajectory)
