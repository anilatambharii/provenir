from __future__ import annotations

from typing import Any, Mapping


class RewardRegistry:
    """A simple registry for named reward functions."""

    def __init__(self) -> None:
        self._rewards: dict[str, Any] = {}

    def register(self, name: str, reward: Any) -> None:
        self._rewards[name] = reward

    def get(self, name: str) -> Any:
        if name not in self._rewards:
            raise KeyError(f"reward {name!r} is not registered")
        return self._rewards[name]

    def score(self, name: str, trajectory: Mapping[str, Any]) -> float:
        result: float = self.get(name).score(trajectory)
        return result
