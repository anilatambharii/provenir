from __future__ import annotations

from typing import Any, Mapping


class ExactMatchReward:
    """Reward 1.0 when prediction exactly matches the reference."""

    def score(self, trajectory: Mapping[str, Any]) -> float:
        prediction = str(trajectory.get("prediction", ""))
        reference = str(trajectory.get("reference", ""))
        return 1.0 if prediction == reference else 0.0


class FormatReward:
    """Reward based on simple length/format constraints."""

    def __init__(self, min_length: int = 1) -> None:
        self.min_length = min_length

    def score(self, trajectory: Mapping[str, Any]) -> float:
        prediction = str(trajectory.get("prediction", ""))
        return 1.0 if len(prediction) >= self.min_length else 0.0
