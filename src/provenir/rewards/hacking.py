from __future__ import annotations

from typing import Any, Mapping


class HackingDetector:
    """Heuristic checks for reward-gameable trajectories."""

    def detect(self, trajectory: Mapping[str, Any]) -> list[str]:
        issues: list[str] = []
        prediction = str(trajectory.get("prediction", ""))

        if len(prediction) > 200:
            issues.append("length_inflation")

        if prediction and prediction.isupper() and len(prediction) < 10:
            issues.append("format_exploit")

        return issues
