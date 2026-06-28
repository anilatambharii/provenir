from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CostBreakdown:
    unit_cost: float
    multiplier: float
    steps: int
    hours: float
    total_cost: float


class CostEstimator:
    """Estimate compute cost with simple unit-cost assumptions."""

    def __init__(self, unit_cost: float, multiplier: float = 1.0) -> None:
        self.unit_cost = unit_cost
        self.multiplier = multiplier

    def estimate_run_cost(self, steps: int, hours: float) -> CostBreakdown:
        total_cost = max(0.0, self.unit_cost * self.multiplier * steps * hours)
        return CostBreakdown(
            unit_cost=self.unit_cost,
            multiplier=self.multiplier,
            steps=steps,
            hours=hours,
            total_cost=total_cost,
        )
