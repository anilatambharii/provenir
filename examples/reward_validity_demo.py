"""Spurious-reward ablation demo.

Demonstrates two scenarios:

1. **Valid reward**: the real reward drives measurable improvement that no
   degenerate control replicates.  ``gate_reward_validity`` allows promotion.

2. **Spurious reward** (Shao repro): the random-reward control achieves the
   same gain as the real reward — the optimizer is amplifying pretrained priors,
   not learning from the signal.  ``gate_reward_validity`` blocks production.

Run::

    python examples/reward_validity_demo.py

No training backend is required; ``TrainEval`` closures are fully synthetic.
"""

from __future__ import annotations

from typing import Any, Mapping

from provenir.core.abstractions import RewardFn
from provenir.environments.reward_validity import (
    RewardValidityBlocked,
    RewardValidityHarness,
    gate_reward_validity,
)

__all__: list[str] = []


# ---------------------------------------------------------------------------
# Minimal real reward (exact-match)
# ---------------------------------------------------------------------------


class ExactMatchReward(RewardFn):
    """1.0 if prediction == reference, else 0.0."""

    kind: str = "real"
    name: str = "exact_match"

    def score(self, trajectory: Mapping[str, Any]) -> float:
        return 1.0 if trajectory.get("prediction") == trajectory.get("reference") else 0.0

    def gameability_check(self, trajectory: Mapping[str, Any]) -> list[str]:
        return []


# ---------------------------------------------------------------------------
# Scenario 1 — valid reward
# ---------------------------------------------------------------------------


def valid_train_eval(reward: RewardFn) -> tuple[float, float, float, dict[str, int]]:
    """Synthetic: real reward yields a large gain; degenerate rewards yield ~0."""
    base = 0.40
    kind = getattr(reward, "kind", "")
    gain = 0.30 if kind == "real" else 0.02
    mean_rwd = 0.85 if kind == "real" else 0.50
    return base, base + gain, mean_rwd, {}


# ---------------------------------------------------------------------------
# Scenario 2 — spurious reward (Shao et al. 2026 repro)
# ---------------------------------------------------------------------------


def spurious_train_eval(reward: RewardFn) -> tuple[float, float, float, dict[str, int]]:
    """Synthetic Shao repro: random gain ≈ real gain — reward is spurious."""
    base = 0.40
    kind = getattr(reward, "kind", "")
    if kind == "real":
        gain = 0.25
    elif kind == "random":
        # Within default tolerance=0.1 of real gain → spurious flag trips.
        gain = 0.24
    else:
        gain = 0.02
    mean_rwd = 0.80 if kind == "real" else 0.50
    return base, base + gain, mean_rwd, {}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    harness = RewardValidityHarness(seed=0)
    real_reward = ExactMatchReward()

    # -- Scenario 1: valid --------------------------------------------------
    print("=" * 60)
    print("Scenario 1: VALID reward")
    print("=" * 60)
    valid_report = harness.evaluate(real_reward, valid_train_eval)
    print(valid_report.summary())
    print(f"  validity  : {valid_report.validity:.3f}")
    print(f"  spurious  : {valid_report.spurious}")
    print(f"  hash      : {valid_report.content_hash()[:16]}...")
    print()
    try:
        gate_reward_validity(valid_report, "production")
        print("  gate_reward_validity('production') -> ALLOWED")
    except RewardValidityBlocked as exc:
        print(f"  gate_reward_validity('production') -> BLOCKED: {exc}")

    # -- Scenario 2: spurious -----------------------------------------------
    print()
    print("=" * 60)
    print("Scenario 2: SPURIOUS reward  (Shao et al. 2026 repro)")
    print("=" * 60)
    spurious_report = harness.evaluate(real_reward, spurious_train_eval)
    print(spurious_report.summary())
    print(f"  validity  : {spurious_report.validity:.3f}")
    print(f"  spurious  : {spurious_report.spurious}")
    print(f"  hash      : {spurious_report.content_hash()[:16]}...")
    print()
    try:
        gate_reward_validity(spurious_report, "production")
        print("  gate_reward_validity('production') -> ALLOWED")
    except RewardValidityBlocked as exc:
        print("  gate_reward_validity('production') -> BLOCKED")
        print(f"    {exc}")

    print()
    print("Non-protected stage 'staging' is never blocked:")
    gate_reward_validity(spurious_report, "staging")
    print("  gate_reward_validity('staging') -> ALLOWED (even though spurious)")


if __name__ == "__main__":
    main()
