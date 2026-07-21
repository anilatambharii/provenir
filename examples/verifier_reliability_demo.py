"""Stress-test a verifier for reliability, then gate a production promotion.

Reward-based RL (RLVR) is only as trustworthy as its verifier. If the verifier is
non-deterministic, brittle, or gameable, the run silently optimizes garbage. The
:class:`ReliabilityHarness` applies metamorphic + perturbation testing across six
failure modes (consistency, invariance, sensitivity, gameability, monotonicity,
boundary) and produces a signed-ready :class:`ReliabilityReport`.

Run::

    python examples/verifier_reliability_demo.py
"""

from __future__ import annotations

from provenir.environments import (
    ExactAnswerVerifier,
    Probe,
    PromotionBlocked,
    ReliabilityHarness,
    gate_promotion,
)
from provenir.environments.base import VerificationResult, Verifier


class LengthRewardVerifier:
    """A deliberately naive reward: 'longer is better'. Trivially gameable."""

    name = "length_reward"

    def verify(self, response: str, reference: object) -> VerificationResult:
        passed = len(response) > 50
        return VerificationResult(
            passed=passed,
            reward=1.0 if passed else 0.0,
            detail=f"len={len(response)}",
        )


def evaluate(label: str, verifier: Verifier, probes: list[Probe]) -> None:
    report = ReliabilityHarness().evaluate(verifier, probes)
    print(f"\n=== {label} ===")
    print(report.summary())
    for outcome in report.failures():
        print(f"  FAIL [{outcome.mode.value}] {outcome.detail} ({outcome.origin_label})")
    try:
        gate_promotion(report, "production")
        print("  promotion to 'production': ALLOWED")
    except PromotionBlocked as exc:
        print(f"  promotion to 'production': BLOCKED — {exc}")


def main() -> None:
    probes = [
        Probe("The answer is \\boxed{42}", "42", should_pass=True, label="cat"),
        Probe("\\boxed{7}", "7", should_pass=True, label="num"),
    ]

    # A real, well-behaved verifier: high reliability, no hard-fail, promotion allowed.
    evaluate("ExactAnswerVerifier (sound)", ExactAnswerVerifier(), probes)

    # A gameable reward: the harness surfaces the hard-fail and blocks production.
    evaluate("LengthRewardVerifier (gameable)", LengthRewardVerifier(), probes)


if __name__ == "__main__":
    main()
