"""Training-Inference Mismatch (TIM) detector demo.

Demonstrates two scenarios:

1. **Matched distributions**: training-time and inference-time log-probs are
   virtually identical.  ``gate_tim`` allows promotion to production.

2. **Diverged distributions**: training concentrates probability mass on
   different tokens than inference.  ``gate_tim`` blocks production but allows
   the non-protected ``"staging"`` stage.

Run::

    python examples/tim_demo.py

No real model backend is required; ``TIMProbeFn`` closures are fully synthetic.
"""

from __future__ import annotations

from provenir.environments.tim import (
    TIMBlocked,
    TIMDetector,
    gate_tim,
)

__all__: list[str] = []


# ---------------------------------------------------------------------------
# Scenario 1 — matched distributions
# ---------------------------------------------------------------------------


def matched_probe_fn(prompt: str) -> tuple[list[float], list[float]]:
    """Training and inference produce the same per-token log-probs."""
    train_lp = [-0.5, -1.2, -0.8, -2.1, -0.3]
    inf_lp = [-0.5, -1.2, -0.8, -2.1, -0.3]
    return train_lp, inf_lp


# ---------------------------------------------------------------------------
# Scenario 2 — diverged distributions
# ---------------------------------------------------------------------------


def diverged_probe_fn(prompt: str) -> tuple[list[float], list[float]]:
    """Training concentrates on token 0; inference concentrates on token 4.

    This simulates a model that learned one sampling strategy during training
    but uses a different one at inference time (e.g. greedy vs. sampling,
    or different temperature/top-p settings).
    """
    train_lp = [0.0, -100.0, -100.0, -100.0, -100.0]
    inf_lp = [-100.0, -100.0, -100.0, -100.0, 0.0]
    return train_lp, inf_lp


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    detector = TIMDetector(threshold=0.1)
    prompts = ["What is 2+2?", "Summarise the paper.", "Translate to French."]

    # -- Scenario 1: matched ------------------------------------------------
    print("=" * 60)
    print("Scenario 1: MATCHED distributions")
    print("=" * 60)
    matched_report = detector.detect(matched_probe_fn, prompts)
    print(matched_report.summary())
    print(f"  mean_kl          : {matched_report.mean_kl:.6f}")
    print(f"  mismatch_detected: {matched_report.mismatch_detected}")
    print(f"  content_hash     : {matched_report.content_hash()[:16]}...")
    print()
    try:
        gate_tim(matched_report, "production")
        print("  gate_tim('production') -> ALLOWED")
    except TIMBlocked as exc:
        print(f"  gate_tim('production') -> BLOCKED: {exc}")

    # -- Scenario 2: diverged -----------------------------------------------
    print()
    print("=" * 60)
    print("Scenario 2: DIVERGED distributions")
    print("=" * 60)
    diverged_report = detector.detect(diverged_probe_fn, prompts)
    print(diverged_report.summary())
    print(f"  mean_kl          : {diverged_report.mean_kl:.4f}")
    print(f"  max_kl           : {diverged_report.max_kl:.4f}")
    print(f"  mismatch_rate    : {diverged_report.mismatch_rate:.3f}")
    print(f"  mismatch_detected: {diverged_report.mismatch_detected}")
    print(f"  content_hash     : {diverged_report.content_hash()[:16]}...")
    print()
    try:
        gate_tim(diverged_report, "production")
        print("  gate_tim('production') -> ALLOWED")
    except TIMBlocked as exc:
        print("  gate_tim('production') -> BLOCKED")
        print(f"    {exc}")

    print()
    print("Non-protected stage 'staging' is never blocked:")
    gate_tim(diverged_report, "staging")
    print("  gate_tim('staging') -> ALLOWED (even though mismatch_detected=True)")


if __name__ == "__main__":
    main()
