# Reward Validity

Benchmark gains alone cannot tell you whether your reward signal is valid.
Shao et al. 2026 (*Spurious Rewards*, arXiv:2506.10947) demonstrated this
starkly: GRPO on Qwen2.5-Math-7B gains **+21.4 pp on MATH-500 with entirely
random rewards** — only 7.7 pp below ground-truth rewards (+29.1 pp). On
Llama 3 and OLMo 2 the same random-reward run yields near-zero gain, exposing
a model-family dependency: some models improve under any reward because the
optimizer is amplifying **pretrained priors**, not learning from the signal.

Provenir's **reward-validity harness** exposes the methodology: run short
diagnostic trainings under a battery of degenerate reward controls and measure
whether improvement is exclusive to the real reward.

---

## The Ablation Battery

| Ablation | Reward definition | What it diagnoses |
|---|---|---|
| `real` | Your actual reward function | Baseline reference |
| `random` | `Random(seed)` uniform in [0, 1], ignores response | Shao's core control — gain here = spurious |
| `constant` | Fixed value (1.0) for every response | Advantage-collapse / prior amplification |
| `format_only` | 1.0 iff response matches a shallow format regex | Format-exploit gaming |
| `length_only` | Monotone in `len(response)` | Length-inflation reward hacking |
| `shuffled` | Real reward against a mismatched reference | Reference-binding validity |

**Validity score:**
`validity = clamp((Δ_real − max(Δ_random, Δ_constant, Δ_format, Δ_length)) / max(Δ_real, ε), 0, 1)`

Near 1 → the real reward drives improvement that degenerate signals cannot replicate.
Near 0 → degenerate signals match or beat the real reward (**spurious**).

The `spurious` hard-fail flag trips when any degenerate Δ ≥ Δ_real − tolerance.

---

## Python Usage

```python
from typing import Any
from provenir.core.abstractions import RewardFn
from provenir.environments.reward_validity import (
    RewardValidityHarness,
    RewardValidityReport,
    gate_reward_validity,
    RewardValidityBlocked,
    TrainEval,
)

# 1. Supply a TrainEval thunk that wires your training backend.
#    Signature: (RewardFn) -> (base_score, final_score, mean_reward, anomaly_counts)
#    In tests: use a synthetic closure.
#    In production: wire RLOrchestrator.run or equivalent.
def my_train_eval(reward: RewardFn) -> tuple[float, float, float, dict[str, int]]:
    # ... run a short diagnostic training, return metrics ...
    return base_score, final_score, mean_reward, {}

# 2. Evaluate your reward function.
harness = RewardValidityHarness(tolerance=0.1, seed=0)
report: RewardValidityReport = harness.evaluate(my_reward, my_train_eval)

print(report.summary())
# reward 'math': validity=0.934 [ok] real_gain=+0.300 (...)

# 3. Inspect the report.
for ablation, run in report.runs.items():
    print(f"  {ablation.value}: gain={run.gain:+.3f}")

# 4. Embed in a Passport (content-addressed).
passport_field = {
    "reward_validity": report.to_dict(),
    "reward_validity_hash": report.content_hash(),
}
```

The harness is **fully deterministic** given `seed`: two calls with the same
reward, `train_eval`, and seed produce identical reports and hashes.

---

## Promotion Gate

Use `gate_reward_validity` to block a model from moving to a protected stage
when the reward is spurious or below a validity floor. Protected stages are
`"production"`, `"prod"`, and `"release"` by default.

```python
from provenir.environments.reward_validity import gate_reward_validity, RewardValidityBlocked

try:
    gate_reward_validity(report, "production")                        # blocks if spurious
    gate_reward_validity(report, "production", min_validity=0.80)     # also enforce a floor
    print("Promotion: ALLOWED")
except RewardValidityBlocked as exc:
    print(f"Promotion: BLOCKED — {exc}")
```

`gate_reward_validity` is a no-op for non-protected stages (e.g. `"staging"`),
so you can call it unconditionally in CI.

Pair it with the verifier-reliability gate and the supply-chain scan gate for
a complete "signed trust before promotion" story:

```python
gate_promotion(reliability_report, stage)      # verifier reliable?
gate_reward_validity(validity_report, stage)   # reward valid?
scan_gate(scan_report, stage)                  # supply chain clean?
```

---

## What "synthetic backend" means

When you supply a stub `TrainEval` (as in the examples and tests), the harness
runs the methodology but the gain numbers are synthetic. The report correctly
reflects the logic of the ablation battery; the *quality* of the verdict scales
with the fidelity of the backend you wire in. This is the same approach as the
verifier-reliability harness, which documents that its default perturbations
target the free-form answer/number verifier family.

---

## Compliance Context

The reward-validity ablation battery supports two EU AI Act obligations for
high-risk and general-purpose AI systems:

- **Article 15 (Accuracy and robustness)**: requires documented evidence that
  high-risk AI systems achieve the claimed accuracy. A `RewardValidityReport`
  with per-ablation gains, a validity score, and a tamper-evident `content_hash`
  constitutes the required audit evidence.

- **Article 55(1)(a) (Adversarial testing for GPAI models)**: requires providers
  of general-purpose AI models to carry out documented adversarial testing. The
  ablation battery is documented adversarial testing of the reward function:
  each degenerate control is a controlled adversarial probe of whether the
  reward signal actually drives the claimed capability gain.

The `content_hash()` method produces a SHA-256 digest suitable for embedding
in a [Model Passport](governance.md), so adversarial testing results travel
with every promoted artifact.
