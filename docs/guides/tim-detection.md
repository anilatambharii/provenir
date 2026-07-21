# Training-Inference Mismatch (TIM) Detection

## The Problem

After RLVR (Reinforcement Learning from Verifiable Rewards) training, a model's
learned policy may diverge from its inference-time behaviour. This divergence is
called a **Training-Inference Mismatch (TIM)** and was characterised by Zhong et
al. (2026) in the context of OpenRLHF (issue #1108).

Common causes include:

- **Different sampling strategies**: the training rollout uses sampling with
  temperature *T* while inference uses greedy decoding or a different temperature.
- **Batching artefacts**: per-sample normalisation during training (e.g. per-group
  advantage normalisation in GRPO) introduces a batch-size dependency that
  disappears at inference time.
- **Context length handling**: a long prompt that is truncated during training but
  not at inference (or vice versa) shifts the token-level distribution.
- **KV-cache behaviour**: cached states during multi-turn inference may not match
  the fresh-context rollouts used during training.

When TIM is present, benchmark gains measured during training may not transfer to
production: the model optimised a subtly different objective than the one it will
encounter at deployment.

## How the Harness Works

Provenir's TIM detector measures the **KL divergence** between the training-time
and inference-time token-level probability distributions on a held-out probe set.

```
KL(train || inference) = Σ_i  p_train(i) · log( p_train(i) / p_inf(i) )
```

A value near zero means the two distributions are essentially the same; a large
value means the model has learned a policy that does not faithfully reproduce
itself at inference time.

The detector is **backend-agnostic**: Provenir defines only the measurement
methodology and the data structures. You wire your own training rollout and
inference server via the `TIMProbeFn` contract — no dependency on any particular
training framework.

### Numerical Stability

Raw log-probabilities are converted to probability distributions via
**log-sum-exp normalisation** (softmax in log-space), which avoids overflow or
underflow when logits have large magnitude. Every probability is then clamped to
a configurable `min_prob` floor (default `1e-9`) to prevent `log(0)` crashes when
one distribution assigns zero mass to a token that the other does not.

## The `TIMProbeFn` Contract

```python
from provenir.environments.tim import TIMProbeFn

def my_probe_fn(prompt: str) -> tuple[list[float], list[float]]:
    """
    Given a prompt, return:
      - train_log_probs: per-token log-probabilities from the training rollout
      - inf_log_probs:   per-token log-probabilities from the inference server

    Both lists must cover the *same* token sequence and should be the same
    length.  If they differ, the detector truncates to the shorter list.
    """
    train_log_probs = [...]  # from your training engine
    inf_log_probs   = [...]  # from your inference server
    return train_log_probs, inf_log_probs
```

Key invariants:
- The lists represent **the same prompt completion** decoded by both the training
  rollout and the inference server.
- Log-probabilities are natural-log (i.e. `log_prob = log P(token | context)`).
- Lists need not be normalised; the detector applies softmax normalisation
  internally.

## Running the Detector

```python
from provenir.environments.tim import TIMDetector, gate_tim

detector = TIMDetector(
    threshold=0.1,   # per-probe KL above which a probe is a mismatch
    min_prob=1e-9,   # probability floor for numerical stability
)

probes = ["Summarise this paper.", "Solve: 2+2=?", "Translate to French."]
report = detector.detect(my_probe_fn, probes)

print(report.summary())
# TIM [ok]: probes=3 mean_kl=0.0012 max_kl=0.0015 mismatch_rate=0.000

# Gate: blocks promotion to production if mismatch detected.
gate_tim(report, "production")
```

### `TIMReport` fields

| Field | Type | Meaning |
|---|---|---|
| `probe_count` | `int` | Number of prompts evaluated |
| `mean_kl` | `float` | Mean KL divergence across all probes |
| `max_kl` | `float` | Maximum per-probe KL |
| `mismatch_rate` | `float` | Fraction of probes with KL > threshold |
| `mismatch_detected` | `bool` | `True` when `mean_kl > threshold` |
| `results` | `list[TIMResult]` | Per-probe breakdown |

### Signing into a Passport

`TIMReport.content_hash()` returns a SHA-256 hex digest of the report's
canonical JSON serialisation. This can be embedded in a Provenir Passport to
provide a tamper-evident audit trail.

```python
passport_entry = {
    "tim_hash": report.content_hash(),
    "tim_report": report.to_dict(),
}
```

## The `gate_tim` Promotion Gate

```python
gate_tim(report, "staging")                        # never blocks
gate_tim(report, "production")                     # blocks if mismatch_detected
gate_tim(report, "release", max_mean_kl=0.05)      # also enforces a KL ceiling
```

- **Non-protected stages** (e.g. `"staging"`, `"dev"`) are never blocked.
- **Protected stages** (`"production"`, `"prod"`, `"release"`) are blocked when
  `report.mismatch_detected is True`.
- The optional `max_mean_kl` parameter adds a hard ceiling on `mean_kl`
  regardless of the threshold-based `mismatch_detected` flag.

`TIMBlocked` (a `RuntimeError` subclass) is raised with a descriptive message
when promotion is blocked.

## Wiring to `RLOrchestrator` Rollout Log-Probs

In a typical OpenRLHF-style pipeline the training rollout stores per-token
log-probs in the experience buffer. To wire these into the TIM detector:

```python
import functools
from provenir.environments.tim import TIMDetector, gate_tim

def make_probe_fn(orchestrator, inference_engine):
    """Closure that queries both the rollout buffer and the inference server."""
    def probe_fn(prompt: str) -> tuple[list[float], list[float]]:
        # 1. Get log-probs from the most-recent training rollout for this prompt.
        train_lp = orchestrator.get_rollout_log_probs(prompt)
        # 2. Get log-probs from the live inference server (greedy or sampled).
        inf_lp = inference_engine.get_log_probs(prompt)
        return train_lp, inf_lp
    return probe_fn

detector = TIMDetector(threshold=0.1)
report = detector.detect(
    make_probe_fn(my_orchestrator, my_inference_engine),
    held_out_prompts,
)
gate_tim(report, stage="production")
```

## References

- Zhong et al. (2026). *Training-Inference Mismatch in RLVR*. OpenRLHF issue #1108.
- Shao et al. (2026). *Spurious Rewards: Rethinking Training Signals for Informal Mathematical Reasoning*. arXiv:2506.10947.
