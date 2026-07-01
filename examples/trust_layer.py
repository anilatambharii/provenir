"""
The Provenir Trust Layer — drop provenance, contamination-safe eval,
reward-hacking detection, and a signed Model Passport into any run.

Everything here is pure-Python (no GPU, no optional deps required).

Usage:
    python examples/trust_layer.py
"""

from __future__ import annotations

import provenir
from provenir.data.dataset import JsonlDataset
from provenir.environments import MathVerifier
from provenir.eval.contamination import ContaminationChecker, ContaminationConfig
from provenir.observability import FlightRecorder, RewardHackingDetector, RLStepMetrics
from provenir.train.algorithms import GRPOConfig
from provenir.train.rl import RLOrchestrator

# ---------------------------------------------------------------------------
# 1. Contamination firewall — is my held-out eval actually held out?
# ---------------------------------------------------------------------------

train = JsonlDataset.from_records([
    {"prompt": "what is 2 + 2", "response": "4", "reference": "4"},
    {"prompt": "what is 3 + 5", "response": "8", "reference": "8"},
])
eval_ds = JsonlDataset.from_records([
    {"prompt": "what is 10 + 15", "response": "25", "reference": "25"},
])

report = ContaminationChecker(ContaminationConfig(method="ngram")).check_datasets(
    train, eval_ds
)
print(f"Contamination: {'CLEAN' if report.is_clean else 'CONTAMINATED'} "
      f"(rate={report.contamination_rate:.3f})")

# ---------------------------------------------------------------------------
# 2. RL Flight Recorder — the black box for RL runs
# ---------------------------------------------------------------------------

recorder = FlightRecorder()
# A healthy step vs a degenerate one (advantage collapse + entropy collapse):
recorder.log_step(RLStepMetrics(step=0, kl=0.02, entropy=2.5, reward_mean=0.4,
                                reward_std=0.3, advantage_std=0.3, grad_norm=1.2))
anomalies = recorder.log_step(
    RLStepMetrics(step=1, kl=0.03, entropy=0.01, reward_mean=0.4,
                  reward_std=0.0, advantage_std=0.0, grad_norm=1.1)
)
print(f"\nFlight recorder caught {len(anomalies)} anomalies on step 1:")
for a in anomalies:
    print(f"  [{a.severity}] {a.kind}: {a.detail}")

# ---------------------------------------------------------------------------
# 3. Reward-hacking detection
# ---------------------------------------------------------------------------

detector = RewardHackingDetector()
signals = detector.detect({
    "prediction": "import sys; sys.exit(0)  # skip the failing tests",
    "proxy_reward": 1.0,
    "true_reward": 0.0,
})
print(f"\nReward-hacking signals: {[s.kind for s in signals]}")

# ---------------------------------------------------------------------------
# 4. Verifiable-reward RL with GRPO (reward + observability are real)
# ---------------------------------------------------------------------------

orchestrator = RLOrchestrator(
    algorithm=GRPOConfig(group_size=4, max_steps=1),
    verifier=MathVerifier(),
    flight_recorder=FlightRecorder(),
    hacking_detector=RewardHackingDetector(),
)
rl_result = orchestrator.run(train_dataset=train)
print(f"\nGRPO run: {rl_result.steps_completed} steps, "
      f"mean_reward={rl_result.mean_reward:.3f}, "
      f"anomalies={rl_result.anomaly_count}")

# ---------------------------------------------------------------------------
# 5. The whole thing in 3 lines — import provenir + a signed Model Passport
# ---------------------------------------------------------------------------

with provenir.track(
    "trust-layer-demo",
    dataset=train,
    base_model="meta-llama/Llama-3.2-1B",
    sign_passport=True,
    signing_key=b"demo-signing-key",
    output_dir="artifacts/trust_demo",
) as run:
    for step in range(3):
        run.log_step({"step": step, "reward_mean": 0.3 + 0.1 * step, "kl": 0.02})
    run.record_eval("math", score=0.82)

print(f"\nManifest:      {run.manifest.run_id}")
print(f"Anomalies:     {len(run.anomalies)}")
print(f"Passport risk: {run.bom.risk_flags()}")
if run.passport is not None:
    print(f"Passport valid: {run.passport.verify(b'demo-signing-key')}")
print("\nAll artifacts written to artifacts/trust_demo/")
