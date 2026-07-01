"""
Two v0.4 capabilities, both pure-Python (no GPU, no optional deps):

1. The Loop Doctor — differential diagnosis of a stalled training loop, with a
   concrete data request when the cause is a data problem.
2. Agentic environments — stateful, multi-turn, tool-use tasks with verifiable
   rewards and multi-turn credit assignment.

Usage:
    python examples/loop_doctor_and_agents.py
"""

from __future__ import annotations

from provenir.environments import (
    CreditConfig,
    EpisodeRunner,
    StubAgentPolicy,
    assign_credit,
    make_lookup_environment,
)
from provenir.loop import LoopController, LoopDoctor, LoopSignals

# ===========================================================================
# 1. Loop Doctor — why did my loop stall?
# ===========================================================================

doctor = LoopDoctor()
controller = LoopController()

print("=== Loop Doctor: differential diagnosis ===\n")

# Case A: reward plateaued, one slice is failing, data is stale -> DATA problem.
data_case = LoopSignals(
    reward_history=[0.40, 0.41, 0.40, 0.41, 0.40],
    slice_failures={"tool_use": 0.82, "math": 0.10},
    data_age_days=400.0,
)
diagnosis = doctor.diagnose(data_case)
action = controller.decide(diagnosis)
print(diagnosis.to_markdown())
print(f"-> action: {action.action}")
if action.data_request is not None:
    req = action.data_request
    print(f"   REQUEST: ~{req.num_examples} examples of {req.slices}, "
          f"from the last {req.freshness_days} days\n")

# Case B: reward rose but so did the reward-hacking rate -> REWARD problem
# (not a data problem — collecting more data would be the wrong fix).
hacking_case = LoopSignals(
    reward_history=[0.40, 0.55, 0.70, 0.82, 0.90],
    hacking_rate=0.35,
    hacking_kinds=["test_tampering", "proxy_divergence"],
)
diag_b = doctor.diagnose(hacking_case)
print(f"Case B primary cause: {diag_b.primary_category}  "
      f"-> action: {controller.decide(diag_b).action}")

# Case C: advantage collapse in the flight recorder -> ALGORITHM problem.
algo_case = LoopSignals(
    reward_history=[0.40, 0.41, 0.40, 0.41, 0.40],
    anomaly_kinds=["advantage_collapse"],
)
diag_c = doctor.diagnose(algo_case)
print(f"Case C primary cause: {diag_c.primary_category}  "
      f"-> action: {controller.decide(diag_c).action}")
print(f"   fix: {diag_c.findings[0].recommended_action}\n")

# ===========================================================================
# 2. Agentic environment — multi-turn tool use with a verifiable reward
# ===========================================================================

print("=== Agentic environment: stateful tool-use task ===\n")

env = make_lookup_environment()

# A policy that uses the lookup tool, then submits the answer.
policy = StubAgentPolicy([
    '{"tool": "lookup", "args": {"key": "capital_of_france"}}',
    "Paris",
])

episode = EpisodeRunner().run(env, policy)
print(f"Task: {env.name}")
print(f"  turns:      {episode.num_turns}")
print(f"  tool calls: {episode.num_tool_calls}")
print(f"  success:    {episode.success}")
print(f"  reward:     {episode.total_reward}")

# Multi-turn credit assignment: spread the terminal reward across turns so the
# tool-use turn gets credit too (addresses the sparse "reward only on the last
# token" problem in agentic RL).
for strategy in ("last_turn", "uniform", "discounted"):
    credit = assign_credit(episode, episode.total_reward, CreditConfig(strategy=strategy))
    print(f"  credit ({strategy:9s}): {[round(c, 3) for c in credit]}")
