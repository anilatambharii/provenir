from __future__ import annotations

import pytest

from provenir.data.dataset import JsonlDataset
from provenir.environments import ExactAnswerVerifier, MathVerifier
from provenir.observability import FlightRecorder
from provenir.train.algorithms import GRPOConfig
from provenir.train.grpo_learner import (
    GRPOUpdater,
    LearnerConfig,
    NoOpUpdater,
    TabularGRPOLearner,
    TRLGRPOAdapter,
    UpdateStats,
    group_relative_advantages,
)
from provenir.train.rl import RLOrchestrator

# --- group_relative_advantages -------------------------------------------

def test_advantages_center_and_scale() -> None:
    adv = group_relative_advantages([0.0, 1.0])
    assert adv[0] == pytest.approx(-1.0)
    assert adv[1] == pytest.approx(1.0)


def test_advantages_zero_when_no_variance() -> None:
    assert group_relative_advantages([0.5, 0.5, 0.5]) == [0.0, 0.0, 0.0]


def test_advantages_empty() -> None:
    assert group_relative_advantages([]) == []


def test_advantages_mean_is_zero() -> None:
    adv = group_relative_advantages([0.1, 0.9, 0.4, 0.6])
    assert sum(adv) == pytest.approx(0.0, abs=1e-9)


# --- LearnerConfig validation --------------------------------------------

def test_learner_config_validation() -> None:
    with pytest.raises(ValueError, match="num_actions"):
        LearnerConfig(num_actions=1)
    with pytest.raises(ValueError, match="learning_rate"):
        LearnerConfig(num_actions=3, learning_rate=0.0)
    with pytest.raises(ValueError, match="entropy_coef"):
        LearnerConfig(num_actions=3, entropy_coef=-0.1)


# --- TabularGRPOLearner: the real learning proof --------------------------

def test_learner_probabilities_start_uniform() -> None:
    learner = TabularGRPOLearner(LearnerConfig(num_actions=4, seed=0))
    probs = learner.probabilities()
    assert probs == pytest.approx([0.25, 0.25, 0.25, 0.25])


def test_learner_converges_to_rewarding_action() -> None:
    learner = TabularGRPOLearner(LearnerConfig(num_actions=4, seed=0))
    curve = learner.train(lambda a: 1.0 if a == 2 else 0.0, iterations=300, group_size=8)
    probs = learner.probabilities()
    # The policy learned to prefer action 2, and reward improved over training.
    assert probs[2] == max(probs)
    assert probs[2] > 0.5
    assert curve[-1] > curve[0]


def test_learner_is_deterministic_by_seed() -> None:
    a = TabularGRPOLearner(LearnerConfig(num_actions=3, seed=7))
    b = TabularGRPOLearner(LearnerConfig(num_actions=3, seed=7))
    ca = a.train(lambda x: float(x == 1), iterations=50, group_size=6)
    cb = b.train(lambda x: float(x == 1), iterations=50, group_size=6)
    assert ca == cb
    assert a.probabilities() == b.probabilities()


def test_learner_different_seeds_diverge() -> None:
    a = TabularGRPOLearner(LearnerConfig(num_actions=5, seed=1))
    b = TabularGRPOLearner(LearnerConfig(num_actions=5, seed=2))
    a.train(lambda x: float(x == 3), iterations=5, group_size=4)
    b.train(lambda x: float(x == 3), iterations=5, group_size=4)
    assert a.logits != b.logits


def test_learner_entropy_decreases_as_it_commits() -> None:
    learner = TabularGRPOLearner(LearnerConfig(num_actions=4, seed=0))
    start = learner.entropy()
    learner.train(lambda a: 1.0 if a == 0 else 0.0, iterations=300, group_size=8)
    assert learner.entropy() < start


def test_learner_logs_to_flight_recorder() -> None:
    learner = TabularGRPOLearner(LearnerConfig(num_actions=3, seed=0))
    recorder = FlightRecorder()
    learner.train(lambda a: float(a == 1), iterations=10, group_size=6, recorder=recorder)
    assert len(recorder.history) == 10


def test_learner_sample_group_validation() -> None:
    learner = TabularGRPOLearner(LearnerConfig(num_actions=3, seed=0))
    with pytest.raises(ValueError, match="group_size"):
        learner.sample_group(1)


def test_learner_update_length_mismatch() -> None:
    learner = TabularGRPOLearner(LearnerConfig(num_actions=3, seed=0))
    with pytest.raises(ValueError, match="equal length"):
        learner.update([0, 1], [1.0])


def test_learner_update_empty() -> None:
    learner = TabularGRPOLearner(LearnerConfig(num_actions=3, seed=0))
    with pytest.raises(ValueError, match="empty group"):
        learner.update([], [])


def test_learner_train_iterations_validation() -> None:
    learner = TabularGRPOLearner(LearnerConfig(num_actions=3, seed=0))
    with pytest.raises(ValueError, match="iterations"):
        learner.train(lambda a: 0.0, iterations=0)


def test_learner_update_returns_stats() -> None:
    learner = TabularGRPOLearner(LearnerConfig(num_actions=3, seed=0))
    stats = learner.update([0, 1, 2], [0.0, 1.0, 0.0])
    assert isinstance(stats, UpdateStats)
    assert stats.updated
    assert stats.backend == "tabular-grpo"
    assert stats.reward_mean == pytest.approx(1.0 / 3.0)


# --- Updaters -------------------------------------------------------------

def test_noop_updater() -> None:
    up = NoOpUpdater()
    assert up.is_available()
    stats = up.apply([0.0, 1.0])
    assert not stats.updated
    assert stats.backend == "noop"
    assert stats.advantage_mean == pytest.approx(0.0, abs=1e-9)


def test_grpo_updater_reports_real_advantages() -> None:
    up = GRPOUpdater(learning_rate=0.5)
    assert up.is_available()
    stats = up.apply([0.0, 1.0, 0.0, 1.0])
    assert stats.updated
    assert stats.advantage_std > 0.0
    assert stats.grad_norm > 0.0
    assert stats.backend == "grpo-reference"


def test_grpo_updater_no_signal_group() -> None:
    up = GRPOUpdater()
    stats = up.apply([0.5, 0.5, 0.5])
    assert not stats.updated  # zero-variance group carries no signal
    assert stats.advantage_std == pytest.approx(0.0)


def test_grpo_updater_validation() -> None:
    with pytest.raises(ValueError, match="learning_rate"):
        GRPOUpdater(learning_rate=0.0)


def test_update_stats_to_dict() -> None:
    stats = UpdateStats(
        loss=0.1, grad_norm=0.2, advantage_mean=0.0, advantage_std=1.0,
        reward_mean=0.5, updated=True, backend="grpo-reference",
    )
    d = stats.to_dict()
    assert d["backend"] == "grpo-reference"
    assert d["updated"] is True


# --- Orchestrator wiring --------------------------------------------------

def test_orchestrator_uses_updater_in_seam() -> None:
    ds = JsonlDataset.from_records([{"prompt": "2+2", "reference": "4"}])
    orch = RLOrchestrator(
        algorithm=GRPOConfig(group_size=4, max_steps=1),
        verifier=MathVerifier(),
        updater=GRPOUpdater(),
    )
    result = orch.run(ds)
    assert result.steps_completed == 1


def test_orchestrator_default_updater_is_none() -> None:
    orch = RLOrchestrator(
        algorithm=GRPOConfig(group_size=4, max_steps=1),
        verifier=ExactAnswerVerifier(),
    )
    assert orch.updater is None
    # Default seam still returns the inert stub shape.
    seam = orch._apply_update([0.0, 1.0])
    assert seam["updated"] is False


def test_orchestrator_seam_with_updater_returns_stats() -> None:
    orch = RLOrchestrator(
        algorithm=GRPOConfig(group_size=4, max_steps=1),
        verifier=MathVerifier(),
        updater=GRPOUpdater(),
    )
    seam = orch._apply_update([0.0, 1.0, 0.0, 1.0])
    assert seam["backend"] == "grpo-reference"
    assert seam["updated"] is True


# --- TRL production adapter (TRL not installed here) -----------------------

def test_trl_adapter_availability() -> None:
    adapter = TRLGRPOAdapter(verifier=MathVerifier())
    # In this environment TRL is not installed; the check is honest either way.
    assert isinstance(adapter.is_available(), bool)


def test_trl_adapter_reward_function_scores_via_verifier() -> None:
    adapter = TRLGRPOAdapter(verifier=MathVerifier())
    reward = adapter.reward_function()
    scores = reward(["4", "5"], reference=["4", "4"])
    assert scores[0] == pytest.approx(1.0)
    assert scores[1] == pytest.approx(0.0)


def test_trl_adapter_reward_function_broadcasts_single_reference() -> None:
    adapter = TRLGRPOAdapter(verifier=MathVerifier())
    reward = adapter.reward_function()
    scores = reward(["4", "4"], reference="4")
    assert scores == pytest.approx([1.0, 1.0])


def test_trl_adapter_build_trainer_requires_trl() -> None:
    adapter = TRLGRPOAdapter(verifier=MathVerifier())
    if not adapter.is_available():
        with pytest.raises(RuntimeError, match="provenir\\[train\\]"):
            adapter.build_trainer(model=object(), train_dataset=[])
