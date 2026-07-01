"""Deterministic tests for the RL orchestration layer and backend adapters."""

from __future__ import annotations

import pytest

from provenir.data.dataset import JsonlDataset
from provenir.environments import ExactAnswerVerifier, MathVerifier
from provenir.observability import FlightRecorder, RewardHackingDetector
from provenir.train.algorithms import GRPOConfig
from provenir.train.backends.adapters import (
    BackendCapabilities,
    BackendSelection,
    BackendSelector,
    TRLAdapter,
    UnslothBackend,
    VerlBackend,
)
from provenir.train.rl import (
    DAPOConfig,
    GSPOConfig,
    RLOrchestrator,
    RLResult,
    RolloutGenerator,
    StubRolloutGenerator,
)
from provenir.train.rl_eval_gate import RLEvalGate, RLGateConfig

# ---------------------------------------------------------------------------
# DAPOConfig validation
# ---------------------------------------------------------------------------


def test_dapo_defaults() -> None:
    cfg = DAPOConfig()
    assert cfg.clip_higher == 0.28
    assert cfg.clip_lower == 0.2
    assert cfg.dynamic_sampling is True
    assert cfg.use_kl is False
    assert cfg.group_size == 16


def test_dapo_clip_lower_must_be_positive() -> None:
    with pytest.raises(ValueError, match="clip_lower"):
        DAPOConfig(clip_lower=0.0, clip_higher=0.28)


def test_dapo_clip_higher_must_exceed_lower() -> None:
    with pytest.raises(ValueError, match="clip_higher"):
        DAPOConfig(clip_higher=0.2, clip_lower=0.2)


def test_dapo_negative_overlong_penalty_rejected() -> None:
    with pytest.raises(ValueError, match="overlong_penalty"):
        DAPOConfig(overlong_penalty=-0.1)


def test_dapo_group_size_min() -> None:
    with pytest.raises(ValueError, match="group_size"):
        DAPOConfig(group_size=1)


def test_dapo_max_steps_min() -> None:
    with pytest.raises(ValueError, match="max_steps"):
        DAPOConfig(max_steps=0)


def test_dapo_is_frozen() -> None:
    cfg = DAPOConfig()
    with pytest.raises(Exception):
        cfg.group_size = 8  # type: ignore[misc]


# ---------------------------------------------------------------------------
# GSPOConfig validation
# ---------------------------------------------------------------------------


def test_gspo_defaults() -> None:
    cfg = GSPOConfig()
    assert cfg.clip_ratio == 0.2
    assert cfg.sequence_level is True
    assert cfg.use_kl is False
    assert cfg.group_size == 16


def test_gspo_clip_ratio_positive() -> None:
    with pytest.raises(ValueError, match="clip_ratio"):
        GSPOConfig(clip_ratio=0.0)


def test_gspo_group_size_min() -> None:
    with pytest.raises(ValueError, match="group_size"):
        GSPOConfig(group_size=1)


def test_gspo_max_steps_min() -> None:
    with pytest.raises(ValueError, match="max_steps"):
        GSPOConfig(max_steps=0)


# ---------------------------------------------------------------------------
# StubRolloutGenerator
# ---------------------------------------------------------------------------


def test_stub_rollout_is_protocol() -> None:
    assert isinstance(StubRolloutGenerator(), RolloutGenerator)


def test_stub_rollout_echoes_prompt() -> None:
    gen = StubRolloutGenerator()
    out = gen.generate("hello", 3)
    assert out == ["hello", "hello (variant 1)", "hello (variant 2)"]


def test_stub_rollout_uses_mapping() -> None:
    gen = StubRolloutGenerator({"2+2": "4"})
    out = gen.generate("2+2", 2)
    assert out == ["4", "4 (variant 1)"]


def test_stub_rollout_deterministic() -> None:
    gen = StubRolloutGenerator({"p": "base"})
    assert gen.generate("p", 5) == gen.generate("p", 5)


def test_stub_rollout_count() -> None:
    gen = StubRolloutGenerator()
    assert len(gen.generate("x", 8)) == 8


def test_stub_rollout_bad_n() -> None:
    with pytest.raises(ValueError, match="n must be"):
        StubRolloutGenerator().generate("x", 0)


# ---------------------------------------------------------------------------
# Test fixtures / helpers
# ---------------------------------------------------------------------------


def _correct_rollout() -> StubRolloutGenerator:
    return StubRolloutGenerator({"2+2": "4", "3+3": "6"})


def _train_ds() -> JsonlDataset:
    return JsonlDataset.from_records(
        [
            {"prompt": "2+2", "reference": "4"},
            {"prompt": "3+3", "reference": "6"},
        ]
    )


# ---------------------------------------------------------------------------
# RLOrchestrator.run
# ---------------------------------------------------------------------------


def test_run_returns_rlresult() -> None:
    orch = RLOrchestrator(
        GRPOConfig(group_size=4, max_steps=1),
        MathVerifier(),
        rollout=_correct_rollout(),
    )
    result = orch.run(_train_ds())
    assert isinstance(result, RLResult)
    assert result.algorithm == "grpo"


def test_run_steps_completed() -> None:
    orch = RLOrchestrator(
        GRPOConfig(group_size=4, max_steps=2),
        MathVerifier(),
        rollout=_correct_rollout(),
    )
    result = orch.run(_train_ds())
    assert result.steps_completed == 2


def test_run_groups_processed_count() -> None:
    # 2 prompts x 2 steps = 4 groups, all with reward variance (some variants wrong).
    orch = RLOrchestrator(
        GRPOConfig(group_size=4, max_steps=2),
        MathVerifier(),
        rollout=_correct_rollout(),
    )
    result = orch.run(_train_ds())
    assert result.groups_processed == 4
    assert result.groups_skipped == 0


def test_run_mean_reward_positive() -> None:
    orch = RLOrchestrator(
        GRPOConfig(group_size=4, max_steps=1),
        MathVerifier(),
        rollout=_correct_rollout(),
    )
    result = orch.run(_train_ds())
    # The base response is correct, variants are not -> mean in (0, 1).
    assert 0.0 < result.mean_reward < 1.0


def test_run_dapo_algorithm_name() -> None:
    orch = RLOrchestrator(
        DAPOConfig(group_size=4, max_steps=1),
        MathVerifier(),
        rollout=_correct_rollout(),
    )
    result = orch.run(_train_ds())
    assert result.algorithm == "dapo"


def test_run_gspo_algorithm_name() -> None:
    orch = RLOrchestrator(
        GSPOConfig(group_size=4, max_steps=1),
        MathVerifier(),
        rollout=_correct_rollout(),
    )
    result = orch.run(_train_ds())
    assert result.algorithm == "gspo"


def test_run_manifest_provenance() -> None:
    orch = RLOrchestrator(
        GSPOConfig(group_size=4, max_steps=1),
        MathVerifier(),
        rollout=_correct_rollout(),
    )
    result = orch.run(_train_ds())
    assert result.manifest.provenance["pipeline"] == "rl"
    assert result.manifest.provenance["algorithm"] == "gspo"
    assert result.manifest.provenance["steps"] == 1


def test_run_uses_response_key_fallback() -> None:
    # No "reference" key -> falls back to "response".
    ds = JsonlDataset.from_records([{"prompt": "2+2", "response": "4"}])
    orch = RLOrchestrator(
        GRPOConfig(group_size=4, max_steps=1),
        MathVerifier(),
        rollout=StubRolloutGenerator({"2+2": "4"}),
    )
    result = orch.run(ds)
    assert result.groups_processed == 1
    assert result.mean_reward > 0.0


def test_run_seed_in_manifest() -> None:
    orch = RLOrchestrator(
        GRPOConfig(group_size=4, max_steps=1),
        MathVerifier(),
        rollout=_correct_rollout(),
        seed=42,
    )
    result = orch.run(_train_ds())
    assert result.manifest.seed == 42


# ---------------------------------------------------------------------------
# Dynamic sampling (DAPO)
# ---------------------------------------------------------------------------


def test_dynamic_sampling_skips_all_equal_groups() -> None:
    # Verifier that always returns reward 0.0 -> every group is all-equal.
    class _AlwaysFail:
        name = "always_fail"

        def verify(self, response: str, reference: object) -> object:
            from provenir.environments import VerificationResult

            return VerificationResult(passed=False, reward=0.0, detail="always 0")

    orch = RLOrchestrator(
        DAPOConfig(group_size=4, max_steps=1, dynamic_sampling=True),
        _AlwaysFail(),  # type: ignore[arg-type]
        rollout=_correct_rollout(),
    )
    result = orch.run(_train_ds())
    assert result.groups_processed == 0
    assert result.groups_skipped == 2


def test_dynamic_sampling_disabled_processes_all() -> None:
    class _AlwaysFail:
        name = "always_fail"

        def verify(self, response: str, reference: object) -> object:
            from provenir.environments import VerificationResult

            return VerificationResult(passed=False, reward=0.0, detail="always 0")

    orch = RLOrchestrator(
        DAPOConfig(group_size=4, max_steps=1, dynamic_sampling=False),
        _AlwaysFail(),  # type: ignore[arg-type]
        rollout=_correct_rollout(),
    )
    result = orch.run(_train_ds())
    assert result.groups_processed == 2
    assert result.groups_skipped == 0


def test_grpo_does_not_skip_all_equal() -> None:
    # Dynamic sampling is DAPO-only; GRPO processes all-equal groups.
    class _AlwaysFail:
        name = "always_fail"

        def verify(self, response: str, reference: object) -> object:
            from provenir.environments import VerificationResult

            return VerificationResult(passed=False, reward=0.0, detail="always 0")

    orch = RLOrchestrator(
        GRPOConfig(group_size=4, max_steps=1),
        _AlwaysFail(),  # type: ignore[arg-type]
        rollout=_correct_rollout(),
    )
    result = orch.run(_train_ds())
    assert result.groups_processed == 2
    assert result.groups_skipped == 0


# ---------------------------------------------------------------------------
# Flight recorder integration
# ---------------------------------------------------------------------------


def test_flight_recorder_receives_steps() -> None:
    recorder = FlightRecorder()
    orch = RLOrchestrator(
        GRPOConfig(group_size=4, max_steps=2),
        MathVerifier(),
        rollout=_correct_rollout(),
        flight_recorder=recorder,
    )
    orch.run(_train_ds())
    # 2 prompts x 2 steps = 4 logged group steps.
    assert len(recorder.history) == 4


def test_anomaly_count_accessible() -> None:
    orch = RLOrchestrator(
        GRPOConfig(group_size=4, max_steps=1),
        MathVerifier(),
        rollout=_correct_rollout(),
    )
    result = orch.run(_train_ds())
    assert isinstance(result.anomaly_count, int)
    assert result.anomaly_count >= 0


def test_flight_summary_present() -> None:
    orch = RLOrchestrator(
        GRPOConfig(group_size=4, max_steps=1),
        MathVerifier(),
        rollout=_correct_rollout(),
    )
    result = orch.run(_train_ds())
    assert "verdict" in result.flight_summary
    assert result.flight_summary["num_steps"] == 2


def test_hacking_rate_reported() -> None:
    orch = RLOrchestrator(
        GRPOConfig(group_size=4, max_steps=1),
        MathVerifier(),
        rollout=_correct_rollout(),
        hacking_detector=RewardHackingDetector(),
    )
    result = orch.run(_train_ds())
    assert 0.0 <= result.hacking_rate <= 1.0


# ---------------------------------------------------------------------------
# Eval gate halting
# ---------------------------------------------------------------------------


def test_eval_gate_halts_on_score_floor() -> None:
    # Predictions won't match eval labels -> exact_match below floor -> halt.
    eval_ds = JsonlDataset.from_records(
        [{"prompt": "2+2", "response": "unreachable-label"}]
    )
    gate = RLEvalGate(
        RLGateConfig(eval_every=1, primary_metric="exact_match", min_primary_score=1.0),
        eval_ds,
    )
    orch = RLOrchestrator(
        GRPOConfig(group_size=4, max_steps=5),
        MathVerifier(),
        rollout=_correct_rollout(),
        eval_gate=gate,
    )
    result = orch.run(_train_ds(), eval_dataset=eval_ds)
    assert result.halted is True
    # Halts on the first step, so it does not run all 5.
    assert result.steps_completed == 1


def test_eval_gate_no_halt_when_disabled() -> None:
    # No eval dataset -> gate is never consulted, runs full course.
    eval_ds = JsonlDataset.from_records([{"prompt": "2+2", "response": "x"}])
    gate = RLEvalGate(RLGateConfig(eval_every=1, min_primary_score=1.0), eval_ds)
    orch = RLOrchestrator(
        GRPOConfig(group_size=4, max_steps=2),
        MathVerifier(),
        rollout=_correct_rollout(),
        eval_gate=gate,
    )
    result = orch.run(_train_ds())  # no eval_dataset passed
    assert result.halted is False
    assert result.steps_completed == 2


def test_eval_gate_halts_on_critical_hacking_signal() -> None:
    # An all-equal (all-fail) train group yields a critical advantage_collapse
    # signal that the gate halts on (halt_on_hacking defaults True). The floor
    # is disabled so the halt is unambiguously the hacking signal.
    eval_ds = JsonlDataset.from_records([{"prompt": "2+2", "response": "4"}])
    gate = RLEvalGate(RLGateConfig(eval_every=1), eval_ds)
    train = JsonlDataset.from_records([{"prompt": "unmapped", "reference": "impossible"}])
    orch = RLOrchestrator(
        GRPOConfig(group_size=4, max_steps=5),
        ExactAnswerVerifier(),
        rollout=StubRolloutGenerator({"2+2": "4"}),
        eval_gate=gate,
    )
    result = orch.run(train, eval_dataset=eval_ds)
    assert result.halted is True
    assert result.steps_completed == 1


def test_eval_gate_no_halt_when_score_met() -> None:
    # Best response for "2+2" is "4"; label is "4" -> exact_match perfect.
    eval_ds = JsonlDataset.from_records([{"prompt": "2+2", "response": "4"}])
    gate = RLEvalGate(
        RLGateConfig(eval_every=1, primary_metric="exact_match", min_primary_score=1.0),
        eval_ds,
    )
    orch = RLOrchestrator(
        GRPOConfig(group_size=4, max_steps=2),
        ExactAnswerVerifier(),
        rollout=StubRolloutGenerator({"2+2": "4", "3+3": "6"}),
        eval_gate=gate,
    )
    result = orch.run(_train_ds(), eval_dataset=eval_ds)
    assert result.halted is False
    assert result.steps_completed == 2


# ---------------------------------------------------------------------------
# Empty dataset + serialization
# ---------------------------------------------------------------------------


def test_empty_dataset_raises() -> None:
    orch = RLOrchestrator(
        GRPOConfig(group_size=4, max_steps=1),
        MathVerifier(),
        rollout=_correct_rollout(),
    )
    with pytest.raises(ValueError, match="empty"):
        orch.run(JsonlDataset.from_records([]))


def test_rlresult_to_dict() -> None:
    orch = RLOrchestrator(
        GRPOConfig(group_size=4, max_steps=1),
        MathVerifier(),
        rollout=_correct_rollout(),
    )
    result = orch.run(_train_ds())
    payload = result.to_dict()
    assert payload["algorithm"] == "grpo"
    assert payload["steps_completed"] == 1
    assert "flight_summary" in payload
    assert payload["manifest"]["provenance"]["pipeline"] == "rl"


def test_default_rollout_is_stub() -> None:
    orch = RLOrchestrator(GRPOConfig(group_size=2, max_steps=1), MathVerifier())
    assert isinstance(orch.rollout, StubRolloutGenerator)


# ---------------------------------------------------------------------------
# Backend adapters: capabilities
# ---------------------------------------------------------------------------


def test_backend_capabilities_validates_scale() -> None:
    with pytest.raises(ValueError, match="max_scale"):
        BackendCapabilities(
            name="x", available=True, algorithms=("grpo",),
            max_scale="galactic", supports_vllm=False,
        )


def test_verl_capabilities() -> None:
    caps = VerlBackend().capabilities()
    assert caps.name == "verl"
    assert caps.max_scale == "multi_node"
    assert caps.supports_vllm is True
    assert caps.supports("DAPO")
    assert caps.supports("rloo")


def test_trl_capabilities() -> None:
    caps = TRLAdapter().capabilities()
    assert caps.name == "trl"
    assert caps.max_scale == "multi_gpu"
    assert "orpo" in caps.algorithms


def test_unsloth_capabilities() -> None:
    caps = UnslothBackend().capabilities()
    assert caps.name == "unsloth"
    assert caps.max_scale == "single_gpu"
    assert caps.supports_vllm is False


def test_capabilities_supports_is_case_insensitive() -> None:
    caps = VerlBackend().capabilities()
    assert caps.supports("GrPo")
    assert not caps.supports("dpo")


def test_is_available_returns_bool() -> None:
    assert isinstance(VerlBackend().is_available(), bool)
    assert isinstance(TRLAdapter().is_available(), bool)
    assert isinstance(UnslothBackend().is_available(), bool)


# ---------------------------------------------------------------------------
# Backend selector
# ---------------------------------------------------------------------------


def test_selector_available_backends() -> None:
    caps = BackendSelector().available_backends()
    names = {c.name for c in caps}
    assert names == {"verl", "trl", "unsloth"}


def test_selector_falls_back_to_stub() -> None:
    # In CI none of verl/trl/unsloth are installed -> stub fallback.
    sel = BackendSelector().select(num_gpus=8, num_nodes=4, algorithm="grpo")
    assert isinstance(sel, BackendSelection)
    if not any(c.available for c in BackendSelector().available_backends()):
        assert sel.backend == "stub"
        assert sel.available is False


def test_selector_multi_node_prefers_verl_tier() -> None:
    sel = BackendSelector().select(num_gpus=8, num_nodes=4, algorithm="grpo")
    # Either verl (if installed) or stub fallback; never trl/unsloth at multi-node.
    assert sel.backend in {"verl", "stub"}


def test_selector_single_gpu_tier() -> None:
    sel = BackendSelector().select(num_gpus=1, num_nodes=1, algorithm="grpo")
    assert sel.backend in {"unsloth", "trl", "stub"}


def test_selector_multi_gpu_tier() -> None:
    sel = BackendSelector().select(num_gpus=4, num_nodes=1, algorithm="grpo")
    assert sel.backend in {"verl", "trl", "stub"}


def test_selector_prefer_honored_when_available() -> None:
    selector = BackendSelector()
    # Force a backend to appear available + supporting the algorithm.
    caps = selector.available_backends()
    trl_available = next(c for c in caps if c.name == "trl").available
    sel = selector.select(num_gpus=1, num_nodes=1, algorithm="sft", prefer="trl")
    if trl_available:
        assert sel.backend == "trl"
    else:
        # Not installed -> preference cannot be honored, falls through.
        assert sel.backend in {"unsloth", "trl", "stub"}


def test_selector_prefer_ignored_for_unsupported_algorithm() -> None:
    # verl does not support 'dpo'; preference must not be honored.
    sel = BackendSelector().select(num_gpus=1, num_nodes=1, algorithm="dpo", prefer="verl")
    assert sel.backend != "verl"


def test_selector_bad_num_gpus() -> None:
    with pytest.raises(ValueError, match="num_gpus"):
        BackendSelector().select(num_gpus=0)


def test_selector_bad_num_nodes() -> None:
    with pytest.raises(ValueError, match="num_nodes"):
        BackendSelector().select(num_nodes=0)


def test_selector_rationale_present() -> None:
    sel = BackendSelector().select(num_gpus=1, num_nodes=1, algorithm="grpo")
    assert isinstance(sel.rationale, str)
    assert sel.rationale
