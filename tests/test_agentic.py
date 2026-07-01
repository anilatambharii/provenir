from __future__ import annotations

import json
from typing import Any

import pytest

from provenir.environments.agentic import (
    AgentPolicy,
    CreditConfig,
    EpisodeResult,
    EpisodeRunner,
    StubAgentPolicy,
    Tool,
    ToolEnvironment,
    ToolRegistry,
    Turn,
    assign_credit,
)
from provenir.environments.base import Environment, Observation
from provenir.environments.tasks import (
    AGENTIC_TASK_REGISTRY,
    make_calculator_environment,
    make_lookup_environment,
    safe_eval,
)
from provenir.environments.verifiers import ExactAnswerVerifier

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _noop_tool(name: str = "noop") -> Tool:
    return Tool(name, "does nothing", lambda args: "ok")


def _echo_env(**kwargs: Any) -> ToolEnvironment:
    def echo(args: dict[str, Any]) -> str:
        return str(args.get("text", ""))

    return ToolEnvironment(
        name="echo",
        tools=[Tool("echo", "echo text", echo)],
        goal_verifier=ExactAnswerVerifier(),
        initial_prompt="echo then answer",
        goal_reference="hi",
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------


def test_tool_fields() -> None:
    t = _noop_tool()
    assert t.name == "noop"
    assert t.func({}) == "ok"


def test_tool_empty_name_rejected() -> None:
    with pytest.raises(ValueError):
        Tool("", "d", lambda a: "")


def test_tool_is_frozen() -> None:
    t = _noop_tool()
    with pytest.raises(Exception):
        t.name = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# ToolRegistry
# ---------------------------------------------------------------------------


def test_registry_register_and_get() -> None:
    reg = ToolRegistry()
    tool = _noop_tool()
    reg.register(tool)
    assert reg.get("noop") is tool


def test_registry_contains() -> None:
    reg = ToolRegistry()
    reg.register(_noop_tool())
    assert "noop" in reg
    assert "missing" not in reg


def test_registry_names_sorted() -> None:
    reg = ToolRegistry()
    reg.register(_noop_tool("b"))
    reg.register(_noop_tool("a"))
    assert reg.names() == ["a", "b"]


def test_registry_get_missing_raises_keyerror() -> None:
    reg = ToolRegistry()
    with pytest.raises(KeyError):
        reg.get("nope")


def test_registry_duplicate_raises() -> None:
    reg = ToolRegistry()
    reg.register(_noop_tool())
    with pytest.raises(ValueError):
        reg.register(_noop_tool())


# ---------------------------------------------------------------------------
# Turn / EpisodeResult
# ---------------------------------------------------------------------------


def test_turn_to_dict() -> None:
    t = Turn(index=0, action="a", observation="o", reward=0.5, tool_called="calc")
    d = t.to_dict()
    assert d == {
        "index": 0,
        "action": "a",
        "observation": "o",
        "reward": 0.5,
        "tool_called": "calc",
    }


def test_episode_result_properties() -> None:
    turns = [
        Turn(0, "a", "o", 0.0, "calc"),
        Turn(1, "b", "o2", 1.0, None),
    ]
    result = EpisodeResult(turns=turns, total_reward=1.0, success=True, num_tool_calls=1)
    assert result.num_turns == 2
    assert result.credited_rewards == [0.0, 1.0]


def test_episode_result_to_dict() -> None:
    turns = [Turn(0, "a", "o", 1.0, None)]
    result = EpisodeResult(turns=turns, total_reward=1.0, success=True, num_tool_calls=0)
    d = result.to_dict()
    assert d["num_turns"] == 1
    assert d["success"] is True
    assert d["turns"][0]["action"] == "a"


# ---------------------------------------------------------------------------
# StubAgentPolicy
# ---------------------------------------------------------------------------


def test_stub_policy_replays_in_order() -> None:
    p = StubAgentPolicy(["a", "b"])
    obs = Observation("x")
    assert p.act(obs, []) == "a"
    assert p.act(obs, []) == "b"


def test_stub_policy_repeats_last_when_exhausted() -> None:
    p = StubAgentPolicy(["a"])
    obs = Observation("x")
    assert p.act(obs, []) == "a"
    assert p.act(obs, []) == "a"


def test_stub_policy_empty_returns_empty() -> None:
    p = StubAgentPolicy([])
    assert p.act(Observation("x"), []) == ""


def test_stub_policy_is_agent_policy() -> None:
    assert isinstance(StubAgentPolicy(["a"]), AgentPolicy)


# ---------------------------------------------------------------------------
# ToolEnvironment: construction + reset
# ---------------------------------------------------------------------------


def test_env_is_environment_protocol() -> None:
    assert isinstance(_echo_env(), Environment)


def test_env_max_turns_validation() -> None:
    with pytest.raises(ValueError):
        _echo_env(max_turns=0)


def test_env_reset_returns_initial_prompt() -> None:
    env = _echo_env()
    obs = env.reset()
    assert obs.text == "echo then answer"
    assert obs.metadata["tools"] == ["echo"]
    assert obs.metadata["max_turns"] == 10


def test_env_reset_clears_state() -> None:
    env = _echo_env()
    env.reset()
    env.state["x"] = 1
    env.reset()
    assert env.state == {}


def test_env_tool_names() -> None:
    assert _echo_env().tool_names == ["echo"]


# ---------------------------------------------------------------------------
# ToolEnvironment: tool-call steps
# ---------------------------------------------------------------------------


def test_env_tool_call_executes_without_ending() -> None:
    env = _echo_env()
    env.reset()
    step = env.step('{"tool": "echo", "args": {"text": "hello"}}')
    assert step.observation.text == "hello"
    assert step.done is False
    assert step.info["tool_called"] == "echo"


def test_env_tool_call_metadata() -> None:
    env = _echo_env()
    env.reset()
    step = env.step('{"tool": "echo", "args": {"text": "hi"}}')
    assert step.observation.metadata["tool"] == "echo"
    assert step.observation.metadata["error"] is False


def test_env_tool_reads_and_writes_state() -> None:
    def remember(args: dict[str, Any]) -> str:
        state = args["__state__"]
        state["memory"] = args["value"]
        return f"remembered {state['memory']}"

    env = ToolEnvironment(
        name="mem",
        tools=[Tool("remember", "store a value", remember)],
        goal_verifier=ExactAnswerVerifier(),
        initial_prompt="p",
        goal_reference="x",
    )
    env.reset()
    step = env.step('{"tool": "remember", "args": {"value": "42"}}')
    assert step.observation.text == "remembered 42"
    assert env.state["memory"] == "42"


def test_env_tool_exception_handled_gracefully() -> None:
    def boom(args: dict[str, Any]) -> str:
        raise RuntimeError("kaboom")

    env = ToolEnvironment(
        name="boom",
        tools=[Tool("boom", "raises", boom)],
        goal_verifier=ExactAnswerVerifier(),
        initial_prompt="p",
        goal_reference="x",
    )
    env.reset()
    step = env.step('{"tool": "boom", "args": {}}')
    assert step.info["error"] is True
    assert step.observation.metadata["error"] is True
    assert "kaboom" in step.observation.text
    assert step.done is False
    assert step.reward == 0.0


def test_env_step_penalty_applied() -> None:
    env = _echo_env(step_penalty=-0.1)
    env.reset()
    step = env.step('{"tool": "echo", "args": {"text": "hi"}}')
    assert step.reward == pytest.approx(-0.1)


# ---------------------------------------------------------------------------
# ToolEnvironment: final-answer steps
# ---------------------------------------------------------------------------


def test_env_final_answer_correct() -> None:
    env = _echo_env()
    env.reset()
    step = env.step("hi")
    assert step.done is True
    assert step.reward == 1.0
    assert step.info["final"] is True
    assert step.info["passed"] is True


def test_env_final_answer_incorrect() -> None:
    env = _echo_env()
    env.reset()
    step = env.step("wrong")
    assert step.done is True
    assert step.reward == 0.0
    assert step.info["passed"] is False


def test_env_answer_tool_uses_answer_arg() -> None:
    env = _echo_env()
    env.reset()
    step = env.step('{"tool": "answer", "args": {"answer": "hi"}}')
    assert step.done is True
    assert step.reward == 1.0


def test_env_submit_tool_finalizes() -> None:
    env = _echo_env()
    env.reset()
    step = env.step('{"tool": "submit", "args": {"answer": "hi"}}')
    assert step.done is True
    assert step.reward == 1.0


def test_env_unknown_tool_treated_as_final_answer() -> None:
    env = _echo_env()
    env.reset()
    # Unknown tool name -> whole action string is the answer (not "hi"), so fails.
    step = env.step('{"tool": "nonexistent", "args": {}}')
    assert step.done is True
    assert step.reward == 0.0


def test_env_invalid_json_treated_as_final_answer() -> None:
    env = _echo_env()
    env.reset()
    step = env.step("hi")
    assert step.done is True
    assert step.reward == 1.0


# ---------------------------------------------------------------------------
# ToolEnvironment: max_turns timeout
# ---------------------------------------------------------------------------


def test_env_max_turns_timeout_ends_episode() -> None:
    env = _echo_env(max_turns=2)
    env.reset()
    s1 = env.step('{"tool": "echo", "args": {"text": "a"}}')
    assert s1.done is False
    s2 = env.step('{"tool": "echo", "args": {"text": "b"}}')
    assert s2.done is True
    assert s2.info["timeout"] is True


def test_env_max_turns_one_tool_call_times_out() -> None:
    env = _echo_env(max_turns=1)
    env.reset()
    step = env.step('{"tool": "echo", "args": {"text": "a"}}')
    assert step.done is True
    assert step.info["timeout"] is True


# ---------------------------------------------------------------------------
# safe_eval
# ---------------------------------------------------------------------------


def test_safe_eval_operations() -> None:
    assert safe_eval("6 * 7") == 42.0
    assert safe_eval("10 + 5") == 15.0
    assert safe_eval("10 - 3") == 7.0
    assert safe_eval("9 / 3") == 3.0


def test_safe_eval_rejects_bad_shape() -> None:
    with pytest.raises(ValueError):
        safe_eval("1 + 2 + 3")


def test_safe_eval_rejects_bad_operator() -> None:
    with pytest.raises(ValueError):
        safe_eval("2 ** 3")


def test_safe_eval_rejects_non_numeric() -> None:
    with pytest.raises(ValueError):
        safe_eval("a + b")


def test_safe_eval_division_by_zero() -> None:
    with pytest.raises(ValueError):
        safe_eval("1 / 0")


def test_safe_eval_does_not_eval_code() -> None:
    with pytest.raises(ValueError):
        safe_eval("__import__('os')")


# ---------------------------------------------------------------------------
# EpisodeRunner on concrete tasks
# ---------------------------------------------------------------------------


def test_runner_lookup_success() -> None:
    env = make_lookup_environment()
    policy = StubAgentPolicy(
        [
            '{"tool": "lookup", "args": {"key": "capital_of_france"}}',
            "Paris",
        ]
    )
    result = EpisodeRunner().run(env, policy)
    assert result.success is True
    assert result.total_reward == 1.0
    assert result.num_turns == 2
    assert result.num_tool_calls == 1


def test_runner_lookup_reads_state() -> None:
    env = make_lookup_environment()
    env.reset()
    env.step('{"tool": "lookup", "args": {"key": "capital_of_france"}}')
    assert env.state["last_lookup"] == "Paris"


def test_runner_calculator_success() -> None:
    env = make_calculator_environment()
    policy = StubAgentPolicy(
        [
            '{"tool": "calc", "args": {"expr": "6 * 7"}}',
            "42",
        ]
    )
    result = EpisodeRunner().run(env, policy)
    assert result.success is True
    assert result.num_tool_calls == 1
    assert result.num_turns == 2


def test_runner_calculator_wrong_answer_fails() -> None:
    env = make_calculator_environment()
    policy = StubAgentPolicy(
        [
            '{"tool": "calc", "args": {"expr": "6 * 7"}}',
            "41",
        ]
    )
    result = EpisodeRunner().run(env, policy)
    assert result.success is False
    assert result.total_reward == 0.0


def test_runner_records_tool_and_answer_turns() -> None:
    env = make_calculator_environment()
    policy = StubAgentPolicy(
        ['{"tool": "calc", "args": {"expr": "6 * 7"}}', "42"]
    )
    result = EpisodeRunner().run(env, policy)
    assert result.turns[0].tool_called == "calc"
    assert result.turns[1].tool_called is None


def test_runner_immediate_answer() -> None:
    env = _echo_env()
    result = EpisodeRunner().run(env, StubAgentPolicy(["hi"]))
    assert result.num_turns == 1
    assert result.num_tool_calls == 0
    assert result.success is True


def test_runner_timeout_no_success() -> None:
    env = _echo_env(max_turns=2)
    policy = StubAgentPolicy(['{"tool": "echo", "args": {"text": "x"}}'])
    result = EpisodeRunner().run(env, policy)
    assert result.success is False
    assert result.num_turns == 2


# ---------------------------------------------------------------------------
# CreditConfig validation
# ---------------------------------------------------------------------------


def test_credit_config_defaults() -> None:
    c = CreditConfig()
    assert c.strategy == "last_turn"
    assert c.gamma == 0.95


def test_credit_config_rejects_bad_strategy() -> None:
    with pytest.raises(ValueError):
        CreditConfig(strategy="bogus")


def test_credit_config_rejects_gamma_zero() -> None:
    with pytest.raises(ValueError):
        CreditConfig(gamma=0.0)


def test_credit_config_rejects_gamma_above_one() -> None:
    with pytest.raises(ValueError):
        CreditConfig(gamma=1.5)


def test_credit_config_gamma_one_ok() -> None:
    assert CreditConfig(gamma=1.0).gamma == 1.0


# ---------------------------------------------------------------------------
# assign_credit
# ---------------------------------------------------------------------------


def _episode(n: int) -> EpisodeResult:
    turns = [Turn(i, "a", "o", 0.0, None) for i in range(n)]
    return EpisodeResult(turns=turns, total_reward=0.0, success=False, num_tool_calls=0)


def test_assign_credit_last_turn() -> None:
    credits = assign_credit(_episode(3), 1.0, CreditConfig("last_turn"))
    assert credits == [0.0, 0.0, 1.0]


def test_assign_credit_uniform() -> None:
    credits = assign_credit(_episode(4), 1.0, CreditConfig("uniform"))
    assert credits == [0.25, 0.25, 0.25, 0.25]
    assert sum(credits) == pytest.approx(1.0)


def test_assign_credit_discounted_sums_to_terminal() -> None:
    credits = assign_credit(_episode(4), 1.0, CreditConfig("discounted", gamma=0.9))
    assert sum(credits) == pytest.approx(1.0)


def test_assign_credit_discounted_monotonic() -> None:
    credits = assign_credit(_episode(5), 1.0, CreditConfig("discounted", gamma=0.8))
    # Later turns receive more credit than earlier ones.
    assert all(credits[i] < credits[i + 1] for i in range(len(credits) - 1))


def test_assign_credit_empty_episode() -> None:
    assert assign_credit(_episode(0), 1.0, CreditConfig("uniform")) == []


def test_assign_credit_default_config_is_last_turn() -> None:
    credits = assign_credit(_episode(2), 1.0)
    assert credits == [0.0, 1.0]


def test_assign_credit_single_turn_all_strategies() -> None:
    for strat in ("last_turn", "uniform", "discounted"):
        credits = assign_credit(_episode(1), 1.0, CreditConfig(strat))
        assert credits == [1.0]


# ---------------------------------------------------------------------------
# AGENTIC_TASK_REGISTRY
# ---------------------------------------------------------------------------


def test_registry_lists_both_tasks() -> None:
    assert AGENTIC_TASK_REGISTRY.list_names() == ["calculator", "lookup"]


def test_registry_creates_lookup() -> None:
    env = AGENTIC_TASK_REGISTRY.create("lookup")
    assert env.name == "lookup"
    assert "lookup" in env.tool_names


def test_registry_creates_calculator() -> None:
    env = AGENTIC_TASK_REGISTRY.create("calculator")
    assert env.name == "calculator"
    assert "calc" in env.tool_names


def test_registry_unknown_task_raises() -> None:
    with pytest.raises(KeyError):
        AGENTIC_TASK_REGISTRY.create("nonexistent")


def test_registry_created_envs_are_independent() -> None:
    e1 = AGENTIC_TASK_REGISTRY.create("lookup")
    e2 = AGENTIC_TASK_REGISTRY.create("lookup")
    e1.reset()
    e1.state["x"] = 1
    e2.reset()
    assert e2.state == {}


# ---------------------------------------------------------------------------
# Round-trip sanity: parse tool-call JSON the way the env expects
# ---------------------------------------------------------------------------


def test_tool_call_json_shape_is_parseable() -> None:
    action = json.dumps({"tool": "echo", "args": {"text": "hi"}})
    env = _echo_env()
    env.reset()
    step = env.step(action)
    assert step.observation.text == "hi"
