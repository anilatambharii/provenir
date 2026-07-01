"""Agentic, stateful, multi-turn tool-use environments with credit assignment.

This module extends the environments package with *agentic* environments: the
agent takes multiple turns, each of which may call a tool (mutating a shared,
per-episode ``state`` dict) or submit a final answer that is scored by a
:class:`~provenir.environments.base.Verifier`. It also solves the classic
"sparse terminal reward on the last token" problem by redistributing the
terminal reward across turns via :func:`assign_credit`.

Example::

    from provenir.environments.verifiers import ExactAnswerVerifier

    def echo(args: dict[str, object]) -> str:
        return str(args.get("text", ""))

    env = ToolEnvironment(
        name="echo",
        tools=[Tool("echo", "echo text", echo)],
        goal_verifier=ExactAnswerVerifier(),
        initial_prompt="Say hi then answer 'hi'.",
        goal_reference="hi",
    )
    policy = StubAgentPolicy(['{"tool": "echo", "args": {"text": "hi"}}', "hi"])
    result = EpisodeRunner().run(env, policy)
    assert result.success
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Protocol, runtime_checkable

from provenir.environments.base import Observation, StepResult, Verifier

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Tool:
    """A named, callable tool the agent can invoke.

    ``func`` receives the (already-parsed) ``args`` dict and returns an
    observation string. To read or write per-episode state, the environment
    injects its mutable ``state`` dict under the reserved ``"__state__"`` key in
    ``args`` before calling ``func`` (see :class:`ToolEnvironment`).

    Example::

        Tool("add", "add two numbers", lambda a: str(a["x"] + a["y"]))
    """

    name: str
    description: str
    func: Callable[[dict[str, Any]], str]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name must be a non-empty string")


class ToolRegistry:
    """Registry mapping tool names to :class:`Tool` instances.

    Example::

        reg = ToolRegistry()
        reg.register(Tool("noop", "does nothing", lambda a: ""))
        "noop" in reg  # True
    """

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"tool {tool.name!r} already registered")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise KeyError(f"unknown tool {name!r}")
        return self._tools[name]

    def names(self) -> list[str]:
        return sorted(self._tools)

    def __contains__(self, name: object) -> bool:
        return name in self._tools


# ---------------------------------------------------------------------------
# Turns + episode result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Turn:
    """A single recorded (action, observation, reward) step in an episode."""

    index: int
    action: str
    observation: str
    reward: float
    tool_called: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "action": self.action,
            "observation": self.observation,
            "reward": self.reward,
            "tool_called": self.tool_called,
        }


@dataclass(frozen=True)
class EpisodeResult:
    """Aggregate outcome of running a policy in a :class:`ToolEnvironment`."""

    turns: list[Turn]
    total_reward: float
    success: bool
    num_tool_calls: int

    @property
    def num_turns(self) -> int:
        return len(self.turns)

    @property
    def credited_rewards(self) -> list[float]:
        """Per-turn rewards as recorded during the episode."""
        return [turn.reward for turn in self.turns]

    def to_dict(self) -> dict[str, Any]:
        return {
            "turns": [turn.to_dict() for turn in self.turns],
            "total_reward": self.total_reward,
            "success": self.success,
            "num_tool_calls": self.num_tool_calls,
            "num_turns": self.num_turns,
        }


# ---------------------------------------------------------------------------
# Policy protocol + stub policy
# ---------------------------------------------------------------------------


@runtime_checkable
class AgentPolicy(Protocol):
    """Structural type for a policy that maps observations to next actions.

    An action is either a JSON tool call ``{"tool": ..., "args": {...}}`` or a
    plain final-answer string.

    Example::

        class Always42:
            def act(self, observation: Observation, history: list[Turn]) -> str:
                return "42"
    """

    def act(self, observation: Observation, history: list[Turn]) -> str: ...


class StubAgentPolicy:
    """Deterministic policy that replays a fixed list of actions in order.

    Once the list is exhausted the last action is returned repeatedly (or the
    empty string if the list was empty). Intended for tests and fixtures.

    Example::

        StubAgentPolicy(["a", "b"]).act(Observation("x"), [])  # "a"
    """

    def __init__(self, actions: list[str]) -> None:
        self.actions = list(actions)
        self._cursor = 0

    def act(self, observation: Observation, history: list[Turn]) -> str:
        if not self.actions:
            return ""
        if self._cursor < len(self.actions):
            action = self.actions[self._cursor]
            self._cursor += 1
            return action
        return self.actions[-1]


# ---------------------------------------------------------------------------
# Tool environment
# ---------------------------------------------------------------------------

_STATE_KEY = "__state__"
_FINAL_TOOLS = frozenset({"answer", "submit"})


def _parse_tool_call(action: str) -> tuple[str, dict[str, Any]] | None:
    """Parse *action* as a ``{"tool","args"}`` JSON object, or return None."""
    try:
        data = json.loads(action)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(data, dict) or "tool" not in data:
        return None
    tool = data["tool"]
    if not isinstance(tool, str):
        return None
    args = data.get("args", {})
    if not isinstance(args, dict):
        return None
    return tool, args


class ToolEnvironment:
    """A stateful, multi-turn, tool-use environment with a verifiable goal.

    The agent submits one *action* per turn:

    * A JSON tool call ``{"tool": name, "args": {...}}`` for a registered,
      non-final tool executes that tool and returns its observation. The tool's
      exceptions are caught and surfaced as an error observation (reward 0.0);
      the episode continues (``done=False``) unless ``max_turns`` is reached.
    * Any other action — a plain string, an unknown/invalid tool call, or a tool
      named ``"answer"``/``"submit"`` — is treated as a *final answer*. It is
      scored against ``goal_reference`` by ``goal_verifier`` and ends the
      episode (``done=True``) with ``reward = verification.reward``. For a
      ``{"tool": "answer"/"submit", "args": {...}}`` call the answer text is
      taken from ``args["answer"]`` (falling back to the raw action).

    Tools read and write shared per-episode state through the mutable ``state``
    dict, which the environment injects into every tool call under the reserved
    ``"__state__"`` key of ``args``. ``state`` is cleared on :meth:`reset`.

    Example::

        env = ToolEnvironment(
            name="demo",
            tools=[Tool("store", "save x", lambda a: a["__state__"].setdefault(
                "x", a["v"]) and "" or "stored")],
            goal_verifier=ExactAnswerVerifier(),
            initial_prompt="store 1 then answer 1",
            goal_reference="1",
        )
        obs = env.reset()
    """

    def __init__(
        self,
        name: str,
        tools: list[Tool],
        goal_verifier: Verifier,
        initial_prompt: str,
        goal_reference: Any,
        max_turns: int = 10,
        step_penalty: float = 0.0,
    ) -> None:
        if max_turns < 1:
            raise ValueError(f"max_turns must be >= 1, got {max_turns}")
        self.name = name
        self.goal_verifier = goal_verifier
        self.initial_prompt = initial_prompt
        self.goal_reference = goal_reference
        self.max_turns = max_turns
        self.step_penalty = step_penalty
        self._registry = ToolRegistry()
        for tool in tools:
            self._registry.register(tool)
        self.state: dict[str, Any] = {}
        self._turn = 0

    @property
    def tool_names(self) -> list[str]:
        return self._registry.names()

    def reset(self) -> Observation:
        """Reset turn counter and per-episode state, returning the prompt."""
        self._turn = 0
        self.state = {}
        return Observation(
            text=self.initial_prompt,
            metadata={"tools": self._registry.names(), "max_turns": self.max_turns},
        )

    def _finalize(self, answer: str) -> StepResult:
        result = self.goal_verifier.verify(answer, self.goal_reference)
        return StepResult(
            observation=Observation(
                text=result.detail,
                metadata={"verified": result.passed, "answer": answer},
            ),
            reward=result.reward,
            done=True,
            info={
                "final": True,
                "passed": result.passed,
                "verifier": self.goal_verifier.name,
            },
        )

    def step(self, action: str) -> StepResult:
        self._turn += 1
        parsed = _parse_tool_call(action)

        # Final-answer path: not a tool call, unknown tool, or a submit/answer tool.
        if parsed is None or parsed[0] not in self._registry or parsed[0] in _FINAL_TOOLS:
            if parsed is not None and parsed[0] in _FINAL_TOOLS:
                args = parsed[1]
                answer = str(args["answer"]) if "answer" in args else action
            else:
                answer = action
            return self._finalize(answer)

        tool_name, args = parsed
        tool = self._registry.get(tool_name)
        call_args = dict(args)
        call_args[_STATE_KEY] = self.state
        try:
            observation_text = tool.func(call_args)
            error = False
        except Exception as exc:  # noqa: BLE001 - tools are untrusted
            observation_text = f"tool {tool_name!r} raised {type(exc).__name__}: {exc}"
            error = True

        timed_out = self._turn >= self.max_turns
        reward = self.step_penalty
        return StepResult(
            observation=Observation(
                text=observation_text,
                metadata={"tool": tool_name, "error": error},
            ),
            reward=reward,
            done=timed_out,
            info={
                "tool_called": tool_name,
                "error": error,
                "timeout": timed_out,
                "turn": self._turn,
            },
        )


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------


class EpisodeRunner:
    """Drive an :class:`AgentPolicy` through a :class:`ToolEnvironment`.

    Example::

        result = EpisodeRunner().run(env, StubAgentPolicy(["42"]))
        assert result.num_turns >= 1
    """

    def run(self, env: ToolEnvironment, policy: AgentPolicy) -> EpisodeResult:
        obs = env.reset()
        history: list[Turn] = []
        total_reward = 0.0
        num_tool_calls = 0
        terminal_reward = 0.0

        for index in range(env.max_turns):
            action = policy.act(obs, history)
            result = env.step(action)
            tool_called = result.info.get("tool_called")
            if tool_called is not None:
                num_tool_calls += 1
            history.append(
                Turn(
                    index=index,
                    action=action,
                    observation=result.observation.text,
                    reward=result.reward,
                    tool_called=tool_called,
                )
            )
            total_reward += result.reward
            obs = result.observation
            if result.done:
                if result.info.get("final"):
                    terminal_reward = result.reward
                break

        return EpisodeResult(
            turns=history,
            total_reward=total_reward,
            success=terminal_reward > 0.0,
            num_tool_calls=num_tool_calls,
        )


# ---------------------------------------------------------------------------
# Multi-turn credit assignment
# ---------------------------------------------------------------------------

_STRATEGIES = frozenset({"uniform", "discounted", "last_turn"})


@dataclass(frozen=True)
class CreditConfig:
    """Configuration for redistributing a terminal reward across turns.

    * ``last_turn`` — all reward on the final turn (the raw sparse signal).
    * ``uniform`` — split equally across every turn.
    * ``discounted`` — ``gamma ** (T - 1 - t)`` weighting, normalized so the
      credits sum to the terminal reward (later turns get more credit).

    Example::

        CreditConfig(strategy="discounted", gamma=0.9)
    """

    strategy: str = "last_turn"
    gamma: float = 0.95

    def __post_init__(self) -> None:
        if self.strategy not in _STRATEGIES:
            raise ValueError(
                f"strategy must be one of {sorted(_STRATEGIES)}, got {self.strategy!r}"
            )
        if not (0.0 < self.gamma <= 1.0):
            raise ValueError(f"gamma must be in (0, 1], got {self.gamma}")


def assign_credit(
    episode: EpisodeResult,
    terminal_reward: float,
    config: CreditConfig = CreditConfig(),
) -> list[float]:
    """Distribute *terminal_reward* across an episode's turns per *config*.

    Returns one credit per turn; the credits always sum to ``terminal_reward``
    (up to floating-point error). An empty episode yields an empty list.

    Example::

        credits = assign_credit(result, 1.0, CreditConfig("uniform"))
        assert abs(sum(credits) - 1.0) < 1e-9
    """
    n = episode.num_turns
    if n == 0:
        return []

    if config.strategy == "last_turn":
        credits = [0.0] * n
        credits[-1] = terminal_reward
        return credits

    if config.strategy == "uniform":
        return [terminal_reward / n] * n

    # discounted: weight w_t = gamma ** (T - 1 - t), normalized to sum to reward.
    weights = [config.gamma ** (n - 1 - t) for t in range(n)]
    total = sum(weights)
    return [terminal_reward * (w / total) for w in weights]
