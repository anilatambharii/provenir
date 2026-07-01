"""Concrete, deterministic agentic environments + a shareable task registry.

These small environments demonstrate the "environments hub" pattern: they are
pre-registered in :data:`AGENTIC_TASK_REGISTRY` so they can be discovered and
instantiated by name. Each is fully deterministic and stdlib-only.

Example::

    env = AGENTIC_TASK_REGISTRY.create("lookup")
    obs = env.reset()
    step = env.step('{"tool": "lookup", "args": {"key": "capital_of_france"}}')
    assert "paris" in step.observation.text.lower()
"""

from __future__ import annotations

from typing import Any

from provenir.environments.agentic import Tool, ToolEnvironment
from provenir.environments.registry import EnvironmentRegistry
from provenir.environments.verifiers import ExactAnswerVerifier

# ---------------------------------------------------------------------------
# Lookup environment
# ---------------------------------------------------------------------------

_FACTS: dict[str, str] = {
    "capital_of_france": "Paris",
    "capital_of_japan": "Tokyo",
    "largest_planet": "Jupiter",
}


def _make_lookup_tool() -> Tool:
    """Build a ``lookup`` tool backed by an in-memory key-value store.

    The tool records the last fact it retrieved into the shared episode state
    under ``state["last_lookup"]`` to demonstrate stateful tools.
    """

    def lookup(args: dict[str, Any]) -> str:
        state = args["__state__"]
        key = str(args.get("key", ""))
        value = _FACTS.get(key)
        if value is None:
            return f"no fact for key {key!r}"
        state["last_lookup"] = value
        return f"{key} = {value}"

    return Tool("lookup", "look up a fact by key from the knowledge base", lookup)


def make_lookup_environment() -> ToolEnvironment:
    """A stateful lookup task: find a fact via the ``lookup`` tool, then submit.

    The agent must call ``lookup`` with ``key="capital_of_france"`` and then
    submit ``"Paris"``; the answer is verified with :class:`ExactAnswerVerifier`.

    Example::

        env = make_lookup_environment()
        env.reset()
        env.step('{"tool": "lookup", "args": {"key": "capital_of_france"}}')
        env.step("Paris").reward  # 1.0
    """
    return ToolEnvironment(
        name="lookup",
        tools=[_make_lookup_tool()],
        goal_verifier=ExactAnswerVerifier(),
        initial_prompt=(
            "Use the 'lookup' tool with key 'capital_of_france' to find the fact, "
            "then submit the value as your final answer."
        ),
        goal_reference="Paris",
        max_turns=5,
    )


# ---------------------------------------------------------------------------
# Calculator environment
# ---------------------------------------------------------------------------

_OPERATORS = {"+", "-", "*", "/"}


def safe_eval(expression: str) -> float:
    """Safely evaluate ``<number> <op> <number>`` with op in ``+ - * /``.

    This deliberately does **not** use :func:`eval`; it parses exactly two
    numeric operands and a single binary operator. Raises ``ValueError`` on any
    malformed or unsupported input (including division by zero).

    Example::

        safe_eval("6 * 7")  # 42.0
    """
    tokens = expression.split()
    if len(tokens) != 3:
        raise ValueError(f"expected '<number> <op> <number>', got {expression!r}")
    left_tok, op, right_tok = tokens
    if op not in _OPERATORS:
        raise ValueError(f"unsupported operator {op!r}")
    try:
        left = float(left_tok)
        right = float(right_tok)
    except ValueError as exc:
        raise ValueError(f"non-numeric operand in {expression!r}") from exc
    if op == "+":
        return left + right
    if op == "-":
        return left - right
    if op == "*":
        return left * right
    if right == 0.0:
        raise ValueError("division by zero")
    return left / right


def _make_calc_tool() -> Tool:
    """Build a ``calc`` tool that safely evaluates a two-operand expression."""

    def calc(args: dict[str, Any]) -> str:
        state = args["__state__"]
        expression = str(args.get("expr", ""))
        value = safe_eval(expression)
        # Store an int-looking result when exact, else the float.
        rendered = str(int(value)) if value.is_integer() else str(value)
        state["last_result"] = rendered
        return rendered

    return Tool("calc", "evaluate a safe two-operand arithmetic expression", calc)


def make_calculator_environment() -> ToolEnvironment:
    """A calculator task: compute ``6 * 7`` via ``calc``, then submit ``42``.

    The answer is verified with :class:`ExactAnswerVerifier` (exact string
    match of the extracted answer).

    Example::

        env = make_calculator_environment()
        env.reset()
        env.step('{"tool": "calc", "args": {"expr": "6 * 7"}}')
        env.step("42").reward  # 1.0
    """
    return ToolEnvironment(
        name="calculator",
        tools=[_make_calc_tool()],
        goal_verifier=ExactAnswerVerifier(),
        initial_prompt=(
            "Use the 'calc' tool to evaluate '6 * 7', then submit the numeric "
            "result as your final answer."
        ),
        goal_reference="42",
        max_turns=5,
    )


# ---------------------------------------------------------------------------
# Task registry ("environments hub" pattern)
# ---------------------------------------------------------------------------

AGENTIC_TASK_REGISTRY = EnvironmentRegistry()
AGENTIC_TASK_REGISTRY.register("lookup", make_lookup_environment)
AGENTIC_TASK_REGISTRY.register("calculator", make_calculator_environment)
