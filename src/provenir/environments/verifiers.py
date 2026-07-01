from __future__ import annotations

import json
import re
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from provenir.environments.base import VerificationResult, Verifier

# ---------------------------------------------------------------------------
# Shared extraction helpers
# ---------------------------------------------------------------------------

_BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")
_NUMBER_RE = re.compile(r"[-+]?\d+(?:/\d+)?(?:\.\d+)?(?:[eE][-+]?\d+)?")


def _extract_answer(text: str) -> str:
    """Extract a final answer: ``\\boxed{...}``, text after ``####``, or whole."""
    boxed = _BOXED_RE.findall(text)
    if boxed:
        return str(boxed[-1]).strip()
    if "####" in text:
        return str(text.rsplit("####", 1)[-1].strip())
    return text.strip()


def _parse_number(token: str) -> float | None:
    """Parse an int/float/fraction token into a float, or ``None``."""
    token = token.strip().replace(",", "")
    try:
        if "/" in token:
            return float(Fraction(token))
        return float(token)
    except (ValueError, ZeroDivisionError):
        return None


def _extract_number(text: str) -> float | None:
    """Extract the last parseable number (preferring ``\\boxed{...}``)."""
    boxed = _BOXED_RE.findall(text)
    if boxed:
        num = _parse_number(boxed[-1])
        if num is not None:
            return num
    matches = _NUMBER_RE.findall(text)
    for token in reversed(matches):
        num = _parse_number(token)
        if num is not None:
            return num
    return None


# ---------------------------------------------------------------------------
# Answer / math verifiers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExactAnswerVerifier:
    """Case-insensitive exact match of an extracted final answer.

    Example::

        ExactAnswerVerifier().verify("The answer is \\boxed{42}", "42").passed  # True
    """

    name: str = "exact_answer"

    def verify(self, response: str, reference: Any) -> VerificationResult:
        got = _extract_answer(response).casefold()
        want = _extract_answer(str(reference)).casefold()
        passed = got == want
        return VerificationResult(
            passed=passed,
            reward=1.0 if passed else 0.0,
            detail=f"expected {want!r}, got {got!r}",
            metadata={"extracted": got, "reference": want},
        )


@dataclass(frozen=True)
class MathVerifier:
    """Numeric equivalence with relative tolerance.

    Example::

        MathVerifier().verify("x = 0.5", "1/2").passed  # True
    """

    tol: float = 1e-6
    name: str = "math"

    def __post_init__(self) -> None:
        if self.tol < 0:
            raise ValueError(f"tol must be non-negative, got {self.tol}")

    def verify(self, response: str, reference: Any) -> VerificationResult:
        got = _extract_number(response)
        want = _extract_number(str(reference))
        if got is None or want is None:
            return VerificationResult(
                passed=False,
                reward=0.0,
                detail="could not parse a number",
                metadata={"response_number": got, "reference_number": want},
            )
        denom = max(abs(want), 1.0)
        passed = abs(got - want) <= self.tol * denom
        return VerificationResult(
            passed=passed,
            reward=1.0 if passed else 0.0,
            detail=f"expected {want}, got {got}",
            metadata={"response_number": got, "reference_number": want},
        )


# ---------------------------------------------------------------------------
# Format / structure verifiers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RegexFormatVerifier:
    """Verify a response against a regular expression.

    Example::

        RegexFormatVerifier(r"\\d+").verify("123", None).passed  # True
    """

    pattern: str
    full_match: bool = True
    name: str = "regex_format"

    def __post_init__(self) -> None:
        try:
            re.compile(self.pattern)
        except re.error as exc:  # pragma: no cover - defensive
            raise ValueError(f"invalid regex: {exc}") from exc

    def verify(self, response: str, reference: Any) -> VerificationResult:
        compiled = re.compile(self.pattern, re.DOTALL)
        match = compiled.fullmatch(response) if self.full_match else compiled.search(response)
        passed = match is not None
        mode = "fullmatch" if self.full_match else "search"
        return VerificationResult(
            passed=passed,
            reward=1.0 if passed else 0.0,
            detail=f"regex {mode} {'ok' if passed else 'failed'}",
            metadata={"pattern": self.pattern, "full_match": self.full_match},
        )


@dataclass(frozen=True)
class JSONSchemaVerifier:
    """Verify a response is JSON with required keys and (optional) value types.

    Reward is the fraction of required keys satisfied::

        JSONSchemaVerifier(["a", "b"]).verify('{"a": 1}', None).reward  # 0.5
    """

    required_keys: list[str]
    types: dict[str, type] | None = None
    name: str = "json_schema"

    def __post_init__(self) -> None:
        if not self.required_keys:
            raise ValueError("required_keys must not be empty")

    def verify(self, response: str, reference: Any) -> VerificationResult:
        try:
            data = json.loads(response)
        except (json.JSONDecodeError, TypeError):
            return VerificationResult(False, 0.0, "response is not valid JSON")
        if not isinstance(data, dict):
            return VerificationResult(False, 0.0, "JSON is not an object")

        satisfied = 0
        for key in self.required_keys:
            if key not in data:
                continue
            if self.types is not None and key in self.types:
                if not isinstance(data[key], self.types[key]):
                    continue
            satisfied += 1

        reward = satisfied / len(self.required_keys)
        passed = satisfied == len(self.required_keys)
        return VerificationResult(
            passed=passed,
            reward=reward,
            detail=f"{satisfied}/{len(self.required_keys)} keys satisfied",
            metadata={"satisfied": satisfied, "total": len(self.required_keys)},
        )


@dataclass(frozen=True)
class ToolCallVerifier:
    """Verify a response is a JSON tool call with an allowed tool and dict args.

    Example::

        v = ToolCallVerifier(["search"])
        v.verify('{"tool": "search", "args": {"q": "x"}}', None).passed  # True
    """

    allowed_tools: list[str]
    name: str = "tool_call"

    def __post_init__(self) -> None:
        if not self.allowed_tools:
            raise ValueError("allowed_tools must not be empty")

    def verify(self, response: str, reference: Any) -> VerificationResult:
        try:
            data = json.loads(response)
        except (json.JSONDecodeError, TypeError):
            return VerificationResult(False, 0.0, "response is not valid JSON")
        if not isinstance(data, dict) or "tool" not in data or "args" not in data:
            return VerificationResult(False, 0.0, "missing 'tool' or 'args'")
        tool = data["tool"]
        args = data["args"]
        if tool not in self.allowed_tools:
            return VerificationResult(
                False, 0.0, f"tool {tool!r} not allowed", metadata={"tool": tool}
            )
        if not isinstance(args, dict):
            return VerificationResult(False, 0.0, "'args' must be a dict")
        return VerificationResult(
            passed=True,
            reward=1.0,
            detail=f"valid call to {tool!r}",
            metadata={"tool": tool},
        )


@dataclass(frozen=True)
class ContainsVerifier:
    """Reward = fraction of required substrings present; 0 if any forbidden present.

    Example::

        ContainsVerifier(["foo", "bar"]).verify("foo baz", None).reward  # 0.5
    """

    required: list[str]
    forbidden: list[str] | None = None
    name: str = "contains"

    def __post_init__(self) -> None:
        if not self.required:
            raise ValueError("required must not be empty")

    def verify(self, response: str, reference: Any) -> VerificationResult:
        forbidden = self.forbidden or []
        present_forbidden = [f for f in forbidden if f in response]
        if present_forbidden:
            return VerificationResult(
                passed=False,
                reward=0.0,
                detail=f"forbidden present: {present_forbidden}",
                metadata={"forbidden_present": present_forbidden},
            )
        hits = sum(1 for r in self.required if r in response)
        reward = hits / len(self.required)
        passed = hits == len(self.required)
        return VerificationResult(
            passed=passed,
            reward=reward,
            detail=f"{hits}/{len(self.required)} required present",
            metadata={"hits": hits, "total": len(self.required)},
        )


# ---------------------------------------------------------------------------
# Composite
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CompositeVerifier:
    """Weighted average of member verifiers; passes if reward >= *threshold*.

    Example::

        cv = CompositeVerifier([(ExactAnswerVerifier(), 1.0), (MathVerifier(), 1.0)])
        cv.verify("42", "42").passed  # True
    """

    verifiers: list[tuple[Verifier, float]]
    threshold: float = 0.5
    name: str = "composite"

    def __post_init__(self) -> None:
        if not self.verifiers:
            raise ValueError("verifiers must not be empty")
        if sum(w for _, w in self.verifiers) <= 0:
            raise ValueError("verifier weights must sum to a positive value")
        if not (0.0 <= self.threshold <= 1.0):
            raise ValueError(f"threshold must be in [0, 1], got {self.threshold}")

    def verify(self, response: str, reference: Any) -> VerificationResult:
        total_weight = sum(w for _, w in self.verifiers)
        weighted = 0.0
        sub: dict[str, float] = {}
        for verifier, weight in self.verifiers:
            result = verifier.verify(response, reference)
            weighted += result.reward * weight
            sub[verifier.name] = result.reward
        reward = weighted / total_weight
        passed = reward >= self.threshold
        return VerificationResult(
            passed=passed,
            reward=reward,
            detail=f"weighted reward {reward:.3f} vs threshold {self.threshold}",
            metadata={"sub_rewards": sub},
        )
