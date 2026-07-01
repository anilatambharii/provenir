from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, runtime_checkable

from provenir.core.abstractions import RewardFn

# ---------------------------------------------------------------------------
# Verification result + verifier protocol
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VerificationResult:
    """Outcome of verifying a single model response against a reference.

    Example::

        result = VerificationResult(passed=True, reward=1.0, detail="exact match")
        assert result.reward == 1.0
    """

    passed: bool
    reward: float  # 0-1
    detail: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not (0.0 <= self.reward <= 1.0):
            raise ValueError(f"reward must be in [0, 1], got {self.reward}")


@runtime_checkable
class Verifier(Protocol):
    """Structural type for a deterministic, hack-resistant verifier.

    Example::

        class LenVerifier:
            name = "len"

            def verify(self, response: str, reference: Any) -> VerificationResult:
                ok = len(response) == int(reference)
                return VerificationResult(ok, 1.0 if ok else 0.0, "length check")
    """

    @property
    def name(self) -> str: ...

    def verify(self, response: str, reference: Any) -> VerificationResult: ...


# ---------------------------------------------------------------------------
# Multi-turn environment protocol (OpenEnv-compatible)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Observation:
    """A single environment observation returned to the agent."""

    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StepResult:
    """Result of taking one action in an :class:`Environment`."""

    observation: Observation
    reward: float
    done: bool
    info: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Environment(Protocol):
    """OpenEnv-compatible, multi-turn environment contract.

    Example::

        env: Environment = MyEnv()
        obs = env.reset()
        step = env.step("my action")
        assert isinstance(step.done, bool)
    """

    name: str

    def reset(self) -> Observation: ...

    def step(self, action: str) -> StepResult: ...


# ---------------------------------------------------------------------------
# Adapter: Verifier -> project RewardFn
# ---------------------------------------------------------------------------


class VerifierReward(RewardFn):
    """Adapt a :class:`Verifier` so it plugs into the project's ``RewardFn``.

    The response and reference are pulled from the trajectory mapping using the
    configured keys::

        reward = VerifierReward(ExactAnswerVerifier())
        reward.score({"prediction": "42", "reference": "42"})  # -> 1.0
    """

    kind: str = "verifier"

    def __init__(
        self,
        verifier: Verifier,
        reference_key: str = "reference",
        response_key: str = "prediction",
    ) -> None:
        self.verifier = verifier
        self.reference_key = reference_key
        self.response_key = response_key

    def _run(self, trajectory: Mapping[str, Any]) -> VerificationResult:
        response = str(trajectory.get(self.response_key, ""))
        reference = trajectory.get(self.reference_key)
        return self.verifier.verify(response, reference)

    def score(self, trajectory: Mapping[str, Any]) -> float:
        return self._run(trajectory).reward

    def gameability_check(self, trajectory: Mapping[str, Any]) -> list[str]:
        return [] if self._run(trajectory).passed else ["verifier_failed"]
