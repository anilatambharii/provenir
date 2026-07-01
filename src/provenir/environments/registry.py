from __future__ import annotations

from typing import Any, Callable

from provenir.environments.base import Environment, Verifier
from provenir.environments.verifiers import ExactAnswerVerifier, MathVerifier

# ---------------------------------------------------------------------------
# Environment registry
# ---------------------------------------------------------------------------


class EnvironmentRegistry:
    """Registry mapping names to :class:`Environment` factories.

    Example::

        reg = EnvironmentRegistry()
        reg.register("echo", lambda: MyEnv())
        env = reg.create("echo")
    """

    def __init__(self) -> None:
        self._factories: dict[str, Callable[..., Environment]] = {}

    def register(self, name: str, factory: Callable[..., Environment]) -> None:
        if not name:
            raise ValueError("name must be a non-empty string")
        if name in self._factories:
            raise ValueError(f"environment {name!r} already registered")
        self._factories[name] = factory

    def create(self, name: str, **kwargs: Any) -> Environment:
        if name not in self._factories:
            raise KeyError(f"unknown environment {name!r}")
        return self._factories[name](**kwargs)

    def list_names(self) -> list[str]:
        return sorted(self._factories)


# ---------------------------------------------------------------------------
# Verifier registry
# ---------------------------------------------------------------------------


class VerifierRegistry:
    """Registry of pre-built (typically stateless) :class:`Verifier` instances.

    Example::

        VERIFIER_REGISTRY.get("math").verify("0.5", "1/2").passed  # True
    """

    def __init__(self) -> None:
        self._verifiers: dict[str, Verifier] = {}

    def register(self, name: str, verifier: Verifier) -> None:
        if not name:
            raise ValueError("name must be a non-empty string")
        if name in self._verifiers:
            raise ValueError(f"verifier {name!r} already registered")
        self._verifiers[name] = verifier

    def get(self, name: str) -> Verifier:
        if name not in self._verifiers:
            raise KeyError(f"unknown verifier {name!r}")
        return self._verifiers[name]

    def list_names(self) -> list[str]:
        return sorted(self._verifiers)


VERIFIER_REGISTRY = VerifierRegistry()
VERIFIER_REGISTRY.register("exact_answer", ExactAnswerVerifier())
VERIFIER_REGISTRY.register("math", MathVerifier())
