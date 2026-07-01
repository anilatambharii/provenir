"""Verifiable-reward environments: sandboxed, hack-resistant verifiers for RLVR.

This package provides deterministic :class:`Verifier` implementations, an
OpenEnv-compatible multi-turn :class:`Environment` protocol, a portable code
sandbox, and registries — plus a :class:`VerifierReward` adapter that exposes any
verifier as the project's ``RewardFn``.
"""

from __future__ import annotations

from provenir.environments.base import (
    Environment,
    Observation,
    StepResult,
    VerificationResult,
    Verifier,
    VerifierReward,
)
from provenir.environments.registry import (
    VERIFIER_REGISTRY,
    EnvironmentRegistry,
    VerifierRegistry,
)
from provenir.environments.sandbox import (
    CodeVerifier,
    PythonSandbox,
    SandboxConfig,
    SandboxResult,
)
from provenir.environments.verifiers import (
    CompositeVerifier,
    ContainsVerifier,
    ExactAnswerVerifier,
    JSONSchemaVerifier,
    MathVerifier,
    RegexFormatVerifier,
    ToolCallVerifier,
)

__all__ = [
    "VERIFIER_REGISTRY",
    "CodeVerifier",
    "CompositeVerifier",
    "ContainsVerifier",
    "Environment",
    "EnvironmentRegistry",
    "ExactAnswerVerifier",
    "JSONSchemaVerifier",
    "MathVerifier",
    "Observation",
    "PythonSandbox",
    "RegexFormatVerifier",
    "SandboxConfig",
    "SandboxResult",
    "StepResult",
    "ToolCallVerifier",
    "VerificationResult",
    "Verifier",
    "VerifierRegistry",
    "VerifierReward",
]
