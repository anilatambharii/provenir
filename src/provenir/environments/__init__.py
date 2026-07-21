"""Verifiable-reward environments: sandboxed, hack-resistant verifiers for RLVR.

This package provides deterministic :class:`Verifier` implementations, an
OpenEnv-compatible multi-turn :class:`Environment` protocol, a portable code
sandbox, and registries — plus a :class:`VerifierReward` adapter that exposes any
verifier as the project's ``RewardFn``.
"""

from __future__ import annotations

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
from provenir.environments.reliability import (
    DerivedProbe,
    FailureMode,
    Probe,
    ProbeOutcome,
    PromotionBlocked,
    ReliabilityHarness,
    ReliabilityReport,
    gate_promotion,
)
from provenir.environments.reward_validity import (
    Ablation,
    AblationRun,
    RewardValidityBlocked,
    RewardValidityHarness,
    RewardValidityReport,
    TrainEval,
    gate_reward_validity,
)
from provenir.environments.sandbox import (
    CodeVerifier,
    PythonSandbox,
    SandboxConfig,
    SandboxResult,
)
from provenir.environments.tasks import (
    AGENTIC_TASK_REGISTRY,
    make_calculator_environment,
    make_lookup_environment,
)
from provenir.environments.tim import (
    TIMBlocked,
    TIMDetector,
    TIMProbeFn,
    TIMReport,
    TIMResult,
    gate_tim,
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
    "AGENTIC_TASK_REGISTRY",
    "VERIFIER_REGISTRY",
    "Ablation",
    "AblationRun",
    "AgentPolicy",
    "CodeVerifier",
    "CompositeVerifier",
    "ContainsVerifier",
    "CreditConfig",
    "DerivedProbe",
    "Environment",
    "EnvironmentRegistry",
    "EpisodeResult",
    "EpisodeRunner",
    "ExactAnswerVerifier",
    "FailureMode",
    "JSONSchemaVerifier",
    "MathVerifier",
    "Observation",
    "Probe",
    "ProbeOutcome",
    "PromotionBlocked",
    "PythonSandbox",
    "RegexFormatVerifier",
    "ReliabilityHarness",
    "ReliabilityReport",
    "RewardValidityBlocked",
    "RewardValidityHarness",
    "RewardValidityReport",
    "SandboxConfig",
    "SandboxResult",
    "StepResult",
    "StubAgentPolicy",
    "TIMBlocked",
    "TIMDetector",
    "TIMProbeFn",
    "TIMReport",
    "TIMResult",
    "Tool",
    "ToolCallVerifier",
    "ToolEnvironment",
    "ToolRegistry",
    "TrainEval",
    "Turn",
    "VerificationResult",
    "Verifier",
    "VerifierRegistry",
    "VerifierReward",
    "assign_credit",
    "gate_promotion",
    "gate_reward_validity",
    "gate_tim",
    "make_calculator_environment",
    "make_lookup_environment",
]
