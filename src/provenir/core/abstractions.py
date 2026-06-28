from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, runtime_checkable
from uuid import uuid4


@dataclass(frozen=True)
class RunManifest:
    """Content-addressed, serializable record for a reproducible run."""

    run_id: str = field(default_factory=lambda: str(uuid4()))
    config_hash: str = ""
    dataset_hash: str = ""
    seed: int = 0
    git_sha: str = ""
    dependencies_lockfile: str = ""
    hardware_fingerprint: str = ""
    metrics_history: list[dict[str, Any]] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Backend(Protocol):
    """Backend adapter contract for training engines."""

    name: str

    def prepare(self, config: Mapping[str, Any]) -> None: ...

    def fit(self, config: Mapping[str, Any], manifest: RunManifest) -> RunManifest: ...

    def save_adapter(self, output_dir: Path, config: Mapping[str, Any]) -> None: ...

    def capabilities(self) -> Mapping[str, Any]: ...


@runtime_checkable
class Dataset(Protocol):
    """Typed dataset contract with validation and hashing."""

    def validate(self) -> None: ...

    def hash(self) -> str: ...

    def iter_records(self) -> Any: ...


class RewardFn(ABC):
    """Pure, composable reward function with an audit hook."""

    kind: str = "rule"

    @abstractmethod
    def score(self, trajectory: Mapping[str, Any]) -> float: ...

    def gameability_check(self, trajectory: Mapping[str, Any]) -> list[str]:
        return []


class Evaluator(ABC):
    """Evaluation harness that returns variance-aware metrics."""

    @abstractmethod
    def evaluate(self, model: Any, dataset: Dataset) -> Mapping[str, Any]: ...


class AdapterRecord(ABC):
    """Versioned adapter record with lineage information."""

    @abstractmethod
    def lineage(self) -> Mapping[str, Any]: ...


class ComputeProvider(ABC):
    """Abstraction for where training, rollout, and eval workers run."""

    @abstractmethod
    def cost_per_run(self, config: Mapping[str, Any]) -> float: ...

    @abstractmethod
    def cost_per_step(self, config: Mapping[str, Any]) -> float: ...
