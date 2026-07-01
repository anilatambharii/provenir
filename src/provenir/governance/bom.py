from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class DataComponent:
    """A single dataset input that contributed to a model.

    Example:
        >>> DataComponent(name="train", content_hash="abc", num_records=100).name
        'train'
    """

    name: str
    content_hash: str
    num_records: int
    license: str = "unknown"
    pii_scanned: bool = False
    contamination_checked: bool = False

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name must be non-empty")
        if not self.content_hash:
            raise ValueError("content_hash must be non-empty")
        if self.num_records < 0:
            raise ValueError("num_records must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "content_hash": self.content_hash,
            "num_records": self.num_records,
            "license": self.license,
            "pii_scanned": self.pii_scanned,
            "contamination_checked": self.contamination_checked,
        }


@dataclass(frozen=True)
class CodeComponent:
    """The code and environment that produced a model.

    Example:
        >>> CodeComponent(git_sha="deadbeef", dependencies_hash="d1", framework="trl").framework
        'trl'
    """

    git_sha: str
    dependencies_hash: str
    framework: str

    def __post_init__(self) -> None:
        if not self.git_sha:
            raise ValueError("git_sha must be non-empty")
        if not self.dependencies_hash:
            raise ValueError("dependencies_hash must be non-empty")
        if not self.framework:
            raise ValueError("framework must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "git_sha": self.git_sha,
            "dependencies_hash": self.dependencies_hash,
            "framework": self.framework,
        }


@dataclass(frozen=True)
class EvalComponent:
    """A single benchmark result for a model.

    Example:
        >>> EvalComponent(benchmark="mmlu", score=0.71).score
        0.71
    """

    benchmark: str
    score: float
    contaminated: bool = False

    def __post_init__(self) -> None:
        if not self.benchmark:
            raise ValueError("benchmark must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark": self.benchmark,
            "score": self.score,
            "contaminated": self.contaminated,
        }


@dataclass(frozen=True)
class ModelBOM:
    """A Model Bill-of-Materials: the full lineage of a trained model.

    The BOM records what data, code, evaluations, and configuration produced a
    model. Its :meth:`canonical_json` is deterministic and forms the basis for
    tamper-evident hashing and signing.

    Example:
        >>> bom = ModelBOM(
        ...     model_id="m1",
        ...     base_model="base",
        ...     run_id="r1",
        ...     data=[DataComponent(name="d", content_hash="h", num_records=1)],
        ...     code=CodeComponent(git_sha="s", dependencies_hash="dh", framework="trl"),
        ...     evals=[EvalComponent(benchmark="mmlu", score=0.5)],
        ...     hyperparameters={"lr": 0.001},
        ... )
        >>> len(bom.content_hash())
        64
    """

    model_id: str
    base_model: str
    run_id: str
    data: list[DataComponent]
    code: CodeComponent
    evals: list[EvalComponent]
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""

    def __post_init__(self) -> None:
        if not self.model_id:
            raise ValueError("model_id must be non-empty")
        if not self.base_model:
            raise ValueError("base_model must be non-empty")
        if not self.run_id:
            raise ValueError("run_id must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "base_model": self.base_model,
            "run_id": self.run_id,
            "data": [component.to_dict() for component in self.data],
            "code": self.code.to_dict(),
            "evals": [component.to_dict() for component in self.evals],
            "hyperparameters": self.hyperparameters,
            "created_at": self.created_at,
        }

    def canonical_json(self) -> str:
        """Return a deterministic, sorted-keys JSON serialization of the BOM.

        The output is independent of dict insertion order, making it a stable
        basis for hashing and signing.
        """
        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )

    def content_hash(self) -> str:
        """Return the SHA-256 hex digest of :meth:`canonical_json`."""
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    def risk_flags(self) -> list[str]:
        """Return derived compliance warnings for this BOM.

        Flags (sorted, de-duplicated):
            - ``"unscanned_pii"`` if any data component was not PII-scanned.
            - ``"unchecked_contamination"`` if any data component was not
              contamination-checked.
            - ``"unknown_license"`` if any data component has an unknown license.
            - ``"contaminated_eval"`` if any evaluation is flagged contaminated.
        """
        flags: set[str] = set()
        for component in self.data:
            if not component.pii_scanned:
                flags.add("unscanned_pii")
            if not component.contamination_checked:
                flags.add("unchecked_contamination")
            if not component.license or component.license.lower() == "unknown":
                flags.add("unknown_license")
        for evaluation in self.evals:
            if evaluation.contaminated:
                flags.add("contaminated_eval")
        return sorted(flags)
