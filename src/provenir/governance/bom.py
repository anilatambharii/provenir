from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from provenir.environments.reward_validity import RewardValidityReport
    from provenir.governance.retraction import RetractionComponent
    from provenir.governance.scan import ScanComponent


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
    # EU AI Act Art. 53 / Annex XII provenance fields (additive, defaulted — zero breaking changes)
    source_category: str = "unknown"
    crawl_domains: dict[str, int] = field(default_factory=dict)
    optout_respected: bool | None = None
    retraction_dois: list[str] = field(default_factory=list)

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
            "source_category": self.source_category,
            "crawl_domains": dict(self.crawl_domains),
            "optout_respected": self.optout_respected,
            "retraction_dois": list(self.retraction_dois),
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
class RewardValidityComponent:
    """A signed-into-the-BOM summary of a reward's spurious-reward ablation result.

    Carries the verdict from a :class:`~provenir.environments.reward_validity.RewardValidityReport`
    so a Passport can attest "reward ``math`` scored validity 0.86, not spurious." Only primitives
    are stored (governance stays independent of the environments layer); build one from a report via
    :meth:`from_report`.

    Example:
        >>> RewardValidityComponent("math", "abc", 0.86, spurious=False).spurious
        False
    """

    reward_name: str
    report_hash: str
    validity: float
    spurious: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "reward_name": self.reward_name,
            "report_hash": self.report_hash,
            "validity": self.validity,
            "spurious": self.spurious,
        }

    @classmethod
    def from_report(cls, report: RewardValidityReport) -> RewardValidityComponent:
        """Build a component from a reward-validity report's signed summary."""
        return cls(
            reward_name=report.reward_name,
            report_hash=report.content_hash(),
            validity=report.validity,
            spurious=report.spurious,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RewardValidityComponent:
        return cls(
            reward_name=data["reward_name"],
            report_hash=data["report_hash"],
            validity=data["validity"],
            spurious=bool(data["spurious"]),
        )


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
    scan: ScanComponent | None = None
    reward_validity: RewardValidityComponent | None = None
    retraction: RetractionComponent | None = None
    parent_passport_hash: str | None = None

    def __post_init__(self) -> None:
        if not self.model_id:
            raise ValueError("model_id must be non-empty")
        if not self.base_model:
            raise ValueError("base_model must be non-empty")
        if not self.run_id:
            raise ValueError("run_id must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "model_id": self.model_id,
            "base_model": self.base_model,
            "run_id": self.run_id,
            "data": [component.to_dict() for component in self.data],
            "code": self.code.to_dict(),
            "evals": [component.to_dict() for component in self.evals],
            "hyperparameters": self.hyperparameters,
            "created_at": self.created_at,
            "scan": self.scan.to_dict() if self.scan is not None else None,
            "reward_validity": (
                self.reward_validity.to_dict() if self.reward_validity is not None else None
            ),
            "retraction": self.retraction.to_dict() if self.retraction is not None else None,
            "parent_passport_hash": self.parent_passport_hash,
        }
        return d

    def canonical_json(self) -> str:
        """Return a deterministic, sorted-keys JSON serialization of the BOM.

        The output is independent of dict insertion order, making it a stable
        basis for hashing and signing.  ``parent_passport_hash`` is included in
        the signed content, making fine-tune lineage tamper-evident: modifying
        any ancestor's BOM changes its content hash, which invalidates every
        descendant's ``parent_passport_hash`` pointer.
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
            - ``"unsafe_model_scan"`` if the attached scan found a critical/high issue.
            - ``"spurious_reward"`` if the attached reward-validity check flagged the
              reward signal as spurious.
            - ``"retracted_training_data"`` if the retraction monitor found retracted DOIs
              in the training corpus.
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
        if self.scan is not None and self.scan.unsafe:
            flags.add("unsafe_model_scan")
        if self.reward_validity is not None and self.reward_validity.spurious:
            flags.add("spurious_reward")
        if self.retraction is not None and self.retraction.retracted_count > 0:
            flags.add("retracted_training_data")
        return sorted(flags)
