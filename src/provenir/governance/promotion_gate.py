from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from provenir.governance.passport import ModelPassport


@dataclass(frozen=True)
class PromotionCheck:
    """Result of a single named promotion check.

    Example:
        >>> PromotionCheck("no_pii", True).passed
        True
    """

    name: str
    passed: bool
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class PromotionResult:
    """Aggregate result of all promotion checks for one model/stage pair.

    Example:
        >>> r = PromotionResult("m1", "production", True, [], [])
        >>> r.summary()
        'PASSED: m1 -> production'
    """

    model_id: str
    stage: str
    passed: bool
    checks: list[PromotionCheck]
    failed_checks: list[str]

    def summary(self) -> str:
        """Return a human-readable one-line (or multi-line if blocked) summary."""
        if self.passed:
            return f"PASSED: {self.model_id} -> {self.stage}"
        lines = [f"BLOCKED: {self.model_id} -> {self.stage}"]
        for check in self.checks:
            if not check.passed:
                lines.append(f"  - {check.name}: {check.detail}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "stage": self.stage,
            "passed": self.passed,
            "checks": [c.to_dict() for c in self.checks],
            "failed_checks": list(self.failed_checks),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, indent=2, ensure_ascii=False)


class PromotionBlocked(RuntimeError):
    """Raised when a passport fails one or more promotion checks.

    Example:
        >>> result = PromotionResult("m1", "production", False, [], ["no_pii"])
        >>> exc = PromotionBlocked(result)
        >>> exc.result.model_id
        'm1'
    """

    def __init__(self, result: PromotionResult) -> None:
        self.result = result
        super().__init__(result.summary())


class PromotionGate:
    """Configurable gate that evaluates a :class:`~provenir.governance.passport.ModelPassport`
    against a set of compliance requirements before promotion to a target stage.

    All checks are opt-in via constructor keyword arguments; a gate constructed
    with all defaults performs no checks and always passes.

    Example::

        gate = PromotionGate(require_no_pii=True, require_signed=True)
        result = gate.evaluate(passport, "production")
        print(result.summary())
    """

    def __init__(
        self,
        *,
        require_scan: bool = False,
        require_no_retraction: bool = False,
        min_validity: float | None = None,
        require_no_pii: bool = False,
        require_no_contamination: bool = False,
        require_signed: bool = False,
    ) -> None:
        self._require_scan = require_scan
        self._require_no_retraction = require_no_retraction
        self._min_validity = min_validity
        self._require_no_pii = require_no_pii
        self._require_no_contamination = require_no_contamination
        self._require_signed = require_signed

    def evaluate(self, passport: ModelPassport, stage: str = "production") -> PromotionResult:
        """Run all enabled checks and return a :class:`PromotionResult`.

        No exception is raised regardless of outcome — use :meth:`gate` to raise
        on failure.
        """
        flags = set(passport.bom.risk_flags())
        checks: list[PromotionCheck] = []

        if self._require_no_pii:
            if "unscanned_pii" in flags:
                checks.append(
                    PromotionCheck(
                        "no_pii",
                        False,
                        "one or more data components were not PII-scanned (unscanned_pii flag)",
                    )
                )
            else:
                checks.append(PromotionCheck("no_pii", True))

        if self._require_no_contamination:
            contamination_flags = flags & {"unchecked_contamination", "contaminated_eval"}
            if contamination_flags:
                checks.append(
                    PromotionCheck(
                        "no_contamination",
                        False,
                        f"contamination flags present: {sorted(contamination_flags)}",
                    )
                )
            else:
                checks.append(PromotionCheck("no_contamination", True))

        if self._require_scan:
            if "unsafe_model_scan" in flags or passport.bom.scan is None:
                detail = (
                    "model scan is missing (bom.scan is None)"
                    if passport.bom.scan is None
                    else "model scan found unsafe findings (unsafe_model_scan flag)"
                )
                checks.append(PromotionCheck("scan_clean", False, detail))
            else:
                checks.append(PromotionCheck("scan_clean", True))

        if self._require_no_retraction:
            if "retracted_training_data" in flags:
                checks.append(
                    PromotionCheck(
                        "no_retraction",
                        False,
                        "retracted training data found (retracted_training_data flag)",
                    )
                )
            else:
                checks.append(PromotionCheck("no_retraction", True))

        if self._min_validity is not None:
            rv = passport.bom.reward_validity
            if rv is None:
                checks.append(
                    PromotionCheck(
                        "reward_valid",
                        False,
                        "no reward_validity component present in BOM",
                    )
                )
            elif rv.spurious:
                checks.append(
                    PromotionCheck(
                        "reward_valid",
                        False,
                        f"reward flagged as spurious (validity={rv.validity})",
                    )
                )
            elif rv.validity < self._min_validity:
                checks.append(
                    PromotionCheck(
                        "reward_valid",
                        False,
                        f"reward validity {rv.validity} < required {self._min_validity}",
                    )
                )
            else:
                checks.append(PromotionCheck("reward_valid", True))

        if self._require_signed:
            if passport.attestation is None:
                checks.append(
                    PromotionCheck(
                        "signed",
                        False,
                        "passport has no attestation (unsigned)",
                    )
                )
            else:
                checks.append(PromotionCheck("signed", True))

        failed = [c.name for c in checks if not c.passed]
        passed = len(failed) == 0

        return PromotionResult(
            model_id=passport.bom.model_id,
            stage=stage,
            passed=passed,
            checks=checks,
            failed_checks=failed,
        )

    def gate(self, passport: ModelPassport, stage: str = "production") -> None:
        """Evaluate the passport and raise :class:`PromotionBlocked` if any check fails."""
        result = self.evaluate(passport, stage)
        if not result.passed:
            raise PromotionBlocked(result)


def load_and_gate(
    passport_path: str | Path,
    stage: str = "production",
    *,
    require_scan: bool = False,
    require_no_retraction: bool = False,
    min_validity: float | None = None,
    require_no_pii: bool = False,
    require_signed: bool = False,
) -> PromotionResult:
    """Load a passport from a JSON file and run the promotion gate.

    Imports :class:`~provenir.governance.passport.ModelPassport` locally to
    avoid circular import issues at module level.

    Raises :class:`PromotionBlocked` if any enabled check fails; otherwise
    returns the :class:`PromotionResult`.
    """
    from provenir.governance.passport import ModelPassport  # local import — avoids circular

    passport_path = Path(passport_path)
    data = json.loads(passport_path.read_text(encoding="utf-8"))
    passport = ModelPassport.from_dict(data)

    gate = PromotionGate(
        require_scan=require_scan,
        require_no_retraction=require_no_retraction,
        min_validity=min_validity,
        require_no_pii=require_no_pii,
        require_signed=require_signed,
    )
    gate.gate(passport, stage)
    return gate.evaluate(passport, stage)
