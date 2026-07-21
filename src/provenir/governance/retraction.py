"""Retraction monitor for Provenir.

Tracks whether DOIs present in a model's training corpus appear in a known-retracted
set (fraud, errors, journal retractions). The result is signed into the BOM so any
downstream consumer can verify that retraction-awareness was applied.

Example::

    from provenir.governance.retraction import RetractionMonitor, gate_retraction

    monitor = RetractionMonitor(known_retracted=frozenset({"10.1234/bad.paper"}))
    report = monitor.check(["10.1234/bad.paper", "10.5678/good.paper"])
    print(report.summary())

    gate_retraction(report, "staging")       # no-op
    gate_retraction(report, "production")    # raises RetractionBlocked
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from provenir.governance.passport import ModelPassport


# ---------------------------------------------------------------------------
# Report (full detail, not stored in BOM)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RetractionReport:
    """Full result of a retraction check over a set of DOIs.

    Example:
        >>> r = RetractionReport(["10.1/a"], [], 0.0, "none")
        >>> r.risk_level
        'none'
    """

    checked_dois: list[str]
    retracted_dois: list[str]
    retraction_rate: float
    risk_level: str  # "none" | "low" | "high"

    def summary(self) -> str:
        """One-line human summary of the retraction check."""
        checked = len(self.checked_dois)
        retracted = len(self.retracted_dois)
        return (
            f"retraction check: {retracted}/{checked} DOIs retracted "
            f"(rate={self.retraction_rate:.2%}, risk={self.risk_level})"
        )

    def content_hash(self) -> str:
        """SHA-256 of canonical JSON — deterministic and signable."""
        canonical = json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "checked_dois": list(self.checked_dois),
            "retracted_dois": list(self.retracted_dois),
            "retraction_rate": self.retraction_rate,
            "risk_level": self.risk_level,
        }


# ---------------------------------------------------------------------------
# Component (primitives only — goes into the BOM)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RetractionComponent:
    """BOM component recording the result of a retraction check.

    Intended to be attached as ``ModelBOM.retraction`` so the retraction verdict
    is inside the signed envelope of any
    :class:`~provenir.governance.passport.ModelPassport`.

    Example:
        >>> rc = RetractionComponent(10, 0, 0.0, "none", "abc")
        >>> rc.to_dict()["risk_level"]
        'none'
    """

    checked_count: int
    retracted_count: int
    retraction_rate: float
    risk_level: str
    report_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "checked_count": self.checked_count,
            "retracted_count": self.retracted_count,
            "retraction_rate": self.retraction_rate,
            "risk_level": self.risk_level,
            "report_hash": self.report_hash,
        }

    @classmethod
    def from_report(cls, report: RetractionReport) -> RetractionComponent:
        """Construct a :class:`RetractionComponent` from a :class:`RetractionReport`."""
        return cls(
            checked_count=len(report.checked_dois),
            retracted_count=len(report.retracted_dois),
            retraction_rate=report.retraction_rate,
            risk_level=report.risk_level,
            report_hash=report.content_hash(),
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RetractionComponent:
        """Reconstruct a :class:`RetractionComponent` from a serialised dict."""
        return cls(
            checked_count=int(data["checked_count"]),
            retracted_count=int(data["retracted_count"]),
            retraction_rate=float(data["retraction_rate"]),
            risk_level=str(data["risk_level"]),
            report_hash=str(data["report_hash"]),
        )


# ---------------------------------------------------------------------------
# Monitor
# ---------------------------------------------------------------------------


def _compute_risk_level(retracted_count: int, retraction_rate: float) -> str:
    """Derive risk level from retraction statistics.

    Any retraction (even a single DOI) is considered high risk because the
    model may have internalised invalid scientific patterns.
    """
    if retracted_count == 0:
        return "none"
    # Any retraction is high risk; rate > 0.1 is also explicitly high
    return "high"


class RetractionMonitor:
    """Check a list of DOIs against a known-retracted set.

    The known-retracted set is supplied by the caller (offline, no network calls).
    If *known_retracted* is ``None`` the monitor uses an empty frozenset — it still
    runs and produces a valid report; it just finds nothing retracted unless a set
    is provided.

    Example::

        monitor = RetractionMonitor(known_retracted=frozenset({"10.1/bad"}))
        report = monitor.check(["10.1/bad", "10.2/good"])
        assert report.retracted_dois == ["10.1/bad"]
    """

    def __init__(self, known_retracted: frozenset[str] | None = None) -> None:
        self._known_retracted: frozenset[str] = (
            known_retracted if known_retracted is not None else frozenset()
        )

    def check(self, dois: list[str]) -> RetractionReport:
        """Check *dois* against the known-retracted set and return a report."""
        # Dedup while preserving order for determinism
        seen: set[str] = set()
        unique_dois: list[str] = []
        for doi in dois:
            if doi not in seen:
                seen.add(doi)
                unique_dois.append(doi)

        retracted = sorted(doi for doi in unique_dois if doi in self._known_retracted)
        checked = len(unique_dois)
        retracted_count = len(retracted)
        rate = retracted_count / checked if checked > 0 else 0.0
        risk = _compute_risk_level(retracted_count, rate)

        return RetractionReport(
            checked_dois=unique_dois,
            retracted_dois=retracted,
            retraction_rate=rate,
            risk_level=risk,
        )

    def check_passport(self, passport: ModelPassport) -> RetractionReport:
        """Collect all ``retraction_dois`` from every DataComponent in the passport's BOM.

        Flattens, deduplicates, then calls :meth:`check`.
        """
        all_dois: list[str] = []
        for data_component in passport.bom.data:
            all_dois.extend(data_component.retraction_dois)
        return self.check(all_dois)


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------


class RetractionBlocked(RuntimeError):
    """Raised when a retraction report blocks promotion to a protected stage."""


def gate_retraction(
    report: RetractionReport,
    stage: str,
    *,
    protected_stages: frozenset[str] = frozenset({"production", "prod", "release"}),
    allow_rate: float = 0.0,
) -> None:
    """Raise :class:`RetractionBlocked` if *report* blocks promotion to *stage*.

    Protected stages (production/prod/release by default) are blocked when
    ``report.retraction_rate > allow_rate``.  Non-protected stages always pass.

    Example::

        gate_retraction(clean_report, "production")          # no-op
        gate_retraction(dirty_report, "staging")             # no-op (not protected)
        gate_retraction(dirty_report, "production")          # raises RetractionBlocked
        gate_retraction(dirty_report, "production", allow_rate=1.0)  # no-op (all allowed)
    """
    if stage not in protected_stages:
        return
    if report.retraction_rate > allow_rate:
        raise RetractionBlocked(
            f"Stage {stage!r} blocked: retraction rate {report.retraction_rate:.2%} "
            f"exceeds allowed {allow_rate:.2%}. "
            f"{report.summary()}"
        )
