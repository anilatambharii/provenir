"""RAG Corpus Trust Layer for Provenir.

Scans a retrieval-augmented-generation corpus directory for PII, training-data
overlap, and retracted documents.  The scan result is summarised into a
:class:`CorpusComponent` that can be attached to a ModelBOM.

Example::

    from provenir.governance.rag_corpus import RAGCorpusScanner, gate_rag_corpus

    scanner = RAGCorpusScanner(
        known_retracted=frozenset({"10.1234/bad"}),
        train_hashes=frozenset({"abc123..."}),
    )
    report = scanner.scan("/path/to/corpus")
    print(report.summary())

    gate_rag_corpus(report, "production", protected_stages=["production"])
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# PII regex patterns (mirrors provenir.governance.pii._PATTERNS, stdlib only)
# ---------------------------------------------------------------------------

_PII_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"),  # email
    re.compile(
        r"\b(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}\b"
    ),  # US phone
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),  # SSN
]

# DOI pattern: "doi: 10.XXXX/..." in the first 500 bytes of a document
_DOI_PATTERN: re.Pattern[str] = re.compile(
    r"doi:\s*(10\.\d{4,}(?:\.\d+)*/\S+)", re.IGNORECASE
)

# File extensions scanned by default
_SCAN_EXTENSIONS: frozenset[str] = frozenset({".txt", ".md", ".json", ".pdf"})


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _has_pii(text: str) -> bool:
    return any(pattern.search(text) for pattern in _PII_PATTERNS)


def _extract_doi(header_text: str) -> str:
    """Return the first DOI found in *header_text*, or empty string."""
    m = _DOI_PATTERN.search(header_text)
    if m:
        return m.group(1).rstrip(".,;)")
    return ""


# ---------------------------------------------------------------------------
# CorpusDocument
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CorpusDocument:
    """Metadata and trust flags for a single file in the RAG corpus.

    Example:
        >>> doc = CorpusDocument(path="a.txt", content_hash="abc", size_bytes=10)
        >>> doc.pii_found
        False
    """

    path: str
    content_hash: str
    size_bytes: int
    pii_found: bool = False
    in_training: bool = False
    retraction_doi: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "content_hash": self.content_hash,
            "size_bytes": self.size_bytes,
            "pii_found": self.pii_found,
            "in_training": self.in_training,
            "retraction_doi": self.retraction_doi,
        }


# ---------------------------------------------------------------------------
# RAGCorpusScanReport
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RAGCorpusScanReport:
    """Full result of scanning a RAG corpus directory.

    Example:
        >>> r = RAGCorpusScanReport("c/", 0, 0, 0, 0, [], "none")
        >>> r.risk_level
        'none'
    """

    corpus_dir: str
    total_docs: int
    pii_count: int
    training_overlap_count: int
    retracted_count: int
    documents: list[CorpusDocument]
    risk_level: str  # "none" | "low" | "medium" | "high"

    def summary(self) -> str:
        """One human-readable line per metric."""
        lines = [
            f"corpus_dir: {self.corpus_dir}",
            f"total_docs: {self.total_docs}",
            f"pii_count: {self.pii_count}",
            f"training_overlap_count: {self.training_overlap_count}",
            f"retracted_count: {self.retracted_count}",
            f"risk_level: {self.risk_level}",
        ]
        return "\n".join(lines)

    def content_hash(self) -> str:
        """SHA-256 of the canonical JSON representation."""
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "corpus_dir": self.corpus_dir,
            "total_docs": self.total_docs,
            "pii_count": self.pii_count,
            "training_overlap_count": self.training_overlap_count,
            "retracted_count": self.retracted_count,
            "documents": [doc.to_dict() for doc in self.documents],
            "risk_level": self.risk_level,
        }


# ---------------------------------------------------------------------------
# CorpusComponent  (summary only — goes into ModelBOM)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CorpusComponent:
    """BOM component that records the RAG corpus trust summary.

    Intended to be attached to a ModelBOM so the corpus verdict is inside the
    signed envelope of any :class:`~provenir.governance.passport.ModelPassport`.

    Example:
        >>> cc = CorpusComponent("c/", 5, 0, 0, 0, "none", "abc")
        >>> cc.to_dict()["risk_level"]
        'none'
    """

    corpus_dir: str
    total_docs: int
    pii_count: int
    training_overlap_count: int
    retracted_count: int
    risk_level: str
    report_hash: str

    @classmethod
    def from_report(cls, report: RAGCorpusScanReport) -> CorpusComponent:
        """Construct a :class:`CorpusComponent` from a :class:`RAGCorpusScanReport`."""
        return cls(
            corpus_dir=report.corpus_dir,
            total_docs=report.total_docs,
            pii_count=report.pii_count,
            training_overlap_count=report.training_overlap_count,
            retracted_count=report.retracted_count,
            risk_level=report.risk_level,
            report_hash=report.content_hash(),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "corpus_dir": self.corpus_dir,
            "total_docs": self.total_docs,
            "pii_count": self.pii_count,
            "training_overlap_count": self.training_overlap_count,
            "retracted_count": self.retracted_count,
            "risk_level": self.risk_level,
            "report_hash": self.report_hash,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CorpusComponent:
        """Reconstruct a :class:`CorpusComponent` from a serialised dict."""
        return cls(
            corpus_dir=str(data["corpus_dir"]),
            total_docs=int(data["total_docs"]),
            pii_count=int(data["pii_count"]),
            training_overlap_count=int(data["training_overlap_count"]),
            retracted_count=int(data["retracted_count"]),
            risk_level=str(data["risk_level"]),
            report_hash=str(data["report_hash"]),
        )


# ---------------------------------------------------------------------------
# RAGCorpusScanner
# ---------------------------------------------------------------------------


def _compute_risk_level(
    pii_count: int,
    retracted_count: int,
    training_overlap_count: int,
    total_docs: int,
) -> str:
    """Derive risk level from scan metrics."""
    if retracted_count > 0 or pii_count > 0:
        return "high"
    if total_docs > 0 and training_overlap_count > total_docs * 0.1:
        return "medium"
    if training_overlap_count > 0:
        return "low"
    return "none"


class RAGCorpusScanner:
    """Scan a RAG corpus directory for PII, training overlap, and retractions.

    Parameters
    ----------
    known_retracted:
        Frozenset of retracted DOI strings (same pattern as RetractionMonitor).
        ``None`` means the empty set — no retraction checking.
    train_hashes:
        Frozenset of SHA-256 hex digests of training-set document bytes.
        ``None`` means the empty set — no overlap checking.

    Example::

        scanner = RAGCorpusScanner(
            known_retracted=frozenset({"10.1234/bad"}),
            train_hashes=frozenset({"deadbeef..."}),
        )
        report = scanner.scan("/my/corpus")
    """

    def __init__(
        self,
        known_retracted: frozenset[str] | None = None,
        train_hashes: frozenset[str] | None = None,
    ) -> None:
        self._known_retracted: frozenset[str] = (
            known_retracted if known_retracted is not None else frozenset()
        )
        self._train_hashes: frozenset[str] = (
            train_hashes if train_hashes is not None else frozenset()
        )

    def scan(self, corpus_dir: str | Path) -> RAGCorpusScanReport:
        """Recursively scan *corpus_dir* and return a :class:`RAGCorpusScanReport`.

        Reads all ``.txt``, ``.md``, ``.json``, and ``.pdf`` files.  For each file
        the scanner:

        1. Reads raw bytes and computes the SHA-256 content hash.
        2. Checks whether the hash is in *train_hashes* (training overlap).
        3. Decodes the first 500 bytes as UTF-8 (errors ignored) and extracts a
           DOI if present; checks the DOI against *known_retracted*.
        4. Decodes the full content as UTF-8 (errors ignored) and runs PII regex
           checks.
        """
        return self._run_scan(Path(corpus_dir), self._train_hashes)

    def check_train_overlap(
        self,
        corpus_dir: str | Path,
        train_hashes: frozenset[str],
    ) -> RAGCorpusScanReport:
        """Convenience: scan with an explicit *train_hashes* set (overrides __init__)."""
        return self._run_scan(Path(corpus_dir), train_hashes)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run_scan(
        self,
        corpus_dir: Path,
        train_hashes: frozenset[str],
    ) -> RAGCorpusScanReport:
        documents: list[CorpusDocument] = []

        if corpus_dir.is_dir():
            candidates = sorted(
                p
                for p in corpus_dir.rglob("*")
                if p.is_file() and p.suffix.lower() in _SCAN_EXTENSIONS
            )
            for file_path in candidates:
                doc = self._scan_file(file_path, corpus_dir, train_hashes)
                documents.append(doc)

        total = len(documents)
        pii_count = sum(1 for d in documents if d.pii_found)
        overlap_count = sum(1 for d in documents if d.in_training)
        retracted_count = sum(1 for d in documents if d.retraction_doi)
        risk = _compute_risk_level(pii_count, retracted_count, overlap_count, total)

        return RAGCorpusScanReport(
            corpus_dir=str(corpus_dir),
            total_docs=total,
            pii_count=pii_count,
            training_overlap_count=overlap_count,
            retracted_count=retracted_count,
            documents=documents,
            risk_level=risk,
        )

    def _scan_file(
        self,
        file_path: Path,
        corpus_dir: Path,
        train_hashes: frozenset[str],
    ) -> CorpusDocument:
        raw_bytes = file_path.read_bytes()
        content_hash = _sha256_hex(raw_bytes)
        size_bytes = len(raw_bytes)

        # Relative path for portability
        try:
            rel_path = str(file_path.relative_to(corpus_dir))
        except ValueError:
            rel_path = str(file_path)

        # Training overlap
        in_training = content_hash in train_hashes

        # DOI extraction from header (first 500 bytes)
        header_text = raw_bytes[:500].decode("utf-8", errors="ignore")
        retraction_doi = _extract_doi(header_text)
        if retraction_doi and retraction_doi not in self._known_retracted:
            retraction_doi = ""

        # PII check on full text
        full_text = raw_bytes.decode("utf-8", errors="ignore")
        pii_found = _has_pii(full_text)

        return CorpusDocument(
            path=rel_path,
            content_hash=content_hash,
            size_bytes=size_bytes,
            pii_found=pii_found,
            in_training=in_training,
            retraction_doi=retraction_doi,
        )


# ---------------------------------------------------------------------------
# Exception and gate
# ---------------------------------------------------------------------------


class RAGCorpusBlocked(RuntimeError):
    """Raised by :func:`gate_rag_corpus` when a protected stage is blocked."""


def gate_rag_corpus(
    report: RAGCorpusScanReport,
    stage: str,
    *,
    protected_stages: list[str],
    allow_pii: bool = False,
    allow_retraction: bool = False,
) -> None:
    """Raise :class:`RAGCorpusBlocked` if *report* blocks promotion to *stage*.

    Only stages listed in *protected_stages* are checked.  Non-protected stages
    always pass through.

    Parameters
    ----------
    report:
        The scan report to evaluate.
    stage:
        The deployment stage being gated (e.g. ``"production"``).
    protected_stages:
        Stages that enforce trust constraints.
    allow_pii:
        If ``True``, PII findings do not block the stage.
    allow_retraction:
        If ``True``, retracted documents do not block the stage.

    Raises
    ------
    RAGCorpusBlocked
        When *stage* is protected and a blocking condition is present.

    Example::

        gate_rag_corpus(clean_report, "production", protected_stages=["production"])
        gate_rag_corpus(pii_report, "staging", protected_stages=["production"])  # ok
    """
    if stage not in protected_stages:
        return

    reasons: list[str] = []
    if report.pii_count > 0 and not allow_pii:
        reasons.append(f"pii_count={report.pii_count}")
    if report.retracted_count > 0 and not allow_retraction:
        reasons.append(f"retracted_count={report.retracted_count}")

    if reasons:
        raise RAGCorpusBlocked(
            f"Stage {stage!r} blocked by RAG corpus trust layer: "
            + ", ".join(reasons)
            + f". risk_level={report.risk_level}"
        )
