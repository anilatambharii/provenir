"""Tests for the RAG Corpus Trust Layer (provenir.governance.rag_corpus)."""

from __future__ import annotations

import hashlib

import pytest

from provenir.governance.rag_corpus import (
    CorpusComponent,
    RAGCorpusBlocked,
    RAGCorpusScanner,
    gate_rag_corpus,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _write(tmp_path, name: str, content: str) -> bytes:
    """Write a file and return its raw bytes (for hash computation)."""
    raw = content.encode("utf-8")
    (tmp_path / name).write_bytes(raw)
    return raw


# ---------------------------------------------------------------------------
# Test 1: Empty corpus directory
# ---------------------------------------------------------------------------


def test_empty_corpus(tmp_path) -> None:
    scanner = RAGCorpusScanner()
    report = scanner.scan(tmp_path)
    assert report.total_docs == 0
    assert report.pii_count == 0
    assert report.training_overlap_count == 0
    assert report.retracted_count == 0
    assert report.documents == []
    assert report.risk_level == "none"


# ---------------------------------------------------------------------------
# Test 2: Single clean text file
# ---------------------------------------------------------------------------


def test_single_clean_file(tmp_path) -> None:
    _write(tmp_path, "doc.txt", "This document contains no sensitive information.")
    scanner = RAGCorpusScanner()
    report = scanner.scan(tmp_path)
    assert report.total_docs == 1
    assert report.pii_count == 0
    assert report.training_overlap_count == 0
    assert report.retracted_count == 0
    assert report.documents[0].pii_found is False
    assert report.documents[0].in_training is False
    assert report.risk_level == "none"


# ---------------------------------------------------------------------------
# Test 3: File with email address
# ---------------------------------------------------------------------------


def test_file_with_email(tmp_path) -> None:
    _write(tmp_path, "email.txt", "Contact us at admin@example.com for support.")
    scanner = RAGCorpusScanner()
    report = scanner.scan(tmp_path)
    assert report.pii_count == 1
    assert report.documents[0].pii_found is True
    assert report.risk_level == "high"


# ---------------------------------------------------------------------------
# Test 4: File with US phone number
# ---------------------------------------------------------------------------


def test_file_with_phone(tmp_path) -> None:
    _write(tmp_path, "phone.txt", "Call our hotline: 555-867-5309 anytime.")
    scanner = RAGCorpusScanner()
    report = scanner.scan(tmp_path)
    assert report.pii_count == 1
    assert report.documents[0].pii_found is True
    assert report.risk_level == "high"


# ---------------------------------------------------------------------------
# Test 5: File with SSN pattern
# ---------------------------------------------------------------------------


def test_file_with_ssn(tmp_path) -> None:
    _write(tmp_path, "ssn.txt", "Patient SSN: 123-45-6789 on file.")
    scanner = RAGCorpusScanner()
    report = scanner.scan(tmp_path)
    assert report.pii_count == 1
    assert report.documents[0].pii_found is True
    assert report.risk_level == "high"


# ---------------------------------------------------------------------------
# Test 6: File hash matches train_hashes
# ---------------------------------------------------------------------------


def test_training_overlap(tmp_path) -> None:
    raw = _write(tmp_path, "train.md", "# Training document\nSome content here.")
    file_hash = _sha256(raw)
    scanner = RAGCorpusScanner(train_hashes=frozenset({file_hash}))
    report = scanner.scan(tmp_path)
    assert report.training_overlap_count == 1
    assert report.documents[0].in_training is True
    # overlap alone with 1/1 docs = 100% > 10% → "medium"
    assert report.risk_level == "medium"


# ---------------------------------------------------------------------------
# Test 7: File with retracted DOI
# ---------------------------------------------------------------------------


def test_retracted_doi(tmp_path) -> None:
    _write(tmp_path, "retracted.txt", "doi: 10.1234/test\n\nSome scientific content.")
    scanner = RAGCorpusScanner(known_retracted=frozenset({"10.1234/test"}))
    report = scanner.scan(tmp_path)
    assert report.retracted_count == 1
    assert report.documents[0].retraction_doi == "10.1234/test"
    assert report.risk_level == "high"


# ---------------------------------------------------------------------------
# Test 8: Multiple files, mixed PII and clean
# ---------------------------------------------------------------------------


def test_mixed_pii_and_clean(tmp_path) -> None:
    _write(tmp_path, "clean1.txt", "All good here, no sensitive data.")
    _write(tmp_path, "clean2.md", "Another clean document.")
    _write(tmp_path, "pii.txt", "Send to alice@corp.org for details.")
    scanner = RAGCorpusScanner()
    report = scanner.scan(tmp_path)
    assert report.total_docs == 3
    assert report.pii_count == 1
    assert report.risk_level == "high"
    pii_docs = [d for d in report.documents if d.pii_found]
    assert len(pii_docs) == 1


# ---------------------------------------------------------------------------
# Test 9: risk_level "medium" when overlap > 10% of docs
# ---------------------------------------------------------------------------


def test_risk_medium_on_high_overlap(tmp_path) -> None:
    # Create 10 docs, 2 of which are in training (20% > 10% threshold)
    train_hashes: set[str] = set()
    for i in range(10):
        content = f"Document number {i} with clean content.\n"
        raw = content.encode("utf-8")
        (tmp_path / f"doc{i}.txt").write_bytes(raw)
        if i < 2:
            train_hashes.add(_sha256(raw))

    scanner = RAGCorpusScanner(train_hashes=frozenset(train_hashes))
    report = scanner.scan(tmp_path)
    assert report.total_docs == 10
    assert report.training_overlap_count == 2
    assert report.pii_count == 0
    assert report.retracted_count == 0
    assert report.risk_level == "medium"


# ---------------------------------------------------------------------------
# Test 10: CorpusComponent.from_report() round-trip through to_dict/from_dict
# ---------------------------------------------------------------------------


def test_corpus_component_round_trip(tmp_path) -> None:
    _write(tmp_path, "a.txt", "Hello world.")
    scanner = RAGCorpusScanner()
    report = scanner.scan(tmp_path)
    component = CorpusComponent.from_report(report)

    d = component.to_dict()
    reconstructed = CorpusComponent.from_dict(d)

    assert reconstructed.corpus_dir == component.corpus_dir
    assert reconstructed.total_docs == component.total_docs
    assert reconstructed.pii_count == component.pii_count
    assert reconstructed.training_overlap_count == component.training_overlap_count
    assert reconstructed.retracted_count == component.retracted_count
    assert reconstructed.risk_level == component.risk_level
    assert reconstructed.report_hash == component.report_hash
    assert reconstructed == component


# ---------------------------------------------------------------------------
# Test 11: RAGCorpusScanReport.content_hash() is stable
# ---------------------------------------------------------------------------


def test_content_hash_stable(tmp_path) -> None:
    _write(tmp_path, "stable.txt", "Deterministic content for hashing.")
    scanner = RAGCorpusScanner()
    report = scanner.scan(tmp_path)
    h1 = report.content_hash()
    h2 = report.content_hash()
    assert h1 == h2
    assert len(h1) == 64  # SHA-256 hex


# ---------------------------------------------------------------------------
# Test 12: gate_rag_corpus raises RAGCorpusBlocked when PII in protected stage
# ---------------------------------------------------------------------------


def test_gate_blocks_pii_in_protected_stage(tmp_path) -> None:
    _write(tmp_path, "leak.txt", "Email: user@leak.com is here.")
    scanner = RAGCorpusScanner()
    report = scanner.scan(tmp_path)
    assert report.pii_count > 0
    with pytest.raises(RAGCorpusBlocked):
        gate_rag_corpus(
            report,
            "production",
            protected_stages=["production"],
        )


# ---------------------------------------------------------------------------
# Test 13: gate_rag_corpus does NOT raise when stage not protected
# ---------------------------------------------------------------------------


def test_gate_passes_unprotected_stage(tmp_path) -> None:
    _write(tmp_path, "leak.txt", "Email: user@leak.com is here.")
    scanner = RAGCorpusScanner()
    report = scanner.scan(tmp_path)
    # staging is not in protected_stages → should not raise
    gate_rag_corpus(
        report,
        "staging",
        protected_stages=["production"],
    )  # no exception expected


# ---------------------------------------------------------------------------
# Test 14: check_train_overlap convenience method
# ---------------------------------------------------------------------------


def test_check_train_overlap(tmp_path) -> None:
    raw = _write(tmp_path, "overlap.json", '{"text": "some training content"}')
    file_hash = _sha256(raw)
    scanner = RAGCorpusScanner()  # no train_hashes set at init
    report = scanner.check_train_overlap(
        tmp_path, train_hashes=frozenset({file_hash})
    )
    assert report.training_overlap_count == 1
    assert report.documents[0].in_training is True


# ---------------------------------------------------------------------------
# Test 15: Non-matching DOI is not flagged as retracted
# ---------------------------------------------------------------------------


def test_non_retracted_doi_not_flagged(tmp_path) -> None:
    _write(tmp_path, "clean_doi.txt", "doi: 10.9999/good\n\nLegitimate content.")
    scanner = RAGCorpusScanner(known_retracted=frozenset({"10.1234/bad"}))
    report = scanner.scan(tmp_path)
    assert report.retracted_count == 0
    assert report.documents[0].retraction_doi == ""
    assert report.risk_level == "none"


# ---------------------------------------------------------------------------
# Test 16: gate_rag_corpus with allow_pii=True does not block
# ---------------------------------------------------------------------------


def test_gate_allow_pii_flag(tmp_path) -> None:
    _write(tmp_path, "pii.txt", "SSN: 999-88-7777 disclosed.")
    scanner = RAGCorpusScanner()
    report = scanner.scan(tmp_path)
    assert report.pii_count == 1
    # allow_pii=True means PII alone should not block
    gate_rag_corpus(
        report,
        "production",
        protected_stages=["production"],
        allow_pii=True,
    )  # no exception


# ---------------------------------------------------------------------------
# Test 17: Only .txt/.md/.json/.pdf files are scanned (others ignored)
# ---------------------------------------------------------------------------


def test_only_supported_extensions_scanned(tmp_path) -> None:
    _write(tmp_path, "doc.txt", "valid text file")
    # write an unsupported extension
    (tmp_path / "image.png").write_bytes(b"\x89PNG\r\n")
    (tmp_path / "script.py").write_text("print('hello')")
    scanner = RAGCorpusScanner()
    report = scanner.scan(tmp_path)
    assert report.total_docs == 1  # only doc.txt
