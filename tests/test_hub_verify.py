"""Tests for provenir.governance.hub_verify."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

from provenir.governance.hub_verify import (
    HubFileEntry,
    HubPassportVerifier,
    HubVerificationReport,
    HubVerifyError,
    verify_model_dir,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _composite(hashes: list[str]) -> str:
    joined = "\n".join(sorted(hashes))
    return hashlib.sha256(joined.encode()).hexdigest()


def _write_file(path: Path, content: bytes) -> str:
    """Write *content* to *path* and return its SHA-256 hex digest."""
    path.write_bytes(content)
    return _sha256(content)


def _make_passport_json(model_id: str = "m1") -> str:
    """Return a minimal valid ModelPassport JSON string."""
    from provenir.governance.bom import (
        CodeComponent,
        DataComponent,
        EvalComponent,
        ModelBOM,
    )
    from provenir.governance.passport import ModelPassport

    bom = ModelBOM(
        model_id=model_id,
        base_model="base",
        run_id="r1",
        data=[DataComponent(name="d", content_hash="h", num_records=1)],
        code=CodeComponent(git_sha="s", dependencies_hash="dh", framework="trl"),
        evals=[EvalComponent(benchmark="mmlu", score=0.5)],
        hyperparameters={},
    )
    passport = ModelPassport(bom=bom, attestation=None)
    return passport.to_json()


# ---------------------------------------------------------------------------
# 1. Empty directory → 0 weight files, composite_hash is hash of empty string
# ---------------------------------------------------------------------------


def test_empty_directory(tmp_path: Path) -> None:
    report = verify_model_dir(tmp_path)
    assert report.files == []
    # composite of no hashes = sha256 of ""
    expected = hashlib.sha256(b"").hexdigest()
    assert report.composite_hash == expected
    assert not report.passport_found
    assert report.passport_hash_match is None
    assert not report.verified


# ---------------------------------------------------------------------------
# 2. Single weight file (*.safetensors) → files has one entry, correct hash
# ---------------------------------------------------------------------------


def test_single_safetensors(tmp_path: Path) -> None:
    content = b"fake safetensor weights"
    _write_file(tmp_path / "model.safetensors", content)
    report = verify_model_dir(tmp_path)
    assert len(report.files) == 1
    entry = report.files[0]
    assert entry.filename == "model.safetensors"
    assert entry.content_hash == _sha256(content)
    assert entry.size_bytes == len(content)


# ---------------------------------------------------------------------------
# 3. Multiple weight files → composite_hash is stable (sorted order)
# ---------------------------------------------------------------------------


def test_multiple_weight_files_stable_composite(tmp_path: Path) -> None:
    h1 = _write_file(tmp_path / "shard1.bin", b"shard one")
    h2 = _write_file(tmp_path / "shard2.safetensors", b"shard two")
    h3 = _write_file(tmp_path / "shard3.pt", b"shard three")

    report1 = verify_model_dir(tmp_path)

    # Re-run; must produce same composite regardless of filesystem order
    report2 = verify_model_dir(tmp_path)
    assert report1.composite_hash == report2.composite_hash

    expected = _composite([h1, h2, h3])
    assert report1.composite_hash == expected
    assert len(report1.files) == 3


# ---------------------------------------------------------------------------
# 4. Non-weight files ignored (*.txt, *.py) unless it's passport.json
# ---------------------------------------------------------------------------


def test_non_weight_files_ignored(tmp_path: Path) -> None:
    _write_file(tmp_path / "readme.txt", b"docs")
    _write_file(tmp_path / "config.py", b"config = {}")
    _write_file(tmp_path / "tokenizer.json", b"{}")
    report = verify_model_dir(tmp_path)
    assert report.files == []


# ---------------------------------------------------------------------------
# 5. All weight extensions are recognised
# ---------------------------------------------------------------------------


def test_all_weight_extensions_recognised(tmp_path: Path) -> None:
    for ext in (".bin", ".safetensors", ".pt", ".ckpt", ".gguf"):
        _write_file(tmp_path / f"model{ext}", b"weights")
    report = verify_model_dir(tmp_path)
    assert len(report.files) == 5


# ---------------------------------------------------------------------------
# 6. passport.json present but hash MISMATCHES (default with real ModelBOM)
# ---------------------------------------------------------------------------


def test_passport_present_hash_mismatch(tmp_path: Path) -> None:
    _write_file(tmp_path / "model.safetensors", b"some weights")
    (tmp_path / "passport.json").write_text(_make_passport_json(), encoding="utf-8")

    report = verify_model_dir(tmp_path)
    assert report.passport_found
    assert report.passport_hash_match is False  # BOM hash != weight composite
    assert not report.verified


# ---------------------------------------------------------------------------
# 7. No passport → passport_found=False, passport_hash_match=None, verified=False
# ---------------------------------------------------------------------------


def test_no_passport(tmp_path: Path) -> None:
    _write_file(tmp_path / "model.bin", b"weights")
    report = verify_model_dir(tmp_path)
    assert not report.passport_found
    assert report.passport_hash_match is None
    assert not report.verified


# ---------------------------------------------------------------------------
# 8. passport in passport_store_dir (not model_dir) → found and checked
# ---------------------------------------------------------------------------


def test_passport_in_store_dir(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    store_dir = tmp_path / "store"
    model_dir.mkdir()
    store_dir.mkdir()

    _write_file(model_dir / "model.safetensors", b"weights")
    (store_dir / "passport.json").write_text(_make_passport_json(), encoding="utf-8")

    report = verify_model_dir(model_dir, passport_store_dir=store_dir)
    assert report.passport_found
    assert report.passport_path == str(store_dir / "passport.json")
    # hash mismatch is expected (BOM hash != weight composite)
    assert report.passport_hash_match is False


# ---------------------------------------------------------------------------
# 9. store_dir passport wins over model_dir passport
# ---------------------------------------------------------------------------


def test_store_dir_passport_takes_priority(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    store_dir = tmp_path / "store"
    model_dir.mkdir()
    store_dir.mkdir()

    (model_dir / "passport.json").write_text(
        _make_passport_json("model_dir_passport"), encoding="utf-8"
    )
    (store_dir / "passport.json").write_text(
        _make_passport_json("store_passport"), encoding="utf-8"
    )

    report = verify_model_dir(model_dir, passport_store_dir=store_dir)
    # Store-dir passport should be used
    assert report.passport_path == str(store_dir / "passport.json")


# ---------------------------------------------------------------------------
# 10. HubVerificationReport.summary() contains "VERIFIED" when verified=True
# ---------------------------------------------------------------------------


def test_summary_verified(tmp_path: Path) -> None:
    # Build a report with verified=True via a subclass that controls composite

    class PatchedVerifier(HubPassportVerifier):
        def verify_local(
            self, model_dir: str | Path, repo_id: str = ""
        ) -> HubVerificationReport:
            # Write a passport whose bom.content_hash() we can retrieve
            from provenir.governance.bom import (
                CodeComponent,
                DataComponent,
                EvalComponent,
                ModelBOM,
            )

            bom = ModelBOM(
                model_id="m1",
                base_model="base",
                run_id="r1",
                data=[
                    DataComponent(
                        name="d",
                        content_hash="h",
                        num_records=1,
                        pii_scanned=True,
                        contamination_checked=True,
                        license="MIT",
                    )
                ],
                code=CodeComponent(git_sha="s", dependencies_hash="dh", framework="trl"),
                evals=[EvalComponent(benchmark="mmlu", score=0.5)],
                hyperparameters={},
            )
            bom_hash = bom.content_hash()

            return HubVerificationReport(
                repo_id=repo_id,
                local_path=str(model_dir),
                files=[],
                composite_hash=bom_hash,  # matches bom.content_hash()
                passport_found=True,
                passport_hash_match=True,
                passport_path=str(Path(model_dir) / "passport.json"),
                risk_flags=[],
                verified=True,
            )

    pv = PatchedVerifier()
    report = pv.verify_local(tmp_path, repo_id="org/model")
    assert report.verified
    assert "VERIFIED" in report.summary()


# ---------------------------------------------------------------------------
# 11. summary() contains "NO PASSPORT" when passport_found=False
# ---------------------------------------------------------------------------


def test_summary_no_passport(tmp_path: Path) -> None:
    report = verify_model_dir(tmp_path)
    assert "NO PASSPORT" in report.summary()


# ---------------------------------------------------------------------------
# 12. summary() contains "MISMATCH" when passport found but hash mismatch
# ---------------------------------------------------------------------------


def test_summary_mismatch(tmp_path: Path) -> None:
    _write_file(tmp_path / "model.safetensors", b"weights")
    (tmp_path / "passport.json").write_text(_make_passport_json(), encoding="utf-8")
    report = verify_model_dir(tmp_path)
    assert "MISMATCH" in report.summary()


# ---------------------------------------------------------------------------
# 13. content_hash() is stable across two calls
# ---------------------------------------------------------------------------


def test_content_hash_stable(tmp_path: Path) -> None:
    report = verify_model_dir(tmp_path)
    assert report.content_hash() == report.content_hash()


# ---------------------------------------------------------------------------
# 14. content_hash() changes when report data changes
# ---------------------------------------------------------------------------


def test_content_hash_differs_for_different_reports(tmp_path: Path) -> None:
    d1 = tmp_path / "dir1"
    d2 = tmp_path / "dir2"
    d1.mkdir()
    d2.mkdir()
    _write_file(d1 / "model.bin", b"weights A")
    _write_file(d2 / "model.bin", b"weights B")

    r1 = verify_model_dir(d1)
    r2 = verify_model_dir(d2)
    assert r1.content_hash() != r2.content_hash()


# ---------------------------------------------------------------------------
# 15. verify_model_dir convenience function works
# ---------------------------------------------------------------------------


def test_verify_model_dir_convenience(tmp_path: Path) -> None:
    report = verify_model_dir(tmp_path)
    assert isinstance(report, HubVerificationReport)
    assert report.local_path == str(tmp_path)


# ---------------------------------------------------------------------------
# 16. HubFileEntry.to_dict() round-trips correctly
# ---------------------------------------------------------------------------


def test_hub_file_entry_to_dict() -> None:
    entry = HubFileEntry("model.bin", "deadbeef", 512)
    d = entry.to_dict()
    assert d == {"filename": "model.bin", "content_hash": "deadbeef", "size_bytes": 512}


# ---------------------------------------------------------------------------
# 17. HubVerificationReport.to_dict() contains expected keys
# ---------------------------------------------------------------------------


def test_hub_verification_report_to_dict(tmp_path: Path) -> None:
    report = verify_model_dir(tmp_path)
    d = report.to_dict()
    for key in (
        "repo_id",
        "local_path",
        "files",
        "composite_hash",
        "passport_found",
        "passport_hash_match",
        "passport_path",
        "risk_flags",
        "verified",
    ):
        assert key in d


# ---------------------------------------------------------------------------
# 18. verify_hub raises HubVerifyError when huggingface_hub not installed
# ---------------------------------------------------------------------------


def test_verify_hub_raises_when_no_hf_hub(monkeypatch: pytest.MonkeyPatch) -> None:
    # Simulate huggingface_hub being absent
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)  # type: ignore[arg-type]

    verifier = HubPassportVerifier()
    with pytest.raises(HubVerifyError, match="huggingface_hub not installed"):
        verifier.verify_hub("org/model")


# ---------------------------------------------------------------------------
# 19. Corrupted passport.json is treated as mismatch (not exception)
# ---------------------------------------------------------------------------


def test_corrupted_passport_treated_as_mismatch(tmp_path: Path) -> None:
    _write_file(tmp_path / "model.bin", b"weights")
    (tmp_path / "passport.json").write_text("not valid json{{{{", encoding="utf-8")

    report = verify_model_dir(tmp_path)
    assert report.passport_found
    assert report.passport_hash_match is False
    assert not report.verified


# ---------------------------------------------------------------------------
# 20. repo_id is propagated to report
# ---------------------------------------------------------------------------


def test_repo_id_propagated(tmp_path: Path) -> None:
    verifier = HubPassportVerifier()
    report = verifier.verify_local(tmp_path, repo_id="meta-llama/Llama-3-8B")
    assert report.repo_id == "meta-llama/Llama-3-8B"
