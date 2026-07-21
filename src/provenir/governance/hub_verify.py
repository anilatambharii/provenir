from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Weight file extensions that count toward the composite hash
_WEIGHT_SUFFIXES = frozenset({".bin", ".safetensors", ".pt", ".ckpt", ".gguf"})


class HubVerifyError(RuntimeError):
    """Raised for missing optional dependencies or unreachable Hub."""


@dataclass(frozen=True)
class HubFileEntry:
    """A single file found during a local model directory scan.

    Example:
        >>> entry = HubFileEntry("model.safetensors", "abc123", 1024)
        >>> entry.filename
        'model.safetensors'
    """

    filename: str
    content_hash: str
    size_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "filename": self.filename,
            "content_hash": self.content_hash,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True)
class HubVerificationReport:
    """Result of verifying a local model directory against its passport.

    Example:
        >>> report = HubVerificationReport(
        ...     repo_id="org/model",
        ...     local_path="/tmp/model",
        ...     files=[],
        ...     composite_hash="abc",
        ...     passport_found=False,
        ...     passport_hash_match=None,
        ...     passport_path="",
        ...     risk_flags=[],
        ...     verified=False,
        ... )
        >>> report.passport_found
        False
    """

    repo_id: str
    local_path: str
    files: list[HubFileEntry]
    composite_hash: str
    passport_found: bool
    passport_hash_match: bool | None
    passport_path: str
    risk_flags: list[str]
    verified: bool

    def summary(self) -> str:
        """Return a human-readable one-line summary of the verification result."""
        if self.verified:
            return (
                f"VERIFIED repo_id={self.repo_id!r} "
                f"files={len(self.files)} composite={self.composite_hash[:12]}..."
            )
        if not self.passport_found:
            return (
                f"NO PASSPORT repo_id={self.repo_id!r} "
                f"files={len(self.files)} composite={self.composite_hash[:12]}..."
            )
        if self.passport_hash_match is False:
            return (
                f"MISMATCH repo_id={self.repo_id!r} "
                f"files={len(self.files)} composite={self.composite_hash[:12]}..."
            )
        # passport found, hash matched but risk flags blocked verification
        flags_str = ", ".join(self.risk_flags)
        return (
            f"RISK_FLAGS repo_id={self.repo_id!r} "
            f"files={len(self.files)} flags=[{flags_str}]"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "repo_id": self.repo_id,
            "local_path": self.local_path,
            "files": [f.to_dict() for f in self.files],
            "composite_hash": self.composite_hash,
            "passport_found": self.passport_found,
            "passport_hash_match": self.passport_hash_match,
            "passport_path": self.passport_path,
            "risk_flags": list(self.risk_flags),
            "verified": self.verified,
        }

    def content_hash(self) -> str:
        """Return SHA-256 of the canonical JSON representation of this report."""
        return hashlib.sha256(
            json.dumps(self.to_dict(), sort_keys=True).encode("utf-8")
        ).hexdigest()


def _hash_file(path: Path) -> str:
    """Return the SHA-256 hex digest of a file's bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _compute_composite(hashes: list[str]) -> str:
    """Return SHA-256 of sorted hashes joined by newlines."""
    joined = "\n".join(sorted(hashes))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


class HubPassportVerifier:
    """Verify a local model directory against a stored ModelPassport.

    If *passport_store_dir* is provided, ``passport.json`` is searched there
    first; the model directory itself is always checked as a fallback.

    Example:
        >>> import tempfile
        >>> verifier = HubPassportVerifier(tempfile.mkdtemp())
        >>> verifier  # doctest: +ELLIPSIS
        <...HubPassportVerifier ...>
    """

    def __init__(self, passport_store_dir: str | Path | None = None) -> None:
        self._passport_store_dir = Path(passport_store_dir) if passport_store_dir else None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _scan_files(self, model_dir: Path) -> list[HubFileEntry]:
        """Collect weight files and passport.json from *model_dir* (non-recursive)."""
        entries: list[HubFileEntry] = []
        for child in sorted(model_dir.iterdir()):
            if not child.is_file():
                continue
            if child.suffix.lower() in _WEIGHT_SUFFIXES or child.name == "passport.json":
                h = _hash_file(child)
                entries.append(HubFileEntry(child.name, h, child.stat().st_size))
        return entries

    def _find_passport_path(self, model_dir: Path) -> Path | None:
        """Return the first passport.json found in store dir then model dir."""
        candidates = []
        if self._passport_store_dir is not None:
            candidates.append(self._passport_store_dir / "passport.json")
        candidates.append(model_dir / "passport.json")
        for p in candidates:
            if p.exists():
                return p
        return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def verify_local(
        self, model_dir: str | Path, repo_id: str = ""
    ) -> HubVerificationReport:
        """Scan *model_dir* and verify it against a local passport.json.

        Weight files (``*.bin``, ``*.safetensors``, ``*.pt``, ``*.ckpt``,
        ``*.gguf``) are hashed; their sorted hashes are combined into a
        *composite_hash*.  If a ``passport.json`` is found (in the store dir
        or in *model_dir*), the passport's ``bom.content_hash()`` is compared
        to the composite hash.
        """
        model_dir = Path(model_dir)
        all_entries = self._scan_files(model_dir)

        # Separate weight files from passport.json for the composite
        weight_entries = [e for e in all_entries if e.filename != "passport.json"]
        weight_hashes = [e.content_hash for e in weight_entries]
        composite = _compute_composite(weight_hashes)

        # Locate passport
        passport_path = self._find_passport_path(model_dir)
        passport_found = passport_path is not None
        passport_hash_match: bool | None = None
        passport_path_str = ""
        risk_flags: list[str] = []

        if passport_path is not None:
            passport_path_str = str(passport_path)
            # Lazy import to avoid mandatory dependency on the full governance stack
            from provenir.governance.passport import ModelPassport

            try:
                data = json.loads(passport_path.read_text(encoding="utf-8"))
                passport = ModelPassport.from_dict(data)
                bom_hash = passport.bom.content_hash()
                passport_hash_match = bom_hash == composite
                risk_flags = passport.bom.risk_flags()
            except Exception:
                # Corrupted or unparseable passport — treat as mismatch
                passport_hash_match = False

        # Critical flags that block verification (all flags are treated as
        # blocking here; callers can implement severity tiers on top)
        _critical_flags = {"unsafe_model_scan", "spurious_reward", "retracted_training_data"}
        has_critical = bool(_critical_flags.intersection(risk_flags))

        verified = (
            passport_found
            and passport_hash_match is True
            and not has_critical
        )

        return HubVerificationReport(
            repo_id=repo_id,
            local_path=str(model_dir),
            files=weight_entries,
            composite_hash=composite,
            passport_found=passport_found,
            passport_hash_match=passport_hash_match,
            passport_path=passport_path_str,
            risk_flags=risk_flags,
            verified=verified,
        )

    def verify_hub(
        self,
        repo_id: str,
        *,
        cache_dir: str | Path | None = None,
    ) -> HubVerificationReport:
        """Download *repo_id* from HuggingFace Hub and call :meth:`verify_local`.

        Raises :class:`HubVerifyError` if ``huggingface_hub`` is not installed.
        """
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise HubVerifyError(
                "huggingface_hub not installed; use verify_local() instead"
            ) from exc

        download_kwargs: dict[str, Any] = {}
        if cache_dir is not None:
            download_kwargs["cache_dir"] = str(cache_dir)

        downloaded_path = snapshot_download(repo_id, **download_kwargs)
        return self.verify_local(downloaded_path, repo_id=repo_id)


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


def verify_model_dir(
    model_dir: str | Path,
    passport_store_dir: str | Path | None = None,
) -> HubVerificationReport:
    """Verify *model_dir* using an optionally-provided passport store.

    Convenience wrapper around :class:`HubPassportVerifier`.

    Example:
        >>> import tempfile, pathlib
        >>> d = pathlib.Path(tempfile.mkdtemp())
        >>> report = verify_model_dir(d)
        >>> report.passport_found
        False
    """
    return HubPassportVerifier(passport_store_dir).verify_local(model_dir)
