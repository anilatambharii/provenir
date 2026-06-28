from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import huggingface_hub

    _HAS_HUB = True
except ImportError:
    _HAS_HUB = False


@dataclass(frozen=True)
class HubConfig:
    """Configuration for HuggingFace Hub push / pull operations."""

    repo_id: str  # "username/model-name"
    private: bool = False
    token: str | None = None
    revision: str = "main"

    def __post_init__(self) -> None:
        if "/" not in self.repo_id:
            raise ValueError(
                f"repo_id must be 'username/repo-name', got {self.repo_id!r}"
            )


@dataclass
class HubPushResult:
    """Result of a push operation."""

    repo_id: str
    url: str
    commit_sha: str | None = None
    files_pushed: list[str] | None = None


class HubClient:
    """HuggingFace Hub integration for pushing and pulling adapters.

    Requires ``huggingface_hub``: ``pip install provenir[hub]``.

    Operations that require the package return stub / empty results when it
    is unavailable, so orchestration code works without a Hub connection.
    ``pull_model`` raises :class:`ImportError` because it cannot return a
    usable path without the package.
    """

    def __init__(self, token: str | None = None) -> None:
        self._token = token

    def push_adapter(
        self,
        adapter_path: Path,
        config: HubConfig,
        model_card: str | None = None,
    ) -> HubPushResult:
        """Push a LoRA adapter directory to the Hub."""
        if not _HAS_HUB:
            return HubPushResult(
                repo_id=config.repo_id,
                url=f"https://huggingface.co/{config.repo_id}",
            )

        token = config.token or self._token
        api = huggingface_hub.HfApi(token=token)
        api.create_repo(
            repo_id=config.repo_id,
            repo_type="model",
            private=config.private,
            exist_ok=True,
        )

        if model_card is not None:
            (adapter_path / "README.md").write_text(model_card, encoding="utf-8")

        commit_info = api.upload_folder(
            folder_path=str(adapter_path),
            repo_id=config.repo_id,
            repo_type="model",
            revision=config.revision,
            commit_message="Upload adapter via Provenir",
        )

        pushed_files = [
            getattr(f, "path_in_repo", getattr(f, "rfilename", str(f)))
            for f in api.list_repo_tree(config.repo_id)
        ]
        return HubPushResult(
            repo_id=config.repo_id,
            url=f"https://huggingface.co/{config.repo_id}",
            commit_sha=getattr(commit_info, "oid", None),
            files_pushed=pushed_files,
        )

    def pull_model(
        self,
        repo_id: str,
        revision: str = "main",
        cache_dir: Path | None = None,
        token: str | None = None,
    ) -> Path:
        """Download a model or adapter from the Hub and return its local path."""
        if not _HAS_HUB:
            raise ImportError(
                "huggingface_hub required: pip install provenir[hub]"
            )
        resolved_token = token or self._token
        local_dir = huggingface_hub.snapshot_download(
            repo_id=repo_id,
            revision=revision,
            cache_dir=str(cache_dir) if cache_dir else None,
            token=resolved_token,
        )
        return Path(local_dir)

    def verify_hash(self, local_path: Path, expected_sha256: str) -> bool:
        """Return True when the file's SHA-256 matches *expected_sha256*."""
        if not local_path.exists():
            return False
        h = hashlib.sha256()
        with local_path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                h.update(chunk)
        return h.hexdigest() == expected_sha256

    def model_info(self, repo_id: str, token: str | None = None) -> dict[str, Any]:
        """Fetch lightweight metadata for a Hub repository."""
        if not _HAS_HUB:
            return {"repo_id": repo_id, "stub": True}
        info = huggingface_hub.model_info(
            repo_id, token=token or self._token
        )
        return {
            "repo_id": repo_id,
            "sha": info.sha,
            "tags": list(info.tags or []),
            "downloads": info.downloads,
            "last_modified": str(info.last_modified) if info.last_modified else None,
        }
