from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

import pytest

from provenir.adapters.hub import HubClient, HubConfig, HubPushResult


class TestHubConfig:
    def test_valid_repo_id(self) -> None:
        cfg = HubConfig(repo_id="alice/my-model")
        assert cfg.repo_id == "alice/my-model"

    def test_missing_slash_raises(self) -> None:
        with pytest.raises(ValueError, match="username/repo"):
            HubConfig(repo_id="noSlashHere")

    def test_default_not_private(self) -> None:
        assert not HubConfig(repo_id="a/b").private

    def test_default_revision(self) -> None:
        assert HubConfig(repo_id="a/b").revision == "main"

    def test_is_frozen(self) -> None:
        cfg = HubConfig(repo_id="a/b")
        with pytest.raises((AttributeError, TypeError)):
            cfg.repo_id = "c/d"  # type: ignore[misc]


class TestHubPushResult:
    def test_url_field(self) -> None:
        r = HubPushResult(repo_id="a/b", url="https://huggingface.co/a/b")
        assert r.url.startswith("https://")

    def test_optional_commit_sha(self) -> None:
        r = HubPushResult(repo_id="a/b", url="u")
        assert r.commit_sha is None


class TestHubClientStub:
    """Tests run without the huggingface_hub package."""

    def setup_method(self) -> None:
        import provenir.adapters.hub as hub_mod

        self._orig_has_hub = hub_mod._HAS_HUB
        hub_mod._HAS_HUB = False

    def teardown_method(self) -> None:
        import provenir.adapters.hub as hub_mod

        hub_mod._HAS_HUB = self._orig_has_hub

    def test_push_returns_result_without_hub(self, tmp_path: Path) -> None:
        client = HubClient()
        cfg = HubConfig(repo_id="alice/test-model")
        result = client.push_adapter(tmp_path, cfg)
        assert isinstance(result, HubPushResult)
        assert result.repo_id == "alice/test-model"

    def test_push_url_contains_repo_id(self, tmp_path: Path) -> None:
        client = HubClient()
        cfg = HubConfig(repo_id="alice/test-model")
        result = client.push_adapter(tmp_path, cfg)
        assert "alice/test-model" in result.url

    def test_pull_raises_import_error_without_hub(self) -> None:
        import provenir.adapters.hub as hub_mod

        original = hub_mod._HAS_HUB
        hub_mod._HAS_HUB = False
        try:
            with pytest.raises(ImportError, match="huggingface_hub"):
                HubClient().pull_model("alice/model")
        finally:
            hub_mod._HAS_HUB = original

    def test_model_info_stub_when_no_hub(self) -> None:
        import provenir.adapters.hub as hub_mod

        original = hub_mod._HAS_HUB
        hub_mod._HAS_HUB = False
        try:
            info = HubClient().model_info("alice/model")
            assert info["stub"] is True
        finally:
            hub_mod._HAS_HUB = original


class TestHubClientVerifyHash:
    def test_correct_hash_returns_true(self) -> None:
        with tempfile.NamedTemporaryFile(delete=False) as f:
            content = b"provenir test content"
            f.write(content)
            fpath = Path(f.name)
        try:
            expected = hashlib.sha256(content).hexdigest()
            assert HubClient().verify_hash(fpath, expected) is True
        finally:
            fpath.unlink()

    def test_wrong_hash_returns_false(self) -> None:
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"hello")
            fpath = Path(f.name)
        try:
            assert HubClient().verify_hash(fpath, "deadbeef") is False
        finally:
            fpath.unlink()

    def test_missing_file_returns_false(self, tmp_path: Path) -> None:
        assert HubClient().verify_hash(tmp_path / "ghost.bin", "abc123") is False
