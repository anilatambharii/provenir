"""Tests for HubClient.push_with_passport and CLI --passport flag."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from provenir.adapters.hub import HubClient, HubConfig, HubPushResult
from provenir.governance.bom import CodeComponent, DataComponent, EvalComponent, ModelBOM
from provenir.governance.passport import PassportSigner


def _make_passport_json() -> str:
    """Build a minimal signed ModelPassport JSON string."""
    bom = ModelBOM(
        model_id="test-model",
        base_model="base-llm",
        run_id="run-passport-test",
        data=[DataComponent(name="train", content_hash="abc123", num_records=100)],
        code=CodeComponent(git_sha="deadbeef", dependencies_hash="dephash", framework="trl"),
        evals=[EvalComponent(benchmark="mmlu", score=0.72)],
        hyperparameters={"lr": 0.001},
    )
    signer = PassportSigner(b"test-key")
    passport = signer.sign(bom, signed_at="2026-01-01T00:00:00Z")
    return passport.to_json()


def _force_stub_mode() -> None:
    """Force hub.py to use the stub path (no huggingface_hub)."""
    import provenir.adapters.hub as hub_mod

    hub_mod._HAS_HUB = False


class TestPushWithPassportStub:
    def setup_method(self) -> None:
        import provenir.adapters.hub as hub_mod

        self._orig = hub_mod._HAS_HUB
        hub_mod._HAS_HUB = False

    def teardown_method(self) -> None:
        import provenir.adapters.hub as hub_mod

        hub_mod._HAS_HUB = self._orig

    def test_push_with_passport_stub(self, tmp_path: Path) -> None:
        """push_with_passport returns a HubPushResult with the right URL even in stub mode."""
        client = HubClient()
        cfg = HubConfig(repo_id="user/repo")
        passport_json = _make_passport_json()
        result = client.push_with_passport(tmp_path, cfg, passport_json)
        assert isinstance(result, HubPushResult)
        assert "huggingface.co" in result.url
        assert result.repo_id == "user/repo"

    def test_push_with_passport_writes_passport_json(self, tmp_path: Path) -> None:
        """push_with_passport writes provenir_passport.json to adapter dir."""
        client = HubClient()
        cfg = HubConfig(repo_id="user/repo")
        passport_json = _make_passport_json()
        client.push_with_passport(tmp_path, cfg, passport_json)
        passport_file = tmp_path / "provenir_passport.json"
        assert passport_file.exists()
        # Round-trip the JSON
        loaded = json.loads(passport_file.read_text(encoding="utf-8"))
        assert loaded["bom"]["model_id"] == "test-model"

    def test_push_adapter_writes_passport_md(self, tmp_path: Path) -> None:
        """push_with_passport also writes provenir_passport.md."""
        client = HubClient()
        cfg = HubConfig(repo_id="user/mymodel")
        passport_json = _make_passport_json()
        client.push_with_passport(tmp_path, cfg, passport_json)
        md_file = tmp_path / "provenir_passport.md"
        assert md_file.exists()
        content = md_file.read_text(encoding="utf-8")
        # The markdown should mention the model id
        assert "test-model" in content

    def test_push_with_passport_invalid_json_still_writes_json(
        self, tmp_path: Path
    ) -> None:
        """Even with invalid JSON for the md step, passport.json is still written."""
        client = HubClient()
        cfg = HubConfig(repo_id="user/repo")
        invalid_json = '{"bom": {}}'  # missing required fields for from_dict but valid JSON
        # Should not raise — the except clause swallows the md-generation error
        client.push_with_passport(tmp_path, cfg, invalid_json)
        assert (tmp_path / "provenir_passport.json").exists()

    def test_push_with_passport_no_model_card(self, tmp_path: Path) -> None:
        client = HubClient()
        cfg = HubConfig(repo_id="user/repo")
        result = client.push_with_passport(tmp_path, cfg, _make_passport_json())
        assert result.url.startswith("https://huggingface.co/")


class TestHubPushCLIWithPassport:
    def setup_method(self) -> None:
        import provenir.adapters.hub as hub_mod

        self._orig = hub_mod._HAS_HUB
        hub_mod._HAS_HUB = False

    def teardown_method(self) -> None:
        import provenir.adapters.hub as hub_mod

        hub_mod._HAS_HUB = self._orig

    def test_hub_push_cli_with_passport(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """CLI hub push --passport should not raise."""
        from provenir.cli.main import main

        # Create a fake adapter directory and a passport file
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()
        passport_file = tmp_path / "passport.json"
        passport_file.write_text(_make_passport_json(), encoding="utf-8")

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "provenir",
                "hub",
                "push",
                str(adapter_dir),
                "user/repo",
                "--passport",
                str(passport_file),
            ],
        )
        # Should complete without raising
        main()

    def test_hub_push_cli_without_passport(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """CLI hub push without --passport should still work."""
        from provenir.cli.main import main

        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()

        monkeypatch.setattr(
            sys,
            "argv",
            ["provenir", "hub", "push", str(adapter_dir), "user/repo"],
        )
        main()

    def test_hub_push_cli_passport_writes_files(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """After CLI hub push --passport, the passport files exist in the adapter dir."""
        from provenir.cli.main import main

        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()
        passport_file = tmp_path / "passport.json"
        passport_file.write_text(_make_passport_json(), encoding="utf-8")

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "provenir",
                "hub",
                "push",
                str(adapter_dir),
                "user/repo",
                "--passport",
                str(passport_file),
            ],
        )
        main()
        assert (adapter_dir / "provenir_passport.json").exists()
        assert (adapter_dir / "provenir_passport.md").exists()
