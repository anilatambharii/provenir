from __future__ import annotations

from pathlib import Path

from provenir.core.abstractions import Backend, RunManifest
from provenir.train.backends.trl import TRLBackend


class TestTRLBackendInterface:
    backend = TRLBackend()

    def test_name_is_trl(self) -> None:
        assert self.backend.name == "trl"

    def test_implements_backend_protocol(self) -> None:
        assert isinstance(self.backend, Backend)

    def test_capabilities_returns_mapping(self) -> None:
        caps = self.backend.capabilities()
        assert isinstance(dict(caps), dict)

    def test_capabilities_has_algorithms(self) -> None:
        caps = self.backend.capabilities()
        assert "algorithms" in caps

    def test_capabilities_includes_sft(self) -> None:
        assert "sft" in self.backend.capabilities()["algorithms"]

    def test_capabilities_includes_dpo(self) -> None:
        assert "dpo" in self.backend.capabilities()["algorithms"]

    def test_capabilities_has_available_flag(self) -> None:
        assert "available" in self.backend.capabilities()

    def test_capabilities_not_available_without_trl(self) -> None:
        import provenir.train.backends.trl as trl_mod

        original_trl = trl_mod._HAS_TRL
        original_torch = trl_mod._HAS_TORCH
        trl_mod._HAS_TRL = False
        trl_mod._HAS_TORCH = False
        try:
            caps = self.backend.capabilities()
            assert not caps["available"]
        finally:
            trl_mod._HAS_TRL = original_trl
            trl_mod._HAS_TORCH = original_torch

    def test_fit_returns_manifest_when_unavailable(self) -> None:
        import provenir.train.backends.trl as trl_mod

        original = trl_mod._HAS_TRL
        trl_mod._HAS_TRL = False
        try:
            manifest = RunManifest(config_hash="abc", dataset_hash="def", seed=0)
            result = self.backend.fit({}, manifest)
            assert result.run_id == manifest.run_id
        finally:
            trl_mod._HAS_TRL = original

    def test_save_adapter_noop(self, tmp_path: "Path") -> None:  # type: ignore[name-defined]

        self.backend.save_adapter(tmp_path, {})  # Should not raise


class TestTRLBackendPrepare:
    def test_prepare_raises_import_error_when_unavailable(self) -> None:
        import pytest

        import provenir.train.backends.trl as trl_mod

        original = trl_mod._HAS_TRL
        trl_mod._HAS_TRL = False
        try:
            with pytest.raises(ImportError, match="TRL backend"):
                TRLBackend().prepare({})
        finally:
            trl_mod._HAS_TRL = original
