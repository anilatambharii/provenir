from __future__ import annotations

from pathlib import Path

import pytest

from provenir.adapters.merging import MergeConfig, MergeResult, ModelMerger


class TestMergeConfig:
    def test_default_strategy(self) -> None:
        assert MergeConfig().strategy == "slerp"

    def test_default_density(self) -> None:
        assert MergeConfig().density == 0.5

    def test_density_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError, match="density"):
            MergeConfig(density=1.5)

    def test_negative_density_raises(self) -> None:
        with pytest.raises(ValueError, match="density"):
            MergeConfig(density=-0.1)

    def test_scale_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="scale"):
            MergeConfig(scale=0.0)

    def test_negative_scale_raises(self) -> None:
        with pytest.raises(ValueError, match="scale"):
            MergeConfig(scale=-1.0)

    def test_valid_strategies(self) -> None:
        for s in ("slerp", "ties", "dare"):
            cfg = MergeConfig(strategy=s)  # type: ignore[arg-type]
            assert cfg.strategy == s

    def test_is_frozen(self) -> None:
        cfg = MergeConfig()
        with pytest.raises((AttributeError, TypeError)):
            cfg.strategy = "ties"  # type: ignore[misc]


class TestModelMergerStub:
    """Merging without torch/safetensors installed produces a stub result."""

    def setup_method(self) -> None:
        import provenir.adapters.merging as merge_mod

        self._orig_torch = merge_mod._HAS_TORCH
        self._orig_st = merge_mod._HAS_SAFETENSORS
        merge_mod._HAS_TORCH = False
        merge_mod._HAS_SAFETENSORS = False

    def teardown_method(self) -> None:
        import provenir.adapters.merging as merge_mod

        merge_mod._HAS_TORCH = self._orig_torch
        merge_mod._HAS_SAFETENSORS = self._orig_st

    def test_single_adapter_raises(self, tmp_path: "Path") -> None:  # type: ignore[name-defined]

        merger = ModelMerger()
        with pytest.raises(ValueError, match="at least 2"):
            merger.merge([tmp_path], MergeConfig(), tmp_path / "out")

    def test_stub_returns_merge_result(self, tmp_path: "Path") -> None:  # type: ignore[name-defined]

        merger = ModelMerger()
        a, b = tmp_path / "a", tmp_path / "b"
        a.mkdir()
        b.mkdir()
        result = merger.merge([a, b], MergeConfig(), tmp_path / "out")
        assert isinstance(result, MergeResult)

    def test_stub_output_path_correct(self, tmp_path: "Path") -> None:  # type: ignore[name-defined]

        merger = ModelMerger()
        a, b = tmp_path / "a", tmp_path / "b"
        a.mkdir()
        b.mkdir()
        out = tmp_path / "merged"
        result = merger.merge([a, b], MergeConfig(), out)
        assert result.output_path == out

    def test_stub_strategy_preserved(self, tmp_path: "Path") -> None:  # type: ignore[name-defined]

        merger = ModelMerger()
        a, b = tmp_path / "a", tmp_path / "b"
        a.mkdir()
        b.mkdir()
        cfg = MergeConfig(strategy="ties")
        result = merger.merge([a, b], cfg, tmp_path / "out")
        assert result.strategy == "ties"

    def test_stub_adapter_paths_preserved(self, tmp_path: "Path") -> None:  # type: ignore[name-defined]

        merger = ModelMerger()
        a, b = tmp_path / "a", tmp_path / "b"
        a.mkdir()
        b.mkdir()
        result = merger.merge([a, b], MergeConfig(), tmp_path / "out")
        assert len(result.adapter_paths) == 2

    def test_output_dir_created(self, tmp_path: "Path") -> None:  # type: ignore[name-defined]

        merger = ModelMerger()
        a, b = tmp_path / "a", tmp_path / "b"
        a.mkdir()
        b.mkdir()
        out = tmp_path / "new_dir" / "merged"
        merger.merge([a, b], MergeConfig(), out)
        assert out.exists()
