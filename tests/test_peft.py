from __future__ import annotations

import pytest

from provenir.train.peft import PEFTConfig


class TestPEFTConfigDefaults:
    def test_default_rank(self) -> None:
        assert PEFTConfig().rank == 16

    def test_default_alpha(self) -> None:
        assert PEFTConfig().alpha == 32

    def test_default_target_modules(self) -> None:
        assert PEFTConfig().target_modules == ("q_proj", "v_proj")

    def test_default_dropout(self) -> None:
        assert PEFTConfig().dropout == 0.05

    def test_default_not_quantised(self) -> None:
        cfg = PEFTConfig()
        assert not cfg.load_in_4bit
        assert not cfg.load_in_8bit

    def test_default_lora_bias(self) -> None:
        assert PEFTConfig().lora_bias == "none"

    def test_is_frozen(self) -> None:
        cfg = PEFTConfig()
        with pytest.raises((AttributeError, TypeError)):
            cfg.rank = 8  # type: ignore[misc]


class TestPEFTConfigValidation:
    def test_rank_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="rank must be >= 1"):
            PEFTConfig(rank=0)

    def test_alpha_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="alpha must be > 0"):
            PEFTConfig(alpha=0)

    def test_negative_alpha_raises(self) -> None:
        with pytest.raises(ValueError, match="alpha must be > 0"):
            PEFTConfig(alpha=-1)

    def test_dropout_one_raises(self) -> None:
        with pytest.raises(ValueError, match="dropout must be in"):
            PEFTConfig(dropout=1.0)

    def test_dropout_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="dropout must be in"):
            PEFTConfig(dropout=-0.1)

    def test_mutual_exclusion_4bit_8bit(self) -> None:
        with pytest.raises(ValueError, match="mutually exclusive"):
            PEFTConfig(load_in_4bit=True, load_in_8bit=True)

    def test_invalid_lora_bias_raises(self) -> None:
        with pytest.raises(ValueError, match="lora_bias"):
            PEFTConfig(lora_bias="invalid")

    def test_valid_lora_biases(self) -> None:
        for bias in ("none", "all", "lora_only"):
            cfg = PEFTConfig(lora_bias=bias)
            assert cfg.lora_bias == bias


class TestPEFTScaling:
    def test_standard_scaling(self) -> None:
        cfg = PEFTConfig(rank=16, alpha=32)
        assert cfg.scaling == pytest.approx(2.0)

    def test_rslora_scaling(self) -> None:
        cfg = PEFTConfig(rank=16, alpha=32, use_rslora=True)
        import math

        expected = 32.0 / math.sqrt(16)
        assert cfg.scaling == pytest.approx(expected)

    def test_rank_equals_alpha_scaling_one(self) -> None:
        cfg = PEFTConfig(rank=8, alpha=8)
        assert cfg.scaling == pytest.approx(1.0)


class TestPEFTToDict:
    def test_to_dict_contains_rank(self) -> None:
        d = PEFTConfig(rank=8).to_dict()
        assert d["rank"] == 8

    def test_to_dict_target_modules_is_list(self) -> None:
        d = PEFTConfig().to_dict()
        assert isinstance(d["target_modules"], list)

    def test_to_dict_scaling_present(self) -> None:
        d = PEFTConfig(rank=4, alpha=8).to_dict()
        assert d["scaling"] == pytest.approx(2.0)

    def test_to_dict_roundtrip_fields(self) -> None:
        cfg = PEFTConfig(rank=4, alpha=8, dropout=0.1, load_in_4bit=True)
        d = cfg.to_dict()
        assert d["rank"] == 4
        assert d["alpha"] == 8
        assert d["dropout"] == 0.1
        assert d["load_in_4bit"] is True
