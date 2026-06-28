from __future__ import annotations

import pytest

from provenir.core.config import DistributedConfig, RunConfig


class TestDistributedConfig:
    def test_default_strategy(self) -> None:
        assert DistributedConfig().strategy == "none"

    def test_default_num_gpus(self) -> None:
        assert DistributedConfig().num_gpus == 1

    def test_valid_strategy_fsdp(self) -> None:
        cfg = DistributedConfig(strategy="fsdp")
        assert cfg.strategy == "fsdp"

    def test_valid_strategy_deepspeed(self) -> None:
        assert DistributedConfig(strategy="deepspeed").strategy == "deepspeed"

    def test_num_gpus_validation(self) -> None:
        with pytest.raises(Exception):
            DistributedConfig(num_gpus=0)

    def test_deepspeed_stage_in_range(self) -> None:
        cfg = DistributedConfig(deepspeed_stage=3)
        assert cfg.deepspeed_stage == 3


class TestRunConfigExtended:
    def test_defaults_backward_compatible(self) -> None:
        cfg = RunConfig()
        assert cfg.name == "default-run"
        assert cfg.backend == "stub"
        assert cfg.seed == 0

    def test_new_field_model_name_default_none(self) -> None:
        assert RunConfig().model_name_or_path is None

    def test_new_field_max_steps(self) -> None:
        assert RunConfig().max_steps == 10

    def test_new_field_batch_size(self) -> None:
        assert RunConfig().batch_size == 1

    def test_peft_field_default_none(self) -> None:
        assert RunConfig().peft is None

    def test_peft_can_accept_dict(self) -> None:
        cfg = RunConfig(peft={"rank": 8, "alpha": 16})
        assert cfg.peft is not None
        assert cfg.peft["rank"] == 8

    def test_distributed_field_default_none(self) -> None:
        assert RunConfig().distributed is None

    def test_distributed_can_accept_config(self) -> None:
        cfg = RunConfig(distributed=DistributedConfig(strategy="fsdp", num_gpus=4))
        assert cfg.distributed is not None
        assert cfg.distributed.strategy == "fsdp"
        assert cfg.distributed.num_gpus == 4

    def test_observability_backend_default(self) -> None:
        assert RunConfig().observability_backend == "none"

    def test_observability_project_default(self) -> None:
        assert RunConfig().observability_project == "provenir"

    def test_max_steps_ge_one(self) -> None:
        with pytest.raises(Exception):
            RunConfig(max_steps=0)

    def test_batch_size_ge_one(self) -> None:
        with pytest.raises(Exception):
            RunConfig(batch_size=0)

    def test_model_dump_includes_new_fields(self) -> None:
        cfg = RunConfig(model_name_or_path="gpt2", max_steps=5)
        d = cfg.model_dump()
        assert d["model_name_or_path"] == "gpt2"
        assert d["max_steps"] == 5
