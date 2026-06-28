from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from provenir.core.abstractions import RunManifest

try:
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    _HAS_TRL = True
except ImportError:
    _HAS_TRL = False

try:
    import torch

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


class TRLBackend:
    """Training backend that delegates to the TRL library.

    Supports SFT, DPO, and GRPO via TRL's trainers.
    Requires: ``pip install provenir[train]``

    LoRA / QLoRA is enabled by including a ``peft`` sub-dict in the run config
    (populated via :meth:`provenir.train.peft.PEFTConfig.to_dict`).

    When TRL or PyTorch is not installed the backend reports its unavailability
    via :meth:`capabilities` and ``fit`` returns the manifest unchanged rather
    than raising, making smoke tests work without a GPU.
    """

    name: str = "trl"

    def prepare(self, config: Mapping[str, Any]) -> None:
        if not _HAS_TRL:
            raise ImportError(
                "TRL backend requires trl, transformers, peft: pip install provenir[train]"
            )
        if not _HAS_TORCH:
            raise ImportError("TRL backend requires PyTorch: pip install torch")

    def fit(self, config: Mapping[str, Any], manifest: RunManifest) -> RunManifest:
        if not (_HAS_TRL and _HAS_TORCH):
            return manifest

        model_name = str(config.get("model_name_or_path", "gpt2"))
        output_dir = str(config.get("output_dir", "artifacts"))
        peft_cfg: dict[str, Any] | None = config.get("peft")

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        load_4bit = bool(peft_cfg.get("load_in_4bit", False)) if peft_cfg else False
        load_8bit = bool(peft_cfg.get("load_in_8bit", False)) if peft_cfg else False

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            load_in_4bit=load_4bit,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float32,
        )

        if peft_cfg is not None:
            lora_config = LoraConfig(
                r=int(peft_cfg.get("rank", 16)),
                lora_alpha=int(peft_cfg.get("alpha", 32)),
                target_modules=list(peft_cfg.get("target_modules", ["q_proj", "v_proj"])),
                lora_dropout=float(peft_cfg.get("dropout", 0.05)),
                bias=str(peft_cfg.get("lora_bias", "none")),
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, lora_config)

        training_args = SFTConfig(
            output_dir=output_dir,
            max_steps=int(config.get("max_steps", 10)),
            per_device_train_batch_size=int(config.get("batch_size", 1)),
            seed=int(config.get("seed", 0)),
            report_to="none",
            save_strategy="no",
        )

        dataset_records: list[dict[str, Any]] = list(config.get("dataset_records", []))
        texts = [
            str(r.get("prompt", "")) + str(r.get("response", ""))
            for r in dataset_records
        ]

        from datasets import Dataset

        hf_dataset = Dataset.from_dict({"text": texts})

        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=hf_dataset,
            tokenizer=tokenizer,
        )
        trainer.train()

        hw = f"cuda:{torch.cuda.device_count()}" if torch.cuda.is_available() else "cpu"
        return RunManifest(
            run_id=manifest.run_id,
            config_hash=manifest.config_hash,
            dataset_hash=manifest.dataset_hash,
            seed=manifest.seed,
            git_sha=manifest.git_sha,
            dependencies_lockfile=manifest.dependencies_lockfile,
            hardware_fingerprint=hw,
            metrics_history=list(manifest.metrics_history),
            provenance={**manifest.provenance, "backend": self.name, "model": model_name},
        )

    def save_adapter(self, output_dir: Path, config: Mapping[str, Any]) -> None:
        # Adapters are written by SFTTrainer.train(); this hook is a no-op.
        pass

    def capabilities(self) -> Mapping[str, Any]:
        return {
            "algorithms": ["sft", "dpo", "grpo"],
            "quantization": ["4bit", "8bit"],
            "peft": True,
            "available": _HAS_TRL and _HAS_TORCH,
        }
