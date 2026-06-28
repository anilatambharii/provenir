from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

try:
    import torch

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

try:
    from safetensors.torch import load_file, save_file

    _HAS_SAFETENSORS = True
except ImportError:
    _HAS_SAFETENSORS = False


@dataclass(frozen=True)
class MergeConfig:
    """Configuration for merging two or more LoRA adapters.

    *strategy*:

    * ``"slerp"`` — spherical linear interpolation; blends weight *direction*
      rather than magnitude.  Best for two adapters trained on similar tasks.
    * ``"ties"`` — magnitude-based trimming then sign-election merge;
      better than averaging for dissimilar tasks.
    * ``"dare"`` — random drop then rescale (DARE-TIES); reduces interference
      between many adapters without collapsing magnitudes.
    """

    strategy: Literal["slerp", "ties", "dare"] = "slerp"
    weights: tuple[float, ...] | None = None  # per-adapter interpolation weights
    density: float = 0.5  # fraction of weights to ZERO OUT in TIES / DARE
    scale: float = 1.0  # rescale factor for DARE survivors

    def __post_init__(self) -> None:
        if not (0.0 <= self.density <= 1.0):
            raise ValueError(f"density must be in [0, 1], got {self.density}")
        if self.scale <= 0.0:
            raise ValueError(f"scale must be > 0, got {self.scale}")


@dataclass
class MergeResult:
    """Output descriptor for a completed merge operation."""

    output_path: Path
    strategy: str
    adapter_paths: list[Path]
    config: MergeConfig
    metadata: dict[str, Any] = field(default_factory=dict)


class ModelMerger:
    """Merge LoRA adapter weight tensors using SLERP, TIES, or DARE.

    Requires PyTorch + safetensors: ``pip install provenir[train]``.

    When the packages are unavailable the merge degrades gracefully: the first
    adapter is copied to *output_dir* unchanged, and the result carries
    ``metadata={stub: True}``.
    """

    def merge(
        self,
        adapter_paths: list[Path],
        config: MergeConfig,
        output_dir: Path,
    ) -> MergeResult:
        if len(adapter_paths) < 2:
            raise ValueError(
                f"merge requires at least 2 adapters, got {len(adapter_paths)}"
            )
        output_dir.mkdir(parents=True, exist_ok=True)

        if not (_HAS_TORCH and _HAS_SAFETENSORS):
            out_file = output_dir / "adapter_model.safetensors"
            src = adapter_paths[0] / "adapter_model.safetensors"
            if src.exists():
                import shutil

                shutil.copy2(src, out_file)
            else:
                out_file.write_bytes(b"")
            return MergeResult(
                output_path=output_dir,
                strategy=config.strategy,
                adapter_paths=list(adapter_paths),
                config=config,
                metadata={"stub": True, "reason": "torch or safetensors not installed"},
            )

        dispatch = {"slerp": self._slerp, "ties": self._ties, "dare": self._dare}
        merged = dispatch[config.strategy](adapter_paths, config)

        out_file = output_dir / "adapter_model.safetensors"
        save_file(merged, str(out_file))

        return MergeResult(
            output_path=output_dir,
            strategy=config.strategy,
            adapter_paths=list(adapter_paths),
            config=config,
            metadata={"num_adapters": len(adapter_paths), "num_tensors": len(merged)},
        )

    def _load(self, path: Path) -> dict[str, Any]:
        af = path / "adapter_model.safetensors"
        if not af.exists():
            raise FileNotFoundError(f"no adapter_model.safetensors in {path}")
        return load_file(str(af))

    def _slerp(self, paths: list[Path], config: MergeConfig) -> dict[str, Any]:
        tensors = [self._load(p) for p in paths]
        n = len(tensors)
        weights = list(config.weights) if config.weights else [1.0 / n] * n
        total = sum(weights)
        weights = [w / total for w in weights]

        merged: dict[str, Any] = {}
        for key in tensors[0]:
            acc = tensors[0][key].float() * weights[0]
            for i in range(1, n):
                v = tensors[i][key].float()
                dot = (acc * v).sum().clamp(-1.0, 1.0)
                theta = torch.acos(dot.abs())
                if theta.abs() < 1e-6:
                    acc = acc + v * weights[i]
                else:
                    t = weights[i]
                    acc = (
                        torch.sin((1 - t) * theta) * acc + torch.sin(t * theta) * v
                    ) / torch.sin(theta)
            merged[key] = acc.to(tensors[0][key].dtype)
        return merged

    def _ties(self, paths: list[Path], config: MergeConfig) -> dict[str, Any]:
        tensors = [self._load(p) for p in paths]
        merged: dict[str, Any] = {}
        for key in tensors[0]:
            stacked = torch.stack([t[key].float() for t in tensors], dim=0)
            threshold = torch.quantile(stacked.abs(), config.density)
            trimmed = stacked * (stacked.abs() >= threshold).float()
            signs = trimmed.sum(dim=0).sign()
            mask = (trimmed.sign() == signs.unsqueeze(0)).float()
            result = (trimmed * mask).sum(dim=0)
            count = mask.sum(dim=0).clamp(min=1.0)
            merged[key] = (result / count).to(tensors[0][key].dtype)
        return merged

    def _dare(self, paths: list[Path], config: MergeConfig) -> dict[str, Any]:
        tensors = [self._load(p) for p in paths]
        rng = torch.Generator()
        rng.manual_seed(42)
        merged: dict[str, Any] = {}
        for key in tensors[0]:
            stacked = torch.stack([t[key].float() for t in tensors], dim=0)
            drop_mask = torch.bernoulli(
                torch.full(stacked.shape, 1.0 - config.density), generator=rng
            )
            dare = stacked * drop_mask * config.scale
            merged[key] = dare.mean(dim=0).to(tensors[0][key].dtype)
        return merged
