from __future__ import annotations

import itertools
import random
from dataclasses import dataclass
from typing import Any, Literal

from provenir.core.abstractions import Backend, RunManifest
from provenir.core.config import RunConfig
from provenir.data.dataset import JsonlDataset
from provenir.train.trainer import Trainer


@dataclass(frozen=True)
class SweepConfig:
    """Describes a hyperparameter search over RunConfig fields.

    *param_grid* maps RunConfig field names to lists of candidate values:

        SweepConfig(
            param_grid={"seed": [0, 1, 2], "backend": ["stub"]},
            strategy="grid",
        )

    *strategy* is ``"grid"`` (exhaustive Cartesian product) or ``"random"``
    (uniform random sampling).  For ``"random"``, *n_trials* is required.
    """

    param_grid: dict[str, list[Any]]
    strategy: Literal["grid", "random"] = "grid"
    n_trials: int | None = None
    seed: int = 0

    def __post_init__(self) -> None:
        if self.strategy == "random" and self.n_trials is None:
            raise ValueError("n_trials is required for random sweep strategy")
        if self.strategy not in {"grid", "random"}:
            raise ValueError(f"unknown strategy {self.strategy!r}")


@dataclass
class TrialResult:
    trial_id: int
    params: dict[str, Any]
    manifest: RunManifest


@dataclass
class SweepResult:
    trials: list[TrialResult]
    sweep_config: SweepConfig

    @property
    def best_trial(self) -> TrialResult | None:
        """Return trial with highest accuracy, if metrics_history carries accuracy."""
        best: TrialResult | None = None
        best_score: float = -1.0
        for trial in self.trials:
            score = float(trial.manifest.provenance.get("accuracy_mean", -1.0))
            if score > best_score:
                best_score = score
                best = trial
        return best


def _param_combinations(param_grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """Cartesian product over all parameter lists."""
    keys = list(param_grid.keys())
    values_lists = [param_grid[k] for k in keys]
    return [dict(zip(keys, combo, strict=True)) for combo in itertools.product(*values_lists)]


def _random_combinations(
    param_grid: dict[str, list[Any]],
    n_trials: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Random sample (with replacement) of *n_trials* parameter combinations."""
    rng = random.Random(seed)
    keys = list(param_grid.keys())
    return [
        {k: rng.choice(param_grid[k]) for k in keys}
        for _ in range(n_trials)
    ]


def _trial_config(base: RunConfig, params: dict[str, Any], trial_id: int) -> RunConfig:
    """Merge *params* into *base* config, appending a trial suffix to output_dir."""
    merged = base.model_dump()
    merged.update(params)
    merged["output_dir"] = f"{merged['output_dir']}/trial_{trial_id}"
    return RunConfig(**merged)


class GridSweep:
    """Exhaustive grid search over *sweep_config.param_grid*."""

    def __init__(
        self,
        base_config: RunConfig,
        sweep_config: SweepConfig,
        backend: Backend,
    ) -> None:
        if sweep_config.strategy != "grid":
            raise ValueError("GridSweep requires strategy='grid'")
        self.base_config = base_config
        self.sweep_config = sweep_config
        self.backend = backend

    def run(self, dataset: JsonlDataset) -> SweepResult:
        combos = _param_combinations(self.sweep_config.param_grid)
        results: list[TrialResult] = []
        for i, params in enumerate(combos):
            config = _trial_config(self.base_config, params, i)
            manifest = Trainer(backend=self.backend, config=config).run(dataset)
            results.append(TrialResult(trial_id=i, params=params, manifest=manifest))
        return SweepResult(trials=results, sweep_config=self.sweep_config)


class RandomSweep:
    """Random search over *sweep_config.param_grid*."""

    def __init__(
        self,
        base_config: RunConfig,
        sweep_config: SweepConfig,
        backend: Backend,
    ) -> None:
        if sweep_config.strategy != "random":
            raise ValueError("RandomSweep requires strategy='random'")
        self.base_config = base_config
        self.sweep_config = sweep_config
        self.backend = backend

    def run(self, dataset: JsonlDataset) -> SweepResult:
        n = self.sweep_config.n_trials or 0
        combos = _random_combinations(
            self.sweep_config.param_grid,
            n,
            self.sweep_config.seed,
        )
        results: list[TrialResult] = []
        for i, params in enumerate(combos):
            config = _trial_config(self.base_config, params, i)
            manifest = Trainer(backend=self.backend, config=config).run(dataset)
            results.append(TrialResult(trial_id=i, params=params, manifest=manifest))
        return SweepResult(trials=results, sweep_config=self.sweep_config)
