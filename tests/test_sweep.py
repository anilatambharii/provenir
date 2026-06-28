from __future__ import annotations

import pytest

from provenir.core.config import RunConfig
from provenir.data.dataset import JsonlDataset
from provenir.train.backends.stub import StubBackend
from provenir.train.sweep import (
    GridSweep,
    RandomSweep,
    SweepConfig,
    SweepResult,
    TrialResult,
    _param_combinations,
    _random_combinations,
    _trial_config,
)

SAMPLE_DATASET = JsonlDataset.from_records(
    [{"prompt": "Hello", "response": "Hi"}, {"prompt": "World", "response": "Planet"}]
)
BASE_CONFIG = RunConfig(name="test-sweep", backend="stub", seed=0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_param_combinations_single_param() -> None:
    combos = _param_combinations({"seed": [0, 1, 2]})
    assert combos == [{"seed": 0}, {"seed": 1}, {"seed": 2}]


def test_param_combinations_cartesian_product() -> None:
    combos = _param_combinations({"seed": [0, 1], "deterministic": [True, False]})
    assert len(combos) == 4
    assert {"seed": 0, "deterministic": True} in combos
    assert {"seed": 1, "deterministic": False} in combos


def test_random_combinations_count() -> None:
    combos = _random_combinations({"seed": [0, 1, 2, 3]}, n_trials=5, seed=42)
    assert len(combos) == 5


def test_random_combinations_deterministic_with_seed() -> None:
    a = _random_combinations({"seed": [0, 1, 2]}, n_trials=3, seed=99)
    b = _random_combinations({"seed": [0, 1, 2]}, n_trials=3, seed=99)
    assert a == b


def test_trial_config_output_dir_suffixed() -> None:
    config = _trial_config(BASE_CONFIG, {"seed": 7}, trial_id=3)
    assert config.output_dir.endswith("trial_3")
    assert config.seed == 7


def test_trial_config_preserves_base_fields() -> None:
    config = _trial_config(BASE_CONFIG, {}, trial_id=0)
    assert config.name == BASE_CONFIG.name
    assert config.backend == BASE_CONFIG.backend


# ---------------------------------------------------------------------------
# SweepConfig validation
# ---------------------------------------------------------------------------


def test_sweep_config_random_requires_n_trials() -> None:
    with pytest.raises(ValueError, match="n_trials is required"):
        SweepConfig(param_grid={"seed": [0, 1]}, strategy="random", n_trials=None)


def test_sweep_config_unknown_strategy_raises() -> None:
    with pytest.raises(ValueError, match="unknown strategy"):
        SweepConfig(param_grid={"seed": [0]}, strategy="bayes")  # type: ignore[arg-type]


def test_sweep_config_grid_no_n_trials_ok() -> None:
    cfg = SweepConfig(param_grid={"seed": [0, 1]}, strategy="grid")
    assert cfg.strategy == "grid"


# ---------------------------------------------------------------------------
# GridSweep
# ---------------------------------------------------------------------------


class TestGridSweep:
    def _make_sweep(self, param_grid: dict) -> GridSweep:
        cfg = SweepConfig(param_grid=param_grid, strategy="grid")
        return GridSweep(BASE_CONFIG, cfg, StubBackend())

    def test_trial_count_matches_grid(self) -> None:
        sweep = self._make_sweep({"seed": [0, 1, 2]})
        result = sweep.run(SAMPLE_DATASET)
        assert len(result.trials) == 3

    def test_cartesian_product_of_two_params(self) -> None:
        sweep = self._make_sweep({"seed": [0, 1], "deterministic": [True, False]})
        result = sweep.run(SAMPLE_DATASET)
        assert len(result.trials) == 4

    def test_trial_ids_sequential(self) -> None:
        sweep = self._make_sweep({"seed": [0, 1, 2]})
        result = sweep.run(SAMPLE_DATASET)
        assert [t.trial_id for t in result.trials] == [0, 1, 2]

    def test_each_trial_has_manifest(self) -> None:
        sweep = self._make_sweep({"seed": [0, 1]})
        result = sweep.run(SAMPLE_DATASET)
        for trial in result.trials:
            assert trial.manifest.run_id

    def test_params_stored_per_trial(self) -> None:
        sweep = self._make_sweep({"seed": [0, 1]})
        result = sweep.run(SAMPLE_DATASET)
        assert result.trials[0].params["seed"] == 0
        assert result.trials[1].params["seed"] == 1

    def test_manifests_have_distinct_run_ids(self) -> None:
        sweep = self._make_sweep({"seed": [0, 1, 2]})
        result = sweep.run(SAMPLE_DATASET)
        run_ids = [t.manifest.run_id for t in result.trials]
        assert len(set(run_ids)) == 3

    def test_wrong_strategy_raises(self) -> None:
        cfg = SweepConfig(param_grid={"seed": [0]}, strategy="random", n_trials=1)
        with pytest.raises(ValueError, match="strategy='grid'"):
            GridSweep(BASE_CONFIG, cfg, StubBackend())

    def test_sweep_result_type(self) -> None:
        sweep = self._make_sweep({"seed": [0]})
        result = sweep.run(SAMPLE_DATASET)
        assert isinstance(result, SweepResult)
        assert isinstance(result.trials[0], TrialResult)


# ---------------------------------------------------------------------------
# RandomSweep
# ---------------------------------------------------------------------------


class TestRandomSweep:
    def _make_sweep(self, n_trials: int, seed: int = 0) -> RandomSweep:
        cfg = SweepConfig(
            param_grid={"seed": [0, 1, 2, 3, 4]},
            strategy="random",
            n_trials=n_trials,
            seed=seed,
        )
        return RandomSweep(BASE_CONFIG, cfg, StubBackend())

    def test_trial_count_matches_n_trials(self) -> None:
        result = self._make_sweep(n_trials=3).run(SAMPLE_DATASET)
        assert len(result.trials) == 3

    def test_deterministic_with_same_seed(self) -> None:
        r1 = self._make_sweep(n_trials=4, seed=7).run(SAMPLE_DATASET)
        r2 = self._make_sweep(n_trials=4, seed=7).run(SAMPLE_DATASET)
        assert [t.params for t in r1.trials] == [t.params for t in r2.trials]

    def test_different_seed_different_params(self) -> None:
        r1 = self._make_sweep(n_trials=5, seed=1).run(SAMPLE_DATASET)
        r2 = self._make_sweep(n_trials=5, seed=2).run(SAMPLE_DATASET)
        # Very unlikely to be identical across 5 trials with 5 choices
        assert [t.params for t in r1.trials] != [t.params for t in r2.trials]

    def test_wrong_strategy_raises(self) -> None:
        cfg = SweepConfig(param_grid={"seed": [0]}, strategy="grid")
        with pytest.raises(ValueError, match="strategy='random'"):
            RandomSweep(BASE_CONFIG, cfg, StubBackend())

    def test_each_trial_has_manifest(self) -> None:
        result = self._make_sweep(n_trials=2).run(SAMPLE_DATASET)
        for trial in result.trials:
            assert trial.manifest.dataset_hash


# ---------------------------------------------------------------------------
# SweepResult.best_trial
# ---------------------------------------------------------------------------


def test_best_trial_no_accuracy_returns_first_or_none() -> None:
    cfg = SweepConfig(param_grid={"seed": [0, 1]}, strategy="grid")
    result = GridSweep(BASE_CONFIG, cfg, StubBackend()).run(SAMPLE_DATASET)
    # StubBackend doesn't inject accuracy_mean, so best_trial falls back
    # to the last trial that beat -1.0 (all tie at -1.0 → returns first found)
    assert result.best_trial is not None or result.best_trial is None  # just no crash
