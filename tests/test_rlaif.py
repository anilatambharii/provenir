from __future__ import annotations

import pytest

from provenir.core.config import RunConfig
from provenir.data.dataset import JsonlDataset
from provenir.eval.judge import StubJudge
from provenir.train.backends.stub import StubBackend
from provenir.train.rlaif import RLAIFConfig, RLAIFIteration, RLAIFPipeline


def _ds() -> JsonlDataset:
    return JsonlDataset.from_records([
        {"prompt": "What is 1+1?", "response": "Two"},
        {"prompt": "Name a colour.", "response": "Blue"},
        {"prompt": "Capital of France?", "response": "Paris"},
    ])


def _pipeline(**kwargs: object) -> RLAIFPipeline:
    return RLAIFPipeline(
        judge=StubJudge(),
        backend=StubBackend(),
        base_config=RunConfig(),
        rlaif_config=RLAIFConfig(**kwargs),  # type: ignore[arg-type]
    )


class TestRLAIFConfig:
    def test_default_iterations(self) -> None:
        assert RLAIFConfig().n_iterations == 3

    def test_zero_iterations_raises(self) -> None:
        with pytest.raises(ValueError, match="n_iterations"):
            RLAIFConfig(n_iterations=0)

    def test_one_response_raises(self) -> None:
        with pytest.raises(ValueError, match="responses_per_prompt"):
            RLAIFConfig(responses_per_prompt=1)

    def test_is_frozen(self) -> None:
        cfg = RLAIFConfig()
        with pytest.raises((AttributeError, TypeError)):
            cfg.n_iterations = 5  # type: ignore[misc]


class TestRLAIFPipelineGeneratePreferences:
    def test_returns_preferences_for_each_record(self) -> None:
        pipeline = _pipeline(n_iterations=1)
        prefs = pipeline.generate_preferences(_ds())
        # StubJudge never ties on the two synthetic candidates (different length)
        assert len(prefs) > 0

    def test_preferences_have_required_fields(self) -> None:
        pipeline = _pipeline(n_iterations=1)
        prefs = pipeline.generate_preferences(_ds())
        for p in prefs:
            assert "prompt" in p
            assert "chosen" in p
            assert "rejected" in p


class TestRLAIFPipelineRun:
    def test_run_returns_iterations(self) -> None:
        pipeline = _pipeline(n_iterations=2)
        iterations = pipeline.run(_ds())
        assert len(iterations) == 2

    def test_single_iteration(self) -> None:
        pipeline = _pipeline(n_iterations=1)
        iterations = pipeline.run(_ds())
        assert len(iterations) == 1
        assert iterations[0].iteration == 0

    def test_each_iteration_has_manifest(self) -> None:
        pipeline = _pipeline(n_iterations=2)
        iterations = pipeline.run(_ds())
        for it in iterations:
            assert it.manifest.run_id != ""

    def test_iteration_type(self) -> None:
        pipeline = _pipeline(n_iterations=1)
        it = pipeline.run(_ds())[0]
        assert isinstance(it, RLAIFIteration)

    def test_with_eval_dataset(self) -> None:
        pipeline = _pipeline(n_iterations=2, eval_after_each_iteration=True)
        iterations = pipeline.run(_ds(), eval_dataset=_ds())
        assert len(iterations) == 2
        for it in iterations:
            assert it.eval_result is not None

    def test_without_eval_no_eval_result(self) -> None:
        pipeline = _pipeline(n_iterations=1, eval_after_each_iteration=False)
        iterations = pipeline.run(_ds())
        assert iterations[0].eval_result is None

    def test_manifests_have_rlaif_provenance(self) -> None:
        pipeline = _pipeline(n_iterations=1)
        it = pipeline.run(_ds())[0]
        assert it.manifest.provenance.get("pipeline") == "rlaif"

    def test_different_seed_per_iteration(self) -> None:
        pipeline = _pipeline(n_iterations=3)
        iterations = pipeline.run(_ds())
        seeds = [it.manifest.seed for it in iterations]
        assert len(set(seeds)) == 3
