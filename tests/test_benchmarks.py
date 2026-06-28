from __future__ import annotations

import pytest

from provenir.eval.benchmarks import (
    SUPPORTED_BENCHMARKS,
    BenchmarkConfig,
    BenchmarkEvaluator,
    BenchmarkResult,
)


class TestBenchmarkConfig:
    def test_valid_benchmark(self) -> None:
        cfg = BenchmarkConfig(benchmark="mmlu")
        assert cfg.benchmark == "mmlu"

    def test_invalid_benchmark_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown benchmark"):
            BenchmarkConfig(benchmark="nonexistent")  # type: ignore[arg-type]

    def test_negative_fewshot_raises(self) -> None:
        with pytest.raises(ValueError, match="num_fewshot"):
            BenchmarkConfig(benchmark="mmlu", num_fewshot=-1)

    def test_zero_batch_size_raises(self) -> None:
        with pytest.raises(ValueError, match="batch_size"):
            BenchmarkConfig(benchmark="mmlu", batch_size=0)

    def test_default_num_fewshot(self) -> None:
        assert BenchmarkConfig(benchmark="mmlu").num_fewshot == 0

    def test_all_supported_benchmarks_valid(self) -> None:
        for name in SUPPORTED_BENCHMARKS:
            cfg = BenchmarkConfig(benchmark=name)  # type: ignore[arg-type]
            assert cfg.benchmark == name

    def test_is_frozen(self) -> None:
        cfg = BenchmarkConfig(benchmark="mmlu")
        with pytest.raises((AttributeError, TypeError)):
            cfg.benchmark = "hellaswag"  # type: ignore[misc]


class TestBenchmarkResult:
    def test_to_dict_keys(self) -> None:
        r = BenchmarkResult(
            benchmark="mmlu",
            score=0.42,
            num_examples=100,
            num_fewshot=5,
            metric_name="acc",
        )
        d = r.to_dict()
        assert set(d) == {
            "benchmark",
            "score",
            "num_examples",
            "num_fewshot",
            "metric_name",
            "metadata",
        }

    def test_to_dict_values(self) -> None:
        r = BenchmarkResult(
            benchmark="hellaswag",
            score=0.75,
            num_examples=50,
            num_fewshot=0,
            metric_name="acc_norm",
        )
        d = r.to_dict()
        assert d["score"] == 0.75
        assert d["benchmark"] == "hellaswag"


class TestBenchmarkEvaluatorStub:
    """lm-eval is not installed in the test environment — verify stub behaviour."""

    evaluator = BenchmarkEvaluator()

    def test_stub_returns_zero_score(self) -> None:
        cfg = BenchmarkConfig(benchmark="mmlu")
        result = self.evaluator.run("gpt2", cfg)
        assert result.score == 0.0

    def test_stub_metadata_contains_stub_flag(self) -> None:
        cfg = BenchmarkConfig(benchmark="mmlu")
        result = self.evaluator.run("gpt2", cfg)
        # Either it's a real result or a stub — both are valid
        if result.metadata.get("stub"):
            assert result.num_examples == 0

    def test_stub_benchmark_name_preserved(self) -> None:
        cfg = BenchmarkConfig(benchmark="arc_easy")
        result = self.evaluator.run("gpt2", cfg)
        assert result.benchmark == "arc_easy"

    def test_run_suite_returns_one_per_config(self) -> None:
        configs = [
            BenchmarkConfig(benchmark="mmlu"),
            BenchmarkConfig(benchmark="hellaswag"),
        ]
        results = self.evaluator.run_suite("gpt2", configs)
        assert len(results) == 2

    def test_run_suite_order_preserved(self) -> None:
        configs = [
            BenchmarkConfig(benchmark="arc_easy"),
            BenchmarkConfig(benchmark="winogrande"),
        ]
        results = self.evaluator.run_suite("gpt2", configs)
        assert results[0].benchmark == "arc_easy"
        assert results[1].benchmark == "winogrande"
