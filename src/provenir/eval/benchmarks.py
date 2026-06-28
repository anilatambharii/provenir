from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

try:
    import lm_eval

    _HAS_LM_EVAL = True
except ImportError:
    _HAS_LM_EVAL = False

SUPPORTED_BENCHMARKS: frozenset[str] = frozenset(
    [
        "mmlu",
        "hellaswag",
        "arc_easy",
        "arc_challenge",
        "truthfulqa_mc1",
        "truthfulqa_mc2",
        "winogrande",
        "gsm8k",
        "humaneval",
    ]
)

BenchmarkName = Literal[
    "mmlu",
    "hellaswag",
    "arc_easy",
    "arc_challenge",
    "truthfulqa_mc1",
    "truthfulqa_mc2",
    "winogrande",
    "gsm8k",
    "humaneval",
]


@dataclass(frozen=True)
class BenchmarkConfig:
    """Configuration for a single benchmark evaluation pass."""

    benchmark: BenchmarkName
    num_fewshot: int = 0
    limit: int | None = None
    batch_size: int = 8

    def __post_init__(self) -> None:
        if self.benchmark not in SUPPORTED_BENCHMARKS:
            raise ValueError(
                f"unknown benchmark {self.benchmark!r}; "
                f"supported: {sorted(SUPPORTED_BENCHMARKS)}"
            )
        if self.num_fewshot < 0:
            raise ValueError(f"num_fewshot must be >= 0, got {self.num_fewshot}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")


@dataclass
class BenchmarkResult:
    """Result of running one benchmark evaluation."""

    benchmark: str
    score: float
    num_examples: int
    num_fewshot: int
    metric_name: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark": self.benchmark,
            "score": self.score,
            "num_examples": self.num_examples,
            "num_fewshot": self.num_fewshot,
            "metric_name": self.metric_name,
            "metadata": self.metadata,
        }


class BenchmarkEvaluator:
    """Wrapper around EleutherAI lm-evaluation-harness for standard benchmarks.

    Requires ``lm-eval``: ``pip install provenir[benchmarks]``.
    Returns stub results with ``score=0.0`` and ``metadata={stub: True}`` when
    the package is unavailable, so pipelines that log benchmark results still
    work in environments without a GPU.
    """

    def run(
        self,
        model_path: str | Path,
        config: BenchmarkConfig,
    ) -> BenchmarkResult:
        if not _HAS_LM_EVAL:
            return BenchmarkResult(
                benchmark=config.benchmark,
                score=0.0,
                num_examples=0,
                num_fewshot=config.num_fewshot,
                metric_name="acc",
                metadata={"stub": True, "reason": "lm-eval not installed"},
            )

        results: dict[str, Any] = lm_eval.simple_evaluate(
            model="hf",
            model_args=f"pretrained={model_path}",
            tasks=[config.benchmark],
            num_fewshot=config.num_fewshot,
            limit=config.limit,
            batch_size=config.batch_size,
        )
        task_results: dict[str, Any] = results["results"][config.benchmark]
        primary_metric = next(iter(task_results))
        score = float(task_results[primary_metric])
        n_examples = int(task_results.get("n", 0))
        return BenchmarkResult(
            benchmark=config.benchmark,
            score=score,
            num_examples=n_examples,
            num_fewshot=config.num_fewshot,
            metric_name=primary_metric,
            metadata=task_results,
        )

    def run_suite(
        self,
        model_path: str | Path,
        configs: list[BenchmarkConfig],
    ) -> list[BenchmarkResult]:
        """Run multiple benchmarks sequentially and return all results."""
        return [self.run(model_path, cfg) for cfg in configs]
