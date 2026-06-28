from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from provenir.data.dataset import JsonlDataset
from provenir.eval.harness import EvaluationResult
from provenir.eval.judge import LLMJudge


@dataclass(frozen=True)
class FlywheelConfig:
    """Configuration for the autonomous data flywheel loop."""

    min_score_threshold: float = 0.5
    max_variants_per_failure: int = 3
    quality_filter_threshold: float = 0.6

    def __post_init__(self) -> None:
        if not (0.0 <= self.min_score_threshold <= 1.0):
            raise ValueError("min_score_threshold must be in [0, 1]")
        if not (0.0 <= self.quality_filter_threshold <= 1.0):
            raise ValueError("quality_filter_threshold must be in [0, 1]")
        if self.max_variants_per_failure < 1:
            raise ValueError("max_variants_per_failure must be >= 1")


class DataFlywheel:
    """Auto-generates training data from model failure cases via an LLM judge.

    Pipeline::

        dataset + eval_result
            → mine_failures()      # records below threshold
            → generate_variants()  # judge-driven augmentation per failure
            → quality filter       # drop variants below threshold
            → augmented dataset

    This creates a self-improving loop: fine-tune → eval → mine failures →
    augment → re-fine-tune.  Use :class:`CurriculumSampler` to integrate the
    augmented dataset back into training with appropriate difficulty ordering.
    """

    def __init__(
        self,
        judge: LLMJudge,
        config: FlywheelConfig | None = None,
    ) -> None:
        self._judge = judge
        self.config = config or FlywheelConfig()

    def mine_failures(
        self,
        dataset: JsonlDataset,
        result: EvaluationResult,
    ) -> list[dict[str, Any]]:
        """Return records where global accuracy is below *min_score_threshold*.

        When per-example scores are unavailable (aggregate-only result), the
        whole dataset is treated as the failure set when accuracy < threshold.
        """
        accuracy = result.accuracy
        if accuracy is None:
            return []
        if accuracy.mean >= self.config.min_score_threshold:
            return []
        return list(dataset.records)

    def generate_variants(
        self,
        failure: dict[str, Any],
        n: int | None = None,
    ) -> list[dict[str, Any]]:
        """Generate *n* augmented variants of *failure* and keep quality ones.

        In a real deployment the candidate responses come from a model
        generation step.  Here the judge acts on synthetic variants so the
        pipeline runs end-to-end without a live model.
        """
        n = n or self.config.max_variants_per_failure
        prompt = str(failure.get("prompt", ""))
        original = str(failure.get("response", ""))
        criteria = ["accuracy", "completeness", "clarity"]

        variants: list[dict[str, Any]] = []
        for i in range(n):
            candidate = f"{original} (elaborated, version {i + 1})"
            scores = self._judge.score_rubric(prompt, candidate, criteria)
            score = sum(s.score for s in scores) / max(len(scores), 1)
            if score >= self.config.quality_filter_threshold:
                pref = self._judge.score_pairwise(prompt, original, candidate)
                variants.append(
                    {
                        **failure,
                        "response": candidate,
                        "_flywheel": {
                            "original_response": original,
                            "variant_index": i,
                            "quality_score": score,
                            "preferred_over_original": pref.preferred == "b",
                        },
                    }
                )
        return variants

    def run(
        self,
        dataset: JsonlDataset,
        result: EvaluationResult,
    ) -> JsonlDataset:
        """Mine failures, generate variants, and return the augmented dataset.

        The original records are always retained; only new variant records are
        appended.  Returns the original dataset unchanged when no failures are
        found.
        """
        failures = self.mine_failures(dataset, result)
        new_records: list[dict[str, Any]] = []
        for failure in failures:
            new_records.extend(self.generate_variants(failure))
        return JsonlDataset.from_records(list(dataset.records) + new_records)
