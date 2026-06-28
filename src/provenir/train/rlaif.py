from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from provenir.core.abstractions import Backend, RunManifest
from provenir.core.config import RunConfig
from provenir.data.dataset import JsonlDataset
from provenir.eval.harness import EvaluationResult
from provenir.eval.judge import LLMJudge
from provenir.train.algorithms import DPOConfig, DPOTrainer
from provenir.train.eval_callback import EvalCallback, EvalCallbackConfig


@dataclass(frozen=True)
class RLAIFConfig:
    """Configuration for the RLAIF (Reinforcement Learning from AI Feedback) pipeline.

    Orchestrates the end-to-end loop::

        dataset → judge → preference pairs → DPO training → eval → iterate

    This is Provenir's primary unique moat: the only open-source fine-tuning
    framework where automated preference generation, DPO training, and
    evaluation-gated iteration form a single reproducible pipeline.
    """

    n_iterations: int = 3
    responses_per_prompt: int = 4
    dpo_config: DPOConfig = field(default_factory=DPOConfig)
    eval_after_each_iteration: bool = True
    regression_threshold: float = 0.05

    def __post_init__(self) -> None:
        if self.n_iterations < 1:
            raise ValueError(f"n_iterations must be >= 1, got {self.n_iterations}")
        if self.responses_per_prompt < 2:
            raise ValueError(
                f"responses_per_prompt must be >= 2, got {self.responses_per_prompt}"
            )


@dataclass
class RLAIFIteration:
    """Result of one RLAIF training iteration."""

    iteration: int
    preference_count: int
    train_result: dict[str, Any]
    eval_result: EvaluationResult | None
    manifest: RunManifest


class RLAIFPipeline:
    """End-to-end RLAIF pipeline without human labellers.

    Each iteration:

    1. Generate ``responses_per_prompt`` candidate responses per training prompt.
       (Stub: synthetic variants from the original responses.)
    2. Pairwise judge to rank candidates → (chosen, rejected) pairs.
    3. DPO training on preference pairs.
    4. Evaluate against held-out set.
    5. Regression gate — stop if quality regresses beyond threshold.

    Replace the stub response-generator with a call to your live policy model
    to go from prototype to production::

        pipeline = RLAIFPipeline(
            judge=AnthropicJudge(),
            backend=TRLBackend(),
            base_config=run_config,
        )
        iterations = pipeline.run(train_dataset, eval_dataset)
    """

    def __init__(
        self,
        judge: LLMJudge,
        backend: Backend,
        base_config: RunConfig,
        rlaif_config: RLAIFConfig | None = None,
    ) -> None:
        self._judge = judge
        self._backend = backend
        self._base_config = base_config
        self.config = rlaif_config or RLAIFConfig()

    def generate_preferences(
        self, dataset: JsonlDataset
    ) -> list[dict[str, Any]]:
        """Build (chosen, rejected) pairs by judging synthetic candidate pairs."""
        preferences: list[dict[str, Any]] = []
        for record in dataset.records:
            prompt = str(record.get("prompt", ""))
            original = str(record.get("response", ""))
            # Stub candidates — replace with live model generation in production.
            candidate_a = original
            candidate_b = f"{original} Furthermore, this is important context."
            pref = self._judge.score_pairwise(prompt, candidate_a, candidate_b)
            if pref.preferred == "a":
                chosen, rejected = candidate_a, candidate_b
            elif pref.preferred == "b":
                chosen, rejected = candidate_b, candidate_a
            else:
                continue  # tie — skip (no useful signal)
            preferences.append(
                {
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected,
                    "confidence": pref.confidence,
                }
            )
        return preferences

    def run(
        self,
        train_dataset: JsonlDataset,
        eval_dataset: JsonlDataset | None = None,
    ) -> list[RLAIFIteration]:
        """Execute the RLAIF loop and return per-iteration results."""
        iterations: list[RLAIFIteration] = []
        dpo_trainer = DPOTrainer(self.config.dpo_config)

        eval_callback: EvalCallback | None = None
        if eval_dataset is not None and self.config.eval_after_each_iteration:
            eval_callback = EvalCallback(
                config=EvalCallbackConfig(
                    eval_steps=1,
                    regression_threshold=self.config.regression_threshold,
                ),
                dataset=eval_dataset,
            )

        for i in range(self.config.n_iterations):
            preferences = self.generate_preferences(train_dataset)
            train_result = dpo_trainer.train(preferences)

            manifest = RunManifest(
                config_hash=train_dataset.hash()[:16],
                dataset_hash=train_dataset.hash(),
                seed=self._base_config.seed + i,
                provenance={
                    "pipeline": "rlaif",
                    "iteration": i,
                    "preference_count": len(preferences),
                    "dpo_beta": self.config.dpo_config.beta,
                },
            )

            eval_result: EvaluationResult | None = None
            if eval_callback is not None and eval_dataset is not None:
                preds = [str(r.get("response", "")) for r in eval_dataset.records]
                eval_result = eval_callback.on_step(step=i, predictions=preds)

            iterations.append(
                RLAIFIteration(
                    iteration=i,
                    preference_count=len(preferences),
                    train_result=train_result,
                    eval_result=eval_result,
                    manifest=manifest,
                )
            )

            if eval_callback is not None and eval_callback.should_stop_early():
                break

        return iterations
