from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Literal

from provenir.data.dataset import JsonlDataset

# ---------------------------------------------------------------------------
# Scoring component weights
# ---------------------------------------------------------------------------

_DEFAULT_WEIGHTS: dict[str, float] = {
    "response_length": 0.4,
    "type_token_ratio": 0.2,
    "prompt_expansion": 0.2,
    "avg_word_length": 0.2,
}


# ---------------------------------------------------------------------------
# Difficulty score dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DifficultyScore:
    """Difficulty estimate for a single training record.

    *score* is normalised to [0, 1] where 0.0 = easiest, 1.0 = hardest.
    *components* carries the percentile-ranked raw values for each proxy.
    """

    record_index: int
    score: float
    components: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _percentile_rank(values: list[float]) -> list[float]:
    """Map a list of raw values to [0, 1] percentile ranks.

    Ties receive the same rank.  All-equal input → all zeros.
    """
    n = len(values)
    if n == 0:
        return []
    unique_sorted = sorted(set(values))
    if len(unique_sorted) == 1:
        return [0.0] * n
    denom = float(len(unique_sorted) - 1)
    rank_map = {v: i / denom for i, v in enumerate(unique_sorted)}
    return [rank_map[v] for v in values]


def _raw_components(record: dict[str, Any]) -> dict[str, float]:
    """Extract raw proxy values from a single record."""
    response = str(record.get("response", ""))
    prompt = str(record.get("prompt", ""))
    tokens = response.split()
    prompt_len = len(prompt.split())
    return {
        "response_length": float(len(tokens)),
        "type_token_ratio": len(set(tokens)) / len(tokens) if tokens else 0.0,
        "prompt_expansion": len(tokens) / max(1, prompt_len),
        "avg_word_length": sum(len(t) for t in tokens) / len(tokens) if tokens else 0.0,
    }


# ---------------------------------------------------------------------------
# DifficultyScorer
# ---------------------------------------------------------------------------


class DifficultyScorer:
    """Heuristic difficulty scorer — no model required.

    Uses four proxy metrics computed from ``prompt`` and ``response`` fields:

    * **response_length** — word count (longer → harder)
    * **type_token_ratio** — unique/total tokens (richer vocabulary → harder)
    * **prompt_expansion** — response length / prompt length (more expansion → harder)
    * **avg_word_length** — mean character length of tokens (longer words → harder)

    Each component is percentile-ranked across the dataset so scores are
    relative within the corpus rather than absolute.
    """

    def __init__(self, weights: dict[str, float] | None = None) -> None:
        self.weights: dict[str, float] = (
            dict(weights) if weights is not None else dict(_DEFAULT_WEIGHTS)
        )

    def score_dataset(self, dataset: JsonlDataset) -> list[DifficultyScore]:
        """Return one :class:`DifficultyScore` per record in dataset order."""
        n = len(dataset.records)
        if n == 0:
            return []

        # Collect raw values column-by-column.
        raw: dict[str, list[float]] = {}
        for record in dataset.records:
            for k, v in _raw_components(record).items():
                raw.setdefault(k, []).append(v)

        # Percentile-rank each column.
        ranked: dict[str, list[float]] = {k: _percentile_rank(vs) for k, vs in raw.items()}

        weight_sum = sum(self.weights.get(k, 0.0) for k in ranked)

        scores: list[DifficultyScore] = []
        for i in range(n):
            composite = (
                sum(self.weights.get(k, 0.0) * ranked[k][i] for k in ranked) / weight_sum
                if weight_sum > 0.0
                else 0.0
            )
            scores.append(
                DifficultyScore(
                    record_index=i,
                    score=max(0.0, min(1.0, composite)),
                    components={k: ranked[k][i] for k in ranked},
                )
            )
        return scores


# ---------------------------------------------------------------------------
# CurriculumConfig
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CurriculumConfig:
    """Configuration for curriculum-ordered training.

    *strategy*:

    * ``"easy_to_hard"`` — sort records ascending by difficulty (default for
      classic curriculum learning)
    * ``"hard_to_easy"`` — descending (anti-curriculum; sometimes better for
      smaller models)
    * ``"staged"`` — divide into *n_stages* buckets of increasing difficulty;
      useful when the trainer iterates through stages explicitly
    * ``"weighted"`` — step-aware weighted sampling; call
      :meth:`CurriculumSampler.sample_for_step` to get the right mix per step
    """

    strategy: Literal["easy_to_hard", "hard_to_easy", "staged", "weighted"]
    n_stages: int = 3
    seed: int = 0
    scorer_weights: dict[str, float] | None = None

    def __post_init__(self) -> None:
        valid = {"easy_to_hard", "hard_to_easy", "staged", "weighted"}
        if self.strategy not in valid:
            raise ValueError(f"unknown strategy {self.strategy!r}; expected one of {sorted(valid)}")
        if self.n_stages < 1:
            raise ValueError(f"n_stages must be >= 1, got {self.n_stages}")


# ---------------------------------------------------------------------------
# CurriculumSampler
# ---------------------------------------------------------------------------


class CurriculumSampler:
    """Applies curriculum ordering or step-aware sampling to a :class:`JsonlDataset`.

    Usage::

        cfg = CurriculumConfig(strategy="easy_to_hard")
        sampler = CurriculumSampler(cfg)
        ordered_ds = sampler.apply(dataset)      # returns new JsonlDataset
        stages = sampler.staged_batches(dataset) # list of index lists
    """

    def __init__(self, config: CurriculumConfig) -> None:
        self.config = config
        self._scorer = DifficultyScorer(weights=config.scorer_weights)

    # ------------------------------------------------------------------
    # Core ordering
    # ------------------------------------------------------------------

    def order(self, dataset: JsonlDataset) -> list[int]:
        """Return record indices in curriculum order.

        For ``"easy_to_hard"`` and ``"hard_to_easy"`` strategies; other strategies
        return indices in dataset order.
        """
        scores = self._scorer.score_dataset(dataset)
        if self.config.strategy == "easy_to_hard":
            return [s.record_index for s in sorted(scores, key=lambda s: s.score)]
        if self.config.strategy == "hard_to_easy":
            return [s.record_index for s in sorted(scores, key=lambda s: s.score, reverse=True)]
        return [s.record_index for s in scores]

    def apply(self, dataset: JsonlDataset) -> JsonlDataset:
        """Return a new :class:`JsonlDataset` reordered by curriculum strategy."""
        return JsonlDataset.from_records([dataset.records[i] for i in self.order(dataset)])

    # ------------------------------------------------------------------
    # Staged batches
    # ------------------------------------------------------------------

    def staged_batches(self, dataset: JsonlDataset) -> list[list[int]]:
        """Divide the dataset into *n_stages* groups sorted by difficulty.

        The first group contains the easiest records; the last the hardest.
        Remaining records after even division are added to the final stage.
        """
        scores = self._scorer.score_dataset(dataset)
        n = len(scores)
        if n == 0:
            return []
        n_stages = min(self.config.n_stages, n)
        sorted_indices = [s.record_index for s in sorted(scores, key=lambda s: s.score)]
        base, remainder = divmod(n, n_stages)
        stages: list[list[int]] = []
        cursor = 0
        for i in range(n_stages):
            size = base + (1 if i < remainder else 0)
            stages.append(sorted_indices[cursor : cursor + size])
            cursor += size
        return stages

    # ------------------------------------------------------------------
    # Weighted / step-aware sampling
    # ------------------------------------------------------------------

    def weighted_sample(
        self,
        dataset: JsonlDataset,
        n: int,
        difficulty_target: float,
        seed: int | None = None,
    ) -> list[dict[str, Any]]:
        """Sample *n* records weighted towards *difficulty_target* ∈ [0, 1].

        Weight = 1 / (1 + (score − target)²), giving a soft preference for
        records near the target difficulty while still drawing from the full
        distribution.
        """
        scores = self._scorer.score_dataset(dataset)
        if not scores:
            return []
        rng = random.Random(seed if seed is not None else self.config.seed)
        weights = [1.0 / (1.0 + (s.score - difficulty_target) ** 2) for s in scores]
        total = sum(weights)
        probs = [w / total for w in weights] if total > 0.0 else [1.0 / len(weights)] * len(weights)
        indices = rng.choices(range(len(dataset.records)), weights=probs, k=n)
        return [dataset.records[i] for i in indices]

    def sample_for_step(
        self,
        dataset: JsonlDataset,
        n: int,
        step: int,
        total_steps: int,
        seed: int | None = None,
    ) -> list[dict[str, Any]]:
        """Sample *n* records appropriate for *step* in a *total_steps* schedule.

        At step 0 the target difficulty is 0.0 (easy); at *total_steps* it is
        1.0 (hard).  Useful for ``"weighted"`` strategy during training.
        """
        target = step / max(1, total_steps)
        return self.weighted_sample(dataset, n, target, seed=seed)
