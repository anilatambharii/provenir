from __future__ import annotations

import pytest

from provenir.data.curriculum import (
    CurriculumConfig,
    CurriculumSampler,
    DifficultyScore,
    DifficultyScorer,
    _percentile_rank,
    _raw_components,
)
from provenir.data.dataset import JsonlDataset

# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

_EASY = {"prompt": "Hi", "response": "Hello"}
_MEDIUM = {
    "prompt": "What is the capital of France?",
    "response": "The capital of France is Paris, which is located in the north of the country.",
}
_HARD = {
    "prompt": "Explain quantum entanglement.",
    "response": (
        "Quantum entanglement is a phenomenon where two or more particles become correlated "
        "in such a way that the quantum state of each particle cannot be described independently "
        "of the others, regardless of the distance separating them. Measuring the state of one "
        "entangled particle instantaneously influences the state of its partner, a feature "
        "Einstein famously called 'spooky action at a distance'."
    ),
}


def _ds(*records: dict) -> JsonlDataset:
    return JsonlDataset.from_records(list(records))


# ---------------------------------------------------------------------------
# _percentile_rank
# ---------------------------------------------------------------------------


def test_percentile_rank_returns_zero_to_one() -> None:
    ranks = _percentile_rank([3.0, 1.0, 2.0])
    assert all(0.0 <= r <= 1.0 for r in ranks)


def test_percentile_rank_preserves_order() -> None:
    values = [10.0, 20.0, 30.0]
    ranks = _percentile_rank(values)
    assert ranks[0] < ranks[1] < ranks[2]


def test_percentile_rank_ties_same_rank() -> None:
    ranks = _percentile_rank([5.0, 5.0, 5.0])
    assert ranks == [0.0, 0.0, 0.0]


def test_percentile_rank_two_values() -> None:
    lo, hi = _percentile_rank([1.0, 9.0])
    assert lo == pytest.approx(0.0)
    assert hi == pytest.approx(1.0)


def test_percentile_rank_empty() -> None:
    assert _percentile_rank([]) == []


def test_percentile_rank_single() -> None:
    assert _percentile_rank([42.0]) == [0.0]


# ---------------------------------------------------------------------------
# _raw_components
# ---------------------------------------------------------------------------


def test_raw_components_keys_present() -> None:
    comps = _raw_components(_EASY)
    expected = {"response_length", "type_token_ratio", "prompt_expansion", "avg_word_length"}
    assert set(comps) == expected


def test_raw_components_empty_response() -> None:
    comps = _raw_components({"prompt": "hi", "response": ""})
    assert comps["response_length"] == 0.0
    assert comps["type_token_ratio"] == 0.0
    assert comps["avg_word_length"] == 0.0


def test_raw_components_ttr_perfect_for_all_unique() -> None:
    comps = _raw_components({"prompt": "", "response": "one two three four"})
    assert comps["type_token_ratio"] == pytest.approx(1.0)


def test_raw_components_ttr_low_for_repeated() -> None:
    comps = _raw_components({"prompt": "", "response": "the the the the"})
    assert comps["type_token_ratio"] == pytest.approx(1.0 / 4)


# ---------------------------------------------------------------------------
# DifficultyScorer
# ---------------------------------------------------------------------------


class TestDifficultyScorer:
    scorer = DifficultyScorer()

    def test_score_count_matches_dataset(self) -> None:
        ds = _ds(_EASY, _MEDIUM, _HARD)
        scores = self.scorer.score_dataset(ds)
        assert len(scores) == 3

    def test_scores_in_range(self) -> None:
        ds = _ds(_EASY, _MEDIUM, _HARD)
        for s in self.scorer.score_dataset(ds):
            assert 0.0 <= s.score <= 1.0

    def test_easy_scores_lower_than_hard(self) -> None:
        ds = _ds(_EASY, _HARD)
        scores = self.scorer.score_dataset(ds)
        easy_score = scores[0].score
        hard_score = scores[1].score
        assert easy_score < hard_score

    def test_record_indices_sequential(self) -> None:
        ds = _ds(_EASY, _MEDIUM, _HARD)
        for i, s in enumerate(self.scorer.score_dataset(ds)):
            assert s.record_index == i

    def test_components_dict_populated(self) -> None:
        ds = _ds(_EASY, _HARD)
        for s in self.scorer.score_dataset(ds):
            assert "response_length" in s.components
            assert "type_token_ratio" in s.components

    def test_empty_dataset_returns_empty(self) -> None:
        assert self.scorer.score_dataset(JsonlDataset.from_records([])) == []

    def test_single_record_scores_zero(self) -> None:
        ds = _ds(_MEDIUM)
        scores = self.scorer.score_dataset(ds)
        assert scores[0].score == pytest.approx(0.0)

    def test_custom_weights_accepted(self) -> None:
        scorer = DifficultyScorer(weights={"response_length": 1.0})
        ds = _ds(_EASY, _HARD)
        scores = scorer.score_dataset(ds)
        assert scores[0].score < scores[1].score

    def test_all_identical_records_same_score(self) -> None:
        ds = _ds(_EASY, _EASY, _EASY)
        scores = self.scorer.score_dataset(ds)
        assert all(s.score == scores[0].score for s in scores)

    def test_difficulty_score_is_frozen(self) -> None:
        score = DifficultyScore(record_index=0, score=0.5)
        with pytest.raises((AttributeError, TypeError)):
            score.score = 0.9  # type: ignore[misc]


# ---------------------------------------------------------------------------
# CurriculumConfig validation
# ---------------------------------------------------------------------------


def test_curriculum_config_valid_strategies() -> None:
    for strategy in ("easy_to_hard", "hard_to_easy", "staged", "weighted"):
        cfg = CurriculumConfig(strategy=strategy)  # type: ignore[arg-type]
        assert cfg.strategy == strategy


def test_curriculum_config_unknown_strategy_raises() -> None:
    with pytest.raises(ValueError, match="unknown strategy"):
        CurriculumConfig(strategy="curriculum")  # type: ignore[arg-type]


def test_curriculum_config_invalid_n_stages_raises() -> None:
    with pytest.raises(ValueError, match="n_stages"):
        CurriculumConfig(strategy="staged", n_stages=0)


def test_curriculum_config_default_n_stages() -> None:
    assert CurriculumConfig(strategy="easy_to_hard").n_stages == 3


# ---------------------------------------------------------------------------
# CurriculumSampler.order
# ---------------------------------------------------------------------------


class TestOrder:
    def test_easy_to_hard_ascending(self) -> None:
        ds = _ds(_HARD, _EASY)
        sampler = CurriculumSampler(CurriculumConfig(strategy="easy_to_hard"))
        indices = sampler.order(ds)
        scores = DifficultyScorer().score_dataset(ds)
        score_of = {s.record_index: s.score for s in scores}
        score_seq = [score_of[i] for i in indices]
        assert score_seq == sorted(score_seq)

    def test_hard_to_easy_descending(self) -> None:
        ds = _ds(_EASY, _MEDIUM, _HARD)
        sampler = CurriculumSampler(CurriculumConfig(strategy="hard_to_easy"))
        indices = sampler.order(ds)
        scores = DifficultyScorer().score_dataset(ds)
        score_of = {s.record_index: s.score for s in scores}
        score_seq = [score_of[i] for i in indices]
        assert score_seq == sorted(score_seq, reverse=True)

    def test_order_is_permutation(self) -> None:
        ds = _ds(_EASY, _MEDIUM, _HARD)
        sampler = CurriculumSampler(CurriculumConfig(strategy="easy_to_hard"))
        assert sorted(sampler.order(ds)) == [0, 1, 2]

    def test_other_strategy_returns_original_order(self) -> None:
        ds = _ds(_EASY, _MEDIUM, _HARD)
        sampler = CurriculumSampler(CurriculumConfig(strategy="weighted"))
        assert sampler.order(ds) == [0, 1, 2]


# ---------------------------------------------------------------------------
# CurriculumSampler.apply
# ---------------------------------------------------------------------------


class TestApply:
    def test_apply_returns_dataset_same_size(self) -> None:
        ds = _ds(_EASY, _MEDIUM, _HARD)
        sampler = CurriculumSampler(CurriculumConfig(strategy="easy_to_hard"))
        result = sampler.apply(ds)
        assert len(result.records) == 3

    def test_apply_first_record_is_easiest(self) -> None:
        ds = _ds(_HARD, _EASY)
        sampler = CurriculumSampler(CurriculumConfig(strategy="easy_to_hard"))
        result = sampler.apply(ds)
        assert result.records[0] == _EASY

    def test_apply_last_record_is_hardest(self) -> None:
        ds = _ds(_EASY, _HARD)
        sampler = CurriculumSampler(CurriculumConfig(strategy="easy_to_hard"))
        result = sampler.apply(ds)
        assert result.records[-1] == _HARD

    def test_apply_hard_to_easy_first_is_hardest(self) -> None:
        ds = _ds(_EASY, _HARD)
        sampler = CurriculumSampler(CurriculumConfig(strategy="hard_to_easy"))
        result = sampler.apply(ds)
        assert result.records[0] == _HARD


# ---------------------------------------------------------------------------
# CurriculumSampler.staged_batches
# ---------------------------------------------------------------------------


class TestStagedBatches:
    def test_stage_count_matches_n_stages(self) -> None:
        ds = _ds(_EASY, _MEDIUM, _HARD)
        sampler = CurriculumSampler(CurriculumConfig(strategy="staged", n_stages=3))
        stages = sampler.staged_batches(ds)
        assert len(stages) == 3

    def test_stages_cover_all_records(self) -> None:
        ds = _ds(_EASY, _MEDIUM, _HARD)
        sampler = CurriculumSampler(CurriculumConfig(strategy="staged", n_stages=3))
        stages = sampler.staged_batches(ds)
        all_indices = sorted(idx for stage in stages for idx in stage)
        assert all_indices == [0, 1, 2]

    def test_first_stage_easier_than_last(self) -> None:
        ds = _ds(_EASY, _MEDIUM, _HARD)
        sampler = CurriculumSampler(CurriculumConfig(strategy="staged", n_stages=3))
        stages = sampler.staged_batches(ds)
        scorer = DifficultyScorer()
        score_of = {s.record_index: s.score for s in scorer.score_dataset(ds)}
        first_avg = sum(score_of[i] for i in stages[0]) / max(1, len(stages[0]))
        last_avg = sum(score_of[i] for i in stages[-1]) / max(1, len(stages[-1]))
        assert first_avg <= last_avg

    def test_stages_capped_at_dataset_size(self) -> None:
        ds = _ds(_EASY, _MEDIUM)
        sampler = CurriculumSampler(CurriculumConfig(strategy="staged", n_stages=10))
        stages = sampler.staged_batches(ds)
        assert len(stages) == 2

    def test_staged_batches_empty_dataset(self) -> None:
        sampler = CurriculumSampler(CurriculumConfig(strategy="staged"))
        assert sampler.staged_batches(JsonlDataset.from_records([])) == []

    def test_five_records_three_stages_no_loss(self) -> None:
        records = [{"prompt": f"q{i}", "response": "x " * (i + 1)} for i in range(5)]
        ds = JsonlDataset.from_records(records)
        sampler = CurriculumSampler(CurriculumConfig(strategy="staged", n_stages=3))
        stages = sampler.staged_batches(ds)
        assert sum(len(s) for s in stages) == 5


# ---------------------------------------------------------------------------
# CurriculumSampler.weighted_sample
# ---------------------------------------------------------------------------


class TestWeightedSample:
    def test_returns_n_records(self) -> None:
        ds = _ds(_EASY, _MEDIUM, _HARD)
        sampler = CurriculumSampler(CurriculumConfig(strategy="weighted", seed=0))
        result = sampler.weighted_sample(ds, n=5, difficulty_target=0.5)
        assert len(result) == 5

    def test_records_come_from_dataset(self) -> None:
        ds = _ds(_EASY, _MEDIUM, _HARD)
        sampler = CurriculumSampler(CurriculumConfig(strategy="weighted", seed=0))
        result = sampler.weighted_sample(ds, n=10, difficulty_target=0.0)
        for record in result:
            assert record in ds.records

    def test_deterministic_with_same_seed(self) -> None:
        ds = _ds(_EASY, _MEDIUM, _HARD)
        sampler = CurriculumSampler(CurriculumConfig(strategy="weighted", seed=99))
        r1 = sampler.weighted_sample(ds, n=6, difficulty_target=0.3, seed=42)
        r2 = sampler.weighted_sample(ds, n=6, difficulty_target=0.3, seed=42)
        assert r1 == r2

    def test_easy_target_prefers_easy_records(self) -> None:
        ds = _ds(_EASY, _HARD)
        sampler = CurriculumSampler(CurriculumConfig(strategy="weighted", seed=0))
        counts: dict[str, int] = {r["response"]: 0 for r in ds.records}  # type: ignore[index]
        for record in sampler.weighted_sample(ds, n=200, difficulty_target=0.0, seed=1):
            counts[record["response"]] += 1  # type: ignore[index]
        assert counts[_EASY["response"]] > counts[_HARD["response"]]

    def test_hard_target_prefers_hard_records(self) -> None:
        ds = _ds(_EASY, _HARD)
        sampler = CurriculumSampler(CurriculumConfig(strategy="weighted", seed=0))
        counts: dict[str, int] = {r["response"]: 0 for r in ds.records}  # type: ignore[index]
        for record in sampler.weighted_sample(ds, n=200, difficulty_target=1.0, seed=2):
            counts[record["response"]] += 1  # type: ignore[index]
        assert counts[_HARD["response"]] > counts[_EASY["response"]]

    def test_empty_dataset_returns_empty(self) -> None:
        sampler = CurriculumSampler(CurriculumConfig(strategy="weighted"))
        result = sampler.weighted_sample(JsonlDataset.from_records([]), n=5, difficulty_target=0.5)
        assert result == []


# ---------------------------------------------------------------------------
# CurriculumSampler.sample_for_step
# ---------------------------------------------------------------------------


class TestSampleForStep:
    def test_returns_n_records(self) -> None:
        ds = _ds(_EASY, _MEDIUM, _HARD)
        sampler = CurriculumSampler(CurriculumConfig(strategy="weighted", seed=0))
        result = sampler.sample_for_step(ds, n=4, step=0, total_steps=10)
        assert len(result) == 4

    def test_step_zero_targets_easy(self) -> None:
        ds = _ds(_EASY, _HARD)
        sampler = CurriculumSampler(CurriculumConfig(strategy="weighted", seed=0))
        # At step 0 the target is 0.0 (easiest); easy record should dominate
        counts: dict[str, int] = {r["response"]: 0 for r in ds.records}  # type: ignore[index]
        for record in sampler.sample_for_step(ds, n=200, step=0, total_steps=10, seed=3):
            counts[record["response"]] += 1  # type: ignore[index]
        assert counts[_EASY["response"]] > counts[_HARD["response"]]

    def test_last_step_targets_hard(self) -> None:
        ds = _ds(_EASY, _HARD)
        sampler = CurriculumSampler(CurriculumConfig(strategy="weighted", seed=0))
        counts: dict[str, int] = {r["response"]: 0 for r in ds.records}  # type: ignore[index]
        for record in sampler.sample_for_step(ds, n=200, step=10, total_steps=10, seed=4):
            counts[record["response"]] += 1  # type: ignore[index]
        assert counts[_HARD["response"]] > counts[_EASY["response"]]

    def test_sample_for_step_deterministic(self) -> None:
        ds = _ds(_EASY, _MEDIUM, _HARD)
        sampler = CurriculumSampler(CurriculumConfig(strategy="weighted", seed=7))
        r1 = sampler.sample_for_step(ds, n=5, step=3, total_steps=10, seed=7)
        r2 = sampler.sample_for_step(ds, n=5, step=3, total_steps=10, seed=7)
        assert r1 == r2
