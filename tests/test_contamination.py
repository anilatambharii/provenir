"""Tests for the contamination firewall and canary leakage guard."""

from __future__ import annotations

import pytest

from provenir.data.dataset import JsonlDataset
from provenir.eval import contamination as contam
from provenir.eval.canary import Canary, CanaryGuard
from provenir.eval.contamination import (
    ContaminationChecker,
    ContaminationConfig,
    ContaminationHit,
    ContaminationReport,
    MinHashDeduplicator,
)


def _ds(prompts: list[str]) -> JsonlDataset:
    return JsonlDataset.from_records([{"prompt": p} for p in prompts])


# -- config validation ------------------------------------------------------


def test_config_defaults_are_the_standard() -> None:
    cfg = ContaminationConfig()
    assert cfg.ngram_n == 13
    assert cfg.method == "ngram"
    assert cfg.text_key == "prompt"


def test_config_rejects_bad_ngram_n() -> None:
    with pytest.raises(ValueError):
        ContaminationConfig(ngram_n=0)


def test_config_rejects_out_of_range_ngram_threshold() -> None:
    with pytest.raises(ValueError):
        ContaminationConfig(ngram_threshold=0.0)
    with pytest.raises(ValueError):
        ContaminationConfig(ngram_threshold=1.5)


def test_config_rejects_out_of_range_embedding_threshold() -> None:
    with pytest.raises(ValueError):
        ContaminationConfig(embedding_threshold=1.1)


def test_config_rejects_unknown_method() -> None:
    with pytest.raises(ValueError):
        ContaminationConfig(method="fuzzy")


def test_config_rejects_empty_text_key() -> None:
    with pytest.raises(ValueError):
        ContaminationConfig(text_key="")


# -- exact method -----------------------------------------------------------


def test_exact_detects_normalized_equality() -> None:
    train = _ds(["What is the CAPITAL of France?", "Unrelated row"])
    checker = ContaminationChecker(ContaminationConfig(method="exact"))
    report = checker.check(train, ["what is the capital of france?"])
    assert report.method == "exact"
    assert report.contaminated_train_indices == {0}
    assert report.hits[0].score == 1.0
    assert report.hits[0].method == "exact"


def test_exact_clean_when_no_match() -> None:
    train = _ds(["alpha", "beta"])
    checker = ContaminationChecker(ContaminationConfig(method="exact"))
    report = checker.check(train, ["gamma"])
    assert report.is_clean
    assert report.contamination_rate == 0.0


def test_exact_ignores_empty_rows() -> None:
    train = _ds(["", ""])
    checker = ContaminationChecker(ContaminationConfig(method="exact"))
    report = checker.check(train, [""])
    assert report.is_clean


# -- ngram method -----------------------------------------------------------


def test_ngram_detects_exact_overlap() -> None:
    text = "the quick brown fox jumps over the lazy dog again"
    train = _ds([text, "a completely different unrelated sentence entirely"])
    cfg = ContaminationConfig(method="ngram", ngram_n=3, ngram_threshold=0.8)
    report = ContaminationChecker(cfg).check(train, [text])
    assert report.contaminated_train_indices == {0}
    assert report.hits[0].method == "ngram"
    assert report.hits[0].score == pytest.approx(1.0)


def test_ngram_detects_containment_of_short_eval() -> None:
    train = _ds([
        "intro words then the quick brown fox jumps over lazy dog then more words",
    ])
    cfg = ContaminationConfig(method="ngram", ngram_n=3, ngram_threshold=0.9)
    report = ContaminationChecker(cfg).check(
        train, ["the quick brown fox jumps over lazy dog"]
    )
    # Short eval fully contained -> containment ratio == 1.0.
    assert report.contaminated_train_indices == {0}
    assert report.hits[0].score == pytest.approx(1.0)


def test_ngram_clean_on_unrelated_text() -> None:
    train = _ds(["completely unrelated training material here today"])
    cfg = ContaminationConfig(method="ngram", ngram_n=3, ngram_threshold=0.8)
    report = ContaminationChecker(cfg).check(
        train, ["an entirely separate evaluation prompt about physics"]
    )
    assert report.is_clean


def test_ngram_threshold_boundary_is_inclusive() -> None:
    # Two 3-grams overlap fully in a 4-word eval item vs itself.
    train = _ds(["one two three four", "one two nine ten"])
    cfg = ContaminationConfig(method="ngram", ngram_n=3, ngram_threshold=0.5)
    report = ContaminationChecker(cfg).check(train, ["one two three four"])
    # Row 0 -> 1.0 overlap. Row 1 shares grams {"one two three"}? no;
    # grams row1 = {"one two nine", "two nine ten"} -> overlap 0 with eval.
    assert report.contaminated_train_indices == {0}


def test_ngram_threshold_too_high_yields_no_hit() -> None:
    train = _ds(["one two three four five six seven"])
    cfg = ContaminationConfig(method="ngram", ngram_n=3, ngram_threshold=1.0)
    report = ContaminationChecker(cfg).check(
        train, ["one two three four five six eight"]
    )
    # Not a full containment, so < 1.0 threshold not met.
    assert report.is_clean


def test_ngram_short_text_falls_back_to_whole_string() -> None:
    train = _ds(["hi there"])
    cfg = ContaminationConfig(method="ngram", ngram_n=13, ngram_threshold=0.8)
    report = ContaminationChecker(cfg).check(train, ["hi there"])
    assert report.contaminated_train_indices == {0}


# -- report math ------------------------------------------------------------


def test_contamination_rate_math() -> None:
    train = _ds(["dup one", "dup two", "clean three", "clean four"])
    cfg = ContaminationConfig(method="exact")
    report = ContaminationChecker(cfg).check(train, ["dup one", "dup two"])
    assert report.train_size == 4
    assert report.eval_size == 2
    assert report.contamination_rate == 0.5


def test_contamination_rate_zero_when_empty_train() -> None:
    report = ContaminationReport(
        hits=[], train_size=0, eval_size=3, method="ngram", threshold=0.8
    )
    assert report.contamination_rate == 0.0
    assert report.is_clean


def test_report_to_dict_roundtrips_fields() -> None:
    train = _ds(["match me", "leave me"])
    report = ContaminationChecker(ContaminationConfig(method="exact")).check(
        train, ["match me"]
    )
    payload = report.to_dict()
    assert payload["train_size"] == 2
    assert payload["method"] == "exact"
    assert payload["contaminated_train_indices"] == [0]
    assert payload["is_clean"] is False
    assert payload["hits"][0]["train_index"] == 0


def test_hit_to_dict() -> None:
    hit = ContaminationHit(
        train_index=1, eval_index=2, score=0.9, method="ngram", snippet="x"
    )
    assert hit.to_dict()["score"] == 0.9


# -- filter_contaminated ----------------------------------------------------


def test_filter_contaminated_removes_flagged_rows() -> None:
    train = _ds(["bad", "good", "bad2", "good2"])
    checker = ContaminationChecker(ContaminationConfig(method="exact"))
    report = checker.check(train, ["bad", "bad2"])
    filtered = checker.filter_contaminated(train, report)
    prompts = [r["prompt"] for r in filtered.records]
    assert prompts == ["good", "good2"]
    assert isinstance(filtered, JsonlDataset)


def test_filter_contaminated_noop_when_clean() -> None:
    train = _ds(["a", "b"])
    checker = ContaminationChecker(ContaminationConfig(method="exact"))
    report = checker.check(train, ["z"])
    filtered = checker.filter_contaminated(train, report)
    assert len(filtered.records) == 2


# -- check_datasets ---------------------------------------------------------


def test_check_datasets_pulls_text_key() -> None:
    train = _ds(["shared prompt here", "unique training row"])
    eval_ds = _ds(["shared prompt here"])
    checker = ContaminationChecker(ContaminationConfig(method="exact"))
    report = checker.check_datasets(train, eval_ds)
    assert report.contaminated_train_indices == {0}


def test_custom_text_key_is_respected() -> None:
    train = JsonlDataset.from_records([{"question": "leaked question text"}])
    eval_ds = JsonlDataset.from_records([{"question": "leaked question text"}])
    cfg = ContaminationConfig(method="exact", text_key="question")
    report = ContaminationChecker(cfg).check_datasets(train, eval_ds)
    assert report.contaminated_train_indices == {0}


# -- embedding method / fallback -------------------------------------------


def test_embedding_falls_back_to_ngram_without_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(contam, "_HAS_SENTENCE_TRANSFORMERS", False)
    text = "the quick brown fox jumps over the lazy dog now"
    train = _ds([text])
    cfg = ContaminationConfig(method="embedding", ngram_n=3, ngram_threshold=0.8)
    report = ContaminationChecker(cfg).check(train, [text])
    # Fallback records the method actually used.
    assert report.method == "ngram"
    assert report.contaminated_train_indices == {0}


@pytest.mark.skipif(
    not contam._HAS_SENTENCE_TRANSFORMERS,
    reason="sentence-transformers not installed",
)
def test_embedding_detects_paraphrase() -> None:
    train = _ds(["What is the capital city of France?"])
    cfg = ContaminationConfig(method="embedding", embedding_threshold=0.85)
    report = ContaminationChecker(cfg).check(
        train, ["What is the capital city of France?"]
    )
    assert report.method == "embedding"
    assert report.contaminated_train_indices == {0}
    assert report.hits[0].score >= 0.85


@pytest.mark.skipif(
    not contam._HAS_SENTENCE_TRANSFORMERS,
    reason="sentence-transformers not installed",
)
def test_embedding_clean_on_unrelated() -> None:
    train = _ds(["The recipe calls for two cups of flour."])
    cfg = ContaminationConfig(method="embedding", embedding_threshold=0.9)
    report = ContaminationChecker(cfg).check(
        train, ["Quantum entanglement links distant particles."]
    )
    assert report.is_clean


# -- MinHash ----------------------------------------------------------------


def test_minhash_identical_texts_estimate_one() -> None:
    dedup = MinHashDeduplicator(num_perm=64, ngram_n=3)
    text = "the quick brown fox jumps over the lazy dog"
    assert dedup.estimated_jaccard(
        dedup.signature(text), dedup.signature(text)
    ) == pytest.approx(1.0)


def test_minhash_disjoint_texts_estimate_low() -> None:
    dedup = MinHashDeduplicator(num_perm=128, ngram_n=3)
    a = dedup.signature("apples bananas cherries dates elderberries figs")
    b = dedup.signature("zebras yaks xerus wombats vultures turtles")
    assert dedup.estimated_jaccard(a, b) < 0.2


def test_minhash_is_deterministic_across_instances() -> None:
    text = "reproducible signatures matter for distributed audits everywhere"
    a = MinHashDeduplicator(num_perm=32, ngram_n=3, seed=7).signature(text)
    b = MinHashDeduplicator(num_perm=32, ngram_n=3, seed=7).signature(text)
    assert a == b


def test_minhash_rejects_mismatched_signature_lengths() -> None:
    dedup = MinHashDeduplicator(num_perm=8)
    with pytest.raises(ValueError):
        dedup.estimated_jaccard((1, 2, 3), (1, 2))


def test_minhash_rejects_bad_params() -> None:
    with pytest.raises(ValueError):
        MinHashDeduplicator(num_perm=0)
    with pytest.raises(ValueError):
        MinHashDeduplicator(ngram_n=0)


# -- canary -----------------------------------------------------------------


def test_canary_mint_is_deterministic() -> None:
    guard = CanaryGuard()
    assert guard.mint("eval-x").token == guard.mint("eval-x").token


def test_canary_mint_is_unique_per_eval_set() -> None:
    guard = CanaryGuard()
    assert guard.mint("eval-x").token != guard.mint("eval-y").token


def test_canary_mint_seed_changes_token() -> None:
    guard = CanaryGuard()
    assert guard.mint("eval-x", seed="a").token != guard.mint("eval-x", seed="b").token


def test_canary_mint_token_shape() -> None:
    canary = CanaryGuard().mint("eval-x")
    assert canary.token.startswith("provenir-canary:")
    assert canary.eval_set_id == "eval-x"


def test_canary_mint_rejects_empty_id() -> None:
    with pytest.raises(ValueError):
        CanaryGuard().mint("")


def test_canary_dataclass_validation() -> None:
    with pytest.raises(ValueError):
        Canary(token="", eval_set_id="x")
    with pytest.raises(ValueError):
        Canary(token="t", eval_set_id="")


def test_canary_tag_then_scan_detects_leakage() -> None:
    guard = CanaryGuard()
    canary = guard.mint("secret-eval")
    eval_ds = _ds(["what is 2 + 2?", "name a prime number"])
    tagged = guard.tag(eval_ds, canary)
    # Simulate leakage: tagged eval rows end up mixed into training.
    train = JsonlDataset.from_records([{"prompt": "clean row"}, *tagged.records])
    leaked = guard.scan(train, canary)
    assert leaked == [1, 2]


def test_canary_scan_no_false_positive_on_clean_train() -> None:
    guard = CanaryGuard()
    canary = guard.mint("secret-eval")
    train = _ds(["ordinary training text", "more ordinary text"])
    assert guard.scan(train, canary) == []


def test_canary_tag_preserves_original_records() -> None:
    guard = CanaryGuard()
    canary = guard.mint("eval-z")
    eval_ds = _ds(["original prompt"])
    tagged = guard.tag(eval_ds, canary)
    assert eval_ds.records[0]["prompt"] == "original prompt"
    assert canary.token in tagged.records[0]["prompt"]
    assert tagged.records[0]["_canary"] == canary.token


def test_canary_scan_detects_via_canary_field_only() -> None:
    guard = CanaryGuard()
    canary = guard.mint("eval-z")
    # Token only in the structured field, not the text.
    train = JsonlDataset.from_records([{"prompt": "harmless", "_canary": canary.token}])
    assert guard.scan(train, canary) == [0]


def test_detect_any_maps_eval_sets_to_leaks() -> None:
    guard = CanaryGuard()
    c1 = guard.mint("eval-1")
    c2 = guard.mint("eval-2")
    texts = [
        "clean",
        f"leaked {c1.token}",
        f"also leaked {c1.token}",
        f"other {c2.token}",
    ]
    result = guard.detect_any(texts, [c1, c2])
    assert result == {"eval-1": [1, 2], "eval-2": [3]}


def test_detect_any_omits_clean_eval_sets() -> None:
    guard = CanaryGuard()
    c1 = guard.mint("eval-1")
    result = guard.detect_any(["nothing here"], [c1])
    assert result == {}
