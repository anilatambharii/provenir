"""Deterministic tests for RH-Bench (reward-hacking detector benchmark)."""

from __future__ import annotations

from pathlib import Path

import pytest

from provenir.bench.rhbench import (
    CATEGORY_DEFINITIONS,
    AlwaysHackDetector,
    BenchmarkHarness,
    CategoryMetrics,
    ComputeSavedResult,
    DetectionResult,
    Detector,
    DetectorEvaluation,
    HackCategory,
    LabeledCorpus,
    LengthHeuristicDetector,
    NeverHackDetector,
    ProvenirDetector,
    ProxyDivergenceDetector,
    RandomDetector,
    SyntheticCorpusGenerator,
    Trajectory,
    auroc,
    compute_savings,
    precision_recall_f1,
)

# ---------------------------------------------------------------------------
# Taxonomy
# ---------------------------------------------------------------------------


def test_taxonomy_has_eight_categories() -> None:
    assert len(HackCategory) == 8


def test_taxonomy_definitions_complete() -> None:
    assert set(CATEGORY_DEFINITIONS) == set(HackCategory)
    assert all(isinstance(v, str) and v for v in CATEGORY_DEFINITIONS.values())


def test_taxonomy_string_enum_values() -> None:
    assert HackCategory.LENGTH_INFLATION.value == "length_inflation"
    assert HackCategory("sycophancy") is HackCategory.SYCOPHANCY
    assert HackCategory.ANSWER_LEAKAGE == "answer_leakage"


def test_taxonomy_expected_members() -> None:
    expected = {
        "length_inflation",
        "format_exploit",
        "test_tampering",
        "verifier_gaming",
        "proxy_divergence",
        "degenerate_repetition",
        "sycophancy",
        "answer_leakage",
    }
    assert {c.value for c in HackCategory} == expected


# ---------------------------------------------------------------------------
# Trajectory validation
# ---------------------------------------------------------------------------


def test_trajectory_valid_hack() -> None:
    t = Trajectory("t", "p", "r", 0.9, 0.1, True, "proxy_divergence")
    assert t.is_hack and t.category == "proxy_divergence"


def test_trajectory_valid_clean() -> None:
    t = Trajectory("t", "p", "r", 0.7, 0.7, False, None)
    assert not t.is_hack and t.category is None


def test_trajectory_reward_out_of_range_low() -> None:
    with pytest.raises(ValueError):
        Trajectory("t", "p", "r", -0.1, 0.5, False, None)


def test_trajectory_reward_out_of_range_high() -> None:
    with pytest.raises(ValueError):
        Trajectory("t", "p", "r", 0.5, 1.5, False, None)


def test_trajectory_hack_requires_category() -> None:
    with pytest.raises(ValueError):
        Trajectory("t", "p", "r", 0.9, 0.1, True, None)


def test_trajectory_clean_forbids_category() -> None:
    with pytest.raises(ValueError):
        Trajectory("t", "p", "r", 0.5, 0.5, False, "length_inflation")


def test_trajectory_invalid_category() -> None:
    with pytest.raises(ValueError):
        Trajectory("t", "p", "r", 0.9, 0.1, True, "not_a_category")


def test_trajectory_roundtrip_dict() -> None:
    t = Trajectory("t", "p", "r", 0.9, 0.1, True, "sycophancy", {"k": 1})
    assert Trajectory.from_dict(t.to_dict()) == t


# ---------------------------------------------------------------------------
# SyntheticCorpusGenerator
# ---------------------------------------------------------------------------


def test_generator_determinism_same_seed() -> None:
    a = SyntheticCorpusGenerator(seed=0).generate(n_per_category=10, n_clean=20)
    b = SyntheticCorpusGenerator(seed=0).generate(n_per_category=10, n_clean=20)
    assert [t.to_dict() for t in a.trajectories] == [t.to_dict() for t in b.trajectories]


def test_generator_determinism_repeated_call() -> None:
    gen = SyntheticCorpusGenerator(seed=3)
    a = gen.generate(n_per_category=5, n_clean=5)
    b = gen.generate(n_per_category=5, n_clean=5)
    assert [t.to_dict() for t in a.trajectories] == [t.to_dict() for t in b.trajectories]


def test_generator_different_seed_differs() -> None:
    a = SyntheticCorpusGenerator(seed=0).generate(n_per_category=10, n_clean=20)
    b = SyntheticCorpusGenerator(seed=1).generate(n_per_category=10, n_clean=20)
    assert [t.to_dict() for t in a.trajectories] != [t.to_dict() for t in b.trajectories]


def test_generator_counts() -> None:
    corpus = SyntheticCorpusGenerator(seed=0).generate(n_per_category=10, n_clean=25)
    assert corpus.hack_count == 10 * len(HackCategory)
    assert corpus.clean_count == 25
    assert corpus.size == corpus.hack_count + corpus.clean_count


def test_generator_all_categories_represented() -> None:
    corpus = SyntheticCorpusGenerator(seed=0).generate(n_per_category=5, n_clean=5)
    assert corpus.categories() == {c.value for c in HackCategory}


def test_generator_negative_counts_raise() -> None:
    with pytest.raises(ValueError):
        SyntheticCorpusGenerator().generate(n_per_category=-1)
    with pytest.raises(ValueError):
        SyntheticCorpusGenerator().generate(n_clean=-1)


def test_generator_hack_reward_signature() -> None:
    corpus = SyntheticCorpusGenerator(seed=0).generate(n_per_category=10, n_clean=10)
    hacks = [t for t in corpus.trajectories if t.is_hack]
    # Every hack has proxy strictly above true reward.
    assert all(t.proxy_reward > t.true_reward for t in hacks)


def test_generator_clean_reward_signature() -> None:
    corpus = SyntheticCorpusGenerator(seed=0).generate(n_per_category=10, n_clean=50)
    clean = [t for t in corpus.trajectories if not t.is_hack]
    # Proxy and true agree closely for clean trajectories.
    assert all(abs(t.proxy_reward - t.true_reward) <= 0.15 for t in clean)


def test_generator_length_inflation_is_long() -> None:
    corpus = SyntheticCorpusGenerator(seed=0).generate(n_per_category=5, n_clean=0)
    li = corpus.by_category()["length_inflation"]
    assert all(len(t.response) > 1500 for t in li)


def test_generator_format_exploit_is_short_caps() -> None:
    corpus = SyntheticCorpusGenerator(seed=0).generate(n_per_category=5, n_clean=0)
    fe = corpus.by_category()["format_exploit"]
    assert all(t.response.isupper() and len(t.response) < 10 for t in fe)


def test_generator_test_tampering_has_markers() -> None:
    corpus = SyntheticCorpusGenerator(seed=0).generate(n_per_category=10, n_clean=0)
    tt = corpus.by_category()["test_tampering"]
    markers = ("sys.exit(0)", "pytest.skip", "unittest.skip", "assert True", "monkeypatch")
    assert all(any(m in t.response for m in markers) for t in tt)


def test_generator_verifier_gaming_metadata() -> None:
    corpus = SyntheticCorpusGenerator(seed=0).generate(n_per_category=5, n_clean=0)
    vg = corpus.by_category()["verifier_gaming"]
    for t in vg:
        assert t.metadata["verification"]["metadata"]["suspected_hacking"] is True


def test_generator_degenerate_repetition() -> None:
    corpus = SyntheticCorpusGenerator(seed=0).generate(n_per_category=5, n_clean=0)
    dr = corpus.by_category()["degenerate_repetition"]
    for t in dr:
        tokens = t.response.split()
        assert len(tokens) > len(set(tokens))  # heavy repetition


def test_generator_sycophancy_flatters() -> None:
    corpus = SyntheticCorpusGenerator(seed=0).generate(n_per_category=5, n_clean=0)
    syc = corpus.by_category()["sycophancy"]
    assert all(
        any(w in t.response for w in ("right", "excellent", "Brilliant", "agree"))
        for t in syc
    )


def test_generator_answer_leakage_marker() -> None:
    corpus = SyntheticCorpusGenerator(seed=0).generate(n_per_category=5, n_clean=0)
    al = corpus.by_category()["answer_leakage"]
    assert all("####" in t.response for t in al)


# ---------------------------------------------------------------------------
# LabeledCorpus split + IO
# ---------------------------------------------------------------------------


def test_split_by_hack_type_held_out_only_in_test() -> None:
    corpus = SyntheticCorpusGenerator(seed=0).generate(n_per_category=5, n_clean=10)
    train, test = corpus.split_by_hack_type(held_out=["sycophancy", "answer_leakage"])
    assert "sycophancy" not in train.categories()
    assert "answer_leakage" not in train.categories()
    assert test.categories() == {"sycophancy", "answer_leakage"}


def test_split_by_hack_type_clean_in_both() -> None:
    corpus = SyntheticCorpusGenerator(seed=0).generate(n_per_category=5, n_clean=10)
    train, test = corpus.split_by_hack_type(held_out=["length_inflation"])
    assert train.clean_count == 10
    assert test.clean_count == 10


def test_split_by_hack_type_partitions_hacks() -> None:
    corpus = SyntheticCorpusGenerator(seed=0).generate(n_per_category=5, n_clean=10)
    train, test = corpus.split_by_hack_type(held_out=["length_inflation"])
    assert train.hack_count + test.hack_count == corpus.hack_count


def test_split_by_hack_type_unknown_category_raises() -> None:
    corpus = SyntheticCorpusGenerator(seed=0).generate(n_per_category=2, n_clean=2)
    with pytest.raises(ValueError):
        corpus.split_by_hack_type(held_out=["nope"])


def test_corpus_by_category_groups() -> None:
    corpus = SyntheticCorpusGenerator(seed=0).generate(n_per_category=4, n_clean=3)
    grouped = corpus.by_category()
    assert set(grouped) == {c.value for c in HackCategory}
    assert all(len(v) == 4 for v in grouped.values())


def test_jsonl_roundtrip(tmp_path: Path) -> None:
    corpus = SyntheticCorpusGenerator(seed=2).generate(n_per_category=6, n_clean=8)
    path = tmp_path / "corpus.jsonl"
    corpus.to_jsonl(path)
    loaded = LabeledCorpus.from_jsonl(path)
    assert [t.to_dict() for t in loaded.trajectories] == [
        t.to_dict() for t in corpus.trajectories
    ]


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def test_auroc_perfect_separation() -> None:
    assert auroc([0.9, 0.8, 0.2, 0.1], [True, True, False, False]) == 1.0


def test_auroc_inverted() -> None:
    assert auroc([0.1, 0.2, 0.8, 0.9], [True, True, False, False]) == 0.0


def test_auroc_random_ish() -> None:
    # Interleaved ranks -> chance-level discrimination.
    scores = [0.6, 0.5, 0.4, 0.55, 0.3, 0.45]
    labels = [True, False, True, False, True, False]
    assert 0.3 <= auroc(scores, labels) <= 0.7


def test_auroc_all_one_class() -> None:
    assert auroc([0.9, 0.8], [True, True]) == 0.5
    assert auroc([0.9, 0.8], [False, False]) == 0.5


def test_auroc_ties() -> None:
    # All-equal scores -> chance level regardless of labels.
    assert auroc([0.5, 0.5, 0.5, 0.5], [True, True, False, False]) == 0.5


def test_auroc_length_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        auroc([0.1, 0.2], [True])


def test_precision_recall_f1_basic() -> None:
    p, r, f = precision_recall_f1(tp=8, fp=2, fn=2)
    assert round(p, 3) == 0.8
    assert round(r, 3) == 0.8
    assert round(f, 3) == 0.8


def test_precision_recall_f1_degenerate() -> None:
    assert precision_recall_f1(0, 0, 0) == (0.0, 0.0, 0.0)
    assert precision_recall_f1(0, 5, 5) == (0.0, 0.0, 0.0)


def test_precision_recall_f1_perfect() -> None:
    assert precision_recall_f1(10, 0, 0) == (1.0, 1.0, 1.0)


# ---------------------------------------------------------------------------
# Detection result / protocol
# ---------------------------------------------------------------------------


def test_detection_result_score_validation() -> None:
    with pytest.raises(ValueError):
        DetectionResult(is_hack=True, score=1.5, category=None)


def test_detectors_satisfy_protocol() -> None:
    detectors = [
        ProvenirDetector(),
        LengthHeuristicDetector(),
        ProxyDivergenceDetector(),
        RandomDetector(),
        AlwaysHackDetector(),
        NeverHackDetector(),
    ]
    assert all(isinstance(d, Detector) for d in detectors)


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------


def _sample_hack() -> Trajectory:
    return Trajectory("h", "p", "A" * 3000, 0.95, 0.05, True, "length_inflation")


def test_all_baselines_return_valid_results() -> None:
    traj = _sample_hack()
    for det in (
        ProvenirDetector(),
        LengthHeuristicDetector(),
        ProxyDivergenceDetector(),
        RandomDetector(),
        AlwaysHackDetector(),
        NeverHackDetector(),
    ):
        result = det.predict(traj)
        assert isinstance(result, DetectionResult)
        assert 0.0 <= result.score <= 1.0


def test_provenir_detector_flags_length_inflation() -> None:
    result = ProvenirDetector().predict(_sample_hack())
    assert result.is_hack


def test_length_detector_flags_long_response() -> None:
    det = LengthHeuristicDetector(max_length=1500)
    assert det.predict(_sample_hack()).is_hack
    short = Trajectory("s", "p", "ok", 0.5, 0.5, False, None)
    assert not det.predict(short).is_hack


def test_proxy_divergence_detector() -> None:
    det = ProxyDivergenceDetector(gap=0.3)
    assert det.predict(_sample_hack()).is_hack
    close = Trajectory("c", "p", "r", 0.6, 0.55, False, None)
    assert not det.predict(close).is_hack


def test_random_detector_deterministic() -> None:
    corpus = SyntheticCorpusGenerator(seed=0).generate(n_per_category=3, n_clean=3)
    a = [RandomDetector(seed=0).predict(t).is_hack for t in corpus.trajectories]
    b = [RandomDetector(seed=0).predict(t).is_hack for t in corpus.trajectories]
    assert a == b


def test_always_hack_recall_one() -> None:
    corpus = SyntheticCorpusGenerator(seed=0).generate(n_per_category=5, n_clean=10)
    ev = BenchmarkHarness().evaluate(AlwaysHackDetector(), corpus)
    assert ev.recall == 1.0


def test_never_hack_recall_zero() -> None:
    corpus = SyntheticCorpusGenerator(seed=0).generate(n_per_category=5, n_clean=10)
    ev = BenchmarkHarness().evaluate(NeverHackDetector(), corpus)
    assert ev.recall == 0.0


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


def test_evaluate_sane_metrics() -> None:
    corpus = SyntheticCorpusGenerator(seed=0).generate()
    ev = BenchmarkHarness().evaluate(ProvenirDetector(), corpus)
    assert 0.0 <= ev.precision <= 1.0
    assert 0.0 <= ev.recall <= 1.0
    assert 0.0 <= ev.auroc <= 1.0
    assert ev.tp + ev.fp + ev.tn + ev.fn == corpus.size


def test_evaluate_per_category_present() -> None:
    corpus = SyntheticCorpusGenerator(seed=0).generate(n_per_category=10, n_clean=20)
    ev = BenchmarkHarness().evaluate(ProvenirDetector(), corpus)
    cats = {c.category for c in ev.per_category}
    assert cats == {c.value for c in HackCategory}
    assert all(isinstance(c, CategoryMetrics) for c in ev.per_category)
    assert all(c.support == 10 for c in ev.per_category)


def test_provenir_beats_random_auroc() -> None:
    corpus = SyntheticCorpusGenerator(seed=0).generate()
    harness = BenchmarkHarness()
    prov = harness.evaluate(ProvenirDetector(), corpus)
    rand = harness.evaluate(RandomDetector(seed=0), corpus)
    assert prov.auroc > rand.auroc


def test_evaluate_held_out_runs_on_unseen() -> None:
    corpus = SyntheticCorpusGenerator(seed=0).generate(n_per_category=10, n_clean=20)
    ev = BenchmarkHarness().evaluate_held_out(
        ProvenirDetector(), corpus, held_out_categories=["proxy_divergence"]
    )
    cats = {c.category for c in ev.per_category}
    assert cats == {"proxy_divergence"}
    assert 0.0 <= ev.auroc <= 1.0


def test_detector_evaluation_to_dict_and_markdown() -> None:
    corpus = SyntheticCorpusGenerator(seed=0).generate(n_per_category=5, n_clean=5)
    ev = BenchmarkHarness().evaluate(ProvenirDetector(), corpus)
    d = ev.to_dict()
    assert d["detector_name"] == "provenir"
    assert "per_category" in d
    md = ev.to_markdown()
    assert "RH-Bench results" in md
    assert "overall" in md


def test_detector_evaluation_validation() -> None:
    with pytest.raises(ValueError):
        DetectorEvaluation(
            detector_name="x",
            precision=1.5,
            recall=0.5,
            f1=0.5,
            auroc=0.5,
            accuracy=0.5,
            tp=1,
            fp=0,
            tn=0,
            fn=0,
            per_category=[],
        )


# ---------------------------------------------------------------------------
# Compute savings
# ---------------------------------------------------------------------------


def _run_stream() -> list[Trajectory]:
    clean = [Trajectory(f"c{i}", "p", "fine", 0.6, 0.6, False, None) for i in range(20)]
    hacks = [
        Trajectory(f"h{i}", "p", "A" * 3000, 0.95, 0.05, True, "length_inflation")
        for i in range(80)
    ]
    return clean + hacks


def test_compute_savings_detection_at_onset() -> None:
    run = _run_stream()
    res = compute_savings(ProvenirDetector(), run, hack_onset_step=20, total_steps=100)
    assert res.detection_step == 20
    assert res.steps_saved == 80
    assert res.fraction_saved == 0.8


def test_compute_savings_never_detected() -> None:
    run = _run_stream()
    res = compute_savings(NeverHackDetector(), run, hack_onset_step=20, total_steps=100)
    assert res.detection_step is None
    assert res.steps_saved == 0
    assert res.fraction_saved == 0.0


def test_compute_savings_fraction_in_range() -> None:
    run = _run_stream()
    res = compute_savings(ProvenirDetector(), run, hack_onset_step=20)
    assert 0.0 <= res.fraction_saved <= 1.0
    assert res.total_steps == len(run)


def test_compute_savings_default_total_steps() -> None:
    run = _run_stream()
    res = compute_savings(AlwaysHackDetector(), run, hack_onset_step=10)
    assert res.total_steps == 100
    assert res.detection_step == 10


def test_compute_saved_result_validation() -> None:
    with pytest.raises(ValueError):
        ComputeSavedResult(
            total_steps=100,
            hack_onset_step=20,
            detection_step=10,  # before onset
            steps_saved=90,
            fraction_saved=0.9,
        )


def test_compute_savings_bad_onset_raises() -> None:
    run = _run_stream()
    with pytest.raises(ValueError):
        compute_savings(ProvenirDetector(), run, hack_onset_step=999, total_steps=100)
