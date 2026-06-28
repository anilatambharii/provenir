from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from provenir.eval.judge import (
    AnthropicJudge,
    CachedJudge,
    LLMJudge,
    OpenAIJudge,
    Preference,
    RubricScore,
    StubJudge,
)


class TestPreference:
    def test_fields(self) -> None:
        p = Preference(preferred="a", confidence=0.9, rationale="better")
        assert p.preferred == "a"
        assert p.confidence == 0.9
        assert p.rationale == "better"

    def test_is_frozen(self) -> None:
        p = Preference(preferred="tie", confidence=0.5, rationale="equal")
        with pytest.raises((AttributeError, TypeError)):
            p.preferred = "b"  # type: ignore[misc]


class TestRubricScore:
    def test_fields(self) -> None:
        s = RubricScore(criterion="accuracy", score=0.8, rationale="good")
        assert s.criterion == "accuracy"
        assert s.score == 0.8

    def test_is_frozen(self) -> None:
        s = RubricScore(criterion="c", score=0.5, rationale="r")
        with pytest.raises((AttributeError, TypeError)):
            s.score = 0.9  # type: ignore[misc]


class TestStubJudge:
    j = StubJudge()

    def test_protocol_compliance(self) -> None:
        assert isinstance(self.j, LLMJudge)

    def test_pairwise_prefers_longer(self) -> None:
        pref = self.j.score_pairwise("q", "short", "a much longer response here")
        assert pref.preferred == "b"

    def test_pairwise_prefers_a_when_longer(self) -> None:
        pref = self.j.score_pairwise("q", "this is a longer answer", "short")
        assert pref.preferred == "a"

    def test_pairwise_tie_on_equal_length(self) -> None:
        pref = self.j.score_pairwise("q", "abc", "xyz")
        assert pref.preferred == "tie"

    def test_pairwise_confidence_in_range(self) -> None:
        pref = self.j.score_pairwise("q", "a", "bb")
        assert 0.0 <= pref.confidence <= 1.0

    def test_rubric_returns_one_per_criterion(self) -> None:
        criteria = ["accuracy", "completeness", "clarity"]
        scores = self.j.score_rubric("prompt", "response", criteria)
        assert len(scores) == 3

    def test_rubric_criterion_names_preserved(self) -> None:
        criteria = ["x", "y"]
        scores = self.j.score_rubric("p", "r", criteria)
        assert [s.criterion for s in scores] == criteria

    def test_rubric_scores_in_range(self) -> None:
        scores = self.j.score_rubric("p", "r", ["a", "b", "c"])
        assert all(0.0 <= s.score <= 1.0 for s in scores)

    def test_rubric_empty_criteria(self) -> None:
        scores = self.j.score_rubric("p", "r", [])
        assert scores == []


class TestCachedJudge:
    def setup_method(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.cache_dir = Path(self._tmpdir.name)
        self.inner = StubJudge()
        self.judge = CachedJudge(self.inner, cache_dir=self.cache_dir)

    def teardown_method(self) -> None:
        self._tmpdir.cleanup()

    def test_pairwise_cache_hit(self) -> None:
        p1 = self.judge.score_pairwise("q", "short", "a longer answer here")
        p2 = self.judge.score_pairwise("q", "short", "a longer answer here")
        assert p1 == p2

    def test_pairwise_cache_writes_file(self) -> None:
        self.judge.score_pairwise("q", "a", "bb")
        cache_files = list(self.cache_dir.glob("pw_*.json"))
        assert len(cache_files) == 1

    def test_pairwise_cache_file_is_valid_json(self) -> None:
        self.judge.score_pairwise("q", "a", "bb")
        cache_files = list(self.cache_dir.glob("pw_*.json"))
        raw = json.loads(cache_files[0].read_text(encoding="utf-8"))
        assert "preferred" in raw

    def test_rubric_cache_hit(self) -> None:
        s1 = self.judge.score_rubric("p", "r", ["a", "b"])
        s2 = self.judge.score_rubric("p", "r", ["a", "b"])
        assert s1 == s2

    def test_rubric_cache_writes_file(self) -> None:
        self.judge.score_rubric("p", "r", ["a"])
        assert list(self.cache_dir.glob("rubric_*.json"))

    def test_different_prompts_different_cache_keys(self) -> None:
        self.judge.score_pairwise("q1", "a", "b")
        self.judge.score_pairwise("q2", "a", "b")
        assert len(list(self.cache_dir.glob("pw_*.json"))) == 2


class TestAnthropicJudgeImportError:
    def test_raises_when_not_installed(self) -> None:
        import provenir.eval.judge as judge_mod

        original = judge_mod._HAS_ANTHROPIC
        judge_mod._HAS_ANTHROPIC = False
        try:
            with pytest.raises(ImportError, match="anthropic"):
                AnthropicJudge()
        finally:
            judge_mod._HAS_ANTHROPIC = original


class TestOpenAIJudgeImportError:
    def test_raises_when_not_installed(self) -> None:
        import provenir.eval.judge as judge_mod

        original = judge_mod._HAS_OPENAI
        judge_mod._HAS_OPENAI = False
        try:
            with pytest.raises(ImportError, match="openai"):
                OpenAIJudge()
        finally:
            judge_mod._HAS_OPENAI = original
