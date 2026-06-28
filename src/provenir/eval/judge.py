from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

try:
    import anthropic

    _HAS_ANTHROPIC = True
except ImportError:
    _HAS_ANTHROPIC = False

try:
    import openai

    _HAS_OPENAI = True
except ImportError:
    _HAS_OPENAI = False


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Preference:
    """Pairwise preference between two model responses."""

    preferred: Literal["a", "b", "tie"]
    confidence: float  # 0-1
    rationale: str


@dataclass(frozen=True)
class RubricScore:
    """Score on a single rubric criterion."""

    criterion: str
    score: float  # 0-1 normalised
    rationale: str


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class LLMJudge(Protocol):
    """Structural type for LLM-based evaluation judges."""

    def score_pairwise(
        self, prompt: str, response_a: str, response_b: str
    ) -> Preference: ...

    def score_rubric(
        self, prompt: str, response: str, criteria: list[str]
    ) -> list[RubricScore]: ...


# ---------------------------------------------------------------------------
# Stub (no external deps)
# ---------------------------------------------------------------------------


class StubJudge:
    """Deterministic stub judge — prefers the longer response.

    Provides a dependency-free implementation of :class:`LLMJudge` for unit
    tests and pipelines that run without API credentials.
    """

    def score_pairwise(
        self, prompt: str, response_a: str, response_b: str
    ) -> Preference:
        la, lb = len(response_a), len(response_b)
        if la > lb:
            return Preference(preferred="a", confidence=0.7, rationale="longer response")
        if lb > la:
            return Preference(preferred="b", confidence=0.7, rationale="longer response")
        return Preference(preferred="tie", confidence=0.5, rationale="equal length")

    def score_rubric(
        self, prompt: str, response: str, criteria: list[str]
    ) -> list[RubricScore]:
        return [
            RubricScore(criterion=c, score=0.5, rationale="stub score")
            for c in criteria
        ]


# ---------------------------------------------------------------------------
# Disk cache wrapper
# ---------------------------------------------------------------------------


class CachedJudge:
    """Wraps any :class:`LLMJudge` with a SHA-256-keyed disk cache.

    Avoids repeated API calls for identical (prompt, response_a, response_b)
    triples.  Cache entries are plain JSON files and survive process restarts.
    """

    def __init__(
        self,
        inner: LLMJudge,
        cache_dir: str | Path = ".provenir_judge_cache",
    ) -> None:
        self._inner = inner
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def _key(self, *parts: str) -> str:
        payload = "\n".join(parts).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:32]

    def score_pairwise(
        self, prompt: str, response_a: str, response_b: str
    ) -> Preference:
        key = self._key(prompt, response_a, response_b)
        cache_path = self._cache_dir / f"pw_{key}.json"
        if cache_path.exists():
            raw = json.loads(cache_path.read_text(encoding="utf-8"))
            return Preference(**raw)
        result = self._inner.score_pairwise(prompt, response_a, response_b)
        cache_path.write_text(
            json.dumps(
                {
                    "preferred": result.preferred,
                    "confidence": result.confidence,
                    "rationale": result.rationale,
                }
            ),
            encoding="utf-8",
        )
        return result

    def score_rubric(
        self, prompt: str, response: str, criteria: list[str]
    ) -> list[RubricScore]:
        key = self._key(prompt, response, *criteria)
        cache_path = self._cache_dir / f"rubric_{key}.json"
        if cache_path.exists():
            raw_list: list[dict[str, Any]] = json.loads(
                cache_path.read_text(encoding="utf-8")
            )
            return [RubricScore(**r) for r in raw_list]
        result = self._inner.score_rubric(prompt, response, criteria)
        cache_path.write_text(
            json.dumps(
                [
                    {
                        "criterion": r.criterion,
                        "score": r.score,
                        "rationale": r.rationale,
                    }
                    for r in result
                ]
            ),
            encoding="utf-8",
        )
        return result


# ---------------------------------------------------------------------------
# Anthropic backend
# ---------------------------------------------------------------------------


class AnthropicJudge:
    """LLM judge backed by Claude.

    Requires ``anthropic`` package: ``pip install provenir[judge-anthropic]``
    """

    def __init__(
        self,
        model: str = "claude-haiku-4-5-20251001",
        api_key: str | None = None,
        max_tokens: int = 256,
    ) -> None:
        if not _HAS_ANTHROPIC:
            raise ImportError(
                "anthropic package required: pip install provenir[judge-anthropic]"
            )
        self._client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens

    def score_pairwise(
        self, prompt: str, response_a: str, response_b: str
    ) -> Preference:

        system = (
            'You are a fair evaluator. Reply with JSON only: '
            '{"preferred": "a"|"b"|"tie", "confidence": 0-1, "rationale": "one sentence"}'
        )
        user_msg = (
            f"Prompt: {prompt}\n\nResponse A: {response_a}\n\n"
            f"Response B: {response_b}\n\nWhich is better?"
        )
        message = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user_msg}],
        )
        raw: dict[str, Any] = json.loads(message.content[0].text)  # type: ignore[union-attr]
        return Preference(
            preferred=raw["preferred"],
            confidence=float(raw["confidence"]),
            rationale=raw["rationale"],
        )

    def score_rubric(
        self, prompt: str, response: str, criteria: list[str]
    ) -> list[RubricScore]:

        criteria_str = "\n".join(f"- {c}" for c in criteria)
        system = (
            "You are a fair evaluator. Reply with JSON array only: "
            '[{"criterion": ..., "score": 0-1, "rationale": "one sentence"}, ...]'
        )
        user_msg = (
            f"Prompt: {prompt}\n\nResponse: {response}\n\nCriteria:\n{criteria_str}"
        )
        message = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens * len(criteria),
            system=system,
            messages=[{"role": "user", "content": user_msg}],
        )
        raw_list: list[dict[str, Any]] = json.loads(message.content[0].text)  # type: ignore[union-attr]
        return [
            RubricScore(
                criterion=r["criterion"],
                score=float(r["score"]),
                rationale=r["rationale"],
            )
            for r in raw_list
        ]


# ---------------------------------------------------------------------------
# OpenAI backend
# ---------------------------------------------------------------------------


class OpenAIJudge:
    """LLM judge backed by GPT-4o.

    Requires ``openai`` package: ``pip install provenir[judge-openai]``
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        max_tokens: int = 256,
    ) -> None:
        if not _HAS_OPENAI:
            raise ImportError(
                "openai package required: pip install provenir[judge-openai]"
            )
        self._client = openai.OpenAI(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens

    def score_pairwise(
        self, prompt: str, response_a: str, response_b: str
    ) -> Preference:
        system = (
            'You are a fair evaluator. Reply with JSON only: '
            '{"preferred": "a"|"b"|"tie", "confidence": 0-1, "rationale": "one sentence"}'
        )
        resp = self._client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": (
                        f"Prompt: {prompt}\n\n"
                        f"Response A: {response_a}\n\n"
                        f"Response B: {response_b}"
                    ),
                },
            ],
        )
        raw: dict[str, Any] = json.loads(resp.choices[0].message.content or "{}")
        return Preference(
            preferred=raw.get("preferred", "tie"),
            confidence=float(raw.get("confidence", 0.5)),
            rationale=raw.get("rationale", ""),
        )

    def score_rubric(
        self, prompt: str, response: str, criteria: list[str]
    ) -> list[RubricScore]:
        criteria_str = "\n".join(f"- {c}" for c in criteria)
        system = (
            "You are a fair evaluator. Reply with JSON array only: "
            '[{"criterion": ..., "score": 0-1, "rationale": "..."}, ...]'
        )
        resp = self._client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens * len(criteria),
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": (
                        f"Prompt: {prompt}\n\nResponse: {response}"
                        f"\n\nCriteria:\n{criteria_str}"
                    ),
                },
            ],
        )
        raw_list: list[dict[str, Any]] = json.loads(
            resp.choices[0].message.content or "[]"
        )
        return [
            RubricScore(
                criterion=r.get("criterion", ""),
                score=float(r.get("score", 0.5)),
                rationale=r.get("rationale", ""),
            )
            for r in raw_list
        ]
