"""Contamination firewall: detect overlap between training and evaluation data.

This module is the counterpart to :mod:`provenir.data.quality`. Where the
decontamination checkers there answer "does my training data contain these
reference strings?", the contamination firewall answers the harder, symmetric
question that dominates eval hygiene in practice: *which* training rows overlap
with *which* eval rows, by how much, and with what method.

It supports three detection methods:

* ``"exact"``     — normalized string equality (fastest, zero false positives).
* ``"ngram"``     — word-level n-gram overlap (the 13-gram standard used by
  most contamination audits); catches near-duplicates and light paraphrase.
* ``"embedding"`` — cosine similarity via ``sentence-transformers`` (optional);
  catches heavy paraphrase and translation. Falls back to ``"ngram"`` when the
  dependency is unavailable.

Example
-------
>>> from provenir.data.dataset import JsonlDataset
>>> train = JsonlDataset.from_records([
...     {"prompt": "What is the capital of France?"},
...     {"prompt": "Explain gradient descent in one sentence."},
... ])
>>> checker = ContaminationChecker(ContaminationConfig(method="exact"))
>>> report = checker.check(train, ["What is the capital of France?"])
>>> report.contamination_rate
0.5
>>> report.is_clean
False
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from provenir.data.dataset import JsonlDataset

try:
    from sentence_transformers import SentenceTransformer

    _HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    _HAS_SENTENCE_TRANSFORMERS = False


_VALID_METHODS = frozenset({"ngram", "embedding", "exact"})


def _normalize(text: str) -> str:
    """Lowercase and collapse whitespace for stable comparison."""
    return " ".join(text.lower().split())


def _word_ngrams(text: str, n: int) -> set[str]:
    """Return the set of word-level n-grams for ``text``.

    When the text has fewer than ``n`` words the whole (normalized) text is
    returned as a single gram so short strings still compare meaningfully.
    """
    words = _normalize(text).split()
    if len(words) < n:
        joined = " ".join(words)
        return {joined} if joined else set()
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def _overlap_ratio(a: set[str], b: set[str]) -> float:
    """Containment overlap ``|A ∩ B| / min(|A|, |B|)``.

    Containment (rather than Jaccard) is used because a short eval item wholly
    embedded in a long training row should score ``1.0`` — that is exactly the
    contamination signal we care about.
    """
    if not a or not b:
        return 0.0
    intersection = len(a & b)
    return intersection / min(len(a), len(b))


@dataclass(frozen=True)
class ContaminationHit:
    """A single detected overlap between a training row and an eval row."""

    train_index: int
    eval_index: int
    score: float
    method: str
    snippet: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "train_index": self.train_index,
            "eval_index": self.eval_index,
            "score": self.score,
            "method": self.method,
            "snippet": self.snippet,
        }


@dataclass(frozen=True)
class ContaminationReport:
    """The result of a contamination scan.

    Attributes
    ----------
    hits:
        Every detected train/eval overlap above threshold.
    train_size, eval_size:
        Sizes of the compared corpora (for rate math and provenance).
    method:
        The method actually used (may differ from the requested method when an
        ``"embedding"`` scan falls back to ``"ngram"``).
    threshold:
        The similarity threshold applied for ``method``.
    """

    hits: list[ContaminationHit]
    train_size: int
    eval_size: int
    method: str
    threshold: float

    @property
    def contaminated_train_indices(self) -> set[int]:
        """Unique training indices implicated in at least one hit."""
        return {hit.train_index for hit in self.hits}

    @property
    def contamination_rate(self) -> float:
        """Fraction of training rows contaminated (unique / train_size)."""
        if self.train_size == 0:
            return 0.0
        return len(self.contaminated_train_indices) / self.train_size

    @property
    def is_clean(self) -> bool:
        """``True`` when no contamination was detected."""
        return not self.hits

    def to_dict(self) -> dict[str, Any]:
        return {
            "hits": [hit.to_dict() for hit in self.hits],
            "train_size": self.train_size,
            "eval_size": self.eval_size,
            "method": self.method,
            "threshold": self.threshold,
            "contamination_rate": self.contamination_rate,
            "contaminated_train_indices": sorted(self.contaminated_train_indices),
            "is_clean": self.is_clean,
        }


@dataclass(frozen=True)
class ContaminationConfig:
    """Configuration for :class:`ContaminationChecker`.

    Parameters
    ----------
    ngram_n:
        Word-level n-gram size. ``13`` is the de-facto standard for
        contamination auditing (large enough to avoid incidental collisions).
    ngram_threshold:
        Containment overlap ratio at/above which an n-gram pair is flagged.
    embedding_threshold:
        Cosine similarity at/above which an embedding pair is flagged.
    method:
        One of ``"ngram"``, ``"embedding"``, ``"exact"``.
    text_key:
        Record key holding the comparable text.
    """

    ngram_n: int = 13
    ngram_threshold: float = 0.8
    embedding_threshold: float = 0.9
    method: str = "ngram"
    text_key: str = "prompt"

    def __post_init__(self) -> None:
        if self.ngram_n < 1:
            raise ValueError(f"ngram_n must be >= 1, got {self.ngram_n}")
        if not 0.0 < self.ngram_threshold <= 1.0:
            raise ValueError(
                f"ngram_threshold must be in (0.0, 1.0], got {self.ngram_threshold}"
            )
        if not 0.0 < self.embedding_threshold <= 1.0:
            raise ValueError(
                f"embedding_threshold must be in (0.0, 1.0], got {self.embedding_threshold}"
            )
        if self.method not in _VALID_METHODS:
            raise ValueError(
                f"method must be one of {sorted(_VALID_METHODS)}, got {self.method!r}"
            )
        if not self.text_key:
            raise ValueError("text_key must be a non-empty string")


class MinHashDeduplicator:
    """MinHash signatures for scalable near-duplicate detection.

    The pairwise n-gram scan in :class:`ContaminationChecker` is ``O(n * m)`` and
    therefore fine for test-scale corpora, but quadratic for millions of rows.
    MinHash approximates Jaccard similarity in linear signature-comparison time
    and is the recommended primitive for large-scale contamination auditing.

    Example
    -------
    >>> dedup = MinHashDeduplicator(num_perm=64, ngram_n=3)
    >>> a = dedup.signature("the quick brown fox jumps")
    >>> b = dedup.signature("the quick brown fox jumps")
    >>> dedup.estimated_jaccard(a, b)
    1.0
    """

    _MERSENNE_PRIME = (1 << 61) - 1
    _MAX_HASH = (1 << 32) - 1

    def __init__(self, num_perm: int = 128, ngram_n: int = 13, seed: int = 1) -> None:
        if num_perm < 1:
            raise ValueError(f"num_perm must be >= 1, got {num_perm}")
        if ngram_n < 1:
            raise ValueError(f"ngram_n must be >= 1, got {ngram_n}")
        self.num_perm = num_perm
        self.ngram_n = ngram_n
        # Derive permutation coefficients deterministically from the seed so
        # signatures are reproducible across processes.
        rng = _DeterministicRng(seed)
        self._a = [rng.next() % (self._MERSENNE_PRIME - 1) + 1 for _ in range(num_perm)]
        self._b = [rng.next() % self._MERSENNE_PRIME for _ in range(num_perm)]

    def signature(self, text: str) -> tuple[int, ...]:
        """Return the MinHash signature (one min per permutation) of ``text``."""
        grams = _word_ngrams(text, self.ngram_n)
        if not grams:
            return tuple([self._MAX_HASH] * self.num_perm)
        hashed = [
            int(hashlib.sha1(gram.encode("utf-8")).hexdigest(), 16) & self._MAX_HASH
            for gram in grams
        ]
        signature: list[int] = []
        for a, b in zip(self._a, self._b):
            signature.append(min((a * h + b) % self._MERSENNE_PRIME for h in hashed))
        return tuple(signature)

    def estimated_jaccard(
        self, sig_a: tuple[int, ...], sig_b: tuple[int, ...]
    ) -> float:
        """Estimate Jaccard similarity from two signatures of equal length."""
        if len(sig_a) != len(sig_b):
            raise ValueError("signatures must have equal length")
        if not sig_a:
            return 0.0
        matches = sum(1 for x, y in zip(sig_a, sig_b) if x == y)
        return matches / len(sig_a)


class _DeterministicRng:
    """Tiny deterministic LCG — avoids depending on ``random`` global state."""

    def __init__(self, seed: int) -> None:
        self._state = (seed & 0xFFFFFFFF) or 1

    def next(self) -> int:
        # Numerical Recipes LCG constants.
        self._state = (1664525 * self._state + 1013904223) & 0xFFFFFFFF
        return self._state


class ContaminationChecker:
    """Detect contamination between a training set and evaluation references.

    Extends the spirit of :class:`provenir.data.quality.DecontaminationChecker`
    with structured, per-pair reporting and multiple detection methods.

    Example
    -------
    >>> from provenir.data.dataset import JsonlDataset
    >>> train = JsonlDataset.from_records([{"prompt": "the sky is very blue today"}])
    >>> cfg = ContaminationConfig(method="ngram", ngram_n=3, ngram_threshold=0.5)
    >>> report = ContaminationChecker(cfg).check(train, ["the sky is very blue today"])
    >>> report.is_clean
    False
    """

    def __init__(self, config: ContaminationConfig | None = None) -> None:
        self.config = config or ContaminationConfig()
        self._model: Any = None

    # -- public API ---------------------------------------------------------

    def check(self, train: JsonlDataset, eval_refs: list[str]) -> ContaminationReport:
        """Scan ``train`` against ``eval_refs`` and return a report.

        For ``method="embedding"`` when ``sentence-transformers`` is not
        installed the scan transparently falls back to ``"ngram"`` and the
        returned :attr:`ContaminationReport.method` reflects the actual method
        used.
        """
        train_texts = [self._record_text(record) for record in train.records]

        method = self.config.method
        if method == "exact":
            hits = self._check_exact(train_texts, eval_refs)
            threshold = 1.0
        elif method == "embedding":
            if _HAS_SENTENCE_TRANSFORMERS:
                hits = self._check_embedding(train_texts, eval_refs)
                threshold = self.config.embedding_threshold
            else:
                # Fallback: ngram, and record the method actually used.
                method = "ngram"
                hits = self._check_ngram(train_texts, eval_refs)
                threshold = self.config.ngram_threshold
        else:  # "ngram"
            hits = self._check_ngram(train_texts, eval_refs)
            threshold = self.config.ngram_threshold

        return ContaminationReport(
            hits=hits,
            train_size=len(train_texts),
            eval_size=len(eval_refs),
            method=method,
            threshold=threshold,
        )

    def check_datasets(
        self, train: JsonlDataset, eval_ds: JsonlDataset
    ) -> ContaminationReport:
        """Convenience wrapper pulling eval text from ``config.text_key``."""
        eval_refs = [self._record_text(record) for record in eval_ds.records]
        return self.check(train, eval_refs)

    def filter_contaminated(
        self, train: JsonlDataset, report: ContaminationReport
    ) -> JsonlDataset:
        """Return a new dataset with every contaminated training row removed."""
        drop = report.contaminated_train_indices
        kept = [
            record for index, record in enumerate(train.records) if index not in drop
        ]
        return JsonlDataset.from_records(kept)

    # -- helpers ------------------------------------------------------------

    def _record_text(self, record: dict[str, Any]) -> str:
        return str(record.get(self.config.text_key, ""))

    @staticmethod
    def _snippet(text: str, limit: int = 120) -> str:
        text = " ".join(text.split())
        return text if len(text) <= limit else text[: limit - 1] + "…"

    def _check_exact(
        self, train_texts: list[str], eval_refs: list[str]
    ) -> list[ContaminationHit]:
        eval_norm = {_normalize(ref): idx for idx, ref in enumerate(eval_refs)}
        hits: list[ContaminationHit] = []
        for t_idx, text in enumerate(train_texts):
            norm = _normalize(text)
            if not norm:
                continue
            e_idx = eval_norm.get(norm)
            if e_idx is not None:
                hits.append(
                    ContaminationHit(
                        train_index=t_idx,
                        eval_index=e_idx,
                        score=1.0,
                        method="exact",
                        snippet=self._snippet(text),
                    )
                )
        return hits

    def _check_ngram(
        self, train_texts: list[str], eval_refs: list[str]
    ) -> list[ContaminationHit]:
        n = self.config.ngram_n
        threshold = self.config.ngram_threshold
        # Precompute n-gram sets once (avoids recomputation in the O(n*m) loop).
        train_grams = [_word_ngrams(text, n) for text in train_texts]
        eval_grams = [_word_ngrams(ref, n) for ref in eval_refs]

        hits: list[ContaminationHit] = []
        for t_idx, t_grams in enumerate(train_grams):
            if not t_grams:
                continue
            for e_idx, e_grams in enumerate(eval_grams):
                if not e_grams:
                    continue
                score = _overlap_ratio(t_grams, e_grams)
                if score >= threshold:
                    hits.append(
                        ContaminationHit(
                            train_index=t_idx,
                            eval_index=e_idx,
                            score=score,
                            method="ngram",
                            snippet=self._snippet(train_texts[t_idx]),
                        )
                    )
        return hits

    def _get_model(self) -> Any:
        if self._model is None:
            self._model = SentenceTransformer("all-MiniLM-L6-v2")
        return self._model

    def _check_embedding(
        self, train_texts: list[str], eval_refs: list[str]
    ) -> list[ContaminationHit]:
        from sentence_transformers import util

        model = self._get_model()
        threshold = self.config.embedding_threshold

        # Only compare non-empty texts; keep index mapping intact.
        train_idx = [i for i, t in enumerate(train_texts) if t.strip()]
        eval_idx = [i for i, r in enumerate(eval_refs) if r.strip()]
        if not train_idx or not eval_idx:
            return []

        train_emb = model.encode(
            [train_texts[i] for i in train_idx], convert_to_tensor=True
        )
        eval_emb = model.encode(
            [eval_refs[i] for i in eval_idx], convert_to_tensor=True
        )
        scores = util.cos_sim(train_emb, eval_emb)

        hits: list[ContaminationHit] = []
        for row, t_idx in enumerate(train_idx):
            for col, e_idx in enumerate(eval_idx):
                score = float(scores[row][col])
                if score >= threshold:
                    hits.append(
                        ContaminationHit(
                            train_index=t_idx,
                            eval_index=e_idx,
                            score=score,
                            method="embedding",
                            snippet=self._snippet(train_texts[t_idx]),
                        )
                    )
        return hits
