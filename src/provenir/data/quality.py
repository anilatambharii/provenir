from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from provenir.data.dataset import JsonlDataset

try:
    from sentence_transformers import SentenceTransformer

    _HAS_ST = True
except ImportError:
    _HAS_ST = False


@dataclass
class DataQualityReport:
    score: float
    issues: list[str] = field(default_factory=list)


class QualityScorer:
    """Simple quality scoring for datasets used by Provenir runs."""

    def score(self, dataset: JsonlDataset) -> DataQualityReport:
        if not dataset.records:
            return DataQualityReport(score=0.0, issues=["empty_dataset"])

        issues: list[str] = []
        if len(dataset.records) != len({repr(record) for record in dataset.records}):
            issues.append("duplicate_records")

        score = 1.0
        if issues:
            score = max(0.0, 1.0 - 0.2 * len(issues))
        return DataQualityReport(score=score, issues=issues)


class DecontaminationChecker:
    """Lightweight substring-based decontamination checker."""

    def check(self, dataset: JsonlDataset, references: list[str]) -> list[str]:
        findings: list[str] = []
        for record in dataset.records:
            text = str(record.get("prompt", "")) + " " + str(record.get("response", ""))
            if any(ref.lower() in text.lower() for ref in references):
                findings.append(text)
        return findings


class SemanticDecontaminationChecker:
    """Embedding-based decontamination using cosine similarity.

    Detects near-duplicate contamination that exact substring matching misses.
    Requires ``sentence-transformers``: ``pip install provenir[semantic]``.

    Falls back to substring matching when the package is unavailable.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        threshold: float = 0.9,
    ) -> None:
        self.model_name = model_name
        self.threshold = threshold
        self._model: Any = None

    def _get_model(self) -> Any:
        if self._model is None:
            if not _HAS_ST:
                return None
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def check(
        self,
        dataset: JsonlDataset,
        references: list[str],
    ) -> list[str]:
        """Return dataset texts that are semantically similar to any reference.

        When ``sentence-transformers`` is not installed, falls back to the
        :class:`DecontaminationChecker` substring approach.
        """
        if not references:
            return []

        model = self._get_model()
        if model is None:
            # Fallback to substring matching.
            return DecontaminationChecker().check(dataset, references)

        from sentence_transformers import util

        texts = [
            str(r.get("prompt", "")) + " " + str(r.get("response", ""))
            for r in dataset.records
        ]

        text_embeddings = model.encode(texts, convert_to_tensor=True)
        ref_embeddings = model.encode(references, convert_to_tensor=True)
        cosine_scores = util.cos_sim(text_embeddings, ref_embeddings)
        suspected: list[str] = []
        for i, row in enumerate(cosine_scores):
            if float(row.max()) >= self.threshold:
                suspected.append(texts[i])
        return suspected
