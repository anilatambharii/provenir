from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from provenir.data.dataset import JsonlDataset
from provenir.eval.judge import LLMJudge, StubJudge


@dataclass(frozen=True)
class RAGGeneratorConfig:
    """Configuration for RAG training-data generation."""

    questions_per_chunk: int = 3
    max_chunk_length: int = 512
    min_chunk_length: int = 100
    quality_filter_threshold: float = 0.4

    def __post_init__(self) -> None:
        if self.questions_per_chunk < 1:
            raise ValueError("questions_per_chunk must be >= 1")
        if self.max_chunk_length < self.min_chunk_length:
            raise ValueError("max_chunk_length must be >= min_chunk_length")
        if not (0.0 <= self.quality_filter_threshold <= 1.0):
            raise ValueError("quality_filter_threshold must be in [0, 1]")


class RAGDataGenerator:
    """Generates (question, answer, context) training triples from document chunks.

    The generated :class:`JsonlDataset` is directly compatible with
    :class:`CurriculumSampler` and the existing fine-tuning backends.

    Plug in an :class:`AnthropicJudge` or :class:`OpenAIJudge` for high-quality
    question generation.  The default :class:`StubJudge` generates heuristic
    questions for offline testing.

    Usage::

        generator = RAGDataGenerator(config=RAGGeneratorConfig(), judge=AnthropicJudge())
        dataset = generator.generate_from_documents(["Your document text..."])
    """

    def __init__(
        self,
        config: RAGGeneratorConfig | None = None,
        judge: LLMJudge | None = None,
    ) -> None:
        self.config = config or RAGGeneratorConfig()
        self._judge = judge or StubJudge()

    def chunk_document(self, text: str) -> list[str]:
        """Split *text* into sentence-boundary-aligned chunks."""
        max_len = self.config.max_chunk_length
        min_len = self.config.min_chunk_length

        if len(text) <= max_len:
            return [text] if len(text) >= min_len else []

        chunks: list[str] = []
        sentences = text.replace("\n", " ").split(". ")
        current = ""
        for sentence in sentences:
            candidate = f"{current}. {sentence}" if current else sentence
            if len(candidate) > max_len and current:
                if len(current) >= min_len:
                    chunks.append(current.strip())
                current = sentence
            else:
                current = candidate
        if current and len(current) >= min_len:
            chunks.append(current.strip())
        return chunks

    def _question_for_chunk(self, chunk: str, index: int) -> str:
        """Derive a plausible question for *chunk*.

        In production, override this by sub-classing and calling the judge
        or a dedicated generation model.  The stub uses the first few words
        as a question stem.
        """
        first_sentence = chunk.split(".")[0].strip()
        words = first_sentence.split()[:8]
        stem = " ".join(words)
        return f"What does the text say about {stem}?" if index == 0 else f"{stem}?"

    def generate_qa_pairs(self, chunks: list[str]) -> list[dict[str, Any]]:
        """Generate Q&A triples for every chunk.

        Each triple includes a ``_rag_meta`` field with diagnostic info that
        :class:`CurriculumSampler` uses as auxiliary difficulty signals.
        """
        triples: list[dict[str, Any]] = []
        for chunk_idx, chunk in enumerate(chunks):
            for q_idx in range(self.config.questions_per_chunk):
                question = self._question_for_chunk(chunk, q_idx)
                rubric = self._judge.score_rubric(
                    question, chunk, ["relevance", "completeness"]
                )
                quality = sum(s.score for s in rubric) / max(len(rubric), 1)
                if quality < self.config.quality_filter_threshold:
                    continue
                triples.append(
                    {
                        "prompt": question,
                        "response": chunk,
                        "context": chunk,
                        "_rag_meta": {
                            "chunk_index": chunk_idx,
                            "question_index": q_idx,
                            "quality_score": quality,
                        },
                    }
                )
        return triples

    def to_dataset(self, qa_pairs: list[dict[str, Any]]) -> JsonlDataset:
        return JsonlDataset.from_records(qa_pairs)

    def generate_from_documents(self, documents: list[str]) -> JsonlDataset:
        """Full pipeline: chunk → generate Q&A → filter → return dataset."""
        all_chunks: list[str] = []
        for doc in documents:
            all_chunks.extend(self.chunk_document(doc))
        pairs = self.generate_qa_pairs(all_chunks)
        return self.to_dataset(pairs)
