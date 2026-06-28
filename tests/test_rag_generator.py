from __future__ import annotations

import pytest

from provenir.data.dataset import JsonlDataset
from provenir.data.rag_generator import RAGDataGenerator, RAGGeneratorConfig
from provenir.eval.judge import StubJudge

_DOCUMENT = (
    "Quantum entanglement is a physical phenomenon. "
    "It occurs when particles interact so that the quantum state of each particle "
    "cannot be described independently of the state of the others. "
    "Einstein called this 'spooky action at a distance'."
)


class TestRAGGeneratorConfig:
    def test_defaults(self) -> None:
        cfg = RAGGeneratorConfig()
        assert cfg.questions_per_chunk == 3
        assert cfg.max_chunk_length == 512

    def test_zero_questions_raises(self) -> None:
        with pytest.raises(ValueError, match="questions_per_chunk"):
            RAGGeneratorConfig(questions_per_chunk=0)

    def test_max_less_than_min_raises(self) -> None:
        with pytest.raises(ValueError, match="max_chunk_length"):
            RAGGeneratorConfig(max_chunk_length=50, min_chunk_length=100)

    def test_invalid_quality_threshold_raises(self) -> None:
        with pytest.raises(ValueError, match="quality_filter_threshold"):
            RAGGeneratorConfig(quality_filter_threshold=1.5)

    def test_is_frozen(self) -> None:
        cfg = RAGGeneratorConfig()
        with pytest.raises((AttributeError, TypeError)):
            cfg.questions_per_chunk = 10  # type: ignore[misc]


class TestRAGDataGeneratorChunking:
    gen = RAGDataGenerator(config=RAGGeneratorConfig(max_chunk_length=100, min_chunk_length=10))

    def test_short_document_one_chunk(self) -> None:
        chunks = self.gen.chunk_document("This is a short document.")
        assert len(chunks) == 1

    def test_empty_document_no_chunks(self) -> None:
        chunks = self.gen.chunk_document("")
        assert chunks == []

    def test_long_document_multiple_chunks(self) -> None:
        long_doc = ". ".join(["This is sentence number " + str(i) for i in range(20)])
        chunks = self.gen.chunk_document(long_doc)
        assert len(chunks) > 1

    def test_chunks_respect_max_length(self) -> None:
        # Use sentence-boundary-aligned text so the chunker can split it.
        sentences = [f"Sentence {i} has some content here." for i in range(20)]
        long_doc = " ".join(sentences)
        chunks = self.gen.chunk_document(long_doc)
        for chunk in chunks:
            assert len(chunk) <= self.gen.config.max_chunk_length

    def test_very_short_text_below_min_filtered(self) -> None:
        chunks = self.gen.chunk_document("Hi.")
        # "Hi." is 3 chars < min_chunk_length=10
        assert chunks == []


class TestRAGDataGeneratorQAPairs:
    gen = RAGDataGenerator(
        config=RAGGeneratorConfig(
            questions_per_chunk=2,
            quality_filter_threshold=0.0,  # accept all
        ),
        judge=StubJudge(),
    )

    def test_qa_pairs_for_single_chunk(self) -> None:
        pairs = self.gen.generate_qa_pairs(["A substantive document chunk about physics."])
        assert len(pairs) == 2  # questions_per_chunk=2

    def test_qa_pairs_have_prompt_response_context(self) -> None:
        pairs = self.gen.generate_qa_pairs(["Some text about cats and dogs."])
        for pair in pairs:
            assert "prompt" in pair
            assert "response" in pair
            assert "context" in pair

    def test_qa_pairs_rag_meta_present(self) -> None:
        pairs = self.gen.generate_qa_pairs(["Some text about neural networks."])
        for pair in pairs:
            assert "_rag_meta" in pair

    def test_empty_chunks_empty_pairs(self) -> None:
        assert self.gen.generate_qa_pairs([]) == []


class TestRAGDataGeneratorFullPipeline:
    gen = RAGDataGenerator(
        config=RAGGeneratorConfig(
            questions_per_chunk=1,
            quality_filter_threshold=0.0,
        ),
        judge=StubJudge(),
    )

    def test_generate_from_documents_returns_dataset(self) -> None:
        ds = self.gen.generate_from_documents([_DOCUMENT])
        assert isinstance(ds, JsonlDataset)

    def test_generate_from_documents_non_empty(self) -> None:
        ds = self.gen.generate_from_documents([_DOCUMENT])
        assert len(ds.records) > 0

    def test_to_dataset_wrapper(self) -> None:
        pairs = [{"prompt": "q", "response": "a", "context": "c"}]
        ds = self.gen.to_dataset(pairs)
        assert len(ds.records) == 1

    def test_multiple_documents_combined(self) -> None:
        ds = self.gen.generate_from_documents([_DOCUMENT, _DOCUMENT])
        single_ds = self.gen.generate_from_documents([_DOCUMENT])
        assert len(ds.records) >= len(single_ds.records)
