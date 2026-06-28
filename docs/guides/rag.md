# RAG Dataset Generation

Generate high-quality retrieval-augmented generation training datasets from
raw documents — automatically.

---

## Overview

`RAGDataGenerator` takes a list of raw documents and produces a `JsonlDataset`
of `(question, answer, context)` triples suitable for fine-tuning RAG-capable
models:

```
documents
    │
    ▼
chunk into passages (respects sentence boundaries)
    │
    ▼
generate N questions per chunk (stub or via judge)
    │
    ▼
generate answers grounded in the chunk
    │
    ▼
quality filter (lexical overlap score)
    │
    ▼
JsonlDataset {"prompt": ..., "response": ..., "context": ..., "_rag_meta": ...}
```

---

## Quick Start

```python
from provenir.data.rag_generator import RAGDataGenerator, RAGGeneratorConfig
from provenir.eval.judge import AnthropicJudge
from pathlib import Path

generator = RAGDataGenerator(
    config=RAGGeneratorConfig(
        questions_per_chunk=5,
        max_chunk_length=512,
        quality_filter_threshold=0.6,
    ),
    judge=AnthropicJudge(),
)

docs = [open(p).read() for p in Path("raw_docs/").glob("*.txt")]
dataset = generator.generate_from_documents(docs)
dataset.save("data/rag_train.jsonl")

print(f"Generated {len(dataset.records)} Q&A pairs from {len(docs)} documents")
```

---

## Configuration

### RAGGeneratorConfig

| Field | Default | Description |
|---|---|---|
| `questions_per_chunk` | `3` | Number of questions to generate per passage |
| `max_chunk_length` | `512` | Maximum characters per chunk |
| `min_chunk_length` | `50` | Minimum characters — shorter chunks are discarded |
| `quality_filter_threshold` | `0.5` | Quality score threshold (0.0–1.0) |

The quality filter uses `FaithfulnessMetric` to score how well each answer is
grounded in the source chunk. Pairs below the threshold are discarded.

---

## Output Format

Each record in the generated dataset contains:

```json
{
  "prompt": "What did Einstein call quantum entanglement?",
  "response": "Einstein called quantum entanglement 'spooky action at a distance'.",
  "context": "Quantum entanglement is a physical phenomenon. Einstein called this 'spooky action at a distance'.",
  "_rag_meta": {
    "source_chunk": "Quantum entanglement is a physical phenomenon...",
    "quality_score": 0.82
  }
}
```

The `_rag_meta` field carries provenance information and can be used for
downstream filtering or analysis.

---

## Step-by-Step API

```python
generator = RAGDataGenerator(config=RAGGeneratorConfig())

# 1. Chunk a document into passages
chunks = generator.chunk_document(long_document)

# 2. Generate Q&A pairs from chunks
pairs = generator.generate_qa_pairs(chunks)

# 3. Wrap pairs as a JsonlDataset
dataset = generator.to_dataset(pairs)
```

---

## Evaluating RAG Datasets

After generating the dataset, evaluate it with `RAGEvaluator`:

```python
from provenir.eval.rag_metrics import RAGEvaluator

evaluator = RAGEvaluator()
results = evaluator.evaluate(dataset.records)
summary = evaluator.aggregate(results)

print(f"Faithfulness:      {summary['faithfulness']:.3f}")
print(f"Context precision: {summary['context_precision']:.3f}")
print(f"Answer relevance:  {summary['answer_relevance']:.3f}")
```

---

## Offline Mode

Without a judge, `RAGDataGenerator` uses deterministic templates — useful for
testing and CI:

```python
generator = RAGDataGenerator(config=RAGGeneratorConfig())  # no judge
dataset = generator.generate_from_documents(docs)
```

For CI with a real judge interface but no API calls, use `StubJudge`:

```python
from provenir.eval.judge import StubJudge
generator = RAGDataGenerator(config=RAGGeneratorConfig(), judge=StubJudge())
```
