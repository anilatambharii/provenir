"""
Generate a RAG training dataset from raw text documents.

Uses StubJudge (no API key). For production, swap in:
    AnthropicJudge(model="claude-haiku-4-5-20251001")

Usage:
    python examples/rag_dataset.py
"""

from __future__ import annotations

import json
from pathlib import Path

from provenir.data.rag_generator import RAGDataGenerator, RAGGeneratorConfig
from provenir.eval.judge import StubJudge
from provenir.eval.rag_metrics import RAGEvaluator

# ---------------------------------------------------------------------------
# 1. Sample documents (replace with your own)
# ---------------------------------------------------------------------------

DOCUMENTS = [
    """
    Quantum entanglement is a physical phenomenon that occurs when particles
    interact in ways such that the quantum state of each particle cannot be
    described independently of the state of the others, even when separated
    by large distances. Einstein famously called this 'spooky action at a
    distance'. Measurements performed on one entangled particle instantly
    influence the correlated partner, regardless of the distance between them.
    """,
    """
    The transformer architecture, introduced in 'Attention Is All You Need'
    (Vaswani et al., 2017), relies entirely on self-attention mechanisms and
    dispenses with recurrence and convolution. Each token attends to every
    other token in the sequence, with attention weights computed as the
    scaled dot product of query and key vectors. Transformers form the
    backbone of all modern large language models including GPT-4, Claude,
    and Llama.
    """,
    """
    Retrieval-augmented generation (RAG) combines a retrieval system with a
    language model generator. At inference time, relevant documents are
    retrieved from a knowledge base and prepended to the prompt as context.
    This allows the model to produce grounded, factual responses without
    storing all knowledge in its parameters. RAG significantly reduces
    hallucination on knowledge-intensive tasks.
    """,
]

# ---------------------------------------------------------------------------
# 2. Configure the generator
# ---------------------------------------------------------------------------

config = RAGGeneratorConfig(
    questions_per_chunk=3,        # questions generated per passage
    max_chunk_length=400,         # max chars per chunk
    min_chunk_length=50,          # discard very short chunks
    quality_filter_threshold=0.3, # keep pairs with faithfulness > 0.3
)

# StubJudge = deterministic, no API calls.
# Swap to AnthropicJudge for higher-quality Q&A generation.
generator = RAGDataGenerator(config=config, judge=StubJudge())

# ---------------------------------------------------------------------------
# 3. Generate the dataset
# ---------------------------------------------------------------------------

print("Generating RAG dataset...")
dataset = generator.generate_from_documents(DOCUMENTS)

print(f"Generated {len(dataset.records)} Q&A pairs from {len(DOCUMENTS)} documents\n")

# Show sample records
for i, record in enumerate(dataset.records[:3], 1):
    print(f"--- Record {i} ---")
    print(f"Q: {record['prompt']}")
    print(f"A: {record['response']}")
    print(f"Context: {record['context'][:80]}...")
    print()

# ---------------------------------------------------------------------------
# 4. Evaluate quality of generated pairs
# ---------------------------------------------------------------------------

evaluator = RAGEvaluator()
results   = evaluator.evaluate(
    [{"question": r["prompt"], "answer": r["response"], "context": r["context"]}
     for r in dataset.records]
)
summary = evaluator.aggregate(results)

print("RAG Quality Metrics:")
print(f"  Faithfulness:      {summary['faithfulness']:.3f}")
print(f"  Context precision: {summary['context_precision']:.3f}")
print(f"  Answer relevance:  {summary['answer_relevance']:.3f}")

# ---------------------------------------------------------------------------
# 5. Save
# ---------------------------------------------------------------------------

out = Path("artifacts/rag_train.jsonl")
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w") as f:
    for record in dataset.records:
        f.write(json.dumps(record) + "\n")
print(f"\nSaved {len(dataset.records)} records to {out}")
