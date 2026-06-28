# Installation

## Requirements

- Python â‰¥ 3.11
- pip â‰¥ 23.0

---

## Install Options

### Core (no GPU required)

```bash
pip install provenir
```

Includes: manifests, evaluation, governance, CLI, REST API framework.
All optional features degrade gracefully to stub implementations.

### Training Stack

```bash
pip install "provenir[train]"
```

Adds: TRL â‰¥ 0.9.0, Transformers â‰¥ 4.40.0, PEFT â‰¥ 0.10.0, Accelerate â‰¥ 0.30.0,
Datasets â‰¥ 2.20.0, PyTorch â‰¥ 2.3.0.

### Feature Groups

| Group | Command | What it adds |
|---|---|---|
| `train` | `pip install "provenir[train]"` | SFT, DPO, LoRA, QLoRA via TRL |
| `distributed` | `pip install "provenir[distributed]"` | FSDP, DeepSpeed multi-GPU |
| `serve` | `pip install "provenir[serve]"` | REST API server (FastAPI + uvicorn) |
| `hub` | `pip install "provenir[hub]"` | HuggingFace Hub push / pull |
| `merge` | `pip install "provenir[merge]"` | SLERP / TIES / DARE adapter merging |
| `semantic` | `pip install "provenir[semantic]"` | Embedding-based decontamination |
| `benchmarks` | `pip install "provenir[benchmarks]"` | MMLU, HellaSwag, ARC, â€¦ |
| `judge-anthropic` | `pip install "provenir[judge-anthropic]"` | LLM-as-judge via Claude |
| `judge-openai` | `pip install "provenir[judge-openai]"` | LLM-as-judge via GPT-4o |
| `all` | `pip install "provenir[all]"` | Everything |

### Multiple Groups

```bash
pip install "provenir[train,serve,judge-anthropic]"
```

---

## Development Install

```bash
git clone https://github.com/anilatambharii/provenir
cd provenir
pip install -e ".[dev,all]"
```

Run the quality gate:

```bash
python -m ruff check .
python -m mypy src
python -m pytest -q
```

All 456 tests pass with just `.[dev]` â€” optional features degrade to stubs
automatically when their packages are absent.

---

## Verifying the Install

```bash
provenir --help
```

```
usage: provenir [-h] {train,eval,audit,model-card,reproduce,sweep,compare,
                       benchmark,merge,hub,serve,rlaif} ...
```

---

## Docker

```dockerfile
FROM python:3.11-slim

RUN pip install "provenir[serve]"

EXPOSE 8000
CMD ["provenir", "serve", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
docker build -t provenir-server .
docker run -p 8000:8000 provenir-server
```

---

## Troubleshooting

**`ImportError: No module named 'torch'`**
Optional â€” install `provenir[train]` if you need the TRL backend.

**`ImportError: No module named 'huggingface_hub'`**
Optional â€” install `provenir[hub]` to enable Hub push/pull.

**`ImportError: No module named 'lm_eval'`**
Optional â€” install `provenir[benchmarks]` to enable benchmark evaluation.

All optional imports are guarded â€” missing packages produce clear error
messages rather than silent failures.

