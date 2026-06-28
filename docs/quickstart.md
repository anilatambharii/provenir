# Quickstart

Get from zero to a reproducible fine-tuning run in under 5 minutes.

---

## 1. Install

```bash
# Core package (manifests, eval, governance, CLI)
pip install provenir

# Add the training stack
pip install "provenir[train]"

# Or install everything
pip install "provenir[all]"
```

Requires Python ≥ 3.11.

---

## 2. Write a config

```yaml
# my_run.yaml
name: llama3-sft-v1
backend: trl
model_name_or_path: meta-llama/Llama-3.2-1B
seed: 42
max_steps: 500
batch_size: 4

peft:
  rank: 16
  alpha: 32
  target_modules: ["q_proj", "v_proj"]
  dropout: 0.05

observability_backend: wandb
observability_project: my-project
```

---

## 3. Prepare your dataset

Provenir uses JSONL — one record per line, each with `prompt` and `response` keys:

```json
{"prompt": "Summarize the following article:", "response": "The article discusses..."}
{"prompt": "Translate to French:", "response": "Voici la traduction..."}
```

Save it as `data/train.jsonl`.

---

## 4. Train

```bash
provenir train my_run.yaml --dataset data/train.jsonl
```

This will:

- Hash your config and dataset to produce a reproducible `RunManifest`
- Run SFT training via the TRL backend with your LoRA config
- Log metrics to W&B (or whichever backend you configured)
- Save the manifest to `artifacts/manifests/`

The output looks like:

```
[provenir] Run ID: 3f8a1b2c-...
[provenir] Config hash: sha256:aabbcc...
[provenir] Dataset hash: sha256:112233...
[provenir] Training complete. Manifest saved to artifacts/manifests/3f8a1b2c.json
```

---

## 5. Evaluate

```bash
provenir eval predictions.jsonl --dataset data/eval.jsonl --metrics all
```

Output:

```
exact_match:  0.712  [0.683, 0.741]
token_f1:     0.831  [0.809, 0.853]
bleu4:        0.489  [0.461, 0.517]
rouge_l:      0.761  [0.739, 0.783]
```

Numbers in brackets are Wilson 95% confidence intervals.

---

## 6. Reproduce any run

```bash
provenir reproduce artifacts/manifests/3f8a1b2c.json --output reproduced/
```

Provenir verifies the config hash and dataset hash before starting. If anything
has changed, it raises an error.

---

## 7. Run RLAIF (no human labels)

```bash
pip install "provenir[judge-anthropic,train]"

provenir rlaif my_run.yaml \
  --dataset data/train.jsonl \
  --eval-dataset data/eval.jsonl \
  --judge anthropic \
  --iterations 3
```

Each iteration: generate response variants → LLM judge pairwise ranking → DPO training → eval → regression gate → next iteration.

See the [RLAIF guide](guides/rlaif.md) for full configuration options.

---

## 8. Start the REST API

```bash
pip install "provenir[serve]"
provenir serve --host 0.0.0.0 --port 8000
```

```bash
# Submit a training job
curl -X POST http://localhost:8000/jobs/train \
  -H "Content-Type: application/json" \
  -d '{"config": {"name": "api-run", "backend": "stub"}, "records": [...]}'

# Get audit log
curl http://localhost:8000/audit
```

Interactive API docs at `http://localhost:8000/docs`.

---

## Next Steps

- [Core Concepts](concepts.md) — understand manifests, RLAIF, and governance
- [RLAIF Pipeline](guides/rlaif.md) — deep dive into AI-feedback iteration
- [API Reference](api.md) — full Python API documentation
