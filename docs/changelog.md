# Changelog

All notable changes to Provenir are documented here.

---

## v0.2.0 (2025)

Major release — acquisition-grade feature set. Full parity with and significant
extension beyond Axolotl, TRL, and Unsloth in the orchestration and evaluation
layers.

### New Features

**Training**
- TRL backend: SFT, DPO, LoRA, QLoRA via HuggingFace TRL + PEFT
- `PEFTConfig`: LoRA, QLoRA (4-bit, 8-bit), rsLoRA scaling
- `DistributedConfig`: FSDP, DeepSpeed (stages 1/2/3), DDP
- `DPOTrainer`, `GRPOTrainer`, `PPOTrainer` algorithm classes
- `GridSweep`, `RandomSweep` for hyperparameter search
- Training observability: W&B, MLflow, TensorBoard, in-memory
- `EvalCallback`: mid-training evaluation, early stopping, regression gate
- **`RLAIFPipeline`**: AI-feedback iteration loop (judge → DPO → eval → iterate)

**Data**
- `PEFTConfig` and six prompt templates: Alpaca, ChatML, Llama3, Mistral, Phi3, RawCompletion
- `CurriculumSampler`: difficulty-based data ordering
- `SemanticDecontaminationChecker`: embedding-based decontamination (falls back to substring)
- **`RAGDataGenerator`**: document → chunk → Q&A → quality filter → dataset
- **`DataFlywheel`**: mine failures → generate variants → quality filter → augment

**Evaluation**
- RAG metrics: faithfulness, context_precision, answer_relevance
- LLM-as-Judge: `StubJudge`, `CachedJudge`, `AnthropicJudge`, `OpenAIJudge`
- **`BenchmarkEvaluator`**: MMLU, HellaSwag, ARC, TruthfulQA, Winogrande, GSM8K, HumanEval

**Adapters**
- **`ModelMerger`**: SLERP, TIES, DARE adapter merging algorithms
- **`HubClient`**: HuggingFace Hub push/pull with SHA-256 hash verification

**Infrastructure**
- **REST API server** (FastAPI): `/health`, `/jobs/train`, `/manifests`, `/eval`, `/adapters`, `/audit`
- 13 CLI commands: `train`, `eval`, `audit`, `model-card`, `reproduce`, `sweep`, `compare`, `benchmark`, `merge`, `hub push`, `hub pull`, `serve`, `rlaif`
- `CostEstimator` for pre-run budget estimation

**Governance**
- `PIIScanner`, `PIIMasker`: detect and mask PII before training
- `SecretScanner`: detect accidentally included credentials
- `ModelCardGenerator`: HuggingFace-compatible model card generation

### Test Suite
- 456 tests, 30 test modules — all passing
- CI: ruff + mypy (strict) + pytest on Python 3.11 and 3.12

---

## v0.1.0 (2024)

Initial release — reproducibility-first core.

### Features
- `RunManifest`: content-addressed run records (config hash, dataset hash, git SHA)
- `RunConfig`: YAML-based unified training configuration (Pydantic)
- `JsonlDataset`: JSONL ingestion, filtering, and provenance
- `QualityScorer`: lexical quality scoring and filtering
- `DecontaminationChecker`: substring-based train/eval overlap detection
- `DifficultyScorer`: difficulty estimation for curriculum sampling
- `AdapterRegistry`: versioned adapter lineage tracking
- Evaluation: ExactMatch, TokenF1, BLEU-4, ROUGE-L with Wilson 95% CI
- `RegressionGate`: blocks model promotion on quality regression
- Reward primitives: `ExactMatchReward`, `FormatReward`, `WeightedSumReward`, etc.
- `AuditLogger`: append-only JSONL audit trail
- `SecretScanner`: credential detection in datasets
- `CostEstimator`: pre-run token and compute cost estimation
- `PluginRegistry`: protocol-based plugin registration
- CLI: `train`, `eval`, `audit`, `model-card`, `reproduce`, `sweep`, `compare`
- StubBackend: zero-dependency testing backend
- 206 tests, 100% passing
