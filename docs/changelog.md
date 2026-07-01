# Changelog

All notable changes to Provenir are documented here.

---

## v0.3.1 (2026)

Follow-up to the Trust Layer release.

- **Real GRPO reference learner** (`provenir.train.grpo_learner`): a
  self-contained, dependency-free `TabularGRPOLearner` that implements the
  actual GRPO update — group-relative advantages + softmax policy-gradient
  ascent — and provably maximizes a verifiable reward (tested end to end). It is
  the reference that proves the RL loop learns, and it streams metrics to the
  flight recorder.
- **Pluggable update seam**: `PolicyUpdater` protocol + `GRPOUpdater` /
  `NoOpUpdater`. `RLOrchestrator` now accepts an optional `updater` so its
  gradient seam reports genuine advantage/gradient statistics.
- **TRL production path**: `TRLGRPOAdapter` wraps a Provenir `Verifier` as a
  TRL-compatible reward function and delegates the real LLM policy-gradient step
  to `trl.GRPOTrainer` (requires `pip install 'provenir[train]'`). The reward
  function and availability check work without TRL installed.
- **Docs accuracy pass**: the Trust Layer guide's code snippets are now
  copy-paste accurate against the shipped v0.3 API.
- Test suite: 943 tests, all passing.

---

## v0.3.0 — The Trust Layer (2026)

Major release — Provenir becomes **the trust layer for model post-training**.
It orchestrates the best RL engines (verl, TRL, Unsloth) instead of
reimplementing kernels, and wraps every run with RL observability, reward
verification, contamination-safe evaluation, deterministic replay, and a
signed Model Passport. Zero breaking changes.

### Pillar A — RL Flight Recorder + reward-hacking detection

- **`provenir.observability` — RL Flight Recorder**: a "black box" for RL runs.
  Detects KL blowup/collapse, entropy collapse, response-length explosion, GRPO
  advantage collapse, reward-std collapse, reward spikes, and gradient
  explosion. RL-native observability that verl / TRL / OpenRLHF do not ship.
- **`provenir.observability` — RewardHackingDetector**: catches the #1 RL
  failure mode — length inflation, format exploits, test tampering
  (`unittest.skip` / `sys.exit(0)` / monkeypatch), verifier gaming, proxy-reward
  divergence, degenerate repetition, and advantage collapse.

### Pillar B — Contamination-safe trustworthy eval + judge calibration

- **`provenir.eval.contamination`**: contamination firewall with 13-gram,
  embedding, and exact train/eval overlap detection, plus MinHash for scale.
- **`provenir.eval.canary`**: canary-tagged private eval vaults that detect if
  a held-out set leaks into training.
- **`provenir.eval.judge_calibration`**: measures LLM-judge position bias,
  self-consistency, and flip-rate. Adds `DebiasedJudge` (evaluates both
  orderings) and `EnsembleJudge` (majority vote).

### Pillar C — Verifiable-reward environments + GRPO/DAPO/GSPO orchestration

- **`provenir.environments`**: sandboxed, hack-resistant reward functions for
  RLVR behind an OpenEnv-compatible `Environment` protocol — `ExactAnswerVerifier`,
  `MathVerifier`, `RegexFormatVerifier`, `JSONSchemaVerifier`, `ToolCallVerifier`,
  `ContainsVerifier`, `CompositeVerifier`, and a `CodeVerifier` with a
  `PythonSandbox` (subprocess isolation + reward-hacking detection).
- **`provenir.train.rl`**: GRPO + DAPO (decoupled clip + dynamic sampling,
  ByteDance) + GSPO (sequence-level, stabilizes MoE, Qwen) configs, plus a real
  `RLOrchestrator` loop (rollout → verify → reward → flight recorder → hacking
  detector → eval gate; gradient step delegates to a backend).
- **`provenir.train.backends.adapters`**: backend-agnostic adapters wrapping
  verl / TRL / Unsloth with capability detection + a `BackendSelector` that
  auto-routes by scale tier.
- **`provenir.train.rl_eval_gate`**: fuses contamination-safety + regression +
  reward-hacking into one loop guard that halts a run before it wastes GPU
  budget.

### Pillar D — Deterministic replay + lineage DAG + signed Model Passport

- **`provenir.provenance`**: content-addressed environment fingerprint,
  kernel-determinism flags, a lineage DAG (dataset → run → adapter → eval →
  merge), and a `ReplayEngine`. Maps to EU AI Act Article 12 (tamper-proof
  audit trails + model lineage, enforced Aug 2, 2026).
- **`provenir.governance.bom`**: a portable Bill-of-Materials of what data +
  code + evals + config produced a model.
- **`provenir.governance.passport`**: a signed (HMAC-SHA256) Model Passport
  over the BOM with compliance risk flags (`unscanned_pii`,
  `contaminated_eval`, `unknown_license`).

### `import provenir` wrapper

- **`provenir.integrations`**: the viral 3-line substrate — drop provenance +
  trustworthy eval + reward-hacking detection + a signed passport into ANY
  training run via `with provenir.track(...) as run:`. Exposes `run.manifest`,
  `run.flight_recorder`, `run.hacking_report`, and `run.passport`.

### New CLI commands

- `provenir rl <config.yaml>` — verifiable-reward RL with the flight recorder
  (`--algorithm grpo|dapo|gspo`, `--verifier exact_answer|math|contains`).
- `provenir contamination <train.jsonl> <eval.jsonl>` — train/eval overlap check.
- `provenir passport show|verify <passport.json>` — inspect / verify a signed
  Model Passport.

### Test Suite

- 909 tests — all passing.
- CI: ruff + mypy (strict) + pytest on Python 3.11 and 3.12.

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
