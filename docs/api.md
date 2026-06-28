# API Reference

Complete Python API for Provenir v0.2.0.

---

## provenir.core.config

### RunConfig

```python
class RunConfig(BaseModel):
    name: str
    backend: str = "stub"               # "trl" | "stub"
    seed: int = 42
    deterministic: bool = True
    output_dir: str = "artifacts"
    model_name_or_path: str | None = None
    max_steps: int = 1000
    batch_size: int = 8
    peft: PEFTConfig | None = None
    distributed: DistributedConfig | None = None
    observability_backend: str = "none" # "wandb" | "mlflow" | "tensorboard" | "none"
    observability_project: str | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> RunConfig: ...
```

### PEFTConfig

```python
class PEFTConfig(BaseModel):
    rank: int = 8
    alpha: float = 16.0
    target_modules: list[str] = ["q_proj", "v_proj"]
    dropout: float = 0.05
    load_in_4bit: bool = False    # QLoRA
    load_in_8bit: bool = False
    use_rslora: bool = False      # rank-stabilised scaling
```

### DistributedConfig

```python
class DistributedConfig(BaseModel):
    strategy: str = "none"        # "fsdp" | "deepspeed" | "ddp" | "none"
    num_gpus: int = 1
    num_nodes: int = 1
    deepspeed_stage: int = 2      # 1 | 2 | 3
```

---

## provenir.core.abstractions

### RunManifest

```python
@dataclass
class RunManifest:
    run_id: str
    config_hash: str
    dataset_hash: str
    git_sha: str | None
    seed: int
    timestamp: str
    provenance: list[str]         # parent run IDs

    def save(self, path: Path) -> None: ...

    @classmethod
    def load(cls, path: Path) -> RunManifest: ...
```

---

## provenir.data.dataset

### JsonlDataset

```python
class JsonlDataset:
    records: list[dict[str, Any]]

    @classmethod
    def from_jsonl(cls, path: str | Path) -> JsonlDataset: ...

    @classmethod
    def from_records(cls, records: list[dict[str, Any]]) -> JsonlDataset: ...

    def save(self, path: str | Path) -> None: ...

    def filter(self, predicate: Callable[[dict], bool]) -> JsonlDataset: ...

    def __len__(self) -> int: ...
```

---

## provenir.data.templates

### TemplateRegistry

```python
class TemplateRegistry:
    def register(self, name: str, template: PromptTemplate) -> None: ...
    def format(self, name: str, record: dict[str, Any]) -> str: ...
    def list_names(self) -> list[str]: ...

TEMPLATE_REGISTRY: TemplateRegistry   # global singleton
```

**Built-in templates:** `alpaca`, `chatml`, `llama3`, `mistral`, `phi3`, `raw_completion`

---

## provenir.data.quality

### QualityScorer

```python
class QualityScorer:
    def score(self, record: dict[str, Any]) -> float: ...
    def filter(self, dataset: JsonlDataset, threshold: float) -> JsonlDataset: ...
```

### SemanticDecontaminationChecker

```python
class SemanticDecontaminationChecker:
    def __init__(self, threshold: float = 0.90): ...
    def check(
        self,
        train_dataset: JsonlDataset,
        eval_references: list[str],
    ) -> list[dict[str, Any]]: ...
    # Returns contaminated records (to be filtered out)
```

Falls back to substring matching when `sentence-transformers` is not installed.

---

## provenir.data.rag_generator

### RAGGeneratorConfig

```python
class RAGGeneratorConfig(BaseModel):
    questions_per_chunk: int = 3
    max_chunk_length: int = 512
    min_chunk_length: int = 50
    quality_filter_threshold: float = 0.5
```

### RAGDataGenerator

```python
class RAGDataGenerator:
    def __init__(
        self,
        config: RAGGeneratorConfig = RAGGeneratorConfig(),
        judge: LLMJudge | None = None,
    ): ...

    def chunk_document(self, text: str) -> list[str]: ...

    def generate_qa_pairs(self, chunks: list[str]) -> list[dict[str, Any]]: ...

    def generate_from_documents(self, documents: list[str]) -> JsonlDataset: ...

    def to_dataset(self, pairs: list[dict[str, Any]]) -> JsonlDataset: ...
```

---

## provenir.data.flywheel

### FlywheelConfig

```python
class FlywheelConfig(BaseModel):
    min_score_threshold: float = 0.7
    max_variants_per_failure: int = 3
    quality_filter_threshold: float = 0.5
```

### DataFlywheel

```python
class DataFlywheel:
    def __init__(
        self,
        config: FlywheelConfig = FlywheelConfig(),
        judge: LLMJudge | None = None,
    ): ...

    def run(
        self,
        train_dataset: JsonlDataset,
        eval_dataset: JsonlDataset,
    ) -> JsonlDataset: ...
```

---

## provenir.eval.harness

### MultiMetricEvaluator

```python
class MultiMetricEvaluator:
    def evaluate(
        self,
        dataset: JsonlDataset,
        predictions: list[str],
    ) -> EvalResult: ...

@dataclass
class EvalResult:
    metrics: dict[str, MetricResult]

@dataclass
class MetricResult:
    mean: float
    ci_lower: float
    ci_upper: float
    n: int
```

---

## provenir.eval.judge

### LLMJudge (Protocol)

```python
class LLMJudge(Protocol):
    def score_pairwise(
        self,
        prompt: str,
        response_a: str,
        response_b: str,
    ) -> Preference: ...

    def score_rubric(
        self,
        prompt: str,
        response: str,
        criteria: list[str],
    ) -> dict[str, float]: ...

@dataclass
class Preference:
    preferred: str          # "a" | "b" | "tie"
    confidence: float       # 0.0–1.0
    rationale: str
```

### Available Judges

```python
from provenir.eval.judge import (
    StubJudge,       # deterministic, no API
    CachedJudge,     # SHA-256 disk cache wrapper
    AnthropicJudge,  # requires provenir[judge-anthropic]
    OpenAIJudge,     # requires provenir[judge-openai]
)

# Wrap any judge with caching
judge = CachedJudge(
    inner=AnthropicJudge(model="claude-haiku-4-5-20251001"),
    cache_dir=".judge_cache",
)
```

---

## provenir.eval.rag_metrics

### RAGEvaluator

```python
class RAGEvaluator:
    def evaluate(
        self,
        triples: list[dict[str, str]],
        # each dict: {"question": ..., "answer": ..., "context": ...}
    ) -> list[RAGMetricSummary]: ...

    def aggregate(
        self,
        summaries: list[RAGMetricSummary],
    ) -> dict[str, float]: ...

@dataclass
class RAGMetricSummary:
    faithfulness: float
    context_precision: float
    answer_relevance: float
    combined: float            # harmonic mean of all three
```

---

## provenir.eval.benchmarks

### BenchmarkEvaluator

```python
class BenchmarkEvaluator:
    def run(
        self,
        model_path: str | Path,
        config: BenchmarkConfig,
    ) -> BenchmarkResult: ...

    def run_suite(
        self,
        model_path: str | Path,
        configs: list[BenchmarkConfig],
    ) -> list[BenchmarkResult]: ...

class BenchmarkConfig(BaseModel):
    benchmark: str              # "mmlu" | "hellaswag" | "arc_easy" | ...
    num_fewshot: int = 0
    limit: int | None = None    # limit number of examples (for testing)

@dataclass
class BenchmarkResult:
    benchmark: str
    score: float
    num_examples: int
    config: BenchmarkConfig
```

Requires `pip install "provenir[benchmarks]"`.

---

## provenir.train.rlaif

### RLAIFConfig

```python
class RLAIFConfig(BaseModel):
    n_iterations: int = 3
    responses_per_prompt: int = 4
    regression_tolerance: float = 0.05
```

### RLAIFPipeline

```python
class RLAIFPipeline:
    def __init__(
        self,
        judge: LLMJudge,
        backend: TrainingBackend,
        base_config: RunConfig,
        rlaif_config: RLAIFConfig = RLAIFConfig(),
    ): ...

    def run(
        self,
        train_dataset: JsonlDataset,
        eval_dataset: JsonlDataset,
    ) -> list[RLAIFIteration]: ...

@dataclass
class RLAIFIteration:
    iteration: int
    preference_count: int
    eval_result: EvalResult
    manifest: RunManifest
    regressed: bool
```

---

## provenir.train.eval_callback

### EvalCallback

```python
class EvalCallbackConfig(BaseModel):
    eval_every_n_steps: int = 100
    early_stopping_patience: int = 3
    regression_tolerance: float = 0.02
    primary_metric: str = "exact_match"

class EvalCallback:
    def __init__(
        self,
        config: EvalCallbackConfig,
        evaluator: MultiMetricEvaluator,
        eval_dataset: JsonlDataset,
    ): ...

    def on_step(self, step: int, model: Any) -> EvalCallbackResult | None: ...

@dataclass
class EvalCallbackResult:
    step: int
    metrics: dict[str, float]
    should_stop: bool
    best_step: int
```

---

## provenir.adapters.merging

### ModelMerger

```python
class MergeConfig(BaseModel):
    strategy: Literal["slerp", "ties", "dare"] = "slerp"
    density: float = 0.5        # for TIES/DARE: fraction of weights to keep
    scale: float = 1.0          # post-merge scale factor

class ModelMerger:
    def merge(
        self,
        adapter_paths: list[Path],
        config: MergeConfig,
        output_dir: Path,
    ) -> MergeResult: ...

@dataclass
class MergeResult:
    output_path: Path
    strategy: str
    adapter_paths: list[Path]
```

Requires `pip install "provenir[merge]"`.

---

## provenir.adapters.hub

### HubClient

```python
class HubConfig(BaseModel):
    repo_id: str               # "username/repo-name"
    private: bool = False
    revision: str = "main"
    token: str | None = None

class HubClient:
    def push_adapter(
        self,
        adapter_path: Path,
        config: HubConfig,
    ) -> HubPushResult: ...

    def pull_model(
        self,
        repo_id: str,
        output_dir: Path | None = None,
        revision: str = "main",
    ) -> Path: ...

    def model_info(self, repo_id: str) -> dict[str, Any]: ...

    def verify_hash(self, file_path: Path, expected_sha256: str) -> bool: ...

@dataclass
class HubPushResult:
    repo_id: str
    url: str
    commit_sha: str | None = None
    pushed_files: list[str] = field(default_factory=list)
```

Requires `pip install "provenir[hub]"`.

---

## provenir.governance.audit

### AuditLogger

```python
class AuditLogger:
    def __init__(self, log_dir: str | Path): ...

    def log(self, event: str, actor: str, **details: Any) -> None: ...

    def read(self) -> list[dict[str, Any]]: ...
```

---

## provenir.governance.pii

### PIIScanner / PIIMasker

```python
class PIIScanResult:
    has_pii: bool
    pii_fields: list[str]
    record_count: int

class PIIScanner:
    def scan(self, dataset: JsonlDataset) -> PIIScanResult: ...

class PIIMasker:
    def mask(self, dataset: JsonlDataset) -> JsonlDataset: ...
```

---

## provenir.governance.scanners

### SecretScanner

```python
class SecretScanner:
    def scan(self, dataset: JsonlDataset) -> SecretScanResult: ...

class SecretScanResult:
    has_secrets: bool
    secret_types: list[str]
```

---

## provenir.rewards.primitives

```python
class ExactMatchReward:
    def score(self, prediction: str, reference: str) -> float: ...

class FormatReward:
    def __init__(self, pattern: str): ...
    def score(self, prediction: str, reference: str) -> float: ...

class WeightedSumReward:
    def __init__(self, rewards: list[tuple[RewardFn, float]]): ...
    def score(self, prediction: str, reference: str) -> float: ...

class ThresholdGatedReward:
    def __init__(self, rewards: list[RewardFn], threshold: float): ...
    def score(self, prediction: str, reference: str) -> float: ...

class ClampedReward:
    def __init__(self, reward: RewardFn, min_val: float, max_val: float): ...
    def score(self, prediction: str, reference: str) -> float: ...
```

---

## provenir.server.app

### REST API (FastAPI)

```python
from provenir.server.app import create_app

app = create_app()  # returns FastAPI instance
```

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness check — `{"status": "ok"}` |
| `POST` | `/jobs/train` | Submit a training job |
| `GET` | `/manifests` | List all manifest run IDs |
| `GET` | `/manifests/{run_id}` | Retrieve a manifest by ID |
| `POST` | `/eval` | Run evaluation |
| `GET` | `/adapters` | List registered adapters |
| `GET` | `/audit` | Full audit log (JSONL) |

Interactive docs: `http://localhost:8000/docs`

Start:

```bash
provenir serve --host 0.0.0.0 --port 8000
# or
uvicorn provenir.server.app:create_app --factory --port 8000
```

---

## provenir.train.observability

### TrainingObserver

```python
class ObservabilityConfig(BaseModel):
    backend: str = "none"           # "wandb" | "mlflow" | "tensorboard" | "none"
    project: str | None = None
    run_name: str | None = None
    log_every_n_steps: int = 10
    tags: tuple[str, ...] = ()

class TrainingObserver:
    def __init__(self, config: ObservabilityConfig): ...

    def __enter__(self) -> TrainingObserver: ...
    def __exit__(self, *args: Any) -> None: ...

    def log_step(self, step: int, metrics: dict[str, float]) -> None: ...
    def log_eval(self, metrics: dict[str, float]) -> None: ...
```
