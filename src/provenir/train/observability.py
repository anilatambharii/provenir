from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from provenir.core.abstractions import RunManifest
from provenir.eval.harness import EvaluationResult

try:
    import wandb

    _HAS_WANDB = True
except ImportError:
    _HAS_WANDB = False

try:
    import mlflow

    _HAS_MLFLOW = True
except ImportError:
    _HAS_MLFLOW = False


@dataclass(frozen=True)
class ObservabilityConfig:
    """Configuration for training observability backends."""

    backend: Literal["wandb", "mlflow", "tensorboard", "none"] = "none"
    project: str = "provenir"
    run_name: str | None = None
    log_every_n_steps: int = 10
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.log_every_n_steps < 1:
            raise ValueError(
                f"log_every_n_steps must be >= 1, got {self.log_every_n_steps}"
            )


class TrainingObserver:
    """Emits training metrics to the configured observability backend.

    Falls back to in-memory collection when the requested backend is not
    installed, so code that wraps ``TrainingObserver`` works in any
    environment — including CI runners without W&B / MLflow credentials.

    Usage::

        observer = TrainingObserver(ObservabilityConfig(backend="wandb"), manifest)
        observer.start()
        for step, batch in enumerate(loader):
            loss = train_step(batch)
            observer.log_step(step, {"train/loss": loss})
        observer.finish()
    """

    def __init__(self, config: ObservabilityConfig, manifest: RunManifest) -> None:
        self.config = config
        self.manifest = manifest
        self._step_log: list[dict[str, Any]] = []
        self._active = False

    def start(self) -> None:
        """Open the observability session."""
        if self.config.backend == "wandb" and _HAS_WANDB:
            wandb.init(
                project=self.config.project,
                name=self.config.run_name or self.manifest.run_id[:8],
                tags=list(self.config.tags),
                config=self.manifest.provenance,
            )
        elif self.config.backend == "mlflow" and _HAS_MLFLOW:
            mlflow.set_experiment(self.config.project)
            mlflow.start_run(run_name=self.config.run_name or self.manifest.run_id[:8])
            for k, v in self.manifest.provenance.items():
                try:
                    mlflow.log_param(k, v)
                except Exception:
                    pass
        self._active = True

    def log_step(self, step: int, metrics: dict[str, float]) -> None:
        """Log per-step training metrics."""
        entry: dict[str, Any] = {"step": step, **metrics}
        self._step_log.append(entry)
        if self.config.backend == "wandb" and _HAS_WANDB and self._active:
            wandb.log(metrics, step=step)
        elif self.config.backend == "mlflow" and _HAS_MLFLOW and self._active:
            mlflow.log_metrics(metrics, step=step)

    def log_eval(self, step: int, result: EvaluationResult) -> None:
        """Log an EvaluationResult as a step event."""
        if result.metrics is None:
            return
        flat: dict[str, float] = {
            f"eval/{k}": v.mean for k, v in result.metrics.items()
        }
        self.log_step(step, flat)

    def finish(self) -> None:
        """Close the observability session."""
        if self.config.backend == "wandb" and _HAS_WANDB and self._active:
            wandb.finish()
        elif self.config.backend == "mlflow" and _HAS_MLFLOW and self._active:
            mlflow.end_run()
        self._active = False

    @property
    def history(self) -> list[dict[str, Any]]:
        """All step entries logged since :meth:`start`, in order."""
        return list(self._step_log)
