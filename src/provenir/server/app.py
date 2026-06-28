from pathlib import Path
from typing import Any

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel as _BaseModel

    _HAS_FASTAPI = True
except ImportError:
    _HAS_FASTAPI = False

from provenir.core.config import RunConfig
from provenir.core.manifest import RunManifestStore
from provenir.data.dataset import JsonlDataset
from provenir.eval.harness import MultiMetricEvaluator
from provenir.governance.audit import AuditLogger
from provenir.train.backends.stub import StubBackend
from provenir.train.trainer import Trainer


def _require_fastapi() -> None:
    if not _HAS_FASTAPI:
        raise ImportError(
            "FastAPI required for the Provenir server: pip install provenir[serve]"
        )


def create_app(
    manifest_dir: str = "artifacts/manifests",
    adapter_dir: str = "artifacts/adapters",
    audit_dir: str = "artifacts",
) -> Any:
    """Create and return the FastAPI application.

    Raises :class:`ImportError` when FastAPI is not installed.
    """
    _require_fastapi()

    app = FastAPI(
        title="Provenir API",
        description=(
            "Reproducible, evaluation-first fine-tuning orchestration. "
            "Submit training jobs, retrieve manifests, run evaluations, "
            "and audit every run via REST."
        ),
        version="0.1.0",
    )

    manifest_store = RunManifestStore(root_dir=manifest_dir)
    audit_logger = AuditLogger(log_dir=audit_dir)

    # ------------------------------------------------------------------
    # Request / response schemas
    # ------------------------------------------------------------------

    class TrainRequest(_BaseModel):
        config: dict[str, Any]
        records: list[dict[str, Any]]

    class EvalRequest(_BaseModel):
        records: list[dict[str, Any]]
        predictions: list[str]

    # ------------------------------------------------------------------
    # Routes
    # ------------------------------------------------------------------

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "version": "0.1.0"}

    @app.post("/jobs/train")
    async def train(req: TrainRequest) -> dict[str, Any]:
        """Submit a training job.  Returns the run manifest."""
        run_config = RunConfig(**req.config)
        dataset = JsonlDataset.from_records(req.records)
        backend = StubBackend()
        trainer = Trainer(backend=backend, config=run_config)
        manifest = trainer.run(dataset)
        audit_logger.log(
            "api_train",
            "api",
            payload={"run_id": manifest.run_id, "config": req.config},
        )
        return {
            "run_id": manifest.run_id,
            "config_hash": manifest.config_hash,
            "dataset_hash": manifest.dataset_hash,
        }

    @app.get("/manifests/{run_id}")
    async def get_manifest(run_id: str) -> dict[str, Any]:
        """Retrieve a saved run manifest by run ID."""
        try:
            manifest = manifest_store.load(run_id)
        except (FileNotFoundError, KeyError):
            raise HTTPException(status_code=404, detail=f"Manifest {run_id!r} not found")
        return {
            "run_id": manifest.run_id,
            "config_hash": manifest.config_hash,
            "dataset_hash": manifest.dataset_hash,
            "seed": manifest.seed,
            "provenance": manifest.provenance,
        }

    @app.get("/manifests")
    async def list_manifests() -> dict[str, Any]:
        """List all saved manifest IDs."""
        root = Path(manifest_dir)
        if not root.exists():
            return {"run_ids": []}
        run_ids = [p.stem for p in root.glob("*.json")]
        return {"run_ids": sorted(run_ids)}

    @app.post("/eval")
    async def evaluate(req: EvalRequest) -> dict[str, Any]:
        """Run MultiMetricEvaluator over predictions vs dataset records."""
        dataset = JsonlDataset.from_records(req.records)
        evaluator = MultiMetricEvaluator()
        result = evaluator.evaluate(dataset, req.predictions)
        return result.to_dict()

    @app.get("/adapters")
    async def list_adapters() -> dict[str, Any]:
        """List registered adapters."""
        root = Path(adapter_dir)
        if not root.exists():
            return {"adapters": []}
        adapters = [p.stem for p in root.glob("*.json")]
        return {"adapters": sorted(adapters)}

    @app.get("/audit")
    async def get_audit() -> Any:
        """Return the raw audit log as newline-delimited JSON."""
        audit_path = Path(audit_dir) / "audit.jsonl"
        if not audit_path.exists():
            return JSONResponse(content={"entries": []})
        lines = audit_path.read_text(encoding="utf-8").strip().splitlines()
        return JSONResponse(content={"entries": lines})

    return app


def run_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    manifest_dir: str = "artifacts/manifests",
    adapter_dir: str = "artifacts/adapters",
    audit_dir: str = "artifacts",
) -> None:
    """Start the Provenir REST server with uvicorn."""
    _require_fastapi()
    try:
        import uvicorn
    except ImportError as exc:
        raise ImportError(
            "uvicorn required: pip install provenir[serve]"
        ) from exc
    app = create_app(
        manifest_dir=manifest_dir,
        adapter_dir=adapter_dir,
        audit_dir=audit_dir,
    )
    uvicorn.run(app, host=host, port=port)
