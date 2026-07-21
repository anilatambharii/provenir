from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import HTMLResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel

    _HAS_FASTAPI = True
except ImportError:
    _HAS_FASTAPI = False
    BaseModel = object  # type: ignore[assignment,misc]

from provenir.core.config import RunConfig
from provenir.core.manifest import RunManifestStore
from provenir.data.dataset import JsonlDataset
from provenir.eval.harness import MultiMetricEvaluator
from provenir.governance.audit import AuditLogger
from provenir.train.backends.stub import StubBackend
from provenir.train.trainer import Trainer

_STATIC_DIR = Path(__file__).parent / "static"


# ---------------------------------------------------------------------------
# Pydantic request models — must be at module level for Pydantic V2 resolution
# ---------------------------------------------------------------------------

class TrainRequest(BaseModel):
    config: dict[str, Any]
    records: list[dict[str, Any]]


class EvalRequest(BaseModel):
    records: list[dict[str, Any]]
    predictions: list[str]


class ScanRequest(BaseModel):
    path: str


class PassportLoadRequest(BaseModel):
    path: str


class PassportVerifyRequest(BaseModel):
    path: str
    key: str = ""


class GateRequest(BaseModel):
    path: str
    stage: str = "production"
    require_scan: bool = False
    require_no_retraction: bool = False
    require_no_pii: bool = False
    require_no_contamination: bool = False
    require_signed: bool = False
    min_validity: float | None = None


class RAGScanRequest(BaseModel):
    corpus_dir: str
    known_retracted: list[str] = []
    train_hashes: list[str] = []


class RetractionRequest(BaseModel):
    path: str
    known_retracted: list[str] = []


class RegulationRequest(BaseModel):
    path: str
    framework: str = "art53"  # art53 | annex-iv | fda-pccp | nist-rmf


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_fastapi() -> None:
    if not _HAS_FASTAPI:
        raise ImportError(
            "FastAPI required for the Provenir server: pip install provenir[serve]"
        )


def _load_passport(path: str) -> Any:
    """Load a ModelPassport from a JSON file path."""
    from provenir.governance.passport import ModelPassport

    p = Path(path)
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"Passport file not found: {path}")
    try:
        return ModelPassport.from_dict(json.loads(p.read_text(encoding="utf-8")))
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid passport: {exc}") from exc


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(
    manifest_dir: str = "artifacts/manifests",
    adapter_dir: str = "artifacts/adapters",
    audit_dir: str = "artifacts",
) -> Any:
    """Create and return the FastAPI application."""
    _require_fastapi()

    app = FastAPI(
        title="Provenir API",
        description=(
            "Trust layer for model post-training. "
            "Submit training jobs, scan model weights, verify passports, "
            "run promotion gates, and generate regulatory evidence."
        ),
        version="0.8.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    manifest_store = RunManifestStore(root_dir=manifest_dir)
    audit_logger = AuditLogger(log_dir=audit_dir)

    # ------------------------------------------------------------------
    # Dashboard
    # ------------------------------------------------------------------

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def dashboard() -> HTMLResponse:
        index = _STATIC_DIR / "dashboard.html"
        headers = {"Cache-Control": "no-cache, no-store, must-revalidate"}
        if index.exists():
            return HTMLResponse(index.read_text(encoding="utf-8"), headers=headers)
        return HTMLResponse(
            "<h2>Dashboard not built yet.</h2>"
            "<p>Visit <a href='/docs'>/docs</a> for the API explorer.</p>",
            headers=headers,
        )

    if _STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "version": "0.8.0"}

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------

    @app.post("/jobs/train")
    async def train(req: TrainRequest) -> dict[str, Any]:
        """Submit a training job. Returns the run manifest."""
        run_config = RunConfig(**req.config)
        dataset = JsonlDataset.from_records(req.records)
        trainer = Trainer(backend=StubBackend(), config=run_config)
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

    # ------------------------------------------------------------------
    # Manifests
    # ------------------------------------------------------------------

    @app.get("/manifests")
    async def list_manifests() -> dict[str, Any]:
        root = Path(manifest_dir)
        if not root.exists():
            return {"run_ids": []}
        return {"run_ids": sorted(p.stem for p in root.glob("*.json"))}

    @app.get("/manifests/{run_id}")
    async def get_manifest(run_id: str) -> dict[str, Any]:
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

    # ------------------------------------------------------------------
    # Eval
    # ------------------------------------------------------------------

    @app.post("/eval")
    async def evaluate(req: EvalRequest) -> dict[str, Any]:
        dataset = JsonlDataset.from_records(req.records)
        evaluator = MultiMetricEvaluator()
        result = evaluator.evaluate(dataset, req.predictions)
        return result.to_dict()

    # ------------------------------------------------------------------
    # Supply-chain scan
    # ------------------------------------------------------------------

    @app.post("/scan")
    async def scan_model(req: ScanRequest) -> dict[str, Any]:
        """Scan a model directory or file for supply-chain threats."""
        from provenir.governance.scan import ModelScanner

        target = Path(req.path)
        if not target.exists():
            raise HTTPException(status_code=404, detail=f"Path not found: {req.path}")
        scanner = ModelScanner()
        report = scanner.scan(req.path)
        return {
            "target": report.target,
            "scanned_files": report.scanned_files,
            "unsafe": report.unsafe(),
            "finding_count": len(report.findings),
            "findings": [
                {
                    "severity": f.severity.value,
                    "threat": f.threat.value,
                    "path": f.path,
                    "detail": f.detail,
                }
                for f in report.findings
            ],
        }

    # ------------------------------------------------------------------
    # Passport
    # ------------------------------------------------------------------

    @app.post("/passport/load")
    async def passport_load(req: PassportLoadRequest) -> dict[str, Any]:
        passport = _load_passport(req.path)
        bom = passport.bom
        return {
            "model_id": bom.model_id,
            "base_model": bom.base_model,
            "run_id": bom.run_id,
            "bom_hash": bom.content_hash(),
            "dataset_count": len(bom.data),
            "datasets": [d.name for d in bom.data],
            "evals": [{"benchmark": e.benchmark, "score": e.score} for e in bom.evals],
            "risk_flags": list(bom.risk_flags()),
            "has_scan": bom.scan is not None,
            "has_attestation": passport.attestation is not None,
            "has_lineage": bool(bom.parent_passport_hash),
        }

    @app.post("/passport/verify")
    async def passport_verify(req: PassportVerifyRequest) -> dict[str, Any]:
        passport = _load_passport(req.path)
        if not req.key:
            return {"valid": False, "detail": "No signing key provided"}
        valid = passport.verify(req.key.encode("utf-8"))
        return {"valid": valid, "model_id": passport.bom.model_id}

    # ------------------------------------------------------------------
    # Promotion gate
    # ------------------------------------------------------------------

    @app.post("/gate/promote")
    async def gate_promote(req: GateRequest) -> dict[str, Any]:
        from provenir.governance.promotion_gate import PromotionGate

        passport = _load_passport(req.path)
        gate = PromotionGate(
            require_scan=req.require_scan,
            require_no_retraction=req.require_no_retraction,
            require_no_pii=req.require_no_pii,
            require_no_contamination=req.require_no_contamination,
            require_signed=req.require_signed,
            min_validity=req.min_validity,
        )
        result = gate.evaluate(passport, stage=req.stage)
        return {
            "passed": result.passed,
            "model_id": result.model_id,
            "stage": result.stage,
            "summary": result.summary(),
            "checks": [
                {"name": c.name, "passed": c.passed, "detail": c.detail}
                for c in result.checks
            ],
            "failed_checks": result.failed_checks,
        }

    # ------------------------------------------------------------------
    # RAG corpus scan
    # ------------------------------------------------------------------

    @app.post("/rag-corpus/scan")
    async def rag_corpus_scan(req: RAGScanRequest) -> dict[str, Any]:
        from provenir.governance.rag_corpus import RAGCorpusScanner

        corpus = Path(req.corpus_dir)
        if not corpus.exists():
            raise HTTPException(
                status_code=404, detail=f"Corpus directory not found: {req.corpus_dir}"
            )
        scanner = RAGCorpusScanner(
            known_retracted=frozenset(req.known_retracted),
            train_hashes=frozenset(req.train_hashes),
        )
        report = scanner.scan(req.corpus_dir)
        return {
            "corpus_dir": report.corpus_dir,
            "total_docs": report.total_docs,
            "pii_count": report.pii_count,
            "training_overlap_count": report.training_overlap_count,
            "retracted_count": report.retracted_count,
            "risk_level": report.risk_level,
            "documents": [
                {
                    "path": d.path,
                    "size_bytes": d.size_bytes,
                    "pii_found": d.pii_found,
                    "in_training": d.in_training,
                    "retraction_doi": d.retraction_doi,
                }
                for d in report.documents
            ],
        }

    # ------------------------------------------------------------------
    # Retraction check
    # ------------------------------------------------------------------

    @app.post("/retraction/check")
    async def retraction_check(req: RetractionRequest) -> dict[str, Any]:
        from provenir.governance.retraction import RetractionMonitor

        passport = _load_passport(req.path)
        monitor = RetractionMonitor(known_retracted=frozenset(req.known_retracted))
        report = monitor.check_passport(passport)
        return {
            "checked_dois": list(report.checked_dois),
            "retracted_dois": list(report.retracted_dois),
            "retraction_rate": report.retraction_rate,
            "risk_level": report.risk_level,
            "summary": report.summary(),
        }

    # ------------------------------------------------------------------
    # Regulatory evidence
    # ------------------------------------------------------------------

    @app.post("/regulation")
    async def regulation(req: RegulationRequest) -> dict[str, Any]:
        from provenir.governance.regulation import RegulationGenerator

        passport = _load_passport(req.path)
        gen = RegulationGenerator()
        fw = req.framework
        if fw == "art53":
            report = gen.art53_training_data_summary(passport)
        elif fw == "annex-iv":
            report = gen.annex_iv_technical_file(passport)
        elif fw == "fda-pccp":
            report = gen.fda_pccp_summary(passport)
        elif fw == "nist-rmf":
            report = gen.nist_ai_rmf_summary(passport)
        else:
            raise HTTPException(status_code=422, detail=f"Unknown framework: {fw}")
        return {
            "framework": fw,
            "coverage_score": report.coverage_score(),
            "markdown": report.markdown,
        }

    # ------------------------------------------------------------------
    # Adapters + audit
    # ------------------------------------------------------------------

    @app.get("/adapters")
    async def list_adapters() -> dict[str, Any]:
        root = Path(adapter_dir)
        if not root.exists():
            return {"adapters": []}
        return {"adapters": sorted(p.stem for p in root.glob("*.json"))}

    @app.get("/audit")
    async def get_audit() -> Any:
        audit_path = Path(audit_dir) / "audit.jsonl"
        if not audit_path.exists():
            return JSONResponse(content={"entries": []})
        entries = []
        for line in audit_path.read_text(encoding="utf-8").strip().splitlines():
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                entries.append({"raw": line})
        return JSONResponse(content={"entries": entries})

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
        raise ImportError("uvicorn required: pip install provenir[serve]") from exc
    app = create_app(
        manifest_dir=manifest_dir,
        adapter_dir=adapter_dir,
        audit_dir=audit_dir,
    )
    uvicorn.run(app, host=host, port=port)
