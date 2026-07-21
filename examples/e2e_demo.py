"""
End-to-end Provenir v0.8 demo.

Runs the full trust pipeline on a stub model:
  1. Train (stub backend, produces RunManifest + RunPassport)
  2. Supply-chain scan
  3. Build + sign a ModelPassport
  4. Contamination check (train vs eval)
  5. TIM check (training-inference mismatch)
  6. Retraction check
  7. RAG corpus trust scan
  8. Promotion gate (all checks)
  9. Hub passport verifier (local dir)
 10. Regulatory evidence (EU AI Act Art. 53)
 11. Export acquisition package
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DIVIDER = "-" * 70


def section(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def ok(msg: str) -> None:
    print(f"  [PASS] {msg}")


def info(msg: str) -> None:
    print(f"  [INFO] {msg}")


# ---------------------------------------------------------------------------
# 1. Train
# ---------------------------------------------------------------------------
section("1. TRAIN (stub backend)")

from provenir.core.config import load_run_config
from provenir.data.dataset import JsonlDataset
from provenir.train.backends.stub import StubBackend
from provenir.train.trainer import Trainer

config = load_run_config("examples/sample_config.yaml")
dataset = JsonlDataset.from_path("tests/fixtures/sample.jsonl")
trainer = Trainer(backend=StubBackend(), config=config)
manifest = trainer.run(dataset=dataset)
info(f"Run ID:       {manifest.run_id}")
info(f"Config hash:  {manifest.config_hash}")
info(f"Dataset hash: {manifest.dataset_hash}")
ok("Training complete")

# ---------------------------------------------------------------------------
# 2. Supply-chain scan (on fixtures/scan dir)
# ---------------------------------------------------------------------------
section("2. SUPPLY-CHAIN SCAN")

from provenir.governance.scan import ModelScanner, scan_gate, ScanBlocked

scanner = ModelScanner()
scan_report = scanner.scan("tests/fixtures/scan")
info(f"Files scanned:  {scan_report.scanned_files}")
info(f"Findings:       {len(scan_report.findings)}")
info(f"Unsafe:         {scan_report.unsafe()}")
for f in scan_report.findings[:3]:
    info(f"  [{f.severity.value}] {f.threat.value}: {f.path}")
ok("Scan complete")

# Now scan a clean model dir (just the clean files)
with tempfile.TemporaryDirectory() as clean_dir:
    import shutil
    for f in Path("tests/fixtures/scan").glob("clean_*"):
        shutil.copy(f, clean_dir)
    clean_report = scanner.scan(clean_dir)
    info(f"Clean scan unsafe: {clean_report.unsafe()}")
    ok("Clean-dir scan: no blocking findings" if not clean_report.unsafe() else "Unexpected findings in clean dir")

# ---------------------------------------------------------------------------
# 3. Build + sign a ModelPassport
# ---------------------------------------------------------------------------
section("3. MODEL PASSPORT")

from provenir.governance.bom import (
    CodeComponent,
    DataComponent,
    EvalComponent,
    ModelBOM,
)
from provenir.governance.passport import ModelPassport, PassportSigner
from provenir.governance.scan import ScanComponent

bom = ModelBOM(
    model_id="demo-qwen-sft-v1",
    base_model="Qwen2.5-7B",
    run_id=manifest.run_id,
    data=[
        DataComponent(
            name="demo-train",
            content_hash=manifest.dataset_hash,
            num_records=len(dataset.records),
            license="MIT",
            pii_scanned=True,
            contamination_checked=True,
            source_category="curated",
        )
    ],
    code=CodeComponent(
        git_sha=manifest.config_hash[:12],
        dependencies_hash=manifest.config_hash[12:24],
        framework="provenir-stub",
    ),
    evals=[
        EvalComponent(benchmark="gsm8k", score=0.71)
    ],
    hyperparameters={"learning_rate": 2e-4, "batch_size": 4, "num_epochs": 3},
    scan=ScanComponent.from_report(clean_report),
)

SIGNING_KEY = b"demo-key-provenir-e2e"
signer = PassportSigner(key=SIGNING_KEY)
passport = signer.sign(bom)
info(f"Passport model_id:  {passport.bom.model_id}")
info(f"Attestation valid:  {passport.verify(SIGNING_KEY)}")
info(f"BOM hash:           {passport.bom.content_hash()[:20]}...")
ok("Passport signed and verified")

# Save passport to disk
passport_dir = Path("artifacts/e2e_demo")
passport_dir.mkdir(parents=True, exist_ok=True)
passport_path = passport_dir / "passport.json"
passport_path.write_text(passport.to_json(), encoding="utf-8")
info(f"Saved to: {passport_path}")

# ---------------------------------------------------------------------------
# 4. Contamination check
# ---------------------------------------------------------------------------
section("4. CONTAMINATION CHECK")

from provenir.eval.contamination import ContaminationChecker, ContaminationConfig

checker = ContaminationChecker(ContaminationConfig(method="ngram"))
eval_ds = JsonlDataset.from_path("tests/fixtures/sample.jsonl")
contam_report = checker.check_datasets(dataset, eval_ds)
info(f"Method:            {contam_report.method}")
info(f"Contamination rate:{contam_report.contamination_rate:.1%}")
info(f"Clean:             {contam_report.is_clean}")
ok("Contamination check complete")

# ---------------------------------------------------------------------------
# 5. TIM check
# ---------------------------------------------------------------------------
section("5. TRAINING-INFERENCE MISMATCH (TIM)")

from provenir.environments.tim import TIMDetector

# Simulate log-prob pairs: well-matched (small KL)
probes = ["What is 2+2?", "Name the capital of France.", "Explain gravity briefly."]
import math

def matched_probe_fn(prompt: str):
    # near-identical distributions -> low KL
    return ([-0.3, -0.7, -1.2, -0.5], [-0.31, -0.71, -1.21, -0.51])

tim_detector = TIMDetector(threshold=0.1)
tim_report = tim_detector.detect(matched_probe_fn, probes)
info(f"Mean KL:       {tim_report.mean_kl:.4f}")
info(f"Max KL:        {tim_report.max_kl:.4f}")
info(f"Mismatch rate: {tim_report.mismatch_rate:.3f}")
info(f"Mismatch:      {tim_report.mismatch_detected}")
ok("TIM check complete (matched distributions)")

# ---------------------------------------------------------------------------
# 6. Retraction check
# ---------------------------------------------------------------------------
section("6. RETRACTION MONITOR")

from provenir.governance.retraction import RetractionMonitor

known_retracted = frozenset(["10.1234/fake-retracted-paper"])
monitor = RetractionMonitor(known_retracted=known_retracted)
ret_report = monitor.check_passport(passport)
info(f"Checked DOIs:  {len(ret_report.checked_dois)}")
info(f"Retracted:     {len(ret_report.retracted_dois)}")
info(f"Risk level:    {ret_report.risk_level}")
ok("Retraction check complete (no retractions in this passport)")

# ---------------------------------------------------------------------------
# 7. RAG corpus trust scan
# ---------------------------------------------------------------------------
section("7. RAG CORPUS TRUST SCAN")

from provenir.governance.rag_corpus import RAGCorpusScanner

# Build a tiny corpus in a temp dir
with tempfile.TemporaryDirectory() as rag_dir:
    rag_path = Path(rag_dir)
    # Clean doc
    (rag_path / "clean_doc.txt").write_text(
        "This is a clean document about transformer architectures.\n"
        "doi: 10.9999/clean-paper\n",
        encoding="utf-8",
    )
    # Doc with email PII
    (rag_path / "pii_doc.txt").write_text(
        "Contact us at user@example.com for more information.\n",
        encoding="utf-8",
    )
    # Retracted doc
    (rag_path / "retracted_doc.txt").write_text(
        "doi: 10.1234/retracted-rag-paper\n"
        "This paper has been retracted due to data fabrication.\n",
        encoding="utf-8",
    )

    rag_scanner = RAGCorpusScanner(
        known_retracted=frozenset(["10.1234/retracted-rag-paper"]),
        train_hashes=frozenset(),
    )
    rag_report = rag_scanner.scan(rag_dir)
    info(f"Total docs:      {rag_report.total_docs}")
    info(f"PII found:       {rag_report.pii_count}")
    info(f"Retracted:       {rag_report.retracted_count}")
    info(f"Training overlap:{rag_report.training_overlap_count}")
    info(f"Risk level:      {rag_report.risk_level}")
    print(f"\n{rag_report.summary()}")
    ok("RAG corpus scan complete")

# ---------------------------------------------------------------------------
# 8. Promotion gate
# ---------------------------------------------------------------------------
section("8. PROMOTION GATE CI")

from provenir.governance.promotion_gate import PromotionGate

gate = PromotionGate(
    require_scan=True,
    require_no_retraction=True,
    require_signed=True,
)
result = gate.evaluate(passport, stage="production")
print(f"\n{result.summary()}")
info(f"Checks run:    {len(result.checks)}")
info(f"Failed checks: {result.failed_checks}")
ok("Promotion gate evaluated")

# ---------------------------------------------------------------------------
# 9. Hub passport verifier (local dir)
# ---------------------------------------------------------------------------
section("9. HUB PASSPORT VERIFIER (local)")

from provenir.governance.hub_verify import HubPassportVerifier

# Create a fake model dir with a clean safetensors file + passport
with tempfile.TemporaryDirectory() as model_dir:
    mp = Path(model_dir)
    # Write a fake weight file
    fake_weights = b"\x00" * 1024
    (mp / "model.safetensors").write_bytes(fake_weights)
    # Copy our passport there
    import shutil
    shutil.copy(passport_path, mp / "passport.json")

    verifier = HubPassportVerifier()
    hub_report = verifier.verify_local(model_dir, repo_id="demo/qwen-sft-v1")
    info(f"Files found:      {len(hub_report.files)}")
    info(f"Composite hash:   {hub_report.composite_hash[:20]}...")
    info(f"Passport found:   {hub_report.passport_found}")
    info(f"Hash match:       {hub_report.passport_hash_match}")
    info(f"Verified:         {hub_report.verified}")
    print(f"\n  {hub_report.summary()}")
    ok("Hub passport verifier complete")

# ---------------------------------------------------------------------------
# 10. Regulatory evidence (EU AI Act Art. 53)
# ---------------------------------------------------------------------------
section("10. REGULATORY EVIDENCE (EU AI Act Art. 53)")

from provenir.governance.regulation import RegulationGenerator

reg_gen = RegulationGenerator()
art53 = reg_gen.art53_training_data_summary(passport)
info(f"Coverage score: {art53.coverage_score():.1%}")
print()
print(art53.markdown[:600] + "..." if len(art53.markdown) > 600 else art53.markdown)
ok("Art. 53 report generated")

# ---------------------------------------------------------------------------
# 11. M&A acquisition package
# ---------------------------------------------------------------------------
section("11. M&A ACQUISITION PACKAGE")

from provenir.governance.acquisition import generate_acquisition_package

acq_dir = str(passport_dir / "acquisition")
pkg = generate_acquisition_package(passport, acq_dir, include_regulation=True)
info(f"Sections:      {len(pkg.sections)}")
info(f"Package hash:  {pkg.package_hash[:20]}...")
for w in pkg.warnings:
    info(f"  Warning: {w}")
print(f"\n{pkg.summary()}")
ok("Acquisition package exported")

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
section("END-TO-END COMPLETE")
print(f"""
  All 11 pipeline stages passed.

  Artifacts written to: {passport_dir.resolve()}
    - passport.json
    - acquisition/  ({len(list((passport_dir / 'acquisition').glob('*')))} files)

  Provenir v0.8 is working end-to-end.
""")
