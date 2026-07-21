"""Acquisition Package Export — demonstration script.

Builds a rich ModelPassport (with scan, reward_validity, retraction, and
lineage set) and calls generate_acquisition_package() to produce the full
M&A technical due-diligence package in a temporary directory.

Run:
    python examples/acquisition_demo.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from provenir.governance.acquisition import generate_acquisition_package
from provenir.governance.bom import (
    CodeComponent,
    DataComponent,
    EvalComponent,
    ModelBOM,
    RewardValidityComponent,
)
from provenir.governance.passport import PassportSigner
from provenir.governance.retraction import RetractionComponent
from provenir.governance.scan import ScanComponent

# ---------------------------------------------------------------------------
# Build a rich BOM
# ---------------------------------------------------------------------------

# Simulate a parent passport hash (in a real workflow this comes from the
# parent model's passport.bom.content_hash()).
PARENT_HASH = "a" * 64

scan = ScanComponent(
    scanner_version="0.9.0",
    report_hash="scanhash_demo_abc123",
    unsafe=False,
    finding_counts={"critical": 0, "high": 0, "medium": 1, "low": 3},
)

reward_validity = RewardValidityComponent(
    reward_name="math_verifier",
    report_hash="rewardhash_demo_xyz789",
    validity=0.91,
    spurious=False,
)

retraction = RetractionComponent(
    checked_count=120,
    retracted_count=0,
    retraction_rate=0.0,
    risk_level="none",
    report_hash="retractionhash_demo_456",
)

bom = ModelBOM(
    model_id="acq-demo-model-v2",
    base_model="llama3-70b",
    run_id="run-demo-2026-001",
    data=[
        DataComponent(
            name="openwebtext",
            content_hash="owthash" + "0" * 57,
            num_records=8_000_000,
            license="cc0",
            pii_scanned=True,
            contamination_checked=True,
            source_category="web",
            crawl_domains={
                "reddit.com": 3_200_000,
                "news.ycombinator.com": 1_800_000,
                "arxiv.org": 900_000,
            },
            optout_respected=True,
            retraction_dois=["10.1234/openwebtext.001"],
        ),
        DataComponent(
            name="math-instruct",
            content_hash="mathhash" + "0" * 56,
            num_records=500_000,
            license="apache-2.0",
            pii_scanned=True,
            contamination_checked=True,
            source_category="synthetic",
            optout_respected=True,
        ),
    ],
    code=CodeComponent(
        git_sha="deadbeef12345678",
        dependencies_hash="depshash_demo_aabbccdd",
        framework="trl",
    ),
    evals=[
        EvalComponent(benchmark="mmlu", score=0.74),
        EvalComponent(benchmark="hellaswag", score=0.85),
        EvalComponent(benchmark="gsm8k", score=0.82),
    ],
    hyperparameters={
        "lr": 2e-5,
        "epochs": 2,
        "batch_size": 16,
        "grad_accum": 4,
        "modalities": "text",
    },
    created_at="2026-07-01T08:00:00Z",
    scan=scan,
    reward_validity=reward_validity,
    retraction=retraction,
    parent_passport_hash=PARENT_HASH,
)

passport = PassportSigner(b"demo-signing-key", key_id="demo-ci").sign(
    bom, signed_at="2026-07-01T09:00:00Z"
)

# ---------------------------------------------------------------------------
# Generate the acquisition package
# ---------------------------------------------------------------------------

with tempfile.TemporaryDirectory(prefix="provenir-acq-") as tmp:
    pkg = generate_acquisition_package(passport, tmp, include_regulation=True)

    print(pkg.summary())
    print()
    print("Generated files:")
    for section, filename in sorted(pkg.sections.items()):
        full_path = Path(pkg.output_dir) / filename
        size_kb = full_path.stat().st_size / 1024
        print(f"  [{section:30s}]  {filename}  ({size_kb:.1f} KB)")
