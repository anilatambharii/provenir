"""EU AI Act Art. 53 / Annex IV Evidence Generator — demo.

Builds a rich ModelPassport (with crawl_domains, source_category, optout_respected),
signs it, and generates both the Art. 53 training-data summary and the Annex IV
technical-documentation skeleton.  Prints a coverage summary and writes Markdown
files to /tmp/provenir_demo/ (or a local artifacts/ dir).

Run:
    python examples/art53_evidence_demo.py
"""

from __future__ import annotations

import json
import pathlib
import sys

from provenir.governance.bom import (
    CodeComponent,
    DataComponent,
    EvalComponent,
    ModelBOM,
)
from provenir.governance.passport import PassportSigner
from provenir.governance.regulation import RegulationGenerator

# ---------------------------------------------------------------------------
# 1. Build a rich BOM with Art. 53 provenance fields
# ---------------------------------------------------------------------------

crawl_domains: dict[str, int] = {
    "wikipedia.org": 80_000,
    "arxiv.org": 45_000,
    "github.com": 20_000,
    "stackoverflow.com": 18_000,
    "reddit.com": 12_000,
    "news.ycombinator.com": 7_500,
    "medium.com": 5_000,
    "quora.com": 3_200,
    "nytimes.com": 2_100,
    "bbc.co.uk": 1_800,
    "theguardian.com": 1_400,
    "wired.com": 900,
}

bom = ModelBOM(
    model_id="demo-llm-v1",
    base_model="mistral-7b-instruct",
    run_id="run-demo-2026",
    data=[
        DataComponent(
            name="web-crawl",
            content_hash="a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2",
            num_records=sum(crawl_domains.values()),
            license="various-open",
            pii_scanned=True,
            contamination_checked=True,
            source_category="scraped",
            crawl_domains=crawl_domains,
            optout_respected=True,
        ),
        DataComponent(
            name="licensed-textbooks",
            content_hash="b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3",
            num_records=25_000,
            license="elsevier-research",
            pii_scanned=True,
            contamination_checked=True,
            source_category="licensed",
            optout_respected=None,  # not applicable for licensed data
        ),
        DataComponent(
            name="synthetic-cot",
            content_hash="c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4",
            num_records=50_000,
            license="mit",
            pii_scanned=True,
            contamination_checked=True,
            source_category="synthetic",
            optout_respected=None,  # not applicable for synthetic
        ),
    ],
    code=CodeComponent(
        git_sha="cafebabe1234",
        dependencies_hash="sha256:deps-lockfile-hash",
        framework="trl",
    ),
    evals=[
        EvalComponent(benchmark="mmlu", score=0.812),
        EvalComponent(benchmark="hellaswag", score=0.791),
        EvalComponent(benchmark="arc_easy", score=0.843),
    ],
    hyperparameters={
        "lr": 2e-4,
        "epochs": 3,
        "batch_size": 64,
        "context_length": 4096,
        "modalities": "text",
        "lora_rank": 16,
    },
    created_at="2026-07-01T12:00:00Z",
)

# ---------------------------------------------------------------------------
# 2. Sign the passport (HMAC-SHA256 — swap for asymmetric in production)
# ---------------------------------------------------------------------------

SIGNING_KEY = b"demo-signing-key-replace-in-production"
passport = PassportSigner(SIGNING_KEY, key_id="demo-key-001").sign(
    bom, signed_at="2026-07-01T12:05:00Z"
)

print(f"Passport signed. BOM hash: {bom.content_hash()[:24]}...")
print(f"Signature verified: {passport.verify(SIGNING_KEY)}\n")

# ---------------------------------------------------------------------------
# 3. Generate Art. 53 training-data summary
# ---------------------------------------------------------------------------

gen = RegulationGenerator(sme=False)
art53_report = gen.art53_training_data_summary(passport)

print("=" * 60)
print(art53_report.summary())
print("Missing fields:")
for fc in art53_report.missing():
    print(f"  [{fc.field_id}] {fc.note}")
print()

# ---------------------------------------------------------------------------
# 4. Generate Annex IV technical-documentation skeleton
# ---------------------------------------------------------------------------

annexiv_report = gen.annex_iv_technical_file(passport)

print("=" * 60)
print(annexiv_report.summary())
print("Missing fields:")
for fc in annexiv_report.missing():
    print(f"  [{fc.field_id}] {fc.note}")
print()

# ---------------------------------------------------------------------------
# 5. Write output files
# ---------------------------------------------------------------------------

out_dir = pathlib.Path("artifacts/regulation-demo")
out_dir.mkdir(parents=True, exist_ok=True)

art53_path = out_dir / "art53_training_data_summary.md"
art53_path.write_text(art53_report.markdown, encoding="utf-8")

annexiv_path = out_dir / "annex_iv_technical_file.md"
annexiv_path.write_text(annexiv_report.markdown, encoding="utf-8")

structured_path = out_dir / "evidence_reports.json"
structured_path.write_text(
    json.dumps(
        {
            "art53": art53_report.to_dict(),
            "annex_iv": annexiv_report.to_dict(),
        },
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)

print(f"Art. 53 summary   -> {art53_path}")
print(f"Annex IV skeleton -> {annexiv_path}")
print(f"Structured data   -> {structured_path}")

# ---------------------------------------------------------------------------
# 6. Demonstrate SME threshold (5 % instead of 10 %)
# ---------------------------------------------------------------------------

sme_gen = RegulationGenerator(sme=True)
sme_report = sme_gen.art53_training_data_summary(passport)
sme_rollup = sme_report.data["domain_rollup"]
print(
    f"\nSME mode: top {sme_rollup['threshold_pct']}% threshold -> "
    f"{len(sme_rollup['domains'])} domain(s) listed"
)

# ---------------------------------------------------------------------------
# 7. CI gate demonstration (--fail-under semantics)
# ---------------------------------------------------------------------------

score = annexiv_report.coverage_score()
# Demonstrate fail-under semantics: 10% threshold always passes for a typical passport
# (most organisations cannot auto-fill §4/7/8/9 from the BOM).
threshold_demo = 0.10
print(f"\nAnnex IV coverage: {score:.1%}")
if score < threshold_demo:
    print(f"CI gate demo FAIL: score {score:.1%} < threshold {threshold_demo:.1%}")
    sys.exit(1)
else:
    print(
        f"CI gate demo PASS: score {score:.1%} >= threshold {threshold_demo:.1%} "
        f"(raise threshold once §4/7/8/9 are supplied)"
    )
