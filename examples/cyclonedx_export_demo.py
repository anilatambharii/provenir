"""CycloneDX / SPDX 3.0 export demo for Provenir model passports.

Demonstrates:
- Building a fully populated ModelBOM (data, code, evals, scan, reward validity).
- Signing it into a ModelPassport with PassportSigner.
- Exporting to CycloneDX 1.5 JSON and SPDX 3.0 JSON.
- Printing the first 30 lines of each SBOM to show the expected structure.
- Showing how the CycloneDX output maps to SBOM tooling expectations.

Run with:
    python examples/cyclonedx_export_demo.py
"""

from __future__ import annotations

import json

from provenir.governance.bom import (
    CodeComponent,
    DataComponent,
    EvalComponent,
    ModelBOM,
    RewardValidityComponent,
)
from provenir.governance.export import ExportFormat, export_passport
from provenir.governance.passport import PassportSigner
from provenir.governance.scan import ScanComponent


def main() -> None:
    print("=" * 64)
    print("Provenir CycloneDX / SPDX 3.0 Export Demo")
    print("=" * 64)

    # ------------------------------------------------------------------ #
    # 1. Build a model BOM with realistic provenance data                 #
    # ------------------------------------------------------------------ #
    scan_component = ScanComponent(
        scanner_version="0.7.0",
        report_hash="a1b2c3d4e5f60718",
        unsafe=False,
        finding_counts={"critical": 0, "high": 0, "medium": 2, "low": 1},
    )

    reward_validity = RewardValidityComponent(
        reward_name="math-correctness",
        report_hash="fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210",
        validity=0.912,
        spurious=False,
    )

    bom = ModelBOM(
        model_id="provenir-demo-llm-v1",
        base_model="meta-llama/Llama-3-8B",
        run_id="run-20260701-abc123",
        data=[
            DataComponent(
                name="openhermes-2.5",
                content_hash="sha256:openhermes25contenthash0000000000000000000000000000000000000000",
                num_records=1_000_000,
                license="cc-by-4.0",
                pii_scanned=True,
                contamination_checked=True,
                source_category="web",
                crawl_domains={"redditcom": 120_000, "stackoverflowcom": 80_000},
                optout_respected=True,
            ),
            DataComponent(
                name="math-qa",
                content_hash="sha256:mathqacontenthash00000000000000000000000000000000000000000000000",
                num_records=50_000,
                license="apache-2.0",
                pii_scanned=True,
                contamination_checked=True,
                source_category="curated",
                optout_respected=None,
            ),
        ],
        code=CodeComponent(
            git_sha="abc123def456789012345678901234567890abcd",
            dependencies_hash="sha256:requirementstxthash000000000000000000000000000000000000000000",
            framework="trl",
        ),
        evals=[
            EvalComponent(benchmark="mmlu", score=0.724),
            EvalComponent(benchmark="hellaswag", score=0.812),
            EvalComponent(benchmark="gsm8k", score=0.638),
        ],
        hyperparameters={"lr": 2e-5, "epochs": 3, "batch_size": 64, "grad_accum": 8},
        created_at="2026-07-01T12:00:00Z",
        scan=scan_component,
        reward_validity=reward_validity,
    )

    signer = PassportSigner(b"demo-signing-key", key_id="demo-ci-key")
    passport = signer.sign(bom, signed_at="2026-07-01T12:05:00Z")
    print(f"\nModel: {bom.model_id}")
    print(f"BOM content hash: {bom.content_hash()[:16]}...")
    print(f"Passport signed: {passport.attestation is not None}")
    print(f"Verified: {passport.verify(b'demo-signing-key')}")
    print(f"Risk flags: {bom.risk_flags()}")

    # ------------------------------------------------------------------ #
    # 2. Export to CycloneDX 1.5 JSON                                     #
    # ------------------------------------------------------------------ #
    cdx_json = export_passport(passport, ExportFormat.CYCLONEDX_JSON)
    cdx_lines = cdx_json.splitlines()

    print("\n" + "-" * 64)
    print("CycloneDX 1.5 SBOM (first 30 lines):")
    print("-" * 64)
    for line in cdx_lines[:30]:
        print(line)
    if len(cdx_lines) > 30:
        print(f"  ... ({len(cdx_lines) - 30} more lines)")

    # Highlight SBOM-tooling-relevant fields
    cdx = json.loads(cdx_json)
    print("\n  [CycloneDX tooling expectations]")
    print(f"  bomFormat:     {cdx['bomFormat']}")
    print(f"  specVersion:   {cdx['specVersion']}")
    print(f"  serialNumber:  {cdx['serialNumber']}")
    print(f"  component type:{cdx['metadata']['component']['type']}")
    print(f"  # components:  {len(cdx['components'])}")
    if "vulnerabilities" in cdx:
        print(f"  # vulns:       {len(cdx['vulnerabilities'])}")
        print(f"  vuln id:       {cdx['vulnerabilities'][0]['id']}")
    if "signature" in cdx:
        print(f"  signature alg: {cdx['signature']['algorithm']}")
        print(f"  key id:        {cdx['signature']['keyId']}")

    # ------------------------------------------------------------------ #
    # 3. Export to SPDX 3.0 JSON                                          #
    # ------------------------------------------------------------------ #
    spdx_json = export_passport(passport, ExportFormat.SPDX3_JSON)
    spdx_lines = spdx_json.splitlines()

    print("\n" + "-" * 64)
    print("SPDX 3.0 SBOM (first 30 lines):")
    print("-" * 64)
    for line in spdx_lines[:30]:
        print(line)
    if len(spdx_lines) > 30:
        print(f"  ... ({len(spdx_lines) - 30} more lines)")

    spdx = json.loads(spdx_json)
    print("\n  [SPDX 3.0 structure summary]")
    print(f"  spdxVersion:   {spdx['spdxVersion']}")
    print(f"  dataLicense:   {spdx['dataLicense']}")
    print(f"  name:          {spdx['name']}")
    print(f"  namespace:     {spdx['documentNamespace']}")
    print(f"  # packages:    {len(spdx['packages'])}  (1 model + {len(bom.data)} data)")
    print(f"  # relations:   {len(spdx['relationships'])}")
    print(f"  # annotations: {len(spdx['annotations'])}  "
          f"({len(bom.evals)} evals + 1 scan)")

    # ------------------------------------------------------------------ #
    # 4. Verify determinism                                                #
    # ------------------------------------------------------------------ #
    cdx_json2 = export_passport(passport, ExportFormat.CYCLONEDX_JSON)
    spdx_json2 = export_passport(passport, ExportFormat.SPDX3_JSON)
    assert cdx_json == cdx_json2, "CycloneDX export is not deterministic!"
    assert spdx_json == spdx_json2, "SPDX export is not deterministic!"

    print("\n" + "=" * 64)
    print("Demo complete. Both SBOMs are valid JSON and deterministic.")
    print("=" * 64)


if __name__ == "__main__":
    main()
