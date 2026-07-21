"""CycloneDX 1.5 and SPDX 3.0 export for Provenir model passports.

Converts a :class:`~provenir.governance.passport.ModelPassport` into a
standards-compliant Software Bill of Materials (SBOM) in either CycloneDX JSON
or SPDX 3.0 JSON format.  No external libraries are required — only stdlib
``json`` and ``enum``.

Example::

    from provenir.governance.export import ExportFormat, export_passport
    from provenir.governance.passport import ModelPassport

    sbom_json = export_passport(passport, ExportFormat.CYCLONEDX_JSON)
    spdx_json = export_passport(passport, ExportFormat.SPDX3_JSON)
"""

from __future__ import annotations

import json
from enum import Enum
from typing import Any

from provenir.governance.passport import ModelPassport


class ExportFormat(str, Enum):
    """Supported SBOM export formats."""

    CYCLONEDX_JSON = "cyclonedx-json"
    SPDX3_JSON = "spdx3-json"


# ---------------------------------------------------------------------------
# CycloneDX 1.5
# ---------------------------------------------------------------------------


def to_cyclonedx(passport: ModelPassport) -> dict[str, Any]:
    """Return a CycloneDX 1.5 BOM dict for *passport*.

    Structure follows https://cyclonedx.org/docs/1.5/json/ — implemented
    without any external library.

    Example:
        >>> from provenir.governance.bom import (  # noqa: E501
        ...     CodeComponent, DataComponent, EvalComponent, ModelBOM
        ... )
        >>> from provenir.governance.passport import ModelPassport
        >>> bom = ModelBOM(
        ...     model_id="m1", base_model="base", run_id="r1",
        ...     data=[DataComponent(name="d", content_hash="abc123", num_records=10)],
        ...     code=CodeComponent(git_sha="s", dependencies_hash="dh", framework="trl"),
        ...     evals=[EvalComponent(benchmark="mmlu", score=0.5)],
        ... )
        >>> p = ModelPassport(bom=bom, attestation=None)
        >>> cdx = to_cyclonedx(p)
        >>> cdx["bomFormat"]
        'CycloneDX'
    """
    bom = passport.bom

    # --- metadata.component properties ----------------------------------
    meta_props: list[dict[str, str]] = [
        {"name": "provenir:framework", "value": bom.code.framework},
        {"name": "provenir:git_sha", "value": bom.code.git_sha},
        {"name": "provenir:created_at", "value": bom.created_at},
    ]
    if bom.reward_validity is not None:
        rv = bom.reward_validity
        meta_props.append(
            {
                "name": "provenir:reward_validity",
                "value": f"{rv.validity:.3f} spurious={rv.spurious}",
            }
        )

    metadata_component: dict[str, Any] = {
        "type": "machine-learning-model",
        "name": bom.model_id,
        "version": bom.run_id,
        "description": f"Base: {bom.base_model}",
        "properties": meta_props,
    }

    # --- data components -------------------------------------------------
    components: list[dict[str, Any]] = []
    for component in bom.data:
        comp_entry: dict[str, Any] = {
            "type": "data",
            "name": component.name,
            "version": component.content_hash[:16],
            "licenses": [{"license": {"id": component.license}}],
            "properties": [
                {"name": "provenir:num_records", "value": str(component.num_records)},
                {"name": "provenir:pii_scanned", "value": str(component.pii_scanned).lower()},
                {"name": "provenir:source_category", "value": component.source_category},
            ],
        }
        components.append(comp_entry)

    # --- vulnerabilities (only when scan present) ------------------------
    result: dict[str, Any] = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "version": 1,
        "serialNumber": f"urn:uuid:{bom.content_hash()[:32]}",
        "metadata": {"component": metadata_component},
        "components": components,
    }

    if bom.scan is not None:
        scan_dict = bom.scan.to_dict()
        report_hash: str = str(scan_dict.get("report_hash", ""))
        unsafe: bool = bool(scan_dict.get("unsafe", False))
        finding_counts: object = scan_dict.get("finding_counts", {})
        vuln_id = f"PROVENIR-SCAN-{report_hash[:8].upper()}"
        result["vulnerabilities"] = [
            {
                "id": vuln_id,
                "source": {"name": "Provenir Supply-Chain Scanner"},
                "ratings": [{"severity": "critical" if unsafe else "none"}],
                "description": f"Supply-chain scan: {finding_counts}",
                "analysis": {"state": "in_triage" if unsafe else "false_positive"},
            }
        ]

    # --- signature (only when attestation present) ----------------------
    if passport.attestation is not None:
        attest = passport.attestation
        result["signature"] = {
            "algorithm": attest.algorithm,
            "value": attest.signature,
            "keyId": attest.key_id,
        }

    return result


# ---------------------------------------------------------------------------
# SPDX 3.0
# ---------------------------------------------------------------------------


def to_spdx3(passport: ModelPassport) -> dict[str, Any]:
    """Return an SPDX 3.0 AI-profile JSON dict for *passport*.

    Covers the core SPDX 3.0 document structure plus the AI profile fields
    for the primary model package, without any external library.

    Example:
        >>> from provenir.governance.bom import (
        ...     CodeComponent, DataComponent, EvalComponent, ModelBOM
        ... )
        >>> from provenir.governance.passport import ModelPassport
        >>> bom = ModelBOM(
        ...     model_id="m1", base_model="base", run_id="r1",
        ...     data=[DataComponent(name="d", content_hash="abc123", num_records=10)],
        ...     code=CodeComponent(git_sha="s", dependencies_hash="dh", framework="trl"),
        ...     evals=[EvalComponent(benchmark="mmlu", score=0.5)],
        ... )
        >>> p = ModelPassport(bom=bom, attestation=None)
        >>> spdx = to_spdx3(p)
        >>> spdx["spdxVersion"]
        'SPDX-3.0'
    """
    bom = passport.bom
    created_at = bom.created_at or "NOASSERTION"

    # --- packages --------------------------------------------------------
    packages: list[dict[str, Any]] = [
        {
            "SPDXID": "SPDXRef-Model",
            "name": bom.model_id,
            "versionInfo": bom.run_id,
            "downloadLocation": "NOASSERTION",
            "filesAnalyzed": False,
            "primaryPackagePurpose": "MACHINE-LEARNING-MODEL",
            "checksum": [{"algorithm": "SHA256", "checksumValue": bom.content_hash()}],
            "ai:modelCard": {
                "ai:autonomyType": "NotAutonomous",
                "ai:informationAboutTraining": bom.base_model,
            },
        }
    ]

    relationships: list[dict[str, Any]] = []
    for i, component in enumerate(bom.data):
        spdx_id = f"SPDXRef-Data-{i}"
        packages.append(
            {
                "SPDXID": spdx_id,
                "name": component.name,
                "versionInfo": component.content_hash,
                "downloadLocation": "NOASSERTION",
                "filesAnalyzed": False,
                "primaryPackagePurpose": "DATA",
                "licenseDeclared": component.license,
                "checksum": [{"algorithm": "SHA256", "checksumValue": component.content_hash}],
            }
        )
        relationships.append(
            {
                "spdxElementId": "SPDXRef-Model",
                "relationshipType": "HAS_DATA_INPUT",
                "relatedSpdxElement": spdx_id,
            }
        )

    # --- annotations (evals + scan) -------------------------------------
    annotations: list[dict[str, Any]] = []
    for evaluation in bom.evals:
        annotations.append(
            {
                "annotationType": "REVIEW",
                "annotator": "Tool: Provenir",
                "annotationDate": created_at,
                "comment": f"Eval: {evaluation.benchmark} score={evaluation.score}",
            }
        )
    if bom.scan is not None:
        scan_dict = bom.scan.to_dict()
        annotations.append(
            {
                "annotationType": "REVIEW",
                "annotator": "Tool: Provenir",
                "annotationDate": created_at,
                "comment": (
                    f"Supply-chain scan: unsafe={scan_dict.get('unsafe')} "
                    f"findings={scan_dict.get('finding_counts')}"
                ),
            }
        )

    return {
        "spdxVersion": "SPDX-3.0",
        "dataLicense": "CC0-1.0",
        "SPDXID": "SPDXRef-DOCUMENT",
        "name": f"{bom.model_id}-sbom",
        "documentNamespace": f"https://provenir.ai/sbom/{bom.content_hash()[:16]}",
        "creationInfo": {
            "created": created_at,
            "creators": ["Tool: Provenir"],
            "licenseListVersion": "3.22",
        },
        "packages": packages,
        "relationships": relationships,
        "annotations": annotations,
    }


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def export_passport(passport: ModelPassport, fmt: ExportFormat) -> str:
    """Serialise *passport* to the requested SBOM format and return JSON.

    Args:
        passport: The signed (or unsigned) model passport to export.
        fmt: Target format — :attr:`ExportFormat.CYCLONEDX_JSON` or
             :attr:`ExportFormat.SPDX3_JSON`.

    Returns:
        A pretty-printed, deterministic JSON string.

    Example:
        >>> from provenir.governance.bom import (
        ...     CodeComponent, DataComponent, EvalComponent, ModelBOM
        ... )
        >>> from provenir.governance.passport import ModelPassport
        >>> bom = ModelBOM(
        ...     model_id="m1", base_model="base", run_id="r1",
        ...     data=[DataComponent(name="d", content_hash="abc123", num_records=10)],
        ...     code=CodeComponent(git_sha="s", dependencies_hash="dh", framework="trl"),
        ...     evals=[EvalComponent(benchmark="mmlu", score=0.5)],
        ... )
        >>> p = ModelPassport(bom=bom, attestation=None)
        >>> out = export_passport(p, ExportFormat.CYCLONEDX_JSON)
        >>> import json; json.loads(out)["bomFormat"]
        'CycloneDX'
    """
    if fmt is ExportFormat.CYCLONEDX_JSON:
        data = to_cyclonedx(passport)
    else:
        data = to_spdx3(passport)
    return json.dumps(data, sort_keys=True, indent=2, ensure_ascii=False)
