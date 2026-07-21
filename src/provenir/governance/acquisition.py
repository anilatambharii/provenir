"""Acquisition Package Export for Provenir.

Generates a signed, self-contained due-diligence package from a
:class:`~provenir.governance.passport.ModelPassport`.  The package answers
every standard M&A technical due-diligence question about an AI model:

- What training data was used, and are there IP/license risks?
- Has the model artifact been scanned for supply-chain threats?
- What is the regulatory exposure under EU AI Act, FDA PCCP, and NIST AI RMF?
- Is the reward signal reliable, and have the evals been contamination-checked?
- Can the training run be reproduced from the BOM?
- Are any training DOIs retracted?
- What is the model's fine-tuning lineage?

Call :func:`generate_acquisition_package` to produce the package in seconds.

Example::

    from provenir.governance.acquisition import generate_acquisition_package
    package = generate_acquisition_package(passport, "/tmp/acq-pkg")
    print(package.summary())
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from provenir.governance.passport import ModelPassport

# ---------------------------------------------------------------------------
# Section name constants
# ---------------------------------------------------------------------------

SECTION_BOM_MANIFEST = "bom-manifest"
SECTION_IP_RISK = "ip-risk"
SECTION_SECURITY_AUDIT = "security-audit"
SECTION_REGULATORY_EXPOSURE = "regulatory-exposure"
SECTION_MODEL_RELIABILITY = "model-reliability"
SECTION_REPRODUCIBILITY = "reproducibility"
SECTION_RETRACTION_RISK = "retraction-risk"
SECTION_LINEAGE = "lineage"
SECTION_SIGNED_SUMMARY = "SIGNED-SUMMARY"


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AcquisitionPackage:
    """The result of :func:`generate_acquisition_package`.

    Attributes:
        model_id:     The model identifier from the BOM.
        output_dir:   Absolute path to the directory containing all written files.
        sections:     Mapping from section name to the relative filename written.
        package_hash: SHA-256 of all written file contents (sorted by filename).
        warnings:     Human-readable gap flags collected during generation.

    Example:
        >>> pkg = AcquisitionPackage(
        ...     model_id="m1",
        ...     output_dir="/tmp/pkg",
        ...     sections={"bom-manifest": "bom-manifest.json"},
        ...     package_hash="abc",
        ...     warnings=[],
        ... )
        >>> pkg.to_dict()["model_id"]
        'm1'
    """

    model_id: str
    output_dir: str
    sections: dict[str, str] = field(default_factory=dict)
    package_hash: str = ""
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation of the package."""
        return {
            "model_id": self.model_id,
            "output_dir": self.output_dir,
            "sections": dict(self.sections),
            "package_hash": self.package_hash,
            "warnings": list(self.warnings),
        }

    def summary(self) -> str:
        """Return a multi-line human-readable summary of the package."""
        lines: list[str] = [
            f"Acquisition Package: {self.model_id}",
            f"  Output dir : {self.output_dir}",
            f"  Sections   : {len(self.sections)}",
            f"  Hash       : {self.package_hash[:16]}...",
        ]
        if self.warnings:
            lines.append("  Warnings:")
            for w in self.warnings:
                lines.append(f"    - {w}")
        else:
            lines.append("  Warnings   : none")
        lines.append("")
        lines.append("  Files written:")
        for section, filename in sorted(self.sections.items()):
            lines.append(f"    [{section}]  {filename}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Section generators
# ---------------------------------------------------------------------------


def _write_bom_manifest(output_dir: Path, passport: ModelPassport) -> str:
    filename = "bom-manifest.json"
    (output_dir / filename).write_text(passport.to_json(), encoding="utf-8")
    return filename


def _write_ip_risk(
    output_dir: Path,
    passport: ModelPassport,
    warnings: list[str],
) -> str:
    bom = passport.bom
    lines: list[str] = [f"# IP Risk Assessment: {bom.model_id}", ""]

    # --- Training Data Licenses ---
    lines.append("## Training Data Licenses")
    has_unknown_license = False
    for dc in bom.data:
        lines.append(
            f"- **{dc.name}** — license: `{dc.license}`, "
            f"records: {dc.num_records:,}, category: {dc.source_category}"
        )
        if not dc.license or dc.license.lower() == "unknown":
            lines.append(
                f"  - ⚠ Unknown license: {dc.name} ({dc.num_records:,} records)"
                " — verify IP clearance"
            )
            has_unknown_license = True
    if not bom.data:
        lines.append("- (no data components)")
    if has_unknown_license:
        warnings.append("unknown licenses found — IP clearance required")
    lines.append("")

    # --- Crawl Domain Exposure ---
    lines.append("## Crawl Domain Exposure")
    all_domains: dict[str, int] = {}
    for dc in bom.data:
        for domain, count in dc.crawl_domains.items():
            all_domains[domain] = all_domains.get(domain, 0) + count
    if all_domains:
        top5 = sorted(all_domains.items(), key=lambda x: x[1], reverse=True)[:5]
        lines.append("Top crawled domains by record volume:")
        for domain, count in top5:
            lines.append(f"- `{domain}`: {count:,} records")
    else:
        lines.append(
            "No domain-level provenance — supply crawl manifest for full IP assessment"
        )
    lines.append("")

    # --- Opt-Out Compliance ---
    lines.append("## Opt-Out Compliance")
    unverified = [dc.name for dc in bom.data if dc.optout_respected is None]
    if unverified:
        lines.append(
            f"⚠ Opt-out respect unverified for {len(unverified)} dataset(s):"
            f" {', '.join(unverified)}"
        )
    elif all(dc.optout_respected for dc in bom.data if dc.optout_respected is not None):
        lines.append("All datasets respect opt-out signals (robots.txt/TDM).")
    else:
        partial = [
            dc.name for dc in bom.data if dc.optout_respected is False
        ]
        lines.append(
            f"⚠ Opt-out NOT respected for: {', '.join(partial)}"
        )
    lines.append("")

    # --- Risk Flags ---
    lines.append("## Risk Flags")
    flags = bom.risk_flags()
    if flags:
        for flag in flags:
            lines.append(f"- `{flag}`")
    else:
        lines.append("None")
    lines.append("")

    # --- BOM Hash ---
    lines.append("## BOM Hash")
    lines.append(f"`{bom.content_hash()}`")
    lines.append("")

    filename = "ip-risk.md"
    (output_dir / filename).write_text("\n".join(lines), encoding="utf-8")
    return filename


def _write_security_audit(
    output_dir: Path,
    passport: ModelPassport,
    warnings: list[str],
) -> str:
    bom = passport.bom
    lines: list[str] = [f"# Security Audit: {bom.model_id}", ""]

    # --- Supply-Chain Scan ---
    lines.append("## Supply-Chain Scan")
    if bom.scan is not None:
        sc = bom.scan
        result_str = "FAIL — unsafe findings present" if sc.unsafe else "PASS"
        lines.append(f"- Scanner version: {sc.scanner_version}")
        lines.append(f"- Report hash: `{sc.report_hash}`")
        lines.append(f"- Unsafe: {sc.unsafe}")
        lines.append(f"- Finding counts: {sc.finding_counts}")
        lines.append(f"- Result: {result_str}")
    else:
        lines.append(
            "⚠ No supply-chain scan — model artifact has not been scanned for threats."
        )
        lines.append(
            "  Run `provenir scan <model_path>` and attach ScanComponent to the BOM."
        )
        warnings.append(
            "no scan component — security section is partial"
        )
    lines.append("")

    # --- Retraction Risk (cross-reference) ---
    lines.append("## Retraction Risk")
    if bom.retraction is not None:
        rc = bom.retraction
        lines.append(
            f"- Retracted DOIs: {rc.retracted_count} of {rc.checked_count} checked"
        )
        lines.append(f"- Rate: {rc.retraction_rate:.1%}")
        lines.append(f"- Risk level: {rc.risk_level}")
    else:
        lines.append("No retraction check performed.")
    lines.append("")

    filename = "security-audit.md"
    (output_dir / filename).write_text("\n".join(lines), encoding="utf-8")
    return filename


def _write_regulatory_exposure(
    output_dir: Path,
    passport: ModelPassport,
    *,
    include_regulation: bool,
) -> str:
    bom = passport.bom
    lines: list[str] = [f"# Regulatory Exposure: {bom.model_id}", ""]

    # --- EU AI Act ---
    lines.append("## EU AI Act Art. 53 / Annex IV")
    if include_regulation:
        from provenir.governance.regulation import RegulationGenerator

        gen = RegulationGenerator()
        art53 = gen.art53_training_data_summary(passport)
        annex_iv = gen.annex_iv_technical_file(passport)

        art53_score = art53.coverage_score()
        art53_missing = len(art53.missing())
        lines.append(
            f"- Art. 53 coverage: {art53_score:.0%}  Missing: {art53_missing} field(s)"
        )
        annex_score = annex_iv.coverage_score()
        annex_missing = len(annex_iv.missing())
        lines.append(
            f"- Annex IV coverage: {annex_score:.0%}  Missing: {annex_missing} field(s)"
        )

        all_missing = art53.missing() + annex_iv.missing()
        if all_missing:
            lines.append("")
            lines.append("### Missing Fields")
            for fc in all_missing:
                note = f" — {fc.note}" if fc.note else ""
                lines.append(f"- `{fc.field_id}`{note}")
    else:
        lines.append("Regulation analysis skipped (include_regulation=False).")
    lines.append("")

    # --- FDA PCCP ---
    lines.append("## FDA PCCP (US)")
    if include_regulation:
        from provenir.governance.regulation import RegulationGenerator

        gen2 = RegulationGenerator()
        fda = gen2.fda_pccp_summary(passport)
        lines.append(f"- Coverage: {fda.coverage_score():.0%}")
    else:
        lines.append("Regulation analysis skipped (include_regulation=False).")
    lines.append("")

    # --- NIST AI RMF ---
    lines.append("## NIST AI RMF")
    if include_regulation:
        from provenir.governance.regulation import RegulationGenerator

        gen3 = RegulationGenerator()
        nist = gen3.nist_ai_rmf_summary(passport)
        lines.append(f"- Coverage: {nist.coverage_score():.0%}")
    else:
        lines.append("Regulation analysis skipped (include_regulation=False).")
    lines.append("")

    filename = "regulatory-exposure.md"
    (output_dir / filename).write_text("\n".join(lines), encoding="utf-8")
    return filename


def _write_model_reliability(
    output_dir: Path,
    passport: ModelPassport,
    warnings: list[str],
) -> str:
    bom = passport.bom
    lines: list[str] = [f"# Model Reliability: {bom.model_id}", ""]

    # --- Reward Validity ---
    lines.append("## Reward Validity")
    if bom.reward_validity is not None:
        rv = bom.reward_validity
        spurious_verdict = (
            "FAIL — reward signal not distinguishable from degenerate controls"
            if rv.spurious
            else "PASS"
        )
        lines.append(f"- Reward: {rv.reward_name}")
        lines.append(f"- Validity score: {rv.validity:.3f}")
        lines.append(f"- Spurious: {rv.spurious} — {spurious_verdict}")
        lines.append(f"- Report hash: `{rv.report_hash}`")
    else:
        lines.append(
            "⚠ No reward validity assessment — run `provenir reward-validity`"
            " before production promotion."
        )
        warnings.append(
            "no reward validity assessment — reliability section is partial"
        )
    lines.append("")

    # --- Evaluations ---
    lines.append("## Evaluations")
    if bom.evals:
        for ev in bom.evals:
            contamination_marker = " **(contaminated)**" if ev.contaminated else ""
            lines.append(
                f"- **{ev.benchmark}**: score={ev.score}{contamination_marker}"
            )
    else:
        lines.append("⚠ No evaluations recorded in BOM.")
        warnings.append("no evaluations recorded in BOM")
    lines.append("")

    filename = "model-reliability.md"
    (output_dir / filename).write_text("\n".join(lines), encoding="utf-8")
    return filename


def _write_reproducibility(
    output_dir: Path,
    passport: ModelPassport,
) -> str:
    bom = passport.bom
    lines: list[str] = [f"# Reproducibility: {bom.model_id}", ""]

    # --- Training Run ---
    lines.append("## Training Run")
    lines.append(f"- Run ID: {bom.run_id}")
    lines.append(f"- Base model: {bom.base_model}")
    lines.append(f"- Created at: {bom.created_at or 'unknown'}")
    lines.append("")

    # --- Code Provenance ---
    lines.append("## Code Provenance")
    lines.append(f"- Framework: {bom.code.framework}")
    lines.append(f"- Git SHA: `{bom.code.git_sha}`")
    lines.append(f"- Dependencies hash: `{bom.code.dependencies_hash}`")
    lines.append("")

    # --- Hyperparameters ---
    lines.append("## Hyperparameters")
    if bom.hyperparameters:
        for key in sorted(bom.hyperparameters):
            lines.append(f"- {key}: {bom.hyperparameters[key]}")
    else:
        lines.append("- (none recorded)")
    lines.append("")

    # --- Data Fingerprints ---
    lines.append("## Data Fingerprints")
    for dc in bom.data:
        lines.append(
            f"- **{dc.name}**: hash=`{dc.content_hash}`, records={dc.num_records:,}"
        )
    if not bom.data:
        lines.append("- (no data components)")
    lines.append("")

    # --- BOM Content Hash ---
    lines.append("## BOM Content Hash")
    lines.append(f"`{bom.content_hash()}`")
    lines.append("")

    # --- Attestation ---
    lines.append("## Attestation")
    if passport.attestation is not None:
        att = passport.attestation
        lines.append("- Status: SIGNED")
        lines.append(f"- Algorithm: {att.algorithm}")
        lines.append(f"- Key ID: {att.key_id}")
        lines.append(f"- Signed at: {att.signed_at or 'unknown'}")
    else:
        lines.append("- Status: UNSIGNED — no tamper-evident attestation")
    lines.append("")

    filename = "reproducibility.md"
    (output_dir / filename).write_text("\n".join(lines), encoding="utf-8")
    return filename


def _write_retraction_risk(
    output_dir: Path,
    passport: ModelPassport,
) -> str:
    bom = passport.bom
    lines: list[str] = [f"# Retraction Risk: {bom.model_id}", ""]

    if bom.retraction is not None:
        rc = bom.retraction
        lines.append(f"- DOIs checked: {rc.checked_count}")
        lines.append(f"- Retracted: {rc.retracted_count}")
        lines.append(f"- Rate: {rc.retraction_rate:.1%}")
        lines.append(f"- Risk level: {rc.risk_level}")
        lines.append(f"- Report hash: `{rc.report_hash}`")
    else:
        lines.append("No retraction monitoring configured.")
        lines.append(
            "Add retraction_dois to DataComponents and run"
            " RetractionMonitor.check_passport()."
        )
    lines.append("")

    filename = "retraction-risk.md"
    (output_dir / filename).write_text("\n".join(lines), encoding="utf-8")
    return filename


def _write_lineage(
    output_dir: Path,
    passport: ModelPassport,
) -> str:
    bom = passport.bom
    lines: list[str] = [f"# Model Lineage: {bom.model_id}", ""]

    if bom.parent_passport_hash:
        lines.append(f"- Parent passport hash: `{bom.parent_passport_hash}`")
        lines.append(
            "- This model was fine-tuned from a parent;"
            " provide the parent passport for full chain verification."
        )
        lines.append(
            "- Run `provenir lineage show` with both passports to verify the chain."
        )
    else:
        lines.append("- Base model (no parent passport recorded).")
    lines.append("")

    filename = "lineage.md"
    (output_dir / filename).write_text("\n".join(lines), encoding="utf-8")
    return filename


def _compute_package_hash(output_dir: Path, filenames: list[str]) -> str:
    """SHA-256 of all written file contents concatenated (sorted by filename)."""
    h = hashlib.sha256()
    for name in sorted(filenames):
        content = (output_dir / name).read_bytes()
        h.update(content)
    return h.hexdigest()


def _write_signed_summary(
    output_dir: Path,
    passport: ModelPassport,
    sections: dict[str, str],
    package_hash: str,
    warnings: list[str],
) -> str:
    bom = passport.bom
    summary: dict[str, Any] = {
        "model_id": bom.model_id,
        "bom_hash": bom.content_hash(),
        "package_hash": package_hash,
        "sections": dict(sections),
        "warnings": list(warnings),
        "attestation_present": passport.attestation is not None,
    }
    filename = "SIGNED-SUMMARY.json"
    (output_dir / filename).write_text(
        json.dumps(summary, sort_keys=True, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return filename


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def generate_acquisition_package(
    passport: ModelPassport,
    output_dir: str | Path,
    *,
    include_regulation: bool = True,
) -> AcquisitionPackage:
    """Generate a complete M&A technical due-diligence package from *passport*.

    Writes one file per section into *output_dir* (created if absent), then
    writes a ``SIGNED-SUMMARY.json`` whose ``package_hash`` is the SHA-256 of
    all written file contents sorted by filename.

    Args:
        passport:           A signed (or unsigned) :class:`ModelPassport`.
        output_dir:         Destination directory; created with ``parents=True``
                            and ``exist_ok=True``.
        include_regulation: When ``False``, skips :class:`RegulationGenerator`
                            calls (useful for unit tests and offline environments).

    Returns:
        :class:`AcquisitionPackage` with ``sections``, ``package_hash``, and
        any ``warnings`` collected during generation.

    Example::

        pkg = generate_acquisition_package(passport, "/tmp/acq")
        print(pkg.summary())
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    bom = passport.bom
    warnings: list[str] = []

    if passport.attestation is None:
        warnings.append("unsigned passport — no tamper-evident attestation")

    sections: dict[str, str] = {}

    sections[SECTION_BOM_MANIFEST] = _write_bom_manifest(out, passport)
    sections[SECTION_IP_RISK] = _write_ip_risk(out, passport, warnings)
    sections[SECTION_SECURITY_AUDIT] = _write_security_audit(out, passport, warnings)
    sections[SECTION_REGULATORY_EXPOSURE] = _write_regulatory_exposure(
        out, passport, include_regulation=include_regulation
    )
    sections[SECTION_MODEL_RELIABILITY] = _write_model_reliability(out, passport, warnings)
    sections[SECTION_REPRODUCIBILITY] = _write_reproducibility(out, passport)
    sections[SECTION_RETRACTION_RISK] = _write_retraction_risk(out, passport)
    sections[SECTION_LINEAGE] = _write_lineage(out, passport)

    # Compute package hash over all non-summary files
    package_hash = _compute_package_hash(out, list(sections.values()))

    sections[SECTION_SIGNED_SUMMARY] = _write_signed_summary(
        out, passport, sections, package_hash, warnings
    )

    return AcquisitionPackage(
        model_id=bom.model_id,
        output_dir=str(out),
        sections=sections,
        package_hash=package_hash,
        warnings=warnings,
    )
