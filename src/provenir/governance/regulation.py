"""EU AI Act Art. 53 / Annex IV Evidence Generator.

Turns a signed :class:`~provenir.governance.passport.ModelPassport` into:

1. **Art. 53 / Annex XII training-data summary** — the mandatory AI Office template
   structure (Markdown + structured data), including domain-volume rollup when
   crawl metadata is present.
2. **Annex IV technical-documentation skeleton** — sections 1-9 auto-filled from the
   BOM where data exists; explicit ``MISSING: <what to supply>`` markers elsewhere.
3. A **coverage report** — which fields are satisfied, partial, or missing.

Hard honesty constraint: no value is fabricated. Absent source => ``Coverage.MISSING``
with an actionable note, or the literal sentinel for missing crawl data. Fabricated
compliance evidence is worse than a marked gap.

Example:
    >>> from provenir.governance.bom import (
    ...     CodeComponent, DataComponent, EvalComponent, ModelBOM
    ... )
    >>> from provenir.governance.passport import ModelPassport
    >>> bom = ModelBOM(
    ...     model_id="m1", base_model="llama3", run_id="r1",
    ...     data=[DataComponent(name="d", content_hash="h", num_records=100)],
    ...     code=CodeComponent(git_sha="s", dependencies_hash="dh", framework="trl"),
    ...     evals=[EvalComponent(benchmark="mmlu", score=0.71)],
    ... )
    >>> passport = ModelPassport(bom=bom, attestation=None)
    >>> gen = RegulationGenerator()
    >>> report = gen.art53_training_data_summary(passport)
    >>> report.artifact
    'art53_training_data_summary'
    >>> 0.0 <= report.coverage_score() <= 1.0
    True
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any

from provenir.governance.passport import ModelPassport
from provenir.governance.templates import (
    ANNEX_IV_TEMPLATE,
    ART53_TEMPLATE,
    FDA_PCCP_TEMPLATE,
    NIST_AI_RMF_TEMPLATE,
)

# Sentinel string emitted when crawl-domain data is absent — never invent a value.
_NO_CRAWL_DATA = "domain breakdown unavailable — supply crawl manifest"


class Coverage(str, Enum):
    """Coverage status for a single evidence field."""

    SATISFIED = "satisfied"
    PARTIAL = "partial"
    MISSING = "missing"


@dataclass(frozen=True)
class FieldCoverage:
    """Coverage verdict for a single Annex IV / Art. 53 field.

    Example:
        >>> fc = FieldCoverage("annex_iv.2c", Coverage.SATISFIED, "Data BOM present")
        >>> fc.to_dict()["coverage"]
        'satisfied'
    """

    field_id: str
    coverage: Coverage
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "field_id": self.field_id,
            "coverage": self.coverage.value,
            "note": self.note,
        }


@dataclass(frozen=True)
class EvidenceReport:
    """A generated compliance-evidence artifact plus its coverage metadata.

    Attributes:
        artifact: Identifier — ``"art53_training_data_summary"`` or
            ``"annex_iv_technical_file"``.
        markdown: The generated document in human-readable Markdown.
        data: Structured form of every derived value — designed as the
            intermediate model for a future OSCAL serialiser.
        fields: Per-field coverage verdicts.

    Example:
        >>> fc = FieldCoverage("x.1", Coverage.MISSING, "supply it")
        >>> r = EvidenceReport("test", "# md", {"k": "v"}, [fc])
        >>> r.coverage_score()
        0.0
        >>> r.missing()
        [FieldCoverage(field_id='x.1', coverage=<Coverage.MISSING: 'missing'>, note='supply it')]
    """

    artifact: str
    markdown: str
    data: dict[str, Any]
    fields: list[FieldCoverage]

    def coverage_score(self) -> float:
        """Return fraction of fields with ``Coverage.SATISFIED`` status."""
        if not self.fields:
            return 0.0
        satisfied = sum(1 for f in self.fields if f.coverage == Coverage.SATISFIED)
        return satisfied / len(self.fields)

    def missing(self) -> list[FieldCoverage]:
        """Return fields whose coverage is ``Coverage.MISSING``."""
        return [f for f in self.fields if f.coverage == Coverage.MISSING]

    def summary(self) -> str:
        """Return a one-line human-readable coverage summary."""
        pct = self.coverage_score() * 100
        satisfied = sum(1 for f in self.fields if f.coverage == Coverage.SATISFIED)
        partial = sum(1 for f in self.fields if f.coverage == Coverage.PARTIAL)
        missing = sum(1 for f in self.fields if f.coverage == Coverage.MISSING)
        return (
            f"{self.artifact}: {pct:.1f}% coverage "
            f"({satisfied} satisfied, {partial} partial, {missing} missing)"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact": self.artifact,
            "coverage_score": self.coverage_score(),
            "fields": [f.to_dict() for f in self.fields],
            "data": self.data,
        }


class RegulationGenerator:
    """Generate EU AI Act Art. 53 and Annex IV evidence from a signed passport.

    Args:
        sme: When ``True``, apply the SME threshold of 5 % instead of 10 %
            for the domain-volume rollup in the Art. 53 summary.

    Example:
        >>> gen = RegulationGenerator(sme=True)
        >>> gen._domain_threshold_pct
        5
    """

    def __init__(self, *, sme: bool = False) -> None:
        self._sme = sme
        self._domain_threshold_pct: int = 5 if sme else 10

    # ------------------------------------------------------------------
    # Art. 53 / Annex XII training-data summary
    # ------------------------------------------------------------------

    def art53_training_data_summary(self, passport: ModelPassport) -> EvidenceReport:
        """Generate an Art. 53 / Annex XII training-data summary from *passport*.

        Returns an :class:`EvidenceReport` whose ``markdown`` follows the AI Office
        mandatory template structure and whose ``data`` dict is the structured
        intermediate form.

        Never fabricates a value: absent crawl data → ``_NO_CRAWL_DATA`` sentinel;
        absent optout field → ``Coverage.MISSING`` with an actionable note.
        """
        bom = passport.bom
        fields: list[FieldCoverage] = []

        # --- model identity ---
        model_name = bom.model_id or "MISSING"
        base_model = bom.base_model or "MISSING"
        run_id = bom.run_id or "MISSING"
        generated_at = bom.created_at or "unknown"
        bom_hash = bom.content_hash()

        id_coverage = (
            Coverage.SATISFIED
            if (bom.model_id and bom.base_model and bom.run_id)
            else Coverage.MISSING
        )
        fields.append(
            FieldCoverage(
                "art53.model_identity",
                id_coverage,
                "" if id_coverage == Coverage.SATISFIED else "Provide model_id, base_model, run_id",
            )
        )

        # --- provider (not stored in BOM; always partial) ---
        fields.append(
            FieldCoverage(
                "art53.provider",
                Coverage.MISSING,
                "Supply provider / organisation name in the BOM or config",
            )
        )

        # --- total records ---
        total_records = sum(d.num_records for d in bom.data)
        records_coverage = Coverage.SATISFIED if bom.data else Coverage.MISSING
        fields.append(
            FieldCoverage(
                "art53.dataset_size",
                records_coverage,
                "" if records_coverage == Coverage.SATISFIED else "No data components in BOM",
            )
        )

        # --- modalities (from hyperparameters) ---
        modalities_raw: Any = bom.hyperparameters.get("modalities")
        if modalities_raw:
            modalities = str(modalities_raw)
            modalities_coverage = Coverage.SATISFIED
        else:
            modalities = "MISSING"
            modalities_coverage = Coverage.MISSING
        fields.append(
            FieldCoverage(
                "art53.modalities",
                modalities_coverage,
                "" if modalities_coverage == Coverage.SATISFIED
                else "Add 'modalities' key to BOM hyperparameters (e.g. 'text', 'text,image')",
            )
        )

        # --- source categories ---
        source_cats: dict[str, int] = {}
        for d in bom.data:
            cat = d.source_category or "unknown"
            source_cats[cat] = source_cats.get(cat, 0) + d.num_records
        source_cat_coverage = (
            Coverage.SATISFIED
            if any(c != "unknown" for c in source_cats)
            else Coverage.PARTIAL
        )
        fields.append(
            FieldCoverage(
                "art53.source_categories",
                source_cat_coverage,
                "" if source_cat_coverage == Coverage.SATISFIED
                else "Set DataComponent.source_category on each component "
                     "(e.g. 'public', 'licensed', 'scraped', 'synthetic', 'user')",
            )
        )

        # --- domain rollup ---
        domain_data, domain_coverage = self._build_domain_section(bom.data)
        fields.append(
            FieldCoverage(
                "art53.top_domains",
                domain_coverage,
                "" if domain_coverage == Coverage.SATISFIED
                else "Populate DataComponent.crawl_domains with {domain: record_count} mappings",
            )
        )

        # --- optout ---
        optout_values = [d.optout_respected for d in bom.data if d.optout_respected is not None]
        if optout_values:
            all_respected = all(optout_values)
            if all_respected:
                optout_str = "Yes — all crawled components respect robots.txt/TDM opt-out"
            else:
                optout_str = "Partial — some components do not fully respect opt-out signals"
            optout_coverage = Coverage.SATISFIED if all_respected else Coverage.PARTIAL
        else:
            optout_str = "MISSING"
            optout_coverage = Coverage.MISSING
        fields.append(
            FieldCoverage(
                "art53.optout_respected",
                optout_coverage,
                "" if optout_coverage != Coverage.MISSING
                else "Set DataComponent.optout_respected=True/False on each scraped component",
            )
        )

        # --- PII / contamination ---
        pii_all = all(d.pii_scanned for d in bom.data) if bom.data else False
        contam_all = all(d.contamination_checked for d in bom.data) if bom.data else False
        pii_cov = Coverage.SATISFIED if (pii_all and contam_all) else (
            Coverage.PARTIAL if (pii_all or contam_all) else Coverage.MISSING
        )
        fields.append(
            FieldCoverage(
                "art53.pii_contamination",
                pii_cov,
                "" if pii_cov == Coverage.SATISFIED
                else "Ensure pii_scanned=True and contamination_checked=True on all DataComponents",
            )
        )

        # --- attestation ---
        att = passport.attestation
        attest_algorithm = att.algorithm if att else "MISSING"
        attest_key_id = att.key_id if att else "MISSING"
        attest_signed_at = att.signed_at if att else "MISSING"
        attest_signature = att.signature[:16] + "..." if att else "MISSING"
        attest_cov = Coverage.SATISFIED if att else Coverage.MISSING
        fields.append(
            FieldCoverage(
                "art53.attestation",
                attest_cov,
                "" if attest_cov == Coverage.SATISFIED
                else "Sign the passport with PassportSigner before generating the evidence report",
            )
        )

        # --- render sections ---
        source_cats_table = _render_kv_table(
            source_cats,
            headers=("Source category", "Record count"),
        ) if source_cats else "_No data components._"

        pii_lines = []
        for d in bom.data:
            pii_lines.append(
                f"- **{d.name}**: PII scanned={d.pii_scanned}, "
                f"contamination checked={d.contamination_checked}"
            )
        pii_section = "\n".join(pii_lines) if pii_lines else "_No data components._"

        optout_section = optout_str

        # --- structured data ---
        data: dict[str, Any] = {
            "model_id": model_name,
            "base_model": base_model,
            "run_id": run_id,
            "generated_at": generated_at,
            "bom_hash": bom_hash,
            "total_records": total_records,
            "num_components": len(bom.data),
            "modalities": modalities,
            "source_categories": source_cats,
            "domain_rollup": domain_data,
            "optout_respected": optout_str,
            "pii_scanned_all": pii_all,
            "contamination_checked_all": contam_all,
            "attestation": {
                "algorithm": attest_algorithm,
                "key_id": attest_key_id,
                "signed_at": attest_signed_at,
            },
        }

        md = ART53_TEMPLATE.format(
            model_name=model_name,
            provider="MISSING — supply provider/organisation name",
            base_model=base_model,
            run_id=run_id,
            generated_at=generated_at,
            bom_hash=bom_hash,
            total_records=total_records,
            num_components=len(bom.data),
            modalities=modalities,
            source_categories_table=source_cats_table,
            domain_threshold_pct=self._domain_threshold_pct,
            domain_section=domain_data["markdown"],
            optout_section=optout_section,
            pii_section=pii_section,
            attest_algorithm=attest_algorithm,
            attest_key_id=attest_key_id,
            attest_signed_at=attest_signed_at,
            attest_signature=attest_signature,
        )

        return EvidenceReport(
            artifact="art53_training_data_summary",
            markdown=md,
            data=data,
            fields=fields,
        )

    # ------------------------------------------------------------------
    # Annex IV technical file
    # ------------------------------------------------------------------

    def annex_iv_technical_file(self, passport: ModelPassport) -> EvidenceReport:
        """Generate an Annex IV technical-documentation skeleton from *passport*.

        Sections 2(c) data provenance and 2(d) validation/testing are auto-filled
        from the BOM (these are Provenir's core strength). Sections 4, 7, 8, 9
        are organisational and are explicitly marked MISSING with actionable notes.
        """
        bom = passport.bom
        fields: list[FieldCoverage] = []

        model_name = bom.model_id or "MISSING"
        base_model = bom.base_model or "MISSING"
        run_id = bom.run_id or "MISSING"
        generated_at = bom.created_at or "unknown"
        bom_hash = bom.content_hash()

        att = passport.attestation

        # §1 general description
        if bom.model_id and bom.base_model:
            s1_text = (
                f"**Model ID:** {bom.model_id}\n\n"
                f"**Base model:** {bom.base_model}\n\n"
                f"**Run ID:** {bom.run_id}\n\n"
                f"**Created at:** {bom.created_at or 'unknown'}\n\n"
                "MISSING: Intended purpose / use-case description — "
                "add 'intended_purpose' to BOM hyperparameters or supply separately."
            )
            s1_cov = Coverage.PARTIAL
            s1_note = "Model identity present; intended purpose not in BOM"
        else:
            s1_text = "MISSING: model_id and base_model are absent from the BOM"
            s1_cov = Coverage.MISSING
            s1_note = "Populate model_id and base_model in the BOM"
        fields.append(FieldCoverage("annex_iv.1.general", s1_cov, s1_note))

        # §2(a) development methods
        code = bom.code
        s2a_text = (
            f"**Framework:** {code.framework}\n\n"
            f"**Git SHA:** `{code.git_sha}`\n\n"
            f"**Dependencies hash:** `{code.dependencies_hash}`\n\n"
            "MISSING: Detailed training methodology description — "
            "add 'training_methodology' to BOM hyperparameters or supply separately."
        )
        s2a_cov = Coverage.PARTIAL
        s2a_note = "Code provenance present; methodology description absent"
        fields.append(FieldCoverage("annex_iv.2a.development", s2a_cov, s2a_note))

        # §2(b) design specifications
        if bom.hyperparameters:
            hp_lines = [f"- **{k}:** {v}" for k, v in sorted(bom.hyperparameters.items())]
            s2b_text = "\n".join(hp_lines)
            s2b_cov = Coverage.PARTIAL
            s2b_note = "Hyperparameters present; full design specification may be incomplete"
        else:
            s2b_text = "MISSING: No hyperparameters recorded in the BOM"
            s2b_cov = Coverage.MISSING
            s2b_note = "Add hyperparameters to the BOM"
        fields.append(FieldCoverage("annex_iv.2b.design_specs", s2b_cov, s2b_note))

        # §2(c) data provenance — Provenir's strength → SATISFIED when data present
        if bom.data:
            data_lines = []
            for d in bom.data:
                data_lines.append(
                    f"- **{d.name}**: {d.num_records} records, "
                    f"license={d.license}, "
                    f"category={d.source_category}, "
                    f"pii_scanned={d.pii_scanned}, "
                    f"contamination_checked={d.contamination_checked}, "
                    f"hash=`{d.content_hash}`"
                )
            s2c_text = "\n".join(data_lines)
            s2c_text += (
                "\n\nSee the Art. 53 training-data summary for the full domain rollup "
                "and copyright opt-out evidence."
            )
            s2c_cov = Coverage.SATISFIED
            s2c_note = ""
        else:
            s2c_text = "MISSING: No data components in the BOM"
            s2c_cov = Coverage.MISSING
            s2c_note = "Add DataComponent entries to the BOM"
        fields.append(FieldCoverage("annex_iv.2c.data_provenance", s2c_cov, s2c_note))

        # §2(d) validation/testing — SATISFIED when evals present
        if bom.evals:
            eval_lines = []
            for e in bom.evals:
                marker = " *(contaminated)*" if e.contaminated else ""
                eval_lines.append(f"- **{e.benchmark}:** {e.score:.4f}{marker}")
            s2d_text = "\n".join(eval_lines)
            if bom.scan is not None:
                s2d_text += f"\n\n**Supply-chain scan:** unsafe={bom.scan.unsafe}"
            if bom.reward_validity is not None:
                s2d_text += (
                    f"\n\n**Reward validity:** validity={bom.reward_validity.validity:.3f}, "
                    f"spurious={bom.reward_validity.spurious}"
                )
            s2d_cov = Coverage.SATISFIED
            s2d_note = ""
        else:
            s2d_text = "MISSING: No evaluation results in the BOM"
            s2d_cov = Coverage.MISSING
            s2d_note = "Add EvalComponent entries to the BOM"
        fields.append(FieldCoverage("annex_iv.2d.validation", s2d_cov, s2d_note))

        # §3 monitoring/logging
        s3_text = (
            "Provenir AuditLogger records all governance events (passport_issued, "
            "train_started, …) to an append-only JSONL audit log.\n\n"
            "MISSING: Production runtime-monitoring configuration (metrics, alerts, "
            "drift detection) — supply separately as part of your MLOps runbook."
        )
        s3_cov = Coverage.PARTIAL
        s3_note = "Audit log present; production monitoring config absent"
        fields.append(FieldCoverage("annex_iv.3.monitoring", s3_cov, s3_note))

        # §4 risk management — organisational, cannot be derived
        s4_text = (
            "MISSING: Organisational risk-management process documentation.\n\n"
            "Required by Annex IV §4: describe the risk-management system, "
            "risk identification process, mitigation measures, and residual risk "
            "acceptance criteria. This cannot be derived from the BOM; supply it "
            "as a separate document and reference it here."
        )
        s4_cov = Coverage.MISSING
        s4_note = "Supply risk-management process document (Annex IV §4)"
        fields.append(FieldCoverage("annex_iv.4.risk_management", s4_cov, s4_note))

        # §5 changes over lifecycle
        if att and att.signed_at:
            s5_text = (
                f"Passport signed at: {att.signed_at}\n\n"
                "The signed passport provides a contemporaneous, tamper-evident record "
                "of the model's BOM at the time of signing.\n\n"
                "MISSING: Change log for subsequent updates — append a new signed passport "
                "for each model revision and link them here."
            )
            s5_cov = Coverage.PARTIAL
            s5_note = "Signed passport present; change log for future revisions absent"
        else:
            s5_text = (
                "MISSING: No signed attestation present. Sign the passport with "
                "PassportSigner to establish a contemporaneous record."
            )
            s5_cov = Coverage.MISSING
            s5_note = "Sign the passport to establish contemporaneous record"
        fields.append(FieldCoverage("annex_iv.5.lifecycle_changes", s5_cov, s5_note))

        # §6 standards applied
        s6_text = (
            "- **CycloneDX ML BOM** — BOM structure follows CycloneDX ML extension concepts\n"
            "- **ISO/IEC 42001:2023** — AI management system alignment (planned)\n"
            "- **EU AI Act (Regulation 2024/1689)** — Art. 53, Annex IV, Annex XII\n\n"
            "MISSING: Formal conformity mapping to harmonised standards — "
            "supply a standards-crosswalk document referencing your specific implementations."
        )
        s6_cov = Coverage.PARTIAL
        s6_note = "Provenir references standards; formal conformity mapping absent"
        fields.append(FieldCoverage("annex_iv.6.standards", s6_cov, s6_note))

        # §7 post-market monitoring
        s7_text = (
            "MISSING: Post-market monitoring plan.\n\n"
            "Required by Annex IV §7: describe how the system is monitored after deployment, "
            "including performance metrics, incident reporting, and feedback loops. "
            "Cannot be derived from the BOM; supply separately."
        )
        s7_cov = Coverage.MISSING
        s7_note = "Supply post-market monitoring plan (Annex IV §7)"
        fields.append(FieldCoverage("annex_iv.7.post_market", s7_cov, s7_note))

        # §8 human oversight
        s8_text = (
            "MISSING: Human oversight design documentation.\n\n"
            "Required by Annex IV §8: describe the human-oversight mechanisms — "
            "who can intervene, how, and under what conditions. "
            "Cannot be derived from the BOM; supply separately."
        )
        s8_cov = Coverage.MISSING
        s8_note = "Supply human oversight design (Annex IV §8)"
        fields.append(FieldCoverage("annex_iv.8.human_oversight", s8_cov, s8_note))

        # §9 conformity assessment
        s9_text = (
            "MISSING: Conformity assessment documentation.\n\n"
            "Required by Annex IV §9: reference the conformity assessment procedure followed "
            "(internal check or third-party notified body), the EU declaration of conformity, "
            "and CE marking information. Cannot be derived from the BOM; supply separately."
        )
        s9_cov = Coverage.MISSING
        s9_note = "Supply conformity assessment documentation (Annex IV §9)"
        fields.append(FieldCoverage("annex_iv.9.conformity", s9_cov, s9_note))

        # --- coverage table ---
        cov_rows = []
        for fc in fields:
            cov_rows.append(f"| {fc.field_id} | {fc.coverage.value} | {fc.note or '—'} |")
        cov_table = (
            "| Field | Coverage | Note |\n"
            "|---|---|---|\n"
            + "\n".join(cov_rows)
        )

        satisfied_count = sum(1 for f in fields if f.coverage == Coverage.SATISFIED)
        total_count = len(fields)
        score = satisfied_count / total_count if total_count else 0.0

        md = ANNEX_IV_TEMPLATE.format(
            model_name=model_name,
            base_model=base_model,
            run_id=run_id,
            generated_at=generated_at,
            bom_hash=bom_hash,
            section1=s1_text,
            section2a=s2a_text,
            section2b=s2b_text,
            section2c=s2c_text,
            section2d=s2d_text,
            section3=s3_text,
            section4=s4_text,
            section5=s5_text,
            section6=s6_text,
            section7=s7_text,
            section8=s8_text,
            section9=s9_text,
            coverage_score=score,
            satisfied_count=satisfied_count,
            total_count=total_count,
            coverage_table=cov_table,
        )

        data: dict[str, Any] = {
            "model_id": model_name,
            "base_model": base_model,
            "run_id": run_id,
            "generated_at": generated_at,
            "bom_hash": bom_hash,
            "sections": {
                "1_general": s1_text,
                "2a_development": s2a_text,
                "2b_design_specs": s2b_text,
                "2c_data_provenance": s2c_text,
                "2d_validation": s2d_text,
                "3_monitoring": s3_text,
                "4_risk_management": s4_text,
                "5_lifecycle_changes": s5_text,
                "6_standards": s6_text,
                "7_post_market": s7_text,
                "8_human_oversight": s8_text,
                "9_conformity": s9_text,
            },
        }

        return EvidenceReport(
            artifact="annex_iv_technical_file",
            markdown=md,
            data=data,
            fields=fields,
        )

    # ------------------------------------------------------------------
    # FDA PCCP — AI/ML SaMD Predetermined Change Control Plan
    # ------------------------------------------------------------------

    def fda_pccp_summary(self, passport: ModelPassport) -> EvidenceReport:
        """Generate an FDA PCCP evidence summary for AI/ML SaMD from *passport*.

        Maps BOM fields to the eight PCCP sections required by FDA guidance on
        Predetermined Change Control Plans for AI/ML-based Software as a Medical
        Device. Absent data → explicit ``MISSING`` markers; no fabrication.
        """
        bom = passport.bom
        fields: list[FieldCoverage] = []

        model_name = bom.model_id or "MISSING"
        base_model = bom.base_model or "MISSING"
        run_id = bom.run_id or "MISSING"
        generated_at = bom.created_at or "unknown"
        bom_hash = bom.content_hash()

        # pccp.device_description — model_id + base_model
        if bom.model_id and bom.base_model:
            device_desc_text = (
                f"**Model ID:** {bom.model_id}\n\n"
                f"**Base model:** {bom.base_model}\n\n"
                f"**Run ID:** {bom.run_id}"
            )
            device_desc_cov = Coverage.SATISFIED
            device_desc_note = ""
        else:
            device_desc_text = "MISSING: Supply model_id and base_model in the BOM"
            device_desc_cov = Coverage.MISSING
            device_desc_note = "Provide model_id and base_model in the BOM"
        fields.append(
            FieldCoverage("pccp.device_description", device_desc_cov, device_desc_note)
        )

        # pccp.samd_function — hyperparameters["samd_function"]
        samd_func_val: Any = bom.hyperparameters.get("samd_function")
        if samd_func_val:
            samd_func_text = f"**SaMD Function:** {samd_func_val}"
            samd_func_cov = Coverage.SATISFIED
            samd_func_note = ""
        else:
            samd_func_text = (
                "MISSING: supply samd_function in hyperparameters "
                "(e.g. 'diagnostic_support', 'treatment_recommendation')"
            )
            samd_func_cov = Coverage.MISSING
            samd_func_note = "supply samd_function in hyperparameters"
        fields.append(
            FieldCoverage("pccp.samd_function", samd_func_cov, samd_func_note)
        )

        # pccp.training_data_documentation — data components + pii_scanned
        if bom.data:
            pii_all = all(d.pii_scanned for d in bom.data)
            data_lines = [
                f"- **{d.name}**: {d.num_records:,} records, "
                f"license={d.license}, pii_scanned={d.pii_scanned}"
                for d in bom.data
            ]
            train_doc_text = "\n".join(data_lines)
            if pii_all:
                train_doc_text += "\n\nAll data components have been PII-scanned."
                train_doc_cov = Coverage.SATISFIED
                train_doc_note = ""
            else:
                train_doc_text += (
                    "\n\nPARTIAL: Not all data components have been PII-scanned."
                )
                train_doc_cov = Coverage.PARTIAL
                train_doc_note = "Not all data components have pii_scanned=True"
        else:
            train_doc_text = "MISSING: No data components in the BOM"
            train_doc_cov = Coverage.MISSING
            train_doc_note = "Add DataComponent entries to the BOM"
        fields.append(
            FieldCoverage(
                "pccp.training_data_documentation", train_doc_cov, train_doc_note
            )
        )

        # pccp.performance_testing — evals
        if bom.evals:
            eval_lines = [
                f"- **{e.benchmark}:** {e.score:.4f}"
                + (" *(contaminated)*" if e.contaminated else "")
                for e in bom.evals
            ]
            perf_test_text = "\n".join(eval_lines)
            perf_test_cov = Coverage.SATISFIED
            perf_test_note = ""
        else:
            perf_test_text = "MISSING: No evaluation results in the BOM"
            perf_test_cov = Coverage.MISSING
            perf_test_note = "Add EvalComponent entries to the BOM"
        fields.append(
            FieldCoverage("pccp.performance_testing", perf_test_cov, perf_test_note)
        )

        # pccp.real_world_monitoring — reward_validity
        if bom.reward_validity is not None:
            rwm_text = (
                f"**Reward name:** {bom.reward_validity.reward_name}\n\n"
                f"**Validity score:** {bom.reward_validity.validity:.3f}\n\n"
                f"**Spurious:** {bom.reward_validity.spurious}\n\n"
                "PARTIAL: Reward validity evidence provides training signal quality. "
                "Post-deployment real-world monitoring must be configured separately."
            )
            rwm_cov = Coverage.PARTIAL
            rwm_note = "Reward validity present; post-deployment monitoring is org process"
        else:
            rwm_text = (
                "MISSING: No reward validity data in the BOM. "
                "Supply post-deployment real-world performance monitoring plan."
            )
            rwm_cov = Coverage.MISSING
            rwm_note = "Supply reward_validity in BOM or post-deployment monitoring plan"
        fields.append(
            FieldCoverage("pccp.real_world_monitoring", rwm_cov, rwm_note)
        )

        # pccp.impact_assessment — scan
        if bom.scan is not None:
            impact_text = (
                f"**Supply-chain scan result:** unsafe={bom.scan.unsafe}\n\n"
                f"**Finding counts:** {bom.scan.finding_counts}\n\n"
                "PARTIAL: Scan evidence present; full clinical/patient impact assessment "
                "must be completed and documented separately."
            )
            impact_cov = Coverage.PARTIAL
            impact_note = "Scan present; full impact assessment requires organisational input"
        else:
            impact_text = (
                "MISSING: No supply-chain scan in the BOM. "
                "Complete a supply-chain scan and clinical impact assessment."
            )
            impact_cov = Coverage.MISSING
            impact_note = "Run a supply-chain scan and add to BOM"
        fields.append(
            FieldCoverage("pccp.impact_assessment", impact_cov, impact_note)
        )

        # pccp.change_description — hyperparameters
        if bom.hyperparameters:
            hp_lines = [
                f"- **{k}:** {v}" for k, v in sorted(bom.hyperparameters.items())
            ]
            change_desc_text = (
                "Training hyperparameters (basis for anticipated change scope):\n\n"
                + "\n".join(hp_lines)
            )
            change_desc_cov = Coverage.SATISFIED
            change_desc_note = ""
        else:
            change_desc_text = (
                "MISSING: No hyperparameters in the BOM. "
                "Record hyperparameters to describe anticipated changes."
            )
            change_desc_cov = Coverage.MISSING
            change_desc_note = "Add hyperparameters to the BOM"
        fields.append(
            FieldCoverage("pccp.change_description", change_desc_cov, change_desc_note)
        )

        # pccp.version_control — code.git_sha
        if bom.code.git_sha:
            vc_text = (
                f"**Git SHA:** `{bom.code.git_sha}`\n\n"
                f"**Framework:** {bom.code.framework}\n\n"
                f"**Dependencies hash:** `{bom.code.dependencies_hash}`"
            )
            vc_cov = Coverage.SATISFIED
            vc_note = ""
        else:
            vc_text = "MISSING: No git_sha in CodeComponent — supply version-controlled code"
            vc_cov = Coverage.MISSING
            vc_note = "Populate git_sha in CodeComponent"
        fields.append(
            FieldCoverage("pccp.version_control", vc_cov, vc_note)
        )

        # --- coverage table ---
        cov_rows = [
            f"| {fc.field_id} | {fc.coverage.value} | {fc.note or '—'} |"
            for fc in fields
        ]
        cov_table = (
            "| Field | Coverage | Note |\n"
            "|---|---|---|\n"
            + "\n".join(cov_rows)
        )

        satisfied_count = sum(1 for f in fields if f.coverage == Coverage.SATISFIED)
        total_count = len(fields)
        score = satisfied_count / total_count if total_count else 0.0

        data: dict[str, Any] = {
            "model_id": model_name,
            "base_model": base_model,
            "run_id": run_id,
            "generated_at": generated_at,
            "bom_hash": bom_hash,
            "pccp.device_description": device_desc_cov.value,
            "pccp.samd_function": samd_func_cov.value,
            "pccp.training_data_documentation": train_doc_cov.value,
            "pccp.performance_testing": perf_test_cov.value,
            "pccp.real_world_monitoring": rwm_cov.value,
            "pccp.impact_assessment": impact_cov.value,
            "pccp.change_description": change_desc_cov.value,
            "pccp.version_control": vc_cov.value,
        }

        md = FDA_PCCP_TEMPLATE.format(
            model_name=model_name,
            base_model=base_model,
            run_id=run_id,
            generated_at=generated_at,
            bom_hash=bom_hash,
            device_description=device_desc_text,
            samd_function=samd_func_text,
            training_data_documentation=train_doc_text,
            performance_testing=perf_test_text,
            real_world_monitoring=rwm_text,
            impact_assessment=impact_text,
            change_description=change_desc_text,
            version_control=vc_text,
            coverage_score=score,
            satisfied_count=satisfied_count,
            total_count=total_count,
            coverage_table=cov_table,
        )

        return EvidenceReport(
            artifact="fda_pccp",
            markdown=md,
            data=data,
            fields=fields,
        )

    # ------------------------------------------------------------------
    # NIST AI RMF 1.0 — Four core functions
    # ------------------------------------------------------------------

    def nist_ai_rmf_summary(self, passport: ModelPassport) -> EvidenceReport:
        """Generate a NIST AI RMF 1.0 evidence summary from *passport*.

        Maps BOM fields to the four AI RMF core functions (GOVERN, MAP, MEASURE,
        MANAGE) and their sub-categories. Absent or organisational data is marked
        PARTIAL or MISSING with actionable notes; no value is fabricated.
        """
        bom = passport.bom
        fields: list[FieldCoverage] = []

        model_name = bom.model_id or "MISSING"
        base_model = bom.base_model or "MISSING"
        run_id = bom.run_id or "MISSING"
        generated_at = bom.created_at or "unknown"
        bom_hash = bom.content_hash()

        att = passport.attestation

        # GOVERN.1 — Policies & accountability (attestation)
        if att is not None:
            if att.signature:
                g1_text = (
                    f"**Attestation algorithm:** {att.algorithm}\n\n"
                    f"**Key ID:** {att.key_id}\n\n"
                    f"**Signed at:** {att.signed_at}\n\n"
                    "Signed passport provides a tamper-evident accountability record."
                )
                g1_cov = Coverage.SATISFIED
                g1_note = ""
            else:
                g1_text = (
                    "PARTIAL: Attestation present but unsigned. "
                    "Sign the passport with PassportSigner to establish full accountability."
                )
                g1_cov = Coverage.PARTIAL
                g1_note = "Attestation present but unsigned"
        else:
            g1_text = (
                "PARTIAL: No attestation present. "
                "Sign the passport with PassportSigner to establish accountability."
            )
            g1_cov = Coverage.PARTIAL
            g1_note = "No attestation — sign with PassportSigner"
        fields.append(FieldCoverage("GOVERN.1", g1_cov, g1_note))

        # GOVERN.2 — Organizational practices (created_at as proxy)
        g2_text = (
            f"**BOM created at:** {bom.created_at or 'unknown'}\n\n"
            "PARTIAL: BOM timestamp evidences that a governance process exists. "
            "Full organisational AI policy documentation must be supplied separately."
        )
        g2_cov = Coverage.PARTIAL
        g2_note = "Org process — supply AI policy documentation separately"
        fields.append(FieldCoverage("GOVERN.2", g2_cov, g2_note))

        # MAP.1 — Risk identification (scan)
        if bom.scan is not None:
            m1_text = (
                f"**Supply-chain scan:** unsafe={bom.scan.unsafe}\n\n"
                f"**Finding counts:** {bom.scan.finding_counts}"
            )
            m1_cov = Coverage.SATISFIED
            m1_note = ""
        else:
            m1_text = (
                "PARTIAL: No supply-chain scan in BOM. "
                "Run a supply-chain scan to identify model risks."
            )
            m1_cov = Coverage.PARTIAL
            m1_note = "Run a supply-chain scan and attach to BOM"
        fields.append(FieldCoverage("MAP.1", m1_cov, m1_note))

        # MAP.2 — Scientific basis (git_sha + framework)
        if bom.code.git_sha and bom.code.framework:
            m2_text = (
                f"**Git SHA:** `{bom.code.git_sha}`\n\n"
                f"**Framework:** {bom.code.framework}\n\n"
                f"**Dependencies hash:** `{bom.code.dependencies_hash}`"
            )
            m2_cov = Coverage.SATISFIED
            m2_note = ""
        else:
            m2_text = (
                "MISSING: git_sha or framework absent from CodeComponent. "
                "Populate CodeComponent for reproducibility evidence."
            )
            m2_cov = Coverage.MISSING
            m2_note = "Populate git_sha and framework in CodeComponent"
        fields.append(FieldCoverage("MAP.2", m2_cov, m2_note))

        # MAP.3 — Impact assessment (risk_flags)
        risk_flags = bom.risk_flags()
        if not risk_flags:
            m3_text = "No risk flags detected from BOM analysis."
            m3_cov = Coverage.SATISFIED
            m3_note = ""
        else:
            flags_list = "\n".join(f"- `{f}`" for f in risk_flags)
            m3_text = (
                f"PARTIAL: The following risk flags are present:\n\n{flags_list}\n\n"
                "Address these flags before completing the impact assessment."
            )
            m3_cov = Coverage.PARTIAL
            m3_note = f"Risk flags present: {', '.join(risk_flags)}"
        fields.append(FieldCoverage("MAP.3", m3_cov, m3_note))

        # MEASURE.1 — Performance testing (evals)
        if bom.evals:
            eval_lines = [
                f"- **{e.benchmark}:** {e.score:.4f}"
                + (" *(contaminated)*" if e.contaminated else "")
                for e in bom.evals
            ]
            meas1_text = "\n".join(eval_lines)
            meas1_cov = Coverage.SATISFIED
            meas1_note = ""
        else:
            meas1_text = "MISSING: No evaluation results in the BOM"
            meas1_cov = Coverage.MISSING
            meas1_note = "Add EvalComponent entries to the BOM"
        fields.append(FieldCoverage("MEASURE.1", meas1_cov, meas1_note))

        # MEASURE.2 — Bias/fairness (hyperparameters["fairness_eval"])
        fairness_val: Any = bom.hyperparameters.get("fairness_eval")
        if fairness_val:
            meas2_text = f"**Fairness evaluation:** {fairness_val}"
            meas2_cov = Coverage.SATISFIED
            meas2_note = ""
        else:
            meas2_text = (
                "MISSING: No fairness_eval in hyperparameters. "
                "Add 'fairness_eval' to BOM hyperparameters with evaluation method/results."
            )
            meas2_cov = Coverage.MISSING
            meas2_note = "Add 'fairness_eval' to BOM hyperparameters"
        fields.append(FieldCoverage("MEASURE.2", meas2_cov, meas2_note))

        # MEASURE.3 — Reward validity
        if bom.reward_validity is not None:
            if not bom.reward_validity.spurious:
                meas3_text = (
                    f"**Reward name:** {bom.reward_validity.reward_name}\n\n"
                    f"**Validity score:** {bom.reward_validity.validity:.3f}\n\n"
                    "**Spurious:** False — reward signal validated as genuine."
                )
                meas3_cov = Coverage.SATISFIED
                meas3_note = ""
            else:
                meas3_text = (
                    f"**Reward name:** {bom.reward_validity.reward_name}\n\n"
                    f"**Validity score:** {bom.reward_validity.validity:.3f}\n\n"
                    "PARTIAL: **Spurious: True** — reward signal flagged as spurious. "
                    "Investigate and remediate before deployment."
                )
                meas3_cov = Coverage.PARTIAL
                meas3_note = "Reward flagged as spurious — investigate before deployment"
        else:
            meas3_text = (
                "MISSING: No reward validity data in the BOM. "
                "Run a reward-validity ablation and attach results."
            )
            meas3_cov = Coverage.MISSING
            meas3_note = "Run reward-validity ablation and add to BOM"
        fields.append(FieldCoverage("MEASURE.3", meas3_cov, meas3_note))

        # MANAGE.1 — Incident response (AuditLogger present, org process)
        man1_text = (
            "PARTIAL: Provenir AuditLogger records all governance events "
            "(passport_issued, train_started, …) to an append-only JSONL audit log. "
            "Organisational incident response procedures must be documented separately."
        )
        man1_cov = Coverage.PARTIAL
        man1_note = "Audit log present; org incident response procedures required separately"
        fields.append(FieldCoverage("MANAGE.1", man1_cov, man1_note))

        # MANAGE.2 — Model lifecycle (run_id + created_at)
        if bom.run_id and bom.created_at:
            man2_text = (
                f"**Run ID:** {bom.run_id}\n\n"
                f"**Created at:** {bom.created_at}\n\n"
                "Run ID and creation timestamp provide lifecycle traceability."
            )
            man2_cov = Coverage.SATISFIED
            man2_note = ""
        else:
            man2_text = (
                "MISSING: run_id or created_at absent. "
                "Populate both fields for full model lifecycle traceability."
            )
            man2_cov = Coverage.MISSING
            man2_note = "Provide run_id and created_at in the BOM"
        fields.append(FieldCoverage("MANAGE.2", man2_cov, man2_note))

        # --- coverage table ---
        cov_rows = [
            f"| {fc.field_id} | {fc.coverage.value} | {fc.note or '—'} |"
            for fc in fields
        ]
        cov_table = (
            "| Function | Coverage | Note |\n"
            "|---|---|---|\n"
            + "\n".join(cov_rows)
        )

        satisfied_count = sum(1 for f in fields if f.coverage == Coverage.SATISFIED)
        total_count = len(fields)
        score = satisfied_count / total_count if total_count else 0.0

        data: dict[str, Any] = {
            "model_id": model_name,
            "base_model": base_model,
            "run_id": run_id,
            "generated_at": generated_at,
            "bom_hash": bom_hash,
            "GOVERN.1": g1_cov.value,
            "GOVERN.2": g2_cov.value,
            "MAP.1": m1_cov.value,
            "MAP.2": m2_cov.value,
            "MAP.3": m3_cov.value,
            "MEASURE.1": meas1_cov.value,
            "MEASURE.2": meas2_cov.value,
            "MEASURE.3": meas3_cov.value,
            "MANAGE.1": man1_cov.value,
            "MANAGE.2": man2_cov.value,
        }

        md = NIST_AI_RMF_TEMPLATE.format(
            model_name=model_name,
            base_model=base_model,
            run_id=run_id,
            generated_at=generated_at,
            bom_hash=bom_hash,
            govern_1=g1_text,
            govern_2=g2_text,
            map_1=m1_text,
            map_2=m2_text,
            map_3=m3_text,
            measure_1=meas1_text,
            measure_2=meas2_text,
            measure_3=meas3_text,
            manage_1=man1_text,
            manage_2=man2_text,
            coverage_score=score,
            satisfied_count=satisfied_count,
            total_count=total_count,
            coverage_table=cov_table,
        )

        return EvidenceReport(
            artifact="nist_ai_rmf",
            markdown=md,
            data=data,
            fields=fields,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_domain_section(
        self,
        data_components: list[Any],
    ) -> tuple[dict[str, Any], Coverage]:
        """Aggregate crawl_domains across all data components and pick top-N%.

        Returns a (structured_data_dict, coverage) pair. If no component has
        crawl_domains the sentinel string is used — no fabrication.
        """
        merged: dict[str, int] = {}
        for dc in data_components:
            for domain, count in dc.crawl_domains.items():
                merged[domain] = merged.get(domain, 0) + count

        if not merged:
            return (
                {
                    "available": False,
                    "markdown": _NO_CRAWL_DATA,
                    "domains": {},
                    "threshold_pct": self._domain_threshold_pct,
                },
                Coverage.MISSING,
            )

        # Sort descending by count for determinism, secondary sort by domain name
        sorted_domains = sorted(merged.items(), key=lambda kv: (-kv[1], kv[0]))
        total = sum(merged.values())
        # Pick top-N% by cumulative volume
        threshold_vol = math.ceil(total * self._domain_threshold_pct / 100)
        top_domains: list[tuple[str, int]] = []
        cumulative = 0
        for domain, count in sorted_domains:
            top_domains.append((domain, count))
            cumulative += count
            if cumulative >= threshold_vol:
                break

        md_rows = [f"| {d} | {c:,} | {c / total:.1%} |" for d, c in top_domains]
        md_table = (
            f"Top {self._domain_threshold_pct}% of crawled volume "
            f"({len(top_domains)} domain(s) of {len(merged)} total, "
            f"{total:,} total records):\n\n"
            "| Domain | Records | % of crawl |\n"
            "|---|---|---|\n"
            + "\n".join(md_rows)
        )

        return (
            {
                "available": True,
                "markdown": md_table,
                "domains": dict(top_domains),
                "total_crawl_records": total,
                "threshold_pct": self._domain_threshold_pct,
            },
            Coverage.SATISFIED,
        )


# ------------------------------------------------------------------
# Private rendering helpers
# ------------------------------------------------------------------

def _render_kv_table(
    mapping: dict[str, Any],
    headers: tuple[str, str] = ("Key", "Value"),
) -> str:
    """Render a two-column Markdown table from a dict, sorted by key."""
    k_hdr, v_hdr = headers
    rows = [f"| {k} | {v} |" for k, v in sorted(mapping.items())]
    return f"| {k_hdr} | {v_hdr} |\n|---|---|\n" + "\n".join(rows)
