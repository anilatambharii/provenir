"""HTML report generator for a completed ProvenirRun."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from provenir.integrations.wrapper import ProvenirRun


@dataclass
class RunReport:
    run_id: str
    base_model: str
    hacking_rate: float
    is_clean: bool
    anomaly_count: int
    health_verdict: str
    evals: list[dict[str, Any]]
    risk_flags: list[str]
    flight_summary: dict[str, Any]
    hacking_by_kind: dict[str, int]
    lineage_nodes: list[dict[str, Any]]
    partial: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_run_dir(cls, run_dir: Path) -> "RunReport":
        """Load from the JSON artifacts written by ProvenirRun on exit."""
        manifest_file = next((run_dir / "manifests").glob("*.json"), None)
        manifest: dict[str, Any] = {}
        if manifest_file and manifest_file.exists():
            manifest = json.loads(manifest_file.read_text(encoding="utf-8"))

        flight_file = run_dir / "flight_recorder.json"
        flight: dict[str, Any] = (
            json.loads(flight_file.read_text(encoding="utf-8")) if flight_file.exists() else {}
        )

        hack_file = run_dir / "hacking_report.json"
        hack: dict[str, Any] = (
            json.loads(hack_file.read_text(encoding="utf-8")) if hack_file.exists() else {}
        )

        bom_file = run_dir / "bom.json"
        bom: dict[str, Any] = (
            json.loads(bom_file.read_text(encoding="utf-8")) if bom_file.exists() else {}
        )

        lineage_file = run_dir / "lineage.json"
        lineage: dict[str, Any] = (
            json.loads(lineage_file.read_text(encoding="utf-8")) if lineage_file.exists() else {}
        )

        provenance: dict[str, Any] = manifest.get("provenance", {})

        # Derive risk flags from BOM data components
        risk_flags: list[str] = []
        for comp in bom.get("data", []):
            if not comp.get("pii_scanned", True):
                risk_flags.append(f"unscanned_pii:{comp.get('name', '?')}")
            if comp.get("license", "CC") in ("unknown", ""):
                risk_flags.append(f"unknown_license:{comp.get('name', '?')}")
        for ev in bom.get("evals", []):
            if ev.get("contaminated", False):
                risk_flags.append(f"contaminated_eval:{ev.get('benchmark', '?')}")

        flight_summary: dict[str, Any] = flight.get("summary", flight)

        return cls(
            run_id=manifest.get("run_id", run_dir.name),
            base_model=provenance.get("base_model", bom.get("base_model", "unknown")),
            hacking_rate=hack.get("hacking_rate", 0.0),
            is_clean=hack.get("is_clean", True),
            anomaly_count=flight_summary.get("num_anomalies", 0),
            health_verdict=flight_summary.get("verdict", "unknown"),
            evals=bom.get("evals", []),
            risk_flags=risk_flags,
            flight_summary=flight_summary,
            hacking_by_kind=hack.get("by_kind", {}),
            lineage_nodes=list(lineage.get("nodes", [])),
            partial=provenance.get("partial", False),
        )

    @classmethod
    def from_run(cls, run: "ProvenirRun") -> "RunReport":
        """Build directly from a finalized ProvenirRun (no disk read needed)."""
        from provenir.governance.bom import ModelBOM

        bom: ModelBOM | None = run.bom
        risk_flags: list[str] = bom.risk_flags() if bom is not None else []
        hr = run.hacking_report
        fs = run.flight_recorder.summary()
        manifest = run.manifest
        evals: list[dict[str, Any]] = []
        if bom is not None:
            evals = [
                {
                    "benchmark": e.benchmark,
                    "score": e.score,
                    "contaminated": e.contaminated,
                }
                for e in bom.evals
            ]
        lineage_nodes: list[dict[str, Any]] = []
        if run.lineage is not None:
            lineage_nodes = list(run.lineage.to_dict().get("nodes", []))
        return cls(
            run_id=manifest.run_id if manifest else "unknown",
            base_model=run.config.base_model,
            hacking_rate=hr.hacking_rate if hr else 0.0,
            is_clean=hr.is_clean if hr else True,
            anomaly_count=fs.get("num_anomalies", 0),
            health_verdict=fs.get("verdict", "unknown"),
            evals=evals,
            risk_flags=risk_flags,
            flight_summary=fs,
            hacking_by_kind=hr.by_kind() if hr else {},
            lineage_nodes=lineage_nodes,
        )

    def to_html(self) -> str:
        """Return a self-contained HTML report string."""
        health_color = {
            "healthy": "#22c55e",
            "warning": "#f59e0b",
            "critical": "#ef4444",
        }.get(self.health_verdict.lower(), "#6b7280")
        hack_color = "#22c55e" if self.is_clean else "#ef4444"
        hack_label = (
            "CLEAN" if self.is_clean else f"HACKING DETECTED ({self.hacking_rate:.1%})"
        )

        risk_html = (
            "".join(f'<li style="color:#ef4444">&#9888; {f}</li>' for f in self.risk_flags)
            or "<li style='color:#22c55e'>&#10003; No risk flags</li>"
        )

        eval_rows = "".join(
            f"<tr><td>{e.get('benchmark', '?')}</td><td>{e.get('score', 0):.3f}</td>"
            f"<td>{'&#9888; contaminated' if e.get('contaminated') else '&#10003; clean'}</td></tr>"
            for e in self.evals
        ) or "<tr><td colspan='3'>No evals recorded</td></tr>"

        hack_rows = "".join(
            f"<tr><td>{kind}</td><td>{count}</td></tr>"
            for kind, count in self.hacking_by_kind.items()
        ) or "<tr><td colspan='2'>No hacking signals</td></tr>"

        lineage_rows = "".join(
            f"<tr><td>{n.get('node_id', '?')}</td><td>{n.get('node_type', '?')}</td>"
            f"<td style='font-family:monospace;font-size:0.8em'>"
            f"{str(n.get('content_hash', ''))[:12]}&#8230;</td></tr>"
            for n in self.lineage_nodes
        ) or "<tr><td colspan='3'>No lineage recorded</td></tr>"

        partial_banner = (
            '<div style="background:#f59e0b;color:#fff;padding:8px;margin-bottom:16px;'
            'border-radius:6px">&#9888; Partial run &#8212; exited with exception</div>'
            if self.partial
            else ""
        )

        return (
            f"<!DOCTYPE html>\n"
            f"<html lang=\"en\">\n"
            f"<head>\n"
            f"<meta charset=\"UTF-8\">\n"
            f"<title>Provenir Run Report &#8212; {self.run_id}</title>\n"
            f"<style>\n"
            f"body{{font-family:system-ui,sans-serif;max-width:960px;margin:0 auto;"
            f"padding:24px;background:#f9fafb;color:#111}}\n"
            f"h1{{font-size:1.5rem;margin-bottom:4px}}\n"
            f".badge{{display:inline-block;padding:4px 10px;border-radius:999px;"
            f"font-size:0.85rem;font-weight:600;color:#fff}}\n"
            f".card{{background:#fff;border-radius:10px;padding:20px;margin-bottom:20px;"
            f"box-shadow:0 1px 4px rgba(0,0,0,.08)}}\n"
            f"table{{width:100%;border-collapse:collapse;font-size:0.9rem}}\n"
            f"th{{text-align:left;padding:8px;background:#f3f4f6;"
            f"border-bottom:2px solid #e5e7eb}}\n"
            f"td{{padding:8px;border-bottom:1px solid #f3f4f6}}\n"
            f"ul{{margin:8px 0;padding-left:20px}}\n"
            f".metric{{display:inline-block;margin-right:24px;margin-bottom:8px}}\n"
            f".metric .label{{font-size:0.75rem;color:#6b7280;text-transform:uppercase;"
            f"letter-spacing:.05em}}\n"
            f".metric .value{{font-size:1.5rem;font-weight:700}}\n"
            f"</style>\n"
            f"</head>\n"
            f"<body>\n"
            f"<h1>Provenir Run Report</h1>\n"
            f"<p style=\"color:#6b7280;margin-top:0\">Run: <code>{self.run_id}</code> "
            f"&middot; Base model: <code>{self.base_model}</code></p>\n"
            f"{partial_banner}\n"
            f"\n"
            f"<div class=\"card\">\n"
            f"<h2 style=\"margin-top:0\">Summary</h2>\n"
            f"<div class=\"metric\"><div class=\"label\">Health</div>"
            f"<div class=\"value\"><span class=\"badge\" style=\"background:{health_color}\">"
            f"{self.health_verdict.upper()}</span></div></div>\n"
            f"<div class=\"metric\"><div class=\"label\">Reward Hacking</div>"
            f"<div class=\"value\"><span class=\"badge\" style=\"background:{hack_color}\">"
            f"{hack_label}</span></div></div>\n"
            f"<div class=\"metric\"><div class=\"label\">Anomalies</div>"
            f"<div class=\"value\">{self.anomaly_count}</div></div>\n"
            f"<div class=\"metric\"><div class=\"label\">Evals Recorded</div>"
            f"<div class=\"value\">{len(self.evals)}</div></div>\n"
            f"</div>\n"
            f"\n"
            f"<div class=\"card\">\n"
            f"<h2 style=\"margin-top:0\">Risk Flags</h2>\n"
            f"<ul>{risk_html}</ul>\n"
            f"</div>\n"
            f"\n"
            f"<div class=\"card\">\n"
            f"<h2 style=\"margin-top:0\">Eval Results</h2>\n"
            f"<table><thead><tr><th>Benchmark</th><th>Score</th>"
            f"<th>Contamination</th></tr></thead>\n"
            f"<tbody>{eval_rows}</tbody></table>\n"
            f"</div>\n"
            f"\n"
            f"<div class=\"card\">\n"
            f"<h2 style=\"margin-top:0\">Reward Hacking Signals by Kind</h2>\n"
            f"<table><thead><tr><th>Kind</th><th>Count</th></tr></thead>\n"
            f"<tbody>{hack_rows}</tbody></table>\n"
            f"</div>\n"
            f"\n"
            f"<div class=\"card\">\n"
            f"<h2 style=\"margin-top:0\">Lineage Nodes</h2>\n"
            f"<table><thead><tr><th>Node ID</th><th>Type</th>"
            f"<th>Content Hash</th></tr></thead>\n"
            f"<tbody>{lineage_rows}</tbody></table>\n"
            f"</div>\n"
            f"\n"
            f"<div class=\"card\">\n"
            f"<h2 style=\"margin-top:0\">Flight Recorder Summary</h2>\n"
            f"<pre style=\"background:#f3f4f6;padding:12px;border-radius:6px;"
            f"overflow:auto;font-size:0.85rem\">"
            f"{json.dumps(self.flight_summary, indent=2)}</pre>\n"
            f"</div>\n"
            f"\n"
            f"<footer style=\"color:#9ca3af;font-size:0.8rem;text-align:center;"
            f"margin-top:32px\">Generated by Provenir v0.4.0</footer>\n"
            f"</body>\n"
            f"</html>"
        )

    def save(self, path: Path) -> Path:
        """Write the HTML report to ``path`` (creates parents)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_html(), encoding="utf-8")
        return path


class ReportGenerator:
    """Convenience wrapper: build a report from a dir or a live run."""

    @staticmethod
    def from_run_dir(run_dir: Path) -> RunReport:
        return RunReport.from_run_dir(run_dir)

    @staticmethod
    def from_run(run: "ProvenirRun") -> RunReport:
        return RunReport.from_run(run)
