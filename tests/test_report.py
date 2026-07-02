"""Tests for the provenir.report HTML report generator."""
from __future__ import annotations

import tempfile
from pathlib import Path

from provenir.integrations.wrapper import track
from provenir.report import ReportGenerator, RunReport


def _make_run(tmpdir: str) -> None:
    """Run a minimal ProvenirRun and finalize it to disk."""
    with track("report-test-run", output_dir=tmpdir, base_model="test-model") as run:
        run.log_step({"step": 0, "kl": 0.02, "entropy": 1.4})
        run.log_step({"step": 1, "kl": 0.03, "entropy": 1.3})
        run.record_eval("gsm8k", score=0.71)


class TestRunReportFromRunDir:
    def test_report_from_run_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_run(tmpdir)
            run_dir = Path(tmpdir)
            report = RunReport.from_run_dir(run_dir)
            assert report.run_id == "report-test-run"
            assert report.health_verdict != ""
            html = report.to_html()
            assert html.startswith("<!DOCTYPE html>")
            assert "report-test-run" in html

    def test_report_save_writes_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_run(tmpdir)
            run_dir = Path(tmpdir)
            report = RunReport.from_run_dir(run_dir)
            out_path = run_dir / "report.html"
            result = report.save(out_path)
            assert result == out_path
            assert out_path.exists()
            content = out_path.read_text(encoding="utf-8")
            assert content.startswith("<!DOCTYPE html>")

    def test_report_run_id_is_set(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_run(tmpdir)
            report = RunReport.from_run_dir(Path(tmpdir))
            assert report.run_id != ""

    def test_report_health_verdict_is_set(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_run(tmpdir)
            report = RunReport.from_run_dir(Path(tmpdir))
            assert report.health_verdict in ("HEALTHY", "DEGRADED", "CRITICAL", "unknown")

    def test_report_missing_optional_files(self) -> None:
        """from_run_dir should not crash when optional JSON files are absent."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Don't run anything; the directory is empty
            run_dir = Path(tmpdir)
            report = RunReport.from_run_dir(run_dir)
            assert report.run_id == run_dir.name  # falls back to dir name
            html = report.to_html()
            assert html.startswith("<!DOCTYPE html>")


class TestRunReportFromRun:
    def test_report_from_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with track(
                "live-run", output_dir=tmpdir, base_model="live-model"
            ) as run:
                run.log_step({"step": 0, "kl": 0.01})
                run.record_eval("hellaswag", score=0.82)

            report = RunReport.from_run(run)
            assert report.run_id == "live-run"
            assert report.base_model == "live-model"
            assert len(report.evals) == 1
            assert report.evals[0]["benchmark"] == "hellaswag"
            html = report.to_html()
            assert html.startswith("<!DOCTYPE html>")

    def test_report_from_run_health_verdict(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with track("hv-run", output_dir=tmpdir) as run:
                run.log_step({"step": 0, "kl": 0.01})

            report = RunReport.from_run(run)
            assert report.health_verdict in ("HEALTHY", "DEGRADED", "CRITICAL", "unknown")

    def test_report_from_run_lineage_nodes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with track("lineage-run", output_dir=tmpdir) as run:
                run.log_step({"step": 0})

            report = RunReport.from_run(run)
            assert isinstance(report.lineage_nodes, list)


class TestRunReportWithRiskFlags:
    def test_report_with_risk_flags_in_html(self) -> None:
        report = RunReport(
            run_id="risk-run",
            base_model="test",
            hacking_rate=0.0,
            is_clean=True,
            anomaly_count=0,
            health_verdict="healthy",
            evals=[],
            risk_flags=["unscanned_pii:train", "unknown_license:val"],
            flight_summary={},
            hacking_by_kind={},
            lineage_nodes=[],
        )
        html = report.to_html()
        assert "unscanned_pii:train" in html
        assert "unknown_license:val" in html

    def test_report_no_risk_flags_shows_clean_message(self) -> None:
        report = RunReport(
            run_id="clean-run",
            base_model="test",
            hacking_rate=0.0,
            is_clean=True,
            anomaly_count=0,
            health_verdict="healthy",
            evals=[],
            risk_flags=[],
            flight_summary={},
            hacking_by_kind={},
            lineage_nodes=[],
        )
        html = report.to_html()
        assert "No risk flags" in html

    def test_report_partial_banner_shown(self) -> None:
        report = RunReport(
            run_id="partial-run",
            base_model="test",
            hacking_rate=0.0,
            is_clean=True,
            anomaly_count=0,
            health_verdict="unknown",
            evals=[],
            risk_flags=[],
            flight_summary={},
            hacking_by_kind={},
            lineage_nodes=[],
            partial=True,
        )
        html = report.to_html()
        assert "Partial run" in html

    def test_report_hacking_detected_label(self) -> None:
        report = RunReport(
            run_id="hack-run",
            base_model="test",
            hacking_rate=0.35,
            is_clean=False,
            anomaly_count=2,
            health_verdict="critical",
            evals=[],
            risk_flags=[],
            flight_summary={},
            hacking_by_kind={"shortcut": 5},
            lineage_nodes=[],
        )
        html = report.to_html()
        assert "HACKING DETECTED" in html


class TestReportGeneratorMethods:
    def test_report_generator_from_run_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_run(tmpdir)
            report = ReportGenerator.from_run_dir(Path(tmpdir))
            assert isinstance(report, RunReport)
            assert report.run_id == "report-test-run"

    def test_report_generator_from_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with track("gen-run", output_dir=tmpdir) as run:
                run.log_step({"step": 0})

            report = ReportGenerator.from_run(run)
            assert isinstance(report, RunReport)
            assert report.run_id == "gen-run"

    def test_report_generator_from_run_dir_returns_runreport(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            report = ReportGenerator.from_run_dir(Path(tmpdir))
            assert isinstance(report, RunReport)
