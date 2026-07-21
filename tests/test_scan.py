"""Tests for provenir.governance.scan — supply-chain model scanner.

All fixtures are synthetic and safe: pickle byte streams are hand-crafted
to contain DANGEROUS opcodes that we only READ with pickletools.genops
(never executed/unpickled), GGUF-like JSON blobs with Jinja2 exec constructs
that are only matched with regex, and deliberately-malformed safetensors
headers to exercise fail-closed paths.

Test layout
-----------
- One *unsafe* fixture + one *clean* fixture per ThreatClass  (7 threat classes × 2 = 14 tests).
- Structural tests: content_hash determinism, to_dict round-trip.
- scan_gate: raises ScanBlocked iff unsafe; allow_severities downgrade.
- BOM + ScanComponent: signs/verifies; tamper detection.
- risk_flags + promotion blocking.
- Fail-closed: garbage input produces high finding, never raises.
- CLI: exit 1 on unsafe path.
"""

from __future__ import annotations

import io
import json
import pickle  # noqa: S403 – only used for opcode construction, never for loading
import struct
import textwrap
import zipfile
from pathlib import Path

import pytest

from provenir.governance.bom import CodeComponent, DataComponent, EvalComponent, ModelBOM
from provenir.governance.passport import ModelPassport, PassportSigner
from provenir.governance.scan import (
    Finding,
    ModelScanner,
    ScanBlocked,
    ScanComponent,
    ScanReport,
    Severity,
    ThreatClass,
    scan_gate,
)

# ---------------------------------------------------------------------------
# Helpers to build synthetic fixture bytes
# ---------------------------------------------------------------------------

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "scan"


def _make_pickle_reduce_unsafe() -> bytes:
    """Build a pickle byte stream with a REDUCE opcode calling os.system.

    This stream is NEVER executed — we only inspect it with pickletools.genops.
    The bytes are constructed manually so the test does not depend on pickle
    being willing to serialise a dangerous object.

    Protocol 2 pickle that encodes: os.system("echo pwned")
    Opcodes: PROTO 2, GLOBAL 'os system', STRING 'echo pwned', TUPLE1, REDUCE, STOP
    """
    buf = io.BytesIO()
    # Protocol 2 header
    buf.write(b"\x80\x02")
    # GLOBAL opcode — references os.system (non-allowlisted)
    buf.write(b"c")  # GLOBAL opcode
    buf.write(b"os\nsystem\n")
    # SHORT_BINSTRING 'echo pwned'
    buf.write(b"U")  # SHORT_BINSTRING
    buf.write(bytes([10]))  # length = 10
    buf.write(b"echo pwned")
    # TUPLE1
    buf.write(b"\x85")
    # REDUCE
    buf.write(b"R")
    # STOP
    buf.write(b".")
    return buf.getvalue()


def _make_pickle_clean() -> bytes:
    """Build a safe pickle stream (only torch-style data, no REDUCE/GLOBAL to unsafe modules)."""
    # A simple integer serialized with pickle protocol 2 — completely safe
    return pickle.dumps({"weight": [1.0, 2.0, 3.0]}, protocol=2)


def _make_safetensors_clean() -> bytes:
    """Build a minimal valid safetensors file."""
    header = json.dumps({"__metadata__": {"format": "pt"}}).encode("utf-8")
    header_len = struct.pack("<Q", len(header))
    return header_len + header + b"\x00" * 16  # dummy tensor data


def _make_safetensors_bad_length() -> bytes:
    """Build a safetensors file with a header_len that exceeds the actual data."""
    header = json.dumps({"__metadata__": {}}).encode("utf-8")
    # Declare a header_len much larger than what we actually provide
    header_len = struct.pack("<Q", len(header) + 9999)
    return header_len + header  # truncated — declared length > available data


def _make_zip_with_pickle(pickle_data: bytes) -> bytes:
    """Wrap pickle_data in a ZIP as 'archive.bin' (like PyTorch .pt files)."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("archive.bin", pickle_data)
    return buf.getvalue()


def _make_gguf_unsafe() -> bytes:
    """Return bytes resembling a GGUF file with a dangerous chat_template."""
    # GGUF magic + fake metadata with a Jinja exec construct
    magic = b"GGUF"
    meta = b'"chat_template": "{% for msg in messages %}{{ msg.content }}{% endfor %}"'
    return magic + b"\x00" * 12 + meta


def _make_gguf_clean() -> bytes:
    """Return bytes resembling a GGUF file without dangerous constructs."""
    magic = b"GGUF"
    meta = b'"chat_template": "{{ message.content }}"'
    return magic + b"\x00" * 12 + meta


def _make_tokenizer_config_unsafe() -> str:
    """Return a tokenizer_config.json text with a Jinja exec construct."""
    return json.dumps(
        {
            "chat_template": "{% for message in messages %}{{ message.content }}{% endfor %}",
            "model_type": "llama",
        }
    )


def _make_tokenizer_config_clean() -> str:
    """Return a tokenizer_config.json text without exec constructs."""
    return json.dumps(
        {
            "chat_template": "{{ messages[-1].content }}",
            "model_type": "llama",
        }
    )


def _make_keras_h5_unsafe() -> bytes:
    """Return bytes with Keras Lambda layer markers."""
    return b"\x89HDF\r\n\x1a\n" + b"\x00" * 8 + b'Lambda layer{"class_name": "Lambda"}'


def _make_keras_h5_clean() -> bytes:
    """Return bytes resembling an H5 file without Lambda markers."""
    return b"\x89HDF\r\n\x1a\n" + b"\x00" * 8 + b'{"class_name": "Dense", "units": 128}'


def _make_config_with_secret() -> str:
    """Return a config file text with an embedded AWS key pattern."""
    return textwrap.dedent("""\
        [aws]
        access_key = AKIAIOSFODNN7EXAMPLE
        region = us-east-1
    """)


def _make_config_clean() -> str:
    """Return a clean config file without secrets."""
    return textwrap.dedent("""\
        [aws]
        region = us-east-1
        profile = default
    """)


def _make_config_with_floating_ref() -> str:
    """Return a config.json with an unpinned model reference."""
    return json.dumps({"model_name_or_path": "meta-llama/Llama-3-8B", "task": "text-generation"})


def _make_config_with_pinned_ref() -> str:
    """Return a config.json with a local/clearly-safe model reference."""
    return json.dumps({"model_name_or_path": "./local-model", "task": "text-generation"})


def _make_extension_mismatch() -> bytes:
    """Return bytes that start with a non-ZIP magic (PNG) for a .pt file."""
    return b"\x89PNG\r\n\x1a\n" + b"\x00" * 100


# ---------------------------------------------------------------------------
# Fixtures — write to FIXTURES_DIR so the scanner can read real files
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session", autouse=True)
def _write_fixtures() -> None:
    """Write all synthetic fixtures to tests/fixtures/scan/."""
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

    # pickle_reduce
    (FIXTURES_DIR / "unsafe_pickle.pt").write_bytes(_make_pickle_reduce_unsafe())
    (FIXTURES_DIR / "clean_pickle.pt").write_bytes(_make_pickle_clean())

    # scanner_evasion (extension mismatch)
    (FIXTURES_DIR / "mismatch.pt").write_bytes(_make_extension_mismatch())

    # chat_template_exec
    (FIXTURES_DIR / "unsafe_tokenizer_config.json").write_text(
        _make_tokenizer_config_unsafe(), encoding="utf-8"
    )
    (FIXTURES_DIR / "clean_tokenizer_config.json").write_text(
        _make_tokenizer_config_clean(), encoding="utf-8"
    )

    # keras_lambda
    (FIXTURES_DIR / "unsafe_model.h5").write_bytes(_make_keras_h5_unsafe())
    (FIXTURES_DIR / "clean_model.h5").write_bytes(_make_keras_h5_clean())

    # weight_format (safetensors)
    (FIXTURES_DIR / "clean_model.safetensors").write_bytes(_make_safetensors_clean())
    (FIXTURES_DIR / "bad_header.safetensors").write_bytes(_make_safetensors_bad_length())

    # namespace_pin
    (FIXTURES_DIR / "unsafe_config.json").write_text(
        _make_config_with_floating_ref(), encoding="utf-8"
    )
    (FIXTURES_DIR / "clean_config.json").write_text(
        _make_config_with_pinned_ref(), encoding="utf-8"
    )

    # embedded_secret
    (FIXTURES_DIR / "unsafe.cfg").write_text(_make_config_with_secret(), encoding="utf-8")
    (FIXTURES_DIR / "clean.cfg").write_text(_make_config_clean(), encoding="utf-8")

    # gguf
    (FIXTURES_DIR / "unsafe.gguf").write_bytes(_make_gguf_unsafe())
    (FIXTURES_DIR / "clean.gguf").write_bytes(_make_gguf_clean())

    # garbage / fail-closed
    (FIXTURES_DIR / "garbage.pkl").write_bytes(b"\xff\xfe garbage bytes \x00\x01\x02")


# ---------------------------------------------------------------------------
# ThreatClass: pickle_reduce
# ---------------------------------------------------------------------------


class TestPickleReduce:
    def test_unsafe_pickle_detect_reduce(self) -> None:
        """Pickle stream with GLOBAL→os.system must produce a critical finding."""
        scanner = ModelScanner()
        report = scanner.scan(FIXTURES_DIR / "unsafe_pickle.pt")
        threats = {f.threat for f in report.findings}
        assert ThreatClass.PICKLE_REDUCE in threats
        critical_or_high = [
            f for f in report.findings
            if f.threat == ThreatClass.PICKLE_REDUCE
            and f.severity in (Severity.CRITICAL, Severity.HIGH)
        ]
        assert critical_or_high, "expected critical/high finding for GLOBAL opcode"

    def test_clean_pickle_no_reduce(self) -> None:
        """A safe pickle of plain Python data must not trigger pickle_reduce."""
        scanner = ModelScanner()
        report = scanner.scan(FIXTURES_DIR / "clean_pickle.pt")
        reduce_findings = [f for f in report.findings if f.threat == ThreatClass.PICKLE_REDUCE]
        assert not reduce_findings, f"unexpected findings: {reduce_findings}"


# ---------------------------------------------------------------------------
# ThreatClass: scanner_evasion
# ---------------------------------------------------------------------------


class TestScannerEvasion:
    def test_extension_mismatch_detected(self) -> None:
        """A PNG file with .pt extension must flag scanner_evasion."""
        scanner = ModelScanner()
        report = scanner.scan(FIXTURES_DIR / "mismatch.pt")
        evasion = [f for f in report.findings if f.threat == ThreatClass.SCANNER_EVASION]
        assert evasion, "expected scanner_evasion finding for magic-bytes mismatch"
        assert any(f.severity in (Severity.HIGH, Severity.CRITICAL) for f in evasion)

    def test_clean_pickle_no_evasion(self) -> None:
        """A legitimate .pt zip archive must not trigger scanner_evasion."""
        scanner = ModelScanner()
        # Build a clean .pt file (ZIP with a safe .bin pickle inside)
        import tempfile

        data = _make_zip_with_pickle(_make_pickle_clean())
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            f.write(data)
            tmp_path = Path(f.name)
        try:
            report = scanner.scan(tmp_path)
            evasion = [f for f in report.findings if f.threat == ThreatClass.SCANNER_EVASION]
            assert not evasion, f"unexpected evasion findings: {evasion}"
        finally:
            tmp_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# ThreatClass: chat_template_exec
# ---------------------------------------------------------------------------


class TestChatTemplateExec:
    def test_unsafe_jinja_exec_detected(self) -> None:
        """tokenizer_config.json with {%for%} must trigger chat_template_exec."""
        scanner = ModelScanner()
        report = scanner.scan(FIXTURES_DIR / "unsafe_tokenizer_config.json")
        exec_findings = [f for f in report.findings if f.threat == ThreatClass.CHAT_TEMPLATE_EXEC]
        assert exec_findings, "expected chat_template_exec finding"

    def test_clean_template_no_exec(self) -> None:
        """A safe Jinja2 template (only {{ }}) must not trigger chat_template_exec."""
        scanner = ModelScanner()
        report = scanner.scan(FIXTURES_DIR / "clean_tokenizer_config.json")
        exec_findings = [f for f in report.findings if f.threat == ThreatClass.CHAT_TEMPLATE_EXEC]
        assert not exec_findings, f"unexpected exec findings: {exec_findings}"


# ---------------------------------------------------------------------------
# ThreatClass: keras_lambda
# ---------------------------------------------------------------------------


class TestKerasLambda:
    def test_unsafe_h5_lambda_detected(self) -> None:
        """H5 file with Lambda marker must trigger keras_lambda."""
        scanner = ModelScanner()
        report = scanner.scan(FIXTURES_DIR / "unsafe_model.h5")
        lambda_findings = [f for f in report.findings if f.threat == ThreatClass.KERAS_LAMBDA]
        assert lambda_findings, "expected keras_lambda finding"

    def test_clean_h5_no_lambda(self) -> None:
        """H5 file without Lambda markers must not trigger keras_lambda."""
        scanner = ModelScanner()
        report = scanner.scan(FIXTURES_DIR / "clean_model.h5")
        lambda_findings = [f for f in report.findings if f.threat == ThreatClass.KERAS_LAMBDA]
        assert not lambda_findings, f"unexpected lambda findings: {lambda_findings}"


# ---------------------------------------------------------------------------
# ThreatClass: weight_format
# ---------------------------------------------------------------------------


class TestWeightFormat:
    def test_bad_safetensors_header_detected(self) -> None:
        """Safetensors file with bad header_len must trigger weight_format finding."""
        scanner = ModelScanner()
        report = scanner.scan(FIXTURES_DIR / "bad_header.safetensors")
        wf_findings = [f for f in report.findings if f.threat == ThreatClass.WEIGHT_FORMAT]
        assert wf_findings, "expected weight_format finding for bad safetensors header"
        assert any(f.severity in (Severity.HIGH, Severity.CRITICAL) for f in wf_findings)

    def test_clean_safetensors_no_high_finding(self) -> None:
        """Valid safetensors file must not produce high/critical weight_format findings."""
        scanner = ModelScanner()
        report = scanner.scan(FIXTURES_DIR / "clean_model.safetensors")
        high_wf = [
            f for f in report.findings
            if f.threat == ThreatClass.WEIGHT_FORMAT
            and f.severity in (Severity.HIGH, Severity.CRITICAL)
        ]
        assert not high_wf, f"unexpected high findings: {high_wf}"

    def test_pickle_weight_flagged_medium(self) -> None:
        """Pickle-format .pt file must receive a medium weight_format finding."""
        scanner = ModelScanner()
        report = scanner.scan(FIXTURES_DIR / "clean_pickle.pt")
        medium_wf = [
            f for f in report.findings
            if f.threat == ThreatClass.WEIGHT_FORMAT and f.severity == Severity.MEDIUM
        ]
        assert medium_wf, "expected medium weight_format flag for pickle .pt"


# ---------------------------------------------------------------------------
# ThreatClass: namespace_pin
# ---------------------------------------------------------------------------


class TestNamespacePin:
    def test_floating_ref_detected(self) -> None:
        """A bare HuggingFace model ref without @sha must trigger namespace_pin."""
        scanner = ModelScanner()
        report = scanner.scan(FIXTURES_DIR / "unsafe_config.json")
        ns_findings = [f for f in report.findings if f.threat == ThreatClass.NAMESPACE_PIN]
        assert ns_findings, "expected namespace_pin finding for bare model ref"

    def test_local_path_no_finding(self) -> None:
        """A local ./path ref must not trigger namespace_pin."""
        scanner = ModelScanner()
        report = scanner.scan(FIXTURES_DIR / "clean_config.json")
        ns_findings = [f for f in report.findings if f.threat == ThreatClass.NAMESPACE_PIN]
        assert not ns_findings, f"unexpected namespace_pin findings: {ns_findings}"


# ---------------------------------------------------------------------------
# ThreatClass: embedded_secret
# ---------------------------------------------------------------------------


class TestEmbeddedSecret:
    def test_aws_key_detected(self) -> None:
        """An AWS key pattern in a config file must trigger embedded_secret."""
        scanner = ModelScanner()
        report = scanner.scan(FIXTURES_DIR / "unsafe.cfg")
        sec_findings = [f for f in report.findings if f.threat == ThreatClass.EMBEDDED_SECRET]
        assert sec_findings, "expected embedded_secret finding for AWS key"
        assert any(f.severity == Severity.CRITICAL for f in sec_findings)

    def test_clean_config_no_secret(self) -> None:
        """A clean config without embedded secrets must not trigger embedded_secret."""
        scanner = ModelScanner()
        report = scanner.scan(FIXTURES_DIR / "clean.cfg")
        sec_findings = [f for f in report.findings if f.threat == ThreatClass.EMBEDDED_SECRET]
        assert not sec_findings, f"unexpected secret findings: {sec_findings}"


# ---------------------------------------------------------------------------
# Structural tests: content_hash, to_dict, summary
# ---------------------------------------------------------------------------


class TestScanReportStructural:
    def test_content_hash_deterministic(self) -> None:
        """content_hash must return the same value on repeated calls."""
        findings = [
            Finding(ThreatClass.WEIGHT_FORMAT, Severity.MEDIUM, "model.bin", "pickle weight"),
            Finding(ThreatClass.EMBEDDED_SECRET, Severity.CRITICAL, "config.json", "key found"),
        ]
        report = ScanReport(target="/model", scanned_files=2, findings=findings)
        assert report.content_hash() == report.content_hash()
        assert len(report.content_hash()) == 64

    def test_content_hash_changes_with_findings(self) -> None:
        """Adding a finding must change content_hash."""
        r1 = ScanReport(target="/m", scanned_files=1, findings=[])
        r2 = ScanReport(
            target="/m",
            scanned_files=1,
            findings=[Finding(ThreatClass.WEIGHT_FORMAT, Severity.LOW, "f", "x")],
        )
        assert r1.content_hash() != r2.content_hash()

    def test_to_dict_round_trip(self) -> None:
        """to_dict must include all expected keys and values."""
        findings = [Finding(ThreatClass.PICKLE_REDUCE, Severity.HIGH, "a.pkl", "REDUCE")]
        report = ScanReport(target="/tmp/model", scanned_files=3, findings=findings)
        d = report.to_dict()
        assert d["target"] == "/tmp/model"
        assert d["scanned_files"] == 3
        assert d["unsafe"] is True
        assert len(d["findings"]) == 1
        assert d["findings"][0]["threat"] == "pickle_reduce"
        assert d["findings"][0]["severity"] == "high"
        assert "content_hash" in d

    def test_scan_component_to_dict_round_trip(self) -> None:
        """ScanComponent.from_report → to_dict → from_dict must be identity."""
        findings = [Finding(ThreatClass.EMBEDDED_SECRET, Severity.CRITICAL, "cfg", "key")]
        report = ScanReport(target="/m", scanned_files=1, findings=findings)
        sc = ScanComponent.from_report(report)
        d = sc.to_dict()
        sc2 = ScanComponent.from_dict(d)
        assert sc2.unsafe == sc.unsafe
        assert sc2.report_hash == sc.report_hash
        assert sc2.finding_counts == sc.finding_counts

    def test_unsafe_true_on_critical(self) -> None:
        report = ScanReport(
            "/m", 1, [Finding(ThreatClass.EMBEDDED_SECRET, Severity.CRITICAL, "f", "k")]
        )
        assert report.unsafe() is True

    def test_unsafe_true_on_high(self) -> None:
        report = ScanReport(
            "/m", 1, [Finding(ThreatClass.PICKLE_REDUCE, Severity.HIGH, "f", "k")]
        )
        assert report.unsafe() is True

    def test_unsafe_false_on_medium_only(self) -> None:
        report = ScanReport(
            "/m", 1, [Finding(ThreatClass.WEIGHT_FORMAT, Severity.MEDIUM, "f", "x")]
        )
        assert report.unsafe() is False

    def test_summary_contains_verdict(self) -> None:
        report = ScanReport("/model", 5, [])
        assert "clean" in report.summary()
        unsafe_report = ScanReport(
            "/model", 5, [Finding(ThreatClass.EMBEDDED_SECRET, Severity.CRITICAL, "f", "k")]
        )
        assert "UNSAFE" in unsafe_report.summary()


# ---------------------------------------------------------------------------
# scan_gate
# ---------------------------------------------------------------------------


class TestScanGate:
    def test_clean_report_does_not_raise(self) -> None:
        report = ScanReport("/m", 0, [])
        scan_gate(report)  # must not raise

    def test_unsafe_report_raises_scan_blocked(self) -> None:
        report = ScanReport(
            "/m", 1, [Finding(ThreatClass.PICKLE_REDUCE, Severity.HIGH, "a.pkl", "REDUCE")]
        )
        with pytest.raises(ScanBlocked):
            scan_gate(report)

    def test_allow_severities_downgrades_high(self) -> None:
        """Allowing HIGH severity must prevent ScanBlocked for high-only findings."""
        report = ScanReport(
            "/m", 1, [Finding(ThreatClass.PICKLE_REDUCE, Severity.HIGH, "a.pkl", "REDUCE")]
        )
        scan_gate(report, allow_severities=frozenset({Severity.HIGH}))  # must not raise

    def test_allow_severities_still_blocks_critical(self) -> None:
        """Allowing HIGH must not prevent blocking on CRITICAL findings."""
        report = ScanReport(
            "/m",
            1,
            [Finding(ThreatClass.EMBEDDED_SECRET, Severity.CRITICAL, "cfg", "key")],
        )
        with pytest.raises(ScanBlocked):
            scan_gate(report, allow_severities=frozenset({Severity.HIGH}))


# ---------------------------------------------------------------------------
# BOM + ScanComponent integration: signing, verification, tamper detection
# ---------------------------------------------------------------------------


def _make_bom(scan: ScanComponent | None = None) -> ModelBOM:
    return ModelBOM(
        model_id="test-model",
        base_model="base",
        run_id="run-001",
        data=[DataComponent(name="d", content_hash="ch", num_records=100)],
        code=CodeComponent(git_sha="abc", dependencies_hash="dh", framework="trl"),
        evals=[EvalComponent(benchmark="mmlu", score=0.75)],
        hyperparameters={"lr": 0.001},
        scan=scan,
    )


class TestBOMScanIntegration:
    def test_bom_with_scan_signs_and_verifies(self) -> None:
        """A BOM with a ScanComponent must sign and verify successfully."""
        findings = [Finding(ThreatClass.WEIGHT_FORMAT, Severity.MEDIUM, "m.pt", "pickle")]
        report = ScanReport("/model", 2, findings)
        sc = ScanComponent.from_report(report)
        bom = _make_bom(scan=sc)
        signer = PassportSigner(b"test-secret")
        passport = signer.sign(bom, signed_at="2026-01-01")
        assert passport.verify(b"test-secret") is True

    def test_flipping_unsafe_changes_content_hash(self) -> None:
        """Changing scan.unsafe must change ModelBOM.content_hash (tamper detection)."""
        sc_clean = ScanComponent("0.7.0", "abc123", False, {"critical": 0})
        sc_dirty = ScanComponent("0.7.0", "abc123", True, {"critical": 1})
        bom_clean = _make_bom(scan=sc_clean)
        bom_dirty = _make_bom(scan=sc_dirty)
        assert bom_clean.content_hash() != bom_dirty.content_hash()

    def test_tamper_invalidates_signature(self) -> None:
        """A passport signed over a clean scan must fail verification if scan is replaced."""
        sc_clean = ScanComponent("0.7.0", "abc123", False, {})
        bom = _make_bom(scan=sc_clean)
        signer = PassportSigner(b"key")
        passport = signer.sign(bom)

        # Simulate tamper: rebuild BOM with unsafe scan but keep same attestation
        sc_dirty = ScanComponent("0.7.0", "different_hash", True, {"critical": 1})
        tampered_bom = _make_bom(scan=sc_dirty)
        tampered_passport = ModelPassport(bom=tampered_bom, attestation=passport.attestation)
        assert tampered_passport.verify(b"key") is False

    def test_passport_from_dict_with_scan(self) -> None:
        """ModelPassport.from_dict must reconstruct ScanComponent from serialised form."""
        sc = ScanComponent("0.7.0", "abc123", False, {"medium": 1})
        bom = _make_bom(scan=sc)
        signer = PassportSigner(b"k")
        passport = signer.sign(bom)
        passport2 = ModelPassport.from_dict(json.loads(passport.to_json()))
        assert passport2.bom.scan is not None
        assert passport2.bom.scan.report_hash == "abc123"
        assert passport2.verify(b"k") is True

    def test_passport_from_dict_without_scan_backward_compat(self) -> None:
        """Older passports without a scan field must load with scan=None.

        A passport that was signed with scan=None and re-loaded from a dict
        that omits the 'scan' key entirely must still verify correctly,
        because both the original and the reconstructed BOM have scan=None
        and thus identical canonical_json.
        """
        bom = _make_bom(scan=None)
        signer = PassportSigner(b"k")
        passport = signer.sign(bom)
        d = json.loads(passport.to_json())
        # Simulate old passport format: no scan key at all
        del d["bom"]["scan"]
        passport2 = ModelPassport.from_dict(d)
        assert passport2.bom.scan is None
        # Verification must succeed: scan=None reconstructed == scan=None original
        assert passport2.verify(b"k") is True

    def test_unsafe_scan_trips_risk_flags(self) -> None:
        """ModelBOM.risk_flags must include 'unsafe_model_scan' when scan.unsafe."""
        sc = ScanComponent("0.7.0", "abc", True, {"critical": 1})
        bom = _make_bom(scan=sc)
        assert "unsafe_model_scan" in bom.risk_flags()

    def test_clean_scan_no_risk_flag(self) -> None:
        """A clean ScanComponent must not add 'unsafe_model_scan' to risk_flags."""
        sc = ScanComponent("0.7.0", "abc", False, {})
        bom = _make_bom(scan=sc)
        assert "unsafe_model_scan" not in bom.risk_flags()

    def test_no_scan_no_risk_flag(self) -> None:
        """A BOM without a scan component must not add 'unsafe_model_scan'."""
        bom = _make_bom(scan=None)
        assert "unsafe_model_scan" not in bom.risk_flags()


# ---------------------------------------------------------------------------
# Fail-closed: garbage input → high finding, never an exception
# ---------------------------------------------------------------------------


class TestFailClosed:
    def test_garbage_pickle_produces_high_finding(self) -> None:
        """A garbage .pkl file must produce a high-severity finding, not raise."""
        scanner = ModelScanner()
        report = scanner.scan(FIXTURES_DIR / "garbage.pkl")
        high_or_critical = [
            f for f in report.findings
            if f.severity in (Severity.HIGH, Severity.CRITICAL)
        ]
        assert high_or_critical, "expected fail-closed high finding for garbage input"

    def test_nonexistent_target_produces_finding(self) -> None:
        """Scanning a non-existent path must return a finding, not raise."""
        scanner = ModelScanner()
        report = scanner.scan("/nonexistent/path/does/not/exist")
        assert len(report.findings) > 0

    def test_scan_empty_directory(self) -> None:
        """Scanning an empty directory must return a clean report without raising."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            scanner = ModelScanner()
            report = scanner.scan(tmp)
            assert report.scanned_files == 0
            assert isinstance(report.findings, list)

    def test_scan_returns_scanreport_not_raises(self) -> None:
        """scan() must never raise; always return a ScanReport."""
        scanner = ModelScanner()
        # Even on a path full of garbage, scan() must succeed
        report = scanner.scan(FIXTURES_DIR / "garbage.pkl")
        assert isinstance(report, ScanReport)


# ---------------------------------------------------------------------------
# GGUF threat
# ---------------------------------------------------------------------------


class TestGGUF:
    def test_gguf_unsafe_chat_template(self) -> None:
        """GGUF file with Jinja exec in metadata must trigger chat_template_exec."""
        scanner = ModelScanner()
        report = scanner.scan(FIXTURES_DIR / "unsafe.gguf")
        exec_findings = [f for f in report.findings if f.threat == ThreatClass.CHAT_TEMPLATE_EXEC]
        assert exec_findings, "expected chat_template_exec finding in GGUF"

    def test_gguf_clean_chat_template(self) -> None:
        """GGUF file with a safe template must not trigger chat_template_exec."""
        scanner = ModelScanner()
        report = scanner.scan(FIXTURES_DIR / "clean.gguf")
        exec_findings = [f for f in report.findings if f.threat == ThreatClass.CHAT_TEMPLATE_EXEC]
        assert not exec_findings, f"unexpected findings: {exec_findings}"
