"""Supply-chain scan demo — scan synthetic clean and unsafe model directories.

Demonstrates:
- Scanning a clean synthetic model directory (scan passes).
- Scanning an unsafe model directory containing a dangerous pickle (scan blocks).
- Embedding the scan result in a Model BOM and signing a Passport.
- Using scan_gate to block promotion to production.

Run with:
    python examples/supply_chain_scan_demo.py
"""

from __future__ import annotations

import io
import json
import struct
import tempfile
from pathlib import Path

from provenir.governance.bom import CodeComponent, DataComponent, EvalComponent, ModelBOM
from provenir.governance.passport import PassportSigner
from provenir.governance.scan import (
    ModelScanner,
    ScanBlocked,
    ScanComponent,
    scan_gate,
)

# ---------------------------------------------------------------------------
# Helpers to create synthetic model directories
# ---------------------------------------------------------------------------


def _make_clean_model_dir(root: Path) -> Path:
    """Create a clean synthetic model directory with a valid safetensors file."""
    model_dir = root / "clean_model"
    model_dir.mkdir()

    # Minimal valid safetensors file
    header = json.dumps({"__metadata__": {"format": "pt"}}).encode("utf-8")
    header_len = struct.pack("<Q", len(header))
    (model_dir / "model.safetensors").write_bytes(header_len + header + b"\x00" * 8)

    # Clean tokenizer config (no Jinja exec constructs)
    (model_dir / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": "{{ messages[-1].content }}", "model_type": "llama"}),
        encoding="utf-8",
    )
    # Clean config
    (model_dir / "config.json").write_text(
        json.dumps({"model_name_or_path": "./local-weights", "task": "text-generation"}),
        encoding="utf-8",
    )
    return model_dir


def _make_unsafe_model_dir(root: Path) -> Path:
    """Create an unsafe synthetic model directory containing a dangerous pickle stream."""
    model_dir = root / "unsafe_model"
    model_dir.mkdir()

    # Hand-built pickle stream with a GLOBAL opcode referencing os.system.
    # This is NEVER executed — only inspected with pickletools.genops.
    buf = io.BytesIO()
    buf.write(b"\x80\x02")          # PROTO 2
    buf.write(b"c")                 # GLOBAL opcode
    buf.write(b"os\nsystem\n")      # non-allowlisted module: os.system
    buf.write(b"U\x0becho danger")  # SHORT_BINSTRING "echo danger"
    buf.write(b"\x85")              # TUPLE1
    buf.write(b"R")                 # REDUCE
    buf.write(b".")                 # STOP
    (model_dir / "model.pt").write_bytes(buf.getvalue())

    # Config with an unpinned model reference
    (model_dir / "config.json").write_text(
        json.dumps({"model_name_or_path": "untrusted-org/mystery-model"}),
        encoding="utf-8",
    )
    return model_dir


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 60)
    print("Provenir Supply-Chain Scan Demo")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_root = Path(tmp)
        scanner = ModelScanner()

        # --- Clean model ---
        clean_dir = _make_clean_model_dir(tmp_root)
        print(f"\n[1] Scanning CLEAN model: {clean_dir.name}/")
        clean_report = scanner.scan(clean_dir)
        print(f"    {clean_report.summary()}")
        for finding in clean_report.findings:
            sev = finding.severity.value.upper()
            print(f"    [{sev}] {finding.threat.value}: {finding.detail}")

        try:
            scan_gate(clean_report)
            print("    Promotion gate: ALLOWED")
        except ScanBlocked as exc:
            print(f"    Promotion gate: BLOCKED — {exc}")

        # --- Unsafe model ---
        unsafe_dir = _make_unsafe_model_dir(tmp_root)
        print(f"\n[2] Scanning UNSAFE model: {unsafe_dir.name}/")
        unsafe_report = scanner.scan(unsafe_dir)
        print(f"    {unsafe_report.summary()}")
        for finding in unsafe_report.findings:
            sev = finding.severity.value.upper()
            print(f"    [{sev}] {finding.threat.value}: {finding.detail}")

        try:
            scan_gate(unsafe_report)
            print("    Promotion gate: ALLOWED")
        except ScanBlocked as exc:
            print(f"    Promotion gate: BLOCKED — {exc}")

        # --- Sign a passport with the clean scan embedded in the BOM ---
        print("\n[3] Embedding clean scan result in a Model BOM and signing a Passport")
        clean_sc = ScanComponent.from_report(clean_report)
        bom = ModelBOM(
            model_id="demo-model-v1",
            base_model="base-llm",
            run_id="demo-run-001",
            data=[DataComponent(name="train", content_hash="sha256:abc", num_records=50_000)],
            code=CodeComponent(git_sha="deadbeef", dependencies_hash="depshash", framework="trl"),
            evals=[EvalComponent(benchmark="mmlu", score=0.71)],
            hyperparameters={"lr": 2e-5, "epochs": 3},
            scan=clean_sc,
        )
        signer = PassportSigner(b"demo-signing-key", key_id="demo-key")
        passport = signer.sign(bom, signed_at="2026-07-05")
        valid = passport.verify(b"demo-signing-key")
        print(f"    Passport signed and verified: {valid}")
        print(f"    BOM content hash: {bom.content_hash()[:16]}...")
        print(f"    Scan verdict in BOM: unsafe={clean_sc.unsafe}")
        print(f"    Risk flags: {bom.risk_flags()}")

        # --- Show that an unsafe scan trips risk_flags ---
        print("\n[4] Attaching UNSAFE scan to BOM — risk flags check")
        unsafe_sc = ScanComponent.from_report(unsafe_report)
        unsafe_bom = ModelBOM(
            model_id="unsafe-model-v1",
            base_model="base-llm",
            run_id="demo-run-002",
            data=[DataComponent(name="train", content_hash="sha256:xyz", num_records=1_000)],
            code=CodeComponent(git_sha="badc0de", dependencies_hash="dh2", framework="trl"),
            evals=[],
            scan=unsafe_sc,
        )
        print(f"    Risk flags: {unsafe_bom.risk_flags()}")
        assert "unsafe_model_scan" in unsafe_bom.risk_flags()
        print("    'unsafe_model_scan' flag correctly raised.")

    print("\nDemo complete.")


if __name__ == "__main__":
    main()
