from pathlib import Path

from provenir.governance.audit import AuditLogger
from provenir.governance.scanners import SecretScanner


def test_governance_components_record_and_scan(tmp_path: Path) -> None:
    logger = AuditLogger(log_dir=tmp_path)
    entry = logger.log(event="train_started", actor="alice")
    assert entry["event"] == "train_started"

    scanner = SecretScanner()
    findings = scanner.scan_text("api_key=sk-test-123")
    assert findings
