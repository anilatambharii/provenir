from __future__ import annotations

import re
from typing import Any


class SecretScanner:
    """A lightweight heuristic scanner for obvious secrets in text."""

    def __init__(self, patterns: list[str] | None = None) -> None:
        self.patterns = patterns or [r"sk-[A-Za-z0-9]+", r"api[_-]?key\s*[:=]\s*.+"]

    def scan_text(self, text: str) -> list[dict[str, Any]]:
        findings: list[dict[str, Any]] = []
        for pattern in self.patterns:
            for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                findings.append({"pattern": pattern, "match": match.group(0)})
        return findings
