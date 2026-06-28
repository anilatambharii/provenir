from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any


class PIICategory(str, Enum):
    EMAIL = "email"
    PHONE = "phone"
    SSN = "ssn"
    CREDIT_CARD = "credit_card"
    IP_ADDRESS = "ip_address"


# Patterns keyed by category.  Ordered from most-specific to least-specific so
# credit-card patterns don't shadow phone patterns on overlap.
_PATTERNS: dict[PIICategory, str] = {
    PIICategory.EMAIL: r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}",
    PIICategory.SSN: r"\b\d{3}-\d{2}-\d{4}\b",
    PIICategory.CREDIT_CARD: r"\b(?:\d{4}[- ]){3}\d{4}\b",
    PIICategory.PHONE: (
        r"\b(?:\+?1[-.\s]?)?"
        r"(?:\(?\d{3}\)?[-.\s]?)"
        r"\d{3}[-.\s]?\d{4}\b"
    ),
    PIICategory.IP_ADDRESS: r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
}

_PLACEHOLDERS: dict[PIICategory, str] = {
    PIICategory.EMAIL: "[EMAIL]",
    PIICategory.PHONE: "[PHONE]",
    PIICategory.SSN: "[SSN]",
    PIICategory.CREDIT_CARD: "[CREDIT_CARD]",
    PIICategory.IP_ADDRESS: "[IP_ADDRESS]",
}


@dataclass(frozen=True)
class PIIFinding:
    category: PIICategory
    match: str
    start: int
    end: int


class PIIScanner:
    """Heuristic regex scanner for personally identifiable information.

    By default all categories are enabled; pass *categories* to restrict.
    """

    def __init__(self, categories: list[PIICategory] | None = None) -> None:
        active = set(categories) if categories is not None else set(PIICategory)
        self._compiled = {
            cat: re.compile(_PATTERNS[cat])
            for cat in PIICategory
            if cat in active
        }

    def scan(self, text: str) -> list[PIIFinding]:
        findings: list[PIIFinding] = []
        for cat, pattern in self._compiled.items():
            for m in pattern.finditer(text):
                findings.append(
                    PIIFinding(
                        category=cat,
                        match=m.group(0),
                        start=m.start(),
                        end=m.end(),
                    )
                )
        return findings

    def has_pii(self, text: str) -> bool:
        return any(True for _ in self._iter_matches(text))

    def _iter_matches(self, text: str) -> Any:
        for pattern in self._compiled.values():
            yield from pattern.finditer(text)


class PIIMasker:
    """Replace PII spans with category placeholders or a uniform redaction token.

    *strategy*:
      - ``"placeholder"`` (default) — replaces each span with its category label
        e.g. ``[EMAIL]``, ``[SSN]``
      - ``"redact"`` — replaces every span with ``[REDACTED]``
    """

    def __init__(
        self,
        strategy: str = "placeholder",
        categories: list[PIICategory] | None = None,
    ) -> None:
        if strategy not in {"placeholder", "redact"}:
            raise ValueError(f"unknown strategy {strategy!r}; expected 'placeholder' or 'redact'")
        self.strategy = strategy
        self._scanner = PIIScanner(categories=categories)

    def mask(self, text: str) -> str:
        findings = self._scanner.scan(text)
        if not findings:
            return text
        # Sort by start position descending so replacements don't shift offsets.
        findings_sorted = sorted(findings, key=lambda f: f.start, reverse=True)
        chars = list(text)
        for finding in findings_sorted:
            replacement = (
                "[REDACTED]"
                if self.strategy == "redact"
                else _PLACEHOLDERS[finding.category]
            )
            chars[finding.start : finding.end] = list(replacement)
        return "".join(chars)

    def mask_record(self, record: dict[str, str]) -> dict[str, str]:
        """Apply masking to every string value in *record*."""
        return {k: self.mask(v) if isinstance(v, str) else v for k, v in record.items()}
