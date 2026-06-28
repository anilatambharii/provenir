from __future__ import annotations

import pytest

from provenir.governance.pii import PIICategory, PIIMasker, PIIScanner

# ---------------------------------------------------------------------------
# PIIScanner — detection
# ---------------------------------------------------------------------------


class TestPIIScanner:
    scanner = PIIScanner()

    # --- email ---

    def test_detects_email(self) -> None:
        findings = self.scanner.scan("Contact me at user@example.com please.")
        emails = [f for f in findings if f.category == PIICategory.EMAIL]
        assert len(emails) == 1
        assert emails[0].match == "user@example.com"

    def test_detects_subdomain_email(self) -> None:
        findings = self.scanner.scan("Send to alice@mail.company.org.")
        assert any(f.category == PIICategory.EMAIL for f in findings)

    # --- SSN ---

    def test_detects_ssn(self) -> None:
        findings = self.scanner.scan("My SSN is 123-45-6789.")
        ssns = [f for f in findings if f.category == PIICategory.SSN]
        assert len(ssns) == 1
        assert ssns[0].match == "123-45-6789"

    # --- credit card ---

    def test_detects_credit_card_hyphen(self) -> None:
        findings = self.scanner.scan("Card: 4111-1111-1111-1111")
        cards = [f for f in findings if f.category == PIICategory.CREDIT_CARD]
        assert len(cards) == 1

    def test_detects_credit_card_space(self) -> None:
        findings = self.scanner.scan("Pay with 4111 1111 1111 1111 today.")
        cards = [f for f in findings if f.category == PIICategory.CREDIT_CARD]
        assert len(cards) == 1

    # --- phone ---

    def test_detects_us_phone(self) -> None:
        findings = self.scanner.scan("Call us at 555-867-5309.")
        phones = [f for f in findings if f.category == PIICategory.PHONE]
        assert len(phones) == 1

    def test_detects_phone_with_country_code(self) -> None:
        findings = self.scanner.scan("International: +1 (800) 555-0100")
        phones = [f for f in findings if f.category == PIICategory.PHONE]
        assert len(phones) >= 1

    # --- IP address ---

    def test_detects_ip(self) -> None:
        findings = self.scanner.scan("Server at 192.168.1.1.")
        ips = [f for f in findings if f.category == PIICategory.IP_ADDRESS]
        assert len(ips) == 1
        assert ips[0].match == "192.168.1.1"

    # --- clean text ---

    def test_no_findings_on_clean_text(self) -> None:
        assert self.scanner.scan("Hello, world! No PII here.") == []

    def test_multiple_categories_in_one_string(self) -> None:
        text = "Email: foo@bar.com, SSN: 111-22-3333"
        findings = self.scanner.scan(text)
        cats = {f.category for f in findings}
        assert PIICategory.EMAIL in cats
        assert PIICategory.SSN in cats

    # --- has_pii ---

    def test_has_pii_true(self) -> None:
        assert self.scanner.has_pii("my email is x@y.com") is True

    def test_has_pii_false(self) -> None:
        assert self.scanner.has_pii("nothing sensitive here") is False

    # --- category filter ---

    def test_category_filter_restricts(self) -> None:
        scanner = PIIScanner(categories=[PIICategory.EMAIL])
        email_only_findings = scanner.scan("foo@bar.com and 123-45-6789")
        assert all(f.category == PIICategory.EMAIL for f in email_only_findings)

    # --- span positions ---

    def test_finding_positions_correct(self) -> None:
        text = "Email: test@example.com!"
        findings = self.scanner.scan(text)
        emails = [f for f in findings if f.category == PIICategory.EMAIL]
        assert len(emails) == 1
        assert text[emails[0].start : emails[0].end] == emails[0].match


# ---------------------------------------------------------------------------
# PIIMasker — placeholder strategy
# ---------------------------------------------------------------------------


class TestPIIMasker:
    masker = PIIMasker(strategy="placeholder")

    def test_masks_email(self) -> None:
        result = self.masker.mask("Email: user@example.com.")
        assert "user@example.com" not in result
        assert "[EMAIL]" in result

    def test_masks_ssn(self) -> None:
        result = self.masker.mask("SSN 123-45-6789 is mine.")
        assert "123-45-6789" not in result
        assert "[SSN]" in result

    def test_masks_credit_card(self) -> None:
        result = self.masker.mask("Pay: 4111-1111-1111-1111 now.")
        assert "4111-1111-1111-1111" not in result
        assert "[CREDIT_CARD]" in result

    def test_masks_ip(self) -> None:
        result = self.masker.mask("Host 10.0.0.1 is down.")
        assert "10.0.0.1" not in result
        assert "[IP_ADDRESS]" in result

    def test_no_pii_unchanged(self) -> None:
        text = "Hello, world!"
        assert self.masker.mask(text) == text

    def test_multiple_pii_in_sequence(self) -> None:
        text = "user@a.com and 111-22-3333"
        result = self.masker.mask(text)
        assert "[EMAIL]" in result
        assert "[SSN]" in result

    def test_mask_record_applies_to_all_values(self) -> None:
        record = {"prompt": "email is x@y.com", "response": "SSN: 000-11-2222"}
        masked = self.masker.mask_record(record)
        assert "[EMAIL]" in masked["prompt"]
        assert "[SSN]" in masked["response"]

    def test_mask_record_preserves_keys(self) -> None:
        record = {"k": "v"}
        assert self.masker.mask_record(record) == {"k": "v"}


# ---------------------------------------------------------------------------
# PIIMasker — redact strategy
# ---------------------------------------------------------------------------


class TestPIIMaskerRedact:
    masker = PIIMasker(strategy="redact")

    def test_redacts_email(self) -> None:
        result = self.masker.mask("user@example.com")
        assert result == "[REDACTED]"

    def test_redacts_ssn(self) -> None:
        result = self.masker.mask("SSN: 123-45-6789.")
        assert "[REDACTED]" in result
        assert "123-45-6789" not in result

    def test_invalid_strategy_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown strategy"):
            PIIMasker(strategy="scramble")
