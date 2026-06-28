# Security Policy

## Supported Versions

| Version | Supported |
|---|---|
| 0.2.x | âœ“ |
| 0.1.x | Security fixes only |
| < 0.1 | Not supported |

---

## Reporting a Vulnerability

**Do not open a public GitHub issue for security vulnerabilities.**

Please report security issues privately using one of the following methods:

1. **GitHub private vulnerability reporting** â€” click "Report a vulnerability"
   on the [Security tab](https://github.com/anilatambharii/provenir/security/advisories/new)
   of the repository.

2. **Email** â€” send a description to the maintainers (see `pyproject.toml`
   for contact information). Encrypt with our PGP key if the disclosure is
   sensitive.

Include:

- A description of the vulnerability and its impact
- Steps to reproduce (or a minimal proof of concept)
- Any suggested mitigations

We will acknowledge receipt within 48 hours and aim to publish a fix or
mitigation within 14 days of confirmed vulnerabilities.

---

## Scope

The following are in scope for security reports:

- Arbitrary code execution via malicious dataset files or configs
- Path traversal in file I/O (manifests, audit logs, adapter paths)
- Credential leakage through the audit log or REST API
- Injection attacks via the CLI or REST API
- Insecure defaults that silently disable PII scanning or audit logging

The following are **out of scope**:

- Vulnerabilities in optional dependencies (torch, TRL, etc.) â€” report
  those to the respective upstream projects
- Security issues in user-supplied model weights or datasets
- Denial of service via very large inputs (not a threat model for this project)

---

## Security Design Notes

Provenir is designed with the following security properties:

- **Audit log** â€” append-only JSONL; never deletes or overwrites entries
- **PII scanning** â€” built into the core package; opt-out requires explicit code change
- **Secret scanning** â€” dataset secrets are detected before training starts
- **Manifest integrity** â€” SHA-256 hashes on config and dataset files; tampering is detectable
- **No network calls in core** â€” the base `pip install provenir` package makes no outbound network requests
- **Optional dependencies** â€” network-capable features (Hub, judges) are installed separately and make no calls unless explicitly invoked

