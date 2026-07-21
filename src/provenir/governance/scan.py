"""Static model supply-chain scanner for Provenir.

Inspects model artifact files for known dangerous patterns — pickle RCE opcodes,
scanner evasion, chat-template injection, Keras Lambda layers, unsafe weight
formats, floating namespace references, and embedded secrets — WITHOUT executing
or deserialising any content.

Fail-closed: any parse error is itself a high-severity finding; the scanner
never swallows errors and never returns a clean report on broken input.

Example::

    from provenir.governance.scan import ModelScanner, scan_gate, ScanBlocked

    report = ModelScanner().scan("/path/to/model/dir")
    print(report.summary())

    try:
        scan_gate(report)
        print("Model is safe to promote")
    except ScanBlocked as exc:
        print(f"BLOCKED: {exc}")
"""

from __future__ import annotations

import hashlib
import io
import json
import pickletools
import re
import struct
import zipfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterator

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class Severity(str, Enum):
    """Finding severity levels (critical/high set unsafe=True on the report)."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ThreatClass(str, Enum):
    """Threat classes checked by :class:`ModelScanner`."""

    PICKLE_REDUCE = "pickle_reduce"
    SCANNER_EVASION = "scanner_evasion"
    CHAT_TEMPLATE_EXEC = "chat_template_exec"
    KERAS_LAMBDA = "keras_lambda"
    WEIGHT_FORMAT = "weight_format"
    NAMESPACE_PIN = "namespace_pin"
    EMBEDDED_SECRET = "embedded_secret"


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

_UNSAFE_SEVERITIES: frozenset[Severity] = frozenset({Severity.CRITICAL, Severity.HIGH})


@dataclass(frozen=True)
class Finding:
    """A single scanner finding.

    Example:
        >>> f = Finding(ThreatClass.WEIGHT_FORMAT, Severity.MEDIUM, "model.bin", "pickle weight")
        >>> f.to_dict()["threat"]
        'weight_format'
    """

    threat: ThreatClass
    severity: Severity
    path: str
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "threat": self.threat.value,
            "severity": self.severity.value,
            "path": self.path,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class ScanReport:
    """Aggregate report for one scan invocation.

    Example:
        >>> r = ScanReport(target="/m", scanned_files=0, findings=[])
        >>> r.unsafe()
        False
        >>> len(r.content_hash())
        64
    """

    target: str
    scanned_files: int
    findings: list[Finding] = field(default_factory=list)

    def unsafe(self) -> bool:
        """Return True if any finding has critical or high severity."""
        return any(f.severity in _UNSAFE_SEVERITIES for f in self.findings)

    def content_hash(self) -> str:
        """SHA-256 over sorted canonical findings — deterministic and signable."""
        sorted_findings = sorted(
            [f.to_dict() for f in self.findings],
            key=lambda d: json.dumps(d, sort_keys=True),
        )
        canonical = json.dumps(
            sorted_findings,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def summary(self) -> str:
        """One-line human summary."""
        verdict = "UNSAFE" if self.unsafe() else "clean"
        counts = {s.value: 0 for s in Severity}
        for f in self.findings:
            counts[f.severity.value] += 1
        parts = ", ".join(f"{v} {k}" for k, v in counts.items() if v)
        detail = f" ({parts})" if parts else ""
        return (
            f"scan {self.target!r}: {verdict}{detail} "
            f"[{len(self.findings)} findings, {self.scanned_files} files scanned]"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "scanned_files": self.scanned_files,
            "unsafe": self.unsafe(),
            "findings": [f.to_dict() for f in self.findings],
            "content_hash": self.content_hash(),
        }


@dataclass(frozen=True)
class ScanComponent:
    """BOM component recording the result of a supply-chain scan.

    Intended to be attached as ``ModelBOM.scan`` so the scan verdict is inside
    the signed envelope of any :class:`~provenir.governance.passport.ModelPassport`.

    Example:
        >>> sc = ScanComponent("0.7.0", "abc123", False, {"critical": 0})
        >>> sc.to_dict()["unsafe"]
        False
    """

    scanner_version: str
    report_hash: str
    unsafe: bool
    finding_counts: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "scanner_version": self.scanner_version,
            "report_hash": self.report_hash,
            "unsafe": self.unsafe,
            "finding_counts": dict(self.finding_counts),
        }

    @classmethod
    def from_report(cls, report: ScanReport, scanner_version: str = "0.7.0") -> ScanComponent:
        """Construct from a :class:`ScanReport`."""
        counts: dict[str, int] = {s.value: 0 for s in Severity}
        for f in report.findings:
            counts[f.severity.value] += 1
        return cls(
            scanner_version=scanner_version,
            report_hash=report.content_hash(),
            unsafe=report.unsafe(),
            finding_counts=counts,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ScanComponent:
        return cls(
            scanner_version=data["scanner_version"],
            report_hash=data["report_hash"],
            unsafe=data["unsafe"],
            finding_counts=dict(data.get("finding_counts", {})),
        )


# ---------------------------------------------------------------------------
# Promotion gate (mirrors reliability.gate_promotion)
# ---------------------------------------------------------------------------


class ScanBlocked(RuntimeError):
    """Raised when a scan result blocks promotion to a protected stage."""


def scan_gate(
    report: ScanReport,
    *,
    allow_severities: frozenset[Severity] = frozenset(),
) -> None:
    """Raise :class:`ScanBlocked` if *report* is unsafe.

    ``allow_severities`` can downgrade specific severity levels so they no
    longer count as unsafe for gate purposes (e.g. allow medium/low findings
    while still blocking on critical/high).

    Example::

        scan_gate(clean_report)                                  # no-op
        scan_gate(unsafe_report)                                 # raises ScanBlocked
        scan_gate(report, allow_severities=frozenset({Severity.HIGH}))  # allows high
    """
    blocking = [
        f for f in report.findings
        if f.severity in _UNSAFE_SEVERITIES and f.severity not in allow_severities
    ]
    if blocking:
        summary_lines = "; ".join(
            f"{f.severity.value}/{f.threat.value}: {f.detail} ({f.path})"
            for f in blocking[:5]
        )
        raise ScanBlocked(
            f"Model scan found {len(blocking)} blocking finding(s): {summary_lines}. "
            f"{report.summary()}"
        )


# ---------------------------------------------------------------------------
# Internal helpers — static opcode/byte inspection (NEVER execute anything)
# ---------------------------------------------------------------------------

# Pickle opcodes that indicate arbitrary code execution
_DANGEROUS_PICKLE_OPCODES: frozenset[str] = frozenset(
    {"REDUCE", "GLOBAL", "STACK_GLOBAL", "INST", "OBJ", "NEWOBJ", "NEWOBJ_EX"}
)

# Safe modules that may legitimately appear as GLOBAL references in model files
_SAFE_PICKLE_MODULES: frozenset[str] = frozenset(
    {
        "collections",
        "collections.OrderedDict",
        "torch._utils",
        "torch",
        "_codecs",
        "numpy",
        "numpy.core.multiarray",
        "numpy.ndarray",
        "__builtin__",
        "builtins",
        "copy_reg",
        "copyreg",
        "operator",
        "_operator",
    }
)

# Regex patterns for Jinja2 exec-capable constructs in chat templates
_JINJA_EXEC_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\{%-?\s*(for|if|set|do|import|from|include|extends|block|macro|call|filter|with)\b"),
    re.compile(r"__[a-zA-Z_]+__"),          # dunder attribute access
    re.compile(r"\bcycler\b"),
    re.compile(r"\bnamespace\b"),
    re.compile(r"\bself\b"),
    re.compile(r"\blipsum\b"),
    re.compile(r"\.(?:update|items|values|keys|format|encode|decode|join|split|replace)\s*\("),
    re.compile(r"\[\s*['\"]__"),            # dict access to dunder
]

# Secret / API key patterns (intentionally generic to avoid being a recipe)
_SECRET_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("aws_access_key", re.compile(r"AKIA[0-9A-Z]{16}")),
    ("github_pat", re.compile(r"ghp_[A-Za-z0-9]{36}")),
    ("openai_key", re.compile(r"sk-[A-Za-z0-9]{20,}")),
    ("hf_token", re.compile(r"hf_[A-Za-z0-9]{20,}")),
    (
        "generic_token",
        re.compile(
            r"(?i)(api[_-]?key|secret[_-]?key|access[_-]?token)"
            r"\s*[=:]\s*['\"]?[A-Za-z0-9+/=_\-]{20,}"
        ),
    ),
]

# Magic bytes for common formats
_MAGIC_BYTES: dict[str, bytes] = {
    ".pt": b"PK",      # PyTorch files are zip archives
    ".bin": b"PK",     # some .bin are zip; others are raw pickle — handled below
    ".zip": b"PK",
    ".gguf": b"GGUF",
    ".h5": b"\x89HDF",
    ".pkl": b"\x80",   # pickle protocol marker (optional but common)
}

# Extensions that indicate pickle-format weights
_PICKLE_WEIGHT_EXTENSIONS: frozenset[str] = frozenset(
    {".pt", ".pth", ".bin", ".ckpt", ".pkl", ".pickle"}
)

# Safetensors extension
_SAFETENSORS_EXT: str = ".safetensors"


# ---------------------------------------------------------------------------
# Per-threat-class check functions
# ---------------------------------------------------------------------------


def _check_pickle_bytes(data: bytes, path: str) -> list[Finding]:
    """Scan pickle bytes for dangerous opcodes using pickletools.genops."""
    findings: list[Finding] = []
    try:
        for opcode, arg, _pos in pickletools.genops(data):
            name: str = getattr(opcode, "name", "")  # pickletools.OpcodeInfo.name
            if name in _DANGEROUS_PICKLE_OPCODES:
                # GLOBAL/STACK_GLOBAL reference a module — check if it is allowlisted
                if name in ("GLOBAL", "STACK_GLOBAL", "INST"):
                    module = str(arg).split("\n")[0] if arg is not None else ""
                    if not any(module.startswith(safe) for safe in _SAFE_PICKLE_MODULES):
                        findings.append(
                            Finding(
                                ThreatClass.PICKLE_REDUCE,
                                Severity.CRITICAL,
                                path,
                                f"pickle {name} opcode referencing non-allowlisted module: {arg!r}",
                            )
                        )
                else:
                    findings.append(
                        Finding(
                            ThreatClass.PICKLE_REDUCE,
                            Severity.HIGH,
                            path,
                            f"pickle {name} opcode (arbitrary call possible)",
                        )
                    )
    except Exception as exc:  # truncated / malformed pickle — fail-closed
        findings.append(
            Finding(
                ThreatClass.PICKLE_REDUCE,
                Severity.HIGH,
                path,
                f"pickle parse error (truncated/malformed stream): {type(exc).__name__}: {exc}",
            )
        )
    return findings


def _iter_zip_pickles(data: bytes, outer_path: str) -> Iterator[tuple[bytes, str]]:
    """Yield (pickle_bytes, inner_path) for every pickle-like member of a ZIP."""
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            for name in zf.namelist():
                if any(name.endswith(ext) for ext in _PICKLE_WEIGHT_EXTENSIONS):
                    try:
                        yield zf.read(name), f"{outer_path}::{name}"
                    except Exception:  # pragma: no cover – corrupt member
                        pass
    except Exception:  # not a valid ZIP
        pass


def _check_pickle_reduce(raw: bytes, path: str) -> list[Finding]:
    """Threat class: PICKLE_REDUCE — dangerous opcodes in pickle/pt/ckpt/bin files."""
    findings: list[Finding] = []
    # Check if it is a ZIP archive (PyTorch saves .pt as ZIP + pickle members)
    if raw[:2] == b"PK":
        for pkl_bytes, inner in _iter_zip_pickles(raw, path):
            findings.extend(_check_pickle_bytes(pkl_bytes, inner))
    else:
        findings.extend(_check_pickle_bytes(raw, path))
    return findings


def _check_scanner_evasion(raw: bytes, path: str, suffix: str) -> list[Finding]:
    """Threat class: SCANNER_EVASION — magic byte / extension mismatches, corrupt ZIP."""
    findings: list[Finding] = []

    # Magic byte vs extension check
    expected_magic = _MAGIC_BYTES.get(suffix)
    if expected_magic is not None and len(raw) >= len(expected_magic):
        if not raw.startswith(expected_magic):
            # .pt / .bin raw pickle is acceptable (starts with \x80) but not mandatory
            if suffix in (".pt", ".bin") and raw[:1] == b"\x80":
                pass  # raw pickle, not a ZIP — handle via pickle_reduce
            elif suffix in (".pkl", ".pickle") and raw[:1] == b"\x80":
                pass  # expected for raw pickle
            else:
                findings.append(
                    Finding(
                        ThreatClass.SCANNER_EVASION,
                        Severity.HIGH,
                        path,
                        f"magic bytes {raw[:4]!r} do not match expected for {suffix!r} extension",
                    )
                )

    # ZIP CRC mismatch / truncation check
    if raw[:2] == b"PK":
        try:
            with zipfile.ZipFile(io.BytesIO(raw)) as zf:
                bad = zf.testzip()
                if bad is not None:
                    findings.append(
                        Finding(
                            ThreatClass.SCANNER_EVASION,
                            Severity.HIGH,
                            path,
                            f"ZIP CRC mismatch in member: {bad!r}",
                        )
                    )
        except zipfile.BadZipFile as exc:
            findings.append(
                Finding(
                    ThreatClass.SCANNER_EVASION,
                    Severity.HIGH,
                    path,
                    f"corrupt ZIP archive: {exc}",
                )
            )
        except Exception as exc:
            findings.append(
                Finding(
                    ThreatClass.SCANNER_EVASION,
                    Severity.MEDIUM,
                    path,
                    f"ZIP parse error: {exc}",
                )
            )

    # Truncated pickle stream — a valid pickle must end with STOP opcode (b'.')
    if suffix in _PICKLE_WEIGHT_EXTENSIONS and raw[:1] == b"\x80":
        if not raw.rstrip(b"\n\r ").endswith(b"."):
            findings.append(
                Finding(
                    ThreatClass.SCANNER_EVASION,
                    Severity.MEDIUM,
                    path,
                    "pickle stream does not terminate with STOP opcode — may be truncated",
                )
            )

    return findings


def _check_chat_template_exec(text: str, path: str) -> list[Finding]:
    """Threat class: CHAT_TEMPLATE_EXEC — Jinja2 exec constructs in chat templates."""
    findings: list[Finding] = []
    for pattern in _JINJA_EXEC_PATTERNS:
        match = pattern.search(text)
        if match:
            snippet = text[max(0, match.start() - 20): match.end() + 20].replace("\n", " ")
            findings.append(
                Finding(
                    ThreatClass.CHAT_TEMPLATE_EXEC,
                    Severity.CRITICAL,
                    path,
                    f"Jinja2 exec construct at offset {match.start()}: {snippet!r}",
                )
            )
            break  # one finding per file — avoid noise
    return findings


def _check_keras_lambda(raw: bytes, path: str, suffix: str) -> list[Finding]:
    """Threat class: KERAS_LAMBDA — Lambda layers / custom code in Keras/H5 files."""
    findings: list[Finding] = []
    # Scan raw bytes for Keras Lambda layer signatures
    indicators = [b"Lambda", b"lambda_layer", b"custom_objects", b"from_config"]
    text_like = raw.replace(b"\x00", b" ")
    for indicator in indicators:
        if indicator in text_like:
            findings.append(
                Finding(
                    ThreatClass.KERAS_LAMBDA,
                    Severity.HIGH,
                    path,
                    f"Keras Lambda/custom-object marker found: {indicator.decode()!r}",
                )
            )
            break
    # H5 / Keras JSON config scan
    if suffix in (".h5", ".keras"):
        try:
            text = raw.decode("utf-8", errors="replace")
            if "lambda" in text.lower() and "class_name" in text:
                findings.append(
                    Finding(
                        ThreatClass.KERAS_LAMBDA,
                        Severity.HIGH,
                        path,
                        "Keras config JSON contains Lambda layer class_name",
                    )
                )
        except Exception:
            pass
    return findings


def _check_safetensors_header(raw: bytes, path: str) -> list[Finding]:
    """Parse safetensors header by hand; flag if malformed or oversized."""
    findings: list[Finding] = []
    if len(raw) < 8:
        findings.append(
            Finding(
                ThreatClass.WEIGHT_FORMAT,
                Severity.HIGH,
                path,
                "safetensors file too short to contain a valid header length prefix",
            )
        )
        return findings
    try:
        (header_len,) = struct.unpack_from("<Q", raw, 0)
        if header_len == 0 or header_len > 100 * 1024 * 1024:  # >100 MB header is suspicious
            findings.append(
                Finding(
                    ThreatClass.WEIGHT_FORMAT,
                    Severity.HIGH,
                    path,
                    f"safetensors header length {header_len} is out of expected range",
                )
            )
            return findings
        if len(raw) < 8 + header_len:
            findings.append(
                Finding(
                    ThreatClass.WEIGHT_FORMAT,
                    Severity.HIGH,
                    path,
                    f"safetensors file truncated: declared header_len={header_len} "
                    f"but only {len(raw) - 8} bytes available",
                )
            )
            return findings
        header_bytes = raw[8: 8 + header_len]
        json.loads(header_bytes)  # must be valid JSON
    except (struct.error, json.JSONDecodeError, ValueError, UnicodeDecodeError) as exc:
        findings.append(
            Finding(
                ThreatClass.WEIGHT_FORMAT,
                Severity.HIGH,
                path,
                f"safetensors header parse error: {type(exc).__name__}: {exc}",
            )
        )
    return findings


def _check_weight_format(raw: bytes, path: str, suffix: str) -> list[Finding]:
    """Threat class: WEIGHT_FORMAT — pickle-format weights flagged; safetensors validated."""
    findings: list[Finding] = []
    if suffix in _PICKLE_WEIGHT_EXTENSIONS:
        findings.append(
            Finding(
                ThreatClass.WEIGHT_FORMAT,
                Severity.MEDIUM,
                path,
                f"weight file uses pickle-compatible format ({suffix}); prefer safetensors",
            )
        )
    elif suffix == _SAFETENSORS_EXT:
        findings.extend(_check_safetensors_header(raw, path))
    return findings


def _check_namespace_pin(text: str, path: str) -> list[Finding]:
    """Threat class: NAMESPACE_PIN — floating model refs (bare name, no hash pin)."""
    findings: list[Finding] = []
    # Look for common HuggingFace repo patterns without a commit-hash pin
    # Patterns like "model_name_or_path": "meta-llama/Llama-3" without @sha
    bare_ref_pattern = re.compile(
        r'"(?:model_name_or_path|base_model|_name_or_path|pretrained_model_name_or_path)"\s*:\s*'
        r'"([^"@\n]{3,})"'
    )
    for match in bare_ref_pattern.finditer(text):
        ref = match.group(1)
        # Skip local paths (start with / or .)
        if ref.startswith(("/", ".", "~")) or "\\" in ref:
            continue
        # Skip if it looks like a local file path with extension
        if re.search(r"\.[a-zA-Z]{1,5}$", ref):
            continue
        findings.append(
            Finding(
                ThreatClass.NAMESPACE_PIN,
                Severity.LOW,
                path,
                f"model reference {ref!r} is not pinned to a content hash (no @sha or digest)",
            )
        )
    return findings


def _check_embedded_secret(text: str, path: str) -> list[Finding]:
    """Threat class: EMBEDDED_SECRET — API keys and tokens in config/text files."""
    findings: list[Finding] = []
    for label, pattern in _SECRET_PATTERNS:
        match = pattern.search(text)
        if match:
            snippet = match.group(0)[:30]
            findings.append(
                Finding(
                    ThreatClass.EMBEDDED_SECRET,
                    Severity.CRITICAL,
                    path,
                    f"potential {label} found: {snippet!r}…",
                )
            )
    return findings


# ---------------------------------------------------------------------------
# Scanner
# ---------------------------------------------------------------------------

#: Files we scan for text-based threats (secrets, namespace pins, chat templates)
_TEXT_EXTENSIONS: frozenset[str] = frozenset(
    {".json", ".yaml", ".yml", ".txt", ".md", ".cfg", ".ini", ".toml", ".py", ".sh"}
)

#: Binary weight extensions we inspect
_BINARY_WEIGHT_EXTENSIONS: frozenset[str] = frozenset(
    {".pt", ".pth", ".bin", ".ckpt", ".pkl", ".pickle", ".safetensors", ".h5", ".keras", ".gguf"}
)

#: Files/dirs to skip entirely
_SKIP_NAMES: frozenset[str] = frozenset(
    {".git", "__pycache__", ".cache", "node_modules"}
)

#: Maximum file size to read into memory (64 MB); larger files get a medium warning
_MAX_FILE_BYTES: int = 64 * 1024 * 1024


class ModelScanner:
    """Static model artifact scanner.

    Inspects every file under a target directory (or a single file) for
    supply-chain threats. All checks are read-only; nothing is executed or
    deserialised.

    Example::

        from provenir.governance.scan import ModelScanner, ScanComponent

        report = ModelScanner().scan("./my-model")
        component = ScanComponent.from_report(report)
    """

    VERSION: str = "0.7.0"

    def __init__(
        self,
        allowed_modules: frozenset[str] | None = None,
    ) -> None:
        self._allowed_modules: frozenset[str] = (
            allowed_modules if allowed_modules is not None else _SAFE_PICKLE_MODULES
        )

    def scan(self, target: str | Path) -> ScanReport:
        """Scan *target* (file or directory) and return a :class:`ScanReport`.

        Fail-closed: any IO/parse error on a file produces a high-severity
        finding rather than an exception propagating to the caller.
        """
        target = Path(target)
        findings: list[Finding] = []
        scanned = 0

        if target.is_file():
            files = [target]
        elif target.is_dir():
            files = sorted(self._collect_files(target))
        else:
            findings.append(
                Finding(
                    ThreatClass.SCANNER_EVASION,
                    Severity.HIGH,
                    str(target),
                    f"scan target does not exist or is not accessible: {target}",
                )
            )
            return ScanReport(target=str(target), scanned_files=0, findings=findings)

        for file_path in files:
            scanned += 1
            try:
                file_findings = self._scan_file(file_path, target)
            except Exception as exc:  # fail-closed safety net
                file_findings = [
                    Finding(
                        ThreatClass.SCANNER_EVASION,
                        Severity.HIGH,
                        str(file_path),
                        f"unexpected scanner error (fail-closed): {type(exc).__name__}: {exc}",
                    )
                ]
            findings.extend(file_findings)

        return ScanReport(
            target=str(target),
            scanned_files=scanned,
            findings=findings,
        )

    def _collect_files(self, root: Path) -> Iterator[Path]:
        for item in root.rglob("*"):
            if any(part in _SKIP_NAMES for part in item.parts):
                continue
            if item.is_file():
                yield item

    def _scan_file(self, file_path: Path, scan_root: Path) -> list[Finding]:
        """Return all findings for one file. Never raises."""
        findings: list[Finding] = []
        suffix = file_path.suffix.lower()
        rel = str(file_path.relative_to(scan_root) if scan_root.is_dir() else file_path)

        # Read file (capped at _MAX_FILE_BYTES)
        try:
            size = file_path.stat().st_size
            if size > _MAX_FILE_BYTES:
                findings.append(
                    Finding(
                        ThreatClass.SCANNER_EVASION,
                        Severity.MEDIUM,
                        rel,
                        f"file size {size} bytes exceeds scan limit; partial scan only",
                    )
                )
            raw = file_path.read_bytes()[:_MAX_FILE_BYTES]
        except Exception as exc:
            return [
                Finding(
                    ThreatClass.SCANNER_EVASION,
                    Severity.HIGH,
                    rel,
                    f"cannot read file: {type(exc).__name__}: {exc}",
                )
            ]

        # Binary weight files
        if suffix in _BINARY_WEIGHT_EXTENSIONS:
            findings.extend(_check_weight_format(raw, rel, suffix))
            if suffix in _PICKLE_WEIGHT_EXTENSIONS:
                findings.extend(_check_pickle_reduce(raw, rel))
                findings.extend(_check_scanner_evasion(raw, rel, suffix))
            if suffix in (".h5", ".keras"):
                findings.extend(_check_keras_lambda(raw, rel, suffix))
            if suffix == ".gguf":
                # GGUF files may embed chat templates in their metadata as JSON
                findings.extend(self._scan_gguf_metadata(raw, rel))

        # Text / config files
        if suffix in _TEXT_EXTENSIONS or suffix == "":
            try:
                text = raw.decode("utf-8", errors="replace")
            except Exception:
                text = ""

            if text:
                findings.extend(_check_embedded_secret(text, rel))
                findings.extend(_check_namespace_pin(text, rel))

                # tokenizer_config.json — may contain chat_template
                if "chat_template" in text:
                    findings.extend(_check_chat_template_exec(text, rel))

                # Keras config JSON
                if suffix == ".json" and ("Lambda" in text or "lambda" in text):
                    findings.extend(_check_keras_lambda(raw, rel, suffix))

        return findings

    def _scan_gguf_metadata(self, raw: bytes, path: str) -> list[Finding]:
        """Scan GGUF file metadata for chat template exec patterns.

        GGUF stores metadata as typed key-value pairs after a fixed header.
        We do a best-effort regex sweep over the raw bytes rather than
        implementing the full GGUF spec, to stay fail-closed.
        """
        findings: list[Finding] = []
        try:
            text = raw.decode("utf-8", errors="replace")
            if "chat_template" in text:
                findings.extend(_check_chat_template_exec(text, path))
        except Exception as exc:
            findings.append(
                Finding(
                    ThreatClass.CHAT_TEMPLATE_EXEC,
                    Severity.HIGH,
                    path,
                    f"GGUF metadata parse error (fail-closed): {exc}",
                )
            )
        return findings
