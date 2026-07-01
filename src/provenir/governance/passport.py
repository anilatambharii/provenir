from __future__ import annotations

import base64
import hashlib
import hmac
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from provenir.core.abstractions import RunManifest
from provenir.governance.audit import AuditLogger
from provenir.governance.bom import (
    CodeComponent,
    DataComponent,
    EvalComponent,
    ModelBOM,
)

_HMAC_SHA256 = "HMAC-SHA256"


@dataclass(frozen=True)
class Attestation:
    """A tamper-evident signature over a :class:`ModelBOM`.

    Structured so a real asymmetric signer could be swapped in for the default
    HMAC-SHA256 scheme without changing consumers.

    Example:
        >>> Attestation("HMAC-SHA256", "c2ln", "default", "2026-01-01T00:00:00Z").key_id
        'default'
    """

    algorithm: str
    signature: str
    key_id: str
    signed_at: str

    def __post_init__(self) -> None:
        if not self.algorithm:
            raise ValueError("algorithm must be non-empty")
        if not self.signature:
            raise ValueError("signature must be non-empty")
        if not self.key_id:
            raise ValueError("key_id must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "signature": self.signature,
            "key_id": self.key_id,
            "signed_at": self.signed_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Attestation:
        return cls(
            algorithm=data["algorithm"],
            signature=data["signature"],
            key_id=data["key_id"],
            signed_at=data.get("signed_at", ""),
        )


def _compute_signature(key: bytes, bom: ModelBOM) -> str:
    """Compute the base64 HMAC-SHA256 signature over ``bom.canonical_json``."""
    digest = hmac.new(key, bom.canonical_json().encode("utf-8"), hashlib.sha256).digest()
    return base64.b64encode(digest).decode("ascii")


@dataclass(frozen=True)
class ModelPassport:
    """A portable, optionally-signed record of a model's bill-of-materials.

    Example:
        >>> from provenir.governance.bom import CodeComponent, DataComponent, EvalComponent
        >>> bom = ModelBOM(
        ...     model_id="m1", base_model="base", run_id="r1",
        ...     data=[DataComponent(name="d", content_hash="h", num_records=1)],
        ...     code=CodeComponent(git_sha="s", dependencies_hash="dh", framework="trl"),
        ...     evals=[EvalComponent(benchmark="mmlu", score=0.5)],
        ...     hyperparameters={},
        ... )
        >>> passport = PassportSigner(b"test-key").sign(bom, signed_at="2026-01-01")
        >>> passport.verify(b"test-key")
        True
    """

    bom: ModelBOM
    attestation: Attestation | None

    def verify(self, key: bytes) -> bool:
        """Return True if the attestation is a valid signature over the BOM.

        Recomputes the HMAC over ``bom.canonical_json()`` and compares it to the
        stored signature in constant time. Returns False when there is no
        attestation.
        """
        if self.attestation is None:
            return False
        expected = _compute_signature(key, self.bom)
        return hmac.compare_digest(expected, self.attestation.signature)

    def to_dict(self) -> dict[str, Any]:
        return {
            "bom": self.bom.to_dict(),
            "attestation": self.attestation.to_dict() if self.attestation else None,
        }

    def to_json(self) -> str:
        """Return a sorted-keys JSON serialization of the passport."""
        return json.dumps(self.to_dict(), sort_keys=True, indent=2, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelPassport:
        bom_data = data["bom"]
        bom = ModelBOM(
            model_id=bom_data["model_id"],
            base_model=bom_data["base_model"],
            run_id=bom_data["run_id"],
            data=[
                DataComponent(
                    name=component["name"],
                    content_hash=component["content_hash"],
                    num_records=component["num_records"],
                    license=component.get("license", "unknown"),
                    pii_scanned=component.get("pii_scanned", False),
                    contamination_checked=component.get("contamination_checked", False),
                )
                for component in bom_data["data"]
            ],
            code=CodeComponent(
                git_sha=bom_data["code"]["git_sha"],
                dependencies_hash=bom_data["code"]["dependencies_hash"],
                framework=bom_data["code"]["framework"],
            ),
            evals=[
                EvalComponent(
                    benchmark=component["benchmark"],
                    score=component["score"],
                    contaminated=component.get("contaminated", False),
                )
                for component in bom_data["evals"]
            ],
            hyperparameters=dict(bom_data.get("hyperparameters", {})),
            created_at=bom_data.get("created_at", ""),
        )
        attestation_data = data.get("attestation")
        attestation = Attestation.from_dict(attestation_data) if attestation_data else None
        return cls(bom=bom, attestation=attestation)

    def to_markdown(self) -> str:
        """Return a human-readable passport card in markdown."""
        bom = self.bom
        lines: list[str] = []
        lines.append(f"# Model Passport: {bom.model_id}")
        lines.append("")
        lines.append("## Overview")
        lines.append(f"- Base model: {bom.base_model}")
        lines.append(f"- Run ID: {bom.run_id}")
        lines.append(f"- Created at: {bom.created_at or 'unknown'}")
        lines.append(f"- Content hash: `{bom.content_hash()}`")
        lines.append("")

        lines.append("## Data")
        for component in bom.data:
            lines.append(
                f"- **{component.name}** ({component.num_records} records, "
                f"license: {component.license}, hash: `{component.content_hash}`)"
            )
        if not bom.data:
            lines.append("- (none)")
        lines.append("")

        lines.append("## Code")
        lines.append(f"- Framework: {bom.code.framework}")
        lines.append(f"- Git SHA: `{bom.code.git_sha}`")
        lines.append(f"- Dependencies hash: `{bom.code.dependencies_hash}`")
        lines.append("")

        lines.append("## Evaluations")
        for evaluation in bom.evals:
            marker = " (contaminated)" if evaluation.contaminated else ""
            lines.append(f"- {evaluation.benchmark}: {evaluation.score}{marker}")
        if not bom.evals:
            lines.append("- (none)")
        lines.append("")

        lines.append("## Hyperparameters")
        for name in sorted(bom.hyperparameters):
            lines.append(f"- {name}: {bom.hyperparameters[name]}")
        if not bom.hyperparameters:
            lines.append("- (none)")
        lines.append("")

        lines.append("## Risk Flags")
        flags = bom.risk_flags()
        if flags:
            for flag in flags:
                lines.append(f"- {flag}")
        else:
            lines.append("- none")
        lines.append("")

        lines.append("## Attestation")
        if self.attestation is None:
            lines.append("- Status: UNSIGNED")
        else:
            lines.append("- Status: SIGNED")
            lines.append(f"- Algorithm: {self.attestation.algorithm}")
            lines.append(f"- Key ID: {self.attestation.key_id}")
            lines.append(f"- Signed at: {self.attestation.signed_at or 'unknown'}")
            lines.append(f"- Signature: `{self.attestation.signature}`")
        lines.append("")

        return "\n".join(lines)


class PassportSigner:
    """Sign and verify model passports with HMAC-SHA256.

    The HMAC scheme is a stand-in for a real asymmetric signer; the
    :class:`Attestation` records the algorithm and key id so verification stays
    algorithm-aware.

    Example:
        >>> signer = PassportSigner(b"test-key", key_id="ci")
        >>> signer.key_id
        'ci'
    """

    def __init__(self, key: bytes, key_id: str = "default") -> None:
        if not key:
            raise ValueError("key must be non-empty")
        if not key_id:
            raise ValueError("key_id must be non-empty")
        self._key = key
        self.key_id = key_id

    def sign(self, bom: ModelBOM, signed_at: str = "") -> ModelPassport:
        """Return a signed :class:`ModelPassport` for ``bom``."""
        signature = _compute_signature(self._key, bom)
        attestation = Attestation(
            algorithm=_HMAC_SHA256,
            signature=signature,
            key_id=self.key_id,
            signed_at=signed_at,
        )
        return ModelPassport(bom=bom, attestation=attestation)

    def verify(self, passport: ModelPassport) -> bool:
        """Return True if ``passport`` verifies against this signer's key."""
        return passport.verify(self._key)


class PassportStore:
    """Directory-backed store for issued model passports.

    Every save appends an immutable ``passport_issued`` line to an optional
    :class:`AuditLogger`.

    Example:
        >>> import tempfile
        >>> store = PassportStore(tempfile.mkdtemp())
        >>> store.list_ids()
        []
    """

    def __init__(self, store_dir: str | Path, audit_logger: AuditLogger | None = None) -> None:
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.audit_logger = audit_logger

    def _path_for(self, model_id: str) -> Path:
        return self.store_dir / f"{model_id}.json"

    def save(self, passport: ModelPassport) -> Path:
        """Persist ``passport`` and record it in the audit log if configured.

        The filename is the BOM ``model_id`` when set, otherwise its content
        hash.
        """
        identifier = passport.bom.model_id or passport.bom.content_hash()
        path = self._path_for(identifier)
        path.write_text(passport.to_json(), encoding="utf-8")
        if self.audit_logger is not None:
            self.audit_logger.log(
                event="passport_issued",
                actor="passport_store",
                model_id=identifier,
                content_hash=passport.bom.content_hash(),
                signed=passport.attestation is not None,
            )
        return path

    def load(self, model_id: str) -> ModelPassport:
        """Load a previously saved passport by identifier."""
        path = self._path_for(model_id)
        data = json.loads(path.read_text(encoding="utf-8"))
        return ModelPassport.from_dict(data)

    def list_ids(self) -> list[str]:
        """Return the sorted identifiers of all stored passports."""
        return sorted(path.stem for path in self.store_dir.glob("*.json"))


def generate_passport_from_manifest(
    manifest: RunManifest,
    bom: ModelBOM,
    signer: PassportSigner,
    signed_at: str = "",
) -> ModelPassport:
    """Sign ``bom`` into a passport, tying it to a run ``manifest``.

    The manifest's ``run_id`` must match the BOM's ``run_id`` so the passport is
    provably anchored to a specific reproducible run.

    Example:
        >>> from provenir.core.abstractions import RunManifest
        >>> from provenir.governance.bom import CodeComponent, DataComponent, EvalComponent
        >>> bom = ModelBOM(
        ...     model_id="m1", base_model="base", run_id="r1",
        ...     data=[DataComponent(name="d", content_hash="h", num_records=1)],
        ...     code=CodeComponent(git_sha="s", dependencies_hash="dh", framework="trl"),
        ...     evals=[EvalComponent(benchmark="mmlu", score=0.5)],
        ...     hyperparameters={},
        ... )
        >>> manifest = RunManifest(run_id="r1")
        >>> passport = generate_passport_from_manifest(manifest, bom, PassportSigner(b"k"))
        >>> passport.attestation is not None
        True
    """
    if manifest.run_id != bom.run_id:
        raise ValueError(
            f"manifest run_id {manifest.run_id!r} does not match bom run_id {bom.run_id!r}"
        )
    return signer.sign(bom, signed_at=signed_at)
