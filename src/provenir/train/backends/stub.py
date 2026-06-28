from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from provenir.core.abstractions import RunManifest


class StubBackend:
    """CPU-only backend used for smoke tests and CI."""

    name = "stub"

    def prepare(self, config: Mapping[str, Any]) -> None:
        _ = config

    def fit(self, config: Mapping[str, Any], manifest: RunManifest) -> RunManifest:
        _ = (config, manifest)
        return manifest

    def save_adapter(self, output_dir: Path, config: Mapping[str, Any]) -> None:
        _ = (config,)
        adapter_path = output_dir / "adapter.bin"
        adapter_path.write_bytes(b"stub-adapter")

    def capabilities(self) -> Mapping[str, Any]:
        return {"supports_sft": True, "deterministic": True}
