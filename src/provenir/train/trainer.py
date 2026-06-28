from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping

from provenir.core.abstractions import Backend, RunManifest
from provenir.core.config import RunConfig
from provenir.core.manifest import RunManifestStore
from provenir.data.dataset import JsonlDataset


class Trainer:
    """Minimal training orchestrator for a reproducible SFT-style run."""

    def __init__(self, backend: Backend, config: RunConfig) -> None:
        self.backend = backend
        self.config = config

    def run(self, dataset: JsonlDataset) -> RunManifest:
        dataset.validate()
        self.backend.prepare(self._config_payload())
        manifest = RunManifest(
            config_hash=self._hash_config(),
            dataset_hash=dataset.hash(),
            seed=self.config.seed,
            git_sha=self._git_sha(),
            dependencies_lockfile=self._dependency_lock(),
            hardware_fingerprint="cpu-stub",
            provenance={
                "backend": self.backend.name,
                "algorithm": "sft",
                "dataset_size": len(dataset.records),
                "deterministic": self.config.deterministic,
            },
        )
        self.backend.fit(self._config_payload(), manifest)
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.backend.save_adapter(output_dir, self._config_payload())

        store = RunManifestStore(root_dir=output_dir / "manifests")
        store.save(manifest)
        return manifest

    def _config_payload(self) -> Mapping[str, Any]:
        return self.config.model_dump()

    def _hash_config(self) -> str:
        payload = json.dumps(self._config_payload(), sort_keys=True).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def _git_sha(self) -> str:
        try:
            return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return "unknown"

    def _dependency_lock(self) -> str:
        return "pydantic>=2.7.0\ntyping-extensions>=4.12.0"
