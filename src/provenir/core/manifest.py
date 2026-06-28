from __future__ import annotations

import json
from pathlib import Path

from provenir.core.abstractions import RunManifest


class RunManifestStore:
    """Persist and retrieve run manifests as JSON files."""

    def __init__(self, root_dir: str | Path) -> None:
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def save(self, manifest: RunManifest) -> Path:
        path = self.root_dir / f"{manifest.run_id}.json"
        path.write_text(json.dumps(manifest.__dict__, indent=2), encoding="utf-8")
        return path

    def load(self, run_id: str) -> RunManifest:
        path = self.root_dir / f"{run_id}.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        return RunManifest(**data)
