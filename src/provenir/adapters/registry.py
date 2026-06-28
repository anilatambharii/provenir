from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AdapterRecord:
    adapter_id: str
    base_model: str
    parent_run: str
    dataset_hash: str
    reward_config: str
    lineage: dict[str, Any]


class AdapterRegistry:
    """A JSON-backed adapter registry with lineage metadata."""

    def __init__(self, root_dir: str | Path) -> None:
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def register(
        self,
        adapter_id: str,
        base_model: str,
        parent_run: str,
        dataset_hash: str,
        reward_config: str,
    ) -> AdapterRecord:
        record = AdapterRecord(
            adapter_id=adapter_id,
            base_model=base_model,
            parent_run=parent_run,
            dataset_hash=dataset_hash,
            reward_config=reward_config,
            lineage={
                "parent_run": parent_run,
                "base_model": base_model,
                "dataset_hash": dataset_hash,
                "reward_config": reward_config,
            },
        )
        payload_path = self.root_dir / f"{adapter_id}.json"
        payload_path.write_text(json.dumps(record.__dict__, indent=2), encoding="utf-8")
        return record
