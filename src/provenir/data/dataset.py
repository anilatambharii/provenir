from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterator


class JsonlDataset:
    """Minimal JSONL dataset for CPU-friendly training smoke tests."""

    def __init__(self, records: list[dict[str, Any]]) -> None:
        self.records = records

    @classmethod
    def from_records(cls, records: list[dict[str, Any]]) -> "JsonlDataset":
        return cls(records)

    @classmethod
    def from_path(cls, path: str | Path) -> "JsonlDataset":
        dataset_path = Path(path)
        records: list[dict[str, Any]] = []
        with dataset_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return cls(records)

    def validate(self) -> None:
        if not self.records:
            raise ValueError("dataset is empty")

    def hash(self) -> str:
        payload = json.dumps(self.records, sort_keys=True).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def iter_records(self) -> Iterator[dict[str, Any]]:
        for record in self.records:
            yield record
