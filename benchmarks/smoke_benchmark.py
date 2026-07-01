from __future__ import annotations

import json

from provenir.core.config import RunConfig
from provenir.data.dataset import JsonlDataset
from provenir.train.backends.stub import StubBackend
from provenir.train.trainer import Trainer


def run() -> dict[str, object]:
    dataset = JsonlDataset.from_path("tests/fixtures/sample.jsonl")
    config = RunConfig(
        name="benchmark-run", backend="stub", seed=1,
        deterministic=True, output_dir="artifacts/bench",
    )
    trainer = Trainer(backend=StubBackend(), config=config)
    manifest = trainer.run(dataset=dataset)
    return {
        "run_id": manifest.run_id,
        "dataset_hash": manifest.dataset_hash,
        "config_hash": manifest.config_hash,
    }


if __name__ == "__main__":
    output = run()
    print(json.dumps(output, indent=2))
