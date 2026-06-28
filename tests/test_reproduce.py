from pathlib import Path

from provenir.core.config import RunConfig
from provenir.core.manifest import RunManifestStore
from provenir.data.dataset import JsonlDataset
from provenir.train.backends.stub import StubBackend
from provenir.train.trainer import Trainer


def test_reproduce_uses_manifest_and_replays_run(tmp_path: Path) -> None:
    dataset = JsonlDataset.from_path("tests/fixtures/sample.jsonl")
    config = RunConfig(
        name="repro-run", backend="stub", seed=5, deterministic=True,
        output_dir=str(tmp_path / "artifacts"),
    )
    trainer = Trainer(backend=StubBackend(), config=config)
    manifest = trainer.run(dataset=dataset)

    store = RunManifestStore(root_dir=tmp_path)
    store.save(manifest)

    replayed = store.load(manifest.run_id)
    assert replayed.run_id == manifest.run_id
    assert replayed.config_hash == manifest.config_hash
