from pathlib import Path

from provenir.core.config import RunConfig
from provenir.data.dataset import JsonlDataset
from provenir.train.backends.stub import StubBackend
from provenir.train.trainer import Trainer


def test_training_pipeline_emits_manifest_and_adapter(tmp_path: Path) -> None:
    dataset_path = Path("tests/fixtures/sample.jsonl")
    dataset = JsonlDataset.from_path(dataset_path)

    config = RunConfig(
        name="stub-sft",
        backend="stub",
        seed=7,
        deterministic=True,
        output_dir=str(tmp_path / "artifacts"),
    )

    trainer = Trainer(backend=StubBackend(), config=config)
    manifest = trainer.run(dataset=dataset)

    assert manifest.run_id
    assert manifest.dataset_hash
    assert manifest.config_hash
    assert manifest.provenance["backend"] == "stub"
    assert (tmp_path / "artifacts" / "adapter.bin").exists()
