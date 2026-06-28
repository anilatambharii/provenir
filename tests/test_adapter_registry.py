from pathlib import Path

from provenir.adapters.registry import AdapterRegistry


def test_adapter_registry_persists_lineage(tmp_path: Path) -> None:
    registry = AdapterRegistry(root_dir=tmp_path)
    record = registry.register(
        adapter_id="adapter-001",
        base_model="gpt-4o-mini",
        parent_run="run-1",
        dataset_hash="abc123",
        reward_config="exact-match",
    )

    assert record.adapter_id == "adapter-001"
    assert record.lineage["parent_run"] == "run-1"
    assert (tmp_path / "adapter-001.json").exists()
