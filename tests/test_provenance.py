from __future__ import annotations

from pathlib import Path

import pytest

from provenir.core.abstractions import RunManifest
from provenir.core.manifest import RunManifestStore
from provenir.provenance import (
    EnvironmentFingerprint,
    LineageEdge,
    LineageGraph,
    LineageNode,
    LineageStore,
    ReplayEngine,
    ReplayVerification,
    capture_fingerprint,
    kernel_determinism_flags,
)

# ---------------------------------------------------------------------------
# fingerprint
# ---------------------------------------------------------------------------


def test_capture_fingerprint_is_deterministic_with_injected_packages() -> None:
    packages = {"numpy": "1.26.4", "torch": "2.3.0"}
    a = capture_fingerprint(packages)
    b = capture_fingerprint(packages)
    assert a == b
    assert a.packages_hash == b.packages_hash


def test_capture_fingerprint_package_order_invariant() -> None:
    a = capture_fingerprint({"numpy": "1.26.4", "torch": "2.3.0"})
    b = capture_fingerprint({"torch": "2.3.0", "numpy": "1.26.4"})
    assert a.packages_hash == b.packages_hash


def test_capture_fingerprint_different_packages_differ() -> None:
    a = capture_fingerprint({"numpy": "1.26.4"})
    b = capture_fingerprint({"numpy": "1.26.5"})
    assert a.packages_hash != b.packages_hash


def test_capture_fingerprint_empty_packages() -> None:
    fp = capture_fingerprint({})
    assert isinstance(fp.packages_hash, str)
    assert len(fp.packages_hash) == 64


def test_capture_fingerprint_fields_present() -> None:
    fp = capture_fingerprint({"numpy": "1.26.4"})
    assert fp.python_version.count(".") == 2
    assert fp.platform
    assert fp.cuda_version  # "unknown" or a version
    assert fp.hardware


def test_fingerprint_to_dict_round_trippable() -> None:
    fp = capture_fingerprint({"numpy": "1.26.4"})
    data = fp.to_dict()
    assert set(data) == {
        "python_version",
        "platform",
        "packages_hash",
        "cuda_version",
        "hardware",
    }
    assert EnvironmentFingerprint(**data) == fp


def test_capture_fingerprint_none_discovers_best_effort() -> None:
    fp = capture_fingerprint(None)
    assert isinstance(fp, EnvironmentFingerprint)
    assert len(fp.packages_hash) == 64


def test_kernel_determinism_flags_contents() -> None:
    flags = kernel_determinism_flags()
    assert flags["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert flags["PYTHONHASHSEED"] == "0"
    assert "CUDA_LAUNCH_BLOCKING" in flags
    assert all(isinstance(v, str) for v in flags.values())


# ---------------------------------------------------------------------------
# lineage nodes / edges validation
# ---------------------------------------------------------------------------


def test_lineage_node_rejects_bad_type() -> None:
    with pytest.raises(ValueError):
        LineageNode("n", "not-a-type", "h", {})


def test_lineage_node_accepts_all_valid_types() -> None:
    for node_type in ("dataset", "run", "adapter", "eval", "merge", "model"):
        node = LineageNode(f"n-{node_type}", node_type, "h", {})
        assert node.node_type == node_type


def test_lineage_edge_rejects_bad_relation() -> None:
    with pytest.raises(ValueError):
        LineageEdge("a", "b", "not-a-relation")


def test_lineage_edge_accepts_valid_relations() -> None:
    for relation in (
        "produced",
        "derived_from",
        "evaluated_by",
        "merged_into",
        "trained_on",
    ):
        edge = LineageEdge("a", "b", relation)
        assert edge.relation == relation


# ---------------------------------------------------------------------------
# graph structure
# ---------------------------------------------------------------------------


def _diamond() -> LineageGraph:
    g = LineageGraph()
    g.add_node(LineageNode("ds", "dataset", "h0", {}))
    g.add_node(LineageNode("run", "run", "h1", {}))
    g.add_node(LineageNode("adapter", "adapter", "h2", {}))
    g.add_node(LineageNode("eval", "eval", "h3", {}))
    g.add_edge(LineageEdge("ds", "run", "trained_on"))
    g.add_edge(LineageEdge("run", "adapter", "produced"))
    g.add_edge(LineageEdge("adapter", "eval", "evaluated_by"))
    return g


def test_add_edge_unknown_node_raises() -> None:
    g = LineageGraph()
    g.add_node(LineageNode("a", "dataset", "h", {}))
    with pytest.raises(ValueError):
        g.add_edge(LineageEdge("a", "missing", "produced"))
    with pytest.raises(ValueError):
        g.add_edge(LineageEdge("missing", "a", "produced"))


def test_add_edge_self_loop_raises() -> None:
    g = LineageGraph()
    g.add_node(LineageNode("a", "dataset", "h", {}))
    with pytest.raises(ValueError):
        g.add_edge(LineageEdge("a", "a", "produced"))


def test_add_edge_rejects_direct_cycle() -> None:
    g = LineageGraph()
    g.add_node(LineageNode("a", "dataset", "h", {}))
    g.add_node(LineageNode("b", "run", "h", {}))
    g.add_edge(LineageEdge("a", "b", "trained_on"))
    with pytest.raises(ValueError):
        g.add_edge(LineageEdge("b", "a", "produced"))


def test_add_edge_rejects_transitive_cycle() -> None:
    g = _diamond()
    with pytest.raises(ValueError):
        g.add_edge(LineageEdge("eval", "ds", "produced"))


def test_ancestors() -> None:
    g = _diamond()
    assert g.ancestors("eval") == {"ds", "run", "adapter"}
    assert g.ancestors("ds") == set()


def test_descendants() -> None:
    g = _diamond()
    assert g.descendants("ds") == {"run", "adapter", "eval"}
    assert g.descendants("eval") == set()


def test_ancestors_unknown_node_raises() -> None:
    g = _diamond()
    with pytest.raises(ValueError):
        g.ancestors("nope")


def test_roots_and_leaves() -> None:
    g = _diamond()
    assert g.roots() == ["ds"]
    assert g.leaves() == ["eval"]


def test_roots_multiple_sorted() -> None:
    g = LineageGraph()
    g.add_node(LineageNode("z", "dataset", "h", {}))
    g.add_node(LineageNode("a", "dataset", "h", {}))
    g.add_node(LineageNode("m", "run", "h", {}))
    g.add_edge(LineageEdge("a", "m", "trained_on"))
    assert g.roots() == ["a", "z"]
    assert g.leaves() == ["m", "z"]


def test_provenance_of_is_topologically_ordered() -> None:
    g = _diamond()
    chain = g.provenance_of("eval")
    ids = [n.node_id for n in chain]
    assert ids == ["ds", "run", "adapter"]
    # parents precede children in the ordering
    assert ids.index("ds") < ids.index("run") < ids.index("adapter")


def test_provenance_of_root_is_empty() -> None:
    g = _diamond()
    assert g.provenance_of("ds") == []


# ---------------------------------------------------------------------------
# serialization
# ---------------------------------------------------------------------------


def test_to_dict_from_dict_round_trip() -> None:
    g = _diamond()
    restored = LineageGraph.from_dict(g.to_dict())
    assert restored.to_dict() == g.to_dict()
    assert restored.ancestors("eval") == g.ancestors("eval")


def test_node_from_dict_round_trip() -> None:
    node = LineageNode("n", "adapter", "h", {"k": "v"})
    assert LineageNode.from_dict(node.to_dict()) == node


def test_edge_from_dict_round_trip() -> None:
    edge = LineageEdge("a", "b", "produced")
    assert LineageEdge.from_dict(edge.to_dict()) == edge


def test_to_dot_contains_node_ids_and_relations() -> None:
    g = _diamond()
    dot = g.to_dot()
    assert dot.startswith("digraph lineage {")
    for node_id in ("ds", "run", "adapter", "eval"):
        assert node_id in dot
    assert "trained_on" in dot
    assert "->" in dot


def test_graph_save_and_load(tmp_path: Path) -> None:
    g = _diamond()
    path = g.save(tmp_path / "lineage.json")
    assert path.exists()
    loaded = LineageGraph.load(path)
    assert loaded.to_dict() == g.to_dict()


# ---------------------------------------------------------------------------
# LineageStore
# ---------------------------------------------------------------------------


def test_lineage_store_record_run(tmp_path: Path) -> None:
    store = LineageStore(tmp_path)
    manifest = RunManifest(run_id="run-1", config_hash="cfg", dataset_hash="dsh", seed=3)
    graph = store.record_run(manifest, dataset_hash="dshash", parent_ids=[])

    provenance_ids = [n.node_id for n in graph.provenance_of("run:run-1")]
    assert provenance_ids == ["dataset:dshash"]
    assert graph.ancestors("run:run-1") == {"dataset:dshash"}
    assert (tmp_path / "run-1.lineage.json").exists()


def test_lineage_store_record_run_with_parents(tmp_path: Path) -> None:
    store = LineageStore(tmp_path)
    manifest = RunManifest(run_id="run-2", config_hash="cfg", dataset_hash="dsh")
    graph = store.record_run(
        manifest, dataset_hash="dshash", parent_ids=["prior-adapter"]
    )
    assert graph.ancestors("run:run-2") == {"dataset:dshash", "prior-adapter"}


def test_lineage_store_reload(tmp_path: Path) -> None:
    store = LineageStore(tmp_path)
    manifest = RunManifest(run_id="run-3", config_hash="cfg", dataset_hash="dsh")
    store.record_run(manifest, dataset_hash="dshash", parent_ids=[])
    reloaded = store.load("run-3")
    assert reloaded.ancestors("run:run-3") == {"dataset:dshash"}


def test_lineage_store_run_node_attributes(tmp_path: Path) -> None:
    store = LineageStore(tmp_path)
    manifest = RunManifest(
        run_id="run-4", config_hash="cfg", dataset_hash="dsh", seed=9, git_sha="abcd"
    )
    graph = store.record_run(manifest, dataset_hash="dshash", parent_ids=[])
    run_node = graph.node("run:run-4")
    assert run_node.attributes["seed"] == 9
    assert run_node.attributes["git_sha"] == "abcd"


# ---------------------------------------------------------------------------
# ReplayEngine
# ---------------------------------------------------------------------------


def _store_with_manifest(tmp_path: Path, **kwargs: object) -> RunManifestStore:
    store = RunManifestStore(tmp_path)
    store.save(RunManifest(**kwargs))  # type: ignore[arg-type]
    return store


def test_replay_verify_full_match(tmp_path: Path) -> None:
    fp = capture_fingerprint({"numpy": "1.26.4"})
    store = _store_with_manifest(
        tmp_path,
        run_id="r1",
        config_hash="c",
        dataset_hash="d",
        hardware_fingerprint=fp.packages_hash,
    )
    engine = ReplayEngine(store)
    result = engine.verify("r1", "c", "d", fp)
    assert result.reproducible
    assert result.matches
    assert result.differences == []


def test_replay_verify_config_mismatch(tmp_path: Path) -> None:
    store = _store_with_manifest(tmp_path, run_id="r2", config_hash="c", dataset_hash="d")
    engine = ReplayEngine(store)
    fp = capture_fingerprint({"numpy": "1.26.4"})
    result = engine.verify("r2", "DIFFERENT", "d", fp)
    assert not result.config_hash_match
    assert not result.reproducible
    assert any("config_hash" in diff for diff in result.differences)


def test_replay_verify_dataset_mismatch(tmp_path: Path) -> None:
    store = _store_with_manifest(tmp_path, run_id="r3", config_hash="c", dataset_hash="d")
    engine = ReplayEngine(store)
    fp = capture_fingerprint({"numpy": "1.26.4"})
    result = engine.verify("r3", "c", "OTHER", fp)
    assert not result.dataset_hash_match
    assert any("dataset_hash" in diff for diff in result.differences)


def test_replay_verify_env_mismatch(tmp_path: Path) -> None:
    store = _store_with_manifest(
        tmp_path,
        run_id="r4",
        config_hash="c",
        dataset_hash="d",
        hardware_fingerprint="STORED_HASH",
    )
    engine = ReplayEngine(store)
    fp = capture_fingerprint({"numpy": "9.9.9"})
    result = engine.verify("r4", "c", "d", fp)
    assert not result.env_match
    assert not result.reproducible
    assert any("environment differs" in diff for diff in result.differences)


def test_replay_verify_no_fingerprint_marks_env_unverified(tmp_path: Path) -> None:
    store = _store_with_manifest(tmp_path, run_id="r5", config_hash="c", dataset_hash="d")
    engine = ReplayEngine(store)
    result = engine.verify("r5", "c", "d", None)
    assert not result.env_match
    assert not result.reproducible
    assert any("not verified" in diff for diff in result.differences)


def test_replay_verification_reproducible_property() -> None:
    good = ReplayVerification(True, True, True, True, [])
    assert good.reproducible
    bad = ReplayVerification(False, True, True, True, [])
    assert not bad.reproducible


def test_replay_command_contents(tmp_path: Path) -> None:
    store = _store_with_manifest(
        tmp_path,
        run_id="r6",
        config_hash="cfg",
        dataset_hash="dsh",
        seed=42,
        git_sha="deadbeef",
        hardware_fingerprint="fp",
    )
    engine = ReplayEngine(store)
    recipe = engine.replay_command("r6")
    assert recipe["run_id"] == "r6"
    assert recipe["seed"] == 42
    assert recipe["config_hash"] == "cfg"
    assert recipe["dataset_hash"] == "dsh"
    assert recipe["git_sha"] == "deadbeef"
    assert recipe["hardware_fingerprint"] == "fp"
    assert recipe["env_flags"]["PYTHONHASHSEED"] == "0"


def test_replay_command_includes_all_determinism_flags(tmp_path: Path) -> None:
    store = _store_with_manifest(tmp_path, run_id="r7", config_hash="c", dataset_hash="d")
    engine = ReplayEngine(store)
    recipe = engine.replay_command("r7")
    assert recipe["env_flags"] == kernel_determinism_flags()
