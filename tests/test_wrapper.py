from __future__ import annotations

import json
from pathlib import Path

import pytest

from provenir.core.abstractions import RunManifest
from provenir.data.dataset import JsonlDataset
from provenir.governance.bom import ModelBOM
from provenir.governance.passport import ModelPassport
from provenir.integrations import (
    ProvenirRun,
    TrackingConfig,
    provenance_tracked,
    track,
)
from provenir.integrations.wrapper import _hash_payload
from provenir.observability import HackingReport, RLStepMetrics

KEY = b"test-signing-key"
OTHER_KEY = b"other-key"
TS = "2026-01-01T00:00:00Z"


def _dataset() -> JsonlDataset:
    return JsonlDataset.from_records(
        [
            {"prompt": "1+1", "response": "2"},
            {"prompt": "2+2", "response": "4"},
        ]
    )


def _config(tmp_path: Path, **kwargs: object) -> TrackingConfig:
    kwargs.setdefault("output_dir", str(tmp_path / "artifacts"))
    return TrackingConfig(name="run-1", **kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# TrackingConfig validation
# ---------------------------------------------------------------------------


def test_config_defaults() -> None:
    cfg = TrackingConfig(name="run-1")
    assert cfg.base_model == "unknown"
    assert cfg.seed == 0
    assert cfg.sign_passport is False
    assert cfg.capture_env is True


def test_config_empty_name_raises() -> None:
    with pytest.raises(ValueError, match="name must be non-empty"):
        TrackingConfig(name="")


def test_config_sign_without_key_raises() -> None:
    with pytest.raises(ValueError, match="signing_key must be non-empty"):
        TrackingConfig(name="run-1", sign_passport=True)


def test_config_sign_with_key_ok() -> None:
    cfg = TrackingConfig(name="run-1", sign_passport=True, signing_key=KEY)
    assert cfg.signing_key == KEY


# ---------------------------------------------------------------------------
# Context manager round-trip + on-disk artifacts
# ---------------------------------------------------------------------------


def test_context_manager_writes_all_files(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    root = Path(cfg.output_dir)
    with ProvenirRun(cfg, config_payload={"lr": 0.001}, dataset=_dataset()) as run:
        run.log_step({"step": 0, "kl": 0.02, "entropy": 1.4})
    assert (root / "manifests" / "run-1.json").exists()
    assert (root / "lineage.json").exists()
    assert (root / "flight_recorder.json").exists()
    assert (root / "hacking_report.json").exists()
    assert (root / "bom.json").exists()


def test_manifest_round_trips_from_disk(tmp_path: Path) -> None:
    cfg = _config(tmp_path, seed=7)
    with ProvenirRun(cfg, config_payload={"lr": 0.001}, dataset=_dataset()) as run:
        run.log_step({"kl": 0.01})
    path = Path(cfg.output_dir) / "manifests" / "run-1.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    manifest = RunManifest(**data)
    assert manifest.run_id == "run-1"
    assert manifest.seed == 7
    assert manifest.dataset_hash == _dataset().hash()


def test_manifest_property_matches_disk(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    with ProvenirRun(cfg, config_payload={"lr": 0.001}) as run:
        run.log_step({"kl": 0.01})
    assert run.manifest is not None
    assert run.manifest.config_hash == _hash_payload({"lr": 0.001})


def test_config_hash_is_deterministic(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    with ProvenirRun(cfg, config_payload={"a": 1, "b": 2}) as run:
        pass
    with ProvenirRun(_config(tmp_path / "two"), config_payload={"b": 2, "a": 1}) as run2:
        pass
    assert run.manifest is not None and run2.manifest is not None
    assert run.manifest.config_hash == run2.manifest.config_hash


# ---------------------------------------------------------------------------
# log_step (dict and RLStepMetrics)
# ---------------------------------------------------------------------------


def test_log_step_with_dict(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    with ProvenirRun(cfg) as run:
        run.log_step({"step": 0, "kl": 0.02, "entropy": 1.4})
        run.log_step({"kl": 0.03})
    assert len(run.flight_recorder.history) == 2
    assert run.flight_recorder.history[0].kl == 0.02


def test_log_step_with_rlstepmetrics(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    with ProvenirRun(cfg) as run:
        run.log_step(RLStepMetrics(step=5, kl=0.1, entropy=2.0))
    assert run.flight_recorder.history[0].step == 5


def test_log_step_ignores_unknown_keys(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    with ProvenirRun(cfg) as run:
        run.log_step({"kl": 0.02, "nonsense": 999, "also_unknown": "x"})
    assert len(run.flight_recorder.history) == 1
    assert run.flight_recorder.history[0].kl == 0.02


def test_log_step_defaults_step_index(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    with ProvenirRun(cfg) as run:
        run.log_step({"kl": 0.01})
        run.log_step({"kl": 0.02})
    assert [m.step for m in run.flight_recorder.history] == [0, 1]


def test_metrics_history_written_to_manifest(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    with ProvenirRun(cfg) as run:
        run.log_step({"kl": 0.01})
        run.log_step({"kl": 0.02})
    assert run.manifest is not None
    assert len(run.manifest.metrics_history) == 2


def test_anomalies_property(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    with ProvenirRun(cfg) as run:
        run.log_step({"kl": 5.0})  # kl_blowup > kl_max 0.5
    assert any(a.kind == "kl_blowup" for a in run.anomalies)


# ---------------------------------------------------------------------------
# log_trajectory + hacking detection
# ---------------------------------------------------------------------------


def test_log_trajectory_produces_report(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    with ProvenirRun(cfg) as run:
        run.log_trajectory({"response": "A" * 3000})  # length_inflation
        run.log_trajectory({"response": "fine"})
    assert isinstance(run.hacking_report, HackingReport)
    assert run.hacking_report is not None
    assert run.hacking_report.hacking_rate > 0.0


def test_empty_trajectories_clean_report(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    with ProvenirRun(cfg) as run:
        run.log_step({"kl": 0.01})
    assert run.hacking_report is not None
    assert run.hacking_report.is_clean
    assert run.hacking_report.hacking_rate == 0.0


def test_hacking_report_file_content(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    with ProvenirRun(cfg) as run:
        run.log_trajectory({"response": "A" * 3000})
    data = json.loads(
        (Path(cfg.output_dir) / "hacking_report.json").read_text(encoding="utf-8")
    )
    assert data["num_trajectories"] == 1
    assert "length_inflation" in data["by_kind"]


# ---------------------------------------------------------------------------
# record_eval (explicit score and predictions+dataset)
# ---------------------------------------------------------------------------


def test_record_eval_explicit_score(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    with ProvenirRun(cfg) as run:
        run.record_eval("gsm8k", score=0.71)
    assert run.bom is not None
    assert run.bom.evals[0].benchmark == "gsm8k"
    assert run.bom.evals[0].score == 0.71


def test_record_eval_via_predictions(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    ds = _dataset()
    with ProvenirRun(cfg) as run:
        run.record_eval("exact", predictions=["2", "4"], eval_dataset=ds)
    assert run.bom is not None
    assert run.bom.evals[0].score == 1.0  # both predictions exactly match


def test_record_eval_partial_predictions(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    ds = _dataset()
    with ProvenirRun(cfg) as run:
        run.record_eval("exact", predictions=["2", "wrong"], eval_dataset=ds)
    assert run.bom is not None
    assert run.bom.evals[0].score == 0.5


def test_record_eval_contaminated_flag(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    with ProvenirRun(cfg) as run:
        run.record_eval("mmlu", score=0.9, contaminated=True)
    assert run.bom is not None
    assert run.bom.evals[0].contaminated is True
    assert "contaminated_eval" in run.bom.risk_flags()


def test_record_eval_requires_score_or_predictions(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    with pytest.raises(ValueError, match="either an explicit score"):
        with ProvenirRun(cfg) as run:
            run.record_eval("mmlu")


def test_record_eval_empty_benchmark(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    with pytest.raises(ValueError, match="benchmark must be non-empty"):
        with ProvenirRun(cfg) as run:
            run.record_eval("", score=0.5)


# ---------------------------------------------------------------------------
# Signed passport path
# ---------------------------------------------------------------------------


def test_signed_passport_written_and_verifies(tmp_path: Path) -> None:
    cfg = _config(tmp_path, sign_passport=True, signing_key=KEY)
    with ProvenirRun(
        cfg, config_payload={"lr": 0.001}, dataset=_dataset(), timestamp=TS
    ) as run:
        run.record_eval("gsm8k", score=0.71)
    assert (Path(cfg.output_dir) / "passport.md").exists()
    assert (Path(cfg.output_dir) / "passport.json").exists()
    assert run.passport is not None
    assert run.passport.verify(KEY) is True


def test_signed_passport_wrong_key_fails(tmp_path: Path) -> None:
    cfg = _config(tmp_path, sign_passport=True, signing_key=KEY)
    with ProvenirRun(cfg, timestamp=TS) as run:
        run.record_eval("gsm8k", score=0.71)
    assert run.passport is not None
    assert run.passport.verify(OTHER_KEY) is False


def test_signed_passport_json_round_trips(tmp_path: Path) -> None:
    cfg = _config(tmp_path, sign_passport=True, signing_key=KEY)
    with ProvenirRun(cfg, dataset=_dataset(), timestamp=TS) as run:
        run.record_eval("gsm8k", score=0.71)
    data = json.loads(
        (Path(cfg.output_dir) / "passport.json").read_text(encoding="utf-8")
    )
    passport = ModelPassport.from_dict(data)
    assert passport.verify(KEY) is True
    assert passport.bom.model_id == "run-1"


def test_unsigned_passport_is_none(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    with ProvenirRun(cfg) as run:
        run.record_eval("gsm8k", score=0.71)
    assert run.passport is None
    assert not (Path(cfg.output_dir) / "passport.json").exists()


# ---------------------------------------------------------------------------
# Lineage graph
# ---------------------------------------------------------------------------


def test_lineage_has_dataset_and_run_nodes(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    ds = _dataset()
    with ProvenirRun(cfg, dataset=ds) as run:
        run.log_step({"kl": 0.01})
    assert run.lineage is not None
    node_ids = {n["node_id"] for n in run.lineage.to_dict()["nodes"]}
    assert f"dataset:{ds.hash()}" in node_ids
    assert "run:run-1" in node_ids


def test_lineage_has_eval_nodes(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    with ProvenirRun(cfg, dataset=_dataset()) as run:
        run.record_eval("gsm8k", score=0.71)
        run.record_eval("mmlu", score=0.55)
    assert run.lineage is not None
    types = [n["node_type"] for n in run.lineage.to_dict()["nodes"]]
    assert types.count("eval") == 2


def test_lineage_run_ancestors_include_dataset(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    ds = _dataset()
    with ProvenirRun(cfg, dataset=ds) as run:
        run.log_step({"kl": 0.01})
    assert run.lineage is not None
    assert f"dataset:{ds.hash()}" in run.lineage.ancestors("run:run-1")


# ---------------------------------------------------------------------------
# BOM
# ---------------------------------------------------------------------------


def test_bom_assembled(tmp_path: Path) -> None:
    cfg = _config(tmp_path, base_model="Qwen2.5-7B")
    with ProvenirRun(cfg, config_payload={"lr": 0.001}, dataset=_dataset()) as run:
        run.record_eval("gsm8k", score=0.71)
    assert isinstance(run.bom, ModelBOM)
    assert run.bom is not None
    assert run.bom.base_model == "Qwen2.5-7B"
    assert run.bom.data[0].num_records == 2
    assert run.bom.hyperparameters == {"lr": 0.001}


# ---------------------------------------------------------------------------
# Exception inside the with-block writes partial manifest and re-raises
# ---------------------------------------------------------------------------


def test_exception_writes_partial_manifest_and_reraises(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    with pytest.raises(RuntimeError, match="boom"):
        with ProvenirRun(cfg) as run:
            run.log_step({"kl": 0.01})
            raise RuntimeError("boom")
    path = Path(cfg.output_dir) / "manifests" / "run-1.json"
    assert path.exists()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["provenance"]["partial"] is True


# ---------------------------------------------------------------------------
# track() factory and provenance_tracked decorator
# ---------------------------------------------------------------------------


def test_track_factory(tmp_path: Path) -> None:
    with track("run-1", output_dir=str(tmp_path / "art"), dataset=_dataset()) as run:
        run.log_step({"kl": 0.02})
    assert isinstance(run, ProvenirRun)
    assert run.manifest is not None
    assert (tmp_path / "art" / "manifests" / "run-1.json").exists()


def test_track_factory_forwards_config_payload(tmp_path: Path) -> None:
    with track(
        "run-1", output_dir=str(tmp_path / "art"), config_payload={"lr": 0.1}
    ) as run:
        pass
    assert run.bom is not None
    assert run.bom.hyperparameters == {"lr": 0.1}


def test_provenance_tracked_decorator(tmp_path: Path) -> None:
    @provenance_tracked("run-1", output_dir=str(tmp_path / "art"))
    def train(run: ProvenirRun, epochs: int) -> int:
        run.log_step({"kl": 0.02})
        run.record_eval("gsm8k", score=0.71)
        return epochs

    result = train(3)
    assert result == 3
    assert (tmp_path / "art" / "manifests" / "run-1.json").exists()


def test_provenance_tracked_signed(tmp_path: Path) -> None:
    @provenance_tracked(
        "run-1",
        output_dir=str(tmp_path / "art"),
        sign_passport=True,
        signing_key=KEY,
    )
    def train(run: ProvenirRun) -> None:
        run.record_eval("gsm8k", score=0.71)

    train()
    assert (tmp_path / "art" / "passport.json").exists()
