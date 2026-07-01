from __future__ import annotations

import pytest

from provenir.observability.flight_recorder import (
    ANOMALY_KINDS,
    Anomaly,
    FlightRecorder,
    FlightRecorderConfig,
    RLStepMetrics,
)


def _healthy(step: int) -> RLStepMetrics:
    """A metrics sample that trips no detectors under the default config."""
    return RLStepMetrics(
        step=step,
        kl=0.02,
        entropy=1.5,
        reward_mean=0.5,
        reward_std=0.2,
        response_length_mean=100.0,
        advantage_std=0.3,
        grad_norm=2.0,
        learning_rate=1e-5,
    )


# --------------------------------------------------------------------------
# dataclass validation / serialization
# --------------------------------------------------------------------------


def test_rlstepmetrics_defaults_zero() -> None:
    m = RLStepMetrics(step=3)
    assert m.kl == 0.0 and m.entropy == 0.0 and m.grad_norm == 0.0


def test_rlstepmetrics_to_dict_roundtrip() -> None:
    m = RLStepMetrics(step=1, kl=0.1, entropy=1.2)
    d = m.to_dict()
    assert d["step"] == 1 and d["kl"] == 0.1 and d["entropy"] == 1.2


def test_anomaly_rejects_bad_severity() -> None:
    with pytest.raises(ValueError):
        Anomaly(step=1, kind="kl_blowup", severity="fatal", detail="x", value=1.0)


def test_anomaly_accepts_valid_severities() -> None:
    for sev in ("warn", "critical"):
        a = Anomaly(step=1, kind="reward_spike", severity=sev, detail="x", value=1.0)
        assert a.severity == sev


def test_anomaly_to_dict() -> None:
    a = Anomaly(step=2, kind="grad_explosion", severity="critical", detail="d", value=9.0)
    assert a.to_dict() == {
        "step": 2,
        "kind": "grad_explosion",
        "severity": "critical",
        "detail": "d",
        "value": 9.0,
    }


def test_all_anomaly_kinds_documented() -> None:
    assert set(ANOMALY_KINDS) == {
        "kl_blowup",
        "kl_collapse",
        "entropy_collapse",
        "length_explosion",
        "advantage_collapse",
        "reward_std_collapse",
        "reward_spike",
        "grad_explosion",
    }


# --------------------------------------------------------------------------
# config validation
# --------------------------------------------------------------------------


def test_config_defaults() -> None:
    cfg = FlightRecorderConfig()
    assert cfg.kl_max == 0.5 and cfg.window == 20


@pytest.mark.parametrize(
    "kwargs",
    [
        {"kl_max": 0.0},
        {"kl_min": -1.0},
        {"entropy_min": 0.0},
        {"entropy_drop_frac": -0.1},
        {"length_explosion_ratio": 0.0},
        {"advantage_std_min": 0.0},
        {"reward_std_min": -1.0},
        {"grad_norm_max": 0.0},
        {"reward_spike_z": 0.0},
    ],
)
def test_config_rejects_nonpositive(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        FlightRecorderConfig(**kwargs)


def test_config_rejects_small_window() -> None:
    with pytest.raises(ValueError):
        FlightRecorderConfig(window=1)


# --------------------------------------------------------------------------
# healthy stream stays silent
# --------------------------------------------------------------------------


def test_healthy_stream_no_anomalies() -> None:
    rec = FlightRecorder()
    for step in range(30):
        assert rec.log_step(_healthy(step)) == []
    assert rec.anomalies == []
    assert rec.summary()["verdict"] == "HEALTHY"


# --------------------------------------------------------------------------
# KL detectors
# --------------------------------------------------------------------------


def test_kl_blowup_fires() -> None:
    rec = FlightRecorder()
    out = rec.log_step(RLStepMetrics(step=0, kl=0.9, entropy=1.5, advantage_std=0.3))
    kinds = [a.kind for a in out]
    assert "kl_blowup" in kinds
    assert next(a for a in out if a.kind == "kl_blowup").severity == "critical"


def test_kl_collapse_fires() -> None:
    rec = FlightRecorder()
    out = rec.log_step(RLStepMetrics(step=0, kl=1e-9, entropy=1.5, advantage_std=0.3))
    assert "kl_collapse" in [a.kind for a in out]


def test_kl_zero_is_not_collapse() -> None:
    # kl == 0.0 means "not recorded"; do not flag collapse.
    rec = FlightRecorder()
    out = rec.log_step(RLStepMetrics(step=0, kl=0.0, entropy=1.5, advantage_std=0.3))
    assert "kl_collapse" not in [a.kind for a in out]


def test_kl_healthy_no_fire() -> None:
    rec = FlightRecorder()
    out = rec.log_step(RLStepMetrics(step=0, kl=0.1, entropy=1.5, advantage_std=0.3))
    assert not any(a.kind.startswith("kl_") for a in out)


# --------------------------------------------------------------------------
# entropy detectors
# --------------------------------------------------------------------------


def test_entropy_below_floor_warns() -> None:
    rec = FlightRecorder()
    # First step sets running max at 0.05 (already below floor 0.1) -> warn only,
    # not a drop (drop compares to same-step max).
    out = rec.log_step(RLStepMetrics(step=0, kl=0.02, entropy=0.05, advantage_std=0.3))
    collapse = [a for a in out if a.kind == "entropy_collapse"]
    assert collapse and collapse[0].severity == "warn"


def test_entropy_sharp_drop_is_critical() -> None:
    rec = FlightRecorder()
    rec.log_step(RLStepMetrics(step=0, kl=0.02, entropy=2.0, advantage_std=0.3))
    out = rec.log_step(RLStepMetrics(step=1, kl=0.02, entropy=0.5, advantage_std=0.3))
    collapse = [a for a in out if a.kind == "entropy_collapse"]
    assert collapse and collapse[0].severity == "critical"


def test_entropy_stable_no_fire() -> None:
    rec = FlightRecorder()
    rec.log_step(RLStepMetrics(step=0, kl=0.02, entropy=2.0, advantage_std=0.3))
    out = rec.log_step(RLStepMetrics(step=1, kl=0.02, entropy=1.9, advantage_std=0.3))
    assert "entropy_collapse" not in [a.kind for a in out]


# --------------------------------------------------------------------------
# length explosion
# --------------------------------------------------------------------------


def test_length_explosion_fires_vs_baseline() -> None:
    rec = FlightRecorder()
    rec.log_step(RLStepMetrics(step=0, kl=0.02, entropy=1.5, advantage_std=0.3,
                               response_length_mean=100.0))
    out = rec.log_step(RLStepMetrics(step=1, kl=0.02, entropy=1.5, advantage_std=0.3,
                                     response_length_mean=400.0))
    assert "length_explosion" in [a.kind for a in out]


def test_length_within_ratio_no_fire() -> None:
    rec = FlightRecorder()
    rec.log_step(RLStepMetrics(step=0, kl=0.02, entropy=1.5, advantage_std=0.3,
                               response_length_mean=100.0))
    out = rec.log_step(RLStepMetrics(step=1, kl=0.02, entropy=1.5, advantage_std=0.3,
                                     response_length_mean=250.0))
    assert "length_explosion" not in [a.kind for a in out]


def test_length_first_step_sets_baseline_no_fire() -> None:
    rec = FlightRecorder()
    out = rec.log_step(RLStepMetrics(step=0, kl=0.02, entropy=1.5, advantage_std=0.3,
                                     response_length_mean=5000.0))
    assert "length_explosion" not in [a.kind for a in out]


# --------------------------------------------------------------------------
# advantage collapse (GRPO)
# --------------------------------------------------------------------------


def test_advantage_collapse_fires_when_zero() -> None:
    rec = FlightRecorder()
    out = rec.log_step(RLStepMetrics(step=0, kl=0.02, entropy=1.5, advantage_std=0.0))
    collapse = [a for a in out if a.kind == "advantage_collapse"]
    assert collapse and collapse[0].severity == "critical"


def test_advantage_healthy_no_fire() -> None:
    rec = FlightRecorder()
    out = rec.log_step(RLStepMetrics(step=0, kl=0.02, entropy=1.5, advantage_std=0.5))
    assert "advantage_collapse" not in [a.kind for a in out]


# --------------------------------------------------------------------------
# reward std collapse
# --------------------------------------------------------------------------


def test_reward_std_collapse_fires() -> None:
    rec = FlightRecorder()
    out = rec.log_step(RLStepMetrics(step=0, kl=0.02, entropy=1.5, advantage_std=0.3,
                                     reward_std=0.0))
    assert "reward_std_collapse" in [a.kind for a in out]


def test_reward_std_healthy_no_fire() -> None:
    rec = FlightRecorder()
    out = rec.log_step(RLStepMetrics(step=0, kl=0.02, entropy=1.5, advantage_std=0.3,
                                     reward_std=0.2))
    assert "reward_std_collapse" not in [a.kind for a in out]


# --------------------------------------------------------------------------
# reward spike (z-score over rolling window)
# --------------------------------------------------------------------------


def test_reward_spike_fires_after_stable_window() -> None:
    rec = FlightRecorder()
    # A stable-but-not-constant window (tiny variance) so the z-score is defined.
    stable = [0.50, 0.51, 0.49, 0.50, 0.52, 0.48, 0.50, 0.51, 0.49, 0.50]
    for step, reward in enumerate(stable):
        rec.log_step(RLStepMetrics(step=step, kl=0.02, entropy=1.5, advantage_std=0.3,
                                   reward_mean=reward, reward_std=0.2))
    out = rec.log_step(RLStepMetrics(step=10, kl=0.02, entropy=1.5, advantage_std=0.3,
                                     reward_mean=5.0, reward_std=0.2))
    assert "reward_spike" in [a.kind for a in out]


def test_reward_spike_needs_history() -> None:
    rec = FlightRecorder()
    out = rec.log_step(RLStepMetrics(step=0, kl=0.02, entropy=1.5, advantage_std=0.3,
                                     reward_mean=99.0, reward_std=0.2))
    assert "reward_spike" not in [a.kind for a in out]


def test_reward_spike_stable_stream_silent() -> None:
    rec = FlightRecorder()
    fired = False
    for step in range(15):
        out = rec.log_step(RLStepMetrics(step=step, kl=0.02, entropy=1.5, advantage_std=0.3,
                                         reward_mean=0.5 + 0.001 * step, reward_std=0.2))
        fired = fired or any(a.kind == "reward_spike" for a in out)
    assert not fired


# --------------------------------------------------------------------------
# grad explosion
# --------------------------------------------------------------------------


def test_grad_explosion_fires() -> None:
    rec = FlightRecorder()
    out = rec.log_step(RLStepMetrics(step=0, kl=0.02, entropy=1.5, advantage_std=0.3,
                                     grad_norm=500.0))
    assert "grad_explosion" in [a.kind for a in out]


def test_grad_explosion_on_inf() -> None:
    rec = FlightRecorder()
    out = rec.log_step(RLStepMetrics(step=0, kl=0.02, entropy=1.5, advantage_std=0.3,
                                     grad_norm=float("inf")))
    grad = [a for a in out if a.kind == "grad_explosion"]
    assert grad and grad[0].severity == "critical"


def test_grad_healthy_no_fire() -> None:
    rec = FlightRecorder()
    out = rec.log_step(RLStepMetrics(step=0, kl=0.02, entropy=1.5, advantage_std=0.3,
                                     grad_norm=3.0))
    assert "grad_explosion" not in [a.kind for a in out]


# --------------------------------------------------------------------------
# history / anomalies properties
# --------------------------------------------------------------------------


def test_history_is_ordered_copy() -> None:
    rec = FlightRecorder()
    rec.log_step(_healthy(0))
    rec.log_step(_healthy(1))
    hist = rec.history
    assert [m.step for m in hist] == [0, 1]
    hist.clear()
    assert len(rec.history) == 2  # mutating the copy does not affect the recorder


def test_anomalies_accumulate() -> None:
    rec = FlightRecorder()
    rec.log_step(RLStepMetrics(step=0, kl=0.9, entropy=1.5, advantage_std=0.3))
    rec.log_step(RLStepMetrics(step=1, kl=0.02, entropy=1.5, advantage_std=0.0))
    kinds = [a.kind for a in rec.anomalies]
    assert "kl_blowup" in kinds and "advantage_collapse" in kinds


# --------------------------------------------------------------------------
# summary / health_report / to_dict
# --------------------------------------------------------------------------


def test_summary_counts_and_verdict() -> None:
    rec = FlightRecorder()
    rec.log_step(RLStepMetrics(step=0, kl=0.9, entropy=1.5, advantage_std=0.3))
    summary = rec.summary()
    assert summary["num_steps"] == 1
    assert summary["by_kind"]["kl_blowup"] == 1
    assert summary["by_severity"]["critical"] == 1
    assert summary["verdict"] == "CRITICAL"
    assert summary["final_metrics"]["kl"] == 0.9


def test_summary_empty_recorder() -> None:
    rec = FlightRecorder()
    summary = rec.summary()
    assert summary["num_steps"] == 0
    assert summary["final_metrics"] is None
    assert summary["verdict"] == "HEALTHY"


def test_verdict_degraded_when_only_warnings() -> None:
    rec = FlightRecorder()
    rec.log_step(RLStepMetrics(step=0, kl=1e-9, entropy=1.5, advantage_std=0.3))
    assert rec.summary()["verdict"] == "DEGRADED"


def test_health_report_healthy() -> None:
    rec = FlightRecorder()
    rec.log_step(_healthy(0))
    report = rec.health_report()
    assert "HEALTHY" in report and "healthy" in report


def test_health_report_lists_detectors() -> None:
    rec = FlightRecorder()
    rec.log_step(RLStepMetrics(step=0, kl=0.9, entropy=1.5, advantage_std=0.3))
    report = rec.health_report()
    assert "CRITICAL" in report
    assert "kl_blowup" in report
    assert "latest critical" in report


def test_to_dict_full_serialization() -> None:
    rec = FlightRecorder()
    rec.log_step(RLStepMetrics(step=0, kl=0.9, entropy=1.5, advantage_std=0.3))
    d = rec.to_dict()
    assert set(d) == {"config", "history", "anomalies", "summary"}
    assert d["config"]["kl_max"] == 0.5
    assert len(d["history"]) == 1
    assert d["anomalies"][0]["kind"] == "kl_blowup"
    assert d["summary"]["verdict"] == "CRITICAL"


def test_custom_config_threshold_applies() -> None:
    rec = FlightRecorder(FlightRecorderConfig(kl_max=0.05))
    out = rec.log_step(RLStepMetrics(step=0, kl=0.1, entropy=1.5, advantage_std=0.3))
    assert "kl_blowup" in [a.kind for a in out]
