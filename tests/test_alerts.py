"""Tests for the provenir.alerts webhook alerter."""
from __future__ import annotations

import pytest

from provenir.alerts import Alert, AlertConfig, Alerter
from provenir.integrations.wrapper import TrackingConfig


class TestAlertConfig:
    def test_alert_config_defaults(self) -> None:
        cfg = AlertConfig()
        assert cfg.enabled is False
        assert cfg.on_anomaly is True
        assert cfg.on_hacking is True
        assert cfg.min_severity == "medium"

    def test_alert_config_enabled(self) -> None:
        cfg = AlertConfig(webhook_url="https://example.com/hook")
        assert cfg.enabled is True

    def test_alert_config_invalid_severity(self) -> None:
        with pytest.raises(ValueError, match="min_severity"):
            AlertConfig(min_severity="extreme")

    def test_alert_config_empty_url_disabled(self) -> None:
        cfg = AlertConfig(webhook_url="")
        assert cfg.enabled is False

    def test_alert_config_frozen(self) -> None:
        cfg = AlertConfig()
        with pytest.raises((AttributeError, TypeError)):
            cfg.webhook_url = "http://x"  # type: ignore[misc]


class TestAlert:
    def test_alert_to_dict_has_kind(self) -> None:
        a = Alert(kind="anomaly", severity="high", run_id="run-1", payload={"step": 5})
        d = a.to_dict()
        assert d["kind"] == "anomaly"
        assert d["severity"] == "high"
        assert d["run_id"] == "run-1"
        assert d["source"] == "provenir"
        assert d["step"] == 5

    def test_alert_payload_merged_into_dict(self) -> None:
        a = Alert(kind="test", severity="low", run_id="r", payload={"foo": "bar"})
        d = a.to_dict()
        assert d["foo"] == "bar"


class TestAlerter:
    def test_alerter_no_url_no_error(self) -> None:
        cfg = AlertConfig(webhook_url="", on_anomaly=True)
        alerter = Alerter(cfg, run_id="run-1")
        alert = Alert("test", "low", "run-1", {})
        alerter.fire(alert)  # should not raise

    def test_alerter_records_fired(self) -> None:
        cfg = AlertConfig(webhook_url="")
        alerter = Alerter(cfg, run_id="run-1")
        alerter.fire(Alert("anomaly", "high", "run-1", {}))
        alerter.fire(Alert("reward_hacking", "high", "run-1", {}))
        assert len(alerter.fired) == 2

    def test_alerter_fire_anomaly_severity_gate(self) -> None:
        """low anomaly with min_severity=high should NOT fire."""
        cfg = AlertConfig(webhook_url="", on_anomaly=True, min_severity="high")
        alerter = Alerter(cfg, run_id="run-1")

        class StubAnomaly:
            severity = "low"
            kind = "kl_blowup"
            step = 3
            detail = "test"

        alerter.fire_anomaly(StubAnomaly())
        assert len(alerter.fired) == 0

    def test_alerter_fire_anomaly_passes_when_severity_meets_threshold(self) -> None:
        cfg = AlertConfig(webhook_url="", on_anomaly=True, min_severity="medium")
        alerter = Alerter(cfg, run_id="run-1")

        class StubAnomaly:
            severity = "high"
            kind = "grad_explosion"
            step = 5
            detail = "big grad"

        alerter.fire_anomaly(StubAnomaly())
        assert len(alerter.fired) == 1
        assert alerter.fired[0].kind == "anomaly"

    def test_alerter_fire_hacking(self) -> None:
        cfg = AlertConfig(webhook_url="", on_hacking=True)
        alerter = Alerter(cfg, run_id="run-2")
        alerter.fire_hacking(0.25, {"shortcut": 3})
        assert len(alerter.fired) == 1
        fired = alerter.fired[0]
        assert fired.kind == "reward_hacking"
        assert fired.severity == "high"
        assert fired.payload["hacking_rate"] == pytest.approx(0.25)

    def test_alerter_fire_hacking_disabled(self) -> None:
        cfg = AlertConfig(webhook_url="", on_hacking=False)
        alerter = Alerter(cfg, run_id="run-3")
        alerter.fire_hacking(0.5, {})
        assert len(alerter.fired) == 0

    def test_alerter_fire_anomaly_disabled(self) -> None:
        cfg = AlertConfig(webhook_url="", on_anomaly=False)
        alerter = Alerter(cfg, run_id="run-4")

        class StubAnomaly:
            severity = "critical"
            kind = "kl_blowup"
            step = 1
            detail = ""

        alerter.fire_anomaly(StubAnomaly())
        assert len(alerter.fired) == 0

    def test_fired_returns_copy(self) -> None:
        cfg = AlertConfig(webhook_url="")
        alerter = Alerter(cfg, run_id="run-5")
        fired_copy = alerter.fired
        alerter.fire(Alert("x", "low", "run-5", {}))
        assert len(fired_copy) == 0  # original copy not mutated


class TestTrackingConfigAlertFields:
    def test_tracking_config_alert_fields_defaults(self) -> None:
        cfg = TrackingConfig(name="x")
        assert cfg.alert_webhook_url == ""
        assert cfg.alert_on_anomaly is True
        assert cfg.alert_on_hacking is True

    def test_tracking_config_alert_webhook_url_set(self) -> None:
        cfg = TrackingConfig(name="x", alert_webhook_url="http://x.com")
        assert cfg.alert_webhook_url == "http://x.com"


class TestProvenirRunAlerterProperty:
    def test_provenir_run_alerter_property(self, tmp_path: object) -> None:
        import tempfile

        from provenir.integrations.wrapper import track

        with tempfile.TemporaryDirectory() as tmpdir:
            with track("alert-test-run", output_dir=tmpdir, alert_webhook_url="") as run:
                run.log_step({"kl": 0.01, "step": 0})

            alerter = run.alerter
            # alerter exists and is an Alerter
            assert alerter is not None
            assert isinstance(alerter.fired, list)
