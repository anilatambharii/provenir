"""Webhook/Slack alerter — fires on anomaly or reward-hacking detection.

Uses stdlib ``urllib.request`` only; no new dependencies.
Alert failures are caught and suppressed so they never crash a training run.
"""
from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class AlertConfig:
    """Configuration for the Provenir alerter.

    Args:
        webhook_url: HTTP(S) endpoint that accepts JSON POST payloads.
            Slack incoming webhooks, PagerDuty, and generic HTTP endpoints
            all work. Empty string disables alerting.
        on_anomaly: Fire when the flight recorder detects an anomaly.
        on_hacking: Fire when the reward-hacking batch report is not clean.
        min_severity: Minimum anomaly severity to alert on ('low', 'medium', 'high').

    Example:
        >>> cfg = AlertConfig(webhook_url="", on_anomaly=True)
        >>> cfg.enabled
        False
    """

    webhook_url: str = ""
    on_anomaly: bool = True
    on_hacking: bool = True
    min_severity: str = "medium"

    @property
    def enabled(self) -> bool:
        return bool(self.webhook_url)

    def __post_init__(self) -> None:
        if self.min_severity not in ("low", "medium", "high"):
            raise ValueError(f"min_severity must be low/medium/high, got {self.min_severity!r}")


_SEVERITY_ORDER: dict[str, int] = {"low": 0, "medium": 1, "high": 2}


@dataclass
class Alert:
    """A single alert payload.

    Args:
        kind: Category of event (e.g. ``'anomaly'``, ``'reward_hacking'``).
        severity: ``'low'``, ``'medium'``, or ``'high'``.
        run_id: The ProvenirRun that fired this alert.
        payload: Extra key/value context sent in the webhook body.

    Example:
        >>> a = Alert(kind="anomaly", severity="high", run_id="run-1", payload={"step": 5})
        >>> a.to_dict()["kind"]
        'anomaly'
    """

    kind: str
    severity: str
    run_id: str
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": "provenir",
            "kind": self.kind,
            "severity": self.severity,
            "run_id": self.run_id,
            **self.payload,
        }


class Alerter:
    """Posts :class:`Alert` payloads to a webhook endpoint.

    Designed to be instantiated once per :class:`~provenir.integrations.wrapper.ProvenirRun`
    and kept alive for the duration of the run.  All network errors are caught
    and silently discarded — alerting must never crash a training loop.

    Example:
        >>> cfg = AlertConfig(webhook_url="", on_anomaly=True)
        >>> alerter = Alerter(cfg, run_id="run-1")
        >>> alerter.fire(Alert("test", "low", "run-1", {}))  # no-op; URL empty
    """

    def __init__(self, config: AlertConfig, run_id: str = "") -> None:
        self.config = config
        self.run_id = run_id
        self._fired: list[Alert] = []

    @property
    def fired(self) -> list[Alert]:
        """All alerts fired this session (useful for testing without a real URL)."""
        return list(self._fired)

    def fire(self, alert: Alert) -> None:
        """POST ``alert`` to the webhook; silently drops on any error."""
        self._fired.append(alert)
        if not self.config.enabled:
            return
        try:
            body = json.dumps(alert.to_dict()).encode("utf-8")
            req = urllib.request.Request(
                self.config.webhook_url,
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=5):  # noqa: S310 — user-supplied URL
                pass
        except Exception:  # pragma: no cover — network errors must not crash runs
            pass

    def fire_anomaly(self, anomaly: Any) -> None:
        """Fire if ``on_anomaly`` is set and anomaly severity >= ``min_severity``."""
        if not self.config.on_anomaly:
            return
        severity = getattr(anomaly, "severity", "medium")
        if _SEVERITY_ORDER.get(severity, 1) < _SEVERITY_ORDER.get(self.config.min_severity, 1):
            return
        self.fire(
            Alert(
                kind="anomaly",
                severity=severity,
                run_id=self.run_id,
                payload={
                    "anomaly_kind": getattr(anomaly, "kind", "unknown"),
                    "step": getattr(anomaly, "step", -1),
                    "detail": getattr(anomaly, "detail", ""),
                },
            )
        )

    def fire_hacking(self, hacking_rate: float, by_kind: dict[str, int]) -> None:
        """Fire a hacking alert if ``on_hacking`` is set."""
        if not self.config.on_hacking:
            return
        self.fire(
            Alert(
                kind="reward_hacking",
                severity="high",
                run_id=self.run_id,
                payload={"hacking_rate": hacking_rate, "by_kind": by_kind},
            )
        )
