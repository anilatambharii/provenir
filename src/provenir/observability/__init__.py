"""RL Flight Recorder + comprehensive reward-hacking detection.

RL-native observability for reinforcement-learning runs.  The
:class:`FlightRecorder` is a "black box": feed it one :class:`RLStepMetrics`
per training step and it raises :class:`Anomaly` records for KL blowup/collapse,
entropy collapse, response-length explosion, GRPO advantage collapse, reward-std
collapse, reward spikes, and gradient explosion.  The
:class:`RewardHackingDetector` inspects trajectories and reward groups for
length inflation, format exploits, verifier gaming, test tampering, proxy/true
reward divergence, degenerate repetition, and advantage collapse.

Example::

    from provenir.observability import FlightRecorder, RLStepMetrics

    rec = FlightRecorder()
    rec.log_step(RLStepMetrics(step=0, kl=0.02, entropy=1.5, advantage_std=0.3))
    print(rec.health_report())
"""

from __future__ import annotations

from provenir.observability.flight_recorder import (
    ANOMALY_KINDS,
    Anomaly,
    FlightRecorder,
    FlightRecorderConfig,
    RLStepMetrics,
)
from provenir.observability.reward_hacking import (
    HACKING_KINDS,
    HackingReport,
    HackingSignal,
    RewardHackingConfig,
    RewardHackingDetector,
)

__all__ = [
    "ANOMALY_KINDS",
    "HACKING_KINDS",
    "Anomaly",
    "FlightRecorder",
    "FlightRecorderConfig",
    "HackingReport",
    "HackingSignal",
    "RLStepMetrics",
    "RewardHackingConfig",
    "RewardHackingDetector",
]
