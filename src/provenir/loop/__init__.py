"""The Loop Doctor: differential diagnosis + adaptive control for training loops.

When a loop stalls, :class:`~provenir.loop.doctor.LoopDoctor` attributes the
stall to a *data*, *reward*, *eval*, or *algorithm* cause using Provenir's trust
signals, and — when the cause is data — emits a concrete
:class:`~provenir.loop.doctor.DataRequest`.
:class:`~provenir.loop.doctor.LoopController` turns that verdict into the next
action.
"""

from __future__ import annotations

from provenir.loop.doctor import (
    DataRequest,
    Diagnosis,
    DoctorConfig,
    Finding,
    LoopAction,
    LoopController,
    LoopDoctor,
    LoopSignals,
)
from provenir.loop.slices import SliceAnalyzer, SliceReport

__all__ = [
    "DataRequest",
    "Diagnosis",
    "DoctorConfig",
    "Finding",
    "LoopAction",
    "LoopController",
    "LoopDoctor",
    "LoopSignals",
    "SliceAnalyzer",
    "SliceReport",
]
