"""Provenir — the trust layer for model post-training.

Reproducibility, contamination-safe evaluation, reward-hacking detection, and
signed provenance for any fine-tuning or RL run — one import, any backend.

Quick start::

    import provenir

    with provenir.track("my-run", dataset=train_ds) as run:
        for step, metrics in training_loop():
            run.log_step(metrics)          # RL flight recorder + anomaly detection
        run.record_eval("mmlu", score=0.71)
    # run.manifest, run.flight_recorder, run.hacking_report, run.passport
"""

from __future__ import annotations

from provenir.integrations.wrapper import (
    ProvenirRun,
    TrackingConfig,
    provenance_tracked,
    track,
)

__all__ = [
    "__version__",
    "ProvenirRun",
    "TrackingConfig",
    "track",
    "provenance_tracked",
]
__version__ = "0.4.0"
