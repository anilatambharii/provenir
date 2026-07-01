"""The viral substrate: ``import provenir`` in three lines.

This package is the drop-in wrapper that instruments any existing training
run (TRL / verl / Unsloth / custom) with Provenir's full trust stack —
provenance fingerprinting, an RL flight recorder, reward-hacking detection, a
content-addressed lineage DAG, a Model Bill-of-Materials, and a signed Model
Passport — with almost no code.

Example:
    >>> from provenir.integrations import track
    >>> import tempfile
    >>> with track("my-run", output_dir=tempfile.mkdtemp()) as run:
    ...     run.log_step({"kl": 0.02, "entropy": 1.4})
    ...     run.record_eval("gsm8k", score=0.71)
    >>> run.manifest is not None
    True
"""

from __future__ import annotations

from provenir.integrations.wrapper import (
    ProvenirRun,
    TrackingConfig,
    provenance_tracked,
    track,
)

__all__ = [
    "TrackingConfig",
    "ProvenirRun",
    "track",
    "provenance_tracked",
]
