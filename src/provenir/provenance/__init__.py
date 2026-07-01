"""Deterministic run replay and provenance lineage DAG.

This package implements Provenir's tamper-evident audit trail: environment
fingerprinting for bitwise reproducibility, a content-addressed lineage DAG
linking datasets, runs, adapters and evals, and a replay engine that verifies
a stored run can be deterministically reproduced. Together these map to the
EU AI Act Article 12 requirements for audit trails and model lineage.

Example:
    >>> from provenir.provenance import capture_fingerprint, LineageGraph, LineageNode
    >>> fp = capture_fingerprint({"numpy": "1.26.4"})
    >>> g = LineageGraph()
    >>> g.add_node(LineageNode("ds", "dataset", fp.packages_hash, {}))
    >>> g.roots()
    ['ds']
"""

from __future__ import annotations

from provenir.provenance.fingerprint import (
    EnvironmentFingerprint,
    capture_fingerprint,
    kernel_determinism_flags,
)
from provenir.provenance.lineage import (
    LineageEdge,
    LineageGraph,
    LineageNode,
    LineageStore,
)
from provenir.provenance.replay import ReplayEngine, ReplayVerification

__all__ = [
    "EnvironmentFingerprint",
    "capture_fingerprint",
    "kernel_determinism_flags",
    "LineageNode",
    "LineageEdge",
    "LineageGraph",
    "LineageStore",
    "ReplayVerification",
    "ReplayEngine",
]
