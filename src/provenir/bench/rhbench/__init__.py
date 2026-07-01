"""RH-Bench: a reproducible benchmark for evaluating reward-hacking *detectors*.

RH-Bench does not claim to be the first reward-hacking-detection benchmark —
prior art includes TRACE (arXiv:2601.20103), "Cheap Reward Hacking Detection"
(arXiv:2606.08893), Shihab et al. (arXiv:2507.05619), and EvalStop
(arXiv:2606.04145).  Its differentiators are:

1. **Held-out hack-TYPE generalization** — train on some hack categories, test
   on unseen ones (:meth:`LabeledCorpus.split_by_hack_type` /
   :meth:`BenchmarkHarness.evaluate_held_out`), not merely held-out
   environments.
2. **A compute-saved-by-early-detection metric** wired to a live training gate
   (:func:`compute_savings`, cf. :mod:`provenir.train.rl_eval_gate`).
3. **A single reproducible artifact** — a deterministic synthetic corpus plus
   baseline detectors — integrated into the Provenir fine-tuning framework.

Example::

    from provenir.bench.rhbench import (
        SyntheticCorpusGenerator, BenchmarkHarness, ProvenirDetector,
    )

    corpus = SyntheticCorpusGenerator(seed=0).generate()
    ev = BenchmarkHarness().evaluate(ProvenirDetector(), corpus)
    print(ev.to_markdown())
"""

from __future__ import annotations

from provenir.bench.rhbench.baselines import (
    AlwaysHackDetector,
    LengthHeuristicDetector,
    NeverHackDetector,
    ProvenirDetector,
    ProxyDivergenceDetector,
    RandomDetector,
)
from provenir.bench.rhbench.corpus import (
    LabeledCorpus,
    SyntheticCorpusGenerator,
    Trajectory,
)
from provenir.bench.rhbench.harness import (
    BenchmarkHarness,
    ComputeSavedResult,
    compute_savings,
)
from provenir.bench.rhbench.protocol import (
    CategoryMetrics,
    DetectionResult,
    Detector,
    DetectorEvaluation,
    auroc,
    precision_recall_f1,
)
from provenir.bench.rhbench.taxonomy import CATEGORY_DEFINITIONS, HackCategory

__all__ = [
    "CATEGORY_DEFINITIONS",
    "HackCategory",
    "Trajectory",
    "LabeledCorpus",
    "SyntheticCorpusGenerator",
    "DetectionResult",
    "Detector",
    "CategoryMetrics",
    "DetectorEvaluation",
    "precision_recall_f1",
    "auroc",
    "BenchmarkHarness",
    "ComputeSavedResult",
    "compute_savings",
    "ProvenirDetector",
    "LengthHeuristicDetector",
    "ProxyDivergenceDetector",
    "RandomDetector",
    "AlwaysHackDetector",
    "NeverHackDetector",
]
