"""
RH-Bench: evaluate reward-hacking detectors on a labeled trajectory corpus.

Reproduces the benchmark's headline table and the held-out-hack-type +
compute-saved metrics. Pure-Python, deterministic (no GPU, no deps).

Honest framing: RH-Bench unifies and integrates ideas from TRACE
(arXiv:2601.20103), "Cheap Reward Hacking Detection" (arXiv:2606.08893), and
EvalStop (arXiv:2606.04145); it is not the first reward-hacking-detection
benchmark. Its deltas are held-out hack-TYPE generalization and a compute-saved
metric wired to a live training gate.

Usage:
    python examples/rhbench_demo.py
"""

from __future__ import annotations

from provenir.bench.rhbench import (
    BenchmarkHarness,
    LengthHeuristicDetector,
    ProvenirDetector,
    ProxyDivergenceDetector,
    RandomDetector,
    SyntheticCorpusGenerator,
    compute_savings,
)

corpus = SyntheticCorpusGenerator(seed=0).generate(n_per_category=50, n_clean=100)
print(f"Corpus: {corpus.size} trajectories "
      f"({corpus.hack_count} hacks / {corpus.clean_count} clean), "
      f"{len(corpus.categories())} hack categories\n")

harness = BenchmarkHarness()
detectors = [
    ProvenirDetector(),
    ProxyDivergenceDetector(),
    LengthHeuristicDetector(),
    RandomDetector(seed=0),
]

print("=== Detector evaluation (overall) ===")
print(f"{'detector':<20} {'AUROC':>7} {'F1':>7} {'precision':>10} {'recall':>7}")
for det in detectors:
    ev = harness.evaluate(det, corpus)
    print(f"{ev.detector_name:<20} {ev.auroc:>7.3f} {ev.f1:>7.3f} "
          f"{ev.precision:>10.3f} {ev.recall:>7.3f}")

# Held-out hack-TYPE generalization: hide two categories at "training", test on them.
held_out = ["sycophancy", "answer_leakage"]
print(f"\n=== Held-out hack types {held_out} (generalization to unseen hacks) ===")
ev = harness.evaluate_held_out(ProvenirDetector(), corpus, held_out)
print(f"ProvenirDetector on unseen hack types: AUROC={ev.auroc:.3f} F1={ev.f1:.3f}")

# Compute-saved-by-early-detection: a run that starts hacking at step 20 of 100.
run = [t for t in corpus.trajectories if not t.is_hack][:20]
run += [t for t in corpus.trajectories if t.is_hack][:80]
saved = compute_savings(ProvenirDetector(), run, hack_onset_step=20, total_steps=100)
print("\n=== Compute saved by early detection ===")
print(f"hack onset step:  {saved.hack_onset_step}")
print(f"detected at step: {saved.detection_step}")
print(f"steps saved:      {saved.steps_saved} / {saved.total_steps} "
      f"({saved.fraction_saved:.0%})")

print("\nTable 1 (markdown):\n")
print(harness.evaluate(ProvenirDetector(), corpus).to_markdown())
