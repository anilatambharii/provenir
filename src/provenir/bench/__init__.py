"""Research benchmark modules for Provenir.

Currently hosts :mod:`provenir.bench.rhbench` (RH-Bench), a reproducible
benchmark for evaluating reward-hacking detectors.

Example::

    from provenir.bench.rhbench import SyntheticCorpusGenerator

    corpus = SyntheticCorpusGenerator(seed=0).generate()
"""

from __future__ import annotations

from provenir.bench import rhbench

__all__ = ["rhbench"]
