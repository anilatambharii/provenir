"""Slice-level failure analysis for the Loop Doctor.

When a training loop plateaus, the residual cause is often a *data* problem —
but "data problem" is useless without knowing *which* data. This module
localizes the failure: bucket predictions by a slice key (a record field like
``"category"`` or an arbitrary callable), score each with a
:class:`provenir.eval.metrics.MetricFn`, and compute a per-slice failure rate.

The resulting :class:`SliceReport` feeds
:class:`provenir.loop.doctor.LoopDoctor`, which turns the worst slices into a
concrete, human-facing ``DataRequest`` ("give me N more recent examples of
these slices").

Example
-------
>>> from provenir.eval.metrics import ExactMatchMetric
>>> analyzer = SliceAnalyzer(ExactMatchMetric(), failure_threshold=0.5)
>>> records = [
...     {"category": "math", "response": "4"},
...     {"category": "math", "response": "6"},
...     {"category": "code", "response": "ok"},
... ]
>>> report = analyzer.analyze(records, ["4", "5", "ok"], slice_key="category")
>>> report.slice_failures["math"]
0.5
>>> report.worst_slices(1)
[('math', 0.5)]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from provenir.eval.metrics import MetricFn

#: Record keys that may carry the gold/reference answer, tried in order.
_REFERENCE_KEYS: tuple[str, ...] = ("response", "reference")


@dataclass(frozen=True)
class SliceReport:
    """Per-slice failure rates from a :class:`SliceAnalyzer` run.

    Attributes
    ----------
    slice_failures:
        Slice name -> failure rate in ``[0, 1]`` (fraction of that slice's
        predictions that scored below the analyzer's threshold).
    slice_counts:
        Slice name -> number of examples bucketed into that slice.
    overall_failure_rate:
        Fraction of *all* predictions that failed, across every slice.

    Example::

        report = SliceReport(
            slice_failures={"math": 0.8, "code": 0.1},
            slice_counts={"math": 10, "code": 10},
            overall_failure_rate=0.45,
        )
        assert report.worst_slices(1) == [("math", 0.8)]
    """

    slice_failures: dict[str, float]
    slice_counts: dict[str, int]
    overall_failure_rate: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.overall_failure_rate <= 1.0:
            raise ValueError(
                f"overall_failure_rate must be in [0.0, 1.0], got {self.overall_failure_rate}"
            )
        for name, rate in self.slice_failures.items():
            if not 0.0 <= rate <= 1.0:
                raise ValueError(
                    f"slice_failures[{name!r}] must be in [0.0, 1.0], got {rate}"
                )
        for name, count in self.slice_counts.items():
            if count < 0:
                raise ValueError(f"slice_counts[{name!r}] must be >= 0, got {count}")

    def worst_slices(self, n: int = 3) -> list[tuple[str, float]]:
        """Return the ``n`` slices with the highest failure rate (non-empty only).

        Slices with a count of zero are excluded — an empty bucket has no
        meaningful failure rate. Ties are broken by slice name for determinism.
        """
        ranked = [
            (name, rate)
            for name, rate in self.slice_failures.items()
            if self.slice_counts.get(name, 0) > 0
        ]
        ranked.sort(key=lambda item: (-item[1], item[0]))
        return ranked[:n]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain ``dict`` (JSON-friendly)."""
        return {
            "slice_failures": dict(self.slice_failures),
            "slice_counts": dict(self.slice_counts),
            "overall_failure_rate": self.overall_failure_rate,
        }


class SliceAnalyzer:
    """Bucket predictions by slice and compute per-slice failure rates.

    A prediction "fails" when its metric score is strictly below
    ``failure_threshold``. Each record is scored against its reference (pulled
    from ``record["response"]`` or ``record["reference"]``) and assigned to a
    slice via ``slice_key`` — either a record field name or a callable mapping a
    record to its slice label.

    Example::

        from provenir.eval.metrics import ExactMatchMetric
        analyzer = SliceAnalyzer(ExactMatchMetric())
        report = analyzer.analyze(
            records=[{"category": "a", "response": "x"}],
            predictions=["y"],
            slice_key="category",
        )
        assert report.slice_failures["a"] == 1.0
    """

    def __init__(self, metric: MetricFn, failure_threshold: float = 0.5) -> None:
        if not 0.0 <= failure_threshold <= 1.0:
            raise ValueError(
                f"failure_threshold must be in [0.0, 1.0], got {failure_threshold}"
            )
        self.metric = metric
        self.failure_threshold = failure_threshold

    def analyze(
        self,
        records: list[dict[str, Any]],
        predictions: list[str],
        slice_key: str | Callable[[dict[str, Any]], str] = "category",
    ) -> SliceReport:
        """Score every record/prediction pair and aggregate failures by slice.

        Parameters
        ----------
        records:
            The eval records; each supplies a reference (``"response"`` or
            ``"reference"``) and the slice label (when ``slice_key`` is a str).
        predictions:
            Model predictions, aligned one-to-one with ``records``.
        slice_key:
            A record field name whose value labels the slice, or a callable
            ``record -> slice_label``. Missing/blank field values fall back to
            the label ``"unknown"``.

        Raises
        ------
        ValueError:
            When ``len(records) != len(predictions)``.
        """
        if len(records) != len(predictions):
            raise ValueError(
                f"records and predictions must be the same length, "
                f"got {len(records)} and {len(predictions)}"
            )

        failures: dict[str, int] = {}
        counts: dict[str, int] = {}
        total_failures = 0

        for record, prediction in zip(records, predictions):
            slice_name = self._slice_name(record, slice_key)
            reference = self._reference(record)
            score = self.metric.score(prediction, reference)
            failed = score < self.failure_threshold

            counts[slice_name] = counts.get(slice_name, 0) + 1
            failures[slice_name] = failures.get(slice_name, 0) + (1 if failed else 0)
            total_failures += 1 if failed else 0

        slice_failures = {
            name: failures[name] / counts[name] for name in counts if counts[name] > 0
        }
        overall = total_failures / len(records) if records else 0.0

        return SliceReport(
            slice_failures=slice_failures,
            slice_counts=counts,
            overall_failure_rate=overall,
        )

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _slice_name(
        record: dict[str, Any],
        slice_key: str | Callable[[dict[str, Any]], str],
    ) -> str:
        if callable(slice_key):
            return str(slice_key(record))
        value = record.get(slice_key)
        if value is None or str(value) == "":
            return "unknown"
        return str(value)

    @staticmethod
    def _reference(record: dict[str, Any]) -> str:
        for key in _REFERENCE_KEYS:
            value = record.get(key)
            if value is not None:
                return str(value)
        return ""
