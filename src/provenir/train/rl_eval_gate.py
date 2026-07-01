"""Eval-in-the-loop RL gate: one guard for the RL / post-training loop.

Managed platforms (OpenAI RFT, Fireworks Eval Protocol) close the eval->train
loop but no OSS tool does it framework-agnostically *with contamination
safety*. This module fuses three concerns that are usually handled (or
forgotten) separately into a single decision object:

* **Contamination safety** — is my held-out eval set actually held out? A
  leaked eval row silently inflates every downstream metric, so the gate can
  scan the training set against its own eval set up front
  (:meth:`RLEvalGate.guard_eval_set`) and refuse to trust a compromised eval.
* **Regression** — has the primary metric dropped materially below a baseline
  (via :class:`provenir.eval.harness.RegressionGate`)?
* **Reward hacking** — did the surrounding loop surface a critical hacking
  signal this iteration?

Each iteration yields a single :class:`GateDecision` carrying ``should_halt`` /
``should_warn`` and a human-readable list of ``reasons``.

The hacking-signal handling is duck-typed on purpose: it accepts either an
object exposing a ``.severity`` attribute *or* a mapping with a ``"severity"``
key, so it composes with the observability module's ``HackingSignal`` without a
hard import (and therefore without a cross-module dependency cycle).

Example
-------
>>> from provenir.data.dataset import JsonlDataset
>>> eval_ds = JsonlDataset.from_records([
...     {"prompt": "q1", "response": "a1"},
...     {"prompt": "q2", "response": "a2"},
... ])
>>> gate = RLEvalGate(RLGateConfig(eval_every=1), eval_ds)
>>> decision = gate.on_iteration(0, ["a1", "a2"])
>>> decision.should_halt
False
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from provenir.data.dataset import JsonlDataset
from provenir.eval.contamination import (
    ContaminationChecker,
    ContaminationConfig,
    ContaminationReport,
)
from provenir.eval.harness import MultiMetricEvaluator, RegressionGate
from provenir.eval.metrics import MetricFn


@dataclass(frozen=True)
class GateDecision:
    """The gate's verdict for a single RL iteration.

    Attributes
    ----------
    should_halt:
        The loop should stop — a hard guard fired (regression, floor,
        contamination, or a critical hacking signal).
    should_warn:
        A soft signal worth surfacing but not fatal on its own.
    reasons:
        Human-readable explanations for every guard that fired.
    step:
        The iteration index this decision belongs to.
    """

    should_halt: bool
    should_warn: bool
    reasons: list[str]
    step: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "should_halt": self.should_halt,
            "should_warn": self.should_warn,
            "reasons": list(self.reasons),
            "step": self.step,
        }


@dataclass(frozen=True)
class RLGateConfig:
    """Configuration for :class:`RLEvalGate`.

    Parameters
    ----------
    eval_every:
        Run the eval + guards only when ``step % eval_every == 0``.
    primary_metric:
        Metric name used for the regression baseline and the score floor.
    regression_baseline:
        When set, halt if the primary metric falls more than
        ``regression_threshold`` below this value.
    regression_threshold:
        Allowed downward slack against ``regression_baseline``.
    max_contamination_rate:
        Halt when a prior :meth:`RLEvalGate.guard_eval_set` scan found the eval
        set contaminated *above* this fraction. Defaults to ``0.0`` — any leak
        is disqualifying.
    halt_on_hacking:
        Halt when a ``"critical"`` hacking signal is present this iteration.
    min_primary_score:
        Absolute floor for the primary metric; halt if it drops below this.
    """

    eval_every: int = 1
    primary_metric: str = "exact_match"
    regression_baseline: float | None = None
    regression_threshold: float = 0.05
    max_contamination_rate: float = 0.0
    halt_on_hacking: bool = True
    min_primary_score: float | None = None

    def __post_init__(self) -> None:
        if self.eval_every < 1:
            raise ValueError(f"eval_every must be >= 1, got {self.eval_every}")
        if not self.primary_metric:
            raise ValueError("primary_metric must be a non-empty string")
        if self.regression_baseline is not None and not 0.0 <= self.regression_baseline <= 1.0:
            raise ValueError(
                f"regression_baseline must be in [0.0, 1.0], got {self.regression_baseline}"
            )
        if self.regression_threshold < 0.0:
            raise ValueError(
                f"regression_threshold must be >= 0, got {self.regression_threshold}"
            )
        if not 0.0 <= self.max_contamination_rate <= 1.0:
            raise ValueError(
                f"max_contamination_rate must be in [0.0, 1.0], got {self.max_contamination_rate}"
            )
        if self.min_primary_score is not None and not 0.0 <= self.min_primary_score <= 1.0:
            raise ValueError(
                f"min_primary_score must be in [0.0, 1.0], got {self.min_primary_score}"
            )


def _signal_severity(signal: Any) -> str | None:
    """Extract a severity string from a hacking signal, duck-typed.

    Accepts an object with a ``.severity`` attribute or a mapping with a
    ``"severity"`` key so the gate composes with any observability
    ``HackingSignal`` without importing it. Returns ``None`` when neither shape
    applies.
    """
    if isinstance(signal, dict):
        value = signal.get("severity")
    else:
        value = getattr(signal, "severity", None)
    return str(value) if value is not None else None


class RLEvalGate:
    """Single composing guard for the RL / post-training loop.

    Fuses contamination safety, regression detection, a score floor, and
    reward-hacking checks into one :class:`GateDecision` per iteration. Mirrors
    the config + history + decision shape of
    :class:`provenir.train.eval_callback.EvalCallback` but *composes* the eval,
    contamination, and regression primitives rather than re-implementing them.

    Usage inside an RL loop::

        gate = RLEvalGate(RLGateConfig(eval_every=1), eval_dataset)
        gate.guard_eval_set(train_dataset)  # trust check, once, up front
        for step, preds in enumerate(rollouts()):
            decision = gate.on_iteration(step, preds, hacking_signals=signals)
            if decision.should_halt:
                break
    """

    def __init__(
        self,
        config: RLGateConfig,
        eval_dataset: JsonlDataset,
        metrics: list[MetricFn] | None = None,
    ) -> None:
        self.config = config
        self.eval_dataset = eval_dataset
        self.evaluator = MultiMetricEvaluator(metrics=metrics)
        self._history: list[dict[str, Any]] = []
        self._eval_report: ContaminationReport | None = None
        self._eval_compromised = False

    # -- trust check --------------------------------------------------------

    def guard_eval_set(self, train_dataset: JsonlDataset) -> ContaminationReport:
        """Scan ``train_dataset`` against this gate's eval set.

        This is the "is my held-out eval actually held-out?" check that makes
        the eval trustworthy: if a training row leaked into the eval set, every
        downstream metric is inflated. The resulting
        :class:`ContaminationReport` is stored and, when its
        ``contamination_rate`` exceeds
        :attr:`RLGateConfig.max_contamination_rate`, every subsequent
        :meth:`on_iteration` decision halts with a contamination reason.
        """
        checker = ContaminationChecker(ContaminationConfig())
        report = checker.check_datasets(train_dataset, self.eval_dataset)
        self._eval_report = report
        self._eval_compromised = report.contamination_rate > self.config.max_contamination_rate
        return report

    # -- per-iteration guard ------------------------------------------------

    def on_iteration(
        self,
        step: int,
        predictions: list[str],
        hacking_signals: list[Any] | None = None,
    ) -> GateDecision:
        """Evaluate and guard iteration ``step``.

        Runs the eval + guards only on ``step % eval_every == 0``; off-eval
        steps return a no-op (non-halting) decision. On eval steps it composes
        the regression gate, the score floor, the stored contamination verdict,
        and the reward-hacking check into a single :class:`GateDecision`.
        """
        if step % self.config.eval_every != 0:
            return GateDecision(
                should_halt=False, should_warn=False, reasons=[], step=step
            )

        reasons: list[str] = []
        should_halt = False

        result = self.evaluator.evaluate(self.eval_dataset, predictions)
        primary = self._primary_score(result)

        # Regression vs baseline.
        if self.config.regression_baseline is not None:
            gate = RegressionGate(
                baseline=self.config.regression_baseline,
                threshold=self.config.regression_threshold,
            )
            if not gate.check(result):
                should_halt = True
                reasons.append(
                    f"regression: {self.config.primary_metric}={primary:.4f} fell more than "
                    f"{self.config.regression_threshold:.4f} below baseline "
                    f"{self.config.regression_baseline:.4f}"
                )

        # Absolute score floor.
        if self.config.min_primary_score is not None and primary < self.config.min_primary_score:
            should_halt = True
            reasons.append(
                f"score floor: {self.config.primary_metric}={primary:.4f} below "
                f"min {self.config.min_primary_score:.4f}"
            )

        # Compromised eval set (from a prior guard_eval_set scan).
        if self._eval_compromised:
            rate = self._eval_report.contamination_rate if self._eval_report else 0.0
            should_halt = True
            reasons.append(
                f"contamination: eval set compromised at rate {rate:.4f} > "
                f"max {self.config.max_contamination_rate:.4f}"
            )

        # Reward hacking.
        if self.config.halt_on_hacking and hacking_signals:
            critical = [
                s for s in hacking_signals if _signal_severity(s) == "critical"
            ]
            if critical:
                should_halt = True
                reasons.append(
                    f"reward hacking: {len(critical)} critical signal(s) detected"
                )

        decision = GateDecision(
            should_halt=should_halt,
            should_warn=bool(reasons) and not should_halt,
            reasons=reasons,
            step=step,
        )
        self._history.append(
            {
                "step": step,
                "primary_score": primary,
                "result": result.to_dict(),
                "decision": decision.to_dict(),
            }
        )
        return decision

    # -- introspection ------------------------------------------------------

    def best_step(self) -> int | None:
        """Step with the highest primary-metric score, or ``None`` if no eval ran."""
        if not self._history:
            return None
        best = max(self._history, key=lambda entry: float(entry["primary_score"]))
        return int(best["step"])

    @property
    def history(self) -> list[dict[str, Any]]:
        return list(self._history)

    def summary(self) -> dict[str, Any]:
        """Aggregate view of the gate's run for logging or reporting."""
        halts = [entry["step"] for entry in self._history if entry["decision"]["should_halt"]]
        warns = [entry["step"] for entry in self._history if entry["decision"]["should_warn"]]
        return {
            "evaluations": len(self._history),
            "best_step": self.best_step(),
            "halt_steps": halts,
            "warn_steps": warns,
            "eval_compromised": self._eval_compromised,
            "contamination_rate": (
                self._eval_report.contamination_rate if self._eval_report else None
            ),
        }

    # -- helpers ------------------------------------------------------------

    def _primary_score(self, result: Any) -> float:
        if not result.metrics:
            return 0.0
        metric = result.metrics.get(self.config.primary_metric)
        if metric is None:
            metric = next(iter(result.metrics.values()))
        return float(metric.mean)
