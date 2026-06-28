from provenir.data.dataset import JsonlDataset
from provenir.data.quality import DataQualityReport, DecontaminationChecker, QualityScorer
from provenir.eval.harness import EvaluationResult, MetricSummary
from provenir.orchestrate.cost import CostBreakdown, CostEstimator


def test_data_quality_and_decontamination_detect_issues() -> None:
    dataset = JsonlDataset.from_records([
        {"prompt": "What is the capital of France?", "response": "Paris"},
        {"prompt": "What is the capital of France?", "response": "Paris"},
        {"prompt": "What is the capital of Germany?", "response": "Berlin"},
    ])

    scorer = QualityScorer()
    report = scorer.score(dataset)
    assert isinstance(report, DataQualityReport)
    assert 0.0 <= report.score <= 1.0
    assert "duplicate_records" in report.issues

    checker = DecontaminationChecker()
    findings = checker.check(dataset, references=["Paris", "Berlin"])
    assert findings


def test_evaluation_result_supports_multiple_metrics_and_cost_estimation() -> None:
    result = EvaluationResult(
        metrics={
            "accuracy": MetricSummary(mean=0.8, confidence_interval=(0.7, 0.9), samples=10),
            "f1": MetricSummary(mean=0.75, confidence_interval=(0.65, 0.85), samples=10),
        }
    )

    assert result.to_dict()["metrics"]["accuracy"]["mean"] == 0.8

    estimator = CostEstimator(unit_cost=0.5, multiplier=1.1)
    breakdown = estimator.estimate_run_cost(steps=20, hours=2.0)
    assert isinstance(breakdown, CostBreakdown)
    assert breakdown.total_cost > 0.0
