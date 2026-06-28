from pathlib import Path

from provenir.data.dataset import JsonlDataset
from provenir.eval.harness import Evaluator, RegressionGate


def test_eval_harness_reports_variance_and_regression_gate(tmp_path: Path) -> None:
    dataset = JsonlDataset.from_path("tests/fixtures/sample.jsonl")
    evaluator = Evaluator()
    result = evaluator.evaluate(dataset=dataset, predictions=["Hi", "Planet"])

    assert result.accuracy.mean == 1.0
    assert result.accuracy.confidence_interval[0] <= result.accuracy.confidence_interval[1]

    gate = RegressionGate(baseline=0.9, threshold=0.2)
    assert gate.check(result) is True

    artifact_dir = tmp_path / "eval"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    result.save(artifact_dir / "report.json")
    assert (artifact_dir / "report.json").exists()
