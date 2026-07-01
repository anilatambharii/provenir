from __future__ import annotations

import pytest

from provenir.data.dataset import JsonlDataset
from provenir.eval.contamination import ContaminationChecker, ContaminationConfig
from provenir.eval.metrics import ExactMatchMetric
from provenir.loop.doctor import (
    DataRequest,
    Diagnosis,
    DoctorConfig,
    Finding,
    LoopAction,
    LoopController,
    LoopDoctor,
    LoopSignals,
)
from provenir.loop.slices import SliceAnalyzer, SliceReport
from provenir.observability import FlightRecorder, RewardHackingDetector, RLStepMetrics

_PLATEAU = [0.40, 0.41, 0.40, 0.41, 0.40]
_IMPROVING = [0.30, 0.40, 0.50, 0.60, 0.70]


# --- SliceAnalyzer --------------------------------------------------------

def test_slice_analyzer_failure_rate() -> None:
    analyzer = SliceAnalyzer(ExactMatchMetric(), failure_threshold=0.5)
    records = [
        {"category": "math", "response": "4"},
        {"category": "math", "response": "6"},
        {"category": "code", "response": "ok"},
    ]
    report = analyzer.analyze(records, ["4", "5", "ok"], slice_key="category")
    assert report.slice_failures["math"] == pytest.approx(0.5)
    assert report.slice_failures["code"] == pytest.approx(0.0)
    assert report.overall_failure_rate == pytest.approx(1 / 3)


def test_slice_analyzer_callable_key() -> None:
    analyzer = SliceAnalyzer(ExactMatchMetric())
    records = [{"response": "a"}, {"response": "b"}]
    report = analyzer.analyze(records, ["x", "y"], slice_key=lambda r: "all")
    assert report.slice_failures["all"] == pytest.approx(1.0)


def test_slice_analyzer_length_mismatch() -> None:
    analyzer = SliceAnalyzer(ExactMatchMetric())
    with pytest.raises(ValueError, match="same length"):
        analyzer.analyze([{"response": "a"}], ["x", "y"])


def test_slice_analyzer_threshold_validation() -> None:
    with pytest.raises(ValueError, match="failure_threshold"):
        SliceAnalyzer(ExactMatchMetric(), failure_threshold=1.5)


def test_slice_report_worst_slices() -> None:
    report = SliceReport(
        slice_failures={"a": 0.8, "b": 0.1, "c": 0.5},
        slice_counts={"a": 10, "b": 10, "c": 10},
        overall_failure_rate=0.47,
    )
    assert report.worst_slices(2) == [("a", 0.8), ("c", 0.5)]


def test_slice_report_excludes_empty_slices() -> None:
    report = SliceReport(
        slice_failures={"a": 0.9, "empty": 1.0},
        slice_counts={"a": 5, "empty": 0},
        overall_failure_rate=0.9,
    )
    assert report.worst_slices() == [("a", 0.9)]


def test_slice_report_validation() -> None:
    with pytest.raises(ValueError, match="overall_failure_rate"):
        SliceReport(slice_failures={}, slice_counts={}, overall_failure_rate=2.0)
    with pytest.raises(ValueError, match="slice_failures"):
        SliceReport(slice_failures={"a": 2.0}, slice_counts={"a": 1}, overall_failure_rate=0.0)


# --- LoopSignals ----------------------------------------------------------

def test_loop_signals_validation() -> None:
    with pytest.raises(ValueError, match="hacking_rate"):
        LoopSignals(reward_history=[0.1], hacking_rate=1.5)
    with pytest.raises(ValueError, match="contamination_rate"):
        LoopSignals(reward_history=[0.1], contamination_rate=-0.1)


def test_loop_signals_from_reports() -> None:
    recorder = FlightRecorder()
    recorder.log_step(RLStepMetrics(step=0, entropy=2.0, reward_mean=0.4, advantage_std=0.3))
    recorder.log_step(RLStepMetrics(step=1, entropy=0.001, reward_mean=0.4, advantage_std=0.0))

    detector = RewardHackingDetector()
    hacking = detector.detect_batch([
        {"prediction": "import sys; sys.exit(0)", "proxy_reward": 1.0, "true_reward": 0.0},
    ])

    train = JsonlDataset.from_records([{"prompt": "shared text about cells"}])
    eval_ds = JsonlDataset.from_records([{"prompt": "shared text about cells"}])
    contam = ContaminationChecker(ContaminationConfig(method="exact")).check_datasets(
        train, eval_ds
    )

    slices = SliceAnalyzer(ExactMatchMetric()).analyze(
        [{"category": "x", "response": "a"}], ["b"], slice_key="category"
    )

    signals = LoopSignals.from_reports(
        reward_history=[0.4, 0.4],
        flight_recorder=recorder,
        hacking_report=hacking,
        contamination_report=contam,
        slice_report=slices,
        data_age_days=200.0,
    )
    kinds = signals.anomaly_kinds
    assert "advantage_collapse" in kinds or "entropy_collapse" in kinds
    assert signals.hacking_rate > 0.0
    assert signals.contamination_rate > 0.0
    assert signals.slice_failures["x"] == pytest.approx(1.0)
    assert signals.data_age_days == 200.0


# --- LoopDoctor: differential diagnosis -----------------------------------

def test_diagnose_healthy_when_improving() -> None:
    diagnosis = LoopDoctor().diagnose(LoopSignals(reward_history=_IMPROVING))
    assert diagnosis.is_healthy
    assert diagnosis.primary_category == "none"


def test_diagnose_eval_contamination() -> None:
    signals = LoopSignals(reward_history=_PLATEAU, contamination_rate=0.2)
    diagnosis = LoopDoctor().diagnose(signals)
    assert diagnosis.primary_category == "eval"
    assert diagnosis.findings[0].severity == "critical"


def test_diagnose_reward_hacking() -> None:
    signals = LoopSignals(
        reward_history=_PLATEAU, hacking_rate=0.4, hacking_kinds=["test_tampering"]
    )
    diagnosis = LoopDoctor().diagnose(signals)
    assert diagnosis.primary_category == "reward"


def test_diagnose_algorithm_advantage_collapse() -> None:
    signals = LoopSignals(reward_history=_PLATEAU, anomaly_kinds=["advantage_collapse"])
    diagnosis = LoopDoctor().diagnose(signals)
    assert diagnosis.primary_category == "algorithm"
    assert "dynamic sampling" in diagnosis.findings[0].recommended_action


def test_diagnose_algorithm_entropy_collapse_action() -> None:
    signals = LoopSignals(reward_history=_PLATEAU, anomaly_kinds=["entropy_collapse"])
    diagnosis = LoopDoctor().diagnose(signals)
    assert diagnosis.primary_category == "algorithm"
    assert "entropy" in diagnosis.findings[0].recommended_action


def test_diagnose_data_problem_emits_request() -> None:
    signals = LoopSignals(
        reward_history=_PLATEAU,
        slice_failures={"tool_use": 0.8, "math": 0.1},
    )
    diagnosis = LoopDoctor().diagnose(signals)
    assert diagnosis.primary_category == "data"
    assert diagnosis.data_request is not None
    assert diagnosis.data_request.slices == ["tool_use"]
    assert diagnosis.data_request.num_examples == 50
    assert diagnosis.data_request.freshness_days is None


def test_diagnose_stale_data_sets_freshness() -> None:
    signals = LoopSignals(
        reward_history=_PLATEAU,
        slice_failures={"tool_use": 0.9},
        data_age_days=400.0,
    )
    diagnosis = LoopDoctor().diagnose(signals)
    assert diagnosis.primary_category == "data"
    assert diagnosis.data_request is not None
    assert diagnosis.data_request.freshness_days == 180
    assert "recent" in diagnosis.findings[0].recommended_action


def test_diagnose_data_request_broadens_when_no_bad_slices() -> None:
    signals = LoopSignals(reward_history=_PLATEAU)
    diagnosis = LoopDoctor().diagnose(signals)
    assert diagnosis.primary_category == "data"
    assert diagnosis.data_request is not None
    assert diagnosis.data_request.slices == ["overall"]


def test_diagnose_eval_wins_over_data() -> None:
    # Both a plateau (data) and contamination (eval) present -> eval ranks first.
    signals = LoopSignals(
        reward_history=_PLATEAU,
        contamination_rate=0.3,
        slice_failures={"x": 0.9},
    )
    diagnosis = LoopDoctor().diagnose(signals)
    assert diagnosis.primary_category == "eval"
    # A strong non-data fault suppresses the data finding / request.
    assert diagnosis.data_request is None


def test_diagnose_no_data_finding_when_improving_despite_bad_slice() -> None:
    signals = LoopSignals(reward_history=_IMPROVING, slice_failures={"x": 0.9})
    diagnosis = LoopDoctor().diagnose(signals)
    assert diagnosis.is_healthy


def test_diagnose_declining_reward_is_data_problem() -> None:
    signals = LoopSignals(reward_history=[0.7, 0.6, 0.5, 0.4, 0.3])
    diagnosis = LoopDoctor().diagnose(signals)
    assert diagnosis.primary_category == "data"
    assert "declined" in diagnosis.findings[0].evidence


# --- Diagnosis rendering --------------------------------------------------

def test_diagnosis_to_dict_and_markdown() -> None:
    signals = LoopSignals(reward_history=_PLATEAU, slice_failures={"tool_use": 0.8})
    diagnosis = LoopDoctor().diagnose(signals)
    d = diagnosis.to_dict()
    assert d["primary_category"] == "data"
    assert d["data_request"]["slices"] == ["tool_use"]
    md = diagnosis.to_markdown()
    assert "Data request" in md
    assert "tool_use" in md


def test_healthy_diagnosis_markdown() -> None:
    diagnosis = LoopDoctor().diagnose(LoopSignals(reward_history=_IMPROVING))
    assert "Healthy" in diagnosis.to_markdown()


# --- validation of result types -------------------------------------------

def test_finding_validation() -> None:
    ok = {"evidence": "e", "recommended_action": "a"}
    with pytest.raises(ValueError, match="category"):
        Finding(category="bogus", severity="warn", confidence=0.5, **ok)
    with pytest.raises(ValueError, match="severity"):
        Finding(category="data", severity="bogus", confidence=0.5, **ok)
    with pytest.raises(ValueError, match="confidence"):
        Finding(category="data", severity="warn", confidence=2.0, **ok)


def test_data_request_validation() -> None:
    with pytest.raises(ValueError, match="num_examples"):
        DataRequest(slices=["a"], num_examples=0, freshness_days=None, rationale="")
    with pytest.raises(ValueError, match="freshness_days"):
        DataRequest(slices=["a"], num_examples=5, freshness_days=0, rationale="")


def test_doctor_config_validation() -> None:
    with pytest.raises(ValueError, match="plateau_window"):
        DoctorConfig(plateau_window=1)
    with pytest.raises(ValueError, match="stale_data_days"):
        DoctorConfig(stale_data_days=0.0)
    with pytest.raises(ValueError, match="min_examples_per_slice"):
        DoctorConfig(min_examples_per_slice=0)


# --- LoopController -------------------------------------------------------

def test_controller_continue_when_healthy() -> None:
    action = LoopController().decide(Diagnosis(findings=[]))
    assert action.action == "continue"


def test_controller_maps_each_category() -> None:
    controller = LoopController()
    cases = {
        "eval": "clean_eval",
        "reward": "fix_reward",
        "algorithm": "stabilize",
        "data": "collect_data",
    }
    for category, expected in cases.items():
        finding = Finding(
            category=category, severity="warn", confidence=0.9,
            evidence="e", recommended_action="a",
        )
        action = controller.decide(Diagnosis(findings=[finding]))
        assert action.action == expected


def test_controller_carries_data_request() -> None:
    signals = LoopSignals(reward_history=_PLATEAU, slice_failures={"x": 0.9})
    diagnosis = LoopDoctor().diagnose(signals)
    action = LoopController().decide(diagnosis)
    assert action.action == "collect_data"
    assert action.data_request is not None


def test_controller_priority_eval_over_data() -> None:
    findings = [
        Finding(
            category="data", severity="warn", confidence=0.95,
            evidence="e", recommended_action="a",
        ),
        Finding(
            category="eval", severity="critical", confidence=0.9,
            evidence="e", recommended_action="a",
        ),
    ]
    action = LoopController().decide(Diagnosis(findings=findings))
    assert action.action == "clean_eval"


def test_loop_action_validation() -> None:
    with pytest.raises(ValueError, match="action"):
        LoopAction(action="bogus", reason="")


def test_loop_action_to_dict() -> None:
    action = LoopAction(action="continue", reason="ok")
    assert action.to_dict()["action"] == "continue"
