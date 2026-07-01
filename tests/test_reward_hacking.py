from __future__ import annotations

import pytest

from provenir.environments.base import VerificationResult
from provenir.observability.reward_hacking import (
    HACKING_KINDS,
    HackingReport,
    HackingSignal,
    RewardHackingConfig,
    RewardHackingDetector,
)


def _det() -> RewardHackingDetector:
    return RewardHackingDetector()


# --------------------------------------------------------------------------
# dataclass validation / serialization
# --------------------------------------------------------------------------


def test_signal_rejects_bad_severity() -> None:
    with pytest.raises(ValueError):
        HackingSignal(kind="length_inflation", severity="nope", detail="x")


def test_signal_defaults_index() -> None:
    s = HackingSignal(kind="format_exploit", severity="warn", detail="x")
    assert s.trajectory_index == -1


def test_signal_to_dict() -> None:
    s = HackingSignal(kind="test_tampering", severity="critical", detail="d", trajectory_index=2)
    assert s.to_dict() == {
        "kind": "test_tampering",
        "severity": "critical",
        "detail": "d",
        "trajectory_index": 2,
    }


def test_all_hacking_kinds_documented() -> None:
    assert set(HACKING_KINDS) == {
        "length_inflation",
        "format_exploit",
        "verifier_gaming",
        "test_tampering",
        "proxy_divergence",
        "degenerate_repetition",
        "advantage_collapse",
    }


# --------------------------------------------------------------------------
# config validation
# --------------------------------------------------------------------------


def test_config_defaults() -> None:
    cfg = RewardHackingConfig()
    assert cfg.max_length == 2000 and cfg.repetition_ngram == 4


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_length": 0},
        {"repetition_ngram": 0},
        {"repetition_ratio_max": 0.0},
        {"repetition_ratio_max": 1.5},
        {"proxy_true_gap": 0.0},
    ],
)
def test_config_rejects_bad_values(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        RewardHackingConfig(**kwargs)


# --------------------------------------------------------------------------
# clean trajectory
# --------------------------------------------------------------------------


def test_clean_trajectory_no_signals() -> None:
    traj = {"response": "The answer is 42 because of basic arithmetic."}
    assert _det().detect(traj) == []


def test_empty_trajectory_no_signals() -> None:
    assert _det().detect({}) == []


# --------------------------------------------------------------------------
# length_inflation
# --------------------------------------------------------------------------


def test_length_inflation_fires() -> None:
    out = _det().detect({"response": "A " * 1500})
    assert "length_inflation" in [s.kind for s in out]


def test_length_inflation_reads_prediction_key() -> None:
    out = _det().detect({"prediction": "x" * 2500})
    assert "length_inflation" in [s.kind for s in out]


def test_length_inflation_below_limit_silent() -> None:
    out = _det().detect({"response": "x" * 100})
    assert "length_inflation" not in [s.kind for s in out]


# --------------------------------------------------------------------------
# format_exploit
# --------------------------------------------------------------------------


def test_format_exploit_fires_on_short_caps() -> None:
    out = _det().detect({"response": "YES"})
    assert "format_exploit" in [s.kind for s in out]


def test_format_exploit_silent_on_long_caps() -> None:
    out = _det().detect({"response": "THIS IS A LONGER ALL CAPS SENTENCE"})
    assert "format_exploit" not in [s.kind for s in out]


def test_format_exploit_silent_on_lowercase() -> None:
    out = _det().detect({"response": "yes"})
    assert "format_exploit" not in [s.kind for s in out]


# --------------------------------------------------------------------------
# test_tampering
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "snippet",
    [
        "import unittest\n@unittest.skip('x')\ndef test(): ...",
        "@pytest.skip",
        "import sys; sys.exit(0)",
        "os._exit(0)",
        "@skip\ndef test(): ...",
        "def test(monkeypatch): ...",
        "with open(__file__, 'w') as f: f.write('')",
        "assert True  # always passes",
    ],
)
def test_test_tampering_fires(snippet: str) -> None:
    out = _det().detect({"response": snippet})
    tamper = [s for s in out if s.kind == "test_tampering"]
    assert tamper and tamper[0].severity == "critical"


def test_test_tampering_silent_on_clean_code() -> None:
    out = _det().detect({"response": "def solve(x):\n    return x + 1"})
    assert "test_tampering" not in [s.kind for s in out]


# --------------------------------------------------------------------------
# verifier_gaming
# --------------------------------------------------------------------------


def test_verifier_gaming_from_verification_result() -> None:
    vr = VerificationResult(
        passed=True, reward=1.0, detail="passed", metadata={"suspected_hacking": True}
    )
    out = _det().detect({"response": "ok", "verification": vr})
    gaming = [s for s in out if s.kind == "verifier_gaming"]
    assert gaming and gaming[0].severity == "critical"


def test_verifier_gaming_from_dict_metadata() -> None:
    vr = {"passed": True, "metadata": {"suspected_hacking": True}}
    out = _det().detect({"response": "ok", "verification": vr})
    assert "verifier_gaming" in [s.kind for s in out]


def test_verifier_clean_no_gaming() -> None:
    vr = VerificationResult(passed=True, reward=1.0, detail="passed", metadata={})
    out = _det().detect({"response": "ok", "verification": vr})
    assert "verifier_gaming" not in [s.kind for s in out]


def test_verifier_missing_no_gaming() -> None:
    out = _det().detect({"response": "ok"})
    assert "verifier_gaming" not in [s.kind for s in out]


# --------------------------------------------------------------------------
# proxy_divergence
# --------------------------------------------------------------------------


def test_proxy_divergence_fires() -> None:
    out = _det().detect({"response": "ok", "proxy_reward": 0.95, "true_reward": 0.1})
    div = [s for s in out if s.kind == "proxy_divergence"]
    assert div and div[0].severity == "critical"


def test_proxy_divergence_reads_held_out_reward() -> None:
    out = _det().detect({"response": "ok", "proxy_reward": 0.9, "held_out_reward": 0.2})
    assert "proxy_divergence" in [s.kind for s in out]


def test_proxy_divergence_silent_when_aligned() -> None:
    out = _det().detect({"response": "ok", "proxy_reward": 0.9, "true_reward": 0.85})
    assert "proxy_divergence" not in [s.kind for s in out]


def test_proxy_divergence_needs_both_rewards() -> None:
    out = _det().detect({"response": "ok", "proxy_reward": 0.9})
    assert "proxy_divergence" not in [s.kind for s in out]


# --------------------------------------------------------------------------
# degenerate_repetition
# --------------------------------------------------------------------------


def test_degenerate_repetition_fires() -> None:
    out = _det().detect({"response": "go go go go " * 20})
    assert "degenerate_repetition" in [s.kind for s in out]


def test_repetition_silent_on_varied_text() -> None:
    text = " ".join(f"word{i}" for i in range(200))
    out = _det().detect({"response": text})
    assert "degenerate_repetition" not in [s.kind for s in out]


def test_repetition_silent_on_short_text() -> None:
    out = _det().detect({"response": "a b"})
    assert "degenerate_repetition" not in [s.kind for s in out]


# --------------------------------------------------------------------------
# detect_group -> advantage_collapse
# --------------------------------------------------------------------------


def test_group_advantage_collapse_fires() -> None:
    out = _det().detect_group([0.5, 0.5, 0.5, 0.5])
    assert out and out[0].kind == "advantage_collapse"
    assert out[0].severity == "critical"


def test_group_varied_rewards_no_collapse() -> None:
    assert _det().detect_group([0.1, 0.9, 0.5, 0.3]) == []


def test_group_single_element_no_collapse() -> None:
    assert _det().detect_group([0.5]) == []


def test_group_empty_no_collapse() -> None:
    assert _det().detect_group([]) == []


# --------------------------------------------------------------------------
# detect_batch / HackingReport
# --------------------------------------------------------------------------


def test_batch_report_flags_and_tags_index() -> None:
    trajs = [
        {"response": "clean answer that is fine"},
        {"response": "YES"},
        {"response": "x" * 3000},
    ]
    report = _det().detect_batch(trajs)
    assert report.num_trajectories == 3
    assert not report.is_clean
    flagged = {s.trajectory_index for s in report.signals}
    assert flagged == {1, 2}


def test_batch_clean_report() -> None:
    trajs = [{"response": "a normal helpful answer"} for _ in range(4)]
    report = _det().detect_batch(trajs)
    assert report.is_clean
    assert report.hacking_rate == 0.0


def test_hacking_rate_math() -> None:
    signals = [
        HackingSignal(kind="length_inflation", severity="warn", detail="x", trajectory_index=0),
        HackingSignal(kind="format_exploit", severity="warn", detail="x", trajectory_index=0),
        HackingSignal(kind="test_tampering", severity="critical", detail="x", trajectory_index=3),
    ]
    report = HackingReport(signals=signals, num_trajectories=4)
    # 2 unique flagged trajectories out of 4.
    assert report.hacking_rate == 0.5


def test_hacking_rate_zero_trajectories() -> None:
    assert HackingReport(signals=[], num_trajectories=0).hacking_rate == 0.0


def test_report_by_kind() -> None:
    signals = [
        HackingSignal(kind="length_inflation", severity="warn", detail="x", trajectory_index=0),
        HackingSignal(kind="length_inflation", severity="warn", detail="x", trajectory_index=1),
        HackingSignal(kind="format_exploit", severity="warn", detail="x", trajectory_index=2),
    ]
    report = HackingReport(signals=signals, num_trajectories=3)
    assert report.by_kind() == {"length_inflation": 2, "format_exploit": 1}


def test_report_rejects_negative_num() -> None:
    with pytest.raises(ValueError):
        HackingReport(signals=[], num_trajectories=-1)


def test_report_to_dict() -> None:
    signals = [
        HackingSignal(kind="format_exploit", severity="warn", detail="x", trajectory_index=1),
    ]
    report = HackingReport(signals=signals, num_trajectories=2)
    d = report.to_dict()
    assert d["num_trajectories"] == 2
    assert d["hacking_rate"] == 0.5
    assert d["is_clean"] is False
    assert d["by_kind"] == {"format_exploit": 1}
    assert d["signals"][0]["kind"] == "format_exploit"


# --------------------------------------------------------------------------
# custom config
# --------------------------------------------------------------------------


def test_custom_max_length_applies() -> None:
    det = RewardHackingDetector(RewardHackingConfig(max_length=10))
    out = det.detect({"response": "this is longer than ten characters"})
    assert "length_inflation" in [s.kind for s in out]


def test_custom_proxy_gap_applies() -> None:
    det = RewardHackingDetector(RewardHackingConfig(proxy_true_gap=0.05))
    out = det.detect({"response": "ok", "proxy_reward": 0.5, "true_reward": 0.4})
    assert "proxy_divergence" in [s.kind for s in out]


def test_multiple_signals_on_one_trajectory() -> None:
    traj = {
        "response": "YES",
        "proxy_reward": 0.9,
        "true_reward": 0.1,
    }
    kinds = {s.kind for s in _det().detect(traj)}
    assert {"format_exploit", "proxy_divergence"} <= kinds
