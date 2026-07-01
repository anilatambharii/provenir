from __future__ import annotations

import pytest

from provenir.environments import (
    VERIFIER_REGISTRY,
    CodeVerifier,
    CompositeVerifier,
    ContainsVerifier,
    Environment,
    EnvironmentRegistry,
    ExactAnswerVerifier,
    JSONSchemaVerifier,
    MathVerifier,
    Observation,
    PythonSandbox,
    RegexFormatVerifier,
    SandboxConfig,
    StepResult,
    ToolCallVerifier,
    VerificationResult,
    Verifier,
    VerifierRegistry,
    VerifierReward,
)

# Small timeout to keep sandbox tests fast and deterministic.
FAST = SandboxConfig(timeout_seconds=2.0)


# ---------------------------------------------------------------------------
# VerificationResult
# ---------------------------------------------------------------------------


def test_verification_result_valid() -> None:
    r = VerificationResult(passed=True, reward=1.0, detail="ok")
    assert r.metadata == {}


def test_verification_result_rejects_out_of_range() -> None:
    with pytest.raises(ValueError):
        VerificationResult(passed=True, reward=1.5, detail="bad")
    with pytest.raises(ValueError):
        VerificationResult(passed=False, reward=-0.1, detail="bad")


# ---------------------------------------------------------------------------
# ExactAnswerVerifier
# ---------------------------------------------------------------------------


def test_exact_answer_boxed() -> None:
    v = ExactAnswerVerifier()
    assert v.verify("The answer is \\boxed{42}", "42").passed


def test_exact_answer_hash_delimiter() -> None:
    v = ExactAnswerVerifier()
    assert v.verify("reasoning...\n#### 7", "7").passed


def test_exact_answer_case_insensitive() -> None:
    v = ExactAnswerVerifier()
    assert v.verify("Paris", "paris").passed


def test_exact_answer_fail() -> None:
    v = ExactAnswerVerifier()
    r = v.verify("41", "42")
    assert not r.passed
    assert r.reward == 0.0


# ---------------------------------------------------------------------------
# MathVerifier
# ---------------------------------------------------------------------------


def test_math_fraction() -> None:
    assert MathVerifier().verify("x = 0.5", "1/2").passed


def test_math_boxed_number() -> None:
    assert MathVerifier().verify("\\boxed{3.14159}", "3.14159").passed


def test_math_last_number_used() -> None:
    assert MathVerifier().verify("first 1 then 42", "42").passed


def test_math_tolerance() -> None:
    v = MathVerifier(tol=1e-3)
    assert v.verify("1.0009", "1.0").passed
    assert not v.verify("1.5", "1.0").passed


def test_math_unparseable() -> None:
    r = MathVerifier().verify("no numbers here", "42")
    assert not r.passed
    assert r.reward == 0.0


def test_math_negative_tol_rejected() -> None:
    with pytest.raises(ValueError):
        MathVerifier(tol=-1.0)


# ---------------------------------------------------------------------------
# RegexFormatVerifier
# ---------------------------------------------------------------------------


def test_regex_full_match_pass() -> None:
    assert RegexFormatVerifier(r"\d{3}").verify("123", None).passed


def test_regex_full_match_fail() -> None:
    assert not RegexFormatVerifier(r"\d{3}").verify("123abc", None).passed


def test_regex_search_mode() -> None:
    v = RegexFormatVerifier(r"\d{3}", full_match=False)
    assert v.verify("abc123def", None).passed


def test_regex_invalid_pattern() -> None:
    with pytest.raises(ValueError):
        RegexFormatVerifier(r"(unclosed")


# ---------------------------------------------------------------------------
# JSONSchemaVerifier
# ---------------------------------------------------------------------------


def test_json_all_keys() -> None:
    v = JSONSchemaVerifier(["a", "b"])
    assert v.verify('{"a": 1, "b": 2}', None).passed


def test_json_partial_reward() -> None:
    v = JSONSchemaVerifier(["a", "b"])
    assert v.verify('{"a": 1}', None).reward == 0.5


def test_json_type_checking() -> None:
    v = JSONSchemaVerifier(["a"], types={"a": str})
    assert not v.verify('{"a": 1}', None).passed
    assert v.verify('{"a": "x"}', None).passed


def test_json_invalid() -> None:
    v = JSONSchemaVerifier(["a"])
    assert not v.verify("not json", None).passed


def test_json_empty_required_rejected() -> None:
    with pytest.raises(ValueError):
        JSONSchemaVerifier([])


# ---------------------------------------------------------------------------
# ToolCallVerifier
# ---------------------------------------------------------------------------


def test_tool_call_valid() -> None:
    v = ToolCallVerifier(["search"])
    assert v.verify('{"tool": "search", "args": {"q": "x"}}', None).passed


def test_tool_call_disallowed_tool() -> None:
    v = ToolCallVerifier(["search"])
    assert not v.verify('{"tool": "delete", "args": {}}', None).passed


def test_tool_call_args_not_dict() -> None:
    v = ToolCallVerifier(["search"])
    assert not v.verify('{"tool": "search", "args": []}', None).passed


def test_tool_call_empty_allowed_rejected() -> None:
    with pytest.raises(ValueError):
        ToolCallVerifier([])


# ---------------------------------------------------------------------------
# ContainsVerifier
# ---------------------------------------------------------------------------


def test_contains_fraction() -> None:
    v = ContainsVerifier(["foo", "bar"])
    assert v.verify("foo baz", None).reward == 0.5


def test_contains_all() -> None:
    v = ContainsVerifier(["foo", "bar"])
    assert v.verify("foo bar", None).passed


def test_contains_forbidden_zeroes() -> None:
    v = ContainsVerifier(["foo"], forbidden=["secret"])
    r = v.verify("foo secret", None)
    assert r.reward == 0.0
    assert not r.passed


def test_contains_empty_required_rejected() -> None:
    with pytest.raises(ValueError):
        ContainsVerifier([])


# ---------------------------------------------------------------------------
# CompositeVerifier
# ---------------------------------------------------------------------------


def test_composite_weighting() -> None:
    # exact fails (0.0), math passes (1.0); weights 1:3 -> 0.75
    cv = CompositeVerifier(
        [(ExactAnswerVerifier(), 1.0), (MathVerifier(), 3.0)], threshold=0.5
    )
    r = cv.verify("3.0", "3")
    assert r.reward == pytest.approx(0.75)
    assert r.passed


def test_composite_below_threshold() -> None:
    cv = CompositeVerifier([(ContainsVerifier(["z"]), 1.0)], threshold=0.5)
    assert not cv.verify("nothing", None).passed


def test_composite_empty_rejected() -> None:
    with pytest.raises(ValueError):
        CompositeVerifier([])


def test_composite_zero_weight_rejected() -> None:
    with pytest.raises(ValueError):
        CompositeVerifier([(MathVerifier(), 0.0)])


def test_composite_bad_threshold_rejected() -> None:
    with pytest.raises(ValueError):
        CompositeVerifier([(MathVerifier(), 1.0)], threshold=2.0)


# ---------------------------------------------------------------------------
# VerifierReward adapter
# ---------------------------------------------------------------------------


def test_verifier_reward_pass() -> None:
    reward = VerifierReward(ExactAnswerVerifier())
    traj = {"prediction": "42", "reference": "42"}
    assert reward.score(traj) == 1.0
    assert reward.gameability_check(traj) == []


def test_verifier_reward_fail_gameability() -> None:
    reward = VerifierReward(ExactAnswerVerifier())
    traj = {"prediction": "41", "reference": "42"}
    assert reward.score(traj) == 0.0
    assert reward.gameability_check(traj) == ["verifier_failed"]


def test_verifier_reward_custom_keys() -> None:
    reward = VerifierReward(MathVerifier(), reference_key="gold", response_key="out")
    assert reward.score({"out": "0.5", "gold": "1/2"}) == 1.0


# ---------------------------------------------------------------------------
# Sandbox
# ---------------------------------------------------------------------------


def test_sandbox_success() -> None:
    r = PythonSandbox(FAST).run("print(2 + 2)")
    assert r.success
    assert r.stdout.strip() == "4"
    assert not r.timed_out


def test_sandbox_stdin() -> None:
    r = PythonSandbox(FAST).run("import sys; print(sys.stdin.read().strip())", stdin="hi")
    assert r.stdout.strip() == "hi"


def test_sandbox_syntax_error() -> None:
    r = PythonSandbox(FAST).run("def broken(:")
    assert not r.success
    assert r.returncode != 0
    assert "SyntaxError" in r.stderr


def test_sandbox_timeout() -> None:
    r = PythonSandbox(SandboxConfig(timeout_seconds=1.0)).run("while True: pass")
    assert r.timed_out
    assert not r.success


def test_sandbox_output_truncation() -> None:
    cfg = SandboxConfig(timeout_seconds=2.0, max_output_chars=10)
    r = PythonSandbox(cfg).run("print('x' * 1000)")
    assert "truncated" in r.stdout
    assert len(r.stdout) < 100


def test_sandbox_config_validation() -> None:
    with pytest.raises(ValueError):
        SandboxConfig(timeout_seconds=0)
    with pytest.raises(ValueError):
        SandboxConfig(max_output_chars=0)


# ---------------------------------------------------------------------------
# CodeVerifier
# ---------------------------------------------------------------------------


def test_code_verifier_passing() -> None:
    v = CodeVerifier(PythonSandbox(FAST))
    r = v.verify("def add(a, b):\n    return a + b", "assert add(1, 2) == 3")
    assert r.passed
    assert r.reward == 1.0


def test_code_verifier_failing() -> None:
    v = CodeVerifier(PythonSandbox(FAST))
    r = v.verify("def add(a, b):\n    return a - b", "assert add(1, 2) == 3")
    assert not r.passed
    assert r.reward == 0.0


def test_code_verifier_dict_reference() -> None:
    v = CodeVerifier(PythonSandbox(FAST))
    r = v.verify("def f():\n    return 1", {"test_code": "assert f() == 1"})
    assert r.passed


def test_code_verifier_multiple_tests_fraction() -> None:
    v = CodeVerifier(PythonSandbox(FAST))
    r = v.verify("def f(x):\n    return x", ["assert f(1) == 1", "assert f(2) == 3"])
    assert r.reward == 0.5
    assert not r.passed


def test_code_verifier_hacking_sys_exit() -> None:
    v = CodeVerifier(PythonSandbox(FAST))
    r = v.verify("import sys\nsys.exit(0)\n", "assert False")
    assert not r.passed
    assert r.metadata["suspected_hacking"] is True


def test_code_verifier_hacking_unittest_skip() -> None:
    v = CodeVerifier(PythonSandbox(FAST))
    r = v.verify("@unittest.skip('x')\ndef f(): pass", "assert True")
    assert r.metadata["suspected_hacking"] is True


# ---------------------------------------------------------------------------
# Registries
# ---------------------------------------------------------------------------


def test_verifier_registry_prepopulated() -> None:
    assert "exact_answer" in VERIFIER_REGISTRY.list_names()
    assert "math" in VERIFIER_REGISTRY.list_names()
    assert VERIFIER_REGISTRY.get("math").verify("0.5", "1/2").passed


def test_verifier_registry_unknown() -> None:
    reg = VerifierRegistry()
    with pytest.raises(KeyError):
        reg.get("nope")


def test_verifier_registry_duplicate() -> None:
    reg = VerifierRegistry()
    reg.register("x", MathVerifier())
    with pytest.raises(ValueError):
        reg.register("x", MathVerifier())


def test_verifier_registry_empty_name() -> None:
    reg = VerifierRegistry()
    with pytest.raises(ValueError):
        reg.register("", MathVerifier())


class _EchoEnv:
    name = "echo"

    def __init__(self, prefix: str = "") -> None:
        self.prefix = prefix

    def reset(self) -> Observation:
        return Observation(text="start")

    def step(self, action: str) -> StepResult:
        return StepResult(
            observation=Observation(text=self.prefix + action),
            reward=1.0,
            done=True,
            info={},
        )


def test_environment_registry_roundtrip() -> None:
    reg = EnvironmentRegistry()
    reg.register("echo", lambda **kw: _EchoEnv(**kw))
    env = reg.create("echo", prefix=">>")
    assert reg.list_names() == ["echo"]
    obs = env.reset()
    assert obs.text == "start"
    step = env.step("hi")
    assert step.observation.text == ">>hi"
    assert step.done


def test_environment_registry_unknown() -> None:
    reg = EnvironmentRegistry()
    with pytest.raises(KeyError):
        reg.create("missing")


def test_environment_registry_duplicate() -> None:
    reg = EnvironmentRegistry()
    reg.register("echo", lambda: _EchoEnv())
    with pytest.raises(ValueError):
        reg.register("echo", lambda: _EchoEnv())


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_protocol_conformance() -> None:
    assert isinstance(ExactAnswerVerifier(), Verifier)
    assert isinstance(_EchoEnv(), Environment)
