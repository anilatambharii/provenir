from __future__ import annotations

import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Any

from provenir.environments.base import VerificationResult

# ---------------------------------------------------------------------------
# Sandbox result + config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SandboxResult:
    """Outcome of executing code inside a sandbox."""

    success: bool
    stdout: str
    stderr: str
    timed_out: bool
    returncode: int


@dataclass(frozen=True)
class SandboxConfig:
    """Resource limits for a sandbox execution."""

    timeout_seconds: float = 10.0
    max_output_chars: int = 10000

    def __post_init__(self) -> None:
        if self.timeout_seconds <= 0:
            raise ValueError(f"timeout_seconds must be positive, got {self.timeout_seconds}")
        if self.max_output_chars <= 0:
            raise ValueError(f"max_output_chars must be positive, got {self.max_output_chars}")


# ---------------------------------------------------------------------------
# Python sandbox
# ---------------------------------------------------------------------------


class PythonSandbox:
    """Run untrusted Python via a subprocess with a wall-clock timeout.

    This is a *portable* sandbox: it relies solely on ``subprocess`` timeouts so
    it works identically on Windows, macOS, and Linux (it deliberately avoids the
    POSIX-only ``resource`` module and ``signal.alarm``).

    .. warning::
        This provides **process isolation only** — it does not restrict
        filesystem, network, or syscall access. For production RLVR use, swap in
        a container / gVisor / firejail backend by subclassing and overriding
        :meth:`run`.

    Example::

        result = PythonSandbox().run("print(2 + 2)")
        assert result.stdout.strip() == "4"
    """

    def __init__(self, config: SandboxConfig | None = None) -> None:
        self.config = config or SandboxConfig()

    def _truncate(self, text: str) -> str:
        limit = self.config.max_output_chars
        if len(text) <= limit:
            return text
        return text[:limit] + "\n...[truncated]"

    def run(self, code: str, stdin: str = "") -> SandboxResult:
        """Execute *code* in a fresh temp directory, returning a SandboxResult."""
        with tempfile.TemporaryDirectory() as workdir:
            try:
                proc = subprocess.run(
                    [sys.executable, "-c", code],
                    input=stdin,
                    capture_output=True,
                    text=True,
                    cwd=workdir,
                    timeout=self.config.timeout_seconds,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                stdout = exc.stdout or ""
                stderr = exc.stderr or ""
                if isinstance(stdout, bytes):
                    stdout = stdout.decode("utf-8", "replace")
                if isinstance(stderr, bytes):
                    stderr = stderr.decode("utf-8", "replace")
                return SandboxResult(
                    success=False,
                    stdout=self._truncate(stdout),
                    stderr=self._truncate(stderr or "TimeoutExpired"),
                    timed_out=True,
                    returncode=-1,
                )
            return SandboxResult(
                success=proc.returncode == 0,
                stdout=self._truncate(proc.stdout),
                stderr=self._truncate(proc.stderr),
                timed_out=False,
                returncode=proc.returncode,
            )


# ---------------------------------------------------------------------------
# Code verifier (executes candidate + tests)
# ---------------------------------------------------------------------------

_HACK_PATTERNS = (
    "unittest.skip",
    "sys.exit(0)",
    "sys.exit()",
    "os._exit",
    "pytest.skip",
    "@skip",
    "builtins.__dict__['assert']",
    "monkeypatch",
    "open(__file__",
    "__loader__",
)


def _normalize_tests(reference: Any) -> list[str]:
    """Coerce a reference into a list of test-code strings."""
    if isinstance(reference, str):
        return [reference]
    if isinstance(reference, dict):
        test_code = reference.get("test_code", "")
        if isinstance(test_code, list):
            return [str(t) for t in test_code]
        return [str(test_code)]
    if isinstance(reference, (list, tuple)):
        return [str(t) for t in reference]
    return [str(reference)]


class CodeVerifier:
    """Verify a candidate code solution by running it against test code.

    ``reference`` may be a test-code string, a ``{"test_code": ...}`` dict, or a
    list of test-code strings (reward becomes the fraction passing). Obvious
    reward-hacking patterns are flagged and fail the verification outright::

        v = CodeVerifier()
        v.verify("def add(a, b): return a + b", "assert add(1, 2) == 3").passed  # True
    """

    def __init__(self, sandbox: PythonSandbox | None = None, name: str = "code") -> None:
        self.sandbox = sandbox or PythonSandbox()
        self.name = name

    def _suspected_hacking(self, response: str) -> list[str]:
        lowered = response.lower()
        return [p for p in _HACK_PATTERNS if p.lower() in lowered]

    def verify(self, response: str, reference: Any) -> VerificationResult:
        hacks = self._suspected_hacking(response)
        if hacks:
            return VerificationResult(
                passed=False,
                reward=0.0,
                detail=f"suspected reward hacking: {hacks}",
                metadata={"suspected_hacking": True, "patterns": hacks},
            )

        tests = _normalize_tests(reference)
        if not tests:
            return VerificationResult(False, 0.0, "no test code provided")

        passed_count = 0
        details: list[str] = []
        for i, test_code in enumerate(tests):
            program = f"{response}\n\n{test_code}\n"
            result = self.sandbox.run(program)
            ok = result.success and result.returncode == 0
            passed_count += int(ok)
            if not ok:
                snippet = (result.stderr or result.stdout).strip().splitlines()
                tail = snippet[-1] if snippet else "no output"
                details.append(f"test {i}: {tail}")

        reward = passed_count / len(tests)
        passed = passed_count == len(tests)
        return VerificationResult(
            passed=passed,
            reward=reward,
            detail=(
                "all tests passed"
                if passed
                else f"{passed_count}/{len(tests)} passed; " + "; ".join(details)
            ),
            metadata={
                "suspected_hacking": False,
                "passed": passed_count,
                "total": len(tests),
            },
        )
