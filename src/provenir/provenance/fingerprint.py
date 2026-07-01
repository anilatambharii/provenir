from __future__ import annotations

import hashlib
import platform
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class EnvironmentFingerprint:
    """Deterministic snapshot of the environment a run executed in.

    A fingerprint captures just enough of the runtime to reason about
    bitwise reproducibility: the Python version, the OS/platform string,
    a content hash over installed packages, and best-effort CUDA/hardware
    identifiers.

    Example:
        >>> fp = capture_fingerprint({"numpy": "1.26.4", "torch": "2.3.0"})
        >>> fp.packages_hash == capture_fingerprint(
        ...     {"torch": "2.3.0", "numpy": "1.26.4"}
        ... ).packages_hash
        True
    """

    python_version: str
    platform: str
    packages_hash: str
    cuda_version: str
    hardware: str

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-serializable mapping of the fingerprint fields."""
        return {
            "python_version": self.python_version,
            "platform": self.platform,
            "packages_hash": self.packages_hash,
            "cuda_version": self.cuda_version,
            "hardware": self.hardware,
        }


def _hash_packages(packages: dict[str, str]) -> str:
    """Return a stable sha256 over a sorted ``name==version`` package list."""
    lines = [f"{name}=={version}" for name, version in sorted(packages.items())]
    payload = "\n".join(lines).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _discover_packages() -> dict[str, str]:
    """Best-effort discovery of installed packages via importlib.metadata.

    Returns an empty mapping if metadata is unavailable so callers stay
    deterministic across environments where discovery fails.
    """
    try:
        from importlib import metadata
    except ImportError:  # pragma: no cover - importlib.metadata is stdlib on 3.11+
        return {}

    discovered: dict[str, str] = {}
    for dist in metadata.distributions():
        name = dist.metadata["Name"]
        if name is None:
            continue
        discovered[name] = dist.version
    return discovered


def _detect_cuda_version() -> str:
    """Best-effort CUDA runtime version; ``"unknown"`` if undetectable."""
    try:
        import torch
    except ImportError:
        return "unknown"

    version = getattr(getattr(torch, "version", None), "cuda", None)
    if isinstance(version, str) and version:
        return version
    return "unknown"


def _detect_hardware() -> str:
    """Best-effort accelerator/CPU descriptor; ``"unknown"`` if undetectable."""
    try:
        import torch
    except ImportError:
        return platform.processor() or "unknown"

    try:
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            if isinstance(name, str) and name:
                return name
    except Exception:  # pragma: no cover - defensive, torch internals vary
        pass
    return platform.processor() or "unknown"


def capture_fingerprint(
    packages: dict[str, str] | None = None,
) -> EnvironmentFingerprint:
    """Capture an :class:`EnvironmentFingerprint` for the current process.

    The function is deterministic when ``packages`` is injected: given the
    same package mapping (and the same interpreter/platform) it always
    produces the same fingerprint. No clocks or randomness are consulted.

    Args:
        packages: Optional mapping of package name to version. Inject this
            for reproducible tests; when ``None`` the installed distributions
            are discovered best-effort via ``importlib.metadata``.

    Returns:
        A frozen :class:`EnvironmentFingerprint`.

    Example:
        >>> fp = capture_fingerprint({"numpy": "1.26.4"})
        >>> fp.packages_hash == capture_fingerprint({"numpy": "1.26.4"}).packages_hash
        True
    """
    resolved = packages if packages is not None else _discover_packages()
    version_info = sys.version_info
    python_version = f"{version_info.major}.{version_info.minor}.{version_info.micro}"
    return EnvironmentFingerprint(
        python_version=python_version,
        platform=platform.platform(),
        packages_hash=_hash_packages(resolved),
        cuda_version=_detect_cuda_version(),
        hardware=_detect_hardware(),
    )


def kernel_determinism_flags() -> dict[str, str]:
    """Return recommended env-var settings for bitwise reproducibility.

    Reinforcement-learning fine-tuning is notoriously non-reproducible
    because the *training* engine and the *inference/rollout* engine can
    execute different GPU kernels for the same op. A trainer computing
    token log-probs with one cuBLAS/attention kernel and a serving stack
    such as vLLM computing them with another will diverge at the bit level;
    those tiny divergences compound through the policy-gradient update, so
    two "identical" RL runs drift apart. Pinning the kernel selection and
    hashing/launch behaviour removes the largest sources of that drift.

    The returned mapping documents each flag:

    * ``CUBLAS_WORKSPACE_CONFIG`` — force a deterministic cuBLAS workspace so
      GEMMs pick a stable algorithm (required by ``use_deterministic_algorithms``).
    * ``PYTHONHASHSEED`` — pin hash randomization so set/dict ordering and any
      hash-derived sampling are stable.
    * ``CUDA_LAUNCH_BLOCKING`` — run kernels synchronously to remove
      launch-order nondeterminism when debugging divergence (has a perf cost).
    * ``CUDA_VISIBLE_DEVICES`` — pin device ordering so device 0 is stable.
    * ``TOKENIZERS_PARALLELISM`` — disable tokenizer thread races.

    Example:
        >>> flags = kernel_determinism_flags()
        >>> flags["PYTHONHASHSEED"]
        '0'
        >>> flags["CUBLAS_WORKSPACE_CONFIG"]
        ':4096:8'
    """
    return {
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "PYTHONHASHSEED": "0",
        "CUDA_LAUNCH_BLOCKING": "1",
        "CUDA_VISIBLE_DEVICES": "0",
        "TOKENIZERS_PARALLELISM": "false",
    }


__all__ = [
    "EnvironmentFingerprint",
    "capture_fingerprint",
    "kernel_determinism_flags",
]
