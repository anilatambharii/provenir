"""Backend-agnostic adapters: wrap verl / TRL / Unsloth behind one contract.

Provenir orchestrates the RL *winners* rather than reimplementing kernels. This
module exposes each supported backend through a uniform capability probe and a
:class:`BackendSelector` that picks the right engine for the requested scale and
algorithm — falling back to the CPU ``stub`` backend when nothing is installed,
so the reward / observability / gating loop still runs in CI.

Scale tiers drive selection:

* **multi_node** -> verl (the only multi-node RL engine here).
* **multi_gpu** -> verl if available, else TRL.
* **single_gpu** -> Unsloth if available, else TRL.

Example
-------
>>> selector = BackendSelector()
>>> sel = selector.select(num_gpus=8, num_nodes=4, algorithm="grpo")
>>> sel.backend in {"verl", "stub"}
True
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Capability probes (conditional imports)
# ---------------------------------------------------------------------------

try:  # pragma: no cover - import guard exercised by availability, not tests
    import verl  # type: ignore[import-not-found]  # noqa: F401

    _HAS_VERL = True
except ImportError:
    _HAS_VERL = False

try:  # pragma: no cover - import guard
    import trl  # noqa: F401

    _HAS_TRL = True
except ImportError:
    _HAS_TRL = False

try:  # pragma: no cover - import guard
    import unsloth  # type: ignore[import-not-found]  # noqa: F401

    _HAS_UNSLOTH = True
except ImportError:
    _HAS_UNSLOTH = False


# ---------------------------------------------------------------------------
# Capabilities
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BackendCapabilities:
    """What a backend supports and whether it is installed.

    Example::

        caps = BackendCapabilities(name="verl", available=False,
                                   algorithms=("grpo",), max_scale="multi_node",
                                   supports_vllm=True)
        assert "grpo" in caps.algorithms
    """

    name: str
    available: bool
    algorithms: tuple[str, ...]
    max_scale: str
    supports_vllm: bool

    def __post_init__(self) -> None:
        valid = {"single_gpu", "multi_gpu", "multi_node"}
        if self.max_scale not in valid:
            raise ValueError(f"max_scale must be one of {sorted(valid)}, got {self.max_scale!r}")

    def supports(self, algorithm: str) -> bool:
        """True when *algorithm* (case-insensitive) is in this backend's list."""
        return algorithm.casefold() in {a.casefold() for a in self.algorithms}


# ---------------------------------------------------------------------------
# Backend adapters
# ---------------------------------------------------------------------------


class VerlBackend:
    """Adapter for `verl <https://github.com/volcengine/verl>`_ — HybridFlow RL.

    verl is the heavy-duty, multi-node RL engine: GRPO / DAPO / GSPO / PPO / RLOO
    with vLLM-accelerated rollouts. Reported unavailable when ``verl`` is not
    importable so selection falls back gracefully.
    """

    name: str = "verl"

    def is_available(self) -> bool:
        return _HAS_VERL

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            name=self.name,
            available=_HAS_VERL,
            algorithms=("grpo", "dapo", "gspo", "ppo", "rloo"),
            max_scale="multi_node",
            supports_vllm=True,
        )


class TRLAdapter:
    """Adapter for `TRL <https://github.com/huggingface/trl>`_ — HF post-training.

    TRL covers the SFT / preference / GRPO space at single- to multi-GPU scale,
    with optional vLLM-backed generation. Reported unavailable when ``trl`` is
    not importable.
    """

    name: str = "trl"

    def is_available(self) -> bool:
        return _HAS_TRL

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            name=self.name,
            available=_HAS_TRL,
            algorithms=("sft", "dpo", "grpo", "kto", "orpo"),
            max_scale="multi_gpu",
            supports_vllm=True,
        )


class UnslothBackend:
    """Adapter for `Unsloth <https://github.com/unslothai/unsloth>`_ — single-GPU.

    Unsloth is the memory-efficient single-GPU choice (SFT / DPO / GRPO) without
    vLLM rollout acceleration. Reported unavailable when ``unsloth`` is not
    importable.
    """

    name: str = "unsloth"

    def is_available(self) -> bool:
        return _HAS_UNSLOTH

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            name=self.name,
            available=_HAS_UNSLOTH,
            algorithms=("sft", "dpo", "grpo"),
            max_scale="single_gpu",
            supports_vllm=False,
        )


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BackendSelection:
    """The selector's verdict: which backend, why, and whether it is installed.

    Example::

        sel = BackendSelection(backend="stub", rationale="nothing installed",
                               available=False)
        assert sel.backend == "stub"
    """

    backend: str
    rationale: str
    available: bool


class BackendSelector:
    """Pick a backend by scale tier and requested algorithm.

    The selection rules mirror how these engines are actually deployed:

    * ``num_nodes > 1`` -> **multi_node** -> verl.
    * ``num_gpus > 1``  -> **multi_gpu**  -> verl, else TRL.
    * otherwise         -> **single_gpu** -> Unsloth, else TRL.

    A ``prefer`` hint wins whenever that backend is available *and* supports the
    algorithm. If no candidate is available the selector falls back to the CPU
    ``stub`` backend so the surrounding loop still runs.
    """

    def __init__(self) -> None:
        self._backends: list[VerlBackend | TRLAdapter | UnslothBackend] = [
            VerlBackend(),
            TRLAdapter(),
            UnslothBackend(),
        ]

    def available_backends(self) -> list[BackendCapabilities]:
        """Capabilities for every registered backend (available or not)."""
        return [b.capabilities() for b in self._backends]

    def select(
        self,
        num_gpus: int = 1,
        num_nodes: int = 1,
        algorithm: str = "grpo",
        prefer: str | None = None,
    ) -> BackendSelection:
        """Choose a backend for the given scale and algorithm."""
        if num_gpus < 1:
            raise ValueError(f"num_gpus must be >= 1, got {num_gpus}")
        if num_nodes < 1:
            raise ValueError(f"num_nodes must be >= 1, got {num_nodes}")

        caps = {c.name: c for c in self.available_backends()}

        # Honor an explicit preference when it can serve the request.
        if prefer is not None:
            pref = caps.get(prefer)
            if pref is not None and pref.available and pref.supports(algorithm):
                return BackendSelection(
                    backend=pref.name,
                    rationale=f"preferred backend {prefer!r} available and supports {algorithm!r}",
                    available=True,
                )

        tier: str
        order: tuple[str, ...]
        if num_nodes > 1:
            tier, order = "multi_node", ("verl",)
        elif num_gpus > 1:
            tier, order = "multi_gpu", ("verl", "trl")
        else:
            tier, order = "single_gpu", ("unsloth", "trl")

        for name in order:
            cap = caps.get(name)
            if cap is not None and cap.available and cap.supports(algorithm):
                return BackendSelection(
                    backend=name,
                    rationale=(
                        f"{tier} scale (gpus={num_gpus}, nodes={num_nodes}); "
                        f"{name!r} supports {algorithm!r}"
                    ),
                    available=True,
                )

        return BackendSelection(
            backend="stub",
            rationale=(
                f"no installed backend serves {algorithm!r} at {tier} scale "
                f"(gpus={num_gpus}, nodes={num_nodes}); falling back to CPU stub"
            ),
            available=False,
        )
