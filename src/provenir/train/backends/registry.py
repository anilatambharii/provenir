from __future__ import annotations

from typing import Any, Callable


class BackendRegistry:
    """Register and instantiate backend adapters by name."""

    def __init__(self) -> None:
        self._backends: dict[str, Callable[[], Any]] = {}

    def register(self, name: str, factory: Callable[[], Any]) -> None:
        self._backends[name] = factory

    def create(self, name: str) -> Any:
        if name not in self._backends:
            raise KeyError(f"backend {name!r} is not registered")
        return self._backends[name]()
