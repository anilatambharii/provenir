from __future__ import annotations

from typing import Any


class PluginRegistry:
    """A simple registry for extension plugins."""

    def __init__(self) -> None:
        self._plugins: dict[str, Any] = {}

    def register(self, name: str, plugin: Any) -> None:
        self._plugins[name] = plugin

    def get(self, name: str) -> Any:
        if name not in self._plugins:
            raise KeyError(f"plugin {name!r} is not registered")
        return self._plugins[name]
