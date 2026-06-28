from provenir.plugins.registry import PluginRegistry
from provenir.train.backends.registry import BackendRegistry
from provenir.train.backends.stub import StubBackend


class DummyBackend:
    name = "dummy"

    def prepare(self, config):
        return None

    def fit(self, config, manifest):
        return manifest

    def save_adapter(self, output_dir, config):
        return None

    def capabilities(self):
        return {"supports_sft": True}


def test_backend_registry_registers_and_creates_backends() -> None:
    registry = BackendRegistry()
    registry.register("stub", StubBackend)
    registry.register("dummy", DummyBackend)

    backend = registry.create("dummy")
    assert backend.name == "dummy"


def test_plugin_registry_registers_plugins() -> None:
    registry = PluginRegistry()
    registry.register("demo", {"kind": "plugin"})

    assert registry.get("demo")["kind"] == "plugin"
