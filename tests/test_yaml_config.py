from pathlib import Path

from provenir.core.config import load_run_config


def test_yaml_config_loads_and_validates(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "name: yaml-run\nbackend: stub\nseed: 3\ndeterministic: true\noutput_dir: artifacts\n",
        encoding="utf-8",
    )

    config = load_run_config(config_path)

    assert config.name == "yaml-run"
    assert config.backend == "stub"
    assert config.seed == 3
