from provenir.core.config import RunConfig


def test_run_config_defaults() -> None:
    cfg = RunConfig()
    assert cfg.backend == "stub"
    assert cfg.seed == 0
    assert cfg.deterministic is True
