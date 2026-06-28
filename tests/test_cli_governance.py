from provenir.governance.audit import AuditLogger
from provenir.governance.model_cards import ModelCardGenerator


def test_cli_supporting_governance_helpers(tmp_path) -> None:
    logger = AuditLogger(log_dir=tmp_path)
    logger.log(event="eval_completed", actor="bob")

    generator = ModelCardGenerator(output_dir=tmp_path)
    card_path = generator.generate("demo", "desc")

    assert card_path.exists()
    assert (tmp_path / "audit.jsonl").exists()
