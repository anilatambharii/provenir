from pathlib import Path

from provenir.governance.model_cards import ModelCardGenerator


def test_model_card_generator_writes_markdown(tmp_path: Path) -> None:
    generator = ModelCardGenerator(output_dir=tmp_path)
    card_path = generator.generate("demo-model", "stub backend")
    assert card_path.exists()
    assert "# demo-model" in card_path.read_text(encoding="utf-8")
