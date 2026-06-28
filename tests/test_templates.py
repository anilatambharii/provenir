from __future__ import annotations

import pytest

from provenir.data.dataset import JsonlDataset
from provenir.data.templates import (
    TEMPLATE_REGISTRY,
    AlpacaTemplate,
    ChatMLTemplate,
    Llama3Template,
    MistralTemplate,
    Phi3Template,
    PromptTemplate,
    RawCompletionTemplate,
    TemplateRegistry,
)

_RECORD = {"prompt": "What is 2+2?", "response": "4"}
_RECORD_WITH_INPUT = {
    "instruction": "Translate to French",
    "input": "Hello",
    "output": "Bonjour",
}


class TestAlpacaTemplate:
    t = AlpacaTemplate()

    def test_name(self) -> None:
        assert self.t.name == "alpaca"

    def test_format_basic(self) -> None:
        out = self.t.format(_RECORD)
        assert "### Instruction:" in out
        assert "### Response:" in out
        assert "What is 2+2?" in out
        assert "4" in out

    def test_format_with_input(self) -> None:
        out = self.t.format(_RECORD_WITH_INPUT)
        assert "### Input:" in out
        assert "Hello" in out

    def test_format_without_input_no_input_section(self) -> None:
        out = self.t.format(_RECORD)
        assert "### Input:" not in out


class TestChatMLTemplate:
    t = ChatMLTemplate()

    def test_name(self) -> None:
        assert self.t.name == "chatml"

    def test_format_contains_tokens(self) -> None:
        out = self.t.format(_RECORD)
        assert "<|im_start|>user" in out
        assert "<|im_start|>assistant" in out
        assert "<|im_end|>" in out

    def test_format_includes_prompt(self) -> None:
        out = self.t.format(_RECORD)
        assert "What is 2+2?" in out

    def test_format_includes_response(self) -> None:
        out = self.t.format(_RECORD)
        assert out.count("4") >= 1


class TestLlama3Template:
    t = Llama3Template()

    def test_name(self) -> None:
        assert self.t.name == "llama3"

    def test_format_contains_begin_of_text(self) -> None:
        out = self.t.format(_RECORD)
        assert "<|begin_of_text|>" in out

    def test_format_contains_header_ids(self) -> None:
        out = self.t.format(_RECORD)
        assert "<|start_header_id|>user<|end_header_id|>" in out
        assert "<|start_header_id|>assistant<|end_header_id|>" in out

    def test_format_contains_eot(self) -> None:
        assert "<|eot_id|>" in self.t.format(_RECORD)


class TestMistralTemplate:
    t = MistralTemplate()

    def test_name(self) -> None:
        assert self.t.name == "mistral"

    def test_format_inst_tags(self) -> None:
        out = self.t.format(_RECORD)
        assert "[INST]" in out
        assert "[/INST]" in out

    def test_format_ends_with_eos(self) -> None:
        assert self.t.format(_RECORD).endswith("</s>")


class TestPhi3Template:
    t = Phi3Template()

    def test_name(self) -> None:
        assert self.t.name == "phi3"

    def test_format_user_token(self) -> None:
        out = self.t.format(_RECORD)
        assert "<|user|>" in out
        assert "<|assistant|>" in out
        assert "<|end|>" in out


class TestRawCompletionTemplate:
    t = RawCompletionTemplate()

    def test_name(self) -> None:
        assert self.t.name == "raw"

    def test_format_concatenates(self) -> None:
        out = self.t.format({"prompt": "Q: ", "response": "A"})
        assert out == "Q: A"


class TestTemplateRegistry:
    def test_list_includes_builtin(self) -> None:
        reg = TemplateRegistry()
        names = reg.list_names()
        for name in ("alpaca", "chatml", "llama3", "mistral", "phi3", "raw"):
            assert name in names

    def test_get_returns_template(self) -> None:
        reg = TemplateRegistry()
        tmpl = reg.get("alpaca")
        assert tmpl.name == "alpaca"

    def test_get_unknown_raises(self) -> None:
        with pytest.raises(KeyError, match="unknown template"):
            TemplateRegistry().get("nonexistent")

    def test_register_custom(self) -> None:
        reg = TemplateRegistry()

        class MyTmpl:
            name: str = "custom"

            def format(self, record: dict) -> str:  # type: ignore[type-arg]
                return f"CUSTOM:{record}"

        reg.register(MyTmpl())
        assert "custom" in reg.list_names()

    def test_format_dataset(self) -> None:
        reg = TemplateRegistry()
        ds = JsonlDataset.from_records([_RECORD, _RECORD])
        results = reg.format_dataset("alpaca", ds)
        assert len(results) == 2
        assert all("### Instruction:" in r for r in results)

    def test_module_level_registry_singleton(self) -> None:
        assert isinstance(TEMPLATE_REGISTRY, TemplateRegistry)
        assert "alpaca" in TEMPLATE_REGISTRY.list_names()

    def test_prompt_template_protocol(self) -> None:
        t = AlpacaTemplate()
        assert isinstance(t, PromptTemplate)
