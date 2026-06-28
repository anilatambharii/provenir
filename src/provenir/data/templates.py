from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from provenir.data.dataset import JsonlDataset


@runtime_checkable
class PromptTemplate(Protocol):
    """Protocol for a chat-format template that converts a record to a training string."""

    @property
    def name(self) -> str: ...

    def format(self, record: dict[str, Any]) -> str: ...


class AlpacaTemplate:
    """Stanford Alpaca prompt format: instruction (+ optional input) → response."""

    name: str = "alpaca"

    def format(self, record: dict[str, Any]) -> str:
        instruction = str(record.get("instruction", record.get("prompt", "")))
        inp = str(record.get("input", ""))
        output = str(record.get("output", record.get("response", "")))
        if inp:
            return (
                f"### Instruction:\n{instruction}\n\n"
                f"### Input:\n{inp}\n\n"
                f"### Response:\n{output}"
            )
        return f"### Instruction:\n{instruction}\n\n### Response:\n{output}"


class ChatMLTemplate:
    """ChatML format used by Mistral / Qwen / OpenHermes and compatible fine-tuning stacks."""

    name: str = "chatml"

    def format(self, record: dict[str, Any]) -> str:
        prompt = str(record.get("prompt", record.get("instruction", "")))
        response = str(record.get("response", record.get("output", "")))
        return (
            f"<|im_start|>user\n{prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n{response}<|im_end|>"
        )


class Llama3Template:
    """Meta Llama-3 instruct format."""

    name: str = "llama3"

    def format(self, record: dict[str, Any]) -> str:
        prompt = str(record.get("prompt", record.get("instruction", "")))
        response = str(record.get("response", record.get("output", "")))
        return (
            "<|begin_of_text|>"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            f"{prompt}<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
            f"{response}<|eot_id|>"
        )


class MistralTemplate:
    """Mistral / Mixtral instruction format."""

    name: str = "mistral"

    def format(self, record: dict[str, Any]) -> str:
        prompt = str(record.get("prompt", record.get("instruction", "")))
        response = str(record.get("response", record.get("output", "")))
        return f"[INST] {prompt} [/INST] {response}</s>"


class Phi3Template:
    """Microsoft Phi-3 chat format."""

    name: str = "phi3"

    def format(self, record: dict[str, Any]) -> str:
        prompt = str(record.get("prompt", record.get("instruction", "")))
        response = str(record.get("response", record.get("output", "")))
        return f"<|user|>\n{prompt}<|end|>\n<|assistant|>\n{response}<|end|>"


class RawCompletionTemplate:
    """Raw completion — prompt concatenated directly with response (no special tokens)."""

    name: str = "raw"

    def format(self, record: dict[str, Any]) -> str:
        prompt = str(record.get("prompt", ""))
        response = str(record.get("response", record.get("output", "")))
        return f"{prompt}{response}"


class TemplateRegistry:
    """Registry of named prompt templates with dataset-level formatting support.

    Built-in templates: ``alpaca``, ``chatml``, ``llama3``, ``mistral``,
    ``phi3``, ``raw``.  Register custom templates with :meth:`register`.
    """

    _BUILTIN: dict[str, PromptTemplate] = {
        "alpaca": AlpacaTemplate(),
        "chatml": ChatMLTemplate(),
        "llama3": Llama3Template(),
        "mistral": MistralTemplate(),
        "phi3": Phi3Template(),
        "raw": RawCompletionTemplate(),
    }

    def __init__(self) -> None:
        self._templates: dict[str, PromptTemplate] = dict(self._BUILTIN)

    def register(self, template: PromptTemplate) -> None:
        """Add or replace a template under its own name."""
        self._templates[template.name] = template

    def get(self, name: str) -> PromptTemplate:
        if name not in self._templates:
            raise KeyError(
                f"unknown template {name!r}; available: {sorted(self._templates)}"
            )
        return self._templates[name]

    def list_names(self) -> list[str]:
        """Alphabetically sorted list of available template names."""
        return sorted(self._templates)

    def format(self, name: str, record: dict[str, Any]) -> str:
        """Format a single record using the named template."""
        return self.get(name).format(record)

    def format_dataset(self, name: str, dataset: JsonlDataset) -> list[str]:
        """Format every record in *dataset* using the named template."""
        tmpl = self.get(name)
        return [tmpl.format(record) for record in dataset.records]


TEMPLATE_REGISTRY: TemplateRegistry = TemplateRegistry()
