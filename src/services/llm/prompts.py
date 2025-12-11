from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, List


class PromptRepository:
    """Loads and renders structured prompt definitions from disk."""

    def __init__(self, source_path: Path) -> None:
        self._source_path = source_path
        self._prompts: Dict[str, Dict[str, Any]] = {}
        self.reload()

    def reload(self) -> None:
        if not self._source_path.exists():
            self._prompts = {}
            return

        raw_payload = self._source_path.read_text(encoding="utf-8")
        if not raw_payload.strip():
            self._prompts = {}
            return

        try:
            records = json.loads(raw_payload)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Prompts file '{self._source_path}' is not valid JSON: {exc}") from exc

        prompts: Dict[str, Dict[str, Any]] = {}
        if isinstance(records, list):
            for record in records:
                if isinstance(record, dict):
                    name = record.get("name")
                    if name:
                        prompts[str(name)] = record
        self._prompts = prompts

    def list_prompts(self) -> List[str]:
        return sorted(self._prompts.keys())

    def get_prompt(self, name: str) -> Dict[str, Any]:
        if name not in self._prompts:
            raise KeyError(f"Prompt '{name}' is not defined.")
        return copy.deepcopy(self._prompts[name])

    def render_prompt(self, name: str) -> str:
        prompt_entry = self.get_prompt(name)
        return self._render_prompt(prompt_entry)

    @staticmethod
    def _render_prompt(prompt_entry: Dict[str, Any]) -> str:
        sections: List[str] = []

        task = prompt_entry.get("task")
        if task:
            sections.append(f"Task:\n{str(task).strip()}")

        instructions = prompt_entry.get("prompt")
        if instructions:
            sections.append(f"Instructions:\n{str(instructions).strip()}")

        expected_output = prompt_entry.get("expected_output")
        if isinstance(expected_output, dict) and expected_output:
            expected_sections: List[str] = []
            description = expected_output.get("description")
            if description:
                expected_sections.append(f"Description:\n{str(description).strip()}")
            schema = expected_output.get("schema")
            if schema is not None:
                expected_sections.append("Schema:\n" + json.dumps(schema, indent=2))
            example = expected_output.get("example")
            if example is not None:
                expected_sections.append("Example:\n" + json.dumps(example, indent=2))
            if expected_sections:
                sections.append("Expected Output (JSON):\n" + "\n".join(expected_sections))

        examples = prompt_entry.get("examples")
        if isinstance(examples, list) and examples:
            rendered_examples: List[str] = []
            for index, example in enumerate(examples, start=1):
                if not isinstance(example, dict):
                    continue
                input_payload = example.get("input", {})
                output_payload = example.get("output", {})
                input_json = json.dumps(input_payload, indent=2)
                output_json = json.dumps(output_payload, indent=2)
                rendered_examples.append(
                    f"Example {index} Input:\n{input_json}\nExample {index} Output:\n{output_json}"
                )
            if rendered_examples:
                sections.append("Examples:\n" + "\n\n".join(rendered_examples))

        return "\n\n".join(part for part in (section.strip() for section in sections) if part)

