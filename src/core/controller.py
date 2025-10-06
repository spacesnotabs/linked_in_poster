from __future__ import annotations

import copy
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Optional, Union

from src.config import MODELS_CONFIG, SYSTEM_PROMPT
from src.core.llm_model import LlmModel, UsageMetrics


@dataclass
class ModelSession:
    system_prompt: str
    chat_history: list[dict[str, str]] = field(default_factory=list)
    pending_context: Optional[str] = None
    pending_context_tokens: int = 0
    last_response: Optional[str] = None
    last_usage: UsageMetrics = field(default_factory=UsageMetrics)
    chat_usage: UsageMetrics = field(default_factory=UsageMetrics)
    usage_history: list[UsageMetrics] = field(default_factory=list)


class LLMController:
    """Manage lifecycle and interaction state for one or more LLM backends."""

    def __init__(
        self,
        models_config: Optional[dict[str, dict[str, Any]]] = None,
        system_prompt: Optional[str] = None,
        prompts_path: Optional[Union[str, Path]] = None,
    ) -> None:
        source_config = models_config or MODELS_CONFIG
        self._models_config: dict[str, dict[str, Any]] = {
            name: config.copy() for name, config in source_config.items()
        }
        self._system_prompt = system_prompt or SYSTEM_PROMPT
        self._active_models: dict[str, LlmModel] = {}
        self._sessions: dict[str, ModelSession] = {}
        self._model_errors: dict[str, str] = {}
        self._active_model_id: Optional[str] = None

        controller_root = Path(__file__).resolve().parents[1]
        default_prompts_path = controller_root / "agents" / "prompts.json"
        self._prompts_path = Path(prompts_path) if prompts_path is not None else default_prompts_path
        self._prompts: dict[str, dict[str, Any]] = {}
        self._load_prompts()

    @property
    def available_models(self) -> list[str]:
        return list(self._models_config.keys())

    @property
    def loaded_models(self) -> list[str]:
        return list(self._active_models.keys())

    @property
    def active_model_id(self) -> Optional[str]:
        return self._active_model_id

    @property
    def chat(self) -> list[dict[str, str]]:
        session = self._get_active_session()
        if not session:
            return []
        return [message.copy() for message in session.chat_history]

    @property
    def last_response(self) -> Optional[str]:
        session = self._get_active_session()
        return session.last_response if session else None

    def list_prompts(self) -> list[str]:
        """Return the available prompt names."""

        return sorted(self._prompts.keys())

    def get_prompt(self, prompt_name: str) -> dict[str, Any]:
        """Retrieve a prompt definition by name."""

        if prompt_name not in self._prompts:
            raise KeyError(f"Prompt '{prompt_name}' is not defined.")
        return copy.deepcopy(self._prompts[prompt_name])

    def render_prompt(self, prompt_name: str) -> str:
        """Return the fully rendered prompt text without applying it."""

        prompt_entry = self.get_prompt(prompt_name)
        return self._render_prompt(prompt_entry)

    def reload_prompts(self) -> dict[str, dict[str, Any]]:
        """Reload prompts from disk and return a copy of the catalog."""

        self._load_prompts()
        return {name: copy.deepcopy(data) for name, data in self._prompts.items()}

    def apply_prompt(self, prompt_name: str, *, model_id: Optional[str] = None) -> str:
        """Apply a stored prompt to the specified model session."""

        prompt_entry = self.get_prompt(prompt_name)
        rendered_prompt = self._render_prompt(prompt_entry)
        self.set_system_prompt(rendered_prompt, model_id=model_id)
        return rendered_prompt

    def set_system_prompt(self, prompt: str, *, model_id: Optional[str] = None) -> None:
        """Set the system prompt for a model and reset its session state."""

        resolved_id, model, session = self._require_model(model_id)
        cleaned_prompt = prompt.strip()

        session.system_prompt = cleaned_prompt
        session.chat_history = []
        if cleaned_prompt:
            session.chat_history.append(self._create_message("system", cleaned_prompt))

        session.pending_context = None
        session.pending_context_tokens = 0
        session.last_response = None
        session.last_usage = UsageMetrics()
        session.chat_usage = UsageMetrics()
        session.usage_history = []

        if hasattr(model, "set_system_prompt"):
            model.set_system_prompt(cleaned_prompt)
        config = self._models_config.get(resolved_id)
        if config is not None:
            config["system_prompt"] = cleaned_prompt

        self._model_errors.pop(resolved_id, None)
        self._set_active_model(resolved_id)

    def get_active_system_prompt(self, model_id: Optional[str] = None) -> str:
        """Return the current system prompt for the active or specified model."""

        resolved_id = model_id or self._active_model_id
        if resolved_id is None:
            return self._system_prompt
        session = self._sessions.get(resolved_id)
        if session is None:
            return self._system_prompt
        return session.system_prompt

    def initialize_model(
        self,
        model_id: str,
        *,
        force_reload: bool = False,
        **overrides: Any,
    ) -> LlmModel:
        config = self._resolve_model_config(model_id=model_id, overrides=overrides)
        if not force_reload and model_id in self._active_models:
            self._set_active_model(model_id)
            return self._active_models[model_id]

        if force_reload:
            self._active_models.pop(model_id, None)
            self._sessions.pop(model_id, None)

        model_path = config.get("model_filename") or config.get("model_path")
        if not model_path:
            raise ValueError(f"Model configuration for '{model_id}' is missing 'model_filename'.")

        chat_format = config.get("chat_format") or "auto"
        context_window = config.get("context_window") or config.get("context_size") or 32768
        try:
            context_size = int(context_window)
        except (TypeError, ValueError):
            raise ValueError(
                f"Invalid context window value '{context_window}' for model '{model_id}'."
            ) from None

        prompt = (config.get("system_prompt") or self._system_prompt or "").strip()
        n_gpu_layers = config.get("n_gpu_layers", -1)

        try:
            model = LlmModel(
                model_path=model_path,
                model_name=model_id,
                system_prompt=prompt,
                chat_format=chat_format,
                n_gpu_layers=n_gpu_layers,
                context_size=context_size,
            )
        except Exception as exc:
            self._model_errors[model_id] = str(exc)
            raise

        session = ModelSession(system_prompt=prompt)
        if prompt:
            session.chat_history.append(self._create_message("system", prompt))

        self._model_errors.pop(model_id, None)
        self._active_models[model_id] = model
        self._sessions[model_id] = session
        self._set_active_model(model_id)
        return model

    def get_state(self) -> dict[str, Any]:
        models_state: dict[str, Any] = {}
        for model_id, model in self._active_models.items():
            session = self._sessions.get(model_id)
            if not session:
                continue
            models_state[model_id] = {
                "is_active": model_id == self._active_model_id,
                "config": model.get_model_config(),
                "pending_context_tokens": session.pending_context_tokens,
                "system_prompt": session.system_prompt,
                "usage": {
                    "last_interaction": session.last_usage.to_dict(),
                    "chat": session.chat_usage.to_dict(),
                    "history": [usage.to_dict() for usage in session.usage_history],
                },
                "last_error": self._model_errors.get(model_id),
            }

        return {
            "available_models": self.available_models,
            "loaded_models": self.loaded_models,
            "active_model": self._active_model_id,
            "chat": self.chat,
            "last_response": self.last_response,
            "models": models_state,
            "errors": self._model_errors.copy(),
            "default_system_prompt": self._system_prompt,
        }

    def send_prompt(
        self,
        prompt: str,
        *,
        model_id: Optional[str] = None,
        max_tokens: int = 1024,
    ) -> Optional[str]:
        resolved_id, model, session = self._require_model(model_id)

        context_value = session.pending_context
        context_tokens = session.pending_context_tokens if context_value else 0
        session.pending_context = None
        session.pending_context_tokens = 0

        user_message = self._create_message("user", prompt, context=context_value)
        session.chat_history.append(user_message)

        messages = [message.copy() for message in session.chat_history]

        try:
            text, raw_response, elapsed = model.generate_response(
                messages=messages,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            self._model_errors[resolved_id] = str(exc)
            raise

        assistant_message = self._create_message("assistant", text)
        session.chat_history.append(assistant_message)
        session.last_response = text

        usage = model.build_usage_metrics(
            user_prompt=user_message,
            assistant_response=text,
            context_tokens=context_tokens,
            response_seconds=elapsed,
            raw_response=raw_response,
        )
        session.last_usage = usage
        session.usage_history.append(self._clone_usage(usage))
        self._update_chat_usage(session.chat_usage, usage)
        self._model_errors.pop(resolved_id, None)

        self._set_active_model(resolved_id)
        return text

    def clear_chat_history(
        self,
        model_id: Optional[str] = None,
        *,
        keep_system_prompt: bool = True,
    ) -> None:
        resolved_id, _, session = self._require_model(model_id)
        session.chat_history = []
        if keep_system_prompt and session.system_prompt:
            session.chat_history.append(self._create_message("system", session.system_prompt))
        elif not keep_system_prompt:
            session.system_prompt = ""

        session.pending_context = None
        session.pending_context_tokens = 0
        session.last_response = None
        session.last_usage = UsageMetrics()
        session.chat_usage = UsageMetrics()
        session.usage_history = []
        self._set_active_model(resolved_id)

    def get_model(self, model_id: Optional[str] = None) -> LlmModel:
        _, model, _ = self._require_model(model_id)
        return model

    def set_current_context(self, context: str, *, model_id: Optional[str] = None) -> None:
        resolved_id, model, session = self._require_model(model_id)
        cleaned_context = context.strip()
        if not cleaned_context:
            session.pending_context = None
            session.pending_context_tokens = 0
        else:
            session.pending_context = cleaned_context
            session.pending_context_tokens = model.estimate_tokens(cleaned_context)
        self._set_active_model(resolved_id)

    def _load_prompts(self) -> None:
        """Load prompts from disk into memory."""

        try:
            raw_payload = self._prompts_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            self._prompts = {}
            return
        except OSError as exc:
            raise RuntimeError(f"Unable to read prompts file '{self._prompts_path}': {exc}") from exc

        if not raw_payload.strip():
            self._prompts = {}
            return

        try:
            records = json.loads(raw_payload)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Prompts file '{self._prompts_path}' is not valid JSON: {exc}") from exc

        if not isinstance(records, list):
            raise ValueError(f"Prompts file '{self._prompts_path}' must contain a JSON list of prompts.")

        prompts: dict[str, dict[str, Any]] = {}
        for record in records:
            if not isinstance(record, dict):
                continue
            name = record.get("name")
            if not name:
                continue
            prompts[str(name)] = record

        self._prompts = prompts

    def _render_prompt(self, prompt_entry: dict[str, Any]) -> str:
        """Compose the full system prompt text from a prompt entry."""

        sections: list[str] = []

        task = prompt_entry.get("task")
        if task:
            sections.append(f"Task:\n{str(task).strip()}")

        instructions = prompt_entry.get("prompt")
        if instructions:
            sections.append(f"Instructions:\n{str(instructions).strip()}")

        expected_output = prompt_entry.get("expected_output")
        if isinstance(expected_output, dict) and expected_output:
            expected_sections: list[str] = []
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
            rendered_examples: list[str] = []
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


    def _resolve_model_config(self, model_id: str, overrides: dict[str, Any]) -> dict[str, Any]:
        base_config = self._models_config.get(model_id)
        if base_config is None:
            raise ValueError(f"Model '{model_id}' is not defined.")
        merged_config = base_config.copy()
        merged_config.update(overrides)
        self._models_config[model_id] = merged_config
        return merged_config

    def _require_model(self, model_id: Optional[str]) -> tuple[str, LlmModel, ModelSession]:
        resolved_id = model_id or self._active_model_id
        if resolved_id is None:
            raise RuntimeError("No model is active. Call 'initialize_model' first.")
        model = self._active_models.get(resolved_id)
        session = self._sessions.get(resolved_id)
        if model is None or session is None:
            raise ValueError(f"Model '{resolved_id}' is not loaded.")
        self._set_active_model(resolved_id)
        return resolved_id, model, session

    def _set_active_model(self, model_id: str) -> None:
        self._active_model_id = model_id

    def _get_active_session(self) -> Optional[ModelSession]:
        if self._active_model_id is None:
            return None
        return self._sessions.get(self._active_model_id)

    def _create_message(
        self,
        role: str,
        content: str,
        *,
        context: Optional[str] = None,
    ) -> dict[str, str]:
        content_value = content.strip()
        if context:
            content_value = f"<context>{context}</context>\n\n{content_value}"
        return {"role": role, "content": content_value}

    def _clone_usage(self, usage: UsageMetrics) -> UsageMetrics:
        return UsageMetrics(
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
            context_tokens=usage.context_tokens,
            response_seconds=usage.response_seconds,
        )

    def _update_chat_usage(self, aggregate: UsageMetrics, usage: UsageMetrics) -> None:
        aggregate.prompt_tokens += usage.prompt_tokens
        aggregate.completion_tokens += usage.completion_tokens
        aggregate.total_tokens += usage.total_tokens
        aggregate.context_tokens += usage.context_tokens
        aggregate.response_seconds = None
