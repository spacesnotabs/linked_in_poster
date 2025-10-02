from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

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
