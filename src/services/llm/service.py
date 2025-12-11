from __future__ import annotations

import copy
import os
from typing import Any, Dict, Optional

from src.infrastructure.config.settings import AppSettings, get_settings
from src.services.llm.prompts import PromptRepository
from src.services.llm.registry import ModelRegistry
from src.services.llm.session_manager import ModelSession, SessionManager


class LLMService:
    """High-level façade composing registry, sessions, and prompt repository."""

    def __init__(
        self,
        *,
        settings: Optional[AppSettings] = None,
        registry: Optional[ModelRegistry] = None,
        session_manager: Optional[SessionManager] = None,
        prompt_repository: Optional[PromptRepository] = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._settings.ensure_runtime_dirs()
        self._registry = registry or ModelRegistry(config_path=self._settings.model_config_path)
        self._sessions = session_manager or SessionManager(self._settings.chat_logs_dir)
        prompt_path = self._settings.prompts_path
        self._prompts = prompt_repository or PromptRepository(prompt_path)
        self._active_model_id: Optional[str] = None

    # ------------------------------------------------------------------ properties
    @property
    def available_models(self) -> list[str]:
        return self._registry.available_models()

    @property
    def loaded_models(self) -> list[str]:
        return self._registry.loaded_models()

    @property
    def active_model_id(self) -> Optional[str]:
        return self._active_model_id

    @property
    def chat(self) -> list[dict[str, str]]:
        if self._active_model_id is None:
            return []
        try:
            session = self._sessions.get_session(self._active_model_id)
            return [message.copy() for message in session.chat_history]
        except KeyError:
            return []

    @property
    def last_response(self) -> Optional[str]:
        if self._active_model_id is None:
            return None
        try:
            session = self._sessions.get_session(self._active_model_id)
            return session.last_response
        except KeyError:
            return None

    # ------------------------------------------------------------------ lifecycle
    def initialize_model(
        self,
        model_id: str,
        *,
        force_reload: bool = False,
        **overrides: Any,
    ):
        if not force_reload and model_id in self._registry.loaded_models():
            runtime = self._registry.get_runtime(model_id)
            self._active_model_id = model_id
            self._sessions.ensure_session(model_id, runtime.get_config().get("system_prompt", ""))
            return runtime

        runtime = self._registry.load_runtime(
            model_id,
            force_reload=force_reload,
            overrides=overrides,
        )
        config = runtime.get_config()
        system_prompt = config.get("system_prompt") or self._registry.default_system_prompt
        self._sessions.reset_session(model_id, system_prompt)
        self._active_model_id = model_id
        return runtime

    # ---------------------------------------------------------------------- prompts
    def list_prompts(self) -> list[str]:
        return self._prompts.list_prompts()

    def get_prompt(self, prompt_name: str) -> dict[str, Any]:
        return self._prompts.get_prompt(prompt_name)

    def render_prompt(self, prompt_name: str) -> str:
        return self._prompts.render_prompt(prompt_name)

    def reload_prompts(self) -> dict[str, dict[str, Any]]:
        self._prompts.reload()
        return {name: self._prompts.get_prompt(name) for name in self._prompts.list_prompts()}

    def apply_prompt(self, prompt_name: str, *, model_id: Optional[str] = None) -> str:
        prompt_text = self.render_prompt(prompt_name)
        self.set_system_prompt(prompt_text, model_id=model_id)
        return prompt_text

    # -------------------------------------------------------------------- sessions
    def send_prompt(
        self,
        prompt: str,
        *,
        model_id: Optional[str] = None,
        max_tokens: int = 1024,
    ) -> Optional[str]:
        resolved_id, runtime, session = self._require_model(model_id)
        user_message = self._sessions.append_user_prompt(session, prompt)
        messages = [message.copy() for message in session.chat_history]

        text, raw_response, elapsed = runtime.generate_response(
            messages=messages,
            max_tokens=max_tokens,
        )
        self._sessions.append_assistant_response(session, text)
        usage = runtime.build_usage_metrics(
            user_prompt=user_message,
            assistant_response=text,
            context_tokens=user_message.get("context_tokens", 0),
            response_seconds=elapsed,
            raw_response=raw_response,
        )
        self._sessions.track_usage(session, usage)
        self._active_model_id = resolved_id
        return text

    def clear_chat_history(
        self,
        model_id: Optional[str] = None,
        *,
        keep_system_prompt: bool = True,
    ) -> None:
        resolved_id, _, _ = self._require_model(model_id)
        self._sessions.clear_history(resolved_id, keep_system_prompt=keep_system_prompt)

    def set_system_prompt(self, prompt: str, *, model_id: Optional[str] = None) -> None:
        resolved_id, runtime, _ = self._require_model(model_id)
        cleaned = prompt.strip()
        runtime.set_system_prompt(cleaned)
        self._registry.update_system_prompt(resolved_id, cleaned)
        self._sessions.reset_session(resolved_id, cleaned)
        self._active_model_id = resolved_id

    def get_active_system_prompt(self, model_id: Optional[str] = None) -> str:
        resolved_id, runtime, session = self._require_model(model_id)
        if session.system_prompt:
            return session.system_prompt
        config = runtime.get_config()
        return config.get("system_prompt") or self._registry.default_system_prompt

    def set_current_context(self, context: str, *, model_id: Optional[str] = None) -> None:
        resolved_id, runtime, _ = self._require_model(model_id)
        self._sessions.set_pending_context(resolved_id, context, runtime.estimate_tokens)
        self._active_model_id = resolved_id

    def get_model(self, model_id: Optional[str] = None):
        resolved_id = model_id or self._active_model_id
        if resolved_id is None:
            raise KeyError("No model is currently active.")
        return self._registry.get_runtime(resolved_id)

    # ---------------------------------------------------------------- configuration
    def describe_model_config(self, model_id: str) -> dict[str, Any]:
        base_config = copy.deepcopy(self._registry.get_model_config(model_id))
        descriptor: dict[str, Any] = {"model_name": model_id}
        base_config.pop("runtime_overrides", None)
        for key, value in base_config.items():
            descriptor[key] = copy.deepcopy(value)
        descriptor.update(self.get_effective_runtime_config(model_id))
        return descriptor

    def describe_all_models(self) -> dict[str, dict[str, Any]]:
        return {model_id: self.describe_model_config(model_id) for model_id in self.available_models}

    def get_last_error(self, model_id: str) -> Optional[str]:
        return self._registry.last_error(model_id)

    def get_error_map(self) -> dict[str, str]:
        return self._registry.errors()

    def get_runtime_overrides(self, model_id: str) -> dict[str, Any]:
        return self._registry.get_runtime_overrides(model_id)

    def set_runtime_overrides(self, model_id: str, overrides: dict[str, Any]) -> dict[str, Any]:
        return self._registry.set_runtime_overrides(model_id, overrides)

    def get_effective_runtime_config(self, model_id: str) -> dict[str, Any]:
        values = self._default_runtime_values(model_id)
        base_config = self._registry.get_model_config(model_id)
        context_value = base_config.get("context_window") or base_config.get("context_size")
        if context_value is not None:
            try:
                coerced = int(context_value)
                values["context_size"] = coerced
                values["n_context_size"] = coerced
            except (TypeError, ValueError):
                pass

        stored = base_config.get("runtime_overrides")
        if isinstance(stored, dict):
            for key in values:
                if key in stored and stored[key] is not None:
                    values[key] = stored[key]

        overrides = self._registry.get_runtime_overrides(model_id)
        for key in values:
            if key in overrides and overrides[key] is not None:
                values[key] = overrides[key]

        if model_id in self._registry.loaded_models():
            current_config = self._registry.get_runtime(model_id).get_config()
            for key in values:
                if key in current_config and current_config[key] is not None:
                    values[key] = current_config[key]

        return values

    # ---------------------------------------------------------------------- state
    def get_state(self) -> dict[str, Any]:
        models_state: dict[str, Any] = {}
        for model_id in self._registry.loaded_models():
            try:
                session = self._sessions.get_session(model_id)
                runtime = self._registry.get_runtime(model_id)
            except KeyError:
                continue
            models_state[model_id] = {
                "is_active": model_id == self._active_model_id,
                "config": runtime.get_config(),
                "pending_context_tokens": session.pending_context_tokens,
                "system_prompt": session.system_prompt,
                "usage": {
                    "last_interaction": session.last_usage.to_dict(),
                    "chat": session.chat_usage.to_dict(),
                    "history": [usage.to_dict() for usage in session.usage_history],
                },
                "last_error": self._registry.last_error(model_id),
            }

        state = {
            "available_models": self.available_models,
            "loaded_models": self.loaded_models,
            "active_model": self._active_model_id,
            "chat": self.chat,
            "last_response": self.last_response,
            "models": models_state,
            "errors": self._registry.errors(),
            "default_system_prompt": self._registry.default_system_prompt,
        }
        return state

    # ------------------------------------------------------------------ helpers
    def _require_model(self, model_id: Optional[str]) -> tuple[str, Any, ModelSession]:
        resolved_id = model_id or self._active_model_id
        if resolved_id is None:
            raise KeyError("No model is currently active.")
        runtime = self._registry.get_runtime(resolved_id)
        session = self._sessions.ensure_session(
            resolved_id,
            runtime.get_config().get("system_prompt") or self._registry.default_system_prompt,
        )
        return resolved_id, runtime, session

    def _default_runtime_values(self, _model_id: str) -> dict[str, Any]:
        cpu_threads = os.cpu_count() or 1
        return {
            "context_size": 32768,
            "n_context_size": 32768,
            "n_threads": cpu_threads,
            "n_threads_batch": max(cpu_threads // 2, 1),
            "temperature": 0.7,
            "verbose": False,
            "use_mmap": True,
            "logits_all": False,
        }
