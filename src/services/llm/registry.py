from __future__ import annotations

import copy
from typing import Any, Dict, Optional

from src.infrastructure.config.settings import load_model_catalog
from src.services.llm.runtime import LlmRuntime


class ModelRegistry:
    """Manage configured models and instantiate runtimes on demand."""

    def __init__(self, *, config_path, default_overrides: Optional[Dict[str, Any]] = None) -> None:
        system_prompt, models_config = load_model_catalog(config_path)
        self._default_system_prompt = system_prompt
        self._models_config: Dict[str, Dict[str, Any]] = {
            name: config.copy() for name, config in models_config.items()
        }
        self._runtime_overrides: Dict[str, Dict[str, Any]] = default_overrides or {}
        self._loaded_runtimes: Dict[str, LlmRuntime] = {}
        self._model_errors: Dict[str, str] = {}

    @property
    def default_system_prompt(self) -> str:
        return self._default_system_prompt

    def available_models(self) -> list[str]:
        return list(self._models_config.keys())

    def loaded_models(self) -> list[str]:
        return list(self._loaded_runtimes.keys())

    def get_model_config(self, model_id: str) -> Dict[str, Any]:
        config = self._models_config.get(model_id)
        if config is None:
            raise KeyError(f"Model '{model_id}' is not defined.")
        return copy.deepcopy(config)

    def describe_all_models(self) -> Dict[str, Dict[str, Any]]:
        return {name: self.get_model_config(name) for name in self._models_config}

    def set_runtime_overrides(self, model_id: str, overrides: Dict[str, Any]) -> Dict[str, Any]:
        sanitized = self._sanitize_runtime_overrides(overrides)
        self._runtime_overrides[model_id] = sanitized
        return sanitized

    def get_runtime_overrides(self, model_id: str) -> Dict[str, Any]:
        return copy.deepcopy(self._runtime_overrides.get(model_id, {}))

    def load_runtime(
        self,
        model_id: str,
        *,
        force_reload: bool = False,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> LlmRuntime:
        if force_reload:
            self._loaded_runtimes.pop(model_id, None)

        if model_id in self._loaded_runtimes:
            return self._loaded_runtimes[model_id]

        config = self._resolve_model_config(model_id, overrides or {})
        model_path = config.get("model_filename") or config.get("model_path")
        if not model_path:
            raise ValueError(f"Model configuration for '{model_id}' is missing 'model_filename'.")

        try:
            runtime = LlmRuntime(
                model_path=model_path,
                model_name=model_id,
                system_prompt=config.get("system_prompt") or self._default_system_prompt,
                chat_format=config.get("chat_format") or "auto",
                n_gpu_layers=config.get("n_gpu_layers", -1),
                context_size=config.get("context_window") or config.get("context_size") or 32768,
                config_overrides=self.get_runtime_overrides(model_id) or None,
            )
        except Exception as exc:
            self._model_errors[model_id] = str(exc)
            raise

        self._loaded_runtimes[model_id] = runtime
        self._model_errors.pop(model_id, None)
        return runtime

    def get_runtime(self, model_id: str) -> LlmRuntime:
        runtime = self._loaded_runtimes.get(model_id)
        if runtime is None:
            raise KeyError(f"Model '{model_id}' is not loaded.")
        return runtime

    def update_system_prompt(self, model_id: str, prompt: str) -> None:
        if model_id in self._models_config:
            self._models_config[model_id]["system_prompt"] = prompt

    def last_error(self, model_id: str) -> Optional[str]:
        return self._model_errors.get(model_id)

    def errors(self) -> Dict[str, str]:
        return self._model_errors.copy()

    def _resolve_model_config(self, model_id: str, overrides: Dict[str, Any]) -> Dict[str, Any]:
        base_config = self.get_model_config(model_id)
        resolved = copy.deepcopy(base_config)
        resolved.update(overrides)
        return resolved

    @staticmethod
    def _sanitize_runtime_overrides(overrides: Dict[str, Any]) -> Dict[str, Any]:
        allowed = {
            "context_size",
            "n_context_size",
            "n_threads",
            "n_threads_batch",
            "temperature",
            "verbose",
            "use_mmap",
            "logits_all",
        }
        int_fields = {"context_size", "n_context_size", "n_threads", "n_threads_batch"}
        float_fields = {"temperature"}
        bool_fields = {"verbose", "use_mmap", "logits_all"}

        sanitized: Dict[str, Any] = {}
        for key in allowed:
            if key not in overrides:
                continue
            value = overrides[key]
            if value is None or value == "":
                continue
            if key in int_fields:
                try:
                    sanitized[key] = int(value)
                except (TypeError, ValueError):
                    continue
            elif key in float_fields:
                try:
                    sanitized[key] = float(value)
                except (TypeError, ValueError):
                    continue
            elif key in bool_fields:
                sanitized[key] = bool(value)
        return sanitized
