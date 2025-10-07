from __future__ import annotations

import os
from dataclasses import dataclass, asdict
from time import perf_counter
from typing import Any, Optional

from llama_cpp import Llama


@dataclass
class UsageMetrics:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    context_tokens: int = 0
    response_seconds: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        data = {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "context_tokens": self.context_tokens,
        }
        if self.response_seconds is not None:
            data["response_seconds"] = self.response_seconds
        return data


@dataclass
class LLMConfig:
    model_path: str
    model_name: str
    system_prompt: str
    context_size: int
    n_context_size: int
    n_threads: int
    n_threads_batch: int
    verbose: bool
    use_mmap: bool
    logits_all: bool
    chat_format: str


class LlmModel:
    """LLM wrapper responsible for loading models and executing completions."""

    def __init__(
        self,
        model_path: str,
        model_name: str,
        system_prompt: str,
        chat_format: str,
        n_gpu_layers: int = -1,
        context_size: int = 32768,
        config_overrides: Optional[dict[str, Any]] = None,
    ) -> None:
        self.initialize_model(
            model_path=model_path,
            model_name=model_name,
            system_prompt=system_prompt,
            chat_format=chat_format,
            n_gpu_layers=n_gpu_layers,
            context_size=context_size,
            config_overrides=config_overrides,
        )

    def initialize_model(
        self,
        model_path: str,
        model_name: str,
        system_prompt: str,
        chat_format: str,
        n_gpu_layers: int = -1,
        context_size: int = 32768,
        config_overrides: Optional[dict[str, Any]] = None,
    ) -> None:
        """Initialize the underlying llama.cpp model."""

        _n_threads = os.cpu_count() or 1
        _n_threads_batch = max(_n_threads // 2, 1)

        self._llm_config = LLMConfig(
            model_path=model_path,
            model_name=model_name,
            system_prompt=system_prompt,
            context_size=context_size,
            n_context_size=context_size,
            n_threads=_n_threads,
            n_threads_batch=_n_threads_batch,
            verbose=False,
            use_mmap=True,
            logits_all=False,
            chat_format=chat_format,
        )
        self._apply_overrides(config_overrides or {})
        self._model_name = model_name

        if model_path == "DUMMY":
            self._model = self._create_dummy_model()
        else:
            self._model = Llama(
                model_path=self._llm_config.model_path,
                n_ctx=self._llm_config.n_context_size,
                n_gpu_layers=n_gpu_layers,
                n_threads=self._llm_config.n_threads,
                n_threads_batch=self._llm_config.n_threads_batch,
                verbose=self._llm_config.verbose,
                use_mmap=self._llm_config.use_mmap,
                logits_all=self._llm_config.logits_all,
            )

    def _apply_overrides(self, overrides: dict[str, Any]) -> None:
        if not overrides:
            return

        int_fields = {"context_size", "n_context_size", "n_threads", "n_threads_batch"}
        bool_fields = {"verbose", "use_mmap", "logits_all"}

        for key, value in overrides.items():
            if value is None or not hasattr(self._llm_config, key):
                continue
            if key in int_fields:
                try:
                    coerced = int(value)
                except (TypeError, ValueError):
                    continue
                setattr(self._llm_config, key, coerced)
            elif key in bool_fields:
                if isinstance(value, str):
                    normalized = value.strip().lower()
                    coerced = normalized in {"1", "true", "yes", "on"}
                else:
                    coerced = bool(value)
                setattr(self._llm_config, key, coerced)

        if self._llm_config.context_size is None and self._llm_config.n_context_size is not None:
            self._llm_config.context_size = self._llm_config.n_context_size
        if self._llm_config.n_context_size is None and self._llm_config.context_size is not None:
            self._llm_config.n_context_size = self._llm_config.context_size

    def _create_dummy_model(self) -> Any:
        class DummyLlama:
            def create_chat_completion(self, messages, **_kwargs):
                response_text = "This is a dummy response from a mock model."
                prompt_tokens = sum(len(msg.get("content", "").split()) for msg in messages)
                completion_tokens = len(response_text.split())
                total_tokens = prompt_tokens + completion_tokens
                return {
                    "choices": [
                        {
                            "message": {
                                "content": response_text,
                            }
                        }
                    ],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": total_tokens,
                    },
                }

            def tokenize(self, text: bytes, add_bos: bool = False):
                content = text.decode("utf-8") if isinstance(text, (bytes, bytearray)) else str(text)
                token_count = len(content.split())
                return [1] * token_count

        return DummyLlama()

    @property
    def model(self) -> Any:
        return self._model

    @property
    def model_name(self) -> str:
        return self._model_name

    def get_model_config(self) -> dict[str, Any]:
        return asdict(self._llm_config)

    def generate_response(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        stream: bool = False,
        **extra_params: Any,
    ) -> tuple[str, dict[str, Any], float]:
        """Execute a chat completion and return text, raw response, and latency."""

        params = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "stream": stream,
        }
        params.update(extra_params)

        start_time = perf_counter()
        response = self.model.create_chat_completion(**params)
        elapsed = perf_counter() - start_time

        text = response["choices"][0]["message"].get("content") if response else None
        return text or "", response, elapsed

    def estimate_tokens(self, text: str) -> int:
        if not text:
            return 0

        tokenizer = getattr(self, "_model", None)
        if tokenizer and hasattr(tokenizer, "tokenize"):
            try:
                tokens = tokenizer.tokenize(text.encode("utf-8"), add_bos=False)
                return len(tokens)
            except TypeError:
                tokens = tokenizer.tokenize(text.encode("utf-8"))
                return len(tokens)
            except Exception:
                pass

        return len(text.split())

    def build_usage_metrics(
        self,
        *,
        user_prompt: dict[str, Any],
        assistant_response: str,
        context_tokens: int,
        response_seconds: float,
        raw_response: dict[str, Any],
    ) -> UsageMetrics:
        usage_block = raw_response.get("usage") or raw_response.get("token_usage") or {}

        prompt_tokens = usage_block.get("prompt_tokens") if isinstance(usage_block, dict) else None
        completion_tokens = usage_block.get("completion_tokens") if isinstance(usage_block, dict) else None
        total_tokens = usage_block.get("total_tokens") if isinstance(usage_block, dict) else None

        if prompt_tokens is None:
            prompt_tokens = self.estimate_tokens(str(user_prompt.get("content", "")))
        if completion_tokens is None:
            completion_tokens = self.estimate_tokens(assistant_response)
        if total_tokens is None:
            total_tokens = prompt_tokens + completion_tokens

        return UsageMetrics(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            context_tokens=context_tokens,
            response_seconds=response_seconds,
        )

    def set_system_prompt(self, system_prompt: str) -> None:
        """Update the cached system prompt for the underlying model."""

        self._llm_config.system_prompt = system_prompt
