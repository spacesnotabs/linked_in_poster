from __future__ import annotations

import os
from dataclasses import dataclass, asdict
from time import perf_counter
from typing import Any, Optional, Tuple

try:  # pragma: no cover - optional dependency
    from llama_cpp import Llama  # type: ignore
except Exception:  # noqa: BLE001
    Llama = None  # type: ignore

from src.domain.models.llm import UsageMetrics


@dataclass(slots=True)
class LLMConfig:
    model_path: str
    model_name: str
    system_prompt: str
    context_size: int
    n_context_size: int
    n_threads: int
    n_threads_batch: int
    temperature: float
    verbose: bool
    use_mmap: bool
    logits_all: bool
    chat_format: str


class LlmRuntime:
    """Adapter for llama.cpp or dummy backends that exposes a chat-completion API."""

    def __init__(
        self,
        *,
        model_path: str,
        model_name: str,
        system_prompt: str,
        chat_format: str,
        n_gpu_layers: int = -1,
        context_size: int = 32768,
        config_overrides: Optional[dict[str, Any]] = None,
    ) -> None:
        self._initialize_model(
            model_path=model_path,
            model_name=model_name,
            system_prompt=system_prompt,
            chat_format=chat_format,
            n_gpu_layers=n_gpu_layers,
            context_size=context_size,
            config_overrides=config_overrides,
        )

    def _initialize_model(
        self,
        *,
        model_path: str,
        model_name: str,
        system_prompt: str,
        chat_format: str,
        n_gpu_layers: int,
        context_size: int,
        config_overrides: Optional[dict[str, Any]],
    ) -> None:
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
            temperature=0.7,
            verbose=False,
            use_mmap=True,
            logits_all=False,
            chat_format=chat_format,
        )
        self._apply_overrides(config_overrides or {})
        self._model_name = model_name

        if model_path == "DUMMY":
            self._model = self._create_dummy_model()
        elif Llama is not None:
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
        else:
            raise RuntimeError(
                "llama_cpp is required for non-dummy models. Install it with `pip install llama-cpp-python`."
            )

    def _apply_overrides(self, overrides: dict[str, Any]) -> None:
        if not overrides:
            return

        int_fields = {"context_size", "n_context_size", "n_threads", "n_threads_batch"}
        float_fields = {"temperature"}
        bool_fields = {"verbose", "use_mmap", "logits_all"}

        for key, value in overrides.items():
            if value is None or not hasattr(self._llm_config, key):
                continue
            if key in int_fields:
                try:
                    setattr(self._llm_config, key, int(value))
                except (TypeError, ValueError):
                    continue
            elif key in float_fields:
                try:
                    setattr(self._llm_config, key, float(value))
                except (TypeError, ValueError):
                    continue
            elif key in bool_fields:
                setattr(self._llm_config, key, bool(value))

    def _create_dummy_model(self) -> Any:
        class _Dummy:
            def __init__(self, name: str) -> None:
                self._name = name

            def create_chat_completion(self, **_kwargs: Any) -> dict[str, Any]:
                return {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": f"[{self._name}] Dummy response.",
                            }
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                    },
                }

            def tokenize(self, text: bytes, add_bos: bool = False, special: bool = False) -> list[int]:
                return list(range(len(text)))

            def detokenize(self, token_ids: list[int]) -> bytes:
                return bytes(token_ids)

        return _Dummy(self._model_name)

    @property
    def model(self) -> Any:
        return self._model

    @property
    def model_name(self) -> str:
        return self._model_name

    def get_config(self) -> dict[str, Any]:
        return asdict(self._llm_config)

    def set_system_prompt(self, system_prompt: str) -> None:
        self._llm_config.system_prompt = system_prompt

    def generate_response(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 1024,
        temperature: Optional[float] = None,
        top_p: float = 0.9,
        top_k: int = 50,
        stream: bool = False,
        **extra_params: Any,
    ) -> Tuple[str, dict[str, Any], float]:
        effective_temp = (
            float(temperature)
            if temperature is not None
            else getattr(self._llm_config, "temperature", 0.7)
        )

        params = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": effective_temp,
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
                tokens = tokenizer.tokenize(text.encode("utf-8"), add_bos=False, special=False)
                return len(tokens)
            except TypeError:
                tokens = tokenizer.tokenize(text.encode("utf-8"))
                return len(tokens)
            except Exception:
                return 0
        return 0

    def tokenize_text(self, text: str) -> list[int]:
        if not text:
            return []
        tokenizer = getattr(self, "_model", None)
        if tokenizer and hasattr(tokenizer, "tokenize"):
            try:
                return tokenizer.tokenize(text.encode("utf-8"), add_bos=False, special=False)
            except TypeError:
                return tokenizer.tokenize(text.encode("utf-8"))
            except Exception:
                return []
        return []

    def detokenize_ids(self, token_ids: list[int]) -> str:
        if not token_ids:
            return ""
        tokenizer = getattr(self, "_model", None)
        if tokenizer and hasattr(tokenizer, "detokenize"):
            try:
                text = tokenizer.detokenize(token_ids)
                return text.decode("utf-8") if isinstance(text, (bytes, bytearray)) else str(text)
            except Exception:
                return ""
        return ""

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
