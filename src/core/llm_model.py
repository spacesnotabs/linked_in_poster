from llama_cpp import Llama
import os
from dataclasses import dataclass, asdict
from typing import Any, Optional
from time import perf_counter


@dataclass
class UsageMetrics:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    context_tokens: int = 0
    response_seconds: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        data = {
            'prompt_tokens': self.prompt_tokens,
            'completion_tokens': self.completion_tokens,
            'total_tokens': self.total_tokens,
            'context_tokens': self.context_tokens,
        }
        if self.response_seconds is not None:
            data['response_seconds'] = self.response_seconds
        return data

@dataclass
class LLMConfig:
    model_path: str
    model_name: str
    system_prompt: str
    n_context_size: int
    n_threads: int
    n_threads_batch: int
    verbose: bool
    use_mmap: bool
    logits_all: bool
    chat_format: str

class LlmModel:
    def __init__(self, 
                 model_path: str, 
                 model_name: str, 
                 system_prompt: str, 
                 chat_format: str,
                 n_gpu_layers: int = -1, 
                 context_size: int = 32768):
        
        self.initialize_model(
            model_path=model_path,
            model_name=model_name,
            system_prompt=system_prompt,
            chat_format=chat_format,
            n_gpu_layers=n_gpu_layers,
            context_size=context_size
        )

    def initialize_model(self, 
                        model_path: str, 
                        model_name: str, 
                        system_prompt: str, 
                        chat_format: str,
                        n_gpu_layers: int = -1, 
                        context_size: int = 32768):
        """Initialize the LLM model with the given configuration."""
        
        _n_threads = os.cpu_count()
        _n_threads_batch = _n_threads // 2

        # create an LLMConfig property for initializing the model and store it
        _llm_config: LLMConfig = LLMConfig(
            model_path=model_path,
            model_name=model_name,
            system_prompt=system_prompt,
            n_context_size=context_size,
            n_threads=_n_threads,
            n_threads_batch=_n_threads_batch,
            verbose=False,
            use_mmap=True,
            logits_all=False,
            chat_format=chat_format
        )

        self._llm_config = _llm_config
        self._model_name = model_name
        self._chat_history: list[dict] = []
        self._current_context: str = ""
        self._reset_usage_tracking()

        if model_path == "DUMMY":
            self._model = self._create_dummy_model()
        else:
            self._model = Llama(model_path=_llm_config.model_path,
                                 n_ctx=_llm_config.n_context_size,
                                 n_gpu_layers=n_gpu_layers,
                                 n_threads=_llm_config.n_threads,
                                 n_batch=_llm_config.n_threads_batch,
                                 verbose=_llm_config.verbose,
                                 use_mmap=_llm_config.use_mmap,
                                 logits_all=_llm_config.logits_all)
                                
        self.set_system_prompt(prompt=system_prompt)

    def _create_dummy_model(self):
        class DummyLlama:
            def create_chat_completion(self, messages, **kwargs):
                response_text = "This is a dummy response from a mock model."
                prompt_tokens = sum(len(msg.get('content', '').split()) for msg in messages)
                completion_tokens = len(response_text.split())
                total_tokens = prompt_tokens + completion_tokens
                return {
                    'choices': [{
                        'message': {
                            'content': response_text
                        }
                    }],
                    'usage': {
                        'prompt_tokens': prompt_tokens,
                        'completion_tokens': completion_tokens,
                        'total_tokens': total_tokens
                    }
                }

            def tokenize(self, text: bytes, add_bos: bool = False):
                content = text.decode('utf-8') if isinstance(text, (bytes, bytearray)) else str(text)
                token_count = len(content.split())
                return [1] * token_count
        return DummyLlama()

    @property
    def model(self):
        return self._model
    
    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def chat_history(self) -> list[str]:
        return self._chat_history

    def get_model_config(self) -> dict[str, Any]:
        # convert the _llm_config dataclass to a dictionary and return it
        return asdict(self._llm_config)

    def set_system_prompt(self, prompt: str) -> None:
        system_prompt = self.create_prompt(role='system', content=prompt)
        self.chat_history.append(system_prompt)

    def set_current_context(self, context: str) -> None:
        self._current_context = context
        self._pending_context_tokens = self._estimate_tokens(context)

    def clear_chat_history(self) -> None:
        """Resets the chat history."""
        self._chat_history = []
        self._reset_usage_tracking()

    def create_prompt(self, role: str, content: str) -> dict:
        """Create the prompt in the expected format."""
        content = content.strip()
        context_used = False
        if self._current_context:
            content = f"<context>{self._current_context}</context>\n\n{content}"
            self._current_context = ""
            context_used = True

        prompt={'role': role, 'content': content}
        if context_used:
            self._pending_context_tokens = 0
        return prompt

    def send_prompt(self, prompt: str) -> str | None:
        """Send the prompt to the LLM and return its response"""
        text = None
        context_tokens = self._pending_context_tokens if self._current_context else 0
        self._last_interaction_usage = UsageMetrics()

        user_prompt = self.create_prompt(role='user', content=prompt)
        self.chat_history.append(user_prompt)

        try:
            start_time = perf_counter()
            resp = self.model.create_chat_completion(
                messages=self._chat_history,
                max_tokens=512,
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                stream=False
            )
            elapsed = perf_counter() - start_time

            # print(resp)
            text = resp['choices'][0]['message']['content']
            assistant_prompt = self.create_prompt(role='assistant', content=text)
            self.chat_history.append(assistant_prompt)

            usage = self._build_usage_metrics(
                user_prompt=user_prompt,
                assistant_response=text,
                context_tokens=context_tokens,
                response_seconds=elapsed,
                raw_response=resp
            )
            self._last_interaction_usage = usage
            self._usage_history.append(
                UsageMetrics(
                    prompt_tokens=usage.prompt_tokens,
                    completion_tokens=usage.completion_tokens,
                    total_tokens=usage.total_tokens,
                    context_tokens=usage.context_tokens,
                    response_seconds=usage.response_seconds
                )
            )
            self._update_chat_usage(usage)

        except Exception as e:
            print(f"Exception resulted in sending prompt: {e}")

        return text

    def get_last_interaction_usage(self) -> dict[str, Any]:
        """Return the usage metrics for the most recent prompt/response."""
        return self._last_interaction_usage.to_dict()

    def get_chat_usage(self) -> dict[str, Any]:
        """Return the cumulative usage metrics for the current chat session."""
        return self._chat_usage.to_dict()

    def get_usage_history(self) -> list[dict[str, Any]]:
        """Return the usage metrics for each completed prompt/response pair."""
        return [usage.to_dict() for usage in self._usage_history]

    def _reset_usage_tracking(self) -> None:
        self._chat_usage = UsageMetrics()
        self._last_interaction_usage = UsageMetrics()
        self._usage_history: list[UsageMetrics] = []
        self._pending_context_tokens: int = 0

    def _update_chat_usage(self, usage: UsageMetrics) -> None:
        self._chat_usage.prompt_tokens += usage.prompt_tokens
        self._chat_usage.completion_tokens += usage.completion_tokens
        self._chat_usage.total_tokens += usage.total_tokens
        self._chat_usage.context_tokens += usage.context_tokens
        self._chat_usage.response_seconds = None

    def _estimate_tokens(self, text: str) -> int:
        if not text:
            return 0

        tokenizer = getattr(self, '_model', None)
        if tokenizer and hasattr(tokenizer, 'tokenize'):
            try:
                tokens = tokenizer.tokenize(text.encode('utf-8'), add_bos=False)
                return len(tokens)
            except TypeError:
                tokens = tokenizer.tokenize(text.encode('utf-8'))
                return len(tokens)
            except Exception:
                pass

        return len(text.split())

    def _build_usage_metrics(self,
                             user_prompt: dict,
                             assistant_response: str,
                             context_tokens: int,
                             response_seconds: float,
                             raw_response: dict) -> UsageMetrics:
        usage_block = raw_response.get('usage') or raw_response.get('token_usage') or {}

        prompt_tokens = usage_block.get('prompt_tokens') if isinstance(usage_block, dict) else None
        completion_tokens = usage_block.get('completion_tokens') if isinstance(usage_block, dict) else None
        total_tokens = usage_block.get('total_tokens') if isinstance(usage_block, dict) else None

        if prompt_tokens is None:
            prompt_tokens = self._estimate_tokens(user_prompt.get('content', ''))
        if completion_tokens is None:
            completion_tokens = self._estimate_tokens(assistant_response)
        if total_tokens is None:
            total_tokens = prompt_tokens + completion_tokens

        return UsageMetrics(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            context_tokens=context_tokens,
            response_seconds=response_seconds
        )
    


