from llama_cpp import Llama
import os
from dataclasses import dataclass, asdict
from typing import Any

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
                return {
                    'choices': [{
                        'message': {
                            'content': "This is a dummy response from a mock model."
                        }
                    }]
                }
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

    def clear_chat_history(self) -> None:
        """Resets the chat history."""
        self._chat_history = []

    def create_prompt(self, role: str, content: str) -> dict:
        """Create the prompt in the expected format."""
        content = content.strip()
        if self._current_context:
            content = f"<context>{self._current_context}</context>\n\n{content}"
            self._current_context = ""

        prompt={'role': role, 'content': content}
        return prompt

    def send_prompt(self, prompt: str) -> str | None:
        """Send the prompt to the LLM and return its response"""
        text = None

        user_prompt = self.create_prompt(role='user', content=prompt)
        self.chat_history.append(user_prompt)

        try:
            resp = self.model.create_chat_completion(
                messages=self._chat_history,
                max_tokens=512,
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                stream=False
            )

            # print(resp)
            text = resp['choices'][0]['message']['content']
            assistant_prompt = self.create_prompt(role='assistant', content=text)
            self.chat_history.append(assistant_prompt)

        except Exception as e:
            print(f"Exception resulted in sending prompt: {e}")

        return text
    


