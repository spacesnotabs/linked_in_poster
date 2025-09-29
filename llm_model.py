from llama_cpp import Llama

class LlmModel:
    def __init__(self, model_path: str, model_name: str, system_prompt: str, n_gpu_layers: int = -1, context_size: int = 32768):
        self._model = Llama(model_path=model_path,
                            n_gpu_layers=n_gpu_layers,
                            n_ctx=context_size,
                            verbose=False,
                            chat_format='mistral-instruct')
        self._chat_history: list[dict] = []
        self._model_name = model_name
        self.set_system_prompt(prompt=system_prompt)

    @property
    def model(self) -> Llama:
        return self._model
    
    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def chat_history(self) -> list[str]:
        return self._chat_history

    def set_system_prompt(self, prompt: str) -> None:
        system_prompt = self.create_prompt(role='system', content=prompt)
        self.chat_history.append(system_prompt)

    def clear_chat_history(self) -> None:
        """Resets the chat history."""
        self._chat_history = []

    def create_prompt(self, role: str, content: str) -> dict:
        """Create the prompt in the expected format."""
        prompt={'role': role, 'content': content.strip()}
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
    
