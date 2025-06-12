from llama_cpp import Llama

class LlmModel:
    def __init__(self, model_path: str, n_gpu_layers: int = -1, context_size: int = 2048):
        self._model = Llama(model_path=model_path, 
                            n_gpu_layers=n_gpu_layers,
                            n_ctx=context_size,
                            verbose=True)

    @property
    def model(self):
        return self._model
    
    def send_prompt(prompt: str) -> str:
        """Send the prompt to the LLM and return its response"""
        return "Testing"
    
