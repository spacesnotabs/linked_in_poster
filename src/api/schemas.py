from pydantic import BaseModel

class ChatRequest(BaseModel):
    prompt: str

class ChatResponse(BaseModel):
    response: str

class ModelSwitchRequest(BaseModel):
    model_name: str