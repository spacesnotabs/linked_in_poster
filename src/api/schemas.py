from typing import Optional

from pydantic import BaseModel


class UsageBreakdown(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    context_tokens: int
    response_seconds: Optional[float] = None

class ChatRequest(BaseModel):
    prompt: str

class ChatResponse(BaseModel):
    response: str
    usage: UsageBreakdown
    chat_usage: UsageBreakdown

class ModelSwitchRequest(BaseModel):
    model_name: str

class PromptApplyRequest(BaseModel):
    prompt_name: Optional[str] = None
