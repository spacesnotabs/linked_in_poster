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


class ModelSettingsPayload(BaseModel):
    model_name: str
    n_context_size: int
    n_threads: int
    n_threads_batch: int
    temperature: float
    verbose: bool
    use_mmap: bool
    logits_all: bool


class ChunkJobRequest(BaseModel):
    paths: list[str]
    include: Optional[list[str]] = None
    exclude: Optional[list[str]] = None
