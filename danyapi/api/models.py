from __future__ import annotations

from typing import Any

from fastapi import HTTPException
from pydantic import BaseModel, Field

MODEL_TYPE_BY_NAME = {
    "deepseek-v4-flash": "default",
    "deepseek-v4-pro": "expert",
    "deepseek-v4-vision": "vision",
}

QWEN_DEFAULT_MODELS = [
    {
        "id": "qwen3.8-max",
        "name": "Qwen3.8-Max",
        "owned_by": "qwen",
        "model_type": "chat",
    },
    {
        "id": "qwen3.7-plus",
        "name": "Qwen3.7-Plus",
        "owned_by": "qwen",
        "model_type": "chat",
    },
    {
        "id": "qwen3.7-max",
        "name": "Qwen3.7-Max",
        "owned_by": "qwen",
        "model_type": "chat",
    },
]

STATUS_TO_FINISH_REASON = {
    "FINISHED": "stop",
    "CONTEXT_LENGTH_EXCEEDED": "length",
    "CONTENT_FILTER": "content_filter",
    "INCOMPLETE": "stop",
    "WIP": "stop",
    "TIMEOUT": "stop",
}

CONTEXT_LENGTH_STATUS = "CONTEXT_LENGTH_EXCEEDED"
INPUT_EXCEEDS_LIMIT = "input_exceeds_limit"
CONTINUE_PROMPT = "Continue"
MAX_CONTINUE_ROUNDS = 5


class ChatMessage(BaseModel):
    role: str = "user"
    content: Any = ""
    tool_calls: list[Any] | None = None
    tool_call_id: str | None = None
    name: str | None = None


class FileSpec(BaseModel):
    name: str
    content: str
    content_type: str = "application/octet-stream"


class ChatCompletionRequest(BaseModel):
    model: str = Field(default="deepseek-v4-flash")
    messages: list[ChatMessage] = Field(default_factory=list)
    stream: bool = False
    temperature: float | None = None
    top_p: float | None = None
    thinking: bool | None = None
    search: bool | None = None
    session_id: str | None = None
    user: str | None = None
    files: list[FileSpec] | None = None
    tools: list[Any] | None = None
    tool_choice: Any = None
    parallel_tool_calls: bool | None = None
    response_format: Any = None
    stream_options: Any = None


def _resolve_model(model: str) -> str:
    model_type = MODEL_TYPE_BY_NAME.get(model)
    if model_type is None:
        raise HTTPException(404, f"Unknown model: {model}")
    return model_type


def _finish_reason(status: Any) -> str:
    if isinstance(status, str):
        return STATUS_TO_FINISH_REASON.get(status, "stop")
    return "stop"


def _deepseek_usage(total: int) -> dict:
    value = max(0, int(total or 0))
    return {"prompt_tokens": 0, "completion_tokens": value, "total_tokens": value}


def _include_usage(req: ChatCompletionRequest) -> bool:
    opts = getattr(req, "stream_options", None)
    return bool((opts or {}).get("include_usage"))
