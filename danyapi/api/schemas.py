from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


class DeepSeekStreamError(Exception):
    pass


ALLOWED_ROLES = frozenset({"user", "assistant", "system", "developer", "tool", "function"})
ALLOWED_ROLES_TEXT = ", ".join(sorted(ALLOWED_ROLES))


class ChatMessage(BaseModel):
    role: str = "user"
    content: Any = ""
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None
    name: str | None = None

    @field_validator("role")
    @classmethod
    def validate_role(cls, v: str) -> str:
        if v not in ALLOWED_ROLES:
            raise ValueError(f"Invalid role: {v}. Allowed roles: {ALLOWED_ROLES_TEXT}")
        return v

    @field_validator("tool_calls", mode="before")
    @classmethod
    def validate_tool_calls(cls, v: list[Any] | None) -> list[dict[str, Any]] | None:
        if v is None:
            return None
        if not isinstance(v, list):
            raise ValueError("tool_calls must be a list")
        validated_tool_calls = []
        for tool_call in v:
            if not isinstance(tool_call, dict):
                raise ValueError("Each tool_call must be a dictionary")
            if "function" not in tool_call and "name" not in tool_call:
                raise ValueError("Each tool_call must contain 'function' or 'name'")
            validated_tool_calls.append(tool_call)
        return validated_tool_calls


class FileSpec(BaseModel):
    name: str
    content: str
    content_type: str = "application/octet-stream"


class ChatCompletionRequest(BaseModel):
    model: str = Field(default="deepseek-v4.1-flash")
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
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    n: int | None = None
    stop: Any = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    seed: int | None = None
    logit_bias: dict[str, float] | None = None
    logprobs: bool | None = None
    top_logprobs: int | None = None
    modalities: list[str] | None = None
    store: bool | None = None
    metadata: dict[str, Any] | None = None
    functions: list[Any] | None = None
    function_call: Any = None


class CompletionRequest(BaseModel):
    model: str = Field(default="deepseek-v4.1-flash")
    prompt: Any = ""
    suffix: str | None = None
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    n: int | None = None
    stream: bool = False
    stop: Any = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    logit_bias: dict[str, float] | None = None
    user: str | None = None
    session_id: str | None = None


class ImageGenerationRequest(BaseModel):
    model: str = Field(default="qwen-image-gen")
    prompt: str
    n: int = Field(default=1, ge=1, le=4)
    size: str | None = None
    response_format: str = Field(default="url")
    session_id: str | None = None
    user: str | None = None


class ResponsesRequest(BaseModel):
    model: str = Field(default="deepseek-v4.1-flash")
    input: Any = ""
    instructions: str | None = None
    stream: bool = False
    temperature: float | None = None
    top_p: float | None = None
    max_output_tokens: int | None = None
    tools: list[Any] | None = None
    tool_choice: Any = None
    parallel_tool_calls: bool | None = None
    text: Any = None
    response_format: Any = None
    reasoning: Any = None
    previous_response_id: str | None = None
    store: bool = True
    metadata: Any = None
    truncation: str = "disabled"
    user: str | None = None
    session_id: str | None = None
    thinking: bool | None = None
    search: bool | None = None
