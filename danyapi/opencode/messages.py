from __future__ import annotations

import json
import logging
from typing import Any

from fastapi import HTTPException

log = logging.getLogger("danyapi.opencode.messages")

EMPTY_PARAMETERS = {"type": "object", "properties": {}}

FINISH_REASON_MAP = {
    "stop": "stop",
    "length": "length",
    "tool_calls": "tool_calls",
    "function_call": "tool_calls",
    "content_filter": "content_filter",
}

PASS_THROUGH_ROLES = ("system", "user", "assistant", "tool")


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") in ("text", "input_text"):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    if content is None or isinstance(content, bool):
        return ""
    if isinstance(content, (int, float)):
        return str(content)
    return ""


def _content_of(content: Any) -> Any:
    if isinstance(content, (str, list)):
        return content
    if content is None:
        return ""
    return _text_of(content)


def _message_dict(message: Any) -> dict[str, Any]:
    role = getattr(message, "role", "user") or "user"
    entry: dict[str, Any] = {"role": role, "content": _content_of(getattr(message, "content", ""))}
    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        normalized: list[dict[str, Any]] = []
        for call in tool_calls:
            if not isinstance(call, dict):
                continue
            function = call.get("function")
            if not isinstance(function, dict):
                function = {"name": call.get("name"), "arguments": call.get("arguments")}
            normalized.append(
                {
                    "id": call.get("id") or f"call_{len(normalized)}",
                    "type": "function",
                    "function": {
                        "name": function.get("name") or "",
                        "arguments": _as_arguments(function.get("arguments")),
                    },
                }
            )
        if normalized:
            entry["tool_calls"] = normalized
    tool_call_id = getattr(message, "tool_call_id", None)
    if isinstance(tool_call_id, str) and tool_call_id:
        entry["tool_call_id"] = tool_call_id
    name = getattr(message, "name", None)
    if isinstance(name, str) and name:
        entry["name"] = name
    return entry


def _as_arguments(value: Any) -> str:
    if isinstance(value, str):
        return value or "{}"
    if value is None:
        return "{}"
    try:
        return json.dumps(value)
    except (TypeError, ValueError):
        return "{}"


def build_messages(messages: list[Any]) -> list[dict]:
    system_parts: list[str] = []
    body: list[dict[str, Any]] = []
    for message in messages:
        role = (getattr(message, "role", "user") or "user").lower()
        if role == "developer":
            role = "system"
        if role == "system":
            text = _text_of(getattr(message, "content", "")).strip()
            if text:
                system_parts.append(text)
            continue
        if role == "function":
            name = getattr(message, "name", None)
            entry: dict[str, Any] = {
                "role": "tool",
                "content": _text_of(getattr(message, "content", "")) or "{}",
            }
            if isinstance(name, str) and name:
                entry["name"] = name
                entry["tool_call_id"] = entry.get("tool_call_id") or name
            body.append(entry)
            continue
        if role not in PASS_THROUGH_ROLES:
            role = "user"
        entry = _message_dict(message)
        entry["role"] = role
        body.append(entry)

    while body and body[0]["role"] != "user":
        body.pop(0)
    if not body:
        raise HTTPException(400, "opencode needs at least one user message")
    if system_parts:
        body.insert(0, {"role": "system", "content": "\n\n".join(system_parts)})
    return body


def _tool_specs(tools: Any) -> list[dict]:
    specs: list[dict] = []
    seen: set[str] = set()
    for item in tools or []:
        if not isinstance(item, dict):
            continue
        function = item.get("function")
        function = function if isinstance(function, dict) else item
        name = function.get("name")
        if not isinstance(name, str) or not name or name in seen:
            continue
        seen.add(name)
        spec: dict[str, Any] = {"name": name}
        description = function.get("description")
        if isinstance(description, str) and description:
            spec["description"] = description
        parameters = function.get("parameters")
        spec["parameters"] = dict(parameters) if isinstance(parameters, dict) and parameters else dict(EMPTY_PARAMETERS)
        specs.append({"type": "function", "function": spec})
    return specs


def request_body(
    messages: list[dict],
    tools: Any = None,
    tool_choice: Any = None,
    temperature: float | None = None,
    top_p: float | None = None,
    max_tokens: int | None = None,
    max_completion_tokens: int | None = None,
    stop: Any = None,
    response_format: Any = None,
    parallel_tool_calls: bool | None = None,
    seed: int | None = None,
    presence_penalty: float | None = None,
    frequency_penalty: float | None = None,
) -> dict:
    body: dict[str, Any] = {"messages": messages}
    specs = _tool_specs(tools)
    if specs:
        body["tools"] = specs
        if tool_choice is not None:
            body["tool_choice"] = tool_choice
        if parallel_tool_calls is not None:
            body["parallel_tool_calls"] = parallel_tool_calls
    if temperature is not None:
        body["temperature"] = temperature
    if top_p is not None:
        body["top_p"] = top_p
    if max_completion_tokens is not None and max_completion_tokens > 0:
        body["max_completion_tokens"] = max_completion_tokens
    elif max_tokens is not None and max_tokens > 0:
        body["max_tokens"] = max_tokens
    if stop is not None:
        body["stop"] = stop
    if isinstance(response_format, dict) and response_format:
        body["response_format"] = response_format
    if seed is not None:
        body["seed"] = seed
    if presence_penalty is not None:
        body["presence_penalty"] = presence_penalty
    if frequency_penalty is not None:
        body["frequency_penalty"] = frequency_penalty
    return body


def normalize_finish_reason(value: Any) -> str:
    if not isinstance(value, str):
        return "stop"
    return FINISH_REASON_MAP.get(value, "stop")


def _token_count(value: Any) -> int:
    if isinstance(value, bool) or value is None:
        return 0
    if isinstance(value, int):
        return max(0, value)
    if isinstance(value, float):
        return int(value) if value > 0 else 0
    if isinstance(value, str):
        try:
            number = int(float(value.strip()))
        except ValueError:
            return 0
        return max(0, number)
    return 0


def normalize_usage(payload: Any) -> dict:
    if not isinstance(payload, dict):
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    prompt = _token_count(payload.get("prompt_tokens"))
    completion = _token_count(payload.get("completion_tokens"))
    total = _token_count(payload.get("total_tokens"))
    if total <= 0 or total < prompt + completion:
        total = prompt + completion
    details = payload.get("prompt_tokens_details")
    cached = _token_count(details.get("cached_tokens")) if isinstance(details, dict) else 0
    completion_details = payload.get("completion_tokens_details")
    reasoning = _token_count(completion_details.get("reasoning_tokens")) if isinstance(completion_details, dict) else 0
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
        "prompt_tokens_details": {"cached_tokens": cached},
        "completion_tokens_details": {"reasoning_tokens": reasoning},
    }
