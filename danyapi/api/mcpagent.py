from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any

from fastapi import HTTPException

from ..config import settings
from ..mcp import McpRegistry
from .schemas import ChatCompletionRequest, ChatMessage

log = logging.getLogger("danyapi.api")


def _registry() -> McpRegistry | None:
    from .state import app

    registry = getattr(app.state, "mcp_registry", None)
    return registry if isinstance(registry, McpRegistry) else None


def mcp_enabled(req: ChatCompletionRequest) -> bool:
    flag = getattr(req, "mcp", None)
    if flag is not None:
        return bool(flag)
    registry = _registry()
    return registry is not None and registry.enabled()


def mcp_tools_openai() -> list[dict[str, Any]]:
    registry = _registry()
    if registry is None or not registry.enabled():
        return []
    schemas: list[dict[str, Any]] = []
    for _server, tool in registry.all_tools():
        schemas.append(tool.openai_schema())
    return schemas


async def _execute_tool_calls(registry: McpRegistry, tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for call in tool_calls:
        raw_function = call.get("function")
        function = raw_function if isinstance(raw_function, dict) else {}
        name = function.get("name") or call.get("name") or ""
        raw_arguments = function.get("arguments") if "arguments" in function else call.get("arguments", "{}")
        if isinstance(raw_arguments, str):
            try:
                arguments = json.loads(raw_arguments) if raw_arguments.strip() else {}
            except ValueError:
                arguments = {}
        elif isinstance(raw_arguments, dict):
            arguments = raw_arguments
        else:
            arguments = {}
        resolved = registry.resolve(str(name))
        if resolved is None:
            text = f"error: tool {name} is not available"
        else:
            server, tool = resolved
            try:
                text = await registry.call(server, tool, arguments if isinstance(arguments, dict) else {})
            except Exception as exc:
                text = f"error: {type(exc).__name__}: {exc}"
        entry: dict[str, Any] = {"role": "tool", "tool_call_id": call.get("id") or "", "content": text}
        if name:
            entry["name"] = str(name)
        messages.append(entry)
    return messages


def _extract_tool_calls(message: dict[str, Any]) -> list[dict[str, Any]]:
    calls = message.get("tool_calls")
    if not isinstance(calls, list) or not calls:
        return []
    valid: list[dict[str, Any]] = []
    for call in calls:
        if not isinstance(call, dict):
            continue
        function = call.get("function")
        name = function.get("name") if isinstance(function, dict) else call.get("name")
        if isinstance(name, str) and name:
            valid.append(call)
    return valid


def _assistant_history_entry(message: dict[str, Any]) -> dict[str, Any]:
    entry: dict[str, Any] = {"role": "assistant", "content": message.get("content") or ""}
    if message.get("tool_calls"):
        entry["tool_calls"] = message["tool_calls"]
    if message.get("reasoning_content"):
        entry["reasoning_content"] = message["reasoning_content"]
    return entry


def _provider_finish_reason(result: Any) -> str:
    if isinstance(result, dict):
        choices = result.get("choices")
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            finish = choices[0].get("finish_reason")
            if isinstance(finish, str) and finish:
                return finish
    return "stop"


def _provider_message(result: Any) -> dict[str, Any]:
    if isinstance(result, dict):
        choices = result.get("choices")
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            message = choices[0].get("message")
            if isinstance(message, dict):
                return message
    return {"role": "assistant", "content": ""}


def _provider_usage(result: Any) -> dict[str, Any]:
    if isinstance(result, dict) and isinstance(result.get("usage"), dict):
        return result["usage"]
    return {}


def _merge_usage(total: dict[str, Any], addition: dict[str, Any]) -> dict[str, Any]:
    for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = addition.get(field)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            total[field] = total.get(field, 0) + int(value)
    for detail_key in ("prompt_tokens_details", "completion_tokens_details"):
        detail = addition.get(detail_key)
        if isinstance(detail, dict):
            target = total.setdefault(detail_key, {})
            for key, value in detail.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    target[key] = target.get(key, 0) + int(value)
    total.setdefault("prompt_tokens", 0)
    total.setdefault("completion_tokens", 0)
    total.setdefault("total_tokens", 0)
    if total.get("total_tokens") == 0:
        total["total_tokens"] = (total.get("prompt_tokens") or 0) + (total.get("completion_tokens") or 0)
    return total


async def run_mcp_chat(req: ChatCompletionRequest, dispatch) -> Any:
    registry = _registry()
    if registry is None or not registry.enabled():
        raise HTTPException(503, "mcp tool execution is not configured")
    if req.stream:
        raise HTTPException(400, "streaming is not supported together with mcp tool execution, send stream=false")

    base_messages = [message.model_dump() for message in req.messages]
    extra_tools = mcp_tools_openai()
    if extra_tools:
        existing = list(req.tools or [])
        req.tools = existing + extra_tools
    if req.tool_choice is None:
        req.tool_choice = "auto"

    iterations_left = registry_iterations()
    history = list(base_messages)
    usage: dict[str, Any] = {}
    last_result: Any = None

    while iterations_left > 0:
        iterations_left -= 1
        req.messages = [ChatMessage(**{key: value for key, value in message.items() if key in _MESSAGE_FIELDS}) for message in history]
        result = await dispatch(req)
        last_result = result
        _merge_usage(usage, _provider_usage(result))
        message = _provider_message(result)
        calls = _extract_tool_calls(message)
        if not calls:
            return _final_response(result, usage)
        history.append(_assistant_history_entry(message))
        history.extend(await _execute_tool_calls(registry, calls))

    final_message = _provider_message(last_result) if last_result is not None else {"role": "assistant", "content": ""}
    if _extract_tool_calls(final_message):
        finish = "tool_calls"
    else:
        finish = _provider_finish_reason(last_result)
        last_result = None
    text = final_message.get("content") or ""
    note = f"\n\n[mcp iteration limit of {registry_iterations()} reached, answer with what you have]"
    message_out = dict(final_message)
    if isinstance(text, str):
        message_out["content"] = (text + note) if text else note.strip()
    else:
        message_out["content"] = text
    return {
        "id": _result_id(last_result),
        "object": "chat.completion",
        "created": _result_created(last_result),
        "model": req.model,
        "system_fingerprint": "fp_danyapi",
        "choices": [{"index": 0, "message": message_out, "finish_reason": finish, "logprobs": None}],
        "usage": usage,
        "session_id": _result_session(last_result),
    }


_MESSAGE_FIELDS = frozenset(ChatMessage.model_fields)


def registry_iterations() -> int:
    return settings.mcp_iterations


def _final_response(result: Any, usage: dict[str, Any]) -> Any:
    if isinstance(result, dict):
        if usage and isinstance(result.get("usage"), dict):
            result["usage"] = usage
        elif usage:
            result = dict(result)
            result["usage"] = usage
        return result
    return result


def _result_id(result: Any) -> str:
    if isinstance(result, dict) and isinstance(result.get("id"), str) and result["id"]:
        return result["id"]
    return f"chatcmpl-{uuid.uuid4().hex}"


def _result_created(result: Any) -> int:
    if isinstance(result, dict) and isinstance(result.get("created"), int):
        return result["created"]
    return int(time.time())


def _result_session(result: Any) -> str | None:
    if isinstance(result, dict) and isinstance(result.get("session_id"), str):
        return result["session_id"]
    return None
