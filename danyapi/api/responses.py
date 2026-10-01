from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import logging
import uuid
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from functools import partial
from typing import Any

from fastapi import HTTPException

from danyapi.tokens import estimate_tokens

from .anthropic import iter_sse_payloads as _iter_sse_payloads
from .anthropic import sse_event

log = logging.getLogger("danyapi.api.responses")


class ResponsesInputError(ValueError):
    pass


TEXT_PART_TYPES = {"input_text", "output_text", "text", "summary_text"}
IMAGE_PART_TYPES = {"input_image", "image_url"}
SUPPORTED_ROLES = {"user", "assistant", "system", "developer", "tool", "function"}
IMAGE_DETAILS = {"low", "high", "auto"}
RESPONSE_INCOMPLETE = "response_incomplete"
REDUCED_CONTEXT_REASON = "max_output_tokens"
UNSUPPORTED_ITEM_TYPES = {
    "reasoning": "reasoning items are produced by the provider and cannot be replayed as input",
    "item_reference": "item references cannot be resolved without the originating response",
    "web_search_call": "server side tool calls cannot be replayed as input",
    "file_search_call": "server side tool calls cannot be replayed as input",
    "code_interpreter_call": "server side tool calls cannot be replayed as input",
}
NON_TERMINAL_STATUSES = {"queued", "in_progress"}
MAX_TEXT_DEPTH = 8


def _as_int(value: Any) -> int:
    if value is None or isinstance(value, bool):
        return 0
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError):
        try:
            number = int(float(value))
        except (TypeError, ValueError, OverflowError):
            return 0
    return max(0, number)


def _as_text(value: Any, depth: int = 0) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if depth >= MAX_TEXT_DEPTH:
        return ""
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
                elif item.get("type") == "output_text" and isinstance(item.get("content"), str):
                    parts.append(item["content"])
            else:
                parts.append(_as_text(item, depth + 1))
        return "".join(parts)
    if isinstance(value, dict):
        if isinstance(value.get("text"), str):
            return value["text"]
        try:
            return json.dumps(value, ensure_ascii=False)
        except (TypeError, ValueError, RecursionError):
            return str(value)
    return str(value)


def _image_source(part: dict) -> tuple[str, Any]:
    image_url = part.get("image_url")
    if isinstance(image_url, dict):
        detail = image_url.get("detail")
        image_url = image_url.get("url")
    else:
        detail = part.get("detail")
    return (image_url if isinstance(image_url, str) else ""), detail


def _image_detail(detail: Any) -> str | None:
    return detail if isinstance(detail, str) and detail in IMAGE_DETAILS else None


def _normalize_content(content: Any) -> Any:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return _as_text(content)
    parts: list[Any] = []
    for item in content:
        if isinstance(item, str):
            parts.append({"type": "text", "text": item})
            continue
        if not isinstance(item, dict):
            raise ResponsesInputError("each content part must be an object")
        part_type = item.get("type")
        if part_type in TEXT_PART_TYPES:
            parts.append({"type": "text", "text": _as_text(item.get("text"))})
        elif part_type in IMAGE_PART_TYPES:
            url, detail = _image_source(item)
            if not url:
                raise ResponsesInputError("input_image requires an image_url string")
            image_part: dict[str, Any] = {"type": "image_url", "image_url": url}
            valid_detail = _image_detail(detail)
            if valid_detail:
                image_part["detail"] = valid_detail
            parts.append(image_part)
        elif part_type == "refusal":
            parts.append({"type": "text", "text": _as_text(item.get("refusal"))})
        elif part_type in ("input_file", "file", "input_audio"):
            continue
        elif isinstance(item.get("text"), str):
            parts.append({"type": "text", "text": item["text"]})
    if not parts:
        return ""
    if all(isinstance(part, dict) and part.get("type") == "text" for part in parts):
        return "".join(part["text"] for part in parts)
    return parts


def _normalize_item(item: Any) -> list[dict]:
    if not isinstance(item, dict):
        raise ResponsesInputError("each input item must be an object")
    item_type = item.get("type")
    if item_type in ("function_call", "computer_call", "custom_tool_call"):
        call_id = item.get("call_id") or item.get("id") or f"call_{uuid.uuid4().hex[:12]}"
        name = item.get("name") or ""
        arguments = item.get("arguments")
        if isinstance(arguments, (dict, list)):
            arguments = json.dumps(arguments, ensure_ascii=False)
        return [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": name, "arguments": arguments if isinstance(arguments, str) else "{}"},
                    }
                ],
            }
        ]
    if item_type in ("function_call_output", "computer_call_output", "custom_tool_call_output"):
        call_id = item.get("call_id") or item.get("id")
        if not isinstance(call_id, str) or not call_id:
            raise ResponsesInputError(f"{item_type} requires a non-empty call_id")
        return [{"role": "tool", "tool_call_id": call_id, "content": _as_text(item.get("output"))}]
    if item_type in UNSUPPORTED_ITEM_TYPES:
        raise ResponsesInputError(f"input item type {item_type!r} is not supported: {UNSUPPORTED_ITEM_TYPES[item_type]}")
    role = item.get("role")
    if role is None and item_type == "message":
        role = "assistant"
    if not isinstance(role, str):
        raise ResponsesInputError("input item requires a role or a supported type")
    if role == "developer":
        role = "system"
    if role not in SUPPORTED_ROLES:
        raise ResponsesInputError(f"unsupported role: {role}")
    return [{"role": role, "content": _normalize_content(item.get("content"))}]


def normalize_input(input_value: Any) -> list[dict]:
    if input_value is None:
        return []
    if isinstance(input_value, str):
        if not input_value.strip():
            return []
        return [{"role": "user", "content": input_value}]
    if isinstance(input_value, dict):
        return _normalize_item(input_value)
    if isinstance(input_value, list):
        messages: list[dict] = []
        for item in input_value:
            messages.extend(_normalize_item(item))
        return messages
    raise ResponsesInputError("input must be a string or an array of input items")


def ensure_input_present(messages: Any) -> list[dict]:
    if not isinstance(messages, list) or not messages:
        raise ResponsesInputError("input must contain at least one input item")
    for message in messages:
        if not isinstance(message, dict):
            continue
        if isinstance(message.get("tool_calls"), list) and message["tool_calls"]:
            continue
        if _as_text(message.get("content")).strip():
            continue
        raise ResponsesInputError("input must contain at least one non-empty input item")
    return messages


def validate_tool_chain(messages: Any) -> None:
    if not isinstance(messages, list):
        return
    answered: set[str] = set()
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        if message.get("role") == "assistant":
            for call in message.get("tool_calls") or []:
                if isinstance(call, dict) and isinstance(call.get("id"), str) and call["id"]:
                    answered.add(call["id"])
        elif message.get("role") == "tool":
            call_id = message.get("tool_call_id")
            if not isinstance(call_id, str) or call_id not in answered:
                raise ResponsesInputError(f"messages[{index}] is a tool result for an unknown call_id: {call_id!r}")


def convert_tools(tools: Any) -> list[Any] | None:
    if not isinstance(tools, list):
        return None
    converted: list[dict] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        if isinstance(tool.get("function"), dict):
            converted.append(tool)
            continue
        tool_type = tool.get("type")
        if isinstance(tool.get("name"), str) and tool_type in (None, "function"):
            function: dict[str, Any] = {"name": tool.get("name") or ""}
            if "description" in tool:
                function["description"] = tool["description"]
            if "parameters" in tool:
                function["parameters"] = tool["parameters"]
            if "strict" in tool:
                function["strict"] = tool["strict"]
            converted.append({"type": "function", "function": function})
    return converted or None


def convert_tool_choice(tool_choice: Any) -> Any:
    if tool_choice is None:
        return None
    if isinstance(tool_choice, str):
        return tool_choice
    if isinstance(tool_choice, dict):
        choice_type = tool_choice.get("type")
        if choice_type in ("auto", "none", "required"):
            return choice_type
        name = tool_choice.get("name")
        if not isinstance(name, str) and isinstance(tool_choice.get("function"), dict):
            name = tool_choice["function"].get("name")
        if isinstance(name, str) and name:
            return {"type": "function", "function": {"name": name}}
    return None


def response_text_format(text: Any) -> dict | None:
    if isinstance(text, dict):
        fmt = text.get("format")
        if isinstance(fmt, dict):
            if fmt.get("type") == "json_schema" and isinstance(fmt.get("json_schema"), dict):
                return {"type": "json_schema", "json_schema": dict(fmt["json_schema"])}
            return fmt
        if isinstance(fmt, str):
            return {"type": fmt}
        return None
    if isinstance(text, str):
        return {"type": text}
    return None


def extract_response_format(text: Any, response_format: Any) -> Any:
    fmt = None
    if isinstance(text, dict):
        fmt = text.get("format")
    if fmt is None:
        fmt = response_format
    if fmt is None:
        return None
    if isinstance(fmt, str):
        return fmt
    if not isinstance(fmt, dict):
        return None
    fmt_type = fmt.get("type")
    if fmt_type == "json_object":
        return {"type": "json_object"}
    if fmt_type != "json_schema":
        return None
    schema = fmt.get("schema")
    if schema is None and isinstance(fmt.get("json_schema"), dict):
        nested = fmt["json_schema"]
        return {"type": "json_schema", "json_schema": nested}
    if schema is None:
        return None
    json_schema: dict[str, Any] = {"name": fmt.get("name") or "response", "schema": schema}
    if "strict" in fmt:
        json_schema["strict"] = fmt["strict"]
    if "description" in fmt:
        json_schema["description"] = fmt["description"]
    return {"type": "json_schema", "json_schema": json_schema}


@dataclass
class RequestInfo:
    model: str
    instructions: str | None = None
    max_output_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    tool_choice: Any = None
    tools: list[Any] | None = None
    parallel_tool_calls: bool | None = None
    previous_response_id: str | None = None
    store: bool = True
    metadata: Any = None
    user: str | None = None
    text_format: dict | None = None
    truncation: str = "disabled"
    reasoning: Any = None


def _reasoning_text_from_output(output: Any) -> str:
    parts: list[str] = []
    if not isinstance(output, list):
        return ""
    for item in output:
        if not isinstance(item, dict):
            continue
        if item.get("type") not in ("reasoning", "reasoning_summary", "summary_text"):
            continue
        text = item.get("text")
        if isinstance(text, str) and text:
            parts.append(text)
            continue
        for key in ("summary", "content"):
            content = item.get(key)
            if not isinstance(content, list):
                continue
            for part in content:
                if not isinstance(part, dict):
                    continue
                part_text = part.get("text")
                if isinstance(part_text, str) and part_text:
                    parts.append(part_text)
    return "".join(parts)


def _usage_to_responses(usage: Any, output: Any = None) -> dict | None:
    if not isinstance(usage, dict):
        return None
    input_tokens = _as_int(usage.get("prompt_tokens") or usage.get("input_tokens"))
    output_tokens = _as_int(usage.get("completion_tokens") or usage.get("output_tokens"))
    reasoning_tokens = _as_int(usage.get("reasoning_tokens"))
    if not reasoning_tokens:
        reasoning_tokens = estimate_tokens(_reasoning_text_from_output(output))
    return {
        "input_tokens": input_tokens,
        "input_tokens_details": {"cached_tokens": _as_int(usage.get("cached_tokens"))},
        "output_tokens": output_tokens,
        "output_tokens_details": {"reasoning_tokens": reasoning_tokens},
        "total_tokens": input_tokens + output_tokens,
    }


def _response_error(error: Any) -> dict | None:
    if error is None:
        return None
    if not isinstance(error, dict):
        return {"code": None, "message": "upstream request failed"}
    code = error.get("code")
    message = error.get("message")
    return {
        "code": code if isinstance(code, str) and code else None,
        "message": message if isinstance(message, str) and message else "upstream request failed",
    }


def build_response_object(
    info: RequestInfo,
    response_id: str,
    created_at: int,
    output: list[dict],
    status: str = "completed",
    usage: Any = None,
    error: dict | None = None,
    incomplete_details: dict | None = None,
) -> dict:
    reported_usage = _usage_to_responses(usage, output)
    if reported_usage is None and status not in NON_TERMINAL_STATUSES:
        reported_usage = {
            "input_tokens": 0,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": 0,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 0,
        }
    return {
        "id": response_id,
        "object": "response",
        "created_at": created_at,
        "status": status,
        "error": _response_error(error),
        "incomplete_details": incomplete_details,
        "instructions": info.instructions,
        "max_output_tokens": info.max_output_tokens,
        "model": info.model,
        "output": output,
        "parallel_tool_calls": True if info.parallel_tool_calls is None else info.parallel_tool_calls,
        "previous_response_id": info.previous_response_id,
        "reasoning": info.reasoning if info.reasoning is not None else {"effort": None, "summary": None},
        "store": info.store,
        "temperature": info.temperature,
        "text": {"format": info.text_format or {"type": "text"}},
        "tool_choice": info.tool_choice if info.tool_choice is not None else "auto",
        "tools": info.tools or [],
        "top_p": info.top_p,
        "truncation": info.truncation,
        "usage": reported_usage,
        "user": info.user,
        "metadata": info.metadata if isinstance(info.metadata, dict) else {},
    }


def output_items_from_message(message: Any) -> list[dict]:
    if not isinstance(message, dict):
        return []
    items: list[dict] = []
    reasoning = message.get("reasoning_content")
    if isinstance(reasoning, str) and reasoning:
        items.append(
            {
                "id": f"rs_{uuid.uuid4().hex}",
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": reasoning}],
            }
        )
    content = message.get("content")
    text = _as_text(content)
    tool_calls = message.get("tool_calls")
    has_calls = isinstance(tool_calls, list) and bool(tool_calls)
    if text or not has_calls:
        items.append(
            {
                "id": f"msg_{uuid.uuid4().hex}",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        )
    if isinstance(tool_calls, list):
        for call in tool_calls:
            if not isinstance(call, dict):
                continue
            function = call.get("function")
            if not isinstance(function, dict):
                function = call
            arguments = function.get("arguments")
            if isinstance(arguments, (dict, list)):
                arguments = json.dumps(arguments, ensure_ascii=False)
            items.append(
                {
                    "id": f"fc_{uuid.uuid4().hex}",
                    "type": "function_call",
                    "call_id": call.get("id") or f"call_{uuid.uuid4().hex[:12]}",
                    "name": function.get("name") or "",
                    "arguments": arguments if isinstance(arguments, str) else "{}",
                    "status": "completed",
                }
            )
    return items


def messages_from_output(output: Any) -> list[dict]:
    if not isinstance(output, list):
        return []
    text_parts: list[str] = []
    calls: list[dict] = []
    for item in output:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type == "message":
            content = item.get("content")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and isinstance(part.get("text"), str):
                        text_parts.append(part["text"])
            elif isinstance(content, str):
                text_parts.append(content)
        elif item_type == "function_call":
            arguments = item.get("arguments")
            if isinstance(arguments, (dict, list)):
                arguments = json.dumps(arguments, ensure_ascii=False)
            calls.append(
                {
                    "id": item.get("call_id") or item.get("id") or f"call_{uuid.uuid4().hex[:12]}",
                    "type": "function",
                    "function": {"name": item.get("name") or "", "arguments": arguments if isinstance(arguments, str) else "{}"},
                }
            )
    message: dict[str, Any] = {"role": "assistant", "content": "".join(text_parts)}
    if calls:
        message["tool_calls"] = calls
    if not text_parts and not calls:
        return []
    return [message]


def _stable_id(prefix: str, payload: Any, occurrences: dict[str, int] | None = None) -> str:
    try:
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    except (TypeError, ValueError):
        encoded = str(payload)
    digest = hashlib.blake2s(encoded.encode("utf-8", "replace"), digest_size=12).hexdigest()
    key = f"{prefix}:{digest}"
    seen = 0 if occurrences is None else occurrences.get(key, 0)
    if occurrences is not None:
        occurrences[key] = seen + 1
    return f"{prefix}_{digest}" if seen == 0 else f"{prefix}_{digest}{seen}"


def _input_message_item(message: dict, occurrences: dict[str, int] | None = None) -> list[dict]:
    role = message.get("role") or "user"
    content = message.get("content")
    if role in ("tool", "function"):
        call_id = message.get("tool_call_id")
        if not isinstance(call_id, str) or not call_id:
            call_id = _stable_id("call", {"role": role, "name": message.get("name"), "content": content}, occurrences)
        return [
            {
                "id": _stable_id("fc", {"call_id": call_id, "output": _as_text(content)}, occurrences),
                "type": "function_call_output",
                "call_id": call_id,
                "output": _as_text(content),
            }
        ]
    parts: list[dict] = []
    if isinstance(content, list):
        for part in content:
            if not isinstance(part, dict):
                continue
            part_type = part.get("type")
            if part_type == "text":
                parts.append({"type": "input_text", "text": part.get("text") or "", "annotations": []})
            elif part_type == "image_url":
                url, detail = _image_source(part)
                image_part: dict[str, Any] = {"type": "input_image", "image_url": url}
                valid_detail = _image_detail(detail)
                if valid_detail:
                    image_part["detail"] = valid_detail
                parts.append(image_part)
            elif part_type in ("input_file", "file") and isinstance(part.get("file"), dict):
                parts.append({"type": "input_file", "file": part["file"], "filename": part.get("filename")})
    elif isinstance(content, str) and content:
        parts.append({"type": "input_text", "text": content, "annotations": []})
    if not parts:
        parts.append({"type": "input_text", "text": "", "annotations": []})
    items: list[dict] = [
        {
            "id": _stable_id("msg", {"role": role, "content": parts}, occurrences),
            "type": "message",
            "status": "completed",
            "role": role,
            "content": parts,
        }
    ]
    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list):
        for call in tool_calls:
            if not isinstance(call, dict):
                continue
            function = call.get("function")
            if not isinstance(function, dict):
                function = call
            arguments = function.get("arguments")
            if isinstance(arguments, (dict, list)):
                arguments = json.dumps(arguments, ensure_ascii=False)
            call_id = call.get("id") or f"call_{uuid.uuid4().hex[:12]}"
            items.append(
                {
                    "id": _stable_id("fc", {"call_id": call_id, "arguments": arguments}, occurrences),
                    "type": "function_call",
                    "call_id": call_id,
                    "name": function.get("name") or "",
                    "arguments": arguments if isinstance(arguments, str) else "{}",
                    "status": "completed",
                }
            )
    reasoning = message.get("reasoning_content")
    if isinstance(reasoning, str) and reasoning:
        items.insert(
            0,
            {
                "id": _stable_id("rs", {"reasoning": reasoning}, occurrences),
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": reasoning}],
            },
        )
    return items


def input_items_from_messages(messages: Any) -> list[dict]:
    if not isinstance(messages, list):
        return []
    items: list[dict] = []
    occurrences: dict[str, int] = {}
    for message in messages:
        if isinstance(message, dict):
            items.extend(_input_message_item(message, occurrences))
    return items


INCOMPLETE_REASONS = {
    "length": "max_output_tokens",
    "content_filter": "content_filter",
}


def _error_finish_reason(error: Any) -> str | None:
    if isinstance(error, dict) and error.get("finish_reason") == RESPONSE_INCOMPLETE:
        return RESPONSE_INCOMPLETE
    return None


def response_from_chat(chat: Any, info: RequestInfo, response_id: str, created_at: int) -> dict:
    if not isinstance(chat, dict):
        raise HTTPException(500, "provider returned an invalid response")
    choices = chat.get("choices")
    choice = choices[0] if isinstance(choices, list) and choices else {}
    message = choice.get("message") if isinstance(choice, dict) else None
    output = output_items_from_message(message)
    finish = choice.get("finish_reason") if isinstance(choice, dict) else None
    error = chat.get("error") if isinstance(chat.get("error"), dict) else None
    status = "completed"
    incomplete_details = None
    if _error_finish_reason(error) is not None or finish == RESPONSE_INCOMPLETE:
        status = "incomplete"
        incomplete_details = {"reason": REDUCED_CONTEXT_REASON}
    elif error is not None:
        status = "failed"
    elif isinstance(finish, str) and finish in INCOMPLETE_REASONS:
        status = "incomplete"
        incomplete_details = {"reason": INCOMPLETE_REASONS[finish]}
    return build_response_object(
        info,
        response_id,
        created_at,
        output=output,
        status=status,
        usage=chat.get("usage"),
        error=error,
        incomplete_details=incomplete_details,
    )


def _error_payload(error: Any) -> dict:
    if isinstance(error, dict):
        code = error.get("code")
        message = error.get("message")
        return {
            "type": error.get("type") or "server_error",
            "code": code if isinstance(code, str) and code else None,
            "message": message if isinstance(message, str) and message else "stream error",
            "param": error.get("param"),
        }
    log.warning("upstream stream produced a non-dict error: %s", type(error).__name__)
    return {"type": "server_error", "code": None, "message": "stream error", "param": None}


def _error_event(error: Any) -> dict:
    payload = _error_payload(error)
    del payload["type"]
    return payload


class _StreamState:
    __slots__ = (
        "active_choice",
        "created_at",
        "finish",
        "finishes",
        "info",
        "message_id",
        "message_index",
        "message_open",
        "message_parts",
        "on_complete",
        "output",
        "output_index",
        "reasoning_id",
        "reasoning_index",
        "reasoning_open",
        "reasoning_parts",
        "response_id",
        "sequence",
        "tool_items",
        "usage",
    )

    def __init__(self, info: RequestInfo, response_id: str, created_at: int, on_complete: Any = None) -> None:
        self.info = info
        self.response_id = response_id
        self.created_at = created_at
        self.on_complete = on_complete
        self.sequence = 0
        self.output_index = 0
        self.output: dict[int, dict] = {}
        self.reasoning_id: str | None = None
        self.reasoning_index: int | None = None
        self.reasoning_parts: list[str] = []
        self.reasoning_open = False
        self.message_id: str | None = None
        self.message_index: int | None = None
        self.message_parts: list[str] = []
        self.message_open = False
        self.tool_items: dict[int, dict] = {}
        self.usage: Any = None
        self.finish: str | None = None
        self.finishes: dict[int, str] = {}
        self.active_choice: int | None = None

    def final_finish(self) -> str | None:
        if self.active_choice is not None and self.active_choice in self.finishes:
            return self.finishes[self.active_choice]
        for position in sorted(self.finishes):
            return self.finishes[position]
        return None

    def next_sequence(self) -> int:
        value = self.sequence
        self.sequence += 1
        return value

    def emit(self, event_type: str, data: dict) -> str:
        payload = {"type": event_type, "sequence_number": self.next_sequence()}
        payload.update(data)
        return sse_event(event_type, payload)

    def reasoning_delta(self, text: str) -> Iterator[str]:
        yield from self.close_message()
        if not self.reasoning_open:
            self.reasoning_open = True
            self.reasoning_id = f"rs_{uuid.uuid4().hex}"
            self.reasoning_index = self.output_index
            self.output_index += 1
            item = {"id": self.reasoning_id, "type": "reasoning", "summary": []}
            yield self.emit("response.output_item.added", {"output_index": self.reasoning_index, "item": item})
            part = {"type": "summary_text", "text": ""}
            yield self.emit(
                "response.reasoning_summary_part.added",
                {"item_id": self.reasoning_id, "output_index": self.reasoning_index, "summary_index": 0, "part": part},
            )
        self.reasoning_parts.append(text)
        yield self.emit(
            "response.reasoning_summary_text.delta",
            {"item_id": self.reasoning_id, "output_index": self.reasoning_index, "summary_index": 0, "delta": text},
        )

    def close_reasoning(self) -> Iterator[str]:
        if not self.reasoning_open:
            return
        self.reasoning_open = False
        text = "".join(self.reasoning_parts)
        yield self.emit(
            "response.reasoning_summary_text.done",
            {"item_id": self.reasoning_id, "output_index": self.reasoning_index, "summary_index": 0, "text": text},
        )
        part = {"type": "summary_text", "text": text}
        yield self.emit(
            "response.reasoning_summary_part.done",
            {"item_id": self.reasoning_id, "output_index": self.reasoning_index, "summary_index": 0, "part": part},
        )
        item = {"id": self.reasoning_id, "type": "reasoning", "summary": [part]}
        self.output[self.reasoning_index or 0] = item
        yield self.emit("response.output_item.done", {"output_index": self.reasoning_index, "item": item})

    def message_delta(self, text: str) -> Iterator[str]:
        yield from self.close_reasoning()
        if not self.message_open:
            self.message_open = True
            self.message_id = f"msg_{uuid.uuid4().hex}"
            self.message_index = self.output_index
            self.output_index += 1
            item = {"id": self.message_id, "type": "message", "status": "in_progress", "role": "assistant", "content": []}
            yield self.emit("response.output_item.added", {"output_index": self.message_index, "item": item})
            part = {"type": "output_text", "text": "", "annotations": []}
            yield self.emit(
                "response.content_part.added",
                {"item_id": self.message_id, "output_index": self.message_index, "content_index": 0, "part": part},
            )
        self.message_parts.append(text)
        yield self.emit(
            "response.output_text.delta",
            {"item_id": self.message_id, "output_index": self.message_index, "content_index": 0, "delta": text},
        )

    def close_message(self) -> Iterator[str]:
        if not self.message_open:
            return
        self.message_open = False
        text = "".join(self.message_parts)
        yield self.emit(
            "response.output_text.done",
            {"item_id": self.message_id, "output_index": self.message_index, "content_index": 0, "text": text},
        )
        part = {"type": "output_text", "text": text, "annotations": []}
        yield self.emit(
            "response.content_part.done",
            {"item_id": self.message_id, "output_index": self.message_index, "content_index": 0, "part": part},
        )
        item = {"id": self.message_id, "type": "message", "status": "completed", "role": "assistant", "content": [part]}
        self.output[self.message_index or 0] = item
        yield self.emit("response.output_item.done", {"output_index": self.message_index, "item": item})

    def _open_tool_item(self, entry: dict) -> Iterator[str]:
        if entry["added"]:
            return
        item = {
            "id": entry["id"],
            "type": "function_call",
            "call_id": entry["call_id"],
            "name": entry["name"],
            "arguments": "",
            "status": "in_progress",
        }
        yield self.emit("response.output_item.added", {"output_index": entry["output_index"], "item": item})
        entry["added"] = True
        buffered = entry["buffered_args"]
        if buffered:
            entry["buffered_args"] = []
            for chunk in buffered:
                yield self.emit(
                    "response.function_call_arguments.delta",
                    {"item_id": entry["id"], "output_index": entry["output_index"], "delta": chunk},
                )

    def tool_delta(self, tool_calls: Any) -> Iterator[str]:
        yield from self.close_message()
        yield from self.close_reasoning()
        if not isinstance(tool_calls, list):
            return
        for call in tool_calls:
            if not isinstance(call, dict):
                continue
            index = call.get("index")
            if isinstance(index, bool) or not isinstance(index, int) or index < 0:
                index = 0
            entry = self.tool_items.get(index)
            function = call.get("function")
            if not isinstance(function, dict):
                function = {}
            if entry is None:
                entry = {
                    "id": f"fc_{uuid.uuid4().hex}",
                    "call_id": call.get("id") or f"call_{uuid.uuid4().hex[:12]}",
                    "name": "",
                    "arguments": "",
                    "output_index": self.output_index,
                    "open": True,
                    "added": False,
                    "buffered_args": [],
                }
                self.output_index += 1
                self.tool_items[index] = entry
            name = function.get("name")
            if isinstance(name, str) and name and not entry["name"]:
                entry["name"] = name
            arguments = function.get("arguments")
            if isinstance(arguments, str) and arguments:
                entry["arguments"] += arguments
                if entry["added"]:
                    yield self.emit(
                        "response.function_call_arguments.delta",
                        {"item_id": entry["id"], "output_index": entry["output_index"], "delta": arguments},
                    )
                else:
                    entry["buffered_args"].append(arguments)
            if entry["name"]:
                yield from self._open_tool_item(entry)

    def _close_tool(self, entry: dict) -> Iterator[str]:
        if not entry.get("open"):
            return
        entry["open"] = False
        yield from self._open_tool_item(entry)
        yield self.emit(
            "response.function_call_arguments.done",
            {"item_id": entry["id"], "output_index": entry["output_index"], "arguments": entry["arguments"]},
        )
        item = {
            "id": entry["id"],
            "type": "function_call",
            "call_id": entry["call_id"],
            "name": entry["name"],
            "arguments": entry["arguments"],
            "status": "completed",
        }
        self.output[entry["output_index"]] = item
        yield self.emit("response.output_item.done", {"output_index": entry["output_index"], "item": item})

    def close_tools(self) -> Iterator[str]:
        for index in sorted(self.tool_items, key=lambda key: self.tool_items[key]["output_index"]):
            yield from self._close_tool(self.tool_items[index])

    def _open_closers(self) -> list[tuple[int, Any]]:
        closers: list[tuple[int, Any]] = []
        if self.reasoning_open and self.reasoning_index is not None:
            closers.append((self.reasoning_index, self.close_reasoning))
        if self.message_open and self.message_index is not None:
            closers.append((self.message_index, self.close_message))
        for entry in self.tool_items.values():
            if entry.get("open"):
                closers.append((entry["output_index"], partial(self._close_tool, entry)))
        closers.sort(key=lambda item: item[0])
        return closers

    def ordered_output(self) -> list[dict]:
        return [self.output[index] for index in sorted(self.output)]

    def close_all(self) -> Iterator[str]:
        for _index, close in self._open_closers():
            yield from close()


async def _close_chat_stream(chat_stream: Any) -> None:
    closer = getattr(chat_stream, "aclose", None)
    if closer is None:
        return
    try:
        await closer()
    except Exception as exc:
        log.debug("upstream chat stream close failed: %s", exc)


async def _terminal_lines(
    state: _StreamState,
    status: str,
    error: dict | None = None,
    incomplete_details: dict | None = None,
) -> AsyncIterator[str]:
    final = build_response_object(
        state.info,
        state.response_id,
        state.created_at,
        output=state.ordered_output(),
        status=status,
        usage=state.usage,
        error=error,
        incomplete_details=incomplete_details,
    )
    await _notify_complete(state, final)
    yield state.emit(f"response.{status}", {"response": final})


async def _failure_lines(state: _StreamState, error: dict) -> AsyncIterator[str]:
    if _error_finish_reason(error) is not None:
        async for line in _terminal_lines(state, "incomplete", None, {"reason": REDUCED_CONTEXT_REASON}):
            yield line
        return
    async for line in _terminal_lines(state, "failed", _error_payload(error)):
        yield line
    payload = {"type": "error", "sequence_number": state.next_sequence()}
    payload.update(_error_event(error))
    yield sse_event("error", payload)


async def _notify_complete(state: _StreamState, final: dict) -> None:
    callback = state.on_complete
    if not callable(callback):
        return
    try:
        outcome = callback(final)
        if inspect.isawaitable(outcome):
            await outcome
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        log.warning("response %s could not be persisted: %s: %s", state.response_id, type(exc).__name__, exc)


def _merge_usage(state: _StreamState, usage: Any) -> None:
    if not isinstance(usage, dict):
        return
    if state.usage is None:
        state.usage = {}
    for key, value in usage.items():
        if value is None:
            continue
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)) and value == 0:
            continue
        if isinstance(value, (dict, list, str, tuple, set)) and not value:
            continue
        state.usage[key] = value


async def translate_stream(
    chat_stream: AsyncIterator[str],
    info: RequestInfo,
    response_id: str,
    created_at: int,
    on_complete: Any = None,
) -> AsyncIterator[str]:
    state = _StreamState(info, response_id, created_at, on_complete)
    try:
        initial = build_response_object(info, response_id, created_at, output=[], status="in_progress")
        yield state.emit("response.created", {"response": initial})
        yield state.emit("response.in_progress", {"response": initial})
        error: dict | None = None
        try:
            async for chunk in chat_stream:
                for payload in _iter_sse_payloads(chunk):
                    if payload is None:
                        continue
                    if isinstance(payload.get("error"), dict):
                        error = payload["error"]
                        continue
                    _merge_usage(state, payload.get("usage"))
                    choices = payload.get("choices")
                    if not isinstance(choices, list):
                        continue
                    for position, choice in enumerate(choices):
                        if not isinstance(choice, dict):
                            continue
                        delta = choice.get("delta")
                        if isinstance(delta, dict):
                            reasoning = delta.get("reasoning_content")
                            if isinstance(reasoning, str) and reasoning:
                                if state.active_choice is None:
                                    state.active_choice = position
                                for line in state.reasoning_delta(reasoning):
                                    yield line
                            content = delta.get("content")
                            if isinstance(content, str) and content:
                                if state.active_choice is None:
                                    state.active_choice = position
                                for line in state.message_delta(content):
                                    yield line
                            if delta.get("tool_calls"):
                                if state.active_choice is None:
                                    state.active_choice = position
                                for line in state.tool_delta(delta["tool_calls"]):
                                    yield line
                        finish = choice.get("finish_reason")
                        if isinstance(finish, str) and finish and position not in state.finishes:
                            state.finishes[position] = finish
        except (GeneratorExit, asyncio.CancelledError):
            raise
        except BaseException as exc:
            log.warning("responses stream aborted by the upstream generator: %s", exc)
            for line in state.close_all():
                yield line
            async for line in _failure_lines(state, {"message": "upstream stream failed"}):
                yield line
            raise
        for line in state.close_all():
            yield line
        if error is not None:
            async for line in _failure_lines(state, error):
                yield line
            return
        state.finish = state.final_finish()
        if state.finish in INCOMPLETE_REASONS:
            status = "incomplete"
            incomplete_details: dict | None = {"reason": INCOMPLETE_REASONS[state.finish]}
        else:
            status = "completed"
            incomplete_details = None
        async for line in _terminal_lines(state, status, None, incomplete_details):
            yield line
    finally:
        await _close_chat_stream(chat_stream)
