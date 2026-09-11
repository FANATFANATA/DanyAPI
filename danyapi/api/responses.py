from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from typing import Any


class ResponsesInputError(ValueError):
    pass


TEXT_PART_TYPES = {"input_text", "output_text", "text", "summary_text"}
IMAGE_PART_TYPES = {"input_image", "image_url"}
SUPPORTED_ROLES = {"user", "assistant", "system", "developer", "tool", "function"}


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
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
                parts.append(_as_text(item))
        return "".join(parts)
    if isinstance(value, dict):
        if isinstance(value.get("text"), str):
            return value["text"]
        return json.dumps(value, ensure_ascii=False)
    return str(value)


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
            image_url = item.get("image_url")
            if isinstance(image_url, dict):
                image_url = image_url.get("url")
            if not isinstance(image_url, str) or not image_url:
                raise ResponsesInputError("input_image requires an image_url string")
            parts.append({"type": "image_url", "image_url": image_url})
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
        call_id = item.get("call_id") or item.get("id") or ""
        return [{"role": "tool", "tool_call_id": call_id, "content": _as_text(item.get("output"))}]
    if item_type in ("reasoning", "item_reference", "web_search_call", "file_search_call", "code_interpreter_call"):
        return []
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
        return [{"role": "user", "content": input_value}]
    if isinstance(input_value, dict):
        return _normalize_item(input_value)
    if isinstance(input_value, list):
        messages: list[dict] = []
        for item in input_value:
            messages.extend(_normalize_item(item))
        return messages
    raise ResponsesInputError("input must be a string or an array of input items")


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
        if tool_type == "function" or (isinstance(tool.get("name"), str) and tool_type in (None, "function")):
            function: dict[str, Any] = {"name": tool.get("name")}
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
    extras: dict[str, Any] = field(default_factory=dict)


def _usage_to_responses(usage: Any) -> dict | None:
    if not isinstance(usage, dict):
        return None
    input_tokens = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
    output_tokens = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
    total_tokens = int(usage.get("total_tokens") or (input_tokens + output_tokens))
    return {
        "input_tokens": input_tokens,
        "input_tokens_details": {"cached_tokens": int(usage.get("cached_tokens") or 0)},
        "output_tokens": output_tokens,
        "output_tokens_details": {"reasoning_tokens": int(usage.get("reasoning_tokens") or 0)},
        "total_tokens": total_tokens,
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
    return {
        "id": response_id,
        "object": "response",
        "created_at": created_at,
        "status": status,
        "error": error,
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
        "usage": _usage_to_responses(usage),
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
            calls.append(
                {
                    "id": item.get("call_id") or item.get("id") or f"call_{uuid.uuid4().hex[:12]}",
                    "type": "function",
                    "function": {"name": item.get("name") or "", "arguments": item.get("arguments") or "{}"},
                }
            )
    messages: list[dict] = []
    text = "".join(text_parts)
    if text or not calls:
        messages.append({"role": "assistant", "content": text})
    if calls:
        messages.append({"role": "assistant", "content": "", "tool_calls": calls})
    return messages


INCOMPLETE_REASONS = {
    "length": "max_output_tokens",
    "content_filter": "content_filter",
}


def response_from_chat(chat: Any, info: RequestInfo, response_id: str, created_at: int) -> dict:
    if not isinstance(chat, dict):
        raise ResponsesInputError("provider returned an invalid response")
    choices = chat.get("choices")
    choice = choices[0] if isinstance(choices, list) and choices else {}
    message = choice.get("message") if isinstance(choice, dict) else None
    output = output_items_from_message(message)
    finish = choice.get("finish_reason") if isinstance(choice, dict) else None
    error = chat.get("error") if isinstance(chat.get("error"), dict) else None
    status = "completed"
    incomplete_details = None
    if error is not None or finish == "response_incomplete":
        status = "incomplete"
    elif finish in INCOMPLETE_REASONS:
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


def sse_event(event_type: str, data: dict) -> str:
    return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _iter_sse_payloads(chunk: Any) -> Iterator[dict | None]:
    if not isinstance(chunk, str):
        return
    for block in chunk.split("\n\n"):
        data_line = None
        for line in block.splitlines():
            if line.startswith("data:"):
                data_line = line[5:].strip()
        if data_line is None:
            continue
        if data_line == "[DONE]":
            yield None
            continue
        try:
            payload = json.loads(data_line)
        except (ValueError, TypeError):
            continue
        if isinstance(payload, dict):
            yield payload


def _error_payload(error: Any) -> dict:
    if isinstance(error, dict):
        return {
            "type": error.get("type") or "server_error",
            "code": error.get("code") or error.get("finish_reason"),
            "message": error.get("message") or "stream error",
            "param": error.get("param"),
        }
    return {"type": "server_error", "code": None, "message": str(error), "param": None}


class _StreamState:
    def __init__(self, info: RequestInfo, response_id: str, created_at: int) -> None:
        self.info = info
        self.response_id = response_id
        self.created_at = created_at
        self.sequence = 0
        self.output_index = 0
        self.output: list[dict] = []
        self.reasoning_id: str | None = None
        self.reasoning_index: int | None = None
        self.reasoning_text = ""
        self.reasoning_open = False
        self.message_id: str | None = None
        self.message_index: int | None = None
        self.message_text = ""
        self.message_open = False
        self.tool_items: dict[int, dict] = {}
        self.usage: Any = None
        self.finish: str | None = None

    def next_sequence(self) -> int:
        value = self.sequence
        self.sequence += 1
        return value

    def emit(self, event_type: str, data: dict) -> str:
        payload = {"type": event_type, "sequence_number": self.next_sequence()}
        payload.update(data)
        return sse_event(event_type, payload)

    def reasoning_delta(self, text: str) -> Iterator[str]:
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
        self.reasoning_text += text
        yield self.emit(
            "response.reasoning_summary_text.delta",
            {"item_id": self.reasoning_id, "output_index": self.reasoning_index, "summary_index": 0, "delta": text},
        )

    def close_reasoning(self) -> Iterator[str]:
        if not self.reasoning_open:
            return
        self.reasoning_open = False
        yield self.emit(
            "response.reasoning_summary_text.done",
            {"item_id": self.reasoning_id, "output_index": self.reasoning_index, "summary_index": 0, "text": self.reasoning_text},
        )
        part = {"type": "summary_text", "text": self.reasoning_text}
        yield self.emit(
            "response.reasoning_summary_part.done",
            {"item_id": self.reasoning_id, "output_index": self.reasoning_index, "summary_index": 0, "part": part},
        )
        item = {"id": self.reasoning_id, "type": "reasoning", "summary": [part]}
        self.output.append(item)
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
        self.message_text += text
        yield self.emit(
            "response.output_text.delta",
            {"item_id": self.message_id, "output_index": self.message_index, "content_index": 0, "delta": text},
        )

    def close_message(self) -> Iterator[str]:
        if not self.message_open:
            return
        self.message_open = False
        yield self.emit(
            "response.output_text.done",
            {"item_id": self.message_id, "output_index": self.message_index, "content_index": 0, "text": self.message_text},
        )
        part = {"type": "output_text", "text": self.message_text, "annotations": []}
        yield self.emit(
            "response.content_part.done",
            {"item_id": self.message_id, "output_index": self.message_index, "content_index": 0, "part": part},
        )
        item = {"id": self.message_id, "type": "message", "status": "completed", "role": "assistant", "content": [part]}
        self.output.append(item)
        yield self.emit("response.output_item.done", {"output_index": self.message_index, "item": item})

    def tool_delta(self, tool_calls: Any) -> Iterator[str]:
        yield from self.close_message()
        yield from self.close_reasoning()
        if not isinstance(tool_calls, list):
            return
        for call in tool_calls:
            if not isinstance(call, dict):
                continue
            index = call.get("index")
            if not isinstance(index, int):
                index = 0
            entry = self.tool_items.get(index)
            function = call.get("function")
            if not isinstance(function, dict):
                function = {}
            if entry is None:
                entry = {
                    "id": f"fc_{uuid.uuid4().hex}",
                    "call_id": call.get("id") or f"call_{uuid.uuid4().hex[:12]}",
                    "name": function.get("name") or "",
                    "arguments": "",
                    "output_index": self.output_index,
                    "open": True,
                }
                self.output_index += 1
                self.tool_items[index] = entry
                item = {
                    "id": entry["id"],
                    "type": "function_call",
                    "call_id": entry["call_id"],
                    "name": entry["name"],
                    "arguments": "",
                    "status": "in_progress",
                }
                yield self.emit("response.output_item.added", {"output_index": entry["output_index"], "item": item})
            if function.get("name") and not entry["name"]:
                entry["name"] = function["name"]
            arguments = function.get("arguments")
            if isinstance(arguments, str) and arguments:
                entry["arguments"] += arguments
                yield self.emit(
                    "response.function_call_arguments.delta",
                    {"item_id": entry["id"], "output_index": entry["output_index"], "delta": arguments},
                )

    def close_tools(self) -> Iterator[str]:
        for index in sorted(self.tool_items):
            entry = self.tool_items[index]
            if not entry.get("open"):
                continue
            entry["open"] = False
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
            self.output.append(item)
            yield self.emit("response.output_item.done", {"output_index": entry["output_index"], "item": item})

    def close_all(self) -> Iterator[str]:
        yield from self.close_reasoning()
        yield from self.close_message()
        yield from self.close_tools()


async def translate_stream(
    chat_stream: AsyncIterator[str],
    info: RequestInfo,
    response_id: str,
    created_at: int,
    on_complete: Any = None,
) -> AsyncIterator[str]:
    state = _StreamState(info, response_id, created_at)
    initial = build_response_object(info, response_id, created_at, output=[], status="in_progress")
    yield state.emit("response.created", {"response": initial})
    yield state.emit("response.in_progress", {"response": initial})
    error: Any = None
    async for chunk in chat_stream:
        for payload in _iter_sse_payloads(chunk):
            if payload is None:
                continue
            if isinstance(payload.get("error"), dict):
                error = payload["error"]
                continue
            usage = payload.get("usage")
            if isinstance(usage, dict):
                state.usage = usage
            choices = payload.get("choices")
            if not isinstance(choices, list):
                continue
            for choice in choices:
                if not isinstance(choice, dict):
                    continue
                delta = choice.get("delta")
                if isinstance(delta, dict):
                    reasoning = delta.get("reasoning_content")
                    if isinstance(reasoning, str) and reasoning:
                        for line in state.reasoning_delta(reasoning):
                            yield line
                    content = delta.get("content")
                    if isinstance(content, str) and content:
                        for line in state.message_delta(content):
                            yield line
                    if delta.get("tool_calls"):
                        for line in state.tool_delta(delta["tool_calls"]):
                            yield line
                finish = choice.get("finish_reason")
                if isinstance(finish, str) and finish:
                    state.finish = finish
    for line in state.close_all():
        yield line
    if error is not None:
        payload = {"type": "error", "sequence_number": state.next_sequence()}
        payload.update(_error_payload(error))
        yield sse_event("error", payload)
        return
    final = build_response_object(info, response_id, created_at, output=state.output, status="completed", usage=state.usage)
    if callable(on_complete):
        on_complete(final)
    yield state.emit("response.completed", {"response": final})
