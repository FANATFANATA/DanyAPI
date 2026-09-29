from __future__ import annotations

import asyncio
import inspect
import json
import logging
import uuid
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from typing import Any

from danyapi.sseutil import parse_sse
from danyapi.tokens import count_messages_tokens, estimate_tokens

log = logging.getLogger("danyapi.api.anthropic")

DEFAULT_MAX_TOKENS = 4096
MAX_TEXT_DEPTH = 8

TEXT_BLOCK_TYPES = {"text", "input_text", "output_text"}
IMAGE_BLOCK_TYPES = {"image", "image_url", "input_image"}
SUPPORTED_ROLES = {"user", "assistant"}
IGNORED_BLOCK_TYPES = {
    "thinking",
    "redacted_thinking",
    "document",
    "search_result",
    "server_tool_use",
    "web_search_tool_result",
    "mcp_tool_use",
    "mcp_tool_result",
    "code_execution_tool_result",
}
TOOL_ERROR_PREFIX = "[tool_error] "

STOP_REASONS = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
    "content_filter": "end_turn",
    "response_incomplete": "max_tokens",
    "stop_sequence": "stop_sequence",
    "error": "end_turn",
}
TRUNCATED_REASONS = {"length", "response_incomplete"}
TOOL_CALL_REASONS = {"tool_calls", "function_call"}


class AnthropicInputError(ValueError):
    pass


@dataclass
class RequestInfo:
    model: str
    max_tokens: int
    prompt_tokens: int = 0
    metadata: Any = None
    stop_sequences: list[str] | None = None


def sse_event(event_type: str, data: dict) -> str:
    return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def error_body(error_type: str, message: str) -> dict:
    return {"type": "error", "error": {"type": error_type, "message": message}}


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
                if isinstance(item.get("text"), str):
                    parts.append(item["text"])
                else:
                    parts.append(_as_text(item.get("content"), depth + 1))
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


def match_stop_sequence(text: str, stop_sequences: Any) -> str | None:
    if not isinstance(stop_sequences, list) or not text:
        return None
    for sequence in stop_sequences:
        if isinstance(sequence, str) and sequence and sequence in text:
            return sequence
    return None


def _image_part(block: dict) -> dict[str, Any]:
    source = block.get("source")
    if isinstance(source, dict):
        source_type = source.get("type")
        if source_type == "base64":
            media_type = source.get("media_type") or "image/png"
            data = source.get("data")
            if not isinstance(data, str) or not data:
                raise AnthropicInputError("image source requires base64 data")
            return {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{data}"}}
        url = source.get("url")
        if isinstance(url, str) and url:
            return {"type": "image_url", "image_url": {"url": url}}
    url = block.get("url") or block.get("image_url")
    if isinstance(url, dict):
        url = url.get("url")
    if isinstance(url, str) and url:
        return {"type": "image_url", "image_url": {"url": url}}
    raise AnthropicInputError("image block requires a source with base64 data or a url")


def _tool_use_block(block: dict) -> dict[str, Any]:
    payload = block.get("input")
    arguments = "{}"
    if isinstance(payload, (dict, list)):
        try:
            arguments = json.dumps(payload, ensure_ascii=False)
        except (TypeError, ValueError):
            arguments = "{}"
    elif isinstance(payload, str):
        arguments = payload
    return {
        "id": block.get("id") or f"call_{uuid.uuid4().hex[:12]}",
        "type": "function",
        "function": {"name": block.get("name") or "", "arguments": arguments},
    }


def _tool_result_block(block: dict) -> dict[str, Any]:
    text = _as_text(block.get("content"))
    if block.get("is_error") is True:
        text = f"{TOOL_ERROR_PREFIX}{text}" if text else TOOL_ERROR_PREFIX.strip()
    return {
        "role": "tool",
        "tool_call_id": block.get("tool_use_id") or block.get("id") or "",
        "content": text,
    }


def _content_blocks(content: Any) -> tuple[str | list[dict], list[dict], list[dict]]:
    if content is None:
        return "", [], []
    if isinstance(content, str):
        return content, [], []
    if not isinstance(content, list):
        return _as_text(content), [], []
    parts: list[dict] = []
    tool_calls: list[dict] = []
    tool_results: list[dict] = []
    for block in content:
        if isinstance(block, str):
            parts.append({"type": "text", "text": block})
            continue
        if not isinstance(block, dict):
            parts.append({"type": "text", "text": _as_text(block)})
            continue
        block_type = block.get("type")
        if block_type in TEXT_BLOCK_TYPES or (block_type is None and isinstance(block.get("text"), str)):
            parts.append({"type": "text", "text": _as_text(block.get("text"))})
        elif block_type in IMAGE_BLOCK_TYPES:
            parts.append(_image_part(block))
        elif block_type == "tool_use":
            tool_calls.append(_tool_use_block(block))
        elif block_type == "tool_result":
            tool_results.append(_tool_result_block(block))
        elif block_type in IGNORED_BLOCK_TYPES:
            continue
        elif isinstance(block.get("text"), str):
            parts.append({"type": "text", "text": block["text"]})
        else:
            raise AnthropicInputError(f"unsupported content block type: {block_type!r}")
    if not parts:
        return "", tool_calls, tool_results
    if all(part.get("type") == "text" for part in parts):
        return "".join(part["text"] for part in parts), tool_calls, tool_results
    return parts, tool_calls, tool_results


def normalize_system(system: Any) -> str | None:
    if system is None:
        return None
    if isinstance(system, str):
        return system or None
    if isinstance(system, list):
        parts: list[str] = []
        for block in system:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                if isinstance(block.get("text"), str):
                    parts.append(block["text"])
                elif block.get("type") == "text":
                    parts.append(_as_text(block.get("text")))
        return "".join(parts) or None
    text = _as_text(system)
    return text or None


def normalize_messages(messages: Any) -> list[dict]:
    if messages is None:
        return []
    if not isinstance(messages, list):
        raise AnthropicInputError("messages must be an array of message objects")
    if not messages:
        raise AnthropicInputError("messages must contain at least one message")
    normalized: list[dict] = []
    for message in messages:
        if not isinstance(message, dict):
            raise AnthropicInputError("each message must be an object")
        role = message.get("role")
        if role not in SUPPORTED_ROLES:
            raise AnthropicInputError(f"unsupported role: {role!r}; anthropic messages accept only user and assistant")
        content, tool_calls, tool_results = _content_blocks(message.get("content"))
        for result in tool_results:
            normalized.append(result)
        if tool_calls:
            entry: dict[str, Any] = {"role": "assistant", "content": content}
            entry["tool_calls"] = tool_calls
            normalized.append(entry)
            continue
        if content == "":
            continue
        normalized.append({"role": role, "content": content})
    if not normalized:
        raise AnthropicInputError("messages must contain at least one message with content")
    return normalized


def convert_tools(tools: Any) -> list[Any] | None:
    if not isinstance(tools, list):
        return None
    converted: list[dict] = []
    for index, tool in enumerate(tools):
        if not isinstance(tool, dict):
            raise AnthropicInputError(f"tools[{index}] must be an object")
        if isinstance(tool.get("function"), dict):
            converted.append(tool)
            continue
        name = tool.get("name")
        if not isinstance(name, str) or not name:
            raise AnthropicInputError(f"tools[{index}] requires a non-empty string name")
        function: dict[str, Any] = {"name": name}
        if isinstance(tool.get("description"), str):
            function["description"] = tool["description"]
        schema = tool.get("input_schema")
        if isinstance(schema, dict):
            function["parameters"] = schema
        converted.append({"type": "function", "function": function})
    return converted or None


def convert_tool_choice(tool_choice: Any) -> Any:
    if tool_choice is None:
        return None
    if isinstance(tool_choice, str):
        if tool_choice in ("auto", "none", "required"):
            return tool_choice
        return {"type": "function", "function": {"name": tool_choice}}
    if not isinstance(tool_choice, dict):
        return None
    choice_type = tool_choice.get("type")
    if choice_type in ("auto", "none"):
        return choice_type
    if choice_type == "any":
        return "required"
    if choice_type == "tool":
        name = tool_choice.get("name")
        if isinstance(name, str) and name:
            return {"type": "function", "function": {"name": name}}
        return None
    if choice_type == "function":
        function = tool_choice.get("function")
        if isinstance(function, dict) and isinstance(function.get("name"), str):
            return {"type": "function", "function": {"name": function["name"]}}
    return None


def convert_stop_sequences(stop_sequences: Any) -> Any:
    if stop_sequences is None:
        return None
    if isinstance(stop_sequences, str):
        return [stop_sequences]
    if not isinstance(stop_sequences, list):
        raise AnthropicInputError("stop_sequences must be an array of strings")
    values: list[str] = []
    for index, item in enumerate(stop_sequences):
        if not isinstance(item, str) or not item:
            raise AnthropicInputError(f"stop_sequences[{index}] must be a non-empty string")
        values.append(item)
    return values or None


def as_max_tokens(value: Any) -> int:
    if value is None:
        return DEFAULT_MAX_TOKENS
    if isinstance(value, bool) or not isinstance(value, int):
        raise AnthropicInputError("max_tokens must be an integer")
    if value <= 0:
        raise AnthropicInputError("max_tokens must be greater than 0")
    return value


def _as_ratio(value: Any, field: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AnthropicInputError(f"{field} must be a number")
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise AnthropicInputError(f"{field} must be between 0 and 1")
    return number


def _as_top_k(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AnthropicInputError("top_k must be an integer")
    if isinstance(value, float) and not value.is_integer():
        raise AnthropicInputError("top_k must be an integer")
    number = int(value)
    if number < 1:
        raise AnthropicInputError("top_k must be greater than 0")
    return number


def build_chat_request(body: dict[str, Any], model: str, session_id: str | None = None) -> dict[str, Any]:
    system = normalize_system(body.get("system"))
    messages = normalize_messages(body.get("messages"))
    tools = convert_tools(body.get("tools"))
    tool_choice = convert_tool_choice(body.get("tool_choice"))
    stop = convert_stop_sequences(body.get("stop_sequences"))
    prefix = [{"role": "system", "content": system}] if system else []
    payload: dict[str, Any] = {"model": model, "messages": [*prefix, *messages]}
    payload["stream"] = bool(body.get("stream"))
    payload["max_tokens"] = as_max_tokens(body.get("max_tokens"))
    if tools:
        payload["tools"] = tools
    if tool_choice is not None:
        payload["tool_choice"] = tool_choice
    if stop:
        payload["stop"] = stop
    temperature = _as_ratio(body.get("temperature"), "temperature")
    if temperature is not None:
        payload["temperature"] = temperature
    top_p = _as_ratio(body.get("top_p"), "top_p")
    if top_p is not None:
        payload["top_p"] = top_p
    top_k = _as_top_k(body.get("top_k"))
    if top_k is not None:
        payload["top_k"] = top_k
    metadata = body.get("metadata")
    if isinstance(metadata, dict):
        user_id = metadata.get("user_id")
        if isinstance(user_id, str) and user_id:
            payload["user"] = user_id
    if session_id:
        payload["session_id"] = session_id
    return payload


def _decode_arguments(raw: Any) -> Any:
    if isinstance(raw, (dict, list)):
        return raw
    if not isinstance(raw, str) or not raw.strip():
        return {}
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return {}


def _usage(usage: Any) -> dict[str, int]:
    if not isinstance(usage, dict):
        return {"input_tokens": 0, "output_tokens": 0}
    prompt = usage.get("prompt_tokens")
    completion = usage.get("completion_tokens")
    if not isinstance(prompt, int):
        prompt = usage.get("input_tokens")
    if not isinstance(completion, int):
        completion = usage.get("output_tokens")
    return {
        "input_tokens": prompt if isinstance(prompt, int) and prompt > 0 else 0,
        "output_tokens": completion if isinstance(completion, int) and completion > 0 else 0,
    }


def stop_reason_for(finish: Any) -> str:
    if not isinstance(finish, str) or not finish:
        return "end_turn"
    return STOP_REASONS.get(finish, "end_turn")


def build_content(choice: dict) -> list[dict]:
    message = choice.get("message")
    if not isinstance(message, dict):
        message = {}
    blocks: list[dict] = []
    reasoning = message.get("reasoning_content")
    if isinstance(reasoning, str) and reasoning:
        blocks.append({"type": "thinking", "thinking": reasoning, "signature": ""})
    text = message.get("content")
    if isinstance(text, str) and text:
        blocks.append({"type": "text", "text": text})
    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list):
        for call in tool_calls:
            if not isinstance(call, dict):
                continue
            function = call.get("function")
            if not isinstance(function, dict):
                function = call
            blocks.append(
                {
                    "type": "tool_use",
                    "id": call.get("id") or f"call_{uuid.uuid4().hex[:12]}",
                    "name": function.get("name") or "",
                    "input": _decode_arguments(function.get("arguments")),
                }
            )
    if not blocks:
        return []
    return blocks


def _content_text(blocks: list[dict]) -> str:
    return "".join(block["text"] for block in blocks if block.get("type") == "text" and isinstance(block.get("text"), str))


def build_message(info: RequestInfo, message_id: str, response: dict) -> dict:
    choices = response.get("choices")
    choice = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else {}
    finish = choice.get("finish_reason")
    if isinstance(response.get("error"), dict):
        finish = "error"
    content = build_content(choice)
    stop_sequence = match_stop_sequence(_content_text(content), info.stop_sequences)
    return {
        "id": message_id,
        "type": "message",
        "role": "assistant",
        "model": info.model,
        "content": content,
        "stop_reason": "stop_sequence" if stop_sequence is not None else stop_reason_for(finish),
        "stop_sequence": stop_sequence,
        "usage": _usage(response.get("usage")),
    }


def count_input_tokens(messages: Any, system: str | None) -> int:
    total = count_messages_tokens(messages)
    if system:
        total += estimate_tokens(system)
    return total


def _decode_chunk(chunk: Any) -> str:
    if isinstance(chunk, str):
        return chunk
    if isinstance(chunk, (bytes, bytearray, memoryview)):
        raw = bytes(chunk)
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            log.warning("sse payload is not valid utf-8 at byte %d (%s), replaced %d byte(s)", exc.start, exc.reason, len(raw))
            return raw.decode("utf-8", errors="replace")
    return ""


def iter_sse_payloads(chunk: Any) -> Iterator[dict | None]:
    text = _decode_chunk(chunk)
    if not text:
        return
    for event in parse_sse(text):
        data = event.data
        if data == "[DONE]":
            yield None
        elif isinstance(data, dict):
            yield data


@dataclass
class _StreamState:
    info: RequestInfo
    message_id: str
    started: bool = False
    finished: bool = False
    text_index: int | None = None
    text_open: bool = False
    text_parts: list[str] = field(default_factory=list)
    thinking_index: int | None = None
    thinking_open: bool = False
    tool_index: dict[int, int] = field(default_factory=dict)
    tool_calls: dict[int, dict[str, str]] = field(default_factory=dict)
    tool_started: set[int] = field(default_factory=set)
    next_index: int = 0
    stop_reason: str = "end_turn"
    stop_sequence: str | None = None
    usage: dict = field(default_factory=lambda: {"input_tokens": 0, "output_tokens": 0})
    has_tool_use: bool = False

    def emit(self, event_type: str, data: dict) -> str:
        payload = {"type": event_type}
        payload.update(data)
        return sse_event(event_type, payload)

    def _take_index(self) -> int:
        value = self.next_index
        self.next_index += 1
        return value

    def start(self) -> Iterator[str]:
        if self.started:
            return
        self.started = True
        message: dict[str, Any] = {
            "id": self.message_id,
            "type": "message",
            "role": "assistant",
            "model": self.info.model,
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": self.usage["input_tokens"], "output_tokens": 0},
        }
        yield self.emit("message_start", {"message": message})

    def close_thinking(self) -> Iterator[str]:
        if not self.thinking_open:
            return
        self.thinking_open = False
        index = self.thinking_index if self.thinking_index is not None else 0
        self.thinking_index = None
        yield self.emit("content_block_stop", {"index": index})

    def close_text(self) -> Iterator[str]:
        if not self.text_open:
            return
        self.text_open = False
        index = self.text_index if self.text_index is not None else 0
        self.text_index = None
        yield self.emit("content_block_stop", {"index": index})

    def close_tools(self) -> Iterator[str]:
        for index in sorted(self.tool_index.values()):
            yield self.emit("content_block_stop", {"index": index})
        self.tool_index.clear()
        self.tool_started.clear()

    def close_for_tool(self) -> Iterator[str]:
        yield from self.close_thinking()
        yield from self.close_text()

    def close_all(self) -> Iterator[str]:
        yield from self.close_thinking()
        yield from self.close_text()
        yield from self.close_tools()

    def thinking_delta(self, text: str) -> Iterator[str]:
        if self.text_open:
            yield from self.close_text()
        if not self.thinking_open:
            self.thinking_open = True
            self.thinking_index = self._take_index()
            yield self.emit(
                "content_block_start",
                {"index": self.thinking_index, "content_block": {"type": "thinking", "thinking": "", "signature": ""}},
            )
        yield self.emit("content_block_delta", {"index": self.thinking_index, "delta": {"type": "thinking_delta", "thinking": text}})

    def text_delta(self, text: str) -> Iterator[str]:
        if self.thinking_open:
            yield from self.close_thinking()
        if not self.text_open:
            self.text_open = True
            self.text_index = self._take_index()
            yield self.emit("content_block_start", {"index": self.text_index, "content_block": {"type": "text", "text": ""}})
        self.text_parts.append(text)
        yield self.emit("content_block_delta", {"index": self.text_index, "delta": {"type": "text_delta", "text": text}})

    def tool_start(self, slot: int, call: dict, name: str) -> tuple[int, bool]:
        identity = self.tool_calls.setdefault(slot, {"id": "", "name": ""})
        call_id = call.get("id")
        if isinstance(call_id, str) and call_id and not identity["id"]:
            identity["id"] = call_id
        if not identity["id"]:
            identity["id"] = f"call_{uuid.uuid4().hex[:12]}"
        if name and not identity["name"]:
            identity["name"] = name
        index = self.tool_index.get(slot)
        if index is None:
            index = self._take_index()
            self.tool_index[slot] = index
        self.has_tool_use = True
        started = index not in self.tool_started
        self.tool_started.add(index)
        return index, started

    def tool_name(self, slot: int) -> str:
        identity = self.tool_calls.get(slot)
        return identity["name"] if identity else ""

    def tool_id(self, slot: int) -> str:
        identity = self.tool_calls.get(slot)
        return identity["id"] if identity else ""

    def finish(self, reason: str, stop_sequence: str | None) -> Iterator[str]:
        if self.finished:
            return
        self.finished = True
        self.stop_reason = reason
        self.stop_sequence = stop_sequence
        yield self.emit(
            "message_delta",
            {
                "delta": {"stop_reason": self.stop_reason, "stop_sequence": self.stop_sequence},
                "usage": {"input_tokens": self.usage["input_tokens"], "output_tokens": self.usage["output_tokens"]},
            },
        )
        yield self.emit("message_stop", {})


def _error_message(error: Any) -> tuple[str, str]:
    if isinstance(error, dict):
        message = error.get("message")
        return ("api_error", message if isinstance(message, str) and message else "upstream stream error")
    log.warning("upstream stream produced a non-dict error: %s", type(error).__name__)
    return ("api_error", "upstream stream error")


def _merge_usage(state_usage: dict, usage: Any) -> None:
    if not isinstance(usage, dict):
        return
    prompt = usage.get("prompt_tokens")
    if not isinstance(prompt, int):
        prompt = usage.get("input_tokens")
    completion = usage.get("completion_tokens")
    if not isinstance(completion, int):
        completion = usage.get("output_tokens")
    if isinstance(prompt, int) and prompt > 0:
        state_usage["input_tokens"] = prompt
    if isinstance(completion, int) and completion > 0:
        state_usage["output_tokens"] = completion


def _tool_slot(call: dict, position: int) -> int:
    index = call.get("index")
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        return position
    return index


async def translate_stream(
    chat_stream: AsyncIterator[str],
    info: RequestInfo,
    message_id: str,
    on_complete: Any = None,
) -> AsyncIterator[str]:
    state = _StreamState(info, message_id)
    state.usage = {"input_tokens": max(0, info.prompt_tokens), "output_tokens": 0}
    try:
        for line in state.start():
            yield line
        error: dict | None = None
        finish: str | None = None
        try:
            async for chunk in chat_stream:
                for payload in iter_sse_payloads(chunk):
                    if payload is None:
                        continue
                    if isinstance(payload.get("error"), dict):
                        error = payload["error"]
                        continue
                    _merge_usage(state.usage, payload.get("usage"))
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
                                for line in state.thinking_delta(reasoning):
                                    yield line
                            content = delta.get("content")
                            if isinstance(content, str) and content:
                                for line in state.text_delta(content):
                                    yield line
                            tool_calls = delta.get("tool_calls")
                            if isinstance(tool_calls, list):
                                for position, call in enumerate(tool_calls):
                                    if not isinstance(call, dict):
                                        continue
                                    function = call.get("function")
                                    if not isinstance(function, dict):
                                        function = call
                                    for line in state.close_for_tool():
                                        yield line
                                    slot = _tool_slot(call, position)
                                    name = function.get("name")
                                    index, started = state.tool_start(slot, call, name if isinstance(name, str) else "")
                                    if started:
                                        yield state.emit(
                                            "content_block_start",
                                            {
                                                "index": index,
                                                "content_block": {
                                                    "type": "tool_use",
                                                    "id": state.tool_id(slot),
                                                    "name": state.tool_name(slot),
                                                    "input": {},
                                                },
                                            },
                                        )
                                    arguments = function.get("arguments")
                                    if isinstance(arguments, str) and arguments:
                                        yield state.emit(
                                            "content_block_delta",
                                            {"index": index, "delta": {"type": "input_json_delta", "partial_json": arguments}},
                                        )
                        reason = choice.get("finish_reason")
                        if isinstance(reason, str) and reason and finish is None:
                            finish = reason
        except (GeneratorExit, asyncio.CancelledError):
            raise
        except BaseException as exc:
            log.warning("anthropic stream aborted by the upstream generator: %s", exc)
            for line in state.close_all():
                yield line
            yield sse_event("error", error_body("api_error", "upstream stream failed"))
            raise
        for line in state.close_all():
            yield line
        if error is not None:
            error_type, message = _error_message(error)
            yield sse_event("error", error_body(error_type, message))
            return
        stop_sequence = match_stop_sequence("".join(state.text_parts), info.stop_sequences)
        if finish in TRUNCATED_REASONS:
            reason = "max_tokens"
        elif stop_sequence is not None:
            reason = "stop_sequence"
        elif finish in TOOL_CALL_REASONS or (finish is None and state.has_tool_use):
            reason = "tool_use"
        else:
            reason = stop_reason_for(finish)
        if callable(on_complete):
            outcome = on_complete(reason, state.usage)
            if inspect.isawaitable(outcome):
                await outcome
        for line in state.finish(reason, stop_sequence):
            yield line
    finally:
        closer = getattr(chat_stream, "aclose", None)
        if closer is not None:
            try:
                await closer()
            except Exception as exc:
                log.debug("upstream chat stream close failed: %s", exc)
