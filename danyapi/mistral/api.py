from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import random
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

import httpx
from fastapi import HTTPException

from ..accounts import account_lock
from ..api.retry import MAX_RETRIES, RETRYABLE_HTTP_STATUSES, _retry_delay
from ..api.shaping import _apply_limits
from ..api.sse import _sse, _stream_error_sse
from ..config import settings
from ..sseutil import StreamStopFilter, split_stop
from ..tokens import StreamBudget, estimate_tokens, trim_to_tokens
from ..tools import _choice_name
from ..usage import record_usage_dict
from .client import DEFAULT_MODEL, MistralChatClient, MistralChatError, MistralEvent

log = logging.getLogger("danyapi.mistral.api")

RETRY_JITTER = 0.5

DONE_LINE = "data: [DONE]\n\n"

PROVIDER = "mistral"

RATE_LIMIT_HINT = "le chat rate limited this session. Free accounts are capped per message count and the cap clears with time. Other providers are unaffected."


def _status_for(error: MistralChatError) -> int:
    if error.is_auth:
        return 401
    if error.rate_limited:
        return 429
    code = error.code if isinstance(error.code, int) else 502
    if code in RETRYABLE_HTTP_STATUSES:
        return 502
    if 400 <= code < 500:
        return 400
    return 502


def _detail_for(error: MistralChatError) -> str:
    if error.rate_limited:
        return f"{RATE_LIMIT_HINT} ({error.message})"
    return f"mistral error: {error.message or error.code}"


def _http_error(error: MistralChatError) -> HTTPException:
    return HTTPException(_status_for(error), _detail_for(error))


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
    if content is None:
        return ""
    return str(content)


def _tool_specs(tools: Any, functions: Any) -> list[dict]:
    specs: list[dict] = []
    for entry in tools or []:
        function = entry.get("function") if isinstance(entry, dict) else None
        if isinstance(function, dict) and function.get("name"):
            specs.append({"name": str(function["name"]), "description": str(function.get("description") or ""), "parameters": function.get("parameters") or {}})
    for entry in functions or []:
        if isinstance(entry, dict) and entry.get("name") and not any(spec["name"] == entry["name"] for spec in specs):
            specs.append({"name": str(entry["name"]), "description": str(entry.get("description") or ""), "parameters": entry.get("parameters") or {}})
    return specs


def _tool_calls_of(message: Any) -> list[dict]:
    if isinstance(message, dict):
        calls = message.get("tool_calls")
    else:
        calls = getattr(message, "tool_calls", None)
    if not isinstance(calls, list):
        return []
    out: list[dict] = []
    for call in calls:
        if not isinstance(call, dict):
            continue
        function = call.get("function")
        if not isinstance(function, dict) or not function.get("name"):
            continue
        raw_arguments = function.get("arguments")
        arguments = raw_arguments if isinstance(raw_arguments, str) else json.dumps(raw_arguments or {}, separators=(",", ":"))
        out.append(
            {
                "id": str(call.get("id") or f"call_{uuid.uuid4().hex[:24]}"),
                "name": str(function["name"]),
                "arguments": arguments,
            }
        )
    return out


def _tools_disabled(tool_choice: Any) -> bool:
    return _choice_name(tool_choice) == "none"


def _render_tools(specs: list[dict]) -> str:
    if not specs:
        return ""
    lines = ["You can call these tools when they help. Reply with a call instead of guessing.", ""]
    for spec in specs:
        lines.append(f"- {spec['name']}: {spec['description']}".rstrip())
        if spec["parameters"]:
            lines.append(f"  arguments schema: {json.dumps(spec['parameters'], separators=(',', ':'))}")
    return "\n".join(lines)


def _render_history(messages: Any, tools: Any, functions: Any, tool_choice: Any) -> str:
    if _tools_disabled(tool_choice):
        tools = None
        functions = None
    preamble = _render_tools(_tool_specs(tools, functions))
    system_chunks: list[str] = []
    turns: list[tuple[str, str]] = []
    for message in messages or []:
        role = getattr(message, "role", None)
        content = getattr(message, "content", None)
        if role in ("system", "developer"):
            text = _text_of(content).strip()
            if text:
                system_chunks.append(text)
            continue
        if role in ("tool", "function"):
            name = getattr(message, "name", None) or "tool"
            turns.append(("user", f"[tool result for {name}]\n{_text_of(content)}"))
            continue
        if role == "assistant":
            text = _text_of(content)
            calls = _tool_calls_of(message)
            if calls:
                rendered = "\n".join(f'<tool-call name="{call["name"]}" arguments="{call["arguments"]}"></tool-call>' for call in calls)
                text = f"{rendered}\n{text}".strip()
            if text:
                turns.append(("assistant", text))
            continue
        text = _text_of(content)
        if text:
            turns.append(("user", text))
    lead: list[str] = []
    if system_chunks:
        lead.append("Follow these instructions for the rest of the conversation:\n" + "\n\n".join(system_chunks))
    if preamble:
        lead.append(preamble)
    if not turns:
        turns = [("user", "Hello")]
    rendered_turns = [f"<{role}>\n{text}\n</{role}>" for role, text in turns]
    return "\n\n".join(lead + rendered_turns)


def _usage_for(prompt: str, content: str) -> dict:
    prompt_tokens = estimate_tokens(prompt)
    completion_tokens = estimate_tokens(content)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "prompt_tokens_details": {"cached_tokens": 0},
        "completion_tokens_details": {"reasoning_tokens": 0},
    }


class _Retry(Exception):
    pass


async def _stream_events(
    account: Any,
    prompt: str,
    prelude: tuple[str, ...] = (),
) -> AsyncIterator[Any]:
    attempt = 0
    started = False
    while True:
        async with contextlib.AsyncExitStack() as stack:
            await stack.enter_async_context(account_lock(account.sem, settings.acquire_timeout))
            if not started:
                started = True
                for line in prelude:
                    yield line
            try:
                async for event in _open_stream(account, prompt, attempt):
                    yield event
                return
            except _Retry:
                pass
        await _sleep_backoff(attempt)
        attempt += 1


async def _open_stream(account: Any, prompt: str, attempt: int) -> AsyncIterator[MistralEvent]:
    delivered = False
    client: MistralChatClient = account.client
    try:
        async with client.chat(prompt) as events:
            async for event in events:
                delivered = True
                yield event
    except httpx.HTTPError as exc:
        status = getattr(getattr(exc, "response", None), "status_code", 0) or 0
        if not delivered and status in RETRYABLE_HTTP_STATUSES and attempt < MAX_RETRIES:
            raise _Retry() from exc
        raise HTTPException(502, f"mistral transport error: {exc}") from exc
    except MistralChatError as exc:
        if exc.is_auth and not delivered and attempt < MAX_RETRIES and await client.relogin():
            raise _Retry() from exc
        if exc.rate_limited and not delivered and attempt < MAX_RETRIES:
            raise _Retry() from exc
        if exc.is_retryable and not delivered and attempt < MAX_RETRIES:
            raise _Retry() from exc
        raise _http_error(exc) from exc


async def _sleep_backoff(attempt: int) -> None:
    base = _retry_delay(attempt + 1)
    if base <= 0:
        return
    await asyncio.sleep(base * (1.0 - RETRY_JITTER + random.random() * RETRY_JITTER))


async def collect_non_stream(
    account,
    messages,
    model: str = DEFAULT_MODEL,
    tools=None,
    tool_choice=None,
    functions=None,
    stop: Any = None,
    max_tokens: int | None = None,
    user: str | None = None,
    session_id: str | None = None,
) -> dict:
    prompt = _render_history(messages, tools, functions, tool_choice)
    content: list[str] = []
    finish = "stop"

    async for event in _stream_events(account, prompt):
        if event.delta:
            content.append(event.delta)
        if event.finish is not None:
            finish = event.finish

    text, limit_finish = _apply_limits("".join(content), max_tokens, stop)
    if limit_finish == "length":
        finish = "length"
    message: dict[str, Any] = {"role": "assistant", "content": text}
    usage = _usage_for(prompt, text)
    record_usage_dict(PROVIDER, model, usage, user=user, session_id=session_id)
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "system_fingerprint": "fp_mistral_chat",
        "choices": [{"index": 0, "message": message, "finish_reason": finish, "logprobs": None}],
        "usage": usage,
        "session_id": session_id,
    }


def _chunk(chunk_id: str, created: int, model: str, delta: dict, finish: str | None = None, usage: dict | None = None) -> str:
    payload: dict[str, Any] = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    if usage is not None:
        payload["usage"] = usage
    return _sse(payload)


async def stream_openai(
    account,
    messages,
    model: str = DEFAULT_MODEL,
    tools=None,
    tool_choice=None,
    functions=None,
    stop: Any = None,
    max_tokens: int | None = None,
    include_usage: bool = False,
    user: str | None = None,
    session_id: str | None = None,
) -> AsyncIterator[str]:
    chunk_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    prompt = _render_history(messages, tools, functions, tool_choice)
    content: list[str] = []
    emitted = False
    finish = "stop"
    stop_markers = split_stop(stop)
    stop_filter = StreamStopFilter(stop_markers) if stop_markers else None
    budget = StreamBudget(max_tokens, trim_to_tokens)
    stop_hit = False

    try:
        async for item in _stream_events(
            account,
            prompt,
            prelude=(_chunk(chunk_id, created, model, {"role": "assistant"}),),
        ):
            if isinstance(item, str):
                yield item
                continue
            event = item
            if event.delta:
                if stop_filter is None:
                    piece = event.delta
                elif stop_hit:
                    piece = ""
                else:
                    piece, hit = stop_filter.feed(event.delta)
                    stop_hit = stop_hit or hit
                piece = budget.feed(piece)
                if piece:
                    content.append(piece)
                    emitted = True
                    yield _chunk(chunk_id, created, model, {"content": piece})
            if event.finish is not None:
                finish = event.finish
    except HTTPException as exc:
        detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
        record_usage_dict(PROVIDER, model, _usage_for(prompt, "".join(content)), user=user, session_id=session_id)
        for line in _stream_error_sse(chunk_id, created, model, detail, session_id):
            yield line
        return

    if stop_filter is not None and not stop_hit:
        tail = budget.feed(stop_filter.flush())
        if tail:
            content.append(tail)
            emitted = True
            yield _chunk(chunk_id, created, model, {"content": tail})
    if budget.done:
        finish = "length"
    if not emitted:
        yield _chunk(chunk_id, created, model, {"content": ""})
    yield _chunk(chunk_id, created, model, {}, finish)
    usage = _usage_for(prompt, "".join(content))
    record_usage_dict(PROVIDER, model, usage, user=user, session_id=session_id)
    if include_usage:
        yield _chunk(chunk_id, created, model, {}, None, usage=usage)
    yield DONE_LINE
