from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

import httpx
from fastapi import HTTPException

from ..accounts import account_lock
from ..api.retry import RETRYABLE_HTTP_STATUSES, _retry_delay
from ..api.shaping import _apply_stop
from ..api.sse import _sse, _stream_error_sse
from ..config import settings
from ..tokens import estimate_tokens
from ..usage import record_usage_dict
from . import attest
from .client import (
    DEFAULT_MODEL,
    DuckAIError,
    DuckAIEvent,
)

log = logging.getLogger("danyapi.duckai.api")

MAX_RETRIES = 5
ATTESTATION_RETRY_DELAY = 0.35

DONE_LINE = "data: [DONE]\n\n"

PROVIDER = "duckai"

MAX_IMAGES_PER_MESSAGE = 3
MAX_IMAGES_PER_REQUEST = 10

SYSTEM_PREFIX = "Follow these instructions for the rest of the conversation:\n"

NO_IMAGE_HINT = "this duckai model does not accept images"

BLOCKED_HINT = (
    "duckai refused this request. Every duck.ai chat request carries a proof that a real browser made it, and this one did not "
    "pass DuckDuckGo's check. This is not a blocklist of your address: a browser on the same network passes. The proof is "
    "computed by danyapi/duckai/jsa_solver.js, so the refusal means that script used a check the solver does not reproduce "
    "yet. Set DANYAPI_LOG_LEVEL=DEBUG for solver timings, then retry, which usually lands on a script that works."
)

ENTRYPOINT_HINT = (
    "duckai refused this request as an unsupported entrypoint. There are two known causes. Either duck.ai is serving a "
    "different build than the one in the x-fe-version header, in which case restarting the server makes it read the "
    "current one, or the client has been recognised and is being refused outright, which is what happens after sustained "
    "automated use and is not cleared by waiting. The other providers are unaffected."
)


def _status_for(error: DuckAIError) -> int:
    if error.is_auth:
        return 401
    if error.is_challenge or error.is_entrypoint:
        return 403
    code = error.code if isinstance(error.code, int) else 502
    if code in RETRYABLE_HTTP_STATUSES:
        return 502
    if 400 <= code < 500:
        return 400
    return 502


def _detail_for(error: DuckAIError) -> str:
    if error.is_entrypoint:
        return ENTRYPOINT_HINT
    if error.is_challenge:
        return BLOCKED_HINT
    message = (error.message or "").strip()
    lowered = message.lower()
    if "image" in lowered and "not" in lowered:
        return f"{NO_IMAGE_HINT}: {message}"
    return f"duckai error: {message or error.code}"


def _http_error(error: DuckAIError) -> HTTPException:
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


def _images_of(content: Any) -> list[tuple[str, str]]:
    images: list[tuple[str, str]] = []
    if not isinstance(content, list):
        return images
    for item in content:
        if not isinstance(item, dict) or item.get("type") != "image_url":
            continue
        image_url = item.get("image_url")
        if isinstance(image_url, str):
            uri = image_url
        elif isinstance(image_url, dict) and isinstance(image_url.get("url"), str):
            uri = image_url["url"]
        else:
            continue
        if not uri.startswith("data:"):
            raise HTTPException(400, "duckai only accepts inline data URI images")
        meta, _, payload = uri[5:].partition(",")
        mime = meta.split(";", 1)[0].strip() or "image/png"
        if payload:
            images.append((mime, uri))
    return images


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


def _render_tools(specs: list[dict]) -> str:
    if not specs:
        return ""
    lines = ["You can call these tools when they help. Reply with a call instead of guessing.", ""]
    for spec in specs:
        lines.append(f"- {spec['name']}: {spec['description']}".rstrip())
        if spec["parameters"]:
            lines.append(f"  arguments schema: {json.dumps(spec['parameters'], separators=(',', ':'))}")
    return "\n".join(lines)


def build_messages(
    messages: Any,
    tools: Any = None,
    functions: Any = None,
) -> list[dict]:
    specs = _tool_specs(tools, functions)
    preamble = _render_tools(specs)
    system_chunks: list[str] = []
    out: list[dict] = []
    image_total = 0
    pending_tool_results: list[dict] = []

    def flush_assistant_parts() -> None:
        if not pending_tool_results:
            return
        if out and out[-1]["role"] == "assistant":
            out[-1]["parts"].extend(pending_tool_results)
        else:
            out.append({"role": "assistant", "content": "", "parts": list(pending_tool_results)})
        pending_tool_results.clear()

    for message in messages or []:
        role = getattr(message, "role", None)
        content = getattr(message, "content", None)
        if role in ("system", "developer"):
            text = _text_of(content).strip()
            if text:
                system_chunks.append(text)
            continue
        if role == "tool":
            call_id = getattr(message, "tool_call_id", None) or f"call_{uuid.uuid4().hex[:24]}"
            pending_tool_results.append(
                {
                    "type": "tool-result",
                    "toolCallId": str(call_id),
                    "result": _text_of(content),
                    "data": None,
                }
            )
            continue
        if role == "assistant":
            flush_assistant_parts()
            text = _text_of(content)
            parts: list[dict] = []
            for call in _tool_calls_of(message):
                parts.append({"type": "tool-call", "toolCallId": call["id"], "toolName": call["name"], "toolArguments": call["arguments"]})
            if text:
                parts.append({"type": "text", "text": text})
            if parts:
                out.append({"role": "assistant", "content": "", "parts": parts})
            continue

        text = _text_of(content)
        images = _images_of(content)
        if images:
            image_total += len(images)
            if image_total > MAX_IMAGES_PER_REQUEST:
                raise HTTPException(400, f"duckai accepts at most {MAX_IMAGES_PER_REQUEST} images per request")
            if len(images) > MAX_IMAGES_PER_MESSAGE:
                raise HTTPException(400, f"duckai accepts at most {MAX_IMAGES_PER_MESSAGE} images per message")
        flush_assistant_parts()
        parts = []
        if text:
            parts.append({"type": "text", "text": text})
        for mime, uri in images:
            parts.append({"type": "image", "mimeType": mime, "image": uri})
        if not parts:
            continue
        if role == "user" and not out and (system_chunks or preamble):
            lead = []
            if system_chunks:
                lead.append(SYSTEM_PREFIX + "\n\n".join(system_chunks))
            if preamble:
                lead.append(preamble)
            parts.insert(0, {"type": "text", "text": "\n\n".join(lead)})
            system_chunks = []
            preamble = ""
        out.append({"role": "user", "content": parts})

    flush_assistant_parts()
    if not out:
        lead = []
        if system_chunks:
            lead.append(SYSTEM_PREFIX + "\n\n".join(system_chunks))
        if preamble:
            lead.append(preamble)
        fallback = "\n\n".join([*lead, "Hello"]).strip()
        out.append({"role": "user", "content": [{"type": "text", "text": fallback or "Hello"}]})
    return out


def _usage_for(prompt: str, content: str, reasoning: str = "") -> dict:
    prompt_tokens = estimate_tokens(prompt)
    completion_tokens = estimate_tokens(content) + estimate_tokens(reasoning)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "prompt_tokens_details": {"cached_tokens": 0},
        "completion_tokens_details": {"reasoning_tokens": estimate_tokens(reasoning)},
    }


def _prompt_text(duck_messages: list[dict]) -> str:
    chunks: list[str] = []
    for message in duck_messages:
        if message.get("role") != "user":
            continue
        for part in message.get("content") or []:
            if isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str):
                chunks.append(part["text"])
    return "\n".join(chunks)


async def _open_stream(
    account: Any, duck_messages: list[dict], model: str, effort: str | None, can_use_tools: bool, can_use_web_search: bool
) -> AsyncIterator[DuckAIEvent]:
    attempt = 0
    while True:
        stream = account.client.chat(
            duck_messages,
            model=model,
            effort=effort,
            can_use_tools=can_use_tools,
            can_use_web_search=can_use_web_search,
        )
        try:
            async for event in stream:
                attempt = 0
                yield event
            return
        except httpx.HTTPError as exc:
            status = getattr(getattr(exc, "response", None), "status_code", 0) or 0
            if status in RETRYABLE_HTTP_STATUSES and attempt < MAX_RETRIES:
                await _sleep_backoff(attempt)
                attempt += 1
                continue
            raise HTTPException(502, f"duckai transport error: {exc}") from exc
        except attest.AttestationError as exc:
            account.client.invalidate_attestation()
            if attempt < MAX_RETRIES:
                await asyncio.sleep(ATTESTATION_RETRY_DELAY)
                attempt += 1
                continue
            raise HTTPException(502, f"duckai attestation failed: {exc}") from exc
        except DuckAIError as exc:
            if exc.is_challenge or exc.is_entrypoint:
                account.client.invalidate_attestation()
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(ATTESTATION_RETRY_DELAY)
                    attempt += 1
                    continue
                raise _http_error(exc) from exc
            if exc.is_retryable and attempt < MAX_RETRIES:
                await _sleep_backoff(attempt)
                attempt += 1
                continue
            raise _http_error(exc) from exc


async def _sleep_backoff(attempt: int) -> None:
    await asyncio.sleep(_retry_delay(attempt + 1))


def _tool_calls_out(event_calls: list[dict], collected: list[dict]) -> list[dict]:
    for index, call in enumerate(event_calls):
        collected.append(call if not index else {**call, "index": index})
    return collected


async def collect_non_stream(
    account,
    messages,
    model: str = DEFAULT_MODEL,
    tools=None,
    tool_choice=None,
    functions=None,
    function_call=None,
    thinking: bool | None = None,
    search: bool = False,
    stop: Any = None,
    user: str | None = None,
    session_id: str | None = None,
) -> dict:
    duck_messages = build_messages(messages, tools, functions)
    prompt = _prompt_text(duck_messages)
    content: list[str] = []
    reasoning: list[str] = []
    tool_calls: list[dict] = []
    sources: list[dict] = []
    refusal = ""
    finish = "stop"

    async with account_lock(account.sem, settings.acquire_timeout):
        try:
            async for event in _open_stream(
                account,
                duck_messages,
                model,
                "none" if thinking is False else None,
                bool(tools or functions),
                bool(search),
            ):
                if event.delta:
                    content.append(event.delta)
                if event.reasoning:
                    reasoning.append(event.reasoning)
                _tool_calls_out(event.tool_calls, tool_calls)
                sources.extend(event.sources)
                if event.refusal and not refusal:
                    refusal = event.refusal
                if event.finish is not None:
                    finish = event.finish
        except HTTPException:
            raise

    text = _apply_stop("".join(content), stop)
    if tool_calls:
        finish = "tool_calls"
    elif refusal and not text:
        finish = "content_filter"
    message: dict[str, Any] = {"role": "assistant", "content": text}
    if reasoning:
        message["reasoning_content"] = "".join(reasoning)
    if tool_calls:
        message["tool_calls"] = [{"id": call["id"], "type": "function", "function": call["function"]} for call in tool_calls]
    if sources:
        message["sources"] = sources
    usage = _usage_for(prompt, text, "".join(reasoning))
    record_usage_dict(PROVIDER, model, usage, user=user, session_id=session_id)
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "system_fingerprint": "fp_duckai",
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
    function_call=None,
    thinking: bool | None = None,
    search: bool = False,
    stop: Any = None,
    include_usage: bool = False,
    user: str | None = None,
    session_id: str | None = None,
) -> AsyncIterator[str]:
    chunk_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    duck_messages = build_messages(messages, tools, functions)
    prompt = _prompt_text(duck_messages)
    content: list[str] = []
    reasoning: list[str] = []
    tool_calls: list[dict] = []
    emitted = False
    finish = "stop"

    async with account_lock(account.sem, settings.acquire_timeout):
        yield _chunk(chunk_id, created, model, {"role": "assistant", "content": ""})
        try:
            async for event in _open_stream(
                account,
                duck_messages,
                model,
                "none" if thinking is False else None,
                bool(tools or functions),
                bool(search),
            ):
                if event.reasoning:
                    reasoning.append(event.reasoning)
                    emitted = True
                    yield _chunk(chunk_id, created, model, {"reasoning_content": event.reasoning})
                if event.delta:
                    piece = _apply_stop(event.delta, stop)
                    if piece:
                        content.append(piece)
                        emitted = True
                        yield _chunk(chunk_id, created, model, {"content": piece})
                for call in _tool_calls_out(event.tool_calls, tool_calls):
                    emitted = True
                    yield _chunk(chunk_id, created, model, {"tool_calls": [call]})
                if event.finish is not None:
                    finish = event.finish
        except HTTPException as exc:
            detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
            for line in _stream_error_sse(chunk_id, created, model, detail, session_id):
                yield line
            return

        if tool_calls:
            finish = "tool_calls"
        if not emitted:
            yield _chunk(chunk_id, created, model, {"content": ""})
        yield _chunk(chunk_id, created, model, {}, finish)
        usage = _usage_for(prompt, "".join(content), "".join(reasoning))
        record_usage_dict(PROVIDER, model, usage, user=user, session_id=session_id)
        if include_usage:
            yield _chunk(chunk_id, created, model, {}, None, usage=usage)
    yield DONE_LINE
