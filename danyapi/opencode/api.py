from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections.abc import AsyncIterator, Iterator
from typing import Any

import httpx
from fastapi import HTTPException

from ..accounts import account_lock
from ..api.retry import MAX_RETRIES, _retry_delay
from ..api.shaping import _apply_stop
from ..api.sse import _sse, _stream_error_sse
from ..config import settings
from ..sseutil import IncrementalSSE
from ..usage import record_usage_dict
from .client import ERROR_STATUS_BY_TYPE, OpenCodeError, error_message, new_request_id
from .messages import build_messages, normalize_finish_reason, normalize_usage, request_body

log = logging.getLogger("danyapi.opencode.api")

DONE_LINE = "data: [DONE]\n\n"


def _status_for(error: OpenCodeError) -> int:
    mapped = ERROR_STATUS_BY_TYPE.get(error.error_type)
    if mapped is not None:
        return mapped
    code = error.code if isinstance(error.code, int) else 500
    if code == 401:
        return 401
    if code == 404:
        return 404
    if code == 429:
        return 429
    if 400 <= code < 500:
        return 400
    return 502


async def _send(
    account: Any,
    body: dict,
    model: str,
    session_id: str | None,
    request_id: str,
) -> httpx.Response:
    attempt = 0
    while True:
        try:
            return await account.client.chat(body, model, session_id=session_id, request_id=request_id)
        except httpx.HTTPError as exc:
            if attempt < MAX_RETRIES:
                await asyncio.sleep(_retry_delay(attempt + 1))
                attempt += 1
                continue
            raise HTTPException(502, f"OpenCode Zen transport error: {exc}") from exc


async def _body_bytes(resp: httpx.Response) -> bytes:
    if not resp.is_stream_consumed:
        try:
            await resp.aread()
        except (httpx.HTTPError, OSError, RuntimeError) as exc:
            log.debug("opencode upstream body could not be read: %s", exc)
            return b""
    try:
        return resp.content
    except (httpx.HTTPError, OSError, RuntimeError):
        return b""


async def _safe_json(resp: httpx.Response) -> Any:
    try:
        return json.loads(await _body_bytes(resp))
    except ValueError:
        return None


async def _close_quietly(resp: httpx.Response | None) -> None:
    if resp is None:
        return
    try:
        await resp.aclose()
    except Exception as exc:
        log.debug("opencode response close failed: %s", exc)


async def _raise_upstream(account: Any, resp: httpx.Response, payload: Any) -> None:
    error_type, message = error_message(payload, resp.status_code)
    if not isinstance(payload, dict):
        text = (await _body_bytes(resp)).decode("utf-8", errors="replace")[:300]
        message = text or f"upstream returned {resp.status_code}"
    error = OpenCodeError(resp.status_code, message, error_type)
    if error.is_auth:
        account.mark_broken()
    raise HTTPException(_status_for(error), error.detail)


def _tool_calls_of(message: dict) -> list[dict]:
    calls = message.get("tool_calls")
    if not isinstance(calls, list):
        return []
    out: list[dict] = []
    for call in calls:
        if not isinstance(call, dict):
            continue
        function = call.get("function")
        function = function if isinstance(function, dict) else {}
        arguments = function.get("arguments")
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments if arguments is not None else {})
        out.append(
            {
                "id": call.get("id") or f"call_{uuid.uuid4().hex[:24]}",
                "type": "function",
                "function": {"name": function.get("name") or "", "arguments": arguments or "{}"},
            }
        )
    return out


def _translate_message(choice: dict) -> dict:
    message = choice.get("message")
    if not isinstance(message, dict):
        message = {}
    text = message.get("content")
    out: dict[str, Any] = {"role": "assistant", "content": text if isinstance(text, str) else ""}
    reasoning = message.get("reasoning_content")
    if isinstance(reasoning, str) and reasoning:
        out["reasoning_content"] = reasoning
    calls = _tool_calls_of(message)
    if calls:
        out["tool_calls"] = calls
    return out


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


def _events(payloads: Any) -> Iterator[dict]:
    for event in payloads:
        data = event.data
        if isinstance(data, dict) and data:
            yield data


async def _iter_sse(resp: httpx.Response) -> AsyncIterator[dict]:
    parser = IncrementalSSE()
    async for raw in resp.aiter_bytes():
        for event in _events(parser.feed(raw)):
            yield event
    for event in _events(parser.finish()):
        yield event


def _delta_from_event(event: dict) -> tuple[dict, str | None]:
    choices = event.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return {}, None
    choice = choices[0]
    delta = choice.get("delta")
    if not isinstance(delta, dict):
        message = choice.get("message")
        delta = message if isinstance(message, dict) else {}
    out: dict[str, Any] = {}
    content = delta.get("content")
    if isinstance(content, str) and content:
        out["content"] = content
    reasoning = delta.get("reasoning_content")
    if isinstance(reasoning, str) and reasoning:
        out["reasoning_content"] = reasoning
    role = delta.get("role")
    if isinstance(role, str) and role:
        out["role"] = role
    tool_calls = delta.get("tool_calls")
    if isinstance(tool_calls, list):
        for call in tool_calls:
            if not isinstance(call, dict):
                continue
            function = call.get("function")
            function = function if isinstance(function, dict) else {}
            out.setdefault("tool_calls", []).append(
                {
                    "index": call.get("index") if isinstance(call.get("index"), int) else 0,
                    "id": call.get("id"),
                    "type": "function",
                    "function": {
                        "name": function.get("name") or "",
                        "arguments": function.get("arguments") if isinstance(function.get("arguments"), str) else "",
                    },
                }
            )
    finish = choice.get("finish_reason")
    return out, finish if isinstance(finish, str) else None


def _body(
    messages: list[Any],
    tools: Any,
    tool_choice: Any,
    temperature: float | None,
    top_p: float | None,
    max_tokens: int | None,
    stop: Any,
    response_format: Any,
) -> dict:
    return request_body(
        build_messages(messages),
        tools,
        tool_choice,
        temperature,
        top_p,
        max_tokens,
        stop=stop,
        response_format=response_format,
    )


async def collect_non_stream(
    account,
    messages,
    model,
    tools=None,
    tool_choice=None,
    temperature=None,
    top_p=None,
    max_tokens: int | None = None,
    stop: Any = None,
    response_format=None,
    user: str | None = None,
    session_id: str | None = None,
) -> dict:
    request_id = new_request_id()
    async with account_lock(account.sem, settings.acquire_timeout):
        body = _body(messages, tools, tool_choice, temperature, top_p, max_tokens, stop, response_format)
        try:
            resp = await _send(account, body, model, session_id, request_id)
        except OpenCodeError as exc:
            if exc.is_auth:
                account.mark_broken()
            raise HTTPException(_status_for(exc), exc.detail) from exc
        try:
            if resp.status_code >= 400:
                await _raise_upstream(account, resp, await _safe_json(resp))
            try:
                payload = json.loads(await _body_bytes(resp))
            except ValueError as exc:
                raise HTTPException(502, "OpenCode Zen returned a malformed response") from exc
        finally:
            await resp.aclose()

    if not isinstance(payload, dict):
        raise HTTPException(502, "OpenCode Zen returned an unexpected payload")
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise HTTPException(502, "OpenCode Zen returned no choices")
    choice = choices[0] if isinstance(choices[0], dict) else {}
    message = _translate_message(choice)
    message["content"] = _apply_stop(message.get("content") or "", stop)
    finish = normalize_finish_reason(choice.get("finish_reason"))
    usage = normalize_usage(payload.get("usage"))
    record_usage_dict("opencode", model, usage, user=user, session_id=session_id)
    return {
        "id": payload.get("id") or f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(payload.get("created") or time.time()),
        "model": model,
        "system_fingerprint": payload.get("model") or "fp_danyapi",
        "choices": [{"index": 0, "message": message, "finish_reason": finish, "logprobs": None}],
        "usage": usage,
        "session_id": session_id,
    }


async def stream_openai(
    account,
    messages,
    model,
    tools=None,
    tool_choice=None,
    temperature=None,
    top_p=None,
    max_tokens: int | None = None,
    stop: Any = None,
    response_format=None,
    include_usage: bool = False,
    user: str | None = None,
    session_id: str | None = None,
) -> AsyncIterator[str]:
    chunk_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    request_id = new_request_id()

    async with account_lock(account.sem, settings.acquire_timeout):
        resp: httpx.Response | None = None
        try:
            body = _body(messages, tools, tool_choice, temperature, top_p, max_tokens, stop, response_format)
            body["stream"] = True
            if include_usage:
                body["stream_options"] = {"include_usage": True}
            resp = await _send(account, body, model, session_id, request_id)
            if resp.status_code >= 400:
                await _raise_upstream(account, resp, await _safe_json(resp))
        except HTTPException as exc:
            await _close_quietly(resp)
            detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
            for line in _stream_error_sse(chunk_id, created, model, detail, session_id):
                yield line
            return
        except Exception as exc:
            await _close_quietly(resp)
            log.warning("opencode request failed: %s", exc)
            for line in _stream_error_sse(chunk_id, created, model, f"OpenCode Zen request failed: {exc}", session_id):
                yield line
            return

        usage_payload: dict | None = None
        emitted = False
        stop_hit = False
        try:
            async for event in _iter_sse(resp):
                usage_raw = event.get("usage")
                if isinstance(usage_raw, dict):
                    usage_payload = usage_raw
                delta, finish = _delta_from_event(event)
                if delta:
                    if "content" in delta and stop is not None:
                        kept = "" if stop_hit else _apply_stop(delta["content"], stop)
                        stop_hit = stop_hit or kept != delta["content"]
                        delta["content"] = kept
                        if not kept:
                            delta.pop("content")
                    if delta:
                        emitted = True
                        yield _chunk(chunk_id, created, model, delta)
                if finish is not None:
                    yield _chunk(chunk_id, created, model, {}, normalize_finish_reason(finish))
        except HTTPException:
            raise
        except httpx.HTTPError as exc:
            log.warning("opencode stream transport error: %s", exc)
            raise HTTPException(502, f"OpenCode Zen stream transport error: {exc}") from exc
        except Exception as exc:
            log.exception("opencode stream failed: %s", exc)
            yield _chunk(chunk_id, created, model, {}, "stop")
        finally:
            await resp.aclose()

        if not emitted:
            yield _chunk(chunk_id, created, model, {"role": "assistant", "content": ""})
        record_usage_dict("opencode", model, normalize_usage(usage_payload), user=user, session_id=session_id)
        if include_usage and usage_payload is not None:
            yield _chunk(chunk_id, created, model, {}, None, usage=normalize_usage(usage_payload))
    yield DONE_LINE
