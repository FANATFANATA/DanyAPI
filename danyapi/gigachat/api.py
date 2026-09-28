from __future__ import annotations

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
from ..usage import record_usage_dict
from .client import GigaChatError
from .messages import build_messages, normalize_finish_reason, normalize_usage, request_body

log = logging.getLogger("danyapi.gigachat.api")

MAX_RETRIES = 3

DONE_LINE = "data: [DONE]\n\n"


def _status_for(error: GigaChatError) -> int:
    code = error.code if isinstance(error.code, int) else 500
    if code == 401 or code == 403:
        return 401
    if code == 404:
        return 404
    if code == 429:
        return 429
    if 400 <= code < 500:
        return 400
    return 502


NO_IMAGE_MODELS_HINT = (
    "this GigaChat model does not accept images, use a Pro, Max or Ultra model such as GigaChat-2-Pro, GigaChat-2-Max, GigaChat-3-Pro or GigaChat-3-Ultra"
)

NO_IMAGE_MARKERS = ("does not support image", "not support image")


def _detail_for(error: GigaChatError) -> str:
    lowered = (error.message or "").lower()
    if any(marker in lowered for marker in NO_IMAGE_MARKERS):
        return f"GigaChat error: {NO_IMAGE_MODELS_HINT}"
    return f"GigaChat error: {error.message or error.code}"


def _translate_message(choice: dict) -> dict:
    message = choice.get("message")
    if not isinstance(message, dict):
        message = {}
    text = message.get("content")
    out: dict[str, Any] = {"role": "assistant", "content": text if isinstance(text, str) else ""}
    function_call = message.get("function_call")
    if isinstance(function_call, dict) and function_call.get("name"):
        arguments = function_call.get("arguments")
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments if arguments is not None else {})
        call_id = function_call.get("id")
        if not isinstance(call_id, str) or not call_id:
            call_id = f"call_{uuid.uuid4().hex[:24]}"
        out["tool_calls"] = [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": function_call["name"], "arguments": arguments or "{}"},
            }
        ]
    return out


async def _send(account: Any, body: dict, model: str) -> httpx.Response:
    attempt = 0
    while True:
        try:
            return await account.client.chat(body, model)
        except httpx.HTTPError as exc:
            status = getattr(getattr(exc, "response", None), "status_code", 0) or 0
            if status in RETRYABLE_HTTP_STATUSES and attempt < MAX_RETRIES:
                await _sleep_backoff(attempt)
                attempt += 1
                continue
            raise HTTPException(502, f"GigaChat transport error: {exc}") from exc
        except GigaChatError as exc:
            if exc.is_auth:
                account.mark_broken()
                raise HTTPException(401, _detail_for(exc)) from exc
            raise


async def _sleep_backoff(attempt: int) -> None:
    import asyncio

    await asyncio.sleep(_retry_delay(attempt + 1))


async def _read_json(resp: httpx.Response) -> Any:
    try:
        return json.loads(resp.content)
    except ValueError as exc:
        raise HTTPException(502, "GigaChat returned a malformed response") from exc


async def _raise_upstream(resp: httpx.Response, payload: Any) -> None:
    message = ""
    if isinstance(payload, dict) and isinstance(payload.get("message"), str):
        message = payload["message"]
    if not message:
        text = resp.text[:300] if resp.content else ""
        message = text or f"upstream returned {resp.status_code}"
    error = GigaChatError(resp.status_code, message)
    raise HTTPException(_status_for(error), _detail_for(error))


async def collect_non_stream(
    account,
    messages,
    model,
    tools=None,
    tool_choice=None,
    functions=None,
    function_call=None,
    temperature=None,
    top_p=None,
    max_tokens: int | None = None,
    stop: Any = None,
    response_format=None,
    user: str | None = None,
    session_id: str | None = None,
) -> dict:
    async with account_lock(account.sem, settings.acquire_timeout):
        gc_messages, specs, call_value = await build_messages(
            account.client,
            messages,
            tools,
            tool_choice,
            functions,
            function_call,
        )
        body = request_body(gc_messages, specs, call_value, temperature, top_p, max_tokens, response_format)
        resp = await _send(account, body, model)
        try:
            if resp.status_code >= 400:
                await _raise_upstream(resp, await _safe_json(resp))
            payload = await _read_json(resp)
        finally:
            await resp.aclose()

    if not isinstance(payload, dict):
        raise HTTPException(502, "GigaChat returned an unexpected payload")
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise HTTPException(502, "GigaChat returned no choices")
    choice = choices[0] if isinstance(choices[0], dict) else {}
    message = _translate_message(choice)
    message["content"] = _apply_stop(message.get("content") or "", stop)
    finish = normalize_finish_reason(choice.get("finish_reason"))
    usage = normalize_usage(payload.get("usage"))
    record_usage_dict("gigachat", model, usage, user=user, session_id=session_id)
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


async def _safe_json(resp: httpx.Response) -> Any:
    try:
        return json.loads(resp.content)
    except ValueError:
        return None


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


async def _iter_sse(resp: httpx.Response) -> AsyncIterator[dict]:
    buffer = ""
    async for raw in resp.aiter_bytes():
        buffer += raw.decode("utf-8", errors="replace")
        while "\n\n" in buffer:
            block, _, buffer = buffer.partition("\n\n")
            for line in block.splitlines():
                if not line.startswith("data: "):
                    continue
                payload = line[len("data: ") :].strip()
                if not payload or payload == "[DONE]":
                    continue
                try:
                    parsed = json.loads(payload)
                except ValueError:
                    continue
                if isinstance(parsed, dict):
                    yield parsed


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
    role = delta.get("role")
    if isinstance(role, str) and role:
        out["role"] = role
    function_call = delta.get("function_call")
    if isinstance(function_call, dict) and function_call.get("name"):
        arguments = function_call.get("arguments")
        call_id = function_call.get("id")
        if not isinstance(call_id, str) or not call_id:
            call_id = f"call_{uuid.uuid4().hex[:24]}"
        out["tool_calls"] = [
            {
                "index": 0,
                "id": call_id,
                "type": "function",
                "function": {
                    "name": function_call.get("name"),
                    "arguments": arguments if isinstance(arguments, str) else "{}",
                },
            }
        ]
    finish = choice.get("finish_reason")
    return out, finish if isinstance(finish, str) else None


async def stream_openai(
    account,
    messages,
    model,
    tools=None,
    tool_choice=None,
    functions=None,
    function_call=None,
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

    async with account_lock(account.sem, settings.acquire_timeout):
        try:
            gc_messages, specs, call_value = await build_messages(
                account.client,
                messages,
                tools,
                tool_choice,
                functions,
                function_call,
            )
            body = request_body(gc_messages, specs, call_value, temperature, top_p, max_tokens, response_format)
            body["stream"] = True
            resp = await _send(account, body, model)
        except HTTPException as exc:
            detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
            for line in _stream_error_sse(chunk_id, created, model, detail, session_id):
                yield line
            return
        except Exception as exc:
            for line in _stream_error_sse(chunk_id, created, model, f"GigaChat request failed: {exc}", session_id):
                yield line
            return

        usage_payload: dict | None = None
        emitted = False
        try:
            if resp.status_code >= 400:
                await _raise_upstream(resp, await _safe_json(resp))
            async for event in _iter_sse(resp):
                usage_raw = event.get("usage")
                if isinstance(usage_raw, dict):
                    usage_payload = usage_raw
                delta, finish = _delta_from_event(event)
                if delta:
                    if "content" in delta and stop is not None:
                        delta["content"] = _apply_stop(delta["content"], stop)
                        if not delta["content"]:
                            delta.pop("content")
                    if delta:
                        emitted = True
                        yield _chunk(chunk_id, created, model, delta)
                if finish is not None:
                    yield _chunk(chunk_id, created, model, {}, normalize_finish_reason(finish))
        except httpx.HTTPError as exc:
            yield _chunk(chunk_id, created, model, {}, "stop")
            log.warning("gigachat stream transport error: %s", exc)
        except Exception as exc:
            log.exception("gigachat stream failed: %s", exc)
            yield _chunk(chunk_id, created, model, {}, "stop")
        finally:
            await resp.aclose()

        if not emitted:
            yield _chunk(chunk_id, created, model, {"role": "assistant", "content": ""})
        if include_usage and usage_payload is not None:
            record_usage_dict("gigachat", model, normalize_usage(usage_payload), user=user, session_id=session_id)
            yield _chunk(chunk_id, created, model, {}, None, usage=normalize_usage(usage_payload))
    yield DONE_LINE
