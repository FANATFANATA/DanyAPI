from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
import uuid

import httpx
from fastapi import HTTPException

from .. import tools as toolemu
from ..deepseek.stream import IncrementalSSE
from .client import QwenClient, QwenError
from .stream import QwenStreamReconstructor, error_code

log = logging.getLogger("danyapi.qwen.api")

MAX_RETRIES = 5
RETRY_BACKOFF_SEC = 1.0
RETRY_BACKOFF_MAX_SEC = 8.0

RETRYABLE_HTTP_STATUSES = {408, 425, 429, 500, 502, 503, 504}


def _is_retryable_http(exc: HTTPException) -> bool:
    return exc.status_code in RETRYABLE_HTTP_STATUSES


RETRYABLE_ERROR_CODES = {
    "Too_Many_Requests",
    "RateLimited",
    "quotaLimited",
    "Internal_Server_Error",
    "Server_Busy",
    "server_busy",
    "Busy",
    "busy",
}

AUTH_ERROR_CODES = {
    "unauthorized",
    "Unauthorized",
    "Invalid token",
    "Forbidden",
    "forbidden",
}

RATE_LIMIT_ERROR_CODES = {
    "Too_Many_Requests",
    "RateLimited",
    "quotaLimited",
}

CONTEXT_LIMIT_MARKERS = ("context", "maxinput", "toolong", "lengthexceeded", "tokenlimit")


class ContextLimitError(Exception):
    pass


def _is_context_limit_code(code) -> bool:
    if not isinstance(code, str) or not code:
        return False
    compact = "".join(ch for ch in code.lower() if ch.isalnum())
    return any(marker in compact for marker in CONTEXT_LIMIT_MARKERS)


def _is_context_limit(rec: QwenStreamReconstructor) -> bool:
    return _is_context_limit_code(error_code(rec.error))


def _drop_session(pool, account, session_key) -> None:
    pool.forget(session_key)
    pool.forget_context(session_key)
    account.sessions.forget(session_key)


def _error_status(code) -> int:
    if code in RATE_LIMIT_ERROR_CODES:
        return 429
    if code in AUTH_ERROR_CODES:
        return 401
    return 502


def _handle_account_error(account, exc: Exception) -> None:
    code = getattr(exc, "code", None)
    if code in AUTH_ERROR_CODES:
        account.mark_broken()
        log.warning("qwen account #%d auth error %s: %s", account.index, code, exc)
    else:
        log.warning("qwen account #%d error: %s", account.index, exc)


async def _prepare_session(account, pool, existing_sid: str | None, model_id: str, context_seq: tuple[str, ...] | None = None):
    try:
        session, session_key = await account.sessions.obtain(existing_sid, model_id)
    except QwenError as exc:
        _handle_account_error(account, exc)
        raise HTTPException(_error_status(exc.code), f"Qwen error: {exc}") from exc
    if session_key != existing_sid:
        pool.register(account.index, session_key)
        if existing_sid:
            pool.forget(existing_sid)
            pool.forget_context(existing_sid)
    if context_seq:
        pool.index_context(session_key, context_seq)
    return session, session_key


async def _send_completion(client: QwenClient, session, prompt: str, model_id: str, thinking: bool, search: bool):
    try:
        resp = await client.completion(
            chat_session_id=session.id,
            prompt=prompt,
            parent_message_id=session.last_response_id,
            model=model_id,
            thinking=thinking,
            search=search,
        )
    except httpx.HTTPStatusError as exc:
        raise HTTPException(exc.response.status_code, exc.response.text[:500]) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(502, f"Qwen request failed: {exc}") from exc

    if resp.status_code != 200:
        body = await resp.aread()
        await resp.aclose()
        raise HTTPException(resp.status_code, body[:500].decode("utf-8", errors="replace"))

    content_type = resp.headers.get("content-type", "")
    if "text/event-stream" not in content_type:
        body = await resp.aread()
        await resp.aclose()
        text = body[:500].decode("utf-8", errors="replace")
        if "text/html" in content_type or b"requestInfo" in body:
            raise HTTPException(502, "Qwen WAF challenge: request blocked by anti-bot, try again later")
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            raise HTTPException(502, f"Qwen request failed: {text}") from None
        if isinstance(payload, dict):
            error = payload.get("error")
            if isinstance(error, dict):
                code = error.get("code")
                if _is_context_limit_code(code):
                    raise ContextLimitError() from None
                raise HTTPException(_error_status(code), error.get("message") or error.get("details") or "Qwen error")
            data = payload.get("data")
            if isinstance(data, dict) and data.get("code"):
                if _is_context_limit_code(data["code"]):
                    raise ContextLimitError() from None
                raise HTTPException(_error_status(data["code"]), data.get("details") or data.get("message") or "Qwen error")
        raise HTTPException(502, f"Qwen request failed: {text}")
    return resp


def _is_retryable_error(rec: QwenStreamReconstructor) -> bool:
    return bool(rec.error and error_code(rec.error) in RETRYABLE_ERROR_CODES and not rec.has_content)


def _error_body(rec: QwenStreamReconstructor) -> str:
    err = rec.error or {}
    return json.dumps(
        {
            "error": {
                "message": err.get("details") or err.get("message") or "Qwen server error, try again later",
                "code": err.get("code"),
            }
        },
        ensure_ascii=False,
    )


async def _try_stop_stream(client, session_id: str, message_id: str | None) -> None:
    if not session_id or not message_id:
        return
    try:
        await client.stop_stream(session_id, message_id)
    except Exception as exc:
        log.debug("stop_stream failed for %s: %s", session_id, exc)


def _accumulate_usage(session, rec: QwenStreamReconstructor) -> dict:
    current = rec.usage_tokens
    prev_input = int(getattr(session, "accumulated_input_tokens", 0) or 0)
    prev_output = int(getattr(session, "accumulated_output_tokens", 0) or 0)
    input_tokens = prev_input + current["prompt_tokens"]
    output_tokens = prev_output + current["completion_tokens"]
    session.accumulated_input_tokens = input_tokens
    session.accumulated_output_tokens = output_tokens
    return {
        "prompt_tokens": input_tokens,
        "completion_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }


async def collect_non_stream(
    account,
    pool,
    existing_sid,
    lock,
    prompt,
    model,
    model_id,
    thinking,
    search,
    tool_mode=False,
    tool_schemas=None,
    include_usage=False,
    context_seq: tuple[str, ...] | None = None,
):
    async with lock:
        session, session_key = await _prepare_session(account, pool, existing_sid, model_id, context_seq)
        stop_response_id: str | None = None
        try:
            rec: QwenStreamReconstructor | None = None
            for attempt in range(MAX_RETRIES + 1):
                if attempt:
                    await asyncio.sleep(min(RETRY_BACKOFF_SEC * (2 ** (attempt - 1)), RETRY_BACKOFF_MAX_SEC))
                try:
                    resp = await _send_completion(account.client, session, prompt, model_id, thinking, search)
                except ContextLimitError:
                    _drop_session(pool, account, session_key)
                    raise HTTPException(400, "context length exceeded: conversation too long, start a new conversation") from None
                except HTTPException as exc:
                    if _is_retryable_http(exc) and attempt < MAX_RETRIES:
                        log.warning(
                            "qwen provider error (%s), attempt %d/%d",
                            exc.status_code,
                            attempt + 1,
                            MAX_RETRIES,
                        )
                        continue
                    raise
                rec = QwenStreamReconstructor()
                incremental = IncrementalSSE()
                try:
                    async for chunk in resp.aiter_bytes():
                        for event in incremental.feed(chunk):
                            rec.handle(event)
                    for event in incremental.finish():
                        rec.handle(event)
                finally:
                    if rec.response_id:
                        stop_response_id = rec.response_id
                    await resp.aclose()
                if _is_retryable_error(rec) and attempt < MAX_RETRIES:
                    log.warning(
                        "qwen retryable error (%s), attempt %d/%d",
                        error_code(rec.error),
                        attempt + 1,
                        MAX_RETRIES,
                    )
                    continue
                break
        except asyncio.CancelledError:
            await _try_stop_stream(account.client, session.id, stop_response_id)
            raise

        assert rec is not None
        if _is_context_limit(rec) and not rec.has_content:
            _drop_session(pool, account, session_key)
            raise HTTPException(400, "context length exceeded: conversation too long, start a new conversation")
        usage = _accumulate_usage(session, rec)
        account.sessions.touch_last_message(session_key, rec.response_id)

        if not rec.has_content and rec.error:
            raise HTTPException(_error_status(error_code(rec.error)), _error_body(rec))

        if tool_mode:
            parsed = toolemu.parse_tool_calls(rec.content, tool_schemas)
            if parsed is not None:
                tool_calls, tool_text = parsed
                message = toolemu.format_tool_message(tool_calls, tool_text, rec.reasoning)
                finish = "tool_calls"
            else:
                message = {"role": "assistant", "content": rec.content}
                if rec.reasoning:
                    message["reasoning_content"] = rec.reasoning
                finish = "stop"
        else:
            message = {"role": "assistant", "content": rec.content}
            if rec.reasoning:
                message["reasoning_content"] = rec.reasoning
            finish = "stop"
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": usage,
            "session_id": session_key,
        }


async def stream_openai(
    account,
    pool,
    existing_sid,
    lock,
    prompt,
    model,
    model_id,
    thinking,
    search,
    tool_mode=False,
    tool_schemas=None,
    include_usage=False,
    context_seq: tuple[str, ...] | None = None,
):
    chunk_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    def sse(data: dict) -> str:
        return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

    async with lock:
        try:
            session, session_key = await _prepare_session(account, pool, existing_sid, model_id, context_seq)
        except HTTPException as exc:
            detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
            yield sse(
                {
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "error": {"message": detail},
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "error"}],
                }
            )
            yield sse(
                {
                    "id": chunk_id,
                    "session_id": None,
                    "object": "chat.completion.chunk",
                    "choices": [],
                }
            )
            yield "data: [DONE]\n\n"
            return

        rec: QwenStreamReconstructor | None = None
        content_buf = ""
        stop_response_id: str | None = None
        for attempt in range(MAX_RETRIES + 1):
            if attempt:
                await asyncio.sleep(min(RETRY_BACKOFF_SEC * (2 ** (attempt - 1)), RETRY_BACKOFF_MAX_SEC))
            try:
                resp = await _send_completion(account.client, session, prompt, model_id, thinking, search)
            except ContextLimitError:
                _drop_session(pool, account, session_key)
                yield sse(
                    {
                        "id": chunk_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "error": {
                            "message": "context length exceeded: conversation too long, start a new conversation",
                            "code": "context_length_exceeded",
                        },
                        "choices": [
                            {
                                "index": 0,
                                "delta": {},
                                "finish_reason": "length",
                            }
                        ],
                    }
                )
                yield sse(
                    {
                        "id": chunk_id,
                        "session_id": session_key,
                        "object": "chat.completion.chunk",
                        "choices": [],
                    }
                )
                yield "data: [DONE]\n\n"
                return
            except HTTPException as exc:
                if _is_retryable_http(exc) and attempt < MAX_RETRIES:
                    log.warning(
                        "qwen provider error (%s), attempt %d/%d",
                        exc.status_code,
                        attempt + 1,
                        MAX_RETRIES,
                    )
                    continue
                detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
                yield sse(
                    {
                        "id": chunk_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "error": {"message": detail},
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "error"}],
                    }
                )
                yield sse(
                    {
                        "id": chunk_id,
                        "session_id": session_key,
                        "object": "chat.completion.chunk",
                        "choices": [],
                    }
                )
                yield "data: [DONE]\n\n"
                return
            rec = QwenStreamReconstructor()
            incremental = IncrementalSSE()
            got_content = False
            pending: list[str] = []
            try:
                async for chunk in resp.aiter_bytes():
                    for event in incremental.feed(chunk):
                        rec.handle(event)
                        c_diff, r_diff = rec.take_diffs()
                        if c_diff or r_diff:
                            got_content = True
                        if not pending:
                            pending.append(
                                sse(
                                    {
                                        "id": chunk_id,
                                        "object": "chat.completion.chunk",
                                        "created": created,
                                        "model": model,
                                        "choices": [
                                            {
                                                "index": 0,
                                                "delta": {"role": "assistant"},
                                                "finish_reason": None,
                                            }
                                        ],
                                    }
                                )
                            )
                        delta: dict = {}
                        if c_diff:
                            if tool_mode:
                                content_buf += c_diff
                            else:
                                delta["content"] = c_diff
                        if r_diff:
                            delta["reasoning_content"] = r_diff
                        if delta:
                            pending.append(
                                sse(
                                    {
                                        "id": chunk_id,
                                        "object": "chat.completion.chunk",
                                        "created": created,
                                        "model": model,
                                        "choices": [
                                            {
                                                "index": 0,
                                                "delta": delta,
                                                "finish_reason": None,
                                            }
                                        ],
                                    }
                                )
                            )
                        if got_content:
                            for line in pending:
                                yield line
                            pending.clear()
            finally:
                if rec.response_id:
                    stop_response_id = rec.response_id
                if sys.exc_info()[0] is not None:
                    await _try_stop_stream(account.client, session.id, stop_response_id)
                await resp.aclose()
            if got_content:
                break
            if _is_retryable_error(rec) and attempt < MAX_RETRIES:
                log.warning(
                    "qwen retryable error (%s), attempt %d/%d",
                    error_code(rec.error),
                    attempt + 1,
                    MAX_RETRIES,
                )
                continue
            break

        assert rec is not None
        if _is_context_limit(rec) and not rec.has_content:
            _drop_session(pool, account, session_key)
            yield sse(
                {
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "error": {
                        "message": "context length exceeded: conversation too long, start a new conversation",
                        "code": "context_length_exceeded",
                    },
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": "length",
                        }
                    ],
                }
            )
            yield sse(
                {
                    "id": chunk_id,
                    "session_id": session_key,
                    "object": "chat.completion.chunk",
                    "choices": [],
                }
            )
            yield "data: [DONE]\n\n"
            return
        usage = _accumulate_usage(session, rec)
        account.sessions.touch_last_message(session_key, rec.response_id)

        if not rec.has_content and rec.error:
            err = rec.error
            code = error_code(rec.error)
            yield sse(
                {
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "error": {
                        "message": err.get("details") or err.get("message") or "Qwen server error, try again later",
                        "code": code,
                    },
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": code or "error",
                        }
                    ],
                }
            )
            yield sse(
                {
                    "id": chunk_id,
                    "session_id": session_key,
                    "object": "chat.completion.chunk",
                    "choices": [],
                }
            )
            yield "data: [DONE]\n\n"
            return

        if tool_mode:
            parsed = toolemu.parse_tool_calls(content_buf, tool_schemas)
            if parsed is not None:
                tool_calls, tool_text = parsed
                for delta in toolemu.tool_call_deltas(tool_calls, tool_text):
                    yield sse(
                        {
                            "id": chunk_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
                        }
                    )
                finish = "tool_calls"
            else:
                if content_buf:
                    yield sse(
                        {
                            "id": chunk_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": content_buf},
                                    "finish_reason": None,
                                }
                            ],
                        }
                    )
                finish = "stop"
        else:
            finish = "stop"

        yield sse(
            {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
            }
        )
        if include_usage:
            yield sse(
                {
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [],
                    "usage": usage,
                }
            )
        yield sse(
            {
                "id": chunk_id,
                "session_id": session_key,
                "object": "chat.completion.chunk",
                "choices": [],
            }
        )
        yield "data: [DONE]\n\n"
