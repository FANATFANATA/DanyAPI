from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
import uuid

from fastapi import HTTPException

from .. import tools as toolemu
from ..accounts import AccountPoolBusy, account_lock
from ..config import settings
from ..deepseek.stream import IncrementalSSE, MessageReconstructor
from .deepseek import (
    MAX_RETRIES,
    RETRY_BACKOFF_MAX_SEC,
    RETRY_BACKOFF_SEC,
    _busy_error_body,
    _collect_continuation,
    _drop_session,
    _fresh_pow_headers,
    _human_delay,
    _input_exceeds_hint_from_http,
    _is_context_limit,
    _is_input_exceeds_limit,
    _is_retryable_hint,
    _is_retryable_http,
    _prepare_session,
    _send_with_auth,
    _try_stop_stream,
)
from .models import CONTEXT_LENGTH_STATUS, MAX_CONTINUE_ROUNDS, _deepseek_usage, _finish_reason

log = logging.getLogger("danyapi.api")


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


async def _collect_non_stream(
    account,
    pool,
    existing_sid,
    lock,
    prompt,
    model,
    model_type,
    thinking,
    search,
    ref_file_ids=None,
    tool_mode=False,
    tool_schemas=None,
    include_usage=False,
    context_seq: tuple[str, ...] | None = None,
):
    await _human_delay()
    async with account_lock(lock, settings.acquire_timeout):
        session, session_key, parent_message_id = await _prepare_session(account, pool, existing_sid, context_seq)
        stop_message_id: str | None = None
        started = time.monotonic()
        try:
            rec: MessageReconstructor | None = None
            response_message_id = None
            for attempt in range(MAX_RETRIES + 1):
                if attempt:
                    await asyncio.sleep(min(RETRY_BACKOFF_SEC * (2 ** (attempt - 1)), RETRY_BACKOFF_MAX_SEC))
                try:
                    pow_headers = await _fresh_pow_headers(account)
                    resp = await _send_with_auth(
                        account,
                        account.client,
                        pow_headers,
                        session.id,
                        parent_message_id,
                        prompt,
                        model_type,
                        thinking,
                        search,
                        ref_file_ids,
                    )
                except HTTPException as exc:
                    input_hint = _input_exceeds_hint_from_http(exc)
                    if input_hint is not None:
                        rec = MessageReconstructor()
                        rec.hint_error = input_hint
                        break
                    if _is_retryable_http(exc) and attempt < MAX_RETRIES:
                        log.warning(
                            "deepseek provider error (%s), attempt %d/%d",
                            exc.status_code,
                            attempt + 1,
                            MAX_RETRIES,
                        )
                        continue
                    raise
                rec = MessageReconstructor()
                incremental = IncrementalSSE()
                response_message_id = None
                try:
                    async for chunk in resp.aiter_bytes():
                        for event in incremental.feed(chunk):
                            if event.event == "ready" and isinstance(event.data, dict):
                                response_message_id = event.data.get("response_message_id")
                                if response_message_id:
                                    stop_message_id = response_message_id
                            rec.handle(event)
                    for event in incremental.finish():
                        rec.handle(event)
                finally:
                    await resp.aclose()
                if rec.id:
                    stop_message_id = rec.id
                if not (rec.content or rec.reasoning) and _is_retryable_hint(rec) and attempt < MAX_RETRIES:
                    log.warning(
                        "retryable hint (%s), attempt %d/%d",
                        (rec.hint_error or {}).get("finish_reason"),
                        attempt + 1,
                        MAX_RETRIES,
                    )
                    continue
                break
        except asyncio.CancelledError:
            await _try_stop_stream(account.client, session.id, stop_message_id)
            raise

        assert rec is not None
        if _is_context_limit(rec) and not (rec.content or rec.reasoning):
            _drop_session(pool, account, session_key)
            raise HTTPException(400, "context length exceeded: conversation too long, start a new conversation")
        if _is_input_exceeds_limit(rec):
            cont_parent = rec.id or response_message_id or parent_message_id
            for _ in range(MAX_CONTINUE_ROUNDS):
                cont_rec = await _collect_continuation(account, session, cont_parent, model_type, thinking, search, ref_file_ids)
                if cont_rec is None:
                    break
                rec.extend_with(cont_rec)
                cont_parent = cont_rec.id or cont_parent
                if not _is_input_exceeds_limit(cont_rec):
                    break
        session.accumulated_tokens = max(getattr(session, "accumulated_tokens", 0) or 0, rec.accumulated_tokens)
        account.sessions.touch_last_message(session_key, rec.id or response_message_id)
        log.info("deepseek completion OK (%.0fms)", (time.monotonic() - started) * 1000)

        if not (rec.content or rec.reasoning) and rec.hint_error:
            raise HTTPException(429, _busy_error_body(rec))

        content = rec.content
        reasoning = rec.reasoning
        if tool_mode:
            parsed = toolemu.parse_tool_calls(content, tool_schemas)
            if parsed is not None:
                tool_calls, tool_text = parsed
                finish = "tool_calls"
                message = toolemu.format_tool_message(tool_calls, tool_text, reasoning)
            else:
                message = {"role": "assistant", "content": content}
                if reasoning:
                    message["reasoning_content"] = reasoning
                finish = _finish_reason(rec.status)
        else:
            finish = _finish_reason(rec.status)
            message = {"role": "assistant", "content": content}
            if reasoning:
                message["reasoning_content"] = reasoning
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": finish,
                }
            ],
            "usage": _deepseek_usage(getattr(session, "accumulated_tokens", 0)),
            "session_id": session_key,
        }


async def _stream_openai(
    account,
    pool,
    existing_sid,
    lock,
    prompt,
    model,
    model_type,
    thinking,
    search,
    ref_file_ids=None,
    tool_mode=False,
    tool_schemas=None,
    include_usage=False,
    context_seq: tuple[str, ...] | None = None,
):
    chunk_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    await _human_delay()
    async with account_lock(lock, settings.acquire_timeout):
        try:
            session, session_key, parent_message_id = await _prepare_session(account, pool, existing_sid, context_seq)
        except HTTPException as exc:
            detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
            yield _sse(
                {
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "error": {"message": detail},
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "error"}],
                }
            )
            yield _sse(
                {
                    "id": chunk_id,
                    "session_id": None,
                    "object": "chat.completion.chunk",
                    "choices": [],
                }
            )
            yield "data: [DONE]\n\n"
            return

        rec: MessageReconstructor | None = None
        response_message_id = None
        stop_message_id: str | None = None
        content_buf = ""
        started = time.monotonic()
        for attempt in range(MAX_RETRIES + 1):
            if attempt:
                await asyncio.sleep(min(RETRY_BACKOFF_SEC * (2 ** (attempt - 1)), RETRY_BACKOFF_MAX_SEC))
            try:
                pow_headers = await _fresh_pow_headers(account)
                resp = await _send_with_auth(
                    account,
                    account.client,
                    pow_headers,
                    session.id,
                    parent_message_id,
                    prompt,
                    model_type,
                    thinking,
                    search,
                    ref_file_ids,
                )
            except HTTPException as exc:
                input_hint = _input_exceeds_hint_from_http(exc)
                if input_hint is not None:
                    rec = MessageReconstructor()
                    rec.hint_error = input_hint
                    break
                if _is_retryable_http(exc) and attempt < MAX_RETRIES:
                    log.warning(
                        "deepseek provider error (%s), attempt %d/%d",
                        exc.status_code,
                        attempt + 1,
                        MAX_RETRIES,
                    )
                    continue
                detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
                yield _sse(
                    {
                        "id": chunk_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "error": {"message": detail},
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "error"}],
                    }
                )
                yield _sse(
                    {
                        "id": chunk_id,
                        "session_id": session_key,
                        "object": "chat.completion.chunk",
                        "choices": [],
                    }
                )
                yield "data: [DONE]\n\n"
                return
            rec = MessageReconstructor()
            incremental = IncrementalSSE()
            response_message_id = None
            got_content = False
            pending: list[str] = []
            try:
                async for chunk in resp.aiter_bytes():
                    for event in incremental.feed(chunk):
                        if event.event == "ready" and isinstance(event.data, dict):
                            response_message_id = event.data.get("response_message_id")
                            if response_message_id:
                                stop_message_id = response_message_id
                        rec.handle(event)
                        c_diff, r_diff = rec.take_diffs()
                        if c_diff or r_diff:
                            got_content = True
                        if not pending:
                            pending.append(
                                _sse(
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
                                _sse(
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
                if rec.id:
                    stop_message_id = rec.id
                if sys.exc_info()[0] is not None:
                    await _try_stop_stream(account.client, session.id, stop_message_id)
                await resp.aclose()
            if got_content:
                break
            if _is_retryable_hint(rec) and attempt < MAX_RETRIES:
                log.warning(
                    "retryable hint (%s), attempt %d/%d",
                    (rec.hint_error or {}).get("finish_reason"),
                    attempt + 1,
                    MAX_RETRIES,
                )
                continue
            break

        assert rec is not None
        if _is_context_limit(rec) and not (rec.content or rec.reasoning):
            _drop_session(pool, account, session_key)
            yield _sse(
                {
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "error": {
                        "message": "context length exceeded: conversation too long, start a new conversation",
                        "finish_reason": CONTEXT_LENGTH_STATUS,
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
            yield _sse(
                {
                    "id": chunk_id,
                    "session_id": session_key,
                    "object": "chat.completion.chunk",
                    "choices": [],
                }
            )
            yield "data: [DONE]\n\n"
            return
        if _is_input_exceeds_limit(rec):
            cont_parent = rec.id or response_message_id or parent_message_id
            for _ in range(MAX_CONTINUE_ROUNDS):
                cont_rec = await _collect_continuation(account, session, cont_parent, model_type, thinking, search, ref_file_ids)
                if cont_rec is None:
                    break
                rec.extend_with(cont_rec)
                cont_parent = cont_rec.id or cont_parent
                if cont_rec.content:
                    if tool_mode:
                        content_buf += cont_rec.content
                    else:
                        yield _sse(
                            {
                                "id": chunk_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": model,
                                "choices": [{"index": 0, "delta": {"content": cont_rec.content}, "finish_reason": None}],
                            }
                        )
                if cont_rec.reasoning:
                    yield _sse(
                        {
                            "id": chunk_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [{"index": 0, "delta": {"reasoning_content": cont_rec.reasoning}, "finish_reason": None}],
                        }
                    )
                if not _is_input_exceeds_limit(cont_rec):
                    break
        if not (rec.content or rec.reasoning) and rec.hint_error:
            hint = rec.hint_error
            yield _sse(
                {
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "error": {
                        "message": hint.get("message") or "DeepSeek server is busy, try again later",
                        "finish_reason": hint.get("finish_reason"),
                    },
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": hint.get("finish_reason") or "error",
                        }
                    ],
                }
            )
            yield _sse(
                {
                    "id": chunk_id,
                    "session_id": session_key,
                    "object": "chat.completion.chunk",
                    "choices": [],
                }
            )
            yield "data: [DONE]\n\n"
            return

        session.accumulated_tokens = max(getattr(session, "accumulated_tokens", 0) or 0, rec.accumulated_tokens)
        account.sessions.touch_last_message(session_key, rec.id or response_message_id)
        log.info("deepseek completion OK (%.0fms)", (time.monotonic() - started) * 1000)

        if tool_mode:
            parsed = toolemu.parse_tool_calls(content_buf, tool_schemas)
            if parsed is not None:
                tool_calls, tool_text = parsed
                for delta in toolemu.tool_call_deltas(tool_calls, tool_text):
                    yield _sse(
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
                    yield _sse(
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
                finish = _finish_reason(rec.status)
        else:
            finish = _finish_reason(rec.status)

        yield _sse(
            {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
            }
        )
        if include_usage:
            yield _sse(
                {
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [],
                    "usage": _deepseek_usage(getattr(session, "accumulated_tokens", 0)),
                }
            )
        yield _sse(
            {
                "id": chunk_id,
                "session_id": session_key,
                "object": "chat.completion.chunk",
                "choices": [],
            }
        )
        yield "data: [DONE]\n\n"


async def _stream_guard(gen, model: str):
    try:
        async for item in gen:
            yield item
    except AccountPoolBusy:
        chunk_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        yield _sse(
            {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "error": {"message": "all accounts are busy, try again later"},
                "choices": [{"index": 0, "delta": {}, "finish_reason": "error"}],
            }
        )
        yield _sse(
            {
                "id": chunk_id,
                "session_id": None,
                "object": "chat.completion.chunk",
                "choices": [],
            }
        )
        yield "data: [DONE]\n\n"
