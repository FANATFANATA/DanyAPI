from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from fastapi import HTTPException

from ..accounts import account_lock
from ..api.retry import MAX_RETRIES, _retry_delay
from ..api.shaping import _apply_limits
from ..api.sse import _sse, _stream_error_sse
from ..config import settings
from ..tokens import estimate_tokens
from ..usage import record_usage_dict
from .client import AUTH_REJECTED, CONNECT_FATAL, DEFAULT_MODEL, RETRYABLE_ERRORS, AliceError, fold_messages

log = logging.getLogger("danyapi.alice.api")

DONE_LINE = "data: [DONE]\n\n"
BROKEN_ERROR_CODES = {AUTH_REJECTED}


def _status_for(error: AliceError) -> int:
    if error.retryable or error.code in RETRYABLE_ERRORS or error.code == CONNECT_FATAL:
        return 502
    return 400


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


def _note_failure(account, exc: AliceError) -> None:
    if exc.code in BROKEN_ERROR_CODES:
        account.mark_broken()
        log.warning(
            "alice account #%d marked broken by upstream error %s: %s",
            getattr(account, "index", 0),
            exc.code,
            exc.message,
        )


async def _ask(account: Any, prompt: str) -> Any:
    attempt = 0
    while True:
        try:
            return await account.client.ask(prompt)
        except AliceError as exc:
            _note_failure(account, exc)
            if not exc.retryable or attempt >= MAX_RETRIES:
                raise
            attempt += 1
            await account.client.aclose()
            delay = _retry_delay(attempt)
            log.debug("alice request failed (%s), retry %d/%d in %.1fs", exc, attempt, MAX_RETRIES, delay)
            await asyncio.sleep(delay)


def _translation_error(exc: AliceError) -> HTTPException:
    return HTTPException(_status_for(exc), f"Alice error: {exc.message}")


async def collect_non_stream(
    account,
    messages=None,
    model: str = DEFAULT_MODEL,
    prompt: str | None = None,
    stop: Any = None,
    max_tokens: int | None = None,
    user: str | None = None,
    session_id: str | None = None,
) -> dict:
    text = prompt if prompt is not None else fold_messages(messages)
    async with account_lock(account.sem, settings.acquire_timeout):
        try:
            stream = await _ask(account, text)
        except AliceError as exc:
            raise _translation_error(exc) from exc
    content, finish = _apply_limits(stream.content, max_tokens, stop)
    usage = _usage_for(text, content)
    record_usage_dict("alice", model, usage, user=user, session_id=session_id)
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "system_fingerprint": stream.version or "fp_danyapi",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish,
                "logprobs": None,
            }
        ],
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
    messages=None,
    model: str = DEFAULT_MODEL,
    prompt: str | None = None,
    stop: Any = None,
    max_tokens: int | None = None,
    include_usage: bool = False,
    user: str | None = None,
    session_id: str | None = None,
) -> AsyncIterator[str]:
    chunk_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    text = prompt if prompt is not None else fold_messages(messages)

    content: str = ""
    finish = "stop"
    error_lines: tuple[str, str] | None = None
    async with account_lock(account.sem, settings.acquire_timeout):
        try:
            stream = await _ask(account, text)
        except AliceError as exc:
            error_lines = _stream_error_sse(chunk_id, created, model, f"Alice error: {exc.message}", session_id)
        else:
            content, finish = _apply_limits(stream.content, max_tokens, stop)

    if error_lines is not None:
        for line in error_lines:
            yield line
        return

    yield _chunk(chunk_id, created, model, {"role": "assistant", "content": content})
    yield _chunk(chunk_id, created, model, {}, finish)
    usage = _usage_for(text, content)
    record_usage_dict("alice", model, usage, user=user, session_id=session_id)
    if include_usage:
        yield _chunk(chunk_id, created, model, {}, None, usage=usage)
    yield DONE_LINE
