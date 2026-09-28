from __future__ import annotations

import logging
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from fastapi import HTTPException

from ..accounts import account_lock
from ..api.shaping import _apply_stop
from ..api.sse import _sse, _stream_error_sse
from ..config import settings
from ..tokens import estimate_tokens
from ..usage import record_usage_dict
from .client import RETRYABLE_ERRORS, AliceError, fold_messages

log = logging.getLogger("danyapi.alice.api")

MAX_RETRIES = 2
DONE_LINE = "data: [DONE]\n\n"

DEFAULT_MODEL = "alice"


def _status_for(error: AliceError) -> int:
    if error.code in RETRYABLE_ERRORS:
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


async def _ask(account: Any, prompt: str) -> Any:
    attempt = 0
    while True:
        try:
            return await account.client.ask(prompt)
        except AliceError as exc:
            if not exc.retryable or attempt >= MAX_RETRIES:
                raise
            attempt += 1
            await account.client.aclose()
            log.debug("alice request failed (%s), retrying", exc)


def _translation_error(exc: AliceError) -> HTTPException:
    return HTTPException(_status_for(exc), f"Alice error: {exc.message}")


async def collect_non_stream(
    account,
    messages=None,
    model: str = DEFAULT_MODEL,
    prompt: str | None = None,
    stop: Any = None,
    user: str | None = None,
    session_id: str | None = None,
) -> dict:
    text = prompt if prompt is not None else fold_messages(messages)
    async with account_lock(account.sem, settings.acquire_timeout):
        try:
            stream = await _ask(account, text)
        except AliceError as exc:
            raise _translation_error(exc) from exc
    content = _apply_stop(stream.content, stop)
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
                "finish_reason": "stop",
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
    include_usage: bool = False,
    user: str | None = None,
    session_id: str | None = None,
) -> AsyncIterator[str]:
    chunk_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    text = prompt if prompt is not None else fold_messages(messages)

    async with account_lock(account.sem, settings.acquire_timeout):
        try:
            stream = await _ask(account, text)
        except AliceError as exc:
            for line in _stream_error_sse(chunk_id, created, model, f"Alice error: {exc.message}", session_id):
                yield line
            return
        content = _apply_stop(stream.content, stop)
        yield _chunk(chunk_id, created, model, {"role": "assistant", "content": content})
        yield _chunk(chunk_id, created, model, {}, "stop")
        usage = _usage_for(text, content)
        record_usage_dict("alice", model, usage, user=user, session_id=session_id)
        if include_usage:
            yield _chunk(chunk_id, created, model, {}, None, usage=usage)
    yield DONE_LINE
