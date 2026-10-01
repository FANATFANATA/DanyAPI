from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import uuid
from collections.abc import AsyncIterator, Iterable, Iterator
from dataclasses import dataclass
from typing import Any

import httpx
from fastapi import HTTPException

from .. import tools as toolemu
from ..accounts import account_lock
from ..api.retry import (
    MAX_RETRIES,
    STALE_SESSION_STATUSES,
    _is_retryable_http,
    _retry_delay,
    _try_stop_stream,
)
from ..api.shaping import _apply_limits, _bounded_choices, _max_calls, _usage_with_details
from ..config import settings
from ..sseutil import IncrementalSSE, StreamStopFilter, split_stop
from ..tokens import StreamBudget, estimate_tokens, trim_to_tokens
from ..usage import record_usage_dict
from .client import QwenClient, QwenError
from .stream import QwenStreamReconstructor, error_code

log = logging.getLogger("danyapi.qwen.api")

AUTH_HTTP_STATUSES = {401, 403}

CONTEXT_LIMIT_MESSAGE = "context length exceeded: conversation too long, start a new conversation"

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

CONTEXT_LIMIT_MARKERS = (
    "context",
    "maxinput",
    "toolong",
    "lengthexceeded",
    "tokenlimit",
)

IMAGE_URI_SCHEMES = ("http://", "https://")
IMAGE_URI_FORBIDDEN = "()<>"
_BASE64_RE = re.compile(r"^[A-Za-z0-9+/]+={0,2}$")


def _valid_image_uri(uri: str) -> bool:
    if not uri or any(char.isspace() or char in IMAGE_URI_FORBIDDEN for char in uri):
        return False
    if uri.startswith(IMAGE_URI_SCHEMES):
        return True
    if not uri.startswith("data:"):
        return False
    meta, _, payload = uri[5:].partition(",")
    if not meta.endswith(";base64") or len(payload) % 4:
        return False
    return _BASE64_RE.match(payload) is not None


def _iter_image_uris(messages: list[Any]) -> Iterator[str]:
    for message in messages:
        content = getattr(message, "content", None)
        if not isinstance(content, list):
            continue
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
            if _valid_image_uri(uri):
                yield uri


def _append_image_markdown(prompt: str, messages: list[Any] | None) -> str:
    if not messages:
        return prompt
    appended: list[str] = []
    seen: set[str] = set()
    for uri in _iter_image_uris(messages):
        tag = f"![image]({uri})"
        if tag in seen or tag in prompt:
            continue
        seen.add(tag)
        appended.append(tag)
    if not appended:
        return prompt
    extra = "\n".join(appended)
    return f"{prompt}\n\n{extra}" if prompt else extra


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
    if pool is not None:
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
        log.warning(
            "qwen account #%d auth error %s: %s",
            getattr(account, "index", 0),
            code,
            exc,
        )
    else:
        log.warning("qwen account #%d error: %s", getattr(account, "index", 0), exc)


async def _prepare_session(
    account,
    pool,
    existing_sid: str | None,
    model_id: str,
    context_seq: tuple[str, ...] | None = None,
):
    try:
        session, session_key = await account.sessions.obtain(existing_sid, model_id)
    except QwenError as exc:
        _handle_account_error(account, exc)
        raise HTTPException(_error_status(exc.code), f"Qwen error: {exc}") from exc
    if pool is not None:
        pool.register(account.index, session_key)
        pool.index_context(session_key, context_seq)
    return session, session_key


async def _send_completion(
    client: QwenClient,
    session,
    prompt: str,
    model_id: str,
    thinking: bool,
    search: bool,
    chat_type: str = "t2t",
):
    try:
        resp = await client.completion(
            chat_session_id=session.id,
            prompt=prompt,
            parent_message_id=session.last_response_id,
            model=model_id,
            thinking=thinking,
            search=search,
            chat_type=chat_type,
        )
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
                raise HTTPException(
                    _error_status(code),
                    error.get("message") or error.get("details") or "Qwen error",
                )
            data = payload.get("data")
            if isinstance(data, dict) and data.get("code"):
                if _is_context_limit_code(data["code"]):
                    raise ContextLimitError() from None
                raise HTTPException(
                    _error_status(data["code"]),
                    data.get("details") or data.get("message") or "Qwen error",
                )
        raise HTTPException(502, f"Qwen request failed: {text}")
    return resp


def _is_retryable_error(rec: QwenStreamReconstructor) -> bool:
    return bool(rec.error and error_code(rec.error) in RETRYABLE_ERROR_CODES and not rec.has_content)


def _error_detail(rec: QwenStreamReconstructor) -> str:
    err = rec.error or {}
    return str(err.get("details") or err.get("message") or "Qwen server error, try again later")


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def _stream_error_lines(
    chunk_id: str,
    created: int,
    model: str,
    message: str,
    session_key: str | None = None,
    code: str | None = None,
    finish_reason: str = "error",
) -> Iterator[str]:
    error: dict = {"message": message}
    if code:
        error["code"] = code
    payload = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "error": error,
        "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
    }
    if session_key:
        payload["session_id"] = session_key
    yield _sse(payload)
    yield "data: [DONE]\n\n"


def _stream_context_limit_lines(chunk_id: str, created: int, model: str, session_key: str | None = None) -> Iterator[str]:
    yield from _stream_error_lines(
        chunk_id,
        created,
        model,
        "context length exceeded: conversation too long, start a new conversation",
        session_key,
        "context_length_exceeded",
        "length",
    )


def _accumulate_usage(session, rec: QwenStreamReconstructor, prompt: str = "", completion_text: str | None = None) -> dict:
    current = rec.usage_tokens
    current_input = current["prompt_tokens"]
    current_output = current["completion_tokens"]
    prev_input = int(getattr(session, "accumulated_input_tokens", 0) or 0)
    prev_output = int(getattr(session, "accumulated_output_tokens", 0) or 0)
    prompt_tokens = max(0, current_input - prev_input)
    completion_tokens = max(0, current_output - prev_output)
    session.accumulated_input_tokens = max(prev_input, current_input)
    session.accumulated_output_tokens = max(prev_output, current_output)
    if not completion_tokens and completion_text:
        completion_tokens = estimate_tokens(completion_text)
        prompt_tokens = max(prompt_tokens, estimate_tokens(prompt) if prompt else estimate_tokens(completion_text))
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def _commit_usage(account, session, session_key: str | None, rec: QwenStreamReconstructor, model: str, prompt: str, user) -> dict:
    usage = _accumulate_usage(session, rec, prompt, completion_text=rec.content or rec.reasoning)
    account.sessions.touch_last_message(session_key, rec.response_id)
    record_usage_dict("qwen", model, usage, user=user, session_id=session_key)
    return usage


def _build_limited_message(
    rec: QwenStreamReconstructor,
    tool_mode: bool,
    tool_schemas: Any,
    max_tokens: int | None,
    stop: Any,
    parallel_tool_calls: bool | None,
) -> tuple[dict, str]:
    message: dict
    if tool_mode:
        parsed = toolemu.parse_tool_calls(rec.content, tool_schemas)
        if parsed is not None:
            tool_calls, tool_text = parsed
            if tool_calls:
                if _max_calls(parallel_tool_calls) is not None:
                    tool_calls = tool_calls[: _max_calls(parallel_tool_calls)]
                message = toolemu.format_tool_message(tool_calls, tool_text, rec.reasoning)
                tail = message.get("content")
                if isinstance(tail, str):
                    trimmed_tail = trim_to_tokens(tail, max_tokens)
                    if trimmed_tail != tail:
                        message["content"] = trimmed_tail
                        return message, "length"
                return message, "tool_calls"
        text, limit_finish = _apply_limits(toolemu.strip_dsml(rec.content or ""), max_tokens, stop)
        message = {"role": "assistant", "content": text}
        if rec.reasoning:
            message["reasoning_content"] = toolemu.strip_dsml(rec.reasoning)
        if limit_finish == "length":
            return message, "length"
        return message, "stop"
    text, limit_finish = _apply_limits(toolemu.strip_dsml(rec.content or ""), max_tokens, stop)
    message = {"role": "assistant", "content": text}
    if rec.reasoning:
        message["reasoning_content"] = toolemu.strip_dsml(rec.reasoning)
    return message, limit_finish


@dataclass
class _CollectRequest:
    account: Any
    pool: Any
    existing_sid: str | None
    model_id: str
    context_seq: tuple[str, ...] | None
    messages: list[Any] | None
    tools: Any
    tool_choice: Any
    response_format: Any
    cached_session: Any
    prompt: str
    tool_mode: bool
    tool_schemas: Any
    thinking: bool = False
    search: bool = False
    chat_type: str = "t2t"
    session: Any = None
    session_key: str | None = None
    prompt_with_images: str = ""
    had_cached_session: bool = False
    prepared: bool = False

    def rebuild_prompt(self) -> None:
        if self.messages is None:
            return
        self.prompt, self.tool_mode = toolemu.build_prompt(self.messages, self.tools, self.tool_choice, False, self.response_format)
        self.tool_schemas = toolemu.tool_schema_map(self.tools)
        self.prompt_with_images = _append_image_markdown(self.prompt, self.messages)


def _make_request(
    account,
    pool,
    existing_sid,
    prompt: str,
    model_id: str,
    thinking: bool,
    search: bool,
    tool_mode: bool,
    tool_schemas: Any,
    context_seq: tuple[str, ...] | None,
    messages: list[Any] | None,
    tools: Any,
    tool_choice: Any,
    response_format: Any,
    cached_session: Any,
    chat_type: str = "t2t",
) -> _CollectRequest:
    return _CollectRequest(
        account=account,
        pool=pool,
        existing_sid=existing_sid,
        model_id=model_id,
        context_seq=context_seq,
        messages=messages,
        tools=tools,
        tool_choice=tool_choice,
        response_format=response_format,
        cached_session=cached_session,
        prompt=prompt,
        tool_mode=tool_mode,
        tool_schemas=tool_schemas,
        thinking=thinking,
        search=search,
        chat_type=chat_type,
        prompt_with_images=_append_image_markdown(prompt, messages),
    )


async def _ensure_prepared(req: _CollectRequest) -> None:
    if req.prepared:
        return
    req.had_cached_session = bool(req.existing_sid) and req.account.sessions.get(req.existing_sid) is not None
    req.session, req.session_key = await _prepare_session(req.account, req.pool, req.existing_sid, req.model_id, req.context_seq)
    req.prepared = True
    if req.messages is not None and (req.session_key != req.existing_sid or req.session is not req.cached_session):
        try:
            req.rebuild_prompt()
        except ValueError:
            pass


def _detail_text(exc: HTTPException) -> str:
    return exc.detail if isinstance(exc.detail, str) else str(exc.detail)


async def _collect_response(req: _CollectRequest, lock, lock_timeout) -> QwenStreamReconstructor:
    stop_response_id: str | None = None
    stale_rebuilt = False
    attempt = 0
    rec: QwenStreamReconstructor | None = None
    try:
        while True:
            delay = 0.0
            result: QwenStreamReconstructor | None = None
            async with account_lock(lock, lock_timeout):
                await _ensure_prepared(req)
                try:
                    resp = await _send_completion(
                        req.account.client,
                        req.session,
                        req.prompt_with_images,
                        req.model_id,
                        req.thinking,
                        req.search,
                        req.chat_type,
                    )
                except ContextLimitError:
                    _drop_session(req.pool, req.account, req.session_key)
                    raise HTTPException(400, CONTEXT_LIMIT_MESSAGE) from None
                except HTTPException as exc:
                    if exc.status_code in AUTH_HTTP_STATUSES:
                        req.account.mark_broken()
                    if exc.status_code in STALE_SESSION_STATUSES and req.had_cached_session and not stale_rebuilt and req.messages is not None:
                        stale_rebuilt = True
                        _drop_session(req.pool, req.account, req.session_key)
                        try:
                            req.rebuild_prompt()
                        except ValueError as build_exc:
                            raise exc from build_exc
                        log.warning(
                            "qwen chat %s is stale (%s), rebuilt full history into a fresh chat",
                            req.session_key,
                            exc.status_code,
                        )
                        req.session, req.session_key = await _prepare_session(req.account, req.pool, req.existing_sid, req.model_id, req.context_seq)
                        stop_response_id = None
                    elif _is_retryable_http(exc) and attempt < MAX_RETRIES:
                        attempt += 1
                        delay = _retry_delay(attempt)
                        log.warning(
                            "qwen provider error (%s), retry %d/%d in %.1fs",
                            exc.status_code,
                            attempt,
                            MAX_RETRIES,
                            delay,
                        )
                    else:
                        raise
                else:
                    rec = QwenStreamReconstructor()
                    incremental = IncrementalSSE()
                    try:
                        async for chunk in resp.aiter_bytes():
                            for event in incremental.feed(chunk):
                                rec.handle(event)
                        for event in incremental.finish():
                            rec.handle(event)
                        rec.finalize()
                    except (httpx.HTTPError, RuntimeError, ValueError) as exc:
                        raise HTTPException(502, f"Stream processing failed: {exc}") from exc
                    finally:
                        if rec.response_id:
                            stop_response_id = rec.response_id
                        try:
                            await resp.aclose()
                        except Exception as exc:
                            log.debug("response close failed: %s", exc)
                    if _is_retryable_error(rec) and attempt < MAX_RETRIES:
                        attempt += 1
                        delay = _retry_delay(attempt)
                        log.warning(
                            "qwen retryable error (%s), retry %d/%d in %.1fs",
                            error_code(rec.error),
                            attempt,
                            MAX_RETRIES,
                            delay,
                        )
                        await _try_stop_stream(req.account.client, req.session.id, stop_response_id)
                    else:
                        result = rec
            if result is None:
                if delay > 0:
                    await asyncio.sleep(delay)
                continue
            return result
    except BaseException:
        if rec is not None and rec.response_id:
            stop_response_id = rec.response_id
        session_id = str(req.session.id) if req.session is not None else ""
        await _try_stop_stream(req.account.client, session_id, stop_response_id)
        raise


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
    context_seq: tuple[str, ...] | None = None,
    messages=None,
    tools=None,
    tool_choice=None,
    response_format=None,
    user=None,
    max_tokens: int | None = None,
    stop: Any = None,
    n: int | None = None,
    parallel_tool_calls: bool | None = None,
    cached_session=None,
):
    req = _make_request(
        account,
        pool,
        existing_sid,
        prompt,
        model_id,
        thinking,
        search,
        tool_mode,
        tool_schemas,
        context_seq,
        messages,
        tools,
        tool_choice,
        response_format,
        cached_session,
    )
    rec = await _collect_response(req, lock, settings.acquire_timeout)
    async with account_lock(lock, settings.acquire_timeout):
        if _is_context_limit(rec):
            _drop_session(pool, account, req.session_key)
            raise HTTPException(400, CONTEXT_LIMIT_MESSAGE)
        usage = _commit_usage(account, req.session, req.session_key, rec, model, req.prompt, user)

        if not rec.has_content and rec.error:
            raise HTTPException(_error_status(error_code(rec.error)), _error_detail(rec))

        message, finish = _build_limited_message(rec, req.tool_mode, req.tool_schemas, max_tokens, stop, parallel_tool_calls)
        response = {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "system_fingerprint": "fp_danyapi",
            "choices": [{"index": 0, "message": message, "finish_reason": finish, "logprobs": None}],
            "usage": _usage_with_details(usage, rec.reasoning),
            "session_id": req.session_key,
        }
        choices = _bounded_choices(n)
        if choices > 1:
            template = response["choices"][0]
            response["choices"] = [dict(template) | {"index": i} for i in range(choices)]
        return response


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
    messages=None,
    tools=None,
    tool_choice=None,
    response_format=None,
    user=None,
    max_tokens: int | None = None,
    stop: Any = None,
    n: int | None = None,
    parallel_tool_calls: bool | None = None,
    cached_session=None,
):
    chunk_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    req = _make_request(
        account,
        pool,
        existing_sid,
        prompt,
        model_id,
        thinking,
        search,
        tool_mode,
        tool_schemas,
        context_seq,
        messages,
        tools,
        tool_choice,
        response_format,
        cached_session,
    )

    content_buf = ""
    content_shown_len = 0
    tool_hidden = False
    role_sent = False
    got_content = False
    budget = StreamBudget(max_tokens, trim_to_tokens)
    stop_markers = split_stop(stop)
    stop_hit = False
    dsml_filter = toolemu.DsmlFilter()
    stop_filter = StreamStopFilter(stop_markers) if stop_markers else None
    reasoning_filter = toolemu.DsmlFilter()

    def content_piece(piece: str | None) -> str:
        nonlocal stop_hit
        if stop_hit:
            return ""
        text = dsml_filter.feed(piece)
        if stop_filter is None:
            return budget.feed(text)
        filtered, hit = stop_filter.feed(text)
        if hit:
            stop_hit = True
        return budget.feed(filtered)

    def flush_piece() -> str:
        nonlocal stop_hit
        if stop_hit:
            return ""
        text = dsml_filter.flush()
        if stop_filter is None:
            return budget.feed(text)
        filtered, hit = stop_filter.feed(text)
        if hit:
            stop_hit = True
            return budget.feed(filtered)
        return budget.feed(filtered) + budget.feed(stop_filter.flush())

    def done_finish() -> str:
        if stop_hit:
            return "stop"
        return "length" if budget.done else "stop"

    def reasoning_piece(piece: str | None) -> str:
        return reasoning_filter.feed(piece)

    def flush_reasoning() -> str:
        return reasoning_filter.flush()

    def role_line() -> str:
        return _sse(
            {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
            }
        )

    def delta_line(delta: dict, index: int = 0) -> str:
        return _sse(
            {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": index, "delta": delta, "finish_reason": None}],
            }
        )

    def finish_line(finish: str, index: int = 0) -> str:
        payload = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": index, "delta": {}, "finish_reason": finish}],
        }
        if req.session_key:
            payload["session_id"] = req.session_key
        return _sse(payload)

    def extra_choice_lines(finish: str, count: int, tool_deltas: list[dict]) -> Iterator[str]:
        for extra_index in range(1, count):
            if tool_deltas:
                for delta in tool_deltas:
                    yield delta_line(delta, extra_index)
            elif budget.text:
                yield delta_line({"content": budget.text}, extra_index)
            yield finish_line(finish, extra_index)

    async def pump(source: QwenStreamReconstructor, events: Iterable[Any]) -> AsyncIterator[str]:
        nonlocal content_buf, content_shown_len, tool_hidden, got_content, role_sent
        for event in events:
            source.handle(event)
            c_diff, r_diff = source.take_diffs()
            if not (c_diff or r_diff):
                continue
            delta: dict = {}
            if c_diff:
                if req.tool_mode:
                    content_buf += c_diff
                    shown, content_shown_len, tool_hidden = toolemu.tool_visible(content_buf, content_shown_len, tool_hidden, req.tool_schemas)
                    allowed = content_piece(shown)
                    if allowed:
                        delta["content"] = allowed
                else:
                    allowed = content_piece(c_diff)
                    if allowed:
                        delta["content"] = allowed
            if r_diff:
                reason = reasoning_piece(r_diff)
                if reason:
                    delta["reasoning_content"] = reason
            if not delta:
                continue
            if not role_sent:
                role_sent = True
                yield role_line()
            got_content = True
            yield delta_line(delta)

    stop_response_id: str | None = None
    stale_rebuilt = False
    attempt = 0
    rec: QwenStreamReconstructor
    while True:
        delay = 0.0
        async with account_lock(lock, settings.acquire_timeout):
            try:
                await _ensure_prepared(req)
            except HTTPException as exc:
                for line in _stream_error_lines(chunk_id, created, model, _detail_text(exc), req.session_key):
                    yield line
                return
            try:
                resp = await _send_completion(
                    account.client,
                    req.session,
                    req.prompt_with_images,
                    model_id,
                    thinking,
                    search,
                )
            except ContextLimitError:
                _drop_session(pool, account, req.session_key)
                for line in _stream_context_limit_lines(chunk_id, created, model, req.session_key):
                    yield line
                return
            except HTTPException as exc:
                if exc.status_code in AUTH_HTTP_STATUSES:
                    account.mark_broken()
                if exc.status_code in STALE_SESSION_STATUSES and req.had_cached_session and not stale_rebuilt and messages is not None:
                    stale_rebuilt = True
                    _drop_session(pool, account, req.session_key)
                    try:
                        req.rebuild_prompt()
                    except ValueError:
                        for line in _stream_error_lines(chunk_id, created, model, _detail_text(exc), req.session_key):
                            yield line
                        return
                    log.warning(
                        "qwen chat %s is stale (%s), rebuilt full history into a fresh chat",
                        req.session_key,
                        exc.status_code,
                    )
                    try:
                        req.session, req.session_key = await _prepare_session(account, pool, existing_sid, model_id, context_seq)
                    except HTTPException as prep_exc:
                        for line in _stream_error_lines(chunk_id, created, model, _detail_text(prep_exc), req.session_key):
                            yield line
                        return
                    stop_response_id = None
                    continue
                if _is_retryable_http(exc) and attempt < MAX_RETRIES:
                    attempt += 1
                    delay = _retry_delay(attempt)
                    log.warning(
                        "qwen provider error (%s), retry %d/%d in %.1fs",
                        exc.status_code,
                        attempt,
                        MAX_RETRIES,
                        delay,
                    )
                else:
                    for line in _stream_error_lines(chunk_id, created, model, _detail_text(exc), req.session_key):
                        yield line
                    return
            else:
                rec = QwenStreamReconstructor()
                incremental = IncrementalSSE()
                got_content = False
                role_sent = False
                content_buf = ""
                content_shown_len = 0
                tool_hidden = False
                budget = StreamBudget(max_tokens, trim_to_tokens)
                stopped = False
                stop_hit = False
                dsml_filter = toolemu.DsmlFilter()
                stop_filter = StreamStopFilter(stop_markers) if stop_markers else None
                reasoning_filter = toolemu.DsmlFilter()
                try:
                    async for chunk in resp.aiter_bytes():
                        async for line in pump(rec, incremental.feed(chunk)):
                            yield line
                    for event in incremental.finish():
                        async for line in pump(rec, [event]):
                            yield line
                    rec.finalize()
                except BaseException:
                    stopped = True
                    if rec.response_id:
                        stop_response_id = rec.response_id
                    await _try_stop_stream(account.client, req.session.id, stop_response_id)
                    raise
                finally:
                    if rec.response_id:
                        stop_response_id = rec.response_id
                    try:
                        await resp.aclose()
                    except Exception as exc:
                        log.debug("response close failed: %s", exc)
                        if not stopped:
                            await _try_stop_stream(account.client, req.session.id, stop_response_id)
                if got_content:
                    break
                if _is_retryable_error(rec) and attempt < MAX_RETRIES:
                    attempt += 1
                    delay = _retry_delay(attempt)
                    log.warning(
                        "qwen retryable error (%s), retry %d/%d in %.1fs",
                        error_code(rec.error),
                        attempt,
                        MAX_RETRIES,
                        delay,
                    )
                    await _try_stop_stream(account.client, req.session.id, stop_response_id)
                else:
                    break
        if delay > 0:
            await asyncio.sleep(delay)

    usage: dict = {}
    context_limited = False
    async with account_lock(lock, settings.acquire_timeout):
        if _is_context_limit(rec):
            _drop_session(pool, account, req.session_key)
            context_limited = True
        else:
            usage = _commit_usage(account, req.session, req.session_key, rec, model, req.prompt, user)
    if context_limited:
        for line in _stream_context_limit_lines(chunk_id, created, model, req.session_key):
            yield line
        return

    if not rec.has_content and rec.error:
        for line in _stream_error_lines(
            chunk_id,
            created,
            model,
            _error_detail(rec),
            req.session_key,
            error_code(rec.error),
        ):
            yield line
        return

    tool_deltas: list[dict] = []
    if req.tool_mode:
        parsed = toolemu.parse_tool_calls(content_buf, req.tool_schemas)
        tool_calls = parsed[0] if parsed is not None else []
        if tool_calls:
            if _max_calls(parallel_tool_calls) is not None:
                tool_calls = tool_calls[: _max_calls(parallel_tool_calls)]
            tool_deltas = toolemu.tool_call_deltas(tool_calls)
            for delta in tool_deltas:
                yield delta_line(delta)
            finish = "tool_calls"
        else:
            remainder = content_piece(content_buf[content_shown_len:])
            if remainder:
                yield delta_line({"content": remainder})
            finish = done_finish()
    else:
        finish = done_finish()

    if not tool_deltas:
        remainder = flush_piece()
        if remainder:
            yield delta_line({"content": remainder})

    reason_tail = flush_reasoning()
    if reason_tail:
        yield delta_line({"reasoning_content": reason_tail})

    if not role_sent:
        role_sent = True
        yield role_line()

    yield finish_line(finish)
    for line in extra_choice_lines(finish, _bounded_choices(n), tool_deltas):
        yield line
    if include_usage:
        usage_payload = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "usage": _usage_with_details(usage, rec.reasoning),
            "choices": [],
        }
        if req.session_key:
            usage_payload["session_id"] = req.session_key
        yield _sse(usage_payload)
    yield "data: [DONE]\n\n"


async def collect_image(
    account,
    pool,
    existing_sid,
    lock,
    prompt,
    model,
    model_id,
    context_seq: tuple[str, ...] | None = None,
    user=None,
):
    req = _make_request(
        account,
        pool,
        existing_sid,
        prompt,
        model_id,
        False,
        False,
        False,
        None,
        context_seq,
        None,
        None,
        None,
        None,
        None,
        "t2i",
    )
    rec = await _collect_response(req, lock, settings.acquire_timeout)
    async with account_lock(lock, settings.acquire_timeout):
        if _is_context_limit(rec):
            _drop_session(pool, account, req.session_key)
            raise HTTPException(400, CONTEXT_LIMIT_MESSAGE)
        usage = _commit_usage(account, req.session, req.session_key, rec, model, req.prompt, user)

        if not rec.has_content and rec.error:
            raise HTTPException(_error_status(error_code(rec.error)), _error_detail(rec))

        return {
            "image_urls": rec.image_urls,
            "revised_prompt": toolemu.strip_dsml(rec.content or ""),
            "usage": usage,
            "session_id": req.session_key,
        }
