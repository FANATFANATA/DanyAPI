from __future__ import annotations

import asyncio
import json
import logging
import random
import time
import uuid
from collections.abc import Iterator
from typing import Any

import httpx
from fastapi import HTTPException

from .. import tools as toolemu
from ..accounts import AccountPool, DeepSeekAccount, account_lock
from ..config import settings
from ..deepseek.client import DeepSeekError, DeepSeekSession
from ..deepseek.stream import IncrementalSSE, MessageReconstructor
from ..sseutil import StreamStopFilter, split_stop
from ..tokens import StreamBudget, estimate_tokens, trim_to_tokens
from ..usage import record_usage
from .attachments import _upload_attachments
from .core import _error_detail
from .models import _finish_reason, _output_truncated
from .powauth import (
    DEEPSEEK_AUTH_ERROR_CODES,
    _deepseek_error_detail,
    _deepseek_status,
    _drop_session,
    _fresh_pow_headers,
    _handle_account_error,
)
from .schemas import DeepSeekStreamError
from .shaping import (
    _advance_session_usage,
    _apply_limits,
    _apply_stop,
    _bounded_choices,
    _deepseek_usage,
    _max_calls,
    _usage_with_details,
)
from .sse import _JSON_ENCODE, _delta_json, _sse, _stream_error_sse

log = logging.getLogger("danyapi.api")


STATUS_TO_FINISH_REASON = {
    "FINISHED": "stop",
    "CONTEXT_LENGTH_EXCEEDED": "length",
    "CONTENT_FILTER": "content_filter",
    "INCOMPLETE": "length",
    "WIP": "length",
    "TIMEOUT": "length",
}

CONTEXT_LENGTH_STATUS = "CONTEXT_LENGTH_EXCEEDED"
INPUT_EXCEEDS_LIMIT = "input_exceeds_limit"
CONTINUE_PROMPT = "Continue"
MAX_CONTINUE_ROUNDS = 5
CONTINUE_DEADLINE_SEC = 120.0
MAX_ERROR_BODY_CHARS = 4096
RESPONSE_INCOMPLETE = "response_incomplete"
RESPONSE_INCOMPLETE_MESSAGE = "Response is incomplete: provider errors interrupted the continuation, please retry"
REDUCED_CONTEXT_MESSAGE = "Response was generated from reduced context because the original input exceeded the model limit and may be incomplete"


SYSTEM_FINGERPRINT = "fp_danyapi"


_UNSET: Any = object()


RETRYABLE_FINISH_REASONS = {
    "expert_busy_use_default",
    "parallel_chat_limit",
    "server_busy",
    "busy",
}
MAX_RETRIES = 5
RETRY_BACKOFF_SEC = 1.0
RETRY_BACKOFF_MAX_SEC = 8.0


MESSAGE_TOO_FREQUENT_MARKERS = ("messagetoofrequent", "messagetofrequent")
MESSAGE_TOO_FREQUENT_WAIT_SEC = 60.0
MESSAGE_TOO_FREQUENT_MAX_RETRIES = 5


async def _human_delay() -> None:
    delay = random.uniform(settings.human_delay_min, settings.human_delay_max)
    if delay > 0:
        await asyncio.sleep(delay)


async def _prepare_session(
    account: DeepSeekAccount,
    pool: AccountPool,
    existing_sid: str | None,
    context_seq: tuple[str, ...] | None = None,
) -> tuple[DeepSeekSession, str, str | None]:
    try:
        session, session_key = await account.sessions.obtain(existing_sid)
    except DeepSeekError as exc:
        _handle_account_error(account, exc)
        raise HTTPException(_deepseek_status(exc), _deepseek_error_detail(exc)) from exc
    pool.register(account.index, session_key)
    if context_seq:
        pool.index_context(session_key, context_seq)
    return session, session_key, session.last_message_id


async def _send_completion(
    client,
    pow_headers,
    session_id,
    parent_message_id,
    prompt,
    model_type,
    thinking,
    search,
    ref_file_ids=None,
):
    try:
        resp = await client.completion(
            chat_session_id=session_id,
            prompt=prompt,
            parent_message_id=parent_message_id,
            model_type=model_type,
            thinking_enabled=thinking,
            search_enabled=search,
            ref_file_ids=ref_file_ids,
            pow_headers=pow_headers,
        )
    except httpx.HTTPError as exc:
        raise HTTPException(502, f"DeepSeek request failed: {exc}") from exc

    if resp is None or not hasattr(resp, "status_code"):
        raise HTTPException(502, "unexpected provider response")

    if resp.status_code != 200:
        body = await resp.aread()
        await resp.aclose()
        text = body[:MAX_ERROR_BODY_CHARS].decode("utf-8", errors="replace")
        if _message_too_frequent_text(text) is not None:
            raise HTTPException(429, text)
        raise HTTPException(resp.status_code, text)

    content_type = resp.headers.get("content-type", "")
    if "text/event-stream" not in content_type:
        body = await resp.aread()
        await resp.aclose()
        try:
            payload = json.loads(body)
        except json.JSONDecodeError as exc:
            text = body[:MAX_ERROR_BODY_CHARS].decode("utf-8", errors="replace")
            if _message_too_frequent_text(text) is not None:
                raise HTTPException(429, text) from exc
            raise HTTPException(502, text) from exc
        data = payload.get("data") or {}
        if data.get("biz_code"):
            code = data["biz_code"]
            detail = f"DeepSeek error {code}: {data.get('biz_msg')}"
            status = 401 if code in DEEPSEEK_AUTH_ERROR_CODES else 502
            if _message_too_frequent_text(detail) is not None:
                status = 429
            raise HTTPException(status, detail)
        if payload.get("code"):
            code = payload["code"]
            detail = f"DeepSeek error {code}: {payload.get('msg') or payload.get('message')}"
            status = 401 if code in DEEPSEEK_AUTH_ERROR_CODES else 502
            if _message_too_frequent_text(detail) is not None:
                status = 429
            raise HTTPException(status, detail)
        raise HTTPException(502, "unexpected non-stream response")
    return resp


def _is_retryable_hint(rec: MessageReconstructor) -> bool:
    hint = rec.hint_error
    return bool(hint and hint.get("finish_reason") in RETRYABLE_FINISH_REASONS)


FAKE_CONTEXT_HINT_MARKERS = ("length limit reached",)


def _is_fake_context_hint(rec: MessageReconstructor) -> bool:
    hint = rec.hint_error
    if not hint:
        return False
    message = hint.get("message")
    if not isinstance(message, str):
        return False
    return any(marker in message.casefold() for marker in FAKE_CONTEXT_HINT_MARKERS)


RETRYABLE_HTTP_STATUSES = {408, 425, 429, 500, 502, 503, 504}
STALE_SESSION_STATUSES = {400, 404}


def _retry_delay(attempt: int) -> float:
    return min(RETRY_BACKOFF_SEC * (2 ** (attempt - 1)), RETRY_BACKOFF_MAX_SEC)


def _compact_error_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return "".join(ch for ch in value.casefold() if ch.isalnum())


def _error_text(detail: Any) -> str:
    if isinstance(detail, str):
        return detail
    try:
        return json.dumps(detail, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(detail)


def _message_too_frequent_text(*values: Any) -> str | None:
    for value in values:
        compact = _compact_error_text(value)
        if not compact:
            continue
        if any(marker in compact for marker in MESSAGE_TOO_FREQUENT_MARKERS):
            return value
    return None


def _is_message_too_frequent_http(exc: HTTPException) -> bool:
    return _message_too_frequent_text(_error_text(exc.detail)) is not None


def _is_message_too_frequent_hint(rec: MessageReconstructor) -> bool:
    hint = rec.hint_error
    if not hint:
        return False
    return _message_too_frequent_text(hint.get("message"), hint.get("finish_reason")) is not None


async def _wait_message_too_frequent(stage: str, attempt: int) -> None:
    log.warning(
        "deepseek message too frequent (%s), retry %d/%d in %.0fs",
        stage,
        attempt,
        MESSAGE_TOO_FREQUENT_MAX_RETRIES,
        MESSAGE_TOO_FREQUENT_WAIT_SEC,
    )
    await asyncio.sleep(MESSAGE_TOO_FREQUENT_WAIT_SEC)


def _is_retryable_http(exc: HTTPException) -> bool:
    return exc.status_code in RETRYABLE_HTTP_STATUSES


def _is_context_limit(rec: MessageReconstructor) -> bool:
    if rec.status == CONTEXT_LENGTH_STATUS:
        return True
    hint = rec.hint_error
    return bool(hint and hint.get("finish_reason") == CONTEXT_LENGTH_STATUS)


def _is_input_exceeds_limit(rec: MessageReconstructor) -> bool:
    if rec.status == INPUT_EXCEEDS_LIMIT:
        return True
    hint = rec.hint_error
    return bool(hint and hint.get("finish_reason") == INPUT_EXCEEDS_LIMIT)


def _incomplete_message(rec: MessageReconstructor) -> str:
    hint = rec.hint_error or {}
    message = hint.get("message")
    return message if isinstance(message, str) and message else RESPONSE_INCOMPLETE_MESSAGE


def _incomplete_error_body(message: str) -> dict:
    return _error_detail(message, RESPONSE_INCOMPLETE)


def _input_exceeds_hint_from_http(exc: HTTPException) -> dict | None:
    detail = exc.detail
    if isinstance(detail, str):
        try:
            detail = json.loads(detail)
        except json.JSONDecodeError:
            return None
    if not isinstance(detail, dict):
        return None
    for node in (detail, detail.get("error"), detail.get("data")):
        if not isinstance(node, dict) or node.get("finish_reason") != INPUT_EXCEEDS_LIMIT:
            continue
        message = node.get("message")
        return {
            "message": message if isinstance(message, str) else "Content is too long",
            "finish_reason": INPUT_EXCEEDS_LIMIT,
        }
    return None


async def _send_with_auth(account, *args, **kwargs):
    try:
        return await _send_completion(*args, **kwargs)
    except HTTPException as exc:
        if exc.status_code in (401, 403):
            account.mark_broken()
        raise


def _busy_error_body(rec: MessageReconstructor) -> dict:
    hint = rec.hint_error or {}
    return _error_detail(hint.get("message") or "DeepSeek server is busy, try again later", hint.get("finish_reason"))


FAKE_CONTEXT_HINT_ERROR_MESSAGE = "DeepSeek returned an unexpected length-limit hint and the response is empty"


def _fake_context_error_body() -> dict:
    return _error_detail(FAKE_CONTEXT_HINT_ERROR_MESSAGE, "server_error")


async def _try_stop_stream(client, session_id: str, message_id: str | None) -> None:
    if not session_id or not message_id:
        return
    try:
        await client.stop_stream(session_id, message_id)
    except Exception as exc:
        log.debug("stop_stream failed for %s: %s", session_id, exc)


def _build_assistant_message(
    content: str,
    reasoning: str | None,
    tool_mode: bool,
    tool_schemas: dict | None,
    max_calls: int | None = None,
) -> tuple[dict, str]:
    if tool_mode:
        parsed = toolemu.parse_tool_calls(content, tool_schemas)
        if parsed is not None:
            tool_calls, tool_text = parsed
            if tool_calls:
                if max_calls is not None:
                    tool_calls = tool_calls[:max_calls]
                return toolemu.format_tool_message(tool_calls, tool_text, reasoning), "tool_calls"
    message = {"role": "assistant", "content": toolemu.strip_dsml(content)}
    if reasoning:
        message["reasoning_content"] = toolemu.strip_dsml(reasoning)
    return message, "stop"


def _build_limited_message(
    content: str,
    reasoning: str | None,
    tool_mode: bool,
    tool_schemas: dict | None,
    max_tokens: int | None,
    stop: Any,
    parallel_tool_calls: bool | None,
    provider_finish: Any,
) -> tuple[dict, str]:
    if tool_mode:
        message, finish = _build_assistant_message(content, reasoning, True, tool_schemas, max_calls=_max_calls(parallel_tool_calls))
        if finish == "tool_calls":
            tool_text = message.get("content")
            if isinstance(tool_text, str):
                text = trim_to_tokens(_apply_stop(tool_text, stop), max_tokens)
                if text != tool_text:
                    finish = "length"
                message["content"] = text
            return message, finish
        text, limit_finish = _apply_limits(str(message.get("content") or ""), max_tokens, stop)
        message["content"] = text
        if limit_finish == "length":
            return message, "length"
        return message, _finish_reason(provider_finish)
    text, limit_finish = _apply_limits(toolemu.strip_dsml(content or ""), max_tokens, stop)
    message = {"role": "assistant", "content": text}
    if reasoning:
        message["reasoning_content"] = toolemu.strip_dsml(reasoning)
    if limit_finish == "length":
        return message, "length"
    return message, _finish_reason(provider_finish)


def _build_completion_response(
    model: str,
    message: dict,
    finish: str,
    usage: dict,
    session_key: str | None,
    reasoning_tokens: int | None = None,
) -> dict:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "system_fingerprint": SYSTEM_FINGERPRINT,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish,
                "logprobs": None,
            }
        ],
        "usage": _usage_with_details(usage, (message or {}).get("reasoning_content"), reasoning_tokens),
        "session_id": session_key,
    }


async def _send_deepseek_stream(
    account,
    session,
    parent_message_id,
    prompt,
    model_type,
    thinking,
    search,
    ref_file_ids=None,
) -> tuple[MessageReconstructor, str | None, str | None]:
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
    rec = MessageReconstructor()
    incremental = IncrementalSSE()
    response_message_id: str | None = None
    stop_message_id: str | None = None
    stopped = False
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
    except (httpx.HTTPError, RuntimeError) as exc:
        stopped = True
        if rec.id:
            stop_message_id = rec.id
        await _try_stop_stream(account.client, session.id, stop_message_id)
        raise DeepSeekStreamError(f"Stream processing failed: {exc}") from exc
    except BaseException:
        stopped = True
        if rec.id:
            stop_message_id = rec.id
        await _try_stop_stream(account.client, session.id, stop_message_id)
        raise
    finally:
        if rec.id:
            stop_message_id = rec.id
        try:
            await resp.aclose()
        except Exception as exc:
            log.debug("response close failed: %s", exc)
            if not stopped:
                await _try_stop_stream(account.client, session.id, stop_message_id)
    return rec, response_message_id, stop_message_id


def _continue_deadline_expired(deadline: float | None) -> bool:
    if deadline is None or time.monotonic() < deadline:
        return False
    log.warning("deepseek continuation deadline of %.0fs reached, stopping", CONTINUE_DEADLINE_SEC)
    return True


async def _collect_continuation(
    account,
    session,
    parent_message_id,
    model_type,
    thinking,
    search,
    ref_file_ids=None,
    deadline: float | None = None,
) -> MessageReconstructor | None:
    attempt = 0
    rate_attempt = 0
    while not _continue_deadline_expired(deadline):
        try:
            rec, _response_message_id, _stop_message_id = await _send_deepseek_stream(
                account,
                session,
                parent_message_id,
                CONTINUE_PROMPT,
                model_type,
                thinking,
                search,
                ref_file_ids,
            )
        except DeepSeekStreamError as exc:
            raise HTTPException(502, str(exc)) from exc
        except HTTPException as exc:
            if _is_message_too_frequent_http(exc) and rate_attempt < MESSAGE_TOO_FREQUENT_MAX_RETRIES:
                rate_attempt += 1
                await _wait_message_too_frequent("continuation request", rate_attempt)
                continue
            if _is_retryable_http(exc) and attempt < MAX_RETRIES:
                attempt += 1
                delay = _retry_delay(attempt)
                log.warning(
                    "deepseek continuation error (%s), retry %d/%d in %.1fs",
                    exc.status_code,
                    attempt,
                    MAX_RETRIES,
                    delay,
                )
                await asyncio.sleep(delay)
                continue
            return None
        if (
            not (rec.content or rec.reasoning)
            and not _is_input_exceeds_limit(rec)
            and _is_message_too_frequent_hint(rec)
            and rate_attempt < MESSAGE_TOO_FREQUENT_MAX_RETRIES
        ):
            rate_attempt += 1
            await _wait_message_too_frequent("continuation hint", rate_attempt)
            continue
        if (
            not (rec.content or rec.reasoning)
            and not _is_input_exceeds_limit(rec)
            and (_is_retryable_hint(rec) or _is_fake_context_hint(rec))
            and attempt < MAX_RETRIES
        ):
            attempt += 1
            delay = _retry_delay(attempt)
            log.warning(
                "deepseek continuation retryable hint (%s), retry %d/%d in %.1fs",
                (rec.hint_error or {}).get("finish_reason"),
                attempt,
                MAX_RETRIES,
                delay,
            )
            await asyncio.sleep(delay)
            continue
        return rec
    return None


def _reduced_prompt_variants(
    messages: list[Any], tools: list[Any] | None, tool_choice: Any, response_format: Any, original_prompt: str
) -> list[tuple[str, bool, dict[str, Any]]]:
    variants: list[tuple[str, bool, dict[str, Any]]] = []
    if tools:
        try:
            prompt, tool_mode = toolemu.build_prompt(messages, None, None, False, response_format)
            if prompt != original_prompt:
                variants.append((prompt, tool_mode, {}))
        except ValueError:
            pass
    system_msgs = [msg for msg in messages if getattr(msg, "role", None) == "system"]
    last_input = None
    for msg in reversed(messages):
        if getattr(msg, "role", None) in ("user", "system"):
            last_input = msg
            break
    reduced_msgs = list(system_msgs)
    if last_input is not None and getattr(last_input, "role", None) == "user":
        reduced_msgs.append(last_input)
    if reduced_msgs:
        try:
            prompt, tool_mode = toolemu.build_prompt(reduced_msgs, None, None, False, response_format)
            if prompt != original_prompt:
                variants.append((prompt, tool_mode, {}))
        except ValueError:
            pass
    if reduced_msgs and tools:
        try:
            prompt, tool_mode = toolemu.build_prompt(reduced_msgs, tools, tool_choice, False, response_format)
            if prompt != original_prompt and not any(v[0] == prompt for v in variants):
                variants.append((prompt, tool_mode, toolemu.tool_schema_map(tools)))
        except ValueError:
            pass
    return variants


async def _collect_reduced(
    account,
    pool,
    reduced_prompts: list[tuple[str, bool, dict[str, Any]]],
    model_type,
    thinking,
    search,
    ref_file_ids=None,
):
    await _human_delay()
    for prompt, variant_tool_mode, variant_tool_schemas in reduced_prompts:
        session_key = None
        try:
            session, session_key, parent_message_id = await _prepare_session(account, pool, None, None)
            rec, _response_message_id, _stop_message_id = await _send_deepseek_stream(
                account,
                session,
                parent_message_id,
                prompt,
                model_type,
                thinking,
                search,
                ref_file_ids,
            )
        except (HTTPException, httpx.HTTPError, DeepSeekStreamError):
            if session_key is not None:
                _drop_session(pool, account, session_key)
            continue
        if (rec.content or rec.reasoning) and not _is_input_exceeds_limit(rec):
            return rec, session, session_key, variant_tool_mode, variant_tool_schemas
        if session_key is not None:
            _drop_session(pool, account, session_key)
    return None


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
    attachments=None,
    tool_mode=False,
    tool_schemas=None,
    context_seq: tuple[str, ...] | None = None,
    reduced_prompts: list[tuple[str, bool, dict[str, Any]]] | None = None,
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
    await _human_delay()
    async with account_lock(lock, settings.acquire_timeout):
        if attachments:
            ref_file_ids = await _upload_attachments(account, attachments, model_type, thinking)
        had_cached_session = bool(existing_sid) and account.sessions.get(existing_sid) is not None
        session, session_key, parent_message_id = await _prepare_session(account, pool, existing_sid, context_seq)
        if (session_key != existing_sid or session is not cached_session) and messages is not None:
            try:
                prompt, tool_mode = toolemu.build_prompt(messages, tools, tool_choice, False, response_format)
            except ValueError:
                pass
            tool_schemas = toolemu.tool_schema_map(tools)
        stop_message_id: str | None = None
        started = time.monotonic()
        deadline = started + CONTINUE_DEADLINE_SEC
        stale_rebuilt = False
        rec: MessageReconstructor | None = None
        response_message_id = None
        attempt = 0
        rate_attempt = 0

        try:
            while True:
                try:
                    rec, response_message_id, stop_message_id = await _send_deepseek_stream(
                        account,
                        session,
                        parent_message_id,
                        prompt,
                        model_type,
                        thinking,
                        search,
                        ref_file_ids,
                    )
                except DeepSeekStreamError as exc:
                    raise HTTPException(502, str(exc)) from exc
                except HTTPException as exc:
                    input_hint = _input_exceeds_hint_from_http(exc)
                    if input_hint is not None:
                        rec = MessageReconstructor()
                        rec.hint_error = input_hint
                        break
                    if exc.status_code in STALE_SESSION_STATUSES and had_cached_session and not stale_rebuilt and messages is not None:
                        stale_rebuilt = True
                        _drop_session(pool, account, session_key)
                        try:
                            prompt, tool_mode = toolemu.build_prompt(messages, tools, tool_choice, False, response_format)
                            tool_schemas = toolemu.tool_schema_map(tools)
                        except (ValueError, TypeError, AttributeError) as build_exc:
                            raise exc from build_exc
                        log.warning("deepseek session %s is stale (%s), rebuilt full history into a fresh chat", session_key, exc.status_code)
                        session, session_key, parent_message_id = await _prepare_session(account, pool, existing_sid, context_seq)
                        stop_message_id = None
                        response_message_id = None
                        continue
                    if _is_message_too_frequent_http(exc) and rate_attempt < MESSAGE_TOO_FREQUENT_MAX_RETRIES:
                        rate_attempt += 1
                        await _wait_message_too_frequent("request", rate_attempt)
                        continue
                    if _is_retryable_http(exc) and attempt < MAX_RETRIES:
                        attempt += 1
                        delay = _retry_delay(attempt)
                        log.warning(
                            "deepseek provider error (%s), retry %d/%d in %.1fs",
                            exc.status_code,
                            attempt,
                            MAX_RETRIES,
                            delay,
                        )
                        await asyncio.sleep(delay)
                        continue
                    raise
                if (
                    not (rec.content or rec.reasoning)
                    and not _is_input_exceeds_limit(rec)
                    and _is_message_too_frequent_hint(rec)
                    and rate_attempt < MESSAGE_TOO_FREQUENT_MAX_RETRIES
                ):
                    rate_attempt += 1
                    await _wait_message_too_frequent("stream hint", rate_attempt)
                    continue
                if (
                    not (rec.content or rec.reasoning)
                    and not _is_input_exceeds_limit(rec)
                    and _is_fake_context_hint(rec)
                    and had_cached_session
                    and not stale_rebuilt
                    and messages is not None
                ):
                    try:
                        rebuilt_prompt, rebuilt_tool_mode = toolemu.build_prompt(messages, tools, tool_choice, False, response_format)
                    except (ValueError, TypeError, AttributeError):
                        rebuilt_prompt = None
                    if rebuilt_prompt is not None:
                        stale_rebuilt = True
                        _drop_session(pool, account, session_key)
                        prompt = rebuilt_prompt
                        tool_mode = rebuilt_tool_mode
                        tool_schemas = toolemu.tool_schema_map(tools)
                        log.warning("deepseek session %s hit a fake length-limit hint, rebuilt full history into a fresh chat", session_key)
                        session, session_key, parent_message_id = await _prepare_session(account, pool, existing_sid, context_seq)
                        stop_message_id = None
                        response_message_id = None
                        continue
                if (
                    not (rec.content or rec.reasoning)
                    and not _is_input_exceeds_limit(rec)
                    and (_is_retryable_hint(rec) or _is_fake_context_hint(rec))
                    and attempt < MAX_RETRIES
                ):
                    attempt += 1
                    delay = _retry_delay(attempt)
                    log.warning(
                        "deepseek retryable hint (%s), retry %d/%d in %.1fs",
                        (rec.hint_error or {}).get("finish_reason"),
                        attempt,
                        MAX_RETRIES,
                        delay,
                    )
                    await asyncio.sleep(delay)
                    continue
                break
        except BaseException:
            await _try_stop_stream(account.client, session.id, stop_message_id)
            raise

        assert rec is not None
        if _is_context_limit(rec) and not (rec.content or rec.reasoning):
            _drop_session(pool, account, session_key)
            raise HTTPException(400, "context length exceeded: conversation too long, start a new conversation")
        incomplete_message: str | None = None
        reduced_notice: str | None = None
        if _is_input_exceeds_limit(rec):
            incomplete_message = _incomplete_message(rec)
            cont_parent = rec.id or response_message_id or parent_message_id
            for _ in range(MAX_CONTINUE_ROUNDS):
                if _continue_deadline_expired(deadline):
                    break
                cont_rec = await _collect_continuation(account, session, cont_parent, model_type, thinking, search, ref_file_ids, deadline)
                if cont_rec is None:
                    break
                rec.extend_with(cont_rec)
                cont_parent = cont_rec.id or cont_parent
                if not _is_input_exceeds_limit(cont_rec):
                    incomplete_message = None
                    break
                incomplete_message = _incomplete_message(cont_rec)
                if not (cont_rec.content or cont_rec.reasoning):
                    break
            if incomplete_message is not None and not (rec.content or rec.reasoning):
                if reduced_prompts is None and messages is not None:
                    reduced_prompts = _reduced_prompt_variants(messages, tools, tool_choice, response_format, prompt)
                if reduced_prompts:
                    _drop_session(pool, account, session_key)
                    reduced = await _collect_reduced(account, pool, reduced_prompts, model_type, thinking, search, ref_file_ids)
                    if reduced is not None:
                        rec, session, session_key, variant_tool_mode, variant_tool_schemas = reduced
                        tool_mode = variant_tool_mode
                        tool_schemas = variant_tool_schemas
                        response_message_id = rec.id or response_message_id
                        stop_message_id = response_message_id
                        reduced_notice = REDUCED_CONTEXT_MESSAGE
        if incomplete_message is not None and reduced_notice is None:
            log.warning("deepseek response incomplete: %s", incomplete_message)
            raise HTTPException(502, _incomplete_error_body(incomplete_message))
        content = rec.content
        reasoning = rec.reasoning
        if not (content or reasoning) and rec.hint_error:
            if _is_fake_context_hint(rec):
                log.warning("deepseek fake context-length hint after retries")
                raise HTTPException(502, _fake_context_error_body())
            raise HTTPException(429, _busy_error_body(rec))
        request_tokens = _advance_session_usage(session, rec.accumulated_tokens)
        usage = _deepseek_usage(request_tokens, prompt, rec.usage, completion_text=rec.content or rec.reasoning)
        account.sessions.touch_last_message(session_key, rec.id or response_message_id)
        record_usage(
            "deepseek",
            model,
            usage["prompt_tokens"],
            usage["completion_tokens"],
            usage["total_tokens"],
            user=user,
            session_id=session_key,
        )
        log.info("deepseek completion success (%.0fms)", (time.monotonic() - started) * 1000)
        message, finish = _build_limited_message(content, reasoning, tool_mode, tool_schemas, max_tokens, stop, parallel_tool_calls, rec.status)
        reasoning_tokens = estimate_tokens(reasoning) if reasoning else 0
        response = _build_completion_response(model, message, finish, usage, session_key, reasoning_tokens)
        if isinstance(n, int) and n and n > 1:
            template = response["choices"][0]
            response["choices"] = [dict(template) | {"index": i} for i in range(_bounded_choices(n))]
        if reduced_notice is not None:
            log.warning("deepseek response delivered from reduced context (%s)", model)
            response["error"] = {"message": reduced_notice, "finish_reason": RESPONSE_INCOMPLETE}
            response["choices"][0]["finish_reason"] = RESPONSE_INCOMPLETE
        return response


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
    attachments=None,
    tool_mode=False,
    tool_schemas=None,
    include_usage=False,
    context_seq: tuple[str, ...] | None = None,
    reduced_prompts: list[tuple[str, bool, dict[str, Any]]] | None = None,
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

    await _human_delay()
    async with account_lock(lock, settings.acquire_timeout):
        if attachments:
            try:
                ref_file_ids = await _upload_attachments(account, attachments, model_type, thinking)
            except HTTPException as exc:
                detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
                for line in _stream_error_sse(chunk_id, created, model, detail):
                    yield line
                return
        had_cached_session = bool(existing_sid) and account.sessions.get(existing_sid) is not None
        try:
            session, session_key, parent_message_id = await _prepare_session(account, pool, existing_sid, context_seq)
        except HTTPException as exc:
            detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
            for line in _stream_error_sse(chunk_id, created, model, detail):
                yield line
            return

        if (session_key != existing_sid or session is not cached_session) and messages is not None:
            try:
                prompt, tool_mode = toolemu.build_prompt(messages, tools, tool_choice, False, response_format)
            except ValueError:
                pass
            tool_schemas = toolemu.tool_schema_map(tools)

        rec: MessageReconstructor | None = None
        response_message_id = None
        stop_message_id: str | None = None
        content_buf = ""
        content_shown_len = 0
        tool_hidden = False
        role_sent = False
        got_content = False
        started = time.monotonic()
        deadline = started + CONTINUE_DEADLINE_SEC
        budget = StreamBudget(max_tokens, trim_to_tokens)
        stop_markers = split_stop(stop)
        stop_filter = StreamStopFilter(stop_markers) if stop_markers else None
        stop_hit = False
        dsml_filter = toolemu.DsmlFilter()
        reasoning_filter = toolemu.DsmlFilter()

        def reasoning_piece(piece: str | None, *, final: bool = False) -> str:
            if stop_hit:
                return ""
            if final:
                return reasoning_filter.flush()
            if piece is None:
                return ""
            return reasoning_filter.feed(piece)

        def content_piece(piece: str | None, *, final: bool = False) -> str:
            nonlocal stop_hit
            if stop_hit:
                return ""
            if final:
                text = dsml_filter.flush()
            else:
                if piece is None:
                    return ""
                text = dsml_filter.feed(piece)
            if stop_filter is None:
                return budget.feed(text)
            filtered, hit = stop_filter.feed(text)
            if hit:
                stop_hit = True
            out = budget.feed(filtered)
            if final and not stop_hit:
                out += budget.feed(stop_filter.flush())
            return out

        chunk_head = f'data: {{"id":{_JSON_ENCODE(chunk_id)},"object":"chat.completion.chunk","created":{created},"model":{_JSON_ENCODE(model)},"choices":['

        def _chunk(delta: dict, finish: str | None = None, *, index: int = 0, session_id: Any = _UNSET) -> str:
            if index == 0 and session_id is _UNSET:
                return chunk_head + _delta_json(delta, finish) + "]}\n\n"
            payload: dict[str, Any] = {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
            }
            if session_id is not _UNSET:
                payload["session_id"] = session_id
            payload["choices"] = [{"index": index, "delta": delta, "finish_reason": finish}]
            return _sse(payload)

        def _role_chunk() -> str | None:
            nonlocal role_sent
            if role_sent:
                return None
            role_sent = True
            return _chunk({"role": "assistant"})

        def _content_text(piece: str) -> str:
            nonlocal content_buf, content_shown_len, tool_hidden
            if not tool_mode:
                return content_piece(piece)
            content_buf += piece
            visible, content_shown_len, tool_hidden = toolemu.tool_visible(content_buf, content_shown_len, tool_hidden, tool_schemas)
            return content_piece(visible)

        def _reasoning_text(piece: str) -> str:
            return reasoning_piece(piece)

        def _done_finish(status: Any) -> str:
            return "stop" if stop_hit else ("length" if budget.done else _finish_reason(status))

        def _drain_event(reconstructor: MessageReconstructor, event) -> Iterator[str]:
            nonlocal response_message_id, stop_message_id, got_content
            if event.event == "ready" and isinstance(event.data, dict):
                response_message_id = event.data.get("response_message_id")
                if response_message_id:
                    stop_message_id = response_message_id
            reconstructor.handle(event)
            c_diff, r_diff = reconstructor.take_diffs()
            if not (c_diff or r_diff):
                return
            got_content = True
            role_line = _role_chunk()
            if role_line:
                yield role_line
            delta: dict = {}
            if c_diff:
                allowed = _content_text(c_diff)
                if allowed:
                    delta["content"] = allowed
            if r_diff:
                reason = _reasoning_text(r_diff)
                if reason:
                    delta["reasoning_content"] = reason
            if delta:
                yield _chunk(delta)

        stale_rebuilt = False
        attempt = 0
        rate_attempt = 0
        while True:
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
                if exc.status_code in STALE_SESSION_STATUSES and had_cached_session and not stale_rebuilt and messages is not None:
                    stale_rebuilt = True
                    _drop_session(pool, account, session_key)
                    try:
                        prompt, tool_mode = toolemu.build_prompt(messages, tools, tool_choice, False, response_format)
                        tool_schemas = toolemu.tool_schema_map(tools)
                    except ValueError:
                        detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
                        for line in _stream_error_sse(chunk_id, created, model, detail, session_key):
                            yield line
                        return
                    log.warning("deepseek session %s is stale (%s), rebuilt full history into a fresh chat", session_key, exc.status_code)
                    session, session_key, parent_message_id = await _prepare_session(account, pool, existing_sid, context_seq)
                    stop_message_id = None
                    response_message_id = None
                    continue
                if _is_message_too_frequent_http(exc) and rate_attempt < MESSAGE_TOO_FREQUENT_MAX_RETRIES:
                    rate_attempt += 1
                    await _wait_message_too_frequent("request", rate_attempt)
                    continue
                if _is_retryable_http(exc) and attempt < MAX_RETRIES:
                    attempt += 1
                    delay = _retry_delay(attempt)
                    log.warning(
                        "deepseek provider error (%s), retry %d/%d in %.1fs",
                        exc.status_code,
                        attempt,
                        MAX_RETRIES,
                        delay,
                    )
                    await asyncio.sleep(delay)
                    continue
                detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
                for line in _stream_error_sse(chunk_id, created, model, detail, session_key):
                    yield line
                return
            rec = MessageReconstructor()
            incremental = IncrementalSSE()
            response_message_id = None
            stop_message_id = None
            got_content = False
            role_sent = False
            stopped = False
            try:
                async for chunk in resp.aiter_bytes():
                    for event in incremental.feed(chunk):
                        for line in _drain_event(rec, event):
                            yield line
                for event in incremental.finish():
                    for line in _drain_event(rec, event):
                        yield line
            except BaseException:
                stopped = True
                if rec.id:
                    stop_message_id = rec.id
                await _try_stop_stream(account.client, session.id, stop_message_id)
                raise
            finally:
                if rec.id:
                    stop_message_id = rec.id
                try:
                    await resp.aclose()
                except Exception as exc:
                    log.debug("response close failed: %s", exc)
                    if not stopped:
                        await _try_stop_stream(account.client, session.id, stop_message_id)
            if got_content:
                break
            if not _is_input_exceeds_limit(rec) and _is_message_too_frequent_hint(rec) and rate_attempt < MESSAGE_TOO_FREQUENT_MAX_RETRIES:
                rate_attempt += 1
                await _wait_message_too_frequent("stream hint", rate_attempt)
                continue
            if not _is_input_exceeds_limit(rec) and _is_fake_context_hint(rec) and had_cached_session and not stale_rebuilt and messages is not None:
                try:
                    rebuilt_prompt, rebuilt_tool_mode = toolemu.build_prompt(messages, tools, tool_choice, False, response_format)
                except (ValueError, TypeError, AttributeError):
                    rebuilt_prompt = None
                if rebuilt_prompt is not None:
                    stale_rebuilt = True
                    _drop_session(pool, account, session_key)
                    prompt = rebuilt_prompt
                    tool_mode = rebuilt_tool_mode
                    tool_schemas = toolemu.tool_schema_map(tools)
                    log.warning("deepseek session %s hit a fake length-limit hint, rebuilt full history into a fresh chat", session_key)
                    session, session_key, parent_message_id = await _prepare_session(account, pool, existing_sid, context_seq)
                    stop_message_id = None
                    response_message_id = None
                    content_buf = ""
                    content_shown_len = 0
                    tool_hidden = False
                    role_sent = False
                    stop_hit = False
                    dsml_filter = toolemu.DsmlFilter()
                    reasoning_filter = toolemu.DsmlFilter()
                    continue
            if not _is_input_exceeds_limit(rec) and (_is_retryable_hint(rec) or _is_fake_context_hint(rec)) and attempt < MAX_RETRIES:
                attempt += 1
                delay = _retry_delay(attempt)
                log.warning(
                    "deepseek retryable hint (%s), retry %d/%d in %.1fs",
                    (rec.hint_error or {}).get("finish_reason"),
                    attempt,
                    MAX_RETRIES,
                    delay,
                )
                await asyncio.sleep(delay)
                continue
            break

        assert rec is not None
        if _is_context_limit(rec) and not (rec.content or rec.reasoning):
            _drop_session(pool, account, session_key)
            for line in _stream_error_sse(
                chunk_id,
                created,
                model,
                "context length exceeded: conversation too long, start a new conversation",
                session_key,
                CONTEXT_LENGTH_STATUS,
                "length",
            ):
                yield line
            return
        incomplete_message: str | None = None
        reduced_notice: str | None = None
        if _is_input_exceeds_limit(rec):
            incomplete_message = _incomplete_message(rec)
            cont_parent = rec.id or response_message_id or parent_message_id
            for _ in range(MAX_CONTINUE_ROUNDS):
                if _continue_deadline_expired(deadline):
                    break
                cont_rec = await _collect_continuation(account, session, cont_parent, model_type, thinking, search, ref_file_ids, deadline)
                if cont_rec is None:
                    break
                rec.extend_with(cont_rec)
                cont_parent = cont_rec.id or cont_parent
                if cont_rec.content:
                    role_line = _role_chunk()
                    if role_line:
                        yield role_line
                    allowed = _content_text(cont_rec.content)
                    if allowed:
                        yield _chunk({"content": allowed})
                if cont_rec.reasoning:
                    role_line = _role_chunk()
                    if role_line:
                        yield role_line
                    cont_reason = _reasoning_text(cont_rec.reasoning)
                    if cont_reason:
                        yield _chunk({"reasoning_content": cont_reason})
                if not _is_input_exceeds_limit(cont_rec):
                    incomplete_message = None
                    break
                incomplete_message = _incomplete_message(cont_rec)
                if not (cont_rec.content or cont_rec.reasoning):
                    break
            if incomplete_message is not None and not (rec.content or rec.reasoning):
                if reduced_prompts is None and messages is not None:
                    reduced_prompts = _reduced_prompt_variants(messages, tools, tool_choice, response_format, prompt)
                if reduced_prompts:
                    _drop_session(pool, account, session_key)
                    reduced = await _collect_reduced(account, pool, reduced_prompts, model_type, thinking, search, ref_file_ids)
                    if reduced is not None:
                        rec, session, session_key, variant_tool_mode, variant_tool_schemas = reduced
                        tool_mode = variant_tool_mode
                        tool_schemas = variant_tool_schemas
                        response_message_id = rec.id or response_message_id
                        stop_message_id = response_message_id
                        reduced_notice = REDUCED_CONTEXT_MESSAGE
                        if rec.content:
                            role_line = _role_chunk()
                            if role_line:
                                yield role_line
                            allowed = _content_text(rec.content)
                            if allowed:
                                yield _chunk({"content": allowed})
                        if rec.reasoning:
                            role_line = _role_chunk()
                            if role_line:
                                yield role_line
                            reduced_reason = _reasoning_text(rec.reasoning)
                            if reduced_reason:
                                yield _chunk({"reasoning_content": reduced_reason})
        if incomplete_message is not None and reduced_notice is None:
            log.warning("deepseek response incomplete: %s", incomplete_message)
            for line in _stream_error_sse(chunk_id, created, model, incomplete_message, session_key, RESPONSE_INCOMPLETE, RESPONSE_INCOMPLETE):
                yield line
            return
        if not (rec.content or rec.reasoning) and rec.hint_error:
            if _is_fake_context_hint(rec):
                log.warning("deepseek fake context-length hint after retries")
                for line in _stream_error_sse(chunk_id, created, model, FAKE_CONTEXT_HINT_ERROR_MESSAGE, session_key, "server_error"):
                    yield line
                return
            hint = rec.hint_error
            for line in _stream_error_sse(
                chunk_id,
                created,
                model,
                hint.get("message") or "DeepSeek server is busy, try again later",
                session_key,
                hint.get("finish_reason"),
            ):
                yield line
            return

        request_tokens = _advance_session_usage(session, rec.accumulated_tokens)
        usage = _deepseek_usage(request_tokens, prompt, rec.usage, completion_text=rec.content or rec.reasoning)
        account.sessions.touch_last_message(session_key, rec.id or response_message_id)
        record_usage(
            "deepseek",
            model,
            usage["prompt_tokens"],
            usage["completion_tokens"],
            usage["total_tokens"],
            user=user,
            session_id=session_key,
        )
        log.info("deepseek completion success (%.0fms)", (time.monotonic() - started) * 1000)

        finish = _done_finish(rec.status)
        tool_call_deltas: list[dict] = []
        if tool_mode:
            parsed = toolemu.parse_tool_calls(content_buf or rec.content, tool_schemas)
            tool_calls = parsed[0] if parsed is not None and parsed[0] else None
            if tool_calls:
                if _output_truncated(rec.status):
                    for line in _stream_error_sse(
                        chunk_id,
                        created,
                        model,
                        "the provider output ended before the tool call completed",
                        session_key,
                        "length",
                        "length",
                    ):
                        yield line
                    return
                max_calls = _max_calls(parallel_tool_calls)
                if max_calls is not None:
                    tool_calls = tool_calls[:max_calls]
                tool_call_deltas = toolemu.tool_call_deltas(tool_calls)
                for delta in tool_call_deltas:
                    yield _chunk(delta)
                finish = "tool_calls"
            else:
                shown = content_buf or rec.content
                tail_text = content_piece(shown[content_shown_len:] if shown else "")
                if tail_text:
                    yield _chunk({"content": tail_text})
                finish = _done_finish(rec.status)

        if not tool_call_deltas:
            tail_text = content_piece(None, final=True)
            if tail_text:
                yield _chunk({"content": tail_text})

        tail_reason = reasoning_piece(None, final=True)
        if tail_reason:
            yield _chunk({"reasoning_content": tail_reason})

        role_line = _role_chunk()
        if role_line:
            yield role_line

        if reduced_notice is not None:
            yield _sse(
                {
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "session_id": session_key,
                    "error": {"message": reduced_notice, "finish_reason": RESPONSE_INCOMPLETE},
                    "choices": [{"index": 0, "delta": {}, "finish_reason": RESPONSE_INCOMPLETE}],
                }
            )
        else:
            yield _chunk({}, finish, session_id=session_key)
            tail = budget.text
            for extra_index in range(1, _bounded_choices(n)):
                if tool_call_deltas:
                    for delta in tool_call_deltas:
                        yield _chunk(delta, index=extra_index, session_id=session_key)
                elif tail:
                    yield _chunk({"content": tail}, index=extra_index, session_id=session_key)
                yield _chunk({}, finish, index=extra_index, session_id=session_key)
        if include_usage:
            yield _sse(
                {
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "session_id": session_key,
                    "usage": _usage_with_details(usage, rec.reasoning),
                    "choices": [],
                }
            )
        yield "data: [DONE]\n\n"
