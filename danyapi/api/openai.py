from __future__ import annotations

import asyncio
import base64
import json
import logging
import random
import sys
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .. import tools as toolemu
from ..accounts import AccountPool, AccountPoolBusy, DeepSeekAccount, account_lock
from ..config import settings
from ..deepseek.client import DeepSeekClient, DeepSeekError, DeepSeekSession
from ..deepseek.stream import IncrementalSSE, MessageReconstructor
from ..qwen import api as qwen_api
from ..qwen.accounts import QwenAccount
from ..qwen.client import QwenClient, QwenError
from ..store import JsonStore

log = logging.getLogger("danyapi.api")

MODEL_TYPE_BY_NAME = {
    "deepseek-v4-flash": "default",
    "deepseek-v4-pro": "expert",
    "deepseek-v4-vision": "vision",
}

QWEN_DEFAULT_MODELS = [
    {
        "id": "qwen3.8-max",
        "name": "Qwen3.8-Max",
        "owned_by": "qwen",
        "model_type": "chat",
    },
    {
        "id": "qwen3.7-plus",
        "name": "Qwen3.7-Plus",
        "owned_by": "qwen",
        "model_type": "chat",
    },
    {
        "id": "qwen3.7-max",
        "name": "Qwen3.7-Max",
        "owned_by": "qwen",
        "model_type": "chat",
    },
]

STATUS_TO_FINISH_REASON = {
    "FINISHED": "stop",
    "CONTEXT_LENGTH_EXCEEDED": "length",
    "CONTENT_FILTER": "content_filter",
    "INCOMPLETE": "stop",
    "WIP": "stop",
    "TIMEOUT": "stop",
}

CONTEXT_LENGTH_STATUS = "CONTEXT_LENGTH_EXCEEDED"
INPUT_EXCEEDS_LIMIT = "input_exceeds_limit"
CONTINUE_PROMPT = "Continue"
MAX_CONTINUE_ROUNDS = 5


class ChatMessage(BaseModel):
    role: str = "user"
    content: Any = ""
    tool_calls: list[Any] | None = None
    tool_call_id: str | None = None
    name: str | None = None


class FileSpec(BaseModel):
    name: str
    content: str
    content_type: str = "application/octet-stream"


class ChatCompletionRequest(BaseModel):
    model: str = Field(default="deepseek-v4-flash")
    messages: list[ChatMessage] = Field(default_factory=list)
    stream: bool = False
    temperature: float | None = None
    top_p: float | None = None
    thinking: bool | None = None
    search: bool | None = None
    session_id: str | None = None
    user: str | None = None
    files: list[FileSpec] | None = None
    tools: list[Any] | None = None
    tool_choice: Any = None
    parallel_tool_calls: bool | None = None
    response_format: Any = None
    stream_options: Any = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    accounts: list[DeepSeekAccount] = []
    qwen_accounts: list[QwenAccount] = []
    cache_enabled = settings.cache_enabled
    deepseek_session_store = JsonStore("deepseek-sessions", "default" if cache_enabled else None)
    qwen_session_store = JsonStore("qwen-sessions", "default" if cache_enabled else None)
    deepseek_context_store = JsonStore("deepseek-contexts", "default" if cache_enabled else None)
    qwen_context_store = JsonStore("qwen-contexts", "default" if cache_enabled else None)
    deepseek_affinity_store = JsonStore("deepseek-affinities", "default" if cache_enabled else None)
    qwen_affinity_store = JsonStore("qwen-affinities", "default" if cache_enabled else None)
    try:
        if settings.deepseek_tokens:
            for i, token in enumerate(settings.deepseek_tokens):
                client = DeepSeekClient(token=token, timeout=settings.timeout)
                if not await client.check_auth():
                    log.warning("deepseek token #%d invalid/expired, skipping", i)
                    await client.aclose()
                    continue
                accounts.append(
                    DeepSeekAccount(
                        len(accounts),
                        client,
                        session_cache_size=settings.session_cache_size,
                        ttl=settings.session_ttl,
                        store=deepseek_session_store,
                    )
                )
            log.info("deepseek accounts ready: %d", len(accounts))
        elif settings.deepseek_email:
            client = DeepSeekClient(timeout=settings.timeout)
            await client.login(
                email=settings.deepseek_email,
                password=settings.deepseek_password,
            )
            accounts.append(DeepSeekAccount(0, client))
            log.info("deepseek login ok, device_id=%s", client.device_id)
        if settings.qwen_tokens:
            for i, token in enumerate(settings.qwen_tokens):
                client = QwenClient(token=token, timeout=settings.timeout)
                if not await client.check_auth():
                    log.warning("qwen token #%d invalid/expired, skipping", i)
                    await client.aclose()
                    continue
                qwen_accounts.append(
                    QwenAccount(
                        len(qwen_accounts),
                        client,
                        session_cache_size=settings.session_cache_size,
                        ttl=settings.session_ttl,
                        store=qwen_session_store,
                    )
                )
            log.info("qwen accounts ready: %d", len(qwen_accounts))
        elif settings.qwen_email:
            client = QwenClient(timeout=settings.timeout)
            await client.login(
                email=settings.qwen_email,
                password=settings.qwen_password,
            )
            qwen_accounts.append(QwenAccount(0, client))
            log.info("qwen login ok")
        if accounts:
            app.state.pool = AccountPool(
                accounts,
                session_cache_size=settings.session_cache_size,
                ttl=settings.session_ttl,
                context_store=deepseek_context_store,
                affinity_store=deepseek_affinity_store,
            )
        else:
            app.state.pool = None
        if qwen_accounts:
            app.state.qwen_pool = AccountPool(
                qwen_accounts,
                label="qwen",
                session_cache_size=settings.session_cache_size,
                ttl=settings.session_ttl,
                context_store=qwen_context_store,
                affinity_store=qwen_affinity_store,
            )
            app.state.qwen_models = await _fetch_qwen_models(qwen_accounts[0].client)
        else:
            app.state.qwen_pool = None
            app.state.qwen_models = []
        if not accounts and not qwen_accounts:
            raise RuntimeError("no valid credentials: set DEEPSEEK_TOKENS/DEEPSEEK_EMAIL or QWEN_TOKENS/QWEN_EMAIL")
        yield
    finally:
        for acct in accounts:
            await acct.client.aclose()
        for acct in qwen_accounts:
            await acct.client.aclose()


async def _fetch_qwen_models(client: QwenClient) -> list[dict]:
    try:
        raw = await client.fetch_models()
    except QwenError as exc:
        log.warning("qwen models fetch failed, using defaults: %s", exc)
        return QWEN_DEFAULT_MODELS
    models = []
    for model in raw:
        if not isinstance(model, dict) or not model.get("id"):
            continue
        info = model.get("info")
        meta = (info or {}).get("meta") if isinstance(info, dict) else None
        chat_types = (meta or {}).get("chat_type") if isinstance(meta, dict) else None
        if chat_types is not None and "t2t" not in chat_types:
            continue
        models.append(
            {
                "id": model["id"],
                "name": model.get("name") or model["id"],
                "owned_by": "qwen",
                "model_type": "chat",
            }
        )
    if not models:
        log.warning("qwen models fetch returned nothing, using defaults")
        return QWEN_DEFAULT_MODELS
    return models


app = FastAPI(title="DanyAPI", lifespan=lifespan)


MAX_FILES_PER_REQUEST = 50
MAX_FILE_SIZE = 100 * 1024 * 1024


@dataclass
class Attachment:
    data: bytes
    name: str
    content_type: str
    is_image: bool


def _split_data_uri(uri: str) -> tuple[str, bytes]:
    if not uri.startswith("data:"):
        raise HTTPException(400, "image_url must be a data URI (data:<mime>;base64,...)")
    meta, _, payload = uri[5:].partition(",")
    if not payload:
        raise HTTPException(400, "invalid data URI: missing base64 payload")
    content_type = meta.split(";", 1)[0] or "application/octet-stream"
    try:
        data = base64.b64decode(payload, validate=True)
    except ValueError as exc:
        raise HTTPException(400, "invalid base64 in image_url") from exc
    return content_type, data


def _collect_attachments(req: ChatCompletionRequest) -> list[Attachment]:
    attachments: list[Attachment] = []
    for msg in req.messages:
        if not isinstance(msg.content, list):
            continue
        for item in msg.content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "image_url":
                image_url = item.get("image_url")
                if isinstance(image_url, str):
                    uri = image_url
                elif isinstance(image_url, dict) and isinstance(image_url.get("url"), str):
                    uri = image_url["url"]
                else:
                    raise HTTPException(400, "invalid image_url value")
                content_type, data = _split_data_uri(uri)
                name = f"image_{len(attachments)}.{content_type.split('/')[-1] or 'bin'}"
                attachments.append(Attachment(data, name, content_type, True))
    for f in req.files or []:
        if not f.name or not f.content:
            raise HTTPException(400, "each file needs name and base64 content")
        try:
            data = base64.b64decode(f.content)
        except ValueError as exc:
            raise HTTPException(400, f"invalid base64 in file {f.name}") from exc
        attachments.append(Attachment(data, f.name, f.content_type or "application/octet-stream", (f.content_type or "").startswith("image/")))
    return attachments


def _validate_attachments(attachments: list[Attachment], model_type: str) -> None:
    if not attachments:
        return
    if len(attachments) > MAX_FILES_PER_REQUEST:
        raise HTTPException(400, f"too many files: max {MAX_FILES_PER_REQUEST} per request")
    for att in attachments:
        if len(att.data) > MAX_FILE_SIZE:
            raise HTTPException(400, f"file {att.name} exceeds 100 MB limit")
    if model_type == "expert":
        raise HTTPException(400, "deepseek-v4-pro does not support file attachments")
    if model_type == "vision" and any(not att.is_image for att in attachments):
        raise HTTPException(400, "deepseek-v4-vision accepts images only")


async def _fresh_pow_upload_headers(account) -> dict:
    try:
        return await account.pow_upload.make_header(lambda: account.client.create_pow_challenge("/api/v0/file/upload_file"))
    except DeepSeekError as exc:
        _handle_account_error(account, exc)
        raise HTTPException(_deepseek_status(exc), _deepseek_error_detail(exc)) from exc


async def _upload_attachments(account, attachments: list[Attachment], model_type: str, thinking: bool) -> list[str]:
    file_ids: list[str] = []
    for att in attachments:
        pow_headers = await _fresh_pow_upload_headers(account)
        try:
            info = await account.client.upload_file(
                att.data,
                att.name,
                att.content_type,
                model_type,
                thinking_enabled=thinking,
                pow_headers=pow_headers,
            )
        except DeepSeekError as exc:
            _handle_account_error(account, exc)
            raise HTTPException(_deepseek_status(exc), f"file upload failed: {exc}") from exc
        file_id = info.get("id")
        if not file_id:
            raise HTTPException(502, f"file upload failed for {att.name}: no file id")
        file_ids.append(file_id)
    return file_ids


def _resolve_model(model: str) -> str:
    model_type = MODEL_TYPE_BY_NAME.get(model)
    if model_type is None:
        raise HTTPException(404, f"Unknown model: {model}")
    return model_type


def _finish_reason(status: Any) -> str:
    if isinstance(status, str):
        return STATUS_TO_FINISH_REASON.get(status, "stop")
    return "stop"


def _pool_stats(pool) -> dict | None:
    if pool is None:
        return None
    try:
        return pool.stats()
    except Exception:
        return None


@app.get("/health")
async def health() -> dict:
    pool = getattr(app.state, "pool", None)
    qwen_pool = getattr(app.state, "qwen_pool", None)
    return {
        "status": "ok",
        "deepseek": pool is not None,
        "qwen": qwen_pool is not None,
        "deepseek_stats": _pool_stats(pool),
        "qwen_stats": _pool_stats(qwen_pool),
    }


@app.get("/v1/models")
async def list_models() -> dict:
    models = [
        {
            "id": name,
            "object": "model",
            "created": 0,
            "owned_by": "deepseek",
            "model_type": model_type,
        }
        for name, model_type in MODEL_TYPE_BY_NAME.items()
    ]
    qwen_models: list[dict] = getattr(app.state, "qwen_models", [])
    for model in qwen_models:
        models.append(
            {
                "id": model["id"],
                "object": "model",
                "created": 0,
                "owned_by": model.get("owned_by", "qwen"),
                "name": model.get("name"),
                "model_type": model.get("model_type", "chat"),
            }
        )
    return {"object": "list", "data": models}


RETRYABLE_FINISH_REASONS = {
    "expert_busy_use_default",
    "parallel_chat_limit",
    "server_busy",
    "busy",
}
MAX_RETRIES = 5
RETRY_BACKOFF_SEC = 1.0
RETRY_BACKOFF_MAX_SEC = 8.0

DEEPSEEK_AUTH_ERROR_CODES = {40001, 40002, 40003, 40012, 40029}


async def _human_delay() -> None:
    delay = random.uniform(settings.human_delay_min, settings.human_delay_max)
    if delay > 0:
        await asyncio.sleep(delay)


def _resolve_provider(model: str) -> str:
    if model.startswith("qwen"):
        return "qwen"
    if model in MODEL_TYPE_BY_NAME or model.startswith("deepseek"):
        return "deepseek"
    raise HTTPException(404, f"Unknown model: {model}")


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest) -> Any:
    provider = _resolve_provider(req.model)
    if provider == "qwen":
        return await _chat_completions_qwen(req)
    return await _chat_completions_deepseek(req)


async def _acquire_account(pool: AccountPool, session_id: str | None):
    try:
        return await pool.acquire(session_id, settings.acquire_timeout)
    except AccountPoolBusy:
        raise HTTPException(429, "all accounts are busy, try again later") from None
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc


def _can_reuse_session(account: Any, session_id: str | None, **kwargs: Any) -> bool:
    if not session_id:
        return False
    session = account.sessions.get(session_id)
    return session is not None and account.sessions._reuse(session, session_id, **kwargs)


def _include_usage(req: ChatCompletionRequest) -> bool:
    opts = getattr(req, "stream_options", None)
    return bool((opts or {}).get("include_usage"))


def _deepseek_usage(total: int) -> dict:
    value = max(0, int(total or 0))
    return {"prompt_tokens": 0, "completion_tokens": value, "total_tokens": value}


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def _stream_error_sse(
    chunk_id: str,
    created: int,
    model: str,
    message: str,
    session_key: str | None = None,
    error_finish: str | None = None,
    choice_finish: str | None = None,
) -> tuple[str, str, str]:
    error: dict = {"message": message}
    if error_finish is not None:
        error["finish_reason"] = error_finish
    error_chunk = _sse(
        {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "error": error,
            "choices": [{"index": 0, "delta": {}, "finish_reason": choice_finish or error_finish or "error"}],
        }
    )
    tail_chunk = _sse(
        {
            "id": chunk_id,
            "session_id": session_key,
            "object": "chat.completion.chunk",
            "choices": [],
        }
    )
    return error_chunk, tail_chunk, "data: [DONE]\n\n"


async def _stream_guard(gen, model: str):
    chunk_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    try:
        async for item in gen:
            yield item
    except AccountPoolBusy:
        for line in _stream_error_sse(chunk_id, created, model, "all accounts are busy, try again later"):
            yield line
    except Exception as exc:
        log.exception("stream generator failed: %s", exc)
        for line in _stream_error_sse(chunk_id, created, model, f"stream error: {exc}"):
            yield line


async def _chat_completions_deepseek(req: ChatCompletionRequest) -> Any:
    pool: AccountPool = app.state.pool
    if pool is None:
        raise HTTPException(503, "deepseek provider is not configured")

    model_type = _resolve_model(req.model)
    thinking = req.thinking if req.thinking is not None else (model_type == "expert")
    search = bool(req.search) and model_type == "default"

    context_seq = toolemu.context_sequence(req.messages, user=getattr(req, "user", None))
    if req.session_id:
        account, existing_sid = await _acquire_account(pool, req.session_id)
    else:
        cached_sid = pool.resolve_context(context_seq) if context_seq else None
        account, existing_sid = await _acquire_account(pool, cached_sid)
    has_session = _can_reuse_session(account, existing_sid)

    try:
        prompt, tool_mode = toolemu.build_prompt(
            req.messages,
            getattr(req, "tools", None),
            getattr(req, "tool_choice", None),
            has_session,
            getattr(req, "response_format", None),
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    attachments = _collect_attachments(req)
    _validate_attachments(attachments, model_type)
    ref_file_ids = None
    if attachments:
        ref_file_ids = await _upload_attachments(account, attachments, model_type, thinking)

    common = {
        "account": account,
        "pool": pool,
        "existing_sid": existing_sid,
        "prompt": prompt,
        "model": req.model,
        "model_type": model_type,
        "thinking": thinking,
        "search": search,
        "ref_file_ids": ref_file_ids,
        "tool_schemas": toolemu.tool_schema_map(getattr(req, "tools", None)),
        "tool_mode": tool_mode,
        "include_usage": _include_usage(req),
        "context_seq": context_seq,
        "reduced_prompts": _reduced_prompt_variants(
            req.messages, getattr(req, "tools", None), getattr(req, "tool_choice", None), getattr(req, "response_format", None), prompt
        ),
    }
    if req.stream:
        return StreamingResponse(
            _stream_guard(_stream_openai(lock=account.sem, **common), req.model),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    try:
        return await _collect_non_stream(lock=account.sem, **common)
    except AccountPoolBusy:
        raise HTTPException(429, "all accounts are busy, try again later") from None


async def _chat_completions_qwen(req: ChatCompletionRequest) -> Any:
    pool: AccountPool = app.state.qwen_pool
    if pool is None:
        raise HTTPException(503, "qwen provider is not configured")

    thinking = req.thinking if req.thinking is not None else True
    search = bool(req.search)

    context_seq = toolemu.context_sequence(req.messages, user=getattr(req, "user", None))
    if req.session_id:
        account, existing_sid = await _acquire_account(pool, req.session_id)
    else:
        cached_sid = pool.resolve_context(context_seq) if context_seq else None
        account, existing_sid = await _acquire_account(pool, cached_sid)
    has_session = _can_reuse_session(account, existing_sid, model=req.model)

    try:
        prompt, tool_mode = toolemu.build_prompt(
            req.messages,
            getattr(req, "tools", None),
            getattr(req, "tool_choice", None),
            has_session,
            getattr(req, "response_format", None),
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    common = {
        "account": account,
        "pool": pool,
        "existing_sid": existing_sid,
        "prompt": prompt,
        "model": req.model,
        "model_id": req.model,
        "thinking": thinking,
        "search": search,
        "tool_schemas": toolemu.tool_schema_map(getattr(req, "tools", None)),
        "tool_mode": tool_mode,
        "include_usage": _include_usage(req),
        "context_seq": context_seq,
    }
    if req.stream:
        return StreamingResponse(
            _stream_guard(qwen_api.stream_openai(lock=account.sem, **common), req.model),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    try:
        return await qwen_api.collect_non_stream(lock=account.sem, **common)
    except AccountPoolBusy:
        raise HTTPException(429, "all accounts are busy, try again later") from None


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
    if session_key != existing_sid:
        pool.register(account.index, session_key)
        if existing_sid:
            pool.forget(existing_sid)
            pool.forget_context(existing_sid)
            account.sessions.forget(existing_sid)
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
    except httpx.HTTPStatusError as exc:
        raise HTTPException(exc.response.status_code, exc.response.text[:500]) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(502, f"DeepSeek request failed: {exc}") from exc

    if resp.status_code != 200:
        body = await resp.aread()
        await resp.aclose()
        raise HTTPException(resp.status_code, body[:500].decode("utf-8", errors="replace"))

    content_type = resp.headers.get("content-type", "")
    if "text/event-stream" not in content_type:
        body = await resp.aread()
        await resp.aclose()
        try:
            payload = json.loads(body)
        except json.JSONDecodeError as exc:
            raise HTTPException(502, body[:500].decode("utf-8", errors="replace")) from exc
        data = payload.get("data") or {}
        if data.get("biz_code"):
            code = data["biz_code"]
            status = 401 if code in DEEPSEEK_AUTH_ERROR_CODES else 502
            raise HTTPException(status, f"DeepSeek error {code}: {data.get('biz_msg')}")
        if payload.get("code"):
            code = payload["code"]
            status = 401 if code in DEEPSEEK_AUTH_ERROR_CODES else 502
            raise HTTPException(
                status,
                f"DeepSeek error {code}: {payload.get('msg') or payload.get('message')}",
            )
        raise HTTPException(502, "unexpected non-stream response")
    return resp


def _is_retryable_hint(rec: MessageReconstructor) -> bool:
    hint = rec.hint_error
    return bool(hint and hint.get("finish_reason") in RETRYABLE_FINISH_REASONS)


RETRYABLE_HTTP_STATUSES = {408, 425, 429, 500, 502, 503, 504}


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


def _input_exceeds_hint_from_http(exc: HTTPException) -> dict | None:
    detail = exc.detail
    if isinstance(detail, str):
        try:
            detail = json.loads(detail)
        except json.JSONDecodeError:
            return None
    if not isinstance(detail, dict):
        return None
    if detail.get("finish_reason") != INPUT_EXCEEDS_LIMIT:
        return None
    message = detail.get("message")
    return {
        "message": message if isinstance(message, str) else "Content is too long",
        "finish_reason": INPUT_EXCEEDS_LIMIT,
    }


def _drop_session(pool, account, session_key) -> None:
    pool.forget(session_key)
    pool.forget_context(session_key)
    account.sessions.forget(session_key)


def _deepseek_status(exc: DeepSeekError) -> int:
    return 401 if exc.biz_code in DEEPSEEK_AUTH_ERROR_CODES else 502


async def _send_with_auth(account, *args, **kwargs):
    try:
        return await _send_completion(*args, **kwargs)
    except HTTPException as exc:
        if exc.status_code in (401, 403):
            account.mark_broken()
        raise


def _deepseek_error_detail(exc: DeepSeekError) -> str:
    if exc.biz_code in DEEPSEEK_AUTH_ERROR_CODES:
        return f"DeepSeek auth error: {exc}"
    return f"DeepSeek error: {exc}"


def _handle_account_error(account: DeepSeekAccount, exc: Exception) -> None:
    code = getattr(exc, "biz_code", None)
    if code in DEEPSEEK_AUTH_ERROR_CODES:
        account.mark_broken()
        log.warning("account #%d auth error %s: %s", account.index, code, exc)
    else:
        log.warning("account #%d error: %s", account.index, exc)


async def _fresh_pow_headers(account) -> dict:
    try:
        return await account.pow.make_header(account.client.create_pow_challenge)
    except DeepSeekError as exc:
        _handle_account_error(account, exc)
        raise HTTPException(_deepseek_status(exc), _deepseek_error_detail(exc)) from exc


def _busy_error_body(rec: MessageReconstructor) -> str:
    hint = rec.hint_error or {}
    return json.dumps(
        {
            "error": {
                "message": hint.get("message") or "DeepSeek server is busy, try again later",
                "finish_reason": hint.get("finish_reason"),
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

# ненавижу
async def _collect_continuation(
    account,
    session,
    parent_message_id,
    model_type,
    thinking,
    search,
    ref_file_ids=None,
) -> MessageReconstructor | None:
    attempt = 0
    while True:
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
                CONTINUE_PROMPT,
                model_type,
                thinking,
                search,
                ref_file_ids,
            )
        except HTTPException as exc:
            if _is_retryable_http(exc) and attempt < MAX_RETRIES:
                attempt += 1
                log.warning(
                    "deepseek continuation error (%s), attempt %d/%d",
                    exc.status_code,
                    attempt,
                    MAX_RETRIES,
                )
                continue
            return None
        rec = MessageReconstructor()
        incremental = IncrementalSSE()
        try:
            async for chunk in resp.aiter_bytes():
                for event in incremental.feed(chunk):
                    rec.handle(event)
            for event in incremental.finish():
                rec.handle(event)
        finally:
            await resp.aclose()
        if not (rec.content or rec.reasoning) and _is_retryable_hint(rec) and attempt < MAX_RETRIES:
            attempt += 1
            continue
        return rec


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
    last_user = None
    for msg in reversed(messages):
        if getattr(msg, "role", None) == "user":
            last_user = msg
            break
    reduced_msgs = list(system_msgs)
    if last_user is not None:
        reduced_msgs.append(last_user)
    if reduced_msgs:
        try:
            prompt, tool_mode = toolemu.build_prompt(reduced_msgs, None, None, False, response_format)
            if prompt != original_prompt:
                variants.append((prompt, tool_mode, {}))
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
    for prompt, _tool_mode, _tool_schemas in reduced_prompts:
        session_key = None
        try:
            session, session_key, parent_message_id = await _prepare_session(account, pool, None, None)
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
            try:
                async for chunk in resp.aiter_bytes():
                    for event in incremental.feed(chunk):
                        rec.handle(event)
                for event in incremental.finish():
                    rec.handle(event)
            finally:
                await resp.aclose()
        except (HTTPException, httpx.HTTPError):
            if session_key is not None:
                _drop_session(pool, account, session_key)
            continue
        if (rec.content or rec.reasoning) and not _is_input_exceeds_limit(rec):
            return rec, session, session_key
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
    tool_mode=False,
    tool_schemas=None,
    include_usage=False,
    context_seq: tuple[str, ...] | None = None,
    reduced_prompts: list[tuple[str, bool, dict[str, Any]]] | None = None,
):
    await _human_delay()
    async with account_lock(lock, settings.acquire_timeout):
        session, session_key, parent_message_id = await _prepare_session(account, pool, existing_sid, context_seq)
        stop_message_id: str | None = None
        started = time.monotonic()
        try:
            rec: MessageReconstructor | None = None
            response_message_id = None
            attempt = 0
            while True:
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
                        attempt += 1
                        log.warning(
                            "deepseek provider error (%s), attempt %d/%d",
                            exc.status_code,
                            attempt,
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
                    attempt += 1
                    log.warning(
                        "retryable hint (%s), attempt %d/%d",
                        (rec.hint_error or {}).get("finish_reason"),
                        attempt,
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
                if not cont_rec.content:
                    break
        if _is_input_exceeds_limit(rec) and not (rec.content or rec.reasoning) and reduced_prompts:
            _drop_session(pool, account, session_key)
            reduced = await _collect_reduced(account, pool, reduced_prompts, model_type, thinking, search, ref_file_ids)
            if reduced is not None:
                rec, session, session_key = reduced
                response_message_id = rec.id or response_message_id
                stop_message_id = response_message_id
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
                if tool_calls:
                    finish = "tool_calls"
                    message = toolemu.format_tool_message(tool_calls, tool_text, reasoning)
                else:
                    message = {"role": "assistant", "content": content}
                    if reasoning:
                        message["reasoning_content"] = reasoning
                    finish = _finish_reason(rec.status)
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
    reduced_prompts: list[tuple[str, bool, dict[str, Any]]] | None = None,
):
    chunk_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    await _human_delay()
    async with account_lock(lock, settings.acquire_timeout):
        try:
            session, session_key, parent_message_id = await _prepare_session(account, pool, existing_sid, context_seq)
        except HTTPException as exc:
            detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
            for line in _stream_error_sse(chunk_id, created, model, detail):
                yield line
            return

        rec: MessageReconstructor | None = None
        response_message_id = None
        stop_message_id: str | None = None
        content_buf = ""
        started = time.monotonic()
        attempt = 0
        while True:
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
                    attempt += 1
                    log.warning(
                        "deepseek provider error (%s), attempt %d/%d",
                        exc.status_code,
                        attempt,
                        MAX_RETRIES,
                    )
                    continue
                detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
                for line in _stream_error_sse(chunk_id, created, model, detail, session_key):
                    yield line
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
                attempt += 1
                log.warning(
                    "retryable hint (%s), attempt %d/%d",
                    (rec.hint_error or {}).get("finish_reason"),
                    attempt,
                    MAX_RETRIES,
                )
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
                if not cont_rec.content:
                    break
        if _is_input_exceeds_limit(rec) and not (rec.content or rec.reasoning) and reduced_prompts:
            _drop_session(pool, account, session_key)
            reduced = await _collect_reduced(account, pool, reduced_prompts, model_type, thinking, search, ref_file_ids)
            if reduced is not None:
                rec, session, session_key = reduced
                response_message_id = rec.id or response_message_id
                stop_message_id = response_message_id
                if rec.content:
                    yield _sse(
                        {
                            "id": chunk_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [{"index": 0, "delta": {"content": rec.content}, "finish_reason": None}],
                        }
                    )
                if rec.reasoning:
                    yield _sse(
                        {
                            "id": chunk_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [{"index": 0, "delta": {"reasoning_content": rec.reasoning}, "finish_reason": None}],
                        }
                    )
        if not (rec.content or rec.reasoning) and rec.hint_error:
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

        session.accumulated_tokens = max(getattr(session, "accumulated_tokens", 0) or 0, rec.accumulated_tokens)
        account.sessions.touch_last_message(session_key, rec.id or response_message_id)
        log.info("deepseek completion OK (%.0fms)", (time.monotonic() - started) * 1000)

        if tool_mode:
            parsed = toolemu.parse_tool_calls(content_buf or rec.content, tool_schemas)
            if parsed is not None:
                tool_calls, tool_text = parsed
                if tool_calls:
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
