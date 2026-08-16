from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from .. import tools as toolemu
from ..accounts import AccountPool, AccountPoolBusy, DeepSeekAccount
from ..config import settings
from ..deepseek.client import DeepSeekClient
from ..qwen import api as qwen_api
from ..qwen.accounts import QwenAccount
from ..qwen.client import QwenClient, QwenError
from ..store import JsonStore

# Import ChatCompletionRequest early so it's available for route signatures.
# The full re-export block at the bottom brings in every other name used by tests.
from .models import ChatCompletionRequest

log = logging.getLogger("danyapi.api")


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

    try:
        prompt, tool_mode = toolemu.build_prompt(
            req.messages,
            getattr(req, "tools", None),
            getattr(req, "tool_choice", None),
            existing_sid is not None,
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

    try:
        prompt, tool_mode = toolemu.build_prompt(
            req.messages,
            getattr(req, "tools", None),
            getattr(req, "tool_choice", None),
            existing_sid is not None,
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


# ---------------------------------------------------------------------------
# Re-exports: submodules of this package (models / attachments / deepseek /
# streaming). Kept here so `danyapi.api.openai` exposes the same names as
# before the split, and module-level monkeypatching in tests keeps working.
# ---------------------------------------------------------------------------

from .attachments import (  # noqa: E402,F401
    MAX_FILE_SIZE,
    MAX_FILES_PER_REQUEST,
    Attachment,
    _collect_attachments,
    _fresh_pow_upload_headers,
    _split_data_uri,
    _upload_attachments,
    _validate_attachments,
)
from .deepseek import (  # noqa: E402,F401
    DEEPSEEK_AUTH_ERROR_CODES,
    MAX_RETRIES,
    RETRY_BACKOFF_MAX_SEC,
    RETRY_BACKOFF_SEC,
    RETRYABLE_FINISH_REASONS,
    _busy_error_body,
    _collect_continuation,
    _deepseek_error_detail,
    _deepseek_status,
    _drop_session,
    _fresh_pow_headers,
    _handle_account_error,
    _human_delay,
    _input_exceeds_hint_from_http,
    _is_context_limit,
    _is_input_exceeds_limit,
    _is_retryable_hint,
    _is_retryable_http,
    _prepare_session,
    _send_completion,
    _send_with_auth,
    _try_stop_stream,
)
from .models import (  # noqa: E402,F401
    MODEL_TYPE_BY_NAME,
    QWEN_DEFAULT_MODELS,
    ChatMessage,
    FileSpec,
    _deepseek_usage,
    _finish_reason,
    _include_usage,
    _resolve_model,
)
from .streaming import (  # noqa: E402,F401
    _collect_non_stream,
    _sse,
    _stream_guard,
    _stream_openai,
)
