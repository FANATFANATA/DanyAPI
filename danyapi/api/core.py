from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import time
import uuid
import weakref
from collections.abc import Iterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from ..accounts import AccountPool, AccountPoolBusy, DeepSeekAccount
from ..alice.accounts import AliceAccount
from ..alice.client import AliceClient
from ..config import settings
from ..deepseek.client import DeepSeekClient
from ..duckai.accounts import DuckAIAccount
from ..duckai.client import DuckAIClient
from ..gigachat.accounts import GigaChatAccount
from ..gigachat.client import GigaChatClient
from ..qwen.accounts import QwenAccount
from ..qwen.client import QwenClient
from ..store import JsonStore
from ..tokens import count_messages_tokens
from ..usage import init_tracker
from .models import model_refresh_loop, refresh_models
from .state import BYOK_PROVIDERS, MODEL_ATTRS, app

log = logging.getLogger("danyapi.api")

POOL_ATTRS = ("pool", "qwen_pool", "gigachat_pool", "alice_pool", "duckai_pool")


def _token_stable_id(token: str) -> str:
    return hashlib.sha1(token.encode("utf-8"), usedforsecurity=False).hexdigest()[:16]


_STATE_STORE_ATTRS = (
    "deepseek_session_store",
    "qwen_session_store",
    "deepseek_context_store",
    "qwen_context_store",
    "deepseek_affinity_store",
    "qwen_affinity_store",
    "responses_store",
)


def _iter_pools() -> Iterator[Any]:
    for attr in POOL_ATTRS:
        pool_obj = getattr(app.state, attr, None)
        if pool_obj is not None:
            yield pool_obj
    byok_pools = getattr(app.state, "byok_pools", None)
    if not isinstance(byok_pools, dict):
        return
    for entries in byok_pools.values():
        if isinstance(entries, dict):
            yield from entries.values()


def _flush_state_stores() -> None:
    tracker = getattr(app.state, "usage", None)
    if tracker is not None:
        try:
            tracker.flush()
        except Exception as exc:
            log.debug("usage flush failed: %s", exc)
    for attr in _STATE_STORE_ATTRS:
        store = getattr(app.state, attr, None)
        if store is None:
            continue
        try:
            store.flush()
        except Exception as exc:
            log.debug("store flush failed for %s: %s", attr, exc)
    seen_pools: set[int] = set()
    for pool_obj in _iter_pools():
        if id(pool_obj) in seen_pools:
            continue
        seen_pools.add(id(pool_obj))
        flush = getattr(pool_obj, "flush", None)
        if flush is None:
            continue
        try:
            flush()
        except Exception as exc:
            log.debug("pool flush failed: %s", exc)


def _close_pow_managers(accts: list[Any]) -> None:
    seen: set[int] = set()
    for acct in accts:
        if id(acct) in seen:
            continue
        seen.add(id(acct))
        for attr in ("pow", "pow_upload"):
            close = getattr(getattr(acct, attr, None), "close", None)
            if close is None:
                continue
            try:
                close()
            except Exception as exc:
                log.debug("pow manager close failed for %s: %s", getattr(acct, "label", acct), exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    accounts: list[DeepSeekAccount] = []
    qwen_accounts: list[QwenAccount] = []
    gigachat_accounts: list[GigaChatAccount] = []
    alice_accounts: list[AliceAccount] = []
    duckai_accounts: list[DuckAIAccount] = []
    byok_mode = settings.byok
    app.state.byok = byok_mode
    app.state.byok_pools = {provider: {} for provider in BYOK_PROVIDERS}
    app.state.byok_locks = {provider: asyncio.Lock() for provider in BYOK_PROVIDERS}
    app.state.byok_auth = {provider: {} for provider in BYOK_PROVIDERS}
    cache_enabled = settings.cache_enabled
    deepseek_session_store = JsonStore("deepseek-sessions", "default" if cache_enabled else None)
    qwen_session_store = JsonStore("qwen-sessions", "default" if cache_enabled else None)
    deepseek_context_store = JsonStore("deepseek-contexts", "default" if cache_enabled else None)
    qwen_context_store = JsonStore("qwen-contexts", "default" if cache_enabled else None)
    deepseek_affinity_store = JsonStore("deepseek-affinities", "default" if cache_enabled else None)
    qwen_affinity_store = JsonStore("qwen-affinities", "default" if cache_enabled else None)
    responses_store = JsonStore("responses", "default" if cache_enabled else None, maxsize=settings.responses_max_records)
    app.state.responses_store = responses_store
    app.state.deepseek_session_store = deepseek_session_store
    app.state.qwen_session_store = qwen_session_store
    app.state.deepseek_context_store = deepseek_context_store
    app.state.qwen_context_store = qwen_context_store
    app.state.deepseek_affinity_store = deepseek_affinity_store
    app.state.qwen_affinity_store = qwen_affinity_store
    for provider in BYOK_PROVIDERS:
        setattr(app.state, MODEL_ATTRS[provider], [])
    if settings.usage_enabled:
        app.state.usage = init_tracker(store=JsonStore("usage", "default"), max_records=settings.usage_max_records)
    else:
        app.state.usage = None
    try:
        if not byok_mode:
            ds_clients = [DeepSeekClient(token=token, timeout=settings.timeout) for token in settings.deepseek_tokens] if settings.deepseek_tokens else []
            qw_clients = [QwenClient(token=token, timeout=settings.timeout) for token in settings.qwen_tokens] if settings.qwen_tokens else []
            gc_clients: list[GigaChatClient] = []
            for key in settings.gigachat_keys:
                try:
                    gc_clients.append(GigaChatClient(key=key, scope=settings.gigachat_scope, timeout=settings.timeout))
                except (RuntimeError, OSError) as exc:
                    log.error("gigachat client disabled, CA unusable: %s", exc)
            alice_clients: list[AliceClient] = []
            if settings.alice_enabled:
                for _ in range(settings.alice_accounts):
                    alice_clients.append(AliceClient(timeout=settings.timeout))
            duckai_clients: list[DuckAIClient] = []
            if settings.duckai_enabled:
                for _ in range(settings.duckai_accounts):
                    duckai_clients.append(DuckAIClient(timeout=settings.timeout))
            ds_checks = [client.check_auth() for client in ds_clients]
            qw_checks = [client.check_auth() for client in qw_clients]
            gc_checks = [client.check_auth() for client in gc_clients]
            alice_checks = [client.check_auth() for client in alice_clients]
            duckai_checks = [client.check_auth() for client in duckai_clients]
            if ds_checks or qw_checks or gc_checks or alice_checks or duckai_checks:
                auth_results = await asyncio.gather(
                    *(ds_checks + qw_checks + gc_checks + alice_checks + duckai_checks),
                    return_exceptions=True,
                )
                for index, outcome in enumerate(auth_results):
                    if isinstance(outcome, BaseException):
                        log.warning("auth check #%d failed: %s", index, outcome)
                auth_flags = [outcome is True for outcome in auth_results]
                ds_end = len(ds_checks)
                qw_end = ds_end + len(qw_checks)
                gc_end = qw_end + len(gc_checks)
                ds_auth = auth_flags[:ds_end]
                qw_auth = auth_flags[ds_end:qw_end]
                gc_auth = auth_flags[qw_end:gc_end]
                alice_auth = auth_flags[gc_end:]
                duckai_auth = auth_flags[gc_end + len(alice_checks) :]
            else:
                ds_auth = []
                qw_auth = []
                gc_auth = []
                alice_auth = []
                duckai_auth = []
            if settings.deepseek_tokens:
                for i, (token, ds_client, ok) in enumerate(zip(settings.deepseek_tokens, ds_clients, ds_auth, strict=True)):
                    if not ok:
                        log.warning("deepseek token #%d invalid/expired, skipping", i)
                        await ds_client.aclose()
                        continue
                    accounts.append(
                        DeepSeekAccount(
                            len(accounts),
                            ds_client,
                            session_cache_size=settings.session_cache_size,
                            ttl=settings.session_ttl,
                            store=deepseek_session_store,
                            stable_id=_token_stable_id(token),
                        )
                    )
                log.info("deepseek accounts ready: %d", len(accounts))
            if settings.qwen_tokens:
                for i, (token, qw_client, ok) in enumerate(zip(settings.qwen_tokens, qw_clients, qw_auth, strict=True)):
                    if not ok:
                        log.warning("qwen token #%d invalid/expired, skipping", i)
                        await qw_client.aclose()
                        continue
                    qwen_accounts.append(
                        QwenAccount(
                            len(qwen_accounts),
                            qw_client,
                            session_cache_size=settings.session_cache_size,
                            ttl=settings.session_ttl,
                            store=qwen_session_store,
                            stable_id=_token_stable_id(token),
                        )
                    )
                log.info("qwen accounts ready: %d", len(qwen_accounts))
            if settings.gigachat_keys:
                for i, (key, gc_client, ok) in enumerate(zip(settings.gigachat_keys, gc_clients, gc_auth, strict=False)):
                    if not ok:
                        log.warning("gigachat key #%d invalid/expired, skipping", i)
                        await gc_client.aclose()
                        continue
                    gigachat_accounts.append(
                        GigaChatAccount(
                            len(gigachat_accounts),
                            gc_client,
                            stable_id=_token_stable_id(key),
                        )
                    )
                log.info("gigachat accounts ready: %d", len(gigachat_accounts))
            for i, (alice_client, ok) in enumerate(zip(alice_clients, alice_auth, strict=False)):
                if not ok:
                    log.warning("alice endpoint unreachable, skipping account #%d", i)
                    await alice_client.aclose()
                    continue
                alice_accounts.append(AliceAccount(len(alice_accounts), alice_client, stable_id="alice"))
            if alice_clients:
                log.info("alice accounts ready: %d", len(alice_accounts))
            for i, (duckai_client, ok) in enumerate(zip(duckai_clients, duckai_auth, strict=False)):
                if not ok:
                    log.warning("duckai bot check missed on account #%d, keeping it and relying on retries", i)
                duckai_accounts.append(DuckAIAccount(len(duckai_accounts), duckai_client, stable_id="duckai"))
            if duckai_clients:
                log.info("duckai accounts ready: %d", len(duckai_accounts))
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
        else:
            app.state.qwen_pool = None
        if gigachat_accounts:
            app.state.gigachat_pool = AccountPool(gigachat_accounts, label="gigachat")
        else:
            app.state.gigachat_pool = None
        if alice_accounts:
            app.state.alice_pool = AccountPool(alice_accounts, label="alice")
        else:
            app.state.alice_pool = None
        if duckai_accounts:
            app.state.duckai_pool = AccountPool(duckai_accounts, label="duckai")
        else:
            app.state.duckai_pool = None
        if not accounts and not qwen_accounts and not gigachat_accounts and not alice_accounts and not duckai_accounts and not byok_mode:
            raise RuntimeError("no valid credentials: set DEEPSEEK_TOKENS, QWEN_TOKENS, GIGACHAT_KEYS, ALICE_ENABLED=1 or DUCKAI_ENABLED=1")
        await refresh_models()
        refresh_task = asyncio.create_task(model_refresh_loop())
        try:
            yield
        finally:
            refresh_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await refresh_task
    finally:
        http_client = getattr(app.state, "http_client", None)
        if http_client is not None:
            await http_client.aclose()
        all_accounts: list[Any] = [*accounts, *qwen_accounts, *gigachat_accounts, *alice_accounts, *duckai_accounts]
        for pool_obj in _iter_pools():
            all_accounts.extend(pool_obj.accounts)
        _close_pow_managers(all_accounts)
        await asyncio.to_thread(_flush_state_stores)
        seen: set[int] = set()
        for acct in all_accounts:
            client = acct.client
            if id(client) in seen:
                continue
            seen.add(id(client))
            await client.aclose()


def _shared_store(attr: str, name: str, *, maxsize: int = 0) -> JsonStore:
    store = getattr(app.state, attr, None)
    if store is None:
        store = JsonStore(name, "default" if settings.cache_enabled else None, maxsize=maxsize)
        setattr(app.state, attr, store)
    return store


def _responses_store() -> JsonStore:
    return _shared_store("responses_store", "responses", maxsize=settings.responses_max_records)


MAX_LOGGED_BODY = 256 * 1024
MAX_REQUEST_BODY = 100 * 1024 * 1024


async def _read_request_body(request: Request, limit: int) -> bytes:
    raw_length = -1
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            raw_length = int(content_length)
        except ValueError:
            raw_length = -1
    if raw_length > 0:
        if raw_length > limit:
            raise HTTPException(413, "request body too large")
        body = await request.body()
        if len(body) > limit:
            raise HTTPException(413, "request body too large")
        request._body = body
        return body
    cached = getattr(request, "_body", None)
    if cached:
        if len(cached) > limit:
            raise HTTPException(413, "request body too large")
        return cached
    chunks: list[bytes] = []
    total = 0
    async for chunk in request.stream():
        total += len(chunk)
        if total > limit:
            raise HTTPException(413, "request body too large")
        chunks.append(chunk)
    body = b"".join(chunks)
    request._body = body
    return body


def _parse_logged_body(body: bytes) -> dict[str, Any]:
    if not body or len(body) > MAX_LOGGED_BODY:
        return {}
    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError, TypeError):
        return {}
    if isinstance(payload, dict):
        return payload
    return {}


async def _extract_request_body(request: Request) -> dict[str, Any]:
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            raw_length = int(content_length)
            if raw_length <= 0:
                return {}
            if raw_length > MAX_LOGGED_BODY:
                return {}
        except ValueError:
            return {}
    if getattr(request, "method", None) in ("GET", "DELETE", "HEAD", "OPTIONS"):
        return {}
    cached = getattr(request, "_body", b"")
    if cached:
        return _parse_logged_body(cached)
    if content_length is None:
        return {}
    try:
        body = await _read_request_body(request, MAX_REQUEST_BODY)
    except HTTPException:
        raise
    except Exception:
        return {}
    return _parse_logged_body(body)


def _request_client_ip(request: Request) -> str:
    headers = request.headers
    forwarded = headers.get("x-forwarded-for")
    if forwarded:
        first = forwarded.split(",", 1)[0].strip()
        if first:
            return first
    real_ip = headers.get("x-real-ip")
    if real_ip:
        return real_ip.strip()
    client = request.client
    if client is not None and client.host:
        return client.host
    return "-"


def _request_details(request: Request, payload: dict[str, Any], count_tokens: bool = True) -> str:
    parts = []
    user_agent = request.headers.get("user-agent")
    if user_agent:
        parts.append(f"ua={user_agent[:120]}")
    model = payload.get("model")
    if isinstance(model, str) and model:
        parts.append(f"model={model}")
    session_id = payload.get("session_id")
    if isinstance(session_id, str) and session_id:
        parts.append(f"sid={session_id}")
    user = payload.get("user")
    if isinstance(user, str) and user:
        parts.append(f"user={user}")
    stream = payload.get("stream")
    if isinstance(stream, bool):
        parts.append(f"stream={int(stream)}")
    messages = payload.get("messages")
    if isinstance(messages, list):
        parts.append(f"msgs={len(messages)}")
        if count_tokens:
            parts.append(f"tokens={count_messages_tokens(messages)}")
    return " ".join(parts)


def _log_request_failure(request: Request, payload: dict[str, Any], duration: float, status: int | None = None, exc: Exception | None = None) -> None:
    if not log.isEnabledFor(logging.WARNING):
        return
    details = _request_details(request, payload, count_tokens=log.isEnabledFor(logging.DEBUG))
    details_part = f" {details}" if details else ""
    ip = _request_client_ip(request)
    if status is not None:
        reason = f"status={status}"
    else:
        reason = f"error={str(exc) if exc else 'unknown'}"
    log.warning(
        "%s %s %s%s failed: %s (%.0fms)",
        request.method,
        request.url.path,
        ip,
        details_part,
        reason,
        duration,
    )


def _log_request_success(request: Request, payload: dict[str, Any], duration: float) -> None:
    if not log.isEnabledFor(logging.INFO):
        return
    details = _request_details(request, payload, count_tokens=log.isEnabledFor(logging.DEBUG))
    details_part = f" {details}" if details else ""
    ip = _request_client_ip(request)
    log.info(
        "%s %s %s%s ok (%.0fms)",
        request.method,
        request.url.path,
        ip,
        details_part,
        duration,
    )


@app.middleware("http")
async def _log_requests(request: Request, call_next):
    started = time.monotonic()
    payload: dict[str, Any] = {}
    try:
        if log.isEnabledFor(logging.INFO) or log.isEnabledFor(logging.WARNING):
            payload = await _extract_request_body(request)
    except HTTPException as exc:
        _log_request_failure(
            request,
            payload,
            (time.monotonic() - started) * 1000,
            status=exc.status_code,
        )
        return await _on_http_exception(request, exc)
    try:
        response = await call_next(request)
    except Exception as exc:
        _log_request_failure(
            request,
            payload,
            (time.monotonic() - started) * 1000,
            exc=exc,
        )
        raise
    duration = (time.monotonic() - started) * 1000
    if response.status_code >= 400:
        _log_request_failure(
            request,
            payload,
            duration,
            status=response.status_code,
        )
    else:
        _log_request_success(request, payload, duration)
    return response


def _error_type_for_status(status: int) -> str:
    if status == 401:
        return "authentication_error"
    if status == 403:
        return "permission_error"
    if status == 404:
        return "not_found_error"
    if status == 408:
        return "request_timeout"
    if status == 409:
        return "conflict_error"
    if status == 413:
        return "request_too_large"
    if status == 429:
        return "rate_limit_error"
    if status == 501 or status == 503:
        return "api_error"
    if status == 502 or status == 504:
        return "server_error"
    if status >= 500:
        return "server_error"
    return "invalid_request_error"


def _error_code_for_status(status: int) -> str | None:
    if status == 429:
        return "rate_limit_exceeded"
    if status == 400:
        return "invalid_request_error"
    return None


INTERNAL_ERROR_MESSAGE = "internal server error"


def _exception_message(exc: Exception) -> str:
    log.exception("unhandled api error: %s", exc)
    return INTERNAL_ERROR_MESSAGE


def _error_detail(message: str, finish_reason: Any = None) -> dict:
    return {"error": {"message": message, "finish_reason": finish_reason}}


def _openai_error_payload(status: int, message: str, request_id: str | None = None) -> dict:
    payload = {
        "error": {
            "message": message,
            "type": _error_type_for_status(status),
            "param": None,
            "code": _error_code_for_status(status),
        }
    }
    if request_id:
        payload["error"]["request_id"] = request_id
    return payload


def _request_id_header(request: Request) -> str:
    provided = request.headers.get("x-request-id")
    if provided and len(provided) <= 128:
        return provided
    return uuid.uuid4().hex


@app.exception_handler(RequestValidationError)
async def _on_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    request_id = _request_id_header(request)
    return JSONResponse(
        status_code=400,
        content=_openai_error_payload(400, f"invalid request body: {exc.errors()}", request_id),
        headers={"x-request-id": request_id},
    )


@app.exception_handler(HTTPException)
async def _on_http_exception(request: Request, exc: HTTPException) -> JSONResponse:
    request_id = _request_id_header(request)
    headers = {"x-request-id": request_id}
    if exc.headers:
        headers.update({str(k): str(v) for k, v in exc.headers.items()})
    detail = exc.detail
    if isinstance(detail, dict):
        inner = detail.get("error")
        if not isinstance(inner, dict):
            inner = detail
        message = inner.get("message")
        content = _openai_error_payload(exc.status_code, message if isinstance(message, str) else str(detail), request_id)
        content["error"].update({key: value for key, value in inner.items() if key != "message" and value is not None})
    else:
        content = _openai_error_payload(exc.status_code, str(detail), request_id)
    return JSONResponse(
        status_code=exc.status_code,
        content=content,
        headers=headers,
    )


@app.exception_handler(Exception)
async def _on_uncaught_exception(request: Request, exc: Exception) -> JSONResponse:
    request_id = _request_id_header(request)
    return JSONResponse(
        status_code=500,
        content=_openai_error_payload(500, _exception_message(exc), request_id),
        headers={"x-request-id": request_id},
    )


def _account_busy_count(pool: Any) -> int:
    busy = 0
    for acct in getattr(pool, "accounts", None) or []:
        sem = getattr(acct, "sem", None)
        if sem is not None and sem.locked():
            busy += 1
    return busy


_POOL_RATE_CACHE: dict[int, tuple[float, dict[str, str], weakref.ReferenceType[Any]]] = {}
_POOL_RATE_TTL = 1.0
_POOL_RATE_CACHE_MAX = 16


def _pool_rate_headers(pool: Any | None) -> dict[str, str]:
    if pool is None or not hasattr(pool, "stats"):
        return {}
    now = time.monotonic()
    key = id(pool)
    entry = _POOL_RATE_CACHE.get(key)
    if entry is not None and entry[2]() is pool and now - entry[0] < _POOL_RATE_TTL:
        return entry[1]
    try:
        total = int(pool.stats().get("healthy", 0) or 0)
    except Exception:
        return {}
    busy = _account_busy_count(pool)
    headers = {
        "x-ratelimit-limit-requests": str(max(total, 0)),
        "x-ratelimit-remaining-requests": str(max(total - busy, 0)),
        "x-ratelimit-reset-requests": str(int(time.time())),
    }
    if len(_POOL_RATE_CACHE) > _POOL_RATE_CACHE_MAX:
        _POOL_RATE_CACHE.clear()
    try:
        _POOL_RATE_CACHE[key] = (now, headers, weakref.ref(pool))
    except TypeError:
        _POOL_RATE_CACHE.pop(key, None)
    return headers


@app.middleware("http")
async def _openai_headers(request: Request, call_next):
    response = await call_next(request)
    headers = response.headers
    if not headers.get("x-request-id"):
        headers["x-request-id"] = _request_id_header(request)
    if not headers.get("x-ratelimit-limit-requests"):
        for attr in POOL_ATTRS:
            candidate = getattr(app.state, attr, None)
            if candidate is None:
                continue
            for key, value in _pool_rate_headers(candidate).items():
                headers[key] = value
            break
    return response


async def _acquire_account(pool: AccountPool, session_id: str | None):
    try:
        return await pool.acquire(session_id, settings.acquire_timeout)
    except AccountPoolBusy:
        raise HTTPException(429, "all accounts are busy, try again later") from None
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc


cors_origins = settings.cors_origins or ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=bool(settings.cors_origins),
    allow_methods=["*"],
    allow_headers=["*"],
)

docs_path = Path(__file__).resolve().parents[2] / "docs"
if docs_path.is_dir():
    app.mount("/docs", StaticFiles(directory=str(docs_path), html=True), name="docs")

app.router.lifespan_context = lifespan
