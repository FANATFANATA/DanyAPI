from __future__ import annotations

import asyncio
import contextlib
import hashlib
import ipaddress
import json
import logging
import posixpath
import re
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
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.types import Scope

from ..accounts import AccountPool, AccountPoolBusy, DeepSeekAccount
from ..alice.accounts import AliceAccount
from ..alice.client import AliceClient
from ..config import settings
from ..deepseek.client import DeepSeekClient
from ..duckai.accounts import DuckAIAccount
from ..duckai.client import DuckAIClient
from ..gigachat.accounts import GigaChatAccount
from ..gigachat.client import GigaChatClient
from ..mistral.accounts import MistralChatAccount
from ..mistral.client import MistralChatClient
from ..opencode.accounts import OpenCodeAccount
from ..opencode.client import OpenCodeClient
from ..qwen.accounts import QwenAccount
from ..qwen.client import QwenClient
from ..store import JsonStore
from ..tokens import count_messages_tokens
from ..usage import init_tracker
from .attachments import MAX_ATTACHMENT_TOTAL_SIZE
from .models import _resolve_provider, model_refresh_loop, refresh_models
from .state import (
    BYOK_PROVIDERS,
    MODEL_ATTRS,
    POOL_ATTRS_BY_PROVIDER,
    _blank_byok_locks,
    _blank_byok_state,
    _blank_byok_stores,
    app,
    provider_pool,
)

log = logging.getLogger("danyapi.api")

POOL_ATTRS = tuple(POOL_ATTRS_BY_PROVIDER[provider] for provider in BYOK_PROVIDERS)


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
    opencode_accounts: list[OpenCodeAccount] = []
    alice_accounts: list[AliceAccount] = []
    duckai_accounts: list[DuckAIAccount] = []
    mistral_accounts: list[MistralChatAccount] = []
    byok_mode = settings.byok
    app.state.byok = byok_mode
    app.state.byok_pools = _blank_byok_state()
    app.state.byok_locks = _blank_byok_locks()
    app.state.byok_auth = _blank_byok_state()
    app.state.byok_stores = _blank_byok_stores()
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
            ds_tokens = settings.deepseek_tokens if settings.provider_enabled("deepseek") else []
            qw_tokens = settings.qwen_tokens if settings.provider_enabled("qwen") else []
            ds_clients = [DeepSeekClient(token=token, timeout=settings.timeout) for token in ds_tokens]
            qw_clients = [QwenClient(token=token, timeout=settings.timeout) for token in qw_tokens]
            gc_clients: list[GigaChatClient] = []
            gc_key_indexes: list[int] = []
            if settings.provider_enabled("gigachat"):
                for key_index, key in enumerate(settings.gigachat_keys):
                    try:
                        gc_clients.append(GigaChatClient(key=key, scope=settings.gigachat_scope, timeout=settings.timeout))
                        gc_key_indexes.append(key_index)
                    except (RuntimeError, OSError) as exc:
                        log.error("gigachat client disabled, CA unusable: %s", exc)
            oc_clients: list[OpenCodeClient] = []
            if settings.provider_enabled("opencode"):
                oc_clients = [OpenCodeClient(key=key, timeout=settings.timeout) for key in settings.opencode_keys]
                if not oc_clients and settings.opencode_enabled:
                    oc_clients.append(OpenCodeClient(timeout=settings.timeout))
            alice_clients: list[AliceClient] = []
            if settings.alice_enabled and settings.provider_enabled("alice"):
                for _ in range(settings.alice_accounts):
                    alice_clients.append(AliceClient(timeout=settings.timeout))
            duckai_clients: list[DuckAIClient] = []
            if settings.duckai_enabled and settings.provider_enabled("duckai"):
                for _ in range(settings.duckai_accounts):
                    duckai_clients.append(DuckAIClient(timeout=settings.timeout))
            mistral_clients: list[MistralChatClient] = []
            if settings.mistral_enabled and settings.provider_enabled("mistral") and settings.mistral_logins:
                for login in settings.mistral_logins:
                    mistral_clients.append(MistralChatClient(timeout=settings.timeout, login=login))
            ds_checks = [client.check_auth() for client in ds_clients]
            qw_checks = [client.check_auth() for client in qw_clients]
            gc_checks = [client.check_auth() for client in gc_clients]
            oc_checks = [client.check_auth() for client in oc_clients]
            alice_checks = [client.check_auth() for client in alice_clients]
            duckai_checks = [client.check_auth() for client in duckai_clients]
            mistral_checks = [client.check_auth() for client in mistral_clients]
            groups = (
                ("deepseek", ds_checks),
                ("qwen", qw_checks),
                ("gigachat", gc_checks),
                ("opencode", oc_checks),
                ("alice", alice_checks),
                ("duckai", duckai_checks),
                ("mistral", mistral_checks),
            )
            pending = [check for _name, checks in groups for check in checks]
            auth_by_provider: dict[str, list[bool]] = {name: [] for name, _checks in groups}
            if pending:
                auth_results = await asyncio.gather(*pending, return_exceptions=True)
                for index, outcome in enumerate(auth_results):
                    if isinstance(outcome, BaseException):
                        log.warning("auth check #%d failed: %s", index, outcome)
                offset = 0
                for name, checks in groups:
                    auth_by_provider[name] = [outcome is True for outcome in auth_results[offset : offset + len(checks)]]
                    offset += len(checks)
            ds_auth = auth_by_provider["deepseek"]
            qw_auth = auth_by_provider["qwen"]
            gc_auth = auth_by_provider["gigachat"]
            oc_auth = auth_by_provider["opencode"]
            alice_auth = auth_by_provider["alice"]
            duckai_auth = auth_by_provider["duckai"]
            mistral_auth = auth_by_provider["mistral"]
            if ds_tokens:
                for i, (token, ds_client, ok) in enumerate(zip(ds_tokens, ds_clients, ds_auth, strict=True)):
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
            if qw_tokens:
                for i, (token, qw_client, ok) in enumerate(zip(qw_tokens, qw_clients, qw_auth, strict=True)):
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
                for key_index, gc_client, ok in zip(gc_key_indexes, gc_clients, gc_auth, strict=True):
                    if not ok:
                        log.warning("gigachat key #%d invalid/expired, skipping", key_index)
                        await gc_client.aclose()
                        continue
                    gigachat_accounts.append(
                        GigaChatAccount(
                            len(gigachat_accounts),
                            gc_client,
                            stable_id=_token_stable_id(settings.gigachat_keys[key_index]),
                        )
                    )
                log.info("gigachat accounts ready: %d", len(gigachat_accounts))
            for i, (oc_client, ok) in enumerate(zip(oc_clients, oc_auth, strict=True)):
                if not ok:
                    log.warning("opencode key #%d could not reach the Zen catalog, skipping", i)
                    await oc_client.aclose()
                    continue
                stable_id = _token_stable_id(settings.opencode_keys[i]) if i < len(settings.opencode_keys) else "opencode"
                opencode_accounts.append(OpenCodeAccount(len(opencode_accounts), oc_client, stable_id=stable_id))
            if opencode_accounts:
                log.info("opencode accounts ready: %d", len(opencode_accounts))
            for i, (alice_client, ok) in enumerate(zip(alice_clients, alice_auth, strict=False)):
                if not ok:
                    log.warning("alice endpoint unreachable, skipping account #%d", i)
                    await alice_client.aclose()
                    continue
                alice_accounts.append(AliceAccount(len(alice_accounts), alice_client, stable_id="alice"))
            if alice_accounts:
                log.info("alice accounts ready: %d", len(alice_accounts))
            for i, (duckai_client, ok) in enumerate(zip(duckai_clients, duckai_auth, strict=False)):
                if not ok:
                    log.warning("duckai bot check missed on account #%d, keeping it and relying on retries", i)
                duckai_accounts.append(DuckAIAccount(len(duckai_accounts), duckai_client, stable_id="duckai"))
            if duckai_clients:
                log.info("duckai accounts ready: %d", len(duckai_accounts))
            for i, (mistral_client, ok) in enumerate(zip(mistral_clients, mistral_auth, strict=False)):
                if not ok:
                    log.warning("mistral le chat unreachable on account #%d, skipping it", i)
                    await mistral_client.aclose()
                    continue
                mistral_accounts.append(MistralChatAccount(len(mistral_accounts), mistral_client, stable_id="mistral"))
            if mistral_clients:
                log.info("mistral accounts ready: %d", len(mistral_accounts))
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
        if opencode_accounts:
            app.state.opencode_pool = AccountPool(opencode_accounts, label="opencode")
        else:
            app.state.opencode_pool = None
        if alice_accounts:
            app.state.alice_pool = AccountPool(alice_accounts, label="alice")
        else:
            app.state.alice_pool = None
        if duckai_accounts:
            app.state.duckai_pool = AccountPool(duckai_accounts, label="duckai")
        else:
            app.state.duckai_pool = None
        if mistral_accounts:
            app.state.mistral_pool = AccountPool(mistral_accounts, label="mistral")
        else:
            app.state.mistral_pool = None
        if (
            not accounts
            and not qwen_accounts
            and not gigachat_accounts
            and not opencode_accounts
            and not alice_accounts
            and not duckai_accounts
            and not mistral_accounts
            and not byok_mode
        ):
            raise RuntimeError(
                "no valid credentials: set DEEPSEEK_TOKENS, QWEN_TOKENS, GIGACHAT_KEYS, OPENCODE_KEYS, "
                "ALICE_ENABLED=1, OPENCODE_ENABLED=1, DUCKAI_ENABLED=1 or MISTRAL_ENABLED=1"
            )
        await refresh_models()
        refresh_task = asyncio.create_task(model_refresh_loop())
        try:
            yield
        finally:
            refresh_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await refresh_task
    finally:
        all_accounts: list[Any] = [*accounts, *qwen_accounts, *gigachat_accounts, *opencode_accounts, *alice_accounts, *duckai_accounts, *mistral_accounts]
        for pool_obj in _iter_pools():
            all_accounts.extend(pool_obj.accounts)
        await _run_lifespan_cleanup(all_accounts)


async def _run_lifespan_cleanup(all_accounts: list[Any]) -> None:
    task = asyncio.ensure_future(_close_everything(all_accounts))
    cancelled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled = True
        except Exception as exc:
            log.warning("lifespan cleanup failed: %s", exc)
            break
    if cancelled:
        raise asyncio.CancelledError


async def _close_everything(all_accounts: list[Any]) -> None:
    http_client = getattr(app.state, "http_client", None)
    if http_client is not None:
        try:
            await http_client.aclose()
        except Exception as exc:
            log.warning("shared http client close failed: %s", exc)
    _close_pow_managers(all_accounts)
    try:
        await asyncio.to_thread(_flush_state_stores)
    except Exception as exc:
        log.warning("state store flush failed during shutdown: %s", exc)
    seen: set[int] = set()
    for acct in all_accounts:
        client = acct.client
        if id(client) in seen:
            continue
        seen.add(id(client))
        try:
            await client.aclose()
        except Exception as exc:
            log.warning("client close failed for %s: %s", getattr(acct, "label", acct), exc)


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
MAX_CHAT_BODY_BYTES = 3 * MAX_ATTACHMENT_TOTAL_SIZE
CHAT_BODY_PATHS = frozenset({"/v1/chat/completions", "/v1/completions"})

_BODY_LIMIT_SCOPE_KEY = "danyapi_body_limit"


class _BodyLimit:
    __slots__ = ("detail", "exceeded", "limit", "seen")

    def __init__(self, limit: int, detail: str) -> None:
        self.limit = limit
        self.detail = detail
        self.seen = 0
        self.exceeded = False


def _body_limit_for(path: str) -> _BodyLimit:
    if path not in CHAT_BODY_PATHS:
        return _BodyLimit(MAX_REQUEST_BODY, "request body too large")
    if MAX_CHAT_BODY_BYTES < MAX_REQUEST_BODY:
        return _BodyLimit(MAX_CHAT_BODY_BYTES, f"request body too large, max {MAX_CHAT_BODY_BYTES // (1024 * 1024)} MB")
    return _BodyLimit(min(MAX_CHAT_BODY_BYTES, MAX_REQUEST_BODY), "request body too large")


def _install_body_limit(request: Request) -> _BodyLimit:
    scope = getattr(request, "scope", None)
    if not isinstance(scope, dict):
        url = getattr(request, "url", None)
        return _body_limit_for(getattr(url, "path", "") or "")
    state = scope.get(_BODY_LIMIT_SCOPE_KEY)
    if isinstance(state, _BodyLimit):
        return state
    state = _body_limit_for(scope.get("path", ""))
    scope[_BODY_LIMIT_SCOPE_KEY] = state
    receive = request._receive

    async def _bounded_receive() -> Any:
        message: Any = await receive()
        if isinstance(message, dict) and message.get("type") == "http.request":
            state.seen += len(message.get("body") or b"")
            if state.seen > state.limit:
                state.exceeded = True
                raise HTTPException(413, state.detail)
        return message

    request._receive = _bounded_receive
    return state


def _unwrap_http_exception(exc: BaseException) -> HTTPException | None:
    depth = 0
    while exc is not None and depth < 8:
        if isinstance(exc, HTTPException):
            return exc
        nested: Any = getattr(exc, "exceptions", None)
        if not isinstance(nested, (list, tuple)) or not nested:
            return None
        exc = nested[0]
        depth += 1
    return None


def _declared_body_length(request: Request) -> int:
    content_length = request.headers.get("content-length")
    if not content_length:
        return -1
    try:
        return int(content_length)
    except ValueError:
        return -1


async def _read_request_body(request: Request, limit: int) -> bytes:
    cached = getattr(request, "_body", None)
    if cached:
        if len(cached) > limit:
            raise HTTPException(413, "request body too large")
        return cached
    if _declared_body_length(request) > limit:
        raise HTTPException(413, "request body too large")
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


async def _read_request_body_detail(request: Request, state: _BodyLimit) -> bytes:
    try:
        return await _read_request_body(request, state.limit)
    except HTTPException:
        state.exceeded = True
        raise HTTPException(413, state.detail) from None


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
    if getattr(request, "method", None) in ("GET", "DELETE", "HEAD", "OPTIONS"):
        return {}
    state = _install_body_limit(request)
    raw_length = _declared_body_length(request)
    if raw_length > state.limit:
        raise HTTPException(413, state.detail)
    if raw_length <= 0 or raw_length > MAX_LOGGED_BODY:
        return {}
    cached = getattr(request, "_body", None)
    if cached:
        return _parse_logged_body(cached)
    try:
        body = await _read_request_body_detail(request, state)
    except HTTPException:
        raise
    except Exception:
        return {}
    if state.exceeded:
        raise HTTPException(413, state.detail)
    return _parse_logged_body(body)


LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "::ffff:127.0.0.1", "localhost"})
LOOPBACK_NETWORKS = (ipaddress.ip_network("127.0.0.0/8"), ipaddress.ip_network("::1/128"))


def _is_loopback_peer(host: str) -> bool:
    if host.strip().lower() in LOOPBACK_HOSTS:
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    return any(address in network for network in LOOPBACK_NETWORKS)


def _forwarded_client_ip(request: Request) -> str | None:
    headers = request.headers
    forwarded = headers.get("x-forwarded-for")
    if forwarded:
        first = forwarded.split(",", 1)[0].strip()
        if first:
            return first
    real_ip = headers.get("x-real-ip")
    if real_ip and real_ip.strip():
        return real_ip.strip()
    return None


def _request_client_ip(request: Request) -> str:
    client = request.client
    peer = client.host if client is not None and client.host else ""
    forwarded = _forwarded_client_ip(request)
    if forwarded is None or not peer:
        return peer or "-"
    if not _is_loopback_peer(peer):
        return peer
    return forwarded


MAX_LOGGED_FIELD = 120
_LOG_UNSAFE_RE = re.compile(r"[\x00-\x1f\x7f]")


def _log_field(value: str) -> str:
    return _LOG_UNSAFE_RE.sub(" ", value)[:MAX_LOGGED_FIELD]


def _request_details(request: Request, payload: dict[str, Any], count_tokens: bool = True) -> str:
    parts = []
    user_agent = request.headers.get("user-agent")
    if user_agent:
        parts.append(f"ua={_log_field(user_agent)}")
    model = payload.get("model")
    if isinstance(model, str) and model:
        parts.append(f"model={_log_field(model)}")
    session_id = payload.get("session_id")
    if isinstance(session_id, str) and session_id:
        parts.append(f"sid={_log_field(session_id)}")
    user = payload.get("user")
    if isinstance(user, str) and user:
        parts.append(f"user={_log_field(user)}")
    stream = payload.get("stream")
    if isinstance(stream, bool):
        parts.append(f"stream={int(stream)}")
    messages = payload.get("messages")
    if isinstance(messages, list):
        parts.append(f"msgs={len(messages)}")
        if count_tokens:
            parts.append(f"tokens={count_messages_tokens(messages)}")
    return " ".join(parts)


def _log_request_failure(request: Request, payload: dict[str, Any], duration: float, status: int | None = None, exc: BaseException | None = None) -> None:
    if not log.isEnabledFor(logging.WARNING):
        return
    details = _request_details(request, payload, count_tokens=log.isEnabledFor(logging.DEBUG))
    details_part = f" {details}" if details else ""
    ip = _request_client_ip(request)
    if status is None and isinstance(exc, HTTPException):
        status = exc.status_code
    if status is not None:
        reason = f"status={status}"
    else:
        reason = f"error={_log_field(str(exc)) if exc else 'unknown'}"
    log.warning(
        "%s %s %s%s failed: %s (%.0fms)",
        request.method,
        _log_field(request.url.path),
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
        _log_field(request.url.path),
        ip,
        details_part,
        duration,
    )


@app.middleware("http")
async def _log_requests(request: Request, call_next):
    started = time.monotonic()
    payload: dict[str, Any] = {}
    state = _install_body_limit(request)
    try:
        if _declared_body_length(request) > state.limit:
            raise HTTPException(413, state.detail)
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
        duration = (time.monotonic() - started) * 1000
        http_exc = _unwrap_http_exception(exc)
        if isinstance(http_exc, HTTPException):
            _log_request_failure(request, payload, duration, exc=http_exc)
            return await _on_http_exception(request, http_exc)
        _log_request_failure(request, payload, duration, exc=exc)
        raise
    duration = (time.monotonic() - started) * 1000
    if state.exceeded:
        _log_request_failure(request, payload, duration, status=413)
        return await _on_http_exception(request, HTTPException(413, state.detail))
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


MAX_VALIDATION_ERRORS = 10
MAX_VALIDATION_FIELD = 64
ERROR_ENVELOPE_EXTRA_KEYS = ("finish_reason",)
CLIENT_REQUEST_ID_HEADER = "x-client-request-id"
MAX_CLIENT_REQUEST_ID = 128


def _validation_summary(errors: Any) -> str:
    if not isinstance(errors, list) or not errors:
        return "request validation failed"
    parts: list[str] = []
    for entry in errors[:MAX_VALIDATION_ERRORS]:
        if not isinstance(entry, dict):
            continue
        loc = ".".join(str(part) for part in entry.get("loc", ()) if isinstance(part, (str, int)))[:MAX_VALIDATION_FIELD]
        message = str(entry.get("msg", ""))[:MAX_VALIDATION_FIELD]
        parts.append(f"{loc}: {message}" if loc else message)
    if not parts:
        return "request validation failed"
    if len(errors) > MAX_VALIDATION_ERRORS:
        parts.append(f"and {len(errors) - MAX_VALIDATION_ERRORS} more")
    return "; ".join(parts)


def _request_id_header(request: Request) -> str:
    return uuid.uuid4().hex


def _client_request_id(request: Request) -> str | None:
    provided = request.headers.get("x-request-id") or request.headers.get(CLIENT_REQUEST_ID_HEADER)
    if provided:
        return _LOG_UNSAFE_RE.sub(" ", provided)[:MAX_CLIENT_REQUEST_ID]
    return None


def _response_headers(request: Request, request_id: str) -> dict[str, str]:
    headers = {"x-request-id": request_id}
    client_request_id = _client_request_id(request)
    if client_request_id:
        headers[CLIENT_REQUEST_ID_HEADER] = client_request_id
    return headers


@app.exception_handler(RequestValidationError)
async def _on_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    request_id = _request_id_header(request)
    return JSONResponse(
        status_code=400,
        content=_openai_error_payload(400, f"invalid request body: {_validation_summary(exc.errors())}", request_id),
        headers=_response_headers(request, request_id),
    )


@app.exception_handler(HTTPException)
async def _on_http_exception(request: Request, exc: HTTPException) -> JSONResponse:
    request_id = _request_id_header(request)
    headers = _response_headers(request, request_id)
    if exc.headers:
        headers.update({str(k): str(v) for k, v in exc.headers.items()})
    detail = exc.detail
    if isinstance(detail, dict):
        inner = detail.get("error")
        if not isinstance(inner, dict):
            inner = detail
        message = inner.get("message")
        content = _openai_error_payload(exc.status_code, message if isinstance(message, str) else str(detail), request_id)
        for key in ERROR_ENVELOPE_EXTRA_KEYS:
            value = inner.get(key)
            if value is not None:
                content["error"][key] = value
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
        headers=_response_headers(request, request_id),
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
_POOL_RATE_RESET_SEC = 5
_POOL_RATE_CACHE_MAX = 16


def _prune_pool_rate_cache(now: float) -> None:
    for cached_key, entry in list(_POOL_RATE_CACHE.items()):
        if entry[2]() is None or now - entry[0] >= _POOL_RATE_TTL:
            del _POOL_RATE_CACHE[cached_key]
    while len(_POOL_RATE_CACHE) >= _POOL_RATE_CACHE_MAX:
        oldest = min(_POOL_RATE_CACHE, key=lambda cached_key: _POOL_RATE_CACHE[cached_key][0])
        del _POOL_RATE_CACHE[oldest]


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
        "x-ratelimit-reset-requests": str(int(time.time()) + _POOL_RATE_RESET_SEC),
    }
    try:
        ref = weakref.ref(pool)
    except TypeError:
        _POOL_RATE_CACHE.pop(key, None)
        return headers
    _prune_pool_rate_cache(now)
    _POOL_RATE_CACHE[key] = (now, headers, ref)
    return headers


RATE_LIMITED_PATHS = ("/v1/chat/completions", "/v1/completions", "/v1/messages", "/v1/responses")
QWEN_ONLY_PATH_PREFIXES = ("/v1/images/", "/v1/videos/")

DASHBOARD_CSP = (
    "default-src 'none'; "
    "script-src 'sha256-Jch6M7OSilyKf0vfgH4jDQrbxyJr5L+SIt3Cc57hgvg='; "
    "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
    "font-src 'self' https://fonts.gstatic.com data:; "
    "img-src 'self' data:; "
    "connect-src 'self'; "
    "base-uri 'none'; form-action 'none'; object-src 'none'; frame-ancestors 'none'"
)


def _apply_security_headers(request: Request, response: Response) -> None:
    headers = response.headers
    if "x-content-type-options" not in headers:
        headers["x-content-type-options"] = "nosniff"
    if "x-frame-options" not in headers:
        headers["x-frame-options"] = "DENY"
    if "content-security-policy" in headers:
        return
    if request.url.path != "/" or not headers.get("content-type", "").startswith("text/html"):
        return
    headers["content-security-policy"] = DASHBOARD_CSP


async def _rate_limit_pool(request: Request) -> Any:
    path = request.url.path
    if path.startswith(QWEN_ONLY_PATH_PREFIXES):
        return provider_pool("qwen")
    if path not in RATE_LIMITED_PATHS:
        return None
    try:
        payload = await _extract_request_body(request)
    except HTTPException:
        return None
    model = payload.get("model")
    if not isinstance(model, str) or not model:
        return None
    try:
        return provider_pool(_resolve_provider(model))
    except HTTPException:
        return None


@app.middleware("http")
async def _openai_headers(request: Request, call_next):
    pool = await _rate_limit_pool(request)
    response = await call_next(request)
    headers = response.headers
    if not headers.get("x-request-id"):
        headers["x-request-id"] = _request_id_header(request)
    if not headers.get("x-ratelimit-limit-requests"):
        for key, value in _pool_rate_headers(pool).items():
            headers[key] = value
    _apply_security_headers(request, response)
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
DOCS_ASSETS = frozenset({"index.html", "style.css", "script.js", "deepseek-logo.svg", "qwen-logo.svg"})


class _DocsAssets(StaticFiles):
    async def get_response(self, path: str, scope: Scope) -> Response:
        requested = posixpath.normpath(path)
        if requested not in (".", "index.html") and requested not in DOCS_ASSETS:
            raise HTTPException(404)
        return await super().get_response(path, scope)


if docs_path.is_dir():
    app.mount("/docs", _DocsAssets(directory=str(docs_path), html=True), name="docs")

app.router.lifespan_context = lifespan
