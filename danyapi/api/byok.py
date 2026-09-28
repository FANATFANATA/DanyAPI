from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Sequence
from typing import Any

from fastapi import HTTPException, Request

from ..accounts import AccountPool, DeepSeekAccount
from ..alice.accounts import AliceAccount
from ..alice.client import AliceClient
from ..config import settings
from ..deepseek.client import DeepSeekClient
from ..gigachat.accounts import GigaChatAccount
from ..gigachat.client import GigaChatClient
from ..qwen.accounts import QwenAccount
from ..qwen.client import QwenClient
from ..store import JsonStore
from .core import MAX_REQUEST_BODY, _read_request_body, _token_stable_id
from .models import _fetch_alice_models, _fetch_gigachat_models, _fetch_qwen_models
from .state import BYOK_PROVIDERS, _byok_auth_state, _byok_locks_state, _byok_pools_state, _byok_stores_state, app

log = logging.getLogger("danyapi.api")


BYOK_POOL_LIMIT = 512
BYOK_AUTH_LIMIT = 4096


def _cached_auth(store: dict[str, Any], stable: str, ttl: float, now: float) -> bool | None:
    if ttl <= 0:
        return None
    record = store.get(stable)
    if not isinstance(record, (list, tuple)) or len(record) != 2:
        return None
    try:
        ts = float(record[1])
    except (TypeError, ValueError):
        return None
    if now - ts > ttl:
        return None
    return bool(record[0])


def _evict_auth(store: dict[str, Any]) -> None:
    while len(store) > BYOK_AUTH_LIMIT:
        store.pop(next(iter(store)), None)


async def _api_key_from_form(request: Request) -> str | None:
    try:
        form = await request.form()
    except Exception:
        return None
    value = form.get("api_key")
    return value.strip() if isinstance(value, str) and value.strip() else None


async def _extract_request_api_key(request: Request) -> str | None:
    auth = request.headers.get("authorization") or ""
    if auth.lower().startswith("bearer "):
        key = auth[7:].strip()
        if key:
            return key
    key = (request.headers.get("x-api-key") or "").strip()
    if key:
        return key
    content_type = request.headers.get("content-type") or ""
    if content_type.startswith("multipart/form-data"):
        return await _api_key_from_form(request)
    if not content_type.startswith("application/json"):
        return None
    try:
        body = await _read_request_body(request, MAX_REQUEST_BODY)
    except HTTPException:
        raise
    except Exception:
        return None
    if not body:
        return None
    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError, TypeError):
        return None
    if isinstance(payload, dict):
        api_key = payload.get("api_key")
        if isinstance(api_key, str) and api_key.strip():
            return api_key.strip()
    return None


_deferred_close_tasks: set[asyncio.Task] = set()


async def _close_pool(pool: Any, stores: Sequence[JsonStore] | None = None) -> None:
    def _release_stores() -> None:
        for acct in pool.accounts:
            try:
                acct.sessions.close_all()
            except Exception as exc:
                log.info("session cleanup failed for byok account %r: %s", getattr(acct, "label", acct), exc)
        flush = getattr(pool, "flush", None)
        if flush is not None:
            try:
                flush()
            except Exception as exc:
                log.info("pool store flush failed: %s", exc)
        for store in stores or ():
            try:
                store.remove()
            except Exception as exc:
                log.info("byok cache file delete failed: %s", exc)

    await asyncio.to_thread(_release_stores)
    for acct in pool.accounts:
        sem = getattr(acct, "sem", None)
        if sem is not None and sem.locked():
            log.info("schedule deferred client close for busy byok account %r", getattr(acct, "label", acct))
            task = asyncio.create_task(_close_busy_client(acct, sem))
            _deferred_close_tasks.add(task)
            task.add_done_callback(_deferred_close_tasks.discard)
            continue
        try:
            await acct.client.aclose()
        except Exception as exc:
            log.info("client close failed for byok account %r: %s", getattr(acct, "label", acct), exc)


async def _close_busy_client(account: Any, sem: asyncio.Semaphore) -> None:
    acquired = False
    try:
        await asyncio.wait_for(sem.acquire(), timeout=300)
        acquired = True
    except (TimeoutError, asyncio.TimeoutError, asyncio.CancelledError):
        log.info("give up deferred client close for busy byok account %r", getattr(account, "label", account))
        return
    try:
        await account.client.aclose()
    except Exception as exc:
        log.info("client close failed for byok account %r: %s", getattr(account, "label", account), exc)
    finally:
        if acquired:
            sem.release()


async def _byok_validate(
    provider: str,
    token: str,
    client: Any,
) -> bool:
    auth = await _byok_auth_state()
    store = auth[provider]
    stable = _token_stable_id(token)
    ttl = settings.byok_auth_ttl
    cached = _cached_auth(store, stable, ttl, time.monotonic())
    if cached is not None:
        return cached
    try:
        ok = bool(await client.check_auth())
    except Exception:
        ok = False
    store.pop(stable, None)
    store[stable] = [ok, time.monotonic()]
    _evict_auth(store)
    return ok


def _byok_cache_key(tokens: list[str]) -> str:
    return "|".join(sorted(_token_stable_id(token) for token in tokens))


async def _byok_pool(provider: str, tokens: list[str]) -> AccountPool:
    if provider not in BYOK_PROVIDERS:
        raise HTTPException(400, f"unknown provider: {provider}")
    tokens = list(dict.fromkeys(tokens))
    if provider == "alice":
        return await _byok_alice_pool()
    pools = await _byok_pools_state()
    cache = pools[provider]
    cache_key = _byok_cache_key(tokens)
    pool = cache.get(cache_key)
    if pool is not None and pool.healthy:
        cache.pop(cache_key)
        cache[cache_key] = pool
        return pool
    locks = await _byok_locks_state()
    async with locks[provider]:
        pool = cache.get(cache_key)
        if pool is not None and pool.healthy:
            cache.pop(cache_key)
            cache[cache_key] = pool
            return pool
        stores = await _byok_stores_state()
        scoped_stores = stores[provider]
        if pool is not None:
            cache.pop(cache_key, None)
            await _close_pool(pool, scoped_stores.pop(cache_key, None))
        scope = ("byok-" + _token_stable_id(cache_key)) if settings.cache_enabled else None
        created: list[JsonStore] = []
        if provider == "deepseek":
            session_store = JsonStore("deepseek-sessions", scope) if settings.cache_enabled else None
            context_store = JsonStore("deepseek-contexts", scope) if settings.cache_enabled else None
            affinity_store = JsonStore("deepseek-affinities", scope) if settings.cache_enabled else None
            created = [store for store in (session_store, context_store, affinity_store) if store is not None]
            accounts: list[DeepSeekAccount] = []
            for i, token in enumerate(tokens):
                ds_client = DeepSeekClient(token=token, timeout=settings.timeout)
                if not await _byok_validate("deepseek", token, ds_client):
                    log.warning("byok deepseek token invalid/expired, skipping")
                    await ds_client.aclose()
                    continue
                accounts.append(
                    DeepSeekAccount(
                        i,
                        ds_client,
                        session_cache_size=settings.session_cache_size,
                        ttl=settings.session_ttl,
                        store=session_store,
                        stable_id=_token_stable_id(token),
                    )
                )
            if not accounts:
                raise HTTPException(401, "invalid deepseek api key")
            pool = AccountPool(
                accounts,
                session_cache_size=settings.session_cache_size,
                ttl=settings.session_ttl,
                context_store=context_store,
                affinity_store=affinity_store,
            )
        elif provider == "qwen":
            session_store = JsonStore("qwen-sessions", scope) if settings.cache_enabled else None
            context_store = JsonStore("qwen-contexts", scope) if settings.cache_enabled else None
            affinity_store = JsonStore("qwen-affinities", scope) if settings.cache_enabled else None
            created = [store for store in (session_store, context_store, affinity_store) if store is not None]
            qwen_accounts: list[QwenAccount] = []
            for i, token in enumerate(tokens):
                qw_client = QwenClient(token=token, timeout=settings.timeout)
                if not await _byok_validate("qwen", token, qw_client):
                    log.warning("byok qwen token invalid/expired, skipping")
                    await qw_client.aclose()
                    continue
                qwen_accounts.append(
                    QwenAccount(
                        i,
                        qw_client,
                        session_cache_size=settings.session_cache_size,
                        ttl=settings.session_ttl,
                        store=session_store,
                        stable_id=_token_stable_id(token),
                    )
                )
            if not qwen_accounts:
                raise HTTPException(401, "invalid qwen api key")
            pool = AccountPool(
                qwen_accounts,
                label="qwen",
                session_cache_size=settings.session_cache_size,
                ttl=settings.session_ttl,
                context_store=context_store,
                affinity_store=affinity_store,
            )
            if not getattr(app.state, "qwen_models", None):
                app.state.qwen_models = await _fetch_qwen_models(qwen_accounts[0].client)
        else:
            gigachat_accounts = await _byok_gigachat_accounts(tokens, "byok")
            if not gigachat_accounts:
                raise HTTPException(401, "invalid gigachat authorization key")
            pool = AccountPool(gigachat_accounts, label="gigachat")
            if not getattr(app.state, "gigachat_models", None):
                app.state.gigachat_models = await _fetch_gigachat_models(gigachat_accounts[0].client)
        cache[cache_key] = pool
        scoped_stores[cache_key] = created
        while len(cache) > BYOK_POOL_LIMIT:
            oldest_key, oldest_pool = next(iter(cache.items()))
            cache.pop(oldest_key)
            await _close_pool(oldest_pool, scoped_stores.pop(oldest_key, None))
        return pool


async def _byok_gigachat_accounts(tokens: list[str], log_prefix: str) -> list[GigaChatAccount]:
    accounts: list[GigaChatAccount] = []
    for i, key in enumerate(tokens):
        try:
            gc_client = GigaChatClient(key=key, scope=settings.gigachat_scope, timeout=settings.timeout)
        except (RuntimeError, OSError) as exc:
            log.error("%s gigachat CA unusable, skipping key #%d: %s", log_prefix, i, exc)
            continue
        if not await _byok_validate("gigachat", key, gc_client):
            log.warning("%s gigachat key invalid/expired, skipping", log_prefix)
            await gc_client.aclose()
            continue
        accounts.append(GigaChatAccount(len(accounts), gc_client, stable_id=_token_stable_id(key)))
    return accounts


async def _byok_alice_accounts() -> list[AliceAccount]:
    client = AliceClient(timeout=settings.timeout)
    if not await client.check_auth():
        await client.aclose()
        return []
    return [AliceAccount(0, client, stable_id="alice")]


_ALICE_BYOK_LOCK = asyncio.Lock()
_ALICE_BYOK_POOL: list[AccountPool | None] = [None]


async def _byok_alice_pool() -> AccountPool:
    pool = _ALICE_BYOK_POOL[0]
    if pool is not None and pool.healthy:
        return pool
    async with _ALICE_BYOK_LOCK:
        cached = _ALICE_BYOK_POOL[0]
        if cached is not None and cached.healthy:
            return cached
        accounts = await _byok_alice_accounts()
        if not accounts:
            raise HTTPException(502, "alice endpoint is unreachable")
        created = AccountPool(accounts, label="alice")
        _ALICE_BYOK_POOL[0] = created
        if not getattr(app.state, "alice_models", None):
            app.state.alice_models = await _fetch_alice_models(accounts[0].client)
        return created


async def _byok_pool_for(provider: str, request: Request) -> AccountPool:
    token = await _extract_request_api_key(request)
    tokens = [t.strip() for t in (token or "").split(",") if t.strip()]
    if not tokens:
        raise HTTPException(401, f"missing api key for {provider} provider")
    return await _byok_pool(provider, tokens)
