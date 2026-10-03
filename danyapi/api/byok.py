from __future__ import annotations

import asyncio
import hashlib
import hmac
import itertools
import json
import logging
import os
import time
from collections.abc import Callable, Sequence
from contextvars import ContextVar
from typing import Any

from fastapi import HTTPException, Request

from ..accounts import AccountPool, DeepSeekAccount
from ..alice.accounts import AliceAccount
from ..alice.client import AliceClient
from ..config import settings
from ..deepseek.client import DeepSeekClient
from ..duckai import api as duckai_api
from ..duckai.accounts import DuckAIAccount
from ..duckai.client import DuckAIClient
from ..gigachat.accounts import GigaChatAccount
from ..gigachat.client import GigaChatClient
from ..opencode.accounts import OpenCodeAccount
from ..opencode.client import EMPTY_CREDENTIAL, OpenCodeClient
from ..qwen.accounts import QwenAccount
from ..qwen.client import QwenClient
from ..store import JsonStore, cache_root
from .core import MAX_REQUEST_BODY, _read_request_body
from .models import _header_api_key, refresh_provider_models
from .state import (
    BYOK_PROVIDERS,
    KEY_OPTIONAL_PROVIDERS,
    _byok_auth_state,
    _byok_pools_state,
    _byok_stores_state,
    app,
    provider_needs_api_key,
)

log = logging.getLogger("danyapi.api")


BYOK_POOL_LIMIT = 512
BYOK_TOTAL_POOL_LIMIT = 512
BYOK_AUTH_LIMIT = 4096
BYOK_MAX_KEYS = 16
BYOK_MAX_JSON_BODY = 1024 * 1024
BYOK_FORM_MAX_BYTES = 2 * 1024 * 1024
BYOK_FORM_MAX_FILES = 8
BYOK_FORM_MAX_FIELDS = 32
BYOK_KEY_LOCK_LIMIT = 4096
BYOK_SALT_FILE = "byok-affinity-salt.bin"
KEYLESS_POOL_KEY = "__keyless__"

_INVALID_KEY_DETAIL = "api key for {provider} is missing or invalid, check the key and the key format"
_UNREACHABLE_DETAIL = "{provider} could not be reached, the api key could not be verified, try again"
_POOL_LIMIT_DETAIL = "too many cached api key sets are live at once, retry in a moment"

_CACHE_MISS = object()
_KEY_LOCKS: dict[str, asyncio.Lock] = {}
_POOL_TOUCH_SEQ = itertools.count()
_POOL_TOUCH_ATTR = "_danyapi_touched"
_EVICT_LOCK = asyncio.Lock()
_AUTH_INDETERMINATE: ContextVar[int] = ContextVar("danyapi_byok_auth_indeterminate", default=0)
_CALLER_ID: ContextVar[str] = ContextVar("danyapi_byok_caller_id", default="")


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


def _touch_auth(store: dict[str, Any], stable: str) -> None:
    record = store.get(stable, _CACHE_MISS)
    if record is _CACHE_MISS:
        return
    store.pop(stable, None)
    store[stable] = record


def _touch_pool_cache(cache: dict[str, Any], cache_key: str) -> None:
    pool = cache.get(cache_key, _CACHE_MISS)
    if pool is _CACHE_MISS:
        return
    cache.pop(cache_key, None)
    cache[cache_key] = pool
    _mark_pool_touched(pool)


def _mark_pool_touched(pool: Any) -> None:
    try:
        setattr(pool, _POOL_TOUCH_ATTR, next(_POOL_TOUCH_SEQ))
    except (AttributeError, TypeError):
        pass


def _pool_touched(pool: Any) -> int:
    value = getattr(pool, _POOL_TOUCH_ATTR, None)
    return value if isinstance(value, int) else -1


def _pool_is_busy(pool: Any) -> bool:
    for acct in getattr(pool, "accounts", None) or []:
        sem = getattr(acct, "sem", None)
        if sem is not None and sem.locked():
            return True
    return False


def _live_pool_entries() -> list[tuple[int, str, str, Any]]:
    entries: list[tuple[int, str, str, Any]] = []
    for provider, cache in _byok_pools_state().items():
        if not isinstance(cache, dict):
            continue
        for cache_key, pool in list(cache.items()):
            if pool is None:
                continue
            entries.append((_pool_touched(pool), provider, cache_key, pool))
    entries.sort(key=lambda entry: entry[0])
    return entries


def _evictable_pool_entries(limit: int, protect: tuple[str, str], reserve: int = 1) -> list[tuple[str, str, Any]]:
    if limit <= 0:
        return []
    entries = _live_pool_entries()
    excess = len(entries) - limit + reserve
    if excess <= 0:
        return []
    victims: list[tuple[str, str, Any]] = []
    for _touched, provider, cache_key, pool in entries:
        if excess <= 0:
            break
        if (provider, cache_key) == protect:
            continue
        if _pool_is_busy(pool):
            continue
        victims.append((provider, cache_key, pool))
        excess -= 1
    return victims


def _evict_pools(victims: Sequence[tuple[str, str, Any]]) -> None:
    stores_state = _byok_stores_state()
    for provider, cache_key, pool in victims:
        cache = _byok_pools_state().get(provider)
        if isinstance(cache, dict):
            if cache.get(cache_key) is not pool:
                continue
            cache.pop(cache_key, None)
        scoped = stores_state.get(provider)
        created = scoped.pop(cache_key, None) if isinstance(scoped, dict) else None
        _close_pool_later(pool, created)


async def _make_room_for_pool(protect: tuple[str, str]) -> bool:
    async with _EVICT_LOCK:
        victims = _evictable_pool_entries(BYOK_TOTAL_POOL_LIMIT, protect)
        if not victims:
            return _live_pool_count() < BYOK_TOTAL_POOL_LIMIT
        _evict_pools(victims)
    return True


def _live_pool_count() -> int:
    return len(_live_pool_entries())


def _auth_indeterminate_count() -> int:
    return _AUTH_INDETERMINATE.get()


def _note_auth_indeterminate() -> None:
    _AUTH_INDETERMINATE.set(_AUTH_INDETERMINATE.get() + 1)


def _load_byok_salt() -> bytes:
    path = cache_root() / BYOK_SALT_FILE
    try:
        existing = path.read_bytes()
        if len(existing) >= 32:
            return existing[:32]
    except OSError as exc:
        log.warning("byok affinity salt is not readable, a new one is generated: %s", exc)
    salt = os.urandom(32)
    try:
        path.write_bytes(salt)
        os.chmod(path, 0o600)
    except OSError as exc:
        log.warning("byok affinity salt cannot be persisted, session affinity is lost on restart: %s", exc)
    return salt


_BYOK_SALT = _load_byok_salt()


def _byok_stable_id(token: str) -> str:
    return hmac.new(_BYOK_SALT, token.encode("utf-8"), hashlib.sha256).hexdigest()[:16]


def _byok_scope(cache_key: str) -> str | None:
    if not settings.cache_enabled:
        return None
    return "byok-" + hashlib.sha256(cache_key.encode("utf-8")).hexdigest()[:16]


def _key_lock(provider: str, cache_key: str) -> asyncio.Lock:
    name = f"{provider}:{cache_key}"
    lock = _KEY_LOCKS.get(name)
    if lock is None:
        if len(_KEY_LOCKS) >= BYOK_KEY_LOCK_LIMIT:
            for stale, candidate in list(_KEY_LOCKS.items()):
                if not candidate.locked():
                    _KEY_LOCKS.pop(stale, None)
        lock = asyncio.Lock()
        _KEY_LOCKS[name] = lock
    return lock


async def _api_key_from_form(request: Request) -> str | None:
    cached = getattr(request, "_form", None)
    if cached is not None:
        value = cached.get("api_key")
        return value.strip() if isinstance(value, str) and value.strip() else None
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > BYOK_FORM_MAX_BYTES:
                raise HTTPException(413, "multipart body too large")
        except ValueError:
            raise HTTPException(400, "invalid content-length header") from None
    try:
        form = await request.form(max_files=BYOK_FORM_MAX_FILES, max_fields=BYOK_FORM_MAX_FIELDS)
        try:
            value = form.get("api_key")
        finally:
            await form.close()
    except Exception as exc:
        log.info("byok multipart body could not be parsed: %s", exc)
        raise HTTPException(400, "malformed multipart request body") from exc
    return value.strip() if isinstance(value, str) and value.strip() else None


async def _extract_request_api_key(request: Request) -> str | None:
    key = _header_api_key(request)
    if key:
        return key
    content_type = request.headers.get("content-type") or ""
    if content_type.startswith("multipart/form-data"):
        return await _api_key_from_form(request)
    if not content_type.startswith("application/json"):
        return None
    body = getattr(request, "_body", None)
    if body is None:
        try:
            body = await _read_request_body(request, MAX_REQUEST_BODY)
        except HTTPException:
            raise
        except Exception:
            return None
    if not body:
        return None
    if len(body) > BYOK_MAX_JSON_BODY:
        log.info("byok api key in the json body is ignored for bodies over %d bytes, send it in a header", BYOK_MAX_JSON_BODY)
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


async def _close_client(client: Any) -> None:
    try:
        await client.aclose()
    except Exception as exc:
        log.info("byok client close failed: %s", exc)


def _close_client_later(client: Any) -> None:
    if client is None:
        return
    task = asyncio.create_task(_close_client(client))
    _deferred_close_tasks.add(task)
    task.add_done_callback(_deferred_close_tasks.discard)


async def _close_pool(pool: Any, stores: Sequence[JsonStore] | None = None) -> None:
    busy = _pool_is_busy(pool)

    def _release_stores() -> None:
        if not busy:
            for acct in pool.accounts:
                try:
                    acct.sessions.close_all()
                except Exception as exc:
                    log.info("session cleanup failed for byok account %r: %s", getattr(acct, "label", acct), exc)
        else:
            log.info("skip session cleanup for busy byok pool %r", getattr(pool, "label", pool))
        flush = getattr(pool, "flush", None)
        if flush is not None:
            try:
                flush()
            except Exception as exc:
                log.info("pool store flush failed: %s", exc)
        if busy:
            log.info("keep the cache file of a busy byok pool so in-flight writes are not dropped")
            return
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


def _close_pool_later(pool: Any, stores: Sequence[JsonStore] | None = None) -> None:
    task = asyncio.create_task(_close_pool(pool, stores))
    _deferred_close_tasks.add(task)
    task.add_done_callback(_deferred_close_tasks.discard)


async def _close_busy_client(account: Any, sem: asyncio.Semaphore) -> None:
    try:
        await asyncio.wait_for(sem.acquire(), timeout=300)
    except (TimeoutError, asyncio.TimeoutError):
        log.info("give up deferred client close for busy byok account %r", getattr(account, "label", account))
        return
    except asyncio.CancelledError:
        _close_client_later(account.client)
        raise
    try:
        await account.client.aclose()
    except Exception as exc:
        log.info("client close failed for byok account %r: %s", getattr(account, "label", account), exc)
    finally:
        sem.release()


async def _byok_validate(
    provider: str,
    token: str,
    client: Any,
) -> bool:
    auth = _byok_auth_state()
    store = auth[provider]
    stable = _byok_stable_id(token)
    now = time.monotonic()
    cached = _cached_auth(store, stable, settings.byok_auth_ttl, now)
    if cached is not None:
        _touch_auth(store, stable)
        return cached
    try:
        ok = bool(await client.check_auth())
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        log.warning("byok %s auth check failed before a verdict, the key is not cached: %s: %s", provider, type(exc).__name__, exc)
        _note_auth_indeterminate()
        return False
    if not ok:
        log.warning("byok %s api key was rejected upstream", provider)
    store.pop(stable, None)
    store[stable] = [ok, time.monotonic()]
    _evict_auth(store)
    return ok


def _byok_cache_key(tokens: list[str]) -> str:
    return "|".join(sorted(_byok_stable_id(token) for token in tokens))


async def _build_accounts(
    provider: str,
    tokens: list[str],
    make_client: Callable[[str], Any],
    make_account: Callable[[int, Any, str], Any],
) -> list[Any]:
    accounts: list[Any] = []
    pending: Any = None
    try:
        for token in tokens:
            try:
                pending = make_client(token)
            except (RuntimeError, OSError) as exc:
                log.error("byok %s client unusable, skipping key #%d: %s", provider, len(accounts), exc)
                pending = None
                continue
            accepted = await _byok_validate(provider, token, pending)
            if not accepted:
                log.warning("byok %s token invalid/expired, skipping", provider)
                await _close_client(pending)
                pending = None
                continue
            accounts.append(make_account(len(accounts), pending, token))
            pending = None
    except BaseException:
        _close_client_later(pending)
        for acct in accounts:
            _close_client_later(acct.client)
        raise
    return accounts


def _no_valid_key(provider: str, indeterminate: int) -> HTTPException:
    if indeterminate:
        return HTTPException(503, _UNREACHABLE_DETAIL.format(provider=provider))
    return HTTPException(401, _INVALID_KEY_DETAIL.format(provider=provider))


async def _build_byok_pool(provider: str, tokens: list[str], scope: str | None) -> tuple[AccountPool, list[JsonStore]]:
    created: list[JsonStore] = []
    before = _auth_indeterminate_count()
    if provider == "deepseek":
        session_store = JsonStore("deepseek-sessions", scope) if settings.cache_enabled else None
        context_store = JsonStore("deepseek-contexts", scope) if settings.cache_enabled else None
        affinity_store = JsonStore("deepseek-affinities", scope) if settings.cache_enabled else None
        created = [store for store in (session_store, context_store, affinity_store) if store is not None]
        accounts = await _build_accounts(
            provider,
            tokens,
            lambda token: DeepSeekClient(token=token, timeout=settings.timeout),
            lambda index, client, token: DeepSeekAccount(
                index,
                client,
                session_cache_size=settings.session_cache_size,
                ttl=settings.session_ttl,
                store=session_store,
                stable_id=_byok_stable_id(token),
            ),
        )
        if not accounts:
            raise _no_valid_key(provider, _auth_indeterminate_count() - before)
        pool = AccountPool(
            accounts,
            session_cache_size=settings.session_cache_size,
            ttl=settings.session_ttl,
            context_store=context_store,
            affinity_store=affinity_store,
        )
        await refresh_provider_models("deepseek", accounts[0].client)
        return pool, created
    if provider == "qwen":
        session_store = JsonStore("qwen-sessions", scope) if settings.cache_enabled else None
        context_store = JsonStore("qwen-contexts", scope) if settings.cache_enabled else None
        affinity_store = JsonStore("qwen-affinities", scope) if settings.cache_enabled else None
        created = [store for store in (session_store, context_store, affinity_store) if store is not None]
        accounts = await _build_accounts(
            provider,
            tokens,
            lambda token: QwenClient(token=token, timeout=settings.timeout),
            lambda index, client, token: QwenAccount(
                index,
                client,
                session_cache_size=settings.session_cache_size,
                ttl=settings.session_ttl,
                store=session_store,
                stable_id=_byok_stable_id(token),
            ),
        )
        if not accounts:
            raise _no_valid_key(provider, _auth_indeterminate_count() - before)
        pool = AccountPool(
            accounts,
            label="qwen",
            session_cache_size=settings.session_cache_size,
            ttl=settings.session_ttl,
            context_store=context_store,
            affinity_store=affinity_store,
        )
        await refresh_provider_models("qwen", accounts[0].client)
        return pool, created
    if provider == "gigachat":
        accounts = await _byok_gigachat_accounts(tokens, "byok")
        if not accounts:
            raise _no_valid_key(provider, _auth_indeterminate_count() - before)
        pool = AccountPool(accounts, label="gigachat")
        await refresh_provider_models("gigachat", accounts[0].client)
        return pool, created
    if provider == "opencode":
        accounts = await _byok_opencode_accounts(tokens, "byok")
        if not accounts:
            raise _no_valid_key(provider, _auth_indeterminate_count() - before)
        pool = AccountPool(accounts, label="opencode")
        await refresh_provider_models("opencode", accounts[0].client)
        return pool, created
    raise HTTPException(400, f"provider {provider} does not accept a caller supplied api key")


async def _byok_pool(provider: str, tokens: list[str]) -> AccountPool:
    if provider not in BYOK_PROVIDERS:
        raise HTTPException(400, f"unknown provider: {provider}")
    tokens = list(dict.fromkeys(tokens))
    if len(tokens) > BYOK_MAX_KEYS:
        raise HTTPException(400, f"too many api keys for {provider}: at most {BYOK_MAX_KEYS} keys per request")
    if provider == "alice":
        return await _byok_alice_pool()
    if provider == "duckai":
        return await _byok_duckai_pool()
    pools = _byok_pools_state()
    cache = pools[provider]
    cache_key = _byok_cache_key(tokens)
    pool = cache.get(cache_key)
    if pool is not None and pool.healthy:
        _touch_pool_cache(cache, cache_key)
        return pool
    _evict_stale_cache(cache, _byok_stores_state().get(provider), incoming=1)
    async with _key_lock(provider, cache_key):
        pool = cache.get(cache_key)
        if pool is not None and pool.healthy:
            _touch_pool_cache(cache, cache_key)
            return pool
        if not await _make_room_for_pool((provider, cache_key)):
            raise HTTPException(503, _POOL_LIMIT_DETAIL)
        stores = _byok_stores_state()
        scoped_stores = stores[provider]
        new_pool, created = await _build_byok_pool(provider, tokens, _byok_scope(cache_key))
        stale_pool = cache.get(cache_key)
        stale_stores = scoped_stores.pop(cache_key, None)
        cache.pop(cache_key, None)
        cache[cache_key] = new_pool
        scoped_stores[cache_key] = created
        _mark_pool_touched(new_pool)
        if stale_pool is not None and stale_pool is not new_pool:
            live = {str(store.path) for store in created if store.path is not None}
            orphans = [store for store in (stale_stores or ()) if str(store.path) not in live]
            _close_pool_later(stale_pool, orphans)
        _evict_stale_cache(cache, scoped_stores)
        _evict_pools(_evictable_pool_entries(BYOK_TOTAL_POOL_LIMIT, (provider, cache_key), reserve=0))
        return new_pool


def _evict_stale_cache(cache: dict[str, Any], scoped_stores: dict[str, list[JsonStore]] | None, incoming: int = 0) -> None:
    while cache and len(cache) + incoming > BYOK_POOL_LIMIT:
        oldest_key, oldest_pool = next(iter(cache.items()))
        if oldest_pool is not None and _pool_is_busy(oldest_pool):
            cache.pop(oldest_key, None)
            cache[oldest_key] = oldest_pool
            break
        cache.pop(oldest_key)
        created = scoped_stores.pop(oldest_key, None) if scoped_stores is not None else None
        _close_pool_later(oldest_pool, created)


async def _byok_gigachat_accounts(tokens: list[str], log_prefix: str) -> list[GigaChatAccount]:
    log.debug("%s gigachat key set received with %d key(s)", log_prefix, len(tokens))
    return await _build_accounts(
        "gigachat",
        tokens,
        lambda key: GigaChatClient(key=key, scope=settings.gigachat_scope, timeout=settings.timeout),
        lambda index, client, key: GigaChatAccount(index, client, stable_id=_byok_stable_id(key)),
    )


async def _byok_opencode_accounts(tokens: list[str], log_prefix: str) -> list[OpenCodeAccount]:
    log.debug("%s opencode key set received with %d key(s)", log_prefix, len(tokens))
    return await _build_accounts(
        "opencode",
        tokens,
        lambda key: OpenCodeClient(key=key, timeout=settings.timeout),
        lambda index, client, key: OpenCodeAccount(index, client, stable_id=_byok_stable_id(key)),
    )


async def _byok_alice_accounts() -> list[AliceAccount]:
    client = AliceClient(timeout=settings.timeout)
    try:
        if not await client.check_auth():
            await _close_client(client)
            return []
    except BaseException:
        _close_client_later(client)
        raise
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
        stale = _ALICE_BYOK_POOL[0]
        _ALICE_BYOK_POOL[0] = created
        app.state.byok_alice_pool = created
        await _register_keyless_pool("alice", created)
        await refresh_provider_models("alice", None)
        if stale is not None and stale is not created:
            await _close_pool(stale)
        return created


async def _byok_duckai_accounts() -> list[DuckAIAccount]:
    client = DuckAIClient(timeout=settings.timeout)
    try:
        if not await client.check_auth():
            await _close_client(client)
            return []
    except BaseException:
        _close_client_later(client)
        raise
    return [DuckAIAccount(0, client, stable_id="duckai")]


_DUCKAI_BYOK_LOCK = asyncio.Lock()
_DUCKAI_BYOK_POOL: list[AccountPool | None] = [None]


async def _byok_duckai_pool() -> AccountPool:
    pool = _DUCKAI_BYOK_POOL[0]
    if pool is not None and pool.healthy:
        return pool
    async with _DUCKAI_BYOK_LOCK:
        cached = _DUCKAI_BYOK_POOL[0]
        if cached is not None and cached.healthy:
            return cached
        accounts = await _byok_duckai_accounts()
        if not accounts:
            raise HTTPException(502, duckai_api.BLOCKED_HINT)
        created = AccountPool(accounts, label="duckai")
        stale = _DUCKAI_BYOK_POOL[0]
        _DUCKAI_BYOK_POOL[0] = created
        app.state.byok_duckai_pool = created
        await _register_keyless_pool("duckai", created)
        await refresh_provider_models("duckai", accounts[0].client)
        if stale is not None and stale is not created:
            await _close_pool(stale)
        return created


async def _register_keyless_pool(provider: str, pool: AccountPool) -> None:
    entries = _byok_pools_state().get(provider)
    if not isinstance(entries, dict):
        return
    for key in list(entries):
        if key != KEYLESS_POOL_KEY:
            entries.pop(key, None)
    entries[KEYLESS_POOL_KEY] = pool


def _byok_caller_id() -> str:
    return _CALLER_ID.get()


def _caller_id_for(tokens: list[str]) -> str:
    return _byok_stable_id("|".join(sorted(tokens)))


async def _byok_pool_for(provider: str, request: Request) -> AccountPool:
    if provider not in BYOK_PROVIDERS:
        raise HTTPException(400, f"unknown provider: {provider}")
    if provider == "mistral":
        raise HTTPException(400, "mistral does not accept a caller supplied api key, configure MISTRAL_LOGINS instead")
    if provider in KEY_OPTIONAL_PROVIDERS:
        _CALLER_ID.set("")
        return await _byok_pool(provider, [EMPTY_CREDENTIAL])
    if not provider_needs_api_key(provider):
        _CALLER_ID.set("")
        return await _byok_pool(provider, [])
    token = await _extract_request_api_key(request)
    tokens = [t.strip() for t in (token or "").split(",") if t.strip()]
    if not tokens:
        raise HTTPException(401, _INVALID_KEY_DETAIL.format(provider=provider))
    if len(tokens) > BYOK_MAX_KEYS:
        raise HTTPException(400, f"too many api keys for {provider}: at most {BYOK_MAX_KEYS} keys per request")
    _CALLER_ID.set(_caller_id_for(tokens))
    return await _byok_pool(provider, tokens)
