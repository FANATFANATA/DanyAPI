from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections import OrderedDict
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from typing import Any, Generic, Protocol, TypeVar

from .deepseek.client import DeepSeekClient
from .pow import PowManager
from .sessions import SessionRegistry
from .store import _MAX_AFFINITY, JsonStore

log = logging.getLogger("danyapi.accounts")


class AccountPoolBusy(Exception):
    pass


async def _free_account(candidates: Sequence[Any], timeout: float) -> list[tuple[Any, bool]]:
    waiters = [asyncio.ensure_future(acct.sem.acquire()) for acct in candidates]
    try:
        await asyncio.wait(waiters, timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
    finally:
        for waiter in waiters:
            if not waiter.done():
                waiter.cancel()
        await asyncio.gather(*waiters, return_exceptions=True)
    return [(acct, waiter.done() and not waiter.cancelled() and waiter.exception() is None) for acct, waiter in zip(candidates, waiters, strict=True)]


@asynccontextmanager
async def account_lock(sem: asyncio.Semaphore, max_wait: float | None = None) -> AsyncIterator[None]:
    if max_wait is None:
        async with sem:
            yield
        return
    try:
        await asyncio.wait_for(sem.acquire(), timeout=max_wait)
    except (TimeoutError, asyncio.TimeoutError) as exc:
        raise AccountPoolBusy() from exc
    try:
        yield
    finally:
        sem.release()


class ContextIndex:
    def __init__(
        self,
        maxsize: int = 128,
        ttl: float = 0.0,
        store: JsonStore | None = None,
    ) -> None:
        self._seqs: dict[str, tuple[str, ...]] = {}
        self._recency: OrderedDict[str, int] = OrderedDict()
        self._ts: dict[str, float] = {}
        self._lock = threading.Lock()
        self._maxsize = max(1, maxsize)
        self._ttl = max(0.0, ttl)
        self._tick = 0
        self.hits = 0
        self.misses = 0
        self._store = store
        self._restore()

    def _restore(self) -> None:
        store = self._store
        if store is None:
            return
        now = time.monotonic()
        junk: list[str] = []
        for session_id, record in store.items():
            if not isinstance(session_id, str) or not session_id:
                junk.append(session_id)
                continue
            if not isinstance(record, list):
                junk.append(session_id)
                continue
            sequence = tuple(item for item in record if isinstance(item, str))
            if not sequence:
                junk.append(session_id)
                continue
            self._seqs[session_id] = sequence
            self._touch(session_id, now)
        for session_id in junk:
            store.discard(session_id)
        while len(self._seqs) > self._maxsize:
            oldest = next(iter(self._recency))
            self._seqs.pop(oldest, None)
            self._recency.pop(oldest, None)
            self._ts.pop(oldest, None)
            store.discard(oldest)

    def _expired(self, session_id: str, now: float) -> bool:
        ts = self._ts.get(session_id)
        return ts is not None and self._ttl > 0 and now - ts > self._ttl

    @property
    def size(self) -> int:
        return len(self._seqs)

    @property
    def max_size(self) -> int:
        return self._maxsize

    def flush(self) -> None:
        if self._store is not None:
            try:
                self._store.flush()
            except Exception as exc:
                log.warning("context store flush failed: %s", exc)

    def lookup(self, sequence: tuple[str, ...]) -> str | None:
        if not sequence:
            return None
        now = time.monotonic()
        store = self._store
        expired: list[str] = []
        with self._lock:
            if not self._seqs:
                self.misses += 1
                return None
            best_sid: str | None = None
            best_key = (-1, -1)
            for sid, seq in self._seqs.items():
                if self._expired(sid, now):
                    expired.append(sid)
                    continue
                if not seq or len(seq) > len(sequence):
                    continue
                if sequence[: len(seq)] != seq:
                    continue
                key = (len(seq), self._recency.get(sid, 0))
                if key > best_key:
                    best_key = key
                    best_sid = sid
            for sid in expired:
                self._seqs.pop(sid, None)
                self._recency.pop(sid, None)
                self._ts.pop(sid, None)
            if best_sid is not None:
                self.hits += 1
                self._touch(best_sid, now)
            else:
                self.misses += 1
        if store is not None:
            for sid in expired:
                store.discard(sid)
        return best_sid

    def index(self, session_id: str, sequence: tuple[str, ...]) -> None:
        if not session_id or not sequence:
            return
        now = time.monotonic()
        store = self._store
        evicted: list[str] = []
        with self._lock:
            current = self._seqs.get(session_id)
            if current is not None and len(current) >= len(sequence) and sequence == current[: len(sequence)]:
                self._touch(session_id, now)
                return
            self._seqs[session_id] = sequence
            self._touch(session_id, now)
            while len(self._seqs) > self._maxsize:
                oldest = next(iter(self._recency))
                self._seqs.pop(oldest, None)
                self._recency.pop(oldest, None)
                self._ts.pop(oldest, None)
                evicted.append(oldest)
        if store is not None:
            for oldest in evicted:
                store.discard(oldest)
            store.set(session_id, list(sequence))

    def forget(self, session_id: str) -> None:
        with self._lock:
            self._seqs.pop(session_id, None)
            self._recency.pop(session_id, None)
            self._ts.pop(session_id, None)
        if self._store is not None:
            self._store.discard(session_id)

    def _touch(self, session_id: str, now: float) -> None:
        self._recency.pop(session_id, None)
        self._recency[session_id] = self._tick
        self._tick += 1
        self._ts[session_id] = now


class DeepSeekAccount:
    __slots__ = ("broken", "broken_at", "client", "index", "pow", "pow_upload", "sem", "sessions", "stable_id")

    def __init__(
        self,
        index: int,
        client: DeepSeekClient,
        session_cache_size: int = 128,
        ttl: float = 0.0,
        store: JsonStore | None = None,
        stable_id: str | None = None,
    ) -> None:
        self.index = index
        self.client = client
        self.pow = PowManager()
        self.pow_upload = PowManager()
        self.sem = asyncio.Semaphore(1)
        self.sessions = SessionRegistry(client, session_cache_size, ttl, store=store, key_prefix=f"{index}:")
        self.stable_id = stable_id
        self.broken = False
        self.broken_at: float | None = None

    def mark_broken(self) -> None:
        if not self.broken:
            self.broken = True
            self.broken_at = time.monotonic()
            log.warning("account #%d marked broken (invalid/expired token)", self.index)

    @property
    def label(self) -> str:
        return f"acct#{self.index}"


class _PoolAccount(Protocol):
    broken: bool
    broken_at: float | None
    sem: asyncio.Semaphore

    @property
    def label(self) -> str: ...


AccountT = TypeVar("AccountT", bound=_PoolAccount)


class AccountPool(Generic[AccountT]):
    _REVIVE_COOLDOWN = 300.0
    _REVIVE_AUTH_TIMEOUT = 10.0

    def __init__(
        self,
        accounts: Sequence[AccountT],
        label: str = "deepseek",
        session_cache_size: int = 128,
        ttl: float = 0.0,
        context_store: JsonStore | None = None,
        affinity_store: JsonStore | None = None,
    ) -> None:
        self.accounts = list(accounts)
        self.label = label
        self._by_session: OrderedDict[str, tuple[int, float]] = OrderedDict()
        self._stable_to_idx: dict[str, int] = {}
        for i, acct in enumerate(accounts):
            sid = getattr(acct, "stable_id", None)
            if isinstance(sid, str) and sid:
                self._stable_to_idx[sid] = i
        self._rr = 0
        self._ttl = max(0.0, ttl)
        self._affinity_store = affinity_store
        self._affinity_lock = threading.Lock()
        self._revive_lock = asyncio.Lock()
        self._contexts = ContextIndex(session_cache_size, ttl, store=context_store)
        self._restore_affinities()

    def _affinity_record(self, account_index: int) -> int | str:
        sid = getattr(self.accounts[account_index], "stable_id", None)
        if isinstance(sid, str) and sid:
            return sid
        return account_index

    def _resolve_affinity(self, record: Any) -> int | None:
        idx: Any = record
        if isinstance(record, list) and record:
            idx = record[0]
        if isinstance(idx, bool):
            return None
        if isinstance(idx, int):
            return idx if 0 <= idx < len(self.accounts) else None
        if isinstance(idx, str):
            return self._stable_to_idx.get(idx)
        return None

    def _restore_affinities(self) -> None:
        if self._affinity_store is None:
            return
        now = time.monotonic()
        for session_id, record in self._affinity_store.items():
            if not isinstance(session_id, str) or not session_id:
                continue
            if len(self._by_session) >= _MAX_AFFINITY:
                break
            idx = self._resolve_affinity(record)
            if idx is None:
                continue
            self._by_session[session_id] = (idx, now)

    @property
    def healthy(self) -> list[AccountT]:
        return [a for a in self.accounts if not a.broken]

    def register(self, account_index: int, session_id: str) -> None:
        if not isinstance(account_index, int) or isinstance(account_index, bool):
            return
        if not 0 <= account_index < len(self.accounts) or not session_id:
            return
        now = time.monotonic()
        record = self._affinity_record(account_index)
        evicted: list[str] = []
        with self._affinity_lock:
            self._by_session.pop(session_id, None)
            self._by_session[session_id] = (account_index, now)
            while len(self._by_session) > _MAX_AFFINITY:
                oldest, _ = self._by_session.popitem(last=False)
                evicted.append(oldest)
            if self._ttl > 0 and len(self._by_session) >= min(4096, _MAX_AFFINITY // 2):
                stale = [sid for sid, (_, ts) in self._by_session.items() if now - ts > self._ttl]
                for sid in stale:
                    self._by_session.pop(sid, None)
                    evicted.append(sid)
        store = self._affinity_store
        if store is not None:
            for sid in evicted:
                store.discard(sid)
            if store.get(session_id) != record:
                store.set(session_id, record)

    def forget(self, session_id: str) -> None:
        with self._affinity_lock:
            self._by_session.pop(session_id, None)
        if self._affinity_store is not None:
            self._affinity_store.discard(session_id)

    def resolve_context(self, sequence: tuple[str, ...]) -> str | None:
        return self._contexts.lookup(sequence)

    def index_context(self, session_id: str, sequence: tuple[str, ...]) -> None:
        self._contexts.index(session_id, sequence)

    def forget_context(self, session_id: str) -> None:
        self._contexts.forget(session_id)

    def account_for_session(self, session_id: str) -> AccountT | None:
        store = self._affinity_store
        dirty = False
        acct: AccountT | None = None
        with self._affinity_lock:
            entry = self._by_session.get(session_id)
            if entry is None:
                return None
            idx, ts = entry
            now = time.monotonic()
            if self._ttl > 0 and now - ts > self._ttl:
                self._by_session.pop(session_id, None)
                dirty = True
            else:
                acct = self.accounts[idx]
                if acct is None or acct.broken:
                    self._by_session.pop(session_id, None)
                    dirty = True
                else:
                    if now != ts:
                        self._by_session[session_id] = (idx, now)
                    self._by_session.move_to_end(session_id)
        if dirty:
            self._contexts.forget(session_id)
            if store is not None:
                store.discard(session_id)
            return None
        return acct

    def stats(self) -> dict[str, Any]:
        with self._affinity_lock:
            affinities = len(self._by_session)
        healthy = [a for a in self.accounts if not a.broken]
        return {
            "label": self.label,
            "accounts": len(self.accounts),
            "healthy": len(healthy),
            "broken": len(self.accounts) - len(healthy),
            "session_affinities": affinities,
            "context_entries": self._contexts.size,
            "context_hits": self._contexts.hits,
            "context_misses": self._contexts.misses,
            "context_cache_size": self._contexts.max_size,
            "ttl_seconds": self._ttl,
        }

    def flush(self) -> None:
        self._contexts.flush()
        if self._affinity_store is not None:
            try:
                self._affinity_store.flush()
            except Exception as exc:
                log.warning("affinity store flush failed: %s", exc)
        for acct in self.accounts:
            sessions = getattr(acct, "sessions", None)
            if sessions is None:
                continue
            try:
                sessions.flush()
            except Exception as exc:
                log.warning("session store flush failed for %s: %s", getattr(acct, "label", acct), exc)

    async def acquire(self, session_id: str | None, max_wait: float | None = None) -> tuple[AccountT, str | None]:
        healthy = [a for a in self.accounts if not a.broken]
        if not healthy:
            revived = await self.revive_broken(max_wait)
            if revived is None:
                if any(getattr(acct, "broken_at", None) is not None for acct in self.accounts):
                    raise AccountPoolBusy()
                raise RuntimeError(f"all {self.label} accounts are unavailable")
            healthy = [a for a in self.accounts if not a.broken]
        if session_id:
            acct = self.account_for_session(session_id)
            if acct is not None:
                if acct.sem.locked() and max_wait is not None:
                    return await self._wait_free(acct, max_wait, session_id)
                return acct, session_id
        n = len(healthy)
        start = self._rr % n
        for i in range(n):
            idx = (start + i) % n
            acct = healthy[idx]
            if not acct.sem.locked():
                self._rr = (idx + 1) % n
                return acct, None
        if max_wait is not None:
            return await self._wait_free(None, max_wait, None)
        acct = healthy[start]
        return acct, None

    def _advance_cursor(self, account: AccountT) -> None:
        healthy = self.healthy
        for index, candidate in enumerate(healthy):
            if candidate is account:
                self._rr = (index + 1) % len(healthy)
                return

    async def _wait_free(
        self,
        preferred: AccountT | None,
        max_wait: float,
        session_id: str | None,
    ) -> tuple[AccountT, str | None]:
        deadline = time.monotonic() + max_wait
        while True:
            if preferred is not None and not preferred.broken:
                candidates = [preferred]
            else:
                candidates = self.healthy
            if not candidates:
                raise AccountPoolBusy()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise AccountPoolBusy()
            ready = await _free_account(candidates, remaining)
            winners = [account for account, held in ready if held]
            if winners:
                chosen = winners[0]
                for extra in winners[1:]:
                    extra.sem.release()
                if session_id is None:
                    self._advance_cursor(chosen)
                return chosen, session_id
            if time.monotonic() >= deadline:
                raise AccountPoolBusy()

    async def revive_broken(self, max_wait: float | None = None) -> AccountT | None:
        if self._revive_lock.locked():
            timeout = None if max_wait is None else max(0.0, max_wait)
            try:
                await asyncio.wait_for(self._revive_lock.acquire(), timeout=timeout)
            except (TimeoutError, asyncio.TimeoutError):
                return self._first_healthy()
            self._revive_lock.release()
            return self._first_healthy()
        async with self._revive_lock:
            return await self._revive_pass(max_wait)

    def _first_healthy(self) -> AccountT | None:
        for account in self.healthy:
            return account
        return None

    async def _revive_pass(self, max_wait: float | None) -> AccountT | None:
        now = time.monotonic()
        deadline = None if max_wait is None else now + max_wait
        for acct in self.accounts:
            if not acct.broken:
                continue
            broken_at = getattr(acct, "broken_at", None)
            if broken_at is None or now - broken_at < self._REVIVE_COOLDOWN:
                continue
            client = getattr(acct, "client", None)
            if client is None:
                continue
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    log.warning("%s revive budget exhausted after %d account(s)", self.label, len(self.accounts))
                    return None
            else:
                remaining = self._REVIVE_AUTH_TIMEOUT
            try:
                ok = await asyncio.wait_for(client.check_auth(), timeout=remaining)
            except (TimeoutError, asyncio.TimeoutError):
                log.warning("%s auth recheck timed out", acct.label)
                ok = False
            except Exception as exc:
                log.warning("%s auth recheck failed: %s", acct.label, exc)
                ok = False
            if ok:
                acct.broken = False
                acct.broken_at = None
                log.info("%s revived after auth recheck", acct.label)
                return acct
            acct.broken_at = time.monotonic()
        return None

    def add_account(self, account: AccountT) -> None:
        with self._affinity_lock:
            idx = len(self.accounts)
            self.accounts.append(account)
            sid = getattr(account, "stable_id", None)
            if isinstance(sid, str) and sid:
                self._stable_to_idx[sid] = idx
