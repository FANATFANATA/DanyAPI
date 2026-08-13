from __future__ import annotations

import asyncio
import logging
import threading
import time

from .deepseek.client import DeepSeekClient
from .pow import PowManager
from .sessions import SessionRegistry
from .store import JsonStore

log = logging.getLogger("danyapi.accounts")


class AccountPoolBusy(Exception):
    pass


class ContextIndex:
    def __init__(
        self,
        maxsize: int = 128,
        ttl: float = 0.0,
        store: JsonStore | None = None,
    ) -> None:
        self._seqs: dict[str, tuple[str, ...]] = {}
        self._recency: dict[str, int] = {}
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
        if self._store is None:
            return
        now = time.monotonic()
        for session_id, record in self._store.items():
            if not isinstance(session_id, str) or not session_id:
                continue
            if not isinstance(record, list):
                continue
            sequence = tuple(item for item in record if isinstance(item, str))
            if not sequence:
                continue
            self._seqs[session_id] = sequence
            self._touch(session_id, now)

    def _expired(self, session_id: str, now: float) -> bool:
        ts = self._ts.get(session_id)
        return ts is not None and self._ttl > 0 and now - ts > self._ttl

    def lookup(self, sequence: tuple[str, ...]) -> str | None:
        if not sequence:
            return None
        now = time.monotonic()
        with self._lock:
            if not self._seqs:
                self.misses += 1
                return None
            best_sid: str | None = None
            best_key = (-1, -1)
            expired: list[str] = []
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
                if self._store is not None:
                    self._store.discard(sid)
            if best_sid is not None:
                self.hits += 1
                self._touch(best_sid, now)
            else:
                self.misses += 1
            return best_sid

    def index(self, session_id: str, sequence: tuple[str, ...]) -> None:
        if not session_id or not sequence:
            return
        now = time.monotonic()
        with self._lock:
            current = self._seqs.get(session_id)
            if current is not None and len(current) >= len(sequence) and sequence == current[: len(sequence)]:
                self._touch(session_id, now)
                return
            self._seqs[session_id] = sequence
            self._touch(session_id, now)
            while len(self._seqs) > self._maxsize:
                oldest = min(self._recency, key=lambda sid: self._recency[sid])
                self._seqs.pop(oldest, None)
                self._recency.pop(oldest, None)
                self._ts.pop(oldest, None)
                if self._store is not None:
                    self._store.discard(oldest)
            if self._store is not None:
                self._store.set(session_id, list(sequence))

    def forget(self, session_id: str) -> None:
        with self._lock:
            self._seqs.pop(session_id, None)
            self._recency.pop(session_id, None)
            self._ts.pop(session_id, None)
        if self._store is not None:
            self._store.discard(session_id)

    def _touch(self, session_id: str, now: float) -> None:
        self._recency[session_id] = self._tick
        self._tick += 1
        self._ts[session_id] = now


class DeepSeekAccount:
    __slots__ = ("broken", "client", "index", "pow", "pow_upload", "sem", "sessions")

    def __init__(
        self,
        index: int,
        client: DeepSeekClient,
        session_cache_size: int = 128,
        ttl: float = 0.0,
        store: JsonStore | None = None,
    ) -> None:
        self.index = index
        self.client = client
        self.pow = PowManager()
        self.pow_upload = PowManager()
        self.sem = asyncio.Semaphore(1)
        self.sessions = SessionRegistry(client, session_cache_size, ttl, store=store, key_prefix=f"{index}:")
        self.broken = False

    def mark_broken(self) -> None:
        if not self.broken:
            self.broken = True
            log.warning("account #%d marked broken (invalid/expired token)", self.index)

    @property
    def label(self) -> str:
        return f"acct#{self.index}"


class AccountPool:
    def __init__(
        self,
        accounts: list,
        label: str = "deepseek",
        session_cache_size: int = 128,
        ttl: float = 0.0,
        context_store: JsonStore | None = None,
        affinity_store: JsonStore | None = None,
    ) -> None:
        self.accounts = accounts
        self.label = label
        self._by_session: dict[str, tuple[int, float]] = {}
        self._rr = 0
        self._ttl = max(0.0, ttl)
        self._affinity_store = affinity_store
        self._contexts = ContextIndex(session_cache_size, ttl, store=context_store)
        self._restore_affinities()

    def _restore_affinities(self) -> None:
        if self._affinity_store is None:
            return
        now = time.monotonic()
        for session_id, record in self._affinity_store.items():
            if not isinstance(session_id, str) or not session_id:
                continue
            idx = record
            if isinstance(record, list) and record:
                idx = record[0]
            if isinstance(idx, bool) or not isinstance(idx, int):
                continue
            if 0 <= idx < len(self.accounts):
                self._by_session[session_id] = (idx, now)

    @property
    def healthy(self) -> list[DeepSeekAccount]:
        return [a for a in self.accounts if not a.broken]

    def register(self, account_index: int, session_id: str) -> None:
        now = time.monotonic()
        self._by_session[session_id] = (account_index, now)
        if self._affinity_store is not None:
            self._affinity_store.set(session_id, account_index)
        if self._ttl > 0 and len(self._by_session) > max(4096, len(self.accounts) * 1024):
            stale = [sid for sid, (_, ts) in self._by_session.items() if now - ts > self._ttl]
            for sid in stale:
                self._by_session.pop(sid, None)
                if self._affinity_store is not None:
                    self._affinity_store.discard(sid)

    def forget(self, session_id: str) -> None:
        self._by_session.pop(session_id, None)
        if self._affinity_store is not None:
            self._affinity_store.discard(session_id)

    def resolve_context(self, sequence: tuple[str, ...]) -> str | None:
        return self._contexts.lookup(sequence)

    def index_context(self, session_id: str, sequence: tuple[str, ...]) -> None:
        self._contexts.index(session_id, sequence)

    def forget_context(self, session_id: str) -> None:
        self._contexts.forget(session_id)

    def account_for_session(self, session_id: str) -> DeepSeekAccount | None:
        entry = self._by_session.get(session_id)
        if entry is None:
            return None
        idx, ts = entry
        if self._ttl > 0 and time.monotonic() - ts > self._ttl:
            self._by_session.pop(session_id, None)
            self._contexts.forget(session_id)
            if self._affinity_store is not None:
                self._affinity_store.discard(session_id)
            return None
        acct = self.accounts[idx]
        if acct is None or acct.broken:
            self._by_session.pop(session_id, None)
            self._contexts.forget(session_id)
            if self._affinity_store is not None:
                self._affinity_store.discard(session_id)
            return None
        return acct

    def stats(self) -> dict:
        return {
            "label": self.label,
            "accounts": len(self.accounts),
            "healthy": len(self.healthy),
            "broken": len(self.accounts) - len(self.healthy),
            "session_affinities": len(self._by_session),
            "context_entries": len(self._contexts._seqs),
            "context_hits": self._contexts.hits,
            "context_misses": self._contexts.misses,
            "context_cache_size": self._contexts._maxsize,
            "ttl_seconds": self._ttl,
        }

    async def acquire(self, session_id: str | None, max_wait: float | None = None) -> tuple[DeepSeekAccount, str | None]:
        healthy = self.healthy
        if not healthy:
            raise RuntimeError(f"all {self.label} accounts are unavailable")
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

    async def _wait_free(
        self,
        preferred: DeepSeekAccount | None,
        max_wait: float,
        session_id: str | None,
    ) -> tuple[DeepSeekAccount, str | None]:
        deadline = time.monotonic() + max_wait
        while True:
            if preferred is not None:
                candidates = [preferred]
            else:
                candidates = self.healthy
            for acct in candidates:
                if not acct.sem.locked():
                    if session_id is not None:
                        return acct, session_id
                    self._rr = (candidates.index(acct) + 1) % len(candidates)
                    return acct, None
            if time.monotonic() >= deadline:
                raise AccountPoolBusy()
            await asyncio.sleep(0.05)
