from __future__ import annotations

import asyncio
import logging
import threading
import time

from .deepseek.client import DeepSeekClient
from .pow import PowManager
from .sessions import SessionRegistry

log = logging.getLogger("danyapi.accounts")


class AccountPoolBusy(Exception):
    pass


class ContextIndex:
    def __init__(self, maxsize: int = 128) -> None:
        self._seqs: dict[str, tuple[str, ...]] = {}
        self._recency: dict[str, int] = {}
        self._lock = threading.Lock()
        self._maxsize = max(1, maxsize)
        self._tick = 0

    def lookup(self, sequence: tuple[str, ...]) -> str | None:
        if not sequence:
            return None
        with self._lock:
            if not self._seqs:
                return None
            best_sid: str | None = None
            best_key = (-1, -1)
            for sid, seq in self._seqs.items():
                if not seq or len(seq) > len(sequence):
                    continue
                if sequence[: len(seq)] != seq:
                    continue
                key = (len(seq), self._recency.get(sid, 0))
                if key > best_key:
                    best_key = key
                    best_sid = sid
            if best_sid is not None:
                self._touch(best_sid)
            return best_sid

    def index(self, session_id: str, sequence: tuple[str, ...]) -> None:
        if not session_id or not sequence:
            return
        with self._lock:
            current = self._seqs.get(session_id)
            if current is not None and len(current) >= len(sequence) and sequence == current[: len(sequence)]:
                self._touch(session_id)
                return
            self._seqs[session_id] = sequence
            self._touch(session_id)
            while len(self._seqs) > self._maxsize:
                oldest = min(self._recency, key=lambda sid: self._recency[sid])
                self._seqs.pop(oldest, None)
                self._recency.pop(oldest, None)

    def forget(self, session_id: str) -> None:
        with self._lock:
            self._seqs.pop(session_id, None)
            self._recency.pop(session_id, None)

    def _touch(self, session_id: str) -> None:
        self._recency[session_id] = self._tick
        self._tick += 1


class DeepSeekAccount:
    __slots__ = ("broken", "client", "index", "pow", "pow_upload", "sem", "sessions")

    def __init__(self, index: int, client: DeepSeekClient, session_cache_size: int = 128) -> None:
        self.index = index
        self.client = client
        self.pow = PowManager()
        self.pow_upload = PowManager()
        self.sem = asyncio.Semaphore(1)
        self.sessions = SessionRegistry(client, session_cache_size)
        self.broken = False

    def mark_broken(self) -> None:
        if not self.broken:
            self.broken = True
            log.warning("account #%d marked broken (invalid/expired token)", self.index)

    @property
    def label(self) -> str:
        return f"acct#{self.index}"


class AccountPool:
    def __init__(self, accounts: list, label: str = "deepseek", session_cache_size: int = 128) -> None:
        self.accounts = accounts
        self.label = label
        self._by_session: dict[str, int] = {}
        self._rr = 0
        self._contexts = ContextIndex(session_cache_size)

    @property
    def healthy(self) -> list[DeepSeekAccount]:
        return [a for a in self.accounts if not a.broken]

    def register(self, account_index: int, session_id: str) -> None:
        self._by_session[session_id] = account_index

    def forget(self, session_id: str) -> None:
        self._by_session.pop(session_id, None)

    def resolve_context(self, sequence: tuple[str, ...]) -> str | None:
        return self._contexts.lookup(sequence)

    def index_context(self, session_id: str, sequence: tuple[str, ...]) -> None:
        self._contexts.index(session_id, sequence)

    def forget_context(self, session_id: str) -> None:
        self._contexts.forget(session_id)

    def account_for_session(self, session_id: str) -> DeepSeekAccount | None:
        idx = self._by_session.get(session_id)
        if idx is None:
            return None
        acct = self.accounts[idx]
        if acct is None or acct.broken:
            self._by_session.pop(session_id, None)
            self._contexts.forget(session_id)
            return None
        return acct

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
                    self._rr = (acct.index + 1) % len(self.healthy)
                    return acct, None
            if time.monotonic() >= deadline:
                raise AccountPoolBusy()
            await asyncio.sleep(0.05)
