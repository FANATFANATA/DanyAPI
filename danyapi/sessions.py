from __future__ import annotations

import asyncio
import logging
import math
import threading
import time
from collections import OrderedDict
from typing import Any

from .store import JsonStore

log = logging.getLogger("danyapi.sessions")


def _as_int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return default
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            pass
        try:
            number = float(value)
        except (TypeError, ValueError):
            return default
        if not math.isfinite(number):
            return default
        return int(number)
    return default


class SessionRegistry:
    def __init__(
        self,
        client: Any,
        maxsize: int = 128,
        ttl: float = 0.0,
        store: JsonStore | None = None,
        key_prefix: str = "",
    ) -> None:
        self._client = client
        self._sessions: OrderedDict[str, tuple[Any, float]] = OrderedDict()
        self._lock = threading.Lock()
        self._maxsize = max(1, maxsize)
        self._ttl = max(0.0, ttl)
        self._store = store
        self._key_prefix = key_prefix
        self._session_locks: dict[str, asyncio.Lock] = {}
        self._session_refs: dict[str, int] = {}
        self._session_locks_guard = asyncio.Lock()
        self._session_locks_guard_sync = threading.RLock()
        self._restore()

    def _now(self) -> float:
        return time.monotonic()

    def _expired(self, session_id: str, now: float) -> bool:
        if self._ttl <= 0:
            return False
        entry = self._sessions.get(session_id)
        return entry is not None and now - entry[1] > self._ttl

    def _session_key(self, session_id: str) -> str:
        return f"{self._key_prefix}{session_id}"

    def _drop_session_lock(self, session_key: str) -> None:
        with self._session_locks_guard_sync:
            if self._session_refs.get(session_key, 0) > 0:
                return
            self._session_locks.pop(session_key, None)
            self._session_refs.pop(session_key, None)

    def _serialize(self, session: Any) -> dict[str, Any]:
        return {
            "id": session.id,
            "title": getattr(session, "title", ""),
            "last_message_id": getattr(session, "last_message_id", None),
            "accumulated_tokens": getattr(session, "accumulated_tokens", 0),
        }

    def _deserialize(self, record: Any) -> Any:
        if not isinstance(record, dict) or not record.get("id"):
            raise ValueError("invalid session record")
        from .deepseek.client import DeepSeekSession

        accumulated = record.get("accumulated_tokens")
        accumulated_tokens = _as_int(accumulated)
        return DeepSeekSession(
            id=record["id"],
            title=record.get("title") or "",
            last_message_id=record.get("last_message_id"),
            accumulated_tokens=accumulated_tokens,
        )

    def _restore(self) -> None:
        if self._store is None:
            return
        prefix = self._key_prefix
        by_canonical: dict[str, Any] = {}
        now = self._now()
        for key, record in self._store.items():
            if prefix:
                if not key.startswith(prefix):
                    continue
                session_id = key[len(prefix) :]
            else:
                session_id = key
            if not session_id:
                self._store.discard(key)
                continue
            try:
                session = self._deserialize(record)
            except Exception as exc:
                log.warning("discarding unparsable session record %s: %s", key, exc)
                self._store.discard(key)
                continue
            canonical = by_canonical.get(session.id)
            if canonical is not None:
                session = canonical
            else:
                by_canonical[session.id] = session
            self._sessions[session_id] = (session, now)
        while len(self._sessions) > self._maxsize:
            oldest, _ = self._sessions.popitem(last=False)
            self._store.discard(self._session_key(oldest))
            self._drop_session_lock(oldest)

    async def _create(self, **kwargs: Any) -> Any:
        return await self._client.create_session(**kwargs)

    def _reuse(self, session: Any, session_id: str, **kwargs: Any) -> bool:
        return True

    def can_reuse(self, session_id: str | None, **kwargs: Any) -> bool:
        if not session_id:
            return False
        session = self.get(session_id)
        return session is not None and self._reuse(session, session_id, **kwargs)

    def _update_last(self, session: Any, message_id: str) -> None:
        session.last_message_id = message_id

    def get(self, session_id: str | None) -> Any | None:
        if not session_id:
            return None
        now = self._now()
        store = self._store
        expired = False
        with self._lock:
            entry = self._sessions.get(session_id)
            if entry is None:
                return None
            if self._expired(session_id, now):
                self._sessions.pop(session_id, None)
                expired = True
            else:
                session = entry[0]
                self._sessions.move_to_end(session_id)
                self._sessions[session_id] = (session, now)
        if expired:
            self._drop_session_lock(session_id)
            if store is not None:
                store.discard(self._session_key(session_id))
            return None
        return session

    async def _session_lock(self, session_key: str) -> asyncio.Lock:
        with self._session_locks_guard_sync:
            lock = self._session_locks.get(session_key)
            if lock is None:
                lock = asyncio.Lock()
                self._session_locks[session_key] = lock
            self._session_refs[session_key] = self._session_refs.get(session_key, 0) + 1
        return lock

    async def _release_session_lock(self, session_key: str) -> None:
        with self._session_locks_guard_sync:
            refs = self._session_refs.get(session_key)
            if refs is None:
                return
            if refs <= 1:
                self._session_refs.pop(session_key, None)
                self._session_locks.pop(session_key, None)
            else:
                self._session_refs[session_key] = refs - 1

    async def obtain(self, session_id: str | None, **kwargs: Any) -> tuple[Any, str]:
        if session_id:
            lock = await self._session_lock(session_id)
            try:
                async with lock:
                    return await self._obtain(session_id, **kwargs)
            finally:
                await self._release_session_lock(session_id)
        return await self._obtain(session_id, **kwargs)

    async def _obtain(self, session_id: str | None, **kwargs: Any) -> tuple[Any, str]:
        if session_id:
            existing = self.get(session_id)
            if existing is not None and self._reuse(existing, session_id, **kwargs):
                return existing, session_id
        session = await self._create(**kwargs)
        new_id = session.id
        bind_key = session_id or new_id
        now = self._now()
        evicted: list[str] = []
        record: dict[str, Any] | None = None
        with self._lock:
            self._sessions[new_id] = (session, now)
            if bind_key != new_id:
                self._sessions[bind_key] = (session, now)
            self._sessions.move_to_end(bind_key)
            if bind_key != new_id:
                self._sessions.move_to_end(new_id)
            allowed = self._maxsize + (1 if bind_key != new_id else 0)
            while self._sessions and len(self._sessions) > allowed:
                oldest, _entry = self._sessions.popitem(last=False)
                evicted.append(oldest)
                self._drop_session_lock(oldest)
            if self._store is not None:
                record = self._serialize(session)
        store = self._store
        if store is not None and record is not None:
            store.set(self._session_key(new_id), record)
            if bind_key != new_id:
                store.set(self._session_key(bind_key), record)
            for oldest in evicted:
                store.discard(self._session_key(oldest))
        return session, bind_key

    def touch_last_message(self, session_id: str, message_id: str | None) -> None:
        store = self._store
        with self._lock:
            entry = self._sessions.get(session_id)
            if entry is None:
                return
            session = entry[0]
            if not message_id:
                return
            self._update_last(session, message_id)
            record = self._serialize(session) if store is not None else None
        if store is not None and record is not None:
            store.set(self._session_key(session_id), record)
            if session_id != session.id:
                store.set(self._session_key(session.id), record)

    def forget(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)
        if self._store is not None:
            self._store.discard(self._session_key(session_id))
        self._drop_session_lock(session_id)

    def close_all(self) -> None:
        with self._lock:
            known = list(self._sessions)
            self._sessions.clear()
        with self._session_locks_guard_sync:
            for key in list(self._session_locks):
                lock = self._session_locks.get(key)
                in_use = self._session_refs.get(key, 0) > 0 or (lock is not None and lock.locked())
                if not in_use:
                    self._session_locks.pop(key, None)
                    self._session_refs.pop(key, None)
        store = self._store
        if store is None:
            return
        if self._key_prefix:
            for key, _ in store.items():
                if key.startswith(self._key_prefix):
                    store.discard(key)
            return
        for session_id in known:
            store.discard(self._session_key(session_id))

    def flush(self) -> None:
        if self._store is not None:
            self._store.flush()
