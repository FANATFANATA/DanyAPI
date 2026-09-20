from __future__ import annotations

import asyncio
import threading
import time
from collections import OrderedDict
from typing import Any

from .store import JsonStore


def _as_int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            try:
                return int(float(value))
            except ValueError:
                return default
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
        self._session_locks_guard = asyncio.Lock()
        self._restore()

    def _now(self) -> float:
        return time.monotonic()

    def _expired(self, session_id: str, now: float) -> bool:
        return self._ttl > 0 and now - self._sessions[session_id][1] > self._ttl

    def _session_key(self, session_id: str) -> str:
        return f"{self._key_prefix}{session_id}"

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
        for key, record in self._store.items():
            if prefix:
                if not key.startswith(prefix):
                    continue
                session_id = key[len(prefix) :]
            else:
                session_id = key
            if not session_id:
                if prefix and key == prefix:
                    self._store.discard(key)
                continue
            try:
                session = self._deserialize(record)
            except Exception:
                continue
            canonical = by_canonical.get(session.id)
            if canonical is not None:
                session = canonical
            else:
                by_canonical[session.id] = session
            self._sessions[session_id] = (session, self._now())
        while len(self._sessions) > self._maxsize:
            oldest, _ = self._sessions.popitem(last=False)
            self._store.discard(self._session_key(oldest))
            self._session_locks.pop(oldest, None)

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
        with self._lock:
            entry = self._sessions.get(session_id)
            if entry is None:
                return None
            if self._expired(session_id, now):
                self._sessions.pop(session_id, None)
                if self._store is not None:
                    self._store.discard(self._session_key(session_id))
                self._session_locks.pop(session_id, None)
                return None
            session = entry[0]
            self._sessions.move_to_end(session_id)
            self._sessions[session_id] = (session, now)
            return session

    async def _session_lock(self, session_key: str) -> asyncio.Lock:
        async with self._session_locks_guard:
            lock = self._session_locks.get(session_key)
            if lock is None:
                lock = asyncio.Lock()
                self._session_locks[session_key] = lock
            return lock

    async def obtain(self, session_id: str | None, **kwargs: Any) -> tuple[Any, str]:
        if session_id:
            lock = await self._session_lock(session_id)
            async with lock:
                return await self._obtain(session_id, **kwargs)
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
        with self._lock:
            self._sessions[new_id] = (session, now)
            if bind_key != new_id:
                self._sessions[bind_key] = (session, now)
            self._sessions.move_to_end(bind_key)
            if bind_key != new_id:
                self._sessions.move_to_end(new_id)
            while len(self._sessions) > self._maxsize:
                protect = {new_id, bind_key}
                evictable = [k for k in self._sessions if k not in protect]
                if not evictable:
                    break
                oldest = min(evictable, key=lambda k: self._sessions[k][1])
                self._sessions.pop(oldest, None)
                if self._store is not None:
                    self._store.discard(self._session_key(oldest))
                self._session_locks.pop(oldest, None)
            if self._store is not None:
                record = self._serialize(session)
                self._store.set(self._session_key(new_id), record)
                if bind_key != new_id:
                    self._store.set(self._session_key(bind_key), record)
        return session, bind_key

    def touch_last_message(self, session_id: str, message_id: str | None) -> None:
        session = self.get(session_id)
        if session is not None and message_id:
            self._update_last(session, message_id)
            if self._store is not None:
                record = self._serialize(session)
                self._store.set(self._session_key(session_id), record)
                if session_id != session.id:
                    self._store.set(self._session_key(session.id), record)

    def forget(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)
        if self._store is not None:
            self._store.discard(self._session_key(session_id))
        self._session_locks.pop(session_id, None)

    def close_all(self) -> None:
        with self._lock:
            self._sessions.clear()
        self._session_locks.clear()
        if self._store is not None:
            prefix = self._key_prefix
            for key, _ in self._store.items():
                if key.startswith(prefix):
                    self._store.discard(key)

    def flush(self) -> None:
        if self._store is not None:
            self._store.flush()
