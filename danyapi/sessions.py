from __future__ import annotations

import threading
import time
from collections import OrderedDict
from typing import Any


class SessionRegistry:
    def __init__(self, client: Any, maxsize: int = 128, ttl: float = 0.0) -> None:
        self._client = client
        self._sessions: OrderedDict[str, tuple[Any, float]] = OrderedDict()
        self._lock = threading.Lock()
        self._maxsize = max(1, maxsize)
        self._ttl = max(0.0, ttl)

    def _now(self) -> float:
        return time.monotonic()

    def _expired(self, session_id: str, now: float) -> bool:
        return self._ttl > 0 and now - self._sessions[session_id][1] > self._ttl

    async def _create(self, **kwargs) -> Any:
        return await self._client.create_session(**kwargs)

    def _reuse(self, session: Any, session_id: str, **kwargs) -> bool:
        return True

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
                return None
            session = entry[0]
            self._sessions.move_to_end(session_id)
            self._sessions[session_id] = (session, now)
            return session

    async def obtain(self, session_id: str | None, **kwargs) -> tuple[Any, str]:
        if session_id:
            existing = self.get(session_id)
            if existing is not None and self._reuse(existing, session_id, **kwargs):
                return existing, session_id
        session = await self._create(**kwargs)
        new_id = session.id
        now = self._now()
        with self._lock:
            self._sessions[new_id] = (session, now)
            self._sessions.move_to_end(new_id)
            while len(self._sessions) > self._maxsize:
                self._sessions.popitem(last=False)
        return session, new_id

    def touch_last_message(self, session_id: str, message_id: str | None) -> None:
        session = self.get(session_id)
        if session is not None and message_id:
            self._update_last(session, message_id)

    def forget(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    def close_all(self) -> None:
        with self._lock:
            self._sessions.clear()
