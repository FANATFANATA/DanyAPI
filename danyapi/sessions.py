from __future__ import annotations

import threading
from collections import OrderedDict

from .deepseek.client import DeepSeekClient, DeepSeekSession


class SessionRegistry:
    def __init__(self, client: DeepSeekClient, maxsize: int = 128) -> None:
        self._client = client
        self._sessions: OrderedDict[str, DeepSeekSession] = OrderedDict()
        self._lock = threading.Lock()
        self._maxsize = max(1, maxsize)

    def get(self, session_id: str | None) -> DeepSeekSession | None:
        if not session_id:
            return None
        with self._lock:
            session = self._sessions.get(session_id)
            if session is not None:
                self._sessions.move_to_end(session_id)
            return session

    async def obtain(self, session_id: str | None) -> tuple[DeepSeekSession, str]:
        if session_id:
            existing = self.get(session_id)
            if existing is not None:
                return existing, session_id
        session = await self._client.create_session()
        new_id = session.id
        with self._lock:
            self._sessions[new_id] = session
            self._sessions.move_to_end(new_id)
            while len(self._sessions) > self._maxsize:
                self._sessions.popitem(last=False)
        return session, new_id

    def touch_last_message(self, session_id: str, message_id: str | None) -> None:
        session = self.get(session_id)
        if session is not None and message_id:
            session.last_message_id = message_id

    def forget(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    def close_all(self) -> None:
        with self._lock:
            self._sessions.clear()
