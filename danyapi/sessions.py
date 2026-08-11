"""Реестр DeepSeek-сессий: сопоставляет сессии OpenAI-пользователей
с серверными чат-сессиями chat.deepseek.com и цепочкой parent_message_id.
"""

from __future__ import annotations

import threading
from typing import Optional

from .deepseek.client import DeepSeekClient, DeepSeekSession


class SessionRegistry:
    def __init__(self, client: DeepSeekClient) -> None:
        self._client = client
        self._sessions: dict[str, DeepSeekSession] = {}
        self._lock = threading.Lock()

    def get(self, session_id: Optional[str]) -> Optional[DeepSeekSession]:
        if not session_id:
            return None
        with self._lock:
            return self._sessions.get(session_id)

    async def obtain(self, session_id: Optional[str]) -> tuple[DeepSeekSession, str]:
        """Возвращает (сессию, её id). Если session_id не задан - создаёт новую."""
        existing = self.get(session_id)
        if existing is not None:
            return existing, session_id  # type: ignore[return-value]
        session = await self._client.create_session()
        new_id = session.id
        with self._lock:
            self._sessions[new_id] = session
        return session, new_id

    def touch_last_message(self, session_id: str, message_id: Optional[str]) -> None:
        session = self.get(session_id)
        if session is not None and message_id:
            session.last_message_id = message_id

    def close_all(self) -> None:
        with self._lock:
            self._sessions.clear()
