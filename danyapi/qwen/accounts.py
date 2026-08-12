from __future__ import annotations

import asyncio
import logging
import threading

from .client import QwenClient, QwenSession

log = logging.getLogger("danyapi.qwen")


class QwenSessionRegistry:
    def __init__(self, client: QwenClient) -> None:
        self._client = client
        self._sessions: dict[str, QwenSession] = {}
        self._lock = threading.Lock()

    def get(self, session_id: str | None) -> QwenSession | None:
        if not session_id:
            return None
        with self._lock:
            return self._sessions.get(session_id)

    async def obtain(self, session_id: str | None, model: str) -> tuple[QwenSession, str]:
        if session_id:
            existing = self.get(session_id)
            if existing is not None:
                return existing, session_id
        chat_id = await self._client.create_chat(model=model, chat_mode="normal")
        session = QwenSession(id=chat_id)
        with self._lock:
            self._sessions[chat_id] = session
        return session, chat_id

    def touch_last_message(self, session_id: str, message_id: str | None) -> None:
        session = self.get(session_id)
        if session is not None and message_id:
            session.last_response_id = message_id


class QwenAccount:
    __slots__ = ("broken", "client", "index", "sem", "sessions")

    def __init__(self, index: int, client: QwenClient) -> None:
        self.index = index
        self.client = client
        self.sem = asyncio.Semaphore(1)
        self.sessions = QwenSessionRegistry(client)
        self.broken = False

    def mark_broken(self) -> None:
        if not self.broken:
            self.broken = True
            log.warning("qwen account #%d marked broken (invalid/expired token)", self.index)

    @property
    def label(self) -> str:
        return f"qwen-acct#{self.index}"
