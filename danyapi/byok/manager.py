from __future__ import annotations

import hashlib
import logging
import secrets
import time
from pathlib import Path

from ..config import settings
from .models import ByokAccount
from .store import UserStore

log = logging.getLogger("danyapi.byok")


def _hash_password(password: str, salt: str) -> str:
    return hashlib.sha256(f"{password}::{salt}".encode()).hexdigest()


class ByokManager:
    def __init__(self, store: UserStore, salt: str) -> None:
        self._store = store
        self._salt = salt
        self._session_keys: dict[str, tuple[str, float]] = {}
        self._max_sessions = 1024

    @classmethod
    def from_settings(cls) -> ByokManager | None:
        if not settings.byok_mode:
            return None
        try:
            base_dir = None
            cache_dir = settings.cache_dir
            if cache_dir:
                base_dir = Path(cache_dir) / "byok"
            store = UserStore(base_dir)
            return cls(store=store, salt=settings.byok_salt)
        except Exception as exc:
            log.error("failed to initialize byok manager: %s", exc)
            return None

    def login(self, username: str, password: str) -> str | None:
        user = self._store.get_user_by_name(username)
        if user is None:
            return None
        expected = _hash_password(password, self._salt)
        if user.get("password_hash") != expected:
            return None
        session_key = secrets.token_urlsafe(32)
        now = time.monotonic()
        self._session_keys[session_key] = (user["id"], now)
        self._cleanup_sessions()
        return session_key

    def register(self, username: str, password: str) -> tuple[bool, str]:
        existing = self._store.get_user_by_name(username)
        if existing is not None:
            return False, "username already taken"
        user_id = secrets.token_urlsafe(16)
        pw_hash = _hash_password(password, self._salt)
        created = self._store.register_user(user_id, username, pw_hash)
        if not created:
            return False, "registration failed"
        return True, user_id

    def logout(self, session_key: str) -> bool:
        return self._session_keys.pop(session_key, None) is not None

    def get_session_user(self, session_key: str) -> str | None:
        entry = self._session_keys.get(session_key)
        if entry is None:
            return None
        user_id, ts = entry
        if 0 < settings.session_ttl < time.monotonic() - ts:
            del self._session_keys[session_key]
            return None
        return user_id

    def add_token(self, user_id: str, provider: str, token_str: str) -> dict[str, str]:
        return self._store.add_token(user_id, provider, token_str)

    def remove_token(self, user_id: str, provider: str) -> bool:
        return self._store.remove_token(user_id, provider)

    def get_user_tokens(self, user_id: str) -> list[dict[str, str]]:
        return self._store.get_tokens_list(user_id)

    def get_provider_token(self, user_id: str, provider: str) -> str | None:
        return self._store.get_token(user_id, provider)

    def get_token_count(self, user_id: str, provider: str) -> int:
        return self._store.get_token_count(user_id, provider)

    def user_has_token(self, user_id: str, provider: str) -> bool:
        return self.get_token_count(user_id, provider) > 0

    def build_account(self, user_id: str, provider: str) -> ByokAccount | None:
        token = self.get_provider_token(user_id, provider)
        if token is None:
            return None
        acc_id = f"byok-{secrets.token_urlsafe(8)}"
        return ByokAccount(id=acc_id, provider=provider, token=token)

    def delete_account(self, user_id: str) -> bool:
        return self._store.delete_user(user_id)

    def auth_check(self, session_key: str) -> str | None:
        return self.get_session_user(session_key)

    def _cleanup_sessions(self) -> None:
        cutoff = time.monotonic() - (settings.session_ttl or 86400)
        stale = [k for k, (_, ts) in self._session_keys.items() if ts < cutoff]
        for k in stale:
            del self._session_keys[k]
        if len(self._session_keys) > self._max_sessions:
            oldest = min(self._session_keys, key=lambda k: self._session_keys[k][1])
            del self._session_keys[oldest]

    def restore(self) -> None:
        pass
