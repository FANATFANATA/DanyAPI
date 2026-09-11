from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any

log = logging.getLogger("danyapi.byok.store")


def _cache_dir() -> Path:
    raw = os.environ.get("DANYAPI_CACHE_DIR", "")
    if raw and os.path.isdir(raw):
        return Path(raw)
    try:
        tmp = Path(os.getenv("TEMP", "/tmp"))  # nosec B108
        return tmp / "danyapi"
    except (OSError, ValueError):
        return Path("/tmp")  # nosec B108


class UserStore:
    def __init__(self, base_dir: Path | None = None) -> None:
        self._base = base_dir or (_cache_dir() / "byok")
        self._lock = threading.Lock()
        self._users: dict[str, dict[str, Any]] = {}
        self._names: dict[str, str] = {}
        self._loaded = False
        self._load()

    def _path_for_user(self, user_id: str) -> Path:
        return self._base / f"user_{user_id}.json"

    def _path_for_names(self) -> Path:
        return self._base / "names.json"

    def _ensure_dir(self) -> None:
        try:
            self._base.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass

    def _load(self) -> None:
        self._ensure_dir()
        names_path = self._path_for_names()
        if names_path.exists():
            try:
                data = json.loads(names_path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    self._users.update(data)
                    self._names = {v["username"]: k for k, v in data.items()}
            except (json.JSONDecodeError, OSError):
                self._users.clear()
                self._names.clear()
        self._loaded = True

    def save_all(self) -> None:
        self._ensure_dir()
        try:
            self._path_for_names().write_text(
                json.dumps(self._users, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError as exc:
            log.warning("store save failed: %s", exc)

    def get_user_by_name(self, username: str) -> dict[str, Any] | None:
        user_id = self._names.get(username)
        if user_id is None:
            return None
        return self._users.get(user_id)

    def get_user(self, user_id: str) -> dict[str, Any] | None:
        return self._users.get(user_id)

    def register_user(self, user_id: str, username: str, password_hash: str) -> bool:
        if user_id in self._users:
            return False
        self._users[user_id] = {"id": user_id, "username": username, "password_hash": password_hash}
        self._names[username] = user_id
        self.save_all()
        return True

    def update_password(self, user_id: str, password_hash: str) -> None:
        user = self._users.get(user_id)
        if user is not None:
            user["password_hash"] = password_hash

    def delete_user(self, user_id: str) -> bool:
        if user_id not in self._users:
            return False
        name = self._users.pop(user_id, {}).get("username", "")
        self._names.pop(name, None)
        path = self._path_for_user(user_id)
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
        self.save_all()
        return True

    def _match_provider(self, acc: dict[str, Any], provider: str) -> bool:
        return acc.get("provider") == provider

    def get_token_count(self, user_id: str, provider: str) -> int:
        accs = self._load_tokens(user_id)
        return sum(1 for a in accs if self._match_provider(a, provider))

    def get_tokens_list(self, user_id: str) -> list[dict[str, Any]]:
        accs = self._load_tokens(user_id)
        return [{"provider": a.get("provider", ""), "token": a.get("token", "")} for a in accs]

    def add_token(self, user_id: str, provider: str, token: str) -> dict[str, Any]:
        accs = self._load_tokens(user_id)
        found = False
        for acc in accs:
            if self._match_provider(acc, provider):
                acc["token"] = token
                found = True
                break
        if not found:
            from uuid import uuid4

            accs.append({"id": uuid4().hex, "provider": provider, "token": token})
        self._save_tokens(user_id, accs)
        return {"provider": provider, "token": token}

    def remove_token(self, user_id: str, provider: str) -> bool:
        accs = self._load_tokens(user_id)
        before = len(accs)
        accs = [a for a in accs if not self._match_provider(a, provider)]
        if len(accs) < before:
            self._save_tokens(user_id, accs)
            return True
        return False

    def get_token(self, user_id: str, provider: str) -> str | None:
        accs = self._load_tokens(user_id)
        for acc in accs:
            if self._match_provider(acc, provider):
                return acc.get("token")
        return None

    def _tokens_path(self, user_id: str) -> Path:
        return self._base / f"tokens_{user_id}.json"

    def _load_tokens(self, user_id: str) -> list[dict[str, Any]]:
        path = self._tokens_path(user_id)
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    return data
            except (json.JSONDecodeError, OSError):
                pass
        return []

    def _save_tokens(self, user_id: str, accs: list[dict[str, Any]]) -> None:
        path = self._tokens_path(user_id)
        try:
            path.write_text(json.dumps(accs, indent=2, ensure_ascii=False), encoding="utf-8")
        except OSError as exc:
            log.warning("tokens save failed: %s", exc)
