from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
from pathlib import Path
from typing import Any

from .config import settings

log = logging.getLogger("danyapi.store")

DEFAULT_CACHE_SUBDIR = "danyapi"


def cache_root() -> Path:
    override = settings.cache_dir
    if override:
        root = Path(override)
    else:
        root = Path(tempfile.gettempdir()) / DEFAULT_CACHE_SUBDIR
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        log.warning("cannot create cache dir %s: %s", root, exc)
    return root


class JsonStore:
    def __init__(self, name: str, scope: str | None = None) -> None:
        self._scope = scope
        self._data: dict[str, Any] = {}
        self._lock = threading.Lock()
        self._path: Path | None = None
        if scope:
            safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in f"{name}-{scope}")
            self._path = cache_root() / f"{safe}.json"
            self._load()

    @property
    def enabled(self) -> bool:
        return self._path is not None

    def _load(self) -> None:
        if self._path is None:
            return
        try:
            raw = self._path.read_text(encoding="utf-8")
        except (FileNotFoundError, OSError):
            return
        try:
            data = json.loads(raw)
        except ValueError:
            return
        if isinstance(data, dict):
            self._data = data

    def _flush(self) -> None:
        if self._path is None:
            return
        with self._lock:
            try:
                tmp = self._path.with_name(self._path.name + ".tmp")
                tmp.write_text(json.dumps(self._data, ensure_ascii=False), encoding="utf-8")
                os.replace(tmp, self._path)
            except OSError as exc:
                log.warning("cache write failed for %s: %s", self._path, exc)

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self._data[key] = value
        self._flush()

    def pop(self, key: str, default: Any = None) -> Any:
        if key not in self._data:
            return default
        value = self._data.pop(key)
        self._flush()
        return value

    def discard(self, key: str) -> None:
        if key in self._data:
            self._data.pop(key)
            self._flush()

    def clear(self) -> None:
        self._data.clear()
        self._flush()

    def items(self):
        return list(self._data.items())

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, key: str) -> bool:
        return key in self._data
