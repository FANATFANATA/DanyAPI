from __future__ import annotations

import asyncio
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

_MAX_AFFINITY = 8192


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
    def __init__(self, name: str, scope: str | None = None, maxsize: int = 0) -> None:
        self._scope = scope
        self._maxsize = max(0, int(maxsize))
        self._data: dict[str, Any] = {}
        self._lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._pending = False
        self._dirty = False
        self._idle = threading.Event()
        self._idle.set()
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
        except (FileNotFoundError, OSError, UnicodeError):
            return
        try:
            data = json.loads(raw)
        except ValueError:
            return
        if isinstance(data, dict):
            self._data = data
            self._evict()

    def _evict(self) -> None:
        while self._maxsize > 0 and len(self._data) > self._maxsize:
            self._data.pop(next(iter(self._data)))

    def _commit(self, data: Any) -> None:
        if self._path is None:
            return
        tmp = self._path.with_name(self._path.name + ".tmp")
        try:
            with self._write_lock:
                tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
                os.replace(tmp, self._path)
        except (OSError, TypeError, ValueError) as exc:
            log.warning("cache write failed for %s: %s", self._path, exc)
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass

    def _write(self) -> None:
        if self._path is None:
            return
        with self._lock:
            snapshot = dict(self._data)
        self._commit(snapshot)

    def _flush_background(self) -> None:
        while True:
            with self._lock:
                if not self._dirty:
                    self._pending = False
                    self._idle.set()
                    return
                self._dirty = False
                snapshot = dict(self._data)
            self._commit(snapshot)

    def _note_changed(self) -> None:
        if self._path is None:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            self._write()
            return
        with self._lock:
            self._dirty = True
            if self._pending:
                return
            self._pending = True
            self._idle.clear()
        try:
            loop.run_in_executor(None, self._flush_background)
        except RuntimeError:
            with self._lock:
                self._dirty = False
                self._pending = False
                self._idle.set()
            self._write()

    def flush(self) -> None:
        if self._path is None:
            return
        while True:
            with self._lock:
                if not self._pending:
                    return
            self._idle.wait()

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            if key in self._data and self._data[key] == value:
                return
            self._data.pop(key, None)
            self._data[key] = value
            self._evict()
        self._note_changed()

    def pop(self, key: str, default: Any = None) -> Any:
        with self._lock:
            if key not in self._data:
                return default
            value = self._data.pop(key)
        self._note_changed()
        return value

    def discard(self, key: str) -> None:
        with self._lock:
            if key in self._data:
                self._data.pop(key)
        self._note_changed()

    def clear(self) -> None:
        with self._lock:
            if not self._data:
                return
            self._data.clear()
        self._note_changed()

    def items(self) -> list[tuple[str, Any]]:
        with self._lock:
            return list(self._data.items())

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)

    def __contains__(self, key: str) -> bool:
        with self._lock:
            return key in self._data
