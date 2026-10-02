from __future__ import annotations

import asyncio
import getpass
import json
import logging
import os
import stat
import tempfile
import threading
import time
import weakref
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .config import settings

log = logging.getLogger("danyapi.store")

DEFAULT_CACHE_SUBDIR = "danyapi"

_MAX_AFFINITY = 8192

_FLUSH_WAIT_INTERVAL = 0.5
_FLUSH_MAX_WAIT = 5.0
_FLUSH_DEBOUNCE = 0.25
_MAX_STORE_BYTES = 64 * 1024 * 1024
_FLUSH_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="danyapi-cache")

_PATH_LOCKS: dict[str, threading.Lock] = {}
_PATH_LOCKS_GUARD = threading.Lock()
_LIVE_STORES: dict[str, weakref.ReferenceType[JsonStore]] = {}
_LIVE_STORES_GUARD = threading.Lock()


def _path_write_lock(path: Path) -> threading.Lock:
    key = str(path)
    with _PATH_LOCKS_GUARD:
        lock = _PATH_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _PATH_LOCKS[key] = lock
        return lock


class _StoreState:
    __slots__ = ("data", "lock")

    def __init__(self) -> None:
        self.data: dict[str, Any] = {}
        self.lock = threading.Lock()


_STATES: dict[str, _StoreState] = {}
_STATES_GUARD = threading.Lock()


def _forget_path_registries(path: Path, store: JsonStore | None = None) -> None:
    key = str(path)
    with _STATES_GUARD:
        _STATES.pop(key, None)
    with _PATH_LOCKS_GUARD:
        _PATH_LOCKS.pop(key, None)
    with _LIVE_STORES_GUARD:
        existing = _LIVE_STORES.get(key)
        if existing is None or existing() is None or store is None or existing() is store:
            _LIVE_STORES.pop(key, None)


def _fsync_dir(path: Path) -> None:
    if os.name == "nt":
        return
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _report_flush_failure(future: Any) -> None:
    try:
        exc = future.exception()
    except (asyncio.CancelledError, asyncio.InvalidStateError):
        return
    if exc is not None:
        log.warning("cache flush failed: %s", exc)


def _getuid() -> int | None:
    getter = getattr(os, "getuid", None)
    return None if getter is None else int(getter())  # pylint: disable=not-callable


def _root_rejection(root: Path) -> str | None:
    try:
        info = os.lstat(root)
    except FileNotFoundError:
        return None
    except OSError as exc:
        return f"cannot be inspected ({exc})"
    if stat.S_ISLNK(info.st_mode):
        return "it is a symbolic link"
    if not stat.S_ISDIR(info.st_mode):
        return None
    uid = _getuid()
    if uid is not None and info.st_uid != uid:
        return f"it belongs to uid {info.st_uid}"
    return None


def _user_root_tag() -> str:
    uid = _getuid()
    if uid is not None:
        return f"{DEFAULT_CACHE_SUBDIR}-{uid}"
    try:
        user = getpass.getuser()
    except (KeyError, OSError, ImportError):
        user = "default"
    return f"{DEFAULT_CACHE_SUBDIR}-{user or 'default'}"


def _private_cache_root() -> Path:
    return Path(tempfile.gettempdir()) / _user_root_tag()


def cache_root() -> Path:
    override = settings.cache_dir
    root = Path(override) if override else _private_cache_root()
    reason = _root_rejection(root)
    if reason is not None:
        fallback = _private_cache_root()
        log.warning("cache dir %s is not safe to use (%s), using %s instead", root, reason, fallback)
        root = fallback
        reason = _root_rejection(root)
    if reason is not None:
        log.warning("cannot use cache dir %s: %s", root, reason)
        root = Path(tempfile.mkdtemp(prefix=_user_root_tag() + "-fallback-"))
        try:
            os.chmod(root, 0o700)
        except OSError as exc:
            log.warning("cannot secure the fallback cache dir %s: %s", root, exc)
    try:
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(root, 0o700)
    except OSError as exc:
        log.warning("cannot create cache dir %s: %s", root, exc)
    return root


class JsonStore:
    def __init__(self, name: str, scope: str | None = None, maxsize: int = 0) -> None:
        self._scope = scope
        self._maxsize = max(0, int(maxsize))
        self._state = _StoreState()
        self._lock = self._state.lock
        self._data = self._state.data
        self._pending = False
        self._dirty = False
        self._removed = False
        self._generation = 0
        self._committed_generation = -1
        self._commit_failed = False
        self._idle = threading.Event()
        self._idle.set()
        self._path: Path | None = None
        if scope and settings.cache_enabled:
            safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in f"{name}-{scope}")
            self._path = cache_root() / f"{safe}.json"
            with _STATES_GUARD:
                shared = _STATES.get(str(self._path))
                if shared is not None:
                    self._state = shared
                    self._lock = shared.lock
                    self._data = shared.data
                else:
                    _STATES[str(self._path)] = self._state
            self._register_live(self._path)
            self._load()

    def _register_live(self, path: Path) -> None:
        key = str(path)
        with _LIVE_STORES_GUARD:
            for other, reference in list(_LIVE_STORES.items()):
                if other != key and reference() is None:
                    del _LIVE_STORES[other]
            existing = _LIVE_STORES.get(key)
            if existing is not None and existing() is not None and existing() is not self:
                log.warning("another JsonStore instance already holds %s, both share one in-memory snapshot", path)
            _LIVE_STORES[key] = weakref.ref(self)

    @property
    def enabled(self) -> bool:
        return self._path is not None and not self._removed

    @property
    def path(self) -> Path | None:
        return self._path

    def _load(self) -> None:
        if self._path is None:
            return
        try:
            if self._path.stat().st_size > _MAX_STORE_BYTES:
                log.warning("cache file %s is larger than %d bytes and was ignored", self._path, _MAX_STORE_BYTES)
                return
            raw = self._path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return
        except (OSError, UnicodeError, MemoryError, ValueError) as exc:
            log.warning("cache read failed for %s: %s", self._path, exc)
            return
        try:
            data = json.loads(raw)
        except ValueError as exc:
            log.warning("cache file %s is corrupt and was ignored: %s", self._path, exc)
            return
        if not isinstance(data, dict):
            log.warning("cache file %s has unexpected root type %s and was ignored", self._path, type(data).__name__)
            return
        with self._lock:
            self._data.clear()
            self._data.update(data)
            self._evict()

    def _evict(self) -> None:
        while self._maxsize > 0 and len(self._data) > self._maxsize:
            self._data.pop(next(iter(self._data)))

    def _commit(self, data: Any, generation: int | None = None) -> None:
        path = self._path
        if path is None or self._removed:
            return
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        try:
            with _path_write_lock(path):
                if self._removed:
                    return
                if generation is not None and generation < self._committed_generation:
                    return
                fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    handle.write(json.dumps(data, ensure_ascii=False))
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(tmp, path)
                _fsync_dir(path.parent)
        except (OSError, TypeError, ValueError, MemoryError) as exc:
            self._commit_failed = True
            log.warning("cache write failed for %s: %s", path, exc)
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
            return
        self._commit_failed = False
        if generation is not None and generation > self._committed_generation:
            self._committed_generation = generation

    def _write(self) -> None:
        if self._path is None or self._removed:
            return
        with self._lock:
            snapshot = dict(self._data)
            generation = self._generation
        self._commit(snapshot, generation)

    def _flush_background(self) -> None:
        while True:
            with self._lock:
                generation = self._generation
            deadline = time.monotonic() + _FLUSH_DEBOUNCE
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                time.sleep(min(0.01, remaining))
            with self._lock:
                if self._removed or not self._dirty:
                    self._dirty = False
                    self._pending = False
                    self._idle.set()
                    return
                snapshot = dict(self._data)
                current = self._generation
                settled = current == generation
            self._commit(snapshot, current)
            if settled:
                with self._lock:
                    self._dirty = False
                    self._pending = False
                    self._idle.set()
                return

    def _note_changed(self) -> None:
        if self._path is None or self._removed:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            with self._lock:
                self._generation += 1
            self._write()
            return
        with self._lock:
            self._dirty = True
            self._generation += 1
            if self._pending:
                return
            self._pending = True
            self._idle.clear()
        try:
            future = loop.run_in_executor(_FLUSH_EXECUTOR, self._flush_background)
        except RuntimeError:
            with self._lock:
                self._dirty = False
                self._pending = False
                self._idle.set()
            self._write()
            return
        future.add_done_callback(_report_flush_failure)

    def flush(self, timeout: float = _FLUSH_MAX_WAIT) -> None:
        if self._path is None:
            return
        deadline = time.monotonic() + timeout
        while True:
            with self._lock:
                if not self._pending:
                    return
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                log.warning("cache flush for %s timed out after %gs with a write still pending", self._path, timeout)
                return
            self._idle.wait(min(remaining, _FLUSH_WAIT_INTERVAL))

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        if self._path is not None and not self._removed:
            try:
                json.dumps(value, ensure_ascii=False)
            except (TypeError, ValueError, RecursionError, MemoryError) as exc:
                log.warning("cache value for %s is not serialisable and was not stored: %s", key, exc)
                return
        with self._lock:
            if not self._commit_failed and key in self._data and self._data[key] == value:
                return
            self._data.pop(key, None)
            self._data[key] = value
            self._evict()
        self._note_changed()

    def pop(self, key: str, default: Any = None) -> Any:
        with self._lock:
            if key in self._data:
                value = self._data.pop(key)
            else:
                value = default
                if not self._commit_failed:
                    return value
        self._note_changed()
        return value

    def discard(self, key: str) -> None:
        with self._lock:
            if key in self._data:
                self._data.pop(key)
            elif not self._commit_failed:
                return
        self._note_changed()

    def clear(self) -> None:
        with self._lock:
            if not self._data and not self._commit_failed:
                return
            self._data.clear()
        self._note_changed()

    def remove(self) -> None:
        path = self._path
        if path is None:
            return
        self.flush()
        self._removed = True
        with self._lock:
            self._data.clear()
            self._dirty = False
            self._pending = False
            self._idle.set()
        try:
            with _path_write_lock(path):
                path.unlink(missing_ok=True)
        except OSError as exc:
            log.warning("cache file delete failed for %s: %s", path, exc)
        _forget_path_registries(path, self)

    def items(self) -> list[tuple[str, Any]]:
        with self._lock:
            return list(self._data.items())

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)

    def __contains__(self, key: str) -> bool:
        with self._lock:
            return key in self._data
