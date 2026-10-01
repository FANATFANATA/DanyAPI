from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections import deque
from typing import Any

from .store import JsonStore

log = logging.getLogger("danyapi.usage")

_tracker: list[UsageTracker | None] = [None]
_tracker_lock = threading.Lock()


def init_tracker(store: JsonStore | None = None, max_records: int = 1000) -> UsageTracker:
    with _tracker_lock:
        tracker = UsageTracker(store=store, max_records=max_records)
        _tracker[0] = tracker
        return tracker


def get_tracker() -> UsageTracker | None:
    return _tracker[0]


def reset_tracker() -> None:
    with _tracker_lock:
        _tracker[0] = None


def _loop_active() -> bool:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


def _as_count(value: Any) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError):
        try:
            number = int(float(value))
        except (TypeError, ValueError, OverflowError):
            return 0
    return max(0, number)


def record_usage(
    provider: str,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int,
    user: str | None = None,
    session_id: str | None = None,
) -> None:
    tracker = get_tracker()
    if tracker is not None:
        tracker.record(provider, model, prompt_tokens, completion_tokens, total_tokens, user=user, session_id=session_id)


def _usage_count(usage: Any, key: str) -> int:
    if not isinstance(usage, dict):
        return 0
    return _as_count(usage.get(key))


def record_usage_dict(
    provider: str,
    model: str,
    usage: dict,
    user: str | None = None,
    session_id: str | None = None,
) -> None:
    record_usage(
        provider,
        model,
        _usage_count(usage, "prompt_tokens"),
        _usage_count(usage, "completion_tokens"),
        _usage_count(usage, "total_tokens"),
        user=user,
        session_id=session_id,
    )


class UsageTracker:
    _USAGE_PERSIST_INTERVAL = 5.0
    _RECENT_PERSIST_INTERVAL = 5.0

    def __init__(self, store: JsonStore | None = None, max_records: int = 1000) -> None:
        self._store = store
        self._max_records = max(1, max_records)
        self._lock = threading.Lock()
        self._persist_lock = threading.Lock()
        self._totals: dict[str, int] = {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self._by_model: dict[str, dict[str, int]] = {}
        self._by_provider: dict[str, dict[str, int]] = {}
        self._by_user: dict[str, dict[str, int]] = {}
        self._recent: deque[dict[str, Any]] = deque(maxlen=max_records)
        self._last_recent_persist = 0.0
        self._last_usage_persist = 0.0
        self._restore()

    def _restore(self) -> None:
        if self._store is None:
            return
        data = self._store.get("usage")
        if not isinstance(data, dict):
            return
        totals = data.get("totals")
        if isinstance(totals, dict):
            merged = {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            for key, value in totals.items():
                if key in merged:
                    merged[key] = _as_count(value)
            self._totals = merged
        for attr, key in (("_by_model", "by_model"), ("_by_provider", "by_provider"), ("_by_user", "by_user")):
            bucket = data.get(key)
            if isinstance(bucket, dict):
                restored: dict[str, dict[str, int]] = {}
                for name, entry in bucket.items():
                    if isinstance(entry, dict):
                        row = {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                        for field, value in entry.items():
                            if field in row:
                                row[field] = _as_count(value)
                        restored[name] = row
                self._evict_overflow(restored)
                setattr(self, attr, restored)
        recent = self._store.get("usage_recent")
        if isinstance(recent, list):
            restored_recent = [entry for entry in recent if isinstance(entry, dict)]
            self._recent = deque(restored_recent[-self._max_records :], maxlen=self._max_records)

    def _snapshot_locked(self) -> dict[str, Any]:
        return {
            "totals": dict(self._totals),
            "by_model": {key: dict(value) for key, value in self._by_model.items()},
            "by_provider": {key: dict(value) for key, value in self._by_provider.items()},
            "by_user": {key: dict(value) for key, value in self._by_user.items()},
        }

    @staticmethod
    def _add(bucket: dict[str, dict[str, int]], key: str, prompt_tokens: int, completion_tokens: int, total_tokens: int) -> None:
        entry = bucket.pop(key, None)
        if entry is None:
            entry = {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        entry["requests"] += 1
        entry["prompt_tokens"] += prompt_tokens
        entry["completion_tokens"] += completion_tokens
        entry["total_tokens"] += total_tokens
        bucket[key] = entry

    def _evict_overflow(self, bucket: dict[str, dict[str, int]]) -> None:
        while len(bucket) > self._max_records:
            bucket.pop(next(iter(bucket)))

    def record(
        self,
        provider: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        user: str | None = None,
        session_id: str | None = None,
    ) -> None:
        prompt_tokens = _as_count(prompt_tokens)
        completion_tokens = _as_count(completion_tokens)
        total_tokens = _as_count(total_tokens)
        if total_tokens == 0:
            total_tokens = prompt_tokens + completion_tokens
        with self._persist_lock:
            with self._lock:
                self._totals["requests"] += 1
                self._totals["prompt_tokens"] += prompt_tokens
                self._totals["completion_tokens"] += completion_tokens
                self._totals["total_tokens"] += total_tokens
                self._add(self._by_model, model or "unknown", prompt_tokens, completion_tokens, total_tokens)
                self._evict_overflow(self._by_model)
                self._add(self._by_provider, provider or "unknown", prompt_tokens, completion_tokens, total_tokens)
                self._evict_overflow(self._by_provider)
                if user:
                    self._add(self._by_user, user, prompt_tokens, completion_tokens, total_tokens)
                    self._evict_overflow(self._by_user)
                self._recent.append(
                    {
                        "ts": time.time(),
                        "provider": provider,
                        "model": model,
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": total_tokens,
                        "user": user,
                        "session_id": session_id,
                    }
                )
                usage_payload: dict[str, Any] | None = None
                recent_payload: list[dict[str, Any]] | None = None
                if self._store is not None:
                    now = time.time()
                    if not _loop_active() or now - self._last_usage_persist >= self._USAGE_PERSIST_INTERVAL:
                        self._last_usage_persist = now
                        usage_payload = self._snapshot_locked()
                    if now - self._last_recent_persist >= self._RECENT_PERSIST_INTERVAL:
                        self._last_recent_persist = now
                        recent_payload = list(self._recent)
            store = self._store
            if store is not None and (usage_payload is not None or recent_payload is not None):
                try:
                    if usage_payload is not None:
                        store.set("usage", usage_payload)
                    if recent_payload is not None:
                        store.set("usage_recent", recent_payload)
                except Exception as exc:
                    log.warning("usage store write failed: %s", exc)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            data = self._snapshot_locked()
            data["recent"] = [dict(entry) for entry in self._recent]
            return data

    def flush(self) -> None:
        if self._store is None:
            return
        try:
            with self._persist_lock:
                with self._lock:
                    data = self._snapshot_locked()
                    recent = [dict(entry) for entry in self._recent]
                    self._last_recent_persist = time.time()
                    self._last_usage_persist = self._last_recent_persist
                self._store.set("usage", data)
                self._store.set("usage_recent", recent)
                self._store.flush()
        except Exception as exc:
            log.warning("usage flush failed: %s", exc)

    def reset(self) -> None:
        with self._persist_lock:
            with self._lock:
                self._totals = {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                self._by_model.clear()
                self._by_provider.clear()
                self._by_user.clear()
                self._recent.clear()
                self._last_recent_persist = 0.0
                self._last_usage_persist = 0.0
            if self._store is not None:
                try:
                    self._store.discard("usage")
                    self._store.discard("usage_recent")
                except Exception as exc:
                    log.warning("usage store clear failed: %s", exc)
