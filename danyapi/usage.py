from __future__ import annotations

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


class UsageTracker:
    _RECENT_PERSIST_INTERVAL = 5.0

    def __init__(self, store: JsonStore | None = None, max_records: int = 1000) -> None:
        self._store = store
        self._max_records = max(1, max_records)
        self._lock = threading.Lock()
        self._totals: dict[str, int] = {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self._by_model: dict[str, dict[str, int]] = {}
        self._by_provider: dict[str, dict[str, int]] = {}
        self._by_user: dict[str, dict[str, int]] = {}
        self._recent: deque[dict[str, Any]] = deque(maxlen=max_records)
        self._last_recent_persist = 0.0
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
                if key in merged and isinstance(value, (int, float)):
                    merged[key] = int(value)
            self._totals = merged
        for attr, key in (("_by_model", "by_model"), ("_by_provider", "by_provider"), ("_by_user", "by_user")):
            bucket = data.get(key)
            if isinstance(bucket, dict):
                restored: dict[str, dict[str, int]] = {}
                for name, entry in bucket.items():
                    if isinstance(entry, dict):
                        row = {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                        for field, value in entry.items():
                            if field in row and isinstance(value, (int, float)):
                                row[field] = int(value)
                        restored[name] = row
                self._evict_overflow(restored)
                setattr(self, attr, restored)
        recent = self._store.get("usage_recent")
        if isinstance(recent, list):
            restored_recent = [entry for entry in recent if isinstance(entry, dict)]
            self._recent = deque(restored_recent[-self._max_records :], maxlen=self._max_records)

    def _serialize(self) -> dict[str, Any]:
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
        prompt_tokens = max(0, int(prompt_tokens or 0))
        completion_tokens = max(0, int(completion_tokens or 0))
        total_tokens = max(0, int(total_tokens or 0))
        if total_tokens == 0:
            total_tokens = prompt_tokens + completion_tokens
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
            if self._store is not None:
                try:
                    self._store.set("usage", self._serialize())
                    now = time.time()
                    if now - self._last_recent_persist >= self._RECENT_PERSIST_INTERVAL:
                        self._last_recent_persist = now
                        self._store.set("usage_recent", list(self._recent))
                except Exception as exc:
                    log.debug("usage store write failed: %s", exc)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "totals": dict(self._totals),
                "by_model": {key: dict(value) for key, value in self._by_model.items()},
                "by_provider": {key: dict(value) for key, value in self._by_provider.items()},
                "by_user": {key: dict(value) for key, value in self._by_user.items()},
                "recent": list(self._recent),
            }

    def flush(self) -> None:
        if self._store is None:
            return
        try:
            with self._lock:
                data = self._serialize()
                recent = list(self._recent)
            self._store.set("usage", data)
            self._store.set("usage_recent", recent)
            self._store.flush()
            self._last_recent_persist = time.time()
        except Exception as exc:
            log.debug("usage flush failed: %s", exc)

    def reset(self) -> None:
        with self._lock:
            self._totals = {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            self._by_model.clear()
            self._by_provider.clear()
            self._by_user.clear()
            self._recent.clear()
            self._last_recent_persist = 0.0
            if self._store is not None:
                try:
                    self._store.discard("usage")
                    self._store.discard("usage_recent")
                except Exception as exc:
                    log.debug("usage store clear failed: %s", exc)
