from __future__ import annotations

import logging
import threading
import time
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
    def __init__(self, store: JsonStore | None = None, max_records: int = 1000) -> None:
        self._store = store
        self._max_records = max(1, max_records)
        self._lock = threading.Lock()
        self._totals: dict[str, int] = {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self._by_model: dict[str, dict[str, int]] = {}
        self._by_provider: dict[str, dict[str, int]] = {}
        self._by_user: dict[str, dict[str, int]] = {}
        self._recent: list[dict[str, Any]] = []
        self._restore()

    def _restore(self) -> None:
        if self._store is None:
            return
        data = self._store.get("usage")
        if not isinstance(data, dict):
            return
        totals = data.get("totals")
        if isinstance(totals, dict):
            self._totals = {key: int(value) for key, value in totals.items() if isinstance(value, (int, float))}
        for attr, key in (("_by_model", "by_model"), ("_by_provider", "by_provider"), ("_by_user", "by_user")):
            bucket = data.get(key)
            if isinstance(bucket, dict):
                restored: dict[str, dict[str, int]] = {}
                for name, entry in bucket.items():
                    if isinstance(entry, dict):
                        restored[name] = {k: int(v) for k, v in entry.items() if isinstance(v, (int, float))}
                setattr(self, attr, restored)

    def _serialize(self) -> dict[str, Any]:
        return {
            "totals": self._totals,
            "by_model": self._by_model,
            "by_provider": self._by_provider,
            "by_user": self._by_user,
        }

    @staticmethod
    def _add(bucket: dict[str, dict[str, int]], key: str, prompt_tokens: int, completion_tokens: int, total_tokens: int) -> None:
        entry = bucket.setdefault(key, {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
        entry["requests"] += 1
        entry["prompt_tokens"] += prompt_tokens
        entry["completion_tokens"] += completion_tokens
        entry["total_tokens"] += total_tokens

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
            self._add(self._by_provider, provider or "unknown", prompt_tokens, completion_tokens, total_tokens)
            if user:
                self._add(self._by_user, user, prompt_tokens, completion_tokens, total_tokens)
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
            if len(self._recent) > self._max_records:
                del self._recent[: len(self._recent) - self._max_records]
            if self._store is not None:
                try:
                    self._store.set("usage", self._serialize())
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

    def reset(self) -> None:
        with self._lock:
            self._totals = {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            self._by_model.clear()
            self._by_provider.clear()
            self._by_user.clear()
            self._recent.clear()
            if self._store is not None:
                try:
                    self._store.discard("usage")
                except Exception as exc:
                    log.debug("usage store clear failed: %s", exc)
