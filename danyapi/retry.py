# Shared retry/error constants and helpers.
#
# MAX_RETRIES, RETRY_BACKOFF_SEC, RETRY_BACKOFF_MAX_SEC,
# RETRYABLE_HTTP_STATUSES, _is_retryable_http, _drop_session — all duplicated
# between the DeepSeek API module and the Qwen API module; they live here so
# both sides can import them as local names (for monkeypatching in tests) while
# sharing a single source of truth.

from __future__ import annotations

from fastapi import HTTPException

MAX_RETRIES: int = 5
RETRY_BACKOFF_SEC: float = 1.0
RETRY_BACKOFF_MAX_SEC: float = 8.0

RETRYABLE_HTTP_STATUSES: set[int] = {408, 425, 429, 500, 502, 503, 504}


def _is_retryable_http(exc: HTTPException) -> bool:
    return exc.status_code in RETRYABLE_HTTP_STATUSES


def _drop_session(pool, account, session_key: str) -> None:
    pool.forget(session_key)
    pool.forget_context(session_key)
    account.sessions.forget(session_key)
