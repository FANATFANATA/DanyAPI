from __future__ import annotations

import logging

from fastapi import HTTPException

from ..accounts import DeepSeekAccount
from ..deepseek.client import DeepSeekError

log = logging.getLogger("danyapi.api")


DEEPSEEK_AUTH_ERROR_CODES = {40001, 40002, 40003, 40012, 40029}


async def _fresh_pow_upload_headers(account) -> dict:
    try:
        return await account.pow_upload.make_header(lambda: account.client.create_pow_challenge("/api/v0/file/upload_file"))
    except DeepSeekError as exc:
        _handle_account_error(account, exc)
        raise HTTPException(_deepseek_status(exc), _deepseek_error_detail(exc)) from exc


def _drop_session(pool, account, session_key) -> None:
    pool.forget(session_key)
    pool.forget_context(session_key)
    account.sessions.forget(session_key)


def _deepseek_status(exc: DeepSeekError) -> int:
    return 401 if exc.biz_code in DEEPSEEK_AUTH_ERROR_CODES else 502


def _deepseek_error_detail(exc: DeepSeekError) -> str:
    if exc.biz_code in DEEPSEEK_AUTH_ERROR_CODES:
        return f"DeepSeek auth error: {exc}"
    return f"DeepSeek error: {exc}"


def _handle_account_error(account: DeepSeekAccount, exc: Exception) -> None:
    code = getattr(exc, "biz_code", None)
    if code in DEEPSEEK_AUTH_ERROR_CODES:
        account.mark_broken()
        log.warning("account #%d auth error %s: %s", account.index, code, exc)
    else:
        log.warning("account #%d error: %s", account.index, exc)


async def _fresh_pow_headers(account) -> dict:
    try:
        return await account.pow.make_header(account.client.create_pow_challenge)
    except DeepSeekError as exc:
        _handle_account_error(account, exc)
        raise HTTPException(_deepseek_status(exc), _deepseek_error_detail(exc)) from exc
