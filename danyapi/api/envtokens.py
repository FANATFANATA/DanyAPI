from __future__ import annotations

import asyncio
import hmac
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

from fastapi import Depends, HTTPException, Request

from ..accounts import AccountPool, DeepSeekAccount
from ..config import settings
from ..deepseek.client import DeepSeekClient
from ..qwen.accounts import QwenAccount
from ..qwen.client import QwenClient
from .core import _shared_store, _token_stable_id
from .models import _fetch_qwen_models
from .state import _byok_mode, app

log = logging.getLogger("danyapi.api")

_TOKENS_LOCK = asyncio.Lock()


def _env_path() -> Path:
    return Path(__file__).resolve().parents[2] / ".env"


def _unquote_env_value(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
        return value[1:-1]
    return value


def _atomic_write_text(target: Path, text: str) -> None:
    handle, tmp_name = tempfile.mkstemp(dir=str(target.parent), suffix=f".{target.name}.tmp")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as handle_out:
            handle_out.write(text)
        os.replace(tmp_path, target)
    except OSError:
        tmp_path.unlink(missing_ok=True)
        raise


def _read_env_tokens_sync() -> tuple[list[str], list[str]]:
    env_file = _env_path()
    if not env_file.exists():
        return [], []
    ds_tokens = ""
    qw_tokens = ""
    for line in env_file.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("DEEPSEEK_TOKENS="):
            ds_tokens = _unquote_env_value(stripped.split("=", 1)[1].strip())
        elif stripped.startswith("QWEN_TOKENS="):
            qw_tokens = _unquote_env_value(stripped.split("=", 1)[1].strip())
    ds_list = [t.strip() for t in ds_tokens.split(",") if t.strip()] if ds_tokens else []
    qw_list = [t.strip() for t in qw_tokens.split(",") if t.strip()] if qw_tokens else []
    return ds_list, qw_list


async def _read_env_tokens() -> tuple[list[str], list[str]]:
    return await asyncio.to_thread(_read_env_tokens_sync)


def _write_env_tokens_sync(ds_tokens: list[str], qw_tokens: list[str]) -> None:
    env_file = _env_path()
    ds_line = f"DEEPSEEK_TOKENS={','.join(ds_tokens)}"
    qw_line = f"QWEN_TOKENS={','.join(qw_tokens)}"
    new_lines: list[str] = []
    ds_set = False
    qw_set = False
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("DEEPSEEK_TOKENS="):
                new_lines.append(ds_line)
                ds_set = True
            elif stripped.startswith("QWEN_TOKENS="):
                new_lines.append(qw_line)
                qw_set = True
            else:
                new_lines.append(line)
    if not ds_set:
        new_lines.append(ds_line)
    if not qw_set:
        new_lines.append(qw_line)
    _atomic_write_text(env_file, "\n".join(new_lines) + "\n")


async def _write_env_tokens(ds_tokens: list[str], qw_tokens: list[str]) -> None:
    await asyncio.to_thread(_write_env_tokens_sync, ds_tokens, qw_tokens)


def _env_token_list(value: Any, field: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise HTTPException(400, f"{field} must be a list of strings")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise HTTPException(400, f"{field} must contain only strings")
        token = item.strip()
        if token:
            result.append(token)
    return result


def _require_admin_token(request: Request) -> None:
    if _byok_mode():
        raise HTTPException(404, "Unknown /v1 endpoint: /v1/tokens")
    expected = settings.admin_token
    if not expected:
        raise HTTPException(404, "token management is disabled, set DANYAPI_ADMIN_TOKEN to enable it")
    provided = (request.headers.get("x-api-key") or "").strip()
    authorization = request.headers.get("authorization") or ""
    if not provided and authorization.lower().startswith("bearer "):
        provided = authorization[7:].strip()
    if not provided or not hmac.compare_digest(provided, expected):
        raise HTTPException(401, "invalid or missing admin token")


def _pool_account_by_stable(pool: AccountPool | None, stable_id: str) -> Any | None:
    if pool is None:
        return None
    for acct in pool.accounts:
        if getattr(acct, "stable_id", None) == stable_id:
            return acct
    return None


@app.post("/v1/tokens", dependencies=[Depends(_require_admin_token)])
async def add_tokens(tokens: dict) -> dict:
    async with _TOKENS_LOCK:
        new_ds = _env_token_list(tokens.get("deepseek_tokens"), "deepseek_tokens")
        new_qw = _env_token_list(tokens.get("qwen_tokens"), "qwen_tokens")
        if not new_ds and not new_qw:
            raise HTTPException(400, "no tokens provided")

        existing_ds, existing_qw = await _read_env_tokens()
        ds_candidates = [t for t in dict.fromkeys(new_ds) if t not in existing_ds]
        qw_candidates = [t for t in dict.fromkeys(new_qw) if t not in existing_qw]

        pool: AccountPool | None = getattr(app.state, "pool", None)
        qwen_pool: AccountPool | None = getattr(app.state, "qwen_pool", None)

        activated_ds = 0
        activated_qw = 0

        for token in dict.fromkeys(new_ds):
            if token in existing_ds:
                acct = _pool_account_by_stable(pool, _token_stable_id(token))
                if acct is None or not acct.broken:
                    continue
                try:
                    valid_auth = await acct.client.check_auth()
                except Exception as exc:
                    log.warning("reactivation check failed for existing deepseek token: %s", exc)
                    continue
                if not valid_auth:
                    continue
                acct.broken = False
                acct.broken_at = None
                activated_ds += 1
                log.info("reactivated deepseek token (total accounts: %d)", len(pool.accounts) if pool else 0)

        for token in dict.fromkeys(new_qw):
            if token in existing_qw:
                acct = _pool_account_by_stable(qwen_pool, _token_stable_id(token))
                if acct is None or not acct.broken:
                    continue
                try:
                    valid_auth = await acct.client.check_auth()
                except Exception as exc:
                    log.warning("reactivation check failed for existing qwen token: %s", exc)
                    continue
                if not valid_auth:
                    continue
                acct.broken = False
                acct.broken_at = None
                activated_qw += 1
                log.info("reactivated qwen token (total accounts: %d)", len(qwen_pool.accounts) if qwen_pool else 0)

        if not ds_candidates and not qw_candidates:
            if activated_ds or activated_qw:
                return {
                    "success": True,
                    "message": "Tokens reactivated.",
                    "added": {"deepseek": 0, "qwen": 0},
                    "skipped": {"deepseek": 0, "qwen": 0},
                    "reactivated": {"deepseek": activated_ds, "qwen": activated_qw},
                }
            raise HTTPException(400, "all provided tokens already exist")

        added_ds = 0
        added_qw = 0
        skipped_ds = 0
        skipped_qw = 0

        ds_store = _shared_store("deepseek_session_store", "deepseek-sessions")
        qw_store = _shared_store("qwen_session_store", "qwen-sessions")
        ds_context_store = _shared_store("deepseek_context_store", "deepseek-contexts")
        qw_context_store = _shared_store("qwen_context_store", "qwen-contexts")
        ds_affinity_store = _shared_store("deepseek_affinity_store", "deepseek-affinities")
        qw_affinity_store = _shared_store("qwen_affinity_store", "qwen-affinities")

        accepted_ds: list[str] = []
        accepted_qw: list[str] = []

        for token in ds_candidates:
            client = DeepSeekClient(token=token, timeout=settings.timeout)
            if not await client.check_auth():
                log.warning("new deepseek token invalid/expired, skipping")
                await client.aclose()
                skipped_ds += 1
                continue
            acct = DeepSeekAccount(
                len(pool.accounts) if pool else 0,
                client,
                session_cache_size=settings.session_cache_size,
                ttl=settings.session_ttl,
                store=ds_store,
                stable_id=_token_stable_id(token),
            )
            if pool is None:
                pool = AccountPool(
                    [acct],
                    session_cache_size=settings.session_cache_size,
                    ttl=settings.session_ttl,
                    context_store=ds_context_store,
                    affinity_store=ds_affinity_store,
                )
                app.state.pool = pool
            else:
                pool.add_account(acct)
            accepted_ds.append(token)
            added_ds += 1
            log.info("hot-added deepseek token (total accounts: %d)", len(pool.accounts))

        for token in qw_candidates:
            qw_client = QwenClient(token=token, timeout=settings.timeout)
            if not await qw_client.check_auth():
                log.warning("new qwen token invalid/expired, skipping")
                await qw_client.aclose()
                skipped_qw += 1
                continue
            qw_acct = QwenAccount(
                len(qwen_pool.accounts) if qwen_pool else 0,
                qw_client,
                session_cache_size=settings.session_cache_size,
                ttl=settings.session_ttl,
                store=qw_store,
                stable_id=_token_stable_id(token),
            )
            if qwen_pool is None:
                qwen_pool = AccountPool(
                    [qw_acct],
                    label="qwen",
                    session_cache_size=settings.session_cache_size,
                    ttl=settings.session_ttl,
                    context_store=qw_context_store,
                    affinity_store=qw_affinity_store,
                )
                app.state.qwen_pool = qwen_pool
            else:
                qwen_pool.add_account(qw_acct)
            accepted_qw.append(token)
            added_qw += 1
            log.info("hot-added qwen token (total accounts: %d)", len(qwen_pool.accounts))

        if added_qw and qwen_pool is not None:
            try:
                app.state.qwen_models = await _fetch_qwen_models(qwen_pool.accounts[0].client)
            except Exception as exc:
                log.warning("failed to refresh qwen models: %s", exc)

        merged_ds = existing_ds + accepted_ds
        merged_qw = existing_qw + accepted_qw
        await _write_env_tokens(merged_ds, merged_qw)
        settings.deepseek_tokens = merged_ds
        settings.qwen_tokens = merged_qw

        return {
            "success": True,
            "message": "Tokens added and activated." if (added_ds or added_qw) else "No valid tokens to add.",
            "added": {"deepseek": added_ds, "qwen": added_qw},
            "skipped": {"deepseek": skipped_ds, "qwen": skipped_qw},
            "reactivated": {"deepseek": activated_ds, "qwen": activated_qw},
        }
