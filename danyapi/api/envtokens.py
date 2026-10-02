from __future__ import annotations

import asyncio
import hmac
import logging
import os
import re
import tempfile
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from fastapi import Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict, ValidationError

from .. import config as config_mod
from ..accounts import AccountPool, DeepSeekAccount
from ..config import settings
from ..deepseek.client import DeepSeekClient
from ..qwen.accounts import QwenAccount
from ..qwen.client import QwenClient
from .core import _shared_store, _token_stable_id, _validation_summary
from .models import _fetch_qwen_models, refresh_provider_models
from .state import _byok_mode, app

log = logging.getLogger("danyapi.api")

_TOKENS_LOCK = asyncio.Lock()
_ENV_WRITE_LOCK = threading.Lock()

MAX_TOKENS_PER_REQUEST = 64
MAX_TOKEN_LENGTH = 4096
AUTH_CONCURRENCY = 8

_TOKEN_LINE_RE = re.compile(r"^(?P<indent>[ \t]*)(?:export[ \t]+)?(?P<name>DEEPSEEK_TOKENS|QWEN_TOKENS)(?P<gap>[ \t]*=[ \t]*)(?P<value>.*)$")
_FORBIDDEN_TOKEN_RE = re.compile("""[, "'#\\\\]|[\\x00-\\x1f\\x7f]""")

_PLAN_LIVE = "live"
_PLAN_BROKEN = "broken"
_PLAN_UNTRACKED = "untracked"
_PLAN_NEW = "new"

_CREDENTIAL_NAMES = ("DEEPSEEK_TOKENS", "QWEN_TOKENS")

_dotenv_values: Callable[[Path], dict[str, str | None]] | None
try:
    from dotenv import dotenv_values as _loaded_dotenv_values

    _dotenv_values = _loaded_dotenv_values
except ImportError:
    _dotenv_values = None


class AddTokensRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    deepseek_tokens: list[str] | None = None
    qwen_tokens: list[str] | None = None


def _env_path() -> Path:
    for name in ("ENV_PATH", "_ENV_PATH"):
        value = getattr(config_mod, name, None)
        if isinstance(value, Path):
            return value
    return Path(config_mod.__file__).resolve().parents[1] / ".env"


def _unquote_env_value(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
        return value[1:-1]
    return value


def _strip_inline_comment(value: str) -> str:
    quote = ""
    for index, char in enumerate(value):
        if quote:
            if char == quote:
                quote = ""
            continue
        if char in ('"', "'"):
            quote = char
        elif char == "#" and (index == 0 or value[index - 1] in " \t"):
            return value[:index]
    return value


def _fallback_env_values(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("export "):
            stripped = stripped[7:].strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        name, _, raw = stripped.partition("=")
        name = name.strip()
        if name in _CREDENTIAL_NAMES:
            values[name] = _unquote_env_value(_strip_inline_comment(raw).strip())
    return values


def _split_token_list(raw: str) -> list[str]:
    items: list[str] = []
    current: list[str] = []
    escaped = False
    for char in raw:
        if escaped:
            escaped = False
            current.append(char if char == "," else f"\\{char}")
            continue
        if char == "\\":
            escaped = True
            continue
        if char == ",":
            items.append("".join(current).strip())
            current = []
            continue
        current.append(char)
    if escaped:
        current.append("\\")
    items.append("".join(current).strip())
    return [item for item in items if item]


def _read_env_values(env_file: Path) -> dict[str, str]:
    try:
        if not env_file.exists():
            return {}
        if _dotenv_values is not None:
            return {key: value for key, value in _dotenv_values(env_file).items() if isinstance(value, str)}
        return _fallback_env_values(env_file.read_text(encoding="utf-8"))
    except HTTPException:
        raise
    except (OSError, UnicodeError, MemoryError, ValueError) as exc:
        log.warning("cannot read the credentials file: %s", exc)
        raise HTTPException(500, "cannot read the credentials file") from exc


def _fsync_dir(path: Path) -> None:
    if os.name == "nt":
        return
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_write_text(target: Path, text: str) -> None:
    with _ENV_WRITE_LOCK:
        handle, tmp_name = tempfile.mkstemp(dir=str(target.parent), suffix=f".{target.name}.tmp")
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as handle_out:
                handle_out.write(text)
                handle_out.flush()
                os.fsync(handle_out.fileno())
            os.replace(tmp_path, target)
            _fsync_dir(target.parent)
        except OSError:
            tmp_path.unlink(missing_ok=True)
            raise


def _read_env_tokens_sync() -> tuple[list[str], list[str]]:
    values = _read_env_values(_env_path())
    return _split_token_list(values.get("DEEPSEEK_TOKENS", "")), _split_token_list(values.get("QWEN_TOKENS", ""))


async def _read_env_tokens() -> tuple[list[str], list[str]]:
    return await asyncio.to_thread(_read_env_tokens_sync)


def _write_env_tokens_sync(ds_tokens: list[str], qw_tokens: list[str]) -> None:
    env_file = _env_path()
    lines: list[str] = []
    if env_file.exists():
        try:
            lines = env_file.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeError, MemoryError) as exc:
            log.warning("cannot read the credentials file before writing: %s", exc)
            raise HTTPException(500, "cannot read the credentials file") from None
    replacements = {"DEEPSEEK_TOKENS": ",".join(ds_tokens), "QWEN_TOKENS": ",".join(qw_tokens)}
    new_lines: list[str] = []
    replaced: set[str] = set()
    for line in lines:
        match = _TOKEN_LINE_RE.match(line)
        if match is None:
            new_lines.append(line)
            continue
        name = match.group("name")
        if name in replaced:
            continue
        replaced.add(name)
        new_lines.append(f"{match.group('indent')}{name}{match.group('gap')}{replacements[name]}")
    for name, value in replacements.items():
        if name not in replaced:
            new_lines.append(f"{name}={value}")
    try:
        _atomic_write_text(env_file, "\n".join(new_lines) + "\n")
    except OSError as exc:
        log.warning("cannot write the credentials file: %s", exc)
        raise HTTPException(500, "cannot write the credentials file") from exc


async def _write_env_tokens(ds_tokens: list[str], qw_tokens: list[str]) -> None:
    await asyncio.to_thread(_write_env_tokens_sync, ds_tokens, qw_tokens)


def _validate_token(token: str, field: str) -> str:
    if len(token) > MAX_TOKEN_LENGTH:
        raise HTTPException(400, f"{field} entries must be at most {MAX_TOKEN_LENGTH} characters")
    if "\n" in token or "\r" in token:
        raise HTTPException(400, f"{field} entries must not contain line breaks")
    if _FORBIDDEN_TOKEN_RE.search(token):
        raise HTTPException(400, f"{field} entries must not contain commas, quotes, backslashes, '#' or spaces")
    return token


def _env_token_list(value: Any, field: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise HTTPException(400, f"{field} must be a list of strings")
    if len(value) > MAX_TOKENS_PER_REQUEST:
        raise HTTPException(400, f"too many tokens in {field}: max {MAX_TOKENS_PER_REQUEST} per request")
    result: list[str] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, str):
            raise HTTPException(400, f"{field} must contain only strings")
        token = item.strip()
        if not token or token in seen:
            continue
        seen.add(token)
        result.append(_validate_token(token, field))
    if len(result) > MAX_TOKENS_PER_REQUEST:
        raise HTTPException(400, f"too many tokens in {field}: max {MAX_TOKENS_PER_REQUEST} per request")
    return result


def _request_client(request: Request) -> str:
    client = request.client
    return client.host if client is not None and client.host else "-"


def _presented_admin_token(request: Request) -> str:
    provided = (request.headers.get("x-api-key") or "").strip()
    authorization = request.headers.get("authorization") or ""
    if not provided and authorization.lower().startswith("bearer "):
        provided = authorization[7:].strip()
    return provided


def admin_token_matches(request: Request) -> bool:
    expected = settings.admin_token
    if not expected:
        return False
    provided = _presented_admin_token(request)
    if not provided:
        return False
    return hmac.compare_digest(provided.encode("utf-8"), expected.encode("utf-8"))


def _require_admin_token(request: Request) -> None:
    if _byok_mode():
        raise HTTPException(404, "Unknown /v1 endpoint: /v1/tokens")
    if not settings.admin_token:
        raise HTTPException(404, "token management is disabled, set DANYAPI_ADMIN_TOKEN to enable it")
    if not _presented_admin_token(request):
        log.warning("rejected POST /v1/tokens from %s: no admin token presented", _request_client(request))
        raise HTTPException(401, "invalid or missing admin token")
    if not hmac.compare_digest(_presented_admin_token(request).encode("utf-8"), settings.admin_token.encode("utf-8")):
        log.warning("rejected POST /v1/tokens from %s: wrong admin token", _request_client(request))
        raise HTTPException(401, "invalid or missing admin token")


def _coerce_tokens(value: Any) -> AddTokensRequest:
    if isinstance(value, AddTokensRequest):
        return value
    try:
        return AddTokensRequest.model_validate(value)
    except ValidationError as exc:
        raise HTTPException(400, f"invalid request body: {_validation_summary(exc.errors())}") from None


def _pool_account_by_stable(pool: AccountPool | None, stable_id: str) -> Any | None:
    if pool is None:
        return None
    for acct in pool.accounts:
        if getattr(acct, "stable_id", None) == stable_id:
            return acct
    return None


def _plan_tokens(tokens: list[str], known: set[str], pool: Any) -> list[tuple[str, str, Any]]:
    plan: list[tuple[str, str, Any]] = []
    for token in tokens:
        acct = _pool_account_by_stable(pool, _token_stable_id(token))
        if acct is None:
            state = _PLAN_UNTRACKED if token in known else _PLAN_NEW
        elif getattr(acct, "broken", False):
            state = _PLAN_BROKEN
        else:
            state = _PLAN_LIVE
        plan.append((token, state, acct))
    return plan


def _known_tokens(stored: list[str], loaded: list[str]) -> set[str]:
    return {token for token in [*stored, *loaded] if token}


class TokenCheckFailed(Exception):
    pass


async def _check_token_auth(client: Any) -> bool:
    try:
        return bool(await client.check_auth())
    except Exception as exc:
        log.warning("token auth check failed: %s: %s", type(exc).__name__, exc)
        raise TokenCheckFailed(f"token auth check failed: {type(exc).__name__}: {exc}") from exc


async def _close_client(client: Any) -> None:
    try:
        await client.aclose()
    except Exception as exc:
        log.warning("token client close failed: %s", exc)


_CLOSE_TASKS: set[asyncio.Task] = set()


def _close_client_later(client: Any) -> None:
    if client is None:
        return
    task = asyncio.create_task(_close_client(client))
    _CLOSE_TASKS.add(task)
    task.add_done_callback(_CLOSE_TASKS.discard)


async def _validated_tokens(
    plan: list[tuple[str, str, Any]],
    provider: str,
    factory: Callable[[str], Any],
) -> tuple[list[tuple[str, str, Any, Any]], int]:
    sem = asyncio.Semaphore(AUTH_CONCURRENCY)
    skipped: list[int] = [0]
    unchecked: list[str] = []

    async def _one(item: tuple[str, str, Any]) -> tuple[str, str, Any, Any] | None:
        token, state, acct = item
        if state == _PLAN_LIVE:
            return None
        async with sem:
            if state == _PLAN_BROKEN:
                try:
                    revived = await _check_token_auth(acct.client)
                except TokenCheckFailed as exc:
                    unchecked.append(f"{provider}: {exc}")
                    return None
                return (token, state, acct, acct.client) if revived else None
            client = factory(token)
            try:
                accepted = await _check_token_auth(client)
            except TokenCheckFailed as exc:
                unchecked.append(f"{provider}: {exc}")
                await _close_client(client)
                return None
            except BaseException:
                _close_client_later(client)
                raise
            if not accepted:
                log.warning("new %s token invalid/expired, skipping", provider)
                await _close_client(client)
                if state == _PLAN_NEW:
                    skipped[0] += 1
                return None
            return (token, state, acct, client)

    results = await asyncio.gather(*(_one(item) for item in plan), return_exceptions=True)
    validated: list[tuple[str, str, Any, Any]] = []
    for result in results:
        if isinstance(result, BaseException):
            log.warning("token auth check did not finish: %s: %s", type(result).__name__, result)
            continue
        if result is not None:
            validated.append(result)
    if unchecked:
        for _token, _state, acct, client in validated:
            if acct is None or client is not acct.client:
                await _close_client(client)
        raise HTTPException(503, f"token auth check could not reach the upstream, retry later: {unchecked[0]}")
    return validated, skipped[0]


def _account_decision(pool: Any, stable_id: str) -> tuple[str, Any]:
    current = _pool_account_by_stable(pool, stable_id)
    if current is None:
        return "add", None
    if getattr(current, "broken", False):
        return "revive", current
    return "skip", current


def _apply_deepseek(
    prepared: list[tuple[str, str, Any, Any]],
    pool: AccountPool | None,
    persisted: set[str],
    store: Any,
    context_store: Any,
    affinity_store: Any,
) -> tuple[AccountPool | None, dict[str, int]]:
    counts = {"added": 0, "reactivated": 0, "activated": 0}
    fresh: list[tuple[str, DeepSeekAccount]] = []
    for token, _state, acct, client in prepared:
        stable = _token_stable_id(token)
        decision, current = _account_decision(pool, stable)
        if decision == "skip":
            if acct is None:
                _close_client_later(client)
            continue
        if decision == "revive":
            current.broken = False
            current.broken_at = None
            counts["reactivated"] += 1
            if acct is not None and current is not acct:
                _close_client_later(acct.client)
            continue
        counts["activated" if token in persisted else "added"] += 1
        fresh.append(
            (
                token,
                DeepSeekAccount(
                    (len(pool.accounts) if pool is not None else 0) + len(fresh),
                    client,
                    session_cache_size=settings.session_cache_size,
                    ttl=settings.session_ttl,
                    store=store,
                    stable_id=stable,
                ),
            )
        )
    if not fresh:
        return pool, counts
    if pool is None:
        pool = AccountPool(
            [account for _token, account in fresh],
            session_cache_size=settings.session_cache_size,
            ttl=settings.session_ttl,
            context_store=context_store,
            affinity_store=affinity_store,
        )
        app.state.pool = pool
        log.info("created deepseek pool with %d account(s)", len(pool.accounts))
    else:
        for _token, account in fresh:
            pool.add_account(account)
        log.info("hot-added deepseek accounts (total accounts: %d)", len(pool.accounts))
    return pool, counts


def _apply_qwen(
    prepared: list[tuple[str, str, Any, Any]],
    pool: AccountPool | None,
    persisted: set[str],
    store: Any,
    context_store: Any,
    affinity_store: Any,
) -> tuple[AccountPool | None, dict[str, int]]:
    counts = {"added": 0, "reactivated": 0, "activated": 0}
    fresh: list[tuple[str, QwenAccount]] = []
    for token, _state, acct, client in prepared:
        stable = _token_stable_id(token)
        decision, current = _account_decision(pool, stable)
        if decision == "skip":
            if acct is None:
                _close_client_later(client)
            continue
        if decision == "revive":
            current.broken = False
            current.broken_at = None
            counts["reactivated"] += 1
            if acct is not None and current is not acct:
                _close_client_later(acct.client)
            continue
        counts["activated" if token in persisted else "added"] += 1
        fresh.append(
            (
                token,
                QwenAccount(
                    (len(pool.accounts) if pool is not None else 0) + len(fresh),
                    client,
                    session_cache_size=settings.session_cache_size,
                    ttl=settings.session_ttl,
                    store=store,
                    stable_id=stable,
                ),
            )
        )
    if not fresh:
        return pool, counts
    if pool is None:
        pool = AccountPool(
            [account for _token, account in fresh],
            label="qwen",
            session_cache_size=settings.session_cache_size,
            ttl=settings.session_ttl,
            context_store=context_store,
            affinity_store=affinity_store,
        )
        app.state.qwen_pool = pool
        log.info("created qwen pool with %d account(s)", len(pool.accounts))
    else:
        for _token, account in fresh:
            pool.add_account(account)
        log.info("hot-added qwen accounts (total accounts: %d)", len(pool.accounts))
    return pool, counts


async def _refresh_models(provider: str, pool: AccountPool) -> None:
    accounts = getattr(pool, "accounts", None) or []
    if not accounts:
        return
    client = accounts[0].client
    try:
        if provider == "qwen":
            app.state.qwen_models = await _fetch_qwen_models(client)
        else:
            await refresh_provider_models(provider, client)
    except Exception as exc:
        log.warning("failed to refresh %s models: %s", provider, exc)


def _token_result(
    added_ds: int,
    added_qw: int,
    skipped_ds: int,
    skipped_qw: int,
    reactivated_ds: int,
    reactivated_qw: int,
    activated_ds: int,
    activated_qw: int,
) -> dict:
    if added_ds or added_qw:
        message = "Tokens added and activated."
    elif reactivated_ds or reactivated_qw or activated_ds or activated_qw:
        message = "Tokens reactivated."
    else:
        message = "No valid tokens to add."
    return {
        "success": True,
        "message": message,
        "added": {"deepseek": added_ds, "qwen": added_qw},
        "skipped": {"deepseek": skipped_ds, "qwen": skipped_qw},
        "reactivated": {"deepseek": reactivated_ds, "qwen": reactivated_qw},
        "activated": {"deepseek": activated_ds, "qwen": activated_qw},
    }


@app.post("/v1/tokens", dependencies=[Depends(_require_admin_token)])
async def add_tokens(tokens: AddTokensRequest) -> dict:
    body = _coerce_tokens(tokens)
    new_ds = _env_token_list(body.deepseek_tokens, "deepseek_tokens")
    new_qw = _env_token_list(body.qwen_tokens, "qwen_tokens")
    if not new_ds and not new_qw:
        raise HTTPException(400, "no tokens provided")

    async with _TOKENS_LOCK:
        existing_ds, existing_qw = await _read_env_tokens()
        plan_ds = _plan_tokens(new_ds, _known_tokens(existing_ds, settings.deepseek_tokens), getattr(app.state, "pool", None))
        plan_qw = _plan_tokens(new_qw, _known_tokens(existing_qw, settings.qwen_tokens), getattr(app.state, "qwen_pool", None))

    ok_ds, skipped_ds = await _validated_tokens(plan_ds, "deepseek", lambda token: DeepSeekClient(token=token, timeout=settings.timeout))
    ok_qw, skipped_qw = await _validated_tokens(plan_qw, "qwen", lambda token: QwenClient(token=token, timeout=settings.timeout))

    async with _TOKENS_LOCK:
        existing_ds, existing_qw = await _read_env_tokens()
        merged_ds = list(dict.fromkeys([*existing_ds, *(token for token, state, _acct, _client in ok_ds if state == _PLAN_NEW)]))
        merged_qw = list(dict.fromkeys([*existing_qw, *(token for token, state, _acct, _client in ok_qw if state == _PLAN_NEW)]))
        if merged_ds != existing_ds or merged_qw != existing_qw:
            await _write_env_tokens(merged_ds, merged_qw)
            settings.deepseek_tokens = merged_ds
            settings.qwen_tokens = merged_qw
        pool, ds_counts = _apply_deepseek(
            ok_ds,
            getattr(app.state, "pool", None),
            set(existing_ds),
            _shared_store("deepseek_session_store", "deepseek-sessions"),
            _shared_store("deepseek_context_store", "deepseek-contexts"),
            _shared_store("deepseek_affinity_store", "deepseek-affinities"),
        )
        qwen_pool, qw_counts = _apply_qwen(
            ok_qw,
            getattr(app.state, "qwen_pool", None),
            set(existing_qw),
            _shared_store("qwen_session_store", "qwen-sessions"),
            _shared_store("qwen_context_store", "qwen-contexts"),
            _shared_store("qwen_affinity_store", "qwen-affinities"),
        )

    if ds_counts["added"] + ds_counts["activated"] and pool is not None:
        await _refresh_models("deepseek", pool)
    if qw_counts["added"] + qw_counts["activated"] and qwen_pool is not None:
        await _refresh_models("qwen", qwen_pool)

    changed = (
        ds_counts["added"],
        qw_counts["added"],
        skipped_ds,
        skipped_qw,
        ds_counts["reactivated"],
        qw_counts["reactivated"],
        ds_counts["activated"],
        qw_counts["activated"],
    )
    if not any(changed):
        raise HTTPException(400, "all provided tokens already exist")
    return _token_result(
        ds_counts["added"],
        qw_counts["added"],
        skipped_ds,
        skipped_qw,
        ds_counts["reactivated"],
        qw_counts["reactivated"],
        ds_counts["activated"],
        qw_counts["activated"],
    )
