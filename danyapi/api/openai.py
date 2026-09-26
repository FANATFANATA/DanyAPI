from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import random
import re
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from .. import tools as toolemu
from ..accounts import AccountPool, AccountPoolBusy, DeepSeekAccount, account_lock
from ..config import settings
from ..deepseek.client import DeepSeekClient, DeepSeekError, DeepSeekSession
from ..deepseek.stream import IncrementalSSE, MessageReconstructor
from ..qwen import api as qwen_api
from ..qwen.accounts import QwenAccount
from ..qwen.client import QwenClient, QwenError
from ..store import JsonStore
from ..tokens import StreamBudget, count_messages_tokens, estimate_tokens
from ..usage import init_tracker, record_usage
from . import responses as responses_api

log = logging.getLogger("danyapi.api")

MODEL_TYPE_BY_NAME = {
    "deepseek-v4.1-flash": "default",
}

REASONING_SUFFIXES = ("-thinking",)

QWEN_DEFAULT_MODELS = [
    {
        "id": "qwen3.8-max",
        "name": "Qwen3.8-Max",
        "owned_by": "qwen",
        "model_type": "chat",
    },
    {
        "id": "qwen3.7-plus",
        "name": "Qwen3.7-Plus",
        "owned_by": "qwen",
        "model_type": "chat",
    },
    {
        "id": "qwen3.7-max",
        "name": "Qwen3.7-Max",
        "owned_by": "qwen",
        "model_type": "chat",
    },
]

STATUS_TO_FINISH_REASON = {
    "FINISHED": "stop",
    "CONTEXT_LENGTH_EXCEEDED": "length",
    "CONTENT_FILTER": "content_filter",
    "INCOMPLETE": "length",
    "WIP": "length",
    "TIMEOUT": "length",
}

CONTEXT_LENGTH_STATUS = "CONTEXT_LENGTH_EXCEEDED"
INPUT_EXCEEDS_LIMIT = "input_exceeds_limit"
CONTINUE_PROMPT = "Continue"
MAX_CONTINUE_ROUNDS = 5
RESPONSE_INCOMPLETE = "response_incomplete"
RESPONSE_INCOMPLETE_MESSAGE = "Response is incomplete: provider errors interrupted the continuation, please retry"
REDUCED_CONTEXT_MESSAGE = "Response was generated from reduced context because the original input exceeded the model limit and may be incomplete"

_TOKENS_LOCK = asyncio.Lock()
SYSTEM_FINGERPRINT = "fp_danyapi"
MODEL_CREATED_AT = int(time.time())


class DeepSeekStreamError(Exception):
    pass


class ChatMessage(BaseModel):
    role: str = "user"
    content: Any = ""
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None
    name: str | None = None

    @field_validator("role")
    @classmethod
    def validate_role(cls, v: str) -> str:
        allowed_roles = {"user", "assistant", "system", "developer", "tool", "function"}
        if v not in allowed_roles:
            raise ValueError(f"Invalid role: {v}. Allowed roles: {allowed_roles}")
        return v

    @field_validator("tool_calls", mode="before")
    @classmethod
    def validate_tool_calls(cls, v: list[Any] | None) -> list[dict[str, Any]] | None:
        if v is None:
            return None
        if not isinstance(v, list):
            raise ValueError("tool_calls must be a list")
        validated_tool_calls = []
        for tool_call in v:
            if not isinstance(tool_call, dict):
                raise ValueError("Each tool_call must be a dictionary")
            if "function" not in tool_call and "name" not in tool_call:
                raise ValueError("Each tool_call must contain 'function' or 'name'")
            validated_tool_calls.append(tool_call)
        return validated_tool_calls


class FileSpec(BaseModel):
    name: str
    content: str
    content_type: str = "application/octet-stream"


class ChatCompletionRequest(BaseModel):
    model: str = Field(default="deepseek-v4.1-flash")
    messages: list[ChatMessage] = Field(default_factory=list)
    stream: bool = False
    temperature: float | None = None
    top_p: float | None = None
    thinking: bool | None = None
    search: bool | None = None
    session_id: str | None = None
    user: str | None = None
    files: list[FileSpec] | None = None
    tools: list[Any] | None = None
    tool_choice: Any = None
    parallel_tool_calls: bool | None = None
    response_format: Any = None
    stream_options: Any = None
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    n: int | None = None
    stop: Any = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    seed: int | None = None
    logit_bias: dict[str, float] | None = None
    logprobs: bool | None = None
    top_logprobs: int | None = None
    modalities: list[str] | None = None
    store: bool | None = None
    metadata: dict[str, Any] | None = None
    functions: list[Any] | None = None
    function_call: Any = None


class CompletionRequest(BaseModel):
    model: str = Field(default="deepseek-v4.1-flash")
    prompt: Any = ""
    suffix: str | None = None
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    n: int | None = None
    stream: bool = False
    stop: Any = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    logit_bias: dict[str, float] | None = None
    user: str | None = None
    session_id: str | None = None


class ImageGenerationRequest(BaseModel):
    model: str = Field(default="qwen-image-gen")
    prompt: str
    n: int = Field(default=1, ge=1, le=4)
    size: str | None = None
    response_format: str = Field(default="url")
    session_id: str | None = None
    user: str | None = None


class ResponsesRequest(BaseModel):
    model: str = Field(default="deepseek-v4.1-flash")
    input: Any = ""
    instructions: str | None = None
    stream: bool = False
    temperature: float | None = None
    top_p: float | None = None
    max_output_tokens: int | None = None
    tools: list[Any] | None = None
    tool_choice: Any = None
    parallel_tool_calls: bool | None = None
    text: Any = None
    response_format: Any = None
    reasoning: Any = None
    previous_response_id: str | None = None
    store: bool = True
    metadata: Any = None
    truncation: str = "disabled"
    user: str | None = None
    session_id: str | None = None
    thinking: bool | None = None
    search: bool | None = None


IMAGE_SIZE_RE = re.compile(r"^(\d{2,5})\s*[*x\u00d7,]\s*(\d{2,5})$", re.IGNORECASE)
MIN_IMAGE_DIM = 16
MAX_IMAGE_DIM = 8192


def _parse_image_size(size: str | None) -> tuple[int, int] | None:
    if size is None or not size.strip():
        return None
    match = IMAGE_SIZE_RE.fullmatch(size.strip())
    if match is None:
        raise HTTPException(400, f"invalid size {size!r}: expected WIDTHxHEIGHT (e.g. 1152x2048 or 1152*2048)")
    width, height = int(match.group(1)), int(match.group(2))
    if not (MIN_IMAGE_DIM <= width <= MAX_IMAGE_DIM and MIN_IMAGE_DIM <= height <= MAX_IMAGE_DIM):
        raise HTTPException(400, f"size out of range: both dimensions must be within {MIN_IMAGE_DIM}..{MAX_IMAGE_DIM}")
    return width, height


def _resize_image_bytes(content: bytes, dims: tuple[int, int] | None) -> bytes:
    if dims is None:
        return content
    try:
        from io import BytesIO

        from PIL import Image

        with Image.open(BytesIO(content)) as img:
            fmt = img.format or "PNG"
            resized = img.resize(dims, Image.Resampling.LANCZOS)
            if fmt.upper() == "JPEG" and resized.mode not in ("RGB", "L"):
                resized = resized.convert("RGB")
            buffer = BytesIO()
            resized.save(buffer, format=fmt)
            return buffer.getvalue()
    except Exception as exc:
        log.warning("image resize to %s failed, returning original: %s", dims, exc)
        return content


def _token_stable_id(token: str) -> str:
    return hashlib.sha1(token.encode("utf-8"), usedforsecurity=False).hexdigest()[:16]


_STATE_STORE_ATTRS = (
    "deepseek_session_store",
    "qwen_session_store",
    "deepseek_context_store",
    "qwen_context_store",
    "deepseek_affinity_store",
    "qwen_affinity_store",
    "responses_store",
)


def _flush_state_stores() -> None:
    tracker = getattr(app.state, "usage", None)
    if tracker is not None:
        try:
            tracker.flush()
        except Exception as exc:
            log.debug("usage flush failed: %s", exc)
    for attr in _STATE_STORE_ATTRS:
        store = getattr(app.state, attr, None)
        if store is None:
            continue
        try:
            store.flush()
        except Exception as exc:
            log.debug("store flush failed for %s: %s", attr, exc)
    pools: list[Any] = []
    for attr in ("pool", "qwen_pool"):
        pool_obj = getattr(app.state, attr, None)
        if pool_obj is not None:
            pools.append(pool_obj)
    byok_pools = getattr(app.state, "byok_pools", None)
    if isinstance(byok_pools, dict):
        for cache in byok_pools.values():
            if isinstance(cache, dict):
                pools.extend(cache.values())
    seen_pools: set[int] = set()
    for pool_obj in pools:
        if id(pool_obj) in seen_pools:
            continue
        seen_pools.add(id(pool_obj))
        flush = getattr(pool_obj, "flush", None)
        if flush is None:
            continue
        try:
            flush()
        except Exception as exc:
            log.debug("pool flush failed: %s", exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    accounts: list[DeepSeekAccount] = []
    qwen_accounts: list[QwenAccount] = []
    byok_mode = settings.byok
    app.state.byok = byok_mode
    app.state.byok_pools = {"deepseek": {}, "qwen": {}}
    app.state.byok_locks = {"deepseek": asyncio.Lock(), "qwen": asyncio.Lock()}
    app.state.byok_auth = {"deepseek": {}, "qwen": {}}
    cache_enabled = settings.cache_enabled
    deepseek_session_store = JsonStore("deepseek-sessions", "default" if cache_enabled else None)
    qwen_session_store = JsonStore("qwen-sessions", "default" if cache_enabled else None)
    deepseek_context_store = JsonStore("deepseek-contexts", "default" if cache_enabled else None)
    qwen_context_store = JsonStore("qwen-contexts", "default" if cache_enabled else None)
    deepseek_affinity_store = JsonStore("deepseek-affinities", "default" if cache_enabled else None)
    qwen_affinity_store = JsonStore("qwen-affinities", "default" if cache_enabled else None)
    responses_store = JsonStore("responses", "default" if cache_enabled else None, maxsize=settings.responses_max_records)
    app.state.responses_store = responses_store
    app.state.deepseek_session_store = deepseek_session_store
    app.state.qwen_session_store = qwen_session_store
    app.state.deepseek_context_store = deepseek_context_store
    app.state.qwen_context_store = qwen_context_store
    app.state.deepseek_affinity_store = deepseek_affinity_store
    app.state.qwen_affinity_store = qwen_affinity_store
    if settings.usage_enabled:
        app.state.usage = init_tracker(store=JsonStore("usage", "default"), max_records=settings.usage_max_records)
    else:
        app.state.usage = None
    try:
        if not byok_mode:
            ds_clients = [DeepSeekClient(token=token, timeout=settings.timeout) for token in settings.deepseek_tokens] if settings.deepseek_tokens else []
            qw_clients = [QwenClient(token=token, timeout=settings.timeout) for token in settings.qwen_tokens] if settings.qwen_tokens else []
            ds_checks = [client.check_auth() for client in ds_clients]
            qw_checks = [client.check_auth() for client in qw_clients]
            if ds_checks or qw_checks:
                auth_results = await asyncio.gather(*(ds_checks + qw_checks))
                ds_auth = auth_results[: len(ds_checks)]
                qw_auth = auth_results[len(ds_checks) :]
            else:
                ds_auth = []
                qw_auth = []
            if settings.deepseek_tokens:
                for i, (token, ds_client, ok) in enumerate(zip(settings.deepseek_tokens, ds_clients, ds_auth, strict=True)):
                    if not ok:
                        log.warning("deepseek token #%d invalid/expired, skipping", i)
                        await ds_client.aclose()
                        continue
                    accounts.append(
                        DeepSeekAccount(
                            len(accounts),
                            ds_client,
                            session_cache_size=settings.session_cache_size,
                            ttl=settings.session_ttl,
                            store=deepseek_session_store,
                            stable_id=_token_stable_id(token),
                        )
                    )
                log.info("deepseek accounts ready: %d", len(accounts))
            if settings.qwen_tokens:
                for i, (token, qw_client, ok) in enumerate(zip(settings.qwen_tokens, qw_clients, qw_auth, strict=True)):
                    if not ok:
                        log.warning("qwen token #%d invalid/expired, skipping", i)
                        await qw_client.aclose()
                        continue
                    qwen_accounts.append(
                        QwenAccount(
                            len(qwen_accounts),
                            qw_client,
                            session_cache_size=settings.session_cache_size,
                            ttl=settings.session_ttl,
                            store=qwen_session_store,
                            stable_id=_token_stable_id(token),
                        )
                    )
                log.info("qwen accounts ready: %d", len(qwen_accounts))
        if accounts:
            app.state.pool = AccountPool(
                accounts,
                session_cache_size=settings.session_cache_size,
                ttl=settings.session_ttl,
                context_store=deepseek_context_store,
                affinity_store=deepseek_affinity_store,
            )
        else:
            app.state.pool = None
        if qwen_accounts:
            app.state.qwen_pool = AccountPool(
                qwen_accounts,
                label="qwen",
                session_cache_size=settings.session_cache_size,
                ttl=settings.session_ttl,
                context_store=qwen_context_store,
                affinity_store=qwen_affinity_store,
            )
            app.state.qwen_models = await _fetch_qwen_models(qwen_accounts[0].client)
        else:
            app.state.qwen_pool = None
            app.state.qwen_models = []
        if not accounts and not qwen_accounts and not byok_mode:
            raise RuntimeError("no valid credentials: set DEEPSEEK_TOKENS or QWEN_TOKENS")
        yield
    finally:
        http_client = getattr(app.state, "http_client", None)
        if http_client is not None:
            await http_client.aclose()
        _flush_state_stores()
        seen: set[int] = set()
        clients = [acct.client for acct in accounts] + [acct.client for acct in qwen_accounts]
        for pool_obj in (getattr(app.state, "pool", None), getattr(app.state, "qwen_pool", None)):
            if pool_obj is not None:
                clients.extend(acct.client for acct in pool_obj.accounts)
        byok_pools = getattr(app.state, "byok_pools", None)
        if byok_pools is not None:
            for pool_obj in list(byok_pools["deepseek"].values()) + list(byok_pools["qwen"].values()):
                clients.extend(acct.client for acct in pool_obj.accounts)
        for client in clients:
            if id(client) in seen:
                continue
            seen.add(id(client))
            await client.aclose()


async def _fetch_qwen_models(client: QwenClient) -> list[dict]:
    try:
        raw = await client.fetch_models()
    except QwenError as exc:
        log.warning("qwen models fetch failed, using defaults: %s", exc)
        return QWEN_DEFAULT_MODELS
    models = []
    for model in raw:
        if not isinstance(model, dict) or not model.get("id"):
            continue
        info = model.get("info")
        meta = (info or {}).get("meta") if isinstance(info, dict) else None
        chat_types = (meta or {}).get("chat_type") if isinstance(meta, dict) else None
        chat_types = chat_types or []
        if "t2t" in chat_types:
            model_type = "chat"
        elif "t2i" in chat_types:
            model_type = "image"
        elif "t2v" in chat_types:
            model_type = "video"
        else:
            model_type = "chat"
        models.append(
            {
                "id": model["id"],
                "name": model.get("name") or model["id"],
                "owned_by": "qwen",
                "model_type": model_type,
                "chat_types": chat_types,
            }
        )
    if not models:
        log.warning("qwen models fetch returned nothing, using defaults")
        return QWEN_DEFAULT_MODELS
    return models


app = FastAPI(title="DanyAPI", lifespan=lifespan)

cors_origins = settings.cors_origins or ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=bool(settings.cors_origins),
    allow_methods=["*"],
    allow_headers=["*"],
)

docs_path = Path(__file__).resolve().parents[2] / "docs"
if docs_path.is_dir():
    app.mount("/docs", StaticFiles(directory=str(docs_path), html=True), name="docs")


@dataclass
class _RootContext:
    html: str | None = None
    checked: bool = False


_root_ctx = _RootContext()


@app.get("/", response_class=HTMLResponse)
async def root():
    if not _root_ctx.checked:
        web_path = Path(__file__).resolve().parents[2] / "web" / "index.html"
        if web_path.exists():
            _root_ctx.html = await asyncio.to_thread(web_path.read_text, encoding="utf-8")
        else:
            _root_ctx.html = None
        _root_ctx.checked = True
    if _root_ctx.html is not None:
        return _root_ctx.html
    return HTMLResponse("<h1>DanyAPI</h1><p>Web interface not found</p>", status_code=404)


@app.get("/favicon.ico")
async def favicon():
    return Response(status_code=204)


def _env_path() -> Path:
    return Path(__file__).resolve().parents[2] / ".env"


def _unquote_env_value(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
        return value[1:-1]
    return value


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
    env_file.write_text("\n".join(new_lines) + "\n", encoding="utf-8")


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


def _shared_store(attr: str, name: str, *, maxsize: int = 0) -> JsonStore:
    store = getattr(app.state, attr, None)
    if store is None:
        store = JsonStore(name, "default" if settings.cache_enabled else None, maxsize=maxsize)
        setattr(app.state, attr, store)
    return store


def _responses_store() -> JsonStore:
    return _shared_store("responses_store", "responses", maxsize=settings.responses_max_records)


def _pool_account_by_stable(pool: AccountPool | None, stable_id: str) -> Any | None:
    if pool is None:
        return None
    for acct in pool.accounts:
        if getattr(acct, "stable_id", None) == stable_id:
            return acct
    return None


@app.post("/v1/tokens")
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

        parts = []
        if added_ds:
            parts.append(f"deepseek: +{added_ds}")
        if added_qw:
            parts.append(f"qwen: +{added_qw}")
        if skipped_ds:
            parts.append(f"deepseek skipped: {skipped_ds}")
        if skipped_qw:
            parts.append(f"qwen skipped: {skipped_qw}")

        return {
            "success": True,
            "message": "Tokens added and activated." if (added_ds or added_qw) else "No valid tokens to add.",
            "added": {"deepseek": added_ds, "qwen": added_qw},
            "skipped": {"deepseek": skipped_ds, "qwen": skipped_qw},
            "reactivated": {"deepseek": activated_ds, "qwen": activated_qw},
        }


MAX_LOGGED_BODY = 256 * 1024
MAX_REQUEST_BODY = 100 * 1024 * 1024


async def _read_request_body(request: Request, limit: int) -> bytes:
    raw_length = -1
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            raw_length = int(content_length)
        except ValueError:
            raw_length = -1
    if raw_length > 0:
        if raw_length > limit:
            raise HTTPException(413, "request body too large")
        body = await request.body()
        if len(body) > limit:
            raise HTTPException(413, "request body too large")
        request._body = body
        return body
    cached = getattr(request, "_body", None)
    if cached:
        if len(cached) > limit:
            raise HTTPException(413, "request body too large")
        return cached
    chunks: list[bytes] = []
    total = 0
    async for chunk in request.stream():
        total += len(chunk)
        if total > limit:
            raise HTTPException(413, "request body too large")
        chunks.append(chunk)
    body = b"".join(chunks)
    request._body = body
    return body


def _parse_logged_body(body: bytes) -> dict[str, Any]:
    if not body or len(body) > MAX_LOGGED_BODY:
        return {}
    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError, TypeError):
        return {}
    if isinstance(payload, dict):
        return payload
    return {}


async def _extract_request_body(request: Request) -> dict[str, Any]:
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            raw_length = int(content_length)
            if raw_length <= 0:
                return {}
            if raw_length > MAX_LOGGED_BODY:
                return {}
        except ValueError:
            return {}
    if getattr(request, "method", None) in ("GET", "DELETE", "HEAD", "OPTIONS"):
        return {}
    cached = getattr(request, "_body", b"")
    if cached:
        return _parse_logged_body(cached)
    if content_length is None:
        return {}
    try:
        body = await _read_request_body(request, MAX_REQUEST_BODY)
    except HTTPException:
        raise
    except Exception:
        return {}
    return _parse_logged_body(body)


def _request_client_ip(request: Request) -> str:
    headers = request.headers
    forwarded = headers.get("x-forwarded-for")
    if forwarded:
        first = forwarded.split(",", 1)[0].strip()
        if first:
            return first
    real_ip = headers.get("x-real-ip")
    if real_ip:
        return real_ip.strip()
    client = request.client
    if client is not None and client.host:
        return client.host
    return "-"


def _request_details(request: Request, payload: dict[str, Any], count_tokens: bool = True) -> str:
    parts = []
    user_agent = request.headers.get("user-agent")
    if user_agent:
        parts.append(f"ua={user_agent[:120].replace('{', '{{').replace('}', '}}')}")
    model = payload.get("model")
    if isinstance(model, str) and model:
        parts.append(f"model={model}")
    session_id = payload.get("session_id")
    if isinstance(session_id, str) and session_id:
        parts.append(f"sid={session_id}")
    user = payload.get("user")
    if isinstance(user, str) and user:
        parts.append(f"user={user}")
    stream = payload.get("stream")
    if isinstance(stream, bool):
        parts.append(f"stream={int(stream)}")
    messages = payload.get("messages")
    if isinstance(messages, list):
        parts.append(f"msgs={len(messages)}")
        if count_tokens:
            parts.append(f"tokens={count_messages_tokens(messages)}")
    return " ".join(parts)


def _log_request_failure(request: Request, payload: dict[str, Any], duration: float, status: int | None = None, exc: Exception | None = None) -> None:
    if not log.isEnabledFor(logging.WARNING):
        return
    details = _request_details(request, payload, count_tokens=log.isEnabledFor(logging.DEBUG))
    details_part = f" {details}" if details else ""
    ip = _request_client_ip(request)
    if status is not None:
        reason = f"status={status}"
    else:
        reason = f"error={str(exc) if exc else 'unknown'}"
    log.warning(
        "%s %s %s%s failed: %s (%.0fms)",
        request.method,
        request.url.path,
        ip,
        details_part,
        reason,
        duration,
    )


def _log_request_success(request: Request, payload: dict[str, Any], duration: float) -> None:
    if not log.isEnabledFor(logging.INFO):
        return
    details = _request_details(request, payload, count_tokens=log.isEnabledFor(logging.DEBUG))
    details_part = f" {details}" if details else ""
    ip = _request_client_ip(request)
    log.info(
        "%s %s %s%s ok (%.0fms)",
        request.method,
        request.url.path,
        ip,
        details_part,
        duration,
    )


@app.middleware("http")
async def _log_requests(request: Request, call_next):
    started = time.monotonic()
    payload: dict[str, Any] = {}
    if log.isEnabledFor(logging.INFO) or log.isEnabledFor(logging.WARNING):
        payload = await _extract_request_body(request)
    try:
        response = await call_next(request)
    except Exception as exc:
        _log_request_failure(
            request,
            payload,
            (time.monotonic() - started) * 1000,
            exc=exc,
        )
        raise
    duration = (time.monotonic() - started) * 1000
    if response.status_code >= 400:
        _log_request_failure(
            request,
            payload,
            duration,
            status=response.status_code,
        )
    else:
        _log_request_success(request, payload, duration)
    return response


def _error_type_for_status(status: int) -> str:
    if status == 401:
        return "authentication_error"
    if status == 403:
        return "permission_error"
    if status == 404:
        return "not_found_error"
    if status == 408:
        return "request_timeout"
    if status == 409:
        return "conflict_error"
    if status == 413:
        return "request_too_large"
    if status == 429:
        return "rate_limit_error"
    if status == 501 or status == 503:
        return "api_error"
    if status == 502 or status == 504:
        return "server_error"
    if status >= 500:
        return "server_error"
    return "invalid_request_error"


def _error_code_for_status(status: int) -> str | None:
    if status == 429:
        return "rate_limit_exceeded"
    if status == 400:
        return "invalid_request_error"
    return None


def _exception_message(exc: Exception) -> str:
    text = str(exc).strip()
    if not text:
        return "An unexpected error occurred"
    return text


def _openai_error_payload(status: int, message: str, request_id: str | None = None) -> dict:
    payload = {
        "error": {
            "message": message,
            "type": _error_type_for_status(status),
            "param": None,
            "code": _error_code_for_status(status),
        }
    }
    if request_id:
        payload["error"]["request_id"] = request_id
    return payload


def _request_id_header(request: Request) -> str:
    provided = request.headers.get("x-request-id")
    if provided and len(provided) <= 128:
        return provided
    return uuid.uuid4().hex


@app.exception_handler(RequestValidationError)
async def _on_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    request_id = _request_id_header(request)
    return JSONResponse(
        status_code=400,
        content=_openai_error_payload(400, f"invalid request body: {exc.errors()}", request_id),
        headers={"x-request-id": request_id},
    )


@app.exception_handler(HTTPException)
async def _on_http_exception(request: Request, exc: HTTPException) -> JSONResponse:
    request_id = _request_id_header(request)
    headers = {"x-request-id": request_id}
    if exc.headers:
        headers.update({str(k): str(v) for k, v in exc.headers.items()})
    return JSONResponse(
        status_code=exc.status_code,
        content=_openai_error_payload(exc.status_code, str(exc.detail), request_id),
        headers=headers,
    )


@app.exception_handler(Exception)
async def _on_uncaught_exception(request: Request, exc: Exception) -> JSONResponse:
    request_id = _request_id_header(request)
    return JSONResponse(
        status_code=500,
        content=_openai_error_payload(500, _exception_message(exc), request_id),
        headers={"x-request-id": request_id},
    )


def _account_busy_count(pool: Any) -> int:
    busy = 0
    for acct in getattr(pool, "accounts", None) or []:
        sem = getattr(acct, "sem", None)
        if sem is not None and sem.locked():
            busy += 1
    return busy


_POOL_RATE_CACHE: dict[int, tuple[float, dict[str, str]]] = {}
_POOL_RATE_TTL = 1.0


def _pool_rate_headers(pool: Any | None) -> dict[str, str]:
    if pool is None or not hasattr(pool, "stats"):
        return {}
    now = time.monotonic()
    entry = _POOL_RATE_CACHE.get(id(pool))
    if entry is not None and now - entry[0] < _POOL_RATE_TTL:
        return entry[1]
    stats = pool.stats()
    total = int(stats.get("healthy", 0) or 0)
    busy = _account_busy_count(pool)
    headers = {
        "x-ratelimit-limit-requests": str(max(total, 0)),
        "x-ratelimit-remaining-requests": str(max(total - busy, 0)),
        "x-ratelimit-reset-requests": str(int(time.time())),
    }
    if len(_POOL_RATE_CACHE) > 16:
        _POOL_RATE_CACHE.clear()
    _POOL_RATE_CACHE[id(pool)] = (now, headers)
    return headers


@app.middleware("http")
async def _openai_headers(request: Request, call_next):
    request_id = _request_id_header(request)
    response = await call_next(request)
    headers = response.headers
    if not headers.get("x-request-id"):
        headers["x-request-id"] = request_id
    if not headers.get("x-ratelimit-limit-requests"):
        for candidate in (getattr(app.state, "pool", None), getattr(app.state, "qwen_pool", None)):
            if candidate is not None:
                for key, value in _pool_rate_headers(candidate).items():
                    headers[key] = value
                break
    return response


MAX_FILES_PER_REQUEST = 50
MAX_FILE_SIZE = 100 * 1024 * 1024
MAX_ATTACHMENT_TOTAL_SIZE = 10 * 1024 * 1024


def _data_uri_parts(uri: str) -> tuple[str, str]:
    if not uri.startswith("data:"):
        raise HTTPException(400, "image_url must be a data URI (data:<mime>;base64,...)")
    meta, _, payload = uri[5:].partition(",")
    if not payload:
        raise HTTPException(400, "invalid data URI: missing base64 payload")
    return meta, "".join(payload.split())


def _compact_data_uri_length(compact: str) -> int:
    stripped = compact.rstrip("=")
    units, remainder = divmod(len(stripped), 4)
    decoded = units * 3
    if remainder == 2:
        decoded += 1
    elif remainder == 3:
        decoded += 2
    return decoded


def _raw_data_uri_length(uri: str) -> int:
    _meta, compact = _data_uri_parts(uri)
    return _compact_data_uri_length(compact)


@dataclass
class Attachment:
    data: bytes
    name: str
    content_type: str
    is_image: bool


def _split_data_uri(uri: str) -> tuple[str, bytes]:
    meta, compact = _data_uri_parts(uri)
    content_type = meta.split(";", 1)[0] or "application/octet-stream"
    try:
        data = base64.b64decode(compact, validate=True)
    except ValueError as exc:
        raise HTTPException(400, "invalid base64 in image_url") from exc
    return content_type, data


def _collect_attachments(req: ChatCompletionRequest) -> list[Attachment]:
    attachments: list[Attachment] = []
    raw_total = 0
    for msg in req.messages:
        if not isinstance(msg.content, list):
            continue
        for item in msg.content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "image_url":
                image_url = item.get("image_url")
                if isinstance(image_url, str):
                    uri = image_url
                elif isinstance(image_url, dict) and isinstance(image_url.get("url"), str):
                    uri = image_url["url"]
                else:
                    raise HTTPException(400, "invalid image_url value")
                meta, compact = _data_uri_parts(uri)
                raw_total += _compact_data_uri_length(compact)
                if raw_total > MAX_ATTACHMENT_TOTAL_SIZE:
                    raise HTTPException(413, "attachments too large")
                content_type = meta.split(";", 1)[0] or "application/octet-stream"
                try:
                    data = base64.b64decode(compact, validate=True)
                except ValueError as exc:
                    raise HTTPException(400, "invalid base64 in image_url") from exc
                name = f"image_{len(attachments)}.{content_type.split('/')[-1] or 'bin'}"
                attachments.append(Attachment(data, name, content_type, True))
    for f in req.files or []:
        if not f.name or not f.content:
            raise HTTPException(400, "each file needs name and base64 content")
        try:
            data = base64.b64decode(f.content)
        except ValueError as exc:
            raise HTTPException(400, f"invalid base64 in file {f.name}") from exc
        attachments.append(Attachment(data, f.name, f.content_type or "application/octet-stream", (f.content_type or "").startswith("image/")))
    return attachments


def _validate_attachments(attachments: list[Attachment]) -> None:
    if not attachments:
        return
    if len(attachments) > MAX_FILES_PER_REQUEST:
        raise HTTPException(400, f"too many files: max {MAX_FILES_PER_REQUEST} per request")
    for att in attachments:
        if len(att.data) > MAX_FILE_SIZE:
            raise HTTPException(400, f"file {att.name} exceeds {MAX_FILE_SIZE // (1024 * 1024)} MB limit")


async def _fresh_pow_upload_headers(account) -> dict:
    try:
        return await account.pow_upload.make_header(lambda: account.client.create_pow_challenge("/api/v0/file/upload_file"))
    except DeepSeekError as exc:
        _handle_account_error(account, exc)
        raise HTTPException(_deepseek_status(exc), _deepseek_error_detail(exc)) from exc


async def _upload_attachments(account, attachments: list[Attachment], model_type: str, thinking: bool) -> list[str]:
    file_ids: list[str] = []
    if attachments:
        pow_headers_list = await asyncio.gather(*(_fresh_pow_upload_headers(account) for _ in attachments))
    else:
        pow_headers_list = []
    if not pow_headers_list:
        return file_ids
    sem = asyncio.Semaphore(4)

    async def _upload_one(att: Attachment, pow_headers) -> str:
        async with sem:
            try:
                info = await account.client.upload_file(
                    att.data,
                    att.name,
                    att.content_type,
                    model_type,
                    thinking_enabled=thinking,
                    pow_headers=pow_headers,
                )
            except DeepSeekError as exc:
                _handle_account_error(account, exc)
                raise HTTPException(_deepseek_status(exc), f"file upload failed: {exc}") from exc
            file_id = info.get("id")
            if not file_id:
                raise HTTPException(502, f"file upload failed for {att.name}: no file id")
            return file_id

    results = await asyncio.gather(
        *(_upload_one(att, pow_headers) for att, pow_headers in zip(attachments, pow_headers_list, strict=True)),
        return_exceptions=True,
    )
    for item in results:
        if isinstance(item, BaseException):
            raise item
        file_ids.append(item)
    return file_ids


def _resolve_model(model: str) -> str:
    model_type = MODEL_TYPE_BY_NAME.get(model)
    if model_type is not None:
        return model_type
    for suffix in REASONING_SUFFIXES:
        if not model.endswith(suffix):
            continue
        base_type = MODEL_TYPE_BY_NAME.get(model[: -len(suffix)])
        if base_type is not None:
            return base_type
        break
    raise HTTPException(404, f"Unknown model: {model}")


def _is_reasoning_model(model: str) -> bool:
    return any(model.endswith(suffix) for suffix in REASONING_SUFFIXES)


def _finish_reason(status: Any) -> str:
    if isinstance(status, str):
        return STATUS_TO_FINISH_REASON.get(status, "stop")
    return "stop"


def _pool_stats(pool) -> dict | None:
    if pool is None:
        return None
    try:
        return pool.stats()
    except Exception:
        return None


def _usage_summary() -> dict | None:
    tracker = getattr(app.state, "usage", None)
    if tracker is None:
        return None
    try:
        return tracker.snapshot()["totals"]
    except Exception:
        return None


@app.get("/health")
async def health() -> dict:
    pool = getattr(app.state, "pool", None)
    qwen_pool = getattr(app.state, "qwen_pool", None)
    result = {
        "status": "ok",
        "deepseek": pool is not None,
        "qwen": qwen_pool is not None,
        "deepseek_stats": _pool_stats(pool),
        "qwen_stats": _pool_stats(qwen_pool),
        "usage": _usage_summary(),
    }
    if _byok_mode():
        byok_pools = await _byok_pools_state()
        result["byok"] = True
        result["byok_pools"] = {
            "deepseek": len(byok_pools["deepseek"]),
            "qwen": len(byok_pools["qwen"]),
        }
    return result


@app.get("/v1/usage")
async def usage_stats() -> dict:
    tracker = getattr(app.state, "usage", None)
    if tracker is None:
        raise HTTPException(404, "usage tracking is disabled")
    return tracker.snapshot()


_MODEL_CACHE: dict[str, Any] = {"key": None, "models": None, "index": None, "qwen_ids": None}


def _model_source() -> list[dict]:
    qwen_models = getattr(app.state, "qwen_models", None) or []
    if not qwen_models and _byok_mode():
        return QWEN_DEFAULT_MODELS
    return qwen_models


def _model_cache_key() -> tuple[tuple[Any, ...], ...]:
    return tuple((m.get("id"), m.get("name"), m.get("owned_by"), m.get("model_type")) for m in _model_source())


def _models_state() -> list[dict]:
    key = _model_cache_key()
    cached = _MODEL_CACHE
    if cached["key"] == key and cached["models"] is not None:
        return cached["models"]
    models: list[dict] = []
    for name, model_type in MODEL_TYPE_BY_NAME.items():
        models.append(
            {
                "id": name,
                "object": "model",
                "created": MODEL_CREATED_AT,
                "owned_by": "deepseek",
                "model_type": model_type,
            }
        )
        for suffix in REASONING_SUFFIXES:
            models.append(
                {
                    "id": f"{name}{suffix}",
                    "object": "model",
                    "created": MODEL_CREATED_AT,
                    "owned_by": "deepseek",
                    "model_type": model_type,
                }
            )
    for model in _model_source():
        models.append(
            {
                "id": model["id"],
                "object": "model",
                "created": MODEL_CREATED_AT,
                "owned_by": model.get("owned_by", "qwen"),
                "name": model.get("name"),
                "model_type": model.get("model_type", "chat"),
            }
        )
    cached["key"] = key
    cached["models"] = models
    cached["index"] = {m["id"]: m for m in models}
    cached["qwen_ids"] = {m.get("id") for m in _model_source()}
    return models


def _all_models() -> list[dict]:
    return _models_state()


@app.get("/v1/models")
async def list_models() -> dict:
    return {"object": "list", "data": _models_state()}


@app.get("/v1/models/{model_id}")
async def get_model(model_id: str) -> dict:
    _models_state()
    model = _MODEL_CACHE["index"].get(model_id)
    if model is None:
        raise HTTPException(404, f"The model '{model_id}' does not exist")
    return model


RETRYABLE_FINISH_REASONS = {
    "expert_busy_use_default",
    "parallel_chat_limit",
    "server_busy",
    "busy",
}
MAX_RETRIES = 5
RETRY_BACKOFF_SEC = 1.0
RETRY_BACKOFF_MAX_SEC = 8.0

DEEPSEEK_AUTH_ERROR_CODES = {40001, 40002, 40003, 40012, 40029}

MESSAGE_TOO_FREQUENT_MARKERS = ("messagetoofrequent", "messagetofrequent")
MESSAGE_TOO_FREQUENT_WAIT_SEC = 60.0
MESSAGE_TOO_FREQUENT_MAX_RETRIES = 5


async def _human_delay() -> None:
    delay = random.uniform(settings.human_delay_min, settings.human_delay_max)
    if delay > 0:
        await asyncio.sleep(delay)


def _resolve_provider(model: str) -> str:
    if model.startswith("qwen"):
        return "qwen"
    if model in MODEL_TYPE_BY_NAME or model.startswith("deepseek"):
        return "deepseek"
    _models_state()
    qwen_ids = _MODEL_CACHE.get("qwen_ids")
    if isinstance(qwen_ids, set) and model in qwen_ids:
        return "qwen"
    raise HTTPException(404, f"Unknown model: {model}")


BYOK_POOL_LIMIT = 512
BYOK_AUTH_LIMIT = 4096


def _byok_mode() -> bool:
    return bool(getattr(app.state, "byok", False))


async def _byok_pools_state() -> dict[str, dict[str, Any]]:
    pools = getattr(app.state, "byok_pools", None)
    if pools is None:
        pools = {"deepseek": {}, "qwen": {}}
        app.state.byok_pools = pools
    return pools


async def _byok_locks_state() -> dict[str, asyncio.Lock]:
    locks = getattr(app.state, "byok_locks", None)
    if locks is None:
        locks = {"deepseek": asyncio.Lock(), "qwen": asyncio.Lock()}
        app.state.byok_locks = locks
    return locks


async def _byok_auth_state() -> dict[str, dict[str, Any]]:
    auth = getattr(app.state, "byok_auth", None)
    if auth is None:
        auth = {"deepseek": {}, "qwen": {}}
        app.state.byok_auth = auth
    return auth


def _cached_auth(store: dict[str, Any], stable: str, ttl: float, now: float) -> bool | None:
    if ttl <= 0:
        return None
    record = store.get(stable)
    if not isinstance(record, (list, tuple)) or len(record) != 2:
        return None
    try:
        ts = float(record[1])
    except (TypeError, ValueError):
        return None
    if now - ts > ttl:
        return None
    return bool(record[0])


def _evict_auth(store: dict[str, Any]) -> None:
    while len(store) > BYOK_AUTH_LIMIT:
        store.pop(next(iter(store)), None)


async def _extract_request_api_key(request: Request) -> str | None:
    auth = request.headers.get("authorization") or ""
    if auth.lower().startswith("bearer "):
        key = auth[7:].strip()
        if key:
            return key
    key = (request.headers.get("x-api-key") or "").strip()
    if key:
        return key
    content_type = request.headers.get("content-type") or ""
    if not content_type.startswith("application/json"):
        return None
    try:
        body = await _read_request_body(request, MAX_REQUEST_BODY)
    except HTTPException:
        raise
    except Exception:
        return None
    if not body:
        return None
    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError, TypeError):
        return None
    if isinstance(payload, dict):
        api_key = payload.get("api_key")
        if isinstance(api_key, str) and api_key.strip():
            return api_key.strip()
    return None


_deferred_close_tasks: set[asyncio.Task] = set()


async def _close_pool(pool: Any) -> None:
    for acct in pool.accounts:
        try:
            acct.sessions.close_all()
        except Exception as exc:
            log.info("session cleanup failed for byok account %r: %s", getattr(acct, "label", acct), exc)
    flush = getattr(pool, "flush", None)
    if flush is not None:
        try:
            flush()
        except Exception as exc:
            log.info("pool store flush failed: %s", exc)
    for acct in pool.accounts:
        sem = getattr(acct, "sem", None)
        if sem is not None and sem.locked():
            log.info("schedule deferred client close for busy byok account %r", getattr(acct, "label", acct))
            task = asyncio.create_task(_close_busy_client(acct, sem))
            _deferred_close_tasks.add(task)
            task.add_done_callback(_deferred_close_tasks.discard)
            continue
        try:
            await acct.client.aclose()
        except Exception as exc:
            log.info("client close failed for byok account %r: %s", getattr(acct, "label", acct), exc)


async def _close_busy_client(account: Any, sem: asyncio.Semaphore) -> None:
    acquired = False
    try:
        await asyncio.wait_for(sem.acquire(), timeout=300)
        acquired = True
    except (TimeoutError, asyncio.TimeoutError, asyncio.CancelledError):
        log.info("give up deferred client close for busy byok account %r", getattr(account, "label", account))
        return
    try:
        await account.client.aclose()
    except Exception as exc:
        log.info("client close failed for byok account %r: %s", getattr(account, "label", account), exc)
    finally:
        if acquired:
            sem.release()


async def _byok_validate(
    provider: str,
    token: str,
    client: Any,
) -> bool:
    auth = await _byok_auth_state()
    store = auth[provider]
    stable = _token_stable_id(token)
    ttl = settings.byok_auth_ttl
    cached = _cached_auth(store, stable, ttl, time.monotonic())
    if cached is not None:
        return cached
    try:
        ok = bool(await client.check_auth())
    except Exception:
        ok = False
    store.pop(stable, None)
    store[stable] = [ok, time.monotonic()]
    _evict_auth(store)
    return ok


def _byok_cache_key(tokens: list[str]) -> str:
    return "|".join(sorted(_token_stable_id(token) for token in tokens))


async def _byok_pool(provider: str, tokens: list[str]) -> AccountPool:
    if provider not in ("deepseek", "qwen"):
        raise HTTPException(400, f"unknown provider: {provider}")
    tokens = list(dict.fromkeys(tokens))
    pools = await _byok_pools_state()
    cache = pools[provider]
    cache_key = _byok_cache_key(tokens)
    pool = cache.get(cache_key)
    if pool is not None and pool.healthy:
        cache.pop(cache_key)
        cache[cache_key] = pool
        return pool
    locks = await _byok_locks_state()
    async with locks[provider]:
        pool = cache.get(cache_key)
        if pool is not None and pool.healthy:
            cache.pop(cache_key)
            cache[cache_key] = pool
            return pool
        if pool is not None:
            cache.pop(cache_key, None)
            await _close_pool(pool)
        scope = ("byok-" + _token_stable_id(cache_key)) if settings.cache_enabled else None
        if provider == "deepseek":
            session_store = JsonStore("deepseek-sessions", scope) if settings.cache_enabled else None
            context_store = JsonStore("deepseek-contexts", scope) if settings.cache_enabled else None
            affinity_store = JsonStore("deepseek-affinities", scope) if settings.cache_enabled else None
            accounts: list[DeepSeekAccount] = []
            for i, token in enumerate(tokens):
                ds_client = DeepSeekClient(token=token, timeout=settings.timeout)
                if not await _byok_validate("deepseek", token, ds_client):
                    log.warning("byok deepseek token invalid/expired, skipping")
                    await ds_client.aclose()
                    continue
                accounts.append(
                    DeepSeekAccount(
                        i,
                        ds_client,
                        session_cache_size=settings.session_cache_size,
                        ttl=settings.session_ttl,
                        store=session_store,
                        stable_id=_token_stable_id(token),
                    )
                )
            if not accounts:
                raise HTTPException(401, "invalid deepseek api key")
            pool = AccountPool(
                accounts,
                session_cache_size=settings.session_cache_size,
                ttl=settings.session_ttl,
                context_store=context_store,
                affinity_store=affinity_store,
            )
        else:
            session_store = JsonStore("qwen-sessions", scope) if settings.cache_enabled else None
            context_store = JsonStore("qwen-contexts", scope) if settings.cache_enabled else None
            affinity_store = JsonStore("qwen-affinities", scope) if settings.cache_enabled else None
            qwen_accounts: list[QwenAccount] = []
            for i, token in enumerate(tokens):
                qw_client = QwenClient(token=token, timeout=settings.timeout)
                if not await _byok_validate("qwen", token, qw_client):
                    log.warning("byok qwen token invalid/expired, skipping")
                    await qw_client.aclose()
                    continue
                qwen_accounts.append(
                    QwenAccount(
                        i,
                        qw_client,
                        session_cache_size=settings.session_cache_size,
                        ttl=settings.session_ttl,
                        store=session_store,
                        stable_id=_token_stable_id(token),
                    )
                )
            if not qwen_accounts:
                raise HTTPException(401, "invalid qwen api key")
            pool = AccountPool(
                qwen_accounts,
                label="qwen",
                session_cache_size=settings.session_cache_size,
                ttl=settings.session_ttl,
                context_store=context_store,
                affinity_store=affinity_store,
            )
            if not getattr(app.state, "qwen_models", None):
                app.state.qwen_models = await _fetch_qwen_models(qwen_accounts[0].client)
        cache[cache_key] = pool
        while len(cache) > BYOK_POOL_LIMIT:
            oldest_key, oldest_pool = next(iter(cache.items()))
            cache.pop(oldest_key)
            await _close_pool(oldest_pool)
        return pool


async def _byok_pool_for(provider: str, request: Request) -> AccountPool:
    token = await _extract_request_api_key(request)
    tokens = [t.strip() for t in (token or "").split(",") if t.strip()]
    if not tokens:
        raise HTTPException(401, f"missing api key for {provider} provider")
    return await _byok_pool(provider, tokens)


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest, request: Request) -> Any:
    return await _dispatch_chat(req, request)


async def _dispatch_chat(req: ChatCompletionRequest, request: Request) -> Any:
    provider = _resolve_provider(req.model)
    if _byok_mode():
        pool = await _byok_pool_for(provider, request)
        if provider == "qwen":
            return await _chat_completions_qwen(req, pool=pool)
        return await _chat_completions_deepseek(req, pool=pool)
    if provider == "qwen":
        return await _chat_completions_qwen(req)
    return await _chat_completions_deepseek(req)


def _completion_prompts(prompt: Any) -> list[str]:
    if isinstance(prompt, str):
        return [prompt]
    if isinstance(prompt, list):
        prompts: list[str] = []
        for item in prompt:
            if isinstance(item, str):
                prompts.append(item)
            elif isinstance(item, list):
                prompts.append(" ".join(str(token) for token in item))
            else:
                raise HTTPException(400, "prompt must be a string, a list of strings, or a list of token lists")
        if not prompts:
            raise HTTPException(400, "prompt must not be empty")
        return prompts
    raise HTTPException(400, "prompt must be a string, a list of strings, or a list of token lists")


def _completion_chat_request(req: CompletionRequest, prompt_text: str, stream: bool) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model=req.model,
        messages=[ChatMessage(role="user", content=prompt_text)],
        stream=stream,
        temperature=req.temperature,
        top_p=req.top_p,
        max_tokens=req.max_tokens,
        n=req.n,
        stop=req.stop,
        presence_penalty=req.presence_penalty,
        frequency_penalty=req.frequency_penalty,
        logit_bias=req.logit_bias,
        user=req.user,
        session_id=req.session_id,
    )


def _legacy_choice_from_chat(chat_choice: dict, index: int) -> dict:
    message = chat_choice.get("message") or {}
    text = message.get("content") if isinstance(message, dict) else ""
    return {
        "index": index,
        "text": text if isinstance(text, str) else "",
        "logprobs": None,
        "finish_reason": chat_choice.get("finish_reason") or "stop",
    }


def _legacy_completion_response(chat_dict: dict, base_index: int) -> dict:
    choices: list[dict] = []
    for i, chat_choice in enumerate(chat_dict.get("choices") or []):
        choices.append(_legacy_choice_from_chat(chat_choice, base_index + i))
    return {
        "id": chat_dict.get("id"),
        "object": "text_completion",
        "created": chat_dict.get("created", int(time.time())),
        "model": chat_dict.get("model"),
        "choices": choices,
        "usage": chat_dict.get("usage"),
        "session_id": chat_dict.get("session_id"),
    }


def _translate_chat_chunk_to_completion(chunk: dict) -> dict:
    piece: dict[str, Any] = {
        "id": chunk.get("id", ""),
        "object": "text_completion",
        "created": chunk.get("created", int(time.time())),
        "model": chunk.get("model", ""),
        "choices": [],
    }
    if "usage" in chunk:
        piece["usage"] = chunk["usage"]
    if "error" in chunk:
        error = chunk["error"]
        piece["error"] = {"message": error.get("message") if isinstance(error, dict) else error}
    for choice in chunk.get("choices") or []:
        delta = choice.get("delta") or {}
        text = delta.get("content") if isinstance(delta, dict) else ""
        piece["choices"].append(
            {
                "index": choice.get("index", 0),
                "text": text if isinstance(text, str) else "",
                "logprobs": None,
                "finish_reason": choice.get("finish_reason"),
            }
        )
    return piece


async def _translate_completion_stream(chat_gen):
    try:
        async for line in chat_gen:
            if not line.startswith("data: "):
                yield line
                continue
            payload = line[len("data: ") :].strip()
            if payload == "[DONE]":
                continue
            try:
                chunk = json.loads(payload)
            except ValueError:
                yield line
                continue
            yield _sse(_translate_chat_chunk_to_completion(chunk))
    finally:
        await _close_generator(chat_gen)


async def _completions_stream(req: CompletionRequest, prompts: list[str], request: Request):
    for prompt_text in prompts:
        chat_req = _completion_chat_request(req, prompt_text, stream=True)
        chat_resp = await _dispatch_chat(chat_req, request)
        if isinstance(chat_resp, StreamingResponse):
            async for line in _translate_completion_stream(chat_resp.body_iterator):
                yield line
        else:
            data = _legacy_completion_response(chat_resp, 0)
            for choice in data["choices"]:
                yield _sse(
                    {
                        "id": data["id"],
                        "object": "text_completion",
                        "created": data["created"],
                        "model": data["model"],
                        "choices": [choice],
                    }
                )
    yield "data: [DONE]\n\n"


@app.post("/v1/completions")
async def completions(req: CompletionRequest, request: Request) -> Any:
    prompts = _completion_prompts(req.prompt)
    if req.stream:
        return StreamingResponse(
            _completions_stream(req, prompts, request),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    choices: list[dict] = []
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    base_index = 0
    created = 0
    completion_id = ""
    completion_model = req.model
    for prompt_text in prompts:
        chat_req = _completion_chat_request(req, prompt_text, stream=False)
        chat_dict = await _dispatch_chat(chat_req, request)
        choices.extend(_legacy_choice_from_chat(choice, base_index + i) for i, choice in enumerate(chat_dict.get("choices") or []))
        base_index += len(chat_dict.get("choices") or [])
        if not completion_id:
            completion_id = chat_dict.get("id")
        created = chat_dict.get("created", created)
        u = chat_dict.get("usage")
        if isinstance(u, dict):
            prompt_tokens += int(u.get("prompt_tokens") or 0)
            completion_tokens += int(u.get("completion_tokens") or 0)
            total_tokens += int(u.get("total_tokens") or 0)
    return {
        "id": completion_id or f"cmpl-{uuid.uuid4().hex}",
        "object": "text_completion",
        "created": created or int(time.time()),
        "model": completion_model,
        "choices": choices,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        },
    }


@app.post("/v1/embeddings")
async def embeddings_not_supported() -> dict:
    raise HTTPException(501, "embeddings are not supported by DanyAPI")


@app.post("/v1/moderations")
async def moderations_not_supported() -> dict:
    raise HTTPException(501, "moderations are not supported by DanyAPI")


def _responses_provider_call(req: ResponsesRequest) -> Any:
    if _resolve_provider(req.model) == "qwen":
        return _chat_completions_qwen
    return _chat_completions_deepseek


def _responses_chat_request(req: ResponsesRequest, provider_messages: list[dict], session_id: str | None) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model=req.model,
        messages=[ChatMessage(**message) for message in provider_messages],
        stream=req.stream,
        temperature=req.temperature,
        top_p=req.top_p,
        thinking=req.thinking,
        search=req.search,
        session_id=session_id,
        user=req.user,
        tools=responses_api.convert_tools(req.tools),
        tool_choice=responses_api.convert_tool_choice(req.tool_choice),
        parallel_tool_calls=req.parallel_tool_calls,
        response_format=responses_api.extract_response_format(req.text, req.response_format),
        stream_options={"include_usage": True} if req.stream else None,
    )


@app.post("/v1/responses")
async def create_response(req: ResponsesRequest, request: Request) -> Any:
    if _byok_mode():
        provider = _resolve_provider(req.model)
        pool = await _byok_pool_for(provider, request)
        if provider == "qwen":
            provider_call = partial(_chat_completions_qwen, pool=pool)
        else:
            provider_call = partial(_chat_completions_deepseek, pool=pool)
    else:
        provider_call = _responses_provider_call(req)
    store = _responses_store()
    try:
        new_input = responses_api.normalize_input(req.input)
    except responses_api.ResponsesInputError as exc:
        raise HTTPException(400, str(exc)) from exc

    base_conversation: list[dict] = []
    if req.previous_response_id:
        record = store.get(req.previous_response_id)
        if not isinstance(record, dict):
            raise HTTPException(404, f"response {req.previous_response_id} not found")
        stored = record.get("conversation")
        if isinstance(stored, list):
            base_conversation = stored
    conversation = list(base_conversation) + new_input

    provider_messages: list[dict] = []
    if req.instructions:
        provider_messages.append({"role": "system", "content": req.instructions})
    provider_messages.extend(conversation)

    session_id = None if req.previous_response_id else req.session_id
    chat_req = _responses_chat_request(req, provider_messages, session_id)

    info = responses_api.RequestInfo(
        model=req.model,
        instructions=req.instructions,
        max_output_tokens=req.max_output_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        tool_choice=req.tool_choice,
        tools=req.tools,
        parallel_tool_calls=req.parallel_tool_calls,
        previous_response_id=req.previous_response_id,
        store=req.store,
        metadata=req.metadata,
        user=req.user,
        text_format=responses_api.response_text_format(req.text),
        truncation=req.truncation,
        reasoning=req.reasoning,
    )

    response_id = f"resp_{uuid.uuid4().hex}"
    created_at = int(time.time())

    if req.stream:
        chat_resp = await provider_call(chat_req)
        conversation_snapshot = conversation

        def _on_complete(final: dict) -> None:
            if not req.store:
                return
            stored_conversation = conversation_snapshot + responses_api.messages_from_output(final.get("output"))
            store.set(response_id, {"public": final, "conversation": stored_conversation})

        return StreamingResponse(
            responses_api.translate_stream(chat_resp.body_iterator, info, response_id, created_at, _on_complete),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    chat_dict = await provider_call(chat_req)
    result = responses_api.response_from_chat(chat_dict, info, response_id, created_at)
    if req.store:
        stored_conversation = conversation + responses_api.messages_from_output(result.get("output"))
        store.set(response_id, {"public": result, "conversation": stored_conversation})
    return result


@app.get("/v1/responses/{response_id}")
async def get_response(response_id: str) -> dict:
    record = _responses_store().get(response_id)
    if not isinstance(record, dict):
        raise HTTPException(404, f"response {response_id} not found")
    public = record.get("public")
    return public if isinstance(public, dict) else {}


@app.delete("/v1/responses/{response_id}")
async def delete_response(response_id: str) -> dict:
    store = _responses_store()
    if not isinstance(store.get(response_id), dict):
        raise HTTPException(404, f"response {response_id} not found")
    store.discard(response_id)
    return {"id": response_id, "object": "response.deleted", "deleted": True}


@app.get("/v1/responses/{response_id}/input_items")
async def get_response_input_items(response_id: str) -> dict:
    record = _responses_store().get(response_id)
    if not isinstance(record, dict):
        raise HTTPException(404, f"response {response_id} not found")
    stored = record.get("conversation")
    messages = stored if isinstance(stored, list) else []
    if not messages:
        public = record.get("public")
        if isinstance(public, dict) and isinstance(public.get("input"), list):
            messages = public["input"].copy()
    items = responses_api.input_items_from_messages(messages)
    return {
        "object": "response.input_items_list",
        "data": items,
        "first_id": items[0]["id"] if items else None,
        "last_id": items[-1]["id"] if items else None,
        "has_more": False,
    }


@app.post("/v1/responses/{response_id}/cancel")
async def cancel_response(response_id: str) -> dict:
    record = _responses_store().get(response_id)
    if not isinstance(record, dict):
        raise HTTPException(404, f"response {response_id} not found")
    public = record.get("public")
    if isinstance(public, dict) and public.get("status") in ("in_progress", "queued"):
        record["public"] = dict(public) | {"status": "cancelled", "incomplete_details": {"reason": "cancelled"}}
        return record["public"]
    return {"id": response_id, "object": "response", "status": "cancelled"}


@app.post("/v1/images/generations")
async def image_generations(req: ImageGenerationRequest, request: Request) -> dict:
    pool: AccountPool | None
    if _byok_mode():
        pool = await _byok_pool_for("qwen", request)
    else:
        pool = getattr(app.state, "qwen_pool", None)
    if pool is None:
        raise HTTPException(503, "qwen provider is not configured (required for image generation)")
    return await _image_generations(req, pool)


def _image_http_client() -> httpx.AsyncClient:
    client = getattr(app.state, "http_client", None)
    if client is None:
        client = httpx.AsyncClient(follow_redirects=True, timeout=30)
        app.state.http_client = client
    return client


async def _image_generations(req: ImageGenerationRequest, pool: AccountPool | None = None) -> dict:
    if pool is None:
        pool = getattr(app.state, "qwen_pool", None)
    if pool is None:
        raise HTTPException(503, "qwen provider is not configured (required for image generation)")

    dims = _parse_image_size(req.size)
    count = max(1, int(getattr(req, "n", 1) or 1))

    account, existing_sid = await _acquire_account(pool, req.session_id)

    want_b64 = req.response_format == "b64_json"
    use_http = want_b64 or dims
    data: list[dict] = []
    usage = None
    result_sid = existing_sid
    hc = _image_http_client()
    download_sem = asyncio.Semaphore(4)

    async def _fetch_image(url: str) -> dict:
        if not use_http:
            return {"url": url}
        async with download_sem:
            try:
                img_resp = await hc.get(url)
                if img_resp.status_code != 200:
                    log.warning("image download failed (%s) for %s, returning url", img_resp.status_code, url)
                    return {"url": url}
                if dims is not None:
                    payload_bytes = await asyncio.to_thread(_resize_image_bytes, img_resp.content, dims)
                else:
                    payload_bytes = img_resp.content
                return {"b64_json": await _b64encode(payload_bytes)}
            except Exception as exc:
                log.warning("image fetch failed for %s, returning url: %s", url, exc)
                return {"url": url}

    try:
        for _ in range(count):
            result = await qwen_api.collect_image(
                account=account,
                pool=pool,
                existing_sid=result_sid,
                lock=account.sem,
                prompt=req.prompt,
                model=req.model,
                model_id=req.model,
                user=req.user,
            )
            result_sid = result.get("session_id") or result_sid
            if result.get("usage"):
                usage = result.get("usage")
            if result["image_urls"]:
                data.extend(await asyncio.gather(*(_fetch_image(url) for url in result["image_urls"])))
    except AccountPoolBusy:
        raise HTTPException(429, "all accounts are busy, try again later") from None

    if not data:
        raise HTTPException(502, "image generation returned no data")

    return {
        "created": int(time.time()),
        "data": data,
        "usage": usage,
        "session_id": result_sid,
    }


async def _image_pool(request: Request) -> AccountPool:
    pool: AccountPool | None
    if _byok_mode():
        pool = await _byok_pool_for("qwen", request)
    else:
        pool = getattr(app.state, "qwen_pool", None)
    if pool is None:
        raise HTTPException(503, "qwen provider is not configured (required for image generation)")
    return pool


_ASYNC_B64_THRESHOLD = 1 << 20


async def _b64encode(data: bytes) -> str:
    if len(data) > _ASYNC_B64_THRESHOLD:
        data = await asyncio.to_thread(base64.b64encode, data)
    else:
        data = base64.b64encode(data)
    return data.decode("ascii")


async def _image_markdown(data: bytes, content_type: str) -> str:
    return f"![image](data:{content_type or 'image/png'};base64,{await _b64encode(data)})"


async def _read_upload(file: UploadFile) -> tuple[bytes, str]:
    data = await file.read(MAX_FILE_SIZE + 1)
    if len(data) > MAX_FILE_SIZE:
        raise HTTPException(413, f"uploaded file exceeds {MAX_FILE_SIZE // (1024 * 1024)} MB limit")
    content_type = (file.content_type or "application/octet-stream").split(";", 1)[0].strip() or "application/octet-stream"
    return data, content_type


def _image_edit_req(prompt: str, image_md: str, mask_md: str | None) -> str:
    parts: list[str] = []
    if prompt.strip():
        parts.append(prompt.strip())
    parts.append(image_md)
    if mask_md:
        parts.append(mask_md)
    return "\n".join(parts)


@app.post("/v1/images/edits")
async def image_edits(
    request: Request,
    image: UploadFile = File(...),
    prompt: str = Form(default=""),
    mask: UploadFile | None = File(default=None),
    model: str = Form(default="qwen-image-gen"),
    n: int = Form(default=1),
    size: str | None = Form(default=None),
    response_format: str = Form(default="url"),
    user: str | None = Form(default=None),
    session_id: str | None = Form(default=None),
) -> dict:
    pool = await _image_pool(request)
    image_data, image_type = await _read_upload(image)
    edited = _image_edit_req(prompt, await _image_markdown(image_data, image_type), None)
    if mask is not None:
        mask_data, mask_type = await _read_upload(mask)
        edited = f"{edited}\n{await _image_markdown(mask_data, mask_type)}"
    req = ImageGenerationRequest(
        model=model,
        prompt=edited,
        n=n,
        size=size,
        response_format=response_format,
        session_id=session_id,
        user=user,
    )
    return await _image_generations(req, pool)


@app.post("/v1/images/variations")
async def image_variations(
    request: Request,
    image: UploadFile = File(...),
    model: str = Form(default="qwen-image-gen"),
    n: int = Form(default=1),
    size: str | None = Form(default=None),
    response_format: str = Form(default="url"),
    user: str | None = Form(default=None),
    session_id: str | None = Form(default=None),
) -> dict:
    pool = await _image_pool(request)
    image_data, image_type = await _read_upload(image)
    req = ImageGenerationRequest(
        model=model,
        prompt=await _image_markdown(image_data, image_type),
        n=n,
        size=size,
        response_format=response_format,
        session_id=session_id,
        user=user,
    )
    return await _image_generations(req, pool)


async def _acquire_account(pool: AccountPool, session_id: str | None):
    try:
        return await pool.acquire(session_id, settings.acquire_timeout)
    except AccountPoolBusy:
        raise HTTPException(429, "all accounts are busy, try again later") from None
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc


def _can_reuse_session(account: Any, session_id: str | None, **kwargs: Any) -> bool:
    return bool(account.sessions.can_reuse(session_id, **kwargs))


def _materialize_tools(req: ChatCompletionRequest) -> tuple[Any, Any]:
    tools = getattr(req, "tools", None)
    tool_choice = getattr(req, "tool_choice", None)
    functions = getattr(req, "functions", None)
    if functions:
        converted: list[dict] = []
        for fn in functions:
            if not isinstance(fn, dict):
                continue
            function: dict[str, Any] = {"name": fn.get("name") or ""}
            if "description" in fn:
                function["description"] = fn["description"]
            if "parameters" in fn:
                function["parameters"] = fn["parameters"]
            converted.append({"type": "function", "function": function})
        if converted:
            if isinstance(tools, list):
                tools = list(tools) + converted
            else:
                tools = converted
    if tool_choice is None and getattr(req, "function_call", None) is not None:
        function_call = req.function_call
        if isinstance(function_call, str):
            if function_call in ("auto", "none"):
                tool_choice = function_call
            elif function_call:
                tool_choice = {"type": "function", "function": {"name": function_call}}
        elif isinstance(function_call, dict) and isinstance(function_call.get("name"), str) and function_call["name"]:
            tool_choice = {"type": "function", "function": {"name": function_call["name"]}}
    return tools, tool_choice


def _max_calls(parallel_tool_calls: bool | None) -> int | None:
    return 1 if parallel_tool_calls is False else None


def _split_stop(stop: Any) -> list[str]:
    if stop is None:
        return []
    if isinstance(stop, str):
        return [stop] if stop else []
    if isinstance(stop, list):
        return [item for item in stop if isinstance(item, str) and item]
    return []


_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\u3040-\u30ff\uac00-\ud7af]")


def _cjk_units(text: str) -> int:
    return len(_CJK_RE.findall(text))


def _trim_to_tokens(text: str, budget: int | None) -> str:
    if budget is None or not text or estimate_tokens(text) <= budget:
        return text
    words = text.split(" ")
    parts: list[str] = []
    total_len = 0
    total_cjk = 0
    n_words = 0
    for word in words:
        cand_len = total_len + len(word) + (1 if n_words else 0)
        cand_cjk = total_cjk + _cjk_units(word)
        other = cand_len - cand_cjk
        if other == 0:
            candidate_tokens = cand_cjk
        else:
            candidate_tokens = cand_cjk + max(1, other // 4)
        if candidate_tokens > budget:
            break
        parts.append(word)
        total_len = cand_len
        total_cjk = cand_cjk
        n_words += 1
    return " ".join(parts)


def _apply_limits(content: str, max_tokens: int | None, stop: Any) -> tuple[str, str]:
    text = content or ""
    finish = "stop"
    stops = _split_stop(stop)
    if stops:
        cut = -1
        for marker in stops:
            position = text.find(marker)
            if position != -1 and (cut == -1 or position < cut):
                cut = position
        if cut != -1:
            text = text[:cut]
    trimmed = _trim_to_tokens(text, max_tokens)
    if trimmed != text:
        finish = "length"
    return trimmed, finish


async def _acquire_and_build(
    pool: AccountPool,
    req: ChatCompletionRequest,
    reuse_kwargs: dict[str, Any] | None = None,
    tools: Any = None,
    tool_choice: Any = None,
) -> tuple[Any, str | None, tuple[str, ...], str, bool]:
    context_seq = toolemu.context_sequence(req.messages, user=getattr(req, "user", None))
    if req.session_id:
        account, existing_sid = await _acquire_account(pool, req.session_id)
        if existing_sid is None:
            existing_sid = req.session_id
    else:
        cached_sid = pool.resolve_context(context_seq) if context_seq else None
        account, existing_sid = await _acquire_account(pool, cached_sid)
    has_session = _can_reuse_session(account, existing_sid, **(reuse_kwargs or {}))
    if tools is None:
        tools, tool_choice = _materialize_tools(req)
    try:
        prompt, tool_mode = toolemu.build_prompt(
            req.messages,
            tools,
            tool_choice,
            has_session,
            getattr(req, "response_format", None),
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return account, existing_sid, context_seq, prompt, tool_mode


def _include_usage(req: ChatCompletionRequest) -> bool:
    opts = getattr(req, "stream_options", None)
    if not isinstance(opts, dict):
        return False
    return bool(opts.get("include_usage"))


def _deepseek_usage(total: int, prompt: str = "", provider_usage: dict | None = None, completion_text: str | None = None) -> dict:
    prompt_tokens = 0
    if isinstance(provider_usage, dict):
        p_tokens = provider_usage.get("prompt_tokens")
        if isinstance(p_tokens, int) and p_tokens > 0:
            prompt_tokens = p_tokens
    if not prompt_tokens:
        prompt_tokens = estimate_tokens(prompt)
    total_tokens = max(0, int(total or 0))
    if total_tokens < prompt_tokens:
        total_tokens = prompt_tokens
    completion_tokens = max(0, total_tokens - prompt_tokens)
    if not completion_tokens and completion_text:
        completion_tokens = estimate_tokens(completion_text)
        total_tokens = prompt_tokens + completion_tokens
    return {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": total_tokens}


def _usage_with_details(usage: dict, reasoning_text: str | None = None, reasoning_tokens: int | None = None) -> dict:
    result = dict(usage)
    if not isinstance(result.get("prompt_tokens_details"), dict):
        result["prompt_tokens_details"] = {"cached_tokens": 0}
    if not isinstance(result.get("completion_tokens_details"), dict):
        if reasoning_tokens is None:
            reasoning_tokens = estimate_tokens(reasoning_text or "")
        result["completion_tokens_details"] = {"reasoning_tokens": reasoning_tokens}
    return result


def _advance_session_usage(session, accumulated_total: int) -> int:
    prev = max(0, int(getattr(session, "accumulated_tokens", 0) or 0))
    current = max(0, int(accumulated_total or 0))
    session.accumulated_tokens = max(prev, current)
    return max(0, current - prev)


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def _stream_error_sse(
    chunk_id: str,
    created: int,
    model: str,
    message: str,
    session_key: str | None = None,
    error_finish: str | None = None,
    choice_finish: str | None = None,
) -> tuple[str, str]:
    error: dict = {"message": message}
    if error_finish is not None:
        error["finish_reason"] = error_finish
    error_chunk = _sse(
        {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "session_id": session_key,
            "error": error,
            "choices": [{"index": 0, "delta": {}, "finish_reason": choice_finish or error_finish or "error"}],
        }
    )
    return error_chunk, "data: [DONE]\n\n"


def _chunk_id_from_line(line: str) -> str | None:
    if not line.startswith("data: "):
        return None
    payload = line[len("data: ") :].strip()
    if not payload or payload == "[DONE]":
        return None
    try:
        parsed = json.loads(payload)
    except ValueError:
        return None
    chunk_id = parsed.get("id") if isinstance(parsed, dict) else None
    return chunk_id if isinstance(chunk_id, str) and chunk_id else None


async def _close_generator(gen) -> None:
    aclose = getattr(gen, "aclose", None)
    if aclose is None:
        return
    try:
        await aclose()
    except Exception as exc:
        log.debug("stream generator close failed: %s", exc)


async def _stream_guard(gen, model: str):
    chunk_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    seen_id: str | None = None
    try:
        async for item in gen:
            if seen_id is None:
                seen_id = _chunk_id_from_line(item)
            yield item
    except AccountPoolBusy:
        for line in _stream_error_sse(seen_id or chunk_id, created, model, "all accounts are busy, try again later"):
            yield line
    except Exception as exc:
        log.exception("stream generator failed: %s", exc)
        msg = str(exc) or repr(exc) or "unknown stream error"
        for line in _stream_error_sse(seen_id or chunk_id, created, model, f"stream error: {msg}"):
            yield line
    finally:
        await _close_generator(gen)


async def _chat_completions_deepseek(req: ChatCompletionRequest, pool: AccountPool | None = None) -> Any:
    if pool is None:
        pool = getattr(app.state, "pool", None)
    if pool is None:
        raise HTTPException(503, "deepseek provider is not configured")

    model_type = _resolve_model(req.model)
    thinking = req.thinking if req.thinking is not None else _is_reasoning_model(req.model)
    search = bool(req.search)

    tools, tool_choice = _materialize_tools(req)
    account, existing_sid, context_seq, prompt, tool_mode = await _acquire_and_build(pool, req, tools=tools, tool_choice=tool_choice)

    attachments = _collect_attachments(req)
    _validate_attachments(attachments)

    max_tokens = getattr(req, "max_tokens", None)
    if max_tokens is None:
        max_tokens = getattr(req, "max_completion_tokens", None)

    common = {
        "account": account,
        "pool": pool,
        "existing_sid": existing_sid,
        "prompt": prompt,
        "model": req.model,
        "model_type": model_type,
        "thinking": thinking,
        "search": search,
        "attachments": attachments,
        "tool_schemas": toolemu.tool_schema_map(tools),
        "tool_mode": tool_mode,
        "include_usage": _include_usage(req),
        "context_seq": context_seq,
        "reduced_prompts": None,
        "messages": req.messages,
        "tools": tools,
        "tool_choice": tool_choice,
        "response_format": getattr(req, "response_format", None),
        "user": getattr(req, "user", None),
        "max_tokens": max_tokens,
        "stop": getattr(req, "stop", None),
        "n": getattr(req, "n", None),
        "parallel_tool_calls": getattr(req, "parallel_tool_calls", None),
    }
    if req.stream:
        return StreamingResponse(
            _stream_guard(_stream_openai(lock=account.sem, **common), req.model),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    try:
        return await _collect_non_stream(lock=account.sem, **{k: v for k, v in common.items() if k != "include_usage"})
    except AccountPoolBusy:
        raise HTTPException(429, "all accounts are busy, try again later") from None


async def _chat_completions_qwen(req: ChatCompletionRequest, pool: AccountPool | None = None) -> Any:
    if pool is None:
        pool = getattr(app.state, "qwen_pool", None)
    if pool is None:
        raise HTTPException(503, "qwen provider is not configured")

    thinking = req.thinking if req.thinking is not None else True
    search = bool(req.search)

    tools, tool_choice = _materialize_tools(req)
    account, existing_sid, context_seq, prompt, tool_mode = await _acquire_and_build(pool, req, {"model": req.model}, tools, tool_choice)

    attachments = _collect_attachments(req)
    if attachments:
        _validate_attachments(attachments)
        for att in attachments:
            if not att.is_image:
                raise HTTPException(400, "qwen only supports image attachments, use deepseek for files")
            prompt = f"{prompt}\n![image](data:{att.content_type};base64,{await _b64encode(att.data)})"

    max_tokens = getattr(req, "max_tokens", None)
    if max_tokens is None:
        max_tokens = getattr(req, "max_completion_tokens", None)

    common = {
        "account": account,
        "pool": pool,
        "existing_sid": existing_sid,
        "prompt": prompt,
        "model": req.model,
        "model_id": req.model,
        "thinking": thinking,
        "search": search,
        "tool_schemas": toolemu.tool_schema_map(tools),
        "tool_mode": tool_mode,
        "include_usage": _include_usage(req),
        "context_seq": context_seq,
        "messages": req.messages,
        "tools": tools,
        "tool_choice": tool_choice,
        "response_format": getattr(req, "response_format", None),
        "user": getattr(req, "user", None),
        "max_tokens": max_tokens,
        "stop": getattr(req, "stop", None),
        "n": getattr(req, "n", None),
        "parallel_tool_calls": getattr(req, "parallel_tool_calls", None),
    }
    if req.stream:
        return StreamingResponse(
            _stream_guard(qwen_api.stream_openai(lock=account.sem, **common), req.model),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    try:
        return await qwen_api.collect_non_stream(lock=account.sem, **{k: v for k, v in common.items() if k != "include_usage"})
    except AccountPoolBusy:
        raise HTTPException(429, "all accounts are busy, try again later") from None


async def _prepare_session(
    account: DeepSeekAccount,
    pool: AccountPool,
    existing_sid: str | None,
    context_seq: tuple[str, ...] | None = None,
) -> tuple[DeepSeekSession, str, str | None]:
    try:
        session, session_key = await account.sessions.obtain(existing_sid)
    except DeepSeekError as exc:
        _handle_account_error(account, exc)
        raise HTTPException(_deepseek_status(exc), _deepseek_error_detail(exc)) from exc
    pool.register(account.index, session_key)
    if existing_sid and session_key != existing_sid:
        pool.forget(existing_sid)
        pool.forget_context(existing_sid)
        account.sessions.forget(existing_sid)
    if context_seq:
        pool.index_context(session_key, context_seq)
    return session, session_key, session.last_message_id


async def _send_completion(
    client,
    pow_headers,
    session_id,
    parent_message_id,
    prompt,
    model_type,
    thinking,
    search,
    ref_file_ids=None,
):
    try:
        resp = await client.completion(
            chat_session_id=session_id,
            prompt=prompt,
            parent_message_id=parent_message_id,
            model_type=model_type,
            thinking_enabled=thinking,
            search_enabled=search,
            ref_file_ids=ref_file_ids,
            pow_headers=pow_headers,
        )
    except httpx.HTTPStatusError as exc:
        body = exc.response.text[:500]
        if _message_too_frequent_text(body) is not None:
            raise HTTPException(429, body) from exc
        raise HTTPException(exc.response.status_code, body) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(502, f"DeepSeek request failed: {exc}") from exc

    if resp is None or not hasattr(resp, "status_code"):
        raise HTTPException(502, "unexpected provider response")

    if resp.status_code != 200:
        body = await resp.aread()
        await resp.aclose()
        text = body[:500].decode("utf-8", errors="replace")
        if _message_too_frequent_text(text) is not None:
            raise HTTPException(429, text)
        raise HTTPException(resp.status_code, text)

    content_type = resp.headers.get("content-type", "")
    if "text/event-stream" not in content_type:
        body = await resp.aread()
        await resp.aclose()
        try:
            payload = json.loads(body)
        except json.JSONDecodeError as exc:
            text = body[:500].decode("utf-8", errors="replace")
            if _message_too_frequent_text(text) is not None:
                raise HTTPException(429, text) from exc
            raise HTTPException(502, text) from exc
        data = payload.get("data") or {}
        if data.get("biz_code"):
            code = data["biz_code"]
            detail = f"DeepSeek error {code}: {data.get('biz_msg')}"
            status = 401 if code in DEEPSEEK_AUTH_ERROR_CODES else 502
            if _message_too_frequent_text(detail) is not None:
                status = 429
            raise HTTPException(status, detail)
        if payload.get("code"):
            code = payload["code"]
            detail = f"DeepSeek error {code}: {payload.get('msg') or payload.get('message')}"
            status = 401 if code in DEEPSEEK_AUTH_ERROR_CODES else 502
            if _message_too_frequent_text(detail) is not None:
                status = 429
            raise HTTPException(status, detail)
        raise HTTPException(502, "unexpected non-stream response")
    return resp


def _is_retryable_hint(rec: MessageReconstructor) -> bool:
    hint = rec.hint_error
    return bool(hint and hint.get("finish_reason") in RETRYABLE_FINISH_REASONS)


FAKE_CONTEXT_HINT_MARKERS = ("length limit reached",)


def _is_fake_context_hint(rec: MessageReconstructor) -> bool:
    hint = rec.hint_error
    if not hint:
        return False
    message = hint.get("message")
    if not isinstance(message, str):
        return False
    return any(marker in message.casefold() for marker in FAKE_CONTEXT_HINT_MARKERS)


RETRYABLE_HTTP_STATUSES = {408, 425, 429, 500, 502, 503, 504}
STALE_SESSION_STATUSES = {400, 404}


def _retry_delay(attempt: int) -> float:
    return min(RETRY_BACKOFF_SEC * (2 ** (attempt - 1)), RETRY_BACKOFF_MAX_SEC)


def _compact_error_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return "".join(ch for ch in value.casefold() if ch.isalnum())


def _error_text(detail: Any) -> str:
    if isinstance(detail, str):
        return detail
    try:
        return json.dumps(detail, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(detail)


def _message_too_frequent_text(*values: Any) -> str | None:
    for value in values:
        compact = _compact_error_text(value)
        if not compact:
            continue
        if any(marker in compact for marker in MESSAGE_TOO_FREQUENT_MARKERS):
            return value
    return None


def _is_message_too_frequent_http(exc: HTTPException) -> bool:
    return _message_too_frequent_text(_error_text(exc.detail)) is not None


def _is_message_too_frequent_hint(rec: MessageReconstructor) -> bool:
    hint = rec.hint_error
    if not hint:
        return False
    return _message_too_frequent_text(hint.get("message"), hint.get("finish_reason")) is not None


async def _wait_message_too_frequent(stage: str, attempt: int) -> None:
    log.warning(
        "deepseek message too frequent (%s), retry %d/%d in %.0fs",
        stage,
        attempt,
        MESSAGE_TOO_FREQUENT_MAX_RETRIES,
        MESSAGE_TOO_FREQUENT_WAIT_SEC,
    )
    await asyncio.sleep(MESSAGE_TOO_FREQUENT_WAIT_SEC)


def _is_retryable_http(exc: HTTPException) -> bool:
    return exc.status_code in RETRYABLE_HTTP_STATUSES


def _is_context_limit(rec: MessageReconstructor) -> bool:
    if rec.status == CONTEXT_LENGTH_STATUS:
        return True
    hint = rec.hint_error
    return bool(hint and hint.get("finish_reason") == CONTEXT_LENGTH_STATUS)


def _is_input_exceeds_limit(rec: MessageReconstructor) -> bool:
    if rec.status == INPUT_EXCEEDS_LIMIT:
        return True
    hint = rec.hint_error
    return bool(hint and hint.get("finish_reason") == INPUT_EXCEEDS_LIMIT)


def _incomplete_message(rec: MessageReconstructor) -> str:
    hint = rec.hint_error or {}
    message = hint.get("message")
    return message if isinstance(message, str) and message else RESPONSE_INCOMPLETE_MESSAGE


def _incomplete_error_body(message: str) -> str:
    return json.dumps({"error": {"message": message, "finish_reason": RESPONSE_INCOMPLETE}}, ensure_ascii=False)


def _input_exceeds_hint_from_http(exc: HTTPException) -> dict | None:
    detail = exc.detail
    if isinstance(detail, str):
        try:
            detail = json.loads(detail)
        except json.JSONDecodeError:
            return None
    if not isinstance(detail, dict):
        return None
    if detail.get("finish_reason") != INPUT_EXCEEDS_LIMIT:
        return None
    message = detail.get("message")
    return {
        "message": message if isinstance(message, str) else "Content is too long",
        "finish_reason": INPUT_EXCEEDS_LIMIT,
    }


def _drop_session(pool, account, session_key) -> None:
    pool.forget(session_key)
    pool.forget_context(session_key)
    account.sessions.forget(session_key)


def _deepseek_status(exc: DeepSeekError) -> int:
    return 401 if exc.biz_code in DEEPSEEK_AUTH_ERROR_CODES else 502


async def _send_with_auth(account, *args, **kwargs):
    try:
        return await _send_completion(*args, **kwargs)
    except HTTPException as exc:
        if exc.status_code in (401, 403):
            account.mark_broken()
        raise


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


def _busy_error_body(rec: MessageReconstructor) -> str:
    hint = rec.hint_error or {}
    return json.dumps(
        {
            "error": {
                "message": hint.get("message") or "DeepSeek server is busy, try again later",
                "finish_reason": hint.get("finish_reason"),
            }
        },
        ensure_ascii=False,
    )


FAKE_CONTEXT_HINT_ERROR_MESSAGE = "DeepSeek returned an unexpected length-limit hint and the response is empty"


def _fake_context_error_body() -> str:
    return json.dumps(
        {"error": {"message": FAKE_CONTEXT_HINT_ERROR_MESSAGE, "finish_reason": "server_error"}},
        ensure_ascii=False,
    )


async def _try_stop_stream(client, session_id: str, message_id: str | None) -> None:
    if not session_id or not message_id:
        return
    try:
        await client.stop_stream(session_id, message_id)
    except Exception as exc:
        log.debug("stop_stream failed for %s: %s", session_id, exc)


def _build_assistant_message(
    content: str,
    reasoning: str | None,
    tool_mode: bool,
    tool_schemas: dict | None,
    max_calls: int | None = None,
) -> tuple[dict, str]:
    if tool_mode:
        parsed = toolemu.parse_tool_calls(content, tool_schemas)
        if parsed is not None:
            tool_calls, tool_text = parsed
            if tool_calls:
                if max_calls is not None:
                    tool_calls = tool_calls[:max_calls]
                return toolemu.format_tool_message(tool_calls, tool_text, reasoning), "tool_calls"
    message = {"role": "assistant", "content": content}
    if reasoning:
        message["reasoning_content"] = reasoning
    return message, "stop"


def _build_limited_message(
    content: str,
    reasoning: str | None,
    tool_mode: bool,
    tool_schemas: dict | None,
    max_tokens: int | None,
    stop: Any,
    parallel_tool_calls: bool | None,
    provider_finish: Any,
) -> tuple[dict, str]:
    if tool_mode:
        message, finish = _build_assistant_message(content, reasoning, True, tool_schemas, max_calls=_max_calls(parallel_tool_calls))
        if finish == "tool_calls":
            tool_text = message.get("content")
            if isinstance(tool_text, str):
                message["content"] = _trim_to_tokens(tool_text, max_tokens)
            return message, finish
        text, limit_finish = _apply_limits(str(message.get("content") or ""), max_tokens, stop)
        message["content"] = text
        if limit_finish == "length":
            return message, "length"
        return message, _finish_reason(provider_finish)
    text, limit_finish = _apply_limits(content or "", max_tokens, stop)
    message = {"role": "assistant", "content": text}
    if reasoning:
        message["reasoning_content"] = reasoning
    if limit_finish == "length":
        return message, "length"
    return message, _finish_reason(provider_finish)


def _build_completion_response(
    model: str,
    message: dict,
    finish: str,
    usage: dict,
    session_key: str | None,
    reasoning_tokens: int | None = None,
) -> dict:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "system_fingerprint": SYSTEM_FINGERPRINT,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish,
                "logprobs": None,
            }
        ],
        "usage": _usage_with_details(usage, (message or {}).get("reasoning_content"), reasoning_tokens),
        "session_id": session_key,
    }


async def _send_deepseek_stream(
    account,
    session,
    parent_message_id,
    prompt,
    model_type,
    thinking,
    search,
    ref_file_ids=None,
) -> tuple[MessageReconstructor, str | None, str | None]:
    pow_headers = await _fresh_pow_headers(account)
    resp = await _send_with_auth(
        account,
        account.client,
        pow_headers,
        session.id,
        parent_message_id,
        prompt,
        model_type,
        thinking,
        search,
        ref_file_ids,
    )
    rec = MessageReconstructor()
    incremental = IncrementalSSE()
    response_message_id: str | None = None
    stop_message_id: str | None = None
    stopped = False
    try:
        async for chunk in resp.aiter_bytes():
            for event in incremental.feed(chunk):
                if event.event == "ready" and isinstance(event.data, dict):
                    response_message_id = event.data.get("response_message_id")
                    if response_message_id:
                        stop_message_id = response_message_id
                rec.handle(event)
        for event in incremental.finish():
            rec.handle(event)
    except (httpx.HTTPError, RuntimeError) as exc:
        stopped = True
        if rec.id:
            stop_message_id = rec.id
        await _try_stop_stream(account.client, session.id, stop_message_id)
        raise DeepSeekStreamError(f"Stream processing failed: {exc}") from exc
    except BaseException:
        stopped = True
        if rec.id:
            stop_message_id = rec.id
        await _try_stop_stream(account.client, session.id, stop_message_id)
        raise
    finally:
        if rec.id:
            stop_message_id = rec.id
        try:
            await resp.aclose()
        except Exception as exc:
            log.debug("response close failed: %s", exc)
            if not stopped:
                await _try_stop_stream(account.client, session.id, stop_message_id)
    return rec, response_message_id, stop_message_id


async def _collect_continuation(
    account,
    session,
    parent_message_id,
    model_type,
    thinking,
    search,
    ref_file_ids=None,
) -> MessageReconstructor | None:
    attempt = 0
    rate_attempt = 0
    while True:
        try:
            rec, _response_message_id, _stop_message_id = await _send_deepseek_stream(
                account,
                session,
                parent_message_id,
                CONTINUE_PROMPT,
                model_type,
                thinking,
                search,
                ref_file_ids,
            )
        except DeepSeekStreamError as exc:
            raise HTTPException(502, str(exc)) from exc
        except HTTPException as exc:
            if _is_message_too_frequent_http(exc) and rate_attempt < MESSAGE_TOO_FREQUENT_MAX_RETRIES:
                rate_attempt += 1
                await _wait_message_too_frequent("continuation request", rate_attempt)
                continue
            if _is_retryable_http(exc) and attempt < MAX_RETRIES:
                attempt += 1
                delay = _retry_delay(attempt)
                log.warning(
                    "deepseek continuation error (%s), retry %d/%d in %.1fs",
                    exc.status_code,
                    attempt,
                    MAX_RETRIES,
                    delay,
                )
                await asyncio.sleep(delay)
                continue
            return None
        if (
            not (rec.content or rec.reasoning)
            and not _is_input_exceeds_limit(rec)
            and _is_message_too_frequent_hint(rec)
            and rate_attempt < MESSAGE_TOO_FREQUENT_MAX_RETRIES
        ):
            rate_attempt += 1
            await _wait_message_too_frequent("continuation hint", rate_attempt)
            continue
        if (
            not (rec.content or rec.reasoning)
            and not _is_input_exceeds_limit(rec)
            and (_is_retryable_hint(rec) or _is_fake_context_hint(rec))
            and attempt < MAX_RETRIES
        ):
            attempt += 1
            delay = _retry_delay(attempt)
            log.warning(
                "deepseek continuation retryable hint (%s), retry %d/%d in %.1fs",
                (rec.hint_error or {}).get("finish_reason"),
                attempt,
                MAX_RETRIES,
                delay,
            )
            await asyncio.sleep(delay)
            continue
        return rec


def _reduced_prompt_variants(
    messages: list[Any], tools: list[Any] | None, tool_choice: Any, response_format: Any, original_prompt: str
) -> list[tuple[str, bool, dict[str, Any]]]:
    variants: list[tuple[str, bool, dict[str, Any]]] = []
    if tools:
        try:
            prompt, tool_mode = toolemu.build_prompt(messages, None, None, False, response_format)
            if prompt != original_prompt:
                variants.append((prompt, tool_mode, {}))
        except ValueError:
            pass
    system_msgs = [msg for msg in messages if getattr(msg, "role", None) == "system"]
    last_input = None
    for msg in reversed(messages):
        if getattr(msg, "role", None) in ("user", "system"):
            last_input = msg
            break
    reduced_msgs = list(system_msgs)
    if last_input is not None and getattr(last_input, "role", None) == "user":
        reduced_msgs.append(last_input)
    if reduced_msgs:
        try:
            prompt, tool_mode = toolemu.build_prompt(reduced_msgs, None, None, False, response_format)
            if prompt != original_prompt:
                variants.append((prompt, tool_mode, {}))
        except ValueError:
            pass
    if reduced_msgs and tools:
        try:
            prompt, tool_mode = toolemu.build_prompt(reduced_msgs, tools, tool_choice, False, response_format)
            if prompt != original_prompt and not any(v[0] == prompt for v in variants):
                variants.append((prompt, tool_mode, toolemu.tool_schema_map(tools)))
        except ValueError:
            pass
    return variants


async def _collect_reduced(
    account,
    pool,
    reduced_prompts: list[tuple[str, bool, dict[str, Any]]],
    model_type,
    thinking,
    search,
    ref_file_ids=None,
):
    await _human_delay()
    for prompt, variant_tool_mode, variant_tool_schemas in reduced_prompts:
        session_key = None
        try:
            session, session_key, parent_message_id = await _prepare_session(account, pool, None, None)
            rec, _response_message_id, _stop_message_id = await _send_deepseek_stream(
                account,
                session,
                parent_message_id,
                prompt,
                model_type,
                thinking,
                search,
                ref_file_ids,
            )
        except (HTTPException, httpx.HTTPError, DeepSeekStreamError):
            if session_key is not None:
                _drop_session(pool, account, session_key)
            continue
        if (rec.content or rec.reasoning) and not _is_input_exceeds_limit(rec):
            return rec, session, session_key, variant_tool_mode, variant_tool_schemas
        if session_key is not None:
            _drop_session(pool, account, session_key)
    return None


async def _collect_non_stream(
    account,
    pool,
    existing_sid,
    lock,
    prompt,
    model,
    model_type,
    thinking,
    search,
    ref_file_ids=None,
    attachments=None,
    tool_mode=False,
    tool_schemas=None,
    context_seq: tuple[str, ...] | None = None,
    reduced_prompts: list[tuple[str, bool, dict[str, Any]]] | None = None,
    messages=None,
    tools=None,
    tool_choice=None,
    response_format=None,
    user=None,
    max_tokens: int | None = None,
    stop: Any = None,
    n: int | None = None,
    parallel_tool_calls: bool | None = None,
):
    await _human_delay()
    async with account_lock(lock, settings.acquire_timeout):
        if attachments:
            ref_file_ids = await _upload_attachments(account, attachments, model_type, thinking)
        session, session_key, parent_message_id = await _prepare_session(account, pool, existing_sid, context_seq)
        if session_key != existing_sid and messages is not None:
            try:
                prompt, tool_mode = toolemu.build_prompt(messages, tools, tool_choice, False, response_format)
            except ValueError:
                pass
            tool_schemas = toolemu.tool_schema_map(tools)
        stop_message_id: str | None = None
        started = time.monotonic()
        had_cached_session = bool(existing_sid) and account.sessions.get(existing_sid) is not None
        stale_rebuilt = False
        rec: MessageReconstructor | None = None
        response_message_id = None
        attempt = 0
        rate_attempt = 0

        try:
            while True:
                try:
                    rec, response_message_id, stop_message_id = await _send_deepseek_stream(
                        account,
                        session,
                        parent_message_id,
                        prompt,
                        model_type,
                        thinking,
                        search,
                        ref_file_ids,
                    )
                except DeepSeekStreamError as exc:
                    raise HTTPException(502, str(exc)) from exc
                except HTTPException as exc:
                    input_hint = _input_exceeds_hint_from_http(exc)
                    if input_hint is not None:
                        rec = MessageReconstructor()
                        rec.hint_error = input_hint
                        break
                    if exc.status_code in STALE_SESSION_STATUSES and had_cached_session and not stale_rebuilt and messages is not None:
                        stale_rebuilt = True
                        _drop_session(pool, account, session_key)
                        try:
                            prompt, tool_mode = toolemu.build_prompt(messages, tools, tool_choice, False, response_format)
                            tool_schemas = toolemu.tool_schema_map(tools)
                        except (ValueError, TypeError, AttributeError) as build_exc:
                            raise exc from build_exc
                        log.warning("deepseek session %s is stale (%s), rebuilt full history into a fresh chat", session_key, exc.status_code)
                        session, session_key, parent_message_id = await _prepare_session(account, pool, existing_sid, context_seq)
                        stop_message_id = None
                        response_message_id = None
                        continue
                    if _is_message_too_frequent_http(exc) and rate_attempt < MESSAGE_TOO_FREQUENT_MAX_RETRIES:
                        rate_attempt += 1
                        await _wait_message_too_frequent("request", rate_attempt)
                        continue
                    if _is_retryable_http(exc) and attempt < MAX_RETRIES:
                        attempt += 1
                        delay = _retry_delay(attempt)
                        log.warning(
                            "deepseek provider error (%s), retry %d/%d in %.1fs",
                            exc.status_code,
                            attempt,
                            MAX_RETRIES,
                            delay,
                        )
                        await asyncio.sleep(delay)
                        continue
                    raise
                if (
                    not (rec.content or rec.reasoning)
                    and not _is_input_exceeds_limit(rec)
                    and _is_message_too_frequent_hint(rec)
                    and rate_attempt < MESSAGE_TOO_FREQUENT_MAX_RETRIES
                ):
                    rate_attempt += 1
                    await _wait_message_too_frequent("stream hint", rate_attempt)
                    continue
                if (
                    not (rec.content or rec.reasoning)
                    and not _is_input_exceeds_limit(rec)
                    and (_is_retryable_hint(rec) or _is_fake_context_hint(rec))
                    and attempt < MAX_RETRIES
                ):
                    attempt += 1
                    delay = _retry_delay(attempt)
                    log.warning(
                        "deepseek retryable hint (%s), retry %d/%d in %.1fs",
                        (rec.hint_error or {}).get("finish_reason"),
                        attempt,
                        MAX_RETRIES,
                        delay,
                    )
                    await asyncio.sleep(delay)
                    continue
                break
        except BaseException:
            await _try_stop_stream(account.client, session.id, stop_message_id)
            raise

        assert rec is not None
        if _is_context_limit(rec) and not (rec.content or rec.reasoning):
            _drop_session(pool, account, session_key)
            raise HTTPException(400, "context length exceeded: conversation too long, start a new conversation")
        incomplete_message: str | None = None
        reduced_notice: str | None = None
        if _is_input_exceeds_limit(rec):
            incomplete_message = _incomplete_message(rec)
            cont_parent = rec.id or response_message_id or parent_message_id
            for _ in range(MAX_CONTINUE_ROUNDS):
                cont_rec = await _collect_continuation(account, session, cont_parent, model_type, thinking, search, ref_file_ids)
                if cont_rec is None:
                    break
                rec.extend_with(cont_rec)
                cont_parent = cont_rec.id or cont_parent
                if not _is_input_exceeds_limit(cont_rec):
                    incomplete_message = None
                    break
                incomplete_message = _incomplete_message(cont_rec)
                if not cont_rec.content:
                    break
            if incomplete_message is not None and not (rec.content or rec.reasoning):
                if reduced_prompts is None and messages is not None:
                    reduced_prompts = _reduced_prompt_variants(messages, tools, tool_choice, response_format, prompt)
                if reduced_prompts:
                    _drop_session(pool, account, session_key)
                    reduced = await _collect_reduced(account, pool, reduced_prompts, model_type, thinking, search, ref_file_ids)
                    if reduced is not None:
                        rec, session, session_key, variant_tool_mode, variant_tool_schemas = reduced
                        if variant_tool_mode:
                            tool_mode = variant_tool_mode
                            tool_schemas = variant_tool_schemas
                        response_message_id = rec.id or response_message_id
                        stop_message_id = response_message_id
                        reduced_notice = REDUCED_CONTEXT_MESSAGE
        if incomplete_message is not None and reduced_notice is None:
            log.warning("deepseek response incomplete: %s", incomplete_message)
            raise HTTPException(502, _incomplete_error_body(incomplete_message))
        content = rec.content
        reasoning = rec.reasoning
        if not (content or reasoning) and rec.hint_error:
            if _is_fake_context_hint(rec):
                log.warning("deepseek fake context-length hint after retries")
                raise HTTPException(502, _fake_context_error_body())
            raise HTTPException(429, _busy_error_body(rec))
        request_tokens = _advance_session_usage(session, rec.accumulated_tokens)
        usage = _deepseek_usage(request_tokens, prompt, rec.usage, completion_text=rec.content or rec.reasoning)
        account.sessions.touch_last_message(session_key, rec.id or response_message_id)
        record_usage(
            "deepseek",
            model,
            usage["prompt_tokens"],
            usage["completion_tokens"],
            usage["total_tokens"],
            user=user,
            session_id=session_key,
        )
        log.info("deepseek completion success (%.0fms)", (time.monotonic() - started) * 1000)
        message, finish = _build_limited_message(content, reasoning, tool_mode, tool_schemas, max_tokens, stop, parallel_tool_calls, rec.status)
        reasoning_tokens = estimate_tokens(reasoning) if reasoning else 0
        response = _build_completion_response(model, message, finish, usage, session_key, reasoning_tokens)
        if isinstance(n, int) and n and n > 1:
            template = response["choices"][0]
            response["choices"] = [dict(template) | {"index": i} for i in range(n)]
        if reduced_notice is not None:
            log.warning("deepseek response delivered from reduced context (%s)", model)
            response["error"] = {"message": reduced_notice, "finish_reason": RESPONSE_INCOMPLETE}
            response["choices"][0]["finish_reason"] = RESPONSE_INCOMPLETE
        return response


async def _stream_openai(
    account,
    pool,
    existing_sid,
    lock,
    prompt,
    model,
    model_type,
    thinking,
    search,
    ref_file_ids=None,
    attachments=None,
    tool_mode=False,
    tool_schemas=None,
    include_usage=False,
    context_seq: tuple[str, ...] | None = None,
    reduced_prompts: list[tuple[str, bool, dict[str, Any]]] | None = None,
    messages=None,
    tools=None,
    tool_choice=None,
    response_format=None,
    user=None,
    max_tokens: int | None = None,
    stop: Any = None,
    n: int | None = None,
    parallel_tool_calls: bool | None = None,
):
    chunk_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    await _human_delay()
    async with account_lock(lock, settings.acquire_timeout):
        if attachments:
            ref_file_ids = await _upload_attachments(account, attachments, model_type, thinking)
        try:
            session, session_key, parent_message_id = await _prepare_session(account, pool, existing_sid, context_seq)
        except HTTPException as exc:
            detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
            for line in _stream_error_sse(chunk_id, created, model, detail):
                yield line
            return

        if session_key != existing_sid and messages is not None:
            try:
                prompt, tool_mode = toolemu.build_prompt(messages, tools, tool_choice, False, response_format)
            except ValueError:
                pass
            tool_schemas = toolemu.tool_schema_map(tools)

        rec: MessageReconstructor | None = None
        response_message_id = None
        stop_message_id: str | None = None
        content_parts: list[str] = []
        content_buf = ""
        content_shown_len = 0
        tool_hidden = False
        role_sent = False
        started = time.monotonic()
        budget = StreamBudget(max_tokens, _trim_to_tokens)

        def content_piece(piece: str | None) -> str:
            return budget.feed(piece)

        had_cached_session = bool(existing_sid) and account.sessions.get(existing_sid) is not None
        stale_rebuilt = False
        attempt = 0
        rate_attempt = 0
        while True:
            try:
                pow_headers = await _fresh_pow_headers(account)
                resp = await _send_with_auth(
                    account,
                    account.client,
                    pow_headers,
                    session.id,
                    parent_message_id,
                    prompt,
                    model_type,
                    thinking,
                    search,
                    ref_file_ids,
                )
            except HTTPException as exc:
                input_hint = _input_exceeds_hint_from_http(exc)
                if input_hint is not None:
                    rec = MessageReconstructor()
                    rec.hint_error = input_hint
                    break
                if exc.status_code in STALE_SESSION_STATUSES and had_cached_session and not stale_rebuilt and messages is not None:
                    stale_rebuilt = True
                    _drop_session(pool, account, session_key)
                    try:
                        prompt, tool_mode = toolemu.build_prompt(messages, tools, tool_choice, False, response_format)
                        tool_schemas = toolemu.tool_schema_map(tools)
                    except ValueError:
                        detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
                        for line in _stream_error_sse(chunk_id, created, model, detail, session_key):
                            yield line
                        return
                    log.warning("deepseek session %s is stale (%s), rebuilt full history into a fresh chat", session_key, exc.status_code)
                    session, session_key, parent_message_id = await _prepare_session(account, pool, existing_sid, context_seq)
                    stop_message_id = None
                    response_message_id = None
                    continue
                if _is_message_too_frequent_http(exc) and rate_attempt < MESSAGE_TOO_FREQUENT_MAX_RETRIES:
                    rate_attempt += 1
                    await _wait_message_too_frequent("request", rate_attempt)
                    continue
                if _is_retryable_http(exc) and attempt < MAX_RETRIES:
                    attempt += 1
                    delay = _retry_delay(attempt)
                    log.warning(
                        "deepseek provider error (%s), retry %d/%d in %.1fs",
                        exc.status_code,
                        attempt,
                        MAX_RETRIES,
                        delay,
                    )
                    await asyncio.sleep(delay)
                    continue
                detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
                for line in _stream_error_sse(chunk_id, created, model, detail, session_key):
                    yield line
                return
            rec = MessageReconstructor()
            incremental = IncrementalSSE()
            response_message_id = None
            got_content = False
            role_sent = False
            stopped = False
            try:
                async for chunk in resp.aiter_bytes():
                    for event in incremental.feed(chunk):
                        if event.event == "ready" and isinstance(event.data, dict):
                            response_message_id = event.data.get("response_message_id")
                            if response_message_id:
                                stop_message_id = response_message_id
                        rec.handle(event)
                        c_diff, r_diff = rec.take_diffs()
                        if not (c_diff or r_diff):
                            continue
                        got_content = True
                        if not role_sent:
                            role_sent = True
                            yield _sse(
                                {
                                    "id": chunk_id,
                                    "object": "chat.completion.chunk",
                                    "created": created,
                                    "model": model,
                                    "choices": [
                                        {
                                            "index": 0,
                                            "delta": {"role": "assistant"},
                                            "finish_reason": None,
                                        }
                                    ],
                                }
                            )
                        delta: dict = {}
                        if c_diff:
                            if tool_mode:
                                content_parts.append(c_diff)
                                content_buf = "".join(content_parts)
                                visible, content_shown_len, tool_hidden = toolemu.tool_visible(content_buf, content_shown_len, tool_hidden, tool_schemas)
                                allowed = content_piece(visible)
                                if allowed:
                                    delta["content"] = allowed
                            else:
                                allowed = content_piece(c_diff)
                                if allowed:
                                    delta["content"] = allowed
                        if r_diff:
                            delta["reasoning_content"] = r_diff
                        if delta:
                            yield _sse(
                                {
                                    "id": chunk_id,
                                    "object": "chat.completion.chunk",
                                    "created": created,
                                    "model": model,
                                    "choices": [
                                        {
                                            "index": 0,
                                            "delta": delta,
                                            "finish_reason": None,
                                        }
                                    ],
                                }
                            )
                for event in incremental.finish():
                    if event.event == "ready" and isinstance(event.data, dict):
                        response_message_id = event.data.get("response_message_id")
                        if response_message_id:
                            stop_message_id = response_message_id
                    rec.handle(event)
                    c_diff, r_diff = rec.take_diffs()
                    if not (c_diff or r_diff):
                        continue
                    got_content = True
                    if not role_sent:
                        role_sent = True
                        yield _sse(
                            {
                                "id": chunk_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": model,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {"role": "assistant"},
                                        "finish_reason": None,
                                    }
                                ],
                            }
                        )
                    delta2: dict = {}
                    if c_diff:
                        if tool_mode:
                            content_parts.append(c_diff)
                            content_buf = "".join(content_parts)
                            visible, content_shown_len, tool_hidden = toolemu.tool_visible(content_buf, content_shown_len, tool_hidden, tool_schemas)
                            allowed = content_piece(visible)
                            if allowed:
                                delta2["content"] = allowed
                        else:
                            allowed = content_piece(c_diff)
                            if allowed:
                                delta2["content"] = allowed
                    if r_diff:
                        delta2["reasoning_content"] = r_diff
                    if delta2:
                        yield _sse(
                            {
                                "id": chunk_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": model,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": delta2,
                                        "finish_reason": None,
                                    }
                                ],
                            }
                        )
            except BaseException:
                stopped = True
                if rec.id:
                    stop_message_id = rec.id
                await _try_stop_stream(account.client, session.id, stop_message_id)
                raise
            finally:
                if rec.id:
                    stop_message_id = rec.id
                try:
                    await resp.aclose()
                except Exception as exc:
                    log.debug("response close failed: %s", exc)
                    if not stopped:
                        await _try_stop_stream(account.client, session.id, stop_message_id)
            if got_content:
                break
            if not _is_input_exceeds_limit(rec) and _is_message_too_frequent_hint(rec) and rate_attempt < MESSAGE_TOO_FREQUENT_MAX_RETRIES:
                rate_attempt += 1
                await _wait_message_too_frequent("stream hint", rate_attempt)
                continue
            if not _is_input_exceeds_limit(rec) and (_is_retryable_hint(rec) or _is_fake_context_hint(rec)) and attempt < MAX_RETRIES:
                attempt += 1
                delay = _retry_delay(attempt)
                log.warning(
                    "deepseek retryable hint (%s), retry %d/%d in %.1fs",
                    (rec.hint_error or {}).get("finish_reason"),
                    attempt,
                    MAX_RETRIES,
                    delay,
                )
                await asyncio.sleep(delay)
                continue
            break

        assert rec is not None
        if _is_context_limit(rec) and not (rec.content or rec.reasoning):
            _drop_session(pool, account, session_key)
            for line in _stream_error_sse(
                chunk_id,
                created,
                model,
                "context length exceeded: conversation too long, start a new conversation",
                session_key,
                CONTEXT_LENGTH_STATUS,
                "length",
            ):
                yield line
            return
        incomplete_message: str | None = None
        reduced_notice: str | None = None
        if _is_input_exceeds_limit(rec):
            incomplete_message = _incomplete_message(rec)
            cont_parent = rec.id or response_message_id or parent_message_id
            for _ in range(MAX_CONTINUE_ROUNDS):
                cont_rec = await _collect_continuation(account, session, cont_parent, model_type, thinking, search, ref_file_ids)
                if cont_rec is None:
                    break
                rec.extend_with(cont_rec)
                cont_parent = cont_rec.id or cont_parent
                if cont_rec.content:
                    if not role_sent:
                        role_sent = True
                        yield _sse(
                            {
                                "id": chunk_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": model,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {"role": "assistant"},
                                        "finish_reason": None,
                                    }
                                ],
                            }
                        )
                    if tool_mode:
                        content_parts.append(cont_rec.content)
                        content_buf = "".join(content_parts)
                        c_visible, content_shown_len, tool_hidden = toolemu.tool_visible(content_buf, content_shown_len, tool_hidden, tool_schemas)
                        allowed = content_piece(c_visible)
                        if allowed:
                            yield _sse(
                                {
                                    "id": chunk_id,
                                    "object": "chat.completion.chunk",
                                    "created": created,
                                    "model": model,
                                    "choices": [{"index": 0, "delta": {"content": allowed}, "finish_reason": None}],
                                }
                            )
                    else:
                        allowed = content_piece(cont_rec.content)
                        if allowed:
                            yield _sse(
                                {
                                    "id": chunk_id,
                                    "object": "chat.completion.chunk",
                                    "created": created,
                                    "model": model,
                                    "choices": [{"index": 0, "delta": {"content": allowed}, "finish_reason": None}],
                                }
                            )
                if cont_rec.reasoning:
                    if not role_sent:
                        role_sent = True
                        yield _sse(
                            {
                                "id": chunk_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": model,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {"role": "assistant"},
                                        "finish_reason": None,
                                    }
                                ],
                            }
                        )
                    yield _sse(
                        {
                            "id": chunk_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [{"index": 0, "delta": {"reasoning_content": cont_rec.reasoning}, "finish_reason": None}],
                        }
                    )
                if not _is_input_exceeds_limit(cont_rec):
                    incomplete_message = None
                    break
                incomplete_message = _incomplete_message(cont_rec)
                if not (cont_rec.content or cont_rec.reasoning):
                    break
            if incomplete_message is not None and not (rec.content or rec.reasoning):
                if reduced_prompts is None and messages is not None:
                    reduced_prompts = _reduced_prompt_variants(messages, tools, tool_choice, response_format, prompt)
                if reduced_prompts:
                    _drop_session(pool, account, session_key)
                    reduced = await _collect_reduced(account, pool, reduced_prompts, model_type, thinking, search, ref_file_ids)
                    if reduced is not None:
                        rec, session, session_key, variant_tool_mode, variant_tool_schemas = reduced
                        if variant_tool_mode:
                            tool_mode = variant_tool_mode
                            tool_schemas = variant_tool_schemas
                        response_message_id = rec.id or response_message_id
                        stop_message_id = response_message_id
                        reduced_notice = REDUCED_CONTEXT_MESSAGE
                        if rec.content:
                            if not role_sent:
                                role_sent = True
                                yield _sse(
                                    {
                                        "id": chunk_id,
                                        "object": "chat.completion.chunk",
                                        "created": created,
                                        "model": model,
                                        "choices": [
                                            {
                                                "index": 0,
                                                "delta": {"role": "assistant"},
                                                "finish_reason": None,
                                            }
                                        ],
                                    }
                                )
                            if tool_mode:
                                content_parts.append(rec.content)
                                content_buf = "".join(content_parts)
                                r_visible, content_shown_len, tool_hidden = toolemu.tool_visible(content_buf, content_shown_len, tool_hidden, tool_schemas)
                                allowed = content_piece(r_visible)
                                if allowed:
                                    yield _sse(
                                        {
                                            "id": chunk_id,
                                            "object": "chat.completion.chunk",
                                            "created": created,
                                            "model": model,
                                            "choices": [{"index": 0, "delta": {"content": allowed}, "finish_reason": None}],
                                        }
                                    )
                            else:
                                allowed = content_piece(rec.content)
                                if allowed:
                                    yield _sse(
                                        {
                                            "id": chunk_id,
                                            "object": "chat.completion.chunk",
                                            "created": created,
                                            "model": model,
                                            "choices": [{"index": 0, "delta": {"content": allowed}, "finish_reason": None}],
                                        }
                                    )
                        if rec.reasoning:
                            if not role_sent:
                                role_sent = True
                                yield _sse(
                                    {
                                        "id": chunk_id,
                                        "object": "chat.completion.chunk",
                                        "created": created,
                                        "model": model,
                                        "choices": [
                                            {
                                                "index": 0,
                                                "delta": {"role": "assistant"},
                                                "finish_reason": None,
                                            }
                                        ],
                                    }
                                )
                            yield _sse(
                                {
                                    "id": chunk_id,
                                    "object": "chat.completion.chunk",
                                    "created": created,
                                    "model": model,
                                    "choices": [{"index": 0, "delta": {"reasoning_content": rec.reasoning}, "finish_reason": None}],
                                }
                            )
        if incomplete_message is not None and reduced_notice is None:
            log.warning("deepseek response incomplete: %s", incomplete_message)
            for line in _stream_error_sse(chunk_id, created, model, incomplete_message, session_key, RESPONSE_INCOMPLETE, RESPONSE_INCOMPLETE):
                yield line
            return
        if not (rec.content or rec.reasoning) and rec.hint_error:
            if _is_fake_context_hint(rec):
                log.warning("deepseek fake context-length hint after retries")
                for line in _stream_error_sse(chunk_id, created, model, FAKE_CONTEXT_HINT_ERROR_MESSAGE, session_key, "server_error", "error"):
                    yield line
                return
            hint = rec.hint_error
            for line in _stream_error_sse(
                chunk_id,
                created,
                model,
                hint.get("message") or "DeepSeek server is busy, try again later",
                session_key,
                hint.get("finish_reason"),
            ):
                yield line
            return

        request_tokens = _advance_session_usage(session, rec.accumulated_tokens)
        usage = _deepseek_usage(request_tokens, prompt, rec.usage, completion_text=rec.content or rec.reasoning)
        account.sessions.touch_last_message(session_key, rec.id or response_message_id)
        record_usage(
            "deepseek",
            model,
            usage["prompt_tokens"],
            usage["completion_tokens"],
            usage["total_tokens"],
            user=user,
            session_id=session_key,
        )
        log.info("deepseek completion success (%.0fms)", (time.monotonic() - started) * 1000)

        if tool_mode:
            parsed = toolemu.parse_tool_calls(content_buf or rec.content, tool_schemas)
            if parsed is not None:
                tool_calls, _ = parsed
                if tool_calls:
                    if budget.done:
                        finish = "length"
                        for line in _stream_error_sse(
                            chunk_id,
                            created,
                            model,
                            "max_tokens reached before the tool call completed",
                            session_key,
                            "length",
                            "length",
                        ):
                            yield line
                        return
                    if _max_calls(parallel_tool_calls) is not None:
                        tool_calls = tool_calls[: _max_calls(parallel_tool_calls)]
                    for delta in toolemu.tool_call_deltas(tool_calls):
                        yield _sse(
                            {
                                "id": chunk_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": model,
                                "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
                            }
                        )
                    finish = "tool_calls"
                else:
                    raw_tail = (content_buf or rec.content)[content_shown_len:] if (content_buf or rec.content) else ""
                    tail_text = content_piece(raw_tail)
                    if tail_text:
                        yield _sse(
                            {
                                "id": chunk_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": model,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {"content": tail_text},
                                        "finish_reason": None,
                                    }
                                ],
                            }
                        )
                    finish = "length" if budget.done else _finish_reason(rec.status)
            else:
                raw_tail = (content_buf or rec.content)[content_shown_len:] if (content_buf or rec.content) else ""
                tail_text = content_piece(raw_tail)
                if tail_text:
                    yield _sse(
                        {
                            "id": chunk_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": tail_text},
                                    "finish_reason": None,
                                }
                            ],
                        }
                    )
                finish = "length" if budget.done else _finish_reason(rec.status)
        else:
            finish = "length" if budget.done else _finish_reason(rec.status)

        if not role_sent:
            role_sent = True
            yield _sse(
                {
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                }
            )

        if reduced_notice is not None:
            yield _sse(
                {
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "session_id": session_key,
                    "error": {"message": reduced_notice, "finish_reason": RESPONSE_INCOMPLETE},
                    "choices": [{"index": 0, "delta": {}, "finish_reason": RESPONSE_INCOMPLETE}],
                }
            )
        else:
            yield _sse(
                {
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "session_id": session_key,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
                }
            )
        if include_usage:
            yield _sse(
                {
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "session_id": session_key,
                    "usage": _usage_with_details(usage, rec.reasoning),
                    "choices": [],
                }
            )
        yield "data: [DONE]\n\n"


@app.api_route("/v1/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
async def unknown_v1_route(path: str) -> dict:
    raise HTTPException(404, f"Unknown /v1 endpoint: /v1/{path}")
