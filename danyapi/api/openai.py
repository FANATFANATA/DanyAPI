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
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
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

log = logging.getLogger("danyapi.api")

MODEL_TYPE_BY_NAME = {
    "deepseek-v4-flash": "default",
    "deepseek-v4-pro": "expert",
    "deepseek-v4-vision": "vision",
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
    "INCOMPLETE": "stop",
    "WIP": "stop",
    "TIMEOUT": "stop",
}

CONTEXT_LENGTH_STATUS = "CONTEXT_LENGTH_EXCEEDED"
INPUT_EXCEEDS_LIMIT = "input_exceeds_limit"
CONTINUE_PROMPT = "Continue"
MAX_CONTINUE_ROUNDS = 5
RESPONSE_INCOMPLETE = "response_incomplete"
RESPONSE_INCOMPLETE_MESSAGE = "Response is incomplete: provider errors interrupted the continuation, please retry"
REDUCED_CONTEXT_MESSAGE = "Response was generated from reduced context because the original input exceeded the model limit and may be incomplete"


class ChatMessage(BaseModel):
    role: str = "user"
    content: Any = ""
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None
    name: str | None = None

    @field_validator("role")
    @classmethod
    def validate_role(cls, v: str) -> str:
        allowed_roles = {"user", "assistant", "system", "tool", "function"}
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
    model: str = Field(default="deepseek-v4-flash")
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


class ImageGenerationRequest(BaseModel):
    model: str = Field(default="qwen-image-gen")
    prompt: str
    n: int = Field(default=1, ge=1, le=4)
    size: str | None = None
    response_format: str = Field(default="url")
    session_id: str | None = None
    user: str | None = None


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


@asynccontextmanager
async def lifespan(app: FastAPI):
    accounts: list[DeepSeekAccount] = []
    qwen_accounts: list[QwenAccount] = []
    cache_enabled = settings.cache_enabled
    deepseek_session_store = JsonStore("deepseek-sessions", "default" if cache_enabled else None)
    qwen_session_store = JsonStore("qwen-sessions", "default" if cache_enabled else None)
    deepseek_context_store = JsonStore("deepseek-contexts", "default" if cache_enabled else None)
    qwen_context_store = JsonStore("qwen-contexts", "default" if cache_enabled else None)
    deepseek_affinity_store = JsonStore("deepseek-affinities", "default" if cache_enabled else None)
    qwen_affinity_store = JsonStore("qwen-affinities", "default" if cache_enabled else None)
    try:
        if settings.deepseek_tokens:
            for i, token in enumerate(settings.deepseek_tokens):
                ds_client = DeepSeekClient(token=token, timeout=settings.timeout)
                if not await ds_client.check_auth():
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
            for i, token in enumerate(settings.qwen_tokens):
                qw_client = QwenClient(token=token, timeout=settings.timeout)
                if not await qw_client.check_auth():
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
        if not accounts and not qwen_accounts:
            raise RuntimeError("no valid credentials: set DEEPSEEK_TOKENS or QWEN_TOKENS")
        yield
    finally:
        for ds_acct in accounts:
            await ds_acct.client.aclose()
        for qw_acct in qwen_accounts:
            await qw_acct.client.aclose()


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
        elif chat_types:
            model_type = "chat"
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


async def _extract_request_model(request: Request) -> str | None:
    try:
        body = await request.body()
    except Exception:
        return None
    if not body:
        return None
    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError, TypeError):
        return None
    if isinstance(payload, dict):
        model = payload.get("model")
        if isinstance(model, str):
            return model
    return None


def _log_request_failure(request: Request, model: str | None, duration: float, status: int | None = None, exc: Exception | None = None) -> None:
    model_part = f"model={model}" if model else "model=?"
    if status is not None:
        reason = f"status={status}"
    else:
        reason = f"error={str(exc) if exc else 'unknown'}"
    log.warning(
        "%s %s failed: %s %s (%.0fms)",
        request.method,
        request.url.path,
        model_part,
        reason.replace("{", "{{").replace("}", "}}"),
        duration,
    )


def _log_request_success(request: Request, duration: float) -> None:
    log.info(
        "%s %s success (%.0fms)",
        request.method,
        request.url.path,
        duration,
    )


@app.middleware("http")
async def _log_request_failures(request: Request, call_next):
    started = time.monotonic()
    try:
        response = await call_next(request)
    except Exception as exc:
        _log_request_failure(
            request,
            await _extract_request_model(request),
            (time.monotonic() - started) * 1000,
            exc=exc,
        )
        raise
    if response.status_code >= 400:
        _log_request_failure(
            request,
            await _extract_request_model(request),
            (time.monotonic() - started) * 1000,
            status=response.status_code,
        )
    elif request.url.path == "/v1/chat/completions":
        _log_request_success(request, (time.monotonic() - started) * 1000)
    return response


MAX_FILES_PER_REQUEST = 50
MAX_FILE_SIZE = 100 * 1024 * 1024


@dataclass
class Attachment:
    data: bytes
    name: str
    content_type: str
    is_image: bool


def _split_data_uri(uri: str) -> tuple[str, bytes]:
    if not uri.startswith("data:"):
        raise HTTPException(400, "image_url must be a data URI (data:<mime>;base64,...)")
    meta, _, payload = uri[5:].partition(",")
    if not payload:
        raise HTTPException(400, "invalid data URI: missing base64 payload")
    content_type = meta.split(";", 1)[0] or "application/octet-stream"
    try:
        data = base64.b64decode(payload, validate=True)
    except ValueError as exc:
        raise HTTPException(400, "invalid base64 in image_url") from exc
    return content_type, data


def _collect_attachments(req: ChatCompletionRequest) -> list[Attachment]:
    attachments: list[Attachment] = []
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
                content_type, data = _split_data_uri(uri)
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


def _validate_attachments(attachments: list[Attachment], model_type: str) -> None:
    if not attachments:
        return
    if len(attachments) > MAX_FILES_PER_REQUEST:
        raise HTTPException(400, f"too many files: max {MAX_FILES_PER_REQUEST} per request")
    for att in attachments:
        if len(att.data) > MAX_FILE_SIZE:
            raise HTTPException(400, f"file {att.name} exceeds 100 MB limit")
    if model_type == "expert":
        raise HTTPException(400, "deepseek-v4-pro does not support file attachments")
    if model_type == "vision" and any(not att.is_image for att in attachments):
        raise HTTPException(400, "deepseek-v4-vision accepts images only")


async def _fresh_pow_upload_headers(account) -> dict:
    try:
        return await account.pow_upload.make_header(lambda: account.client.create_pow_challenge("/api/v0/file/upload_file"))
    except DeepSeekError as exc:
        _handle_account_error(account, exc)
        raise HTTPException(_deepseek_status(exc), _deepseek_error_detail(exc)) from exc


async def _upload_attachments(account, attachments: list[Attachment], model_type: str, thinking: bool) -> list[str]:
    file_ids: list[str] = []
    for att in attachments:
        pow_headers = await _fresh_pow_upload_headers(account)
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
        file_ids.append(file_id)
    return file_ids


def _resolve_model(model: str) -> str:
    model_type = MODEL_TYPE_BY_NAME.get(model)
    if model_type is not None:
        return model_type
    for suffix in REASONING_SUFFIXES:
        if model.endswith(suffix):
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


@app.get("/health")
async def health() -> dict:
    pool = getattr(app.state, "pool", None)
    qwen_pool = getattr(app.state, "qwen_pool", None)
    return {
        "status": "ok",
        "deepseek": pool is not None,
        "qwen": qwen_pool is not None,
        "deepseek_stats": _pool_stats(pool),
        "qwen_stats": _pool_stats(qwen_pool),
    }


@app.get("/v1/models")
async def list_models() -> dict:
    models: list[dict] = []
    for name, model_type in MODEL_TYPE_BY_NAME.items():
        models.append(
            {
                "id": name,
                "object": "model",
                "created": 0,
                "owned_by": "deepseek",
                "model_type": model_type,
            }
        )
        for suffix in REASONING_SUFFIXES:
            models.append(
                {
                    "id": f"{name}{suffix}",
                    "object": "model",
                    "created": 0,
                    "owned_by": "deepseek",
                    "model_type": model_type,
                }
            )
    qwen_models: list[dict] = getattr(app.state, "qwen_models", [])
    for model in qwen_models:
        models.append(
            {
                "id": model["id"],
                "object": "model",
                "created": 0,
                "owned_by": model.get("owned_by", "qwen"),
                "name": model.get("name"),
                "model_type": model.get("model_type", "chat"),
            }
        )
    return {"object": "list", "data": models}


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


async def _human_delay() -> None:
    delay = random.uniform(settings.human_delay_min, settings.human_delay_max)
    if delay > 0:
        await asyncio.sleep(delay)


def _resolve_provider(model: str) -> str:
    if model.startswith("qwen"):
        return "qwen"
    if model in MODEL_TYPE_BY_NAME or model.startswith("deepseek"):
        return "deepseek"
    qwen_models = getattr(app.state, "qwen_models", [])
    for m in qwen_models:
        if m.get("id") == model:
            return "qwen"
    raise HTTPException(404, f"Unknown model: {model}")


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest) -> Any:
    provider = _resolve_provider(req.model)
    if provider == "qwen":
        return await _chat_completions_qwen(req)
    return await _chat_completions_deepseek(req)


@app.post("/v1/images/generations")
async def image_generations(req: ImageGenerationRequest) -> dict:
    pool: AccountPool = app.state.qwen_pool
    if pool is None:
        raise HTTPException(503, "qwen provider is not configured (required for image generation)")

    dims = _parse_image_size(req.size)

    account, existing_sid = await _acquire_account(pool, req.session_id)

    try:
        result = await qwen_api.collect_image(
            account=account,
            pool=pool,
            existing_sid=existing_sid,
            lock=account.sem,
            prompt=req.prompt,
            model=req.model,
            model_id=req.model,
        )
    except AccountPoolBusy:
        raise HTTPException(429, "all accounts are busy, try again later") from None

    want_b64 = req.response_format == "b64_json"
    data: list[dict] = []
    for url in result["image_urls"]:
        if not (want_b64 or dims):
            data.append({"url": url})
            continue
        try:
            async with httpx.AsyncClient(follow_redirects=True, timeout=30) as hc:
                img_resp = await hc.get(url)
            if img_resp.status_code != 200:
                log.warning("image download failed (%s) for %s, returning url", img_resp.status_code, url)
                data.append({"url": url})
                continue
            payload_bytes = _resize_image_bytes(img_resp.content, dims)
            data.append({"b64_json": base64.b64encode(payload_bytes).decode()})
        except Exception as exc:
            log.warning("image fetch failed for %s, returning url: %s", url, exc)
            data.append({"url": url})

    if not data:
        data.append({"url": "", "revised_prompt": result.get("revised_prompt", "")})

    return {
        "created": int(time.time()),
        "data": data,
        "usage": result.get("usage"),
        "session_id": result.get("session_id"),
    }


async def _acquire_account(pool: AccountPool, session_id: str | None):
    try:
        return await pool.acquire(session_id, settings.acquire_timeout)
    except AccountPoolBusy:
        raise HTTPException(429, "all accounts are busy, try again later") from None
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc


def _can_reuse_session(account: Any, session_id: str | None, **kwargs: Any) -> bool:
    return bool(account.sessions.can_reuse(session_id, **kwargs))


def _include_usage(req: ChatCompletionRequest) -> bool:
    opts = getattr(req, "stream_options", None)
    if not isinstance(opts, dict):
        return False
    return bool(opts.get("include_usage"))


def _deepseek_usage(total: int) -> dict:
    value = max(0, int(total or 0))
    return {"prompt_tokens": 0, "completion_tokens": value, "total_tokens": value}


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
) -> tuple[str, str, str]:
    error: dict = {"message": message}
    if error_finish is not None:
        error["finish_reason"] = error_finish
    error_chunk = _sse(
        {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "error": error,
            "choices": [{"index": 0, "delta": {}, "finish_reason": choice_finish or error_finish or "error"}],
        }
    )
    tail_chunk = _sse(
        {
            "id": chunk_id,
            "session_id": session_key,
            "object": "chat.completion.chunk",
            "choices": [],
        }
    )
    return error_chunk, tail_chunk, "data: [DONE]\n\n"


async def _stream_guard(gen, model: str):
    chunk_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    try:
        async for item in gen:
            yield item
    except AccountPoolBusy:
        for line in _stream_error_sse(chunk_id, created, model, "all accounts are busy, try again later"):
            yield line
    except Exception as exc:
        log.exception("stream generator failed: %s", exc)
        msg = str(exc) or repr(exc) or "unknown stream error"
        for line in _stream_error_sse(chunk_id, created, model, f"stream error: {msg}"):
            yield line


async def _chat_completions_deepseek(req: ChatCompletionRequest) -> Any:
    pool: AccountPool = app.state.pool
    if pool is None:
        raise HTTPException(503, "deepseek provider is not configured")

    model_type = _resolve_model(req.model)
    thinking = req.thinking if req.thinking is not None else _is_reasoning_model(req.model)
    search = bool(req.search) and model_type == "default"

    context_seq = toolemu.context_sequence(req.messages, user=getattr(req, "user", None))
    if req.session_id:
        account, existing_sid = await _acquire_account(pool, req.session_id)
        if existing_sid is None:
            existing_sid = req.session_id
    else:
        cached_sid = pool.resolve_context(context_seq) if context_seq else None
        account, existing_sid = await _acquire_account(pool, cached_sid)
    has_session = _can_reuse_session(account, existing_sid)

    try:
        prompt, tool_mode = toolemu.build_prompt(
            req.messages,
            getattr(req, "tools", None),
            getattr(req, "tool_choice", None),
            has_session,
            getattr(req, "response_format", None),
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    attachments = _collect_attachments(req)
    _validate_attachments(attachments, model_type)
    ref_file_ids = None
    if attachments:
        ref_file_ids = await _upload_attachments(account, attachments, model_type, thinking)

    common = {
        "account": account,
        "pool": pool,
        "existing_sid": existing_sid,
        "prompt": prompt,
        "model": req.model,
        "model_type": model_type,
        "thinking": thinking,
        "search": search,
        "ref_file_ids": ref_file_ids,
        "tool_schemas": toolemu.tool_schema_map(getattr(req, "tools", None)),
        "tool_mode": tool_mode,
        "include_usage": _include_usage(req),
        "context_seq": context_seq,
        "reduced_prompts": _reduced_prompt_variants(
            req.messages, getattr(req, "tools", None), getattr(req, "tool_choice", None), getattr(req, "response_format", None), prompt
        ),
        "messages": req.messages,
        "tools": getattr(req, "tools", None),
        "tool_choice": getattr(req, "tool_choice", None),
        "response_format": getattr(req, "response_format", None),
    }
    if req.stream:
        return StreamingResponse(
            _stream_guard(_stream_openai(lock=account.sem, **common), req.model),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    try:
        return await _collect_non_stream(lock=account.sem, **common)
    except AccountPoolBusy:
        raise HTTPException(429, "all accounts are busy, try again later") from None


async def _chat_completions_qwen(req: ChatCompletionRequest) -> Any:
    pool: AccountPool = app.state.qwen_pool
    if pool is None:
        raise HTTPException(503, "qwen provider is not configured")

    thinking = req.thinking if req.thinking is not None else True
    search = bool(req.search)

    context_seq = toolemu.context_sequence(req.messages, user=getattr(req, "user", None))
    if req.session_id:
        account, existing_sid = await _acquire_account(pool, req.session_id)
        if existing_sid is None:
            existing_sid = req.session_id
    else:
        cached_sid = pool.resolve_context(context_seq) if context_seq else None
        account, existing_sid = await _acquire_account(pool, cached_sid)
    has_session = _can_reuse_session(account, existing_sid, model=req.model)

    try:
        prompt, tool_mode = toolemu.build_prompt(
            req.messages,
            getattr(req, "tools", None),
            getattr(req, "tool_choice", None),
            has_session,
            getattr(req, "response_format", None),
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    common = {
        "account": account,
        "pool": pool,
        "existing_sid": existing_sid,
        "prompt": prompt,
        "model": req.model,
        "model_id": req.model,
        "thinking": thinking,
        "search": search,
        "tool_schemas": toolemu.tool_schema_map(getattr(req, "tools", None)),
        "tool_mode": tool_mode,
        "include_usage": _include_usage(req),
        "context_seq": context_seq,
        "messages": req.messages,
        "tools": getattr(req, "tools", None),
        "tool_choice": getattr(req, "tool_choice", None),
        "response_format": getattr(req, "response_format", None),
    }
    if req.stream:
        return StreamingResponse(
            _stream_guard(qwen_api.stream_openai(lock=account.sem, **common), req.model),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    try:
        return await qwen_api.collect_non_stream(lock=account.sem, **common)
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
        raise HTTPException(exc.response.status_code, exc.response.text[:500]) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(502, f"DeepSeek request failed: {exc}") from exc

    if resp.status_code != 200:
        body = await resp.aread()
        await resp.aclose()
        raise HTTPException(resp.status_code, body[:500].decode("utf-8", errors="replace"))

    content_type = resp.headers.get("content-type", "")
    if "text/event-stream" not in content_type:
        body = await resp.aread()
        await resp.aclose()
        try:
            payload = json.loads(body)
        except json.JSONDecodeError as exc:
            raise HTTPException(502, body[:500].decode("utf-8", errors="replace")) from exc
        data = payload.get("data") or {}
        if data.get("biz_code"):
            code = data["biz_code"]
            status = 401 if code in DEEPSEEK_AUTH_ERROR_CODES else 502
            raise HTTPException(status, f"DeepSeek error {code}: {data.get('biz_msg')}")
        if payload.get("code"):
            code = payload["code"]
            status = 401 if code in DEEPSEEK_AUTH_ERROR_CODES else 502
            raise HTTPException(
                status,
                f"DeepSeek error {code}: {payload.get('msg') or payload.get('message')}",
            )
        raise HTTPException(502, "unexpected non-stream response")
    return resp


def _is_retryable_hint(rec: MessageReconstructor) -> bool:
    hint = rec.hint_error
    return bool(hint and hint.get("finish_reason") in RETRYABLE_FINISH_REASONS)


RETRYABLE_HTTP_STATUSES = {408, 425, 429, 500, 502, 503, 504}
STALE_SESSION_STATUSES = {400, 404}


def _retry_delay(attempt: int) -> float:
    return min(RETRY_BACKOFF_SEC * (2 ** (attempt - 1)), RETRY_BACKOFF_MAX_SEC)


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
) -> tuple[dict, str]:
    if tool_mode:
        parsed = toolemu.parse_tool_calls(content, tool_schemas)
        if parsed is not None:
            tool_calls, tool_text = parsed
            if tool_calls:
                return toolemu.format_tool_message(tool_calls, tool_text, reasoning), "tool_calls"
    message = {"role": "assistant", "content": content}
    if reasoning:
        message["reasoning_content"] = reasoning
    return message, "stop"


def _build_completion_response(
    model: str,
    message: dict,
    finish: str,
    session,
    session_key: str | None,
) -> dict:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish,
            }
        ],
        "usage": _deepseek_usage(getattr(session, "accumulated_tokens", 0)),
        "session_id": session_key,
    }


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
    while True:
        try:
            pow_headers = await _fresh_pow_headers(account)
            resp = await _send_with_auth(
                account,
                account.client,
                pow_headers,
                session.id,
                parent_message_id,
                CONTINUE_PROMPT,
                model_type,
                thinking,
                search,
                ref_file_ids,
            )
        except HTTPException as exc:
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
        rec = MessageReconstructor()
        incremental = IncrementalSSE()
        try:
            async for chunk in resp.aiter_bytes():
                for event in incremental.feed(chunk):
                    rec.handle(event)
            for event in incremental.finish():
                rec.handle(event)
        finally:
            await resp.aclose()
        if not (rec.content or rec.reasoning) and _is_retryable_hint(rec) and attempt < MAX_RETRIES:
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
    for prompt, _tool_mode, _tool_schemas in reduced_prompts:
        session_key = None
        try:
            session, session_key, parent_message_id = await _prepare_session(account, pool, None, None)
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
            try:
                async for chunk in resp.aiter_bytes():
                    for event in incremental.feed(chunk):
                        rec.handle(event)
                for event in incremental.finish():
                    rec.handle(event)
            finally:
                await resp.aclose()
        except (HTTPException, httpx.HTTPError):
            if session_key is not None:
                _drop_session(pool, account, session_key)
            continue
        if (rec.content or rec.reasoning) and not _is_input_exceeds_limit(rec):
            return rec, session, session_key
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
    tool_mode=False,
    tool_schemas=None,
    include_usage=False,
    context_seq: tuple[str, ...] | None = None,
    reduced_prompts: list[tuple[str, bool, dict[str, Any]]] | None = None,
    messages=None,
    tools=None,
    tool_choice=None,
    response_format=None,
):
    await _human_delay()
    async with account_lock(lock, settings.acquire_timeout):
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

        try:
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
                        except (ValueError, TypeError, AttributeError) as build_exc:
                            raise exc from build_exc
                        log.warning("deepseek session %s is stale (%s), rebuilt full history into a fresh chat", session_key, exc.status_code)
                        session, session_key, parent_message_id = await _prepare_session(account, pool, existing_sid, context_seq)
                        stop_message_id = None
                        response_message_id = None
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

                rec = MessageReconstructor()
                incremental = IncrementalSSE()
                response_message_id = None
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
                    await _try_stop_stream(account.client, session.id, stop_message_id)
                    raise HTTPException(502, f"Stream processing failed: {exc}") from exc
                finally:
                    try:
                        await resp.aclose()
                    except Exception:
                        await _try_stop_stream(account.client, session.id, stop_message_id)
                        raise

                if rec.id:
                    stop_message_id = rec.id
                if not (rec.content or rec.reasoning) and _is_retryable_hint(rec) and attempt < MAX_RETRIES:
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
            if incomplete_message is not None and not (rec.content or rec.reasoning) and reduced_prompts:
                _drop_session(pool, account, session_key)
                reduced = await _collect_reduced(account, pool, reduced_prompts, model_type, thinking, search, ref_file_ids)
                if reduced is not None:
                    rec, session, session_key = reduced
                    response_message_id = rec.id or response_message_id
                    stop_message_id = response_message_id
                    reduced_notice = REDUCED_CONTEXT_MESSAGE
        if incomplete_message is not None and reduced_notice is None:
            log.warning("deepseek response incomplete: %s", incomplete_message)
            raise HTTPException(502, _incomplete_error_body(incomplete_message))
        content = rec.content
        reasoning = rec.reasoning
        if not (content or reasoning) and rec.hint_error:
            raise HTTPException(429, _busy_error_body(rec))
        session.accumulated_tokens = max(getattr(session, "accumulated_tokens", 0) or 0, rec.accumulated_tokens)
        account.sessions.touch_last_message(session_key, rec.id or response_message_id)
        log.info("deepseek completion success (%.0fms)", (time.monotonic() - started) * 1000)
        message, finish = _build_assistant_message(content, reasoning, tool_mode, tool_schemas)
        if finish == "stop":
            finish = _finish_reason(rec.status)
        response = _build_completion_response(model, message, finish, session, session_key)
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
    tool_mode=False,
    tool_schemas=None,
    include_usage=False,
    context_seq: tuple[str, ...] | None = None,
    reduced_prompts: list[tuple[str, bool, dict[str, Any]]] | None = None,
    messages=None,
    tools=None,
    tool_choice=None,
    response_format=None,
):
    chunk_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    await _human_delay()
    async with account_lock(lock, settings.acquire_timeout):
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
        content_buf = ""
        started = time.monotonic()
        had_cached_session = bool(existing_sid) and account.sessions.get(existing_sid) is not None
        stale_rebuilt = False
        attempt = 0
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
                                content_buf += c_diff
                            else:
                                delta["content"] = c_diff
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
                except Exception:
                    if not stopped:
                        await _try_stop_stream(account.client, session.id, stop_message_id)
                    raise
            if got_content:
                break
            if _is_retryable_hint(rec) and attempt < MAX_RETRIES:
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
                    if tool_mode:
                        content_buf += cont_rec.content
                    else:
                        yield _sse(
                            {
                                "id": chunk_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": model,
                                "choices": [{"index": 0, "delta": {"content": cont_rec.content}, "finish_reason": None}],
                            }
                        )
                if cont_rec.reasoning:
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
            if incomplete_message is not None and not (rec.content or rec.reasoning) and reduced_prompts:
                _drop_session(pool, account, session_key)
                reduced = await _collect_reduced(account, pool, reduced_prompts, model_type, thinking, search, ref_file_ids)
                if reduced is not None:
                    rec, session, session_key = reduced
                    response_message_id = rec.id or response_message_id
                    stop_message_id = response_message_id
                    reduced_notice = REDUCED_CONTEXT_MESSAGE
                    if rec.content:
                        yield _sse(
                            {
                                "id": chunk_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": model,
                                "choices": [{"index": 0, "delta": {"content": rec.content}, "finish_reason": None}],
                            }
                        )
                    if rec.reasoning:
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

        session.accumulated_tokens = max(getattr(session, "accumulated_tokens", 0) or 0, rec.accumulated_tokens)
        account.sessions.touch_last_message(session_key, rec.id or response_message_id)
        log.info("deepseek completion success (%.0fms)", (time.monotonic() - started) * 1000)

        if tool_mode:
            parsed = toolemu.parse_tool_calls(content_buf or rec.content, tool_schemas)
            if parsed is not None:
                tool_calls, tool_text = parsed
                if tool_calls:
                    for delta in toolemu.tool_call_deltas(tool_calls, tool_text):
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
                    if content_buf:
                        yield _sse(
                            {
                                "id": chunk_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": model,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {"content": content_buf},
                                        "finish_reason": None,
                                    }
                                ],
                            }
                        )
                    finish = _finish_reason(rec.status)
            else:
                if content_buf:
                    yield _sse(
                        {
                            "id": chunk_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": content_buf},
                                    "finish_reason": None,
                                }
                            ],
                        }
                    )
                finish = _finish_reason(rec.status)
        else:
            finish = _finish_reason(rec.status)

        if reduced_notice is not None:
            yield _sse(
                {
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
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
                    "choices": [],
                    "usage": _deepseek_usage(getattr(session, "accumulated_tokens", 0)),
                }
            )
        yield _sse(
            {
                "id": chunk_id,
                "session_id": session_key,
                "object": "chat.completion.chunk",
                "choices": [],
            }
        )
        yield "data: [DONE]\n\n"
