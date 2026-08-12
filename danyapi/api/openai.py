from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ..accounts import AccountPool, DeepSeekAccount
from ..config import settings
from ..deepseek.client import DeepSeekClient, DeepSeekError, DeepSeekSession
from ..deepseek.stream import IncrementalSSE, MessageReconstructor

log = logging.getLogger("danyapi.api")

MODEL_TYPE_BY_NAME = {
    "deepseek-chat": "default",
    "deepseek-reasoner": "expert",
    "deepseek-vision": "vision",
}

STATUS_TO_FINISH_REASON = {
    "FINISHED": "stop",
    "CONTEXT_LENGTH_EXCEEDED": "length",
    "CONTENT_FILTER": "content_filter",
    "INCOMPLETE": "stop",
    "WIP": "stop",
    "TIMEOUT": "stop",
}


class ChatMessage(BaseModel):
    role: str = "user"
    content: Any = ""


class ChatCompletionRequest(BaseModel):
    model: str = Field(default="deepseek-chat")
    messages: list[ChatMessage] = Field(default_factory=list)
    stream: bool = False
    temperature: float | None = None
    top_p: float | None = None
    thinking: bool | None = None
    search: bool | None = None
    session_id: str | None = None
    user: str | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    accounts: list[DeepSeekAccount] = []
    try:
        if settings.deepseek_tokens:
            for i, token in enumerate(settings.deepseek_tokens):
                client = DeepSeekClient(token=token, timeout=settings.timeout)
                if not await client.check_auth():
                    log.warning("token #%d invalid/expired, skipping", i)
                    await client.aclose()
                    continue
                accounts.append(DeepSeekAccount(len(accounts), client))
            log.info("deepseek accounts ready: %d", len(accounts))
        elif settings.deepseek_email:
            client = DeepSeekClient(timeout=settings.timeout)
            await client.login(
                email=settings.deepseek_email,
                password=settings.deepseek_password,
            )
            accounts.append(DeepSeekAccount(0, client))
            log.info("deepseek login ok, device_id=%s", client.device_id)
        if not accounts:
            raise RuntimeError("no valid DeepSeek credentials: set DEEPSEEK_TOKENS or DEEPSEEK_EMAIL/DEEPSEEK_PASSWORD")
        app.state.pool = AccountPool(accounts)
        yield
    finally:
        for acct in accounts:
            await acct.client.aclose()


app = FastAPI(title="DanyAPI", lifespan=lifespan)


def _extract_prompt(messages: list[ChatMessage]) -> str:
    if not messages:
        raise HTTPException(400, "messages is required")
    last_user: ChatMessage | None = None
    for msg in reversed(messages):
        if msg.role in ("user", "system"):
            last_user = msg
            break
    if last_user is None:
        raise HTTPException(400, "no user message found")
    content = last_user.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if item.get("type") == "text" and isinstance(item.get("text"), str):
                    parts.append(item["text"])
                elif item.get("type") == "image_url":
                    raise HTTPException(
                        400,
                        "image_url input requires file upload support, not implemented yet",
                    )
        return "".join(parts).strip()
    raise HTTPException(400, "unsupported message content")


def _resolve_model(model: str) -> str:
    model_type = MODEL_TYPE_BY_NAME.get(model)
    if model_type is None:
        raise HTTPException(404, f"Unknown model: {model}")
    return model_type


def _finish_reason(status: Any) -> str:
    if isinstance(status, str):
        return STATUS_TO_FINISH_REASON.get(status, "stop")
    return "stop"


@app.get("/v1/models")
async def list_models() -> dict:
    models = [
        {
            "id": name,
            "object": "model",
            "created": 0,
            "owned_by": "deepseek",
            "model_type": model_type,
        }
        for name, model_type in MODEL_TYPE_BY_NAME.items()
    ]
    return {"object": "list", "data": models}


RETRYABLE_FINISH_REASONS = {
    "expert_busy_use_default",
    "parallel_chat_limit",
    "server_busy",
    "busy",
}
MAX_RETRIES = 3
RETRY_BACKOFF_SEC = 1.0


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest) -> Any:
    pool: AccountPool = app.state.pool

    model_type = _resolve_model(req.model)
    thinking = req.thinking if req.thinking is not None else (model_type == "expert")
    search = bool(req.search)
    prompt = _extract_prompt(req.messages)

    account, existing_sid = await pool.acquire(req.session_id)

    common = {
        "account": account,
        "pool": pool,
        "existing_sid": existing_sid,
        "prompt": prompt,
        "model": req.model,
        "model_type": model_type,
        "thinking": thinking,
        "search": search,
    }
    if req.stream:
        return StreamingResponse(
            _stream_openai(lock=account.sem, **common),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return await _collect_non_stream(lock=account.sem, **common)


async def _prepare_session(account: DeepSeekAccount, pool: AccountPool, existing_sid: str | None) -> tuple[DeepSeekSession, str, str | None]:
    try:
        session, session_key = await account.sessions.obtain(existing_sid)
    except DeepSeekError as exc:
        _handle_account_error(account, exc)
        raise HTTPException(401, f"DeepSeek auth error: {exc}") from exc
    if existing_sid is None:
        pool.register(account.index, session_key)
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
):
    try:
        resp = await client.completion(
            chat_session_id=session_id,
            prompt=prompt,
            parent_message_id=parent_message_id,
            model_type=model_type,
            thinking_enabled=thinking,
            search_enabled=search,
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
            raise HTTPException(502, f"DeepSeek error {data['biz_code']}: {data.get('biz_msg')}")
        if payload.get("code"):
            raise HTTPException(
                502,
                f"DeepSeek error {payload['code']}: {payload.get('msg') or payload.get('message')}",
            )
        raise HTTPException(502, "unexpected non-stream response")
    return resp


def _is_retryable_hint(rec: MessageReconstructor) -> bool:
    hint = rec.hint_error
    return bool(hint and hint.get("finish_reason") in RETRYABLE_FINISH_REASONS)


def _handle_account_error(account: DeepSeekAccount, exc: Exception) -> None:
    code = getattr(exc, "biz_code", None)
    if code in (40001, 40002, 40003, 40012, 40029):
        account.mark_broken()
        log.warning("account #%d auth error %s: %s", account.index, code, exc)
    else:
        log.warning("account #%d error: %s", account.index, exc)


async def _fresh_pow_headers(account) -> dict:
    try:
        return await account.pow.make_header(account.client.create_pow_challenge)
    except DeepSeekError as exc:
        _handle_account_error(account, exc)
        raise HTTPException(401, f"DeepSeek auth error: {exc}") from exc


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
):
    async with lock:
        session, session_key, parent_message_id = await _prepare_session(account, pool, existing_sid)
        rec: MessageReconstructor | None = None
        response_message_id = None
        for attempt in range(MAX_RETRIES + 1):
            if attempt:
                await asyncio.sleep(RETRY_BACKOFF_SEC * (2 ** (attempt - 1)))
            pow_headers = await _fresh_pow_headers(account)
            resp = await _send_completion(
                account.client,
                pow_headers,
                session.id,
                parent_message_id,
                prompt,
                model_type,
                thinking,
                search,
            )
            rec = MessageReconstructor()
            incremental = IncrementalSSE()
            response_message_id = None
            try:
                async for chunk in resp.aiter_bytes():
                    for event in incremental.feed(chunk):
                        if event.event == "ready" and isinstance(event.data, dict):
                            response_message_id = event.data.get("response_message_id")
                        rec.handle(event)
                for event in incremental.finish():
                    rec.handle(event)
            finally:
                await resp.aclose()
            if not (rec.content or rec.reasoning) and _is_retryable_hint(rec) and attempt < MAX_RETRIES:
                log.warning(
                    "retryable hint (%s), attempt %d/%d",
                    (rec.hint_error or {}).get("finish_reason"),
                    attempt + 1,
                    MAX_RETRIES,
                )
                continue
            break

        assert rec is not None
        account.sessions.touch_last_message(session_key, rec.id or response_message_id)

        if not (rec.content or rec.reasoning) and rec.hint_error:
            raise HTTPException(429, _busy_error_body(rec))

        content = rec.content
        reasoning = rec.reasoning
        finish = _finish_reason(rec.status)
        message = {"role": "assistant", "content": content}
        if reasoning:
            message["reasoning_content"] = reasoning
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
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "session_id": session_key,
        }


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
):
    chunk_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    def sse(data: dict) -> str:
        return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

    async with lock:
        try:
            session, session_key, parent_message_id = await _prepare_session(account, pool, existing_sid)
        except HTTPException as exc:
            detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
            yield sse(
                {
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "error": {"message": detail},
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "error"}],
                }
            )
            yield sse(
                {
                    "id": chunk_id,
                    "session_id": None,
                    "object": "chat.completion.chunk",
                    "choices": [],
                }
            )
            yield "data: [DONE]\n\n"
            return

        rec: MessageReconstructor | None = None
        response_message_id = None
        for attempt in range(MAX_RETRIES + 1):
            if attempt:
                await asyncio.sleep(RETRY_BACKOFF_SEC * (2 ** (attempt - 1)))
            pow_headers = await _fresh_pow_headers(account)
            resp = await _send_completion(
                account.client,
                pow_headers,
                session.id,
                parent_message_id,
                prompt,
                model_type,
                thinking,
                search,
            )
            rec = MessageReconstructor()
            incremental = IncrementalSSE()
            response_message_id = None
            got_content = False
            pending: list[str] = []
            try:
                async for chunk in resp.aiter_bytes():
                    for event in incremental.feed(chunk):
                        if event.event == "ready" and isinstance(event.data, dict):
                            response_message_id = event.data.get("response_message_id")
                        rec.handle(event)
                        c_diff, r_diff = rec.take_diffs()
                        if c_diff or r_diff:
                            got_content = True
                        if not pending:
                            pending.append(
                                sse(
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
                            )
                        delta: dict = {}
                        if c_diff:
                            delta["content"] = c_diff
                        if r_diff:
                            delta["reasoning_content"] = r_diff
                        if delta:
                            pending.append(
                                sse(
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
                            )
                        if got_content:
                            for line in pending:
                                yield line
                            pending.clear()
            finally:
                await resp.aclose()
            if got_content:
                break
            if _is_retryable_hint(rec) and attempt < MAX_RETRIES:
                log.warning(
                    "retryable hint (%s), attempt %d/%d",
                    (rec.hint_error or {}).get("finish_reason"),
                    attempt + 1,
                    MAX_RETRIES,
                )
                continue
            break

        assert rec is not None
        if not (rec.content or rec.reasoning) and rec.hint_error:
            hint = rec.hint_error
            yield sse(
                {
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "error": {
                        "message": hint.get("message") or "DeepSeek server is busy, try again later",
                        "finish_reason": hint.get("finish_reason"),
                    },
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": hint.get("finish_reason") or "error",
                        }
                    ],
                }
            )
            yield sse(
                {
                    "id": chunk_id,
                    "session_id": session_key,
                    "object": "chat.completion.chunk",
                    "choices": [],
                }
            )
            yield "data: [DONE]\n\n"
            return

        account.sessions.touch_last_message(session_key, rec.id or response_message_id)
        finish = _finish_reason(rec.status)
        yield sse(
            {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
            }
        )
        yield sse(
            {
                "id": chunk_id,
                "session_id": session_key,
                "object": "chat.completion.chunk",
                "choices": [],
            }
        )
        yield "data: [DONE]\n\n"
