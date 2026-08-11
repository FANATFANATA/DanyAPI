"""OpenAI-совместимый HTTP API поверх DeepSeek web-клиента.

Эндпоинты:
  GET  /v1/models
  POST /v1/chat/completions
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Optional

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ..accounts import AccountPool, DeepSeekAccount
from ..config import settings
from ..deepseek.client import DeepSeekClient
from ..deepseek.stream import IncrementalSSE, MessageReconstructor

log = logging.getLogger("danyapi.api")

MODEL_TYPE_BY_NAME = {
    "deepseek-chat": "default",
    "deepseek-reasoner": "expert",
    "deepseek-vision": "vision",
}
NAME_BY_MODEL_TYPE = {v: k for k, v in MODEL_TYPE_BY_NAME.items()}

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
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    thinking: Optional[bool] = None
    search: Optional[bool] = None
    session_id: Optional[str] = None
    user: Optional[str] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    accounts: list[DeepSeekAccount] = []
    try:
        if settings.deepseek_tokens:
            for i, token in enumerate(settings.deepseek_tokens):
                client = DeepSeekClient(token=token, timeout=settings.timeout)
                accounts.append(DeepSeekAccount(i, client))
            log.info("deepseek accounts from tokens: %d", len(accounts))
        elif settings.deepseek_email:
            client = DeepSeekClient(timeout=settings.timeout)
            await client.login(
                email=settings.deepseek_email,
                password=settings.deepseek_password,
            )
            accounts.append(DeepSeekAccount(0, client))
            log.info("deepseek login ok, device_id=%s", client.device_id)
        if not accounts:
            raise RuntimeError(
                "no DeepSeek credentials: set DEEPSEEK_TOKENS or DEEPSEEK_EMAIL/DEEPSEEK_PASSWORD"
            )
        app.state.pool = AccountPool(accounts)
        yield
    finally:
        for acct in accounts:
            await acct.client.aclose()


app = FastAPI(title="DanyAPI", lifespan=lifespan)


def _extract_prompt(messages: list[ChatMessage]) -> str:
    if not messages:
        raise HTTPException(400, "messages is required")
    last_user: Optional[ChatMessage] = None
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


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest) -> Any:
    pool: AccountPool = app.state.pool

    model_type = _resolve_model(req.model)
    thinking = req.thinking if req.thinking is not None else (model_type == "expert")
    search = bool(req.search)
    prompt = _extract_prompt(req.messages)

    account, existing_sid = await pool.acquire(req.session_id)
    session, session_key = await account.sessions.obtain(existing_sid)
    if existing_sid is None:
        pool.register(account.index, session_key)
    parent_message_id = session.last_message_id

    pow_headers = await account.pow.make_header(account.client.create_pow_challenge)

    common = dict(
        account=account,
        pow_headers=pow_headers,
        session_id=session.id,
        session_key=session_key,
        parent_message_id=parent_message_id,
        prompt=prompt,
        model=req.model,
        model_type=model_type,
        thinking=thinking,
        search=search,
    )
    if req.stream:
        return StreamingResponse(
            _stream_openai(lock=account.sem, **common),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return await _collect_non_stream(lock=account.sem, **common)


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

    if resp.status_code != 200:
        body = await resp.aread()
        raise HTTPException(
            resp.status_code, body[:500].decode("utf-8", errors="replace")
        )

    content_type = resp.headers.get("content-type", "")
    if "text/event-stream" not in content_type:
        body = await resp.aread()
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            raise HTTPException(502, body[:500].decode("utf-8", errors="replace"))
        data = payload.get("data") or {}
        if data.get("biz_code"):
            raise HTTPException(
                502, f"DeepSeek error {data['biz_code']}: {data.get('biz_msg')}"
            )
        if payload.get("code"):
            raise HTTPException(
                502,
                f"DeepSeek error {payload['code']}: {payload.get('msg') or payload.get('message')}",
            )
        raise HTTPException(502, "unexpected non-stream response")
    return resp


async def _collect_non_stream(
    account,
    session_key,
    lock,
    pow_headers,
    session_id,
    parent_message_id,
    prompt,
    model,
    model_type,
    thinking,
    search,
):
    async with lock:
        resp = await _send_completion(
            account.client,
            pow_headers,
            session_id,
            parent_message_id,
            prompt,
            model_type,
            thinking,
            search,
        )
        try:
            reconstructor = MessageReconstructor()
            incremental = IncrementalSSE()
            response_message_id = None
            async for chunk in resp.aiter_bytes():
                for event in incremental.feed(chunk):
                    if event.event == "ready" and isinstance(event.data, dict):
                        response_message_id = event.data.get("response_message_id")
                    reconstructor.handle(event)
            for event in incremental.finish():
                reconstructor.handle(event)
        finally:
            await resp.aclose()

    account.sessions.touch_last_message(
        session_key, reconstructor.id or response_message_id
    )

    content = reconstructor.content
    reasoning = reconstructor.reasoning
    finish = _finish_reason(reconstructor.status)
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
    session_key,
    lock,
    pow_headers,
    session_id,
    parent_message_id,
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
        resp = await _send_completion(
            account.client,
            pow_headers,
            session_id,
            parent_message_id,
            prompt,
            model_type,
            thinking,
            search,
        )
        reconstructor = MessageReconstructor()
        incremental = IncrementalSSE()
        response_message_id = None
        sent_role = False
        try:
            async for chunk in resp.aiter_bytes():
                for event in incremental.feed(chunk):
                    if event.event == "ready" and isinstance(event.data, dict):
                        response_message_id = event.data.get("response_message_id")
                    if event.event in ("toast", "hint"):
                        continue
                    reconstructor.handle(event)
                    if not sent_role:
                        yield sse(
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
                        sent_role = True
                    c_diff, r_diff = reconstructor.take_diffs()
                    delta: dict = {}
                    if c_diff:
                        delta["content"] = c_diff
                    if r_diff:
                        delta["reasoning_content"] = r_diff
                    if delta:
                        yield sse(
                            {
                                "id": chunk_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": model,
                                "choices": [
                                    {"index": 0, "delta": delta, "finish_reason": None}
                                ],
                            }
                        )
            for event in incremental.finish():
                reconstructor.handle(event)
                c_diff, r_diff = reconstructor.take_diffs()
                delta: dict = {}
                if c_diff:
                    delta["content"] = c_diff
                if r_diff:
                    delta["reasoning_content"] = r_diff
                if delta:
                    yield sse(
                        {
                            "id": chunk_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [
                                {"index": 0, "delta": delta, "finish_reason": None}
                            ],
                        }
                    )
        finally:
            await resp.aclose()
            account.sessions.touch_last_message(
                session_key, reconstructor.id or response_message_id
            )

    finish = _finish_reason(reconstructor.status)
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
