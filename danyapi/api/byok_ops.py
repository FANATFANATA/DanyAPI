from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import StreamingResponse

from .. import tools as toolemu
from ..accounts import AccountPoolBusy
from ..byok import get_manager as _get_byok_manager
from ..config import settings
from ..qwen import api as qwen_api
from .attachments import (
    _collect_attachments,
    _upload_attachments,
    _validate_attachments,
)
from .deepseek_ops import (
    _collect_non_stream,
    _include_usage,
    _is_reasoning_model,
    _resolve_model,
    _stream_guard,
    _stream_openai,
)
from .schemas import (
    ByokLoginRequest,
    ByokRegisterRequest,
    ByokTokenAddRequest,
    ChatCompletionRequest,
)

router = APIRouter(prefix="/byok", tags=["byok"])


async def _byok_deepseek(req: ChatCompletionRequest, request: Request | None) -> Any:
    mgr = _get_byok_manager()
    if mgr is None:
        raise HTTPException(503, "BYOK manager not initialized")

    session_key = request.cookies.get("byok_session") if request is not None else None
    user_id = mgr.auth_check(session_key) if session_key else None
    if user_id is None:
        raise HTTPException(401, "authentication required")

    account = await mgr.get_or_create_account(user_id, "deepseek")
    if account is None:
        raise HTTPException(400, "no deepseek token configured for your account")

    model_type = _resolve_model(req.model)
    thinking = (
        req.thinking if req.thinking is not None else _is_reasoning_model(req.model)
    )
    search = bool(req.search)

    has_session = bool(account.sessions.can_reuse(req.session_id))
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

    context_seq = toolemu.context_sequence(
        req.messages, user=getattr(req, "user", None)
    )
    existing_sid = req.session_id

    attachments = _collect_attachments(req)
    _validate_attachments(attachments)
    ref_file_ids = None
    if attachments:
        ref_file_ids = await _upload_attachments(
            account=account,
            attachments=attachments,
            model_type=model_type,
            thinking=thinking,
        )

    common = {
        "account": account,
        "pool": None,
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
        "reduced_prompts": None,
        "messages": req.messages,
        "tools": getattr(req, "tools", None),
        "tool_choice": getattr(req, "tool_choice", None),
        "response_format": getattr(req, "response_format", None),
        "user": getattr(req, "user", None),
    }

    if req.stream:
        return StreamingResponse(
            _stream_guard(_stream_openai(lock=account.sem, **common), req.model),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    try:
        return await _collect_non_stream(
            lock=account.sem,
            **{k: v for k, v in common.items() if k != "include_usage"},
        )
    except AccountPoolBusy:
        raise HTTPException(429, "account busy, try again later") from None


async def _byok_qwen(req: ChatCompletionRequest, request: Request | None) -> Any:
    mgr = _get_byok_manager()
    if mgr is None:
        raise HTTPException(503, "BYOK manager not initialized")

    session_key = request.cookies.get("byok_session") if request is not None else None
    user_id = mgr.auth_check(session_key) if session_key else None
    if user_id is None:
        raise HTTPException(401, "authentication required")

    account = await mgr.get_or_create_account(user_id, "qwen")
    if account is None:
        raise HTTPException(400, "no qwen token configured for your account")

    thinking = req.thinking if req.thinking is not None else True
    search = bool(req.search)

    has_session = bool(account.sessions.can_reuse(req.session_id, model=req.model))
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

    context_seq = toolemu.context_sequence(
        req.messages, user=getattr(req, "user", None)
    )
    existing_sid = req.session_id

    common = {
        "account": account,
        "pool": None,
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
        "user": getattr(req, "user", None),
    }

    if req.stream:
        return StreamingResponse(
            _stream_guard(
                qwen_api.stream_openai(lock=account.sem, **common), req.model
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    try:
        return await qwen_api.collect_non_stream(
            lock=account.sem,
            **{k: v for k, v in common.items() if k != "include_usage"},
        )
    except AccountPoolBusy:
        raise HTTPException(429, "account busy, try again later") from None


@router.get("/status")
async def byok_status():
    if not settings.byok_mode:
        return {"enabled": False}
    mgr = _get_byok_manager()
    if mgr is None:
        return {"enabled": True, "authenticated": False}
    return {"enabled": True, "initialized": True}


@router.post("/register")
async def byok_register(body: ByokRegisterRequest):
    if not settings.byok_mode:
        raise HTTPException(403, "BYOK mode is disabled")
    mgr = _get_byok_manager()
    if mgr is None:
        raise HTTPException(503, "BYOK not initialized")
    ok, result = mgr.register(body.username, body.password)
    if ok:
        return {"registered": True, "user_id": result}
    return {"registered": False, "error": result}


@router.post("/login")
async def byok_login(body: ByokLoginRequest):
    if not settings.byok_mode:
        raise HTTPException(403, "BYOK mode is disabled")
    mgr = _get_byok_manager()
    if mgr is None:
        raise HTTPException(503, "BYOK not initialized")
    session_key = mgr.login(body.username, body.password)
    if session_key is None:
        raise HTTPException(401, "invalid credentials")
    resp = Response(
        content=json.dumps({"session_key": session_key}), media_type="application/json"
    )
    resp.set_cookie(
        key="byok_session",
        value=session_key,
        httponly=True,
        samesite="lax",
        max_age=int(settings.session_ttl) if settings.session_ttl else 86400,
    )
    return resp


@router.get("/logout")
async def byok_logout(request: Request):
    session_key = request.cookies.get("byok_session")
    if not session_key:
        raise HTTPException(401, "not logged in")
    mgr = _get_byok_manager()
    if mgr is None:
        raise HTTPException(503, "BYOK not initialized")
    mgr.logout(session_key)
    resp = Response(
        content=json.dumps({"logged_out": True}), media_type="application/json"
    )
    resp.delete_cookie(key="byok_session")
    return resp


@router.get("/me")
async def byok_me(request: Request):
    if not settings.byok_mode:
        raise HTTPException(403, "BYOK mode is disabled")
    mgr = _get_byok_manager()
    if mgr is None:
        raise HTTPException(503, "BYOK not initialized")
    session_key = request.cookies.get("byok_session")
    if not session_key:
        raise HTTPException(401, "not authenticated")
    user_id = mgr.auth_check(session_key)
    if user_id is None:
        raise HTTPException(401, "invalid or expired session")
    user = mgr._store.get_user(user_id)
    return {
        "authenticated": True,
        "user_id": user_id,
        "username": user.get("username") if user else None,
    }


@router.post("/token")
async def byok_add_token(request: Request, body: ByokTokenAddRequest):
    if not settings.byok_mode:
        raise HTTPException(403, "BYOK mode is disabled")
    mgr = _get_byok_manager()
    if mgr is None:
        raise HTTPException(503, "BYOK not initialized")
    session_key = request.cookies.get("byok_session")
    if not session_key:
        raise HTTPException(401, "not authenticated")
    user_id = mgr.auth_check(session_key)
    if user_id is None:
        raise HTTPException(401, "invalid or expired session")
    provider = body.provider.lower().strip()
    if provider not in ("deepseek", "qwen"):
        raise HTTPException(400, f"invalid provider: {provider!r}")
    token_str = body.token.strip()
    if not token_str:
        raise HTTPException(400, "token is empty")
    result = mgr.add_token(user_id, provider, token_str)
    return result


@router.get("/tokens")
async def byok_get_tokens(request: Request):
    if not settings.byok_mode:
        raise HTTPException(403, "BYOK mode is disabled")
    mgr = _get_byok_manager()
    if mgr is None:
        raise HTTPException(503, "BYOK not initialized")
    session_key = request.cookies.get("byok_session")
    if not session_key:
        raise HTTPException(401, "not authenticated")
    user_id = mgr.auth_check(session_key)
    if user_id is None:
        raise HTTPException(401, "invalid or expired session")
    tokens_list = mgr.get_user_tokens(user_id)
    return {"tokens": [{"provider": t["provider"]} for t in tokens_list]}


@router.delete("/token/{provider}")
async def byok_delete_token(request: Request, provider: str):
    if not settings.byok_mode:
        raise HTTPException(403, "BYOK mode is disabled")
    mgr = _get_byok_manager()
    if mgr is None:
        raise HTTPException(503, "BYOK not initialized")
    session_key = request.cookies.get("byok_session")
    if not session_key:
        raise HTTPException(401, "not authenticated")
    user_id = mgr.auth_check(session_key)
    if user_id is None:
        raise HTTPException(401, "invalid or expired session")
    provider = provider.lower().strip()
    if provider not in ("deepseek", "qwen"):
        raise HTTPException(400, f"invalid provider: {provider!r}")
    removed = mgr.remove_token(user_id, provider)
    return {"removed": removed}
