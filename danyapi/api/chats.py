from __future__ import annotations

import json
import logging
import time
import uuid
from functools import partial
from typing import Any

from fastapi import HTTPException, Request
from fastapi.responses import StreamingResponse

from .. import tools as toolemu
from ..accounts import AccountPool, AccountPoolBusy
from ..qwen import api as qwen_api
from .attachments import _collect_attachments, _validate_attachments
from .byok import _byok_pool_for
from .core import _acquire_account
from .deepseek import _collect_non_stream, _stream_openai
from .images import _b64encode
from .models import _is_reasoning_model, _resolve_model, _resolve_provider
from .schemas import ChatCompletionRequest, ChatMessage, CompletionRequest
from .shaping import _bounded_choices, _include_usage
from .sse import _close_generator, _sse, _stream_guard
from .state import _byok_mode, app

log = logging.getLogger("danyapi.api")


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest, request: Request) -> Any:
    return await _dispatch_chat(req, request)


async def _chat_dispatcher(model: str, request: Request) -> Any:
    provider = _resolve_provider(model)
    call = _chat_completions_qwen if provider == "qwen" else _chat_completions_deepseek
    if not _byok_mode():
        return call
    pool = await _byok_pool_for(provider, request)
    return partial(call, pool=pool)


async def _dispatch_chat(req: ChatCompletionRequest, request: Request) -> Any:
    dispatch = await _chat_dispatcher(req.model, request)
    return await dispatch(req)


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


def _completion_chat_request(req: CompletionRequest, prompt_text: str, stream: bool, prompt_count: int) -> ChatCompletionRequest:
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
        session_id=req.session_id if prompt_count <= 1 else None,
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


async def _completions_stream(req: CompletionRequest, prompts: list[str], dispatch: Any):
    for prompt_text in prompts:
        chat_req = _completion_chat_request(req, prompt_text, True, len(prompts))
        chat_resp = await dispatch(chat_req)
        async for line in _translate_completion_stream(chat_resp.body_iterator):
            yield line
    yield "data: [DONE]\n\n"


@app.post("/v1/completions")
async def completions(req: CompletionRequest, request: Request) -> Any:
    prompts = _completion_prompts(req.prompt)
    dispatch = await _chat_dispatcher(req.model, request)
    if req.stream:
        return StreamingResponse(
            _completions_stream(req, prompts, dispatch),
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
        chat_req = _completion_chat_request(req, prompt_text, False, len(prompts))
        chat_dict = await dispatch(chat_req)
        prompt_choices = chat_dict.get("choices") or []
        choices.extend(_legacy_choice_from_chat(choice, base_index + i) for i, choice in enumerate(prompt_choices))
        base_index += len(prompt_choices)
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


async def _acquire_and_build(
    pool: AccountPool,
    req: ChatCompletionRequest,
    reuse_kwargs: dict[str, Any] | None = None,
    *,
    tools: Any,
    tool_choice: Any,
) -> tuple[Any, str | None, tuple[str, ...], str, bool, Any]:
    context_seq = toolemu.context_sequence(req.messages, user=getattr(req, "user", None))
    if req.session_id:
        account, existing_sid = await _acquire_account(pool, req.session_id)
        if existing_sid is None:
            existing_sid = req.session_id
    else:
        cached_sid = pool.resolve_context(context_seq) if context_seq else None
        account, existing_sid = await _acquire_account(pool, cached_sid)
    has_session = _can_reuse_session(account, existing_sid, **(reuse_kwargs or {}))
    cached_session = account.sessions.get(existing_sid) if has_session else None
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
    return account, existing_sid, context_seq, prompt, tool_mode, cached_session


async def _chat_completions_deepseek(req: ChatCompletionRequest, pool: AccountPool | None = None) -> Any:
    if pool is None:
        pool = getattr(app.state, "pool", None)
    if pool is None:
        raise HTTPException(503, "deepseek provider is not configured")

    model_type = _resolve_model(req.model)
    thinking = req.thinking if req.thinking is not None else _is_reasoning_model(req.model)
    search = bool(req.search)

    tools, tool_choice = _materialize_tools(req)
    account, existing_sid, context_seq, prompt, tool_mode, cached_session = await _acquire_and_build(
        pool,
        req,
        tools=tools,
        tool_choice=tool_choice,
    )

    attachments = _collect_attachments(req)
    _validate_attachments(attachments)

    max_tokens = getattr(req, "max_tokens", None)
    if max_tokens is None:
        max_tokens = getattr(req, "max_completion_tokens", None)

    common = {
        "account": account,
        "pool": pool,
        "existing_sid": existing_sid,
        "cached_session": cached_session,
        "prompt": prompt,
        "model": req.model,
        "model_type": model_type,
        "thinking": thinking,
        "search": search,
        "attachments": attachments,
        "tool_schemas": toolemu.tool_schema_map(tools),
        "tool_mode": tool_mode,
        "context_seq": context_seq,
        "messages": req.messages,
        "tools": tools,
        "tool_choice": tool_choice,
        "response_format": getattr(req, "response_format", None),
        "user": getattr(req, "user", None),
        "max_tokens": max_tokens,
        "stop": getattr(req, "stop", None),
        "n": _bounded_choices(getattr(req, "n", None)),
        "parallel_tool_calls": getattr(req, "parallel_tool_calls", None),
    }
    if req.stream:
        return StreamingResponse(
            _stream_guard(_stream_openai(lock=account.sem, include_usage=_include_usage(req), **common), req.model),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    try:
        return await _collect_non_stream(lock=account.sem, **common)
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
    account, existing_sid, context_seq, prompt, tool_mode, cached_session = await _acquire_and_build(
        pool,
        req,
        {"model": req.model},
        tools=tools,
        tool_choice=tool_choice,
    )

    attachments = _collect_attachments(req, allow_remote=True)
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
        "context_seq": context_seq,
        "messages": req.messages,
        "tools": tools,
        "tool_choice": tool_choice,
        "response_format": getattr(req, "response_format", None),
        "user": getattr(req, "user", None),
        "max_tokens": max_tokens,
        "stop": getattr(req, "stop", None),
        "n": _bounded_choices(getattr(req, "n", None)),
        "parallel_tool_calls": getattr(req, "parallel_tool_calls", None),
        "cached_session": cached_session,
    }
    if req.stream:
        return StreamingResponse(
            _stream_guard(qwen_api.stream_openai(lock=account.sem, include_usage=_include_usage(req), **common), req.model),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    try:
        return await qwen_api.collect_non_stream(lock=account.sem, **common)
    except AccountPoolBusy:
        raise HTTPException(429, "all accounts are busy, try again later") from None
