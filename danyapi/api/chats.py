from __future__ import annotations

import json
import logging
import time
import uuid
from collections import OrderedDict
from functools import partial
from typing import Any, NoReturn

from fastapi import HTTPException, Request
from fastapi.responses import StreamingResponse

from .. import tools as toolemu
from ..accounts import AccountPool, AccountPoolBusy
from ..alice import api as alice_api
from ..duckai import api as duckai_api
from ..gigachat import api as gigachat_api
from ..opencode import api as opencode_api
from ..qwen import api as qwen_api
from .attachments import _collect_attachments, _validate_attachments
from .byok import _byok_caller_id, _byok_pool_for, _extract_request_api_key
from .core import MAX_CHAT_BODY_BYTES, _acquire_account
from .deepseek import _collect_non_stream, _stream_openai
from .images import _b64encode
from .models import _is_reasoning_model, _resolve_model, _resolve_provider
from .schemas import ChatCompletionRequest, ChatMessage, CompletionRequest
from .shaping import _bounded_choices, _include_usage
from .sse import _close_generator, _sse, _stream_guard
from .state import _byok_mode, app

log = logging.getLogger("danyapi.api")

CHAT_HANDLERS = {
    "deepseek": "_chat_completions_deepseek",
    "qwen": "_chat_completions_qwen",
    "gigachat": "_chat_completions_gigachat",
    "opencode": "_chat_completions_opencode",
    "alice": "_chat_completions_alice",
    "duckai": "_chat_completions_duckai",
}

MAX_COMPLETION_PROMPTS = 8
MAX_MESSAGES_PER_REQUEST = 2000
MAX_PROVIDER_ERROR_CHARS = 300

GIGACHAT_UNSUPPORTED_PARAMS = ("n", "presence_penalty", "frequency_penalty", "logit_bias")
OPENCODE_UNSUPPORTED_PARAMS = ("n", "logprobs", "top_logprobs")
ALICE_UNSUPPORTED_PARAMS = ("n", "top_p", "presence_penalty", "frequency_penalty", "logit_bias")
DUCKAI_UNSUPPORTED_PARAMS = ("n", "top_p", "presence_penalty", "frequency_penalty", "logit_bias")

_SESSION_OWNERS: OrderedDict[str, str] = OrderedDict()
MAX_SESSION_OWNERS = 4096


def _chat_handler(provider: str) -> Any:
    name = CHAT_HANDLERS.get(provider)
    if name is None:
        raise HTTPException(404, f"Unknown provider: {provider}")
    call = globals().get(name)
    if not callable(call):
        raise HTTPException(500, f"chat handler for {provider} is not callable")
    return call


def _check_chat_request_limits(req: ChatCompletionRequest, request: Request) -> None:
    body = getattr(request, "_body", None)
    if isinstance(body, (bytes, bytearray)) and len(body) > MAX_CHAT_BODY_BYTES:
        raise HTTPException(413, f"request body too large, max {MAX_CHAT_BODY_BYTES // (1024 * 1024)} MB")
    if len(req.messages) > MAX_MESSAGES_PER_REQUEST:
        raise HTTPException(400, f"too many messages: max {MAX_MESSAGES_PER_REQUEST} per request")


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest, request: Request) -> Any:
    _check_chat_request_limits(req, request)
    return await _dispatch_chat(req, request)


async def _chat_dispatcher(model: str, request: Request) -> Any:
    provider = _resolve_provider(model)
    call = _chat_handler(provider)
    if provider == "deepseek" and getattr(app.state, "deepseek_models", None):
        _resolve_model(model)
    if not _byok_mode():
        return call
    pool = await _byok_pool_for(provider, request)
    return partial(call, pool=pool)


async def _dispatch_chat(req: ChatCompletionRequest, request: Request) -> Any:
    dispatch = await _chat_dispatcher(req.model, request)
    return await dispatch(req)


def _caller_scope() -> str:
    return _byok_caller_id()


def _bind_session_owner(session_id: str, owner: str) -> None:
    if not session_id or not owner:
        return
    known = _SESSION_OWNERS.get(session_id)
    if known is not None and known != owner:
        log.warning("rejected session_id reuse across callers")
        raise HTTPException(403, "session_id belongs to another client")
    if known is None:
        _SESSION_OWNERS[session_id] = owner
        while len(_SESSION_OWNERS) > MAX_SESSION_OWNERS:
            _SESSION_OWNERS.popitem(last=False)
    else:
        _SESSION_OWNERS.move_to_end(session_id)


def _completion_prompts(prompt: Any) -> list[str]:
    if isinstance(prompt, str):
        if not prompt.strip():
            raise HTTPException(400, "prompt must not be empty")
        return [prompt]
    if isinstance(prompt, list):
        prompts: list[str] = []
        for item in prompt:
            if isinstance(item, str):
                text = item
            elif isinstance(item, list):
                text = " ".join(str(token) for token in item)
            else:
                raise HTTPException(400, "prompt must be a string, a list of strings, or a list of token lists")
            if not text.strip():
                raise HTTPException(400, "prompt must not contain empty strings")
            prompts.append(text)
        if not prompts:
            raise HTTPException(400, "prompt must not be empty")
        return prompts
    raise HTTPException(400, "prompt must be a string, a list of strings, or a list of token lists")


def _completion_chat_request(req: CompletionRequest, prompt_text: str, stream: bool, prompt_count: int) -> ChatCompletionRequest:
    if req.suffix is not None:
        raise HTTPException(400, "suffix is not supported by the upstream providers")
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


def _legacy_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        text = item.get("text")
        if isinstance(text, str) and text:
            parts.append(text)
    return "".join(parts)


def _legacy_choice_from_chat(chat_choice: dict, index: int) -> dict:
    message = chat_choice.get("message")
    if not isinstance(message, dict):
        message = {}
    return {
        "index": index,
        "text": _legacy_text(message.get("content")),
        "logprobs": None,
        "finish_reason": chat_choice.get("finish_reason") or "stop",
    }


def _exception_detail(exc: BaseException) -> str:
    if isinstance(exc, HTTPException) and isinstance(exc.detail, str):
        return exc.detail
    return str(exc)


def _safe_error_message(value: Any) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str) if not isinstance(value, (int, float, bool)) else str(value)
    text = text.strip()
    if len(text) > MAX_PROVIDER_ERROR_CHARS:
        text = text[:MAX_PROVIDER_ERROR_CHARS]
    return text or "upstream error"


def _translate_chat_chunk_to_completion(chunk: dict, base_index: int = 0) -> dict:
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
        message = error.get("message") if isinstance(error, dict) else error
        piece["error"] = {"message": _safe_error_message(message)}
    raw_choices = chunk.get("choices")
    if not isinstance(raw_choices, list):
        raw_choices = []
    for choice in raw_choices:
        if not isinstance(choice, dict):
            continue
        delta = choice.get("delta")
        if not isinstance(delta, dict):
            delta = {}
        text = delta.get("content")
        index = choice.get("index")
        piece["choices"].append(
            {
                "index": base_index + (index if isinstance(index, int) and not isinstance(index, bool) else 0),
                "text": text if isinstance(text, str) else "",
                "logprobs": None,
                "finish_reason": choice.get("finish_reason"),
            }
        )
    return piece


def _usage_count(usage: dict, field: str) -> int:
    value = usage.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    return int(value)


async def _translate_completion_stream(chat_gen, base_index: int = 0, seen_indexes: dict[int, None] | None = None):
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
            if seen_indexes is not None:
                for choice in chunk.get("choices") or ():
                    if isinstance(choice, dict):
                        raw = choice.get("index")
                        seen_indexes[raw if isinstance(raw, int) and not isinstance(raw, bool) else 0] = None
            yield _sse(_translate_chat_chunk_to_completion(chunk, base_index))
    finally:
        await _close_generator(chat_gen)


async def _completions_stream(req: CompletionRequest, prompts: list[str], dispatch: Any):
    base_index = 0
    for index, prompt_text in enumerate(prompts):
        prompt_base = base_index
        seen_indexes: dict[int, None] = {}
        try:
            chat_req = _completion_chat_request(req, prompt_text, True, len(prompts))
            chat_resp = await dispatch(chat_req)
            async for line in _translate_completion_stream(chat_resp.body_iterator, prompt_base, seen_indexes):
                yield line
            base_index = prompt_base + max(seen_indexes, default=-1) + 1
        except Exception as exc:
            detail = _exception_detail(exc)
            log.warning("completions prompt #%d of %d failed: %s: %s", index + 1, len(prompts), type(exc).__name__, exc)
            yield _sse(
                {
                    "id": f"cmpl-{uuid.uuid4().hex}",
                    "object": "text_completion",
                    "created": int(time.time()),
                    "model": req.model,
                    "choices": [],
                    "error": {"message": _safe_error_message(detail)},
                }
            )
            break
    yield "data: [DONE]\n\n"


@app.post("/v1/completions")
async def completions(req: CompletionRequest, request: Request) -> Any:
    prompts = _completion_prompts(req.prompt)
    if len(prompts) > MAX_COMPLETION_PROMPTS:
        raise HTTPException(400, f"too many prompts: max {MAX_COMPLETION_PROMPTS} per request")
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
        prompt_choices = chat_dict.get("choices")
        if not isinstance(prompt_choices, list):
            prompt_choices = []
        prompt_choices = [choice for choice in prompt_choices if isinstance(choice, dict)]
        choices.extend(_legacy_choice_from_chat(choice, base_index + i) for i, choice in enumerate(prompt_choices))
        base_index += len(prompt_choices)
        if not completion_id:
            completion_id = chat_dict.get("id")
        created = chat_dict.get("created", created)
        u = chat_dict.get("usage")
        if isinstance(u, dict):
            prompt_tokens += _usage_count(u, "prompt_tokens")
            completion_tokens += _usage_count(u, "completion_tokens")
            total_tokens += _usage_count(u, "total_tokens")
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


@app.post("/v1/embeddings", response_model=None)
async def embeddings_not_supported(request: Request) -> NoReturn:
    await _reject_unsupported_endpoint(request, "embeddings")
    raise HTTPException(501, "embeddings are not supported by DanyAPI")


@app.post("/v1/moderations", response_model=None)
async def moderations_not_supported(request: Request) -> NoReturn:
    await _reject_unsupported_endpoint(request, "moderations")
    raise HTTPException(501, "moderations are not supported by DanyAPI")


async def _reject_unsupported_endpoint(request: Request, endpoint: str) -> None:
    if not _byok_mode():
        return
    token = await _extract_request_api_key(request)
    if not token:
        raise HTTPException(401, f"api key is required in byok mode, {endpoint} cannot be used without one")


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


def _request_scope(req: ChatCompletionRequest) -> str | None:
    user = getattr(req, "user", None)
    if isinstance(user, str) and user:
        return f"u:{user}"
    if _byok_mode() and _caller_scope():
        return f"k:{_caller_scope()}"
    return None


async def _acquire_session_account(pool: AccountPool, req: Any) -> tuple[Any, str | None]:
    if req.session_id:
        scope = _request_scope(req)
        if scope:
            _bind_session_owner(req.session_id, scope)
    return await _acquire_account(pool, req.session_id)


async def _acquire_and_build(
    pool: AccountPool,
    req: ChatCompletionRequest,
    reuse_kwargs: dict[str, Any] | None = None,
    *,
    tools: Any,
    tool_choice: Any,
) -> tuple[Any, str | None, tuple[str, ...], str, bool, Any]:
    scope = _request_scope(req)
    context_seq = toolemu.context_sequence(req.messages, user=scope) if scope else ()
    if req.session_id:
        if scope:
            _bind_session_owner(req.session_id, scope)
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

    max_tokens = _max_tokens_of(req)

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

    max_tokens = _max_tokens_of(req)

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


def _max_tokens_of(req: ChatCompletionRequest) -> int | None:
    max_tokens = getattr(req, "max_tokens", None)
    if max_tokens is None:
        max_tokens = getattr(req, "max_completion_tokens", None)
    return max_tokens


def _reject_unsupported_params(req: ChatCompletionRequest, provider: str, params: tuple[str, ...]) -> None:
    for name in params:
        value = getattr(req, name, None)
        if value is None or (isinstance(value, (list, dict)) and not value):
            continue
        if name == "n" and value == 1:
            continue
        raise HTTPException(400, f"{provider} does not support the {name} parameter")


async def _chat_completions_gigachat(req: ChatCompletionRequest, pool: AccountPool | None = None) -> Any:
    if pool is None:
        pool = getattr(app.state, "gigachat_pool", None)
    if pool is None:
        raise HTTPException(503, "gigachat provider is not configured")

    _reject_unsupported_params(req, "gigachat", GIGACHAT_UNSUPPORTED_PARAMS)
    tools, tool_choice = _materialize_tools(req)
    account, existing_sid = await _acquire_session_account(pool, req)
    max_tokens = _max_tokens_of(req)

    common = {
        "account": account,
        "messages": req.messages,
        "model": req.model,
        "tools": tools,
        "tool_choice": tool_choice,
        "functions": getattr(req, "functions", None),
        "function_call": getattr(req, "function_call", None),
        "temperature": req.temperature,
        "top_p": req.top_p,
        "max_tokens": max_tokens,
        "stop": getattr(req, "stop", None),
        "response_format": getattr(req, "response_format", None),
        "user": getattr(req, "user", None),
        "session_id": existing_sid,
    }
    if req.stream:
        return StreamingResponse(
            _stream_guard(
                gigachat_api.stream_openai(include_usage=_include_usage(req), **common),
                req.model,
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    try:
        return await gigachat_api.collect_non_stream(**common)
    except AccountPoolBusy:
        raise HTTPException(429, "all accounts are busy, try again later") from None


async def _chat_completions_opencode(req: ChatCompletionRequest, pool: AccountPool | None = None) -> Any:
    if pool is None:
        pool = getattr(app.state, "opencode_pool", None)
    if pool is None:
        raise HTTPException(503, "opencode provider is not configured (set OPENCODE_KEYS to enable it)")

    if getattr(req, "files", None):
        raise HTTPException(400, "opencode does not support file attachments, send images inline instead")
    _reject_unsupported_params(req, "opencode", OPENCODE_UNSUPPORTED_PARAMS)
    account, existing_sid = await _acquire_session_account(pool, req)

    tools, tool_choice = _materialize_tools(req)
    common = {
        "account": account,
        "messages": req.messages,
        "model": req.model,
        "tools": tools,
        "tool_choice": tool_choice,
        "temperature": req.temperature,
        "top_p": req.top_p,
        "max_tokens": _max_tokens_of(req),
        "stop": getattr(req, "stop", None),
        "response_format": getattr(req, "response_format", None),
        "user": getattr(req, "user", None),
        "session_id": existing_sid,
    }
    if req.stream:
        return StreamingResponse(
            _stream_guard(opencode_api.stream_openai(include_usage=_include_usage(req), **common), req.model),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    try:
        return await opencode_api.collect_non_stream(**common)
    except AccountPoolBusy:
        raise HTTPException(429, "all accounts are busy, try again later") from None


async def _chat_completions_alice(req: ChatCompletionRequest, pool: AccountPool | None = None) -> Any:
    if pool is None:
        pool = getattr(app.state, "alice_pool", None)
    if pool is None:
        raise HTTPException(503, "alice provider is not configured (set ALICE_ENABLED=1 to enable)")

    if getattr(req, "files", None):
        raise HTTPException(400, "alice does not support file attachments")
    _reject_unsupported_params(req, "alice", ALICE_UNSUPPORTED_PARAMS)
    account, existing_sid = await _acquire_session_account(pool, req)

    common = {
        "account": account,
        "messages": req.messages,
        "model": req.model,
        "stop": getattr(req, "stop", None),
        "user": getattr(req, "user", None),
        "session_id": existing_sid,
    }
    if req.stream:
        return StreamingResponse(
            _stream_guard(alice_api.stream_openai(include_usage=_include_usage(req), **common), req.model),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    try:
        return await alice_api.collect_non_stream(**common)
    except AccountPoolBusy:
        raise HTTPException(429, "all accounts are busy, try again later") from None


async def _chat_completions_duckai(req: ChatCompletionRequest, pool: AccountPool | None = None) -> Any:
    if pool is None:
        pool = getattr(app.state, "duckai_pool", None)
    if pool is None:
        raise HTTPException(503, "duckai provider is not configured (set DUCKAI_ENABLED=1 to enable)")

    if getattr(req, "files", None):
        raise HTTPException(400, "duckai does not support file attachments, send images inline instead")
    _reject_unsupported_params(req, "duck.ai", DUCKAI_UNSUPPORTED_PARAMS)
    account, existing_sid = await _acquire_session_account(pool, req)

    tools, tool_choice = _materialize_tools(req)
    common = {
        "account": account,
        "messages": req.messages,
        "model": req.model,
        "tools": tools,
        "tool_choice": tool_choice,
        "functions": getattr(req, "functions", None),
        "function_call": getattr(req, "function_call", None),
        "thinking": req.thinking,
        "search": bool(req.search),
        "stop": getattr(req, "stop", None),
        "user": getattr(req, "user", None),
        "session_id": existing_sid,
    }
    if req.stream:
        return StreamingResponse(
            _stream_guard(duckai_api.stream_openai(include_usage=_include_usage(req), **common), req.model),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    try:
        return await duckai_api.collect_non_stream(**common)
    except AccountPoolBusy:
        raise HTTPException(429, "all accounts are busy, try again later") from None


def _validate_chat_handlers() -> None:
    missing = sorted(name for name in CHAT_HANDLERS.values() if name not in globals())
    if missing:
        raise RuntimeError(f"CHAT_HANDLERS references undefined handlers: {', '.join(missing)}")


_validate_chat_handlers()
