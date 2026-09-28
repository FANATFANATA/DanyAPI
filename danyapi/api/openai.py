from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse

from .. import tools as toolemu
from ..accounts import AccountPool, DeepSeekAccount, account_lock
from ..alice import api as alice_api
from ..alice.accounts import AliceAccount
from ..alice.client import AliceClient, AliceError
from ..config import MAX_CHOICES, settings
from ..deepseek.client import DeepSeekClient, DeepSeekError, DeepSeekSession
from ..deepseek.stream import IncrementalSSE, MessageReconstructor
from ..duckai import api as duckai_api
from ..gigachat import api as gigachat_api
from ..gigachat.accounts import GigaChatAccount
from ..gigachat.client import GigaChatClient, GigaChatError
from ..gigachat.messages import build_messages, normalize_usage, request_body
from ..gigachat.tls import resolve_ca
from ..qwen import api as qwen_api
from ..qwen.accounts import QwenAccount
from ..qwen.client import QwenClient
from ..tokens import count_message_tokens
from . import anthropic as anthropic_api
from . import responses as responses_api
from .attachments import (
    MAX_ATTACHMENT_TOTAL_SIZE,
    MAX_FILE_SIZE,
    MAX_FILES_PER_REQUEST,
    REMOTE_IMAGE_SCHEMES,
    Attachment,
    _collect_attachments,
    _compact_data_uri_length,
    _data_uri_parts,
    _decode_data_uri,
    _raw_data_uri_length,
    _split_data_uri,
    _upload_attachments,
    _validate_attachments,
)
from .byok import (
    BYOK_AUTH_LIMIT,
    BYOK_POOL_LIMIT,
    _api_key_from_form,
    _byok_cache_key,
    _byok_pool,
    _byok_pool_for,
    _byok_validate,
    _cached_auth,
    _close_busy_client,
    _close_pool,
    _deferred_close_tasks,
    _evict_auth,
    _extract_request_api_key,
)
from .chats import (
    _acquire_and_build,
    _can_reuse_session,
    _chat_completions_alice,
    _chat_completions_deepseek,
    _chat_completions_duckai,
    _chat_completions_gigachat,
    _chat_completions_qwen,
    _chat_dispatcher,
    _completion_chat_request,
    _completion_prompts,
    _completions_stream,
    _legacy_choice_from_chat,
    _materialize_tools,
    _max_tokens_of,
    _translate_chat_chunk_to_completion,
    _translate_completion_stream,
    completions,
    embeddings_not_supported,
    moderations_not_supported,
)
from .core import (
    _POOL_RATE_CACHE,
    _POOL_RATE_CACHE_MAX,
    _POOL_RATE_TTL,
    _STATE_STORE_ATTRS,
    INTERNAL_ERROR_MESSAGE,
    MAX_LOGGED_BODY,
    MAX_REQUEST_BODY,
    POOL_ATTRS,
    _account_busy_count,
    _acquire_account,
    _close_pow_managers,
    _error_code_for_status,
    _error_detail,
    _error_type_for_status,
    _exception_message,
    _extract_request_body,
    _flush_state_stores,
    _log_request_failure,
    _log_request_success,
    _log_requests,
    _on_http_exception,
    _on_uncaught_exception,
    _on_validation_error,
    _openai_error_payload,
    _openai_headers,
    _parse_logged_body,
    _pool_rate_headers,
    _read_request_body,
    _request_client_ip,
    _request_details,
    _request_id_header,
    _responses_store,
    _shared_store,
    _token_stable_id,
    lifespan,
)
from .deepseek import (
    _UNSET,
    CONTEXT_LENGTH_STATUS,
    CONTINUE_DEADLINE_SEC,
    CONTINUE_PROMPT,
    FAKE_CONTEXT_HINT_ERROR_MESSAGE,
    FAKE_CONTEXT_HINT_MARKERS,
    INPUT_EXCEEDS_LIMIT,
    MAX_CONTINUE_ROUNDS,
    MAX_ERROR_BODY_CHARS,
    MESSAGE_TOO_FREQUENT_MARKERS,
    MESSAGE_TOO_FREQUENT_MAX_RETRIES,
    MESSAGE_TOO_FREQUENT_WAIT_SEC,
    REDUCED_CONTEXT_MESSAGE,
    RESPONSE_INCOMPLETE,
    RESPONSE_INCOMPLETE_MESSAGE,
    RETRYABLE_FINISH_REASONS,
    STATUS_TO_FINISH_REASON,
    SYSTEM_FINGERPRINT,
    _build_assistant_message,
    _build_completion_response,
    _build_limited_message,
    _busy_error_body,
    _collect_continuation,
    _collect_non_stream,
    _collect_reduced,
    _compact_error_text,
    _continue_deadline_expired,
    _error_text,
    _fake_context_error_body,
    _incomplete_error_body,
    _incomplete_message,
    _input_exceeds_hint_from_http,
    _is_context_limit,
    _is_fake_context_hint,
    _is_input_exceeds_limit,
    _is_message_too_frequent_hint,
    _is_message_too_frequent_http,
    _is_retryable_hint,
    _message_too_frequent_text,
    _prepare_session,
    _reduced_prompt_variants,
    _send_completion,
    _send_deepseek_stream,
    _send_with_auth,
    _stream_openai,
    _wait_message_too_frequent,
)
from .envtokens import (
    _TOKENS_LOCK,
    _atomic_write_text,
    _env_path,
    _env_token_list,
    _pool_account_by_stable,
    _read_env_tokens,
    _read_env_tokens_sync,
    _require_admin_token,
    _unquote_env_value,
    _write_env_tokens,
    _write_env_tokens_sync,
    add_tokens,
)
from .images import (
    _ASYNC_B64_THRESHOLD,
    IMAGE_SIZE_RE,
    MAX_IMAGE_DIM,
    MIN_IMAGE_DIM,
    _b64encode,
    _image_client_lock,
    _image_edit_req,
    _image_generations,
    _image_http_client,
    _image_markdown,
    _image_pool,
    _parse_image_size,
    _read_upload,
    _resize_image_bytes,
    image_edits,
    image_generations,
    image_variations,
)
from .models import (
    ALICE_MODEL_IDS,
    DEEPSEEK_LEGACY_ALIASES,
    MODEL_CREATED_AT,
    MODEL_FETCHERS,
    REASONING_SUFFIXES,
    _all_models,
    _default_deepseek_model_type,
    _fetch_alice_models,
    _fetch_deepseek_models,
    _fetch_duckai_models,
    _fetch_gigachat_models,
    _fetch_qwen_models,
    _finish_reason,
    _header_api_key,
    _is_deepseek_model,
    _is_reasoning_model,
    _model_cache_key,
    _model_source,
    _models_state,
    _output_truncated,
    _resolve_model,
    _resolve_provider,
    _store_models,
    get_model,
    list_models,
    model_refresh_loop,
    provider_enabled,
    refresh_models,
    refresh_provider_models,
)
from .powauth import (
    DEEPSEEK_AUTH_ERROR_CODES,
    _deepseek_error_detail,
    _deepseek_status,
    _drop_session,
    _fresh_pow_headers,
    _fresh_pow_upload_headers,
    _handle_account_error,
)
from .retry import (
    MAX_RETRIES,
    RETRY_BACKOFF_MAX_SEC,
    RETRY_BACKOFF_SEC,
    RETRYABLE_HTTP_STATUSES,
    STALE_SESSION_STATUSES,
    _is_retryable_http,
    _retry_delay,
    _try_stop_stream,
)
from .schemas import (
    ChatCompletionRequest,
    ChatMessage,
    CompletionRequest,
    DeepSeekStreamError,
    FileSpec,
    ImageGenerationRequest,
    ResponsesRequest,
)
from .shaping import (
    MAX_STREAM_CHOICES,
    USAGE_TOTAL_FIELDS,
    _advance_session_usage,
    _apply_limits,
    _apply_stop,
    _bounded_choices,
    _deepseek_usage,
    _include_usage,
    _max_calls,
    _merge_usage,
    _usage_with_details,
)
from .sse import (
    _JSON_ENCODE,
    _chunk_id_from_line,
    _close_generator,
    _delta_json,
    _sse,
    _stream_error_sse,
    _stream_guard,
)
from .state import (
    BYOK_PROVIDERS,
    KEYLESS_PROVIDERS,
    MODEL_ATTRS,
    POOL_ATTRS_BY_PROVIDER,
    _byok_auth_state,
    _byok_locks_state,
    _byok_mode,
    _byok_pools_state,
    _byok_stores_state,
    app,
    provider_models,
    provider_needs_api_key,
    provider_pool,
)


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


def _pool_stats(pool) -> dict | None:
    if pool is None:
        return None
    try:
        return pool.stats()
    except Exception:
        return None


def _byok_pools_for(provider: str, byok_pools: dict[str, Any]) -> list[Any]:
    pools = [pool for pool in (byok_pools.get(provider) or {}).values() if pool is not None]
    if provider_needs_api_key(provider):
        return pools
    singleton = getattr(app.state, f"byok_{provider}_pool", None)
    if singleton is not None and singleton not in pools:
        pools.insert(0, singleton)
    return pools


def _byok_provider_stats(provider: str, byok_pools: dict[str, Any]) -> dict:
    pools = _byok_pools_for(provider, byok_pools)
    accounts = healthy = broken = 0
    for pool in pools:
        stats = _pool_stats(pool) or {}
        accounts += int(stats.get("accounts") or 0)
        healthy += int(stats.get("healthy") or 0)
        broken += int(stats.get("broken") or 0)
    return {
        "pools": len(pools),
        "accounts": accounts,
        "healthy": healthy,
        "broken": broken,
        "models": len(provider_models(provider)),
    }


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
    byok_mode = _byok_mode()
    result: dict[str, Any] = {
        "status": "ok",
        "usage": _usage_summary(),
    }
    if byok_mode:
        byok_pools = await _byok_pools_state()
        result["byok"] = True
        result["byok_pools"] = {provider: len(entries or {}) for provider, entries in byok_pools.items()}
        result["byok_api_key_required"] = {provider: provider_needs_api_key(provider) for provider in BYOK_PROVIDERS}
        for provider in BYOK_PROVIDERS:
            result[provider] = True
            result[f"{provider}_stats"] = _byok_provider_stats(provider, byok_pools)
        return result
    for provider in BYOK_PROVIDERS:
        pool = provider_pool(provider)
        result[provider] = provider_enabled(provider)
        stats = _pool_stats(pool)
        if stats is not None:
            stats = {**stats, "models": len(provider_models(provider))}
        result[f"{provider}_stats"] = stats
    return result


@app.get("/v1/usage")
async def usage_stats() -> dict:
    tracker = getattr(app.state, "usage", None)
    if tracker is None:
        raise HTTPException(404, "usage tracking is disabled")
    return tracker.snapshot()


_MODEL_CACHE: dict[str, Any] = {
    "key": None,
    "models": None,
    "index": None,
    "qwen_ids": None,
    "gigachat_ids": None,
    "alice_ids": None,
    "duckai_ids": None,
}


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
        max_tokens=req.max_output_tokens,
    )


@app.post("/v1/responses")
async def create_response(req: ResponsesRequest, request: Request) -> Any:
    provider_call = await _chat_dispatcher(req.model, request)
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


ANTHROPIC_ERROR_TYPES = {
    400: "invalid_request_error",
    401: "authentication_error",
    403: "permission_error",
    404: "not_found_error",
    413: "request_too_large",
    429: "rate_limit_error",
    529: "overloaded_error",
}


def _anthropic_model(body: dict) -> str:
    model = body.get("model")
    if not isinstance(model, str) or not model.strip():
        raise HTTPException(400, "model is required")
    _resolve_provider(model)
    return model


def _anthropic_error(exc: anthropic_api.AnthropicInputError) -> JSONResponse:
    return JSONResponse(status_code=400, content=anthropic_api.error_body("invalid_request_error", str(exc)))


def _anthropic_http_error(exc: HTTPException) -> JSONResponse:
    detail = exc.detail
    if isinstance(detail, dict):
        inner = detail.get("error")
        detail = inner.get("message") if isinstance(inner, dict) and isinstance(inner.get("message"), str) else str(detail)
    error_type = ANTHROPIC_ERROR_TYPES.get(exc.status_code, "api_error" if exc.status_code >= 500 else "invalid_request_error")
    headers = {str(k): str(v) for k, v in exc.headers.items()} if exc.headers else None
    return JSONResponse(
        status_code=exc.status_code,
        content=anthropic_api.error_body(error_type, str(detail)),
        headers=headers,
    )


@app.post("/v1/messages")
async def anthropic_messages(body: dict, request: Request) -> Any:
    try:
        return await _anthropic_messages(body, request)
    except HTTPException as exc:
        return _anthropic_http_error(exc)


def _upstream_model_for(provider: str, model: str) -> str:
    if provider == "deepseek":
        return _resolve_model(model)
    return model


async def _anthropic_messages(body: dict, request: Request) -> Any:
    if not isinstance(body, dict):
        raise HTTPException(400, "request body must be a JSON object")
    try:
        model = _anthropic_model(body)
    except HTTPException as exc:
        error_type = "not_found_error" if exc.status_code == 404 else "invalid_request_error"
        return JSONResponse(status_code=exc.status_code, content=anthropic_api.error_body(error_type, str(exc.detail)))
    try:
        chat_payload = anthropic_api.build_chat_request(body, model)
    except anthropic_api.AnthropicInputError as exc:
        return _anthropic_error(exc)
    chat_req = ChatCompletionRequest(**chat_payload)
    provider = _resolve_provider(model)
    info = anthropic_api.RequestInfo(
        model=model,
        upstream_model=_upstream_model_for(provider, model),
        max_tokens=chat_req.max_tokens or anthropic_api.DEFAULT_MAX_TOKENS,
        prompt_tokens=anthropic_api.count_input_tokens(chat_req.messages, anthropic_api.normalize_system(body.get("system"))),
        metadata=body.get("metadata"),
        system_present=bool(anthropic_api.normalize_system(body.get("system"))),
    )
    message_id = f"msg_{uuid.uuid4().hex}"
    provider_call = await _chat_dispatcher(model, request)

    if chat_req.stream:
        chat_req.stream_options = {"include_usage": True}
        chat_resp = await provider_call(chat_req)
        return StreamingResponse(
            anthropic_api.translate_stream(chat_resp.body_iterator, info, message_id),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    chat_dict = await provider_call(chat_req)
    if isinstance(chat_dict, dict) and isinstance(chat_dict.get("error"), dict):
        error = chat_dict["error"]
        message = error.get("message") if isinstance(error.get("message"), str) else "upstream request failed"
        return JSONResponse(status_code=502, content=anthropic_api.error_body("api_error", message))
    return anthropic_api.build_message(info, message_id, chat_dict)


@app.post("/v1/messages/count_tokens")
async def anthropic_count_tokens(body: dict) -> Any:
    try:
        return await _anthropic_count_tokens(body)
    except HTTPException as exc:
        return _anthropic_http_error(exc)


async def _anthropic_count_tokens(body: dict) -> Any:
    if not isinstance(body, dict):
        raise HTTPException(400, "request body must be a JSON object")
    try:
        messages = anthropic_api.normalize_messages(body.get("messages"))
        system = anthropic_api.normalize_system(body.get("system"))
    except anthropic_api.AnthropicInputError as exc:
        return _anthropic_error(exc)
    tools = anthropic_api.convert_tools(body.get("tools"))
    total = anthropic_api.count_input_tokens(messages, system)
    if tools:
        for tool in tools:
            function = tool.get("function") if isinstance(tool, dict) else None
            if isinstance(function, dict):
                total += count_message_tokens({"role": "system", "content": str(function.get("description") or "")})
    return {"input_tokens": total}


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
    store = _responses_store()
    record = store.get(response_id)
    if not isinstance(record, dict):
        raise HTTPException(404, f"response {response_id} not found")
    public = record.get("public")
    if isinstance(public, dict) and public.get("status") in ("in_progress", "queued"):
        cancelled = dict(public) | {"status": "cancelled", "incomplete_details": {"reason": "cancelled"}}
        store.set(response_id, dict(record) | {"public": cancelled})
        return cancelled
    raise HTTPException(409, f"response {response_id} is not cancellable in its current state")


@app.api_route("/v1/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
async def unknown_v1_route(path: str) -> dict:
    raise HTTPException(404, f"Unknown /v1 endpoint: /v1/{path}")


__all__ = [
    "ALICE_MODEL_IDS",
    "BYOK_AUTH_LIMIT",
    "BYOK_POOL_LIMIT",
    "BYOK_PROVIDERS",
    "CONTEXT_LENGTH_STATUS",
    "CONTINUE_DEADLINE_SEC",
    "CONTINUE_PROMPT",
    "DEEPSEEK_AUTH_ERROR_CODES",
    "DEEPSEEK_LEGACY_ALIASES",
    "FAKE_CONTEXT_HINT_ERROR_MESSAGE",
    "FAKE_CONTEXT_HINT_MARKERS",
    "IMAGE_SIZE_RE",
    "INPUT_EXCEEDS_LIMIT",
    "INTERNAL_ERROR_MESSAGE",
    "KEYLESS_PROVIDERS",
    "MAX_ATTACHMENT_TOTAL_SIZE",
    "MAX_CHOICES",
    "MAX_CONTINUE_ROUNDS",
    "MAX_ERROR_BODY_CHARS",
    "MAX_FILES_PER_REQUEST",
    "MAX_FILE_SIZE",
    "MAX_IMAGE_DIM",
    "MAX_LOGGED_BODY",
    "MAX_REQUEST_BODY",
    "MAX_RETRIES",
    "MAX_STREAM_CHOICES",
    "MESSAGE_TOO_FREQUENT_MARKERS",
    "MESSAGE_TOO_FREQUENT_MAX_RETRIES",
    "MESSAGE_TOO_FREQUENT_WAIT_SEC",
    "MIN_IMAGE_DIM",
    "MODEL_ATTRS",
    "MODEL_CREATED_AT",
    "MODEL_FETCHERS",
    "POOL_ATTRS",
    "POOL_ATTRS_BY_PROVIDER",
    "REASONING_SUFFIXES",
    "REDUCED_CONTEXT_MESSAGE",
    "REMOTE_IMAGE_SCHEMES",
    "RESPONSE_INCOMPLETE",
    "RESPONSE_INCOMPLETE_MESSAGE",
    "RETRYABLE_FINISH_REASONS",
    "RETRYABLE_HTTP_STATUSES",
    "RETRY_BACKOFF_MAX_SEC",
    "RETRY_BACKOFF_SEC",
    "STALE_SESSION_STATUSES",
    "STATUS_TO_FINISH_REASON",
    "SYSTEM_FINGERPRINT",
    "USAGE_TOTAL_FIELDS",
    "_ASYNC_B64_THRESHOLD",
    "_JSON_ENCODE",
    "_POOL_RATE_CACHE",
    "_POOL_RATE_CACHE_MAX",
    "_POOL_RATE_TTL",
    "_STATE_STORE_ATTRS",
    "_TOKENS_LOCK",
    "_UNSET",
    "AccountPool",
    "AliceAccount",
    "AliceClient",
    "AliceError",
    "Attachment",
    "ChatCompletionRequest",
    "ChatMessage",
    "CompletionRequest",
    "DeepSeekAccount",
    "DeepSeekClient",
    "DeepSeekError",
    "DeepSeekSession",
    "DeepSeekStreamError",
    "FileSpec",
    "GigaChatAccount",
    "GigaChatClient",
    "GigaChatError",
    "HTTPException",
    "ImageGenerationRequest",
    "IncrementalSSE",
    "JSONResponse",
    "MessageReconstructor",
    "QwenAccount",
    "QwenClient",
    "Request",
    "ResponsesRequest",
    "StreamingResponse",
    "_RootContext",
    "_account_busy_count",
    "_acquire_account",
    "_acquire_and_build",
    "_advance_session_usage",
    "_all_models",
    "_anthropic_count_tokens",
    "_anthropic_error",
    "_anthropic_http_error",
    "_anthropic_messages",
    "_anthropic_model",
    "_api_key_from_form",
    "_apply_limits",
    "_apply_stop",
    "_atomic_write_text",
    "_b64encode",
    "_bounded_choices",
    "_build_assistant_message",
    "_build_completion_response",
    "_build_limited_message",
    "_busy_error_body",
    "_byok_auth_state",
    "_byok_cache_key",
    "_byok_locks_state",
    "_byok_mode",
    "_byok_pool",
    "_byok_pool_for",
    "_byok_pools_state",
    "_byok_stores_state",
    "_byok_validate",
    "_cached_auth",
    "_can_reuse_session",
    "_chat_completions_alice",
    "_chat_completions_deepseek",
    "_chat_completions_duckai",
    "_chat_completions_gigachat",
    "_chat_completions_qwen",
    "_chunk_id_from_line",
    "_close_busy_client",
    "_close_generator",
    "_close_pool",
    "_close_pow_managers",
    "_collect_attachments",
    "_collect_continuation",
    "_collect_non_stream",
    "_collect_reduced",
    "_compact_data_uri_length",
    "_compact_error_text",
    "_completion_chat_request",
    "_completion_prompts",
    "_completions_stream",
    "_continue_deadline_expired",
    "_data_uri_parts",
    "_decode_data_uri",
    "_deepseek_error_detail",
    "_deepseek_status",
    "_deepseek_usage",
    "_default_deepseek_model_type",
    "_deferred_close_tasks",
    "_delta_json",
    "_drop_session",
    "_env_path",
    "_env_token_list",
    "_error_code_for_status",
    "_error_detail",
    "_error_text",
    "_error_type_for_status",
    "_evict_auth",
    "_exception_message",
    "_extract_request_api_key",
    "_extract_request_body",
    "_fake_context_error_body",
    "_fetch_alice_models",
    "_fetch_deepseek_models",
    "_fetch_duckai_models",
    "_fetch_gigachat_models",
    "_fetch_qwen_models",
    "_finish_reason",
    "_flush_state_stores",
    "_fresh_pow_headers",
    "_fresh_pow_upload_headers",
    "_handle_account_error",
    "_header_api_key",
    "_image_client_lock",
    "_image_edit_req",
    "_image_generations",
    "_image_http_client",
    "_image_markdown",
    "_image_pool",
    "_include_usage",
    "_incomplete_error_body",
    "_incomplete_message",
    "_input_exceeds_hint_from_http",
    "_is_context_limit",
    "_is_deepseek_model",
    "_is_fake_context_hint",
    "_is_input_exceeds_limit",
    "_is_message_too_frequent_hint",
    "_is_message_too_frequent_http",
    "_is_reasoning_model",
    "_is_retryable_hint",
    "_is_retryable_http",
    "_legacy_choice_from_chat",
    "_log_request_failure",
    "_log_request_success",
    "_log_requests",
    "_materialize_tools",
    "_max_calls",
    "_max_tokens_of",
    "_merge_usage",
    "_message_too_frequent_text",
    "_model_cache_key",
    "_model_source",
    "_models_state",
    "_on_http_exception",
    "_on_uncaught_exception",
    "_on_validation_error",
    "_openai_error_payload",
    "_openai_headers",
    "_output_truncated",
    "_parse_image_size",
    "_parse_logged_body",
    "_pool_account_by_stable",
    "_pool_rate_headers",
    "_prepare_session",
    "_raw_data_uri_length",
    "_read_env_tokens",
    "_read_env_tokens_sync",
    "_read_request_body",
    "_read_upload",
    "_reduced_prompt_variants",
    "_request_client_ip",
    "_request_details",
    "_request_id_header",
    "_require_admin_token",
    "_resize_image_bytes",
    "_resolve_model",
    "_resolve_provider",
    "_responses_store",
    "_retry_delay",
    "_send_completion",
    "_send_deepseek_stream",
    "_send_with_auth",
    "_shared_store",
    "_split_data_uri",
    "_sse",
    "_store_models",
    "_stream_error_sse",
    "_stream_guard",
    "_stream_openai",
    "_token_stable_id",
    "_translate_chat_chunk_to_completion",
    "_translate_completion_stream",
    "_try_stop_stream",
    "_unquote_env_value",
    "_upload_attachments",
    "_upstream_model_for",
    "_usage_with_details",
    "_validate_attachments",
    "_wait_message_too_frequent",
    "_write_env_tokens",
    "_write_env_tokens_sync",
    "account_lock",
    "add_tokens",
    "alice_api",
    "anthropic_api",
    "anthropic_count_tokens",
    "anthropic_messages",
    "build_messages",
    "cancel_response",
    "completions",
    "create_response",
    "delete_response",
    "duckai_api",
    "embeddings_not_supported",
    "favicon",
    "get_model",
    "get_response",
    "get_response_input_items",
    "gigachat_api",
    "health",
    "image_edits",
    "image_generations",
    "image_variations",
    "lifespan",
    "list_models",
    "model_refresh_loop",
    "moderations_not_supported",
    "normalize_usage",
    "provider_enabled",
    "provider_models",
    "provider_needs_api_key",
    "provider_pool",
    "qwen_api",
    "refresh_models",
    "refresh_provider_models",
    "request_body",
    "resolve_ca",
    "responses_api",
    "root",
    "settings",
    "toolemu",
    "unknown_v1_route",
    "usage_stats",
]
