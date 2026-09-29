import base64 as b64
import functools
import inspect
import json
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

import danyapi.api.chats as chats_mod
from danyapi import tools as toolemu
from danyapi.accounts import AccountPoolBusy
from danyapi.alice import api as alice_api
from danyapi.api.openai import ChatCompletionRequest, ChatMessage, CompletionRequest, FileSpec, app
from danyapi.duckai import api as duckai_api
from danyapi.gigachat import api as gigachat_api

GIGACHAT_ACCEPTED = set(inspect.signature(gigachat_api.collect_non_stream).parameters)
ALICE_ACCEPTED = set(inspect.signature(alice_api.collect_non_stream).parameters)
DUCKAI_ACCEPTED = set(inspect.signature(duckai_api.collect_non_stream).parameters)


class _Sessions:
    def can_reuse(self, session_id, **kwargs):
        return False

    def get(self, session_id):
        return None


class _Account:
    def __init__(self):
        self.sem = MagicMock()
        self.sessions = _Sessions()
        self.label = "acct#0"
        self.client = MagicMock()


class _Pool:
    def __init__(self, account=None, sid=None):
        self.account = account or _Account()
        self.sid = sid
        self.acquire = AsyncMock(return_value=(self.account, self.sid))
        self.resolve_context = MagicMock(return_value=None)
        self.accounts = [self.account]


def _request(headers=None):
    class _Request:
        def __init__(self):
            self.headers = headers or {}
            self.client = None
            self.url = type("_Url", (), {"path": "/v1/chat/completions"})()

    return _Request()


class _ChatResponse:
    def __init__(self, chunks):
        self.body_iterator = _agen(*chunks)


async def _agen(*chunks):
    for chunk in chunks:
        yield chunk


async def _drain(stream):
    return [line async for line in stream]


def _dispatcher(handler):
    async def dispatcher(model, request):
        return handler

    return dispatcher


def _file(name="notes.txt", content=b"file"):
    return FileSpec(name=name, content=b64.b64encode(content).decode(), content_type="text/plain")


def _chat(model="GigaChat-Pro", messages=None, **kwargs):
    return ChatCompletionRequest(model=model, messages=messages if messages is not None else [ChatMessage(role="user", content="hi")], **kwargs)


@pytest.fixture(autouse=True)
def _isolated_state(monkeypatch):
    monkeypatch.setattr(chats_mod, "_byok_mode", lambda: False)
    saved = dict(app.state._state)
    chats_mod._SESSION_OWNERS.clear()
    for attr in ("pool", "qwen_pool", "gigachat_pool", "alice_pool", "duckai_pool"):
        setattr(app.state, attr, None)
    app.state.deepseek_models = []
    yield
    chats_mod._SESSION_OWNERS.clear()
    app.state._state.clear()
    app.state._state.update(saved)


def test_chat_handler_rejects_an_unknown_provider():
    with pytest.raises(HTTPException) as excinfo:
        chats_mod._chat_handler("nope")
    assert excinfo.value.status_code == 404
    assert excinfo.value.detail == "Unknown provider: nope"


def test_chat_handler_rejects_a_stale_entry(monkeypatch):
    monkeypatch.setitem(chats_mod.CHAT_HANDLERS, "gigachat", "_chat_completions_removed")
    with pytest.raises(HTTPException) as excinfo:
        chats_mod._chat_handler("gigachat")
    assert excinfo.value.status_code == 500
    assert excinfo.value.detail == "chat handler for gigachat is not callable"


def test_validate_chat_handlers_fails_on_a_stale_entry(monkeypatch):
    monkeypatch.setitem(chats_mod.CHAT_HANDLERS, "duckai", "_chat_completions_removed")
    with pytest.raises(RuntimeError) as excinfo:
        chats_mod._validate_chat_handlers()
    assert str(excinfo.value) == "CHAT_HANDLERS references undefined handlers: _chat_completions_removed"
    monkeypatch.setitem(chats_mod.CHAT_HANDLERS, "duckai", "_chat_completions_duckai")
    chats_mod._validate_chat_handlers()


def test_check_chat_request_limits_caps_the_declared_body():
    request = _request({"content-length": str(chats_mod.MAX_CHAT_BODY_BYTES + 1)})
    with pytest.raises(HTTPException) as excinfo:
        chats_mod._check_chat_request_limits(_chat(), request)
    assert excinfo.value.status_code == 413
    assert excinfo.value.detail == f"request body too large, max {chats_mod.MAX_CHAT_BODY_BYTES // (1024 * 1024)} MB"
    chats_mod._check_chat_request_limits(_chat(), _request({"content-length": str(chats_mod.MAX_CHAT_BODY_BYTES)}))
    chats_mod._check_chat_request_limits(_chat(), _request())


def test_check_chat_request_limits_rejects_a_bad_content_length():
    with pytest.raises(HTTPException) as excinfo:
        chats_mod._check_chat_request_limits(_chat(), _request({"content-length": "abc"}))
    assert excinfo.value.status_code == 400
    assert excinfo.value.detail == "invalid content-length header"


def test_check_chat_request_limits_caps_the_message_count():
    req = _chat(messages=[ChatMessage(role="user", content="x") for _ in range(chats_mod.MAX_MESSAGES_PER_REQUEST + 1)])
    with pytest.raises(HTTPException) as excinfo:
        chats_mod._check_chat_request_limits(req, _request())
    assert excinfo.value.status_code == 400
    assert excinfo.value.detail == f"too many messages: max {chats_mod.MAX_MESSAGES_PER_REQUEST} per request"
    ok = _chat(messages=[ChatMessage(role="user", content="x") for _ in range(chats_mod.MAX_MESSAGES_PER_REQUEST)])
    chats_mod._check_chat_request_limits(ok, _request())


def test_chat_request_limits_return_the_documented_status_codes():
    client = TestClient(app)
    too_many = client.post(
        "/v1/chat/completions", json={"model": "GigaChat-Pro", "messages": [{"role": "user", "content": "x"}] * (chats_mod.MAX_MESSAGES_PER_REQUEST + 1)}
    )
    assert too_many.status_code == 400
    assert too_many.json()["error"]["message"] == f"too many messages: max {chats_mod.MAX_MESSAGES_PER_REQUEST} per request"
    oversized = client.post(
        "/v1/chat/completions",
        json={"model": "GigaChat-Pro", "messages": [{"role": "user", "content": "x" * chats_mod.MAX_CHAT_BODY_BYTES}]},
    )
    assert oversized.status_code == 413
    assert oversized.json()["error"]["message"] == f"request body too large, max {chats_mod.MAX_CHAT_BODY_BYTES // (1024 * 1024)} MB"
    client.close()


async def test_chat_dispatcher_resolves_the_deepseek_model_before_the_byok_pool(monkeypatch):
    app.state.deepseek_models = [{"id": "deepseek-chat", "upstream_type": "chat"}]
    monkeypatch.setattr(chats_mod, "_byok_mode", lambda: True)
    calls = []

    async def pool_for(provider, request):
        calls.append(provider)
        return _Pool()

    monkeypatch.setattr(chats_mod, "_byok_pool_for", pool_for)
    with pytest.raises(HTTPException) as excinfo:
        await chats_mod._chat_dispatcher("deepseek-unknown", _request())
    assert excinfo.value.status_code == 404
    assert excinfo.value.detail == "Unknown model: deepseek-unknown"
    assert calls == []
    dispatch = await chats_mod._chat_dispatcher("deepseek-chat", _request())
    assert calls == ["deepseek"]
    assert isinstance(dispatch, functools.partial)
    assert dispatch.func is chats_mod._chat_completions_deepseek


async def test_chat_dispatcher_outside_byok_returns_the_bare_handler(monkeypatch):
    resolved = []

    async def pool_for(provider, request):
        resolved.append(provider)
        return _Pool()

    monkeypatch.setattr(chats_mod, "_byok_pool_for", pool_for)
    assert await chats_mod._chat_dispatcher("GigaChat-Pro", _request()) is chats_mod._chat_completions_gigachat
    assert resolved == []


def test_caller_scope_reads_the_byok_caller_id(monkeypatch):
    monkeypatch.setattr(chats_mod, "_byok_caller_id", lambda: "caller-1")
    assert chats_mod._caller_scope() == "caller-1"


def test_bind_session_owner_ignores_empty_inputs_and_rejects_a_foreign_scope(caplog):
    chats_mod._bind_session_owner("", "u:a")
    chats_mod._bind_session_owner("s1", "")
    assert chats_mod._SESSION_OWNERS == {}
    chats_mod._bind_session_owner("s1", "u:a")
    chats_mod._bind_session_owner("s1", "u:a")
    assert chats_mod._SESSION_OWNERS == {"s1": "u:a"}
    with caplog.at_level(logging.WARNING, logger="danyapi.api"):
        with pytest.raises(HTTPException) as excinfo:
            chats_mod._bind_session_owner("s1", "u:b")
    assert excinfo.value.status_code == 403
    assert excinfo.value.detail == "session_id belongs to another client"
    assert "rejected session_id reuse across callers" in caplog.records[0].getMessage()


def test_bind_session_owner_evicts_the_oldest_entry_over_the_limit():
    for index in range(chats_mod.MAX_SESSION_OWNERS):
        chats_mod._bind_session_owner(f"s{index}", "u:a")
    assert len(chats_mod._SESSION_OWNERS) == chats_mod.MAX_SESSION_OWNERS
    chats_mod._bind_session_owner("newest", "u:a")
    assert len(chats_mod._SESSION_OWNERS) == chats_mod.MAX_SESSION_OWNERS
    assert "s0" not in chats_mod._SESSION_OWNERS
    assert "newest" in chats_mod._SESSION_OWNERS
    chats_mod._bind_session_owner("newest", "u:a")
    assert list(chats_mod._SESSION_OWNERS)[-1] == "newest"


def test_request_scope_prefers_the_user_then_the_byok_caller(monkeypatch):
    assert chats_mod._request_scope(_chat(user="alice")) == "u:alice"
    assert chats_mod._request_scope(_chat(user="")) is None
    assert chats_mod._request_scope(_chat()) is None
    monkeypatch.setattr(chats_mod, "_byok_mode", lambda: True)
    monkeypatch.setattr(chats_mod, "_caller_scope", lambda: "caller-1")
    assert chats_mod._request_scope(_chat()) == "k:caller-1"
    monkeypatch.setattr(chats_mod, "_caller_scope", lambda: "")
    assert chats_mod._request_scope(_chat()) is None


def test_context_sequence_digest_differs_between_scopes():
    messages = [ChatMessage(role="user", content="the same prompt")]
    first = toolemu.context_sequence(messages, user="u:alice")
    second = toolemu.context_sequence(messages, user="u:bob")
    assert first != second
    assert first == toolemu.context_sequence(messages, user="u:alice")
    assert first != toolemu.context_sequence(messages, user=None)


async def test_acquire_and_build_skips_the_digest_without_a_scope(monkeypatch):
    seen = []
    monkeypatch.setattr(chats_mod.toolemu, "context_sequence", lambda messages, user=None: seen.append(user) or ("x",))
    bound = []
    monkeypatch.setattr(chats_mod, "_bind_session_owner", lambda session_id, scope: bound.append((session_id, scope)))
    pool = _Pool()
    account, existing_sid, context_seq, prompt, tool_mode, cached = await chats_mod._acquire_and_build(
        pool, _chat(session_id="sess-1"), tools=None, tool_choice=None
    )
    assert seen == []
    assert bound == []
    assert context_seq == ()
    assert existing_sid == "sess-1"
    assert account is pool.account
    assert cached is None
    assert tool_mode is False
    assert "hi" in prompt
    pool.resolve_context.assert_not_called()


async def test_acquire_and_build_binds_the_session_owner_and_computes_the_digest(monkeypatch):
    seen = []
    monkeypatch.setattr(chats_mod.toolemu, "context_sequence", lambda messages, user=None: seen.append(user) or ("digest",))
    bound = []
    monkeypatch.setattr(chats_mod, "_bind_session_owner", lambda session_id, scope: bound.append((session_id, scope)))
    pool = _Pool()
    account, existing_sid, context_seq, _prompt, _tool_mode, _cached = await chats_mod._acquire_and_build(
        pool, _chat(user="alice", session_id="sess-1"), tools=None, tool_choice=None
    )
    assert seen == ["u:alice"]
    assert bound == [("sess-1", "u:alice")]
    assert context_seq == ("digest",)
    assert existing_sid == "sess-1"
    assert pool.acquire.await_args.args[0] == "sess-1"
    assert account is pool.account


async def test_acquire_and_build_uses_a_cached_context_session_when_no_session_id_is_given(monkeypatch):
    monkeypatch.setattr(chats_mod.toolemu, "context_sequence", lambda messages, user=None: ("digest",))
    pool = _Pool(sid="resolved-sid")
    pool.resolve_context.return_value = "resolved"
    _account, existing_sid, context_seq, _prompt, _tool_mode, _cached = await chats_mod._acquire_and_build(
        pool, _chat(user="alice"), tools=None, tool_choice=None
    )
    pool.resolve_context.assert_called_once_with(("digest",))
    assert pool.acquire.await_args.args[0] == "resolved"
    assert existing_sid == "resolved-sid"
    assert context_seq == ("digest",)


async def test_acquire_and_build_turns_a_prompt_error_into_a_bad_request(monkeypatch):
    def boom(*args, **kwargs):
        raise ValueError("tool schema is broken")

    monkeypatch.setattr(chats_mod.toolemu, "build_prompt", boom)
    with pytest.raises(HTTPException) as excinfo:
        await chats_mod._acquire_and_build(_Pool(), _chat(), tools=None, tool_choice=None)
    assert excinfo.value.status_code == 400
    assert excinfo.value.detail == "tool schema is broken"


def test_completion_prompts_accepts_the_documented_shapes():
    assert chats_mod._completion_prompts("one") == ["one"]
    assert chats_mod._completion_prompts(["a", "b"]) == ["a", "b"]
    assert chats_mod._completion_prompts([["a", "b"], "c"]) == ["a b", "c"]


@pytest.mark.parametrize(
    ("prompt", "detail"),
    [
        ("   ", "prompt must not be empty"),
        ([""], "prompt must not contain empty strings"),
        (["  ", "a"], "prompt must not contain empty strings"),
        ([[]], "prompt must not contain empty strings"),
        ([["a"], ""], "prompt must not contain empty strings"),
        ([5.5], "prompt must be a string, a list of strings, or a list of token lists"),
        (5, "prompt must be a string, a list of strings, or a list of token lists"),
        (None, "prompt must be a string, a list of strings, or a list of token lists"),
        ({}, "prompt must be a string, a list of strings, or a list of token lists"),
        ([], "prompt must not be empty"),
    ],
)
def test_completion_prompts_rejects_blank_and_malformed_input(prompt, detail):
    with pytest.raises(HTTPException) as excinfo:
        chats_mod._completion_prompts(prompt)
    assert excinfo.value.status_code == 400
    assert excinfo.value.detail == detail


def test_completion_chat_request_rejects_a_suffix():
    req = CompletionRequest(model="GigaChat-Pro", prompt="hi", suffix="\nbye")
    with pytest.raises(HTTPException) as excinfo:
        chats_mod._completion_chat_request(req, "hi", False, 1)
    assert excinfo.value.status_code == 400
    assert excinfo.value.detail == "suffix is not supported by the upstream providers"


def test_completion_chat_request_drops_the_session_id_for_multi_prompt_requests():
    req = CompletionRequest(model="GigaChat-Pro", prompt="hi", session_id="s1", max_tokens=32, n=2, stop=["x"], presence_penalty=0.1, logit_bias={"a": 1})
    single = chats_mod._completion_chat_request(req, "hi", True, 1)
    assert single.session_id == "s1"
    assert single.stream is True
    assert single.max_tokens == 32
    assert single.n == 2
    assert single.stop == ["x"]
    assert single.presence_penalty == 0.1
    assert single.logit_bias == {"a": 1}
    multi = chats_mod._completion_chat_request(req, "hi", False, 2)
    assert multi.session_id is None
    assert (multi.messages[0].role, multi.messages[0].content) == ("user", "hi")


def test_legacy_text_reads_text_parts_out_of_list_content():
    assert chats_mod._legacy_text("plain") == "plain"
    assert chats_mod._legacy_text(7) == ""
    assert chats_mod._legacy_text(None) == ""
    assert (
        chats_mod._legacy_text([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}, "c", 5, {"text": ""}, {"text": 7}, {"type": "image_url"}]) == "ab"
    )


def test_legacy_choice_from_chat_defaults_the_message_and_finish_reason():
    assert chats_mod._legacy_choice_from_chat({}, 2) == {"index": 2, "text": "", "logprobs": None, "finish_reason": "stop"}
    assert chats_mod._legacy_choice_from_chat({"message": {"content": [{"type": "text", "text": "hi"}]}, "finish_reason": "length"}, 0) == {
        "index": 0,
        "text": "hi",
        "logprobs": None,
        "finish_reason": "length",
    }


def test_exception_detail_handles_http_exceptions_and_other_errors():
    assert chats_mod._exception_detail(HTTPException(400, "bad request")) == "bad request"
    assert chats_mod._exception_detail(HTTPException(400, {"error": {"message": "m"}})) == "400: {'error': {'message': 'm'}}"
    assert chats_mod._exception_detail(RuntimeError("boom")) == "boom"


def test_safe_error_message_bounds_and_types_every_value():
    assert chats_mod._safe_error_message("  padded  ") == "padded"
    assert chats_mod._safe_error_message("") == "upstream error"
    assert chats_mod._safe_error_message("   ") == "upstream error"
    assert chats_mod._safe_error_message(7) == "7"
    assert chats_mod._safe_error_message(1.5) == "1.5"
    assert chats_mod._safe_error_message(True) == "True"
    assert chats_mod._safe_error_message(None) == "null"
    assert chats_mod._safe_error_message({"message": "m"}) == '{"message": "m"}'
    assert chats_mod._safe_error_message({1, 2}) == '"{1, 2}"'
    assert len(chats_mod._safe_error_message("x" * 5000)) == chats_mod.MAX_PROVIDER_ERROR_CHARS


def test_translate_chat_chunk_to_completion_maps_every_field():
    piece = chats_mod._translate_chat_chunk_to_completion(
        {"id": "c1", "created": 7, "model": "m", "usage": {"total_tokens": 3}, "choices": [{"index": 2, "delta": {"content": "hi"}, "finish_reason": "stop"}]}
    )
    assert piece == {
        "id": "c1",
        "object": "text_completion",
        "created": 7,
        "model": "m",
        "usage": {"total_tokens": 3},
        "choices": [{"index": 2, "text": "hi", "logprobs": None, "finish_reason": "stop"}],
    }


def test_translate_chat_chunk_to_completion_tolerates_malformed_input():
    assert chats_mod._translate_chat_chunk_to_completion({})["choices"] == []
    assert chats_mod._translate_chat_chunk_to_completion({})["created"] > 0
    assert chats_mod._translate_chat_chunk_to_completion({"choices": "nope"})["choices"] == []
    assert chats_mod._translate_chat_chunk_to_completion({"choices": ["nope"]})["choices"] == []
    coerced = chats_mod._translate_chat_chunk_to_completion({"choices": [{"index": "1", "delta": {"content": 7}}, {"index": True, "delta": None}]})
    assert [choice["index"] for choice in coerced["choices"]] == [0, 0]
    assert [choice["text"] for choice in coerced["choices"]] == ["", ""]
    assert coerced["choices"][0]["finish_reason"] is None


def test_translate_chat_chunk_to_completion_bounds_an_upstream_error():
    piece = chats_mod._translate_chat_chunk_to_completion({"error": {"message": "x" * 5000}})
    assert len(piece["error"]["message"]) == chats_mod.MAX_PROVIDER_ERROR_CHARS
    assert chats_mod._translate_chat_chunk_to_completion({"error": "plain"})["error"] == {"message": "plain"}


def test_usage_count_type_checks_provider_usage():
    assert chats_mod._usage_count({"prompt_tokens": "7"}, "prompt_tokens") == 0
    assert chats_mod._usage_count({"prompt_tokens": True}, "prompt_tokens") == 0
    assert chats_mod._usage_count({"prompt_tokens": None}, "prompt_tokens") == 0
    assert chats_mod._usage_count({"prompt_tokens": 7.9}, "prompt_tokens") == 7
    assert chats_mod._usage_count({}, "prompt_tokens") == 0


async def test_translate_completion_stream_rewrites_chunks_and_passes_everything_else_through():
    chunks = [
        ": comment\n\n",
        "data: not json\n\n",
        "data: [DONE]\n\n",
        'data: {"id": "c1", "choices": [{"index": 0, "delta": {"content": "hi"}}]}\n\n',
    ]
    lines = await _drain(chats_mod._translate_completion_stream(_agen(*chunks)))
    assert lines[0] == ": comment\n\n"
    assert lines[1] == "data: not json\n\n"
    payload = json.loads(lines[2][len("data: ") :].strip())
    assert payload["id"] == "c1"
    assert payload["object"] == "text_completion"
    assert payload["choices"] == [{"index": 0, "text": "hi", "logprobs": None, "finish_reason": None}]


async def test_completions_stream_emits_a_terminating_error_chunk_and_done():
    calls = []

    async def dispatch(chat_req):
        calls.append(chat_req.messages[0].content)
        if len(calls) == 2:
            raise HTTPException(503, "provider pool is gone")
        return _ChatResponse(['data: {"id": "c1", "choices": [{"index": 0, "delta": {"content": "hi"}}]}\n\n'])

    req = CompletionRequest(model="GigaChat-Pro", prompt=["first", "second", "third"])
    lines = await _drain(chats_mod._completions_stream(req, ["first", "second", "third"], dispatch))
    assert calls == ["first", "second"]
    assert lines[-1] == "data: [DONE]\n\n"
    assert json.loads(lines[0][len("data: ") :].strip())["choices"][0]["text"] == "hi"
    error = json.loads(lines[1][len("data: ") :].strip())
    assert error["error"] == {"message": "provider pool is gone"}
    assert error["model"] == "GigaChat-Pro"
    assert error["object"] == "text_completion"
    assert error["choices"] == []


async def test_completions_stream_terminates_after_a_prompt_failure(caplog):
    async def dispatch(chat_req):
        raise RuntimeError("upstream exploded")

    req = CompletionRequest(model="GigaChat-Pro", prompt="only")
    with caplog.at_level(logging.WARNING, logger="danyapi.api"):
        lines = await _drain(chats_mod._completions_stream(req, ["only"], dispatch))
    assert lines[-1] == "data: [DONE]\n\n"
    assert len(lines) == 2
    assert "upstream exploded" in caplog.records[0].getMessage()


async def test_completions_non_stream_aggregates_every_prompt(monkeypatch):
    seen = []

    async def dispatch(chat_req):
        seen.append(chat_req.messages[0].content)
        return {
            "id": "c1",
            "created": 42,
            "choices": [
                {"message": {"content": [{"type": "text", "text": "!"}]}, "finish_reason": "stop"},
                "junk",
                {"message": "not a dict"},
            ],
            "usage": {"prompt_tokens": "x", "completion_tokens": 2, "total_tokens": 5},
        }

    monkeypatch.setattr(chats_mod, "_chat_dispatcher", _dispatcher(dispatch))
    req = CompletionRequest(model="GigaChat-Pro", prompt=["a", "b"])
    body = await chats_mod.completions(req, _request())
    assert seen == ["a", "b"]
    assert body["id"] == "c1"
    assert body["created"] == 42
    assert body["model"] == "GigaChat-Pro"
    assert [choice["index"] for choice in body["choices"]] == [0, 1, 2, 3]
    assert [choice["text"] for choice in body["choices"]] == ["!", "", "!", ""]
    assert body["usage"] == {"prompt_tokens": 0, "completion_tokens": 4, "total_tokens": 10}


async def test_completions_non_stream_falls_back_to_a_generated_id(monkeypatch):
    async def dispatch(chat_req):
        return {"choices": "not a list"}

    monkeypatch.setattr(chats_mod, "_chat_dispatcher", _dispatcher(dispatch))
    body = await chats_mod.completions(CompletionRequest(model="GigaChat-Pro", prompt="a"), _request())
    assert body["id"].startswith("cmpl-")
    assert body["created"] > 0
    assert body["choices"] == []
    assert body["usage"] == {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


async def test_completions_non_stream_handles_a_provider_result_without_usage(monkeypatch):
    async def dispatch(chat_req):
        return {"id": "c1", "choices": [{"message": {"content": "hi"}}], "usage": "not a dict"}

    monkeypatch.setattr(chats_mod, "_chat_dispatcher", _dispatcher(dispatch))
    body = await chats_mod.completions(CompletionRequest(model="GigaChat-Pro", prompt="a"), _request())
    assert body["choices"][0]["text"] == "hi"
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["usage"] == {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


async def test_completions_rejects_too_many_prompts():
    req = CompletionRequest(model="GigaChat-Pro", prompt=[f"p{index}" for index in range(chats_mod.MAX_COMPLETION_PROMPTS + 1)])
    with pytest.raises(HTTPException) as excinfo:
        await chats_mod.completions(req, _request())
    assert excinfo.value.status_code == 400
    assert excinfo.value.detail == f"too many prompts: max {chats_mod.MAX_COMPLETION_PROMPTS} per request"


async def test_completions_stream_endpoint_returns_sse(monkeypatch):
    async def dispatch(chat_req):
        return _ChatResponse(['data: {"id": "c1", "choices": [{"index": 0, "delta": {"content": "hi"}}]}\n\n'])

    monkeypatch.setattr(chats_mod, "_chat_dispatcher", _dispatcher(dispatch))
    response = await chats_mod.completions(CompletionRequest(model="GigaChat-Pro", prompt="a", stream=True), _request())
    assert response.media_type == "text/event-stream"
    assert response.headers["cache-control"] == "no-cache"
    lines = await _drain(response.body_iterator)
    assert lines[-1] == "data: [DONE]\n\n"


def test_embeddings_and_moderations_are_not_implemented():
    client = TestClient(app)
    embeddings = client.post("/v1/embeddings", json={"model": "GigaChat-Pro", "input": "hi"})
    assert embeddings.status_code == 501
    assert embeddings.json()["error"]["message"] == "embeddings are not supported by DanyAPI"
    moderations = client.post("/v1/moderations", json={"input": "hi"})
    assert moderations.status_code == 501
    assert moderations.json()["error"]["message"] == "moderations are not supported by DanyAPI"
    client.close()


def test_embeddings_and_moderations_demand_a_key_in_byok_mode(monkeypatch):
    monkeypatch.setattr(chats_mod, "_byok_mode", lambda: True)
    client = TestClient(app)
    no_key = client.post("/v1/embeddings", json={"input": "hi"})
    assert no_key.status_code == 401
    assert no_key.json()["error"]["message"] == "api key is required in byok mode, embeddings cannot be used without one"
    with_key = client.post("/v1/embeddings", json={"input": "hi"}, headers={"x-api-key": "k"})
    assert with_key.status_code == 501
    no_key_moderations = client.post("/v1/moderations", json={"input": "hi"})
    assert no_key_moderations.json()["error"]["message"] == "api key is required in byok mode, moderations cannot be used without one"
    with_key_moderations = client.post("/v1/moderations", json={"input": "hi"}, headers={"x-api-key": "k"})
    assert with_key_moderations.status_code == 501
    client.close()


async def test_reject_unsupported_endpoint_never_looks_for_a_key_outside_byok(monkeypatch):
    seen = []

    async def extract(request):
        seen.append(request)
        return "key"

    monkeypatch.setattr(chats_mod, "_extract_request_api_key", extract)
    await chats_mod._reject_unsupported_endpoint(_request(), "embeddings")
    assert seen == []


def test_materialize_tools_converts_legacy_functions():
    req = ChatCompletionRequest(
        model="GigaChat-Pro",
        messages=[ChatMessage(role="user", content="hi")],
        functions=[{"name": "f", "description": "d", "parameters": {"type": "object"}}, "junk", {"description": "no name"}],
    )
    tools, tool_choice = chats_mod._materialize_tools(req)
    assert tools == [
        {"type": "function", "function": {"name": "f", "description": "d", "parameters": {"type": "object"}}},
        {"type": "function", "function": {"name": "", "description": "no name"}},
    ]
    assert tool_choice is None


def test_materialize_tools_merges_functions_with_existing_tools():
    req = ChatCompletionRequest(
        model="GigaChat-Pro",
        messages=[ChatMessage(role="user", content="hi")],
        tools=[{"type": "function", "function": {"name": "existing"}}],
        functions=[{"name": "f"}],
    )
    tools, _ = chats_mod._materialize_tools(req)
    assert [tool["function"]["name"] for tool in tools] == ["existing", "f"]


def test_materialize_tools_ignores_functions_that_convert_to_nothing():
    req = ChatCompletionRequest(
        model="GigaChat-Pro",
        messages=[ChatMessage(role="user", content="hi")],
        functions=["junk"],
        tools=[{"type": "function", "function": {"name": "existing"}}],
    )
    tools, _ = chats_mod._materialize_tools(req)
    assert tools == [{"type": "function", "function": {"name": "existing"}}]


@pytest.mark.parametrize(
    ("function_call", "expected"),
    [
        ("auto", "auto"),
        ("none", "none"),
        ("f", {"type": "function", "function": {"name": "f"}}),
        ({"name": "g"}, {"type": "function", "function": {"name": "g"}}),
        ({"name": ""}, None),
        ("", None),
        (5, None),
    ],
)
def test_materialize_tools_maps_legacy_function_call(function_call, expected):
    req = ChatCompletionRequest(model="m", messages=[], function_call=function_call)
    assert chats_mod._materialize_tools(req) == (None, expected)


def test_materialize_tools_keeps_an_explicit_tool_choice():
    req = ChatCompletionRequest(model="m", messages=[], function_call="f", tool_choice="none")
    assert chats_mod._materialize_tools(req) == (None, "none")


async def test_qwen_rejects_a_non_image_attachment(monkeypatch):
    captured = {}

    async def collect_non_stream(**kwargs):
        captured.update(kwargs)
        return {"id": "c1", "choices": []}

    monkeypatch.setattr(chats_mod.qwen_api, "collect_non_stream", collect_non_stream)
    encoded = []

    async def b64encode(data):
        encoded.append(data)
        return b64.b64encode(data).decode()

    monkeypatch.setattr(chats_mod, "_b64encode", b64encode)
    data_uri = f"data:image/png;base64,{b64.b64encode(b'png').decode()}"
    image = ChatCompletionRequest(
        model="qwen3-max",
        messages=[ChatMessage(role="user", content=[{"type": "text", "text": "look"}, {"type": "image_url", "image_url": data_uri}])],
    )
    await chats_mod._chat_completions_qwen(image, pool=_Pool())
    assert encoded == [b"png"]
    assert f"![image]({data_uri})" in captured["prompt"]

    document = ChatCompletionRequest(model="qwen3-max", messages=[ChatMessage(role="user", content="hi")], files=[_file()])
    with pytest.raises(HTTPException) as excinfo:
        await chats_mod._chat_completions_qwen(document, pool=_Pool())
    assert excinfo.value.status_code == 400
    assert excinfo.value.detail == "qwen only supports image attachments, use deepseek for files"


def test_reject_unsupported_params_allows_the_defaults():
    req = ChatCompletionRequest(model="GigaChat-Pro", messages=[], n=1, logit_bias={}, presence_penalty=None)
    chats_mod._reject_unsupported_params(req, "gigachat", chats_mod.GIGACHAT_UNSUPPORTED_PARAMS)
    with pytest.raises(HTTPException) as excinfo:
        chats_mod._reject_unsupported_params(ChatCompletionRequest(model="m", messages=[], n=2), "gigachat", chats_mod.GIGACHAT_UNSUPPORTED_PARAMS)
    assert excinfo.value.status_code == 400
    assert excinfo.value.detail == "gigachat does not support the n parameter"


async def test_gigachat_rejects_the_unsupported_params(monkeypatch):
    for name, value in (("n", 2), ("presence_penalty", 0.5), ("frequency_penalty", 0.5), ("logit_bias", {"a": 1})):
        reached = []

        async def collect_non_stream(_reached=reached, **kwargs):
            _reached.append(kwargs)
            return {"id": "c1", "choices": []}

        monkeypatch.setattr(chats_mod.gigachat_api, "collect_non_stream", collect_non_stream)
        with pytest.raises(HTTPException) as excinfo:
            await chats_mod._chat_completions_gigachat(_chat(**{name: value}), pool=_Pool())
        assert excinfo.value.status_code == 400
        assert excinfo.value.detail == f"gigachat does not support the {name} parameter"
        assert reached == []


async def test_gigachat_forwards_only_supported_params(monkeypatch):
    captured = {}

    async def collect_non_stream(**kwargs):
        captured.update(kwargs)
        return {"id": "c1", "choices": []}

    monkeypatch.setattr(chats_mod.gigachat_api, "collect_non_stream", collect_non_stream)
    body = await chats_mod._chat_completions_gigachat(
        _chat(temperature=0.5, top_p=0.9, max_tokens=64, stop=["x"], user="alice", session_id="s1"), pool=_Pool(sid="s1")
    )
    assert body == {"id": "c1", "choices": []}
    assert set(captured) <= GIGACHAT_ACCEPTED
    assert "presence_penalty" not in captured
    assert "frequency_penalty" not in captured
    assert captured["max_tokens"] == 64
    assert captured["session_id"] == "s1"


async def test_gigachat_reports_a_missing_pool():
    with pytest.raises(HTTPException) as excinfo:
        await chats_mod._chat_completions_gigachat(_chat(), pool=None)
    assert excinfo.value.status_code == 503
    assert excinfo.value.detail == "gigachat provider is not configured"


async def test_gigachat_stream_wraps_the_provider_stream(monkeypatch):
    async def stream_openai(**kwargs):
        yield 'data: {"id": "c1", "choices": []}\n\n'
        yield "data: [DONE]\n\n"

    monkeypatch.setattr(chats_mod.gigachat_api, "stream_openai", stream_openai)
    response = await chats_mod._chat_completions_gigachat(_chat(stream=True, max_tokens=8), pool=_Pool())
    assert response.media_type == "text/event-stream"
    assert await _drain(response.body_iterator) == ['data: {"id": "c1", "choices": []}\n\n', "data: [DONE]\n\n"]


async def test_gigachat_maps_a_busy_pool_to_a_rate_limit(monkeypatch):
    async def collect_non_stream(**kwargs):
        raise AccountPoolBusy("all busy")

    monkeypatch.setattr(chats_mod.gigachat_api, "collect_non_stream", collect_non_stream)
    with pytest.raises(HTTPException) as excinfo:
        await chats_mod._chat_completions_gigachat(_chat(), pool=_Pool())
    assert excinfo.value.status_code == 429
    assert excinfo.value.detail == "all accounts are busy, try again later"


async def test_alice_rejects_files_and_the_unsupported_params():
    with pytest.raises(HTTPException) as excinfo:
        await chats_mod._chat_completions_alice(_chat(model="yagpt", files=[_file()]), pool=_Pool())
    assert excinfo.value.status_code == 400
    assert excinfo.value.detail == "alice does not support file attachments"
    for name, value in (("n", 2), ("top_p", 0.5), ("presence_penalty", 0.5), ("frequency_penalty", 0.5), ("logit_bias", {"a": 1})):
        with pytest.raises(HTTPException) as excinfo:
            await chats_mod._chat_completions_alice(_chat(model="yagpt", **{name: value}), pool=_Pool())
        assert excinfo.value.status_code == 400
        assert excinfo.value.detail == f"alice does not support the {name} parameter"


async def test_alice_forwards_only_supported_params(monkeypatch):
    captured = {}

    async def collect_non_stream(**kwargs):
        captured.update(kwargs)
        return {"id": "c1", "choices": []}

    monkeypatch.setattr(chats_mod.alice_api, "collect_non_stream", collect_non_stream)
    body = await chats_mod._chat_completions_alice(_chat(model="yagpt", stop=["x"], user="alice", session_id="s1"), pool=_Pool(sid="s1"))
    assert body == {"id": "c1", "choices": []}
    assert set(captured) <= ALICE_ACCEPTED
    assert captured["session_id"] == "s1"


async def test_alice_stream_and_busy_and_missing_pool(monkeypatch):
    async def stream_openai(**kwargs):
        yield 'data: {"id": "c1", "choices": []}\n\n'
        yield "data: [DONE]\n\n"

    monkeypatch.setattr(chats_mod.alice_api, "stream_openai", stream_openai)
    response = await chats_mod._chat_completions_alice(_chat(model="yagpt", stream=True), pool=_Pool())
    assert await _drain(response.body_iterator) == ['data: {"id": "c1", "choices": []}\n\n', "data: [DONE]\n\n"]

    async def collect_non_stream(**kwargs):
        raise AccountPoolBusy("busy")

    monkeypatch.setattr(chats_mod.alice_api, "collect_non_stream", collect_non_stream)
    with pytest.raises(HTTPException) as excinfo:
        await chats_mod._chat_completions_alice(_chat(model="yagpt"), pool=_Pool())
    assert excinfo.value.status_code == 429

    with pytest.raises(HTTPException) as excinfo:
        await chats_mod._chat_completions_alice(_chat(model="yagpt"), pool=None)
    assert excinfo.value.status_code == 503
    assert excinfo.value.detail == "alice provider is not configured (set ALICE_ENABLED=1 to enable)"


async def test_duckai_rejects_files_and_the_unsupported_params():
    with pytest.raises(HTTPException) as excinfo:
        await chats_mod._chat_completions_duckai(_chat(model="duckai-free", files=[_file()]), pool=_Pool())
    assert excinfo.value.status_code == 400
    assert excinfo.value.detail == "duckai does not support file attachments, send images inline instead"
    for name, value in (("n", 2), ("top_p", 0.5), ("presence_penalty", 0.5), ("frequency_penalty", 0.5), ("logit_bias", {"a": 1})):
        with pytest.raises(HTTPException) as excinfo:
            await chats_mod._chat_completions_duckai(_chat(model="duckai-free", **{name: value}), pool=_Pool())
        assert excinfo.value.status_code == 400
        assert excinfo.value.detail == f"duck.ai does not support the {name} parameter"


async def test_duckai_forwards_only_supported_params(monkeypatch):
    captured = {}

    async def collect_non_stream(**kwargs):
        captured.update(kwargs)
        return {"id": "c1", "choices": []}

    monkeypatch.setattr(chats_mod.duckai_api, "collect_non_stream", collect_non_stream)
    body = await chats_mod._chat_completions_duckai(
        _chat(model="duckai-free", thinking=True, search=True, stop=["x"], user="alice", session_id="s1"), pool=_Pool(sid="s1")
    )
    assert body == {"id": "c1", "choices": []}
    assert set(captured) <= DUCKAI_ACCEPTED
    assert captured["thinking"] is True
    assert captured["search"] is True
    assert captured["session_id"] == "s1"


async def test_duckai_stream_and_busy_and_missing_pool(monkeypatch):
    async def stream_openai(**kwargs):
        yield 'data: {"id": "c1", "choices": []}\n\n'
        yield "data: [DONE]\n\n"

    monkeypatch.setattr(chats_mod.duckai_api, "stream_openai", stream_openai)
    response = await chats_mod._chat_completions_duckai(_chat(model="duckai-free", stream=True), pool=_Pool())
    assert await _drain(response.body_iterator) == ['data: {"id": "c1", "choices": []}\n\n', "data: [DONE]\n\n"]

    async def collect_non_stream(**kwargs):
        raise AccountPoolBusy("busy")

    monkeypatch.setattr(chats_mod.duckai_api, "collect_non_stream", collect_non_stream)
    with pytest.raises(HTTPException) as excinfo:
        await chats_mod._chat_completions_duckai(_chat(model="duckai-free"), pool=_Pool())
    assert excinfo.value.status_code == 429

    with pytest.raises(HTTPException) as excinfo:
        await chats_mod._chat_completions_duckai(_chat(model="duckai-free"), pool=None)
    assert excinfo.value.status_code == 503
    assert excinfo.value.detail == "duckai provider is not configured (set DUCKAI_ENABLED=1 to enable)"
