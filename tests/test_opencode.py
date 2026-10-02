import asyncio
import json
import os

import httpx
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from danyapi.accounts import AccountPool
from danyapi.api import models as models_mod
from danyapi.api.core import _acquire_account
from danyapi.api.models import _resolve_provider
from danyapi.api.openai import app
from danyapi.api.schemas import ChatMessage
from danyapi.api.state import BYOK_PROVIDERS, KEY_OPTIONAL_PROVIDERS, KEYLESS_PROVIDERS, MODEL_ATTRS, POOL_ATTRS_BY_PROVIDER
from danyapi.opencode import client as client_mod
from danyapi.opencode import messages as om
from danyapi.opencode.accounts import OpenCodeAccount
from danyapi.opencode.api import _status_for
from danyapi.opencode.client import BASE_URL, USER_AGENT, OpenCodeClient, OpenCodeError, error_message, upstream_model

KEY = "sk-zen-test-key"


def _client(transport: httpx.MockTransport, key: str = KEY) -> OpenCodeClient:
    client = OpenCodeClient(key=key)
    client.http = httpx.AsyncClient(transport=transport, base_url=BASE_URL)
    return client


def _models() -> dict:
    return {"object": "list", "data": [{"id": "space-bunny-free"}, {"id": "qwen3.8-max"}, {"id": "gpt-5.6-sol"}]}


def _completion(content: str = "ok", reasoning: str = "", tool_calls: list | None = None) -> dict:
    message: dict = {"role": "assistant", "content": content}
    if reasoning:
        message["reasoning_content"] = reasoning
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return {
        "id": "chatcmpl-up",
        "created": 1790774725,
        "model": "space-bunny-free",
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 157, "completion_tokens": 8, "total_tokens": 165, "prompt_tokens_details": {"cached_tokens": 149}},
    }


def _msg(**kwargs) -> ChatMessage:
    return ChatMessage(**kwargs)


def _account(key: str = KEY) -> "_Account":
    return _Account(key)


STREAM_CHUNKS = (
    'data: {"id":"c1","object":"chat.completion.chunk","choices":'
    '[{"index":0,"delta":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}\n\n'
    'data: {"id":"c1","object":"chat.completion.chunk","choices":[],'
    '"usage":{"prompt_tokens":5,"completion_tokens":1,"total_tokens":6}}\n\n'
    "data: [DONE]\n\n"
)


class _Account:
    """Minimal stand-in for OpenCodeAccount that records the outbound request."""

    def __init__(self, key: str = KEY) -> None:
        self.client = _client(httpx.MockTransport(self.handler), key)
        self.sem = asyncio.Semaphore(1)
        self.index = 0
        self.broken = False
        self.requests: list[httpx.Request] = []

    @property
    def label(self) -> str:
        return "opencode-acct#0"

    def mark_broken(self) -> None:
        self.broken = True

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json=_models())
        if json.loads(request.content).get("stream"):
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=STREAM_CHUNKS.encode())
        return httpx.Response(200, json=_completion())


def test_upstream_model_strips_only_the_routing_prefix():
    assert upstream_model("opencode/kimi-k3") == "kimi-k3"
    assert upstream_model("OPENCODE/kimi-k3") == "kimi-k3"
    assert upstream_model("kimi-k3") == "kimi-k3"
    assert upstream_model("  space-bunny-free  ") == "space-bunny-free"
    assert upstream_model("") == ""


def test_account_shape_and_slots():
    account = OpenCodeAccount(3, OpenCodeClient(key=KEY), stable_id="abc")
    assert OpenCodeAccount.__slots__ == ("broken", "broken_at", "client", "index", "sem", "stable_id")
    assert account.label == "opencode-acct#3"
    assert account.stable_id == "abc"
    assert account.broken is False
    account.mark_broken()
    assert account.broken is True
    assert account.broken_at is not None
    account.mark_broken()


def test_error_status_is_driven_by_the_body_type_not_the_code():
    assert _status_for(OpenCodeError(401, "Invalid API key.", "AuthError")) == 401
    assert _status_for(OpenCodeError(401, "not supported for format", "ModelError")) == 404
    assert _status_for(OpenCodeError(403, "free tier", "FreeTierError")) == 403
    assert _status_for(OpenCodeError(403, "not in your country", "RegionError")) == 403
    assert _status_for(OpenCodeError(402, "no balance", "BillingError")) == 402
    assert _status_for(OpenCodeError(429, "slow down", "RateLimitError")) == 429
    assert _status_for(OpenCodeError(401, "untyped")) == 401
    assert _status_for(OpenCodeError(404, "untyped")) == 404
    assert _status_for(OpenCodeError(429, "untyped")) == 429
    assert _status_for(OpenCodeError(400, "bad")) == 400
    assert _status_for(OpenCodeError(500, "boom")) == 502
    assert _status_for(OpenCodeError(500, "boom", "SomeUnknownError")) == 502


def test_only_auth_error_is_treated_as_a_bad_key():
    assert OpenCodeError(401, "x", "AuthError").is_auth is True
    assert OpenCodeError(401, "x", "ModelError").is_auth is False
    assert OpenCodeError(403, "x", "FreeTierError").is_auth is False
    assert OpenCodeError(403, "x", "RegionError").is_auth is False
    assert OpenCodeError(401, "x").is_auth is True
    assert OpenCodeError(403, "x").is_auth is True
    assert OpenCodeError(400, "x").is_auth is False


def test_error_message_is_read_from_the_nested_shape():
    payload = {"type": "error", "error": {"type": "AuthError", "message": "Missing API key."}}
    assert error_message(payload, 401) == ("AuthError", "Missing API key.")
    assert error_message({"message": "flat"}, 400) == ("", "flat")
    assert error_message(None, 500) == ("", "upstream returned 500")
    assert error_message({"error": "not a dict"}, 502) == ("", "upstream returned 502")


def test_error_detail_carries_the_type_hint():
    detail = OpenCodeError(403, "tier gone", "FreeTierError").detail
    assert detail.startswith("OpenCode Zen error: tier gone")
    assert "OPENCODE_KEYS" in detail
    assert "space-bunny-free" in detail
    assert "reserved for an authenticated account" in detail
    assert OpenCodeError(500, "").detail == "OpenCode Zen error: 500"


def test_the_free_tier_hint_does_not_blame_the_balance():
    assert "credit" not in client_mod.ERROR_TYPE_HINTS["FreeTierError"]
    assert "exhausted" not in client_mod.ERROR_TYPE_HINTS["FreeTierError"]


CATALOG = {
    "opencode": {
        "id": "opencode",
        "models": {
            "chatty": {"name": "Chatty", "cost": {"input": 1, "output": 2}, "limit": {"context": 4096}, "modalities": {"input": ["text"]}},
            "freebie": {"name": "Freebie", "cost": {"input": 0, "output": 0}, "limit": {"context": 8192}, "modalities": {"input": ["text", "image"]}},
            "responses-only": {"name": "Responses Only", "provider": {"npm": "@ai-sdk/openai"}},
            "anthropic-only": {"name": "Anthropic Only", "provider": {"npm": "@ai-sdk/anthropic"}},
            "broken": "not a dict",
        },
    }
}


def _catalog_transport(payload=None):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "models.opencode.ai":
            return httpx.Response(200, json=CATALOG if payload is None else payload)
        return httpx.Response(200, json={"object": "list", "data": [{"id": k} for k in ("chatty", "freebie", "responses-only", "anthropic-only")]})

    return handler


@pytest.mark.asyncio
async def test_fetch_catalog_reads_the_nested_provider_models(monkeypatch):
    monkeypatch.setattr(client_mod, "_CATALOG_CACHE", (None, 0.0))
    client = _client(httpx.MockTransport(_catalog_transport()))
    catalog = await client_mod.fetch_catalog(client)
    assert set(catalog) == {"chatty", "freebie", "responses-only", "anthropic-only"}
    assert catalog["freebie"]["name"] == "Freebie"


@pytest.mark.asyncio
async def test_fetch_catalog_is_cached_across_calls(monkeypatch):
    monkeypatch.setattr(client_mod, "_CATALOG_CACHE", (None, 0.0))
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json=CATALOG)

    client = _client(httpx.MockTransport(handler))
    await client_mod.fetch_catalog(client)
    await client_mod.fetch_catalog(client)
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_fetch_catalog_ignores_a_payload_without_the_provider(monkeypatch):
    monkeypatch.setattr(client_mod, "_CATALOG_CACHE", (None, 0.0))
    client = _client(httpx.MockTransport(_catalog_transport({"someone-else": {"models": {"x": {}}}})))
    assert await client_mod.fetch_catalog(client) == {}


@pytest.mark.asyncio
async def test_fetch_catalog_raises_on_a_non_json_body(monkeypatch):
    monkeypatch.setattr(client_mod, "_CATALOG_CACHE", (None, 0.0))
    client = _client(httpx.MockTransport(lambda r: httpx.Response(200, content=b"nope")))
    with pytest.raises(OpenCodeError):
        await client_mod.fetch_catalog(client)


@pytest.mark.asyncio
async def test_fetch_catalog_raises_on_an_error_status(monkeypatch):
    monkeypatch.setattr(client_mod, "_CATALOG_CACHE", (None, 0.0))
    client = _client(httpx.MockTransport(lambda r: httpx.Response(500, json={"error": {"type": "server_error", "message": "down"}})))
    with pytest.raises(OpenCodeError):
        await client_mod.fetch_catalog(client)


def test_model_format_defaults_to_chat_and_reads_the_override():
    assert client_mod.model_format({}) == client_mod.CHAT_FORMAT
    assert client_mod.model_format({"provider": {}}) == client_mod.CHAT_FORMAT
    assert client_mod.model_format({"provider": {"npm": "@ai-sdk/openai"}}) == "@ai-sdk/openai"
    assert client_mod.model_format({"provider": {"npm": ""}}) == client_mod.CHAT_FORMAT


def test_is_free_needs_both_directions_at_zero():
    assert client_mod.is_free({"cost": {"input": 0, "output": 0}}) is True
    assert client_mod.is_free({"cost": {"input": 0, "output": 1}}) is False
    assert client_mod.is_free({"cost": {"input": 1, "output": 0}}) is False
    assert client_mod.is_free({}) is False
    assert client_mod.is_free({"cost": "junk"}) is False


def test_context_limit_only_returns_a_positive_int():
    assert client_mod.context_limit({"limit": {"context": 4096}}) == 4096
    assert client_mod.context_limit({"limit": {"context": 0}}) is None
    assert client_mod.context_limit({"limit": {"context": "4096"}}) is None
    assert client_mod.context_limit({"limit": {}}) is None
    assert client_mod.context_limit({}) is None


@pytest.mark.asyncio
async def test_fetcher_hides_models_this_gateway_cannot_serve(monkeypatch):
    from danyapi.api.models import _fetch_opencode_models

    monkeypatch.setattr(client_mod, "_CATALOG_CACHE", (None, 0.0))
    client = _client(httpx.MockTransport(_catalog_transport()))
    models = await _fetch_opencode_models(client)
    assert [m["id"] for m in models] == ["freebie"]
    assert models[0]["name"] == "Freebie"
    assert models[0]["free"] is True
    assert models[0]["supports_vision"] is True


@pytest.mark.asyncio
async def test_fetcher_survives_a_broken_catalog(monkeypatch):
    from danyapi.api.models import _fetch_opencode_models

    async def _boom(_client):
        raise RuntimeError("catalog down")

    monkeypatch.setattr(models_mod.opencode_zen, "fetch_catalog", _boom)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"object": "list", "data": [{"id": "chatty"}, {"id": "freebie"}]})

    client = _client(httpx.MockTransport(handler))
    models = await _fetch_opencode_models(client)
    assert [m["id"] for m in models] == ["freebie"]
    assert models[0]["name"] == "freebie"
    assert "free" not in models[0]


def test_error_detail_is_truncated():
    assert len(OpenCodeError(500, "x" * 900).detail) < 400


def test_registries_all_carry_the_provider():
    assert "opencode" in BYOK_PROVIDERS
    assert "opencode" not in KEYLESS_PROVIDERS
    assert "opencode" in KEY_OPTIONAL_PROVIDERS
    assert MODEL_ATTRS["opencode"] == "opencode_models"
    assert POOL_ATTRS_BY_PROVIDER["opencode"] == "opencode_pool"


def test_the_provider_is_off_until_a_key_or_the_flag_is_given():
    from danyapi.config import Settings

    saved = {name: os.environ.pop(name, None) for name in ("OPENCODE_KEYS", "OPENCODE_ENABLED")}
    try:
        assert Settings().opencode_keys == []
        assert Settings().opencode_enabled is False
        os.environ["OPENCODE_ENABLED"] = "1"
        assert Settings().opencode_enabled is True
        os.environ["OPENCODE_KEYS"] = "k1,k2"
        assert Settings().opencode_keys == ["k1", "k2"]
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def test_prefix_wins_over_the_colliding_native_providers():
    assert _resolve_provider("opencode/qwen3.8-max") == "opencode"
    assert _resolve_provider("opencode/deepseek-v4.1-flash") == "opencode"
    assert _resolve_provider("opencode/anything-unknown") == "opencode"
    assert _resolve_provider("qwen3.8-max") == "qwen"
    assert _resolve_provider("deepseek-v4.1-flash") == "deepseek"


@pytest.mark.asyncio
async def test_check_auth_reads_the_catalog_and_never_raises():
    ok = _client(httpx.MockTransport(lambda r: httpx.Response(200, json=_models())))
    assert await ok.check_auth() is True

    empty = _client(httpx.MockTransport(lambda r: httpx.Response(200, json={"object": "list", "data": []})))
    assert await empty.check_auth() is False

    denied = _client(httpx.MockTransport(lambda r: httpx.Response(401, json={"error": {"type": "AuthError", "message": "bad"}})))
    assert await denied.check_auth() is False

    def broken(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route")

    assert await _client(httpx.MockTransport(broken)).check_auth() is False


@pytest.mark.asyncio
async def test_fetch_models_returns_only_dicts():
    client = _client(httpx.MockTransport(lambda r: httpx.Response(200, json={"object": "list", "data": [{"id": "a"}, "junk", 5]})))
    assert await client.fetch_models() == [{"id": "a"}]
    await client.aclose()


@pytest.mark.asyncio
async def test_fetch_models_on_a_non_dict_payload():
    client = _client(httpx.MockTransport(lambda r: httpx.Response(200, json=[1, 2])))
    assert await client.fetch_models() == []


@pytest.mark.asyncio
async def test_headers_carry_the_opencode_identification():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_models())

    client = _client(httpx.MockTransport(handler))
    await client.fetch_models()
    headers = seen[0].headers
    assert client.key == client_mod.PUBLIC_KEY
    assert "Authorization" not in headers
    assert headers["User-Agent"] == USER_AGENT
    assert headers["x-opencode-client"] == "danyapi"
    assert "x-opencode-session" not in headers
    await client.aclose()


@pytest.mark.asyncio
async def test_authorization_is_never_sent_under_the_hardcoded_public_key():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_models())

    client = _client(httpx.MockTransport(handler), key="")
    assert client.key == client_mod.PUBLIC_KEY
    await client.fetch_models()
    assert "Authorization" not in seen[0].headers


@pytest.mark.asyncio
async def test_user_agent_is_never_the_cloudflare_blocked_signature():
    assert "urllib" not in USER_AGENT.lower()
    assert USER_AGENT.startswith("opencode/")


@pytest.mark.asyncio
async def test_chat_sends_session_and_request_headers_and_bare_model():
    account = _Account()
    await account.client.chat({"messages": []}, "opencode/kimi-k3", session_id="ses_1", request_id="msg_1")
    request = account.requests[-1]
    body = json.loads(request.content)
    assert body["model"] == "kimi-k3"
    assert request.headers["x-opencode-session"] == "ses_1"
    assert request.headers["x-opencode-request"] == "msg_1"
    assert "Authorization" not in request.headers


@pytest.mark.asyncio
async def test_chat_streaming_sets_the_sse_accept_header():
    account = _Account()
    resp = await account.client.chat({"messages": [], "stream": True}, "kimi-k3")
    assert resp.request.headers["Accept"] == "text/event-stream"
    await resp.aclose()


@pytest.mark.asyncio
async def test_collect_non_stream_returns_the_openai_shape():
    from danyapi.opencode import api as opencode_api

    account = _Account()
    result = await opencode_api.collect_non_stream(account=account, messages=[_msg(role="user", content="hi")], model="space-bunny-free")
    assert result["object"] == "chat.completion"
    assert result["model"] == "space-bunny-free"
    assert result["choices"][0]["message"]["content"] == "ok"
    assert result["choices"][0]["finish_reason"] == "stop"
    assert result["usage"]["prompt_tokens"] == 157
    assert result["usage"]["total_tokens"] == 165
    assert result["usage"]["prompt_tokens_details"]["cached_tokens"] == 149
    assert json.loads(account.requests[-1].content)["model"] == "space-bunny-free"


@pytest.mark.asyncio
async def test_collect_non_stream_keeps_reasoning_and_tool_calls():
    from danyapi.opencode import api as opencode_api

    account = _Account()
    calls = [{"id": "call_1", "type": "function", "function": {"name": "f", "arguments": '{"a":1}'}}]
    account.client = _client(httpx.MockTransport(lambda r: httpx.Response(200, json=_completion(content="", reasoning="thinking", tool_calls=calls))))
    result = await opencode_api.collect_non_stream(account=account, messages=[_msg(role="user", content="hi")], model="m")
    message = result["choices"][0]["message"]
    assert message["reasoning_content"] == "thinking"
    assert message["tool_calls"][0]["function"] == {"name": "f", "arguments": '{"a":1}'}
    assert message["content"] == ""


@pytest.mark.asyncio
async def test_collect_non_stream_serialises_non_string_tool_arguments():
    from danyapi.opencode import api as opencode_api

    account = _Account()
    calls = [{"id": "call_1", "function": {"name": "f", "arguments": {"a": 1}}}]
    account.client = _client(httpx.MockTransport(lambda r: httpx.Response(200, json=_completion(tool_calls=calls))))
    result = await opencode_api.collect_non_stream(account=account, messages=[_msg(role="user", content="hi")], model="m")
    assert result["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] == '{"a": 1}'


@pytest.mark.asyncio
async def test_collect_non_stream_applies_stop_sequences():
    from danyapi.opencode import api as opencode_api

    account = _Account()
    account.client = _client(httpx.MockTransport(lambda r: httpx.Response(200, json=_completion(content="keep DROP rest"))))
    result = await opencode_api.collect_non_stream(account=account, messages=[_msg(role="user", content="hi")], model="m", stop=["DROP"])
    assert result["choices"][0]["message"]["content"] == "keep "


@pytest.mark.asyncio
async def test_collect_non_stream_rejects_a_body_without_choices():
    from danyapi.opencode import api as opencode_api

    account = _Account()
    account.client = _client(httpx.MockTransport(lambda r: httpx.Response(200, json={"choices": []})))
    with pytest.raises(HTTPException) as excinfo:
        await opencode_api.collect_non_stream(account=account, messages=[_msg(role="user", content="hi")], model="m")
    assert excinfo.value.status_code == 502


@pytest.mark.asyncio
async def test_collect_non_stream_rejects_a_non_dict_body():
    from danyapi.opencode import api as opencode_api

    account = _Account()
    account.client = _client(httpx.MockTransport(lambda r: httpx.Response(200, json=[1])))
    with pytest.raises(HTTPException):
        await opencode_api.collect_non_stream(account=account, messages=[_msg(role="user", content="hi")], model="m")


@pytest.mark.asyncio
async def test_collect_non_stream_rejects_malformed_json():
    from danyapi.opencode import api as opencode_api

    account = _Account()
    account.client = _client(httpx.MockTransport(lambda r: httpx.Response(200, content=b"not json")))
    with pytest.raises(HTTPException) as excinfo:
        await opencode_api.collect_non_stream(account=account, messages=[_msg(role="user", content="hi")], model="m")
    assert excinfo.value.status_code == 502


@pytest.mark.asyncio
async def test_collect_non_stream_marks_the_account_broken_on_auth_error():
    from danyapi.opencode import api as opencode_api

    account = _Account()
    account.client = _client(httpx.MockTransport(lambda r: httpx.Response(401, json={"error": {"type": "AuthError", "message": "Invalid API key."}})))
    with pytest.raises(HTTPException) as excinfo:
        await opencode_api.collect_non_stream(account=account, messages=[_msg(role="user", content="hi")], model="m")
    assert excinfo.value.status_code == 401
    assert "Invalid API key." in excinfo.value.detail
    assert account.broken is True


@pytest.mark.asyncio
async def test_collect_non_stream_does_not_break_the_account_on_a_model_error():
    from danyapi.opencode import api as opencode_api

    account = _Account()
    account.client = _client(httpx.MockTransport(lambda r: httpx.Response(401, json={"error": {"type": "ModelError", "message": "not supported for format"}})))
    with pytest.raises(HTTPException) as excinfo:
        await opencode_api.collect_non_stream(account=account, messages=[_msg(role="user", content="hi")], model="m")
    assert excinfo.value.status_code == 404
    assert account.broken is False


@pytest.mark.asyncio
async def test_collect_non_stream_maps_free_tier_and_region_errors():
    from danyapi.opencode import api as opencode_api

    for error_type, expected in (("FreeTierError", 403), ("RegionError", 403)):
        account = _Account()
        account.client = _client(httpx.MockTransport(lambda r, t=error_type: httpx.Response(403, json={"error": {"type": t, "message": "nope"}})))
        with pytest.raises(HTTPException) as excinfo:
            await opencode_api.collect_non_stream(account=account, messages=[_msg(role="user", content="hi")], model="m")
        assert excinfo.value.status_code == expected
        assert account.broken is False


@pytest.mark.asyncio
async def test_collect_non_stream_surfaces_a_non_json_error_body():
    from danyapi.opencode import api as opencode_api

    account = _Account()
    account.client = _client(httpx.MockTransport(lambda r: httpx.Response(400, content=b"<html>bad</html>")))
    with pytest.raises(HTTPException) as excinfo:
        await opencode_api.collect_non_stream(account=account, messages=[_msg(role="user", content="hi")], model="m")
    assert excinfo.value.status_code == 400


@pytest.mark.asyncio
async def test_stream_openai_emits_chunks_usage_and_done():
    from danyapi.opencode import api as opencode_api

    account = _Account()
    lines = [line async for line in opencode_api.stream_openai(account=account, messages=[_msg(role="user", content="hi")], model="m", include_usage=True)]
    text = "".join(lines)
    assert '"content":"ok"' in text or '"content": "ok"' in text
    assert '"total_tokens":6' in text or '"total_tokens": 6' in text
    assert text.endswith("data: [DONE]\n\n")


@pytest.mark.asyncio
async def test_stream_openai_reports_an_error_as_sse_not_an_exception():
    from danyapi.opencode import api as opencode_api

    account = _Account()
    account.client = _client(httpx.MockTransport(lambda r: httpx.Response(401, json={"error": {"type": "AuthError", "message": "Invalid API key."}})))
    lines = [line async for line in opencode_api.stream_openai(account=account, messages=[_msg(role="user", content="hi")], model="m")]
    text = "".join(lines)
    assert "Invalid API key." in text
    assert "data: [DONE]" in text


@pytest.mark.asyncio
async def test_stream_openai_reports_a_build_error_as_sse():
    from danyapi.opencode import api as opencode_api

    account = _Account()
    lines = [line async for line in opencode_api.stream_openai(account=account, messages=[_msg(role="system", content="no user turn at all")], model="m")]
    text = "".join(lines)
    assert "at least one user message" in text
    assert "data: [DONE]" in text


@pytest.mark.asyncio
async def test_stream_openai_asks_for_usage_only_when_wanted():
    from danyapi.opencode import api as opencode_api

    account = _Account()
    [line async for line in opencode_api.stream_openai(account=account, messages=[_msg(role="user", content="hi")], model="m")]
    assert "stream_options" not in json.loads(account.requests[-1].content)

    account2 = _Account()
    [line async for line in opencode_api.stream_openai(account=account2, messages=[_msg(role="user", content="hi")], model="m", include_usage=True)]
    assert json.loads(account2.requests[-1].content)["stream_options"] == {"include_usage": True}


def test_build_messages_folds_system_and_developer_and_maps_function_role():
    built = om.build_messages(
        [
            _msg(role="system", content="one"),
            _msg(role="developer", content="two"),
            _msg(role="user", content="hi"),
            _msg(role="assistant", content="call"),
            _msg(role="function", name="f", content="result"),
        ]
    )
    assert built[0] == {"role": "system", "content": "one\n\ntwo"}
    assert built[1] == {"role": "user", "content": "hi"}
    assert built[2]["role"] == "assistant"
    assert built[3]["role"] == "tool"
    assert built[3]["tool_call_id"] == "f"
    assert built[3]["content"] == "result"


def test_build_messages_drops_leading_non_user_turns():
    built = om.build_messages([_msg(role="assistant", content="hi"), _msg(role="user", content="q")])
    assert built == [{"role": "user", "content": "q"}]


def test_build_messages_requires_a_user_turn():
    with pytest.raises(HTTPException) as excinfo:
        om.build_messages([_msg(role="system", content="only system")])
    assert excinfo.value.status_code == 400


def test_build_messages_falls_back_an_unknown_role_to_user():
    message = _msg(role="user", content="hi")
    object.__setattr__(message, "role", "wizard")
    built = om.build_messages([message])
    assert built[0]["role"] == "user"


def test_build_messages_keeps_inline_image_parts():
    content = [{"type": "text", "text": "look"}, {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}}]
    built = om.build_messages([_msg(role="user", content=content)])
    assert built[0]["content"] == content


def test_build_messages_normalises_tool_calls():
    built = om.build_messages(
        [
            _msg(role="user", content="hi"),
            _msg(role="assistant", content="", tool_calls=[{"id": "c1", "function": {"name": "f", "arguments": {"a": 1}}}]),
            _msg(role="user", content="and now"),
        ]
    )
    call = built[1]["tool_calls"][0]
    assert call == {"id": "c1", "type": "function", "function": {"name": "f", "arguments": '{"a": 1}'}}


def test_build_messages_gives_an_untagged_tool_call_an_id():
    built = om.build_messages(
        [
            _msg(role="user", content="hi"),
            _msg(role="assistant", content="", tool_calls=[{"name": "f", "arguments": "{}"}]),
            _msg(role="user", content="more"),
        ]
    )
    assert built[1]["tool_calls"][0]["id"].startswith("call_")


def test_build_messages_drops_an_empty_system_turn():
    built = om.build_messages([_msg(role="system", content="   "), _msg(role="user", content="hi")])
    assert built == [{"role": "user", "content": "hi"}]


def test_build_messages_stringifies_non_string_content():
    built = om.build_messages([_msg(role="user", content=7)])
    assert built[0]["content"] == "7"
    built2 = om.build_messages([_msg(role="user", content=None)])
    assert built2[0]["content"] == ""


def test_request_body_omits_everything_unset():
    assert om.request_body([{"role": "user", "content": "hi"}]) == {"messages": [{"role": "user", "content": "hi"}]}


def test_request_body_passes_the_sampling_parameters_through():
    body = om.request_body(
        [{"role": "user", "content": "hi"}],
        None,
        None,
        0.4,
        0.8,
        256,
        stop=["END"],
        response_format={"type": "json_object"},
        seed=11,
        presence_penalty=0.1,
        frequency_penalty=0.2,
    )
    assert body["temperature"] == 0.4
    assert body["top_p"] == 0.8
    assert body["max_tokens"] == 256
    assert body["stop"] == ["END"]
    assert body["response_format"] == {"type": "json_object"}
    assert body["seed"] == 11
    assert body["presence_penalty"] == 0.1
    assert body["frequency_penalty"] == 0.2
    assert "max_completion_tokens" not in body


def test_request_body_prefers_max_completion_tokens():
    body = om.request_body([{"role": "user", "content": "hi"}], None, None, None, None, 100, 500)
    assert body["max_completion_tokens"] == 500
    assert "max_tokens" not in body


@pytest.mark.parametrize("budget", [0, -5, None])
def test_request_body_drops_a_useless_token_budget(budget):
    body = om.request_body([{"role": "user", "content": "hi"}], None, None, None, None, budget)
    assert "max_tokens" not in body
    assert "max_completion_tokens" not in body


def test_request_body_builds_tools_and_drops_duplicates():
    tools = [
        {"type": "function", "function": {"name": "f", "description": "d", "parameters": {"type": "object"}}},
        {"type": "function", "function": {"name": "f"}},
        {"name": "bare"},
        "junk",
    ]
    body = om.request_body([{"role": "user", "content": "hi"}], tools, "auto", None, None, None, parallel_tool_calls=True)
    assert [t["function"]["name"] for t in body["tools"]] == ["f", "bare"]
    assert body["tools"][0]["function"]["parameters"] == {"type": "object"}
    assert body["tools"][1]["function"]["parameters"] == om.EMPTY_PARAMETERS
    assert body["tools"][1]["function"].get("description") is None
    assert body["tool_choice"] == "auto"
    assert body["parallel_tool_calls"] is True


def test_request_body_drops_tool_choice_when_there_are_no_tools():
    body = om.request_body([{"role": "user", "content": "hi"}], [], "auto")
    assert "tools" not in body
    assert "tool_choice" not in body


def test_normalize_finish_reason_map():
    assert om.normalize_finish_reason("tool_calls") == "tool_calls"
    assert om.normalize_finish_reason("function_call") == "tool_calls"
    assert om.normalize_finish_reason("length") == "length"
    assert om.normalize_finish_reason("content_filter") == "content_filter"
    assert om.normalize_finish_reason("weird") == "stop"
    assert om.normalize_finish_reason(None) == "stop"


def test_normalize_usage_reads_the_nested_details():
    usage = om.normalize_usage(
        {
            "prompt_tokens": 10,
            "completion_tokens": 4,
            "total_tokens": 2,
            "prompt_tokens_details": {"cached_tokens": 6},
            "completion_tokens_details": {"reasoning_tokens": 3},
        }
    )
    assert usage["total_tokens"] == 14
    assert usage["prompt_tokens_details"]["cached_tokens"] == 6
    assert usage["completion_tokens_details"]["reasoning_tokens"] == 3
    assert om.normalize_usage(None)["total_tokens"] == 0
    assert om.normalize_usage({"prompt_tokens": "3", "completion_tokens": "2"})["total_tokens"] == 5


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"prompt_tokens": True}, 0),
        ({"prompt_tokens": -4}, 0),
        ({"prompt_tokens": "nonsense"}, 0),
        ({"prompt_tokens": 2.7}, 2),
        ({"prompt_tokens": {"nested": 1}}, 0),
    ],
)
def test_normalize_usage_ignores_junk_token_counts(payload, expected):
    assert om.normalize_usage(payload)["prompt_tokens"] == expected


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ("plain", "plain"),
        ([{"type": "text", "text": "a"}, {"type": "image_url", "image_url": {"url": "u"}}, "b", {"type": "text", "text": "c"}], "abc"),
        (None, ""),
        (True, ""),
        (["junk", 5], "junk"),
    ],
)
def test_text_of_reads_only_text_parts(content, expected):
    assert om._text_of(content) == expected


def test_system_text_is_trimmed():
    built = om.build_messages([_msg(role="system", content="  padded  "), _msg(role="user", content="hi")])
    assert built[0] == {"role": "system", "content": "padded"}


def test_message_dict_keeps_the_tool_call_id_and_name():
    built = om.build_messages(
        [
            _msg(role="user", content="hi"),
            _msg(role="tool", tool_call_id="c1", name="f", content="r"),
            _msg(role="user", content="next"),
        ]
    )
    assert built[1]["tool_call_id"] == "c1"
    assert built[1]["name"] == "f"


@pytest.mark.parametrize(
    "tools",
    [
        [{"type": "function", "function": {"name": ""}}],
        [{"function": "not a dict"}],
        [{"type": "function"}],
        ["junk", 5, None],
    ],
)
def test_tool_specs_drops_a_call_with_no_usable_name(tools):
    assert "tools" not in om.request_body([{"role": "user", "content": "hi"}], tools)


def test_tool_specs_rejects_a_non_dict_parameters_block():
    body = om.request_body([{"role": "user", "content": "hi"}], [{"name": "f", "parameters": "junk"}])
    assert body["tools"][0]["function"]["parameters"] == om.EMPTY_PARAMETERS


def test_response_format_and_stop_are_only_sent_when_meaningful():
    assert "response_format" not in om.request_body([{"role": "user", "content": "hi"}], response_format={})
    assert "response_format" not in om.request_body([{"role": "user", "content": "hi"}], response_format="junk")
    assert "stop" not in om.request_body([{"role": "user", "content": "hi"}], stop=None)


def test_arguments_serialisation_falls_back_on_unserialisable_values():
    assert om._as_arguments({"a": 1}) == '{"a": 1}'
    assert om._as_arguments(None) == "{}"
    assert om._as_arguments("") == "{}"
    assert om._as_arguments({"bad": object()}) == "{}"


@pytest.mark.asyncio
async def test_pool_round_robin_and_the_account_label():
    accounts = [OpenCodeAccount(i, _client(httpx.MockTransport(lambda r: httpx.Response(200, json=_models())))) for i in range(2)]
    pool = AccountPool(accounts, label="opencode")
    first, _sid = await _acquire_account(pool, None)
    second, _sid = await _acquire_account(pool, None)
    assert first is not second
    assert pool.stats()["accounts"] == 2
    assert accounts[0].label == "opencode-acct#0"


class _Stub(OpenCodeClient):
    """Real client over a mock transport, so the outbound body and headers are the genuine ones."""

    def __init__(self, payload: dict | None = None, error: tuple[int, dict] | None = None) -> None:
        super().__init__(key=KEY)
        self.requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            self.requests.append(request)
            if error is not None:
                status, body = error
                return httpx.Response(status, json=body)
            return httpx.Response(200, json=payload if payload is not None else _completion("привет"))

        self.http = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url=BASE_URL)

    @property
    def sent(self) -> list[dict]:
        return [json.loads(request.content) for request in self.requests]

    async def aclose(self) -> None:
        return None


@pytest.fixture
def _opencode_pool():
    saved = {attr: getattr(app.state, attr, None) for attr in ("opencode_pool", "opencode_models")}
    app.state.opencode_pool = None
    app.state.byok = False
    app.state.opencode_models = [
        {"id": "space-bunny-free", "name": "Space Bunny", "owned_by": "opencode", "model_type": "chat"},
        {"id": "qwen3.8-max", "name": "Qwen 3.8 Max", "owned_by": "opencode", "model_type": "chat"},
    ]
    yield
    for attr, value in saved.items():
        setattr(app.state, attr, value)


def _pool(client) -> AccountPool:
    pool = AccountPool([OpenCodeAccount(0, client, stable_id="zen1")], label="opencode")
    app.state.opencode_pool = pool
    return pool


def test_chat_completions_503_when_not_configured(_opencode_pool):
    resp = TestClient(app).post(
        "/v1/chat/completions",
        json={"model": "space-bunny-free", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 503
    assert "OPENCODE_KEYS" in resp.json()["error"]["message"]


def test_chat_completions_end_to_end(_opencode_pool):
    stub = _Stub()
    _pool(stub)
    resp = TestClient(app).post(
        "/v1/chat/completions",
        json={"model": "opencode/space-bunny-free", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 32},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "chat.completion"
    assert body["model"] == "opencode/space-bunny-free"
    assert body["choices"][0]["message"]["content"] == "привет"
    assert body["usage"]["prompt_tokens"] == 157
    assert body["usage"]["prompt_tokens_details"]["cached_tokens"] == 149
    assert stub.sent[0]["model"] == "space-bunny-free"
    assert stub.sent[0]["max_tokens"] == 32


def test_chat_completions_routes_the_bare_id_too(_opencode_pool):
    stub = _Stub()
    _pool(stub)
    resp = TestClient(app).post(
        "/v1/chat/completions",
        json={"model": "space-bunny-free", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 200
    assert stub.sent[0]["model"] == "space-bunny-free"


def test_chat_completions_stream_end_to_end(_opencode_pool):
    stub = _Stub()
    _pool(stub)
    resp = TestClient(app).post(
        "/v1/chat/completions",
        json={"model": "space-bunny-free", "messages": [{"role": "user", "content": "hi"}], "stream": True},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    assert "data: [DONE]" in resp.text


def test_chat_completions_rejects_file_attachments(_opencode_pool):
    _pool(_Stub())
    resp = TestClient(app).post(
        "/v1/chat/completions",
        json={
            "model": "space-bunny-free",
            "messages": [{"role": "user", "content": "hi"}],
            "files": [{"name": "a.txt", "content": "x"}],
        },
    )
    assert resp.status_code == 400
    assert "does not support file attachments" in resp.json()["error"]["message"]


def test_chat_completions_rejects_unsupported_params(_opencode_pool):
    _pool(_Stub())
    resp = TestClient(app).post(
        "/v1/chat/completions",
        json={"model": "space-bunny-free", "messages": [{"role": "user", "content": "hi"}], "n": 2},
    )
    assert resp.status_code == 400
    assert "does not support the n parameter" in resp.json()["error"]["message"]


def test_chat_completions_forwards_tools(_opencode_pool):
    stub = _Stub()
    _pool(stub)
    tools = [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}]
    resp = TestClient(app).post(
        "/v1/chat/completions",
        json={"model": "space-bunny-free", "messages": [{"role": "user", "content": "hi"}], "tools": tools, "tool_choice": "auto"},
    )
    assert resp.status_code == 200
    assert stub.sent[0]["tools"][0]["function"]["name"] == "lookup"
    assert stub.sent[0]["tool_choice"] == "auto"


def test_chat_completions_maps_a_model_error_to_404_without_breaking_the_account(_opencode_pool):
    stub = _Stub(error=(401, {"error": {"type": "ModelError", "message": "not supported for format openai-chat"}}))
    pool = _pool(stub)
    resp = TestClient(app).post(
        "/v1/chat/completions",
        json={"model": "space-bunny-free", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 404
    assert "not supported for format" in resp.json()["error"]["message"]
    assert pool.accounts[0].broken is False


def test_completions_endpoint_serves_the_provider(_opencode_pool):
    _pool(_Stub())
    resp = TestClient(app).post("/v1/completions", json={"model": "space-bunny-free", "prompt": "hi"})
    assert resp.status_code == 200
    assert resp.json()["choices"][0]["text"] == "привет"


def test_anthropic_messages_serves_the_provider(_opencode_pool):
    _pool(_Stub())
    resp = TestClient(app).post(
        "/v1/messages",
        json={"model": "opencode/space-bunny-free", "max_tokens": 64, "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 200
    assert resp.json()["content"][0]["text"] == "привет"


def test_responses_endpoint_serves_the_provider(_opencode_pool):
    _pool(_Stub())
    resp = TestClient(app).post("/v1/responses", json={"model": "opencode/space-bunny-free", "input": "hi"})
    assert resp.status_code == 200
    assert resp.json()["object"] == "response"


def test_models_endpoint_lists_the_provider_catalog(_opencode_pool):
    data = TestClient(app).get("/v1/models").json()["data"]
    owners = {m["id"]: m["owned_by"] for m in data}
    assert owners["space-bunny-free"] == "opencode"
    assert owners["qwen3.8-max"] == "opencode"


@pytest.fixture
def _no_backoff(monkeypatch):
    from danyapi.opencode import api as opencode_api
    from danyapi.opencode import client as client_mod

    for module in (client_mod, opencode_api):
        monkeypatch.setattr(module, "_retry_delay", lambda attempt: 0.0)
    return client_mod


@pytest.mark.asyncio
async def test_a_retryable_status_is_retried_then_succeeds(monkeypatch, _no_backoff):
    monkeypatch.setattr(_no_backoff, "MAX_RETRIES", 2)
    attempts: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(request)
        if len(attempts) < 3:
            return httpx.Response(429, headers={"Retry-After": "0"}, json={"error": {"type": "RateLimitError", "message": "slow"}})
        return httpx.Response(200, json=_models())

    client = _client(httpx.MockTransport(handler))
    assert await client.fetch_models() != []
    assert len(attempts) == 3


@pytest.mark.asyncio
async def test_a_retryable_status_gives_up_and_raises(monkeypatch, _no_backoff):
    monkeypatch.setattr(_no_backoff, "MAX_RETRIES", 1)
    attempts: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(request)
        return httpx.Response(503, json={"error": {"type": "SomeError", "message": "down"}})

    client = _client(httpx.MockTransport(handler))
    with pytest.raises(OpenCodeError) as excinfo:
        await client.fetch_models()
    assert excinfo.value.code == 503
    assert len(attempts) == 2


@pytest.mark.asyncio
async def test_a_non_json_error_body_becomes_a_client_error():
    client = _client(httpx.MockTransport(lambda r: httpx.Response(500, content=b"<html>")))
    with pytest.raises(OpenCodeError) as excinfo:
        await client.fetch_models()
    assert "non-JSON" in excinfo.value.message


@pytest.mark.parametrize(
    ("header", "expected"),
    [("5", 5.0), ("0", 7.0), ("-3", 7.0), ("not-a-number", 7.0), (None, 7.0), ("999", 8.0)],
)
def test_retry_after_header_parsing(monkeypatch, _no_backoff, header, expected):
    monkeypatch.setattr(_no_backoff, "_retry_delay", lambda attempt: 7.0)
    headers = {"Retry-After": header} if header is not None else {}
    assert _no_backoff._retry_after_seconds(httpx.Response(429, headers=headers)) == expected


@pytest.mark.asyncio
async def test_stream_openai_applies_stop_sequences_and_drops_the_emptied_delta():
    from danyapi.opencode import api as opencode_api

    payload = (
        'data: {"choices":[{"index":0,"delta":{"content":"DROP rest"},"finish_reason":null}]}\n\n'
        'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
        "data: [DONE]\n\n"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=payload.encode())

    account = _Account()
    account.client = _client(httpx.MockTransport(handler))
    lines = [line async for line in opencode_api.stream_openai(account=account, messages=[_msg(role="user", content="hi")], model="m", stop=["DROP"])]
    text = "".join(lines)
    assert "DROP" not in text
    assert '"finish_reason": "stop"' in text


@pytest.mark.asyncio
async def test_stream_openai_emits_an_empty_first_chunk_when_upstream_said_nothing():
    from danyapi.opencode import api as opencode_api

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=b"data: [DONE]\n\n")

    account = _Account()
    account.client = _client(httpx.MockTransport(handler))
    lines = [line async for line in opencode_api.stream_openai(account=account, messages=[_msg(role="user", content="hi")], model="m")]
    text = "".join(lines)
    assert '"content": ""' in text
    assert lines[-1] == "data: [DONE]\n\n"


@pytest.mark.asyncio
async def test_stream_openai_finishes_the_stream_on_an_unexpected_mid_stream_error():
    from danyapi.opencode import api as opencode_api

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=b'data: {"choices":[{"index":0,"delta":{"content":"partial"}}]}\n\ndata: {"choices": "not a list"}\n\n',
        )

    account = _Account()
    account.client = _client(httpx.MockTransport(handler))
    lines = [line async for line in opencode_api.stream_openai(account=account, messages=[_msg(role="user", content="hi")], model="m")]
    assert "partial" in "".join(lines)
    assert lines[-1] == "data: [DONE]\n\n"


@pytest.mark.asyncio
async def test_the_http_client_is_built_lazily_against_the_zen_base_url():
    client = OpenCodeClient(key=KEY)
    built = client.http
    assert str(built.base_url) == f"{BASE_URL}/"
    assert built.headers["user-agent"] == USER_AGENT
    assert built.headers["accept"] == "application/json"
    await client.aclose()
    assert client.http is not built


def test_a_missing_client_name_falls_back_to_the_default():
    assert OpenCodeClient(client_id="").client_id == "danyapi"


@pytest.mark.asyncio
async def test_collect_non_stream_retries_a_transport_error_then_fails(monkeypatch, _no_backoff):
    from danyapi.opencode import api as opencode_api

    monkeypatch.setattr(opencode_api, "MAX_RETRIES", 1)

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route to host")

    account = _Account()
    account.client = _client(httpx.MockTransport(handler))
    with pytest.raises(HTTPException) as excinfo:
        await opencode_api.collect_non_stream(account=account, messages=[_msg(role="user", content="hi")], model="m")
    assert excinfo.value.status_code == 502
    assert "transport error" in excinfo.value.detail


@pytest.mark.asyncio
async def test_stream_openai_turns_a_mid_stream_transport_error_into_an_finished_stream(monkeypatch, _no_backoff):
    from danyapi.opencode import api as opencode_api

    monkeypatch.setattr(opencode_api, "MAX_RETRIES", 0)
    truncated = b'data: {"choices":[{"index":0,"delta":{"content":"partial"},"finish_reason":null}]}\n\ndata: [DO'

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=truncated)

    account = _Account()
    account.client = _client(httpx.MockTransport(handler))
    lines = [line async for line in opencode_api.stream_openai(account=account, messages=[_msg(role="user", content="hi")], model="m")]
    text = "".join(lines)
    assert "partial" in text
    assert "data: [DONE]" in text


@pytest.mark.asyncio
async def test_stream_openai_forwards_tool_call_deltas():
    from danyapi.opencode import api as opencode_api

    payload = (
        'data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"c1","function":{"name":"f","arguments":"{\\"a\\""}}]}}]}\n\n'
        'data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":":1}"}}]},"finish_reason":"tool_calls"}]}\n\n'
        "data: [DONE]\n\n"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=payload.encode())

    account = _Account()
    account.client = _client(httpx.MockTransport(handler))
    lines = [line async for line in opencode_api.stream_openai(account=account, messages=[_msg(role="user", content="hi")], model="m")]
    text = "".join(lines)
    assert '"tool_calls"' in text
    assert '"tool_calls"' in text
    assert "tool_calls" in text


def test_delta_from_event_ignores_a_chunk_without_choices():
    from danyapi.opencode import api as opencode_api

    assert opencode_api._delta_from_event({"choices": []}) == ({}, None)
    assert opencode_api._delta_from_event({}) == ({}, None)
    assert opencode_api._delta_from_event({"choices": [{}]}) == ({}, None)


def test_delta_from_event_falls_back_to_a_message_shape():
    from danyapi.opencode import api as opencode_api

    delta, finish = opencode_api._delta_from_event({"choices": [{"message": {"content": "hi", "reasoning_content": "why"}, "finish_reason": 7}]})
    assert delta == {"content": "hi", "reasoning_content": "why"}
    assert finish is None


def test_delta_from_event_drops_non_dict_tool_call_entries():
    from danyapi.opencode import api as opencode_api

    delta, _finish = opencode_api._delta_from_event({"choices": [{"delta": {"tool_calls": ["junk"]}}]})
    assert delta == {}


def test_delta_from_event_defaults_a_missing_tool_call_index():
    from danyapi.opencode import api as opencode_api

    delta, _finish = opencode_api._delta_from_event({"choices": [{"delta": {"tool_calls": [{"id": "c1"}]}}]})
    assert delta["tool_calls"][0]["index"] == 0
    assert delta["tool_calls"][0]["function"]["arguments"] == ""


async def test_a_non_dict_error_body_is_reported_verbatim():
    from danyapi.opencode.api import _raise_upstream

    resp = httpx.Response(502, content=b"upstream exploded", request=httpx.Request("POST", BASE_URL))
    account = _Account()
    with pytest.raises(HTTPException) as excinfo:
        await _raise_upstream(account, resp, None)
    assert excinfo.value.status_code == 502
    assert "upstream exploded" in excinfo.value.detail


async def test_an_unconsumed_error_stream_is_read_before_it_is_parsed():
    from danyapi.opencode import api as opencode_api

    resp = httpx.Response(
        402,
        headers={"content-type": "application/json"},
        stream=httpx.ByteStream(b'{"error":{"type":"FreeTierError","message":"free tier spent"}}'),
        request=httpx.Request("POST", BASE_URL),
    )
    assert resp.is_stream_consumed is False
    account = _Account()
    with pytest.raises(HTTPException) as excinfo:
        await opencode_api._raise_upstream(account, resp, await opencode_api._safe_json(resp))
    assert excinfo.value.status_code == 403
    assert "free tier spent" in excinfo.value.detail
    assert "content not read" not in excinfo.value.detail
    assert account.broken is False


async def test_an_unreadable_error_stream_keeps_the_upstream_reason():
    from danyapi.opencode import api as opencode_api

    resp = httpx.Response(401, stream=httpx.ByteStream(b'{"error":{"type":"AuthError"}}'), request=httpx.Request("POST", BASE_URL))

    async def _boom() -> None:
        raise httpx.ReadError("socket closed")

    resp.aread = _boom  # type: ignore[method-assign]
    account = _Account()
    with pytest.raises(HTTPException) as excinfo:
        await opencode_api._raise_upstream(account, resp, await opencode_api._safe_json(resp))
    assert excinfo.value.status_code == 401
    assert excinfo.value.detail == "OpenCode Zen error: upstream returned 401"
    assert "content not read" not in excinfo.value.detail
    assert account.broken is True


async def test_a_stream_error_marks_the_account_broken_on_an_unconsumed_error_stream():
    from danyapi.opencode import api as opencode_api

    body = b'{"error":{"type":"AuthError","message":"Invalid API key."}}'
    account = _Account()
    account.client = _client(httpx.MockTransport(lambda r: httpx.Response(401, headers={"content-type": "application/json"}, stream=httpx.ByteStream(body))))

    lines = [line async for line in opencode_api.stream_openai(account=account, messages=[_msg(role="user", content="hi")], model="m")]

    assert "Invalid API key." in "".join(lines)
    assert account.broken is True


async def test_stream_openai_reports_the_real_reason_from_an_unconsumed_error_stream():
    from danyapi.opencode import api as opencode_api

    body = b'{"error":{"type":"FreeTierError","message":"free tier spent"}}'
    account = _Account()
    account.client = _client(httpx.MockTransport(lambda r: httpx.Response(402, headers={"content-type": "application/json"}, stream=httpx.ByteStream(body))))

    lines = [line async for line in opencode_api.stream_openai(account=account, messages=[_msg(role="user", content="hi")], model="m")]
    text = "".join(lines)

    assert "free tier spent" in text
    assert "content not read" not in text
    assert text.endswith("data: [DONE]\n\n")
    assert account.broken is False


def test_health_reports_the_provider(_opencode_pool):
    from danyapi.config import settings as live_settings

    saved = live_settings.admin_token
    live_settings.admin_token = "tok"
    try:
        _pool(_Stub())
        body = TestClient(app).get("/health", headers={"x-api-key": "tok"}).json()
    finally:
        live_settings.admin_token = saved
    assert body["opencode"] is True
    assert body["opencode_stats"]["accounts"] == 1


def _no_key_request():
    from starlette.requests import Request

    scope = {"type": "http", "method": "POST", "path": "/v1/chat/completions", "headers": []}

    async def receive() -> dict:
        return {"type": "http.request", "body": b"", "more_body": False}

    return Request(scope, receive)


@pytest.fixture
def _fresh_byok():
    from danyapi.api.state import BYOK_PROVIDERS as providers

    saved = {attr: getattr(app.state, attr, None) for attr in ("byok", "byok_pools", "byok_auth", "byok_stores", "byok_locks")}
    app.state.byok = True
    app.state.byok_pools = {provider: {} for provider in providers}
    app.state.byok_locks = {provider: asyncio.Lock() for provider in providers}
    app.state.byok_auth = {provider: {} for provider in providers}
    app.state.byok_stores = {provider: {} for provider in providers}
    yield
    for attr, value in saved.items():
        setattr(app.state, attr, value)


@pytest.mark.asyncio
async def test_byok_falls_back_to_a_keyless_account_when_no_key_is_sent(monkeypatch, _fresh_byok):
    from danyapi.api import byok as byok_mod
    from danyapi.opencode import client as client_mod

    async def _reachable(self) -> bool:
        return True

    async def _no_models(self) -> list[dict]:
        return []

    monkeypatch.setattr(client_mod.OpenCodeClient, "check_auth", _reachable)
    monkeypatch.setattr(client_mod.OpenCodeClient, "fetch_models", _no_models)
    pool = await byok_mod._byok_pool_for("opencode", _no_key_request())
    try:
        assert len(pool.accounts) == 1
        assert pool.accounts[0].client.key == client_mod.PUBLIC_KEY
        assert pool.label == "opencode"
    finally:
        await byok_mod._close_pool(pool)


@pytest.mark.asyncio
async def test_byok_builds_one_account_per_comma_separated_key(monkeypatch, _fresh_byok):
    from danyapi.api import byok as byok_mod
    from danyapi.opencode import client as client_mod

    async def _reachable(self) -> bool:
        return True

    async def _no_models(self) -> list[dict]:
        return []

    monkeypatch.setattr(client_mod.OpenCodeClient, "check_auth", _reachable)
    monkeypatch.setattr(client_mod.OpenCodeClient, "fetch_models", _no_models)
    request = _keyed_request("sk-a,sk-b")
    pool = await byok_mod._byok_pool_for("opencode", request)
    try:
        assert [a.client.key for a in pool.accounts] == [client_mod.PUBLIC_KEY]
        assert len(pool.accounts) == 1
    finally:
        await byok_mod._close_pool(pool)


@pytest.mark.asyncio
async def test_byok_rejects_every_key_when_the_catalog_is_unreachable(monkeypatch, _fresh_byok):
    from danyapi.api import byok as byok_mod
    from danyapi.opencode import client as client_mod

    async def _unreachable(self) -> bool:
        return False

    monkeypatch.setattr(client_mod.OpenCodeClient, "check_auth", _unreachable)
    with pytest.raises(HTTPException) as excinfo:
        await byok_mod._byok_pool_for("opencode", _keyed_request("sk-a"))
    assert excinfo.value.status_code == 401


@pytest.mark.asyncio
async def test_byok_still_rejects_a_keyless_request_for_a_keyed_provider():
    from danyapi.api import byok as byok_mod

    with pytest.raises(HTTPException) as excinfo:
        await byok_mod._byok_pool_for("gigachat", _no_key_request())
    assert excinfo.value.status_code == 401


def _keyed_request(value: str):
    from starlette.requests import Request

    headers = [(b"authorization", f"Bearer {value}".encode())]
    scope = {"type": "http", "method": "POST", "path": "/v1/chat/completions", "headers": headers}

    async def receive() -> dict:
        return {"type": "http.request", "body": b"", "more_body": False}

    return Request(scope, receive)
