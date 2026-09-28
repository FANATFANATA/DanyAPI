import base64
import json
from collections.abc import Callable

import httpx
import pytest

from danyapi.gigachat import messages as gm
from danyapi.gigachat.client import DEFAULT_SCOPE, GigaChatClient, GigaChatError, is_authorization_key

KEY = base64.b64encode(b"client-id:client-secret").decode()


def _client(monkeypatch, transport: httpx.MockTransport) -> GigaChatClient:
    client = GigaChatClient(key=KEY)
    client.http = httpx.AsyncClient(transport=transport, base_url="https://api.giga.chat/v1")
    return client


def _token_response(request: httpx.Request, payload: dict | None = None, status: int = 200) -> httpx.Response:
    assert request.url.path.endswith("/api/v2/oauth")
    assert request.headers["RqUID"]
    assert request.headers["Authorization"] == f"Basic {KEY}"
    assert request.content == b"scope=GIGACHAT_API_PERS"
    body = payload if payload is not None else {"access_token": "tok", "expires_at": 4102444800}
    return httpx.Response(status, json=body)


def _router(token_calls: list[int], models_handler) -> Callable[[httpx.Request], httpx.Response]:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/api/v2/oauth"):
            token_calls.append(1)
            return _token_response(request, {"access_token": f"tok{len(token_calls)}", "expires_at": 4102444800})
        return models_handler(request)

    return handler


def test_is_authorization_key():
    assert is_authorization_key(KEY) is True
    assert is_authorization_key("not base64!!") is False
    assert is_authorization_key("") is False


def test_default_scope_and_bad_scope_fallback():
    assert GigaChatClient(key=KEY).scope == DEFAULT_SCOPE
    assert GigaChatClient(key=KEY, scope="nonsense").scope == DEFAULT_SCOPE


@pytest.mark.asyncio
async def test_token_is_cached_between_calls():
    calls: list[int] = []
    client = _client(None, httpx.MockTransport(_router(calls, lambda r: httpx.Response(404, json={}))))
    first = await client.access_token()
    second = await client.access_token()
    assert first == second == "tok1"
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_check_auth_true_and_false():
    ok = _client(None, httpx.MockTransport(_router([], lambda r: httpx.Response(200, json={"data": [{"id": "GigaChat"}]}))))
    assert await ok.check_auth() is True

    def models_unauthorized(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"status": 401, "message": "Unauthorized"})

    bad = _client(None, httpx.MockTransport(_router([], models_unauthorized)))
    assert await bad.check_auth() is False


@pytest.mark.asyncio
async def test_expired_token_is_refreshed_once_on_401():
    token_calls: list[int] = []
    models_calls: list[int] = []

    def models_handler(request: httpx.Request) -> httpx.Response:
        models_calls.append(1)
        if len(models_calls) == 1:
            return httpx.Response(401, json={"status": 401, "message": "Unauthorized"})
        return httpx.Response(200, json={"object": "list", "data": [{"id": "GigaChat"}]})

    client = _client(None, httpx.MockTransport(_router(token_calls, models_handler)))
    models = await client.fetch_models()
    assert models == [{"id": "GigaChat"}]
    assert len(token_calls) == 2


@pytest.mark.asyncio
async def test_upload_file_requires_id():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/api/v2/oauth"):
            return _token_response(request)
        return httpx.Response(200, json={"id": "file-uuid"})

    client = _client(None, httpx.MockTransport(handler))
    assert await client.upload_file("a.png", b"data", "image/png") == "file-uuid"


@pytest.mark.asyncio
async def test_upload_file_without_id_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/api/v2/oauth"):
            return _token_response(request)
        return httpx.Response(200, json={"object": "file"})

    client = _client(None, httpx.MockTransport(handler))
    with pytest.raises(GigaChatError):
        await client.upload_file("a.png", b"data", "image/png")


@pytest.mark.asyncio
async def test_chat_injects_model_and_stream_flag():
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/api/v2/oauth"):
            return _token_response(request)
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": []})

    client = _client(None, httpx.MockTransport(handler))
    resp = await client.chat({"messages": [{"role": "user", "content": "hi"}]}, "GigaChat-2-Max")
    await resp.aclose()
    assert seen[0]["model"] == "GigaChat-2-Max"
    assert seen[0]["stream"] is False

    resp2 = await client.chat({"messages": [], "stream": True}, "GigaChat")
    await resp2.aclose()
    assert seen[1]["stream"] is True


@pytest.mark.asyncio
async def test_expires_at_accepts_seconds_and_milliseconds():
    from danyapi.gigachat.client import _expires_at_seconds

    assert _expires_at_seconds(4102444800) == 4102444800.0
    assert _expires_at_seconds(4102444800000) == 4102444800.0
    assert _expires_at_seconds("4102444800") == 4102444800.0
    assert _expires_at_seconds(None) == 0.0
    assert _expires_at_seconds("nonsense") == 0.0


def test_normalize_finish_reason_maps_function_call():
    assert gm.normalize_finish_reason("function_call") == "tool_calls"
    assert gm.normalize_finish_reason("blacklist") == "content_filter"
    assert gm.normalize_finish_reason("length") == "length"
    assert gm.normalize_finish_reason(None) == "stop"
    assert gm.normalize_finish_reason("weird") == "stop"


def test_normalize_usage_handles_cached_and_missing():
    usage = gm.normalize_usage({"prompt_tokens": 10, "completion_tokens": 5, "precached_prompt_tokens": 3})
    assert usage["prompt_tokens"] == 10
    assert usage["completion_tokens"] == 5
    assert usage["total_tokens"] == 15
    assert usage["prompt_tokens_details"]["cached_tokens"] == 3

    empty = gm.normalize_usage(None)
    assert empty["total_tokens"] == 0


def test_request_body_omits_unset_and_keeps_functions():
    body = gm.request_body([{"role": "user", "content": "hi"}], [], None, None, None, None, None)
    assert body == {"messages": [{"role": "user", "content": "hi"}]}

    with_functions = gm.request_body(
        [{"role": "user", "content": "hi"}],
        [{"name": "f"}],
        "auto",
        0.5,
        0.9,
        100,
        None,
    )
    assert with_functions["functions"] == [{"name": "f"}]
    assert with_functions["function_call"] == "auto"
    assert with_functions["max_tokens"] == 100
