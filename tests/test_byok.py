import asyncio
import json
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient
from starlette.requests import Request

import danyapi.api.openai as openai_mod
from danyapi.api.openai import app, settings


@pytest.fixture(autouse=True)
async def _reset_byok_state():
    app.state.byok = False
    app.state.byok_pools = {"deepseek": {}, "qwen": {}}
    app.state.byok_locks = {"deepseek": asyncio.Lock(), "qwen": asyncio.Lock()}
    saved_models = getattr(app.state, "qwen_models", None)
    yield
    pools = getattr(app.state, "byok_pools", {})
    for cache in pools.values():
        for key in list(cache.keys()):
            pool = cache.pop(key)
            await openai_mod._close_pool(pool)
    app.state.byok = False
    app.state.byok_pools = {"deepseek": {}, "qwen": {}}
    app.state.byok_locks = {"deepseek": asyncio.Lock(), "qwen": asyncio.Lock()}
    app.state.qwen_models = saved_models


def _make_request(headers: dict[str, str] | None = None, body: bytes = b"") -> Request:
    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    header_bytes = [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()]
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/v1/chat/completions",
        "query_string": b"",
        "headers": header_bytes,
        "client": ("127.0.0.1", 5555),
        "server": ("testserver", 80),
        "scheme": "http",
    }
    return Request(scope, receive=receive)


async def test_extract_api_key_bearer():
    request = _make_request(headers={"Authorization": "Bearer tok123"})
    assert await openai_mod._extract_request_api_key(request) == "tok123"


async def test_extract_api_key_x_api_key():
    request = _make_request(headers={"x-api-key": "tok456"})
    assert await openai_mod._extract_request_api_key(request) == "tok456"


async def test_extract_api_key_body():
    body = json.dumps({"api_key": "tok789", "model": "deepseek-v4.1-flash"}).encode()
    request = _make_request(headers={"Content-Type": "application/json"}, body=body)
    assert await openai_mod._extract_request_api_key(request) == "tok789"


async def test_extract_api_key_bearer_precedence():
    headers = {"Authorization": "Bearer aaa", "x-api-key": "bbb", "Content-Type": "application/json"}
    body = json.dumps({"api_key": "ccc"}).encode()
    request = _make_request(headers=headers, body=body)
    assert await openai_mod._extract_request_api_key(request) == "aaa"


async def test_extract_api_key_missing():
    assert await openai_mod._extract_request_api_key(_make_request(headers={})) is None


async def test_byok_pool_for_missing_key_401():
    with pytest.raises(openai_mod.HTTPException) as excinfo:
        await openai_mod._byok_pool_for("deepseek", _make_request(headers={}))
    assert excinfo.value.status_code == 401


async def test_byok_pool_deepseek_builds_and_caches(monkeypatch):
    monkeypatch.setattr(settings, "cache_enabled", False)
    calls = 0

    async def fake_check_auth(self):
        nonlocal calls
        calls += 1
        return True

    monkeypatch.setattr(openai_mod.DeepSeekClient, "check_auth", fake_check_auth)
    request = _make_request(headers={"Authorization": "Bearer tok-a"})
    pool = await openai_mod._byok_pool_for("deepseek", request)
    assert len(pool.accounts) == 1
    assert pool.healthy
    cached = await openai_mod._byok_pool_for("deepseek", request)
    assert cached is pool
    assert calls == 1


async def test_byok_pool_deepseek_invalid_401(monkeypatch):
    monkeypatch.setattr(settings, "cache_enabled", False)
    monkeypatch.setattr(openai_mod.DeepSeekClient, "check_auth", AsyncMock(return_value=False))
    request = _make_request(headers={"Authorization": "Bearer bad-token"})
    with pytest.raises(openai_mod.HTTPException) as excinfo:
        await openai_mod._byok_pool_for("deepseek", request)
    assert excinfo.value.status_code == 401
    assert app.state.byok_pools["deepseek"] == {}


async def test_byok_pool_comma_separated_keys(monkeypatch):
    monkeypatch.setattr(settings, "cache_enabled", False)
    monkeypatch.setattr(openai_mod.DeepSeekClient, "check_auth", AsyncMock(return_value=True))
    request = _make_request(headers={"Authorization": "Bearer tok1,tok2"})
    pool = await openai_mod._byok_pool_for("deepseek", request)
    assert len(pool.accounts) == 2
    assert all(account.healthy if hasattr(account, "healthy") else not account.broken for account in pool.accounts)


async def test_byok_pool_partial_invalid_keys(monkeypatch):
    monkeypatch.setattr(settings, "cache_enabled", False)
    monkeypatch.setattr(openai_mod.DeepSeekClient, "check_auth", AsyncMock(side_effect=[True, False]))
    request = _make_request(headers={"Authorization": "Bearer ok,broken"})
    pool = await openai_mod._byok_pool_for("deepseek", request)
    assert len(pool.accounts) == 1


async def test_close_pool_flushes_pool_stores(monkeypatch):
    monkeypatch.setattr(settings, "cache_enabled", False)
    pool = openai_mod.AccountPool([])
    flushed = {"count": 0}

    def fake_flush():
        flushed["count"] += 1

    pool.flush = fake_flush
    await openai_mod._close_pool(pool)
    assert flushed["count"] == 1


async def test_close_pool_without_flush_attribute():
    class _Bare:
        def __init__(self):
            self.accounts = []

    await openai_mod._close_pool(_Bare())


async def test_byok_pool_qwen_fetches_models(monkeypatch):
    monkeypatch.setattr(settings, "cache_enabled", False)
    app.state.qwen_models = []
    monkeypatch.setattr(openai_mod.QwenClient, "check_auth", AsyncMock(return_value=True))
    monkeypatch.setattr(
        openai_mod.QwenClient,
        "fetch_models",
        AsyncMock(return_value=[{"id": "q1", "info": {"meta": {"chat_type": ["t2t"]}}}]),
    )
    request = _make_request(headers={"Authorization": "Bearer qwen-token"})
    pool = await openai_mod._byok_pool_for("qwen", request)
    assert len(pool.accounts) == 1
    assert app.state.qwen_models


async def test_byok_pool_qwen_invalid_401(monkeypatch):
    monkeypatch.setattr(settings, "cache_enabled", False)
    monkeypatch.setattr(openai_mod.QwenClient, "check_auth", AsyncMock(return_value=False))
    request = _make_request(headers={"Authorization": "Bearer qwen-bad"})
    with pytest.raises(openai_mod.HTTPException) as excinfo:
        await openai_mod._byok_pool_for("qwen", request)
    assert excinfo.value.status_code == 401


def test_chat_completions_byok_401_without_key():
    app.state.byok = True
    client = TestClient(app)
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "deepseek-v4.1-flash",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": False,
        },
    )
    assert response.status_code == 401


def test_chat_completions_byok_invalid_key_401(monkeypatch):
    app.state.byok = True
    monkeypatch.setattr(settings, "cache_enabled", False)
    monkeypatch.setattr(openai_mod.DeepSeekClient, "check_auth", AsyncMock(return_value=False))
    client = TestClient(app)
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "deepseek-v4.1-flash",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": False,
        },
        headers={"Authorization": "Bearer bad-token"},
    )
    assert response.status_code == 401
    assert app.state.byok_pools["deepseek"] == {}


def test_chat_completions_unauthenticated_route_when_byok_off():
    app.state.byok = False
    client = TestClient(app)
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "deepseek-v4.1-flash",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": False,
        },
    )
    assert response.status_code == 503


def test_create_response_byok_401_without_key():
    app.state.byok = True
    client = TestClient(app)
    response = client.post(
        "/v1/responses",
        json={
            "model": "deepseek-v4.1-flash",
            "input": "hello",
            "stream": False,
        },
    )
    assert response.status_code == 401


def test_image_generations_byok_401_without_key():
    app.state.byok = True
    client = TestClient(app)
    response = client.post(
        "/v1/images/generations",
        json={"model": "qwen-image-gen", "prompt": "dog"},
    )
    assert response.status_code == 401


def test_health_reports_byok_mode():
    app.state.byok = True
    client = TestClient(app)
    payload = client.get("/health").json()
    assert payload["byok"] is True
    assert payload["byok_pools"]["deepseek"] == 0
    assert payload["byok_pools"]["qwen"] == 0


def test_list_models_byok_without_env_tokens_lists_qwen_defaults():
    app.state.byok = True
    app.state.qwen_models = []
    client = TestClient(app)
    data = client.get("/v1/models").json()
    client.close()
    ids = [m["id"] for m in data["data"]]
    assert "deepseek-v4.1-flash" in ids
    for model in openai_mod.QWEN_DEFAULT_MODELS:
        assert model["id"] in ids


def test_health_no_byok_key_when_disabled():
    app.state.byok = False
    client = TestClient(app)
    payload = client.get("/health").json()
    assert "byok" not in payload
