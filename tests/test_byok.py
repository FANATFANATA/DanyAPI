import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from starlette.requests import Request

import danyapi.api.byok as byok_mod
import danyapi.api.images as images_mod
import danyapi.api.openai as openai_mod
from danyapi.api.openai import app, settings


@pytest.fixture(autouse=True)
async def _reset_byok_state():
    app.state.byok = False
    app.state.byok_pools = {provider: {} for provider in openai_mod.BYOK_PROVIDERS}
    app.state.byok_locks = {provider: asyncio.Lock() for provider in openai_mod.BYOK_PROVIDERS}
    saved_models = getattr(app.state, "qwen_models", None)
    app.state.byok_alice_pool = None
    yield
    pools = getattr(app.state, "byok_pools", {})
    for cache in pools.values():
        for key in list(cache.keys()):
            pool = cache.pop(key)
            await openai_mod._close_pool(pool)
    app.state.byok = False
    app.state.byok_pools = {provider: {} for provider in openai_mod.BYOK_PROVIDERS}
    app.state.byok_locks = {provider: asyncio.Lock() for provider in openai_mod.BYOK_PROVIDERS}
    app.state.qwen_models = saved_models
    app.state.byok_alice_pool = None


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


async def test_extract_api_key_multipart_form_field():
    boundary = "----danyapitest"
    parts = [
        f"--{boundary}\r\n",
        'Content-Disposition: form-data; name="api_key"\r\n\r\n',
        "  form-token \r\n",
        f"--{boundary}--\r\n",
    ]
    body = "".join(parts).encode()
    request = _make_request(headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}, body=body)
    assert await openai_mod._extract_request_api_key(request) == "form-token"


async def test_extract_api_key_multipart_without_field():
    boundary = "----danyapitest"
    body = f'--{boundary}\r\nContent-Disposition: form-data; name="model"\r\n\r\nqwen\r\n--{boundary}--\r\n'.encode()
    request = _make_request(headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}, body=body)
    assert await openai_mod._extract_request_api_key(request) is None


async def test_extract_api_key_multipart_malformed_body():
    request = _make_request(headers={"Content-Type": "multipart/form-data; boundary=zzz"}, body=b"garbage")
    with pytest.raises(openai_mod.HTTPException) as excinfo:
        await openai_mod._extract_request_api_key(request)
    assert excinfo.value.status_code == 400
    assert "malformed multipart request body" in excinfo.value.detail


async def test_extract_api_key_multipart_oversized_body():
    body = b'--zzz\r\nContent-Disposition: form-data; name="api_key"\r\n\r\n' + b"x" * 4096
    request = _make_request(
        headers={"Content-Type": "multipart/form-data; boundary=zzz", "content-length": str(byok_mod.BYOK_FORM_MAX_BYTES + 1)},
        body=body,
    )
    with pytest.raises(openai_mod.HTTPException) as excinfo:
        await openai_mod._extract_request_api_key(request)
    assert excinfo.value.status_code == 413


async def test_extract_api_key_multipart_bad_content_length():
    request = _make_request(
        headers={"Content-Type": "multipart/form-data; boundary=zzz", "content-length": "abc"},
        body=b"x",
    )
    with pytest.raises(openai_mod.HTTPException) as excinfo:
        await openai_mod._extract_request_api_key(request)
    assert excinfo.value.status_code == 400


async def test_byok_pool_for_multipart_400():
    request = _make_request(headers={"Content-Type": "multipart/form-data; boundary=zzz"}, body=b"garbage")
    with pytest.raises(openai_mod.HTTPException) as excinfo:
        await openai_mod._byok_pool_for("qwen", request)
    assert excinfo.value.status_code == 400


async def test_byok_pool_for_missing_key_401():
    with pytest.raises(openai_mod.HTTPException) as excinfo:
        await openai_mod._byok_pool_for("deepseek", _make_request(headers={}))
    assert excinfo.value.status_code == 401
    assert excinfo.value.detail == byok_mod._INVALID_KEY_DETAIL.format(provider="deepseek")


async def test_byok_pool_for_missing_and_rejected_keys_are_indistinguishable(monkeypatch):
    monkeypatch.setattr(settings, "cache_enabled", False)
    missing = _make_request(headers={})
    with pytest.raises(openai_mod.HTTPException) as missing_exc:
        await openai_mod._byok_pool_for("deepseek", missing)

    monkeypatch.setattr(openai_mod.DeepSeekClient, "check_auth", AsyncMock(return_value=False))
    rejected = _make_request(headers={"Authorization": "Bearer rejected-key"})
    with pytest.raises(openai_mod.HTTPException) as rejected_exc:
        await openai_mod._byok_pool_for("deepseek", rejected)

    assert missing_exc.value.status_code == rejected_exc.value.status_code == 401
    assert missing_exc.value.detail == rejected_exc.value.detail


async def test_byok_pool_for_transport_failure_is_503(monkeypatch):
    monkeypatch.setattr(settings, "cache_enabled", False)

    async def boom(self):
        raise OSError("network down")

    monkeypatch.setattr(openai_mod.DeepSeekClient, "check_auth", boom)
    request = _make_request(headers={"Authorization": "Bearer unreachable-key"})
    with pytest.raises(openai_mod.HTTPException) as excinfo:
        await openai_mod._byok_pool_for("deepseek", request)
    assert excinfo.value.status_code == 503
    assert excinfo.value.detail == byok_mod._UNREACHABLE_DETAIL.format(provider="deepseek")


async def test_byok_pool_for_rejects_more_than_the_key_limit():
    request = _make_request(headers={"Authorization": "Bearer " + ",".join(f"k{index}" for index in range(byok_mod.BYOK_MAX_KEYS + 1))})
    with pytest.raises(openai_mod.HTTPException) as excinfo:
        await openai_mod._byok_pool_for("deepseek", request)
    assert excinfo.value.status_code == 400
    assert str(byok_mod.BYOK_MAX_KEYS) in excinfo.value.detail


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


def test_completions_stream_byok_401_without_key():
    app.state.byok = True
    client = TestClient(app)
    response = client.post(
        "/v1/completions",
        json={"model": "deepseek-v4.1-flash", "prompt": "hello", "stream": True},
    )
    client.close()
    assert response.status_code == 401
    assert response.headers.get("content-type", "").startswith("application/json")
    assert "text/event-stream" not in response.text


def test_completions_non_stream_byok_401_without_key():
    app.state.byok = True
    client = TestClient(app)
    response = client.post(
        "/v1/completions",
        json={"model": "deepseek-v4.1-flash", "prompt": "hello"},
    )
    client.close()
    assert response.status_code == 401


def test_image_edits_byok_401_without_key():
    app.state.byok = True
    client = TestClient(app)
    response = client.post(
        "/v1/images/edits",
        files={"image": ("a.png", b"png-bytes", "image/png")},
        data={"prompt": "make it blue"},
    )
    client.close()
    assert response.status_code == 401


def test_image_edits_byok_auth_from_api_key_field(monkeypatch):
    monkeypatch.setattr(settings, "cache_enabled", False)
    monkeypatch.setattr(openai_mod.QwenClient, "check_auth", AsyncMock(return_value=True))
    monkeypatch.setattr(openai_mod.QwenClient, "fetch_models", AsyncMock(return_value=[{"id": "q1", "info": {"meta": {"chat_type": ["t2t"]}}}]))
    app.state.byok = True
    app.state.qwen_models = []
    captured = {}

    async def fake_image_generations(req, resolved_pool):
        captured["pool"] = resolved_pool
        return {"created": 1, "data": [], "usage": None, "session_id": None}

    monkeypatch.setattr(images_mod, "_image_generations", fake_image_generations)
    client = TestClient(app)
    response = client.post(
        "/v1/images/edits",
        files={"image": ("a.png", b"png-bytes", "image/png")},
        data={"prompt": "make it blue", "api_key": "form-key"},
    )
    client.close()
    assert response.status_code == 200
    assert captured["pool"] is not None
    assert captured["pool"].label == "qwen"
    assert openai_mod._byok_cache_key(["form-key"]) in app.state.byok_pools["qwen"]


def test_health_detail_reports_byok_mode():
    app.state.byok = True
    detail = openai_mod._health_detail(True, app.state.byok_pools)
    assert detail["byok"] is True
    assert detail["byok_pools"]["deepseek"] == 0
    assert detail["byok_pools"]["qwen"] == 0
    for provider in openai_mod.BYOK_PROVIDERS:
        assert detail[provider] is True
        assert detail[f"{provider}_stats"]["pools"] == 0
    assert detail["byok_api_key_required"] == {
        "deepseek": True,
        "qwen": True,
        "gigachat": True,
        "opencode": True,
        "alice": False,
        "duckai": False,
    }


def test_health_hides_byok_details_from_an_anonymous_caller():
    app.state.byok = True
    client = TestClient(app)
    payload = client.get("/health").json()
    client.close()
    assert payload == {"status": "ok"}
    assert "byok" not in payload
    assert "byok_pools" not in payload
    assert not [key for key in payload if key.endswith("_stats")]


def test_health_byok_reports_pool_stats():
    app.state.byok = True
    pool = MagicMock()
    pool.stats.return_value = {"accounts": 2, "healthy": 2, "broken": 0}
    app.state.byok_pools["deepseek"]["key-a"] = pool
    app.state.byok_alice_pool = pool
    saved_deepseek = list(getattr(app.state, "deepseek_models", None) or [])
    app.state.deepseek_models = []
    try:
        detail = openai_mod._health_detail(True, app.state.byok_pools)
        assert detail["byok_pools"]["deepseek"] == 1
        assert detail["deepseek_stats"] == {"pools": 1, "accounts": 2, "healthy": 2, "broken": 0, "models": 0}
        assert detail["alice_stats"]["pools"] == 1
        assert detail["alice_stats"]["accounts"] == 2
    finally:
        app.state.byok_pools["deepseek"].clear()
        app.state.byok_alice_pool = None
        app.state.deepseek_models = saved_deepseek


async def test_keyless_pool_is_registered_under_byok_pools(monkeypatch):
    monkeypatch.setattr(byok_mod.AliceClient, "check_auth", AsyncMock(return_value=True))
    monkeypatch.setattr(byok_mod, "refresh_provider_models", AsyncMock(return_value=None))
    saved = byok_mod._ALICE_BYOK_POOL[0]
    byok_mod._ALICE_BYOK_POOL[0] = None
    app.state.byok_alice_pool = None
    try:
        pool = await openai_mod._byok_pool_for("alice", _make_request(headers={}))
        assert app.state.byok_pools["alice"][byok_mod.KEYLESS_POOL_KEY] is pool
        assert app.state.byok_alice_pool is pool
        assert pool.healthy
    finally:
        byok_mod._ALICE_BYOK_POOL[0] = saved
        app.state.byok_alice_pool = None
        app.state.byok_pools["alice"].pop(byok_mod.KEYLESS_POOL_KEY, None)


async def test_byok_key_lock_is_per_key_not_per_provider():
    first = byok_mod._key_lock("deepseek", "k1")
    second = byok_mod._key_lock("deepseek", "k2")
    third = byok_mod._key_lock("qwen", "k1")
    assert first is not second
    assert first is not third
    assert byok_mod._key_lock("deepseek", "k1") is first


async def test_byok_alice_needs_no_api_key(monkeypatch):
    app.state.byok = True
    captured = {}

    async def fake_byok_pool(provider, tokens):
        captured["provider"] = provider
        captured["tokens"] = tokens
        return MagicMock()

    monkeypatch.setattr(byok_mod, "_byok_pool", fake_byok_pool)
    pool = await openai_mod._byok_pool_for("alice", _make_request(headers={}))
    assert pool is not None
    assert captured == {"provider": "alice", "tokens": []}


async def test_byok_duckai_needs_no_api_key(monkeypatch):
    app.state.byok = True
    captured = {}

    async def fake_byok_pool(provider, tokens):
        captured["provider"] = provider
        captured["tokens"] = tokens
        return MagicMock()

    monkeypatch.setattr(byok_mod, "_byok_pool", fake_byok_pool)
    await openai_mod._byok_pool_for("duckai", _make_request(headers={}))
    assert captured == {"provider": "duckai", "tokens": []}


async def test_byok_keyed_provider_still_requires_api_key():
    app.state.byok = True
    with pytest.raises(HTTPException) as excinfo:
        await openai_mod._byok_pool_for("gigachat", _make_request(headers={}))
    assert excinfo.value.status_code == 401
    assert "gigachat" in excinfo.value.detail


def test_list_models_byok_without_env_tokens_lists_live_models():
    app.state.byok = True
    app.state.qwen_models = [{"id": "qwen-live", "name": "Q", "owned_by": "qwen", "model_type": "chat"}]
    try:
        client = TestClient(app)
        data = client.get("/v1/models").json()
        client.close()
        ids = [m["id"] for m in data["data"]]
        assert "qwen-live" in ids
    finally:
        app.state.qwen_models = []


def test_health_no_byok_key_when_disabled():
    app.state.byok = False
    client = TestClient(app)
    payload = client.get("/health").json()
    assert "byok" not in payload
