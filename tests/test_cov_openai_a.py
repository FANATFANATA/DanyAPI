import asyncio
import base64
import time
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient

import danyapi.api.byok as byok_mod
import danyapi.api.chats as chats_mod
import danyapi.api.core as core_mod
import danyapi.api.envtokens as envtokens_mod
import danyapi.api.images as images_mod
import danyapi.api.models as models_mod
import danyapi.api.openai as openai_mod
import danyapi.api.retry as retry_mod
from danyapi.accounts import AccountPoolBusy
from danyapi.api.openai import ChatMessage, app, settings

ADMIN_HEADERS = {"authorization": "Bearer test-admin-token"}

OK_SSE = (
    "event: ready\n"
    'data: {"request_message_id":1,"response_message_id":2,"model_type":"default"}\n'
    "\n"
    'data: {"v":{"response":{"message_id":2,"parent_id":1,"status":"WIP","fragments":[{"id":2,"type":"RESPONSE","content":"Hi"}]}}}\n'
    "\n"
    'data: {"p":"response/status","o":"SET","v":"FINISHED"}\n'
    "\n"
)


class FakeSession:
    def __init__(self, sid="c1", last_message_id=None):
        self.id = sid
        self.last_message_id = last_message_id
        self.accumulated_tokens = 0


class FakeResp:
    def __init__(self, body=None, sse_text=None, status=200, content_type="text/event-stream; charset=utf-8"):
        self.status_code = status
        self.headers = {"content-type": content_type}
        self._b = (sse_text if sse_text is not None else (body or "")).encode()

    async def aiter_bytes(self):
        yield self._b

    async def aclose(self):
        pass

    async def aread(self):
        return self._b


class FakeAccount:
    def __init__(self, sse_list=None):
        self.index = 0
        self.broken = False
        self.client = MagicMock()
        self.client.completion = AsyncMock(side_effect=[FakeResp(sse_text=s) for s in (sse_list or [OK_SSE])])
        self.client.create_pow_challenge = AsyncMock(return_value={})
        self.pow = MagicMock()
        self.pow.make_header = AsyncMock(return_value={})
        self.pow_upload = MagicMock()
        self.pow_upload.make_header = AsyncMock(return_value={})
        self.sem = asyncio.Semaphore(1)
        self.sessions = MagicMock()
        self.sessions.obtain = AsyncMock(return_value=(FakeSession(), "s1"))
        self.sessions.touch_last_message = MagicMock()
        self.sessions.forget = MagicMock()

    def mark_broken(self):
        self.broken = True


def make_pool(acct=None):
    pool = MagicMock()
    acct = acct or FakeAccount()
    pool.acquire = AsyncMock(return_value=(acct, None))
    return pool, acct


class FakeRequest:
    def __init__(self, headers=None, body=b"", stream_chunks=None, _body=None, method="POST"):
        self.headers = headers or {}
        self._raw_body = body
        self._chunks = stream_chunks or []
        self._body = _body
        self.consumed: list[int] = []
        self.body_calls = 0
        self.method = method

    async def body(self):
        self.body_calls += 1
        return self._raw_body

    async def stream(self):
        for chunk in self._chunks:
            self.consumed.append(len(chunk))
            yield chunk


class FakeImageResponse:
    def __init__(self, status_code=200, chunks=(b"abc",), headers=None, url="http://images.test/1.png"):
        self.status_code = status_code
        self.chunks = list(chunks)
        self.headers = headers or {}
        self.url = httpx.URL(url)

    async def aiter_bytes(self):
        for chunk in self.chunks:
            yield chunk


class FakeStreamContext:
    def __init__(self, response):
        self.response = response

    async def __aenter__(self):
        return self.response

    async def __aexit__(self, *exc_info):
        return False


class FakeImageClient:
    def __init__(self, responses=None):
        self.responses = {url: list(scripted) for url, scripted in (responses or {}).items()}
        self.calls: list[tuple[str, str, float, bool]] = []

    def stream(self, method, url, timeout=None, follow_redirects=True):
        self.calls.append((method, str(url), timeout, follow_redirects))
        scripted = self.responses.get(str(url), [])
        response = scripted.pop(0) if scripted else FakeImageResponse(status_code=404, url=str(url))
        if isinstance(response, BaseException):
            raise response
        return FakeStreamContext(response)


@pytest.fixture(autouse=True)
def clean_state():
    saved = (
        getattr(app.state, "pool", None),
        getattr(app.state, "qwen_pool", None),
        getattr(app.state, "qwen_models", None),
        getattr(app.state, "responses_store", None),
    )
    app.state.pool = None
    app.state.qwen_pool = None
    app.state.qwen_models = []
    openai_mod._MODEL_CACHE["key"] = None
    openai_mod._MODEL_CACHE["models"] = None
    openai_mod._MODEL_CACHE["index"] = None
    openai_mod._MODEL_CACHE["qwen_ids"] = None
    yield
    app.state.pool, app.state.qwen_pool, app.state.qwen_models, app.state.responses_store = saved


@pytest.fixture(autouse=True)
def zero_backoff(monkeypatch):
    monkeypatch.setattr(retry_mod, "RETRY_BACKOFF_SEC", 0.0)


@pytest.fixture
def reset_app_state():
    yield
    app.state.pool = None
    app.state.qwen_pool = None
    app.state.qwen_models = []


def _patch_creds(ds_tokens=None, qwen_tokens=None, cache=True):
    from danyapi.deepseek.client import DeepSeekClient as DSC
    from danyapi.qwen.client import QwenClient as QC

    stack = ExitStack()
    for patch_cm in (
        patch.object(settings, "deepseek_tokens", ds_tokens or []),
        patch.object(settings, "qwen_tokens", qwen_tokens or []),
        patch.object(settings, "cache_enabled", cache),
        patch.object(DSC, "check_auth", new=AsyncMock(return_value=True)),
        patch.object(QC, "check_auth", new=AsyncMock(return_value=True)),
        patch.object(QC, "fetch_models", new=AsyncMock(return_value=[{"id": "m1", "info": {"meta": {"chat_type": ["t2t"]}}}])),
        patch.object(DSC, "aclose", new=AsyncMock()),
        patch.object(QC, "aclose", new=AsyncMock()),
    ):
        stack.enter_context(patch_cm)
    return stack


def test_chat_message_invalid_role():
    with pytest.raises(ValueError):
        ChatMessage(role="bogus", content="x")


def test_chat_message_tool_calls_not_list():
    with pytest.raises(ValueError):
        ChatMessage(role="user", tool_calls="nope")


def test_chat_message_tool_calls_bad_item():
    with pytest.raises(ValueError):
        ChatMessage(role="user", tool_calls=[42])


def test_chat_message_tool_calls_missing_function():
    with pytest.raises(ValueError):
        ChatMessage(role="user", tool_calls=[{"id": "1"}])


def test_chat_message_tool_calls_ok():
    assert ChatMessage(role="user", tool_calls=None).tool_calls is None
    assert ChatMessage(role="user", tool_calls=[{"id": "1", "function": {"name": "f"}}]).tool_calls
    assert ChatMessage(role="user", tool_calls=[{"id": "1", "name": "f"}]).tool_calls


def test_resize_image_bytes_bad_data():
    assert openai_mod._resize_image_bytes(b"notimage", (16, 16)) == b"notimage"


def test_resize_image_bytes_jpeg_cmyk():
    from io import BytesIO

    from PIL import Image

    img = Image.new("CMYK", (32, 32), (0, 0, 0, 0))
    buf = BytesIO()
    img.save(buf, format="JPEG")
    out = openai_mod._resize_image_bytes(buf.getvalue(), (16, 16))
    assert isinstance(out, bytes)


def test_flush_state_stores_errors(monkeypatch):
    tracker = MagicMock()
    tracker.flush.side_effect = RuntimeError("x")
    monkeypatch.setattr(app.state, "usage", tracker, raising=False)
    store = MagicMock()
    store.flush.side_effect = RuntimeError("x")
    monkeypatch.setattr(app.state, "deepseek_session_store", store, raising=False)
    monkeypatch.setattr(app.state, "qwen_session_store", None, raising=False)
    pool = MagicMock()
    pool.flush.side_effect = RuntimeError("x")
    monkeypatch.setattr(app.state, "pool", pool, raising=False)
    monkeypatch.setattr(app.state, "qwen_pool", None, raising=False)
    monkeypatch.setattr(app.state, "byok_pools", {"deepseek": {"k": pool}, "qwen": {}}, raising=False)
    openai_mod._flush_state_stores()


def test_flush_state_stores_skips(monkeypatch):
    monkeypatch.setattr(app.state, "usage", None, raising=False)
    for attr in openai_mod._STATE_STORE_ATTRS:
        monkeypatch.setattr(app.state, attr, None, raising=False)
    plain = SimpleNamespace()
    monkeypatch.setattr(app.state, "pool", plain, raising=False)
    monkeypatch.setattr(app.state, "qwen_pool", plain, raising=False)
    monkeypatch.setattr(app.state, "byok_pools", None, raising=False)
    openai_mod._flush_state_stores()


@pytest.mark.usefixtures("reset_app_state")
def test_lifespan_usage_disabled():
    with patch.object(settings, "usage_enabled", False):
        with _patch_creds(ds_tokens=["tok"]):
            with TestClient(app):
                assert app.state.usage is None


@pytest.mark.usefixtures("reset_app_state")
def test_lifespan_closes_http_client_and_byok_pools():
    http_client = MagicMock()
    http_client.aclose = AsyncMock()
    with _patch_creds(ds_tokens=["tok"]):
        with TestClient(app):
            assert set(app.state.byok_pools) == set(openai_mod.BYOK_PROVIDERS)
            assert set(app.state.byok_locks) == set(openai_mod.BYOK_PROVIDERS)
            assert set(app.state.byok_auth) == set(openai_mod.BYOK_PROVIDERS)
            assert set(app.state.byok_stores) == set(openai_mod.BYOK_PROVIDERS)
            app.state.http_client = http_client
            fake_pool = MagicMock()
            fake_pool.accounts = []
            app.state.byok_pools = {"deepseek": {"k": fake_pool}, "qwen": {}}
    http_client.aclose.assert_awaited()


async def test_fetch_qwen_models_types():
    client = MagicMock()
    client.fetch_models = AsyncMock(
        return_value=[
            {"id": "i1", "info": {"meta": {"chat_type": ["t2i"]}}},
            {"id": "v1", "info": {"meta": {"chat_type": ["t2v"]}}},
        ]
    )
    result = await openai_mod._fetch_qwen_models(client)
    types = {m["id"]: m["model_type"] for m in result}
    assert types["i1"] == "image"
    assert types["v1"] == "video"


def test_root_endpoint():
    client = TestClient(app)
    resp = client.get("/")
    client.close()
    assert resp.status_code in (200, 404)


def test_favicon():
    client = TestClient(app)
    resp = client.get("/favicon.ico")
    client.close()
    assert resp.status_code == 204


def test_env_path():
    assert openai_mod._env_path().name == ".env"


def test_unquote_env_value():
    assert openai_mod._unquote_env_value('"abc"') == "abc"
    assert openai_mod._unquote_env_value("'abc'") == "abc"
    assert openai_mod._unquote_env_value("abc") == "abc"
    assert openai_mod._unquote_env_value('"') == '"'


def test_read_env_tokens_sync_no_file(tmp_path, monkeypatch):
    monkeypatch.setattr(envtokens_mod, "_env_path", lambda: tmp_path / "missing.env")
    assert openai_mod._read_env_tokens_sync() == ([], [])


def test_read_env_tokens_sync_parses(tmp_path, monkeypatch):
    path = tmp_path / "t.env"
    path.write_text("DEEPSEEK_TOKENS=\"a,b\"\nQWEN_TOKENS='c'\n", encoding="utf-8")
    monkeypatch.setattr(envtokens_mod, "_env_path", lambda: path)
    assert openai_mod._read_env_tokens_sync() == (["a", "b"], ["c"])


def test_write_env_tokens_sync(tmp_path, monkeypatch):
    path = tmp_path / "t.env"
    path.write_text("DEEPSEEK_TOKENS=old\nOTHER=1\n", encoding="utf-8")
    monkeypatch.setattr(envtokens_mod, "_env_path", lambda: path)
    openai_mod._write_env_tokens_sync(["a"], ["b"])
    text = path.read_text(encoding="utf-8")
    assert "DEEPSEEK_TOKENS=a" in text
    assert "QWEN_TOKENS=b" in text
    assert "OTHER=1" in text


def test_pool_account_by_stable_none():
    assert openai_mod._pool_account_by_stable(None, "x") is None


def test_pool_account_by_stable_missing():
    acct = SimpleNamespace(stable_id="a")
    pool = SimpleNamespace(accounts=[acct])
    assert openai_mod._pool_account_by_stable(pool, "a") is acct
    assert openai_mod._pool_account_by_stable(pool, "z") is None


def test_add_tokens_reactivates_deepseek(monkeypatch):
    token = "tok"
    acct = SimpleNamespace(stable_id=openai_mod._token_stable_id(token), broken=True, broken_at=1, client=MagicMock())
    acct.client.check_auth = AsyncMock(return_value=True)
    app.state.pool = SimpleNamespace(accounts=[acct])
    monkeypatch.setattr(envtokens_mod, "_read_env_tokens", AsyncMock(return_value=([token], [])))
    client = TestClient(app)
    resp = client.post("/v1/tokens", headers=ADMIN_HEADERS, json={"deepseek_tokens": [token]})
    client.close()
    assert resp.status_code == 200
    assert resp.json()["reactivated"]["deepseek"] == 1
    assert acct.broken is False


def test_add_tokens_reactivate_check_raises(monkeypatch):
    token = "tok"
    acct = SimpleNamespace(stable_id=openai_mod._token_stable_id(token), broken=True, client=MagicMock())
    acct.client.check_auth = AsyncMock(side_effect=RuntimeError("x"))
    app.state.pool = SimpleNamespace(accounts=[acct])
    monkeypatch.setattr(envtokens_mod, "_read_env_tokens", AsyncMock(return_value=([token], [])))
    client = TestClient(app)
    resp = client.post("/v1/tokens", headers=ADMIN_HEADERS, json={"deepseek_tokens": [token]})
    client.close()
    assert resp.status_code == 503
    assert resp.json()["error"]["type"] == "api_error"
    assert "token auth check could not reach the upstream" in resp.json()["error"]["message"]
    assert "RuntimeError: x" in resp.json()["error"]["message"]


def test_add_tokens_reactivate_not_valid(monkeypatch):
    token = "tok"
    acct = SimpleNamespace(stable_id=openai_mod._token_stable_id(token), broken=True, client=MagicMock())
    acct.client.check_auth = AsyncMock(return_value=False)
    app.state.pool = SimpleNamespace(accounts=[acct])
    monkeypatch.setattr(envtokens_mod, "_read_env_tokens", AsyncMock(return_value=([token], [])))
    client = TestClient(app)
    resp = client.post("/v1/tokens", headers=ADMIN_HEADERS, json={"deepseek_tokens": [token]})
    client.close()
    assert resp.status_code == 400


def test_add_tokens_reactivates_qwen(monkeypatch):
    token = "qt"
    acct = SimpleNamespace(stable_id=openai_mod._token_stable_id(token), broken=True, broken_at=1, client=MagicMock())
    acct.client.check_auth = AsyncMock(return_value=True)
    app.state.qwen_pool = SimpleNamespace(accounts=[acct])
    monkeypatch.setattr(envtokens_mod, "_read_env_tokens", AsyncMock(return_value=([], [token])))
    client = TestClient(app)
    resp = client.post("/v1/tokens", headers=ADMIN_HEADERS, json={"qwen_tokens": [token]})
    client.close()
    assert resp.status_code == 200
    assert resp.json()["reactivated"]["qwen"] == 1


def test_add_tokens_hot_adds_both(monkeypatch):
    monkeypatch.setattr(settings, "deepseek_tokens", [])
    monkeypatch.setattr(settings, "qwen_tokens", [])
    monkeypatch.setattr(envtokens_mod, "_write_env_tokens", AsyncMock())
    monkeypatch.setattr(envtokens_mod, "_read_env_tokens", AsyncMock(return_value=([], [])))
    ds_client = MagicMock()
    ds_client.check_auth = AsyncMock(return_value=True)
    qw_client = MagicMock()
    qw_client.check_auth = AsyncMock(return_value=True)
    monkeypatch.setattr(envtokens_mod, "DeepSeekClient", MagicMock(return_value=ds_client))
    monkeypatch.setattr(envtokens_mod, "QwenClient", MagicMock(return_value=qw_client))
    pool = MagicMock()
    pool.accounts = []
    qwen_pool = MagicMock()
    qwen_pool.accounts = []
    app.state.pool = pool
    app.state.qwen_pool = qwen_pool
    monkeypatch.setattr(envtokens_mod, "_fetch_qwen_models", AsyncMock(side_effect=RuntimeError("boom")))
    client = TestClient(app)
    resp = client.post("/v1/tokens", headers=ADMIN_HEADERS, json={"deepseek_tokens": ["d"], "qwen_tokens": ["q"]})
    client.close()
    assert resp.status_code == 200
    assert resp.json()["added"] == {"deepseek": 1, "qwen": 1}
    pool.add_account.assert_called_once()
    qwen_pool.add_account.assert_called_once()


def test_add_tokens_skips_invalid(monkeypatch):
    monkeypatch.setattr(settings, "deepseek_tokens", [])
    monkeypatch.setattr(settings, "qwen_tokens", [])
    monkeypatch.setattr(envtokens_mod, "_write_env_tokens", AsyncMock())
    monkeypatch.setattr(envtokens_mod, "_read_env_tokens", AsyncMock(return_value=([], [])))
    ds_client = MagicMock()
    ds_client.check_auth = AsyncMock(return_value=False)
    ds_client.aclose = AsyncMock()
    qw_client = MagicMock()
    qw_client.check_auth = AsyncMock(return_value=False)
    qw_client.aclose = AsyncMock()
    monkeypatch.setattr(envtokens_mod, "DeepSeekClient", MagicMock(return_value=ds_client))
    monkeypatch.setattr(envtokens_mod, "QwenClient", MagicMock(return_value=qw_client))
    app.state.pool = None
    app.state.qwen_pool = None
    client = TestClient(app)
    resp = client.post("/v1/tokens", headers=ADMIN_HEADERS, json={"deepseek_tokens": ["d"], "qwen_tokens": ["q"]})
    client.close()
    assert resp.status_code == 200
    assert resp.json()["skipped"] == {"deepseek": 1, "qwen": 1}
    assert resp.json()["added"] == {"deepseek": 0, "qwen": 0}
    assert resp.json()["activated"] == {"deepseek": 0, "qwen": 0}
    assert resp.json()["reactivated"] == {"deepseek": 0, "qwen": 0}
    assert resp.json()["message"] == "No valid tokens to add."
    assert app.state.pool is None
    assert app.state.qwen_pool is None


def test_add_tokens_all_exist(monkeypatch):
    monkeypatch.setattr(envtokens_mod, "_read_env_tokens", AsyncMock(return_value=(["a"], ["b"])))
    ds = SimpleNamespace(stable_id=openai_mod._token_stable_id("a"), broken=False, client=MagicMock())
    qw = SimpleNamespace(stable_id=openai_mod._token_stable_id("b"), broken=False, client=MagicMock())
    app.state.pool = SimpleNamespace(accounts=[ds])
    app.state.qwen_pool = SimpleNamespace(accounts=[qw])
    client = TestClient(app)
    resp = client.post("/v1/tokens", headers=ADMIN_HEADERS, json={"deepseek_tokens": ["a"], "qwen_tokens": ["b"]})
    client.close()
    assert resp.status_code == 400
    assert resp.json()["error"]["message"] == "all provided tokens already exist"
    ds.client.check_auth.assert_not_called()
    qw.client.check_auth.assert_not_called()


def test_add_tokens_activates_token_already_in_env_file(monkeypatch):
    monkeypatch.setattr(settings, "deepseek_tokens", [])
    monkeypatch.setattr(settings, "qwen_tokens", [])
    write = AsyncMock()
    monkeypatch.setattr(envtokens_mod, "_write_env_tokens", write)
    monkeypatch.setattr(envtokens_mod, "_read_env_tokens", AsyncMock(return_value=(["tok"], [])))
    ds_client = MagicMock()
    ds_client.check_auth = AsyncMock(return_value=True)
    monkeypatch.setattr(envtokens_mod, "DeepSeekClient", MagicMock(return_value=ds_client))
    app.state.pool = None
    client = TestClient(app)
    resp = client.post("/v1/tokens", headers=ADMIN_HEADERS, json={"deepseek_tokens": ["tok"]})
    client.close()
    assert resp.status_code == 200
    assert resp.json()["activated"] == {"deepseek": 1, "qwen": 0}
    assert resp.json()["added"] == {"deepseek": 0, "qwen": 0}
    assert resp.json()["message"] == "Tokens reactivated."
    assert app.state.pool is not None
    write.assert_not_awaited()


def test_add_tokens_does_not_rewrite_env_for_environment_token(monkeypatch):
    monkeypatch.setattr(settings, "deepseek_tokens", ["envtok"])
    monkeypatch.setattr(settings, "qwen_tokens", [])
    write = AsyncMock()
    monkeypatch.setattr(envtokens_mod, "_write_env_tokens", write)
    monkeypatch.setattr(envtokens_mod, "_read_env_tokens", AsyncMock(return_value=([], [])))
    ds_client = MagicMock()
    ds_client.check_auth = AsyncMock(return_value=True)
    monkeypatch.setattr(envtokens_mod, "DeepSeekClient", MagicMock(return_value=ds_client))
    app.state.pool = None
    client = TestClient(app)
    resp = client.post("/v1/tokens", headers=ADMIN_HEADERS, json={"deepseek_tokens": ["envtok"]})
    client.close()
    assert resp.status_code == 200
    assert resp.json()["added"] == {"deepseek": 1, "qwen": 0}
    assert resp.json()["activated"] == {"deepseek": 0, "qwen": 0}
    write.assert_not_awaited()
    assert settings.deepseek_tokens == ["envtok"]
    assert app.state.pool is not None
    assert len(app.state.pool.accounts) == 1


def test_add_tokens_no_tokens():
    client = TestClient(app)
    resp = client.post("/v1/tokens", headers=ADMIN_HEADERS, json={})
    client.close()
    assert resp.status_code == 400


def test_add_tokens_non_list():
    client = TestClient(app)
    resp = client.post("/v1/tokens", headers=ADMIN_HEADERS, json={"deepseek_tokens": "x"})
    client.close()
    assert resp.status_code == 400


def test_add_tokens_bad_item():
    client = TestClient(app)
    resp = client.post("/v1/tokens", headers=ADMIN_HEADERS, json={"deepseek_tokens": [42]})
    client.close()
    assert resp.status_code == 400


async def test_read_request_body_invalid_content_length():
    assert core_mod._declared_body_length(FakeRequest(headers={"content-length": "abc"})) == -1
    assert core_mod._declared_body_length(FakeRequest(headers={})) == -1
    assert core_mod._declared_body_length(FakeRequest(headers={"content-length": ""})) == -1
    assert core_mod._declared_body_length(FakeRequest(headers={"content-length": "12"})) == 12
    req = FakeRequest(headers={"content-length": "abc"}, _body=b"x")
    assert await openai_mod._read_request_body(req, 10) == b"x"
    streamed = FakeRequest(headers={"content-length": "abc"}, stream_chunks=[b"x"])
    assert await openai_mod._read_request_body(streamed, 10) == b"x"


async def test_read_request_body_header_too_large():
    req = FakeRequest(headers={"content-length": "100"}, stream_chunks=[b"x" * 100])
    with pytest.raises(HTTPException) as excinfo:
        await openai_mod._read_request_body(req, 10)
    assert excinfo.value.status_code == 413
    assert req.consumed == []


async def test_read_request_body_actual_too_large():
    req = FakeRequest(headers={"content-length": "5"}, stream_chunks=[b"x" * 5, b"y" * 6, b"z" * 100])
    with pytest.raises(HTTPException) as excinfo:
        await openai_mod._read_request_body(req, 10)
    assert excinfo.value.status_code == 413
    assert req.consumed == [5, 6]


async def test_read_request_body_cached_too_large():
    req = FakeRequest(headers={}, _body=b"x" * 20)
    with pytest.raises(HTTPException) as excinfo:
        await openai_mod._read_request_body(req, 10)
    assert excinfo.value.status_code == 413


async def test_read_request_body_stream_too_large():
    req = FakeRequest(headers={}, stream_chunks=[b"x" * 8, b"y" * 8, b"z" * 100])
    with pytest.raises(HTTPException) as excinfo:
        await openai_mod._read_request_body(req, 10)
    assert excinfo.value.status_code == 413
    assert req.consumed == [8, 8]


async def test_read_request_body_streams_and_caches():
    req = FakeRequest(headers={"content-length": "7"}, stream_chunks=[b'{"a"', b":1}"])
    assert await openai_mod._read_request_body(req, 10) == b'{"a":1}'
    assert req._body == b'{"a":1}'
    req.consumed.clear()
    assert await openai_mod._read_request_body(req, 10) == b'{"a":1}'
    assert req.consumed == []


def test_parse_logged_body():
    assert openai_mod._parse_logged_body(b"") == {}
    assert openai_mod._parse_logged_body(b"x" * (openai_mod.MAX_LOGGED_BODY + 1)) == {}
    assert openai_mod._parse_logged_body(b"notjson") == {}
    assert openai_mod._parse_logged_body(b"[1,2]") == {}
    assert openai_mod._parse_logged_body(b'{"a":1}') == {"a": 1}


async def test_extract_request_body_variants(monkeypatch):
    assert await openai_mod._extract_request_body(FakeRequest(headers={"content-length": "0"})) == {}
    assert await openai_mod._extract_request_body(FakeRequest(headers={"content-length": str(openai_mod.MAX_LOGGED_BODY + 1)})) == {}
    assert await openai_mod._extract_request_body(FakeRequest(headers={"content-length": "abc"})) == {}
    assert await openai_mod._extract_request_body(FakeRequest(headers={}, method="GET")) == {}
    assert await openai_mod._extract_request_body(FakeRequest(headers={}, method="POST")) == {}
    assert await openai_mod._extract_request_body(FakeRequest(headers={}, method="POST", _body=b'{"a":1}')) == {}
    assert await openai_mod._extract_request_body(FakeRequest(headers={"content-length": "7"}, method="POST", _body=b'{"a":1}')) == {"a": 1}
    streamed = FakeRequest(headers={"content-length": "7"}, method="POST", stream_chunks=[b'{"a"', b":1}"])
    assert await openai_mod._extract_request_body(streamed) == {"a": 1}
    assert streamed._body == b'{"a":1}'
    assert streamed.consumed == [4, 3]
    undeclared = FakeRequest(headers={}, method="POST", stream_chunks=[b'{"a":1}'])
    assert await openai_mod._extract_request_body(undeclared) == {}
    assert undeclared.consumed == []


async def test_extract_request_body_generic_error(monkeypatch):
    class BadReq(FakeRequest):
        async def stream(self):
            raise RuntimeError("x")
            yield b""

    assert await openai_mod._extract_request_body(BadReq(headers={"content-length": "7"}, method="POST")) == {}


async def test_extract_request_body_http_exception(monkeypatch):
    monkeypatch.setattr(core_mod, "MAX_REQUEST_BODY", openai_mod.MAX_LOGGED_BODY + 1)
    req = FakeRequest(headers={"content-length": str(openai_mod.MAX_LOGGED_BODY + 2)}, method="POST")
    with pytest.raises(HTTPException) as excinfo:
        await openai_mod._extract_request_body(req)
    assert excinfo.value.status_code == 413
    assert req.consumed == []


async def test_extract_request_body_rejects_declared_length_over_request_limit():
    req = FakeRequest(headers={"content-length": str(openai_mod.MAX_REQUEST_BODY + 1)}, method="POST")
    with pytest.raises(HTTPException) as excinfo:
        await openai_mod._extract_request_body(req)
    assert excinfo.value.status_code == 413
    assert req.consumed == []
    cached = FakeRequest(headers={"content-length": str(openai_mod.MAX_REQUEST_BODY + 1)}, method="POST", _body=b'{"a":1}')
    with pytest.raises(HTTPException) as excinfo:
        await openai_mod._extract_request_body(cached)
    assert excinfo.value.status_code == 413


def test_error_type_for_status():
    assert openai_mod._error_type_for_status(403) == "permission_error"
    assert openai_mod._error_type_for_status(408) == "request_timeout"
    assert openai_mod._error_type_for_status(409) == "conflict_error"
    assert openai_mod._error_type_for_status(413) == "request_too_large"
    assert openai_mod._error_type_for_status(504) == "server_error"
    assert openai_mod._error_type_for_status(599) == "server_error"
    assert openai_mod._error_type_for_status(400) == "invalid_request_error"


def test_exception_message():
    assert openai_mod._exception_message(RuntimeError("")) == "internal server error"
    assert openai_mod._exception_message(RuntimeError("boom")) == "internal server error"


def test_request_id_header():
    first = openai_mod._request_id_header(FakeRequest(headers={"x-request-id": "abc"}))
    assert first != "abc"
    assert len(first) == 32
    assert int(first, 16) >= 0
    second = openai_mod._request_id_header(FakeRequest(headers={"x-request-id": "abc"}))
    assert second != first


def test_client_request_id_is_sanitised_and_capped():
    assert core_mod._client_request_id(FakeRequest(headers={})) is None
    assert core_mod._client_request_id(FakeRequest(headers={"x-request-id": "abc"})) == "abc"
    assert core_mod._client_request_id(FakeRequest(headers={"x-request-id": "a\r\nb\tc\x00d"})) == "a  b c d"
    assert core_mod._client_request_id(FakeRequest(headers={"x-request-id": "x" * 200})) == "x" * core_mod.MAX_CLIENT_REQUEST_ID
    assert core_mod._client_request_id(FakeRequest(headers={core_mod.CLIENT_REQUEST_ID_HEADER: "via-client-header"})) == "via-client-header"


def test_response_headers_separate_server_and_client_ids():
    headers = core_mod._response_headers(FakeRequest(headers={"x-request-id": "abc"}), "server-id")
    assert headers == {"x-request-id": "server-id", core_mod.CLIENT_REQUEST_ID_HEADER: "abc"}
    assert core_mod._response_headers(FakeRequest(headers={}), "server-id") == {"x-request-id": "server-id"}


def test_validation_error_handler():
    client = TestClient(app)
    resp = client.post("/v1/chat/completions", json={"messages": "notalist"}, headers={"x-request-id": "client-id"})
    client.close()
    assert resp.status_code == 400
    assert resp.headers["x-request-id"] != "client-id"
    assert len(resp.headers["x-request-id"]) == 32
    assert resp.headers[core_mod.CLIENT_REQUEST_ID_HEADER] == "client-id"
    assert resp.json()["error"]["request_id"] == resp.headers["x-request-id"]


async def test_http_exception_handler_with_headers():
    exc = HTTPException(429, "slow", headers={"retry-after": "1"})
    resp = await openai_mod._on_http_exception(FakeRequest(headers={}), exc)
    assert resp.status_code == 429
    assert resp.headers.get("retry-after") == "1"


async def test_uncaught_exception_handler():
    resp = await openai_mod._on_uncaught_exception(FakeRequest(headers={}), RuntimeError("boom"))
    assert resp.status_code == 500


def test_account_busy_count():
    sem = MagicMock()
    sem.locked.return_value = True
    busy_acct = SimpleNamespace(sem=sem)
    idle_acct = SimpleNamespace(sem=None)
    pool = SimpleNamespace(accounts=[busy_acct, idle_acct])
    assert openai_mod._account_busy_count(pool) == 1


def test_pool_rate_headers():
    assert openai_mod._pool_rate_headers(None) == {}
    assert openai_mod._pool_rate_headers(SimpleNamespace()) == {}
    pool = MagicMock()
    pool.stats.return_value = {"healthy": 2}
    pool.accounts = []
    headers = openai_mod._pool_rate_headers(pool)
    assert headers["x-ratelimit-limit-requests"] == "2"
    assert openai_mod._pool_rate_headers(pool) == headers


def test_pool_rate_cache_clear():
    openai_mod._POOL_RATE_CACHE.clear()
    for _ in range(18):
        pool = MagicMock()
        pool.stats.return_value = {"healthy": 1}
        pool.accounts = []
        openai_mod._pool_rate_headers(pool)
    assert len(openai_mod._POOL_RATE_CACHE) <= 18
    openai_mod._POOL_RATE_CACHE.clear()


def test_raw_data_uri_length():
    uri = "data:image/png;base64," + base64.b64encode(b"abc").decode()
    assert openai_mod._raw_data_uri_length(uri) == 3


def test_collect_attachments_invalid_b64():
    req = SimpleNamespace(
        messages=[ChatMessage(role="user", content=[{"type": "image_url", "image_url": {"url": "data:image/png;base64,@@@"}}])],
        files=None,
    )
    with pytest.raises(HTTPException) as excinfo:
        openai_mod._collect_attachments(req)
    assert excinfo.value.status_code == 400


async def test_upload_attachments_empty():
    acct = FakeAccount()
    assert await openai_mod._upload_attachments(acct, [], "default", False) == []


def test_resolve_model_suffix_unknown():
    assert not hasattr(openai_mod, "_resolve_model")
    assert not hasattr(openai_mod, "_upstream_model_for")
    with pytest.raises(HTTPException) as excinfo:
        models_mod._resolve_model("nope-thinking")
    assert excinfo.value.status_code == 404


def test_usage_summary_exception(monkeypatch):
    tracker = MagicMock()
    tracker.snapshot.side_effect = RuntimeError("x")
    monkeypatch.setattr(app.state, "usage", tracker, raising=False)
    assert openai_mod._usage_summary() is None


def test_all_models():
    assert isinstance(openai_mod._all_models(), list)


def test_get_model_found_and_missing():
    app.state.qwen_models = [{"id": "q1", "name": "Q", "owned_by": "qwen", "model_type": "chat"}]
    app.state.deepseek_models = [
        {
            "id": "default",
            "name": "Instant",
            "owned_by": "deepseek",
            "model_type": "chat",
            "upstream_type": "default",
            "is_default": True,
        }
    ]
    openai_mod._MODEL_CACHE["key"] = None
    try:
        client = TestClient(app)
        ok = client.get("/v1/models/default")
        alias = client.get("/v1/models/default-thinking")
        missing = client.get("/v1/models/does-not-exist")
        client.close()
        assert ok.status_code == 200
        assert alias.status_code == 200
        assert missing.status_code == 404
    finally:
        app.state.qwen_models = []
        app.state.deepseek_models = []


def test_resolve_provider_via_cache():
    app.state.qwen_models = [{"id": "special-1", "name": "S", "owned_by": "qwen", "model_type": "chat"}]
    openai_mod._MODEL_CACHE["key"] = None
    assert openai_mod._resolve_provider("special-1") == "qwen"


async def test_byok_state_helpers(monkeypatch):
    monkeypatch.setattr(app.state, "byok_pools", None, raising=False)
    monkeypatch.setattr(app.state, "byok_locks", None, raising=False)
    monkeypatch.setattr(app.state, "byok_auth", None, raising=False)
    monkeypatch.setattr(app.state, "byok_stores", None, raising=False)
    pools = openai_mod._byok_pools_state()
    locks = openai_mod._byok_locks_state()
    auth = openai_mod._byok_auth_state()
    stores = openai_mod._byok_stores_state()
    for state in (pools, locks, auth, stores):
        assert set(state) == set(openai_mod.BYOK_PROVIDERS)
    assert app.state.byok_pools is pools
    assert app.state.byok_locks is locks
    assert app.state.byok_auth is auth
    assert app.state.byok_stores is stores
    assert openai_mod._byok_pools_state() is pools


def test_pool_attrs_follow_provider_map():
    assert openai_mod.POOL_ATTRS == tuple(openai_mod.POOL_ATTRS_BY_PROVIDER[provider] for provider in openai_mod.BYOK_PROVIDERS)


def test_provider_state_for_unknown_provider():
    assert openai_mod.provider_models("not-a-provider") == []
    assert openai_mod.provider_pool("not-a-provider") is None


def test_cached_auth_variants():
    assert openai_mod._cached_auth({}, "s", 0, 100.0) is None
    assert openai_mod._cached_auth({"s": ("x",)}, "s", 10, 100.0) is None
    assert openai_mod._cached_auth({"s": (True, "bad")}, "s", 10, 100.0) is None
    assert openai_mod._cached_auth({"s": (True, 1.0)}, "s", 10, 100.0) is None
    assert openai_mod._cached_auth({"s": (True, 99.0)}, "s", 10, 100.0) is True


def test_evict_auth(monkeypatch):
    monkeypatch.setattr(byok_mod, "BYOK_AUTH_LIMIT", 1)
    store = {"a": 1, "b": 2}
    openai_mod._evict_auth(store)
    assert len(store) == 1


async def test_extract_api_key_variants(monkeypatch):
    body = b'{"api_key":"k1"}'
    cached = FakeRequest(headers={"content-type": "application/json", "content-length": str(len(body))}, _body=body)
    assert await openai_mod._extract_request_api_key(cached) == "k1"
    assert cached.body_calls == 0
    assert cached.consumed == []
    assert await openai_mod._extract_request_api_key(FakeRequest(headers={"content-type": "text/plain"})) is None
    assert await openai_mod._extract_request_api_key(FakeRequest(headers={"content-type": "application/json"}, body=b"")) is None
    bad = b"notjson"
    assert (
        await openai_mod._extract_request_api_key(FakeRequest(headers={"content-type": "application/json", "content-length": str(len(bad))}, body=bad)) is None
    )
    nokey = b'{"x":1}'
    assert (
        await openai_mod._extract_request_api_key(FakeRequest(headers={"content-type": "application/json", "content-length": str(len(nokey))}, body=nokey))
        is None
    )
    oversized = b'{"api_key":"' + b"x" * (byok_mod.BYOK_MAX_JSON_BODY + 1) + b'"}'
    assert await openai_mod._extract_request_api_key(FakeRequest(headers={"content-type": "application/json"}, _body=oversized)) is None


async def test_extract_api_key_body_raises():
    class RaisingReq(FakeRequest):
        async def body(self):
            raise RuntimeError("x")

    req = RaisingReq(headers={"content-type": "application/json", "content-length": "5"})
    assert await openai_mod._extract_request_api_key(req) is None
    with_bad_length = FakeRequest(headers={"content-type": "application/json", "content-length": "abc"})
    assert await openai_mod._extract_request_api_key(with_bad_length) is None
    assert with_bad_length.body_calls == 0


async def test_extract_api_key_http_exception(monkeypatch):
    req = FakeRequest(headers={"content-type": "application/json", "content-length": str(openai_mod.MAX_REQUEST_BODY + 1)})
    with pytest.raises(HTTPException):
        await openai_mod._extract_request_api_key(req)


async def test_close_pool_errors():
    acct = MagicMock()
    acct.sessions.close_all.side_effect = RuntimeError("x")
    sem = MagicMock()
    sem.locked.return_value = False
    acct.sem = sem
    acct.client.aclose = AsyncMock(side_effect=RuntimeError("x"))
    acct.label = "a"
    pool = SimpleNamespace(accounts=[acct])
    pool.flush = MagicMock(side_effect=RuntimeError("x"))
    await openai_mod._close_pool(pool)


async def test_close_pool_busy_defers(monkeypatch):
    acct = MagicMock()
    acct.sessions.close_all = MagicMock()
    sem = MagicMock()
    sem.locked.return_value = True
    acct.sem = sem
    acct.label = "a"
    pool = SimpleNamespace(accounts=[acct])
    pool.flush = None
    called = []
    monkeypatch.setattr(byok_mod, "_close_busy_client", AsyncMock(side_effect=lambda *a: called.append(a)))
    await openai_mod._close_pool(pool)
    await asyncio.sleep(0)
    assert called


async def test_close_busy_client_timeout():
    sem = MagicMock()

    async def fake_acquire():
        raise asyncio.TimeoutError()

    sem.acquire = fake_acquire
    acct = MagicMock()
    acct.label = "a"
    await openai_mod._close_busy_client(acct, sem)


async def test_close_busy_client_acquires():
    sem = asyncio.Semaphore(1)
    acct = MagicMock()
    acct.label = "a"
    acct.client.aclose = AsyncMock(side_effect=RuntimeError("x"))
    await openai_mod._close_busy_client(acct, sem)
    assert sem.locked() is False


async def test_byok_validate_cached(monkeypatch):
    monkeypatch.setattr(settings, "byok_auth_ttl", 100.0)
    token = "tok"
    stable = byok_mod._byok_stable_id(token)
    assert stable != openai_mod._token_stable_id(token)
    monkeypatch.setattr(app.state, "byok_auth", {"deepseek": {stable: (True, time.monotonic())}, "qwen": {}}, raising=False)
    client = MagicMock()
    client.check_auth = AsyncMock(return_value=False)
    assert await openai_mod._byok_validate("deepseek", token, client) is True
    client.check_auth.assert_not_awaited()


async def test_byok_validate_caches_verdict(monkeypatch):
    monkeypatch.setattr(settings, "byok_auth_ttl", 100.0)
    token = "tok"
    monkeypatch.setattr(app.state, "byok_auth", {"deepseek": {}, "qwen": {}}, raising=False)
    client = MagicMock()
    client.check_auth = AsyncMock(return_value=True)
    assert await openai_mod._byok_validate("deepseek", token, client) is True
    assert app.state.byok_auth["deepseek"][byok_mod._byok_stable_id(token)][0] is True
    assert await openai_mod._byok_validate("deepseek", token, client) is True
    client.check_auth.assert_awaited_once()


async def test_byok_validate_transport_failure_is_not_cached(monkeypatch):
    monkeypatch.setattr(settings, "byok_auth_ttl", 100.0)
    token = "tok"
    monkeypatch.setattr(app.state, "byok_auth", {"deepseek": {}, "qwen": {}}, raising=False)
    client = MagicMock()
    client.check_auth = AsyncMock(side_effect=RuntimeError("x"))
    assert await openai_mod._byok_validate("deepseek", token, client) is False
    assert byok_mod._byok_stable_id(token) not in app.state.byok_auth["deepseek"]


async def test_byok_validate_exception(monkeypatch):
    monkeypatch.setattr(settings, "byok_auth_ttl", 0.0)
    monkeypatch.setattr(app.state, "byok_auth", {"deepseek": {}, "qwen": {}}, raising=False)
    client = MagicMock()
    client.check_auth = AsyncMock(side_effect=RuntimeError("x"))
    assert await openai_mod._byok_validate("deepseek", "tok", client) is False


async def test_byok_pool_unknown_provider():
    with pytest.raises(HTTPException) as excinfo:
        await openai_mod._byok_pool("bogus", ["t"])
    assert excinfo.value.status_code == 400


async def test_byok_pool_cache_hit(monkeypatch):
    tokens = ["t1"]
    key = openai_mod._byok_cache_key(tokens)
    healthy = MagicMock()
    healthy.healthy = True
    monkeypatch.setattr(app.state, "byok_pools", {"deepseek": {key: healthy}, "qwen": {}}, raising=False)
    pool = await openai_mod._byok_pool("deepseek", tokens)
    assert pool is healthy


async def test_byok_pool_rebuilds_unhealthy(monkeypatch):
    monkeypatch.setattr(settings, "cache_enabled", False)
    monkeypatch.setattr(settings, "byok_auth_ttl", 0.0)
    tokens = ["t1"]
    key = openai_mod._byok_cache_key(tokens)
    old = MagicMock()
    old.healthy = False
    old.accounts = []
    old.flush = None
    monkeypatch.setattr(app.state, "byok_pools", {"deepseek": {key: old}, "qwen": {}}, raising=False)
    monkeypatch.setattr(app.state, "byok_locks", {"deepseek": asyncio.Lock(), "qwen": asyncio.Lock()}, raising=False)
    monkeypatch.setattr(app.state, "byok_auth", {"deepseek": {}, "qwen": {}}, raising=False)
    client = MagicMock()
    client.check_auth = AsyncMock(return_value=True)
    monkeypatch.setattr(byok_mod, "DeepSeekClient", MagicMock(return_value=client))
    pool = await openai_mod._byok_pool("deepseek", tokens)
    assert pool.accounts


async def test_byok_pool_evicts(monkeypatch):
    monkeypatch.setattr(settings, "cache_enabled", False)
    monkeypatch.setattr(settings, "byok_auth_ttl", 0.0)
    monkeypatch.setattr(byok_mod, "BYOK_POOL_LIMIT", 0)
    monkeypatch.setattr(app.state, "byok_pools", {"deepseek": {}, "qwen": {}}, raising=False)
    monkeypatch.setattr(app.state, "byok_locks", {"deepseek": asyncio.Lock(), "qwen": asyncio.Lock()}, raising=False)
    monkeypatch.setattr(app.state, "byok_auth", {"deepseek": {}, "qwen": {}}, raising=False)
    client = MagicMock()
    client.check_auth = AsyncMock(return_value=True)
    client.aclose = AsyncMock()
    monkeypatch.setattr(byok_mod, "DeepSeekClient", MagicMock(return_value=client))
    pool = await openai_mod._byok_pool("deepseek", ["t1"])
    assert pool.accounts


async def test_byok_pool_qwen_invalid(monkeypatch):
    monkeypatch.setattr(settings, "cache_enabled", False)
    monkeypatch.setattr(settings, "byok_auth_ttl", 0.0)
    monkeypatch.setattr(app.state, "byok_pools", {"deepseek": {}, "qwen": {}}, raising=False)
    monkeypatch.setattr(app.state, "byok_locks", {"deepseek": asyncio.Lock(), "qwen": asyncio.Lock()}, raising=False)
    monkeypatch.setattr(app.state, "byok_auth", {"deepseek": {}, "qwen": {}}, raising=False)
    client = MagicMock()
    client.check_auth = AsyncMock(return_value=False)
    client.aclose = AsyncMock()
    monkeypatch.setattr(byok_mod, "QwenClient", MagicMock(return_value=client))
    with pytest.raises(HTTPException) as excinfo:
        await openai_mod._byok_pool("qwen", ["t1"])
    assert excinfo.value.status_code == 401


async def test_dispatch_chat_byok(monkeypatch):
    monkeypatch.setattr(app.state, "byok", True, raising=False)
    monkeypatch.setattr(chats_mod, "_byok_pool_for", AsyncMock(return_value=MagicMock()))
    monkeypatch.setattr(chats_mod, "_chat_completions_deepseek", AsyncMock(return_value={"ok": 1}))
    monkeypatch.setattr(chats_mod, "_chat_completions_qwen", AsyncMock(return_value={"q": 1}))
    out_ds = await chats_mod._dispatch_chat(SimpleNamespace(model="deepseek-v4.1-flash"), SimpleNamespace())
    assert out_ds == {"ok": 1}
    out_qw = await chats_mod._dispatch_chat(SimpleNamespace(model="qwen3.8-max"), SimpleNamespace())
    assert out_qw == {"q": 1}


def test_completion_prompts():
    assert openai_mod._completion_prompts("hi") == ["hi"]
    assert openai_mod._completion_prompts(["a", ["b", "c"]]) == ["a", "b c"]
    with pytest.raises(HTTPException):
        openai_mod._completion_prompts(42)
    with pytest.raises(HTTPException):
        openai_mod._completion_prompts([])
    with pytest.raises(HTTPException):
        openai_mod._completion_prompts([42])


def test_completion_chat_request():
    req = openai_mod.CompletionRequest(model="deepseek-v4.1-flash", prompt="x", session_id="s1")
    chat = openai_mod._completion_chat_request(req, "hi", False, 1)
    assert chat.messages[0].content == "hi"
    assert chat.stream is False
    assert chat.session_id == "s1"
    assert openai_mod._completion_chat_request(req, "hi", False, 2).session_id is None


def test_legacy_choice_from_chat():
    choice = openai_mod._legacy_choice_from_chat({"message": {"content": "hi"}, "finish_reason": "stop"}, 0)
    assert choice["text"] == "hi"
    assert choice["finish_reason"] == "stop"
    assert openai_mod._legacy_choice_from_chat({"message": "bad"}, 1)["text"] == ""
    assert openai_mod._legacy_choice_from_chat({}, 2)["finish_reason"] == "stop"


def test_translate_chat_chunk_to_completion():
    chunk = {
        "id": "1",
        "choices": [{"index": 0, "delta": {"content": "hi"}, "finish_reason": None}],
        "usage": {"total_tokens": 1},
        "error": {"message": "e"},
    }
    out = openai_mod._translate_chat_chunk_to_completion(chunk)
    assert out["choices"][0]["text"] == "hi"
    assert out["usage"] == {"total_tokens": 1}
    assert out["error"]["message"] == "e"
    out2 = openai_mod._translate_chat_chunk_to_completion({"error": "plain"})
    assert out2["error"]["message"] == "plain"
    out3 = openai_mod._translate_chat_chunk_to_completion({"choices": [{"delta": 5}]})
    assert out3["choices"][0]["text"] == ""


async def test_translate_completion_stream():
    async def gen():
        yield "not data\n"
        yield "data: [DONE]\n\n"
        yield "data: {bad\n\n"
        yield 'data: {"id":"1","choices":[{"delta":{"content":"hi"}}]}\n\n'

    out = [line async for line in openai_mod._translate_completion_stream(gen())]
    assert any("not data" in line for line in out)
    assert any('"text": "hi"' in line for line in out)


async def _sse_body():
    yield "data: {}\n\n"


async def test_completions_stream_emits_done():
    async def fake_dispatch(chat_req):
        return StreamingResponse(_sse_body(), media_type="text/event-stream")

    req = openai_mod.CompletionRequest(model="deepseek-v4.1-flash", prompt="x")
    out = [line async for line in openai_mod._completions_stream(req, ["x"], fake_dispatch)]
    assert out[-1] == "data: [DONE]\n\n"


async def test_completions_stream_translates_chat_chunks():
    seen = []

    async def fake_dispatch(chat_req):
        seen.append(chat_req)

        async def body():
            yield 'data: {"id":"1","choices":[{"delta":{"content":"hi"}}]}\n\n'
            yield "data: [DONE]\n\n"

        return StreamingResponse(body(), media_type="text/event-stream")

    req = openai_mod.CompletionRequest(model="deepseek-v4.1-flash", prompt=["a", "b"], session_id="s1")
    out = [line async for line in openai_mod._completions_stream(req, ["a", "b"], fake_dispatch)]
    assert out[-1] == "data: [DONE]\n\n"
    assert any('"text": "hi"' in line for line in out)
    assert all(chat_req.session_id is None for chat_req in seen)


def test_completions_endpoint_non_stream():
    pool, _ = make_pool()
    app.state.pool = pool
    client = TestClient(app)
    resp = client.post("/v1/completions", json={"model": "deepseek-v4.1-flash", "prompt": "hi"})
    client.close()
    assert resp.status_code == 200
    assert resp.json()["choices"][0]["text"] == "Hi"


def test_completions_endpoint_stream():
    pool, _ = make_pool()
    app.state.pool = pool
    client = TestClient(app)
    resp = client.post("/v1/completions", json={"model": "deepseek-v4.1-flash", "prompt": "hi", "stream": True})
    client.close()
    assert resp.status_code == 200
    assert resp.text.rstrip().endswith("data: [DONE]")


def test_completions_endpoint_batch_drops_session():
    pool, acct = make_pool(FakeAccount([OK_SSE, OK_SSE]))
    app.state.pool = pool
    client = TestClient(app)
    resp = client.post("/v1/completions", json={"model": "deepseek-v4.1-flash", "prompt": ["one", "two"], "session_id": "s9"})
    client.close()
    assert resp.status_code == 200
    assert len(resp.json()["choices"]) == 2
    prompts = [call.kwargs["prompt"] for call in acct.client.completion.await_args_list]
    assert prompts == ["one", "two"]
    assert all(call.kwargs["chat_session_id"] == "c1" for call in acct.client.completion.await_args_list)


def test_completions_endpoint_single_prompt_keeps_session():
    pool, acct = make_pool()
    app.state.pool = pool
    client = TestClient(app)
    resp = client.post("/v1/completions", json={"model": "deepseek-v4.1-flash", "prompt": "one", "session_id": "s9"})
    client.close()
    assert resp.status_code == 200
    assert acct.client.completion.await_args.kwargs["prompt"] == "one"
    assert acct.sessions.obtain.await_args.args[0] == "s9"


def test_completions_endpoint_stream_batch_drops_session():
    pool, acct = make_pool(FakeAccount([OK_SSE, OK_SSE]))
    app.state.pool = pool
    client = TestClient(app)
    resp = client.post("/v1/completions", json={"model": "deepseek-v4.1-flash", "prompt": ["one", "two"], "session_id": "s9", "stream": True})
    client.close()
    assert resp.status_code == 200
    assert resp.text.rstrip().endswith("data: [DONE]")
    assert acct.sessions.obtain.await_args.args[0] is None


def test_embeddings_and_moderations_not_supported():
    client = TestClient(app)
    assert client.post("/v1/embeddings").status_code == 501
    assert client.post("/v1/moderations").status_code == 501
    client.close()


async def test_chat_dispatcher_picks_qwen():
    app.state.qwen_models = [{"id": "qwen3.8-max", "name": "Q", "owned_by": "qwen", "model_type": "chat"}]
    openai_mod._MODEL_CACHE["key"] = None
    assert await openai_mod._chat_dispatcher("qwen3.8-max", SimpleNamespace()) is openai_mod._chat_completions_qwen


async def test_chat_dispatcher_picks_deepseek():
    assert await openai_mod._chat_dispatcher("deepseek-v4.1-flash", SimpleNamespace()) is openai_mod._chat_completions_deepseek


async def test_chat_dispatcher_byok_binds_pool(monkeypatch):
    pool = MagicMock()
    monkeypatch.setattr(chats_mod, "_byok_mode", lambda: True)
    monkeypatch.setattr(chats_mod, "_byok_pool_for", AsyncMock(return_value=pool))
    call = await openai_mod._chat_dispatcher("deepseek-v4.1-flash", SimpleNamespace())
    assert call.func is openai_mod._chat_completions_deepseek
    assert call.keywords == {"pool": pool}


def test_create_response_non_stream_and_crud():
    pool, _ = make_pool()
    app.state.pool = pool
    app.state.responses_store = None
    client = TestClient(app)
    created = client.post("/v1/responses", json={"model": "deepseek-v4.1-flash", "input": "hi"})
    client.close()
    assert created.status_code == 200
    response_id = created.json()["id"]

    client = TestClient(app)
    got = client.get(f"/v1/responses/{response_id}")
    items = client.get(f"/v1/responses/{response_id}/input_items")
    deleted = client.delete(f"/v1/responses/{response_id}")
    missing = client.get("/v1/responses/nope")
    missing_delete = client.delete("/v1/responses/nope")
    client.close()
    assert got.status_code == 200
    assert items.status_code == 200
    assert deleted.status_code == 200
    assert missing.status_code == 404
    assert missing_delete.status_code == 404


def test_cancel_response_in_progress():
    app.state.responses_store = None
    store = openai_mod._responses_store()
    store.set("resp_x", {"public": {"status": "in_progress"}})
    client = TestClient(app)
    resp = client.post("/v1/responses/resp_x/cancel")
    client.close()
    assert resp.status_code == 200
    assert resp.json()["status"] == "cancelled"


def test_input_items_fallback():
    app.state.responses_store = None
    store = openai_mod._responses_store()
    store.set("resp_y", {"public": {"input": [{"role": "user", "content": "hi"}, {"role": "user", "content": "again"}]}})
    client = TestClient(app)
    first = client.get("/v1/responses/resp_y/input_items")
    second = client.get("/v1/responses/resp_y/input_items")
    ascending = client.get("/v1/responses/resp_y/input_items?order=asc")
    client.close()
    assert first.status_code == 200
    data = first.json()["data"]
    assert data
    assert first.json()["data"] == second.json()["data"]
    assert first.json()["first_id"] == data[0]["id"]
    assert first.json()["last_id"] == data[-1]["id"]
    assert first.json()["has_more"] is False
    assert len({item["id"] for item in data}) == len(data)
    assert ascending.json()["data"] == list(reversed(data))
    assert ascending.json()["first_id"] == ascending.json()["data"][0]["id"]
    assert [item["content"][0]["text"] for item in data] == ["again", "hi"]


def test_image_generations_endpoint_no_pool():
    app.state.qwen_pool = None
    client = TestClient(app)
    resp = client.post("/v1/images/generations", json={"prompt": "x"})
    client.close()
    assert resp.status_code == 503


async def test_image_pool_none():
    app.state.qwen_pool = None
    with pytest.raises(HTTPException) as excinfo:
        await openai_mod._image_pool(SimpleNamespace())
    assert excinfo.value.status_code == 503


async def test_image_generations_busy(monkeypatch):
    pool = MagicMock()
    account = MagicMock()
    account.sem = asyncio.Semaphore(1)
    pool.acquire = AsyncMock(return_value=(account, None))

    async def fake_collect(**kwargs):
        raise AccountPoolBusy()

    monkeypatch.setattr(openai_mod.qwen_api, "collect_image", fake_collect)
    with pytest.raises(HTTPException) as excinfo:
        await openai_mod._image_generations(openai_mod.ImageGenerationRequest(prompt="x"), pool)
    assert excinfo.value.status_code == 429


async def test_image_generations_no_data(monkeypatch):
    pool = MagicMock()
    account = MagicMock()
    account.sem = asyncio.Semaphore(1)
    pool.acquire = AsyncMock(return_value=(account, None))

    async def fake_collect(**kwargs):
        return {"image_urls": [], "session_id": None}

    monkeypatch.setattr(openai_mod.qwen_api, "collect_image", fake_collect)
    with pytest.raises(HTTPException) as excinfo:
        await openai_mod._image_generations(openai_mod.ImageGenerationRequest(prompt="x"), pool)
    assert excinfo.value.status_code == 502


async def test_image_generations_b64(monkeypatch):
    pool = MagicMock()
    account = MagicMock()
    account.sem = asyncio.Semaphore(1)
    pool.acquire = AsyncMock(return_value=(account, None))

    async def fake_collect(**kwargs):
        return {"image_urls": ["http://images.test/1.png"], "session_id": "s1", "usage": {"prompt_tokens": 1}}

    monkeypatch.setattr(openai_mod.qwen_api, "collect_image", fake_collect)
    monkeypatch.setattr(images_mod, "_host_is_public", lambda host: True)
    hc = FakeImageClient({"http://images.test/1.png": [FakeImageResponse(chunks=[b"ab", b"c"])]})
    monkeypatch.setattr(app.state, "http_client", hc, raising=False)
    out = await openai_mod._image_generations(openai_mod.ImageGenerationRequest(prompt="x", response_format="b64_json"), pool)
    assert out["data"][0]["b64_json"] == base64.b64encode(b"abc").decode()
    assert "url" not in out["data"][0]
    assert out["usage"] == {"prompt_tokens": 1}
    assert out["session_id"] == "s1"
    assert hc.calls == [("GET", "http://images.test/1.png", images_mod.IMAGE_FETCH_TIMEOUT_SEC, False)]


async def test_image_generations_fetch_failures(monkeypatch):
    pool = MagicMock()
    account = MagicMock()
    account.sem = asyncio.Semaphore(1)
    pool.acquire = AsyncMock(return_value=(account, None))

    async def fake_collect(**kwargs):
        return {"image_urls": ["http://images.test/1.png", "http://images.test/2.png"], "session_id": None}

    monkeypatch.setattr(openai_mod.qwen_api, "collect_image", fake_collect)
    monkeypatch.setattr(images_mod, "_host_is_public", lambda host: True)
    hc = FakeImageClient(
        {
            "http://images.test/1.png": [FakeImageResponse(status_code=500)],
            "http://images.test/2.png": [RuntimeError("net")],
        }
    )
    monkeypatch.setattr(app.state, "http_client", hc, raising=False)
    with pytest.raises(HTTPException) as excinfo:
        await openai_mod._image_generations(openai_mod.ImageGenerationRequest(prompt="x", response_format="b64_json"), pool)
    assert excinfo.value.status_code == 502
    assert excinfo.value.detail == "image download failed"


async def test_image_generations_refuses_non_public_host(monkeypatch):
    pool = MagicMock()
    account = MagicMock()
    account.sem = asyncio.Semaphore(1)
    pool.acquire = AsyncMock(return_value=(account, None))

    async def fake_collect(**kwargs):
        return {"image_urls": ["http://169.254.169.254/latest/meta-data/iam/security-credentials/"], "session_id": None}

    monkeypatch.setattr(openai_mod.qwen_api, "collect_image", fake_collect)
    hc = FakeImageClient()
    monkeypatch.setattr(app.state, "http_client", hc, raising=False)
    with pytest.raises(HTTPException) as excinfo:
        await openai_mod._image_generations(openai_mod.ImageGenerationRequest(prompt="x", response_format="b64_json"), pool)
    assert excinfo.value.status_code == 502
    assert hc.calls == []
    assert "169.254.169.254" not in excinfo.value.detail


async def test_image_generations_refuses_non_http_scheme(monkeypatch):
    pool = MagicMock()
    account = MagicMock()
    account.sem = asyncio.Semaphore(1)
    pool.acquire = AsyncMock(return_value=(account, None))

    async def fake_collect(**kwargs):
        return {"image_urls": ["file:///etc/passwd"], "session_id": None}

    monkeypatch.setattr(openai_mod.qwen_api, "collect_image", fake_collect)
    hc = FakeImageClient()
    monkeypatch.setattr(app.state, "http_client", hc, raising=False)
    with pytest.raises(HTTPException) as excinfo:
        await openai_mod._image_generations(openai_mod.ImageGenerationRequest(prompt="x", response_format="b64_json"), pool)
    assert excinfo.value.status_code == 502
    assert hc.calls == []


async def test_image_generations_follows_public_redirects(monkeypatch):
    pool = MagicMock()
    account = MagicMock()
    account.sem = asyncio.Semaphore(1)
    pool.acquire = AsyncMock(return_value=(account, None))

    async def fake_collect(**kwargs):
        return {"image_urls": ["http://images.test/1.png"], "session_id": None}

    monkeypatch.setattr(openai_mod.qwen_api, "collect_image", fake_collect)
    monkeypatch.setattr(images_mod, "_host_is_public", lambda host: host in ("images.test", "cdn.test"))
    hc = FakeImageClient(
        {
            "http://images.test/1.png": [FakeImageResponse(status_code=302, headers={"location": "http://cdn.test/2.png"})],
            "http://cdn.test/2.png": [FakeImageResponse(chunks=[b"abc"])],
        }
    )
    monkeypatch.setattr(app.state, "http_client", hc, raising=False)
    out = await openai_mod._image_generations(openai_mod.ImageGenerationRequest(prompt="x", response_format="b64_json"), pool)
    assert out["data"][0]["b64_json"] == base64.b64encode(b"abc").decode()
    assert [call[:2] for call in hc.calls] == [("GET", "http://images.test/1.png"), ("GET", "http://cdn.test/2.png")]
    assert all(call[3] is False for call in hc.calls)


async def test_image_generations_revalidates_redirect_target(monkeypatch):
    pool = MagicMock()
    account = MagicMock()
    account.sem = asyncio.Semaphore(1)
    pool.acquire = AsyncMock(return_value=(account, None))

    async def fake_collect(**kwargs):
        return {"image_urls": ["http://images.test/1.png"], "session_id": None}

    monkeypatch.setattr(openai_mod.qwen_api, "collect_image", fake_collect)
    monkeypatch.setattr(images_mod, "_host_is_public", lambda host: host == "images.test")
    hc = FakeImageClient(
        {
            "http://images.test/1.png": [FakeImageResponse(status_code=302, headers={"location": "http://169.254.169.254/latest/"})],
        }
    )
    monkeypatch.setattr(app.state, "http_client", hc, raising=False)
    with pytest.raises(HTTPException) as excinfo:
        await openai_mod._image_generations(openai_mod.ImageGenerationRequest(prompt="x", response_format="b64_json"), pool)
    assert excinfo.value.status_code == 502
    assert [call[1] for call in hc.calls] == ["http://images.test/1.png"]


async def test_image_generations_redirect_limit(monkeypatch):
    pool = MagicMock()
    account = MagicMock()
    account.sem = asyncio.Semaphore(1)
    pool.acquire = AsyncMock(return_value=(account, None))

    async def fake_collect(**kwargs):
        return {"image_urls": ["http://images.test/1.png"], "session_id": None}

    monkeypatch.setattr(openai_mod.qwen_api, "collect_image", fake_collect)
    monkeypatch.setattr(images_mod, "_host_is_public", lambda host: True)
    hops = range(1, images_mod.IMAGE_FETCH_REDIRECTS + 2)
    script = {f"http://images.test/{hop}.png": [FakeImageResponse(status_code=302, headers={"location": f"http://images.test/{hop + 1}.png"})] for hop in hops}
    hc = FakeImageClient(script)
    monkeypatch.setattr(app.state, "http_client", hc, raising=False)
    with pytest.raises(HTTPException) as excinfo:
        await openai_mod._image_generations(openai_mod.ImageGenerationRequest(prompt="x", response_format="b64_json"), pool)
    assert excinfo.value.status_code == 502
    assert [call[1] for call in hc.calls] == [f"http://images.test/{hop}.png" for hop in hops]


async def test_image_generations_download_size_cap(monkeypatch):
    pool = MagicMock()
    account = MagicMock()
    account.sem = asyncio.Semaphore(1)
    pool.acquire = AsyncMock(return_value=(account, None))

    async def fake_collect(**kwargs):
        return {"image_urls": ["http://images.test/1.png"], "session_id": None}

    monkeypatch.setattr(openai_mod.qwen_api, "collect_image", fake_collect)
    monkeypatch.setattr(images_mod, "_host_is_public", lambda host: True)
    monkeypatch.setattr(images_mod, "MAX_FILE_SIZE", 8)
    chunks = [b"x" * 4, b"y" * 4, b"z" * 100]
    hc = FakeImageClient({"http://images.test/1.png": [FakeImageResponse(chunks=chunks)]})
    monkeypatch.setattr(app.state, "http_client", hc, raising=False)
    with pytest.raises(HTTPException) as excinfo:
        await openai_mod._image_generations(openai_mod.ImageGenerationRequest(prompt="x", response_format="b64_json"), pool)
    assert excinfo.value.status_code == 502
    assert hc.calls


def test_host_is_public_refuses_internal_addresses():
    internal = (
        "127.0.0.1",
        "localhost",
        "169.254.169.254",
        "10.1.2.3",
        "192.168.0.1",
        "172.16.0.1",
        "0.0.0.0",
        "::1",
        "fd00::1",
        "224.0.0.1",
        "not-a-host.invalid",
    )
    for host in internal:
        assert images_mod._host_is_public(host) is False, host


async def test_b64encode_variants():
    assert await openai_mod._b64encode(b"abc") == "YWJj"
    big = b"x" * (openai_mod._ASYNC_B64_THRESHOLD + 1)
    assert await openai_mod._b64encode(big)


async def test_image_markdown():
    md = await openai_mod._image_markdown(b"abc", "image/png")
    assert md.startswith("![image](data:image/png;base64,")
    md2 = await openai_mod._image_markdown(b"abc", "")
    assert "image/png" in md2


async def test_read_upload():
    file = SimpleNamespace(read=AsyncMock(return_value=b"abc"), content_type="text/plain; charset=utf-8")
    data, content_type = await openai_mod._read_upload(file)
    assert data == b"abc"
    assert content_type == "text/plain"


async def test_read_upload_too_large():
    file = SimpleNamespace(read=AsyncMock(return_value=b"x" * (openai_mod.MAX_FILE_SIZE + 1)), content_type="text/plain")
    with pytest.raises(HTTPException) as excinfo:
        await openai_mod._read_upload(file)
    assert excinfo.value.status_code == 413


async def test_read_upload_stops_reading_at_the_cap(monkeypatch):
    monkeypatch.setattr(images_mod, "MAX_FILE_SIZE", 100)
    chunks = [b"x" * 101] + [b"y" * 101] * 100
    file = SimpleNamespace(read=AsyncMock(side_effect=chunks), content_type="text/plain")
    with pytest.raises(HTTPException) as excinfo:
        await openai_mod._read_upload(file)
    assert excinfo.value.status_code == 413
    assert file.read.await_count == 1


def test_image_edits_rejects_out_of_range_n(monkeypatch):
    pool = MagicMock()
    account = MagicMock()
    account.sem = asyncio.Semaphore(1)
    pool.acquire = AsyncMock(return_value=(account, None))
    app.state.qwen_pool = pool
    collect = AsyncMock(return_value={"image_urls": ["http://x/1.png"], "session_id": None})
    monkeypatch.setattr(openai_mod.qwen_api, "collect_image", collect)
    client = TestClient(app)
    for count in ("0", "99", "-1"):
        resp = client.post(
            "/v1/images/edits",
            files={"image": ("a.png", b"data", "image/png")},
            data={"prompt": "p", "n": count},
        )
        assert resp.status_code == 400, count
    assert "n" in resp.json()["error"]["message"]
    ok = client.post(
        "/v1/images/edits",
        files={"image": ("a.png", b"data", "image/png")},
        data={"prompt": "p", "n": "4"},
    )
    client.close()
    assert ok.status_code == 200
    collect.assert_awaited()


def test_image_edit_req():
    assert openai_mod._image_edit_req("p", "img", None) == "p\nimg"
    assert openai_mod._image_edit_req("  ", "img", "mask") == "img\nmask"


def test_image_edits_endpoint(monkeypatch):
    pool = MagicMock()
    account = MagicMock()
    account.sem = asyncio.Semaphore(1)
    pool.acquire = AsyncMock(return_value=(account, None))
    app.state.qwen_pool = pool

    async def fake_collect(**kwargs):
        return {"image_urls": ["http://x/1.png"], "session_id": None}

    monkeypatch.setattr(openai_mod.qwen_api, "collect_image", fake_collect)
    client = TestClient(app)
    resp = client.post(
        "/v1/images/edits",
        files={"image": ("a.png", b"data", "image/png")},
        data={"prompt": "p"},
    )
    client.close()
    assert resp.status_code == 200


def test_image_variations_endpoint(monkeypatch):
    pool = MagicMock()
    account = MagicMock()
    account.sem = asyncio.Semaphore(1)
    pool.acquire = AsyncMock(return_value=(account, None))
    app.state.qwen_pool = pool

    async def fake_collect(**kwargs):
        return {"image_urls": ["http://x/1.png"], "session_id": None}

    monkeypatch.setattr(openai_mod.qwen_api, "collect_image", fake_collect)
    client = TestClient(app)
    resp = client.post("/v1/images/variations", files={"image": ("a.png", b"data", "image/png")})
    client.close()
    assert resp.status_code == 200


def test_materialize_tools_functions():
    req = SimpleNamespace(
        tools=None,
        tool_choice=None,
        functions=[{"name": "f", "description": "d", "parameters": {"type": "object"}}],
        function_call=None,
    )
    tools, _tool_choice = openai_mod._materialize_tools(req)
    assert tools[0]["function"]["name"] == "f"
    assert tools[0]["function"]["description"] == "d"

    req2 = SimpleNamespace(
        tools=[{"type": "function", "function": {"name": "existing"}}],
        tool_choice=None,
        functions=[42, {"name": "g"}],
        function_call=None,
    )
    tools2, _ = openai_mod._materialize_tools(req2)
    names = [t["function"]["name"] for t in tools2]
    assert "existing" in names
    assert "g" in names


def test_materialize_tools_function_call():
    req = SimpleNamespace(tools=None, tool_choice=None, functions=None, function_call="auto")
    _, tc = openai_mod._materialize_tools(req)
    assert tc == "auto"
    req = SimpleNamespace(tools=None, tool_choice=None, functions=None, function_call="my_func")
    _, tc = openai_mod._materialize_tools(req)
    assert tc == {"type": "function", "function": {"name": "my_func"}}
    req = SimpleNamespace(tools=None, tool_choice=None, functions=None, function_call={"name": "x"})
    _, tc = openai_mod._materialize_tools(req)
    assert tc["function"]["name"] == "x"
    req = SimpleNamespace(tools=None, tool_choice="none", functions=None, function_call="auto")
    _, tc = openai_mod._materialize_tools(req)
    assert tc == "none"


def test_bounded_choices():
    assert openai_mod._bounded_choices(None) == 1
    assert openai_mod._bounded_choices(1) == 1
    assert openai_mod._bounded_choices(3) == 3
    assert openai_mod._bounded_choices(100) == openai_mod.MAX_STREAM_CHOICES


def test_max_calls():
    assert openai_mod._max_calls(False) == 1
    assert openai_mod._max_calls(True) is None
    assert openai_mod._max_calls(None) is None


def test_apply_stop():
    assert openai_mod._apply_stop("hello END world", "END") == "hello "
    assert openai_mod._apply_stop("hello", None) == "hello"
    assert openai_mod._apply_stop("ab STOP cd", ["STOP", "cd"]) == "ab "


def test_apply_limits():
    text, finish = openai_mod._apply_limits("hello END world", None, "END")
    assert text == "hello "
    assert finish == "stop"
    text, finish = openai_mod._apply_limits("alpha beta gamma delta", 1, None)
    assert text == "alpha"
    assert finish == "length"


def test_include_usage_non_dict():
    assert not openai_mod._include_usage(SimpleNamespace(stream_options="x"))


def test_deepseek_usage_provider():
    out = openai_mod._deepseek_usage(10, "hello", {"prompt_tokens": 4})
    assert out["prompt_tokens"] == 4
    out2 = openai_mod._deepseek_usage(0, "hello", None, "world")
    assert out2["completion_tokens"] > 0


def test_usage_with_details():
    out = openai_mod._usage_with_details({"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3})
    assert out["prompt_tokens_details"] == {"cached_tokens": 0}
    assert "reasoning_tokens" in out["completion_tokens_details"]


def test_sse_helpers():
    line = openai_mod._sse({"a": 1})
    assert line.startswith("data: ")
    assert openai_mod._chunk_id_from_line("not data") is None
    assert openai_mod._chunk_id_from_line("data: [DONE]") is None
    assert openai_mod._chunk_id_from_line("data: {bad") is None
    assert openai_mod._chunk_id_from_line('data: {"id":"x"}') == "x"
    assert openai_mod._chunk_id_from_line('data: {"id":5}') is None


async def test_close_generator():
    class NoClose:
        pass

    await openai_mod._close_generator(NoClose())

    class BadClose:
        async def aclose(self):
            raise RuntimeError("x")

    await openai_mod._close_generator(BadClose())


def test_build_assistant_message():
    msg, finish = openai_mod._build_assistant_message("plain", "why", False, None)
    assert finish == "stop"
    assert msg["reasoning_content"] == "why"
    _msg2, finish2 = openai_mod._build_assistant_message("plain", None, True, {})
    assert finish2 == "stop"


def test_build_limited_message():
    _msg, finish = openai_mod._build_limited_message("hello", None, False, None, None, None, None, "FINISHED")
    assert finish == "stop"
    _msg2, finish2 = openai_mod._build_limited_message("alpha beta gamma delta", None, False, None, 1, None, None, "FINISHED")
    assert finish2 == "length"
    _msg3, finish3 = openai_mod._build_limited_message("x", None, False, None, None, "END", None, "FINISHED")
    assert finish3 == "stop"


def test_build_completion_response():
    out = openai_mod._build_completion_response("m", {"role": "assistant", "content": "hi"}, "stop", {}, None)
    assert out["model"] == "m"
    assert out["choices"][0]["message"]["content"] == "hi"


def test_is_retryable_hint_and_fake(monkeypatch):
    rec = SimpleNamespace(hint_error={"finish_reason": "server_busy"})
    assert openai_mod._is_retryable_hint(rec)
    rec2 = SimpleNamespace(hint_error={"message": 5})
    assert not openai_mod._is_fake_context_hint(rec2)


def test_compact_and_error_text():
    assert openai_mod._compact_error_text(None) == ""
    assert openai_mod._compact_error_text("A b!") == "ab"
    assert openai_mod._error_text("x") == "x"
    assert openai_mod._error_text({"a": 1})
    assert openai_mod._error_text({1, 2})


def test_is_message_too_frequent_hint():
    rec = SimpleNamespace(hint_error=None)
    assert not openai_mod._is_message_too_frequent_hint(rec)
    rec2 = SimpleNamespace(hint_error={"message": "message_too_frequent"})
    assert openai_mod._is_message_too_frequent_hint(rec2)


def test_incomplete_message_helpers():
    rec = SimpleNamespace(hint_error={"message": "custom"})
    assert openai_mod._incomplete_message(rec) == "custom"
    rec2 = SimpleNamespace(hint_error={})
    assert openai_mod._incomplete_message(rec2) == openai_mod.RESPONSE_INCOMPLETE_MESSAGE
    assert openai_mod._incomplete_error_body("m")["error"]["finish_reason"] == openai_mod.RESPONSE_INCOMPLETE


def test_retry_delay():
    assert openai_mod._retry_delay(1) >= 0


def test_unknown_v1_route():
    client = TestClient(app)
    resp = client.get("/v1/nope")
    client.close()
    assert resp.status_code == 404


@pytest.mark.usefixtures("reset_app_state")
def test_lifespan_qwen_invalid_skipped():
    from danyapi.qwen.client import QwenClient as QC

    with (
        patch.object(settings, "deepseek_tokens", []),
        patch.object(settings, "qwen_tokens", ["bad"]),
        patch.object(QC, "check_auth", new=AsyncMock(return_value=False)),
    ):
        with pytest.raises(RuntimeError):
            with TestClient(app):
                pass


def test_health_byok(monkeypatch):
    monkeypatch.setattr(app.state, "byok", True, raising=False)
    monkeypatch.setattr(app.state, "byok_pools", {"deepseek": {"a": 1}, "qwen": {}}, raising=False)
    client = TestClient(app)
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/health", headers={"x-api-key": "nope"}).json() == {"status": "ok"}
    detail = client.get("/health", headers=ADMIN_HEADERS).json()
    client.close()
    assert detail["status"] == "ok"
    assert detail["byok"] is True
    assert detail["byok_pools"]["deepseek"] == 1
    assert detail["qwen_stats"] is not None


def test_health_without_admin_token_is_minimal(monkeypatch):
    monkeypatch.setattr(app.state, "byok", False, raising=False)
    monkeypatch.setattr(app.state, "usage", None, raising=False)
    client = TestClient(app)
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/health", headers={"authorization": "Bearer nope"}).json() == {"status": "ok"}
    assert client.get("/health", headers={"x-api-key": "nope"}).json() == {"status": "ok"}
    detail = client.get("/health", headers=ADMIN_HEADERS).json()
    client.close()
    assert detail["status"] == "ok"
    assert detail["usage"] is None
    assert "byok" not in detail
    expected = set(openai_mod.BYOK_PROVIDERS) | {f"{provider}_stats" for provider in openai_mod.BYOK_PROVIDERS}
    assert set(detail) >= expected


def test_usage_endpoint_is_gated(monkeypatch):
    tracker = MagicMock()
    tracker.snapshot.return_value = {
        "totals": {"requests": 3},
        "by_model": {"m": {"requests": 3}},
        "by_provider": {"deepseek": {"requests": 3}},
        "by_user": {"u": {"requests": 3}},
        "recent": [{"model": "m"}],
    }
    monkeypatch.setattr(app.state, "usage", tracker, raising=False)
    client = TestClient(app)
    public = client.get("/v1/usage")
    admin = client.get("/v1/usage", headers=ADMIN_HEADERS)
    client.close()
    assert public.status_code == 200
    assert public.json() == {"totals": {"requests": 3}, "by_model": {"m": {"requests": 3}}}
    assert admin.status_code == 200
    assert "by_provider" in admin.json()
    assert "recent" in admin.json()


def test_usage_endpoint_disabled(monkeypatch):
    monkeypatch.setattr(app.state, "usage", None, raising=False)
    client = TestClient(app)
    assert client.get("/v1/usage").status_code == 404
    client.close()


def test_build_limited_message_tool_mode():

    content = "some text"
    _msg, finish = openai_mod._build_limited_message(content, None, True, {}, None, None, None, "FINISHED")
    assert finish in ("stop", "length", "tool_calls")


def test_split_data_uri_empty_payload():
    with pytest.raises(HTTPException) as excinfo:
        openai_mod._split_data_uri("data:image/png;base64,")
    assert excinfo.value.status_code == 400


def test_parse_image_size():
    assert openai_mod._parse_image_size(None) is None
    assert openai_mod._parse_image_size("  ") is None
    assert openai_mod._parse_image_size("16x16") == (16, 16)
    with pytest.raises(HTTPException):
        openai_mod._parse_image_size("bad")
    with pytest.raises(HTTPException):
        openai_mod._parse_image_size("1x1")


def test_add_tokens_requires_admin_token():
    client = TestClient(app)
    resp = client.post("/v1/tokens", json={"deepseek_tokens": ["x"]})
    client.close()
    assert resp.status_code == 401
    assert resp.json()["error"]["type"] == "authentication_error"


def test_add_tokens_rejects_wrong_admin_token():
    client = TestClient(app)
    resp = client.post("/v1/tokens", headers={"authorization": "Bearer nope"}, json={"deepseek_tokens": ["x"]})
    client.close()
    assert resp.status_code == 401


def test_add_tokens_disabled_without_configured_token(monkeypatch):
    monkeypatch.setattr(settings, "admin_token", "", raising=False)
    client = TestClient(app)
    resp = client.post("/v1/tokens", headers=ADMIN_HEADERS, json={"deepseek_tokens": ["x"]})
    client.close()
    assert resp.status_code == 404


def test_add_tokens_absent_in_byok_mode(monkeypatch):
    monkeypatch.setattr(app.state, "byok", True, raising=False)
    client = TestClient(app)
    resp = client.post("/v1/tokens", headers=ADMIN_HEADERS, json={"deepseek_tokens": ["x"]})
    client.close()
    assert resp.status_code == 404


def test_add_tokens_accepts_x_api_key(monkeypatch):
    monkeypatch.setattr(settings, "deepseek_tokens", [])
    monkeypatch.setattr(settings, "qwen_tokens", [])
    write = AsyncMock()
    monkeypatch.setattr(envtokens_mod, "_write_env_tokens", write)
    monkeypatch.setattr(envtokens_mod, "_read_env_tokens", AsyncMock(return_value=(["tok"], [])))
    ds_client = MagicMock()
    ds_client.check_auth = AsyncMock(return_value=True)
    monkeypatch.setattr(envtokens_mod, "DeepSeekClient", MagicMock(return_value=ds_client))
    app.state.pool = None
    client = TestClient(app)
    resp = client.post("/v1/tokens", headers={"x-api-key": "test-admin-token"}, json={"deepseek_tokens": ["tok"]})
    client.close()
    assert resp.status_code == 200
    assert resp.json()["activated"] == {"deepseek": 1, "qwen": 0}
    write.assert_not_awaited()


def test_add_tokens_rejects_unknown_field():
    client = TestClient(app)
    resp = client.post("/v1/tokens", headers=ADMIN_HEADERS, json={"deepseek_tokens": ["x"], "gigachat_keys": ["y"]})
    client.close()
    assert resp.status_code == 400


def test_write_env_tokens_is_atomic(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("A=1\nDEEPSEEK_TOKENS=old\n", encoding="utf-8")
    monkeypatch.setattr(envtokens_mod, "_env_path", lambda: env_file)
    openai_mod._write_env_tokens_sync(["new"], ["q"])
    text = env_file.read_text(encoding="utf-8")
    assert "DEEPSEEK_TOKENS=new" in text
    assert "QWEN_TOKENS=q" in text
    assert "A=1" in text
    assert list(tmp_path.glob("*.tmp")) == []


async def test_fetch_qwen_models_survives_network_error():
    client = SimpleNamespace(fetch_models=AsyncMock(side_effect=RuntimeError("connection reset")))
    app.state.qwen_models = [{"id": "kept", "name": "kept", "owned_by": "qwen", "model_type": "chat"}]
    try:
        kept = await openai_mod._store_models("qwen", client)
        assert [model["id"] for model in kept] == ["kept"]
    finally:
        app.state.qwen_models = []


async def test_image_http_client_is_shared():
    app.state.http_client = None
    first, second = await asyncio.gather(openai_mod._image_http_client(), openai_mod._image_http_client())
    assert first is second
    await first.aclose()
    app.state.http_client = None


async def test_close_pool_removes_scoped_store_files(monkeypatch, tmp_path):
    removed: list[str] = []

    class FakeStore:
        def remove(self):
            removed.append("gone")

    acct = SimpleNamespace(label="a", client=SimpleNamespace(aclose=AsyncMock()), sessions=SimpleNamespace(close_all=lambda: None))
    pool = SimpleNamespace(accounts=[acct], flush=lambda: None)
    await openai_mod._close_pool(pool, [FakeStore()])
    assert removed == ["gone"]
    acct.client.aclose.assert_awaited_once()
