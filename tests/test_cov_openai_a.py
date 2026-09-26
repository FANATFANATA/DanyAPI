import asyncio
import base64
import time
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient

import danyapi.api.openai as openai_mod
from danyapi.accounts import AccountPoolBusy
from danyapi.api.openai import ChatMessage, app, settings

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
        self.method = method

    async def body(self):
        return self._raw_body

    async def stream(self):
        for chunk in self._chunks:
            yield chunk


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
def zero_backoff():
    orig = openai_mod.RETRY_BACKOFF_SEC
    openai_mod.RETRY_BACKOFF_SEC = 0.0
    yield
    openai_mod.RETRY_BACKOFF_SEC = orig


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
    monkeypatch.setattr(openai_mod, "_env_path", lambda: tmp_path / "missing.env")
    assert openai_mod._read_env_tokens_sync() == ([], [])


def test_read_env_tokens_sync_parses(tmp_path, monkeypatch):
    path = tmp_path / "t.env"
    path.write_text("DEEPSEEK_TOKENS=\"a,b\"\nQWEN_TOKENS='c'\n", encoding="utf-8")
    monkeypatch.setattr(openai_mod, "_env_path", lambda: path)
    assert openai_mod._read_env_tokens_sync() == (["a", "b"], ["c"])


def test_write_env_tokens_sync(tmp_path, monkeypatch):
    path = tmp_path / "t.env"
    path.write_text("DEEPSEEK_TOKENS=old\nOTHER=1\n", encoding="utf-8")
    monkeypatch.setattr(openai_mod, "_env_path", lambda: path)
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
    monkeypatch.setattr(openai_mod, "_read_env_tokens", AsyncMock(return_value=([token], [])))
    client = TestClient(app)
    resp = client.post("/v1/tokens", json={"deepseek_tokens": [token]})
    client.close()
    assert resp.status_code == 200
    assert resp.json()["reactivated"]["deepseek"] == 1
    assert acct.broken is False


def test_add_tokens_reactivate_check_raises(monkeypatch):
    token = "tok"
    acct = SimpleNamespace(stable_id=openai_mod._token_stable_id(token), broken=True, client=MagicMock())
    acct.client.check_auth = AsyncMock(side_effect=RuntimeError("x"))
    app.state.pool = SimpleNamespace(accounts=[acct])
    monkeypatch.setattr(openai_mod, "_read_env_tokens", AsyncMock(return_value=([token], [])))
    client = TestClient(app)
    resp = client.post("/v1/tokens", json={"deepseek_tokens": [token]})
    client.close()
    assert resp.status_code == 400


def test_add_tokens_reactivate_not_valid(monkeypatch):
    token = "tok"
    acct = SimpleNamespace(stable_id=openai_mod._token_stable_id(token), broken=True, client=MagicMock())
    acct.client.check_auth = AsyncMock(return_value=False)
    app.state.pool = SimpleNamespace(accounts=[acct])
    monkeypatch.setattr(openai_mod, "_read_env_tokens", AsyncMock(return_value=([token], [])))
    client = TestClient(app)
    resp = client.post("/v1/tokens", json={"deepseek_tokens": [token]})
    client.close()
    assert resp.status_code == 400


def test_add_tokens_reactivates_qwen(monkeypatch):
    token = "qt"
    acct = SimpleNamespace(stable_id=openai_mod._token_stable_id(token), broken=True, broken_at=1, client=MagicMock())
    acct.client.check_auth = AsyncMock(return_value=True)
    app.state.qwen_pool = SimpleNamespace(accounts=[acct])
    monkeypatch.setattr(openai_mod, "_read_env_tokens", AsyncMock(return_value=([], [token])))
    client = TestClient(app)
    resp = client.post("/v1/tokens", json={"qwen_tokens": [token]})
    client.close()
    assert resp.status_code == 200
    assert resp.json()["reactivated"]["qwen"] == 1


def test_add_tokens_hot_adds_both(monkeypatch):
    monkeypatch.setattr(openai_mod, "_write_env_tokens", AsyncMock())
    monkeypatch.setattr(openai_mod, "_read_env_tokens", AsyncMock(return_value=([], [])))
    ds_client = MagicMock()
    ds_client.check_auth = AsyncMock(return_value=True)
    qw_client = MagicMock()
    qw_client.check_auth = AsyncMock(return_value=True)
    monkeypatch.setattr(openai_mod, "DeepSeekClient", MagicMock(return_value=ds_client))
    monkeypatch.setattr(openai_mod, "QwenClient", MagicMock(return_value=qw_client))
    pool = MagicMock()
    pool.accounts = []
    qwen_pool = MagicMock()
    qwen_pool.accounts = []
    app.state.pool = pool
    app.state.qwen_pool = qwen_pool
    monkeypatch.setattr(openai_mod, "_fetch_qwen_models", AsyncMock(side_effect=RuntimeError("boom")))
    client = TestClient(app)
    resp = client.post("/v1/tokens", json={"deepseek_tokens": ["d"], "qwen_tokens": ["q"]})
    client.close()
    assert resp.status_code == 200
    pool.add_account.assert_called_once()
    qwen_pool.add_account.assert_called_once()


def test_add_tokens_skips_invalid(monkeypatch):
    monkeypatch.setattr(openai_mod, "_write_env_tokens", AsyncMock())
    monkeypatch.setattr(openai_mod, "_read_env_tokens", AsyncMock(return_value=([], [])))
    ds_client = MagicMock()
    ds_client.check_auth = AsyncMock(return_value=False)
    ds_client.aclose = AsyncMock()
    qw_client = MagicMock()
    qw_client.check_auth = AsyncMock(return_value=False)
    qw_client.aclose = AsyncMock()
    monkeypatch.setattr(openai_mod, "DeepSeekClient", MagicMock(return_value=ds_client))
    monkeypatch.setattr(openai_mod, "QwenClient", MagicMock(return_value=qw_client))
    app.state.pool = None
    app.state.qwen_pool = None
    client = TestClient(app)
    resp = client.post("/v1/tokens", json={"deepseek_tokens": ["d"], "qwen_tokens": ["q"]})
    client.close()
    assert resp.status_code == 200
    assert resp.json()["skipped"] == {"deepseek": 1, "qwen": 1}


def test_add_tokens_all_exist(monkeypatch):
    app.state.pool = None
    app.state.qwen_pool = None
    monkeypatch.setattr(openai_mod, "_read_env_tokens", AsyncMock(return_value=(["a"], ["b"])))
    client = TestClient(app)
    resp = client.post("/v1/tokens", json={"deepseek_tokens": ["a"], "qwen_tokens": ["b"]})
    client.close()
    assert resp.status_code == 400


def test_add_tokens_no_tokens():
    client = TestClient(app)
    resp = client.post("/v1/tokens", json={})
    client.close()
    assert resp.status_code == 400


def test_add_tokens_non_list():
    client = TestClient(app)
    resp = client.post("/v1/tokens", json={"deepseek_tokens": "x"})
    client.close()
    assert resp.status_code == 400


def test_add_tokens_bad_item():
    client = TestClient(app)
    resp = client.post("/v1/tokens", json={"deepseek_tokens": [42]})
    client.close()
    assert resp.status_code == 400


async def test_read_request_body_invalid_content_length():
    req = FakeRequest(headers={"content-length": "abc"}, _body=b"x")
    assert await openai_mod._read_request_body(req, 10) == b"x"


async def test_read_request_body_header_too_large():
    req = FakeRequest(headers={"content-length": "100"}, _body=b"x")
    with pytest.raises(HTTPException) as excinfo:
        await openai_mod._read_request_body(req, 10)
    assert excinfo.value.status_code == 413


async def test_read_request_body_actual_too_large():
    req = FakeRequest(headers={"content-length": "5"}, body=b"x" * 20)
    with pytest.raises(HTTPException) as excinfo:
        await openai_mod._read_request_body(req, 10)
    assert excinfo.value.status_code == 413


async def test_read_request_body_cached_too_large():
    req = FakeRequest(headers={}, _body=b"x" * 20)
    with pytest.raises(HTTPException) as excinfo:
        await openai_mod._read_request_body(req, 10)
    assert excinfo.value.status_code == 413


async def test_read_request_body_stream_too_large():
    req = FakeRequest(headers={}, stream_chunks=[b"x" * 8, b"y" * 8])
    with pytest.raises(HTTPException) as excinfo:
        await openai_mod._read_request_body(req, 10)
    assert excinfo.value.status_code == 413


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
    assert await openai_mod._extract_request_body(FakeRequest(headers={}, method="POST", _body=b'{"a":1}')) == {"a": 1}
    assert await openai_mod._extract_request_body(FakeRequest(headers={"content-length": "7"}, body=b'{"a":1}', method="POST")) == {"a": 1}


async def test_extract_request_body_generic_error():
    class BadReq(FakeRequest):
        async def body(self):
            raise RuntimeError("x")

    assert await openai_mod._extract_request_body(BadReq(headers={"content-length": "7"}, method="POST")) == {}


async def test_extract_request_body_http_exception(monkeypatch):
    monkeypatch.setattr(openai_mod, "MAX_REQUEST_BODY", 5)
    req = FakeRequest(headers={"content-length": "7"}, method="POST", body=b"x" * 7)
    with pytest.raises(HTTPException) as excinfo:
        await openai_mod._extract_request_body(req)
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
    assert openai_mod._exception_message(RuntimeError("")) == "An unexpected error occurred"
    assert openai_mod._exception_message(RuntimeError("boom")) == "boom"


def test_request_id_header():
    assert openai_mod._request_id_header(FakeRequest(headers={"x-request-id": "abc"})) == "abc"
    generated = openai_mod._request_id_header(FakeRequest(headers={"x-request-id": "x" * 200}))
    assert len(generated) == 32


def test_validation_error_handler():
    client = TestClient(app)
    resp = client.post("/v1/chat/completions", json={"messages": "notalist"})
    client.close()
    assert resp.status_code == 400
    assert "x-request-id" in resp.headers


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
    with pytest.raises(HTTPException) as excinfo:
        openai_mod._resolve_model("nope-thinking")
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
    openai_mod._MODEL_CACHE["key"] = None
    client = TestClient(app)
    ok = client.get("/v1/models/deepseek-v4.1-flash")
    missing = client.get("/v1/models/does-not-exist")
    client.close()
    assert ok.status_code == 200
    assert missing.status_code == 404


def test_resolve_provider_via_cache():
    app.state.qwen_models = [{"id": "special-1", "name": "S", "owned_by": "qwen", "model_type": "chat"}]
    openai_mod._MODEL_CACHE["key"] = None
    assert openai_mod._resolve_provider("special-1") == "qwen"


async def test_byok_state_helpers(monkeypatch):
    monkeypatch.setattr(app.state, "byok_pools", None, raising=False)
    monkeypatch.setattr(app.state, "byok_locks", None, raising=False)
    monkeypatch.setattr(app.state, "byok_auth", None, raising=False)
    pools = await openai_mod._byok_pools_state()
    locks = await openai_mod._byok_locks_state()
    auth = await openai_mod._byok_auth_state()
    assert "deepseek" in pools and "qwen" in pools
    assert "deepseek" in locks
    assert "deepseek" in auth


def test_cached_auth_variants():
    assert openai_mod._cached_auth({}, "s", 0, 100.0) is None
    assert openai_mod._cached_auth({"s": ("x",)}, "s", 10, 100.0) is None
    assert openai_mod._cached_auth({"s": (True, "bad")}, "s", 10, 100.0) is None
    assert openai_mod._cached_auth({"s": (True, 1.0)}, "s", 10, 100.0) is None
    assert openai_mod._cached_auth({"s": (True, 99.0)}, "s", 10, 100.0) is True


def test_evict_auth(monkeypatch):
    monkeypatch.setattr(openai_mod, "BYOK_AUTH_LIMIT", 1)
    store = {"a": 1, "b": 2}
    openai_mod._evict_auth(store)
    assert len(store) == 1


async def test_extract_api_key_variants(monkeypatch):
    body = b'{"api_key":"k1"}'
    req = FakeRequest(headers={"content-type": "application/json", "content-length": str(len(body))}, body=body)
    assert await openai_mod._extract_request_api_key(req) == "k1"
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


async def test_extract_api_key_body_raises():
    class RaisingReq(FakeRequest):
        async def body(self):
            raise RuntimeError("x")

    req = RaisingReq(headers={"content-type": "application/json", "content-length": "5"})
    assert await openai_mod._extract_request_api_key(req) is None


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
    monkeypatch.setattr(openai_mod, "_close_busy_client", AsyncMock(side_effect=lambda *a: called.append(a)))
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
    stable = openai_mod._token_stable_id(token)
    monkeypatch.setattr(app.state, "byok_auth", {"deepseek": {stable: (True, time.monotonic())}, "qwen": {}}, raising=False)
    client = MagicMock()
    client.check_auth = AsyncMock(return_value=False)
    assert await openai_mod._byok_validate("deepseek", token, client) is True
    client.check_auth.assert_not_awaited()


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
    monkeypatch.setattr(openai_mod, "DeepSeekClient", MagicMock(return_value=client))
    pool = await openai_mod._byok_pool("deepseek", tokens)
    assert pool.accounts


async def test_byok_pool_evicts(monkeypatch):
    monkeypatch.setattr(settings, "cache_enabled", False)
    monkeypatch.setattr(settings, "byok_auth_ttl", 0.0)
    monkeypatch.setattr(openai_mod, "BYOK_POOL_LIMIT", 0)
    monkeypatch.setattr(app.state, "byok_pools", {"deepseek": {}, "qwen": {}}, raising=False)
    monkeypatch.setattr(app.state, "byok_locks", {"deepseek": asyncio.Lock(), "qwen": asyncio.Lock()}, raising=False)
    monkeypatch.setattr(app.state, "byok_auth", {"deepseek": {}, "qwen": {}}, raising=False)
    client = MagicMock()
    client.check_auth = AsyncMock(return_value=True)
    client.aclose = AsyncMock()
    monkeypatch.setattr(openai_mod, "DeepSeekClient", MagicMock(return_value=client))
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
    monkeypatch.setattr(openai_mod, "QwenClient", MagicMock(return_value=client))
    with pytest.raises(HTTPException) as excinfo:
        await openai_mod._byok_pool("qwen", ["t1"])
    assert excinfo.value.status_code == 401


async def test_dispatch_chat_byok(monkeypatch):
    monkeypatch.setattr(app.state, "byok", True, raising=False)
    monkeypatch.setattr(openai_mod, "_byok_pool_for", AsyncMock(return_value=MagicMock()))
    monkeypatch.setattr(openai_mod, "_chat_completions_deepseek", AsyncMock(return_value={"ok": 1}))
    monkeypatch.setattr(openai_mod, "_chat_completions_qwen", AsyncMock(return_value={"q": 1}))
    out_ds = await openai_mod._dispatch_chat(SimpleNamespace(model="deepseek-v4.1-flash"), SimpleNamespace())
    assert out_ds == {"ok": 1}
    out_qw = await openai_mod._dispatch_chat(SimpleNamespace(model="qwen3.8-max"), SimpleNamespace())
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
    req = openai_mod.CompletionRequest(model="deepseek-v4.1-flash", prompt="x")
    chat = openai_mod._completion_chat_request(req, "hi", False)
    assert chat.messages[0].content == "hi"
    assert chat.stream is False


def test_legacy_choice_from_chat():
    choice = openai_mod._legacy_choice_from_chat({"message": {"content": "hi"}, "finish_reason": "stop"}, 0)
    assert choice["text"] == "hi"
    assert choice["finish_reason"] == "stop"
    assert openai_mod._legacy_choice_from_chat({"message": "bad"}, 1)["text"] == ""
    assert openai_mod._legacy_choice_from_chat({}, 2)["finish_reason"] == "stop"


def test_legacy_completion_response():
    data = openai_mod._legacy_completion_response(
        {"choices": [{"message": {"content": "x"}, "finish_reason": "stop"}], "id": "1", "created": 5},
        0,
    )
    assert data["choices"][0]["text"] == "x"
    assert data["created"] == 5


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


async def test_completions_stream_non_stream(monkeypatch):
    async def fake_dispatch(req, request):
        return {"id": "1", "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}], "created": 1, "model": "m"}

    monkeypatch.setattr(openai_mod, "_dispatch_chat", fake_dispatch)
    req = openai_mod.CompletionRequest(model="deepseek-v4.1-flash", prompt="x")
    out = [line async for line in openai_mod._completions_stream(req, ["x"], None)]
    assert out[-1] == "data: [DONE]\n\n"


async def test_completions_stream_streaming(monkeypatch):
    async def inner():
        yield "data: {}\n\n"

    async def fake_dispatch(req, request):
        return StreamingResponse(inner(), media_type="text/event-stream")

    monkeypatch.setattr(openai_mod, "_dispatch_chat", fake_dispatch)
    req = openai_mod.CompletionRequest(model="deepseek-v4.1-flash", prompt="x")
    out = [line async for line in openai_mod._completions_stream(req, ["x"], None)]
    assert out[-1] == "data: [DONE]\n\n"


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


def test_embeddings_and_moderations_not_supported():
    client = TestClient(app)
    assert client.post("/v1/embeddings").status_code == 501
    assert client.post("/v1/moderations").status_code == 501
    client.close()


def test_responses_provider_call_qwen():
    app.state.qwen_models = [{"id": "qwen3.8-max", "name": "Q", "owned_by": "qwen", "model_type": "chat"}]
    openai_mod._MODEL_CACHE["key"] = None
    req = SimpleNamespace(model="qwen3.8-max")
    assert openai_mod._responses_provider_call(req) is openai_mod._chat_completions_qwen


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
    store.set("resp_y", {"public": {"input": [{"role": "user", "content": "hi"}]}})
    client = TestClient(app)
    resp = client.get("/v1/responses/resp_y/input_items")
    client.close()
    assert resp.status_code == 200
    assert resp.json()["data"]


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
        return {"image_urls": ["http://x/1.png"], "session_id": "s1", "usage": {"prompt_tokens": 1}}

    monkeypatch.setattr(openai_mod.qwen_api, "collect_image", fake_collect)
    hc = MagicMock()
    resp = MagicMock()
    resp.status_code = 200
    resp.content = b"abc"
    hc.get = AsyncMock(return_value=resp)
    monkeypatch.setattr(app.state, "http_client", hc, raising=False)
    out = await openai_mod._image_generations(openai_mod.ImageGenerationRequest(prompt="x", response_format="b64_json"), pool)
    assert out["data"][0]["b64_json"]
    assert out["usage"] == {"prompt_tokens": 1}


async def test_image_generations_fetch_failures(monkeypatch):
    pool = MagicMock()
    account = MagicMock()
    account.sem = asyncio.Semaphore(1)
    pool.acquire = AsyncMock(return_value=(account, None))

    async def fake_collect(**kwargs):
        return {"image_urls": ["http://x/1.png", "http://x/2.png"], "session_id": None}

    monkeypatch.setattr(openai_mod.qwen_api, "collect_image", fake_collect)
    hc = MagicMock()
    bad = MagicMock()
    bad.status_code = 500
    hc.get = AsyncMock(side_effect=[bad, RuntimeError("net")])
    monkeypatch.setattr(app.state, "http_client", hc, raising=False)
    out = await openai_mod._image_generations(openai_mod.ImageGenerationRequest(prompt="x", response_format="b64_json"), pool)
    assert all(item.get("url") for item in out["data"])


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


def test_split_stop():
    assert openai_mod._split_stop(None) == []
    assert openai_mod._split_stop("") == []
    assert openai_mod._split_stop("x") == ["x"]
    assert openai_mod._split_stop(["a", "", 5, "b"]) == ["a", "b"]
    assert openai_mod._split_stop(42) == []


def test_stream_stop_filter():
    f = openai_mod._StreamStopFilter(["END"])
    assert f.feed("") == ("", False)
    _out, hit = f.feed("hello")
    assert not hit
    _out, hit = f.feed(" END")
    assert hit
    assert f.flush() == ""
    f2 = openai_mod._StreamStopFilter(["abc"])
    f2.feed("xxab")
    assert f2.flush() == "ab"


def test_cjk_units():
    assert openai_mod._cjk_units("abc") == 0
    assert openai_mod._cjk_units("aあb") == 1


def test_trim_to_tokens():
    assert openai_mod._trim_to_tokens("", 10) == ""
    assert openai_mod._trim_to_tokens("hi", None) == "hi"
    assert openai_mod._trim_to_tokens("alpha beta gamma delta", 1) == "alpha"


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
    assert openai_mod.RESPONSE_INCOMPLETE in openai_mod._incomplete_error_body("m")


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
    data = client.get("/health").json()
    client.close()
    assert data["byok"] is True
    assert data["byok_pools"]["deepseek"] == 1


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
