import asyncio
import base64 as b64
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

import danyapi.api.openai as openai_mod
from danyapi import tools as toolemu
from danyapi.accounts import AccountPoolBusy
from danyapi.api.openai import app, settings
from danyapi.deepseek.client import DeepSeekError

OK_SSE = (
    "event: ready\n"
    'data: {"request_message_id":1,"response_message_id":2,"model_type":"default"}\n'
    "\n"
    'data: {"v":{"response":{"message_id":2,"parent_id":1,"status":"WIP","fragments":[{"id":2,"type":"RESPONSE","content":"Hi"}]}}}\n'
    "\n"
    'data: {"p":"response/status","o":"SET","v":"FINISHED"}\n'
    "\n"
)

BUSY_SSE = (
    "event: ready\n"
    'data: {"request_message_id":1,"response_message_id":2,"model_type":"expert"}\n'
    "\n"
    "event: hint\n"
    'data: {"type":"error","content":"Server is busy.","finish_reason":"server_busy"}\n'
    "\n"
)

CTX_SSE = (
    "event: ready\n"
    'data: {"request_message_id":1,"response_message_id":2,"model_type":"default"}\n'
    "\n"
    'data: {"p":"response/status","o":"SET","v":"CONTEXT_LENGTH_EXCEEDED"}\n'
    "\n"
)


@pytest.fixture(autouse=True)
def clean_state():
    saved = (getattr(app.state, "pool", None), getattr(app.state, "qwen_pool", None), getattr(app.state, "qwen_models", None))
    app.state.pool = None
    app.state.qwen_pool = None
    app.state.qwen_models = []
    yield
    app.state.pool, app.state.qwen_pool, app.state.qwen_models = saved


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


class FakeSession:
    def __init__(self, sid: str = "c1", last_message_id: str | None = None) -> None:
        self.id = sid
        self.last_message_id = last_message_id
        self.accumulated_tokens = 0


class FakeResp:
    def __init__(self, body=None, sse_text=None, status=200, content_type="text/event-stream; charset=utf-8"):
        self.status_code = status
        self.headers = {"content-type": content_type}
        if sse_text is not None:
            self._b = sse_text.encode()
        else:
            self._b = (body or "").encode()

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


def _rec(status=None, hint=None):
    rec = MagicMock()
    rec.status = status
    rec.hint_error = hint
    return rec


def test_deepseek_usage():
    assert openai_mod._deepseek_usage(10) == {"prompt_tokens": 0, "completion_tokens": 10, "total_tokens": 10}
    assert openai_mod._deepseek_usage(-5) == {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    assert openai_mod._deepseek_usage(None) == {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def test_finish_reason():
    assert openai_mod._finish_reason("FINISHED") == "stop"
    assert openai_mod._finish_reason("CONTEXT_LENGTH_EXCEEDED") == "length"
    assert openai_mod._finish_reason("CONTENT_FILTER") == "content_filter"
    assert openai_mod._finish_reason("INCOMPLETE") == "stop"
    assert openai_mod._finish_reason("WIP") == "stop"
    assert openai_mod._finish_reason("TIMEOUT") == "stop"
    assert openai_mod._finish_reason("WEIRD") == "stop"
    assert openai_mod._finish_reason(None) == "stop"
    assert openai_mod._finish_reason(42) == "stop"


def test_include_usage():
    req = SimpleNamespace(stream_options=None)
    assert not openai_mod._include_usage(req)
    req = SimpleNamespace(stream_options={"include_usage": True})
    assert openai_mod._include_usage(req)
    req = SimpleNamespace(stream_options={"include_usage": False})
    assert not openai_mod._include_usage(req)


def test_pool_stats():
    assert openai_mod._pool_stats(None) is None
    pool = MagicMock()
    pool.stats.return_value = {"a": 1}
    assert openai_mod._pool_stats(pool) == {"a": 1}
    pool = MagicMock()
    pool.stats.side_effect = RuntimeError("boom")
    assert openai_mod._pool_stats(pool) is None


def test_resolve_model():
    assert openai_mod._resolve_model("deepseek-v4-flash") == "default"
    assert openai_mod._resolve_model("deepseek-v4-pro") == "expert"
    assert openai_mod._resolve_model("deepseek-v4-vision") == "vision"


def test_resolve_model_unknown():
    with pytest.raises(Exception) as excinfo:
        openai_mod._resolve_model("nope")
    assert excinfo.value.status_code == 404


def test_resolve_provider():
    assert openai_mod._resolve_provider("qwen3.8-max") == "qwen"
    assert openai_mod._resolve_provider("deepseek-v4-flash") == "deepseek"
    assert openai_mod._resolve_provider("deepseek-whatever") == "deepseek"


def test_resolve_provider_unknown():
    with pytest.raises(Exception) as excinfo:
        openai_mod._resolve_provider("gpt-4")
    assert excinfo.value.status_code == 404


def test_is_retryable_hint():
    assert openai_mod._is_retryable_hint(_rec(hint={"finish_reason": "server_busy"}))
    assert openai_mod._is_retryable_hint(_rec(hint={"finish_reason": "parallel_chat_limit"}))
    assert not openai_mod._is_retryable_hint(_rec(hint={"finish_reason": "other"}))
    assert not openai_mod._is_retryable_hint(_rec(hint=None))


def test_is_retryable_http():
    assert openai_mod._is_retryable_http(openai_mod.HTTPException(429, "x"))
    assert openai_mod._is_retryable_http(openai_mod.HTTPException(502, "x"))
    assert not openai_mod._is_retryable_http(openai_mod.HTTPException(404, "x"))


def test_is_context_limit():
    assert openai_mod._is_context_limit(_rec(status="CONTEXT_LENGTH_EXCEEDED"))
    assert openai_mod._is_context_limit(_rec(hint={"finish_reason": "CONTEXT_LENGTH_EXCEEDED"}))
    assert not openai_mod._is_context_limit(_rec(status="FINISHED"))


def test_is_input_exceeds_limit():
    assert openai_mod._is_input_exceeds_limit(_rec(status="input_exceeds_limit"))
    assert openai_mod._is_input_exceeds_limit(_rec(hint={"finish_reason": "input_exceeds_limit"}))
    assert not openai_mod._is_input_exceeds_limit(_rec(status="FINISHED"))


def test_input_exceeds_hint_from_http():
    body = '{"message":"Content is too long. Please shorten it and try again.","finish_reason":"input_exceeds_limit"}'
    hint = openai_mod._input_exceeds_hint_from_http(openai_mod.HTTPException(400, body))
    assert hint is not None
    assert hint["finish_reason"] == "input_exceeds_limit"
    assert hint["message"] == "Content is too long. Please shorten it and try again."
    assert openai_mod._input_exceeds_hint_from_http(openai_mod.HTTPException(400, "plain")) is None
    assert openai_mod._input_exceeds_hint_from_http(openai_mod.HTTPException(400, {"finish_reason": "other"})) is None
    assert openai_mod._input_exceeds_hint_from_http(openai_mod.HTTPException(400, {"finish_reason": "input_exceeds_limit"})) is not None
    assert openai_mod._input_exceeds_hint_from_http(openai_mod.HTTPException(400, 42)) is None


def test_deepseek_status():
    err = DeepSeekError(40001, "bad")
    assert openai_mod._deepseek_status(err) == 401
    err = DeepSeekError(5000, "bad")
    assert openai_mod._deepseek_status(err) == 502


def test_deepseek_error_detail():
    err = DeepSeekError(40001, "bad")
    assert "auth" in openai_mod._deepseek_error_detail(err)
    err = DeepSeekError(5000, "bad")
    assert "error" in openai_mod._deepseek_error_detail(err)


def test_handle_account_error_auth():
    acct = FakeAccount()
    openai_mod._handle_account_error(acct, DeepSeekError(40001, "bad"))
    assert acct.broken


def test_handle_account_error_other():
    acct = FakeAccount()
    openai_mod._handle_account_error(acct, DeepSeekError(5000, "bad"))
    assert not acct.broken


def test_drop_session():
    pool = MagicMock()
    acct = FakeAccount()
    openai_mod._drop_session(pool, acct, "s1")
    pool.forget.assert_called_once_with("s1")
    pool.forget_context.assert_called_once_with("s1")
    acct.sessions.forget.assert_called_once_with("s1")


def test_busy_error_body():
    rec = MagicMock()
    rec.hint_error = {"message": "busy", "finish_reason": "server_busy"}
    body = json.loads(openai_mod._busy_error_body(rec))
    assert body["error"]["message"] == "busy"
    rec = MagicMock()
    rec.hint_error = {}
    body = json.loads(openai_mod._busy_error_body(rec))
    assert "busy" in body["error"]["message"]


async def test_send_with_auth_marks_broken():
    acct = FakeAccount()
    acct.client.completion = AsyncMock(side_effect=openai_mod.HTTPException(401, "unauthorized"))
    with pytest.raises(Exception) as excinfo:
        await openai_mod._send_with_auth(acct, acct.client, {}, "s", None, "p", "default", False, False)
    assert excinfo.value.status_code == 401
    assert acct.broken


async def test_send_with_auth_other_status():
    acct = FakeAccount()
    acct.client.completion = AsyncMock(side_effect=openai_mod.HTTPException(502, "boom"))
    with pytest.raises(Exception) as excinfo:
        await openai_mod._send_with_auth(acct, acct.client, {}, "s", None, "p", "default", False, False)
    assert excinfo.value.status_code == 502
    assert not acct.broken


async def test_new_session_registered():
    acct = FakeAccount()
    pool = MagicMock()
    acct.sessions.obtain = AsyncMock(return_value=(FakeSession(sid="new1"), "new1"))
    _, key, _ = await openai_mod._prepare_session(acct, pool, None, ("u1",))
    assert key == "new1"
    pool.register.assert_called_once_with(0, "new1")
    pool.index_context.assert_called_once_with("new1", ("u1",))


async def test_existing_session_reused():
    acct = FakeAccount()
    pool = MagicMock()
    _, key, _ = await openai_mod._prepare_session(acct, pool, "s1")
    assert key == "s1"
    pool.register.assert_called_once_with(0, "s1")


async def test_swapped_session_forgets_old():
    acct = FakeAccount()
    pool = MagicMock()
    acct.sessions.obtain = AsyncMock(return_value=(FakeSession(sid="new1"), "new1"))
    await openai_mod._prepare_session(acct, pool, "old1", ("u1",))
    pool.forget.assert_called_once_with("old1")
    pool.forget_context.assert_called_once_with("old1")
    acct.sessions.forget.assert_called_once_with("old1")


async def test_auth_error_401():
    acct = FakeAccount()
    acct.sessions.obtain = AsyncMock(side_effect=DeepSeekError(40001, "bad"))
    with pytest.raises(Exception) as excinfo:
        await openai_mod._prepare_session(acct, MagicMock(), "s1")
    assert excinfo.value.status_code == 401
    assert acct.broken


async def test_other_error_502():
    acct = FakeAccount()
    acct.sessions.obtain = AsyncMock(side_effect=DeepSeekError(5000, "bad"))
    with pytest.raises(Exception) as excinfo:
        await openai_mod._prepare_session(acct, MagicMock(), "s1")
    assert excinfo.value.status_code == 502
    assert not acct.broken


async def _send(resp):
    client = MagicMock()
    client.completion = AsyncMock(return_value=resp)
    return await openai_mod._send_completion(client, {}, "s", None, "p", "default", False, False)


async def test_non_stream_biz_code():
    resp = FakeResp(body=json.dumps({"data": {"biz_code": 40001, "biz_msg": "bad"}}), content_type="application/json")
    with pytest.raises(Exception) as excinfo:
        await _send(resp)
    assert excinfo.value.status_code == 401


async def test_non_stream_code():
    resp = FakeResp(body=json.dumps({"code": 5000, "msg": "oops"}), content_type="application/json")
    with pytest.raises(Exception) as excinfo:
        await _send(resp)
    assert excinfo.value.status_code == 502


async def test_non_stream_bad_json():
    resp = FakeResp(body="not json", content_type="application/json")
    with pytest.raises(Exception) as excinfo:
        await _send(resp)
    assert excinfo.value.status_code == 502


async def test_non_stream_unexpected():
    resp = FakeResp(body=json.dumps({"code": 0}), content_type="application/json")
    with pytest.raises(Exception) as excinfo:
        await _send(resp)
    assert excinfo.value.status_code == 502


async def test_http_status_error():
    client = MagicMock()
    client.completion = AsyncMock(side_effect=httpx.HTTPStatusError("500", request=MagicMock(), response=MagicMock(status_code=500)))
    with pytest.raises(Exception) as excinfo:
        await openai_mod._send_completion(client, {}, "s", None, "p", "default", False, False)
    assert excinfo.value.status_code == 500


async def test_http_error():
    client = MagicMock()
    client.completion = AsyncMock(side_effect=httpx.ConnectError("boom"))
    with pytest.raises(Exception) as excinfo:
        await openai_mod._send_completion(client, {}, "s", None, "p", "default", False, False)
    assert excinfo.value.status_code == 502


async def test_non_200_status():
    resp = FakeResp(sse_text="data: x\n\n", status=429)
    with pytest.raises(Exception) as excinfo:
        await _send(resp)
    assert excinfo.value.status_code == 429


async def test_sse_ok():
    resp = FakeResp(sse_text="data: {}\n\n")
    result = await _send(resp)
    assert result.status_code == 200


async def test_fresh_pow_error():
    acct = FakeAccount()
    acct.pow.make_header = AsyncMock(side_effect=DeepSeekError(40001, "bad"))
    with pytest.raises(Exception) as excinfo:
        await openai_mod._fresh_pow_headers(acct)
    assert excinfo.value.status_code == 401


async def test_fresh_pow_ok():
    acct = FakeAccount()
    result = await openai_mod._fresh_pow_headers(acct)
    assert result == {}


async def test_continuation_success():
    acct = FakeAccount([OK_SSE])
    acct.sessions.obtain = AsyncMock(return_value=(FakeSession(), "s1"))
    rec = await openai_mod._collect_continuation(acct, FakeSession(), None, "default", False, False)
    assert rec is not None
    assert rec.content == "Hi"


async def test_continuation_retries_then_success():
    acct = FakeAccount([BUSY_SSE, OK_SSE])
    rec = await openai_mod._collect_continuation(acct, FakeSession(), None, "default", False, False)
    assert rec is not None
    assert rec.content == "Hi"


async def test_continuation_gives_up():
    acct = FakeAccount([BUSY_SSE] * (openai_mod.MAX_RETRIES + 1))
    rec = await openai_mod._collect_continuation(acct, FakeSession(), None, "default", False, False)
    assert rec is not None
    assert not rec.content


async def test_continuation_http_error_returns_none():
    acct = FakeAccount()
    acct.client.completion = AsyncMock(side_effect=openai_mod.HTTPException(404, "x"))
    rec = await openai_mod._collect_continuation(acct, FakeSession(), None, "default", False, False)
    assert rec is None


async def test_guard_relays():
    async def gen():
        yield "a"
        yield "b"

    result = await _collect_agen(openai_mod._stream_guard(gen(), "m"))
    assert result == ["a", "b"]


async def test_guard_emits_error_on_busy():
    async def gen():
        raise AccountPoolBusy()
        yield "never"

    lines = await _collect_agen(openai_mod._stream_guard(gen(), "m"))
    joined = "".join(lines)
    assert '"error"' in joined
    assert "all accounts are busy" in joined
    assert joined.rstrip().endswith("data: [DONE]")


def test_split_data_uri_invalid_prefix():
    with pytest.raises(Exception) as excinfo:
        openai_mod._split_data_uri("http://x/y.png")
    assert excinfo.value.status_code == 400


def test_split_data_uri_invalid_base64():
    with pytest.raises(Exception) as excinfo:
        openai_mod._split_data_uri("data:image/png;base64,@@@")
    assert excinfo.value.status_code == 400


def test_split_data_uri_ok():
    ct, data = openai_mod._split_data_uri("data:image/png;base64," + b64.b64encode(b"abc").decode())
    assert ct == "image/png"
    assert data == b"abc"


def test_collect_invalid_image_url():
    req = SimpleNamespace(
        messages=[openai_mod.ChatMessage(role="user", content=[{"type": "image_url", "image_url": 42}])],
        files=None,
    )
    with pytest.raises(Exception) as excinfo:
        openai_mod._collect_attachments(req)
    assert excinfo.value.status_code == 400


def test_collect_file_requires_content():
    req = SimpleNamespace(messages=[], files=[SimpleNamespace(name="a.txt", content="", content_type="text/plain")])
    with pytest.raises(Exception) as excinfo:
        openai_mod._collect_attachments(req)
    assert excinfo.value.status_code == 400


def test_collect_file_invalid_base64():
    req = SimpleNamespace(messages=[], files=[SimpleNamespace(name="a.txt", content="a", content_type="text/plain")])
    with pytest.raises(Exception) as excinfo:
        openai_mod._collect_attachments(req)
    assert excinfo.value.status_code == 400


def test_collect_attachment_from_image_url():
    img = b64.b64encode(b"png").decode()
    req = SimpleNamespace(
        messages=[openai_mod.ChatMessage(role="user", content=[{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}}])],
        files=[],
    )
    atts = openai_mod._collect_attachments(req)
    assert len(atts) == 1
    assert atts[0].is_image


def test_too_many_files():
    req = SimpleNamespace(
        messages=[],
        files=[SimpleNamespace(name=f"{i}.txt", content="aGk=", content_type="text/plain") for i in range(openai_mod.MAX_FILES_PER_REQUEST + 1)],
    )
    atts = openai_mod._collect_attachments(req)
    with pytest.raises(Exception) as excinfo:
        openai_mod._validate_attachments(atts, "default")
    assert excinfo.value.status_code == 400


async def test_upload_error_path():
    acct = FakeAccount()
    acct.client.upload_file = AsyncMock(side_effect=DeepSeekError(5000, "boom"))
    with pytest.raises(Exception) as excinfo:
        await openai_mod._upload_attachments(acct, [openai_mod.Attachment(b"a", "a.txt", "text/plain", False)], "default", False)
    assert excinfo.value.status_code == 502


async def test_upload_no_file_id():
    acct = FakeAccount()
    acct.client.upload_file = AsyncMock(return_value={})
    with pytest.raises(Exception) as excinfo:
        await openai_mod._upload_attachments(acct, [openai_mod.Attachment(b"a", "a.txt", "text/plain", False)], "default", False)
    assert excinfo.value.status_code == 502


async def test_success_filters():
    client = MagicMock()
    client.fetch_models = AsyncMock(
        return_value=[
            {"id": "m1", "info": {"meta": {"chat_type": ["t2t", "rag"]}}},
            {"id": "m2", "info": {"meta": {"chat_type": ["video"]}}},
            {"id": "m3"},
            "garbage",
        ]
    )
    result = await openai_mod._fetch_qwen_models(client)
    ids = [m["id"] for m in result]
    assert "m1" in ids
    assert "m2" in ids
    assert "m3" in ids
    assert len(result) == 3


async def test_error_uses_defaults():
    from danyapi.qwen.client import QwenError

    client = MagicMock()
    client.fetch_models = AsyncMock(side_effect=QwenError(500, "boom"))
    result = await openai_mod._fetch_qwen_models(client)
    assert result == openai_mod.QWEN_DEFAULT_MODELS


async def test_empty_uses_defaults():
    client = MagicMock()
    client.fetch_models = AsyncMock(return_value=[])
    result = await openai_mod._fetch_qwen_models(client)
    assert result == openai_mod.QWEN_DEFAULT_MODELS


async def test_busy():
    pool = MagicMock()
    pool.acquire = AsyncMock(side_effect=AccountPoolBusy())
    with pytest.raises(Exception) as excinfo:
        await openai_mod._acquire_account(pool, None)
    assert excinfo.value.status_code == 429


async def test_runtime_error():
    pool = MagicMock()
    pool.acquire = AsyncMock(side_effect=RuntimeError("all down"))
    with pytest.raises(Exception) as excinfo:
        await openai_mod._acquire_account(pool, None)
    assert excinfo.value.status_code == 503


def test_health_no_pools():
    client = TestClient(app)
    resp = client.get("/health")
    client.close()
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert not data["deepseek"]
    assert not data["qwen"]


def test_health_with_pools():
    pool = MagicMock()
    pool.stats.return_value = {"accounts": 2}
    app.state.pool = pool
    app.state.qwen_pool = pool
    client = TestClient(app)
    data = client.get("/health").json()
    client.close()
    assert data["deepseek"]
    assert data["qwen"]
    assert data["deepseek_stats"] == {"accounts": 2}


def test_list_models():
    app.state.qwen_models = [{"id": "qwen3.8-max", "name": "Q", "owned_by": "qwen", "model_type": "chat"}]
    client = TestClient(app)
    data = client.get("/v1/models").json()
    client.close()
    ids = [m["id"] for m in data["data"]]
    assert "deepseek-v4-flash" in ids
    assert "qwen3.8-max" in ids


def test_chat_unknown_model():
    client = TestClient(app)
    resp = client.post("/v1/chat/completions", json={"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]})
    client.close()
    assert resp.status_code == 404


def test_chat_deepseek_not_configured():
    client = TestClient(app)
    resp = client.post("/v1/chat/completions", json={"model": "deepseek-v4-flash", "messages": [{"role": "user", "content": "hi"}]})
    client.close()
    assert resp.status_code == 503


def test_chat_qwen_not_configured():
    client = TestClient(app)
    resp = client.post("/v1/chat/completions", json={"model": "qwen3.8-max", "messages": [{"role": "user", "content": "hi"}]})
    client.close()
    assert resp.status_code == 503


def test_chat_deepseek_non_stream():
    pool, _ = make_pool()
    app.state.pool = pool
    client = TestClient(app)
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "deepseek-v4-flash", "messages": [{"role": "user", "content": "hi"}]},
    )
    client.close()
    assert resp.status_code == 200
    data = resp.json()
    assert data["choices"][0]["message"]["content"] == "Hi"
    assert data["session_id"] == "s1"


def test_chat_deepseek_stream():
    pool, _ = make_pool()
    app.state.pool = pool
    client = TestClient(app)
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "deepseek-v4-flash", "messages": [{"role": "user", "content": "hi"}], "stream": True},
    )
    client.close()
    assert resp.status_code == 200
    assert '"content": "Hi"' in resp.text
    assert resp.text.rstrip().endswith("data: [DONE]")


def test_chat_deepseek_context_length():
    pool, acct = make_pool()
    acct.client.completion = AsyncMock(return_value=FakeResp(sse_text=CTX_SSE))
    pool.acquire = AsyncMock(return_value=(acct, "s1"))
    app.state.pool = pool
    client = TestClient(app)
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "deepseek-v4-flash", "messages": [{"role": "user", "content": "hi"}]},
    )
    client.close()
    assert resp.status_code == 400


def test_chat_qwen_non_stream():
    from danyapi.qwen.client import QwenSession

    acct = MagicMock()
    acct.index = 0
    acct.broken = False
    acct.sem = asyncio.Semaphore(1)
    acct.client = MagicMock()
    acct.client.completion = AsyncMock(
        return_value=FakeResp(
            sse_text=(
                'data: {"response.created":{"chat_id":"c1","parent_id":"p0","response_id":"r1"}} \n\n'
                'data: {"choices": [{"delta": {"content": "Hello", "phase": "answer"}}], "response_id": "r1"}\n\n'
                'data: {"choices": [{"delta": {"status": "finished", "phase": "answer"}}], "response_id": "r1"}\n\n'
            )
        )
    )
    acct.sessions = MagicMock()
    acct.sessions.obtain = AsyncMock(return_value=(QwenSession(id="c1"), "c1"))
    acct.sessions.touch_last_message = MagicMock()
    acct.sessions.forget = MagicMock()
    pool = MagicMock()
    pool.acquire = AsyncMock(return_value=(acct, None))
    app.state.qwen_pool = pool
    client = TestClient(app)
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "qwen3.8-max", "messages": [{"role": "user", "content": "hi"}]},
    )
    client.close()
    assert resp.status_code == 200
    assert resp.json()["choices"][0]["message"]["content"] == "Hello"


def test_chat_deepseek_attachment_via_endpoint():
    pool, acct = make_pool()
    acct.client.upload_file = AsyncMock(return_value={"id": "f1"})
    app.state.pool = pool
    client = TestClient(app)
    payload = {"name": "a.png", "content": b64.b64encode(b"hello").decode(), "content_type": "image/png"}
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "deepseek-v4-vision", "messages": [{"role": "user", "content": "x"}], "files": [payload]},
    )
    client.close()
    assert resp.status_code == 200
    acct.client.upload_file.assert_awaited_once()


def _patch_creds(ds_tokens=None, qwen_tokens=None, cache=True):
    from contextlib import ExitStack

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


@pytest.mark.usefixtures("reset_app_state")
def test_deepseek_tokens_ok():
    with _patch_creds(ds_tokens=["tok"]):
        with TestClient(app):
            pool = app.state.pool
            assert pool is not None
            assert len(pool.accounts) == 1
            assert app.state.qwen_pool is None


@pytest.mark.usefixtures("reset_app_state")
def test_deepseek_invalid_skipped():
    from danyapi.deepseek.client import DeepSeekClient as DSC

    with (
        patch.object(settings, "deepseek_tokens", ["bad"]),
        patch.object(settings, "qwen_tokens", []),
        patch.object(DSC, "check_auth", new=AsyncMock(return_value=False)),
    ):
        with pytest.raises(RuntimeError):
            with TestClient(app):
                pass


@pytest.mark.usefixtures("reset_app_state")
def test_qwen_tokens_ok():
    with _patch_creds(qwen_tokens=["tok"]):
        with TestClient(app):
            assert app.state.qwen_pool is not None
            assert len(app.state.qwen_pool.accounts) == 1
            assert app.state.qwen_models


@pytest.mark.usefixtures("reset_app_state")
def test_no_credentials_raises():
    with (
        patch.object(settings, "deepseek_tokens", []),
        patch.object(settings, "qwen_tokens", []),
    ):
        with pytest.raises(RuntimeError, match="DEEPSEEK_TOKENS or QWEN_TOKENS"):
            with TestClient(app):
                pass


@pytest.mark.usefixtures("reset_app_state")
def test_cache_disabled_runs():
    with _patch_creds(ds_tokens=["tok"], cache=False):
        with TestClient(app):
            assert app.state.pool is not None


@pytest.mark.usefixtures("reset_app_state")
def test_both_providers():
    with _patch_creds(ds_tokens=["t1"], qwen_tokens=["t2"]):
        with TestClient(app):
            assert app.state.pool is not None
            assert app.state.qwen_pool is not None
            assert app.state.qwen_models


def test_image_url_string_form():
    img = b64.b64encode(b"png").decode()
    req = SimpleNamespace(
        messages=[openai_mod.ChatMessage(role="user", content=[{"type": "image_url", "image_url": f"data:image/png;base64,{img}"}])],
        files=[],
    )
    atts = openai_mod._collect_attachments(req)
    assert len(atts) == 1
    assert atts[0].is_image


def test_file_too_large():
    req = SimpleNamespace(
        messages=[],
        files=[SimpleNamespace(name="big.bin", content=b64.b64encode(b"x" * (openai_mod.MAX_FILE_SIZE + 1)).decode(), content_type="application/octet-stream")],
    )
    atts = openai_mod._collect_attachments(req)
    with pytest.raises(Exception) as excinfo:
        openai_mod._validate_attachments(atts, "default")
    assert excinfo.value.status_code == 400


async def test_fresh_pow_upload_error():
    acct = FakeAccount()
    acct.pow_upload.make_header = AsyncMock(side_effect=DeepSeekError(40001, "bad"))
    with pytest.raises(Exception) as excinfo:
        await openai_mod._fresh_pow_upload_headers(acct)
    assert excinfo.value.status_code == 401


async def test_try_stop_stream():
    client = MagicMock()
    client.stop_stream = AsyncMock()
    await openai_mod._try_stop_stream(client, "s1", "m1")
    client.stop_stream.assert_awaited_once_with("s1", "m1")
    await openai_mod._try_stop_stream(client, "", "m1")
    await openai_mod._try_stop_stream(client, "s1", None)
    client.stop_stream.assert_awaited_once()


async def test_try_stop_stream_error():
    client = MagicMock()
    client.stop_stream = AsyncMock(side_effect=Exception("boom"))
    await openai_mod._try_stop_stream(client, "s1", "m1")


async def test_retries_retryable_http():
    acct = FakeAccount([OK_SSE])
    acct.client.completion = AsyncMock(
        side_effect=[
            openai_mod.HTTPException(429, "slow down"),
            FakeResp(sse_text=OK_SSE),
        ]
    )
    rec = await openai_mod._collect_continuation(acct, FakeSession(), None, "default", False, False)
    assert rec is not None
    assert rec.content == "Hi"


INPUT_SSE = (
    "event: ready\n"
    'data: {"request_message_id":1,"response_message_id":2,"model_type":"default"}\n'
    "\n"
    'data: {"p":"response/status","o":"SET","v":"input_exceeds_limit"}\n'
    "\n"
)


async def test_non_stream_auto_continues():
    acct = FakeAccount([INPUT_SSE, OK_SSE])
    acct.sessions.obtain = AsyncMock(return_value=(FakeSession(), "s1"))
    result = await openai_mod._collect_non_stream(
        account=acct,
        pool=MagicMock(),
        existing_sid="s1",
        lock=acct.sem,
        prompt="x",
        model="deepseek-v4-flash",
        model_type="default",
        thinking=False,
        search=False,
    )
    assert result["choices"][0]["message"]["content"] == "Hi"
    assert acct.client.completion.await_count == 2


async def test_stream_auto_continues():
    acct = FakeAccount([INPUT_SSE, OK_SSE])
    acct.sessions.obtain = AsyncMock(return_value=(FakeSession(), "s1"))
    gen = openai_mod._stream_openai(
        account=acct,
        pool=MagicMock(),
        existing_sid="s1",
        lock=acct.sem,
        prompt="x",
        model="deepseek-v4-flash",
        model_type="default",
        thinking=False,
        search=False,
    )
    lines = await _collect_agen(gen)
    joined = "".join(lines)
    assert '"content": "Hi"' in joined
    assert joined.rstrip().endswith("data: [DONE]")


INPUT_HTTP_BODY = '{"message":"Content is too long. Please shorten it and try again.","finish_reason":"input_exceeds_limit"}'


async def test_non_stream_input_exceeds_http_continues():
    acct = FakeAccount([OK_SSE])
    acct.client.completion = AsyncMock(
        side_effect=[
            openai_mod.HTTPException(400, INPUT_HTTP_BODY),
            FakeResp(sse_text=OK_SSE),
        ]
    )
    acct.sessions.obtain = AsyncMock(return_value=(FakeSession(), "s1"))
    result = await openai_mod._collect_non_stream(
        account=acct,
        pool=MagicMock(),
        existing_sid="s1",
        lock=acct.sem,
        prompt="x",
        model="deepseek-v4-flash",
        model_type="default",
        thinking=False,
        search=False,
    )
    assert result["choices"][0]["message"]["content"] == "Hi"
    assert acct.client.completion.await_count == 2


async def test_stream_input_exceeds_http_continues():
    acct = FakeAccount([OK_SSE])
    acct.client.completion = AsyncMock(
        side_effect=[
            openai_mod.HTTPException(400, INPUT_HTTP_BODY),
            FakeResp(sse_text=OK_SSE),
        ]
    )
    acct.sessions.obtain = AsyncMock(return_value=(FakeSession(), "s1"))
    gen = openai_mod._stream_openai(
        account=acct,
        pool=MagicMock(),
        existing_sid="s1",
        lock=acct.sem,
        prompt="x",
        model="deepseek-v4-flash",
        model_type="default",
        thinking=False,
        search=False,
    )
    joined = "".join(await _collect_agen(gen))
    assert '"content": "Hi"' in joined
    assert joined.rstrip().endswith("data: [DONE]")


async def test_non_stream_input_exceeds_http_continuation_none():
    acct = FakeAccount([])
    acct.client.completion = AsyncMock(
        side_effect=[
            openai_mod.HTTPException(400, INPUT_HTTP_BODY),
            openai_mod.HTTPException(404, "gone"),
        ]
    )
    acct.sessions.obtain = AsyncMock(return_value=(FakeSession(), "s1"))
    with pytest.raises(openai_mod.HTTPException) as excinfo:
        await openai_mod._collect_non_stream(
            account=acct,
            pool=MagicMock(),
            existing_sid="s1",
            lock=acct.sem,
            prompt="x",
            model="deepseek-v4-flash",
            model_type="default",
            thinking=False,
            search=False,
        )
    assert excinfo.value.status_code == 502
    assert "Content is too long" in excinfo.value.detail
    assert "response_incomplete" in excinfo.value.detail


async def test_stream_input_exceeds_http_continuation_none():
    acct = FakeAccount([])
    acct.client.completion = AsyncMock(
        side_effect=[
            openai_mod.HTTPException(400, INPUT_HTTP_BODY),
            openai_mod.HTTPException(404, "gone"),
        ]
    )
    acct.sessions.obtain = AsyncMock(return_value=(FakeSession(), "s1"))
    gen = openai_mod._stream_openai(
        account=acct,
        pool=MagicMock(),
        existing_sid="s1",
        lock=acct.sem,
        prompt="x",
        model="deepseek-v4-flash",
        model_type="default",
        thinking=False,
        search=False,
    )
    lines = await _collect_agen(gen)
    joined = "".join(lines)
    assert '"error"' in joined
    assert "Content is too long" in joined
    assert '"response_incomplete"' in joined
    assert joined.rstrip().endswith("data: [DONE]")


REDUCED_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
    },
}


def test_reduced_prompt_variants():
    from danyapi.api.openai import ChatMessage

    msgs = [
        ChatMessage(role="system", content="sys"),
        ChatMessage(role="user", content="hello"),
        ChatMessage(role="assistant", content="hi"),
        ChatMessage(role="user", content="world"),
    ]
    variants = openai_mod._reduced_prompt_variants(msgs, [REDUCED_TOOL], None, None, "original")
    assert len(variants) == 3
    for prompt, tool_mode, schemas in variants[:2]:
        assert not tool_mode
        assert schemas == {}
        assert "world" in prompt
    assert "hello" in variants[0][0]
    assert "hello" not in variants[1][0]
    assert "sys" in variants[1][0]
    assert variants[2][1] is True
    assert "get_weather" in variants[2][0]
    assert "world" in variants[2][0]
    variants = openai_mod._reduced_prompt_variants(msgs, None, None, None, "original")
    assert len(variants) == 1
    assert "hello" not in variants[0][0]
    variants = openai_mod._reduced_prompt_variants(msgs, None, None, None, variants[0][0])
    assert variants == []
    variants = openai_mod._reduced_prompt_variants([ChatMessage(role="user", content=123)], [REDUCED_TOOL], None, None, "original")
    assert variants == []
    variants = openai_mod._reduced_prompt_variants([ChatMessage(role="user", content=123)], None, None, None, "original")
    assert variants == []


async def test_non_stream_input_exceeds_reduced_retry():
    acct = FakeAccount([])
    acct.client.completion = AsyncMock(
        side_effect=[
            openai_mod.HTTPException(400, INPUT_HTTP_BODY),
            openai_mod.HTTPException(404, "gone"),
            FakeResp(sse_text=OK_SSE.rstrip("\n")),
        ]
    )
    acct.sessions.obtain = AsyncMock(return_value=(FakeSession(), "s1"))
    result = await openai_mod._collect_non_stream(
        account=acct,
        pool=MagicMock(),
        existing_sid="s1",
        lock=acct.sem,
        prompt="x",
        model="deepseek-v4-flash",
        model_type="default",
        thinking=False,
        search=False,
        reduced_prompts=[("short prompt", False, {})],
    )
    assert result["choices"][0]["message"]["content"] == "Hi"
    assert result["choices"][0]["finish_reason"] == "response_incomplete"
    assert result["error"]["finish_reason"] == "response_incomplete"
    assert acct.client.completion.await_count == 3
    acct.sessions.forget.assert_called_once_with("s1")


async def test_stream_input_exceeds_reduced_retry():
    acct = FakeAccount([])
    acct.client.completion = AsyncMock(
        side_effect=[
            openai_mod.HTTPException(400, INPUT_HTTP_BODY),
            openai_mod.HTTPException(404, "gone"),
            FakeResp(sse_text=OK_SSE.rstrip("\n")),
        ]
    )
    acct.sessions.obtain = AsyncMock(return_value=(FakeSession(), "s1"))
    gen = openai_mod._stream_openai(
        account=acct,
        pool=MagicMock(),
        existing_sid="s1",
        lock=acct.sem,
        prompt="x",
        model="deepseek-v4-flash",
        model_type="default",
        thinking=False,
        search=False,
        reduced_prompts=[("short prompt", False, {})],
    )
    joined = "".join(await _collect_agen(gen))
    assert '"content": "Hi"' in joined
    assert '"error"' in joined
    assert '"response_incomplete"' in joined
    assert joined.rstrip().endswith("data: [DONE]")
    acct.sessions.forget.assert_called_once_with("s1")


async def test_stream_input_exceeds_reduced_retry_reasoning():
    acct = FakeAccount([])
    acct.client.completion = AsyncMock(
        side_effect=[
            openai_mod.HTTPException(400, INPUT_HTTP_BODY),
            openai_mod.HTTPException(404, "gone"),
            FakeResp(sse_text=THINK_SSE),
        ]
    )
    acct.sessions.obtain = AsyncMock(return_value=(FakeSession(), "s1"))
    gen = openai_mod._stream_openai(
        account=acct,
        pool=MagicMock(),
        existing_sid="s1",
        lock=acct.sem,
        prompt="x",
        model="deepseek-v4-flash",
        model_type="default",
        thinking=True,
        search=False,
        reduced_prompts=[("short prompt", False, {})],
    )
    joined = "".join(await _collect_agen(gen))
    assert '"reasoning_content": "why"' in joined
    assert '"content": "Answer"' in joined
    assert '"response_incomplete"' in joined
    assert joined.rstrip().endswith("data: [DONE]")


async def test_non_stream_input_exceeds_continuation_still_input_exceeds():
    acct = FakeAccount([INPUT_SSE, INPUT_SSE, OK_SSE])
    acct.sessions.obtain = AsyncMock(return_value=(FakeSession(), "s1"))
    result = await openai_mod._collect_non_stream(
        account=acct,
        pool=MagicMock(),
        existing_sid="s1",
        lock=acct.sem,
        prompt="x",
        model="deepseek-v4-flash",
        model_type="default",
        thinking=False,
        search=False,
        reduced_prompts=[("short prompt", False, {})],
    )
    assert result["choices"][0]["message"]["content"] == "Hi"
    assert acct.client.completion.await_count == 3


async def test_stream_input_exceeds_continuation_still_input_exceeds():
    acct = FakeAccount([INPUT_SSE, INPUT_SSE, OK_SSE])
    acct.sessions.obtain = AsyncMock(return_value=(FakeSession(), "s1"))
    gen = openai_mod._stream_openai(
        account=acct,
        pool=MagicMock(),
        existing_sid="s1",
        lock=acct.sem,
        prompt="x",
        model="deepseek-v4-flash",
        model_type="default",
        thinking=False,
        search=False,
        reduced_prompts=[("short prompt", False, {})],
    )
    joined = "".join(await _collect_agen(gen))
    assert '"content": "Hi"' in joined
    assert joined.rstrip().endswith("data: [DONE]")


async def test_non_stream_input_exceeds_reduced_retry_fails():
    acct = FakeAccount([])
    acct.client.completion = AsyncMock(
        side_effect=[
            openai_mod.HTTPException(400, INPUT_HTTP_BODY),
            openai_mod.HTTPException(404, "gone"),
            openai_mod.HTTPException(400, INPUT_HTTP_BODY),
        ]
    )
    acct.sessions.obtain = AsyncMock(return_value=(FakeSession(), "s1"))
    with pytest.raises(openai_mod.HTTPException) as excinfo:
        await openai_mod._collect_non_stream(
            account=acct,
            pool=MagicMock(),
            existing_sid="s1",
            lock=acct.sem,
            prompt="x",
            model="deepseek-v4-flash",
            model_type="default",
            thinking=False,
            search=False,
            reduced_prompts=[("short prompt", False, {})],
        )
    assert excinfo.value.status_code == 502
    assert "Content is too long" in excinfo.value.detail
    assert "response_incomplete" in excinfo.value.detail


async def test_stream_input_exceeds_reduced_retry_fails():
    acct = FakeAccount([])
    acct.client.completion = AsyncMock(
        side_effect=[
            openai_mod.HTTPException(400, INPUT_HTTP_BODY),
            openai_mod.HTTPException(404, "gone"),
            openai_mod.HTTPException(400, INPUT_HTTP_BODY),
        ]
    )
    acct.sessions.obtain = AsyncMock(return_value=(FakeSession(), "s1"))
    gen = openai_mod._stream_openai(
        account=acct,
        pool=MagicMock(),
        existing_sid="s1",
        lock=acct.sem,
        prompt="x",
        model="deepseek-v4-flash",
        model_type="default",
        thinking=False,
        search=False,
        reduced_prompts=[("short prompt", False, {})],
    )
    joined = "".join(await _collect_agen(gen))
    assert '"error"' in joined
    assert "Content is too long" in joined
    assert '"response_incomplete"' in joined
    assert joined.rstrip().endswith("data: [DONE]")


async def test_stream_input_exceeds_reduced_retry_tool_mode():
    acct = FakeAccount([])
    acct.client.completion = AsyncMock(
        side_effect=[
            FakeResp(sse_text=INPUT_SSE),
            openai_mod.HTTPException(404, "gone"),
            FakeResp(sse_text=OK_SSE),
        ]
    )
    acct.sessions.obtain = AsyncMock(return_value=(FakeSession(), "s1"))
    gen = openai_mod._stream_openai(
        account=acct,
        pool=MagicMock(),
        existing_sid="s1",
        lock=acct.sem,
        prompt="x",
        model="deepseek-v4-flash",
        model_type="default",
        thinking=False,
        search=False,
        tool_mode=True,
        reduced_prompts=[("short prompt", False, {})],
    )
    joined = "".join(await _collect_agen(gen))
    assert '"content": "Hi"' in joined
    assert '"finish_reason": "response_incomplete"' in joined
    assert '"error"' in joined
    assert joined.rstrip().endswith("data: [DONE]")


async def test_stream_prepare_session_error():
    acct = FakeAccount([OK_SSE])
    acct.sessions.obtain = AsyncMock(side_effect=openai_mod.HTTPException(401, "bad"))
    gen = openai_mod._stream_openai(
        account=acct,
        pool=MagicMock(),
        existing_sid="s1",
        lock=acct.sem,
        prompt="x",
        model="deepseek-v4-flash",
        model_type="default",
        thinking=False,
        search=False,
    )
    lines = await _collect_agen(gen)
    joined = "".join(lines)
    assert '"error"' in joined
    assert "bad" in joined
    assert joined.rstrip().endswith("data: [DONE]")


def test_stream_emits_usage():
    pool, _ = make_pool()
    app.state.pool = pool
    client = TestClient(app)
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "deepseek-v4-flash",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
            "stream_options": {"include_usage": True},
        },
    )
    assert '"usage"' in resp.text
    client.close()


async def test_deepseek_human_delay_sleeps():
    from danyapi.api.openai import _human_delay

    with patch.object(settings, "human_delay_min", 0.01), patch.object(settings, "human_delay_max", 0.02):
        await _human_delay()


async def test_qwen_human_delay_sleeps():
    from danyapi.qwen.api import _human_delay as qwen_delay
    from danyapi.qwen.api import settings as qwen_settings

    with patch.object(qwen_settings, "human_delay_min", 0.01), patch.object(qwen_settings, "human_delay_max", 0.02):
        await qwen_delay()


async def _collect_agen(agen):
    out = []
    async for item in agen:
        out.append(item)
    return out


def test_content_non_dict_item_skipped():
    req = SimpleNamespace(
        messages=[openai_mod.ChatMessage(role="user", content=[{"type": "image_url", "image_url": {"url": "data:image/png;base64,eA=="}}, 42, "str"])], files=[]
    )
    atts = openai_mod._collect_attachments(req)
    assert len(atts) == 1


def test_chat_deepseek_build_prompt_error():
    pool, _ = make_pool()
    app.state.pool = pool
    client = TestClient(app)
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "deepseek-v4-flash", "messages": [{"role": "user", "content": 42}]},
    )
    assert resp.status_code == 400
    client.close()


async def test_chat_deepseek_busy_non_stream():
    pool, _ = make_pool()
    app.state.pool = pool
    client = TestClient(app)
    with patch("danyapi.api.openai.account_lock", side_effect=AccountPoolBusy()):
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "deepseek-v4-flash", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert resp.status_code == 429
    client.close()


async def test_chat_qwen_with_session_id():
    from danyapi.qwen.client import QwenSession

    acct = MagicMock()
    acct.index = 0
    acct.broken = False
    acct.sem = asyncio.Semaphore(1)
    acct.client = MagicMock()
    acct.client.completion = AsyncMock(
        return_value=FakeResp(
            sse_text=(
                'data: {"response.created":{"chat_id":"c1","parent_id":"p0","response_id":"r1"}} \n\n'
                'data: {"choices": [{"delta": {"content": "Hello", "phase": "answer"}}], "response_id": "r1"}\n\n'
                'data: {"choices": [{"delta": {"status": "finished", "phase": "answer"}}], "response_id": "r1"}\n\n'
            )
        )
    )
    acct.sessions = MagicMock()
    acct.sessions.obtain = AsyncMock(return_value=(QwenSession(id="c1"), "c1"))
    acct.sessions.touch_last_message = MagicMock()
    acct.sessions.forget = MagicMock()
    pool = MagicMock()
    pool.acquire = AsyncMock(return_value=(acct, "sid-q"))
    app.state.qwen_pool = pool
    client = TestClient(app)
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "qwen3.8-max", "messages": [{"role": "user", "content": "hi"}], "session_id": "sid-q"},
    )
    assert resp.status_code == 200
    pool.acquire.assert_awaited_once()
    assert pool.acquire.await_args.args[0] == "sid-q"
    client.close()


def test_chat_qwen_build_prompt_error():
    pool = MagicMock()
    pool.acquire = AsyncMock(return_value=(MagicMock(), None))
    app.state.qwen_pool = pool
    client = TestClient(app)
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "qwen3.8-max", "messages": [{"role": "user", "content": 42}]},
    )
    assert resp.status_code == 400
    client.close()


async def test_chat_qwen_busy_non_stream():
    from danyapi.qwen.client import QwenSession

    acct = MagicMock()
    acct.index = 0
    acct.broken = False
    acct.sem = asyncio.Semaphore(1)
    acct.client = MagicMock()
    acct.sessions = MagicMock()
    acct.sessions.obtain = AsyncMock(return_value=(QwenSession(id="c1"), "c1"))
    acct.sessions.touch_last_message = MagicMock()
    acct.sessions.forget = MagicMock()
    pool = MagicMock()
    pool.acquire = AsyncMock(return_value=(acct, None))
    app.state.qwen_pool = pool
    client = TestClient(app)
    with patch("danyapi.qwen.api.account_lock", side_effect=AccountPoolBusy()):
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "qwen3.8-max", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert resp.status_code == 429
    client.close()


def test_chat_qwen_stream_endpoint():
    from danyapi.qwen.client import QwenSession

    acct = MagicMock()
    acct.index = 0
    acct.broken = False
    acct.sem = asyncio.Semaphore(1)
    acct.client = MagicMock()
    acct.client.completion = AsyncMock(
        return_value=FakeResp(
            sse_text=(
                'data: {"response.created":{"chat_id":"c1","parent_id":"p0","response_id":"r1"}} \n\n'
                'data: {"choices": [{"delta": {"content": "Hello", "phase": "answer"}}], "response_id": "r1"}\n\n'
                'data: {"choices": [{"delta": {"status": "finished", "phase": "answer"}}], "response_id": "r1"}\n\n'
            )
        )
    )
    acct.sessions = MagicMock()
    acct.sessions.obtain = AsyncMock(return_value=(QwenSession(id="c1"), "c1"))
    acct.sessions.touch_last_message = MagicMock()
    acct.sessions.forget = MagicMock()
    pool = MagicMock()
    pool.acquire = AsyncMock(return_value=(acct, None))
    app.state.qwen_pool = pool
    client = TestClient(app)
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "qwen3.8-max", "messages": [{"role": "user", "content": "hi"}], "stream": True},
    )
    assert resp.status_code == 200
    assert '"content": "Hello"' in resp.text
    assert resp.text.rstrip().endswith("data: [DONE]")
    client.close()


def test_lifespan_qwen_invalid_token():
    from contextlib import ExitStack

    from danyapi.qwen.client import QwenClient as QC

    stack = ExitStack()
    for patch_cm in (
        patch.object(settings, "deepseek_tokens", []),
        patch.object(settings, "qwen_tokens", ["bad"]),
        patch.object(QC, "check_auth", new=AsyncMock(return_value=False)),
    ):
        stack.enter_context(patch_cm)
    with stack:
        with pytest.raises(RuntimeError):
            with TestClient(app):
                pass


async def test_deepseek_non_stream_retryable_http():
    acct = FakeAccount([OK_SSE])
    acct.client.completion = AsyncMock(
        side_effect=[
            openai_mod.HTTPException(429, "slow down"),
            FakeResp(sse_text=OK_SSE),
        ]
    )
    acct.sessions.obtain = AsyncMock(return_value=(FakeSession(), "s1"))
    result = await openai_mod._collect_non_stream(
        account=acct,
        pool=MagicMock(),
        existing_sid="s1",
        lock=acct.sem,
        prompt="x",
        model="deepseek-v4-flash",
        model_type="default",
        thinking=False,
        search=False,
    )
    assert result["choices"][0]["message"]["content"] == "Hi"
    assert acct.client.completion.await_count == 2


async def test_continuation_finish_buffer():
    sse = (
        "event: ready\n"
        'data: {"request_message_id":1,"response_message_id":2,"model_type":"default"}\n'
        "\n"
        'data: {"v":{"response":{"message_id":2,"parent_id":1,"status":"WIP","fragments":[{"id":2,"type":"RESPONSE","content":"Tail"}]}}}\n'
        "\n"
        'data: {"p":"response/status","o":"SET","v":"FINISHED"}'
    )
    acct = FakeAccount([sse])
    acct.sessions.obtain = AsyncMock(return_value=(FakeSession(), "s1"))
    rec = await openai_mod._collect_continuation(acct, FakeSession(), None, "default", False, False)
    assert rec is not None
    assert rec.content == "Tail"


async def test_deepseek_non_stream_finish_buffer():
    sse = (
        "event: ready\n"
        'data: {"request_message_id":1,"response_message_id":2,"model_type":"default"}\n'
        "\n"
        'data: {"v":{"response":{"message_id":2,"parent_id":1,"status":"WIP","fragments":[{"id":2,"type":"RESPONSE","content":"Buf"}]}}}\n'
        "\n"
        'data: {"p":"response/status","o":"SET","v":"FINISHED"}'
    )
    acct = FakeAccount([sse])
    acct.sessions.obtain = AsyncMock(return_value=(FakeSession(), "s1"))
    result = await openai_mod._collect_non_stream(
        account=acct,
        pool=MagicMock(),
        existing_sid="s1",
        lock=acct.sem,
        prompt="x",
        model="deepseek-v4-flash",
        model_type="default",
        thinking=False,
        search=False,
    )
    assert result["choices"][0]["message"]["content"] == "Buf"


THINK_SSE = (
    "event: ready\n"
    'data: {"request_message_id":1,"response_message_id":2,"model_type":"default"}\n'
    "\n"
    'data: {"v":{"response":{"message_id":2,"parent_id":1,"status":"WIP","fragments":'
    '[{"id":2,"type":"THINK","content":"why"},{"id":3,"type":"RESPONSE","content":"Answer"}]}}}\n'
    "\n"
    'data: {"p":"response/status","o":"SET","v":"FINISHED"}\n'
    "\n"
)


async def test_non_stream_tool_mode_with_reasoning():
    acct = FakeAccount([THINK_SSE])
    acct.sessions.obtain = AsyncMock(return_value=(FakeSession(), "s1"))
    result = await openai_mod._collect_non_stream(
        account=acct,
        pool=MagicMock(),
        existing_sid="s1",
        lock=acct.sem,
        prompt="x",
        model="deepseek-v4-flash",
        model_type="default",
        thinking=True,
        search=False,
        tool_mode=True,
    )
    message = result["choices"][0]["message"]
    assert message["content"] == "Answer"
    assert message["reasoning_content"] == "why"


async def test_non_stream_reasoning_without_tools():
    acct = FakeAccount([THINK_SSE])
    acct.sessions.obtain = AsyncMock(return_value=(FakeSession(), "s1"))
    result = await openai_mod._collect_non_stream(
        account=acct,
        pool=MagicMock(),
        existing_sid="s1",
        lock=acct.sem,
        prompt="x",
        model="deepseek-v4-flash",
        model_type="default",
        thinking=True,
        search=False,
    )
    message = result["choices"][0]["message"]
    assert message["content"] == "Answer"
    assert message["reasoning_content"] == "why"


async def test_stream_reasoning_delta():
    acct = FakeAccount([THINK_SSE])
    acct.sessions.obtain = AsyncMock(return_value=(FakeSession(), "s1"))
    gen = openai_mod._stream_openai(
        account=acct,
        pool=MagicMock(),
        existing_sid="s1",
        lock=acct.sem,
        prompt="x",
        model="deepseek-v4-flash",
        model_type="default",
        thinking=True,
        search=False,
    )
    joined = "".join(await _collect_agen(gen))
    assert '"reasoning_content": "why"' in joined
    assert '"content": "Answer"' in joined


async def test_non_stream_input_exceeds_continuation_none():
    acct = FakeAccount([INPUT_SSE])
    acct.client.completion = AsyncMock(
        side_effect=[
            FakeResp(sse_text=INPUT_SSE),
            openai_mod.HTTPException(404, "gone"),
        ]
    )
    acct.sessions.obtain = AsyncMock(return_value=(FakeSession(), "s1"))
    with pytest.raises(openai_mod.HTTPException) as excinfo:
        await openai_mod._collect_non_stream(
            account=acct,
            pool=MagicMock(),
            existing_sid="s1",
            lock=acct.sem,
            prompt="x",
            model="deepseek-v4-flash",
            model_type="default",
            thinking=False,
            search=False,
        )
    assert excinfo.value.status_code == 502
    assert "response_incomplete" in excinfo.value.detail


async def test_stream_input_exceeds_tool_mode_continuation():
    acct = FakeAccount([INPUT_SSE, THINK_SSE])
    acct.sessions.obtain = AsyncMock(return_value=(FakeSession(), "s1"))
    gen = openai_mod._stream_openai(
        account=acct,
        pool=MagicMock(),
        existing_sid="s1",
        lock=acct.sem,
        prompt="x",
        model="deepseek-v4-flash",
        model_type="default",
        thinking=True,
        search=False,
        tool_mode=True,
    )
    joined = "".join(await _collect_agen(gen))
    assert '"reasoning_content"' in joined
    assert '"content": "Answer"' in joined


async def test_stream_input_exceeds_continuation_none():
    acct = FakeAccount([INPUT_SSE])
    acct.client.completion = AsyncMock(
        side_effect=[
            FakeResp(sse_text=INPUT_SSE),
            openai_mod.HTTPException(404, "gone"),
        ]
    )
    acct.sessions.obtain = AsyncMock(return_value=(FakeSession(), "s1"))
    gen = openai_mod._stream_openai(
        account=acct,
        pool=MagicMock(),
        existing_sid="s1",
        lock=acct.sem,
        prompt="x",
        model="deepseek-v4-flash",
        model_type="default",
        thinking=False,
        search=False,
    )
    lines = await _collect_agen(gen)
    joined = "".join(lines)
    assert '"error"' in joined
    assert '"response_incomplete"' in joined
    assert joined.rstrip().endswith("data: [DONE]")


def test_reduced_prompt_variants_original_matches():
    msgs = [openai_mod.ChatMessage(role="user", content="hello")]
    original, _ = toolemu.build_prompt(msgs, None, None, False, None)
    variants = openai_mod._reduced_prompt_variants(msgs, [REDUCED_TOOL], None, None, original)
    assert len(variants) == 1
    assert variants[0][1] is True
    assert "get_weather" in variants[0][0]


def test_reduced_prompt_variants_no_user():
    msgs = [openai_mod.ChatMessage(role="assistant", content="hi"), openai_mod.ChatMessage(role="assistant", content="yo")]
    variants = openai_mod._reduced_prompt_variants(msgs, [REDUCED_TOOL], None, None, "original")
    assert len(variants) == 1


async def test_collect_reduced_second_variant_succeeds():
    acct = FakeAccount([OK_SSE])
    acct.client.completion = AsyncMock(
        side_effect=[
            openai_mod.HTTPException(404, "gone"),
            FakeResp(sse_text=OK_SSE),
        ]
    )
    acct.sessions.obtain = AsyncMock(return_value=(FakeSession(), "s1"))
    result = await openai_mod._collect_reduced(acct, MagicMock(), [("p1", False, {}), ("p2", False, {})], "default", False, False)
    assert result is not None
    assert result[0].content == "Hi"


async def test_collect_reduced_input_exceeds_then_success():
    acct = FakeAccount([INPUT_SSE, OK_SSE])
    acct.sessions.obtain = AsyncMock(return_value=(FakeSession(), "s1"))
    result = await openai_mod._collect_reduced(acct, MagicMock(), [("p1", False, {}), ("p2", False, {})], "default", False, False)
    assert result is not None
    assert result[0].content == "Hi"


NO_RID_READY_SSE = 'event: ready\ndata: {"request_message_id":1,"model_type":"default"}\n\ndata: {"p":"response/status","o":"SET","v":"FINISHED"}\n\n'


async def test_non_stream_ready_without_message_id():
    acct = FakeAccount([NO_RID_READY_SSE])
    acct.sessions.obtain = AsyncMock(return_value=(FakeSession(), "s1"))
    result = await openai_mod._collect_non_stream(
        account=acct,
        pool=MagicMock(),
        existing_sid="s1",
        lock=acct.sem,
        prompt="x",
        model="deepseek-v4-flash",
        model_type="default",
        thinking=False,
        search=False,
    )
    assert result["choices"][0]["message"]["content"] == ""


async def test_stream_ready_without_message_id():
    acct = FakeAccount([NO_RID_READY_SSE])
    acct.sessions.obtain = AsyncMock(return_value=(FakeSession(), "s1"))
    gen = openai_mod._stream_openai(
        account=acct,
        pool=MagicMock(),
        existing_sid="s1",
        lock=acct.sem,
        prompt="x",
        model="deepseek-v4-flash",
        model_type="default",
        thinking=False,
        search=False,
    )
    joined = "".join(await _collect_agen(gen))
    assert '"finish_reason": "stop"' in joined


INPUT_CONT_SSE = (
    "event: ready\n"
    'data: {"request_message_id":1,"response_message_id":2,"model_type":"default"}\n'
    "\n"
    'data: {"v":{"response":{"message_id":2,"parent_id":1,"status":"input_exceeds_limit","fragments":[{"id":2,"type":"RESPONSE","content":"part"}]}}}\n'
    "\n"
)


async def test_non_stream_continuation_rounds_exhausted():
    acct = FakeAccount([INPUT_SSE] + [INPUT_CONT_SSE] * openai_mod.MAX_CONTINUE_ROUNDS)
    acct.sessions.obtain = AsyncMock(return_value=(FakeSession(), "s1"))
    with pytest.raises(openai_mod.HTTPException) as excinfo:
        await openai_mod._collect_non_stream(
            account=acct,
            pool=MagicMock(),
            existing_sid="s1",
            lock=acct.sem,
            prompt="x",
            model="deepseek-v4-flash",
            model_type="default",
            thinking=False,
            search=False,
        )
    assert excinfo.value.status_code == 502
    assert "response_incomplete" in excinfo.value.detail
    assert acct.client.completion.await_count == 1 + openai_mod.MAX_CONTINUE_ROUNDS


async def test_stream_continuation_rounds_exhausted():
    acct = FakeAccount([INPUT_SSE] + [INPUT_CONT_SSE] * openai_mod.MAX_CONTINUE_ROUNDS)
    acct.sessions.obtain = AsyncMock(return_value=(FakeSession(), "s1"))
    gen = openai_mod._stream_openai(
        account=acct,
        pool=MagicMock(),
        existing_sid="s1",
        lock=acct.sem,
        prompt="x",
        model="deepseek-v4-flash",
        model_type="default",
        thinking=False,
        search=False,
    )
    joined = "".join(await _collect_agen(gen))
    assert '"content": "part"' in joined
    assert '"error"' in joined
    assert '"response_incomplete"' in joined
    assert joined.rstrip().endswith("data: [DONE]")


THINK_ONLY_SSE = (
    "event: ready\n"
    'data: {"request_message_id":1,"response_message_id":2,"model_type":"default"}\n'
    "\n"
    'data: {"v":{"response":{"message_id":2,"parent_id":1,"status":"FINISHED","fragments":[{"id":2,"type":"THINK","content":"why"}]}}}\n'
    "\n"
)


async def test_stream_input_exceeds_reduced_reasoning_only():
    acct = FakeAccount([])
    acct.client.completion = AsyncMock(
        side_effect=[
            openai_mod.HTTPException(400, INPUT_HTTP_BODY),
            openai_mod.HTTPException(404, "gone"),
            FakeResp(sse_text=THINK_ONLY_SSE),
        ]
    )
    acct.sessions.obtain = AsyncMock(return_value=(FakeSession(), "s1"))
    gen = openai_mod._stream_openai(
        account=acct,
        pool=MagicMock(),
        existing_sid="s1",
        lock=acct.sem,
        prompt="x",
        model="deepseek-v4-flash",
        model_type="default",
        thinking=False,
        search=False,
        reduced_prompts=[("short prompt", False, {})],
    )
    joined = "".join(await _collect_agen(gen))
    assert '"reasoning_content": "why"' in joined
    assert '"content": "Answer"' not in joined
    assert '"response_incomplete"' in joined


TOOL_JSON_SSE = (
    "event: ready\n"
    'data: {"request_message_id":1,"response_message_id":2,"model_type":"default"}\n'
    "\n"
    'data: {"v":{"response":{"message_id":2,"parent_id":1,"status":"FINISHED","fragments":'
    '[{"id":2,"type":"RESPONSE","content":"{\\"tool_calls\\": [{\\"name\\": \\"get_weather\\", \\"arguments\\": {\\"city\\": \\"Moscow\\"}}]}"}]}}}\n'
    "\n"
)


async def test_stream_tool_mode_reduced_emits_tool_calls():
    acct = FakeAccount([])
    acct.client.completion = AsyncMock(
        side_effect=[
            openai_mod.HTTPException(400, INPUT_HTTP_BODY),
            openai_mod.HTTPException(404, "gone"),
            FakeResp(sse_text=TOOL_JSON_SSE),
        ]
    )
    acct.sessions.obtain = AsyncMock(return_value=(FakeSession(), "s1"))
    gen = openai_mod._stream_openai(
        account=acct,
        pool=MagicMock(),
        existing_sid="s1",
        lock=acct.sem,
        prompt="x",
        model="deepseek-v4-flash",
        model_type="default",
        thinking=False,
        search=False,
        tool_mode=True,
        reduced_prompts=[("short prompt", False, {})],
    )
    joined = "".join(await _collect_agen(gen))
    assert '"tool_calls"' in joined
    assert '"finish_reason": "response_incomplete"' in joined
    assert '"error"' in joined
    assert joined.rstrip().endswith("data: [DONE]")
