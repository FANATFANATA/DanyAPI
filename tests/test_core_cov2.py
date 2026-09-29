import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from starlette.requests import Request

import danyapi.api.core as core
import danyapi.api.openai as openai_mod
from danyapi.api import state as state_mod
from danyapi.api.core import app
from danyapi.store import JsonStore

assert openai_mod.app is app

POOL_ATTRS = state_mod.POOL_ATTRS_BY_PROVIDER
BYOK_PROVIDERS = state_mod.BYOK_PROVIDERS


class _Spec:
    def __init__(self, auth=True, auth_exc=None, close_error=None, construct_error=None):
        self.auth = auth
        self.auth_exc = auth_exc
        self.close_error = close_error
        self.construct_error = construct_error


class _FakeClient:
    def __init__(self, kind, spec):
        self.kind = kind
        self.spec = spec
        self.closed = 0

    async def check_auth(self):
        if self.spec.auth_exc is not None:
            raise self.spec.auth_exc
        return self.spec.auth

    async def aclose(self):
        self.closed += 1
        if self.spec.close_error is not None:
            raise self.spec.close_error


class _FakePool:
    def __init__(self, accounts=(), label=None, healthy=None, stats_error=None, **kwargs):
        self.accounts = list(accounts)
        self.label = label
        self.healthy = len(self.accounts) if healthy is None else healthy
        self.stats_error = stats_error
        self.flushed = 0

    def stats(self):
        if self.stats_error is not None:
            raise self.stats_error
        return {"accounts": len(self.accounts), "healthy": self.healthy, "broken": 0}

    def flush(self):
        self.flushed += 1


_CLIENT_CLASSES = {
    "ds": "DeepSeekClient",
    "qw": "QwenClient",
    "gc": "GigaChatClient",
    "alice": "AliceClient",
    "duckai": "DuckAIClient",
}


class _ClientPlan:
    def __init__(self):
        self.queues = {}
        self.clients = []

    def add(self, kind, *specs):
        self.queues[kind] = list(specs)
        return self

    def install(self, monkeypatch):
        for kind, specs in self.queues.items():
            plan = self
            pending = list(specs)

            def build(_pending=pending, _kind=kind, _plan=plan, **kwargs):
                spec = _pending.pop(0)
                if spec.construct_error is not None:
                    raise spec.construct_error
                client = _FakeClient(_kind, spec)
                _plan.clients.append(client)
                return client

            monkeypatch.setattr(core, _CLIENT_CLASSES[kind], build)

        async def refresh_models():
            return None

        async def model_refresh_loop():
            await asyncio.Event().wait()

        monkeypatch.setattr(core, "refresh_models", refresh_models)
        monkeypatch.setattr(core, "model_refresh_loop", model_refresh_loop)
        return self

    def closed_counts(self):
        return [client.closed for client in self.clients]


@asynccontextmanager
async def _run_lifespan():
    async with core.lifespan(app):
        yield


class _PoolFactory:
    def __init__(self):
        self.pools = []

    def __call__(self, accounts, label=None, **kwargs):
        pool = _FakePool(accounts, label=label, **kwargs)
        self.pools.append(pool)
        return pool


@pytest.fixture(autouse=True)
def _isolated_state(monkeypatch):
    monkeypatch.setattr(core.settings, "deepseek_tokens", [])
    monkeypatch.setattr(core.settings, "qwen_tokens", [])
    monkeypatch.setattr(core.settings, "gigachat_keys", [])
    monkeypatch.setattr(core.settings, "alice_enabled", False)
    monkeypatch.setattr(core.settings, "alice_accounts", 0)
    monkeypatch.setattr(core.settings, "duckai_enabled", False)
    monkeypatch.setattr(core.settings, "duckai_accounts", 0)
    monkeypatch.setattr(core.settings, "cache_enabled", False)
    monkeypatch.setattr(core.settings, "usage_enabled", False)
    monkeypatch.setattr(core.settings, "byok", False)
    monkeypatch.setattr(core, "_POOL_RATE_CACHE", {})
    saved_state = dict(app.state._state)
    for attr in POOL_ATTRS.values():
        setattr(app.state, attr, None)
    app.state.byok_pools = {}
    yield
    app.state._state.clear()
    app.state._state.update(saved_state)


def _request(method="POST", path="/v1/chat/completions", body=b"", headers=None, client_host="10.0.0.5", chunks=None):
    state = {"read": False, "index": 0}

    async def receive():
        state["read"] = True
        if chunks is None:
            return {"type": "http.request", "body": body, "more_body": False}
        index = state["index"]
        state["index"] = index + 1
        if index >= len(chunks):
            return {"type": "http.request", "body": b"", "more_body": False}
        return {"type": "http.request", "body": chunks[index], "more_body": index + 1 < len(chunks)}

    raw = [(b"content-type", b"application/json")]
    supplied = dict(headers or {})
    if "content-length" not in supplied:
        supplied["content-length"] = str(len(body))
    for key, value in supplied.items():
        raw.append((key.encode(), str(value).encode()))
    scope = {
        "type": "http",
        "method": method,
        "path": path,
        "headers": raw,
        "client": (client_host, 1234) if client_host else None,
        "server": ("localhost", 8008),
        "scheme": "http",
        "query_string": b"",
        "root_path": "",
    }
    return Request(scope, receive=receive), state


def _json_request(payload, **kwargs):
    body = json.dumps(payload).encode()
    return _request(body=body, headers={"content-length": str(len(body))}, **kwargs)


class _FakeRequest:
    def __init__(self, headers=None, client_host="10.0.0.5"):
        self.headers = headers or {}
        if client_host is None:
            self.client = None
        else:
            self.client = type("_Client", (), {"host": client_host})()


def test_iter_pools_stops_when_byok_pools_is_not_a_mapping():
    app.state.byok_pools = ["not", "a", "mapping"]
    assert list(core._iter_pools()) == []


def test_iter_pools_yields_named_pools_and_byok_pools():
    shared = _FakePool()
    app.state.pool = _FakePool()
    app.state.qwen_pool = shared
    app.state.byok_pools = {"deepseek": {"k1": _FakePool()}, "qwen": "ignored"}
    assert len(list(core._iter_pools())) == 3


def test_flush_state_stores_reports_every_failure(caplog):
    class _BoomTracker:
        def flush(self):
            raise RuntimeError("usage flush failed")

        def snapshot(self):
            raise RuntimeError("usage snapshot failed")

    class _BoomStore:
        def flush(self):
            raise RuntimeError("store flush failed")

        def get(self, *args, **kwargs):
            raise RuntimeError("store get failed")

        def discard(self, *args, **kwargs):
            raise RuntimeError("store discard failed")

    class _BoomPool(_FakePool):
        def flush(self):
            raise RuntimeError("pool flush failed")

    class _SilentPool:
        def __init__(self):
            self.accounts = []

    shared_pool = _BoomPool()
    app.state.usage = _BoomTracker()
    app.state.deepseek_session_store = _BoomStore()
    app.state.qwen_session_store = None
    app.state.responses_store = _BoomStore()
    app.state.pool = shared_pool
    app.state.qwen_pool = shared_pool
    app.state.gigachat_pool = _SilentPool()
    with caplog.at_level(logging.DEBUG, logger="danyapi.api"):
        core._flush_state_stores()
    messages = [record.getMessage() for record in caplog.records]
    assert messages[0] == "usage flush failed: usage flush failed"
    assert "store flush failed for deepseek_session_store: store flush failed" in messages
    assert messages.count("pool flush failed: pool flush failed") == 1
    assert shared_pool.flushed == 0


def test_flush_state_stores_skips_absent_stores_and_pools(caplog):
    class _Tracker:
        def flush(self):
            return None

    app.state.usage = _Tracker()
    for attr in core._STATE_STORE_ATTRS:
        if hasattr(app.state, attr):
            delattr(app.state, attr)
    with caplog.at_level(logging.DEBUG, logger="danyapi.api"):
        core._flush_state_stores()
    assert caplog.records == []


def test_token_stable_id_is_a_stable_sha1_prefix():
    assert core._token_stable_id("abc") == core._token_stable_id("abc")
    assert len(core._token_stable_id("abc")) == 16
    assert core._token_stable_id("abc") != core._token_stable_id("abd")


async def test_lifespan_initialises_byok_state_and_disables_usage(monkeypatch):
    monkeypatch.setattr(core.settings, "deepseek_tokens", ["tok"])
    monkeypatch.setattr(core, "AccountPool", _PoolFactory())
    _ClientPlan().add("ds", _Spec()).install(monkeypatch)
    async with _run_lifespan():
        assert app.state.usage is None
        assert app.state.byok is False
        assert app.state.byok_pools == {provider: {} for provider in BYOK_PROVIDERS}
        assert app.state.byok_auth == {provider: {} for provider in BYOK_PROVIDERS}
        assert app.state.byok_stores == {provider: {} for provider in BYOK_PROVIDERS}
        assert all(isinstance(lock, asyncio.Lock) for lock in app.state.byok_locks.values())
        assert app.state.responses_store is not None
        assert app.state.responses_store.enabled is False
        for provider in BYOK_PROVIDERS:
            assert getattr(app.state, state_mod.MODEL_ATTRS[provider]) == []


async def test_lifespan_reports_gigachat_key_position_not_client_position(monkeypatch, caplog):
    monkeypatch.setattr(core.settings, "gigachat_keys", ["key0", "key1", "key2"])
    monkeypatch.setattr(core.settings, "deepseek_tokens", ["tok"])
    monkeypatch.setattr(core, "AccountPool", _PoolFactory())
    plan = _ClientPlan().add("ds", _Spec())
    plan.add("gc", _Spec(construct_error=OSError("ca bundle missing")), _Spec(auth=False), _Spec(auth=True))
    plan.install(monkeypatch)
    with caplog.at_level(logging.WARNING, logger="danyapi.api"):
        async with _run_lifespan():
            assert len(app.state.gigachat_pool.accounts) == 1
            assert app.state.gigachat_pool.accounts[0].stable_id == core._token_stable_id("key2")
    messages = [record.getMessage() for record in caplog.records]
    assert "gigachat client disabled, CA unusable: ca bundle missing" in messages
    assert "gigachat key #1 invalid/expired, skipping" in messages
    assert "gigachat key #0 invalid/expired, skipping" not in messages
    assert plan.closed_counts() == [1, 1, 1]


async def test_lifespan_builds_alice_and_duckai_accounts(monkeypatch, caplog):
    monkeypatch.setattr(core.settings, "alice_enabled", True)
    monkeypatch.setattr(core.settings, "alice_accounts", 2)
    monkeypatch.setattr(core.settings, "duckai_enabled", True)
    monkeypatch.setattr(core.settings, "duckai_accounts", 1)
    monkeypatch.setattr(core.settings, "deepseek_tokens", ["tok"])
    monkeypatch.setattr(core, "AccountPool", _PoolFactory())
    plan = _ClientPlan().add("ds", _Spec())
    plan.add("alice", _Spec(auth=False), _Spec(auth=True))
    plan.add("duckai", _Spec(auth=False))
    plan.install(monkeypatch)
    with caplog.at_level(logging.INFO, logger="danyapi.api"):
        async with _run_lifespan():
            assert len(app.state.alice_pool.accounts) == 1
            assert len(app.state.duckai_pool.accounts) == 1
    messages = [record.getMessage() for record in caplog.records]
    assert "alice endpoint unreachable, skipping account #0" in messages
    assert "duckai bot check missed on account #0, keeping it and relying on retries" in messages
    assert "alice accounts ready: 1" in messages
    assert "duckai accounts ready: 1" in messages
    assert plan.closed_counts() == [1, 1, 1, 1]


async def test_lifespan_leaves_unused_pools_none(monkeypatch):
    monkeypatch.setattr(core.settings, "deepseek_tokens", ["tok"])
    monkeypatch.setattr(core, "AccountPool", _PoolFactory())
    _ClientPlan().add("ds", _Spec()).install(monkeypatch)
    async with _run_lifespan():
        assert app.state.qwen_pool is None
        assert app.state.gigachat_pool is None
        assert app.state.alice_pool is None
        assert app.state.duckai_pool is None
        assert app.state.pool is not None


async def test_lifespan_skips_a_provider_whose_auth_check_raised(monkeypatch, caplog):
    monkeypatch.setattr(core.settings, "deepseek_tokens", ["bad"])
    monkeypatch.setattr(core.settings, "qwen_tokens", ["good"])
    monkeypatch.setattr(core, "AccountPool", _PoolFactory())
    _ClientPlan().add("ds", _Spec(auth_exc=RuntimeError("network down"))).add("qw", _Spec()).install(monkeypatch)
    with caplog.at_level(logging.WARNING, logger="danyapi.api"):
        async with _run_lifespan():
            assert app.state.pool is None
            assert len(app.state.qwen_pool.accounts) == 1
    messages = [record.getMessage() for record in caplog.records]
    assert "auth check #0 failed: network down" in messages
    assert "deepseek token #0 invalid/expired, skipping" in messages


async def test_lifespan_shutdown_closes_the_http_client_and_survives_a_failed_close(monkeypatch, caplog):
    monkeypatch.setattr(core.settings, "deepseek_tokens", ["tok", "tok2"])
    monkeypatch.setattr(core, "AccountPool", _PoolFactory())

    class _HttpClient:
        closed = 0

        async def aclose(self):
            self.closed += 1

    http_client = _HttpClient()
    app.state.http_client = http_client
    plan = _ClientPlan().add("ds", _Spec(close_error=RuntimeError("close failed")), _Spec()).install(monkeypatch)
    with caplog.at_level(logging.WARNING, logger="danyapi.api"):
        async with _run_lifespan():
            assert len(app.state.pool.accounts) == 2
    assert http_client.closed == 1
    assert plan.closed_counts() == [1, 1]
    assert "client close failed for acct#0: close failed" in [record.getMessage() for record in caplog.records]


async def test_lifespan_without_credentials_raises(monkeypatch):
    monkeypatch.setattr(core, "AccountPool", _PoolFactory())
    _ClientPlan().install(monkeypatch)
    with pytest.raises(RuntimeError, match="no valid credentials"):
        async with _run_lifespan():
            pass


async def test_lifespan_in_byok_mode_needs_no_credentials(monkeypatch):
    monkeypatch.setattr(core.settings, "byok", True)
    monkeypatch.setattr(core, "AccountPool", _PoolFactory())
    _ClientPlan().install(monkeypatch)
    async with _run_lifespan():
        assert app.state.byok is True
        assert app.state.pool is None


async def test_lifespan_starts_the_usage_tracker_when_enabled(monkeypatch):
    monkeypatch.setattr(core.settings, "deepseek_tokens", ["tok"])
    monkeypatch.setattr(core.settings, "usage_enabled", True)
    monkeypatch.setattr(core, "AccountPool", _PoolFactory())
    _ClientPlan().add("ds", _Spec()).install(monkeypatch)
    async with _run_lifespan():
        assert app.state.usage is not None
        assert app.state.usage.snapshot()["totals"] == {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        assert app.state.usage._store.enabled is False


def test_declared_body_length_handles_missing_and_invalid_headers():
    assert core._declared_body_length(_request(headers={"content-length": "not-a-number"})[0]) == -1
    assert core._declared_body_length(_request(headers={"content-length": ""})[0]) == -1
    assert core._declared_body_length(_request(headers={"content-length": "42"})[0]) == 42


async def test_read_request_body_rejects_an_over_limit_declared_length_without_buffering():
    request, state = _request(headers={"content-length": "104857600"}, body=b"x" * 16)
    with pytest.raises(HTTPException) as excinfo:
        await core._read_request_body(request, 1024)
    assert excinfo.value.status_code == 413
    assert excinfo.value.detail == "request body too large"
    assert state["read"] is False


async def test_read_request_body_catches_an_understated_declared_length():
    request, _ = _request(headers={"content-length": "8"}, chunks=[b"x" * 16, b"y" * 16])
    with pytest.raises(HTTPException) as excinfo:
        await core._read_request_body(request, 24)
    assert excinfo.value.status_code == 413
    assert excinfo.value.detail == "request body too large"


async def test_read_request_body_bounds_a_genuine_stream():
    request, _ = _request(headers={"content-length": "6"}, chunks=[b"ab", b"cd", b"ef"])
    assert await core._read_request_body(request, 6) == b"abcdef"
    assert request._body == b"abcdef"


async def test_read_request_body_rejects_a_cached_body_over_the_limit():
    request, state = _request(body=b"x" * 32)
    request._body = b"x" * 32
    with pytest.raises(HTTPException) as excinfo:
        await core._read_request_body(request, 8)
    assert excinfo.value.status_code == 413
    assert state["read"] is False


async def test_read_request_body_returns_a_cached_body_under_the_limit():
    request, _ = _request(body=b"abc")
    request._body = b"abc"
    assert await core._read_request_body(request, 8) == b"abc"


def test_parse_logged_body_bounds_and_rejects_every_shape():
    assert core._parse_logged_body(b"") == {}
    assert core._parse_logged_body(b"x" * (core.MAX_LOGGED_BODY + 1)) == {}
    assert core._parse_logged_body(b"{not json") == {}
    assert core._parse_logged_body(b"\xff\xfe") == {}
    assert core._parse_logged_body(memoryview(b'{"a": 1}')) == {}
    assert core._parse_logged_body(b"[1, 2]") == {}
    assert core._parse_logged_body(b'{"a": 1}') == {"a": 1}


async def test_extract_request_body_rejects_over_the_declared_limit(monkeypatch):
    monkeypatch.setattr(core, "MAX_REQUEST_BODY", 16)
    request, state = _request(headers={"content-length": "64"}, body=b"x" * 64)
    with pytest.raises(HTTPException) as excinfo:
        await core._extract_request_body(request)
    assert excinfo.value.status_code == 413
    assert state["read"] is False


async def test_extract_request_body_uses_the_cached_body(monkeypatch):
    monkeypatch.setattr(core, "MAX_REQUEST_BODY", 1024)
    request, state = _request(body=b'{"model": "cached"}')
    request._body = b'{"model": "cached"}'
    assert await core._extract_request_body(request) == {"model": "cached"}
    assert state["read"] is False


async def test_extract_request_body_reraises_the_incremental_cap(monkeypatch):
    monkeypatch.setattr(core, "MAX_REQUEST_BODY", 32)
    request, _ = _request(headers={"content-length": "4"}, chunks=[b"x" * 64, b"y" * 64])
    with pytest.raises(HTTPException) as excinfo:
        await core._extract_request_body(request)
    assert excinfo.value.status_code == 413


async def test_extract_request_body_ignores_a_body_over_the_log_limit(monkeypatch):
    monkeypatch.setattr(core, "MAX_LOGGED_BODY", 8)
    request, _ = _request(body=b'{"model": "deepseek-v4.1-flash"}')
    assert await core._extract_request_body(request) == {}


async def test_extract_request_body_swallows_a_transport_failure():
    async def receive():
        raise OSError("connection reset")

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/v1/chat/completions",
        "headers": [(b"content-length", b"4")],
        "client": ("10.0.0.5", 1),
        "server": ("localhost", 8008),
        "scheme": "http",
        "query_string": b"",
        "root_path": "",
    }
    assert await core._extract_request_body(Request(scope, receive=receive)) == {}


def test_request_client_ip_prefers_the_first_forwarded_hop():
    request = _FakeRequest({"x-forwarded-for": " 1.2.3.4 , 10.0.0.1", "x-real-ip": "5.6.7.8"})
    assert core._request_client_ip(request) == "1.2.3.4"


def test_request_client_ip_falls_back_to_x_real_ip():
    assert core._request_client_ip(_FakeRequest({"x-forwarded-for": "  ", "x-real-ip": " 5.6.7.8 "})) == "5.6.7.8"


def test_request_client_ip_uses_the_peer_when_no_proxy_headers():
    assert core._request_client_ip(_FakeRequest()) == "10.0.0.5"


def test_log_field_replaces_control_characters_and_caps_the_length():
    assert core._log_field("a\nb\tc\x00d\x7fe") == "a b c d e"
    assert core._log_field("m" * 500) == "m" * core.MAX_LOGGED_FIELD


def test_request_details_reports_every_sanitised_field():
    request = _FakeRequest({"user-agent": "curl/8.0\r\nforged"})
    payload = {
        "model": "deepseek-v4.1-flash",
        "session_id": "sid-1",
        "user": "alice",
        "stream": False,
        "messages": [{"role": "user", "content": "hello"}, {"role": "user", "content": "world"}],
    }
    details = core._request_details(request, payload)
    assert details.split() == ["ua=curl/8.0", "forged", "model=deepseek-v4.1-flash", "sid=sid-1", "user=alice", "stream=0", "msgs=2", "tokens=8"]


def test_request_details_skips_token_counting_when_asked():
    assert core._request_details(_FakeRequest({}), {"messages": [{"role": "user", "content": "hello"}]}, count_tokens=False) == "msgs=1"


def test_log_request_failure_is_silent_below_warning(caplog):
    with caplog.at_level(logging.ERROR, logger="danyapi.api"):
        core._log_request_failure(_FakeRequest({}), {}, 1.0, status=500)
    assert caplog.records == []


def test_log_request_success_is_silent_below_info(caplog):
    with caplog.at_level(logging.WARNING, logger="danyapi.api"):
        core._log_request_success(_FakeRequest({}), {}, 1.0)
    assert caplog.records == []


def test_log_request_success_reports_the_path_and_peer(caplog):
    class _Request(_FakeRequest):
        url = type("_Url", (), {"path": "/v1/chat/completions"})()
        method = "POST"

    with caplog.at_level(logging.INFO, logger="danyapi.api"):
        core._log_request_success(_Request({}), {"model": "m"}, 12.4)
    assert caplog.records[0].getMessage().startswith("POST /v1/chat/completions 10.0.0.5 model=m ok (")


def test_log_request_failure_reports_the_exception(caplog):
    class _Request(_FakeRequest):
        url = type("_Url", (), {"path": "/v1/chat/completions"})()
        method = "POST"

    with caplog.at_level(logging.WARNING, logger="danyapi.api"):
        core._log_request_failure(_Request({}), {}, 3.0, exc=RuntimeError("boom"))
    assert "error=boom" in caplog.records[0].getMessage()


def test_middleware_rejects_an_oversized_declared_length(monkeypatch, caplog):
    monkeypatch.setattr(core, "MAX_REQUEST_BODY", 8)
    with caplog.at_level(logging.WARNING, logger="danyapi.api"):
        client = TestClient(app)
        response = client.post("/v1/chat/completions", content=b"x" * 200, headers={"content-type": "application/json"})
        client.close()
    assert response.status_code == 413
    body = response.json()["error"]
    assert body["message"] == "request body too large"
    assert body["type"] == "request_too_large"
    assert response.headers["x-request-id"] == body["request_id"]
    assert "status=413" in next(record.getMessage() for record in caplog.records)


def test_error_type_for_status_covers_every_bucket():
    assert core._error_type_for_status(401) == "authentication_error"
    assert core._error_type_for_status(403) == "permission_error"
    assert core._error_type_for_status(404) == "not_found_error"
    assert core._error_type_for_status(408) == "request_timeout"
    assert core._error_type_for_status(409) == "conflict_error"
    assert core._error_type_for_status(413) == "request_too_large"
    assert core._error_type_for_status(429) == "rate_limit_error"
    assert core._error_type_for_status(501) == "api_error"
    assert core._error_type_for_status(503) == "api_error"
    assert core._error_type_for_status(502) == "server_error"
    assert core._error_type_for_status(504) == "server_error"
    assert core._error_type_for_status(500) == "server_error"
    assert core._error_type_for_status(418) == "invalid_request_error"


def test_error_code_for_status_only_codes_rate_limit_and_bad_request():
    assert core._error_code_for_status(429) == "rate_limit_exceeded"
    assert core._error_code_for_status(400) == "invalid_request_error"
    assert core._error_code_for_status(500) is None


def test_exception_message_logs_and_returns_the_constant(caplog):
    with caplog.at_level(logging.ERROR, logger="danyapi.api"):
        message = core._exception_message(RuntimeError("secret internal detail"))
    assert message == core.INTERNAL_ERROR_MESSAGE == "internal server error"
    assert "unhandled api error: secret internal detail" in caplog.records[0].getMessage()


def test_error_detail_shape():
    assert core._error_detail("m") == {"error": {"message": "m", "finish_reason": None}}
    assert core._error_detail("m", "length")["error"]["finish_reason"] == "length"


def test_openai_error_payload_without_a_request_id():
    assert core._openai_error_payload(400, "m") == {"error": {"message": "m", "type": "invalid_request_error", "param": None, "code": "invalid_request_error"}}
    assert core._openai_error_payload(500, "m", "rid")["error"]["request_id"] == "rid"


def test_validation_summary_falls_back_for_unusable_error_lists():
    assert core._validation_summary("not a list") == "request validation failed"
    assert core._validation_summary([]) == "request validation failed"
    assert core._validation_summary(["not a dict"]) == "request validation failed"
    assert core._validation_summary([{"msg": "no loc"}]) == "no loc"


def test_validation_summary_bounds_loc_and_message():
    summary = core._validation_summary([{"loc": ("body", "messages"), "msg": "x" * 200, "input": "y" * 100}])
    assert summary == "body.messages: " + "x" * 64
    assert len(summary) == 79


def test_validation_summary_counts_the_overflow():
    errors = [{"loc": ("body", f"f{index}"), "msg": f"m{index}"} for index in range(12)]
    summary = core._validation_summary(errors)
    assert summary == "; ".join(f"body.f{index}: m{index}" for index in range(10)) + "; and 2 more"


def test_validation_summary_ignores_non_string_loc_parts():
    assert core._validation_summary([{"loc": ("body", None, 3, "x"), "msg": "bad"}]) == "body.3.x: bad"


def test_validation_error_body_is_bounded_and_names_the_field():
    huge = "m" * 200000
    client = TestClient(app)
    response = client.post("/v1/chat/completions", json={"model": "deepseek-v4.1-flash", "messages": huge})
    client.close()
    assert response.status_code == 400
    text = response.text
    assert len(text) < 1024
    assert huge not in text
    assert "body.messages" in text
    assert response.headers["x-request-id"] == response.json()["error"]["request_id"]


def test_client_request_id_is_sanitised_and_capped():
    value = core._client_request_id(_FakeRequest({"x-request-id": "c" * 200 + "\nforged"}))
    assert value == "c" * 128
    assert core._client_request_id(_FakeRequest({"x-client-request-id": "cid"})) == "cid"
    assert core._client_request_id(_FakeRequest({})) is None


def test_request_id_header_is_always_server_generated():
    request = _FakeRequest({"x-request-id": "client-chosen"})
    first = core._request_id_header(request)
    second = core._request_id_header(request)
    assert first != second
    assert len(first) == 32
    assert "client-chosen" not in first


def test_response_headers_echo_the_client_id_only_in_the_client_header():
    headers = core._response_headers(_FakeRequest({"x-request-id": "client-chosen"}), "server-id")
    assert headers == {"x-request-id": "server-id", "x-client-request-id": "client-chosen"}
    assert core._response_headers(_FakeRequest({}), "server-id") == {"x-request-id": "server-id"}


async def test_http_exception_handler_merges_only_the_envelope_extra_keys():
    detail = {"error": {"message": "upstream said no", "type": "forged", "code": "forged_code", "param": "forged", "finish_reason": "length"}}
    response = await core._on_http_exception(_FakeRequest({}), HTTPException(502, detail))
    body = json.loads(response.body)["error"]
    assert body["message"] == "upstream said no"
    assert body["type"] == "server_error"
    assert body["code"] is None
    assert body["param"] is None
    assert body["finish_reason"] == "length"


async def test_http_exception_handler_uses_the_detail_itself_without_an_inner_error():
    response = await core._on_http_exception(_FakeRequest({}), HTTPException(400, {"message": "flat detail"}))
    assert json.loads(response.body)["error"]["message"] == "flat detail"


async def test_http_exception_handler_stringifies_a_detail_without_a_message():
    response = await core._on_http_exception(_FakeRequest({}), HTTPException(400, {"code": 7}))
    assert json.loads(response.body)["error"]["message"] == "{'code': 7}"


async def test_http_exception_handler_copies_explicit_headers():
    response = await core._on_http_exception(_FakeRequest({}), HTTPException(429, "slow down", headers={"retry-after": "12"}))
    assert response.headers["retry-after"] == "12"
    assert json.loads(response.body)["error"]["code"] == "rate_limit_exceeded"


async def test_uncaught_exception_handler_returns_the_constant_message(caplog):
    with caplog.at_level(logging.ERROR, logger="danyapi.api"):
        response = await core._on_uncaught_exception(_FakeRequest({}), RuntimeError("internal boom"))
    body = json.loads(response.body)
    assert response.status_code == 500
    assert body["error"]["message"] == core.INTERNAL_ERROR_MESSAGE
    assert body["error"]["type"] == "server_error"
    assert "internal boom" in caplog.records[0].getMessage()


def test_account_busy_count_counts_locked_semaphores():
    class _Locked:
        def __init__(self):
            self.sem = asyncio.Semaphore(0)

    class _Free:
        def __init__(self):
            self.sem = asyncio.Semaphore(1)

    class _NoSem:
        sem = None

    assert core._account_busy_count(_FakePool([_Locked(), _Free(), _NoSem()])) == 1
    assert core._account_busy_count(_FakePool([])) == 0


def test_pool_rate_headers_cache_per_pool():
    pool = _FakePool(healthy=3)
    first = core._pool_rate_headers(pool)
    second = core._pool_rate_headers(pool)
    assert first is second
    assert first["x-ratelimit-limit-requests"] == "3"
    assert first["x-ratelimit-remaining-requests"] == "3"
    assert int(first["x-ratelimit-reset-requests"]) >= int(time.time()) + core._POOL_RATE_RESET_SEC - 2


def test_pool_rate_headers_subtract_busy_accounts():
    class _Busy:
        def __init__(self):
            self.sem = asyncio.Semaphore(0)

    pool = _FakePool([_Busy(), _Busy()], healthy=4)
    headers = core._pool_rate_headers(pool)
    assert headers["x-ratelimit-limit-requests"] == "4"
    assert headers["x-ratelimit-remaining-requests"] == "2"


def test_pool_rate_headers_absent_for_none_and_broken_pools():
    assert core._pool_rate_headers(None) == {}
    assert core._pool_rate_headers(object()) == {}
    assert core._pool_rate_headers(_FakePool(stats_error=RuntimeError("stats failed"))) == {}


def test_pool_rate_headers_survives_a_non_weakrefable_pool():
    class _NoWeak:
        __slots__ = ("accounts",)

        def __init__(self):
            self.accounts = []

        def stats(self):
            return {"healthy": 2}

    pool = _NoWeak()
    core._POOL_RATE_CACHE[id(pool)] = (0.0, {"stale": "1"}, lambda: None)
    headers = core._pool_rate_headers(pool)
    assert headers["x-ratelimit-limit-requests"] == "2"
    assert id(pool) not in core._POOL_RATE_CACHE


def test_pool_rate_cache_drops_expired_and_dead_entries():
    alive = _FakePool()
    core._POOL_RATE_CACHE[12345] = (0.0, {"a": "1"}, lambda: None)
    core._POOL_RATE_CACHE[67890] = (0.0, {"b": "2"}, lambda: alive)
    core._POOL_RATE_CACHE[22222] = (10**6 - 0.5, {"c": "3"}, lambda: alive)
    core._prune_pool_rate_cache(10**6)
    assert list(core._POOL_RATE_CACHE) == [22222]


def test_pool_rate_cache_evicts_only_the_oldest_entry():
    keep = _FakePool()
    for index in range(core._POOL_RATE_CACHE_MAX - 1):
        core._POOL_RATE_CACHE[1000 + index] = (100.0 + index, {}, lambda: keep)
    core._POOL_RATE_CACHE[9999] = (1e9, {}, lambda: keep)
    assert len(core._POOL_RATE_CACHE) == core._POOL_RATE_CACHE_MAX
    core._prune_pool_rate_cache(100.9)
    assert len(core._POOL_RATE_CACHE) == core._POOL_RATE_CACHE_MAX - 1
    assert 1000 not in core._POOL_RATE_CACHE
    assert 1001 in core._POOL_RATE_CACHE
    assert 9999 in core._POOL_RATE_CACHE


async def test_rate_limit_pool_picks_the_qwen_pool_for_images_and_videos():
    qwen = _FakePool()
    app.state.qwen_pool = qwen
    assert await core._rate_limit_pool(_request(path="/v1/images/generations")[0]) is qwen
    assert await core._rate_limit_pool(_request(path="/v1/videos/clip")[0]) is qwen


@pytest.mark.parametrize(
    ("provider", "model"),
    [
        ("deepseek", "deepseek-v4.1-flash"),
        ("qwen", "qwen3-max"),
        ("gigachat", "GigaChat-Pro"),
        ("alice", "yagpt"),
        ("duckai", "duckai-free"),
    ],
)
async def test_rate_limit_pool_picks_the_pool_of_the_requested_model(provider, model):
    pool = _FakePool()
    setattr(app.state, POOL_ATTRS[provider], pool)
    request, _ = _json_request({"model": model}, path="/v1/chat/completions")
    assert await core._rate_limit_pool(request) is pool


async def test_rate_limit_pool_returns_none_for_unknown_and_typeless_models():
    assert await core._rate_limit_pool(_json_request({"model": 7})[0]) is None
    assert await core._rate_limit_pool(_json_request({"model": "nope-1"})[0]) is None
    assert await core._rate_limit_pool(_json_request({})[0]) is None


async def test_rate_limit_pool_ignores_unlisted_paths():
    assert await core._rate_limit_pool(_request(method="GET", path="/v1/models")[0]) is None


async def test_rate_limit_pool_swallows_a_body_rejection(monkeypatch):
    async def boom(request):
        raise HTTPException(413, "request body too large")

    monkeypatch.setattr(core, "_extract_request_body", boom)
    request, _ = _json_request({"model": "deepseek-v4.1-flash"})
    assert await core._rate_limit_pool(request) is None


def test_pool_attrs_are_derived_from_the_provider_mapping():
    assert core.POOL_ATTRS == ("pool", "qwen_pool", "gigachat_pool", "alice_pool", "duckai_pool")
    assert core.POOL_ATTRS == tuple(POOL_ATTRS[provider] for provider in BYOK_PROVIDERS)


def test_docs_assets_allowlist_matches_the_mounted_dashboard():
    assert core.DOCS_ASSETS == frozenset({"index.html", "style.css", "script.js", "deepseek-logo.svg", "qwen-logo.svg"})


def test_docs_mount_uses_the_allowlisting_subclass():
    from fastapi.staticfiles import StaticFiles

    route = next(route for route in app.routes if getattr(route, "name", "") == "docs")
    assert isinstance(route.app, StaticFiles)
    assert type(route.app).__name__ == "_DocsAssets"


def test_shared_store_reuses_the_state_attribute():
    app.state.core_cov2_store = None
    try:
        first = core._shared_store("core_cov2_store", "core-cov2")
        assert first is core._shared_store("core_cov2_store", "core-cov2")
        assert first.enabled is False
    finally:
        del app.state.core_cov2_store


def test_responses_store_shares_the_state_store():
    saved = getattr(app.state, "responses_store", None)
    try:
        app.state.responses_store = JsonStore("core-cov2-responses", None)
        assert core._responses_store() is app.state.responses_store
    finally:
        if saved is None:
            del app.state.responses_store
        else:
            app.state.responses_store = saved
