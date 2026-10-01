import logging

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

import danyapi.api.core as core_mod
import danyapi.api.openai as openai_mod
from danyapi.api.openai import app


def test_request_client_ip_ignores_forwarded_headers_from_a_remote_peer():
    request = _req({"x-forwarded-for": "1.2.3.4, 10.0.0.1"}, "10.0.0.5")
    assert openai_mod._request_client_ip(request) == "10.0.0.5"


def test_request_client_ip_honours_forwarded_headers_from_a_loopback_proxy():
    request = _req({"x-forwarded-for": " 1.2.3.4 , 10.0.0.1", "x-real-ip": "5.6.7.8"}, "127.0.0.1")
    assert openai_mod._request_client_ip(request) == "1.2.3.4"
    assert openai_mod._request_client_ip(_req({"x-forwarded-for": "  ", "x-real-ip": " 5.6.7.8 "}, "127.0.0.1")) == "5.6.7.8"
    assert openai_mod._request_client_ip(_req({"x-real-ip": "5.6.7.8"}, "::1")) == "5.6.7.8"


def test_request_client_ip_remote_peer_wins_over_a_forged_real_ip():
    assert openai_mod._request_client_ip(_req({"x-real-ip": "5.6.7.8"}, "203.0.113.7")) == "203.0.113.7"


def test_request_client_ip_uses_the_peer_when_no_proxy_headers():
    request = _req({}, "10.0.0.5")
    assert openai_mod._request_client_ip(request) == "10.0.0.5"
    assert openai_mod._request_client_ip(_req({}, "127.0.0.1")) == "127.0.0.1"


def test_request_client_ip_no_client():
    request = _req({}, None)
    assert openai_mod._request_client_ip(request) == "-"
    assert openai_mod._request_client_ip(_req({"x-forwarded-for": "1.2.3.4"}, None)) == "-"


def test_request_details_all_fields():
    request = _req({"user-agent": "curl/8.0"})
    payload = {
        "model": "deepseek-v4.1-flash",
        "session_id": "s123",
        "user": "alice",
        "stream": True,
        "messages": [{"role": "user", "content": "hello"}],
    }
    details = openai_mod._request_details(request, payload)
    assert "ua=curl/8.0" in details
    assert "model=deepseek-v4.1-flash" in details
    assert "sid=s123" in details
    assert "user=alice" in details
    assert "stream=1" in details
    assert "msgs=1" in details
    assert "tokens=" in details


def test_request_details_empty_payload():
    request = _req({})
    assert openai_mod._request_details(request, {}) == ""


def test_request_details_non_list_messages():
    request = _req({})
    payload = {"model": "x", "stream": "yes", "messages": "garbage"}
    details = openai_mod._request_details(request, payload)
    assert "model=x" in details
    assert "msgs=" not in details
    assert "stream=1" not in details


def _req(headers, client_host="10.0.0.5"):
    class _Client:
        def __init__(self, host):
            self.host = host

    class _R:
        def __init__(self):
            self.headers = headers
            self.client = _Client(client_host) if client_host else None

    return _R()


def _post_request(body: bytes, content_length: bool = True):
    from starlette.requests import Request

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    headers = [(b"content-type", b"application/json")]
    if content_length:
        headers.append((b"content-length", str(len(body)).encode()))
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/v1/chat/completions",
        "headers": headers,
        "client": ("1.2.3.4", 1234),
        "server": ("localhost", 8008),
        "scheme": "http",
        "query_string": b"",
        "root_path": "",
    }
    return Request(scope, receive=receive)


def test_extract_request_body_valid():
    import asyncio

    request = _post_request(b'{"model": "deepseek-v4.1-flash", "user": "bob"}')
    payload = asyncio.run(openai_mod._extract_request_body(request))
    assert payload == {"model": "deepseek-v4.1-flash", "user": "bob"}


def test_extract_request_body_without_a_declared_length_is_not_read():
    import asyncio

    request = _post_request(b'{"model": "deepseek-v4.1-flash"}', content_length=False)
    request._body = b'{"model": "deepseek-v4.1-flash"}'
    payload = asyncio.run(openai_mod._extract_request_body(request))
    assert payload == {}


def test_extract_request_body_rejects_oversized_declared_length(monkeypatch):
    import asyncio

    monkeypatch.setattr(core_mod, "MAX_REQUEST_BODY", 16)
    request = _post_request(b"x" * 64)
    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(openai_mod._extract_request_body(request))
    assert excinfo.value.status_code == 413
    assert "too large" in excinfo.value.detail


def test_extract_request_body_truncation_is_separate_from_the_rejection_limit(monkeypatch):
    import asyncio

    monkeypatch.setattr(core_mod, "MAX_LOGGED_BODY", 8)
    request = _post_request(b'{"model": "deepseek-v4.1-flash"}')
    assert asyncio.run(openai_mod._extract_request_body(request)) == {}


def test_extract_request_body_ignores_get_and_non_json_bodies():
    import asyncio

    body = b'{"model": "x"}'
    request = _post_request(body)
    request.scope["method"] = "GET"
    assert asyncio.run(openai_mod._extract_request_body(request)) == {}
    request = _post_request(b"not json")
    assert asyncio.run(openai_mod._extract_request_body(request)) == {}
    request = _post_request(b"[1, 2]")
    assert asyncio.run(openai_mod._extract_request_body(request)) == {}


def test_log_fields_are_capped_and_control_characters_are_removed():
    request = _req({"user-agent": "curl/8.0\r\nforged line"})
    details = openai_mod._request_details(request, {"model": "deepseek-v4.1-flash", "user": "bob"})
    assert "\n" not in details
    assert "\r" not in details
    assert "forged line" in details
    assert details.count("model=") == 1

    long_details = openai_mod._request_details(_req({}), {"model": "a" * 5000, "session_id": "b" * 5000, "user": "c" * 5000})
    for key in ("model=", "sid=", "user="):
        value = long_details.split(key, 1)[1].split(" ", 1)[0]
        assert len(value) <= core_mod.MAX_LOGGED_FIELD

    nul_details = openai_mod._request_details(_req({"user-agent": "a\x00b"}), {"model": "m\x07del"})
    assert "\x00" not in nul_details
    assert "\x07" not in nul_details


def test_default_format_declares_the_level_and_the_logger_name():
    import danyapi.logging as logging_mod

    assert "%(levelname)s" in logging_mod.DEFAULT_FORMAT
    assert "%(name)s" in logging_mod.DEFAULT_FORMAT
    record = logging.LogRecord("danyapi.api", logging.WARNING, __file__, 1, "a warning", None, None)
    line = logging_mod.DEFAULT_FORMAT % {
        "asctime": "10:00:00",
        "levelname": record.levelname,
        "name": record.name,
        "message": record.getMessage(),
    }
    assert line == "(10:00:00) WARNING danyapi.api a warning"


def test_logged_line_carries_the_level_and_the_logger_name():
    import danyapi.logging as logging_mod

    record = logging.LogRecord("danyapi.api", logging.WARNING, __file__, 1, "a warning", None, None)
    formatter = logging_mod._EscapingFormatter(logging_mod.DEFAULT_FORMAT, logging_mod.DEFAULT_DATEFMT)
    formatted = formatter.format(record)
    assert " WARNING " in formatted
    assert " danyapi.api " in formatted
    assert formatted.endswith("a warning")


def test_formatter_escapes_control_characters_in_the_message():
    import danyapi.logging as logging_mod

    formatter = logging_mod._EscapingFormatter(logging_mod.DEFAULT_FORMAT, logging_mod.DEFAULT_DATEFMT)
    record = logging.LogRecord("danyapi.api", logging.INFO, __file__, 1, "model=deepseek\nforged line", None, None)
    formatted = formatter.format(record)
    assert formatted.count("\n") == 0
    assert "\\n" in formatted
    assert "forged line" in formatted
    assert logging_mod.escape_control("a\x00b\tc") == "a\\x00b\\tc"


def test_log_requests_success_via_client(caplog):
    app.state.pool = None
    app.state.qwen_pool = None
    with caplog.at_level(logging.INFO, logger="danyapi.api"):
        client = TestClient(app, client=("198.51.100.7", 1234))
        resp = client.get("/health", headers={"x-forwarded-for": "9.9.9.9"})
        client.close()
    assert resp.status_code == 200
    logged = [r for r in caplog.records if r.name == "danyapi.api" and r.getMessage().startswith("GET /health ")]
    assert logged
    assert "198.51.100.7" in logged[0].getMessage()
    assert "9.9.9.9" not in logged[0].getMessage()
    assert "ok" in logged[0].getMessage()


def test_log_requests_failure_via_client(caplog):
    app.state.pool = None
    app.state.qwen_pool = None
    app.state.qwen_models = []
    with caplog.at_level(logging.WARNING, logger="danyapi.api"):
        client = TestClient(app)
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]},
        )
        client.close()
    assert resp.status_code == 404
    logged = [r for r in caplog.records if r.name == "danyapi.api" and r.getMessage().startswith("POST /v1/chat/completions ")]
    assert logged
    assert "status=404" in logged[0].getMessage()
    assert "failed" in logged[0].getMessage()


def test_oversized_body_returns_413_and_is_logged(caplog, monkeypatch):
    monkeypatch.setattr(core_mod, "MAX_REQUEST_BODY", 32)
    app.state.pool = None
    app.state.qwen_pool = None
    with caplog.at_level(logging.WARNING, logger="danyapi.api"):
        client = TestClient(app)
        resp = client.post("/v1/chat/completions", json={"model": "deepseek-v4.1-flash", "messages": [{"role": "user", "content": "x" * 200}]})
        client.close()
    assert resp.status_code == 413
    assert resp.json()["error"]["type"] == "request_too_large"
    logged = [r for r in caplog.records if r.name == "danyapi.api" and r.getMessage().startswith("POST /v1/chat/completions ")]
    assert logged
    assert "status=413" in logged[0].getMessage()
