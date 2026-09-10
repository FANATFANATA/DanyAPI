import logging

from fastapi.testclient import TestClient

import danyapi.api.openai as openai_mod
from danyapi.api.openai import app


def test_request_client_ip_forwarded_first():
    request = _req({"x-forwarded-for": "1.2.3.4, 10.0.0.1"}, "10.0.0.5")
    assert openai_mod._request_client_ip(request) == "1.2.3.4"


def test_request_client_ip_empty_forwarded_falls_back():
    request = _req({"x-forwarded-for": "  ", "x-real-ip": "5.6.7.8"}, "10.0.0.5")
    assert openai_mod._request_client_ip(request) == "5.6.7.8"


def test_request_client_ip_real_ip():
    request = _req({"x-real-ip": "5.6.7.8"}, "10.0.0.5")
    assert openai_mod._request_client_ip(request) == "5.6.7.8"


def test_request_client_ip_client_fallback():
    request = _req({}, "10.0.0.5")
    assert openai_mod._request_client_ip(request) == "10.0.0.5"


def test_request_client_ip_no_client():
    request = _req({}, None)
    assert openai_mod._request_client_ip(request) == "-"


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


def test_extract_request_body_valid():
    import asyncio

    from starlette.requests import Request

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/v1/chat/completions",
        "headers": [(b"content-type", b"application/json")],
        "client": ("1.2.3.4", 1234),
        "server": ("localhost", 8008),
        "scheme": "http",
        "query_string": b"",
        "root_path": "",
    }
    request = Request(scope, receive=None)
    request._body = b'{"model": "deepseek-v4.1-flash", "user": "bob"}'
    payload = asyncio.run(openai_mod._extract_request_body(request))
    assert payload == {"model": "deepseek-v4.1-flash", "user": "bob"}


def test_log_requests_success_via_client(caplog):
    app.state.pool = None
    app.state.qwen_pool = None
    with caplog.at_level(logging.INFO, logger="danyapi.api"):
        client = TestClient(app)
        resp = client.get("/health", headers={"x-forwarded-for": "9.9.9.9"})
        client.close()
    assert resp.status_code == 200
    logged = [r for r in caplog.records if r.name == "danyapi.api" and r.getMessage().startswith("GET /health ")]
    assert logged
    assert "9.9.9.9" in logged[0].getMessage()
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
