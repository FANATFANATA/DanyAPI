import base64
import json
from collections.abc import Callable

import httpx
import pytest

from danyapi.duckai import attest
from danyapi.duckai.client import (
    BASE_URL,
    CHAT_PATH,
    STATUS_PATH,
    USER_AGENT,
    DuckAIClient,
    DuckAIError,
    _iter_lines,
    _parse_json,
)

ATTESTATION = {
    "server_hashes": ["a==", "b==", "c=="],
    "client_hashes": ["ua", "1", "0"],
    "signals": {},
    "meta": {"v": "4", "challenge_id": "cid", "timestamp": "1", "debug": "d"},
}

SCRIPT = b"(async function(){return 1})()"
REFRESHED_SCRIPT = b"(async function(){return 2})()"

STREAM_HEADERS = {"content-type": "text/event-stream"}


def _stub_solve(monkeypatch) -> list[str]:
    seen: list[str] = []

    async def _fake(script: str, user_agent: str) -> dict:
        seen.append(script)
        return dict(ATTESTATION)

    monkeypatch.setattr(attest, "evaluate", _fake)
    return seen


def _stream_body(lines: list[str]) -> bytes:
    return ("\n".join(f"data: {line}" for line in lines) + "\n\n").encode()


def _client(
    chat_handler: Callable[[httpx.Request], httpx.Response] | None,
    *,
    status_handler: Callable[[httpx.Request], httpx.Response] | None = None,
    seen: list[httpx.Request] | None = None,
) -> DuckAIClient:
    seen = seen if seen is not None else []

    def router(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path == STATUS_PATH:
            if status_handler is not None:
                return status_handler(request)
            return httpx.Response(200, json={"status": "0", "statusV2": 0})
        if chat_handler is None:
            return httpx.Response(200, content=_stream_body(["[DONE]"]), headers=STREAM_HEADERS)
        return chat_handler(request)

    client = DuckAIClient()
    client.http = httpx.AsyncClient(transport=httpx.MockTransport(router), base_url=BASE_URL)
    return client


def test_default_headers_ask_for_the_vqd():
    client = DuckAIClient()
    try:
        headers = client._headers()
        assert headers["x-vqd-accept"] == "1"
        assert headers["cache-control"] == "no-store"
    finally:
        pass


@pytest.mark.asyncio
async def test_status_reads_status_payload(monkeypatch):
    _stub_solve(monkeypatch)
    seen: list[httpx.Request] = []
    client = _client(None, seen=seen)
    try:
        payload = await client.status()
        assert payload["status"] == "0"
        assert seen[0].url.path == STATUS_PATH
        assert seen[0].headers["x-vqd-accept"] == "1"
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_status_solves_attestation_from_response_header(monkeypatch):
    scripts = _stub_solve(monkeypatch)
    encoded = base64.b64encode(SCRIPT).decode()
    client = _client(None, status_handler=lambda request: httpx.Response(200, json={"status": "0"}, headers={attest.JSA_HEADER: encoded}))
    try:
        await client.status()
        assert scripts == [SCRIPT.decode()]
        decoded = json.loads(base64.b64decode(client._jsa))
        assert decoded["meta"]["origin"] == BASE_URL
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_status_raises_on_error_payload(monkeypatch):
    _stub_solve(monkeypatch)
    client = _client(None, status_handler=lambda request: httpx.Response(503, json={"type": "ERR_SERVICE_UNAVAILABLE", "message": "down"}))
    try:
        with pytest.raises(DuckAIError) as excinfo:
            await client.status()
        assert excinfo.value.code == "ERR_SERVICE_UNAVAILABLE"
        assert excinfo.value.message == "down"
        assert excinfo.value.is_retryable is True
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_status_raises_on_non_json(monkeypatch):
    _stub_solve(monkeypatch)
    client = _client(None, status_handler=lambda request: httpx.Response(200, text="<html>"))
    try:
        with pytest.raises(DuckAIError):
            await client.status()
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_status_raises_on_bare_status_code(monkeypatch):
    _stub_solve(monkeypatch)
    client = _client(None, status_handler=lambda request: httpx.Response(500, json={"message": "boom"}))
    try:
        with pytest.raises(DuckAIError) as excinfo:
            await client.status()
        assert excinfo.value.code == 500
        assert excinfo.value.message == "boom"
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_check_auth_true_and_false(monkeypatch):
    _stub_solve(monkeypatch)
    good = _client(None)
    bad = _client(None, status_handler=lambda request: httpx.Response(200, text="<html>"))
    try:
        assert await good.check_auth() is True
        assert await bad.check_auth() is False
    finally:
        await good.aclose()
        await bad.aclose()


@pytest.mark.asyncio
async def test_check_auth_survives_an_unsolvable_attestation(monkeypatch):
    _stub_solve(monkeypatch)
    client = _client(
        None, status_handler=lambda request: httpx.Response(200, json={"status": "0"}, headers={attest.JSA_HEADER: base64.b64encode(SCRIPT).decode()})
    )

    async def _boom(script: str, user_agent: str) -> dict:
        raise attest.AttestationError("unsupported fragment")

    monkeypatch.setattr(attest, "evaluate", _boom)
    try:
        assert await client.check_auth() is True
        assert client._jsa == attest.INITIAL_JSA
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_status_tolerates_an_unsolvable_attestation(monkeypatch):
    _stub_solve(monkeypatch)
    client = _client(
        None, status_handler=lambda request: httpx.Response(200, json={"status": "0"}, headers={attest.JSA_HEADER: base64.b64encode(SCRIPT).decode()})
    )

    async def _boom(script: str, user_agent: str) -> dict:
        raise attest.AttestationError("unsupported fragment")

    monkeypatch.setattr(attest, "evaluate", _boom)
    try:
        assert await client.status() == {"status": "0"}
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_attestation_is_cached(monkeypatch):
    scripts = _stub_solve(monkeypatch)
    client = _client(None)
    try:
        client._jsa = "cached-value"
        assert await client.attestation() == "cached-value"
        assert scripts == []
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_invalidate_attestation_resets_to_initial():
    client = _client(None)
    try:
        client._jsa = "something"
        client.invalidate_attestation()
        assert client._jsa == attest.INITIAL_JSA
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_fetch_models_returns_the_catalog():
    client = _client(None)
    try:
        models = await client.fetch_models()
        assert {entry["owned_by"] for entry in models} == {"duckai"}
        assert all(entry["model_type"] == "chat" for entry in models)
        assert all(entry["provider"] for entry in models)
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_chat_streams_assistant_deltas(monkeypatch):
    _stub_solve(monkeypatch)
    seen: list[httpx.Request] = []
    body = _stream_body(
        [
            json.dumps({"action": "success", "role": "assistant", "message": "he"}),
            json.dumps({"action": "success", "role": "assistant", "message": "llo"}),
            "[DONE]",
        ]
    )
    client = _client(lambda request: httpx.Response(200, content=body, headers=STREAM_HEADERS), seen=seen)
    try:
        deltas = [event.delta async for event in client.chat([{"role": "user", "content": [{"type": "text", "text": "hi"}]}])]
        assert "".join(deltas) == "hello"
        chat = next(request for request in seen if request.url.path == CHAT_PATH)
        assert chat.headers["accept"] == "text/event-stream"
        assert chat.headers["x-vqd-accept"] == "1"
        assert attest.JSA_HEADER in chat.headers
        assert "x-fe-signals" in chat.headers
        sent = json.loads(chat.content)
        assert sent["model"] == "gpt-5.4-mini"
        assert sent["messages"][0]["role"] == "user"
        assert sent["canUseTools"] is False
        assert sent["canShowGreeting"] is False
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_chat_picks_up_a_new_attestation_header(monkeypatch):
    scripts = _stub_solve(monkeypatch)
    refreshed = base64.b64encode(REFRESHED_SCRIPT).decode()
    client = _client(lambda request: httpx.Response(200, content=_stream_body(["[DONE]"]), headers={**STREAM_HEADERS, attest.JSA_HEADER: refreshed}))
    try:
        client._jsa = "old-value"
        _ = [event async for event in client.chat([{"role": "user", "content": []}])]
        assert scripts == [REFRESHED_SCRIPT.decode()]
        assert client._jsa != "old-value"
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_chat_keeps_the_old_attestation_when_the_refresh_fails(monkeypatch):
    _stub_solve(monkeypatch)
    client = _client(lambda request: httpx.Response(200, content=_stream_body(["[DONE]"]), headers={**STREAM_HEADERS, attest.JSA_HEADER: "not-base64!!"}))
    try:
        client._jsa = "old-value"
        _ = [event async for event in client.chat([{"role": "user", "content": []}])]
        assert client._jsa == "old-value"
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_chat_raises_on_error_events(monkeypatch):
    _stub_solve(monkeypatch)
    body = _stream_body([json.dumps({"action": "error", "type": "ERR_UPSTREAM", "message": "nope"})])
    client = _client(lambda request: httpx.Response(200, content=body, headers=STREAM_HEADERS))
    try:
        with pytest.raises(DuckAIError) as excinfo:
            _ = [event async for event in client.chat([{"role": "user", "content": []}])]
        assert excinfo.value.code == "ERR_UPSTREAM"
        assert excinfo.value.message == "nope"
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_chat_ignores_unknown_actions(monkeypatch):
    _stub_solve(monkeypatch)
    body = _stream_body([json.dumps({"action": "keepalive"}), json.dumps({"action": "success", "role": "assistant", "content": "c"}), "[DONE]"])
    client = _client(lambda request: httpx.Response(200, content=body, headers=STREAM_HEADERS))
    try:
        deltas = [event.delta async for event in client.chat([{"role": "user", "content": []}])]
        assert "".join(deltas) == "c"
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_chat_raises_on_http_error_status(monkeypatch):
    _stub_solve(monkeypatch)
    client = _client(lambda request: httpx.Response(418, json={"type": "ERR_CHALLENGE", "cd": {"gk": "abc"}}))
    try:
        with pytest.raises(DuckAIError) as excinfo:
            _ = [event async for event in client.chat([{"role": "user", "content": []}])]
        assert excinfo.value.is_challenge is True
        assert "abc" in excinfo.value.message
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_chat_raises_on_http_error_without_json(monkeypatch):
    _stub_solve(monkeypatch)
    client = _client(lambda request: httpx.Response(500, text="oops"))
    try:
        with pytest.raises(DuckAIError) as excinfo:
            _ = [event async for event in client.chat([{"role": "user", "content": []}])]
        assert excinfo.value.code == 500
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_chat_wraps_transport_errors(monkeypatch):
    _stub_solve(monkeypatch)

    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route", request=request)

    client = _client(boom)
    try:
        with pytest.raises(DuckAIError) as excinfo:
            _ = [event async for event in client.chat([{"role": "user", "content": []}])]
        assert excinfo.value.code == 502
        assert "no route" in excinfo.value.message
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_chat_body_respects_tool_search_and_effort(monkeypatch):
    _stub_solve(monkeypatch)
    seen: list[httpx.Request] = []
    client = _client(None, seen=seen)
    try:
        _ = [
            event
            async for event in client.chat(
                [{"role": "user", "content": []}],
                model="claude-opus-4-8",
                effort="medium",
                can_use_tools=True,
                can_use_web_search=True,
            )
        ]
        chat = next(request for request in seen if request.url.path == CHAT_PATH)
        sent = json.loads(chat.content)
        assert sent["canUseTools"] is True
        assert sent["canUseWebSearch"] is True
        assert sent["reasoningEffort"] == "medium"
        assert chat.headers["user-agent"] == USER_AGENT
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_iter_lines_skips_sse_metadata_and_flushes_the_tail():
    class _Resp:
        def __init__(self, chunks: list[str]) -> None:
            self._chunks = chunks

        async def aiter_text(self):
            for chunk in self._chunks:
                yield chunk

    lines = [line async for line in _iter_lines(_Resp(["event: ping\ndata: a\n\nid: 7\nretry: 5\n", "data: b"]))]
    assert lines == ["a", "b"]


def test_parse_json_ignores_non_objects():
    assert _parse_json("not json") is None
    assert _parse_json("[1,2]") is None
    assert _parse_json('{"a":1}') == {"a": 1}


def test_raise_for_payload_falls_back_to_status_text():
    with pytest.raises(DuckAIError) as excinfo:
        DuckAIClient._raise_for_payload(500, None)
    assert excinfo.value.message == "upstream returned 500"
    assert excinfo.value.is_retryable is True


def test_duckai_error_flags():
    auth = DuckAIError(401, "nope")
    assert auth.is_auth is True
    assert auth.is_challenge is False
    challenge = DuckAIError(418, "")
    assert challenge.is_challenge is True
    assert challenge.is_auth is False
    assert challenge.is_retryable is False
