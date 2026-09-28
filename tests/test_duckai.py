import base64
import json
from typing import Any

import pytest
from fastapi.testclient import TestClient

from danyapi.accounts import AccountPool
from danyapi.api.models import _resolve_provider
from danyapi.api.openai import app
from danyapi.api.schemas import ChatMessage
from danyapi.duckai import api as duckai_api
from danyapi.duckai import attest
from danyapi.duckai.accounts import DuckAIAccount
from danyapi.duckai.api import BLOCKED_HINT, build_messages
from danyapi.duckai.client import (
    DEFAULT_MODEL,
    MODEL_CATALOG,
    DuckAIError,
    DuckAIEvent,
    model_efforts,
    normalize_effort,
    normalize_finish_reason,
    parse_control,
    parse_event,
)

MODEL = "gpt-5.4-mini"


@pytest.fixture(autouse=True)
def _no_backoff(monkeypatch):
    async def _instant(attempt: int) -> None:
        return None

    monkeypatch.setattr(duckai_api, "_sleep_backoff", _instant)
    monkeypatch.setattr(duckai_api, "ATTESTATION_RETRY_DELAY", 0.0)


@pytest.fixture(autouse=True)
def _clean():
    saved = {attr: getattr(app.state, attr, None) for attr in ("duckai_pool", "duckai_models")}
    for attr in saved:
        setattr(app.state, attr, None)
    app.state.byok = False
    yield
    for attr, value in saved.items():
        setattr(app.state, attr, value)


def _event(**kwargs) -> DuckAIEvent:
    event = DuckAIEvent()
    for key, value in kwargs.items():
        setattr(event, key, value)
    return event


class _StubClient:
    def __init__(self, events: list[DuckAIEvent] | None = None, error: DuckAIError | None = None) -> None:
        self.events = events if events is not None else [_event(delta="ok")]
        self.error = error
        self.calls: list[dict] = []
        self.closed = False

    async def chat(self, messages, model=DEFAULT_MODEL, effort=None, *, can_use_tools=False, can_use_web_search=False):
        self.calls.append(
            {
                "messages": messages,
                "model": model,
                "effort": effort,
                "can_use_tools": can_use_tools,
                "can_use_web_search": can_use_web_search,
            }
        )
        if self.error is not None:
            raise self.error
        for event in self.events:
            yield event

    def invalidate_attestation(self) -> None:
        pass

    async def aclose(self) -> None:
        self.closed = True


def _pool(client: Any) -> AccountPool:
    return AccountPool([DuckAIAccount(0, client, stable_id="duckai")], label="duckai")


def _post(payload: dict):
    return TestClient(app).post("/v1/chat/completions", json=payload)


def test_resolve_provider_routes_duckai():
    for entry in MODEL_CATALOG:
        assert _resolve_provider(entry["id"]) == "duckai"
    assert _resolve_provider("gpt-5.4-mini") == "duckai"
    assert _resolve_provider("duckai") == "duckai"
    assert _resolve_provider("DuckAI") == "duckai"


def test_resolve_provider_does_not_hijack_foreign_model_names():
    from fastapi import HTTPException

    for name in ("gpt-4o", "gpt-4", "claude-3-haiku-20240307", "llama-3.1-70b"):
        with pytest.raises(HTTPException) as excinfo:
            _resolve_provider(name)
        assert excinfo.value.status_code == 404


def test_resolve_provider_still_routes_other_providers():
    assert _resolve_provider("deepseek-v4.1-flash") == "deepseek"
    assert _resolve_provider("qwen3.8-max") == "qwen"
    assert _resolve_provider("GigaChat") == "gigachat"
    assert _resolve_provider("alice") == "alice"


def test_handler_registry_covers_duckai():
    from danyapi.api.chats import CHAT_HANDLERS

    assert CHAT_HANDLERS["duckai"] == "_chat_completions_duckai"
    assert set(CHAT_HANDLERS) == {"deepseek", "qwen", "gigachat", "alice", "duckai"}


def test_health_reports_duckai():
    body = TestClient(app).get("/health").json()
    assert "duckai" in body
    assert body["duckai"] is False


def test_models_endpoint_includes_duckai():
    app.state.duckai_models = [{"id": entry["id"], "name": entry["name"], "owned_by": "duckai", "model_type": "chat"} for entry in MODEL_CATALOG]
    data = TestClient(app).get("/v1/models").json()["data"]
    owners = {model["id"]: model["owned_by"] for model in data}
    assert owners[MODEL] == "duckai"
    assert owners["claude-opus-4-8"] == "duckai"


def test_chat_completions_503_when_duckai_not_configured():
    resp = _post({"model": MODEL, "messages": [{"role": "user", "content": "hi"}]})
    assert resp.status_code == 503
    assert "DUCKAI_ENABLED" in resp.json()["error"]["message"]


def test_pool_with_only_blocked_account_returns_429():
    client = _StubClient()
    pool = _pool(client)
    pool.accounts[0].mark_broken()
    app.state.duckai_pool = pool
    resp = _post({"model": MODEL, "messages": [{"role": "user", "content": "hi"}]})
    assert resp.status_code == 429


def test_challenge_is_retried_and_does_not_break_the_account():
    client = _StubClient(error=DuckAIError(418, "ERR_CHALLENGE"))
    app.state.duckai_pool = _pool(client)
    resp = _post({"model": MODEL, "messages": [{"role": "user", "content": "hi"}]})
    assert resp.status_code == 403
    assert "did not pass DuckDuckGo's check" in resp.json()["error"]["message"]
    assert len(client.calls) == 6, "a refused attestation must be retried with a fresh one"
    assert app.state.duckai_pool.accounts[0].broken is False
    assert app.state.duckai_pool.accounts[0].sem.locked() is False


def test_challenge_never_leaves_the_pool_exhausted():
    client = _StubClient(error=DuckAIError(418, "ERR_CHALLENGE"))
    app.state.duckai_pool = _pool(client)
    for _ in range(3):
        resp = _post({"model": MODEL, "messages": [{"role": "user", "content": "hi"}]})
        assert resp.status_code == 403
    resp = _post({"model": MODEL, "messages": [{"role": "user", "content": "hi"}]})
    assert resp.status_code == 403


def test_chat_completions_rejects_file_attachments():
    app.state.duckai_pool = _pool(_StubClient())
    resp = _post(
        {
            "model": MODEL,
            "messages": [{"role": "user", "content": "hi"}],
            "files": [{"name": "a.txt", "content": "eA=="}],
        }
    )
    assert resp.status_code == 400
    assert "file attachments" in resp.json()["error"]["message"]


def test_duckai_chat_completion_end_to_end():
    client = _StubClient([_event(delta="при"), _event(delta="вет"), _event(finish="stop")])
    app.state.duckai_pool = _pool(client)
    resp = _post({"model": MODEL, "messages": [{"role": "user", "content": "hi"}]})
    assert resp.status_code == 200
    body = resp.json()
    assert body["choices"][0]["message"]["content"] == "привет"
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["usage"]["total_tokens"] > 0
    assert client.calls[0]["model"] == MODEL
    assert client.calls[0]["messages"][0]["role"] == "user"


def test_duckai_stream_end_to_end():
    client = _StubClient([_event(delta="a"), _event(delta="b"), _event(finish="stop")])
    app.state.duckai_pool = _pool(client)
    resp = _post({"model": MODEL, "messages": [{"role": "user", "content": "hi"}], "stream": True})
    assert resp.status_code == 200
    assert '"content":"a"' in resp.text.replace(" ", "")
    assert '"content":"b"' in resp.text.replace(" ", "")
    assert "data: [DONE]" in resp.text


def test_duckai_stream_emits_usage_when_requested():
    client = _StubClient([_event(delta="x"), _event(finish="stop")])
    app.state.duckai_pool = _pool(client)
    resp = _post(
        {
            "model": MODEL,
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    )
    assert resp.status_code == 200
    assert '"usage"' in resp.text


def test_duckai_surfaces_reasoning_and_tool_calls():
    call = {
        "index": 0,
        "id": "call_1",
        "type": "function",
        "function": {"name": "lookup", "arguments": '{"q":"x"}'},
    }
    client = _StubClient(
        [
            _event(reasoning="thinking"),
            _event(delta="answer"),
            _event(tool_calls=[call]),
            _event(finish="stop"),
        ]
    )
    app.state.duckai_pool = _pool(client)
    resp = _post({"model": MODEL, "messages": [{"role": "user", "content": "hi"}]})
    assert resp.status_code == 200
    message = resp.json()["choices"][0]["message"]
    assert message["content"] == "answer"
    assert message["reasoning_content"] == "thinking"
    assert message["tool_calls"][0]["function"]["name"] == "lookup"
    assert resp.json()["choices"][0]["finish_reason"] == "tool_calls"


def test_duckai_challenge_maps_to_403_with_hint():
    client = _StubClient(error=DuckAIError(418, "ERR_CHALLENGE"))
    app.state.duckai_pool = _pool(client)
    resp = _post({"model": MODEL, "messages": [{"role": "user", "content": "hi"}]})
    assert resp.status_code == 403
    assert resp.json()["error"]["message"] == BLOCKED_HINT


def test_duckai_stream_reports_challenge_as_sse_error():
    client = _StubClient(error=DuckAIError(418, "ERR_CHALLENGE"))
    app.state.duckai_pool = _pool(client)
    resp = _post({"model": MODEL, "messages": [{"role": "user", "content": "hi"}], "stream": True})
    assert resp.status_code == 200
    assert "ERR_CHALLENGE" not in resp.text
    assert "did not pass DuckDuckGo's check" in resp.text
    assert "data: [DONE]" in resp.text


def test_duckai_applies_stop_sequences():
    client = _StubClient([_event(delta="keep this DROP"), _event(finish="stop")])
    app.state.duckai_pool = _pool(client)
    resp = _post({"model": MODEL, "messages": [{"role": "user", "content": "hi"}], "stop": ["DROP"]})
    assert resp.json()["choices"][0]["message"]["content"] == "keep this "


def test_duckai_completions_endpoint():
    client = _StubClient([_event(delta="done"), _event(finish="stop")])
    app.state.duckai_pool = _pool(client)
    resp = TestClient(app).post("/v1/completions", json={"model": MODEL, "prompt": "hi"})
    assert resp.status_code == 200
    assert resp.json()["choices"][0]["text"] == "done"


def test_anthropic_messages_accepts_duckai_model():
    client = _StubClient([_event(delta="hi"), _event(finish="stop")])
    app.state.duckai_pool = _pool(client)
    resp = TestClient(app).post(
        "/v1/messages",
        json={"model": MODEL, "max_tokens": 64, "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 200
    assert resp.json()["content"][0]["text"] == "hi"


def test_build_messages_folds_system_into_first_user_turn():
    built = build_messages(
        [
            ChatMessage(role="system", content="be terse"),
            ChatMessage(role="user", content="hi"),
        ]
    )
    assert len(built) == 1
    parts = built[0]["content"]
    assert parts[0]["type"] == "text"
    assert "be terse" in parts[0]["text"]
    assert parts[1] == {"type": "text", "text": "hi"}


def test_build_messages_keeps_assistant_history_as_parts():
    built = build_messages(
        [
            ChatMessage(role="user", content="hi"),
            ChatMessage(role="assistant", content="hello"),
            ChatMessage(role="user", content="again"),
        ]
    )
    assert [message["role"] for message in built] == ["user", "assistant", "user"]
    assert built[1]["content"] == ""
    assert built[1]["parts"] == [{"type": "text", "text": "hello"}]


def test_build_messages_passes_inline_images():
    built = build_messages(
        [
            ChatMessage(
                role="user",
                content=[
                    {"type": "text", "text": "what is this"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                ],
            )
        ]
    )
    parts = built[0]["content"]
    assert parts[0]["text"] == "what is this"
    assert parts[1]["type"] == "image"
    assert parts[1]["mimeType"] == "image/png"


def test_build_messages_rejects_remote_images():
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as excinfo:
        build_messages([ChatMessage(role="user", content=[{"type": "image_url", "image_url": {"url": "https://x/y.png"}}])])
    assert excinfo.value.status_code == 400


def test_build_messages_caps_images_per_request():
    from fastapi import HTTPException

    content = [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}} for _ in range(11)]
    with pytest.raises(HTTPException) as excinfo:
        build_messages([ChatMessage(role="user", content=content)])
    assert excinfo.value.status_code == 400


def test_build_messages_attaches_tool_results_to_assistant_turn():
    built = build_messages(
        [
            ChatMessage(role="user", content="hi"),
            ChatMessage(
                role="assistant",
                content="",
                tool_calls=[{"id": "call_1", "type": "function", "function": {"name": "f", "arguments": "{}"}}],
            ),
            ChatMessage(role="tool", tool_call_id="call_1", content="42"),
            ChatMessage(role="user", content="thanks"),
        ]
    )
    roles = [message["role"] for message in built]
    assert roles == ["user", "assistant", "user"]
    kinds = [part["type"] for part in built[1]["parts"]]
    assert kinds == ["tool-call", "tool-result"]
    assert built[1]["parts"][0]["toolName"] == "f"
    assert built[1]["parts"][1]["result"] == "42"


def test_build_messages_injects_tool_catalog():
    built = build_messages(
        [ChatMessage(role="user", content="hi")],
        tools=[{"type": "function", "function": {"name": "f", "description": "does f", "parameters": {"type": "object"}}}],
    )
    text = built[0]["content"][0]["text"]
    assert "does f" in text
    assert '"type":"object"' in text


def test_build_messages_never_returns_empty():
    built = build_messages([])
    assert built[0]["role"] == "user"
    assert built[0]["content"][0]["text"] == "Hello"


def test_build_messages_enables_native_tools_on_the_client():
    client = _StubClient()
    app.state.duckai_pool = _pool(client)
    tools = [{"type": "function", "function": {"name": "f", "parameters": {}}}]
    resp = _post({"model": MODEL, "messages": [{"role": "user", "content": "hi"}], "tools": tools})
    assert resp.status_code == 200
    assert client.calls[0]["can_use_tools"] is True


def test_search_flag_reaches_the_client():
    client = _StubClient()
    app.state.duckai_pool = _pool(client)
    resp = _post({"model": MODEL, "messages": [{"role": "user", "content": "hi"}], "search": True})
    assert resp.status_code == 200
    assert client.calls[0]["can_use_web_search"] is True


def test_parse_event_reads_assistant_delta():
    event = parse_event({"action": "success", "role": "assistant", "message": "hi"})
    assert event.delta == "hi"
    assert event.finish is None


def test_parse_event_reads_reasoning_delta():
    event = parse_event({"action": "success", "role": "reasoning", "state": "text-delta", "text": "why"})
    assert event.reasoning == "why"
    assert event.delta == ""


def test_parse_event_reads_tool_call():
    event = parse_event(
        {
            "action": "success",
            "role": "tool-invocation",
            "state": "call",
            "toolCallId": "c1",
            "toolName": "WebSearch",
            "toolArguments": {"query": "x"},
        }
    )
    assert event.tool_calls[0]["id"] == "c1"
    assert event.tool_calls[0]["function"]["name"] == "WebSearch"
    assert json.loads(event.tool_calls[0]["function"]["arguments"]) == {"query": "x"}


def test_parse_event_reads_sources_and_refusal():
    sourced = parse_event({"action": "success", "role": "source", "source": {"url": "https://a", "title": "A"}})
    assert sourced.sources == [{"url": "https://a", "title": "A", "site": ""}]
    refused = parse_event({"action": "success", "role": "refusal", "detectedBy": "model"})
    assert refused.refusal == "model"


def test_parse_control_handles_ping_and_title():
    assert parse_control("[PING]") is None
    title = parse_control("[CHAT_TITLE:Hello there]")
    assert title is not None
    assert title.title == "Hello there"


def test_parse_control_handles_done_with_stop_reason():
    done = parse_control("[DONE][STOP_REASON:max_tokens]")
    assert done is not None
    assert done.finish == "length"
    limited = parse_control("[DONE][LIMIT_CONVERSATION]")
    assert limited is not None
    assert limited.limit == "ERR_CONVERSATION_LIMIT"


def test_parse_control_ignores_json_lines():
    assert parse_control('{"action":"success","role":"assistant","message":"x"}') is None


def test_normalize_finish_reason_maps_upstream_values():
    assert normalize_finish_reason("max_tokens") == "length"
    assert normalize_finish_reason("end_turn") == "stop"
    assert normalize_finish_reason(None) == "stop"
    assert normalize_finish_reason("whatever") == "stop"


def test_normalize_effort_respects_model_support():
    assert normalize_effort("gpt-5.4-mini", "medium") == "medium"
    assert normalize_effort("gpt-5.4-mini", "high") == "none"
    assert normalize_effort("claude-sonnet-4-6", "medium") == "none"
    assert normalize_effort("claude-opus-4-8", "medium") == "medium"
    assert normalize_effort("gpt-5.4-mini", None) == "none"
    assert "low" in model_efforts("claude-sonnet-4-6")


def test_attestation_client_hashes_are_sha256_base64():
    hashed = attest.client_hashes(["abc"])
    assert hashed == [base64.b64encode(bytes.fromhex("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")).decode()]


def test_attestation_decode_script_rejects_garbage():
    with pytest.raises(attest.AttestationError):
        attest.decode_script("not base64!!")
    with pytest.raises(attest.AttestationError):
        attest.decode_script("")


def test_attestation_decode_script_round_trips():
    payload = b"(async function(){return 1})()"
    assert attest.decode_script(base64.b64encode(payload).decode()) == payload.decode()


def test_attestation_build_header_shape():
    header = attest.build_header(
        {
            "server_hashes": ["a==", "b==", "c=="],
            "client_hashes": ["ua", "1", "0"],
            "signals": {},
            "meta": {"v": "4", "challenge_id": "cid", "timestamp": "1", "debug": "d"},
        },
        duration_ms=12,
    )
    decoded = json.loads(base64.b64decode(header))
    assert list(decoded) == ["server_hashes", "client_hashes", "signals", "meta"]
    assert list(decoded["meta"]) == ["v", "challenge_id", "timestamp", "debug", "origin", "stack", "duration"]
    assert decoded["meta"]["origin"] == attest.ORIGIN
    assert decoded["meta"]["duration"] == "12"
    assert len(decoded["client_hashes"]) == 3


def test_attestation_build_header_validates_payload():
    for payload in ({}, {"client_hashes": []}, {"client_hashes": ["x"]}, {"client_hashes": ["x"], "server_hashes": ["a"]}):
        with pytest.raises(attest.AttestationError):
            attest.build_header(payload)


def test_attestation_header_for_falls_back_to_initial():
    assert attest.INITIAL_JSA == "initial"


def test_fraud_signals_is_base64_json():
    decoded = json.loads(base64.b64decode(attest.fraud_signals()))
    assert decoded["events"] == []
    assert isinstance(decoded["start"], int)
    assert decoded["end"] >= 0


def test_account_label_and_mark_broken():
    account = DuckAIAccount(2, _StubClient())
    assert account.label == "duckai-acct#2"
    assert account.broken is False
    account.mark_broken()
    assert account.broken is True
    assert account.broken_at is not None
    account.mark_broken()


def test_provider_api_is_importable():
    assert callable(duckai_api.collect_non_stream)
    assert callable(duckai_api.stream_openai)
