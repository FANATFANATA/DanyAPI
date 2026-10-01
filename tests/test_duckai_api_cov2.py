import asyncio
import base64
import json
from types import SimpleNamespace

import httpx
import pytest
from fastapi import HTTPException

import danyapi.api.retry as retry_mod
from danyapi.api.retry import MAX_RETRIES, RETRY_BACKOFF_JITTER, RETRY_BACKOFF_MAX_SEC, RETRY_BACKOFF_SEC
from danyapi.api.schemas import ChatMessage
from danyapi.duckai import api as duckai_api
from danyapi.duckai import attest
from danyapi.duckai.accounts import DuckAIAccount
from danyapi.duckai.api import (
    ALLOWED_IMAGE_MIME_TYPES,
    ATTESTATION_RETRY_DELAY,
    BLOCKED_HINT,
    ENTRYPOINT_HINT,
    MAX_IMAGE_BYTES,
    MAX_IMAGES_PER_MESSAGE,
    MAX_IMAGES_PER_REQUEST,
    NO_IMAGE_HINT,
    RETRY_JITTER,
    _detail_for,
    _images_of,
    _prompt_text,
    _status_for,
    _text_of,
    _tool_calls_of,
    _tool_specs,
    build_messages,
    collect_non_stream,
    stream_openai,
)
from danyapi.duckai.client import DEFAULT_MODEL, DuckAIError, DuckAIEvent

MODEL = "gpt-5.4-mini"

_REAL_SLEEP_BACKOFF = duckai_api._sleep_backoff


def _neutral_jitter(low: float, high: float) -> float:
    return (low + high) / 2


def _retry_delay_bounds(attempt: int) -> tuple[float, float]:
    base = min(RETRY_BACKOFF_MAX_SEC, RETRY_BACKOFF_SEC * 2 ** (attempt - 1))
    return max(0.0, base * (1 - RETRY_BACKOFF_JITTER)), min(RETRY_BACKOFF_MAX_SEC, base * (1 + RETRY_BACKOFF_JITTER))


class SleepRecorder:
    def __init__(self):
        self.asked: list[float] = []

    async def sleep(self, delay: float) -> None:
        self.asked.append(delay)


class TransportError(httpx.HTTPError):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"transport failed with {status_code}")
        self.response = SimpleNamespace(status_code=status_code)


class FakeStream:
    def __init__(self, script: list) -> None:
        self.script = script
        self.closed = 0
        self.index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.index >= len(self.script):
            raise StopAsyncIteration
        item = self.script[self.index]
        self.index += 1
        if isinstance(item, BaseException):
            raise item
        return item

    async def aclose(self) -> None:
        self.closed += 1


class FakeClient:
    def __init__(self, scripts: list[list]) -> None:
        self.scripts = list(scripts)
        self.calls: list[dict] = []
        self.streams: list[FakeStream] = []
        self.invalidated = 0

    def chat(self, messages, model=DEFAULT_MODEL, effort=None, *, can_use_tools=False, can_use_web_search=False):
        self.calls.append({"model": model, "effort": effort, "can_use_tools": can_use_tools, "can_use_web_search": can_use_web_search})
        stream = FakeStream(self.scripts[len(self.streams)])
        self.streams.append(stream)
        return stream

    def invalidate_attestation(self) -> None:
        self.invalidated += 1

    @property
    def closed(self) -> int:
        return sum(stream.closed for stream in self.streams)


def _event(**kwargs) -> DuckAIEvent:
    event = DuckAIEvent()
    for key, value in kwargs.items():
        setattr(event, key, value)
    return event


def _account(scripts):
    client = FakeClient(scripts)
    return DuckAIAccount(0, client), client


async def _drain(gen) -> list[str]:
    try:
        return [item async for item in gen]
    finally:
        await gen.aclose()


def _payloads(lines: list[str]) -> list[dict]:
    out: list[dict] = []
    for line in lines:
        if not line.startswith("data: ") or line == "data: [DONE]\n\n":
            continue
        out.append(json.loads(line[len("data: ") :]))
    return out


def _all_deltas(lines: list[str]) -> list[dict]:
    return [payload["choices"][0]["delta"] for payload in _payloads(lines) if payload.get("choices")]


def _deltas(lines: list[str], key: str) -> list[str]:
    return [delta[key] for delta in _all_deltas(lines) if key in delta]


def _content(lines: list[str]) -> str:
    return "".join(_deltas(lines, "content"))


def _finish(lines: list[str]) -> str | None:
    reasons = [payload["choices"][0]["finish_reason"] for payload in _payloads(lines) if payload.get("choices")]
    return next(reason for reason in reversed(reasons) if reason is not None)


@pytest.fixture(autouse=True)
def _no_backoff(monkeypatch):
    seen: list[int] = []

    async def _instant(attempt: int) -> None:
        seen.append(attempt)

    monkeypatch.setattr(duckai_api, "_sleep_backoff", _instant)
    return seen


def test_retry_constants_come_from_the_shared_retry_module(monkeypatch):
    monkeypatch.setattr(retry_mod, "RETRY_BACKOFF_SEC", RETRY_BACKOFF_SEC)
    assert duckai_api.MAX_RETRIES is retry_mod.MAX_RETRIES
    assert MAX_RETRIES == 5
    assert RETRY_JITTER == 0.5
    first_low, first_high = _retry_delay_bounds(1)
    assert first_low <= ATTESTATION_RETRY_DELAY <= first_high
    monkeypatch.setattr(retry_mod.random, "uniform", _neutral_jitter)
    assert retry_mod._retry_delay(1) == RETRY_BACKOFF_SEC


def test_status_for_maps_every_error_family():
    assert _status_for(DuckAIError(401, "no")) == 401
    assert _status_for(DuckAIError(403, "no")) == 401
    assert _status_for(DuckAIError(418, "teapot")) == 403
    assert _status_for(DuckAIError("ERR_BN_LIMIT", "")) == 403
    assert _status_for(DuckAIError(408, "")) == 502
    assert _status_for(DuckAIError(504, "")) == 502
    assert _status_for(DuckAIError(409, "")) == 400
    assert _status_for(DuckAIError("ERR_UPSTREAM", "")) == 502
    assert _status_for(DuckAIError(302, "")) == 502


def test_detail_for_entrypoint_returns_the_entrypoint_hint():
    assert _detail_for(DuckAIError("ERR_BN_LIMIT", "blocked")) == ENTRYPOINT_HINT
    assert _detail_for(DuckAIError(500, "this is an unsupported entrypoint here")) == ENTRYPOINT_HINT


def test_detail_for_challenge_and_plain_error():
    assert _detail_for(DuckAIError(418, "")) == BLOCKED_HINT
    assert _detail_for(DuckAIError(500, "  boom  ")) == "duckai error: boom"
    assert _detail_for(DuckAIError(500, "")) == "duckai error: 500"


def test_hints_do_not_disclose_the_solver_or_the_version_header():
    for hint in (BLOCKED_HINT, ENTRYPOINT_HINT):
        assert "jsa_solver" not in hint
        assert "jsa_solver.js" not in hint
        assert "x-fe-version" not in hint.lower()
    assert "did not pass DuckDuckGo's check" in BLOCKED_HINT


def test_no_image_markers_match_capability_failures_only():
    for message in ("Model does not support image input", "unsupported image type", "no image support for this model"):
        assert _detail_for(DuckAIError(400, message)) == f"{NO_IMAGE_HINT}: {message}"
    unrelated = "tool could not process image bytes from the request"
    assert _detail_for(DuckAIError(500, unrelated)) == f"duckai error: {unrelated}"


def test_text_of_reads_strings_lists_and_scalars():
    assert _text_of("plain") == "plain"
    assert _text_of(["a", {"type": "text", "text": "b"}, {"type": "input_text", "text": "c"}, {"type": "image_url"}, 7]) == "abc"
    assert _text_of(None) == ""
    assert _text_of(42) == "42"


def test_images_of_accepts_both_image_url_shapes():
    assert _images_of([{"type": "image_url", "image_url": "data:image/png;base64,AAAA"}]) == [("image/png", "data:image/png;base64,AAAA")]
    assert _images_of([{"type": "image_url", "image_url": {"url": "data:image/gif;base64,AAAA"}}]) == [("image/gif", "data:image/gif;base64,AAAA")]
    assert _images_of([{"type": "text", "text": "x"}]) == []
    assert _images_of([{"type": "image_url", "image_url": 42}]) == []
    assert _images_of([{"type": "image_url", "image_url": {"url": 7}}]) == []
    assert _images_of("not a list") == []


def test_images_of_rejects_a_non_data_uri():
    with pytest.raises(HTTPException) as excinfo:
        _images_of([{"type": "image_url", "image_url": {"url": "https://x/y.png"}}])
    assert excinfo.value.status_code == 400
    assert excinfo.value.detail == "duckai only accepts inline data URI images"


def test_images_of_rejects_a_mime_outside_the_allowlist():
    with pytest.raises(HTTPException) as excinfo:
        _images_of([{"type": "image_url", "image_url": {"url": "data:text/html;base64,PGI+aGk8L2I+"}}])
    assert excinfo.value.status_code == 400
    assert excinfo.value.detail == f"duckai accepts only these inline image types: {', '.join(sorted(ALLOWED_IMAGE_MIME_TYPES))}, got text/html"


def test_images_of_rejects_a_payload_without_the_base64_parameter():
    with pytest.raises(HTTPException) as excinfo:
        _images_of([{"type": "image_url", "image_url": {"url": "data:image/png,AAAA"}}])
    assert excinfo.value.status_code == 400
    assert excinfo.value.detail == "duckai only accepts base64 encoded inline image data"


def test_images_of_rejects_payloads_that_are_not_base64():
    with pytest.raises(HTTPException) as excinfo:
        _images_of([{"type": "image_url", "image_url": {"url": "data:image/png;base64,not*base64"}}])
    assert excinfo.value.status_code == 400
    assert excinfo.value.detail == "duckai inline image data is not valid base64"


def test_images_of_skips_a_data_uri_with_no_payload():
    assert _images_of([{"type": "image_url", "image_url": {"url": "data:image/png;base64,"}}]) == []


def test_images_of_defaults_a_missing_mime_to_png():
    payload = base64.b64encode(b"x").decode()
    assert _images_of([{"type": "image_url", "image_url": {"url": f"data:;base64,{payload}"}}]) == [("image/png", f"data:;base64,{payload}")]


def test_tool_specs_merges_tools_and_functions():
    tools = [{"type": "function", "function": {"name": "a", "description": "does a", "parameters": {"type": "object"}}}]
    functions = [{"name": "a", "description": "dup"}, {"name": "b"}, "not a dict", {"description": "no name"}]
    assert _tool_specs(tools, functions) == [
        {"name": "a", "description": "does a", "parameters": {"type": "object"}},
        {"name": "b", "description": "", "parameters": {}},
    ]
    assert _tool_specs(None, None) == []
    assert _tool_specs([{"type": "function"}, "x"], None) == []


def test_tool_calls_of_reads_dicts_and_parsed_events():
    call = {"id": "c1", "type": "function", "function": {"name": "f", "arguments": '{"a":1}'}}
    assert _tool_calls_of({"tool_calls": [call]}) == [{"id": "c1", "name": "f", "arguments": '{"a":1}'}]
    event = parse_tool_event()
    parsed = _tool_calls_of(event)
    assert [item["name"] for item in parsed] == ["f"]
    assert [item["arguments"] for item in parsed] == ['{"a":1}']
    generated = _tool_calls_of({"tool_calls": ["x", {"function": {}}, {"function": {"name": "g"}}]})
    assert [item["name"] for item in generated] == ["g"]
    assert generated[0]["id"].startswith("call_")
    assert generated[0]["arguments"] == "{}"
    assert _tool_calls_of({"tool_calls": "not a list"}) == []


def parse_tool_event():
    from danyapi.duckai.client import parse_event

    return parse_event({"action": "success", "role": "tool-invocation", "state": "call", "toolName": "f", "toolArguments": {"a": 1}})


def test_build_messages_attaches_orphan_tool_results_to_a_new_assistant_turn():
    built = build_messages([ChatMessage(role="user", content="hi"), ChatMessage(role="tool", tool_call_id="call_9", content="42")])
    assert [message["role"] for message in built] == ["user", "assistant"]
    assert built[1] == {"role": "assistant", "content": "", "parts": [{"type": "tool-result", "toolCallId": "call_9", "result": "42", "data": None}]}


def test_build_messages_caps_images_per_message_before_per_request():
    content = [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}} for _ in range(MAX_IMAGES_PER_MESSAGE + 1)]
    with pytest.raises(HTTPException) as excinfo:
        build_messages([ChatMessage(role="user", content=content)])
    assert excinfo.value.status_code == 400
    assert excinfo.value.detail == f"duckai accepts at most {MAX_IMAGES_PER_MESSAGE} images per message"


def test_build_messages_caps_inline_image_bytes_per_request():
    assert MAX_IMAGE_BYTES == 15 * 1024 * 1024
    chunk = base64.b64encode(bytes(2 * 1024 * 1024)).decode()
    uri = f"data:image/png;base64,{chunk}"
    messages = [ChatMessage(role="user", content=[{"type": "image_url", "image_url": {"url": uri}}]) for _ in range(8)]
    with pytest.raises(HTTPException) as excinfo:
        build_messages(messages)
    assert excinfo.value.status_code == 400
    assert excinfo.value.detail == "duckai accepts at most 15 MiB of inline image data per request"


def test_build_messages_drops_a_user_turn_with_no_text_and_no_images():
    built = build_messages([ChatMessage(role="user", content="hi"), ChatMessage(role="user", content=[{"type": "video_url", "url": "x"}])])
    assert [message["role"] for message in built] == ["user"]
    assert built[0]["content"] == [{"type": "text", "text": "hi"}]


def test_build_messages_fallback_carries_system_prompt_and_tool_catalog():
    tools = [{"type": "function", "function": {"name": "f", "description": "does f"}}]
    built = build_messages([ChatMessage(role="system", content="be terse")], tools=tools)
    text = built[0]["content"][0]["text"]
    assert text.startswith("Follow these instructions for the rest of the conversation:\nbe terse\n\nYou can call these tools")
    assert text.endswith("- f: does f\n\nHello")


def test_prompt_text_skips_non_user_turns():
    messages = [
        ChatMessage(role="user", content="first"),
        ChatMessage(role="assistant", content="answer"),
        ChatMessage(role="tool", tool_call_id="c1", content="42"),
        ChatMessage(role="user", content="second"),
    ]
    assert _prompt_text(build_messages(messages)) == "first\nsecond"


async def test_transport_failure_without_a_delivered_event_retries_to_the_limit(_no_backoff):
    account, client = _account([[TransportError(503)] * 2] * (MAX_RETRIES + 1))
    with pytest.raises(HTTPException) as excinfo:
        await collect_non_stream(account, [ChatMessage(role="user", content="hi")], model=MODEL)
    assert excinfo.value.status_code == 502
    assert excinfo.value.detail == "duckai transport error: transport failed with 503"
    assert len(client.calls) == MAX_RETRIES + 1
    assert client.closed == MAX_RETRIES + 1
    assert _no_backoff == list(range(MAX_RETRIES))


async def test_failure_after_a_delivered_event_propagates_without_retrying(_no_backoff):
    account, client = _account([[_event(delta="partial"), TransportError(503)], [_event(delta="unused")]])
    with pytest.raises(HTTPException) as excinfo:
        await collect_non_stream(account, [ChatMessage(role="user", content="hi")], model=MODEL)
    assert excinfo.value.status_code == 502
    assert len(client.calls) == 1
    assert client.closed == 1
    assert _no_backoff == []


async def test_transport_failure_with_a_non_retryable_status_raises_immediately():
    account, client = _account([[TransportError(400)], [_event(delta="unused")]])
    with pytest.raises(HTTPException) as excinfo:
        await collect_non_stream(account, [ChatMessage(role="user", content="hi")], model=MODEL)
    assert excinfo.value.detail == "duckai transport error: transport failed with 400"
    assert len(client.calls) == 1
    assert client.closed == 1


async def test_transport_failure_without_a_response_status_raises_immediately():
    account, client = _account([[httpx.ConnectError("no route")], [_event(delta="unused")]])
    with pytest.raises(HTTPException) as excinfo:
        await collect_non_stream(account, [ChatMessage(role="user", content="hi")], model=MODEL)
    assert excinfo.value.status_code == 502
    assert excinfo.value.detail == "duckai transport error: no route"
    assert len(client.calls) == 1
    assert client.closed == 1


async def test_attestation_failure_invalidates_and_retries_to_the_limit(_no_backoff):
    account, client = _account([[attest.AttestationError("unsupported fragment")]] * (MAX_RETRIES + 1))
    with pytest.raises(HTTPException) as excinfo:
        await collect_non_stream(account, [ChatMessage(role="user", content="hi")], model=MODEL)
    assert excinfo.value.status_code == 502
    assert excinfo.value.detail == "duckai attestation failed: unsupported fragment"
    assert client.invalidated == MAX_RETRIES + 1
    assert client.closed == MAX_RETRIES + 1
    assert _no_backoff == list(range(MAX_RETRIES))


async def test_attestation_failure_after_a_delivered_event_does_not_retry():
    account, client = _account([[_event(delta="partial"), attest.AttestationError("late")], [_event(delta="unused")]])
    with pytest.raises(HTTPException) as excinfo:
        await collect_non_stream(account, [ChatMessage(role="user", content="hi")], model=MODEL)
    assert excinfo.value.detail == "duckai attestation failed: late"
    assert client.invalidated == 1
    assert len(client.calls) == 1
    assert client.closed == 1


async def test_challenge_error_retries_then_maps_to_the_blocked_hint():
    account, client = _account([[DuckAIError(418, "ERR_CHALLENGE")]] * (MAX_RETRIES + 1))
    with pytest.raises(HTTPException) as excinfo:
        await collect_non_stream(account, [ChatMessage(role="user", content="hi")], model=MODEL)
    assert excinfo.value.status_code == 403
    assert excinfo.value.detail == BLOCKED_HINT
    assert client.invalidated == MAX_RETRIES + 1
    assert client.closed == MAX_RETRIES + 1


async def test_challenge_error_after_a_delivered_event_does_not_retry():
    account, client = _account([[_event(delta="partial"), DuckAIError(418, "ERR_CHALLENGE")], [_event(delta="unused")]])
    with pytest.raises(HTTPException) as excinfo:
        await collect_non_stream(account, [ChatMessage(role="user", content="hi")], model=MODEL)
    assert excinfo.value.status_code == 403
    assert client.invalidated == 1
    assert len(client.calls) == 1
    assert client.closed == 1


async def test_retryable_duck_error_retries_then_raises_502():
    account, client = _account([[DuckAIError(500, "boom")]] * (MAX_RETRIES + 1))
    with pytest.raises(HTTPException) as excinfo:
        await collect_non_stream(account, [ChatMessage(role="user", content="hi")], model=MODEL)
    assert excinfo.value.status_code == 502
    assert excinfo.value.detail == "duckai error: boom"
    assert client.invalidated == 0
    assert client.closed == MAX_RETRIES + 1


async def test_retryable_duck_error_after_a_delivered_event_does_not_retry():
    account, client = _account([[_event(delta="partial"), DuckAIError(500, "boom")], [_event(delta="unused")]])
    with pytest.raises(HTTPException) as excinfo:
        await collect_non_stream(account, [ChatMessage(role="user", content="hi")], model=MODEL)
    assert excinfo.value.status_code == 502
    assert client.closed == 1
    assert len(client.calls) == 1


async def test_non_retryable_duck_error_raises_without_retrying():
    account, client = _account([[DuckAIError(409, "conflict")], [_event(delta="unused")]])
    with pytest.raises(HTTPException) as excinfo:
        await collect_non_stream(account, [ChatMessage(role="user", content="hi")], model=MODEL)
    assert excinfo.value.status_code == 400
    assert excinfo.value.detail == "duckai error: conflict"
    assert len(client.calls) == 1
    assert client.closed == 1


async def test_sleep_backoff_requests_a_jittered_delay(monkeypatch):
    recorder = SleepRecorder()
    monkeypatch.setattr(duckai_api, "asyncio", recorder)
    monkeypatch.setattr(retry_mod, "RETRY_BACKOFF_SEC", RETRY_BACKOFF_SEC)
    rolls = ((0, 0.25), (1, 0.0), (3, 1.0))
    for attempt, roll in rolls:
        monkeypatch.setattr(duckai_api.random, "random", lambda roll=roll: roll)
        await _REAL_SLEEP_BACKOFF(attempt)
    assert len(recorder.asked) == len(rolls)
    for asked, (attempt, roll) in zip(recorder.asked, rolls, strict=True):
        retry_low, retry_high = _retry_delay_bounds(attempt + 1)
        floor = duckai_api.ATTESTATION_RETRY_DELAY
        factor = 1.0 - duckai_api.RETRY_JITTER + roll * duckai_api.RETRY_JITTER
        assert max(floor, retry_low) * factor <= asked <= max(floor, retry_high) * factor


async def test_sleep_backoff_uses_the_growing_backoff_floor(monkeypatch):
    recorder = SleepRecorder()
    monkeypatch.setattr(duckai_api, "asyncio", recorder)
    monkeypatch.setattr(duckai_api.random, "random", lambda: 1.0)
    monkeypatch.setattr(duckai_api, "ATTESTATION_RETRY_DELAY", 0.25)
    monkeypatch.setattr(retry_mod, "RETRY_BACKOFF_SEC", 4.0)
    monkeypatch.setattr(retry_mod.random, "uniform", _neutral_jitter)
    await _REAL_SLEEP_BACKOFF(1)
    assert recorder.asked == [min(retry_mod.RETRY_BACKOFF_MAX_SEC, retry_mod.RETRY_BACKOFF_SEC * 2)]


async def test_zero_delay_removes_the_wait(monkeypatch):
    recorder = SleepRecorder()
    monkeypatch.setattr(duckai_api, "asyncio", recorder)
    monkeypatch.setattr(duckai_api, "ATTESTATION_RETRY_DELAY", 0.0)
    await _REAL_SLEEP_BACKOFF(2)
    assert recorder.asked == []


async def test_collect_non_stream_reports_a_refusal_without_text():
    account, _ = _account([[_event(refusal="model_safety"), _event(delta="")]])
    result = await collect_non_stream(account, [ChatMessage(role="user", content="hi")], model=MODEL)
    assert result["choices"][0]["finish_reason"] == "content_filter"
    assert result["choices"][0]["message"]["content"] == ""


async def test_collect_non_stream_keeps_the_first_refusal_and_prefers_text():
    account, _ = _account([[_event(refusal="first"), _event(refusal="second"), _event(delta="text")]])
    result = await collect_non_stream(account, [ChatMessage(role="user", content="hi")], model=MODEL)
    assert result["choices"][0]["message"]["content"] == "text"
    assert result["choices"][0]["finish_reason"] == "stop"


async def test_collect_non_stream_attaches_sources():
    sources = [{"url": "https://a", "title": "A", "site": "a.com"}]
    account, _ = _account([[_event(sources=sources), _event(delta="x", finish="stop")]])
    result = await collect_non_stream(account, [ChatMessage(role="user", content="hi")], model=MODEL)
    assert result["choices"][0]["message"]["sources"] == sources
    assert result["choices"][0]["finish_reason"] == "stop"


async def test_collect_non_stream_enforces_max_tokens_and_stop():
    account, _ = _account([[_event(delta="one two three four"), _event(finish="stop")]])
    result = await collect_non_stream(account, [ChatMessage(role="user", content="hi")], model=MODEL, max_tokens=1)
    assert result["choices"][0]["message"]["content"] == "one two"
    assert result["choices"][0]["finish_reason"] == "length"
    account, _ = _account([[_event(delta="keep DROP"), _event(finish="stop")]])
    stopped = await collect_non_stream(account, [ChatMessage(role="user", content="hi")], model=MODEL, stop=["DROP"])
    assert stopped["choices"][0]["message"]["content"] == "keep "
    assert stopped["choices"][0]["finish_reason"] == "stop"


async def test_collect_non_stream_reports_a_limit_as_a_truncated_answer():
    account, _ = _account([[_event(delta="half"), _event(finish="stop", limit="ERR_OUTPUT_LIMIT")]])
    result = await collect_non_stream(account, [ChatMessage(role="user", content="hi")], model=MODEL)
    assert result["choices"][0]["message"]["content"] == "half"
    assert result["choices"][0]["finish_reason"] == "length"


async def test_collect_non_stream_honours_thinking_and_tool_choice():
    account, client = _account([[_event(delta="ok"), _event(finish="stop")]])
    await collect_non_stream(account, [ChatMessage(role="user", content="hi")], model=MODEL, thinking=True)
    assert client.calls[0]["effort"] == "medium"
    account, client = _account([[_event(delta="ok"), _event(finish="stop")]])
    await collect_non_stream(account, [ChatMessage(role="user", content="hi")], model=MODEL, thinking=False)
    assert client.calls[0]["effort"] == "none"
    tools = [{"type": "function", "function": {"name": "f"}}]
    account, client = _account([[_event(delta="ok"), _event(finish="stop")]])
    await collect_non_stream(
        account,
        [ChatMessage(role="user", content="hi")],
        model=MODEL,
        tools=tools,
        tool_choice="none",
    )
    assert client.calls[0]["can_use_tools"] is False


async def test_stream_openai_trims_at_max_tokens_and_reports_a_refusal():
    account, _ = _account([[_event(delta="one two three"), _event(finish="stop")]])
    lines = await _drain(stream_openai(account, [ChatMessage(role="user", content="hi")], model=MODEL, max_tokens=1))
    assert _content(lines) == "one two"
    assert _finish(lines) == "length"
    account, _ = _account([[_event(refusal="model_safety"), _event(finish="stop")]])
    lines = await _drain(stream_openai(account, [ChatMessage(role="user", content="hi")], model=MODEL))
    assert _finish(lines) == "content_filter"


async def test_stream_openai_numbers_parallel_tool_calls():
    calls = [
        {"index": 0, "id": "c1", "type": "function", "function": {"name": "a", "arguments": "{}"}},
        {"index": 0, "id": "c2", "type": "function", "function": {"name": "b", "arguments": "{}"}},
    ]
    account, _ = _account([[_event(tool_calls=calls), _event(finish="stop")]])
    lines = await _drain(stream_openai(account, [ChatMessage(role="user", content="hi")], model=MODEL))
    emitted = [delta["tool_calls"][0] for delta in _all_deltas(lines) if "tool_calls" in delta]
    assert [call["index"] for call in emitted] == [0, 1]


async def test_stream_openai_rejects_bad_input_in_band():
    account, client = _account([[_event(delta="unused")]])
    lines = await _drain(
        stream_openai(
            account,
            [ChatMessage(role="user", content=[{"type": "image_url", "image_url": {"url": "https://x/y.png"}}])],
            model=MODEL,
        )
    )
    joined = "".join(lines)
    assert "duckai only accepts inline data URI images" in joined
    assert client.calls == []


async def test_retry_backoff_runs_without_the_account_held():
    order: list[str] = []
    seen: list[bool] = []
    account, _ = _account([[TransportError(503)], [_event(delta="ok"), _event(finish="stop")]])

    async def _instant(attempt: int) -> None:
        order.append(f"sleep-{attempt}")
        seen.append(account.sem.locked())

    duckai_api._sleep_backoff = _instant
    try:
        result = await collect_non_stream(account, [ChatMessage(role="user", content="hi")], model=MODEL)
    finally:
        duckai_api._sleep_backoff = _REAL_SLEEP_BACKOFF
    assert result["choices"][0]["message"]["content"] == "ok"
    assert order == ["sleep-0"]
    assert seen == [False]


def test_build_messages_drops_the_tool_catalog_for_tool_choice_none():
    tools = [{"type": "function", "function": {"name": "f", "description": "does f"}}]
    built = build_messages([ChatMessage(role="user", content="hi")], tools=tools, tool_choice="none")
    assert built[0]["content"] == [{"type": "text", "text": "hi"}]


def test_build_messages_maps_a_legacy_function_result_to_a_tool_result():
    built = build_messages(
        [
            ChatMessage(role="user", content="weather?"),
            ChatMessage(role="assistant", content="", tool_calls=[{"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]),
            ChatMessage(role="function", name="f", content="42"),
        ]
    )
    assert [message["role"] for message in built] == ["user", "assistant"]
    assert built[1]["parts"][-1] == {"type": "tool-result", "toolCallId": "f", "result": "42", "data": None}


def test_build_messages_keeps_an_unanswered_tool_result_on_its_own_turn():
    built = build_messages(
        [
            ChatMessage(role="user", content="hi"),
            ChatMessage(role="assistant", content="thinking out loud"),
            ChatMessage(role="tool", tool_call_id="other", content="42"),
        ]
    )
    assert [message["role"] for message in built] == ["user", "assistant", "assistant"]
    assert built[1]["parts"] == [{"type": "text", "text": "thinking out loud"}]
    assert built[2]["parts"][0]["toolCallId"] == "other"


def test_build_messages_keeps_a_trailing_system_turn():
    built = build_messages([ChatMessage(role="user", content="hi"), ChatMessage(role="system", content="be terse")])
    assert len(built) == 1
    assert built[0]["content"][0]["text"].endswith("be terse")
    assert built[0]["content"][1] == {"type": "text", "text": "hi"}


def test_image_limits_are_checked_before_the_payloads_are_decoded():
    content = [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64.b64encode(bytes(4 * 1024 * 1024)).decode()}"}}]
    messages = [ChatMessage(role="user", content=content) for _ in range(MAX_IMAGES_PER_REQUEST + 1)]
    with pytest.raises(HTTPException) as excinfo:
        build_messages(messages)
    assert excinfo.value.status_code == 400


async def test_collect_non_stream_rejects_a_remote_image():
    account, _ = _account([[_event(delta="x")]])
    with pytest.raises(HTTPException) as excinfo:
        await collect_non_stream(account, [ChatMessage(role="user", content=[{"type": "image_url", "image_url": {"url": "https://x/y.png"}}])], model=MODEL)
    assert excinfo.value.status_code == 400


async def test_stream_openai_stops_emitting_at_a_marker_split_across_deltas():
    account, _ = _account([[_event(delta="keep "), _event(delta="DROP the rest"), _event(delta=" never emitted"), _event(finish="stop")]])
    lines = await _drain(stream_openai(account, [ChatMessage(role="user", content="hi")], model=MODEL, stop=["DROP"]))
    assert _deltas(lines, "content") == ["ke", "ep "]
    assert _content(lines) == "keep "
    assert _finish(lines) == "stop"


async def test_stream_openai_flushes_an_unmatched_stop_tail():
    account, _ = _account([[_event(delta="ab"), _event(delta="c"), _event(finish="stop")]])
    lines = await _drain(stream_openai(account, [ChatMessage(role="user", content="hi")], model=MODEL, stop="XYZ"))
    assert _deltas(lines, "content") == ["a", "bc"]
    assert _content(lines) == "abc"
    assert _finish(lines) == "stop"


async def test_stream_openai_emits_reasoning_and_tool_call_deltas():
    call = {"index": 0, "id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}
    account, _ = _account([[_event(reasoning="thinking"), _event(delta="answer"), _event(tool_calls=[call]), _event(finish="stop")]])
    lines = await _drain(stream_openai(account, [ChatMessage(role="user", content="hi")], model=MODEL))
    assert _deltas(lines, "reasoning_content") == ["thinking"]
    assert _deltas(lines, "content") == ["answer"]
    assert [delta for delta in _all_deltas(lines) if "tool_calls" in delta] == [{"tool_calls": [call]}]
    assert _finish(lines) == "tool_calls"


async def test_stream_openai_emits_one_empty_content_delta_for_an_empty_answer():
    account, _ = _account([[]])
    lines = await _drain(stream_openai(account, [ChatMessage(role="user", content="hi")], model=MODEL))
    assert [delta for delta in _all_deltas(lines) if "content" in delta] == [{"content": ""}]
    assert _finish(lines) == "stop"
    assert lines[-1] == "data: [DONE]\n\n"


async def test_stream_openai_emits_no_empty_content_delta_for_a_normal_answer():
    account, _ = _account([[_event(delta="a"), _event(finish="stop")]])
    lines = await _drain(stream_openai(account, [ChatMessage(role="user", content="hi")], model=MODEL))
    assert [delta for delta in _all_deltas(lines) if "content" in delta] == [{"content": "a"}]
    assert _content(lines) == "a"


async def test_stream_openai_releases_the_account_before_the_terminator():
    account, _ = _account([[_event(delta="a"), _event(reasoning="r"), _event(finish="stop")]])
    seen: list[tuple[str, bool]] = []
    async for line in stream_openai(account, [ChatMessage(role="user", content="hi")], model=MODEL, include_usage=True):
        seen.append((line, account.sem.locked()))
    finish_index = next(index for index, (line, _) in enumerate(seen) if '"finish_reason": "stop"' in line)
    assert [locked for _, locked in seen[:finish_index]] == [True] * finish_index
    assert [locked for _, locked in seen[finish_index:]] == [False] * (len(seen) - finish_index)
    assert '"usage"' in seen[finish_index + 1][0]
    assert seen[-1][0] == "data: [DONE]\n\n"


async def test_stream_openai_leaves_the_semaphore_free_for_the_next_session_at_the_tail():
    account, _ = _account([[_event(delta="a"), _event(finish="stop")]])
    acquired = asyncio.Event()
    order: list[str] = []
    finish_at = -1

    async def probe() -> None:
        await account.sem.acquire()
        order.append("probe")
        acquired.set()
        account.sem.release()

    async for line in stream_openai(account, [ChatMessage(role="user", content="hi")], model=MODEL):
        order.append("chunk")
        if '"finish_reason": "stop"' in line:
            finish_at = len(order) - 1
            assert account.sem.locked() is False
            task = asyncio.create_task(probe())
            await asyncio.wait_for(acquired.wait(), 1)
            assert task.done()
    assert order.count("probe") == 1
    assert order.index("probe") > finish_at
    assert account.sem.locked() is False


async def test_stream_openai_reports_a_transport_error_as_an_in_band_event():
    account, _ = _account([[TransportError(503)]] * (MAX_RETRIES + 1))
    lines = await _drain(stream_openai(account, [ChatMessage(role="user", content="hi")], model=MODEL))
    joined = "".join(lines)
    assert '"error"' in joined
    assert "duckai transport error: transport failed with 503" in joined
    assert lines[-1] == "data: [DONE]\n\n"
