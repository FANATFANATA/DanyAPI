import asyncio
import json
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from fastapi import HTTPException

import danyapi.api.deepseek as ds
import danyapi.api.retry as retry_mod
from danyapi.api.schemas import ChatMessage, DeepSeekStreamError
from danyapi.deepseek.client import DeepSeekClient, DeepSeekError

MODEL = "deepseek-v4.1-flash"

TOOL_CALLS_JSON = json.dumps({"tool_calls": [{"name": "get_weather", "arguments": {"city": "Moscow"}}, {"name": "get_time", "arguments": {}}]})
PARTIAL_TOOL_CALL = '{"tool_calls": [{"name": "get_weather", "argum'
WEATHER_TOOL = {"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}

READY = 'event: ready\ndata: {"request_message_id":1,"response_message_id":2,"model_type":"default"}\n\n'
INPUT_HTTP_BODY = '{"message":"Content is too long. Please shorten it and try again.","finish_reason":"input_exceeds_limit"}'


def _initial(content: str = "", frag_type: str = "RESPONSE", message_id: int = 2, accumulated: int | None = None) -> str:
    fragment: dict = {"id": message_id, "type": frag_type, "content": content}
    response: dict = {"message_id": message_id, "parent_id": 1, "status": "WIP", "fragments": [fragment]}
    if accumulated is not None:
        response["accumulated_token_usage"] = accumulated
    return "data: " + json.dumps({"v": {"response": response}}) + "\n\n"


def _append(text: str, index: int = 0) -> str:
    return "data: " + json.dumps({"p": f"response/fragments/{index}/content", "o": "APPEND", "v": text}) + "\n\n"


def _batch(*ops: tuple[str, str, object]) -> str:
    return "data: " + json.dumps({"o": "BATCH", "v": [{"o": op, "p": path, "v": value} for op, path, value in ops]}) + "\n\n"


def _two_fragments() -> str:
    response = {
        "message_id": 2,
        "parent_id": 1,
        "status": "WIP",
        "fragments": [
            {"id": 2, "type": "RESPONSE", "content": "done"},
            {"id": 3, "type": "THINK", "content": "why"},
        ],
    }
    return "data: " + json.dumps({"v": {"response": response}}) + "\n\n"


def _status(value: str) -> str:
    return "data: " + json.dumps({"p": "response/status", "o": "SET", "v": value}) + "\n\n"


def _hint(message: str, finish_reason: str) -> str:
    payload = {"type": "error", "content": message, "finish_reason": finish_reason}
    return "event: hint\ndata: " + json.dumps(payload) + "\n\n"


OK_SSE = READY + _initial("Hi") + _status("FINISHED")
OK_SSE_TOKENS = READY + _initial("Hi", accumulated=40) + _status("FINISHED")
INPUT_SSE = READY + _status("input_exceeds_limit")
BUSY_SSE = READY + _hint("Server is busy.", "server_busy")
RATE_SSE = READY + _hint("Message too frequent", "rate_limited")
FAKE_CTX_SSE = READY + _hint("Length limit reached. Please start a new chat.", "context_length_exceeded")
CTX_SSE = READY + _status("CONTEXT_LENGTH_EXCEEDED")
THINK_SSE = READY + _initial("why", "THINK", 2) + _append("Answer", 1) + _status("FINISHED")
THINK_ONLY_SSE = READY + _initial("why", "THINK", 2) + _status("FINISHED")
TOOL_SSE = READY + _initial(TOOL_CALLS_JSON) + _status("FINISHED")
TOOL_WIP_SSE = READY + _initial(TOOL_CALLS_JSON) + _status("WIP")
PARTIAL_TOOL_SSE = READY + _initial(PARTIAL_TOOL_CALL) + _status("FINISHED")
REASONING_TAIL_SSE = READY + _two_fragments() + _append("<", 1) + _status("FINISHED")


class FakeResp:
    def __init__(self, body=None, sse_text=None, status=200, content_type="text/event-stream; charset=utf-8") -> None:
        self.status_code = status
        self.headers = {"content-type": content_type}
        self.close_calls = 0
        self.close_error: Exception | None = None
        self._b = sse_text.encode() if sse_text is not None else (body or "").encode()

    async def aiter_bytes(self):
        yield self._b

    async def aclose(self):
        self.close_calls += 1
        if self.close_error is not None:
            raise self.close_error

    async def aread(self):
        return self._b


class StreamErrorResp(FakeResp):
    def __init__(self, exc: BaseException, **kwargs) -> None:
        super().__init__(**kwargs)
        self._exc = exc

    async def aiter_bytes(self):
        yield self._b
        raise self._exc


class FakeSession:
    def __init__(self, sid: str = "c1", last_message_id: str | None = None) -> None:
        self.id = sid
        self.last_message_id = last_message_id
        self.accumulated_tokens = 0


class FakeAccount:
    def __init__(self, responses=()) -> None:
        self.index = 0
        self.broken = False
        self.broken_calls = 0
        self.client = MagicMock()
        self.client.completion = AsyncMock(side_effect=list(responses))
        self.client.stop_stream = AsyncMock()
        self.client.upload_file = AsyncMock(return_value={"id": "f1"})
        self.pow = MagicMock()
        self.pow.make_header = AsyncMock(return_value={})
        self.pow_upload = MagicMock()
        self.pow_upload.make_header = AsyncMock(return_value={})
        self.sem = asyncio.Semaphore(1)
        self.sessions = MagicMock()
        self.sessions.obtain = AsyncMock(return_value=(FakeSession(), "s1"))
        self.sessions.touch_last_message = MagicMock()
        self.sessions.forget = MagicMock()
        self.sessions.get = MagicMock(return_value=None)

    def mark_broken(self):
        self.broken = True
        self.broken_calls += 1


@pytest.fixture(autouse=True)
def _no_real_sleeps(monkeypatch):
    slept: list[float] = []

    async def fake_sleep(delay):
        slept.append(delay)

    monkeypatch.setattr(ds, "asyncio", SimpleNamespace(sleep=fake_sleep))
    monkeypatch.setattr(retry_mod, "RETRY_BACKOFF_SEC", 0.0)
    return slept


@pytest.fixture(autouse=True)
def _recorded_usage(monkeypatch):
    records: list[dict] = []

    def record(provider, model, usage, user=None, session_id=None):
        records.append({"provider": provider, "model": model, "usage": dict(usage), "user": user, "session_id": session_id})

    monkeypatch.setattr(ds, "record_usage_dict", record)
    return records


def _messages(text: str = "hello") -> list[ChatMessage]:
    return [ChatMessage(role="user", content=text)]


async def _drain(agen):
    out = []
    async for item in agen:
        out.append(item)
    return out


def _frames(lines: list[str]) -> list[dict]:
    frames = []
    for line in lines:
        if not line.startswith("data: ") or line[6:].strip() == "[DONE]":
            continue
        frames.append(json.loads(line[6:]))
    return frames


def _client_text(lines: list[str]) -> str:
    out = ""
    for frame in _frames(lines):
        for choice in frame.get("choices") or []:
            delta = choice.get("delta") or {}
            if isinstance(delta.get("content"), str):
                out += delta["content"]
    return out


async def _send(resp):
    client = MagicMock()
    client.completion = AsyncMock(return_value=resp)
    return await ds._send_completion(client, {}, "s", None, "p", "default", False, False)


async def _send_completion(resp):
    return await _send(resp)


async def _collect_non_stream(account, **overrides):
    kwargs = {
        "account": account,
        "pool": MagicMock(),
        "existing_sid": None,
        "lock": account.sem,
        "prompt": "x",
        "model": MODEL,
        "model_type": "default",
        "thinking": False,
        "search": False,
    }
    kwargs.update(overrides)
    return await ds._collect_non_stream(**kwargs)


async def _stream(account, **overrides):
    kwargs = {
        "account": account,
        "pool": MagicMock(),
        "existing_sid": None,
        "lock": account.sem,
        "prompt": "x",
        "model": MODEL,
        "model_type": "default",
        "thinking": False,
        "search": False,
    }
    kwargs.update(overrides)
    return await _drain(ds._stream_openai(**kwargs))


async def test_send_completion_rejects_a_response_without_a_status_code():
    for resp in (None, object()):
        with pytest.raises(HTTPException) as excinfo:
            await _send_completion(resp)
        assert excinfo.value.status_code == 502
        assert excinfo.value.detail == "unexpected provider response"


async def test_send_completion_maps_a_too_frequent_status_to_429():
    resp = FakeResp(body="Message too frequent, please retry", status=503)
    with pytest.raises(HTTPException) as excinfo:
        await _send_completion(resp)
    assert excinfo.value.status_code == 429
    assert excinfo.value.detail == "Message too frequent, please retry"


async def test_send_completion_maps_a_too_frequent_body_to_429():
    resp = FakeResp(body="message too frequent", content_type="application/json")
    with pytest.raises(HTTPException) as excinfo:
        await _send_completion(resp)
    assert excinfo.value.status_code == 429
    assert excinfo.value.detail == "message too frequent"


async def test_send_completion_maps_a_too_frequent_biz_code_to_429():
    resp = FakeResp(body=json.dumps({"data": {"biz_code": 5000, "biz_msg": "message too frequent"}}), content_type="application/json")
    with pytest.raises(HTTPException) as excinfo:
        await _send_completion(resp)
    assert excinfo.value.status_code == 429
    assert excinfo.value.detail == "DeepSeek error 5000: message too frequent"


async def test_send_completion_maps_a_too_frequent_code_to_429():
    resp = FakeResp(body=json.dumps({"code": 5000, "msg": "message too frequent"}), content_type="application/json")
    with pytest.raises(HTTPException) as excinfo:
        await _send_completion(resp)
    assert excinfo.value.status_code == 429
    assert excinfo.value.detail == "DeepSeek error 5000: message too frequent"


async def test_send_completion_maps_auth_codes_to_401():
    resp = FakeResp(body=json.dumps({"data": {"biz_code": 40001, "biz_msg": "expired"}}), content_type="application/json")
    with pytest.raises(HTTPException) as excinfo:
        await _send_completion(resp)
    assert excinfo.value.status_code == 401
    assert excinfo.value.detail == "DeepSeek error 40001: expired"


def test_is_fake_context_hint_requires_a_string_message():
    assert ds._is_fake_context_hint(SimpleNamespace(hint_error={"message": 42, "finish_reason": "context_length_exceeded"})) is False
    assert ds._is_fake_context_hint(SimpleNamespace(hint_error=None)) is False
    assert ds._is_fake_context_hint(SimpleNamespace(hint_error={"message": "LENGTH LIMIT REACHED", "finish_reason": "x"})) is True


def test_error_text_handles_non_serialisable_details():
    assert ds._compact_error_text(42) == ""
    assert ds._compact_error_text("Message-Too Frequent!") == "messagetoofrequent"
    assert ds._error_text(42) == "42"
    assert ds._error_text(None) == "null"
    assert ds._error_text({"message": "x"}) == '{"message": "x"}'
    assert ds._error_text("plain") == "plain"

    class Opaque:
        pass

    assert ds._error_text(Opaque()).startswith("<")


def test_message_too_frequent_text_skips_blank_candidates():
    assert ds._message_too_frequent_text(None, "", "   ") is None
    assert ds._message_too_frequent_text(None, "Message-Too Frequent") == "Message-Too Frequent"
    assert ds._message_too_frequent_text("all good") is None


def test_fake_context_error_body_is_stable():
    assert ds._fake_context_error_body() == {
        "error": {"message": "DeepSeek returned an unexpected length-limit hint and the response is empty", "finish_reason": "server_error"}
    }


async def test_wait_message_too_frequent_logs_and_waits(_no_real_sleeps, caplog):
    with caplog.at_level(logging.WARNING, logger="danyapi.api"):
        await ds._wait_message_too_frequent("continuation request", 3)
    assert ds.MESSAGE_TOO_FREQUENT_WAIT_SEC == 60.0
    assert _no_real_sleeps == [60.0]
    assert "deepseek message too frequent (continuation request), retry 3/5 in 60s" in caplog.text


def test_build_assistant_message_truncates_tool_calls_for_max_calls():
    message, finish = ds._build_assistant_message(TOOL_CALLS_JSON, None, True, None, max_calls=1)
    assert finish == "tool_calls"
    assert [call["function"]["name"] for call in message["tool_calls"]] == ["get_weather"]
    assert message["role"] == "assistant"


def test_build_assistant_message_without_tool_calls_is_stop():
    message, finish = ds._build_assistant_message("plain answer", "because", True, None)
    assert finish == "stop"
    assert message == {"role": "assistant", "content": "plain answer", "reasoning_content": "because"}


def test_build_limited_message_reports_length_without_tool_calls():
    text, finish = ds._build_limited_message("a b c d e f g h i j k l m n", None, False, None, 2, None, None, "FINISHED")
    assert finish == "length"
    assert text == {"role": "assistant", "content": "a b c d e f"}


def test_build_limited_message_reports_length_in_tool_mode_without_calls():
    message, finish = ds._build_limited_message("a b c d e f g h i j k l m n", None, True, None, 2, None, None, "FINISHED")
    assert finish == "length"
    assert message == {"role": "assistant", "content": "a b c d e f"}


def test_build_limited_message_maps_the_provider_finish_reason():
    message, finish = ds._build_limited_message("hi", None, False, None, None, None, None, "CONTENT_FILTER")
    assert finish == "content_filter"
    assert message == {"role": "assistant", "content": "hi"}


def test_build_limited_message_truncates_tool_text_and_keeps_tool_calls():
    text, finish = ds._build_limited_message(TOOL_CALLS_JSON, None, True, None, 1, None, False, "FINISHED")
    assert finish == "tool_calls"
    assert len(text["tool_calls"]) == 1
    assert text["content"] == ""


async def test_send_deepseek_stream_stops_upstream_on_a_transport_error():
    account = FakeAccount([StreamErrorResp(httpx.ReadError("connection reset"), sse_text=OK_SSE)])
    with pytest.raises(DeepSeekStreamError) as excinfo:
        await ds._send_deepseek_stream(account, FakeSession(), None, "p", "default", False, False)
    assert str(excinfo.value) == "Stream processing failed: connection reset"
    account.client.stop_stream.assert_awaited_once_with("c1", 2)


async def test_send_deepseek_stream_reraises_other_failures():
    account = FakeAccount([StreamErrorResp(ValueError("bad frame"), sse_text=OK_SSE)])
    with pytest.raises(ValueError, match="bad frame"):
        await ds._send_deepseek_stream(account, FakeSession(), None, "p", "default", False, False)
    account.client.stop_stream.assert_awaited_once_with("c1", 2)


async def test_send_deepseek_stream_stops_upstream_when_the_response_cannot_be_closed(caplog):
    resp = FakeResp(sse_text=OK_SSE)
    resp.close_error = RuntimeError("close failed")
    account = FakeAccount([resp])
    with caplog.at_level(logging.DEBUG, logger="danyapi.api"):
        rec, _response_id, _stop_id = await ds._send_deepseek_stream(account, FakeSession(), None, "p", "default", False, False)
    assert rec.content == "Hi"
    assert resp.close_calls == 1
    assert "response close failed: close failed" in caplog.text
    account.client.stop_stream.assert_awaited_once_with("c1", 2)


async def test_continuation_retries_a_too_frequent_http_error(_no_real_sleeps):
    account = FakeAccount([HTTPException(429, "message too frequent"), FakeResp(sse_text=OK_SSE)])
    rec = await ds._collect_continuation(account, FakeSession(), None, "default", False, False)
    assert rec is not None
    assert rec.content == "Hi"
    assert _no_real_sleeps == [60.0]


async def test_continuation_retries_a_too_frequent_hint(_no_real_sleeps):
    account = FakeAccount([FakeResp(sse_text=RATE_SSE), FakeResp(sse_text=OK_SSE)])
    rec = await ds._collect_continuation(account, FakeSession(), None, "default", False, False)
    assert rec is not None
    assert rec.content == "Hi"
    assert _no_real_sleeps == [60.0]


async def test_non_stream_survives_an_unbuildable_prompt():
    account = FakeAccount([FakeResp(sse_text=OK_SSE)])
    result = await _collect_non_stream(account, messages=[ChatMessage(role="user", content=42)])
    assert result["choices"][0]["message"]["content"] == "Hi"


async def test_non_stream_translates_a_stream_error_to_502():
    account = FakeAccount([StreamErrorResp(httpx.ReadError("reset"), sse_text=OK_SSE)])
    with pytest.raises(HTTPException) as excinfo:
        await _collect_non_stream(account)
    assert excinfo.value.status_code == 502
    assert excinfo.value.detail == "Stream processing failed: reset"
    account.client.stop_stream.assert_awaited_once_with("c1", 2)


async def test_non_stream_rebuilds_a_stale_cached_session():
    account = FakeAccount([HTTPException(404, "session gone"), FakeResp(sse_text=OK_SSE)])
    account.sessions.get = MagicMock(return_value=FakeSession("s1"))
    result = await _collect_non_stream(account, existing_sid="s1", messages=_messages())
    assert result["choices"][0]["message"]["content"] == "Hi"
    account.sessions.forget.assert_called_once_with("s1")
    assert account.client.completion.await_count == 2


async def test_non_stream_retries_a_too_frequent_http_error(_no_real_sleeps):
    account = FakeAccount([HTTPException(429, "message too frequent"), FakeResp(sse_text=OK_SSE)])
    result = await _collect_non_stream(account)
    assert result["choices"][0]["message"]["content"] == "Hi"
    assert _no_real_sleeps == [60.0]


async def test_non_stream_propagates_a_non_retryable_http_error():
    account = FakeAccount([HTTPException(404, "gone")])
    with pytest.raises(HTTPException) as excinfo:
        await _collect_non_stream(account)
    assert excinfo.value.status_code == 404
    assert excinfo.value.detail == "gone"
    assert account.client.completion.await_count == 1


async def test_non_stream_retries_a_too_frequent_stream_hint(_no_real_sleeps):
    account = FakeAccount([FakeResp(sse_text=RATE_SSE), FakeResp(sse_text=OK_SSE)])
    result = await _collect_non_stream(account)
    assert result["choices"][0]["message"]["content"] == "Hi"
    assert _no_real_sleeps == [60.0]


async def test_non_stream_rebuilds_after_a_fake_context_hint():
    account = FakeAccount([FakeResp(sse_text=FAKE_CTX_SSE), FakeResp(sse_text=OK_SSE)])
    account.sessions.get = MagicMock(return_value=FakeSession("s1"))
    result = await _collect_non_stream(account, existing_sid="s1", messages=_messages())
    assert result["choices"][0]["message"]["content"] == "Hi"
    account.sessions.forget.assert_called_once_with("s1")
    assert account.client.completion.await_count == 2


async def test_non_stream_keeps_the_prompt_when_the_rebuild_fails():
    account = FakeAccount([FakeResp(sse_text=FAKE_CTX_SSE), FakeResp(sse_text=OK_SSE)])
    account.sessions.get = MagicMock(return_value=FakeSession("s1"))
    result = await _collect_non_stream(account, existing_sid="s1", messages=[ChatMessage(role="user", content=42)])
    assert result["choices"][0]["message"]["content"] == "Hi"
    assert account.client.completion.await_count == 2


async def test_non_stream_retries_a_retryable_hint(_no_real_sleeps):
    account = FakeAccount([FakeResp(sse_text=BUSY_SSE), FakeResp(sse_text=OK_SSE)])
    result = await _collect_non_stream(account)
    assert result["choices"][0]["message"]["content"] == "Hi"
    assert _no_real_sleeps == [retry_mod._retry_delay(1)]


async def test_non_stream_fake_context_hint_after_retries_is_502():
    account = FakeAccount([FakeResp(sse_text=FAKE_CTX_SSE) for _ in range(ds.MAX_RETRIES + 1)])
    with pytest.raises(HTTPException) as excinfo:
        await _collect_non_stream(account)
    assert excinfo.value.status_code == 502
    assert excinfo.value.detail == {
        "error": {"message": "DeepSeek returned an unexpected length-limit hint and the response is empty", "finish_reason": "server_error"}
    }


async def test_non_stream_busy_hint_after_retries_is_429():
    account = FakeAccount([FakeResp(sse_text=BUSY_SSE) for _ in range(ds.MAX_RETRIES + 1)])
    with pytest.raises(HTTPException) as excinfo:
        await _collect_non_stream(account)
    assert excinfo.value.status_code == 429
    assert excinfo.value.detail == {"error": {"message": "Server is busy.", "finish_reason": "server_busy"}}


async def test_non_stream_computes_reduced_prompts_and_delivers_them(monkeypatch):
    account = FakeAccount([HTTPException(400, INPUT_HTTP_BODY), HTTPException(404, "gone"), FakeResp(sse_text=OK_SSE)])
    reduced_prompt_variants = MagicMock(return_value=[("short prompt", False, {})])
    monkeypatch.setattr(ds, "_reduced_prompt_variants", reduced_prompt_variants)
    result = await _collect_non_stream(account, messages=_messages("a long prompt"))
    assert reduced_prompt_variants.call_count == 1
    assert "a long prompt" in reduced_prompt_variants.call_args.args[4]
    assert result["choices"][0]["message"]["content"] == "Hi"
    assert result["choices"][0]["finish_reason"] == "response_incomplete"
    assert result["error"] == {"message": ds.REDUCED_CONTEXT_MESSAGE, "finish_reason": "response_incomplete"}


async def test_non_stream_context_length_without_content_is_400():
    account = FakeAccount([FakeResp(sse_text=CTX_SSE)])
    with pytest.raises(HTTPException) as excinfo:
        await _collect_non_stream(account)
    assert excinfo.value.status_code == 400
    assert excinfo.value.detail == "context length exceeded: conversation too long, start a new conversation"
    account.sessions.forget.assert_called_once_with("s1")


async def test_non_stream_expands_choices_for_n_greater_than_one(_recorded_usage):
    account = FakeAccount([FakeResp(sse_text=OK_SSE_TOKENS)])
    result = await _collect_non_stream(account, n=3, user="alice")
    assert [choice["index"] for choice in result["choices"]] == [0, 1, 2]
    assert {choice["finish_reason"] for choice in result["choices"]} == {"stop"}
    assert result["session_id"] == "s1"
    assert result["system_fingerprint"] == "fp_danyapi"
    assert _recorded_usage == [
        {
            "provider": "deepseek",
            "model": MODEL,
            "usage": {"prompt_tokens": 1, "completion_tokens": 39, "total_tokens": 40},
            "user": "alice",
            "session_id": "s1",
        }
    ]


async def test_non_stream_records_usage_from_the_provider_tokens(_recorded_usage):
    account = FakeAccount([FakeResp(sse_text=OK_SSE_TOKENS)])
    await _collect_non_stream(account)
    assert _recorded_usage[0]["usage"] == {"prompt_tokens": 1, "completion_tokens": 39, "total_tokens": 40}
    assert account.sessions.touch_last_message.call_args.args[0] == "s1"


async def test_stream_interleaved_set_and_batch_events_emit_the_upstream_text_once():
    upstream = (
        READY
        + _initial("")
        + _append("Hello")
        + _append(" world")
        + _batch(("SET", "response/status", "WIP"), ("SET", "response/fragments/0/content", "Hello world!"))
        + _append(" again")
        + _status("FINISHED")
    )
    account = FakeAccount([FakeResp(sse_text=upstream)])
    lines = await _stream(account)
    assert _client_text(lines) == "Hello world! again"
    assert "Hello world! again" not in _client_text(lines).replace("Hello world! again", "")
    assert lines[-1] == "data: [DONE]\n\n"


async def test_stream_emits_the_reasoning_tail_held_by_the_dsml_filter():
    account = FakeAccount([FakeResp(sse_text=REASONING_TAIL_SSE)])
    lines = await _stream(account, thinking=True)
    frames = _frames(lines)
    assert _client_text(lines) == "done"
    reasoning = [part for frame in frames for choice in frame.get("choices") or [] for part in [(choice.get("delta") or {}).get("reasoning_content")] if part]
    assert "".join(reasoning) == "why<"
    finishes = [(frame["choices"][0].get("finish_reason")) for frame in frames if frame.get("choices")]
    assert finishes[-1] == "stop"


async def test_stream_flushes_the_stop_filter_tail_on_the_final_chunk():
    account = FakeAccount([FakeResp(sse_text=READY + _initial("abc") + _status("FINISHED"))])
    lines = await _stream(account, stop=["STOP"])
    assert _client_text(lines) == "abc"
    assert '"finish_reason": "stop"' in "".join(lines)


async def test_stream_stops_emitting_after_a_stop_marker_is_seen():
    upstream = READY + _initial("a") + _append("STOPb") + _append("more", 1) + _initial("why", "THINK", 2) + _append("hidden", 1) + _status("FINISHED")
    account = FakeAccount([FakeResp(sse_text=upstream)])
    lines = await _stream(account, stop=["STOP"], thinking=True)
    assert _client_text(lines) == "a"
    assert "STOP" not in "".join(lines)
    assert "hidden" not in "".join(lines)


async def test_stream_survives_an_unbuildable_prompt():
    account = FakeAccount([FakeResp(sse_text=OK_SSE)])
    lines = await _stream(account, messages=[ChatMessage(role="user", content=42)])
    assert _client_text(lines) == "Hi"
    assert lines[-1] == "data: [DONE]\n\n"


async def test_stream_rebuilds_a_stale_cached_session():
    account = FakeAccount([HTTPException(404, "session gone"), FakeResp(sse_text=OK_SSE)])
    account.sessions.get = MagicMock(return_value=FakeSession("s1"))
    lines = await _stream(account, existing_sid="s1", messages=_messages())
    assert _client_text(lines) == "Hi"
    account.sessions.forget.assert_called_once_with("s1")


async def test_stream_reports_a_stale_session_prompt_failure():
    account = FakeAccount([HTTPException(404, "session gone"), FakeResp(sse_text=OK_SSE)])
    account.sessions.get = MagicMock(return_value=FakeSession("s1"))
    lines = await _stream(account, existing_sid="s1", messages=[ChatMessage(role="user", content=42)])
    assert "session gone" in "".join(lines)
    assert lines[-1] == "data: [DONE]\n\n"


async def test_stream_retries_a_too_frequent_http_error(_no_real_sleeps):
    account = FakeAccount([HTTPException(429, "message too frequent"), FakeResp(sse_text=OK_SSE)])
    lines = await _stream(account)
    assert _client_text(lines) == "Hi"
    assert _no_real_sleeps == [60.0]


async def test_stream_retries_a_retryable_http_error(_no_real_sleeps):
    account = FakeAccount([HTTPException(502, "upstream down"), FakeResp(sse_text=OK_SSE)])
    lines = await _stream(account)
    assert _client_text(lines) == "Hi"
    assert _no_real_sleeps == [retry_mod._retry_delay(1)]


async def test_stream_emits_the_terminal_error_for_a_dead_session():
    account = FakeAccount([HTTPException(404, "session gone")])
    lines = await _stream(account, existing_sid=None)
    body = "".join(lines)
    assert '"error"' in body
    assert "session gone" in body
    assert lines[-1] == "data: [DONE]\n\n"


async def test_stream_stops_upstream_and_reraises_on_a_transport_failure():
    account = FakeAccount([StreamErrorResp(ValueError("bad frame"), sse_text=OK_SSE)])
    gen = ds._stream_openai(
        account=account,
        pool=MagicMock(),
        existing_sid=None,
        lock=account.sem,
        prompt="x",
        model=MODEL,
        model_type="default",
        thinking=False,
        search=False,
    )
    with pytest.raises(ValueError, match="bad frame"):
        await _drain(gen)
    account.client.stop_stream.assert_awaited_once_with("c1", 2)


async def test_stream_stops_upstream_when_the_response_cannot_be_closed(caplog):
    resp = FakeResp(sse_text=OK_SSE)
    resp.close_error = RuntimeError("close failed")
    account = FakeAccount([resp])
    with caplog.at_level(logging.DEBUG, logger="danyapi.api"):
        lines = await _stream(account)
    assert _client_text(lines) == "Hi"
    assert "response close failed: close failed" in caplog.text
    account.client.stop_stream.assert_awaited_once_with("c1", 2)


async def test_stream_retries_a_too_frequent_hint(_no_real_sleeps):
    account = FakeAccount([FakeResp(sse_text=RATE_SSE), FakeResp(sse_text=OK_SSE)])
    lines = await _stream(account)
    assert _client_text(lines) == "Hi"
    assert _no_real_sleeps == [60.0]


async def test_stream_rebuilds_after_a_fake_context_hint():
    account = FakeAccount([FakeResp(sse_text=FAKE_CTX_SSE), FakeResp(sse_text=OK_SSE)])
    account.sessions.get = MagicMock(return_value=FakeSession("s1"))
    lines = await _stream(account, existing_sid="s1", messages=_messages())
    assert _client_text(lines) == "Hi"
    account.sessions.forget.assert_called_once_with("s1")


async def test_stream_rebuilds_keeps_the_prompt_when_the_rebuild_fails():
    account = FakeAccount([FakeResp(sse_text=FAKE_CTX_SSE), FakeResp(sse_text=OK_SSE)])
    account.sessions.get = MagicMock(return_value=FakeSession("s1"))
    lines = await _stream(account, existing_sid="s1", messages=[ChatMessage(role="user", content=42)])
    assert _client_text(lines) == "Hi"
    assert account.client.completion.await_count == 2


async def test_stream_retries_a_retryable_hint(_no_real_sleeps):
    account = FakeAccount([FakeResp(sse_text=BUSY_SSE), FakeResp(sse_text=OK_SSE)])
    lines = await _stream(account)
    assert _client_text(lines) == "Hi"
    assert _no_real_sleeps == [retry_mod._retry_delay(1)]


async def test_stream_context_length_without_content_emits_a_terminated_error():
    account = FakeAccount([FakeResp(sse_text=CTX_SSE)])
    lines = await _stream(account)
    frames = _frames(lines)
    error = frames[-1]["error"]
    assert error["message"] == "context length exceeded: conversation too long, start a new conversation"
    assert error["finish_reason"] == "CONTEXT_LENGTH_EXCEEDED"
    assert frames[-1]["choices"][0]["finish_reason"] == "length"
    assert lines[-1] == "data: [DONE]\n\n"


async def test_stream_busy_hint_after_retries_emits_the_provider_message():
    account = FakeAccount([FakeResp(sse_text=BUSY_SSE) for _ in range(ds.MAX_RETRIES + 1)])
    lines = await _stream(account)
    frames = _frames(lines)
    assert frames[-1]["error"] == {"message": "Server is busy.", "finish_reason": "server_busy"}
    assert lines[-1] == "data: [DONE]\n\n"


async def test_stream_fake_context_hint_after_retries_emits_the_generic_body():
    account = FakeAccount([FakeResp(sse_text=FAKE_CTX_SSE) for _ in range(ds.MAX_RETRIES + 1)])
    lines = await _stream(account)
    frames = _frames(lines)
    assert frames[-1]["error"] == {"message": ds.FAKE_CONTEXT_HINT_ERROR_MESSAGE, "finish_reason": "server_error"}
    assert lines[-1] == "data: [DONE]\n\n"


async def test_stream_busy_hint_without_a_message_uses_the_default_text():
    upstream = READY + _hint("", "server_busy")
    account = FakeAccount([FakeResp(sse_text=upstream) for _ in range(ds.MAX_RETRIES + 1)])
    lines = await _stream(account)
    frames = _frames(lines)
    assert frames[-1]["error"] == {"message": "DeepSeek server is busy, try again later", "finish_reason": "server_busy"}


async def test_stream_tool_call_truncated_by_the_provider_is_an_error():
    account = FakeAccount([FakeResp(sse_text=TOOL_WIP_SSE)])
    lines = await _stream(account, tool_mode=True)
    frames = _frames(lines)
    assert frames[-1]["error"] == {"message": "the provider output ended before the tool call completed", "finish_reason": "length"}
    assert frames[-1]["choices"][0]["finish_reason"] == "length"
    assert lines[-1] == "data: [DONE]\n\n"


async def test_stream_emits_a_single_tool_call_when_parallel_is_disabled():
    account = FakeAccount([FakeResp(sse_text=TOOL_SSE)])
    lines = await _stream(account, tool_mode=True, parallel_tool_calls=False)
    body = "".join(lines)
    assert '"tool_calls"' in body
    assert '"get_weather"' in body
    assert '"get_time"' not in body
    assert '"finish_reason": "tool_calls"' in body


async def test_stream_repeats_tool_calls_for_every_choice():
    account = FakeAccount([FakeResp(sse_text=TOOL_SSE)])
    lines = await _stream(account, tool_mode=True, n=2)

    names: dict[int, list[str]] = {0: [], 1: []}
    arguments: dict[int, str] = {0: "", 1: ""}
    for frame in _frames(lines):
        for choice in frame.get("choices") or []:
            for call in (choice.get("delta") or {}).get("tool_calls") or []:
                function = call["function"]
                if function.get("name"):
                    names[choice["index"]].append(function["name"])
                arguments[choice["index"]] += function.get("arguments", "")
    assert arguments[0] == '{"city": "Moscow"}{}'
    assert arguments[1] == '{"city": "Moscow"}{}'

    assert names[0] == ["get_weather", "get_time"]
    assert names[1] == ["get_weather", "get_time"]


async def test_stream_repeats_the_content_tail_for_every_choice():
    account = FakeAccount([FakeResp(sse_text=OK_SSE)])
    lines = await _stream(account, n=2)
    frames = [frame for frame in _frames(lines) if frame.get("session_id") == "s1"]
    assert [frame["choices"][0]["index"] for frame in frames if frame["choices"]] == [0, 1, 1]
    assert _client_text(lines) == "HiHi"


async def test_stream_emits_the_tool_tail_when_no_call_parses():
    account = FakeAccount([FakeResp(sse_text=PARTIAL_TOOL_SSE)])
    lines = await _stream(account, tool_mode=True)
    assert _client_text(lines) == PARTIAL_TOOL_CALL
    assert lines[-1] == "data: [DONE]\n\n"


async def test_stream_continuation_deadline_stops_the_retry_loop(monkeypatch):
    monkeypatch.setattr(ds, "CONTINUE_DEADLINE_SEC", 0.0)
    account = FakeAccount([FakeResp(sse_text=INPUT_SSE)])
    lines = await _stream(account)
    frames = _frames(lines)
    assert frames[-1]["error"] == {"message": ds.RESPONSE_INCOMPLETE_MESSAGE, "finish_reason": "response_incomplete"}
    assert account.client.completion.await_count == 1
    assert lines[-1] == "data: [DONE]\n\n"


async def test_stream_continuation_emits_reasoning_from_the_follow_up_turn():
    account = FakeAccount([FakeResp(sse_text=INPUT_SSE), FakeResp(sse_text=THINK_ONLY_SSE)])
    lines = await _stream(account, thinking=True)
    frames = _frames(lines)
    assert _client_text(lines) == ""
    assert any((choice.get("delta") or {}).get("reasoning_content") == "why" for frame in frames for choice in frame.get("choices") or [])
    assert lines[-1] == "data: [DONE]\n\n"


async def test_stream_computes_reduced_prompts_and_delivers_them(monkeypatch):
    account = FakeAccount([HTTPException(400, INPUT_HTTP_BODY), HTTPException(404, "gone"), FakeResp(sse_text=OK_SSE)])
    reduced_prompt_variants = MagicMock(return_value=[("short prompt", False, {})])
    monkeypatch.setattr(ds, "_reduced_prompt_variants", reduced_prompt_variants)
    lines = await _stream(account, messages=_messages("a long prompt"))
    assert reduced_prompt_variants.call_count == 1
    assert _client_text(lines) == "Hi"
    assert '"finish_reason": "response_incomplete"' in "".join(lines)
    assert account.sessions.forget.call_args_list[-1].args[0] == "s1"


async def test_stream_upload_failure_is_reported_without_leaking_the_reason():
    account = FakeAccount([])
    account.client.upload_file = AsyncMock(side_effect=DeepSeekError(5000, "upload rejected"))
    attachment = SimpleNamespace(data=b"x", name="a.png", content_type="image/png", is_image=True)
    lines = await _stream(account, attachments=[attachment])
    frames = _frames(lines)
    assert frames[0]["error"]["message"] == "file upload failed: DeepSeek biz error 5000: upload rejected"
    assert account.client.completion.await_count == 0
    assert lines[-1] == "data: [DONE]\n\n"


async def test_stream_usage_chunk_is_emitted_only_when_requested(_recorded_usage):
    plain = await _stream(FakeAccount([FakeResp(sse_text=OK_SSE_TOKENS)]))
    assert '"usage"' not in "".join(plain)
    with_usage = await _stream(FakeAccount([FakeResp(sse_text=OK_SSE_TOKENS)]), include_usage=True)
    usage_frames = [frame for frame in _frames(with_usage) if "usage" in frame]
    assert usage_frames[-1]["usage"] == {
        "prompt_tokens": 1,
        "completion_tokens": 39,
        "total_tokens": 40,
        "prompt_tokens_details": {"cached_tokens": 0},
        "completion_tokens_details": {"reasoning_tokens": 0},
    }
    assert usage_frames[-1]["choices"] == []
    assert _recorded_usage[0]["usage"]["total_tokens"] == 40


async def test_send_deepseek_stream_stops_the_message_identified_by_the_stream():
    upstream = READY + _initial("partial", "RESPONSE", 7)
    account = FakeAccount([StreamErrorResp(httpx.ReadError("reset"), sse_text=upstream)])
    with pytest.raises(DeepSeekStreamError):
        await ds._send_deepseek_stream(account, FakeSession(), None, "p", "default", False, False)
    account.client.stop_stream.assert_awaited_once_with("c1", 7)


async def test_send_deepseek_stream_stops_the_message_identified_after_a_foreign_error():
    upstream = READY + _initial("partial", "RESPONSE", 7)
    account = FakeAccount([StreamErrorResp(ValueError("bad frame"), sse_text=upstream)])
    with pytest.raises(ValueError, match="bad frame"):
        await ds._send_deepseek_stream(account, FakeSession(), None, "p", "default", False, False)
    account.client.stop_stream.assert_awaited_once_with("c1", 7)


async def test_non_stream_reports_the_stale_session_when_the_rebuild_fails():
    account = FakeAccount([HTTPException(404, "session gone")])
    account.sessions.get = MagicMock(return_value=FakeSession("s1"))
    with pytest.raises(HTTPException) as excinfo:
        await _collect_non_stream(account, existing_sid="s1", messages=[ChatMessage(role="user", content=42)])
    assert excinfo.value.status_code == 404
    assert excinfo.value.detail == "session gone"
    assert account.client.completion.await_count == 1


async def test_stream_emits_the_event_left_in_the_incremental_buffer():
    account = FakeAccount([FakeResp(sse_text=READY + _initial("Tail").rstrip("\n"))])
    lines = await _stream(account)
    assert _client_text(lines) == "Tail"
    assert lines[-1] == "data: [DONE]\n\n"


async def test_stream_stops_the_message_identified_by_the_stream():
    upstream = READY + _initial("partial", "RESPONSE", 7)
    account = FakeAccount([StreamErrorResp(ValueError("bad frame"), sse_text=upstream)])
    gen = ds._stream_openai(
        account=account,
        pool=MagicMock(),
        existing_sid=None,
        lock=account.sem,
        prompt="x",
        model=MODEL,
        model_type="default",
        thinking=False,
        search=False,
    )
    with pytest.raises(ValueError, match="bad frame"):
        await _drain(gen)
    account.client.stop_stream.assert_awaited_once_with("c1", 7)


def _client_response(payload=None, status: int = 200, text: str = ""):
    resp = SimpleNamespace(status_code=status, text=text)
    resp.raise_for_status = MagicMock()
    resp.json = MagicMock(return_value=payload) if payload is not None else MagicMock(side_effect=ValueError("no json"))
    return resp


def _http_client() -> DeepSeekClient:
    client = DeepSeekClient(token="tok")
    client.http = MagicMock()
    return client


async def test_check_auth_rejects_an_unparseable_body():
    client = _http_client()
    client.http.get = AsyncMock(return_value=_client_response())
    assert await client.check_auth() is False


async def test_fetch_models_maps_a_transport_failure():
    client = _http_client()
    client.http.get = AsyncMock(side_effect=httpx.ConnectError("boom"))
    with pytest.raises(DeepSeekError) as excinfo:
        await client.fetch_models()
    assert excinfo.value.biz_code == -1
    assert str(excinfo.value) == "DeepSeek biz error -1: http request failed: boom"


async def test_fetch_models_rejects_a_non_200_settings_response():
    client = _http_client()
    client.http.get = AsyncMock(return_value=_client_response(status=503, text="down"))
    with pytest.raises(DeepSeekError) as excinfo:
        await client.fetch_models()
    assert excinfo.value.biz_code == 503
    assert str(excinfo.value) == "DeepSeek biz error 503: model settings returned 503"


async def test_fetch_models_rejects_an_unparseable_settings_body():
    client = _http_client()
    client.http.get = AsyncMock(return_value=_client_response(text="<html>"))
    with pytest.raises(DeepSeekError) as excinfo:
        await client.fetch_models()
    assert str(excinfo.value) == "DeepSeek biz error -1: invalid JSON from /api/v0/client/settings: <html>"


async def test_fetch_models_returns_nothing_when_the_catalog_has_no_list():
    client = _http_client()
    client.http.get = AsyncMock(return_value=_client_response({"code": 0, "data": {"biz_data": {"settings": {"model_configs": {"value": {"a": 1}}}}}}))
    assert await client.fetch_models() == []
    client.http.get = AsyncMock(return_value=_client_response({"code": 0, "data": {"biz_data": {"settings": []}}}))
    assert await client.fetch_models() == []
    client.http.get = AsyncMock(return_value=_client_response({"code": 0, "data": {}}))
    assert await client.fetch_models() == []


async def test_fetch_models_skips_malformed_config_entries():
    configs = ["garbage", {"name": "no type"}, {"model_type": ""}, {"model_type": 7}, {"model_type": "live", "enabled": True}]
    client = _http_client()
    client.http.get = AsyncMock(return_value=_client_response({"code": 0, "data": {"biz_data": {"settings": {"model_configs": {"value": configs}}}}}))
    models = await client.fetch_models()
    assert models == [
        {
            "id": "live",
            "name": "live",
            "owned_by": "deepseek",
            "model_type": "live",
            "is_default": False,
            "switchable": False,
            "supports_thinking": False,
            "supports_search": False,
        }
    ]


async def test_fetch_models_logs_every_disabled_type_in_one_debug_record(caplog):
    disabled = [{"model_type": f"m{index}", "enabled": False} for index in range(7)]
    configs = [*disabled, {"model_type": "live", "enabled": True, "name": "Live", "is_default": True, "switchable": True, "think_feature": True}]
    client = _http_client()
    client.http.get = AsyncMock(return_value=_client_response({"code": 0, "data": {"biz_data": {"settings": {"model_configs": {"value": configs}}}}}))
    with caplog.at_level(logging.DEBUG, logger="danyapi.deepseek"):
        models = await client.fetch_models()
    assert [model["id"] for model in models] == ["live"]
    assert models[0]["name"] == "Live"
    records = [record for record in caplog.records if record.name == "danyapi.deepseek"]
    assert len(records) == 1
    assert records[0].levelno == logging.DEBUG
    assert records[0].getMessage() == "deepseek model types disabled upstream: m0, m1, m2, m3, m4, m5, m6"


async def test_fetch_models_is_silent_when_nothing_is_disabled(caplog):
    configs = [{"model_type": "live", "enabled": True}]
    client = _http_client()
    client.http.get = AsyncMock(return_value=_client_response({"code": 0, "data": {"biz_data": {"settings": {"model_configs": {"value": configs}}}}}))
    with caplog.at_level(logging.DEBUG, logger="danyapi.deepseek"):
        assert len(await client.fetch_models()) == 1
    assert [record for record in caplog.records if record.name == "danyapi.deepseek"] == []


async def test_post_and_request_json_report_the_same_decode_error():
    entry_points = [
        lambda client: client._post("/api/v0/x"),
        lambda client: client._request_json("POST", "/api/v0/x", json=None),
    ]
    messages = []
    for entry in entry_points:
        client = _http_client()
        client.http.post = AsyncMock(return_value=_client_response(text="not json"))
        with pytest.raises(DeepSeekError) as excinfo:
            await entry(client)
        messages.append(str(excinfo.value))
    assert messages[0] == messages[1]
    assert messages[0] == "DeepSeek biz error -1: invalid JSON response from /api/v0/x: not json"


async def test_request_json_uses_the_caller_supplied_decode_error():
    client = _http_client()
    client.http.post = AsyncMock(return_value=_client_response(text="not json"))
    with pytest.raises(DeepSeekError) as excinfo:
        await client.upload_file(b"d", "a.txt", "text/plain", "default")
    assert str(excinfo.value) == "DeepSeek biz error -1: invalid JSON from file upload"


async def test_request_json_selects_get_for_other_methods():
    client = _http_client()
    client.http.get = AsyncMock(return_value=_client_response({"code": 0, "data": {"biz_data": {"files": []}}}))
    assert await client.fetch_files(["f1"]) == []
    client.http.get.assert_awaited_once_with("/api/v0/file/fetch_files", params={"file_ids": "f1"})


async def test_stream_completion_closes_the_response_on_normal_exit():
    client = _http_client()
    resp = MagicMock()
    resp.aclose = AsyncMock()
    client.completion = AsyncMock(return_value=resp)
    async with client.stream_completion("cs1", "hi", "p1") as streamed:
        assert streamed is resp
    resp.aclose.assert_awaited_once()
    client.completion.assert_awaited_once_with("cs1", "hi", "p1", "default", False, False, None, None)


async def test_stream_completion_closes_the_response_when_the_body_raises():
    client = _http_client()
    resp = MagicMock()
    resp.aclose = AsyncMock()
    client.completion = AsyncMock(return_value=resp)
    with pytest.raises(ValueError, match="boom"):
        async with client.stream_completion("cs1", "hi", None):
            raise ValueError("boom")
    resp.aclose.assert_awaited_once()


async def test_stream_completion_closes_the_response_on_cancellation():
    client = _http_client()
    resp = MagicMock()
    resp.aclose = AsyncMock()
    client.completion = AsyncMock(return_value=resp)
    started = asyncio.Event()

    async def consume():
        async with client.stream_completion("cs1", "hi", None):
            started.set()
            await asyncio.Event().wait()

    task = asyncio.create_task(consume())
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    resp.aclose.assert_awaited_once()


async def test_stream_completion_swallows_a_failing_close():
    client = _http_client()
    resp = MagicMock()
    resp.aclose = AsyncMock(side_effect=RuntimeError("already closed"))
    client.completion = AsyncMock(return_value=resp)
    async with client.stream_completion("cs1", "hi", None) as streamed:
        assert streamed is resp
