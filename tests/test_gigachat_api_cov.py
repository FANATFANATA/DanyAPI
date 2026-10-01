import asyncio
import json
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from fastapi import HTTPException

from danyapi.api import retry as retry_module
from danyapi.api.retry import RETRY_BACKOFF_MAX_SEC, RETRY_BACKOFF_SEC
from danyapi.api.schemas import ChatMessage
from danyapi.api.sse import STREAM_ERROR_FINISH
from danyapi.gigachat import api as ga
from danyapi.gigachat.client import GigaChatError

MESSAGES = [ChatMessage(role="user", content="hi")]
MODEL = "GigaChat-2-Max"

ONE_TOOL = [{"type": "function", "function": {"name": "f", "description": "d", "parameters": {"type": "object"}}}]
TWO_TOOLS = [
    {"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}},
    {"type": "function", "function": {"name": "g", "parameters": {"type": "object"}}},
]

ZERO_USAGE = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
STREAM_USAGE = {
    "prompt_tokens": 5,
    "completion_tokens": 7,
    "total_tokens": 12,
    "prompt_tokens_details": {"cached_tokens": 0},
    "completion_tokens_details": {"reasoning_tokens": 0},
}
CREATED = 1700000000
HEX = "0123456789abcdef0123456789abcdef"
CHUNK_ID = f"chatcmpl-{HEX}"


def _neutral_jitter(low: float, high: float) -> float:
    return (low + high) / 2


class _FakeResponse(httpx.Response):
    def __init__(
        self,
        chunks: list[bytes] | None = None,
        status_code: int = 200,
        headers: dict[str, str] | None = None,
        fail: Exception | None = None,
    ) -> None:
        super().__init__(status_code, content=b"".join(chunks or []), headers=headers)
        self._chunks = list(chunks or [])
        self._fail = fail
        self.closed = False

    async def aiter_bytes(self, chunk_size: int | None = None) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            yield chunk
        if self._fail is not None:
            raise self._fail

    async def aclose(self) -> None:
        self.closed = True


class _Client:
    def __init__(self, responses: list[Any]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[dict, str]] = []

    async def chat(self, body: dict, model: str) -> Any:
        self.calls.append((body, model))
        item = self.responses.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


class _Account:
    def __init__(self, client: Any) -> None:
        self.client = client
        self.sem = asyncio.Semaphore(1)
        self.broken = False

    def mark_broken(self) -> None:
        self.broken = True


def _json_response(payload: Any, status_code: int = 200) -> _FakeResponse:
    return _FakeResponse([json.dumps(payload, ensure_ascii=False).encode()], status_code)


def _completion(content: str = "hello", finish: str = "stop", usage: dict | None = None) -> _FakeResponse:
    payload: dict[str, Any] = {
        "id": "cmpl-1",
        "created": 1700000000,
        "model": MODEL,
        "choices": [{"message": {"role": "assistant", "content": content}, "finish_reason": finish}],
    }
    if usage is not None:
        payload["usage"] = usage
    return _json_response(payload)


def _sse(*payloads: Any) -> bytes:
    return b"".join(f"data: {json.dumps(p, ensure_ascii=False)}\n\n".encode() for p in payloads)


def _parse(lines: list[str]) -> list[dict]:
    return [json.loads(line[len("data: ") :]) for line in lines if line.startswith("data: ") and line != ga.DONE_LINE]


def _backoff_recorder(recorded: list[int]) -> Any:
    async def backoff(attempt: int) -> None:
        recorded.append(attempt)
        await asyncio.sleep(0)

    return backoff


def _usage_recorder(recorded: list[tuple]) -> Any:
    def record(*args: Any, **kwargs: Any) -> None:
        recorded.append((args, kwargs))

    return record


def _fixed_clock(monkeypatch) -> None:
    monkeypatch.setattr(ga, "time", SimpleNamespace(time=lambda: float(CREATED)))
    monkeypatch.setattr(ga, "uuid", SimpleNamespace(uuid4=lambda: SimpleNamespace(hex=HEX)))


def _chunk(payload: dict) -> dict:
    return {"id": CHUNK_ID, "object": "chat.completion.chunk", "created": CREATED, "model": MODEL, **payload}


def _chunk_line(delta: dict, finish: str | None) -> str:
    return f"data: {json.dumps(_chunk({'choices': [{'index': 0, 'delta': delta, 'finish_reason': finish}]}), ensure_ascii=False)}\n\n"


def _error_line(message: str, session_id: str | None = None) -> str:
    payload = _chunk({"session_id": session_id, "error": {"message": message}, "choices": [{"index": 0, "delta": {}, "finish_reason": STREAM_ERROR_FINISH}]})
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def test_translate_message_defaults_a_missing_or_non_dict_message():
    assert ga._translate_message({}) == {"role": "assistant", "content": ""}
    assert ga._translate_message({"message": "raw text"}) == {"role": "assistant", "content": ""}
    assert ga._translate_message({"message": {"content": 42}}) == {"role": "assistant", "content": ""}
    assert ga._translate_message({"message": {"content": "ok"}}) == {"role": "assistant", "content": "ok"}


def test_translate_message_fills_missing_function_call_arguments_and_id():
    message = ga._translate_message({"message": {"content": "", "function_call": {"name": "f", "arguments": None}}})
    call = message["tool_calls"][0]
    assert call["id"].startswith("call_")
    assert call["function"] == {"name": "f", "arguments": "{}"}


def test_translate_message_keeps_a_supplied_call_id_and_string_arguments():
    message = ga._translate_message({"message": {"content": "", "function_call": {"id": "fc-7", "name": "f", "arguments": '{"a": 1}'}}})
    assert message["tool_calls"][0]["id"] == "fc-7"
    assert message["tool_calls"][0]["function"]["arguments"] == '{"a": 1}'


def test_translate_message_replaces_a_non_string_call_id():
    message = ga._translate_message({"message": {"content": "", "function_call": {"id": 12, "name": "f", "arguments": "{}"}}})
    assert message["tool_calls"][0]["id"].startswith("call_")


def test_translate_message_ignores_a_function_call_without_a_name():
    assert ga._translate_message({"message": {"content": "hi", "function_call": {"name": "", "arguments": "{}"}}}) == {"role": "assistant", "content": "hi"}


async def test_send_retries_a_transport_error_and_then_succeeds(monkeypatch):
    attempts: list[int] = []
    monkeypatch.setattr(ga, "_sleep_backoff", _backoff_recorder(attempts))
    client = _Client([httpx.ConnectError("connection refused"), _json_response({"ok": True})])
    account = _Account(client)

    resp = await ga._send(account, {"messages": []}, MODEL)

    assert json.loads(resp.content) == {"ok": True}
    assert attempts == [0]
    assert len(client.calls) == 2
    assert account.broken is False


async def test_send_exhausting_the_shared_retry_budget_raises_502(monkeypatch):
    attempts: list[int] = []
    monkeypatch.setattr(ga, "_sleep_backoff", _backoff_recorder(attempts))
    client = _Client([httpx.ReadTimeout("upstream stalled") for _ in range(ga.MAX_RETRIES + 1)])
    account = _Account(client)

    with pytest.raises(HTTPException) as exc:
        await ga._send(account, {"messages": []}, MODEL)

    assert exc.value.status_code == 502
    assert exc.value.detail == "GigaChat transport error: upstream stalled"
    assert attempts == list(range(ga.MAX_RETRIES))
    assert len(client.calls) == ga.MAX_RETRIES + 1
    assert ga.MAX_RETRIES == 5


async def test_send_marks_the_account_broken_on_an_auth_error():
    client = _Client([GigaChatError(401, "authorization key is invalid")])
    account = _Account(client)

    with pytest.raises(HTTPException) as exc:
        await ga._send(account, {"messages": []}, MODEL)

    assert exc.value.status_code == 401
    assert exc.value.detail == "GigaChat error: authorization key is invalid"
    assert account.broken is True
    assert len(client.calls) == 1


async def test_send_reraises_a_non_auth_gigachat_error():
    client = _Client([GigaChatError(400, "bad request")])
    account = _Account(client)

    with pytest.raises(GigaChatError) as exc:
        await ga._send(account, {"messages": []}, MODEL)

    assert exc.value.code == 400
    assert exc.value.message == "bad request"
    assert account.broken is False
    assert len(client.calls) == 1


async def test_sleep_backoff_uses_the_shared_retry_delay(monkeypatch):
    from danyapi.api.retry import _retry_delay

    delays: list[float] = []

    async def fake_sleep(delay: float) -> None:
        delays.append(delay)

    monkeypatch.setattr(retry_module, "RETRY_BACKOFF_SEC", RETRY_BACKOFF_SEC)
    monkeypatch.setattr(retry_module.random, "uniform", _neutral_jitter)
    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    await ga._sleep_backoff(0)
    await ga._sleep_backoff(2)

    assert delays == [_retry_delay(1), _retry_delay(3)]
    assert delays == [RETRY_BACKOFF_SEC, min(RETRY_BACKOFF_MAX_SEC, RETRY_BACKOFF_SEC * 4)]


async def test_read_json_turns_a_malformed_body_into_502():
    resp = _FakeResponse([b"<html>not json</html>"])

    with pytest.raises(HTTPException) as exc:
        await ga._read_json(resp)

    assert exc.value.status_code == 502
    assert exc.value.detail == "GigaChat returned a malformed response"


async def test_read_json_decodes_a_valid_body():
    assert await ga._read_json(_FakeResponse([b'{"a": 1}'])) == {"a": 1}


async def test_safe_json_returns_none_instead_of_raising():
    assert await ga._safe_json(_FakeResponse([b"gateway timeout"])) is None
    assert await ga._safe_json(_FakeResponse([b'{"message": "boom"}'])) == {"message": "boom"}


def _unread_response(status_code: int, body: bytes) -> httpx.Response:
    resp = httpx.Response(status_code, headers={"content-type": "application/json"}, stream=httpx.ByteStream(body))
    assert resp.is_stream_consumed is False
    return resp


async def test_safe_json_reads_an_unconsumed_stream_before_parsing():
    resp = _unread_response(401, b'{"message": "expired key"}')

    assert await ga._safe_json(resp) == {"message": "expired key"}


async def test_safe_json_reports_an_unreadable_stream_as_a_parse_failure():
    resp = _unread_response(502, b'{"message": "boom"}')

    async def _boom() -> None:
        raise httpx.ReadError("socket closed")

    resp.aread = _boom  # type: ignore[method-assign]

    assert await ga._safe_json(resp) is None


async def test_raise_upstream_uses_the_upstream_message():
    resp = _FakeResponse([b'{"message": "quota exceeded"}'], 429)

    with pytest.raises(HTTPException) as exc:
        await ga._raise_upstream(_Account(_Client([])), resp, {"message": "quota exceeded"})

    assert exc.value.status_code == 429
    assert exc.value.detail == "GigaChat error: quota exceeded"


async def test_raise_upstream_falls_back_to_the_raw_body():
    resp = _FakeResponse([b"upstream exploded"], 400)

    with pytest.raises(HTTPException) as exc:
        await ga._raise_upstream(_Account(_Client([])), resp, {"message": 7})

    assert exc.value.status_code == 400
    assert exc.value.detail == "GigaChat error: upstream exploded"


async def test_raise_upstream_without_a_body_reports_the_status():
    resp = _FakeResponse([], 503)

    with pytest.raises(HTTPException) as exc:
        await ga._raise_upstream(_Account(_Client([])), resp, None)

    assert exc.value.status_code == 502
    assert exc.value.detail == "GigaChat error: upstream returned 503"


async def test_raise_upstream_reads_an_unconsumed_stream_and_marks_the_account_broken():
    account = _Account(_Client([]))
    resp = _unread_response(403, b'{"message": "key revoked"}')

    with pytest.raises(HTTPException) as exc:
        await ga._raise_upstream(account, resp, {"message": "key revoked"})

    assert exc.value.status_code == 401
    assert exc.value.detail == "GigaChat error: key revoked"
    assert account.broken is True


async def test_collect_non_stream_marks_the_account_broken_when_the_client_returns_401():
    account = _Account(_Client([_unread_response(401, b'{"message": "invalid access token"}')]))

    with pytest.raises(HTTPException) as exc:
        await ga.collect_non_stream(account, MESSAGES, MODEL)

    assert exc.value.status_code == 401
    assert exc.value.detail == "GigaChat error: invalid access token"
    assert account.broken is True


async def test_collect_non_stream_marks_the_account_broken_after_the_client_refreshed_the_token():
    from danyapi.gigachat.client import BASE_URL, GigaChatClient

    key = "9f2c1a4e-0b7d-4c8e-9a1b-2f3c4d5e6f70"
    calls: list[str] = []

    def _handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        if request.url.path.endswith("/api/v2/oauth"):
            return httpx.Response(200, json={"access_token": "tok" * 12, "expires_at": 4102444800})
        return httpx.Response(401, json={"message": "invalid access token"})

    client = GigaChatClient(key=key)
    client.http = httpx.AsyncClient(base_url=BASE_URL, transport=httpx.MockTransport(_handler))
    account = _Account(client)

    with pytest.raises(HTTPException) as exc:
        await ga.collect_non_stream(account, MESSAGES, MODEL)

    assert exc.value.status_code == 401
    assert exc.value.detail == "GigaChat error: invalid access token"
    assert account.broken is True
    assert calls.count("/v1/chat/completions") == 2


async def test_stream_reports_the_real_upstream_reason_from_an_unconsumed_error_stream():
    account = _Account(_Client([_unread_response(401, b'{"message": "access token expired"}')]))

    with pytest.raises(HTTPException) as exc:
        async for _line in ga.stream_openai(account, MESSAGES, MODEL):
            pass

    assert exc.value.status_code == 401
    assert exc.value.detail == "GigaChat error: access token expired"
    assert account.broken is True


async def test_stream_reports_an_unreadable_error_body_as_an_upstream_status():
    account = _Account(_Client([_unread_response(429, b'{"message": "rate limited"}')]))
    resp = account.client.responses[0]

    async def _boom() -> None:
        raise httpx.ReadError("socket closed")

    resp.aread = _boom  # type: ignore[method-assign]

    with pytest.raises(HTTPException) as exc:
        async for _line in ga.stream_openai(account, MESSAGES, MODEL):
            pass

    assert exc.value.status_code == 429
    assert exc.value.detail == "GigaChat error: upstream returned 429"
    assert account.broken is False


def test_chunk_carries_the_delta_and_only_adds_usage_when_given():
    plain = ga._chunk("c1", 7, MODEL, {"content": "Привет"}, "stop")
    expected = (
        'data: {"id": "c1", "object": "chat.completion.chunk", "created": 7, "model": "GigaChat-2-Max",'
        ' "choices": [{"index": 0, "delta": {"content": "Привет"}, "finish_reason": "stop"}]}\n\n'
    )
    assert plain == expected
    assert json.loads(plain[len("data: ") :])["choices"][0]["delta"] == {"content": "Привет"}

    with_usage = ga._chunk("c1", 7, MODEL, {}, None, usage=STREAM_USAGE)
    payload = json.loads(with_usage[len("data: ") :])
    assert payload["usage"] == STREAM_USAGE
    assert payload["choices"] == [{"index": 0, "delta": {}, "finish_reason": None}]
    assert "usage" not in json.loads(ga._chunk("c1", 7, MODEL, {})[len("data: ") :])


def test_events_drops_done_empty_and_non_dict_frames():
    parser = ga.IncrementalSSE()
    raw = b'data: {"a": 1}\n\ndata: [DONE]\n\ndata: "plain"\n\ndata: {}\n\ndata: 7\n\n'

    assert list(ga._events(parser.feed(raw))) == [{"a": 1}]


async def test_iter_sse_survives_a_multibyte_character_split_across_chunks():
    first = {"choices": [{"delta": {"content": "Привет, мир"}}]}
    second = {"choices": [{"delta": {"content": "!"}}]}
    raw = _sse(first, second)
    cut = raw.index(b"\xd0\x9f") + 1
    resp = _FakeResponse([raw[:cut], raw[cut:]])

    assert [event async for event in ga._iter_sse(resp)] == [first, second]


async def test_iter_sse_delivers_a_final_event_without_a_blank_line():
    resp = _FakeResponse([b'data: {"a": 1}\n\ndata: {"a": 2}'])

    assert [event async for event in ga._iter_sse(resp)] == [{"a": 1}, {"a": 2}]


async def test_iter_sse_accepts_crlf_framing():
    resp = _FakeResponse([b'data: {"a": 1}\r\n\r\ndata: {"a": 2}\r\n\r\n'])

    assert [event async for event in ga._iter_sse(resp)] == [{"a": 1}, {"a": 2}]


async def test_iter_sse_accepts_data_without_a_space():
    resp = _FakeResponse([b'data:{"a": 1}\n\ndata:{"b": 2}'])

    assert [event async for event in ga._iter_sse(resp)] == [{"a": 1}, {"b": 2}]


async def test_iter_sse_of_an_empty_body_yields_nothing():
    assert [event async for event in ga._iter_sse(_FakeResponse([]))] == []


def test_delta_from_event_reads_a_message_when_delta_is_absent():
    assert ga._delta_from_event({"choices": [{"message": {"content": "hi", "role": "assistant"}}]}) == ({"content": "hi", "role": "assistant"}, None)


@pytest.mark.parametrize("event", [{}, {"choices": []}, {"choices": "x"}, {"choices": ["x"]}, {"choices": [{"content": "raw"}]}])
def test_delta_from_event_ignores_frames_without_a_usable_choice(event):
    assert ga._delta_from_event(event) == ({}, None)


def test_delta_from_event_ignores_empty_content_role_and_unnamed_function_call():
    assert ga._delta_from_event({"choices": [{"delta": {"content": "", "role": "", "function_call": {"name": "", "arguments": "{}"}}}]}) == ({}, None)


def test_delta_from_event_ignores_content_and_role_of_the_wrong_type():
    assert ga._delta_from_event({"choices": [{"delta": {"content": 7, "role": 9, "function_call": "f"}}]}) == ({}, None)


def test_delta_from_event_normalises_a_function_call():
    event = {"choices": [{"delta": {"function_call": {"name": "f", "arguments": {"city": "Moscow"}, "id": 7}}, "finish_reason": "function_call"}]}

    delta, finish = ga._delta_from_event(event)

    call = delta["tool_calls"][0]
    assert call["index"] == 0
    assert call["id"].startswith("call_")
    assert call["type"] == "function"
    assert call["function"] == {"name": "f", "arguments": "{}"}
    assert finish == "function_call"


def test_delta_from_event_keeps_a_supplied_call_id_and_string_arguments():
    event = {"choices": [{"delta": {"function_call": {"id": "call_9", "name": "f", "arguments": '{"a": 1}'}}, "finish_reason": "stop"}]}

    delta, finish = ga._delta_from_event(event)

    assert delta["tool_calls"][0]["id"] == "call_9"
    assert delta["tool_calls"][0]["function"] == {"name": "f", "arguments": '{"a": 1}'}
    assert finish == "stop"


def test_delta_from_event_drops_a_non_string_finish_reason():
    assert ga._delta_from_event({"choices": [{"delta": {"content": "a"}, "finish_reason": 5}]}) == ({"content": "a"}, None)


async def test_collect_non_stream_returns_an_openai_shaped_completion(monkeypatch):
    recorded: list[tuple] = []
    monkeypatch.setattr(ga, "record_usage_dict", _usage_recorder(recorded))
    client = _Client([_completion("Привет", "stop", {"prompt_tokens": 3, "completion_tokens": 4})])
    account = _Account(client)

    result = await ga.collect_non_stream(account, MESSAGES, MODEL, user="u1", session_id="s1")

    assert result["id"] == "cmpl-1"
    assert result["object"] == "chat.completion"
    assert result["created"] == CREATED
    assert result["model"] == MODEL
    assert result["system_fingerprint"] == MODEL
    assert result["choices"] == [{"index": 0, "message": {"role": "assistant", "content": "Привет"}, "finish_reason": "stop", "logprobs": None}]
    assert result["usage"] == {
        "prompt_tokens": 3,
        "completion_tokens": 4,
        "total_tokens": 7,
        "prompt_tokens_details": {"cached_tokens": 0},
        "completion_tokens_details": {"reasoning_tokens": 0},
    }
    assert result["session_id"] == "s1"
    assert recorded == [(("gigachat", MODEL, result["usage"]), {"user": "u1", "session_id": "s1"})]
    assert client.calls == [({"messages": [{"role": "user", "content": "hi"}]}, MODEL)]


async def test_collect_non_stream_invents_id_and_fingerprint_when_upstream_omits_them(monkeypatch):
    _fixed_clock(monkeypatch)
    monkeypatch.setattr(ga, "record_usage_dict", _usage_recorder([]))
    client = _Client([_json_response({"choices": ["oops"]})])

    result = await ga.collect_non_stream(_Account(client), MESSAGES, MODEL)

    assert result == {
        "id": CHUNK_ID,
        "object": "chat.completion",
        "created": CREATED,
        "model": MODEL,
        "system_fingerprint": "fp_danyapi",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": ""}, "finish_reason": "stop", "logprobs": None}],
        "usage": ZERO_USAGE,
        "session_id": None,
    }


async def test_collect_non_stream_truncates_the_answer_at_the_stop_marker():
    client = _Client([_completion("keep STOP drop", "stop", {"prompt_tokens": 1, "completion_tokens": 1})])

    result = await ga.collect_non_stream(_Account(client), MESSAGES, MODEL, stop=["STOP"])

    assert result["choices"][0]["message"]["content"] == "keep "


async def test_collect_non_stream_normalises_the_finish_reason():
    client = _Client([_completion("hi", "function_call", {})])

    result = await ga.collect_non_stream(_Account(client), MESSAGES, MODEL)

    assert result["choices"][0]["finish_reason"] == "tool_calls"


async def test_collect_non_stream_sends_functions_and_sampling_settings():
    client = _Client([_completion()])

    await ga.collect_non_stream(
        _Account(client),
        MESSAGES,
        "GigaChat",
        tools=ONE_TOOL,
        tool_choice="required",
        temperature=0.3,
        top_p=0.8,
        max_tokens=64,
        response_format={"type": "json_object"},
    )

    body, model = client.calls[0]
    assert model == "GigaChat"
    assert body == {
        "messages": [{"role": "user", "content": "hi"}],
        "functions": [{"name": "f", "description": "d", "parameters": {"type": "object"}}],
        "function_call": {"name": "f"},
        "temperature": 0.3,
        "top_p": 0.8,
        "max_tokens": 64,
        "response_format": {"type": "json_object"},
    }


async def test_collect_non_stream_rejects_a_non_dict_payload():
    client = _Client([_FakeResponse([b"[1, 2, 3]"])])

    with pytest.raises(HTTPException) as exc:
        await ga.collect_non_stream(_Account(client), MESSAGES, MODEL)

    assert exc.value.status_code == 502
    assert exc.value.detail == "GigaChat returned an unexpected payload"


@pytest.mark.parametrize("payload", [{}, {"choices": []}, {"choices": "x"}])
async def test_collect_non_stream_rejects_a_payload_without_choices(payload):
    client = _Client([_json_response(payload)])

    with pytest.raises(HTTPException) as exc:
        await ga.collect_non_stream(_Account(client), MESSAGES, MODEL)

    assert exc.value.status_code == 502
    assert exc.value.detail == "GigaChat returned no choices"


async def test_collect_non_stream_closes_the_response_on_success():
    resp = _completion()
    client = _Client([resp])

    await ga.collect_non_stream(_Account(client), MESSAGES, MODEL)

    assert resp.closed is True


async def test_collect_non_stream_maps_an_upstream_error_status():
    resp = _FakeResponse([b'{"message": "too many requests"}'], 429)
    client = _Client([resp])

    with pytest.raises(HTTPException) as exc:
        await ga.collect_non_stream(_Account(client), MESSAGES, MODEL)

    assert exc.value.status_code == 429
    assert exc.value.detail == "GigaChat error: too many requests"
    assert resp.closed is True


async def test_collect_non_stream_reports_a_non_json_upstream_error_body():
    client = _Client([_FakeResponse([b"gateway down"], 502)])

    with pytest.raises(HTTPException) as exc:
        await ga.collect_non_stream(_Account(client), MESSAGES, MODEL)

    assert exc.value.status_code == 502
    assert exc.value.detail == "GigaChat error: gateway down"


async def test_collect_non_stream_converts_a_non_auth_gigachat_error():
    client = _Client([GigaChatError(400, "Model does not support image")])

    with pytest.raises(HTTPException) as exc:
        await ga.collect_non_stream(_Account(client), MESSAGES, MODEL)

    assert exc.value.status_code == 400
    assert exc.value.detail == f"GigaChat error: {ga.NO_IMAGE_MODELS_HINT}"
    assert "does not support image" not in exc.value.detail


async def test_collect_non_stream_converts_a_non_auth_gigachat_error_with_a_non_int_code():
    client = _Client([GigaChatError("oops", "upstream said no")])

    with pytest.raises(HTTPException) as exc:
        await ga.collect_non_stream(_Account(client), MESSAGES, MODEL)

    assert exc.value.status_code == 502
    assert exc.value.detail == "GigaChat error: upstream said no"


async def test_collect_non_stream_propagates_the_401_from_an_auth_failure():
    account = _Account(_Client([GigaChatError(403, "key revoked")]))

    with pytest.raises(HTTPException) as exc:
        await ga.collect_non_stream(account, MESSAGES, MODEL)

    assert exc.value.status_code == 401
    assert account.broken is True


async def test_collect_non_stream_turns_exhausted_retries_into_502(monkeypatch):
    attempts: list[int] = []
    monkeypatch.setattr(ga, "_sleep_backoff", _backoff_recorder(attempts))
    client = _Client([httpx.ConnectError("no route to host") for _ in range(ga.MAX_RETRIES + 1)])

    with pytest.raises(HTTPException) as exc:
        await ga.collect_non_stream(_Account(client), MESSAGES, MODEL)

    assert exc.value.status_code == 502
    assert exc.value.detail.startswith("GigaChat transport error: ")
    assert len(client.calls) == ga.MAX_RETRIES + 1


async def test_stream_openai_reassembles_a_clean_stream(monkeypatch):
    recorded: list[tuple] = []
    monkeypatch.setattr(ga, "record_usage_dict", _usage_recorder(recorded))
    resp = _FakeResponse(
        [
            _sse(
                {"choices": [{"delta": {"role": "assistant", "content": "Привет"}}]},
                {"choices": [{"delta": {"content": ", мир"}}]},
                {"choices": [{"delta": {}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 5, "completion_tokens": 7}},
            )
            + ga.DONE_LINE.encode()
        ],
        headers={"content-type": "text/event-stream"},
    )
    client = _Client([resp])

    lines = [line async for line in ga.stream_openai(_Account(client), MESSAGES, MODEL)]

    assert lines[-1] == ga.DONE_LINE
    events = _parse(lines)
    assert [e["choices"][0]["delta"] for e in events] == [
        {"role": "assistant", "content": "Привет"},
        {"content": ", мир"},
        {},
    ]
    assert [e["choices"][0]["finish_reason"] for e in events] == [None, None, "stop"]
    assert "".join(e["choices"][0]["delta"].get("content", "") for e in events) == "Привет, мир"
    assert "usage" not in events[-1]
    assert recorded == [(("gigachat", MODEL, STREAM_USAGE), {"user": None, "session_id": None})]
    assert client.calls[0][0]["stream"] is True
    assert resp.closed is True


async def test_stream_records_usage_even_without_include_usage(monkeypatch):
    recorded: list[tuple] = []
    monkeypatch.setattr(ga, "record_usage_dict", _usage_recorder(recorded))
    raw = _sse(
        {"choices": [{"delta": {"role": "assistant", "content": "hi"}}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 5, "completion_tokens": 7}},
    )
    client = _Client([_FakeResponse([raw])])

    lines = [line async for line in ga.stream_openai(_Account(client), MESSAGES, MODEL, include_usage=False)]

    assert recorded == [(("gigachat", MODEL, STREAM_USAGE), {"user": None, "session_id": None})]
    assert all("usage" not in event for event in _parse(lines))
    assert len(_parse(lines)) == 2


async def test_stream_emits_the_usage_chunk_when_include_usage_is_set(monkeypatch):
    recorded: list[tuple] = []
    monkeypatch.setattr(ga, "record_usage_dict", _usage_recorder(recorded))
    raw = _sse(
        {"choices": [{"delta": {"content": "hi"}}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 5, "completion_tokens": 7}},
    )
    client = _Client([_FakeResponse([raw])])

    lines = [line async for line in ga.stream_openai(_Account(client), MESSAGES, MODEL, include_usage=True)]

    events = _parse(lines)
    assert len(events) == 3
    assert events[-1]["usage"] == STREAM_USAGE
    assert events[-1]["choices"] == [{"index": 0, "delta": {}, "finish_reason": None}]
    assert recorded == [(("gigachat", MODEL, STREAM_USAGE), {"user": None, "session_id": None})]


async def test_stream_records_zero_usage_when_upstream_sends_none(monkeypatch):
    recorded: list[tuple] = []
    monkeypatch.setattr(ga, "record_usage_dict", _usage_recorder(recorded))
    raw = _sse({"choices": [{"delta": {"content": "hi"}, "finish_reason": "stop"}]})
    client = _Client([_FakeResponse([raw])])

    lines = [line async for line in ga.stream_openai(_Account(client), MESSAGES, MODEL, include_usage=True)]

    assert recorded == [(("gigachat", MODEL, ZERO_USAGE), {"user": None, "session_id": None})]
    assert all("usage" not in event for event in _parse(lines))


async def test_stream_surfaces_an_upstream_4xx_instead_of_an_empty_completion():
    resp = _FakeResponse([b'{"message": "rate limited"}'], 429)
    client = _Client([resp])
    emitted: list[str] = []

    with pytest.raises(HTTPException) as exc:
        async for line in ga.stream_openai(_Account(client), MESSAGES, MODEL):
            emitted.append(line)

    assert exc.value.status_code == 429
    assert exc.value.detail == "GigaChat error: rate limited"
    assert emitted == []
    assert resp.closed is True


async def test_stream_wraps_a_mid_stream_transport_error_in_502():
    resp = _FakeResponse([_sse({"choices": [{"delta": {"content": "hi"}}]})], fail=httpx.ReadError("connection reset by peer"))
    client = _Client([resp])

    with pytest.raises(HTTPException) as exc:
        async for _line in ga.stream_openai(_Account(client), MESSAGES, MODEL):
            pass

    assert exc.value.status_code == 502
    assert exc.value.detail == "GigaChat stream transport error: connection reset by peer"
    assert resp.closed is True


async def test_stream_falls_back_to_an_empty_completion_on_an_unexpected_error(monkeypatch):
    _fixed_clock(monkeypatch)
    resp = _FakeResponse([b"data: not json at all\n\n"], fail=RuntimeError("frame decoder exploded"))
    client = _Client([resp])

    lines = [line async for line in ga.stream_openai(_Account(client), MESSAGES, MODEL)]

    assert lines == [
        _chunk_line({}, "stop"),
        _chunk_line({"role": "assistant", "content": ""}, None),
        ga.DONE_LINE,
    ]
    assert _parse(lines) == [
        _chunk({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}),
        _chunk({"choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]}),
    ]
    assert resp.closed is True


async def test_stream_emits_an_empty_completion_when_nothing_was_delivered(monkeypatch):
    _fixed_clock(monkeypatch)
    monkeypatch.setattr(ga, "record_usage_dict", _usage_recorder([]))
    client = _Client([_FakeResponse([b"data: [DONE]\n\n"])])

    events = _parse([line async for line in ga.stream_openai(_Account(client), MESSAGES, MODEL)])

    assert events == [_chunk({"choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]})]


async def test_stream_applies_the_stop_marker_inside_a_delta():
    client = _Client([_FakeResponse([_sse({"choices": [{"delta": {"content": "keep STOP drop"}}]})])])

    events = _parse([line async for line in ga.stream_openai(_Account(client), MESSAGES, MODEL, stop=["STOP"])])

    assert events[0]["choices"][0]["delta"] == {"content": "keep "}


async def test_stream_drops_the_content_when_the_stop_marker_empties_it():
    client = _Client([_FakeResponse([_sse({"choices": [{"delta": {"role": "assistant", "content": "STOP tail"}}]})])])

    events = _parse([line async for line in ga.stream_openai(_Account(client), MESSAGES, MODEL, stop="STOP")])

    assert events[0]["choices"][0]["delta"] == {"role": "assistant"}


async def test_stream_normalises_the_finish_reason_and_ignores_a_non_string_one():
    raw = _sse(
        {"choices": [{"delta": {"content": "hi"}, "finish_reason": "function_call"}]},
        {"choices": [{"delta": {"content": "there"}, "finish_reason": 7}]},
    )
    client = _Client([_FakeResponse([raw])])

    events = _parse([line async for line in ga.stream_openai(_Account(client), MESSAGES, MODEL)])

    assert [event["choices"][0]["delta"] for event in events] == [{"content": "hi"}, {}, {"content": "there"}]
    assert [event["choices"][0]["finish_reason"] for event in events] == [None, "tool_calls", None]


async def test_stream_carries_the_tool_call_from_the_stream():
    raw = _sse(
        {"choices": [{"delta": {"role": "assistant", "function_call": {"name": "f", "arguments": '{"city": "Moscow"}', "id": "call_3"}}}]},
    )
    client = _Client([_FakeResponse([raw])])

    events = _parse([line async for line in ga.stream_openai(_Account(client), MESSAGES, MODEL)])

    assert events[0]["choices"][0]["delta"]["tool_calls"] == [
        {"index": 0, "id": "call_3", "type": "function", "function": {"name": "f", "arguments": '{"city": "Moscow"}'}}
    ]


async def test_stream_reports_a_build_failure_as_an_error_chunk(monkeypatch):
    _fixed_clock(monkeypatch)
    client = _Client([])

    lines = [line async for line in ga.stream_openai(_Account(client), MESSAGES, MODEL, tools=TWO_TOOLS, tool_choice="required", session_id="s9")]

    assert lines == [
        _error_line("gigachat requires a function call only when exactly one function is provided, name it in tool_choice", "s9"),
        ga.DONE_LINE,
    ]
    assert client.calls == []


async def test_stream_wraps_an_unexpected_setup_failure_in_an_error_chunk(monkeypatch):
    _fixed_clock(monkeypatch)

    async def boom(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("account pool exploded")

    monkeypatch.setattr(ga, "build_messages", boom)

    lines = [line async for line in ga.stream_openai(_Account(_Client([])), MESSAGES, MODEL)]

    assert lines == [_error_line("GigaChat request failed: account pool exploded"), ga.DONE_LINE]


async def test_stream_stringifies_a_non_string_error_detail(monkeypatch):
    _fixed_clock(monkeypatch)

    async def failing_send(*args: Any, **kwargs: Any) -> Any:
        raise HTTPException(400, {"why": "quota"})

    monkeypatch.setattr(ga, "_send", failing_send)

    lines = [line async for line in ga.stream_openai(_Account(_Client([])), MESSAGES, MODEL)]

    assert _parse(lines)[0]["error"] == {"message": str({"why": "quota"})}
    assert _parse(lines)[0]["session_id"] is None
    assert _parse(lines)[0]["choices"] == [{"index": 0, "delta": {}, "finish_reason": STREAM_ERROR_FINISH}]


async def test_stream_closes_the_upstream_response_when_the_client_stops_early():
    resp = _FakeResponse([_sse({"choices": [{"delta": {"content": "hi"}}]}, {"choices": [{"delta": {"content": "more"}}]})])
    stream = ga.stream_openai(_Account(_Client([resp])), MESSAGES, MODEL)

    first = await stream.__anext__()
    assert '"content": "hi"' in first
    assert resp.closed is False

    await stream.aclose()

    assert resp.closed is True
