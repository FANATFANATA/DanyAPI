from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
from types import SimpleNamespace
from typing import Any, cast

import pytest
from websockets.exceptions import WebSocketException

from danyapi.alice import client as client_module
from danyapi.alice.client import (
    AUTH_REJECTED,
    CONNECT_DROPPED,
    CONNECT_FAILED,
    CONNECT_FATAL,
    CONTINUATION_DELAY,
    EMPTY_ANSWER,
    GOAWAY,
    HANDSHAKE_TIMEOUT,
    HARD_MAX_PROMPT,
    MAX_CONTINUATIONS,
    MAX_PENDING_FRAMES,
    MIN_PROMPT_TAIL,
    MODEL_NAMES,
    PING_TIMEOUT,
    TRIM_ELLIPSIS,
    UPSTREAM_TIMEOUT,
    AliceClient,
    AliceError,
    AliceStream,
    _Directive,
    _text_of,
    fold_messages,
    is_placeholder,
    looks_like_refusal,
    trim_prompt,
)

EMPTY_MARKER = client_module.EMPTY_MARKERS[0]
CAPITALISED_EMPTY_MARKER = client_module.EMPTY_MARKERS[1]
AUTH_MARKER = client_module.AUTH_FINISH_MARKERS[0]
DOTTED_PLACEHOLDER = next(text for text in sorted(client_module.PLACEHOLDER_TEXTS) if text.endswith("..."))
PLAIN_PLACEHOLDER = next(text for text in sorted(client_module.PLACEHOLDER_TEXTS) if not text.endswith("."))


class _Boom(BaseException):
    pass


class _FakeWS:
    def __init__(self) -> None:
        self.sent: list[dict] = []
        self.closed = False
        self.close_error: BaseException | None = None
        self.send_error: BaseException | None = None
        self.incoming: asyncio.Queue = asyncio.Queue()
        self.waiting = asyncio.Event()
        self.ended = asyncio.Event()

    async def send(self, message: str) -> None:
        if self.send_error is not None:
            raise self.send_error
        self.sent.append(json.loads(message))

    async def close(self) -> None:
        self.closed = True
        if self.close_error is not None:
            raise self.close_error

    def __aiter__(self) -> _FakeWS:
        return self

    async def __anext__(self) -> str:
        self.waiting.set()
        item = await self.incoming.get()
        if isinstance(item, BaseException):
            self.ended.set()
            raise item
        return str(item)

    def feed(self, *frames: Any) -> None:
        for frame in frames:
            if isinstance(frame, (str, BaseException)):
                self.incoming.put_nowait(frame)
            else:
                self.incoming.put_nowait(json.dumps(frame))


class _FakeConnect:
    def __init__(self, sockets: list[_FakeWS] | None = None, error: BaseException | None = None) -> None:
        self.sockets = list(sockets or [])
        self.error = error
        self.calls: list[tuple[str, dict]] = []

    async def __call__(self, url: str, **kwargs: Any) -> _FakeWS:
        self.calls.append((url, kwargs))
        if self.error is not None:
            raise self.error
        if self.sockets:
            return self.sockets.pop(0)
        return _FakeWS()


class _ScriptedClock:
    def __init__(self, values: list[float]) -> None:
        self.values = list(values)
        self.calls = 0

    def monotonic(self) -> float:
        self.calls += 1
        if len(self.values) > 1:
            return self.values.pop(0)
        if not self.values:
            raise AssertionError("the client clock was read more times than the test scripted")
        return self.values[0]

    def __getattr__(self, name: str) -> Any:
        return getattr(time, name)


def _install_connect(monkeypatch, sockets: list[_FakeWS] | None = None, error: BaseException | None = None) -> _FakeConnect:
    connector = _FakeConnect(sockets, error)
    monkeypatch.setattr(client_module, "websockets", SimpleNamespace(connect=connector))
    return connector


def _sync_response() -> dict:
    return {"directive": {"header": {"namespace": "System", "name": "SynchronizeStateResponse", "messageId": "s1"}, "payload": {}}}


def _text_output(text: str, version: str = "") -> dict:
    return {
        "directive": {
            "header": {"namespace": "Vins", "name": "TextOutput"},
            "payload": {"version": version, "response": {"directives": [{"name": "print_text", "payload": {"text": text}}]}},
        }
    }


def _silent_output() -> dict:
    update = {"directive": {"header": {"namespace": "Vins", "name": "TextOutput"}, "payload": {"response": {}}}}
    return update


def _system(name: str) -> dict:
    return {"directive": {"header": {"namespace": "System", "name": name}, "payload": {}}}


async def _connected(monkeypatch, *frames: Any) -> tuple[AliceClient, _FakeWS, _FakeConnect]:
    ws = _FakeWS()
    connector = _install_connect(monkeypatch, [ws])
    ws.feed(_sync_response(), *frames)
    client = AliceClient()
    await client._ensure_connected()
    return client, ws, connector


def _ready(timeout: float = 60.0) -> tuple[AliceClient, _FakeWS]:
    client = AliceClient(timeout=timeout)
    client._ws = _FakeWS()
    return client, client._ws


@pytest.fixture
def instant_continuation(monkeypatch) -> None:
    monkeypatch.setattr(client_module, "CONTINUATION_DELAY", 0.0)


def test_default_timeout_reproduces_the_previous_frame_handshake_and_request_budgets() -> None:
    client = AliceClient()
    assert client.timeout == 60.0
    assert client.frame_timeout == 30.0
    assert client.handshake_timeout == 20.0
    assert client.request_timeout == 60.0


def test_custom_timeout_scales_the_frame_and_handshake_budgets() -> None:
    client = AliceClient(timeout=12.0)
    assert client.frame_timeout == 6.0
    assert client.handshake_timeout == 4.0
    assert client.request_timeout == 12.0


def test_large_timeout_is_capped_by_the_ping_and_handshake_constants() -> None:
    client = AliceClient(timeout=600.0)
    assert PING_TIMEOUT == 30.0
    assert HANDSHAKE_TIMEOUT == 20.0
    assert client.frame_timeout == PING_TIMEOUT
    assert client.handshake_timeout == HANDSHAKE_TIMEOUT
    assert client.request_timeout == 600.0


def test_tiny_timeout_is_floored_at_one_second() -> None:
    client = AliceClient(timeout=0.25)
    assert client.frame_timeout == 1.0
    assert client.handshake_timeout == 1.0
    assert client.request_timeout == 1.0


def test_configured_timeout_is_coerced_to_float() -> None:
    client = AliceClient(timeout=cast(Any, "90"))
    assert client.timeout == 90.0
    assert client.request_timeout == 90.0


async def test_frame_queue_is_bounded_at_construction_and_after_a_reconnect(monkeypatch) -> None:
    client = AliceClient()
    assert MAX_PENDING_FRAMES == 64
    assert client._frames.maxsize == MAX_PENDING_FRAMES
    first = client._frames
    client, _ws, _connector = await _connected(monkeypatch)
    assert client._frames is not first
    assert client._frames.maxsize == MAX_PENDING_FRAMES


def test_directive_narrows_name_and_payload_to_concrete_types() -> None:
    directive = _Directive({"name": "print_text", "payload": {"text": "hi"}})
    assert directive.name == "print_text"
    assert directive.payload == {"text": "hi"}


def test_directive_rejects_a_non_mapping_frame() -> None:
    assert _Directive("print_text").name == ""
    assert _Directive("print_text").payload == {}
    assert _Directive(None).payload == {}


def test_directive_ignores_a_non_string_name_and_a_non_mapping_payload() -> None:
    directive = _Directive({"name": 5, "payload": "text"})
    assert directive.name == ""
    assert directive.payload == {}


def test_text_of_flattens_content_parts_and_drops_unusable_items() -> None:
    parts = [
        "plain",
        {"type": "text", "text": "one"},
        {"type": "input_text", "text": "two"},
        {"type": "image", "url": "ignored"},
        {"type": "text", "text": 7},
        9,
    ]
    assert _text_of(parts) == "plainonetwo"


def test_text_of_renders_a_bare_value_and_empties_none() -> None:
    assert _text_of(None) == ""
    assert _text_of(391) == "391"
    assert _text_of("already text") == "already text"


def test_fold_messages_reads_a_plain_dict_message() -> None:
    assert fold_messages([{"role": "assistant", "content": "hello"}]) == "Assistant: hello"


def test_fold_messages_defaults_a_missing_role_to_user() -> None:
    assert fold_messages([{"content": "hello"}]) == "User: hello"


def test_fold_messages_defaults_a_non_string_role_to_user() -> None:
    assert fold_messages([{"role": 5, "content": "hello"}]) == "User: hello"
    assert fold_messages([{"role": None, "content": "hello"}]) == "User: hello"


def test_fold_messages_falls_back_to_a_greeting() -> None:
    assert fold_messages(None) == "Hello"
    assert fold_messages([{"role": "user", "content": None}]) == "Hello"
    assert fold_messages([SimpleNamespace(content="hi")]) == "User: hi"


def test_fold_messages_folds_a_developer_role_into_system() -> None:
    assert fold_messages([{"role": "developer", "content": "rules"}]) == "System: rules"


def test_looks_like_refusal_normalises_whitespace_and_case_before_matching() -> None:
    assert looks_like_refusal(f"  {EMPTY_MARKER.upper()}   ") is True
    assert looks_like_refusal(f"\n{AUTH_MARKER.upper()}\t") is True
    assert looks_like_refusal("") is False


def test_extract_uses_a_dialog_update_when_no_directive_text_is_present() -> None:
    client = AliceClient()
    stream = AliceStream()
    client._extract(
        {"payload": {"response": {"chat_dialog_update": [{"add_message_request": {"messages": [{"content": {"plain_response_text": "from the update"}}]}}]}}},
        stream,
    )
    assert stream.content == "from the update"
    assert stream.done is True


def test_trim_prompt_keeps_head_marker_and_tail_within_the_cap(caplog) -> None:
    head = "SYSTEM: answer in one word. "
    tail = "USER: 17*23?"
    text = head + "m" * (HARD_MAX_PROMPT * 2) + tail
    tail_len = max(MIN_PROMPT_TAIL, HARD_MAX_PROMPT // 8)
    head_len = HARD_MAX_PROMPT - tail_len - len(TRIM_ELLIPSIS)
    with caplog.at_level(logging.WARNING, logger="danyapi.alice"):
        trimmed = trim_prompt(text)
    assert len(trimmed) == HARD_MAX_PROMPT
    assert trimmed == text[:head_len] + TRIM_ELLIPSIS + text[-tail_len:]
    assert trimmed.startswith(head)
    assert TRIM_ELLIPSIS in trimmed
    assert trimmed.endswith(tail)
    assert caplog.messages == [f"alice prompt trimmed from {len(text)} to {HARD_MAX_PROMPT} chars, dropped {len(text) - HARD_MAX_PROMPT} chars from the middle"]


def test_trim_prompt_leaves_a_short_prompt_untouched() -> None:
    assert trim_prompt("short prompt") == "short prompt"
    assert trim_prompt("") == ""
    assert trim_prompt("x" * HARD_MAX_PROMPT) == "x" * HARD_MAX_PROMPT


def test_looks_like_refusal_matches_a_canned_line() -> None:
    assert looks_like_refusal(f"{CAPITALISED_EMPTY_MARKER}.") is True
    assert looks_like_refusal(f"   {AUTH_MARKER.upper()}   ") is True


def test_looks_like_refusal_matches_a_long_answer_ending_the_marker_clause() -> None:
    answer = EMPTY_MARKER + ". " + "y" * 300
    assert len(answer) > 240
    assert looks_like_refusal(answer) is True


def test_looks_like_refusal_ignores_a_marker_glued_to_a_word() -> None:
    answer = EMPTY_MARKER + "x" + " and more " * 40
    assert len(answer) > 240
    assert looks_like_refusal(answer) is False


def test_looks_like_refusal_ignores_a_legitimate_answer_mentioning_it_late() -> None:
    answer = "Tell me a long story about the weather in Moscow today, please. " * 6 + EMPTY_MARKER + "."
    assert len(answer) > 240
    assert answer.find(EMPTY_MARKER) > 48
    assert looks_like_refusal(answer) is False


def test_looks_like_refusal_is_false_for_empty_and_unrelated_text() -> None:
    assert looks_like_refusal("") is False
    assert looks_like_refusal("   ") is False
    assert looks_like_refusal("391") is False


def test_is_placeholder_rejects_blank_and_dotted_text() -> None:
    assert is_placeholder("") is False
    assert is_placeholder("   ") is False
    assert is_placeholder("...") is False
    assert is_placeholder(DOTTED_PLACEHOLDER) is True
    assert is_placeholder(f" {PLAIN_PLACEHOLDER}. ") is True
    assert is_placeholder("391") is False


async def test_send_without_a_socket_raises_a_retryable_connect_failure() -> None:
    client = AliceClient()
    with pytest.raises(AliceError) as excinfo:
        await client._send({"event": {}})
    assert excinfo.value.code == CONNECT_FAILED
    assert excinfo.value.message == "socket is not connected"
    assert excinfo.value.retryable is True


@pytest.mark.parametrize("error", [WebSocketException("closed"), OSError("pipe broke"), RuntimeError("loop is closed")])
async def test_send_wraps_a_write_failure_as_a_retryable_drop(error) -> None:
    client, ws = _ready()
    ws.send_error = error
    with pytest.raises(AliceError) as excinfo:
        await client._send({"event": {}})
    assert excinfo.value.code == CONNECT_DROPPED
    assert excinfo.value.message == f"alice socket write failed: {error}"
    assert excinfo.value.retryable is True
    assert isinstance(excinfo.value.__cause__, type(error))


async def test_aclose_cancels_the_reader_and_closes_the_socket() -> None:
    client, ws = _ready()
    reader = asyncio.create_task(_blocked(ws))
    client._reader = reader
    await ws.waiting.wait()
    await client.aclose()
    assert client._closed is True
    assert client._ws is None
    assert client._reader is None
    assert reader.cancelled() is True
    assert ws.closed is True


async def test_aclose_swallows_a_failed_reader_and_a_failing_socket_close() -> None:
    client, ws = _ready()
    ws.close_error = OSError("already gone")

    async def _boom() -> None:
        raise RuntimeError("reader died")

    client._reader = asyncio.create_task(_boom())
    await client.aclose()
    assert client._ws is None
    assert client._reader is None
    assert ws.closed is True


async def test_aclose_is_a_no_op_once_closed() -> None:
    client = AliceClient()
    await client.aclose()
    assert client._closed is True
    await client.aclose()
    assert client._closed is True


async def _blocked(ws: _FakeWS) -> None:
    async for _frame in ws:
        pass


async def test_read_loop_drops_unparsable_and_shapeless_frames(caplog) -> None:
    client, ws = _ready()
    reader = asyncio.create_task(client._read_loop())
    ws.feed(
        "not json at all",
        "[1, 2, 3]",
        '{"no": "directive"}',
        {"directive": "not a mapping"},
        {"directive": {"header": "not a mapping"}},
        {"directive": {"header": {"name": 5, "namespace": 7}}},
        _text_output("kept", version="v7"),
    )
    while client._frames.qsize() < 3:
        await asyncio.sleep(0)
    drained = [client._frames.get_nowait() for _ in range(3)]
    assert drained == [
        {"header": "not a mapping"},
        {"header": {"name": 5, "namespace": 7}},
        _text_output("kept", version="v7")["directive"],
    ]
    reader.cancel()
    await asyncio.wait({reader})
    assert client._frames.qsize() == 0
    assert ws.sent == []
    assert caplog.records == []


async def test_read_loop_answers_a_ping_with_a_pong() -> None:
    client, ws = _ready()
    reader = asyncio.create_task(client._read_loop())
    ws.feed({"directive": {"header": {"namespace": "System", "name": "Ping", "messageId": "ping-1"}, "payload": {}}})
    while not ws.sent:
        await asyncio.sleep(0)
    header = ws.sent[0]["event"]["header"]
    assert header["namespace"] == "System"
    assert header["name"] == "Pong"
    assert header["refMessageId"] == "ping-1"
    assert len(header["messageId"]) == 36
    assert header["messageId"] != "ping-1"
    assert ws.sent[0]["event"]["payload"] == {}
    reader.cancel()
    await asyncio.wait({reader})


async def test_read_loop_sets_the_sync_event_on_a_state_response() -> None:
    client, ws = _ready()
    reader = asyncio.create_task(client._read_loop())
    assert client._sync.is_set() is False
    ws.feed(_sync_response())
    while not client._sync.is_set():
        await asyncio.sleep(0)
    assert client._sync.is_set() is True
    reader.cancel()
    await asyncio.wait({reader})


@pytest.mark.parametrize("error", [WebSocketException("closed"), OSError("pipe broke"), RuntimeError("loop is closed"), _Boom("odd")])
async def test_read_loop_signals_every_socket_failure_with_none(error, caplog) -> None:
    client, ws = _ready()
    reader = asyncio.create_task(client._read_loop())
    with caplog.at_level(logging.DEBUG, logger="danyapi.alice"):
        ws.feed(error)
        await asyncio.wait({reader})
    assert reader.cancelled() is False
    assert reader.exception() is None
    assert client._frames.get_nowait() is None
    assert caplog.messages == [f"alice socket reader stopped: {error!r}"]


async def test_read_loop_reraises_cancellation_without_signalling_frames(caplog) -> None:
    client, ws = _ready()
    reader = asyncio.create_task(client._read_loop())
    await ws.waiting.wait()
    with caplog.at_level(logging.DEBUG, logger="danyapi.alice"):
        reader.cancel()
        await asyncio.wait({reader})
    assert reader.cancelled() is True
    assert client._frames.qsize() == 0
    assert caplog.messages == []


@pytest.mark.parametrize("error", [OSError("dns is down"), WebSocketException("handshake refused"), asyncio.TimeoutError()])
async def test_ensure_connected_reports_a_failed_connect_as_retryable(monkeypatch, error) -> None:
    connector = _install_connect(monkeypatch, error=error)
    client = AliceClient()
    with pytest.raises(AliceError) as excinfo:
        await client._ensure_connected()
    assert excinfo.value.code == CONNECT_FAILED
    assert excinfo.value.message == f"could not reach alice: {error}"
    assert excinfo.value.retryable is True
    assert client._ws is None
    assert client._reader is None
    assert connector.calls[0][0] == client_module.WS_URL


async def test_ensure_connected_passes_the_derived_timeouts_to_the_transport(monkeypatch) -> None:
    _client, _ws, connector = await _connected(monkeypatch)
    assert connector.calls == [
        (
            client_module.WS_URL,
            {
                "origin": client_module.ORIGIN,
                "ping_interval": 30.0,
                "ping_timeout": 30.0,
                "close_timeout": 5.0,
                "open_timeout": 60.0,
            },
        )
    ]


async def test_ensure_connected_sends_the_synchronize_state_handshake(monkeypatch) -> None:
    client, ws, _connector = await _connected(monkeypatch)
    assert len(ws.sent) == 1
    header = ws.sent[0]["event"]["header"]
    payload = ws.sent[0]["event"]["payload"]
    assert header["namespace"] == "System"
    assert header["name"] == "SynchronizeState"
    assert header["seqNumber"] == 1
    assert payload["vins"] == {"application": {"app_id": "ru.yandex.web.desktop", "platform": "windows"}}
    assert payload["uuid"] == client._uuid
    assert len(payload["auth_token"]) == 36
    assert payload["auth_token"] != client._uuid
    assert client._sync.is_set() is True


async def test_ensure_connected_reuses_a_live_connection(monkeypatch) -> None:
    client, ws, connector = await _connected(monkeypatch)
    await client._ensure_connected()
    await client._ensure_connected()
    assert len(connector.calls) == 1
    assert ws.closed is False


async def test_ensure_connected_reconnects_once_the_reader_died(monkeypatch) -> None:
    first = _FakeWS()
    second = _FakeWS()
    connector = _install_connect(monkeypatch, [first, second])
    first.feed(_sync_response(), OSError("remote closed"))
    second.feed(_sync_response())
    client = AliceClient()
    await client._ensure_connected()
    dead_reader = client._reader
    assert dead_reader is not None
    await asyncio.wait({dead_reader})
    assert client._frames.get_nowait() == _sync_response()["directive"]
    assert client._frames.get_nowait() is None
    await client._ensure_connected()
    assert len(connector.calls) == 2
    assert first.closed is True
    assert client._ws is second
    assert client._frames.maxsize == MAX_PENDING_FRAMES
    assert client._frames is not first.incoming


async def test_ensure_connected_closes_both_halves_when_the_handshake_write_fails(monkeypatch) -> None:
    ws = _FakeWS()
    ws.send_error = OSError("write failed")
    _install_connect(monkeypatch, [ws])
    client = AliceClient()
    with pytest.raises(AliceError) as excinfo:
        await client._ensure_connected()
    assert excinfo.value.code == CONNECT_DROPPED
    assert excinfo.value.message == "alice handshake failed: alice socket write failed: write failed"
    assert excinfo.value.retryable is True
    assert client._ws is None
    assert client._reader is None
    assert ws.closed is True


async def test_ensure_connected_closes_both_halves_when_the_sync_response_never_arrives(monkeypatch) -> None:
    ws = _FakeWS()
    _install_connect(monkeypatch, [ws])
    client = AliceClient()
    client.handshake_timeout = 0.01
    with pytest.raises(AliceError) as excinfo:
        await client._ensure_connected()
    assert excinfo.value.code == CONNECT_FAILED
    assert excinfo.value.message == "alice handshake failed: "
    assert excinfo.value.retryable is True
    assert client._ws is None
    assert client._reader is None
    assert ws.closed is True


def test_extract_skips_malformed_chat_dialog_updates() -> None:
    client = AliceClient()
    stream = AliceStream()
    client._extract(
        {
            "payload": {
                "response": {
                    "chat_dialog_update": [
                        "not a mapping",
                        {"add_message_request": "not a mapping"},
                        {"add_message_request": {"messages": ["not a mapping"]}},
                        {"add_message_request": {"messages": [{"content": "not a mapping"}]}},
                    ]
                }
            }
        },
        stream,
    )
    assert stream.content == ""
    assert stream.placeholder == ""
    assert stream.done is False
    assert stream.cards == []


def test_extract_prefers_the_directive_text_and_records_the_card() -> None:
    client = AliceClient()
    stream = AliceStream()
    client._extract(
        {
            "payload": {
                "version": "tags/releases/alice/stable-1@1",
                "response": {
                    "directives": [{"payload": {"text": "from directive"}}, {"payload": {"text": "final"}}],
                    "card": {"type": "simple_text", "text": "from card"},
                    "chat_dialog_update": [{"add_message_request": {"messages": [{"content": {"plain_response_text": "from update"}}]}}],
                },
            }
        },
        stream,
    )
    assert stream.content == "final"
    assert stream.version == "tags/releases/alice/stable-1@1"
    assert stream.cards == [{"type": "simple_text", "text": "from card"}]
    assert stream.done is True


def test_extract_falls_back_to_a_card_without_a_type() -> None:
    client = AliceClient()
    stream = AliceStream()
    client._extract({"payload": {"response": {"card": {"text": "from card"}}}}, stream)
    assert stream.content == "from card"
    assert stream.cards == []
    assert stream.done is True


def test_extract_keeps_the_first_placeholder_and_never_marks_it_done() -> None:
    client = AliceClient()
    stream = AliceStream()
    client._extract({"payload": {"response": {"directives": [{"payload": {"text": DOTTED_PLACEHOLDER}}]}}}, stream)
    client._extract({"payload": {"response": {"directives": [{"payload": {"text": PLAIN_PLACEHOLDER}}]}}}, stream)
    assert stream.placeholder == DOTTED_PLACEHOLDER
    assert stream.content == ""
    assert stream.done is False


def test_extract_ignores_a_directive_without_text() -> None:
    client = AliceClient()
    stream = AliceStream()
    client._extract({"payload": {"response": {"directives": [{}, "raw", {"payload": {"text": ""}}]}}}, stream)
    assert stream.content == ""
    assert stream.done is False


async def test_pump_returns_the_first_answer(monkeypatch) -> None:
    client, ws = _ready()
    client._frames.put_nowait(_text_output("391", version="v3")["directive"])
    stream = AliceStream()
    await client._pump({"event": {}}, stream)
    assert stream.content == "391"
    assert stream.version == "v3"
    assert stream.done is True
    assert len(ws.sent) == 1


async def test_pump_sends_a_continuation_after_the_continuation_delay(monkeypatch) -> None:
    client, ws = _ready()
    delays: list[float] = []
    real_sleep = asyncio.sleep

    async def _sleep(delay: float) -> None:
        delays.append(delay)
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", _sleep)
    client._frames.put_nowait(_system("Progress"))
    client._frames.put_nowait(_text_output("391")["directive"])
    stream = AliceStream()
    await client._pump({"event": {}}, stream)
    assert stream.content == "391"
    assert delays == [CONTINUATION_DELAY]
    assert CONTINUATION_DELAY > 0
    assert len(ws.sent) == 2
    continuation = ws.sent[1]["event"]["payload"]["request"]["event"]
    assert continuation["type"] == "server_action"
    assert continuation["name"] == "@@mm_stack_engine_get_next"
    assert continuation["payload"]["@scenario_name"] == "Dialogovo"
    assert ws.sent[1]["event"]["payload"]["header"]["prev_req_id"] is None


async def test_pump_clamps_every_frame_wait_to_the_remaining_request_budget(monkeypatch) -> None:
    client, _ws = _ready(timeout=60.0)
    client.frame_timeout = 30.0
    monkeypatch.setattr(client_module, "time", _ScriptedClock([0.0, 5.0, 58.0]))
    monkeypatch.setattr(client_module, "CONTINUATION_DELAY", 0.0)
    timeouts: list[float] = []
    real_wait_for = asyncio.wait_for

    async def _wait_for(awaitable, timeout):
        timeouts.append(timeout)
        return await real_wait_for(awaitable, timeout)

    monkeypatch.setattr(asyncio, "wait_for", _wait_for)
    client._frames.put_nowait(_system("Progress"))
    client._frames.put_nowait(_text_output("391")["directive"])
    stream = AliceStream()
    await client._pump({"event": {}}, stream)
    assert stream.content == "391"
    assert timeouts == [30.0, 2.0]


async def test_pump_stops_at_the_overall_request_deadline(monkeypatch, caplog) -> None:
    client, ws = _ready()
    monkeypatch.setattr(client_module, "time", _ScriptedClock([0.0, 99.0]))
    with caplog.at_level(logging.WARNING, logger="danyapi.alice"):
        with pytest.raises(AliceError) as excinfo:
            await client._pump({"event": {}}, AliceStream())
    assert excinfo.value.code == UPSTREAM_TIMEOUT
    assert excinfo.value.message == "alice did not answer within the request budget"
    assert excinfo.value.retryable is True
    assert caplog.messages == ["alice continuation deadline of 60s reached after 0 steps"]
    assert len(ws.sent) == 1


async def test_pump_reports_a_frame_timeout(monkeypatch) -> None:
    client, _ws = _ready()
    client.frame_timeout = 0.01
    with pytest.raises(AliceError) as excinfo:
        await client._pump({"event": {}}, AliceStream())
    assert excinfo.value.code == UPSTREAM_TIMEOUT
    assert excinfo.value.message == "alice did not answer in time"
    assert excinfo.value.retryable is True


async def test_pump_reports_a_dropped_connection() -> None:
    client, _ws = _ready()
    client._frames.put_nowait(None)
    with pytest.raises(AliceError) as excinfo:
        await client._pump({"event": {}}, AliceStream())
    assert excinfo.value.code == CONNECT_DROPPED
    assert excinfo.value.message == "alice closed the connection"
    assert excinfo.value.retryable is True


@pytest.mark.parametrize("name,code", [("GoAway", GOAWAY), ("InvalidAuth", AUTH_REJECTED)])
async def test_pump_maps_system_disconnects_to_retryable_errors(name, code) -> None:
    client, _ws = _ready()
    client._frames.put_nowait(_system(name)["directive"])
    with pytest.raises(AliceError) as excinfo:
        await client._pump({"event": {}}, AliceStream())
    assert excinfo.value.code == code
    assert excinfo.value.message == f"alice sent {name}"
    assert excinfo.value.retryable is True


async def test_pump_maps_an_event_exception_to_a_fatal_error() -> None:
    client, _ws = _ready()
    client._frames.put_nowait({"header": {"namespace": "System", "name": "EventException"}, "payload": {"error": {"message": "internal alice failure"}}})
    with pytest.raises(AliceError) as excinfo:
        await client._pump({"event": {}}, AliceStream())
    assert excinfo.value.code == CONNECT_FATAL
    assert excinfo.value.message == "alice error: internal alice failure"
    assert excinfo.value.retryable is False


async def test_pump_reports_an_event_exception_without_a_message_as_unknown() -> None:
    client, _ws = _ready()
    client._frames.put_nowait({"header": {"namespace": "System", "name": "EventException"}, "payload": "not a mapping"})
    with pytest.raises(AliceError) as excinfo:
        await client._pump({"event": {}}, AliceStream())
    assert excinfo.value.code == CONNECT_FATAL
    assert excinfo.value.message == "alice error: unknown"


async def test_pump_gives_up_after_the_continuation_budget(instant_continuation) -> None:
    client, ws = _ready()
    for _ in range(MAX_CONTINUATIONS):
        client._frames.put_nowait(_silent_output()["directive"])
    with pytest.raises(AliceError) as excinfo:
        await client._pump({"event": {}}, AliceStream())
    assert MAX_CONTINUATIONS == 24
    assert excinfo.value.code == EMPTY_ANSWER
    assert excinfo.value.message == "alice produced no answer after continuations"
    assert excinfo.value.retryable is True
    assert len(ws.sent) == MAX_CONTINUATIONS + 1


async def test_pump_names_the_placeholder_that_never_became_an_answer(instant_continuation) -> None:
    client, _ws = _ready()
    client._frames.put_nowait(_text_output(DOTTED_PLACEHOLDER)["directive"])
    for _ in range(MAX_CONTINUATIONS - 1):
        client._frames.put_nowait(_silent_output()["directive"])
    with pytest.raises(AliceError) as excinfo:
        await client._pump({"event": {}}, AliceStream())
    assert excinfo.value.code == EMPTY_ANSWER
    assert excinfo.value.message == f"alice produced no answer after continuations: {DOTTED_PLACEHOLDER}"


async def test_ask_returns_the_streamed_answer(monkeypatch) -> None:
    client, ws, _connector = await _connected(monkeypatch, _text_output("391", version="v4"))
    stream = await client.ask("17*23?")
    assert stream.content == "391"
    assert stream.version == "v4"
    assert stream.done is True
    assert stream.placeholder == ""
    assert client._ws is ws


async def test_ask_trims_the_prompt_before_sending_it(monkeypatch) -> None:
    client, ws, _connector = await _connected(monkeypatch, _text_output("391"))
    huge = "q" * (HARD_MAX_PROMPT + 10) + "TAIL"
    await client.ask(huge)
    sent_prompt = ws.sent[1]["event"]["payload"]["request"]["event"]["text"]
    assert len(sent_prompt) == HARD_MAX_PROMPT
    assert sent_prompt.endswith("TAIL")


async def test_ask_rejects_an_empty_prompt() -> None:
    client = AliceClient()
    with pytest.raises(AliceError) as excinfo:
        await client.ask("")
    assert excinfo.value.code == EMPTY_ANSWER
    assert excinfo.value.message == "prompt is empty"
    assert excinfo.value.retryable is False
    assert client._ws is None


async def test_ask_rejects_a_refusal_as_a_retryable_empty_answer(monkeypatch) -> None:
    client, _ws, _connector = await _connected(monkeypatch, _text_output(EMPTY_MARKER))
    with pytest.raises(AliceError) as excinfo:
        await client.ask("hi")
    assert excinfo.value.code == EMPTY_ANSWER
    assert excinfo.value.message == f"alice declined to answer: {EMPTY_MARKER}"
    assert excinfo.value.retryable is True


async def test_check_auth_is_true_for_a_live_connection(monkeypatch) -> None:
    client, _ws, _connector = await _connected(monkeypatch)
    assert await client.check_auth() is True
    assert client._ws is not None
    assert client._reader is not None


async def test_check_auth_is_false_when_the_connect_fails(monkeypatch) -> None:
    _install_connect(monkeypatch, error=OSError("dns is down"))
    assert await AliceClient().check_auth() is False


async def test_check_auth_is_false_when_the_reader_task_died(monkeypatch) -> None:
    client, ws = _ready()

    async def _already_connected() -> None:
        return None

    async def _dead() -> None:
        return None

    monkeypatch.setattr(client, "_ensure_connected", _already_connected)
    dead_reader = asyncio.create_task(_dead())
    client._reader = dead_reader
    await asyncio.wait({dead_reader})
    assert ws is client._ws
    assert await client.check_auth() is False


async def test_check_auth_is_false_without_a_reader(monkeypatch) -> None:
    client = AliceClient()

    async def _already_connected() -> None:
        return None

    monkeypatch.setattr(client, "_ensure_connected", _already_connected)
    assert await client.check_auth() is False


async def test_fetch_models_lists_every_alias() -> None:
    models = await AliceClient().fetch_models()
    assert models == [
        {"id": "alice", "name": "Alice AI (Yandex)", "owned_by": "alice", "model_type": "chat"},
        {"id": "alice-ai", "name": "Alice AI (Yandex)", "owned_by": "alice", "model_type": "chat"},
        {"id": "yagpt", "name": "YaGPT (Yandex)", "owned_by": "alice", "model_type": "chat"},
    ]
    assert {model["id"] for model in models} == set(MODEL_NAMES)


class _FakeResponse:
    def __init__(self, text: str, status_code: int = 200) -> None:
        self.text = text
        self.status_code = status_code


class _FakeHttpClient:
    def __init__(self, response: _FakeResponse | None = None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.init_kwargs: dict = {}
        self.requests: list[tuple[str, Any]] = []

    def __call__(self, **kwargs: Any) -> _FakeHttpClient:
        self.init_kwargs = kwargs
        return self

    async def __aenter__(self) -> _FakeHttpClient:
        return self

    async def __aexit__(self, *exc: Any) -> bool:
        return False

    async def get(self, url: str, headers: dict | None = None) -> _FakeResponse:
        self.requests.append((url, headers))
        if self.error is not None:
            raise self.error
        if self.response is None:
            raise AssertionError("the fake http client was asked for a response it does not have")
        return self.response


async def test_resolve_version_keeps_the_configured_version_without_httpx(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "httpx", None)
    client = AliceClient()
    assert await client.resolve_version() == client_module.APP_VERSION
    assert client.app_version == client_module.APP_VERSION


async def test_resolve_version_adopts_the_production_version(monkeypatch) -> None:
    import httpx

    http = _FakeHttpClient(_FakeResponse('...,"production", version:"4.2.1" ...'))
    monkeypatch.setattr(httpx, "AsyncClient", http)
    client = AliceClient()
    assert await client.resolve_version() == "4.2.1"
    assert client.app_version == "4.2.1"
    assert http.init_kwargs == {"timeout": 10.0}
    assert http.requests == [("https://ya.ru/alisa_davay_pridumaem", {"User-Agent": client_module.OS_VERSION})]


async def test_resolve_version_falls_back_to_a_json_version_field(monkeypatch) -> None:
    import httpx

    http = _FakeHttpClient(_FakeResponse('{"version": "1.2.3-beta", "other": 1}'))
    monkeypatch.setattr(httpx, "AsyncClient", http)
    client = AliceClient()
    assert await client.resolve_version() == "1.2.3-beta"


async def test_resolve_version_keeps_the_configured_version_on_a_non_200(monkeypatch) -> None:
    import httpx

    http = _FakeHttpClient(_FakeResponse('"production", version:"9.9.9"', status_code=503))
    monkeypatch.setattr(httpx, "AsyncClient", http)
    client = AliceClient()
    assert await client.resolve_version() == client_module.APP_VERSION
    assert len(http.requests) == 1


async def test_resolve_version_keeps_the_configured_version_on_a_transport_error(monkeypatch) -> None:
    import httpx

    http = _FakeHttpClient(error=httpx.ConnectError("no route"))
    monkeypatch.setattr(httpx, "AsyncClient", http)
    client = AliceClient()
    assert await client.resolve_version() == client_module.APP_VERSION


async def test_resolve_version_keeps_the_configured_version_when_nothing_matches(monkeypatch) -> None:
    import httpx

    http = _FakeHttpClient(_FakeResponse("nothing useful here"))
    monkeypatch.setattr(httpx, "AsyncClient", http)
    client = AliceClient()
    assert await client.resolve_version() == client_module.APP_VERSION
