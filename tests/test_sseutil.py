import logging

import pytest

from danyapi.deepseek.sse import SSEEvent, parse_sse
from danyapi.deepseek.stream import IncrementalSSE, MessageReconstructor
from danyapi.sseutil import MAX_BUFFER_BYTES, StreamStopFilter, _set_path


def _ready() -> SSEEvent:
    return SSEEvent(
        None,
        {
            "o": "SET",
            "p": "response",
            "v": {
                "response": {
                    "message_id": "asst_1",
                    "role": "assistant",
                    "accumulated_token_usage": 0,
                    "status": "WIP",
                    "fragments": [
                        {"type": "THINK", "content": ""},
                        {"type": "RESPONSE", "content": ""},
                    ],
                }
            },
        },
    )


def _append(piece: str) -> SSEEvent:
    return SSEEvent(None, {"o": "APPEND", "p": "response/fragments/1/content", "v": piece})


def _usage(value: int) -> SSEEvent:
    return SSEEvent(None, {"o": "SET", "p": "response/accumulated_token_usage", "v": value})


def _primed() -> MessageReconstructor:
    rec = MessageReconstructor()
    rec.handle(_ready())
    rec.take_diffs()
    return rec


def test_appends_before_a_non_tail_set_are_never_re_emitted():
    rec = _primed()
    emitted = ""
    for event in (_append("He"), _append("llo"), _append(" world"), _usage(12)):
        rec.handle(event)
        c_diff, r_diff = rec.take_diffs()
        assert r_diff == ""
        emitted += c_diff
    assert emitted == "Hello world"
    assert rec.content == "Hello world"
    assert rec.take_diffs() == ("", "")


def test_a_non_tail_set_between_appends_emits_only_the_new_text():
    rec = _primed()
    rec.handle(_append("He"))
    rec.handle(_append("llo"))
    assert rec.take_diffs() == ("Hello", "")
    rec.handle(_usage(4))
    assert rec.take_diffs() == ("", "")
    rec.handle(_append(" world"))
    assert rec.take_diffs() == (" world", "")
    assert rec.content == "Hello world"


def test_reading_content_does_not_mutate_the_diff_state():
    rec = _primed()
    rec.handle(_append("Hello"))
    assert rec.content == "Hello"
    assert rec.content == "Hello"
    assert rec.take_diffs() == ("Hello", "")
    assert rec.content == "Hello"
    assert rec.take_diffs() == ("", "")


def test_reasoning_and_content_are_folded_independently():
    rec = _primed()
    rec.handle(SSEEvent(None, {"o": "APPEND", "p": "response/fragments/0/content", "v": "think "}))
    assert rec.take_diffs() == ("", "think ")
    rec.handle(_usage(7))
    assert rec.take_diffs() == ("", "")
    rec.handle(_append("answer"))
    assert rec.take_diffs() == ("answer", "")
    assert rec.reasoning == "think "
    assert rec.content == "answer"


def test_parse_sse_splits_on_a_lone_carriage_return():
    events = parse_sse('data: {"a":1}\r\rdata: {"b":2}\r\r')
    assert [event.data for event in events] == [{"a": 1}, {"b": 2}]


def test_incremental_sse_handles_a_lone_carriage_return_stream():
    inc = IncrementalSSE()
    seen = []
    raw = b'data: {"a":1}\r\rdata: {"b":2}\r\r'
    mid = 10
    for chunk in (raw[:mid], raw[mid:]):
        for event in inc.feed(chunk):
            seen.append(event.data)
    assert seen == [{"a": 1}, {"b": 2}]


def test_incremental_sse_raises_when_the_buffer_cap_is_passed():
    inc = IncrementalSSE()
    chunk = b"data: " + b"a" * 65536
    with pytest.raises(ValueError) as excinfo:
        for _ in range((MAX_BUFFER_BYTES // len(chunk)) + 2):
            list(inc.feed(chunk))
    assert str(MAX_BUFFER_BYTES) in str(excinfo.value)
    assert inc._buffer == bytearray()
    assert inc._pos == 0


def test_incremental_sse_compacts_the_consumed_prefix():
    inc = IncrementalSSE()
    consumed = 0
    for index in range(200):
        chunk = f'data: {{"i":{index}}}\n\n'.encode()
        for _event in inc.feed(chunk):
            consumed += 1
        if inc._pos:
            assert inc._pos <= len(inc._buffer)
            assert inc._buffer[: inc._pos] == bytearray()
    assert consumed == 200
    assert len(inc._buffer) < 4096


def test_incremental_sse_logs_and_replaces_invalid_utf8(caplog):
    inc = IncrementalSSE()
    raw = b'data: {"a":"\xff\xfe"}\n\n'
    with caplog.at_level(logging.WARNING, logger="danyapi.sseutil"):
        events = list(inc.feed(raw))
    assert any("not valid utf-8" in record.getMessage() for record in caplog.records)
    assert len(events) == 1
    assert events[0].data == {"a": "\ufffd\ufffd"}


def test_incremental_sse_decodes_multibyte_characters_split_across_chunks():
    inc = IncrementalSSE()
    raw = 'data: {"a":"Привет"}\n\n'.encode()
    seen = []
    for index in range(len(raw)):
        for event in inc.feed(raw[index : index + 1]):
            seen.append(event.data)
    assert seen == [{"a": "Привет"}]


def test_stream_stop_filter_accepts_an_empty_marker_list():
    filt = StreamStopFilter([])
    assert filt.feed("hello") == ("hello", False)
    assert filt.flush() == ""


def test_set_path_only_creates_the_root_intermediate():
    target: dict = {}
    _set_path(target, ("fragments", "xx", "content"), "x")
    assert target == {"fragments": {}}
    target = {}
    _set_path(target, ("accumulated_token_usage",), 5)
    assert target == {"accumulated_token_usage": 5}
    target = {"fragments": [{"content": "a"}]}
    _set_path(target, ("fragments", "0", "content"), "b")
    assert target["fragments"][0]["content"] == "b"
