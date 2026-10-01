import asyncio
import json
import logging

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

import danyapi.api.retry as retry_mod
from danyapi.api import anthropic as ant
from danyapi.api.openai import app


def _info(model="claude-sonnet-4-5", **kwargs):
    return ant.RequestInfo(model=model, max_tokens=100, **kwargs)


def _sse(*payloads):
    return "".join(f"data: {item}\n\n" for item in payloads)


def _frames(text):
    frames = []
    for block in text.split("\n\n"):
        if not block.strip():
            continue
        event = ""
        data = ""
        for line in block.splitlines():
            if line.startswith("event: "):
                event = line[len("event: ") :]
            elif line.startswith("data: "):
                data = line[len("data: ") :]
        if data:
            frames.append((event, json.loads(data)))
    return frames


def _named(frames, name):
    return [payload for event, payload in frames if event == name]


async def _agen_chunks(*texts):
    for text in texts:
        yield text


async def _run_stream(stream):
    return [line async for line in stream]


class _ChatResponse:
    def __init__(self, chunks):
        self.body_iterator = _agen_chunks(*chunks)


class _Unserialisable:
    def __repr__(self):
        return "<unserialisable>"


def _fake_request():
    class _Request:
        def __init__(self):
            self.headers: dict = {}
            self.url = type("_Url", (), {"path": "/v1/messages"})()

    return _Request()


def _dispatcher(handler):
    async def dispatcher(model, request):
        return handler

    return dispatcher


@pytest.fixture(autouse=True)
def _isolated_state():
    saved = dict(app.state._state)
    app.state.pool = None
    app.state.qwen_pool = None
    app.state.gigachat_pool = None
    app.state.deepseek_models = []
    original = retry_mod.RETRY_BACKOFF_SEC
    retry_mod.RETRY_BACKOFF_SEC = 0.0
    yield
    retry_mod.RETRY_BACKOFF_SEC = original
    app.state._state.clear()
    app.state._state.update(saved)


def test_as_text_renders_scalars_objects_and_nested_scalars():
    assert ant._as_text(None) == ""
    assert ant._as_text(5) == "5"
    assert ant._as_text(True) == "True"
    assert ant._as_text(1.5) == "1.5"
    assert ant._as_text([5, "a", 2.5]) == "5a2.5"
    assert ant._as_text(_Unserialisable()) == "<unserialisable>"


def test_as_text_prefers_a_text_key_and_falls_back_to_json():
    assert ant._as_text({"text": "hi"}) == "hi"
    assert ant._as_text({"a": 1, "b": [1, 2]}) == '{"a": 1, "b": [1, 2]}'


def test_as_text_falls_back_to_repr_when_the_value_cannot_be_serialised():
    value = {"weird": {1, 2}}
    assert ant._as_text(value) == str(value)


def test_as_text_returns_empty_beyond_the_depth_bound():
    at_bound: object = "leaf"
    for _ in range(ant.MAX_TEXT_DEPTH):
        at_bound = [{"content": at_bound}]
    assert ant._as_text(at_bound) == "leaf"
    past_bound: object = "leaf"
    for _ in range(ant.MAX_TEXT_DEPTH + 1):
        past_bound = [{"content": past_bound}]
    assert ant._as_text(past_bound) == ""


def test_image_part_requires_usable_base64_data():
    with pytest.raises(ant.AnthropicInputError) as excinfo:
        ant._image_part({"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": 7}})
    assert str(excinfo.value) == "image source requires base64 data"
    assert ant._image_part({"type": "image", "source": {"type": "base64", "data": "QQ=="}}) == {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,QQ=="},
    }


def test_image_part_accepts_a_url_source_and_a_nested_image_url():
    assert ant._image_part({"type": "image", "source": {"type": "url", "url": "https://x/y.png"}}) == {
        "type": "image_url",
        "image_url": {"url": "https://x/y.png"},
    }
    assert ant._image_part({"type": "image", "image_url": {"url": "https://z/w.png"}}) == {"type": "image_url", "image_url": {"url": "https://z/w.png"}}


def test_image_part_without_any_usable_source_raises():
    with pytest.raises(ant.AnthropicInputError) as excinfo:
        ant._image_part({"type": "image", "source": {"type": "url"}})
    assert str(excinfo.value) == "image block requires a source with base64 data or a url"
    with pytest.raises(ant.AnthropicInputError):
        ant._image_part({"type": "image", "image_url": {}})


def test_tool_use_block_serialises_dict_list_and_string_inputs():
    assert ant._tool_use_block({"id": "c1", "name": "f", "input": {"a": 1}})["function"] == {"name": "f", "arguments": '{"a": 1}'}
    assert ant._tool_use_block({"id": "c1", "name": "f", "input": [1, 2]})["function"]["arguments"] == "[1, 2]"
    assert ant._tool_use_block({"id": "c1", "name": "f", "input": '{"raw":true}'})["function"]["arguments"] == '{"raw":true}'
    assert ant._tool_use_block({"id": "c1", "name": "f"})["function"]["arguments"] == "{}"


def test_tool_use_block_falls_back_when_the_input_cannot_be_serialised():
    block = ant._tool_use_block({"id": "c1", "name": "f", "input": {"bad": {1, 2}}})
    assert block["function"]["arguments"] == "{}"


def test_tool_use_block_generates_an_id_and_a_name_when_absent():
    block = ant._tool_use_block({})
    assert block["id"].startswith("call_")
    assert len(block["id"]) == len("call_") + 12
    assert block["function"] == {"name": "", "arguments": "{}"}


def test_tool_result_block_marks_errors_and_uses_a_synthesised_call_id():
    assert ant._tool_result_block({"tool_use_id": "c1", "content": "boom", "is_error": True}) == {
        "role": "tool",
        "tool_call_id": "c1",
        "content": "[tool_error] boom",
    }
    assert ant._tool_result_block({"is_error": True}) == {"role": "tool", "tool_call_id": "", "content": "[tool_error]"}
    assert ant._tool_result_block({"id": "c9", "content": "ok"}) == {"role": "tool", "tool_call_id": "c9", "content": "ok"}


def test_content_blocks_handles_none_and_scalar_content():
    assert ant._content_blocks(None) == ("", [], [])
    assert ant._content_blocks("plain") == ("plain", [], [])
    assert ant._content_blocks(7) == ("7", [], [])
    assert ant._content_blocks({"text": "hi"}) == ("hi", [], [])


def test_content_blocks_flattens_strings_and_scalars_inside_a_list():
    content, tool_calls, tool_results = ant._content_blocks(["a", 5, {"text": "b"}])
    assert content == "a5b"
    assert tool_calls == []
    assert tool_results == []


def test_content_blocks_keeps_an_unknown_block_that_carries_text():
    content, tool_calls, tool_results = ant._content_blocks([{"type": "custom_widget", "text": "payload"}])
    assert content == "payload"
    assert (tool_calls, tool_results) == ([], [])


def test_content_blocks_returns_no_text_when_only_tools_are_present():
    content, tool_calls, tool_results = ant._content_blocks(
        [
            {"type": "tool_use", "id": "c1", "name": "f", "input": {}},
            {"type": "tool_result", "tool_use_id": "c1", "content": "done"},
        ]
    )
    assert content == ""
    assert tool_calls[0]["id"] == "c1"
    assert tool_results[0]["content"] == "done"


def test_content_blocks_raises_on_an_unusable_block_type():
    with pytest.raises(ant.AnthropicInputError) as excinfo:
        ant._content_blocks([{"type": "video", "source": {}}])
    assert str(excinfo.value) == "unsupported content block type: 'video'"


def test_content_blocks_skips_ignored_block_types():
    for block_type in sorted(ant.IGNORED_BLOCK_TYPES):
        assert ant._content_blocks([{"type": block_type, "text": "dropped"}]) == ("", [], [])


def test_normalize_system_handles_blocks_and_scalars():
    assert ant.normalize_system("") is None
    assert ant.normalize_system(["a", {"text": "b"}, {"type": "text"}, {"type": "text", "text": 5}, {"type": "image"}]) == "ab5"
    assert ant.normalize_system(["a", {"type": "text"}]) == "a"
    assert ant.normalize_system(12) == "12"
    assert ant.normalize_system(0) == "0"


def test_normalize_messages_rejects_non_list_and_non_dict_entries():
    with pytest.raises(ant.AnthropicInputError) as excinfo:
        ant.normalize_messages({"role": "user"})
    assert str(excinfo.value) == "messages must be an array of message objects"
    with pytest.raises(ant.AnthropicInputError) as excinfo:
        ant.normalize_messages(["plain string"])
    assert str(excinfo.value) == "each message must be an object"
    assert ant.normalize_messages(None) == []


def test_normalize_messages_rejects_a_conversation_that_normalises_to_nothing():
    with pytest.raises(ant.AnthropicInputError) as excinfo:
        ant.normalize_messages([{"role": "user", "content": [{"type": "thinking", "thinking": "x"}]}])
    assert str(excinfo.value) == "messages must contain at least one message with content"


def test_normalize_messages_keeps_a_tool_result_only_conversation():
    normalized = ant.normalize_messages(
        [
            {"role": "user", "content": [{"type": "tool_use", "id": "c1", "name": "f", "input": {"a": 1}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "c1", "content": "42"}]},
        ]
    )
    assert normalized == [
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "f", "arguments": '{"a": 1}'}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "42"},
    ]


def test_convert_tools_returns_none_for_empty_and_non_list_input():
    assert ant.convert_tools("nope") is None
    assert ant.convert_tools([]) is None
    assert ant.convert_tools([{"function": {"name": "already"}}]) == [{"function": {"name": "already"}}]
    assert ant.convert_tools([{"name": "f"}]) == [{"type": "function", "function": {"name": "f"}}]
    assert ant.convert_tools([{"name": "f", "description": "d", "input_schema": {"type": "object"}, "extra": 1}]) == [
        {"type": "function", "function": {"name": "f", "description": "d", "parameters": {"type": "object"}}}
    ]


def test_convert_tool_choice_covers_every_branch():
    assert ant.convert_tool_choice(None) is None
    assert ant.convert_tool_choice(7) is None
    assert ant.convert_tool_choice("auto") == "auto"
    assert ant.convert_tool_choice("none") == "none"
    assert ant.convert_tool_choice("required") == "required"
    assert ant.convert_tool_choice("get_weather") == {"type": "function", "function": {"name": "get_weather"}}
    assert ant.convert_tool_choice({"type": "auto"}) == "auto"
    assert ant.convert_tool_choice({"type": "none"}) == "none"
    assert ant.convert_tool_choice({"type": "any"}) == "required"
    assert ant.convert_tool_choice({"type": "tool", "name": "f"}) == {"type": "function", "function": {"name": "f"}}
    assert ant.convert_tool_choice({"type": "tool", "name": ""}) is None
    assert ant.convert_tool_choice({"type": "tool"}) is None
    assert ant.convert_tool_choice({"type": "function", "function": {"name": "f"}}) == {"type": "function", "function": {"name": "f"}}
    assert ant.convert_tool_choice({"type": "function"}) is None
    assert ant.convert_tool_choice({"type": "function", "function": {"name": 5}}) is None
    assert ant.convert_tool_choice({"type": "wat"}) is None


def test_convert_stop_sequences_rejects_bad_shapes():
    assert ant.convert_stop_sequences(None) is None
    assert ant.convert_stop_sequences("END") == ["END"]
    with pytest.raises(ant.AnthropicInputError) as excinfo:
        ant.convert_stop_sequences(7)
    assert str(excinfo.value) == "stop_sequences must be an array of strings"
    with pytest.raises(ant.AnthropicInputError) as excinfo:
        ant.convert_stop_sequences(["ok", ""])
    assert str(excinfo.value) == "stop_sequences[1] must be a non-empty string"
    assert ant.convert_stop_sequences([]) is None


def test_as_max_tokens_defaults_and_bounds():
    assert ant.as_max_tokens(None) == ant.DEFAULT_MAX_TOKENS == 4096
    assert ant.as_max_tokens(1) == 1
    for bad in (0, -5, "8", 8.5, True):
        with pytest.raises(ant.AnthropicInputError):
            ant.as_max_tokens(bad)


def test_as_ratio_and_top_k_validation():
    assert ant._as_ratio(None, "temperature") is None
    assert ant._as_ratio(0, "temperature") == 0.0
    assert ant._as_ratio(1, "temperature") == 1.0
    for bad in ("0.5", True, 1.5, -0.1):
        with pytest.raises(ant.AnthropicInputError):
            ant._as_ratio(bad, "top_p")
    assert ant._reject_top_k(None) is None
    for bad in ("5", True, 5.5, 0, -3, 7):
        with pytest.raises(ant.AnthropicInputError):
            ant._reject_top_k(bad)


def test_build_chat_request_reports_the_message_of_a_bad_sampling_param():
    messages = [{"role": "user", "content": "x"}]
    with pytest.raises(ant.AnthropicInputError) as excinfo:
        ant.build_chat_request({"messages": messages, "temperature": 2.0}, "m")
    assert str(excinfo.value) == "temperature must be between 0 and 1"
    with pytest.raises(ant.AnthropicInputError) as excinfo:
        ant.build_chat_request({"messages": messages, "top_p": "x"}, "m")
    assert str(excinfo.value) == "top_p must be a number"
    with pytest.raises(ant.AnthropicInputError) as excinfo:
        ant.build_chat_request({"messages": messages, "top_k": 5.5}, "m")
    assert str(excinfo.value) == "top_k is not supported by the upstream providers, remove it from the request"


def test_build_chat_request_passes_a_session_id_and_omits_a_blank_user_id():
    payload = ant.build_chat_request({"messages": [{"role": "user", "content": "x"}], "metadata": {"user_id": ""}}, "m", "sess-1")
    assert payload["session_id"] == "sess-1"
    assert "user" not in payload
    assert "session_id" not in ant.build_chat_request({"messages": [{"role": "user", "content": "x"}], "metadata": "not a dict"}, "m", None)


def test_decode_arguments_accepts_structures_and_rejects_junk():
    assert ant._decode_arguments({"a": 1}) == {"a": 1}
    assert ant._decode_arguments([1]) == [1]
    assert ant._decode_arguments(7) == {}
    assert ant._decode_arguments("   ") == {}
    assert ant._decode_arguments('{"a": 1}') == {"a": 1}
    assert ant._decode_arguments("{not json") == {}


def test_build_content_skips_broken_tool_calls_and_uses_the_call_as_the_function():
    blocks = ant.build_content({"message": {"content": "", "tool_calls": ["broken", {"id": "c1", "name": "f", "arguments": "{}"}]}})
    assert blocks == [{"type": "tool_use", "id": "c1", "name": "f", "input": {}}]
    generated = ant.build_content({"message": {"content": "", "tool_calls": [{"function": {"name": "f"}}]}})
    assert generated[0]["id"].startswith("call_")
    assert generated[0]["input"] == {}


def test_build_message_reports_a_stop_sequence_found_in_the_text():
    message = ant.build_message(
        _info(stop_sequences=["END"]),
        "msg_1",
        {"choices": [{"message": {"content": "before END after"}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 1, "completion_tokens": 2}},
    )
    assert message["stop_reason"] == "stop_sequence"
    assert message["stop_sequence"] == "END"
    assert message["usage"] == {"input_tokens": 1, "output_tokens": 2}


def test_build_message_error_branch_is_a_defensive_fallback():
    message = ant.build_message(_info(), "msg_1", {"error": {"message": "boom"}, "choices": [{"message": {"content": "partial"}, "finish_reason": "length"}]})
    assert message["stop_reason"] == "end_turn"
    assert message["content"] == [{"type": "text", "text": "partial"}]
    assert message["usage"] == {"input_tokens": 0, "output_tokens": 0}


def test_count_input_tokens_adds_the_system_prompt_once():
    payload = ant.build_chat_request({"system": "be brief", "messages": [{"role": "user", "content": "hi"}]}, "m")
    with_system = payload["messages"]
    assert with_system[0] == {"role": "system", "content": "be brief"}
    counted_once = ant.count_input_tokens(with_system, None)
    assert counted_once > ant.count_input_tokens(with_system[1:], None)
    assert counted_once < ant.count_input_tokens(with_system, "be brief")
    assert ant.count_input_tokens(with_system, None) == ant.count_input_tokens(with_system, None)


def test_decode_chunk_handles_str_bytes_and_garbage():
    assert ant._decode_chunk("plain") == "plain"
    assert ant._decode_chunk(b"bytes") == "bytes"
    assert ant._decode_chunk(bytearray(b"array")) == "array"
    assert ant._decode_chunk(memoryview(b"view")) == "view"
    assert ant._decode_chunk(7) == ""


def test_decode_chunk_replaces_invalid_utf8_and_warns(caplog):
    with caplog.at_level(logging.WARNING, logger="danyapi.api.anthropic"):
        assert ant._decode_chunk(b"ok\xff") == "ok\ufffd"
    assert "sse payload is not valid utf-8 at byte 2" in caplog.records[0].getMessage()


def test_iter_sse_payloads_ignores_empty_and_non_object_frames():
    assert list(ant.iter_sse_payloads("")) == []
    assert list(ant.iter_sse_payloads(7)) == []
    assert list(ant.iter_sse_payloads(": comment\n\n")) == []
    assert list(ant.iter_sse_payloads("data: [DONE]\n\n")) == [None]
    assert list(ant.iter_sse_payloads(b'data: {"a": 1}\n\n')) == [{"a": 1}]


def test_stream_state_start_is_idempotent():
    state = ant._StreamState(_info(), "msg_1")
    first = list(state.start())
    assert list(state.start()) == []
    assert first
    assert json.loads(first[0].split("data: ", 1)[1])["type"] == "message_start"


def test_stream_state_finish_is_idempotent():
    state = ant._StreamState(_info(), "msg_1")
    assert len(list(state.finish("end_turn", None))) == 2
    assert list(state.finish("max_tokens", "X")) == []


def test_stream_state_thinking_delta_closes_an_open_text_block():
    state = ant._StreamState(_info(), "msg_1")
    list(state.text_delta("hi"))
    events = list(state.thinking_delta("why"))
    types = [json.loads(event.split("data: ", 1)[1])["type"] for event in events]
    assert types == ["content_block_stop", "content_block_start", "content_block_delta"]
    assert state.text_index is None
    assert state.thinking_index == 1


def test_tool_slot_falls_back_to_the_position_for_unusable_indexes():
    assert ant._tool_slot({"index": 3}, 0) == 3
    assert ant._tool_slot({"index": -1}, 1) == 1
    assert ant._tool_slot({"index": "0"}, 2) == 2
    assert ant._tool_slot({"index": True}, 3) == 3
    assert ant._tool_slot({}, 4) == 4


def test_error_message_refuses_a_non_dict_error(caplog):
    with caplog.at_level(logging.WARNING, logger="danyapi.api.anthropic"):
        assert ant._error_message("boom") == ("api_error", "upstream stream error")
    assert "non-dict error: str" in caplog.records[0].getMessage()
    assert ant._error_message({}) == ("api_error", "upstream stream error")
    assert ant._error_message({"message": 7}) == ("api_error", "upstream stream error")
    assert ant._error_message({"message": "boom"}) == ("api_error", "boom")


def test_merge_usage_ignores_non_dict_and_non_numeric_values():
    usage = {"input_tokens": 0, "output_tokens": 0}
    ant._merge_usage(usage, None)
    assert usage == {"input_tokens": 0, "output_tokens": 0}
    ant._merge_usage(usage, {"prompt_tokens": "7", "completion_tokens": "9"})
    assert usage == {"input_tokens": 0, "output_tokens": 0}
    ant._merge_usage(usage, {"prompt_tokens": 3, "completion_tokens": -1})
    assert usage == {"input_tokens": 3, "output_tokens": 0}
    ant._merge_usage(usage, {"input_tokens": 4, "output_tokens": 6})
    assert usage == {"input_tokens": 4, "output_tokens": 6}


async def test_translate_stream_skips_malformed_choices_and_tool_calls():
    chunks = [
        _sse(json.dumps({"choices": "not a list"})),
        _sse(json.dumps({"choices": ["not a dict"]})),
        _sse(json.dumps({"choices": [{"index": 0, "delta": {"tool_calls": ["not a dict", {"name": "f", "arguments": "{}"}]}, "finish_reason": "tool_calls"}]})),
    ]
    stream = ant.translate_stream(_agen_chunks(*chunks), _info(), "msg_1")
    frames = _frames("".join(await _run_stream(stream)))
    assert [entry["content_block"]["name"] for entry in _named(frames, "content_block_start")] == ["f"]
    assert _named(frames, "message_delta")[0]["delta"] == {"stop_reason": "tool_use", "stop_sequence": None}


async def test_translate_stream_uses_the_call_itself_when_function_is_absent():
    chunk = _sse(
        json.dumps({"choices": [{"index": 0, "delta": {"tool_calls": [{"id": "c1", "name": "direct", "arguments": "{}"}]}, "finish_reason": "tool_calls"}]})
    )
    stream = ant.translate_stream(_agen_chunks(chunk), _info(), "msg_1")
    frames = _frames("".join(await _run_stream(stream)))
    assert _named(frames, "content_block_start")[0]["content_block"] == {"type": "tool_use", "id": "c1", "name": "direct", "input": {}}


async def test_translate_stream_logs_and_swallows_a_failing_upstream_close(caplog):
    class _Stream:
        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

        async def aclose(self):
            raise RuntimeError("close exploded")

    with caplog.at_level(logging.DEBUG, logger="danyapi.api.anthropic"):
        stream = ant.translate_stream(_Stream(), _info(), "msg_1")
        assert len(await _run_stream(stream)) == 3
    assert "upstream chat stream close failed: close exploded" in caplog.records[0].getMessage()


async def test_translate_stream_tolerates_a_stream_without_aclose():
    stream = ant.translate_stream(_agen_chunks(), _info(), "msg_1")
    frames = _frames("".join(await _run_stream(stream)))
    assert _named(frames, "message_start")[0]["message"]["id"] == "msg_1"
    assert _named(frames, "message_stop") == [{"type": "message_stop"}]


async def test_endpoint_error_branch_returns_the_anthropic_envelope_for_unknown_exceptions(monkeypatch):
    from danyapi.api.openai import anthropic_messages

    async def provider_call(chat_req):
        raise RuntimeError("provider exploded")

    monkeypatch.setattr("danyapi.api.openai._chat_dispatcher", _dispatcher(provider_call))
    response = await anthropic_messages({"model": "deepseek-v4.1-flash", "messages": [{"role": "user", "content": "hi"}]}, _fake_request())
    assert response.status_code == 500
    assert json.loads(response.body) == {"type": "error", "error": {"type": "api_error", "message": "internal server error"}}


async def test_endpoint_error_branch_covers_an_anthropic_input_error(monkeypatch):
    from danyapi.api.openai import anthropic_messages

    async def provider_call(chat_req):
        raise ant.AnthropicInputError("stop_sequences[0] must be a non-empty string")

    monkeypatch.setattr("danyapi.api.openai._chat_dispatcher", _dispatcher(provider_call))
    response = await anthropic_messages({"model": "deepseek-v4.1-flash", "messages": [{"role": "user", "content": "hi"}]}, _fake_request())
    assert response.status_code == 400
    assert json.loads(response.body) == {"type": "error", "error": {"type": "invalid_request_error", "message": "stop_sequences[0] must be a non-empty string"}}


async def test_endpoint_rejects_a_non_object_body():
    from danyapi.api.openai import anthropic_messages

    response = await anthropic_messages(["not", "an", "object"], _fake_request())
    assert response.status_code == 400
    assert json.loads(response.body) == {"type": "error", "error": {"type": "invalid_request_error", "message": "request body must be a JSON object"}}


async def test_endpoint_rejects_a_blank_model():
    from danyapi.api.openai import anthropic_messages

    response = await anthropic_messages({"model": "  ", "messages": [{"role": "user", "content": "hi"}]}, _fake_request())
    assert response.status_code == 400
    assert json.loads(response.body)["error"]["message"] == "model is required"


async def test_endpoint_reports_an_unknown_model_as_not_found():
    from danyapi.api.openai import anthropic_messages

    response = await anthropic_messages({"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]}, _fake_request())
    assert response.status_code == 404
    assert json.loads(response.body)["error"] == {"type": "not_found_error", "message": "Unknown model: gpt-4"}


def test_http_error_rewrites_an_unusable_dict_detail_and_keeps_explicit_headers():
    from danyapi.api.openai import _anthropic_http_error

    response = _anthropic_http_error(HTTPException(429, {"error": {"code": "busy"}}))
    assert response.status_code == 429
    assert json.loads(response.body) == {"type": "error", "error": {"type": "rate_limit_error", "message": "request could not be completed"}}
    kept = _anthropic_http_error(HTTPException(503, {"error": {"message": "pool warming"}}, headers={"retry-after": "3"}))
    assert json.loads(kept.body)["error"] == {"type": "api_error", "message": "pool warming"}
    assert kept.headers["retry-after"] == "3"
    flat = _anthropic_http_error(HTTPException(400, {"message": "flat"}))
    assert json.loads(flat.body)["error"] == {"type": "invalid_request_error", "message": "request could not be completed"}
    plain = _anthropic_http_error(HTTPException(404, "Unknown model: x"))
    assert json.loads(plain.body)["error"] == {"type": "not_found_error", "message": "Unknown model: x"}
    server = _anthropic_http_error(HTTPException(500, "boom"))
    assert json.loads(server.body)["error"] == {"type": "api_error", "message": "boom"}


async def test_endpoint_reports_a_non_dict_upstream_result_as_a_bad_gateway(monkeypatch):
    from danyapi.api.openai import anthropic_messages

    async def provider_call(chat_req):
        return ["not", "a", "dict"]

    monkeypatch.setattr("danyapi.api.openai._chat_dispatcher", _dispatcher(provider_call))
    response = await anthropic_messages({"model": "deepseek-v4.1-flash", "messages": [{"role": "user", "content": "hi"}]}, _fake_request())
    assert response.status_code == 502
    assert json.loads(response.body) == {"type": "error", "error": {"type": "api_error", "message": "upstream returned an invalid response"}}


async def test_endpoint_upstream_error_dict_messages(monkeypatch):
    from danyapi.api.openai import anthropic_messages

    async def with_message(chat_req):
        return {"error": {"message": "upstream said no"}}

    async def without_message(chat_req):
        return {"error": {"message": 7}}

    monkeypatch.setattr("danyapi.api.openai._chat_dispatcher", _dispatcher(with_message))
    response = await anthropic_messages({"model": "deepseek-v4.1-flash", "messages": [{"role": "user", "content": "hi"}]}, _fake_request())
    assert response.status_code == 502
    assert json.loads(response.body) == {"type": "error", "error": {"type": "api_error", "message": "upstream said no"}}

    monkeypatch.setattr("danyapi.api.openai._chat_dispatcher", _dispatcher(without_message))
    other = await anthropic_messages({"model": "deepseek-v4.1-flash", "messages": [{"role": "user", "content": "hi"}]}, _fake_request())
    assert json.loads(other.body) == {"type": "error", "error": {"type": "api_error", "message": "upstream request failed"}}


async def test_endpoint_passes_a_pydantic_failure_through_the_anthropic_envelope(monkeypatch):
    from danyapi.api.openai import anthropic_messages

    def exploding_chat_request(**kwargs):
        raise ValueError("max_tokens must be a string")

    monkeypatch.setattr("danyapi.api.openai.ChatCompletionRequest", exploding_chat_request)
    response = await anthropic_messages({"model": "deepseek-v4.1-flash", "messages": [{"role": "user", "content": "hi"}]}, _fake_request())
    assert response.status_code == 400
    assert json.loads(response.body)["error"] == {"type": "invalid_request_error", "message": "invalid request body: max_tokens must be a string"}


async def test_endpoint_supplies_on_complete_to_the_stream_translator(monkeypatch):
    from danyapi.api.openai import anthropic_messages

    seen: list = []
    real_translate_stream = ant.translate_stream

    async def recording_translate_stream(chat_stream, info, message_id, on_complete=None):
        seen.append(on_complete)
        assert on_complete is not None

        async def spy(reason, usage):
            seen.append((reason, dict(usage)))
            return on_complete(reason, usage)

        async for line in real_translate_stream(chat_stream, info, message_id, spy):
            yield line

    async def provider_call(chat_req):
        assert chat_req.stream_options == {"include_usage": True}
        return _ChatResponse(
            [_sse(json.dumps({"choices": [{"index": 0, "delta": {"content": "hi"}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 3}}))]
        )

    monkeypatch.setattr(ant, "translate_stream", recording_translate_stream)
    monkeypatch.setattr("danyapi.api.openai._chat_dispatcher", _dispatcher(provider_call))
    response = await anthropic_messages({"model": "deepseek-v4.1-flash", "messages": [{"role": "user", "content": "hi"}], "stream": True}, _fake_request())
    assert response.media_type == "text/event-stream"
    assert response.headers["cache-control"] == "no-cache"
    assert response.headers["x-accel-buffering"] == "no"
    await _run_stream(response.body_iterator)
    assert callable(seen[0])
    assert seen[1:] == [("end_turn", {"input_tokens": 3, "output_tokens": 0})]


async def test_endpoint_count_tokens_passes_stop_sequences_into_request_info():
    from danyapi.api.openai import anthropic_messages

    body = {
        "model": "deepseek-v4.1-flash",
        "system": "you are a very long system prompt that must be counted exactly once and not twice",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
        "stop_sequences": ["END"],
    }
    captured: list = []

    async def provider_call(chat_req):
        captured.append(chat_req)
        return _ChatResponse(
            [_sse(json.dumps({"choices": [{"index": 0, "delta": {"content": "hi"}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 3}}))]
        )

    request = _fake_request()
    request.stop_sequences = None

    async def dispatcher(model, _request):
        async def call(chat_req):
            captured.append(chat_req)
            return await provider_call(chat_req)

        return call

    import danyapi.api.openai as openai_mod

    original = openai_mod._chat_dispatcher
    openai_mod._chat_dispatcher = dispatcher
    try:
        response = await anthropic_messages(body, request)
        frames = _frames("".join(await _run_stream(response.body_iterator)))
    finally:
        openai_mod._chat_dispatcher = original
    system = body["system"]
    reported = _named(frames, "message_start")[0]["message"]["usage"]["input_tokens"]
    without_double_count = ant.count_input_tokens([{"role": "system", "content": system}, {"role": "user", "content": "hi"}], None)
    assert reported == without_double_count
    assert reported < without_double_count + ant.count_input_tokens([{"role": "system", "content": system}], None)
    assert captured[0].stop == ["END"]
    assert (captured[0].messages[0].role, captured[0].messages[0].content) == ("system", system)
    assert [(message.role, message.content) for message in captured[0].messages] == [("system", system), ("user", "hi")]


def test_count_tokens_endpoint_requires_messages_and_a_known_model():
    client = TestClient(app)
    no_messages = client.post("/v1/messages/count_tokens", json={"model": "deepseek-v4.1-flash"})
    assert no_messages.status_code == 400
    assert no_messages.json() == {"type": "error", "error": {"type": "invalid_request_error", "message": "messages is required"}}
    bad_model = client.post("/v1/messages/count_tokens", json={"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]})
    assert bad_model.status_code == 404
    assert bad_model.json()["error"]["type"] == "not_found_error"
    no_model = client.post("/v1/messages/count_tokens", json={"messages": [{"role": "user", "content": "hi"}]})
    assert no_model.status_code == 400
    assert no_model.json()["error"]["message"] == "model is required"
    client.close()


def test_count_tokens_endpoint_reports_bad_messages_tools_and_body():
    from danyapi.api.openai import anthropic_count_tokens

    bad_messages = asyncio.run(anthropic_count_tokens({"model": "deepseek-v4.1-flash", "messages": [{"role": "system", "content": "x"}]}))
    assert bad_messages.status_code == 400
    bad_tools = asyncio.run(anthropic_count_tokens({"model": "deepseek-v4.1-flash", "messages": [{"role": "user", "content": "x"}], "tools": ["nope"]}))
    assert bad_tools.status_code == 400
    assert json.loads(bad_tools.body)["error"] == {"type": "invalid_request_error", "message": "tools[0] must be an object"}


def test_count_tokens_endpoint_sums_the_tool_schemas():
    from danyapi.api.openai import anthropic_count_tokens
    from danyapi.tokens import estimate_tokens

    body = {
        "model": "deepseek-v4.1-flash",
        "messages": [{"role": "user", "content": "weather in paris"}],
        "system": "be brief",
        "tools": [
            {"name": "get_weather", "description": "Get the weather in a city", "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}}}
        ],
    }
    response = asyncio.run(anthropic_count_tokens(body))
    expected = ant.count_input_tokens(ant.normalize_messages(body["messages"]), "be brief") + estimate_tokens(
        "get_weather\nGet the weather in a city\n" + json.dumps(body["tools"][0]["input_schema"], ensure_ascii=False, sort_keys=True)
    )
    assert response == {"input_tokens": expected}


def test_count_tokens_endpoint_accepts_nested_openai_tools():
    from danyapi.api.openai import _tool_token_text

    assert _tool_token_text("not a dict") == ""
    assert _tool_token_text({"name": "f"}) == ""
    assert _tool_token_text({"function": {"name": "f"}}) == "f"
    assert _tool_token_text({"function": {"name": "f", "description": "d", "parameters": [1, 2]}}) == "f\nd\n[1, 2]"
    assert _tool_token_text({"function": {"name": "f", "parameters": "not a schema"}}) == "f"


def test_count_tokens_endpoint_rejects_a_non_object_body():
    from danyapi.api.openai import anthropic_count_tokens

    response = asyncio.run(anthropic_count_tokens(["nope"]))
    assert response.status_code == 400
    assert json.loads(response.body)["error"]["message"] == "request body must be a JSON object"


def test_count_tokens_endpoint_error_envelopes(monkeypatch):
    from danyapi.api.openai import anthropic_count_tokens

    def http_error(body):
        raise HTTPException(409, "conflict")

    monkeypatch.setattr("danyapi.api.openai._anthropic_model", http_error)
    conflict = asyncio.run(anthropic_count_tokens({"model": "deepseek-v4.1-flash", "messages": [{"role": "user", "content": "x"}]}))
    assert conflict.status_code == 409
    assert json.loads(conflict.body)["error"]["type"] == "invalid_request_error"

    def input_error(body):
        raise ant.AnthropicInputError("bad model")

    monkeypatch.setattr("danyapi.api.openai._anthropic_model", input_error)
    invalid = asyncio.run(anthropic_count_tokens({"model": "deepseek-v4.1-flash", "messages": [{"role": "user", "content": "x"}]}))
    assert invalid.status_code == 400
    assert json.loads(invalid.body)["error"] == {"type": "invalid_request_error", "message": "bad model"}

    def explode(body):
        raise RuntimeError("count tokens exploded")

    monkeypatch.setattr("danyapi.api.openai._anthropic_count_tokens", explode)
    internal = asyncio.run(anthropic_count_tokens({"model": "deepseek-v4.1-flash", "messages": [{"role": "user", "content": "x"}]}))
    assert internal.status_code == 500
    assert json.loads(internal.body) == {"type": "error", "error": {"type": "api_error", "message": "internal server error"}}


def test_count_tokens_endpoint_ignores_an_unusable_stop_sequences_value():
    from danyapi.api.openai import anthropic_count_tokens

    response = asyncio.run(anthropic_count_tokens({"model": "deepseek-v4.1-flash", "messages": [{"role": "user", "content": "x"}], "system": 5}))
    assert response == {"input_tokens": ant.count_input_tokens([{"role": "user", "content": "x"}], "5")}
