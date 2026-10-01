import asyncio
import json
import logging

import pytest
from fastapi import HTTPException

from danyapi.api import responses as resp


async def _agen(items):
    for item in items:
        yield item


async def _collect(agen):
    out = []
    async for item in agen:
        out.append(item)
    return out


def test_as_text_none():
    assert resp._as_text(None) == ""


def test_as_text_scalars():
    assert resp._as_text(5) == "5"
    assert resp._as_text(1.5) == "1.5"
    assert resp._as_text(True) == "True"


def test_as_text_list_variants():
    assert resp._as_text(["a", {"text": "b"}, {"type": "output_text", "content": "c"}, 7, {"x": 1}]) == "abc7"


def test_as_text_dict_and_fallback():
    assert resp._as_text({"x": 1}) == json.dumps({"x": 1}, ensure_ascii=False)
    assert resp._as_text({"text": "hi"}) == "hi"
    assert resp._as_text(b"x") == "b'x'"


def test_normalize_content_none_and_scalar():
    assert resp._normalize_content(None) == ""
    assert resp._normalize_content(5) == "5"


def test_normalize_content_string_items():
    assert resp._normalize_content(["a", "b"]) == "ab"


def test_normalize_content_non_dict_item_raises():
    with pytest.raises(resp.ResponsesInputError):
        resp._normalize_content([5])


def test_normalize_content_text_part():
    assert resp._normalize_content([{"type": "input_text", "text": "x"}]) == "x"


def test_normalize_content_image_variants():
    assert resp._normalize_content([{"type": "input_image", "image_url": {"url": "data:x"}}]) == [{"type": "image_url", "image_url": "data:x"}]
    with pytest.raises(resp.ResponsesInputError):
        resp._normalize_content([{"type": "input_image", "image_url": ""}])


def test_normalize_content_refusal_file_and_fallback():
    assert resp._normalize_content([{"type": "refusal", "refusal": "no"}]) == "no"
    assert resp._normalize_content([{"type": "input_file"}]) == ""
    assert resp._normalize_content([{"text": "hi"}]) == "hi"


def test_normalize_item_not_dict():
    with pytest.raises(resp.ResponsesInputError):
        resp._normalize_item(5)


def test_normalize_item_function_call_dict_arguments():
    items = resp._normalize_item({"type": "function_call", "name": "f", "arguments": {"a": 1}})
    assert items[0]["tool_calls"][0]["function"]["arguments"] == json.dumps({"a": 1})


def test_normalize_item_function_call_output():
    items = resp._normalize_item({"type": "function_call_output", "call_id": "c", "output": "42"})
    assert items == [{"role": "tool", "tool_call_id": "c", "content": "42"}]
    by_id = resp._normalize_item({"type": "function_call_output", "id": "c", "output": 42})
    assert by_id == [{"role": "tool", "tool_call_id": "c", "content": "42"}]


def test_normalize_item_function_call_output_without_call_id():
    broken = (
        {"type": "function_call_output", "output": "42"},
        {"type": "function_call_output", "id": "", "output": "42"},
        {"type": "function_call_output", "call_id": 7},
    )
    for item in broken:
        with pytest.raises(resp.ResponsesInputError):
            resp._normalize_item(item)


def test_normalize_item_reasoning():
    with pytest.raises(resp.ResponsesInputError):
        resp._normalize_item({"type": "reasoning"})


def test_normalize_item_unreplayable_provider_items():
    for item_type in ("reasoning", "item_reference", "web_search_call", "file_search_call", "code_interpreter_call"):
        with pytest.raises(resp.ResponsesInputError) as excinfo:
            resp._normalize_item({"type": item_type})
        assert item_type in str(excinfo.value)


def test_normalize_item_message_default_role():
    items = resp._normalize_item({"type": "message", "content": "hi"})
    assert items[0]["role"] == "assistant"


def test_normalize_item_requires_role():
    with pytest.raises(resp.ResponsesInputError):
        resp._normalize_item({"content": "hi"})


def test_normalize_item_developer_role():
    assert resp._normalize_item({"role": "developer", "content": "x"}) == [{"role": "system", "content": "x"}]


def test_normalize_item_unsupported_role():
    with pytest.raises(resp.ResponsesInputError):
        resp._normalize_item({"role": "bogus"})


def test_normalize_input_none_and_dict():
    assert resp.normalize_input(None) == []
    assert resp.normalize_input({"role": "user", "content": "x"}) == [{"role": "user", "content": "x"}]


def test_normalize_input_string():
    assert resp.normalize_input("hi") == [{"role": "user", "content": "hi"}]


def test_normalize_input_list():
    assert resp.normalize_input([{"role": "user", "content": "x"}]) == [{"role": "user", "content": "x"}]


def test_normalize_input_invalid():
    with pytest.raises(resp.ResponsesInputError):
        resp.normalize_input(123)


def test_ensure_input_present_keeps_valid_input():
    messages = [{"role": "user", "content": "hi"}]
    assert resp.ensure_input_present(messages) is messages
    tool_calls = [{"role": "assistant", "content": "", "tool_calls": [{"id": "c", "type": "function", "function": {"name": "f", "arguments": "{}"}}]}]
    assert resp.ensure_input_present(tool_calls) is tool_calls


def test_ensure_input_present_rejects_empty_input():
    for value in ([], "hi", [{}], [{"role": "user", "content": ""}], [{"role": "user", "content": "   "}], [{"role": "user", "content": []}]):
        with pytest.raises(resp.ResponsesInputError):
            resp.ensure_input_present(value)


def test_validate_tool_chain_accepts_answered_calls():
    resp.validate_tool_chain(
        [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "42"},
            {"role": "assistant", "content": "done"},
        ]
    )
    resp.validate_tool_chain(5)


def test_validate_tool_chain_rejects_unmatched_tool_result():
    with pytest.raises(resp.ResponsesInputError):
        resp.validate_tool_chain([{"role": "tool", "tool_call_id": "c1", "content": "42"}])
    with pytest.raises(resp.ResponsesInputError) as excinfo:
        resp.validate_tool_chain(
            [
                {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]},
                {"role": "tool", "tool_call_id": "c2", "content": "42"},
            ]
        )
    assert "c2" in str(excinfo.value)
    with pytest.raises(resp.ResponsesInputError):
        resp.validate_tool_chain([{"role": "assistant", "content": "", "tool_calls": [5, {"id": ""}]}, {"role": "tool", "tool_call_id": "", "content": "42"}])
    with pytest.raises(resp.ResponsesInputError):
        resp.validate_tool_chain([{"role": "assistant", "content": "", "tool_calls": [5]}, {"role": "tool", "tool_call_id": "c1", "content": "42"}])


def test_convert_tools_non_list():
    assert resp.convert_tools("x") is None


def test_convert_tools_skips_non_dict():
    assert resp.convert_tools(["x"]) is None


def test_convert_tools_nested():
    nested = {"type": "function", "function": {"name": "f"}}
    assert resp.convert_tools([nested]) == [nested]


def test_convert_tools_flat_fields():
    tools = resp.convert_tools([{"type": "function", "name": "f", "description": "d", "parameters": {"type": "object"}}])
    assert tools == [{"type": "function", "function": {"name": "f", "description": "d", "parameters": {"type": "object"}}}]


def test_convert_tools_strict():
    tools = resp.convert_tools([{"type": "function", "name": "f", "strict": True}])
    assert tools == [{"type": "function", "function": {"name": "f", "strict": True}}]


def test_convert_tool_choice_none():
    assert resp.convert_tool_choice(None) is None


def test_convert_tool_choice_str():
    assert resp.convert_tool_choice("auto") == "auto"


def test_convert_tool_choice_type():
    assert resp.convert_tool_choice({"type": "required"}) == "required"


def test_convert_tool_choice_nested_and_unknown():
    assert resp.convert_tool_choice({"type": "function", "function": {"name": "f"}}) == {
        "type": "function",
        "function": {"name": "f"},
    }
    assert resp.convert_tool_choice({"type": "weird"}) is None


def test_response_text_format_dict():
    assert resp.response_text_format({"format": {"type": "json_object"}}) == {"type": "json_object"}


def test_response_text_format_branches():
    assert resp.response_text_format({"format": "json_object"}) == {"type": "json_object"}
    assert resp.response_text_format({"format": 123}) is None


def test_response_text_format_str_and_other():
    assert resp.response_text_format("json_object") == {"type": "json_object"}
    assert resp.response_text_format(5) is None


def test_extract_response_format_none():
    assert resp.extract_response_format(None, None) is None


def test_extract_response_format_json_object():
    assert resp.extract_response_format({"format": {"type": "json_object"}}, None) == {"type": "json_object"}


def test_extract_response_format_unknown_type():
    assert resp.extract_response_format({"format": {"type": "text"}}, None) is None


def test_extract_response_format_str_and_non_dict():
    assert resp.extract_response_format(None, "json_object") == "json_object"
    assert resp.extract_response_format(None, 5) is None


def test_extract_response_format_nested_schema():
    fmt = resp.extract_response_format({"format": {"type": "json_schema", "json_schema": {"name": "n", "schema": {}}}}, None)
    assert fmt == {"type": "json_schema", "json_schema": {"name": "n", "schema": {}}}


def test_extract_response_format_schema_missing():
    assert resp.extract_response_format({"format": {"type": "json_schema"}}, None) is None


def test_extract_response_format_strict_and_description():
    fmt = resp.extract_response_format({"format": {"type": "json_schema", "name": "n", "schema": {}, "strict": True, "description": "d"}}, None)
    assert fmt["json_schema"]["strict"] is True
    assert fmt["json_schema"]["description"] == "d"


def test_usage_total_is_recomputed_from_parts():
    usage = resp._usage_to_responses({"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 999})
    assert usage is not None
    assert usage["input_tokens"] == 5
    assert usage["output_tokens"] == 2
    assert usage["total_tokens"] == 7
    assert resp._usage_to_responses(None) is None
    empty = resp._usage_to_responses({"total_tokens": 999})
    assert empty is not None
    assert empty["total_tokens"] == 0


def test_build_response_object_usage_by_status():
    info = resp.RequestInfo(model="m")
    for status in ("completed", "incomplete", "failed"):
        reported = resp.build_response_object(info, "r", 1, output=[], status=status)["usage"]
        assert reported == {
            "input_tokens": 0,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": 0,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 0,
        }
    for status in ("in_progress", "queued"):
        assert resp.build_response_object(info, "r", 1, output=[], status=status)["usage"] is None


def test_reasoning_text_from_output_variants():
    assert resp._reasoning_text_from_output(5) == ""
    out = [
        5,
        {"type": "reasoning", "text": "hello"},
        {"type": "reasoning", "summary": [5, {"text": "world"}]},
    ]
    assert resp._reasoning_text_from_output(out) == "helloworld"


def test_output_items_from_message_non_dict():
    assert resp.output_items_from_message(5) == []


def test_output_items_from_message_reasoning():
    items = resp.output_items_from_message({"role": "assistant", "content": "x", "reasoning_content": "r"})
    assert items[0]["type"] == "reasoning"


def test_output_items_from_message_skips_bad_call():
    items = resp.output_items_from_message({"role": "assistant", "content": "hi", "tool_calls": [5]})
    assert len(items) == 1


def test_output_items_from_message_call_fallback():
    items = resp.output_items_from_message({"role": "assistant", "content": "", "tool_calls": [{"name": "f", "arguments": {"a": 1}}]})
    calls = [item for item in items if item["type"] == "function_call"]
    assert calls[0]["name"] == "f"
    assert calls[0]["arguments"] == json.dumps({"a": 1})


def test_messages_from_output_variants():
    assert resp.messages_from_output(5) == []
    assert resp.messages_from_output([5, {"type": "message", "content": "hi"}]) == [{"role": "assistant", "content": "hi"}]


def test_messages_from_output_content_parts():
    assert resp.messages_from_output([{"type": "message", "content": [{"text": "hi"}]}]) == [{"role": "assistant", "content": "hi"}]


def test_messages_from_output_calls():
    output = [{"type": "function_call", "call_id": "c", "name": "f", "arguments": "{}"}]
    messages = resp.messages_from_output(output)
    assert messages[-1]["tool_calls"][0]["id"] == "c"


def test_messages_from_output_reasoning_only_is_empty():
    output = [{"id": "rs_1", "type": "reasoning", "summary": [{"type": "summary_text", "text": "think"}]}]
    assert resp.messages_from_output(output) == []


def test_input_message_item_tool_role():
    items = resp._input_message_item({"role": "tool", "tool_call_id": "c1", "content": "42"})
    assert items[0]["type"] == "function_call_output"
    assert items[0]["output"] == "42"


def test_input_message_item_parts():
    items = resp._input_message_item(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "hi"},
                {"type": "image_url", "image_url": {"url": "data:x"}},
                {"type": "input_file", "filename": "f.txt", "file": {"id": "x"}},
                5,
            ],
        }
    )
    parts = items[0]["content"]
    assert parts[0] == {"type": "input_text", "text": "hi", "annotations": []}
    assert parts[1]["type"] == "input_image"
    assert parts[1]["image_url"] == "data:x"
    assert parts[2]["type"] == "input_file"


def test_input_message_item_empty_and_str_content():
    empty = resp._input_message_item({"role": "user", "content": ""})
    assert empty[0]["content"] == [{"type": "input_text", "text": "", "annotations": []}]
    text = resp._input_message_item({"role": "user", "content": "hello"})
    assert text[0]["content"][0]["text"] == "hello"


def test_input_message_item_tool_calls_and_reasoning():
    items = resp._input_message_item(
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "think",
            "tool_calls": [5, {"name": "f", "arguments": {"a": 1}}],
        }
    )
    assert items[0]["type"] == "reasoning"
    calls = [item for item in items if item["type"] == "function_call"]
    assert calls[0]["name"] == "f"


def test_input_items_from_messages_variants():
    assert resp.input_items_from_messages(5) == []
    items = resp.input_items_from_messages([{"role": "user", "content": "hi"}, 5])
    assert len(items) == 1


def test_input_items_from_messages_ids_are_stable_and_unique():
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello", "reasoning_content": "think"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "function": {"name": "f", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "42"},
        {"role": "user", "content": "hi"},
    ]
    first = resp.input_items_from_messages(messages)
    second = resp.input_items_from_messages(messages)
    assert [item["id"] for item in first] == [item["id"] for item in second]
    assert len({item["id"] for item in first}) == len(first)
    assert first[0]["id"].startswith("msg_")
    duplicate = [item for item in first if item.get("content") == [{"type": "input_text", "text": "hi", "annotations": []}]]
    assert len(duplicate) == 2
    assert duplicate[0]["id"] != duplicate[1]["id"]
    outputs = [item for item in first if item["type"] == "function_call_output"]
    assert len(outputs) == 1
    assert outputs[0]["call_id"] == "c1"
    assert outputs[0]["output"] == "42"
    assert [item["type"] for item in first] == ["message", "reasoning", "message", "message", "function_call", "function_call_output", "message"]


def test_response_from_chat_invalid():
    with pytest.raises(HTTPException) as excinfo:
        resp.response_from_chat(5, resp.RequestInfo(model="m"), "r", 1)
    assert excinfo.value.status_code == 500


def test_response_from_chat_incomplete_reason():
    info = resp.RequestInfo(model="m")
    chat = {"choices": [{"message": {"role": "assistant", "content": "x"}, "finish_reason": "length"}]}
    obj = resp.response_from_chat(chat, info, "r", 1)
    assert obj["status"] == "incomplete"
    assert obj["incomplete_details"] == {"reason": "max_output_tokens"}


def test_response_from_chat_error_and_incomplete():
    info = resp.RequestInfo(model="m")
    errored = resp.response_from_chat(
        {"choices": [{"message": {"role": "assistant", "content": "x"}, "finish_reason": None}], "error": {"message": "bad"}},
        info,
        "r",
        1,
    )
    assert errored["status"] == "failed"
    assert errored["error"]["message"] == "bad"
    unfinished = resp.response_from_chat(
        {"choices": [{"message": {"role": "assistant", "content": "x"}, "finish_reason": "response_incomplete"}]},
        info,
        "r",
        1,
    )
    assert unfinished["status"] == "incomplete"
    assert unfinished["incomplete_details"] == {"reason": "max_output_tokens"}


def test_iter_sse_payloads_non_str():
    assert list(resp._iter_sse_payloads(5)) == []


def test_iter_sse_payloads_crlf():
    assert list(resp._iter_sse_payloads('data: {"a": 1}\r\ndata: {"b": 2}\n\n')) == []


def test_iter_sse_payloads_bad_json():
    assert list(resp._iter_sse_payloads("data: nope\n\n")) == []


def test_iter_sse_payloads_done():
    assert list(resp._iter_sse_payloads("data: [DONE]\n\n")) == [None]


def test_error_payload_dict():
    payload = resp._error_payload({"message": "bad"})
    assert payload == {"type": "server_error", "code": None, "message": "bad", "param": None}
    typed = resp._error_payload({"type": "rate_limit_error", "code": "rate_limit_exceeded", "message": "slow", "param": "model"})
    assert typed == {"type": "rate_limit_error", "code": "rate_limit_exceeded", "message": "slow", "param": "model"}
    assert resp._error_payload({"message": 5}) == {"type": "server_error", "code": None, "message": "stream error", "param": None}


def test_error_payload_does_not_expose_upstream_finish_reason():
    payload = resp._error_payload({"message": "reduced", "finish_reason": "response_incomplete", "code": 42})
    assert payload == {"type": "server_error", "code": None, "message": "reduced", "param": None}
    assert "finish_reason" not in json.dumps(payload)


def test_error_payload_non_dict():
    payload = resp._error_payload("boom")
    assert payload["type"] == "server_error"
    assert payload["message"] == "stream error"
    assert payload["code"] is None
    assert payload["param"] is None
    assert "boom" not in json.dumps(payload)
    for value in (None, 5, ["boom"], b"boom"):
        assert resp._error_payload(value)["message"] == "stream error"


def test_response_error_normalises_to_code_message():
    assert resp._response_error(None) is None
    assert resp._response_error({"message": "bad"}) == {"code": None, "message": "bad"}
    assert resp._response_error({"message": "bad", "finish_reason": "server_busy", "param": "model"}) == {"code": None, "message": "bad"}
    assert resp._response_error({"message": "bad", "code": "rate_limit_exceeded"}) == {"code": "rate_limit_exceeded", "message": "bad"}
    assert resp._response_error("boom") == {"code": None, "message": "upstream request failed"}
    assert resp._response_error({}) == {"code": None, "message": "upstream request failed"}


def test_stream_state_reasoning_and_message():
    state = resp._StreamState(resp.RequestInfo(model="m"), "r", 1)
    lines = list(state.reasoning_delta("think"))
    assert any("response.reasoning_summary_text.delta" in line for line in lines)
    lines = list(state.message_delta("Hi"))
    assert any("response.reasoning_summary_text.done" in line for line in lines)
    lines = list(state.close_all())
    assert any("response.output_item.done" in line for line in lines)


def test_stream_state_reasoning_closes_an_open_message_first():
    state = resp._StreamState(resp.RequestInfo(model="m"), "r", 1)
    list(state.message_delta("hi"))
    lines = list(state.reasoning_delta("think"))
    assert state.message_open is False
    assert state.reasoning_open is True
    names = [line.split("\n")[0] for line in lines]
    assert "event: response.output_item.done" in names
    assert names.index("event: response.output_item.done") < names.index("event: response.output_item.added")


async def test_translate_stream_closes_the_message_before_the_reasoning_item_opens():
    info = resp.RequestInfo(model="m")
    stream = _agen(
        [
            'data: {"id":"x","choices":[{"index":0,"delta":{"content":"Hi"},"finish_reason":null}]}\n\n',
            'data: {"id":"x","choices":[{"index":0,"delta":{"reasoning_content":"think"},"finish_reason":null}]}\n\n',
            'data: {"id":"x","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n',
        ]
    )
    lines = "".join(await _collect(resp.translate_stream(stream, info, "r", 1)))
    events = [block.split("\n")[0] for block in lines.split("event: ")[1:]]
    added = [position for position, name in enumerate(events) if name == "response.output_item.added"]
    done = [position for position, name in enumerate(events) if name == "response.output_item.done"]
    assert len(added) == 2
    assert len(done) == 2
    assert done[0] < added[1]
    final = json.loads(lines.rsplit("event: response.completed\ndata: ", 1)[1])["response"]
    assert [item["type"] for item in final["output"]] == ["message", "reasoning"]


def test_stream_state_tool_delta_non_list():
    state = resp._StreamState(resp.RequestInfo(model="m"), "r", 1)
    assert list(state.tool_delta("x")) == []


def test_stream_state_tool_delta_odd_calls():
    state = resp._StreamState(resp.RequestInfo(model="m"), "r", 1)
    list(state.tool_delta([5, {"function": 5, "id": "c"}]))
    lines = list(state.close_tools())
    assert any("response.output_item.done" in line for line in lines)
    assert list(state.close_tools()) == []


def test_stream_state_tool_delta_full():
    state = resp._StreamState(resp.RequestInfo(model="m"), "r", 1)
    list(state.tool_delta([{"index": 0, "function": {"arguments": '{"a"'}}]))
    list(state.tool_delta([{"index": 0, "function": {"name": "f"}}]))
    lines = list(state.tool_delta([{"index": 0, "function": {"arguments": ":1}"}}]))
    assert any("response.function_call_arguments.delta" in line for line in lines)


def test_close_tools_buffered():
    state = resp._StreamState(resp.RequestInfo(model="m"), "r", 1)
    list(state.tool_delta([{"index": 0, "function": {"arguments": '{"a"'}}]))
    lines = list(state.close_tools())
    assert any("response.output_item.added" in line for line in lines)


async def test_translate_stream_reasoning_and_incomplete():
    info = resp.RequestInfo(model="m")
    stream = _agen(
        [
            'data: {"id":"x","choices":[{"index":0,"delta":{"reasoning_content":"think"},"finish_reason":null}]}\n\n',
            'data: {"id":"x","choices":[{"index":0,"delta":{"content":"Hi"},"finish_reason":"length"}],'
            '"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}\n\n',
        ]
    )
    seen = []
    lines = "".join(await _collect(resp.translate_stream(stream, info, "r", 1, on_complete=seen.append)))
    assert "response.reasoning_summary_text.delta" in lines
    assert "response.incomplete" in lines
    assert seen


async def test_translate_stream_completed_on_complete():
    info = resp.RequestInfo(model="m")
    stream = _agen(
        [
            'data: {"id":"x","choices":[{"index":0,"delta":{"content":"Hi"},"finish_reason":"stop"}]}\n\n',
            "data: [DONE]\n\n",
        ]
    )
    seen = []
    lines = "".join(await _collect(resp.translate_stream(stream, info, "r", 1, on_complete=seen.append)))
    assert "response.completed" in lines
    assert seen


async def test_translate_stream_still_emits_the_terminal_event_when_persisting_fails(caplog):
    info = resp.RequestInfo(model="m")
    stream = _agen(['data: {"id":"x","choices":[{"index":0,"delta":{"content":"Hi"},"finish_reason":"stop"}]}\n\n'])

    async def on_complete(_final):
        raise RuntimeError("the responses store is full")

    with caplog.at_level(logging.WARNING, logger="danyapi.api.responses"):
        lines = "".join(await _collect(resp.translate_stream(stream, info, "r", 1, on_complete=on_complete)))
    assert "response.completed" in lines
    assert "could not be persisted" in caplog.text


async def test_translate_stream_propagates_a_cancellation_from_persisting():
    info = resp.RequestInfo(model="m")
    stream = _agen(['data: {"id":"x","choices":[{"index":0,"delta":{"content":"Hi"},"finish_reason":"stop"}]}\n\n'])

    async def on_complete(_final):
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await _collect(resp.translate_stream(stream, info, "r", 1, on_complete=on_complete))


async def test_translate_stream_tool_calls_and_error():
    info = resp.RequestInfo(model="m")
    tool_stream = _agen(
        [
            'data: {"id":"x","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"c","function":'
            '{"name":"f","arguments":"{}"}}]},"finish_reason":"tool_calls"}]}\n\n',
        ]
    )
    joined = "".join(await _collect(resp.translate_stream(tool_stream, info, "r", 1)))
    assert "response.function_call_arguments.done" in joined

    error_stream = _agen(['data: {"id":"x","error":{"message":"boom"},"choices":[]}\n\n'])
    joined = "".join(await _collect(resp.translate_stream(error_stream, info, "r", 1)))
    assert "event: error" in joined
    assert "response.completed" not in joined


async def test_translate_stream_skips_malformed():
    info = resp.RequestInfo(model="m")
    stream = _agen(
        [
            'data: {"id":"x","usage":{"prompt_tokens":1}}\n\n',
            'data: {"id":"x","choices":[5]}\n\n',
        ]
    )
    lines = "".join(await _collect(resp.translate_stream(stream, info, "r", 1)))
    assert "response.completed" in lines
