import json

import pytest

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


def test_normalize_item_reasoning():
    assert resp._normalize_item({"type": "reasoning"}) == []


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


def test_response_from_chat_invalid():
    with pytest.raises(resp.ResponsesInputError):
        resp.response_from_chat(5, resp.RequestInfo(model="m"), "r", 1)


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
    assert errored["status"] == "incomplete"
    unfinished = resp.response_from_chat(
        {"choices": [{"message": {"role": "assistant", "content": "x"}, "finish_reason": "response_incomplete"}]},
        info,
        "r",
        1,
    )
    assert unfinished["status"] == "incomplete"


def test_iter_sse_payloads_non_str():
    assert list(resp._iter_sse_payloads(5)) == []


def test_iter_sse_payloads_crlf():
    assert list(resp._iter_sse_payloads('data: {"a": 1}\r\ndata: {"b": 2}\n\n')) == [{"b": 2}]


def test_iter_sse_payloads_bad_json():
    assert list(resp._iter_sse_payloads("data: nope\n\n")) == []


def test_iter_sse_payloads_done():
    assert list(resp._iter_sse_payloads("data: [DONE]\n\n")) == [None]


def test_error_payload_dict():
    payload = resp._error_payload({"message": "bad"})
    assert payload["message"] == "bad"


def test_error_payload_non_dict():
    payload = resp._error_payload("boom")
    assert payload["type"] == "server_error"
    assert payload["message"] == "boom"


def test_stream_state_reasoning_and_message():
    state = resp._StreamState(resp.RequestInfo(model="m"), "r", 1)
    lines = list(state.reasoning_delta("think"))
    assert any("response.reasoning_summary_text.delta" in line for line in lines)
    lines = list(state.message_delta("Hi"))
    assert any("response.reasoning_summary_text.done" in line for line in lines)
    lines = list(state.close_all())
    assert any("response.output_item.done" in line for line in lines)


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
