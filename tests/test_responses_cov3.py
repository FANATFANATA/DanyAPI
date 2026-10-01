import hashlib
import json
import logging

import pytest
from fastapi.testclient import TestClient

from danyapi.api import responses as resp
from danyapi.api.openai import app


async def _agen(items):
    for item in items:
        yield item


async def _collect(agen):
    out = []
    async for item in agen:
        out.append(item)
    return out


def _frames(text: str) -> list[tuple[str, dict]]:
    frames: list[tuple[str, dict]] = []
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
        if not data:
            continue
        frames.append((event, json.loads(data)))
    return frames


def _event(frames, name):
    return [payload for event, payload in frames if event == name]


def test_usage_reads_float_string_token_counts():
    usage = resp._usage_to_responses({"prompt_tokens": "12.7", "completion_tokens": "3"})
    assert usage == {
        "input_tokens": 12,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens": 3,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": 15,
    }


def test_usage_survives_a_non_finite_token_count():
    usage = resp._usage_to_responses({"prompt_tokens": float("inf"), "completion_tokens": float("nan")})
    assert usage is not None
    assert usage["input_tokens"] == 0
    assert usage["output_tokens"] == 0
    assert usage["total_tokens"] == 0


def test_as_text_is_bounded_in_depth():
    assert resp._as_text([[[[[[[["deep"]]]]]]]]) == "deep"
    assert resp._as_text([[[[[[[[["deep"]]]]]]]]]) == ""


def test_as_text_falls_back_to_repr_for_a_circular_dict():
    circular: dict = {}
    circular["self"] = circular
    assert resp._as_text(circular) == str(circular)


def test_as_text_falls_back_to_repr_for_an_unserialisable_dict():
    text = resp._as_text({"x": object()})
    assert text.startswith("{'x': <object object at ")
    assert text.endswith(">}")


def test_ensure_input_present_skips_non_dict_messages():
    messages = [5, {"role": "user", "content": "hi"}]
    assert resp.ensure_input_present(messages) is messages


def test_validate_tool_chain_skips_non_dict_messages():
    assert resp.validate_tool_chain([5, {"role": "user", "content": "hi"}]) is None


def test_response_text_format_keeps_the_json_schema_discriminator():
    assert resp.response_text_format({"format": {"type": "json_schema", "json_schema": {"name": "n", "schema": {"type": "object"}}}}) == {
        "type": "json_schema",
        "json_schema": {"name": "n", "schema": {"type": "object"}},
    }
    assert resp.response_text_format({"format": {"type": "json_schema", "json_schema": {"name": "n"}}}) != resp.response_text_format(
        {"format": {"type": "text"}}
    )


def test_response_text_format_does_not_alias_the_request_payload():
    payload = {"format": {"type": "json_schema", "json_schema": {"name": "n"}}}
    formatted = resp.response_text_format(payload)
    assert formatted is not None
    formatted["json_schema"]["name"] = "changed"
    assert payload["format"]["json_schema"]["name"] == "n"


def test_messages_from_output_encodes_structured_arguments():
    messages = resp.messages_from_output([{"type": "function_call", "call_id": "c", "name": "f", "arguments": {"a": [1, 2]}}])
    assert messages[0]["tool_calls"][0]["function"]["arguments"] == json.dumps({"a": [1, 2]}, ensure_ascii=False)


def test_stable_id_falls_back_to_repr_for_unserialisable_payloads():
    circular: dict = {}
    circular["self"] = circular
    digest = hashlib.blake2s(str(circular).encode("utf-8", "replace"), digest_size=12).hexdigest()
    assert resp._stable_id("msg", circular) == f"msg_{digest}"
    assert resp._stable_id("msg", circular) == f"msg_{digest}"


def test_stable_id_marks_each_occurrence():
    occurrences: dict[str, int] = {}
    first = resp._stable_id("fc", {"a": 1}, occurrences)
    second = resp._stable_id("fc", {"a": 1}, occurrences)
    third = resp._stable_id("fc", {"a": 1}, occurrences)
    assert first.startswith("fc_")
    assert second == f"{first}1"
    assert third == f"{first}2"
    assert resp._stable_id("rs", {"a": 1}, occurrences).startswith("rs_")
    assert resp._stable_id("msg", {"a": 1}) == resp._stable_id("msg", {"a": 1})


def test_legacy_tool_message_gets_a_deterministic_non_name_call_id():
    message = {"role": "function", "name": "lookup", "content": "42"}
    first = resp._input_message_item(message)
    second = resp._input_message_item(message)
    assert first[0]["type"] == "function_call_output"
    assert first[0]["call_id"].startswith("call_")
    assert first[0]["call_id"] != "lookup"
    assert first[0]["call_id"] == second[0]["call_id"]
    assert first[0]["id"] == second[0]["id"]
    assert first[0]["id"].startswith("fc_")


def test_stored_ids_carry_the_item_kind_prefix():
    items = resp.input_items_from_messages(
        [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "", "reasoning_content": "think"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "function": {"name": "f", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "42"},
        ]
    )
    kinds = [item["id"].split("_", 1)[0] for item in items]
    assert kinds == ["msg", "rs", "msg", "msg", "fc", "fc"]
    assert [item["type"] for item in items] == ["message", "reasoning", "message", "message", "function_call", "function_call_output"]


def test_message_item_with_only_an_empty_text_part_yields_an_empty_assistant_message():
    output = [{"id": "msg_1", "type": "message", "role": "assistant", "content": [{"type": "output_text", "text": ""}]}]
    assert resp.messages_from_output(output) == [{"role": "assistant", "content": ""}]


def test_input_file_without_a_file_object_is_skipped():
    items = resp._input_message_item({"role": "user", "content": [{"type": "input_file", "filename": "g.txt"}]})
    assert items[0]["content"] == [{"type": "input_text", "text": "", "annotations": []}]


def test_normalize_input_treats_a_blank_string_as_no_input():
    assert resp.normalize_input("   ") == []
    assert resp.normalize_input("\n\t") == []


def test_final_finish_without_an_active_choice_uses_the_lowest_index():
    state = resp._StreamState(resp.RequestInfo(model="m"), "r", 1)
    state.finishes = {1: "length", 0: "stop"}
    assert state.active_choice is None
    assert state.final_finish() == "stop"
    state.active_choice = 1
    assert state.final_finish() == "length"
    state.finishes = {}
    assert state.final_finish() is None


def test_merge_usage_skips_empty_and_placeholder_values():
    state = resp._StreamState(resp.RequestInfo(model="m"), "r", 1)
    resp._merge_usage(state, None)
    assert state.usage is None
    resp._merge_usage(state, "not a dict")
    assert state.usage is None
    resp._merge_usage(state, {"prompt_tokens": 5, "completion_tokens": None, "cached_tokens": 0, "flag": True, "details": {}, "tags": [], "note": ""})
    assert state.usage == {"prompt_tokens": 5}
    resp._merge_usage(state, {"prompt_tokens": 6, "completion_tokens": 2})
    assert state.usage == {"prompt_tokens": 6, "completion_tokens": 2}


def test_error_payload_logs_a_non_dict_error_instead_of_forwarding_it(caplog):
    with caplog.at_level(logging.WARNING, logger="danyapi.api.responses"):
        payload = resp._error_payload(["boom"])
    assert payload == {"type": "server_error", "code": None, "message": "stream error", "param": None}
    assert [record.getMessage() for record in caplog.records] == ["upstream stream produced a non-dict error: list"]


def test_bytes_chunks_are_decoded_and_other_types_are_dropped():
    assert list(resp._iter_sse_payloads(b'data: {"a": 1}\n\n')) == [{"a": 1}]
    assert list(resp._iter_sse_payloads(bytearray(b'data: {"b": 2}\n\n'))) == [{"b": 2}]
    assert list(resp._iter_sse_payloads(5)) == []
    assert list(resp._iter_sse_payloads(None)) == []


def _post_responses(payload):
    app.state.pool = None
    app.state.qwen_pool = None
    client = TestClient(app)
    try:
        return client.post("/v1/responses", json=payload)
    finally:
        client.close()


def _error_body(response):
    body = response.json()
    assert sorted(body["error"]) == ["code", "message", "param", "request_id", "type"]
    assert body["error"]["code"] == "invalid_request_error"
    assert body["error"]["type"] == "invalid_request_error"
    assert body["error"]["param"] is None
    assert len(body["error"]["request_id"]) == 32
    return body["error"]["message"]


@pytest.mark.parametrize("item_type", ["reasoning", "item_reference", "web_search_call", "file_search_call", "code_interpreter_call"])
def test_endpoint_rejects_unreplayable_input_items(item_type):
    response = _post_responses({"model": "deepseek-v4.1-flash", "input": [{"type": item_type}]})
    assert response.status_code == 400
    assert _error_body(response) == f"input item type {item_type!r} is not supported: {resp.UNSUPPORTED_ITEM_TYPES[item_type]}"


@pytest.mark.parametrize("item_type", ["function_call_output", "computer_call_output", "custom_tool_call_output"])
def test_endpoint_rejects_a_tool_result_without_any_call_id(item_type):
    response = _post_responses({"model": "deepseek-v4.1-flash", "input": [{"type": item_type, "output": "42"}]})
    assert response.status_code == 400
    assert _error_body(response) == f"{item_type} requires a non-empty call_id"


def test_endpoint_rejects_an_empty_input():
    response = _post_responses({"model": "deepseek-v4.1-flash", "input": "   "})
    assert response.status_code == 400
    assert _error_body(response) == "input must contain at least one input item"


def test_legacy_tool_message_loses_its_call_id_during_normalisation():
    assert resp.normalize_input([{"role": "tool", "tool_call_id": "c1", "content": "42"}]) == [{"role": "tool", "content": "42"}]


def test_endpoint_reports_a_missing_call_id_for_a_legacy_tool_message():
    response = _post_responses({"model": "deepseek-v4.1-flash", "input": [{"role": "tool", "tool_call_id": "c1", "content": "42"}]})
    assert response.status_code == 400
    assert _error_body(response) == "messages[0] is a tool result for an unknown call_id: None"


def test_endpoint_rejects_a_tool_result_whose_call_was_never_announced():
    response = _post_responses(
        {
            "model": "deepseek-v4.1-flash",
            "input": [
                {"type": "function_call", "call_id": "c1", "name": "f", "arguments": "{}"},
                {"type": "function_call_output", "call_id": "c2", "output": "42"},
            ],
        }
    )
    assert response.status_code == 400
    assert _error_body(response) == "messages[1] is a tool result for an unknown call_id: 'c2'"


async def test_close_chat_stream_ignores_a_stream_without_a_close():
    assert await resp._close_chat_stream(object()) is None


class _FailingClose:
    def __init__(self):
        self.calls = 0

    async def aclose(self):
        self.calls += 1
        raise RuntimeError("cannot close")


async def test_close_chat_stream_logs_a_failing_close(caplog):
    stream = _FailingClose()
    with caplog.at_level(logging.DEBUG, logger="danyapi.api.responses"):
        assert await resp._close_chat_stream(stream) is None
    assert stream.calls == 1
    assert [record.getMessage() for record in caplog.records] == ["upstream chat stream close failed: cannot close"]


async def test_translate_stream_survives_a_failing_upstream_close():
    class _Stream:
        def __aiter__(self):
            return self._gen()

        async def _gen(self):
            yield 'data: {"choices":[{"index":0,"delta":{"content":"Hi"},"finish_reason":"stop"}]}\n\n'

        async def aclose(self):
            raise RuntimeError("cannot close")

    frames = await _collect(resp.translate_stream(_Stream(), resp.RequestInfo(model="m"), "r", 1))
    assert "response.completed" in "".join(frames)


async def test_terminal_lines_await_an_async_on_complete():
    seen: list[dict] = []

    async def on_complete(final):
        seen.append(final)

    state = resp._StreamState(resp.RequestInfo(model="m"), "r", 1, on_complete)
    lines = [line async for line in resp._terminal_lines(state, "completed")]
    assert lines[0].startswith("event: response.completed\n")
    assert json.loads(lines[0].split("data: ", 1)[1])["response"] == seen[0]
    assert seen[0]["status"] == "completed"
    assert seen[0]["usage"] == {
        "input_tokens": 0,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens": 0,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": 0,
    }


async def test_terminal_lines_ignore_a_missing_callback():
    state = resp._StreamState(resp.RequestInfo(model="m"), "r", 1, None)
    lines = [line async for line in resp._terminal_lines(state, "failed", {"message": "boom"})]
    assert lines[0].startswith("event: response.failed\n")
    assert json.loads(lines[0].split("data: ", 1)[1])["response"]["error"] == {"code": None, "message": "boom"}


async def test_output_is_ordered_by_the_announced_indices():
    stream = _agen(
        [
            'data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"c","function":{"name":"f","arguments":"{}"}}]}}]}\n\n',
            'data: {"choices":[{"index":0,"delta":{"reasoning_content":"think"}}]}\n\n',
            'data: {"choices":[{"index":0,"delta":{"content":"Hi"},"finish_reason":"stop"}]}\n\n',
        ]
    )
    frames = _frames("".join(await _collect(resp.translate_stream(stream, resp.RequestInfo(model="m"), "r", 1))))
    added = _event(frames, "response.output_item.added")
    assert [payload["output_index"] for payload in added] == [0, 1, 2]
    completed = _event(frames, "response.completed")[0]["response"]
    assert [item["id"] for item in completed["output"]] == [payload["item"]["id"] for payload in added]
    assert [item["type"] for item in completed["output"]] == ["function_call", "reasoning", "message"]
    assert completed["output"][0]["call_id"] == "c"
    assert completed["output"][1]["summary"] == [{"type": "summary_text", "text": "think"}]
    assert completed["output"][2]["content"] == [{"type": "output_text", "text": "Hi", "annotations": []}]


@pytest.mark.parametrize(
    ("first_finish", "second_finish", "expected_status"),
    [("length", "stop", "incomplete"), ("stop", "length", "completed")],
)
async def test_status_comes_from_the_choice_that_produced_content(first_finish, second_finish, expected_status):
    finishes = {
        "choices": [
            {"index": 0, "delta": {}, "finish_reason": first_finish},
            {"index": 1, "delta": {}, "finish_reason": second_finish},
        ]
    }
    stream = _agen(
        [
            'data: {"choices":[{"index":0,"delta":{"content":"Hi"}}]}\n\n',
            f"data: {json.dumps(finishes)}\n\n",
        ]
    )
    frames = _frames("".join(await _collect(resp.translate_stream(stream, resp.RequestInfo(model="m"), "r", 1))))
    terminal = [event for event in {event for event, _ in frames} if event in ("response.completed", "response.incomplete")]
    assert terminal == [f"response.{expected_status}"]
    final = _event(frames, f"response.{expected_status}")[0]["response"]
    assert final["status"] == expected_status
    assert final["output"][0]["content"][0]["text"] == "Hi"


async def test_partial_mid_stream_usage_does_not_zero_the_counters():
    stream = _agen(
        [
            'data: {"choices":[{"index":0,"delta":{"content":"Hi"}}],"usage":{"prompt_tokens":11,"completion_tokens":7,"total_tokens":999}}\n\n',
            'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"completion_tokens":0}}\n\n',
        ]
    )
    frames = _frames("".join(await _collect(resp.translate_stream(stream, resp.RequestInfo(model="m"), "r", 1))))
    usage = _event(frames, "response.completed")[0]["response"]["usage"]
    assert usage == {
        "input_tokens": 11,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens": 7,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": 18,
    }


async def test_reasoning_tokens_are_estimated_from_the_stored_output():
    stream = _agen(
        [
            'data: {"choices":[{"index":0,"delta":{"reasoning_content":"think hard about it"},"finish_reason":"stop"}],'
            '"usage":{"prompt_tokens":1,"completion_tokens":2}}\n\n',
        ]
    )
    frames = _frames("".join(await _collect(resp.translate_stream(stream, resp.RequestInfo(model="m"), "r", 1))))
    usage = _event(frames, "response.completed")[0]["response"]["usage"]
    assert usage == {
        "input_tokens": 1,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens": 2,
        "output_tokens_details": {"reasoning_tokens": 4},
        "total_tokens": 3,
    }


async def test_upstream_failure_closes_items_emits_failed_and_reraises():
    async def failing():
        yield 'data: {"choices":[{"index":0,"delta":{"content":"Hi"}}]}\n\n'
        yield 'data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"c","function":{"name":"f","arguments":"{}"}}]}}]}\n\n'
        raise RuntimeError("upstream boom")

    frames: list[tuple[str, dict]] = []
    with pytest.raises(RuntimeError) as excinfo:
        async for line in resp.translate_stream(failing(), resp.RequestInfo(model="m"), "r", 1):
            frames.extend(_frames(line))
    assert str(excinfo.value) == "upstream boom"
    events = [event for event, _ in frames]
    assert events.count("response.output_item.done") == 2
    failed = _event(frames, "response.failed")[0]["response"]
    assert failed["status"] == "failed"
    assert failed["error"] == {"code": None, "message": "upstream stream failed"}
    assert [item["type"] for item in failed["output"]] == ["message", "function_call"]
    errors = _event(frames, "error")
    assert len(errors) == 1
    assert errors[0]["message"] == "upstream stream failed"
    assert events.index("response.output_item.done") < events.index("response.failed") < events.index("error")
