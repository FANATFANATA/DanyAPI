from danyapi.deepseek.sse import SSEEvent, parse_sse
from danyapi.deepseek.stream import IncrementalSSE, MessageReconstructor, _init_message, _navigate, _set_path


def first_set_delta():
    return SSEEvent(
        None,
        {
            "o": "SET",
            "p": "response",
            "v": {
                "response": {
                    "message_id": "asst_1",
                    "parent_id": "user_1",
                    "role": "assistant",
                    "thinking_enabled": True,
                    "inserted_at": 1786469543117,
                    "accumulated_token_usage": 0,
                    "search_enabled": False,
                    "search_triggered": None,
                    "status": "WIP",
                    "ban_regenerate": False,
                    "feedback": None,
                    "fragments": [
                        {"type": "THINK", "content": ""},
                        {"type": "RESPONSE", "content": ""},
                    ],
                    "has_pending_fragment": False,
                    "conversation_mode": "DEFAULT",
                }
            },
        },
    )


def test_parse_sse():
    raw = (
        "event: ready\n"
        'data: {"response_message_id":"a","request_message_id":"b","model_type":"default"}\n'
        "\n"
        'data: {"o":"BATCH","v":[{"o":"SET","p":"response/fragments/0/content","v":"hi"}]}\n'
        "\n"
        "event: finish\n"
        "data: {}\n"
        "\n"
    )
    events = parse_sse(raw)
    assert len(events) == 3
    assert events[0].event == "ready"
    assert events[0].data["response_message_id"] == "a"
    assert events[1].event is None
    assert events[1].data["o"] == "BATCH"
    assert events[2].event == "finish"


def test_parse_sse_non_json_data():
    events = parse_sse("event: ping\ndata: hello world\n\n")
    assert len(events) == 1
    assert events[0].event == "ping"
    assert events[0].data == "hello world"


def test_parse_sse_multiple_data_lines():
    events = parse_sse('data: {"a":1}\ndata: {"b":2}\n\n')
    assert events[0].data == '{"a":1}\n{"b":2}'


def test_incremental_split():
    raw = 'event: ready\ndata: {"a":1}\n\ndata: {"o":"SET","p":"response/x","v":1}\n\n'
    inc = IncrementalSSE()
    collected = []
    mid = len(raw) // 2
    collected.extend(inc.feed(raw[:mid].encode()))
    collected.extend(inc.feed(raw[mid:].encode()))
    collected.extend(inc.finish())
    assert len(collected) == 2
    assert collected[0].event == "ready"


def test_crlf_separators():
    raw = 'event: ready\r\ndata: {"a":1}\r\n\r\ndata: {"o":"SET","p":"response/x","v":1}\r\n\r\n'
    inc = IncrementalSSE()
    collected = list(inc.feed(raw.encode())) + list(inc.finish())
    assert len(collected) == 2
    assert collected[0].event == "ready"
    assert collected[1].data["v"] == 1


def test_finish_with_leftover():
    inc = IncrementalSSE()
    inc._buffer = b'data: {"x": 1}\n\n'
    events = list(inc.finish())
    assert len(events) == 1
    assert events[0].data == {"x": 1}


def test_real_stream_pattern():
    rec = MessageReconstructor()
    rec.handle(
        SSEEvent(
            "ready",
            {
                "request_message_id": 1,
                "response_message_id": 2,
                "model_type": "default",
            },
        )
    )
    assert rec.response_message_id == 2
    rec.handle(
        SSEEvent(
            None,
            {
                "v": {
                    "response": {
                        "message_id": 2,
                        "parent_id": 1,
                        "role": "ASSISTANT",
                        "status": "WIP",
                        "fragments": [{"id": 2, "type": "RESPONSE", "content": "Г"}],
                    }
                }
            },
        )
    )
    assert rec.message["id"] == 2
    rec.handle(SSEEvent(None, {"p": "response/fragments/-1/content", "o": "APPEND", "v": "рави"}))
    for token in ("тация", " -", " это"):
        rec.handle(SSEEvent(None, {"v": token}))
    assert rec.content == "Гравитация - это"
    rec.handle(
        SSEEvent(
            None,
            {
                "p": "response",
                "o": "BATCH",
                "v": [
                    {"p": "accumulated_token_usage", "v": 103},
                    {"p": "quasi_status", "v": "FINISHED"},
                ],
            },
        )
    )
    rec.handle(SSEEvent(None, {"p": "response/status", "o": "SET", "v": "FINISHED"}))
    assert rec.status == "FINISHED"
    assert rec.message.get("accumulated_token_usage") == 103


def test_thinking_then_response_fragment():
    rec = MessageReconstructor()
    rec.handle(
        SSEEvent(
            None,
            {
                "v": {
                    "response": {
                        "message_id": 2,
                        "parent_id": 1,
                        "status": "WIP",
                        "fragments": [{"id": 2, "type": "THINK", "content": "Мы"}],
                    }
                }
            },
        )
    )
    rec.handle(SSEEvent(None, {"p": "response/fragments/-1/elapsed_secs", "o": "SET", "v": 3.5}))
    rec.handle(SSEEvent(None, {"p": "response/fragments", "o": "APPEND", "v": [{"id": 3, "type": "RESPONSE", "content": "Если"}]}))
    rec.handle(SSEEvent(None, {"p": "response/fragments/-1/content", "v": " вз"}))
    for token in ("ять", " четыре"):
        rec.handle(SSEEvent(None, {"v": token}))
    assert rec.reasoning == "Мы"
    assert rec.content == "Если взять четыре"


def test_hint_error_capture():
    rec = MessageReconstructor()
    rec.handle(
        SSEEvent(
            "ready",
            {
                "request_message_id": 1,
                "response_message_id": 2,
                "model_type": "expert",
            },
        )
    )
    rec.handle(
        SSEEvent(
            "hint",
            {
                "type": "error",
                "content": "Server is busy. Try again later, or use Instant Mode.",
                "clear_response": True,
                "finish_reason": "expert_busy_use_default",
            },
        )
    )
    assert rec.hint_error is not None
    assert rec.hint_error["finish_reason"] == "expert_busy_use_default"
    assert "busy" in rec.hint_error["message"].lower()
    rec2 = MessageReconstructor()
    rec2.handle(SSEEvent("hint", {"type": "info", "content": "ok", "finish_reason": None}))
    assert rec2.hint_error is None


def test_apply_deltas():
    rec = MessageReconstructor()
    rec.handle(first_set_delta())
    assert rec.message["id"] == "asst_1"
    rec.handle(
        SSEEvent(
            None,
            {
                "o": "BATCH",
                "v": [
                    {"o": "SET", "p": "response/fragments/0/content", "v": "Let me think step by step"},
                    {"o": "SET", "p": "response/fragments/1/content", "v": "Hello"},
                ],
            },
        )
    )
    assert rec.reasoning == "Let me think step by step"
    assert rec.content == "Hello"
    c_diff, r_diff = rec.take_diffs()
    assert c_diff == "Hello"
    assert r_diff == "Let me think step by step"
    rec.handle(
        SSEEvent(
            None,
            {
                "o": "BATCH",
                "v": [
                    {"o": "SET", "p": "response/fragments/1/content", "v": "Hello world"},
                    {"o": "SET", "p": "response/status", "v": "FINISHED"},
                ],
            },
        )
    )
    c_diff, r_diff = rec.take_diffs()
    assert c_diff == " world"
    assert rec.content == "Hello world"
    assert rec.status == "FINISHED"


def test_usage_from_accumulated_token_usage():
    rec = MessageReconstructor()
    rec.handle(first_set_delta())
    assert rec.usage == {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    rec.handle(SSEEvent(None, {"o": "BATCH", "v": [{"p": "accumulated_token_usage", "v": 103}]}))
    assert rec.usage == {"prompt_tokens": 0, "completion_tokens": 103, "total_tokens": 103}


def test_usage_ignores_invalid_values():
    rec = MessageReconstructor()
    rec.handle(SSEEvent(None, {"o": "BATCH", "v": [{"p": "accumulated_token_usage", "v": -5}]}))
    assert rec.usage["completion_tokens"] == 0


def test_context_length_status():
    rec = MessageReconstructor()
    rec.handle(SSEEvent(None, {"p": "response/status", "o": "SET", "v": "CONTEXT_LENGTH_EXCEEDED"}))
    assert rec.status == "CONTEXT_LENGTH_EXCEEDED"
    assert rec.content == ""
    assert rec.reasoning == ""


def test_navigate_list_out_of_range():
    assert _navigate({"a": [1]}, ["a", "5"]) is None
    assert _navigate({"a": [1]}, ["a", "-2"]) is None


def test_navigate_negative_index():
    assert _navigate({"a": [1, 2]}, ["a", "-1"]) == 2


def test_navigate_missing_key():
    assert _navigate({"a": 1}, ["b"]) is None


def test_navigate_non_dict():
    assert _navigate({"a": 1}, ["a", "b"]) is None
    assert _navigate(42, ["a"]) is None


def test_navigate_invalid_index():
    assert _navigate({"a": ["x"]}, ["a", "zz"]) is None


def test_set_path_index_out_of_range():
    target = {"a": []}
    _set_path(target, ["a", "0", "b"], 1)
    assert target == {"a": []}


def test_set_path_creates_nested():
    target = {}
    _set_path(target, ["response", "fragments"], [])
    assert target == {"response": {"fragments": []}}


def test_set_path_through_list():
    target = {"a": [{"b": {}}]}
    _set_path(target, ["a", "0", "c"], 5)
    assert target["a"][0]["c"] == 5


def test_set_path_non_dict_node():
    target = {"a": "str"}
    _set_path(target, ["a", "b"], 1)
    assert target["a"] == {"b": 1}


def test_set_path_through_scalar():
    target = {"a": [5]}
    _set_path(target, ["a", "0", "b"], 1)
    assert target == {"a": [5]}


def test_set_path_scalar_mid_path():
    target = {"a": [5]}
    _set_path(target, ["a", "0", "b", "c"], 1)
    assert target == {"a": [5]}


def test_set_path_bad_index():
    target = {"a": [5]}
    _set_path(target, ["a", "zz", "b"], 1)
    assert target == {"a": [5]}


def test_append_list_positive_index():
    rec = MessageReconstructor()
    rec.message = {"items": ["a", "b"]}
    rec.handle(SSEEvent(None, {"p": "response/items/1", "o": "APPEND", "v": "X"}))
    assert rec.message["items"] == ["a", "bX"]


def test_append_list_positive_index_out_of_range():
    rec = MessageReconstructor()
    rec.message = {"items": ["a"]}
    rec.handle(SSEEvent(None, {"p": "response/items/1", "o": "APPEND", "v": "X"}))
    assert rec.message["items"] == ["a"]


def test_append_list_negative_index():
    rec = MessageReconstructor()
    rec.message = {"items": ["a", "b"]}
    rec.handle(SSEEvent(None, {"p": "response/items/-1", "o": "APPEND", "v": "X"}))
    assert rec.message["items"] == ["a", "b"]


def test_init_message_non_dict():
    msg = {"x": 1}
    _init_message(msg, "garbage")
    assert msg == {"x": 1}


def test_batch_with_non_dict_sub():
    rec = MessageReconstructor()
    rec.handle(SSEEvent(None, {"o": "BATCH", "v": [42, "str"]}))
    assert rec.message == {}


def test_append_list_index_out_of_range():
    rec = MessageReconstructor()
    rec.message = {"fragments": [{"content": "x"}]}
    rec.handle(SSEEvent(None, {"p": "response/fragments/9/content", "o": "APPEND", "v": "y"}))
    assert rec.message["fragments"][0]["content"] == "x"


def test_append_negative_index_out_of_range():
    rec = MessageReconstructor()
    rec.message = {"fragments": [{"content": "x"}]}
    rec.handle(SSEEvent(None, {"p": "response/fragments/-5/content", "o": "APPEND", "v": "y"}))
    assert rec.message["fragments"][0]["content"] == "x"


def test_append_missing_key_sets():
    rec = MessageReconstructor()
    rec.message = {"fragments": [{}]}
    rec.handle(SSEEvent(None, {"p": "response/fragments/0/content", "o": "APPEND", "v": "y"}))
    assert rec.message["fragments"][0]["content"] == "y"


def test_append_list_value():
    rec = MessageReconstructor()
    rec.message = {"fragments": []}
    rec.handle(SSEEvent(None, {"p": "response/fragments", "o": "APPEND", "v": [{"id": 1}]}))
    assert rec.message["fragments"] == [{"id": 1}]


def test_append_scalar_to_list():
    rec = MessageReconstructor()
    rec.message = {"ids": [1, 2]}
    rec.handle(SSEEvent(None, {"p": "response/ids", "o": "APPEND", "v": 3}))
    assert rec.message["ids"] == [1, 2, 3]


def test_append_replaces_non_str_value():
    rec = MessageReconstructor()
    rec.message = {"fragments": [{"content": 5}]}
    rec.handle(SSEEvent(None, {"p": "response/fragments/0/content", "o": "APPEND", "v": "y"}))
    assert rec.message["fragments"][0]["content"] == "y"


def test_path_ignored_when_not_response():
    rec = MessageReconstructor()
    rec.handle(SSEEvent(None, {"p": "other/x", "o": "SET", "v": 1}))
    assert rec.message == {}


def test_fragment_text_variants():
    from danyapi.deepseek.stream import _fragment_text

    assert _fragment_text("plain") == "plain"
    assert _fragment_text(42) == ""
    assert _fragment_text({"content": "x"}) == "x"
    assert _fragment_text({"content": ["a", {"text": "b"}, 3]}) == "ab"
    assert _fragment_text({"content": []}) == ""
    assert _fragment_text({"content": 5}) == ""
    assert _fragment_text({}) == ""


def test_delta_without_v_ignored():
    rec = MessageReconstructor()
    rec.handle(SSEEvent(None, {"o": "SET", "p": "response/x"}))
    assert rec.message == {}


def test_delta_non_dict_data():
    rec = MessageReconstructor()
    rec.handle(SSEEvent(None, "garbage"))
    assert rec.message == {}


def test_toast_non_dict():
    rec = MessageReconstructor()
    rec.handle(SSEEvent("toast", "just text"))
    assert rec.hint_error is None


def test_toast_info_not_error():
    rec = MessageReconstructor()
    rec.handle(SSEEvent("hint", {"type": "info", "content": "ok"}))
    assert rec.hint_error is None


def test_ready_non_dict():
    rec = MessageReconstructor()
    rec.handle(SSEEvent("ready", "garbage"))
    assert rec.response_message_id is None


def test_delta_ignored_event():
    rec = MessageReconstructor()
    rec.handle(SSEEvent("close", {"v": 1}))
    assert rec.message == {}


def test_extend_with_other_fragments_list():
    rec = MessageReconstructor()
    rec.message = {"fragments": [{"type": "RESPONSE", "content": "a"}]}
    other = MessageReconstructor()
    other.message = {"fragments": [{"type": "RESPONSE", "content": "b"}], "id": "m2", "status": "FINISHED", "accumulated_token_usage": 5}
    rec.extend_with(other)
    assert rec.content == "ab"
    assert rec.id == "m2"
    assert rec.status == "FINISHED"
    assert rec.accumulated_tokens == 5


def test_extend_with_no_fragments():
    rec = MessageReconstructor()
    other = MessageReconstructor()
    other.message = {}
    rec.extend_with(other)
    assert rec.message == {}


def test_extend_with_own_no_fragments():
    rec = MessageReconstructor()
    rec.message = {}
    other = MessageReconstructor()
    other.message = {"fragments": [{"type": "RESPONSE", "content": "b"}]}
    rec.extend_with(other)
    assert rec.content == "b"


def test_extend_with_resets_hint():
    rec = MessageReconstructor()
    rec.hint_error = {"message": "old"}
    other = MessageReconstructor()
    other.hint_error = {"message": "new", "finish_reason": "busy"}
    rec.extend_with(other)
    assert rec.hint_error == {"message": "new", "finish_reason": "busy"}


def test_accumulated_tokens_invalid():
    rec = MessageReconstructor()
    rec.message = {"accumulated_token_usage": "not a number"}
    assert rec.accumulated_tokens == 0


def test_reasoning_and_content_types():
    rec = MessageReconstructor()
    rec.message = {"fragments": [{"type": "THINK", "content": "why"}, {"type": "RESPONSE", "content": "because"}]}
    assert rec.reasoning == "why"
    assert rec.content == "because"


def test_parse_sse_ignores_non_event_lines():
    events = parse_sse(": comment\nignored line\ndata: {}\n\n")
    assert len(events) == 1
    assert events[0].data == {}


def test_set_path_empty_parts():
    target = {"a": 1}
    _set_path(target, [], 99)
    assert target == {"a": 1}


def test_apply_delta_unknown_op():
    rec = MessageReconstructor()
    rec.handle(SSEEvent(None, {"o": "OTHER", "p": "response/x", "v": 1}))
    assert rec.message == {}


def test_apply_delta_append_to_response_root():
    rec = MessageReconstructor()
    rec.handle(SSEEvent(None, {"o": "APPEND", "p": "response", "v": 1}))
    assert rec.message == {}


def test_append_non_str_in_list():
    rec = MessageReconstructor()
    rec.message = {"items": [5]}
    rec.handle(SSEEvent(None, {"p": "response/items/0", "o": "APPEND", "v": "x"}))
    assert rec.message["items"] == [5]
