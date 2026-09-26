import json

import pytest

from danyapi import tools as toolemu
from danyapi.tools import (
    ToolCall,
    _argument_summary,
    _array_hold,
    _boundary_hold,
    _choice_name,
    _extract_json_object,
    _find_xml_close,
    _IntervalSet,
    _json_hold,
    _literal_hold,
    _msg_field,
    _scan_xml_pairs,
    _tag_hold,
    _url_like_after,
    build_prompt,
    extract_last_user,
    tool_call_boundary,
)

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the weather in a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}


class Message:
    def __init__(self, role="user", content="", tool_calls=None, tool_call_id=None, name=None):
        self.role = role
        self.content = content
        self.tool_calls = tool_calls
        self.tool_call_id = tool_call_id
        self.name = name


def test_find_xml_close_name_space():
    assert _find_xml_close("</ bash>", "bash", 0, True, False) == (0, 8)


def test_find_xml_close_name_space_no_match():
    assert _find_xml_close("</ bash>", "other", 0, True, False) is None


def test_find_xml_close_tail_space():
    assert _find_xml_close("</bash >", "bash", 0, False, True) == (0, 8)


def test_find_xml_close_tail_space_no_match():
    assert _find_xml_close("</bash >", "bash", 0, False, False) is None


def test_scan_xml_pairs_truncated_close():
    result = list(_scan_xml_pairs('<invoke name="x">body', frozenset({"invoke"})))
    assert len(result) == 1
    assert result[0][2] == "invoke"


def test_interval_set_merges_chain():
    intervals = _IntervalSet()
    intervals.add(0, 10)
    intervals.add(20, 30)
    intervals.add(10, 40)
    assert intervals.contains(0, 40)
    assert not intervals.contains(0, 41)


def test_literal_hold_marker_prefix():
    assert _literal_hold('{"', 0) == 0


def test_json_hold_empty_body():
    assert _json_hold("abc{", 0) == 3


def test_json_hold_key_prefix():
    assert _json_hold('{"tool', 0) == 0


def test_json_hold_closed_brace_rejected():
    assert _json_hold('{"tool"}', 0) == -1


def test_tag_hold_body():
    assert _tag_hold("<inv", 0, ()) == 0


def test_tag_hold_closing_slash():
    assert _tag_hold("</inv", 0, ()) == 0


def test_tag_hold_empty_body():
    assert _tag_hold("<", 0, ()) == 0


def test_tag_hold_pipe_and_non_ascii():
    assert _tag_hold("<|", 0, ()) == 0
    assert _tag_hold("<\u00a6", 0, ()) == 0


def test_tag_hold_no_name():
    assert _tag_hold("<=", 0, ()) == -1


def test_tag_hold_html_tag():
    assert _tag_hold("<div", 0, ()) == -1


def test_tag_hold_stream_tag():
    assert _tag_hold("<inv", 0, ()) == 0


def test_tag_hold_schema_name():
    assert _tag_hold("<myt", 0, ("mytool",)) == 0


def test_tag_hold_name_suffix():
    assert _tag_hold("<name", 0, ()) == 0


def test_tag_hold_name_in_word():
    assert _tag_hold("<myname", 0, ()) == 0


def test_tag_hold_unknown():
    assert _tag_hold("<xyzzy", 0, ()) == -1


def test_array_hold_empty_body():
    assert _array_hold("[  ", 0) == 0


def test_boundary_hold_returns_first_candidate():
    assert _boundary_hold('{"', 0, ()) == 0


def test_tool_call_boundary_cached_hold():
    toolemu._boundary_cache.clear()
    toolemu._boundary_cache[()] = ("hi <tool", 8, True)
    assert tool_call_boundary("hi <tool", 0, None) == (3, False)


def test_boundary_dsml_stream_start():
    toolemu._boundary_cache.clear()
    assert tool_call_boundary("<|DSML|tool_calls>", 0, None) == (0, True)


def test_boundary_marker_only():
    toolemu._boundary_cache.clear()
    assert tool_call_boundary("<toolinvoke", 0, None) == (0, True)


def test_boundary_hold_only():
    toolemu._boundary_cache.clear()
    assert tool_call_boundary("<tool", 0, None) == (0, False)


def test_tool_call_create_edit_non_string_arguments():
    call = ToolCall.create("edit", {"oldString": 5, "newString": 6})
    assert json.loads(call.arguments) == {"oldString": "5", "newString": "6"}


def test_choice_name_none_and_required():
    assert _choice_name({"type": "none"}) == "none"
    assert _choice_name({"type": "required"}) == "required"


def test_choice_name_custom_type():
    assert _choice_name({"type": "custom_tool"}) == "custom_tool"


def test_argument_summary_bad_key_skipped():
    fn = {"parameters": {"properties": {1: {"type": "string"}}, "required": []}}
    assert _argument_summary(fn) is None


def test_argument_summary_required_without_type():
    fn = {"parameters": {"properties": {"x": {}}, "required": ["x"]}}
    assert _argument_summary(fn) == "x (required)"


def test_argument_summary_optional_without_type():
    fn = {"parameters": {"properties": {"y": {}}, "required": []}}
    assert _argument_summary(fn) == "y"


def test_msg_field_dict():
    assert _msg_field({"role": "user"}, "role") == "user"
    assert _msg_field({"role": "user"}, "missing", "d") == "d"


def test_extract_last_user_none_content_skipped():
    with pytest.raises(ValueError):
        extract_last_user([Message(role="user", content=None)])


def test_build_prompt_session_required_choice():
    prompt, tool_mode = build_prompt([Message("user", "hi")], [WEATHER_TOOL], "required", True)
    assert tool_mode
    assert "MUST call" in prompt


def test_url_like_after_empty_fragment():
    assert _url_like_after("abc", 10) is False


def test_url_like_after_double_slash():
    assert _url_like_after("://x", 0) is True


def test_url_like_after_scheme():
    assert _url_like_after("http://x", 0) is True


def test_url_like_after_colon_port():
    assert _url_like_after(":8080", 0) is True


def test_url_like_after_colon_bracket_stop():
    assert _url_like_after(":a[", 0) is False


def test_url_like_after_colon_space_break():
    assert _url_like_after(":a b", 0) is False


def test_url_like_after_colon_long_break():
    assert _url_like_after(":abcdefghijklmnopq", 0) is False


def test_extract_json_object_non_string():
    assert _extract_json_object(42) is None
