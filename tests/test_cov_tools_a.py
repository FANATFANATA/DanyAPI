import json
from typing import Any

import pytest

from danyapi import tools as toolemu
from danyapi.tools import (
    CHOICE_INSTRUCTIONS,
    DsmlFilter,
    ToolCall,
    _argument_summary,
    _array_hold,
    _boundary_hold,
    _choice_name,
    _coerce_scalar,
    _dsml_scan_cut,
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
    fix_tool_calls,
    parse_tool_calls,
    render_tool_schema,
    strip_dsml,
    tool_call_boundary,
    tool_schema_detail,
    tool_schema_map,
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


_DSML_TAG = "\uff5c\uff5c"


def test_dsml_filter_streams_cjk_after_angle_bracket():
    text = "a<\u4e2d more text"
    flt = DsmlFilter()
    chunks = [flt.feed(text[index : index + 3]) for index in range(0, len(text), 3)]
    chunks.append(flt.flush())
    assert "".join(chunks) == text
    assert "".join(chunks[:2]) != ""


def test_dsml_filter_holds_partial_dsml_marker():
    flt = DsmlFilter()
    assert flt.feed("<\uff5c") == ""
    assert flt.feed("DS") == ""
    assert flt.flush() == " "


def test_dsml_scan_cut_releases_cjk_after_angle_bracket():
    text = "a<\u4e2d more text"
    assert _dsml_scan_cut(text, False) == len(text)


def test_dsml_scan_cut_holds_partial_marker_name():
    assert _dsml_scan_cut("<\uff5cDS", False) == 0


def test_dsml_filter_holds_dead_tag_that_is_still_a_dangling_tail():
    text = "<<\uff5cDplain text "
    flt = DsmlFilter()
    chunks = [flt.feed(text[index : index + 3]) for index in range(0, len(text), 3)]
    chunks.append(flt.flush())
    assert "".join(chunks) == strip_dsml(text) == "< "


def test_dsml_filter_releases_angle_bracket_without_a_pipe_run():
    flt = DsmlFilter()
    assert flt.feed("<b") == "<b"
    assert flt.feed("more") == "more"
    assert flt.flush() == ""


def test_dsml_filter_holds_pipe_tail_until_flush():
    flt = DsmlFilter()
    assert flt.feed("<\uff5cDplain") == ""
    assert flt.feed(" 2") == ""
    assert flt.flush() == strip_dsml("<\uff5cDplain 2")


def test_dsml_filter_releases_a_buffer_that_outgrows_the_cap():
    import danyapi.tools.dsml as dsml_mod

    original = dsml_mod.MAX_BUFFER_CHARS
    dsml_mod.MAX_BUFFER_CHARS = 16
    try:
        flt = DsmlFilter()
        emitted = "".join(flt.feed("<|") for _ in range(64))
        assert emitted
        assert len(flt._buf) <= 16
    finally:
        dsml_mod.MAX_BUFFER_CHARS = original


def test_strip_dsml_removes_interrupted_tag_with_attributes():
    assert strip_dsml('ok <\uff5cDSML\uff5cparameter name="city"') == "ok  "


def test_strip_dsml_keeps_ordinary_markup():
    assert strip_dsml('<a href="x">link</a>') == '<a href="x">link</a>'
    assert strip_dsml("1 < 2 and 3 > 2") == "1 < 2 and 3 > 2"
    assert strip_dsml("ok text \uff5c\uff5cDSML\uff5c") == "ok text  "


def test_strip_dsml_dangling_no_catastrophic_backtracking():
    import time

    started = time.monotonic()
    strip_dsml("\uff5c" * 40)
    strip_dsml("<" + "\uff5c" * 40 + "DSML")
    strip_dsml("x" * 5000 + "\uff5c" * 40)
    assert time.monotonic() - started < 2.0


def test_dsml_close_cache_cleared_at_cap():
    toolemu._DSML_CLOSE_CACHE.clear()
    for index in range(toolemu._DSML_CLOSE_CACHE_MAX + 5):
        toolemu._dsml_close_pattern(f"tag{index}")
    assert len(toolemu._DSML_CLOSE_CACHE) <= toolemu._DSML_CLOSE_CACHE_MAX
    assert toolemu._dsml_close_pattern("x") is not None


def test_fix_tool_calls_full_resolves_types_from_schemas():
    schemas = {"add": {"a": "integer", "b": "string"}}
    calls = [ToolCall("c1", "add", '{"a": 1, "b": "x"}')]
    assert fix_tool_calls(list(calls), schemas, None, "full")[0].arguments == '{"a": 1, "b": "x"}'


def test_fix_tool_calls_full_keeps_coercion_from_schemas():
    schemas = {"add": {"a": "integer"}}
    calls = [ToolCall("c1", "add", '{"a": "12"}')]
    assert fix_tool_calls(list(calls), schemas, None, "full")[0].arguments == '{"a": 12}'


def test_fix_tool_calls_full_reports_unknown_tool_from_schemas():
    report: dict[str, Any] = {}
    calls = [ToolCall("c1", "nope", '{"a": 1}')]
    result = fix_tool_calls(list(calls), {"add": {"a": "integer"}}, None, "full", report)
    assert result[0].arguments == '{"a": 1}'
    assert report["warnings"] == [{"call_id": "c1", "kind": "unknown_tool", "name": "nope"}]


def test_parse_tool_calls_non_finite_number_stays_string():
    schemas = {"f": {"x": "number"}}
    details = {"f": {"types": {"x": "number"}}}
    parsed = parse_tool_calls('<invoke name="f"><parameter name="x">1e400</parameter></invoke>', schemas, details, "full")
    assert parsed is not None
    calls, _ = parsed
    assert json.loads(calls[0].arguments) == {"x": "1e400"}


def test_coerce_scalar_rejects_infinities():
    assert _coerce_scalar("1e400", "number") == "1e400"
    assert _coerce_scalar("-1e400", "integer") == "-1e400"
    assert _coerce_scalar("inf", "number") == "inf"


def test_yaml_value_rejects_infinities():
    assert toolemu._yaml_value("1e400") == "1e400"
    assert toolemu._yaml_value("-Infinity") == "-Infinity"
    assert toolemu._yaml_value("1.5") == 1.5


def test_parse_tool_calls_duplicate_xml_parameters_fold_into_list():
    parsed = parse_tool_calls('<invoke name="add"><parameter name="tag">a</parameter><parameter name="tag">b</parameter></invoke>')
    assert parsed is not None
    calls, _ = parsed
    assert json.loads(calls[0].arguments) == {"tag": ["a", "b"]}


def test_parse_tool_calls_duplicate_dsml_parameters_fold_into_list():
    mark = _DSML_TAG
    text = (
        f"<{mark}DSML{mark}tool_calls>"
        f'<{mark}DSML{mark}invoke name="add">'
        f'<{mark}DSML{mark}parameter name="tag">a</{mark}DSML{mark}parameter>'
        f'<{mark}DSML{mark}parameter name="tag">b</{mark}DSML{mark}parameter>'
        f"</{mark}DSML{mark}invoke>"
        f"</{mark}DSML{mark}tool_calls>"
    )
    parsed = parse_tool_calls(text)
    assert parsed is not None
    calls, _ = parsed
    assert json.loads(calls[0].arguments) == {"tag": ["a", "b"]}


def test_parse_tool_calls_duplicate_lax_dsml_parameters_fold_into_list():
    mark = _DSML_TAG
    text = (
        f'<{mark}DSML{mark}invoke name="add">'
        f'<{mark}DSML{mark}parameter name="tag">a</{mark}DSML{mark}parameter>'
        f'<{mark}DSML{mark}parameter name="tag">b</{mark}DSML{mark}parameter>'
        f"</{mark}DSML{mark}invoke>"
    )
    parsed = parse_tool_calls(text)
    assert parsed is not None
    calls, _ = parsed
    assert json.loads(calls[0].arguments) == {"tag": ["a", "b"]}


def test_parse_tool_calls_unquoted_attributes():
    parsed = parse_tool_calls("<invoke name=f param=1 flag=true />")
    assert parsed is not None
    calls, _ = parsed
    assert calls[0].name == "f"
    assert json.loads(calls[0].arguments) == {"param": "1", "flag": "true"}


def test_parse_tool_calls_quoted_attributes_unchanged():
    parsed = parse_tool_calls('<invoke name="f" param="1" flag="true" />')
    assert parsed is not None
    calls, _ = parsed
    assert json.loads(calls[0].arguments) == {"param": "1", "flag": "true"}


def test_parse_tool_calls_nested_dsml_parameter_is_object():
    mark = _DSML_TAG
    text = (
        f"<{mark}DSML{mark}tool_calls>"
        f'<{mark}DSML{mark}invoke name="f">'
        f'<{mark}DSML{mark}parameter name="cfg">'
        f"<{mark}DSML{mark}city>Moscow</{mark}DSML{mark}city>"
        f"</{mark}DSML{mark}parameter>"
        f"</{mark}DSML{mark}invoke>"
        f"</{mark}DSML{mark}tool_calls>"
    )
    parsed = parse_tool_calls(text, {"f": {"cfg": "object"}})
    assert parsed is not None
    calls, _ = parsed
    assert json.loads(calls[0].arguments) == {"cfg": {"city": "Moscow"}}


def test_parse_dsml_lax_wrapper_drops_tool_close_tags():
    mark = _DSML_TAG
    text = f'lead <{mark}DSML{mark}invoke name="add">x</{mark}DSML{mark}invoke> tail'
    parsed = parse_tool_calls(text, {"add": {"a": "integer"}})
    assert parsed is not None
    calls, wrapper = parsed
    assert calls[0].name == "add"
    assert wrapper == "lead x tail"


def test_render_tool_schema_tool_choice_any_is_required():
    rendered = render_tool_schema([WEATHER_TOOL], "any")
    assert rendered is not None
    assert rendered.rstrip().endswith(CHOICE_INSTRUCTIONS["required"])


def test_render_tool_schema_tool_choice_auto_unchanged():
    rendered = render_tool_schema([WEATHER_TOOL], "auto")
    assert rendered is not None
    assert "You MUST call" not in rendered


def test_schema_helpers_agree_with_public_functions():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "f",
                "parameters": json.dumps(
                    {"properties": {"a": {"type": ["null", "integer"]}, "b": {"type": ["string"]}}},
                ),
                "aliases": ["ff"],
            },
        }
    ]
    assert tool_schema_map(tools) == {"f": {"a": "integer", "b": "string", "_aliases": ["ff"]}}
    detail = tool_schema_detail(tools)
    assert detail["f"]["types"] == {"a": "integer", "b": "string"}
    assert detail["f"]["name_aliases"] == ["ff"]


def test_schema_helpers_tolerate_broken_parameter_json():
    tools = [{"type": "function", "function": {"name": "f", "parameters": "{not json"}}]
    assert tool_schema_map(tools) == {"f": {}}
    assert tool_schema_detail(tools)["f"]["types"] == {}


def test_resolved_prop_type_passthrough():
    assert toolemu._resolved_prop_type({"type": "object"}) == "object"
    assert toolemu._resolved_prop_type({"type": []}) == []
    assert toolemu._resolved_prop_type("nope") is None
    assert toolemu._schema_params({"parameters": 5}) == 5
