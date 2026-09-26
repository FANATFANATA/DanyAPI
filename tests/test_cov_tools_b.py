import json

from danyapi import tools as toolemu
from danyapi.tools import (
    ToolCall,
    _alias_rows,
    _coerce_by_type,
    _extract_calls,
    _fuzzy_known_name,
    _infer_tool_name_from_schemas,
    _iter_json_objects,
    _match_enum,
    _parse_yaml_calls,
    _resolve_alias,
    _resolve_arg_key,
    _schema_for_name,
    _xml_tag_attrs,
    fix_tool_calls,
    parse_tool_calls,
    parse_tool_calls_debug,
    tool_schema_detail,
    tool_schema_map,
)

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the weather in a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    },
}

RICH_TOOL = {
    "type": "function",
    "function": {
        "name": "do_thing",
        "aliases": ["doit", "perform"],
        "parameters": {
            "type": "object",
            "required": ["mode"],
            "properties": {
                "mode": {"type": ["string", "null"], "enum": ["fast", "slow"], "default": "fast", "aliases": ["m"]},
                "count": {"type": "integer", "minimum": 1, "maximum": 10},
                "ratio": {"type": "number"},
                "flag": {"type": "boolean"},
                "nothing": {"type": "null"},
            },
        },
    },
}


def _weather_schemas():
    return tool_schema_map([WEATHER_TOOL])


def _rich_details():
    return tool_schema_detail([RICH_TOOL])


def test_extract_calls_non_dict():
    assert _extract_calls("nope") is None


def test_schema_for_name_non_dict():
    assert _schema_for_name(["a"], "b") is None


def test_alias_rows_skips_non_string():
    assert _alias_rows((("known", (None, "")),)) == ()


def test_resolve_alias_none_and_empty_key():
    assert _resolve_alias("x", None) is None
    assert _resolve_alias("!!!", {"a": {}}) is None


def test_fuzzy_known_name_short_and_miss():
    assert _fuzzy_known_name("ab", {}) is None
    assert _fuzzy_known_name("zzzz", {"get_weather": {}}) is None


def test_tool_schema_map_skips_non_dict_tool():
    toolemu._tool_schema_map_cache.clear()
    assert tool_schema_map([42, {"function": {"name": "x"}}]) == {"x": {}}


def test_tool_schema_map_cache_eviction():
    toolemu._tool_schema_map_cache.clear()
    for i in range(toolemu._TOOL_SCHEMA_MAP_CACHE_MAX):
        toolemu._tool_schema_map_cache[i] = ((), {})
    assert len(toolemu._tool_schema_map_cache) == toolemu._TOOL_SCHEMA_MAP_CACHE_MAX
    result = tool_schema_map([WEATHER_TOOL])
    assert "get_weather" in result


def test_tool_schema_detail_rich():
    details = tool_schema_detail([RICH_TOOL])
    spec = details["do_thing"]
    assert spec["required"] == ["mode"]
    assert spec["types"]["mode"] == "null"
    assert spec["enums"]["mode"] == ["fast", "slow"]
    assert spec["defaults"]["mode"] == "fast"
    assert spec["bounds"]["count"] == {"minimum": 1, "maximum": 10}
    assert spec["param_aliases"]["mode"] == ["m"]
    assert spec["name_aliases"] == ["doit", "perform"]


def test_tool_schema_detail_ignores_bad_tools():
    assert tool_schema_detail(None) == {}
    assert tool_schema_detail([42, {"function": {"name": ""}}]) == {}


def test_resolve_arg_key_branches():
    known = ("mode", "count")
    assert _resolve_arg_key("mode", known, {}) == ("mode", "exact")
    assert _resolve_arg_key("other", (), {}) == (None, "unknown")
    assert _resolve_arg_key("MODE", known, {}) == ("mode", "casefold")
    assert _resolve_arg_key("m_o_d_e", known, {}) == ("mode", "compact")
    assert _resolve_arg_key("m", known, {"mode": ["m"]}) == ("mode", "alias")
    assert _resolve_arg_key("modw", known, {}) == (None, "unknown")
    assert _resolve_arg_key("zzz", known, {}) == (None, "unknown")


def test_match_enum_branches():
    assert _match_enum("fast", ["fast", "slow"]) == ("fast", "exact")
    assert _match_enum("FAST", ["fast", "slow"]) == ("fast", "casefold")
    assert _match_enum("fastt", ["fast", "slow"]) == ("fast", "fuzzy")
    assert _match_enum("zzz", ["fast", "slow"]) is None


def test_coerce_by_type_branches():
    assert _coerce_by_type(5, None) == (5, False)
    assert _coerce_by_type("x", "string") == ("x", False)
    assert _coerce_by_type({"a": 1}, "string") == ('{"a": 1}', True)
    assert _coerce_by_type(None, "string") == ("", True)
    assert _coerce_by_type(5, "string") == ("5", True)
    assert _coerce_by_type(True, "integer") == (True, False)
    assert _coerce_by_type(2.0, "integer") == (2, True)
    assert _coerce_by_type("3", "integer") == (3, True)
    assert _coerce_by_type("x", "integer") == ("x", False)
    assert _coerce_by_type(True, "boolean") == (True, False)
    assert _coerce_by_type("true", "boolean") == (True, True)
    assert _coerce_by_type("false", "boolean") == (False, True)
    assert _coerce_by_type("nope", "boolean") == ("nope", False)
    assert _coerce_by_type(None, "null") == (None, False)
    assert _coerce_by_type("null", "null") == (None, True)
    assert _coerce_by_type("x", "null") == ("x", False)
    assert _coerce_by_type("x", "object") == ("x", False)


def _rich_call(arguments, call_id="c1", name="do_thing"):
    return ToolCall(call_id, name, json.dumps(arguments))


def test_fix_tool_calls_safe_mode():
    details = _rich_details()
    call = _rich_call({"mode": "FAST", "count": "3", "ratio": "1.5", "flag": "true", "nothing": "null", "extra": "x"})
    report = {}
    result = fix_tool_calls([call], _weather_schemas(), details, "safe", report)
    parsed = json.loads(result[0].arguments)
    assert parsed["mode"] == "fast"
    assert parsed["count"] == 3
    assert parsed["ratio"] == 1.5
    assert parsed["flag"] is True
    assert parsed["nothing"] is None
    assert parsed["extra"] == "x"
    assert any(item["kind"] == "coerce" for item in report["fixes"])
    assert any(item["kind"] == "enum" for item in report["fixes"])
    assert any(item["kind"] == "unknown_param" for item in report["warnings"])


def test_fix_tool_calls_full_mode_drops_unknown():
    details = _rich_details()
    call = _rich_call({"mode": "fast", "extra": "x"})
    result = fix_tool_calls([call], _weather_schemas(), details, "full")
    assert "extra" not in json.loads(result[0].arguments)


def test_fix_tool_calls_report_mode_keeps_arguments():
    details = _rich_details()
    call = _rich_call({"mode": "FAST"})
    result = fix_tool_calls([call], _weather_schemas(), details, "report")
    assert result[0].arguments == call.arguments


def test_fix_tool_calls_invalid_mode_and_defaults_and_required():
    details = _rich_details()
    call = _rich_call({"count": 5})
    report = {}
    result = fix_tool_calls([call], _weather_schemas(), details, "safe", report)
    parsed = json.loads(result[0].arguments)
    assert parsed["mode"] == "fast"
    assert any(item["kind"] == "default" for item in report["fixes"])


def test_fix_tool_calls_bounds_warnings():
    details = _rich_details()
    call = _rich_call({"mode": "fast", "count": 99})
    report = {}
    fix_tool_calls([call], _weather_schemas(), details, "report", report)
    assert "out_of_range" in {item["kind"] for item in report["warnings"]}


def test_fix_tool_calls_enum_mismatch_warning():
    details = _rich_details()
    call = _rich_call({"mode": "zzz"})
    report = {}
    fix_tool_calls([call], _weather_schemas(), details, "report", report)
    assert any(item["kind"] == "enum_mismatch" for item in report["warnings"])


def test_fix_tool_calls_empty_and_non_dict_schemas():
    assert fix_tool_calls([], None, None, "safe") == []
    call = ToolCall("c1", "do_thing", json.dumps({"mode": "fast"}))
    result = fix_tool_calls([call], "notadict", _rich_details(), "safe")
    assert json.loads(result[0].arguments)["mode"] == "fast"


def test_fix_tool_calls_unknown_tool_warning():
    call = ToolCall("c1", "nope", "{}")
    report = {}
    result = fix_tool_calls([call], _weather_schemas(), {}, "report", report)
    assert result[0].name == "nope"
    assert any(item["kind"] == "unknown_tool" for item in report["warnings"])


def test_fix_tool_calls_bad_json_and_non_dict_parsed():
    bad = ToolCall("c1", "do_thing", "{broken")
    non_dict = ToolCall("c2", "do_thing", "[1, 2]")
    details = _rich_details()
    result = fix_tool_calls([bad, non_dict], _weather_schemas(), details, "safe")
    assert result[0].arguments == "{broken"
    assert result[1].arguments == "[1, 2]"


def test_fix_tool_calls_report_accumulates_existing():
    details = _rich_details()
    call = _rich_call({"mode": "FAST"})
    report = {"fixes": [{"kind": "old"}], "warnings": [{"kind": "oldw"}]}
    fix_tool_calls([call], _weather_schemas(), details, "report", report)
    assert report["fixes"][0]["kind"] == "old"
    assert report["warnings"][0]["kind"] == "oldw"


def test_fix_tool_calls_rename_via_alias_and_compact():
    details = _rich_details()
    call = _rich_call({"m": "fast"})
    report = {}
    result = fix_tool_calls([call], _weather_schemas(), details, "safe", report)
    assert json.loads(result[0].arguments)["mode"] == "fast"
    assert any(item["kind"] == "rename" for item in report["fixes"])


def test_xml_tag_attrs_json_type_skip():
    assert _xml_tag_attrs('integer="true" string="false"') == {}
    assert _xml_tag_attrs('city="Moscow"') == {"city": "Moscow"}


def test_iter_json_objects_whitespace():
    objects = list(_iter_json_objects('{  "a": 1}'))
    assert objects and objects[0][0] == {"a": 1}


def test_iter_json_objects_bad_key_start():
    assert list(_iter_json_objects("{,}")) == []


def test_parse_yaml_calls_name_field():
    text = "tool_calls:\n- name: get_weather\n  city: Moscow"
    calls = _parse_yaml_calls(text)
    assert calls is not None
    assert calls[0].name == "get_weather"


def test_parse_yaml_calls_plain_name():
    text = "tool_calls:\n- get_weather\n"
    calls = _parse_yaml_calls(text)
    assert calls is not None
    assert calls[0].name == "get_weather"


def test_parse_yaml_calls_array_inline():
    text = 'tool_calls: [{"name": "get_weather", "arguments": {"city": "A"}}]'
    calls = _parse_yaml_calls(text)
    assert calls is not None
    assert calls[0].name == "get_weather"


def test_parse_yaml_calls_inline_non_array():
    assert _parse_yaml_calls("tool_calls: junk") is None


def test_parse_yaml_calls_non_tool_root():
    assert _parse_yaml_calls("hello: world") is None


def test_infer_tool_name_from_schemas_branches():
    assert _infer_tool_name_from_schemas(set(), {"a": {}}) is None
    assert _infer_tool_name_from_schemas({"x"}, None) is None
    assert _infer_tool_name_from_schemas({"x"}, {"a": "notadict"}) is None
    assert _infer_tool_name_from_schemas({"x"}, {"a": {"_aliases": ["x"]}}) is None
    assert _infer_tool_name_from_schemas({"x", "y"}, {"a": {"x": 1}, "b": {"y": 2}}) is None
    assert _infer_tool_name_from_schemas({"city"}, {"get_weather": {"city": "string"}}) == "get_weather"


def test_parse_tool_calls_fix_mode():
    text = "tool_calls:\n- name: do_thing\n  mode: FAST"
    schemas = tool_schema_map([RICH_TOOL])
    details = tool_schema_detail([RICH_TOOL])
    result = parse_tool_calls(text, schemas, details, "safe")
    assert result is not None
    calls, _ = result
    assert calls[0].name == "do_thing"
    assert json.loads(calls[0].arguments)["mode"] == "fast"


def test_parse_tool_calls_without_fix_mode():
    text = "tool_calls:\n- name: do_thing\n  mode: FAST"
    result = parse_tool_calls(text, tool_schema_map([RICH_TOOL]), tool_schema_detail([RICH_TOOL]))
    assert result is not None
    calls, _ = result
    assert calls[0].name == "do_thing"


def test_parse_tool_calls_debug_fix_report():
    text = "tool_calls:\n- name: do_thing\n  mode: FAST"
    report = parse_tool_calls_debug(text, tool_schema_map([RICH_TOOL]), tool_schema_detail([RICH_TOOL]))
    assert report["parsed"] is True


def test_toolemu_cache_clear():
    toolemu._tool_schema_map_cache.clear()
    assert toolemu._tool_schema_map_cache == {}
