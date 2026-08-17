import json
from typing import Any

import pytest

from danyapi.tools import (
    ToolCall,
    _coerce_scalar,
    _content_fingerprint,
    _content_text,
    _extract_calls,
    _extract_one_call,
    _fix_unbalanced_json,
    _has_history,
    _is_jsonish_arguments,
    _is_tool_round_tail,
    _loads_lenient,
    _normalize_single_quotes,
    _parse_bare_array_calls,
    _parse_xml_tool_calls,
    _render_tool_call_mention,
    _strip_dsml,
    _tail_after_last_user,
    _tool_function,
    build_prompt,
    extract_last_user,
    extract_system,
    format_tool_message,
    is_tool_round,
    parse_tool_calls,
    parse_tool_calls_debug,
    render_json_mode,
    render_message,
    render_tool_schema,
    tool_call_deltas,
    tool_names,
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
    def __init__(self, role="user", content: Any = "", tool_calls=None, tool_call_id=None, name=None):
        self.role = role
        self.content = content
        self.tool_calls = tool_calls
        self.tool_call_id = tool_call_id
        self.name = name


def test_render_tool_schema_basic():
    schema = render_tool_schema([WEATHER_TOOL])
    assert schema is not None
    assert schema is not None
    assert "get_weather" in schema
    assert '"city"' in schema
    assert '"tool_calls"' in schema


def test_render_tool_schema_empty_tools():
    assert render_tool_schema([]) is None
    assert render_tool_schema(None) is None


def test_render_tool_schema_tool_choice_none():
    assert render_tool_schema([WEATHER_TOOL], "none") is None


def test_render_tool_schema_tool_choice_required():
    schema = render_tool_schema([WEATHER_TOOL], "required")
    assert schema is not None
    assert schema is not None
    assert "MUST call" in schema


def test_render_tool_schema_tool_choice_function_dict():
    schema = render_tool_schema([WEATHER_TOOL], {"type": "function", "function": {"name": "get_weather"}})
    assert schema is not None
    assert schema is not None
    assert "get_weather" in schema


def test_render_tool_schema_strict_flag_rendered():
    tool = {
        "type": "function",
        "function": {"name": "calc", "strict": True, "parameters": {"type": "object", "properties": {"x": {"type": "number"}}}},
    }
    schema = render_tool_schema([tool])
    assert schema is not None
    assert schema is not None
    assert "strict" in schema


def test_render_tool_schema_compact_parameters_json():
    schema = render_tool_schema([WEATHER_TOOL])
    assert schema is not None
    assert schema is not None
    assert '{"type":"object"' in schema


def test_is_tool_round_plain_user():
    assert not is_tool_round([Message(role="user", content="hi")])


def test_is_tool_round_tool_role():
    assert is_tool_round([Message(role="tool", content="22C", tool_call_id="call_1")])


def test_is_tool_round_function_role():
    assert is_tool_round([Message(role="function", content="42", name="calc")])


def test_is_tool_round_assistant_tool_calls():
    msg = Message(
        role="assistant",
        tool_calls=[{"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "{}"}}],
    )
    assert is_tool_round([msg])


def test_is_tool_round_assistant_content_list_tool_call():
    msg = Message(
        role="assistant",
        content=[{"type": "tool_call", "id": "c1", "function": {"name": "bash", "arguments": '{"command": "ls"}'}}],
    )
    assert is_tool_round([msg])


def test_extract_last_user_last_user():
    messages = [
        Message(role="system", content="sys"),
        Message(role="user", content="q1"),
        Message(role="assistant", content="a1"),
        Message(role="user", content="q2"),
    ]
    assert extract_last_user(messages) == "q2"


def test_extract_last_user_list_content():
    msg = Message(role="user", content=[{"type": "text", "text": "look"}, {"type": "image_url", "image_url": {"url": "data:image/png;base64,xx"}}])
    assert extract_last_user([msg]) == "look"


def test_extract_last_user_no_user():
    with pytest.raises(ValueError):
        extract_last_user([Message(role="assistant", content="a")])


def test_extract_last_user_empty():
    with pytest.raises(ValueError):
        extract_last_user([])


def test_parse_tool_calls_pure_json():
    text = '{"tool_calls": [{"name": "get_weather", "arguments": {"city": "Moscow"}}]}'
    parsed = parse_tool_calls(text)
    assert parsed is not None
    assert parsed is not None
    calls, wrapper = parsed
    assert calls is not None
    assert len(calls) == 1
    assert calls[0].name == "get_weather"
    assert calls[0].arguments == '{"city": "Moscow"}'
    assert calls[0].id.startswith("call_")
    assert wrapper == ""


def test_parse_tool_calls_markdown_fences():
    text = '```json\n{"tool_calls": [{"name": "get_weather", "arguments": {"city": "London"}}]}\n```'
    parsed = parse_tool_calls(text)
    assert parsed is not None
    assert parsed is not None
    calls, _ = parsed
    assert calls is not None
    assert calls[0].name == "get_weather"


def test_parse_tool_calls_prose_around():
    text = 'I will help you.\n\n{"tool_calls": [{"name": "get_weather", "arguments": {"city": "Rome"}}]}\nHope that helps.'
    parsed = parse_tool_calls(text)
    assert parsed is not None
    assert parsed is not None
    calls, wrapper = parsed
    assert calls is not None
    assert calls[0].name == "get_weather"
    assert "I will help you" in wrapper


def test_parse_tool_calls_legacy_function_call():
    text = '{"function_call": {"name": "get_weather", "arguments": {"city": "Paris"}}}'
    parsed = parse_tool_calls(text)
    assert parsed is not None
    assert parsed is not None
    calls, _ = parsed
    assert calls is not None
    assert calls[0].name == "get_weather"


def test_parse_tool_calls_multiple_calls():
    text = '{"tool_calls": [{"name": "a", "arguments": {"x": 1}}, {"name": "b", "arguments": {"y": 2}}]}'
    parsed = parse_tool_calls(text)
    assert parsed is not None
    assert parsed is not None
    calls, _ = parsed
    assert calls is not None
    assert [c.name for c in calls] == ["a", "b"]


def test_parse_tool_calls_content_with_calls():
    text = '{"content": "checking", "tool_calls": [{"name": "get_weather", "arguments": {"city": "Kyiv"}}]}'
    parsed = parse_tool_calls(text)
    assert parsed is not None
    assert parsed is not None
    calls, wrapper = parsed
    assert calls is not None
    assert wrapper == "checking"


def test_parse_tool_calls_arguments_as_string():
    text = '{"tool_calls": [{"name": "get_weather", "arguments": "{\\"city\\": \\"Oslo\\"}"}]}'
    parsed = parse_tool_calls(text)
    assert parsed is not None
    assert parsed is not None
    calls, _ = parsed
    assert calls is not None
    assert calls[0].arguments == '{"city": "Oslo"}'


def test_parse_tool_calls_not_a_tool_call():
    assert parse_tool_calls("Just a normal answer.") is None
    assert parse_tool_calls('{"answer": 42}') is None
    assert parse_tool_calls("") is None
    assert parse_tool_calls("   ") is None


def test_parse_tool_calls_trailing_comma():
    text = '{"tool_calls": [{"name": "f", "arguments": {"x": 1},}]}'
    parsed = parse_tool_calls(text)
    assert parsed is not None
    assert parsed is not None
    calls, _ = parsed
    assert calls is not None
    assert calls[0].name == "f"
    assert json.loads(calls[0].arguments) == {"x": 1}


def test_parse_tool_calls_single_quotes():
    text = '{"tool_calls": [{"name": "f", "arguments": {"x": "it\'s"}}]}'
    parsed = parse_tool_calls(text)
    assert parsed is not None
    assert parsed is not None
    calls, _ = parsed
    assert calls is not None
    assert calls[0].name == "f"
    assert json.loads(calls[0].arguments) == {"x": "it's"}


def test_parse_tool_calls_single_quotes_with_double_quotes_inside():
    text = "{'tool_calls': [{'name': 'f', 'arguments': {'x': 'say \"hi\"'}}]}"
    parsed = parse_tool_calls(text)
    assert parsed is not None
    assert parsed is not None
    calls, _ = parsed
    assert calls is not None
    assert json.loads(calls[0].arguments) == {"x": 'say "hi"'}


def test_parse_tool_calls_single_quotes_with_backslashes_inside():
    text = r"{'tool_calls': [{'name': 'f', 'arguments': {'path': 'C:\\Windows'}}]}"
    parsed = parse_tool_calls(text)
    assert parsed is not None
    assert parsed is not None
    calls, _ = parsed
    assert calls is not None
    assert json.loads(calls[0].arguments) == {"path": r"C:\Windows"}


def test_parse_tool_calls_truncated_json_not_misparsed():
    text = '{"tool_calls": [{"name": "f", "arguments": {"command": "ls"}}'
    parsed = parse_tool_calls(text)
    assert parsed is None


def test_parse_tool_calls_truncated_missing_argument_key_not_accepted():
    text = '{"name": "f", "arguments": {"command": "ls"}'
    parsed = parse_tool_calls(text)
    assert parsed is None


def test_parse_tool_calls_bare_dict_trailing_comma():
    text = '{"name": "f", "arguments": {"x": 1,}}'
    parsed = parse_tool_calls(text)
    assert parsed is not None
    assert parsed is not None
    calls, _ = parsed
    assert calls is not None
    assert json.loads(calls[0].arguments) == {"x": 1}


def test_parse_xml_tool_calls_bash_invoke():
    text = '<tool_calls>\n<invoke name="bash">\n<command>Get-ChildItem -Name</command>\n</invoke>\n</tool_calls>'
    parsed = parse_tool_calls(text)
    assert parsed is not None
    assert parsed is not None
    calls, wrapper = parsed
    assert calls is not None
    assert len(calls) == 1
    assert calls[0].name == "bash"
    assert json.loads(calls[0].arguments) == {"command": "Get-ChildItem -Name"}
    assert wrapper == ""


def test_parse_xml_tool_calls_multiple_invokes():
    text = '<tool_calls><invoke name="a"><x>1</x></invoke><invoke name="b"><y>2</y></invoke></tool_calls>'
    parsed = parse_tool_calls(text)
    assert parsed is not None
    calls, _ = parsed
    assert calls is not None
    assert [c.name for c in calls] == ["a", "b"]
    assert json.loads(calls[0].arguments) == {"x": "1"}
    assert json.loads(calls[1].arguments) == {"y": "2"}


def test_parse_xml_tool_calls_parameter_tag():
    text = '<tool_calls><invoke name="get_weather"><parameter name="city">Moscow</parameter></invoke></tool_calls>'
    parsed = parse_tool_calls(text)
    assert parsed is not None
    calls, _ = parsed
    assert calls is not None
    assert json.loads(calls[0].arguments) == {"city": "Moscow"}


def test_parse_xml_tool_calls_xml_entities():
    text = '<tool_calls><invoke name="bash"><command>echo &quot;a&quot; &amp; b</command></invoke></tool_calls>'
    parsed = parse_tool_calls(text)
    assert parsed is not None
    calls, _ = parsed
    assert calls is not None
    assert json.loads(calls[0].arguments) == {"command": 'echo "a" & b'}


def test_parse_xml_tool_calls_prose_wrapper():
    text = 'Let me check.\n\n<tool_calls><invoke name="bash"><command>ls</command></invoke></tool_calls>'
    parsed = parse_tool_calls(text)
    assert parsed is not None
    calls, wrapper = parsed
    assert calls is not None
    assert calls[0].name == "bash"
    assert wrapper == "Let me check."


def test_parse_xml_tool_calls_unquoted_name():
    text = "<tool_calls><invoke name=bash><command>pwd</command></invoke></tool_calls>"
    parsed = parse_tool_calls(text)
    assert parsed is not None
    calls, _ = parsed
    assert calls is not None
    assert calls[0].name == "bash"


def test_parse_xml_tool_calls_plain_text_invoke():
    text = '<tool_calls><invoke name="bash">Get-ChildItem</invoke></tool_calls>'
    parsed = parse_tool_calls(text)
    assert parsed is not None
    calls, _ = parsed
    assert calls is not None
    assert json.loads(calls[0].arguments) == {"content": "Get-ChildItem"}


def test_parse_xml_tool_calls_fenced_xml():
    text = '```\n<tool_calls>\n<invoke name="bash">\n<command>dir</command>\n</invoke>\n</tool_calls>\n```'
    parsed = parse_tool_calls(text)
    assert parsed is not None
    calls, _ = parsed
    assert calls is not None
    assert calls[0].name == "bash"


def test_parse_xml_tool_calls_invoke_inside_tool_call_not_duplicated():
    text = '<tool_call><invoke name="bash"><command>ls</command></invoke></tool_call>'
    parsed = parse_tool_calls(text)
    assert parsed is not None
    calls, _ = parsed
    assert calls is not None
    assert [c.name for c in calls] == ["bash"]


def test_parse_bare_array_calls_bare_array():
    text = '[{"name": "bash", "arguments": {"command": "ls"}}, {"name": "get_weather", "arguments": {"city": "Moscow"}}]'
    parsed = parse_tool_calls(text)
    assert parsed is not None
    calls, _ = parsed
    assert calls is not None
    assert [c.name for c in calls] == ["bash", "get_weather"]
    assert json.loads(calls[1].arguments) == {"city": "Moscow"}


def test_parse_bare_array_calls_fenced_array():
    text = '```json\n[{"name": "bash", "arguments": {"command": "pwd"}}]\n```'
    parsed = parse_tool_calls(text)
    assert parsed is not None
    calls, _ = parsed
    assert calls is not None
    assert calls[0].name == "bash"


def test_parse_bare_dict_call_bare_name_arguments():
    text = '{"name": "bash", "arguments": {"command": "Get-ChildItem -Name"}}'
    parsed = parse_tool_calls(text)
    assert parsed is not None
    calls, _ = parsed
    assert calls is not None
    assert calls[0].name == "bash"
    assert json.loads(calls[0].arguments) == {"command": "Get-ChildItem -Name"}


def test_parse_bare_dict_call_many_tools_prose():
    text = (
        "I need to look at the code first.\n\n"
        '{"name": "bash", "arguments": {"command": "git status"}}\n\n'
        'Then I will read the file: {"name": "read", "arguments": {"filePath": "src/main.py"}}\n\n'
        "After that I will fix it."
    )
    parsed = parse_tool_calls(text)
    assert parsed is not None
    calls, wrapper = parsed
    assert calls is not None
    assert calls[0].name == "bash"
    assert "git status" in json.loads(calls[0].arguments)["command"]
    assert calls[1].name == "read"
    assert json.loads(calls[1].arguments)["filePath"] == "src/main.py"
    assert "I need to look at the code first" in wrapper


def test_parse_json_in_xml_json_inside_invoke():
    text = '<tool_calls><invoke name="bash">\n{"command": "Get-ChildItem -Name"}\n</invoke></tool_calls>'
    parsed = parse_tool_calls(text)
    assert parsed is not None
    calls, _ = parsed
    assert calls is not None
    assert calls[0].name == "bash"
    assert json.loads(calls[0].arguments) == {"command": "Get-ChildItem -Name"}


def test_parse_json_in_xml_json_inside_tool_call_block():
    text = '<tool_call>{"name": "edit", "arguments": {"filePath": "a.py", "oldString": "x", "newString": "y"}}</tool_call>'
    parsed = parse_tool_calls(text)
    assert parsed is not None
    calls, _ = parsed
    assert calls is not None
    assert calls[0].name == "edit"
    args = json.loads(calls[0].arguments)
    assert args["filePath"] == "a.py"


def test_parse_json_in_xml_many_xml_tools():
    text = (
        "<tool_calls>\n"
        '<invoke name="bash">\n<command>Get-ChildItem -Name</command>\n</invoke>\n'
        '<invoke name="read">\n<filePath>README.md</filePath>\n</invoke>\n'
        '<invoke name="write">\n<filePath>note.txt</filePath>\n<content>hello</content>\n</invoke>\n'
        "</tool_calls>"
    )
    parsed = parse_tool_calls(text)
    assert parsed is not None
    calls, _ = parsed
    assert calls is not None
    assert [c.name for c in calls] == ["bash", "read", "write"]
    assert json.loads(calls[1].arguments) == {"filePath": "README.md"}
    assert json.loads(calls[2].arguments) == {"filePath": "note.txt", "content": "hello"}


def test_parse_json_in_xml_parameter_style_xml():
    text = (
        '<tool_calls><invoke name="edit">'
        '<parameter name="filePath">a.py</parameter>'
        '<parameter name="oldString">1</parameter>'
        '<parameter name="newString">2</parameter>'
        "</invoke></tool_calls>"
    )
    parsed = parse_tool_calls(text)
    assert parsed is not None
    calls, _ = parsed
    assert calls is not None
    assert calls[0].name == "edit"
    assert json.loads(calls[0].arguments) == {"filePath": "a.py", "oldString": "1", "newString": "2"}


def test_format_tool_message():
    calls = [ToolCall.create("get_weather", {"city": "Moscow"})]
    message = format_tool_message(calls, "", "think step by step")
    assert message["role"] == "assistant"
    assert message["content"] == ""
    assert message["reasoning_content"] == "think step by step"
    assert len(message["tool_calls"]) == 1
    tool_call = message["tool_calls"][0]
    assert tool_call["type"] == "function"
    assert tool_call["function"]["name"] == "get_weather"


def test_tool_call_deltas():
    calls = [ToolCall.create("get_weather", {"city": "Moscow"})]
    deltas = tool_call_deltas(calls)
    assert len(deltas) >= 2
    assert deltas[0]["role"] == "assistant"
    assert deltas[0]["tool_calls"][0]["function"]["name"] == "get_weather"
    arguments = "".join(d["tool_calls"][0]["function"]["arguments"] for d in deltas[1:])
    assert arguments == '{"city": "Moscow"}'


def test_render_message_content_list_tool_call():
    msg = Message(
        role="assistant",
        content=[{"type": "tool_call", "id": "c1", "function": {"name": "bash", "arguments": '{"command": "ls"}'}}],
    )
    text = render_message(msg)
    assert "bash" in text
    assert "ls" in text


def test_render_json_mode_none():
    assert render_json_mode(None) is None


def test_render_json_mode_string():
    block = render_json_mode("json_object")
    assert block is not None
    assert block is not None
    assert "valid JSON object" in block


def test_render_json_mode_unknown_type():
    assert render_json_mode("text") is None
    assert render_json_mode({"type": "text"}) is None


def test_render_json_mode_schema():
    block = render_json_mode({"type": "json_schema", "json_schema": {"schema": {"type": "object"}}})
    assert block is not None
    assert block is not None
    assert "JSON Schema" in block
    assert '"type": "object"' in block


def test_extract_system_collects_system():
    messages = [
        Message(role="system", content="one"),
        Message(role="user", content="x"),
        Message(role="system", content="two"),
    ]
    assert extract_system(messages) == "one\ntwo"


def test_extract_system_no_system():
    assert extract_system([Message(role="user", content="x")]) == ""


def test_build_prompt_plain():
    prompt, tool_mode = build_prompt([Message(role="user", content="hello")])
    assert prompt == "hello"
    assert not tool_mode


def test_build_prompt_system_injected():
    messages = [Message(role="system", content="Be concise."), Message(role="user", content="Explain X")]
    prompt, tool_mode = build_prompt(messages)
    assert not tool_mode
    assert prompt.startswith("Be concise.")
    assert "Explain X" in prompt
    assert prompt.index("Explain X") > prompt.index("Be concise.")


def test_build_prompt_json_mode():
    messages = [Message(role="user", content="Extract JSON")]
    prompt, tool_mode = build_prompt(messages, response_format="json_object")
    assert not tool_mode
    assert "valid JSON object" in prompt
    assert "Extract JSON" in prompt


def test_build_prompt_system_and_tools_and_json():
    messages = [Message(role="system", content="sys"), Message(role="user", content="q")]
    prompt, tool_mode = build_prompt(messages, [WEATHER_TOOL], None, False, {"type": "json_schema", "json_schema": {"schema": {"type": "object"}}})
    assert tool_mode
    assert prompt.startswith("sys")
    assert "get_weather" in prompt
    assert "JSON Schema" in prompt
    assert "q" in prompt


def test_build_prompt_first_tool_round():
    messages = [Message(role="user", content="What is the weather?")]
    prompt, tool_mode = build_prompt(messages, [WEATHER_TOOL], None, has_session=False)
    assert tool_mode
    assert "get_weather" in prompt
    assert "What is the weather?" in prompt


def test_build_prompt_continuation_with_session():
    messages = [
        Message(role="user", content="What is the weather?"),
        Message(role="assistant", tool_calls=[{"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "Moscow"}'}}]),
        Message(role="tool", content="22C, sunny", tool_call_id="call_1"),
    ]
    prompt, tool_mode = build_prompt(messages, None, None, has_session=True)
    assert tool_mode
    assert "22C, sunny" in prompt
    assert "Continue the conversation" in prompt
    assert "What is the weather?" not in prompt


def test_build_prompt_continuation_no_session():
    messages = [
        Message(role="user", content="What is the weather?"),
        Message(role="assistant", tool_calls=[{"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "Moscow"}'}}]),
        Message(role="tool", content="22C, sunny", tool_call_id="call_1"),
    ]
    prompt, tool_mode = build_prompt(messages, None, None, has_session=False)
    assert tool_mode
    assert "What is the weather?" in prompt
    assert "22C, sunny" in prompt
    assert "get_weather" in prompt


def test_build_prompt_continuation_with_session_skips_schema():
    messages = [
        Message(role="user", content="What is the weather?"),
        Message(role="assistant", content="It is 22C."),
        Message(role="user", content="And in Rome?"),
    ]
    prompt, tool_mode = build_prompt(messages, [WEATHER_TOOL], None, has_session=True)
    assert tool_mode
    assert "And in Rome?" in prompt
    assert "get_weather" not in prompt


def test_build_prompt_tool_round_with_session_skips_schema():
    messages = [
        Message(role="user", content="What is the weather?"),
        Message(role="assistant", tool_calls=[{"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "Moscow"}'}}]),
        Message(role="tool", content="22C, sunny", tool_call_id="call_1"),
    ]
    prompt, tool_mode = build_prompt(messages, [WEATHER_TOOL], None, has_session=True)
    assert tool_mode
    assert "22C, sunny" in prompt
    assert "You have access to the following functions" not in prompt


def test_build_prompt_new_chat_with_history_renders_full_context():
    messages = [
        Message(role="user", content="What is the weather?"),
        Message(role="assistant", content="It is 22C."),
        Message(role="user", content="And in Rome?"),
    ]
    prompt, tool_mode = build_prompt(messages, None, None, has_session=False)
    assert not tool_mode
    assert "What is the weather?" in prompt
    assert "It is 22C." in prompt
    assert "And in Rome?" in prompt


def test_build_prompt_new_chat_with_history_and_tools_renders_full_context():
    messages = [
        Message(role="user", content="What is the weather?"),
        Message(role="assistant", content="It is 22C."),
        Message(role="user", content="And in Rome?"),
    ]
    prompt, tool_mode = build_prompt(messages, [WEATHER_TOOL], None, has_session=False)
    assert tool_mode
    assert "get_weather" in prompt
    assert "What is the weather?" in prompt
    assert "It is 22C." in prompt
    assert "And in Rome?" in prompt


def test_fix_unbalanced_json_stray_closers_dropped():
    assert _fix_unbalanced_json("]{") == "{}"
    assert _fix_unbalanced_json("}") == ""
    assert _fix_unbalanced_json("[}") == "[]"


def test_fix_unbalanced_json_missing_brace_before_bracket():
    assert _fix_unbalanced_json('{"a": 1]') == '{"a": 1}'


def test_fix_unbalanced_json_wrong_closer_repaired():
    assert _fix_unbalanced_json('{"a": [1}]') == '{"a": [1]}'


def test_fix_unbalanced_json_truncated_nested():
    assert _fix_unbalanced_json('[{"a": 1],') == '[{"a": 1}],'


def test_fix_unbalanced_json_unterminated_string_closed():
    assert _fix_unbalanced_json('{"a": "abc') == '{"a": "abc"}'
    assert _fix_unbalanced_json('{"a": "unterminated{[') == '{"a": "unterminated{["}'
    assert _fix_unbalanced_json('[{"a": "x') == '[{"a": "x"}]'


def test_fix_unbalanced_json_escaped_backslash_at_end_closed():
    fixed = _fix_unbalanced_json('[{"a": "x\\')
    assert fixed is not None
    assert fixed is not None
    assert fixed == '[{"a": "x\\\\"}]'
    assert json.loads(fixed) == [{"a": "x\\"}]


def test_fix_unbalanced_json_balanced_returns_none():
    assert _fix_unbalanced_json('{"a": [1, 2]}') is None
    assert _fix_unbalanced_json('"just string"') is None
    assert _fix_unbalanced_json('{"a": "\\"b\\"}", "c": [1, 2]}') is None


def test_strip_dsml_empty():
    assert _strip_dsml("") == ""
    assert _strip_dsml(None) is None


def test_strip_dsml_unicode_markers():
    for pipe in ("\u2016", "\uff5c", "\u01c0", "\u01c1", "\u05c0", "\u00a6", "\u2551", "\ufe31", "\u2223", "\u2758"):
        stripped = _strip_dsml(f"{pipe}DSML{pipe}<thinking>x</thinking>{pipe}DSML{pipe}\nHello")
        assert "<thinking>" not in stripped
        assert f"{pipe}DSML{pipe}" not in stripped
        assert "Hello" in stripped


def test_strip_dsml_tags_with_suffix_preserves_json():
    text = '<\u2016DSML\u2016tool_calls>{"tool_calls":[{"name":"f","arguments":{"city":"Moscow"}}]}</\u2016DSML\u2016tool_calls>'
    calls, wrapper = parse_tool_calls(text)
    assert calls is not None
    assert calls[0].name == "f"
    assert wrapper == ""


def test_strip_dsml_removes_hidden_reasoning():
    text = '<||DSML||thinking>step 1 step 2</||DSML||thinking>\n{"tool_calls":[{"name":"f","arguments":{"x":1}}]}'
    calls, wrapper = parse_tool_calls(text)
    assert calls is not None
    assert calls[0].name == "f"
    assert "step 1" not in wrapper


def test_strip_dsml_json_in_attrs_preserved():
    text = '<||DSML||tool_calls {"tool_calls":[{"name":"f","arguments":{"city":"Moscow"}}]}>'
    calls, _ = parse_tool_calls(text)
    assert calls is not None
    assert calls[0].name == "f"


def test_strip_dsml_render_message_cleans_dsml():
    msg = Message(role="assistant", content="<||DSML||thinking>secret</||DSML||thinking>answer")
    rendered = render_message(msg)
    assert "<thinking>" not in rendered
    assert "DSML" not in rendered
    assert "answer" in rendered


_DSML_JUNK_MARKER = "\u044f\u255c\u042c\u044f\u255c\u042c"


def test_strip_dsml_junk_marker_normalizes_xml():
    text = (
        f"<{_DSML_JUNK_MARKER}DSML{_DSML_JUNK_MARKER}tool_calls>"
        f'<{_DSML_JUNK_MARKER}DSML{_DSML_JUNK_MARKER}invoke name="edit">'
        f'<{_DSML_JUNK_MARKER}DSML{_DSML_JUNK_MARKER}parameter name="filePath">a.py'
        f"</{_DSML_JUNK_MARKER}DSML{_DSML_JUNK_MARKER}parameter>"
        f"</{_DSML_JUNK_MARKER}DSML{_DSML_JUNK_MARKER}invoke>"
        f"</{_DSML_JUNK_MARKER}DSML{_DSML_JUNK_MARKER}tool_calls>"
    )
    stripped = _strip_dsml(text)
    assert "<tool_calls>" in stripped
    assert '<invoke name="edit">' in stripped
    assert '<parameter name="filePath">a.py</parameter>' in stripped
    assert "DSML" not in stripped


def test_parse_tool_calls_junk_marker_edit():
    text = (
        f"<{_DSML_JUNK_MARKER}DSML{_DSML_JUNK_MARKER}tool_calls>\n"
        f'<{_DSML_JUNK_MARKER}DSML{_DSML_JUNK_MARKER}invoke name="edit">\n'
        f'<{_DSML_JUNK_MARKER}DSML{_DSML_JUNK_MARKER}parameter name="filePath">D:\\steam\\a.lua</{_DSML_JUNK_MARKER}DSML{_DSML_JUNK_MARKER}parameter>\n'
        f'<{_DSML_JUNK_MARKER}DSML{_DSML_JUNK_MARKER}parameter name="oldString">old</{_DSML_JUNK_MARKER}DSML{_DSML_JUNK_MARKER}parameter>\n'
        f'<{_DSML_JUNK_MARKER}DSML{_DSML_JUNK_MARKER}parameter name="newString">new</{_DSML_JUNK_MARKER}DSML{_DSML_JUNK_MARKER}parameter>\n'
        f"</{_DSML_JUNK_MARKER}DSML{_DSML_JUNK_MARKER}invoke>\n"
        f"</{_DSML_JUNK_MARKER}DSML{_DSML_JUNK_MARKER}tool_calls>"
    )
    calls, wrapper = parse_tool_calls(text)
    assert calls is not None
    assert calls[0].name == "edit"
    assert json.loads(calls[0].arguments) == {"filePath": "D:\\steam\\a.lua", "oldString": "old", "newString": "new"}
    assert wrapper == ""


def test_parse_tool_calls_junk_marker_glob():
    text = (
        f"<{_DSML_JUNK_MARKER}DSML{_DSML_JUNK_MARKER}tool_calls>\n"
        f"<{_DSML_JUNK_MARKER}DSML{_DSML_JUNK_MARKER}glob>\n"
        f"<{_DSML_JUNK_MARKER}DSML{_DSML_JUNK_MARKER}pattern>**/*.py</{_DSML_JUNK_MARKER}DSML{_DSML_JUNK_MARKER}pattern>\n"
        f"</{_DSML_JUNK_MARKER}DSML{_DSML_JUNK_MARKER}glob>\n"
        f"</{_DSML_JUNK_MARKER}DSML{_DSML_JUNK_MARKER}tool_calls>"
    )
    calls, wrapper = parse_tool_calls(text)
    assert calls is not None
    assert calls[0].name == "glob"
    assert json.loads(calls[0].arguments) == {"pattern": "**/*.py"}
    assert wrapper == ""


def test_parse_tool_calls_wrapped_xml_tag_as_tool_name():
    text = "<tool_calls>\n<glob>\n<pattern>*/</pattern>\n</glob>\n</tool_calls>"
    calls, wrapper = parse_tool_calls(text)
    assert calls is not None
    assert calls[0].name == "glob"
    assert json.loads(calls[0].arguments) == {"pattern": "*/"}
    assert wrapper == ""


def test_parse_tool_calls_bare_xml_tag_as_tool_name():
    text = "<glob>\n<pattern>*/</pattern>\n</glob>"
    calls, wrapper = parse_tool_calls(text)
    assert calls is not None
    assert calls[0].name == "glob"
    assert json.loads(calls[0].arguments) == {"pattern": "*/"}
    assert wrapper == ""


def test_parse_tool_calls_bare_xml_multiple_calls():
    text = "<glob><pattern>a</pattern></glob>\n<glob><pattern>b</pattern></glob>"
    calls, wrapper = parse_tool_calls(text)
    assert calls is not None
    assert [call.name for call in calls] == ["glob", "glob"]
    assert [json.loads(call.arguments) for call in calls] == [{"pattern": "a"}, {"pattern": "b"}]
    assert wrapper == ""


def test_parse_tool_calls_bare_xml_selfclose():
    text = '<glob pattern="*/*.py"/>'
    calls, wrapper = parse_tool_calls(text)
    assert calls is not None
    assert calls[0].name == "glob"
    assert json.loads(calls[0].arguments) == {"pattern": "*/*.py"}
    assert wrapper == ""


def test_parse_tool_calls_bare_xml_attrs_and_children():
    text = '<glob recursive="true"><pattern>**/*.py</pattern></glob>'
    calls, wrapper = parse_tool_calls(text)
    assert calls is not None
    assert calls[0].name == "glob"
    assert json.loads(calls[0].arguments) == {"recursive": "true", "pattern": "**/*.py"}
    assert wrapper == ""


def test_parse_tool_calls_bare_xml_with_prose():
    text = "Search now\n<glob>\n<pattern>*.py</pattern>\n</glob>\nDone"
    calls, wrapper = parse_tool_calls(text)
    assert calls is not None
    assert calls[0].name == "glob"
    assert json.loads(calls[0].arguments) == {"pattern": "*.py"}
    assert wrapper == "Search now Done"


def test_parse_tool_calls_bare_xml_skips_html():
    assert parse_tool_calls("<b>bold</b>") is None
    assert parse_tool_calls('<div class="x">text</div>') is None
    assert parse_tool_calls("<code>func(x)</code>") is None


def test_parse_tool_calls_bare_xml_skips_content_only():
    assert parse_tool_calls("<custom>hello</custom>") is None
    assert parse_tool_calls("<tool><pattern>*/</pattern></tool>") is None


def test_parse_tool_calls_bare_xml_param_selfclose_ignored():
    assert parse_tool_calls('<glob><pattern value="*"/></glob>') is None


def test_parse_tool_calls_bare_xml_junk_marker():
    text = (
        f"<{_DSML_JUNK_MARKER}DSML{_DSML_JUNK_MARKER}glob>\n"
        f"<{_DSML_JUNK_MARKER}DSML{_DSML_JUNK_MARKER}pattern>**/*.py</{_DSML_JUNK_MARKER}DSML{_DSML_JUNK_MARKER}pattern>\n"
        f"</{_DSML_JUNK_MARKER}DSML{_DSML_JUNK_MARKER}glob>"
    )
    calls, wrapper = parse_tool_calls(text)
    assert calls is not None
    assert calls[0].name == "glob"
    assert json.loads(calls[0].arguments) == {"pattern": "**/*.py"}
    assert wrapper == ""


def test_parse_tool_calls_junk_marker_json_preserved():
    payload = json.dumps({"tool_calls": [{"name": "f", "arguments": {"city": "Moscow"}}]})
    text = f"<{_DSML_JUNK_MARKER}DSML{_DSML_JUNK_MARKER}tool_calls>{payload}</{_DSML_JUNK_MARKER}DSML{_DSML_JUNK_MARKER}tool_calls>"
    calls, wrapper = parse_tool_calls(text)
    assert calls is not None
    assert calls[0].name == "f"
    assert json.loads(calls[0].arguments) == {"city": "Moscow"}
    assert wrapper == ""


def test_strip_dsml_junk_marker_hidden_reasoning():
    text = f"<{_DSML_JUNK_MARKER}DSML{_DSML_JUNK_MARKER}ds_safety>secret</{_DSML_JUNK_MARKER}DSML{_DSML_JUNK_MARKER}ds_safety>answer"
    stripped = _strip_dsml(text)
    assert "secret" not in stripped
    assert "DSML" not in stripped
    assert "answer" in stripped


def test_strip_dsml_any_unicode_marker():
    markers = (
        "\u03b1\u03b2",
        "\u4e2d\u6587",
        "\U0001f600",
        "\u3042\u3044",
        "\u20ac\u00a9",
        "\u05e9\u05dc",
        "\u0627\u0628",
        "\u00e9\u00ea",
        "\u2500\u2501",
        "\uff0d\uff3f",
        "\u0416\u0419",
        "\u0301\u0300",
    )
    for junk in markers:
        text = (
            f"<{junk}DSML{junk}tool_calls>"
            f'<{junk}DSML{junk}invoke name="f">'
            f'<{junk}DSML{junk}parameter name="x">1</{junk}DSML{junk}parameter>'
            f"</{junk}DSML{junk}invoke>"
            f"</{junk}DSML{junk}tool_calls>"
        )
        calls, wrapper = parse_tool_calls(text)
        assert calls is not None
        assert calls[0].name == "f"
        assert json.loads(calls[0].arguments) == {"x": "1"}
        assert wrapper == ""


def test_strip_dsml_any_unicode_marker_hidden():
    for junk in ("\u03b1", "\U0001f600", "\u4e2d"):
        text = f"<{junk}DSML{junk}thinking>secret</{junk}DSML{junk}thinking>answer"
        stripped = _strip_dsml(text)
        assert "secret" not in stripped
        assert "DSML" not in stripped
        assert "answer" in stripped


def test_strip_dsml_paired_tag_block():
    text = "hello <|ds_middle|>DSML<|ds_end|> world"
    stripped = _strip_dsml(text)
    assert "DSML" not in stripped
    assert "ds_middle" not in stripped
    assert "ds_end" not in stripped
    assert stripped == "hello   world"


def test_strip_dsml_paired_tag_same_name():
    text = "<|ds_safety|>DSML<|ds_safety|>secret<|ds_safety|>DSML<|ds_safety|>answer"
    stripped = _strip_dsml(text)
    assert "DSML" not in stripped
    assert "ds_safety" not in stripped
    assert "answer" in stripped


def test_strip_dsml_paired_tag_spaces():
    text = "a <|ds_middle|> DSML <|ds_end|> b"
    stripped = _strip_dsml(text)
    assert "DSML" not in stripped
    assert "ds_middle" not in stripped
    assert "ds_end" not in stripped


def test_strip_dsml_two_char_wrap():
    text = "before |>DSML<| after"
    stripped = _strip_dsml(text)
    assert "DSML" not in stripped
    assert "before" in stripped
    assert "after" in stripped


def test_strip_dsml_two_char_pipe_wrap():
    text = "before ||DSML|| after"
    stripped = _strip_dsml(text)
    assert "DSML" not in stripped
    assert "before" in stripped
    assert "after" in stripped


def test_strip_dsml_json_tag_untouched_by_paired_rules():
    text = '<|DSML|{"tool_calls":[{"name":"f","arguments":{"x":1}}]}>'
    stripped = _strip_dsml(text)
    assert stripped == '{"tool_calls":[{"name":"f","arguments":{"x":1}}]}'


def test_parse_tool_calls_paired_tag_block_cleaned():
    text = '<|ds_middle|>DSML<|ds_end|>{"tool_calls":[{"name":"f","arguments":{"x":1}}]}'
    calls, wrapper = parse_tool_calls(text)
    assert calls is not None
    assert calls[0].name == "f"
    assert json.loads(calls[0].arguments) == {"x": 1}
    assert wrapper == ""


def test_strip_dsml_render_message_paired_tags():
    msg = Message(role="assistant", content="<|ds_middle|>DSML<|ds_end|>answer")
    rendered = render_message(msg)
    assert "DSML" not in rendered
    assert "ds_middle" not in rendered
    assert "answer" in rendered


def test_tool_names():
    assert tool_names(None) == set()
    assert tool_names([]) == set()
    assert tool_names([{"function": {"name": "a"}}]) == {"a"}
    assert tool_names([{"name": "b"}]) == {"b"}
    assert tool_names([{"function": {}}, {"function": {"name": ""}}, "x"]) == set()


def test_strip_dsml_tag_json_without_name_preserved():
    text = '<|DSML|{"tool_calls":[{"name":"f","arguments":{"x":1}}]}>'
    stripped = _strip_dsml(text)
    assert stripped == '{"tool_calls":[{"name":"f","arguments":{"x":1}}]}'


def test_strip_dsml_tag_without_name_replaced():
    assert _strip_dsml("<|DSML|123>") == " "


def test_strip_dsml_tag_json_without_calls_replaced():
    text = '<|DSML|{"answer": 42}>'
    stripped = _strip_dsml(text)
    assert "answer" not in stripped


def test_tool_call_create_variants():
    assert ToolCall.create("f", None).arguments == "{}"
    assert ToolCall.create("f", "raw").arguments == "raw"
    assert ToolCall.create("f", {"a": 1}).arguments == '{"a": 1}'
    assert ToolCall.create("f", [1, 2]).arguments == "[1, 2]"
    assert ToolCall.create("f", 5).arguments == "5"


def test_tool_function_variants():
    assert _tool_function(None) is None
    assert _tool_function("str") is None
    assert _tool_function({"function": 42}) is None
    assert _tool_function({"name": "x"}) == {"name": "x"}
    assert _tool_function({"function": {"name": "y"}}) == {"name": "y"}


def test_render_tool_schema_empty_names():
    assert render_tool_schema([{"function": {"name": ""}}]) is None
    assert render_tool_schema([{"function": {"parameters": {}}}]) is None


def test_render_tool_schema_string_params():
    tool = {"function": {"name": "f", "parameters": '{"type":"object"}'}}
    schema = render_tool_schema([tool])
    assert schema is not None
    assert schema is not None
    assert '{"type":"object"}' in schema


def test_content_text_variants():
    assert _content_text(42) == ""
    assert _content_text([42, {"type": "image_url"}]) == ""


def test_content_fingerprint_variants():
    assert _content_fingerprint(42) == ""
    assert _content_fingerprint([{"type": "image_url", "image_url": "data:x"}]) == "data:x"
    assert _content_fingerprint([42, "str"]) == "str"
    assert _content_fingerprint([{"type": "other", "text": 5}]) == ""
    assert _content_fingerprint([{"type": "image_url", "image_url": 42}]) == ""


def test_render_tool_call_mention_variants():
    assert _render_tool_call_mention(None) == ""
    assert _render_tool_call_mention({"name": "f", "arguments": [1]}) == "[assistant called f([1])]"
    assert _render_tool_call_mention({"function": {"name": "g", "arguments": "{}"}}) == "[assistant called g({})]"


def test_render_message_roles():
    msg = Message(role="function", name="calc", content="42")
    assert render_message(msg) == "Function calc returned: 42"
    msg = Message(role="tool", content="ok")
    assert render_message(msg) == "Tool result: ok"
    msg = Message(role="other", content="x")
    assert render_message(msg) == "x"


def test_extract_last_user_image_only():
    msg = Message(role="user", content=[{"type": "image_url", "image_url": {"url": "data:image/png;base64,x"}}])
    with pytest.raises(ValueError):
        extract_last_user([msg])


def test_has_history_multiple_users():
    assert _has_history([Message("user", "a"), Message("user", "b")])
    assert not _has_history([Message("user", "a")])


def test_tail_after_last_user_no_user():
    msgs = [Message("assistant", "a"), Message("assistant", "b")]
    assert _tail_after_last_user(msgs) == msgs


def test_render_json_mode_unknown():
    assert render_json_mode({"type": "text"}) is None
    assert render_json_mode(42) is None
    assert render_json_mode("text") is None


def test_normalize_single_quotes_escapes():
    assert _normalize_single_quotes(r"{'a': 'say \"hi\"'}") == '{"a": "say \\"hi\\""}'
    assert _normalize_single_quotes(r"{'a': 'it\'s'}") == '{"a": "it\'s"}'
    assert _normalize_single_quotes(r"{'a': 'x\\y'}") == '{"a": "x\\\\y"}'


def test_coerce_scalar():
    assert _coerce_scalar("5", "integer") == 5
    assert _coerce_scalar("5.5", "number") == 5.5
    assert _coerce_scalar("abc", "integer") == "abc"
    assert _coerce_scalar("true", "boolean") is True
    assert _coerce_scalar("false", "boolean") is False
    assert _coerce_scalar("maybe", "boolean") == "maybe"
    assert _coerce_scalar("x", "null") is None
    assert _coerce_scalar(5, "integer") == 5
    assert _coerce_scalar("str", "string") == "str"


def test_is_jsonish_arguments():
    assert _is_jsonish_arguments({"a": 1})
    assert _is_jsonish_arguments('{"a": 1}')
    assert not _is_jsonish_arguments("nope")
    assert not _is_jsonish_arguments(42)


def test_extract_one_call_variants():
    call = _extract_one_call({"function": {"name": "f", "arguments": {}}})
    assert call.name == "f"
    call = _extract_one_call({"name": "g", "arguments": {}})
    assert call.name == "g"
    assert _extract_one_call({"name": ""}) is None
    assert _extract_one_call("str") is None


def test_extract_calls_bare_dict():
    calls = _extract_calls({"name": "f", "arguments": {"x": 1}})
    assert calls is not None
    assert calls is not None
    assert calls[0].name == "f"


def test_tool_schema_map():
    tools = [
        {
            "function": {
                "name": "a",
                "parameters": {"properties": {"x": {"type": ["null", "integer"]}, "y": {"type": "string"}, "z": {}}},
            }
        },
        {"function": {"name": "b"}},
    ]
    result = tool_schema_map(tools)
    assert result == {"a": {"x": "integer", "y": "string"}}


def test_xml_invoke_inline_json():
    from danyapi.tools import _xml_invoke_arguments

    args = _xml_invoke_arguments('{"cmd": "ls"}')
    assert args == {"cmd": "ls"}
    assert _xml_invoke_arguments("  ") is None


def test_xml_tag_attrs():
    from danyapi.tools import _xml_tag_attrs

    attrs = _xml_tag_attrs('key="value" num="5" flag="true"', {"num": "integer", "flag": "boolean"})
    assert attrs == {"key": "value", "num": 5, "flag": True}
    attrs = _xml_tag_attrs('key="value"')
    assert attrs == {"key": "value"}


def test_parse_xml_self_closing():
    text = '<tool_calls><bash command="ls"/></tool_calls>'
    parsed = _parse_xml_tool_calls(text, {"bash": {"command": "string"}})
    assert parsed is not None
    calls, _ = parsed
    assert calls is not None
    assert calls[0].name == "bash"
    assert json.loads(calls[0].arguments) == {"command": "ls"}


def test_parse_bare_array_non_array():
    assert _parse_bare_array_calls("hello") is None
    assert _parse_bare_array_calls("not [json") is None


def test_loads_lenient_paths():
    assert _loads_lenient('{"a": 1,}') == {"a": 1}
    assert _loads_lenient("{'a': 1}") == {"a": 1}
    assert _loads_lenient('{"a": 1}') == {"a": 1}
    with pytest.raises(ValueError):
        _loads_lenient("not json at all")


def test_loads_lenient_bare_quote_fails_late():
    with pytest.raises(ValueError):
        _loads_lenient("{a: b c}")
    with pytest.raises(ValueError):
        _loads_lenient("{'a': b c}")


def test_build_prompt_history_no_session():
    messages = [Message("user", "q1"), Message("assistant", "a1"), Message("user", "q2")]
    prompt, tool_mode = build_prompt(messages, has_session=False)
    assert "q1" in prompt
    assert "q2" in prompt
    assert not tool_mode


def test_build_prompt_session_json():
    messages = [Message("user", "hello")]
    prompt, _ = build_prompt(messages, has_session=True, response_format="json_object")
    assert "valid JSON object" in prompt
    assert "hello" in prompt


def test_tool_call_deltas_with_text():
    calls = [ToolCall.create("f", {"x": 1})]
    deltas = tool_call_deltas(calls, "pre")
    assert deltas[0] == {"role": "assistant", "content": "pre"}


def test_tool_function_non_string_name():
    assert _tool_function({"name": 5}) is None


def test_content_text_string_items():
    assert _content_text(["hello"]) == "hello"
    assert _content_text([{"text": "x"}]) == "x"


def test_content_fingerprint_string_items():
    assert _content_fingerprint(["a"]) == "a"


def test_extract_last_user_string_items():
    msg = Message(role="user", content=["a", {"type": "text", "text": "b"}])
    assert extract_last_user([msg]) == "ab"


def test_has_history_tool_role():
    assert _has_history([Message("user", "a"), Message("tool", "42", tool_call_id="c")])


def test_is_tool_round_tail_variants():
    assert _is_tool_round_tail([Message("tool", "x")])
    assert _is_tool_round_tail([Message("assistant", tool_calls=[{"id": "c", "type": "function", "function": {"name": "f", "arguments": "{}"}}])])
    assert _is_tool_round_tail([Message("assistant", content=[{"type": "tool_call", "id": "c", "function": {"name": "f", "arguments": "{}"}}])])
    assert not _is_tool_round_tail([Message("user", "hi")])


def test_build_prompt_history_with_json_mode():
    messages = [Message("user", "q1"), Message("assistant", "a1"), Message("user", "q2")]
    prompt, _ = build_prompt(messages, has_session=False, response_format="json_object")
    assert "valid JSON object" in prompt
    assert prompt.index("valid JSON") < prompt.index("q1")


def test_build_prompt_empty_history_fallback():
    messages = [Message("user", ""), Message("user", "")]
    _prompt, tool_mode = build_prompt(messages, has_session=False)
    assert tool_mode is False


def test_normalize_single_quotes_escaped_other():
    assert _normalize_single_quotes(r"{'a': '\t'}") == '{"a": "\\t"}'


def test_extract_wrapped_calls_tool_calls_dict():
    from danyapi.tools import _extract_wrapped_calls

    calls = _extract_wrapped_calls({"tool_calls": {"name": "f", "arguments": {"x": 1}}})
    assert calls is not None
    assert calls[0].name == "f"


def test_parse_two_objects_second_is_tool_call():
    text = '{"a": 1} {"tool_calls": [{"name": "f", "arguments": {"x": 1}}]}'
    parsed = parse_tool_calls(text)
    assert parsed is not None
    calls, _ = parsed
    assert calls is not None
    assert calls[0].name == "f"


def test_xml_invoke_empty_args_skipped():
    text = '<tool_calls><invoke name="bash">   </invoke></tool_calls>'
    parsed = _parse_xml_tool_calls(text, {"bash": {"cmd": "string"}})
    assert parsed == (None, "")


def test_xml_open_pattern_overlap_skipped():
    text = '<invoke name="other"><bash>x</bash></invoke>'
    parsed = _parse_xml_tool_calls(text, {"other": {"a": "string"}, "bash": {"cmd": "string"}})
    assert parsed is not None
    calls, _ = parsed
    assert calls is not None
    assert [c.name for c in calls] == ["other"]


def test_xml_open_pattern_empty_merged_skipped():
    text = "<bash></bash>"
    parsed = _parse_xml_tool_calls(text, {"bash": {"cmd": "string"}})
    assert parsed == (None, "")


def test_xml_open_pattern_with_schema():
    text = "<bash><cmd>ls</cmd></bash>"
    parsed = _parse_xml_tool_calls(text, {"bash": {"cmd": "string"}})
    assert parsed is not None
    calls, _wrapper = parsed
    assert calls is not None
    assert calls[0].name == "bash"
    assert json.loads(calls[0].arguments) == {"cmd": "ls"}


def test_xml_selfclose_overlap_skipped():
    text = '<invoke name="other"><bash/></invoke>'
    parsed = _parse_xml_tool_calls(text, {"other": {"a": "string"}, "bash": {"cmd": "string"}})
    assert parsed is not None
    calls, _ = parsed
    assert calls is not None
    assert [c.name for c in calls] == ["other"]


def test_tool_schema_map_skips_bad_tools():
    tools = [
        {"function": {}},
        {"function": {"name": ""}},
        {"function": {"name": "a", "parameters": "not dict"}},
        {"function": {"name": "b", "parameters": {"properties": "not dict"}}},
        {"function": {"name": "c", "parameters": {"properties": {"x": "not dict"}}}},
    ]
    assert tool_schema_map(tools) == {}


def test_xml_invoke_broken_json():
    from danyapi.tools import _xml_invoke_arguments

    args = _xml_invoke_arguments("{broken")
    assert args == {"content": "{broken"}


def test_parse_xml_empty_merged_skipped():
    text = "<bash />"
    parsed = _parse_xml_tool_calls(text, {"bash": {"cmd": "string"}})
    assert parsed == (None, "")


def test_parse_xml_selfclose_no_args():
    text = "<bash />"
    parsed = _parse_xml_tool_calls(text, {"bash": {}})
    assert parsed == (None, "")


def test_balanced_json_no_brace():
    from danyapi.tools import _balanced_json

    assert _balanced_json("no braces here") is None


def test_extract_json_object_balanced_but_invalid():
    from danyapi.tools import _extract_json_object

    assert _extract_json_object('pre {"a": } post') is None


def test_parse_bare_array_broken_json():
    assert _parse_bare_array_calls("[{broken") is None


def test_iter_json_objects_skips_bad_candidates():
    from danyapi.tools import _iter_json_objects

    text = '{"ok": 1} not-json {"bad": }'
    objs = [obj for obj, _, _ in _iter_json_objects(text)]
    assert objs == [{"ok": 1}]


def test_parse_bare_keys_json():
    parsed = parse_tool_calls('{"tool_calls": [{name: glob, arguments: {pattern: "*/"}}]}')
    assert parsed is not None
    calls, _ = parsed
    assert calls is not None
    assert calls[0].name == "glob"
    assert json.loads(calls[0].arguments) == {"pattern": "*/"}


def test_parse_bare_keys_json_unquoted_values():
    parsed = parse_tool_calls("{name: get_weather, arguments: {city: Moscow, temp: 22, sunny: true}}")
    assert parsed is not None
    calls, _ = parsed
    assert calls is not None
    assert calls[0].name == "get_weather"
    assert json.loads(calls[0].arguments) == {"city": "Moscow", "temp": 22, "sunny": True}


def test_parse_alias_keys_json():
    parsed = parse_tool_calls('{"tool": "glob", "input": {"pattern": "*/"}}')
    assert parsed is not None
    calls, _ = parsed
    assert calls is not None
    assert calls[0].name == "glob"
    assert json.loads(calls[0].arguments) == {"pattern": "*/"}


def test_parse_alias_keys_json_bare_array():
    parsed = parse_tool_calls('[{"action": "read", "args": {"filePath": "a.py"}}]')
    assert parsed is not None
    calls, _ = parsed
    assert calls is not None
    assert calls[0].name == "read"
    assert json.loads(calls[0].arguments) == {"filePath": "a.py"}


def test_parse_xml_nameless_invoke_with_name_and_input():
    text = '<tool_use><name>glob</name><input>{"pattern": "*/"}</input></tool_use>'
    parsed = parse_tool_calls(text)
    assert parsed is not None
    calls, _ = parsed
    assert calls is not None
    assert calls[0].name == "glob"
    assert json.loads(calls[0].arguments) == {"pattern": "*/"}


def test_parse_xml_nameless_invoke_with_parameter_children():
    text = "<tool><name>get_weather</name><city>Moscow</city></tool>"
    parsed = parse_tool_calls(text)
    assert parsed is not None
    calls, _ = parsed
    assert calls is not None
    assert calls[0].name == "get_weather"
    assert json.loads(calls[0].arguments) == {"city": "Moscow"}


def test_parse_xml_function_wrapper():
    text = '<functions><function name="glob"><pattern>*/</pattern></function></functions>'
    parsed = parse_tool_calls(text)
    assert parsed is not None
    calls, wrapper = parsed
    assert calls is not None
    assert calls[0].name == "glob"
    assert json.loads(calls[0].arguments) == {"pattern": "*/"}
    assert wrapper == ""


def test_parse_xml_arguments_container_unwrapped():
    text = '<invoke name="bash"><arguments>{"cmd": "ls"}</arguments></invoke>'
    parsed = parse_tool_calls(text)
    assert parsed is not None
    calls, _ = parsed
    assert calls is not None
    assert calls[0].name == "bash"
    assert json.loads(calls[0].arguments) == {"cmd": "ls"}


def test_parse_xml_name_and_input_flat_wrapper():
    text = "<tool_calls><name>glob</name><input><pattern>*/</pattern></input></tool_calls>"
    parsed = parse_tool_calls(text)
    assert parsed is not None
    calls, _ = parsed
    assert calls is not None
    assert calls[0].name == "glob"
    assert json.loads(calls[0].arguments) == {"pattern": "*/"}


def test_parse_python_call_single():
    parsed = parse_tool_calls('get_weather(city="Moscow", units="celsius")')
    assert parsed is not None
    calls, _ = parsed
    assert calls is not None
    assert calls[0].name == "get_weather"
    assert json.loads(calls[0].arguments) == {"city": "Moscow", "units": "celsius"}


def test_parse_python_call_typed_values():
    parsed = parse_tool_calls("read(filePath='a.py', limit=10, cached=False, note=None)")
    assert parsed is not None
    calls, _ = parsed
    assert calls is not None
    assert calls[0].name == "read"
    assert json.loads(calls[0].arguments) == {"filePath": "a.py", "limit": 10, "cached": False, "note": None}


def test_parse_python_call_multiline_and_multiple():
    text = 'glob(pattern="*/")\nread(filePath="a.py", limit=5)'
    parsed = parse_tool_calls(text)
    assert parsed is not None
    calls, _ = parsed
    assert calls is not None
    assert [c.name for c in calls] == ["glob", "read"]
    assert json.loads(calls[1].arguments) == {"filePath": "a.py", "limit": 5}


def test_parse_python_call_not_detected_in_prose():
    assert parse_tool_calls("The answer is read(filePath='a.py').") is None
    assert parse_tool_calls("Just check read(filePath='a.py') please.") is None


def test_parse_python_call_after_prose():
    parsed = parse_tool_calls('Let me check.\n\nglob(pattern="*/")')
    assert parsed is not None
    calls, wrapper = parsed
    assert calls is not None
    assert calls[0].name == "glob"
    assert json.loads(calls[0].arguments) == {"pattern": "*/"}
    assert wrapper == "Let me check."


def test_parse_yaml_block():
    text = 'tool_calls:\n- name: glob\n  arguments:\n    pattern: "*/"\n- name: read\n  arguments:\n    filePath: a.py'
    parsed = parse_tool_calls(text)
    assert parsed is not None
    calls, _ = parsed
    assert calls is not None
    assert [c.name for c in calls] == ["glob", "read"]
    assert json.loads(calls[0].arguments) == {"pattern": "*/"}
    assert json.loads(calls[1].arguments) == {"filePath": "a.py"}


def test_parse_yaml_inline_flow():
    text = 'tool_calls:\n- name: glob\n  arguments: {pattern: "*/"}'
    parsed = parse_tool_calls(text)
    assert parsed is not None
    calls, _ = parsed
    assert calls is not None
    assert calls[0].name == "glob"
    assert json.loads(calls[0].arguments) == {"pattern": "*/"}


def test_parse_yaml_inline_array():
    text = 'tool_calls: [{"name": "glob", "arguments": {"pattern": "*/"}}]'
    parsed = parse_tool_calls(text)
    assert parsed is not None
    calls, _ = parsed
    assert calls is not None
    assert calls[0].name == "glob"
    assert json.loads(calls[0].arguments) == {"pattern": "*/"}


def test_parse_tool_calls_debug():
    report = parse_tool_calls_debug('glob(pattern="*/")')
    assert report["parsed"]
    assert report["strategies"] == ["python_call"]
    assert report["calls"][0]["name"] == "glob"
    assert report["calls"][0]["arguments"] == '{"pattern": "*/"}'
    assert report["wrapper"] == ""

    report = parse_tool_calls_debug("Just a normal answer.")
    assert not report["parsed"]
    assert report["strategies"] == []
    assert report["calls"] == []
    assert report["unrecognized"] == "Just a normal answer."


def test_parse_tool_calls_debug_strategies():
    assert parse_tool_calls_debug('{"tool_calls": [{"name": "a", "arguments": {}}]}')["strategies"] == ["json_wrapped"]
    assert parse_tool_calls_debug('[{"name": "a", "arguments": {}}]')["strategies"] == ["json_array"]
    assert parse_tool_calls_debug('<invoke name="a"><x>1</x></invoke>')["strategies"] == ["xml"]
    assert parse_tool_calls_debug("tool_calls:\n- name: a\n  arguments: {x: 1}")["strategies"] == ["yaml"]
    assert parse_tool_calls_debug('First check. {"name": "a", "arguments": {}}')["strategies"] == ["json_in_prose"]


def test_parse_dsml_strategy_debug():
    text = (
        f"<{_DSML_JUNK_MARKER}DSML{_DSML_JUNK_MARKER}tool_calls>"
        f'<{_DSML_JUNK_MARKER}DSML{_DSML_JUNK_MARKER}invoke name="a">'
        f'<{_DSML_JUNK_MARKER}DSML{_DSML_JUNK_MARKER}parameter name="x">1</{_DSML_JUNK_MARKER}DSML{_DSML_JUNK_MARKER}parameter>'
        f"</{_DSML_JUNK_MARKER}DSML{_DSML_JUNK_MARKER}invoke>"
        f"</{_DSML_JUNK_MARKER}DSML{_DSML_JUNK_MARKER}tool_calls>"
    )
    report = parse_tool_calls_debug(text)
    assert report["parsed"]
    assert report["strategies"] == ["dsml"]
    assert report["calls"][0]["name"] == "a"
    assert json.loads(report["calls"][0]["arguments"]) == {"x": "1"}


def test_parse_dsml_invoke_attrs_before_after_name():
    for attrs in ('type="function" name="edit"', 'name="edit" type="function"', 'name = "edit"'):
        text = (
            f"<{_DSML_JUNK_MARKER}DSML{_DSML_JUNK_MARKER}tool_calls>"
            f"<{_DSML_JUNK_MARKER}DSML{_DSML_JUNK_MARKER}invoke {attrs}>"
            f'<{_DSML_JUNK_MARKER}DSML{_DSML_JUNK_MARKER}parameter name="filePath">a.py</{_DSML_JUNK_MARKER}DSML{_DSML_JUNK_MARKER}parameter>'
            f"</{_DSML_JUNK_MARKER}DSML{_DSML_JUNK_MARKER}invoke>"
            f"</{_DSML_JUNK_MARKER}DSML{_DSML_JUNK_MARKER}tool_calls>"
        )
        calls, wrapper = parse_tool_calls(text)
        assert calls is not None
        assert calls[0].name == "edit"
        assert json.loads(calls[0].arguments) == {"filePath": "a.py"}
        assert wrapper == ""


def test_parse_dsml_invoke_child_elements_without_parameter():
    text = (
        f"<{_DSML_JUNK_MARKER}DSML{_DSML_JUNK_MARKER}tool_calls>"
        f'<{_DSML_JUNK_MARKER}DSML{_DSML_JUNK_MARKER}invoke name="glob">'
        f"<{_DSML_JUNK_MARKER}DSML{_DSML_JUNK_MARKER}pattern>**/*.py</{_DSML_JUNK_MARKER}DSML{_DSML_JUNK_MARKER}pattern>"
        f"</{_DSML_JUNK_MARKER}DSML{_DSML_JUNK_MARKER}invoke>"
        f"</{_DSML_JUNK_MARKER}DSML{_DSML_JUNK_MARKER}tool_calls>"
    )
    calls, wrapper = parse_tool_calls(text)
    assert calls is not None
    assert calls[0].name == "glob"
    assert json.loads(calls[0].arguments) == {"pattern": "**/*.py"}
    assert wrapper == ""


def test_parse_dsml_multiple_invokes():
    text = (
        f"<{_DSML_JUNK_MARKER}DSML{_DSML_JUNK_MARKER}tool_calls>"
        f'<{_DSML_JUNK_MARKER}DSML{_DSML_JUNK_MARKER}invoke name="a">'
        f'<{_DSML_JUNK_MARKER}DSML{_DSML_JUNK_MARKER}parameter name="x">1</{_DSML_JUNK_MARKER}DSML{_DSML_JUNK_MARKER}parameter>'
        f"</{_DSML_JUNK_MARKER}DSML{_DSML_JUNK_MARKER}invoke>"
        f'<{_DSML_JUNK_MARKER}DSML{_DSML_JUNK_MARKER}invoke name="b">'
        f'<{_DSML_JUNK_MARKER}DSML{_DSML_JUNK_MARKER}parameter name="y">2</{_DSML_JUNK_MARKER}DSML{_DSML_JUNK_MARKER}parameter>'
        f"</{_DSML_JUNK_MARKER}DSML{_DSML_JUNK_MARKER}invoke>"
        f"</{_DSML_JUNK_MARKER}DSML{_DSML_JUNK_MARKER}tool_calls>"
    )
    calls, wrapper = parse_tool_calls(text)
    assert calls is not None
    assert [c.name for c in calls] == ["a", "b"]
    assert json.loads(calls[0].arguments) == {"x": "1"}
    assert json.loads(calls[1].arguments) == {"y": "2"}
    assert wrapper == ""


def test_parse_dsml_reasoning_stripped_from_wrapper():
    text = (
        f"<{_DSML_JUNK_MARKER}DSML{_DSML_JUNK_MARKER}thinking>secret</{_DSML_JUNK_MARKER}DSML{_DSML_JUNK_MARKER}thinking>"
        f"<{_DSML_JUNK_MARKER}DSML{_DSML_JUNK_MARKER}tool_calls>"
        f'<{_DSML_JUNK_MARKER}DSML{_DSML_JUNK_MARKER}invoke name="a">'
        f'<{_DSML_JUNK_MARKER}DSML{_DSML_JUNK_MARKER}parameter name="x">1</{_DSML_JUNK_MARKER}DSML{_DSML_JUNK_MARKER}parameter>'
        f"</{_DSML_JUNK_MARKER}DSML{_DSML_JUNK_MARKER}invoke>"
        f"</{_DSML_JUNK_MARKER}DSML{_DSML_JUNK_MARKER}tool_calls>"
    )
    calls, wrapper = parse_tool_calls(text)
    assert calls is not None
    assert calls[0].name == "a"
    assert "secret" not in wrapper


def test_parse_xml_repeated_parameters_become_list():
    text = '<invoke name="t"><parameter name="x">1</parameter><parameter name="x">2</parameter><parameter name="x">3</parameter></invoke>'
    parsed = parse_tool_calls(text)
    assert parsed is not None
    calls, _ = parsed
    assert calls is not None
    assert json.loads(calls[0].arguments) == {"x": ["1", "2", "3"]}


def test_parse_xml_broken_json_param_value():
    text = '<invoke name="t"><x>{broken</x></invoke>'
    parsed = parse_tool_calls(text)
    assert parsed is not None
    calls, _ = parsed
    assert calls is not None
    assert json.loads(calls[0].arguments) == {"x": "{broken"}


def test_parse_xml_nested_param_value():
    text = '<invoke name="t"><filters><name>a</name></filters></invoke>'
    parsed = parse_tool_calls(text)
    assert parsed is not None
    calls, _ = parsed
    assert calls is not None
    assert json.loads(calls[0].arguments) == {"filters": {"name": "a"}}


def test_parse_xml_skip_element_child():
    text = '<invoke name="t"><thinking>z</thinking><city>Moscow</city></invoke>'
    parsed = parse_tool_calls(text)
    assert parsed is not None
    calls, _ = parsed
    assert calls is not None
    assert json.loads(calls[0].arguments) == {"city": "Moscow"}


def test_parse_xml_unparseable_tag_child():
    text = '<invoke name="t"><x><raw></x></invoke>'
    parsed = parse_tool_calls(text)
    assert parsed is not None
    calls, _ = parsed
    assert calls is not None
    assert json.loads(calls[0].arguments) == {"x": "<raw>"}


def test_parse_xml_broken_wrapper_close():
    text = "<tool_calls><glob><pattern>*/</pattern></glob><tool_calls>"
    parsed = parse_tool_calls(text)
    assert parsed is not None
    calls, _ = parsed
    assert calls is not None
    assert calls[0].name == "glob"
    assert json.loads(calls[0].arguments) == {"pattern": "*/"}


def test_parse_xml_invoke_child_name_empty():
    assert parse_tool_calls("<invoke><name>   </name></invoke>") is None


def test_parse_xml_invoke_without_name_or_child():
    assert parse_tool_calls("<invoke><city>Moscow</city></invoke>") is None


def test_parse_xml_selfclose_overlap_nested():
    text = '<invoke name="a"><invoke/></invoke>'
    parsed = parse_tool_calls(text)
    assert parsed is not None
    calls, _ = parsed
    assert calls is not None
    assert calls[0].name == "a"


def test_parse_xml_selfclose_no_name():
    assert parse_tool_calls("<tool_use/>") is None


def test_parse_xml_selfclose_with_attrs():
    text = '<function name="x" pattern="*/"/>'
    parsed = parse_tool_calls(text)
    assert parsed is not None
    calls, _ = parsed
    assert calls is not None
    assert calls[0].name == "x"
    assert json.loads(calls[0].arguments) == {"pattern": "*/"}


def test_parse_xml_wrapper_array_json():
    text = '<tool_calls>[{"name": "a", "arguments": {"x": 1}}]</tool_calls>'
    parsed = parse_tool_calls(text)
    assert parsed is not None
    calls, _ = parsed
    assert calls is not None
    assert calls[0].name == "a"
    assert json.loads(calls[0].arguments) == {"x": 1}


def test_parse_xml_wrapper_object_json():
    text = '<tool_calls>{"name": "a", "arguments": {"x": 1}}</tool_calls>'
    parsed = parse_tool_calls(text)
    assert parsed is not None
    calls, _ = parsed
    assert calls is not None
    assert calls[0].name == "a"
    assert json.loads(calls[0].arguments) == {"x": 1}


def test_parse_xml_wrapper_element_overlap_mismatched_close():
    text = '<tool_calls><invoke name="a"><x>1</x></use_tool></tool_calls>'
    parsed = parse_tool_calls(text)
    assert parsed is not None
    calls, _ = parsed
    assert calls is not None
    assert calls[0].name == "a"
    assert json.loads(calls[0].arguments) == {"x": "1"}


def test_parse_xml_wrapper_element_basic():
    text = "<tool_calls><glob><pattern>*/</pattern></glob></tool_calls>"
    parsed = parse_tool_calls(text)
    assert parsed is not None
    calls, _ = parsed
    assert calls is not None
    assert calls[0].name == "glob"
    assert json.loads(calls[0].arguments) == {"pattern": "*/"}


def test_parse_xml_wrapper_element_empty_args():
    text = "<tool_calls><glob></glob></tool_calls>"
    parsed = parse_tool_calls(text)
    assert parsed is None


def test_parse_xml_wrapper_selfclose_skip_list():
    text = "<tool_calls><invoke/></tool_calls>"
    parsed = parse_tool_calls(text)
    assert parsed is None


def test_parse_xml_wrapper_selfclose_overlap():
    text = "<tool_calls><glob><a/></glob></tool_calls>"
    parsed = parse_tool_calls(text)
    assert parsed is not None
    calls, _ = parsed
    assert calls is not None
    assert calls[0].name == "glob"


def test_parse_xml_wrapper_selfclose_no_args():
    text = "<tool_calls><glob/></tool_calls>"
    parsed = parse_tool_calls(text)
    assert parsed is None


def test_parse_xml_generic_selfclose():
    from danyapi.tools import _parse_xml_tool_calls

    parsed = _parse_xml_tool_calls('<glob pattern="*/"/>', {"glob": {"pattern": "string"}})
    assert parsed is not None
    calls, _ = parsed
    assert calls is not None
    assert calls[0].name == "glob"
    assert json.loads(calls[0].arguments) == {"pattern": "*/"}


def test_parse_python_call_escaped_quotes():
    parsed = parse_tool_calls('f(a="x\\"y")')
    assert parsed is not None
    calls, _ = parsed
    assert calls is not None
    assert json.loads(calls[0].arguments) == {"a": 'x"y'}


def test_parse_python_call_nested_values():
    parsed = parse_tool_calls('f(a={"b": 1}, c=[1, 2])')
    assert parsed is not None
    calls, _ = parsed
    assert calls is not None
    assert json.loads(calls[0].arguments) == {"a": {"b": 1}, "c": [1, 2]}


def test_parse_python_call_empty_value():
    parsed = parse_tool_calls("f(a=)")
    assert parsed is not None
    calls, _ = parsed
    assert calls is not None
    assert json.loads(calls[0].arguments) == {"a": None}


def test_parse_python_call_broken_json_value():
    parsed = parse_tool_calls("f(a={broken)")
    assert parsed is not None
    calls, _ = parsed
    assert calls is not None
    assert json.loads(calls[0].arguments) == {"a": "{broken"}


def test_parse_python_call_mismatched_quotes():
    parsed = parse_tool_calls("f(a='x'x)")
    assert parsed is not None
    calls, _ = parsed
    assert calls is not None
    assert json.loads(calls[0].arguments) == {"a": "'x'x"}


def test_parse_python_call_invalid_escape():
    parsed = parse_tool_calls('f(a="x\\q")')
    assert parsed is not None
    calls, _ = parsed
    assert calls is not None
    assert json.loads(calls[0].arguments) == {"a": "x\\q"}


def test_parse_python_call_single_quote_invalid_escape():
    parsed = parse_tool_calls("f(a='x\\q')")
    assert parsed is not None
    calls, _ = parsed
    assert calls is not None
    assert json.loads(calls[0].arguments) == {"a": "x\\q"}


def test_parse_python_call_bare_value():
    parsed = parse_tool_calls("f(city=Moscow)")
    assert parsed is not None
    calls, _ = parsed
    assert calls is not None
    assert json.loads(calls[0].arguments) == {"city": "Moscow"}


def test_parse_python_call_trailing_comma():
    parsed = parse_tool_calls("f(a=1,)")
    assert parsed is not None
    calls, _ = parsed
    assert calls is not None
    assert json.loads(calls[0].arguments) == {"a": 1}


def test_parse_python_call_positional_rejected():
    assert parse_tool_calls("f(1)") is None


def test_parse_python_call_invalid_key():
    assert parse_tool_calls("f(1a=2)") is None


def test_parse_python_call_escaped_backslash():
    parsed = parse_tool_calls('f(a="x\\\\y")')
    assert parsed is not None
    calls, _ = parsed
    assert calls is not None
    assert json.loads(calls[0].arguments) == {"a": "x\\y"}


def test_parse_python_call_nested_parens():
    assert parse_tool_calls("f(g(x=1))") is None


def test_parse_python_call_unclosed():
    assert parse_tool_calls("f(a=1") is None


def test_parse_python_call_no_args():
    parsed = parse_tool_calls("f()")
    assert parsed is not None
    calls, _ = parsed
    assert calls is not None
    assert calls[0].name == "f"
    assert json.loads(calls[0].arguments) == {}


def test_parse_python_call_empty_text():
    from danyapi.tools import _parse_python_calls

    assert _parse_python_calls("   ") is None


def test_parse_python_call_prose_mid_lines():
    assert parse_tool_calls("read(a=1)\nhello\nglob(b=2)") is None


def test_parse_python_call_multi_no_args():
    parsed = parse_tool_calls("glob()\nread(a=1)")
    assert parsed is not None
    calls, _ = parsed
    assert calls is not None
    assert [c.name for c in calls] == ["glob", "read"]


def test_parse_python_call_multi_bad_args():
    assert parse_tool_calls("glob(a=1)\nread(1)") is None


def test_parse_yaml_quoted_name():
    text = 'tool_calls:\n- name: "glob"\n  arguments:\n    pattern: "*/"'
    parsed = parse_tool_calls(text)
    assert parsed is not None
    calls, _ = parsed
    assert calls is not None
    assert calls[0].name == "glob"
    assert json.loads(calls[0].arguments) == {"pattern": "*/"}


def test_parse_yaml_empty_arguments():
    text = 'tool_calls:\n- name: glob\n  arguments:\n    pattern: "*/"\n- name: read\n  arguments:'
    parsed = parse_tool_calls(text)
    assert parsed is not None
    calls, _ = parsed
    assert calls is not None
    assert [c.name for c in calls] == ["glob", "read"]


def test_parse_yaml_broken_flow_value():
    text = "tool_calls:\n- name: glob\n  arguments: {broken"
    parsed = parse_tool_calls(text)
    assert parsed is not None
    calls, _ = parsed
    assert calls is not None
    assert calls[0].name == "glob"
    assert json.loads(calls[0].arguments) == {}


def test_parse_yaml_mismatched_quote():
    text = 'tool_calls:\n- name: glob\n  arguments:\n    pattern: "unclosed'
    parsed = parse_tool_calls(text)
    assert parsed is not None
    calls, _ = parsed
    assert calls is not None
    assert json.loads(calls[0].arguments) == {"pattern": '"unclosed'}


def test_parse_yaml_invalid_escape():
    text = 'tool_calls:\n- name: glob\n  arguments:\n    pattern: "x\\q"'
    parsed = parse_tool_calls(text)
    assert parsed is not None
    calls, _ = parsed
    assert calls is not None
    assert json.loads(calls[0].arguments) == {"pattern": "x\\q"}


def test_parse_yaml_bool_and_null_values():
    text = "tool_calls:\n- name: glob\n  arguments:\n    flag: on\n    off: off\n    none: ~"
    parsed = parse_tool_calls(text)
    assert parsed is not None
    calls, _ = parsed
    assert calls is not None
    assert json.loads(calls[0].arguments) == {"flag": True, "off": False, "none": None}


def test_parse_yaml_empty_text():
    from danyapi.tools import _parse_yaml_calls

    assert _parse_yaml_calls("") is None


def test_parse_yaml_inline_non_array():
    assert parse_tool_calls("tool_calls: hello") is None


def test_parse_yaml_dash_colon_line():
    text = "tool_calls:\n- : x\n- name: glob"
    parsed = parse_tool_calls(text)
    assert parsed is not None
    calls, _ = parsed
    assert calls is not None
    assert calls[0].name == "glob"


def test_parse_yaml_dash_non_name():
    text = 'tool_calls:\n- a: 1\n- name: glob\n  arguments:\n    pattern: "*/"'
    parsed = parse_tool_calls(text)
    assert parsed is not None
    calls, _ = parsed
    assert calls is not None
    assert calls[0].name == "glob"


def test_parse_yaml_direct_args():
    text = 'tool_calls:\n- name: glob\n  pattern: "*/"'
    parsed = parse_tool_calls(text)
    assert parsed is not None
    calls, _ = parsed
    assert calls is not None
    assert json.loads(calls[0].arguments) == {"pattern": "*/"}


def test_parse_yaml_unknown_line():
    text = "tool_calls:\n- name: glob\nhello"
    parsed = parse_tool_calls(text)
    assert parsed is not None
    calls, _ = parsed
    assert calls is not None
    assert calls[0].name == "glob"


def test_parse_xml_selfclose_empty_attrs():
    assert parse_tool_calls('<function name="x"/>') is None


def test_parse_yaml_empty_name():
    assert parse_tool_calls("tool_calls:\n- name:") is None


def test_parse_yaml_empty_value_key():
    text = "tool_calls:\n- name: glob\n  arguments:\n    pattern:"
    parsed = parse_tool_calls(text)
    assert parsed is not None
    calls, _ = parsed
    assert calls is not None
    assert json.loads(calls[0].arguments) == {"pattern": None}


def test_parse_yaml_single_quoted_value():
    text = "tool_calls:\n- name: glob\n  arguments:\n    pattern: '*/'"
    parsed = parse_tool_calls(text)
    assert parsed is not None
    calls, _ = parsed
    assert calls is not None
    assert json.loads(calls[0].arguments) == {"pattern": "*/"}


def test_parse_yaml_empty_dash_item():
    text = "tool_calls:\n- \n- name: glob"
    parsed = parse_tool_calls(text)
    assert parsed is not None
    calls, _ = parsed
    assert calls is not None
    assert calls[0].name == "glob"


def test_parse_yaml_bare_name_item():
    text = 'tool_calls:\n- glob\n  arguments:\n    pattern: "*/"'
    parsed = parse_tool_calls(text)
    assert parsed is not None
    calls, _ = parsed
    assert calls is not None
    assert calls[0].name == "glob"
    assert json.loads(calls[0].arguments) == {"pattern": "*/"}


def test_parse_yaml_key_before_any_item():
    assert parse_tool_calls('tool_calls:\npattern: "*/"') is None


def test_choice_name_unknown_dict():
    schema = render_tool_schema([{"function": {"name": "a"}}], tool_choice={"function": {}})
    assert schema is not None
    assert "1. name: a" in schema


def test_render_schema_tool_without_parameters():
    schema = render_tool_schema([{"function": {"name": "a", "description": "d"}}])
    assert schema is not None
    assert "parameters" not in schema


def test_render_message_tool_calls_non_dict():
    msg = Message(role="assistant", content="x", tool_calls=[42, {"function": {"name": "f", "arguments": '{"a":1}'}}])
    rendered = render_message(msg)
    assert "[assistant called f" in rendered


def test_render_message_content_list_no_tool_call():
    msg = Message(role="assistant", content=[{"type": "text", "text": "hi"}])
    assert render_message(msg) == "hi"


def test_render_message_content_tool_call_item():
    msg = Message(role="assistant", content=[{"type": "tool_call", "name": "f", "arguments": '{"x": 1}'}])
    rendered = render_message(msg)
    assert "[assistant called f" in rendered


def test_render_tool_tail_empty():
    from danyapi.tools import _render_tool_tail

    text = _render_tool_tail([Message("tool", "", tool_call_id="c1"), Message("assistant", "done")])
    assert "Continue" in text


def test_extract_last_user_content_variants():
    msg = Message("user", [42, {"type": "other"}, {"type": "text", "text": "go"}])
    assert extract_last_user([msg]) == "go"


def test_is_tool_round_content_list():
    msg = Message("assistant", content=[{"type": "text", "text": "x"}, {"type": "tool_call"}])
    assert is_tool_round([msg])


def test_is_tool_round_non_list_content():
    assert not is_tool_round([Message("assistant", "plain"), Message("user", "x")])


def test_is_tool_round_content_without_calls():
    assert not is_tool_round([Message("assistant", content=[{"type": "text", "text": "x"}])])


def test_is_tool_round_tail_content_list():
    msg = Message("assistant", content=[{"type": "text", "text": "x"}])
    assert not _is_tool_round_tail([msg])


def test_extract_system_empty_text():
    assert extract_system([Message("system", [42])]) == ""


def test_render_json_mode_json_object():
    result = render_json_mode({"type": "json_object"})
    assert result is not None
    assert "JSON object" in result


def test_extract_json_object_non_dict():
    from danyapi.tools import _extract_json_object

    assert _extract_json_object("x {[1, 2]} y") is None


def test_extract_wrapped_calls_variants():
    from danyapi.tools import _extract_wrapped_calls

    calls = _extract_wrapped_calls({"tool_calls": [42, {"name": "f", "arguments": {"x": 1}}]})
    assert calls is not None
    assert [call.name for call in calls] == ["f"]
    assert _extract_wrapped_calls({"tool_calls": {"bad": 1}}) is None
    assert _extract_wrapped_calls({"function_call": {"bad": 1}}) is None


def test_tool_schema_map_type_lists():
    tools = [
        {"function": {"name": "a", "parameters": {"type": "object", "properties": {"x": {"type": ["array"]}}}}},
        {"function": {"name": "b", "parameters": {"type": "object", "properties": {"y": {"type": ["string", "null"]}}}}},
    ]
    result = tool_schema_map(tools)
    assert result["a"]["x"] == ["array"]
    assert result["b"]["y"] == "null"


def test_xml_invoke_arguments_broken_json():
    from danyapi.tools import _xml_invoke_arguments

    result = _xml_invoke_arguments('{"a": 1}, {"b": 2}')
    assert result == {"content": '{"a": 1}, {"b": 2}'}


def test_xml_invoke_arguments_parameter_tags():
    from danyapi.tools import _xml_invoke_arguments

    result = _xml_invoke_arguments('<parameter name="x">1</parameter>')
    assert result == {"x": "1"}


def test_parse_xml_block_pattern_no_calls():
    text = '<tool_call>{"answer": 42}</tool_call><tool_call>{"tool_calls":[{"name":"f","arguments":{"x":1}}]}</tool_call>'
    calls, _ = parse_tool_calls(text)
    assert calls is not None
    assert calls[0].name == "f"


def test_parse_wrapper_array_no_calls():
    assert parse_tool_calls('<tool_calls>[{"bad": 1}]</tool_calls>') is None


def test_parse_wrapper_empty_name_element():
    text = "<tool_calls><name></name><get_weather><city>Moscow</city></get_weather></tool_calls>"
    calls, _ = parse_tool_calls(text)
    assert calls is not None
    assert calls[0].name == "get_weather"


def test_parse_wrapper_arguments_without_name():
    assert parse_tool_calls('<tool_calls><arguments>{"x": 1}</arguments></tool_calls>') is None


def test_parse_bare_array_bad_item():
    from danyapi.tools import _parse_bare_array_calls

    calls = _parse_bare_array_calls('[{"bad": 1}, {"name": "f", "arguments": {"x": 1}}]')
    assert calls is not None
    assert calls[0].name == "f"


def test_iter_json_objects_non_dict():
    from danyapi.tools import _iter_json_objects

    assert list(_iter_json_objects("x {[1, 2]} y")) == []


def test_parse_yaml_inline_not_array():
    from danyapi.tools import _parse_yaml_calls

    assert _parse_yaml_calls("tool_calls: something") is None
    assert _parse_yaml_calls("tool_calls: [bad]") is None


def test_parse_yaml_duplicate_name_key():
    from danyapi.tools import _parse_yaml_calls

    calls = _parse_yaml_calls("tool_calls:\n- name: f\n  name: g\n  x: 1")
    assert calls is not None
    assert calls[0].name == "f"
    assert json.loads(calls[0].arguments) == {"x": 1}


def test_parse_dsml_invoke_empty():
    text = (
        "<||DSML||tool_calls>"
        '<||DSML||invoke name="f"></||DSML||invoke>'
        '<||DSML||invoke name="g"><||DSML||parameter name="x">1</||DSML||parameter></||DSML||invoke>'
        "</||DSML||tool_calls>"
    )
    calls, _ = parse_tool_calls(text)
    assert calls is not None
    assert calls[0].name == "g"
    assert json.loads(calls[0].arguments) == {"x": "1"}


def test_tool_call_deltas_empty_arguments():
    call = ToolCall("id1", "f", "")
    deltas = tool_call_deltas([call], "text")
    assert len(deltas) == 2
    assert deltas[0] == {"role": "assistant", "content": "text"}
    assert deltas[1]["tool_calls"][0]["function"]["arguments"] == ""
