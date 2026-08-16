import json
from typing import Any

import pytest

from danyapi.tools import (
    TOOL_RESULT_MAX_CHARS,
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
    render_json_mode,
    render_message,
    render_tool_schema,
    response_format_schema,
    tool_call_deltas,
    tool_schema_map,
    validate_json_response,
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

# Mirrors the kimi-code Read tool: line_offset is a union of two integer
# ranges, so its JSON Schema has no top-level "type" — only an anyOf.
READ_TOOL = {
    "type": "function",
    "function": {
        "name": "Read",
        "description": "Read a text file",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "line_offset": {
                    "anyOf": [
                        {"type": "integer", "minimum": 1},
                        {"type": "integer", "minimum": -1000, "maximum": -1},
                    ]
                },
                "n_lines": {"type": "integer"},
            },
            "required": ["path"],
        },
    },
}

# Mirrors the kimi-code TodoList tool: a single array-typed property.
TODO_TOOL = {
    "type": "function",
    "function": {
        "name": "TodoList",
        "description": "Update the todo list",
        "parameters": {
            "type": "object",
            "properties": {
                "todos": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "status": {"type": "string", "enum": ["pending", "in_progress", "done"]},
                        },
                        "required": ["title", "status"],
                    },
                }
            },
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


def test_response_format_schema_returns_schema():
    schema = {"type": "object", "required": ["answer"], "properties": {"answer": {"type": "integer"}}}
    rf = {"type": "json_schema", "json_schema": {"name": "answer", "schema": schema}}
    assert response_format_schema(rf) is schema


def test_response_format_schema_rejects_other_shapes():
    assert response_format_schema(None) is None
    assert response_format_schema("json_object") is None
    assert response_format_schema({"type": "json_object"}) is None
    assert response_format_schema({"type": "json_schema"}) is None
    assert response_format_schema({"type": "json_schema", "json_schema": {"schema": 42}}) is None


def test_validate_json_response_none_schema_is_noop():
    validate_json_response({"anything": "goes"}, None)


def test_validate_json_response_valid():
    schema = {"type": "object", "required": ["answer"], "properties": {"answer": {"type": "integer"}}}
    validate_json_response({"answer": 42}, schema)


def test_validate_json_response_invalid():
    schema = {"type": "object", "required": ["answer"], "properties": {"answer": {"type": "integer"}}}
    with pytest.raises(ValueError, match="does not match JSON schema"):
        validate_json_response({"answer": "not a number"}, schema)


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
    # Empty body = zero-argument call → {}, not "no call".
    assert _xml_invoke_arguments("  ") == {}


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


def test_xml_invoke_empty_args_zero_arg_call():
    # Empty invoke body = zero-argument call → "{}", not a skipped call.
    text = '<tool_calls><invoke name="bash">   </invoke></tool_calls>'
    parsed = _parse_xml_tool_calls(text, {"bash": {"cmd": "string"}})
    assert parsed is not None
    calls, _ = parsed
    assert calls is not None
    assert [c.name for c in calls] == ["bash"]
    assert calls[0].arguments == "{}"


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

    text = '{"ok": 1} not-json {"bad": broken}'
    objs = [obj for obj, _, _ in _iter_json_objects(text)]
    assert objs == [{"ok": 1}]


def _tool_result_msg(text: str) -> Message:
    return Message(role="tool", content=text, tool_call_id="call_1")


def test_tool_result_short_untouched():
    assert render_message(_tool_result_msg("42")) == "Tool result (call_1): 42"


def test_tool_result_long_truncated():
    long_text = "x" * 20000
    out = render_message(_tool_result_msg(long_text))
    assert out.startswith("Tool result (call_1): " + "x" * TOOL_RESULT_MAX_CHARS)
    assert f"[truncated: {20000 - TOOL_RESULT_MAX_CHARS} more characters]" in out
    assert len(out) < len(long_text)


def test_tool_result_custom_limit():
    out = render_message(_tool_result_msg("y" * 100), tool_result_limit=40)
    assert out.endswith("y" * 40 + " ... [truncated: 60 more characters]")


def test_tool_result_limit_zero_disables():
    long_text = "z" * 500
    assert render_message(_tool_result_msg(long_text), tool_result_limit=0) == f"Tool result (call_1): {long_text}"


def test_function_result_truncated():
    msg = Message(role="function", content="f" * 9000, name="get_weather")
    assert "[truncated:" in render_message(msg)


def test_build_prompt_session_tail_truncates_tool_result():
    messages = [
        Message(role="user", content="read the file"),
        Message(role="assistant", tool_calls=[{"id": "c1", "type": "function", "function": {"name": "Read", "arguments": "{}"}}]),
        Message(role="tool", content="r" * 20000, tool_call_id="c1"),
    ]
    prompt, tool_mode = build_prompt(messages, None, None, has_session=True)
    assert tool_mode
    assert "[truncated:" in prompt
    assert len(prompt) < TOOL_RESULT_MAX_CHARS + 500


def test_build_prompt_full_history_custom_limit():
    messages = [
        Message(role="user", content="hi"),
        Message(role="assistant", content="ok"),
        Message(role="user", content="read it"),
        Message(role="tool", content="t" * 5000, tool_call_id="c9"),
    ]
    prompt, tool_mode = build_prompt(messages, has_session=False, tool_result_limit=100)
    assert tool_mode
    assert "[truncated: 4900 more characters]" in prompt


# --- corrupted "tool_calls" key separator ---


def test_repair_tool_calls_key_inserts_colon_and_bracket():
    from danyapi.tools import _repair_tool_calls_key

    assert _repair_tool_calls_key('{"tool_calls"> {"name": "a"}}') == '{"tool_calls": [ {"name": "a"}}'
    # Value already starts with "[" — only the colon is re-inserted.
    assert _repair_tool_calls_key('{"tool_calls"> [1, 2]}') == '{"tool_calls":  [1, 2]}'
    # A valid separator is left untouched.
    assert _repair_tool_calls_key('{"tool_calls": [{"name": "a"}]}') == '{"tool_calls": [{"name": "a"}]}'


def test_parse_tool_calls_corrupted_separator():
    # DeepSeek Pro sometimes emits "tool_calls"> instead of "tool_calls": [
    text = 'Still reading.\n\n```json\n{\n  "tool_calls">\n    {\n      "name": "get_weather",\n      "arguments": {"city": "London"}\n    }\n  ]\n}\n```'
    parsed = parse_tool_calls(text)
    assert parsed is not None
    calls, wrapper = parsed
    assert [c.name for c in calls] == ["get_weather"]
    assert calls[0].arguments == '{"city": "London"}'
    assert "Still reading" in wrapper


def test_parse_tool_calls_corrupted_separator_multiple():
    text = '```json\n{\n  "tool_calls">\n    {"name": "a", "arguments": {"x": 1}},\n    {"name": "b", "arguments": {"y": 2}}\n  ]\n}\n```'
    parsed = parse_tool_calls(text)
    assert parsed is not None
    calls, _ = parsed
    assert [c.name for c in calls] == ["a", "b"]


def test_parse_tool_calls_corrupted_separator_missing_closer():
    # Both the corrupted separator and the closing bracket are dropped.
    text = '{"tool_calls"> {"name": "a", "arguments": {"x": 1}}}'
    parsed = parse_tool_calls(text)
    assert parsed is not None
    calls, _ = parsed
    assert [c.name for c in calls] == ["a"]


# --- tool_schema_map declared types (anyOf/oneOf/allOf) ---


def test_tool_schema_map_anyof_integer_union():
    assert tool_schema_map([READ_TOOL])["Read"]["line_offset"] == "integer"


def test_tool_schema_map_array_property_hint():
    assert tool_schema_map([TODO_TOOL])["TodoList"]["todos"] == "array"


def test_tool_schema_map_array_string_union_prefers_array():
    tool = {
        "type": "function",
        "function": {
            "name": "f",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"anyOf": [{"type": "string"}, {"type": "array"}]},
                },
            },
        },
    }
    assert tool_schema_map([tool])["f"]["a"] == "array"


def test_tool_schema_map_nullable_string_prefers_string():
    tool = {
        "type": "function",
        "function": {
            "name": "f",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                    "b": {"type": ["string", "null"]},
                },
            },
        },
    }
    assert tool_schema_map([tool])["f"] == {"a": "string", "b": "string"}


def test_tool_schema_map_property_without_type_skipped():
    tool = {
        "type": "function",
        "function": {
            "name": "f",
            "parameters": {"type": "object", "properties": {"opts": {"description": "free form"}}},
        },
    }
    assert "f" not in tool_schema_map([tool])


# --- XML parameter coercion ---


def test_coerce_scalar_array_object_hints():
    assert _coerce_scalar("[1, 2]", "array") == [1, 2]
    assert _coerce_scalar('{"a": 1}', "object") == {"a": 1}
    # A failed JSON parse falls back to the raw string.
    assert _coerce_scalar("plain text", "array") == "plain text"


def test_coerce_json_container():
    from danyapi.tools import _coerce_json_container

    assert _coerce_json_container("[1, 2, 3]") == [1, 2, 3]
    assert _coerce_json_container('{"a": 1}') == {"a": 1}
    # Scalars and non-JSON stay strings.
    assert _coerce_json_container("42") == "42"
    assert _coerce_json_container("[ not json ]") == "[ not json ]"
    assert _coerce_json_container("") == ""


def test_coerce_untyped_literals():
    from danyapi.tools import _coerce_untyped

    assert _coerce_untyped("true") is True
    assert _coerce_untyped("false") is False
    assert _coerce_untyped("null") is None
    assert _coerce_untyped("42") == 42
    assert _coerce_untyped("3.5") == 3.5
    assert _coerce_untyped("[1, 2]") == [1, 2]
    assert _coerce_untyped("hello") == "hello"


def test_xml_parameter_string_attrs_with_anyof_schema():
    # DeepSeek emits non-string values annotated with string="false", and the
    # client schema may declare the property only via anyOf (no top-level type).
    text = (
        "<tool_calls>\n"
        '<invoke name="Read">\n'
        '<parameter name="path" string="true">/home/u/proj/src/llm/mod.rs</parameter>\n'
        '<parameter name="line_offset" string="false">370</parameter>\n'
        '<parameter name="n_lines" string="false">40</parameter>\n'
        "</invoke>\n"
        "</tool_calls>"
    )
    parsed = parse_tool_calls(text, tool_schema_map([READ_TOOL]))
    assert parsed is not None
    calls, _ = parsed
    args = json.loads(calls[0].arguments)
    assert args["path"] == "/home/u/proj/src/llm/mod.rs"
    assert args["line_offset"] == 370
    assert args["n_lines"] == 40


def test_xml_anyof_integer_coerced_from_schema_without_attrs():
    text = '<tool_calls><invoke name="Read"><parameter name="path">a.rs</parameter><parameter name="line_offset">-100</parameter></invoke></tool_calls>'
    parsed = parse_tool_calls(text, tool_schema_map([READ_TOOL]))
    assert parsed is not None
    calls, _ = parsed
    args = json.loads(calls[0].arguments)
    assert args["line_offset"] == -100


def test_xml_string_attr_false_without_schema():
    text = (
        '<tool_calls><invoke name="f">'
        '<parameter name="limit" string="false">42</parameter>'
        '<parameter name="block" string="false">false</parameter>'
        '<parameter name="note" string="true">42</parameter>'
        "</invoke></tool_calls>"
    )
    parsed = parse_tool_calls(text)
    assert parsed is not None
    calls, _ = parsed
    assert json.loads(calls[0].arguments) == {"limit": 42, "block": False, "note": "42"}


def test_xml_empty_invoke_zero_arg_tool():
    # EnterPlanMode (and similar zero-arg tools) are invoked with an empty
    # body; before the fix the whole block stayed as plain text.
    text = 'Let me call EnterPlanMode.\n\n<tool_calls><invoke name="EnterPlanMode"></invoke></tool_calls>'
    parsed = parse_tool_calls(text)
    assert parsed is not None
    calls, wrapper = parsed
    assert calls[0].name == "EnterPlanMode"
    assert calls[0].arguments == "{}"
    assert wrapper == "Let me call EnterPlanMode."


def test_xml_array_param_json_string():
    # DeepSeek emits nested arrays as pre-serialized JSON strings; the client
    # schema (zod) rejects a string where an array is expected.
    text = (
        '<tool_calls><invoke name="TodoList">'
        '<parameter name="todos">[{"title": "a", "status": "pending"}, {"title": "b", "status": "in_progress"}]</parameter>'
        "</invoke></tool_calls>"
    )
    parsed = parse_tool_calls(text, tool_schema_map([TODO_TOOL]))
    assert parsed is not None
    calls, _ = parsed
    args = json.loads(calls[0].arguments)
    assert args["todos"][0] == {"title": "a", "status": "pending"}
    assert args["todos"][1]["status"] == "in_progress"


def test_xml_array_param_json_string_without_schema():
    # No schema available, but the model marks the parameter string="false".
    text = '<tool_calls><invoke name="TodoList"><parameter name="todos" string="false">[{"title": "a", "status": "pending"}]</parameter></invoke></tool_calls>'
    parsed = parse_tool_calls(text)
    assert parsed is not None
    calls, _ = parsed
    args = json.loads(calls[0].arguments)
    assert args["todos"] == [{"title": "a", "status": "pending"}]


def test_xml_array_param_plain_text_stays_string():
    text = '<tool_calls><invoke name="TodoList"><parameter name="todos">[ -f /tmp/x ]</parameter></invoke></tool_calls>'
    parsed = parse_tool_calls(text, tool_schema_map([TODO_TOOL]))
    assert parsed is not None
    calls, _ = parsed
    args = json.loads(calls[0].arguments)
    assert args["todos"] == "[ -f /tmp/x ]"


def test_xml_bare_array_body():
    text = '<tool_calls><invoke name="TodoList">[{"title": "a", "status": "pending"}]</invoke></tool_calls>'
    parsed = parse_tool_calls(text, tool_schema_map([TODO_TOOL]))
    assert parsed is not None
    calls, _ = parsed
    args = json.loads(calls[0].arguments)
    assert args == [{"title": "a", "status": "pending"}]
