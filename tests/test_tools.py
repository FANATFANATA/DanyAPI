import json
import unittest
from typing import Any

from danyapi.tools import (
    ToolCall,
    build_prompt,
    extract_last_user,
    extract_system,
    format_tool_message,
    is_tool_round,
    parse_tool_calls,
    render_json_mode,
    render_message,
    render_tool_schema,
    tool_call_deltas,
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


class TestRenderToolSchema(unittest.TestCase):
    def test_basic(self):
        schema = render_tool_schema([WEATHER_TOOL])
        self.assertIsNotNone(schema)
        assert schema is not None
        self.assertIn("get_weather", schema)
        self.assertIn('"city"', schema)
        self.assertIn('"tool_calls"', schema)

    def test_empty_tools(self):
        self.assertIsNone(render_tool_schema([]))
        self.assertIsNone(render_tool_schema(None))

    def test_tool_choice_none(self):
        self.assertIsNone(render_tool_schema([WEATHER_TOOL], "none"))

    def test_tool_choice_required(self):
        schema = render_tool_schema([WEATHER_TOOL], "required")
        self.assertIsNotNone(schema)
        assert schema is not None
        self.assertIn("MUST call", schema)

    def test_tool_choice_function_dict(self):
        schema = render_tool_schema([WEATHER_TOOL], {"type": "function", "function": {"name": "get_weather"}})
        self.assertIsNotNone(schema)
        assert schema is not None
        self.assertIn("get_weather", schema)

    def test_strict_flag_rendered(self):
        tool = {
            "type": "function",
            "function": {"name": "calc", "strict": True, "parameters": {"type": "object", "properties": {"x": {"type": "number"}}}},
        }
        schema = render_tool_schema([tool])
        self.assertIsNotNone(schema)
        assert schema is not None
        self.assertIn("strict", schema)

    def test_compact_parameters_json(self):
        schema = render_tool_schema([WEATHER_TOOL])
        self.assertIsNotNone(schema)
        assert schema is not None
        self.assertIn('{"type":"object"', schema)


class TestIsToolRound(unittest.TestCase):
    def test_plain_user(self):
        self.assertFalse(is_tool_round([Message(role="user", content="hi")]))

    def test_tool_role(self):
        self.assertTrue(is_tool_round([Message(role="tool", content="22C", tool_call_id="call_1")]))

    def test_function_role(self):
        self.assertTrue(is_tool_round([Message(role="function", content="42", name="calc")]))

    def test_assistant_tool_calls(self):
        msg = Message(
            role="assistant",
            tool_calls=[{"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "{}"}}],
        )
        self.assertTrue(is_tool_round([msg]))

    def test_assistant_content_list_tool_call(self):
        msg = Message(
            role="assistant",
            content=[{"type": "tool_call", "id": "c1", "function": {"name": "bash", "arguments": '{"command": "ls"}'}}],
        )
        self.assertTrue(is_tool_round([msg]))


class TestExtractLastUser(unittest.TestCase):
    def test_last_user(self):
        messages = [
            Message(role="system", content="sys"),
            Message(role="user", content="q1"),
            Message(role="assistant", content="a1"),
            Message(role="user", content="q2"),
        ]
        self.assertEqual(extract_last_user(messages), "q2")

    def test_list_content(self):
        msg = Message(role="user", content=[{"type": "text", "text": "look"}, {"type": "image_url", "image_url": {"url": "data:image/png;base64,xx"}}])
        self.assertEqual(extract_last_user([msg]), "look")

    def test_no_user(self):
        with self.assertRaises(ValueError):
            extract_last_user([Message(role="assistant", content="a")])

    def test_empty(self):
        with self.assertRaises(ValueError):
            extract_last_user([])


class TestParseToolCalls(unittest.TestCase):
    def test_pure_json(self):
        text = '{"tool_calls": [{"name": "get_weather", "arguments": {"city": "Moscow"}}]}'
        parsed = parse_tool_calls(text)
        self.assertIsNotNone(parsed)
        assert parsed is not None
        calls, wrapper = parsed
        assert calls is not None
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "get_weather")
        self.assertEqual(calls[0].arguments, '{"city": "Moscow"}')
        self.assertTrue(calls[0].id.startswith("call_"))
        self.assertEqual(wrapper, "")

    def test_markdown_fences(self):
        text = '```json\n{"tool_calls": [{"name": "get_weather", "arguments": {"city": "London"}}]}\n```'
        parsed = parse_tool_calls(text)
        self.assertIsNotNone(parsed)
        assert parsed is not None
        calls, _ = parsed
        assert calls is not None
        self.assertEqual(calls[0].name, "get_weather")

    def test_prose_around(self):
        text = 'I will help you.\n\n{"tool_calls": [{"name": "get_weather", "arguments": {"city": "Rome"}}]}\nHope that helps.'
        parsed = parse_tool_calls(text)
        self.assertIsNotNone(parsed)
        assert parsed is not None
        calls, wrapper = parsed
        assert calls is not None
        self.assertEqual(calls[0].name, "get_weather")
        self.assertIn("I will help you", wrapper)

    def test_legacy_function_call(self):
        text = '{"function_call": {"name": "get_weather", "arguments": {"city": "Paris"}}}'
        parsed = parse_tool_calls(text)
        self.assertIsNotNone(parsed)
        assert parsed is not None
        calls, _ = parsed
        assert calls is not None
        self.assertEqual(calls[0].name, "get_weather")

    def test_multiple_calls(self):
        text = '{"tool_calls": [{"name": "a", "arguments": {"x": 1}}, {"name": "b", "arguments": {"y": 2}}]}'
        parsed = parse_tool_calls(text)
        self.assertIsNotNone(parsed)
        assert parsed is not None
        calls, _ = parsed
        assert calls is not None
        self.assertEqual([c.name for c in calls], ["a", "b"])

    def test_content_with_calls(self):
        text = '{"content": "checking", "tool_calls": [{"name": "get_weather", "arguments": {"city": "Kyiv"}}]}'
        parsed = parse_tool_calls(text)
        self.assertIsNotNone(parsed)
        assert parsed is not None
        calls, wrapper = parsed
        assert calls is not None
        self.assertEqual(wrapper, "checking")

    def test_arguments_as_string(self):
        text = '{"tool_calls": [{"name": "get_weather", "arguments": "{\\"city\\": \\"Oslo\\"}"}]}'
        parsed = parse_tool_calls(text)
        self.assertIsNotNone(parsed)
        assert parsed is not None
        calls, _ = parsed
        assert calls is not None
        self.assertEqual(calls[0].arguments, '{"city": "Oslo"}')

    def test_not_a_tool_call(self):
        self.assertIsNone(parse_tool_calls("Just a normal answer."))
        self.assertIsNone(parse_tool_calls('{"answer": 42}'))
        self.assertIsNone(parse_tool_calls(""))
        self.assertIsNone(parse_tool_calls("   "))

    def test_trailing_comma(self):
        text = '{"tool_calls": [{"name": "f", "arguments": {"x": 1},}]}'
        parsed = parse_tool_calls(text)
        self.assertIsNotNone(parsed)
        assert parsed is not None
        calls, _ = parsed
        assert calls is not None
        self.assertEqual(calls[0].name, "f")
        self.assertEqual(json.loads(calls[0].arguments), {"x": 1})

    def test_single_quotes(self):
        text = '{"tool_calls": [{"name": "f", "arguments": {"x": "it\'s"}}]}'
        parsed = parse_tool_calls(text)
        self.assertIsNotNone(parsed)
        assert parsed is not None
        calls, _ = parsed
        assert calls is not None
        self.assertEqual(calls[0].name, "f")
        self.assertEqual(json.loads(calls[0].arguments), {"x": "it's"})

    def test_bare_dict_trailing_comma(self):
        text = '{"name": "f", "arguments": {"x": 1,}}'
        parsed = parse_tool_calls(text)
        self.assertIsNotNone(parsed)
        assert parsed is not None
        calls, _ = parsed
        assert calls is not None
        self.assertEqual(json.loads(calls[0].arguments), {"x": 1})


class TestParseXmlToolCalls(unittest.TestCase):
    def test_bash_invoke(self):
        text = '<tool_calls>\n<invoke name="bash">\n<command>Get-ChildItem -Name</command>\n</invoke>\n</tool_calls>'
        parsed = parse_tool_calls(text)
        self.assertIsNotNone(parsed)
        assert parsed is not None
        calls, wrapper = parsed
        assert calls is not None
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "bash")
        self.assertEqual(json.loads(calls[0].arguments), {"command": "Get-ChildItem -Name"})
        self.assertEqual(wrapper, "")

    def test_multiple_invokes(self):
        text = '<tool_calls><invoke name="a"><x>1</x></invoke><invoke name="b"><y>2</y></invoke></tool_calls>'
        parsed = parse_tool_calls(text)
        assert parsed is not None
        calls, _ = parsed
        assert calls is not None
        self.assertEqual([c.name for c in calls], ["a", "b"])
        self.assertEqual(json.loads(calls[0].arguments), {"x": "1"})
        self.assertEqual(json.loads(calls[1].arguments), {"y": "2"})

    def test_parameter_tag(self):
        text = '<tool_calls><invoke name="get_weather"><parameter name="city">Moscow</parameter></invoke></tool_calls>'
        parsed = parse_tool_calls(text)
        assert parsed is not None
        calls, _ = parsed
        assert calls is not None
        self.assertEqual(json.loads(calls[0].arguments), {"city": "Moscow"})

    def test_xml_entities(self):
        text = '<tool_calls><invoke name="bash"><command>echo &quot;a&quot; &amp; b</command></invoke></tool_calls>'
        parsed = parse_tool_calls(text)
        assert parsed is not None
        calls, _ = parsed
        assert calls is not None
        self.assertEqual(json.loads(calls[0].arguments), {"command": 'echo "a" & b'})

    def test_prose_wrapper(self):
        text = 'Let me check.\n\n<tool_calls><invoke name="bash"><command>ls</command></invoke></tool_calls>'
        parsed = parse_tool_calls(text)
        assert parsed is not None
        calls, wrapper = parsed
        assert calls is not None
        self.assertEqual(calls[0].name, "bash")
        self.assertEqual(wrapper, "Let me check.")

    def test_unquoted_name(self):
        text = "<tool_calls><invoke name=bash><command>pwd</command></invoke></tool_calls>"
        parsed = parse_tool_calls(text)
        assert parsed is not None
        calls, _ = parsed
        assert calls is not None
        self.assertEqual(calls[0].name, "bash")

    def test_plain_text_invoke(self):
        text = '<tool_calls><invoke name="bash">Get-ChildItem</invoke></tool_calls>'
        parsed = parse_tool_calls(text)
        assert parsed is not None
        calls, _ = parsed
        assert calls is not None
        self.assertEqual(json.loads(calls[0].arguments), {"content": "Get-ChildItem"})

    def test_fenced_xml(self):
        text = '```\n<tool_calls>\n<invoke name="bash">\n<command>dir</command>\n</invoke>\n</tool_calls>\n```'
        parsed = parse_tool_calls(text)
        assert parsed is not None
        calls, _ = parsed
        assert calls is not None
        self.assertEqual(calls[0].name, "bash")

    def test_invoke_inside_tool_call_not_duplicated(self):
        text = '<tool_call><invoke name="bash"><command>ls</command></invoke></tool_call>'
        parsed = parse_tool_calls(text)
        assert parsed is not None
        calls, _ = parsed
        assert calls is not None
        self.assertEqual([c.name for c in calls], ["bash"])


class TestParseBareArrayCalls(unittest.TestCase):
    def test_bare_array(self):
        text = '[{"name": "bash", "arguments": {"command": "ls"}}, {"name": "get_weather", "arguments": {"city": "Moscow"}}]'
        parsed = parse_tool_calls(text)
        assert parsed is not None
        calls, _ = parsed
        assert calls is not None
        self.assertEqual([c.name for c in calls], ["bash", "get_weather"])
        self.assertEqual(json.loads(calls[1].arguments), {"city": "Moscow"})

    def test_fenced_array(self):
        text = '```json\n[{"name": "bash", "arguments": {"command": "pwd"}}]\n```'
        parsed = parse_tool_calls(text)
        assert parsed is not None
        calls, _ = parsed
        assert calls is not None
        self.assertEqual(calls[0].name, "bash")


class TestParseBareDictCall(unittest.TestCase):
    def test_bare_name_arguments(self):
        text = '{"name": "bash", "arguments": {"command": "Get-ChildItem -Name"}}'
        parsed = parse_tool_calls(text)
        assert parsed is not None
        calls, _ = parsed
        assert calls is not None
        self.assertEqual(calls[0].name, "bash")
        self.assertEqual(json.loads(calls[0].arguments), {"command": "Get-ChildItem -Name"})

    def test_many_tools_prose(self):
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
        self.assertEqual(calls[0].name, "bash")
        self.assertIn("git status", json.loads(calls[0].arguments)["command"])
        self.assertEqual(calls[1].name, "read")
        self.assertEqual(json.loads(calls[1].arguments)["filePath"], "src/main.py")
        self.assertIn("I need to look at the code first", wrapper)


class TestParseJsonInXml(unittest.TestCase):
    def test_json_inside_invoke(self):
        text = '<tool_calls><invoke name="bash">\n{"command": "Get-ChildItem -Name"}\n</invoke></tool_calls>'
        parsed = parse_tool_calls(text)
        assert parsed is not None
        calls, _ = parsed
        assert calls is not None
        self.assertEqual(calls[0].name, "bash")
        self.assertEqual(json.loads(calls[0].arguments), {"command": "Get-ChildItem -Name"})

    def test_json_inside_tool_call_block(self):
        text = '<tool_call>{"name": "edit", "arguments": {"filePath": "a.py", "oldString": "x", "newString": "y"}}</tool_call>'
        parsed = parse_tool_calls(text)
        assert parsed is not None
        calls, _ = parsed
        assert calls is not None
        self.assertEqual(calls[0].name, "edit")
        args = json.loads(calls[0].arguments)
        self.assertEqual(args["filePath"], "a.py")

    def test_many_xml_tools(self):
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
        self.assertEqual([c.name for c in calls], ["bash", "read", "write"])
        self.assertEqual(json.loads(calls[1].arguments), {"filePath": "README.md"})
        self.assertEqual(json.loads(calls[2].arguments), {"filePath": "note.txt", "content": "hello"})

    def test_parameter_style_xml(self):
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
        self.assertEqual(calls[0].name, "edit")
        self.assertEqual(json.loads(calls[0].arguments), {"filePath": "a.py", "oldString": "1", "newString": "2"})


class TestFormatting(unittest.TestCase):
    def test_format_tool_message(self):
        calls = [ToolCall.create("get_weather", {"city": "Moscow"})]
        message = format_tool_message(calls, "", "think step by step")
        self.assertEqual(message["role"], "assistant")
        self.assertEqual(message["content"], "")
        self.assertEqual(message["reasoning_content"], "think step by step")
        self.assertEqual(len(message["tool_calls"]), 1)
        tool_call = message["tool_calls"][0]
        self.assertEqual(tool_call["type"], "function")
        self.assertEqual(tool_call["function"]["name"], "get_weather")

    def test_tool_call_deltas(self):
        calls = [ToolCall.create("get_weather", {"city": "Moscow"})]
        deltas = tool_call_deltas(calls)
        self.assertGreaterEqual(len(deltas), 2)
        self.assertEqual(deltas[0]["role"], "assistant")
        self.assertEqual(deltas[0]["tool_calls"][0]["function"]["name"], "get_weather")
        arguments = "".join(d["tool_calls"][0]["function"]["arguments"] for d in deltas[1:])
        self.assertEqual(arguments, '{"city": "Moscow"}')

    def test_render_message_content_list_tool_call(self):
        msg = Message(
            role="assistant",
            content=[{"type": "tool_call", "id": "c1", "function": {"name": "bash", "arguments": '{"command": "ls"}'}}],
        )
        text = render_message(msg)
        self.assertIn("bash", text)
        self.assertIn("ls", text)


class TestRenderJsonMode(unittest.TestCase):
    def test_none(self):
        self.assertIsNone(render_json_mode(None))

    def test_string(self):
        block = render_json_mode("json_object")
        self.assertIsNotNone(block)
        assert block is not None
        self.assertIn("valid JSON object", block)

    def test_unknown_type(self):
        self.assertIsNone(render_json_mode("text"))
        self.assertIsNone(render_json_mode({"type": "text"}))

    def test_schema(self):
        block = render_json_mode({"type": "json_schema", "json_schema": {"schema": {"type": "object"}}})
        self.assertIsNotNone(block)
        assert block is not None
        self.assertIn("JSON Schema", block)
        self.assertIn('"type": "object"', block)


class TestExtractSystem(unittest.TestCase):
    def test_collects_system(self):
        messages = [
            Message(role="system", content="one"),
            Message(role="user", content="x"),
            Message(role="system", content="two"),
        ]
        self.assertEqual(extract_system(messages), "one\ntwo")

    def test_no_system(self):
        self.assertEqual(extract_system([Message(role="user", content="x")]), "")


class TestBuildPrompt(unittest.TestCase):
    def test_plain(self):
        prompt, tool_mode = build_prompt([Message(role="user", content="hello")])
        self.assertEqual(prompt, "hello")
        self.assertFalse(tool_mode)

    def test_system_injected(self):
        messages = [Message(role="system", content="Be concise."), Message(role="user", content="Explain X")]
        prompt, tool_mode = build_prompt(messages)
        self.assertFalse(tool_mode)
        self.assertTrue(prompt.startswith("Be concise."))
        self.assertIn("Explain X", prompt)
        self.assertGreater(prompt.index("Explain X"), prompt.index("Be concise."))

    def test_json_mode(self):
        messages = [Message(role="user", content="Extract JSON")]
        prompt, tool_mode = build_prompt(messages, response_format="json_object")
        self.assertFalse(tool_mode)
        self.assertIn("valid JSON object", prompt)
        self.assertIn("Extract JSON", prompt)

    def test_system_and_tools_and_json(self):
        messages = [Message(role="system", content="sys"), Message(role="user", content="q")]
        prompt, tool_mode = build_prompt(messages, [WEATHER_TOOL], None, False, {"type": "json_schema", "json_schema": {"schema": {"type": "object"}}})
        self.assertTrue(tool_mode)
        self.assertTrue(prompt.startswith("sys"))
        self.assertIn("get_weather", prompt)
        self.assertIn("JSON Schema", prompt)
        self.assertIn("q", prompt)

    def test_first_tool_round(self):
        messages = [Message(role="user", content="What is the weather?")]
        prompt, tool_mode = build_prompt(messages, [WEATHER_TOOL], None, has_session=False)
        self.assertTrue(tool_mode)
        self.assertIn("get_weather", prompt)
        self.assertIn("What is the weather?", prompt)

    def test_continuation_with_session(self):
        messages = [
            Message(role="user", content="What is the weather?"),
            Message(
                role="assistant", tool_calls=[{"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "Moscow"}'}}]
            ),
            Message(role="tool", content="22C, sunny", tool_call_id="call_1"),
        ]
        prompt, tool_mode = build_prompt(messages, None, None, has_session=True)
        self.assertTrue(tool_mode)
        self.assertIn("22C, sunny", prompt)
        self.assertIn("Continue the conversation", prompt)
        self.assertNotIn("What is the weather?", prompt)

    def test_continuation_no_session(self):
        messages = [
            Message(role="user", content="What is the weather?"),
            Message(
                role="assistant", tool_calls=[{"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "Moscow"}'}}]
            ),
            Message(role="tool", content="22C, sunny", tool_call_id="call_1"),
        ]
        prompt, tool_mode = build_prompt(messages, None, None, has_session=False)
        self.assertTrue(tool_mode)
        self.assertIn("What is the weather?", prompt)
        self.assertIn("22C, sunny", prompt)
        self.assertIn("get_weather", prompt)

    def test_continuation_with_session_skips_schema(self):
        messages = [
            Message(role="user", content="What is the weather?"),
            Message(role="assistant", content="It is 22C."),
            Message(role="user", content="And in Rome?"),
        ]
        prompt, tool_mode = build_prompt(messages, [WEATHER_TOOL], None, has_session=True)
        self.assertTrue(tool_mode)
        self.assertIn("And in Rome?", prompt)
        self.assertNotIn("get_weather", prompt)

    def test_tool_round_with_session_skips_schema(self):
        messages = [
            Message(role="user", content="What is the weather?"),
            Message(
                role="assistant", tool_calls=[{"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "Moscow"}'}}]
            ),
            Message(role="tool", content="22C, sunny", tool_call_id="call_1"),
        ]
        prompt, tool_mode = build_prompt(messages, [WEATHER_TOOL], None, has_session=True)
        self.assertTrue(tool_mode)
        self.assertIn("22C, sunny", prompt)
        self.assertNotIn("You have access to the following functions", prompt)

    def test_new_chat_with_history_renders_full_context(self):
        messages = [
            Message(role="user", content="What is the weather?"),
            Message(role="assistant", content="It is 22C."),
            Message(role="user", content="And in Rome?"),
        ]
        prompt, tool_mode = build_prompt(messages, None, None, has_session=False)
        self.assertFalse(tool_mode)
        self.assertIn("What is the weather?", prompt)
        self.assertIn("It is 22C.", prompt)
        self.assertIn("And in Rome?", prompt)

    def test_new_chat_with_history_and_tools_renders_full_context(self):
        messages = [
            Message(role="user", content="What is the weather?"),
            Message(role="assistant", content="It is 22C."),
            Message(role="user", content="And in Rome?"),
        ]
        prompt, tool_mode = build_prompt(messages, [WEATHER_TOOL], None, has_session=False)
        self.assertTrue(tool_mode)
        self.assertIn("get_weather", prompt)
        self.assertIn("What is the weather?", prompt)
        self.assertIn("It is 22C.", prompt)
        self.assertIn("And in Rome?", prompt)


if __name__ == "__main__":
    unittest.main()
