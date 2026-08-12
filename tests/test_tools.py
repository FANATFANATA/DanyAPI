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


if __name__ == "__main__":
    unittest.main()
