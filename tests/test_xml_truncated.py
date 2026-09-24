import json

from danyapi.tools import parse_tool_calls

LT = "\x3c"
GT = "\x3e"
SLASH = "\x2f"
Q = "\x22"
NL = "\x0a"
SP = "\x20"


def wrap(body, w="calls"):
    return LT + w + GT + NL + body + NL + LT + SLASH + w + GT


def invoke(name, body):
    return LT + "invoke name=" + Q + name + Q + GT + body + LT + SLASH + "invoke" + GT


def param(name, value):
    return LT + "parameter name=" + Q + name + Q + GT + value + LT + SLASH + "parameter" + GT


def check(text, name, args):
    parsed = parse_tool_calls(text)
    assert parsed is not None, repr(text)
    calls, _ = parsed
    assert calls[0].name == name
    assert json.loads(calls[0].arguments) == args


def test_truncated_invoke_without_closing_tag():
    text = wrap(invoke("read", NL + param("filePath", "a.py")))
    check(text, "read", {"filePath": "a.py"})


def test_space_inside_parameter_tag():
    body = invoke("terminal", NL + LT + "  parameter name=" + Q + "command" + Q + GT + "echo hi" + LT + SLASH + "parameter" + GT)
    check(wrap(body), "terminal", {"command": "echo hi"})


def test_space_before_closing_parameter_tag():
    body = invoke("terminal", NL + LT + "parameter name=" + Q + "command" + Q + GT + "ls" + LT + SLASH + "  parameter" + GT)
    check(wrap(body), "terminal", {"command": "ls"})


def test_truncated_parameter_value_stops_at_wrapper_close():
    body = invoke("read", NL + param("filePath", "a.py" + NL))
    text = LT + "calls" + GT + NL + body + LT + SLASH + "calls" + GT
    check(text, "read", {"filePath": "a.py"})


def test_wrapper_invoke_selfclose_leaves_no_content():
    body = LT + "invoke name=" + Q + "terminal" + Q + SP + "command=" + Q + "ls" + Q + SP + SLASH + GT
    check(wrap(body), "terminal", {"command": "ls"})


def test_truncated_head_only_is_not_parsed():
    text = LT + "calls" + GT + SP + LT + "invoke name=" + Q + "read" + Q + GT + SP + LT + "  parameter name=" + Q + "filePath" + Q
    assert parse_tool_calls(text) is None


def test_pipe_in_command_survives():
    check(wrap(invoke("terminal", param("command", "echo hi | wc -l"))), "terminal", {"command": "echo hi | wc -l"})


def test_html_in_string_argument_survives():
    html = LT + "html" + GT + "x" + LT + SLASH + "html" + GT
    parsed = parse_tool_calls(wrap(invoke("write", NL + param("content", html))))
    assert parsed is not None
    calls, _ = parsed
    assert calls[0].name == "write"
    assert html in calls[0].arguments


def test_bare_selfclose_has_no_trailing_content():
    text = LT + "glob pattern=" + Q + "*/*.py" + Q + SLASH + GT
    parsed = parse_tool_calls(text)
    assert parsed is not None
    calls, wrapper = parsed
    assert calls[0].name == "glob"
    assert json.loads(calls[0].arguments) == {"pattern": "*/*.py"}
    assert wrapper == ""
