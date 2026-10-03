import pytest
from fastapi import HTTPException

from danyapi.api import mcpagent
from danyapi.api.schemas import ChatCompletionRequest


class FakeRegistry:
    def __init__(self, iterations=8):
        self.iterations = iterations
        self.calls = []

    def enabled(self):
        return True

    def all_tools(self):
        from danyapi.mcp import McpTool

        return [(self, McpTool(name="echo", input_schema={"type": "object"}))]

    def resolve(self, name):
        from danyapi.mcp import McpTool

        if name in ("echo", "boom"):
            return self, McpTool(name=name)
        return None

    async def call(self, server, tool, arguments):
        self.calls.append((tool.name, arguments))
        if tool.name == "boom":
            raise RuntimeError("exploded")
        return f"echo:{arguments.get('value', '')}"

    async def close(self):
        return None


def _message(role, content, **extra):
    return {"role": role, "content": content, **extra}


def _tool_call_message(name, arguments, call_id="call_1"):
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}],
    }


def _provider_result(message, finish="tool_calls", call_id="chatcmpl-1"):
    return {
        "id": call_id,
        "object": "chat.completion",
        "created": 123,
        "model": "m",
        "choices": [{"index": 0, "message": message, "finish_reason": finish, "logprobs": None}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8},
        "session_id": None,
    }


@pytest.fixture
def registry(monkeypatch):
    fake = FakeRegistry()

    class State:
        mcp_registry = fake

    monkeypatch.setattr(mcpagent, "_registry", lambda: fake)
    monkeypatch.setattr(mcpagent, "registry_iterations", lambda: fake.iterations)
    return fake


def _request(messages=None, stream=False):
    return ChatCompletionRequest(
        model="deepseek-v4.1-flash",
        messages=messages or [{"role": "user", "content": "hi"}],
        stream=stream,
    )


@pytest.mark.asyncio
async def test_mcp_loop_executes_and_finishes(registry):
    seen_requests = []

    async def dispatch(req):
        seen_requests.append(req)
        if len(seen_requests) == 1:
            return _provider_result(_tool_call_message("echo", '{"value": "x"}'))
        return _provider_result(_message("assistant", "done"), finish="stop", call_id="chatcmpl-2")

    result = await mcpagent.run_mcp_chat(_request(), dispatch)
    assert result["choices"][0]["message"]["content"] == "done"
    assert result["choices"][0]["finish_reason"] == "stop"
    assert result["usage"]["total_tokens"] == 16
    assert registry.calls == [("echo", {"value": "x"})]
    second = seen_requests[1]
    roles = [message.role for message in second.messages]
    assert roles == ["user", "assistant", "tool"]
    assert second.tools[-1]["function"]["name"] == "echo"
    assert second.tool_choice == "auto"


@pytest.mark.asyncio
async def test_mcp_loop_returns_first_plain_answer(registry):
    async def dispatch(req):
        return _provider_result(_message("assistant", "plain"), finish="stop")

    result = await mcpagent.run_mcp_chat(_request(), dispatch)
    assert result["choices"][0]["message"]["content"] == "plain"
    assert registry.calls == []


@pytest.mark.asyncio
async def test_mcp_tool_error_becomes_tool_result(registry):
    async def dispatch(req):
        if len(registry.calls) == 0:
            return _provider_result(_tool_call_message("boom", "{}"))
        return _provider_result(_message("assistant", "recovered"), finish="stop")

    result = await mcpagent.run_mcp_chat(_request(), dispatch)
    assert result["choices"][0]["message"]["content"] == "recovered"
    assert registry.calls == [("boom", {})]


@pytest.mark.asyncio
async def test_mcp_unknown_tool_becomes_error_result(registry):
    rounds = []

    async def dispatch(req):
        rounds.append(req)
        if len(rounds) == 1:
            return _provider_result(_tool_call_message("ghost", "{}"))
        return _provider_result(_message("assistant", "ok"), finish="stop")

    result = await mcpagent.run_mcp_chat(_request(), dispatch)
    assert result["choices"][0]["message"]["content"] == "ok"
    tool_messages = [message for message in rounds[1].messages if message.role == "tool"]
    assert len(tool_messages) == 1
    assert "not available" in tool_messages[0].content


@pytest.mark.asyncio
async def test_mcp_iteration_limit_appends_note(registry):
    registry.iterations = 1

    async def dispatch(req):
        return _provider_result(_tool_call_message("echo", "{}"))

    result = await mcpagent.run_mcp_chat(_request(), dispatch)
    assert "iteration limit" in result["choices"][0]["message"]["content"]
    assert result["choices"][0]["finish_reason"] == "tool_calls"


@pytest.mark.asyncio
async def test_mcp_stream_is_rejected(registry):
    with pytest.raises(HTTPException) as exc:
        await mcpagent.run_mcp_chat(_request(stream=True), None)
    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_mcp_without_registry_is_503(monkeypatch):
    monkeypatch.setattr(mcpagent, "_registry", lambda: None)
    with pytest.raises(HTTPException) as exc:
        await mcpagent.run_mcp_chat(_request(), None)
    assert exc.value.status_code == 503
