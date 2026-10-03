import asyncio
import json

import pytest

from danyapi.mcp import BuiltinSearchServer, McpError, McpRegistry, McpTool, server_from_config
from danyapi.mcp.registry import _decode_ddg_url, _strip_tags, duckduckgo_search, render_search_results
from danyapi.mcp.transport import HttpTransport, StdioTransport, _parse_http_payload, _parse_tools, _render_result


def test_builtin_search_tool_schema():
    tools = BuiltinSearchServer().tools
    assert len(tools) == 1
    assert tools[0].name == "web_search"
    schema = tools[0].openai_schema()
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "web_search"
    assert schema["function"]["parameters"]["required"] == ["query"]


@pytest.mark.asyncio
async def test_builtin_search_rejects_unknown_tool():
    server = BuiltinSearchServer()
    with pytest.raises(McpError):
        await server.call("nope", {})


@pytest.mark.asyncio
async def test_builtin_search_rejects_empty_query():
    server = BuiltinSearchServer()
    with pytest.raises(McpError):
        await server.call("web_search", {"query": "   "})


@pytest.mark.asyncio
async def test_builtin_search_returns_rendered_results(monkeypatch):
    from danyapi.mcp.registry import SearchResult

    async def fake_search(query, max_results=5):
        return [SearchResult("t1", "https://a", "s1"), SearchResult("t2", "https://b", "s2")]

    monkeypatch.setattr("danyapi.mcp.registry.duckduckgo_search", fake_search)
    server = BuiltinSearchServer()
    text = await server.call("web_search", {"query": "q"})
    assert "t1" in text and "https://a" in text and "s2" in text


@pytest.mark.asyncio
async def test_builtin_search_renders_empty(monkeypatch):
    async def fake_search(query, max_results=5):
        return []

    monkeypatch.setattr("danyapi.mcp.registry.duckduckgo_search", fake_search)
    server = BuiltinSearchServer()
    assert await server.call("web_search", {"query": "q"}) == "no results found"


def test_strip_tags_unescapes_and_strips():
    assert _strip_tags("<b>a &amp; b</b>") == "a & b"


def test_decode_ddg_redirect_url():
    assert _decode_ddg_url("//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fx&r=x") == "https://example.com/x"
    assert _decode_ddg_url("https://plain.example") == "https://plain.example"


@pytest.mark.asyncio
async def test_duckduckgo_search_parses_html(monkeypatch):
    html_body = (
        '<a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fa">Ex&amp;ample</a><a class="result__snippet">Sni&lt;p&gt;pet</a>'
    )

    class FakeResp:
        status_code = 200
        text = html_body

    class FakeClient:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, url, headers=None):
            return FakeResp()

    monkeypatch.setattr("danyapi.mcp.registry.httpx.AsyncClient", FakeClient)
    results = await duckduckgo_search("q", 5)
    assert len(results) == 1
    assert results[0].title == "Ex&ample"
    assert results[0].url == "https://example.com/a"
    assert results[0].snippet == "Sni<p>pet"


def test_render_search_results_limits_length():
    from danyapi.mcp.registry import SearchResult

    results = [SearchResult("t", "https://a", "s") for _ in range(20)]
    assert len(render_search_results(results)) <= 60000


def test_server_from_config_http_and_stdio():
    http = server_from_config("h", "https://mcp.example.com/mcp")
    assert http.name == "h"
    assert isinstance(http.transport, HttpTransport)
    stdio = server_from_config("s", "uvx mcp-server-fetch")
    assert stdio.name == "s"
    assert isinstance(stdio.transport, StdioTransport)


def test_stdio_transport_rejects_empty_command():
    with pytest.raises(McpError):
        StdioTransport("   ")


def test_stdio_transport_rejects_unparseable_command():
    with pytest.raises(McpError):
        StdioTransport('python -c "unclosed')


def test_stdio_transport_request_before_start():
    transport = StdioTransport("python -c pass")
    with pytest.raises(McpError):
        asyncio.run(transport.request("tools/list"))


def test_parse_http_payload_json_object():
    assert _parse_http_payload('{"jsonrpc":"2.0","id":1,"result":{"ok":true}}') == {"ok": True}


def test_parse_http_payload_sse_body():
    text = 'event: message\ndata: {"jsonrpc":"2.0","id":1,"result":{"a":1}}\n\n'
    assert _parse_http_payload(text) == {"a": 1}


def test_parse_http_payload_error():
    with pytest.raises(McpError):
        _parse_http_payload('{"jsonrpc":"2.0","id":1,"error":{"message":"nope"}}')


def test_parse_http_payload_empty():
    with pytest.raises(McpError):
        _parse_http_payload("   ")


def test_parse_tools_handles_shapes():
    tools = _parse_tools({"tools": [{"name": "a", "description": "d", "inputSchema": {"type": "object"}}, {"name": ""}, "junk"]})
    assert [tool.name for tool in tools] == ["a"]
    assert _parse_tools(None) == []
    assert _parse_tools({"tools": "junk"}) == []


def test_render_result_text_content():
    result = {"content": [{"type": "text", "text": "hello"}, {"type": "text", "text": "world"}]}
    assert _render_result(result) == "hello\nworld"


def test_render_result_truncates():
    result = {"content": [{"type": "text", "text": "x" * 100000}]}
    assert len(_render_result(result)) == 60000


def test_render_result_structured():
    assert _render_result({"structuredContent": {"a": 1}}) == '{"a": 1}'


def test_render_result_plain_json():
    assert _render_result({"other": True}) == '{"other": true}'


@pytest.mark.asyncio
async def test_registry_add_skips_failing_server():
    class Failing:
        name = "bad"

        @property
        def server_label(self):
            return "mcp:bad"

        async def start(self):
            raise McpError("nope")

        async def close(self):
            raise RuntimeError("double fault")

    registry = McpRegistry()
    assert await registry.add(Failing()) is False
    assert registry.enabled() is False


@pytest.mark.asyncio
async def test_registry_resolve_and_call():
    class Fake:
        name = "fake"
        server_label = "mcp:fake"

        def __init__(self):
            self.tools = [McpTool(name="echo", input_schema={"type": "object"})]

        async def start(self):
            return None

        async def call(self, name, arguments):
            return json.dumps(arguments)

        async def close(self):
            return None

    registry = McpRegistry()
    assert await registry.add(Fake()) is True
    resolved = registry.resolve("echo")
    assert resolved is not None
    server, tool = resolved
    assert await registry.call(server, tool, {"a": 1}) == '{"a": 1}'
    assert registry.resolve("missing") is None
    await registry.close()
    assert registry.enabled() is False
