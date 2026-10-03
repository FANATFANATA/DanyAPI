from __future__ import annotations

import html
import logging
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote_plus

import httpx

from .transport import MAX_TOOL_RESULT_CHARS, HttpTransport, McpError, McpServer, McpTool, StdioTransport

log = logging.getLogger("danyapi.mcp")

SEARCH_TOOL_NAME = "web_search"
SEARCH_TOOL_DESCRIPTION = "Search the web and return the top results with titles, URLs and snippets."
SEARCH_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "The search query."},
        "max_results": {"type": "integer", "description": "How many results to return, 1 to 8.", "default": 5, "minimum": 1, "maximum": 8},
    },
    "required": ["query"],
}

DDG_HTML_URL = "https://html.duckduckgo.com/html/?q={query}"
DDG_REQUEST_TIMEOUT = 20.0
DDG_RESULT_RE = re.compile(r'<a[^>]+class="result__a"[^>]+href="([^"]*)"[^>]*>(.*?)</a>', re.DOTALL)
DDG_SNIPPET_RE = re.compile(r'<a[^>]+class="result__snippet"[^>]*>(.*?)</a>', re.DOTALL)
DDG_TAG_RE = re.compile(r"<[^>]+>")
MAX_SEARCH_RESULTS = 8


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str


def _strip_tags(markup: str) -> str:
    return html.unescape(DDG_TAG_RE.sub("", markup)).strip()


def _decode_ddg_url(href: str) -> str:
    if href.startswith("//"):
        href = "https:" + href
    marker = "uddg="
    if marker in href:
        tail = href.split(marker, 1)[1]
        encoded = tail.split("&", 1)[0]
        from urllib.parse import unquote

        return unquote(encoded)
    return href


async def duckduckgo_search(query: str, max_results: int = 5) -> list[SearchResult]:
    if not query.strip():
        return []
    limit = max(1, min(MAX_SEARCH_RESULTS, max_results if isinstance(max_results, int) else 5))
    url = DDG_HTML_URL.format(query=quote_plus(query))
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    async with httpx.AsyncClient(timeout=DDG_REQUEST_TIMEOUT, follow_redirects=True) as client:
        resp = await client.get(url, headers=headers)
    if resp.status_code >= 400:
        raise McpError(f"duckduckgo answered {resp.status_code}")
    titles = DDG_RESULT_RE.findall(resp.text)
    snippets = list(DDG_SNIPPET_RE.findall(resp.text))
    results: list[SearchResult] = []
    for index, (href, title_markup) in enumerate(titles[:limit]):
        snippet = _strip_tags(snippets[index]) if index < len(snippets) else ""
        results.append(SearchResult(title=_strip_tags(title_markup), url=_decode_ddg_url(href), snippet=snippet))
    return results


def render_search_results(results: list[SearchResult]) -> str:
    if not results:
        return "no results found"
    lines = []
    for position, result in enumerate(results, 1):
        lines.append(f"{position}. {result.title}\n   {result.url}\n   {result.snippet}")
    return "\n".join(lines)[:MAX_TOOL_RESULT_CHARS]


class BuiltinSearchServer:
    name = "search"

    @property
    def server_label(self) -> str:
        return "builtin:search"

    @property
    def tools(self) -> list[McpTool]:
        return [McpTool(name=SEARCH_TOOL_NAME, description=SEARCH_TOOL_DESCRIPTION, input_schema=SEARCH_INPUT_SCHEMA)]

    async def start(self) -> None:
        return None

    async def call(self, name: str, arguments: dict[str, Any]) -> str:
        if name != SEARCH_TOOL_NAME:
            raise McpError(f"unknown builtin tool: {name}")
        query = arguments.get("query")
        if not isinstance(query, str) or not query.strip():
            raise McpError("query must be a non-empty string")
        raw_limit = arguments.get("max_results")
        limit = raw_limit if isinstance(raw_limit, int) and not isinstance(raw_limit, bool) else 5
        results = await duckduckgo_search(query, limit)
        return render_search_results(results)

    async def close(self) -> None:
        return None


SERVER_TYPES = (McpServer, BuiltinSearchServer)


class McpRegistry:
    def __init__(self) -> None:
        self._servers: list[Any] = []

    @property
    def servers(self) -> list[Any]:
        return list(self._servers)

    def enabled(self) -> bool:
        return bool(self._servers)

    async def add(self, server: Any) -> bool:
        try:
            await server.start()
        except (McpError, OSError) as exc:
            label = getattr(server, "server_label", getattr(server, "name", "mcp"))
            log.warning("%s failed to start and is skipped: %s", label, exc)
            await _safe_close(server)
            return False
        self._servers.append(server)
        return True

    def all_tools(self) -> list[tuple[Any, McpTool]]:
        pairs: list[tuple[Any, McpTool]] = []
        for server in self._servers:
            for tool in server.tools:
                pairs.append((server, tool))
        return pairs

    def resolve(self, name: str) -> tuple[Any, McpTool] | None:
        for server, tool in self.all_tools():
            if tool.name == name:
                return server, tool
        return None

    async def call(self, server: Any, tool: McpTool, arguments: dict[str, Any]) -> str:
        return await server.call(tool.name, arguments)

    async def close(self) -> None:
        for server in self._servers:
            await _safe_close(server)
        self._servers.clear()


async def _safe_close(server: Any) -> None:
    try:
        await server.close()
    except Exception as exc:
        log.debug("mcp server close failed: %s", exc)


def server_from_config(name: str, spec: str) -> Any:
    if spec.startswith("http://") or spec.startswith("https://"):
        return McpServer(name, HttpTransport(spec))
    return McpServer(name, StdioTransport(spec))
