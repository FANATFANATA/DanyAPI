from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import shlex
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import httpx

log = logging.getLogger("danyapi.mcp")

JSONRPC_VERSION = "2.0"
PROTOCOL_VERSION = "2025-06-18"
MAX_TOOL_RESULT_CHARS = 60000
STARTUP_TIMEOUT_SEC = 20.0
CALL_TIMEOUT_SEC = 60.0


class McpError(Exception):
    pass


@dataclass
class McpTool:
    name: str
    description: str = ""
    input_schema: dict[str, Any] = field(default_factory=dict)

    def openai_schema(self) -> dict[str, Any]:
        schema = self.input_schema if isinstance(self.input_schema, dict) else {}
        if not isinstance(schema.get("type"), str):
            schema = {"type": "object", **schema}
        function: dict[str, Any] = {"name": self.name, "parameters": schema}
        if self.description:
            function["description"] = self.description
        return {"type": "function", "function": function}


class McpTransport(ABC):
    @abstractmethod
    async def start(self) -> None:
        raise NotImplementedError

    @abstractmethod
    async def request(self, method: str, params: dict[str, Any] | None = None, timeout: float | None = None) -> Any:
        raise NotImplementedError

    @abstractmethod
    async def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        raise NotImplementedError

    @abstractmethod
    async def close(self) -> None:
        raise NotImplementedError


class StdioTransport(McpTransport):
    def __init__(self, command: str, env: dict[str, str] | None = None, cwd: str | None = None) -> None:
        try:
            argv = shlex.split(command)
        except ValueError as exc:
            raise McpError(f"mcp stdio command is not parseable: {exc}") from exc
        if not argv:
            raise McpError("mcp stdio command is empty")
        self._argv = argv
        self._env = env or {}
        self._cwd = cwd or None
        self._proc: asyncio.subprocess.Process | None = None
        self._next_id = 0
        self._pending: dict[int, asyncio.Future[Any]] = {}
        self._reader_task: asyncio.Task[None] | None = None
        self._write_lock = asyncio.Lock()
        self._closed = False

    def _message_env(self) -> dict[str, str]:
        environment = dict(os.environ)
        environment.update(self._env)
        return environment

    async def start(self) -> None:
        try:
            self._proc = await asyncio.create_subprocess_exec(
                *self._argv,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
                env=self._message_env(),
                cwd=self._cwd,
            )
        except (OSError, ValueError) as exc:
            raise McpError(f"mcp stdio server failed to start: {' '.join(self._argv)}: {exc}") from exc
        self._reader_task = asyncio.create_task(self._read_loop())

    async def _read_loop(self) -> None:
        proc = self._proc
        stdout = proc.stdout if proc is not None else None
        if stdout is None:
            return
        while True:
            try:
                line = await stdout.readline()
            except OSError:
                break
            if not line:
                break
            text = line.decode("utf-8", "replace").strip()
            if not text:
                continue
            try:
                message = json.loads(text)
            except ValueError:
                continue
            if not isinstance(message, dict):
                continue
            pending_id = message.get("id")
            if isinstance(pending_id, bool) or not isinstance(pending_id, int):
                continue
            future = self._pending.pop(pending_id, None)
            if future is None or future.done():
                continue
            if "error" in message:
                future.set_exception(McpError(_error_text(message["error"])))
            else:
                future.set_result(message.get("result"))

    def _next(self) -> int:
        self._next_id += 1
        return self._next_id

    async def request(self, method: str, params: dict[str, Any] | None = None, timeout: float | None = None) -> Any:
        if self._proc is None or self._proc.stdin is None:
            raise McpError("mcp stdio transport is not started")
        call_id = self._next()
        message: dict[str, Any] = {"jsonrpc": JSONRPC_VERSION, "id": call_id, "method": method}
        if params is not None:
            message["params"] = params
        future: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        self._pending[call_id] = future
        payload = json.dumps(message, ensure_ascii=False, separators=(",", ":")) + "\n"
        try:
            async with self._write_lock:
                self._proc.stdin.write(payload.encode("utf-8"))
                await self._proc.stdin.drain()
            return await asyncio.wait_for(future, timeout if timeout is not None else CALL_TIMEOUT_SEC)
        except asyncio.TimeoutError as exc:
            self._pending.pop(call_id, None)
            raise McpError(f"mcp call {method} timed out") from exc
        finally:
            self._pending.pop(call_id, None)

    async def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        if self._proc is None or self._proc.stdin is None:
            return
        message: dict[str, Any] = {"jsonrpc": JSONRPC_VERSION, "method": method}
        if params is not None:
            message["params"] = params
        payload = json.dumps(message, ensure_ascii=False, separators=(",", ":")) + "\n"
        try:
            async with self._write_lock:
                self._proc.stdin.write(payload.encode("utf-8"))
                await self._proc.stdin.drain()
        except (OSError, ValueError):
            pass

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for future in self._pending.values():
            if not future.done():
                future.set_exception(McpError("mcp stdio transport closed"))
        self._pending.clear()
        if self._reader_task is not None:
            self._reader_task.cancel()
        proc = self._proc
        if proc is not None and proc.stdin is not None:
            try:
                proc.stdin.close()
            except OSError:
                pass
        if proc is not None:
            try:
                await asyncio.wait_for(proc.wait(), 5.0)
            except asyncio.TimeoutError:
                with contextlib.suppress(ProcessLookupError):
                    proc.kill()
                with contextlib.suppress(Exception):
                    await proc.wait()
        if self._reader_task is not None:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._reader_task

    def alive(self) -> bool:
        return self._proc is not None and self._proc.returncode is None


class HttpTransport(McpTransport):
    def __init__(self, url: str, headers: dict[str, str] | None = None, timeout: float | None = None) -> None:
        self._url = url
        self._headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
        if headers:
            self._headers.update(headers)
        self._client = httpx.AsyncClient(timeout=timeout if timeout is not None else CALL_TIMEOUT_SEC)
        self._next_id = 0
        self._session_header: str | None = None

    async def start(self) -> None:
        result = await self.request(
            "initialize", {"protocolVersion": PROTOCOL_VERSION, "capabilities": {}, "clientInfo": {"name": "danyapi", "version": "1.0"}}
        )
        session = result.get("session") if isinstance(result, dict) else None
        if isinstance(session, str) and session:
            self._session_header = session
            self._headers["Mcp-Session-Id"] = session
        await self.notify("notifications/initialized")

    async def request(self, method: str, params: dict[str, Any] | None = None, timeout: float | None = None) -> Any:
        self._next_id += 1
        message: dict[str, Any] = {"jsonrpc": JSONRPC_VERSION, "id": self._next_id, "method": method}
        if params is not None:
            message["params"] = params
        try:
            resp = await self._client.post(self._url, json=message, headers=self._headers, timeout=timeout)
        except httpx.HTTPError as exc:
            raise McpError(f"mcp http request failed: {exc}") from exc
        if resp.status_code >= 400:
            raise McpError(f"mcp http server answered {resp.status_code}")
        return _parse_http_payload(resp.text)

    async def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        message: dict[str, Any] = {"jsonrpc": JSONRPC_VERSION, "method": method}
        if params is not None:
            message["params"] = params
        try:
            await self._client.post(self._url, json=message, headers=self._headers)
        except httpx.HTTPError:
            pass

    async def close(self) -> None:
        with contextlib.suppress(Exception):
            await self._client.aclose()


def _parse_http_payload(text: str) -> Any:
    stripped = text.strip()
    if not stripped:
        raise McpError("mcp http server answered an empty body")
    if stripped.startswith("{"):
        try:
            message = json.loads(stripped)
        except ValueError as exc:
            raise McpError("mcp http server answered invalid JSON") from exc
    else:
        message = None
        for line in stripped.splitlines():
            if not line.startswith("data:"):
                continue
            payload = line[len("data:") :].strip()
            if not payload or payload == "[DONE]":
                continue
            try:
                candidate = json.loads(payload)
            except ValueError:
                continue
            if isinstance(candidate, dict) and ("result" in candidate or "error" in candidate):
                message = candidate
                break
        if message is None:
            raise McpError("mcp http response carries no JSON-RPC result")
    if not isinstance(message, dict):
        raise McpError("mcp http response is not a JSON-RPC message")
    if "error" in message:
        raise McpError(_error_text(message["error"]))
    return message.get("result")


def _error_text(error: Any) -> str:
    if isinstance(error, dict):
        message = error.get("message")
        if isinstance(message, str) and message:
            return message
    if isinstance(error, str) and error:
        return error
    return "mcp server error"


class McpServer:
    def __init__(self, name: str, transport: McpTransport) -> None:
        self.name = name
        self.transport = transport
        self.tools: list[McpTool] = []
        self.server_label = f"mcp:{name}"

    async def start(self) -> None:
        await self.transport.start()
        if isinstance(self.transport, StdioTransport):
            result = await self.transport.request(
                "initialize",
                {"protocolVersion": PROTOCOL_VERSION, "capabilities": {}, "clientInfo": {"name": "danyapi", "version": "1.0"}},
                timeout=STARTUP_TIMEOUT_SEC,
            )
            await self.transport.notify("notifications/initialized")
        result = await self.transport.request("tools/list", {}, timeout=STARTUP_TIMEOUT_SEC)
        self.tools = _parse_tools(result)
        if not self.tools:
            raise McpError(f"{self.server_label} exposes no tools")

    async def call(self, name: str, arguments: dict[str, Any]) -> str:
        result = await self.transport.request("tools/call", {"name": name, "arguments": arguments})
        return _render_result(result)

    async def close(self) -> None:
        try:
            await self.transport.close()
        except Exception as exc:
            log.debug("mcp server %s close failed: %s", self.name, exc)


def _parse_tools(result: Any) -> list[McpTool]:
    if not isinstance(result, dict):
        return []
    raw_tools = result.get("tools")
    if not isinstance(raw_tools, list):
        return []
    tools: list[McpTool] = []
    for raw in raw_tools:
        if not isinstance(raw, dict):
            continue
        name = raw.get("name")
        if not isinstance(name, str) or not name:
            continue
        description = raw.get("description")
        schema = raw.get("inputSchema")
        tools.append(McpTool(name, description if isinstance(description, str) else "", schema if isinstance(schema, dict) else {}))
    return tools


def _render_result(result: Any) -> str:
    if isinstance(result, dict) and isinstance(result.get("content"), list):
        parts: list[str] = []
        for item in result["content"]:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text" and isinstance(item.get("text"), str):
                parts.append(item["text"])
            elif item.get("type") == "resource" and isinstance(item.get("resource"), dict):
                resource = item["resource"]
                text = resource.get("text")
                if isinstance(text, str):
                    parts.append(text)
        joined = "\n".join(parts).strip()
        if joined:
            return joined[:MAX_TOOL_RESULT_CHARS]
    if isinstance(result, dict) and isinstance(result.get("structuredContent"), dict):
        return json.dumps(result["structuredContent"], ensure_ascii=False)[:MAX_TOOL_RESULT_CHARS]
    try:
        return json.dumps(result, ensure_ascii=False, default=str)[:MAX_TOOL_RESULT_CHARS]
    except (TypeError, ValueError):
        return str(result)[:MAX_TOOL_RESULT_CHARS]
