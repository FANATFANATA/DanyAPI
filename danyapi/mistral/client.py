from __future__ import annotations

import codecs
import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

import httpx

log = logging.getLogger("danyapi.mistral")

CHAT_BASE = "https://chat.mistral.ai"
AUTH_BASE = "https://auth.mistral.ai"
BOOTSTRAP_PATH = "/self-service/registration/api"
LOGIN_FLOW_PATH = "/self-service/login/api"
LOGIN_PATH = "/self-service/login"
NEW_CHAT_PATH = "/api/trpc/message.newChat?batch=1"
CHAT_PATH = "/api/chat"

APP_UA = "le-chat-mobile/2.8.0 (build:20800191; os_name:android; device_category:smartphone; device_model:unknown; device_manufacturer:unknown)"

AUTH_ERROR_STATUSES = frozenset({401, 403})
RATE_LIMIT_CODE = 6200
RETRYABLE_STATUSES = frozenset({408, 425, 429, 500, 502, 503, 504})

MAX_SSE_LINE_CHARS = 1024 * 1024

SIGNIN_MARKER = "an account is now required"

MODEL_CATALOG: tuple[dict, ...] = (
    {"id": "mistral-small-latest", "name": "Mistral Small 3.2", "owned_by": "mistral", "model_type": "chat"},
    {"id": "mistral-medium-latest", "name": "Mistral Medium 3.1", "owned_by": "mistral", "model_type": "chat"},
    {"id": "mistral-large-latest", "name": "Mistral Large 2.1", "owned_by": "mistral", "model_type": "chat"},
    {"id": "magistral-medium-latest", "name": "Magistral Medium 1.2", "owned_by": "mistral", "model_type": "chat"},
    {"id": "codestral-latest", "name": "Codestral 2508", "owned_by": "mistral", "model_type": "chat"},
    {"id": "mistral-ocr-latest", "name": "Mistral OCR 2503", "owned_by": "mistral", "model_type": "chat"},
)

DEFAULT_MODEL = "mistral-small-latest"


def catalog_models() -> tuple[dict, ...]:
    return MODEL_CATALOG


class MistralAuthError(Exception):
    pass


class MistralChatError(Exception):
    def __init__(self, code: int, message: str, *, rate_limited: bool = False, auth: bool = False, retryable: bool = False) -> None:
        super().__init__(f"Mistral Le Chat error {code}: {message}")
        self.code = code
        self.message = message
        self.rate_limited = rate_limited
        self.is_auth = auth
        self.is_retryable = retryable


def _error(status: int, payload: Any, default: str = "") -> MistralChatError:
    message = default
    rate_limited = False
    retryable = status in RETRYABLE_STATUSES and status != 429
    if isinstance(payload, dict):
        raw = payload.get("detail") or payload.get("message")
        if isinstance(raw, str) and raw:
            message = raw
        elif isinstance(raw, dict):
            inner = raw.get("message")
            if isinstance(inner, str) and inner:
                message = inner
    if status == 429:
        rate_limited = True
        retryable = False
    if not message:
        message = f"upstream returned {status}"
    return MistralChatError(
        status,
        message,
        rate_limited=rate_limited,
        auth=status in AUTH_ERROR_STATUSES,
        retryable=retryable,
    )


def _safe_json(resp: httpx.Response) -> Any:
    try:
        return resp.json()
    except ValueError:
        return None


class MistralEvent:
    __slots__ = ("delta", "finish")

    def __init__(self) -> None:
        self.delta = ""
        self.finish: str | None = None


def _new_chat_body(message: str, files: list[dict] | None = None) -> dict:
    return {
        "0": {
            "json": {
                "files": files or [],
                "content": [{"type": "text", "text": message}],
                "transcriptionsMetadata": [],
                "features": [],
                "integrations": [],
                "libraries": [],
                "productType": "chat",
                "projectId": None,
                "incognito": True,
            }
        }
    }


def parse_line(line: str, state: dict) -> MistralEvent | None:
    stripped = line.strip()
    if not stripped or ":" not in stripped:
        return None
    colon = stripped.index(":")
    try:
        line_type = int(stripped[:colon])
    except ValueError:
        return None
    payload = stripped[colon + 1 :]
    if not payload or payload == "null":
        if line_type == 8:
            event = MistralEvent()
            event.finish = "stop"
            return event
        return None
    try:
        data = json.loads(payload)
    except ValueError:
        return None
    if not isinstance(data, dict):
        return None
    body = data.get("json", data)
    if not isinstance(body, dict):
        return None
    if line_type == 6:
        code = body.get("internalCode", 0)
        message = str(body.get("message") or body)
        retry = body.get("retryAfterSeconds", 0)
        if code == RATE_LIMIT_CODE or retry:
            raise MistralChatError(429, f"le chat rate limited, retry after {retry} seconds", rate_limited=True)
        raise MistralChatError(502, message)
    if line_type == 8:
        event = MistralEvent()
        event.finish = "stop"
        return event
    if line_type != 15:
        return None
    kind = body.get("type")
    if kind == "bootstrap":
        chat = body.get("chat")
        if isinstance(chat, dict) and chat.get("id"):
            state["chat_id"] = chat["id"]
        return None
    if kind != "message":
        return None
    event = MistralEvent()
    for patch in body.get("patches") or []:
        if not isinstance(patch, dict):
            continue
        op = patch.get("op")
        path = patch.get("path", "")
        value = patch.get("value")
        if op == "replace" and path == "/":
            state["assistant_seen"] = True
        elif op == "replace" and "/contentChunks" in path and isinstance(value, list):
            for item in value:
                if isinstance(item, dict) and item.get("type") == "text" and isinstance(item.get("text"), str):
                    event.delta = item["text"]
        elif op == "append" and "/text" in path and isinstance(value, str):
            event.delta = value
    if event.delta:
        state.setdefault("chunks", []).append(event.delta)
    return event


def _split_login(raw: str) -> tuple[str, str]:
    email, separator, password = raw.partition(":")
    if not separator or not email.strip() or not password:
        raise ValueError("MISTRAL_LOGIN must be email:password")
    return email.strip(), password


async def _iter_lines(resp: httpx.Response) -> AsyncIterator[str]:
    decoder = codecs.getincrementaldecoder("utf-8")("replace")
    buffer = ""
    async for chunk in resp.aiter_bytes():
        if not chunk:
            continue
        buffer += decoder.decode(chunk)
        start = 0
        while True:
            end = buffer.find("\n", start)
            if end == -1:
                break
            text = buffer[start:end]
            start = end + 1
            if text.strip():
                yield text
        if start:
            buffer = buffer[start:]
        if len(buffer) > MAX_SSE_LINE_CHARS:
            raise MistralChatError(502, f"le chat sent a line above the {MAX_SSE_LINE_CHARS} character limit without a newline")
    buffer += decoder.decode(b"", final=True)
    if buffer.strip():
        yield buffer


class MistralChatClient:
    def __init__(
        self,
        timeout: float = 60.0,
        user_agent: str = APP_UA,
        session_token: str | None = None,
        login: str | None = None,
    ) -> None:
        self.user_agent = user_agent
        self.timeout = float(timeout)
        self._bootstrapped = False
        self._token = session_token or None
        self._login_email, self._login_password = _split_login(login) if login else ("", "")
        self.http = httpx.AsyncClient(
            base_url=CHAT_BASE,
            headers={
                "User-Agent": user_agent,
                "Accept-Language": "en-US",
            },
            timeout=httpx.Timeout(timeout, read=max(float(timeout) * 5, 300.0)),
            follow_redirects=True,
            limits=httpx.Limits(max_keepalive_connections=20, keepalive_expiry=30.0),
        )

    async def aclose(self) -> None:
        await self.http.aclose()

    @property
    def can_auth(self) -> bool:
        return bool(self._token or self._login_email)

    async def _bootstrap(self) -> None:
        if self._bootstrapped:
            return
        try:
            await self.http.get(
                AUTH_BASE + BOOTSTRAP_PATH,
                headers={"Accept": "application/json"},
            )
        except (httpx.HTTPError, OSError, RuntimeError) as exc:
            log.debug("mistral cookie bootstrap failed: %s", exc)
        self._bootstrapped = True

    async def login(self) -> str:
        if not self._login_email:
            raise MistralAuthError("no mistral login credentials configured")
        await self._bootstrap()
        flow_resp = await self.http.get(
            AUTH_BASE + LOGIN_FLOW_PATH,
            headers={"Accept": "application/json"},
        )
        if flow_resp.status_code >= 400:
            raise MistralAuthError(f"login flow creation failed with {flow_resp.status_code}")
        try:
            flow_id = flow_resp.json()["id"]
        except (ValueError, KeyError) as exc:
            raise MistralAuthError("login flow returned no flow id") from exc
        resp = await self.http.post(
            f"{AUTH_BASE}{LOGIN_PATH}?flow={flow_id}",
            json={
                "method": "password",
                "identifier": self._login_email,
                "password": self._login_password,
                "csrf_token": "",
            },
            headers={"Accept": "application/json", "Content-Type": "application/json"},
        )
        if resp.status_code >= 400:
            detail = _kratos_error(resp)
            raise MistralAuthError(detail or f"login failed with {resp.status_code}")
        try:
            token = resp.json().get("session_token")
        except ValueError as exc:
            raise MistralAuthError("login returned non-JSON") from exc
        if not token:
            raise MistralAuthError("login returned no session token")
        self._token = token
        return token

    async def token(self) -> str:
        if self._token:
            return self._token
        return await self.login()

    def invalidate_token(self) -> None:
        self._token = None

    async def _auth_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {await self.token()}"}

    async def _new_chat(self, message: str, files: list[dict] | None = None) -> str:
        await self._bootstrap()
        resp = await self.http.post(
            NEW_CHAT_PATH,
            json=_new_chat_body(message, files),
            headers={"Content-Type": "application/json", "Accept": "application/json", **await self._auth_headers()},
        )
        if resp.status_code in AUTH_ERROR_STATUSES:
            raise MistralChatError(resp.status_code, "le chat rejected the session token", auth=True)
        if resp.status_code >= 400:
            raise _error(resp.status_code, _safe_json(resp), "chat creation failed")
        try:
            payload = resp.json()
        except ValueError as exc:
            raise MistralChatError(502, "chat creation returned non-JSON") from exc
        if isinstance(payload, list) and payload:
            payload = payload[0]
        data = payload.get("result", payload) if isinstance(payload, dict) else None
        if isinstance(data, dict):
            data = data.get("data", data)
        if isinstance(data, dict) and isinstance(data.get("json"), dict):
            data = data["json"]
        chat_id = None
        if isinstance(data, dict):
            raw = data.get("chatId") or data.get("id")
            if isinstance(raw, str) and raw:
                chat_id = raw
            else:
                chat = data.get("chat")
                if isinstance(chat, dict) and isinstance(chat.get("id"), str):
                    chat_id = chat["id"]
        if chat_id is None:
            raise MistralChatError(502, "chat creation returned no chat id")
        return chat_id

    async def _post_chat(self, chat_id: str) -> httpx.Response:
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")
        body = {
            "mode": "start",
            "chatId": chat_id,
            "platform": "mobile",
            "clientPromptData": {"currentDate": now},
            "supportedTaskCallbacks": [],
            "features": [],
            "libraries": [],
            "integrations": [],
            "disabledFeatures": ["memory-inference"],
        }
        request = self.http.build_request(
            "POST",
            CHAT_PATH,
            json=body,
            headers={"Accept": "text/event-stream", "Content-Type": "application/json", **await self._auth_headers()},
        )
        resp = await self.http.send(request, stream=True)
        if resp.status_code in AUTH_ERROR_STATUSES:
            await resp.aclose()
            raise MistralChatError(resp.status_code, "le chat rejected the session token", auth=True)
        if resp.status_code >= 400:
            payload = None
            try:
                await resp.aread()
                payload = _safe_json(resp)
            except (httpx.HTTPError, ValueError):
                payload = None
            await resp.aclose()
            error = _error(resp.status_code, payload)
            if resp.status_code == 429 and _rate_limited_code(payload):
                error = MistralChatError(429, error.message, rate_limited=True)
            raise error
        return resp

    @asynccontextmanager
    async def chat(self, message: str, files: list[dict] | None = None) -> AsyncIterator[AsyncIterator[MistralEvent]]:
        chat_id = await self._new_chat(message, files)
        resp = await self._post_chat(chat_id)
        try:
            yield self._events(resp)
        finally:
            await resp.aclose()

    async def _events(self, resp: httpx.Response) -> AsyncIterator[MistralEvent]:
        state: dict[str, Any] = {}
        generator = _iter_lines(resp)
        while True:
            try:
                line = await generator.__anext__()
            except StopAsyncIteration:
                break
            except httpx.ReadTimeout:
                if state.get("chunks"):
                    break
                raise
            event = parse_line(line, state)
            if event is None:
                continue
            yield event
            if event.finish is not None:
                break
        text = "".join(state.get("chunks") or "")
        if SIGNIN_MARKER in text.lower():
            raise MistralChatError(401, "le chat answered with the sign-in wall, the session is not authorized", auth=True)

    async def check_auth(self) -> bool:
        if not self.can_auth:
            return False
        try:
            await self.token()
        except MistralAuthError as exc:
            log.warning("mistral login failed: %s", exc)
            return False
        try:
            async with self.chat("hi") as events:
                async for event in events:
                    if event.delta or event.finish is not None:
                        return True
        except (MistralChatError, httpx.HTTPError, OSError, RuntimeError) as exc:
            log.debug("mistral check_auth probe failed: %s", exc)
            return False
        return True

    async def relogin(self) -> bool:
        if not self._login_email:
            return False
        self.invalidate_token()
        try:
            await self.login()
        except MistralAuthError as exc:
            log.warning("mistral relogin failed: %s", exc)
            return False
        return True

    async def fetch_models(self) -> list[dict]:
        return [dict(entry) for entry in catalog_models()]


def _kratos_error(resp: httpx.Response) -> str:
    payload = _safe_json(resp)
    if not isinstance(payload, dict):
        return ""
    messages: list[str] = []
    ui = payload.get("ui")
    if isinstance(ui, dict):
        for item in ui.get("messages") or []:
            if isinstance(item, dict) and item.get("type") == "error":
                text = item.get("text")
                if isinstance(text, str):
                    messages.append(text)
        for node in ui.get("nodes") or []:
            if isinstance(node, dict):
                for item in node.get("messages") or []:
                    if isinstance(item, dict) and item.get("type") == "error":
                        text = item.get("text")
                        if isinstance(text, str):
                            messages.append(text)
    return "; ".join(messages)


def _rate_limited_code(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    return payload.get("code") == RATE_LIMIT_CODE
