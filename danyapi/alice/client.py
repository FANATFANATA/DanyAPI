from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import re
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any, cast

import websockets
from websockets.exceptions import WebSocketException

log = logging.getLogger("danyapi.alice")

WS_URL = "wss://uniproxy.alice.yandex.net/uni.ws"
LEGACY_WS_URL = "wss://uniproxy.alice.ya.ru/uni.ws"
ORIGIN = "https://alice.yandex.ru"
ORIGIN_DOMAIN = "alice.yandex.ru"

APP_ID = "ru.yandex.web.desktop"
APP_VERSION = "1.0.281-home-static/alice-web/15"
PLATFORM = "windows"
LANGUAGE = "ru"
TIMEZONE = "Europe/Moscow"
TOPIC = "desktopgeneral"
AUDIO_MIME = "audio/webm;codecs=opus"
AUDIO_FORMAT = "audio/ogg;codecs=opus"
OS_VERSION = "mozilla/5.0 (windows nt 10.0; win64; x64) applewebkit/537.36 (khtml, like gecko) chrome/151.0.0.0 safari/537.36"

SUPPORTED_FEATURES = [
    "open_link",
    "server_action",
    "cloud_ui",
    "show_promo",
    "print_text_in_message_view",
]

CONTINUATION_NAME = "@@mm_stack_engine_get_next"
SCENARIO_NAME = "Dialogovo"
STACK_SCENARIO_NAME = "dialogovo"

HANDSHAKE_TIMEOUT = 20.0
FRAME_TIMEOUT = 90.0
PING_TIMEOUT = 30.0
MAX_CONTINUATIONS = 24
HARD_MAX_PROMPT = 6000

PLACEHOLDER_TEXTS = frozenset({"одну секунду...", "одну секунду", "секунду...", "подождите", "подумаю"})

_VERSION_RE = re.compile(r'"production",\s*version:"(\S+?)"')
_VERSION_FALLBACK_RE = re.compile(r'"version"\s*:\s*"([^"]{4,60})"')

CONNECT_IN_PROGRESS = 1000
CONNECT_FAILED = 1003
CONNECT_DROPPED = 1004
CONNECT_REFUSED = 1005
CONNECT_LOST = 1006
CONNECT_FATAL = 1011
AUTH_REJECTED = 1002
GOAWAY = 1007
EMPTY_ANSWER = 1008
UPSTREAM_TIMEOUT = 1009

TERMINAL_ERRORS = {CONNECT_FATAL, AUTH_REJECTED, EMPTY_ANSWER}
RETRYABLE_ERRORS = {CONNECT_DROPPED, CONNECT_LOST, CONNECT_IN_PROGRESS, GOAWAY, UPSTREAM_TIMEOUT}

AUTH_FINISH_MARKERS = ("authorization key", "ключ авторизации", "scope is empty", "credentials doesn't match")
EMPTY_MARKERS = ("не могу с вами познакомиться", "не могу с Вами познакомиться", "не знаю, кто вы")
TIMEOUT_MARKERS = ("не успела", "попробуйте еще раз", "попробуйте ещё раз", "слишком долго")


def new_id() -> str:
    return str(uuid.uuid4())


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") in ("text", "input_text"):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    if content is None:
        return ""
    return str(content)


def fold_messages(messages: Any) -> str:
    lines: list[str] = []
    for message in messages or []:
        role = getattr(message, "role", None) or (message.get("role") if isinstance(message, dict) else "user")
        if role == "developer":
            role = "system"
        text = _text_of(getattr(message, "content", None) if not isinstance(message, dict) else message.get("content"))
        text = text.strip()
        if not text:
            continue
        lines.append(f"{str(role).capitalize()}: {text}")
    if not lines:
        return "Hello"
    return "\n".join(lines)


def trim_prompt(prompt: str) -> str:
    text = prompt or ""
    if len(text) <= HARD_MAX_PROMPT:
        return text
    return text[-HARD_MAX_PROMPT:]


def is_placeholder(text: str) -> bool:
    stripped = (text or "").strip().lower().rstrip(".")
    if not stripped:
        return False
    return stripped in PLACEHOLDER_TEXTS


def looks_like_refusal(text: str) -> bool:
    lowered = (text or "").lower()
    return any(marker in lowered for marker in AUTH_FINISH_MARKERS + EMPTY_MARKERS + TIMEOUT_MARKERS)


class AliceError(Exception):
    def __init__(self, code: int, message: str, retryable: bool = False) -> None:
        super().__init__(f"Alice error {code}: {message}")
        self.code = code
        self.message = message
        self.retryable = retryable


class _Directive:
    __slots__ = ("name", "payload")

    def __init__(self, raw: Any) -> None:
        if isinstance(raw, dict):
            self.name = raw.get("name") if isinstance(raw.get("name"), str) else ""
            self.payload = raw.get("payload") if isinstance(raw.get("payload"), dict) else {}
        else:
            self.name = ""
            self.payload = {}


class AliceStream:
    def __init__(self) -> None:
        self.content = ""
        self.placeholder = ""
        self.version = ""
        self.done = False
        self.cards: list[dict] = []


class AliceClient:
    def __init__(
        self,
        url: str = WS_URL,
        timeout: float = 60.0,
        app_version: str = APP_VERSION,
    ) -> None:
        self.url = url
        self.timeout = float(timeout)
        self.app_version = app_version
        self._ws: Any = None
        self._reader: asyncio.Task | None = None
        self._sync = asyncio.Event()
        self._frames: asyncio.Queue = asyncio.Queue()
        self._send_lock = asyncio.Lock()
        self._seq = 1
        self._uuid = new_id()
        self._last_request_id: str | None = None
        self._stack_session_id: str | None = None
        self._closed = False

    async def aclose(self) -> None:
        await self._close()

    async def _close(self) -> None:
        if self._closed:
            return
        self._closed = True
        reader = self._reader
        self._reader = None
        if reader is not None:
            reader.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await reader
        ws = self._ws
        self._ws = None
        if ws is not None:
            with contextlib.suppress(Exception):
                await ws.close()

    async def _send(self, message: dict) -> None:
        ws = self._ws
        if ws is None:
            raise AliceError(CONNECT_FAILED, "socket is not connected")
        async with self._send_lock:
            await ws.send(json.dumps(message, ensure_ascii=False))

    def _application(self) -> dict:
        return {
            "app_id": APP_ID,
            "app_version": self.app_version,
            "platform": PLATFORM,
            "os_version": OS_VERSION,
            "uuid": self._uuid,
            "device_id": self._uuid,
            "lang": LANGUAGE,
            "client_time": time.strftime("%Y%m%dT%H%M%S"),
            "timezone": TIMEZONE,
            "timestamp": str(int(time.time())),
        }

    def _payload(self, event: dict, request_id: str, prev_req_id: str | None) -> dict:
        return {
            "application": self._application(),
            "header": {
                "prev_req_id": prev_req_id,
                "sequence_number": None,
                "request_id": request_id,
                "dialog_id": "",
                "dialog_type": 1,
            },
            "request": {
                "event": event,
                "voice_session": False,
                "experiments": [],
                "uniproxy_options": {
                    "background_response_streaming_options": {},
                    "dialog_options": {},
                },
                "additional_options": {
                    "bass_options": {"user_agent": OS_VERSION, "screen_scale_factor": 1},
                    "origin_domain": ORIGIN_DOMAIN,
                    "supported_features": SUPPORTED_FEATURES,
                    "unsupported_features": [],
                },
            },
            "location": {},
            "reset_session": False,
            "environment_state": {"endpoints": [{"id": ORIGIN_DOMAIN, "capabilities": []}]},
            "format": AUDIO_FORMAT,
            "mime": AUDIO_MIME,
            "topic": TOPIC,
            "punctuation": False,
        }

    def _text_input(self, event: dict, prev_req_id: str | None) -> tuple[dict, str]:
        self._seq += 1
        request_id = new_id()
        return (
            {
                "event": {
                    "header": {
                        "namespace": "Vins",
                        "name": "TextInput",
                        "messageId": new_id(),
                        "seqNumber": self._seq,
                    },
                    "payload": self._payload(event, request_id, prev_req_id),
                }
            },
            request_id,
        )

    def _prompt_message(self, prompt: str) -> tuple[dict, str]:
        message, request_id = self._text_input({"type": "text_input", "text": prompt}, None)
        self._last_request_id = request_id
        self._stack_session_id = request_id
        return message, request_id

    def _continuation_message(self) -> tuple[dict, str]:
        prev = self._last_request_id
        stack = self._stack_session_id or prev
        message, request_id = self._text_input(
            {
                "type": "server_action",
                "name": CONTINUATION_NAME,
                "payload": {
                    "@recovery_params": {},
                    "@request_id": prev,
                    "stack_session_id": stack,
                    "@scenario_name": SCENARIO_NAME,
                    "stack_product_scenario_name": STACK_SCENARIO_NAME,
                },
            },
            prev,
        )
        self._last_request_id = request_id
        return message, request_id

    async def _read_loop(self) -> None:
        ws = self._ws
        try:
            async for raw in ws:
                try:
                    root = json.loads(raw)
                except ValueError:
                    continue
                if not isinstance(root, dict):
                    continue
                directive = root.get("directive")
                if not isinstance(directive, dict):
                    continue
                header = directive.get("header")
                header = header if isinstance(header, dict) else {}
                name = header.get("name") if isinstance(header.get("name"), str) else ""
                namespace = header.get("namespace") if isinstance(header.get("namespace"), str) else ""
                if namespace == "System" and name == "Ping":
                    with contextlib.suppress(Exception):
                        await self._send(
                            {
                                "event": {
                                    "header": {
                                        "namespace": "System",
                                        "name": "Pong",
                                        "messageId": new_id(),
                                        "refMessageId": header.get("messageId"),
                                    },
                                    "payload": {},
                                }
                            }
                        )
                    continue
                if namespace == "System" and name == "SynchronizeStateResponse":
                    self._sync.set()
                await self._frames.put(directive)
        except asyncio.CancelledError:
            raise
        except (WebSocketException, OSError, RuntimeError) as exc:
            log.debug("alice socket closed: %s", exc)
            await self._frames.put(None)

    async def _ensure_connected(self) -> None:
        if self._ws is not None and self._reader is not None and not self._reader.done():
            return
        await self._close()
        self._closed = False
        self._sync = asyncio.Event()
        self._frames = asyncio.Queue()
        self._seq = 1
        self._uuid = new_id()
        self._last_request_id = None
        self._stack_session_id = None
        try:
            self._ws = await asyncio.wait_for(
                websockets.connect(self.url, origin=cast("Any", ORIGIN), ping_interval=None, open_timeout=25.0),
                timeout=25.0,
            )
        except (OSError, WebSocketException, asyncio.TimeoutError) as exc:
            raise AliceError(CONNECT_FAILED, f"could not reach alice: {exc}", retryable=True) from exc
        self._reader = asyncio.create_task(self._read_loop())
        await self._send(
            {
                "event": {
                    "header": {
                        "namespace": "System",
                        "name": "SynchronizeState",
                        "messageId": new_id(),
                        "seqNumber": 1,
                    },
                    "payload": {
                        "auth_token": new_id(),
                        "uuid": self._uuid,
                        "vins": {"application": {"app_id": APP_ID, "platform": PLATFORM}},
                    },
                }
            }
        )
        try:
            await asyncio.wait_for(self._sync.wait(), timeout=HANDSHAKE_TIMEOUT)
        except asyncio.TimeoutError as exc:
            await self._close()
            raise AliceError(CONNECT_FAILED, "alice did not confirm the session", retryable=True) from exc

    @staticmethod
    def _extract(directive: dict, stream: AliceStream) -> None:
        payload = directive.get("payload")
        payload = payload if isinstance(payload, dict) else {}
        version = payload.get("version")
        if isinstance(version, str) and version:
            stream.version = version
        response = payload.get("response")
        response = response if isinstance(response, dict) else {}
        text = ""
        for raw in response.get("directives") or []:
            item = _Directive(raw)
            if not isinstance(item.payload, dict):
                continue
            value = item.payload.get("text")
            if isinstance(value, str) and value:
                text = value
        card = response.get("card")
        if isinstance(card, dict):
            value = card.get("text")
            if not text and isinstance(value, str):
                text = value
            if card.get("type"):
                stream.cards.append({"type": card.get("type"), "text": value if isinstance(value, str) else ""})
        for update in response.get("chat_dialog_update") or []:
            if not isinstance(update, dict):
                continue
            request = update.get("add_message_request")
            if not isinstance(request, dict):
                continue
            for message in request.get("messages") or []:
                if not isinstance(message, dict):
                    continue
                content = message.get("content")
                if not isinstance(content, dict):
                    continue
                value = content.get("plain_response_text")
                if not text and isinstance(value, str):
                    text = value
        if text:
            if is_placeholder(text):
                if not stream.placeholder:
                    stream.placeholder = text
            else:
                stream.content = text
                stream.done = True

    async def _pump(self, message: dict, stream: AliceStream) -> None:
        await self._send(message)
        for _ in range(MAX_CONTINUATIONS):
            try:
                directive = await asyncio.wait_for(self._frames.get(), timeout=PING_TIMEOUT)
            except asyncio.TimeoutError as exc:
                raise AliceError(UPSTREAM_TIMEOUT, "alice did not answer in time", retryable=True) from exc
            if directive is None:
                raise AliceError(CONNECT_DROPPED, "alice closed the connection", retryable=True)
            header = directive.get("header") if isinstance(directive.get("header"), dict) else {}
            name = header.get("name") if isinstance(header.get("name"), str) else ""
            namespace = header.get("namespace") if isinstance(header.get("namespace"), str) else ""
            if namespace == "System":
                if name in ("GoAway", "InvalidAuth"):
                    raise AliceError(GOAWAY if name == "GoAway" else AUTH_REJECTED, f"alice sent {name}", retryable=True)
                if name == "EventException":
                    error = directive.get("payload")
                    message_text = ""
                    if isinstance(error, dict) and isinstance(error.get("error"), dict):
                        message_text = str(error["error"].get("message") or "")
                    raise AliceError(CONNECT_FATAL, f"alice error: {message_text[:200] or 'unknown'}")
                continue
            self._extract(directive, stream)
            if stream.done:
                return
            continuation, _ = self._continuation_message()
            await self._send(continuation)
        raise AliceError(EMPTY_ANSWER, "alice produced no answer after continuations", retryable=True)

    async def ask(self, prompt: str) -> AliceStream:
        text = trim_prompt(prompt)
        if not text:
            raise AliceError(EMPTY_ANSWER, "prompt is empty")
        await self._ensure_connected()
        stream = AliceStream()
        message, _ = self._prompt_message(text)
        await self._pump(message, stream)
        if not stream.content:
            if stream.placeholder:
                raise AliceError(EMPTY_ANSWER, f"alice answered with a placeholder: {stream.placeholder}", retryable=True)
            raise AliceError(EMPTY_ANSWER, "alice returned an empty answer", retryable=True)
        if looks_like_refusal(stream.content):
            raise AliceError(EMPTY_ANSWER, f"alice declined to answer: {stream.content[:160]}", retryable=True)
        log.debug("alice answered %d chars, version=%s", len(stream.content), stream.version or "unknown")
        return stream

    async def stream_ask(self, prompt: str) -> AsyncIterator[AliceStream]:
        stream = await self.ask(prompt)
        yield stream

    async def check_auth(self) -> bool:
        try:
            await self._ensure_connected()
        except AliceError:
            return False
        return self._ws is not None

    async def fetch_models(self) -> list[dict]:
        return []

    async def fetch_version(self) -> str:
        try:
            stream = await self.ask("Привет")
        except AliceError:
            return ""
        return stream.version

    async def resolve_version(self) -> str:
        try:
            import httpx
        except ImportError:
            return self.app_version
        url = "https://ya.ru/alisa_davay_pridumaem"
        try:
            async with httpx.AsyncClient(timeout=10.0) as http:
                resp = await http.get(url, headers={"User-Agent": OS_VERSION})
                if resp.status_code != 200:
                    return self.app_version
                text = resp.text
        except (httpx.HTTPError, OSError):
            return self.app_version
        match = _VERSION_RE.search(text) or _VERSION_FALLBACK_RE.search(text)
        if match:
            found = match.group(1)
            if found:
                self.app_version = found
                log.info("alice app version resolved to %s", found)
        return self.app_version
