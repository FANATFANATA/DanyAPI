from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import re
import threading
import uuid
from bisect import bisect_left
from collections.abc import AsyncIterator
from typing import Any

import httpx

from . import attest

log = logging.getLogger("danyapi.duckai")

BASE_URL = "https://duck.ai"
API_ROOT = "/duckchat/v1"
STATUS_PATH = f"{API_ROOT}/status"
CHAT_PATH = f"{API_ROOT}/chat"

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/151.0.0.0 Safari/537.36"

FE_VERSION_FALLBACK = "dev-hash"

FE_VERSION_RE = re.compile(r'data-version-tag="([^"]+)"')
FE_SHA_RE = re.compile(r'data-version-sha="([^"]+)"')

BROWSER_HEADERS = {
    "accept-encoding": "gzip, deflate, br",
    "sec-ch-ua-mobile": "?0",
    "sec-fetch-dest": "empty",
    "sec-fetch-mode": "cors",
    "sec-fetch-site": "same-origin",
    "priority": "u=1, i",
}

STATUS_OK_CODES = frozenset({"0", ""})
AUTH_ERROR_STATUSES = frozenset({401, 403})
CHALLENGE_STATUSES = frozenset({418})
CHALLENGE_TYPES = frozenset({"ERR_CHALLENGE"})
ENTRYPOINT_TYPES = frozenset({"ERR_BN_LIMIT"})
ENTRYPOINT_MARKERS = ("unsupported entrypoint",)
RETRYABLE_STATUSES = frozenset({408, 425, 429, 500, 502, 503, 504})
RETRYABLE_TYPES = frozenset({"ERR_UPSTREAM", "ERR_SERVICE_UNAVAILABLE", "ERR_SERVICE_OFFLINE", "ERR_NETWORK", "ERR_STREAM_NETWORK"})

STREAM_ERROR_STATUS = {
    "ERR_INPUT_LIMIT": 400,
    "ERR_OUTPUT_LIMIT": 400,
    "ERR_CONVERSATION_LIMIT": 400,
    "ERR_IMAGE_GENERATION_LIMIT": 400,
    "ERR_ACCOUNT_LIMIT": 400,
    "ERR_USER_LIMIT": 400,
}

STREAM_PING = "[PING]"
STREAM_CHAT_TITLE = "[CHAT_TITLE:"
STREAM_DONE = "[DONE]"
STOP_REASON_PREFIX = "[STOP_REASON:"

MAX_SSE_LINE_CHARS = 1024 * 1024

DEFAULT_REASONING_EFFORT = "none"
REASONING_EFFORTS = ("none", "low", "medium")

STOP_REASON_MAP = {
    "max_tokens": "length",
    "end_turn": "stop",
    "stop_sequence": "stop",
    "tool_use": "tool_calls",
    "content_filter": "content_filter",
    "refusal": "content_filter",
}

LIMIT_MARKERS = {
    "[LIMIT_ACCOUNT]": "ERR_ACCOUNT_LIMIT",
    "[LIMIT_ENTITY]": "ERR_USER_LIMIT",
    "[LIMIT_CONVERSATION]": "ERR_CONVERSATION_LIMIT",
    "[LIMIT_IMAGE_GENERATIONS]": "ERR_IMAGE_GENERATION_LIMIT",
    "[LIMIT_INPUT]": "ERR_INPUT_LIMIT",
    "[LIMIT_OUTPUT]": "ERR_OUTPUT_LIMIT",
}

MODEL_CATALOG: tuple[dict, ...] = (
    {
        "id": "gpt-5.4-mini",
        "name": "GPT-5.4 mini",
        "owned_by": "duckai",
        "model_type": "chat",
        "provider": "openai",
        "efforts": ("none", "low", "medium"),
    },
    {
        "id": "gpt-5.6-luna",
        "name": "GPT-5.6 Luna",
        "owned_by": "duckai",
        "model_type": "chat",
        "provider": "openai",
        "efforts": ("none", "low"),
    },
    {
        "id": "gpt-5.6-terra",
        "name": "GPT-5.6 Terra",
        "owned_by": "duckai",
        "model_type": "chat",
        "provider": "openai",
        "efforts": ("none", "low", "medium"),
    },
    {
        "id": "gpt-5.6-sol",
        "name": "GPT-5.6 Sol",
        "owned_by": "duckai",
        "model_type": "chat",
        "provider": "openai",
        "efforts": ("none", "low", "medium"),
    },
    {
        "id": "claude-haiku-4-5",
        "name": "Claude Haiku 4.5",
        "owned_by": "duckai",
        "model_type": "chat",
        "provider": "anthropic",
        "efforts": ("none", "low"),
    },
    {
        "id": "claude-sonnet-4-6",
        "name": "Claude Sonnet 4.6",
        "owned_by": "duckai",
        "model_type": "chat",
        "provider": "anthropic",
        "efforts": ("none", "low"),
    },
    {
        "id": "claude-opus-4-8",
        "name": "Claude Opus 4.8",
        "owned_by": "duckai",
        "model_type": "chat",
        "provider": "anthropic",
        "efforts": ("none", "low", "medium"),
    },
    {
        "id": "mistral-small-2603",
        "name": "Mistral Small 4",
        "owned_by": "duckai",
        "model_type": "chat",
        "provider": "mistral",
        "efforts": ("none",),
    },
)

DEFAULT_MODEL = "gpt-5.4-mini"

CATALOG_ANCHOR = '{model:"'
CATALOG_ENTRY_END = '},{model:"'
CATALOG_TAIL_RE = re.compile(r";\s*var")
CATALOG_ID_LIMIT = 256
CATALOG_BODY_LIMIT = 4000
CATALOG_SCAN_LIMIT = 8 * 1024 * 1024
CATALOG_FIELD_RE = {
    "name": re.compile(r'modelName:"([^"]*)"'),
    "variant": re.compile(r'modelVariant:(?:"([^"]*)"|null)'),
    "short_name": re.compile(r'modelShortName:"([^"]*)"'),
    "created_by": re.compile(r'createdBy:"([^"]*)"'),
    "model_type": re.compile(r'modelType:"([^"]*)"'),
    "efforts": re.compile(r"supportedReasoningEffort:\[([^\]]*)\]"),
    "available_to": re.compile(r"availableTo:\[([^\]]*)\]"),
    "cost_rank": re.compile(r"costRank:(\d+)"),
}
ENTRY_SCRIPT_RE = re.compile(r'src="(?P<path>/dist/[^"]*entry\.duckai\.[^"]*\.js)"')
FREE_TIER = "Free"


def _catalog_field(body: str, key: str) -> str | None:
    found = CATALOG_FIELD_RE[key].search(body)
    return found.group(1) if found else None


def _catalog_efforts(raw: str | None) -> tuple[str, ...]:
    if not raw:
        return REASONING_EFFORTS
    efforts = tuple(item.strip().strip('"') for item in raw.split(",") if item.strip())
    known = tuple(item for item in efforts if item in REASONING_EFFORTS)
    return known or REASONING_EFFORTS


def _offsets(text: str, needle: str) -> list[int]:
    found: list[int] = []
    step = len(needle)
    pos = text.find(needle)
    while pos != -1:
        found.append(pos)
        pos = text.find(needle, pos + step)
    return found


def _catalog_terminators(text: str) -> list[int]:
    marks = _offsets(text, CATALOG_ENTRY_END)
    marks += _offsets(text, "}]")
    marks += [match.start() for match in CATALOG_TAIL_RE.finditer(text)]
    marks.sort()
    return marks


def _first_mark(marks: list[int], start: int, limit: int) -> int | None:
    index = bisect_left(marks, start)
    if index < len(marks) and marks[index] <= limit:
        return marks[index]
    return None


def parse_catalog(bundle: str) -> tuple[dict, ...]:
    text = bundle[:CATALOG_SCAN_LIMIT]
    anchors = _offsets(text, CATALOG_ANCHOR)
    if not anchors:
        return ()
    region_start = anchors[0]
    region = text[region_start : anchors[-1] + CATALOG_BODY_LIMIT]
    terminators = [mark + region_start for mark in _catalog_terminators(region)]
    entries: list[dict] = []
    cursor = 0
    for anchor in anchors:
        if anchor < cursor:
            continue
        id_start = anchor + len(CATALOG_ANCHOR)
        quote = text.find('"', id_start)
        if quote == -1 or quote - id_start > CATALOG_ID_LIMIT:
            continue
        body_start = quote + 1
        stop = _first_mark(terminators, body_start, body_start + CATALOG_BODY_LIMIT)
        if stop is None:
            continue
        cursor = stop
        body = text[body_start:stop]
        short_name = _catalog_field(body, "short_name")
        available_to = _catalog_field(body, "available_to")
        if not short_name or not available_to or f".{FREE_TIER}" not in available_to:
            continue
        model_id = text[id_start:quote]
        name = _catalog_field(body, "name") or short_name
        variant = _catalog_field(body, "variant")
        rank = _catalog_field(body, "cost_rank")
        entries.append(
            {
                "id": model_id,
                "name": f"{name} {variant}" if variant else name,
                "owned_by": "duckai",
                "model_type": "chat",
                "provider": (_catalog_field(body, "created_by") or "").lower(),
                "efforts": _catalog_efforts(_catalog_field(body, "efforts")),
                "cost_rank": int(rank) if rank and rank.isdigit() else 0,
            }
        )
    entries.sort(key=lambda entry: (entry["cost_rank"] or 99, entry["id"]))
    for position, entry in enumerate(entries):
        entry["cost_rank"] = position + 1
    return tuple(entries)


def catalog_models() -> tuple[dict, ...]:
    return MODEL_CATALOG


def model_efforts(model: str) -> tuple[str, ...]:
    for entry in MODEL_CATALOG:
        if entry["id"] == model:
            return entry["efforts"]
    return REASONING_EFFORTS


class DuckAIError(Exception):
    def __init__(self, code: int | str, message: str, error_type: str = "") -> None:
        super().__init__(f"Duck.ai error {code}: {message}")
        self.code = code
        self.message = message
        self.error_type = error_type

    @property
    def is_auth(self) -> bool:
        return self.code in AUTH_ERROR_STATUSES or self.error_type in AUTH_ERROR_STATUSES

    @property
    def is_retryable(self) -> bool:
        return self.code in RETRYABLE_STATUSES or self.error_type in RETRYABLE_STATUSES or self.code in RETRYABLE_TYPES or self.error_type in RETRYABLE_TYPES

    @property
    def is_challenge(self) -> bool:
        return self.code in CHALLENGE_STATUSES or self.error_type in CHALLENGE_STATUSES or self.code in CHALLENGE_TYPES or self.error_type in CHALLENGE_TYPES

    @property
    def is_entrypoint(self) -> bool:
        if self.code in ENTRYPOINT_TYPES or self.error_type in ENTRYPOINT_TYPES:
            return True
        lowered = (self.message or "").lower()
        return any(marker in lowered for marker in ENTRYPOINT_MARKERS)


def normalize_effort_in(allowed: tuple[str, ...], effort: str | None) -> str:
    wanted = (effort or DEFAULT_REASONING_EFFORT).strip().lower()
    if wanted not in REASONING_EFFORTS:
        return DEFAULT_REASONING_EFFORT
    if wanted in allowed:
        return wanted
    known = next((item for item in allowed if item in REASONING_EFFORTS), "")
    return known or DEFAULT_REASONING_EFFORT


def normalize_effort(model: str, effort: str | None) -> str:
    return normalize_effort_in(model_efforts(model), effort)


def normalize_finish_reason(value: Any) -> str:
    if isinstance(value, str) and value:
        return STOP_REASON_MAP.get(value, "stop")
    return "stop"


class DuckAIEvent:
    __slots__ = ("delta", "finish", "limit", "reasoning", "refusal", "sources", "title", "tool_calls")

    def __init__(self) -> None:
        self.delta = ""
        self.reasoning = ""
        self.finish: str | None = None
        self.tool_calls: list[dict] = []
        self.sources: list[dict] = []
        self.refusal = ""
        self.title = ""
        self.limit = ""


def _source_entries(event: dict) -> list[dict]:
    source = event.get("source")
    entries: list[dict] = []
    if isinstance(source, dict):
        entries.append(source)
    raw = event.get("sources")
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                entries.append(item)
    return entries


def parse_event(payload: dict) -> DuckAIEvent:
    event = DuckAIEvent()
    role = payload.get("role")
    if role in (None, "assistant"):
        message = payload.get("message")
        if isinstance(message, str) and message:
            event.delta = message
        else:
            content = payload.get("content")
            if isinstance(content, str) and content:
                event.delta = content
    elif role == "reasoning":
        text = payload.get("text")
        if payload.get("state") == "text-delta" and isinstance(text, str):
            event.reasoning = text
    elif role == "tool-invocation":
        name = payload.get("toolName")
        if payload.get("state") == "call" and isinstance(name, str) and name:
            arguments = payload.get("toolArguments")
            if not isinstance(arguments, str):
                arguments = json.dumps(arguments if arguments is not None else {}, separators=(",", ":"))
            event.tool_calls.append(
                {
                    "index": 0,
                    "id": payload.get("toolCallId") or f"call_{name}_{uuid.uuid4().hex[:16]}",
                    "type": "function",
                    "function": {"name": name, "arguments": arguments},
                }
            )
    elif role == "refusal":
        detected = payload.get("detectedBy")
        event.refusal = detected if isinstance(detected, str) and detected else "content_filter"
    for source in _source_entries(payload):
        url = source.get("url") if isinstance(source.get("url"), str) else None
        if not url:
            continue
        event.sources.append({"url": url, "title": source.get("title") or "", "site": source.get("site") or ""})
    return event


def parse_control(line: str) -> DuckAIEvent | None:
    stripped = line.strip()
    if not stripped or stripped == STREAM_PING:
        return None
    if stripped.startswith(STREAM_CHAT_TITLE):
        event = DuckAIEvent()
        body = stripped[len(STREAM_CHAT_TITLE) :].strip()
        event.title = body[:-1].strip() if body.endswith("]") else body
        return event
    if stripped == STREAM_DONE or stripped.startswith(f"{STREAM_DONE}["):
        event = DuckAIEvent()
        event.finish = "stop"
        for marker, limit in LIMIT_MARKERS.items():
            if marker in stripped:
                event.limit = limit
        marker_at = stripped.find(STOP_REASON_PREFIX)
        if marker_at != -1:
            end = stripped.find("]", marker_at + len(STOP_REASON_PREFIX))
            reason = stripped[marker_at + len(STOP_REASON_PREFIX) : end if end != -1 else None]
            event.finish = normalize_finish_reason(reason)
        return event
    return None


def _error_for_payload(status: int, payload: Any) -> DuckAIError:
    message = ""
    error_type = ""
    if isinstance(payload, dict):
        raw_type = payload.get("type")
        if isinstance(raw_type, str) and raw_type:
            error_type = raw_type
        raw_message = payload.get("message")
        if isinstance(raw_message, str) and raw_message:
            message = raw_message
        challenge = payload.get("cd")
        if isinstance(challenge, dict):
            override = challenge.get("gk")
            if isinstance(override, str) and override:
                message = f"{message or 'bot check failed'} (challenge {override})".strip()
    return DuckAIError(status, message or f"upstream returned {status}", error_type)


def _oversized_bundle() -> DuckAIError:
    return DuckAIError("catalog", f"duck.ai entry bundle is above the {CATALOG_SCAN_LIMIT} byte scan limit")


def _reject_declared_length(headers: Any) -> None:
    declared = headers.get("content-length")
    if isinstance(declared, str) and declared.isdigit() and int(declared) > CATALOG_SCAN_LIMIT:
        raise _oversized_bundle()


def _reject_oversized_bundle(resp: Any) -> None:
    _reject_declared_length(resp.headers)
    if len(resp.content) > CATALOG_SCAN_LIMIT:
        raise _oversized_bundle()


class DuckAIClient:
    def __init__(
        self,
        timeout: float = 60.0,
        user_agent: str = USER_AGENT,
    ) -> None:
        self.user_agent = user_agent
        self.timeout = float(timeout)
        self._jsa: str = attest.INITIAL_JSA
        self._jsa_script = ""
        self._jsa_warm: asyncio.Task[None] | None = None
        self._fe_version: str = ""
        self._jsa_lock = asyncio.Lock()
        self._refresh_lock = asyncio.Lock()
        self._catalog: tuple[dict, ...] = MODEL_CATALOG
        self._catalog_lock = threading.Lock()
        self.http = httpx.AsyncClient(
            base_url=BASE_URL,
            headers={
                "User-Agent": user_agent,
                "Accept": "application/json",
                "Accept-Language": "en-US,en;q=0.9",
                "Origin": BASE_URL,
                "Referer": f"{BASE_URL}/",
            },
            timeout=httpx.Timeout(timeout, read=max(float(timeout) * 5, 300.0)),
            follow_redirects=True,
            limits=httpx.Limits(max_keepalive_connections=20, keepalive_expiry=30.0),
        )

    async def aclose(self) -> None:
        warm = self._jsa_warm
        self._cancel_attestation_warm()
        if warm is not None:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await warm
        await self.http.aclose()

    async def fe_version(self) -> str:
        if self._fe_version:
            return self._fe_version
        try:
            resp = await self.http.get("/", headers=self._headers({"cache-control": "no-cache"}))
        except (httpx.HTTPError, OSError, RuntimeError) as exc:
            log.debug("duckai could not read the build version: %s", exc)
            return FE_VERSION_FALLBACK
        if resp.status_code >= 400:
            log.debug("duckai build version page answered %d", resp.status_code)
            return FE_VERSION_FALLBACK
        tag = FE_VERSION_RE.search(resp.text)
        sha = FE_SHA_RE.search(resp.text)
        if tag and sha:
            self._fe_version = f"{tag.group(1)}-{sha.group(1)}"
        else:
            self._fe_version = FE_VERSION_FALLBACK
        return self._fe_version

    def invalidate_attestation(self) -> None:
        self._cancel_attestation_warm()
        self._jsa = attest.INITIAL_JSA
        self._jsa_script = ""

    def _cancel_attestation_warm(self) -> None:
        warm = self._jsa_warm
        if warm is None:
            return
        self._jsa_warm = None
        if not warm.done():
            warm.cancel()

    def _headers(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        headers = dict(BROWSER_HEADERS)
        headers.update(
            {
                "user-agent": self.user_agent,
                "accept": "application/json",
                "cache-control": "no-store",
                "x-vqd-accept": "1",
            }
        )
        if extra:
            headers.update(extra)
        return headers

    @staticmethod
    def _raise_for_payload(status: int, payload: Any) -> None:
        if status < 400:
            return
        raise _error_for_payload(status, payload)

    async def status(self) -> dict:
        resp = await self.http.get(STATUS_PATH, headers=self._headers())
        try:
            payload = resp.json()
        except ValueError as exc:
            raise DuckAIError(resp.status_code, "status endpoint returned non-JSON") from exc
        self._raise_for_payload(resp.status_code, payload)
        script = resp.headers.get(attest.JSA_HEADER)
        if script:
            try:
                await self._refresh_attestation(script)
            except attest.AttestationError as exc:
                self.invalidate_attestation()
                log.warning("duckai could not solve the attestation served this time, will retry: %s", exc)
        return payload if isinstance(payload, dict) else {}

    async def _refresh_attestation(self, script_b64: str) -> str:
        header = await attest.header_for(script_b64, self.user_agent, BASE_URL)
        async with self._jsa_lock:
            self._jsa = header
            self._jsa_script = script_b64
        return header

    def _start_attestation_warm(self, script_b64: str) -> None:
        if not script_b64 or script_b64 == self._jsa_script:
            return
        warm = self._jsa_warm
        if warm is not None and not warm.done():
            return
        self._jsa_warm = asyncio.create_task(self._warm_attestation(script_b64))

    async def _warm_attestation(self, script_b64: str) -> None:
        try:
            await self._refresh_attestation(script_b64)
        except attest.AttestationError as exc:
            log.debug("duckai attestation refresh failed: %s", exc)

    async def _join_attestation_warm(self) -> None:
        warm = self._jsa_warm
        if warm is not None and not warm.done():
            await warm
        if self._jsa_warm is warm:
            self._jsa_warm = None

    async def attestation(self, force: bool = False) -> str:
        if not force and self._jsa != attest.INITIAL_JSA:
            return self._jsa
        async with self._refresh_lock:
            if not force and self._jsa != attest.INITIAL_JSA:
                return self._jsa
            await self.status()
            return self._jsa

    async def check_auth(self) -> bool:
        try:
            payload = await self.status()
        except (DuckAIError, httpx.HTTPError, OSError, RuntimeError, attest.AttestationError):
            return False
        if not payload or "status" not in payload:
            return False
        return str(payload.get("status")) in STATUS_OK_CODES

    async def fetch_models(self) -> list[dict]:
        catalog = await self._fetch_catalog()
        return [dict(entry) for entry in catalog]

    def efforts_for(self, model: str) -> tuple[str, ...]:
        with self._catalog_lock:
            entries = self._catalog
        for entry in entries:
            if entry["id"] == model:
                return entry["efforts"]
        return model_efforts(model)

    def _known_catalog(self) -> tuple[dict, ...]:
        with self._catalog_lock:
            return self._catalog

    async def _read_bundle(self, path: str) -> str:
        parts: list[str] = []
        size = 0
        async with self.http.stream("GET", path, headers=self._headers({"accept": "*/*"})) as resp:
            _reject_declared_length(resp.headers)
            async for chunk in resp.aiter_text():
                size += len(chunk)
                if size > CATALOG_SCAN_LIMIT:
                    raise _oversized_bundle()
                parts.append(chunk)
        return "".join(parts)

    async def _fetch_catalog(self) -> tuple[dict, ...]:
        known = self._known_catalog()
        try:
            page = await self.http.get("/", headers=self._headers({"accept": "text/html"}))
            script = ENTRY_SCRIPT_RE.search(page.text)
            if not script:
                raise DuckAIError("catalog", "duck.ai entry bundle is not referenced by the page")
            entries = parse_catalog(await self._read_bundle(script.group("path")))
            if not entries:
                raise DuckAIError("catalog", "duck.ai bundle carries no free model entries")
        except (DuckAIError, httpx.HTTPError, OSError, RuntimeError) as exc:
            log.warning("duckai catalog fetch failed, keeping %d known models: %s", len(known), exc)
            return known
        with self._catalog_lock:
            self._catalog = entries
        log.info("duckai catalog refreshed: %d free models", len(entries))
        return entries

    def _request_body(self, messages: list[dict], model: str, effort: str | None, *, can_use_tools: bool, can_use_web_search: bool) -> dict:
        return {
            "model": model,
            "messages": messages,
            "canUseTools": can_use_tools,
            "reasoningEffort": normalize_effort_in(self.efforts_for(model), effort),
            "canUseApproxLocation": False,
            "canDelegateImageGeneration": False,
            "canUseWebSearch": can_use_web_search,
            "canUploadFiles": False,
            "canShowGreeting": False,
        }

    async def chat(
        self,
        messages: list[dict],
        model: str = DEFAULT_MODEL,
        effort: str | None = None,
        *,
        can_use_tools: bool = False,
        can_use_web_search: bool = False,
    ) -> AsyncIterator[DuckAIEvent]:
        body = self._request_body(
            messages,
            model,
            effort,
            can_use_tools=can_use_tools,
            can_use_web_search=can_use_web_search,
        )
        jsa = await self.attestation()
        headers = self._headers(
            {
                "content-type": "application/json",
                "accept": "text/event-stream",
                "x-fe-version": await self.fe_version(),
                "x-fe-signals": attest.fraud_signals(),
                attest.JSA_HEADER.lower(): jsa,
            }
        )
        request = self.http.build_request("POST", CHAT_PATH, json=body, headers=headers)
        try:
            resp = await self.http.send(request, stream=True)
        except httpx.HTTPError as exc:
            raise DuckAIError(502, f"transport error: {exc}") from exc
        if resp.status_code >= 400:
            raise await self._fail(resp)
        next_jsa = resp.headers.get(attest.JSA_HEADER)
        if next_jsa and next_jsa != jsa:
            self._start_attestation_warm(next_jsa)
        seen_sources: set[str] = set()
        try:
            async for line in _iter_lines(resp):
                control = parse_control(line)
                if control is not None:
                    yield control
                    if control.finish is not None:
                        break
                    continue
                payload = _parse_json(line)
                if payload is None:
                    continue
                action = payload.get("action")
                if action == "error":
                    raw_type = payload.get("type")
                    error_type = raw_type if isinstance(raw_type, str) else ""
                    raise DuckAIError(
                        STREAM_ERROR_STATUS.get(error_type, 502),
                        str(payload.get("message") or "upstream error"),
                        error_type,
                    )
                if action != "success":
                    continue
                event = parse_event(payload)
                if event.sources:
                    fresh: list[dict] = []
                    for source in event.sources:
                        if source["url"] in seen_sources:
                            continue
                        seen_sources.add(source["url"])
                        fresh.append(source)
                    event.sources = fresh
                yield event
            await self._join_attestation_warm()
        finally:
            await resp.aclose()

    async def _fail(self, resp: httpx.Response) -> DuckAIError:
        try:
            await resp.aread()
            payload = resp.json()
        except (ValueError, httpx.HTTPError):
            payload = None
        await resp.aclose()
        return _error_for_payload(resp.status_code, payload)


async def _iter_lines(resp: httpx.Response) -> AsyncIterator[str]:
    buffer = ""
    async for chunk in resp.aiter_text():
        if not chunk:
            continue
        buffer += chunk
        start = 0
        while True:
            end = buffer.find("\n", start)
            if end == -1:
                break
            text = _clean_line(buffer[start:end])
            start = end + 1
            if text:
                yield text
        if start:
            buffer = buffer[start:]
        if len(buffer) > MAX_SSE_LINE_CHARS:
            raise DuckAIError(502, f"duckai sent an SSE line above the {MAX_SSE_LINE_CHARS} character limit without a newline")
    text = _clean_line(buffer)
    if text:
        yield text


_SSE_METADATA_PREFIXES = ("event:", "id:", "retry:")


def _clean_line(line: str) -> str:
    text = line.strip()
    if not text or text.startswith(_SSE_METADATA_PREFIXES):
        return ""
    if text.startswith("data:"):
        text = text[len("data:") :].strip()
    return text


def _parse_json(line: str) -> dict | None:
    try:
        payload = json.loads(line)
    except ValueError:
        return None
    return payload if isinstance(payload, dict) else None
