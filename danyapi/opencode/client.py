from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any

import httpx

from ..api.retry import MAX_RETRIES, RETRY_BACKOFF_MAX_SEC, RETRYABLE_HTTP_STATUSES, _retry_delay

log = logging.getLogger("danyapi.opencode")

BASE_URL = "https://opencode.ai/zen/v1"

PROVIDER_ID = "opencode"

MODEL_PREFIX = f"{PROVIDER_ID}/"

USER_AGENT = "opencode/1.18.33"

CLIENT_NAME = "danyapi"

ERROR_STATUS_BY_TYPE = {
    "AuthError": 401,
    "BillingError": 402,
    "FreeTierError": 403,
    "PermissionDeniedError": 403,
    "RegionError": 403,
    "ModelError": 404,
    "NotFoundError": 404,
    "RateLimitError": 429,
}

ERROR_TYPE_HINTS = {
    "AuthError": "the OpenCode Zen API key is missing or was rejected",
    "BillingError": "the OpenCode Zen account has no usable balance, top it up at https://opencode.ai/zen",
    "FreeTierError": "the OpenCode Zen free tier for this model is exhausted, use a different model or add credit",
    "RegionError": "OpenCode Zen does not serve this model in this region",
    "ModelError": "OpenCode Zen cannot serve this model through the chat completions format",
    "RateLimitError": "OpenCode Zen rate limit reached",
}

EMPTY_CREDENTIAL = ""

AUTH_ERROR_TYPES = frozenset({"AuthError"})

MAX_ERROR_CHARS = 300


def upstream_model(model: str) -> str:
    """Strip the ``opencode/`` routing prefix, Zen expects the bare model id."""
    text = (model or "").strip()
    if text.lower().startswith(MODEL_PREFIX):
        return text[len(MODEL_PREFIX) :]
    return text


class OpenCodeError(Exception):
    def __init__(self, code: int | str, message: str, error_type: str = "") -> None:
        super().__init__(f"OpenCode Zen error {code}: {message}")
        self.code = code
        self.message = message
        self.error_type = error_type

    @property
    def is_auth(self) -> bool:
        if self.error_type:
            return self.error_type in AUTH_ERROR_TYPES
        return self.code in (401, 403)

    @property
    def detail(self) -> str:
        text = (self.message or "").strip() or str(self.code)
        if len(text) > MAX_ERROR_CHARS:
            text = text[:MAX_ERROR_CHARS]
        hint = ERROR_TYPE_HINTS.get(self.error_type)
        return f"OpenCode Zen error: {text}" + (f" ({hint})" if hint else "")


def error_message(payload: Any, status: int) -> tuple[str, str]:
    error_type = ""
    message = ""
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            raw_type = error.get("type")
            if isinstance(raw_type, str):
                error_type = raw_type
            raw_message = error.get("message")
            if isinstance(raw_message, str):
                message = raw_message
        raw_message = payload.get("message")
        if not message and isinstance(raw_message, str):
            message = raw_message
    return error_type, message or f"upstream returned {status}"


def _retry_after_seconds(resp: httpx.Response) -> float:
    header = resp.headers.get("Retry-After", "").strip()
    if not header:
        return _retry_delay(1)
    try:
        advertised = float(header)
    except ValueError:
        return _retry_delay(1)
    if advertised <= 0.0:
        return _retry_delay(1)
    return min(advertised, RETRY_BACKOFF_MAX_SEC)


class OpenCodeClient:
    def __init__(
        self,
        key: str = EMPTY_CREDENTIAL,
        timeout: float = 60.0,
        client_id: str = CLIENT_NAME,
    ) -> None:
        self.key = (key or EMPTY_CREDENTIAL).strip()
        self.client_id = client_id or CLIENT_NAME
        self._timeout = timeout
        self._http: httpx.AsyncClient | None = None

    @property
    def http(self) -> httpx.AsyncClient:
        client = self._http
        if client is None:
            client = httpx.AsyncClient(
                base_url=BASE_URL,
                headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
                timeout=httpx.Timeout(self._timeout, read=max(float(self._timeout) * 5, 300.0)),
                follow_redirects=True,
                limits=httpx.Limits(max_keepalive_connections=20, keepalive_expiry=30.0),
            )
            self._http = client
        return client

    @http.setter
    def http(self, client: httpx.AsyncClient) -> None:
        self._http = client

    async def aclose(self) -> None:
        client = self._http
        if client is None:
            return
        self._http = None
        await client.aclose()

    def _headers(
        self,
        extra: dict[str, str] | None = None,
        *,
        json_body: bool = True,
        session_id: str | None = None,
        request_id: str | None = None,
    ) -> dict[str, str]:
        headers: dict[str, str] = {
            "Accept": "application/json",
            "User-Agent": USER_AGENT,
            "x-opencode-client": self.client_id,
        }
        if self.key:
            headers["Authorization"] = f"Bearer {self.key}"
        if session_id:
            headers["x-opencode-session"] = session_id
        if request_id:
            headers["x-opencode-request"] = request_id
        if json_body:
            headers["Content-Type"] = "application/json"
        if extra:
            headers.update(extra)
        return headers

    @staticmethod
    def _raise_for_payload(status: int, payload: Any) -> None:
        if status < 400:
            return
        error_type, message = error_message(payload, status)
        raise OpenCodeError(status, message, error_type)

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: Any = None,
        headers: dict[str, str] | None = None,
        stream: bool = False,
        session_id: str | None = None,
        request_id: str | None = None,
    ) -> httpx.Response:
        retries = 0
        while True:
            req = self.http.build_request(
                method,
                path,
                json=json_body,
                headers=self._headers(headers, session_id=session_id, request_id=request_id),
            )
            resp = await self.http.send(req, stream=stream)
            status = resp.status_code
            if status in RETRYABLE_HTTP_STATUSES and retries < MAX_RETRIES:
                retry_after = _retry_after_seconds(resp)
                await resp.aclose()
                retries += 1
                log.warning("opencode returned %s, retrying in %.1fs (attempt %d)", status, retry_after, retries)
                await asyncio.sleep(retry_after)
                continue
            return resp

    async def _json_request(self, method: str, path: str, **kwargs: Any) -> Any:
        resp = await self._request(method, path, **kwargs)
        try:
            payload = resp.json()
        except ValueError as exc:
            raise OpenCodeError(resp.status_code, f"unexpected non-JSON response from {path}") from exc
        self._raise_for_payload(resp.status_code, payload)
        return payload

    async def check_auth(self) -> bool:
        try:
            payload = await self._json_request("GET", "/models")
        except (OpenCodeError, httpx.HTTPError, OSError, RuntimeError):
            return False
        return isinstance(payload, dict) and bool(payload.get("data"))

    async def fetch_models(self) -> list[dict]:
        payload = await self._json_request("GET", "/models")
        if not isinstance(payload, dict):
            return []
        data = payload.get("data")
        return [item for item in data if isinstance(item, dict)] if isinstance(data, list) else []

    async def chat(
        self,
        body: dict,
        model: str,
        session_id: str | None = None,
        request_id: str | None = None,
    ) -> httpx.Response:
        payload = dict(body)
        payload["model"] = upstream_model(model)
        payload.setdefault("stream", False)
        extra: dict[str, str] = {}
        if payload.get("stream"):
            extra = {"Accept": "text/event-stream", "X-Accel-Buffering": "no"}
        return await self._request(
            "POST",
            "/chat/completions",
            json_body=payload,
            headers=extra,
            stream=bool(payload.get("stream")),
            session_id=session_id,
            request_id=request_id,
        )


def new_request_id() -> str:
    return f"msg_{uuid.uuid4().hex}"
