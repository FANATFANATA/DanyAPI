from __future__ import annotations

import asyncio
import base64
import binascii
import logging
import re
import time
import uuid
from typing import Any

import httpx

from ..api.retry import MAX_RETRIES, RETRY_BACKOFF_MAX_SEC, RETRYABLE_HTTP_STATUSES, _retry_after_hint, _retry_delay
from .tls import resolve_ca

log = logging.getLogger("danyapi.gigachat")

BASE_URL = "https://api.giga.chat/v1"
AUTH_URL = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"

SCOPES = ("GIGACHAT_API_PERS", "GIGACHAT_API_B2B", "GIGACHAT_API_CORP")
DEFAULT_SCOPE = "GIGACHAT_API_PERS"

TOKEN_LIFETIME_SEC = 1800.0
TOKEN_EXPIRY_BUFFER_SEC = 60.0
TOKEN_RETRY_LIMIT_SEC = 30.0

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/151.0.0.0 Safari/537.36"

AUTH_ERROR_STATUSES = {401, 403}
TOKEN_ERROR_STATUSES = {400, 401, 403, 429}

IMAGE_MIME_TYPES = frozenset({"image/jpeg", "image/png", "image/tiff", "image/bmp"})

AUTHORIZATION_KEY_BYTES = 36
AUTHORIZATION_KEY_PATTERN = re.compile(r"[A-Za-z0-9+/]+={0,2}")

EMPTY_CREDENTIAL = ""


class GigaChatError(Exception):
    def __init__(self, code: int | str, message: str) -> None:
        super().__init__(f"GigaChat error {code}: {message}")
        self.code = code
        self.message = message

    @property
    def is_auth(self) -> bool:
        return self.code in AUTH_ERROR_STATUSES


def is_authorization_key(value: str) -> bool:
    text = (value or "").strip()
    if not text or not AUTHORIZATION_KEY_PATTERN.fullmatch(text):
        return False
    try:
        decoded = base64.b64decode(text, validate=True)
    except (ValueError, binascii.Error):
        return False
    if len(decoded) != AUTHORIZATION_KEY_BYTES:
        return False
    try:
        uuid.UUID(decoded.decode("ascii"))
    except (UnicodeDecodeError, ValueError):
        return False
    return True


def _expires_at_seconds(value: Any) -> float:
    if isinstance(value, str):
        try:
            value = float(value)
        except ValueError:
            return 0.0
    if not isinstance(value, (int, float)):
        return 0.0
    number = float(value)
    if number <= 0.0:
        return 0.0
    if number > 1e11:
        return number / 1000.0
    return number


def _retry_after_seconds(resp: httpx.Response) -> float:
    hinted = _retry_after_hint(resp.headers.get("Retry-After"))
    if hinted is None or hinted <= 0.0:
        return _retry_delay(1)
    return min(hinted, RETRY_BACKOFF_MAX_SEC)


class GigaChatClient:
    def __init__(
        self,
        key: str,
        scope: str = DEFAULT_SCOPE,
        timeout: float = 60.0,
    ) -> None:
        self.key = (key or EMPTY_CREDENTIAL).strip()
        self.scope = scope if scope in SCOPES else DEFAULT_SCOPE
        self._token = EMPTY_CREDENTIAL
        self._token_expires_at = 0.0
        self._token_deferred_until = 0.0
        self._token_error: tuple[int, str] = (503, "authorization endpoint unavailable")
        self._token_lock = asyncio.Lock()
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
                verify=resolve_ca(),
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

    def invalidate_token(self) -> None:
        self._token = EMPTY_CREDENTIAL
        self._token_expires_at = 0.0

    @staticmethod
    def _request_id() -> str:
        return str(uuid.uuid4())

    def _auth_headers(self) -> dict[str, str]:
        return {
            "RqUID": self._request_id(),
            "Authorization": f"Basic {self.key}",
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
            "User-Agent": USER_AGENT,
        }

    def _api_headers(self, token: str, extra: dict[str, str] | None = None, *, json_body: bool = True) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
            "X-Request-ID": self._request_id(),
        }
        if json_body:
            headers["Content-Type"] = "application/json"
        if extra:
            headers.update(extra)
        return headers

    @staticmethod
    def _raise_for_payload(status: int, payload: Any) -> None:
        if status < 400:
            return
        message = ""
        code: int | str = status
        if isinstance(payload, dict):
            raw_status = payload.get("status")
            if isinstance(raw_status, int):
                code = raw_status
            raw_message = payload.get("message")
            if isinstance(raw_message, str):
                message = raw_message
        raise GigaChatError(code, message or f"upstream returned {status}")

    def _defer_token(self, code: int, message: str) -> None:
        self._token = EMPTY_CREDENTIAL
        self._token_expires_at = 0.0
        self._token_deferred_until = time.time() + TOKEN_RETRY_LIMIT_SEC
        self._token_error = (code, message)

    def _cached_token(self, now: float, force: bool) -> str | None:
        if force:
            return None
        if self._token and now + TOKEN_EXPIRY_BUFFER_SEC < self._token_expires_at:
            return self._token
        if now < self._token_deferred_until:
            code, message = self._token_error
            raise GigaChatError(code, message)
        return None

    async def _obtain_token(self, force: bool = False) -> str:
        now = time.time()
        cached = self._cached_token(now, force)
        if cached is not None:
            return cached
        async with self._token_lock:
            now = time.time()
            cached = self._cached_token(now, force)
            if cached is not None:
                return cached
            try:
                resp = await self.http.post(
                    AUTH_URL,
                    content=f"scope={self.scope}",
                    headers=self._auth_headers(),
                )
            except httpx.HTTPError as exc:
                message = f"authorization endpoint transport error: {exc}"
                self._defer_token(503, message)
                raise GigaChatError(503, message) from exc
            try:
                payload = resp.json()
            except ValueError as exc:
                raise GigaChatError(resp.status_code, "authorization endpoint returned non-JSON") from exc
            try:
                self._raise_for_payload(resp.status_code, payload)
            except GigaChatError as exc:
                if resp.status_code in TOKEN_ERROR_STATUSES:
                    code = exc.code if isinstance(exc.code, int) else resp.status_code
                    self._defer_token(code, exc.message)
                raise
            if not isinstance(payload, dict):
                raise GigaChatError(resp.status_code, "unexpected authorization payload")
            token = payload.get("access_token")
            if not isinstance(token, str) or not token:
                raise GigaChatError(resp.status_code, "authorization response has no access_token")
            expires_at = _expires_at_seconds(payload.get("expires_at"))
            if expires_at <= 0.0:
                expires_at = time.time() + TOKEN_LIFETIME_SEC
            if expires_at <= time.time() + TOKEN_EXPIRY_BUFFER_SEC:
                self._defer_token(resp.status_code, "authorization response returned an already expired access_token")
                raise GigaChatError(resp.status_code, "authorization response returned an already expired access_token")
            self._token = token
            self._token_expires_at = expires_at
            self._token_deferred_until = 0.0
            self._token_error = (503, "authorization endpoint unavailable")
            log.info("gigachat access token obtained, valid for %.0fs", expires_at - time.time())
            return token

    async def access_token(self) -> str:
        return await self._obtain_token()

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: Any = None,
        content: Any = None,
        files: Any = None,
        data: Any = None,
        params: Any = None,
        headers: dict[str, str] | None = None,
        stream: bool = False,
    ) -> httpx.Response:
        attempt = 0
        retries = 0
        multipart = files is not None or data is not None
        while True:
            token = await self._obtain_token()
            request_headers = self._api_headers(token, headers, json_body=not multipart)
            req = self.http.build_request(
                method,
                path,
                json=json_body,
                content=content,
                files=files,
                data=data,
                params=params,
                headers=request_headers,
            )
            resp = await self.http.send(req, stream=stream)
            status = resp.status_code
            if status in AUTH_ERROR_STATUSES:
                if attempt < 1:
                    await resp.aclose()
                    attempt += 1
                    self.invalidate_token()
                    log.warning("gigachat auth rejected, refreshing access token and retrying")
                    continue
                return resp
            if status in RETRYABLE_HTTP_STATUSES and retries < MAX_RETRIES:
                retry_after = _retry_after_seconds(resp)
                await resp.aclose()
                retries += 1
                log.warning("gigachat returned %s, retrying in %.1fs (attempt %d)", status, retry_after, retries)
                await asyncio.sleep(retry_after)
                continue
            return resp

    async def _json_request(
        self,
        method: str,
        path: str,
        *,
        json_body: Any = None,
        content: Any = None,
        files: Any = None,
        data: Any = None,
        params: Any = None,
        headers: dict[str, str] | None = None,
    ) -> Any:
        resp = await self._request(
            method,
            path,
            json_body=json_body,
            content=content,
            files=files,
            data=data,
            params=params,
            headers=headers,
        )
        try:
            payload = resp.json()
        except ValueError as exc:
            raise GigaChatError(resp.status_code, f"unexpected non-JSON response from {path}") from exc
        self._raise_for_payload(resp.status_code, payload)
        return payload

    async def check_auth(self) -> bool:
        try:
            payload = await self._json_request("GET", "/models")
        except (GigaChatError, httpx.HTTPError, OSError, RuntimeError):
            return False
        return isinstance(payload, dict) and bool(payload.get("data"))

    async def fetch_models(self) -> list[dict]:
        payload = await self._json_request("GET", "/models")
        if not isinstance(payload, dict):
            return []
        data = payload.get("data")
        return [item for item in data if isinstance(item, dict)] if isinstance(data, list) else []

    async def chat(self, body: dict, model: str) -> httpx.Response:
        payload = dict(body)
        payload["model"] = model
        payload.setdefault("stream", False)
        headers: dict[str, str] = {}
        if payload.get("stream"):
            headers = {"Accept": "text/event-stream", "X-Accel-Buffering": "no"}
        return await self._request("POST", "/chat/completions", json_body=payload, headers=headers, stream=bool(payload.get("stream")))

    async def upload_file(self, filename: str, data: bytes, content_type: str, purpose: str = "general") -> str:
        payload = await self._json_request(
            "POST",
            "/files",
            files={"file": (filename, data, content_type)},
            data={"purpose": purpose},
        )
        if not isinstance(payload, dict):
            raise GigaChatError(200, "unexpected file upload payload")
        file_id = payload.get("id")
        if not isinstance(file_id, str) or not file_id:
            raise GigaChatError(200, "file upload returned no id")
        return file_id
