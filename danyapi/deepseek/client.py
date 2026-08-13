from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field

import httpx

log = logging.getLogger("danyapi.deepseek")

BASE_URL = "https://chat.deepseek.com"

CLIENT_HEADERS = {
    "x-client-bundle-id": "com.deepseek.chat",
    "x-client-platform": "web",
    "x-client-version": "2.3.0",
    "x-client-locale": "en-US",
    "x-client-timezone-offset": "0",
}

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"


def new_device_id() -> str:
    return str(uuid.uuid4())


@dataclass
class DeepSeekSession:
    id: str
    title: str = ""
    last_message_id: str | None = None
    accumulated_tokens: int = 0
    extra: dict = field(default_factory=dict)


class DeepSeekError(Exception):
    def __init__(self, biz_code: int, biz_msg: str):
        super().__init__(f"DeepSeek biz error {biz_code}: {biz_msg}")
        self.biz_code = biz_code
        self.biz_msg = biz_msg


class DeepSeekClient:
    def __init__(
        self,
        token: str | None = None,
        device_id: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        self.token = token
        self.device_id = device_id or new_device_id()
        headers = {
            "User-Agent": USER_AGENT,
            "Referer": "https://chat.deepseek.com/",
            "Origin": "https://chat.deepseek.com",
            "Accept": "*/*",
            **CLIENT_HEADERS,
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"
        self.http = httpx.AsyncClient(
            base_url=BASE_URL,
            headers=headers,
            timeout=httpx.Timeout(timeout),
            follow_redirects=True,
        )

    async def aclose(self) -> None:
        await self.http.aclose()

    async def _post(self, path: str, json_body: dict | None = None) -> dict:
        try:
            resp = await self.http.post(path, json=json_body)
        except httpx.HTTPError as exc:
            raise DeepSeekError(-1, f"http request failed: {exc}") from exc
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise DeepSeekError(exc.response.status_code, exc.response.text[:300]) from exc
        return resp.json()

    @staticmethod
    def _biz(resp: dict) -> dict:
        code = resp.get("code")
        if code:
            raise DeepSeekError(code, resp.get("msg") or resp.get("message") or "")
        data = resp.get("data") or {}
        if data.get("biz_code"):
            raise DeepSeekError(data["biz_code"], data.get("biz_msg", ""))
        return data.get("biz_data") or {}

    async def check_auth(self) -> bool:
        try:
            resp = await self.http.get(
                "/api/v0/client/settings",
                params={"did": self.device_id, "scope": "main"},
            )
            return resp.json().get("code") == 0
        except (httpx.HTTPError, ValueError):
            return False

    async def login(
        self,
        email: str | None = None,
        mobile: str | None = None,
        password: str | None = None,
        area_code: str = "",
    ) -> str:
        body = {
            "email": email,
            "mobile": mobile,
            "password": password,
            "area_code": area_code,
            "device_id": self.device_id,
            "os": "web",
        }
        resp = await self._post("/api/v0/users/login", body)
        biz = self._biz(resp)
        user = (biz or {}).get("user")
        if not user or not user.get("token"):
            raise DeepSeekError((resp.get("data") or {}).get("biz_code", -1), "login failed: no token")
        self.token = user["token"]
        self.http.headers["Authorization"] = f"Bearer {self.token}"
        return self.token

    async def get_user(self) -> dict:
        resp = await self._post("/api/v0/users", None)
        biz = self._biz(resp)
        return biz or {}

    async def create_pow_challenge(self, target_path: str = "/api/v0/chat/completion") -> dict:
        resp = await self._post("/api/v0/chat/create_pow_challenge", {"target_path": target_path})
        biz = self._biz(resp)
        challenge = biz.get("challenge")
        if not challenge:
            raise DeepSeekError(-1, "no pow challenge in response")
        log.debug("deepseek pow challenge OK (%s)", target_path)
        return challenge

    async def create_session(self) -> DeepSeekSession:
        started = time.monotonic()
        resp = await self._post("/api/v0/chat_session/create", {})
        biz = self._biz(resp)
        raw = biz["chat_session"]
        session = DeepSeekSession(id=raw["id"], title=raw.get("title") or "")
        log.info("deepseek create session OK (%.0fms)", (time.monotonic() - started) * 1000)
        return session

    async def fetch_page(self, pinned: bool = False, count: int = 20) -> list[dict]:
        body = {"pinned": pinned, "count": count, "mode": "lte"}
        resp = await self._post("/api/v0/chat_session/fetch_page", body)
        biz = self._biz(resp)
        return (biz or {}).get("chat_sessions", []) or []

    async def upload_file(
        self,
        data: bytes,
        filename: str,
        content_type: str,
        model_type: str,
        thinking_enabled: bool = False,
        pow_headers: dict | None = None,
    ) -> dict:
        started = time.monotonic()
        headers = {
            "X-File-Size": str(len(data)),
            "X-Model-Type": model_type,
            "X-Thinking-Enabled": "1" if thinking_enabled else "0",
        }
        if pow_headers:
            headers.update(pow_headers)
        try:
            resp = await self.http.post(
                "/api/v0/file/upload_file",
                files={"file": (filename, data, content_type)},
                headers=headers,
            )
        except httpx.HTTPError as exc:
            raise DeepSeekError(-1, f"http request failed: {exc}") from exc
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise DeepSeekError(exc.response.status_code, exc.response.text[:300]) from exc
        try:
            payload = resp.json()
        except ValueError as exc:
            raise DeepSeekError(-1, "invalid JSON from file upload") from exc
        biz = self._biz(payload)
        file_info = biz.get("id") if isinstance(biz, dict) else None
        if not isinstance(biz, dict) or not file_info:
            raise DeepSeekError(-1, "file upload failed: no file id in response")
        log.info("deepseek upload file OK: %s (%.0fms)", filename, (time.monotonic() - started) * 1000)
        return biz

    async def fetch_files(self, file_ids: list[str]) -> list[dict]:
        if not file_ids:
            return []
        try:
            resp = await self.http.get(
                "/api/v0/file/fetch_files",
                params={"file_ids": ",".join(file_ids)},
            )
        except httpx.HTTPError as exc:
            raise DeepSeekError(-1, f"http request failed: {exc}") from exc
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise DeepSeekError(exc.response.status_code, exc.response.text[:300]) from exc
        try:
            payload = resp.json()
        except ValueError as exc:
            raise DeepSeekError(-1, "invalid JSON from fetch_files") from exc
        biz = self._biz(payload)
        return (biz or {}).get("files", []) if isinstance(biz, dict) else []

    async def history_messages(self, chat_session_id: str) -> list[dict]:
        try:
            resp = await self.http.get(
                "/api/v0/chat/history_messages",
                params={"chat_session_id": chat_session_id},
            )
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise DeepSeekError(-1, f"http request failed: {exc}") from exc
        biz = self._biz(resp.json())
        return (biz or {}).get("chat_messages", [])

    async def rename_session(self, chat_session_id: str, title: str) -> None:
        await self._post(
            "/api/v0/chat_session/update_title",
            {
                "chat_session_id": chat_session_id,
                "title": title,
            },
        )

    async def delete_session(self, chat_session_id: str) -> None:
        await self._post("/api/v0/chat_session/delete", {"chat_session_id": chat_session_id})

    async def completion(
        self,
        chat_session_id: str,
        prompt: str,
        parent_message_id: str | None,
        model_type: str = "default",
        thinking_enabled: bool = False,
        search_enabled: bool = False,
        ref_file_ids: list[str] | None = None,
        pow_headers: dict | None = None,
    ) -> httpx.Response:
        body = {
            "chat_session_id": chat_session_id,
            "parent_message_id": parent_message_id,
            "model_type": model_type,
            "prompt": prompt,
            "ref_file_ids": ref_file_ids or [],
            "thinking_enabled": thinking_enabled,
            "search_enabled": search_enabled,
            "action": None,
            "preempt": False,
        }
        headers = {"Accept": "text/event-stream"}
        if pow_headers:
            headers.update(pow_headers)
        req = self.http.build_request("POST", "/api/v0/chat/completion", json=body, headers=headers)
        return await self.http.send(req, stream=True)

    async def stop_stream(self, chat_session_id: str, message_id: str | None) -> None:
        await self._post(
            "/api/v0/chat/stop_stream",
            {
                "chat_session_id": chat_session_id,
                "message_id": message_id,
            },
        )
