from __future__ import annotations

import contextlib
import logging
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass

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

SETTINGS_PATH = "/api/v0/client/settings"
SETTINGS_SCOPE_MAIN = "main"
SETTINGS_SCOPE_MODEL = "model"
MODEL_CONFIGS_KEY = "model_configs"
DEFAULT_MODEL_TYPE = "default"


def new_device_id() -> str:
    return str(uuid.uuid4())


@dataclass
class DeepSeekSession:
    id: str
    title: str = ""
    last_message_id: str | None = None
    accumulated_tokens: int = 0


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
            timeout=httpx.Timeout(timeout, read=max(float(timeout) * 5, 300.0)),
            follow_redirects=True,
            limits=httpx.Limits(max_keepalive_connections=20, keepalive_expiry=30.0),
        )

    async def aclose(self) -> None:
        await self.http.aclose()

    async def _request_json(self, method: str, path: str, json_error: str | None = None, **kwargs) -> dict:
        send = self.http.post if method == "POST" else self.http.get
        try:
            resp = await send(path, **kwargs)
        except httpx.HTTPError as exc:
            raise DeepSeekError(-1, f"http request failed: {exc}") from exc
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise DeepSeekError(exc.response.status_code, exc.response.text[:300]) from exc
        try:
            return resp.json()
        except ValueError as exc:
            raise DeepSeekError(-1, json_error or f"invalid JSON response from {path}: {resp.text[:300]}") from exc

    async def _post(self, path: str, json_body: dict | None = None) -> dict:
        return await self._request_json("POST", path, json=json_body)

    @staticmethod
    def _biz(resp: dict) -> dict:
        if not isinstance(resp, dict):
            raise DeepSeekError(-1, "unexpected response shape")
        code = resp.get("code")
        if code:
            raise DeepSeekError(code, resp.get("msg") or resp.get("message") or "")
        data = resp.get("data")
        if not isinstance(data, dict):
            data = {}
        if data.get("biz_code"):
            raise DeepSeekError(data["biz_code"], data.get("biz_msg", ""))
        biz_data = data.get("biz_data")
        return biz_data if isinstance(biz_data, dict) else {}

    async def check_auth(self) -> bool:
        try:
            resp = await self.http.get(
                SETTINGS_PATH,
                params={"did": self.device_id, "scope": SETTINGS_SCOPE_MAIN},
            )
        except httpx.HTTPError:
            return False
        if resp.status_code != 200:
            return False
        try:
            payload = resp.json()
        except ValueError:
            return False
        return isinstance(payload, dict) and payload.get("code") == 0

    async def fetch_models(self) -> list[dict]:
        try:
            resp = await self.http.get(
                SETTINGS_PATH,
                params={"did": self.device_id, "scope": SETTINGS_SCOPE_MODEL},
            )
        except httpx.HTTPError as exc:
            raise DeepSeekError(-1, f"http request failed: {exc}") from exc
        if resp.status_code != 200:
            raise DeepSeekError(resp.status_code, f"model settings returned {resp.status_code}")
        try:
            payload = resp.json()
        except ValueError as exc:
            raise DeepSeekError(-1, f"invalid JSON from {SETTINGS_PATH}: {resp.text[:200]}") from exc
        settings = self._biz(payload).get("settings")
        entry = settings.get(MODEL_CONFIGS_KEY) if isinstance(settings, dict) else None
        configs = entry.get("value") if isinstance(entry, dict) else None
        models: list[dict] = []
        if not isinstance(configs, list):
            return models
        disabled: list[str] = []
        for config in configs:
            if not isinstance(config, dict):
                continue
            model_type = config.get("model_type")
            if not isinstance(model_type, str) or not model_type:
                continue
            if not config.get("enabled", True):
                disabled.append(model_type)
                continue
            models.append(
                {
                    "id": model_type,
                    "name": config.get("name") or model_type,
                    "owned_by": "deepseek",
                    "model_type": model_type,
                    "is_default": bool(config.get("is_default")),
                    "switchable": bool(config.get("switchable")),
                    "supports_thinking": bool(config.get("think_feature")),
                    "supports_search": bool(config.get("search_feature")),
                }
            )
        if disabled:
            log.debug("deepseek model types disabled upstream: %s", ", ".join(disabled))
        return models

    async def get_user(self) -> dict:
        resp = await self._post("/api/v0/users", None)
        return self._biz(resp)

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
        raw = biz.get("chat_session")
        if not isinstance(raw, dict) or not raw.get("id"):
            raise DeepSeekError(-1, "no chat_session in response")
        session = DeepSeekSession(id=raw["id"], title=raw.get("title") or "")
        log.info("deepseek create session success (%.0fms)", (time.monotonic() - started) * 1000)
        return session

    async def fetch_page(self, pinned: bool = False, count: int = 20) -> list[dict]:
        body = {"pinned": pinned, "count": count, "mode": "lte"}
        resp = await self._post("/api/v0/chat_session/fetch_page", body)
        sessions = self._biz(resp).get("chat_sessions")
        return sessions if isinstance(sessions, list) else []

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
        payload = await self._request_json(
            "POST",
            "/api/v0/file/upload_file",
            "invalid JSON from file upload",
            files={"file": (filename, data, content_type)},
            headers=headers,
        )
        biz = self._biz(payload)
        if not biz.get("id"):
            raise DeepSeekError(-1, "file upload failed: no file id in response")
        log.info("deepseek upload file success: %s (%.0fms)", filename, (time.monotonic() - started) * 1000)
        return biz

    async def fetch_files(self, file_ids: list[str]) -> list[dict]:
        if not file_ids:
            return []
        payload = await self._request_json(
            "GET",
            "/api/v0/file/fetch_files",
            "invalid JSON from fetch_files",
            params={"file_ids": ",".join(file_ids)},
        )
        files = self._biz(payload).get("files")
        return files if isinstance(files, list) else []

    async def history_messages(self, chat_session_id: str) -> list[dict]:
        payload = await self._request_json(
            "GET",
            "/api/v0/chat/history_messages",
            "invalid JSON from history_messages",
            params={"chat_session_id": chat_session_id},
        )
        messages = self._biz(payload).get("chat_messages")
        return messages if isinstance(messages, list) else []

    async def rename_session(self, chat_session_id: str, title: str) -> None:
        self._biz(
            await self._post(
                "/api/v0/chat_session/update_title",
                {
                    "chat_session_id": chat_session_id,
                    "title": title,
                },
            )
        )

    async def delete_session(self, chat_session_id: str) -> None:
        self._biz(await self._post("/api/v0/chat_session/delete", {"chat_session_id": chat_session_id}))

    @contextlib.asynccontextmanager
    async def stream_completion(
        self,
        chat_session_id: str,
        prompt: str,
        parent_message_id: str | None,
        model_type: str = "default",
        thinking_enabled: bool = False,
        search_enabled: bool = False,
        ref_file_ids: list[str] | None = None,
        pow_headers: dict | None = None,
    ) -> AsyncIterator[httpx.Response]:
        resp = await self.completion(
            chat_session_id,
            prompt,
            parent_message_id,
            model_type,
            thinking_enabled,
            search_enabled,
            ref_file_ids,
            pow_headers,
        )
        try:
            yield resp
        finally:
            with contextlib.suppress(Exception):
                await resp.aclose()

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
        self._biz(
            await self._post(
                "/api/v0/chat/stop_stream",
                {
                    "chat_session_id": chat_session_id,
                    "message_id": message_id,
                },
            )
        )
