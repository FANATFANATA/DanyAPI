from __future__ import annotations

import datetime
import hashlib
import logging
import time
import uuid
from dataclasses import dataclass, field

import httpx

log = logging.getLogger("danyapi.qwen")

BASE_URL = "https://chat.qwen.ai"

WEB_VERSION = "0.2.83"

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/151.0.0.0 Safari/537.36"

COMMON_HEADERS = {
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Origin": BASE_URL,
    "Referer": BASE_URL + "/",
    "source": "web",
    "version": WEB_VERSION,
    "sec-ch-ua": '"Not=A?Brand";v="99", "Google Chrome";v="151", "Chromium";v="151"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"Windows"',
}


def new_uuid() -> str:
    return str(uuid.uuid4())


def sha256_hex(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def timezone_header() -> str:
    now = datetime.datetime.now().astimezone()
    offset = now.strftime("%z")
    return f"{now.strftime('%a %b %d %Y %H:%M:%S')} GMT{offset}"


@dataclass
class QwenSession:
    id: str
    title: str = ""
    last_response_id: str | None = None
    model: str | None = None
    accumulated_input_tokens: int = 0
    accumulated_output_tokens: int = 0
    extra: dict = field(default_factory=dict)


class QwenError(Exception):
    def __init__(self, code: int | str, message: str) -> None:
        super().__init__(f"Qwen error {code}: {message}")
        self.code = code
        self.message = message


class QwenClient:
    def __init__(
        self,
        token: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        self.token = token
        headers = {
            "User-Agent": USER_AGENT,
            **COMMON_HEADERS,
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"
        self.http = httpx.AsyncClient(
            base_url=BASE_URL,
            headers=headers,
            timeout=httpx.Timeout(timeout),
            follow_redirects=True,
        )
        if token:
            self.http.cookies.set("token", token, domain="chat.qwen.ai", path="/")

    async def aclose(self) -> None:
        await self.http.aclose()

    @staticmethod
    def _request_headers(headers: dict | None = None) -> dict:
        base = {
            "X-Request-Id": new_uuid(),
            "Timezone": timezone_header(),
        }
        if headers:
            base.update(headers)
        return base

    @staticmethod
    def _biz(payload: dict) -> dict:
        if not payload.get("success", False):
            data = payload.get("data")
            if isinstance(data, dict) and data.get("code"):
                raise QwenError(data["code"], data.get("details") or data.get("message") or "")
            raise QwenError(payload.get("code") or -1, payload.get("details") or payload.get("message") or "request failed")
        data = payload.get("data")
        return data if isinstance(data, dict) else {}

    def _parse_json(self, resp: httpx.Response, path: str) -> dict:
        content_type = resp.headers.get("content-type", "")
        if "json" not in content_type:
            raise QwenError(-1, f"unexpected non-JSON response from {path}: {resp.text[:200]}")
        try:
            payload = resp.json()
        except ValueError as exc:
            raise QwenError(-1, f"invalid JSON from {path}") from exc
        return self._biz(payload)

    async def _post(self, path: str, json_body: dict | None = None, params: dict | None = None, headers: dict | None = None) -> dict:
        resp = await self.http.post(path, json=json_body, params=params, headers=self._request_headers(headers))
        return self._parse_json(resp, path)

    async def _get(self, path: str, params: dict | None = None, headers: dict | None = None) -> dict:
        resp = await self.http.get(path, params=params, headers=self._request_headers(headers))
        return self._parse_json(resp, path)

    async def check_auth(self) -> bool:
        try:
            resp = await self.http.get("/api/v1/auths/", headers=self._request_headers())
            if resp.status_code != 200:
                return False
            return bool(resp.json().get("success", True))
        except (httpx.HTTPError, ValueError):
            return False

    async def login(self, email: str, password: str) -> str:
        resp = await self.http.post(
            "/api/v2/auths/signin",
            json={"email": email, "password": sha256_hex(password)},
            headers=self._request_headers(),
        )
        biz = self._parse_json(resp, "/api/v2/auths/signin")
        token = biz.get("token") or (biz.get("user") or {}).get("token")
        if not token:
            raise QwenError(-1, "login failed: no token in response")
        self.token = token
        self.http.headers["Authorization"] = f"Bearer {token}"
        self.http.cookies.set("token", token, domain="chat.qwen.ai", path="/")
        return token

    async def fetch_models(self) -> list[dict]:
        biz = await self._get("/api/v2/models/")
        models = biz.get("data")
        return models if isinstance(models, list) else []

    async def create_chat(self, model: str, chat_mode: str = "normal", chat_type: str = "t2t") -> str:
        started = time.monotonic()
        body = {
            "chatId": "",
            "models": [model],
            "project_id": "",
            "timestamp": int(datetime.datetime.now().timestamp() * 1000),
            "chat_type": chat_type,
            "chat_mode": chat_mode,
        }
        biz = await self._post("/api/v2/chats/new", body)
        chat_id = biz.get("id")
        if not chat_id:
            raise QwenError(-1, "create chat failed: no chat id in response")
        log.info("qwen create chat OK (%.0fms)", (time.monotonic() - started) * 1000)
        return chat_id

    async def completion(
        self,
        chat_session_id: str,
        prompt: str,
        parent_message_id: str | None,
        model: str,
        thinking: bool = False,
        search: bool = False,
    ) -> httpx.Response:
        log.debug("qwen completion start session=%s model=%s", chat_session_id, model)
        ts = int(datetime.datetime.now().timestamp())
        user_fid = new_uuid()
        response_fid = new_uuid()
        feature_config = {
            "thinking_enabled": thinking,
            "output_schema": "phase",
            "research_mode": "normal",
            "auto_thinking": thinking,
            "thinking_mode": "Auto" if thinking else "Manual",
            "thinking_format": "summary",
            "auto_search": search,
        }
        message = {
            "id": None,
            "fid": user_fid,
            "parentId": parent_message_id,
            "childrenIds": [response_fid],
            "role": "user",
            "content": prompt,
            "user_action": "chat",
            "files": [],
            "timestamp": ts,
            "models": [model],
            "model": "",
            "chat_type": "t2t",
            "feature_config": feature_config,
            "extra": {"meta": {"subChatType": "t2t"}},
            "sub_chat_type": "t2t",
            "parent_id": parent_message_id,
        }
        body = {
            "stream": True,
            "version": "2.1",
            "incremental_output": True,
            "chatId": chat_session_id,
            "parentId": parent_message_id or "",
            "chat_id": chat_session_id,
            "chat_mode": "normal",
            "model": model,
            "parent_id": parent_message_id,
            "messages": [message],
            "timestamp": ts,
        }
        headers = {
            "Accept": "text/event-stream",
            "X-Accel-Buffering": "no",
            "Referer": f"{BASE_URL}/c/{chat_session_id}",
        }
        req = self.http.build_request(
            "POST",
            "/api/v2/chat/completions",
            params={"chat_id": chat_session_id},
            json=body,
            headers=self._request_headers(headers),
        )
        return await self.http.send(req, stream=True)

    async def stop_stream(self, chat_session_id: str, response_id: str | None) -> None:
        await self._post(
            "/api/v2/chat/completions/stop",
            {"chat_id": chat_session_id, "response_id": response_id},
        )
