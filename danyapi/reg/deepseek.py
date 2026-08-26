from __future__ import annotations

import base64
import json
import logging
import uuid

import httpx

from ..pow import solve_challenge

log = logging.getLogger("danyapi.reg.deepseek")

BASE_URL = "https://chat.deepseek.com"

CLIENT_HEADERS = {
    "x-client-bundle-id": "com.deepseek.chat",
    "x-client-platform": "web",
    "x-client-version": "2.4.0",
    "x-client-locale": "en_US",
    "x-client-timezone-offset": "0",
}

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/151.0.0.0 Safari/537.36"

LOCALE = "en_US"
REGION = "[unset]"

GUEST_CHALLENGE_PATH = "/api/v0/users/create_guest_challenge"
CREATE_EMAIL_CODE_PATH = "/api/v0/users/create_email_verification_code"
REGISTER_PATH = "/api/v0/users/register"
LOGIN_PATH = "/api/v0/users/login"
CURRENT_USER_PATH = "/api/v0/users/current"

EMAIL_CODE_ERRORS = {
    1: "EMAIL_REQUEST_TOO_FREQUENT",
    2: "RECAPTCHA_VERIFY_FAILED",
    3: "INVALID_EMAIL_FORMAT",
    4: "EMAIL_DOMAIN_NOT_IN_WHITELIST",
    98: "TEMP_DISABLED_IN_THIS_CHANNEL",
    99: "CLOUD_ERROR",
}

REGISTER_ERRORS = {
    1: "EMAIL_EXISTS",
    4: "INVALID_PASSWORD",
    5: "EMAIL_VERIFY_TOO_MANY_ATTEMPTS",
    6: "REGISTER_FROM_MAINLAND",
    7: "EMAIL_EXPIRED",
    8: "EMAIL_PASSCODE_FAILED",
    9: "EMAIL_DOMAIN_NOT_SUPPORTED",
    98: "TEMP_DISABLED_IN_THIS_CHANNEL",
}

LOGIN_ERRORS = {
    2: "PASSWORD_OR_USER_NAME_IS_WRONG",
    10: "USER_IS_BANNED",
}


class RegError(Exception):
    def __init__(self, biz_code: int, biz_msg: str) -> None:
        super().__init__(f"DeepSeek reg error {biz_code}: {biz_msg}")
        self.biz_code = biz_code
        self.biz_msg = biz_msg


def new_device_id() -> str:
    return uuid.uuid4().hex


def guest_pow_header(salt: str, answer: int) -> dict[str, str]:
    raw = json.dumps({"salt": salt, "answer": answer}, separators=(",", ":")).encode()
    return {"X-DS-Guest-PoW-Response": base64.b64encode(raw).decode()}


class DeepSeekRegistrar:
    def __init__(self, device_id: str | None = None, timeout: float = 60.0, waf_token: str | None = None) -> None:
        self.device_id = device_id or new_device_id()
        self.http = httpx.AsyncClient(
            base_url=BASE_URL,
            headers={"User-Agent": USER_AGENT, **CLIENT_HEADERS},
            timeout=httpx.Timeout(timeout),
            follow_redirects=True,
        )
        if waf_token:
            self.http.cookies.set("aws-waf-token", waf_token, domain="chat.deepseek.com", path="/")

    async def aclose(self) -> None:
        await self.http.aclose()

    async def _post(self, path: str, json_body: dict, headers: dict[str, str] | None = None) -> dict:
        try:
            resp = await self.http.post(path, json=json_body, headers=headers)
        except httpx.HTTPError as exc:
            raise RegError(-1, f"http request failed: {exc}") from exc
        if resp.status_code == 202 and resp.headers.get("x-amzn-waf-action") == "challenge":
            raise RegError(
                202,
                "AWS WAF challenge on this endpoint: pass --waf-token (aws-waf-token cookie from a browser on chat.deepseek.com)",
            )
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise RegError(exc.response.status_code, exc.response.text[:300]) from exc
        try:
            payload = resp.json()
        except ValueError as exc:
            raise RegError(-1, f"invalid JSON from {path}: {resp.text[:200]}") from exc
        if not isinstance(payload, dict):
            raise RegError(-1, f"unexpected response from {path}: {resp.text[:200]}")
        return payload

    @staticmethod
    def _result(payload: dict, error_names: dict[int, str]) -> dict:
        data = payload.get("data")
        if not isinstance(data, dict):
            raise RegError(-1, f"missing data object in response: {json.dumps(payload)[:300]}")
        biz_code = data.get("biz_code")
        if biz_code != 0:
            name = error_names.get(biz_code, "") if isinstance(biz_code, int) else ""
            biz_msg = data.get("biz_msg") or name or "unknown error"
            raise RegError(biz_code if isinstance(biz_code, int) else -1, f"{name}: {biz_msg}" if name else biz_msg)
        biz_data = data.get("biz_data")
        return biz_data if isinstance(biz_data, dict) else {}

    async def _guest_pow(self, target_path: str) -> dict[str, str]:
        payload = await self._post(GUEST_CHALLENGE_PATH, {"target_path": target_path})
        data = payload.get("data")
        biz_data = data.get("biz_data") if isinstance(data, dict) else None
        challenge = biz_data.get("guest_challenge") if isinstance(biz_data, dict) else None
        if not isinstance(challenge, dict):
            raise RegError(-1, f"guest_challenge missing in response: {json.dumps(payload)[:300]}")
        expire_at = challenge.get("expire_at")
        difficulty = challenge.get("difficulty")
        if isinstance(expire_at, bool) or not isinstance(expire_at, (int, float)):
            raise RegError(-1, "guest challenge has invalid expire_at")
        if isinstance(difficulty, bool) or not isinstance(difficulty, (int, float)) or difficulty <= 0:
            raise RegError(-1, "guest challenge has invalid difficulty")
        answer = await solve_challenge(
            str(challenge.get("challenge") or ""),
            str(challenge.get("salt") or ""),
            int(expire_at),
            int(difficulty),
        )
        if answer is None:
            raise RegError(-1, "guest pow solver returned no answer")
        log.debug("guest pow solved for %s (answer=%s)", target_path, answer)
        return guest_pow_header(str(challenge.get("salt") or ""), answer)

    async def send_email_code(self, email: str, hcaptcha_token: str) -> None:
        pow_headers = await self._guest_pow(CREATE_EMAIL_CODE_PATH)
        body = {
            "email": email,
            "locale": LOCALE,
            "hcaptcha_token": hcaptcha_token,
            "device_id": self.device_id,
            "scenario": "register",
        }
        payload = await self._post(CREATE_EMAIL_CODE_PATH, body, pow_headers)
        self._result(payload, EMAIL_CODE_ERRORS)
        log.info("verification email sent to %s", email)

    async def register(self, email: str, password: str, code: str) -> str:
        pow_headers = await self._guest_pow(REGISTER_PATH)
        body = {
            "locale": LOCALE,
            "region": REGION,
            "payload": {
                "email": email,
                "email_verification_code": code,
                "password": password,
            },
            "device_id": self.device_id,
            "os": "web",
        }
        payload = await self._post(REGISTER_PATH, body, pow_headers)
        biz_data = self._result(payload, REGISTER_ERRORS)
        token = self._extract_token(biz_data, "register")
        log.info("register success for %s", email)
        return token

    async def login(self, email: str, password: str) -> str:
        body = {
            "email": email,
            "password": password,
            "device_id": self.device_id,
            "os": "web",
        }
        payload = await self._post(LOGIN_PATH, body)
        biz_data = self._result(payload, LOGIN_ERRORS)
        token = self._extract_token(biz_data, "login")
        log.info("login success for %s", email)
        return token

    async def current_user(self, token: str) -> dict:
        try:
            resp = await self.http.get(CURRENT_USER_PATH, headers={"Authorization": f"Bearer {token}"})
        except httpx.HTTPError as exc:
            raise RegError(-1, f"http request failed: {exc}") from exc
        if resp.status_code != 200:
            raise RegError(resp.status_code, f"token check failed: {resp.text[:200]}")
        try:
            payload = resp.json()
        except ValueError as exc:
            raise RegError(-1, "invalid JSON from users/current") from exc
        data = payload.get("data")
        biz_data = data.get("biz_data") if isinstance(data, dict) else None
        if not isinstance(biz_data, dict):
            raise RegError(-1, "users/current returned no user")
        return biz_data

    @staticmethod
    def _extract_token(biz_data: dict, stage: str) -> str:
        user = biz_data.get("user")
        token = user.get("token") if isinstance(user, dict) else None
        if not token or not isinstance(token, str):
            raise RegError(-1, f"{stage} succeeded but no token in response: {json.dumps(biz_data)[:300]}")
        return token
