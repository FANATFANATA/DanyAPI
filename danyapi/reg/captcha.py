from __future__ import annotations

import asyncio
import time
from collections.abc import Callable

import httpx

HCAPTCHA_SITEKEY = "352e5376-f2cc-43fe-a744-e51640449610"
HCAPTCHA_PAGE_URL = "https://chat.deepseek.com/sign_up"


class CaptchaError(Exception):
    pass


class HcaptchaSolver:
    async def solve(self) -> str:
        raise NotImplementedError


class StaticSolver(HcaptchaSolver):
    def __init__(self, token: str) -> None:
        self.token = token

    async def solve(self) -> str:
        return self.token


class ManualCaptchaSolver(HcaptchaSolver):
    def __init__(self, input_func: Callable[[str], str] = input, print_func: Callable[[object], None] = print) -> None:
        self._input = input_func
        self._print = print_func

    async def solve(self) -> str:
        self._print("An hCaptcha token is required to send the DeepSeek verification email.")
        self._print("Get one in a browser: open https://chat.deepseek.com/sign_up, click Send code,")
        self._print("then copy hcaptcha_token from the create_email_verification_code request body (DevTools -> Network).")
        token = (await asyncio.to_thread(self._input, "hCaptcha token: ")).strip()
        if not token:
            raise CaptchaError("empty hCaptcha token")
        return token


def _json_body(resp: httpx.Response, source: str) -> dict:
    try:
        data = resp.json()
    except ValueError as exc:
        raise CaptchaError(f"invalid JSON from {source}: {resp.text[:200]}") from exc
    if not isinstance(data, dict):
        raise CaptchaError(f"unexpected response from {source}: {resp.text[:200]}")
    return data


class TwoCaptchaSolver(HcaptchaSolver):
    def __init__(self, api_key: str, timeout: float = 180.0, poll_interval: float = 5.0) -> None:
        self.api_key = api_key
        self.timeout = timeout
        self.poll_interval = poll_interval

    async def solve(self) -> str:
        async with httpx.AsyncClient(timeout=30.0) as http:
            try:
                resp = await http.get(
                    "https://2captcha.com/in.php",
                    params={
                        "key": self.api_key,
                        "method": "hcaptcha",
                        "sitekey": HCAPTCHA_SITEKEY,
                        "pageurl": HCAPTCHA_PAGE_URL,
                        "json": "1",
                    },
                )
            except httpx.HTTPError as exc:
                raise CaptchaError(f"2captcha submit failed: {exc}") from exc
            data = _json_body(resp, "2captcha in.php")
            if data.get("status") != 1:
                raise CaptchaError(f"2captcha submit failed: {data.get('request')}")
            task_id = data["request"]
            deadline = time.monotonic() + self.timeout
            while time.monotonic() < deadline:
                await asyncio.sleep(self.poll_interval)
                try:
                    resp = await http.get(
                        "https://2captcha.com/res.php",
                        params={"key": self.api_key, "action": "get", "id": task_id, "json": "1"},
                    )
                except httpx.HTTPError as exc:
                    raise CaptchaError(f"2captcha poll failed: {exc}") from exc
                data = _json_body(resp, "2captcha res.php")
                if data.get("status") == 1:
                    token = str(data.get("request") or "")
                    if not token:
                        raise CaptchaError("2captcha returned an empty token")
                    return token
                if data.get("request") != "CAPCHA_NOT_READY":
                    raise CaptchaError(f"2captcha solve failed: {data.get('request')}")
        raise CaptchaError("2captcha solve timed out")


class CapSolverSolver(HcaptchaSolver):
    def __init__(self, api_key: str, timeout: float = 180.0, poll_interval: float = 3.0) -> None:
        self.api_key = api_key
        self.timeout = timeout
        self.poll_interval = poll_interval

    async def solve(self) -> str:
        async with httpx.AsyncClient(timeout=30.0) as http:
            try:
                resp = await http.post(
                    "https://api.capsolver.com/createTask",
                    json={
                        "clientKey": self.api_key,
                        "task": {
                            "type": "HCaptchaTaskProxyLess",
                            "websiteURL": HCAPTCHA_PAGE_URL,
                            "websiteKey": HCAPTCHA_SITEKEY,
                        },
                    },
                )
            except httpx.HTTPError as exc:
                raise CaptchaError(f"capsolver createTask failed: {exc}") from exc
            data = _json_body(resp, "capsolver createTask")
            task_id = data.get("taskId")
            if not task_id:
                raise CaptchaError(f"capsolver createTask failed: {data.get('errorDescription') or data}")
            deadline = time.monotonic() + self.timeout
            while time.monotonic() < deadline:
                await asyncio.sleep(self.poll_interval)
                try:
                    resp = await http.post(
                        "https://api.capsolver.com/getTaskResult",
                        json={"clientKey": self.api_key, "taskId": task_id},
                    )
                except httpx.HTTPError as exc:
                    raise CaptchaError(f"capsolver poll failed: {exc}") from exc
                data = _json_body(resp, "capsolver getTaskResult")
                status = data.get("status")
                if status == "ready":
                    token = str((data.get("solution") or {}).get("gRecaptchaResponse") or "")
                    if not token:
                        raise CaptchaError("capsolver returned an empty token")
                    return token
                if status != "processing":
                    raise CaptchaError(f"capsolver solve failed: {data.get('errorDescription') or data}")
        raise CaptchaError("capsolver solve timed out")
