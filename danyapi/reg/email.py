from __future__ import annotations

import asyncio
import email.header
import email.message
import imaplib
import logging
import re
import time
from collections.abc import Callable

log = logging.getLogger("danyapi.reg.email")


class EmailCodeError(Exception):
    pass


class CodeSource:
    async def wait_for_code(self, email_address: str) -> str:
        raise NotImplementedError


class StaticCodeSource(CodeSource):
    def __init__(self, code: str) -> None:
        self.code = code

    async def wait_for_code(self, email_address: str) -> str:
        return self.code


class ManualCodeSource(CodeSource):
    def __init__(self, input_func: Callable[[str], str] = input, print_func: Callable[[object], None] = print) -> None:
        self._input = input_func
        self._print = print_func

    async def wait_for_code(self, email_address: str) -> str:
        self._print(f"Enter the verification code sent to {email_address}:")
        code = (await asyncio.to_thread(self._input, "code: ")).strip()
        if not code:
            raise EmailCodeError("empty verification code")
        return code


_CODE_RE = re.compile(r"\b(\d{6})\b")
_CODE_FALLBACK_RE = re.compile(r"\b(\d{4,8})\b")


def extract_code(text: str) -> str | None:
    match = _CODE_RE.search(text)
    if match:
        return match.group(1)
    fallback = _CODE_FALLBACK_RE.search(text)
    return fallback.group(1) if fallback else None


def _decode_header_value(raw: str) -> str:
    chunks: list[str] = []
    for value, charset in email.header.decode_header(raw):
        if isinstance(value, bytes):
            chunks.append(value.decode(charset or "utf-8", errors="replace"))
        else:
            chunks.append(value)
    return "".join(chunks)


def _message_text(message: email.message.Message) -> str:
    chunks: list[str] = []
    for part in message.walk():
        if part.get_content_type() not in ("text/plain", "text/html"):
            continue
        payload = part.get_payload(decode=True)
        if not isinstance(payload, bytes):
            continue
        charset = part.get_content_charset() or "utf-8"
        chunks.append(payload.decode(charset, errors="replace"))
    return "\n".join(chunks)


class ImapCodeSource(CodeSource):
    def __init__(
        self,
        host: str,
        username: str,
        password: str,
        port: int = 993,
        folder: str = "INBOX",
        sender_filter: str = "deepseek",
        wait_seconds: float = 120.0,
        poll_interval: float = 5.0,
        lookback_seconds: float = 900.0,
    ) -> None:
        self.host = host
        self.username = username
        self.password = password
        self.port = port
        self.folder = folder
        self.sender_filter = sender_filter
        self.wait_seconds = wait_seconds
        self.poll_interval = poll_interval
        self.lookback_seconds = lookback_seconds

    async def wait_for_code(self, email_address: str) -> str:
        deadline = time.monotonic() + self.wait_seconds
        code = await asyncio.to_thread(self._fetch_code_sync, deadline)
        if not code:
            raise EmailCodeError(f"no verification code found in {self.username}:{self.folder} within {self.wait_seconds:.0f}s")
        return code

    def _fetch_code_sync(self, deadline: float) -> str | None:
        while True:
            try:
                code = self._poll_once()
            except imaplib.IMAP4.error as exc:
                raise EmailCodeError(f"imap error: {exc}") from exc
            if code:
                return code
            if time.monotonic() >= deadline:
                return None
            time.sleep(self.poll_interval)

    def _poll_once(self) -> str | None:
        client = imaplib.IMAP4_SSL(self.host, self.port)
        try:
            client.login(self.username, self.password)
            client.select(self.folder, readonly=True)
            since = time.strftime("%d-%b-%Y", time.gmtime(time.time() - self.lookback_seconds))
            status, data = client.search(None, f'(SINCE "{since}" FROM "{self.sender_filter}")')
            if status != "OK":
                return None
            message_ids = (data[0] or b"").split()
            for message_id in reversed(message_ids):
                status, message_data = client.fetch(message_id.decode("ascii"), "(RFC822)")
                if status != "OK" or not message_data or message_data[0] is None:
                    continue
                raw = message_data[0][1]
                if not isinstance(raw, bytes):
                    continue
                message = email.message_from_bytes(raw)
                code = extract_code(_message_text(message))
                if code:
                    return code
            return None
        finally:
            try:
                client.logout()
            except Exception as exc:
                log.debug("logout failed: %s", exc)
