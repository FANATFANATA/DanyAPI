from __future__ import annotations

import asyncio
import base64
import logging
from dataclasses import dataclass

from fastapi import HTTPException

from ..deepseek.client import DeepSeekError
from .powauth import _deepseek_status, _fresh_pow_upload_headers, _handle_account_error
from .schemas import ChatCompletionRequest

log = logging.getLogger("danyapi.api")


MAX_FILES_PER_REQUEST = 50
MAX_FILE_SIZE = 100 * 1024 * 1024
MAX_ATTACHMENT_TOTAL_SIZE = 10 * 1024 * 1024


def _data_uri_parts(uri: str) -> tuple[str, str]:
    if not uri.startswith("data:"):
        raise HTTPException(400, "image_url must be a data URI (data:<mime>;base64,...)")
    meta, _, payload = uri[5:].partition(",")
    if not payload:
        raise HTTPException(400, "invalid data URI: missing base64 payload")
    return meta, "".join(payload.split())


def _compact_data_uri_length(compact: str) -> int:
    stripped = compact.rstrip("=")
    units, remainder = divmod(len(stripped), 4)
    decoded = units * 3
    if remainder == 2:
        decoded += 1
    elif remainder == 3:
        decoded += 2
    return decoded


def _raw_data_uri_length(uri: str) -> int:
    _meta, compact = _data_uri_parts(uri)
    return _compact_data_uri_length(compact)


@dataclass
class Attachment:
    data: bytes
    name: str
    content_type: str
    is_image: bool


def _decode_data_uri(meta: str, compact: str) -> tuple[str, bytes]:
    content_type = meta.split(";", 1)[0].strip() or "application/octet-stream"
    try:
        data = base64.b64decode(compact, validate=True)
    except ValueError as exc:
        raise HTTPException(400, "invalid base64 in image_url") from exc
    return content_type, data


def _split_data_uri(uri: str) -> tuple[str, bytes]:
    return _decode_data_uri(*_data_uri_parts(uri))


REMOTE_IMAGE_SCHEMES = ("http://", "https://")


def _collect_attachments(req: ChatCompletionRequest, allow_remote: bool = False) -> list[Attachment]:
    attachments: list[Attachment] = []
    raw_total = 0
    for msg in req.messages:
        if not isinstance(msg.content, list):
            continue
        for item in msg.content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "image_url":
                image_url = item.get("image_url")
                if isinstance(image_url, str):
                    uri = image_url
                elif isinstance(image_url, dict) and isinstance(image_url.get("url"), str):
                    uri = image_url["url"]
                else:
                    raise HTTPException(400, "invalid image_url value")
                if allow_remote and uri.startswith(REMOTE_IMAGE_SCHEMES):
                    continue
                raw_total += _raw_data_uri_length(uri)
                if raw_total > MAX_ATTACHMENT_TOTAL_SIZE:
                    raise HTTPException(413, "attachments too large")
                content_type, data = _split_data_uri(uri)
                name = f"image_{len(attachments)}.{content_type.split('/')[-1] or 'bin'}"
                attachments.append(Attachment(data, name, content_type, True))
    for f in req.files or []:
        if not f.name or not f.content:
            raise HTTPException(400, "each file needs name and base64 content")
        try:
            data = base64.b64decode(f.content)
        except ValueError as exc:
            raise HTTPException(400, f"invalid base64 in file {f.name}") from exc
        attachments.append(Attachment(data, f.name, f.content_type or "application/octet-stream", (f.content_type or "").startswith("image/")))
    return attachments


def _validate_attachments(attachments: list[Attachment]) -> None:
    if not attachments:
        return
    if len(attachments) > MAX_FILES_PER_REQUEST:
        raise HTTPException(400, f"too many files: max {MAX_FILES_PER_REQUEST} per request")
    for att in attachments:
        if len(att.data) > MAX_FILE_SIZE:
            raise HTTPException(400, f"file {att.name} exceeds {MAX_FILE_SIZE // (1024 * 1024)} MB limit")


async def _upload_attachments(account, attachments: list[Attachment], model_type: str, thinking: bool) -> list[str]:
    if not attachments:
        return []
    pow_headers_list = await asyncio.gather(*(_fresh_pow_upload_headers(account) for _ in attachments))
    file_ids: list[str] = []
    sem = asyncio.Semaphore(4)

    async def _upload_one(att: Attachment, pow_headers) -> str:
        async with sem:
            try:
                info = await account.client.upload_file(
                    att.data,
                    att.name,
                    att.content_type,
                    model_type,
                    thinking_enabled=thinking,
                    pow_headers=pow_headers,
                )
            except DeepSeekError as exc:
                _handle_account_error(account, exc)
                raise HTTPException(_deepseek_status(exc), f"file upload failed: {exc}") from exc
            file_id = info.get("id")
            if not file_id:
                raise HTTPException(502, f"file upload failed for {att.name}: no file id")
            return file_id

    results = await asyncio.gather(
        *(_upload_one(att, pow_headers) for att, pow_headers in zip(attachments, pow_headers_list, strict=True)),
        return_exceptions=True,
    )
    for item in results:
        if isinstance(item, BaseException):
            raise item
        file_ids.append(item)
    return file_ids
