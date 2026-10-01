from __future__ import annotations

import asyncio
import base64
import ipaddress
import logging
import re
import socket
import time
from typing import Any
from urllib.parse import urlsplit

import httpx
from fastapi import File, Form, HTTPException, Request, UploadFile
from pydantic import ValidationError

from ..accounts import AccountPool, AccountPoolBusy
from ..qwen import api as qwen_api
from .attachments import MAX_FILE_SIZE
from .byok import _byok_pool_for
from .core import _validation_summary
from .schemas import ImageGenerationRequest
from .shaping import _merge_usage
from .state import _byok_mode, app

log = logging.getLogger("danyapi.api")


IMAGE_SIZE_RE = re.compile(r"^(\d{2,5})\s*[*x\u00d7,]\s*(\d{2,5})$", re.IGNORECASE)
MIN_IMAGE_DIM = 16
MAX_IMAGE_DIM = 8192

IMAGE_FETCH_SCHEMES = ("http", "https")
IMAGE_FETCH_REDIRECTS = 3
IMAGE_FETCH_TIMEOUT_SEC = 30.0
REDIRECT_STATUSES = (301, 302, 303, 307, 308)
IMAGE_DOWNLOAD_CONCURRENCY = 4
QWEN_UNAVAILABLE_MESSAGE = "qwen provider is not configured (required for image generation)"
UPLOAD_CHUNK_SIZE = 64 * 1024


def _parse_image_size(size: str | None) -> tuple[int, int] | None:
    if size is None or not size.strip():
        return None
    match = IMAGE_SIZE_RE.fullmatch(size.strip())
    if match is None:
        raise HTTPException(400, f"invalid size {size!r}: expected WIDTHxHEIGHT (e.g. 1152x2048 or 1152*2048)")
    width, height = int(match.group(1)), int(match.group(2))
    if not (MIN_IMAGE_DIM <= width <= MAX_IMAGE_DIM and MIN_IMAGE_DIM <= height <= MAX_IMAGE_DIM):
        raise HTTPException(400, f"size out of range: both dimensions must be within {MIN_IMAGE_DIM}..{MAX_IMAGE_DIM}")
    return width, height


def _resize_image_bytes(content: bytes, dims: tuple[int, int] | None) -> bytes:
    if dims is None:
        return content
    try:
        from io import BytesIO

        from PIL import Image

        with Image.open(BytesIO(content)) as img:
            fmt = img.format or "PNG"
            resized = img.resize(dims, Image.Resampling.LANCZOS)
            if fmt.upper() == "JPEG" and resized.mode not in ("RGB", "L"):
                resized = resized.convert("RGB")
            buffer = BytesIO()
            resized.save(buffer, format=fmt)
            return buffer.getvalue()
    except Exception as exc:
        log.warning("image resize to %s failed, returning original: %s", dims, exc)
        return content


def _host_is_public(host: str) -> bool:
    try:
        infos = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except (OSError, UnicodeError):
        return False
    if not infos:
        return False
    for info in infos:
        try:
            address = ipaddress.ip_address(info[4][0])
        except (IndexError, ValueError):
            return False
        if not address.is_global or address.is_multicast:
            return False
    return True


def _peer_is_public(response: httpx.Response) -> bool:
    stream = (getattr(response, "extensions", None) or {}).get("network_stream")
    get_extra_info = getattr(stream, "get_extra_info", None)
    if get_extra_info is None:
        return True
    try:
        peer = get_extra_info("server_addr")
    except Exception:
        return True
    host = peer[0] if isinstance(peer, tuple) and peer else peer
    if not isinstance(host, str):
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    return address.is_global and not address.is_multicast


async def _check_image_url(url: str) -> None:
    parts = urlsplit(url)
    if parts.scheme not in IMAGE_FETCH_SCHEMES:
        raise ValueError(f"unsupported image url scheme {parts.scheme or url[:16]!r}")
    host = parts.hostname or ""
    if not host:
        raise ValueError("image url has no host")
    if not await asyncio.to_thread(_host_is_public, host):
        raise ValueError(f"refusing to fetch image from the non-public address {host}")


async def _download_image(hc: httpx.AsyncClient, url: str) -> bytes:
    current = url
    for _ in range(IMAGE_FETCH_REDIRECTS + 1):
        await _check_image_url(current)
        async with hc.stream("GET", current, timeout=IMAGE_FETCH_TIMEOUT_SEC, follow_redirects=False) as resp:
            if not _peer_is_public(resp):
                raise ValueError("image host resolved to a non-public address on connect")
            if resp.status_code in REDIRECT_STATUSES:
                location = resp.headers.get("location")
                if not location:
                    raise ValueError("image redirect without a location header")
                current = str(resp.url.join(location))
                continue
            if resp.status_code != 200:
                raise ValueError(f"image download failed with status {resp.status_code}")
            payload = bytearray()
            async for chunk in resp.aiter_bytes():
                payload += chunk
                if len(payload) > MAX_FILE_SIZE:
                    raise ValueError(f"image exceeds {MAX_FILE_SIZE // (1024 * 1024)} MB")
            return bytes(payload)
    raise ValueError("image download exceeded the redirect limit")


@app.post("/v1/images/generations")
async def image_generations(req: ImageGenerationRequest, request: Request) -> dict:
    return await _image_generations(req, await _image_pool(request))


_image_client_lock = asyncio.Lock()


async def _image_http_client() -> httpx.AsyncClient:
    client = getattr(app.state, "http_client", None)
    if client is not None:
        return client
    async with _image_client_lock:
        client = getattr(app.state, "http_client", None)
        if client is None:
            client = httpx.AsyncClient(follow_redirects=True, timeout=30)
            app.state.http_client = client
    return client


async def _image_generations(req: ImageGenerationRequest, pool: AccountPool | None = None) -> dict:
    if pool is None:
        raise HTTPException(503, QWEN_UNAVAILABLE_MESSAGE)

    from .chats import _acquire_session_account

    dims = _parse_image_size(req.size)
    count = max(1, int(getattr(req, "n", 1) or 1))
    want_b64 = req.response_format == "b64_json"
    if dims is not None and not want_b64:
        raise HTTPException(400, "size requires response_format=b64_json")

    account, existing_sid = await _acquire_session_account(pool, req)

    data: list[dict] = []
    usage = None
    result_sid = existing_sid
    hc = await _image_http_client()
    download_sem = asyncio.Semaphore(IMAGE_DOWNLOAD_CONCURRENCY)

    async def _fetch_image(url: str) -> dict:
        if not want_b64:
            return {"url": url}
        async with download_sem:
            try:
                payload_bytes = await _download_image(hc, url)
            except Exception as exc:
                log.warning("image download failed for %s: %s", url, exc)
                raise HTTPException(502, "image download failed") from exc
            if dims is not None:
                payload_bytes = await asyncio.to_thread(_resize_image_bytes, payload_bytes, dims)
            return {"b64_json": await _b64encode(payload_bytes)}

    try:
        for _ in range(count):
            result = await qwen_api.collect_image(
                account=account,
                pool=pool,
                existing_sid=result_sid,
                lock=account.sem,
                prompt=req.prompt,
                model=req.model,
                model_id=req.model,
                user=req.user,
            )
            result_sid = result.get("session_id") or result_sid
            step_usage = result.get("usage")
            if step_usage:
                usage = _merge_usage(usage, step_usage)
            if result["image_urls"]:
                settled = await asyncio.gather(*(_fetch_image(url) for url in result["image_urls"]), return_exceptions=True)
                fetched: list[dict] = []
                for item in settled:
                    if isinstance(item, BaseException):
                        raise item
                    fetched.append(item)
                data.extend(fetched)
    except AccountPoolBusy:
        raise HTTPException(429, "all accounts are busy, try again later") from None

    if not data:
        raise HTTPException(502, "image generation returned no data")

    return {
        "created": int(time.time()),
        "data": data,
        "usage": usage,
        "session_id": result_sid,
    }


async def _image_pool(request: Request) -> AccountPool:
    pool: AccountPool | None
    if _byok_mode():
        pool = await _byok_pool_for("qwen", request)
    else:
        pool = getattr(app.state, "qwen_pool", None)
    if pool is None:
        raise HTTPException(503, QWEN_UNAVAILABLE_MESSAGE)
    return pool


_ASYNC_B64_THRESHOLD = 1 << 20


async def _b64encode(data: bytes) -> str:
    if len(data) > _ASYNC_B64_THRESHOLD:
        data = await asyncio.to_thread(base64.b64encode, data)
    else:
        data = base64.b64encode(data)
    return data.decode("ascii")


async def _image_markdown(data: bytes, content_type: str) -> str:
    return f"![image](data:{content_type or 'image/png'};base64,{await _b64encode(data)})"


async def _read_upload(file: UploadFile) -> tuple[bytes, str]:
    chunks: list[bytes] = []
    total = 0
    while True:
        remaining = MAX_FILE_SIZE - total
        chunk = await file.read(min(UPLOAD_CHUNK_SIZE, remaining + 1))
        if not chunk:
            break
        total += len(chunk)
        if total > MAX_FILE_SIZE:
            raise HTTPException(413, f"uploaded file exceeds {MAX_FILE_SIZE // (1024 * 1024)} MB limit")
        chunks.append(chunk)
        if len(chunk) < UPLOAD_CHUNK_SIZE:
            break
    content_type = (file.content_type or "application/octet-stream").split(";", 1)[0].strip() or "application/octet-stream"
    return b"".join(chunks), content_type


def _image_edit_req(prompt: str, image_md: str, mask_md: str | None) -> str:
    parts: list[str] = []
    if prompt.strip():
        parts.append(prompt.strip())
    parts.append(image_md)
    if mask_md:
        parts.append(mask_md)
    return "\n".join(parts)


def _image_request(**fields: Any) -> ImageGenerationRequest:
    try:
        return ImageGenerationRequest(**fields)
    except ValidationError as exc:
        raise HTTPException(400, f"invalid request body: {_validation_summary(exc.errors())}") from exc


@app.post("/v1/images/edits")
async def image_edits(
    request: Request,
    image: UploadFile = File(...),
    prompt: str = Form(default=""),
    mask: UploadFile | None = File(default=None),
    model: str = Form(default="qwen-image-gen"),
    n: int = Form(default=1),
    size: str | None = Form(default=None),
    response_format: str = Form(default="url"),
    user: str | None = Form(default=None),
    session_id: str | None = Form(default=None),
) -> dict:
    pool = await _image_pool(request)
    image_data, image_type = await _read_upload(image)
    mask_md = None
    if mask is not None:
        mask_data, mask_type = await _read_upload(mask)
        mask_md = await _image_markdown(mask_data, mask_type)
    req = _image_request(
        model=model,
        prompt=_image_edit_req(prompt, await _image_markdown(image_data, image_type), mask_md),
        n=n,
        size=size,
        response_format=response_format,
        session_id=session_id,
        user=user,
    )
    return await _image_generations(req, pool)


@app.post("/v1/images/variations")
async def image_variations(
    request: Request,
    image: UploadFile = File(...),
    prompt: str = Form(default=""),
    model: str = Form(default="qwen-image-gen"),
    n: int = Form(default=1),
    size: str | None = Form(default=None),
    response_format: str = Form(default="url"),
    user: str | None = Form(default=None),
    session_id: str | None = Form(default=None),
) -> dict:
    pool = await _image_pool(request)
    image_data, image_type = await _read_upload(image)
    req = _image_request(
        model=model,
        prompt=_image_edit_req(prompt, await _image_markdown(image_data, image_type), None),
        n=n,
        size=size,
        response_format=response_format,
        session_id=session_id,
        user=user,
    )
    return await _image_generations(req, pool)
