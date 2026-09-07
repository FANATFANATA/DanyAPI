# Source reference:
# https://github.com/youssefvdel/qwengate/blob/dev/src/services/qwenFileUpload.ts
# Implements Qwen web UI direct-to-Alibaba OSS file upload and attachment flow.

from __future__ import annotations

import asyncio
import base64
import email.utils
import hmac
import logging
import time
import uuid
from typing import TYPE_CHECKING, Any

import httpx

if TYPE_CHECKING:
    from .client import QwenClient

__all__ = [
    "build_oss_canonical_request",
    "build_qwen_file_attachment",
    "hmac_sha1_base64",
    "parse_and_poll",
    "upload_to_oss",
]

log = logging.getLogger("danyapi.qwen.upload")


def hmac_sha1_base64(key: str, message: str) -> str:
    """HMAC-SHA1 Base64 digest for Alibaba OSS authorization."""
    sig = hmac.new(key.encode("utf-8"), message.encode("utf-8"), "sha1").digest()
    return base64.b64encode(sig).decode("utf-8")


def build_oss_canonical_request(
    method: str,
    content_type: str,
    date_str: str,
    security_token: str,
    bucket: str,
    key: str,
) -> str:
    """Builds CanonicalizedOSSHeaders and CanonicalizedResource string.

    Format:
        VERB + "\\n"
        + Content-MD5 + "\\n"
        + Content-Type + "\\n"
        + Date + "\\n"
        + CanonicalizedOSSHeaders
        + CanonicalizedResource
    """
    return "\n".join(
        [
            method,
            "",  # Content-MD5 (empty)
            content_type,
            date_str,
            f"x-oss-security-token:{security_token}",
            f"/{bucket}/{key}",
        ]
    )


async def upload_to_oss(
    http_client: httpx.AsyncClient,
    sts: dict[str, Any],
    file_bytes: bytes,
    content_type: str,
) -> str:
    """Uploads raw binary bytes to Alibaba Cloud OSS bucket using STS credentials.

    Ref: qwengate/src/services/qwenFileUpload.ts:uploadToOss
    """
    date_str = email.utils.formatdate(usegmt=True)
    key = sts.get("file_path", "")
    bucket = sts.get("bucketname", "qwen-webui-prod")

    bucket_prefix = f"{bucket}/"
    object_key = key.removeprefix(bucket_prefix)

    canonical_req = build_oss_canonical_request(
        method="PUT",
        content_type=content_type,
        date_str=date_str,
        security_token=sts["security_token"],
        bucket=bucket,
        key=object_key,
    )

    signature = hmac_sha1_base64(sts["access_key_secret"], canonical_req)
    auth_header = f"OSS {sts['access_key_id']}:{signature}"

    endpoint = sts.get("endpoint", "").rstrip("/")
    if bucket not in endpoint:
        clean_endpoint = endpoint.replace("https://", "").replace("http://", "")
        endpoint = f"https://{bucket}.{clean_endpoint}"
    upload_url = f"{endpoint}/{object_key}"

    headers = {
        "Content-Type": content_type,
        "Date": date_str,
        "Authorization": auth_header,
        "x-oss-security-token": sts["security_token"],
    }

    log.debug("uploading %d bytes to OSS: %s", len(file_bytes), upload_url)
    resp = await http_client.put(upload_url, content=file_bytes, headers=headers, timeout=60.0)
    if resp.status_code not in (200, 204):
        body_sample = resp.text[:300] if hasattr(resp, "text") else ""
        raise RuntimeError(f"OSS upload failed ({resp.status_code}): {body_sample}")

    return sts.get("file_url", upload_url)


def build_qwen_file_attachment(
    sts: dict[str, Any],
    filename: str,
    filesize: int,
    content_type: str,
    attachment_type: str = "file",
) -> dict[str, Any]:
    """Builds nested file descriptor matching Qwen web UI messages[].files[] format.

    Ref: qwengate/src/services/qwenFileUpload.ts:buildQwenFileAttachment
    """
    file_path = sts.get("file_path", "")
    user_id = file_path.split("/")[0] if "/" in file_path else ""
    file_id = sts.get("file_id", "")
    file_url = sts.get("file_url", "")
    now_ms = int(time.time() * 1000)

    is_image = attachment_type == "image" or content_type.startswith("image/")
    att_type = "image" if is_image else "file"
    file_class = "vision" if is_image else "document"
    show_type = "image" if is_image else "file"

    meta: dict[str, Any] = {
        "name": filename,
        "size": filesize,
        "content_type": content_type,
    }
    if not is_image:
        meta["parse_meta"] = {"parse_status": "success"}

    return {
        "type": att_type,
        "file": {
            "created_at": now_ms,
            "data": {},
            "filename": filename,
            "hash": None,
            "id": file_id,
            "user_id": user_id,
            "meta": meta,
            "update_at": now_ms,
            "lastModified": now_ms,
            "name": filename,
            "webkitRelativePath": "",
            "size": filesize,
            "type": content_type,
        },
        "id": file_id,
        "url": file_url,
        "name": filename,
        "collection_name": "",
        "progress": 0,
        "status": "uploaded",
        "greenNet": "success",
        "size": filesize,
        "error": "",
        "itemId": str(uuid.uuid4()),
        "file_type": content_type,
        "showType": show_type,
        "file_class": file_class,
        "uploadTaskId": str(uuid.uuid4()),
    }


async def parse_and_poll(
    client: QwenClient,
    file_id: str,
    max_wait_sec: float = 5.0,
) -> None:
    """Triggers server-side text/doc parsing and polls until complete.

    Images do not require this step — Qwen vision processes them directly from OSS.
    Ref: qwengate/src/services/qwenFileUpload.ts:parseFile & pollParseStatus
    """
    try:
        await client._post("/api/v2/files/parse", json_body={"file_id": file_id})
    except Exception as exc:
        log.warning("file parse trigger failed for %s: %s", file_id, exc)
        return

    start_time = time.monotonic()
    while time.monotonic() - start_time < max_wait_sec:
        try:
            resp = await client._post(
                "/api/v2/files/parse/status",
                json_body={"file_id_list": [file_id]},
            )
            # Response: {"data": [{"file_id": "...", "status": "success"}]}
            items = resp if isinstance(resp, list) else resp.get("data", [])
            if isinstance(items, list) and items:
                st = items[0].get("status")
                if st == "success":
                    log.debug("parse complete for %s in %.2fs", file_id, time.monotonic() - start_time)
                    return
                if st == "failed":
                    log.warning("file parsing failed for %s", file_id)
                    return
        except Exception as exc:
            log.debug("error polling parse status for %s: %s", file_id, exc)
        await asyncio.sleep(1.0)
