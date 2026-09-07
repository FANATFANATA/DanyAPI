from unittest.mock import AsyncMock, MagicMock

import pytest

from danyapi.qwen.client import QwenClient
from danyapi.qwen.upload import (
    build_oss_canonical_request,
    build_qwen_file_attachment,
    hmac_sha1_base64,
)


def test_hmac_sha1_base64():
    key = "my-secret-key"
    message = "test-message-to-sign"
    sig = hmac_sha1_base64(key, message)
    assert isinstance(sig, str)
    assert len(sig) > 0
    # Deterministic check
    assert sig == hmac_sha1_base64(key, message)


def test_build_oss_canonical_request():
    req = build_oss_canonical_request(
        method="PUT",
        content_type="image/jpeg",
        date_str="Mon, 07 Sep 2026 19:42:56 GMT",
        security_token="tok123",
        bucket="qwen-webui-prod",
        key="user1/file1_test.jpg",
    )
    expected = "PUT\n\nimage/jpeg\nMon, 07 Sep 2026 19:42:56 GMT\nx-oss-security-token:tok123\n/qwen-webui-prod/user1/file1_test.jpg"
    assert req == expected


def test_build_qwen_file_attachment():
    sts = {
        "file_id": "fid-123",
        "file_url": "https://oss.example.com/u1/fid-123_pic.jpg",
        "file_path": "u1/fid-123_pic.jpg",
    }
    att = build_qwen_file_attachment(sts, "pic.jpg", 1024, "image/jpeg", "image")
    assert att["type"] == "image"
    assert att["id"] == "fid-123"
    assert att["file"]["id"] == "fid-123"
    assert att["file"]["user_id"] == "u1"
    assert att["file"]["size"] == 1024
    assert att["showType"] == "image"
    assert att["file_class"] == "vision"


def test_qwen_client_cookie_parsing():
    raw_cookie = "cna=test_cna; _bl_uid=uid123; token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6IjEyMyJ9.sig; atpsida=atp123; isg=isg123"
    client = QwenClient(token=raw_cookie)
    assert client.token == "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6IjEyMyJ9.sig"
    assert client.http.headers.get("Authorization") == f"Bearer {client.token}"
    assert client.http.cookies.get("token") == client.token
    assert client.http.cookies.get("cna") == "test_cna"
    assert client.http.cookies.get("_bl_uid") == "uid123"
    assert client.http.cookies.get("atpsida") == "atp123"
    assert client.http.cookies.get("isg") == "isg123"


@pytest.mark.asyncio
async def test_qwen_upload_file_flow():
    client = QwenClient(token="jwt-token")
    mock_sts = {
        "file_id": "file-qwen-999",
        "file_url": "https://qwen-webui-prod.oss-accelerate.aliyuncs.com/u1/file-qwen-999_test.jpg",
        "file_path": "u1/file-qwen-999_test.jpg",
        "bucketname": "qwen-webui-prod",
        "endpoint": "https://oss-accelerate.aliyuncs.com",
        "access_key_id": "STS.KEY",
        "access_key_secret": "SECRET",
        "security_token": "SEC_TOK",
    }
    client._post = AsyncMock(return_value={"success": True, "data": mock_sts})
    client.http.put = AsyncMock(return_value=MagicMock(status_code=200))

    att = await client.upload_file(b"image-data", "test.jpg", "image/jpeg")

    assert att["id"] == "file-qwen-999"
    assert att["type"] == "image"
    assert att["file"]["size"] == 10
    client._post.assert_awaited_once_with(
        "/api/v2/files/getstsToken",
        json_body={"filename": "test.jpg", "filesize": "10", "filetype": "image"},
    )
    client.http.put.assert_awaited_once()
