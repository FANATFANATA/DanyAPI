import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from danyapi.deepseek.client import (
    CLIENT_HEADERS,
    DeepSeekClient,
    DeepSeekError,
    DeepSeekSession,
    new_device_id,
)


def make_resp(payload=None, status=200, text=""):
    resp = SimpleNamespace(status_code=status, text=text)
    if payload is not None:
        resp.json = MagicMock(return_value=payload)
    else:
        resp.json = MagicMock(side_effect=ValueError("no json"))
    resp.raise_for_status = MagicMock()
    return resp


def make_client():
    client = DeepSeekClient(token="tok")
    client.http = MagicMock()
    return client


async def test_new_device_id_is_uuid():
    value = new_device_id()
    assert str(uuid.UUID(value)) == value


async def test_session_defaults():
    s = DeepSeekSession(id="c1")
    assert s.title == ""
    assert s.last_message_id is None
    assert s.accumulated_tokens == 0
    assert s.extra == {}


async def test_error_attrs():
    err = DeepSeekError(40001, "bad token")
    assert err.biz_code == 40001
    assert err.biz_msg == "bad token"
    assert "40001" in str(err)


async def test_headers_with_token():
    client = DeepSeekClient(token="abc")
    assert client.http.headers["Authorization"] == "Bearer abc"
    for key in CLIENT_HEADERS:
        assert client.http.headers[key] == CLIENT_HEADERS[key]


async def test_aclose():
    client = DeepSeekClient()
    client.http = MagicMock()
    client.http.aclose = AsyncMock()

    await client.aclose()
    client.http.aclose.assert_awaited_once()


async def test_post_success():
    client = make_client()
    resp = make_resp({"code": 0, "data": {"biz_data": {"x": 1}}})
    client.http.post = AsyncMock(return_value=resp)

    result = await client._post("/api/x", {"a": 1})
    assert result == {"code": 0, "data": {"biz_data": {"x": 1}}}
    client.http.post.assert_awaited_once_with("/api/x", json={"a": 1})


async def test_post_http_error():
    client = make_client()
    client.http.post = AsyncMock(side_effect=httpx.ConnectError("boom"))

    with pytest.raises(DeepSeekError) as excinfo:
        await client._post("/api/x")
    assert excinfo.value.biz_code == -1


async def test_post_status_error():
    client = make_client()
    resp = make_resp(status=401, text="denied")
    resp.raise_for_status = MagicMock(side_effect=httpx.HTTPStatusError("401", request=MagicMock(), response=resp))
    client.http.post = AsyncMock(return_value=resp)

    with pytest.raises(DeepSeekError) as excinfo:
        await client._post("/api/x")
    assert excinfo.value.biz_code == 401


async def test_code_raises():
    with pytest.raises(DeepSeekError) as excinfo:
        DeepSeekClient._biz({"code": 123, "msg": "oops"})
    assert excinfo.value.biz_code == 123


async def test_biz_code_raises():
    with pytest.raises(DeepSeekError) as excinfo:
        DeepSeekClient._biz({"code": 0, "data": {"biz_code": 40002, "biz_msg": "nope"}})
    assert excinfo.value.biz_code == 40002


async def test_success_returns_biz_data():
    assert DeepSeekClient._biz({"code": 0, "data": {"biz_data": {"k": 1}}}) == {"k": 1}


async def test_empty_returns_empty():
    assert DeepSeekClient._biz({"code": 0}) == {}


async def test_check_auth_ok():
    client = make_client()
    resp = make_resp({"code": 0})
    client.http.get = AsyncMock(return_value=resp)

    assert await client.check_auth()


async def test_check_auth_bad_code():
    client = make_client()
    resp = make_resp({"code": 500})
    client.http.get = AsyncMock(return_value=resp)

    assert not await client.check_auth()


async def test_check_auth_exception():
    client = make_client()
    client.http.get = AsyncMock(side_effect=httpx.ConnectError("boom"))

    assert not await client.check_auth()


async def test_login_sets_token():
    client = make_client()
    client.http.headers = {}
    resp = make_resp({"code": 0, "data": {"biz_data": {"user": {"token": "fresh"}}}})
    client.http.post = AsyncMock(return_value=resp)

    token = await client.login(email="a@b.c", password="pw")
    assert token == "fresh"
    assert client.token == "fresh"
    assert client.http.headers["Authorization"] == "Bearer fresh"


async def test_login_no_token_raises():
    client = make_client()
    resp = make_resp({"code": 0, "data": {"biz_data": {"user": {}}}})
    client.http.post = AsyncMock(return_value=resp)

    with pytest.raises(DeepSeekError):
        await client.login(email="a@b.c", password="pw")


async def test_get_user():
    client = make_client()
    resp = make_resp({"code": 0, "data": {"biz_data": {"user": {"id": 7}}}})
    client.http.post = AsyncMock(return_value=resp)

    result = await client.get_user()
    assert result == {"user": {"id": 7}}


async def test_create_pow_challenge_ok():
    client = make_client()
    challenge = {"challenge": "x", "salt": "s", "difficulty": 5}
    resp = make_resp({"code": 0, "data": {"biz_data": {"challenge": challenge}}})
    client.http.post = AsyncMock(return_value=resp)

    result = await client.create_pow_challenge("/api/v0/chat/completion")
    assert result == challenge
    client.http.post.assert_awaited_once_with(
        "/api/v0/chat/create_pow_challenge",
        json={"target_path": "/api/v0/chat/completion"},
    )


async def test_create_pow_challenge_missing():
    client = make_client()
    resp = make_resp({"code": 0, "data": {"biz_data": {}}})
    client.http.post = AsyncMock(return_value=resp)

    with pytest.raises(DeepSeekError) as excinfo:
        await client.create_pow_challenge()
    assert excinfo.value.biz_code == -1


async def test_create_session():
    client = make_client()
    resp = make_resp({"code": 0, "data": {"biz_data": {"chat_session": {"id": "cs1", "title": "T"}}}})
    client.http.post = AsyncMock(return_value=resp)

    session = await client.create_session()
    assert session.id == "cs1"
    assert session.title == "T"


async def test_fetch_page():
    client = make_client()
    resp = make_resp({"code": 0, "data": {"biz_data": {"chat_sessions": [{"id": "a"}, {"id": "b"}]}}})
    client.http.post = AsyncMock(return_value=resp)

    result = await client.fetch_page(pinned=True, count=5)
    assert len(result) == 2
    client.http.post.assert_awaited_once_with("/api/v0/chat_session/fetch_page", json={"pinned": True, "count": 5, "mode": "lte"})


async def test_rename_session():
    client = make_client()
    resp = make_resp({"code": 0})
    client.http.post = AsyncMock(return_value=resp)

    await client.rename_session("cs1", "New title")
    client.http.post.assert_awaited_once_with(
        "/api/v0/chat_session/update_title",
        json={"chat_session_id": "cs1", "title": "New title"},
    )


async def test_delete_session():
    client = make_client()
    resp = make_resp({"code": 0})
    client.http.post = AsyncMock(return_value=resp)

    await client.delete_session("cs1")
    client.http.post.assert_awaited_once_with("/api/v0/chat_session/delete", json={"chat_session_id": "cs1"})


async def test_upload_ok():
    client = make_client()
    resp = make_resp({"code": 0, "data": {"biz_data": {"id": "file-1"}}})
    client.http.post = AsyncMock(return_value=resp)

    info = await client.upload_file(b"data", "a.txt", "text/plain", "default", thinking_enabled=True, pow_headers={"X-DS-PoW-Response": "x"})
    assert info == {"id": "file-1"}
    _, kwargs = client.http.post.call_args
    assert kwargs["headers"]["X-File-Size"] == "4"
    assert kwargs["headers"]["X-Model-Type"] == "default"
    assert kwargs["headers"]["X-Thinking-Enabled"] == "1"
    assert kwargs["headers"]["X-DS-PoW-Response"] == "x"


async def test_upload_status_error():
    client = make_client()
    resp = make_resp(status=502, text="bad")
    resp.raise_for_status = MagicMock(side_effect=httpx.HTTPStatusError("502", request=MagicMock(), response=resp))
    client.http.post = AsyncMock(return_value=resp)

    with pytest.raises(DeepSeekError) as excinfo:
        await client.upload_file(b"d", "a", "text/plain", "default")
    assert excinfo.value.biz_code == 502


async def test_upload_http_error():
    client = make_client()
    client.http.post = AsyncMock(side_effect=httpx.ConnectError("boom"))

    with pytest.raises(DeepSeekError):
        await client.upload_file(b"d", "a", "text/plain", "default")


async def test_upload_invalid_json():
    client = make_client()
    resp = make_resp(status=200)
    resp.json = MagicMock(side_effect=ValueError("bad"))
    client.http.post = AsyncMock(return_value=resp)

    with pytest.raises(DeepSeekError):
        await client.upload_file(b"d", "a", "text/plain", "default")


async def test_upload_no_file_id():
    client = make_client()
    resp = make_resp({"code": 0, "data": {"biz_data": {}}})
    client.http.post = AsyncMock(return_value=resp)

    with pytest.raises(DeepSeekError) as excinfo:
        await client.upload_file(b"d", "a", "text/plain", "default")
    assert "no file id" in str(excinfo.value)


async def test_fetch_files_empty():
    client = make_client()

    result = await client.fetch_files([])
    assert result == []


async def test_fetch_files_ok():
    client = make_client()
    resp = make_resp({"code": 0, "data": {"biz_data": {"files": [{"id": "f1"}]}}})
    client.http.get = AsyncMock(return_value=resp)

    result = await client.fetch_files(["f1", "f2"])
    assert result == [{"id": "f1"}]
    client.http.get.assert_awaited_once_with("/api/v0/file/fetch_files", params={"file_ids": "f1,f2"})


async def test_fetch_files_status_error():
    client = make_client()
    resp = make_resp(status=500, text="x")
    resp.raise_for_status = MagicMock(side_effect=httpx.HTTPStatusError("500", request=MagicMock(), response=resp))
    client.http.get = AsyncMock(return_value=resp)

    with pytest.raises(DeepSeekError) as excinfo:
        await client.fetch_files(["f1"])
    assert excinfo.value.biz_code == 500


async def test_fetch_files_http_error():
    client = make_client()
    client.http.get = AsyncMock(side_effect=httpx.ConnectError("boom"))

    with pytest.raises(DeepSeekError):
        await client.fetch_files(["f1"])


async def test_fetch_files_invalid_json():
    client = make_client()
    resp = SimpleNamespace(status_code=200)
    resp.raise_for_status = MagicMock()
    resp.json = MagicMock(side_effect=ValueError("bad"))
    client.http.get = AsyncMock(return_value=resp)

    with pytest.raises(DeepSeekError):
        await client.fetch_files(["f1"])


async def test_history_ok():
    client = make_client()
    resp = make_resp({"code": 0, "data": {"biz_data": {"chat_messages": [{"id": "m1"}]}}})
    client.http.get = AsyncMock(return_value=resp)

    result = await client.history_messages("cs1")
    assert result == [{"id": "m1"}]


async def test_history_status_error():
    client = make_client()
    resp = make_resp(status=500, text="x")
    resp.raise_for_status = MagicMock(side_effect=httpx.HTTPStatusError("500", request=MagicMock(), response=resp))
    client.http.get = AsyncMock(return_value=resp)

    with pytest.raises(DeepSeekError) as excinfo:
        await client.history_messages("cs1")
    assert excinfo.value.biz_code == 500


async def test_history_http_error():
    client = make_client()
    client.http.get = AsyncMock(side_effect=httpx.ConnectError("boom"))

    with pytest.raises(DeepSeekError):
        await client.history_messages("cs1")


async def test_history_invalid_json():
    client = make_client()
    resp = SimpleNamespace(status_code=200)
    resp.raise_for_status = MagicMock()
    resp.json = MagicMock(side_effect=ValueError("bad"))
    client.http.get = AsyncMock(return_value=resp)

    with pytest.raises(DeepSeekError):
        await client.history_messages("cs1")


async def test_completion_builds_request():
    client = make_client()
    sent = SimpleNamespace(status_code=200)
    client.http.build_request = MagicMock(return_value=object())
    client.http.send = AsyncMock(return_value=sent)

    result = await client.completion(
        chat_session_id="cs1",
        prompt="hi",
        parent_message_id="p1",
        model_type="default",
        thinking_enabled=True,
        search_enabled=True,
        ref_file_ids=["f1"],
        pow_headers={"X-DS-PoW-Response": "hdr"},
    )
    assert result is sent
    _, kwargs = client.http.build_request.call_args
    assert kwargs["json"]["chat_session_id"] == "cs1"
    assert kwargs["json"]["prompt"] == "hi"
    assert kwargs["json"]["parent_message_id"] == "p1"
    assert kwargs["json"]["thinking_enabled"] is True
    assert kwargs["json"]["search_enabled"] is True
    assert kwargs["json"]["ref_file_ids"] == ["f1"]
    assert kwargs["headers"]["X-DS-PoW-Response"] == "hdr"
    assert kwargs["headers"]["Accept"] == "text/event-stream"
    client.http.send.assert_awaited_once()
    assert client.http.send.call_args.kwargs["stream"]


async def test_completion_no_ref_files():
    client = make_client()
    client.http.build_request = MagicMock(return_value=object())
    client.http.send = AsyncMock(return_value=SimpleNamespace())
    client.http.send.return_value.status_code = 200

    await client.completion(chat_session_id="cs1", prompt="hi", parent_message_id=None)
    _, kwargs = client.http.build_request.call_args
    assert kwargs["json"]["ref_file_ids"] == []


async def test_stop_stream():
    client = make_client()
    resp = make_resp({"code": 0})
    client.http.post = AsyncMock(return_value=resp)

    await client.stop_stream("cs1", "m1")
    client.http.post.assert_awaited_once_with("/api/v0/chat/stop_stream", json={"chat_session_id": "cs1", "message_id": "m1"})
