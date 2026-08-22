import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from danyapi.qwen.client import (
    QwenClient,
    QwenError,
    QwenSession,
    new_uuid,
    timezone_header,
)


def test_new_uuid():
    value = new_uuid()
    assert str(uuid.UUID(value)) == value


def test_timezone_header():
    value = timezone_header()
    assert "GMT" in value
    assert __import__("re").search(r"GMT[+-]\d{4}$", value)


def test_session_defaults():
    s = QwenSession(id="c1")
    assert s.title == ""
    assert s.last_response_id is None
    assert s.model is None
    assert s.accumulated_input_tokens == 0
    assert s.accumulated_output_tokens == 0
    assert s.extra == {}


def test_session_full():
    s = QwenSession(id="c1", title="t", last_response_id="r1", model="m", accumulated_input_tokens=5, accumulated_output_tokens=7)
    assert s.accumulated_input_tokens == 5
    assert s.accumulated_output_tokens == 7


def test_error_str_and_attrs():
    err = QwenError(401, "bad token")
    assert err.code == 401
    assert err.message == "bad token"
    assert "401" in str(err)


def test_biz_success_returns_data():
    assert QwenClient._biz({"success": True, "data": {"a": 1}}) == {"a": 1}


def test_biz_success_without_data():
    assert QwenClient._biz({"success": True}) == {}


def test_biz_failure_with_data_code():
    with pytest.raises(QwenError) as excinfo:
        QwenClient._biz({"success": False, "data": {"code": 123, "details": "nope"}})
    assert excinfo.value.code == 123
    assert excinfo.value.message == "nope"


def test_biz_failure_generic():
    with pytest.raises(QwenError) as excinfo:
        QwenClient._biz({"success": False, "code": 500, "details": "oops"})
    assert excinfo.value.code == 500
    assert excinfo.value.message == "oops"


def test_biz_failure_no_code_defaults():
    with pytest.raises(QwenError) as excinfo:
        QwenClient._biz({"success": False})
    assert excinfo.value.code == -1


def test_parse_json_non_json_content_type():
    resp = SimpleNamespace(headers={"content-type": "text/html"}, text="<html></html>")
    with pytest.raises(QwenError) as excinfo:
        QwenClient()._parse_json(resp, "/x")
    assert excinfo.value.code == -1


def test_parse_json_invalid_json():
    resp = SimpleNamespace(headers={"content-type": "application/json"}, text="not json")
    resp.json = MagicMock(side_effect=ValueError("boom"))
    with pytest.raises(QwenError):
        QwenClient()._parse_json(resp, "/x")


def test_parse_json_valid_json():
    resp = SimpleNamespace(headers={"content-type": "application/json"})
    resp.json = MagicMock(return_value={"success": True, "data": {"k": 1}})
    assert QwenClient()._parse_json(resp, "/x") == {"k": 1}


def test_request_headers():
    headers = QwenClient._request_headers()
    assert "X-Request-Id" in headers
    assert "Timezone" in headers


def test_request_headers_merge():
    headers = QwenClient._request_headers({"Accept": "text/event-stream"})
    assert headers["Accept"] == "text/event-stream"
    assert "X-Request-Id" in headers


async def test_headers_and_cookie_with_token():
    client = QwenClient(token="tok")
    assert client.token == "tok"
    assert client.http.headers["Authorization"] == "Bearer tok"
    await client.aclose()


async def test_no_token():
    client = QwenClient()
    assert client.token is None
    assert "Authorization" not in client.http.headers
    await client.aclose()


async def test_check_auth_ok():
    client = QwenClient()
    resp = SimpleNamespace(status_code=200)
    resp.json = MagicMock(return_value={"success": True})
    client.http.get = AsyncMock(return_value=resp)
    assert await client.check_auth()


async def test_check_auth_ok_user_object():
    client = QwenClient()
    resp = SimpleNamespace(status_code=200)
    resp.json = MagicMock(return_value={"id": "u1", "email": "a@b.c", "name": "test"})
    client.http.get = AsyncMock(return_value=resp)
    assert await client.check_auth()


async def test_check_auth_no_id_no_success():
    client = QwenClient()
    resp = SimpleNamespace(status_code=200)
    resp.json = MagicMock(return_value={"foo": "bar"})
    client.http.get = AsyncMock(return_value=resp)
    assert not await client.check_auth()


async def test_check_auth_non_200():
    client = QwenClient()
    resp = SimpleNamespace(status_code=403)
    resp.json = MagicMock(return_value={"success": False})
    client.http.get = AsyncMock(return_value=resp)
    assert not await client.check_auth()


async def test_check_auth_exception():
    import httpx

    client = QwenClient()
    client.http.get = AsyncMock(side_effect=httpx.ConnectError("boom"))
    assert not await client.check_auth()


async def test_fetch_models():
    client = QwenClient()
    resp = SimpleNamespace(status_code=200, headers={"content-type": "application/json"})
    resp.json = MagicMock(return_value={"success": True, "data": {"data": [{"id": "m1"}]}})
    client.http.get = AsyncMock(return_value=resp)
    models = await client.fetch_models()
    assert models == [{"id": "m1"}]


async def test_create_chat():
    client = QwenClient()
    resp = SimpleNamespace(status_code=200, headers={"content-type": "application/json"})
    resp.json = MagicMock(return_value={"success": True, "data": {"id": "chat1"}})
    client.http.post = AsyncMock(return_value=resp)
    assert await client.create_chat("qwen3.8-max") == "chat1"


async def test_create_chat_no_id():
    client = QwenClient()
    resp = SimpleNamespace(status_code=200, headers={"content-type": "application/json"})
    resp.json = MagicMock(return_value={"success": True, "data": {}})
    client.http.post = AsyncMock(return_value=resp)
    with pytest.raises(QwenError):
        await client.create_chat("qwen3.8-max")


async def test_completion_builds_request():
    client = QwenClient()
    sent = SimpleNamespace(status_code=200)
    client.http.build_request = MagicMock(return_value=object())
    client.http.send = AsyncMock(return_value=sent)
    result = await client.completion("chat1", "hi", "p1", "m", thinking=True, search=True)
    assert result is sent
    _, kwargs = client.http.build_request.call_args
    assert kwargs["json"]["chatId"] == "chat1"
    assert kwargs["json"]["model"] == "m"


async def test_stop_stream():
    client = QwenClient()
    resp = SimpleNamespace(status_code=200, headers={"content-type": "application/json"})
    resp.json = MagicMock(return_value={"success": True, "data": {}})
    client.http.post = AsyncMock(return_value=resp)
    await client.stop_stream("chat1", "r1")
    client.http.post.assert_awaited_once()
