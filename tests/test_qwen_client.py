import asyncio
import hashlib
import unittest
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from danyapi.qwen.client import (
    QwenClient,
    QwenError,
    QwenSession,
    new_uuid,
    sha256_hex,
    timezone_header,
)


class TestQwenHelpers(unittest.TestCase):
    def test_new_uuid(self):
        value = new_uuid()
        self.assertEqual(str(uuid.UUID(value)), value)

    def test_sha256_hex(self):
        self.assertEqual(sha256_hex("hello"), hashlib.sha256(b"hello").hexdigest())

    def test_timezone_header(self):
        value = timezone_header()
        self.assertIn("GMT", value)
        self.assertRegex(value, r"GMT[+-]\d{4}$")


class TestQwenSession(unittest.TestCase):
    def test_defaults(self):
        s = QwenSession(id="c1")
        self.assertEqual(s.title, "")
        self.assertIsNone(s.last_response_id)
        self.assertIsNone(s.model)
        self.assertEqual(s.accumulated_input_tokens, 0)
        self.assertEqual(s.accumulated_output_tokens, 0)
        self.assertEqual(s.extra, {})

    def test_full(self):
        s = QwenSession(id="c1", title="t", last_response_id="r1", model="m", accumulated_input_tokens=5, accumulated_output_tokens=7)
        self.assertEqual(s.accumulated_input_tokens, 5)
        self.assertEqual(s.accumulated_output_tokens, 7)


class TestQwenError(unittest.TestCase):
    def test_str_and_attrs(self):
        err = QwenError(401, "bad token")
        self.assertEqual(err.code, 401)
        self.assertEqual(err.message, "bad token")
        self.assertIn("401", str(err))


class TestQwenBiz(unittest.TestCase):
    def test_success_returns_data(self):
        self.assertEqual(QwenClient._biz({"success": True, "data": {"a": 1}}), {"a": 1})

    def test_success_without_data_returns_empty(self):
        self.assertEqual(QwenClient._biz({"success": True}), {})

    def test_failure_with_data_code(self):
        with self.assertRaises(QwenError) as ctx:
            QwenClient._biz({"success": False, "data": {"code": 123, "details": "nope"}})
        self.assertEqual(ctx.exception.code, 123)
        self.assertEqual(ctx.exception.message, "nope")

    def test_failure_generic(self):
        with self.assertRaises(QwenError) as ctx:
            QwenClient._biz({"success": False, "code": 500, "details": "oops"})
        self.assertEqual(ctx.exception.code, 500)
        self.assertEqual(ctx.exception.message, "oops")

    def test_failure_no_code_defaults(self):
        with self.assertRaises(QwenError) as ctx:
            QwenClient._biz({"success": False})
        self.assertEqual(ctx.exception.code, -1)


class TestQwenParseJson(unittest.TestCase):
    def _client(self):
        return QwenClient()

    def test_non_json_content_type(self):
        resp = SimpleNamespace(headers={"content-type": "text/html"}, text="<html></html>")
        with self.assertRaises(QwenError) as ctx:
            self._client()._parse_json(resp, "/x")
        self.assertEqual(ctx.exception.code, -1)

    def test_invalid_json(self):
        resp = SimpleNamespace(headers={"content-type": "application/json"}, text="not json")
        resp.json = MagicMock(side_effect=ValueError("boom"))
        with self.assertRaises(QwenError):
            self._client()._parse_json(resp, "/x")

    def test_valid_json(self):
        resp = SimpleNamespace(headers={"content-type": "application/json"})
        resp.json = MagicMock(return_value={"success": True, "data": {"k": 1}})
        self.assertEqual(self._client()._parse_json(resp, "/x"), {"k": 1})


class TestQwenRequestHeaders(unittest.TestCase):
    def test_request_headers(self):
        headers = QwenClient._request_headers()
        self.assertIn("X-Request-Id", headers)
        self.assertIn("Timezone", headers)

    def test_request_headers_merge(self):
        headers = QwenClient._request_headers({"Accept": "text/event-stream"})
        self.assertEqual(headers["Accept"], "text/event-stream")
        self.assertIn("X-Request-Id", headers)


class TestQwenClientConstruction(unittest.TestCase):
    def test_headers_and_cookie_with_token(self):
        client = QwenClient(token="tok")
        self.assertEqual(client.token, "tok")
        self.assertEqual(client.http.headers["Authorization"], "Bearer tok")

        async def run():
            await client.aclose()

        asyncio.run(run())

    def test_no_token(self):
        client = QwenClient()
        self.assertIsNone(client.token)
        self.assertNotIn("Authorization", client.http.headers)

        async def run():
            await client.aclose()

        asyncio.run(run())


class TestQwenClientEndpoints(unittest.TestCase):
    def test_check_auth_ok(self):
        client = QwenClient()
        resp = SimpleNamespace(status_code=200)
        resp.json = MagicMock(return_value={"success": True})
        client.http.get = AsyncMock(return_value=resp)

        async def run():
            self.assertTrue(await client.check_auth())

        asyncio.run(run())

    def test_check_auth_non_200(self):
        client = QwenClient()
        resp = SimpleNamespace(status_code=403)
        resp.json = MagicMock(return_value={"success": False})
        client.http.get = AsyncMock(return_value=resp)

        async def run():
            self.assertFalse(await client.check_auth())

        asyncio.run(run())

    def test_check_auth_exception(self):
        import httpx

        client = QwenClient()
        client.http.get = AsyncMock(side_effect=httpx.ConnectError("boom"))

        async def run():
            self.assertFalse(await client.check_auth())

        asyncio.run(run())

    def test_login(self):
        client = QwenClient()
        resp = SimpleNamespace(status_code=200, headers={"content-type": "application/json"})
        resp.json = MagicMock(return_value={"success": True, "data": {"token": "fresh"}})
        client.http.post = AsyncMock(return_value=resp)

        async def run():
            token = await client.login("a@b.c", "pw")
            self.assertEqual(token, "fresh")
            self.assertEqual(client.http.headers["Authorization"], "Bearer fresh")

        asyncio.run(run())

    def test_login_no_token_raises(self):
        client = QwenClient()
        resp = SimpleNamespace(status_code=200, headers={"content-type": "application/json"})
        resp.json = MagicMock(return_value={"success": True, "data": {}})
        client.http.post = AsyncMock(return_value=resp)

        async def run():
            with self.assertRaises(QwenError):
                await client.login("a@b.c", "pw")

        asyncio.run(run())

    def test_fetch_models(self):
        client = QwenClient()
        resp = SimpleNamespace(status_code=200, headers={"content-type": "application/json"})
        resp.json = MagicMock(return_value={"success": True, "data": {"data": [{"id": "m1"}]}})
        client.http.get = AsyncMock(return_value=resp)

        async def run():
            models = await client.fetch_models()
            self.assertEqual(models, [{"id": "m1"}])

        asyncio.run(run())

    def test_create_chat(self):
        client = QwenClient()
        resp = SimpleNamespace(status_code=200, headers={"content-type": "application/json"})
        resp.json = MagicMock(return_value={"success": True, "data": {"id": "chat1"}})
        client.http.post = AsyncMock(return_value=resp)

        async def run():
            self.assertEqual(await client.create_chat("qwen3.8-max"), "chat1")

        asyncio.run(run())

    def test_create_chat_no_id(self):
        client = QwenClient()
        resp = SimpleNamespace(status_code=200, headers={"content-type": "application/json"})
        resp.json = MagicMock(return_value={"success": True, "data": {}})
        client.http.post = AsyncMock(return_value=resp)

        async def run():
            with self.assertRaises(QwenError):
                await client.create_chat("qwen3.8-max")

        asyncio.run(run())

    def test_completion_builds_request(self):
        client = QwenClient()
        sent = SimpleNamespace(status_code=200)
        client.http.build_request = MagicMock(return_value=object())
        client.http.send = AsyncMock(return_value=sent)

        async def run():
            result = await client.completion("chat1", "hi", "p1", "m", thinking=True, search=True)
            self.assertIs(result, sent)
            _, kwargs = client.http.build_request.call_args
            self.assertEqual(kwargs["json"]["chatId"], "chat1")
            self.assertEqual(kwargs["json"]["model"], "m")

        asyncio.run(run())

    def test_stop_stream(self):
        client = QwenClient()
        resp = SimpleNamespace(status_code=200, headers={"content-type": "application/json"})
        resp.json = MagicMock(return_value={"success": True, "data": {}})
        client.http.post = AsyncMock(return_value=resp)

        async def run():
            await client.stop_stream("chat1", "r1")
            client.http.post.assert_awaited_once()

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
