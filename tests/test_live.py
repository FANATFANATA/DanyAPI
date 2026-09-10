import os

import pytest

from danyapi.deepseek.client import DeepSeekClient
from danyapi.deepseek.stream import IncrementalSSE, MessageReconstructor
from danyapi.pow import PowManager

_TOKEN = os.environ.get("DEEPSEEK_TOKENS", "").split(",")[0].strip()

pytestmark = [pytest.mark.live, pytest.mark.skipif(not _TOKEN, reason="DEEPSEEK_TOKENS not set")]


async def _make_account():
    client = DeepSeekClient(token=_TOKEN, timeout=120)
    assert await client.check_auth(), "auth failed"
    pow_mgr = PowManager()
    session = await client.create_session()
    return client, pow_mgr, session


async def _complete(client, pow_mgr, session, prompt, *, model_type="default", thinking=False, search=False, ref_file_ids=None, parent_message_id=None):
    pow_headers = await pow_mgr.make_header(client.create_pow_challenge)
    resp = await client.completion(
        chat_session_id=session.id,
        prompt=prompt,
        parent_message_id=parent_message_id,
        model_type=model_type,
        thinking_enabled=thinking,
        search_enabled=search,
        ref_file_ids=ref_file_ids,
        pow_headers=pow_headers,
    )
    rec = MessageReconstructor()
    incremental = IncrementalSSE()
    response_message_id = None
    try:
        async for chunk in resp.aiter_bytes():
            for event in incremental.feed(chunk):
                if event.event == "ready" and isinstance(event.data, dict):
                    response_message_id = event.data.get("response_message_id")
                rec.handle(event)
        for event in incremental.finish():
            rec.handle(event)
    finally:
        try:
            await resp.aclose()
        except Exception:
            pass
    return rec, response_message_id


async def test_auth():
    client = DeepSeekClient(token=_TOKEN, timeout=30)
    assert await client.check_auth()
    await client.aclose()


async def test_create_session():
    client, _, session = await _make_account()
    assert session.id
    await client.aclose()


async def test_completion_basic():
    client, pow_mgr, session = await _make_account()
    try:
        rec, _response_message_id = await _complete(client, pow_mgr, session, "Reply with exactly: OK", model_type="default")
        assert rec.content, "empty content"
        assert rec.status, "no status"
    finally:
        await client.aclose()


async def test_completion_thinking():
    client, pow_mgr, session = await _make_account()
    try:
        rec, _ = await _complete(client, pow_mgr, session, "What is 2+2? Reply with just the number.", model_type="default", thinking=True)
        assert rec.content, "empty content"
    finally:
        await client.aclose()


async def test_completion_search():
    client, pow_mgr, session = await _make_account()
    try:
        rec, _ = await _complete(client, pow_mgr, session, "What is the current year? Reply briefly.", model_type="default", thinking=False, search=True)
        assert rec.content, "empty content"
    finally:
        await client.aclose()


async def test_completion_thinking_and_search():
    client, pow_mgr, session = await _make_account()
    try:
        rec, _ = await _complete(client, pow_mgr, session, "What is the capital of France? Reply briefly.", model_type="default", thinking=True, search=True)
        assert rec.content, "empty content"
    finally:
        await client.aclose()


async def test_upload_file():
    client, pow_mgr, session = await _make_account()
    try:
        pow_upload = PowManager()
        pow_headers = await pow_upload.make_header(lambda: client.create_pow_challenge("/api/v0/file/upload_file"))
        file_data = b"Hello, this is a test document."
        info = await client.upload_file(file_data, "test.txt", "text/plain", "default", thinking_enabled=False, pow_headers=pow_headers)
        assert info.get("id"), "no file id returned"
        file_id = info["id"]

        rec, _ = await _complete(client, pow_mgr, session, "What does this document say? Reply briefly.", model_type="default", ref_file_ids=[file_id])
        assert rec.content, "empty content"
    finally:
        await client.aclose()


async def test_conversation_turns():
    client, pow_mgr, session = await _make_account()
    try:
        rec1, _msg_id = await _complete(client, pow_mgr, session, "My name is TestBot. Reply with: OK", model_type="default")
        assert rec1.content, "empty content on first turn"

        rec2, _ = await _complete(client, pow_mgr, session, "What is my name?", model_type="default", parent_message_id=rec1.id)
        assert rec2.content, "empty content on second turn"
    finally:
        await client.aclose()
