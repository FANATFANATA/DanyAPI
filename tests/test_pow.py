import asyncio
import base64
import json
import shutil
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from danyapi.pow import (
    _find_native_solver,
    _run_solver,
    deepseek_hash_v1,
    deepseek_hash_v1_hex,
    solve_challenge,
    solve_native,
    solve_node,
    solve_python,
)


def test_missing_fields_raise():
    from danyapi.pow import PowManager

    async def run():
        pm = PowManager()

        async def fetch_missing():
            return {"challenge": "x", "algorithm": "a", "signature": "s", "target_path": "t"}

        with pytest.raises(RuntimeError):
            await pm.make_header(fetch_missing)

    asyncio.run(run())


def test_invalid_expire_at_raises():
    from danyapi.pow import PowManager

    async def run():
        pm = PowManager()

        async def fetch_bad():
            return {
                "challenge": "x",
                "salt": "s",
                "algorithm": "a",
                "signature": "s",
                "target_path": "t",
                "expire_at": None,
                "difficulty": 5,
            }

        with pytest.raises(RuntimeError):
            await pm.make_header(fetch_bad)

    asyncio.run(run())


def test_invalid_difficulty_raises():
    from danyapi.pow import PowManager

    async def run():
        pm = PowManager()

        async def fetch_bad():
            challenge = _valid_challenge()
            challenge["difficulty"] = 0
            return challenge

        with pytest.raises(RuntimeError):
            await pm.make_header(fetch_bad)

    asyncio.run(run())


def test_solver_failure_raises():
    from danyapi.pow import PowManager

    async def run():
        pm = PowManager()

        async def fetch_ok():
            return _valid_challenge()

        with patch("danyapi.pow.solve_challenge", new=AsyncMock(return_value=None)):
            with pytest.raises(RuntimeError):
                await pm.make_header(fetch_ok)

    asyncio.run(run())


def test_make_header_payload():
    from danyapi.pow import PowManager

    async def run():
        pm = PowManager()

        async def fetch_ok():
            return _valid_challenge()

        with patch("danyapi.pow.solve_challenge", new=AsyncMock(return_value=42)):
            header = await pm.make_header(fetch_ok)
        payload = _decode_header(header)
        assert payload["answer"] == 42
        assert payload["algorithm"] == "alg"
        assert payload["signature"] == "sig"
        assert payload["target_path"] == "/api/v0/chat/completion"

    asyncio.run(run())


def test_make_header_prefetches():
    from danyapi.pow import PowManager

    async def run():
        pm = PowManager()
        fetch = AsyncMock(return_value=_valid_challenge())

        with patch("danyapi.pow.solve_challenge", new=AsyncMock(return_value=42)):
            h1 = await pm.make_header(fetch)
            assert h1 is not None
            assert pm._refill is not None
            if pm._refill is not None:
                await pm._refill
            h2 = await pm.make_header(fetch)
            assert h2 is not None
            assert fetch.await_count == 2

    asyncio.run(run())


def test_refill_failure_logged():
    from danyapi.pow import PowManager

    async def run():
        pm = PowManager()

        async def fetch_fail():
            raise RuntimeError("no challenge")

        pm._kick_refill(fetch_fail)
        await pm._refill
        assert pm._refill is None
        assert pm._header is None

    asyncio.run(run())


def test_refill_skips_when_header_present():
    from danyapi.pow import PowManager

    async def run():
        pm = PowManager()
        pm._header = {"X-DS-PoW-Response": "x"}
        fetch = AsyncMock(side_effect=RuntimeError("must not be called"))
        with patch("danyapi.pow.solve_challenge", new=AsyncMock(return_value=42)):
            await pm._refill_if_empty(fetch)
        assert pm._header == {"X-DS-PoW-Response": "x"}
        fetch.assert_not_called()

    asyncio.run(run())


def test_kick_refill_skips_running_task():
    from danyapi.pow import PowManager

    async def run():
        pm = PowManager()
        pm._refill = asyncio.create_task(asyncio.sleep(1))
        fetch = AsyncMock()
        pm._kick_refill(fetch)
        task = pm._refill
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        fetch.assert_not_called()

    asyncio.run(run())


def test_find_native_solver_missing():
    with patch("danyapi.pow._SOLVER_DIR", Path(tempfile.mkdtemp())):
        assert _find_native_solver() is None


def _challenge(counter: int) -> tuple[str, str, int]:
    salt = "chk"
    expire_at = 1700000000000
    prefix = f"{salt}_{expire_at}_".encode()
    target = deepseek_hash_v1_hex(prefix + str(counter).encode())
    return target, salt, expire_at


def test_vectors():
    vectors = {
        b"": "e594808bc5b7151ac160c6d39a02e0a8e261ed588578403099e3561dc40c26b3",
        b"A": "7157d45adfe495122cb4a12198a2d603b2e09019177c5efe5d0a12b00247e407",
        b"hello": "50605e468e6d6ead913d7d7ccc4687b83ded157cf0a0c5e011eefece12712fa5",
        b"DeepSeekHashV1": "3fc52c4ae40faa946b1bc0eeb747059a35fba6efaa3d616074e720d6e99436cd",
    }
    for data, expected in vectors.items():
        assert deepseek_hash_v1_hex(data) == expected


def test_padlen_zero_case():
    data = b"x" * 135
    digest = deepseek_hash_v1_hex(data)
    assert len(digest) == 64
    assert digest == deepseek_hash_v1_hex(data)


def test_output_longer_than_rate():
    assert len(deepseek_hash_v1(b"data", output_bytes=200)) == 200


def test_hash_byte_output():
    assert deepseek_hash_v1(b"x", output_bytes=16) == deepseek_hash_v1(b"x")[:16]


def test_solve_python_finds_answer():
    salt = "chk"
    expire_at = 1700000000000
    prefix = f"{salt}_{expire_at}_".encode()
    target = deepseek_hash_v1_hex(prefix + b"7")
    assert solve_python(target, salt, expire_at, 1000) == 7


def test_solve_python_not_found():
    salt = "chk"
    expire_at = 1700000000000
    prefix = f"{salt}_{expire_at}_".encode()
    target = deepseek_hash_v1_hex(prefix + b"1000")
    assert solve_python(target, salt, expire_at, 500) is None


def test_solve_python_zero_limit():
    assert solve_python("0" * 64, "s", 1, 0) is None


def _proc(stdout="", returncode=0, stderr=""):
    return MagicMock(stdout=stdout, returncode=returncode, stderr=stderr)


def test_run_solver_ok():
    with patch("danyapi.pow.subprocess.run", return_value=_proc('{"answer":5}')):
        assert _run_solver(Path("x"), "c", "s", 1, 10) == 5


def test_run_solver_nonzero_raises():
    with patch("danyapi.pow.subprocess.run", return_value=_proc("", returncode=1, stderr="boom")):
        with pytest.raises(RuntimeError):
            _run_solver(Path("x"), "c", "s", 1, 10)


def test_run_solver_error_payload_raises():
    with patch("danyapi.pow.subprocess.run", return_value=_proc('{"error":"no answer"}')):
        with pytest.raises(RuntimeError):
            _run_solver(Path("x"), "c", "s", 1, 10)


def test_native_matches_python():
    if _find_native_solver() is None:
        pytest.skip("native pow_solver not built")
    target, salt, expire_at = _challenge(42)
    assert solve_native(target, salt, expire_at, 200000) == 42


@pytest.mark.parametrize("slen", [100, 121, 127, 128, 129, 200, 300, 500, 1000])
def test_native_matches_python_multiblock(slen):
    if _find_native_solver() is None:
        pytest.skip("native pow_solver not built")
    salt = "S" * slen
    expire_at = 1700000000000
    prefix = f"{salt}_{expire_at}_".encode()
    target = deepseek_hash_v1_hex(prefix + b"42")
    assert solve_native(target, salt, expire_at, 200000) == 42


def test_node_matches_python():
    if shutil.which("node") is None:
        pytest.skip("node not available")
    target, salt, expire_at = _challenge(42)
    assert solve_node(target, salt, expire_at, 200000) == 42


def test_native_missing_raises():
    with patch("danyapi.pow._find_native_solver", return_value=None):
        with pytest.raises(FileNotFoundError):
            solve_native("c", "s", 1, 10)


def test_node_missing_raises():
    with patch("danyapi.pow.Path.exists", return_value=False):
        with pytest.raises(FileNotFoundError):
            solve_node("c", "s", 1, 10)


def test_solve_challenge_falls_back():
    def boom_native(*args):
        raise RuntimeError("missing")

    with (
        patch("danyapi.pow.solve_native", new=boom_native),
        patch("danyapi.pow.solve_node", new=boom_native),
        patch("danyapi.pow.solve_python", new=lambda *a: 9),
    ):
        assert asyncio.run(solve_challenge("c", "s", 1, 10)) == 9


def test_solve_challenge_all_fail():
    def boom(*args):
        raise RuntimeError("missing")

    with (
        patch("danyapi.pow.solve_native", new=boom),
        patch("danyapi.pow.solve_node", new=boom),
        patch("danyapi.pow.solve_python", new=lambda *a: None),
    ):
        assert asyncio.run(solve_challenge("c", "s", 1, 10)) is None


def _valid_challenge():
    return {
        "challenge": "00" * 32,
        "salt": "salt",
        "algorithm": "alg",
        "signature": "sig",
        "target_path": "/api/v0/chat/completion",
        "expire_at": 1700000000000,
        "difficulty": 5,
    }


def _decode_header(header):
    raw = base64.b64decode(header["X-DS-PoW-Response"])
    return json.loads(raw)
