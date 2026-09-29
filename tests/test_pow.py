import asyncio
import base64
import gc
import json
import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from danyapi.config import CREDENTIAL_ENV_NAMES
from danyapi.pow import (
    _PYTHON_SOLVE_BUDGET_SEC,
    _SOLVE_TOTAL_BUDGET_SEC,
    _SOLVER_DIFFICULTY_LIMIT,
    _SOLVER_ENV_DENYLIST,
    _SOLVER_TIMEOUT_SEC,
    _find_native_solver,
    _run_solver,
    _solver_env,
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


def test_kick_refill_does_not_overwrite_outstanding_task():
    from danyapi.pow import PowManager

    async def run():
        pm = PowManager()
        pending = asyncio.create_task(asyncio.sleep(5))
        pm._refill = pending
        pm._kick_refill(AsyncMock())
        assert pm._refill is pending
        pm.close()
        assert pm._refill is None
        with pytest.raises(asyncio.CancelledError):
            await pending

    asyncio.run(run())


def test_kick_refill_retrieves_finished_task_result():
    from danyapi.pow import PowManager

    async def run():
        pm = PowManager()
        captured = []
        loop = asyncio.get_running_loop()
        loop.set_exception_handler(lambda _loop, context: captured.append(context.get("message", "")))

        async def failing():
            raise RuntimeError("boom")

        done = asyncio.create_task(failing())
        await asyncio.sleep(0)
        assert done.done()
        pm._refill = done

        async def fetch():
            raise RuntimeError("must not be called")

        pm._kick_refill(fetch)
        assert pm._refill is not done
        await pm._refill
        del done
        gc.collect()
        await asyncio.sleep(0)
        assert captured == []

    asyncio.run(run())


def test_close_without_refill_is_noop():
    from danyapi.pow import PowManager

    pm = PowManager()
    pm.close()
    assert pm._refill is None


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


def test_hash_output_truncated_to_exact_byte_count():
    for n in (1, 3, 5, 7, 33, 135, 137, 200):
        digest = deepseek_hash_v1(b"payload", output_bytes=n)
        assert len(digest) == n
    ref = deepseek_hash_v1(b"payload")
    for n in (1, 3, 5, 7, 16, 31):
        assert deepseek_hash_v1(b"payload", output_bytes=n) == ref[:n]


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


def test_run_solver_accepts_zero_answer():
    with patch("danyapi.pow.subprocess.run", return_value=_proc('{"answer":0}')):
        assert _run_solver(Path("x"), "c", "s", 1, 10) == 0


@pytest.mark.parametrize("payload", ['{"answer":true}', '{"answer":"7"}', '{"answer":-1}', '{"answer":1.5}', "{}"])
def test_run_solver_rejects_invalid_answer(payload):
    with patch("danyapi.pow.subprocess.run", return_value=_proc(payload)):
        with pytest.raises(RuntimeError) as exc:
            _run_solver(Path("x"), "c", "s", 1, 10)
    assert "invalid answer" in str(exc.value)


def test_run_solver_invalid_answer_repr_is_truncated():
    payload = json.dumps({"answer": "x" * 5000})
    with patch("danyapi.pow.subprocess.run", return_value=_proc(payload)):
        with pytest.raises(RuntimeError) as exc:
            _run_solver(Path("x"), "c", "s", 1, 10)
    assert len(str(exc.value)) < 300


def test_solver_env_drops_secrets(monkeypatch):
    for name in CREDENTIAL_ENV_NAMES:
        monkeypatch.setenv(name, "secret-value")
    monkeypatch.setenv("DANYAPI_HOST", "127.0.0.1")
    env = _solver_env()
    for name in CREDENTIAL_ENV_NAMES:
        assert name not in env
    assert env["DANYAPI_HOST"] == "127.0.0.1"
    assert os.environ["DEEPSEEK_TOKENS"] == "secret-value"


def test_solver_env_denylist_is_the_credential_tuple(monkeypatch):
    assert set(CREDENTIAL_ENV_NAMES) <= _SOLVER_ENV_DENYLIST
    for name in ("BYOK", "BYOK_MODE", "DANYAPI_BYOK_MODE", "GIGACHAT_KEYS", "DANYAPI_ADMIN_TOKEN"):
        assert name in _SOLVER_ENV_DENYLIST
    for name in _SOLVER_ENV_DENYLIST:
        monkeypatch.setenv(name, "secret-value")
    assert _solver_env() == {key: value for key, value in os.environ.items() if key not in _SOLVER_ENV_DENYLIST}


def test_run_solver_passes_scrubbed_env():
    with patch("danyapi.pow.subprocess.run", return_value=_proc('{"answer":1}')) as run:
        assert _run_solver(Path("x"), "c", "s", 1, 10) == 1
    env = run.call_args.kwargs["env"]
    assert "DEEPSEEK_TOKENS" not in env


def test_solve_python_respects_budget():
    salt = "S" * 140
    assert solve_python("00" * 32, salt, 1, 100, timeout=0.0) is None


def test_solve_python_budget_expires_mid_search():
    salt = "S" * 140
    ticks = iter([0.0, 0.0] + [1.0] * 200)
    with patch("danyapi.pow.time.monotonic", side_effect=lambda: next(ticks)):
        assert solve_python("00" * 32, salt, 1, 100000, timeout=0.5) is None


def test_solve_python_timeout_is_the_fifth_parameter():
    salt = "S" * 140
    assert solve_python("00" * 32, salt, 1, 100, 0.0) is None
    assert solve_python("00" * 32, salt, 1, 100, timeout=0.0) is None


class _FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now


def _recording_solver(clock: _FakeClock, seen: list, name: str, cost: float):
    def solver(challenge_hex, salt, expire_at, difficulty, budget):
        seen.append((name, clock.now, budget, difficulty))
        clock.now += cost
        return None

    return solver


def test_solve_challenge_gives_each_solver_at_most_the_remaining_time():
    clock = _FakeClock()
    seen: list = []
    with (
        patch("danyapi.pow.time", clock),
        patch("danyapi.pow.solve_native", new=_recording_solver(clock, seen, "native", 40.0)),
        patch("danyapi.pow.solve_node", new=_recording_solver(clock, seen, "node", 40.0)),
        patch("danyapi.pow.solve_python", new=_recording_solver(clock, seen, "python", 0.0)),
    ):
        assert asyncio.run(solve_challenge("c", "s", 1, 10)) is None
    assert [item[0] for item in seen] == ["native", "node", "python"]
    assert seen[0][2] == pytest.approx(_SOLVER_TIMEOUT_SEC)
    assert seen[1][2] == pytest.approx(_SOLVE_TOTAL_BUDGET_SEC - 40.0)
    assert seen[2][2] == pytest.approx(_PYTHON_SOLVE_BUDGET_SEC)
    for _name, started_at, budget, _difficulty in seen:
        assert budget <= max(_SOLVE_TOTAL_BUDGET_SEC - started_at, 0.0)
    assert clock.now <= _SOLVE_TOTAL_BUDGET_SEC


def test_solve_challenge_bounds_the_whole_chain_by_the_total_budget():
    clock = _FakeClock()
    seen: list = []

    def burn_whole_budget(challenge_hex, salt, expire_at, difficulty, budget):
        seen.append(("solver", clock.now, budget, difficulty))
        clock.now += budget
        return None

    def burn_native(*args):
        return burn_whole_budget(*args)

    def burn_node(*args):
        return burn_whole_budget(*args)

    def burn_python(*args):
        return burn_whole_budget(*args)

    with (
        patch("danyapi.pow.time", clock),
        patch("danyapi.pow.solve_native", new=burn_native),
        patch("danyapi.pow.solve_node", new=burn_node),
        patch("danyapi.pow.solve_python", new=burn_python),
    ):
        assert asyncio.run(solve_challenge("c", "s", 1, 10)) is None
    assert [item[2] for item in seen] == [pytest.approx(_SOLVER_TIMEOUT_SEC), pytest.approx(_SOLVE_TOTAL_BUDGET_SEC - _SOLVER_TIMEOUT_SEC)]
    assert clock.now == pytest.approx(_SOLVE_TOTAL_BUDGET_SEC)
    assert clock.now < _SOLVER_TIMEOUT_SEC * 3


def test_solve_challenge_python_gets_at_most_its_own_cap():
    clock = _FakeClock()
    seen: list = []
    with (
        patch("danyapi.pow.time", clock),
        patch("danyapi.pow.solve_native", new=_recording_solver(clock, seen, "native", 0.0)),
        patch("danyapi.pow.solve_node", new=_recording_solver(clock, seen, "node", 0.0)),
        patch("danyapi.pow.solve_python", new=_recording_solver(clock, seen, "python", 0.0)),
    ):
        assert asyncio.run(solve_challenge("c", "s", 1, 10)) is None
    assert seen[0][2] == pytest.approx(_SOLVER_TIMEOUT_SEC)
    assert seen[1][2] == pytest.approx(_SOLVER_TIMEOUT_SEC)
    assert seen[2][2] == pytest.approx(_PYTHON_SOLVE_BUDGET_SEC)
    assert _PYTHON_SOLVE_BUDGET_SEC < _SOLVER_TIMEOUT_SEC


def _solve_with_seen_solvers(difficulty: int) -> list:
    from danyapi.pow import PowManager

    seen: list = []
    clock = _FakeClock()

    async def run():
        pm = PowManager()

        async def fetch_ok():
            challenge = _valid_challenge()
            challenge["difficulty"] = difficulty
            return challenge

        with (
            patch("danyapi.pow.time", clock),
            patch("danyapi.pow.solve_native", new=_recording_solver(clock, seen, "native", 0.0)),
            patch("danyapi.pow.solve_node", new=_recording_solver(clock, seen, "node", 0.0)),
            patch("danyapi.pow.solve_python", new=_recording_solver(clock, seen, "python", 0.0)),
            pytest.raises(RuntimeError),
        ):
            await pm.make_header(fetch_ok)

    asyncio.run(run())
    return seen


def test_pow_manager_clamps_difficulty_for_every_solver():
    seen = _solve_with_seen_solvers(_SOLVER_DIFFICULTY_LIMIT * 1000)
    assert [item[0] for item in seen] == ["native", "node", "python"]
    for _name, _started_at, _budget, difficulty in seen:
        assert difficulty == _SOLVER_DIFFICULTY_LIMIT


def test_pow_manager_passes_a_sane_difficulty_through_unchanged():
    seen = _solve_with_seen_solvers(64)
    for _name, _started_at, _budget, difficulty in seen:
        assert difficulty == 64


def test_close_cancels_the_in_flight_build():
    from danyapi.pow import PowManager

    async def run():
        pm = PowManager()
        building = asyncio.create_task(asyncio.sleep(5))
        pm._building = building
        pm.close()
        assert pm._building is None
        with pytest.raises(asyncio.CancelledError):
            await building

    asyncio.run(run())


def test_ensure_build_propagates_the_build_failure():
    from danyapi.pow import PowManager

    async def run():
        pm = PowManager()

        async def fetch_bad():
            raise RuntimeError("no challenge")

        with pytest.raises(RuntimeError):
            await pm._ensure_build(fetch_bad)
        assert pm._building is None

    asyncio.run(run())


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
