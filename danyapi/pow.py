from __future__ import annotations

import asyncio
import base64
import json
import logging
import math
import os
import struct
import subprocess
import time
from contextlib import suppress
from pathlib import Path

log = logging.getLogger("danyapi.pow")

_MASK64 = (1 << 64) - 1

_RC = [
    0x0000000000000001,
    0x0000000000008082,
    0x800000000000808A,
    0x8000000080008000,
    0x000000000000808B,
    0x0000000080000001,
    0x8000000080008081,
    0x8000000000008009,
    0x000000000000008A,
    0x0000000000000088,
    0x0000000080008009,
    0x000000008000000A,
    0x000000008000808B,
    0x800000000000008B,
    0x8000000000008089,
    0x8000000000008003,
    0x8000000000008002,
    0x8000000000000080,
    0x000000000000800A,
    0x800000008000000A,
    0x8000000080008081,
    0x8000000000008080,
    0x0000000080000001,
    0x8000000080008008,
]

_ROUNDS = 23

_ROUND_CONSTANTS = _RC[1 : _ROUNDS + 1]

_PYTHON_SOLVE_LIMIT = 2_000_000
_PYTHON_SOLVE_BUDGET_SEC = 5.0
_PYTHON_BUDGET_CHECK_INTERVAL = 1024

_SOLVER_TIMEOUT_SEC = 60.0

_SOLVER_ENV_DENYLIST = frozenset({"DEEPSEEK_TOKENS", "QWEN_TOKENS", "BYOK", "BYOK_MODE", "DANYAPI_BYOK_MODE"})


def _parse_number(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return int(text)
        except ValueError:
            pass
        try:
            number = float(text)
        except ValueError:
            return None
        return number if math.isfinite(number) else None
    return None


def _keccak_f(state: list[int]) -> list[int]:
    mask = _MASK64
    c = [0] * 5
    d = [0] * 5
    b = [0] * 25
    for rc in _ROUND_CONSTANTS:
        c[0] = state[0] ^ state[5] ^ state[10] ^ state[15] ^ state[20]
        c[1] = state[1] ^ state[6] ^ state[11] ^ state[16] ^ state[21]
        c[2] = state[2] ^ state[7] ^ state[12] ^ state[17] ^ state[22]
        c[3] = state[3] ^ state[8] ^ state[13] ^ state[18] ^ state[23]
        c[4] = state[4] ^ state[9] ^ state[14] ^ state[19] ^ state[24]
        d[0] = c[4] ^ (((c[1] << 1) | (c[1] >> 63)) & mask)
        d[1] = c[0] ^ (((c[2] << 1) | (c[2] >> 63)) & mask)
        d[2] = c[1] ^ (((c[3] << 1) | (c[3] >> 63)) & mask)
        d[3] = c[2] ^ (((c[4] << 1) | (c[4] >> 63)) & mask)
        d[4] = c[3] ^ (((c[0] << 1) | (c[0] >> 63)) & mask)
        state[0] ^= d[0]
        state[1] ^= d[1]
        state[2] ^= d[2]
        state[3] ^= d[3]
        state[4] ^= d[4]
        state[5] ^= d[0]
        state[6] ^= d[1]
        state[7] ^= d[2]
        state[8] ^= d[3]
        state[9] ^= d[4]
        state[10] ^= d[0]
        state[11] ^= d[1]
        state[12] ^= d[2]
        state[13] ^= d[3]
        state[14] ^= d[4]
        state[15] ^= d[0]
        state[16] ^= d[1]
        state[17] ^= d[2]
        state[18] ^= d[3]
        state[19] ^= d[4]
        state[20] ^= d[0]
        state[21] ^= d[1]
        state[22] ^= d[2]
        state[23] ^= d[3]
        state[24] ^= d[4]
        b[0] = state[0]
        b[16] = ((state[5] << 36) | (state[5] >> 28)) & mask
        b[7] = ((state[10] << 3) | (state[10] >> 61)) & mask
        b[23] = ((state[15] << 41) | (state[15] >> 23)) & mask
        b[14] = ((state[20] << 18) | (state[20] >> 46)) & mask
        b[10] = ((state[1] << 1) | (state[1] >> 63)) & mask
        b[1] = ((state[6] << 44) | (state[6] >> 20)) & mask
        b[17] = ((state[11] << 10) | (state[11] >> 54)) & mask
        b[8] = ((state[16] << 45) | (state[16] >> 19)) & mask
        b[24] = ((state[21] << 2) | (state[21] >> 62)) & mask
        b[20] = ((state[2] << 62) | (state[2] >> 2)) & mask
        b[11] = ((state[7] << 6) | (state[7] >> 58)) & mask
        b[2] = ((state[12] << 43) | (state[12] >> 21)) & mask
        b[18] = ((state[17] << 15) | (state[17] >> 49)) & mask
        b[9] = ((state[22] << 61) | (state[22] >> 3)) & mask
        b[5] = ((state[3] << 28) | (state[3] >> 36)) & mask
        b[21] = ((state[8] << 55) | (state[8] >> 9)) & mask
        b[12] = ((state[13] << 25) | (state[13] >> 39)) & mask
        b[3] = ((state[18] << 21) | (state[18] >> 43)) & mask
        b[19] = ((state[23] << 56) | (state[23] >> 8)) & mask
        b[15] = ((state[4] << 27) | (state[4] >> 37)) & mask
        b[6] = ((state[9] << 20) | (state[9] >> 44)) & mask
        b[22] = ((state[14] << 39) | (state[14] >> 25)) & mask
        b[13] = ((state[19] << 8) | (state[19] >> 56)) & mask
        b[4] = ((state[24] << 14) | (state[24] >> 50)) & mask
        state[0] = b[0] ^ ((~b[1]) & b[2])
        state[1] = b[1] ^ ((~b[2]) & b[3])
        state[2] = b[2] ^ ((~b[3]) & b[4])
        state[3] = b[3] ^ ((~b[4]) & b[0])
        state[4] = b[4] ^ ((~b[0]) & b[1])
        state[5] = b[5] ^ ((~b[6]) & b[7])
        state[6] = b[6] ^ ((~b[7]) & b[8])
        state[7] = b[7] ^ ((~b[8]) & b[9])
        state[8] = b[8] ^ ((~b[9]) & b[5])
        state[9] = b[9] ^ ((~b[5]) & b[6])
        state[10] = b[10] ^ ((~b[11]) & b[12])
        state[11] = b[11] ^ ((~b[12]) & b[13])
        state[12] = b[12] ^ ((~b[13]) & b[14])
        state[13] = b[13] ^ ((~b[14]) & b[10])
        state[14] = b[14] ^ ((~b[10]) & b[11])
        state[15] = b[15] ^ ((~b[16]) & b[17])
        state[16] = b[16] ^ ((~b[17]) & b[18])
        state[17] = b[17] ^ ((~b[18]) & b[19])
        state[18] = b[18] ^ ((~b[19]) & b[15])
        state[19] = b[19] ^ ((~b[15]) & b[16])
        state[20] = b[20] ^ ((~b[21]) & b[22])
        state[21] = b[21] ^ ((~b[22]) & b[23])
        state[22] = b[22] ^ ((~b[23]) & b[24])
        state[23] = b[23] ^ ((~b[24]) & b[20])
        state[24] = b[24] ^ ((~b[20]) & b[21])
        state[0] ^= rc
    return state


_RATE = 136


def deepseek_hash_v1(data: bytes, output_bytes: int = 32) -> bytes:
    state = [0] * 25
    block = bytearray(data)
    block.append(0x06)
    padlen = _RATE - (len(block) % _RATE)
    block += bytes(padlen)
    block[-1] |= 0x80
    mv = memoryview(block).cast("Q")
    stride = _RATE // 8
    for off in range(0, len(block), _RATE):
        base = off // 8
        for i in range(stride):
            state[i] ^= mv[base + i]
        _keccak_f(state)
    out = bytearray()
    while len(out) < output_bytes:
        take = min(_RATE, output_bytes - len(out))
        for i in range(0, take, 8):
            out += struct.pack("<Q", state[i // 8])
        if len(out) < output_bytes:
            _keccak_f(state)
    return bytes(out[:output_bytes])


def _hash_block(data: bytes) -> bytes:
    state = [0] * 25
    mv = memoryview(data).cast("Q")
    for i in range(_RATE // 8):
        state[i] ^= mv[i]
    _keccak_f(state)
    out = bytearray(32)
    for i in range(4):
        struct.pack_into("<Q", out, i * 8, state[i])
    return bytes(out)


def deepseek_hash_v1_hex(data: bytes) -> str:
    return deepseek_hash_v1(data).hex()


_SOLVER_DIR = Path(__file__).resolve().parent / "deepseek"
_NODE_SOLVER = _SOLVER_DIR / "pow_solver.js"


def _native_solver_names() -> tuple[str, ...]:
    if os.name == "nt":
        return ("pow_solver.exe", "pow_solver")
    return ("pow_solver", "pow_solver.exe")


def _find_native_solver() -> Path | None:
    for name in _native_solver_names():
        candidate = _SOLVER_DIR / name
        if candidate.is_file():
            return candidate
    return None


def solve_python(challenge_hex: str, salt: str, expire_at: int, difficulty: int, budget: float = _PYTHON_SOLVE_BUDGET_SEC) -> int | None:
    prefix = f"{salt}_{expire_at}_".encode()
    target = bytes.fromhex(challenge_hex)
    limit = max(0, min(int(difficulty), _PYTHON_SOLVE_LIMIT))
    if limit == 0:
        return None
    deadline = time.monotonic() + budget
    check = _PYTHON_BUDGET_CHECK_INTERVAL
    pfx_len = len(prefix)
    max_width = len(str(limit - 1))
    if pfx_len + max_width <= _RATE - 2:
        tails: dict[int, bytes] = {}
        for width in range(1, max_width + 1):
            tails[width] = bytes([0x06]) + b"\x00" * (_RATE - pfx_len - width - 2) + b"\x80"
        for c in range(limit):
            if c % check == 0 and time.monotonic() >= deadline:
                return None
            digits = str(c).encode()
            if _hash_block(prefix + digits + tails[len(digits)]) == target:
                return c
        return None
    for c in range(limit):
        if c % check == 0 and time.monotonic() >= deadline:
            return None
        if deepseek_hash_v1(prefix + str(c).encode()) == target:
            return c
    return None


def _solver_env() -> dict[str, str]:
    return {key: value for key, value in os.environ.items() if key not in _SOLVER_ENV_DENYLIST}


def _run_solver(script: Path, challenge_hex: str, salt: str, expire_at: int, difficulty: int) -> int | None:
    payload = {
        "challenge": challenge_hex,
        "salt": salt,
        "expire_at": str(expire_at),
        "difficulty": int(difficulty),
    }
    cmd = ["node", str(script)] if script.suffix == ".js" else [str(script)]
    try:
        proc = subprocess.run(
            cmd,
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            timeout=_SOLVER_TIMEOUT_SEC,
            check=False,
            env=_solver_env(),
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"{script.name} timed out after {_SOLVER_TIMEOUT_SEC:g}s") from exc
    except OSError as exc:
        raise RuntimeError(f"{script.name} is not executable: {exc}") from exc
    if proc.returncode != 0:
        raise RuntimeError(f"{script.name} failed: {proc.stderr[:300]}")
    try:
        out = json.loads(proc.stdout.strip())
    except ValueError as exc:
        raise RuntimeError(f"{script.name} returned malformed output: {proc.stdout[:300]}") from exc
    if not isinstance(out, dict):
        raise RuntimeError(f"{script.name} returned malformed output: {proc.stdout[:300]}")
    if "error" in out:
        raise RuntimeError(str(out["error"]))
    answer = out.get("answer")
    if isinstance(answer, bool) or not isinstance(answer, int) or answer < 0:
        raise RuntimeError(f"{script.name} returned an invalid answer: {out!r:.200}")
    return answer


def solve_native(challenge_hex: str, salt: str, expire_at: int, difficulty: int) -> int | None:
    native = _find_native_solver()
    if native is None:
        raise FileNotFoundError("native pow_solver binary not built")
    return _run_solver(native, challenge_hex, salt, expire_at, difficulty)


def solve_node(challenge_hex: str, salt: str, expire_at: int, difficulty: int) -> int | None:
    if not _NODE_SOLVER.exists():
        raise FileNotFoundError("pow_solver.js not found")
    return _run_solver(_NODE_SOLVER, challenge_hex, salt, expire_at, difficulty)


async def solve_challenge(challenge_hex: str, salt: str, expire_at: int, difficulty: int) -> int | None:
    for solver in (solve_native, solve_node, solve_python):
        try:
            answer = await asyncio.to_thread(solver, challenge_hex, salt, expire_at, difficulty)
            if answer is not None:
                return answer
        except Exception as exc:
            log.warning("pow solver %s failed (%s), trying next", solver.__name__, exc)
    return None


class PowManager:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._header: dict | None = None
        self._refill: asyncio.Task | None = None
        self._building: asyncio.Task | None = None

    async def _build(self, fetch) -> dict:
        challenge = await fetch()
        missing = [k for k in ("challenge", "salt", "algorithm", "signature", "target_path") if not challenge.get(k)]
        if missing:
            raise RuntimeError(f"pow challenge missing fields: {', '.join(missing)}")
        expire_at = _parse_number(challenge.get("expire_at"))
        difficulty = _parse_number(challenge.get("difficulty"))
        if expire_at is None or expire_at < 0:
            raise RuntimeError("pow challenge has invalid expire_at")
        if difficulty is None or difficulty <= 0:
            raise RuntimeError("pow challenge has invalid difficulty")
        answer = await solve_challenge(
            challenge["challenge"],
            challenge["salt"],
            int(expire_at),
            int(difficulty),
        )
        if answer is None:
            raise RuntimeError("pow solver returned no answer")
        payload = {
            "algorithm": challenge["algorithm"],
            "challenge": challenge["challenge"],
            "salt": challenge["salt"],
            "answer": answer,
            "signature": challenge["signature"],
            "target_path": challenge["target_path"],
        }
        raw = json.dumps(payload, separators=(",", ":")).encode()
        return {"X-DS-PoW-Response": base64.b64encode(raw).decode()}

    async def _ensure_build(self, fetch) -> dict:
        current = self._building
        if current is None or current.done():
            self._building = asyncio.create_task(self._build(fetch))
            current = self._building
        try:
            return await asyncio.shield(current)
        except Exception:
            if current is self._building and current.done():
                self._building = None
            raise

    async def _refill_if_empty(self, fetch) -> None:
        try:
            async with self._lock:
                if self._header is None:
                    self._header = await self._ensure_build(fetch)
        except Exception as exc:
            log.warning("pow prefetch failed: %s", exc)
        finally:
            self._refill = None

    def _kick_refill(self, fetch) -> None:
        current = self._refill
        if current is not None:
            if not current.done():
                return
            with suppress(asyncio.CancelledError):
                current.exception()
        self._refill = asyncio.create_task(self._refill_if_empty(fetch))

    def close(self) -> None:
        current = self._refill
        self._refill = None
        if current is not None and not current.done():
            current.cancel()

    async def make_header(self, fetch) -> dict:
        async with self._lock:
            if self._header is not None:
                header = self._header
                self._header = None
            else:
                header = None
        if header is not None:
            self._kick_refill(fetch)
            return header
        header = await self._build(fetch)
        self._kick_refill(fetch)
        return header
