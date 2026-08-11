"""DeepSeekHashV1 — proof-of-work hash chat.deepseek.com (полностью реверснут из sha3_wasm_bg.7b9ca65ddd.wasm).

Отличия от стандартного SHA3-256:
  * rate = 136 байт, capacity = 64 байта (как у SHA3-256);
  * padding: msg || 0x06 || 0x00... || 0x80 (последний байт блока);
  * ровно 23 раунда Keccak-f вместо 24;
  * round constants сдвинуты: round i использует RC[i+1] (i = 0..22).

Алгоритм верифицирован против wasm-экспортов `wasm_deepseek_hash_v1`
и `wasm_solve` (см. danyapi/tests/test_pow.py).

Правило решения чалленджа (подтверждено эмпирически):
  prefix = f"{salt}_{expire_at}_"
  answer = минимальный counter c >= 0, для которого
           hex(DeepSeekHashV1(prefix + str(c))) == challenge
  Сервер сам выбирает c из [0, difficulty) и присылает его хэш как challenge,
  клиент перебирает c пока хэш не совпадёт.
"""

from __future__ import annotations

import asyncio
import json
import logging
import struct
import subprocess
from pathlib import Path
from typing import Optional

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

_ROT = [
    [0, 36, 3, 41, 18],
    [1, 44, 10, 45, 2],
    [62, 6, 43, 15, 61],
    [28, 55, 25, 21, 56],
    [27, 20, 39, 8, 14],
]

_ROUNDS = 23

_ROUND_CONSTANTS = _RC[1:24]


def _rol(x: int, n: int) -> int:
    return ((x << n) | (x >> (64 - n))) & _MASK64


def _keccak_f(state: list[int]) -> list[int]:
    for rc in _ROUND_CONSTANTS:
        c = [
            state[x] ^ state[x + 5] ^ state[x + 10] ^ state[x + 15] ^ state[x + 20]
            for x in range(5)
        ]
        d = [c[(x - 1) % 5] ^ _rol(c[(x + 1) % 5], 1) for x in range(5)]
        for x in range(5):
            for y in range(5):
                state[x + 5 * y] ^= d[x]
        b = [0] * 25
        for x in range(5):
            for y in range(5):
                b[y + 5 * ((2 * x + 3 * y) % 5)] = _rol(state[x + 5 * y], _ROT[x][y])
        for x in range(5):
            for y in range(5):
                state[x + 5 * y] = b[x + 5 * y] ^ (
                    (~b[(x + 1) % 5 + 5 * y]) & b[(x + 2) % 5 + 5 * y]
                )
        state[0] ^= rc
    return state


def deepseek_hash_v1(data: bytes, output_bytes: int = 32) -> bytes:
    rate = 136
    state = [0] * 25
    block = bytearray(data)
    block.append(0x06)
    padlen = rate - (len(block) % rate)
    if padlen == 0:
        padlen = rate
    block += bytes(padlen)
    block[-1] |= 0x80
    for off in range(0, len(block), rate):
        chunk = block[off : off + rate]
        for i in range(0, rate, 8):
            state[i // 8] ^= struct.unpack_from("<Q", chunk, i)[0]
        _keccak_f(state)
    out = bytearray()
    while len(out) < output_bytes:
        take = min(rate, output_bytes - len(out))
        for i in range(0, take, 8):
            out += struct.pack("<Q", state[i // 8])
        if len(out) < output_bytes:
            _keccak_f(state)
    return bytes(out)


def deepseek_hash_v1_hex(data: bytes) -> str:
    return deepseek_hash_v1(data).hex()


# --------------------------------------------------------------------------
# Решение чалленджа
# --------------------------------------------------------------------------

_SOLVER_DIR = Path(__file__).resolve().parent / "deepseek"
_NATIVE_SOLVER = _SOLVER_DIR / "pow_solver.exe"
_NODE_SOLVER = _SOLVER_DIR / "pow_solver.js"


def solve_python(
    challenge_hex: str, salt: str, expire_at: int, difficulty: int
) -> Optional[int]:
    """Чисто-питоновский fallback-солвер (медленный, только для аварий)."""
    prefix = f"{salt}_{expire_at}_".encode()
    target = challenge_hex
    limit = max(0, min(int(difficulty), 2_000_000))
    for c in range(limit):
        if deepseek_hash_v1_hex(prefix + str(c).encode()) == target:
            return c
    return None


def _run_solver(
    script: Path, challenge_hex: str, salt: str, expire_at: int, difficulty: int
) -> Optional[int]:
    payload = {
        "challenge": challenge_hex,
        "salt": salt,
        "expire_at": expire_at,
        "difficulty": int(difficulty),
    }
    cmd = [str(script)] if script.suffix == ".exe" else ["node", str(script)]
    proc = subprocess.run(
        cmd,
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        timeout=60,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"{script.name} failed: {proc.stderr[:300]}")
    out = json.loads(proc.stdout.strip())
    if "error" in out:
        raise RuntimeError(out["error"])
    return out.get("answer")


def solve_native(
    challenge_hex: str, salt: str, expire_at: int, difficulty: int
) -> Optional[int]:
    """Быстрый солвер через скомпилированный C-экзешник."""
    if not _NATIVE_SOLVER.exists():
        raise FileNotFoundError("pow_solver.exe not built")
    return _run_solver(_NATIVE_SOLVER, challenge_hex, salt, expire_at, difficulty)


def solve_node(
    challenge_hex: str, salt: str, expire_at: int, difficulty: int
) -> Optional[int]:
    """Солвер через Node + оригинальный wasm-модуль сайта."""
    if not _NODE_SOLVER.exists():
        raise FileNotFoundError("pow_solver.js not found")
    return _run_solver(_NODE_SOLVER, challenge_hex, salt, expire_at, difficulty)


async def solve_challenge(
    challenge_hex: str, salt: str, expire_at: int, difficulty: int
) -> Optional[int]:
    for solver in (solve_native, solve_node, solve_python):
        try:
            answer = await asyncio.to_thread(
                solver, challenge_hex, salt, expire_at, difficulty
            )
            if answer is not None:
                return answer
        except Exception as exc:
            log.warning("pow solver %s failed (%s), trying next", solver.__name__, exc)
    return None


class PowManager:
    """Раздаёт заголовок X-DS-PoW-Response.

    Чаллендж DeepSeek ОДНОРАЗОВЫЙ: после использования сервер его инвалидирует
    (сайт сам вызывает clear у store после retrieveAnswer). Поэтому после
    выдачи заголовка сразу префетчится и решается следующий чаллендж, чтобы
    следующий запрос не ждал (~120мс через нативный солвер).
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._header: Optional[dict] = None
        self._refill: Optional[asyncio.Task] = None

    async def _build(self, fetch) -> dict:
        challenge = await fetch()
        expire_at = challenge.get("expire_at")
        answer = await solve_challenge(
            challenge["challenge"],
            challenge["salt"],
            expire_at,
            challenge["difficulty"],
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
        import base64

        return {"X-DS-PoW-Response": base64.b64encode(raw).decode()}

    async def _refill_if_empty(self, fetch) -> None:
        try:
            async with self._lock:
                if self._header is None:
                    self._header = await self._build(fetch)
        except Exception as exc:
            log.warning("pow prefetch failed: %s", exc)
        finally:
            self._refill = None

    def _kick_refill(self, fetch) -> None:
        if self._refill is None or self._refill.done():
            self._refill = asyncio.create_task(self._refill_if_empty(fetch))

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
