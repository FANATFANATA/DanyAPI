from __future__ import annotations

import asyncio
import base64
import binascii
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess  # nosec B404
import time
from pathlib import Path
from typing import Any

log = logging.getLogger("danyapi.duckai.attest")

ORIGIN = "https://duck.ai"

JSA_HEADER = "X-Vqd-Hash-1"

JSA_SCRIPT = Path(__file__).resolve().parent / "jsa_solver.js"

SOLVER_TIMEOUT_SEC = 20.0

_INITIAL_JSA = "initial"

INITIAL_JSA = _INITIAL_JSA

_BASE64_RE = re.compile(r"^[A-Za-z0-9+/]+={0,2}$")

_EMPTY_CREDENTIAL = ""


class AttestationError(Exception):
    pass


def _b64encode(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")


def decode_script(value: str) -> str:
    text = (value or _EMPTY_CREDENTIAL).strip()
    if not text:
        raise AttestationError("attestation script header is empty")
    if len(text) % 4 != 0 or not _BASE64_RE.match(text):
        raise AttestationError("attestation script header is not base64")
    try:
        raw = base64.b64decode(text, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise AttestationError("attestation script header is not valid base64") from exc
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise AttestationError("attestation script is not utf-8") from exc


def client_hashes(values: list[Any]) -> list[str]:
    hashed: list[str] = []
    for value in values:
        text = value if isinstance(value, str) else json.dumps(value, separators=(",", ":"), sort_keys=True)
        hashed.append(_b64encode(hashlib.sha256(text.encode("utf-8")).digest()))
    return hashed


def build_header(attestation: dict, origin: str = ORIGIN, duration_ms: int = 0) -> str:
    raw_hashes = attestation.get("client_hashes")
    meta = attestation.get("meta")
    server_hashes = attestation.get("server_hashes")
    if not isinstance(raw_hashes, list) or not raw_hashes:
        raise AttestationError("attestation has no client_hashes")
    if not isinstance(server_hashes, list) or not server_hashes:
        raise AttestationError("attestation has no server_hashes")
    if not isinstance(meta, dict):
        raise AttestationError("attestation has no meta")
    extended = dict(meta)
    extended["origin"] = origin
    extended["stack"] = "no-stack"
    extended["duration"] = str(max(0, int(duration_ms)))
    payload = {
        "server_hashes": list(server_hashes),
        "client_hashes": client_hashes(raw_hashes),
        "signals": attestation.get("signals") if isinstance(attestation.get("signals"), dict) else {},
        "meta": extended,
    }
    return _b64encode(json.dumps(payload, separators=(",", ":")).encode("utf-8"))


def _node_executable() -> str | None:
    override = os.environ.get("DANYAPI_DUCKAI_NODE", "").strip()
    if override:
        return override if Path(override).is_file() else None
    return shutil.which("node")


def evaluate_sync(script: str, user_agent: str) -> dict:
    node = _node_executable()
    if node is None:
        raise AttestationError("node is required to solve the duckai attestation, set DANYAPI_DUCKAI_NODE or install node")
    if not JSA_SCRIPT.is_file():
        raise AttestationError(f"attestation solver is missing: {JSA_SCRIPT.name}")
    payload = json.dumps({"script": script, "user_agent": user_agent})
    try:
        proc = subprocess.run(  # nosec B603
            [node, str(JSA_SCRIPT)],
            input=payload,
            capture_output=True,
            text=True,
            timeout=SOLVER_TIMEOUT_SEC,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise AttestationError(f"attestation solver timed out after {SOLVER_TIMEOUT_SEC:g}s") from exc
    except OSError as exc:
        raise AttestationError(f"attestation solver is not runnable: {exc}") from exc
    if proc.returncode != 0:
        raise AttestationError(f"attestation solver failed: {(proc.stderr or '').strip()[:200]}")
    try:
        result = json.loads(proc.stdout.strip())
    except ValueError as exc:
        raise AttestationError(f"attestation solver returned malformed output: {proc.stdout[:200]}") from exc
    if not isinstance(result, dict) or result.get("ok") is not True:
        detail = result.get("error") if isinstance(result, dict) else None
        raise AttestationError(f"attestation solver rejected the script: {str(detail)[:200]}")
    attestation = result.get("result")
    if not isinstance(attestation, dict):
        raise AttestationError("attestation solver returned no attestation")
    return attestation


async def evaluate(script: str, user_agent: str) -> dict:
    return await asyncio.to_thread(evaluate_sync, script, user_agent)


async def header_for(script_b64: str, user_agent: str, origin: str = ORIGIN) -> str:
    if not script_b64:
        return _INITIAL_JSA
    started = time.monotonic()
    script = decode_script(script_b64)
    attestation = await evaluate(script, user_agent)
    header = build_header(attestation, origin=origin, duration_ms=int((time.monotonic() - started) * 1000))
    log.debug("duckai attestation solved in %dms", int((time.monotonic() - started) * 1000))
    return header


def fraud_signals(events: list[dict] | None = None, started: float | None = None) -> str:
    start = started if started is not None else time.time() * 1000.0
    payload = {
        "start": int(start),
        "events": [event for event in (events or []) if isinstance(event, dict)],
        "end": max(0, int(time.time() * 1000.0 - start)),
    }
    return _b64encode(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
