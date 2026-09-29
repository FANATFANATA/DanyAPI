from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess  # nosec B404
import threading
import time
from functools import lru_cache
from pathlib import Path
from typing import IO, Any

from ..pow import _solver_env as _base_solver_env

log = logging.getLogger("danyapi.duckai.attest")

ORIGIN = "https://duck.ai"

JSA_HEADER = "X-Vqd-Hash-1"

JSA_SCRIPT = Path(__file__).resolve().parent / "jsa_solver.js"

SOLVER_TIMEOUT_SEC = 20.0
SOLVER_OUTPUT_LIMIT = 1024 * 1024
SOLVER_ERROR_CHARS = 200
MAX_SCRIPT_BYTES = 1024 * 1024

INITIAL_JSA = "initial"

_SOLVER_ENV_DENYLIST = frozenset({"NODE_OPTIONS", "NODE_PATH", "NODE_REPL_EXTERNAL_MODULE", "ELECTRON_RUN_AS_NODE"})

_BASE64_RE = re.compile(r"^[A-Za-z0-9+/]+={0,2}$")

_EMPTY_CREDENTIAL = ""


class AttestationError(Exception):
    pass


@lru_cache(maxsize=1)
def _note_remote_script_exposure() -> None:
    log.warning("duckai attestation executes javascript served by the remote duck.ai host in a scrubbed subprocess; it carries no digest to verify against")


def _solver_env() -> dict[str, str]:
    return {key: value for key, value in _base_solver_env().items() if key not in _SOLVER_ENV_DENYLIST}


def _b64encode(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")


def decode_script(value: str) -> str:
    text = (value or _EMPTY_CREDENTIAL).strip()
    if not text:
        raise AttestationError("attestation script header is empty")
    if len(text) % 4 != 0 or not _BASE64_RE.match(text):
        raise AttestationError("attestation script header is not base64")
    raw = base64.b64decode(text, validate=True)
    if len(raw) > MAX_SCRIPT_BYTES:
        raise AttestationError(f"attestation script is above the {MAX_SCRIPT_BYTES} byte limit")
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
    try:
        duration = max(0, int(duration_ms))
    except (TypeError, ValueError) as exc:
        raise AttestationError(f"attestation duration is not a number: {duration_ms!r}") from exc
    extended = dict(meta)
    extended["origin"] = origin
    extended["stack"] = "no-stack"
    extended["duration"] = str(duration)
    try:
        hashed = client_hashes(raw_hashes)
        encoded = json.dumps(
            {
                "server_hashes": list(server_hashes),
                "client_hashes": hashed,
                "signals": attestation.get("signals") if isinstance(attestation.get("signals"), dict) else {},
                "meta": extended,
            },
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise AttestationError("attestation value is not JSON serialisable") from exc
    return _b64encode(encoded.encode("utf-8"))


def _is_runnable(path: Path) -> bool:
    if not path.is_file():
        return False
    if os.name == "nt":
        allowed = {item.strip().lower() for item in os.environ.get("PATHEXT", ".COM;.EXE;.BAT;.CMD").split(os.pathsep) if item.strip()}
        return path.suffix.lower() in allowed
    return os.access(path, os.X_OK)


def _node_executable() -> str | None:
    override = os.environ.get("DANYAPI_DUCKAI_NODE", "").strip()
    if override:
        path = Path(override)
        return str(path) if _is_runnable(path) else None
    return shutil.which("node")


def _drain(stream: IO[str], cap: int, out: list[str]) -> None:
    parts: list[str] = []
    kept = 0
    try:
        while True:
            chunk = stream.read(8192)
            if not chunk:
                break
            parts.append(chunk)
            kept += len(chunk)
            if kept > cap:
                drop = kept - cap
                while parts and drop > 0:
                    head = parts[0]
                    if drop < len(head):
                        parts[0] = head[drop:]
                        kept -= drop
                        break
                    drop -= len(head)
                    kept -= len(head)
                    parts.pop(0)
    except (OSError, ValueError):
        pass
    out.append("".join(parts))


def _run_solver(node: str, payload: str) -> tuple[int, str, str]:
    proc = subprocess.Popen(  # nosec B603
        [node, str(JSA_SCRIPT)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=_solver_env(),
        close_fds=True,
    )
    captured: dict[str, list[str]] = {"stdout": [], "stderr": []}
    readers = [
        threading.Thread(target=_drain, args=(proc.stdout, SOLVER_OUTPUT_LIMIT, captured["stdout"]), daemon=True),
        threading.Thread(target=_drain, args=(proc.stderr, SOLVER_OUTPUT_LIMIT, captured["stderr"]), daemon=True),
    ]
    for reader in readers:
        reader.start()
    try:
        if proc.stdin is not None:
            try:
                proc.stdin.write(payload)
                proc.stdin.close()
            except (OSError, ValueError):
                pass
        try:
            returncode = proc.wait(timeout=SOLVER_TIMEOUT_SEC)
        except subprocess.TimeoutExpired as exc:
            raise AttestationError(f"attestation solver timed out after {SOLVER_TIMEOUT_SEC:g}s") from exc
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
    except OSError as exc:
        raise AttestationError(f"attestation solver is not runnable: {exc}") from exc
    finally:
        for reader in readers:
            reader.join(timeout=SOLVER_TIMEOUT_SEC)
        for handle in (proc.stdout, proc.stderr, proc.stdin):
            if handle is not None:
                try:
                    handle.close()
                except OSError:
                    pass
    return returncode, "".join(captured["stdout"]), "".join(captured["stderr"])


def _solver_result(stdout: str) -> Any:
    text = stdout.strip()
    if not text:
        raise AttestationError("attestation solver returned no output")
    for candidate in (text.rsplit("\n", 1)[-1].strip(), text):
        try:
            return json.loads(candidate)
        except ValueError:
            continue
    raise AttestationError(f"attestation solver returned malformed output: {text[:SOLVER_ERROR_CHARS]}")


def evaluate_sync(script: str, user_agent: str) -> dict:
    node = _node_executable()
    if node is None:
        raise AttestationError("node is required to solve the duckai attestation, set DANYAPI_DUCKAI_NODE or install node")
    if not JSA_SCRIPT.is_file():
        raise AttestationError(f"attestation solver is missing: {JSA_SCRIPT.name}")
    payload = json.dumps({"script": script, "user_agent": user_agent})
    returncode, stdout, stderr = _run_solver(node, payload)
    if returncode != 0:
        raise AttestationError(f"attestation solver failed: {stderr.strip()[:SOLVER_ERROR_CHARS]}")
    result = _solver_result(stdout)
    if not isinstance(result, dict) or result.get("ok") is not True:
        detail = result.get("error") if isinstance(result, dict) else None
        raise AttestationError(f"attestation solver rejected the script: {str(detail)[:SOLVER_ERROR_CHARS]}")
    attestation = result.get("result")
    if not isinstance(attestation, dict):
        raise AttestationError("attestation solver returned no attestation")
    return attestation


async def evaluate(script: str, user_agent: str) -> dict:
    return await asyncio.to_thread(evaluate_sync, script, user_agent)


async def header_for(script_b64: str, user_agent: str, origin: str = ORIGIN) -> str:
    if not script_b64:
        return INITIAL_JSA
    started = time.monotonic()
    script = decode_script(script_b64)
    _note_remote_script_exposure()
    attestation = await evaluate(script, user_agent)
    header = build_header(attestation, origin=origin, duration_ms=int((time.monotonic() - started) * 1000))
    log.debug("duckai attestation solved in %dms", int((time.monotonic() - started) * 1000))
    return header


def fraud_signals(started: float | None = None) -> str:
    start = started if started is not None else time.time() * 1000.0
    payload = {
        "start": int(start),
        "events": [],
        "end": max(0, int(time.time() * 1000.0 - start)),
    }
    return _b64encode(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
