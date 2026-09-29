import base64
import json
import logging
import os
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace
from typing import IO, Any, cast

import pytest

from danyapi.duckai import attest


class _FakeStream:
    def __init__(
        self,
        text: str = "",
        chunks: list[str] | None = None,
        read_error: BaseException | None = None,
        fail_after: int = 0,
        close_error: BaseException | None = None,
    ) -> None:
        self.text = text
        self.chunks = list(chunks) if chunks is not None else None
        self.read_error = read_error
        self.fail_after = fail_after
        self.close_error = close_error
        self.position = 0
        self.closed = False
        self.reads = 0

    def read(self, size: int = -1) -> str:
        self.reads += 1
        if self.read_error is not None and self.reads > self.fail_after:
            raise self.read_error
        if self.chunks is not None:
            return self.chunks.pop(0) if self.chunks else ""
        if size is None or size < 0:
            chunk = self.text[self.position :]
        else:
            chunk = self.text[self.position : self.position + size]
        self.position += len(chunk)
        return chunk

    def close(self) -> None:
        self.closed = True
        if self.close_error is not None:
            raise self.close_error


class _FakeStdin:
    def __init__(self, write_error: BaseException | None = None, close_error: BaseException | None = None) -> None:
        self.write_error = write_error
        self.close_error = close_error
        self.written: list[str] = []
        self.closed = False

    def write(self, data: str) -> None:
        if self.write_error is not None:
            raise self.write_error
        self.written.append(data)

    def close(self) -> None:
        self.closed = True
        if self.close_error is not None:
            raise self.close_error


class _FakeProc:
    def __init__(
        self,
        stdout: str = "",
        stderr: str = "",
        returncode: int = 0,
        wait_script: list[Any] | None = None,
        poll_result: int | None = 0,
        with_stdin: bool = True,
        stdin_write_error: BaseException | None = None,
        stdout_close_error: BaseException | None = None,
    ) -> None:
        self.stdin = _FakeStdin(stdin_write_error) if with_stdin else None
        self.stdout = _FakeStream(stdout, close_error=stdout_close_error)
        self.stderr = _FakeStream(stderr)
        self.returncode = returncode
        self.wait_script = list(wait_script or [])
        self.poll_result = poll_result
        self.killed = False
        self.wait_timeouts: list[Any] = []

    def stdin_handle(self) -> _FakeStdin:
        if self.stdin is None:
            raise AssertionError("the fake process was built without a stdin pipe")
        return self.stdin

    def wait(self, timeout: float | None = None) -> int:
        self.wait_timeouts.append(timeout)
        if self.wait_script:
            item = self.wait_script.pop(0)
            if isinstance(item, BaseException):
                raise item
            return int(item)
        return self.returncode

    def poll(self) -> int | None:
        return self.poll_result

    def kill(self) -> None:
        self.killed = True


class _FakePopen:
    def __init__(self, proc: _FakeProc) -> None:
        self.proc = proc
        self.calls: list[tuple[Any, dict]] = []

    def __call__(self, args: list[str], **kwargs: Any) -> _FakeProc:
        self.calls.append((args, kwargs))
        return self.proc


def _as_stream(stream: _FakeStream) -> IO[str]:
    return cast(IO[str], stream)


class _FakeTime:
    def __init__(self, seconds: float) -> None:
        self.seconds = seconds

    def time(self) -> float:
        return self.seconds

    def monotonic(self) -> float:
        return self.seconds

    def __getattr__(self, name: str) -> Any:
        return getattr(time, name)


def _popen(monkeypatch, proc: _FakeProc) -> _FakePopen:
    fake = _FakePopen(proc)
    monkeypatch.setattr(attest.subprocess, "Popen", fake)
    return fake


def _posix_os(access_result: bool, recorded: list) -> SimpleNamespace:
    def _access(path: Path, mode: int) -> bool:
        recorded.append((path, mode))
        return access_result

    return SimpleNamespace(name="posix", environ=os.environ, access=_access, X_OK=os.X_OK)


def test_b64encode_returns_ascii_base64() -> None:
    assert attest._b64encode(b"abc") == "YWJj"


def test_client_hashes_sorts_keys_for_structured_values() -> None:
    hashed = attest.client_hashes(["abc", {"b": 1, "a": 2}])
    assert hashed == [
        "ungWv48Bz+pBQUDeXa4iI7ADYaOWF3qctBD/YfIAFa0=",
        "02JqwwqH5vemQoIzs8aCmZdoZfpVCOQmfFQVx2r3p3I=",
    ]
    assert all(len(item) == 44 for item in hashed)


def test_client_hashes_keeps_plain_strings_verbatim() -> None:
    assert attest.client_hashes(["  spaced  "]) == ["HcwkoUpr+MtNkXUqmNuP/CWeNxVpA6eA+CbCTeYeKgU="]


def test_decode_script_returns_the_utf8_payload() -> None:
    payload = b"() => 1"
    assert attest.decode_script(base64.b64encode(payload).decode()) == payload.decode()
    assert attest.decode_script(f"  {base64.b64encode(payload).decode()}  ") == payload.decode()


def test_decode_script_rejects_an_empty_header() -> None:
    with pytest.raises(attest.AttestationError) as excinfo:
        attest.decode_script("")
    assert str(excinfo.value) == "attestation script header is empty"
    with pytest.raises(attest.AttestationError) as empty:
        attest.decode_script(cast(Any, None))
    assert str(empty.value) == "attestation script header is empty"


def test_decode_script_rejects_a_misaligned_header() -> None:
    with pytest.raises(attest.AttestationError) as excinfo:
        attest.decode_script("not base64!!")
    assert str(excinfo.value) == "attestation script header is not base64"
    with pytest.raises(attest.AttestationError) as symbols:
        attest.decode_script("AAAAAAA*")
    assert str(symbols.value) == "attestation script header is not base64"


def test_decode_script_rejects_a_payload_over_the_cap() -> None:
    assert attest.MAX_SCRIPT_BYTES == 1024 * 1024
    oversized = base64.b64encode(b"a" * (attest.MAX_SCRIPT_BYTES + 1)).decode()
    with pytest.raises(attest.AttestationError) as excinfo:
        attest.decode_script(oversized)
    assert str(excinfo.value) == f"attestation script is above the {attest.MAX_SCRIPT_BYTES} byte limit"


def test_decode_script_rejects_a_non_utf8_payload() -> None:
    with pytest.raises(attest.AttestationError) as excinfo:
        attest.decode_script(base64.b64encode(b"\xff\xfe\x00").decode())
    assert str(excinfo.value) == "attestation script is not utf-8"


def test_build_header_encodes_the_attestation() -> None:
    header = attest.build_header(
        {
            "client_hashes": ["abc"],
            "server_hashes": ["srv"],
            "signals": {"k": 1},
            "meta": {"m": "v"},
        },
        origin="https://duck.ai",
        duration_ms=1234,
    )
    decoded = json.loads(base64.b64decode(header))
    assert decoded["server_hashes"] == ["srv"]
    assert decoded["client_hashes"] == ["ungWv48Bz+pBQUDeXa4iI7ADYaOWF3qctBD/YfIAFa0="]
    assert decoded["signals"] == {"k": 1}
    assert decoded["meta"] == {"m": "v", "origin": "https://duck.ai", "stack": "no-stack", "duration": "1234"}


def test_build_header_clamps_a_negative_duration() -> None:
    header = attest.build_header({"client_hashes": ["a"], "server_hashes": ["s"], "meta": {}}, duration_ms=-5)
    assert json.loads(base64.b64decode(header))["meta"]["duration"] == "0"


def test_build_header_defaults_a_missing_signals_block() -> None:
    header = attest.build_header({"client_hashes": ["a"], "server_hashes": ["s"], "meta": {}, "signals": "nope"})
    assert json.loads(base64.b64decode(header))["signals"] == {}


@pytest.mark.parametrize(
    "attestation,message",
    [
        ({"server_hashes": ["s"], "meta": {}}, "attestation has no client_hashes"),
        ({"client_hashes": [], "server_hashes": ["s"], "meta": {}}, "attestation has no client_hashes"),
        ({"client_hashes": ["c"], "meta": {}}, "attestation has no server_hashes"),
        ({"client_hashes": ["c"], "server_hashes": "s", "meta": {}}, "attestation has no server_hashes"),
        ({"client_hashes": ["c"], "server_hashes": ["s"]}, "attestation has no meta"),
        ({"client_hashes": ["c"], "server_hashes": ["s"], "meta": []}, "attestation has no meta"),
    ],
)
def test_build_header_validates_the_attestation_shape(attestation, message) -> None:
    with pytest.raises(attest.AttestationError) as excinfo:
        attest.build_header(attestation)
    assert str(excinfo.value) == message


def test_build_header_rejects_a_non_numeric_duration() -> None:
    with pytest.raises(attest.AttestationError) as excinfo:
        attest.build_header({"client_hashes": ["c"], "server_hashes": ["s"], "meta": {}}, duration_ms=cast(Any, "soon"))
    assert str(excinfo.value) == "attestation duration is not a number: 'soon'"


def test_build_header_rejects_an_unserialisable_value() -> None:
    with pytest.raises(attest.AttestationError) as excinfo:
        attest.build_header({"client_hashes": [object()], "server_hashes": ["s"], "meta": {}})
    assert str(excinfo.value) == "attestation value is not JSON serialisable"


def test_is_runnable_rejects_a_plain_text_file(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("PATHEXT", ".COM;.EXE;.BAT;.CMD")
    target = tmp_path / "node.txt"
    target.write_text("not an executable", encoding="utf-8")
    assert attest._is_runnable(target) is False


def test_is_runnable_rejects_a_directory(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("PATHEXT", ".COM;.EXE;.BAT;.CMD")
    assert attest._is_runnable(tmp_path) is False


def test_is_runnable_follows_the_pathext_list_on_windows(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("PATHEXT", ".COM;.EXE;.BAT;.CMD")
    accepted = tmp_path / "node.EXE"
    accepted.write_text("MZ", encoding="utf-8")
    ignored = tmp_path / "node.sh"
    ignored.write_text("#!/bin/sh", encoding="utf-8")
    assert attest._is_runnable(accepted) is (os.name == "nt")
    assert attest._is_runnable(ignored) is False


def test_is_runnable_uses_x_ok_off_windows(monkeypatch, tmp_path) -> None:
    recorded: list[tuple[Path, int]] = []
    monkeypatch.setattr(attest, "os", _posix_os(True, recorded))
    target = tmp_path / "node"
    target.write_text("x", encoding="utf-8")
    assert attest._is_runnable(target) is True
    assert recorded == [(target, os.X_OK)]


def test_is_runnable_rejects_a_non_executable_off_windows(monkeypatch, tmp_path) -> None:
    recorded: list[tuple[Path, int]] = []
    monkeypatch.setattr(attest, "os", _posix_os(False, recorded))
    target = tmp_path / "node"
    target.write_text("x", encoding="utf-8")
    assert attest._is_runnable(target) is False
    assert recorded == [(target, os.X_OK)]


def test_node_executable_prefers_a_runnable_override(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("PATHEXT", ".COM;.EXE;.BAT;.CMD")
    target = tmp_path / "custom-node.exe"
    target.write_text("MZ", encoding="utf-8")
    if os.name != "nt":
        target.chmod(0o755)
    monkeypatch.setenv("DANYAPI_DUCKAI_NODE", f"  {target}  ")
    assert attest._node_executable() == str(target)


def test_node_executable_rejects_an_override_without_an_executable_suffix(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("PATHEXT", ".COM;.EXE;.BAT;.CMD")
    target = tmp_path / "node.txt"
    target.write_text("MZ", encoding="utf-8")
    monkeypatch.setenv("DANYAPI_DUCKAI_NODE", str(target))
    assert attest._node_executable() is None


def test_node_executable_falls_back_to_the_path_lookup(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("DANYAPI_DUCKAI_NODE", "   ")
    monkeypatch.setattr(attest.shutil, "which", lambda name: f"C:/fake/{name}.exe" if name == "node" else None)
    assert attest._node_executable() == "C:/fake/node.exe"


def test_node_executable_is_none_when_node_is_absent(monkeypatch) -> None:
    monkeypatch.delenv("DANYAPI_DUCKAI_NODE", raising=False)
    monkeypatch.setattr(attest.shutil, "which", lambda name: None)
    assert attest._node_executable() is None


def test_drain_trims_a_single_oversized_chunk() -> None:
    out: list[str] = []
    attest._drain(_as_stream(_FakeStream("abcdefgh")), 3, out)
    assert out == ["fgh"]


def test_drain_drops_whole_leading_chunks_when_the_overflow_covers_them() -> None:
    out: list[str] = []
    attest._drain(_as_stream(_FakeStream(chunks=["ab", "cd", "ef"])), 2, out)
    assert out == ["ef"]


def test_drain_keeps_everything_below_the_cap() -> None:
    out: list[str] = []
    attest._drain(_as_stream(_FakeStream(chunks=["ab", "cd"])), 64, out)
    assert out == ["abcd"]


def test_drain_swallows_a_read_failure_and_keeps_what_it_read() -> None:
    out: list[str] = []
    stream = _FakeStream(chunks=["ab"], read_error=OSError("pipe closed"), fail_after=1)
    attest._drain(_as_stream(stream), 64, out)
    assert out == ["ab"]
    assert stream.reads == 2


def test_solver_env_drops_node_injection_vectors(monkeypatch) -> None:
    monkeypatch.setenv("NODE_OPTIONS", "--require=C:/tmp/evil.js")
    monkeypatch.setenv("NODE_PATH", "C:/tmp/modules")
    monkeypatch.setenv("NODE_REPL_EXTERNAL_MODULE", "C:/tmp/repl.js")
    monkeypatch.setenv("ELECTRON_RUN_AS_NODE", "1")
    monkeypatch.setenv("PATH", "C:/windows")
    base = attest._base_solver_env()
    assert {"NODE_OPTIONS", "NODE_PATH", "NODE_REPL_EXTERNAL_MODULE", "ELECTRON_RUN_AS_NODE"} <= set(base)
    env = attest._solver_env()
    assert set(env) & attest._SOLVER_ENV_DENYLIST == set()
    assert env["PATH"] == "C:/windows"
    assert len(env) == len(base) - 4


def test_run_solver_hands_the_scrubbed_env_to_the_child(monkeypatch) -> None:
    monkeypatch.setenv("NODE_OPTIONS", "--require=C:/tmp/evil.js")
    monkeypatch.setenv("NODE_PATH", "C:/tmp/modules")
    monkeypatch.setenv("NODE_REPL_EXTERNAL_MODULE", "C:/tmp/repl.js")
    monkeypatch.setenv("ELECTRON_RUN_AS_NODE", "1")
    monkeypatch.setenv("PATH", "C:/windows")
    popen = _popen(monkeypatch, _FakeProc(stdout='{"ok":true}\n'))
    returncode, stdout, stderr = attest._run_solver("C:/node.exe", '{"script":"x"}')
    args, kwargs = popen.calls[0]
    assert returncode == 0
    assert stdout == '{"ok":true}\n'
    assert stderr == ""
    assert args == ["C:/node.exe", str(attest.JSA_SCRIPT)]
    assert set(kwargs["env"]) & attest._SOLVER_ENV_DENYLIST == set()
    assert kwargs["env"]["PATH"] == "C:/windows"
    assert kwargs["stdin"] is subprocess.PIPE
    assert kwargs["stdout"] is subprocess.PIPE
    assert kwargs["stderr"] is subprocess.PIPE
    assert kwargs["text"] is True
    assert kwargs["encoding"] == "utf-8"
    assert kwargs["errors"] == "replace"
    assert kwargs["close_fds"] is True
    assert popen.proc.stdin_handle().written == ['{"script":"x"}']
    assert popen.proc.stdin_handle().closed is True
    assert popen.proc.wait_timeouts == [attest.SOLVER_TIMEOUT_SEC]
    assert popen.proc.stdout.closed is True
    assert popen.proc.stderr.closed is True


def test_run_solver_tolerates_a_process_without_stdin(monkeypatch) -> None:
    popen = _popen(monkeypatch, _FakeProc(stdout="{}\n", with_stdin=False))
    assert attest._run_solver("C:/node.exe", "{}") == (0, "{}\n", "")
    assert popen.proc.stdin is None


def test_run_solver_tolerates_a_stdin_that_refuses_the_payload(monkeypatch) -> None:
    popen = _popen(monkeypatch, _FakeProc(stdout="{}\n", stdin_write_error=OSError("broken pipe")))
    assert attest._run_solver("C:/node.exe", "{}") == (0, "{}\n", "")
    assert popen.proc.stdin_handle().written == []


def test_run_solver_kills_a_hung_solver_and_reports_the_timeout(monkeypatch) -> None:
    proc = _FakeProc(wait_script=[subprocess.TimeoutExpired("node", attest.SOLVER_TIMEOUT_SEC), 0], poll_result=None)
    _popen(monkeypatch, proc)
    with pytest.raises(attest.AttestationError) as excinfo:
        attest._run_solver("C:/node.exe", "{}")
    assert str(excinfo.value) == f"attestation solver timed out after {attest.SOLVER_TIMEOUT_SEC:g}s"
    assert isinstance(excinfo.value.__cause__, subprocess.TimeoutExpired)
    assert proc.killed is True
    assert proc.wait_timeouts == [attest.SOLVER_TIMEOUT_SEC, None]
    assert proc.stdout.closed is True


def test_run_solver_reports_an_unwaitable_process(monkeypatch) -> None:
    _popen(monkeypatch, _FakeProc(wait_script=[OSError("wait failed")]))
    with pytest.raises(attest.AttestationError) as excinfo:
        attest._run_solver("C:/node.exe", "{}")
    assert str(excinfo.value) == "attestation solver is not runnable: wait failed"


def test_run_solver_tolerates_a_handle_that_cannot_be_closed(monkeypatch) -> None:
    popen = _popen(monkeypatch, _FakeProc(stdout="{}\n", stdout_close_error=OSError("already closed")))
    assert attest._run_solver("C:/node.exe", "{}") == (0, "{}\n", "")
    assert popen.proc.stdout.closed is True
    assert popen.proc.stderr.closed is True


def test_run_solver_keeps_only_the_bounded_tail_of_a_chatty_solver(monkeypatch) -> None:
    monkeypatch.setattr(attest, "SOLVER_OUTPUT_LIMIT", 64)
    answer = '{"ok":true,"result":{"client_hashes":["a"]}}'
    _popen(monkeypatch, _FakeProc(stdout="x" * 4096 + "\n" + answer + "\n", stderr="warning" * 32))
    returncode, stdout, stderr = attest._run_solver("C:/node.exe", "{}")
    assert returncode == 0
    assert len(stdout) == 64
    assert stdout.endswith(answer + "\n")
    assert stdout.startswith("x")
    assert len(stderr) == 64
    assert attest._solver_result(stdout) == {"ok": True, "result": {"client_hashes": ["a"]}}


def test_solver_result_reads_the_last_line() -> None:
    assert attest._solver_result('noise\n{"ok": true}') == {"ok": True}
    assert attest._solver_result('{"ok":\ntrue}') == {"ok": True}


def test_solver_result_rejects_empty_output() -> None:
    with pytest.raises(attest.AttestationError) as excinfo:
        attest._solver_result("  \n ")
    assert str(excinfo.value) == "attestation solver returned no output"


def test_solver_result_reports_malformed_output() -> None:
    with pytest.raises(attest.AttestationError) as excinfo:
        attest._solver_result("z" * 400)
    assert str(excinfo.value) == f"attestation solver returned malformed output: {'z' * attest.SOLVER_ERROR_CHARS}"


def test_evaluate_sync_requires_node(monkeypatch) -> None:
    monkeypatch.setattr(attest, "_node_executable", lambda: None)
    with pytest.raises(attest.AttestationError) as excinfo:
        attest.evaluate_sync("script", "ua")
    assert str(excinfo.value) == "node is required to solve the duckai attestation, set DANYAPI_DUCKAI_NODE or install node"


def test_evaluate_sync_requires_the_shipped_solver(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(attest, "_node_executable", lambda: "C:/node.exe")
    monkeypatch.setattr(attest, "JSA_SCRIPT", tmp_path / "missing.js")
    with pytest.raises(attest.AttestationError) as excinfo:
        attest.evaluate_sync("script", "ua")
    assert str(excinfo.value) == "attestation solver is missing: missing.js"


def test_evaluate_sync_returns_the_solved_attestation(monkeypatch) -> None:
    monkeypatch.setattr(attest, "_node_executable", lambda: "C:/node.exe")
    popen = _popen(monkeypatch, _FakeProc(stdout='{"ok":true,"result":{"client_hashes":["a"],"meta":{}}}\n'))
    attestation = attest.evaluate_sync("(() => 1)", "Mozilla/5.0")
    assert attestation == {"client_hashes": ["a"], "meta": {}}
    args, _kwargs = popen.calls[0]
    assert args == ["C:/node.exe", str(attest.JSA_SCRIPT)]
    assert json.loads(popen.proc.stdin_handle().written[0]) == {"script": "(() => 1)", "user_agent": "Mozilla/5.0"}


def test_evaluate_sync_reports_a_failed_solver(monkeypatch) -> None:
    monkeypatch.setattr(attest, "_node_executable", lambda: "C:/node.exe")
    _popen(monkeypatch, _FakeProc(stderr="ReferenceError: x is not defined\n", returncode=1))
    with pytest.raises(attest.AttestationError) as excinfo:
        attest.evaluate_sync("script", "ua")
    assert str(excinfo.value) == "attestation solver failed: ReferenceError: x is not defined"


def test_evaluate_sync_rejects_a_non_dict_result(monkeypatch) -> None:
    monkeypatch.setattr(attest, "_node_executable", lambda: "C:/node.exe")
    _popen(monkeypatch, _FakeProc(stdout="[1, 2, 3]"))
    with pytest.raises(attest.AttestationError) as excinfo:
        attest.evaluate_sync("script", "ua")
    assert str(excinfo.value) == "attestation solver rejected the script: None"


def test_evaluate_sync_reports_a_rejected_script(monkeypatch) -> None:
    monkeypatch.setattr(attest, "_node_executable", lambda: "C:/node.exe")
    _popen(monkeypatch, _FakeProc(stdout='{"ok": false, "error": "unsupported fragment"}'))
    with pytest.raises(attest.AttestationError) as excinfo:
        attest.evaluate_sync("script", "ua")
    assert str(excinfo.value) == "attestation solver rejected the script: unsupported fragment"


def test_evaluate_sync_requires_an_attestation_payload(monkeypatch) -> None:
    monkeypatch.setattr(attest, "_node_executable", lambda: "C:/node.exe")
    _popen(monkeypatch, _FakeProc(stdout='{"ok": true}'))
    with pytest.raises(attest.AttestationError) as excinfo:
        attest.evaluate_sync("script", "ua")
    assert str(excinfo.value) == "attestation solver returned no attestation"


async def test_evaluate_runs_the_solver_off_the_event_loop(monkeypatch) -> None:
    seen: list[tuple[str, str]] = []

    def _evaluate_sync(script: str, user_agent: str) -> dict:
        seen.append((script, user_agent))
        return {"client_hashes": ["a"]}

    monkeypatch.setattr(attest, "evaluate_sync", _evaluate_sync)
    assert await attest.evaluate("(() => 1)", "Mozilla/5.0") == {"client_hashes": ["a"]}
    assert seen == [("(() => 1)", "Mozilla/5.0")]


async def test_header_for_returns_the_initial_marker_without_a_script() -> None:
    assert attest.INITIAL_JSA == "initial"
    assert await attest.header_for("", "ua") == "initial"
    assert not hasattr(attest, "_INITIAL_JSA")


async def test_header_for_warns_once_about_the_remote_script_exposure(monkeypatch, caplog) -> None:
    attest._note_remote_script_exposure.cache_clear()
    monkeypatch.setattr(attest, "time", _FakeTime(100.0))

    async def _evaluate(script: str, user_agent: str) -> dict:
        return {"client_hashes": ["abc"], "server_hashes": ["srv"], "meta": {"m": 1}, "signals": {"s": 2}}

    monkeypatch.setattr(attest, "evaluate", _evaluate)
    with caplog.at_level(logging.WARNING, logger="danyapi.duckai.attest"):
        first = await attest.header_for(base64.b64encode(b"script").decode(), "ua", origin="https://duck.ai")
        second = await attest.header_for(base64.b64encode(b"script").decode(), "ua")
    decoded = json.loads(base64.b64decode(first))
    assert decoded["server_hashes"] == ["srv"]
    assert decoded["signals"] == {"s": 2}
    assert decoded["meta"]["origin"] == "https://duck.ai"
    assert decoded["meta"]["duration"] == "0"
    assert first == second
    assert caplog.messages == [
        "duckai attestation executes javascript served by the remote duck.ai host in a scrubbed subprocess; it carries no digest to verify against"
    ]
    attest._note_remote_script_exposure.cache_clear()


def test_fraud_signals_encodes_the_wire_key_as_an_empty_list(monkeypatch) -> None:
    monkeypatch.setattr(attest, "time", _FakeTime(1.5))
    assert json.loads(base64.b64decode(attest.fraud_signals())) == {"start": 1500, "events": [], "end": 0}
    assert json.loads(base64.b64decode(attest.fraud_signals(1000.0))) == {"start": 1000, "events": [], "end": 500}
    assert json.loads(base64.b64decode(attest.fraud_signals(2000.0))) == {"start": 2000, "events": [], "end": 0}
