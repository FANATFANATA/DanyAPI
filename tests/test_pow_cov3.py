import json
import os
import subprocess  # nosec B404
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from danyapi import pow as pow_mod
from danyapi.pow import _find_native_solver, _native_solver_names, _parse_number, _run_solver, deepseek_hash_v1_hex, solve_python


def _proc(stdout="", returncode=0, stderr=""):
    return MagicMock(stdout=stdout, returncode=returncode, stderr=stderr)


def test_parse_number_rejects_booleans():
    assert _parse_number(True) is None
    assert _parse_number(False) is None


def test_parse_number_accepts_numbers():
    assert _parse_number(7) == 7
    assert _parse_number(1.5) == 1.5
    assert _parse_number(float("inf")) is None
    assert _parse_number(float("nan")) is None


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("12", 12), ("  12  ", 12), ("-3", -3), ("1.5", 1.5), (" 2.5 ", 2.5), ("nan", None), ("inf", None), ("abc", None), ("", None), ("   ", None)],
)
def test_parse_number_reads_strings(raw, expected):
    assert _parse_number(raw) == expected


@pytest.mark.parametrize("value", [None, [1], {"a": 1}, object()])
def test_parse_number_rejects_other_types(value):
    assert _parse_number(value) is None


def test_native_solver_names_follow_the_platform():
    expected = ("pow_solver.exe", "pow_solver") if os.name == "nt" else ("pow_solver", "pow_solver.exe")
    assert _native_solver_names() == expected
    other = "posix" if os.name == "nt" else "nt"
    with patch.object(pow_mod.os, "name", other):
        assert _native_solver_names() == tuple(reversed(expected))


def test_find_native_solver_picks_the_existing_name(tmp_path):
    with patch.object(pow_mod, "_SOLVER_DIR", tmp_path):
        assert _find_native_solver() is None
        (tmp_path / "pow_solver.exe").write_bytes(b"")
        assert _find_native_solver() == tmp_path / "pow_solver.exe"
    (tmp_path / "pow_solver.exe").unlink()
    (tmp_path / "pow_solver").write_bytes(b"")
    with patch.object(pow_mod, "_SOLVER_DIR", tmp_path):
        assert _find_native_solver() == tmp_path / "pow_solver"


def test_find_native_solver_prefers_the_extensionless_name_off_windows(tmp_path):
    (tmp_path / "pow_solver").write_bytes(b"")
    with patch.object(pow_mod, "_SOLVER_DIR", tmp_path), patch.object(pow_mod.os, "name", "posix"):
        assert _find_native_solver() == tmp_path / "pow_solver"
    (tmp_path / "pow_solver.exe").write_bytes(b"")
    with patch.object(pow_mod, "_SOLVER_DIR", tmp_path), patch.object(pow_mod.os, "name", "posix"):
        assert _find_native_solver() == tmp_path / "pow_solver"


def test_solve_python_fast_path_stops_at_the_budget():
    assert solve_python("00" * 32, "s", 1, 100000, timeout=0.0) is None


def test_solve_python_fast_path_finds_the_answer():
    prefix = b"s_1_"
    assert solve_python(deepseek_hash_v1_hex(prefix + b"7"), "s", 1, 1000) == 7


def test_solve_python_slow_path_finds_the_answer():
    salt = "S" * 140
    prefix = f"{salt}_1_".encode()
    assert solve_python(deepseek_hash_v1_hex(prefix + b"3"), salt, 1, 100) == 3


def test_solve_python_slow_path_exhausts_every_candidate():
    assert solve_python("00" * 32, "S" * 140, 1, 100) is None


def test_run_solver_maps_a_subprocess_timeout():
    with patch("danyapi.pow.subprocess.run", side_effect=subprocess.TimeoutExpired(cmd=["pow_solver.exe"], timeout=60.0)):
        with pytest.raises(RuntimeError) as excinfo:
            _run_solver(Path("pow_solver.exe"), "c", "s", 1, 10, 60.0)
    assert str(excinfo.value) == "pow_solver.exe timed out after 60s"
    assert excinfo.value.__cause__ is not None


def test_run_solver_maps_a_missing_executable():
    with patch("danyapi.pow.subprocess.run", side_effect=OSError("no such file or directory")):
        with pytest.raises(RuntimeError) as excinfo:
            _run_solver(Path("pow_solver.exe"), "c", "s", 1, 10)
    assert str(excinfo.value) == "pow_solver.exe is not executable: no such file or directory"


def test_run_solver_rejects_unparsable_output():
    with patch("danyapi.pow.subprocess.run", return_value=_proc("not json")):
        with pytest.raises(RuntimeError) as excinfo:
            _run_solver(Path("pow_solver.exe"), "c", "s", 1, 10)
    assert str(excinfo.value) == "pow_solver.exe returned malformed output: not json"


def test_run_solver_rejects_non_object_output():
    with patch("danyapi.pow.subprocess.run", return_value=_proc("[1, 2]")):
        with pytest.raises(RuntimeError) as excinfo:
            _run_solver(Path("pow_solver.exe"), "c", "s", 1, 10)
    assert str(excinfo.value) == "pow_solver.exe returned malformed output: [1, 2]"


def test_run_solver_forwards_the_timeout_and_the_payload():
    with patch("danyapi.pow.subprocess.run", return_value=_proc('{"answer": 3}')) as run:
        assert _run_solver(Path("pow_solver.exe"), "ab", "salt", 1700000000000, 12, 12.5) == 3
    assert run.call_args.args[0] == [str(Path("pow_solver.exe"))]
    assert run.call_args.kwargs["timeout"] == 12.5
    assert run.call_args.kwargs["check"] is False
    assert json.loads(run.call_args.kwargs["input"]) == {"challenge": "ab", "salt": "salt", "expire_at": "1700000000000", "difficulty": 12}


def test_run_solver_runs_a_javascript_solver_through_node():
    script = Path("pow_solver.js")
    with patch("danyapi.pow.subprocess.run", return_value=_proc('{"answer": 1}')) as run:
        assert _run_solver(script, "c", "s", 1, 2) == 1
    assert run.call_args.args[0] == ["node", str(script)]


def test_run_solver_scrubs_the_environment_it_hands_over(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_TOKENS", "secret")
    monkeypatch.setenv("DANYAPI_HOST", "127.0.0.1")
    with patch("danyapi.pow.subprocess.run", return_value=_proc('{"answer": 1}')) as run:
        _run_solver(Path("pow_solver.exe"), "c", "s", 1, 2)
    env = run.call_args.kwargs["env"]
    assert "DEEPSEEK_TOKENS" not in env
    assert env["DANYAPI_HOST"] == "127.0.0.1"
