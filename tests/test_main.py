import os
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import danyapi.__main__ as main_mod


def test_main_runs_uvicorn():
    with (
        patch.object(main_mod.uvicorn, "run") as run,
        patch("danyapi.config.settings") as settings,
        patch("danyapi.logging.uvicorn_log_config", return_value={"version": 1}) as log_cfg,
    ):
        settings.host = "1.2.3.4"
        settings.port = 9999
        main_mod.main()
        log_cfg.assert_called_once_with()
        run.assert_called_once_with(
            "danyapi.api.openai:app",
            host="1.2.3.4",
            port=9999,
            log_config={"version": 1},
        )


def _solver_pair(tmp_path):
    source = tmp_path / "pow_solver.c"
    source.write_text("int main(void) { return 0; }\n", encoding="utf-8")
    binary = tmp_path / ("pow_solver.exe" if sys.platform == "win32" else "pow_solver")
    binary.write_bytes(b"stub")
    return source, binary


def _age(path, seconds):
    stamp = time.time() + seconds
    os.utime(path, (stamp, stamp))


def test_build_solver_skips_a_compile_when_the_binary_is_current(tmp_path, monkeypatch):
    import app as app_mod

    source, binary = _solver_pair(tmp_path)
    _age(binary, 60)
    _age(source, 0)
    monkeypatch.setattr(app_mod, "ROOT", tmp_path)
    monkeypatch.setattr(app_mod.shutil, "which", MagicMock(side_effect=AssertionError("no compiler lookup")))
    monkeypatch.setattr(app_mod.subprocess, "run", MagicMock(side_effect=AssertionError("must not compile")))
    assert app_mod.build_solver() is None


def test_build_solver_compiles_when_the_binary_is_older(tmp_path, monkeypatch, capsys):
    import app as app_mod

    source, binary = _solver_pair(tmp_path)
    _age(binary, -60)
    _age(source, 0)
    monkeypatch.setattr(app_mod, "ROOT", tmp_path)
    monkeypatch.setattr(app_mod.shutil, "which", lambda name: f"C:/fake/{name}")
    result = MagicMock(returncode=0, stdout="", stderr="")
    run = MagicMock(return_value=result)
    monkeypatch.setattr(app_mod.subprocess, "run", run)
    assert app_mod.build_solver() is None
    assert run.call_count == 1
    assert f"pow_solver{'.exe' if sys.platform == 'win32' else ''}" in capsys.readouterr().out


def test_build_solver_compiles_when_there_is_no_binary(tmp_path, monkeypatch):
    import app as app_mod

    _source, binary = _solver_pair(tmp_path)
    binary.unlink()
    monkeypatch.setattr(app_mod, "ROOT", tmp_path)
    monkeypatch.setattr(app_mod.shutil, "which", lambda name: f"C:/fake/{name}")
    run = MagicMock(return_value=MagicMock(returncode=0, stdout="", stderr=""))
    monkeypatch.setattr(app_mod.subprocess, "run", run)
    assert app_mod.build_solver() is None
    assert run.call_count == 1


def test_solver_is_current_needs_both_files(tmp_path):
    import app as app_mod

    source, binary = _solver_pair(tmp_path)
    assert app_mod._solver_is_current(binary, source) is True
    assert app_mod._solver_is_current(tmp_path / "missing", source) is False
    binary.unlink()
    assert app_mod._solver_is_current(binary, source) is False
    assert not (tmp_path / "pow_solver.c").is_symlink()
    assert Path(source).is_file()
