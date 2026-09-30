import logging
import os
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from danyapi import config as config_mod
from danyapi.config import MAX_PORT, MIN_PORT, Settings, audit_credential_env_names


def _settings_for(env):
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("os.environ", dict(env))
        return Settings()


def _warnings(caplog, name="danyapi.config"):
    return [record for record in caplog.records if record.name == name and record.levelno == logging.WARNING]


def _texts(caplog, name="danyapi.config"):
    return [record.getMessage() for record in _warnings(caplog, name)]


class _SpyDotenv(types.ModuleType):
    def __init__(self):
        super().__init__("dotenv")
        self.calls = []

    def load_dotenv(self, **kwargs):
        self.calls.append(kwargs)
        return True


def _run_module_body(monkeypatch, name, dotenv_module) -> dict[str, Any]:
    source = Path(config_mod.__file__).read_text(encoding="utf-8")
    namespace: dict[str, Any] = {"__name__": name, "__file__": config_mod.__file__, "__package__": "danyapi"}
    monkeypatch.setattr("os.environ", {})
    monkeypatch.setitem(sys.modules, "dotenv", dotenv_module)
    exec(compile(source, config_mod.__file__, "exec"), namespace)
    return namespace


def test_bad_port_warns_and_falls_back(caplog):
    with caplog.at_level(logging.WARNING):
        settings = _settings_for({"DANYAPI_PORT": "eighty"})
    assert settings.port == 8000
    assert _texts(caplog) == ["DANYAPI_PORT='eighty' is not a valid integer, using 8000"]


def test_port_zero_clamps_to_min_port(caplog):
    with caplog.at_level(logging.WARNING):
        settings = _settings_for({"DANYAPI_PORT": "0"})
    assert MIN_PORT == 1
    assert settings.port == 1
    assert _texts(caplog) == ["DANYAPI_PORT=0 is below the minimum 1, clamped"]


def test_port_above_maximum_clamps(caplog):
    with caplog.at_level(logging.WARNING):
        settings = _settings_for({"DANYAPI_PORT": "70000"})
    assert settings.port == MAX_PORT == 65535
    assert _texts(caplog) == ["DANYAPI_PORT=70000 is above the maximum 65535, clamped"]


def test_bad_timeout_warns_and_falls_back(caplog):
    with caplog.at_level(logging.WARNING):
        settings = _settings_for({"DANYAPI_TIMEOUT": "soon"})
    assert settings.timeout == 60.0
    assert _texts(caplog) == ["DANYAPI_TIMEOUT='soon' is not a valid number, using 60.0"]


def test_non_finite_ttl_warns_and_uses_the_default(caplog):
    with caplog.at_level(logging.WARNING):
        settings = _settings_for({"DANYAPI_SESSION_TTL_SECONDS": "inf"})
    assert settings.session_ttl == 3600.0
    assert _texts(caplog) == ["DANYAPI_SESSION_TTL_SECONDS=inf is not finite, using 3600.0"]


def test_zero_ttl_never_expires_and_negative_clamps(caplog):
    with caplog.at_level(logging.WARNING):
        zero = _settings_for({"DANYAPI_SESSION_TTL_SECONDS": "0"})
        negative = _settings_for({"DANYAPI_SESSION_TTL_SECONDS": "-30"})
    assert zero.session_ttl == 0.0
    assert negative.session_ttl == 0.0
    assert _texts(caplog) == ["DANYAPI_SESSION_TTL_SECONDS=-30.0 is below the minimum 0.0, clamped"]


def test_acquire_timeout_bad_value_warns(caplog):
    with caplog.at_level(logging.WARNING):
        settings = _settings_for({"DANYAPI_ACQUIRE_TIMEOUT": "later"})
    assert settings.acquire_timeout is None
    assert _texts(caplog) == ["DANYAPI_ACQUIRE_TIMEOUT='later' is not a valid number, ignoring it"]


def test_split_env_list_escaped_comma():
    assert config_mod._split_env_list("a\\,b") == ["a,b"]
    assert config_mod._split_env_list("one\\,two,three") == ["one,two", "three"]


def test_split_env_list_escaped_backslash():
    assert config_mod._split_env_list("a\\\\b") == ["a\\\\b"]


def test_split_env_list_trailing_backslash():
    assert config_mod._split_env_list("a\\") == ["a\\"]


def test_split_env_list_drops_empty_items():
    assert config_mod._split_env_list("") == []
    assert config_mod._split_env_list(" , , ") == []
    assert config_mod._split_env_list("a,,b") == ["a", "b"]


def test_credential_with_a_comma_is_configurable():
    settings = _settings_for({"GIGACHAT_KEYS": "key\\,with,comma"})
    assert settings.gigachat_keys == ["key,with", "comma"]


def test_audit_is_silent_for_the_current_source(caplog):
    read = set(config_mod._ENV_NAME_RE.findall(Path(config_mod.__file__).read_text(encoding="utf-8")))
    assert read
    assert read <= set(config_mod.CREDENTIAL_ENV_NAMES) | config_mod._NON_CREDENTIAL_ENV_NAMES
    with caplog.at_level(logging.WARNING):
        audit_credential_env_names()
    assert _texts(caplog) == []


def test_audit_warns_about_an_unclassified_env_name(caplog, monkeypatch):
    class _FakePath:
        def __init__(self, *args, **kwargs):
            self.source = '_env_str("DANYAPI_MADE_UP_NAME", "")\n_env_list("DEEPSEEK_TOKENS")\n_env_int("DANYAPI_USAGE_MAX_RECORDS", 1)\n'

        def resolve(self):
            return self

        def read_text(self, encoding=None):
            return self.source

    monkeypatch.setattr(config_mod, "Path", _FakePath)
    with caplog.at_level(logging.WARNING):
        audit_credential_env_names()
    assert _texts(caplog) == ["config reads env names that are neither credential nor listed as non credential: DANYAPI_MADE_UP_NAME"]


def test_audit_warns_when_the_source_cannot_be_read(caplog, monkeypatch):
    def _boom(self, *args, **kwargs):
        raise OSError("permission denied")

    monkeypatch.setattr(Path, "read_text", _boom)
    with caplog.at_level(logging.WARNING):
        assert audit_credential_env_names() is None
    assert _texts(caplog) == ["cannot audit credential env names: permission denied"]


def test_env_path_is_derived_from_the_package_location():
    assert config_mod._ENV_PATH == Path(config_mod.__file__).resolve().parents[1] / ".env"
    assert config_mod._ENV_PATH.name == ".env"
    assert (config_mod._ENV_PATH.parent / "danyapi" / "config.py").is_file()


def test_module_import_loads_the_package_env_without_override(monkeypatch):
    spy = _SpyDotenv()
    namespace = _run_module_body(monkeypatch, "danyapi.config_probe", spy)
    assert spy.calls == [{"dotenv_path": config_mod._ENV_PATH, "override": False}]
    assert namespace["_ENV_PATH"] == config_mod._ENV_PATH
    assert namespace["settings"].port == 8000
    assert namespace["audit_credential_env_names"]() is None


def test_module_import_without_python_dotenv_warns_and_still_builds_settings(monkeypatch, caplog):
    with caplog.at_level(logging.WARNING):
        namespace = _run_module_body(monkeypatch, "danyapi.config_no_dotenv", None)
    assert namespace["load_dotenv"] is namespace["_noop_load_dotenv"]
    assert namespace["settings"].timeout == 60.0
    assert _texts(caplog, "danyapi.config_no_dotenv") == [f"python-dotenv is not installed, skipping {config_mod._ENV_PATH}"]


def test_load_dotenv_does_not_override_an_exported_variable(tmp_path, monkeypatch):
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text("DANYAPI_HOST=10.0.0.1\nDANYAPI_PORT=9999\n", encoding="utf-8")
    monkeypatch.setenv("DANYAPI_HOST", "0.0.0.0")
    monkeypatch.delenv("DANYAPI_PORT", raising=False)
    assert config_mod.load_dotenv(dotenv_path=dotenv_path, override=False) is True
    assert os.environ["DANYAPI_HOST"] == "0.0.0.0"
    assert os.environ["DANYAPI_PORT"] == "9999"
