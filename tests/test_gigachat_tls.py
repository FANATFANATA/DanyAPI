from __future__ import annotations

import hashlib
import logging
import os
import re
import ssl
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import certifi
import pytest

from danyapi.config import settings
from danyapi.gigachat import tls

BUNDLED_ROOT = tls.SYSTEM_CA_CANDIDATES[0]
BUNDLED_BYTES = BUNDLED_ROOT.read_bytes()
SYSTEM_PEM = certifi.where()
ROOT_SUBJECT = (
    (("countryName", "RU"),),
    (("organizationName", "The Ministry of Digital Development and Communications"),),
    (("commonName", "Russian Trusted Root CA"),),
)
BUNDLE_NAME = "gigachat-ca-bundle.pem"
ROOT_ONLY = (BUNDLED_BYTES.decode("ascii").rstrip() + "\n").encode("ascii")


def _expected_bundle() -> bytes:
    system = Path(SYSTEM_PEM).read_bytes().decode("ascii").rstrip()
    return (system + "\n" + BUNDLED_BYTES.decode("ascii").rstrip() + "\n").encode("ascii")


def _subjects(context: ssl.SSLContext) -> list[Any]:
    return [cert["subject"] for cert in context.get_ca_certs()]


def _tampered(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / tls.ROOT_CA_FILENAME
    target.write_bytes(BUNDLED_BYTES.replace(b"MIIF", b"MIIE", 1))
    return target


def _unreadable(path: Path) -> bytes:
    raise PermissionError(f"cannot read {path}")


def _context_calls(monkeypatch: pytest.MonkeyPatch) -> list[str | None]:
    calls: list[str | None] = []
    original = ssl.create_default_context

    def _create(*args: Any, **kwargs: Any) -> ssl.SSLContext:
        calls.append(kwargs.get("cafile"))
        return original(*args, **kwargs)

    monkeypatch.setattr(tls.ssl, "create_default_context", _create)
    return calls


def _replace_calls(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str]]:
    calls: list[tuple[str, str]] = []
    original = tls.os.replace

    def _replace(src: str, dst: str) -> None:
        calls.append((Path(src).name, Path(dst).name))
        original(src, dst)

    monkeypatch.setattr(tls.os, "replace", _replace)
    return calls


@pytest.fixture(autouse=True)
def _isolated_ca(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.delenv("DANYAPI_GIGACHAT_CA_FILE", raising=False)
    monkeypatch.setattr(tls, "_resolved", [False])
    monkeypatch.setattr(settings, "cache_dir", "")
    yield
    monkeypatch.setattr(tls, "_resolved", [False])


def test_candidates_are_absolute_and_point_at_the_shipped_root(monkeypatch, tmp_path):
    decoy = tmp_path / tls.ROOT_CA_FILENAME
    decoy.write_text("decoy", encoding="ascii")
    monkeypatch.chdir(tmp_path)

    assert len(tls.SYSTEM_CA_CANDIDATES) == 1
    candidate = tls.SYSTEM_CA_CANDIDATES[0]
    assert candidate.is_absolute()
    assert candidate == tls._PACKAGE_DIR / tls.ROOT_CA_FILENAME
    assert candidate.read_bytes() == BUNDLED_BYTES
    assert tls._is_trusted_root(BUNDLED_BYTES)
    crlf = BUNDLED_BYTES.decode("ascii").replace("\r\n", "\n").replace("\n", "\r\n").encode("ascii")
    assert tls._is_trusted_root(crlf)
    assert hashlib.sha256(BUNDLED_BYTES.decode("ascii").replace("\r\n", "\n").encode("ascii")).hexdigest() == tls.ROOT_CA_SHA256
    assert decoy.read_text(encoding="ascii") == "decoy"


def test_resolve_ca_ignores_a_root_ca_dropped_in_the_working_directory(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    decoy_bytes = b"-----BEGIN CERTIFICATE-----\nZmFrZQ==\n-----END CERTIFICATE-----\n"
    decoy = tmp_path / tls.ROOT_CA_FILENAME
    decoy.write_bytes(decoy_bytes)
    monkeypatch.setattr(settings, "cache_dir", str(tmp_path / "cache"))

    context = tls.resolve_ca()

    assert tls.resolve_ca() is context
    bundle = (tmp_path / "cache" / BUNDLE_NAME).read_bytes()
    assert bundle == _expected_bundle()
    assert b"ZmFrZQ==" not in bundle
    assert decoy.read_bytes() == decoy_bytes


def test_reset_ca_cache_forces_a_fresh_resolve(monkeypatch):
    calls = _context_calls(monkeypatch)

    first = tls.resolve_ca()
    assert tls.resolve_ca() is first
    assert len(calls) == 1

    tls.reset_ca_cache()

    second = tls.resolve_ca()
    assert second is not first
    assert len(calls) == 2
    assert calls[1] == calls[0]
    assert tls.resolve_ca() is second


def test_read_root_accepts_the_shipped_root():
    assert tls._read_root() == (BUNDLED_ROOT, BUNDLED_BYTES, True)


def test_read_root_reports_a_tampered_root_as_unverified(monkeypatch, caplog, tmp_path):
    tampered = _tampered(tmp_path)
    monkeypatch.setattr(tls, "SYSTEM_CA_CANDIDATES", (tampered,))

    with caplog.at_level(logging.WARNING, logger="danyapi.gigachat"):
        root, data, verified = tls._read_root()

    assert root == tampered
    assert data == tampered.read_bytes()
    assert verified is False
    assert f"gigachat root CA digest mismatch for {tampered}" in caplog.text


def test_read_root_skips_missing_and_unreadable_candidates(monkeypatch, caplog, tmp_path):
    missing = tmp_path / "absent.pem"
    locked = tmp_path / "locked.pem"
    locked.write_bytes(b"not readable\n")
    monkeypatch.setattr(tls, "SYSTEM_CA_CANDIDATES", (missing, BUNDLED_ROOT))
    assert tls._read_root() == (BUNDLED_ROOT, BUNDLED_BYTES, True)

    original = Path.read_bytes
    monkeypatch.setattr(Path, "read_bytes", lambda self: _unreadable(self) if self == locked else original(self))
    monkeypatch.setattr(tls, "SYSTEM_CA_CANDIDATES", (missing, locked, BUNDLED_ROOT))

    with caplog.at_level(logging.WARNING, logger="danyapi.gigachat"):
        assert tls._read_root() == (BUNDLED_ROOT, BUNDLED_BYTES, True)
    assert f"gigachat CA read failed for {locked}" in caplog.text

    monkeypatch.setattr(tls, "SYSTEM_CA_CANDIDATES", (missing, locked))
    with caplog.at_level(logging.WARNING, logger="danyapi.gigachat"):
        assert tls._read_root() == (None, b"", False)


def test_certifi_path_is_none_when_the_bundle_is_absent(monkeypatch, tmp_path):
    import certifi as certifi_module

    monkeypatch.setattr(certifi_module, "where", lambda: str(tmp_path / "no-such-bundle.pem"))
    assert tls._certifi_path() is None
    monkeypatch.setattr(certifi_module, "where", lambda: "")
    assert tls._certifi_path() is None
    monkeypatch.setattr(certifi_module, "where", lambda: SYSTEM_PEM)
    assert tls._certifi_path() == SYSTEM_PEM


def test_certifi_path_is_none_when_certifi_is_not_installed(monkeypatch):
    monkeypatch.setitem(sys.modules, "certifi", None)
    assert tls._certifi_path() is None


def test_part_text_preloaded_bytes_are_served_without_touching_the_disk(monkeypatch):
    seen: list[str] = []
    original = Path.read_bytes

    def _counting(self: Path) -> bytes:
        seen.append(self.name)
        return original(self)

    monkeypatch.setattr(Path, "read_bytes", _counting)
    assert tls._part_text(BUNDLED_ROOT) == BUNDLED_BYTES.decode("ascii").rstrip()
    assert seen == [tls.ROOT_CA_FILENAME]

    seen.clear()
    assert tls._part_text(BUNDLED_ROOT, BUNDLED_BYTES) == BUNDLED_BYTES.decode("ascii").rstrip()
    assert seen == []


def test_part_text_rejects_a_tampered_file_named_like_the_root(caplog, tmp_path):
    tampered = _tampered(tmp_path)
    payload = tampered.read_bytes()

    with caplog.at_level(logging.WARNING, logger="danyapi.gigachat"):
        assert tls._part_text(tampered) is None
        assert tls._part_text(tampered, payload) is None

    assert f"gigachat root CA digest mismatch for {tampered}" in caplog.text


def test_part_text_rejects_non_ascii_and_unreadable_parts(tmp_path, caplog):
    binary = tmp_path / "extra.pem"
    binary.write_bytes(b"-----BEGIN CERTIFICATE-----\n\xff\xfe\n-----END CERTIFICATE-----\n")
    missing = tmp_path / "gone.pem"
    padded = tmp_path / "padded.pem"
    padded.write_bytes(b"  -----BEGIN CERTIFICATE-----\n  body\n  \n")
    binary_root = tmp_path / tls.ROOT_CA_FILENAME
    binary_root.write_bytes(b"-----BEGIN CERTIFICATE-----\n\xff\xfe\n-----END CERTIFICATE-----\n")

    with caplog.at_level(logging.WARNING, logger="danyapi.gigachat"):
        assert tls._part_text(binary) is None
        assert tls._part_text(missing) is None
        assert tls._is_trusted_root(binary_root.read_bytes()) is False

    assert f"gigachat CA is not ascii for {binary}" in caplog.text
    assert f"gigachat CA read failed for {missing}" in caplog.text
    assert tls._part_text(padded) == "  -----BEGIN CERTIFICATE-----\n  body"


def test_combined_payload_keeps_the_root_and_drops_the_offending_part(tmp_path):
    binary = tmp_path / "extra.pem"
    binary.write_bytes(b"\xff\xfe\x00\x01")

    assert tls._combined_payload([binary, BUNDLED_ROOT]) == ROOT_ONLY
    assert tls._combined_payload([BUNDLED_ROOT], {str(BUNDLED_ROOT): BUNDLED_BYTES}) == ROOT_ONLY


def test_combined_payload_without_a_usable_part_raises(tmp_path):
    with pytest.raises(RuntimeError, match=r"^no usable CA bundle parts for gigachat$"):
        tls._combined_payload([tmp_path / "gone.pem"])


def test_write_combined_creates_missing_parents_and_replaces_a_stale_file(tmp_path):
    target = tmp_path / "nested" / "deeper" / BUNDLE_NAME
    target.parent.mkdir(parents=True)
    target.write_bytes(b"stale")

    written = tls._write_combined(target, [BUNDLED_ROOT])

    assert written == str(target)
    assert target.read_bytes() == ROOT_ONLY
    assert list(target.parent.glob(f"{BUNDLE_NAME}.*.tmp")) == []


def test_atomic_write_replaces_a_symlink_without_following_it(tmp_path):
    elsewhere = tmp_path / "untouched.pem"
    elsewhere.write_bytes(b"do not touch\n")
    target = tmp_path / BUNDLE_NAME
    os.symlink(elsewhere, target)
    assert target.is_symlink()

    tls._write_combined(target, [BUNDLED_ROOT])

    assert not target.is_symlink()
    assert elsewhere.read_bytes() == b"do not touch\n"
    assert target.read_bytes() == ROOT_ONLY
    assert list(tmp_path.glob(f"{BUNDLE_NAME}.*.tmp")) == []


def test_atomic_write_rolls_back_the_temp_file_when_replace_fails(monkeypatch, tmp_path):
    def _refuse(src: str, dst: str) -> None:
        raise OSError(f"replace refused for {dst}")

    monkeypatch.setattr(tls.os, "replace", _refuse)
    target = tmp_path / BUNDLE_NAME

    with pytest.raises(OSError, match=r"^replace refused for .*gigachat-ca-bundle\.pem$"):
        tls._write_combined(target, [BUNDLED_ROOT])

    assert not target.exists()
    assert list(tmp_path.glob(f"{BUNDLE_NAME}.*.tmp")) == []


def test_cache_dir_follows_the_configured_directory(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "cache_dir", str(tmp_path / "custom"))
    assert tls._cache_dir() == tmp_path / "custom"

    monkeypatch.setattr(settings, "cache_dir", "")
    assert tls._cache_dir() == Path(tempfile.gettempdir()) / "danyapi"


def test_resolve_ca_honours_the_environment_override_on_a_cache_miss(monkeypatch, tmp_path, caplog):
    override = tmp_path / "override.pem"
    override.write_bytes(BUNDLED_BYTES)
    calls = _context_calls(monkeypatch)
    monkeypatch.setenv("DANYAPI_GIGACHAT_CA_FILE", f"  {override}  ")

    with caplog.at_level(logging.INFO, logger="danyapi.gigachat"):
        context = tls.resolve_ca()

    assert calls == [str(override)]
    assert f"gigachat CA override in use: {override}" in caplog.text
    assert _subjects(context) == [ROOT_SUBJECT]
    assert tls.resolve_ca() is context
    assert calls == [str(override)]


def test_resolve_ca_rejects_a_missing_override(monkeypatch, tmp_path):
    missing = tmp_path / "absent.pem"
    monkeypatch.setenv("DANYAPI_GIGACHAT_CA_FILE", str(missing))

    with pytest.raises(RuntimeError, match=f"^{re.escape(f'DANYAPI_GIGACHAT_CA_FILE does not exist: {missing}')}$"):
        tls.resolve_ca()


def test_resolve_ca_ignores_the_override_once_the_cache_is_warm(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "cache_dir", str(tmp_path / "cache"))
    warm = tls.resolve_ca()

    monkeypatch.setenv("DANYAPI_GIGACHAT_CA_FILE", str(tmp_path / "absent.pem"))

    assert tls.resolve_ca() is warm


def test_resolve_ca_raises_when_the_bundled_root_is_absent(monkeypatch, tmp_path):
    monkeypatch.setattr(tls, "SYSTEM_CA_CANDIDATES", (tmp_path / "absent.pem",))

    with pytest.raises(RuntimeError, match=f"^gigachat root CA missing, expected {tls.ROOT_CA_FILENAME} next to the gigachat package$"):
        tls.resolve_ca()


def test_resolve_ca_falls_back_to_the_system_store_when_the_root_is_tampered(monkeypatch, tmp_path, caplog):
    system_subjects = _subjects(ssl.create_default_context(cafile=SYSTEM_PEM))
    _tampered(tmp_path)
    monkeypatch.setattr(tls, "SYSTEM_CA_CANDIDATES", (tmp_path / tls.ROOT_CA_FILENAME,))
    monkeypatch.setattr(settings, "cache_dir", str(tmp_path / "cache"))
    calls = _context_calls(monkeypatch)

    with caplog.at_level(logging.WARNING, logger="danyapi.gigachat"):
        context = tls.resolve_ca()

    assert calls == [SYSTEM_PEM]
    assert "gigachat bundled root CA rejected, falling back to the system trust store" in caplog.text
    assert not (tmp_path / "cache" / BUNDLE_NAME).exists()
    assert _subjects(context) == system_subjects
    assert ROOT_SUBJECT not in _subjects(context)


def test_resolve_ca_uses_the_bundled_root_only_when_certifi_is_absent(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr(tls, "_certifi_path", lambda: None)
    monkeypatch.setattr(settings, "cache_dir", str(tmp_path / "cache"))
    calls = _context_calls(monkeypatch)

    with caplog.at_level(logging.INFO, logger="danyapi.gigachat"):
        context = tls.resolve_ca()

    assert calls == [str(BUNDLED_ROOT)]
    assert _subjects(context) == [ROOT_SUBJECT]
    assert not (tmp_path / "cache" / BUNDLE_NAME).exists()
    assert "gigachat CA: certifi unavailable, using bundled root only" in caplog.text


def test_resolve_ca_builds_the_combined_bundle_from_the_preloaded_root(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "cache_dir", str(tmp_path / "cache"))
    seen: list[str] = []
    original = Path.read_bytes

    def _counting(self: Path) -> bytes:
        seen.append(self.name)
        return original(self)

    monkeypatch.setattr(Path, "read_bytes", _counting)
    context = tls.resolve_ca()

    assert seen == [tls.ROOT_CA_FILENAME, Path(SYSTEM_PEM).name]
    assert (tmp_path / "cache" / BUNDLE_NAME).read_bytes() == _expected_bundle()
    assert ROOT_SUBJECT in _subjects(context)


def test_resolve_ca_rewrites_a_bundle_of_the_same_size_with_other_content(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "cache_dir", str(tmp_path / "cache"))
    expected = _expected_bundle()
    bundle = tmp_path / "cache" / BUNDLE_NAME
    bundle.parent.mkdir(parents=True)
    bundle.write_bytes(b"#" * len(expected))
    replaces = _replace_calls(monkeypatch)

    tls.resolve_ca()

    assert len(replaces) == 1
    assert replaces[0][1] == BUNDLE_NAME
    assert replaces[0][0].startswith(f"{BUNDLE_NAME}.")
    assert replaces[0][0].endswith(".tmp")
    assert bundle.read_bytes() == expected
    assert list(bundle.parent.glob(f"{BUNDLE_NAME}.*.tmp")) == []


def test_resolve_ca_keeps_an_identical_bundle_untouched(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "cache_dir", str(tmp_path / "cache"))
    expected = _expected_bundle()
    bundle = tmp_path / "cache" / BUNDLE_NAME
    bundle.parent.mkdir(parents=True)
    bundle.write_bytes(expected)
    before = bundle.stat().st_mtime_ns
    replaces = _replace_calls(monkeypatch)

    tls.resolve_ca()

    assert replaces == []
    assert bundle.stat().st_mtime_ns == before
    assert bundle.read_bytes() == expected


def test_resolve_ca_falls_back_to_the_root_when_the_bundle_cannot_be_replaced(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr(settings, "cache_dir", str(tmp_path / "cache"))
    (tmp_path / "cache" / BUNDLE_NAME).mkdir(parents=True)
    calls = _context_calls(monkeypatch)

    with caplog.at_level(logging.WARNING, logger="danyapi.gigachat"):
        context = tls.resolve_ca()

    assert calls == [str(BUNDLED_ROOT)]
    assert _subjects(context) == [ROOT_SUBJECT]
    assert "gigachat CA bundle unusable" in caplog.text
    assert list((tmp_path / "cache").glob(f"{BUNDLE_NAME}.*.tmp")) == []
