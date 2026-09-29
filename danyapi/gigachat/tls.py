from __future__ import annotations

import hashlib
import logging
import os
import ssl
import tempfile
import threading
from pathlib import Path

log = logging.getLogger("danyapi.gigachat")

ROOT_CA_FILENAME = "russian_trusted_root_ca.pem"

ROOT_CA_SHA256 = "aa800ef345422d6158c6fafe1c06c429dbda21c3df4bb1ccb45a920ec1111399"

_PACKAGE_DIR = Path(__file__).resolve().parent

SYSTEM_CA_CANDIDATES = ((_PACKAGE_DIR / ROOT_CA_FILENAME).resolve(),)

_lock = threading.Lock()
_resolved: list[ssl.SSLContext | bool] = [False]


def _certifi_path() -> str | None:
    try:
        import certifi
    except ImportError:
        return None
    path = certifi.where()
    return path if path and Path(path).is_file() else None


def _is_trusted_root(data: bytes) -> bool:
    try:
        normalised = data.decode("ascii").replace("\r\n", "\n").replace("\r", "\n")
    except UnicodeDecodeError:
        return False
    return hashlib.sha256(normalised.encode("ascii")).hexdigest() == ROOT_CA_SHA256


def _read_root() -> tuple[Path | None, bytes, bool]:
    tampered: tuple[Path, bytes] | None = None
    for candidate in SYSTEM_CA_CANDIDATES:
        if not candidate.is_file():
            continue
        try:
            data = candidate.read_bytes()
        except OSError as exc:
            log.warning("gigachat CA read failed for %s: %s", candidate, exc)
            continue
        if _is_trusted_root(data):
            return candidate, data, True
        log.warning("gigachat root CA digest mismatch for %s, refusing to trust it", candidate)
        tampered = (candidate, data)
    if tampered is not None:
        return tampered[0], tampered[1], False
    return None, b"", False


def _part_text(part: Path, preloaded: bytes | None = None) -> str | None:
    data = preloaded
    if data is None:
        try:
            data = part.read_bytes()
        except OSError as exc:
            log.warning("gigachat CA read failed for %s: %s", part, exc)
            return None
    if part.name == ROOT_CA_FILENAME and not _is_trusted_root(data):
        log.warning("gigachat root CA digest mismatch for %s, refusing to trust it", part)
        return None
    try:
        text = data.decode("ascii")
    except UnicodeDecodeError as exc:
        log.warning("gigachat CA is not ascii for %s: %s", part, exc)
        return None
    return text.rstrip()


def _combined_payload(parts: list[Path], preloaded: dict[str, bytes] | None = None) -> bytes:
    known = preloaded or {}
    chunks: list[str] = []
    for part in parts:
        text = _part_text(part, known.get(str(part)))
        if text is not None:
            chunks.append(text)
    if not chunks:
        raise RuntimeError("no usable CA bundle parts for gigachat")
    return ("\n".join(chunks) + "\n").encode("ascii")


def _atomic_write(target: Path, payload: bytes) -> None:
    handle, name = tempfile.mkstemp(dir=str(target.parent), prefix=target.name + ".", suffix=".tmp")
    temp = Path(name)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, target)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise


def _write_combined(target: Path, parts: list[Path], expected: bytes | None = None) -> str:
    payload = _combined_payload(parts) if expected is None else expected
    target.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(target, payload)
    return str(target)


def _cache_dir() -> Path:
    from ..config import settings

    if settings.cache_dir:
        return Path(settings.cache_dir)
    return Path(tempfile.gettempdir()) / "danyapi"


def resolve_ca() -> ssl.SSLContext:
    with _lock:
        cached = _resolved[0]
        if cached is not False:
            return cached  # type: ignore[return-value]

        override = os.environ.get("DANYAPI_GIGACHAT_CA_FILE", "").strip()
        if override:
            if not Path(override).is_file():
                raise RuntimeError(f"DANYAPI_GIGACHAT_CA_FILE does not exist: {override}")
            log.info("gigachat CA override in use: %s", override)
            context = ssl.create_default_context(cafile=override)
            _resolved[0] = context
            return context

        root, root_data, verified = _read_root()
        if root is None:
            raise RuntimeError(f"gigachat root CA missing, expected {ROOT_CA_FILENAME} next to the gigachat package")

        system = _certifi_path()
        if not verified:
            log.warning("gigachat bundled root CA rejected, falling back to the system trust store")
            context = ssl.create_default_context(cafile=system) if system else ssl.create_default_context()
            _resolved[0] = context
            return context

        if system is None:
            log.info("gigachat CA: certifi unavailable, using bundled root only")
            context = ssl.create_default_context(cafile=str(root))
            _resolved[0] = context
            return context

        combined = _cache_dir() / "gigachat-ca-bundle.pem"
        try:
            parts = [Path(system), root]
            expected = _combined_payload(parts, {str(root): root_data})
            if not combined.is_file() or combined.read_bytes() != expected:
                _write_combined(combined, parts, expected)
            context = ssl.create_default_context(cafile=str(combined))
        except (OSError, ssl.SSLError) as exc:
            log.warning("gigachat CA bundle unusable (%s), using bundled root only", exc)
            context = ssl.create_default_context(cafile=str(root))
        _resolved[0] = context
        return context


def reset_ca_cache() -> None:
    with _lock:
        _resolved[0] = False
