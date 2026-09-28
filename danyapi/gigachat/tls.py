from __future__ import annotations

import logging
import os
import ssl
import threading
from pathlib import Path

log = logging.getLogger("danyapi.gigachat")

ROOT_CA_FILENAME = "russian_trusted_root_ca.pem"

ROOT_CA_SHA256 = "d26d2d0231b7c39f92cc738512ba54103519e4405d68b5bd703e9788ca8ecf31"

SYSTEM_CA_CANDIDATES = (
    Path(ROOT_CA_FILENAME).resolve(),
    Path(__file__).resolve().parent / ROOT_CA_FILENAME,
)

_lock = threading.Lock()
_resolved: list[ssl.SSLContext | bool] = [False]


def _certifi_path() -> str | None:
    try:
        import certifi
    except ImportError:
        return None
    path = certifi.where()
    return path if path and Path(path).is_file() else None


def _write_combined(target: Path, parts: list[Path]) -> str:
    chunks: list[str] = []
    for part in parts:
        try:
            text = part.read_text(encoding="ascii", errors="strict")
        except OSError as exc:
            log.warning("gigachat CA read failed for %s: %s", part, exc)
            continue
        chunks.append(text.rstrip())
    if not chunks:
        raise RuntimeError("no usable CA bundle parts for gigachat")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(chunks) + "\n", encoding="ascii")
    return str(target)


def _cache_dir() -> Path | None:
    from ..config import settings

    if settings.cache_dir:
        return Path(settings.cache_dir)
    import tempfile

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

        root = next((p for p in SYSTEM_CA_CANDIDATES if p.is_file()), None)
        if root is None:
            raise RuntimeError(f"gigachat root CA missing, expected {ROOT_CA_FILENAME} next to the gigachat package")

        system = _certifi_path()
        if system is None:
            log.info("gigachat CA: certifi unavailable, using bundled root only")
            context = ssl.create_default_context(cafile=str(root))
            _resolved[0] = context
            return context

        cache = _cache_dir()
        if cache is None:
            context = ssl.create_default_context(cafile=str(root))
            _resolved[0] = context
            return context

        combined = cache / "gigachat-ca-bundle.pem"
        try:
            if not combined.is_file() or combined.stat().st_size < root.stat().st_size:
                _write_combined(combined, [Path(system), root])
            context = ssl.create_default_context(cafile=str(combined))
        except (OSError, ssl.SSLError) as exc:
            log.warning("gigachat CA bundle unusable (%s), using bundled root only", exc)
            context = ssl.create_default_context(cafile=str(root))
        _resolved[0] = context
        return context


def reset_ca_cache() -> None:
    with _lock:
        _resolved[0] = False
