from __future__ import annotations

from ..config import settings as _settings
from .manager import ByokManager

_byok_mgr: ByokManager | None = None


def byok_enabled() -> bool:
    return _settings.byok_mode


def set_manager(mgr: ByokManager | None) -> None:
    global _byok_mgr
    _byok_mgr = mgr


def get_manager() -> ByokManager | None:
    return _byok_mgr


def reset_manager() -> None:
    global _byok_mgr
    _byok_mgr = None
