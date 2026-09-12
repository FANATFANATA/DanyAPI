from __future__ import annotations

import logging

from fastapi import Cookie, HTTPException, Request

log = logging.getLogger("danyapi.byok.middleware")

BYOK_SESSION_COOKIE = "byok_session"


class ByokAuth:
    def __init__(self, user_id: str | None) -> None:
        self.user_id = user_id
        self.authenticated: bool = user_id is not None


async def byok_dependency(
    request: Request, session_key: str | None = Cookie(None)
) -> ByokAuth:
    from . import get_manager

    if session_key is None:
        return ByokAuth(user_id=None)

    mgr = get_manager()
    if mgr is None:
        return ByokAuth(user_id=None)

    user_id = mgr.auth_check(session_key)
    if user_id is None:
        raise HTTPException(status_code=401, detail="invalid or expired session")

    return ByokAuth(user_id=user_id)
