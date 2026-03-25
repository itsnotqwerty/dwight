"""Cookie-based authentication helpers for the web UI.

Password is read from the ``DWIGHT_PASSWORD`` environment variable at request
time.  If the variable is unset or empty, all UI routes are accessible without
authentication (existing behaviour).

Signing uses stdlib ``hmac`` + ``hashlib`` + ``secrets`` — no extra deps.
The signing key is derived from the password via SHA-256 so sessions survive
server restarts as long as the password stays the same.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets

from fastapi import Request
from fastapi.responses import RedirectResponse

_COOKIE_NAME = "dwight_session"


def _get_password() -> str:
    return os.environ.get("DWIGHT_PASSWORD", "")


def _signing_key(password: str) -> bytes:
    """Derive a stable signing key from the password."""
    return hashlib.sha256(password.encode()).digest()


def _sign(token: str, password: str) -> str:
    """Return ``{token}.{hmac_hex}``."""
    mac = hmac.new(_signing_key(password), token.encode(), digestmod="sha256")
    return f"{token}.{mac.hexdigest()}"


def _verify(cookie_val: str, password: str) -> bool:
    """Constant-time verification of a signed cookie value."""
    if "." not in cookie_val:
        return False
    token, _, sig = cookie_val.rpartition(".")
    expected = hmac.new(_signing_key(password), token.encode(), digestmod="sha256").hexdigest()
    return hmac.compare_digest(sig, expected)


def check_password(candidate: str) -> bool:
    """Constant-time comparison against the configured password."""
    password = _get_password()
    if not password:
        return False
    return hmac.compare_digest(candidate, password)


def is_authenticated(request: Request) -> bool:
    """Return True if the request carries a valid session cookie."""
    password = _get_password()
    if not password:
        return True  # auth disabled
    cookie = request.cookies.get(_COOKIE_NAME, "")
    return bool(cookie) and _verify(cookie, password)


def set_session_cookie(response, password: str) -> None:
    """Attach a freshly signed session cookie to *response*."""
    token = secrets.token_urlsafe(32)
    value = _sign(token, password)
    response.set_cookie(
        key=_COOKIE_NAME,
        value=value,
        httponly=True,
        samesite="lax",
    )


def delete_session_cookie(response) -> None:
    response.delete_cookie(_COOKIE_NAME)


def require_auth(request: Request):
    """FastAPI dependency — redirects to /login if not authenticated."""
    if not is_authenticated(request):
        raise _LoginRedirect()


class _LoginRedirect(Exception):
    """Sentinel used by the auth dependency; caught by the exception handler."""


def login_redirect_response() -> RedirectResponse:
    return RedirectResponse("/login", status_code=302)
