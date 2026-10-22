"""Email + password authentication for the XPLORA API.

Passwords are bcrypt-hashed and the session is a signed JWT stored in an
HTTP-only cookie, so the token is never readable from JavaScript. A Bearer
header is accepted as an alternative for API clients and tests.
"""

import os
import secrets
import time
from typing import Optional

import bcrypt
import jwt
from dotenv import load_dotenv

load_dotenv()

COOKIE_NAME = "xplora_session"
ALGORITHM = "HS256"
# bcrypt only ever sees the first 72 bytes of a password.
MAX_PASSWORD_BYTES = 72
MIN_PASSWORD_LEN = 8


def _jwt_secret() -> bytes:
    secret = (os.getenv("JWT_SECRET") or "").strip()
    if secret:
        return secret.encode()
    # Never sign with a hardcoded key: fall back to a per-process random secret
    # so a misconfigured deployment simply invalidates its own sessions.
    generated = secrets.token_bytes(32)
    print("[AUTH] JWT_SECRET is not set — using a random per-process secret. "
          "Sessions will be invalidated on restart.")
    return generated


def cookie_max_age() -> int:
    try:
        return max(3600, int(os.getenv("AUTH_COOKIE_MAX_AGE", 60 * 60 * 24 * 30)))
    except ValueError:
        return 60 * 60 * 24 * 30


def cookie_secure() -> bool:
    """Cookies must be Secure in production; local http:// dev needs the opt-out."""
    raw = (os.getenv("AUTH_COOKIE_SECURE") or "").strip().lower()
    if raw in ("1", "true", "yes"):
        return True
    if raw in ("0", "false", "no"):
        return False
    return (os.getenv("ENVIRONMENT") or os.getenv("NODE_ENV") or "").strip().lower() in (
        "production", "prod"
    )


# ─────────────────────────── passwords ───────────────────────────

def _truncate(password: str) -> bytes:
    """bcrypt's 72-byte input limit, enforced without splitting a UTF-8 sequence."""
    raw = password.encode("utf-8")[:MAX_PASSWORD_BYTES]
    while raw:
        try:
            raw.decode("utf-8")
            return raw
        except UnicodeDecodeError:
            raw = raw[:-1]
    return raw


def hash_password(password: str) -> str:
    return bcrypt.hashpw(_truncate(password), bcrypt.gensalt(rounds=12)).decode()


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(_truncate(password), password_hash.encode())
    except (ValueError, TypeError):
        return False


def password_problem(password: str, confirm: Optional[str] = None) -> Optional[str]:
    """Returns a human-readable validation error, or None when the password is fine."""
    if not password or len(password) < MIN_PASSWORD_LEN:
        return f"Password must be at least {MIN_PASSWORD_LEN} characters."
    if len(password.encode("utf-8")) > MAX_PASSWORD_BYTES:
        return f"Password must be at most {MAX_PASSWORD_BYTES} bytes."
    if confirm is not None and password != confirm:
        return "Passwords do not match."
    return None


# ───────────────────────────── tokens ─────────────────────────────

def create_token(user: dict) -> str:
    now = int(time.time())
    payload = {
        "sub": str(user["id"]),
        "email": user.get("email"),
        "name": user.get("display_name"),
        "iat": now,
        "exp": now + cookie_max_age(),
    }
    return jwt.encode(payload, _jwt_secret(), algorithm=ALGORITHM)


def decode_token(token: str) -> Optional[int]:
    """Returns the user id carried by a valid token, else None."""
    if not token:
        return None
    try:
        payload = jwt.decode(token, _jwt_secret(), algorithms=[ALGORITHM])
        return int(payload["sub"])
    except (jwt.PyJWTError, KeyError, TypeError, ValueError):
        return None
