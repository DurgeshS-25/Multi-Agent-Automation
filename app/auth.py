"""Supabase-backed login gate.

How access works:
- You create your one account from a single-use Supabase invite link and set
  your own password there. Public sign-ups are switched off in Supabase, so
  no one else can register. No password lives in code or env vars.
- The browser signs in with Supabase and sends the access token as
  `Authorization: Bearer <token>` on every API call.
- This middleware checks that token with Supabase (cached briefly) and, if
  APP_OWNER_EMAIL is set, that it belongs to you.

Public (no token needed): the HTML pages under /ui/ and /login (they hold no
data and redirect to sign-in themselves), /health, and /auth/config.
Everything else, including every /research endpoint and /docs, needs a token.

Env vars:
  SUPABASE_URL          https://<project>.supabase.co
  SUPABASE_ANON_KEY     the publishable / anon key (safe to expose to browsers)
  APP_OWNER_EMAIL       optional; only this account may use the app

If SUPABASE_URL or SUPABASE_ANON_KEY is missing, the gate is OFF (handy for
local development) and the pages skip sign-in.
"""

import asyncio
import hashlib
import json
import logging
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse, JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)

LOGIN_PAGE = Path(__file__).parent / "static" / "login.html"

PUBLIC_PATHS = {"/", "/health", "/login", "/auth/config", "/favicon.ico"}
PUBLIC_PREFIXES = ("/ui",)

TOKEN_CACHE_SECONDS = 60
VERIFY_TIMEOUT_SECONDS = 10


class SupabaseUnavailable(Exception):
    """Supabase couldn't be reached to verify a token."""


class SupabaseVerifier:
    """Checks an access token by asking Supabase who it belongs to.

    Asking Supabase (GET /auth/v1/user) works for every key type and signing
    algorithm, and immediately rejects tokens from signed-out sessions. Results
    are cached for a minute so each request doesn't cost a round trip.
    """

    def __init__(self, supabase_url: str, anon_key: str):
        self._user_url = supabase_url.rstrip("/") + "/auth/v1/user"
        self._anon_key = anon_key
        self._cache: dict[str, tuple[float, dict | None]] = {}

    def _fetch_user(self, token: str) -> dict | None:
        req = urllib.request.Request(
            self._user_url,
            headers={"apikey": self._anon_key, "Authorization": f"Bearer {token}"},
        )
        try:
            with urllib.request.urlopen(req, timeout=VERIFY_TIMEOUT_SECONDS) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            if e.code in (401, 403):
                return None
            raise SupabaseUnavailable(f"Supabase returned {e.code}") from e
        except (urllib.error.URLError, TimeoutError) as e:
            raise SupabaseUnavailable(str(e)) from e

    async def user_for(self, token: str) -> dict | None:
        key = hashlib.sha256(token.encode()).hexdigest()
        now = time.monotonic()
        cached = self._cache.get(key)
        if cached and cached[0] > now:
            return cached[1]

        user = await asyncio.to_thread(self._fetch_user, token)
        if len(self._cache) > 1000:
            self._cache = {k: v for k, v in self._cache.items() if v[0] > now}
        self._cache[key] = (now + TOKEN_CACHE_SECONDS, user)
        return user


def _is_public(path: str) -> bool:
    if path in PUBLIC_PATHS:
        return True
    return any(path == p or path.startswith(p + "/") for p in PUBLIC_PREFIXES)


def _bearer_token(request: Request) -> str | None:
    scheme, _, token = request.headers.get("authorization", "").partition(" ")
    return token.strip() if scheme.lower() == "bearer" and token.strip() else None


class _AuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, verifier: SupabaseVerifier, owner_email: str | None):
        super().__init__(app)
        self.verifier = verifier
        self.owner_email = owner_email.strip().lower() if owner_email else None

    async def dispatch(self, request: Request, call_next):
        if _is_public(request.url.path):
            return await call_next(request)

        token = _bearer_token(request)
        if not token:
            return JSONResponse({"detail": "Sign in required."}, status_code=401)

        try:
            user = await self.verifier.user_for(token)
        except SupabaseUnavailable as e:
            logger.error("Token check failed, Supabase unreachable: %s", e)
            return JSONResponse({"detail": "Sign-in service unavailable. Try again shortly."}, status_code=503)

        if not user:
            return JSONResponse({"detail": "Session expired. Sign in again."}, status_code=401)

        email = (user.get("email") or "").lower()
        if self.owner_email and email != self.owner_email:
            logger.warning("Rejected signed-in non-owner account %s", email or user.get("id"))
            return JSONResponse({"detail": "This account doesn't have access."}, status_code=403)

        request.state.user = {"id": user.get("id"), "email": user.get("email")}
        return await call_next(request)


def _build_router(config: dict) -> APIRouter:
    router = APIRouter(include_in_schema=False)

    @router.get("/login")
    def login_page():
        return FileResponse(LOGIN_PAGE, headers={"Cache-Control": "no-store"})

    @router.get("/auth/config")
    def auth_config():
        # Only public values: the URL and the publishable/anon key are meant to
        # be used by browsers. Never put the secret/service key here.
        return config

    return router


def supabase_base_url(raw: str | None) -> str | None:
    """Reduce any pasted Supabase URL (e.g. one ending in /rest/v1/) to scheme://host."""
    if not raw:
        return None
    parts = urlsplit(raw.strip())
    if not (parts.scheme and parts.netloc):
        return raw.strip().rstrip("/")
    return f"{parts.scheme}://{parts.netloc}"


def setup_auth(app) -> bool:
    """Attach the login page, config endpoint and token gate. Returns True if the gate is on."""
    url = supabase_base_url(os.getenv("SUPABASE_URL"))
    anon_key = os.getenv("SUPABASE_ANON_KEY") or os.getenv("SUPABASE_PUBLISHABLE_KEY")

    if not (url and anon_key):
        logger.warning("SUPABASE_URL / SUPABASE_ANON_KEY not set: sign-in is DISABLED.")
        app.include_router(_build_router({"enabled": False}))
        return False

    app.include_router(_build_router({"enabled": True, "supabaseUrl": url, "supabaseAnonKey": anon_key}))
    app.add_middleware(
        _AuthMiddleware,
        verifier=SupabaseVerifier(url, anon_key),
        owner_email=os.getenv("APP_OWNER_EMAIL"),
    )
    return True