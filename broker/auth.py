"""
TOTP-based two-factor authentication for broker APIs.
Handles TOTP generation, session creation, validation, and auto-refresh.

Kite Connect Login Flow (fully automated, no browser):
  1. POST /api/login → get request_id
  2. POST /api/twofa → get redirect with request_token
  3. POST /session/token → exchange for access_token
"""

import hashlib
import logging
import re
import time
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import parse_qs, urlparse

import pyotp
import requests

from config.settings import load_settings

logger = logging.getLogger(__name__)


def _mask(value: str, visible: int = 4) -> str:
    """Mask a sensitive string, keeping only the last *visible* characters."""
    if not value or len(value) <= visible:
        return "***"
    return "*" * (len(value) - visible) + value[-visible:]


class TOTPAuthenticator:
    """Generates time-based OTPs and manages broker authentication sessions.

    Args:
        totp_secret: Base32-encoded TOTP secret key.
        session_ttl_seconds: How long an access token is considered valid.
            Defaults to 6 hours (Zerodha tokens expire at 06:00 IST next day,
            but we re-authenticate well before that).
    """

    DEFAULT_SESSION_TTL_SECONDS: int = 6 * 60 * 60  # 6 hours
    LOGIN_TIMEOUT_SECONDS: int = 30
    MAX_LOGIN_RETRIES: int = 3
    RETRY_BACKOFF_BASE: float = 1.0

    KITE_LOGIN_URL = "https://kite.zerodha.com/api/login"
    KITE_TWOFA_URL = "https://kite.zerodha.com/api/twofa"
    KITE_SESSION_URL = "https://api.kite.trade/session/token"

    def __init__(
        self,
        totp_secret: str,
        session_ttl_seconds: int = DEFAULT_SESSION_TTL_SECONDS,
    ) -> None:
        self._totp = pyotp.TOTP(totp_secret)
        self._session_ttl = session_ttl_seconds
        self._manual_totp: str | None = None
        logger.info(
            "TOTPAuthenticator initialised",
            extra={"session_ttl_seconds": session_ttl_seconds},
        )

    def set_manual_totp(self, code: str) -> None:
        """Set a manually-provided TOTP code for the next login attempt.

        Use this when the auto-generated TOTP doesn't match (e.g., Kite's
        built-in authenticator). The code is consumed on the next login.
        """
        self._manual_totp = code.strip()
        logger.info("Manual TOTP code set (will be used for next login)")

    # ── TOTP Generation ─────────────────────────────────────────────────

    def generate_totp(self) -> str:
        """Generate the current 6-digit TOTP."""
        otp = self._totp.now()
        logger.debug("TOTP generated", extra={"otp_prefix": otp[:2] + "****"})
        return otp

    # ── Session Management ──────────────────────────────────────────────

    def authenticate_session(
        self,
        api_key: str,
        api_secret: str,
        user_id: str,
        password: str,
        totp_secret: str,
        login_url: Optional[str] = None,
    ) -> dict:
        """Perform full Kite Connect login: login → 2FA → token exchange.

        Kite Connect Flow:
          Step 1: POST to /api/login with user_id + password → request_id
          Step 2: POST to /api/twofa with request_id + TOTP → request_token
          Step 3: POST to /session/token with api_key + request_token + checksum
                  → access_token

        Returns:
            Session dict with access_token, expires_at, user_id.

        Raises:
            ConnectionError: If login fails after retries.
            ValueError: If response is unexpected.
        """
        logger.info(
            "Authenticating broker session",
            extra={"user_id": _mask(user_id), "api_key": _mask(api_key)},
        )

        last_error: Optional[Exception] = None
        for attempt in range(1, self.MAX_LOGIN_RETRIES + 1):
            try:
                logger.info("Login attempt %d/%d", attempt, self.MAX_LOGIN_RETRIES)

                # Use a session to maintain cookies
                sess = requests.Session()

                # ── Step 1: Login with user_id + password ──
                login_resp = sess.post(
                    self.KITE_LOGIN_URL,
                    data={
                        "user_id": user_id,
                        "password": password,
                    },
                    timeout=self.LOGIN_TIMEOUT_SECONDS,
                )
                login_resp.raise_for_status()
                login_data = login_resp.json()

                if login_data.get("status") != "success":
                    raise ValueError(f"Login failed: {login_data}")

                request_id = login_data.get("data", {}).get("request_id", "")
                if not request_id:
                    raise ValueError(f"No request_id in login response: {login_data}")

                logger.info("Step 1 (login) succeeded | request_id=%s", _mask(request_id))

                # ── Step 2: 2FA with TOTP ──
                # Try auto-generated TOTP first, fall back to manual input
                twofa_type = login_data.get("data", {}).get("twofa_type", "app_code")

                if self._manual_totp:
                    totp_value = self._manual_totp
                    self._manual_totp = None  # Use only once
                else:
                    totp_auth = pyotp.TOTP(totp_secret)
                    totp_value = totp_auth.now()

                twofa_resp = sess.post(
                    self.KITE_TWOFA_URL,
                    data={
                        "user_id": user_id,
                        "request_id": request_id,
                        "twofa_value": totp_value,
                        "twofa_type": twofa_type,
                    },
                    timeout=self.LOGIN_TIMEOUT_SECONDS,
                    allow_redirects=False,  # Don't follow redirect — we need the URL
                )

                # The 2FA response may redirect with request_token in URL
                # or return it in the JSON response
                request_token = None

                if twofa_resp.status_code in (200, 302, 303):
                    # Check redirect location for request_token
                    redirect_url = twofa_resp.headers.get("Location", "")
                    if "request_token=" in redirect_url:
                        parsed = parse_qs(urlparse(redirect_url).query)
                        request_token = parsed.get("request_token", [None])[0]

                    # Or check JSON body
                    if not request_token:
                        try:
                            twofa_data = twofa_resp.json()
                            if twofa_data.get("status") == "success":
                                request_token = twofa_data.get("data", {}).get("request_token", "")
                        except Exception:
                            pass

                    # Or check the response history for redirects
                    if not request_token:
                        for hist_resp in twofa_resp.history:
                            loc = hist_resp.headers.get("Location", "")
                            if "request_token=" in loc:
                                parsed = parse_qs(urlparse(loc).query)
                                request_token = parsed.get("request_token", [None])[0]
                                break

                if not request_token:
                    # After successful 2FA, the session cookie is set.
                    # Visit Kite Connect login URL — step through redirects
                    # manually to capture request_token.
                    request_token = self._follow_connect_redirects(
                        sess, api_key
                    )

                if not request_token:
                    raise ValueError(
                        "Could not extract request_token from 2FA response. "
                        f"Status: {twofa_resp.status_code}, "
                        f"Headers: {dict(twofa_resp.headers)}"
                    )

                logger.info("Step 2 (2FA) succeeded | request_token=%s", _mask(request_token))

                # ── Step 3: Exchange request_token for access_token ──
                checksum = hashlib.sha256(
                    (api_key + request_token + api_secret).encode()
                ).hexdigest()

                token_resp = sess.post(
                    self.KITE_SESSION_URL,
                    data={
                        "api_key": api_key,
                        "request_token": request_token,
                        "checksum": checksum,
                    },
                    timeout=self.LOGIN_TIMEOUT_SECONDS,
                )
                token_resp.raise_for_status()
                token_data = token_resp.json()

                access_token = (
                    token_data.get("data", {}).get("access_token")
                    or token_data.get("access_token")
                )

                if not access_token:
                    raise ValueError(
                        f"No access_token in response: {list(token_data.keys())}"
                    )

                now = time.time()
                session = {
                    "access_token": access_token,
                    "expires_at": now + self._session_ttl,
                    "user_id": user_id,
                    "authenticated_at": datetime.now(timezone.utc).isoformat(),
                }

                logger.info(
                    "Step 3 (token exchange) succeeded | Authenticated!",
                    extra={
                        "user_id": _mask(user_id),
                        "token_prefix": _mask(access_token, 6),
                    },
                )
                return session

            except Exception as exc:
                last_error = exc
                wait = self.RETRY_BACKOFF_BASE * (2 ** (attempt - 1))
                logger.warning(
                    "Login attempt %d failed: %s — retrying in %.1fs",
                    attempt, exc, wait,
                )
                time.sleep(wait)

        raise ConnectionError(
            f"Broker login failed after {self.MAX_LOGIN_RETRIES} attempts: {last_error}"
        ) from last_error

    # ── Connect Redirect Handling ──────────────────────────────────────

    def _follow_connect_redirects(
        self, sess: requests.Session, api_key: str
    ) -> str | None:
        """Step through the Kite Connect redirect chain to get request_token.

        Handles two scenarios:
        * **Previously authorized app** — the chain redirects straight to the
          localhost callback URL containing ``request_token``.
        * **First-time authorization** — the chain lands on the authorize page.
          We POST to ``/connect/finish`` with the ``authorize`` field set to
          the user's ``public_token`` (as Kite's frontend does).
        """
        connect_url = f"https://kite.trade/connect/login?api_key={api_key}&v=3"

        # --- Approach 1: step manually with allow_redirects=False ----------
        try:
            url = connect_url
            sess_id = ""
            for _ in range(10):
                r = sess.get(
                    url, timeout=self.LOGIN_TIMEOUT_SECONDS,
                    allow_redirects=False,
                )
                loc = r.headers.get("Location", "")

                if r.status_code in (301, 302, 303, 307, 308):
                    # Resolve relative redirects
                    if loc.startswith("/"):
                        loc = f"https://kite.zerodha.com{loc}"

                    # Capture request_token from any redirect URL
                    if "request_token=" in loc:
                        parsed = parse_qs(urlparse(loc).query)
                        token = parsed.get("request_token", [None])[0]
                        if token:
                            logger.info("Got request_token from redirect chain")
                            return token

                    # Localhost callback — extract token even though server
                    # is not running (we never follow this redirect).
                    if loc.startswith(("http://127.0.0.1", "http://localhost")):
                        parsed = parse_qs(urlparse(loc).query)
                        token = parsed.get("request_token", [None])[0]
                        if token:
                            logger.info("Got request_token from callback redirect")
                            return token

                    # Track sess_id for authorize step
                    if "sess_id=" in loc:
                        parsed = parse_qs(urlparse(loc).query)
                        sess_id = parsed.get("sess_id", [sess_id])[0]

                    url = loc
                    continue

                if r.status_code == 200 and "authorize" in url:
                    # Landed on the authorize page — need to submit the form.
                    return self._authorize_app(sess, api_key, sess_id)

                # Unexpected status
                break

        except Exception as e:
            logger.debug("Redirect stepping failed: %s", e)

        # --- Approach 2: allow_redirects=True and catch ConnectionError ----
        try:
            sess.get(
                connect_url,
                timeout=self.LOGIN_TIMEOUT_SECONDS,
                allow_redirects=True,
            )
        except requests.exceptions.ConnectionError as ce:
            error_url = str(ce)
            if "request_token=" in error_url:
                match = re.search(r"request_token=([a-zA-Z0-9]+)", error_url)
                if match:
                    logger.info("Extracted request_token from ConnectionError")
                    return match.group(1)

        return None

    def _authorize_app(
        self, sess: requests.Session, api_key: str, sess_id: str
    ) -> str | None:
        """Programmatically authorize a Kite Connect app (first-time consent).

        Kite's authorize form sends ``authorize=<public_token>`` — not a
        boolean.  We replicate this by reading the public_token from the
        session cookies or the ``/api/connect/session`` endpoint.
        """
        # Get public_token from cookies (set during 2FA)
        public_token = sess.cookies.get("public_token", "")

        if not public_token and sess_id:
            # Fallback: fetch from session endpoint
            try:
                sr = sess.get(
                    "https://kite.zerodha.com/api/connect/session"
                    f"?api_key={api_key}&sess_id={sess_id}",
                    timeout=self.LOGIN_TIMEOUT_SECONDS,
                )
                if sr.status_code == 200:
                    public_token = sr.json().get("data", {}).get("public_token", "")
            except Exception:
                pass

        if not public_token:
            logger.error("Cannot authorize app: no public_token available")
            return None

        logger.info("First-time app authorization — submitting consent form")

        try:
            r = sess.post(
                "https://kite.zerodha.com/connect/finish",
                data={
                    "api_key": api_key,
                    "sess_id": sess_id,
                    "authorize": public_token,
                },
                timeout=self.LOGIN_TIMEOUT_SECONDS,
                allow_redirects=False,
            )

            loc = r.headers.get("Location", "")
            if "request_token=" in loc:
                parsed = parse_qs(urlparse(loc).query)
                token = parsed.get("request_token", [None])[0]
                if token:
                    logger.info("App authorized successfully — got request_token")
                    return token

            # Follow localhost redirect
            if loc.startswith(("http://127.0.0.1", "http://localhost")):
                parsed = parse_qs(urlparse(loc).query)
                token = parsed.get("request_token", [None])[0]
                if token:
                    logger.info("App authorized — got request_token from callback")
                    return token

            logger.error(
                "App authorization failed: status=%d body=%s",
                r.status_code, r.text[:200],
            )
        except Exception as e:
            logger.error("App authorization error: %s", e)

        return None

    # ── Validation / Refresh ────────────────────────────────────────────

    def is_session_valid(self, session: dict) -> bool:
        """Check whether *session* is still valid (not expired)."""
        if not session or "expires_at" not in session:
            return False
        valid = time.time() < session["expires_at"]
        if not valid:
            logger.info("Session expired", extra={"user_id": _mask(session.get("user_id", ""))})
        return valid

    def refresh_if_needed(
        self,
        session: dict,
        api_key: str = "",
        api_secret: str = "",
        user_id: str = "",
        password: str = "",
        totp_secret: str = "",
    ) -> dict:
        """Return existing session if valid, otherwise re-authenticate."""
        if self.is_session_valid(session):
            return session

        logger.info("Session expired or invalid; re-authenticating")
        settings = load_settings()
        return self.authenticate_session(
            api_key=api_key or settings.broker.api_key,
            api_secret=api_secret or settings.broker.api_secret,
            user_id=user_id or session.get("user_id", "") or settings.broker.user_id,
            password=password or settings.broker.password,
            totp_secret=totp_secret or settings.broker.totp_secret,
        )
