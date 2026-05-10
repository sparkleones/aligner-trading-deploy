"""
Zerodha Kite auto-login via HTTP (no browser needed).

Uses direct POST requests to Kite's login endpoints with credentials +
TOTP, captures request_token from redirect, generates access_token via
KiteConnect. Lightweight alternative to Selenium-based flow — runs in a
plain Docker container with just `requests` + `pyotp` + `kiteconnect`.

WARNING: relies on Zerodha's undocumented login endpoints. Stable for
years but may change. If Zerodha alters the flow, the Selenium-based
fallback in the Antigravity reference still works (slower, requires
Chromium in container).

Required env vars (in `.env`):
  KITE_API_KEY       (or BROKER_API_KEY)
  KITE_API_SECRET    (or BROKER_API_SECRET)
  ZERODHA_USER_ID
  ZERODHA_PASSWORD
  ZERODHA_TOTP_SECRET

On success: writes new KITE_ACCESS_TOKEN to .env (if writable) and
returns a dict with the token. Caller can also use it in-memory.

Usage as CLI:
    python -m broker.kite_auto_login
Usage as library:
    from broker.kite_auto_login import auto_login_kite
    result = auto_login_kite()
    if result["success"]:
        access_token = result["access_token"]
"""
import os
import logging
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse, parse_qs

logger = logging.getLogger(__name__)


def _get_env(*names) -> Optional[str]:
    """Return first non-empty env var from given names."""
    for n in names:
        v = os.getenv(n)
        if v:
            return v
    return None


def _persist_token_to_env(env_path: Path, access_token: str) -> bool:
    """Write KITE_ACCESS_TOKEN to .env file (creates or updates the line).

    Returns True if write succeeded, False otherwise (e.g. read-only fs).
    """
    try:
        if not env_path.exists():
            env_path.write_text(f"KITE_ACCESS_TOKEN={access_token}\n", encoding="utf-8")
            return True
        lines = env_path.read_text(encoding="utf-8").splitlines()
        updated = False
        for i, line in enumerate(lines):
            if line.strip().startswith("KITE_ACCESS_TOKEN="):
                lines[i] = f"KITE_ACCESS_TOKEN={access_token}"
                updated = True
                break
        if not updated:
            lines.append(f"KITE_ACCESS_TOKEN={access_token}")
        env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        # Also update in-process env so the same Python process sees the new value
        os.environ["KITE_ACCESS_TOKEN"] = access_token
        return True
    except Exception as e:
        logger.warning("Could not persist token to .env: %s", e)
        return False


def auto_login_kite(persist: bool = True) -> dict:
    """Run end-to-end auto-login flow. Returns structured result.

    Result schema:
        {
            "success": bool,
            "access_token": str | None,
            "user_id": str | None,
            "error": str | None,
            "stage": str,         # last stage reached, for debugging
            "persisted": bool,    # whether .env was updated
        }
    """
    result = {
        "success": False, "access_token": None, "user_id": None,
        "error": None, "stage": "init", "persisted": False,
    }

    # ── Imports deferred so module loads even without optional deps ──
    try:
        import requests
        import pyotp
        from kiteconnect import KiteConnect
    except ImportError as e:
        result["error"] = f"missing dep: {e}"
        return result

    # ── Reload .env so newly-added credentials are visible without restart ──
    try:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)
    except Exception:
        pass  # not critical — falls back to current process env

    # ── Read credentials (support ZERODHA_*, BROKER_*, and KITE_* prefixes) ──
    api_key = _get_env("KITE_API_KEY", "BROKER_API_KEY")
    api_secret = _get_env("KITE_API_SECRET", "BROKER_API_SECRET")
    user_id = _get_env("ZERODHA_USER_ID", "BROKER_USER_ID", "KITE_USER_ID")
    password = _get_env("ZERODHA_PASSWORD", "BROKER_PASSWORD", "KITE_PASSWORD")
    totp_secret = _get_env("ZERODHA_TOTP_SECRET", "BROKER_TOTP_SECRET", "KITE_TOTP_SECRET")

    missing = [name for name, val in [
        ("API_KEY", api_key), ("API_SECRET", api_secret),
        ("USER_ID", user_id), ("PASSWORD", password),
        ("TOTP_SECRET", totp_secret),
    ] if not val]
    if missing:
        result["error"] = f"missing env vars: {', '.join(missing)}"
        result["stage"] = "credentials"
        return result

    result["user_id"] = user_id

    # ── HTTP session for cookie persistence across the 3 calls ──
    sess = requests.Session()
    sess.headers.update({"User-Agent": "Mozilla/5.0 (compatible; aligner-trading/1.0)"})

    try:
        # Stage 1: POST userid + password to Zerodha login endpoint
        result["stage"] = "login"
        r1 = sess.post(
            "https://kite.zerodha.com/api/login",
            data={"user_id": user_id, "password": password},
            timeout=15,
        )
        if r1.status_code != 200:
            result["error"] = f"login HTTP {r1.status_code}: {r1.text[:200]}"
            return result
        login_data = r1.json()
        if login_data.get("status") != "success":
            result["error"] = f"login failed: {login_data.get('message', 'unknown')}"
            return result
        request_id = login_data["data"]["request_id"]

        # Stage 2: POST TOTP for 2FA
        result["stage"] = "twofa"
        totp = pyotp.TOTP(totp_secret).now()
        r2 = sess.post(
            "https://kite.zerodha.com/api/twofa",
            data={
                "user_id": user_id,
                "request_id": request_id,
                "twofa_value": totp,
                "twofa_type": "totp",
                "skip_session": "",
            },
            timeout=15,
        )
        if r2.status_code != 200:
            result["error"] = f"twofa HTTP {r2.status_code}: {r2.text[:200]}"
            return result
        twofa_data = r2.json()
        if twofa_data.get("status") != "success":
            result["error"] = f"twofa failed: {twofa_data.get('message', 'unknown')}"
            return result

        # Stage 3: GET kite connect login URL, follow redirects MANUALLY so we
        # can capture the request_token from the final redirect URL WITHOUT
        # actually trying to fetch the configured redirect_uri (which is set
        # to your trading dashboard — may be on a different port or unreachable).
        result["stage"] = "request_token"
        kite = KiteConnect(api_key=api_key)
        login_url = kite.login_url()

        request_token = None
        current_url = login_url
        for _ in range(10):  # cap redirects
            try:
                r3 = sess.get(current_url, allow_redirects=False, timeout=15)
            except Exception as fetch_err:
                # Connection refused etc. on the final redirect target.
                # The request_token may already be in current_url from the
                # previous Location header — check before giving up.
                qs = parse_qs(urlparse(current_url).query)
                if "request_token" in qs:
                    request_token = qs["request_token"][0]
                    break
                raise fetch_err

            # Check the CURRENT URL we just fetched (handles 200 endpoint cases)
            qs = parse_qs(urlparse(current_url).query)
            if "request_token" in qs:
                request_token = qs["request_token"][0]
                break

            if r3.status_code in (301, 302, 303, 307, 308) and r3.headers.get("Location"):
                next_url = r3.headers["Location"]
                # Resolve relative URLs against current host
                if next_url.startswith("/"):
                    parsed = urlparse(current_url)
                    next_url = f"{parsed.scheme}://{parsed.netloc}{next_url}"

                # Check the redirect target BEFORE following — captures the
                # request_token from the configured redirect_uri without
                # needing to actually fetch it.
                qs = parse_qs(urlparse(next_url).query)
                if "request_token" in qs:
                    request_token = qs["request_token"][0]
                    break

                current_url = next_url
                continue

            # Non-redirect response with no token — we're stuck
            break

        if not request_token:
            result["error"] = (
                f"could not extract request_token from redirect chain. "
                f"final URL: {current_url[:120]}"
            )
            return result

        # Stage 4: Generate access_token via KiteConnect
        result["stage"] = "access_token"
        session_data = kite.generate_session(request_token, api_secret=api_secret)
        access_token = session_data["access_token"]
        result["access_token"] = access_token

        # Stage 5: persist to .env (optional)
        if persist:
            project_root = Path(__file__).resolve().parent.parent
            env_path = project_root / ".env"
            result["persisted"] = _persist_token_to_env(env_path, access_token)

        result["success"] = True
        result["stage"] = "done"
        return result

    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        return result


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
    print("Running Zerodha Kite auto-login...")
    res = auto_login_kite(persist=True)
    if res["success"]:
        print(f"\n✅ SUCCESS")
        print(f"   user_id:      {res['user_id']}")
        print(f"   access_token: {res['access_token'][:20]}... (truncated)")
        print(f"   persisted:    {res['persisted']}")
    else:
        print(f"\n❌ FAILED at stage '{res['stage']}'")
        print(f"   error: {res['error']}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
