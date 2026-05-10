"""
Trading Dashboard — FastAPI Backend

Serves the real-time trading dashboard with:
- WebSocket streaming of live session events
- REST endpoints for strategy comparison, backtesting, and system status
- Paper trading session management
"""

import asyncio
import hashlib
import json
import logging
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

# Add project root to path
PROJECT_ROOT = str(Path(__file__).parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from config.constants import INDEX_CONFIG, STT_RATES
from backtesting.data_loader import generate_synthetic_ohlcv
from backtesting.backtest_engine import BacktestEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d | %(levelname)-8s | %(name)-20s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("dashboard")

app = FastAPI(title="Indian Options Trading Dashboard", version="1.0.0")

# ── Simple API Key Auth for remote access ──
# Set DASHBOARD_API_KEY in .env to protect the dashboard
# If not set, dashboard is open (local development mode)
DASHBOARD_API_KEY = os.getenv("DASHBOARD_API_KEY", "")

from fastapi import Depends, HTTPException, Query
from fastapi.security import APIKeyQuery

api_key_query = APIKeyQuery(name="key", auto_error=False)

def verify_api_key(key: str = Depends(api_key_query)):
    """Verify API key if DASHBOARD_API_KEY is set."""
    if not DASHBOARD_API_KEY:
        return True  # No auth in dev mode
    if key != DASHBOARD_API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return True


# Serve static files (Chart.js, etc.)
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ── State ────────────────────────────────────────────────────────────────────
connected_clients: list[WebSocket] = []
session_state = {
    "running": False,
    "strategy": None,
    "start_time": None,
    "events": [],
}

# Kite Connect state
kite_state = {
    "connected": False,
    "kite": None,
    "access_token": None,
    "user_profile": None,
    "api_key": os.getenv("KITE_API_KEY") or os.getenv("BROKER_API_KEY", ""),
    "api_secret": os.getenv("KITE_API_SECRET") or os.getenv("BROKER_API_SECRET", ""),
}


def _try_restore_kite_session():
    """On dashboard startup, if KITE_ACCESS_TOKEN is set in .env, try to
    restore the session in-memory so we don't force a re-login on every
    container/launcher restart. The token is fetched once a day by
    auto-login and persists in .env until expiry next midnight."""
    token = os.getenv("KITE_ACCESS_TOKEN", "").strip()
    if not token or not kite_state["api_key"]:
        return
    try:
        from kiteconnect import KiteConnect
        kite = KiteConnect(api_key=kite_state["api_key"])
        kite.set_access_token(token)
        profile = kite.profile()  # validates the token
        kite_state["kite"] = kite
        kite_state["access_token"] = token
        kite_state["connected"] = True
        kite_state["user_profile"] = profile
        logger.info("Kite session restored from .env | user=%s",
                    profile.get("user_name", ""))
    except Exception as e:
        logger.info("Could not restore Kite session from .env (%s) — user must AUTO-LOGIN", e)


# Attempt restore at import time (after logger is configured above)
_try_restore_kite_session()


# ── WebSocket Manager ────────────────────────────────────────────────────────

async def broadcast(event: dict):
    """Send event to all connected WebSocket clients."""
    msg = json.dumps(event, default=str)
    disconnected = []
    for ws in connected_clients:
        try:
            await ws.send_text(msg)
        except Exception:
            disconnected.append(ws)
    for ws in disconnected:
        connected_clients.remove(ws)


# ── API Endpoints ────────────────────────────────────────────────────────────

@app.api_route("/", methods=["GET", "HEAD"])
async def index():
    """Redirect to the live trading dashboard."""
    from starlette.responses import RedirectResponse
    return RedirectResponse(url="/live")

@app.get("/old", response_class=HTMLResponse)
async def old_dashboard():
    """Serve the legacy dashboard (kept at /old for reference)."""
    html_path = Path(__file__).parent / "templates" / "index.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


# ── Live Dashboard PWA ──────────────────────────────────────────────────────

@app.get("/live", response_class=HTMLResponse)
async def live_dashboard(key: str = Query(default="")):
    """Serve the PWA live trading dashboard (mobile-first, real-time).

    If DASHBOARD_API_KEY is set, requires ?key=<your-key> in URL.
    The key is passed to WebSocket connection automatically.
    """
    if DASHBOARD_API_KEY and key != DASHBOARD_API_KEY:
        return HTMLResponse(
            "<html><body style='background:#0a0e17;color:#ef4444;font-family:sans-serif;"
            "display:flex;justify-content:center;align-items:center;height:100vh'>"
            "<div style='text-align:center'><h1>Access Denied</h1>"
            "<p style='color:#94a3b8'>Add ?key=YOUR_KEY to the URL</p></div>"
            "</body></html>",
            status_code=403,
        )
    html_path = Path(__file__).parent / "templates" / "live_pwa.html"
    if html_path.exists():
        html = html_path.read_text(encoding="utf-8")
        # Inject API key into the page so WebSocket can use it
        if DASHBOARD_API_KEY:
            html = html.replace(
                "// ── Init ──",
                f"const API_KEY = '{key}';\n// ── Init ──",
            )
        return HTMLResponse(html)
    return HTMLResponse("<h1>live_pwa.html not found</h1>", status_code=404)


@app.get("/manifest.json")
async def pwa_manifest():
    """PWA manifest for Add to Home Screen."""
    return JSONResponse({
        "name": "Aligner Trading",
        "short_name": "Aligner",
        "start_url": "/live",
        "display": "standalone",
        "background_color": "#0a0e17",
        "theme_color": "#00d4ff",
        "orientation": "portrait",
        "icons": [
            {"src": "/static/icon-192.svg", "sizes": "192x192", "type": "image/svg+xml"},
            {"src": "/static/icon-512.svg", "sizes": "512x512", "type": "image/svg+xml"},
        ],
    })


@app.get("/api/live/state")
async def get_live_state():
    """REST endpoint returning current dashboard state."""
    state_file = Path(PROJECT_ROOT) / "data" / "dashboard_state.json"
    if state_file.exists():
        return JSONResponse(json.loads(state_file.read_text(encoding="utf-8")))
    return JSONResponse({"error": "No state file"}, status_code=404)


@app.post("/api/live/manual_order")
async def post_manual_order(request: Request):
    """Accept manual order request from PWA dashboard."""
    body = await request.json()
    order_file = Path(PROJECT_ROOT) / "data" / "manual_order_request.json"
    body["status"] = "PENDING"
    body["timestamp"] = datetime.now().isoformat()
    order_file.write_text(json.dumps(body, indent=2), encoding="utf-8")
    return JSONResponse({"status": "queued", "order": body})


@app.websocket("/ws/live")
async def ws_live_state(websocket: WebSocket):
    """WebSocket that pushes dashboard_state.json diffs every 500ms.

    Much faster than Streamlit's full-page reload approach.
    Only sends data when something actually changed (hash-based diff).
    """
    await websocket.accept()
    logger.info("PWA WebSocket connected")
    state_file = Path(PROJECT_ROOT) / "data" / "dashboard_state.json"
    last_hash = ""

    try:
        while True:
            try:
                if state_file.exists():
                    raw = state_file.read_text(encoding="utf-8")
                    h = hashlib.md5(raw.encode()).hexdigest()
                    if h != last_hash:
                        last_hash = h
                        await websocket.send_text(raw)
            except (json.JSONDecodeError, FileNotFoundError):
                pass
            except Exception as e:
                logger.debug("ws/live read error: %s", e)

            # Check for client messages (manual orders, ping)
            try:
                msg = await asyncio.wait_for(websocket.receive_text(), timeout=0.5)
                # Handle plain-text ping from keepalive
                if not msg or msg.strip() in ("ping", "pong"):
                    await websocket.send_text("pong")
                    continue
                data = json.loads(msg)
                if data.get("type") == "manual_order":
                    order_file = Path(PROJECT_ROOT) / "data" / "manual_order_request.json"
                    data["status"] = "PENDING"
                    data["timestamp"] = datetime.now().isoformat()
                    order_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
                    await websocket.send_text(json.dumps({"type": "order_ack", "status": "queued"}))
                elif data.get("type") == "ping":
                    await websocket.send_text(json.dumps({"type": "pong"}))
            except asyncio.TimeoutError:
                pass
            except json.JSONDecodeError:
                pass  # Ignore non-JSON client messages

    except WebSocketDisconnect:
        logger.info("PWA WebSocket disconnected")
    except Exception as e:
        logger.error("PWA WebSocket error: %s", e)


# ── Kite Connect OAuth2 Login Flow ──────────────────────────────────────────

@app.get("/login/zerodha")
async def zerodha_login():
    """Redirect user to Zerodha login page for OAuth2 authorization."""
    api_key = kite_state["api_key"]
    if not api_key:
        return JSONResponse(status_code=400, content={
            "error": "BROKER_API_KEY not set in .env"
        })
    login_url = f"https://kite.zerodha.com/connect/login?v=3&api_key={api_key}"
    return RedirectResponse(url=login_url)


@app.get("/callback")
async def zerodha_callback(request_token: str = "", action: str = "", status: str = ""):
    """Handle Zerodha OAuth2 callback after user logs in."""
    if status != "success" or not request_token:
        return HTMLResponse(f"""
        <html><body style="background:#0a0e17;color:#ef4444;font-family:sans-serif;
        display:flex;align-items:center;justify-content:center;height:100vh">
        <div><h2>Login Failed</h2><p>Status: {status}, Action: {action}</p>
        <a href="/" style="color:#3b82f6">Back to Dashboard</a></div>
        </body></html>""")

    api_key = kite_state["api_key"]
    api_secret = kite_state["api_secret"]

    try:
        from kiteconnect import KiteConnect
        kite = KiteConnect(api_key=api_key)
        data = kite.generate_session(request_token, api_secret=api_secret)
        kite.set_access_token(data["access_token"])

        kite_state["connected"] = True
        kite_state["kite"] = kite
        kite_state["access_token"] = data["access_token"]

        # Fetch user profile
        profile = kite.profile()
        kite_state["user_profile"] = profile
        logger.info("Zerodha login successful | user=%s", profile.get("user_name", ""))

        return RedirectResponse(url="/?broker=connected")

    except Exception as e:
        logger.error("Zerodha OAuth callback failed: %s", e)
        return HTMLResponse(f"""
        <html><body style="background:#0a0e17;color:#ef4444;font-family:sans-serif;
        display:flex;align-items:center;justify-content:center;height:100vh">
        <div><h2>Authentication Error</h2><p>{e}</p>
        <a href="/" style="color:#3b82f6">Back to Dashboard</a></div>
        </body></html>""")


@app.post("/api/broker/auto_login")
async def broker_auto_login():
    """Trigger Zerodha auto-login (HTTP-based, no browser).

    Uses .env credentials (ZERODHA_USER_ID, ZERODHA_PASSWORD,
    ZERODHA_TOTP_SECRET) to log in programmatically and refresh the
    access_token. Updates kite_state in-memory and writes new token to
    .env if writable.

    Required env vars:
      KITE_API_KEY (or BROKER_API_KEY)
      KITE_API_SECRET (or BROKER_API_SECRET)
      ZERODHA_USER_ID
      ZERODHA_PASSWORD
      ZERODHA_TOTP_SECRET
    """
    try:
        from broker.kite_auto_login import auto_login_kite
        result = auto_login_kite(persist=True)
    except Exception as e:
        logger.error("auto_login import or call failed: %s", e)
        return JSONResponse(status_code=500, content={
            "success": False, "error": f"{type(e).__name__}: {e}",
        })

    if not result["success"]:
        logger.warning("Kite auto-login failed at stage=%s: %s",
                       result.get("stage"), result.get("error"))
        return JSONResponse(status_code=400, content=result)

    # Wire the new token into kite_state in-memory so existing endpoints
    # (broker/status, broker/positions, etc.) start working immediately
    # without a container restart.
    try:
        from kiteconnect import KiteConnect
        api_key = kite_state["api_key"] or os.getenv("KITE_API_KEY", "")
        kite = KiteConnect(api_key=api_key)
        kite.set_access_token(result["access_token"])
        kite_state["kite"] = kite
        kite_state["connected"] = True
        kite_state["access_token"] = result["access_token"]
        try:
            kite_state["user_profile"] = kite.profile()
        except Exception as e:
            logger.warning("auto_login: token saved but profile fetch failed: %s", e)
        logger.info("Kite auto-login SUCCESS | user_id=%s", result.get("user_id"))
        return {
            "success": True,
            "user_id": result["user_id"],
            "persisted": result["persisted"],
            "user_name": (kite_state.get("user_profile") or {}).get("user_name", ""),
        }
    except Exception as e:
        logger.error("auto_login wiring failed: %s", e)
        return JSONResponse(status_code=500, content={
            "success": False,
            "error": f"token saved but wiring failed: {e}",
            "access_token_len": len(result["access_token"]),
        })


@app.get("/api/broker/auto_login/check")
async def broker_auto_login_check():
    """Check whether all auto-login env vars are configured.

    Used by dashboard UI to show whether the auto-login button can work.
    Does NOT expose secrets — returns presence flags only.
    Reloads .env each call so newly-added vars are picked up without restart.
    """
    # Reload .env so dashboard reflects fresh credentials without restart
    try:
        from dotenv import load_dotenv
        load_dotenv(Path(PROJECT_ROOT) / ".env", override=True)
    except Exception:
        pass
    api_key_v     = os.getenv("KITE_API_KEY") or os.getenv("BROKER_API_KEY")
    api_secret_v  = os.getenv("KITE_API_SECRET") or os.getenv("BROKER_API_SECRET")
    user_id_v     = os.getenv("ZERODHA_USER_ID") or os.getenv("BROKER_USER_ID") or os.getenv("KITE_USER_ID")
    password_v    = os.getenv("ZERODHA_PASSWORD") or os.getenv("BROKER_PASSWORD") or os.getenv("KITE_PASSWORD")
    totp_v        = os.getenv("ZERODHA_TOTP_SECRET") or os.getenv("BROKER_TOTP_SECRET") or os.getenv("KITE_TOTP_SECRET")
    return {
        "api_key_set":     bool(api_key_v),
        "api_secret_set":  bool(api_secret_v),
        "user_id_set":     bool(user_id_v),
        "password_set":    bool(password_v),
        "totp_secret_set": bool(totp_v),
        "ready": all([api_key_v, api_secret_v, user_id_v, password_v, totp_v]),
    }


@app.get("/api/broker/status")
async def broker_status():
    """Check if broker is connected and return account info."""
    if not kite_state["connected"] or kite_state["kite"] is None:
        return {
            "connected": False,
            "login_url": f"/login/zerodha",
            "api_key_set": bool(kite_state["api_key"]),
        }
    try:
        kite = kite_state["kite"]
        margins = kite.margins()
        equity = margins.get("equity", {})
        return {
            "connected": True,
            "user": kite_state["user_profile"].get("user_name", ""),
            "user_id": kite_state["user_profile"].get("user_id", ""),
            "broker": "Zerodha",
            "available_margin": equity.get("available", {}).get("live_balance", 0),
            "used_margin": equity.get("utilised", {}).get("debits", 0),
        }
    except Exception as e:
        kite_state["connected"] = False
        return {"connected": False, "error": str(e), "login_url": "/login/zerodha"}


@app.get("/api/broker/positions")
async def broker_positions():
    """Fetch live positions from Zerodha."""
    if not kite_state["connected"]:
        return JSONResponse(status_code=401, content={"error": "Not connected. Login at /login/zerodha"})
    try:
        kite = kite_state["kite"]
        positions = kite.positions()
        orders = kite.orders()
        return {
            "net_positions": positions.get("net", []),
            "day_positions": positions.get("day", []),
            "orders": orders,
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/api/broker/place_order")
async def place_live_order(
    symbol: str = "",
    side: str = "BUY",
    qty: int = 25,
    order_type: str = "MARKET",
    price: float = 0,
    product: str = "MIS",
):
    """Place a real order on Zerodha (requires broker connection)."""
    if not kite_state["connected"]:
        return JSONResponse(status_code=401, content={"error": "Not connected"})

    kite = kite_state["kite"]
    try:
        params = {
            "tradingsymbol": symbol,
            "exchange": "NFO",
            "transaction_type": side,
            "quantity": qty,
            "order_type": order_type,
            "product": product,
            "variety": "regular",
        }
        if order_type == "LIMIT" and price > 0:
            params["price"] = price

        order_id = kite.place_order(**params)
        logger.info("Live order placed | id=%s symbol=%s side=%s qty=%d", order_id, symbol, side, qty)
        return {"success": True, "order_id": str(order_id), "symbol": symbol}
    except Exception as e:
        logger.error("Live order failed: %s", e)
        return JSONResponse(status_code=400, content={"error": str(e)})


@app.get("/api/broker/option_chain")
async def get_option_chain(symbol: str = "NIFTY", expiry: str = ""):
    """Fetch option chain from Zerodha for strike selection."""
    if not kite_state["connected"]:
        return JSONResponse(status_code=401, content={"error": "Not connected"})
    try:
        kite = kite_state["kite"]
        instruments = kite.instruments("NFO")

        # Filter for the symbol and nearest expiry
        from datetime import date
        today = date.today()
        options = [
            i for i in instruments
            if i.get("name") == symbol
            and i.get("instrument_type") in ("CE", "PE")
            and i.get("expiry") is not None
            and i["expiry"] >= today
        ]

        if not options:
            return {"chain": {}, "expiries": []}

        # Get available expiries
        expiries = sorted(set(str(o["expiry"]) for o in options))
        target_expiry = expiry or str(min(o["expiry"] for o in options))

        chain = {}
        for o in options:
            if str(o["expiry"]) != target_expiry:
                continue
            strike = float(o["strike"])
            opt_type = o["instrument_type"]
            chain.setdefault(strike, {})
            chain[strike][opt_type] = {
                "tradingsymbol": o["tradingsymbol"],
                "lot_size": o["lot_size"],
                "instrument_token": o["instrument_token"],
            }

        return {"chain": chain, "expiries": expiries, "selected_expiry": target_expiry}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# ── Trading Terminal (Professional UI) ─────────────────────────────────────

@app.get("/terminal", response_class=HTMLResponse)
async def terminal_dashboard(key: str = Query(default="")):
    """Serve the professional trading terminal UI.

    Full-featured dashboard with option chain, market pulse, decision engine,
    positions, orders, and strategy panels. All real-time via WebSocket.
    """
    if DASHBOARD_API_KEY and key != DASHBOARD_API_KEY:
        return HTMLResponse(
            "<html><body style='background:#0b0f19;color:#ef4444;font-family:sans-serif;"
            "display:flex;justify-content:center;align-items:center;height:100vh'>"
            "<div style='text-align:center'><h1>Access Denied</h1>"
            "<p style='color:#94a3b8'>Add ?key=YOUR_KEY to the URL</p></div>"
            "</body></html>",
            status_code=403,
        )
    html_path = Path(__file__).parent / "templates" / "terminal.html"
    if html_path.exists():
        html = html_path.read_text(encoding="utf-8")
        if DASHBOARD_API_KEY:
            html = html.replace(
                "// ── Init ──",
                f"const API_KEY = '{key}';\n// ── Init ──",
            )
        return HTMLResponse(html)
    return HTMLResponse("<h1>terminal.html not found</h1>", status_code=404)


# ── Engine Control Endpoints ───────────────────────────────────────────────

engine_control = {
    "paused": False,
    "kill_switch": False,
}

# ── Engine Process Management ──
_engine_process: Optional[subprocess.Popen] = None
_engine_log_file = None


def _find_external_engine_pids() -> list[int]:
    """Find any run_autonomous.py processes NOT managed by this dashboard."""
    pids = []
    try:
        import psutil
        for proc in psutil.process_iter(["pid", "cmdline"]):
            try:
                cmdline = proc.info.get("cmdline") or []
                cmd_str = " ".join(cmdline).lower()
                if "run_autonomous" in cmd_str and proc.pid != os.getpid():
                    # Skip our own managed process
                    if _engine_process and proc.pid == _engine_process.pid:
                        continue
                    pids.append(proc.pid)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    except ImportError:
        # psutil not available — fall back to platform command
        if sys.platform == "win32":
            try:
                result = subprocess.run(
                    ["wmic", "process", "where",
                     "commandline like '%run_autonomous%'",
                     "get", "processid"],
                    capture_output=True, text=True, timeout=5,
                )
                for line in result.stdout.strip().split("\n"):
                    line = line.strip()
                    if line.isdigit():
                        pid = int(line)
                        if _engine_process and pid == _engine_process.pid:
                            continue
                        if pid != os.getpid():
                            pids.append(pid)
            except Exception:
                pass
    return pids


def _is_engine_running() -> bool:
    """Check if the trading engine process is alive (managed or external)."""
    global _engine_process
    # Check our managed process first
    if _engine_process is not None:
        retcode = _engine_process.poll()
        if retcode is not None:
            # Process exited
            _engine_process = None
            if _engine_log_file:
                try:
                    _engine_log_file.close()
                except Exception:
                    pass
            return False
        return True

    # Also check for externally-started engine processes
    external_pids = _find_external_engine_pids()
    if external_pids:
        logger.info("Found external engine process(es): %s", external_pids)
        return True

    return False


@app.post("/api/engine/start")
async def start_engine(mode: str = "paper"):
    """Start the trading engine (run_autonomous.py) as a subprocess.

    Args:
        mode: 'paper' (default) or 'live'
    """
    global _engine_process, _engine_log_file

    # If engine is already running (either managed by us or by watchdog), don't kill it
    external_pids = _find_external_engine_pids()
    if external_pids:
        logger.info("Engine already running via watchdog (PIDs: %s) — not starting another",
                     external_pids)
        return {"status": "ALREADY_RUNNING", "pid": external_pids[0],
                "message": "Engine managed by watchdog"}

    if _is_engine_running():
        return {"status": "ALREADY_RUNNING", "pid": _engine_process.pid}

    python_exe = sys.executable
    engine_script = str(Path(PROJECT_ROOT) / "run_autonomous.py")
    cmd = [python_exe, engine_script]
    if mode == "paper":
        cmd.append("--paper")

    # Log file for engine output
    log_dir = Path(PROJECT_ROOT) / "logs"
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / f"engine_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    try:
        _engine_log_file = open(log_path, "w", encoding="utf-8")
        _engine_process = subprocess.Popen(
            cmd,
            cwd=PROJECT_ROOT,
            stdout=_engine_log_file,
            stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
        )
        logger.info("Engine started | PID=%d | mode=%s | log=%s", _engine_process.pid, mode, log_path)

        # Reset control state
        engine_control["paused"] = False
        engine_control["kill_switch"] = False

        return {"status": "STARTED", "pid": _engine_process.pid, "mode": mode, "log": str(log_path)}
    except Exception as e:
        logger.error("Failed to start engine: %s", e)
        return JSONResponse(status_code=500, content={"status": "FAILED", "error": str(e)})


@app.post("/api/engine/stop")
async def stop_engine():
    """Stop the running trading engine."""
    global _engine_process, _engine_log_file

    # If engine is managed by watchdog, don't stop it (watchdog will restart it anyway)
    if _engine_process is None:
        external = _find_external_engine_pids()
        if external:
            logger.info("Engine managed by watchdog (PIDs: %s) — use watchdog to stop", external)
            return {"status": "WATCHDOG_MANAGED", "pid": external[0],
                    "message": "Engine is managed by watchdog. Use Ctrl+C on watchdog terminal to stop."}
        return {"status": "NOT_RUNNING"}

    if not _is_engine_running():
        return {"status": "NOT_RUNNING"}

    pid = _engine_process.pid
    try:
        if sys.platform == "win32":
            _engine_process.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            _engine_process.send_signal(signal.SIGTERM)

        # Wait up to 15 seconds for graceful shutdown
        try:
            _engine_process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            logger.warning("Engine didn't stop in 15s — killing PID %d", pid)
            _engine_process.kill()
            _engine_process.wait(timeout=5)

        logger.info("Engine stopped | PID=%d", pid)
    except Exception as e:
        logger.error("Error stopping engine: %s", e)
        try:
            _engine_process.kill()
        except Exception:
            pass

    _engine_process = None
    if _engine_log_file:
        try:
            _engine_log_file.close()
        except Exception:
            pass
        _engine_log_file = None

    return {"status": "STOPPED", "pid": pid}


@app.post("/api/engine/toggle")
async def toggle_engine():
    """Toggle engine: start if stopped, pause/resume if running."""
    if not _is_engine_running():
        # Engine not running — read mode from .env (PAPER_TRADING=false means LIVE)
        from config.settings import load_settings
        _settings = load_settings()
        _mode = "paper" if _settings.trading.paper_trading else "live"
        return await start_engine(mode=_mode)

    # Engine running — toggle pause state
    engine_control["paused"] = not engine_control["paused"]
    status = "PAUSED" if engine_control["paused"] else "TRADING"

    # Write control signal to a file the live agent can poll
    control_file = Path(PROJECT_ROOT) / "data" / "engine_control.json"
    control_file.parent.mkdir(parents=True, exist_ok=True)
    control_file.write_text(
        json.dumps({"paused": engine_control["paused"], "kill_switch": engine_control["kill_switch"],
                     "timestamp": datetime.now().isoformat()}),
        encoding="utf-8",
    )
    logger.info("Engine toggled: %s", status)
    return {"status": status, "paused": engine_control["paused"]}


@app.post("/api/kill-switch")
async def activate_kill_switch():
    """Activate kill switch — closes all positions and halts trading."""
    engine_control["kill_switch"] = True
    engine_control["paused"] = True

    control_file = Path(PROJECT_ROOT) / "data" / "engine_control.json"
    control_file.parent.mkdir(parents=True, exist_ok=True)
    control_file.write_text(
        json.dumps({"paused": True, "kill_switch": True,
                     "timestamp": datetime.now().isoformat()}),
        encoding="utf-8",
    )
    logger.warning("KILL SWITCH ACTIVATED — all trading halted")
    return {"status": "KILL_SWITCH_ACTIVE", "kill_switch": True}


@app.post("/api/kill-switch/reset")
async def reset_kill_switch():
    """Reset kill switch (re-enable trading)."""
    engine_control["kill_switch"] = False
    engine_control["paused"] = False

    control_file = Path(PROJECT_ROOT) / "data" / "engine_control.json"
    control_file.parent.mkdir(parents=True, exist_ok=True)
    control_file.write_text(
        json.dumps({"paused": False, "kill_switch": False,
                     "timestamp": datetime.now().isoformat()}),
        encoding="utf-8",
    )
    logger.info("Kill switch reset — trading re-enabled")
    return {"status": "TRADING", "kill_switch": False}


@app.get("/api/engine/status")
async def engine_status():
    """Get current engine control state."""
    running = _is_engine_running()
    pid = _engine_process.pid if running and _engine_process else None
    external = _find_external_engine_pids()

    # Warn if multiple engines detected (cause of LIVE/PAPER flickering)
    if _engine_process and external:
        logger.warning("DUAL ENGINE DETECTED: managed PID=%s + external PIDs=%s — "
                        "this causes LIVE/PAPER mode flickering!", pid, external)

    # If engine is managed by watchdog, report its PID
    watchdog_managed = bool(external and not _engine_process)
    effective_pid = pid or (external[0] if external else None)

    return {
        "running": running,
        "pid": effective_pid,
        "paused": engine_control["paused"],
        "kill_switch": engine_control["kill_switch"],
        "external_pids": external,
        "dual_engine_warning": bool(_engine_process and external),
        "watchdog_managed": watchdog_managed,
    }


@app.get("/api/engine/mode")
async def get_engine_mode():
    """Get current trading mode (LIVE or PAPER)."""
    env_path = Path(PROJECT_ROOT) / ".env"
    paper = True  # default safe
    if env_path.exists():
        content = env_path.read_text(encoding="utf-8")
        for line in content.splitlines():
            if line.strip().startswith("PAPER_TRADING="):
                val = line.split("=", 1)[1].strip().lower()
                paper = val in ("true", "1", "yes")
                break
    return {"mode": "PAPER" if paper else "LIVE", "paper_trading": paper}


@app.post("/api/engine/mode")
async def set_engine_mode(request: Request):
    """Switch between LIVE and PAPER mode. Requires engine restart to take effect."""
    body = await request.json()
    new_mode = body.get("mode", "").upper()
    if new_mode not in ("LIVE", "PAPER"):
        return {"error": "mode must be 'LIVE' or 'PAPER'"}

    paper_val = "true" if new_mode == "PAPER" else "false"

    # Update .env file
    env_path = Path(PROJECT_ROOT) / ".env"
    if env_path.exists():
        lines = env_path.read_text(encoding="utf-8").splitlines()
        updated = False
        for i, line in enumerate(lines):
            if line.strip().startswith("PAPER_TRADING="):
                lines[i] = f"PAPER_TRADING={paper_val}"
                updated = True
                break
        if not updated:
            lines.append(f"PAPER_TRADING={paper_val}")
        env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    else:
        env_path.write_text(f"PAPER_TRADING={paper_val}\n", encoding="utf-8")

    logger.info("Trading mode changed to %s (PAPER_TRADING=%s)", new_mode, paper_val)

    # If engine is running, stop it so it restarts with new mode
    if _is_engine_running():
        await stop_engine()
        # Brief delay then start with new mode
        import asyncio
        await asyncio.sleep(2)
        result = await start_engine(mode=new_mode.lower())
        return {"status": "RESTARTED", "mode": new_mode, "engine": result}

    return {"status": "MODE_SET", "mode": new_mode, "message": "Engine not running. Start it to trade in " + new_mode + " mode."}


@app.get("/api/strategy/info")
async def strategy_info():
    """Return CURRENTLY-DEPLOYED strategy config + walk-forward-validated metrics.

    Reads V15_CONFIG live (the actual deployed strategy), not V14 base.
    Backtest metrics reflect the directional gate at lb=3 / thr=0.5.
    """
    try:
        from scoring.config import V15_CONFIG
        cfg = V15_CONFIG
        gate_thr = cfg.get("directional_gate_threshold")
        gate_lb = cfg.get("directional_gate_lookback_days")
        has_gate = gate_thr is not None and gate_thr > 0

        # Strategy display name based on what's actually deployed
        if has_gate:
            name = f"V17_PROD_ONLY + Directional Gate (lb={gate_lb}, thr={gate_thr})"
        else:
            name = "V17_PROD_ONLY (Option B)"

        # Backtest metrics — match the deployed config exactly
        if has_gate and gate_lb == 3 and gate_thr == 0.5:
            backtest = {
                "return_multiple": 44.36,         # +Rs 86.72L / Rs 2L
                "profit_factor": 3.53,
                "win_rate": 52.2,
                "max_drawdown_pct": -10.1,
                "trades": 134,
                "walk_forward": "6/6 PnL wins, 6/6 PF wins, 0 catastrophic",
                "period": "Jul 2024 - Apr 2026 (21mo)",
                "lift_vs_no_gate": "+Rs 29.42L (+51%) PnL, PF 1.94 -> 3.53",
            }
        elif has_gate:
            backtest = {
                "return_multiple": 39.77,
                "profit_factor": 2.85,
                "win_rate": 50.0,
                "max_drawdown_pct": -8.8,
                "trades": 150,
                "walk_forward": "6/6 PnL wins, 6/6 PF wins",
                "period": "Jul 2024 - Apr 2026 (21mo)",
            }
        else:
            backtest = {
                "return_multiple": 29.65,
                "profit_factor": 1.94,
                "win_rate": 42.3,
                "max_drawdown_pct": -12.2,
                "trades": 194,
                "period": "Jul 2024 - Apr 2026 (21mo)",
            }

        return {
            "name": name,
            "bar_interval": cfg.get("bar_interval_min", 5),
            "entry_windows": cfg.get("entry_windows_bars", []),
            "avoid_windows": cfg.get("avoid_windows_bars", []),
            "avoid_days": cfg.get("avoid_days", []),
            "max_lots_cap": cfg.get("max_lots_cap", 27),
            "max_trades_per_day": cfg.get("max_trades_per_day", 7),
            "max_concurrent": cfg.get("max_concurrent", 3),
            "vix_floor": cfg.get("vix_floor"),
            "vix_ceil": cfg.get("vix_ceil"),
            "put_score_min": cfg.get("put_score_min", 5.0),
            "call_score_min": cfg.get("call_score_min", 6.0),
            "directional_gate": {
                "active": has_gate,
                "threshold_pct": gate_thr,
                "lookback_days": gate_lb,
                "description": (
                    f"Blocks PUT when {gate_lb}-day spot return > +{gate_thr}%; "
                    f"blocks CALL when {gate_lb}-day return < -{gate_thr}%."
                    if has_gate else "Disabled."
                ),
            },
            "features": {
                "directional_gate":   has_gate,
                "live_trade_monitor": True,  # Push #2 added it
                "dte_target_tuesday": True,  # Push #2 fixed Thu->Tue
                "psar_confluence":    cfg.get("use_psar_confluence", False),
                "vwap_filter":        cfg.get("use_vwap_filter", False),
                "squeeze_filter":     cfg.get("use_squeeze_filter", False),
                "theta_exit":         cfg.get("theta_exit_enabled", False),
            },
            "backtest": backtest,
        }
    except ImportError:
        return JSONResponse(status_code=500, content={"error": "Config not found"})


# ─────────────────────────────────────────────────────────────────────
# Strategy Research / Comparison Endpoints
# ─────────────────────────────────────────────────────────────────────

@app.get("/api/strategies/comparison")
async def strategies_comparison():
    """Side-by-side validation of deployed config evolution.

    Returns Option A (vix_ceil=35) vs Option B (vix_ceil=25) backtest
    results — full window + post-Sep + monthly + drawdown.
    Loaded from reports/oos/option_b_validation.json.
    """
    json_path = Path(PROJECT_ROOT) / "reports" / "oos" / "option_b_validation.json"
    if not json_path.exists():
        return JSONResponse(status_code=404, content={
            "error": "validation file not found",
            "hint": "Run: python -m backtesting.validate_option_b",
        })
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
        # Sanitize: JSON spec disallows inf/nan; convert to None
        def _sanitize(obj):
            if isinstance(obj, dict):
                return {k: _sanitize(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_sanitize(x) for x in obj]
            if isinstance(obj, float):
                if obj == float("inf") or obj == float("-inf") or obj != obj:
                    return None
            return obj
        data = _sanitize(data)
        # Compute deltas for the UI summary
        a, b = data.get("option_a", {}), data.get("option_b", {})
        summary = {
            "deployed": "OPTION B",
            "deployed_since": "2026-05-08",
            "deltas": {
                "full_pnl":       round(b.get("full", {}).get("pnl", 0) - a.get("full", {}).get("pnl", 0), 2),
                "post_sep_pnl":   round(b.get("post", {}).get("pnl", 0) - a.get("post", {}).get("pnl", 0), 2),
                "full_pf":        round(b.get("full", {}).get("pf", 0) - a.get("full", {}).get("pf", 0), 4),
                "post_sep_pf":    round(b.get("post", {}).get("pf", 0) - a.get("post", {}).get("pf", 0), 4),
                "max_dd":         round(b.get("max_dd", 0) - a.get("max_dd", 0), 2),
                "trades":         b.get("full", {}).get("n", 0) - a.get("full", {}).get("n", 0),
            },
        }
        return {"summary": summary, "option_a": a, "option_b": b}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/api/strategies/research")
async def strategies_research():
    """Catalog of strategies tested and their verdicts.

    Each entry summarizes a research experiment with the conclusion,
    so the user can see what's been validated, rejected, and why.
    """
    # Build deployed entry dynamically from current V15_CONFIG
    deployed_config = {}
    try:
        from scoring.config import V15_CONFIG
        deployed_config = {
            "avoid_days": V15_CONFIG.get("avoid_days"),
            "vix_floor": V15_CONFIG.get("vix_floor"),
            "vix_ceil": V15_CONFIG.get("vix_ceil"),
            "directional_gate_threshold": V15_CONFIG.get("directional_gate_threshold"),
            "directional_gate_lookback_days": V15_CONFIG.get("directional_gate_lookback_days"),
        }
    except ImportError:
        pass

    # Determine current strategy phase based on whether directional gate is active
    gate_thr = deployed_config.get("directional_gate_threshold")
    gate_lb = deployed_config.get("directional_gate_lookback_days")
    has_gate = gate_thr is not None and gate_thr > 0

    if has_gate and gate_lb == 3 and gate_thr == 0.5:
        # Push #2 amendment (refined directional gate)
        deployed_metrics = {
            "full_pnl": 8_672_000, "full_pf": 3.53,
            "post_sep_pnl": None, "post_sep_pf": None,
            "wr_full": 52.2, "max_dd_pct": -10.1, "n_full": 134,
            "walk_forward": "6/6 STRONG EDGE (6 windows)",
        }
        deployed_name = "V17_PROD_ONLY + Directional Gate (lb=3, thr=0.5)"
        deployed_since = "2026-05-10"
    elif has_gate:
        # Earlier directional gate config (lb=5, thr=1.0)
        deployed_metrics = {
            "full_pnl": 7_954_899, "full_pf": 2.85,
            "wr_full": 50.0, "max_dd_pct": -8.8, "n_full": 150,
            "walk_forward": "6/6 STRONG EDGE",
        }
        deployed_name = "V17_PROD_ONLY + Directional Gate (lb=5, thr=1.0)"
        deployed_since = "2026-05-09"
    else:
        # Option B baseline (no gate)
        deployed_metrics = {
            "full_pnl": 5_729_880, "full_pf": 1.94,
            "post_sep_pnl": 1_554_471, "post_sep_pf": 1.87,
            "wr_full": 42.3, "max_dd_pct": -12.2, "n_full": 194,
        }
        deployed_name = "V17_PROD_ONLY (Option B)"
        deployed_since = "2026-05-08"

    return {
        "deployed": [
            {
                "id": "v17_current",
                "name": deployed_name,
                "config": deployed_config,
                "metrics": deployed_metrics,
                "status": "deployed",
                "deployed_since": deployed_since,
            },
        ],
        "tested_rejected": [
            {
                "id": "trail_stop_tight_or_disable",
                "name": "Trail-stop fixes (TIGHT_TRAIL / NO_TRAIL / DELAYED / PE_ONLY)",
                "metrics": {"best_full_pf": 1.99, "best_full_pnl_delta": 264_000,
                            "walk_fwd_pnl_wins": "3/6", "walk_fwd_pf_wins": "4/6"},
                "verdict": "All MARGINAL — 3/6 PnL wins doesn't clear strict 4/6 bar. The trail leak is real but the fix only marginally lifts PnL.",
                "documented_in": "reports/oos/trail_asymmetric_test.log",
            },
            {
                "id": "regime_gate_variant_c",
                "name": "Variant C (use_v17_regime_gate=True)",
                "metrics": {"last_1y_pnl_lift": 1_385_000, "walk_fwd_pnl_wins": "3/6",
                            "catastrophic_windows": 1},
                "verdict": "Looked +6× on last-1Y, failed walk-forward (3/6, W5 catastrophic)",
                "documented_in": "reports/oos/walk_forward_variant_c.log",
            },
            {
                "id": "avoid_012_mon_tue_wed",
                "name": "avoid=[0,1,2] (Mon+Tue+Wed)",
                "metrics": {"last_1y_pf": 2.11, "pre_1y_oos_pnl_delta": -757_000},
                "verdict": "Post-Sep concentration artifact; OOS lost Rs 7.57L vs deployed",
                "documented_in": "reports/oos/validate_avoid_012_oos.log",
            },
            {
                "id": "vix_floor_10_lift",
                "name": "vix_floor=10 (max return)",
                "metrics": {"full_pnl": 5_962_497, "full_pf": 1.76,
                            "post_sep_pnl": 1_744_318, "post_sep_pf": 1.57},
                "verdict": "Higher gross PnL but worse PF; more noisy entries",
                "kept_alternative": "vix_floor=12 (Option A)",
            },
            {
                "id": "avoid_023_floor12",
                "name": "avoid=[0,2,3], vix_floor=12 (max quality)",
                "metrics": {"full_pf": 2.45, "post_sep_pnl": 1_598_145,
                            "post_sep_pf": 2.45, "wr_post_sep": 50.0,
                            "max_dd_pct": -50.5},
                "verdict": "Best risk profile but Tuesday post-Sep is expiry — blocking it kills the lift",
                "kept_alternative": "avoid=[0,2]",
            },
            {
                "id": "gap_classifier_skip_huge",
                "name": "Gap-classifier: skip days with |gap|>0.6%",
                "metrics": {"full_pnl_delta": 419_000, "post_sep_pnl_delta": -146_000},
                "verdict": "Lifts pre-Sep, degrades post-Sep — regime is shifting",
                "source": "OptionWise Auto-Router idea, inverted",
            },
            {
                "id": "optionwise_mean_reversion",
                "name": "OptionWise Mean Reversion strategy",
                "claimed": {"wr": 68.0},
                "actual": {"wr_honest": 28.8, "wr_their_method": 77.3, "pf_honest": 0.57},
                "verdict": "68% WR is daily-OHLC backtest artifact (47pp inflation)",
                "documented_in": "reports/oos/optionwise_mr_replication.log",
            },
            {
                "id": "intraday_short_strangle",
                "name": "Intraday short strangle (V17 filters)",
                "metrics": {"wr": 13.1, "pf": 0.02, "full_pnl": -498_792},
                "verdict": "Slippage round-trip > intraday theta gain",
                "documented_in": "reports/oos/short_premium_backtest.log",
            },
            {
                "id": "multi_day_short_strangle_wed",
                "name": "Multi-day short strangle (Wed entry, Tue expiry)",
                "metrics": {"wr": 68.0, "pf": 0.70, "full_pnl": -85_028},
                "verdict": "Literature WR confirmed (68%) but PF<1.0 — high-WR, negative-expectancy trap",
                "documented_in": "reports/oos/multi_day_short_premium.log",
            },
        ],
        "summary": {
            "total_experiments": 30,
            "deployed_winners": 1,
            "rejected": 9,
            "key_insight": (
                "Directional sanity gate (block PUT in uptrend, CALL in downtrend, "
                "lb=3 thr=0.5) is the breakthrough — first variant in 30+ tested to "
                "clear strict walk-forward (6/6 PnL+PF wins). 21mo: PF 1.94 -> 3.53, "
                "WR 42.3% -> 52.2%, DD improved. Mechanism: rescues low-vol uptrend "
                "regimes (e.g. June 2025) where V14 entries systematically fade trends."
            ),
        },
    }


@app.get("/api/ai/brain")
async def get_ai_brain():
    """Return latest AI Market Brain analysis."""
    brain_file = Path(PROJECT_ROOT) / "data" / "claude_brain.json"
    if brain_file.exists():
        try:
            return JSONResponse(json.loads(brain_file.read_text(encoding="utf-8")))
        except Exception:
            pass
    return JSONResponse({"enabled": False, "analysis": None})


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "broker_connected": kite_state["connected"],
    }


@app.get("/api/config")
async def get_config():
    """Return system configuration."""
    return {
        "indices": INDEX_CONFIG,
        "stt_rates": STT_RATES,
        "strategies": [
            "short_straddle", "delta_neutral", "bull_put_spread",
            "iron_condor", "pairs_trade", "ddqn_agent",
        ],
        "capital": 1_000_000,
    }


@app.get("/api/paper_trading/results")
async def get_paper_trading_results():
    """Load and return 15-day paper trading simulation results."""
    results_path = Path(__file__).parent.parent / "data" / "paper_trading_15day_results.json"
    if not results_path.exists():
        return JSONResponse(status_code=404, content={"error": "No paper trading results found. Run paper_trading_15day_fast.py first."})
    try:
        with open(results_path, "r") as f:
            data = json.load(f)
        return data
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/api/paper_trading/run")
async def run_paper_trading_sim():
    """Run the 15-day paper trading simulation and return results."""
    logger.info("Starting 15-day paper trading simulation...")
    t_start = time.monotonic()
    try:
        def _run():
            from backtesting.paper_trading_15day_fast import run_15_day_paper_trading
            return run_15_day_paper_trading()

        results = await asyncio.to_thread(_run)
        elapsed = time.monotonic() - t_start
        logger.info("Paper trading simulation complete in %.1fs", elapsed)
        return {"status": "complete", "elapsed_seconds": round(elapsed, 2), "results": results}
    except Exception as e:
        logger.error("Paper trading simulation failed: %s", e, exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/api/paper_trading/real_data/results")
async def get_real_data_paper_trading_results():
    """Load and return 6-month real-data paper trading results."""
    results_path = Path(__file__).parent.parent / "data" / "paper_trading_realdata_results.json"
    if not results_path.exists():
        return JSONResponse(status_code=404, content={"error": "No real-data results found. Run paper_trading_real_data.py first."})
    try:
        with open(results_path, "r") as f:
            data = json.load(f)
        return data
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/api/paper_trading/real_data/run")
async def run_real_data_paper_trading_sim():
    """Run the 6-month real-data paper trading simulation and return results."""
    logger.info("Starting 6-month real-data paper trading simulation...")
    t_start = time.monotonic()
    try:
        def _run():
            from backtesting.paper_trading_real_data import run_real_data_paper_trading
            return run_real_data_paper_trading()

        results = await asyncio.to_thread(_run)
        elapsed = time.monotonic() - t_start
        logger.info("Real-data paper trading simulation complete in %.1fs", elapsed)
        return {"status": "complete", "elapsed_seconds": round(elapsed, 2), "results": results}
    except Exception as e:
        logger.error("Real-data paper trading simulation failed: %s", e, exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/api/model_comparison/results")
async def get_model_comparison_results():
    """Load and return cached model comparison results."""
    results_path = Path(__file__).parent.parent / "data" / "model_comparison_results.json"
    if not results_path.exists():
        return JSONResponse(status_code=404, content={"error": "No model comparison results found. Run comparison first."})
    try:
        with open(results_path, "r") as f:
            data = json.load(f)
        return data
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/api/model_comparison/run")
async def run_model_comparison_sim():
    """Run all model variants across multiple date periods and return results."""
    logger.info("Starting model comparison (all variants x 3 periods)...")
    t_start = time.monotonic()
    try:
        def _run():
            from backtesting.model_comparison import run_comparison
            return run_comparison()

        results = await asyncio.to_thread(_run)
        elapsed = time.monotonic() - t_start
        logger.info("Model comparison complete in %.1fs", elapsed)
        return {"status": "complete", "elapsed_seconds": round(elapsed, 2), "results": results}
    except Exception as e:
        logger.error("Model comparison failed: %s", e, exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/api/backtest/compare")
async def run_comparison(days: int = 21):
    """Run all 6 strategies and return comparative results."""
    days = min(max(days, 5), 63)  # Clamp between 5-63 days
    logger.info("Starting multi-strategy backtest comparison (days=%d)...", days)
    t_start = time.monotonic()

    try:
        from dashboard.strategies import run_all_strategies, generate_comparison_report

        def _run_backtest():
            nd = generate_synthetic_ohlcv("NIFTY", days=days, interval_minutes=15, base_price=24000, seed=42)
            bnd = generate_synthetic_ohlcv("BANKNIFTY", days=days, interval_minutes=15, base_price=51000, seed=43)
            res = run_all_strategies(data=nd, initial_capital=1_000_000, nifty_data=nd, banknifty_data=bnd)
            rep = generate_comparison_report(res)
            return res, rep

        results, report = await asyncio.to_thread(_run_backtest)

        # Convert for JSON serialization
        comparison = {}
        for name, r in results.items():
            res = r["result"]
            comparison[name] = {
                "net_pnl": round(res.net_pnl, 2),
                "win_rate": round(res.win_rate * 100, 1),
                "sharpe_ratio": round(res.sharpe_ratio, 4),
                "profit_factor": round(res.profit_factor, 2),
                "max_drawdown_pct": round(res.max_drawdown_pct * 100, 2),
                "total_trades": res.total_trades,
                "avg_hold_minutes": round(res.avg_hold_minutes, 1),
                "transaction_costs": round(res.total_transaction_costs, 2),
                "slippage_costs": round(res.total_slippage_costs, 2),
                "kill_switch_triggers": res.kill_switch_triggers,
                "equity_curve": [round(x, 2) for x in res.equity_curve[::max(1, len(res.equity_curve)//200)]],
                "daily_pnl": [round(x, 2) for x in res.daily_pnl],
                "trades": [
                    {
                        "entry_time": str(t.entry_time),
                        "exit_time": str(t.exit_time),
                        "symbol": t.symbol,
                        "side": t.side,
                        "qty": t.quantity,
                        "entry_price": round(t.entry_price, 2),
                        "exit_price": round(t.exit_price, 2),
                        "net_pnl": round(t.net_pnl, 2),
                        "costs": round(t.transaction_costs, 2),
                    }
                    for t in (res.trades[:100] if res.trades else [])
                ],
                "metadata": r.get("metadata", {}),
            }

        elapsed = time.monotonic() - t_start
        logger.info("Comparison complete | strategies=%d elapsed=%.2fs", len(comparison), elapsed)

        return {
            "comparison": comparison,
            "report": report,
            "elapsed_seconds": round(elapsed, 2),
        }
    except Exception as e:
        logger.error("Backtest comparison failed: %s", e, exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(e)})


# ── Live Trading Orchestrator Endpoints ──────────────────────────────────────

live_state = {
    "orchestrator": None,
    "running": False,
    "task": None,
}


@app.post("/api/live/start")
async def start_live_trading(
    capital: float = 200000.0,
    strategies: str = "learned_rules",
    mode: str = "paper",
    symbol: str = "NIFTY",
):
    """Start the fully automated live trading orchestrator.

    All 3 agents run simultaneously. MarketAnalyzer (12 indicators)
    decides strategy, timing, and direction automatically.

    mode='paper' uses the paper broker (safe testing).
    mode='live' uses the real Kite Connect broker (requires login).
    symbol: NIFTY, BANKNIFTY, or FINNIFTY.
    """
    if live_state["running"]:
        return {"error": "Live trading already running"}

    from orchestrator.live_orchestrator import LiveTradingOrchestrator

    strategy_list = [s.strip() for s in strategies.split(",")]

    if mode == "live":
        if not kite_state["connected"]:
            return JSONResponse(status_code=401, content={
                "error": "Broker not connected. Login at /login/zerodha first."
            })
        broker = kite_state["kite"]
    else:
        from backtesting.paper_trading import PaperTradingBroker
        broker = PaperTradingBroker(initial_capital=capital)

    async def event_callback(event):
        await broadcast(event)

    orchestrator = LiveTradingOrchestrator(
        broker=broker,
        capital=capital,
        strategies=strategy_list,
        callback=event_callback,
        symbol=symbol,
    )

    live_state["orchestrator"] = orchestrator
    live_state["running"] = True

    async def _run():
        try:
            await orchestrator.run()
        except Exception as e:
            logger.error("Orchestrator error: %s", e, exc_info=True)
        finally:
            live_state["running"] = False

    live_state["task"] = asyncio.create_task(_run())

    return {
        "status": "started",
        "mode": mode,
        "symbol": symbol,
        "capital": capital,
        "strategies": strategy_list,
    }


@app.post("/api/live/stop")
async def stop_live_trading():
    """Stop the live trading orchestrator (squares off all positions)."""
    orch = live_state.get("orchestrator")
    if orch is None or not live_state["running"]:
        return {"error": "No live session running"}

    await orch.shutdown()
    live_state["running"] = False
    return {"status": "stopped"}


@app.get("/api/live/status")
async def live_trading_status():
    """Get live trading orchestrator status."""
    orch = live_state.get("orchestrator")
    if orch is None:
        return {"running": False}
    return {
        "running": live_state["running"],
        **orch.get_status(),
    }


# ── Paper Trading Session Endpoints ─────────────────────────────────────────

@app.post("/api/session/start")
async def start_session(
    strategy: str = "short_straddle",
    capital: float = 1_000_000,
    speed: float = 50,
    market_trend: str = "sideways",
):
    """Start a live paper trading session."""
    if session_state["running"]:
        return {"error": "Session already running"}

    session_state["running"] = True
    session_state["strategy"] = strategy
    session_state["start_time"] = datetime.now().isoformat()
    session_state["events"] = []

    asyncio.create_task(_run_session(strategy, capital, speed, market_trend))
    return {"status": "started", "strategy": strategy}


@app.post("/api/session/stop")
async def stop_session():
    """Stop the running session."""
    session_state["running"] = False
    await broadcast({"type": "session_stopped"})
    return {"status": "stopped"}


@app.get("/api/session/status")
async def session_status():
    return session_state


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time dashboard updates."""
    await websocket.accept()
    connected_clients.append(websocket)
    logger.info("WebSocket client connected | total=%d", len(connected_clients))

    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)

            if msg.get("action") == "start_session":
                if not session_state["running"]:
                    asyncio.create_task(_run_session(
                        msg.get("strategy", "short_straddle"),
                        msg.get("capital", 1_000_000),
                        msg.get("speed", 50),
                        msg.get("market_trend", "sideways"),
                    ))

            elif msg.get("action") == "stop_session":
                session_state["running"] = False

            elif msg.get("action") == "run_comparison":
                asyncio.create_task(_run_comparison_ws())

    except WebSocketDisconnect:
        connected_clients.remove(websocket)
        logger.info("WebSocket client disconnected | remaining=%d", len(connected_clients))


async def _run_comparison_ws():
    """Run comparison and stream results via WebSocket."""
    await broadcast({"type": "comparison_started"})
    try:
        from dashboard.strategies import run_all_strategies, generate_comparison_report

        def _run_bt():
            nd = generate_synthetic_ohlcv("NIFTY", days=21, interval_minutes=15, base_price=24000, seed=42)
            bnd = generate_synthetic_ohlcv("BANKNIFTY", days=21, interval_minutes=15, base_price=51000, seed=43)
            r = run_all_strategies(nd, 1_000_000, nd, bnd)
            return r, generate_comparison_report(r)

        results, report = await asyncio.to_thread(_run_bt)

        comparison = {}
        for name, r in results.items():
            res = r["result"]
            comparison[name] = {
                "net_pnl": round(res.net_pnl, 2),
                "win_rate": round(res.win_rate * 100, 1),
                "sharpe_ratio": round(res.sharpe_ratio, 4),
                "profit_factor": round(res.profit_factor, 2),
                "max_drawdown_pct": round(res.max_drawdown_pct * 100, 2),
                "total_trades": res.total_trades,
                "transaction_costs": round(res.total_transaction_costs, 2),
                "equity_curve": [round(x, 2) for x in res.equity_curve[::max(1, len(res.equity_curve)//200)]],
                "daily_pnl": [round(x, 2) for x in res.daily_pnl],
            }

        await broadcast({
            "type": "comparison_complete",
            "comparison": comparison,
            "report": report,
        })
    except Exception as e:
        await broadcast({"type": "comparison_error", "error": str(e)})


async def _run_session(strategy: str, capital: float, speed: float, market_trend: str):
    """Run a simulated trading session and broadcast events."""
    logger.info("Session starting | strategy=%s capital=%.0f speed=%.0fx trend=%s",
                strategy, capital, speed, market_trend)

    session_state["running"] = True

    try:
        from dashboard.session_runner import SessionRunner, generate_intraday_data

        data = generate_intraday_data(
            base_price=24000.0,
            volatility=0.14,
            trend=market_trend,
            seed=None,
        )

        runner = SessionRunner(strategy, capital, speed_multiplier=speed)

        async def event_callback(event):
            session_state["events"].append(event)
            if len(session_state["events"]) > 1000:
                session_state["events"] = session_state["events"][-1000:]
            await broadcast(event)

        await runner.run_session(data, event_callback)

    except Exception as e:
        logger.error("Session error: %s", e, exc_info=True)
        await broadcast({"type": "error", "message": str(e)})
    finally:
        session_state["running"] = False
        logger.info("Session ended")


@app.post("/api/session/run_all")
async def run_all_strategies_session(
    capital: float = 1_000_000,
    speed: float = 200,
    market_trend: str = "sideways",
):
    """Run all 6 strategies on the same data and return comparative results."""
    if session_state["running"]:
        return {"error": "Session already running"}

    session_state["running"] = True
    session_state["events"] = []

    try:
        from dashboard.session_runner import (
            run_multi_strategy_session,
            generate_intraday_data,
        )

        data = generate_intraday_data(
            base_price=24000.0,
            volatility=0.14,
            trend=market_trend,
            seed=None,
        )

        strategies = [
            "short_straddle", "delta_neutral", "bull_put_spread",
            "iron_condor", "pairs_trade", "ddqn_agent",
        ]

        async def event_callback(event):
            session_state["events"].append(event)
            if len(session_state["events"]) > 2000:
                session_state["events"] = session_state["events"][-2000:]
            await broadcast(event)

        results = await run_multi_strategy_session(
            data, strategies, capital, speed, event_callback,
        )

        return {
            "status": "complete",
            "results": results,
            "market_trend": market_trend,
        }
    except Exception as e:
        logger.error("Run-all error: %s", e, exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        session_state["running"] = False


# ── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("DASHBOARD_PORT", "8501"))
    logger.info("Starting Trading Dashboard on http://localhost:%d", port)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
