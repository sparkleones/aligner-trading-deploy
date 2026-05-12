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


@app.get("/api/health/full")
async def health_full():
    """Comprehensive system health: every subsystem status + counters + errors.

    Aggregates from kite_state, engine state file, .env config, and process info.
    Dashboard uses this to surface "why isn't anything happening" without asking.
    """
    import time
    import psutil

    now = time.time()
    state_path = Path(PROJECT_ROOT) / "data" / "dashboard_state.json"
    engine_state = {}
    state_age_sec = None
    if state_path.exists():
        try:
            engine_state = json.loads(state_path.read_text(encoding="utf-8"))
            mtime = state_path.stat().st_mtime
            state_age_sec = round(now - mtime, 1)
        except Exception:
            pass

    # ── Engine subsystem ──
    engine_pids = _find_external_engine_pids()
    engine_running = bool(engine_pids)
    engine_mem_mb = 0
    if engine_pids:
        try:
            engine_mem_mb = round(psutil.Process(engine_pids[0]).memory_info().rss / 1024 / 1024, 0)
        except Exception:
            pass

    sys_status = (engine_state.get("system") or {}).get("status", "UNKNOWN")
    bars_processed = (engine_state.get("system") or {}).get("bars_processed", 0)
    market = engine_state.get("market") or {}

    # ── Kite subsystem ──
    kite_subsystem = {
        "connected": kite_state.get("connected", False),
        "user": (kite_state.get("user_profile") or {}).get("user_name", ""),
        "user_id": (kite_state.get("user_profile") or {}).get("user_id", ""),
        "api_key_set": bool(kite_state.get("api_key")),
        "access_token_set": bool(kite_state.get("access_token")),
    }
    if kite_state.get("connected") and kite_state.get("kite"):
        try:
            t0 = time.time()
            margins = kite_state["kite"].margins()
            kite_subsystem["api_response_ms"] = round((time.time() - t0) * 1000, 0)
            eq = margins.get("equity", {})
            kite_subsystem["available_margin"] = eq.get("available", {}).get("live_balance", 0)
            kite_subsystem["used_margin"] = eq.get("utilised", {}).get("debits", 0)
            kite_subsystem["api_status"] = "ok"
        except Exception as e:
            kite_subsystem["api_status"] = "error"
            kite_subsystem["api_error"] = str(e)[:200]

    # ── Data feeds ──
    data_feeds = {
        "nifty_spot": {
            "value": market.get("spot_price"),
            "stale": (state_age_sec is None) or (state_age_sec > 30 and sys_status == "TRADING"),
        },
        "vix": {
            "value": market.get("vix"),
            "stale": (state_age_sec is None) or (state_age_sec > 30 and sys_status == "TRADING"),
        },
        "state_age_sec": state_age_sec,
    }

    # ── AI Brain ──
    # Display the CONFIGURED (top-priority) provider+model first, NOT what was
    # stored in stale claude_brain.json from a previous run. Counters (calls,
    # cost) can come from brain.json since they're per-session totals.
    ai_subsystem = {"enabled": False, "provider": None, "model": None}
    configured = []
    try:
        from orchestrator.claude_market_brain import PROVIDERS, PROVIDER_PRIORITY
        for pname in PROVIDER_PRIORITY:
            p = PROVIDERS.get(pname, {})
            if os.getenv(p.get("env_key", "")):
                configured.append({
                    "provider": pname,
                    "model": p.get("model"),
                    "cost_per_1m": p.get("cost_per_1m_input"),
                })
        if configured:
            # Top-priority configured provider = what the engine WILL use
            ai_subsystem["enabled"] = True
            ai_subsystem["provider"] = configured[0]["provider"]
            ai_subsystem["model"] = configured[0]["model"]
            ai_subsystem["cost_per_1m"] = configured[0]["cost_per_1m"]
        ai_subsystem["configured_providers"] = configured
    except Exception:
        pass

    # Layer in session counters (calls, cost) from brain.json
    brain_file = Path(PROJECT_ROOT) / "data" / "claude_brain.json"
    if brain_file.exists():
        try:
            brain = json.loads(brain_file.read_text(encoding="utf-8"))
            ai_subsystem["calls_today"] = brain.get("total_calls", 0)
            ai_subsystem["cost_usd_today"] = brain.get("total_cost_usd", 0.0)
            ai_subsystem["last_call_age_sec"] = (
                round(now - brain.get("last_analysis_time", 0), 1)
                if brain.get("last_analysis_time") else None
            )
            ai_subsystem["last_error"] = brain.get("last_error")
            # If the running engine is using a DIFFERENT model than configured,
            # surface that as a warning (means engine needs restart to pick up
            # new PROVIDERS config)
            engine_model = brain.get("model")
            if (engine_model and ai_subsystem.get("model")
                    and engine_model != ai_subsystem["model"]):
                ai_subsystem["engine_running_model"] = engine_model
                ai_subsystem["model_mismatch"] = True
        except Exception:
            pass

    # ── Telegram ──
    telegram_subsystem = {
        "enabled": bool(os.getenv("TELEGRAM_BOT_TOKEN")) and bool(os.getenv("TELEGRAM_CHAT_ID")),
        "chat_id": os.getenv("TELEGRAM_CHAT_ID", ""),
    }

    # ── Position / order counters from engine state ──
    positions = engine_state.get("positions", {}).get("open", [])
    risk = engine_state.get("risk", {})
    today_pnl = (engine_state.get("pnl") or {}).get("gross") or (engine_state.get("pnl") or {}).get("total") or 0

    # ── Recent log tail (last 5 ERROR/WARN entries) ──
    log_path = Path(PROJECT_ROOT) / "data" / "daemon_feed.log"
    recent_alerts = []
    if log_path.exists():
        try:
            lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-200:]
            for line in reversed(lines):
                low = line.lower()
                if "error" in low or "warning" in low or "warn" in low:
                    recent_alerts.append(line[:200])
                if len(recent_alerts) >= 5:
                    break
        except Exception:
            pass

    # ── Disk / memory ──
    try:
        disk = psutil.disk_usage(str(PROJECT_ROOT))
        disk_free_gb = round(disk.free / 1024 / 1024 / 1024, 1)
        disk_used_pct = round(disk.percent, 1)
    except Exception:
        disk_free_gb = None
        disk_used_pct = None

    # ── Config validation ──
    config_subsystem = {
        "kite_api_key":     bool(os.getenv("KITE_API_KEY") or os.getenv("BROKER_API_KEY")),
        "kite_api_secret":  bool(os.getenv("KITE_API_SECRET") or os.getenv("BROKER_API_SECRET")),
        "zerodha_user_id":  bool(os.getenv("ZERODHA_USER_ID") or os.getenv("BROKER_USER_ID")),
        "zerodha_password": bool(os.getenv("ZERODHA_PASSWORD") or os.getenv("BROKER_PASSWORD")),
        "totp_secret":      bool(os.getenv("ZERODHA_TOTP_SECRET") or os.getenv("BROKER_TOTP_SECRET")),
        "anthropic_key":    bool(os.getenv("ANTHROPIC_API_KEY")),
        "groq_key":         bool(os.getenv("GROQ_API_KEY")),
        "gemini_key":       bool(os.getenv("GEMINI_API_KEY")),
        "telegram":         bool(os.getenv("TELEGRAM_BOT_TOKEN")) and bool(os.getenv("TELEGRAM_CHAT_ID")),
    }

    return {
        "ts": now,
        "engine": {
            "running": engine_running,
            "pid": engine_pids[0] if engine_pids else None,
            "status": sys_status,
            "bars_processed": bars_processed,
            "state_age_sec": state_age_sec,
            "memory_mb": engine_mem_mb,
        },
        "kite": kite_subsystem,
        "data_feeds": data_feeds,
        "ai_brain": ai_subsystem,
        "telegram": telegram_subsystem,
        "trade_stats": {
            "today_pnl_inr": today_pnl,
            "win_rate": risk.get("win_rate", 0),
            "winners": risk.get("winners", 0),
            "losers": risk.get("losers", 0),
            "profit_factor": risk.get("profit_factor", 0),
            "open_positions": len(positions),
        },
        "config": config_subsystem,
        "system": {
            "disk_free_gb": disk_free_gb,
            "disk_used_pct": disk_used_pct,
        },
        "alerts": recent_alerts,
    }


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

        # Backtest metrics — sized for NIFTY lot=65 (SEBI Oct 2025 revision).
        # Earlier numbers were computed with stale lot=75; rescaled by 65/75=0.867.
        # PF/WR/MaxDD% are ratios, unchanged. Absolute PnL and DD scale linearly.
        if has_gate and gate_lb == 3 and gate_thr == 0.5:
            backtest = {
                "return_multiple": 38.58,                # +Rs 75.16L / Rs 2L cap
                "profit_factor": 3.53,                   # ratio — unchanged
                "win_rate": 52.2,                        # ratio — unchanged
                "max_drawdown_pct": -10.1,               # ratio — unchanged
                "max_dd_inr": -439_000,                  # -Rs 4.39L
                "trades": 134,
                "walk_forward": "6/6 PnL wins, 6/6 PF wins, 0 catastrophic",
                "period": "Jul 2024 - Apr 2026 (21mo)",
                "lift_vs_no_gate": "+Rs 25.5L (+51%) PnL, PF 1.94 -> 3.53",
                "lot_size_note": "NIFTY lot=65 (SEBI Oct 2025 revision; prior lot=75 numbers were 15% higher in INR)",
            }
        elif has_gate:
            backtest = {
                "return_multiple": 34.46,
                "profit_factor": 2.85,
                "win_rate": 50.0,
                "max_drawdown_pct": -8.8,
                "max_dd_inr": -383_000,
                "trades": 150,
                "walk_forward": "6/6 PnL wins, 6/6 PF wins",
                "period": "Jul 2024 - Apr 2026 (21mo)",
            }
        else:
            backtest = {
                "return_multiple": 25.69,
                "profit_factor": 1.94,
                "win_rate": 42.3,
                "max_drawdown_pct": -12.2,
                "max_dd_inr": -527_000,
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

    # All PnL/DD values rescaled for NIFTY lot=65 (SEBI Oct 2025 revision).
    # Earlier metrics at lot=75 were 15% higher in INR; PF/WR/DD% unchanged.
    if has_gate and gate_lb == 3 and gate_thr == 0.5:
        # Push #2 amendment (refined directional gate)
        deployed_metrics = {
            "full_pnl": 7_516_000, "full_pf": 3.53,    # was 8_672_000 at lot=75
            "post_sep_pnl": None, "post_sep_pf": None,
            "wr_full": 52.2, "max_dd_pct": -10.1, "n_full": 134,
            "walk_forward": "6/6 STRONG EDGE (6 windows)",
            "lot_size": 65,
        }
        deployed_name = "V17_PROD_ONLY + Directional Gate (lb=3, thr=0.5)"
        deployed_since = "2026-05-10"
    elif has_gate:
        # Earlier directional gate config (lb=5, thr=1.0)
        deployed_metrics = {
            "full_pnl": 6_894_245, "full_pf": 2.85,    # was 7_954_899 at lot=75
            "wr_full": 50.0, "max_dd_pct": -8.8, "n_full": 150,
            "walk_forward": "6/6 STRONG EDGE",
            "lot_size": 65,
        }
        deployed_name = "V17_PROD_ONLY + Directional Gate (lb=5, thr=1.0)"
        deployed_since = "2026-05-09"
    else:
        # Option B baseline (no gate)
        deployed_metrics = {
            "full_pnl": 4_965_896, "full_pf": 1.94,    # was 5_729_880 at lot=75
            "post_sep_pnl": 1_347_208, "post_sep_pf": 1.87,
            "wr_full": 42.3, "max_dd_pct": -12.2, "n_full": 194,
            "lot_size": 65,
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
                "lb=3 thr=0.5) is the breakthrough — first of 30+ tested variants to "
                "clear strict walk-forward (6/6 PnL+PF wins). 21mo at NIFTY lot=65: "
                "PnL +Rs 75.16L, PF 3.53, WR 52.2%. Live forward expectation: "
                "+Rs 7-13L/yr on Rs 22.9K capital (1 ATM lot/trade, SEBI compliant)."
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


# ── Stock Screener Endpoints ─────────────────────────────────────────────────

@app.get("/api/screener/signals")
async def screener_signals(capital: float = 100000.0, n_picks: int = 2, force: int = 0):
    """
    Return today's top 2-3 stock picks with full trade plans.
    Cached for 6 hours; pass force=1 to refresh.
    """
    try:
        from screener.live_signal import generate_signals
        n_picks = max(1, min(int(n_picks), 5))
        payload = generate_signals(
            account_capital=float(capital),
            n_picks=n_picks,
            force_refresh=bool(force),
        )
        return payload
    except Exception as e:
        logger.error("Screener signals error: %s", e, exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/api/screener/backtest_stats")
async def screener_backtest_stats():
    """Return the cached backtest stats from the most recent run."""
    try:
        path = Path(__file__).resolve().parent.parent / "reports" / "screener" / "backtest_stats.json"
        if not path.exists():
            return {"error": "no backtest stats; run `python -m screener.run_backtest`"}
        with open(path, "r") as f:
            stats = json.load(f)
        # Layer in walk-forward summary
        walkforward = [
            {"window": "2022-01 to 2023-06", "screener_cagr": 0.1591, "nifty_cagr": 0.0588, "sharpe": 1.28, "max_dd": -0.1864},
            {"window": "2022-07 to 2023-12", "screener_cagr": 0.4787, "nifty_cagr": 0.2402, "sharpe": 2.66, "max_dd": -0.1856},
            {"window": "2023-01 to 2024-06", "screener_cagr": 0.5932, "nifty_cagr": 0.2050, "sharpe": 2.35, "max_dd": -0.1427},
            {"window": "2024-01 to 2025-06", "screener_cagr": 0.2135, "nifty_cagr": 0.1130, "sharpe": 1.03, "max_dd": -0.2260},
            {"window": "2024-07 to 2026-04", "screener_cagr": -0.0601, "nifty_cagr": -0.0033, "sharpe": -0.30, "max_dd": -0.2424},
        ]
        return {
            "stats": stats,
            "walkforward": walkforward,
            "config_label": "HTR Monthly 2pk default",
            "benchmark_nifty_cagr": 0.0868,
            "verdict": (
                "Beat NIFTY in 4/5 walk-forward windows. Lagged in the most "
                "recent flat-NIFTY window. Recommend paper-tracking before "
                "deploying real capital."
            ),
        }
    except Exception as e:
        logger.error("Screener backtest_stats error: %s", e, exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/api/screener/universe")
async def screener_universe():
    """List the F&O screening universe."""
    try:
        from screener.universe import get_universe, SECTOR_MAP
        universe = get_universe()
        return {
            "universe": universe,
            "count": len(universe),
            "by_sector": {sec: [s for s in universe if SECTOR_MAP.get(s) == sec]
                          for sec in set(SECTOR_MAP.values())},
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/api/screener/picks_v2")
async def screener_picks_v2(
    capital: float = 100000.0,
    tier: str = "BLEND",
    enable_ai: int = 1,
    force: int = 0,
):
    """
    V2 screener producer. Combines:
      - LARGE: composite (stage2 + breakout)  -- 70% allocation
      - MID:   mean_reversion (Connors RSI-2) -- 30% allocation
      - AI agent review of every pick
      - Concrete entry/SL/target/qty with 42-day hold

    Params:
      capital:    total equity capital in Rs (default 100000)
      tier:       LARGE | MID | BLEND (default BLEND = 70/30)
      enable_ai:  1 = run LLM review, 0 = skip (faster)
      force:      1 = bypass 6h cache
    """
    try:
        from screener.live_picks_v2 import generate
        # tier override: SOLO LARGE or SOLO MID
        n_large = 2
        n_mid = 1
        if tier.upper() == "LARGE":
            n_large = 3
            n_mid = 0
        elif tier.upper() == "MID":
            n_large = 0
            n_mid = 3
        payload = generate(
            capital=float(capital),
            n_large=n_large,
            n_mid=n_mid,
            enable_ai=bool(int(enable_ai)),
            force_refresh=bool(int(force)),
        )
        return payload
    except Exception as e:
        logger.error("Screener picks_v2 error: %s", e, exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/api/screener/strategy_comparison")
async def screener_strategy_comparison():
    """Return strategy_comparison.csv as JSON rows."""
    try:
        path = Path(__file__).resolve().parent.parent / "reports" / "screener" / "strategy_comparison.csv"
        if not path.exists():
            return {"error": "run 'python -m screener.compare_all_strategies' first"}
        import csv as _csv
        rows = []
        with open(path, "r", newline="") as f:
            reader = _csv.DictReader(f)
            for r in reader:
                rows.append(r)
        return {"rows": rows, "count": len(rows)}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/api/screener/train_test_findings")
async def screener_train_test():
    """Return train/test backtest findings."""
    try:
        path = Path(__file__).resolve().parent.parent / "reports" / "screener" / "train_test_findings.json"
        if not path.exists():
            return {"error": "no findings yet"}
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# ── Market Timing + DCA Endpoints ────────────────────────────────────────────

@app.get("/api/agent_team/review")
async def agent_team_review(symbol: str, sector: str = "OTHER"):
    """
    Run the full 7-agent research team on any NSE symbol on demand.
    Returns each specialist's report + the PM's synthesis.
    """
    try:
        from agent_team import ResearchTeam
        from agent_team.context_builder import build_context
        from screener.data_loader import load_history
        from screener.fundamentals import fetch_fundamentals
        from screener.market_timing_analyzer import _fetch_nifty
        from screener.universe import get_sector as _get_sector
        from screener.universe_extended import LARGE_CAP, MID_CAP

        sym = symbol.upper().strip()
        # Resolve sector via the existing map if user didn't supply
        if sector == "OTHER" or not sector:
            sector = _get_sector(sym)

        df = load_history(sym, period="3y", use_cache=True)
        if df.empty or len(df) < 252:
            return {"error": f"insufficient history for {sym}"}
        fund = fetch_fundamentals(sym)
        try:
            nifty_h = _fetch_nifty()
        except Exception:
            nifty_h = None

        # Same-sector peers for sector context
        peers = {}
        for s in LARGE_CAP + MID_CAP:
            if _get_sector(s) == sector and s != sym:
                pdf = load_history(s, period="3y", use_cache=True)
                if not pdf.empty and len(pdf) >= 252:
                    peers[s] = pdf
                if len(peers) >= 12:
                    break

        macro_extra = {}
        if nifty_h is not None and not nifty_h.empty:
            c = nifty_h["Close"]
            close_n = float(c.iloc[-1])
            ma_200 = float(c.rolling(200).mean().iloc[-1]) if len(c) >= 200 else close_n
            ma_50  = float(c.rolling(50).mean().iloc[-1]) if len(c) >= 50 else close_n
            high_252 = float(nifty_h["High"].tail(252).max())
            macro_extra = {
                "nifty_close": close_n,
                "nifty_dist_high_pct": (close_n/high_252 - 1.0) if high_252 > 0 else 0,
                "nifty_dist_200dma_pct": (close_n/ma_200 - 1.0) if ma_200 > 0 else 0,
                "above_200dma": close_n > ma_200,
                "golden_cross": ma_50 > ma_200,
            }

        ctx = build_context(
            symbol=sym, sector=sector, history=df,
            fundamentals=fund, nifty_history=nifty_h,
            sector_peers=peers, macro_extra=macro_extra,
        )

        team = ResearchTeam(prefer_fast=True, enable_llm_arbitration=False)
        tv = team.review(sym, ctx)
        return {
            "symbol": sym, "sector": sector,
            "final_action": tv.final_action,
            "final_score": tv.final_score,
            "confidence": tv.confidence,
            "suggested_qty_mult": tv.suggested_qty_mult,
            "hold_days": tv.hold_days,
            "coordinator_note": tv.coordinator_note,
            "reports": [
                {"agent": r.agent_name, "score": r.score, "verdict": r.verdict,
                 "confidence": r.confidence, "flags": r.flags,
                 "one_liner": r.one_liner, "error": r.error,
                 "provider": r.provider_used}
                for r in tv.reports
            ],
            "context_summary": {
                "technical": ctx.get("technical", {}),
                "risk": ctx.get("risk", {}),
                "sector_metrics": ctx.get("sector_metrics", {}),
                "events": ctx.get("events", {}),
            },
        }
    except Exception as e:
        logger.error("agent_team review error: %s", e, exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/api/agent_team/options_review")
async def agent_team_options_review():
    """
    Run the OPTIONS research team on the current NIFTY F&O state.
    Returns strategy recommendation + all specialist reports.
    """
    try:
        from agent_team.options import OptionsResearchTeam
        from agent_team.options.context_builder import build_options_context
        from screener.market_timing_analyzer import _fetch_nifty, _fetch_vix

        nifty_h = _fetch_nifty()
        vix_h = _fetch_vix()

        # Try to pull engine state for Greeks if available
        portfolio_greeks = {}
        capital_deployed = 0
        capital_available = 0
        try:
            engine_state_path = Path(__file__).resolve().parent.parent / "data" / "live_state.json"
            if engine_state_path.exists():
                with open(engine_state_path, "r") as f:
                    es = json.load(f)
                # Simple proxies — engine may not expose Greeks directly
                capital_deployed = float(es.get("system", {}).get("used_margin", 0))
                capital_available = float(es.get("system", {}).get("available_margin", 0))
        except Exception:
            pass

        ctx = build_options_context(
            nifty_history=nifty_h,
            vix_history=vix_h,
            option_chain={},
            portfolio_greeks=portfolio_greeks,
            capital_deployed=capital_deployed,
            capital_available=capital_available,
        )

        team = OptionsResearchTeam(prefer_fast=True, enable_llm_arbitration=False)
        tv = team.review("NIFTY", ctx)
        chosen_strategy = getattr(tv, "chosen_strategy", "HOLD")
        return {
            "symbol": "NIFTY",
            "final_action": tv.final_action,
            "chosen_strategy": chosen_strategy,
            "final_score": tv.final_score,
            "confidence": tv.confidence,
            "suggested_qty_mult": tv.suggested_qty_mult,
            "hold_days": tv.hold_days,
            "coordinator_note": tv.coordinator_note,
            "reports": [
                {"agent": r.agent_name, "score": r.score, "verdict": r.verdict,
                 "confidence": r.confidence, "flags": r.flags,
                 "one_liner": r.one_liner, "error": r.error,
                 "provider": r.provider_used}
                for r in tv.reports
            ],
            "context_summary": {
                "macro": ctx.get("macro", {}),
                "vol": ctx.get("vol", {}),
                "greeks": ctx.get("greeks", {}),
                "events": ctx.get("events", {}),
            },
        }
    except Exception as e:
        logger.error("options_review error: %s", e, exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/api/screener/mf_recommendations")
async def screener_mf_recommendations(category: str = "ALL"):
    """
    Mutual fund recommendations. For categories where our screener
    underperforms passive/active MFs (especially small-cap), point user
    to the better external alternative.
    """
    try:
        from screener.mf_recommendations import get_recommendations
        return {"recommendations": get_recommendations(category)}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/api/screener/smallcap_postmortem")
async def screener_smallcap_postmortem():
    """Return the small-cap backtest findings + honest report."""
    try:
        path = Path(__file__).resolve().parent.parent / "reports" / "screener" / "smallcap_strategies.json"
        if not path.exists():
            return {"error": "run 'python -m screener.test_smallcap_strategies' first"}
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/api/screener/current_regime")
async def screener_current_regime():
    """
    Classify the current market regime. INFORMATIONAL ONLY — does not
    drive picks. Read the postmortem for why regime-adaptive ensemble
    was rejected (regime detection at quarterly frequency is too lagged
    to improve picks; it hurt 5+/6 windows in backtest).
    """
    try:
        from screener.market_timing_analyzer import _fetch_nifty, _fetch_vix
        from screener.strategies.regime_adaptive import (
            classify_regime, compute_breadth,
        )
        from screener.data_loader import load_universe
        from screener.universe_extended import LARGE_CAP

        nifty = _fetch_nifty()
        vix = _fetch_vix()
        if nifty.empty:
            return {"error": "no NIFTY data"}
        asof = nifty.index[-1]
        history = load_universe(LARGE_CAP, period="2y", use_cache=True, progress=False)
        breadth = compute_breadth(history, asof)
        snap = classify_regime(nifty, vix, breadth, asof)
        return {
            "regime": snap.regime.value if hasattr(snap.regime, "value") else str(snap.regime),
            "reason": snap.reason,
            "nifty_close": snap.nifty_close,
            "nifty_ma_50": snap.nifty_ma_50,
            "nifty_ma_200": snap.nifty_ma_200,
            "above_200dma": snap.above_200dma,
            "golden_cross": snap.golden_cross,
            "rsi_14": snap.rsi_14,
            "vix_percentile": snap.vix_percentile,
            "breadth_pct": snap.breadth_pct,
            "informational_note": (
                "Regime detection is INFORMATIONAL only. Backtest showed "
                "regime-adaptive ensembles UNDERPERFORM pure momentum "
                "(2-3 of 6 windows beat NIFTY vs 4 of 6 for momentum). "
                "Picks remain pure momentum. Read regime_adaptation_postmortem.md."
            ),
        }
    except Exception as e:
        logger.error("current_regime error: %s", e, exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/api/screener/rolling_walkforward")
async def screener_rolling_walkforward():
    """Return the rolling walk-forward findings if saved."""
    try:
        path = Path(__file__).resolve().parent.parent / "reports" / "screener" / "rolling_walkforward.json"
        if not path.exists():
            return {"error": "run 'python -m screener.rolling_walkforward' first"}
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/api/screener/market_timing")
async def screener_market_timing():
    """Run the market timing analyzer on live NIFTY/VIX data. ~10-15s."""
    try:
        from screener.market_timing_analyzer import (
            _fetch_nifty, _fetch_vix,
            analyze_price_position, analyze_technical,
            analyze_volatility, analyze_breadth, historical_analog,
            make_verdict,
        )
        from screener.universe_extended import LARGE_CAP

        nifty = _fetch_nifty()
        if nifty.empty:
            return {"error": "could not fetch NIFTY data"}
        vix_df = _fetch_vix()
        price = analyze_price_position(nifty)
        tech = analyze_technical(nifty)
        vol = analyze_volatility(vix_df)
        breadth = analyze_breadth(LARGE_CAP)
        analog = historical_analog(nifty, price["dist_from_high_pct"])
        verdict = make_verdict(price, tech, vol, breadth, analog)
        return {
            "price": price,
            "technical": tech,
            "volatility": vol,
            "breadth": breadth,
            "historical_analog": analog,
            "verdict": verdict,
        }
    except Exception as e:
        logger.error("Market timing error: %s", e, exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/api/dca/status")
async def dca_status():
    """Return current DCA plan state. Returns null if no plan exists."""
    try:
        from screener.dca_state import load_plan
        plan = load_plan()
        if plan is None:
            return {"plan": None, "message": "No DCA plan started. Click INIT to create one."}
        return {"plan": plan.to_dict(),
                "remaining_capital": plan.remaining_capital(),
                "remaining_tranches": plan.remaining_tranches(),
                "base_tranche_size": plan.base_tranche_size()}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/api/dca/init")
async def dca_init(capital: float = 100000.0, tranches: int = 4):
    """Initialize a fresh DCA plan (wipes existing)."""
    try:
        from screener.dca_state import reset_plan, new_plan
        reset_plan()
        plan = new_plan(total_capital=float(capital), base_tranches=int(tranches))
        return {"ok": True, "plan": plan.to_dict()}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/api/dca/reset")
async def dca_reset():
    """Wipe the DCA plan."""
    try:
        from screener.dca_state import reset_plan
        reset_plan()
        return {"ok": True, "message": "Plan wiped"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/api/dca/resume")
async def dca_resume():
    """Resume a PAUSED plan."""
    try:
        from screener.dca_state import load_plan, save_plan
        plan = load_plan()
        if plan is None:
            return {"error": "no plan"}
        if plan.status == "PAUSED":
            plan.status = "ACTIVE"
            plan.pause_reason = None
            save_plan(plan)
        return {"ok": True, "status": plan.status}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/api/dca/run")
async def dca_run(dry_run: int = 0, enable_ai: int = 0):
    """
    Execute the weekly DCA flow on demand.
      dry_run=1: don't update state, don't send Telegram — return the
                 message that WOULD have been sent.
      dry_run=0: real run — records the week + sends Telegram if envs set.
    """
    try:
        from screener.dca_state import load_plan, record_week, pause_plan
        from screener.dca_triggers import evaluate as evaluate_triggers
        from screener.live_picks_v2 import generate as generate_picks
        from screener.weekly_dca import format_message
        from notifications import telegram_notifier as tg
        import re as _re

        plan = load_plan()
        if plan is None:
            return {"error": "no plan — call /api/dca/init first"}
        if plan.status == "DONE":
            return {"ok": False, "message": "Plan already DONE", "plan": plan.to_dict()}
        if plan.status == "STOPPED":
            return {"ok": False, "message": "Plan STOPPED — call /api/dca/resume", "plan": plan.to_dict()}

        snap = evaluate_triggers()
        base = plan.base_tranche_size()
        tranche = base * snap.tranche_multiplier
        if plan.status == "PAUSED" and not snap.stop_fired:
            plan.status = "ACTIVE"
            plan.pause_reason = None
        if snap.stop_fired:
            tranche = 0.0
            if not dry_run and plan.status != "PAUSED":
                pause_plan(plan, "; ".join(snap.stop_fired))
        tranche = min(tranche, plan.remaining_capital())

        picks_payload = {}
        if tranche > 0:
            picks_payload = generate_picks(
                capital=tranche, n_large=2, n_mid=1,
                enable_ai=bool(int(enable_ai)),
                force_refresh=True,
            )

        msg_html = format_message(plan, snap, tranche, picks_payload)
        msg_plain = _re.sub(r"<[^>]+>", "", msg_html)

        sent = False
        if not dry_run:
            bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
            chat_id = os.getenv("TELEGRAM_CHAT_ID")
            if bot_token and chat_id:
                try:
                    tg.configure(bot_token, chat_id)
                    tg.notify(msg_html)
                    sent = True
                except Exception as e:
                    logger.warning("Telegram send failed: %s", e)
            # Record the week
            record_week(
                plan=plan, tranche=tranche,
                triggers_fired=snap.accelerate_fired + snap.stop_fired,
                market_score=0, market_verdict=snap.recommended_action,
                picks=picks_payload.get("picks", []),
                notes="; ".join(snap.notes),
            )

        return {
            "ok": True,
            "dry_run": bool(dry_run),
            "tranche": tranche,
            "multiplier": snap.tranche_multiplier,
            "action": snap.recommended_action,
            "accelerate_fired": snap.accelerate_fired,
            "stop_fired": snap.stop_fired,
            "notes": snap.notes,
            "message_html": msg_html,
            "message_plain": msg_plain,
            "telegram_sent": sent,
            "picks": picks_payload.get("picks", []),
            "plan": plan.to_dict(),
        }
    except Exception as e:
        logger.error("DCA run error: %s", e, exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(e)})


# ── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("DASHBOARD_PORT", "8501"))
    logger.info("Starting Trading Dashboard on http://localhost:%d", port)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
