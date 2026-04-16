"""
Telegram Notifier — Aligner Trading System
==========================================
Sends real-time alerts to Telegram for:
  • System start / stop
  • Trade entry / exit
  • Daily P&L summary
  • Kill switch / risk events
  • Heartbeat (hourly, proves system is alive)

Setup (one-time):
  1. Message @BotFather on Telegram → /newbot → copy the token
  2. Message your bot once, then run:
       python -c "from notifications.telegram_notifier import get_chat_id; get_chat_id()"
  3. Add to .env:
       TELEGRAM_BOT_TOKEN=<token>
       TELEGRAM_CHAT_ID=<chat_id>
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from datetime import datetime
from typing import Optional

import httpx

logger = logging.getLogger("telegram")

# ── Config (read from env / .env) ──────────────────────────────────────────────
_BOT_TOKEN: str = ""
_CHAT_ID: str = ""
_ENABLED: bool = False


def configure(bot_token: str = "", chat_id: str = "") -> bool:
    """Initialise notifier from args or env vars. Returns True if enabled."""
    global _BOT_TOKEN, _CHAT_ID, _ENABLED
    _BOT_TOKEN = bot_token or os.environ.get("TELEGRAM_BOT_TOKEN", "")
    _CHAT_ID = chat_id or os.environ.get("TELEGRAM_CHAT_ID", "")
    _ENABLED = bool(_BOT_TOKEN and _CHAT_ID)
    if _ENABLED:
        logger.info("Telegram notifier enabled (chat_id=%s)", _CHAT_ID)
    else:
        logger.info("Telegram notifier disabled — set TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID in .env")
    return _ENABLED


def get_chat_id() -> None:
    """Helper: print your chat_id after you've messaged your bot."""
    import requests
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    if not token:
        print("Set TELEGRAM_BOT_TOKEN in .env first")
        return
    r = requests.get(f"https://api.telegram.org/bot{token}/getUpdates")
    data = r.json()
    for update in data.get("result", []):
        chat = update.get("message", {}).get("chat", {})
        print(f"chat_id: {chat.get('id')}  username: {chat.get('username')}")


# ── Core send ──────────────────────────────────────────────────────────────────

def _send(text: str, parse_mode: str = "HTML") -> None:
    """Send a message synchronously (fire-and-forget, never raises)."""
    if not _ENABLED:
        return
    try:
        url = f"https://api.telegram.org/bot{_BOT_TOKEN}/sendMessage"
        with httpx.Client(timeout=5.0) as client:
            client.post(url, json={
                "chat_id": _CHAT_ID,
                "text": text,
                "parse_mode": parse_mode,
                "disable_web_page_preview": True,
            })
    except Exception as e:
        logger.debug("Telegram send failed: %s", e)


async def _send_async(text: str, parse_mode: str = "HTML") -> None:
    """Send a message asynchronously (non-blocking)."""
    if not _ENABLED:
        return
    try:
        url = f"https://api.telegram.org/bot{_BOT_TOKEN}/sendMessage"
        async with httpx.AsyncClient(timeout=5.0) as client:
            await client.post(url, json={
                "chat_id": _CHAT_ID,
                "text": text,
                "parse_mode": parse_mode,
                "disable_web_page_preview": True,
            })
    except Exception as e:
        logger.debug("Telegram send failed: %s", e)


def notify(text: str) -> None:
    """Send from sync context (or when no event loop is running)."""
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(_send_async(text))
    except RuntimeError:
        _send(text)


# ── Event helpers ──────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _pnl_emoji(pnl: float) -> str:
    return "🟢" if pnl >= 0 else "🔴"


def on_system_start(symbol: str, capital: float, mode: str = "LIVE") -> None:
    emoji = "🚀" if mode == "LIVE" else "📄"
    notify(
        f"{emoji} <b>Aligner Trading Started</b>\n"
        f"Mode: <b>{mode}</b> | Symbol: {symbol}\n"
        f"Capital: ₹{capital:,.0f}\n"
        f"Time: {_now()}"
    )


def on_system_stop(reason: str = "Market close") -> None:
    notify(
        f"🛑 <b>Trading Stopped</b>\n"
        f"Reason: {reason}\n"
        f"Time: {_now()}"
    )


def on_trade_entry(symbol: str, side: str, qty: int, price: float,
                   lots: int, strategy: str = "", underlying: float = 0) -> None:
    arrow = "📈" if "CALL" in side or side == "BUY" else "📉"
    notify(
        f"{arrow} <b>TRADE ENTRY</b>\n"
        f"Symbol: <code>{symbol}</code>\n"
        f"Side: <b>{side}</b> | Lots: {lots} ({qty} qty)\n"
        f"Price: ₹{price:.2f}\n"
        f"NIFTY: {underlying:.0f} | {_now()}"
    )


def on_trade_exit(symbol: str, entry: float, exit_price: float,
                  qty: int, pnl: float, reason: str = "") -> None:
    emoji = _pnl_emoji(pnl)
    pct = ((exit_price - entry) / entry * 100) if entry > 0 else 0
    notify(
        f"{emoji} <b>TRADE EXIT</b>\n"
        f"Symbol: <code>{symbol}</code>\n"
        f"Entry: ₹{entry:.2f} → Exit: ₹{exit_price:.2f} ({pct:+.1f}%)\n"
        f"P&amp;L: <b>₹{pnl:+,.0f}</b> | Reason: {reason}\n"
        f"Time: {_now()}"
    )


def on_kill_switch(day_pnl: float, threshold_pct: float) -> None:
    notify(
        f"⚠️ <b>KILL SWITCH TRIGGERED</b>\n"
        f"Day P&amp;L: ₹{day_pnl:+,.0f}\n"
        f"Threshold: {threshold_pct:.1f}% exceeded\n"
        f"No new trades for rest of session.\n"
        f"Time: {_now()}"
    )


def on_daily_summary(realized: float, unrealized: float, trades: int,
                     wins: int, losses: int, capital: float) -> None:
    total = realized + unrealized
    win_rate = (wins / trades * 100) if trades > 0 else 0
    emoji = _pnl_emoji(total)
    notify(
        f"{emoji} <b>Daily Summary</b> — {datetime.now().strftime('%d %b %Y')}\n"
        f"Total P&amp;L: <b>₹{total:+,.0f}</b>\n"
        f"  Realized: ₹{realized:+,.0f}\n"
        f"  Unrealized: ₹{unrealized:+,.0f}\n"
        f"Trades: {trades} ({wins}W / {losses}L | {win_rate:.0f}% WR)\n"
        f"Capital: ₹{capital:,.0f}"
    )


def on_heartbeat(spot: float, vix: float, bias: str,
                 realized: float, open_positions: int) -> None:
    notify(
        f"💓 <b>Heartbeat</b> — {_now()}\n"
        f"NIFTY: {spot:.0f} | VIX: {vix:.1f}\n"
        f"Bias: {bias} | Open: {open_positions} pos\n"
        f"Day P&amp;L: ₹{realized:+,.0f}"
    )


def on_error(message: str) -> None:
    notify(f"❌ <b>Error</b>\n{message}\nTime: {_now()}")


# ── Hourly heartbeat task ──────────────────────────────────────────────────────

_last_heartbeat: float = 0.0
HEARTBEAT_INTERVAL = 3600  # 1 hour


def maybe_heartbeat(spot: float, vix: float, bias: str,
                    realized: float, open_positions: int) -> None:
    """Call this every loop tick — sends heartbeat at most once per hour."""
    global _last_heartbeat
    now = time.monotonic()
    if now - _last_heartbeat >= HEARTBEAT_INTERVAL:
        _last_heartbeat = now
        on_heartbeat(spot, vix, bias, realized, open_positions)
