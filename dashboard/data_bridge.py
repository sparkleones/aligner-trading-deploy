"""
Dashboard Data Bridge — shared state between trading engine and dashboard.

The autonomous trader writes state periodically, the Streamlit dashboard reads it.
This avoids the single-session Kite Connect limitation (only one access token at a time).

Write side: DashboardStateWriter (used by run_autonomous.py)
Read side:  DashboardStateReader (used by live_dashboard.py)
"""

import json
import logging
import time
from datetime import datetime, date
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"
DASHBOARD_STATE_FILE = DATA_DIR / "dashboard_state.json"
PNL_CURVE_FILE = DATA_DIR / "pnl_curve.json"
POSITION_STATE_FILE = DATA_DIR / "position_state.json"
BTST_FILE = DATA_DIR / "btst_positions.json"
ORDER_LOG_FILE = DATA_DIR / "order_log.json"
MANUAL_ORDER_FILE = DATA_DIR / "manual_order_request.json"


class DashboardStateWriter:
    """Writes trading engine state to JSON for the dashboard to consume.

    Called periodically by the autonomous trader (every bar / every 5s).
    """

    def __init__(self):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        self._pnl_curve: list[dict] = []
        self._logs: list[str] = []
        self._max_logs = 200
        self._last_write = 0.0
        self._write_interval = 0.1  # Write every 100ms for near-realtime dashboard

    def add_log(self, message: str) -> None:
        """Add a log message for dashboard display."""
        ts = datetime.now().strftime("%H:%M:%S")
        entry = f"[{ts}] {message}"
        self._logs.append(entry)
        if len(self._logs) > self._max_logs:
            self._logs = self._logs[-self._max_logs:]

    def update_pnl_curve(self, pnl: float, timestamp: str = "") -> None:
        """Append a PnL data point (max 1 per minute)."""
        ts = timestamp or datetime.now().strftime("%H:%M:%S")
        # Only add if timestamp minute changed (avoid per-second bloat)
        if self._pnl_curve and ts[:5] == self._pnl_curve[-1]["time"][:5]:
            # Same minute — just update the last point
            self._pnl_curve[-1]["pnl"] = round(pnl, 2)
            return
        self._pnl_curve.append({"time": ts, "pnl": round(pnl, 2)})

    def write_state(
        self,
        system_status: str = "TRADING",
        mode: str = "LIVE",
        symbol: str = "NIFTY",
        capital: float = 30000,
        bars_processed: int = 0,
        total_bars: int = 375,
        # Market data
        spot_price: float = 0.0,
        prev_close: float = 0.0,
        vix: float = 0.0,
        pcr: float = 0.0,
        market_bias: str = "",
        confidence: float = 0.0,
        support: float = 0.0,
        resistance: float = 0.0,
        is_expiry_day: bool = False,
        bars: list[dict] | None = None,
        # Positions
        open_positions: list[dict] | None = None,
        closed_positions: list[dict] | None = None,
        btst_positions: list[dict] | None = None,
        # P&L
        realized_pnl: float = 0.0,
        unrealized_pnl: float = 0.0,
        estimated_charges: float = 0.0,
        # Risk
        kill_switch_pct: float = 0.03,
        kill_switch_triggered: bool = False,
        max_positions: int = 4,
        trades_today: int = 0,
        max_trades_per_day: int = 5,
        winners: int = 0,
        losers: int = 0,
        # Orders
        orders: list[dict] | None = None,
        # Agent
        agent_name: str = "",
        signals_generated: int = 0,
        signals_accepted: int = 0,
        signals_filtered: int = 0,
        last_signal: str = "",
        # Additional
        started_at: str = "",
        # Live indices
        indices: list[dict] | None = None,
        # V14 decision state
        decision_state: dict | None = None,
        # AI Brain state
        ai_brain: dict | None = None,
        # Option chain for terminal display
        option_chain_display: list[dict] | None = None,
    ) -> None:
        """Write complete dashboard state to disk."""
        now = time.time()
        if now - self._last_write < self._write_interval:
            return  # Throttle writes
        self._last_write = now

        total_pnl_gross = realized_pnl + unrealized_pnl
        # Use GROSS P&L as headline (matches what Zerodha positions page shows)
        # Charges shown separately — Zerodha API PnL is before brokerage/charges
        total_pnl = total_pnl_gross  # Matches Zerodha P&L display
        day_change = spot_price - prev_close if prev_close > 0 else 0
        day_change_pct = (day_change / prev_close * 100) if prev_close > 0 else 0

        open_pos = open_positions or []
        closed_pos = closed_positions or []

        state = {
            "last_updated": datetime.now().isoformat(),
            "system": {
                "status": system_status,
                "mode": mode,
                "started_at": started_at or datetime.now().isoformat(),
                "symbol": symbol,
                "capital": capital,
                "bars_processed": bars_processed,
                "total_bars": total_bars,
            },
            "market": {
                "spot_price": round(spot_price, 2),
                "prev_close": round(prev_close, 2),
                "day_change": round(day_change, 2),
                "day_change_pct": round(day_change_pct, 2),
                "vix": round(vix, 2),
                "pcr": round(pcr, 2),
                "market_bias": market_bias,
                "confidence": round(confidence, 2),
                "support": round(support, 2),
                "resistance": round(resistance, 2),
                "is_expiry_day": is_expiry_day,
                "bars": bars or [],
            },
            "positions": {
                "open": open_pos,
                "closed": closed_pos,
                "btst": btst_positions or [],
            },
            "pnl": {
                "realized": round(realized_pnl, 2),
                "unrealized": round(unrealized_pnl, 2),
                "gross": round(total_pnl_gross, 2),
                "charges": round(estimated_charges, 2),
                "total": round(total_pnl, 2),
                "total_pct": round((total_pnl / capital * 100) if capital > 0 else 0, 2),
                "curve": self._pnl_curve,
            },
            "risk": {
                "daily_loss_pct": round(
                    abs(total_pnl / capital) if capital > 0 and total_pnl < 0 else 0, 4
                ),
                "kill_switch_pct": kill_switch_pct,
                "kill_switch_triggered": kill_switch_triggered,
                "max_positions": max_positions,
                "open_count": len(open_pos),
                "trades_today": trades_today,
                "max_trades_per_day": max_trades_per_day,
                "winners": winners,
                "losers": losers,
                "win_rate": round(
                    winners / (winners + losers) * 100 if (winners + losers) > 0 else 0, 1
                ),
                "profit_factor": round(
                    sum(t.get("pnl", 0) for t in closed_pos if t.get("pnl", 0) > 0)
                    / max(abs(sum(t.get("pnl", 0) for t in closed_pos if t.get("pnl", 0) < 0)), 1)
                    if closed_pos else 0, 2
                ),
            },
            "orders": orders or [],
            "agent": {
                "name": agent_name,
                "signals_generated": signals_generated,
                "signals_accepted": signals_accepted,
                "signals_filtered": signals_filtered,
                "last_signal": last_signal,
            },
            "logs": self._logs[-100:],
            "indices": indices or [],
            "decision": decision_state or {},
            "ai_brain": ai_brain or {},
            "option_chain": option_chain_display or [],
        }

        try:
            tmp = DASHBOARD_STATE_FILE.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(state, f, default=str)
            tmp.replace(DASHBOARD_STATE_FILE)
        except Exception as e:
            logger.debug("Failed to write dashboard state: %s", e)


class DashboardStateReader:
    """Reads trading state for the Streamlit dashboard."""

    @staticmethod
    def read_state() -> dict:
        """Read the latest dashboard state."""
        try:
            if DASHBOARD_STATE_FILE.exists():
                with open(DASHBOARD_STATE_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
        return {}

    @staticmethod
    def read_manual_order() -> dict:
        """Read the manual order request/result."""
        try:
            if MANUAL_ORDER_FILE.exists():
                with open(MANUAL_ORDER_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
        return {}

    @staticmethod
    def read_claude_brain() -> dict:
        """Read Claude AI brain analysis."""
        brain_file = DATA_DIR / "claude_brain.json"
        try:
            if brain_file.exists():
                with open(brain_file, "r", encoding="utf-8") as f:
                    return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
        return {}

    @staticmethod
    def read_order_log() -> list[dict]:
        """Read the order log."""
        try:
            if ORDER_LOG_FILE.exists():
                with open(ORDER_LOG_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
        return []

    @staticmethod
    def read_position_state() -> dict:
        """Read the position state file."""
        try:
            if POSITION_STATE_FILE.exists():
                with open(POSITION_STATE_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
        return {}

    @staticmethod
    def read_btst() -> list[dict]:
        """Read BTST positions."""
        try:
            if BTST_FILE.exists():
                with open(BTST_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
        return []

    @staticmethod
    def get_daily_reports() -> list[dict]:
        """Read all daily report files."""
        reports = []
        report_dir = DATA_DIR / "daily_reports"
        if report_dir.exists():
            for f in sorted(report_dir.glob("report_*.json"), reverse=True):
                try:
                    with open(f, "r", encoding="utf-8") as fh:
                        reports.append(json.load(fh))
                except Exception:
                    pass
        return reports

    @staticmethod
    def get_recent_logs(n: int = 50) -> list[str]:
        """Get recent log entries from today's log file."""
        log_dir = Path(__file__).parent.parent / "logs"
        log_file = log_dir / f"autonomous_{date.today().isoformat()}.log"
        lines = []
        try:
            if log_file.exists():
                with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
                    all_lines = f.readlines()
                    lines = all_lines[-n:]
        except Exception:
            pass
        return [l.rstrip() for l in lines]
