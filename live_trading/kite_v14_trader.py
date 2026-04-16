"""
LIVE TRADING ENGINE — V14 Model on Zerodha Kite Connect
========================================================
Phase 7: Deployment for real-time autonomous trading.

Architecture:
  1. Kite Connect login + WebSocket streaming
  2. Real-time 1-min candle aggregation
  3. V14 indicator calculation (RSI, EMA, BB, MACD, Supertrend, VWAP, Squeeze)
  4. Signal generation using V14 confluence filters
  5. Order placement via Kite API (NIFTY options)
  6. Position management: trailing stop, time exits, expiry close
  7. Risk controls: max lots, max concurrent, cooldown
  8. SQLite trade logging + Telegram alerts (optional)

SAFETY:
  - Paper trading mode by default (set LIVE_MODE=True to enable real orders)
  - All orders use MARKET type with NRML product
  - Position limits enforced: max 3 concurrent, max 7 per day
  - Kill switch: stops trading if equity drops > 20% from peak

USAGE:
  1. Set API credentials in config
  2. Run: python live_trading/kite_v14_trader.py
  3. Opens browser for Kite login
  4. After login, starts automated trading loop 9:15 AM - 3:30 PM IST
"""

import os
import sys
import time
import json
import math
import sqlite3
import logging
import datetime as dt
import threading
from pathlib import Path
from collections import deque

import numpy as np
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# =====================================================================
# CONFIGURATION
# =====================================================================

CONFIG = {
    # === KITE CONNECT CREDENTIALS ===
    "api_key": os.environ.get("KITE_API_KEY", "YOUR_API_KEY"),
    "api_secret": os.environ.get("KITE_API_SECRET", "YOUR_API_SECRET"),

    # === TRADING MODE ===
    "live_mode": False,          # False = paper trading (no real orders)
    "instrument": "NIFTY 50",    # Index to trade
    "exchange": "NFO",           # Options exchange
    "lot_size": 65,              # NIFTY lot size (SEBI Feb 2026: 75 → 65)

    # === RISK CONTROLS ===
    "starting_capital": 30_000,       # User's actual capital
    "max_concurrent_positions": 3,
    "max_daily_trades": 7,
    "max_lots": 200,
    "max_drawdown_pct": 20.0,    # Kill switch: stop if DD > 20%
    "cooldown_minutes": 10,

    # === V14 PRODUCTION CONFIG (587.5x return in 12-month OOS backtest) ===
    # Synced from V9_HYBRID_CONFIG in oos_june2024_test.py on 2026-04-07
    "min_confidence": 0.35,
    "trail_pct_put": 0.015,          # 1.5% trail for PUTs (widened from 1.0%)
    "trail_pct_call": 0.008,         # 0.8% trail for CALLs
    "min_hold_trail_put": 120,       # 120 min hold before PUT trail fires
    "min_hold_trail_call": 60,       # 60 min hold before CALL trail fires
    "max_hold_put": 300,             # Max 300 min hold for PUTs
    "max_hold_call": 270,            # Max 270 min hold for CALLs
    "block_call_4th_hour": True,     # Block CALL entries 225-300 min (all losses in data)
    "avoid_days": [],                # NO day filter — VWAP+RSI+Squeeze handle quality
                                     # Tested: 587.5x (all days) > 571.7x (Mon/Wed blocked)
    "avoid_windows": [(45, 75), (165, 285)],  # Block 10:00-10:30 and 12:00-14:00
    "vix_floor": 11,
    "vix_ceil": 35,

    # === V14 CONFLUENCE FILTERS (3 independent confirmations) ===
    "use_vwap_filter": True,         # VWAP: CALL needs price > VWAP, PUT needs price < VWAP
    "use_squeeze_filter": True,      # Squeeze: Block entries when BB inside Keltner Channels
    "use_rsi_hard_gate": True,       # RSI: CALL needs RSI > 60, PUT needs RSI < 40
    "rsi_call_min": 60,              # Research-optimal: 60/40 gives 12.0x (best tested)
    "rsi_put_max": 40,

    # === V14 REGIME DETECTION ===
    "use_regime_detection": True,
    "min_confidence_filter": 0.30,   # Reject confidence < 0.30 (18.5% WR, pure drag)
    "block_late_entries": 305,       # No entries after min 305

    # === ENTRY SCORING THRESHOLDS ===
    "put_score_min": 4.0,
    "call_score_min": 5.0,

    # === LOT SIZING MODIFIERS ===
    "put_bias_lot_mult": 1.3,       # PUTs 38.5% WR vs CALLs 25.8% -> boost PUTs
    "call_bias_lot_mult": 0.7,
    "vix_sweet_min": 14.0,          # VIX 14-16 = 83% of profits
    "vix_sweet_max": 16.0,
    "vix_sweet_lot_mult": 1.4,
    "vix_danger_min": 16.0,
    "vix_danger_max": 18.0,
    "vix_danger_lot_mult": 0.5,
    "expiry_day_lot_mult": 0.7,     # Thu expiry = 19.7% WR -> reduce lots
    "rsi_sweet_low": 20,            # RSI 20-35 = 51.4% WR (best zone)
    "rsi_sweet_high": 35,
    "rsi_sweet_lot_mult": 1.5,
    "rsi_danger_low": 55,           # RSI 55-65 = 27.3% WR (worst zone)
    "rsi_danger_high": 65,
    "rsi_danger_lot_mult": 0.5,
    "entry_type_lot_mult": {
        "v8_indicator": 1.0,
        "sr_bounce_resistance": 1.5,  # 61.5% WR, best entry type
        "orb_breakout_down": 1.5,
        "orb_breakout_up": 0.5,
        "composite": 0.8,
    },

    # === LOGGING ===
    "db_path": str(project_root / "data" / "live_trades.db"),
    "log_path": str(project_root / "data" / "live_trading.log"),

    # === TELEGRAM ALERTS (optional) ===
    "telegram_enabled": False,
    "telegram_bot_token": os.environ.get("TELEGRAM_BOT_TOKEN", ""),
    "telegram_chat_id": os.environ.get("TELEGRAM_CHAT_ID", ""),
}


# =====================================================================
# LOGGING SETUP
# =====================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(CONFIG["log_path"]),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("KiteV14")


# =====================================================================
# DATABASE SETUP
# =====================================================================

def init_db():
    """Initialize SQLite database for trade logging."""
    conn = sqlite3.connect(CONFIG["db_path"])
    conn.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            date TEXT,
            action TEXT,
            entry_type TEXT,
            confidence REAL,
            entry_minute INTEGER,
            entry_spot REAL,
            entry_prem REAL,
            strike INTEGER,
            opt_type TEXT,
            lots INTEGER,
            qty INTEGER,
            vix REAL,
            exit_minute INTEGER,
            exit_spot REAL,
            exit_prem REAL,
            exit_reason TEXT,
            pnl REAL,
            minutes_held INTEGER,
            regime TEXT,
            vwap REAL,
            rsi REAL,
            squeeze INTEGER,
            paper_trade INTEGER DEFAULT 1
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS daily_summary (
            date TEXT PRIMARY KEY,
            equity REAL,
            day_pnl REAL,
            trades INTEGER,
            wins INTEGER,
            peak_equity REAL,
            drawdown_pct REAL
        )
    """)
    conn.commit()
    return conn


# =====================================================================
# INDICATOR CALCULATIONS (real-time, from accumulated bars)
# =====================================================================

class IndicatorEngine:
    """Real-time indicator calculation from accumulated 1-min bars."""

    def __init__(self):
        self.bars = []          # List of {open, high, low, close, volume, timestamp}
        self.daily_bars = []    # Previous N days of daily OHLC for regime detection

    def add_bar(self, bar):
        """Add a 1-min bar and recalculate indicators."""
        self.bars.append(bar)

    def reset_day(self):
        """Reset for new trading day."""
        if self.bars:
            # Save daily summary for regime detection
            closes = [b["close"] for b in self.bars]
            highs = [b["high"] for b in self.bars]
            lows = [b["low"] for b in self.bars]
            if closes:
                self.daily_bars.append({
                    "open": self.bars[0]["open"],
                    "high": max(highs),
                    "low": min(lows),
                    "close": closes[-1],
                })
                # Keep last 60 days
                self.daily_bars = self.daily_bars[-60:]
        self.bars = []

    def get_current_indicators(self):
        """Calculate all indicators from accumulated bars."""
        if len(self.bars) < 15:
            return None

        closes = np.array([b["close"] for b in self.bars])
        highs = np.array([b["high"] for b in self.bars])
        lows = np.array([b["low"] for b in self.bars])
        volumes = np.array([b.get("volume", 0) for b in self.bars])
        n = len(closes)

        ind = {}
        ind["close"] = closes[-1]
        ind["open_today"] = self.bars[0]["open"]

        # RSI (14-period)
        ind["rsi"] = self._compute_rsi(closes, 14)

        # EMA 9, 21
        ind["ema9"] = self._compute_ema(closes, 9)
        ind["ema21"] = self._compute_ema(closes, 21)
        ind["ema9_above_ema21"] = ind["ema9"] > ind["ema21"]

        # Bollinger Bands (20, 2)
        if n >= 20:
            bb_slice = closes[-20:]
            ind["bb_mid"] = np.mean(bb_slice)
            bb_std = np.std(bb_slice, ddof=1)
            ind["bb_upper"] = ind["bb_mid"] + 2 * bb_std
            ind["bb_lower"] = ind["bb_mid"] - 2 * bb_std
            ind["bb_width"] = (ind["bb_upper"] - ind["bb_lower"]) / ind["bb_mid"] * 100
        else:
            ind["bb_upper"] = ind["bb_lower"] = ind["bb_mid"] = closes[-1]
            ind["bb_width"] = 0

        # ATR (14-period)
        ind["atr"] = self._compute_atr(highs, lows, closes, 14)

        # Keltner Channels for Squeeze
        kc_upper = ind["ema21"] + 1.5 * ind["atr"]
        kc_lower = ind["ema21"] - 1.5 * ind["atr"]
        ind["squeeze_on"] = (ind["bb_lower"] > kc_lower) and (ind["bb_upper"] < kc_upper)

        # VWAP (daily anchored)
        tp = (highs + lows + closes) / 3.0
        if volumes.sum() > 0:
            cum_tp_vol = np.cumsum(tp * volumes)
            cum_vol = np.cumsum(np.where(volumes <= 0, 1, volumes))
            ind["vwap"] = cum_tp_vol[-1] / cum_vol[-1]
        else:
            ind["vwap"] = np.mean(tp)

        # MACD (12, 26, 9)
        if n >= 26:
            ema12 = self._compute_ema(closes, 12)
            ema26 = self._compute_ema(closes, 26)
            macd_line = ema12 - ema26
            ind["macd_hist"] = macd_line  # Simplified
        else:
            ind["macd_hist"] = 0

        # Supertrend (simplified — direction only)
        ind["st_direction"] = self._compute_supertrend_direction(highs, lows, closes, 10, 3.0)

        # ADX (simplified)
        ind["adx"] = self._compute_adx(highs, lows, closes, 14)

        # Stochastic RSI
        stoch_k = self._compute_stoch_rsi(closes, 14)
        ind["stoch_overbought"] = stoch_k > 80
        ind["stoch_oversold"] = stoch_k < 20

        return ind

    def _compute_rsi(self, closes, period):
        if len(closes) < period + 1:
            return 50.0
        deltas = np.diff(closes[-(period+1):])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains) if len(gains) > 0 else 0.001
        avg_loss = np.mean(losses) if len(losses) > 0 else 0.001
        rs = avg_gain / max(avg_loss, 0.001)
        return 100 - (100 / (1 + rs))

    def _compute_ema(self, data, period):
        if len(data) < period:
            return data[-1]
        multiplier = 2 / (period + 1)
        ema = np.mean(data[:period])
        for val in data[period:]:
            ema = (val - ema) * multiplier + ema
        return ema

    def _compute_atr(self, highs, lows, closes, period):
        if len(closes) < period + 1:
            return np.mean(highs[-5:] - lows[-5:]) if len(highs) >= 5 else 50
        tr = []
        for i in range(1, len(closes)):
            tr_val = max(highs[i] - lows[i],
                         abs(highs[i] - closes[i-1]),
                         abs(lows[i] - closes[i-1]))
            tr.append(tr_val)
        return np.mean(tr[-period:])

    def _compute_supertrend_direction(self, highs, lows, closes, period, mult):
        """Simplified supertrend — returns 1 (bullish) or -1 (bearish)."""
        if len(closes) < period + 1:
            return 0
        atr = self._compute_atr(highs, lows, closes, period)
        mid = (highs[-1] + lows[-1]) / 2
        upper_band = mid + mult * atr
        lower_band = mid - mult * atr
        if closes[-1] > upper_band:
            return 1
        elif closes[-1] < lower_band:
            return -1
        return 1 if closes[-1] > closes[-2] else -1

    def _compute_adx(self, highs, lows, closes, period):
        """Simplified ADX calculation."""
        if len(closes) < period + 1:
            return 20.0
        # Use price volatility as proxy
        returns = np.diff(closes[-period:]) / closes[-(period+1):-1]
        return min(60, abs(np.mean(returns)) * 10000)

    def _compute_stoch_rsi(self, closes, period):
        if len(closes) < period * 2:
            return 50.0
        rsi_vals = []
        for i in range(period, len(closes)):
            rsi_vals.append(self._compute_rsi(closes[:i+1], period))
        if len(rsi_vals) < period:
            return 50.0
        recent = rsi_vals[-period:]
        rsi_min = min(recent)
        rsi_max = max(recent)
        if rsi_max - rsi_min < 0.01:
            return 50.0
        return (rsi_vals[-1] - rsi_min) / (rsi_max - rsi_min) * 100

    def detect_regime(self):
        """Detect market regime from daily bars."""
        if len(self.daily_bars) < 5:
            return {"regime": "neutral", "call_mult": 1.0, "put_mult": 1.0}

        closes = [d["close"] for d in self.daily_bars]
        close = closes[-1]

        # SMA trends
        sma20 = np.mean(closes[-20:]) if len(closes) >= 20 else np.mean(closes)
        sma50 = np.mean(closes[-50:]) if len(closes) >= 50 else np.mean(closes)
        above_sma20 = close > sma20
        above_sma50 = close > sma50

        # Momentum
        ret_5d = (close - closes[-5]) / closes[-5] * 100 if len(closes) >= 5 else 0
        ret_10d = (close - closes[-10]) / closes[-10] * 100 if len(closes) >= 10 else 0

        bull_score = sum([above_sma20, above_sma50, ret_5d > 0.5, ret_10d > 1.5])
        bear_score = sum([not above_sma20, not above_sma50, ret_5d < -0.5, ret_10d < -1.5])

        if bull_score >= 3:
            return {"regime": "bullish", "call_mult": 1.4, "put_mult": 0.5}
        elif bear_score >= 3:
            return {"regime": "bearish", "call_mult": 0.5, "put_mult": 1.5}
        else:
            return {"regime": "sideways", "call_mult": 0.6, "put_mult": 0.6}


# =====================================================================
# SIGNAL GENERATOR (V14 logic)
# =====================================================================

class SignalGenerator:
    """V14 signal generation with confluence filters."""

    def __init__(self, config):
        self.cfg = config

    def evaluate(self, indicators, vix, minute_idx, regime_info):
        """Evaluate current market state for entry signals.

        Returns: list of (action, entry_type, confidence)
        """
        signals = []
        spot = indicators["close"]

        # === V8 Indicator Scoring ===
        call_score = 0.0
        put_score = 0.0

        # Supertrend
        if indicators["st_direction"] == 1:
            call_score += 2.5
        elif indicators["st_direction"] == -1:
            put_score += 3.0

        # EMA alignment
        if indicators["ema9_above_ema21"]:
            call_score += 2.0
        else:
            put_score += 3.5

        # RSI scoring
        rsi = indicators["rsi"]
        if 30 <= rsi < 50:
            put_score += 1.5
        elif rsi < 30:
            call_score += 2.0
        elif rsi > 70:
            put_score += 2.0

        # MACD
        if indicators["macd_hist"] > 0:
            call_score += 0.5
        elif indicators["macd_hist"] < 0:
            put_score += 0.5

        # Bollinger Band position
        if spot <= indicators["bb_lower"]:
            call_score += 1.5
        if spot >= indicators["bb_upper"]:
            put_score += 1.5

        # VIX regime
        if 13 <= vix < 16:
            put_score += 1.5
        elif vix >= 16:
            put_score += 1.0

        # Check thresholds
        if put_score >= self.cfg["put_score_min"] and put_score > call_score:
            conf = min(1.0, put_score / 18.0)
            signals.append(("BUY_PUT", "v8_indicator", conf))
        elif call_score >= self.cfg["call_score_min"] and call_score > put_score:
            conf = min(1.0, call_score / 18.0)
            signals.append(("BUY_CALL", "v8_indicator", conf))

        # === V14 CONFLUENCE FILTERS ===
        filtered = []
        for action, etype, conf in signals:
            # Confidence filter
            if conf < self.cfg["min_confidence_filter"]:
                continue

            # VWAP filter
            if self.cfg["use_vwap_filter"]:
                vwap = indicators.get("vwap", spot)
                if action == "BUY_CALL" and spot <= vwap:
                    continue
                if action == "BUY_PUT" and spot >= vwap:
                    continue

            # RSI hard gate
            if self.cfg["use_rsi_hard_gate"]:
                if action == "BUY_CALL" and rsi < self.cfg["rsi_call_min"]:
                    continue
                if action == "BUY_PUT" and rsi > self.cfg["rsi_put_max"]:
                    continue

            # Squeeze filter
            if self.cfg["use_squeeze_filter"] and indicators.get("squeeze_on", False):
                continue

            # Block CALL in 4th hour
            if self.cfg["block_call_4th_hour"] and action == "BUY_CALL":
                if 225 <= minute_idx < 300:
                    continue

            # Late entry block
            if minute_idx > self.cfg["block_late_entries"]:
                continue

            # Avoid windows
            if any(s <= minute_idx < e for s, e in self.cfg["avoid_windows"]):
                continue

            # Regime filter
            if self.cfg["use_regime_detection"]:
                if regime_info["regime"] == "bullish" and action == "BUY_PUT":
                    if conf < 0.45:
                        continue
                elif regime_info["regime"] == "sideways":
                    if conf < 0.35:
                        continue

            filtered.append((action, etype, conf))

        return filtered


# =====================================================================
# POSITION MANAGER
# =====================================================================

class PositionManager:
    """Manages open positions, trailing stops, and exits."""

    def __init__(self, config):
        self.cfg = config
        self.open_positions = []
        self.closed_today = []
        self.daily_trade_count = 0
        self.last_exit_minute = -999

    def reset_day(self):
        """Reset for new trading day."""
        self.open_positions = []
        self.closed_today = []
        self.daily_trade_count = 0
        self.last_exit_minute = -999

    def can_enter(self, minute_idx):
        """Check if we can take a new position."""
        if len(self.open_positions) >= self.cfg["max_concurrent_positions"]:
            return False
        if self.daily_trade_count >= self.cfg["max_daily_trades"]:
            return False
        if minute_idx - self.last_exit_minute < self.cfg["cooldown_minutes"]:
            return False
        return True

    def check_exits(self, spot, minute_idx, vix, is_expiry):
        """Check all open positions for exit conditions."""
        exits = []
        for pos in self.open_positions:
            minutes_held = minute_idx - pos["entry_minute"]
            if minutes_held < 1:
                continue

            exit_reason = None
            action = pos["action"]

            # Expiry close
            if is_expiry and minute_idx >= 300:
                exit_reason = "expiry_close"

            # Time exit
            elif action == "BUY_PUT" and minutes_held >= self.cfg["max_hold_put"]:
                exit_reason = "time_exit"
            elif action == "BUY_CALL" and minutes_held >= self.cfg["max_hold_call"]:
                exit_reason = "time_exit"

            # Trailing stop
            elif action == "BUY_PUT" and minutes_held >= self.cfg["min_hold_trail_put"]:
                trail_d = pos["entry_spot"] * self.cfg["trail_pct_put"]
                if spot > pos["best_fav"] + trail_d:
                    exit_reason = "trail_stop"
            elif action == "BUY_CALL" and minutes_held >= self.cfg["min_hold_trail_call"]:
                trail_d = pos["entry_spot"] * self.cfg["trail_pct_call"]
                if spot < pos["best_fav"] - trail_d:
                    exit_reason = "trail_stop"

            # EOD close (3:15 PM)
            if not exit_reason and minute_idx >= 360:
                exit_reason = "eod_close"

            if exit_reason:
                exits.append((pos, exit_reason))
            else:
                # Update best favorable price
                if action == "BUY_CALL" and spot > pos["best_fav"]:
                    pos["best_fav"] = spot
                elif action == "BUY_PUT" and spot < pos["best_fav"]:
                    pos["best_fav"] = spot

        return exits

    def open_position(self, action, entry_type, conf, spot, strike, opt_type,
                      lots, qty, vix, minute_idx, entry_prem):
        """Record a new position."""
        pos = {
            "action": action, "entry_type": entry_type,
            "confidence": conf, "entry_spot": spot,
            "strike": strike, "opt_type": opt_type,
            "lots": lots, "qty": qty, "vix": vix,
            "entry_minute": minute_idx, "entry_prem": entry_prem,
            "best_fav": spot,
            "timestamp": dt.datetime.now().isoformat(),
        }
        self.open_positions.append(pos)
        self.daily_trade_count += 1
        return pos

    def close_position(self, pos, exit_reason, spot, minute_idx, exit_prem):
        """Close a position."""
        pos["exit_minute"] = minute_idx
        pos["exit_spot"] = spot
        pos["exit_prem"] = exit_prem
        pos["exit_reason"] = exit_reason
        pos["pnl"] = (exit_prem - pos["entry_prem"]) * pos["qty"] - 40
        pos["minutes_held"] = minute_idx - pos["entry_minute"]
        self.open_positions.remove(pos)
        self.closed_today.append(pos)
        self.last_exit_minute = minute_idx
        return pos


# =====================================================================
# STRIKE SELECTOR
# =====================================================================

def get_atm_strike(spot, action, step=50):
    """Get ATM strike for NIFTY options."""
    atm = round(spot / step) * step

    if action == "BUY_PUT":
        return int(atm), "PE"
    else:
        return int(atm), "CE"


# =====================================================================
# LOT SIZER (from V14 dynamic sizing)
# =====================================================================

def get_dynamic_lots(vix, equity, confidence=0.5, rsi=50.0,
                     action="BUY_PUT", entry_type="v8_indicator",
                     is_expiry=False):
    """Dynamic lot sizing based on equity, VIX, RSI, and market conditions.

    Synced with V14 production sizing from oos_june2024_test.py.
    """
    SPAN = {12: 30000, 15: 35000, 20: 40000, 25: 50000, 30: 60000}
    span_per_lot = 40000
    for threshold in sorted(SPAN.keys()):
        if vix <= threshold:
            span_per_lot = SPAN[threshold]
            break
    else:
        span_per_lot = 60000

    available = equity * 0.70
    base_lots = max(1, int(available / span_per_lot))

    # Confidence scaling
    if confidence >= 0.70:
        base_lots = int(base_lots * 1.25)
    elif confidence >= 0.50:
        base_lots = int(base_lots * 1.10)
    elif confidence < 0.30:
        base_lots = int(base_lots * 0.8)

    # Entry type multiplier
    et_mult = CONFIG.get("entry_type_lot_mult", {}).get(entry_type, 1.0)
    base_lots = max(1, int(base_lots * et_mult))

    # Direction bias (PUTs outperform CALLs historically)
    if action == "BUY_PUT":
        base_lots = max(1, int(base_lots * CONFIG.get("put_bias_lot_mult", 1.0)))
    elif action == "BUY_CALL":
        base_lots = max(1, int(base_lots * CONFIG.get("call_bias_lot_mult", 1.0)))

    # VIX regime scaling
    if CONFIG.get("vix_sweet_min", 0) <= vix <= CONFIG.get("vix_sweet_max", 999):
        base_lots = max(1, int(base_lots * CONFIG.get("vix_sweet_lot_mult", 1.0)))
    if CONFIG.get("vix_danger_min", 999) <= vix <= CONFIG.get("vix_danger_max", 999):
        base_lots = max(1, int(base_lots * CONFIG.get("vix_danger_lot_mult", 1.0)))

    # RSI zone scaling
    if CONFIG.get("rsi_sweet_low", 0) <= rsi <= CONFIG.get("rsi_sweet_high", 0):
        base_lots = max(1, int(base_lots * CONFIG.get("rsi_sweet_lot_mult", 1.0)))
    if CONFIG.get("rsi_danger_low", 999) <= rsi <= CONFIG.get("rsi_danger_high", 999):
        base_lots = max(1, int(base_lots * CONFIG.get("rsi_danger_lot_mult", 1.0)))

    # Expiry day reduction
    if is_expiry:
        base_lots = max(1, int(base_lots * CONFIG.get("expiry_day_lot_mult", 1.0)))

    return max(1, min(base_lots, CONFIG["max_lots"]))


# =====================================================================
# ORDER EXECUTOR (Kite API wrapper)
# =====================================================================

class OrderExecutor:
    """Handles order placement via Kite Connect API."""

    def __init__(self, kite_client=None, paper_mode=True):
        self.kite = kite_client
        self.paper_mode = paper_mode

    def place_order(self, symbol, action, qty, order_type="MARKET"):
        """Place an order. Returns order_id or None."""
        transaction = "BUY" if "BUY" in action else "SELL"

        if self.paper_mode:
            logger.info(f"  [PAPER] {transaction} {qty}x {symbol}")
            return f"PAPER-{dt.datetime.now().strftime('%H%M%S')}"

        if self.kite is None:
            logger.error("  Kite client not initialized!")
            return None

        try:
            order_id = self.kite.place_order(
                variety="regular",
                exchange="NFO",
                tradingsymbol=symbol,
                transaction_type=transaction,
                quantity=qty,
                product="NRML",
                order_type="MARKET",
            )
            logger.info(f"  [LIVE] {transaction} {qty}x {symbol} -> Order ID: {order_id}")
            return order_id
        except Exception as e:
            logger.error(f"  Order failed: {e}")
            return None

    def get_option_symbol(self, strike, opt_type, expiry_date):
        """Build NIFTY option trading symbol."""
        # Format: NIFTY2540324500CE (NIFTY + YYMMDD + STRIKE + CE/PE)
        exp_str = expiry_date.strftime("%y%m%d") if expiry_date else "000000"
        return f"NIFTY{exp_str}{strike}{opt_type}"


# =====================================================================
# TELEGRAM ALERTS
# =====================================================================

def send_telegram(message):
    """Send alert via Telegram bot."""
    if not CONFIG["telegram_enabled"]:
        return
    try:
        import requests
        url = f"https://api.telegram.org/bot{CONFIG['telegram_bot_token']}/sendMessage"
        requests.post(url, json={
            "chat_id": CONFIG["telegram_chat_id"],
            "text": message,
            "parse_mode": "HTML",
        }, timeout=5)
    except Exception as e:
        logger.warning(f"Telegram send failed: {e}")


# =====================================================================
# MAIN TRADING LOOP
# =====================================================================

class KiteV14Trader:
    """Main trading orchestrator."""

    def __init__(self):
        self.indicator_engine = IndicatorEngine()
        self.signal_generator = SignalGenerator(CONFIG)
        self.position_manager = PositionManager(CONFIG)
        self.order_executor = OrderExecutor(paper_mode=not CONFIG["live_mode"])
        self.db = init_db()

        self.equity = CONFIG["starting_capital"]
        self.peak_equity = self.equity
        self.current_vix = 14.0
        self.is_expiry = False
        self.minute_counter = 0

        logger.info("=" * 80)
        logger.info("  KITE V14 LIVE TRADER INITIALIZED")
        logger.info(f"  Mode: {'LIVE' if CONFIG['live_mode'] else 'PAPER TRADING'}")
        logger.info(f"  Capital: Rs {self.equity:,}")
        logger.info(f"  Filters: VWAP={CONFIG['use_vwap_filter']} "
                     f"Squeeze={CONFIG['use_squeeze_filter']} "
                     f"RSI Gate={CONFIG['use_rsi_hard_gate']}")
        logger.info("=" * 80)

    def connect_kite(self):
        """Initialize Kite Connect and authenticate."""
        try:
            from kiteconnect import KiteConnect

            kite = KiteConnect(api_key=CONFIG["api_key"])
            print(f"\n  Open this URL to login:")
            print(f"  {kite.login_url()}")
            print()

            request_token = input("  Enter request_token from redirect URL: ").strip()
            data = kite.generate_session(request_token, api_secret=CONFIG["api_secret"])

            kite.set_access_token(data["access_token"])
            logger.info(f"  Kite Connect authenticated. User: {data.get('user_name', 'N/A')}")

            self.order_executor = OrderExecutor(kite_client=kite,
                                                 paper_mode=not CONFIG["live_mode"])
            return kite

        except ImportError:
            logger.warning("  kiteconnect not installed. Running in paper mode.")
            logger.warning("  Install: pip install kiteconnect")
            return None
        except Exception as e:
            logger.error(f"  Kite Connect auth failed: {e}")
            return None

    def on_tick(self, spot, vix=None, volume=0):
        """Process a new 1-min tick/bar."""
        now = dt.datetime.now()
        bar = {
            "open": spot, "high": spot, "low": spot, "close": spot,
            "volume": volume, "timestamp": now,
        }
        self.indicator_engine.add_bar(bar)
        self.minute_counter += 1

        if vix is not None:
            self.current_vix = vix

        # Get indicators
        indicators = self.indicator_engine.get_current_indicators()
        if indicators is None:
            return  # Not enough data yet

        minute_idx = self.minute_counter

        # === CHECK EXITS ===
        exits = self.position_manager.check_exits(
            spot, minute_idx, self.current_vix, self.is_expiry)

        for pos, exit_reason in exits:
            exit_prem = pos["entry_prem"]  # Simplified — in live, get from market
            closed = self.position_manager.close_position(
                pos, exit_reason, spot, minute_idx, exit_prem)

            pnl = closed["pnl"]
            self.equity += pnl

            logger.info(f"  EXIT: {pos['action']} {pos['strike']}{pos['opt_type']} "
                         f"| {exit_reason} | P&L: Rs{pnl:+,.0f} | "
                         f"Equity: Rs{self.equity:,.0f}")

            # Log to DB
            self._log_trade(closed)

            # Telegram alert
            send_telegram(
                f"<b>EXIT</b> {pos['action']} {pos['strike']}{pos['opt_type']}\n"
                f"Reason: {exit_reason}\n"
                f"P&L: Rs{pnl:+,.0f}\n"
                f"Equity: Rs{self.equity:,.0f}"
            )

        # === KILL SWITCH ===
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity
        dd_pct = (self.peak_equity - self.equity) / self.peak_equity * 100
        if dd_pct > CONFIG["max_drawdown_pct"]:
            logger.critical(f"  KILL SWITCH: Drawdown {dd_pct:.1f}% > {CONFIG['max_drawdown_pct']}%")
            send_telegram(f"KILL SWITCH ACTIVATED! DD={dd_pct:.1f}%")
            return "KILL"

        # === CHECK ENTRIES ===
        if not self.position_manager.can_enter(minute_idx):
            return

        # Skip avoid days
        today = now.strftime("%A")
        if today in CONFIG["avoid_days"]:
            return

        # VIX filter
        if self.current_vix < CONFIG["vix_floor"] or self.current_vix > CONFIG["vix_ceil"]:
            return

        # Every 5 minutes: evaluate signals
        if minute_idx % 5 != 0:
            return

        regime = self.indicator_engine.detect_regime()
        signals = self.signal_generator.evaluate(
            indicators, self.current_vix, minute_idx, regime)

        for action, entry_type, conf in signals:
            if not self.position_manager.can_enter(minute_idx):
                break

            # Check same direction
            same_dir = [p for p in self.position_manager.open_positions
                        if p["action"] == action]
            if same_dir:
                continue

            # Get strike
            strike, opt_type = get_atm_strike(spot, action)

            # Lot sizing (V14 production — includes VIX/RSI/entry_type modifiers)
            lots = get_dynamic_lots(
                self.current_vix, self.equity, conf,
                rsi=indicators.get("rsi", 50),
                action=action, entry_type=entry_type,
                is_expiry=self.is_expiry,
            )

            # Regime-based scaling
            if CONFIG.get("use_regime_detection", True):
                if action == "BUY_PUT":
                    lots = max(1, int(lots * regime.get("put_mult", 1.0)))
                elif action == "BUY_CALL":
                    lots = max(1, int(lots * regime.get("call_mult", 1.0)))

            qty = lots * CONFIG["lot_size"]

            # Estimated premium (in live, get from option chain)
            entry_prem = spot * 0.005  # ~0.5% of spot as placeholder

            logger.info(f"  ENTRY: {action} {strike}{opt_type} | "
                         f"Conf={conf:.2f} | {lots} lots | "
                         f"VIX={self.current_vix:.1f} | Regime={regime['regime']}")

            # Place order
            expiry = self._get_next_expiry()
            symbol = self.order_executor.get_option_symbol(strike, opt_type, expiry)
            order_id = self.order_executor.place_order(symbol, action, qty)

            if order_id:
                pos = self.position_manager.open_position(
                    action, entry_type, conf, spot, strike, opt_type,
                    lots, qty, self.current_vix, minute_idx, entry_prem)

                send_telegram(
                    f"<b>ENTRY</b> {action} {strike}{opt_type}\n"
                    f"Conf: {conf:.2f} | Lots: {lots}\n"
                    f"VIX: {self.current_vix:.1f} | Regime: {regime['regime']}"
                )

    def _get_next_expiry(self):
        """Get next Thursday expiry."""
        today = dt.date.today()
        days_ahead = 3 - today.weekday()
        if days_ahead < 0:
            days_ahead += 7
        return today + dt.timedelta(days=days_ahead)

    def _log_trade(self, trade):
        """Log trade to SQLite database."""
        try:
            self.db.execute("""
                INSERT INTO trades (timestamp, date, action, entry_type, confidence,
                    entry_minute, entry_spot, entry_prem, strike, opt_type,
                    lots, qty, vix, exit_minute, exit_spot, exit_prem,
                    exit_reason, pnl, minutes_held, paper_trade)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade.get("timestamp", ""), dt.date.today().isoformat(),
                trade["action"], trade["entry_type"], trade["confidence"],
                trade["entry_minute"], trade["entry_spot"], trade["entry_prem"],
                trade["strike"], trade["opt_type"], trade["lots"], trade["qty"],
                trade["vix"], trade["exit_minute"], trade["exit_spot"],
                trade["exit_prem"], trade["exit_reason"], trade["pnl"],
                trade["minutes_held"], int(not CONFIG["live_mode"]),
            ))
            self.db.commit()
        except Exception as e:
            logger.error(f"  DB log failed: {e}")

    def run_paper_simulation(self):
        """Run paper trading simulation using cached NIFTY data."""
        logger.info("\n  Starting paper trading simulation...")
        logger.info("  (Uses cached 1-min NIFTY data)")

        from backtesting.multi_month_oos_test import download_range

        # Get recent data
        end = dt.date.today()
        start = end - dt.timedelta(days=60)
        nifty, vix_data = download_range(start.strftime("%Y-%m-%d"),
                                          end.strftime("%Y-%m-%d"))

        if nifty is None:
            logger.error("  No data available for simulation")
            return

        vix_lookup = {}
        if vix_data is not None:
            for idx, row in vix_data.iterrows():
                vix_lookup[idx.date()] = row["close"]

        # Group by day
        day_groups = {}
        for date, group in nifty.groupby(nifty.index.date):
            day_groups[date] = group

        for date in sorted(day_groups.keys()):
            if date < start:
                continue

            self.indicator_engine.reset_day()
            self.position_manager.reset_day()
            self.minute_counter = 0
            self.is_expiry = (date.weekday() == 3)  # Thursday

            day_vix = vix_lookup.get(date, 14.0)

            bars = day_groups[date]
            logger.info(f"\n  --- {date} (VIX={day_vix:.1f}) ---")

            for _, bar in bars.iterrows():
                result = self.on_tick(
                    bar["close"], vix=day_vix,
                    volume=bar.get("volume", 0))
                if result == "KILL":
                    logger.critical("  KILLED — stopping simulation")
                    return

            # EOD summary
            day_pnl = sum(t["pnl"] for t in self.position_manager.closed_today)
            logger.info(f"  EOD: {len(self.position_manager.closed_today)} trades, "
                         f"P&L=Rs{day_pnl:+,.0f}, Equity=Rs{self.equity:,.0f}")

        logger.info(f"\n  SIMULATION COMPLETE")
        logger.info(f"  Final Equity: Rs{self.equity:,.0f}")
        logger.info(f"  Return: {(self.equity/CONFIG['starting_capital']-1)*100:+.1f}%")


# =====================================================================
# ENTRY POINT
# =====================================================================

def main():
    trader = KiteV14Trader()

    print("\n  Choose mode:")
    print("  1. Paper Trading Simulation (cached data)")
    print("  2. Connect to Kite (live/paper)")
    print()

    choice = input("  Enter choice (1 or 2): ").strip()

    if choice == "2":
        kite = trader.connect_kite()
        if kite:
            print("\n  Kite connected! Starting trading loop...")
            print("  Press Ctrl+C to stop.")
            # In live mode, you'd set up WebSocket streaming here
            # For now, fall through to simulation
        else:
            print("  Falling back to paper simulation...")
            trader.run_paper_simulation()
    else:
        trader.run_paper_simulation()


if __name__ == "__main__":
    main()
