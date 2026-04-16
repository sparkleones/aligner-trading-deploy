"""
V6/V7 UNLOCKED — Test what happens when we keep V6's SAFETY but unlock entries.

V6 only takes 43 trades in 104 days (0.4/day) because of 5 compounding filters:
  1. sr_gap_min = 150 (blocks most S/R bounces)
  2. sr_bias_required = True (blocks 50% of remaining S/R bounces)
  3. min_confidence = 0.55 (blocks 60% of composite signals)
  4. cooldown = 5 bars (loses 15% of entry windows)
  5. No zero-to-hero, restrictive BTST

This script tests 6 configurations:
  V6     = Current V6 (baseline, 43 trades)
  V7     = Current V7 hybrid (90 trades)
  V6_2T  = V6 + 2 trade/day target (relaxed confidence + S/R)
  V6_3T  = V6 + 3 trade/day target (+ zero-to-hero + BTST)
  V7_3T  = V7 + 3 trade/day target (+ zero-to-hero + BTST)
  V7_5T  = V7 + 5 trade/day target (full V3-style entries, V7 exits)

All variants keep V6/V7's critical safety features:
  - 0.6-1.0% trail stop (not V3's deadly 0.3%)
  - Entry-based sr_stop with 0.4% buffer
  - Expiry day close by bar 19
  - SPAN margin sizing (not V3's backwards lot calc)

Data: 38,685 real 1-min bars from Kite Connect (Oct 2025 - Apr 2026)
"""

import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backtesting.paper_trading_real_data import (
    sr_multi_method,
    bs_premium,
    get_strike_and_type,
    LOT_SIZE,
    BROKERAGE,
    STRIKE_INTERVAL,
)

from backtesting.v7_hybrid_comparison import (
    compute_composite,
    get_lot_count_span,
)

CAPITAL = 200_000
BARS_PER_DAY = 375

# ===========================================================================
# EXTENDED VERSION CONFIG — V6/V7 variants with unlocked entries
# ===========================================================================

VERSIONS = {
    "V6": {
        "cooldown": 5,
        "min_confidence": 0.55,
        "trail_pct": 0.010,       # 1.0% trail (V6 proven safe)
        "put_max_hold": 22,
        "call_max_hold": 20,
        "sr_stop_buffer": 0.004,
        "vix_floor": 12,
        "vix_ceil": 35,
        "sr_gap_min": 150,
        "sr_bias_required": True,
        "sr_breakout": False,
        "zero_hero": False,
        "btst_enabled": True,
        "btst_trail_ok": False,    # V6: no BTST on trail exits
        "btst_vix_cap": 20,
        "composite_windows_put": [(3, 5), (8, 10)],
        "composite_windows_call": [(4, 8)],
        "gap_fade": True,
        "orb_scale": 10,
        "orb_cap": 0.80,
        "max_entry_bar": 19,
        "expiry_close_bar": 19,
        "max_trades_per_day": 5,
        "max_concurrent": 2,
    },
    "V7": {
        "cooldown": 2,
        "min_confidence": 0.45,
        "trail_pct": 0.006,       # 0.6% trail (V7 sweet spot)
        "put_max_hold": 18,
        "call_max_hold": 15,
        "sr_stop_buffer": 0.004,
        "vix_floor": 11,
        "vix_ceil": 35,
        "sr_gap_min": 75,
        "sr_bias_required": True,
        "sr_breakout": False,
        "zero_hero": False,
        "btst_enabled": True,
        "btst_trail_ok": False,
        "btst_vix_cap": 20,
        "composite_windows_put": [(3, 5), (8, 10)],
        "composite_windows_call": [(4, 8)],
        "gap_fade": True,
        "orb_scale": 10,
        "orb_cap": 0.80,
        "max_entry_bar": 20,
        "expiry_close_bar": 19,
        "max_trades_per_day": 5,
        "max_concurrent": 2,
    },
    # ---- V6 with 2 trades/day target ----
    "V6_2T": {
        "cooldown": 3,             # Relaxed from 5
        "min_confidence": 0.45,    # Relaxed from 0.55
        "trail_pct": 0.010,        # Keep V6 safe trail
        "put_max_hold": 22,
        "call_max_hold": 20,
        "sr_stop_buffer": 0.004,
        "vix_floor": 11,           # Relaxed from 12
        "vix_ceil": 35,
        "sr_gap_min": 75,          # Relaxed from 150
        "sr_bias_required": True,  # Keep bias safety
        "sr_breakout": False,
        "zero_hero": False,
        "btst_enabled": True,
        "btst_trail_ok": False,
        "btst_vix_cap": 20,
        "composite_windows_put": [(2, 6), (8, 12)],  # Wider windows
        "composite_windows_call": [(3, 10)],          # Wider windows
        "gap_fade": True,
        "orb_scale": 10,
        "orb_cap": 0.80,
        "max_entry_bar": 20,
        "expiry_close_bar": 19,
        "max_trades_per_day": 5,
        "max_concurrent": 2,
    },
    # ---- V6 with 3 trades/day + zero-hero + BTST ----
    "V6_3T": {
        "cooldown": 2,             # Aggressive cooldown
        "min_confidence": 0.40,    # More signals
        "trail_pct": 0.010,        # Keep V6 safe trail
        "put_max_hold": 22,
        "call_max_hold": 20,
        "sr_stop_buffer": 0.004,
        "vix_floor": 11,
        "vix_ceil": 35,
        "sr_gap_min": 50,          # Very relaxed S/R
        "sr_bias_required": False, # Allow all S/R bounces
        "sr_breakout": False,
        "zero_hero": True,         # ENABLED
        "btst_enabled": True,
        "btst_trail_ok": True,     # BTST even on trail exits
        "btst_vix_cap": 25,        # Higher VIX cap for BTST
        "composite_windows_put": [(2, 7), (8, 14)],   # Much wider
        "composite_windows_call": [(3, 12)],
        "gap_fade": True,
        "orb_scale": 10,
        "orb_cap": 0.80,
        "max_entry_bar": 20,
        "expiry_close_bar": 19,
        "max_trades_per_day": 5,
        "max_concurrent": 2,
    },
    # ---- V7 with 3 trades/day + zero-hero + BTST ----
    "V7_3T": {
        "cooldown": 2,
        "min_confidence": 0.40,
        "trail_pct": 0.006,        # V7's 0.6% trail
        "put_max_hold": 18,
        "call_max_hold": 15,
        "sr_stop_buffer": 0.004,
        "vix_floor": 11,
        "vix_ceil": 35,
        "sr_gap_min": 50,
        "sr_bias_required": False,
        "sr_breakout": False,
        "zero_hero": True,         # ENABLED
        "btst_enabled": True,
        "btst_trail_ok": True,
        "btst_vix_cap": 25,
        "composite_windows_put": [(2, 7), (8, 14)],
        "composite_windows_call": [(3, 12)],
        "gap_fade": True,
        "orb_scale": 10,
        "orb_cap": 0.85,
        "max_entry_bar": 20,
        "expiry_close_bar": 19,
        "max_trades_per_day": 5,
        "max_concurrent": 2,
    },
    # ---- V7 with 5 trades/day (full aggressive, V7 safety exits) ----
    "V7_5T": {
        "cooldown": 1,             # Minimal cooldown
        "min_confidence": 0.35,    # Accept more signals
        "trail_pct": 0.006,        # V7's 0.6% trail
        "put_max_hold": 18,
        "call_max_hold": 15,
        "sr_stop_buffer": 0.004,
        "vix_floor": 10,           # Accept VIX 10+
        "vix_ceil": 40,
        "sr_gap_min": 0,           # No S/R gap filter (like V3)
        "sr_bias_required": False, # No bias filter (like V3)
        "sr_breakout": False,      # Still no breakout (0% WR proven)
        "zero_hero": True,
        "btst_enabled": True,
        "btst_trail_ok": True,
        "btst_vix_cap": 30,
        "composite_windows_put": [(2, 8), (8, 16)],
        "composite_windows_call": [(2, 14)],
        "gap_fade": True,
        "orb_scale": 15,
        "orb_cap": 0.85,
        "max_entry_bar": 22,
        "expiry_close_bar": 19,
        "max_trades_per_day": 5,
        "max_concurrent": 2,
    },
}


# ===========================================================================
# LOAD DATA (reuse from real_1min_backtest)
# ===========================================================================

def load_1min_data():
    data_dir = project_root / "data" / "historical"
    nifty_path = data_dir / "nifty_min_2025-10-01_2026-04-06.csv"
    if not nifty_path.exists():
        print(f"ERROR: {nifty_path} not found!")
        sys.exit(1)

    nifty_1min = pd.read_csv(nifty_path, parse_dates=["timestamp"], index_col="timestamp")
    vix_path = data_dir / "vix_min_2025-10-01_2026-04-06.csv"
    vix_daily = None
    if vix_path.exists():
        vix_daily = pd.read_csv(vix_path, parse_dates=["timestamp"], index_col="timestamp")
    return nifty_1min, vix_daily


def build_daily_data(nifty_1min, vix_daily):
    daily = nifty_1min.resample("D").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum",
    }).dropna()
    daily.columns = ["Open", "High", "Low", "Close", "Volume"]

    if vix_daily is not None and not vix_daily.empty:
        vix_close = vix_daily["close"].reindex(daily.index, method="ffill")
        daily["VIX"] = vix_close.fillna(14.0)
    else:
        daily["VIX"] = 14.0
    daily["VIX"] = daily["VIX"].ffill().bfill().fillna(14.0)

    daily["PrevVIX"] = daily["VIX"].shift(1).fillna(daily["VIX"].iloc[0])
    daily["Change%"] = daily["Close"].pct_change() * 100
    daily["PrevChange%"] = daily["Change%"].shift(1).fillna(0)
    daily["DOW"] = daily.index.day_name()
    daily["SMA50"] = daily["Close"].rolling(50, min_periods=1).mean()
    daily["SMA20"] = daily["Close"].rolling(20, min_periods=1).mean()
    daily["AboveSMA50"] = daily["Close"] > daily["SMA50"]
    daily["AboveSMA20"] = daily["Close"] > daily["SMA20"]
    daily["EMA9"] = daily["Close"].ewm(span=9).mean()
    daily["EMA21"] = daily["Close"].ewm(span=21).mean()
    daily["WeeklySMA"] = daily["Close"].rolling(5).mean().rolling(4, min_periods=1).mean()

    delta = daily["Close"].diff()
    gain = delta.clip(lower=0).rolling(14, min_periods=1).mean()
    loss = (-delta.clip(upper=0)).rolling(14, min_periods=1).mean()
    daily["RSI"] = 100 - (100 / (1 + gain / loss.replace(0, 0.001)))
    daily["PrevHigh"] = daily["High"].shift(1)
    daily["PrevLow"] = daily["Low"].shift(1)
    daily["VIXSpike"] = daily["VIX"] > daily["PrevVIX"] * 1.15

    daily["IsExpiry"] = daily.index.map(
        lambda d: d.strftime("%A") == ("Tuesday" if d >= pd.Timestamp("2025-11-01") else "Thursday")
    )

    dte_values = []
    for idx in daily.index:
        current_dow = idx.weekday()
        target = 1 if idx >= pd.Timestamp("2025-11-01") else 3
        if current_dow <= target:
            dte = target - current_dow
        else:
            dte = 7 - current_dow + target
        dte_values.append(max(dte, 0.5))
    daily["DTE"] = dte_values

    daily["PrevClose"] = daily["Close"].shift(1)
    daily["GapPct"] = (daily["Open"] - daily["PrevClose"]) / daily["PrevClose"] * 100
    daily["GapPct"] = daily["GapPct"].fillna(0)

    return daily


# ===========================================================================
# ENTRY DETECTION (unified for all variants)
# ===========================================================================

def detect_entries_v2(cfg, bar_idx, path, support, resistance, vix, gap_pct,
                      composite_action, composite_conf, is_expiry,
                      prev_high, prev_low, above_sma50, above_sma20,
                      bias_val="neutral"):
    """Entry detection — same logic as v7_hybrid but driven by cfg dict."""
    signals = []
    spot = path[bar_idx] if bar_idx < len(path) else path[-1]
    MIN_CONF = cfg["min_confidence"]

    # No entries before bar 1 or after max_entry_bar
    if bar_idx < 1 or bar_idx > cfg["max_entry_bar"]:
        return signals
    if is_expiry and bar_idx > 17:
        return signals

    # 1. GAP ENTRY
    if bar_idx == 0 and abs(gap_pct) >= 0.3:
        if cfg["gap_fade"]:
            is_large_gap = abs(gap_pct) > 1.2
            if gap_pct < -0.3:
                if is_large_gap:
                    conf = min(0.85, 0.65 + abs(gap_pct) * 0.05)
                    if conf >= MIN_CONF:
                        signals.append(("BUY_CALL", "gap_fade", conf, False))
                else:
                    conf = min(0.90, 0.60 + abs(gap_pct) * 0.10)
                    if conf >= MIN_CONF:
                        signals.append(("BUY_PUT", "gap_entry", conf, False))
                # Zero-to-hero on gap
                if cfg["zero_hero"] and -1.2 <= gap_pct < -0.5 and vix >= 13:
                    signals.append(("BUY_PUT", "gap_zero_hero", 0.70, True))
            elif gap_pct > 0.3:
                if is_large_gap:
                    conf = min(0.85, 0.65 + gap_pct * 0.05)
                    if conf >= MIN_CONF:
                        signals.append(("BUY_PUT", "gap_fade", conf, False))
                elif gap_pct > 0.5 and above_sma50:
                    conf = min(0.85, 0.55 + gap_pct * 0.08)
                    if conf >= MIN_CONF:
                        signals.append(("BUY_CALL", "gap_entry", conf, False))
                # Zero-to-hero on gap up
                if cfg["zero_hero"] and gap_pct > 1.0 and vix >= 13:
                    signals.append(("BUY_PUT", "gap_zero_hero", 0.65, True))

    # 2. ORB ENTRY (bar 1-2)
    if bar_idx in (1, 2) and len(path) >= 2:
        orb_high = max(path[0], path[1])
        orb_low = min(path[0], path[1])
        orb_range = orb_high - orb_low
        if orb_range > spot * 0.0015:
            if spot > orb_high:
                conf = min(cfg["orb_cap"], 0.55 + (spot - orb_high) / orb_high * cfg["orb_scale"])
                if conf >= MIN_CONF and (above_sma50 or vix < 14):
                    signals.append(("BUY_CALL", "orb_breakout_up", conf, False))
            elif spot < orb_low:
                conf = min(cfg["orb_cap"], 0.55 + (orb_low - spot) / orb_low * cfg["orb_scale"])
                if conf >= MIN_CONF:
                    signals.append(("BUY_PUT", "orb_breakout_down", conf, False))

    # 3. S/R BOUNCE
    sr_dist = (resistance - support) if (support and resistance) else 0
    sr_valid = sr_dist >= cfg["sr_gap_min"] if cfg["sr_gap_min"] > 0 else True

    if bar_idx >= 2 and bar_idx <= 18 and sr_valid:
        prev_spot = path[bar_idx - 1] if bar_idx - 1 < len(path) else path[-1]

        if support and abs(spot - support) / spot < 0.003:
            if spot > prev_spot:
                if cfg["sr_bias_required"]:
                    if bias_val in ("bullish", "strong_bullish", "neutral"):
                        signals.append(("BUY_CALL", "sr_bounce_support", 0.65, False))
                else:
                    signals.append(("BUY_CALL", "sr_bounce_support", 0.65, False))

        if resistance and abs(spot - resistance) / spot < 0.003:
            if spot < prev_spot:
                if cfg["sr_bias_required"]:
                    if bias_val in ("bearish", "strong_bearish", "neutral"):
                        signals.append(("BUY_PUT", "sr_bounce_resistance", 0.70, False))
                else:
                    signals.append(("BUY_PUT", "sr_bounce_resistance", 0.70, False))

    # 4. COMPOSITE
    if composite_conf >= MIN_CONF:
        in_put_window = any(lo <= bar_idx <= hi for lo, hi in cfg["composite_windows_put"])
        in_call_window = any(lo <= bar_idx <= hi for lo, hi in cfg["composite_windows_call"])

        if composite_action == "BUY_PUT" and in_put_window:
            signals.append(("BUY_PUT", "composite", composite_conf, False))
        elif composite_action == "BUY_CALL" and in_call_window and vix < 12:
            if composite_conf >= 0.75:
                signals.append(("BUY_CALL", "composite", composite_conf, False))

    # 5. ZERO-TO-HERO: Standalone (not gap-based) — expiry day cheap puts
    if cfg["zero_hero"] and is_expiry and bar_idx in (1, 2, 3):
        if vix >= 13 and not above_sma50:
            signals.append(("BUY_PUT", "zero_hero_expiry", 0.60, True))

    return signals


# ===========================================================================
# EXIT LOGIC (unified, same as V7 but driven by cfg)
# ===========================================================================

def check_exit_v2(cfg, trade, bar_idx, bar_spot, bar_dte, vix,
                  support, resistance, is_expiry, path):
    """Exit logic — keeps V6/V7 safety features for ALL variants."""
    action = trade["action"]
    entry_bar = trade["entry_bar"]
    bars_held = bar_idx - entry_bar
    best_fav = trade["best_fav"]
    entry_spot = trade["entry_spot"]
    is_zero_hero = trade.get("is_zero_hero", False)

    if bars_held < 1:
        return None

    trail_dist = entry_spot * cfg["trail_pct"]

    # Expiry day close
    if cfg["expiry_close_bar"] and is_expiry and bar_idx >= cfg["expiry_close_bar"]:
        return "expiry_day_close"

    # Zero-hero exits
    if is_zero_hero:
        zh_trail = entry_spot * 0.008
        entry_prem = trade["entry_prem"]
        bar_prem = bs_premium(bar_spot, trade["strike"], bar_dte, vix, trade["opt_type"])
        current_return = (bar_prem - entry_prem) / max(entry_prem, 1)
        if current_return >= 2.0: return "zero_hero_target"
        if current_return <= -0.60: return "zero_hero_stop"
        if current_return >= 0.5:
            if action == "BUY_PUT" and bar_spot > best_fav + zh_trail:
                return "zero_hero_trail"
            elif action == "BUY_CALL" and bar_spot < best_fav - zh_trail:
                return "zero_hero_trail"
        if bars_held >= 10: return "zero_hero_time"
        return None

    # PUT exit
    if action == "BUY_PUT":
        if bars_held >= 3:
            if bar_spot > best_fav + trail_dist:
                return "trail_pct"
        if bars_held >= cfg["put_max_hold"]:
            return "time_exit"
        return None

    # CALL exit
    if action == "BUY_CALL":
        call_stop = entry_spot * (1 - cfg["sr_stop_buffer"])
        if not trade.get("sr_target_hit", False):
            if resistance and bar_spot >= resistance:
                trade["sr_target_hit"] = True
                trade["best_fav"] = bar_spot
            if bar_spot < call_stop and bars_held >= 3:
                return "sr_stop"
        else:
            if bar_spot < best_fav - trail_dist:
                return "sr_combo_trail"
        if bars_held >= cfg["call_max_hold"]:
            return "time_exit"
        return None

    return None


# ===========================================================================
# DAY SIMULATION (1-min bars)
# ===========================================================================

def simulate_day_1min(daily_row, row_idx, daily_df, close_prices,
                      day_1min_bars, version_name, cfg):
    """Simulate a single day on real 1-minute bars."""
    entry_spot = float(daily_row["Open"])
    vix = float(daily_row["VIX"]) if pd.notna(daily_row["VIX"]) else 14.0
    dow = str(daily_row["DOW"])
    above_sma50 = bool(daily_row["AboveSMA50"]) if pd.notna(daily_row.get("AboveSMA50")) else True
    above_sma20 = bool(daily_row["AboveSMA20"]) if pd.notna(daily_row.get("AboveSMA20")) else True
    rsi = float(daily_row["RSI"]) if pd.notna(daily_row.get("RSI")) else 50
    prev_change = float(daily_row["PrevChange%"]) if pd.notna(daily_row.get("PrevChange%")) else 0
    vix_spike = bool(daily_row["VIXSpike"]) if pd.notna(daily_row.get("VIXSpike")) else False
    sma20 = float(daily_row["SMA20"]) if pd.notna(daily_row.get("SMA20")) else None
    sma50 = float(daily_row["SMA50"]) if pd.notna(daily_row.get("SMA50")) else None
    prev_high = float(daily_row["PrevHigh"]) if pd.notna(daily_row.get("PrevHigh")) else entry_spot * 1.01
    prev_low = float(daily_row["PrevLow"]) if pd.notna(daily_row.get("PrevLow")) else entry_spot * 0.99
    is_expiry = bool(daily_row.get("IsExpiry", False))
    dte_market = float(daily_row.get("DTE", 2.0))
    ema9 = float(daily_row["EMA9"]) if pd.notna(daily_row.get("EMA9")) else None
    ema21 = float(daily_row["EMA21"]) if pd.notna(daily_row.get("EMA21")) else None
    weekly_sma = float(daily_row["WeeklySMA"]) if pd.notna(daily_row.get("WeeklySMA")) else None
    gap_pct = float(daily_row["GapPct"]) if pd.notna(daily_row.get("GapPct")) else 0
    date_str = str(daily_df.index[row_idx].date())
    day_close = float(daily_row["Close"])

    # VIX skip
    if vix < cfg["vix_floor"] or vix > cfg["vix_ceil"]:
        return 0, [{"action": "SKIP", "reason": f"VIX {vix:.1f} out of range", "date": date_str}]

    # S/R levels
    support, resistance = sr_multi_method(
        entry_spot, prev_high, prev_low, sma20, sma50,
        close_history=close_prices, idx=row_idx)

    # Composite scoring (use V6/V7 fixed RSI for ALL variants)
    version_for_composite = "V6"  # All variants use V6's fixed RSI
    scores = compute_composite(
        version_for_composite, vix, above_sma50, above_sma20, rsi, dow,
        prev_change, vix_spike, entry_spot, support, resistance,
        ema9=ema9, ema21=ema21, weekly_sma=weekly_sma)
    best_composite = max(scores, key=scores.get)
    total_score = sum(scores.values())
    composite_conf = scores[best_composite] / total_score if total_score > 0 else 0

    # Bias
    bias_val = "neutral"
    if above_sma50 and above_sma20:
        bias_val = "strong_bullish" if ema9 and ema21 and ema9 > ema21 else "bullish"
    elif not above_sma50 and not above_sma20:
        bias_val = "strong_bearish" if ema9 and ema21 and ema9 < ema21 else "bearish"

    # 1-min close prices
    minute_closes = day_1min_bars["close"].values
    n_minutes = len(minute_closes)

    # Build 15-min path for entry detection
    n_15min_bars = min(25, n_minutes // 15 + 1)
    path_15min = []
    for i in range(n_15min_bars):
        idx = min(i * 15, n_minutes - 1)
        path_15min.append(minute_closes[idx])

    open_trades = []
    closed_trades = []
    total_day_trades = 0
    last_exit_bar = -99

    MAX_CONCURRENT = cfg["max_concurrent"]
    MAX_TRADES = cfg["max_trades_per_day"]

    for minute_idx in range(n_minutes):
        bar_spot = minute_closes[minute_idx]
        bar_15min = minute_idx // 15
        bar_dte = max(0.05, dte_market - minute_idx / 1440)

        # ====== 1. CHECK EXITS EVERY 1-MIN ======
        trades_to_close = []
        for ti, trade in enumerate(open_trades):
            entry_minute = trade["entry_minute"]
            minutes_held = minute_idx - entry_minute
            bars_held_15min = minutes_held // 15
            virtual_bar_idx = trade["entry_bar_15min"] + bars_held_15min

            exit_signal = check_exit_v2(
                cfg, trade, virtual_bar_idx, bar_spot, bar_dte,
                vix, support, resistance, is_expiry, path_15min)

            if exit_signal:
                exit_prem = bs_premium(bar_spot, trade["strike"], bar_dte,
                                       vix, trade["opt_type"])
                pnl = (exit_prem - trade["entry_prem"]) * trade["qty"] - 80
                trade["exit_bar"] = bar_15min
                trade["exit_minute"] = minute_idx
                trade["exit_spot"] = round(bar_spot, 0)
                trade["exit_prem"] = round(exit_prem, 2)
                trade["exit_reason"] = exit_signal
                trade["intraday_pnl"] = round(pnl, 0)
                trade["total_pnl"] = round(pnl, 0)
                trade["minutes_held"] = minutes_held
                trades_to_close.append(ti)
                last_exit_bar = bar_15min

        for ti in reversed(trades_to_close):
            closed_trades.append(open_trades.pop(ti))

        # ====== 2. ENTRIES AT 15-MIN BOUNDARIES ======
        is_15min_boundary = (minute_idx % 15 == 0)

        if (is_15min_boundary
                and len(open_trades) < MAX_CONCURRENT
                and total_day_trades < MAX_TRADES
                and bar_15min - last_exit_bar >= cfg["cooldown"]
                and bar_15min < cfg["max_entry_bar"]):

            entries = detect_entries_v2(
                cfg, bar_15min, path_15min, support, resistance, vix, gap_pct,
                best_composite, composite_conf, is_expiry,
                prev_high, prev_low, above_sma50, above_sma20,
                bias_val=bias_val)

            if entries:
                entries.sort(key=lambda x: x[2], reverse=True)

                for action, entry_type, conf, is_zero_hero in entries:
                    if len(open_trades) >= MAX_CONCURRENT:
                        break
                    if total_day_trades >= MAX_TRADES:
                        break

                    same_dir = [t for t in open_trades if t["action"] == action]
                    if same_dir:
                        continue

                    strike, opt_type = get_strike_and_type(action, bar_spot, vix, is_zero_hero)
                    num_lots = get_lot_count_span(vix, is_zero_hero)
                    qty = min(num_lots * LOT_SIZE, 1800)
                    entry_prem = bs_premium(bar_spot, strike, bar_dte, vix, opt_type)

                    trade = {
                        "day": row_idx + 1,
                        "date": date_str,
                        "dow": dow,
                        "action": action,
                        "entry_type": entry_type,
                        "is_zero_hero": is_zero_hero,
                        "confidence": round(conf, 2),
                        "entry_bar": bar_15min,
                        "entry_bar_15min": bar_15min,
                        "entry_minute": minute_idx,
                        "entry_spot": round(bar_spot, 0),
                        "entry_prem": round(entry_prem, 2),
                        "strike": int(strike),
                        "opt_type": opt_type,
                        "lots": num_lots,
                        "qty": qty,
                        "vix": round(vix, 1),
                        "is_expiry": is_expiry,
                        "dte": round(bar_dte, 1),
                        "support": support,
                        "resistance": resistance,
                        "best_fav": bar_spot,
                        "sr_target_hit": False,
                        "exit_bar": -1, "exit_minute": -1,
                        "exit_spot": 0, "exit_prem": 0,
                        "exit_reason": "", "intraday_pnl": 0,
                        "overnight_pnl": 0, "total_pnl": 0,
                        "minutes_held": 0,
                    }
                    open_trades.append(trade)
                    total_day_trades += 1

        # ====== 3. UPDATE TRACKING ======
        for trade in open_trades:
            if trade["action"] == "BUY_CALL" and bar_spot > trade["best_fav"]:
                trade["best_fav"] = bar_spot
            elif trade["action"] == "BUY_PUT" and bar_spot < trade["best_fav"]:
                trade["best_fav"] = bar_spot

    # ====== 4. FORCE CLOSE ======
    for trade in open_trades:
        exit_prem = bs_premium(day_close, trade["strike"],
                               max(0.05, dte_market - n_minutes / 1440),
                               vix, trade["opt_type"])
        pnl = (exit_prem - trade["entry_prem"]) * trade["qty"] - 80
        trade["exit_bar"] = n_minutes // 15
        trade["exit_minute"] = n_minutes - 1
        trade["exit_spot"] = round(day_close, 0)
        trade["exit_prem"] = round(exit_prem, 2)
        trade["exit_reason"] = "eod_close"
        trade["intraday_pnl"] = round(pnl, 0)
        trade["total_pnl"] = round(pnl, 0)
        trade["minutes_held"] = n_minutes - 1 - trade["entry_minute"]
        closed_trades.append(trade)

    # ====== 5. BTST ======
    if cfg["btst_enabled"]:
        for trade in closed_trades:
            valid_exits = ("eod_close", "time_exit")
            if cfg["btst_trail_ok"]:
                valid_exits = ("eod_close", "time_exit", "trail_pct")

            btst_ok = (trade["action"] == "BUY_PUT"
                       and trade["intraday_pnl"] > 0
                       and not is_expiry
                       and vix < cfg["btst_vix_cap"]
                       and trade["exit_reason"] in valid_exits
                       and row_idx + 1 < len(daily_df))

            if btst_ok:
                next_row = daily_df.iloc[row_idx + 1]
                next_open = float(next_row["Open"])
                gap = (next_open - day_close) / day_close * 100
                if gap < 0:
                    on_pnl = (day_close - next_open) * trade["qty"] * 0.5 - 50
                    on_pnl = max(on_pnl, -trade["intraday_pnl"] * 0.5)
                else:
                    on_pnl = -abs(gap) * trade["qty"] * 0.3
                    on_pnl = max(on_pnl, -trade["intraday_pnl"] * 0.5)
                trade["overnight_pnl"] = round(on_pnl, 0)
                trade["total_pnl"] = round(trade["intraday_pnl"] + on_pnl, 0)

    if not closed_trades:
        return 0, [{"action": "SKIP", "reason": "No signals", "date": date_str}]

    return sum(t["total_pnl"] for t in closed_trades), closed_trades


# ===========================================================================
# RUN VERSION
# ===========================================================================

def run_version(daily_df, close_prices, day_bars_dict, version_name):
    cfg = VERSIONS[version_name]
    equity = CAPITAL
    equity_curve = [CAPITAL]
    all_trades = []
    peak_equity = CAPITAL
    max_dd = 0
    daily_pnl_list = []
    entry_stats = defaultdict(lambda: {"count": 0, "pnl": 0, "wins": 0})
    exit_stats = defaultdict(lambda: {"count": 0, "pnl": 0, "wins": 0})
    btst_count = 0
    btst_pnl = 0

    for i in range(len(daily_df)):
        date = daily_df.index[i].date()
        if date not in day_bars_dict:
            equity_curve.append(equity)
            daily_pnl_list.append(0)
            continue

        day_1min = day_bars_dict[date]
        if len(day_1min) < 30:
            equity_curve.append(equity)
            daily_pnl_list.append(0)
            continue

        row = daily_df.iloc[i]
        day_pnl, day_trades = simulate_day_1min(
            row, i, daily_df, close_prices, day_1min, version_name, cfg)

        if len(day_trades) == 1 and day_trades[0].get("action") == "SKIP":
            equity_curve.append(equity)
            daily_pnl_list.append(0)
            continue

        equity += day_pnl
        daily_pnl_list.append(day_pnl)
        if equity > peak_equity:
            peak_equity = equity
        dd = (peak_equity - equity) / peak_equity * 100
        if dd > max_dd:
            max_dd = dd

        for t in day_trades:
            all_trades.append(t)
            et = t.get("entry_type", "?")
            entry_stats[et]["count"] += 1
            entry_stats[et]["pnl"] += t["total_pnl"]
            if t["total_pnl"] > 0:
                entry_stats[et]["wins"] += 1
            er = t.get("exit_reason", "?")
            exit_stats[er]["count"] += 1
            exit_stats[er]["pnl"] += t["total_pnl"]
            if t["total_pnl"] > 0:
                exit_stats[er]["wins"] += 1
            if t.get("overnight_pnl", 0) != 0:
                btst_count += 1
                btst_pnl += t["overnight_pnl"]

        equity_curve.append(equity)

    net_pnl = equity - CAPITAL
    total = len(all_trades)
    wins = [t for t in all_trades if t["total_pnl"] > 0]
    losses = [t for t in all_trades if t["total_pnl"] <= 0]
    wr = len(wins) / total * 100 if total else 0

    daily_arr = np.array([d for d in daily_pnl_list if d != 0])
    sharpe = 0
    if len(daily_arr) > 1 and daily_arr.std() > 0:
        sharpe = (daily_arr.mean() / daily_arr.std()) * np.sqrt(252)

    gw = sum(t["total_pnl"] for t in wins)
    gl = abs(sum(t["total_pnl"] for t in losses))
    pf = gw / gl if gl > 0 else float("inf")

    trail_killed = [t for t in all_trades if t.get("exit_reason") == "trail_pct"]
    time_exits = [t for t in all_trades if t.get("exit_reason") == "time_exit"]
    eod_exits = [t for t in all_trades if t.get("exit_reason") == "eod_close"]
    zh_trades = [t for t in all_trades if t.get("is_zero_hero")]

    # Trades per day
    trading_days = len([d for d in daily_pnl_list if d != 0])
    trades_per_day = total / max(trading_days, 1)

    return {
        "version": version_name,
        "net_pnl": round(net_pnl),
        "return_pct": round(net_pnl / CAPITAL * 100, 1),
        "total_trades": total,
        "trades_per_day": round(trades_per_day, 1),
        "winning_trades": len(wins),
        "losing_trades": len(losses),
        "win_rate": round(wr, 1),
        "sharpe": round(sharpe, 2),
        "profit_factor": round(pf, 2),
        "max_drawdown": round(max_dd, 2),
        "avg_win": round(np.mean([t["total_pnl"] for t in wins]) if wins else 0),
        "avg_loss": round(np.mean([t["total_pnl"] for t in losses]) if losses else 0),
        "final_equity": round(equity),
        "pnl_per_trade": round(net_pnl / max(total, 1)),
        "trail_kills": len(trail_killed),
        "time_exits": len(time_exits),
        "eod_exits": len(eod_exits),
        "btst_count": btst_count,
        "btst_pnl": round(btst_pnl),
        "zero_hero_trades": len(zh_trades),
        "zero_hero_pnl": round(sum(t["total_pnl"] for t in zh_trades)),
        "entry_stats": dict(entry_stats),
        "exit_stats": dict(exit_stats),
    }


# ===========================================================================
# DISPLAY
# ===========================================================================

def print_table(results_list, title):
    versions = [r["version"] for r in results_list]
    print(f"\n{'=' * 130}")
    print(f"  {title}")
    print(f"{'=' * 130}")

    header = f"{'Metric':<28}"
    for v in versions:
        header += f" {v:>16}"
    print(header)
    print("-" * 130)

    metrics = [
        ("Net P&L", lambda r: f"Rs {r['net_pnl']:>+,}"),
        ("Return %", lambda r: f"{r['return_pct']:>+.1f}%"),
        ("Total Trades", lambda r: f"{r['total_trades']}"),
        ("Trades / Day", lambda r: f"{r['trades_per_day']:.1f}"),
        ("Win / Lose", lambda r: f"{r['winning_trades']}W/{r['losing_trades']}L"),
        ("Win Rate", lambda r: f"{r['win_rate']:.1f}%"),
        ("Sharpe Ratio", lambda r: f"{r['sharpe']:.2f}"),
        ("Profit Factor", lambda r: f"{r['profit_factor']:.2f}"),
        ("Max Drawdown", lambda r: f"{r['max_drawdown']:.1f}%"),
        ("Avg Win", lambda r: f"Rs {r['avg_win']:>+,}"),
        ("Avg Loss", lambda r: f"Rs {r['avg_loss']:>+,}"),
        ("P&L per Trade", lambda r: f"Rs {r['pnl_per_trade']:>+,}"),
        ("", lambda r: ""),
        ("Trail Stop Kills", lambda r: f"{r['trail_kills']}"),
        ("Time Exits", lambda r: f"{r['time_exits']}"),
        ("EOD Closes", lambda r: f"{r['eod_exits']}"),
        ("BTST Trades", lambda r: f"{r['btst_count']}"),
        ("BTST P&L", lambda r: f"Rs {r['btst_pnl']:>+,}"),
        ("Zero-Hero Trades", lambda r: f"{r['zero_hero_trades']}"),
        ("Zero-Hero P&L", lambda r: f"Rs {r['zero_hero_pnl']:>+,}"),
    ]

    for name, fmt in metrics:
        if name == "":
            print(f"  {'--- TRADE BREAKDOWN ---':<26}")
            continue
        row = f"  {name:<26}"
        for r in results_list:
            row += f" {fmt(r):>16}"
        print(row)

    # Exit reasons
    print(f"\n  EXIT REASONS:")
    all_exits = set()
    for r in results_list:
        all_exits.update(r["exit_stats"].keys())
    for er in sorted(all_exits):
        row = f"    {er:<24}"
        for r in results_list:
            s = r["exit_stats"].get(er, {"count": 0, "pnl": 0, "wins": 0})
            wr_val = s["wins"] / s["count"] * 100 if s["count"] > 0 else 0
            row += f" {s['count']:>3}t {s['pnl']:>+8,.0f} {wr_val:>3.0f}%"
        print(row)

    # Entry types
    print(f"\n  ENTRY TYPES:")
    all_entries = set()
    for r in results_list:
        all_entries.update(r["entry_stats"].keys())
    for et in sorted(all_entries):
        row = f"    {et:<24}"
        for r in results_list:
            s = r["entry_stats"].get(et, {"count": 0, "pnl": 0, "wins": 0})
            wr_val = s["wins"] / s["count"] * 100 if s["count"] > 0 else 0
            row += f" {s['count']:>3}t {s['pnl']:>+8,.0f} {wr_val:>3.0f}%"
        print(row)

    print("=" * 130)


# ===========================================================================
# MAIN
# ===========================================================================

if __name__ == "__main__":
    print("=" * 130)
    print("  V6/V7 UNLOCKED — CAN MORE TRADES = MORE PROFITS?")
    print("  Data: 38,685 real 1-min bars from Kite Connect (Oct 2025 - Apr 2026)")
    print("  All variants keep V6/V7 SAFETY exits (wide trail, entry-based stops)")
    print("  Testing: relaxed entries + BTST + zero-to-hero")
    print("=" * 130)

    # Load data
    print("\n--- Loading data ---")
    nifty_1min, vix_daily = load_1min_data()
    day_bars_dict = {}
    for date, group in nifty_1min.groupby(nifty_1min.index.date):
        day_bars_dict[date] = group

    daily_df = build_daily_data(nifty_1min, vix_daily)
    close_prices = daily_df["Close"].values.tolist()

    print(f"Trading days: {len(day_bars_dict)} | NIFTY {daily_df['Close'].min():.0f}-{daily_df['Close'].max():.0f}")
    print(f"VIX range: {daily_df['VIX'].min():.1f} - {daily_df['VIX'].max():.1f}")

    # Run all versions
    print("\n--- Running backtests ---")
    results = []
    for ver in ["V6", "V7", "V6_2T", "V6_3T", "V7_3T", "V7_5T"]:
        print(f"  {ver}...", end=" ", flush=True)
        r = run_version(daily_df, close_prices, day_bars_dict, ver)
        results.append(r)
        print(f"Rs {r['net_pnl']:>+,} | {r['total_trades']}t ({r['trades_per_day']:.1f}/day) | "
              f"WR {r['win_rate']:.1f}% | Sharpe {r['sharpe']:.2f}")

    # Print comparison
    print_table(results, "V6/V7 UNLOCKED: More Trades + BTST + Zero-Hero (Real 1-min Bars)")

    # ---- FIND OPTIMAL ----
    print("\n" + "=" * 130)
    print("  FINDING THE OPTIMAL CONFIGURATION")
    print("=" * 130)

    # Score: weighted combination of Sharpe, PF, P&L, and drawdown
    for r in results:
        # Composite score: higher is better
        score = (r["sharpe"] * 2.0 +
                 r["profit_factor"] * 1.5 +
                 r["return_pct"] / 50 +
                 (30 - r["max_drawdown"]) * 0.3 +
                 min(r["trades_per_day"], 3) * 1.0)  # Reward up to 3 trades/day
        r["composite_score"] = round(score, 2)

    results_sorted = sorted(results, key=lambda r: r["composite_score"], reverse=True)

    print(f"\n  {'Rank':<6} {'Version':<12} {'Score':>8} {'P&L':>12} {'Trades/Day':>12} "
          f"{'Sharpe':>8} {'PF':>6} {'MaxDD':>8} {'WR':>6}")
    print("-" * 90)
    for i, r in enumerate(results_sorted):
        marker = " <-- WINNER" if i == 0 else ""
        print(f"  {i+1:<6} {r['version']:<12} {r['composite_score']:>8.1f} "
              f"Rs {r['net_pnl']:>+9,} {r['trades_per_day']:>10.1f} "
              f"{r['sharpe']:>8.2f} {r['profit_factor']:>6.2f} {r['max_drawdown']:>7.1f}% "
              f"{r['win_rate']:>5.1f}%{marker}")

    winner = results_sorted[0]
    print(f"\n  RECOMMENDED FOR LIVE: {winner['version']}")
    print(f"    P&L: Rs {winner['net_pnl']:>+,} ({winner['return_pct']:>+.1f}%)")
    print(f"    Trades/day: {winner['trades_per_day']:.1f}")
    print(f"    Sharpe: {winner['sharpe']:.2f} | PF: {winner['profit_factor']:.2f} | Max DD: {winner['max_drawdown']:.1f}%")
    print(f"    BTST: {winner['btst_count']} trades, Rs {winner['btst_pnl']:>+,}")
    print(f"    Zero-Hero: {winner['zero_hero_trades']} trades, Rs {winner['zero_hero_pnl']:>+,}")
    print("=" * 130)

    # Save
    save_data = {}
    for r in results:
        save_data[r["version"]] = {k: v for k, v in r.items()
                                    if k not in ("entry_stats", "exit_stats")}
        save_data[r["version"]]["entry_type_stats"] = {
            k: {"count": s["count"], "pnl": float(s["pnl"]), "wins": s["wins"]}
            for k, s in r["entry_stats"].items()
        }
        save_data[r["version"]]["exit_reason_stats"] = {
            k: {"count": s["count"], "pnl": float(s["pnl"]), "wins": s["wins"]}
            for k, s in r["exit_stats"].items()
        }

    out_path = project_root / "data" / "v6v7_unlocked_results.json"
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")
