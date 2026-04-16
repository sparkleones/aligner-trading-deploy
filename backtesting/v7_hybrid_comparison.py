"""
V3 vs V4 vs V6 vs V7 Comparison — January 2025 OOS Data.

ALSO: Demonstrates WHY V3 fails in live trading by running all models
on REALISTIC (noisy) intraday paths vs the smooth synthetic paths.

The backtest's generate_intraday_path() creates SMOOTH paths:
  Open → Low → High → Close  (up day)
  Open → High → Low → Close  (down day)

Real market has NOISE, WHIPSAWS, and MICRO-PULLBACKS:
  Open → dip → spike → pullback → dip → recovery → High → pullback → Close

This noise triggers V3's tight 0.3% trail stop on EVERY pullback,
killing the trades that would have been winners via time_exit.

V7 (Hybrid — Best of V6 Safety + V3/V4 Performance):
  - RSI FIXED (from V6)
  - SPAN margin sizing (from V6)
  - Expiry handling (from V6)
  - COOLDOWN_BARS = 2 (from V3, not V6's 5)
  - MIN_CONFIDENCE = 0.45 (between V3's 0.25 and V6's 0.55)
  - TRAIL_PCT = 0.006 (0.6% — between V3's 0.3% and V6's 1.0%)
  - S/R bounce: bias-aware but gap lowered to 75pt (from V6's 150pt)
  - VIX floor: 11 (from V6's 12)
  - Composite windows: V4's bars 3-5 + 8-10

Capital: Rs 200,000 | Data: January 2025 from Yahoo Finance
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
    download_real_data,
    generate_intraday_path,
    sr_multi_method,
    bs_premium,
    get_strike_and_type,
    check_trade_exit,
    LOT_SIZE,
    BROKERAGE,
    STRIKE_INTERVAL,
    TOTAL_BARS,
    MAX_TRADES_PER_DAY,
    MAX_CONCURRENT,
)

CAPITAL = 200_000


# ===========================================================================
# REALISTIC INTRADAY PATH (with market noise)
# ===========================================================================

def generate_intraday_path_realistic(open_p, high, low, close, n_bars=TOTAL_BARS):
    """Generate REALISTIC intraday path with noise and whipsaws.

    The original generate_intraday_path() creates perfectly smooth paths
    that go Open→Low→High→Close (up) or Open→High→Low→Close (down).

    Real markets have:
      - 0.1-0.3% random noise on every bar
      - Mean-reverting whipsaws (50-100pt pullbacks)
      - Gap traps in first 15 minutes
      - Momentum bursts followed by consolidation

    This function adds these realistic elements while still hitting the
    same OHLC levels, making the path representative of actual 1-min
    tick behavior aggregated to 15-min bars.
    """
    # Start with the smooth base path
    base_path = generate_intraday_path(open_p, high, low, close, n_bars)

    # Add realistic noise
    np.random.seed(int(abs(open_p * 100 + close * 10)) % 2**31)

    noisy_path = [open_p]
    for i in range(1, n_bars):
        base = base_path[i]

        # Noise component: 0.05-0.15% of spot (realistic for NIFTY 15-min bars)
        noise_pct = np.random.uniform(0.0005, 0.0015)
        noise = base * noise_pct * np.random.choice([-1, 1])

        # Whipsaw component: occasional 0.2-0.4% pullbacks
        whipsaw = 0
        if np.random.random() < 0.25:  # 25% of bars have a whipsaw
            whipsaw_pct = np.random.uniform(0.002, 0.004)
            # Whipsaw against the trend direction
            if i > 1 and base > base_path[i-1]:
                whipsaw = -base * whipsaw_pct  # Pullback on up move
            elif i > 1 and base < base_path[i-1]:
                whipsaw = base * whipsaw_pct   # Bounce on down move

        # Gap trap: extra noise in first 2 bars
        gap_noise = 0
        if i <= 2:
            gap_noise = base * np.random.uniform(-0.002, 0.002)

        noisy_val = base + noise + whipsaw + gap_noise

        # Clamp to stay within day's range (with small overflow allowed)
        range_buffer = (high - low) * 0.05
        noisy_val = max(low - range_buffer, min(high + range_buffer, noisy_val))

        noisy_path.append(noisy_val)

    # Last bar must be close
    noisy_path[-1] = close

    return noisy_path


# ===========================================================================
# COMPOSITE SCORING — V3, V4, V6, V7
# ===========================================================================

def compute_composite(version, vix, above_sma50, above_sma20, rsi, dow,
                      prev_change, vix_spike, spot, support, resistance,
                      ema9=None, ema21=None, weekly_sma=None):
    """Version-aware composite scoring."""
    scores = {"BUY_CALL": 0.0, "BUY_PUT": 0.0}

    if vix < 12: scores["BUY_CALL"] += 3.0
    elif vix < 17: scores["BUY_PUT"] += 3.0
    elif vix < 25: scores["BUY_PUT"] += 3.5
    else: scores["BUY_PUT"] += 4.0

    if not above_sma50: scores["BUY_PUT"] += 2.0
    else: scores["BUY_CALL"] += 2.0

    if not above_sma20: scores["BUY_PUT"] += 1.0
    else: scores["BUY_CALL"] += 1.0

    # RSI: V3/V4 have the BUG, V6/V7 have the FIX
    if version in ("V3", "V4"):
        if rsi < 30: scores["BUY_PUT"] += 1.5    # BUG: oversold → PUT
        elif rsi > 70: scores["BUY_PUT"] += 1.5  # BUG: overbought → PUT
    else:  # V6, V7
        if rsi < 30: scores["BUY_CALL"] += 1.5   # FIX: oversold → CALL (bounce)
        elif rsi > 70: scores["BUY_PUT"] += 1.5  # FIX: overbought → PUT (pullback)

    # DOW rules
    if version == "V3":
        dow_rules = {"Monday": "BUY_PUT", "Tuesday": "BUY_PUT",
                     "Wednesday": "BUY_CALL", "Thursday": "BUY_PUT",
                     "Friday": "BUY_CALL"}
    else:  # V4, V6, V7
        dow_rules = {"Monday": "BUY_CALL", "Tuesday": "BUY_PUT",
                     "Wednesday": "BUY_CALL", "Thursday": "BUY_PUT",
                     "Friday": "BUY_PUT"}
    d = dow_rules.get(dow)
    if d: scores[d] += 0.5

    if vix_spike: scores["BUY_CALL"] += 2.0
    if prev_change < -1.0: scores["BUY_CALL"] += 1.0
    elif prev_change > 1.0: scores["BUY_PUT"] += 1.0

    if support and spot:
        dp = (spot - support) / spot * 100
        if 0 < dp < 1.0: scores["BUY_CALL"] += 1.0
        elif dp < 0: scores["BUY_PUT"] += 1.0
    if resistance and spot:
        dp = (resistance - spot) / spot * 100
        if 0 < dp < 1.0: scores["BUY_PUT"] += 1.0
        elif dp < 0: scores["BUY_CALL"] += 1.0

    if ema9 is not None and ema21 is not None and weekly_sma is not None:
        alignment = 0
        if spot > weekly_sma: alignment += 1
        if ema9 > ema21: alignment += 1
        best = max(scores, key=scores.get)
        if best == "BUY_CALL" and alignment == 0:
            scores["BUY_CALL"] *= 0.5
        elif best == "BUY_PUT" and alignment == 2:
            scores["BUY_PUT"] *= 0.5

    return scores


# ===========================================================================
# VERSION CONFIG
# ===========================================================================

VERSION_CONFIG = {
    "V3": {
        "cooldown": 2,
        "min_confidence": 0.25,
        "trail_pct": 0.003,
        "put_max_hold": 16,
        "call_max_hold": 12,
        "sr_stop_buffer": None,  # Uses support level directly
        "vix_floor": 10,
        "vix_ceil": 999,
        "sr_gap_min": 0,
        "sr_bias_required": False,
        "sr_breakout": True,
        "composite_windows_put": [(2, 4)],
        "composite_windows_call": [(2, 4)],
        "death_zone": True,
        "gap_fade": False,
        "orb_scale": 50,
        "orb_cap": 0.85,
        "max_entry_bar": 20,
        "expiry_close_bar": None,
    },
    "V4": {
        "cooldown": 0,
        "min_confidence": 0.25,
        "trail_pct": 0.003,
        "put_max_hold": 16,
        "call_max_hold": 12,
        "sr_stop_buffer": None,
        "vix_floor": 10,
        "vix_ceil": 999,
        "sr_gap_min": 0,
        "sr_bias_required": False,
        "sr_breakout": False,
        "composite_windows_put": [(3, 5), (8, 10)],
        "composite_windows_call": [(4, 8)],
        "death_zone": True,
        "gap_fade": True,
        "orb_scale": 50,
        "orb_cap": 0.85,
        "max_entry_bar": 20,
        "expiry_close_bar": None,
    },
    "V6": {
        "cooldown": 5,
        "min_confidence": 0.55,
        "trail_pct": 0.010,
        "put_max_hold": 22,
        "call_max_hold": 20,
        "sr_stop_buffer": 0.004,
        "vix_floor": 12,
        "vix_ceil": 35,
        "sr_gap_min": 150,
        "sr_bias_required": True,
        "sr_breakout": False,
        "composite_windows_put": [(3, 5), (8, 10)],
        "composite_windows_call": [(4, 8)],
        "death_zone": False,
        "gap_fade": True,
        "orb_scale": 10,
        "orb_cap": 0.80,
        "max_entry_bar": 19,
        "expiry_close_bar": 19,
    },
    "V7": {
        "cooldown": 2,
        "min_confidence": 0.45,
        "trail_pct": 0.006,       # 0.6% — sweet spot
        "put_max_hold": 18,       # 4.5 hours
        "call_max_hold": 15,      # 3.75 hours
        "sr_stop_buffer": 0.004,  # Entry-based stop
        "vix_floor": 11,          # Allow 11-12 range
        "vix_ceil": 35,
        "sr_gap_min": 75,         # Relaxed from 150
        "sr_bias_required": True, # Keep bias awareness
        "sr_breakout": False,     # Still removed (0% WR)
        "composite_windows_put": [(3, 5), (8, 10)],
        "composite_windows_call": [(4, 8)],
        "death_zone": False,      # Removed (from V6)
        "gap_fade": True,         # Keep (from V4)
        "orb_scale": 10,          # Fixed scaling (from V6)
        "orb_cap": 0.80,
        "max_entry_bar": 20,
        "expiry_close_bar": 19,   # Keep expiry safety
    },
}


# ===========================================================================
# UNIFIED ENTRY DETECTION
# ===========================================================================

def detect_entries(version, bar_idx, path, support, resistance, vix, gap_pct,
                   composite_action, composite_conf, is_expiry,
                   prev_high, prev_low, above_sma50, above_sma20,
                   bias_val="neutral"):
    """Unified entry detection driven by VERSION_CONFIG."""
    cfg = VERSION_CONFIG[version]
    signals = []
    spot = path[bar_idx]
    MIN_CONF = cfg["min_confidence"]

    # V6/V7: No entries before bar 1 or after max_entry_bar
    if version in ("V6", "V7"):
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
                if -1.2 <= gap_pct < -0.5 and vix >= 13:
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
        else:
            # V3 style: no gap fade
            if gap_pct < -0.3:
                conf = min(0.90, 0.60 + abs(gap_pct) * 0.10)
                signals.append(("BUY_PUT", "gap_entry", conf, False))
                if gap_pct <= -0.8 and vix >= 15:
                    signals.append(("BUY_PUT", "gap_zero_hero", 0.70, True))
            elif gap_pct > 0.5 and above_sma50:
                conf = min(0.85, 0.55 + gap_pct * 0.08)
                signals.append(("BUY_CALL", "gap_entry", conf, False))

    # 2. ORB ENTRY
    orb_bar = 1 if version in ("V3", "V4") else (1, 2)
    if (bar_idx == 1 or (isinstance(orb_bar, tuple) and bar_idx in orb_bar)) and len(path) >= 2:
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
        prev_spot = path[bar_idx - 1]

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

    # 4. S/R BREAKOUT (V3 only)
    if cfg["sr_breakout"] and bar_idx >= 2 and bar_idx <= 18:
        prev_spot = path[bar_idx - 1]
        momentum = abs(spot - prev_spot) / spot * 100
        if support and spot < support and momentum > 0.1:
            conf = min(0.75, 0.55 + momentum * 0.10)
            signals.append(("BUY_PUT", "sr_breakout", conf, False))
        if resistance and spot > resistance and momentum > 0.1:
            conf = min(0.75, 0.55 + momentum * 0.10)
            signals.append(("BUY_CALL", "sr_breakout", conf, False))

    # 5. COMPOSITE
    if composite_conf >= MIN_CONF:
        in_put_window = any(lo <= bar_idx <= hi for lo, hi in cfg["composite_windows_put"])
        in_call_window = any(lo <= bar_idx <= hi for lo, hi in cfg["composite_windows_call"])

        if composite_action == "BUY_PUT" and in_put_window:
            if cfg["death_zone"] and (0.60 <= composite_conf < 0.70):
                pass  # Skip death zone
            else:
                signals.append(("BUY_PUT", "composite", composite_conf, False))
        elif composite_action == "BUY_CALL" and in_call_window and vix < 12:
            if composite_conf >= 0.75:
                signals.append(("BUY_CALL", "composite", composite_conf, False))

    return signals


# ===========================================================================
# UNIFIED EXIT LOGIC
# ===========================================================================

def check_exit(version, trade, bar_idx, bar_spot, bar_dte, vix,
               support, resistance, is_expiry, path):
    """Unified exit logic driven by VERSION_CONFIG."""
    cfg = VERSION_CONFIG[version]
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

    # Zero-hero exits (same for all)
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
        if cfg["sr_stop_buffer"]:
            # V6/V7: entry-based stop with buffer
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
        else:
            # V3/V4: support-based stop
            if not trade.get("sr_target_hit", False):
                if resistance and bar_spot >= resistance:
                    trade["sr_target_hit"] = True
                    trade["best_fav"] = bar_spot
                if support and bar_spot < support:
                    return "sr_stop"
            else:
                if bar_spot < best_fav - trail_dist:
                    return "sr_combo_trail"

        if bars_held >= cfg["call_max_hold"]:
            return "time_exit"
        return None

    return None


# ===========================================================================
# V3/V4 POSITION SIZING (backwards)
# ===========================================================================

def get_lot_count_legacy(vix, zero_hero=False):
    if zero_hero: return 1
    if vix < 12: mult = 2.0
    elif vix < 15: mult = 1.5
    elif vix < 20: mult = 1.0
    elif vix < 25: mult = 0.7
    elif vix < 30: mult = 0.5
    else: mult = 0.3
    base = max(1, int(CAPITAL * 0.08 / (50 * LOT_SIZE)))
    return min(5, max(1, int(base * mult)))


def get_lot_count_span(vix, zero_hero=False):
    """V6/V7: SPAN margin-based sizing."""
    if zero_hero: return 1
    SPAN = {12: 30000, 15: 35000, 20: 40000, 25: 50000, 30: 60000}
    span_per_lot = 40000
    for threshold in sorted(SPAN.keys()):
        if vix <= threshold:
            span_per_lot = SPAN[threshold]
            break
    else:
        span_per_lot = 60000
    available = CAPITAL * 0.70
    max_lots = max(1, int(available / span_per_lot))
    return min(2, max_lots)


# ===========================================================================
# DAY SIMULATION
# ===========================================================================

def simulate_day(row, row_idx, nifty_df, equity, close_prices,
                 version="V7", use_realistic_path=False):
    """Simulate a single trading day."""
    entry_spot = float(row["Open"])
    day_high = float(row["High"])
    day_low = float(row["Low"])
    day_close = float(row["Close"])
    vix = float(row["VIX"]) if pd.notna(row["VIX"]) else 14.0
    dow = str(row["DOW"])
    above_sma50 = bool(row["AboveSMA50"]) if pd.notna(row.get("AboveSMA50")) else True
    above_sma20 = bool(row["AboveSMA20"]) if pd.notna(row.get("AboveSMA20")) else True
    rsi = float(row["RSI"]) if pd.notna(row.get("RSI")) else 50
    prev_change = float(row["PrevChange%"]) if pd.notna(row.get("PrevChange%")) else 0
    vix_spike = bool(row["VIXSpike"]) if pd.notna(row.get("VIXSpike")) else False
    sma20 = float(row["SMA20"]) if pd.notna(row.get("SMA20")) else None
    sma50 = float(row["SMA50"]) if pd.notna(row.get("SMA50")) else None
    prev_high = float(row["PrevHigh"]) if pd.notna(row.get("PrevHigh")) else day_high
    prev_low = float(row["PrevLow"]) if pd.notna(row.get("PrevLow")) else day_low
    is_expiry = bool(row.get("IsExpiry", False))
    dte_market = float(row.get("DTE", 2.0))
    ema9 = float(row["EMA9"]) if pd.notna(row.get("EMA9")) else None
    ema21 = float(row["EMA21"]) if pd.notna(row.get("EMA21")) else None
    weekly_sma = float(row["WeeklySMA"]) if pd.notna(row.get("WeeklySMA")) else None
    gap_pct = float(row["GapPct"]) if pd.notna(row.get("GapPct")) else 0
    date_str = str(nifty_df.index[row_idx].date())

    cfg = VERSION_CONFIG[version]

    # VIX skip
    if vix < cfg["vix_floor"] or vix > cfg["vix_ceil"]:
        return 0, [{"action": "SKIP", "reason": f"VIX {vix:.1f} out of range",
                     "date": date_str}]

    # S/R levels
    support, resistance = sr_multi_method(
        entry_spot, prev_high, prev_low, sma20, sma50,
        close_history=close_prices, idx=row_idx)

    # Composite scoring
    scores = compute_composite(
        version, vix, above_sma50, above_sma20, rsi, dow, prev_change,
        vix_spike, entry_spot, support, resistance,
        ema9=ema9, ema21=ema21, weekly_sma=weekly_sma)
    best_composite = max(scores, key=scores.get)
    total_score = sum(scores.values())
    composite_conf = scores[best_composite] / total_score if total_score > 0 else 0

    # Bias for V6/V7
    bias_val = "neutral"
    if above_sma50 and above_sma20:
        bias_val = "strong_bullish" if ema9 and ema21 and ema9 > ema21 else "bullish"
    elif not above_sma50 and not above_sma20:
        bias_val = "strong_bearish" if ema9 and ema21 and ema9 < ema21 else "bearish"

    # Generate path
    np.random.seed(int(abs(entry_spot * 100)) % 2**31 + row_idx)
    if use_realistic_path:
        path = generate_intraday_path_realistic(entry_spot, day_high, day_low, day_close)
    else:
        path = generate_intraday_path(entry_spot, day_high, day_low, day_close)

    open_trades = []
    closed_trades = []
    total_day_trades = 0
    last_exit_bar = -99

    for bar_idx in range(TOTAL_BARS):
        bar_spot = path[bar_idx]
        bar_dte = max(0.05, dte_market - bar_idx * 15 / 1440)

        # 1. CHECK EXITS
        trades_to_close = []
        for ti, trade in enumerate(open_trades):
            exit_signal = check_exit(version, trade, bar_idx, bar_spot, bar_dte,
                                     vix, support, resistance, is_expiry, path)
            if exit_signal:
                exit_prem = bs_premium(bar_spot, trade["strike"], bar_dte,
                                       vix, trade["opt_type"])
                pnl = (exit_prem - trade["entry_prem"]) * trade["qty"] - 80
                trade["exit_bar"] = bar_idx
                trade["exit_spot"] = round(bar_spot, 0)
                trade["exit_prem"] = round(exit_prem, 2)
                trade["exit_reason"] = exit_signal
                trade["intraday_pnl"] = round(pnl, 0)
                trade["total_pnl"] = round(pnl, 0)
                trades_to_close.append(ti)
                last_exit_bar = bar_idx

        for ti in reversed(trades_to_close):
            closed_trades.append(open_trades.pop(ti))

        # 2. CHECK ENTRIES
        if (len(open_trades) < MAX_CONCURRENT
                and total_day_trades < MAX_TRADES_PER_DAY
                and bar_idx - last_exit_bar >= cfg["cooldown"]
                and bar_idx < cfg["max_entry_bar"]):

            entries = detect_entries(
                version, bar_idx, path, support, resistance, vix, gap_pct,
                best_composite, composite_conf, is_expiry,
                prev_high, prev_low, above_sma50, above_sma20,
                bias_val=bias_val)

            if entries:
                entries.sort(key=lambda x: x[2], reverse=True)
                action, entry_type, conf, is_zero_hero = entries[0]

                same_dir = [t for t in open_trades if t["action"] == action]
                if not same_dir:
                    strike, opt_type = get_strike_and_type(action, bar_spot, vix, is_zero_hero)

                    if version in ("V6", "V7"):
                        num_lots = get_lot_count_span(vix, is_zero_hero)
                    else:
                        num_lots = get_lot_count_legacy(vix, is_zero_hero)

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
                        "entry_bar": bar_idx,
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
                        "exit_bar": -1, "exit_spot": 0, "exit_prem": 0,
                        "exit_reason": "", "intraday_pnl": 0,
                        "overnight_pnl": 0, "total_pnl": 0,
                    }
                    open_trades.append(trade)
                    total_day_trades += 1

        # 3. UPDATE tracking
        for trade in open_trades:
            if trade["action"] == "BUY_CALL" and bar_spot > trade["best_fav"]:
                trade["best_fav"] = bar_spot
            elif trade["action"] == "BUY_PUT" and bar_spot < trade["best_fav"]:
                trade["best_fav"] = bar_spot

    # 4. FORCE CLOSE
    for trade in open_trades:
        exit_prem = bs_premium(day_close, trade["strike"],
                               max(0.05, dte_market - 24 * 15 / 1440),
                               vix, trade["opt_type"])
        pnl = (exit_prem - trade["entry_prem"]) * trade["qty"] - 80
        trade["exit_bar"] = TOTAL_BARS - 1
        trade["exit_spot"] = round(day_close, 0)
        trade["exit_prem"] = round(exit_prem, 2)
        trade["exit_reason"] = "eod_close"
        trade["intraday_pnl"] = round(pnl, 0)
        trade["total_pnl"] = round(pnl, 0)
        closed_trades.append(trade)

    # 5. BTST
    for trade in closed_trades:
        if version in ("V3",):
            btst_ok = (trade["action"] == "BUY_PUT" and trade["intraday_pnl"] > 0
                       and not is_expiry
                       and trade["exit_reason"] in ("eod_close", "trail_pct")
                       and row_idx + 1 < len(nifty_df))
        elif version == "V4":
            btst_ok = (trade["action"] == "BUY_PUT" and trade["intraday_pnl"] >= 0
                       and not is_expiry
                       and trade["exit_reason"] in ("eod_close", "trail_pct", "time_exit")
                       and row_idx + 1 < len(nifty_df))
        else:  # V6, V7
            btst_ok = (trade["action"] == "BUY_PUT" and trade["intraday_pnl"] > 0
                       and not is_expiry and vix < 20
                       and trade["exit_reason"] in ("eod_close", "time_exit")
                       and row_idx + 1 < len(nifty_df))

        if btst_ok:
            next_row = nifty_df.iloc[row_idx + 1]
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

def run_version(nifty, version, close_prices, use_realistic_path=False):
    equity = CAPITAL
    equity_curve = [CAPITAL]
    all_trades = []
    peak_equity = CAPITAL
    max_dd = 0
    daily_pnl_list = []
    entry_stats = defaultdict(lambda: {"count": 0, "pnl": 0, "wins": 0})
    exit_stats = defaultdict(lambda: {"count": 0, "pnl": 0, "wins": 0})

    for i in range(len(nifty)):
        row = nifty.iloc[i]
        day_pnl, day_trades = simulate_day(
            row, i, nifty, equity, close_prices, version, use_realistic_path)

        if len(day_trades) == 1 and day_trades[0].get("action") == "SKIP":
            equity_curve.append(equity)
            daily_pnl_list.append(0)
            continue

        equity += day_pnl
        daily_pnl_list.append(day_pnl)
        if equity > peak_equity: peak_equity = equity
        dd = (peak_equity - equity) / peak_equity * 100
        if dd > max_dd: max_dd = dd

        for t in day_trades:
            all_trades.append(t)
            et = t.get("entry_type", "?")
            entry_stats[et]["count"] += 1
            entry_stats[et]["pnl"] += t["total_pnl"]
            if t["total_pnl"] > 0: entry_stats[et]["wins"] += 1
            er = t.get("exit_reason", "?")
            exit_stats[er]["count"] += 1
            exit_stats[er]["pnl"] += t["total_pnl"]
            if t["total_pnl"] > 0: exit_stats[er]["wins"] += 1

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

    return {
        "version": version,
        "net_pnl": round(net_pnl),
        "return_pct": round(net_pnl / CAPITAL * 100, 1),
        "total_trades": total,
        "winning_trades": len(wins),
        "losing_trades": len(losses),
        "win_rate": round(wr, 1),
        "sharpe": round(sharpe, 2),
        "profit_factor": round(pf, 2),
        "max_drawdown": round(max_dd, 2),
        "avg_win": round(np.mean([t["total_pnl"] for t in wins]) if wins else 0),
        "avg_loss": round(np.mean([t["total_pnl"] for t in losses]) if losses else 0),
        "final_equity": round(equity),
        "entry_stats": dict(entry_stats),
        "exit_stats": dict(exit_stats),
    }


def print_table(results_dict, title):
    """Print comparison table."""
    versions = list(results_dict.keys())
    print(f"\n{'=' * 100}")
    print(f"  {title}")
    print(f"{'=' * 100}")

    header = f"{'Metric':<25}"
    for v in versions:
        header += f" {v:>18}"
    print(header)
    print("-" * 100)

    metrics = [
        ("Net P&L", lambda r: f"Rs {r['net_pnl']:>+,}"),
        ("Return %", lambda r: f"{r['return_pct']:>+.1f}%"),
        ("Total Trades", lambda r: f"{r['total_trades']}"),
        ("Win / Lose", lambda r: f"{r['winning_trades']}W / {r['losing_trades']}L"),
        ("Win Rate", lambda r: f"{r['win_rate']:.1f}%"),
        ("Sharpe Ratio", lambda r: f"{r['sharpe']:.2f}"),
        ("Profit Factor", lambda r: f"{r['profit_factor']:.2f}"),
        ("Max Drawdown", lambda r: f"{r['max_drawdown']:.2f}%"),
        ("Avg Win", lambda r: f"Rs {r['avg_win']:>+,}"),
        ("Avg Loss", lambda r: f"Rs {r['avg_loss']:>+,}"),
    ]

    for name, fmt in metrics:
        row = f"  {name:<23}"
        for v in versions:
            row += f" {fmt(results_dict[v]):>18}"
        print(row)

    # Exit reason breakdown
    print(f"\n  EXIT REASONS:")
    all_exits = set()
    for r in results_dict.values():
        all_exits.update(r["exit_stats"].keys())
    for er in sorted(all_exits):
        row = f"    {er:<21}"
        for v in versions:
            s = results_dict[v]["exit_stats"].get(er, {"count": 0, "pnl": 0, "wins": 0})
            wr = s["wins"] / s["count"] * 100 if s["count"] > 0 else 0
            row += f" {s['count']:>3}t Rs{s['pnl']:>+8,.0f} {wr:>3.0f}%"
        print(row)

    print("=" * 100)


# ===========================================================================
# MAIN
# ===========================================================================

if __name__ == "__main__":
    print("=" * 100)
    print("  BACKTEST-TO-LIVE GAP ANALYSIS + V7 HYBRID")
    print("  Period: January 2025 | Capital: Rs 200,000")
    print("  Tests: (1) Smooth paths, (2) Realistic noisy paths")
    print("=" * 100)

    nifty = download_real_data(start="2025-01-01", end="2025-02-01")
    close_prices = nifty["Close"].values.tolist()
    print(f"\nJan 2025: {len(nifty)} days | NIFTY {nifty['Close'].min():.0f}-{nifty['Close'].max():.0f} | VIX {nifty['VIX'].min():.1f}-{nifty['VIX'].max():.1f}")

    # ── TEST 1: SMOOTH PATHS (standard backtest) ──
    print("\n\n" + "#" * 100)
    print("  TEST 1: SMOOTH SYNTHETIC PATHS (how backtest normally runs)")
    print("  This is WHY V3 looks amazing — smooth paths don't trigger trail stops")
    print("#" * 100)

    smooth_results = {}
    for ver in ["V3", "V4", "V6", "V7"]:
        smooth_results[ver] = run_version(nifty, ver, close_prices, use_realistic_path=False)
        print(f"  {ver}: Rs {smooth_results[ver]['net_pnl']:>+,} | "
              f"{smooth_results[ver]['total_trades']}t | WR {smooth_results[ver]['win_rate']:.1f}%")

    print_table(smooth_results, "SMOOTH PATHS — V3 vs V4 vs V6 vs V7 (January 2025)")

    # ── TEST 2: REALISTIC NOISY PATHS (simulates live market) ──
    print("\n\n" + "#" * 100)
    print("  TEST 2: REALISTIC NOISY PATHS (simulates actual live market)")
    print("  This is WHAT HAPPENS in live — noise triggers V3's 0.3% trail on every pullback")
    print("#" * 100)

    noisy_results = {}
    for ver in ["V3", "V4", "V6", "V7"]:
        noisy_results[ver] = run_version(nifty, ver, close_prices, use_realistic_path=True)
        print(f"  {ver}: Rs {noisy_results[ver]['net_pnl']:>+,} | "
              f"{noisy_results[ver]['total_trades']}t | WR {noisy_results[ver]['win_rate']:.1f}%")

    print_table(noisy_results, "REALISTIC PATHS — V3 vs V4 vs V6 vs V7 (January 2025)")

    # ── DEGRADATION ANALYSIS ──
    print("\n\n" + "=" * 100)
    print("  DEGRADATION: SMOOTH -> REALISTIC (how much each version loses to noise)")
    print("=" * 100)
    print(f"  {'Version':<10} {'Smooth P&L':>15} {'Noisy P&L':>15} {'Degradation':>15} {'% Lost':>10}")
    print("-" * 70)
    for ver in ["V3", "V4", "V6", "V7"]:
        sp = smooth_results[ver]["net_pnl"]
        np_ = noisy_results[ver]["net_pnl"]
        deg = sp - np_
        pct = deg / sp * 100 if sp != 0 else 0
        marker = " <-- WORST" if pct == max((smooth_results[v]["net_pnl"] - noisy_results[v]["net_pnl"]) / max(smooth_results[v]["net_pnl"], 1) * 100 for v in ["V3", "V4", "V6", "V7"]) else ""
        best = " <-- BEST" if pct == min((smooth_results[v]["net_pnl"] - noisy_results[v]["net_pnl"]) / max(smooth_results[v]["net_pnl"], 1) * 100 for v in ["V3", "V4", "V6", "V7"]) else ""
        print(f"  {ver:<10} Rs {sp:>+10,} Rs {np_:>+10,} Rs {deg:>+10,} {pct:>+8.1f}%{marker}{best}")

    print("\n  CONCLUSION:")
    best_noisy = max(noisy_results.values(), key=lambda r: r["net_pnl"])
    best_sharpe = max(noisy_results.values(), key=lambda r: r["sharpe"])
    print(f"    Best absolute P&L on noisy data: {best_noisy['version']} (Rs {best_noisy['net_pnl']:>+,})")
    print(f"    Best risk-adjusted on noisy data: {best_sharpe['version']} (Sharpe {best_sharpe['sharpe']:.2f})")

    least_degraded = min(["V3", "V4", "V6", "V7"],
                         key=lambda v: (smooth_results[v]["net_pnl"] - noisy_results[v]["net_pnl"]) / max(smooth_results[v]["net_pnl"], 1))
    print(f"    Most noise-resistant:             {least_degraded}")
    print("=" * 100)

    # Save
    save_data = {
        "smooth": {v: {k: val for k, val in r.items() if k not in ("entry_stats", "exit_stats")}
                   for v, r in smooth_results.items()},
        "noisy": {v: {k: val for k, val in r.items() if k not in ("entry_stats", "exit_stats")}
                  for v, r in noisy_results.items()},
    }
    out_path = project_root / "data" / "v7_comparison_jan2025.json"
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")
