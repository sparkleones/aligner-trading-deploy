"""
V3 vs V4 vs V6 Comparison on January 2025 Out-of-Sample Data.

Tests all three model versions on January 2025 NIFTY data (unseen by any version).

V3 (Original Baseline):
  - COOLDOWN_BARS = 2, MIN_CONFIDENCE = 0.25, TRAIL_PCT = 0.3%
  - RSI: both <30 and >70 scored BUY_PUT (BUG)
  - DOW: Mon=PUT, Tue=PUT, Wed=CALL, Thu=PUT, Fri=CALL
  - sr_breakout entries ENABLED
  - Composite window: bars 2-4
  - VIX sizing: 2x at low VIX (backwards for buyers)

V4 (Optimized Entries):
  - COOLDOWN_BARS = 0, MIN_CONFIDENCE = 0.25, TRAIL_PCT = 0.3%
  - RSI: same bug (both <30 and >70 → BUY_PUT)
  - DOW: Mon=CALL, Tue=PUT, Wed=CALL, Thu=PUT, Fri=PUT
  - sr_breakout REMOVED
  - Composite window: bars 3-5 + 8-10
  - gap_fade for |gap| > 1.2%
  - Confidence death zone: skip 0.60-0.70

V6 (Live Fixes - Current Production):
  - COOLDOWN_BARS = 5, MIN_CONFIDENCE = 0.55, TRAIL_PCT = 1.0%
  - RSI: FIXED — RSI<30→BUY_CALL, RSI>70→BUY_PUT
  - S/R bounce: bias-aware, 150pt gap required
  - S/R stop: entry-based 0.4% buffer
  - VIX: SPAN margin sizing, guardrails <12 and >35
  - VIX smoothing: 3-bar average
  - ORB: scale by 10 not 50, cap 0.80
  - Expiry: auto-close by bar 19, no entries after bar 17
  - No entries before bar 1 or after bar 19

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
# COMPOSITE SCORING — Three Versions
# ===========================================================================

def compute_composite_v3(vix, above_sma50, above_sma20, rsi, dow,
                         prev_change, vix_spike, spot, support, resistance,
                         ema9=None, ema21=None, weekly_sma=None):
    """V3 composite: original DOW rules + RSI BUG (both → BUY_PUT)."""
    scores = {"BUY_CALL": 0.0, "BUY_PUT": 0.0}

    if vix < 12: scores["BUY_CALL"] += 3.0
    elif vix < 17: scores["BUY_PUT"] += 3.0
    elif vix < 25: scores["BUY_PUT"] += 3.5
    else: scores["BUY_PUT"] += 4.0

    if not above_sma50: scores["BUY_PUT"] += 2.0
    else: scores["BUY_CALL"] += 2.0

    if not above_sma20: scores["BUY_PUT"] += 1.0
    else: scores["BUY_CALL"] += 1.0

    # V3 RSI BUG: both oversold AND overbought → BUY_PUT
    if rsi < 30: scores["BUY_PUT"] += 1.5
    elif rsi > 70: scores["BUY_PUT"] += 1.5

    # V3 DOW rules
    dow_rules = {"Monday": "BUY_PUT", "Tuesday": "BUY_PUT",
                 "Wednesday": "BUY_CALL", "Thursday": "BUY_PUT",
                 "Friday": "BUY_CALL"}
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


def compute_composite_v4(vix, above_sma50, above_sma20, rsi, dow,
                         prev_change, vix_spike, spot, support, resistance,
                         ema9=None, ema21=None, weekly_sma=None):
    """V4 composite: corrected DOW rules, same RSI BUG."""
    scores = {"BUY_CALL": 0.0, "BUY_PUT": 0.0}

    if vix < 12: scores["BUY_CALL"] += 3.0
    elif vix < 17: scores["BUY_PUT"] += 3.0
    elif vix < 25: scores["BUY_PUT"] += 3.5
    else: scores["BUY_PUT"] += 4.0

    if not above_sma50: scores["BUY_PUT"] += 2.0
    else: scores["BUY_CALL"] += 2.0

    if not above_sma20: scores["BUY_PUT"] += 1.0
    else: scores["BUY_CALL"] += 1.0

    # V4: same RSI BUG as V3 (both → BUY_PUT)
    if rsi < 30: scores["BUY_PUT"] += 1.5
    elif rsi > 70: scores["BUY_PUT"] += 1.5

    # V4 DOW rules (data-driven corrections)
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


def compute_composite_v6(vix, above_sma50, above_sma20, rsi, dow,
                         prev_change, vix_spike, spot, support, resistance,
                         ema9=None, ema21=None, weekly_sma=None):
    """V6 composite: RSI FIXED, corrected DOW."""
    scores = {"BUY_CALL": 0.0, "BUY_PUT": 0.0}

    if vix < 12: scores["BUY_CALL"] += 3.0
    elif vix < 17: scores["BUY_PUT"] += 3.0
    elif vix < 25: scores["BUY_PUT"] += 3.5
    else: scores["BUY_PUT"] += 4.0

    if not above_sma50: scores["BUY_PUT"] += 2.0
    else: scores["BUY_CALL"] += 2.0

    if not above_sma20: scores["BUY_PUT"] += 1.0
    else: scores["BUY_CALL"] += 1.0

    # V6 RSI FIX: oversold → bounce (BUY_CALL), overbought → pullback (BUY_PUT)
    if rsi < 30: scores["BUY_CALL"] += 1.5
    elif rsi > 70: scores["BUY_PUT"] += 1.5

    # V6 uses same DOW as V4
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
# ENTRY DETECTION — Three Versions
# ===========================================================================

def detect_entries_v3(bar_idx, path, support, resistance, vix, gap_pct,
                      composite_action, composite_conf, is_expiry,
                      prev_high, prev_low, above_sma50, above_sma20):
    """V3: sr_breakout enabled, composite bars 2-4, no gap_fade."""
    signals = []
    spot = path[bar_idx]
    MIN_CONFIDENCE = 0.25

    # 1. GAP ENTRY (bar 0)
    if bar_idx == 0 and abs(gap_pct) >= 0.3:
        if gap_pct < -0.3:
            conf = min(0.90, 0.60 + abs(gap_pct) * 0.10)
            signals.append(("BUY_PUT", "gap_entry", conf, False))
            if gap_pct <= -0.8 and vix >= 15:
                signals.append(("BUY_PUT", "gap_zero_hero", 0.70, True))
        elif gap_pct > 0.5 and above_sma50:
            conf = min(0.85, 0.55 + gap_pct * 0.08)
            signals.append(("BUY_CALL", "gap_entry", conf, False))

    # 2. ORB ENTRY (bar 1)
    if bar_idx == 1 and len(path) >= 2:
        orb_high = max(path[0], path[1])
        orb_low = min(path[0], path[1])
        orb_range = orb_high - orb_low
        if orb_range > spot * 0.0015:
            if spot > orb_high:
                conf = min(0.85, 0.60 + (spot - orb_high) / spot * 50)
                if above_sma50 or vix < 14:
                    signals.append(("BUY_CALL", "orb_breakout_up", conf, False))
            elif spot < orb_low:
                conf = min(0.85, 0.60 + (orb_low - spot) / spot * 50)
                signals.append(("BUY_PUT", "orb_breakout_down", conf, False))

    # 3. S/R BOUNCE (bar 2+)
    if bar_idx >= 2 and bar_idx <= 18:
        prev_spot = path[bar_idx - 1]
        if support and abs(spot - support) / spot < 0.003:
            if spot > prev_spot:
                signals.append(("BUY_CALL", "sr_bounce_support", 0.65, False))
        if resistance and abs(spot - resistance) / spot < 0.003:
            if spot < prev_spot:
                signals.append(("BUY_PUT", "sr_bounce_resistance", 0.70, False))

    # 4. S/R BREAKOUT (V3 only)
    if bar_idx >= 2 and bar_idx <= 18:
        prev_spot = path[bar_idx - 1]
        momentum = abs(spot - prev_spot) / spot * 100
        if support and spot < support and momentum > 0.1:
            conf = min(0.75, 0.55 + momentum * 0.10)
            signals.append(("BUY_PUT", "sr_breakout", conf, False))
            if momentum > 0.3 and vix >= 15:
                signals.append(("BUY_PUT", "sr_zero_hero", 0.65, True))
        if resistance and spot > resistance and momentum > 0.1:
            conf = min(0.75, 0.55 + momentum * 0.10)
            signals.append(("BUY_CALL", "sr_breakout", conf, False))

    # 5. COMPOSITE (bars 2-4)
    if composite_conf >= MIN_CONFIDENCE:
        if composite_action == "BUY_PUT" and 2 <= bar_idx <= 4:
            if not (0.60 <= composite_conf < 0.70):
                signals.append(("BUY_PUT", "composite", composite_conf, False))
        elif composite_action == "BUY_CALL" and 2 <= bar_idx <= 4 and vix < 12:
            if composite_conf >= 0.75:
                signals.append(("BUY_CALL", "composite", composite_conf, False))

    return signals


def detect_entries_v4(bar_idx, path, support, resistance, vix, gap_pct,
                      composite_action, composite_conf, is_expiry,
                      prev_high, prev_low, above_sma50, above_sma20):
    """V4: no sr_breakout, gap_fade added, composite bars 3-5 + 8-10."""
    signals = []
    spot = path[bar_idx]
    MIN_CONFIDENCE = 0.25

    # 1. GAP ENTRY with gap_fade
    if bar_idx == 0 and abs(gap_pct) >= 0.3:
        is_large_gap = abs(gap_pct) > 1.2
        if gap_pct < -0.3:
            if is_large_gap:
                conf = min(0.85, 0.65 + abs(gap_pct) * 0.05)
                signals.append(("BUY_CALL", "gap_fade", conf, False))
            else:
                conf = min(0.90, 0.60 + abs(gap_pct) * 0.10)
                signals.append(("BUY_PUT", "gap_entry", conf, False))
            if -1.2 <= gap_pct < -0.5 and vix >= 13:
                signals.append(("BUY_PUT", "gap_zero_hero", 0.70, True))
        elif gap_pct > 0.3:
            if is_large_gap:
                conf = min(0.85, 0.65 + gap_pct * 0.05)
                signals.append(("BUY_PUT", "gap_fade", conf, False))
            elif gap_pct > 0.5 and above_sma50:
                conf = min(0.85, 0.55 + gap_pct * 0.08)
                signals.append(("BUY_CALL", "gap_entry", conf, False))

    # 2. ORB ENTRY
    if bar_idx == 1 and len(path) >= 2:
        orb_high = max(path[0], path[1])
        orb_low = min(path[0], path[1])
        orb_range = orb_high - orb_low
        if orb_range > spot * 0.0015:
            if spot > orb_high:
                conf = min(0.85, 0.60 + (spot - orb_high) / spot * 50)
                if above_sma50 or vix < 14:
                    signals.append(("BUY_CALL", "orb_breakout_up", conf, False))
            elif spot < orb_low:
                conf = min(0.85, 0.60 + (orb_low - spot) / spot * 50)
                signals.append(("BUY_PUT", "orb_breakout_down", conf, False))

    # 3. S/R BOUNCE
    if bar_idx >= 2 and bar_idx <= 18:
        prev_spot = path[bar_idx - 1]
        if support and abs(spot - support) / spot < 0.003:
            if spot > prev_spot:
                signals.append(("BUY_CALL", "sr_bounce_support", 0.65, False))
        if resistance and abs(spot - resistance) / spot < 0.003:
            if spot < prev_spot:
                signals.append(("BUY_PUT", "sr_bounce_resistance", 0.70, False))

    # 4. sr_breakout REMOVED in V4

    # 5. COMPOSITE (bars 3-5 + 8-10)
    if composite_conf >= MIN_CONFIDENCE:
        put_window = (3 <= bar_idx <= 5) or (8 <= bar_idx <= 10)
        if composite_action == "BUY_PUT" and put_window:
            if not (0.60 <= composite_conf < 0.70):
                signals.append(("BUY_PUT", "composite", composite_conf, False))
        elif composite_action == "BUY_CALL" and 4 <= bar_idx <= 8 and vix < 12:
            if composite_conf >= 0.75:
                signals.append(("BUY_CALL", "composite", composite_conf, False))

    return signals


def detect_entries_v6(bar_idx, path, support, resistance, vix, gap_pct,
                      composite_action, composite_conf, is_expiry,
                      prev_high, prev_low, above_sma50, above_sma20,
                      bias_val="neutral"):
    """V6: all live fixes — bias-aware S/R, wider ORB, VIX guardrails, expiry handling."""
    signals = []
    spot = path[bar_idx]
    MIN_CONFIDENCE = 0.55

    # V6: No entries before bar 1 (9:30 equivalent) or after bar 19 (14:45 equivalent)
    if bar_idx < 1 or bar_idx > 19:
        return signals

    # V6: On expiry day, no entries after bar 17 (1 PM equivalent)
    if is_expiry and bar_idx > 17:
        return signals

    # V6: VIX guardrails
    if vix < 12 or vix > 35:
        return signals

    # 1. GAP ENTRY with gap_fade (same as V4 but higher confidence threshold)
    if bar_idx == 1 and abs(gap_pct) >= 0.3:
        is_large_gap = abs(gap_pct) > 1.2
        if gap_pct < -0.3:
            if is_large_gap:
                conf = min(0.85, 0.65 + abs(gap_pct) * 0.05)
                if conf >= MIN_CONFIDENCE:
                    signals.append(("BUY_CALL", "gap_fade", conf, False))
            elif gap_pct < -0.5 and above_sma50:
                conf = min(0.85, 0.55 + abs(gap_pct) * 0.08)
                if conf >= MIN_CONFIDENCE:
                    signals.append(("BUY_PUT", "gap_entry", conf, False))
        elif gap_pct > 0.3:
            if is_large_gap:
                conf = min(0.85, 0.65 + gap_pct * 0.05)
                if conf >= MIN_CONFIDENCE:
                    signals.append(("BUY_PUT", "gap_fade", conf, False))
            elif gap_pct > 0.5 and above_sma50:
                conf = min(0.85, 0.55 + gap_pct * 0.08)
                if conf >= MIN_CONFIDENCE:
                    signals.append(("BUY_CALL", "gap_entry", conf, False))

    # 2. ORB ENTRY — V6: scale by 10 not 50, cap at 0.80
    if bar_idx in (1, 2) and len(path) >= 2:
        orb_high = max(path[0], path[1])
        orb_low = min(path[0], path[1])
        orb_range = orb_high - orb_low
        if orb_range > spot * 0.0015:
            if spot > orb_high:
                conf = min(0.80, 0.55 + (spot - orb_high) / orb_high * 10)
                if conf >= MIN_CONFIDENCE and (above_sma50 or vix < 14):
                    signals.append(("BUY_CALL", "orb_breakout_up", conf, False))
            elif spot < orb_low:
                conf = min(0.80, 0.55 + (orb_low - spot) / orb_low * 10)
                if conf >= MIN_CONFIDENCE:
                    signals.append(("BUY_PUT", "orb_breakout_down", conf, False))

    # 3. S/R BOUNCE — V6: bias-aware, 150pt gap required
    sr_dist = (resistance - support) if (support and resistance) else 0
    sr_valid = sr_dist >= 150

    if bar_idx >= 2 and bar_idx <= 18 and sr_valid:
        prev_spot = path[bar_idx - 1]

        # Bounce off support (need bullish bias)
        if support and abs(spot - support) / spot < 0.003:
            if spot > prev_spot and bias_val in ("bullish", "strong_bullish", "neutral"):
                signals.append(("BUY_CALL", "sr_bounce_support", 0.65, False))

        # Rejection at resistance (need bearish bias)
        if resistance and abs(spot - resistance) / spot < 0.003:
            if spot < prev_spot and bias_val in ("bearish", "strong_bearish", "neutral"):
                signals.append(("BUY_PUT", "sr_bounce_resistance", 0.70, False))

    # 4. sr_breakout REMOVED (same as V4)

    # 5. COMPOSITE — V6: no death zone, MIN_CONFIDENCE=0.55
    if composite_conf >= MIN_CONFIDENCE:
        put_window = (3 <= bar_idx <= 5) or (8 <= bar_idx <= 10)
        call_window = (4 <= bar_idx <= 8)
        if composite_action == "BUY_PUT" and put_window:
            signals.append(("BUY_PUT", "composite", composite_conf, False))
        elif composite_action == "BUY_CALL" and call_window and vix < 12 and composite_conf >= 0.75:
            signals.append(("BUY_CALL", "composite", composite_conf, False))

    return signals


# ===========================================================================
# V6 EXIT LOGIC (different from V3/V4)
# ===========================================================================

def check_trade_exit_v6(trade, bar_idx, bar_spot, bar_dte, vix,
                         support, resistance, is_expiry, path):
    """V6 exit logic: wider trail, entry-based S/R stop."""
    action = trade["action"]
    entry_bar = trade["entry_bar"]
    bars_held = bar_idx - entry_bar
    best_fav = trade["best_fav"]
    entry_spot = trade["entry_spot"]
    is_zero_hero = trade.get("is_zero_hero", False)

    TRAIL_PCT = 0.010    # V6: 1.0% trail (was 0.3%)
    SR_STOP_BUFFER = 0.004  # V6: 0.4% entry-based stop
    PUT_MAX_HOLD = 22    # V6: 330 bars / 15 = 22
    CALL_MAX_HOLD = 20   # V6: 300 bars / 15 = 20

    if bars_held < 1:
        return None

    trail_dist = entry_spot * TRAIL_PCT

    # Expiry day: force close after bar 19 (2:45 PM equivalent)
    if is_expiry and bar_idx >= 19:
        return "expiry_day_close"

    # Zero-hero exits (same as V3/V4)
    if is_zero_hero:
        zh_trail = entry_spot * 0.008
        entry_prem = trade["entry_prem"]
        bar_prem = bs_premium(bar_spot, trade["strike"], bar_dte, vix, trade["opt_type"])
        current_return = (bar_prem - entry_prem) / max(entry_prem, 1)
        if current_return >= 2.0:
            return "zero_hero_target"
        if current_return <= -0.60:
            return "zero_hero_stop"
        if current_return >= 0.5:
            if action == "BUY_PUT" and bar_spot > best_fav + zh_trail:
                return "zero_hero_trail"
            elif action == "BUY_CALL" and bar_spot < best_fav - zh_trail:
                return "zero_hero_trail"
        if bars_held >= 10:
            return "zero_hero_time"
        return None

    # PUT exit: V6 wider trail
    if action == "BUY_PUT":
        if bars_held >= 3:
            if bar_spot > best_fav + trail_dist:
                return "trail_pct"
        if bars_held >= PUT_MAX_HOLD:
            return "time_exit"
        return None

    # CALL exit: V6 entry-based S/R stop with buffer
    if action == "BUY_CALL":
        call_stop = entry_spot * (1 - SR_STOP_BUFFER)
        if spot_val := bar_spot:
            if not trade.get("sr_target_hit", False):
                if resistance and bar_spot >= resistance:
                    trade["sr_target_hit"] = True
                    trade["best_fav"] = bar_spot
                if bar_spot < call_stop and bars_held >= 3:
                    return "sr_stop"
            else:
                if bar_spot < best_fav - trail_dist:
                    return "sr_combo_trail"
        if bars_held >= CALL_MAX_HOLD:
            return "time_exit"
        return None

    return None


# ===========================================================================
# V6 POSITION SIZING (SPAN margin based)
# ===========================================================================

def get_lot_count_v6(vix, zero_hero=False):
    """V6: SPAN margin-based sizing with 30% capital buffer."""
    if zero_hero:
        return 1

    # SPAN margin estimation by VIX level
    SPAN_MARGIN_EST = {12: 30000, 15: 35000, 20: 40000, 25: 50000, 30: 60000}

    span_per_lot = 40000  # default
    for threshold in sorted(SPAN_MARGIN_EST.keys()):
        if vix <= threshold:
            span_per_lot = SPAN_MARGIN_EST[threshold]
            break
    else:
        span_per_lot = 60000

    available_margin = CAPITAL * 0.70  # 30% buffer
    max_lots = max(1, int(available_margin / span_per_lot))
    return min(2, max_lots)


def get_lot_count_v3v4(vix, zero_hero=False):
    """V3/V4 original sizing (backwards — 2x at low VIX)."""
    if zero_hero:
        return 1
    if vix < 12: mult = 2.0
    elif vix < 15: mult = 1.5
    elif vix < 20: mult = 1.0
    elif vix < 25: mult = 0.7
    elif vix < 30: mult = 0.5
    else: mult = 0.3
    base = max(1, int(CAPITAL * 0.08 / (50 * LOT_SIZE)))
    return min(5, max(1, int(base * mult)))


# ===========================================================================
# UNIFIED DAY SIMULATION (version-aware)
# ===========================================================================

def simulate_day(row, row_idx, nifty_df, equity, close_prices, version="V6"):
    """Simulate a single trading day with version-specific logic."""
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

    # Version-specific parameters
    if version == "V3":
        cooldown = 2
    elif version == "V4":
        cooldown = 0
    else:  # V6
        cooldown = 5

    # VIX skip (V6 has different thresholds)
    if version == "V6":
        if vix < 12 or vix > 35:
            return 0, [{"action": "SKIP", "reason": f"VIX out of range ({vix:.1f})",
                         "date": date_str, "dow": dow, "vix": round(vix, 1),
                         "is_expiry": is_expiry}]
    else:
        if vix < 10:
            return 0, [{"action": "SKIP", "reason": f"VIX too low ({vix:.1f})",
                         "date": date_str, "dow": dow, "vix": round(vix, 1),
                         "is_expiry": is_expiry}]

    # S/R levels
    support, resistance = sr_multi_method(
        entry_spot, prev_high, prev_low, sma20, sma50,
        close_history=close_prices, idx=row_idx
    )

    # Composite scoring (version-aware)
    if version == "V3":
        scores = compute_composite_v3(
            vix, above_sma50, above_sma20, rsi, dow, prev_change, vix_spike,
            entry_spot, support, resistance, ema9=ema9, ema21=ema21, weekly_sma=weekly_sma)
    elif version == "V4":
        scores = compute_composite_v4(
            vix, above_sma50, above_sma20, rsi, dow, prev_change, vix_spike,
            entry_spot, support, resistance, ema9=ema9, ema21=ema21, weekly_sma=weekly_sma)
    else:
        scores = compute_composite_v6(
            vix, above_sma50, above_sma20, rsi, dow, prev_change, vix_spike,
            entry_spot, support, resistance, ema9=ema9, ema21=ema21, weekly_sma=weekly_sma)

    best_composite = max(scores, key=scores.get)
    total_score = sum(scores.values())
    composite_conf = scores[best_composite] / total_score if total_score > 0 else 0

    # Derive bias for V6
    bias_val = "neutral"
    if above_sma50 and above_sma20:
        bias_val = "strong_bullish" if ema9 and ema21 and ema9 > ema21 else "bullish"
    elif not above_sma50 and not above_sma20:
        bias_val = "strong_bearish" if ema9 and ema21 and ema9 < ema21 else "bearish"

    # Intraday path (deterministic)
    np.random.seed(int(abs(entry_spot * 100)) % 2**31 + row_idx)
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
            if version == "V6":
                exit_signal = check_trade_exit_v6(
                    trade, bar_idx, bar_spot, bar_dte, vix,
                    support, resistance, is_expiry, path)
            else:
                exit_signal = check_trade_exit(
                    trade, bar_idx, bar_spot, bar_dte, vix,
                    support, resistance, is_expiry, path)

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
        max_bar_entry = 19 if version == "V6" else 20
        if (len(open_trades) < MAX_CONCURRENT
                and total_day_trades < MAX_TRADES_PER_DAY
                and bar_idx - last_exit_bar >= cooldown
                and bar_idx < max_bar_entry):

            if version == "V3":
                entries = detect_entries_v3(
                    bar_idx, path, support, resistance, vix, gap_pct,
                    best_composite, composite_conf, is_expiry,
                    prev_high, prev_low, above_sma50, above_sma20)
            elif version == "V4":
                entries = detect_entries_v4(
                    bar_idx, path, support, resistance, vix, gap_pct,
                    best_composite, composite_conf, is_expiry,
                    prev_high, prev_low, above_sma50, above_sma20)
            else:
                entries = detect_entries_v6(
                    bar_idx, path, support, resistance, vix, gap_pct,
                    best_composite, composite_conf, is_expiry,
                    prev_high, prev_low, above_sma50, above_sma20,
                    bias_val=bias_val)

            if entries:
                entries.sort(key=lambda x: x[2], reverse=True)
                action, entry_type, conf, is_zero_hero = entries[0]

                same_dir = [t for t in open_trades if t["action"] == action]
                if not same_dir:
                    strike, opt_type = get_strike_and_type(action, bar_spot, vix, is_zero_hero)

                    if version == "V6":
                        num_lots = get_lot_count_v6(vix, is_zero_hero)
                    else:
                        num_lots = get_lot_count_v3v4(vix, is_zero_hero)

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
                        "exit_bar": -1,
                        "exit_spot": 0,
                        "exit_prem": 0,
                        "exit_reason": "",
                        "intraday_pnl": 0,
                        "overnight_pnl": 0,
                        "total_pnl": 0,
                    }
                    open_trades.append(trade)
                    total_day_trades += 1

        # 3. UPDATE tracking
        for trade in open_trades:
            if trade["action"] == "BUY_CALL" and bar_spot > trade["best_fav"]:
                trade["best_fav"] = bar_spot
            elif trade["action"] == "BUY_PUT" and bar_spot < trade["best_fav"]:
                trade["best_fav"] = bar_spot

    # 4. FORCE CLOSE remaining at EOD
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

    # 5. BTST (version-aware)
    for trade in closed_trades:
        if version == "V3":
            btst_eligible = (
                trade["action"] == "BUY_PUT"
                and trade["intraday_pnl"] > 0
                and not is_expiry
                and trade["exit_reason"] in ("eod_close", "trail_pct")
                and row_idx + 1 < len(nifty_df))
        elif version == "V4":
            btst_eligible = (
                trade["action"] == "BUY_PUT"
                and trade["intraday_pnl"] >= 0
                and not is_expiry
                and trade["exit_reason"] in ("eod_close", "trail_pct", "time_exit")
                and row_idx + 1 < len(nifty_df))
        else:  # V6
            btst_eligible = (
                trade["action"] == "BUY_PUT"
                and trade["intraday_pnl"] > 0
                and not is_expiry
                and vix < 20
                and trade["exit_reason"] in ("eod_close", "time_exit")
                and row_idx + 1 < len(nifty_df))

        if btst_eligible:
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
            trade["overnight_held"] = True
        else:
            trade["overnight_held"] = False

    if not closed_trades:
        return 0, [{"action": "SKIP", "reason": "No entry signals",
                     "date": date_str, "dow": dow, "vix": round(vix, 1),
                     "is_expiry": is_expiry}]

    total_pnl = sum(t["total_pnl"] for t in closed_trades)
    return total_pnl, closed_trades


# ===========================================================================
# RUN COMPARISON
# ===========================================================================

def run_version(nifty, version, close_prices):
    """Run a single version and return results dict."""
    equity = CAPITAL
    equity_curve = [CAPITAL]
    all_trades = []
    peak_equity = CAPITAL
    max_dd = 0
    daily_pnl_list = []
    entry_type_stats = defaultdict(lambda: {"count": 0, "pnl": 0, "wins": 0})
    exit_reason_stats = defaultdict(lambda: {"count": 0, "pnl": 0, "wins": 0})

    for i in range(len(nifty)):
        row = nifty.iloc[i]
        day_pnl, day_trades = simulate_day(row, i, nifty, equity, close_prices, version)

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
            et = t.get("entry_type", "unknown")
            entry_type_stats[et]["count"] += 1
            entry_type_stats[et]["pnl"] += t["total_pnl"]
            if t["total_pnl"] > 0:
                entry_type_stats[et]["wins"] += 1

            er = t.get("exit_reason", "unknown")
            exit_reason_stats[er]["count"] += 1
            exit_reason_stats[er]["pnl"] += t["total_pnl"]
            if t["total_pnl"] > 0:
                exit_reason_stats[er]["wins"] += 1

        equity_curve.append(equity)

    # Calculate stats
    net_pnl = equity - CAPITAL
    total_trades = len(all_trades)
    wins = [t for t in all_trades if t["total_pnl"] > 0]
    losses = [t for t in all_trades if t["total_pnl"] <= 0]
    win_rate = len(wins) / total_trades * 100 if total_trades else 0

    daily_arr = np.array([d for d in daily_pnl_list if d != 0])
    sharpe = 0
    if len(daily_arr) > 1 and daily_arr.std() > 0:
        sharpe = (daily_arr.mean() / daily_arr.std()) * np.sqrt(252)

    gross_wins = sum(t["total_pnl"] for t in wins)
    gross_losses = abs(sum(t["total_pnl"] for t in losses))
    profit_factor = gross_wins / gross_losses if gross_losses > 0 else float("inf")

    avg_win = np.mean([t["total_pnl"] for t in wins]) if wins else 0
    avg_loss = np.mean([t["total_pnl"] for t in losses]) if losses else 0

    return {
        "version": version,
        "net_pnl": round(net_pnl),
        "return_pct": round(net_pnl / CAPITAL * 100, 1),
        "total_trades": total_trades,
        "winning_trades": len(wins),
        "losing_trades": len(losses),
        "win_rate": round(win_rate, 1),
        "sharpe": round(sharpe, 2),
        "profit_factor": round(profit_factor, 2),
        "max_drawdown": round(max_dd, 2),
        "avg_win": round(avg_win),
        "avg_loss": round(avg_loss),
        "final_equity": round(equity),
        "equity_curve": equity_curve,
        "all_trades": all_trades,
        "entry_type_stats": dict(entry_type_stats),
        "exit_reason_stats": dict(exit_reason_stats),
    }


def print_comparison(results):
    """Print side-by-side comparison table."""
    print("\n" + "=" * 90)
    print("  V3 vs V4 vs V6 COMPARISON — JANUARY 2025 (OUT-OF-SAMPLE)")
    print("  Capital: Rs 200,000 | Data: Yahoo Finance (^NSEI + ^INDIAVIX)")
    print("=" * 90)

    header = f"{'Metric':<25} {'V3 (Original)':>18} {'V4 (Optimized)':>18} {'V6 (Live Fixes)':>18}"
    print(header)
    print("-" * 90)

    metrics = [
        ("Net P&L", lambda r: f"Rs {r['net_pnl']:>+,}"),
        ("Return %", lambda r: f"{r['return_pct']:>+.1f}%"),
        ("Total Trades", lambda r: f"{r['total_trades']}"),
        ("Winning", lambda r: f"{r['winning_trades']}"),
        ("Losing", lambda r: f"{r['losing_trades']}"),
        ("Win Rate", lambda r: f"{r['win_rate']:.1f}%"),
        ("Sharpe Ratio", lambda r: f"{r['sharpe']:.2f}"),
        ("Profit Factor", lambda r: f"{r['profit_factor']:.2f}"),
        ("Max Drawdown", lambda r: f"{r['max_drawdown']:.2f}%"),
        ("Avg Win", lambda r: f"Rs {r['avg_win']:>+,}"),
        ("Avg Loss", lambda r: f"Rs {r['avg_loss']:>+,}"),
        ("Final Equity", lambda r: f"Rs {r['final_equity']:>,}"),
    ]

    for name, fmt in metrics:
        v3_val = fmt(results["V3"])
        v4_val = fmt(results["V4"])
        v6_val = fmt(results["V6"])
        print(f"  {name:<23} {v3_val:>18} {v4_val:>18} {v6_val:>18}")

    # Entry type breakdown
    print("\n" + "-" * 90)
    print("  ENTRY TYPE BREAKDOWN:")
    print("-" * 90)
    all_entry_types = set()
    for v in results.values():
        all_entry_types.update(v["entry_type_stats"].keys())

    for et in sorted(all_entry_types):
        row = f"  {et:<23}"
        for ver in ["V3", "V4", "V6"]:
            stats = results[ver]["entry_type_stats"].get(et, {"count": 0, "pnl": 0, "wins": 0})
            cnt = stats["count"]
            pnl = stats["pnl"]
            wr = stats["wins"] / cnt * 100 if cnt > 0 else 0
            row += f" {cnt:>3}t Rs{pnl:>+7,.0f} {wr:>4.0f}%"
        print(row)

    # Exit reason breakdown
    print("\n" + "-" * 90)
    print("  EXIT REASON BREAKDOWN:")
    print("-" * 90)
    all_exit_reasons = set()
    for v in results.values():
        all_exit_reasons.update(v["exit_reason_stats"].keys())

    for er in sorted(all_exit_reasons):
        row = f"  {er:<23}"
        for ver in ["V3", "V4", "V6"]:
            stats = results[ver]["exit_reason_stats"].get(er, {"count": 0, "pnl": 0, "wins": 0})
            cnt = stats["count"]
            pnl = stats["pnl"]
            wr = stats["wins"] / cnt * 100 if cnt > 0 else 0
            row += f" {cnt:>3}t Rs{pnl:>+7,.0f} {wr:>4.0f}%"
        print(row)

    # Winner determination
    print("\n" + "=" * 90)
    best = max(results.values(), key=lambda r: r["net_pnl"])
    best_risk = max(results.values(), key=lambda r: r["sharpe"])
    print(f"  WINNER (Absolute P&L):     {best['version']} with Rs {best['net_pnl']:>+,}")
    print(f"  WINNER (Risk-Adjusted):    {best_risk['version']} with Sharpe {best_risk['sharpe']:.2f}")
    print("=" * 90)


# ===========================================================================
# MAIN
# ===========================================================================

if __name__ == "__main__":
    print("=" * 90)
    print("  V3 vs V4 vs V6 BACKTEST COMPARISON")
    print("  Period: January 2025 (Out-of-Sample)")
    print("  Capital: Rs 200,000 per version")
    print("=" * 90)

    # Download January 2025 data (with warmup from Oct 2024 for SMA50)
    nifty = download_real_data(start="2025-01-01", end="2025-02-01")

    if len(nifty) == 0:
        print("ERROR: No data downloaded for January 2025!")
        sys.exit(1)

    print(f"\nJanuary 2025 data: {len(nifty)} trading days")
    print(f"NIFTY range: {nifty['Close'].min():.0f} - {nifty['Close'].max():.0f}")
    print(f"VIX range: {nifty['VIX'].min():.1f} - {nifty['VIX'].max():.1f}")

    close_prices = nifty["Close"].values.tolist()

    # Run all three versions
    results = {}
    for ver in ["V3", "V4", "V6"]:
        print(f"\n{'-' * 40}")
        print(f"  Running {ver}...")
        print(f"{'-' * 40}")
        results[ver] = run_version(nifty, ver, close_prices)
        print(f"  {ver} done: Rs {results[ver]['net_pnl']:>+,} | "
              f"{results[ver]['total_trades']} trades | "
              f"WR {results[ver]['win_rate']:.1f}%")

    # Print comparison
    print_comparison(results)

    # Save results
    save_data = {}
    for ver, r in results.items():
        save_data[ver] = {k: v for k, v in r.items()
                          if k not in ("equity_curve", "all_trades",
                                       "entry_type_stats", "exit_reason_stats")}
        save_data[ver]["entry_type_stats"] = r["entry_type_stats"]
        save_data[ver]["exit_reason_stats"] = r["exit_reason_stats"]

    out_path = project_root / "data" / "v6_jan2025_comparison.json"
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")
