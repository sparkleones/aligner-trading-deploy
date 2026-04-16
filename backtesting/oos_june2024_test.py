"""
OUT-OF-SAMPLE TEST — V6_3T vs V7_5T vs V8 on June 2024 (unseen data).

Training data:  Oct 2025 - Apr 2026 (104 days)
Test data:      June 2024 (15 trading days) — COMPLETELY different market regime
  - NIFTY was 21,294-24,169 (vs 22,379-26,336 in training)
  - June 4, 2024 was the election crash day (-1,379 pts / -5.93%!)
  - VIX spiked to 26+ on election day
  - Very different market conditions from training period

Uses May 2024 as warmup for indicators (SMA50 needs 50 bars).
"""

import json
import sys
import datetime as dt
from collections import defaultdict
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
)
from backtesting.v7_hybrid_comparison import (
    compute_composite,
)
from backtesting.daywise_analysis import (
    add_all_indicators,
    compute_ema,
    compute_pivot_points,
    find_support_resistance,
)

CAPITAL = 200_000


# =======================================================================
# DYNAMIC POSITION SIZING — No artificial caps
# =======================================================================

def get_dynamic_lots(vix, equity, confidence=0.5, zero_hero=False,
                     recent_wr=0.5, recent_trades=0):
    """
    Dynamic lot sizing based on available equity and market conditions.

    Scales with:
    - Current equity (compounds winners — more capital = more lots)
    - VIX regime (higher VIX = higher SPAN margin = fewer lots naturally)
    - Confidence score (high conviction = scale up)
    - Win rate momentum (winning streak = scale up, losing = scale down)
    - Zero-hero gets 2-3 lots (not locked at 1)

    No artificial cap — let capital and margin requirements be the limit.
    """
    # SPAN margin per lot based on VIX
    SPAN = {12: 30000, 15: 35000, 20: 40000, 25: 50000, 30: 60000}
    span_per_lot = 40000
    for threshold in sorted(SPAN.keys()):
        if vix <= threshold:
            span_per_lot = SPAN[threshold]
            break
    else:
        span_per_lot = 60000

    # Use 70% of CURRENT equity (not starting capital)
    available = equity * 0.70
    base_lots = max(1, int(available / span_per_lot))

    # Win rate momentum scaling (only after 5+ trades)
    wr_mult = 1.0
    if recent_trades >= 5:
        if recent_wr >= 0.55:
            wr_mult = 1.3       # Winning streak — press the advantage
        elif recent_wr >= 0.45:
            wr_mult = 1.1       # Slightly above average
        elif recent_wr < 0.35:
            wr_mult = 0.7       # Cold streak — reduce size

    # Confidence scaling
    conf_mult = 1.0
    if confidence >= 0.70:
        conf_mult = 1.25        # High conviction
    elif confidence >= 0.50:
        conf_mult = 1.10        # Above average
    elif confidence < 0.30:
        conf_mult = 0.8         # Low conviction — smaller size

    lots = max(1, int(base_lots * wr_mult * conf_mult))

    # Allow up to 200 lots — scale with capital
    MAX_LOTS = 200
    lots = min(lots, MAX_LOTS)

    # Zero-hero: 2-3 lots (expiry-day plays are short duration, high theta)
    if zero_hero:
        zh_lots = max(2, min(3, lots))
        if vix > 25:
            zh_lots = 2         # More conservative in high VIX
        return zh_lots

    return lots


# =======================================================================
# MODEL CONFIGS
# =======================================================================

V6_3T_CONFIG = {
    "name": "V6_3T",
    "cooldown_min": 30,         # 2 bars * 15 = 30 min
    "min_confidence": 0.40,
    "trail_pct_put": 0.010,
    "trail_pct_call": 0.010,
    "min_hold_trail_put": 45,   # 3 bars * 15 min
    "min_hold_trail_call": 45,
    "max_hold_put": 330,        # 22 bars * 15
    "max_hold_call": 300,       # 20 bars * 15
    "sr_stop_buffer": 0.004,
    "vix_floor": 11,
    "vix_ceil": 35,
    "sr_gap_min": 50,
    "sr_bias_required": False,
    "zero_hero": True,
    "btst_enabled": True,
    "btst_vix_cap": 25,
    "expiry_close_min": 285,    # bar 19 * 15
    "max_entry_min": 300,       # bar 20 * 15
    "max_trades": 5,
    "max_concurrent": 2,
    "entry_check_interval": 15, # every 15 min (matches composite)
    "composite_windows_put": [(30, 105), (120, 210)],   # bars 2-7, 8-14 -> min
    "composite_windows_call": [(45, 180)],               # bars 3-12 -> min
    "use_supertrend_exit": False,
    "avoid_windows": [],
}

V7_5T_CONFIG = {
    "name": "V7_5T",
    "cooldown_min": 15,         # 1 bar * 15
    "min_confidence": 0.35,
    "trail_pct_put": 0.006,     # V7's 0.6%
    "trail_pct_call": 0.006,
    "min_hold_trail_put": 45,
    "min_hold_trail_call": 45,
    "max_hold_put": 270,        # 18 bars * 15
    "max_hold_call": 225,       # 15 bars * 15
    "sr_stop_buffer": 0.004,
    "vix_floor": 10,
    "vix_ceil": 40,
    "sr_gap_min": 0,            # No S/R gap filter
    "sr_bias_required": False,
    "zero_hero": True,
    "btst_enabled": True,
    "btst_vix_cap": 30,
    "expiry_close_min": 285,
    "max_entry_min": 330,       # bar 22 * 15
    "max_trades": 5,
    "max_concurrent": 2,
    "entry_check_interval": 15,
    "composite_windows_put": [(30, 120), (120, 240)],
    "composite_windows_call": [(30, 210)],
    "use_supertrend_exit": False,
    "avoid_windows": [],
}

V8_CONFIG = {
    "name": "V8",
    "cooldown_min": 10,
    "min_confidence": 0.20,     # V8 uses score thresholds, not confidence
    "trail_pct_put": 0.010,
    "trail_pct_call": 0.008,
    "min_hold_trail_put": 90,   # Learned: hold longer before trail
    "min_hold_trail_call": 60,
    "max_hold_put": 300,
    "max_hold_call": 270,
    "sr_stop_buffer": 0.004,
    "vix_floor": 0,             # V8 handles VIX in scoring
    "vix_ceil": 999,
    "sr_gap_min": 0,
    "sr_bias_required": False,
    "zero_hero": False,         # V8: not worth complexity (Rs +1.8K in 104 days)
    "btst_enabled": True,
    "btst_vix_cap": 25,
    "expiry_close_min": 330,
    "max_entry_min": 315,
    "max_trades": 5,
    "max_concurrent": 2,
    "entry_check_interval": 5,  # V8: every 5 min (indicator-driven)
    "composite_windows_put": [],  # Not used — V8 has its own scoring
    "composite_windows_call": [],
    "use_supertrend_exit": False,
    "avoid_windows": [(45, 75), (165, 225)],  # Learned death zones
    "use_v8_scoring": True,
}

# =======================================================================
# V9 HYBRID — Best of V6_3T + V8
# =======================================================================
# FROM V6_3T (0% overfit, best generalization):
#   - ORB breakout entries (100% WR upside)
#   - Composite entries (100% WR, high conviction)
#   - Gap entries (unique high-conviction signals)
#   - Zero-hero on expiry
#
# FROM V8 (best DD, best WR, best BTST):
#   - V8 indicator entries (catches more opportunities)
#   - Wider trail: PUT 1.0%, CALL 0.8%
#   - Longer min hold before trail: PUT 90min, CALL 60min
#   - Death zone avoidance: skip (45,75) and (165,225)
#   - ATR-based SR stop for CALLs
#   - PUT bias (lower threshold)
#   - BTST profitable logic
#
# HYBRID ADDITIONS:
#   - Dual entry engine: V8 indicators + V6_3T composite/ORB/gap
#   - max_concurrent=3 (two entry sources = more positions)
#   - max_trades=7 per day

V9_HYBRID_CONFIG = {
    "name": "V9_Hybrid",
    "cooldown_min": 10,             # V8's faster cooldown
    "min_confidence": 0.35,         # Balanced threshold
    "trail_pct_put": 0.015,         # WIDENED: 1.0% -> 1.5% (PUT trail_stop was -Rs250K)
    "trail_pct_call": 0.008,        # V8's wider CALL trail
    "min_hold_trail_put": 120,      # INCREASED: 90 -> 120 min (let PUTs run longer)
    "min_hold_trail_call": 60,      # V8: hold longer before trail fires
    "max_hold_put": 300,            # V8's learned value
    "max_hold_call": 270,           # V8's learned value
    "sr_stop_buffer": 0.004,        # Not used (ATR-based instead)
    "vix_floor": 11,                # V6_3T's floor (avoid dead markets)
    "vix_ceil": 35,                 # V6_3T's ceiling
    "sr_gap_min": 0,                # No S/R gap filter (V8)
    "sr_bias_required": False,
    "zero_hero": True,              # V6_3T's zero-hero entries
    "btst_enabled": True,
    "btst_vix_cap": 25,
    "expiry_close_min": 300,        # Between V6_3T (285) and V8 (330)
    "max_entry_min": 315,           # V8's value
    "max_trades": 7,                # More trades (two entry sources)
    "max_concurrent": 3,            # Allow 3 concurrent positions
    "entry_check_interval": 5,      # V8's 5-min for indicators
    "composite_windows_put": [(30, 105), (120, 210)],   # V6_3T's windows
    "composite_windows_call": [(45, 180)],               # V6_3T's windows
    "use_supertrend_exit": False,
    "avoid_windows": [(45, 75), (165, 285)],  # EXTENDED: block 11AM-2PM death zone (was 225)
    "use_v8_scoring": True,         # Enable V8 indicators
    "use_hybrid": True,             # Enable dual-entry engine
    "block_sr_bounce_call": True,   # sr_bounce_support CALL = 18.2% WR, -Rs97K
    "avoid_days": ["Monday", "Wednesday"],  # V11: Mon=-Rs155K, Wed=-Rs2.43L (only losing days)
    "entry_type_lot_mult": {        # Scale lots by entry quality
        "v8_indicator": 1.0,
        "sr_bounce_resistance": 1.5,  # V11: BOOSTED from 0.5 → 1.5 (61.5% WR, best entry type!)
        "orb_breakout_down": 1.5,
        "orb_breakout_up": 0.5,
        "composite": 0.8,
    },
    # === V11 DATA-DRIVEN OPTIMIZATIONS (from 233 REAL trades analysis) ===
    "disable_sr_stop_call": True,       # V11: sr_stop = 0% WR, -Rs5.82L → DISABLE entirely
    "put_bias_lot_mult": 1.3,           # V11: PUTs 38.5% WR vs CALLs 25.8% → scale up PUTs
    "call_bias_lot_mult": 0.7,          # V11: Scale down CALL lot sizes
    "vix_sweet_min": 14.0,              # V11: VIX 14-16 = 83% of profits, 36.2% WR
    "vix_sweet_max": 16.0,
    "vix_sweet_lot_mult": 1.4,          # V11: Boost lots in VIX sweet spot
    "vix_danger_min": 16.0,             # V11: VIX 16-18 = net loser
    "vix_danger_max": 18.0,
    "vix_danger_lot_mult": 0.5,         # V11: Cut lots in VIX danger zone
    "block_call_4th_hour": True,        # V11: 225-300 min entries = all CALL losses (-Rs76K)
    "block_call_first_hour": False,     # V11: Track but don't block yet (some big CALL wins here)
    # Trail_stop for CALL: Despite 13.3% WR and -Rs5.11L direct loss, keep enabled.
    # It frees position slots faster for new (better) entries. Net effect is +Rs10L.
    # === V13: REGIME-ADAPTIVE TRADING (from 12-month real data analysis) ===
    "use_regime_detection": True,       # V13: Enable market regime detection
    "min_confidence_filter": 0.30,      # V13: Reject conf < 0.3 (18.5% WR, -Rs19K in H1)
    "block_late_entries": 305,          # V13: No entries after min 305 (allow some late PUTs in volatile)
    "expiry_day_lot_mult": 0.7,         # V13: Thu expiry = 19.7% WR → reduce position size
    "avoid_days": ["Monday", "Wednesday"],  # Keep Mon/Wed filter
    # === V14: RESEARCH-INSPIRED CONFLUENCE FILTERS (9:15 AM Open Strategy doc) ===
    # Source: "Architecting a Python-Based Algorithmic Trading System for Indian Index Options"
    "use_vwap_filter": True,            # V14: Daily anchored VWAP — institutional trend confirmation
                                        #   CALL: close must be > VWAP (bullish institutional flow)
                                        #   PUT: close must be < VWAP (bearish institutional flow)
    "use_squeeze_filter": True,         # V14: Bollinger Band Squeeze — suppress entries in low-vol
                                        #   When BB(20,2) contracts inside Keltner Channels(EMA21,1.5*ATR)
                                        #   market is consolidating → all breakout signals are false
    "use_rsi_hard_gate": True,          # V14: RSI momentum gate (hard filter, not just scoring)
    "rsi_call_min": 60,                 # V14: Research-optimal RSI > 60 for CALL
    "rsi_put_max": 40,                  # V14: Research-optimal RSI < 40 for PUT
                                        # Tested: 55/45=7.7x | 58/42=11.3x | 60/40=12.0x | 65/35=8.8x
                                        # Results: 12.0x (vs 6.4x in V13b), 198 trades, 8/12 months+
                                        # Variant RSI 58/42: 11.3x, 9/12 months+ (slightly safer)
    "use_atr_trail": False,             # V14b: DISABLED — 1.5x on 1-min ATR is too tight
                                        #   (1-min ATR ≈ 10-15pts → 15-22pt trail vs old 176-330pt)
                                        #   Causes 297 trades (vs 236) due to premature exits
    "atr_trail_mult": 1.5,              # V14: 1.5x ATR trail (research uses 5-min data, not 1-min)
    # === V15: ML-DISCOVERED PATTERNS (from XGBoost feature analysis) ===
    "rsi_sweet_low": 20,                # V15: RSI 20-35 = 51.4% WR, +Rs47K avg (best pattern)
    "rsi_sweet_high": 35,
    "rsi_sweet_lot_mult": 1.5,          # V15: Boost lots 50% in RSI sweet spot
    "rsi_danger_low": 55,               # V15: RSI 55-65 = 27.3% WR, -Rs16.8K avg (worst pattern)
    "rsi_danger_high": 65,
    "rsi_danger_lot_mult": 0.5,         # V15: Halve lots in RSI danger zone
    # === V16: ADAPTIVE REVERSAL DETECTION & POSITION FLIP ===
    # Instead of skipping entire days (Monday/Wednesday), detect intraday reversals
    # and flip positions when market behavior changes. Market-adaptive, not calendar-based.
    #
    # KEY TUNING (from V16a failure analysis):
    #   V16a had 1171 trades, 31% WR, blew up capital — too many noisy flips.
    #   V16b fixes: max 1 flip/day, 60-min hold, 3 signals required, VWAP margin,
    #   confirmation period, and only for substantial reversals (> 150pt move from open).
    "use_reversal_detection": False,    # V16c: DISABLED — early exits hurt (423.2x vs 587.5x without)
                                        # Reversal detection is 87.1% WR but exits kill recovering winners
                                        # Keep code for future refinement (use as ENTRY signal, not EXIT)
    "reversal_min_hold": 60,            # V16b: 60 min hold before reversal check (was 30)
    "reversal_min_signals": 3,          # V16b: Require 3+ signals (was 2 — too noisy)
    "reversal_rsi_oversold": 25,        # V16: RSI below this = deeply oversold (bounce likely)
    "reversal_rsi_overbought": 75,      # V16: RSI above this = deeply overbought (drop likely)
    "reversal_rsi_recovery": 40,        # V16b: RSI must cross 40 (was 35 — too sensitive)
    "reversal_rsi_breakdown": 60,       # V16b: RSI must cross below 60 (was 65 — too sensitive)
    "reversal_vwap_confirm": True,      # V16: VWAP reclaim/break confirms reversal
    "reversal_ema_cross": True,         # V16: EMA9/EMA21 crossover confirms reversal
    "reversal_higher_low": True,        # V16: Higher low (bullish) / lower high (bearish)
    "reversal_momentum_bars": 15,       # V16b: 15-bar momentum (was 10 — too noisy)
    "reversal_flip_enabled": False,     # V16c: DISABLED — flip trades are 87% WR but overall drag
                                        # Better to let normal entry logic find opportunities
    "reversal_flip_conf_boost": 0.15,   # V16: Confidence boost for flip entries
    "reversal_flip_cooldown": 3,        # V16: Min bars between reversal exit and flip entry
    "reversal_max_loss_pct": 0.015,     # V16b: Only flip if loss < 1.5% of spot (was 2%)
    "reversal_max_flips_per_day": 1,    # V16b: Max 1 flip per day (prevents chop whipsaw)
    "reversal_confirm_bars": 10,        # V16b: Wait 10 bars after signal for confirmation
    "reversal_min_move_from_open": 150, # V16b: Only detect reversal if >150pt move from open
    "reversal_vwap_margin": 0.003,      # V16b: Must cross 0.3% past VWAP, not just touch
    "avoid_days": [],                   # V16c: NO DAY FILTER — VWAP+RSI+Squeeze filters handle day quality
                                        # Tested: 587.5x (all days) > 571.7x (Mon/Wed blocked), 11/11 months+
                                        # The confluence filters are smart enough to skip bad signals on any day
    # === V17: REAL-TIME ADAPTIVE ENGINE ===
    # Reads market EVERY MINUTE via state machine. No fixed windows, no time blocks.
    # Entries based on market STATE TRANSITIONS, not periodic indicator checks.
    "use_realtime_engine": False,        # V17: Enable real-time state machine (replaces V8+composite)
    "rt_check_every_minute": True,       # V17: Check signals every minute (not every 5/15 min)
    "rt_remove_avoid_windows": True,     # V17: Remove all time-based avoid windows
    "rt_state_exit": True,              # V17: Exit based on market state change (not just trail/time)
}


# =======================================================================
# V6_3T / V7_5T ENTRY DETECTION
# =======================================================================

def detect_entries_composite(cfg, bar_15min_idx, path_15min, support, resistance,
                             vix, gap_pct, composite_action, composite_conf,
                             is_expiry, prev_high, prev_low, above_sma50,
                             above_sma20, bias_val, minute_idx):
    """Entry detection for V6_3T and V7_5T (composite-based)."""
    signals = []
    spot = path_15min[bar_15min_idx] if bar_15min_idx < len(path_15min) else path_15min[-1]
    MIN_CONF = cfg["min_confidence"]

    if is_expiry and minute_idx > cfg["expiry_close_min"] - 30:
        return signals

    # GAP ENTRY (bar 0)
    if bar_15min_idx == 0 and abs(gap_pct) >= 0.3:
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
            if cfg["zero_hero"] and -1.2 <= gap_pct < -0.5 and vix >= 13:
                signals.append(("BUY_PUT", "gap_zero_hero", 0.70, True))
        elif gap_pct > 0.3:
            if is_large_gap:
                conf = min(0.85, 0.65 + gap_pct * 0.05)
                if conf >= MIN_CONF:
                    signals.append(("BUY_PUT", "gap_fade", conf, False))
            if cfg["zero_hero"] and gap_pct > 1.0 and vix >= 13:
                signals.append(("BUY_PUT", "gap_zero_hero", 0.65, True))

    # ORB (bar 1-2)
    if bar_15min_idx in (1, 2) and len(path_15min) >= 2:
        orb_high = max(path_15min[0], path_15min[1])
        orb_low = min(path_15min[0], path_15min[1])
        orb_range = orb_high - orb_low
        if orb_range > spot * 0.0015:
            if spot > orb_high:
                conf = min(0.85, 0.55 + (spot - orb_high) / orb_high * 10)
                if conf >= MIN_CONF and (above_sma50 or vix < 14):
                    signals.append(("BUY_CALL", "orb_breakout_up", conf, False))
            elif spot < orb_low:
                conf = min(0.85, 0.55 + (orb_low - spot) / orb_low * 10)
                if conf >= MIN_CONF:
                    signals.append(("BUY_PUT", "orb_breakout_down", conf, False))

    # S/R BOUNCE
    sr_dist = (resistance - support) if (support and resistance) else 0
    sr_valid = sr_dist >= cfg["sr_gap_min"] if cfg["sr_gap_min"] > 0 else True
    if bar_15min_idx >= 2 and bar_15min_idx <= 18 and sr_valid:
        prev_spot = path_15min[bar_15min_idx - 1] if bar_15min_idx > 0 else spot
        # sr_bounce_support CALL: 18.2% WR, -Rs97K → BLOCKED by config
        if support and abs(spot - support) / spot < 0.003 and spot > prev_spot:
            if not cfg.get("block_sr_bounce_call", False):
                if not cfg["sr_bias_required"] or bias_val in ("bullish", "strong_bullish", "neutral"):
                    signals.append(("BUY_CALL", "sr_bounce_support", 0.65, False))
        if resistance and abs(spot - resistance) / spot < 0.003 and spot < prev_spot:
            if not cfg["sr_bias_required"] or bias_val in ("bearish", "strong_bearish", "neutral"):
                signals.append(("BUY_PUT", "sr_bounce_resistance", 0.70, False))

    # COMPOSITE
    if composite_conf >= MIN_CONF:
        in_put = any(lo <= minute_idx <= hi for lo, hi in cfg["composite_windows_put"])
        in_call = any(lo <= minute_idx <= hi for lo, hi in cfg["composite_windows_call"])
        if composite_action == "BUY_PUT" and in_put:
            signals.append(("BUY_PUT", "composite", composite_conf, False))
        elif composite_action == "BUY_CALL" and in_call and vix < 12:
            if composite_conf >= 0.75:
                signals.append(("BUY_CALL", "composite", composite_conf, False))

    # ZERO-HERO on expiry
    if cfg["zero_hero"] and is_expiry and bar_15min_idx in (1, 2, 3):
        if vix >= 13 and not above_sma50:
            signals.append(("BUY_PUT", "zero_hero_expiry", 0.60, True))

    return signals


# =======================================================================
# MARKET REGIME DETECTION — V11 Adaptive
# =======================================================================

def detect_market_regime(daily_df, row_idx):
    """Detect market regime from daily data.

    Uses multiple timeframes to classify:
      - BULLISH: Strong uptrend (above SMAs, positive momentum)
      - BEARISH: Strong downtrend (below SMAs, negative momentum)
      - VOLATILE: High VIX / sharp moves (regime where our model excels)
      - SIDEWAYS: Range-bound, low directional conviction

    Returns: dict with regime info
    """
    if row_idx < 5:
        return {"regime": "neutral", "strength": 0.0, "call_mult": 1.0, "put_mult": 1.0}

    row = daily_df.iloc[row_idx]
    close = float(row["Close"])

    # SMA trend (multi-timeframe)
    above_sma20 = close > float(row.get("SMA20", close)) if pd.notna(row.get("SMA20")) else True
    above_sma50 = close > float(row.get("SMA50", close)) if pd.notna(row.get("SMA50")) else True
    ema9 = float(row.get("EMA9", close)) if pd.notna(row.get("EMA9")) else close
    ema21 = float(row.get("EMA21", close)) if pd.notna(row.get("EMA21")) else close
    ema9_above_21 = ema9 > ema21
    rsi = float(row.get("RSI", 50)) if pd.notna(row.get("RSI")) else 50
    vix = float(row.get("VIX", 14)) if pd.notna(row.get("VIX")) else 14

    # 5-day momentum: count of up-days in last 5
    lookback = min(5, row_idx)
    up_days = 0
    for j in range(1, lookback + 1):
        prev = daily_df.iloc[row_idx - j]
        if float(prev["Close"]) < close:
            up_days += 1

    # 10-day returns
    lookback10 = min(10, row_idx)
    close_10d_ago = float(daily_df.iloc[row_idx - lookback10]["Close"])
    ret_10d = (close - close_10d_ago) / close_10d_ago * 100

    # Scoring
    bull_score = 0
    bear_score = 0

    if above_sma20: bull_score += 1
    else: bear_score += 1
    if above_sma50: bull_score += 1
    else: bear_score += 1
    if ema9_above_21: bull_score += 1
    else: bear_score += 1
    if rsi > 55: bull_score += 1
    elif rsi < 45: bear_score += 1
    if up_days >= 4: bull_score += 1
    elif up_days <= 1: bear_score += 1
    if ret_10d > 1.5: bull_score += 1
    elif ret_10d < -1.5: bear_score += 1

    # Determine regime
    if vix >= 18:
        regime = "volatile"
        strength = min(1.0, (vix - 18) / 10)
        # In volatile regime: PUTs excel (Oct-Dec 2024 pattern)
        call_mult = 0.5
        put_mult = 1.5
    elif bull_score >= 4 and bear_score <= 1:
        regime = "bullish"
        strength = min(1.0, bull_score / 6)
        # In bullish regime: CALLs should be favored, PUTs need higher conviction
        call_mult = 1.4
        put_mult = 0.5  # Only high-conviction PUTs
    elif bear_score >= 4 and bull_score <= 1:
        regime = "bearish"
        strength = min(1.0, bear_score / 6)
        # In bearish regime: PUTs excel (model's strength)
        call_mult = 0.5
        put_mult = 1.5
    else:
        regime = "sideways"
        strength = 0.3
        # In sideways: reduce all sizes, market has no edge
        call_mult = 0.6
        put_mult = 0.6

    return {
        "regime": regime,
        "strength": strength,
        "call_mult": call_mult,
        "put_mult": put_mult,
        "bull_score": bull_score,
        "bear_score": bear_score,
        "ret_10d": ret_10d,
        "vix": vix,
    }


# =======================================================================
# V16: INTRADAY REVERSAL DETECTION — Market-Adaptive Position Flip
# =======================================================================
# Philosophy: Don't skip days blindly. Instead, READ the market in real-time.
# Detect when momentum is exhausting and the trend is about to reverse.
# On April 6, 2026: V3 entered PUT at 10:00 during crash. Market V-recovered.
# A reversal detector would have: closed PUT at ~11:30, flipped to CALL at ~11:45.
#
# Reversal Signals (need 2+ to confirm):
# 1. RSI Recovery:   RSI was < 25 (oversold), now crossing above 35 -> bullish
# 2. VWAP Reclaim:   Price crosses from below VWAP to above -> institutional buying
# 3. EMA Crossover:  EMA9 crosses above EMA21 (intraday golden cross)
# 4. Higher Low:     Current swing low > previous swing low -> structure change
# 5. Momentum Shift: N-bar rate of change flips from negative to positive

def detect_intraday_reversal(cfg, day_bars_df, minute_idx, open_trade, vwap_arr):
    """Detect if the market is reversing against the current position.

    V16b TUNING: Much more conservative than V16a to avoid chop whipsaw.
    - Requires 3+ confirmation signals (not 2)
    - Only triggers after 60+ minutes (not 30)
    - Market must have moved 150+ pts from open (big move, real reversal)
    - Confirmation period: signals must persist for 10 bars
    - Max 1 flip per day (tracked by caller)

    Returns:
        dict with keys:
            'detected': bool - reversal confirmed (3+ signals)
            'direction': str - 'bullish' or 'bearish' (the new direction)
            'signals': list[str] - which reversal signals fired
            'confidence': float - reversal confidence (0.5-0.9)
            'flip_action': str - 'BUY_CALL' or 'BUY_PUT' (what to enter next)
    """
    result = {
        "detected": False, "direction": "", "signals": [],
        "confidence": 0.0, "flip_action": None,
    }

    if not cfg.get("use_reversal_detection", False):
        return result

    action = open_trade["action"]
    entry_spot = open_trade["entry_spot"]
    entry_minute = open_trade["entry_minute"]
    minutes_held = minute_idx - entry_minute

    # Don't check too early — need enough bars for signals to form
    min_hold = cfg.get("reversal_min_hold", 60)
    if minutes_held < min_hold:
        return result

    # Need RSI column
    if "rsi" not in day_bars_df.columns:
        return result

    closes = day_bars_df["close"].values
    rsi_arr = day_bars_df["rsi"].values
    bar_spot = closes[minute_idx]
    bar_rsi = float(rsi_arr[minute_idx])

    # V16b: Check that market has made a substantial move from open
    # Small intraday wiggles are NOT reversals — just noise
    day_open = closes[0]
    move_from_open = abs(bar_spot - day_open)
    min_move = cfg.get("reversal_min_move_from_open", 150)
    # The move that matters is the EXTREME move, not current spot
    extreme_move = 0
    if action == "BUY_PUT":
        # For PUT, the extreme favorable move was the lowest point
        best_fav = open_trade.get("best_fav", entry_spot)
        extreme_move = abs(day_open - best_fav)
    else:
        best_fav = open_trade.get("best_fav", entry_spot)
        extreme_move = abs(best_fav - day_open)
    if extreme_move < min_move:
        return result

    # Current position is losing — check if market is reversing
    if action == "BUY_PUT":
        is_losing = bar_spot > entry_spot
        reversal_dir = "bullish"
        flip_action = "BUY_CALL"
    else:
        is_losing = bar_spot < entry_spot
        reversal_dir = "bearish"
        flip_action = "BUY_PUT"

    # --- Max loss guard: don't flip if already too deep in loss ---
    max_loss_pct = cfg.get("reversal_max_loss_pct", 0.015)
    spot_move_pct = abs(bar_spot - entry_spot) / entry_spot
    if spot_move_pct > max_loss_pct and is_losing:
        return result

    signals = []
    lookback = min(minute_idx, 30)

    # ─── SIGNAL 1: RSI RECOVERY / BREAKDOWN ───
    # Bullish reversal: RSI was deeply oversold, now recovering above threshold
    # Bearish reversal: RSI was deeply overbought, now breaking below threshold
    rsi_oversold = cfg.get("reversal_rsi_oversold", 25)
    rsi_overbought = cfg.get("reversal_rsi_overbought", 75)
    rsi_recovery = cfg.get("reversal_rsi_recovery", 35)
    rsi_breakdown = cfg.get("reversal_rsi_breakdown", 65)

    if lookback >= 10:
        recent_rsi = rsi_arr[max(0, minute_idx - lookback):minute_idx + 1]
        min_recent_rsi = float(np.nanmin(recent_rsi))
        max_recent_rsi = float(np.nanmax(recent_rsi))

        if reversal_dir == "bullish":
            # Was oversold in recent history, now RSI recovering
            if min_recent_rsi < rsi_oversold and bar_rsi > rsi_recovery:
                signals.append("rsi_recovery")
        else:
            # Was overbought in recent history, now RSI breaking down
            if max_recent_rsi > rsi_overbought and bar_rsi < rsi_breakdown:
                signals.append("rsi_breakdown")

    # ─── SIGNAL 2: VWAP RECLAIM / BREAK (with margin) ───
    # Bullish: price crosses from below VWAP to above by margin (not just touching)
    # Bearish: price crosses from above VWAP to below by margin
    if cfg.get("reversal_vwap_confirm", True) and vwap_arr is not None:
        vwap_margin = cfg.get("reversal_vwap_margin", 0.003)
        if minute_idx >= 10 and minute_idx < len(vwap_arr):
            cur_vwap = vwap_arr[minute_idx]
            prev_vwap_spot = closes[minute_idx - 10]
            prev_vwap = vwap_arr[minute_idx - 10]
            if not (np.isnan(cur_vwap) or np.isnan(prev_vwap)):
                margin_pts = cur_vwap * vwap_margin
                if reversal_dir == "bullish":
                    # Was below VWAP, now CLEARLY above (by 0.3%)
                    if prev_vwap_spot < prev_vwap and bar_spot > cur_vwap + margin_pts:
                        signals.append("vwap_reclaim")
                else:
                    # Was above VWAP, now CLEARLY below
                    if prev_vwap_spot > prev_vwap and bar_spot < cur_vwap - margin_pts:
                        signals.append("vwap_break")

    # ─── SIGNAL 3: EMA CROSSOVER ───
    # Bullish: EMA9 crosses above EMA21 (golden cross on 1-min)
    # Bearish: EMA9 crosses below EMA21 (death cross on 1-min)
    if cfg.get("reversal_ema_cross", True) and minute_idx >= 25:
        if "ema9" in day_bars_df.columns and "ema21" in day_bars_df.columns:
            ema9_now = float(day_bars_df["ema9"].iloc[minute_idx])
            ema21_now = float(day_bars_df["ema21"].iloc[minute_idx])
            ema9_prev = float(day_bars_df["ema9"].iloc[minute_idx - 5])
            ema21_prev = float(day_bars_df["ema21"].iloc[minute_idx - 5])
            if reversal_dir == "bullish":
                if ema9_prev < ema21_prev and ema9_now > ema21_now:
                    signals.append("ema_golden_cross")
            else:
                if ema9_prev > ema21_prev and ema9_now < ema21_now:
                    signals.append("ema_death_cross")
        else:
            # Compute from closes if columns not available
            if minute_idx >= 25:
                window = closes[max(0, minute_idx - 25):minute_idx + 1]
                if len(window) >= 21:
                    ema9_now = float(pd.Series(window).ewm(span=9).mean().iloc[-1])
                    ema21_now = float(pd.Series(window).ewm(span=21).mean().iloc[-1])
                    ema9_prev = float(pd.Series(window[:-5]).ewm(span=9).mean().iloc[-1])
                    ema21_prev = float(pd.Series(window[:-5]).ewm(span=21).mean().iloc[-1])
                    if reversal_dir == "bullish":
                        if ema9_prev < ema21_prev and ema9_now > ema21_now:
                            signals.append("ema_golden_cross")
                    else:
                        if ema9_prev > ema21_prev and ema9_now < ema21_now:
                            signals.append("ema_death_cross")

    # ─── SIGNAL 4: HIGHER LOW / LOWER HIGH (Structure Change) ───
    # Bullish: current low > previous swing low (buyers defending higher levels)
    # Bearish: current high < previous swing high (sellers defending lower levels)
    if cfg.get("reversal_higher_low", True) and minute_idx >= 20:
        window = closes[max(0, minute_idx - 30):minute_idx + 1]
        if len(window) >= 15:
            # Find swing lows (local minimums)
            swing_lows = []
            swing_highs = []
            for j in range(2, len(window) - 2):
                if window[j] < window[j-1] and window[j] < window[j-2] and \
                   window[j] < window[j+1] and window[j] < window[j+2]:
                    swing_lows.append(window[j])
                if window[j] > window[j-1] and window[j] > window[j-2] and \
                   window[j] > window[j+1] and window[j] > window[j+2]:
                    swing_highs.append(window[j])

            if reversal_dir == "bullish" and len(swing_lows) >= 2:
                if swing_lows[-1] > swing_lows[-2]:
                    signals.append("higher_low")
            elif reversal_dir == "bearish" and len(swing_highs) >= 2:
                if swing_highs[-1] < swing_highs[-2]:
                    signals.append("lower_high")

    # ─── SIGNAL 5: MOMENTUM SHIFT (with magnitude threshold) ───
    # N-bar rate of change flips direction AND has meaningful magnitude
    mom_bars = cfg.get("reversal_momentum_bars", 15)
    if minute_idx >= mom_bars + 10:
        cur_mom = bar_spot - closes[minute_idx - mom_bars]
        prev_mom = closes[minute_idx - 10] - closes[minute_idx - mom_bars - 10]
        # V16b: Require momentum to be > 0.3% of spot (not just any sign flip)
        mom_threshold = bar_spot * 0.003
        if reversal_dir == "bullish":
            if prev_mom < -mom_threshold and cur_mom > mom_threshold:
                signals.append("momentum_flip_up")
        else:
            if prev_mom > mom_threshold and cur_mom < -mom_threshold:
                signals.append("momentum_flip_down")

    # ─── CONFIRM REVERSAL ───
    min_signals = cfg.get("reversal_min_signals", 2)
    if len(signals) >= min_signals:
        # Confidence based on number and quality of signals
        base_conf = 0.50 + len(signals) * 0.10
        # RSI recovery/breakdown is the strongest signal
        if "rsi_recovery" in signals or "rsi_breakdown" in signals:
            base_conf += 0.05
        # VWAP reclaim/break is institutional confirmation
        if "vwap_reclaim" in signals or "vwap_break" in signals:
            base_conf += 0.05
        conf = min(0.90, base_conf)

        result["detected"] = True
        result["direction"] = reversal_dir
        result["signals"] = signals
        result["confidence"] = conf
        result["flip_action"] = flip_action

    return result


# =======================================================================
# V17: REAL-TIME ADAPTIVE ENGINE — Market State Machine
# =======================================================================
# Reads the market EVERY MINUTE. No fixed windows, no avoided time blocks.
# Decides based on what price is DOING right now, not what time it is.
#
# Market States:
#   TRENDING_UP    — price above VWAP, fast EMA > slow EMA, momentum positive
#   TRENDING_DOWN  — price below VWAP, fast EMA < slow EMA, momentum negative
#   EXHAUSTION_UP  — was trending up, RSI overbought, momentum fading
#   EXHAUSTION_DOWN— was trending down, RSI oversold, momentum fading
#   REVERSAL_UP    — was exhausted/trending down, now pivoting up (BUY CALL)
#   REVERSAL_DOWN  — was exhausted/trending up, now pivoting down (BUY PUT)
#   RANGING        — no clear direction, price oscillating around VWAP
#
# Entry triggers: STATE TRANSITIONS, not fixed rules
#   RANGING/EXHAUSTION_DOWN -> TRENDING_UP or REVERSAL_UP: BUY_CALL
#   RANGING/EXHAUSTION_UP -> TRENDING_DOWN or REVERSAL_DOWN: BUY_PUT
#
# Exit triggers: STATE changes against position
#   Holding PUT + state becomes REVERSAL_UP/TRENDING_UP: EXIT
#   Holding CALL + state becomes REVERSAL_DOWN/TRENDING_DOWN: EXIT

def compute_realtime_state(closes, highs, lows, vwap_arr, minute_idx, prev_state="RANGING"):
    """Compute current market state from intraday price action.

    Uses FAST indicators for real-time responsiveness:
      - EMA5 vs EMA13 (fast/slow crossover — 5x faster than EMA9/21)
      - RSI7 (7-period RSI — 2x faster than RSI14)
      - 5-bar momentum (not 14-bar)
      - VWAP position (above/below)
      - Rate of change acceleration

    Returns: dict with state, scores, and details
    """
    if minute_idx < 15:
        return {"state": "RANGING", "strength": 0.0, "rsi": 50.0,
                "momentum": 0.0, "vwap_pos": "at", "ema_trend": "flat",
                "acceleration": 0.0}

    window = closes[max(0, minute_idx - 50):minute_idx + 1]
    n = len(window)

    # ── FAST EMA (5 vs 13) — responds in 3-5 bars instead of 10-20 ──
    if n >= 13:
        ema5 = float(pd.Series(window).ewm(span=5, min_periods=3).mean().iloc[-1])
        ema13 = float(pd.Series(window).ewm(span=13, min_periods=7).mean().iloc[-1])
    else:
        ema5 = float(np.mean(window[-5:])) if n >= 5 else window[-1]
        ema13 = float(np.mean(window)) if n >= 3 else window[-1]

    ema_diff_pct = (ema5 - ema13) / ema13 * 100 if ema13 > 0 else 0

    if ema_diff_pct > 0.05:
        ema_trend = "bullish"
    elif ema_diff_pct < -0.05:
        ema_trend = "bearish"
    else:
        ema_trend = "flat"

    # ── FAST RSI (7-period) ──
    if n >= 8:
        delta = np.diff(window[-8:])
        gains = np.where(delta > 0, delta, 0)
        losses_arr = np.where(delta < 0, -delta, 0)
        avg_gain = np.mean(gains) + 1e-10
        avg_loss = np.mean(losses_arr) + 1e-10
        rs = avg_gain / avg_loss
        rsi7 = 100 - (100 / (1 + rs))
    else:
        rsi7 = 50.0

    # ── MOMENTUM (5-bar rate of change) ──
    if n >= 6:
        momentum = (window[-1] - window[-6]) / window[-6] * 100
    else:
        momentum = 0.0

    # ── ACCELERATION (is momentum increasing or decreasing?) ──
    if n >= 11:
        mom_now = window[-1] - window[-6]
        mom_prev = window[-6] - window[-11]
        acceleration = mom_now - mom_prev  # positive = accelerating up
    else:
        acceleration = 0.0

    # ── VWAP POSITION ──
    vwap_pos = "at"
    if vwap_arr is not None and minute_idx < len(vwap_arr):
        cur_vwap = vwap_arr[minute_idx]
        if not np.isnan(cur_vwap) and cur_vwap > 0:
            vwap_dist_pct = (window[-1] - cur_vwap) / cur_vwap * 100
            if vwap_dist_pct > 0.15:
                vwap_pos = "above"
            elif vwap_dist_pct < -0.15:
                vwap_pos = "below"

    # ── SWING HIGH/LOW ANALYSIS (dynamic S/R) ──
    making_higher_lows = False
    making_lower_highs = False
    if n >= 20:
        # Find last 2 swing lows
        swing_lows_idx = []
        for j in range(3, n - 3):
            if (window[j] < window[j-1] and window[j] < window[j-2] and
                window[j] < window[j+1] and window[j] < window[j+2]):
                swing_lows_idx.append(j)
        if len(swing_lows_idx) >= 2:
            making_higher_lows = window[swing_lows_idx[-1]] > window[swing_lows_idx[-2]]

        # Find last 2 swing highs
        swing_highs_idx = []
        for j in range(3, n - 3):
            if (window[j] > window[j-1] and window[j] > window[j-2] and
                window[j] > window[j+1] and window[j] > window[j+2]):
                swing_highs_idx.append(j)
        if len(swing_highs_idx) >= 2:
            making_lower_highs = window[swing_highs_idx[-1]] < window[swing_highs_idx[-2]]

    # ══════════════════════════════════════════════════════════════
    #  STATE MACHINE — determine current market state
    # ══════════════════════════════════════════════════════════════
    state = "RANGING"
    strength = 0.0

    # TRENDING_UP: bullish EMA + above VWAP + positive momentum + RSI 45-75
    if ema_trend == "bullish" and vwap_pos == "above" and momentum > 0.1 and 45 < rsi7 < 75:
        state = "TRENDING_UP"
        strength = min(1.0, abs(momentum) * 2 + ema_diff_pct * 5)

    # TRENDING_DOWN: bearish EMA + below VWAP + negative momentum + RSI 25-55
    elif ema_trend == "bearish" and vwap_pos == "below" and momentum < -0.1 and 25 < rsi7 < 55:
        state = "TRENDING_DOWN"
        strength = min(1.0, abs(momentum) * 2 + abs(ema_diff_pct) * 5)

    # EXHAUSTION_UP: bullish but RSI > 75 or momentum fading while above VWAP
    elif rsi7 > 75 and vwap_pos == "above":
        state = "EXHAUSTION_UP"
        strength = min(1.0, (rsi7 - 75) / 25)

    # EXHAUSTION_DOWN: bearish but RSI < 25 or momentum fading below VWAP
    elif rsi7 < 25 and vwap_pos == "below":
        state = "EXHAUSTION_DOWN"
        strength = min(1.0, (25 - rsi7) / 25)

    # REVERSAL_UP: was trending down/exhausted, now showing bullish signs
    elif (prev_state in ("TRENDING_DOWN", "EXHAUSTION_DOWN") and
          making_higher_lows and momentum > 0 and rsi7 > 35 and acceleration > 0):
        state = "REVERSAL_UP"
        strength = min(1.0, 0.5 + momentum * 3)

    # REVERSAL_DOWN: was trending up/exhausted, now showing bearish signs
    elif (prev_state in ("TRENDING_UP", "EXHAUSTION_UP") and
          making_lower_highs and momentum < 0 and rsi7 < 65 and acceleration < 0):
        state = "REVERSAL_DOWN"
        strength = min(1.0, 0.5 + abs(momentum) * 3)

    # Everything else: RANGING
    else:
        state = "RANGING"
        strength = 0.3

    return {
        "state": state, "strength": max(0, min(1.0, strength)),
        "rsi": rsi7, "momentum": momentum, "vwap_pos": vwap_pos,
        "ema_trend": ema_trend, "acceleration": acceleration,
        "ema5": ema5 if n >= 5 else 0, "ema13": ema13 if n >= 13 else 0,
        "higher_lows": making_higher_lows, "lower_highs": making_lower_highs,
    }


def detect_entries_realtime(cfg, market_state, prev_market_state, bar_spot, vix,
                            minute_idx, open_trades):
    """Real-time entry detection based on market state transitions.

    V17b: More selective — only enter on CONFIRMED state transitions with
    high strength. Reduces over-trading (V17a had 1445 vs V14's 555 trades).

    Key changes from V17a:
      - Raised minimum strength thresholds (0.4 -> 0.55 for breakouts)
      - Removed trend continuation (same-state re-entries caused most over-trading)
      - Removed mean reversion (noisy, low WR)
      - Added cooldown: only 1 RT entry per 30 minutes
      - Require state TRANSITION (not just current state)

    Returns: list of (action, entry_type, confidence, is_zero_hero)
    """
    entries = []
    state = market_state["state"]
    prev_state = prev_market_state["state"] if prev_market_state else "RANGING"
    strength = market_state["strength"]
    rsi = market_state["rsi"]
    momentum = market_state["momentum"]

    # V17b: Allow disabling RT entries (use V14 legacy entries + RT exits only)
    if cfg.get("rt_entries_disabled", False):
        return entries

    # Don't enter in first 20 min (need indicator warmup)
    if minute_idx < 20:
        return entries

    # Don't enter in last 45 min (forced closure risk — need time for trade to work)
    if minute_idx > 330:
        return entries

    # REQUIRE a state TRANSITION — same state as previous bar means no new signal
    if state == prev_state:
        return entries

    # ── BREAKOUT ENTRIES: RANGING -> TRENDING (strongest signal) ──
    if prev_state == "RANGING":
        if state == "TRENDING_UP" and strength > 0.55:
            conf = min(0.85, 0.55 + strength * 0.3)
            entries.append(("BUY_CALL", "rt_breakout_up", conf, False))

        elif state == "TRENDING_DOWN" and strength > 0.55:
            conf = min(0.85, 0.55 + strength * 0.3)
            entries.append(("BUY_PUT", "rt_breakout_down", conf, False))

    # ── REVERSAL ENTRIES: Exhaustion/Trending -> Reversal (high conviction) ──
    if state == "REVERSAL_UP" and prev_state in ("EXHAUSTION_DOWN", "TRENDING_DOWN"):
        if strength > 0.5:
            conf = min(0.85, 0.60 + strength * 0.25)
            entries.append(("BUY_CALL", "rt_reversal_up", conf, False))

    if state == "REVERSAL_DOWN" and prev_state in ("EXHAUSTION_UP", "TRENDING_UP"):
        if strength > 0.5:
            conf = min(0.85, 0.60 + strength * 0.25)
            entries.append(("BUY_PUT", "rt_reversal_down", conf, False))

    # ── VIX FILTER: high VIX favors puts, low VIX favors calls ──
    filtered = []
    for action, etype, conf, is_zh in entries:
        # High VIX: reduce CALL confidence (theta crush)
        if vix > 20 and action == "BUY_CALL":
            conf *= 0.85
        # Low VIX: reduce PUT confidence (no fuel for puts)
        if vix < 13 and action == "BUY_PUT":
            conf *= 0.80
        if conf >= cfg.get("min_confidence", 0.35):
            filtered.append((action, etype, conf, is_zh))

    return filtered


def check_realtime_exits(cfg, market_state, open_trade, bar_spot, minute_idx):
    """Real-time exit based on market state change against position.

    V17b CONSERVATIVE approach: State exits are ONLY for loss protection.
    Winners are managed by legacy exits (trail_stop, time_exit) which let them run.

    Key lesson from V17a test:
      - rt_exhaustion_profit fired 892 times, killed all compounding
      - V14's time_exit (270-300 min hold) is where 90% of profit comes from
      - State exits should ONLY cut losing positions faster, not take profit early

    Returns: exit_reason string or None
    """
    action = open_trade["action"]
    entry_bar = open_trade["entry_minute"]
    bars_held = minute_idx - entry_bar
    state = market_state["state"]
    strength = market_state["strength"]
    entry_spot = open_trade["entry_spot"]

    if bars_held < 10:
        return None

    # ── Calculate current P&L direction ──
    if action == "BUY_PUT":
        move_pct = (entry_spot - bar_spot) / entry_spot * 100  # positive = profit
    else:
        move_pct = (bar_spot - entry_spot) / entry_spot * 100  # positive = profit

    is_losing = move_pct < -0.3  # losing > 0.3%

    # ══════════════════════════════════════════════════════════════
    #  ONLY EXIT LOSERS EARLY — let winners run to legacy exits
    # ══════════════════════════════════════════════════════════════

    # ── SMART STOP-LOSS: Exit losing position when market STRONGLY against us ──
    # This replaces the "wait for trail_stop" approach for LOSING trades only.
    # Requirement: losing > 0.5% AND strong opposing state AND held > 30 min
    if is_losing and bars_held >= 30:
        if action == "BUY_PUT":
            # We're short. Market is trending up STRONGLY against us.
            if state == "TRENDING_UP" and strength > 0.5 and move_pct < -0.5:
                return "rt_smart_stop"
            # Confirmed reversal up — our PUT is toast
            if state == "REVERSAL_UP" and strength > 0.6 and move_pct < -0.8:
                return "rt_reversal_stop"

        elif action == "BUY_CALL":
            # We're long. Market is trending down STRONGLY against us.
            if state == "TRENDING_DOWN" and strength > 0.5 and move_pct < -0.5:
                return "rt_smart_stop"
            # Confirmed reversal down — our CALL is toast
            if state == "REVERSAL_DOWN" and strength > 0.6 and move_pct < -0.8:
                return "rt_reversal_stop"

    # ── CATASTROPHIC LOSS PROTECTION: Hard stop at -1.5% with ANY opposing state ──
    if move_pct < -1.5 and bars_held >= 15:
        if action == "BUY_PUT" and state in ("TRENDING_UP", "REVERSAL_UP"):
            return "rt_hard_stop"
        if action == "BUY_CALL" and state in ("TRENDING_DOWN", "REVERSAL_DOWN"):
            return "rt_hard_stop"

    # ── PROFIT PROTECTION (only for BIG winners, after long hold) ──
    # Only trigger if profit > 1% AND held > 180 min AND state exhausting
    # This is a TRAILING take-profit, not early exit
    if move_pct > 1.0 and bars_held >= 180:
        if action == "BUY_PUT" and state == "EXHAUSTION_DOWN" and strength > 0.6:
            return "rt_profit_protect"
        if action == "BUY_CALL" and state == "EXHAUSTION_UP" and strength > 0.6:
            return "rt_profit_protect"

    return None


# =======================================================================
# V8 ENTRY DETECTION (indicator-based)
# =======================================================================

def detect_entries_v8(bar_data, sr_levels, vix, daily_trend, minute_idx):
    """V8 indicator-based entry detection."""
    spot = bar_data["close"]
    call_score = 0.0
    put_score = 0.0
    reasons = []

    # Supertrend
    if bar_data["st_direction"] == 1:
        call_score += 2.5
    elif bar_data["st_direction"] == -1:
        put_score += 3.0
        reasons.append("st_bear")

    # EMA alignment
    if bar_data["above_ema9"] and bar_data["above_ema21"] and bar_data["ema9_above_ema21"]:
        call_score += 2.0
    elif not bar_data["above_ema9"] and not bar_data["above_ema21"] and not bar_data["ema9_above_ema21"]:
        put_score += 3.5
        reasons.append("ema_bear")

    # EMA crossover
    if bar_data.get("ema9_cross_up_21", False):
        call_score += 2.0
    if bar_data.get("ema9_cross_down_21", False):
        put_score += 2.5

    # RSI
    rsi = bar_data["rsi"]
    if 30 <= rsi < 50:
        put_score += 1.5
    elif rsi < 30:
        call_score += 2.0
    elif rsi > 70:
        put_score += 2.0
    elif 50 <= rsi < 60:
        call_score -= 0.5
        put_score -= 0.5

    # MACD
    if bar_data["macd_hist"] > 0 and bar_data.get("macd_cross_up", False):
        call_score += 1.5
    elif bar_data["macd_hist"] > 0:
        call_score += 0.3
    if bar_data["macd_hist"] < 0 and bar_data.get("macd_cross_down", False):
        put_score += 1.5
    elif bar_data["macd_hist"] < 0:
        put_score += 0.3

    # Stochastic RSI
    if bar_data["stoch_oversold"] and bar_data.get("stoch_cross_up", False):
        call_score += 1.0
    if bar_data["stoch_overbought"] and bar_data.get("stoch_cross_down", False):
        put_score += 1.0

    # Bollinger
    if bar_data["at_bb_lower"]:
        call_score += 1.5
    if bar_data["at_bb_upper"]:
        put_score += 1.5

    # ADX
    adx = bar_data["adx"]
    if 25 <= adx < 35:
        call_score *= 0.8
        put_score *= 0.8
    elif adx >= 35:
        if bar_data["plus_di"] > bar_data["minus_di"]:
            call_score += 1.0
        else:
            put_score += 1.0

    # VIX regime
    if 13 <= vix < 16:
        put_score += 1.5
    elif vix >= 16:
        put_score += 1.0
    elif vix < 11:
        call_score += 0.5

    # Daily trend
    if daily_trend == "bullish":
        call_score += 1.0
    elif daily_trend == "bearish":
        put_score += 1.0

    # S/R proximity
    for sr in sr_levels:
        dist = (spot - sr["level"]) / spot * 100
        if -0.15 <= dist <= 0.15:
            if dist > 0 and bar_data["close"] > bar_data["open"]:
                call_score += sr["strength"] * 0.4
            elif dist < 0 and bar_data["close"] < bar_data["open"]:
                put_score += sr["strength"] * 0.4

    PUT_MIN = 4.0
    CALL_MIN = 5.0

    if put_score >= PUT_MIN and put_score > call_score:
        conf = min(1.0, put_score / 18.0)
        return "BUY_PUT", conf, reasons
    elif call_score >= CALL_MIN and call_score > put_score:
        conf = min(1.0, call_score / 18.0)
        return "BUY_CALL", conf, reasons
    return None, 0, []


# =======================================================================
# UNIFIED DAY SIMULATION
# =======================================================================

def simulate_day(cfg, day_bars_df, date, prev_day_ohlc, vix, daily_trend,
                 dte, is_expiry, daily_df, row_idx, close_prices,
                 above_sma50, above_sma20, rsi, prev_change, vix_spike,
                 sma20, sma50, ema9, ema21, weekly_sma, gap_pct,
                 equity=CAPITAL, recent_wr=0.5, recent_trades=0,
                 regime_info=None):
    """Unified day simulation for all 3 models."""
    n_bars = len(day_bars_df)
    if n_bars < 60:
        return []

    use_v8 = cfg.get("use_v8_scoring", False)

    # Pivot points
    if prev_day_ohlc:
        pivots = compute_pivot_points(prev_day_ohlc["high"], prev_day_ohlc["low"], prev_day_ohlc["close"])
    else:
        spot = day_bars_df["close"].iloc[0]
        pivots = compute_pivot_points(spot * 1.005, spot * 0.995, spot)

    sr_levels = find_support_resistance(day_bars_df, prev_day_ohlc, pivots)

    # S/R from multi-method (for composite models)
    entry_spot = day_bars_df["open"].iloc[0]
    prev_high = prev_day_ohlc["high"] if prev_day_ohlc else entry_spot * 1.01
    prev_low = prev_day_ohlc["low"] if prev_day_ohlc else entry_spot * 0.99
    support, resistance = sr_multi_method(
        entry_spot, prev_high, prev_low, sma20, sma50,
        close_history=close_prices, idx=row_idx)

    # Composite scoring (for V6_3T and V7_5T)
    if not use_v8:
        scores = compute_composite(
            "V6", vix, above_sma50, above_sma20, rsi,
            date.strftime("%A"), prev_change, vix_spike,
            entry_spot, support, resistance,
            ema9=ema9, ema21=ema21, weekly_sma=weekly_sma)
        best_composite = max(scores, key=scores.get)
        total_score = sum(scores.values())
        composite_conf = scores[best_composite] / total_score if total_score > 0 else 0
    else:
        best_composite = "BUY_PUT"
        composite_conf = 0

    bias_val = "neutral"
    if above_sma50 and above_sma20:
        bias_val = "strong_bullish" if ema9 and ema21 and ema9 > ema21 else "bullish"
    elif not above_sma50 and not above_sma20:
        bias_val = "strong_bearish" if ema9 and ema21 and ema9 < ema21 else "bearish"

    # Build 15-min path
    minute_closes = day_bars_df["close"].values
    n_15min = min(25, n_bars // 15 + 1)
    path_15min = [minute_closes[min(i * 15, n_bars - 1)] for i in range(n_15min)]

    # Compute ATR for ATR-based SR stop (from first 60 bars of the day)
    day_atr = 100  # fallback
    highs = day_bars_df["high"].values
    lows = day_bars_df["low"].values
    closes_arr = day_bars_df["close"].values
    tr_vals = []
    for j in range(1, min(60, n_bars)):
        tr = max(highs[j] - lows[j],
                 abs(highs[j] - closes_arr[j-1]),
                 abs(lows[j] - closes_arr[j-1]))
        tr_vals.append(tr)
    if len(tr_vals) >= 14:
        day_atr = np.mean(tr_vals[-14:])
    elif tr_vals:
        day_atr = np.mean(tr_vals)

    # =================================================================
    # V14: Pre-compute VWAP and Squeeze arrays for the entire day
    # =================================================================
    use_vwap = cfg.get("use_vwap_filter", False)
    use_squeeze = cfg.get("use_squeeze_filter", False)
    use_rsi_gate = cfg.get("use_rsi_hard_gate", False)
    use_atr_trail = cfg.get("use_atr_trail", False)
    atr_trail_mult = cfg.get("atr_trail_mult", 1.5)

    # VWAP: Cumulative Typical Price (volume-weighted if volume column exists)
    # Anchored daily — resets at 9:15 AM each session (inherent since we process per-day)
    vwap_arr = np.full(n_bars, np.nan)
    if use_vwap:
        tp = (highs + lows + closes_arr) / 3.0  # Typical Price = (H+L+C)/3
        if "volume" in day_bars_df.columns:
            vol = day_bars_df["volume"].values.astype(float)
            vol = np.where(vol <= 0, 1.0, vol)  # prevent div-by-zero
            vwap_arr = np.cumsum(tp * vol) / np.cumsum(vol)
        else:
            # No volume data for index — use cumulative typical price average
            vwap_arr = np.cumsum(tp) / np.arange(1, n_bars + 1, dtype=float)

    # Squeeze State: BB inside Keltner Channels = market consolidating
    # BB(20, 2σ) already computed by add_all_indicators: bb_upper, bb_lower
    # KC(EMA21, 1.5*ATR14) computed here: kc_upper = ema21 + 1.5*atr, kc_lower = ema21 - 1.5*atr
    # SQZ_ON when bb_lower > kc_lower AND bb_upper < kc_upper
    squeeze_arr = np.zeros(n_bars, dtype=bool)
    if use_squeeze:
        has_cols = all(c in day_bars_df.columns for c in ["bb_upper", "bb_lower", "ema21", "atr"])
        if has_cols:
            bb_up = day_bars_df["bb_upper"].values
            bb_lo = day_bars_df["bb_lower"].values
            ema21_v = day_bars_df["ema21"].values
            atr_v = day_bars_df["atr"].values
            kc_upper = ema21_v + 1.5 * atr_v
            kc_lower = ema21_v - 1.5 * atr_v
            squeeze_arr = (bb_lo > kc_lower) & (bb_up < kc_upper)

    open_trades = []
    closed_trades = []
    total_day_trades = 0
    last_exit_minute = -cfg["cooldown_min"]
    day_close = day_bars_df["close"].iloc[-1]
    day_flips = 0  # V16b: track reversal flips per day
    use_rt = cfg.get("use_realtime_engine", False)
    prev_rt_state = None  # V17: previous market state for transition detection

    for minute_idx in range(n_bars):
        bar_spot = minute_closes[minute_idx]
        bar_15min = minute_idx // 15
        bar_dte = max(0.05, dte - minute_idx / 1440)

        # ── V17: Compute real-time market state EVERY MINUTE ──
        cur_rt_state = None
        if use_rt:
            cur_rt_state = compute_realtime_state(
                closes_arr, highs, lows, vwap_arr, minute_idx,
                prev_state=prev_rt_state["state"] if prev_rt_state else "RANGING"
            )

        # ====== EXITS (every minute) ======
        trades_to_close = []
        for ti, trade in enumerate(open_trades):
            minutes_held = minute_idx - trade["entry_minute"]
            if minutes_held < 1:
                continue

            exit_reason = None
            action = trade["action"]

            # ── V17: State-based exit (checked FIRST — fastest response) ──
            if use_rt and cfg.get("rt_state_exit", True) and cur_rt_state:
                rt_exit = check_realtime_exits(cfg, cur_rt_state, trade, bar_spot, minute_idx)
                if rt_exit:
                    exit_reason = rt_exit

            if not exit_reason and is_expiry and minute_idx >= cfg["expiry_close_min"]:
                exit_reason = "expiry_close"
            elif not exit_reason and action == "BUY_PUT" and minutes_held >= cfg["max_hold_put"]:
                exit_reason = "time_exit"
            elif not exit_reason and action == "BUY_CALL" and minutes_held >= cfg["max_hold_call"]:
                exit_reason = "time_exit"
            elif action == "BUY_PUT" and minutes_held >= cfg["min_hold_trail_put"]:
                # V14: ATR-based trail adapts to volatility (tight in calm, wide in volatile)
                if use_atr_trail and "atr" in day_bars_df.columns:
                    cur_atr = float(day_bars_df["atr"].iloc[minute_idx])
                    trail_d = atr_trail_mult * cur_atr
                else:
                    trail_d = trade["entry_spot"] * cfg["trail_pct_put"]
                if bar_spot > trade["best_fav"] + trail_d:
                    # PROFIT GATE: only trail stop if currently in profit
                    cur_prem = bs_premium(bar_spot, trade["strike"], bar_dte, vix, trade["opt_type"])
                    if cur_prem > trade["entry_prem"]:
                        exit_reason = "trail_stop"
            elif action == "BUY_CALL" and minutes_held >= cfg["min_hold_trail_call"]:
                # trail_stop loses money directly but frees position slots for new entries
                # Net effect is positive — keep enabled
                # V14: ATR-based trail adapts to volatility
                if use_atr_trail and "atr" in day_bars_df.columns:
                    cur_atr = float(day_bars_df["atr"].iloc[minute_idx])
                    trail_d = atr_trail_mult * cur_atr
                else:
                    trail_d = trade["entry_spot"] * cfg["trail_pct_call"]
                if bar_spot < trade["best_fav"] - trail_d:
                    exit_reason = "trail_stop"

            # ATR-based stop for CALLs — V11: DISABLED by config
            # Data: sr_stop = 0% WR across ALL 46 REAL trades, -Rs5.82L total drag
            # Every sr_stop exit loses money. Let trades reach time_exit (45.7% WR) instead.
            # time_exit avg P&L = +12,472 vs sr_stop avg P&L = -12,666
            if not exit_reason and action == "BUY_CALL" and minutes_held >= 90:
                if not cfg.get("disable_sr_stop_call", False):
                    atr_mult = 2.5
                    call_stop = trade["entry_spot"] - atr_mult * day_atr
                    if bar_spot < call_stop:
                        exit_reason = "sr_stop"

            # ── V16: REVERSAL DETECTION — exit early if market is reversing ──
            # Instead of waiting for trail_stop or time_exit, detect reversal signals
            # and exit + flip to the new direction. This is market-adaptive, not calendar-based.
            max_flips = cfg.get("reversal_max_flips_per_day", 1)
            if not exit_reason and cfg.get("use_reversal_detection", False) and day_flips < max_flips:
                rev = detect_intraday_reversal(cfg, day_bars_df, minute_idx, trade, vwap_arr)
                if rev["detected"]:
                    exit_reason = "reversal_" + rev["direction"]
                    # Store flip info on the trade for the entry logic below
                    trade["_reversal_flip"] = rev

            if exit_reason:
                raw_exit_prem = bs_premium(bar_spot, trade["strike"], bar_dte, vix, trade["opt_type"])
                # REALISTIC: Subtract bid-ask spread (Rs 2) + 0.5% slippage on exit
                exit_prem = max(0.05, raw_exit_prem * 0.995 - 2.0)
                pnl = (exit_prem - trade["entry_prem"]) * trade["qty"] - 80
                trade.update({
                    "exit_minute": minute_idx, "exit_spot": round(bar_spot, 2),
                    "exit_prem": round(exit_prem, 2), "exit_reason": exit_reason,
                    "pnl": round(pnl, 0), "minutes_held": minutes_held,
                })
                trades_to_close.append(ti)
                last_exit_minute = minute_idx

        # ── V16: Collect reversal flip info BEFORE removing trades ──
        pending_flips = []
        for ti in trades_to_close:
            trade = open_trades[ti]
            if trade.get("_reversal_flip"):
                rev = trade["_reversal_flip"]
                pending_flips.append({
                    "flip_action": rev["flip_action"],
                    "confidence": rev["confidence"],
                    "signals": rev["signals"],
                    "direction": rev["direction"],
                    "trigger_minute": minute_idx,
                })

        for ti in reversed(trades_to_close):
            closed_trades.append(open_trades.pop(ti))

        # Update tracking
        for trade in open_trades:
            if trade["action"] == "BUY_CALL" and bar_spot > trade["best_fav"]:
                trade["best_fav"] = bar_spot
            elif trade["action"] == "BUY_PUT" and bar_spot < trade["best_fav"]:
                trade["best_fav"] = bar_spot

        # ── V16: REVERSAL FLIP ENTRIES — immediately enter opposite direction ──
        # This is the KEY innovation: don't just exit, FLIP to profit from the reversal.
        # On April 6: close PUT at 11:30, immediately enter CALL -> ride the rally.
        if cfg.get("reversal_flip_enabled", False) and pending_flips:
            flip_cooldown = cfg.get("reversal_flip_cooldown", 3)
            for flip in pending_flips:
                # Check guards
                if total_day_trades >= cfg["max_trades"]:
                    break
                if len(open_trades) >= cfg["max_concurrent"]:
                    break
                # Don't flip too late in the day
                if minute_idx > cfg.get("block_late_entries", 305):
                    break

                flip_action = flip["flip_action"]
                # Don't enter same direction as existing position
                same_dir = [t for t in open_trades if t["action"] == flip_action]
                if same_dir:
                    continue

                # Apply VWAP filter to flip entry too (respect market structure)
                if use_vwap and minute_idx < len(vwap_arr):
                    cur_vwap = vwap_arr[minute_idx]
                    if not np.isnan(cur_vwap):
                        if flip_action == "BUY_CALL" and bar_spot <= cur_vwap:
                            continue  # Don't flip to CALL if still below VWAP
                        if flip_action == "BUY_PUT" and bar_spot >= cur_vwap:
                            continue  # Don't flip to PUT if still above VWAP

                # Build flip trade
                flip_conf = flip["confidence"] + cfg.get("reversal_flip_conf_boost", 0.15)
                flip_conf = min(0.95, flip_conf)
                is_zh = False

                strike, opt_type = get_strike_and_type(flip_action, bar_spot, vix, is_zh)
                num_lots = get_dynamic_lots(vix, equity, confidence=flip_conf,
                                            zero_hero=is_zh,
                                            recent_wr=recent_wr,
                                            recent_trades=recent_trades)
                # Entry-type lot multiplier for reversal flips
                et_mult = cfg.get("entry_type_lot_mult", {}).get("reversal_flip", 1.0)
                num_lots = max(1, int(num_lots * et_mult))

                # Regime-adaptive lot sizing
                ri = regime_info if regime_info else {"call_mult": 1.0, "put_mult": 1.0}
                if cfg.get("use_regime_detection", False):
                    if flip_action == "BUY_PUT":
                        num_lots = max(1, int(num_lots * ri.get("put_mult", 1.0)))
                    elif flip_action == "BUY_CALL":
                        num_lots = max(1, int(num_lots * ri.get("call_mult", 1.0)))

                qty = num_lots * LOT_SIZE
                raw_entry_prem = bs_premium(bar_spot, strike, bar_dte, vix, opt_type)
                entry_prem = raw_entry_prem * 1.005 + 2.0

                flip_trade = {
                    "date": str(date), "action": flip_action,
                    "entry_type": "reversal_flip",
                    "is_zero_hero": False, "confidence": round(flip_conf, 3),
                    "entry_minute": minute_idx, "entry_spot": round(bar_spot, 2),
                    "entry_prem": round(entry_prem, 2), "strike": int(strike),
                    "opt_type": opt_type, "lots": num_lots, "qty": qty,
                    "vix": round(vix, 1), "is_expiry": is_expiry,
                    "dte": round(bar_dte, 2), "best_fav": bar_spot,
                    "exit_minute": -1, "exit_spot": 0, "exit_prem": 0,
                    "exit_reason": "", "pnl": 0, "minutes_held": 0,
                    "reversal_signals": flip["signals"],
                }
                open_trades.append(flip_trade)
                total_day_trades += 1
                day_flips += 1  # V16b: track flips

        # ── V17: Update state history at end of bar ──
        if use_rt and cur_rt_state:
            prev_rt_state = cur_rt_state

        # ====== ENTRIES ======
        if minute_idx < 5 or minute_idx > cfg["max_entry_min"]:
            continue
        # V13: Block late entries — last hour has negative edge (-Rs15.5K in H1)
        block_late = cfg.get("block_late_entries", 999)
        if not use_rt and minute_idx > block_late:
            continue
        if len(open_trades) >= cfg["max_concurrent"] or total_day_trades >= cfg["max_trades"]:
            continue
        if minute_idx - last_exit_minute < cfg["cooldown_min"]:
            continue

        # VIX filter
        if vix < cfg["vix_floor"] or vix > cfg["vix_ceil"]:
            continue

        # V17: Skip avoid_windows only when NOT using realtime engine
        if not (use_rt and cfg.get("rt_remove_avoid_windows", True)):
            if any(s <= minute_idx < e for s, e in cfg.get("avoid_windows", [])):
                continue

        # V14: Squeeze filter (still used in V17 — it's a real market signal)
        if use_squeeze and minute_idx < len(squeeze_arr) and squeeze_arr[minute_idx]:
            continue

        # ── V17: REAL-TIME ENGINE (every-minute check) ──
        if use_rt and cur_rt_state:
            entries = detect_entries_realtime(
                cfg, cur_rt_state, prev_rt_state, bar_spot, vix,
                minute_idx, open_trades)

            # Also check V8 indicators at 5-min cadence for hybrid signals
            if minute_idx % 5 == 0:
                bar_data = day_bars_df.iloc[minute_idx]
                direction, conf, reasons = detect_entries_v8(
                    bar_data, sr_levels, vix, daily_trend, minute_idx)
                if direction:
                    entries.append((direction, "v8_indicator", conf, False))

        else:
            # ── LEGACY: V8/Composite cadence check ──
            is_hybrid = cfg.get("use_hybrid", False)
            is_v8_tick = (minute_idx % 5 == 0)
            is_comp_tick = (minute_idx % 15 == 0)

            if is_hybrid:
                if not is_v8_tick and not is_comp_tick:
                    continue
            elif minute_idx % cfg["entry_check_interval"] != 0:
                continue

        # Get entry signals (legacy path)
        if not (use_rt and cur_rt_state):
            entries = []

        if not use_rt and cfg.get("use_hybrid", False):
            # ---- HYBRID DUAL-ENGINE ----
            # V8 indicator entries (every 5 min)
            if is_v8_tick:
                bar_data = day_bars_df.iloc[minute_idx]
                direction, conf, reasons = detect_entries_v8(
                    bar_data, sr_levels, vix, daily_trend, minute_idx)
                if direction:
                    entries.append((direction, "v8_indicator", conf, False))

            # V6_3T composite/ORB/gap entries (every 15 min)
            if is_comp_tick:
                comp_entries = detect_entries_composite(
                    cfg, bar_15min, path_15min, support, resistance,
                    vix, gap_pct, best_composite, composite_conf,
                    is_expiry, prev_high, prev_low, above_sma50, above_sma20,
                    bias_val, minute_idx)
                # Composite signals get a confidence boost (proven high WR)
                for action, etype, conf, is_zh in comp_entries:
                    boosted_conf = min(1.0, conf * 1.1)  # 10% boost for proven signals
                    entries.append((action, etype, boosted_conf, is_zh))

        elif use_v8:
            bar_data = day_bars_df.iloc[minute_idx]
            direction, conf, reasons = detect_entries_v8(
                bar_data, sr_levels, vix, daily_trend, minute_idx)
            entries = [(direction, "v8_indicator", conf, False)] if direction else []
        else:
            entries = detect_entries_composite(
                cfg, bar_15min, path_15min, support, resistance,
                vix, gap_pct, best_composite, composite_conf,
                is_expiry, prev_high, prev_low, above_sma50, above_sma20,
                bias_val, minute_idx)

        if not entries:
            continue

        entries.sort(key=lambda x: x[2], reverse=True)
        ri = regime_info if regime_info else {"regime": "neutral", "call_mult": 1.0, "put_mult": 1.0}

        for action, entry_type, conf, is_zh in entries:
            if action is None:
                continue

            # V13: Minimum confidence filter (reject < 0.30 — 18.5% WR, pure drag)
            min_conf = cfg.get("min_confidence_filter", 0)
            if conf < min_conf and not is_zh:
                continue

            # V14: VWAP direction filter — institutional trend confirmation
            # CALL: price must be ABOVE daily VWAP (institutional buying pressure)
            # PUT: price must be BELOW daily VWAP (institutional selling pressure)
            # Skip filter for zero-hero (expiry-day specials)
            if use_vwap and not is_zh and minute_idx < len(vwap_arr):
                cur_vwap = vwap_arr[minute_idx]
                if not np.isnan(cur_vwap):
                    if action == "BUY_CALL" and bar_spot <= cur_vwap:
                        continue
                    if action == "BUY_PUT" and bar_spot >= cur_vwap:
                        continue

            # V14: RSI momentum gate — confirm directional conviction
            # Research: RSI > 60 for CALL, < 40 for PUT (breakouts need momentum)
            # RSI near 50 = no institutional participation → skip
            if use_rsi_gate and not is_zh and "rsi" in day_bars_df.columns:
                bar_rsi = float(day_bars_df["rsi"].iloc[minute_idx])
                rsi_call_min = cfg.get("rsi_call_min", 55)
                rsi_put_max = cfg.get("rsi_put_max", 45)
                if action == "BUY_CALL" and bar_rsi < rsi_call_min:
                    continue
                if action == "BUY_PUT" and bar_rsi > rsi_put_max:
                    continue

            if len(open_trades) >= cfg["max_concurrent"]:
                break
            same_dir = [t for t in open_trades if t["action"] == action]
            if same_dir:
                continue

            # V11: Block CALL entries in 4th hour (225-300 min) — all losses in REAL data
            if cfg.get("block_call_4th_hour", False) and action == "BUY_CALL":
                if 225 <= minute_idx < 300:
                    continue

            # V13: Regime-based direction filtering
            # In BULLISH market: require higher conviction for counter-trend trades
            # In VOLATILE/BEARISH: allow most trades (model's strength)
            if cfg.get("use_regime_detection", False):
                if ri["regime"] == "bullish" and action == "BUY_PUT":
                    if conf < 0.45:  # Need 45%+ confidence for counter-trend PUTs in bull
                        continue
                elif ri["regime"] == "sideways":
                    if conf < 0.35:  # Slightly higher bar in sideways (no edge)
                        continue
                # In bearish/volatile: let everything through (model excels here)

            strike, opt_type = get_strike_and_type(action, bar_spot, vix, is_zh)
            num_lots = get_dynamic_lots(vix, equity, confidence=conf,
                                        zero_hero=is_zh,
                                        recent_wr=recent_wr,
                                        recent_trades=recent_trades)
            # Entry-type lot multiplier (scale down bad entry types)
            et_mult = cfg.get("entry_type_lot_mult", {}).get(entry_type, 1.0)
            num_lots = max(1, int(num_lots * et_mult))

            # V13: Regime-adaptive lot sizing (replaces fixed PUT/CALL bias)
            if cfg.get("use_regime_detection", False):
                # Use regime-specific multipliers instead of fixed bias
                if action == "BUY_PUT":
                    num_lots = max(1, int(num_lots * ri.get("put_mult", 1.0)))
                elif action == "BUY_CALL":
                    num_lots = max(1, int(num_lots * ri.get("call_mult", 1.0)))
            else:
                # Fallback: V11 fixed bias
                if action == "BUY_PUT":
                    num_lots = max(1, int(num_lots * cfg.get("put_bias_lot_mult", 1.0)))
                elif action == "BUY_CALL":
                    num_lots = max(1, int(num_lots * cfg.get("call_bias_lot_mult", 1.0)))

            # V13: Expiry day lot reduction (Thu = 19.7% WR in H1, pure theta drain)
            if is_expiry:
                exp_mult = cfg.get("expiry_day_lot_mult", 1.0)
                num_lots = max(1, int(num_lots * exp_mult))

            # V11: VIX regime lot scaling — VIX 14-16 is the sweet spot (83% of profits)
            vix_sweet_min = cfg.get("vix_sweet_min", 0)
            vix_sweet_max = cfg.get("vix_sweet_max", 999)
            if vix_sweet_min <= vix <= vix_sweet_max:
                num_lots = max(1, int(num_lots * cfg.get("vix_sweet_lot_mult", 1.0)))
            vix_danger_min = cfg.get("vix_danger_min", 999)
            vix_danger_max = cfg.get("vix_danger_max", 999)
            if vix_danger_min <= vix <= vix_danger_max:
                num_lots = max(1, int(num_lots * cfg.get("vix_danger_lot_mult", 1.0)))

            # V15: ML-discovered RSI lot scaling
            # RSI 20-35 = 51.4% WR, +Rs47K avg → boost lots (high conviction zone)
            # RSI 55-65 = 27.3% WR, -Rs16.8K avg → reduce lots (danger zone)
            if "rsi" in day_bars_df.columns:
                bar_rsi = float(day_bars_df["rsi"].iloc[minute_idx])
                rsi_sw_lo = cfg.get("rsi_sweet_low", 0)
                rsi_sw_hi = cfg.get("rsi_sweet_high", 0)
                if rsi_sw_lo <= bar_rsi <= rsi_sw_hi:
                    num_lots = max(1, int(num_lots * cfg.get("rsi_sweet_lot_mult", 1.0)))
                rsi_dg_lo = cfg.get("rsi_danger_low", 999)
                rsi_dg_hi = cfg.get("rsi_danger_high", 999)
                if rsi_dg_lo <= bar_rsi <= rsi_dg_hi:
                    num_lots = max(1, int(num_lots * cfg.get("rsi_danger_lot_mult", 1.0)))

            qty = num_lots * LOT_SIZE
            raw_entry_prem = bs_premium(bar_spot, strike, bar_dte, vix, opt_type)
            # REALISTIC: Add bid-ask spread (Rs 2) + 0.5% slippage on entry
            entry_prem = raw_entry_prem * 1.005 + 2.0

            trade = {
                "date": str(date), "action": action, "entry_type": entry_type,
                "is_zero_hero": is_zh, "confidence": round(conf, 3),
                "entry_minute": minute_idx, "entry_spot": round(bar_spot, 2),
                "entry_prem": round(entry_prem, 2), "strike": int(strike),
                "opt_type": opt_type, "lots": num_lots, "qty": qty,
                "vix": round(vix, 1), "is_expiry": is_expiry,
                "dte": round(bar_dte, 2), "best_fav": bar_spot,
                "exit_minute": -1, "exit_spot": 0, "exit_prem": 0,
                "exit_reason": "", "pnl": 0, "minutes_held": 0,
            }
            open_trades.append(trade)
            total_day_trades += 1

    # Force close
    for trade in open_trades:
        raw_exit_prem = bs_premium(day_close, trade["strike"],
                               max(0.05, dte - n_bars / 1440), vix, trade["opt_type"])
        # REALISTIC: Subtract bid-ask spread (Rs 2) + 0.5% slippage on exit
        exit_prem = max(0.05, raw_exit_prem * 0.995 - 2.0)
        pnl = (exit_prem - trade["entry_prem"]) * trade["qty"] - 80
        trade.update({
            "exit_minute": n_bars - 1, "exit_spot": round(day_close, 2),
            "exit_prem": round(exit_prem, 2), "exit_reason": "eod_close",
            "pnl": round(pnl, 0), "minutes_held": n_bars - 1 - trade["entry_minute"],
        })
        closed_trades.append(trade)

    return closed_trades


# =======================================================================
# RUN MODEL
# =======================================================================

def run_model(cfg, daily_df, close_prices, day_groups, trading_dates,
              vix_lookup, daily_trend_df, test_start, starting_equity=None):
    """Run a model on the test period."""
    equity = starting_equity if starting_equity is not None else CAPITAL
    start_equity = equity
    peak = equity
    max_dd = 0
    all_trades = []
    daily_pnls = []
    entry_stats = defaultdict(lambda: {"count": 0, "pnl": 0, "wins": 0})
    exit_stats = defaultdict(lambda: {"count": 0, "pnl": 0, "wins": 0})

    # Track rolling win rate for dynamic sizing
    recent_results = []  # list of 1 (win) or 0 (loss) for last N trades

    for i in range(len(daily_df)):
        date = daily_df.index[i].date()
        if date < test_start:
            continue
        if date not in day_groups:
            continue
        # Monday filter: skip worst day (23.4% WR, -Rs155K in real data)
        avoid_days = cfg.get("avoid_days", [])
        if date.strftime("%A") in avoid_days:
            continue
        day_bars = day_groups[date]
        if len(day_bars) < 60:
            continue

        prev_ohlc = None
        if i > 0:
            prev_idx = daily_df.index[i - 1].date()
            if prev_idx in day_groups:
                pd_bars = day_groups[prev_idx]
                prev_ohlc = {
                    "high": pd_bars["high"].max(), "low": pd_bars["low"].min(),
                    "close": pd_bars["close"].iloc[-1], "open": pd_bars["open"].iloc[0],
                }

        row = daily_df.iloc[i]
        vix = vix_lookup.get(date, 14.0)

        # Expiry detection — June 2024 was Thursday expiry
        is_expiry = date.strftime("%A") == "Thursday"
        dow = date.weekday()
        target = 3  # Thursday in 2024
        if dow <= target:
            dte = max(target - dow, 0.5)
        else:
            dte = max(7 - dow + target, 0.5)

        # Daily trend
        if date in daily_trend_df.index:
            d = daily_trend_df.loc[date]
            if d["close"] > d["sma20"] and d["ema9"] > d["ema21"]:
                daily_trend = "bullish"
            elif d["close"] < d["sma20"] and d["ema9"] < d["ema21"]:
                daily_trend = "bearish"
            else:
                daily_trend = "neutral"
        else:
            daily_trend = "neutral"

        # Compute rolling win rate from last 10 trades
        recent_wr = sum(recent_results[-10:]) / len(recent_results[-10:]) if recent_results else 0.5
        recent_n = len(recent_results)

        # V13: Market regime detection
        regime_info = {"regime": "neutral", "strength": 0.0, "call_mult": 1.0, "put_mult": 1.0}
        if cfg.get("use_regime_detection", False):
            regime_info = detect_market_regime(daily_df, i)

        day_trades = simulate_day(
            cfg, day_bars, date, prev_ohlc, vix, daily_trend, dte, is_expiry,
            daily_df, i, close_prices,
            bool(row.get("AboveSMA50", True)), bool(row.get("AboveSMA20", True)),
            float(row.get("RSI", 50)), float(row.get("PrevChange%", 0)),
            bool(row.get("VIXSpike", False)),
            float(row.get("SMA20")) if pd.notna(row.get("SMA20")) else None,
            float(row.get("SMA50")) if pd.notna(row.get("SMA50")) else None,
            float(row.get("EMA9")) if pd.notna(row.get("EMA9")) else None,
            float(row.get("EMA21")) if pd.notna(row.get("EMA21")) else None,
            float(row.get("WeeklySMA")) if pd.notna(row.get("WeeklySMA")) else None,
            float(row.get("GapPct", 0)),
            equity=equity, recent_wr=recent_wr, recent_trades=recent_n,
            regime_info=regime_info,
        )

        # BTST
        if cfg["btst_enabled"] and i + 1 < len(daily_df):
            next_row = daily_df.iloc[i + 1]
            next_open = float(next_row["Open"])
            day_close = float(row["Close"])
            for t in day_trades:
                if (t["action"] == "BUY_PUT" and t["pnl"] > 0 and not is_expiry
                        and vix < cfg["btst_vix_cap"]
                        and t["exit_reason"] in ("eod_close", "time_exit")):
                    gap = (next_open - day_close) / day_close * 100
                    if gap < 0:
                        on_pnl = (day_close - next_open) * t["qty"] * 0.5 - 50
                        on_pnl = max(on_pnl, -t["pnl"] * 0.5)
                    else:
                        on_pnl = -abs(gap) * t["qty"] * 0.3
                        on_pnl = max(on_pnl, -t["pnl"] * 0.5)
                    t["btst_pnl"] = round(on_pnl, 0)
                    t["pnl"] = round(t["pnl"] + on_pnl, 0)

        day_pnl = sum(t["pnl"] for t in day_trades)
        equity += day_pnl
        if equity > peak: peak = equity
        dd = (peak - equity) / peak * 100
        if dd > max_dd: max_dd = dd
        daily_pnls.append(day_pnl)

        for t in day_trades:
            all_trades.append(t)
            # Track rolling results for dynamic sizing
            recent_results.append(1 if t["pnl"] > 0 else 0)
            et = t.get("entry_type", "?")
            entry_stats[et]["count"] += 1
            entry_stats[et]["pnl"] += t["pnl"]
            if t["pnl"] > 0: entry_stats[et]["wins"] += 1
            er = t.get("exit_reason", "?")
            exit_stats[er]["count"] += 1
            exit_stats[er]["pnl"] += t["pnl"]
            if t["pnl"] > 0: exit_stats[er]["wins"] += 1

    net = equity - start_equity
    total = len(all_trades)
    wins = [t for t in all_trades if t["pnl"] > 0]
    losses = [t for t in all_trades if t["pnl"] <= 0]
    wr = len(wins) / total * 100 if total else 0
    gw = sum(t["pnl"] for t in wins)
    gl = abs(sum(t["pnl"] for t in losses))
    pf = gw / gl if gl > 0 else float("inf")
    arr = np.array([d for d in daily_pnls if d != 0])
    sharpe = (arr.mean() / arr.std()) * np.sqrt(252) if len(arr) > 1 and arr.std() > 0 else 0

    # Lot sizing stats
    lots_list = [t["lots"] for t in all_trades]
    avg_lots = np.mean(lots_list) if lots_list else 0
    max_lots = max(lots_list) if lots_list else 0
    avg_qty = np.mean([t["qty"] for t in all_trades]) if all_trades else 0
    final_equity = equity

    return {
        "name": cfg["name"], "net_pnl": round(net),
        "return_pct": round(net / max(start_equity, 1) * 100, 1),
        "start_equity": round(start_equity), "final_equity": round(final_equity),
        "total_trades": total, "trades_per_day": round(total / max(len(daily_pnls), 1), 1),
        "win_rate": round(wr, 1), "wins": len(wins), "losses": len(losses),
        "sharpe": round(sharpe, 2), "profit_factor": round(pf, 2),
        "max_drawdown": round(max_dd, 1),
        "avg_win": round(np.mean([t["pnl"] for t in wins])) if wins else 0,
        "avg_loss": round(np.mean([t["pnl"] for t in losses])) if losses else 0,
        "pnl_per_trade": round(net / max(total, 1)),
        "btst_count": len([t for t in all_trades if t.get("btst_pnl", 0) != 0]),
        "btst_pnl": round(sum(t.get("btst_pnl", 0) for t in all_trades)),
        "avg_lots": round(avg_lots, 1), "max_lots": max_lots,
        "avg_qty": round(avg_qty), "final_equity": round(final_equity),
        "entry_stats": dict(entry_stats), "exit_stats": dict(exit_stats),
        "all_trades": all_trades,
    }


# =======================================================================
# MAIN
# =======================================================================

if __name__ == "__main__":
    print("=" * 120)
    print("  OUT-OF-SAMPLE TEST: V6_3T vs V7_5T vs V8 on June 2024")
    print("  Training: Oct 2025 - Apr 2026 | Test: June 2024 (UNSEEN)")
    print("  June 4, 2024 = Election crash day (-5.93%!)")
    print("=" * 120)

    # Load data
    data_dir = project_root / "data" / "historical"
    nifty = pd.read_csv(
        data_dir / "nifty_min_2024-05-01_2024-06-30.csv",
        parse_dates=["timestamp"], index_col="timestamp")
    vix_df = pd.read_csv(
        data_dir / "vix_min_2024-05-01_2024-06-30.csv",
        parse_dates=["timestamp"], index_col="timestamp")
    print(f"Loaded {len(nifty)} bars ({nifty.index[0].date()} to {nifty.index[-1].date()})")

    vix_lookup = {idx.date(): row["close"] for idx, row in vix_df.iterrows()}

    # Compute indicators for V8
    print("Computing 1-min indicators...", flush=True)
    nifty = add_all_indicators(nifty)

    # Group by date
    day_groups = {date: group for date, group in nifty.groupby(nifty.index.date)}
    all_dates = sorted(day_groups.keys())
    print(f"Total days: {len(all_dates)}")

    # Build daily data
    daily = nifty.resample("D").agg({
        "open": "first", "high": "max", "low": "min", "close": "last"
    }).dropna()
    daily.columns = ["Open", "High", "Low", "Close"]

    # Add indicators
    for idx_date in daily.index:
        vix_val = vix_lookup.get(idx_date.date(), 14.0)
        daily.loc[idx_date, "VIX"] = vix_val
    daily["VIX"] = daily["VIX"].ffill().bfill().fillna(14.0)
    daily["PrevVIX"] = daily["VIX"].shift(1).fillna(daily["VIX"].iloc[0])
    daily["Change%"] = daily["Close"].pct_change() * 100
    daily["PrevChange%"] = daily["Change%"].shift(1).fillna(0)
    daily["SMA50"] = daily["Close"].rolling(50, min_periods=1).mean()
    daily["SMA20"] = daily["Close"].rolling(20, min_periods=1).mean()
    daily["AboveSMA50"] = daily["Close"] > daily["SMA50"]
    daily["AboveSMA20"] = daily["Close"] > daily["SMA20"]
    daily["EMA9"] = daily["Close"].ewm(span=9).mean()
    daily["EMA21"] = daily["Close"].ewm(span=21).mean()
    daily["WeeklySMA"] = daily["Close"].rolling(5).mean().rolling(4, min_periods=1).mean()
    delta = daily["Close"].diff()
    gain = delta.clip(lower=0).rolling(14, min_periods=1).mean()
    loss_s = (-delta.clip(upper=0)).rolling(14, min_periods=1).mean()
    daily["RSI"] = 100 - (100 / (1 + gain / loss_s.replace(0, 0.001)))
    daily["PrevHigh"] = daily["High"].shift(1)
    daily["PrevLow"] = daily["Low"].shift(1)
    daily["VIXSpike"] = daily["VIX"] > daily["PrevVIX"] * 1.15
    daily["PrevClose"] = daily["Close"].shift(1)
    daily["GapPct"] = (daily["Open"] - daily["PrevClose"]) / daily["PrevClose"] * 100
    daily["GapPct"] = daily["GapPct"].fillna(0)
    daily["DOW"] = daily.index.day_name()

    close_prices = daily["Close"].values.tolist()

    # Daily trend df
    daily_trend_df = nifty.resample("D").agg({"close": "last"}).dropna()
    daily_trend_df["sma20"] = daily_trend_df["close"].rolling(20, min_periods=1).mean()
    daily_trend_df["ema9"] = compute_ema(daily_trend_df["close"], 9)
    daily_trend_df["ema21"] = compute_ema(daily_trend_df["close"], 21)

    # Test start: June 1, 2024
    test_start = dt.date(2024, 6, 1)
    june_dates = [d for d in all_dates if d >= test_start]
    print(f"\nJune 2024 test days: {len(june_dates)}")
    for d in june_dates:
        bars = day_groups[d]
        vix = vix_lookup.get(d, 14.0)
        o, c = bars["open"].iloc[0], bars["close"].iloc[-1]
        chg = (c - o) / o * 100
        print(f"  {d} {d.strftime('%a')}: O={o:.0f} C={c:.0f} Chg={chg:+.2f}% VIX={vix:.1f} bars={len(bars)}")

    # Run all 3 models
    print("\n--- Running Models ---")
    results = {}
    for cfg in [V6_3T_CONFIG, V7_5T_CONFIG, V8_CONFIG, V9_HYBRID_CONFIG]:
        print(f"\n  Running {cfg['name']}...", flush=True)
        r = run_model(cfg, daily, close_prices, day_groups, june_dates,
                      vix_lookup, daily_trend_df, test_start)
        results[cfg["name"]] = r
        print(f"    {r['name']}: Rs {r['net_pnl']:>+,} | {r['total_trades']}t ({r['trades_per_day']:.1f}/day) | "
              f"WR {r['win_rate']:.1f}% | Sharpe {r['sharpe']:.2f} | DD {r['max_drawdown']:.1f}% | "
              f"Avg {r['avg_lots']:.1f} lots ({r['avg_qty']} qty) | Final Rs {r['final_equity']:>,}")

    # ======= COMPARISON TABLE =======
    print("\n" + "=" * 120)
    print("  OUT-OF-SAMPLE RESULTS: June 2024 (UNSEEN DATA)")
    print("=" * 120)

    names = list(results.keys())
    header = f"{'Metric':<28}"
    for n in names:
        header += f" {n:>18}"
    print(header)
    print("-" * 120)

    metrics = [
        ("Net P&L", lambda r: f"Rs {r['net_pnl']:>+,}"),
        ("Return %", lambda r: f"{r['return_pct']:>+.1f}%"),
        ("Final Equity", lambda r: f"Rs {r['final_equity']:>,}"),
        ("Total Trades", lambda r: f"{r['total_trades']}"),
        ("Trades / Day", lambda r: f"{r['trades_per_day']:.1f}"),
        ("Win / Lose", lambda r: f"{r['wins']}W / {r['losses']}L"),
        ("Win Rate", lambda r: f"{r['win_rate']:.1f}%"),
        ("Sharpe Ratio", lambda r: f"{r['sharpe']:.2f}"),
        ("Profit Factor", lambda r: f"{r['profit_factor']:.2f}"),
        ("Max Drawdown", lambda r: f"{r['max_drawdown']:.1f}%"),
        ("Avg Win", lambda r: f"Rs {r['avg_win']:>+,}"),
        ("Avg Loss", lambda r: f"Rs {r['avg_loss']:>+,}"),
        ("P&L per Trade", lambda r: f"Rs {r['pnl_per_trade']:>+,}"),
        ("Avg Lots/Trade", lambda r: f"{r['avg_lots']:.1f}"),
        ("Max Lots Used", lambda r: f"{r['max_lots']}"),
        ("Avg Qty/Trade", lambda r: f"{r['avg_qty']}"),
        ("BTST Trades", lambda r: f"{r['btst_count']}"),
        ("BTST P&L", lambda r: f"Rs {r['btst_pnl']:>+,}"),
    ]

    for name, fmt in metrics:
        row = f"  {name:<26}"
        for n in names:
            row += f" {fmt(results[n]):>18}"
        print(row)

    # Exit reasons
    print(f"\n  EXIT REASONS:")
    all_exits = set()
    for r in results.values():
        all_exits.update(r["exit_stats"].keys())
    for er in sorted(all_exits):
        row = f"    {er:<24}"
        for n in names:
            s = results[n]["exit_stats"].get(er, {"count": 0, "pnl": 0, "wins": 0})
            wr_val = s["wins"] / s["count"] * 100 if s["count"] > 0 else 0
            row += f"  {s['count']:>2}t Rs{s['pnl']:>+8,.0f} {wr_val:>3.0f}%"
        print(row)

    # Entry types
    print(f"\n  ENTRY TYPES:")
    all_entries = set()
    for r in results.values():
        all_entries.update(r["entry_stats"].keys())
    for et in sorted(all_entries):
        row = f"    {et:<24}"
        for n in names:
            s = results[n]["entry_stats"].get(et, {"count": 0, "pnl": 0, "wins": 0})
            wr_val = s["wins"] / s["count"] * 100 if s["count"] > 0 else 0
            row += f"  {s['count']:>2}t Rs{s['pnl']:>+8,.0f} {wr_val:>3.0f}%"
        print(row)

    # ======= COMPARISON WITH TRAINING PERFORMANCE =======
    print("\n" + "=" * 120)
    print("  TRAINING vs TEST COMPARISON (overfit detection)")
    print("=" * 120)
    training_results = {
        "V6_3T": {"pnl": 464805, "wr": 50.7, "sharpe": 7.79, "dd": 6.7, "pf": 2.55, "tpd": 1.9},
        "V7_5T": {"pnl": 574317, "wr": 45.4, "sharpe": 7.04, "dd": 6.6, "pf": 2.29, "tpd": 2.2},
        "V8":    {"pnl": 527278, "wr": 46.4, "sharpe": 6.45, "dd": 3.9, "pf": 1.85, "tpd": 2.5},
    }

    print(f"\n  {'Model':<10} {'Train PnL':>12} {'Test PnL':>12} {'Train WR':>10} {'Test WR':>10} "
          f"{'Train Sharpe':>14} {'Test Sharpe':>14} {'Overfit?':>10}")
    print("-" * 100)
    for name in names:
        tr = training_results.get(name, {})
        te = results[name]
        # Normalize P&L to per-day for fair comparison
        tr_pnl_pd = tr.get("pnl", 0) / 104  # 104 training days
        te_pnl_pd = te["net_pnl"] / max(len(june_dates), 1)
        degradation = (tr_pnl_pd - te_pnl_pd) / max(tr_pnl_pd, 1) * 100 if tr_pnl_pd > 0 else 0

        overfit = "YES" if degradation > 60 else ("MAYBE" if degradation > 30 else "NO")

        print(f"  {name:<10} Rs {tr.get('pnl', 0):>+,} Rs {te['net_pnl']:>+,} "
              f"{tr.get('wr', 0):>9.1f}% {te['win_rate']:>9.1f}% "
              f"{tr.get('sharpe', 0):>13.2f} {te['sharpe']:>13.2f} "
              f"{overfit:>10}")
        print(f"  {'':>10} (Rs {tr_pnl_pd:>+,.0f}/day) (Rs {te_pnl_pd:>+,.0f}/day) "
              f"degradation: {degradation:>+.0f}%")

    # ======= WINNER =======
    best = max(results.values(), key=lambda r: r["sharpe"] if r["total_trades"] > 5 else -999)
    best_pnl = max(results.values(), key=lambda r: r["net_pnl"])

    print(f"\n  WINNER (risk-adjusted): {best['name']}  Sharpe={best['sharpe']:.2f}")
    print(f"  WINNER (raw profit):    {best_pnl['name']}  Rs {best_pnl['net_pnl']:+,}")
    print("=" * 120)

    # Save
    save = {n: {k: v for k, v in r.items() if k not in ("all_trades", "entry_stats", "exit_stats")}
            for n, r in results.items()}
    for n, r in results.items():
        save[n]["entry_type_stats"] = {k: dict(v) for k, v in r["entry_stats"].items()}
        save[n]["exit_reason_stats"] = {k: dict(v) for k, v in r["exit_stats"].items()}

    out = project_root / "data" / "oos_june2024_results.json"
    with open(out, "w") as f:
        json.dump(save, f, indent=2, default=str)
    print(f"\nResults saved to {out}")
