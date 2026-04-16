"""
DAY-BY-DAY ANALYSIS ON REAL 1-MIN DATA — Learn from actual price action.

This script:
  1. Computes TradingView-style indicators on real 1-min bars
  2. Walks through each day, identifies entries/exits from real price action
  3. Records EVERY trade with full indicator snapshot
  4. After all days, analyzes what separates winners from losers
  5. Builds optimized model from learned patterns

Indicators (TradingView standard):
  - EMA 9/21 (fast trend on 1-min)
  - EMA 50 (slow trend on 15-min resampled)
  - RSI 14 (momentum)
  - Supertrend (10, 3.0) — ATR-based trend follower
  - MACD (12, 26, 9)
  - Bollinger Bands (20, 2.0)
  - ADX (14) — trend strength
  - Pivot Points (daily — previous day's high/low/close)
  - Stochastic RSI (14, 14, 3, 3)
  - ATR (14) — volatility measure

Data: 38,685 real 1-min bars from Kite Connect (Oct 2025 - Apr 2026)
Volume: NOT available (NIFTY 50 index) — all price-based indicators
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backtesting.paper_trading_real_data import (
    bs_premium,
    get_strike_and_type,
    LOT_SIZE,
)

CAPITAL = 200_000


# =====================================================================
# TECHNICAL INDICATORS (TradingView-compatible)
# =====================================================================

def compute_ema(series, span):
    """Exponential Moving Average."""
    return series.ewm(span=span, adjust=False).mean()


def compute_rsi(series, period=14):
    """RSI — Relative Strength Index."""
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / loss.replace(0, 1e-10)
    return 100 - (100 / (1 + rs))


def compute_atr(high, low, close, period=14):
    """Average True Range."""
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()


def compute_supertrend(high, low, close, period=10, multiplier=3.0):
    """Supertrend indicator — TradingView standard."""
    atr = compute_atr(high, low, close, period)
    hl2 = (high + low) / 2

    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr

    supertrend = pd.Series(np.nan, index=close.index)
    direction = pd.Series(1, index=close.index)  # 1=up, -1=down

    for i in range(1, len(close)):
        if close.iloc[i] > upper_band.iloc[i-1]:
            direction.iloc[i] = 1
        elif close.iloc[i] < lower_band.iloc[i-1]:
            direction.iloc[i] = -1
        else:
            direction.iloc[i] = direction.iloc[i-1]

        if direction.iloc[i] == 1:
            lower_band.iloc[i] = max(lower_band.iloc[i],
                                      lower_band.iloc[i-1] if direction.iloc[i-1] == 1 else lower_band.iloc[i])
            supertrend.iloc[i] = lower_band.iloc[i]
        else:
            upper_band.iloc[i] = min(upper_band.iloc[i],
                                      upper_band.iloc[i-1] if direction.iloc[i-1] == -1 else upper_band.iloc[i])
            supertrend.iloc[i] = upper_band.iloc[i]

    return supertrend, direction


def compute_macd(close, fast=12, slow=26, signal=9):
    """MACD — TradingView standard."""
    ema_fast = compute_ema(close, fast)
    ema_slow = compute_ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = compute_ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_bollinger(close, period=20, std_dev=2.0):
    """Bollinger Bands."""
    sma = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    return upper, sma, lower


def compute_adx(high, low, close, period=14):
    """ADX — Average Directional Index (trend strength)."""
    plus_dm = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)

    # Zero out when the other is larger
    mask = plus_dm > minus_dm
    minus_dm[mask & (plus_dm > 0)] = 0
    plus_dm[~mask & (minus_dm > 0)] = 0

    atr = compute_atr(high, low, close, period)

    plus_di = 100 * (plus_dm.ewm(alpha=1/period, adjust=False).mean() / atr.replace(0, 1e-10))
    minus_di = 100 * (minus_dm.ewm(alpha=1/period, adjust=False).mean() / atr.replace(0, 1e-10))

    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1e-10))
    adx = dx.ewm(alpha=1/period, adjust=False).mean()

    return adx, plus_di, minus_di


def compute_stoch_rsi(close, rsi_period=14, stoch_period=14, k_smooth=3, d_smooth=3):
    """Stochastic RSI — TradingView standard."""
    rsi = compute_rsi(close, rsi_period)
    rsi_min = rsi.rolling(stoch_period).min()
    rsi_max = rsi.rolling(stoch_period).max()
    stoch_rsi = (rsi - rsi_min) / (rsi_max - rsi_min).replace(0, 1e-10) * 100
    k = stoch_rsi.rolling(k_smooth).mean()
    d = k.rolling(d_smooth).mean()
    return k, d


def compute_pivot_points(prev_high, prev_low, prev_close):
    """Standard Pivot Points — most used S/R on TradingView."""
    pp = (prev_high + prev_low + prev_close) / 3
    r1 = 2 * pp - prev_low
    s1 = 2 * pp - prev_high
    r2 = pp + (prev_high - prev_low)
    s2 = pp - (prev_high - prev_low)
    r3 = prev_high + 2 * (pp - prev_low)
    s3 = prev_low - 2 * (prev_high - pp)
    return {"PP": pp, "R1": r1, "R2": r2, "R3": r3, "S1": s1, "S2": s2, "S3": s3}


def add_all_indicators(df):
    """Add ALL indicators to a 1-min DataFrame."""
    c = df["close"]
    h = df["high"]
    l = df["low"]

    # EMAs
    df["ema9"] = compute_ema(c, 9)
    df["ema21"] = compute_ema(c, 21)
    df["ema50"] = compute_ema(c, 50)

    # RSI
    df["rsi"] = compute_rsi(c, 14)

    # Supertrend
    df["supertrend"], df["st_direction"] = compute_supertrend(h, l, c, 10, 3.0)

    # MACD
    df["macd"], df["macd_signal"], df["macd_hist"] = compute_macd(c, 12, 26, 9)

    # Bollinger Bands
    df["bb_upper"], df["bb_mid"], df["bb_lower"] = compute_bollinger(c, 20, 2.0)

    # ADX
    df["adx"], df["plus_di"], df["minus_di"] = compute_adx(h, l, c, 14)

    # Stochastic RSI
    df["stoch_k"], df["stoch_d"] = compute_stoch_rsi(c, 14, 14, 3, 3)

    # ATR
    df["atr"] = compute_atr(h, l, c, 14)
    df["atr_pct"] = df["atr"] / c * 100  # ATR as % of price

    # Price position relative to indicators
    df["above_ema9"] = c > df["ema9"]
    df["above_ema21"] = c > df["ema21"]
    df["above_ema50"] = c > df["ema50"]
    df["ema9_above_ema21"] = df["ema9"] > df["ema21"]

    # EMA crossover detection
    df["ema9_cross_up_21"] = (df["ema9"] > df["ema21"]) & (df["ema9"].shift(1) <= df["ema21"].shift(1))
    df["ema9_cross_down_21"] = (df["ema9"] < df["ema21"]) & (df["ema9"].shift(1) >= df["ema21"].shift(1))

    # MACD crossover
    df["macd_cross_up"] = (df["macd"] > df["macd_signal"]) & (df["macd"].shift(1) <= df["macd_signal"].shift(1))
    df["macd_cross_down"] = (df["macd"] < df["macd_signal"]) & (df["macd"].shift(1) >= df["macd_signal"].shift(1))

    # Stoch RSI signals
    df["stoch_oversold"] = df["stoch_k"] < 20
    df["stoch_overbought"] = df["stoch_k"] > 80
    df["stoch_cross_up"] = (df["stoch_k"] > df["stoch_d"]) & (df["stoch_k"].shift(1) <= df["stoch_d"].shift(1))
    df["stoch_cross_down"] = (df["stoch_k"] < df["stoch_d"]) & (df["stoch_k"].shift(1) >= df["stoch_d"].shift(1))

    # Bollinger Band touches
    df["at_bb_upper"] = c >= df["bb_upper"]
    df["at_bb_lower"] = c <= df["bb_lower"]
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"] * 100  # Squeeze detection

    return df


# =====================================================================
# DAY-BY-DAY TRADE ENGINE
# =====================================================================

def find_support_resistance(day_bars, prev_day_ohlc, pivot_points):
    """Find real S/R levels from price action + pivots."""
    levels = []

    # Pivot levels
    for name, val in pivot_points.items():
        levels.append({"level": val, "type": name, "strength": 2})

    # Previous day high/low
    if prev_day_ohlc:
        levels.append({"level": prev_day_ohlc["high"], "type": "prev_high", "strength": 3})
        levels.append({"level": prev_day_ohlc["low"], "type": "prev_low", "strength": 3})
        levels.append({"level": prev_day_ohlc["close"], "type": "prev_close", "strength": 1})

    # Round numbers (psychological)
    spot = day_bars["close"].iloc[0]
    nearest_100 = round(spot / 100) * 100
    for offset in [-200, -100, 0, 100, 200]:
        levels.append({"level": nearest_100 + offset, "type": "round_number", "strength": 1})

    # Sort and deduplicate (merge levels within 10 points)
    levels.sort(key=lambda x: x["level"])
    merged = []
    for lev in levels:
        if merged and abs(lev["level"] - merged[-1]["level"]) < 10:
            # Merge: keep higher strength
            if lev["strength"] > merged[-1]["strength"]:
                merged[-1] = lev
            merged[-1]["strength"] += 1
        else:
            merged.append(lev)

    return merged


def compute_confluence_score(bar, day_bars, idx, sr_levels, vix, daily_trend):
    """Compute entry confluence score from all indicators at a specific bar.

    Returns (direction, score, reasons) where:
      direction: "CALL" or "PUT"
      score: 0.0 to 1.0 (higher = stronger signal)
      reasons: list of indicator signals that fired
    """
    call_score = 0.0
    put_score = 0.0
    call_reasons = []
    put_reasons = []
    spot = bar["close"]

    # 1. SUPERTREND DIRECTION (weight: 3)
    if bar["st_direction"] == 1:
        call_score += 3.0
        call_reasons.append("supertrend_bullish")
    elif bar["st_direction"] == -1:
        put_score += 3.0
        put_reasons.append("supertrend_bearish")

    # 2. EMA ALIGNMENT (weight: 2.5)
    if bar["above_ema9"] and bar["above_ema21"] and bar["ema9_above_ema21"]:
        call_score += 2.5
        call_reasons.append("ema_aligned_bull")
    elif not bar["above_ema9"] and not bar["above_ema21"] and not bar["ema9_above_ema21"]:
        put_score += 2.5
        put_reasons.append("ema_aligned_bear")

    # 3. EMA CROSSOVER (weight: 2.0 — fresh signal)
    if bar.get("ema9_cross_up_21", False):
        call_score += 2.0
        call_reasons.append("ema_cross_up")
    if bar.get("ema9_cross_down_21", False):
        put_score += 2.0
        put_reasons.append("ema_cross_down")

    # 4. RSI ZONES (weight: 1.5)
    rsi = bar["rsi"]
    if rsi < 30:
        call_score += 1.5  # Oversold = bounce expected
        call_reasons.append(f"rsi_oversold_{rsi:.0f}")
    elif rsi < 40:
        call_score += 0.5
        call_reasons.append(f"rsi_low_{rsi:.0f}")
    elif rsi > 70:
        put_score += 1.5  # Overbought = pullback expected
        put_reasons.append(f"rsi_overbought_{rsi:.0f}")
    elif rsi > 60:
        put_score += 0.5
        put_reasons.append(f"rsi_high_{rsi:.0f}")

    # 5. MACD (weight: 2.0)
    if bar["macd_hist"] > 0 and bar.get("macd_cross_up", False):
        call_score += 2.0
        call_reasons.append("macd_bull_cross")
    elif bar["macd_hist"] > 0:
        call_score += 0.5
        call_reasons.append("macd_positive")
    if bar["macd_hist"] < 0 and bar.get("macd_cross_down", False):
        put_score += 2.0
        put_reasons.append("macd_bear_cross")
    elif bar["macd_hist"] < 0:
        put_score += 0.5
        put_reasons.append("macd_negative")

    # 6. STOCHASTIC RSI (weight: 1.5)
    if bar["stoch_oversold"] and bar.get("stoch_cross_up", False):
        call_score += 1.5
        call_reasons.append("stoch_oversold_cross_up")
    elif bar["stoch_oversold"]:
        call_score += 0.5
        call_reasons.append("stoch_oversold")
    if bar["stoch_overbought"] and bar.get("stoch_cross_down", False):
        put_score += 1.5
        put_reasons.append("stoch_overbought_cross_down")
    elif bar["stoch_overbought"]:
        put_score += 0.5
        put_reasons.append("stoch_overbought")

    # 7. BOLLINGER BANDS (weight: 1.5)
    if bar["at_bb_lower"]:
        call_score += 1.5  # Bounce from lower band
        call_reasons.append("bb_lower_touch")
    if bar["at_bb_upper"]:
        put_score += 1.5
        put_reasons.append("bb_upper_touch")

    # 8. ADX TREND STRENGTH (weight: 1.0)
    adx = bar["adx"]
    if adx > 25:  # Strong trend
        if bar["plus_di"] > bar["minus_di"]:
            call_score += 1.0
            call_reasons.append(f"adx_strong_bull_{adx:.0f}")
        else:
            put_score += 1.0
            put_reasons.append(f"adx_strong_bear_{adx:.0f}")

    # 9. S/R LEVEL PROXIMITY (weight: 2.0)
    for sr in sr_levels:
        dist = (spot - sr["level"]) / spot * 100
        if -0.15 <= dist <= 0.15:  # Within 0.15% of S/R level
            if dist > 0 and bar["close"] > bar["open"]:  # Bouncing up from support
                call_score += sr["strength"] * 0.5
                call_reasons.append(f"sr_bounce_up_{sr['type']}")
            elif dist < 0 and bar["close"] < bar["open"]:  # Rejecting at resistance
                put_score += sr["strength"] * 0.5
                put_reasons.append(f"sr_reject_down_{sr['type']}")

    # 10. DAILY TREND CONTEXT (weight: 1.5)
    if daily_trend == "bullish":
        call_score += 1.5
        call_reasons.append("daily_trend_bull")
    elif daily_trend == "bearish":
        put_score += 1.5
        put_reasons.append("daily_trend_bear")

    # 11. VIX REGIME (weight: 1.0)
    if vix < 12:
        call_score += 1.0
        call_reasons.append(f"low_vix_{vix:.1f}")
    elif vix > 20:
        put_score += 1.0
        put_reasons.append(f"high_vix_{vix:.1f}")

    # Normalize to 0-1
    max_possible = 20.0  # Approximate max score
    call_conf = min(1.0, call_score / max_possible)
    put_conf = min(1.0, put_score / max_possible)

    if call_score > put_score and call_score >= 4.0:  # Minimum 4 points for entry
        return "BUY_CALL", call_conf, call_reasons, call_score, put_score
    elif put_score > call_score and put_score >= 4.0:
        return "BUY_PUT", put_conf, put_reasons, call_score, put_score
    else:
        return None, 0, [], call_score, put_score


def simulate_day_learning(day_bars_df, date, prev_day_ohlc, vix, daily_trend,
                          dte, is_expiry):
    """Simulate one day: find entries from indicators, track exits on 1-min bars.

    Returns list of trade dicts with full indicator snapshot.
    """
    n_bars = len(day_bars_df)
    if n_bars < 60:
        return []

    # Compute pivot points
    if prev_day_ohlc:
        pivots = compute_pivot_points(
            prev_day_ohlc["high"], prev_day_ohlc["low"], prev_day_ohlc["close"])
    else:
        spot = day_bars_df["close"].iloc[0]
        pivots = compute_pivot_points(spot * 1.005, spot * 0.995, spot)

    sr_levels = find_support_resistance(day_bars_df, prev_day_ohlc, pivots)

    open_trades = []
    closed_trades = []
    total_day_trades = 0
    last_exit_minute = -30  # Cooldown in minutes
    COOLDOWN_MINUTES = 15
    MAX_TRADES = 5
    MAX_CONCURRENT = 2

    # Trail stop: 0.6% (V7 level — proven sweet spot)
    TRAIL_PCT = 0.006
    # Max hold: 270 min (~4.5 hours)
    MAX_HOLD_MINUTES = 270
    # Expiry close by 2:45 PM (330 min from 9:15)
    EXPIRY_CLOSE_MINUTE = 330
    # No new entries after 2:30 PM (315 min from 9:15)
    NO_ENTRY_AFTER_MINUTE = 315
    # No entries in first 5 minutes (gap noise)
    NO_ENTRY_BEFORE_MINUTE = 5

    day_open = day_bars_df["close"].iloc[0]

    for minute_idx in range(n_bars):
        bar = day_bars_df.iloc[minute_idx]
        spot = bar["close"]
        bar_dte = max(0.05, dte - minute_idx / 1440)

        # ====== 1. CHECK EXITS EVERY MINUTE ======
        trades_to_close = []
        for ti, trade in enumerate(open_trades):
            entry_minute = trade["entry_minute"]
            minutes_held = minute_idx - entry_minute
            if minutes_held < 1:
                continue

            trail_dist = trade["entry_spot"] * TRAIL_PCT
            exit_reason = None

            # Expiry close
            if is_expiry and minute_idx >= EXPIRY_CLOSE_MINUTE:
                exit_reason = "expiry_close"

            # Max hold time exit
            elif minutes_held >= MAX_HOLD_MINUTES:
                exit_reason = "time_exit"

            # Trail stop (after 45 min minimum hold)
            elif minutes_held >= 45:
                if trade["action"] == "BUY_PUT":
                    if spot > trade["best_fav"] + trail_dist:
                        exit_reason = "trail_stop"
                elif trade["action"] == "BUY_CALL":
                    if spot < trade["best_fav"] - trail_dist:
                        exit_reason = "trail_stop"

            # Indicator-based exit: Supertrend flip
            elif minutes_held >= 30:
                if trade["action"] == "BUY_CALL" and bar["st_direction"] == -1:
                    # Check if supertrend just flipped
                    if minute_idx > 0 and day_bars_df.iloc[minute_idx - 1]["st_direction"] == 1:
                        exit_reason = "supertrend_flip"
                elif trade["action"] == "BUY_PUT" and bar["st_direction"] == 1:
                    if minute_idx > 0 and day_bars_df.iloc[minute_idx - 1]["st_direction"] == -1:
                        exit_reason = "supertrend_flip"

            if exit_reason:
                exit_prem = bs_premium(spot, trade["strike"], bar_dte, vix, trade["opt_type"])
                pnl = (exit_prem - trade["entry_prem"]) * trade["qty"] - 80
                trade["exit_minute"] = minute_idx
                trade["exit_time"] = str(bar.name)
                trade["exit_spot"] = round(spot, 2)
                trade["exit_prem"] = round(exit_prem, 2)
                trade["exit_reason"] = exit_reason
                trade["pnl"] = round(pnl, 0)
                trade["minutes_held"] = minutes_held
                # Snapshot exit indicators
                trade["exit_rsi"] = round(bar["rsi"], 1) if pd.notna(bar["rsi"]) else None
                trade["exit_macd_hist"] = round(bar["macd_hist"], 2) if pd.notna(bar["macd_hist"]) else None
                trade["exit_st_dir"] = int(bar["st_direction"]) if pd.notna(bar["st_direction"]) else None
                trade["max_favorable_move"] = round(
                    (trade["entry_spot"] - trade["best_fav"]) / trade["entry_spot"] * 100
                    if trade["action"] == "BUY_PUT" else
                    (trade["best_fav"] - trade["entry_spot"]) / trade["entry_spot"] * 100, 3)
                trades_to_close.append(ti)
                last_exit_minute = minute_idx

        for ti in reversed(trades_to_close):
            closed_trades.append(open_trades.pop(ti))

        # ====== 2. UPDATE TRACKING EVERY MINUTE ======
        for trade in open_trades:
            if trade["action"] == "BUY_CALL" and spot > trade["best_fav"]:
                trade["best_fav"] = spot
            elif trade["action"] == "BUY_PUT" and spot < trade["best_fav"]:
                trade["best_fav"] = spot

        # ====== 3. CHECK ENTRIES EVERY 5 MINUTES ======
        if minute_idx % 5 != 0:
            continue
        if minute_idx < NO_ENTRY_BEFORE_MINUTE:
            continue
        if minute_idx > NO_ENTRY_AFTER_MINUTE:
            continue
        if len(open_trades) >= MAX_CONCURRENT:
            continue
        if total_day_trades >= MAX_TRADES:
            continue
        if minute_idx - last_exit_minute < COOLDOWN_MINUTES:
            continue

        # Compute confluence
        direction, conf, reasons, call_score, put_score = compute_confluence_score(
            bar, day_bars_df, minute_idx, sr_levels, vix, daily_trend)

        if direction is None:
            continue

        # Don't take same direction as existing trade
        same_dir = [t for t in open_trades if t["action"] == direction]
        if same_dir:
            continue

        # ENTRY
        is_zero_hero = False
        strike, opt_type = get_strike_and_type(direction, spot, vix, is_zero_hero)

        # SPAN margin sizing
        SPAN = {12: 30000, 15: 35000, 20: 40000, 25: 50000, 30: 60000}
        span_per_lot = 40000
        for threshold in sorted(SPAN.keys()):
            if vix <= threshold:
                span_per_lot = SPAN[threshold]
                break
        else:
            span_per_lot = 60000
        available = CAPITAL * 0.70
        num_lots = min(2, max(1, int(available / span_per_lot)))
        qty = num_lots * LOT_SIZE

        entry_prem = bs_premium(spot, strike, bar_dte, vix, opt_type)

        trade = {
            "date": str(date),
            "action": direction,
            "entry_minute": minute_idx,
            "entry_time": str(bar.name),
            "entry_spot": round(spot, 2),
            "entry_prem": round(entry_prem, 2),
            "strike": int(strike),
            "opt_type": opt_type,
            "lots": num_lots,
            "qty": qty,
            "vix": round(vix, 1),
            "is_expiry": is_expiry,
            "dte": round(bar_dte, 2),
            "confidence": round(conf, 3),
            "call_score": round(call_score, 1),
            "put_score": round(put_score, 1),
            "reasons": reasons,
            "best_fav": spot,
            # Indicator snapshot at entry
            "entry_rsi": round(bar["rsi"], 1) if pd.notna(bar["rsi"]) else None,
            "entry_ema9": round(bar["ema9"], 1) if pd.notna(bar["ema9"]) else None,
            "entry_ema21": round(bar["ema21"], 1) if pd.notna(bar["ema21"]) else None,
            "entry_st_dir": int(bar["st_direction"]) if pd.notna(bar["st_direction"]) else None,
            "entry_macd_hist": round(bar["macd_hist"], 2) if pd.notna(bar["macd_hist"]) else None,
            "entry_adx": round(bar["adx"], 1) if pd.notna(bar["adx"]) else None,
            "entry_bb_width": round(bar["bb_width"], 3) if pd.notna(bar["bb_width"]) else None,
            "entry_stoch_k": round(bar["stoch_k"], 1) if pd.notna(bar["stoch_k"]) else None,
            "entry_atr_pct": round(bar["atr_pct"], 4) if pd.notna(bar["atr_pct"]) else None,
            "above_ema9": bool(bar["above_ema9"]),
            "above_ema21": bool(bar["above_ema21"]),
            "above_ema50": bool(bar["above_ema50"]),
            "ema9_above_21": bool(bar["ema9_above_ema21"]),
            "spot_vs_pivot": round((spot - pivots["PP"]) / spot * 100, 3),
            # Placeholder
            "exit_minute": -1, "exit_time": "", "exit_spot": 0,
            "exit_prem": 0, "exit_reason": "", "pnl": 0,
            "minutes_held": 0, "max_favorable_move": 0,
        }
        open_trades.append(trade)
        total_day_trades += 1

    # ====== 4. FORCE CLOSE AT EOD ======
    day_close = day_bars_df["close"].iloc[-1]
    for trade in open_trades:
        exit_prem = bs_premium(day_close, trade["strike"],
                               max(0.05, dte - n_bars / 1440), vix, trade["opt_type"])
        pnl = (exit_prem - trade["entry_prem"]) * trade["qty"] - 80
        minutes_held = n_bars - 1 - trade["entry_minute"]
        trade["exit_minute"] = n_bars - 1
        trade["exit_time"] = str(day_bars_df.index[-1])
        trade["exit_spot"] = round(day_close, 2)
        trade["exit_prem"] = round(exit_prem, 2)
        trade["exit_reason"] = "eod_close"
        trade["pnl"] = round(pnl, 0)
        trade["minutes_held"] = minutes_held
        trade["exit_rsi"] = None
        trade["exit_macd_hist"] = None
        trade["exit_st_dir"] = None
        trade["max_favorable_move"] = round(
            (trade["entry_spot"] - trade["best_fav"]) / trade["entry_spot"] * 100
            if trade["action"] == "BUY_PUT" else
            (trade["best_fav"] - trade["entry_spot"]) / trade["entry_spot"] * 100, 3)
        closed_trades.append(trade)

    return closed_trades


# =====================================================================
# PATTERN LEARNING
# =====================================================================

def analyze_trades(all_trades):
    """Analyze all trades to find what separates winners from losers."""
    if not all_trades:
        return {}

    winners = [t for t in all_trades if t["pnl"] > 0]
    losers = [t for t in all_trades if t["pnl"] <= 0]

    analysis = {
        "total_trades": len(all_trades),
        "total_pnl": sum(t["pnl"] for t in all_trades),
        "winners": len(winners),
        "losers": len(losers),
        "win_rate": round(len(winners) / len(all_trades) * 100, 1),
        "avg_win": round(np.mean([t["pnl"] for t in winners])) if winners else 0,
        "avg_loss": round(np.mean([t["pnl"] for t in losers])) if losers else 0,
    }

    # ---- INDICATOR ANALYSIS: What values predict winners? ----
    def safe_mean(trades, key):
        vals = [t[key] for t in trades if t.get(key) is not None and not (isinstance(t[key], float) and np.isnan(t[key]))]
        return round(np.mean(vals), 2) if vals else None

    indicator_keys = [
        "entry_rsi", "entry_adx", "entry_bb_width", "entry_stoch_k",
        "entry_atr_pct", "confidence", "call_score", "put_score",
        "minutes_held", "max_favorable_move",
    ]

    analysis["indicator_comparison"] = {}
    for key in indicator_keys:
        w_val = safe_mean(winners, key)
        l_val = safe_mean(losers, key)
        analysis["indicator_comparison"][key] = {
            "winners_avg": w_val,
            "losers_avg": l_val,
            "diff": round(w_val - l_val, 3) if w_val is not None and l_val is not None else None
        }

    # ---- EXIT REASON ANALYSIS ----
    exit_analysis = defaultdict(lambda: {"count": 0, "pnl": 0, "wins": 0})
    for t in all_trades:
        er = t["exit_reason"]
        exit_analysis[er]["count"] += 1
        exit_analysis[er]["pnl"] += t["pnl"]
        if t["pnl"] > 0:
            exit_analysis[er]["wins"] += 1
    analysis["exit_reasons"] = dict(exit_analysis)

    # ---- ENTRY REASON FREQUENCY IN WINNERS vs LOSERS ----
    reason_in_winners = defaultdict(int)
    reason_in_losers = defaultdict(int)
    for t in winners:
        for r in t.get("reasons", []):
            # Extract base reason (remove numeric suffixes)
            base = r.split("_")
            base_reason = "_".join(base[:3]) if len(base) > 3 else r
            reason_in_winners[base_reason] += 1
    for t in losers:
        for r in t.get("reasons", []):
            base = r.split("_")
            base_reason = "_".join(base[:3]) if len(base) > 3 else r
            reason_in_losers[base_reason] += 1

    all_reasons = set(list(reason_in_winners.keys()) + list(reason_in_losers.keys()))
    reason_analysis = {}
    for r in all_reasons:
        w_count = reason_in_winners.get(r, 0)
        l_count = reason_in_losers.get(r, 0)
        total = w_count + l_count
        wr = w_count / total * 100 if total > 0 else 0
        reason_analysis[r] = {
            "in_winners": w_count,
            "in_losers": l_count,
            "win_rate": round(wr, 1),
            "total": total,
        }
    # Sort by win rate
    analysis["reason_win_rates"] = dict(sorted(
        reason_analysis.items(), key=lambda x: x[1]["win_rate"], reverse=True))

    # ---- SUPERTREND DIRECTION AT ENTRY ----
    st_up_trades = [t for t in all_trades if t.get("entry_st_dir") == 1]
    st_down_trades = [t for t in all_trades if t.get("entry_st_dir") == -1]
    analysis["supertrend_stats"] = {
        "up_trend": {
            "count": len(st_up_trades),
            "pnl": sum(t["pnl"] for t in st_up_trades),
            "win_rate": round(len([t for t in st_up_trades if t["pnl"] > 0]) / max(len(st_up_trades), 1) * 100, 1),
        },
        "down_trend": {
            "count": len(st_down_trades),
            "pnl": sum(t["pnl"] for t in st_down_trades),
            "win_rate": round(len([t for t in st_down_trades if t["pnl"] > 0]) / max(len(st_down_trades), 1) * 100, 1),
        },
    }

    # ---- ACTION BREAKDOWN ----
    calls = [t for t in all_trades if t["action"] == "BUY_CALL"]
    puts = [t for t in all_trades if t["action"] == "BUY_PUT"]
    analysis["action_stats"] = {
        "BUY_CALL": {
            "count": len(calls),
            "pnl": sum(t["pnl"] for t in calls),
            "win_rate": round(len([t for t in calls if t["pnl"] > 0]) / max(len(calls), 1) * 100, 1),
        },
        "BUY_PUT": {
            "count": len(puts),
            "pnl": sum(t["pnl"] for t in puts),
            "win_rate": round(len([t for t in puts if t["pnl"] > 0]) / max(len(puts), 1) * 100, 1),
        },
    }

    # ---- TIME-OF-DAY ANALYSIS ----
    hour_stats = defaultdict(lambda: {"count": 0, "pnl": 0, "wins": 0})
    for t in all_trades:
        # entry_minute: 0-374, 0=9:15AM
        entry_min = t["entry_minute"]
        hour = 9 + (entry_min + 15) // 60
        hour_key = f"{hour}:00"
        hour_stats[hour_key]["count"] += 1
        hour_stats[hour_key]["pnl"] += t["pnl"]
        if t["pnl"] > 0:
            hour_stats[hour_key]["wins"] += 1
    analysis["time_of_day"] = dict(sorted(hour_stats.items()))

    # ---- ADX THRESHOLD ANALYSIS ----
    high_adx = [t for t in all_trades if t.get("entry_adx", 0) and t["entry_adx"] > 25]
    low_adx = [t for t in all_trades if t.get("entry_adx", 0) and t["entry_adx"] <= 25]
    analysis["adx_threshold"] = {
        "high_adx_gt25": {
            "count": len(high_adx),
            "pnl": sum(t["pnl"] for t in high_adx),
            "win_rate": round(len([t for t in high_adx if t["pnl"] > 0]) / max(len(high_adx), 1) * 100, 1),
        },
        "low_adx_le25": {
            "count": len(low_adx),
            "pnl": sum(t["pnl"] for t in low_adx),
            "win_rate": round(len([t for t in low_adx if t["pnl"] > 0]) / max(len(low_adx), 1) * 100, 1),
        },
    }

    # ---- CONFIDENCE THRESHOLD ANALYSIS ----
    for threshold in [0.20, 0.25, 0.30, 0.35, 0.40]:
        above = [t for t in all_trades if t["confidence"] >= threshold]
        if above:
            wr = len([t for t in above if t["pnl"] > 0]) / len(above) * 100
            total_pnl = sum(t["pnl"] for t in above)
            analysis[f"conf_ge_{threshold}"] = {
                "count": len(above), "pnl": total_pnl, "win_rate": round(wr, 1)
            }

    return analysis


# =====================================================================
# MAIN
# =====================================================================

if __name__ == "__main__":
    print("=" * 120)
    print("  DAY-BY-DAY ANALYSIS ON REAL 1-MIN DATA")
    print("  Indicators: EMA, RSI, Supertrend, MACD, Bollinger, ADX, StochRSI, Pivots")
    print("  Data: 38,685 real 1-min bars (Oct 2025 - Apr 2026)")
    print("=" * 120)

    # Load data
    data_dir = project_root / "data" / "historical"
    nifty_1min = pd.read_csv(
        data_dir / "nifty_min_2025-10-01_2026-04-06.csv",
        parse_dates=["timestamp"], index_col="timestamp")
    print(f"Loaded {len(nifty_1min)} bars")

    vix_df = pd.read_csv(
        data_dir / "vix_min_2025-10-01_2026-04-06.csv",
        parse_dates=["timestamp"], index_col="timestamp")

    # Build VIX lookup
    vix_lookup = {}
    for idx, row in vix_df.iterrows():
        vix_lookup[idx.date()] = row["close"]

    # Get daily data for trend
    daily = nifty_1min.resample("D").agg({
        "open": "first", "high": "max", "low": "min", "close": "last"
    }).dropna()
    daily["sma20"] = daily["close"].rolling(20, min_periods=1).mean()
    daily["sma50"] = daily["close"].rolling(50, min_periods=1).mean()
    daily["ema9"] = compute_ema(daily["close"], 9)
    daily["ema21"] = compute_ema(daily["close"], 21)

    # Group by date
    day_groups = {}
    for date, group in nifty_1min.groupby(nifty_1min.index.date):
        day_groups[date] = group

    trading_dates = sorted(day_groups.keys())
    print(f"Trading days: {len(trading_dates)}")

    # Compute indicators for ALL 1-min bars (whole series at once for continuity)
    print("Computing indicators on full 1-min series...", flush=True)
    nifty_1min = add_all_indicators(nifty_1min)
    print("  Done.")

    # Re-group after adding indicators
    day_groups = {}
    for date, group in nifty_1min.groupby(nifty_1min.index.date):
        day_groups[date] = group

    # Expiry detection
    def is_expiry_day(d):
        import datetime
        if d >= datetime.date(2025, 11, 1):
            return d.strftime("%A") == "Tuesday"
        else:
            return d.strftime("%A") == "Thursday"

    # DTE calculation
    def calc_dte(d):
        import datetime
        dow = d.weekday()
        target = 1 if d >= datetime.date(2025, 11, 1) else 3
        if dow <= target:
            return max(target - dow, 0.5)
        else:
            return max(7 - dow + target, 0.5)

    # ======= DAY-BY-DAY WALK =======
    all_trades = []
    equity = CAPITAL
    equity_curve = [CAPITAL]
    daily_results = []
    peak_equity = CAPITAL
    max_dd = 0

    print("\n--- Day-by-Day Walk-Through ---\n")
    print(f"{'Date':<12} {'DOW':<5} {'Open':>8} {'Close':>8} {'Chg%':>6} {'VIX':>5} "
          f"{'Trades':>6} {'Wins':>5} {'PnL':>10} {'Equity':>12} {'DD%':>6}")
    print("-" * 110)

    for di, date in enumerate(trading_dates):
        day_bars = day_groups[date]
        if len(day_bars) < 60:
            continue

        # Previous day OHLC
        prev_day_ohlc = None
        if di > 0:
            prev_date = trading_dates[di - 1]
            prev_d = day_groups[prev_date]
            prev_day_ohlc = {
                "high": prev_d["high"].max(),
                "low": prev_d["low"].min(),
                "close": prev_d["close"].iloc[-1],
                "open": prev_d["open"].iloc[0],
            }

        # VIX
        vix = vix_lookup.get(date, 14.0)

        # Daily trend
        if date in daily.index:
            d_row = daily.loc[date]
            if d_row["close"] > d_row["sma20"] and d_row["ema9"] > d_row["ema21"]:
                daily_trend = "bullish"
            elif d_row["close"] < d_row["sma20"] and d_row["ema9"] < d_row["ema21"]:
                daily_trend = "bearish"
            else:
                daily_trend = "neutral"
        else:
            daily_trend = "neutral"

        dte = calc_dte(date)
        is_exp = is_expiry_day(date)

        # Simulate
        day_trades = simulate_day_learning(
            day_bars, date, prev_day_ohlc, vix, daily_trend, dte, is_exp)

        day_pnl = sum(t["pnl"] for t in day_trades)
        day_wins = len([t for t in day_trades if t["pnl"] > 0])

        equity += day_pnl
        if equity > peak_equity:
            peak_equity = equity
        dd = (peak_equity - equity) / peak_equity * 100
        if dd > max_dd:
            max_dd = dd
        equity_curve.append(equity)

        # Day summary
        day_open = day_bars["open"].iloc[0]
        day_close = day_bars["close"].iloc[-1]
        day_chg = (day_close - day_open) / day_open * 100

        dow = date.strftime("%a")
        print(f"{date}  {dow:<5} {day_open:>8.0f} {day_close:>8.0f} {day_chg:>+5.2f}% "
              f"{vix:>5.1f} {len(day_trades):>5}t {day_wins:>4}W "
              f"Rs {day_pnl:>+9,.0f} Rs {equity:>11,.0f} {dd:>5.1f}%")

        daily_results.append({
            "date": str(date), "dow": dow,
            "open": round(day_open), "close": round(day_close),
            "change_pct": round(day_chg, 2), "vix": round(vix, 1),
            "trades": len(day_trades), "wins": day_wins,
            "pnl": round(day_pnl), "equity": round(equity),
            "drawdown": round(dd, 2), "trend": daily_trend,
        })

        all_trades.extend(day_trades)

    # ======= FINAL RESULTS =======
    print("\n" + "=" * 120)
    print(f"  FINAL RESULTS")
    print("=" * 120)
    net_pnl = equity - CAPITAL
    total = len(all_trades)
    wins = len([t for t in all_trades if t["pnl"] > 0])
    losses = total - wins
    wr = wins / total * 100 if total else 0
    gw = sum(t["pnl"] for t in all_trades if t["pnl"] > 0)
    gl = abs(sum(t["pnl"] for t in all_trades if t["pnl"] <= 0))
    pf = gw / gl if gl > 0 else float("inf")
    daily_pnls = [r["pnl"] for r in daily_results if r["pnl"] != 0]
    sharpe = 0
    if len(daily_pnls) > 1:
        arr = np.array(daily_pnls)
        if arr.std() > 0:
            sharpe = (arr.mean() / arr.std()) * np.sqrt(252)

    print(f"  Net P&L:        Rs {net_pnl:>+,}")
    print(f"  Return:         {net_pnl/CAPITAL*100:>+.1f}%")
    print(f"  Total Trades:   {total}")
    print(f"  Trades/Day:     {total/len(trading_dates):.1f}")
    print(f"  Win Rate:       {wr:.1f}% ({wins}W / {losses}L)")
    print(f"  Avg Win:        Rs {gw/max(wins,1):>+,.0f}")
    print(f"  Avg Loss:       Rs {-gl/max(losses,1):>+,.0f}")
    print(f"  Profit Factor:  {pf:.2f}")
    print(f"  Sharpe Ratio:   {sharpe:.2f}")
    print(f"  Max Drawdown:   {max_dd:.1f}%")

    # ======= PATTERN ANALYSIS =======
    print("\n" + "=" * 120)
    print("  PATTERN ANALYSIS — What separates WINNERS from LOSERS?")
    print("=" * 120)

    analysis = analyze_trades(all_trades)

    # Indicator comparison
    print("\n  INDICATOR VALUES AT ENTRY (Winners vs Losers):")
    print(f"  {'Indicator':<25} {'Winners Avg':>15} {'Losers Avg':>15} {'Difference':>15} {'Insight'}")
    print("-" * 100)
    for key, vals in analysis.get("indicator_comparison", {}).items():
        w = vals["winners_avg"]
        l = vals["losers_avg"]
        d = vals["diff"]
        if w is not None and l is not None:
            insight = ""
            if key == "entry_rsi" and d is not None:
                if d > 3: insight = "Winners enter at HIGHER RSI"
                elif d < -3: insight = "Winners enter at LOWER RSI"
            elif key == "entry_adx" and d is not None:
                if d > 2: insight = "Winners have STRONGER trend"
                elif d < -2: insight = "Winners prefer WEAKER trend"
            elif key == "confidence" and d is not None:
                if d > 0.02: insight = "Higher confidence = MORE wins"
            elif key == "minutes_held" and d is not None:
                if d > 20: insight = "Winners hold LONGER"
                elif d < -20: insight = "Winners exit FASTER"
            elif key == "max_favorable_move" and d is not None:
                if d > 0.05: insight = "Winners get BIGGER favorable moves"
            print(f"  {key:<25} {w:>15.2f} {l:>15.2f} {d:>+15.3f} {insight}")

    # Action stats
    print("\n  ACTION BREAKDOWN:")
    for action, stats in analysis.get("action_stats", {}).items():
        print(f"    {action}: {stats['count']} trades, Rs {stats['pnl']:>+,}, WR {stats['win_rate']:.1f}%")

    # Exit reasons
    print("\n  EXIT REASONS:")
    for er, stats in sorted(analysis.get("exit_reasons", {}).items(), key=lambda x: x[1]["pnl"], reverse=True):
        wr_val = stats["wins"] / stats["count"] * 100 if stats["count"] > 0 else 0
        print(f"    {er:<20} {stats['count']:>4}t  Rs {stats['pnl']:>+9,}  WR {wr_val:.0f}%")

    # Top indicator signals by win rate
    print("\n  INDICATOR SIGNAL WIN RATES (best to worst):")
    for reason, stats in list(analysis.get("reason_win_rates", {}).items())[:20]:
        if stats["total"] >= 5:  # Only show signals with 5+ occurrences
            print(f"    {reason:<35} {stats['total']:>4} trades  WR {stats['win_rate']:>5.1f}%  "
                  f"(W:{stats['in_winners']} L:{stats['in_losers']})")

    # Supertrend
    print("\n  SUPERTREND DIRECTION AT ENTRY:")
    for dir_name, stats in analysis.get("supertrend_stats", {}).items():
        print(f"    {dir_name}: {stats['count']} trades, Rs {stats['pnl']:>+,}, WR {stats['win_rate']:.1f}%")

    # ADX
    print("\n  ADX THRESHOLD (>25 = strong trend):")
    for name, stats in analysis.get("adx_threshold", {}).items():
        print(f"    {name}: {stats['count']} trades, Rs {stats['pnl']:>+,}, WR {stats['win_rate']:.1f}%")

    # Time of day
    print("\n  TIME OF DAY:")
    for hour, stats in sorted(analysis.get("time_of_day", {}).items()):
        wr_val = stats["wins"] / stats["count"] * 100 if stats["count"] > 0 else 0
        print(f"    {hour}: {stats['count']:>4}t  Rs {stats['pnl']:>+9,}  WR {wr_val:.0f}%")

    # Confidence thresholds
    print("\n  CONFIDENCE THRESHOLD ANALYSIS:")
    for key in sorted([k for k in analysis if k.startswith("conf_ge_")]):
        stats = analysis[key]
        print(f"    {key}: {stats['count']} trades, Rs {stats['pnl']:>+,}, WR {stats['win_rate']:.1f}%")

    print("=" * 120)

    # ======= SAVE EVERYTHING =======
    save_data = {
        "summary": {
            "net_pnl": round(net_pnl),
            "return_pct": round(net_pnl / CAPITAL * 100, 1),
            "total_trades": total,
            "trades_per_day": round(total / len(trading_dates), 1),
            "win_rate": round(wr, 1),
            "sharpe": round(sharpe, 2),
            "profit_factor": round(pf, 2),
            "max_drawdown": round(max_dd, 1),
        },
        "analysis": {k: v for k, v in analysis.items()
                     if k not in ("indicator_comparison",)},
        "indicator_comparison": analysis.get("indicator_comparison", {}),
        "daily_results": daily_results,
    }

    out_path = project_root / "data" / "daywise_analysis_results.json"
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)

    # Save all trades with full details
    trades_path = project_root / "data" / "daywise_all_trades.json"
    # Convert trades for JSON
    trades_for_json = []
    for t in all_trades:
        t_clean = {k: (v if not isinstance(v, (np.integer, np.floating)) else
                       int(v) if isinstance(v, np.integer) else float(v))
                   for k, v in t.items() if k != "reasons"}
        t_clean["reasons"] = t.get("reasons", [])
        trades_for_json.append(t_clean)

    with open(trades_path, "w") as f:
        json.dump(trades_for_json, f, indent=2, default=str)

    print(f"\nResults saved to {out_path}")
    print(f"All trades saved to {trades_path}")
