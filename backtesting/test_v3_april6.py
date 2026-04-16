"""
V3 MODEL REPLAY — April 6, 2026 (Monday)
Simulates the exact V3MultiTradeLiveAgent logic on April 6 minute bars.
Compares with user's actual live loss of Rs -2,500 on Rs 30,000 capital.
"""

import sys
import datetime as dt
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backtesting.paper_trading_real_data import bs_premium, get_strike_and_type

# ─── V3 CONSTANTS (from v3_multi_trade_agent.py) ───
STRIKE_INTERVAL = 50
MAX_TRADES_PER_DAY = 5
MAX_CONCURRENT = 2
COOLDOWN_BARS = 5
MIN_CONFIDENCE = 0.55
TRAIL_PCT = 0.010          # 1.0%
PUT_MAX_HOLD_BARS = 330    # 5.5 hrs
CALL_MAX_HOLD_BARS = 300   # 5 hrs
SR_STOP_BUFFER = 0.004     # 0.4%
NO_ENTRY_BEFORE_BAR = 15   # bar 15 = 9:30 AM (skips first 15 min)
NO_ENTRY_AFTER_BAR = 330   # bar 330 = 14:45 (no entries last 75 min)

TARGET_DATE = dt.date(2026, 4, 6)
USER_CAPITAL = 30_000
LOT_SIZE = 65


@dataclass
class MarketState:
    """Simulated MarketAnalysis object."""
    ema_trend: str = ""        # "BULLISH" or "BEARISH"
    market_bias: str = "neutral"
    rsi: float = 50.0
    vix: float = 15.0
    vix_spike: bool = False
    supertrend_direction: str = ""
    is_expiry_day: bool = False
    vwap: float = 0.0


def compute_market_state(bars_df, bar_idx, vix_val, prev_vix):
    """Compute simulated MarketAnalysis from bar data."""
    ms = MarketState()
    ms.vix = vix_val
    ms.vix_spike = vix_val > prev_vix * 1.15 if prev_vix > 0 else False
    ms.is_expiry_day = (TARGET_DATE.weekday() == 3)  # Thursday

    closes = bars_df["close"].values[:bar_idx + 1]
    if len(closes) < 2:
        return ms

    # EMA trend (9 vs 21 period)
    if len(closes) >= 21:
        ema9 = pd.Series(closes).ewm(span=9).mean().iloc[-1]
        ema21 = pd.Series(closes).ewm(span=21).mean().iloc[-1]
        if ema9 > ema21:
            ms.ema_trend = "BULLISH"
        else:
            ms.ema_trend = "BEARISH"
    elif len(closes) >= 9:
        ema9 = pd.Series(closes).ewm(span=9).mean().iloc[-1]
        sma = np.mean(closes)
        ms.ema_trend = "BULLISH" if ema9 > sma else "BEARISH"
    else:
        ms.ema_trend = "BEARISH" if closes[-1] < closes[0] else "BULLISH"

    # RSI (14 period)
    if len(closes) >= 15:
        delta = np.diff(closes)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.mean(gain[-14:])
        avg_loss = np.mean(loss[-14:]) + 1e-10
        rs = avg_gain / avg_loss
        ms.rsi = 100 - (100 / (1 + rs))
    else:
        # Approximate RSI from price change
        pct_change = (closes[-1] - closes[0]) / closes[0] * 100
        ms.rsi = 50 + pct_change * 5  # rough

    ms.rsi = max(0, min(100, ms.rsi))

    # Market bias from price action
    if len(closes) >= 20:
        sma20 = np.mean(closes[-20:])
        if closes[-1] > sma20 * 1.003:
            ms.market_bias = "bullish"
        elif closes[-1] < sma20 * 0.997:
            ms.market_bias = "bearish"
        else:
            ms.market_bias = "neutral"
    else:
        pct = (closes[-1] - closes[0]) / closes[0] * 100
        if pct > 0.3:
            ms.market_bias = "bullish"
        elif pct < -0.3:
            ms.market_bias = "bearish"
        else:
            ms.market_bias = "neutral"

    # Supertrend (simplified: based on 10-bar EMA trend)
    if len(closes) >= 10:
        ema10 = pd.Series(closes).ewm(span=10).mean().iloc[-1]
        ms.supertrend_direction = "UP" if closes[-1] > ema10 else "DOWN"
    else:
        ms.supertrend_direction = "UP" if closes[-1] > closes[0] else "DOWN"

    # VWAP
    if bar_idx > 0:
        highs = bars_df["high"].values[:bar_idx + 1]
        lows = bars_df["low"].values[:bar_idx + 1]
        tp = (highs + lows + closes) / 3.0
        ms.vwap = np.cumsum(tp)[-1] / (bar_idx + 1)

    return ms


def update_sr(spot, session_spots, sr_rules=None):
    """Multi-method S/R computation (from V3 code)."""
    support_cands = []
    resist_cands = []

    # Round 500 levels (weight 3.0)
    for level in range(int(spot // 500) * 500 - 1500,
                       int(spot // 500) * 500 + 2000, 500):
        if level < spot:
            support_cands.append((level, 3.0))
        elif level > spot:
            resist_cands.append((level, 3.0))

    # Round 100 levels (weight 1.5)
    for level in range(int(spot // 100) * 100 - 500,
                       int(spot // 100) * 100 + 600, 100):
        if level % 500 != 0:
            if level < spot:
                support_cands.append((level, 1.5))
            elif level > spot:
                resist_cands.append((level, 1.5))

    # Swing points from session (weight 1.5)
    if len(session_spots) >= 10:
        window = session_spots[-100:]
        for i in range(1, len(window) - 1):
            if window[i] > window[i-1] and window[i] > window[i+1]:
                if window[i] > spot:
                    resist_cands.append((window[i], 1.5))
            if window[i] < window[i-1] and window[i] < window[i+1]:
                if window[i] < spot:
                    support_cands.append((window[i], 1.5))

    # Pick closest
    if support_cands:
        support_cands.sort(key=lambda x: (spot - x[0], -x[1]))
        support = support_cands[0][0]
    else:
        support = round((spot * 0.99) / 50) * 50

    if resist_cands:
        resist_cands.sort(key=lambda x: (x[0] - spot, -x[1]))
        resistance = resist_cands[0][0]
    else:
        resistance = round((spot * 1.01) / 50) * 50

    return support, resistance


def compute_composite(spot, vix, ms, session_spots, support, resistance):
    """9-rule composite scoring (exact V3 logic)."""
    scores = {"BUY_CALL": 0.0, "BUY_PUT": 0.0}

    # Rule 1: VIX regime
    if vix < 12:
        scores["BUY_CALL"] += 3.0
    elif vix < 17:
        scores["BUY_PUT"] += 3.0
    elif vix < 25:
        scores["BUY_PUT"] += 3.5
    else:
        scores["BUY_PUT"] += 4.0

    # Rule 2: EMA trend
    if ms.ema_trend == "BEARISH":
        scores["BUY_PUT"] += 2.0
    elif ms.ema_trend == "BULLISH":
        scores["BUY_CALL"] += 2.0

    # Rule 3: Market bias
    if ms.market_bias in ("bearish", "strong_bearish"):
        scores["BUY_PUT"] += 1.0
    elif ms.market_bias in ("bullish", "strong_bullish"):
        scores["BUY_CALL"] += 1.0

    # Rule 4: RSI
    if ms.rsi < 30:
        scores["BUY_CALL"] += 1.5
    elif ms.rsi > 70:
        scores["BUY_PUT"] += 1.5

    # Rule 5: Day of week
    dow = TARGET_DATE.strftime("%A")
    dow_map = {"Monday": "BUY_CALL", "Tuesday": "BUY_PUT",
               "Wednesday": "BUY_CALL", "Thursday": "BUY_PUT",
               "Friday": "BUY_PUT"}
    d = dow_map.get(dow)
    if d:
        scores[d] += 0.5

    # Rule 6: VIX spike
    if ms.vix_spike:
        scores["BUY_CALL"] += 2.0

    # Rule 7: S/R proximity
    if support and spot:
        dp = (spot - support) / spot * 100
        if 0 < dp < 1.0:
            scores["BUY_CALL"] += 1.0
        elif dp < 0:
            scores["BUY_PUT"] += 1.0
    if resistance and spot:
        dp = (resistance - spot) / spot * 100
        if 0 < dp < 1.0:
            scores["BUY_PUT"] += 1.0
        elif dp < 0:
            scores["BUY_CALL"] += 1.0

    # Rule 8: Momentum (last 5 bars)
    if len(session_spots) >= 6:
        mom = session_spots[-1] - session_spots[-6]
        if mom < -spot * 0.003:
            scores["BUY_PUT"] += 1.0
        elif mom > spot * 0.003:
            scores["BUY_CALL"] += 1.0

    # Rule 9: Multi-TF alignment (supertrend)
    if ms.supertrend_direction == "DOWN":
        best = max(scores, key=scores.get)
        if best == "BUY_CALL":
            scores["BUY_CALL"] *= 0.5
    elif ms.supertrend_direction == "UP":
        best = max(scores, key=scores.get)
        if best == "BUY_PUT":
            scores["BUY_PUT"] *= 0.5

    return scores


def detect_entries(bar, bar_idx, spot, vix, ms, prev_close, prev_spot,
                   orb_high, orb_low, session_spots, support, resistance):
    """V3 entry detection — all 4 types."""
    signals = []
    above_sma50 = (ms.ema_trend == "BULLISH")

    # 1. GAP ENTRY (bar 0 only)
    if bar_idx == 0 and prev_close > 0:
        gap_pct = (spot - prev_close) / prev_close * 100
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

    # 2. ORB ENTRY (bar 1-2)
    if bar_idx in (1, 2) and orb_high > 0:
        orb_range = orb_high - orb_low
        if orb_range > spot * 0.0015:
            if spot > orb_high:
                conf = min(0.80, 0.55 + (spot - orb_high) / orb_high * 10)
                if above_sma50 or vix < 14:
                    signals.append(("BUY_CALL", "orb_breakout_up", conf, False))
            elif spot < orb_low:
                conf = min(0.80, 0.55 + (orb_low - spot) / orb_low * 10)
                signals.append(("BUY_PUT", "orb_breakout_down", conf, False))

    # 3. S/R BOUNCE (bar 2+)
    bias_val = ms.market_bias
    sr_dist = (resistance - support) if (support and resistance) else 0
    sr_valid = sr_dist >= 150

    if bar_idx >= 2 and prev_spot > 0 and sr_valid:
        if support and abs(spot - support) / spot < 0.003:
            if spot > prev_spot and bias_val in ("bullish", "strong_bullish", "neutral"):
                sr_call_conf = 0.65 if bias_val in ("bullish", "strong_bullish") else 0.55
                signals.append(("BUY_CALL", "sr_bounce_support", sr_call_conf, False))

        if resistance and abs(spot - resistance) / spot < 0.003:
            if spot < prev_spot and bias_val not in ("strong_bullish",):
                sr_put_conf = 0.75 if bias_val in ("bearish", "strong_bearish") else 0.70
                signals.append(("BUY_PUT", "sr_bounce_resistance", sr_put_conf, False))

    # 4. COMPOSITE (windowed)
    if bar_idx > 25:
        put_window = (45 <= bar_idx <= 75) or (120 <= bar_idx <= 150)
        call_window = (60 <= bar_idx <= 120)
    else:
        put_window = (3 <= bar_idx <= 5) or (8 <= bar_idx <= 10)
        call_window = (4 <= bar_idx <= 8)

    if put_window or call_window:
        scores = compute_composite(spot, vix, ms, session_spots, support, resistance)
        best_action = max(scores, key=scores.get)
        total = sum(scores.values())
        conf = scores[best_action] / total if total > 0 else 0

        if conf >= MIN_CONFIDENCE:
            if best_action == "BUY_PUT" and put_window:
                signals.append(("BUY_PUT", "composite", conf, False))
            elif best_action == "BUY_CALL" and call_window and vix < 12 and conf >= 0.75:
                signals.append(("BUY_CALL", "composite", conf, False))

    return signals


def check_exits(positions, spot, bar_idx, support, resistance):
    """V3 exit logic — exact replication."""
    to_close = []

    for pos in positions:
        action = pos["action"]
        entry_bar = pos["entry_bar"]
        bars_held = bar_idx - entry_bar
        best_fav = pos["best_fav"]
        entry_spot = pos["entry_spot"]
        is_zh = pos.get("is_zero_hero", False)

        if bars_held < 1:
            continue

        trail_dist = entry_spot * TRAIL_PCT  # 1.0% trail
        exit_reason = None

        # Track best favorable move
        if action == "BUY_CALL" and spot > best_fav:
            pos["best_fav"] = spot
            best_fav = spot
        elif action == "BUY_PUT" and spot < best_fav:
            pos["best_fav"] = spot
            best_fav = spot

        # Zero-hero exits
        if is_zh:
            zh_trail = entry_spot * 0.008
            if action == "BUY_PUT":
                move = (entry_spot - spot) / entry_spot
            else:
                move = (spot - entry_spot) / entry_spot

            if move >= 0.02:
                exit_reason = "zero_hero_target"
            elif move <= -0.008:
                exit_reason = "zero_hero_stop"
            elif move >= 0.01:
                if action == "BUY_PUT" and spot > best_fav + zh_trail:
                    exit_reason = "zero_hero_trail"
                elif action == "BUY_CALL" and spot < best_fav - zh_trail:
                    exit_reason = "zero_hero_trail"
            elif bars_held >= 150:
                exit_reason = "zero_hero_time"

        # Regular PUT exit
        elif action == "BUY_PUT":
            if bars_held >= 45:
                if spot > best_fav + trail_dist:
                    exit_reason = "trail_pct"
            if bars_held >= PUT_MAX_HOLD_BARS and not exit_reason:
                exit_reason = "time_exit"

        # Regular CALL exit
        elif action == "BUY_CALL":
            call_stop = entry_spot * (1 - SR_STOP_BUFFER)
            if not pos.get("sr_target_hit", False):
                if resistance and spot >= resistance:
                    pos["sr_target_hit"] = True
                    pos["best_fav"] = spot
                if spot < call_stop and bars_held >= 3:
                    exit_reason = "sr_stop"
            else:
                if spot < best_fav - trail_dist:
                    exit_reason = "sr_combo_trail"
            if bars_held >= CALL_MAX_HOLD_BARS and not exit_reason:
                exit_reason = "time_exit"

        if exit_reason:
            to_close.append((pos, exit_reason))

    return to_close


def compute_option_premium(spot, strike, opt_type, minutes_from_open, dte_days, vix):
    """Compute option premium using Black-Scholes.

    dte_days: days to expiry (e.g., 3 for Thursday expiry on Monday)
    vix: India VIX value (e.g., 25.47)
    """
    try:
        # Adjust DTE for intraday decay: subtract fraction of day elapsed
        intraday_fraction = minutes_from_open / 375.0
        effective_dte = max(dte_days - intraday_fraction, 0.02)  # min ~7 min

        # bs_premium -> price_option expects dte in DAYS and VIX as raw value
        premium = bs_premium(spot, strike, effective_dte, vix, opt_type=opt_type)
        return max(premium, 0.5)
    except Exception:
        # Fallback: intrinsic + time value approximation
        if opt_type == "CE":
            intrinsic = max(0, spot - strike)
        else:
            intrinsic = max(0, strike - spot)
        time_val = spot * (vix / 100) * np.sqrt(max(dte_days, 0.1) / 252) * 0.4
        return max(intrinsic + time_val, 0.5)


def get_strike_for_v3(spot, action, vix, is_zero_hero):
    """V3 strike selection logic."""
    atm = round(spot / STRIKE_INTERVAL) * STRIKE_INTERVAL
    opt_type = "CE" if action == "BUY_CALL" else "PE"

    if is_zero_hero:
        strike = atm + 200 if action == "BUY_CALL" else atm - 200
    else:
        if action == "BUY_CALL":
            strike = atm + (-50 if vix < 12 else (100 if vix < 20 else 200))
        else:
            strike = atm - (0 if vix < 12 else 150)

    return int(strike), opt_type


def get_v3_lots(capital, vix, is_zero_hero):
    """V3 position sizing."""
    if is_zero_hero:
        return 1

    SPAN_MARGIN_EST = {12: 30000, 15: 35000, 20: 40000, 25: 50000, 30: 60000}
    span_per_lot = 60000
    for vix_threshold in sorted(SPAN_MARGIN_EST.keys()):
        if vix < vix_threshold:
            span_per_lot = SPAN_MARGIN_EST[vix_threshold]
            break

    available_margin = capital * 0.70
    max_lots = max(1, int(available_margin / span_per_lot))
    return min(2, max_lots)


def main():
    print("=" * 120)
    print(f"  V3 MODEL REPLAY — {TARGET_DATE} ({TARGET_DATE.strftime('%A')})")
    print(f"  Simulating EXACT V3MultiTradeLiveAgent logic on 1-min bars")
    print(f"  User's live loss: Rs -2,500 on Rs {USER_CAPITAL:,} capital")
    print("=" * 120)

    # ─── LOAD DATA ───
    nifty = pd.read_csv(
        project_root / "data" / "historical" / "nifty_min_2026-03-01_2026-04-07.csv",
        parse_dates=["timestamp"], index_col="timestamp",
    )
    vix_df = pd.read_csv(
        project_root / "data" / "historical" / "vix_min_2026-03-01_2026-04-07.csv",
        parse_dates=["timestamp"], index_col="timestamp",
    )

    apr6 = nifty[nifty.index.date == TARGET_DATE].copy()
    if len(apr6) == 0:
        print("ERROR: No data for April 6!")
        return

    # Get previous close (April 2 — last trading day)
    apr2 = nifty[nifty.index.date == dt.date(2026, 4, 2)]
    prev_close = apr2["close"].iloc[-1] if len(apr2) > 0 else 22700.70

    # Get VIX
    vix_apr6 = vix_df[vix_df.index.date == TARGET_DATE]
    if len(vix_apr6) > 0:
        vix_val = float(vix_apr6["close"].iloc[0])
    else:
        vix_apr2 = vix_df[vix_df.index.date == dt.date(2026, 4, 2)]
        vix_val = float(vix_apr2["close"].iloc[0]) if len(vix_apr2) > 0 else 15.0

    # Get prev VIX for spike detection
    vix_apr2_df = vix_df[vix_df.index.date == dt.date(2026, 4, 2)]
    prev_vix = float(vix_apr2_df["close"].iloc[0]) if len(vix_apr2_df) > 0 else vix_val

    # DTE: days to Thursday expiry
    days_to_thu = (3 - TARGET_DATE.weekday()) % 7
    if days_to_thu == 0:
        days_to_thu = 7
    dte = days_to_thu

    bars = apr6.reset_index()
    n_bars = len(bars)

    print(f"\n  DATA SUMMARY:")
    print(f"  Date: {TARGET_DATE} ({TARGET_DATE.strftime('%A')})")
    print(f"  Bars: {n_bars}")
    print(f"  Prev Close (Apr 2): {prev_close:.2f}")
    print(f"  Open: {bars['open'].iloc[0]:.2f}")
    print(f"  High: {bars['high'].max():.2f}")
    print(f"  Low: {bars['low'].min():.2f}")
    print(f"  Close: {bars['close'].iloc[-1]:.2f}")
    print(f"  Day Change: {(bars['close'].iloc[-1] - prev_close):+.2f} pts "
          f"({(bars['close'].iloc[-1] - prev_close) / prev_close * 100:+.2f}%)")
    print(f"  VIX: {vix_val:.2f} (prev: {prev_vix:.2f}, spike: {vix_val > prev_vix * 1.15})")
    print(f"  DTE: {dte} (expiry: {'Yes' if TARGET_DATE.weekday() == 3 else 'No'})")
    gap_pct = (bars['close'].iloc[0] - prev_close) / prev_close * 100
    print(f"  Gap (bar 0 close vs prev close): {gap_pct:+.3f}%")
    gap_open = (bars['open'].iloc[0] - prev_close) / prev_close * 100
    print(f"  Gap (open vs prev close): {gap_open:+.3f}%")

    # Intraday profile
    low_bar = bars["low"].values.argmin()
    high_bar = bars["high"].values.argmax()
    low_h, low_m = 9 + (15 + low_bar) // 60, (15 + low_bar) % 60
    high_h, high_m = 9 + (15 + high_bar) // 60, (15 + high_bar) % 60
    print(f"  Intraday Low: {bars['low'].iloc[low_bar]:.2f} at bar {low_bar} ({low_h:02d}:{low_m:02d})")
    print(f"  Intraday High: {bars['high'].iloc[high_bar]:.2f} at bar {high_bar} ({high_h:02d}:{high_m:02d})")
    print(f"  Pattern: V-RECOVERY (crash to {bars['low'].min():.0f} then rally to {bars['high'].max():.0f})")

    # ─── V3 SIMULATION ───
    print(f"\n{'='*120}")
    print(f"  V3 MODEL SIMULATION")
    print(f"{'='*120}")

    # State variables
    trades_today = 0
    open_positions = []
    last_exit_bar = -10
    orb_high = 0.0
    orb_low = 0.0
    prev_spot = 0.0
    session_spots = []
    vix_history = []
    completed_trades = []
    all_signals_log = []  # Log ALL entry signals for debugging

    for bar_idx in range(n_bars):
        bar_data = bars.iloc[bar_idx]
        spot = float(bar_data["close"])
        session_spots.append(spot)
        vix_history.append(vix_val)

        # 3-bar smoothed VIX
        recent_vix = vix_history[-3:]
        smoothed_vix = sum(recent_vix) / len(recent_vix)

        # VIX guardrails
        if smoothed_vix < 12 or smoothed_vix > 35:
            prev_spot = spot
            continue

        # Compute market state
        ms = compute_market_state(bars, bar_idx, smoothed_vix, prev_vix)

        # Check exits FIRST
        exits = check_exits(open_positions, spot, bar_idx,
                           *update_sr(spot, session_spots))
        for pos, reason in exits:
            exit_spot = spot
            entry_spot = pos["entry_spot"]
            action = pos["action"]
            strike, opt_type = pos["strike"], pos["opt_type"]
            entry_prem = pos["entry_prem"]

            # Compute exit premium
            exit_prem = compute_option_premium(exit_spot, strike, opt_type,
                                               bar_idx, dte, smoothed_vix)

            # Compute P&L (option buyer)
            pnl = (exit_prem - entry_prem) * pos["lots"] * LOT_SIZE
            # If PUT and spot went down from entry → premium went up → profit
            # But BS premium captures this — it depends on spot vs strike

            entry_h = 9 + (15 + pos["entry_bar"]) // 60
            entry_m = (15 + pos["entry_bar"]) % 60
            exit_h = 9 + (15 + bar_idx) // 60
            exit_m = (15 + bar_idx) % 60

            trade_record = {
                "action": action,
                "entry_type": pos["entry_type"],
                "entry_bar": pos["entry_bar"],
                "exit_bar": bar_idx,
                "entry_time": f"{entry_h:02d}:{entry_m:02d}",
                "exit_time": f"{exit_h:02d}:{exit_m:02d}",
                "entry_spot": entry_spot,
                "exit_spot": exit_spot,
                "strike": strike,
                "opt_type": opt_type,
                "entry_prem": entry_prem,
                "exit_prem": exit_prem,
                "lots": pos["lots"],
                "bars_held": bar_idx - pos["entry_bar"],
                "exit_reason": reason,
                "pnl": pnl,
                "is_zero_hero": pos.get("is_zero_hero", False),
            }
            completed_trades.append(trade_record)
            open_positions.remove(pos)
            last_exit_bar = bar_idx

        # Entry guards
        if trades_today >= MAX_TRADES_PER_DAY:
            prev_spot = spot
            continue
        if len(open_positions) >= MAX_CONCURRENT:
            prev_spot = spot
            continue
        if bar_idx - last_exit_bar < COOLDOWN_BARS:
            prev_spot = spot
            continue
        if bar_idx < NO_ENTRY_BEFORE_BAR or bar_idx >= NO_ENTRY_AFTER_BAR:
            prev_spot = spot
            continue

        # ORB tracking
        if bar_idx == 0:
            orb_high = float(bar_data.get("high", spot))
            orb_low = float(bar_data.get("low", spot))
        elif bar_idx == 1:
            orb_high = max(orb_high, float(bar_data.get("high", spot)))
            orb_low = min(orb_low, float(bar_data.get("low", spot)))

        # S/R
        support, resistance = update_sr(spot, session_spots)

        # Detect entries
        entries = detect_entries(
            bar_data, bar_idx, spot, smoothed_vix, ms, prev_close, prev_spot,
            orb_high, orb_low, session_spots, support, resistance
        )

        if entries:
            # Log all detected signals
            for sig in entries:
                bar_h = 9 + (15 + bar_idx) // 60
                bar_m = (15 + bar_idx) % 60
                all_signals_log.append({
                    "bar": bar_idx,
                    "time": f"{bar_h:02d}:{bar_m:02d}",
                    "spot": spot,
                    "action": sig[0],
                    "type": sig[1],
                    "conf": sig[2],
                    "is_zh": sig[3],
                    "passed": False,  # will update if taken
                })

            # Pick best entry
            entries.sort(key=lambda x: x[2], reverse=True)
            action, entry_type, confidence, is_zero_hero = entries[0]

            # Conflict check
            conflict = False
            for pos in open_positions:
                if pos["action"] == action:
                    conflict = True
                    break

            if not conflict and confidence >= MIN_CONFIDENCE:
                # Build trade
                strike, opt_type = get_strike_for_v3(spot, action, smoothed_vix, is_zero_hero)
                num_lots = get_v3_lots(USER_CAPITAL, smoothed_vix, is_zero_hero)

                # Option premium at entry
                entry_prem = compute_option_premium(
                    spot, strike, opt_type, bar_idx, dte, smoothed_vix
                )

                position = {
                    "action": action,
                    "entry_type": entry_type,
                    "entry_bar": bar_idx,
                    "entry_spot": spot,
                    "best_fav": spot,
                    "is_zero_hero": is_zero_hero,
                    "sr_target_hit": False,
                    "strike": strike,
                    "opt_type": opt_type,
                    "entry_prem": entry_prem,
                    "lots": num_lots,
                    "confidence": confidence,
                }
                open_positions.append(position)
                trades_today += 1

                # Mark signal as taken
                for sig_log in all_signals_log:
                    if (sig_log["bar"] == bar_idx and sig_log["action"] == action
                            and sig_log["type"] == entry_type):
                        sig_log["passed"] = True

                bar_h = 9 + (15 + bar_idx) // 60
                bar_m = (15 + bar_idx) % 60
                print(f"  >> ENTRY #{trades_today}: {action} via {entry_type} | "
                      f"{bar_h:02d}:{bar_m:02d} | spot={spot:.0f} | "
                      f"strike={strike} {opt_type} | prem={entry_prem:.1f} | "
                      f"lots={num_lots} | conf={confidence:.3f}"
                      f"{' [ZERO-HERO]' if is_zero_hero else ''}")

        prev_spot = spot

    # Force close any remaining positions at market close
    if open_positions:
        spot = float(bars["close"].iloc[-1])
        for pos in open_positions:
            exit_prem = compute_option_premium(
                spot, pos["strike"], pos["opt_type"], n_bars - 1, dte, smoothed_vix
            )
            pnl = (exit_prem - pos["entry_prem"]) * pos["lots"] * LOT_SIZE

            entry_h = 9 + (15 + pos["entry_bar"]) // 60
            entry_m = (15 + pos["entry_bar"]) % 60

            trade_record = {
                "action": pos["action"],
                "entry_type": pos["entry_type"],
                "entry_bar": pos["entry_bar"],
                "exit_bar": n_bars - 1,
                "entry_time": f"{entry_h:02d}:{entry_m:02d}",
                "exit_time": "15:29",
                "entry_spot": pos["entry_spot"],
                "exit_spot": spot,
                "strike": pos["strike"],
                "opt_type": pos["opt_type"],
                "entry_prem": pos["entry_prem"],
                "exit_prem": exit_prem,
                "lots": pos["lots"],
                "bars_held": n_bars - 1 - pos["entry_bar"],
                "exit_reason": "eod_close",
                "pnl": pnl,
                "is_zero_hero": pos.get("is_zero_hero", False),
            }
            completed_trades.append(trade_record)
            print(f"  >> EOD CLOSE: {pos['action']} | spot={spot:.0f}")

    # ─── RESULTS ───
    print(f"\n{'='*120}")
    print(f"  DETAILED TRADE LOG")
    print(f"{'='*120}")

    if not completed_trades:
        print("\n  >>> ZERO TRADES — No valid V3 signals on this day <<<")
        print(f"\n  ANALYSIS:")
        print(f"  - Gap: {gap_pct:+.3f}% (needs > 0.3% for gap entry)")
        print(f"  - VIX: {vix_val:.2f} (needs 12-35 range, within range)")
        print(f"  - Time filter: bars 0-14 (9:15-9:29) blocked")
        print(f"  - Market crashed to {bars['low'].min():.0f} before rallying to {bars['high'].max():.0f}")

        # Show what signals were detected but filtered
        if all_signals_log:
            print(f"\n  DETECTED BUT FILTERED SIGNALS:")
            for sig in all_signals_log:
                status = "TAKEN" if sig["passed"] else "FILTERED"
                print(f"    bar {sig['bar']:>3} ({sig['time']}) | {sig['action']:>9} via {sig['type']:<25} | "
                      f"conf={sig['conf']:.3f} | spot={sig['spot']:.0f} | {status}")
    else:
        total_pnl = 0
        print(f"\n  {'#':>3} {'Action':>9} {'Entry Time':>11} {'Exit Time':>10} {'Type':<25} "
              f"{'Spot In':>8} {'Spot Out':>8} {'Strike':>7} {'Prem In':>8} {'Prem Out':>8} "
              f"{'Lots':>4} {'Held':>6} {'Exit Reason':<16} {'P&L':>10}")
        print(f"  {'-'*155}")

        for i, t in enumerate(completed_trades):
            zh_tag = " [ZH]" if t["is_zero_hero"] else ""
            print(f"  {i+1:>3} {t['action']:>9} {t['entry_time']:>11}      "
                  f"{t['exit_time']:>10}  {t['entry_type']+zh_tag:<25} "
                  f"{t['entry_spot']:>8.0f} {t['exit_spot']:>8.0f} "
                  f"{t['strike']:>7} "
                  f"{t['entry_prem']:>8.1f} {t['exit_prem']:>8.1f} "
                  f"{t['lots']:>4} {t['bars_held']:>5}m "
                  f"{t['exit_reason']:<16} "
                  f"Rs{t['pnl']:>+9,.0f}")
            total_pnl += t["pnl"]

        wins = sum(1 for t in completed_trades if t["pnl"] > 0)
        losses = len(completed_trades) - wins
        print(f"  {'-'*155}")
        print(f"  TOTAL: {len(completed_trades)} trades ({wins}W/{losses}L) | "
              f"Day P&L: Rs{total_pnl:>+,}")
        print(f"  Capital after: Rs {USER_CAPITAL + total_pnl:,.0f} "
              f"({total_pnl / USER_CAPITAL * 100:+.1f}%)")

    # ─── SIGNAL ANALYSIS ───
    print(f"\n{'='*120}")
    print(f"  ALL DETECTED SIGNALS (including filtered)")
    print(f"{'='*120}")
    if all_signals_log:
        for sig in all_signals_log:
            status = "[TAKEN]  " if sig["passed"] else "[SKIPPED]"
            print(f"    {status} bar {sig['bar']:>3} ({sig['time']}) | "
                  f"{sig['action']:>9} via {sig['type']:<25} | "
                  f"conf={sig['conf']:.3f} | spot={sig['spot']:.0f}"
                  f"{' [ZH]' if sig['is_zh'] else ''}")
    else:
        print("    No entry signals detected at all!")

    # ─── COMPARISON WITH USER'S LIVE RESULTS ───
    print(f"\n{'='*120}")
    print(f"  COMPARISON WITH YOUR LIVE TRADING")
    print(f"{'='*120}")
    user_loss = -2500
    sim_pnl = sum(t["pnl"] for t in completed_trades) if completed_trades else 0
    print(f"  Your live result:     Rs {user_loss:>+,} ({user_loss/USER_CAPITAL*100:+.1f}%)")
    print(f"  V3 simulation result: Rs {sim_pnl:>+,.0f} ({sim_pnl/USER_CAPITAL*100:+.1f}%)")
    if completed_trades:
        print(f"  V3 trades taken:      {len(completed_trades)}")
    else:
        print(f"  V3 trades taken:      0 (model found no valid signals)")

    print(f"\n  MARKET CONTEXT:")
    print(f"  - Monday (V3 trades all days, no Monday filter)")
    print(f"  - VIX: {vix_val:.2f} (high volatility)")
    print(f"  - Classic V-recovery: dropped -237 pts then rallied +455 pts")
    print(f"  - Gap open: {gap_open:+.2f}% -> bearish bar 0 -> massive selloff -> reversal")
    print(f"  - The WORST pattern for PUT holders (entered during crash, then reversed)")

    # ─── BAR-BY-BAR COMPOSITE WINDOW ANALYSIS ───
    print(f"\n{'='*120}")
    print(f"  COMPOSITE WINDOW ANALYSIS (bars 45-75, 60-120, 120-150)")
    print(f"{'='*120}")
    for check_bar in [45, 60, 75, 90, 105, 120, 135, 150]:
        if check_bar >= n_bars:
            continue
        check_spot = float(bars["close"].iloc[check_bar])
        check_ms = compute_market_state(bars, check_bar, vix_val, prev_vix)
        check_sup, check_res = update_sr(check_spot, session_spots[:check_bar+1])
        scores = compute_composite(check_spot, vix_val, check_ms,
                                  session_spots[:check_bar+1], check_sup, check_res)
        best = max(scores, key=scores.get)
        total = sum(scores.values())
        conf = scores[best] / total if total > 0 else 0
        ch = 9 + (15 + check_bar) // 60
        cm = (15 + check_bar) % 60

        put_w = (45 <= check_bar <= 75) or (120 <= check_bar <= 150)
        call_w = (60 <= check_bar <= 120)
        window = []
        if put_w: window.append("PUT_WIN")
        if call_w: window.append("CALL_WIN")

        print(f"  bar {check_bar:>3} ({ch:02d}:{cm:02d}) | spot={check_spot:>8.0f} | "
              f"CALL={scores['BUY_CALL']:.1f} PUT={scores['BUY_PUT']:.1f} | "
              f"best={best} conf={conf:.3f} | RSI={check_ms.rsi:.0f} | "
              f"trend={check_ms.ema_trend} | bias={check_ms.market_bias} | "
              f"windows: {','.join(window) if window else 'NONE'}")

    print(f"\n{'='*120}")


if __name__ == "__main__":
    main()
