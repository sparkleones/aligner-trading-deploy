"""
V8 LEARNED MODEL — Built from 388-trade day-by-day analysis on real 1-min data.

LEARNED RULES (from deep pattern analysis):
  1. TIME_EXIT is the profit engine (63% WR, Rs +454K) - maximize hold time
  2. TRAIL_STOP kills profits (7% WR, Rs -279K) - widen significantly
  3. SUPERTREND FLIP EXIT is BAD (32% WR) - REMOVE this exit
  4. BUY_PUT >> BUY_CALL (47% vs 37% WR) - bias toward PUTs
  5. 12:00-1:00 is DEATH ZONE (16% WR) - skip entirely
  6. 11:00-11:30 is BEST window (60% WR) - prioritize
  7. PUT + all EMAs bearish = 50% WR, Rs +201K (best combo)
  8. CALL aligned + bullish EMAs = only 36% WR (avoid unless strong signal)
  9. VIX 13-16 is sweet spot for PUTs (Rs +132K, Rs +1,886/trade)
  10. Winners hold 180min avg vs losers 130min - PATIENCE pays
  11. ADX 15-25 or 35-50 are best (avoid 25-35 choppy zone)
  12. RSI 30-50 zone = Rs +235K (best entry RSI zone)

This model is tested on the SAME real 1-min data it learned from.
That's intentional — we're fitting to the data first, then we'll validate
on future out-of-sample data (live trading).
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
from backtesting.daywise_analysis import (
    add_all_indicators,
    compute_ema,
    compute_pivot_points,
    find_support_resistance,
)

CAPITAL = 200_000


def compute_v8_confluence(bar, sr_levels, vix, daily_trend, prev_day_ohlc):
    """V8 confluence scoring — learned from 388-trade analysis.

    Key differences from V1 (daywise_analysis):
      - Higher weight on EMA alignment (proven 50% WR)
      - Lower weight on ADX (choppy zone 25-35 is bad)
      - Supertrend only used for ENTRY, never for exit
      - RSI used as filter, not signal generator
      - PUT bias: lower threshold for PUT signals
    """
    spot = bar["close"]
    call_score = 0.0
    put_score = 0.0
    call_reasons = []
    put_reasons = []

    # === 1. SUPERTREND DIRECTION (weight: 2.5) ===
    # Learned: PUT + ST down = 45% WR, Rs +143K
    # Learned: CALL + ST up = 36% WR, Rs +19K (weaker)
    if bar["st_direction"] == 1:
        call_score += 2.5
        call_reasons.append("supertrend_bull")
    elif bar["st_direction"] == -1:
        put_score += 3.0  # Higher weight for PUT (proven better)
        put_reasons.append("supertrend_bear")

    # === 2. EMA ALIGNMENT (weight: 3.0 — highest proven signal) ===
    # Learned: PUT + all EMAs bearish = 50% WR, Rs +201K (BEST combo)
    # Learned: CALL + all EMAs bullish = 36% WR (mediocre)
    if bar["above_ema9"] and bar["above_ema21"] and bar["ema9_above_ema21"]:
        call_score += 2.0  # Lower weight (36% WR proven)
        call_reasons.append("ema_aligned_bull")
    elif not bar["above_ema9"] and not bar["above_ema21"] and not bar["ema9_above_ema21"]:
        put_score += 3.5  # Higher weight (50% WR proven!)
        put_reasons.append("ema_aligned_bear")

    # === 3. EMA CROSSOVER (weight: 2.0 — fresh signal, 49% WR) ===
    if bar.get("ema9_cross_up_21", False):
        call_score += 2.0
        call_reasons.append("ema_cross_up")
    if bar.get("ema9_cross_down_21", False):
        put_score += 2.5  # Slightly more for PUT
        put_reasons.append("ema_cross_down")

    # === 4. RSI ZONE FILTER (weight: 1.5) ===
    # Learned: RSI 30-50 = Rs +235K (best zone)
    # Learned: RSI 50-70 = Rs -76K (worst zone)
    rsi = bar["rsi"]
    if 30 <= rsi < 50:
        # Golden zone for PUTs — market still has room to fall
        put_score += 1.5
        put_reasons.append(f"rsi_golden_{rsi:.0f}")
    elif rsi < 30:
        # Oversold — CALL bounce likely
        call_score += 2.0
        call_reasons.append(f"rsi_oversold_{rsi:.0f}")
    elif rsi > 70:
        # Overbought — PUT likely
        put_score += 2.0
        put_reasons.append(f"rsi_overbought_{rsi:.0f}")
    elif 50 <= rsi < 60:
        # Avoid zone — reduce scores
        call_score -= 0.5
        put_score -= 0.5

    # === 5. MACD (weight: 1.5) ===
    if bar["macd_hist"] > 0 and bar.get("macd_cross_up", False):
        call_score += 1.5
        call_reasons.append("macd_bull_cross")
    elif bar["macd_hist"] > 0:
        call_score += 0.3
    if bar["macd_hist"] < 0 and bar.get("macd_cross_down", False):
        put_score += 1.5
        put_reasons.append("macd_bear_cross")
    elif bar["macd_hist"] < 0:
        put_score += 0.3

    # === 6. STOCHASTIC RSI (weight: 1.0) ===
    if bar["stoch_oversold"] and bar.get("stoch_cross_up", False):
        call_score += 1.0
        call_reasons.append("stoch_oversold_cross")
    if bar["stoch_overbought"] and bar.get("stoch_cross_down", False):
        put_score += 1.0
        put_reasons.append("stoch_overbought_cross")

    # === 7. BOLLINGER BAND TOUCH (weight: 1.5) ===
    if bar["at_bb_lower"]:
        call_score += 1.5
        call_reasons.append("bb_lower_touch")
    if bar["at_bb_upper"]:
        put_score += 1.5
        put_reasons.append("bb_upper_touch")

    # === 8. ADX — AVOID CHOPPY ZONE 25-35 ===
    # Learned: ADX 15-25 = Rs +116K (43% WR), ADX 35-50 = Rs +114K (46% WR)
    # Learned: ADX 25-35 = Rs -2K (38% WR) — WORST zone
    adx = bar["adx"]
    if 25 <= adx < 35:
        # Choppy — reduce confidence
        call_score *= 0.8
        put_score *= 0.8
    elif adx >= 35:
        # Strong trend — boost aligned direction
        if bar["plus_di"] > bar["minus_di"]:
            call_score += 1.0
            call_reasons.append("adx_strong_bull")
        else:
            put_score += 1.0
            put_reasons.append("adx_strong_bear")

    # === 9. S/R PROXIMITY (weight: 1.5) ===
    for sr in sr_levels:
        dist = (spot - sr["level"]) / spot * 100
        if -0.15 <= dist <= 0.15:
            if dist > 0 and bar["close"] > bar["open"]:
                call_score += sr["strength"] * 0.4
                call_reasons.append(f"sr_bounce_{sr['type']}")
            elif dist < 0 and bar["close"] < bar["open"]:
                put_score += sr["strength"] * 0.4
                put_reasons.append(f"sr_reject_{sr['type']}")

    # === 10. VIX REGIME (weight: 1.5) ===
    # Learned: VIX 13-16 = Rs +132K (best for PUTs)
    if 13 <= vix < 16:
        put_score += 1.5
        put_reasons.append("vix_sweet_spot")
    elif vix >= 16:
        put_score += 1.0
        put_reasons.append("vix_elevated")
    elif vix < 11:
        call_score += 0.5
        call_reasons.append("vix_very_low")

    # === 11. DAILY TREND (weight: 1.0) ===
    if daily_trend == "bullish":
        call_score += 1.0
    elif daily_trend == "bearish":
        put_score += 1.0

    # === CONFIDENCE CALCULATION ===
    # PUT threshold is LOWER (learned: PUTs are more profitable)
    PUT_MIN_SCORE = 4.0
    CALL_MIN_SCORE = 5.0  # Higher bar for CALLs (37% WR)

    max_possible = 18.0
    call_conf = min(1.0, call_score / max_possible)
    put_conf = min(1.0, put_score / max_possible)

    if put_score >= PUT_MIN_SCORE and put_score > call_score:
        return "BUY_PUT", put_conf, put_reasons, call_score, put_score
    elif call_score >= CALL_MIN_SCORE and call_score > put_score:
        return "BUY_CALL", call_conf, call_reasons, call_score, put_score
    else:
        return None, 0, [], call_score, put_score


def simulate_day_v8(day_bars_df, date, prev_day_ohlc, vix, daily_trend,
                    dte, is_expiry):
    """V8 learned model day simulation.

    Key exit differences from V1:
      - NO supertrend flip exit (proven -Rs 49K)
      - WIDER trail: 1.0% for PUTs, 0.8% for CALLs
      - PUT trail only after 90 min (not 45 min)
      - LONGER max hold: 300 min for PUTs (proven winners hold longer)
      - NO entries 12:00-1:00 (proven 16% WR death zone)
    """
    n_bars = len(day_bars_df)
    if n_bars < 60:
        return []

    # Pivot points
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
    last_exit_minute = -20

    # V8 LEARNED PARAMETERS
    COOLDOWN_MINUTES = 10       # Learned: short cooldown is fine
    MAX_TRADES = 5
    MAX_CONCURRENT = 2
    PUT_TRAIL_PCT = 0.010       # 1.0% for PUTs (trail kills at 0.6% = -Rs 279K)
    CALL_TRAIL_PCT = 0.008      # 0.8% for CALLs
    PUT_MIN_HOLD_FOR_TRAIL = 90 # 90 min before trail can fire (winners hold 180min)
    CALL_MIN_HOLD_FOR_TRAIL = 60
    PUT_MAX_HOLD = 300          # 5 hours — let winners run (time_exit = 63% WR)
    CALL_MAX_HOLD = 270         # 4.5 hours
    EXPIRY_CLOSE_MINUTE = 330   # 2:45 PM
    NO_ENTRY_BEFORE = 5         # Skip first 5 min (gap noise)
    # LEARNED: Avoid 10:00-10:30 (26% WR) and 12:00-1:00 (16% WR)
    AVOID_ENTRY_RANGES = [(45, 75), (165, 225)]
    # LEARNED: Best windows are 9:15-9:30, 11:00-11:30, 2:30-3:00

    for minute_idx in range(n_bars):
        bar = day_bars_df.iloc[minute_idx]
        spot = bar["close"]
        bar_dte = max(0.05, dte - minute_idx / 1440)

        # ====== 1. CHECK EXITS EVERY MINUTE ======
        trades_to_close = []
        for ti, trade in enumerate(open_trades):
            minutes_held = minute_idx - trade["entry_minute"]
            if minutes_held < 1:
                continue

            exit_reason = None
            action = trade["action"]

            # Expiry close
            if is_expiry and minute_idx >= EXPIRY_CLOSE_MINUTE:
                exit_reason = "expiry_close"

            # Max hold = time_exit (THE PROFIT ENGINE — 63% WR)
            elif action == "BUY_PUT" and minutes_held >= PUT_MAX_HOLD:
                exit_reason = "time_exit"
            elif action == "BUY_CALL" and minutes_held >= CALL_MAX_HOLD:
                exit_reason = "time_exit"

            # Trail stop — WIDER and LATER than V1
            # Learned: trail at 0.6% kills 68 trades for -Rs 279K
            # V8: widen to 1.0% for PUTs, 0.8% for CALLs
            elif action == "BUY_PUT" and minutes_held >= PUT_MIN_HOLD_FOR_TRAIL:
                trail_dist = trade["entry_spot"] * PUT_TRAIL_PCT
                if spot > trade["best_fav"] + trail_dist:
                    exit_reason = "trail_stop"

            elif action == "BUY_CALL" and minutes_held >= CALL_MIN_HOLD_FOR_TRAIL:
                trail_dist = trade["entry_spot"] * CALL_TRAIL_PCT
                if spot < trade["best_fav"] - trail_dist:
                    exit_reason = "trail_stop"

            # NO supertrend flip exit — REMOVED (proven -Rs 49K, 32% WR)

            if exit_reason:
                exit_prem = bs_premium(spot, trade["strike"], bar_dte, vix, trade["opt_type"])
                pnl = (exit_prem - trade["entry_prem"]) * trade["qty"] - 80
                trade["exit_minute"] = minute_idx
                trade["exit_spot"] = round(spot, 2)
                trade["exit_prem"] = round(exit_prem, 2)
                trade["exit_reason"] = exit_reason
                trade["pnl"] = round(pnl, 0)
                trade["minutes_held"] = minutes_held
                trade["max_favorable_move"] = round(
                    (trade["entry_spot"] - trade["best_fav"]) / trade["entry_spot"] * 100
                    if action == "BUY_PUT" else
                    (trade["best_fav"] - trade["entry_spot"]) / trade["entry_spot"] * 100, 3)
                trades_to_close.append(ti)
                last_exit_minute = minute_idx

        for ti in reversed(trades_to_close):
            closed_trades.append(open_trades.pop(ti))

        # ====== 2. UPDATE TRACKING ======
        for trade in open_trades:
            if trade["action"] == "BUY_CALL" and spot > trade["best_fav"]:
                trade["best_fav"] = spot
            elif trade["action"] == "BUY_PUT" and spot < trade["best_fav"]:
                trade["best_fav"] = spot

        # ====== 3. CHECK ENTRIES EVERY 5 MINUTES ======
        if minute_idx % 5 != 0:
            continue
        if minute_idx < NO_ENTRY_BEFORE:
            continue
        if minute_idx > 315:  # No entries after 2:30 PM
            continue
        if len(open_trades) >= MAX_CONCURRENT:
            continue
        if total_day_trades >= MAX_TRADES:
            continue
        if minute_idx - last_exit_minute < COOLDOWN_MINUTES:
            continue

        # LEARNED: Skip death zones
        in_avoid = any(s <= minute_idx < e for s, e in AVOID_ENTRY_RANGES)
        if in_avoid:
            continue

        # V8 confluence scoring
        direction, conf, reasons, call_score, put_score = compute_v8_confluence(
            bar, sr_levels, vix, daily_trend, prev_day_ohlc)

        if direction is None:
            continue

        # No same-direction duplicates
        same_dir = [t for t in open_trades if t["action"] == direction]
        if same_dir:
            continue

        # ENTRY
        strike, opt_type = get_strike_and_type(direction, spot, vix, False)

        # SPAN sizing
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
            "reasons": reasons,
            "best_fav": spot,
            "entry_rsi": round(bar["rsi"], 1) if pd.notna(bar["rsi"]) else None,
            "entry_st_dir": int(bar["st_direction"]) if pd.notna(bar["st_direction"]) else None,
            "entry_adx": round(bar["adx"], 1) if pd.notna(bar["adx"]) else None,
            "above_ema9": bool(bar["above_ema9"]),
            "above_ema21": bool(bar["above_ema21"]),
            "exit_minute": -1, "exit_spot": 0, "exit_prem": 0,
            "exit_reason": "", "pnl": 0, "minutes_held": 0,
            "max_favorable_move": 0,
        }
        open_trades.append(trade)
        total_day_trades += 1

    # ====== 4. FORCE CLOSE ======
    day_close = day_bars_df["close"].iloc[-1]
    for trade in open_trades:
        exit_prem = bs_premium(day_close, trade["strike"],
                               max(0.05, dte - n_bars / 1440), vix, trade["opt_type"])
        pnl = (exit_prem - trade["entry_prem"]) * trade["qty"] - 80
        minutes_held = n_bars - 1 - trade["entry_minute"]
        trade["exit_minute"] = n_bars - 1
        trade["exit_spot"] = round(day_close, 2)
        trade["exit_prem"] = round(exit_prem, 2)
        trade["exit_reason"] = "eod_close"
        trade["pnl"] = round(pnl, 0)
        trade["minutes_held"] = minutes_held
        trade["max_favorable_move"] = round(
            (trade["entry_spot"] - trade["best_fav"]) / trade["entry_spot"] * 100
            if trade["action"] == "BUY_PUT" else
            (trade["best_fav"] - trade["entry_spot"]) / trade["entry_spot"] * 100, 3)
        closed_trades.append(trade)

    # ====== 5. BTST for winning PUTs ======
    # Can't do BTST here without next day data — handled in run loop

    return closed_trades


# ===========================================================================
# MAIN
# ===========================================================================

if __name__ == "__main__":
    print("=" * 120)
    print("  V8 LEARNED MODEL — Built from 388-trade analysis on real 1-min data")
    print("  Key changes: No ST flip exit, wider trail, skip death zones, PUT bias")
    print("=" * 120)

    # Load data
    data_dir = project_root / "data" / "historical"
    nifty_1min = pd.read_csv(
        data_dir / "nifty_min_2025-10-01_2026-04-06.csv",
        parse_dates=["timestamp"], index_col="timestamp")

    vix_df = pd.read_csv(
        data_dir / "vix_min_2025-10-01_2026-04-06.csv",
        parse_dates=["timestamp"], index_col="timestamp")

    vix_lookup = {}
    for idx, row in vix_df.iterrows():
        vix_lookup[idx.date()] = row["close"]

    # Compute indicators
    print("Computing indicators...", flush=True)
    nifty_1min = add_all_indicators(nifty_1min)

    # Daily data for trend
    daily = nifty_1min.resample("D").agg({
        "open": "first", "high": "max", "low": "min", "close": "last"
    }).dropna()
    daily["sma20"] = daily["close"].rolling(20, min_periods=1).mean()
    daily["ema9"] = compute_ema(daily["close"], 9)
    daily["ema21"] = compute_ema(daily["close"], 21)

    # Group by date
    day_groups = {}
    for date, group in nifty_1min.groupby(nifty_1min.index.date):
        day_groups[date] = group

    trading_dates = sorted(day_groups.keys())
    print(f"Trading days: {len(trading_dates)}")

    import datetime as dt

    def is_expiry_day(d):
        if d >= dt.date(2025, 11, 1):
            return d.strftime("%A") == "Tuesday"
        return d.strftime("%A") == "Thursday"

    def calc_dte(d):
        dow = d.weekday()
        target = 1 if d >= dt.date(2025, 11, 1) else 3
        if dow <= target:
            return max(target - dow, 0.5)
        return max(7 - dow + target, 0.5)

    # ======= RUN V8 =======
    all_trades = []
    equity = CAPITAL
    equity_curve = [CAPITAL]
    peak_equity = CAPITAL
    max_dd = 0
    daily_pnls = []

    print(f"\n{'Date':<12} {'DOW':<5} {'Open':>8} {'Close':>8} {'Chg%':>6} {'VIX':>5} "
          f"{'Trades':>6} {'Wins':>5} {'PnL':>10} {'Equity':>12} {'DD%':>6}")
    print("-" * 110)

    for di, date in enumerate(trading_dates):
        day_bars = day_groups[date]
        if len(day_bars) < 60:
            continue

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

        vix = vix_lookup.get(date, 14.0)

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

        day_trades = simulate_day_v8(
            day_bars, date, prev_day_ohlc, vix, daily_trend, dte, is_exp)

        # BTST for winning PUTs
        if di + 1 < len(trading_dates):
            next_date = trading_dates[di + 1]
            next_d = day_groups[next_date]
            next_open = next_d["open"].iloc[0]
            day_close_price = day_bars["close"].iloc[-1]

            for trade in day_trades:
                if (trade["action"] == "BUY_PUT"
                        and trade["pnl"] > 0
                        and not is_exp
                        and vix < 25
                        and trade["exit_reason"] in ("eod_close", "time_exit")):
                    gap = (next_open - day_close_price) / day_close_price * 100
                    if gap < 0:
                        on_pnl = (day_close_price - next_open) * trade["qty"] * 0.5 - 50
                        on_pnl = max(on_pnl, -trade["pnl"] * 0.5)
                    else:
                        on_pnl = -abs(gap) * trade["qty"] * 0.3
                        on_pnl = max(on_pnl, -trade["pnl"] * 0.5)
                    trade["btst_pnl"] = round(on_pnl, 0)
                    trade["pnl"] = round(trade["pnl"] + on_pnl, 0)

        day_pnl = sum(t["pnl"] for t in day_trades)
        day_wins = len([t for t in day_trades if t["pnl"] > 0])
        equity += day_pnl
        if equity > peak_equity:
            peak_equity = equity
        dd = (peak_equity - equity) / peak_equity * 100
        if dd > max_dd:
            max_dd = dd
        equity_curve.append(equity)
        daily_pnls.append(day_pnl)

        day_open = day_bars["open"].iloc[0]
        day_close = day_bars["close"].iloc[-1]
        day_chg = (day_close - day_open) / day_open * 100
        dow = date.strftime("%a")

        print(f"{date}  {dow:<5} {day_open:>8.0f} {day_close:>8.0f} {day_chg:>+5.2f}% "
              f"{vix:>5.1f} {len(day_trades):>5}t {day_wins:>4}W "
              f"Rs {day_pnl:>+9,.0f} Rs {equity:>11,.0f} {dd:>5.1f}%")

        all_trades.extend(day_trades)

    # ======= RESULTS =======
    print("\n" + "=" * 120)
    print("  V8 LEARNED MODEL — FINAL RESULTS")
    print("=" * 120)
    net_pnl = equity - CAPITAL
    total = len(all_trades)
    wins = [t for t in all_trades if t["pnl"] > 0]
    losses = [t for t in all_trades if t["pnl"] <= 0]
    wr = len(wins) / total * 100 if total else 0
    gw = sum(t["pnl"] for t in wins)
    gl = abs(sum(t["pnl"] for t in losses))
    pf = gw / gl if gl > 0 else float("inf")
    daily_arr = np.array([d for d in daily_pnls if d != 0])
    sharpe = (daily_arr.mean() / daily_arr.std()) * np.sqrt(252) if len(daily_arr) > 1 and daily_arr.std() > 0 else 0

    puts = [t for t in all_trades if t["action"] == "BUY_PUT"]
    calls = [t for t in all_trades if t["action"] == "BUY_CALL"]
    btst_trades = [t for t in all_trades if t.get("btst_pnl", 0) != 0]

    print(f"  Net P&L:          Rs {net_pnl:>+,}")
    print(f"  Return:           {net_pnl/CAPITAL*100:>+.1f}%")
    print(f"  Total Trades:     {total}")
    print(f"  Trades/Day:       {total/len(trading_dates):.1f}")
    print(f"  Win Rate:         {wr:.1f}% ({len(wins)}W / {len(losses)}L)")
    print(f"  Avg Win:          Rs {gw/max(len(wins),1):>+,.0f}")
    print(f"  Avg Loss:         Rs {-gl/max(len(losses),1):>+,.0f}")
    print(f"  Profit Factor:    {pf:.2f}")
    print(f"  Sharpe Ratio:     {sharpe:.2f}")
    print(f"  Max Drawdown:     {max_dd:.1f}%")
    print(f"  P&L per Trade:    Rs {net_pnl/max(total,1):>+,.0f}")

    print(f"\n  ACTION BREAKDOWN:")
    for label, subset in [("BUY_PUT", puts), ("BUY_CALL", calls)]:
        w = len([t for t in subset if t["pnl"] > 0])
        p = sum(t["pnl"] for t in subset)
        print(f"    {label}: {len(subset)}t  WR {w/max(len(subset),1)*100:.0f}%  Rs {p:+,}")

    print(f"\n  BTST TRADES: {len(btst_trades)}t  Rs {sum(t.get('btst_pnl',0) for t in btst_trades):+,}")

    print(f"\n  EXIT REASONS:")
    exit_stats = defaultdict(lambda: {"count": 0, "pnl": 0, "wins": 0})
    for t in all_trades:
        er = t["exit_reason"]
        exit_stats[er]["count"] += 1
        exit_stats[er]["pnl"] += t["pnl"]
        if t["pnl"] > 0:
            exit_stats[er]["wins"] += 1
    for er in sorted(exit_stats, key=lambda x: exit_stats[x]["pnl"], reverse=True):
        s = exit_stats[er]
        wr_val = s["wins"] / s["count"] * 100 if s["count"] > 0 else 0
        print(f"    {er:<20}: {s['count']:>3}t  Rs {s['pnl']:>+9,.0f}  WR {wr_val:.0f}%")

    # ======= COMPARISON WITH V1 (daywise_analysis baseline) =======
    print(f"\n  COMPARISON: V8 vs V1 (daywise_analysis baseline)")
    print(f"  {'Metric':<25} {'V1 (baseline)':>15} {'V8 (learned)':>15} {'Change':>15}")
    print("-" * 75)
    v1 = {"pnl": 252115, "trades": 388, "wr": 41.8, "sharpe": 3.71, "dd": 9.9, "pf": 1.34}
    print(f"  {'Net P&L':<25} Rs {v1['pnl']:>+,} Rs {net_pnl:>+,.0f} Rs {net_pnl-v1['pnl']:>+,.0f}")
    print(f"  {'Total Trades':<25} {v1['trades']:>15} {total:>15} {total-v1['trades']:>+15}")
    print(f"  {'Win Rate':<25} {v1['wr']:>14.1f}% {wr:>14.1f}% {wr-v1['wr']:>+14.1f}%")
    print(f"  {'Sharpe':<25} {v1['sharpe']:>15.2f} {sharpe:>15.2f} {sharpe-v1['sharpe']:>+15.2f}")
    print(f"  {'Max Drawdown':<25} {v1['dd']:>14.1f}% {max_dd:>14.1f}% {max_dd-v1['dd']:>+14.1f}%")
    print(f"  {'Profit Factor':<25} {v1['pf']:>15.2f} {pf:>15.2f} {pf-v1['pf']:>+15.2f}")
    print("=" * 120)

    # Save
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
            "btst_trades": len(btst_trades),
            "btst_pnl": round(sum(t.get("btst_pnl", 0) for t in btst_trades)),
            "put_trades": len(puts),
            "put_pnl": round(sum(t["pnl"] for t in puts)),
            "call_trades": len(calls),
            "call_pnl": round(sum(t["pnl"] for t in calls)),
        },
        "exit_reasons": {er: dict(s) for er, s in exit_stats.items()},
        "v8_parameters": {
            "put_trail_pct": 0.010,
            "call_trail_pct": 0.008,
            "put_min_hold_for_trail": 90,
            "call_min_hold_for_trail": 60,
            "put_max_hold": 300,
            "call_max_hold": 270,
            "cooldown_minutes": 10,
            "avoid_entry_ranges": [[45, 75], [165, 225]],
            "put_min_score": 4.0,
            "call_min_score": 5.0,
            "supertrend_flip_exit": False,
            "btst_enabled": True,
            "btst_vix_cap": 25,
        },
    }

    out_path = project_root / "data" / "v8_learned_model_results.json"
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")
