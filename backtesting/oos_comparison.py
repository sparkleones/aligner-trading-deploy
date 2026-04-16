"""
Out-of-Sample (OOS) Comparison: V3 vs V4 on December 2024 (Unseen Data).

Tests BOTH the original V3 logic and the V4 optimized logic on Dec 2024 data
that was NOT used during any training/optimization cycle.

V3 characteristics:
  - COOLDOWN_BARS = 2
  - sr_breakout entries ENABLED
  - Composite window: bars 2-4
  - DOW rules: Mon=PUT, Tue=PUT, Wed=CALL, Thu=PUT, Fri=CALL
  - No gap_fade entry type
  - BTST: only if intraday_pnl > 0 (strict)
  - Trail: 0.3% trail, no profit gate

V4 characteristics:
  - COOLDOWN_BARS = 0
  - sr_breakout REMOVED
  - Composite window: bars 3-5 + bars 8-10
  - DOW rules: Mon=CALL, Tue=PUT, Wed=CALL, Thu=PUT, Fri=PUT
  - gap_fade for |gap| > 1.2%
  - Enhanced BTST: intraday_pnl >= 0 (includes breakeven), time_exit eligible
  - Zero-hero: threshold -0.5%, VIX >= 13 (lowered from -0.8% / VIX >= 15)

Capital: Rs 200,000 | Data: December 2024 from Yahoo Finance
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

# Import shared functions from the V4 engine
from backtesting.paper_trading_real_data import (
    download_real_data,
    generate_intraday_path,
    sr_multi_method,
    bs_premium,
    get_strike_and_type,
    get_lot_count,
    check_trade_exit,
    compute_composite_scores,
    LOT_SIZE,
    BROKERAGE,
    STRIKE_INTERVAL,
    CAPITAL,
    TOTAL_BARS,
    MAX_TRADES_PER_DAY,
    MAX_CONCURRENT,
    MIN_CONFIDENCE,
    TRAIL_PCT,
    PUT_MAX_HOLD,
    CALL_MAX_HOLD,
)


# ===========================================================================
# V3 COMPOSITE SCORING (original DOW rules)
# ===========================================================================

def compute_composite_scores_v3(vix, above_sma50, above_sma20, rsi, dow,
                                 prev_change, vix_spike, spot, support, resistance,
                                 ema9=None, ema21=None, weekly_sma=None):
    """V3 composite scoring with ORIGINAL DOW rules.

    V3 DOW rules: Mon=PUT, Tue=PUT, Wed=CALL, Thu=PUT, Fri=CALL
    (Different from V4: Mon=CALL, Tue=PUT, Wed=CALL, Thu=PUT, Fri=PUT)
    """
    scores = {"BUY_CALL": 0.0, "BUY_PUT": 0.0}

    if vix < 12: scores["BUY_CALL"] += 3.0
    elif vix < 17: scores["BUY_PUT"] += 3.0
    elif vix < 25: scores["BUY_PUT"] += 3.5
    else: scores["BUY_PUT"] += 4.0

    if not above_sma50: scores["BUY_PUT"] += 2.0
    else: scores["BUY_CALL"] += 2.0

    if not above_sma20: scores["BUY_PUT"] += 1.0
    else: scores["BUY_CALL"] += 1.0

    if rsi < 30: scores["BUY_PUT"] += 1.5
    elif rsi > 70: scores["BUY_PUT"] += 1.5

    # V3 ORIGINAL DOW rules (different from V4)
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


# ===========================================================================
# V3 ENTRY DETECTION (original logic with sr_breakout, no gap_fade)
# ===========================================================================

def detect_entries_v3(bar_idx, path, support, resistance, vix, gap_pct,
                      composite_action, composite_conf, is_expiry,
                      prev_high, prev_low, above_sma50, above_sma20):
    """V3 original entry detection.

    Differences from V4:
      - sr_breakout entries ENABLED
      - No gap_fade entry type (large gaps just skip)
      - Composite window: bars 2-4 (not 3-5 + 8-10)
      - Zero-hero threshold: gap <= -0.8% and VIX >= 15 (stricter)
    """
    signals = []
    spot = path[bar_idx]

    # 1. GAP ENTRY (bar 0 only) - V3: no gap_fade, no large-gap reversal
    if bar_idx == 0 and abs(gap_pct) >= 0.3:
        if gap_pct < -0.3:
            conf = min(0.90, 0.60 + abs(gap_pct) * 0.10)
            signals.append(("BUY_PUT", "gap_entry", conf, False))
            # V3 zero-hero: stricter thresholds (-0.8% gap, VIX >= 15)
            if gap_pct <= -0.8 and vix >= 15:
                signals.append(("BUY_PUT", "gap_zero_hero", 0.70, True))
        elif gap_pct > 0.5 and above_sma50:
            conf = min(0.85, 0.55 + gap_pct * 0.08)
            signals.append(("BUY_CALL", "gap_entry", conf, False))

    # 2. ORB ENTRY (bar 1) - same as V4
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

    # 3. S/R BOUNCE (bar 2+) - same as V4
    if bar_idx >= 2 and bar_idx <= 18:
        prev_spot = path[bar_idx - 1]

        if support and abs(spot - support) / spot < 0.003:
            if spot > prev_spot and prev_spot <= spot:
                conf = 0.65
                signals.append(("BUY_CALL", "sr_bounce_support", conf, False))

        if resistance and abs(spot - resistance) / spot < 0.003:
            if spot < prev_spot:
                conf = 0.70
                signals.append(("BUY_PUT", "sr_bounce_resistance", conf, False))

    # 4. S/R BREAKOUT (V3: ENABLED, V4 removed this)
    if bar_idx >= 2 and bar_idx <= 18:
        prev_spot = path[bar_idx - 1]
        momentum = abs(spot - prev_spot) / spot * 100

        # Breakdown through support (bearish)
        if support and spot < support and momentum > 0.1:
            conf = min(0.75, 0.55 + momentum * 0.10)
            signals.append(("BUY_PUT", "sr_breakout", conf, False))
            # Zero-hero on strong breakdowns
            if momentum > 0.3 and vix >= 15:
                signals.append(("BUY_PUT", "sr_zero_hero", 0.65, True))

        # Breakout through resistance (bullish)
        if resistance and spot > resistance and momentum > 0.1:
            conf = min(0.75, 0.55 + momentum * 0.10)
            signals.append(("BUY_CALL", "sr_breakout", conf, False))

    # 5. COMPOSITE SCORING - V3: window bars 2-4 (not V4's 3-5 + 8-10)
    if composite_conf >= MIN_CONFIDENCE:
        if composite_action == "BUY_PUT" and 2 <= bar_idx <= 4:
            if not (0.60 <= composite_conf < 0.70):
                signals.append(("BUY_PUT", "composite", composite_conf, False))
        elif composite_action == "BUY_CALL" and 2 <= bar_idx <= 4 and vix < 12:
            if composite_conf >= 0.75:
                signals.append(("BUY_CALL", "composite", composite_conf, False))

    return signals


# ===========================================================================
# V4 ENTRY DETECTION (imported logic, re-implemented for clarity)
# ===========================================================================

def detect_entries_v4(bar_idx, path, support, resistance, vix, gap_pct,
                      composite_action, composite_conf, is_expiry,
                      prev_high, prev_low, above_sma50, above_sma20):
    """V4 optimized entry detection (same as paper_trading_real_data.detect_entries)."""
    signals = []
    spot = path[bar_idx]

    # 1. GAP ENTRY (bar 0) - V4: includes gap_fade for |gap| > 1.2%
    if bar_idx == 0 and abs(gap_pct) >= 0.3:
        is_large_gap = abs(gap_pct) > 1.2

        if gap_pct < -0.3:
            if is_large_gap:
                conf = min(0.85, 0.65 + abs(gap_pct) * 0.05)
                signals.append(("BUY_CALL", "gap_fade", conf, False))
            else:
                conf = min(0.90, 0.60 + abs(gap_pct) * 0.10)
                signals.append(("BUY_PUT", "gap_entry", conf, False))
            # V4 zero-hero: lowered thresholds (-0.5% gap, VIX >= 13)
            if -1.2 <= gap_pct < -0.5 and vix >= 13:
                signals.append(("BUY_PUT", "gap_zero_hero", 0.70, True))
        elif gap_pct > 0.3:
            if is_large_gap:
                conf = min(0.85, 0.65 + gap_pct * 0.05)
                signals.append(("BUY_PUT", "gap_fade", conf, False))
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
            if spot > prev_spot and prev_spot <= spot:
                conf = 0.65
                signals.append(("BUY_CALL", "sr_bounce_support", conf, False))
        if resistance and abs(spot - resistance) / spot < 0.003:
            if spot < prev_spot:
                conf = 0.70
                signals.append(("BUY_PUT", "sr_bounce_resistance", conf, False))

    # 4. S/R BREAKOUT - REMOVED in V4

    # 5. COMPOSITE SCORING - V4: bars 3-5 + 8-10
    if composite_conf >= MIN_CONFIDENCE:
        put_window = (3 <= bar_idx <= 5) or (8 <= bar_idx <= 10)
        if composite_action == "BUY_PUT" and put_window:
            if not (0.60 <= composite_conf < 0.70):
                signals.append(("BUY_PUT", "composite", composite_conf, False))
        elif composite_action == "BUY_CALL" and 4 <= bar_idx <= 8 and vix < 12:
            if composite_conf >= 0.75:
                signals.append(("BUY_CALL", "composite", composite_conf, False))

    return signals


# ===========================================================================
# UNIFIED DAY SIMULATION (version-aware)
# ===========================================================================

def simulate_day_oos(row, row_idx, nifty_df, equity, close_prices, version="V4"):
    """Simulate a single day with version-specific logic.

    version: "V3" or "V4"
    """
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
    cooldown = 2 if version == "V3" else 0

    # VIX < 10: skip day (both versions)
    if vix < 10:
        return 0, [{"action": "SKIP", "reason": f"VIX too low ({vix:.1f})",
                     "date": date_str, "dow": dow, "vix": round(vix, 1),
                     "is_expiry": is_expiry}]

    # S/R levels
    support, resistance = sr_multi_method(
        entry_spot, prev_high, prev_low, sma20, sma50,
        close_history=close_prices, idx=row_idx
    )

    # Composite scoring (version-aware DOW rules)
    if version == "V3":
        scores = compute_composite_scores_v3(
            vix, above_sma50, above_sma20, rsi, dow, prev_change, vix_spike,
            entry_spot, support, resistance,
            ema9=ema9, ema21=ema21, weekly_sma=weekly_sma,
        )
    else:
        scores = compute_composite_scores(
            vix, above_sma50, above_sma20, rsi, dow, prev_change, vix_spike,
            entry_spot, support, resistance,
            ema9=ema9, ema21=ema21, weekly_sma=weekly_sma,
        )

    best_composite = max(scores, key=scores.get)
    total_score = sum(scores.values())
    composite_conf = scores[best_composite] / total_score if total_score > 0 else 0

    # Generate intraday path (deterministic per day)
    np.random.seed(int(abs(entry_spot * 100)) % 2**31 + row_idx)
    path = generate_intraday_path(entry_spot, day_high, day_low, day_close)

    # Multi-trade tracking
    open_trades = []
    closed_trades = []
    total_day_trades = 0
    last_exit_bar = -99

    # BAR-BY-BAR SIMULATION
    for bar_idx in range(TOTAL_BARS):
        bar_spot = path[bar_idx]
        bar_dte = max(0.05, dte_market - bar_idx * 15 / 1440)

        # 1. CHECK EXITS for all open trades
        trades_to_close = []
        for ti, trade in enumerate(open_trades):
            exit_signal = check_trade_exit(
                trade, bar_idx, bar_spot, bar_dte, vix, support, resistance,
                is_expiry, path
            )
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
                and bar_idx - last_exit_bar >= cooldown
                and bar_idx < 20):

            if version == "V3":
                entries = detect_entries_v3(
                    bar_idx, path, support, resistance, vix, gap_pct,
                    best_composite, composite_conf, is_expiry,
                    prev_high, prev_low, above_sma50, above_sma20,
                )
            else:
                entries = detect_entries_v4(
                    bar_idx, path, support, resistance, vix, gap_pct,
                    best_composite, composite_conf, is_expiry,
                    prev_high, prev_low, above_sma50, above_sma20,
                )

            if entries:
                entries.sort(key=lambda x: x[2], reverse=True)
                action, entry_type, conf, is_zero_hero = entries[0]

                same_dir = [t for t in open_trades if t["action"] == action]
                if not same_dir:
                    strike, opt_type = get_strike_and_type(action, bar_spot, vix, is_zero_hero)
                    num_lots = get_lot_count(vix, is_zero_hero)
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
                        "vix_regime": str(row.get("VIXRegime", "")),
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

    # 5. BTST LOGIC (version-aware)
    for trade in closed_trades:
        if version == "V3":
            # V3 BTST: only if intraday_pnl > 0 (strict), no time_exit
            btst_eligible = (
                trade["action"] == "BUY_PUT"
                and trade["intraday_pnl"] > 0  # V3: strictly > 0
                and not is_expiry
                and trade["exit_reason"] in ("eod_close", "trail_pct")
                and row_idx + 1 < len(nifty_df)
            )
        else:
            # V4 BTST: includes breakeven (>= 0), includes time_exit
            btst_eligible = (
                trade["action"] == "BUY_PUT"
                and trade["intraday_pnl"] >= 0  # V4: >= 0
                and not is_expiry
                and trade["exit_reason"] in ("eod_close", "trail_pct", "time_exit")
                and row_idx + 1 < len(nifty_df)
            )

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
# RUN A FULL BACKTEST FOR ONE VERSION
# ===========================================================================

def run_backtest(nifty, version="V4"):
    """Run complete backtest for a given version, return results dict."""
    close_prices = nifty["Close"].values.tolist()
    equity = CAPITAL
    equity_curve = [CAPITAL]
    all_trades = []
    daily_results = []
    peak_equity = CAPITAL
    max_dd = 0

    entry_type_stats = defaultdict(lambda: {"count": 0, "pnl": 0, "wins": 0})
    exit_reason_stats = defaultdict(lambda: {"count": 0, "pnl": 0, "wins": 0})

    for i in range(len(nifty)):
        row = nifty.iloc[i]
        date_str = str(nifty.index[i].date())

        day_pnl, day_trades = simulate_day_oos(
            row, i, nifty, equity, close_prices, version=version
        )

        if len(day_trades) == 1 and day_trades[0].get("action") == "SKIP":
            daily_results.append({
                "day": i + 1, "date": date_str,
                "dow": day_trades[0].get("dow", ""),
                "action": "SKIP", "day_pnl": 0,
                "equity": round(equity, 0), "num_trades": 0,
            })
            equity_curve.append(equity)
            continue

        equity += day_pnl

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
        n_trades = len(day_trades)
        day_wins = len([t for t in day_trades if t["total_pnl"] > 0])

        daily_results.append({
            "day": i + 1, "date": date_str,
            "dow": str(row["DOW"]),
            "day_pnl": round(day_pnl, 0),
            "equity": round(equity, 0),
            "num_trades": n_trades,
            "num_wins": day_wins,
            "vix": round(float(row["VIX"]), 1),
        })

    # Compute metrics
    active_trades = [t for t in all_trades if t.get("action") != "SKIP"]
    win_trades = [t for t in active_trades if t["total_pnl"] > 0]
    loss_trades = [t for t in active_trades if t["total_pnl"] < 0]

    total_pnl = equity - CAPITAL
    daily_pnls = [d["day_pnl"] for d in daily_results if d.get("num_trades", 0) > 0]
    avg_daily = np.mean(daily_pnls) if daily_pnls else 0
    std_daily = np.std(daily_pnls) if daily_pnls else 1
    sharpe = (avg_daily / std_daily) * np.sqrt(252) if std_daily > 0 else 0
    gross_w = sum(t["total_pnl"] for t in win_trades)
    gross_l = abs(sum(t["total_pnl"] for t in loss_trades))
    pf = gross_w / max(1, gross_l)
    active_days = len([d for d in daily_results if d.get("num_trades", 0) > 0])
    trades_per_day = len(active_trades) / max(1, active_days)

    best_trade = max(t["total_pnl"] for t in active_trades) if active_trades else 0
    worst_trade = min(t["total_pnl"] for t in active_trades) if active_trades else 0

    return {
        "version": version,
        "capital": CAPITAL,
        "final_equity": round(equity, 0),
        "total_pnl": round(total_pnl, 0),
        "return_pct": round(total_pnl / CAPITAL * 100, 2),
        "max_dd_pct": round(max_dd, 2),
        "sharpe": round(sharpe, 2),
        "profit_factor": round(pf, 2),
        "win_rate": round(len(win_trades) / max(1, len(active_trades)) * 100, 1),
        "total_trades": len(active_trades),
        "active_days": active_days,
        "trades_per_day": round(trades_per_day, 1),
        "best_trade": round(best_trade, 0),
        "worst_trade": round(worst_trade, 0),
        "wins": len(win_trades),
        "losses": len(loss_trades),
        "entry_type_stats": {k: dict(v) for k, v in entry_type_stats.items()},
        "exit_reason_stats": {k: dict(v) for k, v in exit_reason_stats.items()},
        "daily_results": daily_results,
        "trades": active_trades,
        "equity_curve": [round(e, 0) for e in equity_curve],
    }


# ===========================================================================
# COMPARISON OUTPUT
# ===========================================================================

def winner(v3_val, v4_val, higher_better=True):
    """Determine which version wins for a metric."""
    if higher_better:
        if v3_val > v4_val: return "V3"
        elif v4_val > v3_val: return "V4"
        else: return "TIE"
    else:
        if v3_val < v4_val: return "V3"
        elif v4_val < v3_val: return "V4"
        else: return "TIE"


def print_comparison(v3, v4):
    """Print side-by-side comparison table."""
    print()
    print("=" * 80)
    print("  OOS COMPARISON: V3 vs V4 on DECEMBER 2024 (Unseen Data)")
    print("=" * 80)

    sep = "-" * 80

    rows = [
        ("Return",
         f"+{v3['return_pct']:.1f}%" if v3['return_pct'] >= 0 else f"{v3['return_pct']:.1f}%",
         f"+{v4['return_pct']:.1f}%" if v4['return_pct'] >= 0 else f"{v4['return_pct']:.1f}%",
         winner(v3['return_pct'], v4['return_pct'])),
        ("Total P&L",
         f"Rs {v3['total_pnl']:>+,.0f}",
         f"Rs {v4['total_pnl']:>+,.0f}",
         winner(v3['total_pnl'], v4['total_pnl'])),
        ("Max Drawdown",
         f"{v3['max_dd_pct']:.1f}%",
         f"{v4['max_dd_pct']:.1f}%",
         winner(v3['max_dd_pct'], v4['max_dd_pct'], higher_better=False)),
        ("Sharpe Ratio",
         f"{v3['sharpe']:.2f}",
         f"{v4['sharpe']:.2f}",
         winner(v3['sharpe'], v4['sharpe'])),
        ("Profit Factor",
         f"{v3['profit_factor']:.2f}",
         f"{v4['profit_factor']:.2f}",
         winner(v3['profit_factor'], v4['profit_factor'])),
        ("Win Rate",
         f"{v3['win_rate']:.0f}%",
         f"{v4['win_rate']:.0f}%",
         winner(v3['win_rate'], v4['win_rate'])),
        ("Total Trades",
         f"{v3['total_trades']}",
         f"{v4['total_trades']}",
         ""),
        ("Avg Trades/Day",
         f"{v3['trades_per_day']:.1f}",
         f"{v4['trades_per_day']:.1f}",
         ""),
        ("Best Trade",
         f"Rs {v3['best_trade']:>+,.0f}",
         f"Rs {v4['best_trade']:>+,.0f}",
         winner(v3['best_trade'], v4['best_trade'])),
        ("Worst Trade",
         f"Rs {v3['worst_trade']:>+,.0f}",
         f"Rs {v4['worst_trade']:>+,.0f}",
         winner(v3['worst_trade'], v4['worst_trade'], higher_better=True)),
    ]

    print(f"\n  {'Metric':<24s} {'V3':>16s} {'V4':>16s} {'Winner':>10s}")
    print(f"  {sep}")
    for label, v3_str, v4_str, w in rows:
        print(f"  {label:<24s} {v3_str:>16s} {v4_str:>16s} {w:>10s}")

    # Entry type comparison
    print(f"\n  ENTRY TYPE COMPARISON:")
    print(f"  {sep}")
    all_entry_types = sorted(set(list(v3["entry_type_stats"].keys()) +
                                 list(v4["entry_type_stats"].keys())))
    print(f"  {'Entry Type':<24s} {'V3 Count':>8s} {'V3 P&L':>12s} {'V3 WR':>7s}"
          f" {'V4 Count':>8s} {'V4 P&L':>12s} {'V4 WR':>7s}")
    print(f"  {sep}")
    for et in all_entry_types:
        v3e = v3["entry_type_stats"].get(et, {"count": 0, "pnl": 0, "wins": 0})
        v4e = v4["entry_type_stats"].get(et, {"count": 0, "pnl": 0, "wins": 0})
        v3_wr = v3e["wins"] / max(1, v3e["count"]) * 100
        v4_wr = v4e["wins"] / max(1, v4e["count"]) * 100
        print(f"  {et:<24s} {v3e['count']:>8d} {v3e['pnl']:>+12,.0f} {v3_wr:>6.0f}%"
              f" {v4e['count']:>8d} {v4e['pnl']:>+12,.0f} {v4_wr:>6.0f}%")

    # Exit reason comparison
    print(f"\n  EXIT REASON COMPARISON:")
    print(f"  {sep}")
    all_exit_reasons = sorted(set(list(v3["exit_reason_stats"].keys()) +
                                   list(v4["exit_reason_stats"].keys())))
    print(f"  {'Exit Reason':<24s} {'V3 Count':>8s} {'V3 P&L':>12s} {'V3 WR':>7s}"
          f" {'V4 Count':>8s} {'V4 P&L':>12s} {'V4 WR':>7s}")
    print(f"  {sep}")
    for er in all_exit_reasons:
        v3e = v3["exit_reason_stats"].get(er, {"count": 0, "pnl": 0, "wins": 0})
        v4e = v4["exit_reason_stats"].get(er, {"count": 0, "pnl": 0, "wins": 0})
        v3_wr = v3e["wins"] / max(1, v3e["count"]) * 100
        v4_wr = v4e["wins"] / max(1, v4e["count"]) * 100
        print(f"  {er:<24s} {v3e['count']:>8d} {v3e['pnl']:>+12,.0f} {v3_wr:>6.0f}%"
              f" {v4e['count']:>8d} {v4e['pnl']:>+12,.0f} {v4_wr:>6.0f}%")

    # Daily breakdown
    print(f"\n  DAILY P&L COMPARISON (V3 vs V4):")
    print(f"  {sep}")
    print(f"  {'Date':<12s} {'DOW':<5s} {'V3 P&L':>12s} {'V4 P&L':>12s} {'Diff':>12s} {'Better':>8s}")
    print(f"  {sep}")
    v3_daily = {d["date"]: d for d in v3["daily_results"]}
    v4_daily = {d["date"]: d for d in v4["daily_results"]}
    all_dates = sorted(set(list(v3_daily.keys()) + list(v4_daily.keys())))
    for dt in all_dates:
        d3 = v3_daily.get(dt, {"day_pnl": 0, "dow": "?"})
        d4 = v4_daily.get(dt, {"day_pnl": 0, "dow": "?"})
        p3 = d3["day_pnl"]
        p4 = d4["day_pnl"]
        diff = p4 - p3
        better = "V4" if p4 > p3 else ("V3" if p3 > p4 else "TIE")
        dow = d3.get("dow", d4.get("dow", "?"))
        print(f"  {dt:<12s} {dow:<5s} {p3:>+12,.0f} {p4:>+12,.0f} {diff:>+12,.0f} {better:>8s}")

    # Summary verdict
    print(f"\n  {sep}")
    v3_score = 0
    v4_score = 0
    metrics = [
        ("Return", v3["return_pct"], v4["return_pct"], True),
        ("Max DD", v3["max_dd_pct"], v4["max_dd_pct"], False),
        ("Sharpe", v3["sharpe"], v4["sharpe"], True),
        ("PF", v3["profit_factor"], v4["profit_factor"], True),
        ("Win Rate", v3["win_rate"], v4["win_rate"], True),
    ]
    for name, v3v, v4v, hb in metrics:
        w = winner(v3v, v4v, hb)
        if w == "V3": v3_score += 1
        elif w == "V4": v4_score += 1

    overall = "V4" if v4_score > v3_score else ("V3" if v3_score > v4_score else "TIE")
    print(f"  OVERALL VERDICT: {overall} wins ({v3_score}-{v4_score} on key metrics)")
    print(f"  V3 Return: {v3['return_pct']:+.1f}% | V4 Return: {v4['return_pct']:+.1f}%")
    print(f"  V3 Sharpe: {v3['sharpe']:.2f} | V4 Sharpe: {v4['sharpe']:.2f}")
    print(f"  V3 Max DD: {v3['max_dd_pct']:.1f}% | V4 Max DD: {v4['max_dd_pct']:.1f}%")
    print("=" * 80)


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    print("=" * 80)
    print("  OUT-OF-SAMPLE TEST: V3 vs V4 on DECEMBER 2024")
    print("  Data: Yahoo Finance (^NSEI + ^INDIAVIX)")
    print("  Period: Dec 1-31, 2024 (unseen/unknown data)")
    print("  Capital: Rs 200,000")
    print("=" * 80)

    # Download Dec 2024 data (with warmup for SMA50 handled internally)
    nifty = download_real_data(start="2024-12-01", end="2025-01-01")

    print(f"\n  OOS test period: {nifty.index[0].date()} to {nifty.index[-1].date()}")
    print(f"  Trading days: {len(nifty)}")
    print(f"  NIFTY range: {nifty['Low'].min():.0f} - {nifty['High'].max():.0f}")
    print(f"  VIX range: {nifty['VIX'].min():.1f} - {nifty['VIX'].max():.1f}")

    # Run V3 backtest
    print(f"\n{'='*80}")
    print("  Running V3 (Original) backtest...")
    print(f"{'='*80}")
    v3_results = run_backtest(nifty, version="V3")
    print(f"  V3 complete: {v3_results['total_trades']} trades, "
          f"P&L: Rs {v3_results['total_pnl']:+,.0f} ({v3_results['return_pct']:+.1f}%)")

    # Run V4 backtest
    print(f"\n{'='*80}")
    print("  Running V4 (Optimized) backtest...")
    print(f"{'='*80}")
    v4_results = run_backtest(nifty, version="V4")
    print(f"  V4 complete: {v4_results['total_trades']} trades, "
          f"P&L: Rs {v4_results['total_pnl']:+,.0f} ({v4_results['return_pct']:+.1f}%)")

    # Print comparison
    print_comparison(v3_results, v4_results)

    # Save results to JSON
    output = {
        "test_date": datetime.now().isoformat(),
        "oos_period": "December 2024",
        "data_range": f"{nifty.index[0].date()} to {nifty.index[-1].date()}",
        "trading_days": len(nifty),
        "capital": CAPITAL,
        "v3": {k: v for k, v in v3_results.items() if k not in ("trades", "daily_results", "equity_curve")},
        "v4": {k: v for k, v in v4_results.items() if k not in ("trades", "daily_results", "equity_curve")},
        "v3_trades": v3_results["trades"],
        "v4_trades": v4_results["trades"],
        "v3_equity_curve": v3_results["equity_curve"],
        "v4_equity_curve": v4_results["equity_curve"],
    }

    output_path = project_root / "data" / "oos_dec2024_comparison.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to: {output_path}")


if __name__ == "__main__":
    main()
