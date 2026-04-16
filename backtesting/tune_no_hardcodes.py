"""
Tune for maximum flexibility: NO hard-coded day/timing blocks.
Let scoring engine + confluence filters be the ONLY gatekeeper.

Tests progressively from current to fully open market-driven approach.
"""

import sys
import datetime as dt
from pathlib import Path
from copy import deepcopy
from collections import defaultdict

import numpy as np
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scoring.config import V15_CONFIG
from scoring.indicators import compute_indicators
from scoring.engine import (
    score_entry,
    passes_confluence,
    evaluate_exit,
    compute_lots,
    detect_composite_entries,
)
from backtesting.option_pricer import price_option

DATA_DIR = project_root / "data" / "historical"
CAPITAL = 200_000
LOT_SIZE = 75
STRIKE_INTERVAL = 50
SLIPPAGE_PCT = 0.005
SPREAD_RS = 2.0
BROKERAGE_RT = 80.0


def load_data():
    end_date = dt.date(2025, 1, 31)
    start_date = dt.date(2024, 7, 1)
    warmup_start = start_date - dt.timedelta(days=10)

    all_dfs = []
    for f in sorted(DATA_DIR.glob("nifty_min_*.csv")):
        try:
            df = pd.read_csv(f, parse_dates=["timestamp"], index_col="timestamp")
            mask = (df.index.date >= warmup_start) & (df.index.date <= end_date)
            chunk = df[mask]
            if len(chunk) > 0:
                all_dfs.append(chunk)
        except Exception:
            continue
    combined = pd.concat(all_dfs)
    combined = combined[~combined.index.duplicated(keep="first")]
    combined.sort_index(inplace=True)

    df_5min = combined.resample("5min").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum",
    }).dropna(subset=["open"])
    df_5min = df_5min[(df_5min.index.time >= dt.time(9, 15)) & (df_5min.index.time <= dt.time(15, 30))]

    vix_lookup = {}
    for f in sorted(DATA_DIR.glob("vix_min_*.csv")):
        try:
            vdf = pd.read_csv(f, parse_dates=["timestamp"], index_col="timestamp")
            mask = (vdf.index.date >= warmup_start) & (vdf.index.date <= end_date)
            chunk = vdf[mask]
            if len(chunk) > 0:
                for d, group in chunk.groupby(chunk.index.date):
                    vix_lookup[d] = float(group["close"].iloc[-1])
        except Exception:
            continue

    day_groups = {}
    for d, group in df_5min.groupby(df_5min.index.date):
        day_groups[d] = [{
            "open": float(r["open"]), "high": float(r["high"]),
            "low": float(r["low"]), "close": float(r["close"]),
            "volume": float(r.get("volume", 0)),
            "date": str(d), "time": str(ts.time()),
        } for ts, r in group.iterrows()]

    return start_date, end_date, vix_lookup, day_groups


def get_strike_and_type(action, spot, vix, zero_hero=False):
    atm = round(spot / STRIKE_INTERVAL) * STRIKE_INTERVAL
    if zero_hero:
        offset = 200
        return (atm - offset, "PE") if action == "BUY_PUT" else (atm + offset, "CE")
    return (atm, "PE") if action == "BUY_PUT" else (atm, "CE")


def calc_premium(spot, strike, dte, vix, opt_type, slippage_sign=1):
    try:
        result = price_option(spot, strike, dte, vix, opt_type)
        prem = result["premium"] if isinstance(result, dict) else float(result)
    except Exception:
        intrinsic = max(0, spot - strike) if opt_type == "CE" else max(0, strike - spot)
        prem = intrinsic + 50
    prem += slippage_sign * prem * SLIPPAGE_PCT
    prem += slippage_sign * SPREAD_RS
    return max(1.0, prem)


def simulate_day(bars_5min, date, vix, cfg, prev_close, equity,
                 warmup_bars=None, is_expiry=False, consecutive_down_days=0):
    date_str = str(date)
    avoid_days = cfg.get("avoid_days", [])
    if date.weekday() in avoid_days:
        eod_spot = bars_5min[-1]["close"] if bars_5min else 0
        return [], 0.0, eod_spot

    bar_history = list(warmup_bars or [])
    open_trades = []
    closed_trades = []
    trades_today = 0
    last_exit_bar = -10
    orb_high = 0.0
    orb_low = 0.0
    gap_detected = False
    day_open = bars_5min[0]["close"] if bars_5min else 0

    span_per_lot = 40000
    if vix >= 25:
        span_per_lot = 60000
    elif vix >= 20:
        span_per_lot = 50000
    base_lots = max(1, int(equity * 0.70 / span_per_lot))

    dte = 7
    if is_expiry:
        dte = 0.5

    for bar_idx, bar in enumerate(bars_5min):
        spot = bar["close"]
        bar_history.append(bar)

        if bar_idx == 0:
            orb_high = bar["high"]
            orb_low = bar["low"]

        for trade in open_trades:
            if trade["action"] == "BUY_PUT":
                trade["best_fav"] = min(trade["best_fav"], spot)
            else:
                trade["best_fav"] = max(trade["best_fav"], spot)

        for trade in list(open_trades):
            indicators = compute_indicators(bar_history)
            if indicators is None:
                continue
            exit_reason = evaluate_exit(trade, bar_idx, spot, indicators, cfg,
                                         day_of_week=date.weekday())
            if exit_reason:
                exit_prem = calc_premium(spot, trade["strike"], dte, vix, trade["opt_type"], slippage_sign=-1)
                qty = trade["qty"]
                if trade["action"] == "BUY_PUT":
                    pnl = (exit_prem - trade["entry_prem"]) * qty
                    pnl += qty * (trade["entry_spot"] - spot) * 0.01
                else:
                    pnl = (exit_prem - trade["entry_prem"]) * qty
                    pnl += qty * (spot - trade["entry_spot"]) * 0.01
                pnl -= BROKERAGE_RT
                trade["pnl"] = pnl
                trade["exit_bar"] = bar_idx
                trade["exit_reason"] = exit_reason
                closed_trades.append(trade)
                open_trades.remove(trade)
                last_exit_bar = bar_idx

        if len(open_trades) >= cfg.get("max_concurrent", 3):
            continue
        if trades_today >= cfg.get("max_trades_per_day", 7):
            continue
        if bar_idx - last_exit_bar < cfg.get("cooldown_bars", 2):
            continue

        indicators = compute_indicators(bar_history)
        if indicators is None:
            continue

        if vix < cfg.get("vix_floor", 11) or vix > cfg.get("vix_ceil", 35):
            continue

        # Entry windows (if configured)
        entry_windows = cfg.get("entry_windows_bars", [])
        if entry_windows:
            in_window = any(s <= bar_idx <= e for s, e in entry_windows)
            if not in_window and bar_idx > 2:
                continue

        action, conf = score_entry(
            indicators, vix, cfg,
            pcr=0.0, bar_idx=bar_idx,
            consecutive_down_days=consecutive_down_days,
            regime_block_reversion=False,
            is_expiry=is_expiry,
        )
        entry_type = "v8_indicator"
        is_zero_hero = False

        if action is None:
            composites = detect_composite_entries(
                bar, bar_idx, spot, vix, cfg,
                prev_close=prev_close,
                gap_detected=gap_detected,
                orb_high=orb_high, orb_low=orb_low,
            )
            if cfg.get("disable_zero_hero", False):
                composites = [c for c in composites if not c[3]]
            if composites:
                composites.sort(key=lambda x: x[2], reverse=True)
                action, entry_type, conf, is_zero_hero = composites[0]
                if bar_idx == 0:
                    gap_detected = True

        if action is None:
            continue

        if not passes_confluence(
            action, conf, indicators, bar_idx, cfg,
            current_spot=spot,
            prev_close=prev_close,
            day_open=day_open,
        ):
            continue

        if any(t["action"] == action for t in open_trades):
            continue

        cfg_with_atr = cfg
        if cfg.get("use_atr_sizing"):
            cfg_with_atr = cfg.copy()
            cfg_with_atr["_current_atr"] = indicators.get("atr", 0)
        lots = compute_lots(action, conf, vix, indicators.get("rsi", 50),
                            is_expiry, base_lots, cfg_with_atr)
        if is_zero_hero:
            lots = min(cfg.get("zero_hero_max_lots", 3), max(1, lots))

        if cfg.get("vix_lot_scaling"):
            if vix < 13:
                lots = max(1, int(lots * cfg.get("vix_below13_mult", 0.3)))
            elif 14 <= vix < 15:
                lots = max(1, int(lots * cfg.get("vix_14_15_mult", 0.5)))
            elif 15 <= vix < 17:
                lots = max(1, int(lots * cfg.get("vix_15_17_mult", 1.5)))
            elif vix >= 17:
                lots = max(1, int(lots * cfg.get("vix_17plus_mult", 2.0)))
        if entry_type and "orb" in entry_type:
            lots = max(1, int(lots * cfg.get("orb_lot_mult", 1.0)))
        lots = min(lots, cfg.get("max_lots_cap", 999))

        strike, opt_type = get_strike_and_type(action, spot, vix, zero_hero=is_zero_hero)
        qty = lots * LOT_SIZE
        entry_prem = calc_premium(spot, strike, dte, vix, opt_type, slippage_sign=1)

        trade = {
            "date": date_str, "action": action, "entry_bar": bar_idx,
            "entry_spot": spot, "entry_prem": entry_prem,
            "strike": strike, "opt_type": opt_type,
            "qty": qty, "lots": lots, "best_fav": spot,
            "is_zero_hero": is_zero_hero, "entry_type": entry_type,
        }
        open_trades.append(trade)
        trades_today += 1

    eod_spot = bars_5min[-1]["close"] if bars_5min else 0
    for trade in open_trades:
        exit_prem = calc_premium(eod_spot, trade["strike"], dte, vix, trade["opt_type"], slippage_sign=-1)
        qty = trade["qty"]
        if trade["action"] == "BUY_PUT":
            pnl = (exit_prem - trade["entry_prem"]) * qty
            pnl += qty * (trade["entry_spot"] - eod_spot) * 0.01
        else:
            pnl = (exit_prem - trade["entry_prem"]) * qty
            pnl += qty * (eod_spot - trade["entry_spot"]) * 0.01
        pnl -= BROKERAGE_RT
        trade["pnl"] = pnl
        trade["exit_reason"] = "eod_force_close"
        closed_trades.append(trade)

    day_pnl = sum(t.get("pnl", 0) for t in closed_trades)
    return closed_trades, day_pnl, eod_spot


def run_backtest(cfg, label, start_date, end_date, vix_lookup, day_groups):
    all_dates = sorted(d for d in day_groups.keys() if d >= start_date)
    warmup_dates = sorted(d for d in day_groups.keys() if d < start_date)

    warmup_bars = []
    for d in warmup_dates[-3:]:
        warmup_bars.extend(day_groups[d])

    equity = CAPITAL
    all_trades = []
    prev_close = 0.0
    consecutive_down_days = 0
    max_equity = CAPITAL
    max_drawdown = 0.0
    trading_days = 0

    for date in all_dates:
        bars = day_groups[date]
        if len(bars) < 5:
            continue

        vix = vix_lookup.get(date, 14.0)
        is_expiry = (date.weekday() == 3)
        trading_days += 1

        trades, day_pnl, eod_close = simulate_day(
            bars, date, vix, cfg,
            prev_close=prev_close,
            equity=equity,
            warmup_bars=warmup_bars,
            is_expiry=is_expiry,
            consecutive_down_days=consecutive_down_days,
        )

        equity += day_pnl
        all_trades.extend(trades)
        max_equity = max(max_equity, equity)
        dd = (max_equity - equity) / max_equity * 100
        max_drawdown = max(max_drawdown, dd)

        if len(bars) >= 2:
            if bars[-1]["close"] < bars[0]["open"]:
                consecutive_down_days += 1
            else:
                consecutive_down_days = 0

        warmup_bars = warmup_bars[-(75 * 2):] + bars
        prev_close = eod_close

    total_pnl = sum(t.get("pnl", 0) for t in all_trades)
    wins = [t for t in all_trades if t.get("pnl", 0) > 0]
    losses = [t for t in all_trades if t.get("pnl", 0) <= 0]
    win_rate = len(wins) / len(all_trades) * 100 if all_trades else 0
    gross_win = sum(t["pnl"] for t in wins)
    gross_loss = sum(abs(t["pnl"]) for t in losses)
    profit_factor = gross_win / gross_loss if gross_loss > 0 else float("inf")
    trades_per_day = len(all_trades) / trading_days if trading_days > 0 else 0

    return {
        "label": label,
        "trades": len(all_trades),
        "trading_days": trading_days,
        "trades_per_day": trades_per_day,
        "wins": len(wins),
        "win_rate": win_rate,
        "total_pnl": total_pnl,
        "equity": equity,
        "return_x": equity / CAPITAL,
        "profit_factor": profit_factor,
        "max_drawdown": max_drawdown,
    }


def main():
    print("=" * 80)
    print("  NO HARD RULES — Market-Driven Trade Frequency Optimization")
    print("=" * 80)
    print()

    start_date, end_date, vix_lookup, day_groups = load_data()
    trading_days = len([d for d in day_groups if d >= start_date])
    print(f"  Period: {start_date} to {end_date} ({trading_days} trading days)")
    print()

    configs = []

    # ── BASELINE ──
    s0 = deepcopy(V15_CONFIG)
    s0["name"] = "S0_BASELINE"
    configs.append((s0, "S0_BASELINE"))

    # ── A1: Remove ALL hard timing + day blocks, fix RSI ceiling ──
    # Let scoring engine be the only gatekeeper
    a1 = deepcopy(V15_CONFIG)
    a1["name"] = "A1_NoBlocks"
    a1["avoid_days"] = []                      # No day blocks
    a1["entry_windows_bars"] = []              # No time windows — ALL bars allowed
    a1["avoid_windows_bars"] = []              # No lunch block
    a1["block_late_entries"] = 74              # Only block last bar (3:25 PM)
    a1["block_call_4th_hour"] = False          # Allow calls any time
    a1["rsi_call_kill_ceiling"] = 75           # Fix RSI ceiling bug
    a1["vix_floor"] = 13                       # Lower VIX floor
    a1["cooldown_bars"] = 1                    # 5 min cooldown (from 10)
    configs.append((a1, "A1_NoBlocks"))

    # ── A2: A1 + lower score thresholds (more signals)
    a2 = deepcopy(a1)
    a2["name"] = "A2_LowThresh"
    a2["put_score_min"] = 4.0
    a2["call_score_min"] = 5.0
    configs.append((a2, "A2_LowThresh"))

    # ── A3: A2 + relax RSI gates (more opportunities)
    a3 = deepcopy(a2)
    a3["name"] = "A3_RelaxRSI"
    a3["rsi_call_min"] = 50                    # From 60 -> 50
    a3["rsi_put_max"] = 50                     # From 40 -> 50
    configs.append((a3, "A3_RelaxRSI"))

    # ── A4: A3 + enable zero-to-hero + higher concurrent
    a4 = deepcopy(a3)
    a4["name"] = "A4_Z2H+Multi"
    a4["disable_zero_hero"] = False            # Enable zero-to-hero
    a4["max_concurrent"] = 5                   # From 3 -> 5
    a4["max_trades_per_day"] = 10              # From 7 -> 10
    configs.append((a4, "A4_Z2H+Multi"))

    # ── A5: A4 + soften VWAP filter (biggest remaining blocker)
    a5 = deepcopy(a4)
    a5["name"] = "A5_NoVWAP"
    a5["use_vwap_filter"] = False              # Let scoring handle VWAP bias
    configs.append((a5, "A5_NoVWAP"))

    # ── A6: A4 but keep VWAP, remove squeeze filter
    a6 = deepcopy(a4)
    a6["name"] = "A6_NoSqueeze"
    a6["use_squeeze_filter"] = False
    configs.append((a6, "A6_NoSqueeze"))

    # ── A7: Conservative market-driven (no day/time blocks but keep quality filters)
    a7 = deepcopy(V15_CONFIG)
    a7["name"] = "A7_SmartOpen"
    a7["avoid_days"] = []                      # No day blocks
    a7["entry_windows_bars"] = []              # No time windows
    a7["avoid_windows_bars"] = []              # No lunch block
    a7["block_late_entries"] = 72              # Block last 15 min (3:15 PM)
    a7["block_call_4th_hour"] = False
    a7["rsi_call_kill_ceiling"] = 72           # Fix ceiling but not too wide
    a7["vix_floor"] = 13
    a7["cooldown_bars"] = 2
    a7["put_score_min"] = 4.5                  # Slight relaxation
    a7["call_score_min"] = 5.5
    a7["rsi_call_min"] = 55
    a7["rsi_put_max"] = 45
    a7["disable_zero_hero"] = False            # Enable zero-to-hero
    a7["max_concurrent"] = 4
    a7["max_trades_per_day"] = 8
    configs.append((a7, "A7_SmartOpen"))

    # ── A8: A7 + lower thresholds
    a8 = deepcopy(a7)
    a8["name"] = "A8_SmartAggr"
    a8["put_score_min"] = 4.0
    a8["call_score_min"] = 5.0
    a8["rsi_call_min"] = 50
    a8["rsi_put_max"] = 50
    a8["max_concurrent"] = 5
    a8["max_trades_per_day"] = 10
    configs.append((a8, "A8_SmartAggr"))

    # Run all
    results = []
    for cfg, label in configs:
        print(f"  Running {label}...", end="", flush=True)
        result = run_backtest(cfg, label, start_date, end_date, vix_lookup, day_groups)
        results.append(result)
        print(f" {result['trades']} trades ({result['trades_per_day']:.1f}/day), "
              f"{result['return_x']:.2f}x, PF {result['profit_factor']:.2f}")

    print()
    print("=" * 110)
    print(f"{'Config':<16} {'Trades':>6} {'T/Day':>5} {'Days':>4} {'WR%':>6} {'P&L':>12} {'Return':>7} {'PF':>6} {'MaxDD':>7}")
    print("-" * 110)
    for r in results:
        marker = ""
        if r["trades_per_day"] >= 2.0 and r["profit_factor"] > 1.0:
            marker = " <--"
        elif r["trades_per_day"] >= 1.5 and r["profit_factor"] > 1.5:
            marker = " *"
        print(f"{r['label']:<16} {r['trades']:>6} {r['trades_per_day']:>5.1f} {r['trading_days']:>4} "
              f"{r['win_rate']:>5.1f}% {r['total_pnl']:>+11,.0f} {r['return_x']:>6.2f}x "
              f"{r['profit_factor']:>5.2f} {r['max_drawdown']:>6.1f}%{marker}")

    print()
    # Rank by P&L with PF > 1.0 constraint
    profitable = [r for r in results if r["profit_factor"] > 1.0]
    if profitable:
        by_pnl = sorted(profitable, key=lambda x: x["total_pnl"], reverse=True)
        print("  TOP 3 by P&L (PF > 1.0):")
        for i, r in enumerate(by_pnl[:3], 1):
            print(f"    {i}. {r['label']}: {r['trades_per_day']:.1f} trades/day, "
                  f"{r['return_x']:.2f}x, PF {r['profit_factor']:.2f}, MaxDD {r['max_drawdown']:.1f}%")

    # Find best balance of frequency + quality
    balanced = [r for r in results if r["trades_per_day"] >= 1.0 and r["profit_factor"] > 1.3]
    if balanced:
        best = max(balanced, key=lambda x: x["total_pnl"])
        print(f"\n  RECOMMENDED: {best['label']} -> {best['trades_per_day']:.1f} trades/day, "
              f"{best['return_x']:.2f}x return, PF {best['profit_factor']:.2f}")


if __name__ == "__main__":
    main()
