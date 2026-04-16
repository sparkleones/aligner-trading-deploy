"""
V14 1-Min vs 5-Min Bar Resolution Comparison.
==============================================
Tests the SAME scoring engine on both 1-min and 5-min bars
to determine which resolution is better.

For 1-min bars, config values in bar_idx units are scaled by 5x
(since 1 bar = 1 min instead of 5 min).
"""

import sys
import copy
import datetime as dt
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scoring.config import V14_CONFIG
from scoring.indicators import compute_indicators
from scoring.engine import (
    score_entry, passes_confluence, evaluate_exit,
    compute_lots, detect_composite_entries,
)
from backtesting.option_pricer import price_option

DATA_DIR = project_root / "data" / "historical"
CAPITAL = 200_000
LOT_SIZE = 75
STRIKE_INTERVAL = 50
SLIPPAGE_PCT = 0.005
SPREAD_RS = 2.0
BROKERAGE_RT = 80.0


def load_nifty_1min(start_date, end_date):
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
    if not all_dfs:
        raise FileNotFoundError("No data")
    nifty = pd.concat(all_dfs)
    nifty = nifty[~nifty.index.duplicated(keep="first")].sort_index()
    nifty = nifty[(nifty.index.time >= dt.time(9, 15)) & (nifty.index.time <= dt.time(15, 30))]
    return nifty


def load_vix(start_date, end_date):
    warmup_start = start_date - dt.timedelta(days=10)
    vix_lookup = {}
    for f in sorted(DATA_DIR.glob("vix_min_*.csv")):
        try:
            df = pd.read_csv(f, parse_dates=["timestamp"], index_col="timestamp")
            mask = (df.index.date >= warmup_start) & (df.index.date <= end_date)
            chunk = df[mask]
            for d, g in chunk.groupby(chunk.index.date):
                vix_lookup[d] = float(g["close"].iloc[-1])
        except Exception:
            continue
    return vix_lookup


def calc_premium(spot, strike, dte, vix, opt_type, slippage_sign=1):
    try:
        result = price_option(spot=spot, strike=strike, dte_days=max(0.1, dte),
                              vix=vix, option_type=opt_type)
        prem = result["premium"]
    except Exception:
        prem = 30.0
    prem = prem * (1 + slippage_sign * SLIPPAGE_PCT) + slippage_sign * SPREAD_RS
    return max(0.05, prem)


def scale_config_for_1min(cfg):
    """Scale bar_idx-based config values by 5x for 1-min bars."""
    c = copy.deepcopy(cfg)
    c["cooldown_bars"] = c["cooldown_bars"] * 5
    c["min_hold_trail_put"] = c["min_hold_trail_put"] * 5
    c["min_hold_trail_call"] = c["min_hold_trail_call"] * 5
    c["max_hold_put"] = c["max_hold_put"] * 5
    c["max_hold_call"] = c["max_hold_call"] * 5
    c["block_late_entries"] = c["block_late_entries"] * 5
    c["eod_close_bar"] = c["eod_close_bar"] * 5
    if "theta_exit_monday_bar" in c:
        c["theta_exit_monday_bar"] = c["theta_exit_monday_bar"] * 5
    if "zero_hero_time_bars" in c:
        c["zero_hero_time_bars"] = c["zero_hero_time_bars"] * 5
    # Scale avoid_windows
    if c.get("avoid_windows_bars"):
        c["avoid_windows_bars"] = [(s * 5, e * 5) for s, e in c["avoid_windows_bars"]]
    # Scale entry windows
    if c.get("entry_windows_bars"):
        c["entry_windows_bars"] = [(s * 5, e * 5) for s, e in c["entry_windows_bars"]]
    return c


def simulate_day(bars, date, vix, cfg, prev_close, equity, warmup_bars,
                 is_expiry, consecutive_down_days, is_1min=False):
    """Run day simulation — works for both 1-min and 5-min bars."""
    date_str = str(date)
    bar_history = list(warmup_bars or [])
    open_trades = []
    closed_trades = []
    trades_today = 0
    last_exit_bar = -10
    orb_high = orb_low = 0.0
    gap_detected = False
    day_open = bars[0]["close"] if bars else 0
    day_of_week = date.weekday()

    span_per_lot = 40000 if vix < 20 else (50000 if vix < 25 else 60000)
    base_lots = max(1, int(equity * 0.70 / span_per_lot))

    days_to_expiry = (3 - day_of_week) % 7
    dte = max(0.2, days_to_expiry) if days_to_expiry > 0 else 0.2

    # For 1-min: composite entries happen in first 15 min (bar 0-14)
    # For 5-min: composite entries happen in first 3 bars (bar 0-2)
    composite_cutoff = 15 if is_1min else 3

    day_pnl = 0.0

    for bar_idx, bar in enumerate(bars):
        bar_history.append(bar)
        if len(bar_history) > 500:
            bar_history = bar_history[-500:]
        spot = bar["close"]

        # ORB tracking
        if bar_idx == 0:
            orb_high = bar["high"]; orb_low = bar["low"]
        elif bar_idx < composite_cutoff:
            orb_high = max(orb_high, bar["high"])
            orb_low = min(orb_low, bar["low"])

        # ── EXITS ──
        for trade in list(open_trades):
            if trade["action"] == "BUY_CALL" and spot > trade["best_fav"]:
                trade["best_fav"] = spot
            elif trade["action"] == "BUY_PUT" and spot < trade["best_fav"]:
                trade["best_fav"] = spot

            indicators = compute_indicators(bar_history, date_str)
            exit_reason = evaluate_exit(trade, bar_idx, spot, indicators or {}, cfg,
                                         day_of_week=day_of_week)
            if exit_reason:
                exit_prem = calc_premium(spot, trade["strike"],
                                          max(0.05, dte - bar_idx / (375 if is_1min else 75)),
                                          vix, trade["opt_type"], -1)
                pnl = (exit_prem - trade["entry_premium"]) * trade["qty"] - BROKERAGE_RT
                trade.update({"exit_bar": bar_idx, "exit_spot": spot, "exit_premium": exit_prem,
                               "exit_reason": exit_reason, "pnl": pnl,
                               "bars_held": bar_idx - trade["entry_bar"]})
                closed_trades.append(trade)
                open_trades.remove(trade)
                day_pnl += pnl
                last_exit_bar = bar_idx

        # ── ENTRIES ──
        indicators = compute_indicators(bar_history, date_str)
        if indicators is None:
            continue
        if len(open_trades) >= cfg["max_concurrent"]:
            continue
        if trades_today >= cfg["max_trades_per_day"]:
            continue

        # Entry window gating
        entry_windows = cfg.get("entry_windows_bars")

        action = None; conf = 0; entry_type = "v8_indicator"; is_zero_hero = False

        if bar_idx < composite_cutoff:
            prev_spot = bars[bar_idx - 1]["close"] if bar_idx > 0 else spot
            composites = detect_composite_entries(
                bar, bar_idx, spot, vix, cfg,
                prev_close=prev_close, gap_detected=gap_detected,
                orb_high=orb_high, orb_low=orb_low, prev_spot=prev_spot,
            )
            if bar_idx == 0:
                gap_detected = True
            if composites:
                composites.sort(key=lambda x: x[2], reverse=True)
                action, entry_type, conf, is_zero_hero = composites[0]
        else:
            # Entry window check
            if entry_windows:
                in_window = any(s <= bar_idx <= e for s, e in entry_windows)
                if not in_window:
                    continue

            if bar_idx - last_exit_bar < cfg["cooldown_bars"]:
                continue

            action, conf = score_entry(indicators, vix, cfg, bar_idx=bar_idx,
                                        consecutive_down_days=consecutive_down_days)
            if action is None:
                prev_spot = bars[bar_idx - 1]["close"] if bar_idx > 0 else spot
                composites = detect_composite_entries(
                    bar, bar_idx, spot, vix, cfg,
                    prev_close=prev_close, gap_detected=gap_detected,
                    orb_high=orb_high, orb_low=orb_low, prev_spot=prev_spot,
                )
                if composites:
                    composites.sort(key=lambda x: x[2], reverse=True)
                    action, entry_type, conf, is_zero_hero = composites[0]

        if action is None:
            continue
        if vix < cfg["vix_floor"] or vix > cfg["vix_ceil"]:
            continue
        if not passes_confluence(action, conf, indicators, bar_idx, cfg,
                                  current_spot=spot, prev_close=prev_close, day_open=day_open):
            continue
        if any(t["action"] == action for t in open_trades):
            continue

        lots = compute_lots(action, conf, vix, indicators.get("rsi", 50),
                            is_expiry, base_lots, cfg)
        if is_zero_hero:
            lots = min(cfg.get("zero_hero_max_lots", 3), max(1, lots))

        opt_type = "CE" if action == "BUY_CALL" else "PE"
        atm = round(spot / STRIKE_INTERVAL) * STRIKE_INTERVAL
        strike = (atm + 200 if action == "BUY_CALL" else atm - 200) if is_zero_hero else atm
        qty = lots * LOT_SIZE
        entry_prem = calc_premium(spot, strike, dte, vix, opt_type, 1)

        trade = {
            "date": date_str, "action": action, "entry_bar": bar_idx,
            "entry_spot": spot, "best_fav": spot, "strike": strike,
            "opt_type": opt_type, "lots": lots, "qty": qty,
            "entry_premium": entry_prem, "entry_type": entry_type,
            "is_zero_hero": is_zero_hero, "confidence": conf, "vix": vix,
        }
        open_trades.append(trade)
        trades_today += 1

    # Force close
    eod_spot = bars[-1]["close"] if bars else 0
    for trade in open_trades:
        exit_prem = calc_premium(eod_spot, trade["strike"], max(0.05, dte * 0.1),
                                  vix, trade["opt_type"], -1)
        pnl = (exit_prem - trade["entry_premium"]) * trade["qty"] - BROKERAGE_RT
        trade.update({"exit_bar": len(bars) - 1, "exit_spot": eod_spot,
                       "exit_premium": exit_prem, "exit_reason": "eod_close",
                       "pnl": pnl, "bars_held": len(bars) - 1 - trade["entry_bar"]})
        closed_trades.append(trade)
        day_pnl += pnl

    return closed_trades, day_pnl, eod_spot


def run_test(label, nifty_df, start_date, end_date, vix_lookup, cfg, is_1min=False):
    """Run full backtest with equity compounding."""
    resample = "1min" if is_1min else "5min"
    if not is_1min:
        nifty_df = nifty_df.resample("5min").agg({
            "open": "first", "high": "max", "low": "min",
            "close": "last", "volume": "sum"
        }).dropna(subset=["open"])
        nifty_df = nifty_df[(nifty_df.index.time >= dt.time(9, 15)) &
                            (nifty_df.index.time <= dt.time(15, 30))]

    # Group by date
    day_groups = {}
    for ts, row in nifty_df.iterrows():
        d = ts.date()
        if d not in day_groups:
            day_groups[d] = []
        day_groups[d].append({
            "open": float(row["open"]), "high": float(row["high"]),
            "low": float(row["low"]), "close": float(row["close"]),
            "volume": float(row.get("volume", 0)),
            "date": str(d), "time": str(ts),
        })

    all_dates = sorted(d for d in day_groups if d >= start_date)
    warmup_dates = sorted(d for d in day_groups if d < start_date)
    warmup_bars = []
    for d in warmup_dates[-3:]:
        warmup_bars.extend(day_groups[d])

    equity = CAPITAL
    all_trades = []
    prev_close = 0.0
    consecutive_down_days = 0
    monthly_pnl = defaultdict(float)
    max_equity = equity
    max_dd = 0.0

    for date in all_dates:
        bars = day_groups[date]
        if len(bars) < (20 if is_1min else 5):
            continue
        vix = vix_lookup.get(date, 14.0)
        is_expiry = (date.weekday() == 3)

        trades, day_pnl, eod_close = simulate_day(
            bars, date, vix, cfg, prev_close, equity,
            warmup_bars, is_expiry, consecutive_down_days, is_1min=is_1min,
        )

        equity += day_pnl
        all_trades.extend(trades)
        monthly_pnl[f"{date.year}-{date.month:02d}"] += day_pnl

        if equity > max_equity:
            max_equity = equity
        dd = (max_equity - equity) / max_equity if max_equity > 0 else 0
        if dd > max_dd:
            max_dd = dd

        consecutive_down_days = consecutive_down_days + 1 if len(bars) >= 2 and bars[-1]["close"] < bars[0]["open"] else 0
        warmup_bars = warmup_bars[-(375 * 2 if is_1min else 75 * 2):] + bars
        prev_close = eod_close

    total_pnl = sum(t.get("pnl", 0) for t in all_trades)
    wins = [t for t in all_trades if t.get("pnl", 0) > 0]
    wr = len(wins) / len(all_trades) * 100 if all_trades else 0
    pf_num = sum(t["pnl"] for t in wins) if wins else 0
    pf_den = abs(sum(t["pnl"] for t in all_trades if t["pnl"] <= 0)) + 1

    return {
        "name": label, "trades": len(all_trades), "wins": len(wins),
        "win_rate": wr, "total_pnl": total_pnl, "equity": equity,
        "return_x": equity / CAPITAL, "profit_factor": pf_num / pf_den,
        "max_dd_pct": max_dd * 100, "monthly_pnl": dict(monthly_pnl),
    }


def main():
    start_date = dt.date(2024, 7, 1)
    end_date = dt.date(2025, 1, 1)

    print("V14 — 1-Min vs 5-Min Bar Resolution Comparison")
    print("=" * 90)
    print(f"Period: {start_date} to {end_date} (6 months)")
    print(f"Capital: Rs {CAPITAL:,.0f} | Equity COMPOUNDED")
    print()

    print("Loading 1-min data...", flush=True)
    nifty_1min = load_nifty_1min(start_date, end_date)
    vix_lookup = load_vix(start_date, end_date)
    print(f"1-min bars: {len(nifty_1min):,}")
    print()

    # ── Test 1: 5-min with entry windows (current R4 = baseline 3.48x) ──
    cfg_5min = copy.deepcopy(V14_CONFIG)
    print("Test 1: 5-Min + Entry Windows (R4 baseline)...", flush=True)
    r1 = run_test("5min_R4_Windows", nifty_1min.copy(), start_date, end_date,
                  vix_lookup, cfg_5min, is_1min=False)
    print(f"  {r1['trades']} trades | {r1['win_rate']:.1f}% WR | "
          f"Rs {r1['total_pnl']:+,.0f} | {r1['return_x']:.2f}x | PF {r1['profit_factor']:.2f}")

    # ── Test 2: 5-min without entry windows ──
    cfg_5min_nw = copy.deepcopy(V14_CONFIG)
    cfg_5min_nw["entry_windows_bars"] = None
    print("Test 2: 5-Min + No Windows...", flush=True)
    r2 = run_test("5min_NoWindows", nifty_1min.copy(), start_date, end_date,
                  vix_lookup, cfg_5min_nw, is_1min=False)
    print(f"  {r2['trades']} trades | {r2['win_rate']:.1f}% WR | "
          f"Rs {r2['total_pnl']:+,.0f} | {r2['return_x']:.2f}x | PF {r2['profit_factor']:.2f}")

    # ── Test 3: 1-min with entry windows (scaled) ──
    cfg_1min = scale_config_for_1min(V14_CONFIG)
    print("Test 3: 1-Min + Entry Windows (scaled)...", flush=True)
    r3 = run_test("1min_R4_Windows", nifty_1min.copy(), start_date, end_date,
                  vix_lookup, cfg_1min, is_1min=True)
    print(f"  {r3['trades']} trades | {r3['win_rate']:.1f}% WR | "
          f"Rs {r3['total_pnl']:+,.0f} | {r3['return_x']:.2f}x | PF {r3['profit_factor']:.2f}")

    # ── Test 4: 1-min without entry windows ──
    cfg_1min_nw = scale_config_for_1min(V14_CONFIG)
    cfg_1min_nw["entry_windows_bars"] = None
    print("Test 4: 1-Min + No Windows...", flush=True)
    r4 = run_test("1min_NoWindows", nifty_1min.copy(), start_date, end_date,
                  vix_lookup, cfg_1min_nw, is_1min=True)
    print(f"  {r4['trades']} trades | {r4['win_rate']:.1f}% WR | "
          f"Rs {r4['total_pnl']:+,.0f} | {r4['return_x']:.2f}x | PF {r4['profit_factor']:.2f}")

    # Summary
    print()
    print("=" * 100)
    print(f"{'Test':<25} {'Trades':>6} {'WR':>6} {'PnL':>14} {'Return':>8} {'PF':>6} {'MaxDD':>7}")
    print("-" * 100)
    for r in [r1, r2, r3, r4]:
        print(f"{r['name']:<25} {r['trades']:>6} {r['win_rate']:>5.1f}% "
              f"{r['total_pnl']:>+14,.0f} {r['return_x']:>7.2f}x {r['profit_factor']:>6.2f} "
              f"{r['max_dd_pct']:>6.1f}%")

    # Monthly for best
    results = sorted([r1, r2, r3, r4], key=lambda x: x["total_pnl"], reverse=True)
    best = results[0]
    print(f"\nBEST: {best['name']} — {best['return_x']:.2f}x | Rs {best['total_pnl']:+,.0f}")
    cum = CAPITAL
    for m in sorted(best["monthly_pnl"]):
        cum += best["monthly_pnl"][m]
        print(f"  {m}: Rs {best['monthly_pnl'][m]:>+12,.0f}  (Equity: Rs {cum:>12,.0f})")


if __name__ == "__main__":
    main()
