"""
V14 Advanced Tuner — Incorporate ALL research findings into model variants.
============================================================================
Tests 15+ improved configs with:
  - Full equity compounding (month-to-month)
  - Multiple trades per day (up to max_trades_per_day)
  - Lot sizing based on SPAN margin from compounded equity
  - 200K starting capital

Research findings incorporated:
  - VIX-adaptive strike selection (OTM in high VIX, ATM in low)
  - Hour-based entry windows (best: 9:30-10, 11-12)
  - Entry bar gating (skip bar 0 noise, skip lunch)
  - Wider trails for PUTs (2.0% vs 1.5%)
  - Time exit as primary (shorter max hold)
  - Scale-out exits
  - Day-of-week filters
  - ADX stronger filtering
  - Consecutive down day bias
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


def load_data(start_date, end_date):
    """Load NIFTY + VIX data and resample to 5-min."""
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
        raise FileNotFoundError(f"No data for {start_date} to {end_date}")
    nifty = pd.concat(all_dfs)
    nifty = nifty[~nifty.index.duplicated(keep="first")].sort_index()

    # Resample to 5-min
    n5 = nifty.resample("5min").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum"
    }).dropna(subset=["open"])
    n5 = n5[(n5.index.time >= dt.time(9, 15)) & (n5.index.time <= dt.time(15, 30))]

    # VIX
    vix_lookup = {}
    for f in sorted(DATA_DIR.glob("vix_min_*.csv")):
        try:
            df = pd.read_csv(f, parse_dates=["timestamp"], index_col="timestamp")
            mask = (df.index.date >= warmup_start) & (df.index.date <= end_date)
            chunk = df[mask]
            for d, group in chunk.groupby(chunk.index.date):
                vix_lookup[d] = float(group["close"].iloc[-1])
        except Exception:
            continue

    # Group by date
    day_groups = {}
    for ts, row in n5.iterrows():
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

    return day_groups, all_dates, warmup_bars, vix_lookup


def get_strike(action, spot, vix, zero_hero=False, vix_adaptive=False):
    """Strike selection — optionally VIX-adaptive."""
    opt_type = "CE" if action == "BUY_CALL" else "PE"
    atm = round(spot / STRIKE_INTERVAL) * STRIKE_INTERVAL
    if zero_hero:
        strike = atm + 200 if action == "BUY_CALL" else atm - 200
    elif vix_adaptive:
        if action == "BUY_CALL":
            offset = -50 if vix < 12 else (0 if vix < 16 else (100 if vix < 22 else 150))
        else:
            offset = 0 if vix < 12 else (50 if vix < 20 else 100)
        strike = atm + offset
    else:
        strike = atm
    return strike, opt_type


def calc_premium(spot, strike, dte, vix, opt_type, slippage_sign=1):
    try:
        result = price_option(spot=spot, strike=strike, dte_days=max(0.1, dte), vix=vix, option_type=opt_type)
        prem = result["premium"]
    except Exception:
        prem = 30.0
    prem = prem * (1 + slippage_sign * SLIPPAGE_PCT) + slippage_sign * SPREAD_RS
    return max(0.05, prem)


def simulate_day_advanced(bars, date, vix, cfg, prev_close, equity, warmup_bars,
                          is_expiry, consecutive_down_days):
    """Advanced day simulation with all research improvements."""
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

    # SPAN margin lot sizing (compounding)
    span_per_lot = 40000 if vix < 20 else (50000 if vix < 25 else 60000)
    base_lots = max(1, int(equity * 0.70 / span_per_lot))

    # DTE approximation (Thursday expiry for 2024)
    days_to_expiry = (3 - day_of_week) % 7
    dte = max(0.2, days_to_expiry) if days_to_expiry > 0 else 0.2

    vix_adaptive_strikes = cfg.get("vix_adaptive_strikes", False)
    entry_windows = cfg.get("entry_windows_bars", None)  # List of (start, end) allowed windows
    skip_first_bar = cfg.get("skip_first_bar", False)
    scale_out = cfg.get("scale_out", False)

    day_pnl = 0.0

    for bar_idx, bar in enumerate(bars):
        bar_history.append(bar)
        if len(bar_history) > 500:
            bar_history = bar_history[-500:]
        spot = bar["close"]

        # ORB tracking
        if bar_idx == 0:
            orb_high = bar["high"]; orb_low = bar["low"]
        elif bar_idx == 1:
            orb_high = max(orb_high, bar["high"])
            orb_low = min(orb_low, bar["low"])

        # ── EXITS ──
        for trade in list(open_trades):
            if trade["action"] == "BUY_CALL" and spot > trade["best_fav"]:
                trade["best_fav"] = spot
            elif trade["action"] == "BUY_PUT" and spot < trade["best_fav"]:
                trade["best_fav"] = spot

            indicators = compute_indicators(bar_history, date_str)
            exit_reason = evaluate_exit(trade, bar_idx, spot, indicators or {}, cfg, day_of_week=day_of_week)

            # Scale-out: partial exit at 50% of position
            if exit_reason and scale_out and trade.get("qty", 0) > LOT_SIZE and not trade.get("scaled_out"):
                partial_qty = trade["qty"] // 2
                partial_prem = calc_premium(spot, trade["strike"],
                                            max(0.05, dte - bar_idx * 5 / (6.25 * 60)),
                                            vix, trade["opt_type"], -1)
                partial_pnl = (partial_prem - trade["entry_premium"]) * partial_qty - BROKERAGE_RT / 2
                day_pnl += partial_pnl
                trade["qty"] -= partial_qty
                trade["scaled_out"] = True
                closed_trades.append({
                    **trade, "exit_bar": bar_idx, "exit_spot": spot,
                    "exit_premium": partial_prem, "exit_reason": f"{exit_reason}_partial",
                    "pnl": partial_pnl, "bars_held": bar_idx - trade["entry_bar"],
                    "qty": partial_qty,
                })
                continue  # Keep remainder open

            if exit_reason:
                exit_prem = calc_premium(spot, trade["strike"],
                                         max(0.05, dte - bar_idx * 5 / (6.25 * 60)),
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

        # Skip first bar (gap noise) if configured
        if skip_first_bar and bar_idx == 0:
            if bar_idx == 0:
                gap_detected = True
            continue

        # Entry window gating
        if entry_windows:
            in_window = any(s <= bar_idx <= e for s, e in entry_windows)
            if not in_window and bar_idx >= 3:
                continue

        action = None; conf = 0; entry_type = "v8_indicator"; is_zero_hero = False

        if bar_idx < 3:
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
            if bar_idx - last_exit_bar < cfg["cooldown_bars"]:
                continue
            action, conf = score_entry(
                indicators, vix, cfg, bar_idx=bar_idx,
                consecutive_down_days=consecutive_down_days,
            )
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

        strike, opt_type = get_strike(action, spot, vix, zero_hero=is_zero_hero,
                                       vix_adaptive=vix_adaptive_strikes)
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

    # Force close remaining
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


def make_variants():
    """Build 15+ config variants using all research findings."""
    v = {}
    base = copy.deepcopy(V14_CONFIG)  # This is V5_RSI6040 (the current winner at 3.87x)

    # ── V0: Current best (V5_RSI6040) — baseline ──
    v["V0_Current"] = base

    # ── V1: + VIX-adaptive strikes (research: OTM in high VIX, ATM in low) ──
    c = copy.deepcopy(base); c["vix_adaptive_strikes"] = True
    v["V1_VIXStrikes"] = c

    # ── V2: + Wider PUT trail (research: 2.0% PUT trail better than 1.5%) ──
    c = copy.deepcopy(base)
    c["trail_pct_put"] = 0.020; c["trail_pct_call"] = 0.012
    c["min_hold_trail_put"] = 30; c["min_hold_trail_call"] = 18
    v["V2_WiderTrail"] = c

    # ── V3: + Entry windows (research: 9:30-10, 11-12, 14:00-15:00 best) ──
    c = copy.deepcopy(base)
    c["entry_windows_bars"] = [(3, 12), (21, 33), (57, 69)]  # 9:30-10, 11-12, 14:00-15:00
    v["V3_EntryWindows"] = c

    # ── V4: + Skip first bar (research: gap noise at 9:15) ──
    c = copy.deepcopy(base); c["skip_first_bar"] = True
    v["V4_SkipBar0"] = c

    # ── V5: + Shorter max hold (research: time exit at 45 min = best) ──
    c = copy.deepcopy(base)
    c["max_hold_put"] = 40; c["max_hold_call"] = 36  # 200/180 min
    v["V5_ShorterHold"] = c

    # ── V6: + Higher max trades (research: allow more entries with compounding) ──
    c = copy.deepcopy(base)
    c["max_trades_per_day"] = 10; c["max_concurrent"] = 4
    v["V6_MoreTrades"] = c

    # ── V7: V1+V2+V3 combined ──
    c = copy.deepcopy(base)
    c["vix_adaptive_strikes"] = True
    c["trail_pct_put"] = 0.020; c["trail_pct_call"] = 0.012
    c["min_hold_trail_put"] = 30; c["min_hold_trail_call"] = 18
    c["entry_windows_bars"] = [(3, 12), (21, 33), (57, 69)]
    v["V7_Combined123"] = c

    # ── V8: V7 + more trades + shorter hold ──
    c = copy.deepcopy(v["V7_Combined123"])
    c["max_trades_per_day"] = 10; c["max_concurrent"] = 4
    c["max_hold_put"] = 45; c["max_hold_call"] = 40
    v["V8_Aggressive"] = c

    # ── V9: V7 + scale-out exits ──
    c = copy.deepcopy(v["V7_Combined123"])
    c["scale_out"] = True
    v["V9_ScaleOut"] = c

    # ── V10: V8 + VIX ceil raised to 40 (research: VIX>25 = contrarian buy) ──
    c = copy.deepcopy(v["V8_Aggressive"])
    c["vix_ceil"] = 40
    v["V10_HighVIX"] = c

    # ── V11: V7 + no lunch block (research: lunch block removes 13% trades) ──
    c = copy.deepcopy(v["V7_Combined123"])
    c["avoid_windows_bars"] = []
    v["V11_NoLunch"] = c

    # ── V12: V7 + no trail stops (research: trail_stop 0% WR) ──
    c = copy.deepcopy(v["V7_Combined123"])
    c["min_hold_trail_put"] = 999; c["min_hold_trail_call"] = 999
    v["V12_NoTrail"] = c

    # ── V13: V7 + RSI 55/45 (middle ground between 60/40 and 50/50) ──
    c = copy.deepcopy(v["V7_Combined123"])
    c["rsi_call_min"] = 55; c["rsi_put_max"] = 45
    v["V13_RSI5545"] = c

    # ── V14: Ultra aggressive — all improvements + max trades + wider VIX ──
    c = copy.deepcopy(base)
    c["vix_adaptive_strikes"] = True
    c["trail_pct_put"] = 0.020; c["trail_pct_call"] = 0.012
    c["min_hold_trail_put"] = 30; c["min_hold_trail_call"] = 18
    c["max_trades_per_day"] = 12; c["max_concurrent"] = 5
    c["max_hold_put"] = 45; c["max_hold_call"] = 40
    c["vix_ceil"] = 40
    c["entry_windows_bars"] = [(3, 12), (21, 33), (57, 69)]
    c["scale_out"] = True
    v["V14_Ultra"] = c

    # ── V15: Conservative — V7 + max_concurrent=1 + lower score thresholds ──
    c = copy.deepcopy(v["V7_Combined123"])
    c["max_concurrent"] = 1
    c["put_score_min"] = 6.0; c["call_score_min"] = 7.0
    v["V15_Conservative"] = c

    return v


def run_variant(name, cfg, day_groups, all_dates, warmup_bars_init, vix_lookup):
    """Run with full equity compounding."""
    equity = CAPITAL
    all_trades = []
    prev_close = 0.0
    consecutive_down_days = 0
    warmup_bars = list(warmup_bars_init)
    monthly_pnl = defaultdict(float)
    max_equity = equity
    max_dd = 0.0

    for date in all_dates:
        bars = day_groups[date]
        if len(bars) < 5:
            continue
        vix = vix_lookup.get(date, 14.0)
        is_expiry = (date.weekday() == 3)

        trades, day_pnl, eod_close = simulate_day_advanced(
            bars, date, vix, cfg, prev_close, equity,
            warmup_bars, is_expiry, consecutive_down_days,
        )

        equity += day_pnl
        all_trades.extend(trades)
        month_key = f"{date.year}-{date.month:02d}"
        monthly_pnl[month_key] += day_pnl

        # Track max drawdown
        if equity > max_equity:
            max_equity = equity
        dd = (max_equity - equity) / max_equity if max_equity > 0 else 0
        if dd > max_dd:
            max_dd = dd

        if len(bars) >= 2:
            consecutive_down_days = consecutive_down_days + 1 if bars[-1]["close"] < bars[0]["open"] else 0

        warmup_bars = warmup_bars[-(75 * 2):] + bars
        prev_close = eod_close

    total_pnl = sum(t.get("pnl", 0) for t in all_trades)
    wins = [t for t in all_trades if t.get("pnl", 0) > 0]
    wr = len(wins) / len(all_trades) * 100 if all_trades else 0
    pf_num = sum(t["pnl"] for t in wins) if wins else 0
    pf_den = abs(sum(t["pnl"] for t in all_trades if t["pnl"] <= 0)) + 1
    pf = pf_num / pf_den

    return {
        "name": name, "trades": len(all_trades), "wins": len(wins),
        "win_rate": wr, "total_pnl": total_pnl, "equity": equity,
        "return_x": equity / CAPITAL, "profit_factor": pf,
        "max_dd_pct": max_dd * 100, "monthly_pnl": dict(monthly_pnl),
    }


def main():
    start_date = dt.date(2024, 7, 1)
    end_date = dt.date(2025, 1, 1)

    print("V14 Advanced Tuner — All Research Improvements")
    print("=" * 80)
    print(f"Period: {start_date} to {end_date} (6 months)")
    print(f"Capital: Rs {CAPITAL:,.0f} | Lot size: {LOT_SIZE}")
    print(f"Equity COMPOUNDED month-to-month")
    print()

    print("Loading data...", flush=True)
    day_groups, all_dates, warmup_bars, vix_lookup = load_data(start_date, end_date)
    print(f"Trading days: {len(all_dates)}")
    print()

    variants = make_variants()
    results = []

    for name, cfg in variants.items():
        print(f"  {name}...", end="", flush=True)
        r = run_variant(name, cfg, day_groups, all_dates, warmup_bars, vix_lookup)
        results.append(r)
        print(f" {r['trades']} trades | {r['win_rate']:.1f}% WR | "
              f"Rs {r['total_pnl']:+,.0f} | {r['return_x']:.2f}x | "
              f"PF {r['profit_factor']:.2f} | MaxDD {r['max_dd_pct']:.1f}%")

    results.sort(key=lambda x: x["total_pnl"], reverse=True)

    print()
    print("=" * 105)
    print(f"{'Rank':<4} {'Config':<25} {'Trades':>6} {'WR':>6} {'PnL':>14} {'Return':>8} {'PF':>6} {'MaxDD':>7}")
    print("-" * 105)
    for i, r in enumerate(results, 1):
        print(f"{i:<4} {r['name']:<25} {r['trades']:>6} {r['win_rate']:>5.1f}% "
              f"{r['total_pnl']:>+14,.0f} {r['return_x']:>7.2f}x {r['profit_factor']:>6.2f} "
              f"{r['max_dd_pct']:>6.1f}%")

    # Show monthly breakdown for top 3
    print()
    print("=" * 80)
    print("MONTHLY P&L — TOP 3 MODELS")
    print("=" * 80)
    for r in results[:3]:
        print(f"\n{r['name']} ({r['return_x']:.2f}x, Rs {r['total_pnl']:+,.0f}):")
        cum = CAPITAL
        for m in sorted(r["monthly_pnl"]):
            cum += r["monthly_pnl"][m]
            print(f"  {m}: Rs {r['monthly_pnl'][m]:>+12,.0f}  (Equity: Rs {cum:>12,.0f})")

    best = results[0]
    print()
    print(f"WINNER: {best['name']}")
    print(f"  Return: {best['return_x']:.2f}x ({(best['return_x']-1)*100:.0f}%)")
    print(f"  P&L: Rs {best['total_pnl']:+,.0f}")
    print(f"  Trades: {best['trades']} | WR: {best['win_rate']:.1f}%")
    print(f"  Profit Factor: {best['profit_factor']:.2f}")
    print(f"  Max Drawdown: {best['max_dd_pct']:.1f}%")

    # Save best config name for deployment
    with open(DATA_DIR / "best_v14_config.txt", "w") as f:
        f.write(best["name"])

    return results


if __name__ == "__main__":
    main()
