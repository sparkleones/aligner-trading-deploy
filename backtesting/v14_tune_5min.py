"""
V14 5-Min Config Tuner — find the most profitable config on 6 months data.
Tests multiple config variations using the shared scoring engine.
"""

import sys
import copy
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scoring.config import V14_CONFIG
from backtesting.v14_unified_backtest import (
    load_nifty_data, load_vix_data, resample_to_5min,
    simulate_day, CAPITAL, LOT_SIZE,
)

DATA_DIR = project_root / "data" / "historical"


def make_variants():
    """Create config variants to test."""
    variants = {}

    # ── V0: Baseline (current shared config) ──
    variants["V0_Baseline"] = copy.deepcopy(V14_CONFIG)

    # ── V1: Disable gap entries (gap_entry lost -144K) ──
    # Keep Z2H gaps and ORB, disable regular gap_entry
    cfg1 = copy.deepcopy(V14_CONFIG)
    cfg1["name"] = "V1_NoGapEntry"
    cfg1["disable_gap_entry"] = True  # Flag for backtester
    variants["V1_NoGapEntry"] = cfg1

    # ── V2: Tighter scoring thresholds ──
    cfg2 = copy.deepcopy(V14_CONFIG)
    cfg2["name"] = "V2_TighterScores"
    cfg2["put_score_min"] = 5.0   # Was 4.0
    cfg2["call_score_min"] = 6.0  # Was 5.0
    variants["V2_TighterScores"] = cfg2

    # ── V3: V2 + No gap entry ──
    cfg3 = copy.deepcopy(cfg2)
    cfg3["name"] = "V3_Tight+NoGap"
    cfg3["disable_gap_entry"] = True
    variants["V3_Tight+NoGap"] = cfg3

    # ── V4: V3 + Wider trail + longer hold ──
    cfg4 = copy.deepcopy(cfg3)
    cfg4["name"] = "V4_WiderTrail"
    cfg4["trail_pct_put"] = 0.020   # Was 0.015
    cfg4["trail_pct_call"] = 0.012  # Was 0.008
    cfg4["min_hold_trail_put"] = 30  # Was 24 (150 min)
    cfg4["min_hold_trail_call"] = 18  # Was 12 (90 min)
    variants["V4_WiderTrail"] = cfg4

    # ── V5: V3 + RSI gate tightened (60/40 like original backtest) ──
    cfg5 = copy.deepcopy(cfg3)
    cfg5["name"] = "V5_RSI6040"
    cfg5["rsi_call_min"] = 60
    cfg5["rsi_put_max"] = 40
    variants["V5_RSI6040"] = cfg5

    # ── V6: V3 + No lunch block (avoid_windows empty) ──
    cfg6 = copy.deepcopy(cfg3)
    cfg6["name"] = "V6_NoLunchBlock"
    cfg6["avoid_windows_bars"] = []
    variants["V6_NoLunchBlock"] = cfg6

    # ── V7: V3 + Disable trail stops entirely (let time_exit handle) ──
    cfg7 = copy.deepcopy(cfg3)
    cfg7["name"] = "V7_NoTrail"
    cfg7["min_hold_trail_put"] = 999   # Never trail
    cfg7["min_hold_trail_call"] = 999
    variants["V7_NoTrail"] = cfg7

    # ── V8: V7 + ADX filter stricter (only trade ADX > 25) ──
    cfg8 = copy.deepcopy(cfg7)
    cfg8["name"] = "V8_ADX25+NoTrail"
    cfg8["adx_weak_threshold"] = 25   # Was 18
    cfg8["adx_weak_mult"] = 0.3       # Was 0.6 — heavy dampening
    variants["V8_ADX25+NoTrail"] = cfg8

    # ── V9: Best of V3 + max concurrent = 1 (focus quality) ──
    cfg9 = copy.deepcopy(cfg3)
    cfg9["name"] = "V9_Single"
    cfg9["max_concurrent"] = 1
    variants["V9_Single"] = cfg9

    # ── V10: V3 + No squeeze filter (squeeze was blocking entries) ──
    cfg10 = copy.deepcopy(cfg3)
    cfg10["name"] = "V10_NoSqueeze"
    cfg10["use_squeeze_filter"] = False
    variants["V10_NoSqueeze"] = cfg10

    return variants


def run_variant(name, cfg, day_groups, all_dates, warmup_bars_init, vix_lookup):
    """Run one config variant through the full backtest."""
    equity = CAPITAL
    all_trades = []
    prev_close = 0.0
    consecutive_down_days = 0
    warmup_bars = list(warmup_bars_init)

    for date in all_dates:
        bars = day_groups[date]
        if len(bars) < 5:
            continue

        vix = vix_lookup.get(date, 14.0)
        is_expiry = (date.weekday() == 3)

        # Filter out gap_entry if disabled
        trades, day_pnl, eod_close = simulate_day(
            bars, date, vix, cfg,
            prev_close=prev_close,
            equity=equity,
            warmup_bars=warmup_bars,
            is_expiry=is_expiry,
            consecutive_down_days=consecutive_down_days,
        )

        # If this config disables gap_entry, filter those out
        if cfg.get("disable_gap_entry"):
            kept = [t for t in trades if t.get("entry_type") != "gap_entry"]
            removed_pnl = sum(t.get("pnl", 0) for t in trades if t.get("entry_type") == "gap_entry")
            trades = kept
            day_pnl -= removed_pnl

        equity += day_pnl
        all_trades.extend(trades)

        if len(bars) >= 2:
            if bars[-1]["close"] < bars[0]["open"]:
                consecutive_down_days += 1
            else:
                consecutive_down_days = 0

        warmup_bars = warmup_bars[-(75 * 2):] + bars
        prev_close = eod_close

    total_pnl = sum(t.get("pnl", 0) for t in all_trades)
    wins = [t for t in all_trades if t.get("pnl", 0) > 0]
    win_rate = len(wins) / len(all_trades) * 100 if all_trades else 0
    pf_num = sum(t["pnl"] for t in wins)
    pf_den = abs(sum(t["pnl"] for t in all_trades if t["pnl"] <= 0)) + 1
    profit_factor = pf_num / pf_den
    ret_x = equity / CAPITAL

    return {
        "name": name,
        "trades": len(all_trades),
        "wins": len(wins),
        "win_rate": win_rate,
        "total_pnl": total_pnl,
        "equity": equity,
        "return_x": ret_x,
        "profit_factor": profit_factor,
    }


def main():
    start_date = dt.date(2024, 7, 1)
    end_date = dt.date(2025, 1, 1)
    warmup_start = start_date - dt.timedelta(days=10)

    print("V14 5-Min Config Tuner")
    print("=" * 70)
    print(f"Period: {start_date} to {end_date}")
    print(f"Capital: Rs {CAPITAL:,.0f}")
    print()

    # Load data once
    print("Loading data...", flush=True)
    nifty_1min = load_nifty_data(warmup_start, end_date)
    vix_lookup = load_vix_data(warmup_start, end_date)
    nifty_5min = resample_to_5min(nifty_1min)

    # Group by date
    day_groups = {}
    for ts, row in nifty_5min.iterrows():
        d = ts.date()
        if d not in day_groups:
            day_groups[d] = []
        day_groups[d].append({
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row.get("volume", 0)),
            "date": str(d),
            "time": str(ts),
        })

    all_dates = sorted(d for d in day_groups.keys() if d >= start_date)
    warmup_dates = sorted(d for d in day_groups.keys() if d < start_date)
    warmup_bars = []
    for d in warmup_dates[-3:]:
        warmup_bars.extend(day_groups[d])

    print(f"Trading days: {len(all_dates)}")
    print()

    # Run all variants
    variants = make_variants()
    results = []

    for name, cfg in variants.items():
        print(f"  Testing {name}...", end="", flush=True)
        result = run_variant(name, cfg, day_groups, all_dates, warmup_bars, vix_lookup)
        results.append(result)
        print(f" {result['trades']} trades | {result['win_rate']:.1f}% WR | "
              f"Rs {result['total_pnl']:+,.0f} | {result['return_x']:.2f}x | "
              f"PF {result['profit_factor']:.2f}")

    # Sort by total P&L
    results.sort(key=lambda x: x["total_pnl"], reverse=True)

    print()
    print("=" * 90)
    print(f"{'Rank':<5} {'Config':<25} {'Trades':>7} {'WR':>6} {'PnL':>12} {'Return':>8} {'PF':>6}")
    print("-" * 90)
    for i, r in enumerate(results, 1):
        print(f"{i:<5} {r['name']:<25} {r['trades']:>7} {r['win_rate']:>5.1f}% "
              f"{r['total_pnl']:>+12,.0f} {r['return_x']:>7.2f}x {r['profit_factor']:>6.2f}")

    best = results[0]
    print()
    print(f"BEST: {best['name']} -> {best['return_x']:.2f}x, Rs {best['total_pnl']:+,.0f}")


if __name__ == "__main__":
    main()
