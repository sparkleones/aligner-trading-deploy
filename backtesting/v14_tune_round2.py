"""
V14 Round 2 Tuner — Fine-tune around V3_EntryWindows (the Round 1 winner).
============================================================================
Base: V3_EntryWindows (2.30x, 93 trades, 35.5% WR, PF 1.19, MaxDD 69.3%)
  - entry_windows_bars = [(3, 12), (21, 33), (57, 69)]

Tests 16 variations around V3 to squeeze maximum profit:
  - Window sizes (wider/narrower)
  - Score thresholds
  - Trail parameters
  - Concurrent trades
  - Skip bar 0
  - Cooldown
  - Max hold
  - Lunch block interaction
"""

import sys
import copy
import datetime as dt
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scoring.config import V14_CONFIG
from backtesting.v14_advanced_tune import (
    load_data, simulate_day_advanced, run_variant,
    CAPITAL, LOT_SIZE,
)


def make_round2_variants():
    """All variants start from V3_EntryWindows."""
    v = {}

    # Build V3 base
    v3 = copy.deepcopy(V14_CONFIG)
    v3["entry_windows_bars"] = [(3, 12), (21, 33), (57, 69)]

    # ── R0: V3 baseline ──
    v["R0_V3Base"] = copy.deepcopy(v3)

    # ── R1: Wider morning window (9:15-10:30 instead of 9:30-10) ──
    c = copy.deepcopy(v3)
    c["entry_windows_bars"] = [(0, 15), (21, 33), (57, 69)]
    v["R1_WiderMorn"] = c

    # ── R2: Tighter morning (9:30-9:50) + wider afternoon ──
    c = copy.deepcopy(v3)
    c["entry_windows_bars"] = [(3, 7), (21, 33), (54, 72)]
    v["R2_TightMorn+WideAftn"] = c

    # ── R3: Add 10:30-11:00 window (4 windows) ──
    c = copy.deepcopy(v3)
    c["entry_windows_bars"] = [(3, 12), (15, 21), (21, 33), (57, 69)]
    v["R3_4Windows"] = c

    # ── R4: Only 2 windows — morning + afternoon (skip lunch) ──
    c = copy.deepcopy(v3)
    c["entry_windows_bars"] = [(3, 15), (54, 69)]
    v["R4_2Windows"] = c

    # ── R5: V3 + skip bar 0 (R1 winner finding: gap noise) ──
    c = copy.deepcopy(v3); c["skip_first_bar"] = True
    v["R5_V3+SkipBar0"] = c

    # ── R6: V3 + lower PUT score (4.5 instead of 5.0) ──
    c = copy.deepcopy(v3); c["put_score_min"] = 4.5
    v["R6_V3+LowerPutScore"] = c

    # ── R7: V3 + higher CALL score (7.0 instead of 6.0) ──
    c = copy.deepcopy(v3); c["call_score_min"] = 7.0
    v["R7_V3+HighCallScore"] = c

    # ── R8: V3 + both scores relaxed (4.5/5.5) ──
    c = copy.deepcopy(v3)
    c["put_score_min"] = 4.5; c["call_score_min"] = 5.5
    v["R8_V3+RelaxScores"] = c

    # ── R9: V3 + wider PUT trail (2.0%) ──
    c = copy.deepcopy(v3); c["trail_pct_put"] = 0.020
    v["R9_V3+WiderPutTrail"] = c

    # ── R10: V3 + narrower CALL trail (0.5%) ──
    c = copy.deepcopy(v3); c["trail_pct_call"] = 0.005
    v["R10_V3+TightCallTrail"] = c

    # ── R11: V3 + max concurrent = 4 ──
    c = copy.deepcopy(v3); c["max_concurrent"] = 4
    v["R11_V3+4Concurrent"] = c

    # ── R12: V3 + max concurrent = 1 (quality over quantity) ──
    c = copy.deepcopy(v3); c["max_concurrent"] = 1
    v["R12_V3+1Concurrent"] = c

    # ── R13: V3 + zero cooldown ──
    c = copy.deepcopy(v3); c["cooldown_bars"] = 0
    v["R13_V3+NoCooldown"] = c

    # ── R14: V3 + longer max hold (PUT 72, CALL 60) ──
    c = copy.deepcopy(v3)
    c["max_hold_put"] = 72; c["max_hold_call"] = 60
    v["R14_V3+LongerHold"] = c

    # ── R15: V3 + no lunch avoid block ──
    c = copy.deepcopy(v3); c["avoid_windows_bars"] = []
    v["R15_V3+NoLunchBlock"] = c

    # ── R16: V3 + more trades per day (10) ──
    c = copy.deepcopy(v3); c["max_trades_per_day"] = 10
    v["R16_V3+10Trades"] = c

    # ── R17: Best combo — V3 + skip bar0 + wider PUT trail + no lunch block ──
    c = copy.deepcopy(v3)
    c["skip_first_bar"] = True
    c["trail_pct_put"] = 0.020
    c["avoid_windows_bars"] = []
    v["R17_V3+BestCombo"] = c

    # ── R18: V3 + lower min_confidence (0.30 from 0.35) ──
    c = copy.deepcopy(v3); c["min_confidence"] = 0.30
    v["R18_V3+LowConf"] = c

    # ── R19: V3 + higher min_confidence (0.40) ──
    c = copy.deepcopy(v3); c["min_confidence"] = 0.40
    v["R19_V3+HighConf"] = c

    return v


def main():
    start_date = dt.date(2024, 7, 1)
    end_date = dt.date(2025, 1, 1)

    print("V14 Round 2 Tuner — Fine-Tuning V3_EntryWindows")
    print("=" * 80)
    print(f"Period: {start_date} to {end_date} (6 months)")
    print(f"Capital: Rs {CAPITAL:,.0f} | Equity COMPOUNDED")
    print()

    print("Loading data...", flush=True)
    day_groups, all_dates, warmup_bars, vix_lookup = load_data(start_date, end_date)
    print(f"Trading days: {len(all_dates)}")
    print()

    variants = make_round2_variants()
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
    print("=" * 110)
    print(f"{'Rank':<4} {'Config':<30} {'Trades':>6} {'WR':>6} {'PnL':>14} {'Return':>8} {'PF':>6} {'MaxDD':>7}")
    print("-" * 110)
    for i, r in enumerate(results, 1):
        marker = " <-- WINNER" if i == 1 else ""
        print(f"{i:<4} {r['name']:<30} {r['trades']:>6} {r['win_rate']:>5.1f}% "
              f"{r['total_pnl']:>+14,.0f} {r['return_x']:>7.2f}x {r['profit_factor']:>6.2f} "
              f"{r['max_dd_pct']:>6.1f}%{marker}")

    # Monthly breakdown for top 3
    print()
    print("=" * 80)
    print("MONTHLY P&L — TOP 3")
    print("=" * 80)
    for r in results[:3]:
        print(f"\n{r['name']} ({r['return_x']:.2f}x, Rs {r['total_pnl']:+,.0f}):")
        cum = CAPITAL
        for m in sorted(r["monthly_pnl"]):
            cum += r["monthly_pnl"][m]
            print(f"  {m}: Rs {r['monthly_pnl'][m]:>+12,.0f}  (Equity: Rs {cum:>12,.0f})")

    best = results[0]
    print()
    print(f"ROUND 2 WINNER: {best['name']}")
    print(f"  Return: {best['return_x']:.2f}x ({(best['return_x']-1)*100:.0f}%)")
    print(f"  P&L: Rs {best['total_pnl']:+,.0f}")
    print(f"  Trades: {best['trades']} | WR: {best['win_rate']:.1f}%")
    print(f"  Profit Factor: {best['profit_factor']:.2f}")
    print(f"  Max Drawdown: {best['max_dd_pct']:.1f}%")

    return results


if __name__ == "__main__":
    main()
