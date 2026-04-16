"""
V14 Round 3 — Final fine-tuning around R4_2Windows (3.48x winner).
Base: entry_windows_bars = [(3, 15), (54, 69)]
"""

import sys
import copy
import datetime as dt
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scoring.config import V14_CONFIG
from backtesting.v14_advanced_tune import (
    load_data, run_variant, CAPITAL,
)


def make_round3_variants():
    v = {}
    # R4 base
    r4 = copy.deepcopy(V14_CONFIG)
    r4["entry_windows_bars"] = [(3, 15), (54, 69)]

    v["S0_R4Base"] = copy.deepcopy(r4)

    # ── Window boundary experiments ──
    # S1: Wider morning (0-18 = 9:15-10:45)
    c = copy.deepcopy(r4); c["entry_windows_bars"] = [(0, 18), (54, 69)]
    v["S1_WiderMorn0_18"] = c

    # S2: Narrower morning (3-10 = 9:30-10:05)
    c = copy.deepcopy(r4); c["entry_windows_bars"] = [(3, 10), (54, 69)]
    v["S2_NarrowMorn3_10"] = c

    # S3: Wider afternoon (48-72 = 13:15-15:15)
    c = copy.deepcopy(r4); c["entry_windows_bars"] = [(3, 15), (48, 72)]
    v["S3_WiderAftn48_72"] = c

    # S4: Narrower afternoon (57-66 = 13:45-14:30)
    c = copy.deepcopy(r4); c["entry_windows_bars"] = [(3, 15), (57, 66)]
    v["S4_NarrowAftn57_66"] = c

    # S5: Morning only (no afternoon)
    c = copy.deepcopy(r4); c["entry_windows_bars"] = [(3, 18)]
    v["S5_MornOnly"] = c

    # S6: Afternoon only (no morning)
    c = copy.deepcopy(r4); c["entry_windows_bars"] = [(48, 72)]
    v["S6_AftnOnly"] = c

    # S7: Extend morning to 15, start afternoon at 48
    c = copy.deepcopy(r4); c["entry_windows_bars"] = [(3, 15), (48, 69)]
    v["S7_EarlierAftn"] = c

    # S8: Morning 3-12, Afternoon 54-72
    c = copy.deepcopy(r4); c["entry_windows_bars"] = [(3, 12), (54, 72)]
    v["S8_TightMorn_WideAftn"] = c

    # ── Score tweaks on R4 ──
    # S9: Lower PUT score (4.5)
    c = copy.deepcopy(r4); c["put_score_min"] = 4.5
    v["S9_LowerPutScore"] = c

    # S10: Relaxed both scores (4.5/5.5)
    c = copy.deepcopy(r4); c["put_score_min"] = 4.5; c["call_score_min"] = 5.5
    v["S10_RelaxScores"] = c

    # ── Trail tweaks on R4 ──
    # S11: Wider PUT trail (2.0%)
    c = copy.deepcopy(r4); c["trail_pct_put"] = 0.020
    v["S11_WiderPutTrail"] = c

    # S12: Both trails wider
    c = copy.deepcopy(r4); c["trail_pct_put"] = 0.020; c["trail_pct_call"] = 0.012
    v["S12_BothTrailsWider"] = c

    # ── Concurrent/Cooldown tweaks ──
    # S13: No cooldown
    c = copy.deepcopy(r4); c["cooldown_bars"] = 0
    v["S13_NoCooldown"] = c

    # S14: Max concurrent = 4
    c = copy.deepcopy(r4); c["max_concurrent"] = 4
    v["S14_4Concurrent"] = c

    # ── Best combos ──
    # S15: R4 + lower PUT score + no cooldown
    c = copy.deepcopy(r4); c["put_score_min"] = 4.5; c["cooldown_bars"] = 0
    v["S15_LowPut+NoCool"] = c

    # S16: R4 + wider morning (0-15) + lower PUT score
    c = copy.deepcopy(r4); c["entry_windows_bars"] = [(0, 15), (54, 69)]
    c["put_score_min"] = 4.5
    v["S16_WideMorn+LowPut"] = c

    # S17: R4 + no lunch block
    c = copy.deepcopy(r4); c["avoid_windows_bars"] = []
    v["S17_NoLunchBlock"] = c

    # S18: R4 + max trades 10
    c = copy.deepcopy(r4); c["max_trades_per_day"] = 10
    v["S18_10Trades"] = c

    return v


def main():
    start_date = dt.date(2024, 7, 1)
    end_date = dt.date(2025, 1, 1)

    print("V14 Round 3 — Final Fine-Tuning R4_2Windows")
    print("=" * 80)
    print(f"Period: {start_date} to {end_date} | Capital: Rs {CAPITAL:,.0f}")
    print()

    print("Loading data...", flush=True)
    day_groups, all_dates, warmup_bars, vix_lookup = load_data(start_date, end_date)
    print(f"Trading days: {len(all_dates)}")
    print()

    variants = make_round3_variants()
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
        marker = " ***" if i == 1 else ""
        print(f"{i:<4} {r['name']:<30} {r['trades']:>6} {r['win_rate']:>5.1f}% "
              f"{r['total_pnl']:>+14,.0f} {r['return_x']:>7.2f}x {r['profit_factor']:>6.2f} "
              f"{r['max_dd_pct']:>6.1f}%{marker}")

    # Monthly for top 3
    print()
    print("MONTHLY P&L — TOP 3")
    print("=" * 80)
    for r in results[:3]:
        print(f"\n{r['name']} ({r['return_x']:.2f}x):")
        cum = CAPITAL
        for m in sorted(r["monthly_pnl"]):
            cum += r["monthly_pnl"][m]
            print(f"  {m}: Rs {r['monthly_pnl'][m]:>+12,.0f}  (Equity: Rs {cum:>12,.0f})")

    best = results[0]
    print(f"\nFINAL WINNER: {best['name']}")
    print(f"  {best['return_x']:.2f}x | Rs {best['total_pnl']:+,.0f} | "
          f"{best['trades']} trades | {best['win_rate']:.1f}% WR | "
          f"PF {best['profit_factor']:.2f} | MaxDD {best['max_dd_pct']:.1f}%")

    return results


if __name__ == "__main__":
    main()
