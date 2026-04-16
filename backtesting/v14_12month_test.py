"""
V14 12-Month Test — Compare 6-month vs 12-month compounding.
Also test with and without entry windows.

The previous 12.0x result was on 12 months of data with full compounding.
Our 3.48x was on 6 months. Math: 3.48^2 = 12.1x — the difference is
primarily the compounding period, not the model.
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


def main():
    print("V14 — 6-Month vs 12-Month Compounding Test")
    print("=" * 90)
    print(f"Capital: Rs {CAPITAL:,.0f} | Equity COMPOUNDED")
    print()

    # ── Test configs ──
    # A: Current R4 (with entry windows)
    r4 = copy.deepcopy(V14_CONFIG)  # Already has entry_windows_bars

    # B: R4 WITHOUT entry windows (like the old 12.0x model)
    no_win = copy.deepcopy(V14_CONFIG)
    no_win["entry_windows_bars"] = None  # Remove windows
    no_win["name"] = "V14_NoWindows"

    tests = [
        # (label, start, end, config, config_name)
        ("6M_R4_Windows",    dt.date(2024, 7, 1),  dt.date(2025, 1, 1),  r4,     "R4_2Windows"),
        ("6M_NoWindows",     dt.date(2024, 7, 1),  dt.date(2025, 1, 1),  no_win, "NoWindows"),
        ("12M_R4_Windows",   dt.date(2024, 1, 1),  dt.date(2025, 1, 1),  r4,     "R4_2Windows"),
        ("12M_NoWindows",    dt.date(2024, 1, 1),  dt.date(2025, 1, 1),  no_win, "NoWindows"),
        ("15M_R4_Windows",   dt.date(2023, 12, 1), dt.date(2025, 1, 31), r4,     "R4_2Windows"),
        ("15M_NoWindows",    dt.date(2023, 12, 1), dt.date(2025, 1, 31), no_win, "NoWindows"),
    ]

    results = []
    for label, start, end, cfg, cfg_name in tests:
        months = (end.year - start.year) * 12 + (end.month - start.month)
        print(f"\n{'='*70}")
        print(f"  {label} ({months}M: {start} to {end}) — {cfg_name}")
        print(f"{'='*70}")
        try:
            day_groups, all_dates, warmup_bars, vix_lookup = load_data(start, end)
            print(f"  Trading days: {len(all_dates)}")
            r = run_variant(label, cfg, day_groups, all_dates, warmup_bars, vix_lookup)
            results.append(r)
            print(f"  {r['trades']} trades | {r['win_rate']:.1f}% WR | "
                  f"Rs {r['total_pnl']:+,.0f} | {r['return_x']:.2f}x | "
                  f"PF {r['profit_factor']:.2f} | MaxDD {r['max_dd_pct']:.1f}%")

            # Monthly breakdown
            cum = CAPITAL
            for m in sorted(r["monthly_pnl"]):
                cum += r["monthly_pnl"][m]
                print(f"    {m}: Rs {r['monthly_pnl'][m]:>+12,.0f}  (Equity: Rs {cum:>12,.0f})")
        except Exception as e:
            print(f"  ERROR: {e}")

    # Summary
    print()
    print("=" * 100)
    print(f"{'Test':<25} {'Trades':>6} {'WR':>6} {'PnL':>14} {'Return':>8} {'PF':>6} {'MaxDD':>7}")
    print("-" * 100)
    for r in results:
        print(f"{r['name']:<25} {r['trades']:>6} {r['win_rate']:>5.1f}% "
              f"{r['total_pnl']:>+14,.0f} {r['return_x']:>7.2f}x {r['profit_factor']:>6.2f} "
              f"{r['max_dd_pct']:>6.1f}%")


if __name__ == "__main__":
    main()
