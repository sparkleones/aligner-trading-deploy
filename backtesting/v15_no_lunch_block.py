"""Backtest: compare V15 baseline vs removing the avoid_windows_bars lunch block.

Tests whether removing the 11:55 AM - 1:55 PM block (bars 33-57) improves
21-month performance.
"""
import sys
import datetime as dt
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backtesting.v14_unified_backtest import run_backtest, load_period_data, CAPITAL


VARIANTS = [
    ("V15 baseline (lunch block ON)",        {"avoid_days": [2]}),
    ("No lunch block",                        {"avoid_days": [2], "avoid_windows_bars": []}),
    ("No lunch block + no call 4th hr block", {"avoid_days": [2], "avoid_windows_bars": [], "block_call_4th_hour": False}),
    ("All blocks removed",                    {"avoid_days": [2], "avoid_windows_bars": [], "block_call_4th_hour": False, "block_late_entries": 999}),
]


def main():
    start = dt.date(2024, 7, 1)
    months = 22  # Jul 2024 through Apr 2026

    print(f"LUNCH BLOCK COMPARISON: {start} for {months} months")
    print(f"Baseline: avoid_windows_bars=[(33,57)] — lunch hour 11:55 AM - 1:55 PM")
    print("=" * 96, flush=True)

    print("Loading data once...", flush=True)
    preloaded = load_period_data(start_date=start, months=months, quiet=False)
    print(flush=True)

    print(f"{'Variant':<50} {'Trades':>6} {'P&L':>13} {'WR':>6} {'Return':>7} {'MaxDD':>8}", flush=True)
    print("-" * 96, flush=True)

    results = []
    for label, override in VARIANTS:
        try:
            trades, equity = run_backtest(
                start_date=start, months=months,
                cfg_override=override, quiet=True, preloaded=preloaded,
            )
            pnl = sum(t.get("pnl", 0) for t in trades)
            wins = sum(1 for t in trades if t.get("pnl", 0) > 0)
            wr = wins / len(trades) * 100 if trades else 0
            ret = equity / CAPITAL

            # compute max drawdown
            running = CAPITAL
            peak = CAPITAL
            max_dd = 0.0
            for t in trades:
                running += t.get("pnl", 0)
                if running > peak:
                    peak = running
                dd = (peak - running) / peak * 100
                if dd > max_dd:
                    max_dd = dd

            results.append((label, len(trades), pnl, wr, ret, max_dd))
            print(f"{label:<50} {len(trades):>6} {pnl:>+13,.0f} {wr:>5.1f}% {ret:>6.2f}x {max_dd:>7.1f}%", flush=True)
        except Exception as e:
            print(f"{label:<50} ERROR: {e}", flush=True)
            import traceback
            traceback.print_exc()

    print("=" * 96)
    if results:
        baseline = results[0]
        print(f"\nVERDICT:")
        for r in results[1:]:
            delta_ret = r[4] - baseline[4]
            delta_trades = r[2] - baseline[2]
            sign = "+" if delta_ret >= 0 else ""
            print(f"  {r[0]}: {sign}{delta_ret:.2f}x vs baseline | {delta_trades:+d} trades | MaxDD {r[5]:.1f}%")


if __name__ == "__main__":
    main()
