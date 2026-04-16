"""V15 Full Period Test — run V3 (avoid Mon+Wed) contiguously Jul 2024 - Apr 2026.

This produces the true compounded equity curve that respects period boundaries.
"""
import sys
import datetime as dt
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backtesting.v14_unified_backtest import run_backtest, load_period_data, CAPITAL


VARIANTS = [
    ("V15 baseline",            {}),
    ("V3 (avoid Mon+Wed)",      {"avoid_days": [0, 2]}),
    ("V2 (avoid Wed only)",     {"avoid_days": [2]}),
]


def main():
    start = dt.date(2024, 7, 1)
    months = 22  # Jul 2024 through Apr 2026

    print(f"V15 FULL PERIOD: {start} for {months} months (~21 calendar months)")
    print("=" * 86, flush=True)

    print("Loading data once...", flush=True)
    preloaded = load_period_data(start_date=start, months=months, quiet=False)
    print(flush=True)

    print(f"{'Variant':<46} {'Trades':>6} {'P&L':>13} {'WR':>6} {'Return':>7}", flush=True)
    print("-" * 86, flush=True)

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
            print(f"{label:<46} {len(trades):>6} {pnl:>+13,.0f} {wr:>5.1f}% {ret:>6.2f}x", flush=True)
        except Exception as e:
            print(f"{label:<46} ERROR: {e}", flush=True)
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
