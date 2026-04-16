"""V15 Out-of-Sample Validation — test top sweep variants on Feb 2025+ data.

The sweep was tuned on Jul 2024 - Jan 2025 (in-sample). This script validates
the winning variants on Feb 2025 onward to check whether the improvements hold
out of sample (i.e. weren't just regime-fitted to a bearish IS period).
"""
import sys
import datetime as dt
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backtesting.v14_unified_backtest import run_backtest, load_period_data, CAPITAL


# Top variants from sweep
TOP_VARIANTS = [
    ("V15 baseline (no override)",       {}),
    ("V8 (regime + Wed + VIX14)",        {"use_trend_regime_gate": True, "avoid_days": [2], "vix_floor": 14}),
    ("V10 (regime + Mon+Wed + VIX14)",   {"use_trend_regime_gate": True, "avoid_days": [0, 2], "vix_floor": 14}),
    ("V3 (avoid Mon+Wed)",               {"avoid_days": [0, 2]}),
    ("V2 (avoid Wed)",                   {"avoid_days": [2]}),
]

# OOS periods
PERIODS = [
    ("OOS Feb–Jul 2025",  dt.date(2025, 2, 1),  6),
    ("OOS Aug 2025–Jan 2026", dt.date(2025, 8, 1),  6),
    ("OOS Feb–Apr 2026",  dt.date(2026, 2, 1),  2),
]


def main():
    for label, start, months in PERIODS:
        print(f"\n{'='*86}")
        print(f"{label}: {start} for {months} months")
        print(f"{'='*86}")
        try:
            preloaded = load_period_data(start_date=start, months=months, quiet=True)
        except Exception as e:
            print(f"  Could not load data: {e}")
            continue

        print(f"{'Variant':<46} {'Trades':>6} {'P&L':>13} {'WR':>6} {'Return':>7}", flush=True)
        print("-" * 86, flush=True)
        for vlabel, override in TOP_VARIANTS:
            try:
                trades, equity = run_backtest(
                    start_date=start, months=months,
                    cfg_override=override, quiet=True, preloaded=preloaded,
                )
                pnl = sum(t.get("pnl", 0) for t in trades)
                wins = sum(1 for t in trades if t.get("pnl", 0) > 0)
                wr = wins / len(trades) * 100 if trades else 0
                ret = equity / CAPITAL
                print(f"{vlabel:<46} {len(trades):>6} {pnl:>+13,.0f} {wr:>5.1f}% {ret:>6.2f}x", flush=True)
            except Exception as e:
                print(f"{vlabel:<46} ERROR: {e}", flush=True)


if __name__ == "__main__":
    main()
