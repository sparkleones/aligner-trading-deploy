"""V15 Config Sweep — test improvement variants vs baseline.

Loads NIFTY+VIX data ONCE and runs multiple V15_CONFIG variants against the
same period via run_backtest(cfg_override=...).
"""
import sys
import datetime as dt
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backtesting.v14_unified_backtest import run_backtest, load_period_data, CAPITAL


# Each variant: (label, cfg_override_dict)
VARIANTS = [
    ("V15 baseline (no override)",                          {}),
    ("V1: trend regime gate ON",                            {"use_trend_regime_gate": True}),
    ("V2: avoid Wednesday",                                 {"avoid_days": [2]}),
    ("V3: avoid Mon+Wed",                                   {"avoid_days": [0, 2]}),
    ("V4: regime gate + avoid Wed",                         {"use_trend_regime_gate": True, "avoid_days": [2]}),
    ("V5: regime gate + avoid Mon+Wed",                     {"use_trend_regime_gate": True, "avoid_days": [0, 2]}),
    ("V6: VIX floor 14 (was 13)",                           {"vix_floor": 14}),
    ("V7: VIX floor 15",                                    {"vix_floor": 15}),
    ("V8: regime + Wed + VIX14",                            {"use_trend_regime_gate": True, "avoid_days": [2], "vix_floor": 14}),
    ("V9: regime + Wed + VIX15",                            {"use_trend_regime_gate": True, "avoid_days": [2], "vix_floor": 15}),
    ("V10: regime + Mon+Wed + VIX14",                       {"use_trend_regime_gate": True, "avoid_days": [0, 2], "vix_floor": 14}),
    ("V11: tighter call_score_min=6.5",                     {"call_score_min": 6.5}),
    ("V12: tighter call_score_min=8",                       {"call_score_min": 8.0}),
    ("V13: regime + Wed + call_min=6.5",                    {"use_trend_regime_gate": True, "avoid_days": [2], "call_score_min": 6.5}),
    ("V14: regime + Wed + call_min=8",                      {"use_trend_regime_gate": True, "avoid_days": [2], "call_score_min": 8.0}),
    ("V15: regime + Wed + call_min=8 + VIX14",              {"use_trend_regime_gate": True, "avoid_days": [2], "call_score_min": 8.0, "vix_floor": 14}),
    ("V16: block_late_entries=55 (was 61)",                 {"block_late_entries": 55}),
    ("V17: max_concurrent=2 (was 3)",                       {"max_concurrent": 2}),
    ("V18: regime + Wed + max_conc=2",                      {"use_trend_regime_gate": True, "avoid_days": [2], "max_concurrent": 2}),
]


def main():
    print(f"V15 SWEEP — {len(VARIANTS)} variants over Jul 2024 - Jan 2025", flush=True)
    print("Loading data once...", flush=True)
    preloaded = load_period_data(start_date=dt.date(2024, 7, 1), months=6, quiet=False)
    print(flush=True)
    print(f"{'Variant':<46} {'Trades':>6} {'P&L':>13} {'WR':>6} {'Return':>7}", flush=True)
    print("-" * 86, flush=True)

    results = []
    for label, override in VARIANTS:
        try:
            trades, equity = run_backtest(
                start_date=dt.date(2024, 7, 1),
                months=6,
                cfg_override=override,
                quiet=True,
                preloaded=preloaded,
            )
            pnl = sum(t.get("pnl", 0) for t in trades)
            wins = sum(1 for t in trades if t.get("pnl", 0) > 0)
            wr = wins / len(trades) * 100 if trades else 0
            ret = equity / CAPITAL
            print(f"{label:<46} {len(trades):>6} {pnl:>+13,.0f} {wr:>5.1f}% {ret:>6.2f}x", flush=True)
            results.append((label, len(trades), pnl, wr, ret, override))
        except Exception as e:
            print(f"{label:<46} ERROR: {e}", flush=True)
            import traceback
            traceback.print_exc()

    # Summary: top 5 by return
    print("\n" + "=" * 86)
    print("TOP 5 by Return (compounded):")
    print("=" * 86)
    for label, n, pnl, wr, ret, override in sorted(results, key=lambda x: x[4], reverse=True)[:5]:
        print(f"  {ret:.2f}x  |  {n} trades  |  {wr:.1f}% WR  |  P&L Rs {pnl:+,.0f}")
        print(f"          override: {override}")


if __name__ == "__main__":
    main()
