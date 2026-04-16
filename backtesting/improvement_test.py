"""
Improvement Testing — Measure the impact of intelligent filters.
=================================================================
Tests improvements one at a time against V15 (V17_PROD_ONLY) baseline.

Usage:
    python -m backtesting.improvement_test
"""

import sys
import copy
import datetime as dt
from pathlib import Path
from collections import defaultdict

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np

from scoring.config import V15_CONFIG
from backtesting.v14_advanced_tune import load_data, CAPITAL
from backtesting.v16_comparison import simulate_day



def run_variant(name, cfg, day_groups, all_dates, warmup_bars_init, vix_lookup):
    """Run backtest with equity compounding using v16_comparison.simulate_day."""
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

        trades, day_pnl, eod_close = simulate_day(
            bars, date, vix, cfg, prev_close, equity,
            warmup_bars, is_expiry, consecutive_down_days,
        )

        equity += day_pnl
        all_trades.extend(trades)
        month_key = f"{date.year}-{date.month:02d}"
        monthly_pnl[month_key] += day_pnl

        if equity > max_equity:
            max_equity = equity
        dd = (max_equity - equity) / max_equity if max_equity > 0 else 0
        if dd > max_dd:
            max_dd = dd

        if len(bars) >= 2:
            consecutive_down_days = (
                consecutive_down_days + 1 if bars[-1]["close"] < bars[0]["open"] else 0
            )

        warmup_bars = warmup_bars[-(75 * 2):] + bars
        prev_close = eod_close

    total_pnl = sum(t.get("pnl", 0) for t in all_trades)
    wins = [t for t in all_trades if t.get("pnl", 0) > 0]
    losses = [t for t in all_trades if t.get("pnl", 0) <= 0]
    wr = len(wins) / len(all_trades) * 100 if all_trades else 0
    pf_num = sum(t["pnl"] for t in wins) if wins else 0
    pf_den = abs(sum(t["pnl"] for t in losses)) + 1
    pf = pf_num / pf_den
    avg_win = np.mean([t["pnl"] for t in wins]) if wins else 0
    avg_loss = np.mean([t["pnl"] for t in losses]) if losses else 0

    # Sharpe
    daily_returns = defaultdict(float)
    for t in all_trades:
        daily_returns[t.get("date", "")] += t.get("pnl", 0) / CAPITAL
    daily_ret_list = list(daily_returns.values())
    sharpe = 0.0
    if daily_ret_list and np.std(daily_ret_list) > 0:
        sharpe = np.mean(daily_ret_list) / np.std(daily_ret_list) * np.sqrt(252)

    return {
        "name": name,
        "trades": len(all_trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": wr,
        "total_pnl": total_pnl,
        "equity": equity,
        "return_x": equity / CAPITAL,
        "profit_factor": pf,
        "max_dd_pct": max_dd * 100,
        "monthly_pnl": dict(monthly_pnl),
        "avg_win": float(avg_win),
        "avg_loss": float(avg_loss),
        "sharpe": sharpe,
    }


def print_result(r, show_monthly=False):
    """Pretty-print one variant result."""
    max_dd = r.get("max_dd_pct", r.get("max_drawdown", 0) * 100)
    if max_dd < 1:
        max_dd = max_dd * 100  # handle fractional format
    print(f"  {'Trades:':<14} {r['trades']:>4}  ({r['wins']}W / {r['losses']}L)")
    print(f"  {'Win Rate:':<14} {r['win_rate']:>6.1f}%")
    print(f"  {'Total PnL:':<14} Rs {r['total_pnl']:>+12,.0f}")
    print(f"  {'Return:':<14} {r['return_x']:>7.2f}x")
    print(f"  {'Profit Factor:':<14} {r['profit_factor']:>6.2f}")
    print(f"  {'Max Drawdown:':<14} {max_dd:>6.1f}%")
    print(f"  {'Avg Win:':<14} Rs {r['avg_win']:>+10,.0f}")
    print(f"  {'Avg Loss:':<14} Rs {r['avg_loss']:>+10,.0f}")
    if r.get("sharpe"):
        print(f"  {'Sharpe:':<14} {r['sharpe']:>6.2f}")
    if show_monthly:
        cum = CAPITAL
        for m in sorted(r["monthly_pnl"]):
            cum += r["monthly_pnl"][m]
            print(f"    {m}: Rs {r['monthly_pnl'][m]:>+12,.0f}  (Equity: Rs {cum:>12,.0f})")


def main():
    # ── Period aligned with available data (earliest file: 2023-11-27) ──
    start_date = dt.date(2024, 1, 1)
    end_date = dt.date(2025, 10, 1)

    print("=" * 80)
    print("  IMPROVEMENT TESTING — Intelligent Filters on V15 Baseline")
    print("=" * 80)
    print(f"Period: {start_date} to {end_date}")
    print(f"Capital: Rs {CAPITAL:,.0f}")
    print()

    print("Loading data...", flush=True)
    day_groups, all_dates, warmup_bars, vix_lookup = load_data(start_date, end_date)
    print(f"Trading days: {len(all_dates)}, VIX days: {len(vix_lookup)}")
    print()

    results = []

    def test(name, cfg, show_monthly=False):
        print(f"  Running {name}...", end="", flush=True)
        r = run_variant(name, cfg, day_groups, all_dates, warmup_bars, vix_lookup)
        print(f" done ({r['trades']} trades)")
        print_result(r, show_monthly=show_monthly)
        results.append(r)
        return r

    # ─── BASELINE: V15 (= V17_PROD_ONLY for signal purposes) ─────────
    print("-" * 80)
    print("  [BASELINE] V15_CONFIG (avoid Mon+Wed, vix_floor=13)")
    print("-" * 80)
    baseline_cfg = copy.deepcopy(V15_CONFIG)
    test("V15_BASELINE", baseline_cfg, show_monthly=True)

    # ─── IMPROVEMENT #1A: EMA50 Trend Regime Gate ─────────────────────
    # Block counter-trend entries: no CALLs in confirmed downtrend,
    # no PUTs in confirmed uptrend. Uses the existing `trend_regime`
    # indicator (EMA50 slope + price position) which is already
    # computed every bar but never checked.
    print()
    print("-" * 80)
    print("  [IMP-1A] + EMA50 Trend Regime Gate (block counter-trend entries)")
    print("-" * 80)
    cfg_1a = copy.deepcopy(V15_CONFIG)
    cfg_1a["use_trend_regime_gate"] = True
    test("V15+RegimeGate", cfg_1a)

    # ─── IMPROVEMENT #1B: Regime Gate + Remove Mon/Wed block ──────────
    # Hypothesis: the regime gate is smarter than blanket day blocking.
    # If it works, we can trade Mon/Wed when the regime is favorable.
    print()
    print("-" * 80)
    print("  [IMP-1B] + EMA50 Regime Gate, NO avoid_days (regime replaces calendar)")
    print("-" * 80)
    cfg_1b = copy.deepcopy(V15_CONFIG)
    cfg_1b["use_trend_regime_gate"] = True
    cfg_1b["avoid_days"] = []
    test("RegimeGate_NoDayBlock", cfg_1b)

    # ─── IMPROVEMENT #1C: Regime Gate + Mon/Wed + VIX floor 14 ────────
    # From v15_sweep.py: V10 was tested as "regime + Mon+Wed + VIX14".
    # Let's see if higher VIX floor helps.
    print()
    print("-" * 80)
    print("  [IMP-1C] + Regime Gate + avoid Mon+Wed + VIX floor=14")
    print("-" * 80)
    cfg_1c = copy.deepcopy(V15_CONFIG)
    cfg_1c["use_trend_regime_gate"] = True
    cfg_1c["vix_floor"] = 14
    test("RegimeGate+VIX14", cfg_1c)

    # ─── IMPROVEMENT #1D: Full V17 (regime gate + monwed gate) ────────
    from scoring.config import V17_CONFIG
    print()
    print("-" * 80)
    print("  [IMP-1D] Full V17_CONFIG (regime gate + Mon/Wed conditional gate)")
    print("-" * 80)
    cfg_1d = copy.deepcopy(V17_CONFIG)
    test("V17_Full", cfg_1d)

    # ══════════════════════════════════════════════════════════════════
    #  SUMMARY TABLE
    # ══════════════════════════════════════════════════════════════════
    print()
    print("=" * 120)
    print(f"{'Variant':<30} {'Trades':>6} {'WR':>6} {'PnL':>14} {'Return':>8} "
          f"{'PF':>6} {'MaxDD':>7} {'Sharpe':>7} {'AvgW':>10} {'AvgL':>10}")
    print("-" * 120)
    for r in results:
        name = r.get("name", r.get("config", "?"))
        max_dd = r.get("max_dd_pct", r.get("max_drawdown", 0) * 100)
        if max_dd < 1:
            max_dd = max_dd * 100
        delta = ""
        if r != results[0]:
            d = r["return_x"] - results[0]["return_x"]
            delta = f" ({d:+.2f}x)"
        print(f"{name:<30} {r['trades']:>6} {r['win_rate']:>5.1f}% "
              f"{r['total_pnl']:>+14,.0f} {r['return_x']:>7.2f}x{delta:<9} "
              f"{r['profit_factor']:>5.2f} {max_dd:>6.1f}% "
              f"{r.get('sharpe', 0):>6.2f} "
              f"{r['avg_win']:>+10,.0f} {r['avg_loss']:>+10,.0f}")
    print("=" * 120)
    print()

    # Identify best
    best = max(results, key=lambda x: x["return_x"])
    best_name = best.get("name", best.get("config", "?"))
    baseline_name = results[0].get("name", results[0].get("config", "?"))
    if best_name != baseline_name:
        print(f"WINNER: {best_name} at {best['return_x']:.2f}x "
              f"(+{best['return_x'] - results[0]['return_x']:.2f}x vs baseline)")
    else:
        print("No improvement found — baseline is still the best.")


if __name__ == "__main__":
    main()
