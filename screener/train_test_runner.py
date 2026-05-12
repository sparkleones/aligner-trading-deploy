"""
Train/Test split runner — honest evaluation methodology.

  TRAIN: 2021-11-13 to 2025-11-12  (4.5 years)
         Used to:
           - sweep hold times
           - sweep portfolio weights
           - select winning strategy
  TEST:  2025-11-13 to 2026-05-13  (6 months)
         HELD OUT. Never seen during selection. Only one final run.

Capital: Rs 1,00,000 initial.
Tests the COMPOSITE blend (70% LARGE + 30% MID mean-rev).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data_loader import load_universe
from .strategies.composite_top import CompositeTopStrategy, CompositeMidStrategy
from .strategy_backtest import StrategyBacktestConfig, run_strategy_backtest
from .universe import get_sector
from .universe_extended import LARGE_CAP, MID_CAP

TRAIN_START = "2021-11-13"
TRAIN_END   = "2025-11-12"
TEST_START  = "2025-11-13"
TEST_END    = "2026-05-13"
INITIAL_CAPITAL = 100_000.0


def run_blend(lg_hist, mid_hist, start, end, init=INITIAL_CAPITAL, hold_bars=60, n_picks=2):
    cfg = StrategyBacktestConfig(
        start_date=start, end_date=end, initial_capital=init,
        n_picks=n_picks, rebalance_freq="BMS",
        time_stop_bars=hold_bars, max_per_sector=1,
    )
    r1 = run_strategy_backtest(CompositeTopStrategy(), lg_hist, get_sector, cfg=cfg)
    r2 = run_strategy_backtest(CompositeMidStrategy(), mid_hist, get_sector, cfg=cfg)
    eq_lg = r1["equity_curve"]["equity"]
    eq_mid = r2["equity_curve"]["equity"]
    if eq_lg.empty or eq_mid.empty:
        return None, None, None
    common = eq_lg.index.intersection(eq_mid.index)
    if len(common) < 5:
        return None, None, None
    blend = 0.70 * (eq_lg.loc[common] / eq_lg.iloc[0]) + 0.30 * (eq_mid.loc[common] / eq_mid.iloc[0])
    eq = blend * init
    final = float(eq.iloc[-1])
    years = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-6)
    cagr = (final / init) ** (1.0 / years) - 1.0
    daily = eq.pct_change().dropna()
    sharpe = float((daily.mean() / daily.std()) * np.sqrt(252)) if daily.std() > 0 else 0.0
    dd = (eq / eq.cummax() - 1.0).min()
    return (dict(cagr=cagr, sharpe=sharpe, dd=float(dd), final=final, years=years),
            r1["trades"], r2["trades"])


def main():
    print("="*70)
    print(" TRAIN / TEST EVALUATION")
    print("="*70)
    print(f"  TRAIN: {TRAIN_START} -> {TRAIN_END}  (4.5 yrs)")
    print(f"  TEST:  {TEST_START} -> {TEST_END}   (6 months)")
    print(f"  Capital: Rs {INITIAL_CAPITAL:,.0f}")

    print("\nLoading 5-yr history...")
    lg = load_universe(LARGE_CAP, period="5y", use_cache=True, progress=False)
    mid = load_universe(MID_CAP, period="5y", use_cache=True, progress=False)
    print(f"  LARGE: {len(lg)}  MID: {len(mid)}\n")

    # ── TRAIN: sweep hold periods ──
    print("=== TRAIN — sweep hold periods ===")
    print(f"  {'Hold (bars)':>12} {'Hold (days)':>13} {'CAGR':>10} {'Sharpe':>8} {'MaxDD':>10}")
    best = None
    for hold in [10, 20, 30, 45, 60, 90, 120]:
        r, _, _ = run_blend(lg, mid, TRAIN_START, TRAIN_END, hold_bars=hold)
        if not r:
            continue
        print(f"  {hold:>12d} {hold*7/5:>12.0f}  {r['cagr']*100:>8.2f}% {r['sharpe']:>8.2f} {r['dd']*100:>9.2f}%")
        if best is None or r["sharpe"] > best[1]["sharpe"]:
            best = (hold, r)

    if not best:
        print("No backtest ran successfully — exiting")
        return

    best_hold, best_r = best
    print(f"\n  BEST HOLD: {best_hold} bars = ~{best_hold*7/5:.0f} calendar days")
    print(f"  Train CAGR={best_r['cagr']*100:.2f}%  Sharpe={best_r['sharpe']:.2f}  DD={best_r['dd']*100:.2f}%")

    # ── TEST: single OOS run with chosen hold ──
    print("\n=== TEST — held-out 6 months, single run ===")
    test_r, test_lg_trades, test_mid_trades = run_blend(
        lg, mid, TEST_START, TEST_END, hold_bars=best_hold
    )
    if test_r:
        print(f"  CAGR (annualized): {test_r['cagr']*100:>6.2f}%")
        print(f"  Sharpe:            {test_r['sharpe']:>6.2f}")
        print(f"  MaxDD:             {test_r['dd']*100:>6.2f}%")
        print(f"  Final equity:      Rs {test_r['final']:>10,.0f}")
        print(f"  Absolute return:   Rs {test_r['final'] - INITIAL_CAPITAL:>+10,.0f}  ({(test_r['final']/INITIAL_CAPITAL-1)*100:+.2f}%)")

    # ── NIFTY benchmark for TEST window ──
    from .config_sweep import benchmark_nifty
    nb_train = benchmark_nifty(TRAIN_START, TRAIN_END, INITIAL_CAPITAL)
    nb_test = benchmark_nifty(TEST_START, TEST_END, INITIAL_CAPITAL)

    print("\n=== COMPARISON ===")
    print(f"  {'Window':<15} {'Strategy':<20} {'CAGR':>10} {'Sharpe':>8} {'MaxDD':>10}")
    print(f"  {'TRAIN':<15} {'NIFTY 50 B&H':<20} {nb_train['cagr_pct']*100:>8.2f}% {nb_train['sharpe']:>8.2f} {nb_train['max_drawdown_pct']*100:>9.2f}%")
    print(f"  {'TRAIN':<15} {'Blend (selected)':<20} {best_r['cagr']*100:>8.2f}% {best_r['sharpe']:>8.2f} {best_r['dd']*100:>9.2f}%")
    print(f"  {'TEST (OOS)':<15} {'NIFTY 50 B&H':<20} {nb_test['cagr_pct']*100:>8.2f}% {nb_test['sharpe']:>8.2f} {nb_test['max_drawdown_pct']*100:>9.2f}%")
    print(f"  {'TEST (OOS)':<15} {'Blend (held-out)':<20} {test_r['cagr']*100:>8.2f}% {test_r['sharpe']:>8.2f} {test_r['dd']*100:>9.2f}%")

    # Save findings
    out_dir = Path(__file__).resolve().parent.parent / "reports" / "screener"
    out_dir.mkdir(parents=True, exist_ok=True)
    findings = {
        "train_window": [TRAIN_START, TRAIN_END],
        "test_window": [TEST_START, TEST_END],
        "initial_capital": INITIAL_CAPITAL,
        "best_hold_bars": best_hold,
        "best_hold_days": int(best_hold * 7 / 5),
        "train_cagr": best_r["cagr"],
        "train_sharpe": best_r["sharpe"],
        "train_dd": best_r["dd"],
        "test_cagr_annualized": test_r["cagr"] if test_r else None,
        "test_sharpe": test_r["sharpe"] if test_r else None,
        "test_dd": test_r["dd"] if test_r else None,
        "test_final_equity": test_r["final"] if test_r else None,
        "test_absolute_pct": (test_r["final"] / INITIAL_CAPITAL - 1.0) if test_r else None,
        "benchmark_nifty_test_cagr": nb_test.get("cagr_pct", 0),
    }
    with open(out_dir / "train_test_findings.json", "w") as f:
        json.dump(findings, f, indent=2, default=str)
    print(f"\n[OK] Saved findings to {out_dir/'train_test_findings.json'}")

    # Save test trades for inspection
    if test_lg_trades is not None and not test_lg_trades.empty:
        test_lg_trades.to_csv(out_dir / "test_trades_large.csv", index=False)
    if test_mid_trades is not None and not test_mid_trades.empty:
        test_mid_trades.to_csv(out_dir / "test_trades_mid.csv", index=False)


if __name__ == "__main__":
    main()
