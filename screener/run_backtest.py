"""
CLI runner for the stock screener backtest.

Usage:
    python -m screener.run_backtest
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from .universe import get_universe
from .data_loader import load_universe
from .backtest import BacktestConfig, run_backtest


def main():
    print("="*70)
    print(" STOCK SCREENER — BACKTEST")
    print("="*70)
    cfg = BacktestConfig(
        start_date="2023-01-01",
        end_date="2026-04-30",
        initial_capital=100_000.0,
        n_picks=2,
    )

    universe = get_universe()
    print(f"\nUniverse size: {len(universe)} F&O equities")
    print("Downloading 3-year history (cached on disk)...")
    history = load_universe(universe, period="4y", use_cache=True)
    print(f"\nLoaded valid history for {len(history)}/{len(universe)} symbols")

    if len(history) < 20:
        print("ERROR: Too few stocks loaded — backtest needs at least 20")
        return

    print(f"\nRunning backtest {cfg.start_date} -> {cfg.end_date}...")
    print(f"  initial_capital = Rs {cfg.initial_capital:,.0f}")
    print(f"  n_concurrent_positions = {cfg.n_picks}")
    print(f"  rebalance = {cfg.rebalance_freq}")

    result = run_backtest(history, cfg=cfg, verbose=True)
    stats = result["stats"]

    print("\n" + "="*70)
    print(" RESULTS")
    print("="*70)
    print(f"  Initial capital:    Rs {stats['initial_capital']:>12,.0f}")
    print(f"  Final equity:       Rs {stats['final_equity']:>12,.0f}")
    print(f"  Total return:          {stats['total_return_pct']:>12.2%}")
    print(f"  CAGR:                  {stats['cagr_pct']:>12.2%}")
    print(f"  Sharpe ratio:          {stats['sharpe']:>12.2f}")
    print(f"  Max drawdown:          {stats['max_drawdown_pct']:>12.2%}")
    print(f"  Calmar ratio:          {stats['calmar']:>12.2f}")
    print(f"  Years tested:          {stats['years']:>12.2f}")
    print()
    print(f"  N trades:              {stats['n_trades']:>12d}")
    print(f"  Win rate:              {stats['win_rate_pct']:>12.2%}")
    print(f"  Avg win:               {stats['avg_win_pct']:>12.2%}")
    print(f"  Avg loss:              {stats['avg_loss_pct']:>12.2%}")
    print(f"  Profit factor:         {stats['profit_factor']:>12.2f}")
    print(f"  Avg hold (bars):       {stats['avg_hold_bars']:>12.1f}")
    print()
    print("  Exit reason breakdown:")
    for reason, count in stats["exit_reasons"].items():
        pct = count / stats["n_trades"] if stats["n_trades"] else 0
        print(f"    {reason:>15s}: {count:>4d} ({pct:.0%})")

    # Save results
    out_dir = Path(__file__).resolve().parent.parent / "reports" / "screener"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Strip non-JSON-serializable bits
    stats_json = {k: (float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v)
                  for k, v in stats.items()}
    with open(out_dir / "backtest_stats.json", "w") as f:
        json.dump(stats_json, f, indent=2, default=str)
    result["equity_curve"].to_csv(out_dir / "equity_curve.csv")
    result["trades"].to_csv(out_dir / "trades.csv", index=False)

    print(f"\n[OK] Results saved to {out_dir}")
    print("="*70)

    return result


if __name__ == "__main__":
    main()
