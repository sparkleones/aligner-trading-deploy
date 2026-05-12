"""
Diagnose WHY composite_top underperformed NIFTY in:
  Window 5: 2024-01-01 to 2025-07-01  (strat +6.43% vs NIFTY +11.36%)
  Window 6: 2024-07-01 to 2026-01-01  (strat -13.59% vs NIFTY +5.45%)

Analyzes:
  1. Which picks were chosen each quarter
  2. How long they held
  3. Exit reasons (stage 2 break vs rebalance vs end)
  4. Sector concentration over time
  5. NIFTY itself vs the strategy's PnL distribution
"""
from __future__ import annotations

import pandas as pd

from .data_loader import load_universe
from .strategies.composite_top import CompositeTopStrategy
from .backtest_mf_style import MFStyleConfig, run_mf_backtest
from .universe import get_sector
from .universe_extended import LARGE_CAP


def diagnose(start, end):
    lg = load_universe(LARGE_CAP, period="5y", use_cache=True, progress=False)
    cfg = MFStyleConfig(start_date=start, end_date=end,
                         initial_capital=100_000, n_picks=5,
                         rebalance_freq="QS", apply_stcg_tax=True)
    res = run_mf_backtest(CompositeTopStrategy(), lg, get_sector, cfg=cfg)
    eq = res["equity_curve"]
    trades = res["trades"]
    stats = res["stats"]

    print(f"\n{'='*78}\n WINDOW: {start} to {end}\n{'='*78}")
    print(f"  Final equity:     Rs {stats.get('final_equity', 0):,.0f}")
    print(f"  CAGR:             {stats.get('cagr_pct', 0)*100:+.2f}%")
    print(f"  Sharpe:           {stats.get('sharpe', 0):.2f}")
    print(f"  Max DD:           {stats.get('max_dd_pct', 0)*100:+.2f}%")
    print(f"  Trades:           {stats.get('n_trades', 0)}")
    print(f"  Win rate:         {stats.get('win_rate_pct', 0)*100:.0f}%")

    if trades.empty:
        return

    print(f"\n  Exit reasons:")
    print(trades.groupby("exit_reason").size().to_string())

    print(f"\n  Trade P&L distribution:")
    print(f"    Mean P&L %:   {trades['pnl_pct'].mean()*100:+.2f}%")
    print(f"    Median P&L %: {trades['pnl_pct'].median()*100:+.2f}%")
    print(f"    Best trade:   {trades['pnl_pct'].max()*100:+.2f}%  ({trades.loc[trades['pnl_pct'].idxmax(), 'symbol']})")
    print(f"    Worst trade:  {trades['pnl_pct'].min()*100:+.2f}%  ({trades.loc[trades['pnl_pct'].idxmin(), 'symbol']})")
    print(f"    Avg hold:     {trades['bars_held'].mean():.0f} bars")

    # By sector
    trades["sector"] = trades["symbol"].map(get_sector)
    print(f"\n  By sector (P&L %):")
    by_sector = trades.groupby("sector").agg(
        trades=("symbol", "count"),
        avg_pnl_pct=("pnl_pct", "mean"),
        total_pnl=("pnl_net", "sum"),
    ).sort_values("total_pnl")
    print(by_sector.to_string())

    # Worst 5 trades
    print(f"\n  Worst 5 trades:")
    worst = trades.nsmallest(5, "pnl_pct")[["symbol", "entry", "exit", "pnl_pct", "exit_reason", "bars_held"]]
    worst["pnl_pct"] = (worst["pnl_pct"] * 100).round(2).astype(str) + "%"
    worst["bars_held"] = worst["bars_held"].astype(int)
    print(worst.to_string(index=False))

    print(f"\n  Best 5 trades:")
    best = trades.nlargest(5, "pnl_pct")[["symbol", "entry", "exit", "pnl_pct", "exit_reason", "bars_held"]]
    best["pnl_pct"] = (best["pnl_pct"] * 100).round(2).astype(str) + "%"
    best["bars_held"] = best["bars_held"].astype(int)
    print(best.to_string(index=False))


def main():
    diagnose("2024-01-01", "2025-07-01")
    diagnose("2024-07-01", "2026-01-01")


if __name__ == "__main__":
    main()
