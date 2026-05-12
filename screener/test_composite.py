"""Backtest the composite strategy on LARGE cap, plus a portfolio mix."""
from __future__ import annotations

import pandas as pd

from .data_loader import load_universe
from .strategies.composite_top import CompositeTopStrategy, CompositeMidStrategy
from .strategy_backtest import StrategyBacktestConfig, run_strategy_backtest
from .universe_extended import LARGE_CAP, MID_CAP
from .universe import get_sector


def test():
    print("="*70)
    print(" COMPOSITE STRATEGY BACKTEST")
    print("="*70)

    print("\nLoading 4-yr history...")
    history_lg = load_universe(LARGE_CAP, period="4y", use_cache=True, progress=False)
    history_mid = load_universe(MID_CAP, period="4y", use_cache=True, progress=False)
    print(f"  LARGE: {len(history_lg)}/{len(LARGE_CAP)}")
    print(f"  MID:   {len(history_mid)}/{len(MID_CAP)}\n")

    cfg = StrategyBacktestConfig(
        start_date="2023-01-01",
        end_date="2026-04-30",
        initial_capital=100_000.0,
        n_picks=2,
        rebalance_freq="BMS",
        time_stop_bars=60,
        max_per_sector=1,
    )

    print("=== COMPOSITE (LARGE cap, stage2 + breakout blend) ===")
    r1 = run_strategy_backtest(CompositeTopStrategy(), history_lg, get_sector, cfg=cfg)
    s = r1["stats"]
    print(f"  CAGR:     {s['cagr_pct']*100:>6.2f}%")
    print(f"  Sharpe:   {s['sharpe']:>6.2f}")
    print(f"  MaxDD:    {s['max_dd_pct']*100:>6.2f}%")
    print(f"  Calmar:   {s['calmar']:>6.2f}")
    print(f"  PF:       {s['profit_factor']:>6.2f}")
    print(f"  WinRate:  {s['win_rate_pct']*100:>6.2f}%")
    print(f"  Trades:   {s['n_trades']}")
    print(f"  Final Eq: Rs {s['final_equity']:>12,.0f}")

    print("\n=== COMPOSITE MID (mean_reversion, mid caps) ===")
    r2 = run_strategy_backtest(CompositeMidStrategy(), history_mid, get_sector, cfg=cfg)
    s2 = r2["stats"]
    print(f"  CAGR:     {s2['cagr_pct']*100:>6.2f}%")
    print(f"  Sharpe:   {s2['sharpe']:>6.2f}")
    print(f"  MaxDD:    {s2['max_dd_pct']*100:>6.2f}%")
    print(f"  Calmar:   {s2['calmar']:>6.2f}")
    print(f"  PF:       {s2['profit_factor']:>6.2f}")
    print(f"  Trades:   {s2['n_trades']}")
    print(f"  Final Eq: Rs {s2['final_equity']:>12,.0f}")

    # ── Portfolio mix: 70% LARGE composite + 30% MID mean-reversion ──
    print("\n=== PORTFOLIO BLEND: 70% LARGE composite + 30% MID mean-rev ===")
    eq_lg = r1["equity_curve"]["equity"]
    eq_mid = r2["equity_curve"]["equity"]
    common = eq_lg.index.intersection(eq_mid.index)
    blend = 0.70 * (eq_lg.loc[common] / eq_lg.iloc[0]) + 0.30 * (eq_mid.loc[common] / eq_mid.iloc[0])
    blend_eq = blend * cfg.initial_capital
    final = float(blend_eq.iloc[-1])
    years = max((blend_eq.index[-1] - blend_eq.index[0]).days / 365.25, 1e-6)
    cagr = (final / cfg.initial_capital) ** (1.0 / years) - 1.0
    daily = blend_eq.pct_change().dropna()
    import numpy as np
    sharpe = float((daily.mean() / daily.std()) * np.sqrt(252)) if daily.std() > 0 else 0.0
    dd = (blend_eq / blend_eq.cummax() - 1.0).min()
    calmar = (cagr / abs(dd)) if dd < 0 else float("inf")
    print(f"  CAGR:     {cagr*100:>6.2f}%")
    print(f"  Sharpe:   {sharpe:>6.2f}")
    print(f"  MaxDD:    {float(dd)*100:>6.2f}%")
    print(f"  Calmar:   {calmar:>6.2f}")
    print(f"  Final Eq: Rs {final:>12,.0f}")

    print("\n=== COMPARISON ===")
    print(f"  NIFTY 50 B&H:                 CAGR  8.68%  Sharpe 0.73  DD -15.77%")
    print(f"  Original screener (HTR M2):   CAGR 18.71%  Sharpe 1.02  DD -25.98%")
    print(f"  NEW Composite (LARGE only):   CAGR {s['cagr_pct']*100:>5.2f}%  Sharpe {s['sharpe']:.2f}  DD {s['max_dd_pct']*100:>6.2f}%")
    print(f"  NEW Portfolio Blend 70/30:    CAGR {cagr*100:>5.2f}%  Sharpe {sharpe:.2f}  DD {float(dd)*100:>6.2f}%")

if __name__ == "__main__":
    test()
