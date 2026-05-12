"""Walk-forward validation on the composite portfolio blend."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data_loader import load_universe
from .strategies.composite_top import CompositeTopStrategy, CompositeMidStrategy
from .strategy_backtest import StrategyBacktestConfig, run_strategy_backtest
from .universe_extended import LARGE_CAP, MID_CAP
from .universe import get_sector
from .config_sweep import benchmark_nifty


def run_blend(lg_hist, mid_hist, start, end, init=100_000.0):
    cfg = StrategyBacktestConfig(
        start_date=start, end_date=end, initial_capital=init,
        n_picks=2, rebalance_freq="BMS", time_stop_bars=60, max_per_sector=1,
    )
    r1 = run_strategy_backtest(CompositeTopStrategy(), lg_hist, get_sector, cfg=cfg)
    r2 = run_strategy_backtest(CompositeMidStrategy(), mid_hist, get_sector, cfg=cfg)

    eq_lg = r1["equity_curve"]["equity"]
    eq_mid = r2["equity_curve"]["equity"]
    common = eq_lg.index.intersection(eq_mid.index)
    if len(common) < 5:
        return None
    blend = 0.70 * (eq_lg.loc[common] / eq_lg.iloc[0]) + 0.30 * (eq_mid.loc[common] / eq_mid.iloc[0])
    eq = blend * init
    final = float(eq.iloc[-1])
    years = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-6)
    cagr = (final / init) ** (1.0 / years) - 1.0
    daily = eq.pct_change().dropna()
    sharpe = float((daily.mean() / daily.std()) * np.sqrt(252)) if daily.std() > 0 else 0.0
    dd = (eq / eq.cummax() - 1.0).min()
    return dict(cagr=cagr, sharpe=sharpe, dd=float(dd), final=final, years=years)


def main():
    print("Loading 5-yr history for walk-forward...")
    lg = load_universe(LARGE_CAP, period="5y", use_cache=True, progress=False)
    mid = load_universe(MID_CAP, period="5y", use_cache=True, progress=False)
    print(f"  LARGE: {len(lg)}  MID: {len(mid)}\n")

    windows = [
        ("2022-01-01", "2023-06-30"),
        ("2022-07-01", "2023-12-31"),
        ("2023-01-01", "2024-06-30"),
        ("2024-01-01", "2025-06-30"),
        ("2024-07-01", "2026-04-30"),
    ]
    print(f"{'Window':<22} {'Blend CAGR':>12} {'Sharpe':>8} {'DD':>10} {'NIFTY CAGR':>12} {'Alpha':>10} {'Beats?':>8}")
    print("-" * 90)
    rows = []
    for start, end in windows:
        b = run_blend(lg, mid, start, end)
        n = benchmark_nifty(start, end)
        if not b or not n:
            continue
        alpha = b["cagr"] - n.get("cagr_pct", 0)
        beats = "YES" if alpha > 0 else "NO"
        win = f"{start[:7]}..{end[:7]}"
        print(f"{win:<22} {b['cagr']*100:>11.2f}% {b['sharpe']:>8.2f} {b['dd']*100:>9.2f}% "
              f"{n.get('cagr_pct',0)*100:>11.2f}% {alpha*100:>9.2f}% {beats:>8}")
        rows.append(dict(window=win, blend_cagr=b["cagr"], blend_sharpe=b["sharpe"],
                         blend_dd=b["dd"], nifty_cagr=n.get("cagr_pct", 0), alpha=alpha))

    # Summary
    if rows:
        n_beats = sum(1 for r in rows if r["alpha"] > 0)
        avg_alpha = np.mean([r["alpha"] for r in rows])
        avg_sharpe = np.mean([r["blend_sharpe"] for r in rows])
        print()
        print(f"VERDICT: Beats NIFTY in {n_beats}/{len(rows)} walk-forward windows.")
        print(f"         Avg alpha = {avg_alpha*100:>5.2f}%/year   Avg Sharpe = {avg_sharpe:.2f}")


if __name__ == "__main__":
    main()
