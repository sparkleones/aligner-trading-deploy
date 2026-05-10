"""
Round 2 sweep: try configs that actually have a shot.
- Pure momentum (no other factors muddying signal)
- No-stop hold-to-rebalance (let winners run)
- Tighter universe (top liquidity quartile)
- Longer momentum lookback (6M)
"""
from __future__ import annotations

import pandas as pd
import numpy as np

from .universe import get_universe
from .data_loader import load_universe
from .backtest import BacktestConfig, run_backtest


def sweep():
    universe = get_universe()
    history = load_universe(universe, period="4y", use_cache=True, progress=False)
    print(f"Loaded {len(history)} symbols\n")

    base = dict(
        start_date="2023-01-01",
        end_date="2026-04-30",
        initial_capital=100_000.0,
        n_picks=2,
    )

    configs = [
        # label, overrides
        ("Pure momentum 12-1 (W)", {
            "rebalance_freq": "W-MON",
            "factor_weights": {"momentum_12_1": 1.0},
        }),
        ("Pure momentum 12-1 (M)", {
            "rebalance_freq": "BMS",
            "factor_weights": {"momentum_12_1": 1.0},
        }),
        ("Mom + Trend (W)", {
            "rebalance_freq": "W-MON",
            "factor_weights": {"momentum_12_1": 0.6, "trend": 0.4},
        }),
        ("Hold-to-rotate W (no SL hit)", {
            "rebalance_freq": "W-MON",
            "min_stop_pct": 0.20,    # effectively disable hard stop (>20% gap)
            "max_stop_pct": 0.20,
            "reward_risk_ratio": 5.0,
            "time_stop_bars": 60,
        }),
        ("Hold-to-rotate M", {
            "rebalance_freq": "BMS",
            "min_stop_pct": 0.20,
            "max_stop_pct": 0.20,
            "reward_risk_ratio": 5.0,
            "time_stop_bars": 60,
        }),
        ("Top liquidity only (>1e10)", {
            "rebalance_freq": "W-MON",
            "min_liquidity_log10_inr": 10.0,  # >₹1000 Cr/day
        }),
        ("3 picks weekly", {
            "rebalance_freq": "W-MON",
            "n_picks": 3,
        }),
    ]

    rows = []
    for label, override in configs:
        kwargs = {**base, **override}
        cfg = BacktestConfig(**kwargs)
        r = run_backtest(history, cfg=cfg)
        s = r["stats"]
        rows.append({
            "config": label,
            "CAGR%":  f"{s['cagr_pct']*100:.2f}",
            "Sharpe": f"{s['sharpe']:.2f}",
            "MaxDD%": f"{s['max_drawdown_pct']*100:.2f}",
            "Calmar": f"{s['calmar']:.2f}",
            "WinRt%": f"{s['win_rate_pct']*100:.1f}",
            "PF":     f"{s['profit_factor']:.2f}",
            "N":      s["n_trades"],
        })
        print(f"  {label:>32s}: CAGR={s['cagr_pct']*100:>6.2f}%  "
              f"Sharpe={s['sharpe']:.2f}  DD={s['max_drawdown_pct']*100:>6.2f}%  "
              f"PF={s['profit_factor']:.2f}  N={s['n_trades']}")

    print("\n" + pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    sweep()
