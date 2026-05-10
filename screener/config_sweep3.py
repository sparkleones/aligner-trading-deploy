"""
Round 3: focus on hold-to-rotate winners. Vary holding period, picks,
sector caps, factor mix. Also test walk-forward robustness — split into
4 sub-windows.
"""
from __future__ import annotations

import pandas as pd

from .universe import get_universe
from .data_loader import load_universe
from .backtest import BacktestConfig, run_backtest


HOLD_TO_ROTATE = dict(
    min_stop_pct=0.20,
    max_stop_pct=0.20,
    reward_risk_ratio=5.0,
    time_stop_bars=60,
)


def sweep_main(history):
    print("\n=== ROUND 3: focused sweep around hold-to-rotate ===\n")
    base = dict(
        start_date="2023-01-01",
        end_date="2026-04-30",
        initial_capital=100_000.0,
        n_picks=2,
        **HOLD_TO_ROTATE,
    )
    configs = [
        ("HTR Monthly 2pk default",       {"rebalance_freq": "BMS"}),
        ("HTR Monthly 2pk pure-mom",      {"rebalance_freq": "BMS",
                                           "factor_weights": {"momentum_12_1": 1.0}}),
        ("HTR Monthly 3pk default",       {"rebalance_freq": "BMS", "n_picks": 3}),
        ("HTR Monthly 2pk no-sec-cap",    {"rebalance_freq": "BMS", "max_per_sector": 5}),
        ("HTR Fortnight 2pk default",     {"rebalance_freq": "2W-MON"}),
        ("HTR Weekly 2pk default",        {"rebalance_freq": "W-MON"}),
        ("HTR Monthly 2pk above-200dma=N",{"rebalance_freq": "BMS", "require_above_200dma": False}),
        ("HTR Monthly 2pk no-rev",        {"rebalance_freq": "BMS",
                                           "factor_weights": {"momentum_12_1": 0.5,
                                                              "trend": 0.3,
                                                              "low_vol": 0.2}}),
    ]
    rows = []
    for label, override in configs:
        cfg = BacktestConfig(**{**base, **override})
        r = run_backtest(history, cfg=cfg)
        s = r["stats"]
        rows.append({"config": label,
                     "CAGR%":  f"{s['cagr_pct']*100:.2f}",
                     "Sharpe": f"{s['sharpe']:.2f}",
                     "MaxDD%": f"{s['max_drawdown_pct']*100:.2f}",
                     "Calmar": f"{s['calmar']:.2f}",
                     "WinRt%": f"{s['win_rate_pct']*100:.1f}",
                     "PF":     f"{s['profit_factor']:.2f}",
                     "N":      s["n_trades"]})
        print(f"  {label:>32s}: CAGR={s['cagr_pct']*100:>6.2f}%  "
              f"Sharpe={s['sharpe']:.2f}  DD={s['max_drawdown_pct']*100:>6.2f}%  "
              f"PF={s['profit_factor']:.2f}  N={s['n_trades']}")
    print("\n" + pd.DataFrame(rows).to_string(index=False))


def walkforward(history):
    """Test that the winning config holds up across multiple time windows."""
    print("\n=== WALK-FORWARD: HTR Monthly 2pk default across 4 windows ===\n")
    windows = [
        ("2022-01-01", "2023-06-30"),  # bull
        ("2022-07-01", "2023-12-31"),  # mixed
        ("2023-01-01", "2024-06-30"),  # bull
        ("2024-01-01", "2025-06-30"),  # mid
        ("2024-07-01", "2026-04-30"),  # recent
    ]
    rows = []
    for start, end in windows:
        cfg = BacktestConfig(
            start_date=start, end_date=end,
            initial_capital=100_000.0,
            n_picks=2,
            rebalance_freq="BMS",
            **HOLD_TO_ROTATE,
        )
        r = run_backtest(history, cfg=cfg)
        s = r["stats"]
        rows.append({"window": f"{start[:7]} - {end[:7]}",
                     "CAGR%":  f"{s['cagr_pct']*100:.2f}",
                     "Sharpe": f"{s['sharpe']:.2f}",
                     "MaxDD%": f"{s['max_drawdown_pct']*100:.2f}",
                     "PF":     f"{s['profit_factor']:.2f}",
                     "N":      s["n_trades"]})
        print(f"  {start[:7]}..{end[:7]}: CAGR={s['cagr_pct']*100:>6.2f}%  "
              f"Sharpe={s['sharpe']:.2f}  DD={s['max_drawdown_pct']*100:>6.2f}%  "
              f"PF={s['profit_factor']:.2f}  N={s['n_trades']}")
    print("\n" + pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    history = load_universe(get_universe(), period="5y", use_cache=True, progress=False)
    print(f"Loaded {len(history)} symbols")
    sweep_main(history)
    walkforward(history)
