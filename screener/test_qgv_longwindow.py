"""
Test QGV with longer history window — try 2019-2026 (7 years).
The 2020-2021 bull market is where SC MFs made most of their alpha.
If our QGV strategy works in that window, the recent chop is just bad luck.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data_loader import load_universe
from .universe import get_sector
from .universe_extended import SMALL_CAP, MID_CAP
from .strategies.qgv import QGVStrategy
from .backtest_qgv_mf import QGVMFConfig, run_qgv_backtest


def run(label, history, sector_fn, start, end, **cfg_overrides):
    cfg = QGVMFConfig(
        start_date=start, end_date=end,
        initial_capital=100_000, **cfg_overrides,
    )
    res = run_qgv_backtest(QGVStrategy(), history, sector_fn, cfg=cfg)
    return res.get("stats", {})


def _row(label, s, init_cap=100_000):
    cagr = s.get("cagr_pct", 0)
    sharpe = s.get("sharpe", 0)
    dd = s.get("max_dd_pct", 0)
    final = s.get("final_equity", 0)
    total_inv = s.get("total_invested", init_cap)
    profit = final - total_inv
    n = s.get("n_trades", 0)
    yrs = s.get("years", 0)
    print(f"  {label:<48} {yrs:>4.1f}y  {cagr*100:>8.2f}% {sharpe:>7.2f} {dd*100:>9.2f}% "
          f"Rs {final:>12,.0f}  Rs {profit:>+11,.0f}  N {n:>3d}")


def main():
    print("=" * 120)
    print(" QGV LONGER-WINDOW TEST — extend back to 2019/2020 to capture bull cycle")
    print("=" * 120)

    print("\nLoading max history...")
    sc = load_universe(SMALL_CAP, period="max", use_cache=True, progress=False)
    mc = load_universe(MID_CAP, period="max", use_cache=True, progress=False)
    print(f"  SC: {len(sc)}  MC: {len(mc)}\n")

    print(f"  {'Scenario':<48} {'Years':>5} {'CAGR':>9} {'Sharpe':>7} {'MaxDD':>10} "
          f"{'Final':>14}  {'Profit':>13}  {'N':>5}")
    print("-" * 130)

    # 7-year windows
    for label_prefix, history, sector_fn in [("SC", sc, get_sector), ("MC", mc, get_sector),
                                              ("Combined SC+MC", {**sc, **mc}, get_sector)]:
        for start, end in [("2019-01-01", "2026-04-30"),
                             ("2020-04-01", "2026-04-30"),  # post-COVID start
                             ("2021-01-01", "2026-04-30")]:
            scenario = f"{label_prefix} {start[:7]} -> {end[:7]}, 1L lump"
            s = run(scenario, history, sector_fn, start, end, n_picks=12, monthly_sip=0)
            _row(scenario, s)
        # With SIP
        s = run(f"{label_prefix} 2020-04 SIP 10k/mo + 1L",
                  history, sector_fn, "2020-04-01", "2026-04-30",
                  n_picks=12, monthly_sip=10_000)
        _row(f"{label_prefix} 2020-04 SIP 10k/mo + 1L", s)
        print()

    print("-" * 130)
    print(" BENCHMARKS (~5-7y CAGR typical):")
    print(f"  {'Nippon India SC':<48}  ~31% / 5y")
    print(f"  {'HDFC SC':<48}  ~28% / 5y")
    print(f"  {'NIFTY Smallcap 250 (passive)':<48}  ~20% / 5y")
    print(f"  {'NIFTY Midcap 150 (passive)':<48}  ~22% / 5y")


if __name__ == "__main__":
    main()
