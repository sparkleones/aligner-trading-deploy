"""
Hypothesis: run composite_top STRATEGY through the QGV-MF ENGINE
(annual rebal, dynamic hold, dip-buy, sector-weighted, no fixed SL).

This isolates the question: does the ENGINE help even if the STRATEGY
stays the same?

Compare:
  A. composite_top via backtest_mf_style (the original 4/6 wins)
  B. composite_top via QGV-MF engine (annual rebal + dip-buy + sector)
  C. QGV strategy via QGV-MF engine (the 2/6 wins from prior test)
  D. Ensemble: 70% (B) + 30% (C)
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
from pathlib import Path

from .data_loader import load_universe
from .market_timing_analyzer import _fetch_nifty
from .universe import get_sector
from .universe_extended import LARGE_CAP
from .strategies.qgv import QGVStrategy
from .strategies.composite_top import CompositeTopStrategy
from .backtest_qgv_mf import QGVMFConfig, run_qgv_backtest
from .backtest_mf_style import MFStyleConfig, run_mf_backtest


def _nifty_bench(nifty_df, start, end, init=100_000):
    sub = nifty_df[(nifty_df.index >= pd.Timestamp(start)) & (nifty_df.index <= pd.Timestamp(end))]
    if sub.empty or len(sub) < 10:
        return {"cagr": 0}
    s = float(sub["Close"].iloc[0]); e = float(sub["Close"].iloc[-1])
    yrs = max((sub.index[-1] - sub.index[0]).days / 365.25, 1e-6)
    return {"cagr": (e / s) ** (1.0 / yrs) - 1.0}


def rolling_windows(start="2022-01-01", end="2026-04-30", window_months=18, step_months=6):
    cur = pd.Timestamp(start)
    end_d = pd.Timestamp(end)
    out = []
    while cur + pd.DateOffset(months=window_months) <= end_d:
        win_end = cur + pd.DateOffset(months=window_months)
        out.append((cur.strftime("%Y-%m-%d"), win_end.strftime("%Y-%m-%d")))
        cur = cur + pd.DateOffset(months=step_months)
    return out


def main():
    print("=" * 130)
    print(" Test: composite_top STRATEGY in QGV ENGINE + ensemble")
    print("=" * 130)

    print("\nLoading data (period=max)...")
    lg = load_universe(LARGE_CAP, period="max", use_cache=True, progress=False)
    nifty = _fetch_nifty()
    print(f"  Universe: {len(lg)}  NIFTY: {len(nifty)} bars\n")

    windows = rolling_windows()
    rows = []

    print(f"{'Window':<28} {'A: comp/orig':>12} {'B: comp/QGV-eng':>16} {'C: QGV/QGV':>12} {'D: 70B+30C':>11} {'NIFTY':>8}  {'B beats':>8} {'D beats':>8}")
    print("-" * 130)

    for start, end in windows:
        # A: composite_top via original MF-style engine
        cfg_a = MFStyleConfig(start_date=start, end_date=end, initial_capital=100_000,
                                n_picks=5, rebalance_freq="QS", apply_stcg_tax=True)
        ra = run_mf_backtest(CompositeTopStrategy(), lg, get_sector, cfg=cfg_a)
        a_cagr = ra["stats"].get("cagr_pct", 0)

        # B: composite_top via QGV-MF engine (dynamic hold + dip-buy + sector wt)
        cfg_b = QGVMFConfig(start_date=start, end_date=end, initial_capital=100_000,
                              n_picks=12, annual_rebal_freq="YS", slippage_bps=10)
        rb = run_qgv_backtest(CompositeTopStrategy(), lg, get_sector, cfg=cfg_b)
        b_cagr = rb["stats"].get("cagr_pct", 0)

        # C: QGV via QGV engine (defensive quality)
        rc = run_qgv_backtest(QGVStrategy(), lg, get_sector, cfg=cfg_b)
        c_cagr = rc["stats"].get("cagr_pct", 0)

        # D: 70% B + 30% C (estimated by equity-curve blending)
        eq_b = rb["equity_curve"]["equity"]
        eq_c = rc["equity_curve"]["equity"]
        if not eq_b.empty and not eq_c.empty:
            common = eq_b.index.intersection(eq_c.index)
            if len(common) >= 10:
                blend = 0.7 * (eq_b.loc[common] / eq_b.iloc[0]) + 0.3 * (eq_c.loc[common] / eq_c.iloc[0])
                eq_blend = blend * 100_000
                final = float(eq_blend.iloc[-1])
                yrs = max((eq_blend.index[-1] - eq_blend.index[0]).days / 365.25, 1e-6)
                d_cagr = (final / 100_000) ** (1.0 / yrs) - 1.0
            else:
                d_cagr = 0
        else:
            d_cagr = 0

        n_cagr = _nifty_bench(nifty, start, end)["cagr"]
        b_beats = "Y" if (b_cagr - n_cagr) > 0 else "N"
        d_beats = "Y" if (d_cagr - n_cagr) > 0 else "N"
        a_beats = "Y" if (a_cagr - n_cagr) > 0 else "N"

        print(f"{start} -> {end[:10]}   "
              f"{a_cagr*100:>11.2f}% {b_cagr*100:>15.2f}% {c_cagr*100:>11.2f}% {d_cagr*100:>10.2f}% {n_cagr*100:>7.2f}%  "
              f"{b_beats:>8} {d_beats:>8}")

        rows.append({
            "window": f"{start} -> {end}",
            "A_composite_original": a_cagr,
            "B_composite_qgv_engine": b_cagr,
            "C_qgv_qgv_engine": c_cagr,
            "D_blend_70B_30C": d_cagr,
            "nifty": n_cagr,
            "A_beats": (a_cagr - n_cagr) > 0,
            "B_beats": (b_cagr - n_cagr) > 0,
            "D_beats": (d_cagr - n_cagr) > 0,
        })

    print("\n" + "-" * 130)
    a_w = sum(1 for r in rows if r["A_beats"])
    b_w = sum(1 for r in rows if r["B_beats"])
    d_w = sum(1 for r in rows if r["D_beats"])
    print(f"  A (composite + original engine) beats NIFTY: {a_w}/{len(rows)}  median CAGR {np.median([r['A_composite_original'] for r in rows])*100:+.2f}%")
    print(f"  B (composite + QGV engine)      beats NIFTY: {b_w}/{len(rows)}  median CAGR {np.median([r['B_composite_qgv_engine'] for r in rows])*100:+.2f}%")
    print(f"  D (70%B + 30%C blend)           beats NIFTY: {d_w}/{len(rows)}  median CAGR {np.median([r['D_blend_70B_30C'] for r in rows])*100:+.2f}%")

    out_dir = Path(__file__).resolve().parent.parent / "reports" / "screener"
    findings = {"test": "composite in QGV engine + ensemble", "windows": rows,
                  "summary": {"A_wins": a_w, "B_wins": b_w, "D_wins": d_w}}
    with open(out_dir / "composite_in_qgv_engine.json", "w") as f:
        json.dump(findings, f, indent=2, default=str)


if __name__ == "__main__":
    main()
