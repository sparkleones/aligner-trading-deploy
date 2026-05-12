"""
Test the hypothesis: does QGV-MF engine on LARGE-CAP universe beat
NIFTY in all 6 rolling walk-forward windows?

Compares:
  A. QGV-MF on LARGE_CAP (the test)
  B. composite_top via backtest_mf_style (the previous baseline: 4/6)
  C. NIFTY 50 buy-and-hold (the bar to beat)
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
        return {"cagr": 0, "sharpe": 0, "dd": 0, "final": init}
    s = float(sub["Close"].iloc[0])
    e = float(sub["Close"].iloc[-1])
    years = max((sub.index[-1] - sub.index[0]).days / 365.25, 1e-6)
    cagr = (e / s) ** (1.0 / years) - 1.0
    eq = sub["Close"] / s * init
    daily = eq.pct_change().dropna()
    sharpe = float((daily.mean() / daily.std()) * np.sqrt(252)) if daily.std() > 0 else 0.0
    dd = (eq / eq.cummax() - 1.0).min()
    return {"cagr": cagr, "sharpe": sharpe, "dd": float(dd), "final": float(eq.iloc[-1])}


def rolling_windows(start, end, window_months=18, step_months=6):
    cur = pd.Timestamp(start)
    end_d = pd.Timestamp(end)
    out = []
    while cur + pd.DateOffset(months=window_months) <= end_d:
        win_end = cur + pd.DateOffset(months=window_months)
        out.append((cur.strftime("%Y-%m-%d"), win_end.strftime("%Y-%m-%d")))
        cur = cur + pd.DateOffset(months=step_months)
    return out


def run_qgv_window(history, start, end, n_picks):
    cfg = QGVMFConfig(
        start_date=start, end_date=end, initial_capital=100_000,
        n_picks=n_picks, annual_rebal_freq="YS",
        slippage_bps=10,  # large-cap slippage is lower
    )
    res = run_qgv_backtest(QGVStrategy(), history, get_sector, cfg=cfg)
    return res.get("stats", {})


def run_composite_window(history, start, end, n_picks):
    cfg = MFStyleConfig(
        start_date=start, end_date=end, initial_capital=100_000,
        n_picks=n_picks, rebalance_freq="QS", apply_stcg_tax=True,
    )
    res = run_mf_backtest(CompositeTopStrategy(), history, get_sector, cfg=cfg)
    return res.get("stats", {})


def main():
    print("=" * 120)
    print(" QGV-MF on LARGE-CAP — Can we beat NIFTY in all 6 windows?")
    print("=" * 120)

    print("\nLoading data (period=max for sufficient QGV warmup)...")
    lg = load_universe(LARGE_CAP, period="max", use_cache=True, progress=False)
    nifty = _fetch_nifty()
    # Diagnostic
    print(f"  Universe: {len(lg)} large-caps")
    if lg:
        sample = next(iter(lg.values()))
        print(f"  Sample: {len(sample)} bars, from {sample.index[0].date()} to {sample.index[-1].date()}")
    print(f"  NIFTY: {len(nifty)} bars\n")

    windows = rolling_windows("2022-01-01", "2026-04-30", 18, 6)
    print(f"{'Window':<28} {'QGV CAGR':>10} {'Sharpe':>7} {'DD':>9} {'NIFTY':>9} "
          f"{'QGV-α':>9}  {'Composite-α':>12}  {'QGV Beat?':>10}")
    print("-" * 120)

    rows = []
    for start, end in windows:
        qgv_stats = run_qgv_window(lg, start, end, n_picks=12)
        comp_stats = run_composite_window(lg, start, end, n_picks=5)
        nifty_stats = _nifty_bench(nifty, start, end)

        qgv_cagr = qgv_stats.get("cagr_pct", 0)
        qgv_sh = qgv_stats.get("sharpe", 0)
        qgv_dd = qgv_stats.get("max_dd_pct", 0)
        comp_cagr = comp_stats.get("cagr_pct", 0)
        n_cagr = nifty_stats["cagr"]

        qgv_alpha = qgv_cagr - n_cagr
        comp_alpha = comp_cagr - n_cagr
        beat = "Y" if qgv_alpha > 0 else "N"

        print(f"{start} -> {end[:10]}  "
              f"{qgv_cagr*100:>9.2f}% {qgv_sh:>7.2f} {qgv_dd*100:>8.2f}% "
              f"{n_cagr*100:>8.2f}% "
              f"{qgv_alpha*100:>8.2f}% "
              f"{comp_alpha*100:>11.2f}%  "
              f"{beat:>10}")

        rows.append({
            "window": f"{start} -> {end}",
            "qgv_cagr": qgv_cagr,
            "qgv_sharpe": qgv_sh,
            "qgv_dd": qgv_dd,
            "composite_cagr": comp_cagr,
            "nifty_cagr": n_cagr,
            "qgv_alpha": qgv_alpha,
            "composite_alpha": comp_alpha,
            "qgv_beats_nifty": qgv_alpha > 0,
            "composite_beats_nifty": comp_alpha > 0,
        })

    n_qgv_beats = sum(1 for r in rows if r["qgv_beats_nifty"])
    n_comp_beats = sum(1 for r in rows if r["composite_beats_nifty"])
    qgv_med_cagr = np.median([r["qgv_cagr"] for r in rows])
    qgv_med_alpha = np.median([r["qgv_alpha"] for r in rows])
    comp_med_alpha = np.median([r["composite_alpha"] for r in rows])
    worst_qgv = min(r["qgv_cagr"] for r in rows)
    best_qgv = max(r["qgv_cagr"] for r in rows)

    print("\n" + "=" * 120)
    print(" SUMMARY")
    print("=" * 120)
    print(f"  QGV beats NIFTY:           {n_qgv_beats} of {len(rows)} windows")
    print(f"  composite_top beats NIFTY: {n_comp_beats} of {len(rows)} windows  (baseline)")
    print(f"  QGV median CAGR:           {qgv_med_cagr*100:+.2f}%")
    print(f"  QGV median alpha:          {qgv_med_alpha*100:+.2f}%/yr")
    print(f"  composite_top median alpha:{comp_med_alpha*100:+.2f}%/yr")
    print(f"  QGV worst window:          {worst_qgv*100:+.2f}%")
    print(f"  QGV best window:           {best_qgv*100:+.2f}%")

    out_dir = Path(__file__).resolve().parent.parent / "reports" / "screener"
    out_dir.mkdir(parents=True, exist_ok=True)
    findings = {
        "test": "QGV-MF on LARGE_CAP universe",
        "windows": rows,
        "summary": {
            "n_qgv_beats_nifty": n_qgv_beats,
            "n_composite_beats_nifty": n_comp_beats,
            "qgv_median_cagr": float(qgv_med_cagr),
            "qgv_median_alpha": float(qgv_med_alpha),
            "composite_median_alpha": float(comp_med_alpha),
            "qgv_worst": float(worst_qgv),
            "qgv_best": float(best_qgv),
        },
        "generated_at": pd.Timestamp.now().isoformat(),
    }
    with open(out_dir / "qgv_largecap_walkforward.json", "w") as f:
        json.dump(findings, f, indent=2, default=str)
    print(f"\n[OK] Saved to {out_dir/'qgv_largecap_walkforward.json'}")


if __name__ == "__main__":
    main()
