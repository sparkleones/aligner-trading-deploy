"""
Rolling walk-forward v2 — REGIME-ADAPTIVE ensemble.

Compare against:
  - NIFTY 50 buy & hold (the bar to beat)
  - Old composite_top (pure momentum) — to quantify improvement

If the new approach truly works in all regimes, we expect:
  - Beats NIFTY in 5+/6 windows
  - Lower max DD in choppy windows
  - No outsized drawdowns from forced stage-2 exits
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
from pathlib import Path

from .data_loader import load_universe
from .market_timing_analyzer import _fetch_nifty, _fetch_vix
from .backtest_regime_adaptive import RegimeAdaptiveConfig, run_regime_backtest
from .backtest_mf_style import MFStyleConfig, run_mf_backtest
from .strategies.composite_top import CompositeTopStrategy
from .universe import get_sector
from .universe_extended import LARGE_CAP


def _nifty_bench(nifty_df, start, end, init=100_000):
    sub = nifty_df[(nifty_df.index >= pd.Timestamp(start)) & (nifty_df.index <= pd.Timestamp(end))]
    if sub.empty or len(sub) < 10:
        return {"cagr": 0, "sharpe": 0, "dd": 0}
    s = float(sub["Close"].iloc[0])
    e = float(sub["Close"].iloc[-1])
    years = max((sub.index[-1] - sub.index[0]).days / 365.25, 1e-6)
    cagr = (e / s) ** (1.0 / years) - 1.0
    eq = sub["Close"] / s * init
    daily = eq.pct_change().dropna()
    sharpe = float((daily.mean() / daily.std()) * np.sqrt(252)) if daily.std() > 0 else 0.0
    dd = (eq / eq.cummax() - 1.0).min()
    return {"cagr": cagr, "sharpe": sharpe, "dd": float(dd)}


def rolling_windows(start, end, window_months=18, step_months=6):
    cur = pd.Timestamp(start)
    end_d = pd.Timestamp(end)
    out = []
    while cur + pd.DateOffset(months=window_months) <= end_d:
        win_end = cur + pd.DateOffset(months=window_months)
        out.append((cur.strftime("%Y-%m-%d"), win_end.strftime("%Y-%m-%d")))
        cur = cur + pd.DateOffset(months=step_months)
    return out


def main():
    print("=" * 78)
    print(" ROLLING WALK-FORWARD v2 — Regime-Adaptive Ensemble")
    print(" Allocates dynamically: momentum / low-vol / mean-rev / cash")
    print("=" * 78)

    print("\nLoading data...")
    lg = load_universe(LARGE_CAP, period="5y", use_cache=True, progress=False)
    nifty = _fetch_nifty()
    vix = _fetch_vix()
    print(f"  Universe: {len(lg)} stocks  |  NIFTY: {len(nifty)} bars  |  VIX: {len(vix) if vix is not None else 0} bars\n")

    windows = rolling_windows("2022-01-01", "2026-04-30", 18, 6)
    print(f"Running {len(windows)} windows (REGIME-ADAPTIVE)...\n")

    print(f"{'Window':<28} {'Adaptive':>10} {'Sharpe':>8} {'DD':>9} {'NIFTY':>10} {'Alpha':>10} {'Old Mom':>10} {'Mom Alpha':>10}")
    print("-" * 115)

    results = []
    for start, end in windows:
        # Regime-adaptive run
        cfg_a = RegimeAdaptiveConfig(start_date=start, end_date=end,
                                      initial_capital=100_000, n_picks=5,
                                      rebalance_freq="QS")
        ra = run_regime_backtest(lg, nifty, vix, get_sector, cfg=cfg_a)
        sa = ra["stats"]

        # Old pure-momentum run (baseline)
        cfg_m = MFStyleConfig(start_date=start, end_date=end,
                               initial_capital=100_000, n_picks=5,
                               rebalance_freq="QS", apply_stcg_tax=True)
        rm = run_mf_backtest(CompositeTopStrategy(), lg, get_sector, cfg=cfg_m)
        sm = rm["stats"]

        # NIFTY
        n = _nifty_bench(nifty, start, end)

        alpha_adaptive = sa.get("cagr_pct", 0) - n["cagr"]
        alpha_momentum = sm.get("cagr_pct", 0) - n["cagr"]
        adaptive_beats = "Y" if alpha_adaptive > 0 else "N"

        print(f"{start} -> {end[:10]}  "
              f"{sa.get('cagr_pct', 0)*100:>9.2f}% "
              f"{sa.get('sharpe', 0):>8.2f} "
              f"{sa.get('max_dd_pct', 0)*100:>8.2f}% "
              f"{n['cagr']*100:>9.2f}% "
              f"{alpha_adaptive*100:>9.2f}% "
              f"{sm.get('cagr_pct', 0)*100:>9.2f}% "
              f"{alpha_momentum*100:>9.2f}% {adaptive_beats}")

        # Tally regimes seen this window
        regime_counts = {}
        for entry in ra.get("regime_history", []):
            r = entry.get("regime")
            r = r.value if hasattr(r, "value") else r
            regime_counts[str(r)] = regime_counts.get(str(r), 0) + 1

        results.append({
            "window": f"{start} -> {end}",
            "adaptive_cagr": sa.get("cagr_pct", 0),
            "adaptive_sharpe": sa.get("sharpe", 0),
            "adaptive_dd": sa.get("max_dd_pct", 0),
            "momentum_cagr": sm.get("cagr_pct", 0),
            "nifty_cagr": n["cagr"],
            "adaptive_alpha": alpha_adaptive,
            "momentum_alpha": alpha_momentum,
            "adaptive_beats_nifty": alpha_adaptive > 0,
            "regimes_seen": regime_counts,
        })

    if not results:
        print("No results")
        return

    n_beats = sum(1 for r in results if r["adaptive_beats_nifty"])
    median_alpha = np.median([r["adaptive_alpha"] for r in results])
    median_cagr = np.median([r["adaptive_cagr"] for r in results])
    median_sharpe = np.median([r["adaptive_sharpe"] for r in results])
    median_dd = np.median([r["adaptive_dd"] for r in results])
    worst = min(r["adaptive_cagr"] for r in results)
    best = max(r["adaptive_cagr"] for r in results)
    n_beats_mom = sum(1 for r in results if r["momentum_alpha"] > 0)

    print("\n" + "=" * 78)
    print(" SUMMARY")
    print("=" * 78)
    print(f"  Adaptive beats NIFTY:    {n_beats} of {len(results)} windows")
    print(f"  Old momentum beats:      {n_beats_mom} of {len(results)} windows  (for reference)")
    print(f"  Adaptive median CAGR:    {median_cagr*100:+.2f}%")
    print(f"  Adaptive median Sharpe:  {median_sharpe:.2f}")
    print(f"  Adaptive median MaxDD:   {median_dd*100:+.2f}%")
    print(f"  Adaptive median alpha:   {median_alpha*100:+.2f}%/yr")
    print(f"  Worst window:            {worst*100:+.2f}%")
    print(f"  Best window:             {best*100:+.2f}%")

    # Save findings
    out_dir = Path(__file__).resolve().parent.parent / "reports" / "screener"
    out_dir.mkdir(parents=True, exist_ok=True)
    findings = {
        "engine": "regime-adaptive ensemble",
        "windows": results,
        "summary": {
            "n_windows": len(results),
            "n_beating_nifty": n_beats,
            "median_cagr": float(median_cagr),
            "median_sharpe": float(median_sharpe),
            "median_dd": float(median_dd),
            "median_alpha": float(median_alpha),
            "worst_cagr": float(worst),
            "best_cagr": float(best),
        },
        "generated_at": pd.Timestamp.now().isoformat(),
    }
    with open(out_dir / "rolling_walkforward_v2.json", "w") as f:
        json.dump(findings, f, indent=2, default=str)
    print(f"\n[OK] Saved to {out_dir/'rolling_walkforward_v2.json'}")


if __name__ == "__main__":
    main()
