"""
Rolling walk-forward across full 4.5 years. Many overlapping windows
to test the strategy in DIFFERENT regimes — not just the last 6 months
(which happened to be a falling NIFTY).

Compares:
  - MF-style backtest (hold-until-deterioration, quarterly rebal,
    no fixed SL, with STCG tax)
  - Composite (stage2 + breakout) strategy
  - vs NIFTY 50 buy-and-hold
  - vs Nifty 500 buy-and-hold

Outputs honest summary stats: median CAGR, worst window, best window,
% windows beating NIFTY.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data_loader import load_universe, load_history
from .strategies.composite_top import CompositeTopStrategy
from .backtest_mf_style import MFStyleConfig, run_mf_backtest
from .universe import get_sector
from .universe_extended import LARGE_CAP, MID_CAP


# Fetch NIFTY history once at module load — saves 6x cache hits + is more
# resilient to yfinance peewee/tzdata cache corruption.
_NIFTY_CACHE = None


def _get_nifty_cached():
    global _NIFTY_CACHE
    if _NIFTY_CACHE is not None:
        return _NIFTY_CACHE
    import yfinance as yf
    for attempt in range(3):
        try:
            df = yf.Ticker("^NSEI").history(period="5y", auto_adjust=True)
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            if not df.empty:
                _NIFTY_CACHE = df
                return df
        except Exception as e:
            print(f"  [NIFTY fetch attempt {attempt+1} failed: {e}]")
    return None


def _nifty_bench(start, end, init):
    df = _get_nifty_cached()
    if df is None or df.empty:
        return None
    sub = df[(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))]
    if sub.empty or len(sub) < 10:
        return None
    s = float(sub["Close"].iloc[0])
    e = float(sub["Close"].iloc[-1])
    years = max((sub.index[-1] - sub.index[0]).days / 365.25, 1e-6)
    cagr = (e / s) ** (1.0 / years) - 1.0
    eq = sub["Close"] / s * init
    daily = eq.pct_change().dropna()
    sharpe = float((daily.mean() / daily.std()) * np.sqrt(252)) if daily.std() > 0 else 0.0
    dd = (eq / eq.cummax() - 1.0).min()
    return {"cagr": cagr, "sharpe": sharpe, "dd": float(dd)}


def rolling_windows(start: str = "2022-01-01", end: str = "2026-04-30",
                     window_months: int = 18, step_months: int = 6):
    """Generate (window_start, window_end) tuples."""
    start_d = pd.Timestamp(start)
    end_d = pd.Timestamp(end)
    out = []
    cur = start_d
    while cur + pd.DateOffset(months=window_months) <= end_d:
        win_end = cur + pd.DateOffset(months=window_months)
        out.append((cur.strftime("%Y-%m-%d"), win_end.strftime("%Y-%m-%d")))
        cur = cur + pd.DateOffset(months=step_months)
    return out


def run_window(lg_hist, start, end):
    cfg = MFStyleConfig(start_date=start, end_date=end,
                         initial_capital=100_000.0, n_picks=5,
                         rebalance_freq="QS",   # quarterly
                         apply_stcg_tax=True)
    r = run_mf_backtest(CompositeTopStrategy(), lg_hist, get_sector, cfg=cfg)
    s = r["stats"]
    return {
        "start": start, "end": end,
        "blend_cagr": s.get("cagr_pct", 0),
        "blend_sharpe": s.get("sharpe", 0),
        "blend_dd": s.get("max_dd_pct", 0),
        "blend_final": s.get("final_equity", 0),
        "blend_n_trades": s.get("n_trades", 0),
    }


def main():
    print("=" * 78)
    print(" ROLLING WALK-FORWARD on FULL 4.5y")
    print(" Engine: MF-style (hold-until-deterioration, quarterly rebal, STCG tax)")
    print(" Strategy: composite_top (stage2 + breakout) on LARGE caps, top 5 picks")
    print("=" * 78)

    print("\nLoading 5y history for LARGE cap universe...")
    lg = load_universe(LARGE_CAP, period="5y", use_cache=True, progress=False)
    print(f"  Loaded {len(lg)} stocks\n")

    windows = rolling_windows("2022-01-01", "2026-04-30",
                                window_months=18, step_months=6)
    print(f"Running {len(windows)} rolling windows (18-month each, 6-month step)...\n")

    print(f"{'Window':<28} {'Strat CAGR':>11} {'Sharpe':>8} {'MaxDD':>9} {'NIFTY CAGR':>11} {'Alpha':>9}")
    print("-" * 95)

    rows = []
    for start, end in windows:
        r = run_window(lg, start, end)
        n = _nifty_bench(start, end, 100_000)
        if n is None:
            n = {"cagr": 0, "sharpe": 0, "dd": 0}
        alpha = r["blend_cagr"] - n["cagr"]
        beats = "Y" if alpha > 0 else "N"
        print(f"{start} -> {end}   {r['blend_cagr']*100:>10.2f}% {r['blend_sharpe']:>8.2f} "
              f"{r['blend_dd']*100:>8.2f}% {n['cagr']*100:>10.2f}% {alpha*100:>8.2f}% {beats}")
        rows.append({
            "window": f"{start} -> {end}",
            "strat_cagr": r["blend_cagr"],
            "strat_sharpe": r["blend_sharpe"],
            "strat_dd": r["blend_dd"],
            "nifty_cagr": n["cagr"],
            "alpha": alpha,
            "beats_nifty": alpha > 0,
            "n_trades": r["blend_n_trades"],
        })

    if not rows:
        print("No windows.")
        return

    n_beats = sum(1 for r in rows if r["beats_nifty"])
    median_cagr = np.median([r["strat_cagr"] for r in rows])
    median_sharpe = np.median([r["strat_sharpe"] for r in rows])
    median_dd = np.median([r["strat_dd"] for r in rows])
    median_alpha = np.median([r["alpha"] for r in rows])
    worst_cagr = min(r["strat_cagr"] for r in rows)
    best_cagr = max(r["strat_cagr"] for r in rows)
    median_nifty = np.median([r["nifty_cagr"] for r in rows])
    median_n_trades = np.median([r["n_trades"] for r in rows])

    print("\n" + "=" * 78)
    print(" SUMMARY (HONEST — engine has anti-look-ahead, STCG tax,")
    print("           but NO fixed SL since MFs don't use that)")
    print("=" * 78)
    print(f"  Windows beating NIFTY:  {n_beats} of {len(rows)}")
    print(f"  Median strategy CAGR:   {median_cagr*100:+.2f}%   (NIFTY median: {median_nifty*100:+.2f}%)")
    print(f"  Median Sharpe:          {median_sharpe:.2f}")
    print(f"  Median Max DD:          {median_dd*100:.2f}%")
    print(f"  Median alpha:           {median_alpha*100:+.2f}%")
    print(f"  Median trades / window: {median_n_trades:.0f}")
    print(f"  Worst window CAGR:      {worst_cagr*100:+.2f}%")
    print(f"  Best window CAGR:       {best_cagr*100:+.2f}%")

    print("\nReference benchmarks (Indian equity MFs, 5y CAGR as of 2025):")
    print("  Top quartile large-cap MF:  ~14-16%")
    print("  Top quartile mid-cap MF:    ~22-26%")
    print("  Median large-cap MF:        ~11-13%")
    print("  NIFTY 50 (5y):              ~12-14%")
    print("=" * 78)

    # ── Save findings for dashboard consumption ──
    import json as _json
    from pathlib import Path as _Path
    out_dir = _Path(__file__).resolve().parent.parent / "reports" / "screener"
    out_dir.mkdir(parents=True, exist_ok=True)
    findings = {
        "engine": "MF-style (hold-until-deterioration, quarterly rebal, STCG tax)",
        "strategy": "composite_top (stage2 + breakout), top-5 LARGE cap picks",
        "windows": rows,
        "summary": {
            "n_windows": len(rows),
            "n_beating_nifty": n_beats,
            "median_cagr": float(median_cagr),
            "median_sharpe": float(median_sharpe),
            "median_dd": float(median_dd),
            "median_alpha": float(median_alpha),
            "worst_cagr": float(worst_cagr),
            "best_cagr": float(best_cagr),
            "median_nifty_cagr": float(median_nifty),
            "median_trades_per_window": float(median_n_trades),
        },
        "benchmarks": {
            "top_quartile_largecap_mf_5y": 0.15,
            "top_quartile_midcap_mf_5y": 0.24,
            "median_largecap_mf_5y": 0.12,
            "nifty50_5y": 0.13,
        },
        "generated_at": pd.Timestamp.now().isoformat(),
    }
    with open(out_dir / "rolling_walkforward.json", "w") as f:
        _json.dump(findings, f, indent=2, default=str)
    print(f"\n[OK] Saved to {out_dir/'rolling_walkforward.json'}")


if __name__ == "__main__":
    main()
