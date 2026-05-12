"""
Run every strategy on every cap tier — produce the comparison matrix.

Fundamentals strategies are NOT backtested (snapshot data = look-ahead bias).
They're tested via current-state-only signal generation in a separate run.

Output:
  reports/screener/strategy_comparison.csv  — full matrix
  reports/screener/best_strategies.json     — top 3 per tier
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .data_loader import load_universe
from .strategies import STRATEGIES
from .strategy_backtest import StrategyBacktestConfig, run_strategy_backtest
from .universe_extended import (
    LARGE_CAP, MID_CAP, SMALL_CAP, ALL_STOCKS, get_cap_tier,
)
from .universe import get_sector


# Only backtest the strategies that DON'T need fundamentals (clean backtest)
BACKTESTABLE = {n: s for n, s in STRATEGIES.items() if not s.needs_fundamentals}
FUNDAMENTAL = {n: s for n, s in STRATEGIES.items() if s.needs_fundamentals}


def benchmark_nifty(start, end, init_cap=100_000.0) -> dict:
    """NIFTY 50 buy-and-hold for comparison."""
    import yfinance as yf
    try:
        df = yf.Ticker("^NSEI").history(period="5y", auto_adjust=True)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df = df[(df.index >= start) & (df.index <= end)]
        if df.empty:
            return {}
        s = float(df["Close"].iloc[0])
        e = float(df["Close"].iloc[-1])
        years = max((df.index[-1] - df.index[0]).days / 365.25, 1e-6)
        cagr = (e / s) ** (1.0 / years) - 1.0
        eq = df["Close"] / s * init_cap
        daily = eq.pct_change().dropna()
        import numpy as np
        sharpe = float((daily.mean() / daily.std()) * np.sqrt(252)) if daily.std() > 0 else 0.0
        dd = (eq / eq.cummax() - 1.0).min()
        return {"cagr_pct": cagr, "sharpe": sharpe, "max_dd_pct": float(dd)}
    except Exception as e:
        print(f"NIFTY benchmark failed: {e}")
        return {}


def run_one_tier(tier_name: str, symbols: list[str], history: dict) -> pd.DataFrame:
    """Run all backtestable strategies on one cap tier."""
    sub_history = {s: history[s] for s in symbols if s in history}
    if len(sub_history) < 20:
        print(f"  {tier_name}: only {len(sub_history)} stocks loaded — skipping")
        return pd.DataFrame()

    print(f"\n=== {tier_name} ({len(sub_history)} stocks) ===")
    cfg = StrategyBacktestConfig(
        start_date="2023-01-01",
        end_date="2026-04-30",
        initial_capital=100_000.0,
        n_picks=2,
        rebalance_freq="BMS",
        time_stop_bars=60,
        max_per_sector=1,
    )
    rows = []
    for name, strat in BACKTESTABLE.items():
        result = run_strategy_backtest(strat, sub_history, get_sector, cfg=cfg)
        s = result.get("stats", {})
        if not s:
            continue
        rows.append({
            "tier": tier_name,
            "strategy": name,
            "CAGR%":  s.get("cagr_pct", 0) * 100,
            "Sharpe": s.get("sharpe", 0),
            "MaxDD%": s.get("max_dd_pct", 0) * 100,
            "Calmar": s.get("calmar", 0),
            "WinRt%": s.get("win_rate_pct", 0) * 100,
            "PF":     s.get("profit_factor", 0),
            "N":      s.get("n_trades", 0),
            "Final":  s.get("final_equity", 0),
        })
        print(f"  {name:>22s}: CAGR={s.get('cagr_pct',0)*100:>6.2f}%  "
              f"Sharpe={s.get('sharpe',0):>5.2f}  "
              f"DD={s.get('max_dd_pct',0)*100:>6.2f}%  "
              f"PF={s.get('profit_factor',0):>5.2f}  "
              f"N={s.get('n_trades',0):>3d}")
    return pd.DataFrame(rows)


def main():
    print("="*70)
    print(" 10-STRATEGY COMPARISON: LARGE vs MID vs SMALL vs ALL cap")
    print("="*70)
    print(f"\nBacktestable strategies: {len(BACKTESTABLE)}")
    print(f"Fundamental strategies (live-only): {len(FUNDAMENTAL)}")
    print(f"  Fundamentals NOT backtested (yfinance .info = snapshot bias)")
    print(f"  These will be run live for current signals only")
    print()

    print("Loading 4-yr OHLCV history for entire universe...")
    history = load_universe(ALL_STOCKS, period="4y", use_cache=True, progress=False)
    print(f"  Loaded valid history for {len(history)}/{len(ALL_STOCKS)} stocks\n")

    all_rows = []
    for tier_name, syms in [
        ("LARGE", LARGE_CAP),
        ("MID",   MID_CAP),
        ("SMALL", SMALL_CAP),
        ("ALL",   ALL_STOCKS),
    ]:
        df = run_one_tier(tier_name, syms, history)
        if not df.empty:
            all_rows.append(df)

    if not all_rows:
        print("No results — aborting")
        return

    combined = pd.concat(all_rows, ignore_index=True)

    # Benchmark NIFTY
    print("\n=== BENCHMARK ===")
    nifty = benchmark_nifty("2023-01-01", "2026-04-30")
    if nifty:
        print(f"  {'NIFTY 50 B&H':>22s}: CAGR={nifty['cagr_pct']*100:>6.2f}%  "
              f"Sharpe={nifty['sharpe']:>5.2f}  DD={nifty['max_dd_pct']*100:>6.2f}%")

    # Save
    out_dir = Path(__file__).resolve().parent.parent / "reports" / "screener"
    out_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_dir / "strategy_comparison.csv", index=False)
    print(f"\nSaved: {out_dir/'strategy_comparison.csv'}")

    # Find best 3 per tier by Sharpe (more honest than CAGR alone)
    print("\n" + "="*70)
    print(" TOP 3 STRATEGIES PER CAP TIER (by Sharpe)")
    print("="*70)
    best_picks = {}
    for tier in ["LARGE", "MID", "SMALL", "ALL"]:
        sub = combined[combined["tier"] == tier]
        if sub.empty:
            continue
        top3 = sub.sort_values("Sharpe", ascending=False).head(3)
        print(f"\n{tier}:")
        for _, r in top3.iterrows():
            print(f"  {r['strategy']:>22s}: CAGR={r['CAGR%']:>6.2f}%  "
                  f"Sharpe={r['Sharpe']:>5.2f}  DD={r['MaxDD%']:>6.2f}%  "
                  f"PF={r['PF']:>5.2f}")
        best_picks[tier] = top3["strategy"].tolist()

    with open(out_dir / "best_strategies.json", "w") as f:
        json.dump(best_picks, f, indent=2)

    print(f"\nSaved: {out_dir/'best_strategies.json'}")
    print("\n=== FULL MATRIX (sorted by Sharpe) ===")
    print(combined.sort_values(["tier", "Sharpe"], ascending=[True, False]).to_string(index=False))


if __name__ == "__main__":
    main()
