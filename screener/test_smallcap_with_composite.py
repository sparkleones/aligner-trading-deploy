"""
Cross-test: run the PROVEN composite_top (stage2 + breakout) strategy
on small-cap universe. Also fixed NIFTY Smallcap 250 benchmark fetch.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data_loader import load_universe
from .universe import get_sector
from .universe_extended import SMALL_CAP
from .strategies.composite_top import CompositeTopStrategy
from .backtest_smallcap_mf import SmallCapMFConfig, run_smallcap_backtest


def _smcap_bench(start, end, init):
    """Try multiple tickers for Nifty Smallcap 250."""
    import yfinance as yf
    for ticker in ("NIFTYSMLCAP250.NS", "^CNXSMCP", "NIFTYSMLCAP250.BO",
                   "NIFTY_SMLCAP_100.NS", "0P0001HC1J.BO"):
        try:
            df = yf.Ticker(ticker).history(period="5y", auto_adjust=True)
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            if not df.empty and len(df) > 200:
                df = df[(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))]
                if not df.empty:
                    s = float(df["Close"].iloc[0])
                    e = float(df["Close"].iloc[-1])
                    years = max((df.index[-1] - df.index[0]).days / 365.25, 1e-6)
                    cagr = (e / s) ** (1.0 / years) - 1.0
                    eq = df["Close"] / s * init
                    dd = (eq / eq.cummax() - 1.0).min()
                    return {"ticker": ticker, "cagr": cagr, "dd": float(dd),
                             "final": float(eq.iloc[-1])}
        except Exception:
            pass
    return None


def main():
    print("=" * 78)
    print(" CROSS-TEST: composite_top strategy on SMALL-CAP universe")
    print("=" * 78)

    print(f"\nLoading 5y history...")
    history = load_universe(SMALL_CAP, period="5y", use_cache=True, progress=False)
    print(f"  Loaded: {len(history)} small-caps\n")

    scenarios = [
        ("Composite-top, 5 picks, quarterly rebal",
         dict(n_picks=5, rebalance_freq="QS", monthly_sip=0)),
        ("Composite-top, 10 picks, quarterly rebal",
         dict(n_picks=10, rebalance_freq="QS", monthly_sip=0)),
        ("Composite-top, 10 picks, annual rebal",
         dict(n_picks=10, rebalance_freq="YS", monthly_sip=0)),
        ("Composite-top, 10 picks, quarterly + SIP 10k/m",
         dict(n_picks=10, rebalance_freq="QS", monthly_sip=10_000)),
    ]

    print(f"{'Scenario':<55} {'CAGR':>10} {'Sharpe':>8} {'MaxDD':>10} {'Profit Rs':>14}")
    print("-" * 105)

    for label, override in scenarios:
        cfg = SmallCapMFConfig(
            start_date="2022-01-01", end_date="2026-04-30",
            initial_capital=100_000, max_per_sector=3,
            **override,
        )
        res = run_smallcap_backtest(CompositeTopStrategy(), history, get_sector, cfg=cfg)
        s = res.get("stats", {})
        cagr = s.get("cagr_pct", 0)
        sharpe = s.get("sharpe", 0)
        dd = s.get("max_dd_pct", 0)
        profit = s.get("absolute_profit", s.get("final_equity", 0) - s.get("total_invested", 100_000))
        print(f"  {label:<53} {cagr*100:>9.2f}% {sharpe:>8.2f} {dd*100:>9.2f}% "
              f"Rs {profit:>11,.0f}")

    print("\n" + "-" * 105)
    bench = _smcap_bench("2022-01-01", "2026-04-30", 100_000)
    if bench:
        bench_profit = bench["final"] - 100_000
        print(f"  {'NIFTY Smallcap 250 ' + bench['ticker'] + ' (lump 1L)':<53} "
              f"{bench['cagr']*100:>9.2f}% {'':>9}{bench['dd']*100:>9.2f}% "
              f"Rs {bench_profit:>11,.0f}")
    else:
        print("  NIFTY Smallcap 250 benchmark fetch failed for all tickers.")
        print("  Reference values (AMFI/Value Research, 5y CAGR end-2025):")
        print("    NIFTY Smallcap 250:           ~22%")
        print("    Top quartile SC MF:           ~26-31%")


if __name__ == "__main__":
    main()
