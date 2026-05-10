"""
Sweep across rebalance frequencies + stop/target rules to find better params.
Also benchmark against NIFTY 50 buy-and-hold.
"""
from __future__ import annotations

import pandas as pd

from .universe import get_universe
from .data_loader import load_universe, load_history
from .backtest import BacktestConfig, run_backtest


def benchmark_nifty(start: str, end: str, init_cap: float = 100_000.0) -> dict:
    """Buy-and-hold NIFTY 50 index for comparison."""
    df = load_history("NIFTY", period="5y", use_cache=True)
    if df.empty:
        # Try yahoo's ^NSEI symbol manually
        import yfinance as yf
        df = yf.Ticker("^NSEI").history(period="5y", auto_adjust=True)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
    df = df[(df.index >= start) & (df.index <= end)]
    if df.empty:
        return {"error": "no NIFTY data"}
    start_px = float(df["Close"].iloc[0])
    end_px = float(df["Close"].iloc[-1])
    ret = (end_px / start_px) - 1.0
    n_days = (df.index[-1] - df.index[0]).days
    years = max(n_days / 365.25, 1e-6)
    cagr = (end_px / start_px) ** (1.0 / years) - 1.0
    eq = df["Close"] / start_px * init_cap
    daily = eq.pct_change().dropna()
    sharpe = float((daily.mean() / daily.std()) * (252 ** 0.5)) if daily.std() > 0 else 0.0
    dd = (eq / eq.cummax() - 1.0).min()
    return {
        "name": "NIFTY 50 B&H",
        "total_return_pct": ret,
        "cagr_pct": cagr,
        "sharpe": sharpe,
        "max_drawdown_pct": float(dd),
        "years": round(years, 2),
    }


def sweep():
    universe = get_universe()
    print(f"Universe: {len(universe)} symbols")
    history = load_universe(universe, period="4y", use_cache=True, progress=False)
    print(f"Loaded: {len(history)} valid\n")

    configs = [
        # (label, rebalance, n_picks, min_stop, max_stop, RR, time_stop, min_liq)
        ("WEEKLY  2pk std",  "W-MON",  2, 0.04, 0.08, 3.0, 20, 9.0),
        ("MONTHLY 2pk std",  "BMS",    2, 0.04, 0.08, 3.0, 30, 9.0),  # BMS = business month start
        ("MONTHLY 2pk wide", "BMS",    2, 0.06, 0.12, 2.5, 40, 9.0),
        ("MONTHLY 3pk std",  "BMS",    3, 0.04, 0.08, 3.0, 30, 9.0),
        ("FORTNIGHTLY 2pk",  "2W-MON", 2, 0.05, 0.10, 3.0, 25, 9.0),
        ("MONTHLY 2pk loose","BMS",    2, 0.05, 0.10, 4.0, 40, 9.0),
    ]

    rows = []
    for label, freq, n_picks, min_sl, max_sl, rr, ts, min_liq in configs:
        cfg = BacktestConfig(
            start_date="2023-01-01",
            end_date="2026-04-30",
            initial_capital=100_000.0,
            n_picks=n_picks,
            rebalance_freq=freq,
            min_stop_pct=min_sl,
            max_stop_pct=max_sl,
            reward_risk_ratio=rr,
            time_stop_bars=ts,
            min_liquidity_log10_inr=min_liq,
        )
        r = run_backtest(history, cfg=cfg)
        s = r["stats"]
        rows.append({
            "config": label,
            "CAGR%": f"{s['cagr_pct']*100:.2f}",
            "Sharpe": f"{s['sharpe']:.2f}",
            "MaxDD%": f"{s['max_drawdown_pct']*100:.2f}",
            "Calmar": f"{s['calmar']:.2f}",
            "WinRt%": f"{s['win_rate_pct']*100:.1f}",
            "PF": f"{s['profit_factor']:.2f}",
            "N":  s["n_trades"],
        })
        print(f"  {label:>22s}: CAGR={s['cagr_pct']*100:>5.2f}%  "
              f"Sharpe={s['sharpe']:.2f}  DD={s['max_drawdown_pct']*100:>6.2f}%  "
              f"PF={s['profit_factor']:.2f}  N={s['n_trades']}")

    # Benchmark
    print()
    nifty = benchmark_nifty("2023-01-01", "2026-04-30")
    if "cagr_pct" in nifty:
        print(f"  {'NIFTY 50 B&H':>22s}: CAGR={nifty['cagr_pct']*100:>5.2f}%  "
              f"Sharpe={nifty['sharpe']:.2f}  DD={nifty['max_drawdown_pct']*100:>6.2f}%")
        rows.append({
            "config": "NIFTY 50 B&H",
            "CAGR%":  f"{nifty['cagr_pct']*100:.2f}",
            "Sharpe": f"{nifty['sharpe']:.2f}",
            "MaxDD%": f"{nifty['max_drawdown_pct']*100:.2f}",
            "Calmar": "-",
            "WinRt%": "-",
            "PF":     "-",
            "N":      "-",
        })

    df = pd.DataFrame(rows)
    print("\n" + df.to_string(index=False))


if __name__ == "__main__":
    sweep()
