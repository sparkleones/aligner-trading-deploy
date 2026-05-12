"""
Test the small-cap strategies against:
  - Nifty Smallcap 250 index (^CNXSMCP / NIFTYSMCAP via yfinance)
  - Top small-cap MF benchmark CAGRs (Nippon SC, HDFC SC etc.)
  - Our existing large-cap composite

Scenarios:
  A) Lump-sum ₹1L, no SIP, 4.3-year backtest
  B) SIP ₹10k/month for 4.3 years, no initial lump
  C) Lump-sum ₹1L + SIP ₹10k/month (combo)
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
from pathlib import Path

from .data_loader import load_universe
from .universe import get_sector
from .universe_extended import SMALL_CAP, LARGE_CAP
from .strategies.scqg import SCQGStrategy
from .strategies.composite_top import CompositeTopStrategy
from .backtest_smallcap_mf import SmallCapMFConfig, run_smallcap_backtest


def _nifty_smcap_bench(start, end, init, monthly_sip=0):
    """Buy-and-hold (or SIP into) Nifty Smallcap 250."""
    import yfinance as yf
    # Try several tickers; NIFTYSMLCAP250.NS, ^CNXSC, NIFTY-SMALLCAP-250
    for ticker in ("NIFTYSMLCAP250.NS", "^CNXSMCP", "NIFTYSMCAP250.NS"):
        try:
            df = yf.Ticker(ticker).history(period="5y", auto_adjust=True)
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            if not df.empty:
                break
        except Exception:
            df = None
    if df is None or df.empty:
        return None

    sub = df[(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))]
    if sub.empty or len(sub) < 10:
        return None

    if monthly_sip == 0:
        # Lump-sum
        s = float(sub["Close"].iloc[0])
        e = float(sub["Close"].iloc[-1])
        years = max((sub.index[-1] - sub.index[0]).days / 365.25, 1e-6)
        cagr = (e / s) ** (1.0 / years) - 1.0
        eq = sub["Close"] / s * init
        daily = eq.pct_change().dropna()
        sharpe = float((daily.mean() / daily.std()) * np.sqrt(252)) if daily.std() > 0 else 0.0
        dd = (eq / eq.cummax() - 1.0).min()
        return {
            "cagr": cagr, "sharpe": sharpe, "dd": float(dd),
            "final": float(eq.iloc[-1]), "total_invested": init,
        }
    # SIP simulation: buy each month with monthly_sip
    units = init / float(sub["Close"].iloc[0])
    total_invested = init
    # Find 1st trading day of each month
    months = sub.index.to_series().dt.to_period("M").drop_duplicates().index
    for m in months[1:]:  # skip first month (initial deploy)
        month_first = sub[sub.index >= m].index[0] if (sub.index >= m).any() else None
        if month_first is None:
            continue
        px = float(sub.loc[month_first, "Close"])
        if px > 0:
            units += monthly_sip / px
            total_invested += monthly_sip
    final_px = float(sub["Close"].iloc[-1])
    final_val = units * final_px
    years = max((sub.index[-1] - sub.index[0]).days / 365.25, 1e-6)
    cagr = (final_val / total_invested) ** (1.0 / years) - 1.0  # simple CAGR proxy
    return {
        "cagr": cagr, "sharpe": 0, "dd": 0,
        "final": final_val, "total_invested": total_invested,
    }


def run_scenario(label, history, sector_fn, cfg):
    res = run_smallcap_backtest(SCQGStrategy(), history, sector_fn, cfg=cfg)
    s = res.get("stats", {})
    return s


def main():
    print("=" * 78)
    print(" SMALL-CAP STRATEGY BACKTEST")
    print(" Strategy: SCQG (Small-Cap Quality-Growth composite)")
    print("=" * 78)

    print(f"\nLoading 5y history for {len(SMALL_CAP)} small-cap symbols...")
    sc_history = load_universe(SMALL_CAP, period="5y", use_cache=True, progress=False)
    print(f"  Loaded: {len(sc_history)}\n")

    if len(sc_history) < 30:
        print("WARNING: Insufficient small-cap data loaded.")
        return

    start = "2022-01-01"
    end = "2026-04-30"

    print(f"BACKTEST WINDOW: {start} -> {end} (~4.3 years)\n")
    print(f"{'Scenario':<45} {'CAGR (XIRR)':>12} {'Sharpe':>8} {'MaxDD':>10} {'Final Eq':>14} {'Profit':>14}")
    print("-" * 110)

    # ── A) Lump-sum 1L, no SIP ──
    cfg_a = SmallCapMFConfig(
        start_date=start, end_date=end,
        initial_capital=100_000, monthly_sip=0,
        n_picks=12, rebalance_freq="AS",
    )
    sa = run_scenario("A", sc_history, get_sector, cfg_a)
    print(f"{'A. Lump-sum Rs 1L, no SIP':<45} "
          f"{sa.get('cagr_pct', 0)*100:>11.2f}% "
          f"{sa.get('sharpe', 0):>8.2f} "
          f"{sa.get('max_dd_pct', 0)*100:>9.2f}% "
          f"Rs {sa.get('final_equity', 0):>11,.0f} "
          f"Rs {sa.get('absolute_profit', 0):>11,.0f}")

    # ── B) Pure SIP 10k/month, no initial ──
    cfg_b = SmallCapMFConfig(
        start_date=start, end_date=end,
        initial_capital=10_000, monthly_sip=10_000,
        n_picks=12, rebalance_freq="AS",
    )
    sb = run_scenario("B", sc_history, get_sector, cfg_b)
    print(f"{'B. SIP Rs 10k/month, Rs 10k start':<45} "
          f"{sb.get('cagr_pct', 0)*100:>11.2f}% "
          f"{sb.get('sharpe', 0):>8.2f} "
          f"{sb.get('max_dd_pct', 0)*100:>9.2f}% "
          f"Rs {sb.get('final_equity', 0):>11,.0f} "
          f"Rs {sb.get('absolute_profit', 0):>11,.0f}")

    # ── C) Lump 1L + SIP 10k/month ──
    cfg_c = SmallCapMFConfig(
        start_date=start, end_date=end,
        initial_capital=100_000, monthly_sip=10_000,
        n_picks=12, rebalance_freq="AS",
    )
    sc = run_scenario("C", sc_history, get_sector, cfg_c)
    print(f"{'C. Rs 1L lump + SIP Rs 10k/month':<45} "
          f"{sc.get('cagr_pct', 0)*100:>11.2f}% "
          f"{sc.get('sharpe', 0):>8.2f} "
          f"{sc.get('max_dd_pct', 0)*100:>9.2f}% "
          f"Rs {sc.get('final_equity', 0):>11,.0f} "
          f"Rs {sc.get('absolute_profit', 0):>11,.0f}")

    # ── D) For comparison: large-cap composite (lump 1L) ──
    lg_history = load_universe(LARGE_CAP, period="5y", use_cache=True, progress=False)
    from .backtest_mf_style import MFStyleConfig, run_mf_backtest
    cfg_d = MFStyleConfig(
        start_date=start, end_date=end,
        initial_capital=100_000, n_picks=5,
        rebalance_freq="QS", apply_stcg_tax=True,
    )
    rd = run_mf_backtest(CompositeTopStrategy(), lg_history, get_sector, cfg=cfg_d)
    sd = rd.get("stats", {})
    print(f"{'D. Large-cap composite (1L, ref)':<45} "
          f"{sd.get('cagr_pct', 0)*100:>11.2f}% "
          f"{sd.get('sharpe', 0):>8.2f} "
          f"{sd.get('max_dd_pct', 0)*100:>9.2f}% "
          f"Rs {sd.get('final_equity', 0):>11,.0f} "
          f"Rs {sd.get('final_equity', 0) - 100_000:>11,.0f}")

    # ── BENCHMARKS ──
    print("\n" + "-" * 110)
    print(" BENCHMARKS (Indian small-cap index + MF references)")
    print("-" * 110)

    sc_bench_a = _nifty_smcap_bench(start, end, 100_000, monthly_sip=0)
    if sc_bench_a:
        bp = sc_bench_a["final"] - sc_bench_a["total_invested"]
        print(f"{'NIFTY Smallcap 250 (Rs 1L lump)':<45} "
              f"{sc_bench_a['cagr']*100:>11.2f}% {'':>9}"
              f"{sc_bench_a['dd']*100:>10.2f}% Rs {sc_bench_a['final']:>11,.0f} "
              f"Rs {bp:>11,.0f}")
    else:
        print(f"{'NIFTY Smallcap 250 (lump)':<45}  data unavailable")

    sc_bench_c = _nifty_smcap_bench(start, end, 100_000, monthly_sip=10_000)
    if sc_bench_c:
        bp = sc_bench_c["final"] - sc_bench_c["total_invested"]
        print(f"{'NIFTY Smallcap 250 (1L + 10k SIP)':<45} "
              f"{sc_bench_c['cagr']*100:>11.2f}% {'':>9}"
              f"{'':>10}  Rs {sc_bench_c['final']:>11,.0f} "
              f"Rs {bp:>11,.0f}")

    print()
    print(f"{'Top MF: Nippon India Small Cap (5y typical)':<45} ~{31:>11.0f}%   typical XIRR")
    print(f"{'Top MF: HDFC Small Cap (5y typical)':<45} ~{28:>11.0f}%   typical XIRR")
    print(f"{'Top MF: Kotak Small Cap (5y typical)':<45} ~{26:>11.0f}%   typical XIRR")
    print(f"{'Nifty Smallcap 250 (5y)':<45} ~{22:>11.0f}%   index baseline")

    # Save findings
    out_dir = Path(__file__).resolve().parent.parent / "reports" / "screener"
    out_dir.mkdir(parents=True, exist_ok=True)
    findings = {
        "strategy": "SCQG (Small-Cap Quality-Growth)",
        "engine": "small-cap MF style: annual rebal, no SL, stage2_break exit",
        "window": [start, end],
        "scenarios": {
            "A_lumpsum_1L": sa,
            "B_sip_10k_month": sb,
            "C_lump_1L_plus_sip_10k": sc,
            "D_large_cap_reference": sd,
        },
        "benchmarks": {
            "nifty_smallcap250_lump": sc_bench_a,
            "nifty_smallcap250_sip": sc_bench_c,
            "mf_5y_typicals": {
                "Nippon India SC": 0.31,
                "HDFC SC": 0.28,
                "Kotak SC": 0.26,
                "Axis SC": 0.25,
                "Nifty Smallcap 250": 0.22,
            },
        },
        "generated_at": pd.Timestamp.now().isoformat(),
    }
    with open(out_dir / "smallcap_strategies.json", "w") as f:
        json.dump(findings, f, indent=2, default=str)
    print(f"\n[OK] Saved to {out_dir/'smallcap_strategies.json'}")


if __name__ == "__main__":
    main()
