"""
Test QGV-MF engine on SC, MC, and SC+MC universes vs MF benchmarks.

Goal: beat top-quartile SC MF (~28% CAGR) or get close.
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
from pathlib import Path

from .data_loader import load_universe
from .universe import get_sector
from .universe_extended import SMALL_CAP, MID_CAP, LARGE_CAP
from .strategies.qgv import QGVStrategy
from .backtest_qgv_mf import QGVMFConfig, run_qgv_backtest


def run_scenario(label, history, sector_fn, **cfg_overrides):
    cfg = QGVMFConfig(
        start_date="2022-01-01", end_date="2026-04-30",
        initial_capital=100_000, **cfg_overrides,
    )
    res = run_qgv_backtest(QGVStrategy(), history, sector_fn, cfg=cfg)
    return res.get("stats", {})


def _print_row(label, s, total_invested=None):
    cagr = s.get("cagr_pct", 0)
    sharpe = s.get("sharpe", 0)
    dd = s.get("max_dd_pct", 0)
    final = s.get("final_equity", 0)
    profit = s.get("absolute_profit", final - (total_invested or 100_000))
    wr = s.get("win_rate_pct", 0)
    n = s.get("n_trades", 0)
    print(f"  {label:<46} {cagr*100:>8.2f}% {sharpe:>7.2f} {dd*100:>9.2f}% "
          f"Rs {final:>11,.0f} Rs {profit:>11,.0f}  WR {wr*100:>4.0f}%  N {n:>3d}")


def main():
    print("=" * 110)
    print(" QGV-MF Strategy Test — Quality + Growth + Value + Conviction-weighted + Dip-buy")
    print("=" * 110)

    print("\nLoading data...")
    sc = load_universe(SMALL_CAP, period="5y", use_cache=True, progress=False)
    mc = load_universe(MID_CAP, period="5y", use_cache=True, progress=False)
    lg = load_universe(LARGE_CAP, period="5y", use_cache=True, progress=False)
    print(f"  SC: {len(sc)}  MC: {len(mc)}  LC: {len(lg)}\n")

    print(f"  {'Scenario':<46} {'CAGR':>9} {'Sharpe':>7} {'MaxDD':>10} "
          f"{'Final':>14} {'Profit':>14} {'Win':>10} {'N':>5}")
    print("-" * 130)

    # ── SMALL CAP scenarios ──
    print("\nSMALL-CAP universe (the hard target):")
    s_sc_lump = run_scenario("SC lump", sc, get_sector, n_picks=12, monthly_sip=0)
    _print_row("SC lump Rs 1L, 12 picks, annual rebal", s_sc_lump)
    s_sc_sip = run_scenario("SC SIP", sc, get_sector, n_picks=12, monthly_sip=10_000)
    _print_row("SC + SIP Rs 10k/mo, 12 picks", s_sc_sip, total_invested=s_sc_sip.get("total_invested"))

    # ── MID CAP ──
    print("\nMID-CAP universe:")
    s_mc_lump = run_scenario("MC lump", mc, get_sector, n_picks=12, monthly_sip=0)
    _print_row("MC lump Rs 1L, 12 picks, annual rebal", s_mc_lump)
    s_mc_sip = run_scenario("MC SIP", mc, get_sector, n_picks=12, monthly_sip=10_000)
    _print_row("MC + SIP Rs 10k/mo, 12 picks", s_mc_sip, total_invested=s_mc_sip.get("total_invested"))

    # ── COMBINED SC+MC ──
    print("\nSMALL+MID combined (140 stocks, the way real MFs invest):")
    combined = {**sc, **mc}
    s_comb_lump = run_scenario("Combined lump", combined, get_sector, n_picks=12, monthly_sip=0)
    _print_row("SC+MC lump Rs 1L, 12 picks", s_comb_lump)
    s_comb_sip = run_scenario("Combined SIP", combined, get_sector, n_picks=12, monthly_sip=10_000)
    _print_row("SC+MC + SIP Rs 10k/mo, 12 picks", s_comb_sip, total_invested=s_comb_sip.get("total_invested"))
    s_comb_big_sip = run_scenario("Combined big SIP", combined, get_sector, n_picks=15, monthly_sip=20_000)
    _print_row("SC+MC + SIP Rs 20k/mo, 15 picks", s_comb_big_sip, total_invested=s_comb_big_sip.get("total_invested"))

    # ── LARGE CAP for reference ──
    print("\nLARGE-CAP reference (where we know we work):")
    s_lc_lump = run_scenario("LC lump", lg, get_sector, n_picks=12, monthly_sip=0)
    _print_row("LC lump Rs 1L, 12 picks", s_lc_lump)

    # ── Benchmarks ──
    print("\n" + "-" * 130)
    print(" MUTUAL FUND BENCHMARKS (5-yr typical CAGR):")
    print("-" * 130)
    print(f"  {'Nippon India Small Cap (Direct Growth)':<46} ~31% CAGR    typical 5y XIRR")
    print(f"  {'HDFC Small Cap (Direct Growth)':<46} ~28% CAGR    typical 5y XIRR")
    print(f"  {'Kotak Small Cap (Direct Growth)':<46} ~26% CAGR    typical 5y XIRR")
    print(f"  {'Motilal Oswal Midcap (Direct Growth)':<46} ~28% CAGR    typical 5y XIRR")
    print(f"  {'NIFTY Smallcap 250 Index Fund':<46} ~20% CAGR    passive baseline")
    print(f"  {'NIFTY Midcap 150 Index Fund':<46} ~22% CAGR    passive baseline")

    # Save findings
    out_dir = Path(__file__).resolve().parent.parent / "reports" / "screener"
    out_dir.mkdir(parents=True, exist_ok=True)
    findings = {
        "strategy": "QGV-MF (Quality+Growth+Value, conviction-weighted, dip-buy)",
        "window": ["2022-01-01", "2026-04-30"],
        "scenarios": {
            "sc_lump": s_sc_lump, "sc_sip": s_sc_sip,
            "mc_lump": s_mc_lump, "mc_sip": s_mc_sip,
            "combined_lump": s_comb_lump, "combined_sip": s_comb_sip,
            "combined_big_sip": s_comb_big_sip,
            "lc_lump_ref": s_lc_lump,
        },
        "benchmarks_5y_typical": {
            "Nippon India SC": 0.31, "HDFC SC": 0.28, "Kotak SC": 0.26,
            "Motilal MidCap": 0.28,
            "Nifty Smallcap 250": 0.20, "Nifty Midcap 150": 0.22,
        },
        "generated_at": pd.Timestamp.now().isoformat(),
    }
    with open(out_dir / "qgv_findings.json", "w") as f:
        json.dump(findings, f, indent=2, default=str)
    print(f"\n[OK] Saved to {out_dir/'qgv_findings.json'}")


if __name__ == "__main__":
    main()
