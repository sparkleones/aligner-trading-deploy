"""
Clean robustness sweep with PROPER lookback variation.

The earlier directional_gate_robustness.py had a bug: passed custom
trend_lookup via cfg, but run_backtest overwrote it with the hardcoded
5-day version. So all "lookback" rows were actually 5-day.

Fixed in v14_unified_backtest.py: _build_trend_lookup_5d now accepts
lookback_days, controlled by cfg["directional_gate_lookback_days"]
(default 5).

This sweep tests whether the 5-day choice is empirically optimal or if
a different lookback would beat it.

Grid: 6 thresholds (0.5 to 2.0) × 4 lookbacks (3, 5, 7, 10) = 24 cells.

Usage:
    python -m backtesting.directional_gate_lookback_sweep
"""
import sys
import datetime as dt
from pathlib import Path
from collections import defaultdict

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backtesting.v14_unified_backtest import load_period_data, run_backtest
from scoring.config import V17_CONFIG

CAPITAL = 2_00_000

THRESHOLDS = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
LOOKBACKS = [3, 5, 7, 10]


def to_date(d):
    return dt.date.fromisoformat(d) if isinstance(d, str) else d


def base_cfg():
    cfg = dict(V17_CONFIG)
    cfg["avoid_days"] = [0, 2]
    cfg["use_v17_regime_gate"] = False
    cfg["use_v17_monwed_gate"] = False
    cfg["vix_floor"] = 12
    cfg["vix_ceil"] = 25
    # Explicitly disable directional gate (inherited from V15_CONFIG since Push #2)
    cfg["directional_gate_threshold"] = None
    return cfg


def gate_cfg(threshold, lookback):
    cfg = base_cfg()
    cfg["directional_gate_threshold"] = threshold
    cfg["directional_gate_lookback_days"] = lookback
    return cfg


def metrics(trades):
    if not trades:
        return {"n": 0, "pnl": 0.0, "wr": 0.0, "pf": 0.0}
    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    tw = sum(t["pnl"] for t in wins)
    tl = -sum(t["pnl"] for t in losses)
    pf = tw / tl if tl > 0 else float("inf")
    return {"n": len(trades), "pnl": sum(t["pnl"] for t in trades),
            "wr": len(wins) / len(trades) * 100, "pf": pf}


def equity_dd(trades, capital):
    daily = defaultdict(float)
    for t in trades:
        daily[to_date(t["date"])] += t["pnl"]
    eq = capital
    peak = capital
    max_dd = 0.0
    for d in sorted(daily.keys()):
        eq += daily[d]
        peak = max(peak, eq)
        dd = peak - eq
        if dd > max_dd:
            max_dd = dd
    return max_dd


def fmt_pf(pf):
    return " inf" if pf == float("inf") else f"{pf:5.2f}"


def main():
    full_start = dt.date(2024, 7, 1)
    full_end = dt.date(2026, 4, 6)

    print("=" * 110)
    print("  DIRECTIONAL GATE LOOKBACK SWEEP  (proper varying lookback)")
    print("=" * 110)
    print(f"\nLoading {full_start} -> {full_end} ...", flush=True)
    preloaded = load_period_data(start_date=full_start, end_date=full_end, quiet=True)
    print(f"  {len(preloaded[2])} trading days\n", flush=True)

    # Baseline
    print("Running BASELINE (no gate)...", flush=True)
    base_trades, _ = run_backtest(start_date=full_start, end_date=full_end,
                                  cfg_override=base_cfg(), quiet=True, preloaded=preloaded)
    bm = metrics(base_trades)
    print(f"  BASELINE: n={bm['n']}  PnL=Rs {bm['pnl']:+,.0f}  PF={fmt_pf(bm['pf'])}\n", flush=True)

    # Grid sweep
    print("PHASE 1: 21mo grid sweep", flush=True)
    print("-" * 110, flush=True)
    print(f"  {'lookback':<9} {'thr':<6} {'n':>4} {'PnL':>14} {'PF':>5} {'WR%':>6} "
          f"{'DD':>11} {'dPnL':>13} {'dPF':>6}", flush=True)
    print("  " + "-" * 92, flush=True)

    grid = []
    for lb in LOOKBACKS:
        for thr in THRESHOLDS:
            cfg = gate_cfg(thr, lb)
            trades, _ = run_backtest(start_date=full_start, end_date=full_end,
                                     cfg_override=cfg, quiet=True, preloaded=preloaded)
            m = metrics(trades)
            dd = equity_dd(trades, CAPITAL)
            d_pnl = m["pnl"] - bm["pnl"]
            m_pf = m["pf"] if m["pf"] != float("inf") else 999
            b_pf = bm["pf"] if bm["pf"] != float("inf") else 999
            d_pf = m_pf - b_pf
            grid.append((lb, thr, m, dd, d_pnl, d_pf))
            print(f"  {lb:<9d} {thr:<6.2f} {m['n']:>4d} Rs {m['pnl']:>+12,.0f} "
                  f"{fmt_pf(m['pf'])} {m['wr']:>5.1f}% Rs {dd:>+9,.0f} "
                  f"Rs {d_pnl:>+10,.0f} {d_pf:>+6.2f}", flush=True)

    # Find absolute best
    best = max(grid, key=lambda g: g[2]["pnl"])
    lb_best, thr_best, m_best, dd_best, d_pnl_best, d_pf_best = best

    # Pivot table
    print()
    print("=" * 110)
    print("  PIVOT: rows=lookback, cols=threshold, value=PnL_delta_vs_baseline (Rs L)")
    print("=" * 110)
    header = "  lookback | " + " ".join(f"{t:>9.2f}" for t in THRESHOLDS)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for lb in LOOKBACKS:
        row = f"  {lb:<8d} | "
        for thr in THRESHOLDS:
            entry = next((g for g in grid if g[0] == lb and g[1] == thr), None)
            if entry:
                row += f"{entry[4]/100000:>9.2f} "
            else:
                row += "       -- "
        print(row)

    # Mark best in summary
    print()
    print("=" * 110)
    print(f"  BEST: lookback={lb_best}, threshold={thr_best}: "
          f"PnL=Rs {m_best['pnl']:+,.0f} (Rs {d_pnl_best:+,.0f} vs base), "
          f"PF={fmt_pf(m_best['pf'])}, DD=Rs {dd_best:+,.0f}")
    print(f"  Currently deployed: lookback=5, threshold=1.0")
    deployed = next((g for g in grid if g[0] == 5 and g[1] == 1.0), None)
    if deployed:
        _, _, m_dep, dd_dep, dpnl_dep, dpf_dep = deployed
        print(f"  Deployed result:    PnL=Rs {m_dep['pnl']:+,.0f} (Rs {dpnl_dep:+,.0f} vs base), "
              f"PF={fmt_pf(m_dep['pf'])}, DD=Rs {dd_dep:+,.0f}")
        delta_vs_deployed = m_best['pnl'] - m_dep['pnl']
        print(f"  Best gives Rs {delta_vs_deployed:+,.0f} more than deployed "
              f"({delta_vs_deployed/m_dep['pnl']*100 if m_dep['pnl'] else 0:+.1f}%)")


if __name__ == "__main__":
    main()
