"""
vix_ceil sweep on V17_PROD_ONLY (Option A: avoid=[0,2], vix_floor=12).

Tests whether tightening the upper VIX gate helps. Antigravity uses
vix_ceil=22 ("premiums too expensive"); we currently use 35.

Usage:
    python -m backtesting.vix_ceil_sweep
"""
import sys
import datetime as dt
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backtesting.v14_unified_backtest import load_period_data, run_backtest
from scoring.config import V17_CONFIG

CAPITAL = 2_00_000
POST_SEP = dt.date(2025, 9, 1)

CEILS = [20, 22, 25, 28, 30, 35]


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


def split(trades, cutover):
    pre, post = [], []
    for t in trades:
        d = t.get("date")
        if isinstance(d, str):
            d = dt.date.fromisoformat(d)
        (pre if d < cutover else post).append(t)
    return pre, post


def main():
    start = dt.date(2024, 7, 1)
    end = dt.date(2026, 4, 6)

    print(f"Loading period data {start} -> {end} ...", flush=True)
    preloaded = load_period_data(start_date=start, end_date=end, quiet=True)
    print(f"  {len(preloaded[2])} trading days, vix_floor=12, avoid_days=[0,2]\n", flush=True)

    print(f"{'vix_ceil':>9s}  {'n':>4s}  {'PnL_full':>14s}  {'x':>6s}  {'WR':>5s}  {'PF':>5s}    "
          f"{'post-Sep n':>10s}  {'PnL':>12s}  {'WR':>5s}  {'PF':>5s}")
    print("-" * 130)
    for ceil in CEILS:
        cfg = dict(V17_CONFIG)
        cfg["avoid_days"] = [0, 2]
        cfg["use_v17_regime_gate"] = False
        cfg["use_v17_monwed_gate"] = False
        cfg["vix_floor"] = 12
        cfg["vix_ceil"] = ceil
        trades, _ = run_backtest(start_date=start, end_date=end,
                                 cfg_override=cfg, quiet=True, preloaded=preloaded)
        full = metrics(trades)
        _, post = split(trades, POST_SEP)
        ps = metrics(post)
        ret_x = (CAPITAL + full["pnl"]) / CAPITAL
        print(f"  {ceil:>7d}  {full['n']:>4d}  Rs {full['pnl']:>+12,.0f}  "
              f"{ret_x:>5.2f}x  {full['wr']:>4.1f}%  {full['pf']:>4.2f}    "
              f"{ps['n']:>10d}  Rs {ps['pnl']:>+10,.0f}  "
              f"{ps['wr']:>4.1f}%  {ps['pf']:>4.2f}")


if __name__ == "__main__":
    main()
