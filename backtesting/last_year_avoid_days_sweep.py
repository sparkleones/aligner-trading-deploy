"""
Last 1-year window avoid_days sweep on current Option B config.

Re-runs the day-of-week filter sweep on the realistic 12-month window
(2025-04-07 -> 2026-04-06) using V17_CONFIG with use_v17_regime_gate and
use_v17_monwed_gate disabled, plus vix_floor=12 / vix_ceil=25 (current Option B).

Variants intentionally narrowed to the 7 most relevant for the question:
  - []          (none, baseline)
  - [0,2]       (current production: Mon+Wed)
  - [0,2,3]     (previously rejected "max quality": +Thu)
  - [0,2,4]     (Mon+Wed+Fri)
  - [0,1,2]     (Mon+Tue+Wed)
  - [2]         (Wed only)
  - [0]         (Mon only)

Usage:
    python -m backtesting.last_year_avoid_days_sweep
"""
import sys
import datetime as dt
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backtesting.v14_unified_backtest import (
    load_period_data,
    run_backtest,
)
from scoring.config import V17_CONFIG


CAPITAL = 2_00_000
POST_SEP = dt.date(2025, 9, 1)
START = dt.date(2025, 4, 7)
END = dt.date(2026, 4, 6)


# DOW: Mon=0, Tue=1, Wed=2, Thu=3, Fri=4
SWEEP_VARIANTS = [
    ("none",         []),
    ("mon_wed",      [0, 2]),     # current production
    ("mon_wed_thu",  [0, 2, 3]),  # previously rejected "max quality"
    ("mon_wed_fri",  [0, 2, 4]),
    ("mon_tue_wed",  [0, 1, 2]),
    ("wed",          [2]),
    ("mon",          [0]),
]


def compute_metrics(trades):
    if not trades:
        return {
            "n": 0, "pnl": 0.0, "wins": 0, "losses": 0, "wr": 0.0,
            "pf": 0.0, "avg_win": 0.0, "avg_loss": 0.0,
            "max_dd": 0.0, "max_dd_pct": 0.0,
        }
    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    total_win = sum(t["pnl"] for t in wins)
    total_loss = -sum(t["pnl"] for t in losses)
    pf = (total_win / total_loss) if total_loss > 0 else float("inf")

    # Max drawdown on running equity (capital-relative)
    equity = CAPITAL
    peak = CAPITAL
    max_dd = 0.0
    max_dd_pct = 0.0
    sorted_trades = sorted(
        trades,
        key=lambda t: (
            dt.date.fromisoformat(t["date"]) if isinstance(t.get("date"), str) else t.get("date")
        ),
    )
    for t in sorted_trades:
        equity += t["pnl"]
        if equity > peak:
            peak = equity
        dd = peak - equity
        if dd > max_dd:
            max_dd = dd
            max_dd_pct = (dd / peak * 100) if peak > 0 else 0.0

    return {
        "n": len(trades),
        "pnl": sum(t["pnl"] for t in trades),
        "wins": len(wins),
        "losses": len(losses),
        "wr": len(wins) / len(trades) * 100 if trades else 0.0,
        "pf": pf,
        "avg_win": total_win / max(1, len(wins)),
        "avg_loss": -total_loss / max(1, len(losses)),
        "max_dd": max_dd,
        "max_dd_pct": max_dd_pct,
    }


def split_by_cutover(trades, cutover):
    pre, post = [], []
    for t in trades:
        d = t.get("date")
        if isinstance(d, str):
            d = dt.date.fromisoformat(d)
        if d < cutover:
            pre.append(t)
        else:
            post.append(t)
    return pre, post


def main():
    print(f"Loading period data {START} -> {END} ...", flush=True)
    preloaded = load_period_data(start_date=START, end_date=END, quiet=True)
    _, _, all_dates, _ = preloaded
    print(f"  {len(all_dates)} trading days in window")
    print()

    results = []
    for label, avoid_days in SWEEP_VARIANTS:
        cfg_override = dict(V17_CONFIG)
        cfg_override["avoid_days"] = list(avoid_days)
        cfg_override["use_v17_regime_gate"] = False
        cfg_override["use_v17_monwed_gate"] = False
        cfg_override["vix_floor"] = 12
        cfg_override["vix_ceil"] = 25

        trades, equity = run_backtest(
            start_date=START, end_date=END,
            cfg_override=cfg_override, quiet=True,
            preloaded=preloaded,
        )

        full = compute_metrics(trades)
        _, post = split_by_cutover(trades, POST_SEP)
        sub = compute_metrics(post)

        ret_x = (CAPITAL + full["pnl"]) / CAPITAL
        results.append({
            "label": label,
            "avoid_days": avoid_days,
            "full": full,
            "post": sub,
            "ret_x": ret_x,
            "final_equity": CAPITAL + full["pnl"],
        })
        print(
            f"  [{label:14s}] avoid={str(avoid_days):12s}  "
            f"n={full['n']:3d}  PnL={full['pnl']:+12,.0f}  "
            f"{ret_x:5.2f}x  WR={full['wr']:4.1f}%  PF={full['pf']:5.2f}  "
            f"DD={full['max_dd']:>10,.0f} ({full['max_dd_pct']:4.1f}%)  "
            f"| post-Sep n={sub['n']:2d} PnL={sub['pnl']:+10,.0f} "
            f"WR={sub['wr']:4.1f}% PF={sub['pf']:5.2f} "
            f"DD={sub['max_dd']:>9,.0f} ({sub['max_dd_pct']:4.1f}%)"
        )

    print()
    print("=" * 120)
    print("RANKED BY POST-SEP-2025 PROFIT FACTOR  (window: 2025-04-07 -> 2026-04-06, vix_floor=12, vix_ceil=25)")
    print("=" * 120)
    print(
        f"  {'label':14s} {'avoid_days':14s}  {'n':>3s} {'PnL':>12s} {'ret_x':>6s} "
        f"{'WR':>5s} {'PF':>6s} {'maxDD':>10s} {'DD%':>5s}  | "
        f"{'post_n':>6s} {'post_PnL':>10s} {'post_WR':>7s} {'post_PF':>7s} {'post_DD':>9s} {'DD%':>5s}"
    )
    print("-" * 120)
    for r in sorted(results, key=lambda x: -x["post"]["pf"]):
        full = r["full"]
        sub = r["post"]
        print(
            f"  {r['label']:14s} {str(r['avoid_days']):14s}  "
            f"{full['n']:3d} {full['pnl']:+12,.0f} {r['ret_x']:5.2f}x "
            f"{full['wr']:4.1f}% {full['pf']:5.2f} {full['max_dd']:>10,.0f} {full['max_dd_pct']:4.1f}%  | "
            f"{sub['n']:6d} {sub['pnl']:+10,.0f} {sub['wr']:6.1f}% {sub['pf']:6.2f} "
            f"{sub['max_dd']:>9,.0f} {sub['max_dd_pct']:4.1f}%"
        )

    print()
    print("=" * 120)
    print("RANKED BY FULL-WINDOW P&L")
    print("=" * 120)
    for r in sorted(results, key=lambda x: -x["full"]["pnl"]):
        full = r["full"]
        sub = r["post"]
        print(
            f"  {r['label']:14s} avoid_days={str(r['avoid_days']):14s}  "
            f"full: {r['ret_x']:5.2f}x / Rs {full['pnl']:+12,.0f} "
            f"(n={full['n']}, PF={full['pf']:.2f}, WR={full['wr']:.1f}%, DD={full['max_dd_pct']:.1f}%)   "
            f"post-Sep: Rs {sub['pnl']:+10,.0f} (n={sub['n']}, PF={sub['pf']:.2f})"
        )


if __name__ == "__main__":
    main()
