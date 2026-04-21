"""
avoid_days sweep for V17_PROD_ONLY config.

Shares one data load, runs N variants, reports full-window + post-Sep-2025
sub-metrics so you can see whether the winning DOW filter shifted after the
SEBI expiry cutover.

Usage:
    python -m backtesting.avoid_days_sweep --start 2024-07-01 --end 2026-04-06
"""
import sys
import argparse
import datetime as dt
from pathlib import Path
from collections import defaultdict

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backtesting.v14_unified_backtest import (
    load_period_data,
    run_backtest,
)
from scoring.config import V17_CONFIG


CAPITAL = 2_00_000  # mirror backtest default


# DOW index: Mon=0, Tue=1, Wed=2, Thu=3, Fri=4
SWEEP_VARIANTS = [
    ("none",       []),
    ("mon",        [0]),
    ("tue",        [1]),
    ("wed",        [2]),
    ("thu",        [3]),
    ("fri",        [4]),
    ("mon_wed",    [0, 2]),   # current V17_PROD_ONLY
    ("mon_tue",    [0, 1]),
    ("tue_wed",    [1, 2]),
    ("wed_thu",    [2, 3]),
    ("mon_wed_thu",[0, 2, 3]),
    ("mon_wed_fri",[0, 2, 4]),
]

POST_SEP = dt.date(2025, 9, 1)


def compute_metrics(trades, start_date, end_date):
    if not trades:
        return {
            "n": 0, "pnl": 0.0, "wins": 0, "losses": 0, "wr": 0.0,
            "pf": 0.0, "avg_win": 0.0, "avg_loss": 0.0,
        }
    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    total_win = sum(t["pnl"] for t in wins)
    total_loss = -sum(t["pnl"] for t in losses)
    pf = (total_win / total_loss) if total_loss > 0 else float("inf")
    return {
        "n": len(trades),
        "pnl": sum(t["pnl"] for t in trades),
        "wins": len(wins),
        "losses": len(losses),
        "wr": len(wins) / len(trades) * 100 if trades else 0.0,
        "pf": pf,
        "avg_win": total_win / max(1, len(wins)),
        "avg_loss": -total_loss / max(1, len(losses)),
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=str, default="2024-07-01")
    parser.add_argument("--end", type=str, default="2026-04-06")
    args = parser.parse_args()

    start = dt.date.fromisoformat(args.start)
    end = dt.date.fromisoformat(args.end)

    print(f"Loading period data {start} -> {end} ...", flush=True)
    preloaded = load_period_data(start_date=start, end_date=end, quiet=True)
    _, _, all_dates, _ = preloaded
    print(f"  {len(all_dates)} trading days in window")
    print()

    results = []
    for label, avoid_days in SWEEP_VARIANTS:
        cfg_override = dict(V17_CONFIG)
        cfg_override["avoid_days"] = list(avoid_days)
        cfg_override["use_v17_regime_gate"] = False
        cfg_override["use_v17_monwed_gate"] = False

        trades, equity = run_backtest(
            start_date=start, end_date=end,
            cfg_override=cfg_override, quiet=True,
            preloaded=preloaded,
        )

        # Full-window metrics
        full = compute_metrics(trades, start, end)
        # Post-Sep-2025 sub-metrics
        _, post = split_by_cutover(trades, POST_SEP)
        sub = compute_metrics(post, POST_SEP, end)

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
            f"  [{label:12s}] n={full['n']:3d}  PnL={full['pnl']:+12,.0f}  "
            f"{ret_x:5.2f}x  WR={full['wr']:4.1f}%  PF={full['pf']:4.2f}   "
            f"| post-Sep n={sub['n']:2d} PnL={sub['pnl']:+10,.0f} "
            f"WR={sub['wr']:4.1f}% PF={sub['pf']:4.2f}"
        )

    print()
    print("=" * 110)
    print("RANKED BY FULL-WINDOW P&L")
    print("=" * 110)
    for r in sorted(results, key=lambda x: -x["full"]["pnl"]):
        print(
            f"  {r['label']:14s} avoid_days={str(r['avoid_days']):14s}  "
            f"full: {r['ret_x']:5.2f}x / Rs {r['full']['pnl']:+12,.0f} "
            f"(n={r['full']['n']}, PF={r['full']['pf']:.2f}, WR={r['full']['wr']:.1f}%)   "
            f"post-Sep: Rs {r['post']['pnl']:+10,.0f} "
            f"(n={r['post']['n']}, PF={r['post']['pf']:.2f})"
        )

    print()
    print("=" * 110)
    print("RANKED BY POST-SEP-2025 P&L")
    print("=" * 110)
    for r in sorted(results, key=lambda x: -x["post"]["pnl"]):
        print(
            f"  {r['label']:14s} avoid_days={str(r['avoid_days']):14s}  "
            f"post-Sep: Rs {r['post']['pnl']:+10,.0f} "
            f"(n={r['post']['n']}, PF={r['post']['pf']:.2f}, WR={r['post']['wr']:.1f}%)   "
            f"full: {r['ret_x']:5.2f}x / Rs {r['full']['pnl']:+12,.0f}"
        )


if __name__ == "__main__":
    main()
