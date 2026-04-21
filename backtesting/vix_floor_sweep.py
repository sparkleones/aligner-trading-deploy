"""
vix_floor sweep for V17_PROD_ONLY config.

Tests whether lowering vix_floor (allowing entries in the ultra-low-vol
Jul-Dec 2025 regime) recovers profit or destroys it, at the current
avoid_days and at the winner selected from avoid_days_sweep.

Usage:
    python -m backtesting.vix_floor_sweep --start 2024-07-01 --end 2026-04-06
    python -m backtesting.vix_floor_sweep --avoid-days 0,2
"""
import sys
import argparse
import datetime as dt
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backtesting.v14_unified_backtest import load_period_data, run_backtest
from scoring.config import V17_CONFIG

CAPITAL = 2_00_000
POST_SEP = dt.date(2025, 9, 1)

FLOORS = [10, 11, 12, 13, 14]


def compute_metrics(trades):
    if not trades:
        return {"n": 0, "pnl": 0.0, "wr": 0.0, "pf": 0.0}
    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    total_win = sum(t["pnl"] for t in wins)
    total_loss = -sum(t["pnl"] for t in losses)
    pf = (total_win / total_loss) if total_loss > 0 else float("inf")
    return {
        "n": len(trades),
        "pnl": sum(t["pnl"] for t in trades),
        "wr": len(wins) / len(trades) * 100,
        "pf": pf,
    }


def split_by_cutover(trades, cutover):
    pre, post = [], []
    for t in trades:
        d = t.get("date")
        if isinstance(d, str):
            d = dt.date.fromisoformat(d)
        (pre if d < cutover else post).append(t)
    return pre, post


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=str, default="2024-07-01")
    parser.add_argument("--end", type=str, default="2026-04-06")
    parser.add_argument("--avoid-days", type=str, default="0,2",
                        help="comma-separated DOW indices to block (Mon=0..Fri=4)")
    args = parser.parse_args()

    start = dt.date.fromisoformat(args.start)
    end = dt.date.fromisoformat(args.end)
    avoid_days = [int(x) for x in args.avoid_days.split(",") if x.strip()]

    print(f"Loading period data {start} -> {end} ...", flush=True)
    preloaded = load_period_data(start_date=start, end_date=end, quiet=True)
    _, _, all_dates, _ = preloaded
    print(f"  {len(all_dates)} trading days.  avoid_days={avoid_days}")
    print()

    rows = []
    for floor in FLOORS:
        cfg_override = dict(V17_CONFIG)
        cfg_override["avoid_days"] = avoid_days
        cfg_override["use_v17_regime_gate"] = False
        cfg_override["use_v17_monwed_gate"] = False
        cfg_override["vix_floor"] = floor

        trades, equity = run_backtest(
            start_date=start, end_date=end,
            cfg_override=cfg_override, quiet=True,
            preloaded=preloaded,
        )
        full = compute_metrics(trades)
        _, post = split_by_cutover(trades, POST_SEP)
        sub = compute_metrics(post)
        ret_x = (CAPITAL + full["pnl"]) / CAPITAL
        rows.append((floor, full, sub, ret_x))
        print(
            f"  vix_floor={floor:2d}  n={full['n']:3d}  PnL={full['pnl']:+12,.0f}  "
            f"{ret_x:5.2f}x  WR={full['wr']:4.1f}%  PF={full['pf']:4.2f}   "
            f"| post-Sep n={sub['n']:2d} PnL={sub['pnl']:+10,.0f} "
            f"WR={sub['wr']:4.1f}% PF={sub['pf']:4.2f}"
        )

    print()
    print("Best full-window:   ", max(rows, key=lambda r: r[1]["pnl"])[0])
    print("Best post-Sep:      ", max(rows, key=lambda r: r[2]["pnl"])[0])


if __name__ == "__main__":
    main()
