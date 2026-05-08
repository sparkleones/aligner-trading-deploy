"""
Gap-classifier sweep for V17_PROD_ONLY (Option A: avoid=[0,2], vix_floor=12).

Tests whether OptionWise's "Auto-Router" gap-pct regime idea improves our
existing system as a *filter*. Computes per-day overnight gap% from prev_close
to today's first-bar OPEN, then post-filters the V17_PROD_ONLY trade list by
various gap windows.

NOTE: This is an additive filter test, NOT a re-implementation of OptionWise's
LLM trade-signal architecture (which we explicitly rejected as non-deterministic).
We only test their gap-classifier idea on top of our deterministic V17 entries.

Variants:
  - baseline                 : no gap filter (Option A as-is)
  - skip_flat_15bp           : require |gap| >= 0.15%   (skip OW "scalper" days)
  - skip_flat_30bp           : require |gap| >= 0.30%
  - skip_flat_50bp           : require |gap| >= 0.50%
  - skip_huge_60bp           : require |gap| <  0.60%   (skip OW "MR" days)
  - trend_window_15_60       : require 0.15% <= |gap| <= 0.60%   (OW pure trend)
  - fade_only_60+            : require |gap| > 0.60%   (OW MR territory)
  - flat_only_15-            : require |gap| < 0.15%   (sanity-check inversion)
  - gap_up_only              : require gap > 0
  - gap_down_only            : require gap < 0

Usage:
    python -m backtesting.gap_classifier_sweep
    python -m backtesting.gap_classifier_sweep --start 2024-07-01 --end 2026-04-06
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


def build_gap_lookup(day_groups):
    """Return {date -> signed gap_pct} using prev_close -> today's first-bar OPEN."""
    gap_lookup = {}
    sorted_dates = sorted(day_groups.keys())
    prev_close = 0.0
    for d in sorted_dates:
        bars = day_groups[d]
        if not bars:
            continue
        day_open = bars[0]["open"]
        if prev_close > 0:
            gap_lookup[d] = (day_open - prev_close) / prev_close * 100.0
        prev_close = bars[-1]["close"]
    return gap_lookup


# ── Filter predicates: gap_pct is SIGNED percent (e.g. +0.42 means +0.42%) ──
FILTERS = [
    ("baseline",            lambda g: True),
    ("skip_flat_15bp",      lambda g: abs(g) >= 0.15),
    ("skip_flat_30bp",      lambda g: abs(g) >= 0.30),
    ("skip_flat_50bp",      lambda g: abs(g) >= 0.50),
    ("skip_huge_60bp",      lambda g: abs(g) <  0.60),
    ("trend_window_15_60",  lambda g: 0.15 <= abs(g) <= 0.60),
    ("fade_only_60+",       lambda g: abs(g) >  0.60),
    ("flat_only_15-",       lambda g: abs(g) <  0.15),
    ("gap_up_only",         lambda g: g >  0),
    ("gap_down_only",       lambda g: g <  0),
]


def compute_metrics(trades):
    if not trades:
        return {"n": 0, "pnl": 0.0, "wr": 0.0, "pf": 0.0,
                "avg_win": 0.0, "avg_loss": 0.0}
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
        "avg_win": total_win / max(1, len(wins)),
        "avg_loss": -total_loss / max(1, len(losses)),
    }


def split_by_cutover(trades, cutover):
    pre, post = [], []
    for t in trades:
        d = t.get("date")
        if isinstance(d, str):
            d = dt.date.fromisoformat(d)
        (pre if d < cutover else post).append(t)
    return pre, post


def filter_trades(trades, gap_lookup, predicate):
    out = []
    for t in trades:
        d = t.get("date")
        if isinstance(d, str):
            d = dt.date.fromisoformat(d)
        g = gap_lookup.get(d)
        if g is None:
            continue
        if predicate(g):
            out.append(t)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=str, default="2024-07-01")
    parser.add_argument("--end",   type=str, default="2026-04-06")
    args = parser.parse_args()

    start = dt.date.fromisoformat(args.start)
    end   = dt.date.fromisoformat(args.end)

    print(f"Loading period data {start} -> {end} ...", flush=True)
    preloaded = load_period_data(start_date=start, end_date=end, quiet=True)
    day_groups, _, all_dates, _ = preloaded
    print(f"  {len(all_dates)} trading days in window", flush=True)

    # Build gap% lookup once
    gap_lookup = build_gap_lookup(day_groups)
    print(f"  gap% lookup built for {len(gap_lookup)} days", flush=True)

    # Distribution of gap% (helps interpret the variants)
    in_window_gaps = [g for d, g in gap_lookup.items() if d in set(all_dates)]
    abs_g = [abs(g) for g in in_window_gaps]
    if abs_g:
        n = len(abs_g)
        n_lt_15 = sum(1 for g in abs_g if g < 0.15)
        n_15_60 = sum(1 for g in abs_g if 0.15 <= g <= 0.60)
        n_gt_60 = sum(1 for g in abs_g if g > 0.60)
        print(f"  gap distribution: |g|<0.15%: {n_lt_15} ({n_lt_15/n*100:.1f}%) "
              f"| 0.15-0.60%: {n_15_60} ({n_15_60/n*100:.1f}%) "
              f"| >0.60%: {n_gt_60} ({n_gt_60/n*100:.1f}%)")
    print()

    # Run V17_PROD_ONLY (Option A) ONCE
    cfg_override = dict(V17_CONFIG)
    cfg_override["avoid_days"] = [0, 2]
    cfg_override["use_v17_regime_gate"] = False
    cfg_override["use_v17_monwed_gate"] = False
    cfg_override["vix_floor"] = 12  # Option A in production

    print("Running V17_PROD_ONLY (Option A) baseline backtest...", flush=True)
    trades, equity = run_backtest(
        start_date=start, end_date=end,
        cfg_override=cfg_override, quiet=True,
        preloaded=preloaded,
    )
    print(f"  {len(trades)} trades generated", flush=True)
    print()

    # Apply each filter variant
    rows = []
    print(f"{'variant':22s} {'n':>4s}  {'PnL':>14s}  {'x':>6s}  {'WR':>5s}  {'PF':>5s}   "
          f"{'post-Sep n':>10s} {'PnL':>12s} {'WR':>5s} {'PF':>5s}   retained%")
    print("-" * 130)
    for label, pred in FILTERS:
        sub_trades = filter_trades(trades, gap_lookup, pred)
        full = compute_metrics(sub_trades)
        _, post = split_by_cutover(sub_trades, POST_SEP)
        ps = compute_metrics(post)
        ret_x = (CAPITAL + full["pnl"]) / CAPITAL
        retained_pct = full["n"] / max(1, len(trades)) * 100
        rows.append((label, full, ps, ret_x, retained_pct))
        print(
            f"  {label:20s} {full['n']:>4d}  Rs {full['pnl']:>+12,.0f}  "
            f"{ret_x:>5.2f}x  {full['wr']:>4.1f}%  {full['pf']:>4.2f}   "
            f"   {ps['n']:>4d}  Rs {ps['pnl']:>+10,.0f}  {ps['wr']:>4.1f}%  {ps['pf']:>4.2f}   "
            f"{retained_pct:>5.1f}%"
        )

    print()
    print("=" * 130)
    print("RANKED BY FULL-WINDOW PnL")
    print("=" * 130)
    for label, full, ps, ret_x, ret_pct in sorted(rows, key=lambda r: -r[1]["pnl"]):
        print(
            f"  {label:22s} full: {ret_x:5.2f}x / Rs {full['pnl']:+12,.0f} "
            f"(n={full['n']:>3d}, PF={full['pf']:.2f}, WR={full['wr']:.1f}%, "
            f"retained={ret_pct:.0f}%)   post-Sep: Rs {ps['pnl']:+10,.0f} "
            f"(n={ps['n']}, PF={ps['pf']:.2f})"
        )

    print()
    print("=" * 130)
    print("RANKED BY POST-SEP-2025 PnL")
    print("=" * 130)
    for label, full, ps, ret_x, ret_pct in sorted(rows, key=lambda r: -r[2]["pnl"]):
        print(
            f"  {label:22s} post-Sep: Rs {ps['pnl']:+10,.0f} "
            f"(n={ps['n']:>3d}, PF={ps['pf']:.2f}, WR={ps['wr']:.1f}%)   "
            f"full: {ret_x:5.2f}x / Rs {full['pnl']:+12,.0f}"
        )

    print()
    print("=" * 130)
    print("RANKED BY POST-SEP-2025 PROFIT FACTOR (quality, not gross)")
    print("=" * 130)
    for label, full, ps, ret_x, ret_pct in sorted(
        rows, key=lambda r: -(r[2]["pf"] if r[2]["n"] >= 5 else -1)
    ):
        if ps["n"] < 5:
            continue
        print(
            f"  {label:22s} post-Sep PF={ps['pf']:.2f}  "
            f"PnL=Rs {ps['pnl']:+10,.0f}  n={ps['n']}  WR={ps['wr']:.1f}%   "
            f"full: PF={full['pf']:.2f}  Rs {full['pnl']:+12,.0f}"
        )


if __name__ == "__main__":
    main()
