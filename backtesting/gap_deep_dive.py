"""
Deep-dive analysis of V17_PROD_ONLY trades bucketed by overnight gap%.

Specifically dissects:
  - The flat_only_15- finding (PF 3.72 post-Sep, n=13)
  - The fade_only_60+ slice (where OptionWise's MR thesis says we should win)
  - The trend_window_15_60 slice (where their trend thesis says we should win)

For each bucket, reports:
  - n trades, P&L, WR, PF
  - Day-of-week distribution
  - Exit-reason breakdown
  - Top 5 wins / Top 5 losses (date, spot, P&L, exit reason)
  - Per-trade list (so we can see if 1-2 outliers dominate)

Usage:
    python -m backtesting.gap_deep_dive
"""
import sys
import datetime as dt
from pathlib import Path
from collections import Counter, defaultdict

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backtesting.v14_unified_backtest import load_period_data, run_backtest
from scoring.config import V17_CONFIG

CAPITAL = 2_00_000
POST_SEP = dt.date(2025, 9, 1)
DOW_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def build_gap_lookup(day_groups):
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


def metrics(trades):
    if not trades:
        return {"n": 0, "pnl": 0.0, "wr": 0.0, "pf": 0.0,
                "avg_win": 0.0, "avg_loss": 0.0,
                "med_pnl": 0.0, "max_win": 0.0, "max_loss": 0.0}
    pnls = sorted([t["pnl"] for t in trades])
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    tw = sum(wins)
    tl = -sum(losses)
    pf = tw / tl if tl > 0 else float("inf")
    return {
        "n": len(trades),
        "pnl": sum(pnls),
        "wr": len(wins) / len(trades) * 100,
        "pf": pf,
        "avg_win": tw / max(1, len(wins)),
        "avg_loss": -tl / max(1, len(losses)),
        "med_pnl": pnls[len(pnls) // 2],
        "max_win": max(pnls),
        "max_loss": min(pnls),
    }


def to_date(d):
    if isinstance(d, str):
        return dt.date.fromisoformat(d)
    return d


def bucket_label(gap):
    if gap is None:
        return "unknown"
    a = abs(gap)
    if a < 0.15:
        return "flat (<0.15%)"
    if a <= 0.60:
        return "trend (0.15-0.60%)"
    return "fade (>0.60%)"


def show_bucket(name, trades, gap_lookup, max_show=999):
    print(f"\n{'=' * 120}")
    print(f"BUCKET: {name}   (n={len(trades)})")
    print(f"{'=' * 120}")
    if not trades:
        print("  (empty)")
        return

    m = metrics(trades)
    print(f"  P&L: Rs {m['pnl']:+,.0f}   WR={m['wr']:.1f}%   PF={m['pf']:.2f}")
    print(f"  Avg win: Rs {m['avg_win']:+,.0f}    Avg loss: Rs {m['avg_loss']:+,.0f}    Median: Rs {m['med_pnl']:+,.0f}")
    print(f"  Max win: Rs {m['max_win']:+,.0f}    Max loss: Rs {m['max_loss']:+,.0f}")

    # Day-of-week distribution
    dow_pnl = defaultdict(float)
    dow_n = Counter()
    for t in trades:
        d = to_date(t["date"])
        dow_n[d.weekday()] += 1
        dow_pnl[d.weekday()] += t["pnl"]
    dow_str = "  DOW: "
    for i in range(5):
        if dow_n[i] > 0:
            dow_str += f"{DOW_NAMES[i]} n={dow_n[i]} pnl=Rs {dow_pnl[i]:+,.0f}  | "
    print(dow_str.rstrip(" |"))

    # Exit reasons
    exit_counts = Counter(t["exit_reason"] for t in trades)
    exit_pnl = defaultdict(float)
    for t in trades:
        exit_pnl[t["exit_reason"]] += t["pnl"]
    print("  Exit reasons:")
    for reason, n in exit_counts.most_common():
        print(f"    {reason:25s} n={n:3d}  pnl=Rs {exit_pnl[reason]:+,.0f}")

    # Top 5 wins
    sorted_trades = sorted(trades, key=lambda t: -t["pnl"])
    print("  Top 5 wins:")
    for t in sorted_trades[:5]:
        d = to_date(t["date"])
        print(f"    {d}  ({DOW_NAMES[d.weekday()]:3s})  gap={gap_lookup.get(d, 0):+5.2f}%  "
              f"{t['action']:9s} {t['strike']:.0f}{t['opt_type']}  "
              f"pnl=Rs {t['pnl']:+,.0f}  exit={t['exit_reason']}")
    print("  Top 5 losses:")
    for t in sorted_trades[-5:][::-1]:
        d = to_date(t["date"])
        print(f"    {d}  ({DOW_NAMES[d.weekday()]:3s})  gap={gap_lookup.get(d, 0):+5.2f}%  "
              f"{t['action']:9s} {t['strike']:.0f}{t['opt_type']}  "
              f"pnl=Rs {t['pnl']:+,.0f}  exit={t['exit_reason']}")


def main():
    start = dt.date(2024, 7, 1)
    end   = dt.date(2026, 4, 6)

    print("Loading period data...", flush=True)
    preloaded = load_period_data(start_date=start, end_date=end, quiet=True)
    day_groups, _, all_dates, _ = preloaded
    gap_lookup = build_gap_lookup(day_groups)
    print(f"  {len(all_dates)} days, gap lookup for {len(gap_lookup)} days", flush=True)

    cfg = dict(V17_CONFIG)
    cfg["avoid_days"] = [0, 2]
    cfg["use_v17_regime_gate"] = False
    cfg["use_v17_monwed_gate"] = False
    cfg["vix_floor"] = 12

    print("Running V17_PROD_ONLY (Option A)...", flush=True)
    trades, _ = run_backtest(start_date=start, end_date=end,
                             cfg_override=cfg, quiet=True, preloaded=preloaded)
    print(f"  {len(trades)} trades\n", flush=True)

    # Tag each trade with gap%
    for t in trades:
        d = to_date(t["date"])
        t["gap_pct"] = gap_lookup.get(d)

    # Buckets
    flat = [t for t in trades if t.get("gap_pct") is not None and abs(t["gap_pct"]) < 0.15]
    trend = [t for t in trades if t.get("gap_pct") is not None and 0.15 <= abs(t["gap_pct"]) <= 0.60]
    fade = [t for t in trades if t.get("gap_pct") is not None and abs(t["gap_pct"]) > 0.60]

    flat_post = [t for t in flat if to_date(t["date"]) >= POST_SEP]
    trend_post = [t for t in trend if to_date(t["date"]) >= POST_SEP]
    fade_post = [t for t in fade if to_date(t["date"]) >= POST_SEP]

    print("=" * 120)
    print("SUMMARY: trades by gap bucket")
    print("=" * 120)
    print(f"{'bucket':25s} {'n_full':>8s} {'PnL_full':>14s} {'WR%':>7s} {'PF':>6s}   "
          f"{'n_post':>8s} {'PnL_post':>14s} {'WR%':>7s} {'PF':>6s}")
    for name, full, post in [
        ("flat (<0.15%)",  flat,  flat_post),
        ("trend (0.15-0.6%)", trend, trend_post),
        ("fade (>0.60%)",  fade,  fade_post),
    ]:
        mf = metrics(full)
        mp = metrics(post)
        print(f"  {name:23s} {mf['n']:>8d} {mf['pnl']:>+14,.0f} {mf['wr']:>6.1f}% {mf['pf']:>6.2f}   "
              f"{mp['n']:>8d} {mp['pnl']:>+14,.0f} {mp['wr']:>6.1f}% {mp['pf']:>6.2f}")

    # Detailed dive on each bucket — full window
    show_bucket("FLAT |gap|<0.15% — FULL WINDOW", flat, gap_lookup)
    show_bucket("FLAT |gap|<0.15% — POST-SEP ONLY", flat_post, gap_lookup)
    show_bucket("FADE |gap|>0.60% — FULL WINDOW", fade, gap_lookup)
    show_bucket("FADE |gap|>0.60% — POST-SEP ONLY", fade_post, gap_lookup)
    show_bucket("TREND 0.15-0.60% — POST-SEP ONLY", trend_post, gap_lookup)

    # Concentration check — if 1-2 trades dominate flat_post P&L
    print(f"\n{'=' * 120}")
    print("CONCENTRATION CHECK: how much of flat_post P&L comes from top 3 trades?")
    print(f"{'=' * 120}")
    if flat_post:
        sorted_p = sorted([t["pnl"] for t in flat_post], reverse=True)
        total = sum(sorted_p)
        top3 = sum(sorted_p[:3])
        print(f"  total P&L (n={len(flat_post)}): Rs {total:+,.0f}")
        print(f"  top 3:                            Rs {top3:+,.0f}  ({top3/total*100:.1f}% of total)")
        print(f"  remaining {len(flat_post)-3} trades:           Rs {total-top3:+,.0f}")


if __name__ == "__main__":
    main()
