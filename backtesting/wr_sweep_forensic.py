"""
Forensic deep-dive on the WR-max sweep results.

Specifically investigates:
  1. Why AGGRESSIVE_STALE produced 397 trades (2x baseline) — re-entry churn?
  2. Per-trade size distribution (are AGGRESSIVE_STALE trades all small?)
  3. Trades-per-day distribution comparison
  4. Exit-reason mix per variant
  5. What happens to monthly P&L under each variant

Output: complete analysis table for each variant with deeper metrics.

Usage:
    python -m backtesting.wr_sweep_forensic
"""
import sys
import datetime as dt
from pathlib import Path
from collections import defaultdict, Counter

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backtesting.v14_unified_backtest import load_period_data, run_backtest
from scoring.config import V17_CONFIG

CAPITAL = 2_00_000


def run_variant(label, overrides, preloaded, start, end):
    cfg = dict(V17_CONFIG)
    cfg["avoid_days"] = [0, 2]
    cfg["use_v17_regime_gate"] = False
    cfg["use_v17_monwed_gate"] = False
    cfg["vix_floor"] = 12
    cfg["vix_ceil"] = 25
    cfg.update(overrides)
    trades, _ = run_backtest(start_date=start, end_date=end,
                             cfg_override=cfg, quiet=True, preloaded=preloaded)
    return label, trades


def to_date(d):
    return dt.date.fromisoformat(d) if isinstance(d, str) else d


def analyze(label, trades):
    n = len(trades)
    if n == 0:
        return {"label": label, "n": 0}

    pnls = [t["pnl"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    tw = sum(wins)
    tl = -sum(losses)

    # Trades per day
    by_date = defaultdict(list)
    for t in trades:
        by_date[to_date(t["date"])].append(t)
    n_days = len(by_date)
    trades_per_day = sorted([len(v) for v in by_date.values()])
    median_tpd = trades_per_day[len(trades_per_day) // 2]
    max_tpd = max(trades_per_day)
    days_with_5plus = sum(1 for v in trades_per_day if v >= 5)

    # Exit reason mix
    exit_mix = Counter(t["exit_reason"] for t in trades)
    exit_pnl = defaultdict(float)
    for t in trades:
        exit_pnl[t["exit_reason"]] += t["pnl"]

    # Bars-held distribution
    bars_held = [t.get("bars_held", 0) for t in trades]
    median_bars = sorted(bars_held)[len(bars_held) // 2]
    avg_bars = sum(bars_held) / len(bars_held)

    # P&L magnitude buckets
    tiny_wins = sum(1 for p in pnls if 0 < p < 5000)
    small_wins = sum(1 for p in pnls if 5000 <= p < 50000)
    big_wins = sum(1 for p in pnls if p >= 50000)
    tiny_losses = sum(1 for p in pnls if -5000 < p <= 0)
    small_losses = sum(1 for p in pnls if -50000 < p <= -5000)
    big_losses = sum(1 for p in pnls if p <= -50000)

    return {
        "label": label,
        "n": n,
        "wr": len(wins) / n * 100,
        "pf": tw / tl if tl > 0 else float("inf"),
        "pnl": sum(pnls),
        "avg_win": tw / max(1, len(wins)),
        "avg_loss": -tl / max(1, len(losses)),
        "n_days": n_days,
        "median_tpd": median_tpd,
        "max_tpd": max_tpd,
        "days_with_5plus": days_with_5plus,
        "exit_mix": dict(exit_mix),
        "exit_pnl": dict(exit_pnl),
        "median_bars": median_bars,
        "avg_bars": avg_bars,
        "tiny_wins": tiny_wins, "small_wins": small_wins, "big_wins": big_wins,
        "tiny_losses": tiny_losses, "small_losses": small_losses, "big_losses": big_losses,
    }


def main():
    start = dt.date(2024, 7, 1)
    end = dt.date(2026, 4, 6)
    print(f"Loading period data {start} -> {end} ...", flush=True)
    preloaded = load_period_data(start_date=start, end_date=end, quiet=True)
    print(f"  {len(preloaded[2])} trading days\n", flush=True)

    variants = [
        ("BASELINE",        {}),
        ("TIGHT_TRAIL",     {
            "trail_pct_put": 0.005, "trail_pct_call": 0.003,
            "min_hold_trail_put": 6, "min_hold_trail_call": 6,
        }),
        ("WIDE_TRAIL",      {
            "trail_pct_put": 0.025, "trail_pct_call": 0.020,
        }),
        ("AGGRESSIVE_STALE", {
            "stale_exit_bars": 3, "stale_exit_pct": 0.001,
        }),
        ("MAX_WR",          {
            "trail_pct_put": 0.005, "trail_pct_call": 0.003,
            "min_hold_trail_put": 6, "min_hold_trail_call": 6,
            "trail_min_profit_pct": 0.0,
            "stale_exit_bars": 3, "stale_exit_pct": 0.001,
        }),
    ]

    results = []
    for label, overrides in variants:
        print(f"Running {label} ...", flush=True)
        _, trades = run_variant(label, overrides, preloaded, start, end)
        a = analyze(label, trades)
        results.append(a)

    # ── Headline table ──
    print()
    print("=" * 130)
    print("  HEADLINE METRICS")
    print("=" * 130)
    print(f"  {'variant':<18} {'n':>4} {'WR%':>6} {'PF':>5} {'PnL':>14} {'avg_win':>10} {'avg_loss':>10} {'med_bars':>9}")
    for r in results:
        print(f"  {r['label']:<18} {r['n']:>4d} {r['wr']:>5.1f}% {r['pf']:>5.2f} "
              f"Rs {r['pnl']:>+12,.0f} Rs {r['avg_win']:>+8,.0f} Rs {r['avg_loss']:>+8,.0f} "
              f"{r['median_bars']:>9d}")

    # ── Trade-count explosion analysis ──
    print()
    print("=" * 130)
    print("  TRADE-COUNT EXPLOSION ANALYSIS  (where do extra trades come from?)")
    print("=" * 130)
    print(f"  {'variant':<18} {'n':>4} {'days':>5} {'med_tpd':>8} {'max_tpd':>8} {'days>=5':>9} {'avg_bars':>9}")
    for r in results:
        print(f"  {r['label']:<18} {r['n']:>4d} {r['n_days']:>5d} "
              f"{r['median_tpd']:>8d} {r['max_tpd']:>8d} {r['days_with_5plus']:>9d} "
              f"{r['avg_bars']:>9.1f}")

    # ── Exit-reason mix ──
    print()
    print("=" * 130)
    print("  EXIT-REASON MIX BY VARIANT  (where edge is gained/lost)")
    print("=" * 130)
    all_reasons = sorted(set(r2 for r in results for r2 in r["exit_mix"].keys()))
    hdr = f"  {'variant':<18}"
    for reason in all_reasons:
        hdr += f" {reason[:10]:>11}"
    print(hdr)
    for r in results:
        line = f"  {r['label']:<18}"
        for reason in all_reasons:
            n = r["exit_mix"].get(reason, 0)
            pnl = r["exit_pnl"].get(reason, 0)
            line += f" {n:>4}({pnl/1000:+5.0f}K)"[:12].rjust(11)
        print(line)

    # ── P&L distribution ──
    print()
    print("=" * 130)
    print("  P&L MAGNITUDE DISTRIBUTION  (tiny < 5K, small 5K-50K, big > 50K)")
    print("=" * 130)
    print(f"  {'variant':<18} {'tiny_W':>7} {'small_W':>8} {'big_W':>7} {'tiny_L':>7} {'small_L':>8} {'big_L':>7}")
    for r in results:
        print(f"  {r['label']:<18} {r['tiny_wins']:>7d} {r['small_wins']:>8d} {r['big_wins']:>7d} "
              f"{r['tiny_losses']:>7d} {r['small_losses']:>8d} {r['big_losses']:>7d}")

    # ── The smoking-gun analysis ──
    print()
    print("=" * 130)
    print("  WHY AGGRESSIVE_STALE FAILED  (the math)")
    print("=" * 130)
    base = next(r for r in results if r["label"] == "BASELINE")
    agg = next(r for r in results if r["label"] == "AGGRESSIVE_STALE")
    print(f"  BASELINE:        {base['n']} trades / {base['n_days']} days "
          f"= {base['n']/base['n_days']:.2f} trades/day median")
    print(f"  AGGRESSIVE_STALE: {agg['n']} trades / {agg['n_days']} days "
          f"= {agg['n']/agg['n_days']:.2f} trades/day median")
    print(f"  Trade explosion ratio: {agg['n']/base['n']:.2f}x baseline")
    print(f"  Avg bars held: {base['avg_bars']:.1f} -> {agg['avg_bars']:.1f} "
          f"({(agg['avg_bars']-base['avg_bars'])/base['avg_bars']*100:+.0f}%)")
    print()
    print(f"  Net effect on edge:")
    print(f"    avg_win  Rs {base['avg_win']:+,.0f} -> Rs {agg['avg_win']:+,.0f} "
          f"({(agg['avg_win']-base['avg_win'])/base['avg_win']*100:+.0f}%)")
    print(f"    avg_loss Rs {base['avg_loss']:+,.0f} -> Rs {agg['avg_loss']:+,.0f} "
          f"({(agg['avg_loss']-base['avg_loss'])/base['avg_loss']*100:+.0f}%)")
    print(f"    WR%      {base['wr']:.1f}% -> {agg['wr']:.1f}% "
          f"({agg['wr']-base['wr']:+.1f}pp)")
    print()
    print(f"  Expectancy per trade:")
    base_ev = base['wr']/100 * base['avg_win'] - (1-base['wr']/100) * base['avg_loss']
    agg_ev  =  agg['wr']/100 *  agg['avg_win'] - (1-agg['wr']/100) *  agg['avg_loss']
    print(f"    BASELINE:        WR×avg_win - (1-WR)×avg_loss = "
          f"Rs {base_ev:+,.0f} per trade")
    print(f"    AGGRESSIVE_STALE: same formula                = "
          f"Rs {agg_ev:+,.0f} per trade")
    print(f"  Net P&L = expectancy × n_trades:")
    print(f"    BASELINE:        Rs {base_ev:+,.0f} × {base['n']} = Rs {base_ev*base['n']:+,.0f}")
    print(f"    AGGRESSIVE_STALE: Rs {agg_ev:+,.0f} × {agg['n']} = Rs {agg_ev*agg['n']:+,.0f}")
    print()
    print(f"  Conclusion: stale-exit broke the win/loss size symmetry. avg_win shrank "
          f"{(1-agg['avg_win']/base['avg_win'])*100:.0f}% but avg_loss only shrank "
          f"{(1-agg['avg_loss']/base['avg_loss'])*100:.0f}%, AND WR collapsed "
          f"from {base['wr']:.0f}% to {agg['wr']:.0f}%. Per-trade EV flipped "
          f"from +ve to -ve, then the doubled trade count amplified the bleed.")


if __name__ == "__main__":
    main()
