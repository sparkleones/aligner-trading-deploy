"""
Directional Sanity Gate experiment.

Hypothesis: V14 entries fire PUTs into rallies and CALLs into selloffs
during low-vol regimes (June 2025 forensic: 6 of 9 PUT entries into a
+3.9% rally). A simple 5-day spot trend filter should rescue these
months without significantly hurting genuinely counter-trend trades.

Implementation: post-filter trades (same pattern as gap_classifier_sweep).
Build a date -> 5-day spot return lookup. Drop trades where direction
fights the trend by more than threshold.

Variants:
  baseline    : no filter
  thr_1.0     : block PUT if 5d > +1.0%, CALL if 5d < -1.0%
  thr_1.5     : same, threshold +/-1.5% (the original hypothesis)
  thr_2.0     : same, threshold +/-2.0%
  thr_3.0     : same, threshold +/-3.0% (only block extreme)
  put_only    : only block PUT in uptrend (trend filter is asymmetric)
  call_only   : only block CALL in downtrend (control)

For best variant: walk-forward 6 windows + June 2025 forensic.

Usage:
    python -m backtesting.directional_gate_test
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

WINDOWS = [
    ("W1", dt.date(2024, 7, 1),  dt.date(2024, 12, 31)),
    ("W2", dt.date(2024, 10, 1), dt.date(2025, 3, 31)),
    ("W3", dt.date(2025, 1, 1),  dt.date(2025, 6, 30)),
    ("W4", dt.date(2025, 4, 1),  dt.date(2025, 9, 30)),
    ("W5", dt.date(2025, 7, 1),  dt.date(2025, 12, 31)),
    ("W6", dt.date(2025, 10, 1), dt.date(2026, 3, 31)),
]


def to_date(d):
    return dt.date.fromisoformat(d) if isinstance(d, str) else d


def build_trend_lookup(day_groups, lookback_days=5):
    """Return {date -> N-day spot return percent}."""
    sorted_dates = sorted(day_groups.keys())
    closes = {}
    for d in sorted_dates:
        bars = day_groups.get(d, [])
        if bars:
            closes[d] = bars[-1]["close"]
    trading_dates = sorted(closes.keys())
    out = {}
    for i, d in enumerate(trading_dates):
        if i < lookback_days:
            out[d] = 0.0
            continue
        prev_close = closes[trading_dates[i - lookback_days]]
        today_close = closes[d]
        if prev_close > 0:
            out[d] = (today_close - prev_close) / prev_close * 100.0
        else:
            out[d] = 0.0
    return out


def base_cfg():
    cfg = dict(V17_CONFIG)
    cfg["avoid_days"] = [0, 2]
    cfg["use_v17_regime_gate"] = False
    cfg["use_v17_monwed_gate"] = False
    cfg["vix_floor"] = 12
    cfg["vix_ceil"] = 25
    return cfg


def apply_directional_filter(trades, trend_lookup, *, threshold,
                             block_put_in_up=True, block_call_in_dn=True):
    """Drop trades whose direction fights the trend beyond threshold."""
    out = []
    for t in trades:
        d = to_date(t.get("date"))
        trend_5d = trend_lookup.get(d, 0.0)
        action = t.get("action", "")
        if block_put_in_up and action == "BUY_PUT" and trend_5d > threshold:
            continue
        if block_call_in_dn and action == "BUY_CALL" and trend_5d < -threshold:
            continue
        out.append(t)
    return out


def metrics(trades):
    if not trades:
        return {"n": 0, "pnl": 0.0, "wr": 0.0, "pf": 0.0,
                "n_put": 0, "n_call": 0,
                "put_pnl": 0.0, "call_pnl": 0.0}
    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    tw = sum(t["pnl"] for t in wins)
    tl = -sum(t["pnl"] for t in losses)
    pf = tw / tl if tl > 0 else float("inf")
    n_put = sum(1 for t in trades if t.get("action") == "BUY_PUT")
    n_call = sum(1 for t in trades if t.get("action") == "BUY_CALL")
    put_pnl = sum(t["pnl"] for t in trades if t.get("action") == "BUY_PUT")
    call_pnl = sum(t["pnl"] for t in trades if t.get("action") == "BUY_CALL")
    return {"n": len(trades), "pnl": sum(t["pnl"] for t in trades),
            "wr": len(wins) / len(trades) * 100, "pf": pf,
            "n_put": n_put, "n_call": n_call,
            "put_pnl": put_pnl, "call_pnl": call_pnl}


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

    print("=" * 130)
    print("  DIRECTIONAL SANITY GATE TEST")
    print("=" * 130)

    print(f"\nLoading {full_start} -> {full_end} ...", flush=True)
    preloaded = load_period_data(start_date=full_start, end_date=full_end, quiet=True)
    day_groups, _, all_dates, _ = preloaded
    trend_lookup = build_trend_lookup(day_groups, lookback_days=5)
    print(f"  {len(all_dates)} trading days, trend lookup for {len(trend_lookup)} days\n", flush=True)

    # Distribution of 5-day returns
    in_window = [trend_lookup[d] for d in all_dates if d in trend_lookup]
    if in_window:
        s = sorted(in_window)
        n = len(s)
        print(f"  5-day return distribution:")
        print(f"    pct < -3%: {sum(1 for x in s if x < -3)} ({sum(1 for x in s if x < -3)/n*100:.0f}%)")
        print(f"    pct -3 to -1%: {sum(1 for x in s if -3 <= x < -1)} ({sum(1 for x in s if -3 <= x < -1)/n*100:.0f}%)")
        print(f"    pct -1 to 1%: {sum(1 for x in s if -1 <= x <= 1)} ({sum(1 for x in s if -1 <= x <= 1)/n*100:.0f}%)")
        print(f"    pct 1 to 3%: {sum(1 for x in s if 1 < x <= 3)} ({sum(1 for x in s if 1 < x <= 3)/n*100:.0f}%)")
        print(f"    pct > 3%: {sum(1 for x in s if x > 3)} ({sum(1 for x in s if x > 3)/n*100:.0f}%)")
        print()

    # Run V17_PROD_ONLY baseline ONCE
    print("Running V17_PROD_ONLY baseline (Option B)...", flush=True)
    trades, _ = run_backtest(start_date=full_start, end_date=full_end,
                             cfg_override=base_cfg(), quiet=True, preloaded=preloaded)
    print(f"  {len(trades)} total trades\n", flush=True)

    # Variants
    variants = [
        ("baseline",    {}),
        ("thr_1.0",     dict(threshold=1.0)),
        ("thr_1.5",     dict(threshold=1.5)),
        ("thr_2.0",     dict(threshold=2.0)),
        ("thr_3.0",     dict(threshold=3.0)),
        ("put_only_1.5",dict(threshold=1.5, block_put_in_up=True, block_call_in_dn=False)),
        ("call_only_1.5",dict(threshold=1.5, block_put_in_up=False, block_call_in_dn=True)),
    ]

    print(f"  {'variant':<14} {'n':>4} {'PnL':>14} {'PF':>5} {'WR%':>6} "
          f"{'put_n':>5} {'put_pnl':>11} {'call_n':>6} {'call_pnl':>11} {'DD':>11}")
    print("  " + "-" * 100)

    results_21mo = {}
    for name, kwargs in variants:
        if name == "baseline":
            sub_trades = trades
        else:
            sub_trades = apply_directional_filter(trades, trend_lookup, **kwargs)
        m = metrics(sub_trades)
        dd = equity_dd(sub_trades, CAPITAL)
        results_21mo[name] = (m, dd, sub_trades)
        print(f"  {name:<14} {m['n']:>4d} Rs {m['pnl']:>+12,.0f} {fmt_pf(m['pf'])} "
              f"{m['wr']:>5.1f}% {m['n_put']:>5d} Rs {m['put_pnl']:>+9,.0f} "
              f"{m['n_call']:>6d} Rs {m['call_pnl']:>+9,.0f} Rs {dd:>+9,.0f}", flush=True)

    # Identify best non-baseline by PnL
    base_pnl = results_21mo["baseline"][0]["pnl"]
    best_name = None
    best_pnl = base_pnl
    for name, (m, _, _) in results_21mo.items():
        if name == "baseline":
            continue
        if m["pnl"] > best_pnl:
            best_pnl = m["pnl"]
            best_name = name

    if best_name is None:
        print("\n  No variant beat baseline. Stopping.")
        return
    print(f"\n  Best 21mo variant: {best_name} (+Rs {best_pnl - base_pnl:,.0f} vs baseline)")

    # ── June 2025 forensic ──
    print()
    print("=" * 130)
    print("  JUNE 2025 FORENSIC — did the gate rescue the bad month?")
    print("=" * 130)
    base_june = [t for t in trades if to_date(t["date"]).strftime("%Y-%m") == "2025-06"]
    print(f"  Baseline June 2025: n={len(base_june)} trades, "
          f"PnL=Rs {sum(t['pnl'] for t in base_june):+,.0f}")
    for name, (_, _, sub) in results_21mo.items():
        if name == "baseline":
            continue
        sub_june = [t for t in sub if to_date(t["date"]).strftime("%Y-%m") == "2025-06"]
        m = metrics(sub_june)
        n_dropped = len(base_june) - len(sub_june)
        print(f"  {name:<14}: n={len(sub_june)} (-{n_dropped}), "
              f"PnL=Rs {m['pnl']:+,.0f} ({'rescued' if m['pnl'] > -50000 and m['pnl'] > sum(t['pnl'] for t in base_june) else 'still negative'})")

    # ── Walk-forward best variant ──
    print()
    print("=" * 130)
    print(f"  WALK-FORWARD: {best_name}")
    print("=" * 130)
    best_kwargs = dict(variants)[best_name]
    print(f"  {'win':<3} {'period':<25} | {'A_PnL':>11} {'A_PF':>5} | "
          f"{'X_PnL':>11} {'X_PF':>5} | {'dPnL':>11} {'dPF':>6}")
    print("  " + "-" * 102)

    pnl_wins = pf_wins = 0
    catastrophic = []
    for label, w_start, w_end in WINDOWS:
        pre = load_period_data(start_date=w_start, end_date=w_end, quiet=True)
        day_groups_w, _, _, _ = pre
        trend_w = build_trend_lookup(day_groups_w, lookback_days=5)
        # Run baseline on this window
        ta, _ = run_backtest(start_date=w_start, end_date=w_end,
                             cfg_override=base_cfg(), quiet=True, preloaded=pre)
        ma = metrics(ta)
        # Apply filter
        tx = apply_directional_filter(ta, trend_w, **best_kwargs)
        mx = metrics(tx)
        d_pnl = mx["pnl"] - ma["pnl"]
        a_pf = ma["pf"] if ma["pf"] != float("inf") else 999
        x_pf = mx["pf"] if mx["pf"] != float("inf") else 999
        d_pf = x_pf - a_pf
        if d_pnl > 0:
            pnl_wins += 1
        if d_pf >= 0:
            pf_wins += 1
        if ma["pnl"] > 0 and (mx["pnl"] < 0 or mx["pnl"] < 0.5 * ma["pnl"]):
            catastrophic.append(label)
        period_str = f"{w_start} -> {w_end}"
        print(f"  {label:<3} {period_str:<25} | "
              f"Rs {ma['pnl']:>+9,.0f} {fmt_pf(ma['pf'])} | "
              f"Rs {mx['pnl']:>+9,.0f} {fmt_pf(mx['pf'])} | "
              f"Rs {d_pnl:>+9,.0f} {d_pf:>+6.2f}", flush=True)

    print()
    print("=" * 130)
    print(f"  WALK-FORWARD VERDICT for {best_name}")
    print("=" * 130)
    print(f"  PnL wins: {pnl_wins}/6   (need >=4)")
    print(f"  PF wins:  {pf_wins}/6   (need >=4)")
    print(f"  Catastrophic windows: {len(catastrophic)}")
    if pnl_wins >= 4 and pf_wins >= 4 and not catastrophic:
        print(f"\n  VERDICT: STRONG EDGE — {best_name} clears all criteria. Deploy candidate.")
    elif not catastrophic:
        print(f"\n  VERDICT: MARGINAL — close but not strict pass.")
    else:
        print(f"\n  VERDICT: FAIL — {best_name} does not show consistent edge.")


if __name__ == "__main__":
    main()
