"""
Deep-dive forensic analysis of June 2025 — the worst month in the last-1Y
backtest of V17_PROD_ONLY (Option B) config.

Goal: classify the month as (a) regime break, (b) bad luck, or (c) bug.

Usage:
    python -m backtesting.june_2025_deep_dive
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
DOW_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


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


def month_stats(day_groups, vix_lookup, year, month):
    """Compute spot OHLC and VIX summary for a given month."""
    days = sorted(d for d in day_groups.keys() if d.year == year and d.month == month)
    if not days:
        return None

    daily_ranges = []         # (high - low) / open * 100
    daily_pct_chgs = []        # (close - open) / open * 100
    abs_pct_chgs = []
    trending = 0
    choppy = 0
    vix_vals = []

    first_open = None
    last_close = None

    for d in days:
        bars = day_groups[d]
        if not bars:
            continue
        day_open = bars[0]["open"]
        day_close = bars[-1]["close"]
        day_high = max(b["high"] for b in bars)
        day_low = min(b["low"] for b in bars)

        if first_open is None:
            first_open = day_open
        last_close = day_close

        rng_pct = (day_high - day_low) / day_open * 100.0
        chg_pct = (day_close - day_open) / day_open * 100.0
        daily_ranges.append(rng_pct)
        daily_pct_chgs.append(chg_pct)
        abs_pct_chgs.append(abs(chg_pct))

        if abs(chg_pct) > 0.30:
            trending += 1
        else:
            choppy += 1

        v = vix_lookup.get(d)
        if v is not None:
            vix_vals.append(v)

    daily_ranges_s = sorted(daily_ranges)
    vix_s = sorted(vix_vals)
    n = len(daily_ranges_s)
    nv = len(vix_s)

    return {
        "n_days": len(days),
        "first_open": first_open,
        "last_close": last_close,
        "net_pct": (last_close - first_open) / first_open * 100.0 if first_open else 0.0,
        "range_mean": sum(daily_ranges) / n if n else 0.0,
        "range_med":  daily_ranges_s[n // 2] if n else 0.0,
        "abs_chg_mean": sum(abs_pct_chgs) / n if n else 0.0,
        "trending_days": trending,
        "choppy_days":   choppy,
        "vix_mean": sum(vix_vals) / nv if nv else 0.0,
        "vix_med":  vix_s[nv // 2] if nv else 0.0,
        "vix_min":  min(vix_vals) if vix_vals else 0.0,
        "vix_max":  max(vix_vals) if vix_vals else 0.0,
    }


def main():
    start = dt.date(2025, 5, 1)
    end   = dt.date(2025, 7, 31)

    print("Loading period data 2025-05-01 -> 2025-07-31 ...", flush=True)
    preloaded = load_period_data(start_date=start, end_date=end, quiet=True)
    day_groups, vix_lookup, all_dates, _ = preloaded
    print(f"  {len(all_dates)} trading days\n", flush=True)

    # Option B config (deployed) — V17_PROD_ONLY
    cfg = dict(V17_CONFIG)
    cfg["avoid_days"] = [0, 2]
    cfg["use_v17_regime_gate"] = False
    cfg["use_v17_monwed_gate"] = False
    cfg["vix_floor"] = 12
    cfg["vix_ceil"] = 25

    print("Running V17_PROD_ONLY (Option B) on May-Jul 2025 ...", flush=True)
    trades, _ = run_backtest(start_date=start, end_date=end,
                             cfg_override=cfg, quiet=True, preloaded=preloaded)
    print(f"  {len(trades)} trades total\n", flush=True)

    # Bucket by month
    by_month = defaultdict(list)
    for t in trades:
        d = to_date(t["date"])
        by_month[(d.year, d.month)].append(t)

    print("=" * 100)
    print("MONTHLY SUMMARY")
    print("=" * 100)
    print(f"{'month':10s} {'n':>4s} {'P&L':>14s} {'WR%':>7s} {'PF':>6s} "
          f"{'avg_win':>11s} {'avg_loss':>11s}")
    for ym in sorted(by_month):
        m = metrics(by_month[ym])
        print(f"  {ym[0]}-{ym[1]:02d} {m['n']:>4d} {m['pnl']:>+14,.0f} "
              f"{m['wr']:>6.1f}% {m['pf']:>6.2f} "
              f"{m['avg_win']:>+11,.0f} {m['avg_loss']:>+11,.0f}")
    print()

    june = by_month.get((2025, 6), [])
    may = by_month.get((2025, 5), [])
    july = by_month.get((2025, 7), [])

    # ── PER-TRADE TABLE FOR JUNE ──
    print("=" * 130)
    print(f"JUNE 2025 — TRADE-BY-TRADE  (n={len(june)})")
    print("=" * 130)
    hdr = (f"{'date':12s} {'DOW':3s} {'entry_type':18s} {'act':5s} "
           f"{'strike':>7s} {'opt':3s} {'spot_in':>8s} {'spot_out':>8s} "
           f"{'prem_in':>7s} {'prem_out':>8s} {'exit_reason':22s} "
           f"{'P&L':>11s} {'bars':>4s} {'VIX':>6s} {'dte':>3s}")
    print(hdr)
    print("-" * 130)
    for t in sorted(june, key=lambda x: to_date(x["date"])):
        d = to_date(t["date"])
        v = vix_lookup.get(d, 0.0)
        print(f"{str(d):12s} {DOW_NAMES[d.weekday()]:3s} "
              f"{str(t.get('entry_type',''))[:18]:18s} {t['action']:5s} "
              f"{t['strike']:>7.0f} {t['opt_type']:3s} "
              f"{t['entry_spot']:>8.1f} {t['exit_spot']:>8.1f} "
              f"{t['entry_premium']:>7.2f} {t['exit_premium']:>8.2f} "
              f"{str(t['exit_reason'])[:22]:22s} "
              f"{t['pnl']:>+11,.0f} {t['bars_held']:>4d} "
              f"{v:>6.2f} {float(t.get('dte_at_entry', 0)):>3.0f}")
    print()

    # ── DISTRIBUTIONS ──
    if june:
        m_j = metrics(june)
        print("=" * 100)
        print("JUNE 2025 — DISTRIBUTIONS")
        print("=" * 100)
        print(f"  P&L: Rs {m_j['pnl']:+,.0f}   WR={m_j['wr']:.1f}%   PF={m_j['pf']:.2f}")
        print(f"  Avg win:  Rs {m_j['avg_win']:+,.0f}    Avg loss: Rs {m_j['avg_loss']:+,.0f}")
        print(f"  Max win:  Rs {m_j['max_win']:+,.0f}    Max loss: Rs {m_j['max_loss']:+,.0f}")

        ex_n = Counter(t["exit_reason"] for t in june)
        ex_p = defaultdict(float)
        for t in june:
            ex_p[t["exit_reason"]] += t["pnl"]
        print("  Exit reasons:")
        for r, n in ex_n.most_common():
            print(f"    {r:25s} n={n:2d}  pnl=Rs {ex_p[r]:+,.0f}")

        dow_n = Counter(to_date(t["date"]).weekday() for t in june)
        dow_p = defaultdict(float)
        for t in june:
            dow_p[to_date(t["date"]).weekday()] += t["pnl"]
        print("  DOW (only Tue/Thu/Fri expected — avoid_days=[0,2]):")
        for i in range(5):
            if dow_n[i]:
                print(f"    {DOW_NAMES[i]} n={dow_n[i]:2d}  pnl=Rs {dow_p[i]:+,.0f}")

        et_n = Counter(t.get("entry_type") for t in june)
        et_p = defaultdict(float)
        for t in june:
            et_p[t.get("entry_type")] += t["pnl"]
        print("  Entry types:")
        for et, n in et_n.most_common():
            print(f"    {str(et):25s} n={n:2d}  pnl=Rs {et_p[et]:+,.0f}")

        vix_at_entry = [vix_lookup.get(to_date(t["date"]), 0.0) for t in june]
        dtes = [float(t.get("dte_at_entry", 0)) for t in june]
        print(f"  Mean VIX at entry: {sum(vix_at_entry)/len(vix_at_entry):.2f}  "
              f"(min={min(vix_at_entry):.2f}, max={max(vix_at_entry):.2f})")
        print(f"  Mean dte_at_entry: {sum(dtes)/len(dtes):.2f}  "
              f"(min={min(dtes)}, max={max(dtes)})")
        print()

    # ── MARKET-STATE COMPARISON: May vs Jun vs Jul ──
    print("=" * 100)
    print("MARKET STATE: May 2025 vs June 2025 vs July 2025")
    print("=" * 100)
    months = [
        ("May 2025", 2025, 5),
        ("Jun 2025", 2025, 6),
        ("Jul 2025", 2025, 7),
    ]
    print(f"{'month':12s} {'days':>5s} {'first_open':>11s} {'last_close':>11s} "
          f"{'net%':>7s} {'rng_mean%':>10s} {'rng_med%':>9s} {'abs_chg%':>9s} "
          f"{'trend':>6s} {'chop':>5s} {'VIX_mean':>9s} {'VIX_med':>8s} "
          f"{'VIX_min':>8s} {'VIX_max':>8s}")
    for label, y, m in months:
        s = month_stats(day_groups, vix_lookup, y, m)
        if not s:
            continue
        print(f"  {label:10s} {s['n_days']:>5d} {s['first_open']:>11.1f} "
              f"{s['last_close']:>11.1f} {s['net_pct']:>+7.2f} "
              f"{s['range_mean']:>10.2f} {s['range_med']:>9.2f} "
              f"{s['abs_chg_mean']:>9.2f} {s['trending_days']:>6d} "
              f"{s['choppy_days']:>5d} {s['vix_mean']:>9.2f} "
              f"{s['vix_med']:>8.2f} {s['vix_min']:>8.2f} {s['vix_max']:>8.2f}")
    print()
    print("Notes:")
    print("  - 'trend day' = abs(close-open)/open > 0.30%; else 'chop'")
    print("  - rng_% = (high-low)/open * 100 per day")
    print()

    # ── PNL SUMMARY ROW for context ──
    print("=" * 100)
    print("ADJACENT-MONTH P&L COMPARISON")
    print("=" * 100)
    for label, ts in [("May 2025", may), ("Jun 2025", june), ("Jul 2025", july)]:
        mm = metrics(ts)
        print(f"  {label}: n={mm['n']:>3d}  P&L=Rs {mm['pnl']:>+12,.0f}  "
              f"WR={mm['wr']:>5.1f}%  PF={mm['pf']:>5.2f}  "
              f"max_win=Rs {mm['max_win']:>+10,.0f}  max_loss=Rs {mm['max_loss']:>+10,.0f}")
    print()

    print("Done.")


if __name__ == "__main__":
    main()
