"""
Side-by-side: Option A (vix_ceil=35) vs Option B (vix_ceil=25) on the LAST 1 YEAR.

Window: 2025-04-07 -> 2026-04-06 (last 1 year of available data)

Goal: confirm whether Option B is genuinely better than Option A in the recent
regime, or whether the lift was purely from pre-2025-04 data in the 21-month
validation.

Both configs:
  base = V17_CONFIG with avoid_days=[0,2], use_v17_regime_gate=False,
         use_v17_monwed_gate=False, vix_floor=12
  Option A: vix_ceil=35
  Option B: vix_ceil=25

Usage:
    PYTHONUNBUFFERED=1 python -u -m backtesting.last_year_a_vs_b 2>&1 \
        | tee reports/oos/last_year_a_vs_b.log
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
POST_SEP = dt.date(2025, 9, 1)


def to_date(d):
    return dt.date.fromisoformat(d) if isinstance(d, str) else d


def metrics(trades):
    if not trades:
        return {"n": 0, "pnl": 0.0, "wr": 0.0, "pf": 0.0,
                "avg_win": 0.0, "avg_loss": 0.0}
    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    tw = sum(t["pnl"] for t in wins)
    tl = -sum(t["pnl"] for t in losses)
    pf = tw / tl if tl > 0 else float("inf")
    return {
        "n": len(trades), "pnl": sum(t["pnl"] for t in trades),
        "wr": len(wins) / len(trades) * 100, "pf": pf,
        "avg_win": tw / max(1, len(wins)),
        "avg_loss": -tl / max(1, len(losses)),
    }


def equity_drawdown(trades, capital):
    daily = defaultdict(float)
    for t in trades:
        daily[to_date(t["date"])] += t["pnl"]
    eq = capital
    peak = capital
    max_dd = 0.0
    max_dd_pct = 0.0
    max_dd_date = None
    for d in sorted(daily.keys()):
        eq += daily[d]
        peak = max(peak, eq)
        dd = eq - peak
        dd_pct = (dd / peak * 100) if peak > 0 else 0
        if dd < max_dd:
            max_dd = dd
            max_dd_pct = dd_pct
            max_dd_date = d
    return max_dd, max_dd_pct, max_dd_date, eq


def losing_streaks(trades):
    if not trades:
        return [], []
    sorted_t = sorted(trades, key=lambda t: to_date(t["date"]))
    cur, runs = 0, []
    for t in sorted_t:
        if t["pnl"] <= 0:
            cur += 1
        else:
            if cur > 0:
                runs.append(cur)
            cur = 0
    if cur > 0:
        runs.append(cur)
    daily = defaultdict(float)
    for t in sorted_t:
        daily[to_date(t["date"])] += t["pnl"]
    day_runs = []
    cur_d = 0
    for d in sorted(daily.keys()):
        if daily[d] < 0:
            cur_d += 1
        else:
            if cur_d > 0:
                day_runs.append(cur_d)
            cur_d = 0
    if cur_d > 0:
        day_runs.append(cur_d)
    return sorted(runs, reverse=True)[:5], sorted(day_runs, reverse=True)[:5]


def monthly(trades):
    bm = defaultdict(list)
    for t in trades:
        bm[to_date(t["date"]).strftime("%Y-%m")].append(t)
    rows = []
    for m in sorted(bm.keys()):
        ts = bm[m]
        wins = [t for t in ts if t["pnl"] > 0]
        tw = sum(t["pnl"] for t in wins)
        tl = -sum(t["pnl"] for t in ts if t["pnl"] <= 0)
        pf = tw / tl if tl > 0 else float("inf")
        rows.append({
            "month": m, "n": len(ts), "pnl": sum(t["pnl"] for t in ts),
            "wr": len(wins) / len(ts) * 100, "pf": pf,
        })
    return rows


def split(trades, cutover):
    pre, post = [], []
    for t in trades:
        d = to_date(t.get("date"))
        (pre if d < cutover else post).append(t)
    return pre, post


def run_one(label, cfg, preloaded, start, end):
    trades, _ = run_backtest(start_date=start, end_date=end,
                             cfg_override=cfg, quiet=True, preloaded=preloaded)
    full = metrics(trades)
    pre, post = split(trades, POST_SEP)
    pre_m = metrics(pre)
    post_m = metrics(post)
    max_dd, max_dd_pct, max_dd_date, final_eq = equity_drawdown(trades, CAPITAL)
    trade_runs, day_runs = losing_streaks(trades)
    months = monthly(trades)
    rr = full["avg_win"] / abs(full["avg_loss"]) if full["avg_loss"] else 0

    print()
    print("=" * 96)
    print(f"  {label}")
    print(f"  cfg: avoid={cfg['avoid_days']}, vix_floor={cfg['vix_floor']}, "
          f"vix_ceil={cfg['vix_ceil']}")
    print("=" * 96)
    print(f"  Full window: n={full['n']:>3d}  PnL=Rs {full['pnl']:>+12,.0f}  "
          f"WR={full['wr']:>5.1f}%  PF={full['pf']:.2f}  "
          f"return={(CAPITAL+full['pnl'])/CAPITAL:.2f}x")
    print(f"  Pre-Sep:     n={pre_m['n']:>3d}  PnL=Rs {pre_m['pnl']:>+12,.0f}  "
          f"WR={pre_m['wr']:>5.1f}%  PF={pre_m['pf']:.2f}")
    print(f"  Post-Sep:    n={post_m['n']:>3d}  PnL=Rs {post_m['pnl']:>+12,.0f}  "
          f"WR={post_m['wr']:>5.1f}%  PF={post_m['pf']:.2f}")
    print(f"  Avg win Rs {full['avg_win']:+,.0f}    "
          f"Avg loss Rs {full['avg_loss']:+,.0f}    R:R = {rr:.2f}")
    print(f"  Max DD: Rs {max_dd:+,.0f} ({max_dd_pct:+.1f}%) on {max_dd_date}")
    print(f"  Final equity: Rs {final_eq:,.0f}")
    print(f"  Losing streaks (trade): {trade_runs}")
    print(f"  Losing streaks (day):   {day_runs}")
    return {
        "label": label, "cfg": cfg, "full": full, "pre": pre_m, "post": post_m,
        "max_dd": max_dd, "max_dd_pct": max_dd_pct,
        "max_dd_date": max_dd_date, "trade_runs": trade_runs,
        "day_runs": day_runs, "monthly": months, "rr": rr,
    }


def main():
    end = dt.date(2026, 4, 6)
    start = end - dt.timedelta(days=365)

    print(f"Loading period data {start} -> {end} (last 1 year) ...", flush=True)
    preloaded = load_period_data(start_date=start, end_date=end, quiet=True)
    print(f"  {len(preloaded[2])} trading days\n", flush=True)

    base = dict(V17_CONFIG)
    base["avoid_days"] = [0, 2]
    base["use_v17_regime_gate"] = False
    base["use_v17_monwed_gate"] = False
    base["vix_floor"] = 12

    cfg_a = dict(base); cfg_a["vix_ceil"] = 35
    cfg_b = dict(base); cfg_b["vix_ceil"] = 25

    res_a = run_one("OPTION A  vix_ceil=35  (previously deployed)",
                    cfg_a, preloaded, start, end)
    res_b = run_one("OPTION B  vix_ceil=25  (currently deployed)",
                    cfg_b, preloaded, start, end)

    # ---- Side-by-side deltas ----
    print()
    print("=" * 96)
    print("  SIDE-BY-SIDE DELTAS  (B - A;  positive = B better)")
    print("=" * 96)
    d_full_pnl = res_b["full"]["pnl"] - res_a["full"]["pnl"]
    d_post_pnl = res_b["post"]["pnl"] - res_a["post"]["pnl"]
    d_pre_pnl = res_b["pre"]["pnl"] - res_a["pre"]["pnl"]
    d_full_pf = res_b["full"]["pf"] - res_a["full"]["pf"]
    d_post_pf = res_b["post"]["pf"] - res_a["post"]["pf"]
    d_dd = res_b["max_dd"] - res_a["max_dd"]
    d_n = res_b["full"]["n"] - res_a["full"]["n"]
    d_rr = res_b["rr"] - res_a["rr"]

    print(f"  Full PnL:    A=Rs {res_a['full']['pnl']:>+12,.0f}  "
          f"B=Rs {res_b['full']['pnl']:>+12,.0f}  delta=Rs {d_full_pnl:>+12,.0f}")
    print(f"  Pre-Sep PnL: A=Rs {res_a['pre']['pnl']:>+12,.0f}  "
          f"B=Rs {res_b['pre']['pnl']:>+12,.0f}  delta=Rs {d_pre_pnl:>+12,.0f}")
    print(f"  Post-Sep PnL:A=Rs {res_a['post']['pnl']:>+12,.0f}  "
          f"B=Rs {res_b['post']['pnl']:>+12,.0f}  delta=Rs {d_post_pnl:>+12,.0f}")
    print(f"  Full PF:     A={res_a['full']['pf']:.2f}             "
          f"B={res_b['full']['pf']:.2f}             delta={d_full_pf:+.2f}")
    print(f"  Post-Sep PF: A={res_a['post']['pf']:.2f}             "
          f"B={res_b['post']['pf']:.2f}             delta={d_post_pf:+.2f}")
    print(f"  Max DD:      A=Rs {res_a['max_dd']:>+12,.0f}  "
          f"B=Rs {res_b['max_dd']:>+12,.0f}  delta=Rs {d_dd:>+12,.0f}")
    print(f"  Max DD %:    A={res_a['max_dd_pct']:+.1f}%        "
          f"B={res_b['max_dd_pct']:+.1f}%        "
          f"delta={res_b['max_dd_pct']-res_a['max_dd_pct']:+.1f}%")
    print(f"  Trades:      A={res_a['full']['n']}              "
          f"B={res_b['full']['n']}              delta={d_n:+d}")
    print(f"  R:R:         A={res_a['rr']:.2f}             "
          f"B={res_b['rr']:.2f}             delta={d_rr:+.2f}")
    print(f"  Longest losing streak (trades): A={res_a['trade_runs'][0] if res_a['trade_runs'] else 0}  "
          f"B={res_b['trade_runs'][0] if res_b['trade_runs'] else 0}")
    print(f"  Longest losing streak (days):   A={res_a['day_runs'][0] if res_a['day_runs'] else 0}  "
          f"B={res_b['day_runs'][0] if res_b['day_runs'] else 0}")

    # ---- Side-by-side monthly P&L ----
    print()
    print("=" * 96)
    print("  MONTHLY P&L  (Option A  |  Option B  |  delta = B - A)")
    print("=" * 96)
    months_a = {m["month"]: m for m in res_a["monthly"]}
    months_b = {m["month"]: m for m in res_b["monthly"]}
    all_months = sorted(set(months_a.keys()) | set(months_b.keys()))

    print(f"  {'month':<8}  {'A_n':>4}  {'A_pnl':>13}  {'A_pf':>5}  | "
          f" {'B_n':>4}  {'B_pnl':>13}  {'B_pf':>5}  | "
          f" {'d_n':>4}  {'d_pnl':>13}")
    for m in all_months:
        a = months_a.get(m)
        b = months_b.get(m)
        a_n = a["n"] if a else 0
        a_pnl = a["pnl"] if a else 0.0
        a_pf = a["pf"] if a else 0.0
        b_n = b["n"] if b else 0
        b_pnl = b["pnl"] if b else 0.0
        b_pf = b["pf"] if b else 0.0
        marker = ""
        if abs(b_pnl - a_pnl) > 50000:
            marker = "  <<"
        a_pf_str = f"{a_pf:>5.2f}" if a_pf != float("inf") else "  inf"
        b_pf_str = f"{b_pf:>5.2f}" if b_pf != float("inf") else "  inf"
        print(f"  {m:<8}  {a_n:>4d}  Rs {a_pnl:>+11,.0f}  {a_pf_str}  | "
              f" {b_n:>4d}  Rs {b_pnl:>+11,.0f}  {b_pf_str}  | "
              f" {b_n-a_n:>+4d}  Rs {b_pnl-a_pnl:>+11,.0f}{marker}")

    # ---- Verdict ----
    print()
    print("=" * 96)
    print("  VERDICT (last 1Y, 2025-04-07 -> 2026-04-06)")
    print("=" * 96)
    better_pnl = "B" if res_b["full"]["pnl"] > res_a["full"]["pnl"] else "A"
    better_post = "B" if res_b["post"]["pnl"] > res_a["post"]["pnl"] else "A"
    better_pf = "B" if res_b["full"]["pf"] > res_a["full"]["pf"] else "A"
    better_dd = "B" if res_b["max_dd"] > res_a["max_dd"] else "A"  # less negative
    print(f"  Better full PnL:    {better_pnl}")
    print(f"  Better post-Sep:    {better_post}")
    print(f"  Better PF (full):   {better_pf}")
    print(f"  Better Max DD:      {better_dd}")


if __name__ == "__main__":
    main()
