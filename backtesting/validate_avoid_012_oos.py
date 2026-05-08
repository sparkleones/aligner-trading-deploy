"""
Walk-forward OOS validation of avoid_days=[0,1,2] on the pre-1Y window.

Compares:
  - Current deployed: avoid_days=[0,2],   vix_floor=12, vix_ceil=25
  - Candidate:        avoid_days=[0,1,2], vix_floor=12, vix_ceil=25

OOS window: 2024-07-01 -> 2025-04-06 (the pre-1Y portion, never used for
the avoid_days=[0,1,2] selection — that selection was made on the last 1Y).

Usage:
    PYTHONUNBUFFERED=1 python -u -m backtesting.validate_avoid_012_oos \
        2>&1 | tee reports/oos/validate_avoid_012_oos.log
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
START = dt.date(2024, 7, 1)
END = dt.date(2025, 4, 6)


def to_date(d):
    return dt.date.fromisoformat(d) if isinstance(d, str) else d


def metrics(trades):
    if not trades:
        return {
            "n": 0, "pnl": 0.0, "wins": 0, "losses": 0, "wr": 0.0,
            "pf": 0.0, "avg_win": 0.0, "avg_loss": 0.0,
        }
    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    tw = sum(t["pnl"] for t in wins)
    tl = -sum(t["pnl"] for t in losses)
    pf = tw / tl if tl > 0 else float("inf")
    return {
        "n": len(trades),
        "pnl": sum(t["pnl"] for t in trades),
        "wins": len(wins),
        "losses": len(losses),
        "wr": len(wins) / len(trades) * 100,
        "pf": pf,
        "avg_win": tw / max(1, len(wins)),
        "avg_loss": -tl / max(1, len(losses)),
    }


def equity_curve_dd(trades, capital):
    """Day-by-day cumulative equity, returns max DD info."""
    daily_pnl = defaultdict(float)
    for t in trades:
        d = to_date(t["date"])
        daily_pnl[d] += t["pnl"]
    eq = capital
    peak = capital
    max_dd = 0.0
    max_dd_pct = 0.0
    max_dd_date = None
    for d in sorted(daily_pnl.keys()):
        eq += daily_pnl[d]
        peak = max(peak, eq)
        dd = peak - eq  # positive = drawdown magnitude
        dd_pct = (dd / peak * 100) if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd
            max_dd_pct = dd_pct
            max_dd_date = d
    return max_dd, max_dd_pct, max_dd_date


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

    daily_pnl = defaultdict(float)
    for t in sorted_t:
        daily_pnl[to_date(t["date"])] += t["pnl"]
    day_runs = []
    cur_d = 0
    for d in sorted(daily_pnl.keys()):
        if daily_pnl[d] < 0:
            cur_d += 1
        else:
            if cur_d > 0:
                day_runs.append(cur_d)
            cur_d = 0
    if cur_d > 0:
        day_runs.append(cur_d)
    return sorted(runs, reverse=True)[:5], sorted(day_runs, reverse=True)[:5]


def monthly(trades):
    by_month = defaultdict(list)
    for t in trades:
        d = to_date(t["date"])
        by_month[d.strftime("%Y-%m")].append(t)
    rows = []
    for m in sorted(by_month.keys()):
        ts = by_month[m]
        mw = [t for t in ts if t["pnl"] > 0]
        ml = [t for t in ts if t["pnl"] <= 0]
        tw = sum(t["pnl"] for t in mw)
        tl = -sum(t["pnl"] for t in ml)
        pf = tw / tl if tl > 0 else float("inf")
        rows.append({
            "month": m,
            "n": len(ts),
            "pnl": sum(t["pnl"] for t in ts),
            "wr": len(mw) / len(ts) * 100 if ts else 0,
            "pf": pf,
        })
    return rows


def run_one(label, cfg, preloaded):
    print(f"\n{'=' * 90}")
    print(f"  {label}")
    print(f"  cfg: avoid_days={cfg['avoid_days']}, vix_floor={cfg['vix_floor']}, "
          f"vix_ceil={cfg['vix_ceil']}")
    print(f"{'=' * 90}", flush=True)

    trades, _ = run_backtest(
        start_date=START, end_date=END,
        cfg_override=cfg, quiet=True, preloaded=preloaded,
    )

    full = metrics(trades)
    max_dd, max_dd_pct, max_dd_date = equity_curve_dd(trades, CAPITAL)
    trade_runs, day_runs = losing_streaks(trades)
    months = monthly(trades)
    months_neg = sum(1 for m in months if m["pnl"] < 0)
    ret_x = (CAPITAL + full["pnl"]) / CAPITAL

    print(f"  Window:  {START} -> {END}")
    print(f"  Trades:  n={full['n']}  ({full['wins']} wins / {full['losses']} losses)")
    print(f"  PnL:     Rs {full['pnl']:+,.0f}   return={ret_x:.2f}x")
    print(f"  WR:      {full['wr']:.1f}%   PF: {full['pf']:.2f}")
    print(f"  Avg win: Rs {full['avg_win']:+,.0f}    Avg loss: Rs {full['avg_loss']:+,.0f}    "
          f"R:R = {full['avg_win']/abs(full['avg_loss']) if full['avg_loss'] else 0:.2f}")
    print(f"  Max DD:  Rs {max_dd:+,.0f} ({max_dd_pct:.1f}% of peak) on {max_dd_date}")
    print(f"  Max DD as % of capital: {max_dd / CAPITAL * 100:.1f}%")
    print(f"  Losing streaks (trade): {trade_runs}")
    print(f"  Losing streaks (day):   {day_runs}")
    print(f"  Negative months: {months_neg}/{len(months)}")
    print(f"\n  Monthly P&L:")
    for m in months:
        print(f"    {m['month']}  n={m['n']:>3d}  Rs {m['pnl']:>+12,.0f}  "
              f"WR={m['wr']:>5.1f}%  PF={m['pf']:>5.2f}")

    return {
        "label": label, "cfg": cfg, "trades": trades, "full": full,
        "max_dd": max_dd, "max_dd_pct": max_dd_pct, "max_dd_date": max_dd_date,
        "trade_runs": trade_runs, "day_runs": day_runs,
        "monthly": months, "ret_x": ret_x,
    }


def main():
    print(f"Loading period data {START} -> {END} ...", flush=True)
    preloaded = load_period_data(start_date=START, end_date=END, quiet=True)
    print(f"  {len(preloaded[2])} trading days in OOS window\n", flush=True)

    base = dict(V17_CONFIG)
    base["use_v17_regime_gate"] = False
    base["use_v17_monwed_gate"] = False
    base["vix_floor"] = 12
    base["vix_ceil"] = 25

    cfg_a = dict(base); cfg_a["avoid_days"] = [0, 2]
    cfg_b = dict(base); cfg_b["avoid_days"] = [0, 1, 2]

    res_a = run_one("CURRENT DEPLOYED  avoid_days=[0,2]   (Mon+Wed blocked)", cfg_a, preloaded)
    res_b = run_one("CANDIDATE         avoid_days=[0,1,2] (Mon+Tue+Wed blocked)", cfg_b, preloaded)

    # Side-by-side monthly
    print(f"\n{'=' * 90}")
    print("  MONTHLY P&L SIDE-BY-SIDE (OOS window)")
    print(f"{'=' * 90}")
    print(f"  {'month':>8s}  | {'A:[0,2] n':>10s} {'A PnL':>12s} {'A PF':>6s}  | "
          f"{'B:[0,1,2] n':>11s} {'B PnL':>12s} {'B PF':>6s}  | {'delta PnL':>12s}")
    print("  " + "-" * 96)
    months_a = {m["month"]: m for m in res_a["monthly"]}
    months_b = {m["month"]: m for m in res_b["monthly"]}
    all_months = sorted(set(months_a) | set(months_b))
    for m in all_months:
        ma = months_a.get(m, {"n": 0, "pnl": 0.0, "pf": 0.0})
        mb = months_b.get(m, {"n": 0, "pnl": 0.0, "pf": 0.0})
        dpnl = mb["pnl"] - ma["pnl"]
        pf_a = ma["pf"] if ma["pf"] != float("inf") else 999.0
        pf_b = mb["pf"] if mb["pf"] != float("inf") else 999.0
        print(f"  {m:>8s}  | {ma['n']:>10d} {ma['pnl']:>+12,.0f} {pf_a:>6.2f}  | "
              f"{mb['n']:>11d} {mb['pnl']:>+12,.0f} {pf_b:>6.2f}  | {dpnl:>+12,.0f}")

    # Side-by-side delta block
    print(f"\n{'=' * 90}")
    print("  SIDE-BY-SIDE DELTA  (B - A)  on OOS window 2024-07-01 -> 2025-04-06")
    print(f"{'=' * 90}")
    a, b = res_a["full"], res_b["full"]
    dpnl = b["pnl"] - a["pnl"]
    dpf = b["pf"] - a["pf"]
    ddd = res_b["max_dd"] - res_a["max_dd"]
    dn = b["n"] - a["n"]
    print(f"  Trades:    A={a['n']:>4d}            B={b['n']:>4d}            "
          f"delta={dn:+d} ({dn/a['n']*100 if a['n'] else 0:+.1f}%)")
    print(f"  PnL:       A=Rs {a['pnl']:>+12,.0f}    B=Rs {b['pnl']:>+12,.0f}    "
          f"delta=Rs {dpnl:>+12,.0f}")
    print(f"  Return:    A={res_a['ret_x']:.2f}x          B={res_b['ret_x']:.2f}x          "
          f"delta={res_b['ret_x']-res_a['ret_x']:+.2f}x")
    print(f"  WR:        A={a['wr']:>5.1f}%         B={b['wr']:>5.1f}%         "
          f"delta={b['wr']-a['wr']:+.1f}pp")
    print(f"  PF:        A={a['pf']:>5.2f}          B={b['pf']:>5.2f}          "
          f"delta={dpf:+.2f}")
    print(f"  Max DD:    A=Rs {res_a['max_dd']:>+12,.0f}    "
          f"B=Rs {res_b['max_dd']:>+12,.0f}    delta=Rs {ddd:>+12,.0f}")
    print(f"  DD %cap:   A={res_a['max_dd']/CAPITAL*100:>5.1f}%         "
          f"B={res_b['max_dd']/CAPITAL*100:>5.1f}%         "
          f"delta={(res_b['max_dd']-res_a['max_dd'])/CAPITAL*100:+.1f}pp")

    # Verdict
    print(f"\n{'=' * 90}")
    print("  VERDICT")
    print(f"{'=' * 90}")
    # Real edge if B's PF > A's PF AND B's PnL > A's PnL on OOS.
    pnl_better = dpnl > 0
    pf_better = b["pf"] > a["pf"]
    if pnl_better and pf_better:
        verdict = "REAL EDGE"
        rationale = (
            f"Candidate [0,1,2] beats deployed [0,2] on OOS: PnL +Rs {dpnl:+,.0f}, "
            f"PF {a['pf']:.2f} -> {b['pf']:.2f}; the last-1Y outperformance is not in-sample artifact."
        )
    else:
        verdict = "OVERFIT"
        reasons = []
        if not pnl_better:
            reasons.append(f"PnL worse on OOS (delta=Rs {dpnl:+,.0f})")
        if not pf_better:
            reasons.append(f"PF worse on OOS ({a['pf']:.2f} -> {b['pf']:.2f})")
        rationale = (
            f"Candidate [0,1,2] fails to outperform deployed [0,2] on OOS: "
            f"{'; '.join(reasons)}. Last-1Y win was tuned-to-window."
        )
    print(f"  {verdict}: {rationale}")
    print()


if __name__ == "__main__":
    main()
