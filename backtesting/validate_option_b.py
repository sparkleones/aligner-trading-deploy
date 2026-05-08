"""
Side-by-side validation of Option A (deployed) vs Option B (proposed).

Option A:  avoid=[0,2], vix_floor=12, vix_ceil=35  (in production)
Option B:  avoid=[0,2], vix_floor=12, vix_ceil=25  (proposed tightening)

Reports for each:
  - Full window + Post-Sep PnL, WR, PF
  - Monthly P&L breakdown
  - Max drawdown, longest losing streak (trades + days)
  - Equity curve points (saved to JSON for dashboard later)

Usage:
    python -m backtesting.validate_option_b
"""
import sys
import json
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


def equity_curve(trades, capital):
    """Day-by-day cumulative equity."""
    daily_pnl = defaultdict(float)
    for t in trades:
        d = to_date(t["date"])
        daily_pnl[d] += t["pnl"]
    curve = []
    eq = capital
    peak = capital
    max_dd = 0.0
    max_dd_pct = 0.0
    max_dd_date = None
    for d in sorted(daily_pnl.keys()):
        eq += daily_pnl[d]
        peak = max(peak, eq)
        dd = eq - peak
        dd_pct = (dd / peak * 100) if peak > 0 else 0
        if dd < max_dd:
            max_dd = dd
            max_dd_pct = dd_pct
            max_dd_date = d
        curve.append({"date": str(d), "equity": eq, "peak": peak,
                      "dd": dd, "dd_pct": dd_pct})
    return curve, max_dd, max_dd_pct, max_dd_date


def losing_streaks(trades):
    if not trades:
        return [], []
    sorted_t = sorted(trades, key=lambda t: to_date(t["date"]))
    # Trade-level streaks
    cur, max_run, runs = 0, 0, []
    for t in sorted_t:
        if t["pnl"] <= 0:
            cur += 1
            max_run = max(max_run, cur)
        else:
            if cur > 0:
                runs.append(cur)
            cur = 0
    if cur > 0:
        runs.append(cur)
    # Day-level streaks
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
            "month": m, "n": len(ts), "pnl": sum(t["pnl"] for t in ts),
            "wr": len(mw) / len(ts) * 100 if ts else 0,
            "pf": pf,
        })
    return rows


def split(trades, cutover):
    pre, post = [], []
    for t in trades:
        d = to_date(t.get("date"))
        (pre if d < cutover else post).append(t)
    return pre, post


def run_one(label, cfg, preloaded, start, end):
    print(f"\n{'=' * 90}")
    print(f"  {label}")
    print(f"  cfg: avoid={cfg['avoid_days']}, floor={cfg['vix_floor']}, "
          f"ceil={cfg['vix_ceil']}")
    print(f"{'=' * 90}")
    trades, _ = run_backtest(start_date=start, end_date=end,
                             cfg_override=cfg, quiet=True, preloaded=preloaded)

    full = metrics(trades)
    _, post = split(trades, POST_SEP)
    ps = metrics(post)
    curve, max_dd, max_dd_pct, max_dd_date = equity_curve(trades, CAPITAL)
    trade_runs, day_runs = losing_streaks(trades)
    months = monthly(trades)
    months_neg = sum(1 for m in months if m["pnl"] < 0)

    print(f"  Full window: n={full['n']}  PnL=Rs {full['pnl']:+,.0f}  "
          f"WR={full['wr']:.1f}%  PF={full['pf']:.2f}  return={(CAPITAL+full['pnl'])/CAPITAL:.2f}x")
    print(f"  Post-Sep:    n={ps['n']}  PnL=Rs {ps['pnl']:+,.0f}  "
          f"WR={ps['wr']:.1f}%  PF={ps['pf']:.2f}")
    print(f"  Avg win Rs {full['avg_win']:+,.0f}    Avg loss Rs {full['avg_loss']:+,.0f}    "
          f"R:R = {full['avg_win']/abs(full['avg_loss']) if full['avg_loss'] else 0:.2f}")
    print(f"  Max DD: Rs {max_dd:+,.0f} ({max_dd_pct:+.1f}%) on {max_dd_date}")
    print(f"  Losing streaks (trade): {trade_runs}")
    print(f"  Losing streaks (day):   {day_runs}")
    print(f"  Negative months: {months_neg}/{len(months)}")
    print(f"\n  Monthly P&L:")
    for m in months:
        print(f"    {m['month']}  n={m['n']:>3d}  Rs {m['pnl']:>+12,.0f}  "
              f"WR={m['wr']:>5.1f}%  PF={m['pf']:>5.2f}")
    return {
        "label": label, "cfg": cfg, "trades": trades, "full": full, "post": ps,
        "curve": curve, "max_dd": max_dd, "max_dd_pct": max_dd_pct,
        "max_dd_date": str(max_dd_date) if max_dd_date else None,
        "trade_runs": trade_runs, "day_runs": day_runs,
        "monthly": months,
    }


def main():
    start = dt.date(2024, 7, 1)
    end = dt.date(2026, 4, 6)

    print(f"Loading period data {start} -> {end} ...", flush=True)
    preloaded = load_period_data(start_date=start, end_date=end, quiet=True)
    print(f"  {len(preloaded[2])} trading days\n", flush=True)

    base = dict(V17_CONFIG)
    base["avoid_days"] = [0, 2]
    base["use_v17_regime_gate"] = False
    base["use_v17_monwed_gate"] = False
    base["vix_floor"] = 12

    cfg_a = dict(base); cfg_a["vix_ceil"] = 35
    cfg_b = dict(base); cfg_b["vix_ceil"] = 25

    res_a = run_one("OPTION A (DEPLOYED)  vix_ceil=35", cfg_a, preloaded, start, end)
    res_b = run_one("OPTION B (PROPOSED)  vix_ceil=25", cfg_b, preloaded, start, end)

    # Side-by-side
    print(f"\n{'=' * 90}")
    print("  SIDE-BY-SIDE (proposed - deployed)")
    print(f"{'=' * 90}")
    delta_full = res_b["full"]["pnl"] - res_a["full"]["pnl"]
    delta_post = res_b["post"]["pnl"] - res_a["post"]["pnl"]
    delta_dd = res_b["max_dd"] - res_a["max_dd"]
    print(f"  Full PnL:    A=Rs {res_a['full']['pnl']:>+12,.0f}    "
          f"B=Rs {res_b['full']['pnl']:>+12,.0f}    delta=Rs {delta_full:>+12,.0f}")
    print(f"  Post-Sep:    A=Rs {res_a['post']['pnl']:>+12,.0f}    "
          f"B=Rs {res_b['post']['pnl']:>+12,.0f}    delta=Rs {delta_post:>+12,.0f}")
    print(f"  Full PF:     A={res_a['full']['pf']:.2f}             "
          f"B={res_b['full']['pf']:.2f}             delta={res_b['full']['pf']-res_a['full']['pf']:+.2f}")
    print(f"  Post-Sep PF: A={res_a['post']['pf']:.2f}             "
          f"B={res_b['post']['pf']:.2f}             delta={res_b['post']['pf']-res_a['post']['pf']:+.2f}")
    print(f"  Max DD:      A=Rs {res_a['max_dd']:>+12,.0f}    "
          f"B=Rs {res_b['max_dd']:>+12,.0f}    delta=Rs {delta_dd:>+12,.0f}")
    print(f"  Trades:      A={res_a['full']['n']}            B={res_b['full']['n']}            "
          f"delta={res_b['full']['n']-res_a['full']['n']:+d} ({(res_b['full']['n']-res_a['full']['n'])/res_a['full']['n']*100:+.1f}%)")

    # Save JSON for dashboard
    out_path = Path("reports/oos/option_b_validation.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        # strip trades (large) — keep only metrics + curves
        for r in [res_a, res_b]:
            r.pop("trades", None)
            r.pop("cfg", None)
        json.dump({"option_a": res_a, "option_b": res_b}, f, indent=2, default=str)
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
