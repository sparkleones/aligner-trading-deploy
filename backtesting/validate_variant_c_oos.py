"""
Walk-forward OOS validation of Variant C (avoid_days=[0,2] + use_v17_regime_gate=True)
on the pre-1Y window (2024-07-01 -> 2025-04-06).

Compares:
  A. BASELINE (deployed): avoid_days=[0,2], gate=OFF, monwed=OFF
  C. VARIANT C (candidate): avoid_days=[0,2], gate=ON, monwed=OFF

Both: vix_floor=12, vix_ceil=25.

Question: is the +Rs 16.63L last-1Y lift (vs baseline +Rs 2.78L, PF 1.99 vs 1.20)
real edge or in-sample overfitting?

Pass criteria for "REAL EDGE":
  1. Positive PnL on OOS window
  2. PF >= 1.50
  3. PnL within 60% of baseline OR better

Usage:
    PYTHONUNBUFFERED=1 python -u -m backtesting.validate_variant_c_oos \
        2>&1 | tee reports/oos/validate_variant_c_oos.log
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
        dd = peak - eq  # positive magnitude
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
    print(f"\n{'=' * 96}")
    print(f"  {label}")
    print(f"  cfg: avoid_days={cfg['avoid_days']}, "
          f"gate={cfg['use_v17_regime_gate']}, "
          f"monwed={cfg['use_v17_monwed_gate']}, "
          f"vix=[{cfg['vix_floor']},{cfg['vix_ceil']}]")
    print(f"{'=' * 96}", flush=True)

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
    rr = full["avg_win"] / abs(full["avg_loss"]) if full["avg_loss"] else 0.0

    print(f"  Window:  {START} -> {END}")
    print(f"  Trades:  n={full['n']}  ({full['wins']} wins / {full['losses']} losses)")
    print(f"  PnL:     Rs {full['pnl']:+,.0f}   return={ret_x:.2f}x")
    print(f"  WR:      {full['wr']:.1f}%   PF: {full['pf']:.2f}")
    print(f"  Avg win: Rs {full['avg_win']:+,.0f}    Avg loss: Rs {full['avg_loss']:+,.0f}    "
          f"R:R = {rr:.2f}")
    print(f"  Max DD:  Rs {max_dd:+,.0f} ({max_dd_pct:.1f}% of peak) on {max_dd_date}")
    print(f"  Max DD as % of capital: {max_dd / CAPITAL * 100:.1f}%")
    print(f"  Losing streaks (trade): {trade_runs}")
    print(f"  Losing streaks (day):   {day_runs}")
    print(f"  Negative months: {months_neg}/{len(months)}")
    print(f"\n  Monthly P&L:")
    for m in months:
        pf_str = f"{m['pf']:>5.2f}" if m['pf'] != float("inf") else "  inf"
        print(f"    {m['month']}  n={m['n']:>3d}  Rs {m['pnl']:>+12,.0f}  "
              f"WR={m['wr']:>5.1f}%  PF={pf_str}")

    return {
        "label": label, "cfg": cfg, "trades": trades, "full": full,
        "max_dd": max_dd, "max_dd_pct": max_dd_pct, "max_dd_date": max_dd_date,
        "trade_runs": trade_runs, "day_runs": day_runs,
        "monthly": months, "ret_x": ret_x, "rr": rr,
    }


def main():
    print(f"Loading period data {START} -> {END} ...", flush=True)
    preloaded = load_period_data(start_date=START, end_date=END, quiet=True)
    print(f"  {len(preloaded[2])} trading days in OOS window\n", flush=True)

    base = dict(V17_CONFIG)
    base["avoid_days"] = [0, 2]
    base["use_v17_monwed_gate"] = False
    base["vix_floor"] = 12
    base["vix_ceil"] = 25

    cfg_a = dict(base); cfg_a["use_v17_regime_gate"] = False
    cfg_c = dict(base); cfg_c["use_v17_regime_gate"] = True

    res_a = run_one("A. BASELINE (DEPLOYED)  avoid=[0,2], gate=OFF, monwed=OFF",
                    cfg_a, preloaded)
    res_c = run_one("C. VARIANT C (CANDIDATE)  avoid=[0,2], gate=ON,  monwed=OFF",
                    cfg_c, preloaded)

    # Side-by-side monthly
    print(f"\n{'=' * 96}")
    print("  MONTHLY P&L SIDE-BY-SIDE (OOS window 2024-07-01 -> 2025-04-06)")
    print(f"{'=' * 96}")
    print(f"  {'month':>8s}  | {'A:base n':>9s} {'A PnL':>12s} {'A PF':>6s}  | "
          f"{'C:gate n':>9s} {'C PnL':>12s} {'C PF':>6s}  | {'delta PnL':>12s}")
    print("  " + "-" * 92)
    months_a = {m["month"]: m for m in res_a["monthly"]}
    months_c = {m["month"]: m for m in res_c["monthly"]}
    all_months = sorted(set(months_a) | set(months_c))
    for m in all_months:
        ma = months_a.get(m, {"n": 0, "pnl": 0.0, "pf": 0.0})
        mc = months_c.get(m, {"n": 0, "pnl": 0.0, "pf": 0.0})
        dpnl = mc["pnl"] - ma["pnl"]
        pf_a = ma["pf"] if ma["pf"] != float("inf") else 999.0
        pf_c = mc["pf"] if mc["pf"] != float("inf") else 999.0
        print(f"  {m:>8s}  | {ma['n']:>9d} {ma['pnl']:>+12,.0f} {pf_a:>6.2f}  | "
              f"{mc['n']:>9d} {mc['pnl']:>+12,.0f} {pf_c:>6.2f}  | {dpnl:>+12,.0f}")

    # Side-by-side delta block
    print(f"\n{'=' * 96}")
    print("  SIDE-BY-SIDE DELTA  (C - A)  on OOS window 2024-07-01 -> 2025-04-06")
    print(f"{'=' * 96}")
    a, c = res_a["full"], res_c["full"]
    dpnl = c["pnl"] - a["pnl"]
    dpf = c["pf"] - a["pf"]
    ddd = res_c["max_dd"] - res_a["max_dd"]
    dn = c["n"] - a["n"]
    print(f"  Trades:    A={a['n']:>4d}            C={c['n']:>4d}            "
          f"delta={dn:+d} ({dn/a['n']*100 if a['n'] else 0:+.1f}%)")
    print(f"  PnL:       A=Rs {a['pnl']:>+12,.0f}    C=Rs {c['pnl']:>+12,.0f}    "
          f"delta=Rs {dpnl:>+12,.0f}")
    print(f"  Return:    A={res_a['ret_x']:.2f}x          C={res_c['ret_x']:.2f}x          "
          f"delta={res_c['ret_x']-res_a['ret_x']:+.2f}x")
    print(f"  WR:        A={a['wr']:>5.1f}%         C={c['wr']:>5.1f}%         "
          f"delta={c['wr']-a['wr']:+.1f}pp")
    print(f"  PF:        A={a['pf']:>5.2f}          C={c['pf']:>5.2f}          "
          f"delta={dpf:+.2f}")
    print(f"  Max DD:    A=Rs {res_a['max_dd']:>+12,.0f}    "
          f"C=Rs {res_c['max_dd']:>+12,.0f}    delta=Rs {ddd:>+12,.0f}")
    print(f"  DD %cap:   A={res_a['max_dd']/CAPITAL*100:>5.1f}%         "
          f"C={res_c['max_dd']/CAPITAL*100:>5.1f}%         "
          f"delta={(res_c['max_dd']-res_a['max_dd'])/CAPITAL*100:+.1f}pp")
    print(f"  R:R:       A={res_a['rr']:>5.2f}          C={res_c['rr']:>5.2f}          "
          f"delta={res_c['rr']-res_a['rr']:+.2f}")

    # Pass criteria
    print(f"\n{'=' * 96}")
    print("  PASS CRITERIA")
    print(f"{'=' * 96}")
    crit1_pos_pnl = c["pnl"] > 0
    crit2_pf = c["pf"] >= 1.50
    threshold_60pct = a["pnl"] * 0.60
    crit3_within = (c["pnl"] >= threshold_60pct) or (c["pnl"] >= a["pnl"])
    print(f"  1) Positive PnL:          C PnL=Rs {c['pnl']:+,.0f}    "
          f"PASS={'YES' if crit1_pos_pnl else 'NO'}")
    print(f"  2) PF >= 1.50:            C PF={c['pf']:.2f}            "
          f"PASS={'YES' if crit2_pf else 'NO'}")
    print(f"  3) PnL within 60% of A or better: A=Rs {a['pnl']:+,.0f}, "
          f"60%-floor=Rs {threshold_60pct:+,.0f}, C=Rs {c['pnl']:+,.0f}    "
          f"PASS={'YES' if crit3_within else 'NO'}")

    # Verdict
    print(f"\n{'=' * 96}")
    print("  VERDICT")
    print(f"{'=' * 96}")

    all_pass = crit1_pos_pnl and crit2_pf and crit3_within
    beats_baseline = c["pnl"] > a["pnl"]
    pf_negative_or_under1 = c["pnl"] < 0 or c["pf"] < 1.0

    if pf_negative_or_under1:
        verdict = "REJECT (OVERFIT)"
        rationale = (
            f"Variant C collapses on OOS: PnL=Rs {c['pnl']:+,.0f}, PF={c['pf']:.2f}. "
            f"The last-1Y +Rs 16.63L lift was tuned-to-window. Same pattern as [0,1,2]."
        )
    elif all_pass and beats_baseline:
        verdict = "REAL EDGE - DEPLOY"
        rationale = (
            f"Variant C beats baseline on OOS by Rs {dpnl:+,.0f} with PF "
            f"{a['pf']:.2f} -> {c['pf']:.2f}. The regime gate's lift is not in-sample artifact."
        )
    elif all_pass:
        verdict = "MARGINAL - MONITOR & CONSIDER"
        rationale = (
            f"Variant C is positive on OOS (PnL=Rs {c['pnl']:+,.0f}, PF={c['pf']:.2f}) "
            f"and clears all 3 floors but doesn't outright beat baseline (delta=Rs {dpnl:+,.0f}). "
            f"Consider conditional deployment with monitoring."
        )
    else:
        failed = []
        if not crit1_pos_pnl:
            failed.append(f"PnL not positive (Rs {c['pnl']:+,.0f})")
        if not crit2_pf:
            failed.append(f"PF<1.50 ({c['pf']:.2f})")
        if not crit3_within:
            failed.append(f"PnL <60% of A (Rs {c['pnl']:+,.0f} vs floor Rs {threshold_60pct:+,.0f})")
        verdict = "REJECT (OVERFIT)"
        rationale = (
            f"Variant C fails OOS criteria: {'; '.join(failed)}. "
            f"The last-1Y win was tuned-to-window."
        )

    print(f"  {verdict}: {rationale}")
    print()


if __name__ == "__main__":
    main()
