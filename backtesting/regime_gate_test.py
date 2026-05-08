"""
Regime Gate Test — does enabling `use_v17_regime_gate=True` block June 2025?

Window: 2025-04-07 -> 2026-04-06 (last 1Y)

Variants:
  A. Baseline (current deployed): avoid_days=[0,2], gate=OFF, monwed=OFF
  B. Regime gate ON, calendar OFF: avoid_days=[], gate=ON, monwed=OFF
  C. Both ON:                       avoid_days=[0,2], gate=ON, monwed=OFF
  D. Mon/Wed gate replacement:      avoid_days=[], gate=OFF, monwed=ON

All: vix_floor=12, vix_ceil=25.

Key question: did variants B/C zero out June 2025 (catastrophic vol-crush month)?

Usage:
    PYTHONUNBUFFERED=1 python -u -m backtesting.regime_gate_test 2>&1 \
        | tee reports/oos/regime_gate_test.log
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
JUNE_START = dt.date(2025, 6, 1)
JUNE_END = dt.date(2025, 6, 30)


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


def filter_window(trades, start_d, end_d):
    return [t for t in trades if start_d <= to_date(t["date"]) <= end_d]


def split(trades, cutover):
    pre, post = [], []
    for t in trades:
        d = to_date(t.get("date"))
        (pre if d < cutover else post).append(t)
    return pre, post


def exit_breakdown(trades):
    by = defaultdict(int)
    for t in trades:
        by[t.get("exit_reason", "unknown")] += 1
    return dict(by)


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


def run_one(label, cfg, preloaded, start, end):
    trades, _ = run_backtest(start_date=start, end_date=end,
                             cfg_override=cfg, quiet=True, preloaded=preloaded)
    full = metrics(trades)
    pre, post = split(trades, POST_SEP)
    pre_m = metrics(pre)
    post_m = metrics(post)
    max_dd, max_dd_pct, max_dd_date, final_eq = equity_drawdown(trades, CAPITAL)

    june_trades = filter_window(trades, JUNE_START, JUNE_END)
    june_m = metrics(june_trades)
    june_exits = exit_breakdown(june_trades)

    months = monthly(trades)
    rr = full["avg_win"] / abs(full["avg_loss"]) if full["avg_loss"] else 0

    print()
    print("=" * 96)
    print(f"  {label}")
    print(f"  cfg: avoid={cfg['avoid_days']}, gate={cfg['use_v17_regime_gate']}, "
          f"monwed={cfg['use_v17_monwed_gate']}, "
          f"vix=[{cfg['vix_floor']},{cfg['vix_ceil']}]")
    print("=" * 96)
    print(f"  Full window: n={full['n']:>3d}  PnL=Rs {full['pnl']:>+12,.0f}  "
          f"WR={full['wr']:>5.1f}%  PF={full['pf']:.2f}  "
          f"return={(CAPITAL+full['pnl'])/CAPITAL:.2f}x")
    print(f"  Pre-Sep:     n={pre_m['n']:>3d}  PnL=Rs {pre_m['pnl']:>+12,.0f}  "
          f"WR={pre_m['wr']:>5.1f}%  PF={pre_m['pf']:.2f}")
    print(f"  Post-Sep:    n={post_m['n']:>3d}  PnL=Rs {post_m['pnl']:>+12,.0f}  "
          f"WR={post_m['wr']:>5.1f}%  PF={post_m['pf']:.2f}")
    print(f"  Max DD:      Rs {max_dd:+,.0f} ({max_dd_pct:+.1f}%) on {max_dd_date}")
    print(f"  R:R = {rr:.2f}    Final eq = Rs {final_eq:,.0f}")
    print()
    print(f"  --- JUNE 2025 (catastrophic month in baseline) ---")
    print(f"  n={june_m['n']:>2d}  PnL=Rs {june_m['pnl']:>+12,.0f}  "
          f"WR={june_m['wr']:>5.1f}%  PF={june_m['pf']:.2f}")
    print(f"  exits={june_exits}")
    return {
        "label": label, "cfg": cfg, "full": full, "pre": pre_m, "post": post_m,
        "max_dd": max_dd, "max_dd_pct": max_dd_pct, "max_dd_date": max_dd_date,
        "june": june_m, "june_exits": june_exits,
        "monthly": months, "rr": rr,
    }


def main():
    end = dt.date(2026, 4, 6)
    start = dt.date(2025, 4, 7)

    print(f"Loading period data {start} -> {end} (last 1 year) ...", flush=True)
    preloaded = load_period_data(start_date=start, end_date=end, quiet=True)
    print(f"  {len(preloaded[2])} trading days\n", flush=True)

    base = dict(V17_CONFIG)
    base["vix_floor"] = 12
    base["vix_ceil"] = 25

    cfg_a = dict(base)
    cfg_a["avoid_days"] = [0, 2]
    cfg_a["use_v17_regime_gate"] = False
    cfg_a["use_v17_monwed_gate"] = False

    cfg_b = dict(base)
    cfg_b["avoid_days"] = []
    cfg_b["use_v17_regime_gate"] = True
    cfg_b["use_v17_monwed_gate"] = False

    cfg_c = dict(base)
    cfg_c["avoid_days"] = [0, 2]
    cfg_c["use_v17_regime_gate"] = True
    cfg_c["use_v17_monwed_gate"] = False

    cfg_d = dict(base)
    cfg_d["avoid_days"] = []
    cfg_d["use_v17_regime_gate"] = False
    cfg_d["use_v17_monwed_gate"] = True

    res_a = run_one("A. BASELINE (deployed)  avoid=[0,2], gate=OFF, monwed=OFF",
                    cfg_a, preloaded, start, end)
    res_b = run_one("B. REGIME GATE ONLY     avoid=[],    gate=ON,  monwed=OFF",
                    cfg_b, preloaded, start, end)
    res_c = run_one("C. BOTH ON              avoid=[0,2], gate=ON,  monwed=OFF",
                    cfg_c, preloaded, start, end)
    res_d = run_one("D. MONWED GATE ONLY     avoid=[],    gate=OFF, monwed=ON",
                    cfg_d, preloaded, start, end)

    # ── Comparison table ──
    print()
    print("=" * 110)
    print("  COMPARISON TABLE  (last 1Y window 2025-04-07 -> 2026-04-06)")
    print("=" * 110)
    header = (f"  {'variant':<8}  {'n':>4}  {'PnL':>13}  {'PF':>5}  {'MaxDD':>13}  "
              f"{'Jun_n':>5}  {'Jun_PnL':>12}  {'PostSep_PnL':>13}  {'PostSep_PF':>10}")
    print(header)
    print("  " + "-" * 106)
    for r in (res_a, res_b, res_c, res_d):
        tag = r["label"].split(".")[0].strip()
        pf = r["full"]["pf"]
        pf_str = f"{pf:>5.2f}" if pf != float("inf") else "  inf"
        post_pf = r["post"]["pf"]
        post_pf_str = f"{post_pf:>10.2f}" if post_pf != float("inf") else "       inf"
        print(f"  {tag:<8}  "
              f"{r['full']['n']:>4d}  "
              f"Rs {r['full']['pnl']:>+10,.0f}  {pf_str}  "
              f"Rs {r['max_dd']:>+10,.0f}  "
              f"{r['june']['n']:>5d}  "
              f"Rs {r['june']['pnl']:>+9,.0f}  "
              f"Rs {r['post']['pnl']:>+10,.0f}  "
              f"{post_pf_str}")

    # ── Side-by-side monthly (all 4 variants) ──
    print()
    print("=" * 110)
    print("  MONTHLY P&L  (n / PnL per variant)")
    print("=" * 110)
    months_map = {r["label"][:1]: {m["month"]: m for m in r["monthly"]}
                  for r in (res_a, res_b, res_c, res_d)}
    all_months = sorted(set().union(*[set(m.keys()) for m in months_map.values()]))
    print(f"  {'month':<8}  "
          f"{'A_n':>3} {'A_pnl':>11}  "
          f"{'B_n':>3} {'B_pnl':>11}  "
          f"{'C_n':>3} {'C_pnl':>11}  "
          f"{'D_n':>3} {'D_pnl':>11}")
    for m in all_months:
        row = f"  {m:<8}  "
        for v in ("A", "B", "C", "D"):
            mm = months_map[v].get(m)
            n = mm["n"] if mm else 0
            pnl = mm["pnl"] if mm else 0.0
            marker = ""
            if m == "2025-06":
                marker = "*"
            row += f"{n:>3d} Rs {pnl:>+8,.0f}{marker:1s}  "
        print(row)
    print("  (* = June 2025 catastrophic month in baseline)")

    # ── June detail ──
    print()
    print("=" * 110)
    print("  JUNE 2025 DETAIL  (the catastrophic month — did the gate filter it?)")
    print("=" * 110)
    for r in (res_a, res_b, res_c, res_d):
        tag = r["label"].split(".")[0].strip()
        pf = r["june"]["pf"]
        pf_str = f"{pf:.2f}" if pf != float("inf") else "inf"
        print(f"  {tag}: n={r['june']['n']:>2d}  PnL=Rs {r['june']['pnl']:>+10,.0f}  "
              f"PF={pf_str}  exits={r['june_exits']}")

    # ── Verdict ──
    print()
    print("=" * 110)
    print("  VERDICT SIGNALS")
    print("=" * 110)
    base_june = res_a["june"]["pnl"]
    base_full = res_a["full"]["pnl"]
    for r in (res_b, res_c, res_d):
        tag = r["label"].split(".")[0].strip()
        d_full = r["full"]["pnl"] - base_full
        d_june = r["june"]["pnl"] - base_june
        june_blocked = "YES" if r["june"]["n"] == 0 else (
            "PARTIAL" if r["june"]["n"] < res_a["june"]["n"] else "NO"
        )
        print(f"  {tag}: vs A  d_full=Rs {d_full:>+11,.0f}  "
              f"d_june=Rs {d_june:>+11,.0f}  "
              f"june_blocked={june_blocked}")


if __name__ == "__main__":
    main()
