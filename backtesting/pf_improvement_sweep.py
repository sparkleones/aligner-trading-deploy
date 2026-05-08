"""
PF improvement sweep on Option B (deployed).

Baseline (deployed live):
    avoid_days=[0,2], vix_floor=12, vix_ceil=25, use_v17_regime_gate=False
    Walk-forward verified across 6 rolling 6-month windows.
    21mo: WR ~42%, PF 1.94, +Rs 57.30L

Goal: PF > 2.5 with no PnL/DD degradation.

Hypotheses:
    H1: Daily drawdown circuit-breaker (max_daily_loss_pct=0.03)
    H2: Trail-stop relaxation (trail_pct_put 0.015->0.025; call 0.008->0.015)
    H3: Entry-confidence threshold tightening (call 6->7, put 5->6)
    H4: ATR-based dynamic SL (use_atr_sizing=True)

Acceptance gate (21mo):
    PF >= 2.10 AND PnL within -5% of baseline AND Max DD no worse.
    If passes -> walk-forward across 6 windows; require PnL beat in >=4/6.

Usage:
    PYTHONUNBUFFERED=1 python -u -m backtesting.pf_improvement_sweep \
        2>&1 | tee reports/oos/pf_improvement_sweep.log
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

FULL_START = dt.date(2024, 7, 1)
FULL_END = dt.date(2026, 4, 6)

WF_WINDOWS = [
    ("W1", dt.date(2024, 7, 1),  dt.date(2024, 12, 31)),
    ("W2", dt.date(2024, 10, 1), dt.date(2025, 3, 31)),
    ("W3", dt.date(2025, 1, 1),  dt.date(2025, 6, 30)),
    ("W4", dt.date(2025, 4, 1),  dt.date(2025, 9, 30)),
    ("W5", dt.date(2025, 7, 1),  dt.date(2025, 12, 31)),
    ("W6", dt.date(2025, 10, 1), dt.date(2026, 3, 31)),
]


def to_date(d):
    return dt.date.fromisoformat(d) if isinstance(d, str) else d


def metrics(trades):
    if not trades:
        return {"n": 0, "pnl": 0.0, "wr": 0.0, "pf": 0.0,
                "avg_win": 0.0, "avg_loss": 0.0, "rr": 0.0}
    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    tw = sum(t["pnl"] for t in wins)
    tl = -sum(t["pnl"] for t in losses)
    pf = tw / tl if tl > 0 else float("inf")
    aw = tw / max(1, len(wins))
    al = -tl / max(1, len(losses))
    rr = aw / abs(al) if al else 0.0
    return {
        "n": len(trades), "pnl": sum(t["pnl"] for t in trades),
        "wr": len(wins) / len(trades) * 100, "pf": pf,
        "avg_win": aw, "avg_loss": al, "rr": rr,
    }


def max_drawdown(trades, capital):
    daily = defaultdict(float)
    for t in trades:
        daily[to_date(t["date"])] += t["pnl"]
    eq = capital
    peak = capital
    mdd = 0.0
    mdd_pct = 0.0
    for d in sorted(daily.keys()):
        eq += daily[d]
        peak = max(peak, eq)
        dd = peak - eq
        if dd > mdd:
            mdd = dd
            mdd_pct = (dd / peak * 100) if peak > 0 else 0.0
    return mdd, mdd_pct


def fmt_pf(pf):
    if pf == float("inf"):
        return "  inf"
    return f"{pf:5.2f}"


def base_cfg():
    """Deployed Option B."""
    cfg = dict(V17_CONFIG)
    cfg["avoid_days"] = [0, 2]
    cfg["use_v17_regime_gate"] = False
    cfg["use_v17_monwed_gate"] = False
    cfg["vix_floor"] = 12
    cfg["vix_ceil"] = 25
    return cfg


VARIANTS = [
    # (label, description, cfg_patch)
    ("BASE", "deployed Option B", {}),
    ("H1",   "daily DD circuit-breaker -3%",
     {"max_daily_loss_pct": 0.03}),
    ("H2",   "loose trail (put 2.5%, call 1.5%)",
     {"trail_pct_put": 0.025, "trail_pct_call": 0.015}),
    ("H3",   "tighter entry scores (call 7, put 6)",
     {"call_score_min": 7.0, "put_score_min": 6.0}),
    ("H4",   "ATR-based dynamic SL (use_atr_sizing=True)",
     {"use_atr_sizing": True}),
]


def run_one(label, desc, patch, preloaded, start, end):
    cfg = base_cfg()
    cfg.update(patch)
    trades, _ = run_backtest(
        start_date=start, end_date=end,
        cfg_override=cfg, quiet=True, preloaded=preloaded,
    )
    m = metrics(trades)
    mdd, mdd_pct = max_drawdown(trades, CAPITAL)
    return {
        "label": label, "desc": desc, "cfg": cfg,
        "n": m["n"], "pnl": m["pnl"], "wr": m["wr"], "pf": m["pf"],
        "rr": m["rr"], "avg_win": m["avg_win"], "avg_loss": m["avg_loss"],
        "max_dd": mdd, "max_dd_pct": mdd_pct,
    }


def print_table(rows, title):
    print()
    print("=" * 120)
    print(f"  {title}")
    print("=" * 120)
    hdr = (f"  {'var':<5} {'desc':<46} | {'n':>4} {'PnL':>14} {'WR':>6} "
           f"{'PF':>6} {'R:R':>5} {'MaxDD':>14} {'DD%':>6}")
    print(hdr)
    print("  " + "-" * 117)
    base = rows[0]
    for r in rows:
        delta_pnl = r["pnl"] - base["pnl"]
        marker = " "
        if r["label"] != "BASE":
            if r["pf"] >= 2.10 and r["pnl"] >= base["pnl"] * 0.95 and \
               r["max_dd"] <= base["max_dd"] * 1.02:
                marker = "*"
        print(f"  {r['label']:<5} {r['desc']:<46} | "
              f"{r['n']:>4d} Rs {r['pnl']:>+11,.0f} {r['wr']:>5.1f}% "
              f"{fmt_pf(r['pf'])} {r['rr']:>5.2f} Rs {r['max_dd']:>+11,.0f} "
              f"{r['max_dd_pct']:>5.1f}% {marker}")
        if r["label"] != "BASE":
            print(f"        delta vs BASE: PnL Rs {delta_pnl:+,.0f} "
                  f"({delta_pnl/base['pnl']*100:+.1f}%)   "
                  f"PF delta {(r['pf'] if r['pf'] != float('inf') else 999) - (base['pf'] if base['pf'] != float('inf') else 999):+.2f}   "
                  f"DD delta Rs {r['max_dd']-base['max_dd']:+,.0f}")


def acceptance(base_row, cand_row):
    """Returns (passes, reasons)."""
    reasons = []
    pf_ok = cand_row["pf"] >= 2.10
    pnl_ok = cand_row["pnl"] >= base_row["pnl"] * 0.95
    dd_ok = cand_row["max_dd"] <= base_row["max_dd"] * 1.02  # 2% slack
    reasons.append(f"PF>=2.10: {cand_row['pf']:.2f} -> "
                   f"{'PASS' if pf_ok else 'FAIL'}")
    reasons.append(f"PnL within -5% of BASE: "
                   f"Rs {cand_row['pnl']:+,.0f} vs Rs {base_row['pnl']:+,.0f} -> "
                   f"{'PASS' if pnl_ok else 'FAIL'}")
    reasons.append(f"Max DD no worse: "
                   f"Rs {cand_row['max_dd']:+,.0f} vs Rs {base_row['max_dd']:+,.0f} -> "
                   f"{'PASS' if dd_ok else 'FAIL'}")
    return (pf_ok and pnl_ok and dd_ok), reasons


def walk_forward(label, desc, patch):
    print()
    print("=" * 120)
    print(f"  WALK-FORWARD VALIDATION  candidate={label} ({desc})")
    print("  per-window load (NOT shared) — windowed dates only.")
    print("=" * 120)

    base_patch = {}
    cand_patch = patch
    rows = []
    for wlabel, ws, we in WF_WINDOWS:
        print(f"\n--- {wlabel}: {ws} -> {we} ---", flush=True)
        preloaded_w = load_period_data(start_date=ws, end_date=we, quiet=True)
        rb = run_one("BASE", "deployed", base_patch, preloaded_w, ws, we)
        rc = run_one(label, desc, cand_patch, preloaded_w, ws, we)
        rows.append((wlabel, rb, rc))
        print(f"  BASE: n={rb['n']:>3d}  PnL Rs {rb['pnl']:>+11,.0f}  "
              f"WR={rb['wr']:>5.1f}%  PF={fmt_pf(rb['pf'])}  "
              f"DD Rs {rb['max_dd']:>+11,.0f}")
        print(f"  {label:<4}: n={rc['n']:>3d}  PnL Rs {rc['pnl']:>+11,.0f}  "
              f"WR={rc['wr']:>5.1f}%  PF={fmt_pf(rc['pf'])}  "
              f"DD Rs {rc['max_dd']:>+11,.0f}")
        print(f"  delta PnL: Rs {rc['pnl']-rb['pnl']:+,.0f}    "
              f"PF delta: {(rc['pf'] if rc['pf'] != float('inf') else 999) - (rb['pf'] if rb['pf'] != float('inf') else 999):+.2f}")

    pnl_wins = sum(1 for _, rb, rc in rows if rc["pnl"] > rb["pnl"])
    pf_wins = sum(
        1 for _, rb, rc in rows
        if (rc["pf"] if rc["pf"] != float("inf") else 999.0) >=
           (rb["pf"] if rb["pf"] != float("inf") else 999.0)
    )

    print()
    print("=" * 120)
    print(f"  WALK-FORWARD SUMMARY  {label}")
    print("=" * 120)
    hdr = (f"  {'win':<3} | {'BASE n':>6} {'BASE PnL':>14} {'BASE PF':>7} | "
           f"{'CAND n':>6} {'CAND PnL':>14} {'CAND PF':>7} | "
           f"{'dPnL':>14} {'dPF':>6}")
    print(hdr)
    print("  " + "-" * 116)
    for wlabel, rb, rc in rows:
        d_pnl = rc["pnl"] - rb["pnl"]
        d_pf = (rc["pf"] if rc["pf"] != float("inf") else 999.0) - \
               (rb["pf"] if rb["pf"] != float("inf") else 999.0)
        print(f"  {wlabel:<3} | {rb['n']:>6d} Rs {rb['pnl']:>+11,.0f} {fmt_pf(rb['pf']):>7} | "
              f"{rc['n']:>6d} Rs {rc['pnl']:>+11,.0f} {fmt_pf(rc['pf']):>7} | "
              f"Rs {d_pnl:>+11,.0f} {d_pf:>+6.2f}")

    print()
    print(f"  PnL wins: {pnl_wins}/6   PF wins: {pf_wins}/6")
    real_edge = pnl_wins >= 4
    print(f"  Real edge (>=4/6 PnL wins): {'YES' if real_edge else 'NO'}")
    return real_edge, pnl_wins, pf_wins


def main():
    print("=" * 120)
    print("  PF IMPROVEMENT SWEEP  Option B baseline")
    print(f"  Window: {FULL_START} -> {FULL_END}   Capital: Rs {CAPITAL:,.0f}")
    print(f"  Acceptance gate: PF>=2.10  AND  PnL >= 95% of BASE  AND  MaxDD <= BASE*1.02")
    print("=" * 120)

    print(f"\nLoading 21mo period data {FULL_START} -> {FULL_END} ...", flush=True)
    preloaded = load_period_data(start_date=FULL_START, end_date=FULL_END, quiet=True)
    _, _, all_dates, _ = preloaded
    print(f"  {len(all_dates)} trading days\n", flush=True)

    rows = []
    for label, desc, patch in VARIANTS:
        print(f"Running {label}: {desc} ...", flush=True)
        r = run_one(label, desc, patch, preloaded, FULL_START, FULL_END)
        rows.append(r)
        print(f"  -> n={r['n']}  PnL Rs {r['pnl']:+,.0f}  PF {fmt_pf(r['pf'])}  "
              f"DD Rs {r['max_dd']:+,.0f}", flush=True)

    print_table(rows, "21-MONTH SWEEP RESULTS")

    base_row = rows[0]
    print()
    print("=" * 120)
    print("  ACCEPTANCE GATE")
    print("=" * 120)
    candidates = []
    for r in rows[1:]:
        passes, reasons = acceptance(base_row, r)
        print(f"\n  {r['label']} ({r['desc']}):")
        for line in reasons:
            print(f"     {line}")
        print(f"     -> {'PASS' if passes else 'FAIL'}")
        if passes:
            candidates.append(r)

    if not candidates:
        print()
        print("=" * 120)
        print("  VERDICT: NO CANDIDATE CLEARED 21mo GATE  -> recommendation: do nothing")
        print("=" * 120)
        return

    # Pick best by PF among passers (tie-break: PnL)
    best = max(candidates, key=lambda r: (r["pf"], r["pnl"]))
    print()
    print("=" * 120)
    print(f"  BEST CANDIDATE: {best['label']} ({best['desc']})")
    print(f"     PF {best['pf']:.2f}, PnL Rs {best['pnl']:+,.0f}, DD Rs {best['max_dd']:+,.0f}")
    print("=" * 120)

    # Find original patch for the best
    patch = next(p for lab, _, p in VARIANTS if lab == best["label"])
    real_edge, pw, pfw = walk_forward(best["label"], best["desc"], patch)

    print()
    print("=" * 120)
    print("  FINAL VERDICT")
    print("=" * 120)
    if real_edge:
        print(f"  DEPLOY: {best['label']} ({best['desc']})  "
              f"-- 21mo gate PASS + WF PnL wins {pw}/6")
        print(f"  cfg patch: {patch}")
    else:
        print(f"  DO NOT DEPLOY {best['label']}: only {pw}/6 PnL wins in walk-forward.")
        print(f"  Recommendation: research further; current edge does not generalize.")


if __name__ == "__main__":
    main()
