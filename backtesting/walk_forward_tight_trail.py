"""
Walk-forward validation of TIGHT_TRAIL variant vs deployed Option B (baseline)
across rolling 6-month windows of the 21-month dataset.

The deciding test for whether the tighter trail provides real edge or is
concentrated in 1-2 lucky months.

Windows (6-month, 3-month step):
  W1: 2024-07-01 -> 2024-12-31
  W2: 2024-10-01 -> 2025-03-31
  W3: 2025-01-01 -> 2025-06-30
  W4: 2025-04-01 -> 2025-09-30
  W5: 2025-07-01 -> 2025-12-31
  W6: 2025-10-01 -> 2026-03-31

Configs (both share avoid_days=[0,2], vix=[12,25], regime=OFF, monwed=OFF):
  A. BASELINE (deployed Option B)
  T. TIGHT_TRAIL (candidate):
       trail_pct_put=0.005, trail_pct_call=0.003,
       min_hold_trail_put=6, min_hold_trail_call=6

Usage:
    PYTHONUNBUFFERED=1 python -u -m backtesting.walk_forward_tight_trail \
        2>&1 | tee reports/oos/walk_forward_tight_trail.log
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


def metrics(trades):
    if not trades:
        return {"n": 0, "pnl": 0.0, "wins": 0, "losses": 0,
                "wr": 0.0, "pf": 0.0, "avg_win": 0.0, "avg_loss": 0.0}
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
        daily_pnl[to_date(t["date"])] += t["pnl"]
    eq = capital
    peak = capital
    max_dd = 0.0
    max_dd_pct = 0.0
    max_dd_date = None
    for d in sorted(daily_pnl.keys()):
        eq += daily_pnl[d]
        peak = max(peak, eq)
        dd = peak - eq
        dd_pct = (dd / peak * 100) if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd
            max_dd_pct = dd_pct
            max_dd_date = d
    return max_dd, max_dd_pct, max_dd_date


def run_window(label, start, end, cfg, preloaded):
    trades, _ = run_backtest(
        start_date=start, end_date=end,
        cfg_override=cfg, quiet=True, preloaded=preloaded,
    )
    m = metrics(trades)
    max_dd, max_dd_pct, max_dd_date = equity_curve_dd(trades, CAPITAL)
    ret_x = (CAPITAL + m["pnl"]) / CAPITAL
    return {
        "label": label, "start": start, "end": end,
        "n": m["n"], "pnl": m["pnl"], "wr": m["wr"], "pf": m["pf"],
        "max_dd": max_dd, "max_dd_pct": max_dd_pct, "max_dd_date": max_dd_date,
        "ret_x": ret_x,
    }


def fmt_pf(pf):
    if pf == float("inf"):
        return " inf"
    return f"{pf:5.2f}"


def main():
    base = dict(V17_CONFIG)
    base["avoid_days"] = [0, 2]
    base["use_v17_regime_gate"] = False
    base["use_v17_monwed_gate"] = False
    base["vix_floor"] = 12
    base["vix_ceil"] = 25

    cfg_a = dict(base)
    cfg_t = dict(base)
    cfg_t["trail_pct_put"] = 0.005
    cfg_t["trail_pct_call"] = 0.003
    cfg_t["min_hold_trail_put"] = 6
    cfg_t["min_hold_trail_call"] = 6

    print("=" * 110)
    print("  WALK-FORWARD WINDOWED VALIDATION  TIGHT_TRAIL vs Baseline (Option B)")
    print(f"  Both share: avoid_days=[0,2], vix=[12,25], regime=OFF, monwed=OFF")
    print(f"  A=baseline (deployed)   T=TIGHT_TRAIL (candidate)")
    print(f"  T overrides: trail_pct_put=0.005, trail_pct_call=0.003, "
          f"min_hold_trail_put=6, min_hold_trail_call=6")
    print("=" * 110)
    print()

    results = []
    for label, start, end in WINDOWS:
        print(f"--- {label}: {start} -> {end} ---", flush=True)
        # Load preloaded PER WINDOW (shared preload has known bug).
        preloaded_w = load_period_data(start_date=start, end_date=end, quiet=True)
        ra = run_window(label, start, end, cfg_a, preloaded_w)
        rt = run_window(label, start, end, cfg_t, preloaded_w)
        results.append((label, start, end, ra, rt))
        print(f"  A: n={ra['n']:>3d}  PnL=Rs {ra['pnl']:>+11,.0f}  "
              f"WR={ra['wr']:>5.1f}%  PF={fmt_pf(ra['pf'])}  "
              f"DD=Rs {ra['max_dd']:>+10,.0f} ({ra['max_dd_pct']:.1f}%)")
        print(f"  T: n={rt['n']:>3d}  PnL=Rs {rt['pnl']:>+11,.0f}  "
              f"WR={rt['wr']:>5.1f}%  PF={fmt_pf(rt['pf'])}  "
              f"DD=Rs {rt['max_dd']:>+10,.0f} ({rt['max_dd_pct']:.1f}%)")
        d_pnl = rt["pnl"] - ra["pnl"]
        d_pf = (rt["pf"] if rt["pf"] != float("inf") else 999.0) - \
               (ra["pf"] if ra["pf"] != float("inf") else 999.0)
        d_wr = rt["wr"] - ra["wr"]
        print(f"  delta: PnL=Rs {d_pnl:+,.0f}    PF delta={d_pf:+.2f}    "
              f"WR delta={d_wr:+.1f}pp\n")

    # Per-window table
    print()
    print("=" * 130)
    print("  PER-WINDOW SUMMARY TABLE")
    print("=" * 130)
    hdr = (f"  {'win':<3} {'start':<11} {'end':<11} | "
           f"{'A_n':>4} {'A_PnL':>12} {'A_WR':>5} {'A_PF':>5} {'A_DD':>11} | "
           f"{'T_n':>4} {'T_PnL':>12} {'T_WR':>5} {'T_PF':>5} {'T_DD':>11} | "
           f"{'dPnL':>12} {'dPF':>6}")
    print(hdr)
    print("  " + "-" * 146)
    for label, start, end, ra, rt in results:
        d_pnl = rt["pnl"] - ra["pnl"]
        a_pf = ra["pf"] if ra["pf"] != float("inf") else 999.0
        t_pf = rt["pf"] if rt["pf"] != float("inf") else 999.0
        d_pf = t_pf - a_pf
        print(f"  {label:<3} {str(start):<11} {str(end):<11} | "
              f"{ra['n']:>4d} {ra['pnl']:>+12,.0f} {ra['wr']:>4.1f}% {fmt_pf(ra['pf'])} {ra['max_dd']:>+11,.0f} | "
              f"{rt['n']:>4d} {rt['pnl']:>+12,.0f} {rt['wr']:>4.1f}% {fmt_pf(rt['pf'])} {rt['max_dd']:>+11,.0f} | "
              f"{d_pnl:>+12,.0f} {d_pf:>+6.2f}")

    # Win counts
    pnl_wins = sum(1 for _, _, _, ra, rt in results if rt["pnl"] > ra["pnl"])
    pf_wins = sum(
        1 for _, _, _, ra, rt in results
        if (rt["pf"] if rt["pf"] != float("inf") else 999.0) >=
           (ra["pf"] if ra["pf"] != float("inf") else 999.0)
    )
    both_wins = sum(
        1 for _, _, _, ra, rt in results
        if rt["pnl"] > ra["pnl"] and
           (rt["pf"] if rt["pf"] != float("inf") else 999.0) >=
           (ra["pf"] if ra["pf"] != float("inf") else 999.0)
    )

    # Catastrophic check
    catastrophic = []
    for label, start, end, ra, rt in results:
        if ra["pnl"] > 0:
            if rt["pnl"] < 0:
                catastrophic.append((label, "T neg while A pos",
                                    ra["pnl"], rt["pnl"]))
            elif rt["pnl"] < 0.5 * ra["pnl"]:
                catastrophic.append((label, "T < 50% of A", ra["pnl"], rt["pnl"]))

    print()
    print("=" * 110)
    print("  WIN COUNTS  (out of 6 windows)")
    print("=" * 110)
    print(f"  T beats A on PnL:   {pnl_wins}/6")
    print(f"  T >= A on PF:       {pf_wins}/6")
    print(f"  T beats on BOTH:    {both_wins}/6")
    print()
    if catastrophic:
        print(f"  Catastrophic windows (T <50% of A or T-negative-while-A-positive):")
        for lab, why, a_pnl, t_pnl in catastrophic:
            print(f"    {lab}: {why}  A=Rs {a_pnl:+,.0f}  T=Rs {t_pnl:+,.0f}")
    else:
        print("  No catastrophic windows detected.")

    # Pass criteria
    crit1 = pnl_wins >= 4
    crit2 = pf_wins >= 4
    crit3 = len(catastrophic) == 0

    print()
    print("=" * 110)
    print("  PASS CRITERIA")
    print("=" * 110)
    print(f"  1) T beats A on PnL in >=4/6 windows:      {pnl_wins}/6   "
          f"PASS={'YES' if crit1 else 'NO'}")
    print(f"  2) T >= A on PF in >=4/6 windows:          {pf_wins}/6   "
          f"PASS={'YES' if crit2 else 'NO'}")
    print(f"  3) No catastrophic window for T:           "
          f"{len(catastrophic)} bad   "
          f"PASS={'YES' if crit3 else 'NO'}")

    # Verdict
    print()
    print("=" * 110)
    print("  VERDICT")
    print("=" * 110)

    all_pass = crit1 and crit2 and crit3

    # Check concentration (only last 2 windows winning)
    last2 = results[-2:]
    earlier = results[:-2]
    last2_wins = sum(1 for _, _, _, ra, rt in last2 if rt["pnl"] > ra["pnl"])
    earlier_wins = sum(1 for _, _, _, ra, rt in earlier if rt["pnl"] > ra["pnl"])
    concentration = (last2_wins == 2 and earlier_wins <= 1)

    if all_pass and pnl_wins >= 5:
        verdict = "STRONG EDGE"
        recommendation = "DEPLOY TIGHT_TRAIL — tighter trail consistently lifts results across regimes."
    elif all_pass:
        verdict = "STRONG EDGE"
        recommendation = "DEPLOY TIGHT_TRAIL — clears all 3 pass criteria."
    elif crit3 and (crit1 or crit2) and not concentration:
        verdict = "MARGINAL"
        recommendation = ("WAIT and monitor. T is positive but doesn't dominate A. "
                          "Run another quarter live before switching.")
    elif concentration:
        verdict = "CONCENTRATION RISK"
        recommendation = ("DO NOT DEPLOY. T's edge is concentrated in last 2 windows. "
                          "Pre-Sep windows show no consistent lift "
                          "— it's likely 1-time regime alignment.")
    else:
        verdict = "FAIL"
        recommendation = "REJECT TIGHT_TRAIL — does not show consistent edge across windows."

    print(f"  {verdict}")
    print(f"  Recommendation: {recommendation}")
    print()
    print(f"  (PnL_wins={pnl_wins}/6, PF_wins={pf_wins}/6, both={both_wins}/6, "
          f"catastrophic={len(catastrophic)}, last2_wins={last2_wins}/2, "
          f"earlier_wins={earlier_wins}/4)")
    print()


if __name__ == "__main__":
    main()
