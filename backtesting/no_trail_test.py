"""
NO_TRAIL test: disable the trail-stop mechanism entirely.

Hypothesis: 5 of 8 baseline trail fires were false alarms costing Rs 313K
net. Removing the trail entirely (so trades exit only via EOD/time/SL/TP)
should recover that loss.

Risk: 3 of 8 trail fires correctly cut worsening losses. Without trail,
those 3 trades would have done WORSE. The Rs 313K lift is the NET — we
keep the savings on false alarms but lose the savings on correct cuts.

Tests both:
  1. 21-month full-window comparison (BASELINE vs NO_TRAIL)
  2. 6-window walk-forward validation (same windows used for prior rejections)

Output: pass/fail verdict using the same gate as before:
  - PnL > BASELINE on 4+ of 6 walk-forward windows
  - PF >= BASELINE on 4+ of 6
  - No catastrophic windows

Usage:
    python -m backtesting.no_trail_test
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
        return {"n": 0, "pnl": 0.0, "wr": 0.0, "pf": 0.0}
    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    tw = sum(t["pnl"] for t in wins)
    tl = -sum(t["pnl"] for t in losses)
    pf = tw / tl if tl > 0 else float("inf")
    return {"n": len(trades), "pnl": sum(t["pnl"] for t in trades),
            "wr": len(wins) / len(trades) * 100, "pf": pf}


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


def base_cfg():
    cfg = dict(V17_CONFIG)
    cfg["avoid_days"] = [0, 2]
    cfg["use_v17_regime_gate"] = False
    cfg["use_v17_monwed_gate"] = False
    cfg["vix_floor"] = 12
    cfg["vix_ceil"] = 25
    return cfg


def no_trail_cfg():
    cfg = base_cfg()
    # Disable trail: set trail_pct extremely high so it never engages,
    # AND set min_hold extremely high so trail never activates.
    cfg["trail_pct_put"] = 1.0      # 100% — premium would have to swing 100%
    cfg["trail_pct_call"] = 1.0
    cfg["min_hold_trail_put"] = 999
    cfg["min_hold_trail_call"] = 999
    cfg["monwed_trail_pct_put"] = 1.0
    cfg["monwed_trail_pct_call"] = 1.0
    cfg["monwed_min_hold_trail_put"] = 999
    cfg["monwed_min_hold_trail_call"] = 999
    return cfg


def run_window(cfg, start, end, preloaded):
    trades, _ = run_backtest(start_date=start, end_date=end,
                             cfg_override=cfg, quiet=True, preloaded=preloaded)
    m = metrics(trades)
    dd = equity_dd(trades, CAPITAL)
    return m, dd, trades


def main():
    full_start = dt.date(2024, 7, 1)
    full_end = dt.date(2026, 4, 6)

    print("=" * 110)
    print("  NO_TRAIL TEST")
    print("  A=BASELINE (deployed Option B)   N=NO_TRAIL (trail disabled)")
    print("=" * 110)
    print()

    # ── Phase 1: 21-month full window ──
    print(f"PHASE 1: 21-month full window ({full_start} -> {full_end})", flush=True)
    print("-" * 110, flush=True)
    preloaded = load_period_data(start_date=full_start, end_date=full_end, quiet=True)
    print(f"  Loaded {len(preloaded[2])} trading days\n", flush=True)

    cfg_a = base_cfg()
    cfg_n = no_trail_cfg()
    ma, dd_a, ta = run_window(cfg_a, full_start, full_end, preloaded)
    mn, dd_n, tn = run_window(cfg_n, full_start, full_end, preloaded)

    # Count trail-stop exits in each
    a_trail = sum(1 for t in ta if t.get("exit_reason") == "trail_stop")
    n_trail = sum(1 for t in tn if t.get("exit_reason") == "trail_stop")
    print(f"  BASELINE:  n={ma['n']:3d}  PnL=Rs {ma['pnl']:+12,.0f}  PF={fmt_pf(ma['pf'])}  "
          f"WR={ma['wr']:5.1f}%  DD=Rs {dd_a:+10,.0f}  trail_stops={a_trail}")
    print(f"  NO_TRAIL:  n={mn['n']:3d}  PnL=Rs {mn['pnl']:+12,.0f}  PF={fmt_pf(mn['pf'])}  "
          f"WR={mn['wr']:5.1f}%  DD=Rs {dd_n:+10,.0f}  trail_stops={n_trail}")
    print(f"  delta:     PnL=Rs {mn['pnl']-ma['pnl']:+,.0f}    "
          f"PF delta={(mn['pf'] if mn['pf']!=float('inf') else 999)-(ma['pf'] if ma['pf']!=float('inf') else 999):+.2f}    "
          f"DD delta=Rs {dd_n-dd_a:+,.0f}")
    print()

    if mn["pnl"] <= ma["pnl"]:
        print(f"  21mo: NO_TRAIL did NOT beat baseline. Aborting walk-forward.")
        return

    # ── Phase 2: walk-forward ──
    print(f"PHASE 2: 6-window walk-forward validation", flush=True)
    print("-" * 110, flush=True)
    print(f"  {'win':<3} {'period':<25} | {'A_PnL':>11} {'A_PF':>5} | "
          f"{'N_PnL':>11} {'N_PF':>5} | {'dPnL':>11} {'dPF':>6}")
    print("  " + "-" * 106)
    pnl_wins = pf_wins = 0
    catastrophic = []
    for label, w_start, w_end in WINDOWS:
        pre = load_period_data(start_date=w_start, end_date=w_end, quiet=True)
        ma_w, dd_a_w, _ = run_window(cfg_a, w_start, w_end, pre)
        mn_w, dd_n_w, _ = run_window(cfg_n, w_start, w_end, pre)
        d_pnl = mn_w["pnl"] - ma_w["pnl"]
        a_pf = ma_w["pf"] if ma_w["pf"] != float("inf") else 999
        n_pf = mn_w["pf"] if mn_w["pf"] != float("inf") else 999
        d_pf = n_pf - a_pf
        if d_pnl > 0:
            pnl_wins += 1
        if d_pf >= 0:
            pf_wins += 1
        # Catastrophic: N's PnL < 50% of A's when A positive, OR N negative when A positive
        if ma_w["pnl"] > 0:
            if mn_w["pnl"] < 0:
                catastrophic.append((label, "N negative while A positive"))
            elif mn_w["pnl"] < 0.5 * ma_w["pnl"]:
                catastrophic.append((label, "N < 50% of A"))
        period_str = f"{w_start} -> {w_end}"
        print(f"  {label:<3} {period_str:<25} | "
              f"Rs {ma_w['pnl']:>+9,.0f} {fmt_pf(ma_w['pf'])} | "
              f"Rs {mn_w['pnl']:>+9,.0f} {fmt_pf(mn_w['pf'])} | "
              f"Rs {d_pnl:>+9,.0f} {d_pf:>+6.2f}")
    print()
    print("=" * 110)
    print("  WALK-FORWARD VERDICT")
    print("=" * 110)
    print(f"  PnL wins: {pnl_wins}/6   (need >=4)")
    print(f"  PF wins:  {pf_wins}/6   (need >=4)")
    print(f"  Catastrophic windows: {len(catastrophic)}   (need 0)")
    if catastrophic:
        for lab, why in catastrophic:
            print(f"    {lab}: {why}")
    print()
    crit1 = pnl_wins >= 4
    crit2 = pf_wins >= 4
    crit3 = len(catastrophic) == 0
    if crit1 and crit2 and crit3:
        print(f"  VERDICT: PASS — NO_TRAIL is a real edge. Deploy candidate.")
    elif crit3 and (pnl_wins >= 3 or pf_wins >= 3):
        print(f"  VERDICT: MARGINAL — close to passing but doesn't dominate. Wait/monitor.")
    else:
        print(f"  VERDICT: FAIL — does not show consistent edge across windows. Reject.")


if __name__ == "__main__":
    main()
