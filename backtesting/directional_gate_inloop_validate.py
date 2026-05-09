"""
Re-validate the directional sanity gate in-loop (vs post-filter).

The directional gate was first tested via post-filter (drop trades after
the backtest runs). For correctness, the live deployment will apply the
gate at entry-time. This script tests the in-loop implementation
(via cfg["directional_gate_threshold"]) and confirms it matches the
post-filter result on 21mo + walk-forward.

If the in-loop result matches post-filter (modulo small differences from
freed-up daily trade quota), the implementation is correct and ready
for live agent integration.

Usage:
    python -m backtesting.directional_gate_inloop_validate
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


def base_cfg():
    cfg = dict(V17_CONFIG)
    cfg["avoid_days"] = [0, 2]
    cfg["use_v17_regime_gate"] = False
    cfg["use_v17_monwed_gate"] = False
    cfg["vix_floor"] = 12
    cfg["vix_ceil"] = 25
    return cfg


def gate_cfg(threshold=1.0):
    cfg = base_cfg()
    cfg["directional_gate_threshold"] = threshold
    return cfg


def metrics(trades):
    if not trades:
        return {"n": 0, "pnl": 0.0, "wr": 0.0, "pf": 0.0,
                "n_put": 0, "n_call": 0}
    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    tw = sum(t["pnl"] for t in wins)
    tl = -sum(t["pnl"] for t in losses)
    pf = tw / tl if tl > 0 else float("inf")
    n_put = sum(1 for t in trades if t.get("action") == "BUY_PUT")
    n_call = sum(1 for t in trades if t.get("action") == "BUY_CALL")
    return {"n": len(trades), "pnl": sum(t["pnl"] for t in trades),
            "wr": len(wins) / len(trades) * 100, "pf": pf,
            "n_put": n_put, "n_call": n_call}


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
    print("  DIRECTIONAL GATE IN-LOOP VALIDATION")
    print("  Re-runs the breakthrough variant via in-loop cfg flag (vs post-filter)")
    print("  Expected: same or better than post-filter result (in-loop frees quota for replacement trades)")
    print("=" * 130)

    print(f"\nLoading {full_start} -> {full_end} ...", flush=True)
    preloaded = load_period_data(start_date=full_start, end_date=full_end, quiet=True)
    print(f"  {len(preloaded[2])} trading days\n", flush=True)

    # ── 21-month: BASELINE vs IN_LOOP_GATE ──
    print("PHASE 1: 21-month full window", flush=True)
    print("-" * 130, flush=True)
    print("Running BASELINE (no gate)...", flush=True)
    ta, _ = run_backtest(start_date=full_start, end_date=full_end,
                         cfg_override=base_cfg(), quiet=True, preloaded=preloaded)
    print("Running IN_LOOP_GATE (threshold=1.0)...", flush=True)
    tg, _ = run_backtest(start_date=full_start, end_date=full_end,
                         cfg_override=gate_cfg(1.0), quiet=True, preloaded=preloaded)

    ma = metrics(ta)
    mg = metrics(tg)
    dd_a = equity_dd(ta, CAPITAL)
    dd_g = equity_dd(tg, CAPITAL)

    print()
    print("  21-month results:")
    print(f"    BASELINE:        n={ma['n']:3d}  PnL=Rs {ma['pnl']:+12,.0f}  PF={fmt_pf(ma['pf'])}  "
          f"WR={ma['wr']:.1f}%  DD=Rs {dd_a:+,.0f}  ({ma['n_put']}P / {ma['n_call']}C)")
    print(f"    IN_LOOP_GATE:    n={mg['n']:3d}  PnL=Rs {mg['pnl']:+12,.0f}  PF={fmt_pf(mg['pf'])}  "
          f"WR={mg['wr']:.1f}%  DD=Rs {dd_g:+,.0f}  ({mg['n_put']}P / {mg['n_call']}C)")
    print(f"    delta:                PnL=Rs {mg['pnl']-ma['pnl']:+,.0f}  "
          f"PF={(mg['pf'] if mg['pf']!=float('inf') else 999) - (ma['pf'] if ma['pf']!=float('inf') else 999):+.2f}  "
          f"DD=Rs {dd_g-dd_a:+,.0f}  trades={mg['n']-ma['n']:+d}")
    print()
    print("  Reference (post-filter result, from directional_gate_test.py):")
    print(f"    POST_FILTER thr_1.0: n=150  PnL=Rs +6,767,687  PF= 2.66  WR=50.0%  DD=Rs -441,672")
    print()

    # ── Walk-forward in-loop ──
    print("PHASE 2: 6-window walk-forward (in-loop gate)", flush=True)
    print("-" * 130, flush=True)
    print(f"  {'win':<3} {'period':<25} | {'A_PnL':>11} {'A_PF':>5} | "
          f"{'X_PnL':>11} {'X_PF':>5} | {'dPnL':>11} {'dPF':>6}")
    print("  " + "-" * 102)
    pnl_wins = pf_wins = 0
    catastrophic = []
    for label, w_start, w_end in WINDOWS:
        pre = load_period_data(start_date=w_start, end_date=w_end, quiet=True)
        ta_w, _ = run_backtest(start_date=w_start, end_date=w_end,
                               cfg_override=base_cfg(), quiet=True, preloaded=pre)
        tg_w, _ = run_backtest(start_date=w_start, end_date=w_end,
                               cfg_override=gate_cfg(1.0), quiet=True, preloaded=pre)
        ma_w = metrics(ta_w)
        mg_w = metrics(tg_w)
        d_pnl = mg_w["pnl"] - ma_w["pnl"]
        a_pf = ma_w["pf"] if ma_w["pf"] != float("inf") else 999
        g_pf = mg_w["pf"] if mg_w["pf"] != float("inf") else 999
        d_pf = g_pf - a_pf
        if d_pnl > 0:
            pnl_wins += 1
        if d_pf >= 0:
            pf_wins += 1
        if ma_w["pnl"] > 0 and (mg_w["pnl"] < 0 or mg_w["pnl"] < 0.5 * ma_w["pnl"]):
            catastrophic.append(label)
        period_str = f"{w_start} -> {w_end}"
        print(f"  {label:<3} {period_str:<25} | "
              f"Rs {ma_w['pnl']:>+9,.0f} {fmt_pf(ma_w['pf'])} | "
              f"Rs {mg_w['pnl']:>+9,.0f} {fmt_pf(mg_w['pf'])} | "
              f"Rs {d_pnl:>+9,.0f} {d_pf:>+6.2f}", flush=True)

    print()
    print("=" * 130)
    print("  WALK-FORWARD VERDICT  (in-loop gate)")
    print("=" * 130)
    print(f"  PnL wins: {pnl_wins}/6   (need >=4 for strict pass)")
    print(f"  PF wins:  {pf_wins}/6   (need >=4 for strict pass)")
    print(f"  Catastrophic windows: {len(catastrophic)}")
    if pnl_wins >= 4 and pf_wins >= 4 and not catastrophic:
        print(f"\n  VERDICT: STRONG EDGE — in-loop gate confirmed. Ready for live agent integration.")
    elif not catastrophic:
        print(f"\n  VERDICT: MARGINAL — close but not strict pass.")
    else:
        print(f"\n  VERDICT: FAIL — implementation may differ from post-filter.")


if __name__ == "__main__":
    main()
