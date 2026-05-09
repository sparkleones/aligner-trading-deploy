"""
Walk-forward of candidate (lookback=3, threshold=0.5) vs deployed (5, 1.0).

The lookback sweep showed lb=3/thr=0.5 gives +Rs 86.72L vs deployed
+Rs 79.55L on 21mo (+Rs 7L lift). Need walk-forward across 6 windows
to see if the lift generalizes or is concentrated.

Usage:
    python -m backtesting.directional_gate_lb3_thr05_walkfwd
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
    cfg["directional_gate_threshold"] = None  # disable inherited gate
    return cfg


def gate_cfg(threshold, lookback):
    cfg = base_cfg()
    cfg["directional_gate_threshold"] = threshold
    cfg["directional_gate_lookback_days"] = lookback
    return cfg


def main():
    print("=" * 110)
    print("  WALK-FORWARD: (lb=3, thr=0.50) vs (lb=5, thr=1.00)")
    print("=" * 110)

    cfg_deployed = gate_cfg(1.0, 5)
    cfg_candidate = gate_cfg(0.5, 3)

    print(f"  {'win':<3} {'period':<25} | "
          f"{'D_PnL':>11} {'D_PF':>5} | {'C_PnL':>11} {'C_PF':>5} | "
          f"{'dPnL':>11} {'dPF':>6}", flush=True)
    print("  " + "-" * 102)

    pnl_wins = pf_wins = 0
    catastrophic = []
    for label, w_start, w_end in WINDOWS:
        pre = load_period_data(start_date=w_start, end_date=w_end, quiet=True)
        td, _ = run_backtest(start_date=w_start, end_date=w_end,
                             cfg_override=cfg_deployed, quiet=True, preloaded=pre)
        tc, _ = run_backtest(start_date=w_start, end_date=w_end,
                             cfg_override=cfg_candidate, quiet=True, preloaded=pre)
        md = metrics(td)
        mc = metrics(tc)
        d_pnl = mc["pnl"] - md["pnl"]
        d_pf_v = ((mc["pf"] if mc["pf"] != float("inf") else 999)
                  - (md["pf"] if md["pf"] != float("inf") else 999))
        if d_pnl > 0:
            pnl_wins += 1
        if d_pf_v >= 0:
            pf_wins += 1
        if md["pnl"] > 0 and (mc["pnl"] < 0 or mc["pnl"] < 0.5 * md["pnl"]):
            catastrophic.append(label)
        period_str = f"{w_start} -> {w_end}"
        print(f"  {label:<3} {period_str:<25} | "
              f"Rs {md['pnl']:>+9,.0f} {fmt_pf(md['pf'])} | "
              f"Rs {mc['pnl']:>+9,.0f} {fmt_pf(mc['pf'])} | "
              f"Rs {d_pnl:>+9,.0f} {d_pf_v:>+6.2f}", flush=True)

    print()
    print("=" * 110)
    print("  VERDICT")
    print("=" * 110)
    print(f"  Candidate (lb=3, thr=0.5) vs Deployed (lb=5, thr=1.0)")
    print(f"  PnL wins: {pnl_wins}/6   PF wins: {pf_wins}/6")
    print(f"  Catastrophic windows: {len(catastrophic)}")
    if pnl_wins >= 4 and pf_wins >= 4 and not catastrophic:
        print(f"\n  STRONG: candidate beats deployed across windows. Consider switch.")
    elif not catastrophic and (pnl_wins >= 3 or pf_wins >= 4):
        print(f"\n  MARGINAL: similar to deployed. Stay with deployed (validated).")
    else:
        print(f"\n  REJECT: candidate fails to consistently beat deployed.")


if __name__ == "__main__":
    main()
