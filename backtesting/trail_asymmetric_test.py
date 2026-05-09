"""
Asymmetric trail test: disable CE trail, keep PE trail.

Hypothesis: counter-factual analysis showed the trail-stop leak is almost
entirely on the CALL side. PE trail saved Rs 17.5K net across 4 fires;
CE trail cost Rs 331.2K net across 4 fires (would have won big at EOD).

This suggests the right fix is asymmetric: keep PE trail (it helps),
disable CE trail (it's the leak).

Variants:
  1. BASELINE        — deployed Option B
  2. PE_TRAIL_ONLY   — disable CE trail, keep PE trail (THE HYPOTHESIS)
  3. CE_TRAIL_ONLY   — disable PE trail, keep CE trail (CONTROL — should be worse)
  4. NO_TRAIL        — both disabled (for comparison)
  5. DELAYED         — both delayed (prior best)

Phase 1: 21mo sweep
Phase 2: walk-forward winner

Usage:
    python -m backtesting.trail_asymmetric_test
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


VARIANTS = {
    "BASELINE": lambda: base_cfg(),
    "PE_TRAIL_ONLY": lambda: {**base_cfg(),
        # Disable CE trail
        "trail_pct_call": 1.0,
        "min_hold_trail_call": 999,
        "monwed_trail_pct_call": 1.0,
        "monwed_min_hold_trail_call": 999,
    },
    "CE_TRAIL_ONLY": lambda: {**base_cfg(),
        # Disable PE trail
        "trail_pct_put": 1.0,
        "min_hold_trail_put": 999,
        "monwed_trail_pct_put": 1.0,
        "monwed_min_hold_trail_put": 999,
    },
    "NO_TRAIL": lambda: {**base_cfg(),
        "trail_pct_put": 1.0, "trail_pct_call": 1.0,
        "min_hold_trail_put": 999, "min_hold_trail_call": 999,
        "monwed_trail_pct_put": 1.0, "monwed_trail_pct_call": 1.0,
        "monwed_min_hold_trail_put": 999, "monwed_min_hold_trail_call": 999,
    },
    "DELAYED": lambda: {**base_cfg(),
        "min_hold_trail_put": 48, "min_hold_trail_call": 48,
        "monwed_min_hold_trail_put": 24, "monwed_min_hold_trail_call": 24,
    },
}


def run_window(cfg, start, end, preloaded):
    trades, _ = run_backtest(start_date=start, end_date=end,
                             cfg_override=cfg, quiet=True, preloaded=preloaded)
    m = metrics(trades)
    dd = equity_dd(trades, CAPITAL)
    n_trail = sum(1 for t in trades if t.get("exit_reason") == "trail_stop")
    n_ce_trail = sum(1 for t in trades if t.get("exit_reason") == "trail_stop" and t.get("opt_type") == "CE")
    n_pe_trail = sum(1 for t in trades if t.get("exit_reason") == "trail_stop" and t.get("opt_type") == "PE")
    return m, dd, trades, n_trail, n_ce_trail, n_pe_trail


def main():
    full_start = dt.date(2024, 7, 1)
    full_end = dt.date(2026, 4, 6)

    print("=" * 110)
    print("  ASYMMETRIC TRAIL TEST  (CE vs PE)")
    print("=" * 110)

    print(f"\nPHASE 1: 21-month sweep ({full_start} -> {full_end})", flush=True)
    print("-" * 110, flush=True)
    preloaded = load_period_data(start_date=full_start, end_date=full_end, quiet=True)
    print(f"  Loaded {len(preloaded[2])} trading days\n", flush=True)

    print(f"  {'variant':<16} {'n':>4} {'PnL':>14} {'PF':>5} {'WR%':>6} "
          f"{'DD':>11} {'trail':>6} {'CE_t':>5} {'PE_t':>5}")
    print("  " + "-" * 80)
    results_21mo = {}
    for name, builder in VARIANTS.items():
        cfg = builder()
        m, dd, trades, n_trail, n_ce, n_pe = run_window(cfg, full_start, full_end, preloaded)
        results_21mo[name] = (m, dd, n_trail, n_ce, n_pe)
        print(f"  {name:<16} {m['n']:>4d} Rs {m['pnl']:>+12,.0f} {fmt_pf(m['pf'])} "
              f"{m['wr']:>5.1f}% Rs {dd:>+9,.0f} {n_trail:>6d} {n_ce:>5d} {n_pe:>5d}",
              flush=True)

    # Identify best non-baseline
    best_name = None
    best_pnl = results_21mo["BASELINE"][0]["pnl"]
    for name, (m, _, _, _, _) in results_21mo.items():
        if name == "BASELINE":
            continue
        if m["pnl"] > best_pnl:
            best_pnl = m["pnl"]
            best_name = name

    if best_name is None:
        print("\n  No variant beat baseline. Stopping.")
        return

    print(f"\n  Best 21mo variant: {best_name} (+Rs {best_pnl - results_21mo['BASELINE'][0]['pnl']:,.0f} vs baseline)")

    # Walk-forward best
    print(f"\nPHASE 2: walk-forward {best_name} vs BASELINE", flush=True)
    print("-" * 110, flush=True)
    print(f"  {'win':<3} {'period':<25} | {'A_PnL':>11} {'A_PF':>5} | "
          f"{'X_PnL':>11} {'X_PF':>5} | {'dPnL':>11} {'dPF':>6}")
    print("  " + "-" * 102)
    cfg_a = VARIANTS["BASELINE"]()
    cfg_x = VARIANTS[best_name]()
    pnl_wins = pf_wins = 0
    catastrophic = []
    for label, w_start, w_end in WINDOWS:
        pre = load_period_data(start_date=w_start, end_date=w_end, quiet=True)
        ma, _, _, _, _, _ = run_window(cfg_a, w_start, w_end, pre)
        mx, _, _, _, _, _ = run_window(cfg_x, w_start, w_end, pre)
        d_pnl = mx["pnl"] - ma["pnl"]
        a_pf = ma["pf"] if ma["pf"] != float("inf") else 999
        x_pf = mx["pf"] if mx["pf"] != float("inf") else 999
        d_pf = x_pf - a_pf
        if d_pnl > 0:
            pnl_wins += 1
        if d_pf >= 0:
            pf_wins += 1
        if ma["pnl"] > 0 and (mx["pnl"] < 0 or mx["pnl"] < 0.5 * ma["pnl"]):
            catastrophic.append(label)
        period_str = f"{w_start} -> {w_end}"
        print(f"  {label:<3} {period_str:<25} | "
              f"Rs {ma['pnl']:>+9,.0f} {fmt_pf(ma['pf'])} | "
              f"Rs {mx['pnl']:>+9,.0f} {fmt_pf(mx['pf'])} | "
              f"Rs {d_pnl:>+9,.0f} {d_pf:>+6.2f}", flush=True)

    print()
    print("=" * 110)
    print(f"  WALK-FORWARD VERDICT  for {best_name}")
    print("=" * 110)
    print(f"  PnL wins: {pnl_wins}/6   (need >=4)")
    print(f"  PF wins:  {pf_wins}/6   (need >=4)")
    print(f"  Catastrophic windows: {len(catastrophic)}")
    if catastrophic:
        print(f"    {catastrophic}")
    print()
    if pnl_wins >= 4 and pf_wins >= 4 and not catastrophic:
        print(f"  VERDICT: STRONG EDGE — {best_name} clears all criteria. Deploy candidate.")
    elif not catastrophic and (pnl_wins >= 3 or pf_wins >= 4):
        print(f"  VERDICT: MARGINAL — close but not strict pass. Decision call.")
    else:
        print(f"  VERDICT: FAIL — {best_name} does not show consistent edge.")


if __name__ == "__main__":
    main()
