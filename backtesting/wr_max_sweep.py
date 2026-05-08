"""
WR-maximization parameter sweep on Option B deployed config.

Goal: empirically demonstrate the WR-vs-PF trade-off curve.
Question: can WR be pushed to 90%+ by tightening exits? What does it cost?

Base config: V17_CONFIG with avoid_days=[0,2], vix_floor=12, vix_ceil=25
             V17 regime + monwed gates DISABLED (matches deployed Option B)

Variants:
    1. BASELINE         — deployed config, no exit-knob changes
    2. TIGHT_TRAIL      — trail_pct put 0.005 / call 0.003, min_hold_trail 6/6
    3. EARLY_PROFIT     — trail_min_profit_pct = 0.0
    4. AGGRESSIVE_STALE — stale_exit_bars=3, stale_exit_pct=0.001 (use_stale_exit ON)
    5. MAX_WR           — combine 2+3+4
    6. WIDE_TRAIL       — control: trail_pct put 0.025 / call 0.020 (let winners run)

Usage:
    PYTHONUNBUFFERED=1 python -u -m backtesting.wr_max_sweep
"""
import sys
import datetime as dt
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backtesting.v14_unified_backtest import load_period_data, run_backtest
from scoring.config import V17_CONFIG


CAPITAL = 2_00_000


def make_base():
    cfg = dict(V17_CONFIG)
    # Match deployed Option B
    cfg["avoid_days"] = [0, 2]
    cfg["use_v17_regime_gate"] = False
    cfg["use_v17_monwed_gate"] = False
    cfg["vix_floor"] = 12
    cfg["vix_ceil"] = 25
    return cfg


def variant_baseline():
    return make_base()


def variant_tight_trail():
    cfg = make_base()
    cfg["trail_pct_put"] = 0.005
    cfg["trail_pct_call"] = 0.003
    cfg["min_hold_trail_put"] = 6
    cfg["min_hold_trail_call"] = 6
    return cfg


def variant_early_profit():
    cfg = make_base()
    cfg["trail_min_profit_pct"] = 0.0
    return cfg


def variant_aggressive_stale():
    cfg = make_base()
    cfg["use_stale_exit"] = True
    cfg["stale_exit_bars"] = 3
    cfg["stale_exit_pct"] = 0.001
    return cfg


def variant_max_wr():
    cfg = make_base()
    # tight trail
    cfg["trail_pct_put"] = 0.005
    cfg["trail_pct_call"] = 0.003
    cfg["min_hold_trail_put"] = 6
    cfg["min_hold_trail_call"] = 6
    # early profit
    cfg["trail_min_profit_pct"] = 0.0
    # aggressive stale
    cfg["use_stale_exit"] = True
    cfg["stale_exit_bars"] = 3
    cfg["stale_exit_pct"] = 0.001
    return cfg


def variant_wide_trail():
    cfg = make_base()
    cfg["trail_pct_put"] = 0.025
    cfg["trail_pct_call"] = 0.020
    return cfg


VARIANTS = [
    ("BASELINE",         variant_baseline),
    ("TIGHT_TRAIL",      variant_tight_trail),
    ("EARLY_PROFIT",     variant_early_profit),
    ("AGGRESSIVE_STALE", variant_aggressive_stale),
    ("MAX_WR",           variant_max_wr),
    ("WIDE_TRAIL",       variant_wide_trail),
]


def to_date(d):
    return dt.date.fromisoformat(d) if isinstance(d, str) else d


def compute_metrics(trades):
    if not trades:
        return {"n": 0, "pnl": 0.0, "wr": 0.0, "pf": 0.0,
                "avg_win": 0.0, "avg_loss": 0.0,
                "max_dd": 0.0, "rr": 0.0}
    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    tw = sum(t["pnl"] for t in wins)
    tl = -sum(t["pnl"] for t in losses)
    pf = (tw / tl) if tl > 0 else float("inf")
    avg_win = tw / max(1, len(wins))
    avg_loss = -tl / max(1, len(losses))
    rr = (avg_win / abs(avg_loss)) if avg_loss != 0 else 0.0

    # Daily-equity Max DD
    from collections import defaultdict
    daily = defaultdict(float)
    for t in trades:
        daily[to_date(t["date"])] += t["pnl"]
    eq = CAPITAL
    peak = CAPITAL
    max_dd = 0.0
    for d in sorted(daily.keys()):
        eq += daily[d]
        peak = max(peak, eq)
        dd = eq - peak
        if dd < max_dd:
            max_dd = dd

    return {
        "n": len(trades),
        "pnl": sum(t["pnl"] for t in trades),
        "wr": len(wins) / len(trades) * 100,
        "pf": pf,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "rr": rr,
        "max_dd": max_dd,
    }


def main():
    start = dt.date(2024, 7, 1)
    end = dt.date(2026, 4, 6)

    print(f"Loading period data {start} -> {end} ...", flush=True)
    preloaded = load_period_data(start_date=start, end_date=end, quiet=True)
    print(f"  {len(preloaded[2])} trading days\n", flush=True)

    results = []
    for label, builder in VARIANTS:
        cfg = builder()
        print(f"Running {label} ...", flush=True)
        trades, _ = run_backtest(
            start_date=start, end_date=end,
            cfg_override=cfg, quiet=True,
            preloaded=preloaded,
        )
        m = compute_metrics(trades)
        ret_x = (CAPITAL + m["pnl"]) / CAPITAL
        m["label"] = label
        m["ret_x"] = ret_x
        results.append(m)
        pf_str = f"{m['pf']:.2f}" if m["pf"] != float("inf") else "inf"
        print(
            f"  [{label:18s}] n={m['n']:3d}  WR={m['wr']:5.1f}%  PF={pf_str:>5s}  "
            f"PnL=Rs {m['pnl']:>+12,.0f}  ({ret_x:5.2f}x)  "
            f"DD=Rs {m['max_dd']:>+11,.0f}  "
            f"R:R={m['rr']:.2f}",
            flush=True,
        )

    print()
    print("=" * 120)
    print("RANKED BY WR (descending)")
    print("=" * 120)
    print(f"  {'variant':18s}  {'n':>3s}  {'WR':>6s}  {'PF':>6s}  "
          f"{'PnL':>13s}  {'ret_x':>6s}  {'MaxDD':>13s}  {'avg_win':>10s}  {'avg_loss':>10s}  {'R:R':>5s}")
    print("  " + "-" * 116)
    for r in sorted(results, key=lambda x: -x["wr"]):
        pf_str = f"{r['pf']:.2f}" if r["pf"] != float("inf") else "inf"
        print(
            f"  {r['label']:18s}  {r['n']:>3d}  {r['wr']:>5.1f}%  {pf_str:>6s}  "
            f"Rs {r['pnl']:>+10,.0f}  {r['ret_x']:>5.2f}x  "
            f"Rs {r['max_dd']:>+10,.0f}  "
            f"Rs {r['avg_win']:>+7,.0f}  Rs {r['avg_loss']:>+7,.0f}  {r['rr']:>4.2f}"
        )

    # Comparison vs BASELINE
    base = next(r for r in results if r["label"] == "BASELINE")
    print()
    print("=" * 120)
    print("DELTA vs BASELINE")
    print("=" * 120)
    print(f"  {'variant':18s}  {'dWR':>7s}  {'dPF':>7s}  {'dPnL':>15s}  {'dDD':>15s}  {'dN':>5s}")
    print("  " + "-" * 116)
    for r in sorted(results, key=lambda x: -x["wr"]):
        if r["label"] == "BASELINE":
            continue
        d_wr = r["wr"] - base["wr"]
        d_pf = (r["pf"] - base["pf"]) if (r["pf"] != float("inf") and base["pf"] != float("inf")) else float("nan")
        d_pnl = r["pnl"] - base["pnl"]
        d_dd = r["max_dd"] - base["max_dd"]
        d_n = r["n"] - base["n"]
        print(
            f"  {r['label']:18s}  {d_wr:>+6.1f}%  {d_pf:>+6.2f}  "
            f"Rs {d_pnl:>+12,.0f}  Rs {d_dd:>+12,.0f}  {d_n:>+5d}"
        )

    # Verdict
    print()
    print("=" * 120)
    print("VERDICT")
    print("=" * 120)
    max_wr_r = max(results, key=lambda x: x["wr"])
    pf_str = f"{max_wr_r['pf']:.2f}" if max_wr_r["pf"] != float("inf") else "inf"
    print(f"  Highest WR: {max_wr_r['label']}  WR={max_wr_r['wr']:.1f}%  PF={pf_str}  "
          f"PnL=Rs {max_wr_r['pnl']:+,.0f}")
    above_70 = [r for r in results if r["wr"] > 70]
    above_90 = [r for r in results if r["wr"] > 90]
    if above_90:
        print(f"  WR > 90% achieved by: {[r['label'] for r in above_90]}")
    elif above_70:
        print(f"  WR > 70% achieved by: {[r['label'] for r in above_70]}  -- but no variant > 90%")
    else:
        print(f"  No variant achieved WR > 70%. Realistic ceiling for this strategy class "
              f"is {max_wr_r['wr']:.1f}% (variant {max_wr_r['label']}).")


if __name__ == "__main__":
    main()
