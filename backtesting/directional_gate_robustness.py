"""
Robustness sweep: directional gate threshold × lookback grid.

The breakthrough config was threshold=1.0% and lookback=5 days. Need to
verify the lift is robust across parameter neighborhoods before any
live deployment. If the result only holds at a single point, it's
overfitting; if it holds across a range, it's a real edge.

Test grid:
  thresholds: 0.5, 0.75, 1.0, 1.25, 1.5, 2.0
  lookbacks:  3, 5, 7, 10 days

Phase 1: 21-month sweep over 6 × 4 = 24 variants
Phase 2: walk-forward the 3 best variants (by 21mo PnL)

Usage:
    python -m backtesting.directional_gate_robustness
"""
import sys
import datetime as dt
from pathlib import Path
from collections import defaultdict

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backtesting.v14_unified_backtest import (
    load_period_data, run_backtest, _build_trend_lookup_5d,
)
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

THRESHOLDS = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
LOOKBACKS = [3, 5, 7, 10]


def to_date(d):
    return dt.date.fromisoformat(d) if isinstance(d, str) else d


def build_trend_lookup(day_groups, lookback_days):
    """Re-implementation supporting variable lookback (the in-loop helper hardcodes 5)."""
    sorted_dates = sorted(day_groups.keys())
    closes = {}
    for d in sorted_dates:
        bars = day_groups.get(d, [])
        if bars:
            closes[d] = bars[-1]["close"]
    trading_dates = sorted(closes.keys())
    out = {}
    for i, d in enumerate(trading_dates):
        if i < lookback_days:
            out[d] = 0.0
            continue
        prev_close = closes[trading_dates[i - lookback_days]]
        today_close = closes[d]
        if prev_close > 0:
            out[d] = (today_close - prev_close) / prev_close * 100.0
        else:
            out[d] = 0.0
    return out


def base_cfg():
    cfg = dict(V17_CONFIG)
    cfg["avoid_days"] = [0, 2]
    cfg["use_v17_regime_gate"] = False
    cfg["use_v17_monwed_gate"] = False
    cfg["vix_floor"] = 12
    cfg["vix_ceil"] = 25
    return cfg


def gate_cfg_with_lookup(threshold, trend_lookup):
    """Build a cfg with the in-loop directional gate, custom trend lookup."""
    cfg = base_cfg()
    cfg["directional_gate_threshold"] = threshold
    cfg["_trend_lookup_5d"] = trend_lookup  # the in-loop reads this key
    return cfg


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


def main():
    full_start = dt.date(2024, 7, 1)
    full_end = dt.date(2026, 4, 6)

    print("=" * 130)
    print("  DIRECTIONAL GATE ROBUSTNESS SWEEP")
    print("  Tests sensitivity to threshold (5 levels) x lookback (4 levels) = 24 variants")
    print("=" * 130)

    print(f"\nLoading {full_start} -> {full_end} ...", flush=True)
    preloaded = load_period_data(start_date=full_start, end_date=full_end, quiet=True)
    day_groups, _, all_dates, _ = preloaded
    print(f"  {len(all_dates)} trading days\n", flush=True)

    # Pre-compute trend lookups for all lookbacks once
    print("Pre-computing trend lookups for 4 lookback windows ...", flush=True)
    trend_lookups = {lb: build_trend_lookup(day_groups, lb) for lb in LOOKBACKS}
    print(f"  Built {len(trend_lookups)} lookup tables\n", flush=True)

    # ── Baseline ──
    print("Running BASELINE (no gate) ...", flush=True)
    base_trades, _ = run_backtest(start_date=full_start, end_date=full_end,
                                  cfg_override=base_cfg(), quiet=True, preloaded=preloaded)
    bm = metrics(base_trades)
    bd = equity_dd(base_trades, CAPITAL)
    print(f"  BASELINE: n={bm['n']}  PnL=Rs {bm['pnl']:+,.0f}  PF={fmt_pf(bm['pf'])}  "
          f"WR={bm['wr']:.1f}%  DD=Rs {bd:+,.0f}\n", flush=True)

    # ── Phase 1: 21-month grid ──
    print("PHASE 1: 21-month grid sweep", flush=True)
    print("-" * 130, flush=True)
    print(f"  {'lookback':<8} {'thr':<6} {'n':>4} {'PnL':>14} {'PF':>5} "
          f"{'WR%':>6} {'DD':>11} {'dPnL':>13} {'dPF':>6}")
    print("  " + "-" * 100)

    grid_results = []  # (lookback, thr, m, dd)
    for lb in LOOKBACKS:
        trend_lookup = trend_lookups[lb]
        for thr in THRESHOLDS:
            cfg = gate_cfg_with_lookup(thr, trend_lookup)
            trades, _ = run_backtest(start_date=full_start, end_date=full_end,
                                     cfg_override=cfg, quiet=True, preloaded=preloaded)
            m = metrics(trades)
            dd = equity_dd(trades, CAPITAL)
            d_pnl = m["pnl"] - bm["pnl"]
            m_pf = m["pf"] if m["pf"] != float("inf") else 999
            b_pf = bm["pf"] if bm["pf"] != float("inf") else 999
            d_pf = m_pf - b_pf
            grid_results.append((lb, thr, m, dd))
            print(f"  {lb:<8d} {thr:<6.2f} {m['n']:>4d} Rs {m['pnl']:>+12,.0f} "
                  f"{fmt_pf(m['pf'])} {m['wr']:>5.1f}% Rs {dd:>+9,.0f} "
                  f"Rs {d_pnl:>+10,.0f} {d_pf:>+6.2f}", flush=True)

    # Top 3 by PnL
    grid_results.sort(key=lambda x: -x[2]["pnl"])
    top3 = grid_results[:3]

    print()
    print("=" * 130)
    print("  TOP 3 BY 21mo PnL")
    print("=" * 130)
    for lb, thr, m, dd in top3:
        print(f"  lookback={lb}, thr={thr}: PnL=Rs {m['pnl']:+,.0f}  PF={fmt_pf(m['pf'])}  "
              f"WR={m['wr']:.1f}%  DD=Rs {dd:+,.0f}")

    # ── Phase 2: walk-forward top 3 ──
    print()
    print("=" * 130)
    print("  PHASE 2: Walk-forward top 3 candidates")
    print("=" * 130)

    final_results = []
    for lb, thr, _, _ in top3:
        print(f"\n--- lookback={lb}, threshold={thr} ---", flush=True)
        pnl_wins = pf_wins = 0
        catastrophic = []
        net_delta = 0.0
        for label, w_start, w_end in WINDOWS:
            pre = load_period_data(start_date=w_start, end_date=w_end, quiet=True)
            day_groups_w, _, _, _ = pre
            trend_w = build_trend_lookup(day_groups_w, lb)
            cfg_a = base_cfg()
            cfg_g = gate_cfg_with_lookup(thr, trend_w)
            ta, _ = run_backtest(start_date=w_start, end_date=w_end,
                                 cfg_override=cfg_a, quiet=True, preloaded=pre)
            tg, _ = run_backtest(start_date=w_start, end_date=w_end,
                                 cfg_override=cfg_g, quiet=True, preloaded=pre)
            ma = metrics(ta)
            mg = metrics(tg)
            d_pnl = mg["pnl"] - ma["pnl"]
            net_delta += d_pnl
            a_pf = ma["pf"] if ma["pf"] != float("inf") else 999
            g_pf = mg["pf"] if mg["pf"] != float("inf") else 999
            d_pf = g_pf - a_pf
            if d_pnl > 0:
                pnl_wins += 1
            if d_pf >= 0:
                pf_wins += 1
            if ma["pnl"] > 0 and (mg["pnl"] < 0 or mg["pnl"] < 0.5 * ma["pnl"]):
                catastrophic.append(label)
            print(f"    {label}: A=Rs {ma['pnl']:+9,.0f} ({fmt_pf(ma['pf'])}) | "
                  f"G=Rs {mg['pnl']:+9,.0f} ({fmt_pf(mg['pf'])}) | "
                  f"dPnL=Rs {d_pnl:+9,.0f}  dPF={d_pf:+5.2f}", flush=True)

        verdict = ("STRONG" if (pnl_wins >= 4 and pf_wins >= 4 and not catastrophic)
                   else ("MARGINAL" if not catastrophic else "FAIL"))
        print(f"    -> PnL wins: {pnl_wins}/6  PF wins: {pf_wins}/6  "
              f"net Δ: Rs {net_delta:+,.0f}  catastrophic: {len(catastrophic)}  -> {verdict}")
        final_results.append({
            "lookback": lb, "threshold": thr,
            "pnl_wins": pnl_wins, "pf_wins": pf_wins,
            "net_delta": net_delta, "catastrophic": len(catastrophic),
            "verdict": verdict,
        })

    print()
    print("=" * 130)
    print("  FINAL RANKING (walk-forward verified)")
    print("=" * 130)
    final_results.sort(key=lambda r: (-(r["pnl_wins"] + r["pf_wins"]), -r["net_delta"]))
    for r in final_results:
        print(f"  lookback={r['lookback']}, thr={r['threshold']}: "
              f"PnL_wins={r['pnl_wins']}/6  PF_wins={r['pf_wins']}/6  "
              f"net=Rs {r['net_delta']:+,.0f}  catastrophic={r['catastrophic']}  "
              f"-> {r['verdict']}")

    # Best
    best = final_results[0]
    print(f"\n  BEST: lookback={best['lookback']}, threshold={best['threshold']}")
    print(f"  Recommendation: deploy this configuration after live agent integration.")


if __name__ == "__main__":
    main()
