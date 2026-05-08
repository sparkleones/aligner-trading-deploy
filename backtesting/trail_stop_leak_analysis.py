"""
Forensic on baseline's trail-stop leak.

Background: WR sweep forensic showed deployed Option B fires trail_stop
only 8 times across 21 months, all net negative (-Rs 4.2L). TIGHT_TRAIL
fires it 25 times net positive (+Rs 1.8L). Need to understand: are the
baseline's 8 trail-stops "false alarms" (would have won if held), or
"correct catches" that the system simply mis-times?

Approach:
  1. Run V17 baseline (Option B) and capture all trades
  2. Filter to trail_stop exits
  3. For each: show entry/exit context, P&L, bars held
  4. Counter-factual: re-simulate each trade, what if it had been held
     to EOD instead? Compute hypothetical P&L delta.

Usage:
    python -m backtesting.trail_stop_leak_analysis
"""
import sys
import datetime as dt
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backtesting.v14_unified_backtest import (
    load_period_data, run_backtest, calc_premium, LOT_SIZE,
)
from scoring.config import V17_CONFIG

CAPITAL = 2_00_000


def to_date(d):
    return dt.date.fromisoformat(d) if isinstance(d, str) else d


def main():
    start = dt.date(2024, 7, 1)
    end = dt.date(2026, 4, 6)
    print(f"Loading period data {start} -> {end} ...", flush=True)
    preloaded = load_period_data(start_date=start, end_date=end, quiet=True)
    day_groups, vix_lookup, all_dates, _ = preloaded
    print(f"  {len(all_dates)} trading days\n", flush=True)

    # Run baseline (Option B)
    cfg = dict(V17_CONFIG)
    cfg["avoid_days"] = [0, 2]
    cfg["use_v17_regime_gate"] = False
    cfg["use_v17_monwed_gate"] = False
    cfg["vix_floor"] = 12
    cfg["vix_ceil"] = 25

    print("Running V17_PROD_ONLY baseline (Option B)...", flush=True)
    trades, _ = run_backtest(start_date=start, end_date=end,
                             cfg_override=cfg, quiet=True, preloaded=preloaded)
    print(f"  {len(trades)} total trades\n", flush=True)

    # Filter to trail_stop exits
    trail_trades = [t for t in trades if t["exit_reason"] == "trail_stop"]
    print(f"  {len(trail_trades)} trail_stop exits\n", flush=True)

    print("=" * 130)
    print("  BASELINE TRAIL-STOP TRADES — FULL DETAIL")
    print("=" * 130)
    print(f"  {'date':<11} {'DOW':<4} {'type':<6} {'opt':<3} {'strike':>6} "
          f"{'entry_spot':>9} {'exit_spot':>9} {'entry_pr':>8} {'exit_pr':>8} "
          f"{'bars':>4} {'pnl':>10} {'vix':>5}")
    print("  " + "-" * 126)
    DOW = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    total_actual = 0.0
    for t in sorted(trail_trades, key=lambda x: to_date(x["date"])):
        d = to_date(t["date"])
        total_actual += t["pnl"]
        print(f"  {str(d):<11} {DOW[d.weekday()]:<4} {t.get('entry_type','?'):<6} "
              f"{t.get('opt_type','?'):<3} {int(t.get('strike',0)):>6} "
              f"{t.get('entry_spot',0):>9.0f} {t.get('exit_spot',0):>9.0f} "
              f"{t.get('entry_premium',0):>8.2f} {t.get('exit_premium',0):>8.2f} "
              f"{t.get('bars_held',0):>4d} Rs {t['pnl']:>+8,.0f} {t.get('vix',0):>5.1f}")

    # ── Counter-factual: what if these trades had been held to EOD ──
    print()
    print("=" * 130)
    print("  COUNTER-FACTUAL: hold these trades to EOD instead")
    print("=" * 130)
    print(f"  Approach: re-price the option at the day's last 5-min bar close,")
    print(f"  using same calc_premium model. Compare hypothetical EOD-exit P&L vs actual trail-stop P&L.")
    print()
    print(f"  {'date':<11} {'opt':<3} {'strike':>6} "
          f"{'actual_pnl':>11} {'eod_pnl':>11} {'delta':>10} "
          f"{'verdict':<28}")
    print("  " + "-" * 126)

    total_cf = 0.0
    n_better = 0
    n_worse = 0
    for t in sorted(trail_trades, key=lambda x: to_date(x["date"])):
        d = to_date(t["date"])
        bars = day_groups.get(d, [])
        if not bars:
            continue
        # Get EOD spot
        eod_spot = bars[-1]["close"]
        n_bars_total = len(bars)
        # Re-price option at EOD with reduced DTE (~ same day)
        dte = max(0.05, t.get("dte_at_entry", 1.0) - n_bars_total * 5 / 1440.0)
        try:
            eod_premium = calc_premium(
                eod_spot, t["strike"], dte, t.get("vix", 14),
                t["opt_type"], slippage_sign=-1,  # we'd be selling
            )
        except Exception:
            continue
        qty = t.get("qty", 0)
        if not qty:
            continue
        eod_pnl_calc = (eod_premium - t["entry_premium"]) * qty - 80  # 80 = brokerage RT
        delta = eod_pnl_calc - t["pnl"]
        total_cf += eod_pnl_calc
        verdict = ""
        if eod_pnl_calc > t["pnl"]:
            n_better += 1
            if eod_pnl_calc > 0 and t["pnl"] < 0:
                verdict = "**TRAIL HURT — would have won**"
            elif eod_pnl_calc > 0:
                verdict = "smaller loss avoided"
            else:
                verdict = "less bad if held"
        else:
            n_worse += 1
            verdict = "trail correctly cut"
        print(f"  {str(d):<11} {t.get('opt_type','?'):<3} {int(t.get('strike',0)):>6} "
              f"Rs {t['pnl']:>+8,.0f} Rs {eod_pnl_calc:>+8,.0f} Rs {delta:>+8,.0f} "
              f"{verdict:<28}")

    print()
    print("=" * 130)
    print("  SUMMARY")
    print("=" * 130)
    n = len(trail_trades)
    print(f"  Total trail-stop trades:  {n}")
    print(f"  Actual P&L sum:           Rs {total_actual:+,.0f}")
    print(f"  Hypothetical EOD P&L sum: Rs {total_cf:+,.0f}")
    print(f"  Net delta (EOD - trail):  Rs {total_cf - total_actual:+,.0f}")
    print(f"  Trades trail HURT:        {n_better}/{n}  (would have done better at EOD)")
    print(f"  Trades trail HELPED:      {n_worse}/{n}  (correctly cut a worsening loss)")
    print()
    if n_better > n_worse:
        print(f"  VERDICT: trail mechanism is firing on FALSE ALARMS more than valid signals.")
        print(f"  Recommendation: tighten activation criteria (require deeper drawdown before firing)")
        print(f"  OR replace percent-trail with chandelier-trail (ATR-based).")
    else:
        print(f"  VERDICT: trail mechanism mostly correct catches; the few losses are noise.")


if __name__ == "__main__":
    main()
