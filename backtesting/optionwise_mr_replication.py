"""
Replicate OptionWise's "Dynamic Mean Reversion" strategy on our real 5-min
NIFTY data, under three different cost/methodology assumptions, to surgically
expose how the claimed 68% WR is constructed.

Their stated rules (from their architecture doc):
  - Trigger:  big overnight gap, |gap%| > 0.60%
  - Direction:  gap UP  -> buy PE   (fade — expect profit booking)
                gap DN  -> buy CE   (fade — expect bounce)
  - Strike: ATM (delta 0.5 assumption)
  - Spot SL: 20 points against entry direction  (-> ~10 option pts)
  - Spot Target: 50 points toward pre-gap level  (-> ~25 option pts)
  - Daily OHLC backtest, ₹60 brokerage, 2 pts slippage per leg

Three replication variants:
  A. HONEST       — 5-min bars, first-hit-bar resolution, our realistic costs
                    (Rs 80 RT brokerage + Black-Scholes premium with our
                    slippage model). This is what you'd actually see live.
  B. OPTIMISTIC   — 5-min bars, BUT if both target & SL would hit on the
                    same bar, assume target wins (mild inflation).
  C. OW-METHOD    — Daily OHLC granularity, target-wins-tie, their cheap
                    cost model (Rs 60 + 2 pts slippage). This is closest
                    to their claimed backtest.

Output: WR, PF, P&L per variant. The delta between A and C is the
"backtest-inflation premium" — i.e. the fake portion of their claim.

Usage:
    python -m backtesting.optionwise_mr_replication
"""
import sys
import datetime as dt
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backtesting.v14_unified_backtest import load_period_data, calc_premium, LOT_SIZE

CAPITAL = 2_00_000
POST_SEP = dt.date(2025, 9, 1)
GAP_THRESHOLD = 0.60          # %
SPOT_SL_PTS = 20              # points
SPOT_TARGET_PTS = 50          # points
BROKERAGE_REALISTIC = 80.0    # Rs round-trip (our backtest constant)
BROKERAGE_OW = 60.0           # Rs round-trip (their stated)
# OW slippage: 2 pts per leg = 4 pts round trip in option points
# Our calc_premium has built-in SLIPPAGE_PCT + SPREAD_RS already; for OW
# variant we override with a fixed 2 pt deduction per leg.

DOW_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri"]


def build_gap_lookup(day_groups):
    gap = {}
    prev_close = 0.0
    for d in sorted(day_groups.keys()):
        bars = day_groups[d]
        if not bars:
            continue
        day_open = bars[0]["open"]
        if prev_close > 0:
            gap[d] = (day_open - prev_close) / prev_close * 100.0
        prev_close = bars[-1]["close"]
    return gap


def round_atm(spot, step=50):
    return int(round(spot / step) * step)


def estimate_dte(date_obj):
    """Tuesday expiry post-Sep-2025, Thursday before."""
    target = 1 if date_obj >= dt.date(2025, 9, 1) else 3
    days = (target - date_obj.weekday()) % 7
    return max(0.1, float(days)) + 0.5  # half-day padding for intraday


def simulate_mr_day(bars, entry_spot, direction, opt_type, dte, vix,
                    method="5min", target_wins_tie=False,
                    use_ow_costs=False):
    """Return (exit_spot, exit_reason, bars_held). direction: +1=long-CE/spot-up, -1=long-PE/spot-dn"""
    # SL when spot moves AGAINST direction by SPOT_SL_PTS
    # Target when spot moves WITH direction by SPOT_TARGET_PTS
    if direction == +1:  # bought CE, want spot UP
        sl_level = entry_spot - SPOT_SL_PTS      # if low <= sl_level, SL hit
        tgt_level = entry_spot + SPOT_TARGET_PTS # if high >= tgt_level, target hit
    else:                # bought PE, want spot DOWN
        sl_level = entry_spot + SPOT_SL_PTS      # if high >= sl_level, SL hit
        tgt_level = entry_spot - SPOT_TARGET_PTS # if low <= tgt_level, target hit

    if method == "daily":
        # Use full-day OHLC
        day_high = max(b["high"] for b in bars)
        day_low = min(b["low"] for b in bars)
        if direction == +1:
            sl_hit = day_low <= sl_level
            tgt_hit = day_high >= tgt_level
        else:
            sl_hit = day_high >= sl_level
            tgt_hit = day_low <= tgt_level
        if tgt_hit and sl_hit:
            # Both hit — daily OHLC can't distinguish order
            if target_wins_tie:
                return tgt_level, "target", len(bars)
            else:
                return sl_level, "stop_loss", len(bars)
        if tgt_hit:
            return tgt_level, "target", len(bars)
        if sl_hit:
            return sl_level, "stop_loss", len(bars)
        # Neither hit
        return bars[-1]["close"], "eod_close", len(bars)

    # 5-min granularity — scan bars in order
    for i, b in enumerate(bars[1:], start=1):
        if direction == +1:
            sl_in_bar = b["low"] <= sl_level
            tgt_in_bar = b["high"] >= tgt_level
        else:
            sl_in_bar = b["high"] >= sl_level
            tgt_in_bar = b["low"] <= tgt_level
        if sl_in_bar and tgt_in_bar:
            # Both within same 5-min bar — pessimistic assumption: SL first
            # unless target_wins_tie is on
            if target_wins_tie:
                return tgt_level, "target", i
            return sl_level, "stop_loss", i
        if tgt_in_bar:
            return tgt_level, "target", i
        if sl_in_bar:
            return sl_level, "stop_loss", i
    # Neither hit by EOD
    return bars[-1]["close"], "eod_close", len(bars)


def run_variant(name, day_groups, vix_lookup, gap_lookup, all_dates,
                method, target_wins_tie, use_ow_costs):
    trades = []
    for d in all_dates:
        gap = gap_lookup.get(d)
        if gap is None or abs(gap) <= GAP_THRESHOLD:
            continue
        bars = day_groups.get(d, [])
        if len(bars) < 6:
            continue
        # OW Mean Reversion: fade the gap
        direction = -1 if gap > 0 else +1   # gap UP -> short spot via PE -> direction=-1
        opt_type = "PE" if direction == -1 else "CE"
        entry_spot = bars[0]["close"]
        strike = round_atm(entry_spot)
        vix = vix_lookup.get(d, 14.0)
        dte = estimate_dte(d)

        exit_spot, reason, bars_held = simulate_mr_day(
            bars, entry_spot, direction, opt_type, dte, vix,
            method=method, target_wins_tie=target_wins_tie,
            use_ow_costs=use_ow_costs,
        )

        # Premiums via Black-Scholes
        entry_prem = calc_premium(entry_spot, strike, dte, vix, opt_type, slippage_sign=+1)
        # On exit, dte slightly less (bars_held * 5 minutes)
        dte_exit = max(0.05, dte - bars_held * 5 / (60 * 24 * 365))
        exit_prem = calc_premium(exit_spot, strike, dte_exit, vix, opt_type, slippage_sign=-1)

        # Apply OW-style cheap costs override (replace our slippage with their 2 pts/leg)
        if use_ow_costs:
            # Strip our slippage from the premiums and apply theirs
            base_entry = calc_premium(entry_spot, strike, dte, vix, opt_type, slippage_sign=0)
            base_exit  = calc_premium(exit_spot, strike, dte_exit, vix, opt_type, slippage_sign=0)
            entry_prem = base_entry + 2.0   # they pay 2 pts on entry
            exit_prem  = max(0.05, base_exit - 2.0)  # they receive 2 pts less on exit
            brokerage = BROKERAGE_OW
        else:
            brokerage = BROKERAGE_REALISTIC

        # 1 lot sizing (apples-to-apples)
        lots = 1
        qty = lots * LOT_SIZE
        pnl = (exit_prem - entry_prem) * qty - brokerage

        trades.append({
            "date": d, "gap_pct": gap, "direction": direction, "opt_type": opt_type,
            "entry_spot": entry_spot, "exit_spot": exit_spot,
            "strike": strike, "entry_prem": entry_prem, "exit_prem": exit_prem,
            "exit_reason": reason, "bars_held": bars_held, "pnl": pnl,
        })
    return trades


def metrics(trades):
    if not trades:
        return {"n": 0, "pnl": 0.0, "wr": 0.0, "pf": 0.0, "exits": {}}
    pnls = [t["pnl"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    pf = sum(wins) / -sum(losses) if losses and sum(losses) < 0 else float("inf")
    exits = {}
    for t in trades:
        exits.setdefault(t["exit_reason"], 0)
        exits[t["exit_reason"]] += 1
    return {
        "n": len(trades),
        "pnl": sum(pnls),
        "wr": len(wins) / len(trades) * 100,
        "pf": pf,
        "exits": exits,
    }


def main():
    start = dt.date(2024, 7, 1)
    end = dt.date(2026, 4, 6)
    print(f"Loading period data {start} -> {end} ...", flush=True)
    preloaded = load_period_data(start_date=start, end_date=end, quiet=True)
    day_groups, vix_lookup, all_dates, _ = preloaded
    gap_lookup = build_gap_lookup(day_groups)

    n_big_gap = sum(1 for d in all_dates if (gap_lookup.get(d) or 0) and abs(gap_lookup[d]) > GAP_THRESHOLD)
    print(f"  {len(all_dates)} trading days, {n_big_gap} big-gap days (|gap|>{GAP_THRESHOLD}%)\n", flush=True)

    # ── Three variants ──
    variants = [
        ("A. HONEST       (5-min, our costs, SL-wins-tie)",   "5min",  False, False),
        ("B. OPTIMISTIC   (5-min, our costs, TGT-wins-tie)",  "5min",  True,  False),
        ("C. OW-METHOD    (daily OHLC, OW costs, TGT-wins)",  "daily", True,  True),
    ]

    print(f"{'variant':50s} {'n':>4s} {'PnL':>14s} {'WR%':>6s} {'PF':>6s}   exits")
    print("-" * 130)

    all_results = {}
    for label, method, tgt_tie, ow_costs in variants:
        trades = run_variant(label, day_groups, vix_lookup, gap_lookup, all_dates,
                             method, tgt_tie, ow_costs)
        m = metrics(trades)
        all_results[label] = (trades, m)
        exits_str = ", ".join(f"{r}={n}" for r, n in m["exits"].items())
        print(f"  {label:48s} {m['n']:>4d} Rs {m['pnl']:>+11,.0f} {m['wr']:>5.1f}% {m['pf']:>5.2f}   {exits_str}")

    # Post-Sep slice
    print()
    print("=" * 130)
    print("POST-SEP-2025 sub-slice (the regime that matters now)")
    print("=" * 130)
    print(f"{'variant':50s} {'n':>4s} {'PnL':>14s} {'WR%':>6s} {'PF':>6s}")
    print("-" * 130)
    for label, _ in [(v[0], None) for v in variants]:
        trades, _ = all_results[label]
        post = [t for t in trades if t["date"] >= POST_SEP]
        m = metrics(post)
        print(f"  {label:48s} {m['n']:>4d} Rs {m['pnl']:>+11,.0f} {m['wr']:>5.1f}% {m['pf']:>5.2f}")

    # Cost arbitrage estimate
    print()
    print("=" * 130)
    print("WR INFLATION DECOMPOSITION (full window)")
    print("=" * 130)
    a_m = all_results[variants[0][0]][1]
    b_m = all_results[variants[1][0]][1]
    c_m = all_results[variants[2][0]][1]
    print(f"  A. HONEST       WR = {a_m['wr']:5.1f}%   PF = {a_m['pf']:.2f}   PnL = Rs {a_m['pnl']:+,.0f}")
    print(f"  B. OPTIMISTIC   WR = {b_m['wr']:5.1f}%   PF = {b_m['pf']:.2f}   PnL = Rs {b_m['pnl']:+,.0f}")
    print(f"     -> tie-resolution inflation: +{b_m['wr']-a_m['wr']:.1f}pp WR")
    print(f"  C. OW-METHOD    WR = {c_m['wr']:5.1f}%   PF = {c_m['pf']:.2f}   PnL = Rs {c_m['pnl']:+,.0f}")
    print(f"     -> daily-OHLC + cheap-cost inflation: +{c_m['wr']-b_m['wr']:.1f}pp WR")
    print(f"     -> TOTAL inflation A->C: +{c_m['wr']-a_m['wr']:.1f}pp WR")
    print()
    print(f"Their claimed:  68% WR.  Our honest replication on real data:  {a_m['wr']:.1f}% WR.")


if __name__ == "__main__":
    main()
