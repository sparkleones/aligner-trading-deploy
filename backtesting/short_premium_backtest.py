"""
Short-premium (volatility-selling) backtester for NIFTY weekly options.

Implements three sub-strategies on real 5-min bars:
  1. SHORT STRANGLE  — sell 1 strike OTM CE + 1 strike OTM PE
  2. IRON CONDOR     — short strangle + buy 3 strikes OTM wings (defined risk)
  3. SHORT STRADDLE  — sell ATM CE + ATM PE (most credit, most risk)

Common rules:
  - Entry:  09:20 (after first 5-min bar settles)
  - Exit conditions (whichever first):
      a. Stop-loss: combined position MTM hits -100% of credit received
         (i.e., we lose what we collected — typical practitioner SL)
      b. Profit target: capture 50% of credit
      c. Time stop: 15:15 (intraday only, no overnight)
  - Strike step: 50 pts for NIFTY
  - Lot size: 75 (current NIFTY)
  - Costs: Rs 60 per leg round-trip + slippage via calc_premium

Filters tested:
  - Baseline (every day)
  - avoid_days=[0,2] (same as V17)
  - V17-style (avoid_days + VIX gate)
  - Mon+Tue only (literature: best for premium selling)

Usage:
    python -m backtesting.short_premium_backtest
"""
import sys
import datetime as dt
from pathlib import Path
from collections import defaultdict

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backtesting.v14_unified_backtest import (
    load_period_data, calc_premium, LOT_SIZE,
)

CAPITAL = 2_00_000
POST_SEP = dt.date(2025, 9, 1)
COST_PER_LEG_RT = 30.0   # Rs 30 per leg one-way ≈ Rs 60 round-trip per leg
SL_MULT = 1.0            # SL = -100% of credit (loss = credit received)
TP_FRAC = 0.50           # Take profit at 50% of credit captured
ENTRY_BAR = 1            # 09:20 (bar 1 close, after first 5-min settles)
EXIT_BAR_LIMIT = 72      # 15:15 (bar 72 = end of trading)


def estimate_dte(d):
    target = 1 if d >= dt.date(2025, 9, 1) else 3
    days = (target - d.weekday()) % 7
    return max(0.1, float(days)) + 0.5  # half-day padding for intraday


def round_atm(spot, step=50):
    return int(round(spot / step) * step)


def short_position_pnl(credit_received, current_combined_premium, lots):
    """For a short position, P&L per lot = (credit_received - current_premium) * LOT_SIZE.
    Positive when premium has decayed below entry credit."""
    return (credit_received - current_combined_premium) * lots * LOT_SIZE


def simulate_strangle(bars, vix, dte, strike_step=50, otm_strikes=1,
                      iron_condor_wing=None):
    """Returns trade dict or None if not enough data."""
    if len(bars) <= ENTRY_BAR + 5:
        return None
    entry_bar_data = bars[ENTRY_BAR]
    entry_spot = entry_bar_data["close"]
    atm = round_atm(entry_spot, strike_step)

    # Strike selection
    short_ce_strike = atm + otm_strikes * strike_step
    short_pe_strike = atm - otm_strikes * strike_step

    # Get entry premiums (we SELL — receive premium minus slippage)
    ce_credit = calc_premium(entry_spot, short_ce_strike, dte, vix, "CE", slippage_sign=-1)
    pe_credit = calc_premium(entry_spot, short_pe_strike, dte, vix, "PE", slippage_sign=-1)

    long_ce_strike = long_pe_strike = None
    long_ce_debit = long_pe_debit = 0.0
    if iron_condor_wing:
        long_ce_strike = atm + (otm_strikes + iron_condor_wing) * strike_step
        long_pe_strike = atm - (otm_strikes + iron_condor_wing) * strike_step
        long_ce_debit = calc_premium(entry_spot, long_ce_strike, dte, vix, "CE", slippage_sign=+1)
        long_pe_debit = calc_premium(entry_spot, long_pe_strike, dte, vix, "PE", slippage_sign=+1)

    net_credit = (ce_credit + pe_credit) - (long_ce_debit + long_pe_debit)
    if net_credit <= 0:
        return None

    # Manage intraday — scan bars from ENTRY_BAR+1 to EXIT_BAR_LIMIT
    sl_threshold = -net_credit * SL_MULT * LOT_SIZE
    tp_threshold = +net_credit * TP_FRAC * LOT_SIZE

    exit_reason = "time_exit"
    exit_bar_idx = min(EXIT_BAR_LIMIT, len(bars) - 1)
    exit_spot = bars[exit_bar_idx]["close"]

    # MTM check at each subsequent bar's CLOSE (not high/low — strangles are
    # sensitive to spot moves but not as path-dependent as outright options)
    for i in range(ENTRY_BAR + 1, min(EXIT_BAR_LIMIT, len(bars))):
        spot_now = bars[i]["close"]
        # Time decay over i-ENTRY_BAR bars
        elapsed_min = (i - ENTRY_BAR) * 5
        dte_now = max(0.05, dte - elapsed_min / 1440.0)  # minutes -> days
        # Current combined premium (mark-to-market)
        ce_now = calc_premium(spot_now, short_ce_strike, dte_now, vix, "CE", slippage_sign=+1)
        pe_now = calc_premium(spot_now, short_pe_strike, dte_now, vix, "PE", slippage_sign=+1)
        if iron_condor_wing:
            ce_long_now = calc_premium(spot_now, long_ce_strike, dte_now, vix, "CE", slippage_sign=-1)
            pe_long_now = calc_premium(spot_now, long_pe_strike, dte_now, vix, "PE", slippage_sign=-1)
            current_net = (ce_now + pe_now) - (ce_long_now + pe_long_now)
        else:
            current_net = ce_now + pe_now
        mtm = (net_credit - current_net) * LOT_SIZE

        if mtm <= sl_threshold:
            exit_reason = "stop_loss"
            exit_bar_idx = i
            exit_spot = spot_now
            break
        if mtm >= tp_threshold:
            exit_reason = "target"
            exit_bar_idx = i
            exit_spot = spot_now
            break

    # Final P&L at exit
    elapsed_min = (exit_bar_idx - ENTRY_BAR) * 5
    dte_exit = max(0.05, dte - elapsed_min / 1440.0)  # minutes -> days
    ce_exit = calc_premium(exit_spot, short_ce_strike, dte_exit, vix, "CE", slippage_sign=+1)
    pe_exit = calc_premium(exit_spot, short_pe_strike, dte_exit, vix, "PE", slippage_sign=+1)
    n_legs = 2
    if iron_condor_wing:
        ce_long_exit = calc_premium(exit_spot, long_ce_strike, dte_exit, vix, "CE", slippage_sign=-1)
        pe_long_exit = calc_premium(exit_spot, long_pe_strike, dte_exit, vix, "PE", slippage_sign=-1)
        net_exit = (ce_exit + pe_exit) - (ce_long_exit + pe_long_exit)
        n_legs = 4
    else:
        net_exit = ce_exit + pe_exit

    pnl_per_lot = (net_credit - net_exit) * LOT_SIZE
    cost = n_legs * COST_PER_LEG_RT
    pnl = pnl_per_lot - cost

    return {
        "entry_spot": entry_spot, "exit_spot": exit_spot,
        "atm": atm,
        "short_ce_strike": short_ce_strike, "short_pe_strike": short_pe_strike,
        "long_ce_strike": long_ce_strike, "long_pe_strike": long_pe_strike,
        "net_credit": net_credit, "net_exit": net_exit,
        "exit_reason": exit_reason, "bars_held": exit_bar_idx - ENTRY_BAR,
        "pnl": pnl,
    }


def run_strategy(label, day_groups, vix_lookup, all_dates,
                 otm_strikes=1, iron_condor_wing=None,
                 avoid_days=None, vix_floor=None, vix_ceil=None,
                 only_dows=None):
    avoid_days = avoid_days or []
    trades = []
    for d in all_dates:
        if d.weekday() in avoid_days:
            continue
        if only_dows is not None and d.weekday() not in only_dows:
            continue
        vix = vix_lookup.get(d, 14.0)
        if vix_floor is not None and vix < vix_floor:
            continue
        if vix_ceil is not None and vix > vix_ceil:
            continue
        bars = day_groups.get(d, [])
        if len(bars) < ENTRY_BAR + 10:
            continue
        dte = estimate_dte(d)
        result = simulate_strangle(bars, vix, dte, otm_strikes=otm_strikes,
                                   iron_condor_wing=iron_condor_wing)
        if result is None:
            continue
        result["date"] = d
        result["vix"] = vix
        trades.append(result)
    return label, trades


def metrics(trades):
    if not trades:
        return {"n": 0, "pnl": 0.0, "wr": 0.0, "pf": 0.0,
                "exits": {}, "avg_win": 0.0, "avg_loss": 0.0}
    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    tw = sum(t["pnl"] for t in wins)
    tl = -sum(t["pnl"] for t in losses)
    pf = tw / tl if tl > 0 else float("inf")
    exits = defaultdict(int)
    for t in trades:
        exits[t["exit_reason"]] += 1
    return {
        "n": len(trades), "pnl": sum(t["pnl"] for t in trades),
        "wr": len(wins) / len(trades) * 100, "pf": pf,
        "avg_win": tw / max(1, len(wins)),
        "avg_loss": -tl / max(1, len(losses)),
        "exits": dict(exits),
        "max_win": max(t["pnl"] for t in trades),
        "max_loss": min(t["pnl"] for t in trades),
    }


def split(trades, cutover):
    pre, post = [], []
    for t in trades:
        d = t.get("date")
        if isinstance(d, str):
            d = dt.date.fromisoformat(d)
        (pre if d < cutover else post).append(t)
    return pre, post


def main():
    start = dt.date(2024, 7, 1)
    end = dt.date(2026, 4, 6)
    print(f"Loading period data {start} -> {end} ...", flush=True)
    preloaded = load_period_data(start_date=start, end_date=end, quiet=True)
    day_groups, vix_lookup, all_dates, _ = preloaded
    print(f"  {len(all_dates)} trading days\n", flush=True)

    variants = [
        # (label, otm_strikes, iron_condor_wing, avoid_days, vix_floor, vix_ceil, only_dows)
        ("STRANGLE 1OTM   no filters",            1, None, [],     None, None, None),
        ("STRANGLE 1OTM   avoid_days [0,2]",      1, None, [0,2],  None, None, None),
        ("STRANGLE 1OTM   V17-style filters",     1, None, [0,2],  12,   25,   None),
        ("STRANGLE 1OTM   Mon+Tue only",          1, None, [],     None, None, [0,1]),
        ("STRANGLE 2OTM   V17-style filters",     2, None, [0,2],  12,   25,   None),
        ("CONDOR 1/3      no filters",            1, 2,    [],     None, None, None),
        ("CONDOR 1/3      avoid_days [0,2]",      1, 2,    [0,2],  None, None, None),
        ("CONDOR 1/3      V17-style filters",     1, 2,    [0,2],  12,   25,   None),
        ("STRADDLE ATM    V17-style filters",     0, None, [0,2],  12,   25,   None),
    ]

    print(f"{'variant':38s} {'n':>4s}  {'PnL':>14s}  {'x':>6s}  {'WR':>5s}  {'PF':>5s}     "
          f"{'post-Sep n':>10s}  {'PnL':>12s}  {'WR':>5s}  {'PF':>5s}    exits")
    print("-" * 170)

    rows = []
    for v in variants:
        label, otm, wing, avoid, vfloor, vceil, dows = v
        _, trades = run_strategy(label, day_groups, vix_lookup, all_dates,
                                 otm_strikes=otm, iron_condor_wing=wing,
                                 avoid_days=avoid, vix_floor=vfloor, vix_ceil=vceil,
                                 only_dows=dows)
        full = metrics(trades)
        _, post = split(trades, POST_SEP)
        ps = metrics(post)
        ret_x = (CAPITAL + full["pnl"]) / CAPITAL
        exits_short = ", ".join(f"{r}={n}" for r, n in full["exits"].items())
        rows.append((label, full, ps, ret_x))
        print(f"  {label:36s} {full['n']:>4d}  Rs {full['pnl']:>+12,.0f}  "
              f"{ret_x:>5.2f}x  {full['wr']:>4.1f}%  {full['pf']:>4.2f}     "
              f"{ps['n']:>10d}  Rs {ps['pnl']:>+10,.0f}  "
              f"{ps['wr']:>4.1f}%  {ps['pf']:>4.2f}    {exits_short}")

    print()
    print("=" * 170)
    print("RANKED BY POST-SEP P&L")
    print("=" * 170)
    for label, full, ps, ret_x in sorted(rows, key=lambda r: -r[2]["pnl"]):
        print(f"  {label:38s} post-Sep: Rs {ps['pnl']:>+12,.0f}  "
              f"(n={ps['n']:>3d}, PF={ps['pf']:.2f}, WR={ps['wr']:.1f}%)   "
              f"full: {ret_x:5.2f}x / Rs {full['pnl']:>+12,.0f}")

    # Comparison to V17 baseline
    print()
    print("=" * 170)
    print("COMPARISON: V17_PROD_ONLY (Option A) reference")
    print("=" * 170)
    print("  V17 BUYING (deployed):   28.99x / Rs +5,597,481  WR 42.6%  PF 1.89    "
          "post-Sep: Rs +1,422,073  WR 45.2%  PF 1.73")
    print("  V17 + ceil=25 (proposed): 29.65x / Rs +5,729,880  WR 42.3%  PF 1.94    "
          "post-Sep: Rs +1,554,471  WR 43.6%  PF 1.87")


if __name__ == "__main__":
    main()
