"""
Multi-day short-premium backtester (the literature's actual setup).

Unlike short_premium_backtest.py (intraday only), this holds positions for
1-7 days, letting theta accumulate. This is the framing under which
academic/practitioner literature claims 65-70% WR for short strangles.

Strategy:
  - Enter early in the week (Mon/Tue/Wed/Thu options)
  - Hold across nights, marking MTM bar-by-bar
  - Exit at:
      a. Stop-loss: position drawdown >= SL_MULT × credit received
      b. Profit target: capture TP_FRAC of credit
      c. Force exit: expiry day close (Tuesday post-Sep, Thursday before)
  - Gap-day risk modeled: MTM checked at every bar including next-day open

Variants tested:
  - Entry DOW: Mon, Tue, Wed, Thu, Fri (separately)
  - Strikes: 1 OTM, 2 OTM
  - Wings: bare strangle vs iron condor (1/3, 1/4)

Cost / pricing:
  - calc_premium() with our slippage model (0.5% + Rs 2 per leg)
  - Rs 30 per leg cost (Rs 60 RT for strangle, Rs 120 RT for condor)
  - LOT_SIZE = 75

Usage:
    python -m backtesting.multi_day_short_premium
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
COST_PER_LEG_RT = 30.0   # Rs 30 per leg one-way ≈ Rs 60 RT per leg
SL_MULT = 1.0            # SL at -100% of credit
TP_FRAC = 0.50           # TP at +50% of credit
ENTRY_BAR = 1            # 09:20 IST


def expiry_weekday(d):
    """Tuesday (1) post-Sep-2025, Thursday (3) before."""
    return 1 if d >= POST_SEP else 3


def next_expiry(d, all_dates_set):
    """Find the next expiry trading day on or after d."""
    target_wd = expiry_weekday(d)
    cand = d
    for _ in range(14):
        if cand.weekday() == target_wd and cand in all_dates_set:
            return cand
        cand = cand + dt.timedelta(days=1)
    return None


def round_atm(spot, step=50):
    return int(round(spot / step) * step)


def simulate_multi_day(entry_date, day_groups, vix_lookup, all_dates_set,
                       otm_strikes=1, iron_condor_wing=None,
                       strike_step=50):
    """Open position on entry_date, hold through to expiry. Return trade dict."""
    entry_bars = day_groups.get(entry_date)
    if not entry_bars or len(entry_bars) <= ENTRY_BAR + 5:
        return None

    expiry = next_expiry(entry_date, all_dates_set)
    if expiry is None:
        return None
    days_to_expiry = (expiry - entry_date).days
    if days_to_expiry < 0:
        return None
    dte_at_entry = max(0.1, float(days_to_expiry)) + 0.5  # half-day pad for entry time

    entry_bar_data = entry_bars[ENTRY_BAR]
    entry_spot = entry_bar_data["close"]
    atm = round_atm(entry_spot, strike_step)
    short_ce = atm + otm_strikes * strike_step
    short_pe = atm - otm_strikes * strike_step

    vix_at_entry = vix_lookup.get(entry_date, 14.0)

    ce_credit = calc_premium(entry_spot, short_ce, dte_at_entry, vix_at_entry, "CE", -1)
    pe_credit = calc_premium(entry_spot, short_pe, dte_at_entry, vix_at_entry, "PE", -1)

    long_ce = long_pe = None
    long_ce_debit = long_pe_debit = 0.0
    if iron_condor_wing:
        long_ce = atm + (otm_strikes + iron_condor_wing) * strike_step
        long_pe = atm - (otm_strikes + iron_condor_wing) * strike_step
        long_ce_debit = calc_premium(entry_spot, long_ce, dte_at_entry, vix_at_entry, "CE", +1)
        long_pe_debit = calc_premium(entry_spot, long_pe, dte_at_entry, vix_at_entry, "PE", +1)

    net_credit = (ce_credit + pe_credit) - (long_ce_debit + long_pe_debit)
    if net_credit <= 0:
        return None

    sl_threshold = -net_credit * SL_MULT * LOT_SIZE
    tp_threshold = +net_credit * TP_FRAC * LOT_SIZE

    # Build the chronological bar stream from entry_bar+1 through expiry day close
    cur_date = entry_date
    days_traversed = []
    while cur_date <= expiry:
        if cur_date in day_groups:
            bars = day_groups[cur_date]
            start_idx = (ENTRY_BAR + 1) if cur_date == entry_date else 0
            for bar_idx in range(start_idx, len(bars)):
                days_traversed.append((cur_date, bar_idx, bars[bar_idx]))
        cur_date = cur_date + dt.timedelta(days=1)

    if not days_traversed:
        return None

    exit_reason = "expiry_close"
    exit_date, exit_bar_idx, exit_bar = days_traversed[-1]
    exit_spot = exit_bar["close"]
    bars_held = len(days_traversed)

    for i, (d, bar_idx, bar) in enumerate(days_traversed):
        spot_now = bar["close"]
        elapsed_days = (d - entry_date).days + (bar_idx * 5) / 1440.0
        dte_now = max(0.01, dte_at_entry - elapsed_days)
        vix_now = vix_lookup.get(d, vix_at_entry)

        ce_now = calc_premium(spot_now, short_ce, dte_now, vix_now, "CE", +1)
        pe_now = calc_premium(spot_now, short_pe, dte_now, vix_now, "PE", +1)
        if iron_condor_wing:
            ce_long_now = calc_premium(spot_now, long_ce, dte_now, vix_now, "CE", -1)
            pe_long_now = calc_premium(spot_now, long_pe, dte_now, vix_now, "PE", -1)
            current_net = (ce_now + pe_now) - (ce_long_now + pe_long_now)
        else:
            current_net = ce_now + pe_now

        mtm = (net_credit - current_net) * LOT_SIZE

        if mtm <= sl_threshold:
            exit_reason = "stop_loss"
            exit_date, exit_bar_idx, exit_bar = d, bar_idx, bar
            exit_spot = spot_now
            bars_held = i + 1
            break
        if mtm >= tp_threshold:
            exit_reason = "target"
            exit_date, exit_bar_idx, exit_bar = d, bar_idx, bar
            exit_spot = spot_now
            bars_held = i + 1
            break

    # Final P&L at exit
    elapsed_days = (exit_date - entry_date).days + (exit_bar_idx * 5) / 1440.0
    dte_exit = max(0.01, dte_at_entry - elapsed_days)
    vix_exit = vix_lookup.get(exit_date, vix_at_entry)

    ce_exit = calc_premium(exit_spot, short_ce, dte_exit, vix_exit, "CE", +1)
    pe_exit = calc_premium(exit_spot, short_pe, dte_exit, vix_exit, "PE", +1)
    n_legs = 2
    if iron_condor_wing:
        ce_long_exit = calc_premium(exit_spot, long_ce, dte_exit, vix_exit, "CE", -1)
        pe_long_exit = calc_premium(exit_spot, long_pe, dte_exit, vix_exit, "PE", -1)
        net_exit = (ce_exit + pe_exit) - (ce_long_exit + pe_long_exit)
        n_legs = 4
    else:
        net_exit = ce_exit + pe_exit

    pnl_per_lot = (net_credit - net_exit) * LOT_SIZE
    cost = n_legs * COST_PER_LEG_RT
    pnl = pnl_per_lot - cost

    return {
        "entry_date": entry_date, "exit_date": exit_date,
        "expiry": expiry, "dte_at_entry": dte_at_entry,
        "entry_spot": entry_spot, "exit_spot": exit_spot,
        "atm": atm, "short_ce": short_ce, "short_pe": short_pe,
        "long_ce": long_ce, "long_pe": long_pe,
        "vix_at_entry": vix_at_entry,
        "net_credit": net_credit, "net_exit": net_exit,
        "exit_reason": exit_reason, "bars_held": bars_held,
        "days_held": (exit_date - entry_date).days,
        "pnl": pnl,
    }


def run_strategy(label, entry_dows, otm_strikes, iron_condor_wing,
                 day_groups, vix_lookup, all_dates,
                 vix_floor=None, vix_ceil=None):
    all_set = set(all_dates)
    trades = []
    for d in all_dates:
        if d.weekday() not in entry_dows:
            continue
        # Skip days that are themselves expiry (no point opening on expiry)
        if d.weekday() == expiry_weekday(d):
            continue
        vix = vix_lookup.get(d, 14.0)
        if vix_floor is not None and vix < vix_floor:
            continue
        if vix_ceil is not None and vix > vix_ceil:
            continue
        result = simulate_multi_day(d, day_groups, vix_lookup, all_set,
                                    otm_strikes=otm_strikes,
                                    iron_condor_wing=iron_condor_wing)
        if result is None:
            continue
        trades.append(result)
    return label, trades


def metrics(trades):
    if not trades:
        return {"n": 0, "pnl": 0.0, "wr": 0.0, "pf": 0.0, "exits": {},
                "avg_win": 0.0, "avg_loss": 0.0, "max_win": 0.0, "max_loss": 0.0,
                "avg_days": 0.0}
    pnls = [t["pnl"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    tw = sum(wins)
    tl = -sum(losses)
    pf = tw / tl if tl > 0 else float("inf")
    exits = defaultdict(int)
    for t in trades:
        exits[t["exit_reason"]] += 1
    return {
        "n": len(trades), "pnl": sum(pnls),
        "wr": len(wins) / len(trades) * 100, "pf": pf,
        "avg_win": tw / max(1, len(wins)),
        "avg_loss": -tl / max(1, len(losses)),
        "max_win": max(pnls), "max_loss": min(pnls),
        "exits": dict(exits),
        "avg_days": sum(t.get("days_held", 0) for t in trades) / len(trades),
    }


def split(trades, cutover):
    pre, post = [], []
    for t in trades:
        d = t.get("entry_date")
        (pre if d < cutover else post).append(t)
    return pre, post


def main():
    start = dt.date(2024, 7, 1)
    end = dt.date(2026, 4, 6)
    print(f"Loading period data {start} -> {end} ...", flush=True)
    preloaded = load_period_data(start_date=start, end_date=end, quiet=True)
    day_groups, vix_lookup, all_dates, _ = preloaded
    print(f"  {len(all_dates)} trading days\n", flush=True)

    # ── Variants: (label, entry_dows, otm, iron_condor_wing, vfloor, vceil) ──
    variants = [
        # Entry DOW sweep (1OTM strangle, V17 VIX gates)
        ("STRANGLE 1OTM   Mon entry          ", [0],     1, None, 12, 25),
        ("STRANGLE 1OTM   Tue entry          ", [1],     1, None, 12, 25),
        ("STRANGLE 1OTM   Wed entry          ", [2],     1, None, 12, 25),
        ("STRANGLE 1OTM   Thu entry          ", [3],     1, None, 12, 25),
        ("STRANGLE 1OTM   Fri entry          ", [4],     1, None, 12, 25),
        ("STRANGLE 1OTM   Mon+Wed entry      ", [0, 2],  1, None, 12, 25),
        ("STRANGLE 1OTM   Mon+Thu entry      ", [0, 3],  1, None, 12, 25),
        # Strike-OTM sweep
        ("STRANGLE 2OTM   Wed entry          ", [2],     2, None, 12, 25),
        ("STRANGLE 3OTM   Wed entry          ", [2],     3, None, 12, 25),
        # Iron Condor (defined risk) — best DOW
        ("CONDOR 1/3      Wed entry          ", [2],     1, 2,    12, 25),
        ("CONDOR 1/3      Mon entry          ", [0],     1, 2,    12, 25),
        ("CONDOR 1/4      Wed entry          ", [2],     1, 3,    12, 25),
        # No-VIX-gate variants (does VIX gating help here?)
        ("STRANGLE 1OTM   Wed entry no VIX   ", [2],     1, None, None, None),
        ("CONDOR 1/3      Wed entry no VIX   ", [2],     1, 2,    None, None),
    ]

    print(f"{'variant':38s} {'n':>4s}  {'PnL':>14s}  {'x':>6s}  {'WR':>5s}  {'PF':>5s}  "
          f"{'avg days':>9s}    {'post-Sep n':>10s}  {'PnL':>12s}  {'WR':>5s}  {'PF':>5s}    exits")
    print("-" * 175)

    rows = []
    for v in variants:
        label, dows, otm, wing, vfl, vce = v
        _, trades = run_strategy(label, dows, otm, wing,
                                 day_groups, vix_lookup, all_dates,
                                 vix_floor=vfl, vix_ceil=vce)
        full = metrics(trades)
        _, post = split(trades, POST_SEP)
        ps = metrics(post)
        ret_x = (CAPITAL + full["pnl"]) / CAPITAL
        exits_short = ", ".join(f"{r}={n}" for r, n in full["exits"].items())
        rows.append((label, full, ps, ret_x))
        print(f"  {label:36s} {full['n']:>4d}  Rs {full['pnl']:>+12,.0f}  "
              f"{ret_x:>5.2f}x  {full['wr']:>4.1f}%  {full['pf']:>4.2f}  "
              f"{full['avg_days']:>7.1f}d    "
              f"{ps['n']:>10d}  Rs {ps['pnl']:>+10,.0f}  "
              f"{ps['wr']:>4.1f}%  {ps['pf']:>4.2f}    {exits_short}")

    print()
    print("=" * 175)
    print("RANKED BY POST-SEP PROFIT FACTOR (quality, not gross)")
    print("=" * 175)
    for label, full, ps, ret_x in sorted(
        rows, key=lambda r: -(r[2]["pf"] if r[2]["n"] >= 5 else -1)
    ):
        if ps["n"] < 5:
            continue
        print(f"  {label:38s} post-Sep PF={ps['pf']:.2f}  PnL=Rs {ps['pnl']:>+12,.0f}  "
              f"n={ps['n']:>3d}  WR={ps['wr']:.1f}%   "
              f"full: PF={full['pf']:.2f}  Rs {full['pnl']:>+12,.0f} ({ret_x:.2f}x)")

    print()
    print("=" * 175)
    print("REFERENCE: V17 buying (Option B deployed)")
    print("=" * 175)
    print("  V17 BUYING (deployed):    29.65x / Rs +5,729,880  WR 42.3%  PF 1.94    "
          "post-Sep: Rs +1,554,471  WR 43.6%  PF 1.87")


if __name__ == "__main__":
    main()
