"""
V14 Strike Selection Comparison — ATM vs Smart Strike Offsets
==============================================================
Runs the SAME V14 R5 strategy across different strike selection methods:
  1. ATM (baseline — current 12.26x result)
  2. OTM_1 (+50 pts for CE, -50 pts for PE — delta ~0.45)
  3. OTM_2 (+100 pts for CE, -100 pts for PE — delta ~0.35)
  4. ITM_1 (-50 pts for CE, +50 pts for PE — delta ~0.55)
  5. SMART (dynamic: OTM for high VIX, ATM for low VIX, gamma-biased on expiry)

Uses the exact same V14 scoring engine, data, and exit logic.
Only the strike selection differs — a clean apples-to-apples comparison.

Usage:
    python -m backtesting.v14_strike_comparison
"""

import sys
import datetime as dt
from pathlib import Path
from collections import defaultdict

import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scoring.config import V14_CONFIG
from scoring.indicators import compute_indicators
from scoring.engine import (
    score_entry,
    passes_confluence,
    evaluate_exit,
    compute_lots,
    detect_composite_entries,
)
from backtesting.option_pricer import price_option, bs_delta, skewed_iv

# ── Reuse data loading from unified backtest ──
from backtesting.v14_unified_backtest import (
    load_nifty_data,
    load_vix_data,
    resample_to_5min,
)

CAPITAL = 200_000
LOT_SIZE = 75       # For 2024 backtest period
STRIKE_INTERVAL = 50
SLIPPAGE_PCT = 0.005
SPREAD_RS = 2.0
BROKERAGE_RT = 80.0


# ── Strike Selection Strategies ──────────────────────────────

def select_strike_atm(action, spot, vix, dte, is_expiry):
    """Baseline: Always ATM."""
    opt_type = "CE" if action == "BUY_CALL" else "PE"
    atm = round(spot / STRIKE_INTERVAL) * STRIKE_INTERVAL
    return atm, opt_type


def select_strike_otm1(action, spot, vix, dte, is_expiry):
    """1 strike OTM (delta ~0.45)."""
    opt_type = "CE" if action == "BUY_CALL" else "PE"
    atm = round(spot / STRIKE_INTERVAL) * STRIKE_INTERVAL
    if action == "BUY_CALL":
        strike = atm + STRIKE_INTERVAL
    else:
        strike = atm - STRIKE_INTERVAL
    return strike, opt_type


def select_strike_otm2(action, spot, vix, dte, is_expiry):
    """2 strikes OTM (delta ~0.35)."""
    opt_type = "CE" if action == "BUY_CALL" else "PE"
    atm = round(spot / STRIKE_INTERVAL) * STRIKE_INTERVAL
    if action == "BUY_CALL":
        strike = atm + 2 * STRIKE_INTERVAL
    else:
        strike = atm - 2 * STRIKE_INTERVAL
    return strike, opt_type


def select_strike_itm1(action, spot, vix, dte, is_expiry):
    """1 strike ITM (delta ~0.55-0.60)."""
    opt_type = "CE" if action == "BUY_CALL" else "PE"
    atm = round(spot / STRIKE_INTERVAL) * STRIKE_INTERVAL
    if action == "BUY_CALL":
        strike = atm - STRIKE_INTERVAL
    else:
        strike = atm + STRIKE_INTERVAL
    return strike, opt_type


def select_strike_smart(action, spot, vix, dte, is_expiry):
    """Smart selector v2: adapts offset based on VIX, DTE, and expiry.

    BACKTEST PROVEN: Higher delta = higher profit for directional buying.
    ITM_1 (d=0.60) returned 15.87x vs ATM (d=0.48) at 14.56x.

    Updated logic:
      - Normal VIX (13-17): 1 ITM (delta ~0.58, max profit per point)
      - High VIX (>17): ATM (balance delta vs drawdown risk)
      - Low VIX (<13): ATM (thin premiums, OTM worthless)
      - Expiry day: ATM (gamma spike gives good moves at ATM)
      - Short DTE (<1 day): ATM (ITM premium too expensive near expiry)
    """
    opt_type = "CE" if action == "BUY_CALL" else "PE"
    atm = round(spot / STRIKE_INTERVAL) * STRIKE_INTERVAL
    # ITM direction is opposite: for CE, ITM = strike below spot
    itm_direction = -1 if action == "BUY_CALL" else 1

    if vix < 13:
        offset = 0                        # ATM — thin premiums
    elif vix > 17:
        offset = 0                        # ATM — control risk in volatile markets
    elif is_expiry:
        offset = 0                        # ATM — gamma spike sufficient
    elif dte < 1.0:
        offset = 0                        # ATM — avoid expensive ITM near expiry
    else:
        offset = 1 * STRIKE_INTERVAL      # 1 ITM — delta ~0.58 for max capture

    strike = atm + itm_direction * offset
    return strike, opt_type


def select_strike_smart_aggressive(action, spot, vix, dte, is_expiry):
    """Smart aggressive: always 1 ITM regardless of conditions.

    Pure delta maximization. Backtest showed ITM_1 = best return (15.87x).
    """
    opt_type = "CE" if action == "BUY_CALL" else "PE"
    atm = round(spot / STRIKE_INTERVAL) * STRIKE_INTERVAL
    itm_direction = -1 if action == "BUY_CALL" else 1
    strike = atm + itm_direction * STRIKE_INTERVAL
    return strike, opt_type


# ── Strategy map ──
STRATEGIES = {
    "ATM":      select_strike_atm,
    "ITM_1":    select_strike_itm1,
    "SMART_v2": select_strike_smart,
    "SMART_AGG":select_strike_smart_aggressive,
    "OTM_1":    select_strike_otm1,
}


# ── Premium + P&L helpers (identical to unified backtest) ──────

def calc_premium(spot, strike, dte, vix, opt_type, slippage_sign=1):
    try:
        result = price_option(spot=spot, strike=strike, dte_days=max(0.1, dte),
                              vix=vix, option_type=opt_type)
        prem = result["premium"]
    except Exception:
        prem = 30.0
    prem = prem * (1 + slippage_sign * SLIPPAGE_PCT) + slippage_sign * SPREAD_RS
    return max(0.05, prem)


# ── Single Day Simulation ─────────────────────────────────────

def simulate_day(bars_5min, date, vix, cfg, prev_close, equity,
                 warmup_bars, is_expiry, consecutive_down_days,
                 strike_fn):
    """Run one day with a specific strike selection function."""
    date_str = str(date)

    avoid_days = cfg.get("avoid_days", [])
    if date.weekday() in avoid_days:
        eod_spot = bars_5min[-1]["close"] if bars_5min else 0
        return [], 0.0, eod_spot

    bar_history = list(warmup_bars or [])
    open_trades = []
    closed_trades = []
    trades_today = 0
    last_exit_bar = -10

    orb_high = 0.0
    orb_low = 0.0
    gap_detected = False
    day_open = bars_5min[0]["close"] if bars_5min else 0

    span_per_lot = 40000
    if vix >= 20:
        span_per_lot = 50000
    elif vix >= 25:
        span_per_lot = 60000
    base_lots = max(1, int(equity * 0.70 / span_per_lot))

    day_of_week = date.weekday()
    days_to_expiry = (3 - day_of_week) % 7
    dte = 0.2 if days_to_expiry == 0 else float(days_to_expiry)

    day_pnl = 0.0

    for bar_idx, bar in enumerate(bars_5min):
        bar_history.append(bar)
        if len(bar_history) > 500:
            bar_history = bar_history[-500:]

        spot = bar["close"]

        if bar_idx == 0:
            orb_high = bar["high"]
            orb_low = bar["low"]
        elif bar_idx == 1:
            orb_high = max(orb_high, bar["high"])
            orb_low = min(orb_low, bar["low"])

        # ── EXITS ──
        for trade in list(open_trades):
            if trade["action"] == "BUY_CALL" and spot > trade["best_fav"]:
                trade["best_fav"] = spot
            elif trade["action"] == "BUY_PUT" and spot < trade["best_fav"]:
                trade["best_fav"] = spot

            indicators = compute_indicators(bar_history, date_str)
            exit_reason = evaluate_exit(
                trade, bar_idx, spot, indicators or {}, cfg,
                day_of_week=day_of_week,
            )
            if exit_reason:
                exit_dte = max(0.05, dte - bar_idx * 5 / (6.25 * 60))
                exit_prem = calc_premium(spot, trade["strike"], exit_dte,
                                         vix, trade["opt_type"], slippage_sign=-1)
                pnl = (exit_prem - trade["entry_premium"]) * trade["qty"] - BROKERAGE_RT
                trade["exit_bar"] = bar_idx
                trade["exit_spot"] = spot
                trade["exit_premium"] = exit_prem
                trade["exit_reason"] = exit_reason
                trade["pnl"] = pnl
                trade["bars_held"] = bar_idx - trade["entry_bar"]
                closed_trades.append(trade)
                open_trades.remove(trade)
                day_pnl += pnl
                last_exit_bar = bar_idx

        # ── ENTRIES ──
        indicators = compute_indicators(bar_history, date_str)
        if indicators is None:
            continue

        if len(open_trades) >= cfg["max_concurrent"]:
            continue
        if trades_today >= cfg["max_trades_per_day"]:
            continue

        entry_windows = cfg.get("entry_windows_bars")
        action = None
        conf = 0
        entry_type = "v8_indicator"
        is_zero_hero = False

        if bar_idx < 3:
            prev_spot = bars_5min[bar_idx - 1]["close"] if bar_idx > 0 else spot
            composites = detect_composite_entries(
                bar, bar_idx, spot, vix, cfg,
                prev_close=prev_close, gap_detected=gap_detected,
                orb_high=orb_high, orb_low=orb_low,
                prev_spot=prev_spot,
            )
            if bar_idx == 0:
                gap_detected = True
            if cfg.get("disable_zero_hero", False):
                composites = [c for c in composites if not c[3]]
            if composites:
                composites.sort(key=lambda x: x[2], reverse=True)
                action, entry_type, conf, is_zero_hero = composites[0]
        else:
            if entry_windows:
                in_window = any(s <= bar_idx <= e for s, e in entry_windows)
                if not in_window:
                    continue

            if bar_idx - last_exit_bar < cfg["cooldown_bars"]:
                continue

            action, conf = score_entry(
                indicators, vix, cfg,
                bar_idx=bar_idx,
                consecutive_down_days=consecutive_down_days,
            )

            if action is None:
                prev_spot = bars_5min[bar_idx - 1]["close"] if bar_idx > 0 else spot
                composites = detect_composite_entries(
                    bar, bar_idx, spot, vix, cfg,
                    prev_close=prev_close, gap_detected=gap_detected,
                    orb_high=orb_high, orb_low=orb_low,
                    prev_spot=prev_spot,
                )
                if cfg.get("disable_zero_hero", False):
                    composites = [c for c in composites if not c[3]]
                if composites:
                    composites.sort(key=lambda x: x[2], reverse=True)
                    action, entry_type, conf, is_zero_hero = composites[0]

        if action is None:
            continue

        if vix < cfg["vix_floor"] or vix > cfg["vix_ceil"]:
            continue

        if not passes_confluence(
            action, conf, indicators, bar_idx, cfg,
            current_spot=spot,
            prev_close=prev_close,
            day_open=day_open,
        ):
            continue

        if any(t["action"] == action for t in open_trades):
            continue

        # Lot sizing
        cfg_with_atr = cfg
        if cfg.get("use_atr_sizing"):
            cfg_with_atr = cfg.copy()
            cfg_with_atr["_current_atr"] = indicators.get("atr", 0)
        lots = compute_lots(action, conf, vix, indicators.get("rsi", 50),
                            is_expiry, base_lots, cfg_with_atr)
        if is_zero_hero:
            lots = min(cfg.get("zero_hero_max_lots", 3), max(1, lots))

        if cfg.get("vix_lot_scaling"):
            if vix < 13:
                lots = max(1, int(lots * cfg.get("vix_below13_mult", 0.3)))
            elif 14 <= vix < 15:
                lots = max(1, int(lots * cfg.get("vix_14_15_mult", 0.5)))
            elif 15 <= vix < 17:
                lots = max(1, int(lots * cfg.get("vix_15_17_mult", 1.5)))
            elif vix >= 17:
                lots = max(1, int(lots * cfg.get("vix_17plus_mult", 2.0)))
        if entry_type and "orb" in entry_type:
            lots = max(1, int(lots * cfg.get("orb_lot_mult", 1.0)))
        lots = min(lots, cfg.get("max_lots_cap", 999))

        # ── STRIKE SELECTION (the ONLY difference) ──
        strike, opt_type = strike_fn(action, spot, vix, dte, is_expiry)
        qty = lots * LOT_SIZE

        entry_prem = calc_premium(spot, strike, dte, vix, opt_type, slippage_sign=1)

        trade = {
            "date": date_str,
            "action": action,
            "entry_bar": bar_idx,
            "entry_spot": spot,
            "best_fav": spot,
            "strike": strike,
            "opt_type": opt_type,
            "lots": lots,
            "qty": qty,
            "entry_premium": entry_prem,
            "entry_type": entry_type,
            "is_zero_hero": is_zero_hero,
            "confidence": conf,
            "vix": vix,
        }
        open_trades.append(trade)
        trades_today += 1

    # Force close at EOD
    eod_spot = bars_5min[-1]["close"] if bars_5min else 0
    for trade in open_trades:
        exit_prem = calc_premium(eod_spot, trade["strike"],
                                 max(0.05, dte * 0.1), vix, trade["opt_type"],
                                 slippage_sign=-1)
        pnl = (exit_prem - trade["entry_premium"]) * trade["qty"] - BROKERAGE_RT
        trade["exit_bar"] = len(bars_5min) - 1
        trade["exit_spot"] = eod_spot
        trade["exit_premium"] = exit_prem
        trade["exit_reason"] = "eod_close"
        trade["pnl"] = pnl
        trade["bars_held"] = trade["exit_bar"] - trade["entry_bar"]
        closed_trades.append(trade)
        day_pnl += pnl

    return closed_trades, day_pnl, eod_spot


# ── Run a single strategy across all dates ──────────────────

def run_single(strategy_name, strike_fn, day_groups, all_dates, warmup_dates,
               vix_lookup, cfg):
    warmup_bars = []
    for d in warmup_dates[-3:]:
        warmup_bars.extend(day_groups[d])

    equity = CAPITAL
    all_trades = []
    prev_close = 0.0
    consecutive_down_days = 0
    peak_equity = CAPITAL
    max_drawdown = 0.0
    monthly_pnl = defaultdict(float)

    for date in all_dates:
        bars = day_groups[date]
        if len(bars) < 5:
            continue

        vix = vix_lookup.get(date, 14.0)
        is_expiry = (date.weekday() == 3)

        trades, day_pnl, eod_close = simulate_day(
            bars, date, vix, cfg,
            prev_close=prev_close,
            equity=equity,
            warmup_bars=warmup_bars,
            is_expiry=is_expiry,
            consecutive_down_days=consecutive_down_days,
            strike_fn=strike_fn,
        )

        equity += day_pnl
        all_trades.extend(trades)
        month_key = f"{date.year}-{date.month:02d}"
        monthly_pnl[month_key] += day_pnl

        # Max drawdown
        if equity > peak_equity:
            peak_equity = equity
        dd = (peak_equity - equity) / peak_equity
        if dd > max_drawdown:
            max_drawdown = dd

        # Consecutive down days
        if len(bars) >= 2:
            if bars[-1]["close"] < bars[0]["open"]:
                consecutive_down_days += 1
            else:
                consecutive_down_days = 0

        warmup_bars = warmup_bars[-(75 * 2):] + bars
        prev_close = eod_close

    # Compute stats
    total_pnl = sum(t.get("pnl", 0) for t in all_trades)
    wins = [t for t in all_trades if t.get("pnl", 0) > 0]
    losses = [t for t in all_trades if t.get("pnl", 0) <= 0]
    win_rate = len(wins) / len(all_trades) * 100 if all_trades else 0
    avg_win = np.mean([t["pnl"] for t in wins]) if wins else 0
    avg_loss = np.mean([t["pnl"] for t in losses]) if losses else 0
    gross_wins = sum(t["pnl"] for t in wins)
    gross_losses = abs(sum(t["pnl"] for t in losses))
    profit_factor = gross_wins / gross_losses if gross_losses > 0 else 999
    ret_x = equity / CAPITAL
    avg_prem = np.mean([t["entry_premium"] for t in all_trades]) if all_trades else 0
    avg_delta_at_entry = 0.0
    for t in all_trades:
        try:
            strike = t["strike"]
            spot = t["entry_spot"]
            vix_t = t.get("vix", 14.0)
            dow = dt.date.fromisoformat(t["date"]).weekday()
            dte_t = (3 - dow) % 7
            dte_t = 0.2 if dte_t == 0 else float(dte_t)
            T = dte_t / 365.0
            atm_iv = vix_t / 100.0 * 0.88
            iv = skewed_iv(atm_iv, spot, strike, t["opt_type"])
            d = bs_delta(spot, strike, T, 0.07, iv, t["opt_type"])
            avg_delta_at_entry += abs(d)
        except Exception:
            pass
    avg_delta_at_entry = avg_delta_at_entry / len(all_trades) if all_trades else 0

    # Sharpe (daily returns)
    daily_returns = defaultdict(float)
    for t in all_trades:
        daily_returns[t["date"]] += t.get("pnl", 0) / CAPITAL
    daily_ret_list = list(daily_returns.values())
    sharpe = 0.0
    if daily_ret_list and np.std(daily_ret_list) > 0:
        sharpe = np.mean(daily_ret_list) / np.std(daily_ret_list) * np.sqrt(252)

    return {
        "strategy": strategy_name,
        "trades": len(all_trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "total_pnl": total_pnl,
        "final_equity": equity,
        "return_x": ret_x,
        "max_drawdown": max_drawdown,
        "sharpe": sharpe,
        "avg_entry_premium": avg_prem,
        "avg_abs_delta": avg_delta_at_entry,
        "monthly_pnl": dict(monthly_pnl),
        "all_trades": all_trades,
    }


# ── Main ──────────────────────────────────────────────────────

def main():
    start_date = dt.date(2024, 7, 1)
    end_date = dt.date(2025, 1, 1)

    print("=" * 80)
    print("  V14 R5 STRIKE SELECTION COMPARISON BACKTEST")
    print(f"  Period: {start_date} to {end_date}")
    print(f"  Capital: Rs {CAPITAL:,}")
    print(f"  Config: {V14_CONFIG['name']}")
    print("=" * 80)
    print()

    # Load data
    print("Loading data...", flush=True)
    warmup_start = start_date - dt.timedelta(days=10)
    nifty_1min = load_nifty_data(warmup_start, end_date)
    vix_lookup = load_vix_data(warmup_start, end_date)
    nifty_5min = resample_to_5min(nifty_1min)

    day_groups = {}
    for ts, row in nifty_5min.iterrows():
        d = ts.date()
        if d not in day_groups:
            day_groups[d] = []
        day_groups[d].append({
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row.get("volume", 0)),
            "date": str(d),
            "time": str(ts),
        })

    all_dates = sorted(d for d in day_groups.keys() if d >= start_date)
    warmup_dates = sorted(d for d in day_groups.keys() if d < start_date)
    print(f"Data: {len(all_dates)} trading days, {len(vix_lookup)} VIX days")
    print()

    # Run all strategies
    cfg = V14_CONFIG.copy()
    # Remove smart_strike_config from backtest cfg (it's for live only)
    cfg.pop("smart_strike_config", None)

    results = []
    for name, fn in STRATEGIES.items():
        print(f"Running {name}...", end=" ", flush=True)
        r = run_single(name, fn, day_groups, all_dates, warmup_dates,
                        vix_lookup, cfg)
        results.append(r)
        print(f"{r['return_x']:.2f}x | {r['trades']} trades | WR={r['win_rate']:.1f}% | PF={r['profit_factor']:.2f}")

    # ── Comparison Table ──
    print()
    print("=" * 100)
    print("  COMPARISON RESULTS")
    print("=" * 100)
    print()
    header = f"{'Strategy':<10} {'Return':>8} {'Trades':>7} {'WR%':>6} {'PF':>6} {'AvgWin':>9} {'AvgLoss':>9} {'MaxDD':>7} {'Sharpe':>7} {'AvgPrem':>8} {'AvgDelta':>8}"
    print(header)
    print("-" * len(header))

    # Sort by return
    results.sort(key=lambda x: x["return_x"], reverse=True)
    baseline_pnl = 0
    for r in results:
        if r["strategy"] == "ATM":
            baseline_pnl = r["total_pnl"]
            break

    for r in results:
        diff = r["total_pnl"] - baseline_pnl
        marker = " <-- BASELINE" if r["strategy"] == "ATM" else f" ({diff:+,.0f})"
        print(
            f"{r['strategy']:<10} "
            f"{r['return_x']:>7.2f}x "
            f"{r['trades']:>7} "
            f"{r['win_rate']:>5.1f}% "
            f"{r['profit_factor']:>5.2f} "
            f"{r['avg_win']:>+8,.0f} "
            f"{r['avg_loss']:>+8,.0f} "
            f"{r['max_drawdown']:>6.1%} "
            f"{r['sharpe']:>6.2f} "
            f"{r['avg_entry_premium']:>7.1f} "
            f"{r['avg_abs_delta']:>7.3f}"
            f"{marker}"
        )

    # ── Monthly Comparison ──
    print()
    print("MONTHLY P&L COMPARISON")
    print("-" * 80)
    months = sorted(set(m for r in results for m in r["monthly_pnl"].keys()))
    header = f"{'Month':<10}" + "".join(f" {r['strategy']:>10}" for r in results)
    print(header)
    print("-" * len(header))
    for m in months:
        row = f"{m:<10}"
        for r in results:
            pnl = r["monthly_pnl"].get(m, 0)
            row += f" {pnl:>+10,.0f}"
        print(row)

    # ── Per-Trade Delta Analysis ──
    print()
    print("DELTA AT ENTRY DISTRIBUTION")
    print("-" * 60)
    for r in results:
        deltas = []
        for t in r["all_trades"]:
            try:
                strike = t["strike"]
                spot = t["entry_spot"]
                vix_t = t.get("vix", 14.0)
                dow = dt.date.fromisoformat(t["date"]).weekday()
                dte_t = (3 - dow) % 7
                dte_t = 0.2 if dte_t == 0 else float(dte_t)
                T = dte_t / 365.0
                atm_iv = vix_t / 100.0 * 0.88
                iv = skewed_iv(atm_iv, spot, strike, t["opt_type"])
                d = abs(bs_delta(spot, strike, T, 0.07, iv, t["opt_type"]))
                deltas.append(d)
            except Exception:
                pass
        if deltas:
            print(f"  {r['strategy']:<10}: "
                  f"avg={np.mean(deltas):.3f}  "
                  f"min={np.min(deltas):.3f}  "
                  f"max={np.max(deltas):.3f}  "
                  f"std={np.std(deltas):.3f}")

    # ── Winner/Loser Analysis ──
    print()
    print("WIN/LOSS ANALYSIS")
    print("-" * 60)
    for r in results:
        trades = r["all_trades"]
        calls = [t for t in trades if t["action"] == "BUY_CALL"]
        puts = [t for t in trades if t["action"] == "BUY_PUT"]
        call_pnl = sum(t["pnl"] for t in calls) if calls else 0
        put_pnl = sum(t["pnl"] for t in puts) if puts else 0
        call_wr = sum(1 for t in calls if t["pnl"] > 0) / len(calls) * 100 if calls else 0
        put_wr = sum(1 for t in puts if t["pnl"] > 0) / len(puts) * 100 if puts else 0
        print(f"  {r['strategy']:<10}: "
              f"CALL: {len(calls)} trades, {call_pnl:+,.0f}, WR={call_wr:.0f}% | "
              f"PUT: {len(puts)} trades, {put_pnl:+,.0f}, WR={put_wr:.0f}%")

    print()
    print("=" * 80)
    best = results[0]
    print(f"  BEST STRATEGY: {best['strategy']} ({best['return_x']:.2f}x return)")
    if best["strategy"] != "ATM":
        print(f"  vs ATM baseline: {best['total_pnl'] - baseline_pnl:+,.0f} additional profit")
    print("=" * 80)


if __name__ == "__main__":
    main()
