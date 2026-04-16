"""
V16b — Position Sizing & Hold Time Optimization
================================================
Tests:
  1. Extended hold times (let everything ride to EOD)
  2. Equity curve sizing (reduce lots during drawdown)
  3. Tighter daily loss cap
  4. VIX 13-14 zone: block entirely instead of 0.3x

These modify SIZING and HOLD TIME — not entry scoring or confluence.

Usage:
    python -m backtesting.v16b_sizing_test
"""

import sys
import datetime as dt
from pathlib import Path
from collections import defaultdict

import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scoring.config import V15_CONFIG
from scoring.indicators import compute_indicators
from scoring.engine import (
    score_entry,
    passes_confluence,
    evaluate_exit,
    compute_lots,
    detect_composite_entries,
)
from backtesting.option_pricer import price_option

from backtesting.v14_unified_backtest import (
    load_nifty_data,
    load_vix_data,
    resample_to_5min,
)

CAPITAL = 200_000
LOT_SIZE = 75
STRIKE_INTERVAL = 50
SLIPPAGE_PCT = 0.005
SPREAD_RS = 2.0
BROKERAGE_RT = 80.0


def select_strike_itm1(action, spot, vix, dte, is_expiry):
    opt_type = "CE" if action == "BUY_CALL" else "PE"
    atm = round(spot / STRIKE_INTERVAL) * STRIKE_INTERVAL
    if action == "BUY_CALL":
        strike = atm - STRIKE_INTERVAL
    else:
        strike = atm + STRIKE_INTERVAL
    return strike, opt_type


def calc_premium(spot, strike, dte, vix, opt_type, slippage_sign=1):
    try:
        result = price_option(spot=spot, strike=strike, dte_days=max(0.1, dte),
                              vix=vix, option_type=opt_type)
        prem = result["premium"]
    except Exception:
        prem = 30.0
    prem = prem * (1 + slippage_sign * SLIPPAGE_PCT) + slippage_sign * SPREAD_RS
    return max(0.05, prem)


def simulate_day(bars_5min, date, vix, cfg, prev_close, equity,
                 warmup_bars, is_expiry, consecutive_down_days,
                 peak_equity, recent_losses):
    """Run one day with equity-curve-aware sizing."""
    date_str = str(date)
    day_of_week = date.weekday()

    avoid_days = cfg.get("avoid_days", [])
    if day_of_week in avoid_days:
        eod_spot = bars_5min[-1]["close"] if bars_5min else 0
        return [], 0.0, eod_spot

    bar_history = list(warmup_bars or [])
    open_trades = []
    closed_trades = []
    trades_today = 0
    last_exit_bar = -10
    day_loss = 0.0

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

    # ── Equity curve sizing ──
    if cfg.get("use_equity_curve_sizing") and peak_equity > 0:
        dd_pct = (peak_equity - equity) / peak_equity
        if dd_pct > 0.30:
            base_lots = max(1, int(base_lots * 0.5))
        elif dd_pct > 0.15:
            base_lots = max(1, int(base_lots * 0.7))

    # ── Losing streak sizing ──
    if cfg.get("use_streak_sizing") and recent_losses >= 3:
        streak_mult = max(0.4, 1.0 - recent_losses * 0.15)
        base_lots = max(1, int(base_lots * streak_mult))

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
                if pnl < 0:
                    day_loss += abs(pnl)
                last_exit_bar = bar_idx

        # ── Daily loss cap ──
        daily_cap = cfg.get("daily_loss_cap_pct", 0.0)
        if daily_cap > 0 and day_loss > equity * daily_cap:
            continue  # Stop entering new trades today

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
        cfg_with_atr = cfg.copy()
        if cfg.get("use_atr_sizing"):
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

        strike, opt_type = select_strike_itm1(action, spot, vix, dte, is_expiry)
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
            "day_of_week": day_of_week,
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


def run_config(config_name, cfg, day_groups, all_dates, warmup_dates,
               vix_lookup):
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
    recent_losses = 0  # Consecutive losing trades

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
            peak_equity=peak_equity,
            recent_losses=recent_losses,
        )

        equity += day_pnl
        all_trades.extend(trades)
        month_key = f"{date.year}-{date.month:02d}"
        monthly_pnl[month_key] += day_pnl

        # Track consecutive losses
        for t in trades:
            if t.get("pnl", 0) > 0:
                recent_losses = 0
            else:
                recent_losses += 1

        if equity > peak_equity:
            peak_equity = equity
        dd = (peak_equity - equity) / peak_equity
        if dd > max_drawdown:
            max_drawdown = dd

        if len(bars) >= 2:
            if bars[-1]["close"] < bars[0]["open"]:
                consecutive_down_days += 1
            else:
                consecutive_down_days = 0

        warmup_bars = warmup_bars[-(75 * 2):] + bars
        prev_close = eod_close

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

    daily_returns = defaultdict(float)
    for t in all_trades:
        daily_returns[t["date"]] += t.get("pnl", 0) / CAPITAL
    daily_ret_list = list(daily_returns.values())
    sharpe = 0.0
    if daily_ret_list and np.std(daily_ret_list) > 0:
        sharpe = np.mean(daily_ret_list) / np.std(daily_ret_list) * np.sqrt(252)

    exit_reasons = defaultdict(lambda: {"count": 0, "pnl": 0.0, "wins": 0})
    for t in all_trades:
        r = t.get("exit_reason", "unknown")
        exit_reasons[r]["count"] += 1
        exit_reasons[r]["pnl"] += t.get("pnl", 0)
        if t.get("pnl", 0) > 0:
            exit_reasons[r]["wins"] += 1

    calmar = (ret_x - 1) / max_drawdown if max_drawdown > 0 else 0

    return {
        "config": config_name,
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
        "calmar": calmar,
        "monthly_pnl": dict(monthly_pnl),
        "exit_reasons": dict(exit_reasons),
        "all_trades": all_trades,
    }


def main():
    start_date = dt.date(2024, 7, 1)
    end_date = dt.date(2025, 1, 1)

    print("=" * 90)
    print("  V16b — POSITION SIZING & HOLD TIME OPTIMIZATION")
    print(f"  Period: {start_date} to {end_date}")
    print(f"  Capital: Rs {CAPITAL:,}")
    print("=" * 90)
    print()

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

    # ── Configs to compare ──
    configs = {}

    # Baseline: V15
    configs["V15_Base"] = V15_CONFIG.copy()

    # Test 1: Extended holds (max_hold → EOD bar 72)
    t1 = V15_CONFIG.copy()
    t1["name"] = "AllEOD"
    t1["max_hold_put"] = 72     # Effectively EOD for all trades
    t1["max_hold_call"] = 72
    configs["AllEOD"] = t1

    # Test 2: Longer holds (put=66, call=60 — +10% over baseline)
    t2 = V15_CONFIG.copy()
    t2["name"] = "LongerHold"
    t2["max_hold_put"] = 66     # 5.5 hours (from 5 hours)
    t2["max_hold_call"] = 60    # 5 hours (from 4.5 hours)
    configs["LongerHold"] = t2

    # Test 3: Equity curve sizing (reduce lots in drawdown)
    t3 = V15_CONFIG.copy()
    t3["name"] = "EqCurveSz"
    t3["use_equity_curve_sizing"] = True
    configs["EqCurveSz"] = t3

    # Test 4: Losing streak sizing
    t4 = V15_CONFIG.copy()
    t4["name"] = "StreakSz"
    t4["use_streak_sizing"] = True
    configs["StreakSz"] = t4

    # Test 5: Daily loss cap at 3%
    t5 = V15_CONFIG.copy()
    t5["name"] = "DailyLoss3"
    t5["daily_loss_cap_pct"] = 0.03
    configs["DailyLoss3"] = t5

    # Test 6: Block VIX 13-14 entirely (raise floor from 13 to 14)
    t6 = V15_CONFIG.copy()
    t6["name"] = "VIX14Floor"
    t6["vix_floor"] = 14
    configs["VIX14Floor"] = t6

    # Test 7: Combined: Equity curve + Longer holds
    t7 = V15_CONFIG.copy()
    t7["name"] = "EqCurve+LH"
    t7["use_equity_curve_sizing"] = True
    t7["max_hold_put"] = 66
    t7["max_hold_call"] = 60
    configs["EqCurve+LH"] = t7

    # Test 8: Max lot cap tighter (20 instead of 27)
    t8 = V15_CONFIG.copy()
    t8["name"] = "LotCap20"
    t8["max_lots_cap"] = 20
    configs["LotCap20"] = t8

    # Remove smart_strike_config from all configs
    for cfg in configs.values():
        cfg.pop("smart_strike_config", None)

    # ── Run all configs ──
    results = []
    for name, cfg in configs.items():
        print(f"Running {name}...", end=" ", flush=True)
        r = run_config(name, cfg, day_groups, all_dates, warmup_dates,
                        vix_lookup)
        results.append(r)
        print(f"{r['return_x']:.2f}x | {r['trades']} trades | "
              f"WR={r['win_rate']:.1f}% | PF={r['profit_factor']:.2f} | "
              f"MaxDD={r['max_drawdown']*100:.1f}% | Calmar={r['calmar']:.1f}")

    # ── Comparison Table ──
    print()
    print("=" * 115)
    print("  COMPARISON RESULTS")
    print("=" * 115)
    print()
    header = (f"{'Config':<14} {'Return':>8} {'Trades':>7} {'WR%':>6} {'PF':>6} "
              f"{'AvgWin':>9} {'AvgLoss':>9} {'MaxDD':>7} {'Sharpe':>7} {'Calmar':>8}")
    print(header)
    print("-" * len(header))

    results.sort(key=lambda x: x["return_x"], reverse=True)
    v15_pnl = next((r["total_pnl"] for r in results if r["config"] == "V15_Base"), 0)

    for r in results:
        diff = r["total_pnl"] - v15_pnl
        marker = " <-- BASE" if r["config"] == "V15_Base" else f" ({diff:+,.0f})"
        print(f"{r['config']:<14} {r['return_x']:>7.2f}x {r['trades']:>7} "
              f"{r['win_rate']:>5.1f}% {r['profit_factor']:>6.2f} "
              f"{r['avg_win']:>+9,.0f} {r['avg_loss']:>+9,.0f} "
              f"{r['max_drawdown']*100:>6.1f}% {r['sharpe']:>7.2f} "
              f"{r['calmar']:>8.1f}{marker}")

    # ── Exit Reason for top configs ──
    print()
    print("=" * 90)
    print("  EXIT REASON BREAKDOWN (top 4 configs)")
    print("=" * 90)
    for r in results[:4]:
        print(f"\n  {r['config']}:")
        print(f"  {'Reason':<25} {'Count':>6} {'Total PnL':>12} {'WR%':>7} {'Avg PnL':>10}")
        print(f"  {'-'*65}")
        sorted_exits = sorted(r["exit_reasons"].items(), key=lambda x: -x[1]["count"])
        for reason, stats in sorted_exits:
            wr = stats["wins"] / stats["count"] * 100 if stats["count"] > 0 else 0
            avg = stats["pnl"] / stats["count"] if stats["count"] > 0 else 0
            print(f"  {reason:<25} {stats['count']:>6} {stats['pnl']:>+12,.0f} "
                  f"{wr:>6.1f}% {avg:>+10,.0f}")

    # ── Risk-adjusted ranking ──
    print()
    print("=" * 90)
    print("  RISK-ADJUSTED RANKING (Return per unit MaxDD)")
    print("=" * 90)
    results.sort(key=lambda x: x["calmar"], reverse=True)
    for i, r in enumerate(results):
        risk_adj = r["return_x"] / max(r["max_drawdown"], 0.01)
        print(f"  {i+1}. {r['config']:<14} Calmar={r['calmar']:>6.1f} | "
              f"Return={r['return_x']:.2f}x | MaxDD={r['max_drawdown']*100:.1f}%")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
