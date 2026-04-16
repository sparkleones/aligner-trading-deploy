"""
V15 vs V16 — Regime-Aware Strategy Comparison Backtest
=====================================================
Tests V16 improvements against V15 baseline:
  1. ADX + BB Width regime filter (dampen ranging markets)
  2. OBV divergence scoring (reversal detection)
  3. Volume climax reversal boost
  4. ORB directional confirmation (align with opening range)
  5. Day-of-week parameter optimization (Thursday shorter holds)

Usage:
    python -m backtesting.v16_comparison
"""

import sys
import datetime as dt
from pathlib import Path
from collections import defaultdict

import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scoring.config import V15_CONFIG, V16_CONFIG
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
LOT_SIZE = 75
STRIKE_INTERVAL = 50
SLIPPAGE_PCT = 0.005
SPREAD_RS = 2.0
BROKERAGE_RT = 80.0


def select_strike_itm1(action, spot, vix, dte, is_expiry):
    """1 strike ITM (delta ~0.55-0.60)."""
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
                 warmup_bars, is_expiry, consecutive_down_days):
    """Run one day with a given config — V16 enhanced with ORB + DOW logic."""
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

    orb_high = 0.0
    orb_low = 0.0
    gap_detected = False
    day_open = bars_5min[0]["close"] if bars_5min else 0

    # ── V16: ORB tracking (first 6 bars = 30 min for directional bias) ──
    orb_30_high = 0.0
    orb_30_low = float('inf')
    # Afternoon range tracking (bars 33-57 = 12:00-14:00)
    afternoon_range_high = 0.0
    afternoon_range_low = float('inf')
    session_bias = "NEUTRAL"
    afternoon_bias = "NEUTRAL"

    span_per_lot = 40000
    if vix >= 20:
        span_per_lot = 50000
    elif vix >= 25:
        span_per_lot = 60000
    base_lots = max(1, int(equity * 0.70 / span_per_lot))

    days_to_expiry = (3 - day_of_week) % 7
    dte = 0.2 if days_to_expiry == 0 else float(days_to_expiry)

    # ── V16: Day-of-week parameter overrides ──
    effective_cfg = cfg.copy()
    if cfg.get("use_dow_optimization"):
        if day_of_week == 3:  # Thursday (expiry)
            effective_cfg["max_hold_put"] = cfg.get("thursday_max_hold_put", 36)
            effective_cfg["max_hold_call"] = cfg.get("thursday_max_hold_call", 30)
        elif day_of_week == 0:  # Monday
            boost = cfg.get("monday_hold_boost", 1.1)
            effective_cfg["max_hold_put"] = int(cfg["max_hold_put"] * boost)
            effective_cfg["max_hold_call"] = int(cfg["max_hold_call"] * boost)

    day_pnl = 0.0

    for bar_idx, bar in enumerate(bars_5min):
        bar_history.append(bar)
        if len(bar_history) > 500:
            bar_history = bar_history[-500:]

        spot = bar["close"]

        # Track ORB range (first 6 bars = 30 min)
        if bar_idx < 6:
            orb_30_high = max(orb_30_high, bar["high"])
            orb_30_low = min(orb_30_low, bar["low"])
        elif bar_idx == 6:
            # Establish session bias from 30-min ORB
            if spot > orb_30_high:
                session_bias = "BULLISH"
            elif spot < orb_30_low:
                session_bias = "BEARISH"
            else:
                session_bias = "NEUTRAL"
        elif bar_idx > 6:
            # Update session bias dynamically
            if spot > orb_30_high:
                session_bias = "BULLISH"
            elif spot < orb_30_low:
                session_bias = "BEARISH"

        # Track afternoon range (bars 33-57 = 12:00-14:00)
        if 33 <= bar_idx <= 57:
            afternoon_range_high = max(afternoon_range_high, bar["high"])
            afternoon_range_low = min(afternoon_range_low, bar["low"])
        elif bar_idx == 58 and afternoon_range_high > 0:
            if spot > afternoon_range_high:
                afternoon_bias = "BULLISH"
            elif spot < afternoon_range_low:
                afternoon_bias = "BEARISH"
            else:
                afternoon_bias = "NEUTRAL"
        elif bar_idx > 58:
            if afternoon_range_high > 0:
                if spot > afternoon_range_high:
                    afternoon_bias = "BULLISH"
                elif spot < afternoon_range_low:
                    afternoon_bias = "BEARISH"

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
                trade, bar_idx, spot, indicators or {}, effective_cfg,
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

        if len(open_trades) >= effective_cfg["max_concurrent"]:
            continue
        if trades_today >= effective_cfg["max_trades_per_day"]:
            continue

        entry_windows = effective_cfg.get("entry_windows_bars")
        action = None
        conf = 0
        entry_type = "v8_indicator"
        is_zero_hero = False

        if bar_idx < 3:
            prev_spot = bars_5min[bar_idx - 1]["close"] if bar_idx > 0 else spot
            composites = detect_composite_entries(
                bar, bar_idx, spot, vix, effective_cfg,
                prev_close=prev_close, gap_detected=gap_detected,
                orb_high=orb_high, orb_low=orb_low,
                prev_spot=prev_spot,
            )
            if bar_idx == 0:
                gap_detected = True
            if effective_cfg.get("disable_zero_hero", False):
                composites = [c for c in composites if not c[3]]
            if composites:
                composites.sort(key=lambda x: x[2], reverse=True)
                action, entry_type, conf, is_zero_hero = composites[0]
        else:
            if entry_windows:
                in_window = any(s <= bar_idx <= e for s, e in entry_windows)
                if not in_window:
                    continue

            # ── V16: Thursday block afternoon entries ──
            if (effective_cfg.get("use_dow_optimization") and
                effective_cfg.get("thursday_block_afternoon") and
                day_of_week == 3 and bar_idx >= 59):
                continue

            if bar_idx - last_exit_bar < effective_cfg["cooldown_bars"]:
                continue

            action, conf = score_entry(
                indicators, vix, effective_cfg,
                bar_idx=bar_idx,
                consecutive_down_days=consecutive_down_days,
            )

            if action is None:
                prev_spot = bars_5min[bar_idx - 1]["close"] if bar_idx > 0 else spot
                composites = detect_composite_entries(
                    bar, bar_idx, spot, vix, effective_cfg,
                    prev_close=prev_close, gap_detected=gap_detected,
                    orb_high=orb_high, orb_low=orb_low,
                    prev_spot=prev_spot,
                )
                if effective_cfg.get("disable_zero_hero", False):
                    composites = [c for c in composites if not c[3]]
                if composites:
                    composites.sort(key=lambda x: x[2], reverse=True)
                    action, entry_type, conf, is_zero_hero = composites[0]

        if action is None:
            continue

        # ── V16: ORB directional confirmation ──
        if effective_cfg.get("use_orb_confirmation") and bar_idx >= 6:
            if 3 <= bar_idx <= 15:
                # Morning window: require alignment with 30-min ORB
                if action == "BUY_CALL" and session_bias == "BEARISH":
                    continue
                if action == "BUY_PUT" and session_bias == "BULLISH":
                    continue
            elif 59 <= bar_idx <= 69:
                # Afternoon window: require alignment with afternoon range
                if action == "BUY_CALL" and afternoon_bias == "BEARISH":
                    continue
                if action == "BUY_PUT" and afternoon_bias == "BULLISH":
                    continue

        # ── V16: Thursday higher conviction requirement ──
        if (effective_cfg.get("use_dow_optimization") and
            day_of_week == 3 and not is_zero_hero):
            thursday_boost = effective_cfg.get("thursday_score_boost", 1.5)
            conf_threshold = effective_cfg["min_confidence"] * thursday_boost
            if conf < conf_threshold / 18.0:  # Scale to 0-1 range
                continue

        if vix < effective_cfg["vix_floor"] or vix > effective_cfg["vix_ceil"]:
            continue

        if not passes_confluence(
            action, conf, indicators, bar_idx, effective_cfg,
            current_spot=spot,
            prev_close=prev_close,
            day_open=day_open,
        ):
            continue

        if any(t["action"] == action for t in open_trades):
            continue

        # Lot sizing
        cfg_with_atr = effective_cfg.copy()
        if effective_cfg.get("use_atr_sizing"):
            cfg_with_atr["_current_atr"] = indicators.get("atr", 0)
        lots = compute_lots(action, conf, vix, indicators.get("rsi", 50),
                            is_expiry, base_lots, cfg_with_atr)
        if is_zero_hero:
            lots = min(effective_cfg.get("zero_hero_max_lots", 3), max(1, lots))

        if effective_cfg.get("vix_lot_scaling"):
            if vix < 13:
                lots = max(1, int(lots * effective_cfg.get("vix_below13_mult", 0.3)))
            elif 14 <= vix < 15:
                lots = max(1, int(lots * effective_cfg.get("vix_14_15_mult", 0.5)))
            elif 15 <= vix < 17:
                lots = max(1, int(lots * effective_cfg.get("vix_15_17_mult", 1.5)))
            elif vix >= 17:
                lots = max(1, int(lots * effective_cfg.get("vix_17plus_mult", 2.0)))
        if entry_type and "orb" in entry_type:
            lots = max(1, int(lots * effective_cfg.get("orb_lot_mult", 1.0)))
        lots = min(lots, effective_cfg.get("max_lots_cap", 999))

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
            "session_bias": session_bias,
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
        )

        equity += day_pnl
        all_trades.extend(trades)
        month_key = f"{date.year}-{date.month:02d}"
        monthly_pnl[month_key] += day_pnl

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

    # Day-of-week breakdown
    dow_stats = defaultdict(lambda: {"trades": 0, "pnl": 0.0, "wins": 0})
    for t in all_trades:
        dow = t.get("day_of_week", -1)
        dow_stats[dow]["trades"] += 1
        dow_stats[dow]["pnl"] += t.get("pnl", 0)
        if t.get("pnl", 0) > 0:
            dow_stats[dow]["wins"] += 1

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
        "dow_stats": dict(dow_stats),
        "all_trades": all_trades,
    }


def main():
    start_date = dt.date(2024, 7, 1)
    end_date = dt.date(2025, 1, 1)

    print("=" * 90)
    print("  V15 vs V16 REGIME-AWARE — STRATEGY COMPARISON BACKTEST")
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
    configs = {
        "V15_Baseline": V15_CONFIG.copy(),
        "V16_Full": V16_CONFIG.copy(),
    }

    # V16a: Regime filter only (ADX + BB width)
    v16a = V15_CONFIG.copy()
    v16a["name"] = "V16a_RegimeOnly"
    v16a["use_regime_filter"] = True
    v16a["regime_adx_min"] = 20
    v16a["regime_bb_pctile_min"] = 30
    v16a["regime_ranging_mult"] = 0.5
    configs["V16a_Regime"] = v16a

    # V16b: ORB confirmation only
    v16b = V15_CONFIG.copy()
    v16b["name"] = "V16b_ORBOnly"
    v16b["use_orb_confirmation"] = True
    configs["V16b_ORB"] = v16b

    # V16c: DOW optimization only
    v16c = V15_CONFIG.copy()
    v16c["name"] = "V16c_DOWOnly"
    v16c["use_dow_optimization"] = True
    v16c["thursday_max_hold_put"] = 36
    v16c["thursday_max_hold_call"] = 30
    v16c["thursday_score_boost"] = 1.5
    v16c["thursday_block_afternoon"] = True
    v16c["monday_hold_boost"] = 1.1
    configs["V16c_DOW"] = v16c

    # V16d: OBV divergence + volume climax only
    v16d = V15_CONFIG.copy()
    v16d["name"] = "V16d_OBVDiv"
    v16d["use_obv_divergence"] = True
    v16d["obv_div_score"] = 1.5
    v16d["use_volume_climax"] = True
    v16d["volume_climax_score"] = 1.0
    configs["V16d_OBVDiv"] = v16d

    # V16e: Regime + ORB (no DOW, no OBV div)
    v16e = V15_CONFIG.copy()
    v16e["name"] = "V16e_Regime+ORB"
    v16e["use_regime_filter"] = True
    v16e["regime_adx_min"] = 20
    v16e["regime_bb_pctile_min"] = 30
    v16e["regime_ranging_mult"] = 0.5
    v16e["use_orb_confirmation"] = True
    configs["V16e_Reg+ORB"] = v16e

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
    header = (f"{'Config':<16} {'Return':>8} {'Trades':>7} {'WR%':>6} {'PF':>6} "
              f"{'AvgWin':>9} {'AvgLoss':>9} {'MaxDD':>7} {'Sharpe':>7} {'Calmar':>8}")
    print(header)
    print("-" * len(header))

    results.sort(key=lambda x: x["return_x"], reverse=True)
    v15_pnl = next((r["total_pnl"] for r in results if r["config"] == "V15_Baseline"), 0)

    for r in results:
        diff = r["total_pnl"] - v15_pnl
        marker = " <-- BASELINE" if r["config"] == "V15_Baseline" else f" ({diff:+,.0f})"
        print(f"{r['config']:<16} {r['return_x']:>7.2f}x {r['trades']:>7} "
              f"{r['win_rate']:>5.1f}% {r['profit_factor']:>6.2f} "
              f"{r['avg_win']:>+9,.0f} {r['avg_loss']:>+9,.0f} "
              f"{r['max_drawdown']*100:>6.1f}% {r['sharpe']:>7.2f} "
              f"{r['calmar']:>8.1f}{marker}")

    # ── Monthly P&L Comparison ──
    print()
    print("=" * 90)
    print("  MONTHLY P&L COMPARISON")
    print("=" * 90)
    all_months = sorted(set(m for r in results for m in r["monthly_pnl"]))
    header_months = f"{'Month':<10}"
    for r in results:
        header_months += f" {r['config']:>14}"
    print(header_months)
    print("-" * len(header_months))
    for month in all_months:
        row = f"{month:<10}"
        for r in results:
            pnl = r["monthly_pnl"].get(month, 0)
            row += f" {pnl:>+14,.0f}"
        print(row)

    # ── Exit Reason Analysis ──
    print()
    print("=" * 90)
    print("  EXIT REASON BREAKDOWN")
    print("=" * 90)
    for r in results:
        print(f"\n  {r['config']}:")
        print(f"  {'Reason':<25} {'Count':>6} {'Total PnL':>12} {'WR%':>7} {'Avg PnL':>10}")
        print(f"  {'-'*65}")
        sorted_exits = sorted(r["exit_reasons"].items(), key=lambda x: -x[1]["count"])
        for reason, stats in sorted_exits:
            wr = stats["wins"] / stats["count"] * 100 if stats["count"] > 0 else 0
            avg = stats["pnl"] / stats["count"] if stats["count"] > 0 else 0
            print(f"  {reason:<25} {stats['count']:>6} {stats['pnl']:>+12,.0f} "
                  f"{wr:>6.1f}% {avg:>+10,.0f}")

    # ── Day of Week Analysis (for V16 configs) ──
    print()
    print("=" * 90)
    print("  DAY-OF-WEEK BREAKDOWN")
    print("=" * 90)
    dow_names = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri"}
    for r in results:
        print(f"\n  {r['config']}:")
        print(f"  {'Day':<6} {'Trades':>7} {'WR%':>7} {'Total PnL':>12} {'Avg PnL':>10}")
        print(f"  {'-'*45}")
        for dow in sorted(r["dow_stats"].keys()):
            if dow < 0:
                continue
            s = r["dow_stats"][dow]
            wr = s["wins"] / s["trades"] * 100 if s["trades"] > 0 else 0
            avg = s["pnl"] / s["trades"] if s["trades"] > 0 else 0
            print(f"  {dow_names.get(dow, '?'):<6} {s['trades']:>7} "
                  f"{wr:>6.1f}% {s['pnl']:>+12,.0f} {avg:>+10,.0f}")

    # ── V16 vs V15 Summary ──
    v15_r = next((r for r in results if r["config"] == "V15_Baseline"), None)
    best_v16 = max(
        [r for r in results if r["config"] != "V15_Baseline"],
        key=lambda x: x["return_x"],
        default=None,
    )
    if v15_r and best_v16:
        print()
        print("=" * 60)
        print(f"  BEST V16 ({best_v16['config']}) vs V15 SUMMARY")
        print("=" * 60)
        metrics = [
            ("Return Multiple", v15_r["return_x"], best_v16["return_x"], "x"),
            ("Win Rate", v15_r["win_rate"], best_v16["win_rate"], "%"),
            ("Profit Factor", v15_r["profit_factor"], best_v16["profit_factor"], ""),
            ("Max Drawdown", v15_r["max_drawdown"]*100, best_v16["max_drawdown"]*100, "%"),
            ("Sharpe Ratio", v15_r["sharpe"], best_v16["sharpe"], ""),
            ("Calmar Ratio", v15_r["calmar"], best_v16["calmar"], ""),
            ("Total Trades", v15_r["trades"], best_v16["trades"], ""),
            ("Total P&L", v15_r["total_pnl"], best_v16["total_pnl"], "Rs"),
        ]
        for name, v15_val, v16_val, unit in metrics:
            if unit == "Rs":
                diff_str = f"{v16_val - v15_val:+,.0f}"
            elif unit == "%":
                diff_str = f"{v16_val - v15_val:+.1f}%"
            else:
                diff_str = f"{v16_val - v15_val:+.2f}"
            better = "BETTER" if (
                (name != "Max Drawdown" and v16_val > v15_val) or
                (name == "Max Drawdown" and v16_val < v15_val)
            ) else "WORSE" if v16_val != v15_val else "SAME"
            icon = "+" if better == "BETTER" else "-" if better == "WORSE" else "="
            print(f"  [{icon}] {name:<20} V15: {v15_val:>10.2f}  "
                  f"Best V16: {v16_val:>10.2f}  ({diff_str})")

        if best_v16["return_x"] > v15_r["return_x"]:
            print()
            print(f"  >>> {best_v16['config']} IMPROVES RETURNS — RECOMMEND DEPLOYING <<<")
        elif (best_v16["max_drawdown"] < v15_r["max_drawdown"] and
              best_v16["return_x"] >= v15_r["return_x"] * 0.95):
            print()
            print(f"  >>> {best_v16['config']} REDUCES RISK — RECOMMEND DEPLOYING <<<")
        else:
            print()
            print("  >>> V15 remains the better strategy <<<")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
