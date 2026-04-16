"""
V14 R5 vs V15 Enhanced — Strategy Comparison Backtest
=====================================================
Runs both V14 R5 (baseline) and V15 Enhanced (with improvements)
on the SAME data to measure impact of:

V15 improvements:
  1. Volume confirmation (OBV + spike detection)
  2. Momentum acceleration scoring
  3. Multi-TF EMA stack alignment
  4. Session-specific weights (morning trend / afternoon reversion)
  5. RV/IV entry quality filter
  6. Stale trade exit (45 min no-progress kill)
  7. Chandelier exit (ATR-adaptive trailing)
  8. Momentum exhaustion exit (RSI extreme take-profit)
  9. Volume + velocity confluence filters
 10. Bearish RSI divergence scoring

Usage:
    python -m backtesting.v15_comparison
"""

import sys
import datetime as dt
from pathlib import Path
from collections import defaultdict

import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scoring.config import V14_CONFIG, V15_CONFIG
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


# ── Strike Selection (ITM_1 — proven best) ──────────────────

def select_strike_itm1(action, spot, vix, dte, is_expiry):
    """1 strike ITM (delta ~0.55-0.60)."""
    opt_type = "CE" if action == "BUY_CALL" else "PE"
    atm = round(spot / STRIKE_INTERVAL) * STRIKE_INTERVAL
    if action == "BUY_CALL":
        strike = atm - STRIKE_INTERVAL
    else:
        strike = atm + STRIKE_INTERVAL
    return strike, opt_type


# ── Premium + P&L helpers ────────────────────────────────────

def calc_premium(spot, strike, dte, vix, opt_type, slippage_sign=1):
    try:
        result = price_option(spot=spot, strike=strike, dte_days=max(0.1, dte),
                              vix=vix, option_type=opt_type)
        prem = result["premium"]
    except Exception:
        prem = 30.0
    prem = prem * (1 + slippage_sign * SLIPPAGE_PCT) + slippage_sign * SPREAD_RS
    return max(0.05, prem)


# ── Single Day Simulation ────────────────────────────────────

def simulate_day(bars_5min, date, vix, cfg, prev_close, equity,
                 warmup_bars, is_expiry, consecutive_down_days):
    """Run one day with a given config."""
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

        # Strike selection (ITM_1 for both — apples-to-apples)
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


# ── Run a single config across all dates ──────────────────────

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
    daily_equity = []

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
        daily_equity.append(equity)

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

    # ── Compute stats ──
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

    # Sharpe
    daily_returns = defaultdict(float)
    for t in all_trades:
        daily_returns[t["date"]] += t.get("pnl", 0) / CAPITAL
    daily_ret_list = list(daily_returns.values())
    sharpe = 0.0
    if daily_ret_list and np.std(daily_ret_list) > 0:
        sharpe = np.mean(daily_ret_list) / np.std(daily_ret_list) * np.sqrt(252)

    # Exit reason breakdown
    exit_reasons = defaultdict(lambda: {"count": 0, "pnl": 0.0, "wins": 0})
    for t in all_trades:
        r = t.get("exit_reason", "unknown")
        exit_reasons[r]["count"] += 1
        exit_reasons[r]["pnl"] += t.get("pnl", 0)
        if t.get("pnl", 0) > 0:
            exit_reasons[r]["wins"] += 1

    # Calmar ratio
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


# ── Main ──────────────────────────────────────────────────────

def main():
    start_date = dt.date(2024, 7, 1)
    end_date = dt.date(2025, 1, 1)

    print("=" * 90)
    print("  V14 R5 vs V15 ENHANCED — STRATEGY COMPARISON BACKTEST")
    print(f"  Period: {start_date} to {end_date}")
    print(f"  Capital: Rs {CAPITAL:,}")
    print("=" * 90)
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

    # ── Configs to compare ──
    configs = {
        "V14_R5": V14_CONFIG.copy(),
        "V15_Final": V15_CONFIG.copy(),  # Refined: scoring + exhaustion exit only
    }

    # V15d: Scoring only (no exits at all)
    v15d = V14_CONFIG.copy()
    v15d["name"] = "V15d_ScoreOnly"
    v15d["use_volume_confirmation"] = True
    v15d["low_volume_mult"] = 0.8
    v15d["use_obv_confirmation"] = True
    v15d["obv_score"] = 0.8
    v15d["use_momentum_accel"] = True
    v15d["momentum_accel_score"] = 0.7
    v15d["use_ema_stack"] = True
    v15d["ema_stack_score"] = 1.2
    v15d["use_session_weights"] = True
    v15d["morning_trend_boost"] = 1.1
    v15d["afternoon_adx_min"] = 28
    v15d["use_rv_iv_filter"] = True
    v15d["rv_iv_min"] = 0.8
    v15d["rv_iv_boost_threshold"] = 1.3
    v15d["use_rsi_bear_div"] = True
    v15d["rsi_bear_div_score"] = 1.5
    configs["V15d_ScoreOnly"] = v15d

    # V15e: Exhaustion exit only (no scoring changes)
    v15e = V14_CONFIG.copy()
    v15e["name"] = "V15e_ExhOnly"
    v15e["use_momentum_exhaustion_exit"] = True
    v15e["exhaustion_rsi_call"] = 75
    v15e["exhaustion_rsi_put"] = 25
    configs["V15e_ExhOnly"] = v15e

    # V15f: Scoring + more aggressive EMA stack + session weighting
    v15f = V14_CONFIG.copy()
    v15f["name"] = "V15f_AggrScore"
    v15f["use_volume_confirmation"] = True
    v15f["low_volume_mult"] = 0.75
    v15f["use_obv_confirmation"] = True
    v15f["obv_score"] = 1.0
    v15f["use_momentum_accel"] = True
    v15f["momentum_accel_score"] = 1.0
    v15f["use_ema_stack"] = True
    v15f["ema_stack_score"] = 1.5
    v15f["use_session_weights"] = True
    v15f["morning_trend_boost"] = 1.15
    v15f["afternoon_adx_min"] = 30
    v15f["use_rv_iv_filter"] = True
    v15f["rv_iv_min"] = 0.75
    v15f["rv_iv_boost_threshold"] = 1.2
    v15f["use_rsi_bear_div"] = True
    v15f["rsi_bear_div_score"] = 2.0
    configs["V15f_AggrScore"] = v15f

    # Remove smart_strike_config from all configs (live only)
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
    print("=" * 110)
    print("  COMPARISON RESULTS")
    print("=" * 110)
    print()
    header = (f"{'Config':<16} {'Return':>8} {'Trades':>7} {'WR%':>6} {'PF':>6} "
              f"{'AvgWin':>9} {'AvgLoss':>9} {'MaxDD':>7} {'Sharpe':>7} {'Calmar':>8}")
    print(header)
    print("-" * len(header))

    results.sort(key=lambda x: x["return_x"], reverse=True)
    v14_pnl = next((r["total_pnl"] for r in results if r["config"] == "V14_R5"), 0)

    for r in results:
        diff = r["total_pnl"] - v14_pnl
        marker = " <-- BASELINE" if r["config"] == "V14_R5" else f" ({diff:+,.0f})"
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

    # ── V15 vs V14 Summary ──
    v14_r = next((r for r in results if r["config"] == "V14_R5"), None)
    v15_r = next((r for r in results if r["config"] == "V15_Final"), None)
    if v14_r and v15_r:
        print()
        print("=" * 60)
        print("  V15 vs V14 IMPROVEMENT SUMMARY")
        print("=" * 60)
        metrics = [
            ("Return Multiple", v14_r["return_x"], v15_r["return_x"], "x"),
            ("Win Rate", v14_r["win_rate"], v15_r["win_rate"], "%"),
            ("Profit Factor", v14_r["profit_factor"], v15_r["profit_factor"], ""),
            ("Max Drawdown", v14_r["max_drawdown"]*100, v15_r["max_drawdown"]*100, "%"),
            ("Sharpe Ratio", v14_r["sharpe"], v15_r["sharpe"], ""),
            ("Calmar Ratio", v14_r["calmar"], v15_r["calmar"], ""),
            ("Total Trades", v14_r["trades"], v15_r["trades"], ""),
            ("Total P&L", v14_r["total_pnl"], v15_r["total_pnl"], "Rs"),
        ]
        for name, v14_val, v15_val, unit in metrics:
            if unit == "Rs":
                diff_str = f"{v15_val - v14_val:+,.0f}"
            elif unit == "%":
                diff_str = f"{v15_val - v14_val:+.1f}%"
            else:
                diff_str = f"{v15_val - v14_val:+.2f}"
            better = "BETTER" if (
                (name != "Max Drawdown" and v15_val > v14_val) or
                (name == "Max Drawdown" and v15_val < v14_val)
            ) else "WORSE" if v15_val != v14_val else "SAME"
            icon = "+" if better == "BETTER" else "-" if better == "WORSE" else "="
            print(f"  [{icon}] {name:<20} V14: {v14_val:>10.2f}  V15: {v15_val:>10.2f}  ({diff_str})")

        if v15_r["return_x"] > v14_r["return_x"]:
            print()
            print("  >>> V15 IMPROVES RETURNS — RECOMMEND DEPLOYING <<<")
        elif v15_r["max_drawdown"] < v14_r["max_drawdown"] and v15_r["return_x"] >= v14_r["return_x"] * 0.95:
            print()
            print("  >>> V15 REDUCES RISK with similar returns — RECOMMEND DEPLOYING <<<")
        else:
            print()
            print("  >>> V14 R5 remains the better strategy <<<")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
