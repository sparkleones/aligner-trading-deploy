"""
Research-Driven Enhancement Comparison Backtest
================================================
Compares V14_R5 baseline vs enhanced model with:
  1. IV Percentile entry gate (block entries when IV > 80th pctile)
  2. OI Change scoring (delta OI buildup/unwinding)
  3. Expiry day max pain convergence boost

Runs both configs on the same data and reports comparison metrics.

Usage:
    python -m backtesting.research_comparison_backtest
"""

import sys
import datetime as dt
from pathlib import Path
from collections import defaultdict
from copy import deepcopy

import numpy as np
import pandas as pd

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

DATA_DIR = project_root / "data" / "historical"
CAPITAL = 200_000
LOT_SIZE = 75
STRIKE_INTERVAL = 50
SLIPPAGE_PCT = 0.005
SPREAD_RS = 2.0
BROKERAGE_RT = 80.0


def load_nifty_data(start_date, end_date):
    all_dfs = []
    for f in sorted(DATA_DIR.glob("nifty_min_*.csv")):
        try:
            df = pd.read_csv(f, parse_dates=["timestamp"], index_col="timestamp")
            mask = (df.index.date >= start_date) & (df.index.date <= end_date)
            chunk = df[mask]
            if len(chunk) > 0:
                all_dfs.append(chunk)
        except Exception:
            continue
    if not all_dfs:
        raise FileNotFoundError(f"No NIFTY data found for {start_date} to {end_date}")
    combined = pd.concat(all_dfs)
    combined = combined[~combined.index.duplicated(keep="first")]
    combined.sort_index(inplace=True)
    return combined


def load_vix_data(start_date, end_date):
    vix_lookup = {}
    for f in sorted(DATA_DIR.glob("vix_min_*.csv")):
        try:
            df = pd.read_csv(f, parse_dates=["timestamp"], index_col="timestamp")
            mask = (df.index.date >= start_date) & (df.index.date <= end_date)
            chunk = df[mask]
            if len(chunk) > 0:
                for d, group in chunk.groupby(chunk.index.date):
                    vix_lookup[d] = float(group["close"].iloc[-1])
        except Exception:
            continue
    return vix_lookup


def resample_to_5min(df_1min):
    df_5 = df_1min.resample("5min").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna(subset=["open"])
    df_5 = df_5[(df_5.index.time >= dt.time(9, 15)) & (df_5.index.time <= dt.time(15, 30))]
    return df_5


def bars_to_dicts(df_5min, date_str):
    return [{
        "open": float(r["open"]),
        "high": float(r["high"]),
        "low": float(r["low"]),
        "close": float(r["close"]),
        "volume": float(r.get("volume", 0)),
        "date": date_str,
        "time": str(ts.time()),
    } for ts, r in df_5min.iterrows()]


def get_strike_and_type(action, spot, vix, zero_hero=False):
    atm = round(spot / STRIKE_INTERVAL) * STRIKE_INTERVAL
    if zero_hero:
        offset = 200
        if action == "BUY_PUT":
            return atm - offset, "PE"
        else:
            return atm + offset, "CE"
    if action == "BUY_PUT":
        return atm, "PE"
    else:
        return atm, "CE"


def calc_premium(spot, strike, dte, vix, opt_type, slippage_sign=1):
    try:
        result = price_option(spot, strike, dte, vix, opt_type)
        prem = result["premium"] if isinstance(result, dict) else float(result)
    except Exception:
        intrinsic = max(0, spot - strike) if opt_type == "CE" else max(0, strike - spot)
        prem = intrinsic + 50
    prem += slippage_sign * prem * SLIPPAGE_PCT
    prem += slippage_sign * SPREAD_RS
    return max(1.0, prem)


def simulate_day(bars_5min, date, vix, cfg, prev_close, equity,
                 warmup_bars=None, is_expiry=False, consecutive_down_days=0):
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

    dte = 7      # days to expiry (pricer takes days, not years)
    if is_expiry:
        dte = 0.5

    for bar_idx, bar in enumerate(bars_5min):
        spot = bar["close"]
        bar_history.append(bar)

        if bar_idx == 0:
            orb_high = bar["high"]
            orb_low = bar["low"]

        # Exit check
        for trade in list(open_trades):
            indicators = compute_indicators(bar_history)
            if indicators is None:
                continue
            exit_reason = evaluate_exit(trade, bar_idx, spot, indicators, cfg,
                                         day_of_week=date.weekday())
            if exit_reason:
                strike = trade["strike"]
                opt_type = trade["opt_type"]
                exit_prem = calc_premium(spot, strike, dte, vix, opt_type, slippage_sign=-1)
                qty = trade["qty"]

                if trade["action"] == "BUY_PUT":
                    pnl = (exit_prem - trade["entry_prem"]) * qty
                    pnl += qty * (trade["entry_spot"] - spot) * 0.01  # Delta approx
                else:
                    pnl = (exit_prem - trade["entry_prem"]) * qty
                    pnl += qty * (spot - trade["entry_spot"]) * 0.01

                pnl -= BROKERAGE_RT
                trade["pnl"] = pnl
                trade["exit_bar"] = bar_idx
                trade["exit_reason"] = exit_reason
                closed_trades.append(trade)
                open_trades.remove(trade)
                last_exit_bar = bar_idx

        # Entry check
        if len(open_trades) >= cfg.get("max_concurrent", 3):
            continue
        if trades_today >= cfg.get("max_trades_per_day", 7):
            continue
        if bar_idx - last_exit_bar < cfg.get("cooldown_bars", 2):
            continue

        indicators = compute_indicators(bar_history)
        if indicators is None:
            continue

        if vix < cfg["vix_floor"] or vix > cfg["vix_ceil"]:
            continue

        # Score entry
        action, conf = score_entry(
            indicators, vix, cfg,
            pcr=0.0, bar_idx=bar_idx,
            consecutive_down_days=consecutive_down_days,
            regime_block_reversion=False,
            is_expiry=is_expiry,
        )
        entry_type = "v8_indicator"
        is_zero_hero = False

        if action is None:
            composites = detect_composite_entries(
                bar, bar_idx, spot, vix, cfg,
                prev_close=prev_close,
                gap_detected=gap_detected,
                orb_high=orb_high, orb_low=orb_low,
            )
            if cfg.get("disable_zero_hero", False):
                composites = [c for c in composites if not c[3]]
            if composites:
                composites.sort(key=lambda x: x[2], reverse=True)
                action, entry_type, conf, is_zero_hero = composites[0]
                if bar_idx == 0:
                    gap_detected = True

        if action is None:
            continue

        # Confluence
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

        strike, opt_type = get_strike_and_type(action, spot, vix, zero_hero=is_zero_hero)
        qty = lots * LOT_SIZE
        entry_prem = calc_premium(spot, strike, dte, vix, opt_type, slippage_sign=1)

        trade = {
            "date": date_str,
            "action": action,
            "entry_bar": bar_idx,
            "entry_spot": spot,
            "entry_prem": entry_prem,
            "strike": strike,
            "opt_type": opt_type,
            "qty": qty,
            "lots": lots,
            "best_fav": spot,
            "is_zero_hero": is_zero_hero,
            "entry_type": entry_type,
        }
        open_trades.append(trade)
        trades_today += 1

    # Force close remaining
    eod_spot = bars_5min[-1]["close"] if bars_5min else 0
    for trade in open_trades:
        exit_prem = calc_premium(eod_spot, trade["strike"], dte, vix, trade["opt_type"], slippage_sign=-1)
        qty = trade["qty"]
        if trade["action"] == "BUY_PUT":
            pnl = (exit_prem - trade["entry_prem"]) * qty
            pnl += qty * (trade["entry_spot"] - eod_spot) * 0.01
        else:
            pnl = (exit_prem - trade["entry_prem"]) * qty
            pnl += qty * (eod_spot - trade["entry_spot"]) * 0.01
        pnl -= BROKERAGE_RT
        trade["pnl"] = pnl
        trade["exit_reason"] = "eod_force_close"
        closed_trades.append(trade)

    day_pnl = sum(t.get("pnl", 0) for t in closed_trades)
    return closed_trades, day_pnl, eod_spot


def run_backtest(cfg, label, start_date, end_date, df_5min_all, vix_lookup, day_groups):
    """Run a full backtest with given config and return metrics."""
    all_dates = sorted(d for d in day_groups.keys() if d >= start_date)
    warmup_dates = sorted(d for d in day_groups.keys() if d < start_date)

    warmup_bars = []
    for d in warmup_dates[-3:]:
        warmup_bars.extend(day_groups[d])

    equity = CAPITAL
    all_trades = []
    prev_close = 0.0
    consecutive_down_days = 0
    max_equity = CAPITAL
    max_drawdown = 0.0

    for date in all_dates:
        bars = day_groups[date]
        if len(bars) < 5:
            continue

        vix = vix_lookup.get(date, 14.0)
        is_expiry = (date.weekday() == 3)  # Thursday for 2024 data

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
        max_equity = max(max_equity, equity)
        dd = (max_equity - equity) / max_equity * 100
        max_drawdown = max(max_drawdown, dd)

        if len(bars) >= 2:
            if bars[-1]["close"] < bars[0]["open"]:
                consecutive_down_days += 1
            else:
                consecutive_down_days = 0

        warmup_bars = warmup_bars[-(75 * 2):] + bars
        prev_close = eod_close

    # Metrics
    total_pnl = sum(t.get("pnl", 0) for t in all_trades)
    wins = [t for t in all_trades if t.get("pnl", 0) > 0]
    losses = [t for t in all_trades if t.get("pnl", 0) <= 0]
    win_rate = len(wins) / len(all_trades) * 100 if all_trades else 0
    avg_win = np.mean([t["pnl"] for t in wins]) if wins else 0
    avg_loss = np.mean([abs(t["pnl"]) for t in losses]) if losses else 1
    gross_win = sum(t["pnl"] for t in wins)
    gross_loss = sum(abs(t["pnl"]) for t in losses)
    profit_factor = gross_win / gross_loss if gross_loss > 0 else float("inf")
    ret = (equity - CAPITAL) / CAPITAL

    return {
        "label": label,
        "trades": len(all_trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": win_rate,
        "total_pnl": total_pnl,
        "equity": equity,
        "return_x": equity / CAPITAL,
        "profit_factor": profit_factor,
        "max_drawdown": max_drawdown,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
    }


def main():
    print("=" * 70)
    print("  RESEARCH-DRIVEN ENHANCEMENT COMPARISON BACKTEST")
    print("=" * 70)
    print()

    # Date range: use available data period
    end_date = dt.date(2025, 1, 31)
    start_date = dt.date(2024, 7, 1)
    warmup_start = start_date - dt.timedelta(days=10)

    print(f"  Period: {start_date} to {end_date}")
    print(f"  Capital: Rs {CAPITAL:,.0f}")
    print()

    # Load data once
    print("Loading data...")
    df = load_nifty_data(warmup_start, end_date)
    vix_lookup = load_vix_data(warmup_start, end_date)
    df_5min = resample_to_5min(df)

    # Build day groups
    day_groups = {}
    for d, group in df_5min.groupby(df_5min.index.date):
        day_groups[d] = bars_to_dicts(group, str(d))

    trading_days = len([d for d in day_groups if d >= start_date])
    print(f"  Trading days: {trading_days}")
    print()

    # ── Config variants ──
    # 1. BASELINE: V15_CONFIG as-is (no research enhancements)
    baseline = deepcopy(V15_CONFIG)
    baseline["name"] = "BASELINE_V15"
    baseline["use_iv_pctile_gate"] = False
    baseline["use_oi_change_scoring"] = False
    baseline["use_expiry_max_pain_boost"] = False

    # 2. IV GATE ONLY
    iv_gate_only = deepcopy(V15_CONFIG)
    iv_gate_only["name"] = "IV_GATE_ONLY"
    iv_gate_only["use_iv_pctile_gate"] = True
    iv_gate_only["iv_pctile_gate_threshold"] = 80
    iv_gate_only["use_oi_change_scoring"] = False
    iv_gate_only["use_expiry_max_pain_boost"] = False

    # 3. OI CHANGE ONLY
    oi_change_only = deepcopy(V15_CONFIG)
    oi_change_only["name"] = "OI_CHANGE_ONLY"
    oi_change_only["use_iv_pctile_gate"] = False
    oi_change_only["use_oi_change_scoring"] = True
    oi_change_only["oi_change_buildup_score"] = 0.5
    oi_change_only["oi_change_unwinding_score"] = -0.3
    oi_change_only["use_expiry_max_pain_boost"] = False

    # 4. EXPIRY MAX PAIN ONLY
    expiry_mp_only = deepcopy(V15_CONFIG)
    expiry_mp_only["name"] = "EXPIRY_MP_ONLY"
    expiry_mp_only["use_iv_pctile_gate"] = False
    expiry_mp_only["use_oi_change_scoring"] = False
    expiry_mp_only["use_expiry_max_pain_boost"] = True
    expiry_mp_only["expiry_max_pain_score"] = 1.0
    expiry_mp_only["expiry_max_pain_after_bar"] = 57

    # 5. ALL ENHANCEMENTS
    all_enhanced = deepcopy(V15_CONFIG)
    all_enhanced["name"] = "ALL_ENHANCED"
    all_enhanced["use_iv_pctile_gate"] = True
    all_enhanced["iv_pctile_gate_threshold"] = 80
    all_enhanced["use_oi_change_scoring"] = True
    all_enhanced["oi_change_buildup_score"] = 0.5
    all_enhanced["oi_change_unwinding_score"] = -0.3
    all_enhanced["use_expiry_max_pain_boost"] = True
    all_enhanced["expiry_max_pain_score"] = 1.0
    all_enhanced["expiry_max_pain_after_bar"] = 57

    configs = [
        (baseline, "BASELINE"),
        (iv_gate_only, "IV_GATE"),
        (oi_change_only, "OI_CHANGE"),
        (expiry_mp_only, "EXPIRY_MP"),
        (all_enhanced, "ALL_ENHANCED"),
    ]

    results = []
    for cfg, label in configs:
        print(f"Running {label}...", end="", flush=True)
        result = run_backtest(cfg, label, start_date, end_date, df_5min, vix_lookup, day_groups)
        results.append(result)
        print(f" done ({result['trades']} trades, {result['return_x']:.2f}x)")

    # ── Print comparison ──
    print()
    print("=" * 90)
    print(f"{'Config':<16} {'Trades':>7} {'Wins':>5} {'WR%':>6} {'P&L':>12} {'Return':>8} {'PF':>6} {'MaxDD':>7} {'AvgW':>8} {'AvgL':>8}")
    print("-" * 90)
    for r in results:
        print(f"{r['label']:<16} {r['trades']:>7} {r['wins']:>5} {r['win_rate']:>5.1f}% "
              f"{r['total_pnl']:>+11,.0f} {r['return_x']:>7.2f}x {r['profit_factor']:>5.2f} "
              f"{r['max_drawdown']:>6.1f}% {r['avg_win']:>+7,.0f} {r['avg_loss']:>7,.0f}")

    print()
    baseline_pnl = results[0]["total_pnl"]
    for r in results[1:]:
        delta = r["total_pnl"] - baseline_pnl
        print(f"  {r['label']} vs BASELINE: {delta:+,.0f} Rs ({delta/max(1,abs(baseline_pnl))*100:+.1f}%)")

    print()
    best = max(results, key=lambda x: x["total_pnl"])
    print(f"  BEST: {best['label']} -> {best['return_x']:.2f}x return, PF {best['profit_factor']:.2f}")
    print()


if __name__ == "__main__":
    main()
