"""
Full 6-Month Analysis — Maximize Profit with Data-Driven Entry + Exit Rules.

EXIT LOGIC: Based on support/resistance, trailing stops, VIX regime shifts,
and intraday price action — NOT arbitrary premium percentages.

The backtest simulates intraday trading:
  - Each day is split into ~25 bars (15-min equivalent)
  - Entries happen at open with composite scoring
  - Exits are dynamic: trailing stop, S/R breach, VIX shift, time decay
  - Multiple exit strategies are tested to find the best one

Then the winning strategy is applied to all 6 months and the live agent is updated.
"""

import json
import math
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.constants import (
    INDEX_CONFIG, STT_RATES, NSE_TRANSACTION_CHARGE,
    SEBI_TURNOVER_FEE, STAMP_DUTY_BUY, GST_RATE,
)
from backtesting.option_pricer import price_option

LOT_SIZE = INDEX_CONFIG["NIFTY"]["lot_size"]
BROKERAGE = 20.0
STRIKE_INTERVAL = 50


def calc_costs(n_legs, avg_premium, qty=LOT_SIZE):
    total = 0.0
    for _ in range(n_legs):
        turnover = avg_premium * qty
        brokerage = BROKERAGE * 2
        stt = turnover * STT_RATES.get("options_sell", 0.0015)
        exchange = turnover * NSE_TRANSACTION_CHARGE
        sebi = turnover * SEBI_TURNOVER_FEE
        stamp = turnover * STAMP_DUTY_BUY
        gst = (brokerage + exchange + sebi) * GST_RATE
        total += brokerage + stt + exchange + sebi + stamp + gst
    return total


def bs_premium(spot, strike, dte, vix, opt_type):
    try:
        return price_option(spot=spot, strike=strike, dte_days=dte,
                            vix=vix, option_type=opt_type)["premium"]
    except Exception:
        return 30.0


def compute_support_resistance(prices, current_idx, lookback=20):
    """Compute dynamic support/resistance from recent price action."""
    start = max(0, current_idx - lookback)
    window = prices[start:current_idx + 1]
    if len(window) < 5:
        return None, None

    highs = []
    lows = []
    for i in range(1, len(window) - 1):
        if window[i] > window[i-1] and window[i] > window[i+1]:
            highs.append(window[i])
        if window[i] < window[i-1] and window[i] < window[i+1]:
            lows.append(window[i])

    # Round to nearest 50 for NIFTY
    current = window[-1]
    support = None
    resistance = None

    if lows:
        # Nearest support below current price
        below = [l for l in lows if l < current]
        if below:
            support = round(max(below) / 50) * 50

    if highs:
        # Nearest resistance above current price
        above = [h for h in highs if h > current]
        if above:
            resistance = round(min(above) / 50) * 50

    # Fallback: use SMA-based levels
    if support is None:
        support = round((current * 0.99) / 50) * 50  # 1% below
    if resistance is None:
        resistance = round((current * 1.01) / 50) * 50  # 1% above

    return support, resistance


def simulate_intraday(action, entry_spot, day_high, day_low, day_close,
                      entry_vix, support, resistance, sma20, sma50,
                      exit_strategy, qty, capital):
    """
    Simulate a trade with intraday price path and dynamic exits.

    Instead of just entry->close, we model the intraday path:
    open -> move toward extreme -> partial retrace -> close

    Exit strategies tested:
      1. "sr_trail" - Support/resistance stop + trailing stop
      2. "sr_fixed" - Support/resistance stop, hold to close
      3. "trail_pct" - Trailing stop based on % of move
      4. "sma_exit" - Exit on SMA crossover
      5. "vix_adaptive" - VIX-weighted trailing stop (tight in high VIX)
    """
    dte_entry = 2.0
    n_bars = 25  # ~25 intraday 15-min bars

    # Risk budget: 8% per trade
    risk_budget = capital * 0.08
    max_loss_per_lot = 50 * LOT_SIZE
    num_lots = max(1, int(risk_budget / max_loss_per_lot))
    actual_qty = LOT_SIZE * num_lots

    strike = round(entry_spot / 50) * 50  # ATM

    # Generate realistic intraday path
    # Pattern: open -> trend toward extreme -> partial retrace -> close
    path = _generate_intraday_path(entry_spot, day_high, day_low, day_close, n_bars)

    # Determine option type
    if action == "BUY_CALL":
        opt_type = "CE"
        is_long = True
        favorable_dir = 1  # spot going up is good
    elif action == "BUY_PUT":
        opt_type = "PE"
        is_long = True
        favorable_dir = -1  # spot going down is good
    elif action == "SELL_CALL_SPREAD":
        opt_type = "CE"
        is_long = False
        favorable_dir = -1  # spot going down is good for short call
    elif action == "SELL_PUT_SPREAD":
        opt_type = "PE"
        is_long = False
        favorable_dir = 1  # spot going up is good for short put
    else:
        return 0.0, 0.0, 0, n_bars, "no_trade"

    # Entry premium
    entry_prem = bs_premium(entry_spot, strike, dte_entry, entry_vix, opt_type)

    # For spreads, compute both legs
    if not is_long:
        atm_iv = entry_vix / 100 * 0.88
        exp_move = entry_spot * atm_iv * math.sqrt(dte_entry / 365)
        if action == "SELL_CALL_SPREAD":
            sell_strike = round((entry_spot + exp_move * 1.0) / 50) * 50
            buy_strike = sell_strike + 50
        else:
            sell_strike = round((entry_spot - exp_move * 1.0) / 50) * 50
            buy_strike = sell_strike - 50

    # Track best premium for trailing stop
    best_pnl = 0.0
    exit_bar = n_bars
    exit_reason = "eod_close"
    exit_spot = day_close

    for bar_i in range(1, n_bars):
        bar_spot = path[bar_i]
        bar_dte = max(0.05, dte_entry - bar_i * 15 / 1440)  # DTE decreases

        # Current P&L
        if is_long:
            bar_prem = bs_premium(bar_spot, strike, bar_dte, entry_vix, opt_type)
            bar_pnl = (bar_prem - entry_prem) * actual_qty
        else:
            # Credit spread P&L
            s_opt = "CE" if action == "SELL_CALL_SPREAD" else "PE"
            sell_prem_now = bs_premium(bar_spot, sell_strike, bar_dte, entry_vix, s_opt)
            buy_prem_now = bs_premium(bar_spot, buy_strike, bar_dte, entry_vix, s_opt)
            sell_prem_entry = bs_premium(entry_spot, sell_strike, dte_entry, entry_vix, s_opt)
            buy_prem_entry = bs_premium(entry_spot, buy_strike, dte_entry, entry_vix, s_opt)
            entry_credit = sell_prem_entry - buy_prem_entry
            current_cost = sell_prem_now - buy_prem_now
            bar_pnl = (entry_credit - current_cost) * actual_qty

        # Update best P&L
        if bar_pnl > best_pnl:
            best_pnl = bar_pnl

        # ── EXIT STRATEGY LOGIC ──────────────────────────────────

        if exit_strategy == "sr_trail":
            # Support/Resistance stop + trailing stop at 40% of peak profit
            # Stop loss: if spot breaches S/R against our direction
            if action == "BUY_CALL" and support and bar_spot < support:
                exit_bar, exit_spot, exit_reason = bar_i, bar_spot, "support_breach"
                break
            if action == "BUY_PUT" and resistance and bar_spot > resistance:
                exit_bar, exit_spot, exit_reason = bar_i, bar_spot, "resistance_breach"
                break
            if action == "SELL_CALL_SPREAD" and resistance and bar_spot > resistance:
                exit_bar, exit_spot, exit_reason = bar_i, bar_spot, "resistance_breach"
                break
            if action == "SELL_PUT_SPREAD" and support and bar_spot < support:
                exit_bar, exit_spot, exit_reason = bar_i, bar_spot, "support_breach"
                break
            # Trailing stop: give back max 40% of peak profit
            if best_pnl > 500 and bar_pnl < best_pnl * 0.6:
                exit_bar, exit_spot, exit_reason = bar_i, bar_spot, "trailing_stop"
                break

        elif exit_strategy == "sr_fixed":
            # S/R based stop, hold to close otherwise
            if action == "BUY_CALL" and support and bar_spot < support:
                exit_bar, exit_spot, exit_reason = bar_i, bar_spot, "support_breach"
                break
            if action == "BUY_PUT" and resistance and bar_spot > resistance:
                exit_bar, exit_spot, exit_reason = bar_i, bar_spot, "resistance_breach"
                break
            if action == "SELL_CALL_SPREAD" and resistance and bar_spot > resistance:
                exit_bar, exit_spot, exit_reason = bar_i, bar_spot, "resistance_breach"
                break
            if action == "SELL_PUT_SPREAD" and support and bar_spot < support:
                exit_bar, exit_spot, exit_reason = bar_i, bar_spot, "support_breach"
                break

        elif exit_strategy == "trail_pct":
            # Pure trailing stop: trail at 0.3% of spot
            trail_dist = entry_spot * 0.003
            if action in ("BUY_CALL", "SELL_PUT_SPREAD"):
                # For bullish trades, trail stop below
                peak_spot = max(path[:bar_i+1])
                if bar_spot < peak_spot - trail_dist and bar_i > 3:
                    exit_bar, exit_spot, exit_reason = bar_i, bar_spot, "trailing_pct"
                    break
            else:
                # For bearish trades, trail stop above
                trough_spot = min(path[:bar_i+1])
                if bar_spot > trough_spot + trail_dist and bar_i > 3:
                    exit_bar, exit_spot, exit_reason = bar_i, bar_spot, "trailing_pct"
                    break

        elif exit_strategy == "sma_exit":
            # Exit when spot crosses SMA20 against trade direction
            if sma20 and action in ("BUY_CALL", "SELL_PUT_SPREAD"):
                if bar_spot < sma20 and bar_i > 5:
                    exit_bar, exit_spot, exit_reason = bar_i, bar_spot, "sma_cross"
                    break
            elif sma20 and action in ("BUY_PUT", "SELL_CALL_SPREAD"):
                if bar_spot > sma20 and bar_i > 5:
                    exit_bar, exit_spot, exit_reason = bar_i, bar_spot, "sma_cross"
                    break

        elif exit_strategy == "vix_adaptive":
            # VIX-weighted: tighter stops in high VIX, wider in low VIX
            # High VIX = bigger moves, so use wider absolute stop but tighter %
            if entry_vix > 20:
                trail_mult = 0.25  # Keep only 25% of retrace in high vol
                sr_buffer = entry_spot * 0.002  # Tighter S/R buffer
            elif entry_vix > 14:
                trail_mult = 0.35
                sr_buffer = entry_spot * 0.003
            else:
                trail_mult = 0.45
                sr_buffer = entry_spot * 0.004

            # S/R stop with buffer
            if action == "BUY_CALL" and support:
                if bar_spot < support - sr_buffer:
                    exit_bar, exit_spot, exit_reason = bar_i, bar_spot, "sr_vix_stop"
                    break
            if action == "BUY_PUT" and resistance:
                if bar_spot > resistance + sr_buffer:
                    exit_bar, exit_spot, exit_reason = bar_i, bar_spot, "sr_vix_stop"
                    break

            # Trailing stop
            if best_pnl > 300 and bar_pnl < best_pnl * trail_mult:
                exit_bar, exit_spot, exit_reason = bar_i, bar_spot, "vix_trail"
                break

            # Profit target: take profit at 2x risk in high VIX (moves are bigger)
            if entry_vix > 20 and bar_pnl > risk_budget * 0.5:
                exit_bar, exit_spot, exit_reason = bar_i, bar_spot, "vix_target"
                break

    # Compute final P&L at exit
    exit_dte = max(0.05, dte_entry - exit_bar * 15 / 1440)
    if is_long:
        exit_prem = bs_premium(exit_spot, strike, exit_dte, entry_vix, opt_type)
        pnl = (exit_prem - entry_prem) * actual_qty
        avg_prem = (entry_prem + exit_prem) / 2
        costs = calc_costs(1, avg_prem, actual_qty)
    else:
        s_opt = "CE" if action == "SELL_CALL_SPREAD" else "PE"
        sell_exit = bs_premium(exit_spot, sell_strike, exit_dte, entry_vix, s_opt)
        buy_exit = bs_premium(exit_spot, buy_strike, exit_dte, entry_vix, s_opt)
        sell_entry = bs_premium(entry_spot, sell_strike, dte_entry, entry_vix, s_opt)
        buy_entry = bs_premium(entry_spot, buy_strike, dte_entry, entry_vix, s_opt)
        pnl = ((sell_entry - sell_exit) - (buy_entry - buy_exit)) * actual_qty
        avg_prem = (sell_entry + buy_entry) / 2
        costs = calc_costs(2, avg_prem, actual_qty)

    return round(pnl, 2), round(costs, 2), num_lots, exit_bar, exit_reason


def _generate_intraday_path(open_price, high, low, close, n_bars):
    """Generate a realistic intraday price path using OHLC data."""
    path = [open_price]

    # Determine if trend is up or down
    up_trend = close > open_price

    # Find the extreme point timing (random-ish but realistic)
    np.random.seed(int(abs(open_price * 100)) % 2**31)

    if up_trend:
        # Trend up: might dip first, then rally, then settle
        low_bar = max(1, int(n_bars * 0.15))  # Dip early
        high_bar = max(low_bar + 2, int(n_bars * 0.7))  # Peak later

        for i in range(1, n_bars):
            if i <= low_bar:
                # Initial dip
                t = i / low_bar
                target = open_price + (low - open_price) * t
            elif i <= high_bar:
                # Rally to high
                t = (i - low_bar) / (high_bar - low_bar)
                target = low + (high - low) * t
            else:
                # Settle to close
                t = (i - high_bar) / (n_bars - high_bar)
                target = high + (close - high) * t

            # Add micro noise (0.02% of spot)
            noise = target * 0.0002 * np.random.randn()
            path.append(target + noise)
    else:
        # Trend down: might pop first, then sell off, then settle
        high_bar = max(1, int(n_bars * 0.15))
        low_bar = max(high_bar + 2, int(n_bars * 0.7))

        for i in range(1, n_bars):
            if i <= high_bar:
                t = i / high_bar
                target = open_price + (high - open_price) * t
            elif i <= low_bar:
                t = (i - high_bar) / (low_bar - high_bar)
                target = high + (low - high) * t
            else:
                t = (i - low_bar) / (n_bars - low_bar)
                target = low + (close - low) * t

            noise = target * 0.0002 * np.random.randn()
            path.append(target + noise)

    return path


def run_full_analysis(capital=200000.0):
    import yfinance as yf

    np.random.seed(42)

    print("=" * 80)
    print("  COMPREHENSIVE 6-MONTH ANALYSIS — DYNAMIC EXIT OPTIMIZATION")
    print(f"  Capital: Rs {capital:,.0f} | Lot: {LOT_SIZE}")
    print("=" * 80)

    # Download data
    nifty = yf.download("^NSEI", start="2025-10-01", end="2026-04-04", interval="1d")
    vix_data = yf.download("^INDIAVIX", start="2025-10-01", end="2026-04-04", interval="1d")
    if isinstance(nifty.columns, pd.MultiIndex):
        nifty.columns = nifty.columns.get_level_values(0)
    if isinstance(vix_data.columns, pd.MultiIndex):
        vix_data.columns = vix_data.columns.get_level_values(0)

    nifty["VIX"] = vix_data["Close"]
    nifty["Change%"] = nifty["Close"].pct_change() * 100
    nifty["PrevChange%"] = nifty["Change%"].shift(1)
    nifty["PrevVIX"] = nifty["VIX"].shift(1)
    nifty["DOW"] = nifty.index.day_name()

    # SMAs
    nifty["SMA50"] = nifty["Close"].rolling(50).mean()
    nifty["SMA20"] = nifty["Close"].rolling(20).mean()
    nifty["AboveSMA50"] = nifty["Close"] > nifty["SMA50"]
    nifty["AboveSMA20"] = nifty["Close"] > nifty["SMA20"]

    # RSI
    delta = nifty["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    nifty["RSI"] = 100 - (100 / (1 + gain / loss.replace(0, 0.001)))

    # Close prices array for S/R computation
    close_prices = nifty["Close"].values.tolist()

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PHASE 1: Test ALL exit strategies on ALL actions on ALL days
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    actions = ["BUY_CALL", "BUY_PUT", "SELL_CALL_SPREAD", "SELL_PUT_SPREAD"]
    exit_strategies = ["sr_trail", "sr_fixed", "trail_pct", "sma_exit", "vix_adaptive"]

    print("\n  PHASE 1: Testing 4 actions x 5 exit strategies on all days...")
    print("  " + "-" * 60)

    strategy_results = defaultdict(lambda: {"pnl": 0, "wins": 0, "losses": 0,
                                             "trades": 0, "exit_reasons": defaultdict(int)})

    all_results = []

    for i in range(1, len(nifty)):
        row = nifty.iloc[i]
        prev = nifty.iloc[i - 1]

        entry_spot = float(row["Open"])
        day_high = float(row["High"])
        day_low = float(row["Low"])
        day_close = float(row["Close"])
        vix = float(row["VIX"]) if pd.notna(row["VIX"]) else 14.0
        prev_vix = float(prev["VIX"]) if pd.notna(prev["VIX"]) else vix
        vix_regime = "LOW" if vix < 12 else "NORMAL" if vix < 20 else "HIGH" if vix < 30 else "EXTREME"
        dow = str(row["DOW"])
        above_sma50 = bool(row["AboveSMA50"]) if pd.notna(row["AboveSMA50"]) else True
        above_sma20 = bool(row["AboveSMA20"]) if pd.notna(row["AboveSMA20"]) else True
        rsi = float(row["RSI"]) if pd.notna(row["RSI"]) else 50
        prev_change = float(prev["Change%"]) if pd.notna(prev["Change%"]) else 0
        vix_spike = prev_vix > 0 and vix > prev_vix * 1.15
        sma20 = float(row["SMA20"]) if pd.notna(row["SMA20"]) else None
        sma50 = float(row["SMA50"]) if pd.notna(row["SMA50"]) else None

        # Dynamic support/resistance
        support, resistance = compute_support_resistance(close_prices, i, lookback=20)

        for action in actions:
            for exit_strat in exit_strategies:
                pnl, costs, lots, exit_bar, exit_reason = simulate_intraday(
                    action, entry_spot, day_high, day_low, day_close,
                    vix, support, resistance, sma20, sma50,
                    exit_strat, LOT_SIZE, capital
                )
                net = pnl - costs

                key = f"{action}|{exit_strat}"
                strategy_results[key]["pnl"] += net
                strategy_results[key]["trades"] += 1
                if net > 0:
                    strategy_results[key]["wins"] += 1
                else:
                    strategy_results[key]["losses"] += 1
                strategy_results[key]["exit_reasons"][exit_reason] += 1

                all_results.append({
                    "date": str(nifty.index[i].date()),
                    "action": action,
                    "exit_strategy": exit_strat,
                    "entry": entry_spot,
                    "exit": day_close,
                    "vix": vix,
                    "vix_regime": vix_regime,
                    "day_change": round(float(row["Change%"]) if pd.notna(row["Change%"]) else 0, 3),
                    "dow": dow,
                    "above_sma50": above_sma50,
                    "rsi": round(rsi, 1),
                    "support": support,
                    "resistance": resistance,
                    "gross_pnl": pnl,
                    "costs": costs,
                    "net_pnl": net,
                    "lots": lots,
                    "exit_bar": exit_bar,
                    "exit_reason": exit_reason,
                })

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PHASE 2: Find BEST exit strategy per action
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    print("\n" + "=" * 80)
    print("  PHASE 2: EXIT STRATEGY COMPARISON")
    print("=" * 80)

    best_exits = {}

    for action in actions:
        print(f"\n  {action}:")
        best_pnl = -float('inf')
        best_strat = None

        for exit_strat in exit_strategies:
            key = f"{action}|{exit_strat}"
            sr = strategy_results[key]
            wr = sr["wins"] / max(1, sr["trades"]) * 100
            print(f"    {exit_strat:15s}: P&L=Rs {sr['pnl']:>+12,.0f} "
                  f"WR={wr:.0f}% ({sr['wins']}/{sr['trades']})"
                  f"  exits: {dict(sr['exit_reasons'])}")

            if sr["pnl"] > best_pnl:
                best_pnl = sr["pnl"]
                best_strat = exit_strat

        best_exits[action] = best_strat
        print(f"    >>> BEST: {best_strat} (P&L=Rs {best_pnl:+,.0f})")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PHASE 3: Learn ENTRY rules using the best exit per action
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    df = pd.DataFrame(all_results)

    # Filter to best exit strategy per action
    best_df_parts = []
    for action, exit_strat in best_exits.items():
        part = df[(df.action == action) & (df.exit_strategy == exit_strat)]
        best_df_parts.append(part)
    best_df = pd.concat(best_df_parts)

    print("\n" + "=" * 80)
    print("  PHASE 3: LEARNED ENTRY RULES (using best exit per action)")
    print("=" * 80)

    # 1. Best action per VIX regime
    print("\n  1. BEST ACTION PER VIX REGIME:")
    rules_vix = {}
    for regime in ["LOW", "NORMAL", "HIGH", "EXTREME"]:
        sub = best_df[best_df.vix_regime == regime]
        if sub.empty:
            continue
        action_pnl = sub.groupby("action")["net_pnl"].agg(["sum", "count", "mean"])
        action_wr = sub.groupby("action").apply(
            lambda x: (x.net_pnl > 0).mean() * 100, include_groups=False
        ).round(1)
        action_pnl["win_rate"] = action_wr
        best_action = action_pnl["sum"].idxmax()
        stats = action_pnl.loc[best_action]
        rules_vix[regime] = best_action
        print(f"     {regime:8s} -> {best_action:20s} "
              f"P&L=Rs {stats['sum']:>+10,.0f} WR={stats['win_rate']:.0f}% "
              f"({int(stats['count'])} days)")
        for act in actions:
            if act in action_pnl.index and act != best_action:
                s = action_pnl.loc[act]
                print(f"              {act:20s} P&L=Rs {s['sum']:>+10,.0f} WR={s['win_rate']:.0f}%")

    # 2. Best action by trend
    print("\n  2. BEST ACTION BY TREND:")
    rules_trend = {}
    for trend_label, is_above in [("ABOVE SMA50 (uptrend)", True), ("BELOW SMA50 (downtrend)", False)]:
        sub = best_df[best_df.above_sma50 == is_above]
        if sub.empty:
            continue
        best = sub.groupby("action")["net_pnl"].sum()
        best_action = best.idxmax()
        wr = (sub[sub.action == best_action]["net_pnl"] > 0).mean() * 100
        rules_trend["above_sma50" if is_above else "below_sma50"] = best_action
        print(f"     {trend_label:30s} -> {best_action:20s} "
              f"P&L=Rs {best[best_action]:>+10,.0f} WR={wr:.0f}%")

    # 3. Best action by RSI
    print("\n  3. BEST ACTION BY RSI:")
    rules_rsi = {}
    for rsi_label, lo, hi, key in [
        ("Oversold (RSI<30)", 0, 30, "oversold_lt_30"),
        ("Neutral (30-70)", 30, 70, "neutral"),
        ("Overbought (RSI>70)", 70, 100, "overbought_gt_70"),
    ]:
        sub = best_df[(best_df.rsi >= lo) & (best_df.rsi < hi)]
        if sub.empty or len(sub) < 4:
            continue
        best = sub.groupby("action")["net_pnl"].sum()
        best_action = best.idxmax()
        wr = (sub[sub.action == best_action]["net_pnl"] > 0).mean() * 100
        rules_rsi[key] = best_action
        print(f"     {rsi_label:25s} -> {best_action:20s} "
              f"P&L=Rs {best[best_action]:>+10,.0f} WR={wr:.0f}%")

    # 4. Best action by day of week
    print("\n  4. BEST ACTION BY DAY OF WEEK:")
    rules_dow = {}
    for dow in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]:
        sub = best_df[best_df.dow == dow]
        if sub.empty:
            continue
        best = sub.groupby("action")["net_pnl"].sum()
        best_action = best.idxmax()
        wr = (sub[sub.action == best_action]["net_pnl"] > 0).mean() * 100
        rules_dow[dow] = best_action
        print(f"     {dow:10s} -> {best_action:20s} "
              f"P&L=Rs {best[best_action]:>+8,.0f} WR={wr:.0f}%")

    # 5. Previous day momentum
    print("\n  5. PREVIOUS DAY MOMENTUM:")
    rules_momentum = {}
    for label, lo, hi, key in [
        ("Prev day DOWN >1%", -99, -1, "prev_day_down_gt_1pct"),
        ("Prev day flat", -0.5, 0.5, "prev_day_flat"),
        ("Prev day UP >1%", 1, 99, "prev_day_up_gt_1pct"),
    ]:
        sub = best_df[(best_df.day_change.shift(1) if "prev" in label.lower() else best_df.day_change).between(lo, hi, inclusive="left")]
        # Use day_change as proxy for previous day
        if sub.empty or len(sub) < 4:
            continue
        best = sub.groupby("action")["net_pnl"].sum()
        best_action = best.idxmax()
        wr = (sub[sub.action == best_action]["net_pnl"] > 0).mean() * 100
        rules_momentum[key] = best_action
        print(f"     {label:22s} -> {best_action:20s} "
              f"P&L=Rs {best[best_action]:>+8,.0f} WR={wr:.0f}%")

    # 6. Exit reason analysis
    print("\n  6. EXIT REASON ANALYSIS (using best strategies):")
    exit_reason_pnl = defaultdict(lambda: {"pnl": 0, "count": 0, "wins": 0})
    for _, row_data in best_df.iterrows():
        reason = row_data["exit_reason"]
        net = row_data["net_pnl"]
        exit_reason_pnl[reason]["pnl"] += net
        exit_reason_pnl[reason]["count"] += 1
        if net > 0:
            exit_reason_pnl[reason]["wins"] += 1

    for reason, data in sorted(exit_reason_pnl.items(), key=lambda x: -x[1]["pnl"]):
        wr = data["wins"] / max(1, data["count"]) * 100
        print(f"     {reason:20s}: {data['count']:4d} exits, "
              f"P&L=Rs {data['pnl']:>+10,.0f} WR={wr:.0f}%")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PHASE 4: COMPOSITE STRATEGY with optimal entries + exits
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    print("\n" + "=" * 80)
    print("  PHASE 4: COMPOSITE STRATEGY (Optimal Entries + Exits)")
    print("=" * 80)

    equity = capital
    equity_curve = [capital]
    composite_trades = []

    for i in range(1, len(nifty)):
        row = nifty.iloc[i]
        prev = nifty.iloc[i - 1]

        entry_spot = float(row["Open"])
        day_high = float(row["High"])
        day_low = float(row["Low"])
        day_close = float(row["Close"])
        vix = float(row["VIX"]) if pd.notna(row["VIX"]) else 14.0
        prev_vix = float(prev["VIX"]) if pd.notna(prev["VIX"]) else vix
        vix_regime = "LOW" if vix < 12 else "NORMAL" if vix < 20 else "HIGH" if vix < 30 else "EXTREME"
        dow = str(row["DOW"])
        above_sma50 = bool(row["AboveSMA50"]) if pd.notna(row["AboveSMA50"]) else True
        above_sma20 = bool(row["AboveSMA20"]) if pd.notna(row["AboveSMA20"]) else True
        rsi = float(row["RSI"]) if pd.notna(row["RSI"]) else 50
        prev_change = float(prev["Change%"]) if pd.notna(prev["Change%"]) else 0
        vix_spike = prev_vix > 0 and vix > prev_vix * 1.15
        sma20 = float(row["SMA20"]) if pd.notna(row["SMA20"]) else None
        sma50 = float(row["SMA50"]) if pd.notna(row["SMA50"]) else None

        support, resistance = compute_support_resistance(close_prices, i, lookback=20)

        # ── COMPOSITE SCORING ────────────────────────────────────
        scores = {a: 0.0 for a in actions}

        # Rule 1: VIX regime (weight 3.0)
        vix_best = rules_vix.get(vix_regime, "BUY_PUT")
        scores[vix_best] += 3.0

        # Rule 2: Trend (weight 2.0)
        if not above_sma50:
            trend_action = rules_trend.get("below_sma50", "BUY_PUT")
            scores[trend_action] += 2.0
        else:
            trend_action = rules_trend.get("above_sma50", "BUY_CALL")
            scores[trend_action] += 2.0

        # Rule 3: SMA20 trend (weight 1.0)
        if not above_sma20:
            scores["BUY_PUT"] += 1.0
        else:
            scores["BUY_CALL"] += 1.0

        # Rule 4: RSI (weight 1.5)
        if rsi < 30:
            rsi_action = rules_rsi.get("oversold_lt_30", "BUY_CALL")
            scores[rsi_action] += 1.5
        elif rsi > 70:
            rsi_action = rules_rsi.get("overbought_gt_70", "BUY_PUT")
            scores[rsi_action] += 1.5

        # Rule 5: Day of week (weight 0.5)
        dow_action = rules_dow.get(dow)
        if dow_action:
            scores[dow_action] += 0.5

        # Rule 6: VIX spike recovery (weight 2.0)
        if vix_spike:
            scores["BUY_CALL"] += 2.0

        # Rule 7: Previous day momentum (weight 1.0)
        if prev_change < -1.0:
            mom_action = rules_momentum.get("prev_day_down_gt_1pct", "BUY_CALL")
            scores[mom_action] += 1.0
        elif prev_change > 1.0:
            mom_action = rules_momentum.get("prev_day_up_gt_1pct", "BUY_PUT")
            scores[mom_action] += 1.0

        # Rule 8: S/R proximity (weight 1.0)
        if support and entry_spot:
            dist_to_support = (entry_spot - support) / entry_spot * 100
            if 0 < dist_to_support < 1.0:
                scores["BUY_CALL"] += 1.0  # Near support = bounce potential
            elif dist_to_support < 0:
                scores["BUY_PUT"] += 1.0  # Broken support = bearish

        if resistance and entry_spot:
            dist_to_resistance = (resistance - entry_spot) / entry_spot * 100
            if 0 < dist_to_resistance < 1.0:
                scores["BUY_PUT"] += 1.0  # Near resistance = rejection potential
            elif dist_to_resistance < 0:
                scores["BUY_CALL"] += 1.0  # Broken resistance = bullish

        # Pick best action
        best_action = max(scores, key=scores.get)
        total_score = sum(scores.values())
        confidence = scores[best_action] / total_score if total_score > 0 else 0

        if confidence < 0.25:
            equity_curve.append(equity)
            continue

        # Use the BEST exit strategy for this action
        exit_strat = best_exits[best_action]

        pnl, costs, lots, exit_bar, exit_reason = simulate_intraday(
            best_action, entry_spot, day_high, day_low, day_close,
            vix, support, resistance, sma20, sma50,
            exit_strat, LOT_SIZE, capital
        )
        net = pnl - costs
        equity += net
        equity_curve.append(equity)

        composite_trades.append({
            "date": str(nifty.index[i].date()),
            "action": best_action,
            "exit_strategy": exit_strat,
            "exit_reason": exit_reason,
            "exit_bar": exit_bar,
            "confidence": round(confidence, 2),
            "scores": {k: round(v, 1) for k, v in scores.items()},
            "entry": round(entry_spot, 0),
            "exit": round(day_close, 0),
            "support": support,
            "resistance": resistance,
            "vix": round(vix, 1),
            "vix_regime": vix_regime,
            "net_pnl": round(net, 0),
            "lots": lots,
        })

    # ── RESULTS ──────────────────────────────────────────────────
    wins = len([t for t in composite_trades if t["net_pnl"] > 0])
    total = len(composite_trades)
    total_pnl = sum(t["net_pnl"] for t in composite_trades)

    print(f"\n  Total Days Traded: {total}/122")
    print(f"  Win Rate: {wins}/{total} ({wins/max(1,total)*100:.1f}%)")
    print(f"  Total Net P&L: Rs {total_pnl:+,.0f}")
    print(f"  Return: {total_pnl/capital*100:+.2f}%")
    print(f"  Final Capital: Rs {equity:,.0f}")

    # Max drawdown
    peak = equity_curve[0]
    max_dd = 0
    for v in equity_curve:
        if v > peak:
            peak = v
        dd = (peak - v) / peak * 100
        max_dd = max(max_dd, dd)
    print(f"  Max Drawdown: {max_dd:.2f}%")

    # Sharpe
    daily_pnls = [t["net_pnl"] for t in composite_trades]
    if len(daily_pnls) > 1 and np.std(daily_pnls) > 0:
        sharpe = np.mean(daily_pnls) / np.std(daily_pnls) * np.sqrt(252)
        print(f"  Sharpe Ratio: {sharpe:.2f}")

    # Per-action breakdown
    print("\n  PER-ACTION BREAKDOWN:")
    for action in actions:
        at = [t for t in composite_trades if t["action"] == action]
        if not at:
            continue
        w = len([t for t in at if t["net_pnl"] > 0])
        pnl = sum(t["net_pnl"] for t in at)
        wr = w / len(at) * 100
        exit_strat = best_exits[action]
        print(f"    {action:20s}: {len(at):3d} trades, WR={wr:.0f}%, "
              f"P&L=Rs {pnl:>+10,.0f} | exit={exit_strat}")

    # Exit reason breakdown
    print("\n  EXIT REASON BREAKDOWN:")
    exit_counts = defaultdict(lambda: {"count": 0, "pnl": 0, "wins": 0})
    for t in composite_trades:
        r = t["exit_reason"]
        exit_counts[r]["count"] += 1
        exit_counts[r]["pnl"] += t["net_pnl"]
        if t["net_pnl"] > 0:
            exit_counts[r]["wins"] += 1
    for reason, data in sorted(exit_counts.items(), key=lambda x: -x[1]["pnl"]):
        wr = data["wins"] / max(1, data["count"]) * 100
        print(f"    {reason:20s}: {data['count']:3d} exits, "
              f"P&L=Rs {data['pnl']:>+10,.0f} WR={wr:.0f}%")

    # Monthly P&L
    print("\n  MONTHLY P&L:")
    monthly_pnl = defaultdict(float)
    for t in composite_trades:
        month = t["date"][:7]
        monthly_pnl[month] += t["net_pnl"]
    for month, pnl in sorted(monthly_pnl.items()):
        bar = "#" * max(1, int(abs(pnl) / 5000))
        direction = "+" if pnl > 0 else "-"
        print(f"    {month}: Rs {pnl:>+10,.0f} {direction}{bar}")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PHASE 5: TODAY'S TRADE RECOMMENDATION
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    print("\n" + "=" * 80)
    print("  TODAY'S RECOMMENDED TRADE (Dynamic Exits)")
    print("=" * 80)

    current_spot = float(nifty["Close"].iloc[-1])
    current_vix = float(nifty["VIX"].iloc[-1]) if pd.notna(nifty["VIX"].iloc[-1]) else 25.0
    current_sma50 = float(nifty["SMA50"].iloc[-1]) if pd.notna(nifty["SMA50"].iloc[-1]) else current_spot
    current_sma20 = float(nifty["SMA20"].iloc[-1]) if pd.notna(nifty["SMA20"].iloc[-1]) else current_spot
    current_rsi = float(nifty["RSI"].iloc[-1]) if pd.notna(nifty["RSI"].iloc[-1]) else 50
    prev_change_today = float(nifty["Change%"].iloc[-1]) if pd.notna(nifty["Change%"].iloc[-1]) else 0
    today_dow = datetime.now().strftime("%A")
    vix_regime = "LOW" if current_vix < 12 else "NORMAL" if current_vix < 20 else "HIGH" if current_vix < 30 else "EXTREME"

    # Compute current S/R
    support, resistance = compute_support_resistance(close_prices, len(close_prices) - 1, lookback=20)

    print(f"\n  Current Conditions:")
    print(f"    NIFTY:       {current_spot:.0f}")
    print(f"    VIX:         {current_vix:.1f} ({vix_regime})")
    print(f"    20-SMA:      {current_sma20:.0f} ({'ABOVE' if current_spot > current_sma20 else 'BELOW'})")
    print(f"    50-SMA:      {current_sma50:.0f} ({'ABOVE' if current_spot > current_sma50 else 'BELOW'})")
    print(f"    RSI:         {current_rsi:.1f}")
    print(f"    Prev Day:    {prev_change_today:+.2f}%")
    print(f"    Day:         {today_dow}")
    print(f"    Support:     {support}")
    print(f"    Resistance:  {resistance}")

    # Score actions
    scores = {a: 0.0 for a in actions}
    score_log = []

    # VIX regime
    vix_action = rules_vix.get(vix_regime, "BUY_PUT")
    scores[vix_action] += 3.0
    score_log.append(f"VIX {vix_regime} -> {vix_action} (+3.0)")

    # Trend
    if current_spot < current_sma50:
        trend_act = rules_trend.get("below_sma50", "BUY_PUT")
        scores[trend_act] += 2.0
        score_log.append(f"Below SMA50 -> {trend_act} (+2.0)")
    else:
        trend_act = rules_trend.get("above_sma50", "BUY_CALL")
        scores[trend_act] += 2.0
        score_log.append(f"Above SMA50 -> {trend_act} (+2.0)")

    # SMA20
    if current_spot < current_sma20:
        scores["BUY_PUT"] += 1.0
        score_log.append("Below SMA20 -> BUY_PUT (+1.0)")
    else:
        scores["BUY_CALL"] += 1.0
        score_log.append("Above SMA20 -> BUY_CALL (+1.0)")

    # RSI
    if current_rsi < 30:
        rsi_act = rules_rsi.get("oversold_lt_30", "BUY_CALL")
        scores[rsi_act] += 1.5
        score_log.append(f"RSI {current_rsi:.0f} (oversold) -> {rsi_act} (+1.5)")
    elif current_rsi > 70:
        rsi_act = rules_rsi.get("overbought_gt_70", "BUY_PUT")
        scores[rsi_act] += 1.5
        score_log.append(f"RSI {current_rsi:.0f} (overbought) -> {rsi_act} (+1.5)")
    else:
        score_log.append(f"RSI {current_rsi:.0f} (neutral) -> no adjustment")

    # Day of week
    dow_act = rules_dow.get(today_dow)
    if dow_act:
        scores[dow_act] += 0.5
        score_log.append(f"{today_dow} -> {dow_act} (+0.5)")

    # Prev day momentum
    if prev_change_today < -1.0:
        mom_act = rules_momentum.get("prev_day_down_gt_1pct", "BUY_CALL")
        scores[mom_act] += 1.0
        score_log.append(f"Prev {prev_change_today:+.1f}% (big drop) -> {mom_act} (+1.0)")
    elif prev_change_today > 1.0:
        mom_act = rules_momentum.get("prev_day_up_gt_1pct", "BUY_PUT")
        scores[mom_act] += 1.0
        score_log.append(f"Prev {prev_change_today:+.1f}% (big rally) -> {mom_act} (+1.0)")

    # S/R proximity
    if support:
        dist = (current_spot - support) / current_spot * 100
        if 0 < dist < 1.0:
            scores["BUY_CALL"] += 1.0
            score_log.append(f"Near support {support} ({dist:.1f}%) -> BUY_CALL (+1.0)")
    if resistance:
        dist = (resistance - current_spot) / current_spot * 100
        if 0 < dist < 1.0:
            scores["BUY_PUT"] += 1.0
            score_log.append(f"Near resistance {resistance} ({dist:.1f}%) -> BUY_PUT (+1.0)")

    print(f"\n  Rule Scoring:")
    for log in score_log:
        print(f"    {log}")

    best_action = max(scores, key=scores.get)
    total_score = sum(scores.values())
    confidence = scores[best_action] / total_score if total_score > 0 else 0
    exit_strat = best_exits.get(best_action, "vix_adaptive")

    print(f"\n  Scores: {json.dumps({k: round(v, 1) for k, v in scores.items()})}")
    print(f"  Best Action: {best_action} (confidence: {confidence*100:.0f}%)")
    print(f"  Exit Strategy: {exit_strat}")

    # Trade details
    strike = round(current_spot / 50) * 50
    risk_budget = capital * 0.08
    num_lots = max(1, int(risk_budget / (50 * LOT_SIZE)))
    qty = LOT_SIZE * num_lots

    if best_action in ("BUY_CALL", "BUY_PUT"):
        opt_type = "CE" if "CALL" in best_action else "PE"
        premium = bs_premium(current_spot, strike, 2.0, current_vix, opt_type)
        total_cost = premium * qty

        print(f"\n  {'='*60}")
        print(f"  TRADE: {best_action}")
        print(f"  {'='*60}")
        print(f"    Strike:        NIFTY {strike} {opt_type} (ATM)")
        print(f"    Side:          BUY")
        print(f"    Lots:          {num_lots} x {LOT_SIZE} = {qty} qty")
        print(f"    Premium:       Rs {premium:.2f} per unit")
        print(f"    Total Cost:    Rs {total_cost:,.0f}")
        print(f"    Exit Strategy: {exit_strat}")
        if support:
            print(f"    Stop Loss:     Support breach at {support} "
                  f"(Rs {abs(current_spot - support):.0f} away, "
                  f"{abs(current_spot - support)/current_spot*100:.2f}%)")
        if resistance:
            print(f"    Resistance:    {resistance} "
                  f"(Rs {abs(resistance - current_spot):.0f} away, "
                  f"{abs(resistance - current_spot)/current_spot*100:.2f}%)")
        print(f"    Max Loss:      Rs {total_cost:,.0f} ({total_cost/capital*100:.1f}% of capital)")
        print(f"    Confidence:    {confidence*100:.0f}%")
    else:
        atm_iv = current_vix / 100 * 0.88
        exp_move = current_spot * atm_iv * math.sqrt(2 / 365)
        if "PUT" in best_action:
            sell_strike = round((current_spot - exp_move) / 50) * 50
            buy_strike = sell_strike - 50
            opt_type = "PE"
        else:
            sell_strike = round((current_spot + exp_move) / 50) * 50
            buy_strike = sell_strike + 50
            opt_type = "CE"

        sell_prem = bs_premium(current_spot, sell_strike, 2.0, current_vix, opt_type)
        buy_prem = bs_premium(current_spot, buy_strike, 2.0, current_vix, opt_type)
        credit = sell_prem - buy_prem
        max_loss = (50 - credit) * qty

        print(f"\n  {'='*60}")
        print(f"  TRADE: {best_action}")
        print(f"  {'='*60}")
        print(f"    Sell Strike:   NIFTY {sell_strike} {opt_type}")
        print(f"    Buy Strike:    NIFTY {buy_strike} {opt_type}")
        print(f"    Lots:          {num_lots} x {LOT_SIZE} = {qty} qty")
        print(f"    Net Credit:    Rs {credit:.2f}/unit (Rs {credit * qty:,.0f} total)")
        print(f"    Max Loss:      Rs {max_loss:,.0f}")
        print(f"    Exit Strategy: {exit_strat}")
        if support:
            print(f"    Support:       {support}")
        if resistance:
            print(f"    Resistance:    {resistance}")
        print(f"    Confidence:    {confidence*100:.0f}%")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # SAVE COMPREHENSIVE RULES
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    output_dir = project_root / "data"
    output_dir.mkdir(exist_ok=True)

    comprehensive_rules = {
        "vix_rules": rules_vix,
        "trend_rules": rules_trend,
        "rsi_rules": rules_rsi,
        "dow_rules": rules_dow,
        "momentum_rules": rules_momentum,
        "best_exit_per_action": best_exits,
        "exit_strategy_params": {
            "sr_trail": {
                "description": "S/R stop + trailing stop at 40% retrace of peak profit",
                "trailing_retrace": 0.40,
                "min_profit_for_trail": 500,
            },
            "sr_fixed": {
                "description": "S/R stop only, hold to EOD otherwise",
            },
            "trail_pct": {
                "description": "Trailing stop at 0.3% of spot from peak/trough",
                "trail_pct": 0.003,
                "min_bars_before_trail": 3,
            },
            "sma_exit": {
                "description": "Exit on SMA20 cross against direction",
                "sma_period": 20,
                "min_bars": 5,
            },
            "vix_adaptive": {
                "description": "VIX-weighted trailing: tighter in high VIX, wider in low",
                "high_vix_trail_mult": 0.25,
                "normal_vix_trail_mult": 0.35,
                "low_vix_trail_mult": 0.45,
                "high_vix_sr_buffer_pct": 0.002,
                "normal_vix_sr_buffer_pct": 0.003,
                "low_vix_sr_buffer_pct": 0.004,
                "high_vix_profit_target_mult": 0.5,
            },
        },
        "rule_weights": {
            "vix_regime": 3.0,
            "trend_sma50": 2.0,
            "trend_sma20": 1.0,
            "rsi": 1.5,
            "day_of_week": 0.5,
            "vix_spike": 2.0,
            "prev_momentum": 1.0,
            "sr_proximity": 1.0,
        },
        "risk_management": {
            "risk_budget_pct": 0.08,
            "max_loss_per_lot": 50 * LOT_SIZE,
            "capital": capital,
        },
        "macro_context": {
            "current_regime": "CRISIS_BEARISH",
            "nifty_ath": 26373,
            "nifty_ath_date": "2026-01-05",
            "decline_from_ath_pct": 15.9,
            "iran_war_started": "2026-02-28",
            "brent_crude": 94,
            "fed_rate": "3.50-3.75",
            "rbi_rate": 5.25,
            "fii_march_selling_cr": -113810,
            "fii_trend": "HEAVY_SELLING",
            "key_support": [22000, 21700, 21800],
            "key_resistance": [22800, 23000, 23500],
        },
        "today_recommendation": {
            "action": best_action,
            "confidence": round(confidence, 2),
            "exit_strategy": exit_strat,
            "scores": {k: round(v, 1) for k, v in scores.items()},
            "conditions": {
                "spot": round(current_spot, 0),
                "vix": round(current_vix, 1),
                "vix_regime": vix_regime,
                "rsi": round(current_rsi, 1),
                "above_sma50": current_spot > current_sma50,
                "above_sma20": current_spot > current_sma20,
                "day": today_dow,
                "support": support,
                "resistance": resistance,
            },
        },
        "backtest_performance": {
            "total_trades": total,
            "win_rate": round(wins / max(1, total) * 100, 1),
            "total_pnl": round(total_pnl, 0),
            "return_pct": round(total_pnl / capital * 100, 2),
            "max_drawdown": round(max_dd, 2),
        },
    }

    with open(output_dir / "learned_rules.json", "w") as f:
        json.dump(comprehensive_rules, f, indent=2, default=str)

    trades_df = pd.DataFrame(composite_trades)
    trades_df.to_csv(output_dir / "composite_trades.csv", index=False)

    print(f"\n  Rules saved to data/learned_rules.json")
    print(f"  Trades saved to data/composite_trades.csv")
    print("=" * 80)

    return comprehensive_rules


if __name__ == "__main__":
    run_full_analysis(capital=200000.0)
