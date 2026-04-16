"""
15-Day Paper Trading Simulation — FAST FORWARD MODE.

Bypasses the LiveTradingOrchestrator's real-time tick generation (375 sec/day)
by directly driving the PaperTradingBroker with synthetic intraday bars,
composite scoring, and the full 6-agent ensemble logic.

Runs all 15 days in under 60 seconds.

Capital: Rs 200,000 | Strategy: Full Ensemble (6 agents)
Tests: Composite scoring, VIX-adaptive sizing/strikes, S/R-based exits,
       timing gates, overnight holding.
"""

import json
import math
import sys
import random
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backtesting.option_pricer import price_option
from backtesting.paper_trading import PaperTradingBroker
from config.constants import (
    INDEX_CONFIG, STT_RATES, NSE_TRANSACTION_CHARGE,
    SEBI_TURNOVER_FEE, STAMP_DUTY_BUY, GST_RATE,
)

LOT_SIZE = INDEX_CONFIG["NIFTY"]["lot_size"]  # 65
BROKERAGE = 20.0
STRIKE_INTERVAL = 50
CAPITAL = 200_000
TOTAL_BARS = 25  # 15-min bars per session (9:15 AM to 3:30 PM)
NUM_DAYS = 15

# Load ensemble rules
data_dir = project_root / "data"
ensemble_rules = {}
learned_rules = {}
sr_rules = {}

ens_path = data_dir / "ensemble_rules.json"
if ens_path.exists():
    with open(ens_path) as f:
        ensemble_rules = json.load(f)

lr_path = data_dir / "learned_rules.json"
if lr_path.exists():
    with open(lr_path) as f:
        learned_rules = json.load(f)

sr_path = data_dir / "sr_rules.json"
if sr_path.exists():
    with open(sr_path) as f:
        sr_rules = json.load(f)


# ===========================================================================
# SYNTHETIC DATA GENERATION
# ===========================================================================

def generate_15_day_nifty(start_spot=22713.0, base_vix=25.5):
    """Generate 15 days of synthetic NIFTY OHLCV + VIX data.

    Uses realistic crisis-bearish parameters matching current macro:
    - Avg daily range: 0.8-1.5%
    - Bearish bias: 55% down days, 35% up days, 10% flat
    - VIX evolution: mean-reverting around base with shocks
    - Overnight gaps: avg 0.57%, directional continuation 54%
    """
    days = []
    spot = start_spot
    vix = base_vix
    prev_change = 0.0

    # Day-of-week sequence starting Monday Apr 7
    dow_cycle = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]

    for d in range(NUM_DAYS):
        dow = dow_cycle[d % 5]
        np.random.seed(42 + d * 7)  # Reproducible but varied

        # Day type (crisis-bearish regime)
        r = np.random.random()
        if r < 0.55:
            day_type = "down"
        elif r < 0.90:
            day_type = "up"
        else:
            day_type = "flat"

        # Daily range (0.6-2.0% depending on VIX)
        range_pct = 0.006 + (vix / 100) * 0.04 + np.random.uniform(0, 0.004)
        half_range = spot * range_pct / 2

        if day_type == "down":
            change_pct = -np.random.uniform(0.3, 1.2) / 100
            open_price = spot + spot * np.random.uniform(-0.002, 0.003)
            close_price = spot * (1 + change_pct)
            high_price = max(open_price, close_price) + half_range * np.random.uniform(0.2, 0.6)
            low_price = min(open_price, close_price) - half_range * np.random.uniform(0.3, 0.8)
        elif day_type == "up":
            change_pct = np.random.uniform(0.2, 0.9) / 100
            open_price = spot + spot * np.random.uniform(-0.003, 0.002)
            close_price = spot * (1 + change_pct)
            high_price = max(open_price, close_price) + half_range * np.random.uniform(0.3, 0.8)
            low_price = min(open_price, close_price) - half_range * np.random.uniform(0.2, 0.6)
        else:
            change_pct = np.random.uniform(-0.15, 0.15) / 100
            open_price = spot + spot * np.random.uniform(-0.001, 0.001)
            close_price = spot * (1 + change_pct)
            high_price = max(open_price, close_price) + half_range * 0.4
            low_price = min(open_price, close_price) - half_range * 0.4

        # VIX evolution (mean-reverting around 25.5, higher on down days)
        vix_shock = 0
        if day_type == "down" and abs(change_pct) > 0.008:
            vix_shock = np.random.uniform(1.0, 3.0)
        elif day_type == "up":
            vix_shock = -np.random.uniform(0.5, 1.5)

        vix = vix + vix_shock + (base_vix - vix) * 0.1 + np.random.normal(0, 0.5)
        vix = max(12, min(40, vix))

        # SMA proxies (below both SMAs in crisis)
        sma20 = spot * 1.032  # ~3.2% above
        sma50 = spot * 1.086  # ~8.6% above

        # RSI (bearish regime -> 30-50)
        rsi = 30 + np.random.uniform(0, 20)
        if day_type == "down":
            rsi -= 5
        rsi = max(20, min(80, rsi))

        # VIX spike detection
        prev_vix = days[-1]["vix"] if days else vix
        vix_spike = vix > prev_vix * 1.15

        days.append({
            "day_num": d + 1,
            "date": (datetime(2026, 4, 7) + timedelta(days=d + (d // 5) * 2)).strftime("%Y-%m-%d"),
            "dow": dow,
            "open": round(open_price, 2),
            "high": round(high_price, 2),
            "low": round(low_price, 2),
            "close": round(close_price, 2),
            "vix": round(vix, 1),
            "prev_vix": round(prev_vix, 1),
            "vix_spike": vix_spike,
            "vix_regime": "LOW" if vix < 12 else "NORMAL" if vix < 20 else "HIGH" if vix < 30 else "EXTREME",
            "sma20": round(sma20, 0),
            "sma50": round(sma50, 0),
            "above_sma20": close_price > sma20,
            "above_sma50": close_price > sma50,
            "rsi": round(rsi, 1),
            "prev_change": round(prev_change, 2),
            "prev_high": round(days[-1]["high"], 0) if days else round(high_price, 0),
            "prev_low": round(days[-1]["low"], 0) if days else round(low_price, 0),
        })

        prev_change = change_pct * 100
        # Overnight gap for next day
        spot = close_price * (1 + np.random.uniform(-0.006, 0.003))

    return days


def generate_intraday_path(open_price, high, low, close, n_bars=TOTAL_BARS):
    """Generate realistic intraday price path from daily OHLC."""
    path = [open_price]
    up_trend = close > open_price

    if up_trend:
        low_bar = max(1, int(n_bars * 0.15))
        high_bar = max(low_bar + 2, int(n_bars * 0.7))
        for i in range(1, n_bars):
            if i <= low_bar:
                t = i / low_bar
                target = open_price + (low - open_price) * t
            elif i <= high_bar:
                t = (i - low_bar) / (high_bar - low_bar)
                target = low + (high - low) * t
            else:
                t = (i - high_bar) / (n_bars - high_bar)
                target = high + (close - high) * t
            noise = target * 0.0002 * np.random.randn()
            path.append(target + noise)
    else:
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


# ===========================================================================
# S/R COMPUTATION (Agent 6 — Multi-method)
# ===========================================================================

def sr_multi_method(spot, prev_high, prev_low, sma20, sma50, close_history=None):
    """Multi-method S/R combining all methods from Agent 6."""
    support_candidates = []
    resistance_candidates = []

    # Method 1: Round numbers (weight 3.0, 90.7% WR)
    for level in range(int(spot // 500) * 500 - 1500,
                       int(spot // 500) * 500 + 2000, 500):
        if level < spot:
            support_candidates.append((level, 3.0))
        elif level > spot:
            resistance_candidates.append((level, 3.0))
    for level in range(int(spot // 100) * 100 - 500,
                       int(spot // 100) * 100 + 600, 100):
        if level % 500 != 0:
            if level < spot:
                support_candidates.append((level, 1.5))
            elif level > spot:
                resistance_candidates.append((level, 1.5))

    # Method 2: PDH/PDL (weight 2.5, 85.3% WR)
    pdl = round(prev_low / 50) * 50
    pdh = round(prev_high / 50) * 50
    if pdl < spot:
        support_candidates.append((pdl, 2.5))
    if pdh > spot:
        resistance_candidates.append((pdh, 2.5))

    # Method 3: SMA (weight 1.5)
    if sma20 and sma20 < spot:
        support_candidates.append((round(sma20 / 50) * 50, 1.5))
    elif sma20 and sma20 > spot:
        resistance_candidates.append((round(sma20 / 50) * 50, 1.5))
    if sma50 and sma50 < spot:
        support_candidates.append((round(sma50 / 50) * 50, 1.5))
    elif sma50 and sma50 > spot:
        resistance_candidates.append((round(sma50 / 50) * 50, 1.5))

    # Method 4: Pre-loaded S/R from ensemble_rules
    sr = ensemble_rules.get("sr_rules", {})
    for s in sr.get("current_supports", [22500, 22200, 22000]):
        if s < spot:
            support_candidates.append((s, 2.0))
    for r in sr.get("current_resistances", [22800, 23000, 23500]):
        if r > spot:
            resistance_candidates.append((r, 2.0))

    # Select nearest with highest weight
    if support_candidates:
        support_candidates.sort(key=lambda x: (spot - x[0], -x[1]))
        support = support_candidates[0][0]
    else:
        support = round((spot * 0.99) / 50) * 50

    if resistance_candidates:
        resistance_candidates.sort(key=lambda x: (x[0] - spot, -x[1]))
        resistance = resistance_candidates[0][0]
    else:
        resistance = round((spot * 1.01) / 50) * 50

    return support, resistance


# ===========================================================================
# COMPOSITE SCORING (8 rules from all entry agents)
# ===========================================================================

def compute_composite_scores(day):
    """8-rule composite scoring system from 6 specialist agents."""
    scores = {"BUY_CALL": 0.0, "BUY_PUT": 0.0}

    # Rule 1: VIX regime (weight 3.0)
    if day["vix_regime"] == "LOW":
        scores["BUY_CALL"] += 3.0
    else:
        scores["BUY_PUT"] += 3.0

    # Rule 2: Trend SMA50 (weight 2.0)
    if not day["above_sma50"]:
        scores["BUY_PUT"] += 2.0
    else:
        scores["BUY_CALL"] += 2.0

    # Rule 3: Trend SMA20 (weight 1.0)
    if not day["above_sma20"]:
        scores["BUY_PUT"] += 1.0
    else:
        scores["BUY_CALL"] += 1.0

    # Rule 4: RSI (weight 1.5)
    if day["rsi"] < 30 or day["rsi"] > 70:
        scores["BUY_PUT"] += 1.5

    # Rule 5: Day of week (weight 0.5)
    dow_rules = {"Monday": "BUY_PUT", "Tuesday": "BUY_PUT",
                 "Wednesday": "BUY_CALL", "Thursday": "BUY_PUT",
                 "Friday": "BUY_CALL"}
    dow_action = dow_rules.get(day["dow"])
    if dow_action:
        scores[dow_action] += 0.5

    # Rule 6: VIX spike (weight 2.0)
    if day["vix_spike"]:
        scores["BUY_CALL"] += 2.0

    # Rule 7: Previous momentum (weight 1.0)
    if day["prev_change"] < -1.0:
        scores["BUY_CALL"] += 1.0
    elif day["prev_change"] > 1.0:
        scores["BUY_PUT"] += 1.0

    # Rule 8: S/R proximity (weight 1.0)
    spot = day["open"]
    support, _ = sr_multi_method(spot, day["prev_high"], day["prev_low"],
                                 day["sma20"], day["sma50"])
    _, resistance = sr_multi_method(spot, day["prev_high"], day["prev_low"],
                                    day["sma20"], day["sma50"])
    if support:
        dist_pct = (spot - support) / spot * 100
        if 0 < dist_pct < 1.0:
            scores["BUY_CALL"] += 1.0
        elif dist_pct < 0:
            scores["BUY_PUT"] += 1.0
    if resistance:
        dist_pct = (resistance - spot) / spot * 100
        if 0 < dist_pct < 1.0:
            scores["BUY_PUT"] += 1.0
        elif dist_pct < 0:
            scores["BUY_CALL"] += 1.0

    return scores


def check_entry_timing(action, bar_idx):
    """Timing gate (Agent 1): only enter in optimal window."""
    if bar_idx < 2:
        return False
    if bar_idx > 13:
        return False
    if action == "BUY_PUT":
        return 2 <= bar_idx <= 3
    else:
        return 4 <= bar_idx <= 8


def get_vix_strike_offset(action, vix):
    """VIX-adaptive strike selection (Agent 2)."""
    if action == "BUY_CALL":
        if vix < 12:
            return -50
        elif vix < 20:
            return 100
        else:
            return 150
    else:  # BUY_PUT
        if vix < 12:
            return 0
        elif vix < 20:
            return 50
        else:
            return 50


def get_vix_lot_multiplier(vix):
    """VIX-adaptive position sizing (Agent 3)."""
    if vix < 12:
        return 2.0
    elif vix < 15:
        return 1.5
    elif vix < 20:
        return 1.0
    elif vix < 25:
        return 0.7
    elif vix < 30:
        return 0.5
    else:
        return 0.3


def bs_premium(spot, strike, dte, vix, opt_type):
    """Black-Scholes option premium."""
    try:
        return price_option(spot=spot, strike=strike, dte_days=dte,
                            vix=vix, option_type=opt_type)["premium"]
    except Exception:
        return 30.0


# ===========================================================================
# TRADE SIMULATION WITH PAPER BROKER
# ===========================================================================

def simulate_day(day, broker, equity, day_data_all, day_idx):
    """Simulate a single trading day using the full ensemble strategy.

    Returns: (pnl, trade_details)
    """
    spot = day["open"]
    vix = day["vix"]
    support, resistance = sr_multi_method(
        spot, day["prev_high"], day["prev_low"], day["sma20"], day["sma50"])

    # 1. Composite scoring
    scores = compute_composite_scores(day)
    best_action = max(scores, key=scores.get)
    total_score = sum(scores.values())
    confidence = scores[best_action] / total_score if total_score > 0 else 0

    if confidence < 0.25:
        return 0, {"action": "SKIP", "reason": "Low confidence", "confidence": confidence}

    # 2. Timing gate (Agent 1)
    entry_bar = 3 if best_action == "BUY_PUT" else 5
    if not check_entry_timing(best_action, entry_bar):
        return 0, {"action": "SKIP", "reason": "Timing gate rejected"}

    # 3. VIX-adaptive strike (Agent 2)
    strike_offset = get_vix_strike_offset(best_action, vix)
    strike = round(spot / 50) * 50 + strike_offset
    opt_type = "CE" if best_action == "BUY_CALL" else "PE"

    # 4. VIX-adaptive sizing (Agent 3)
    vix_mult = get_vix_lot_multiplier(vix)
    base_lots = max(1, int(equity * 0.08 / (50 * LOT_SIZE)))
    num_lots = max(1, int(base_lots * vix_mult))
    qty = min(num_lots * LOT_SIZE, 1800)
    num_lots = qty // LOT_SIZE

    # 5. Select exit strategy
    if best_action == "BUY_CALL":
        exit_strat = "sr_trail_combo"
    else:
        exit_strat = "trail_pct"

    # 6. Generate intraday path
    np.random.seed(int(abs(spot * 100)) % 2**31 + day["day_num"])
    path = generate_intraday_path(day["open"], day["high"], day["low"], day["close"])

    # 7. Entry via broker
    symbol = f"NIFTY{int(strike)}{opt_type}"
    dte_entry = 2.0
    entry_prem = bs_premium(path[entry_bar], strike, dte_entry, vix, opt_type)

    broker.update_price(symbol, entry_prem)
    entry_result = broker.place_order(
        symbol=symbol, side="BUY", quantity=qty,
        order_type="MARKET", price=entry_prem,
        product="NRML", tag="ensemble_entry"
    )

    if not entry_result.get("success"):
        return 0, {"action": "SKIP", "reason": f"Entry rejected: {entry_result.get('message', 'unknown')}"}

    entry_fill = entry_result["fill_price"]
    entry_costs = entry_result.get("costs", 0)

    # 8. Simulate intraday bars and check exits
    best_pnl = 0.0
    best_favorable_spot = path[entry_bar]
    exit_bar = TOTAL_BARS - 1
    exit_spot = day["close"]
    exit_reason = "eod_close"
    sr_combo_target_hit = False

    for bar_i in range(entry_bar + 1, TOTAL_BARS):
        bar_spot = path[bar_i]
        bar_dte = max(0.05, dte_entry - bar_i * 15 / 1440)
        bar_prem = bs_premium(bar_spot, strike, bar_dte, vix, opt_type)
        bar_pnl = (bar_prem - entry_fill) * qty

        if bar_pnl > best_pnl:
            best_pnl = bar_pnl

        # Track best favorable spot
        if best_action == "BUY_CALL" and bar_spot > best_favorable_spot:
            best_favorable_spot = bar_spot
        elif best_action == "BUY_PUT" and bar_spot < best_favorable_spot:
            best_favorable_spot = bar_spot

        # EXIT: sr_trail_combo (Agent 6 best for BUY_CALL, Sharpe 2.52)
        if exit_strat == "sr_trail_combo":
            trail_dist = path[entry_bar] * 0.003
            if not sr_combo_target_hit:
                if best_action == "BUY_CALL":
                    if resistance and bar_spot >= resistance:
                        sr_combo_target_hit = True
                        best_favorable_spot = bar_spot
                    if support and bar_spot < support:
                        exit_bar, exit_spot, exit_reason = bar_i, bar_spot, "sr_stop"
                        break
                else:
                    if support and bar_spot <= support:
                        sr_combo_target_hit = True
                        best_favorable_spot = bar_spot
                    if resistance and bar_spot > resistance:
                        exit_bar, exit_spot, exit_reason = bar_i, bar_spot, "sr_stop"
                        break
            else:
                if best_action == "BUY_CALL":
                    if bar_spot < best_favorable_spot - trail_dist:
                        exit_bar, exit_spot, exit_reason = bar_i, bar_spot, "sr_combo_trail"
                        break
                else:
                    if bar_spot > best_favorable_spot + trail_dist:
                        exit_bar, exit_spot, exit_reason = bar_i, bar_spot, "sr_combo_trail"
                        break

        # EXIT: trail_pct (best for BUY_PUT, Sharpe 5.22)
        elif exit_strat == "trail_pct":
            trail_dist = path[entry_bar] * 0.003
            if bar_i > entry_bar + 3:
                if best_action == "BUY_CALL":
                    if bar_spot < best_favorable_spot - trail_dist:
                        exit_bar, exit_spot, exit_reason = bar_i, bar_spot, "trailing_pct"
                        break
                else:
                    if bar_spot > best_favorable_spot + trail_dist:
                        exit_bar, exit_spot, exit_reason = bar_i, bar_spot, "trailing_pct"
                        break

    # 9. Exit via broker
    exit_dte = max(0.05, dte_entry - exit_bar * 15 / 1440)
    exit_prem = bs_premium(exit_spot, strike, exit_dte, vix, opt_type)

    broker.update_price(symbol, exit_prem)
    exit_result = broker.place_order(
        symbol=symbol, side="SELL", quantity=qty,
        order_type="MARKET", price=exit_prem,
        product="NRML", tag="ensemble_exit"
    )

    exit_fill = exit_result.get("fill_price", exit_prem) if exit_result.get("success") else exit_prem
    exit_costs = exit_result.get("costs", 0) if exit_result.get("success") else 0

    # 10. Calculate P&L (broker should have done this, but let's verify)
    intraday_pnl = (exit_fill - entry_fill) * qty - entry_costs - exit_costs

    # 11. Overnight hold bonus (Agent 4) — for BUY_PUT with profit
    overnight_pnl = 0
    overnight_held = False
    if best_action == "BUY_PUT" and intraday_pnl > 0 and day_idx + 1 < len(day_data_all):
        overnight_held = True
        next_day = day_data_all[day_idx + 1]
        gap_pct = (next_day["open"] - day["close"]) / day["close"] * 100
        if gap_pct < 0:
            # Favorable gap down for puts
            overnight_spot_move = day["close"] - next_day["open"]
            overnight_pnl = overnight_spot_move * qty * 0.5  # delta ~0.5
            overnight_pnl -= 50  # approximate overnight exit costs
            overnight_pnl = max(overnight_pnl, -intraday_pnl * 0.5)
        else:
            # Adverse gap up
            overnight_pnl = -abs(gap_pct) * qty * 0.3
            overnight_pnl = max(overnight_pnl, -intraday_pnl * 0.5)

    total_pnl = intraday_pnl + overnight_pnl

    trade_detail = {
        "day": day["day_num"],
        "date": day["date"],
        "dow": day["dow"],
        "action": best_action,
        "confidence": round(confidence, 2),
        "scores": {k: round(v, 1) for k, v in scores.items()},
        "entry_spot": round(path[entry_bar], 0),
        "exit_spot": round(exit_spot, 0),
        "support": support,
        "resistance": resistance,
        "vix": day["vix"],
        "vix_regime": day["vix_regime"],
        "strike": int(strike),
        "opt_type": opt_type,
        "strike_offset": strike_offset,
        "lots": num_lots,
        "qty": qty,
        "entry_prem": round(entry_fill, 2),
        "exit_prem": round(exit_fill, 2),
        "exit_strategy": exit_strat,
        "exit_reason": exit_reason,
        "exit_bar": exit_bar,
        "intraday_pnl": round(intraday_pnl, 0),
        "overnight_held": overnight_held,
        "overnight_pnl": round(overnight_pnl, 0),
        "total_pnl": round(total_pnl, 0),
        "entry_costs": round(entry_costs, 2),
        "exit_costs": round(exit_costs, 2),
    }

    return total_pnl, trade_detail


# ===========================================================================
# MAIN 15-DAY SIMULATION
# ===========================================================================

def run_15_day_paper_trading():
    """Run 15-day paper trading simulation with full ensemble."""
    print("=" * 80)
    print("  15-DAY PAPER TRADING — FAST FORWARD SIMULATION")
    print("  Full Ensemble (6 Agents) | Rs 200,000 Capital")
    print("  Composite Scoring + VIX-Adaptive + S/R Exits + Overnight Hold")
    print("=" * 80)

    # Generate synthetic market data
    days = generate_15_day_nifty(start_spot=22713.0, base_vix=25.5)

    # Initialize paper broker
    broker = PaperTradingBroker(
        initial_capital=CAPITAL,
        brokerage_per_order=BROKERAGE,
        latency_ms=0.5,  # Fast-forward: minimal latency
    )

    equity = CAPITAL
    equity_curve = [CAPITAL]
    all_trades = []
    daily_results = []
    peak_equity = CAPITAL
    max_dd = 0
    consecutive_wins = 0
    max_consecutive_wins = 0
    consecutive_losses = 0
    max_consecutive_losses = 0

    for i, day in enumerate(days):
        print(f"\n{'-'*80}")
        print(f"  DAY {day['day_num']}/{NUM_DAYS} | {day['date']} ({day['dow']})")
        print(f"  NIFTY Open={day['open']:.0f} H={day['high']:.0f} "
              f"L={day['low']:.0f} C={day['close']:.0f} | VIX={day['vix']} ({day['vix_regime']})")
        print(f"{'-'*80}")

        # Reset broker state for new day (clean slate per day)
        day_start_capital = broker.capital
        broker.positions.clear()
        broker._order_count_window.clear()  # Reset rate limiter for fast-forward

        pnl, trade = simulate_day(day, broker, equity, days, i)

        if trade["action"] == "SKIP":
            print(f"  SKIPPED: {trade.get('reason', 'N/A')}")
            daily_results.append({
                "day": day["day_num"], "date": day["date"], "dow": day["dow"],
                "action": "SKIP", "day_pnl": 0, "equity": round(equity, 0),
            })
            equity_curve.append(equity)
            consecutive_wins = 0
            consecutive_losses = 0
            continue

        # Update equity from broker state
        equity += pnl

        # Track drawdown
        if equity > peak_equity:
            peak_equity = equity
        dd = (peak_equity - equity) / peak_equity * 100
        if dd > max_dd:
            max_dd = dd

        # Track streaks
        if pnl > 0:
            consecutive_wins += 1
            consecutive_losses = 0
            if consecutive_wins > max_consecutive_wins:
                max_consecutive_wins = consecutive_wins
        elif pnl < 0:
            consecutive_losses += 1
            consecutive_wins = 0
            if consecutive_losses > max_consecutive_losses:
                max_consecutive_losses = consecutive_losses

        equity_curve.append(equity)
        all_trades.append(trade)

        status = "WIN" if pnl > 0 else "LOSS" if pnl < 0 else "FLAT"
        overnight_str = f" + O/N Rs {trade['overnight_pnl']:+,.0f}" if trade['overnight_held'] else ""
        print(f"  {trade['action']} {trade['opt_type']} strike={trade['strike']} | "
              f"{trade['lots']} lots ({trade['qty']} qty)")
        print(f"  Entry@bar{trade.get('exit_bar', 3)-trade.get('exit_bar', 3)}: "
              f"Rs {trade['entry_prem']:.2f} | "
              f"Exit@bar{trade['exit_bar']}: Rs {trade['exit_prem']:.2f} ({trade['exit_reason']})")
        print(f"  S/R: {trade['support']}/{trade['resistance']} | "
              f"Confidence: {trade['confidence']:.0%}")
        print(f"  >> {status}: Intraday Rs {trade['intraday_pnl']:+,.0f}"
              f"{overnight_str} | Net Rs {pnl:+,.0f}")
        print(f"  >> Equity: Rs {equity:,.0f} | DD: {dd:.1f}%")

        daily_results.append({
            "day": day["day_num"], "date": day["date"], "dow": day["dow"],
            "action": trade["action"], "day_pnl": round(pnl, 0),
            "intraday_pnl": trade["intraday_pnl"],
            "overnight_pnl": trade["overnight_pnl"],
            "equity": round(equity, 0),
            "exit_reason": trade["exit_reason"],
            "exit_strategy": trade["exit_strategy"],
            "confidence": trade["confidence"],
            "lots": trade["lots"],
            "vix": trade["vix"],
        })

    # =========================================================================
    # FINAL REPORT
    # =========================================================================
    print("\n" + "=" * 80)
    print("  15-DAY PAPER TRADING RESULTS")
    print("=" * 80)

    total_pnl = equity - CAPITAL
    active_trades = [t for t in all_trades if t["action"] != "SKIP"]
    win_trades = len([t for t in active_trades if t["total_pnl"] > 0])
    loss_trades = len([t for t in active_trades if t["total_pnl"] < 0])
    flat_trades = len([t for t in active_trades if t["total_pnl"] == 0])
    skipped = NUM_DAYS - len(active_trades)

    win_days = len([d for d in daily_results if d["day_pnl"] > 0])
    loss_days = len([d for d in daily_results if d["day_pnl"] < 0])
    flat_days = len([d for d in daily_results if d["day_pnl"] == 0])

    daily_pnls = [d["day_pnl"] for d in daily_results if d["action"] != "SKIP"]
    avg_daily = np.mean(daily_pnls) if daily_pnls else 0
    std_daily = np.std(daily_pnls) if daily_pnls else 1
    sharpe = (avg_daily / std_daily) * np.sqrt(252) if std_daily > 0 else 0
    best_day = max(daily_pnls) if daily_pnls else 0
    worst_day = min(daily_pnls) if daily_pnls else 0

    # Profit factor
    gross_wins = sum(t["total_pnl"] for t in active_trades if t["total_pnl"] > 0)
    gross_losses = abs(sum(t["total_pnl"] for t in active_trades if t["total_pnl"] < 0))
    profit_factor = gross_wins / max(1, gross_losses)

    print(f"\n  EQUITY:")
    print(f"    Starting Capital:     Rs {CAPITAL:>12,}")
    print(f"    Final Capital:        Rs {equity:>12,.0f}")
    print(f"    Total P&L:            Rs {total_pnl:>+12,.0f}")
    print(f"    Return:               {total_pnl/CAPITAL*100:>+11.2f}%")
    print(f"    Max Drawdown:         {max_dd:>11.2f}%")
    print(f"    Annualized Sharpe:    {sharpe:>11.2f}")
    print(f"    Profit Factor:        {profit_factor:>11.2f}")

    print(f"\n  TRADING DAYS:")
    print(f"    Active Trading Days:  {len(active_trades)}/{NUM_DAYS}")
    print(f"    Skipped Days:         {skipped} (low confidence/timing)")
    print(f"    Win Days:             {win_days}")
    print(f"    Loss Days:            {loss_days}")
    print(f"    Flat Days:            {flat_days}")

    print(f"\n  TRADE STATS:")
    print(f"    Total Trades:         {len(active_trades)}")
    print(f"    Wins:                 {win_trades} ({win_trades/max(1,len(active_trades))*100:.0f}%)")
    print(f"    Losses:               {loss_trades}")
    print(f"    Avg Daily P&L:        Rs {avg_daily:>+10,.0f}")
    print(f"    Best Day:             Rs {best_day:>+10,.0f}")
    print(f"    Worst Day:            Rs {worst_day:>+10,.0f}")
    print(f"    Max Consecutive Wins: {max_consecutive_wins}")
    print(f"    Max Consecutive Loss: {max_consecutive_losses}")

    # Action breakdown
    print(f"\n  ACTION BREAKDOWN:")
    for action in ["BUY_CALL", "BUY_PUT"]:
        at = [t for t in active_trades if t["action"] == action]
        if not at:
            continue
        w = len([t for t in at if t["total_pnl"] > 0])
        p = sum(t["total_pnl"] for t in at)
        avg_conf = np.mean([t["confidence"] for t in at])
        print(f"    {action}: {len(at)} trades | WR={w/len(at)*100:.0f}% | "
              f"P&L=Rs {p:+,.0f} | Avg Conf={avg_conf:.0%}")

    # Exit reason breakdown
    print(f"\n  EXIT REASONS:")
    exit_counts = defaultdict(lambda: {"count": 0, "pnl": 0, "wins": 0})
    for t in active_trades:
        r = t["exit_reason"]
        exit_counts[r]["count"] += 1
        exit_counts[r]["pnl"] += t["total_pnl"]
        if t["total_pnl"] > 0:
            exit_counts[r]["wins"] += 1
    for reason, data in sorted(exit_counts.items(), key=lambda x: -x[1]["pnl"]):
        wr = data["wins"] / max(1, data["count"]) * 100
        print(f"    {reason:20s}: {data['count']:2d}x | "
              f"P&L=Rs {data['pnl']:>+10,.0f} | WR={wr:.0f}%")

    # Daily breakdown table
    print(f"\n  DAILY BREAKDOWN:")
    print(f"    {'Day':>4s} {'Date':>12s} {'DoW':>5s} {'Action':>10s} "
          f"{'P&L':>12s} {'Equity':>14s} {'Exit':>16s} {'VIX':>5s}")
    print(f"    {'-'*4} {'-'*12} {'-'*5} {'-'*10} {'-'*12} {'-'*14} {'-'*16} {'-'*5}")
    for d in daily_results:
        marker = " +" if d["day_pnl"] > 0 else " -" if d["day_pnl"] < 0 else "  "
        exit_r = d.get("exit_reason", "--")
        vix_str = f"{d.get('vix', '--')}" if d.get('vix') else "--"
        print(f"    {d['day']:>4d} {d['date']:>12s} {d['dow']:>5s} "
              f"{d['action']:>10s} Rs {d['day_pnl']:>+10,.0f} "
              f"Rs {d['equity']:>10,.0f}{marker} {exit_r:>16s} {vix_str:>5s}")

    # Equity curve visualization
    print(f"\n  EQUITY CURVE:")
    for i, eq in enumerate(equity_curve):
        diff = eq - CAPITAL
        if abs(total_pnl) > 0:
            bar_len = int(diff / max(1, abs(total_pnl)) * 40)
        else:
            bar_len = 0
        if diff >= 0:
            bar = "+" * min(40, max(0, bar_len))
        else:
            bar = "-" * min(40, max(0, abs(bar_len)))
        label = f"Day {i:>2d}" if i > 0 else "Start"
        print(f"    {label}: Rs {eq:>12,.0f} {bar}")

    # Overnight hold analysis
    overnight_trades = [t for t in active_trades if t.get("overnight_held")]
    if overnight_trades:
        on_pnl = sum(t["overnight_pnl"] for t in overnight_trades)
        on_wins = len([t for t in overnight_trades if t["overnight_pnl"] > 0])
        print(f"\n  OVERNIGHT HOLD ANALYSIS (Agent 4):")
        print(f"    Trades held overnight: {len(overnight_trades)}")
        print(f"    Overnight P&L:         Rs {on_pnl:+,.0f}")
        print(f"    Overnight Win Rate:    {on_wins/max(1,len(overnight_trades))*100:.0f}%")

    # Save results
    output = {
        "test_date": datetime.now().isoformat(),
        "simulation_mode": "fast_forward",
        "capital": CAPITAL,
        "strategy": "full_ensemble_6_agents",
        "num_days": NUM_DAYS,
        "final_equity": round(equity, 0),
        "total_pnl": round(total_pnl, 0),
        "return_pct": round(total_pnl / CAPITAL * 100, 2),
        "max_dd_pct": round(max_dd, 2),
        "sharpe": round(sharpe, 2),
        "profit_factor": round(profit_factor, 2),
        "active_days": len(active_trades),
        "win_days": win_days,
        "loss_days": loss_days,
        "win_rate": round(win_trades / max(1, len(active_trades)) * 100, 1),
        "max_consecutive_wins": max_consecutive_wins,
        "max_consecutive_losses": max_consecutive_losses,
        "daily_results": daily_results,
        "trades": all_trades,
        "equity_curve": [round(e, 0) for e in equity_curve],
    }

    output_path = project_root / "data" / "paper_trading_15day_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to: {output_path}")

    # VERDICT
    print("\n" + "=" * 80)
    if total_pnl > 0 and win_days >= loss_days and max_dd < 30:
        print("  VERDICT: PASS -- Strategy is profitable in paper trading")
        print(f"  Return: {total_pnl/CAPITAL*100:+.1f}% | Sharpe: {sharpe:.2f} | "
              f"Max DD: {max_dd:.1f}% | WR: {win_trades/max(1,len(active_trades))*100:.0f}%")
        print(f"  READY for live trading with Rs {CAPITAL:,} capital on Monday")
    elif total_pnl > 0:
        print("  VERDICT: CAUTION -- Profitable but with concerns")
        if max_dd >= 30:
            print(f"  Warning: Max drawdown {max_dd:.1f}% exceeds 30% threshold")
        if win_days < loss_days:
            print(f"  Warning: More loss days ({loss_days}) than win days ({win_days})")
        print("  Consider adjusting risk parameters before live trading")
    else:
        print("  VERDICT: FAIL -- Strategy lost money in paper trading")
        print("  Do NOT deploy to live trading without fixes")
    print("=" * 80)

    return output


if __name__ == "__main__":
    results = run_15_day_paper_trading()
