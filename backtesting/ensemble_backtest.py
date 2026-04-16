"""
FULL ENSEMBLE BACKTEST — Tests ALL 6 agent strategies over 6 months.

Compares multiple strategy configurations side by side:
  A. BASELINE:      Old exit (vix_adaptive for calls, trail_pct for puts),
                     old S/R (swing points only), fixed sizing
  B. ENSEMBLE_V1:   5-agent ensemble (timing, strike, sizing, holding, global)
                     vix_adaptive exit, swing S/R
  C. ENSEMBLE_V2:   6-agent ensemble with Agent 6 S/R integration
                     sr_trail_combo exit for calls, multi-method S/R
  D. FULL_ENSEMBLE:  V2 + VIX-adaptive sizing + timing gates + overnight hold

Also runs sensitivity tests:
  - S/R method comparison (round numbers vs PDH/PDL vs swing vs SMA)
  - Exit strategy comparison per action (sr_trail_combo vs vix_adaptive vs trail_pct)
  - VIX-adaptive sizing impact
  - Overnight holding impact

Capital: Rs 200,000 | Period: Oct 2025 - Apr 2026 (122 days)
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

LOT_SIZE = INDEX_CONFIG["NIFTY"]["lot_size"]  # 65
BROKERAGE = 20.0
STRIKE_INTERVAL = 50
CAPITAL = 200_000
TOTAL_BARS = 25


# ═══════════════════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

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


def generate_intraday_path(open_price, high, low, close, n_bars=TOTAL_BARS):
    """Generate realistic intraday price path from daily OHLC."""
    path = [open_price]
    np.random.seed(int(abs(open_price * 100)) % 2**31)
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


# ═══════════════════════════════════════════════════════════════════════════
# S/R METHODS (Agent 6)
# ═══════════════════════════════════════════════════════════════════════════

def sr_swing_points(close_prices, idx, lookback=20):
    """S/R from swing highs/lows."""
    start = max(0, idx - lookback)
    window = close_prices[start:idx + 1]
    if len(window) < 5:
        return None, None

    highs, lows = [], []
    for i in range(1, len(window) - 1):
        if window[i] > window[i-1] and window[i] > window[i+1]:
            highs.append(window[i])
        if window[i] < window[i-1] and window[i] < window[i+1]:
            lows.append(window[i])

    current = window[-1]
    support = round(max([l for l in lows if l < current], default=current * 0.99) / 50) * 50
    resistance = round(min([h for h in highs if h > current], default=current * 1.01) / 50) * 50
    return support, resistance


def sr_round_numbers(spot):
    """S/R from round 500-level numbers (best method, 90.7% WR)."""
    nearest_500_below = int(spot // 500) * 500
    nearest_500_above = nearest_500_below + 500

    # Also check 100-levels for closer S/R
    nearest_100_below = int(spot // 100) * 100
    nearest_100_above = nearest_100_below + 100

    support = nearest_100_below if (spot - nearest_100_below) < (spot - nearest_500_below) else nearest_500_below
    resistance = nearest_100_above if (nearest_100_above - spot) < (nearest_500_above - spot) else nearest_500_above

    # Prefer 500-levels when close
    if abs(spot - nearest_500_below) < 200:
        support = nearest_500_below
    if abs(nearest_500_above - spot) < 200:
        resistance = nearest_500_above

    return support, resistance


def sr_pdhl(prev_high, prev_low):
    """S/R from previous day high/low (85.3% WR)."""
    return round(prev_low / 50) * 50, round(prev_high / 50) * 50


def sr_sma(sma20, sma50, spot):
    """S/R from SMA levels."""
    levels = [l for l in [sma20, sma50] if l is not None]
    support = max([l for l in levels if l < spot], default=None)
    resistance = min([l for l in levels if l > spot], default=None)
    return support, resistance


def sr_multi_method(spot, close_prices, idx, prev_high, prev_low, sma20, sma50):
    """Multi-method S/R combining all 5 methods (Agent 6 approach)."""
    support_candidates = []
    resistance_candidates = []

    # Method 1: Round numbers (weight 3.0)
    s, r = sr_round_numbers(spot)
    if s and s < spot:
        support_candidates.append((s, 3.0))
    if r and r > spot:
        resistance_candidates.append((r, 3.0))

    # Method 2: PDH/PDL (weight 2.5)
    s, r = sr_pdhl(prev_high, prev_low)
    if s and s < spot:
        support_candidates.append((s, 2.5))
    if r and r > spot:
        resistance_candidates.append((r, 2.5))

    # Method 3: Swing points (weight 2.0)
    s, r = sr_swing_points(close_prices, idx)
    if s and s < spot:
        support_candidates.append((s, 2.0))
    if r and r > spot:
        resistance_candidates.append((r, 2.0))

    # Method 4: SMA (weight 1.5)
    s, r = sr_sma(sma20, sma50, spot)
    if s and s < spot:
        support_candidates.append((s, 1.5))
    if r and r > spot:
        resistance_candidates.append((r, 1.5))

    # Pick nearest with highest weight
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


# ═══════════════════════════════════════════════════════════════════════════
# EXIT STRATEGIES
# ═══════════════════════════════════════════════════════════════════════════

def simulate_trade(action, entry_spot, day_high, day_low, day_close,
                   vix, support, resistance, exit_strategy,
                   qty, strike_offset=0, entry_bar=3):
    """Simulate a single trade with specified exit strategy."""
    dte_entry = 2.0
    path = generate_intraday_path(entry_spot, day_high, day_low, day_close)

    opt_type = "CE" if action == "BUY_CALL" else "PE"
    strike = round(entry_spot / 50) * 50 + strike_offset

    entry_prem = bs_premium(entry_spot, strike, dte_entry, vix, opt_type)

    best_pnl = 0.0
    best_favorable_spot = entry_spot
    exit_bar = TOTAL_BARS - 1
    exit_spot = day_close
    exit_reason = "eod_close"
    sr_combo_target_hit = False

    for bar_i in range(entry_bar + 1, TOTAL_BARS):
        bar_spot = path[bar_i]
        bar_dte = max(0.05, dte_entry - bar_i * 15 / 1440)
        bar_prem = bs_premium(bar_spot, strike, bar_dte, vix, opt_type)
        bar_pnl = (bar_prem - entry_prem) * qty

        if bar_pnl > best_pnl:
            best_pnl = bar_pnl

        # Track best favorable spot
        if action == "BUY_CALL" and bar_spot > best_favorable_spot:
            best_favorable_spot = bar_spot
        elif action == "BUY_PUT" and bar_spot < best_favorable_spot:
            best_favorable_spot = bar_spot

        # ── EXIT: sr_trail_combo (Agent 6 best for BUY_CALL) ──
        if exit_strategy == "sr_trail_combo":
            trail_dist = entry_spot * 0.003
            if not sr_combo_target_hit:
                # Phase 1: S/R target + stop
                if action == "BUY_CALL":
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
                # Phase 2: Trail after target hit
                if action == "BUY_CALL":
                    if bar_spot < best_favorable_spot - trail_dist:
                        exit_bar, exit_spot, exit_reason = bar_i, bar_spot, "sr_combo_trail"
                        break
                else:
                    if bar_spot > best_favorable_spot + trail_dist:
                        exit_bar, exit_spot, exit_reason = bar_i, bar_spot, "sr_combo_trail"
                        break

        # ── EXIT: trail_pct (best for BUY_PUT, Sharpe 5.22) ──
        elif exit_strategy == "trail_pct":
            trail_dist = entry_spot * 0.003
            if bar_i > entry_bar + 3:  # min bars before trail
                if action == "BUY_CALL":
                    if bar_spot < best_favorable_spot - trail_dist:
                        exit_bar, exit_spot, exit_reason = bar_i, bar_spot, "trailing_pct"
                        break
                else:
                    if bar_spot > best_favorable_spot + trail_dist:
                        exit_bar, exit_spot, exit_reason = bar_i, bar_spot, "trailing_pct"
                        break

        # ── EXIT: vix_adaptive ──
        elif exit_strategy == "vix_adaptive":
            if vix > 20:
                trail_mult = 0.25
                sr_buffer = entry_spot * 0.002
            elif vix > 14:
                trail_mult = 0.35
                sr_buffer = entry_spot * 0.003
            else:
                trail_mult = 0.45
                sr_buffer = entry_spot * 0.004

            if action == "BUY_CALL" and support and bar_spot < support - sr_buffer:
                exit_bar, exit_spot, exit_reason = bar_i, bar_spot, "sr_vix_stop"
                break
            if action == "BUY_PUT" and resistance and bar_spot > resistance + sr_buffer:
                exit_bar, exit_spot, exit_reason = bar_i, bar_spot, "sr_vix_stop"
                break
            if best_pnl > 300 and bar_pnl < best_pnl * trail_mult:
                exit_bar, exit_spot, exit_reason = bar_i, bar_spot, "vix_trail"
                break
            if vix > 20 and bar_pnl > CAPITAL * 0.08 * 0.5:
                exit_bar, exit_spot, exit_reason = bar_i, bar_spot, "vix_target"
                break

        # ── EXIT: sr_fixed (S/R stop, hold to close) ──
        elif exit_strategy == "sr_fixed":
            if action == "BUY_CALL" and support and bar_spot < support:
                exit_bar, exit_spot, exit_reason = bar_i, bar_spot, "support_breach"
                break
            if action == "BUY_PUT" and resistance and bar_spot > resistance:
                exit_bar, exit_spot, exit_reason = bar_i, bar_spot, "resistance_breach"
                break

        # ── EXIT: eod (just hold to close) ──
        elif exit_strategy == "eod":
            pass  # Hold until end of day

    # Compute final P&L
    exit_dte = max(0.05, dte_entry - exit_bar * 15 / 1440)
    exit_prem = bs_premium(exit_spot, strike, exit_dte, vix, opt_type)
    pnl = (exit_prem - entry_prem) * qty
    avg_prem = (entry_prem + exit_prem) / 2
    costs = calc_costs(1, avg_prem, qty)

    return round(pnl - costs, 2), exit_reason, exit_bar


# ═══════════════════════════════════════════════════════════════════════════
# COMPOSITE SCORING (from all entry rules)
# ═══════════════════════════════════════════════════════════════════════════

def compute_composite_scores(vix, vix_regime, above_sma50, above_sma20,
                             rsi, dow, prev_change, vix_spike,
                             spot, support, resistance):
    """8-rule composite scoring system (from 5 specialist agents)."""
    scores = {"BUY_CALL": 0.0, "BUY_PUT": 0.0}

    # Rule 1: VIX regime (weight 3.0)
    if vix_regime in ("LOW",):
        scores["BUY_CALL"] += 3.0
    elif vix_regime in ("NORMAL", "HIGH", "EXTREME"):
        scores["BUY_PUT"] += 3.0

    # Rule 2: Trend SMA50 (weight 2.0)
    if not above_sma50:
        scores["BUY_PUT"] += 2.0
    else:
        scores["BUY_CALL"] += 2.0

    # Rule 3: Trend SMA20 (weight 1.0)
    if not above_sma20:
        scores["BUY_PUT"] += 1.0
    else:
        scores["BUY_CALL"] += 1.0

    # Rule 4: RSI (weight 1.5)
    if rsi < 30:
        scores["BUY_PUT"] += 1.5
    elif rsi > 70:
        scores["BUY_PUT"] += 1.5

    # Rule 5: Day of week (weight 0.5)
    dow_rules = {"Monday": "BUY_PUT", "Tuesday": "BUY_PUT",
                 "Wednesday": "BUY_CALL", "Thursday": "BUY_PUT", "Friday": "BUY_CALL"}
    dow_action = dow_rules.get(dow)
    if dow_action:
        scores[dow_action] += 0.5

    # Rule 6: VIX spike (weight 2.0)
    if vix_spike:
        scores["BUY_CALL"] += 2.0

    # Rule 7: Previous momentum (weight 1.0)
    if prev_change < -1.0:
        scores["BUY_CALL"] += 1.0
    elif prev_change > 1.0:
        scores["BUY_PUT"] += 1.0

    # Rule 8: S/R proximity (weight 1.0)
    if support and spot:
        dist_pct = (spot - support) / spot * 100
        if 0 < dist_pct < 1.0:
            scores["BUY_CALL"] += 1.0
        elif dist_pct < 0:
            scores["BUY_PUT"] += 1.0
    if resistance and spot:
        dist_pct = (resistance - spot) / spot * 100
        if 0 < dist_pct < 1.0:
            scores["BUY_PUT"] += 1.0
        elif dist_pct < 0:
            scores["BUY_CALL"] += 1.0

    return scores


def get_vix_strike_offset(action, vix):
    """VIX-adaptive strike selection (Agent 2)."""
    if action == "BUY_CALL":
        if vix < 12:
            return -50  # ITM
        elif vix < 20:
            return 100  # OTM
        else:
            return 150  # deep OTM
    else:  # BUY_PUT
        if vix < 12:
            return 0  # ATM
        elif vix < 20:
            return 50  # ITM
        else:
            return 50  # ITM


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


def check_entry_timing(action, bar_idx):
    """Timing gate (Agent 1): only enter in optimal window."""
    if bar_idx < 2:  # Wait 30 min
        return False
    if bar_idx > 13:  # Never after 12:30 PM
        return False
    if action == "BUY_PUT":
        return 2 <= bar_idx <= 3  # First hour
    else:  # BUY_CALL
        return 4 <= bar_idx <= 8  # Late morning
    return True


# ═══════════════════════════════════════════════════════════════════════════
# STRATEGY CONFIGURATIONS
# ═══════════════════════════════════════════════════════════════════════════

STRATEGIES = {
    "A_BASELINE": {
        "description": "Old baseline: vix_adaptive exit, swing S/R only, fixed 1 lot",
        "sr_method": "swing",
        "exit_call": "vix_adaptive",
        "exit_put": "trail_pct",
        "vix_sizing": False,
        "vix_strike": False,
        "timing_gate": False,
        "entry_bar": 1,
    },
    "B_ENSEMBLE_V1": {
        "description": "5-agent: composite scoring + vix exits + swing S/R",
        "sr_method": "swing",
        "exit_call": "vix_adaptive",
        "exit_put": "trail_pct",
        "vix_sizing": True,
        "vix_strike": True,
        "timing_gate": True,
        "entry_bar": 3,
    },
    "C_ENSEMBLE_V2": {
        "description": "6-agent: sr_trail_combo exit + multi-method S/R",
        "sr_method": "multi",
        "exit_call": "sr_trail_combo",
        "exit_put": "trail_pct",
        "vix_sizing": True,
        "vix_strike": True,
        "timing_gate": True,
        "entry_bar": 3,
    },
    "D_FULL_ENSEMBLE": {
        "description": "Full 6-agent + overnight hold simulation",
        "sr_method": "multi",
        "exit_call": "sr_trail_combo",
        "exit_put": "trail_pct",
        "vix_sizing": True,
        "vix_strike": True,
        "timing_gate": True,
        "entry_bar": 3,
        "overnight_hold_put": True,
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# MAIN BACKTEST
# ═══════════════════════════════════════════════════════════════════════════

def download_data():
    """Download NIFTY + VIX data."""
    import yfinance as yf

    print("Downloading data...")
    nifty = yf.download("^NSEI", start="2025-10-01", end="2026-04-06", interval="1d", progress=False)
    vix_data = yf.download("^INDIAVIX", start="2025-10-01", end="2026-04-06", interval="1d", progress=False)

    if isinstance(nifty.columns, pd.MultiIndex):
        nifty.columns = nifty.columns.get_level_values(0)
    if isinstance(vix_data.columns, pd.MultiIndex):
        vix_data.columns = vix_data.columns.get_level_values(0)

    nifty["VIX"] = vix_data["Close"]
    nifty["VIX"] = nifty["VIX"].ffill().fillna(14.0)
    nifty["Change%"] = nifty["Close"].pct_change() * 100
    nifty["PrevChange%"] = nifty["Change%"].shift(1)
    nifty["PrevVIX"] = nifty["VIX"].shift(1)
    nifty["DOW"] = nifty.index.day_name()
    nifty["SMA50"] = nifty["Close"].rolling(50).mean()
    nifty["SMA20"] = nifty["Close"].rolling(20).mean()
    nifty["AboveSMA50"] = nifty["Close"] > nifty["SMA50"]
    nifty["AboveSMA20"] = nifty["Close"] > nifty["SMA20"]

    delta = nifty["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    nifty["RSI"] = 100 - (100 / (1 + gain / loss.replace(0, 0.001)))

    nifty["PrevHigh"] = nifty["High"].shift(1)
    nifty["PrevLow"] = nifty["Low"].shift(1)

    print(f"Data: {len(nifty)} trading days | NIFTY {nifty['Low'].min():.0f}-{nifty['High'].max():.0f}")
    return nifty


def run_strategy(nifty, config, strat_name):
    """Run a single strategy over 6 months."""
    close_prices = nifty["Close"].values.tolist()
    equity = CAPITAL
    equity_curve = [CAPITAL]
    trades = []
    overnight_bonus = []

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
        prev_high = float(row["PrevHigh"]) if pd.notna(row["PrevHigh"]) else day_high
        prev_low = float(row["PrevLow"]) if pd.notna(row["PrevLow"]) else day_low

        # Compute S/R based on configured method
        if config["sr_method"] == "swing":
            support, resistance = sr_swing_points(close_prices, i)
        elif config["sr_method"] == "round":
            support, resistance = sr_round_numbers(entry_spot)
        elif config["sr_method"] == "pdhl":
            support, resistance = sr_pdhl(prev_high, prev_low)
        elif config["sr_method"] == "sma":
            support, resistance = sr_sma(sma20, sma50, entry_spot)
        elif config["sr_method"] == "multi":
            support, resistance = sr_multi_method(
                entry_spot, close_prices, i, prev_high, prev_low, sma20, sma50)
        else:
            support, resistance = sr_swing_points(close_prices, i)

        # Composite scoring for entry
        scores = compute_composite_scores(
            vix, vix_regime, above_sma50, above_sma20,
            rsi, dow, prev_change, vix_spike,
            entry_spot, support, resistance)

        best_action = max(scores, key=scores.get)
        total_score = sum(scores.values())
        confidence = scores[best_action] / total_score if total_score > 0 else 0

        if confidence < 0.25:
            equity_curve.append(equity)
            continue

        # Timing gate (Agent 1)
        entry_bar = config.get("entry_bar", 1)
        if config.get("timing_gate"):
            if not check_entry_timing(best_action, entry_bar):
                equity_curve.append(equity)
                continue

        # VIX-adaptive strike (Agent 2)
        if config.get("vix_strike"):
            strike_offset = get_vix_strike_offset(best_action, vix)
        else:
            strike_offset = 0

        # VIX-adaptive sizing (Agent 3) — FIXED: use initial capital, not equity
        # Prevents unrealistic exponential compounding
        if config.get("vix_sizing"):
            vix_mult = get_vix_lot_multiplier(vix)
            base_lots = max(1, int(CAPITAL * 0.08 / (50 * LOT_SIZE)))
            num_lots = max(1, int(base_lots * vix_mult))
            num_lots = min(num_lots, 5)  # Cap at 5 lots realistic for Rs 2L
        else:
            num_lots = 1
        qty = num_lots * LOT_SIZE

        # Cap at freeze limit
        qty = min(qty, 1800)
        num_lots = qty // LOT_SIZE

        # Select exit strategy
        if best_action == "BUY_CALL":
            exit_strat = config["exit_call"]
        else:
            exit_strat = config["exit_put"]

        # Run trade simulation
        pnl, exit_reason, exit_bar_num = simulate_trade(
            best_action, entry_spot, day_high, day_low, day_close,
            vix, support, resistance, exit_strat,
            qty, strike_offset, entry_bar)

        # Overnight hold bonus for BUY_PUT (Agent 4)
        overnight_pnl = 0
        if config.get("overnight_hold_put") and best_action == "BUY_PUT" and pnl > 0:
            # Simulate overnight gap: avg 0.57% continuation, 54.2% of the time
            if i + 1 < len(nifty):
                next_open = float(nifty.iloc[i + 1]["Open"])
                next_close = float(nifty.iloc[i + 1]["Close"])
                gap_pct = (next_open - day_close) / day_close * 100
                # For puts, down gap is good
                if gap_pct < 0:
                    overnight_spot_move = day_close - next_open
                    overnight_pnl = overnight_spot_move * qty * 0.5  # delta ~0.5
                    overnight_pnl -= calc_costs(1, 30, qty)  # exit cost
                    overnight_pnl = max(overnight_pnl, -pnl * 0.5)  # cap loss at 50% of profit
                else:
                    overnight_pnl = -abs(gap_pct) * qty * 0.3  # adverse gap
                    overnight_pnl = max(overnight_pnl, -pnl * 0.5)

        total_pnl = pnl + overnight_pnl
        equity += total_pnl
        equity_curve.append(equity)

        trades.append({
            "date": str(nifty.index[i].date()),
            "action": best_action,
            "exit_strategy": exit_strat,
            "exit_reason": exit_reason,
            "confidence": round(confidence, 2),
            "entry": round(entry_spot, 0),
            "support": support,
            "resistance": resistance,
            "vix": round(vix, 1),
            "vix_regime": vix_regime,
            "lots": num_lots,
            "qty": qty,
            "day_pnl": round(pnl, 0),
            "overnight_pnl": round(overnight_pnl, 0),
            "net_pnl": round(total_pnl, 0),
        })

    return trades, equity_curve


def print_results(strat_name, config, trades, equity_curve):
    """Print strategy results."""
    wins = len([t for t in trades if t["net_pnl"] > 0])
    total = len(trades)
    total_pnl = sum(t["net_pnl"] for t in trades)
    final_equity = equity_curve[-1]

    # Max drawdown
    peak = equity_curve[0]
    max_dd = 0
    for v in equity_curve:
        if v > peak:
            peak = v
        dd = (peak - v) / peak * 100
        max_dd = max(max_dd, dd)

    # Sharpe
    daily_pnls = [t["net_pnl"] for t in trades]
    sharpe = 0
    if len(daily_pnls) > 1 and np.std(daily_pnls) > 0:
        sharpe = np.mean(daily_pnls) / np.std(daily_pnls) * np.sqrt(252)

    # Largest win/loss
    largest_win = max(daily_pnls) if daily_pnls else 0
    largest_loss = min(daily_pnls) if daily_pnls else 0

    wr = wins / max(1, total) * 100
    ret = total_pnl / CAPITAL * 100

    print(f"\n  {'='*75}")
    print(f"  {strat_name}: {config['description']}")
    print(f"  {'='*75}")
    print(f"  Trades: {total} | Win Rate: {wr:.1f}% ({wins}/{total})")
    print(f"  Total P&L: Rs {total_pnl:+,.0f} | Return: {ret:+.1f}%")
    print(f"  Final Equity: Rs {final_equity:,.0f}")
    print(f"  Max DD: {max_dd:.2f}% | Sharpe: {sharpe:.2f}")
    print(f"  Largest Win: Rs {largest_win:+,.0f} | Largest Loss: Rs {largest_loss:+,.0f}")

    # Per-action breakdown
    for action in ["BUY_CALL", "BUY_PUT"]:
        at = [t for t in trades if t["action"] == action]
        if not at:
            continue
        w = len([t for t in at if t["net_pnl"] > 0])
        p = sum(t["net_pnl"] for t in at)
        awr = w / len(at) * 100 if at else 0
        exit_strat = config.get(f"exit_{'call' if action=='BUY_CALL' else 'put'}", "?")
        print(f"    {action}: {len(at)} trades, WR={awr:.0f}%, "
              f"P&L=Rs {p:>+10,.0f} | exit={exit_strat}")

    # Monthly P&L
    monthly = defaultdict(float)
    for t in trades:
        monthly[t["date"][:7]] += t["net_pnl"]
    print(f"\n  Monthly P&L:")
    all_positive = True
    for month, pnl in sorted(monthly.items()):
        bar = "+" * max(1, min(40, int(abs(pnl) / 5000)))
        if pnl < 0:
            bar = "-" * max(1, min(40, int(abs(pnl) / 5000)))
            all_positive = False
        color = "" if pnl > 0 else "(!)"
        print(f"    {month}: Rs {pnl:>+10,.0f} {bar} {color}")
    print(f"  All months profitable: {'YES' if all_positive else 'NO'}")

    # Exit reason breakdown
    exit_counts = defaultdict(lambda: {"count": 0, "pnl": 0, "wins": 0})
    for t in trades:
        r = t["exit_reason"]
        exit_counts[r]["count"] += 1
        exit_counts[r]["pnl"] += t["net_pnl"]
        if t["net_pnl"] > 0:
            exit_counts[r]["wins"] += 1
    print(f"\n  Exit Reasons:")
    for reason, data in sorted(exit_counts.items(), key=lambda x: -x[1]["pnl"]):
        wr = data["wins"] / max(1, data["count"]) * 100
        print(f"    {reason:20s}: {data['count']:3d}x, "
              f"P&L=Rs {data['pnl']:>+10,.0f} WR={wr:.0f}%")

    return {
        "strategy": strat_name,
        "trades": total,
        "win_rate": round(wr, 1),
        "total_pnl": round(total_pnl, 0),
        "return_pct": round(ret, 1),
        "max_dd": round(max_dd, 2),
        "sharpe": round(sharpe, 2),
        "final_equity": round(final_equity, 0),
        "all_months_positive": all_positive,
    }


def run_sr_comparison(nifty):
    """Compare S/R methods head-to-head."""
    print("\n" + "=" * 80)
    print("  S/R METHOD COMPARISON (using same entry/exit, varying only S/R)")
    print("=" * 80)

    sr_methods = ["swing", "round", "pdhl", "multi"]
    base_config = {
        "exit_call": "sr_trail_combo",
        "exit_put": "trail_pct",
        "vix_sizing": True,
        "vix_strike": True,
        "timing_gate": True,
        "entry_bar": 3,
    }

    sr_results = {}
    for method in sr_methods:
        config = {**base_config, "sr_method": method, "description": f"S/R: {method}"}
        trades, eq = run_strategy(nifty, config, f"SR_{method}")
        total_pnl = sum(t["net_pnl"] for t in trades)
        wins = len([t for t in trades if t["net_pnl"] > 0])
        total = len(trades)
        wr = wins / max(1, total) * 100
        daily_pnls = [t["net_pnl"] for t in trades]
        sharpe = (np.mean(daily_pnls) / np.std(daily_pnls) * np.sqrt(252)) if len(daily_pnls) > 1 and np.std(daily_pnls) > 0 else 0

        sr_results[method] = {"pnl": total_pnl, "wr": wr, "sharpe": sharpe, "trades": total}
        print(f"  {method:10s}: P&L=Rs {total_pnl:>+10,.0f}, WR={wr:.1f}%, Sharpe={sharpe:.2f}, trades={total}")

    best = max(sr_results, key=lambda k: sr_results[k]["sharpe"])
    print(f"\n  >>> BEST S/R METHOD: {best} (Sharpe={sr_results[best]['sharpe']:.2f})")
    return sr_results


def run_exit_comparison(nifty):
    """Compare exit strategies for each action."""
    print("\n" + "=" * 80)
    print("  EXIT STRATEGY COMPARISON (per action)")
    print("=" * 80)

    exit_strats = ["sr_trail_combo", "vix_adaptive", "trail_pct", "sr_fixed", "eod"]
    base_config = {
        "sr_method": "multi",
        "vix_sizing": True,
        "vix_strike": True,
        "timing_gate": True,
        "entry_bar": 3,
    }

    for action_label, set_call, set_put in [
        ("BUY_CALL exits", True, False),
        ("BUY_PUT exits", False, True),
    ]:
        print(f"\n  {action_label}:")
        for strat in exit_strats:
            if set_call:
                config = {**base_config, "exit_call": strat, "exit_put": "trail_pct",
                          "description": f"CALL exit: {strat}"}
            else:
                config = {**base_config, "exit_call": "sr_trail_combo", "exit_put": strat,
                          "description": f"PUT exit: {strat}"}

            trades, eq = run_strategy(nifty, config, f"EXIT_{strat}")

            if set_call:
                action_trades = [t for t in trades if t["action"] == "BUY_CALL"]
            else:
                action_trades = [t for t in trades if t["action"] == "BUY_PUT"]

            if not action_trades:
                print(f"    {strat:18s}: no trades")
                continue

            pnl = sum(t["net_pnl"] for t in action_trades)
            wins = len([t for t in action_trades if t["net_pnl"] > 0])
            wr = wins / len(action_trades) * 100
            daily_pnls = [t["net_pnl"] for t in action_trades]
            sharpe = (np.mean(daily_pnls) / np.std(daily_pnls) * np.sqrt(252)) if len(daily_pnls) > 1 and np.std(daily_pnls) > 0 else 0
            print(f"    {strat:18s}: P&L=Rs {pnl:>+10,.0f}, WR={wr:.1f}%, "
                  f"Sharpe={sharpe:.2f}, trades={len(action_trades)}")


def main():
    print("=" * 80)
    print("  FULL ENSEMBLE BACKTEST — ALL 6 AGENTS x 6 MONTHS")
    print(f"  Capital: Rs {CAPITAL:,} | Lot: {LOT_SIZE} | Period: Oct 2025 - Apr 2026")
    print("=" * 80)

    nifty = download_data()

    # ── Run all strategies ────────────────────────────────────────
    all_summaries = []

    for strat_name, config in STRATEGIES.items():
        trades, equity_curve = run_strategy(nifty, config, strat_name)
        summary = print_results(strat_name, config, trades, equity_curve)
        all_summaries.append(summary)

    # ── Comparison table ──────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  STRATEGY COMPARISON TABLE")
    print("=" * 80)
    print(f"  {'Strategy':<20s} {'Trades':>6s} {'WR':>6s} {'Return':>8s} {'Sharpe':>7s} {'MaxDD':>7s} {'Final Equity':>14s}")
    print(f"  {'-'*20} {'-'*6} {'-'*6} {'-'*8} {'-'*7} {'-'*7} {'-'*14}")

    for s in all_summaries:
        print(f"  {s['strategy']:<20s} {s['trades']:>6d} {s['win_rate']:>5.1f}% "
              f"{s['return_pct']:>+7.1f}% {s['sharpe']:>7.2f} {s['max_dd']:>6.2f}% "
              f"Rs {s['final_equity']:>10,.0f}")

    # ── S/R method comparison ─────────────────────────────────────
    sr_results = run_sr_comparison(nifty)

    # ── Exit strategy comparison ──────────────────────────────────
    run_exit_comparison(nifty)

    # ── Save results ──────────────────────────────────────────────
    results = {
        "backtest_date": datetime.now().isoformat(),
        "period": "Oct 2025 - Apr 2026",
        "capital": CAPITAL,
        "strategies": all_summaries,
        "sr_comparison": {k: {"pnl": v["pnl"], "sharpe": v["sharpe"], "wr": v["wr"]}
                         for k, v in sr_results.items()},
    }

    output_path = project_root / "data" / "ensemble_backtest_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to: {output_path}")

    print("\n" + "=" * 80)
    print("  BACKTEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
