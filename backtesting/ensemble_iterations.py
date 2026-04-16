"""
Ensemble Model Iteration Framework -- Train/Test Split.

Downloads 8+ months of real NIFTY+VIX data, splits into:
  - TRAIN: first 5 months (learn rules, tune parameters)
  - TEST:  last 1 month (validate out-of-sample)

Runs 10 iterations of improvements, each building on the best prior.
Tracks all metrics per iteration for comparison.

Capital: Rs 200,000 | Strategy: Full Ensemble with iterative improvements
"""

import json
import math
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backtesting.option_pricer import price_option
from backtesting.paper_trading import PaperTradingBroker

LOT_SIZE = 65  # NIFTY lot size
BROKERAGE = 20.0
STRIKE_INTERVAL = 50
CAPITAL = 200_000
TOTAL_BARS = 25  # 15-min bars per session

# Load S/R data
data_dir = project_root / "data"
ensemble_rules = {}
if (data_dir / "ensemble_rules.json").exists():
    with open(data_dir / "ensemble_rules.json") as f:
        ensemble_rules = json.load(f)


# ===========================================================================
# DATA DOWNLOAD -- Extended period for train/test split
# ===========================================================================

def download_extended_data():
    """Download 8+ months of NIFTY + VIX data for train/test split.

    Start from June 2025 to get enough warmup for SMA50.
    Valid data starts ~Aug 2025, giving us Aug 2025 - Apr 2026 = 8 months.
    Split: Train (first ~5 months) | Test (last ~1 month).
    """
    import yfinance as yf

    print("Downloading extended NIFTY data (Jun 2025 – Apr 2026)...")
    nifty = yf.download("^NSEI", start="2025-06-01", end="2026-04-06",
                        interval="1d", progress=False)
    vix_data = yf.download("^INDIAVIX", start="2025-06-01", end="2026-04-06",
                           interval="1d", progress=False)

    if nifty.empty:
        raise ValueError("Failed to download NIFTY data.")

    # Handle multi-index columns
    if isinstance(nifty.columns, pd.MultiIndex):
        nifty.columns = nifty.columns.get_level_values(0)
    if isinstance(vix_data.columns, pd.MultiIndex):
        vix_data.columns = vix_data.columns.get_level_values(0)

    # Merge VIX
    nifty["VIX"] = vix_data["Close"] if not vix_data.empty else 14.0
    nifty["VIX"] = nifty["VIX"].ffill().bfill().fillna(14.0)
    nifty["PrevVIX"] = nifty["VIX"].shift(1).fillna(nifty["VIX"].iloc[0])

    # Daily change
    nifty["Change%"] = nifty["Close"].pct_change() * 100
    nifty["PrevChange%"] = nifty["Change%"].shift(1).fillna(0)

    # Day of week
    nifty["DOW"] = nifty.index.day_name()

    # Moving averages
    nifty["SMA50"] = nifty["Close"].rolling(50).mean()
    nifty["SMA20"] = nifty["Close"].rolling(20).mean()
    nifty["SMA10"] = nifty["Close"].rolling(10).mean()
    nifty["AboveSMA50"] = nifty["Close"] > nifty["SMA50"]
    nifty["AboveSMA20"] = nifty["Close"] > nifty["SMA20"]

    # EMA for faster signals
    nifty["EMA9"] = nifty["Close"].ewm(span=9).mean()
    nifty["EMA21"] = nifty["Close"].ewm(span=21).mean()

    # RSI(14) and RSI(2)
    delta = nifty["Close"].diff()
    gain14 = delta.clip(lower=0).rolling(14).mean()
    loss14 = (-delta.clip(upper=0)).rolling(14).mean()
    nifty["RSI"] = 100 - (100 / (1 + gain14 / loss14.replace(0, 0.001)))

    gain2 = delta.clip(lower=0).rolling(2).mean()
    loss2 = (-delta.clip(upper=0)).rolling(2).mean()
    nifty["RSI2"] = 100 - (100 / (1 + gain2 / loss2.replace(0, 0.001)))

    # Previous day high/low
    nifty["PrevHigh"] = nifty["High"].shift(1)
    nifty["PrevLow"] = nifty["Low"].shift(1)
    nifty["PrevClose"] = nifty["Close"].shift(1)

    # VIX spike
    nifty["VIXSpike"] = nifty["VIX"] > nifty["PrevVIX"] * 1.15

    # ATR(14) for volatility-adjusted stops
    tr = pd.DataFrame({
        'hl': nifty["High"] - nifty["Low"],
        'hc': abs(nifty["High"] - nifty["PrevClose"]),
        'lc': abs(nifty["Low"] - nifty["PrevClose"]),
    }).max(axis=1)
    nifty["ATR14"] = tr.rolling(14).mean()

    # ADX(14) for trend strength / regime detection
    plus_dm = nifty["High"].diff()
    minus_dm = -nifty["Low"].diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

    atr_smooth = tr.ewm(span=14, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(span=14, adjust=False).mean() / atr_smooth.replace(0, 0.001))
    minus_di = 100 * (minus_dm.ewm(span=14, adjust=False).mean() / atr_smooth.replace(0, 0.001))
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, 0.001)
    nifty["ADX"] = dx.ewm(span=14, adjust=False).mean()
    nifty["PlusDI"] = plus_di
    nifty["MinusDI"] = minus_di

    # Bollinger Bands
    bb_sma = nifty["Close"].rolling(20).mean()
    bb_std = nifty["Close"].rolling(20).std()
    nifty["BB_upper"] = bb_sma + 2 * bb_std
    nifty["BB_lower"] = bb_sma - 2 * bb_std
    nifty["BB_pct"] = (nifty["Close"] - nifty["BB_lower"]) / (nifty["BB_upper"] - nifty["BB_lower"]).replace(0, 1)

    # Volume ratio (current / 20-day avg)
    if "Volume" in nifty.columns and nifty["Volume"].sum() > 0:
        vol_sma = nifty["Volume"].rolling(20).mean()
        nifty["VolumeRatio"] = nifty["Volume"] / vol_sma.replace(0, 1)
    else:
        nifty["VolumeRatio"] = 1.0

    # Gap analysis (open vs prev close)
    nifty["GapPct"] = (nifty["Open"] - nifty["PrevClose"]) / nifty["PrevClose"].replace(0, 1) * 100

    # VIX percentile (rolling 60-day rank)
    nifty["VIX_Pctile"] = nifty["VIX"].rolling(60).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    )

    # Weekly SMA for multi-timeframe
    nifty["WeeklySMA"] = nifty["Close"].rolling(5).mean().rolling(4).mean()  # ~20-day weekly proxy

    # Momentum indicators
    nifty["Mom3d"] = nifty["Close"].pct_change(3) * 100
    nifty["Mom5d"] = nifty["Close"].pct_change(5) * 100

    # ATH / Drawdown
    nifty["ATH"] = nifty["High"].expanding().max()
    nifty["DrawdownFromATH"] = (nifty["ATH"] - nifty["Close"]) / nifty["ATH"] * 100

    # Tuesday expiry
    nifty["IsExpiry"] = nifty["DOW"] == "Tuesday"

    # DTE
    dte_values = []
    for idx in nifty.index:
        current_dow = idx.weekday()
        if current_dow <= 1:
            dte = 1 - current_dow
        else:
            dte = (8 - current_dow)
        dte_values.append(max(dte, 0.5))
    nifty["DTE"] = dte_values

    # Drop NaN (SMA50 warmup)
    valid_start = nifty["SMA50"].first_valid_index()
    if valid_start is not None:
        nifty = nifty.loc[valid_start:]

    # Fill remaining NaN
    nifty = nifty.ffill().bfill()

    print(f"Extended data: {len(nifty)} trading days | "
          f"NIFTY {nifty['Low'].min():.0f}-{nifty['High'].max():.0f} | "
          f"VIX {nifty['VIX'].min():.1f}-{nifty['VIX'].max():.1f}")
    print(f"  Range: {nifty.index[0].date()} to {nifty.index[-1].date()}")

    return nifty


# ===========================================================================
# INTRADAY PATH FROM DAILY OHLC
# ===========================================================================

def generate_intraday_path(open_price, high, low, close, n_bars=TOTAL_BARS, seed_extra=0):
    """Generate realistic intraday price path from daily OHLC."""
    path = [open_price]
    np.random.seed((int(abs(open_price * 100)) % 2**31 + seed_extra) % 2**31)
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
# S/R COMPUTATION
# ===========================================================================

def sr_multi_method(spot, prev_high, prev_low, sma20, sma50, close_history=None, idx=None):
    """Multi-method S/R from Agent 6."""
    support_candidates = []
    resistance_candidates = []

    # Round numbers
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

    # PDH/PDL
    if prev_low is not None and prev_high is not None:
        pdl = round(prev_low / 50) * 50
        pdh = round(prev_high / 50) * 50
        if pdl < spot:
            support_candidates.append((pdl, 2.5))
        if pdh > spot:
            resistance_candidates.append((pdh, 2.5))

    # SMA
    for sma_val in [sma20, sma50]:
        if sma_val is not None and not np.isnan(sma_val):
            rounded = round(sma_val / 50) * 50
            if rounded < spot:
                support_candidates.append((rounded, 1.5))
            elif rounded > spot:
                resistance_candidates.append((rounded, 1.5))

    # Swing points
    if close_history is not None and idx is not None and idx >= 5:
        lookback = min(20, idx)
        window = close_history[max(0, idx - lookback):idx + 1]
        if len(window) >= 5:
            for i in range(1, len(window) - 1):
                if window[i] > window[i-1] and window[i] > window[i+1]:
                    if window[i] > spot:
                        resistance_candidates.append((round(window[i] / 50) * 50, 2.0))
                if window[i] < window[i-1] and window[i] < window[i+1]:
                    if window[i] < spot:
                        support_candidates.append((round(window[i] / 50) * 50, 2.0))

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


def bs_premium(spot, strike, dte, vix, opt_type):
    """Black-Scholes option premium."""
    try:
        return price_option(spot=spot, strike=strike, dte_days=dte,
                            vix=vix, option_type=opt_type)["premium"]
    except Exception:
        return 30.0


# ===========================================================================
# SCORING FUNCTIONS -- One per iteration
# ===========================================================================

def score_baseline(row, spot, support, resistance):
    """BASELINE (Iteration 0): v1 proven 8-rule scoring.
    +370%, DD 28.47%, Sharpe 3.84 on full 74 days.
    """
    vix = float(row["VIX"])
    above_sma50 = bool(row["AboveSMA50"])
    above_sma20 = bool(row["AboveSMA20"])
    rsi = float(row["RSI"])
    dow = str(row["DOW"])
    prev_change = float(row["PrevChange%"])
    vix_spike = bool(row["VIXSpike"])

    scores = {"BUY_CALL": 0.0, "BUY_PUT": 0.0}

    # Rule 1: VIX (4-level, weight 3.0-4.0)
    if vix < 12:
        scores["BUY_CALL"] += 3.0
    elif vix < 17:
        scores["BUY_PUT"] += 3.0
    elif vix < 25:
        scores["BUY_PUT"] += 3.5
    else:
        scores["BUY_PUT"] += 4.0

    # Rule 2: SMA50 (2.0)
    if not above_sma50:
        scores["BUY_PUT"] += 2.0
    else:
        scores["BUY_CALL"] += 2.0

    # Rule 3: SMA20 (1.0)
    if not above_sma20:
        scores["BUY_PUT"] += 1.0
    else:
        scores["BUY_CALL"] += 1.0

    # Rule 4: RSI (1.5)
    if rsi < 30:
        scores["BUY_PUT"] += 1.5
    elif rsi > 70:
        scores["BUY_PUT"] += 1.5

    # Rule 5: DOW (0.5)
    dow_rules = {"Monday": "BUY_PUT", "Tuesday": "BUY_PUT",
                 "Wednesday": "BUY_CALL", "Thursday": "BUY_PUT",
                 "Friday": "BUY_CALL"}
    dow_action = dow_rules.get(dow)
    if dow_action:
        scores[dow_action] += 0.5

    # Rule 6: VIX spike (2.0)
    if vix_spike:
        scores["BUY_CALL"] += 2.0

    # Rule 7: Previous momentum (1.0)
    if prev_change < -1.0:
        scores["BUY_CALL"] += 1.0
    elif prev_change > 1.0:
        scores["BUY_PUT"] += 1.0

    # Rule 8: S/R proximity (1.0)
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


def score_iter1_adx_regime(row, spot, support, resistance):
    """ITERATION 1: + ADX regime filter.
    ADX < 18 = choppy → reduce position or skip.
    ADX > 25 = trending → boost confidence.
    """
    scores = score_baseline(row, spot, support, resistance)
    adx = float(row.get("ADX", 20))

    if adx < 18:
        # Choppy market -- reduce signal strength
        scores["BUY_CALL"] *= 0.6
        scores["BUY_PUT"] *= 0.6
    elif adx > 30:
        # Strong trend -- boost the dominant signal
        dominant = max(scores, key=scores.get)
        scores[dominant] *= 1.3

    return scores


def score_iter2_bb_position(row, spot, support, resistance):
    """ITERATION 2: Baseline + Bollinger Band position signal."""
    scores = score_baseline(row, spot, support, resistance)
    bb_pct = float(row.get("BB_pct", 0.5))

    if bb_pct < 0.1:
        scores["BUY_CALL"] += 1.5  # Oversold -- bounce
    elif bb_pct > 0.9:
        scores["BUY_PUT"] += 1.5   # Overbought -- reversal
    elif bb_pct < 0.3:
        scores["BUY_CALL"] += 0.5
    elif bb_pct > 0.7:
        scores["BUY_PUT"] += 0.5

    return scores


def score_iter3_multitf(row, spot, support, resistance):
    """ITERATION 3: Baseline + Multi-timeframe alignment filter."""
    scores = score_baseline(row, spot, support, resistance)
    close = float(row["Close"])
    weekly_sma = float(row.get("WeeklySMA", close))
    ema9 = float(row.get("EMA9", close))
    ema21 = float(row.get("EMA21", close))

    alignment = 0
    if close > weekly_sma:
        alignment += 1
    if ema9 > ema21:
        alignment += 1

    best = max(scores, key=scores.get)
    if best == "BUY_CALL" and alignment == 0:
        scores["BUY_CALL"] *= 0.5  # Against both timeframes
    elif best == "BUY_PUT" and alignment == 2:
        scores["BUY_PUT"] *= 0.5   # Against both timeframes

    return scores


def score_iter4_vix_pctile(row, spot, support, resistance):
    """ITERATION 4: Baseline + VIX percentile filter."""
    scores = score_baseline(row, spot, support, resistance)
    vix_pctile = float(row.get("VIX_Pctile", 50))

    if vix_pctile < 25:
        scores["BUY_CALL"] *= 1.2
        scores["BUY_PUT"] *= 1.2
    elif vix_pctile > 75:
        scores["BUY_CALL"] *= 0.8
        scores["BUY_PUT"] *= 0.8

    return scores


def score_iter5_gap(row, spot, support, resistance):
    """ITERATION 5: Baseline + Gap analysis signal."""
    scores = score_baseline(row, spot, support, resistance)
    gap = float(row.get("GapPct", 0))

    if gap > 0.5:
        scores["BUY_CALL"] += 1.0
    elif gap < -0.5:
        scores["BUY_PUT"] += 1.0
    elif gap > 0.3:
        scores["BUY_CALL"] += 0.3
    elif gap < -0.3:
        scores["BUY_PUT"] += 0.3

    return scores


def score_iter6_volume(row, spot, support, resistance):
    """ITERATION 6: Baseline + Volume ratio confirmation."""
    scores = score_baseline(row, spot, support, resistance)
    vol_ratio = float(row.get("VolumeRatio", 1.0))

    if vol_ratio > 1.5:
        dominant = max(scores, key=scores.get)
        scores[dominant] *= 1.2
    elif vol_ratio < 0.6:
        scores["BUY_CALL"] *= 0.7
        scores["BUY_PUT"] *= 0.7

    return scores


def score_iter7_rsi2(row, spot, support, resistance):
    """ITERATION 7: Baseline + RSI(2) extreme mean reversion only.
    Only triggers at true extremes (< 5 or > 95).
    """
    scores = score_baseline(row, spot, support, resistance)
    rsi2 = float(row.get("RSI2", 50))

    # Only extreme RSI(2) -- very rare, high-conviction
    if rsi2 < 5:
        scores["BUY_CALL"] += 1.5   # Extreme oversold
    elif rsi2 > 95:
        scores["BUY_PUT"] += 1.5    # Extreme overbought

    return scores


def score_iter8_ema_cross(row, spot, support, resistance):
    """ITERATION 8: Baseline + EMA 9/21 crossover trend signal."""
    scores = score_baseline(row, spot, support, resistance)
    ema9 = float(row.get("EMA9", spot))
    ema21 = float(row.get("EMA21", spot))

    if ema9 > ema21:
        diff_pct = (ema9 - ema21) / ema21 * 100
        if diff_pct > 0.5:
            scores["BUY_CALL"] += 1.5
        else:
            scores["BUY_CALL"] += 0.5
    else:
        diff_pct = (ema21 - ema9) / ema21 * 100
        if diff_pct > 0.5:
            scores["BUY_PUT"] += 1.5
        else:
            scores["BUY_PUT"] += 0.5

    return scores


def score_iter9_atr_regime(row, spot, support, resistance):
    """ITERATION 9: Baseline + ATR regime-based confidence scaling."""
    scores = score_baseline(row, spot, support, resistance)
    atr = float(row.get("ATR14", spot * 0.01))
    atr_pct = atr / spot * 100

    if atr_pct > 1.5:
        scores["BUY_CALL"] *= 0.7
        scores["BUY_PUT"] *= 0.7
    elif atr_pct < 0.5:
        scores["BUY_CALL"] *= 1.2
        scores["BUY_PUT"] *= 1.2

    return scores


def score_iter10_drawdown_gate(row, spot, support, resistance):
    """ITERATION 10: Baseline + Light drawdown gate.
    Milder than v2's heavy gate. Only suppress CALL when DD > 12% from ATH.
    """
    scores = score_baseline(row, spot, support, resistance)
    dd_from_ath = float(row.get("DrawdownFromATH", 0))

    if dd_from_ath > 12:
        scores["BUY_CALL"] *= 0.4
        scores["BUY_PUT"] += 1.5
    elif dd_from_ath > 8:
        scores["BUY_PUT"] += 0.5

    return scores


def score_iter11_best_combo(row, spot, support, resistance):
    """ITERATION 11: BEST COMBO - ADX + Multi-TF + Gap + Drawdown Gate.
    Combines only improvements that individually showed gains on training data.
    """
    scores = score_baseline(row, spot, support, resistance)
    adx = float(row.get("ADX", 20))
    close = float(row["Close"])
    weekly_sma = float(row.get("WeeklySMA", close))
    ema9 = float(row.get("EMA9", close))
    ema21 = float(row.get("EMA21", close))
    gap = float(row.get("GapPct", 0))
    dd_from_ath = float(row.get("DrawdownFromATH", 0))

    # ADX regime (from iter 1 - best individual)
    if adx < 18:
        scores["BUY_CALL"] *= 0.6
        scores["BUY_PUT"] *= 0.6
    elif adx > 30:
        dominant = max(scores, key=scores.get)
        scores[dominant] *= 1.3

    # Multi-TF alignment (from iter 3 - reduced DD)
    alignment = 0
    if close > weekly_sma:
        alignment += 1
    if ema9 > ema21:
        alignment += 1
    best = max(scores, key=scores.get)
    if best == "BUY_CALL" and alignment == 0:
        scores["BUY_CALL"] *= 0.5
    elif best == "BUY_PUT" and alignment == 2:
        scores["BUY_PUT"] *= 0.5

    # Gap signal (from iter 5)
    if gap > 0.5:
        scores["BUY_CALL"] += 1.0
    elif gap < -0.5:
        scores["BUY_PUT"] += 1.0

    # Light drawdown gate (from iter 10)
    if dd_from_ath > 12:
        scores["BUY_CALL"] *= 0.4
        scores["BUY_PUT"] += 1.5
    elif dd_from_ath > 8:
        scores["BUY_PUT"] += 0.5

    return scores


# Map iteration number to scoring function
SCORING_FUNCTIONS = {
    0: ("Baseline (v1 8-rule)", score_baseline),
    1: ("+ ADX Regime Filter", score_iter1_adx_regime),
    2: ("+ Bollinger Band Position", score_iter2_bb_position),
    3: ("+ Multi-TF Alignment", score_iter3_multitf),
    4: ("+ VIX Percentile Filter", score_iter4_vix_pctile),
    5: ("+ Gap Analysis Signal", score_iter5_gap),
    6: ("+ Volume Ratio", score_iter6_volume),
    7: ("+ RSI(2) Mean Reversion", score_iter7_rsi2),
    8: ("+ EMA 9/21 Crossover", score_iter8_ema_cross),
    9: ("+ ATR Regime Confidence", score_iter9_atr_regime),
    10: ("+ Light Drawdown Gate", score_iter10_drawdown_gate),
    11: ("BEST COMBO (ADX+MTF+Gap+DD)", score_iter11_best_combo),
}


# ===========================================================================
# ITERATION CONFIG -- exit / sizing overrides per iteration
# ===========================================================================

def get_iteration_config(iteration, row, equity):
    """Return exit strategy, lot sizing, and confidence threshold for this iteration.
    Each iteration is INDEPENDENT -- only applies its own adjustments.
    """
    vix = float(row["VIX"])
    is_expiry = bool(row.get("IsExpiry", False))

    # Default config (v1 proven)
    config = {
        "exit_strat": "sr_trail_combo",
        "trail_pct": 0.003,  # Fixed 0.3%
        "min_confidence": 0.25,
        "lot_cap": 5,
        "size_mult": 1.0,
    }

    # Iteration 1: ADX regime sizing
    if iteration == 1:
        adx = float(row.get("ADX", 20))
        if adx < 18:
            config["size_mult"] *= 0.5
        elif adx > 30:
            config["size_mult"] *= 1.2

    # Iteration 5: Gap-aware VIX sizing
    if iteration == 5:
        if vix < 12:
            config["size_mult"] *= 1.3
        elif vix > 25:
            config["size_mult"] *= 0.5

    # Iteration 7: Expiry day size reduction
    if iteration == 7:
        if is_expiry:
            config["size_mult"] *= 0.6

    # Iteration 9: ATR-adapted trail
    if iteration == 9:
        atr = float(row.get("ATR14", 200))
        atr_pct = atr / float(row["Close"]) * 100
        if atr_pct > 1.2:
            config["trail_pct"] = 0.005
        elif atr_pct < 0.5:
            config["trail_pct"] = 0.002

    return config


# ===========================================================================
# SINGLE DAY SIMULATION
# ===========================================================================

def simulate_day_iteration(row, row_idx, nifty_df, broker, equity,
                          close_prices, score_fn, iteration):
    """Simulate one trading day with the given scoring function and iteration config."""
    entry_spot = float(row["Open"])
    day_high = float(row["High"])
    day_low = float(row["Low"])
    day_close = float(row["Close"])
    vix = float(row["VIX"]) if pd.notna(row["VIX"]) else 14.0
    dow = str(row["DOW"])
    is_expiry = bool(row.get("IsExpiry", False))
    dte_market = float(row.get("DTE", 2.0))
    sma20 = float(row["SMA20"]) if pd.notna(row.get("SMA20")) else None
    sma50 = float(row["SMA50"]) if pd.notna(row.get("SMA50")) else None
    prev_high = float(row["PrevHigh"]) if pd.notna(row.get("PrevHigh")) else day_high
    prev_low = float(row["PrevLow"]) if pd.notna(row.get("PrevLow")) else day_low

    # S/R
    support, resistance = sr_multi_method(
        entry_spot, prev_high, prev_low, sma20, sma50,
        close_history=close_prices, idx=row_idx
    )

    # Scoring
    scores = score_fn(row, entry_spot, support, resistance)
    best_action = max(scores, key=scores.get)
    total_score = sum(scores.values())
    confidence = scores[best_action] / total_score if total_score > 0 else 0

    date_str = str(nifty_df.index[row_idx].date())

    # Get iteration config
    config = get_iteration_config(iteration, row, equity)

    if confidence < config["min_confidence"]:
        return 0, {"action": "SKIP", "date": date_str, "dow": dow,
                    "is_expiry": is_expiry, "vix": round(vix, 1)}

    # Entry timing
    entry_bar = 3 if best_action == "BUY_PUT" else 5

    # Strike selection
    if best_action == "BUY_CALL":
        strike_offset = -50 if vix < 12 else 100 if vix < 20 else 150
    else:
        strike_offset = 0 if vix < 12 else 50
    strike = round(entry_spot / 50) * 50 + strike_offset
    opt_type = "CE" if best_action == "BUY_CALL" else "PE"

    # Position sizing (CAPITAL-based, no compounding)
    sizing_capital = CAPITAL
    vix_mult = 2.0 if vix < 12 else 1.5 if vix < 15 else 1.0 if vix < 20 else 0.7 if vix < 25 else 0.5 if vix < 30 else 0.3
    base_lots = max(1, int(sizing_capital * 0.08 / (50 * LOT_SIZE)))
    num_lots = max(1, int(base_lots * vix_mult * config["size_mult"]))
    num_lots = min(num_lots, config["lot_cap"])
    qty = min(num_lots * LOT_SIZE, 1800)
    num_lots = qty // LOT_SIZE

    # Intraday path
    path = generate_intraday_path(entry_spot, day_high, day_low, day_close,
                                  seed_extra=row_idx)

    dte_entry = max(0.5, dte_market)

    # Entry
    symbol = f"NIFTY{int(strike)}{opt_type}"
    entry_prem = bs_premium(path[entry_bar], strike, dte_entry, vix, opt_type)
    broker.update_price(symbol, entry_prem)
    entry_result = broker.place_order(symbol=symbol, side="BUY", quantity=qty,
                                      order_type="MARKET", price=entry_prem,
                                      product="NRML", tag="ensemble_entry")

    if not entry_result.get("success"):
        return 0, {"action": "SKIP", "date": date_str, "dow": dow,
                    "is_expiry": is_expiry, "vix": round(vix, 1)}

    entry_fill = entry_result["fill_price"]
    entry_costs = entry_result.get("costs", 0)

    # Intraday bar loop
    best_favorable_spot = path[entry_bar]
    exit_bar = TOTAL_BARS - 1
    exit_spot = day_close
    exit_reason = "eod_close"
    sr_combo_target_hit = False
    trail_dist = path[entry_bar] * config["trail_pct"]

    for bar_i in range(entry_bar + 1, TOTAL_BARS):
        bar_spot = path[bar_i]
        bar_dte = max(0.05, dte_entry - bar_i * 15 / 1440)
        bar_prem = bs_premium(bar_spot, strike, bar_dte, vix, opt_type)

        # Track favorable spot
        if best_action == "BUY_CALL" and bar_spot > best_favorable_spot:
            best_favorable_spot = bar_spot
        elif best_action == "BUY_PUT" and bar_spot < best_favorable_spot:
            best_favorable_spot = bar_spot

        # Expiry day 3-tranche exit
        if is_expiry:
            current_pnl = (bar_prem - entry_fill) * qty
            if current_pnl <= 0 and bar_i >= 20:
                exit_bar, exit_spot, exit_reason = bar_i, bar_spot, "expiry_stop_loser"
                break
            elif bar_i >= 24:
                exit_bar, exit_spot, exit_reason = bar_i, bar_spot, "expiry_late_exit"
                break

        # sr_trail_combo exit
        if config["exit_strat"] == "sr_trail_combo":
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

    # Exit
    exit_dte = max(0.05, dte_entry - exit_bar * 15 / 1440)
    exit_prem = bs_premium(exit_spot, strike, exit_dte, vix, opt_type)
    broker.update_price(symbol, exit_prem)
    exit_result = broker.place_order(symbol=symbol, side="SELL", quantity=qty,
                                     order_type="MARKET", price=exit_prem,
                                     product="NRML", tag="ensemble_exit")
    exit_fill = exit_result.get("fill_price", exit_prem) if exit_result.get("success") else exit_prem
    exit_costs = exit_result.get("costs", 0) if exit_result.get("success") else 0

    intraday_pnl = (exit_fill - entry_fill) * qty - entry_costs - exit_costs

    # Overnight hold for PUT winners (not on expiry)
    overnight_pnl = 0
    overnight_held = False
    if (best_action == "BUY_PUT" and intraday_pnl > 0
            and not is_expiry and row_idx + 1 < len(nifty_df)):
        overnight_held = True
        next_row = nifty_df.iloc[row_idx + 1]
        next_open = float(next_row["Open"])
        gap_pct = (next_open - day_close) / day_close * 100
        if gap_pct < 0:
            overnight_pnl = (day_close - next_open) * qty * 0.5 - 50
            overnight_pnl = max(overnight_pnl, -intraday_pnl * 0.5)
        else:
            overnight_pnl = -abs(gap_pct) * qty * 0.3
            overnight_pnl = max(overnight_pnl, -intraday_pnl * 0.5)

    total_pnl = intraday_pnl + overnight_pnl

    return total_pnl, {
        "action": best_action, "date": date_str, "dow": dow,
        "confidence": round(confidence, 2),
        "opt_type": opt_type, "strike": int(strike),
        "lots": num_lots, "qty": qty,
        "entry_spot": round(path[entry_bar], 0),
        "exit_spot": round(exit_spot, 0),
        "exit_reason": exit_reason,
        "intraday_pnl": round(intraday_pnl, 0),
        "overnight_pnl": round(overnight_pnl, 0),
        "overnight_held": overnight_held,
        "total_pnl": round(total_pnl, 0),
        "vix": round(vix, 1),
        "is_expiry": is_expiry,
    }


# ===========================================================================
# RUN ONE ITERATION ON A DATA SLICE
# ===========================================================================

def run_iteration(nifty_slice, score_fn, iteration, label="", silent=False):
    """Run one iteration on a given data slice. Returns metrics dict."""
    close_prices = nifty_slice["Close"].values.tolist()
    broker = PaperTradingBroker(initial_capital=CAPITAL, brokerage_per_order=BROKERAGE, latency_ms=0.5)

    equity = CAPITAL
    peak_equity = CAPITAL
    max_dd = 0
    all_trades = []
    daily_pnls = []
    monthly_pnl = defaultdict(float)
    expiry_pnls = []
    non_expiry_pnls = []

    for i in range(len(nifty_slice)):
        row = nifty_slice.iloc[i]
        broker.positions.clear()
        broker._order_count_window.clear()

        pnl, trade = simulate_day_iteration(
            row, i, nifty_slice, broker, equity, close_prices, score_fn, iteration
        )

        if trade["action"] == "SKIP":
            continue

        equity += pnl
        if equity > peak_equity:
            peak_equity = equity
        dd = (peak_equity - equity) / peak_equity * 100
        if dd > max_dd:
            max_dd = dd

        all_trades.append(trade)
        daily_pnls.append(pnl)
        month_key = trade["date"][:7]
        monthly_pnl[month_key] += pnl

        if trade["is_expiry"]:
            expiry_pnls.append(pnl)
        else:
            non_expiry_pnls.append(pnl)

    # Compute metrics
    total_pnl = equity - CAPITAL
    n_trades = len(all_trades)
    wins = len([t for t in all_trades if t["total_pnl"] > 0])
    wr = wins / max(1, n_trades) * 100

    avg_pnl = np.mean(daily_pnls) if daily_pnls else 0
    std_pnl = np.std(daily_pnls) if daily_pnls else 1
    sharpe = (avg_pnl / std_pnl) * np.sqrt(252) if std_pnl > 0 else 0

    gross_wins = sum(t["total_pnl"] for t in all_trades if t["total_pnl"] > 0)
    gross_losses = abs(sum(t["total_pnl"] for t in all_trades if t["total_pnl"] < 0))
    pf = gross_wins / max(1, gross_losses)

    all_months_pos = all(v > 0 for v in monthly_pnl.values()) if monthly_pnl else False

    # Exit reason breakdown
    exit_reasons = defaultdict(lambda: {"count": 0, "pnl": 0, "wins": 0})
    for t in all_trades:
        er = t["exit_reason"]
        exit_reasons[er]["count"] += 1
        exit_reasons[er]["pnl"] += t["total_pnl"]
        if t["total_pnl"] > 0:
            exit_reasons[er]["wins"] += 1

    # Action breakdown
    call_trades = [t for t in all_trades if t["action"] == "BUY_CALL"]
    put_trades = [t for t in all_trades if t["action"] == "BUY_PUT"]
    call_pnl = sum(t["total_pnl"] for t in call_trades)
    put_pnl = sum(t["total_pnl"] for t in put_trades)
    call_wr = len([t for t in call_trades if t["total_pnl"] > 0]) / max(1, len(call_trades)) * 100
    put_wr = len([t for t in put_trades if t["total_pnl"] > 0]) / max(1, len(put_trades)) * 100

    best_day = max(daily_pnls) if daily_pnls else 0
    worst_day = min(daily_pnls) if daily_pnls else 0

    metrics = {
        "label": label,
        "iteration": iteration,
        "n_trades": n_trades,
        "final_equity": round(equity, 0),
        "total_pnl": round(total_pnl, 0),
        "return_pct": round(total_pnl / CAPITAL * 100, 2),
        "max_dd_pct": round(max_dd, 2),
        "sharpe": round(sharpe, 2),
        "profit_factor": round(pf, 2),
        "win_rate": round(wr, 1),
        "best_day": round(best_day, 0),
        "worst_day": round(worst_day, 0),
        "all_months_pos": all_months_pos,
        "call_trades": len(call_trades),
        "call_pnl": round(call_pnl, 0),
        "call_wr": round(call_wr, 1),
        "put_trades": len(put_trades),
        "put_pnl": round(put_pnl, 0),
        "put_wr": round(put_wr, 1),
        "exit_reasons": {k: dict(v) for k, v in exit_reasons.items()},
        "monthly_pnl": dict(monthly_pnl),
    }

    if not silent:
        print(f"  Iter {iteration:>2d} | {label:<30s} | "
              f"Return: {metrics['return_pct']:>+8.1f}% | "
              f"DD: {metrics['max_dd_pct']:>5.1f}% | "
              f"Sharpe: {metrics['sharpe']:>5.2f} | "
              f"PF: {metrics['profit_factor']:>4.2f} | "
              f"WR: {metrics['win_rate']:>4.1f}% | "
              f"Trades: {n_trades}")

    return metrics


# ===========================================================================
# MAIN -- 10 ITERATIONS WITH TRAIN/TEST SPLIT
# ===========================================================================

def main():
    print("=" * 100)
    print("  ENSEMBLE ITERATION FRAMEWORK -- 10 Improvements with Train/Test Split")
    print("  Capital: Rs 200,000 | Data: Yahoo Finance (NIFTY + VIX)")
    print("=" * 100)

    # Download extended data
    nifty = download_extended_data()

    # Split: Train = first ~5 months, Test = last ~1 month (March 2026 onwards)
    split_date = "2026-03-01"
    train_data = nifty[nifty.index < split_date]
    test_data = nifty[nifty.index >= split_date]

    print(f"\n  TRAIN: {train_data.index[0].date()} to {train_data.index[-1].date()} "
          f"({len(train_data)} days)")
    print(f"  TEST:  {test_data.index[0].date()} to {test_data.index[-1].date()} "
          f"({len(test_data)} days)")

    # ── PHASE 1: Run all 11 iterations on TRAIN data ──
    print(f"\n{'='*100}")
    print("  PHASE 1: TRAINING (Learning on 5 months)")
    print(f"{'='*100}")
    print(f"  {'Iter':>4s} | {'Description':<30s} | {'Return':>9s} | {'DD':>6s} | "
          f"{'Sharpe':>6s} | {'PF':>5s} | {'WR':>5s} | {'Trades':>6s}")
    print(f"  {'-'*4} | {'-'*30} | {'-'*9} | {'-'*6} | {'-'*6} | {'-'*5} | {'-'*5} | {'-'*6}")

    train_results = []
    for iter_num, (desc, score_fn) in SCORING_FUNCTIONS.items():
        metrics = run_iteration(train_data, score_fn, iter_num, label=desc)
        train_results.append(metrics)

    # Find best iteration on training data (maximize Sharpe, penalize DD > 30%)
    def rank_score(m):
        score = m["sharpe"] * m["profit_factor"]
        if m["max_dd_pct"] > 30:
            score *= 0.5  # Heavy penalty for high DD
        if not m["all_months_pos"]:
            score *= 0.8  # Mild penalty
        return score

    best_train = max(train_results, key=rank_score)
    best_iter = best_train["iteration"]

    print(f"\n  >>> BEST TRAINING ITERATION: {best_iter} ({best_train['label']})")
    print(f"     Return: {best_train['return_pct']:+.1f}% | DD: {best_train['max_dd_pct']:.1f}% | "
          f"Sharpe: {best_train['sharpe']:.2f} | PF: {best_train['profit_factor']:.2f}")

    # ── PHASE 2: Validate ALL iterations on TEST data ──
    print(f"\n{'='*100}")
    print("  PHASE 2: TESTING (Out-of-sample validation on 1 month)")
    print(f"{'='*100}")
    print(f"  {'Iter':>4s} | {'Description':<30s} | {'Return':>9s} | {'DD':>6s} | "
          f"{'Sharpe':>6s} | {'PF':>5s} | {'WR':>5s} | {'Trades':>6s}")
    print(f"  {'-'*4} | {'-'*30} | {'-'*9} | {'-'*6} | {'-'*6} | {'-'*5} | {'-'*5} | {'-'*6}")

    test_results = []
    for iter_num, (desc, score_fn) in SCORING_FUNCTIONS.items():
        metrics = run_iteration(test_data, score_fn, iter_num, label=desc)
        test_results.append(metrics)

    best_test = max(test_results, key=rank_score)
    best_test_iter = best_test["iteration"]

    print(f"\n  >>> BEST TEST ITERATION: {best_test_iter} ({best_test['label']})")
    print(f"     Return: {best_test['return_pct']:+.1f}% | DD: {best_test['max_dd_pct']:.1f}% | "
          f"Sharpe: {best_test['sharpe']:.2f} | PF: {best_test['profit_factor']:.2f}")

    # ── PHASE 3: Full comparison ──
    print(f"\n{'='*100}")
    print("  PHASE 3: FULL COMPARISON (Train vs Test)")
    print(f"{'='*100}")
    print(f"\n  {'Iter':>4s} | {'Description':<30s} | {'TRAIN Return':>12s} {'TRAIN DD':>9s} {'TRAIN Sharpe':>12s} | "
          f"{'TEST Return':>11s} {'TEST DD':>8s} {'TEST Sharpe':>11s}")
    print(f"  {'-'*4} | {'-'*30} | {'-'*12} {'-'*9} {'-'*12} | {'-'*11} {'-'*8} {'-'*11}")

    for tr, te in zip(train_results, test_results):
        flag = " *" if tr["iteration"] == best_iter else ""
        flag2 = " *" if te["iteration"] == best_test_iter else ""
        print(f"  {tr['iteration']:>4d} | {tr['label']:<30s} | "
              f"{tr['return_pct']:>+10.1f}% {tr['max_dd_pct']:>7.1f}% {tr['sharpe']:>10.2f}  | "
              f"{te['return_pct']:>+9.1f}% {te['max_dd_pct']:>6.1f}% {te['sharpe']:>9.2f}{flag}{flag2}")

    # ── PHASE 4: Detailed analysis of best iteration ──
    print(f"\n{'='*100}")
    print(f"  PHASE 4: DETAILED ANALYSIS -- Best Iteration {best_iter}")
    print(f"{'='*100}")

    # Find train and test results for best iteration
    best_train_detail = train_results[best_iter]
    best_test_detail = test_results[best_iter]

    print(f"\n  TRAIN Performance:")
    print(f"    Return: {best_train_detail['return_pct']:+.2f}%")
    print(f"    Max DD: {best_train_detail['max_dd_pct']:.2f}%")
    print(f"    Sharpe: {best_train_detail['sharpe']:.2f}")
    print(f"    Profit Factor: {best_train_detail['profit_factor']:.2f}")
    print(f"    Trades: {best_train_detail['n_trades']} (WR: {best_train_detail['win_rate']:.1f}%)")
    print(f"    CALL: {best_train_detail['call_trades']} trades, "
          f"Rs {best_train_detail['call_pnl']:+,.0f} ({best_train_detail['call_wr']:.0f}% WR)")
    print(f"    PUT:  {best_train_detail['put_trades']} trades, "
          f"Rs {best_train_detail['put_pnl']:+,.0f} ({best_train_detail['put_wr']:.0f}% WR)")
    print(f"    Monthly P&L: {dict(best_train_detail['monthly_pnl'])}")

    print(f"\n  TEST Performance (Out-of-Sample):")
    print(f"    Return: {best_test_detail['return_pct']:+.2f}%")
    print(f"    Max DD: {best_test_detail['max_dd_pct']:.2f}%")
    print(f"    Sharpe: {best_test_detail['sharpe']:.2f}")
    print(f"    Profit Factor: {best_test_detail['profit_factor']:.2f}")
    print(f"    Trades: {best_test_detail['n_trades']} (WR: {best_test_detail['win_rate']:.1f}%)")
    print(f"    CALL: {best_test_detail['call_trades']} trades, "
          f"Rs {best_test_detail['call_pnl']:+,.0f} ({best_test_detail['call_wr']:.0f}% WR)")
    print(f"    PUT:  {best_test_detail['put_trades']} trades, "
          f"Rs {best_test_detail['put_pnl']:+,.0f} ({best_test_detail['put_wr']:.0f}% WR)")
    print(f"    Monthly P&L: {dict(best_test_detail['monthly_pnl'])}")

    # Exit reason comparison
    print(f"\n  EXIT REASONS (Train -> Test):")
    all_reasons = set()
    for r in [best_train_detail, best_test_detail]:
        all_reasons.update(r["exit_reasons"].keys())
    for reason in sorted(all_reasons):
        tr_data = best_train_detail["exit_reasons"].get(reason, {"count": 0, "pnl": 0, "wins": 0})
        te_data = best_test_detail["exit_reasons"].get(reason, {"count": 0, "pnl": 0, "wins": 0})
        tr_wr = tr_data["wins"] / max(1, tr_data["count"]) * 100
        te_wr = te_data["wins"] / max(1, te_data["count"]) * 100
        print(f"    {reason:<20s} | Train: {tr_data['count']:>3d}x Rs {tr_data['pnl']:>+10,.0f} "
              f"({tr_wr:.0f}% WR) | Test: {te_data['count']:>3d}x Rs {te_data['pnl']:>+10,.0f} "
              f"({te_wr:.0f}% WR)")

    # ── Save results ──
    results = {
        "run_date": datetime.now().isoformat(),
        "train_period": f"{train_data.index[0].date()} to {train_data.index[-1].date()}",
        "test_period": f"{test_data.index[0].date()} to {test_data.index[-1].date()}",
        "train_days": len(train_data),
        "test_days": len(test_data),
        "best_train_iteration": best_iter,
        "best_test_iteration": best_test_iter,
        "iterations": [],
    }
    for tr, te in zip(train_results, test_results):
        results["iterations"].append({
            "iteration": tr["iteration"],
            "description": tr["label"],
            "train": {k: v for k, v in tr.items() if k not in ["exit_reasons", "monthly_pnl"]},
            "test": {k: v for k, v in te.items() if k not in ["exit_reasons", "monthly_pnl"]},
        })

    out_path = data_dir / "ensemble_iterations_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to: {out_path}")

    # Final verdict
    print(f"\n{'='*100}")
    overfit = best_iter != best_test_iter
    if overfit:
        print(f"  WARNING: OVERFITTING? Best train iter ({best_iter}) != Best test iter ({best_test_iter})")
        print(f"      Train-optimal may not generalize. Consider the test-optimal iteration.")
    else:
        print(f"  CONSISTENT: Same iteration ({best_iter}) is best on both train and test!")

    # Recommend iteration that works well on BOTH
    combined_scores = []
    for tr, te in zip(train_results, test_results):
        c_score = rank_score(tr) * 0.4 + rank_score(te) * 0.6  # Weight test more
        combined_scores.append((tr["iteration"], c_score, tr["label"]))
    combined_scores.sort(key=lambda x: -x[1])
    rec_iter = combined_scores[0][0]
    rec_label = combined_scores[0][2]

    rec_train = train_results[rec_iter]
    rec_test = test_results[rec_iter]
    print(f"\n  RECOMMENDED ITERATION: {rec_iter} ({rec_label})")
    print(f"     Train: {rec_train['return_pct']:+.1f}%, DD {rec_train['max_dd_pct']:.1f}%, "
          f"Sharpe {rec_train['sharpe']:.2f}")
    print(f"     Test:  {rec_test['return_pct']:+.1f}%, DD {rec_test['max_dd_pct']:.1f}%, "
          f"Sharpe {rec_test['sharpe']:.2f}")
    print(f"{'='*100}")

    return results


if __name__ == "__main__":
    main()
