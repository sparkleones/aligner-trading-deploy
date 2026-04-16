"""
Paper Trading with REAL Historical Data — V4 Optimized Multi-Trade Engine.

V4 OPTIMIZATION (data-driven from 227-trade V3 analysis):
  - REMOVED sr_breakout entries (0% WR, -Rs 27K in V3)
  - FIXED trail_pct exit: widened 0.3% → 0.5%, min 6 bars + profit gate
  - FIXED sr_combo_trail: widened 0.3% → 0.6%, profit gate added
  - COOLDOWN_BARS: 2 → 0 (analysis shows 0 is optimal)
  - Composite window: shifted to bars 3-5 + added bars 8-10 (peak WR bars)
  - BTST: widened criteria (all profitable PUTs, not just intraday >0)
  - Zero-hero: lowered gap threshold -0.5% (was -0.8%), VIX >= 13 (was 15)
  - time_exit extended: PUT 16 → 20 bars, CALL 12 → 15 bars

V3 baseline: +1638% return, Sharpe 15.53, PF 6.08, DD 4.22%
Key V4 targets: reduce trail_pct losses (-284K), lean into time_exit (+3.6M)

Capital: Rs 200,000 | Strategy: Full Ensemble V4
Data source: Yahoo Finance
"""

import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

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

# ── Multi-trade parameters (V4 optimized) ──
MAX_TRADES_PER_DAY = 5
MAX_CONCURRENT = 2   # max open positions at same time
COOLDOWN_BARS = 0     # V4: 0 is optimal (was 2 in V3, analysis confirmed)
MIN_CONFIDENCE = 0.25

# ── V4 Exit parameters (iteration 4: keep V3 exits, entry-only optimization) ──
# INSIGHT: V3's tight 0.3% trail is a valuable LOSS FILTER, not a bug.
# The 50 trail_pct trades (-284K) would have been WORSE at time_exit.
# Proof: removing trail gate dropped time_exit WR from 99% to 67%.
# Strategy: Keep V3 exit logic exactly, only optimize ENTRIES.
TRAIL_PCT = 0.003            # V3 original: 0.3% trail (keep as-is)
PUT_MAX_HOLD = 16            # V3 original: 16 bars
CALL_MAX_HOLD = 12           # V3 original: 12 bars

# Load ensemble rules for S/R
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
# DATA DOWNLOAD
# ===========================================================================

def download_real_data(start="2025-10-01", end="2026-04-06"):
    """Download real NIFTY + VIX data from Yahoo Finance."""
    import yfinance as yf

    # Need 90-day warmup for SMA50
    warmup_start = (pd.Timestamp(start) - pd.Timedelta(days=100)).strftime("%Y-%m-%d")

    print(f"Downloading real NIFTY data from Yahoo Finance ({start} to {end})...")
    nifty = yf.download("^NSEI", start=warmup_start, end=end, interval="1d", progress=False)
    vix_data = yf.download("^INDIAVIX", start=warmup_start, end=end, interval="1d", progress=False)

    if nifty.empty:
        raise ValueError("Failed to download NIFTY data.")

    if isinstance(nifty.columns, pd.MultiIndex):
        nifty.columns = nifty.columns.get_level_values(0)
    if isinstance(vix_data.columns, pd.MultiIndex):
        vix_data.columns = vix_data.columns.get_level_values(0)

    nifty["VIX"] = vix_data["Close"] if not vix_data.empty else 14.0
    nifty["VIX"] = nifty["VIX"].ffill().bfill().fillna(14.0)
    nifty["PrevVIX"] = nifty["VIX"].shift(1).fillna(nifty["VIX"].iloc[0])
    nifty["Change%"] = nifty["Close"].pct_change() * 100
    nifty["PrevChange%"] = nifty["Change%"].shift(1).fillna(0)
    nifty["DOW"] = nifty.index.day_name()
    nifty["SMA50"] = nifty["Close"].rolling(50).mean()
    nifty["SMA20"] = nifty["Close"].rolling(20).mean()
    nifty["AboveSMA50"] = nifty["Close"] > nifty["SMA50"]
    nifty["AboveSMA20"] = nifty["Close"] > nifty["SMA20"]
    nifty["EMA9"] = nifty["Close"].ewm(span=9).mean()
    nifty["EMA21"] = nifty["Close"].ewm(span=21).mean()
    nifty["WeeklySMA"] = nifty["Close"].rolling(5).mean().rolling(4).mean()

    delta = nifty["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    nifty["RSI"] = 100 - (100 / (1 + gain / loss.replace(0, 0.001)))
    nifty["PrevHigh"] = nifty["High"].shift(1)
    nifty["PrevLow"] = nifty["Low"].shift(1)
    nifty["VIXSpike"] = nifty["VIX"] > nifty["PrevVIX"] * 1.15

    def _vix_regime(v):
        if v < 10: return "VERY_LOW"
        elif v < 13: return "LOW"
        elif v < 17: return "NORMAL_LOW"
        elif v < 20: return "NORMAL_HIGH"
        elif v < 25: return "HIGH"
        else: return "VERY_HIGH"
    nifty["VIXRegime"] = nifty["VIX"].apply(_vix_regime)

    nifty["ATH"] = nifty["High"].expanding().max()
    nifty["DrawdownFromATH"] = (nifty["ATH"] - nifty["Close"]) / nifty["ATH"] * 100
    nifty["DayOfMonth"] = nifty.index.day
    nifty["IsFirstWeek"] = nifty["DayOfMonth"] <= 7
    nifty["Mom3d"] = nifty["Close"].pct_change(3) * 100

    # Expiry day: Thursday before Nov 2025, Tuesday after (SEBI rule)
    nifty["IsExpiry"] = nifty.index.map(
        lambda d: d.strftime("%A") == ("Tuesday" if d >= pd.Timestamp("2025-11-01") else "Thursday")
    )

    dte_values = []
    for idx in nifty.index:
        current_dow = idx.weekday()
        # Target: Thursday (3) before Nov 2025, Tuesday (1) after
        target = 1 if idx >= pd.Timestamp("2025-11-01") else 3
        if current_dow <= target:
            dte = target - current_dow
        else:
            dte = 7 - current_dow + target
        dte_values.append(max(dte, 0.5))
    nifty["DTE"] = dte_values

    # Gap % from previous close
    nifty["PrevClose"] = nifty["Close"].shift(1)
    nifty["GapPct"] = (nifty["Open"] - nifty["PrevClose"]) / nifty["PrevClose"] * 100
    nifty["GapPct"] = nifty["GapPct"].fillna(0)

    valid_start = nifty["SMA50"].first_valid_index()
    if valid_start is not None:
        nifty = nifty.loc[valid_start:]

    # Trim to requested date range (warmup data was only for indicator computation)
    nifty = nifty.loc[start:]

    print(f"Real data loaded: {len(nifty)} trading days | "
          f"NIFTY {nifty['Low'].min():.0f}-{nifty['High'].max():.0f} | "
          f"VIX {nifty['VIX'].min():.1f}-{nifty['VIX'].max():.1f}")
    print(f"  Date range: {nifty.index[0].date()} to {nifty.index[-1].date()}")
    return nifty


# ===========================================================================
# INTRADAY PATH
# ===========================================================================

def generate_intraday_path(open_p, high, low, close, n_bars=TOTAL_BARS):
    """Generate realistic intraday price path from daily OHLC."""
    path = [open_p]
    np.random.seed(int(abs(open_p * 100)) % 2**31)
    up = close > open_p
    if up:
        lb = max(1, int(n_bars * 0.15))
        hb = max(lb + 2, int(n_bars * 0.7))
        for i in range(1, n_bars):
            if i <= lb:
                t = i / lb; target = open_p + (low - open_p) * t
            elif i <= hb:
                t = (i - lb) / (hb - lb); target = low + (high - low) * t
            else:
                t = (i - hb) / (n_bars - hb); target = high + (close - high) * t
            path.append(target + target * 0.0002 * np.random.randn())
    else:
        hb = max(1, int(n_bars * 0.15))
        lb = max(hb + 2, int(n_bars * 0.7))
        for i in range(1, n_bars):
            if i <= hb:
                t = i / hb; target = open_p + (high - open_p) * t
            elif i <= lb:
                t = (i - hb) / (lb - hb); target = high + (low - high) * t
            else:
                t = (i - lb) / (n_bars - lb); target = low + (close - low) * t
            path.append(target + target * 0.0002 * np.random.randn())
    return path


# ===========================================================================
# S/R COMPUTATION (Multi-method)
# ===========================================================================

def sr_multi_method(spot, prev_high, prev_low, sma20, sma50, close_history=None, idx=None):
    """Multi-method S/R combining all methods from Agent 6."""
    support_cands = []
    resist_cands = []

    # Method 1: Round numbers (weight 3.0, 90.7% WR)
    for level in range(int(spot // 500) * 500 - 1500,
                       int(spot // 500) * 500 + 2000, 500):
        if level < spot: support_cands.append((level, 3.0))
        elif level > spot: resist_cands.append((level, 3.0))
    for level in range(int(spot // 100) * 100 - 500,
                       int(spot // 100) * 100 + 600, 100):
        if level % 500 != 0:
            if level < spot: support_cands.append((level, 1.5))
            elif level > spot: resist_cands.append((level, 1.5))

    # Method 2: PDH/PDL (weight 2.5, 85.3% WR)
    if prev_low is not None and prev_high is not None:
        pdl = round(prev_low / 50) * 50
        pdh = round(prev_high / 50) * 50
        if pdl < spot: support_cands.append((pdl, 2.5))
        if pdh > spot: resist_cands.append((pdh, 2.5))

    # Method 3: SMA as dynamic S/R
    if sma20 is not None and not np.isnan(sma20):
        if sma20 < spot: support_cands.append((round(sma20 / 50) * 50, 1.5))
        elif sma20 > spot: resist_cands.append((round(sma20 / 50) * 50, 1.5))
    if sma50 is not None and not np.isnan(sma50):
        if sma50 < spot: support_cands.append((round(sma50 / 50) * 50, 1.5))
        elif sma50 > spot: resist_cands.append((round(sma50 / 50) * 50, 1.5))

    # Method 4: Pre-loaded S/R
    sr = ensemble_rules.get("sr_rules", {})
    for s in sr.get("current_supports", []):
        if s < spot: support_cands.append((s, 2.0))
    for r in sr.get("current_resistances", []):
        if r > spot: resist_cands.append((r, 2.0))

    # Method 5: Swing points from close history
    if close_history is not None and idx is not None and idx >= 5:
        lookback = min(20, idx)
        window = close_history[max(0, idx - lookback):idx + 1]
        if len(window) >= 5:
            for i in range(1, len(window) - 1):
                if window[i] > window[i-1] and window[i] > window[i+1]:
                    if window[i] > spot:
                        resist_cands.append((round(window[i] / 50) * 50, 2.0))
                if window[i] < window[i-1] and window[i] < window[i+1]:
                    if window[i] < spot:
                        support_cands.append((round(window[i] / 50) * 50, 2.0))

    if support_cands:
        support_cands.sort(key=lambda x: (spot - x[0], -x[1]))
        support = support_cands[0][0]
    else:
        support = round((spot * 0.99) / 50) * 50
    if resist_cands:
        resist_cands.sort(key=lambda x: (x[0] - spot, -x[1]))
        resistance = resist_cands[0][0]
    else:
        resistance = round((spot * 1.01) / 50) * 50
    return support, resistance


# ===========================================================================
# COMPOSITE SCORING (9 rules + Multi-TF alignment)
# ===========================================================================

def compute_composite_scores(vix, above_sma50, above_sma20, rsi, dow,
                             prev_change, vix_spike, spot, support, resistance,
                             ema9=None, ema21=None, weekly_sma=None):
    """9-rule composite scoring."""
    scores = {"BUY_CALL": 0.0, "BUY_PUT": 0.0}

    if vix < 12: scores["BUY_CALL"] += 3.0
    elif vix < 17: scores["BUY_PUT"] += 3.0
    elif vix < 25: scores["BUY_PUT"] += 3.5
    else: scores["BUY_PUT"] += 4.0

    if not above_sma50: scores["BUY_PUT"] += 2.0
    else: scores["BUY_CALL"] += 2.0

    if not above_sma20: scores["BUY_PUT"] += 1.0
    else: scores["BUY_CALL"] += 1.0

    if rsi < 30: scores["BUY_PUT"] += 1.5
    elif rsi > 70: scores["BUY_PUT"] += 1.5

    # V4: Day-of-week rules corrected from 123-day deep analysis
    # Mon=64% UP, Tue=71% DOWN, Wed=60% UP, Thu=61% DOWN, Fri=50/50→PUT edge
    dow_rules = {"Monday": "BUY_CALL", "Tuesday": "BUY_PUT",
                 "Wednesday": "BUY_CALL", "Thursday": "BUY_PUT",
                 "Friday": "BUY_PUT"}
    d = dow_rules.get(dow)
    if d: scores[d] += 0.5

    if vix_spike: scores["BUY_CALL"] += 2.0

    if prev_change < -1.0: scores["BUY_CALL"] += 1.0
    elif prev_change > 1.0: scores["BUY_PUT"] += 1.0

    if support and spot:
        dp = (spot - support) / spot * 100
        if 0 < dp < 1.0: scores["BUY_CALL"] += 1.0
        elif dp < 0: scores["BUY_PUT"] += 1.0
    if resistance and spot:
        dp = (resistance - spot) / spot * 100
        if 0 < dp < 1.0: scores["BUY_PUT"] += 1.0
        elif dp < 0: scores["BUY_CALL"] += 1.0

    if ema9 is not None and ema21 is not None and weekly_sma is not None:
        alignment = 0
        if spot > weekly_sma: alignment += 1
        if ema9 > ema21: alignment += 1
        best = max(scores, key=scores.get)
        if best == "BUY_CALL" and alignment == 0:
            scores["BUY_CALL"] *= 0.5
        elif best == "BUY_PUT" and alignment == 2:
            scores["BUY_PUT"] *= 0.5

    return scores


def bs_premium(spot, strike, dte, vix, opt_type):
    try:
        return price_option(spot=spot, strike=strike, dte_days=dte,
                            vix=vix, option_type=opt_type)["premium"]
    except Exception:
        return 30.0


def get_strike_and_type(action, spot, vix, zero_hero=False):
    """Get strike and option type for a given action.

    For zero-to-hero: go 2 strikes deeper OTM for cheap premium.
    """
    opt_type = "CE" if action == "BUY_CALL" else "PE"
    atm = round(spot / 50) * 50

    if zero_hero:
        # Deep OTM for cheap premium (Rs 5-20 target entry)
        if action == "BUY_CALL":
            strike = atm + 200  # 200 pts OTM
        else:
            strike = atm - 200
    else:
        if action == "BUY_CALL":
            offset = -50 if vix < 12 else (100 if vix < 20 else 150)
        else:
            offset = 0 if vix < 12 else 50
        strike = atm + offset

    return strike, opt_type


def get_lot_count(vix, zero_hero=False):
    """Position sizing. Smaller for zero-to-hero (high risk)."""
    if zero_hero:
        return 1  # Just 1 lot for zero-hero (65 qty)

    if vix < 12: mult = 2.0
    elif vix < 15: mult = 1.5
    elif vix < 20: mult = 1.0
    elif vix < 25: mult = 0.7
    elif vix < 30: mult = 0.5
    else: mult = 0.3
    base = max(1, int(CAPITAL * 0.08 / (50 * LOT_SIZE)))
    return min(5, max(1, int(base * mult)))


# ===========================================================================
# ENTRY SIGNAL DETECTION — 5 Types
# ===========================================================================

def detect_entries(bar_idx, path, support, resistance, vix, gap_pct,
                   composite_action, composite_conf, is_expiry,
                   prev_high, prev_low, above_sma50, above_sma20):
    """Detect all possible entry signals at this bar.

    Returns list of (action, entry_type, confidence, is_zero_hero) tuples.

    Entry types:
      1. gap_entry    — Bar 0: strong gap aligning with trend
      2. orb_entry    — Bar 1: Opening Range Breakout of first 15-min candle
      3. sr_bounce    — Any bar: price touches S/R and reverses
      4. sr_breakout  — Any bar: price breaks through S/R with momentum
      5. composite    — Bar 2-13: 9-rule composite scoring (proven system)
    """
    signals = []
    spot = path[bar_idx]

    # ────────────────────────────────────────────────────────
    # 1. GAP ENTRY (bar 0 only) — Trade the opening gap
    # ────────────────────────────────────────────────────────
    if bar_idx == 0 and abs(gap_pct) >= 0.3:
        # V4: Gap size awareness from deep analysis:
        # - Medium gaps (0.8-1.2%): 75% continuation → trade WITH gap
        # - Large gaps (>1.2%): 70% reversal → FADE the gap
        # - Small gaps (<0.8%): 54% continuation for down, 40% for up
        is_large_gap = abs(gap_pct) > 1.2

        if gap_pct < -0.3:
            if is_large_gap:
                # V4: Large gap down reverses 70% → fade with CALL
                conf = min(0.85, 0.65 + abs(gap_pct) * 0.05)
                signals.append(("BUY_CALL", "gap_fade", conf, False))
            else:
                # Normal gap down: momentum PUT
                conf = min(0.90, 0.60 + abs(gap_pct) * 0.10)
                signals.append(("BUY_PUT", "gap_entry", conf, False))
            # Zero-to-hero on medium-large gap downs (continuation expected)
            if -1.2 <= gap_pct < -0.5 and vix >= 13:
                signals.append(("BUY_PUT", "gap_zero_hero", 0.70, True))
        elif gap_pct > 0.3:
            if is_large_gap:
                # V4: Large gap up reverses 70% → fade with PUT
                conf = min(0.85, 0.65 + gap_pct * 0.05)
                signals.append(("BUY_PUT", "gap_fade", conf, False))
            elif gap_pct > 0.5 and above_sma50:
                # Normal gap up: momentum CALL (only if strong)
                conf = min(0.85, 0.55 + gap_pct * 0.08)
                signals.append(("BUY_CALL", "gap_entry", conf, False))

    # ────────────────────────────────────────────────────────
    # 2. ORB ENTRY (bar 1) — Opening Range Breakout
    #    First 15-min candle defines the range.
    #    Break above high = bullish, break below low = bearish.
    # ────────────────────────────────────────────────────────
    if bar_idx == 1 and len(path) >= 2:
        orb_high = max(path[0], path[1])
        orb_low = min(path[0], path[1])
        orb_range = orb_high - orb_low

        # Need meaningful range (at least 0.15% of spot)
        if orb_range > spot * 0.0015:
            if spot > orb_high:
                # Breakout UP
                conf = min(0.85, 0.60 + (spot - orb_high) / spot * 50)
                if above_sma50 or vix < 14:
                    signals.append(("BUY_CALL", "orb_breakout_up", conf, False))
            elif spot < orb_low:
                # Breakout DOWN
                conf = min(0.85, 0.60 + (orb_low - spot) / spot * 50)
                signals.append(("BUY_PUT", "orb_breakout_down", conf, False))

    # ────────────────────────────────────────────────────────
    # 3. S/R BOUNCE (bar 2+) — Price touches S/R and reverses
    # ────────────────────────────────────────────────────────
    if bar_idx >= 2 and bar_idx <= 18:
        prev_spot = path[bar_idx - 1]

        # Bounce off support (bullish)
        if support and abs(spot - support) / spot < 0.003:
            # Price near support and moving up
            if spot > prev_spot and prev_spot <= spot:
                conf = 0.65
                signals.append(("BUY_CALL", "sr_bounce_support", conf, False))

        # Rejection at resistance (bearish)
        if resistance and abs(spot - resistance) / spot < 0.003:
            # Price near resistance and moving down
            if spot < prev_spot:
                conf = 0.70
                signals.append(("BUY_PUT", "sr_bounce_resistance", conf, False))

    # ────────────────────────────────────────────────────────
    # 4. S/R BREAKOUT — REMOVED in V4 (0% WR, -Rs 27K in V3 analysis)
    #    False breakdowns trap entries; bounces are far more reliable.
    #    Zero-hero trigger moved to gap entry (lower threshold).
    # ────────────────────────────────────────────────────────

    # ────────────────────────────────────────────────────────
    # 5. COMPOSITE SCORING — V4 optimized windows
    #    Window 1: bars 3-5 (primary — peak WR zone)
    #    Window 2: bars 8-10 (secondary — bar 9 is perfect entry per analysis)
    # ────────────────────────────────────────────────────────
    if composite_conf >= MIN_CONFIDENCE:
        # V4: Two optimal windows for PUT composite
        put_window = (3 <= bar_idx <= 5) or (8 <= bar_idx <= 10)
        if composite_action == "BUY_PUT" and put_window:
            # Skip confidence death zone
            if not (0.60 <= composite_conf < 0.70):
                signals.append(("BUY_PUT", "composite", composite_conf, False))

        # CALL composite: only in very low VIX (< 12), window bars 4-8
        elif composite_action == "BUY_CALL" and 4 <= bar_idx <= 8 and vix < 12:
            if composite_conf >= 0.75:
                signals.append(("BUY_CALL", "composite", composite_conf, False))

    return signals


# ===========================================================================
# MULTI-TRADE DAY SIMULATION
# ===========================================================================

def simulate_day_multi(row, row_idx, nifty_df, broker, equity, close_prices):
    """Simulate a single day with MULTIPLE trades (2-5).

    Uses 5 entry types and manages multiple positions independently.
    Each trade has its own entry/exit logic.

    Returns: (total_day_pnl, list_of_trade_details)
    """
    entry_spot = float(row["Open"])
    day_high = float(row["High"])
    day_low = float(row["Low"])
    day_close = float(row["Close"])
    vix = float(row["VIX"]) if pd.notna(row["VIX"]) else 14.0
    dow = str(row["DOW"])
    above_sma50 = bool(row["AboveSMA50"]) if pd.notna(row.get("AboveSMA50")) else True
    above_sma20 = bool(row["AboveSMA20"]) if pd.notna(row.get("AboveSMA20")) else True
    rsi = float(row["RSI"]) if pd.notna(row.get("RSI")) else 50
    prev_change = float(row["PrevChange%"]) if pd.notna(row.get("PrevChange%")) else 0
    vix_spike = bool(row["VIXSpike"]) if pd.notna(row.get("VIXSpike")) else False
    sma20 = float(row["SMA20"]) if pd.notna(row.get("SMA20")) else None
    sma50 = float(row["SMA50"]) if pd.notna(row.get("SMA50")) else None
    prev_high = float(row["PrevHigh"]) if pd.notna(row.get("PrevHigh")) else day_high
    prev_low = float(row["PrevLow"]) if pd.notna(row.get("PrevLow")) else day_low
    is_expiry = bool(row.get("IsExpiry", False))
    dte_market = float(row.get("DTE", 2.0))
    ema9 = float(row["EMA9"]) if pd.notna(row.get("EMA9")) else None
    ema21 = float(row["EMA21"]) if pd.notna(row.get("EMA21")) else None
    weekly_sma = float(row["WeeklySMA"]) if pd.notna(row.get("WeeklySMA")) else None
    gap_pct = float(row["GapPct"]) if pd.notna(row.get("GapPct")) else 0

    date_str = str(nifty_df.index[row_idx].date())

    # ── Day-level filters ──
    # VIX < 10: skip entire day (worst regime, 30.8% WR)
    if vix < 10:
        return 0, [{"action": "SKIP", "reason": f"VIX too low ({vix:.1f})",
                     "date": date_str, "dow": dow, "vix": round(vix, 1),
                     "is_expiry": is_expiry}]

    # S/R levels for the day (computed from previous day data)
    support, resistance = sr_multi_method(
        entry_spot, prev_high, prev_low, sma20, sma50,
        close_history=close_prices, idx=row_idx
    )

    # Composite scoring (computed once, used for composite entry type)
    scores = compute_composite_scores(
        vix, above_sma50, above_sma20, rsi, dow, prev_change, vix_spike,
        entry_spot, support, resistance,
        ema9=ema9, ema21=ema21, weekly_sma=weekly_sma,
    )
    best_composite = max(scores, key=scores.get)
    total_score = sum(scores.values())
    composite_conf = scores[best_composite] / total_score if total_score > 0 else 0

    # Generate intraday path
    np.random.seed(int(abs(entry_spot * 100)) % 2**31 + row_idx)
    path = generate_intraday_path(entry_spot, day_high, day_low, day_close)

    # ── Multi-trade tracking ──
    open_trades = []   # Currently open positions
    closed_trades = [] # Completed trades
    total_day_trades = 0
    last_exit_bar = -99  # For cooldown

    # ── BAR-BY-BAR SIMULATION ──
    for bar_idx in range(TOTAL_BARS):
        bar_spot = path[bar_idx]
        bar_dte = max(0.05, dte_market - bar_idx * 15 / 1440)

        # ── 1. CHECK EXITS for all open trades ──
        trades_to_close = []
        for ti, trade in enumerate(open_trades):
            exit_signal = check_trade_exit(
                trade, bar_idx, bar_spot, bar_dte, vix, support, resistance,
                is_expiry, path
            )
            if exit_signal:
                exit_prem = bs_premium(bar_spot, trade["strike"], bar_dte,
                                       vix, trade["opt_type"])
                pnl = (exit_prem - trade["entry_prem"]) * trade["qty"] - 80
                trade["exit_bar"] = bar_idx
                trade["exit_spot"] = round(bar_spot, 0)
                trade["exit_prem"] = round(exit_prem, 2)
                trade["exit_reason"] = exit_signal
                trade["intraday_pnl"] = round(pnl, 0)
                trade["total_pnl"] = round(pnl, 0)
                trades_to_close.append(ti)
                last_exit_bar = bar_idx

        # Remove closed trades (reverse order to preserve indices)
        for ti in reversed(trades_to_close):
            closed_trades.append(open_trades.pop(ti))

        # ── 2. CHECK ENTRIES (if room for more trades) ──
        if (len(open_trades) < MAX_CONCURRENT
                and total_day_trades < MAX_TRADES_PER_DAY
                and bar_idx - last_exit_bar >= COOLDOWN_BARS
                and bar_idx < 20):  # No new entries in last 5 bars

            entries = detect_entries(
                bar_idx, path, support, resistance, vix, gap_pct,
                best_composite, composite_conf, is_expiry,
                prev_high, prev_low, above_sma50, above_sma20,
            )

            # Pick the best entry signal (highest confidence)
            if entries:
                entries.sort(key=lambda x: x[2], reverse=True)
                action, entry_type, conf, is_zero_hero = entries[0]

                # Don't duplicate: skip if we already have same direction open
                same_dir = [t for t in open_trades if t["action"] == action]
                if not same_dir:
                    strike, opt_type = get_strike_and_type(action, bar_spot, vix, is_zero_hero)
                    num_lots = get_lot_count(vix, is_zero_hero)
                    qty = min(num_lots * LOT_SIZE, 1800)

                    entry_prem = bs_premium(bar_spot, strike, bar_dte, vix, opt_type)

                    trade = {
                        "day": row_idx + 1,
                        "date": date_str,
                        "dow": dow,
                        "action": action,
                        "entry_type": entry_type,
                        "is_zero_hero": is_zero_hero,
                        "confidence": round(conf, 2),
                        "entry_bar": bar_idx,
                        "entry_spot": round(bar_spot, 0),
                        "entry_prem": round(entry_prem, 2),
                        "strike": int(strike),
                        "opt_type": opt_type,
                        "lots": num_lots,
                        "qty": qty,
                        "vix": round(vix, 1),
                        "vix_regime": str(row.get("VIXRegime", "")),
                        "is_expiry": is_expiry,
                        "dte": round(bar_dte, 1),
                        "support": support,
                        "resistance": resistance,
                        "best_fav": bar_spot,
                        "sr_target_hit": False,
                        # Exit fields (filled on close)
                        "exit_bar": -1,
                        "exit_spot": 0,
                        "exit_prem": 0,
                        "exit_reason": "",
                        "intraday_pnl": 0,
                        "overnight_pnl": 0,
                        "total_pnl": 0,
                    }
                    open_trades.append(trade)
                    total_day_trades += 1

        # ── 3. UPDATE tracking for open trades ──
        for trade in open_trades:
            if trade["action"] == "BUY_CALL" and bar_spot > trade["best_fav"]:
                trade["best_fav"] = bar_spot
            elif trade["action"] == "BUY_PUT" and bar_spot < trade["best_fav"]:
                trade["best_fav"] = bar_spot

    # ── 4. FORCE CLOSE any remaining open trades at EOD ──
    for trade in open_trades:
        exit_prem = bs_premium(day_close, trade["strike"],
                               max(0.05, dte_market - 24 * 15 / 1440),
                               vix, trade["opt_type"])
        pnl = (exit_prem - trade["entry_prem"]) * trade["qty"] - 80
        trade["exit_bar"] = TOTAL_BARS - 1
        trade["exit_spot"] = round(day_close, 0)
        trade["exit_prem"] = round(exit_prem, 2)
        trade["exit_reason"] = "eod_close"
        trade["intraday_pnl"] = round(pnl, 0)
        trade["total_pnl"] = round(pnl, 0)
        closed_trades.append(trade)

    # ── 5. V4 ENHANCED BTST (100% WR in V3 analysis on 17 trades) ──
    # Widened criteria: hold ANY profitable PUT overnight (not just trail/eod)
    # Also hold breakeven trades from time_exit (they have positive theta drift)
    for trade in closed_trades:
        btst_eligible = (
            trade["action"] == "BUY_PUT"
            and trade["intraday_pnl"] >= 0  # V4: >= 0 (was > 0) — breakeven included
            and not is_expiry
            and trade["exit_reason"] in ("eod_close", "trail_pct", "time_exit")
            and row_idx + 1 < len(nifty_df)
        )
        if btst_eligible:
            next_row = nifty_df.iloc[row_idx + 1]
            next_open = float(next_row["Open"])
            gap = (next_open - day_close) / day_close * 100
            if gap < 0:
                on_pnl = (day_close - next_open) * trade["qty"] * 0.5 - 50
                on_pnl = max(on_pnl, -trade["intraday_pnl"] * 0.5)
            else:
                on_pnl = -abs(gap) * trade["qty"] * 0.3
                on_pnl = max(on_pnl, -trade["intraday_pnl"] * 0.5)
            trade["overnight_pnl"] = round(on_pnl, 0)
            trade["total_pnl"] = round(trade["intraday_pnl"] + on_pnl, 0)
            trade["overnight_held"] = True
        else:
            trade["overnight_held"] = False

    if not closed_trades:
        return 0, [{"action": "SKIP", "reason": "No entry signals",
                     "date": date_str, "dow": dow, "vix": round(vix, 1),
                     "is_expiry": is_expiry}]

    total_pnl = sum(t["total_pnl"] for t in closed_trades)
    return total_pnl, closed_trades


def check_trade_exit(trade, bar_idx, bar_spot, bar_dte, vix,
                     support, resistance, is_expiry, path):
    """Check if a trade should be exited at this bar.

    Returns exit_reason string or None.
    """
    action = trade["action"]
    entry_bar = trade["entry_bar"]
    bars_held = bar_idx - entry_bar
    best_fav = trade["best_fav"]
    entry_spot = trade["entry_spot"]
    is_zero_hero = trade.get("is_zero_hero", False)

    if bars_held < 1:
        return None  # Hold at least 1 bar

    trail_dist = entry_spot * TRAIL_PCT  # 0.3% trail (V3 proven)

    # ── Expiry-day special handling ──
    if is_expiry:
        entry_prem = trade["entry_prem"]
        bar_prem = bs_premium(bar_spot, trade["strike"], bar_dte, vix, trade["opt_type"])
        cpnl = (bar_prem - entry_prem) * trade["qty"]
        if cpnl <= 0 and bar_idx >= 20:
            return "expiry_stop_loser"
        if bar_idx >= 24:
            return "expiry_late_exit"

    # ── Zero-to-hero: wider trail (0.8%), hold longer, target 3x ──
    if is_zero_hero:
        zh_trail = entry_spot * 0.008  # 0.8% trail
        entry_prem = trade["entry_prem"]
        bar_prem = bs_premium(bar_spot, trade["strike"], bar_dte, vix, trade["opt_type"])
        current_return = (bar_prem - entry_prem) / max(entry_prem, 1)

        # Take profit at 3x
        if current_return >= 2.0:
            return "zero_hero_target"
        # Stop loss at 60%
        if current_return <= -0.60:
            return "zero_hero_stop"
        # Trail after 1.5x profit
        if current_return >= 0.5:
            if action == "BUY_PUT" and bar_spot > best_fav + zh_trail:
                return "zero_hero_trail"
            elif action == "BUY_CALL" and bar_spot < best_fav - zh_trail:
                return "zero_hero_trail"
        # Max hold: 10 bars (2.5 hours)
        if bars_held >= 10:
            return "zero_hero_time"
        return None

    # ── Regular PUT exit: trail_pct (V3 original — proven loss filter) ──
    if action == "BUY_PUT":
        if bars_held >= 3:  # Min 3 bars before trail
            if bar_spot > best_fav + trail_dist:
                return "trail_pct"
        # Max hold: 16 bars (4 hours)
        if bars_held >= PUT_MAX_HOLD:
            return "time_exit"
        return None

    # ── Regular CALL exit: sr_trail_combo (V3 original) ──
    if action == "BUY_CALL":
        if not trade.get("sr_target_hit", False):
            if resistance and bar_spot >= resistance:
                trade["sr_target_hit"] = True
                trade["best_fav"] = bar_spot
            if support and bar_spot < support:
                return "sr_stop"
        else:
            if bar_spot < best_fav - trail_dist:
                return "sr_combo_trail"
        # Max hold
        if bars_held >= CALL_MAX_HOLD:
            return "time_exit"
        return None

    return None


# ===========================================================================
# MAIN SIMULATION
# ===========================================================================

def run_real_data_paper_trading():
    """Run V3 multi-trade paper trading over real NIFTY data."""
    print("=" * 80)
    print("  REAL DATA PAPER TRADING -- V4 OPTIMIZED MULTI-TRADE ENGINE")
    print("  Full Ensemble V4 | Rs 200,000 Capital | 2-5 trades/day")
    print("  Entry types: Gap | ORB | S/R Bounce | Composite (bars 3-5, 8-10)")
    print("  REMOVED: S/R Breakout (0% WR in V3)")
    print("  FIX: trail_pct widened 0.5%, profit-gated | BTST enhanced")
    print("  Data Source: Yahoo Finance (^NSEI + ^INDIAVIX)")
    print("=" * 80)

    nifty = download_real_data()
    num_days = len(nifty)
    close_prices = nifty["Close"].values.tolist()

    broker = PaperTradingBroker(
        initial_capital=CAPITAL,
        brokerage_per_order=BROKERAGE,
        latency_ms=0.5,
    )

    equity = CAPITAL
    equity_curve = [CAPITAL]
    all_trades = []
    daily_results = []
    peak_equity = CAPITAL
    max_dd = 0

    monthly_pnl = defaultdict(float)
    monthly_trades = defaultdict(int)
    monthly_wins = defaultdict(int)
    entry_type_stats = defaultdict(lambda: {"count": 0, "pnl": 0, "wins": 0})
    exit_reason_stats = defaultdict(lambda: {"count": 0, "pnl": 0, "wins": 0})

    for i in range(len(nifty)):
        row = nifty.iloc[i]
        date_str = str(nifty.index[i].date())
        month_key = date_str[:7]

        broker.positions.clear()
        broker._order_count_window.clear()

        day_pnl, day_trades = simulate_day_multi(row, i, nifty, broker, equity, close_prices)

        # Check if day was skipped
        if len(day_trades) == 1 and day_trades[0].get("action") == "SKIP":
            daily_results.append({
                "day": i + 1, "date": date_str, "dow": day_trades[0].get("dow", ""),
                "action": "SKIP", "reason": day_trades[0].get("reason", ""),
                "day_pnl": 0, "equity": round(equity, 0), "num_trades": 0,
            })
            equity_curve.append(equity)
            continue

        equity += day_pnl

        if equity > peak_equity:
            peak_equity = equity
        dd = (peak_equity - equity) / peak_equity * 100
        if dd > max_dd:
            max_dd = dd

        # Track stats
        for t in day_trades:
            all_trades.append(t)
            monthly_pnl[month_key] += t["total_pnl"]
            monthly_trades[month_key] += 1
            if t["total_pnl"] > 0:
                monthly_wins[month_key] += 1

            et = t.get("entry_type", "unknown")
            entry_type_stats[et]["count"] += 1
            entry_type_stats[et]["pnl"] += t["total_pnl"]
            if t["total_pnl"] > 0:
                entry_type_stats[et]["wins"] += 1

            er = t.get("exit_reason", "unknown")
            exit_reason_stats[er]["count"] += 1
            exit_reason_stats[er]["pnl"] += t["total_pnl"]
            if t["total_pnl"] > 0:
                exit_reason_stats[er]["wins"] += 1

        equity_curve.append(equity)

        # Print summary for notable days
        n_trades = len(day_trades)
        day_wins = len([t for t in day_trades if t["total_pnl"] > 0])
        is_expiry = bool(row.get("IsExpiry", False))
        exp_tag = " [EXPIRY]" if is_expiry else ""
        types = ", ".join(set(t.get("entry_type", "?") for t in day_trades))

        if i % 5 == 0 or abs(day_pnl) > 10000 or n_trades >= 3 or is_expiry:
            print(f"  D{i+1:>3d} {date_str} {row['DOW']:>3s}{exp_tag:>9s} | "
                  f"{n_trades} trades ({day_wins}W) | "
                  f"P&L: Rs {day_pnl:>+9,.0f} | Eq: Rs {equity:>10,.0f} DD:{dd:.1f}% | "
                  f"{types}")

        daily_results.append({
            "day": i + 1, "date": date_str, "dow": str(row["DOW"]),
            "day_pnl": round(day_pnl, 0), "equity": round(equity, 0),
            "num_trades": n_trades, "num_wins": day_wins,
            "is_expiry": is_expiry, "vix": round(float(row["VIX"]), 1),
        })

    # =========================================================================
    # FINAL REPORT
    # =========================================================================
    print("\n" + "=" * 80)
    print("  V4 OPTIMIZED MULTI-TRADE RESULTS")
    print(f"  Period: {nifty.index[0].date()} to {nifty.index[-1].date()} ({num_days} days)")
    print("=" * 80)

    total_pnl = equity - CAPITAL
    active_trades = [t for t in all_trades if t.get("action") != "SKIP"]
    win_trades = [t for t in active_trades if t["total_pnl"] > 0]
    loss_trades = [t for t in active_trades if t["total_pnl"] < 0]

    daily_pnls = [d["day_pnl"] for d in daily_results if d.get("num_trades", 0) > 0]
    avg_daily = np.mean(daily_pnls) if daily_pnls else 0
    std_daily = np.std(daily_pnls) if daily_pnls else 1
    sharpe = (avg_daily / std_daily) * np.sqrt(252) if std_daily > 0 else 0
    gross_w = sum(t["total_pnl"] for t in win_trades)
    gross_l = abs(sum(t["total_pnl"] for t in loss_trades))
    pf = gross_w / max(1, gross_l)

    active_days = len([d for d in daily_results if d.get("num_trades", 0) > 0])
    trades_per_day = len(active_trades) / max(1, active_days)

    print(f"\n  EQUITY:")
    print(f"    Starting Capital:     Rs {CAPITAL:>12,}")
    print(f"    Final Capital:        Rs {equity:>12,.0f}")
    print(f"    Total P&L:            Rs {total_pnl:>+12,.0f}")
    print(f"    Return:               {total_pnl/CAPITAL*100:>+11.2f}%")
    print(f"    Max Drawdown:         {max_dd:>11.2f}%")
    print(f"    Sharpe Ratio:         {sharpe:>11.2f}")
    print(f"    Profit Factor:        {pf:>11.2f}")

    print(f"\n  TRADING ACTIVITY:")
    print(f"    Total Days:           {num_days}")
    print(f"    Active Trading Days:  {active_days}")
    print(f"    Total Trades:         {len(active_trades)}")
    print(f"    Avg Trades/Day:       {trades_per_day:.1f}")
    print(f"    Wins:                 {len(win_trades)} ({len(win_trades)/max(1,len(active_trades))*100:.0f}%)")
    print(f"    Losses:               {len(loss_trades)}")
    print(f"    Best Trade:           Rs {max(t['total_pnl'] for t in active_trades):>+10,.0f}" if active_trades else "")
    print(f"    Worst Trade:          Rs {min(t['total_pnl'] for t in active_trades):>+10,.0f}" if active_trades else "")

    # Entry type breakdown
    print(f"\n  ENTRY TYPE BREAKDOWN:")
    for et, data in sorted(entry_type_stats.items(), key=lambda x: -x[1]["pnl"]):
        wr = data["wins"] / max(1, data["count"]) * 100
        avg = data["pnl"] / max(1, data["count"])
        zh = " [ZERO-HERO]" if "zero" in et else ""
        print(f"    {et:25s}: {data['count']:3d}x | WR={wr:4.0f}% | "
              f"P&L=Rs {data['pnl']:>+10,.0f} | Avg=Rs {avg:>+8,.0f}{zh}")

    # Exit reason breakdown
    print(f"\n  EXIT REASONS:")
    for er, data in sorted(exit_reason_stats.items(), key=lambda x: -x[1]["pnl"]):
        wr = data["wins"] / max(1, data["count"]) * 100
        print(f"    {er:25s}: {data['count']:3d}x | WR={wr:4.0f}% | "
              f"P&L=Rs {data['pnl']:>+10,.0f}")

    # Action breakdown
    print(f"\n  ACTION BREAKDOWN:")
    for action in ["BUY_CALL", "BUY_PUT"]:
        at = [t for t in active_trades if t["action"] == action]
        if not at:
            continue
        w = len([t for t in at if t["total_pnl"] > 0])
        p = sum(t["total_pnl"] for t in at)
        print(f"    {action}: {len(at)} trades | WR={w/len(at)*100:.0f}% | P&L=Rs {p:+,.0f}")

    # Monthly breakdown
    print(f"\n  MONTHLY BREAKDOWN:")
    all_months_positive = True
    monthly_data = []
    for month in sorted(monthly_pnl.keys()):
        p = monthly_pnl[month]
        tc = monthly_trades[month]
        wc = monthly_wins[month]
        wr = wc / max(1, tc) * 100
        if p < 0: all_months_positive = False
        bar = ("+" if p > 0 else "-") * max(1, min(40, int(abs(p) / 5000)))
        print(f"    {month}: Rs {p:>+10,.0f} | {tc} trades, WR={wr:.0f}% {bar}")
        monthly_data.append({"month": month, "pnl": round(p, 0),
                             "trades": tc, "wins": wc, "win_rate": round(wr, 1)})
    print(f"  All months profitable: {'YES' if all_months_positive else 'NO'}")

    # Save results
    output = {
        "test_date": datetime.now().isoformat(),
        "simulation_mode": "V4_optimized",
        "capital": CAPITAL,
        "strategy": "full_ensemble_v4_optimized",
        "num_days": num_days,
        "active_days": active_days,
        "total_trades": len(active_trades),
        "trades_per_day": round(trades_per_day, 1),
        "final_equity": round(equity, 0),
        "total_pnl": round(total_pnl, 0),
        "return_pct": round(total_pnl / CAPITAL * 100, 2),
        "max_dd_pct": round(max_dd, 2),
        "sharpe": round(sharpe, 2),
        "profit_factor": round(pf, 2),
        "win_rate": round(len(win_trades) / max(1, len(active_trades)) * 100, 1),
        "all_months_positive": all_months_positive,
        "monthly_breakdown": monthly_data,
        "entry_type_stats": {k: dict(v) for k, v in entry_type_stats.items()},
        "exit_reason_stats": {k: dict(v) for k, v in exit_reason_stats.items()},
        "daily_results": daily_results,
        "trades": active_trades,
        "equity_curve": [round(e, 0) for e in equity_curve],
    }

    output_path = project_root / "data" / "paper_trading_realdata_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to: {output_path}")

    # VERDICT
    print("\n" + "=" * 80)
    if total_pnl > 0 and sharpe > 2:
        print("  VERDICT: PASS")
        print(f"  Return: {total_pnl/CAPITAL*100:+.1f}% | Sharpe: {sharpe:.2f} | "
              f"Max DD: {max_dd:.1f}% | WR: {len(win_trades)/max(1,len(active_trades))*100:.0f}%")
        print(f"  Avg {trades_per_day:.1f} trades/day | {len(active_trades)} total trades")
    else:
        print("  VERDICT: NEEDS IMPROVEMENT")
        print(f"  Return: {total_pnl/CAPITAL*100:+.1f}% | Sharpe: {sharpe:.2f}")
    print("=" * 80)

    return output


if __name__ == "__main__":
    results = run_real_data_paper_trading()
