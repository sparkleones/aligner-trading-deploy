"""
Deep Study of Support/Resistance Levels for NIFTY Options Entry & Exit.

Downloads 6 months of NIFTY daily data and studies 5 S/R detection methods:
  A. Swing Highs/Lows (pivot points) with multiple lookback windows
  B. Price Clustering (consolidation zones)
  C. Round Number Levels
  D. Moving Average S/R (SMA20, SMA50)
  E. Previous Day High/Low/Close (PDH/PDL/PDC)

For each method, tests:
  - Bounce rate (how often price reverses at S/R)
  - Break rate (how often price breaks through S/R)
  - Optimal S/R-based entry signals (BUY_CALL near support, BUY_PUT near resistance)
  - Optimal S/R-based exit signals (exit at opposite S/R level)
  - S/R strength scoring
  - Multi-timeframe analysis

Saves results to data/sr_rules.json with actionable rules for the ensemble agent.
"""

import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backtesting.option_pricer import price_option
from config.constants import (
    INDEX_CONFIG,
    STT_RATES,
    SEBI_TURNOVER_FEE,
    NSE_TRANSACTION_CHARGE,
    STAMP_DUTY_BUY,
    GST_RATE,
)

# ── Configuration ─────────────────────────────────────────────────────────

LOT_SIZE = INDEX_CONFIG["NIFTY"]["lot_size"]  # 65
STRIKE_INTERVAL = INDEX_CONFIG["NIFTY"]["strike_interval"]  # 50
CAPITAL = 200_000
DTE = 2
RISK_FREE_RATE = 0.07
TOTAL_BARS = 25  # 15-min bars in trading day (9:15-3:30)

# S/R proximity thresholds (points from S/R level)
SR_PROXIMITY_POINTS = 75  # within 75 points = "near" S/R
SR_PROXIMITY_PCT = 0.003  # 0.3% of spot


def compute_transaction_costs(premium: float, lots: int = 1) -> float:
    """Compute round-trip Zerodha transaction costs for options."""
    qty = lots * LOT_SIZE
    turnover = premium * qty
    brokerage = 20.0 * 2
    stt = turnover * STT_RATES["options_sell"]
    exchange_charge = turnover * NSE_TRANSACTION_CHARGE
    sebi_fee = turnover * SEBI_TURNOVER_FEE
    stamp = turnover * STAMP_DUTY_BUY
    gst = (brokerage + exchange_charge) * GST_RATE
    total = brokerage + stt + exchange_charge + sebi_fee + stamp + gst
    return round(total, 2)


def download_data():
    """Download NIFTY and India VIX data using yfinance."""
    import yfinance as yf

    print("=" * 70)
    print("SUPPORT/RESISTANCE DEEP STUDY")
    print("=" * 70)

    start_date = "2025-10-01"
    end_date = "2026-04-06"

    print(f"\nDownloading NIFTY (^NSEI): {start_date} to {end_date}")
    nifty = yf.download("^NSEI", start=start_date, end=end_date, interval="1d", progress=False)

    print(f"Downloading India VIX (^INDIAVIX): {start_date} to {end_date}")
    vix = yf.download("^INDIAVIX", start=start_date, end=end_date, interval="1d", progress=False)

    # Handle multi-level columns
    if isinstance(nifty.columns, pd.MultiIndex):
        nifty.columns = nifty.columns.get_level_values(0)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)

    print(f"\nNIFTY data: {len(nifty)} trading days")
    print(f"India VIX data: {len(vix)} trading days")

    if len(nifty) == 0:
        print("ERROR: No NIFTY data downloaded.")
        sys.exit(1)

    nifty = nifty[["Open", "High", "Low", "Close", "Volume"]].copy()
    nifty.columns = ["open", "high", "low", "close", "volume"]

    if len(vix) > 0:
        vix_close = vix[["Close"]].copy()
        vix_close.columns = ["vix"]
        nifty = nifty.join(vix_close, how="left")
        nifty["vix"] = nifty["vix"].ffill().fillna(14.0)
    else:
        nifty["vix"] = 14.0

    # Compute SMAs
    nifty["sma20"] = nifty["close"].rolling(20).mean()
    nifty["sma50"] = nifty["close"].rolling(50).mean()

    # Previous day values
    nifty["prev_close"] = nifty["close"].shift(1)
    nifty["prev_high"] = nifty["high"].shift(1)
    nifty["prev_low"] = nifty["low"].shift(1)
    nifty["gap_pct"] = (nifty["open"] - nifty["prev_close"]) / nifty["prev_close"] * 100

    nifty = nifty.dropna(subset=["prev_close"])

    print(f"Final dataset: {len(nifty)} trading days")
    print(f"NIFTY range: {nifty['low'].min():.0f} - {nifty['high'].max():.0f}")
    print(f"VIX range: {nifty['vix'].min():.1f} - {nifty['vix'].max():.1f}")

    return nifty


def simulate_intraday_bars(row):
    """Simulate 25 intraday 15-min bars from daily OHLC."""
    o, h, l, c = row["open"], row["high"], row["low"], row["close"]
    daily_range = h - l
    if daily_range == 0:
        return np.full(TOTAL_BARS, c)

    np.random.seed(int(abs(o * 100)) % (2**31))
    raw = np.random.randn(TOTAL_BARS)
    raw = np.cumsum(raw)
    raw = raw - raw[0]
    if raw[-1] != 0:
        target_move = c - o
        raw = raw / raw[-1] * target_move
    raw = raw + o

    if c > o:
        high_bar = np.random.randint(TOTAL_BARS // 2, TOTAL_BARS)
        low_bar = np.random.randint(0, TOTAL_BARS // 2)
    else:
        high_bar = np.random.randint(0, TOTAL_BARS // 2)
        low_bar = np.random.randint(TOTAL_BARS // 2, TOTAL_BARS)

    raw[high_bar] = h
    raw[low_bar] = l
    raw[-1] = c
    raw[0] = o

    from scipy.interpolate import interp1d
    key_points = sorted(set([0, low_bar, high_bar, TOTAL_BARS - 1]))
    key_values = [raw[i] for i in key_points]
    f = interp1d(key_points, key_values, kind="linear")
    smooth = f(np.arange(TOTAL_BARS))

    noise = np.random.randn(TOTAL_BARS) * daily_range * 0.01
    smooth = smooth + noise
    smooth = np.clip(smooth, l, h)
    smooth[0] = o
    smooth[-1] = c

    return smooth


def price_trade(spot_entry, spot_exit, vix_val, action, dte_entry=DTE, dte_exit=None):
    """Price an option trade with entry/exit spots. Returns net PnL per lot."""
    strike = round(spot_entry / STRIKE_INTERVAL) * STRIKE_INTERVAL
    opt_type = "CE" if action == "BUY_CALL" else "PE"

    if dte_exit is None:
        dte_exit = max(dte_entry - 0.8, 0.1)

    entry_result = price_option(
        spot=spot_entry, strike=strike, dte_days=dte_entry,
        vix=vix_val, option_type=opt_type, r=RISK_FREE_RATE,
    )
    exit_result = price_option(
        spot=spot_exit, strike=strike, dte_days=dte_exit,
        vix=vix_val, option_type=opt_type, r=RISK_FREE_RATE,
    )

    pnl = (exit_result["premium"] - entry_result["premium"]) * LOT_SIZE
    avg_premium = (entry_result["premium"] + exit_result["premium"]) / 2
    costs = compute_transaction_costs(avg_premium, lots=1)
    return pnl - costs


# ═══════════════════════════════════════════════════════════════════════════
# METHOD A: SWING HIGH/LOW (PIVOT POINTS)
# ═══════════════════════════════════════════════════════════════════════════

def find_swing_points(df, lookback=10):
    """
    Find swing highs and lows using lookback window.
    Swing high: bar where high > high of N bars before and after.
    Swing low: bar where low < low of N bars before and after.
    """
    highs = df["high"].values
    lows = df["low"].values
    n = len(df)

    swing_highs = []
    swing_lows = []

    for i in range(lookback, n - lookback):
        # Check swing high
        is_swing_high = True
        for j in range(1, lookback + 1):
            if highs[i] <= highs[i - j] or highs[i] <= highs[i + j]:
                is_swing_high = False
                break
        if is_swing_high:
            swing_highs.append({"idx": i, "price": highs[i], "date": df.index[i]})

        # Check swing low
        is_swing_low = True
        for j in range(1, lookback + 1):
            if lows[i] >= lows[i - j] or lows[i] >= lows[i + j]:
                is_swing_low = False
                break
        if is_swing_low:
            swing_lows.append({"idx": i, "price": lows[i], "date": df.index[i]})

    return swing_highs, swing_lows


def study_swing_sr(nifty_df):
    """Test swing point S/R with different lookbacks."""
    print("\n" + "=" * 70)
    print("METHOD A: SWING HIGHS/LOWS (PIVOT POINTS)")
    print("=" * 70)

    lookbacks = [5, 10, 20]
    results = {}

    for lb in lookbacks:
        swing_highs, swing_lows = find_swing_points(nifty_df, lookback=lb)
        print(f"\n--- Lookback={lb} bars ---")
        print(f"  Swing Highs found: {len(swing_highs)}")
        print(f"  Swing Lows found:  {len(swing_lows)}")

        bounce_support = 0
        break_support = 0
        bounce_resistance = 0
        break_resistance = 0
        entry_pnl_support = []
        entry_pnl_resistance = []

        for day_idx in range(lb + 5, len(nifty_df)):
            row = nifty_df.iloc[day_idx]
            day_low = row["low"]
            day_high = row["high"]
            day_close = row["close"]
            vix_val = row["vix"]

            active_supports = [s["price"] for s in swing_lows if s["idx"] < day_idx]
            active_resistances = [s["price"] for s in swing_highs if s["idx"] < day_idx]

            for sup in active_supports[-5:]:
                proximity = abs(day_low - sup) / sup
                if proximity < SR_PROXIMITY_PCT:
                    if day_close > sup:
                        bounce_support += 1
                        pnl = price_trade(sup, day_close, vix_val, "BUY_CALL")
                        entry_pnl_support.append(pnl)
                    else:
                        break_support += 1

            for res in active_resistances[-5:]:
                proximity = abs(day_high - res) / res
                if proximity < SR_PROXIMITY_PCT:
                    if day_close < res:
                        bounce_resistance += 1
                        pnl = price_trade(res, day_close, vix_val, "BUY_PUT")
                        entry_pnl_resistance.append(pnl)
                    else:
                        break_resistance += 1

        total_support = bounce_support + break_support
        total_resistance = bounce_resistance + break_resistance
        support_bounce_rate = (bounce_support / total_support * 100) if total_support > 0 else 0
        resistance_bounce_rate = (bounce_resistance / total_resistance * 100) if total_resistance > 0 else 0

        avg_support_pnl = np.mean(entry_pnl_support) if entry_pnl_support else 0
        avg_resistance_pnl = np.mean(entry_pnl_resistance) if entry_pnl_resistance else 0
        wr_support = (sum(1 for p in entry_pnl_support if p > 0) / len(entry_pnl_support) * 100) if entry_pnl_support else 0
        wr_resistance = (sum(1 for p in entry_pnl_resistance if p > 0) / len(entry_pnl_resistance) * 100) if entry_pnl_resistance else 0

        print(f"  Support bounce rate:    {support_bounce_rate:.1f}% ({bounce_support}/{total_support})")
        print(f"  Resistance bounce rate: {resistance_bounce_rate:.1f}% ({bounce_resistance}/{total_resistance})")
        print(f"  BUY_CALL at support:    avg PnL={avg_support_pnl:.0f}, WR={wr_support:.1f}%, trades={len(entry_pnl_support)}")
        print(f"  BUY_PUT at resistance:  avg PnL={avg_resistance_pnl:.0f}, WR={wr_resistance:.1f}%, trades={len(entry_pnl_resistance)}")

        results[f"lookback_{lb}"] = {
            "swing_highs": len(swing_highs),
            "swing_lows": len(swing_lows),
            "support_bounce_rate": round(support_bounce_rate, 1),
            "support_interactions": total_support,
            "resistance_bounce_rate": round(resistance_bounce_rate, 1),
            "resistance_interactions": total_resistance,
            "buy_call_at_support": {
                "avg_pnl": round(float(avg_support_pnl), 0),
                "win_rate": round(wr_support, 1),
                "trades": len(entry_pnl_support),
                "total_pnl": round(sum(entry_pnl_support), 0),
            },
            "buy_put_at_resistance": {
                "avg_pnl": round(float(avg_resistance_pnl), 0),
                "win_rate": round(wr_resistance, 1),
                "trades": len(entry_pnl_resistance),
                "total_pnl": round(sum(entry_pnl_resistance), 0),
            },
        }

    best_lb = max(results, key=lambda k: (
        results[k]["buy_call_at_support"]["total_pnl"] +
        results[k]["buy_put_at_resistance"]["total_pnl"]
    ))
    print(f"\n  BEST LOOKBACK: {best_lb}")

    return results, best_lb


# ═══════════════════════════════════════════════════════════════════════════
# METHOD B: PRICE CLUSTERING (CONSOLIDATION ZONES)
# ═══════════════════════════════════════════════════════════════════════════

def study_price_clustering(nifty_df):
    """Find price zones where market consolidates — these become S/R."""
    print("\n" + "=" * 70)
    print("METHOD B: PRICE CLUSTERING (CONSOLIDATION ZONES)")
    print("=" * 70)

    closes = nifty_df["close"].values
    bin_size = 50
    min_price = int(np.floor(closes.min() / bin_size) * bin_size)
    max_price = int(np.ceil(closes.max() / bin_size) * bin_size)

    bins = np.arange(min_price, max_price + bin_size, bin_size)
    hist, bin_edges = np.histogram(closes, bins=bins)

    zone_freq = []
    for i in range(len(hist)):
        zone_mid = (bin_edges[i] + bin_edges[i + 1]) / 2
        zone_freq.append({"zone_mid": zone_mid, "count": int(hist[i]),
                          "range": f"{int(bin_edges[i])}-{int(bin_edges[i+1])}"})

    zone_freq.sort(key=lambda x: x["count"], reverse=True)

    print("\n  Top Consolidation Zones (most time spent):")
    for z in zone_freq[:10]:
        print(f"    {z['range']}: {z['count']} days")

    high_freq_zones = [z for z in zone_freq if z["count"] >= 5]
    low_freq_zones = [z for z in zone_freq if 1 <= z["count"] < 5]

    high_freq_bounces = 0
    high_freq_breaks = 0
    low_freq_bounces = 0
    low_freq_breaks = 0
    hf_pnl = []
    lf_pnl = []

    for day_idx in range(20, len(nifty_df)):
        row = nifty_df.iloc[day_idx]
        day_low, day_high, day_close = row["low"], row["high"], row["close"]
        vix_val = row["vix"]

        for z in high_freq_zones:
            zm = z["zone_mid"]
            if abs(day_low - zm) / zm < SR_PROXIMITY_PCT:
                if day_close > zm:
                    high_freq_bounces += 1
                    hf_pnl.append(price_trade(zm, day_close, vix_val, "BUY_CALL"))
                else:
                    high_freq_breaks += 1

        for z in low_freq_zones[:10]:
            zm = z["zone_mid"]
            if abs(day_low - zm) / zm < SR_PROXIMITY_PCT:
                if day_close > zm:
                    low_freq_bounces += 1
                    lf_pnl.append(price_trade(zm, day_close, vix_val, "BUY_CALL"))
                else:
                    low_freq_breaks += 1

    hf_total = high_freq_bounces + high_freq_breaks
    lf_total = low_freq_bounces + low_freq_breaks
    hf_rate = (high_freq_bounces / hf_total * 100) if hf_total > 0 else 0
    lf_rate = (low_freq_bounces / lf_total * 100) if lf_total > 0 else 0

    print(f"\n  High-freq zone bounce rate: {hf_rate:.1f}% ({high_freq_bounces}/{hf_total})")
    print(f"  Low-freq zone bounce rate:  {lf_rate:.1f}% ({low_freq_bounces}/{lf_total})")
    print(f"  High-freq avg PnL: {np.mean(hf_pnl):.0f}" if hf_pnl else "  No high-freq trades")
    print(f"  Low-freq avg PnL:  {np.mean(lf_pnl):.0f}" if lf_pnl else "  No low-freq trades")

    verdict = "high_freq_better" if hf_rate > lf_rate else "low_freq_better"

    return {
        "top_zones": [{"range": z["range"], "count": z["count"]} for z in zone_freq[:10]],
        "high_freq_bounce_rate": round(hf_rate, 1),
        "low_freq_bounce_rate": round(lf_rate, 1),
        "high_freq_avg_pnl": round(float(np.mean(hf_pnl)), 0) if hf_pnl else 0,
        "low_freq_avg_pnl": round(float(np.mean(lf_pnl)), 0) if lf_pnl else 0,
        "high_freq_total_pnl": round(sum(hf_pnl), 0) if hf_pnl else 0,
        "low_freq_total_pnl": round(sum(lf_pnl), 0) if lf_pnl else 0,
        "verdict": verdict,
    }


# ═══════════════════════════════════════════════════════════════════════════
# METHOD C: ROUND NUMBER LEVELS
# ═══════════════════════════════════════════════════════════════════════════

def study_round_numbers(nifty_df):
    """Test round number S/R (psychological levels)."""
    print("\n" + "=" * 70)
    print("METHOD C: ROUND NUMBER LEVELS")
    print("=" * 70)

    data_min = nifty_df["low"].min()
    data_max = nifty_df["high"].max()

    round_500 = list(range(int(data_min // 500 * 500), int(data_max // 500 * 500) + 1000, 500))
    round_100 = list(range(int(data_min // 100 * 100), int(data_max // 100 * 100) + 200, 100))
    round_100_only = [r for r in round_100 if r not in round_500]

    print(f"\n  Testing 500-levels: {round_500}")
    print(f"  Testing 100-levels (non-500): {len(round_100_only)} levels")

    results = {}

    for label, levels in [("round_500", round_500), ("round_100", round_100_only)]:
        bounces = 0
        breaks = 0
        support_pnl = []
        resistance_pnl = []

        for day_idx in range(5, len(nifty_df)):
            row = nifty_df.iloc[day_idx]
            day_low, day_high, day_close = row["low"], row["high"], row["close"]
            vix_val = row["vix"]

            for level in levels:
                if abs(day_low - level) / level < SR_PROXIMITY_PCT and day_close > level:
                    bounces += 1
                    support_pnl.append(price_trade(level, day_close, vix_val, "BUY_CALL"))
                elif abs(day_low - level) / level < SR_PROXIMITY_PCT and day_close <= level:
                    breaks += 1

                if abs(day_high - level) / level < SR_PROXIMITY_PCT and day_close < level:
                    bounces += 1
                    resistance_pnl.append(price_trade(level, day_close, vix_val, "BUY_PUT"))
                elif abs(day_high - level) / level < SR_PROXIMITY_PCT and day_close >= level:
                    breaks += 1

        total = bounces + breaks
        bounce_rate = (bounces / total * 100) if total > 0 else 0
        avg_sup_pnl = np.mean(support_pnl) if support_pnl else 0
        avg_res_pnl = np.mean(resistance_pnl) if resistance_pnl else 0

        print(f"\n  {label}:")
        print(f"    Bounce rate: {bounce_rate:.1f}% ({bounces}/{total})")
        print(f"    Support BUY_CALL avg PnL: {avg_sup_pnl:.0f} ({len(support_pnl)} trades)")
        print(f"    Resistance BUY_PUT avg PnL: {avg_res_pnl:.0f} ({len(resistance_pnl)} trades)")

        results[label] = {
            "bounce_rate": round(bounce_rate, 1),
            "total_interactions": total,
            "support_avg_pnl": round(float(avg_sup_pnl), 0),
            "resistance_avg_pnl": round(float(avg_res_pnl), 0),
            "support_trades": len(support_pnl),
            "resistance_trades": len(resistance_pnl),
            "support_total_pnl": round(sum(support_pnl), 0) if support_pnl else 0,
            "resistance_total_pnl": round(sum(resistance_pnl), 0) if resistance_pnl else 0,
            "levels_tested": levels if label == "round_500" else len(levels),
        }

    return results


# ═══════════════════════════════════════════════════════════════════════════
# METHOD D: MOVING AVERAGE S/R
# ═══════════════════════════════════════════════════════════════════════════

def study_sma_sr(nifty_df):
    """Test SMAs as dynamic support/resistance."""
    print("\n" + "=" * 70)
    print("METHOD D: MOVING AVERAGE S/R")
    print("=" * 70)

    results = {}

    for sma_period in [20, 50]:
        col = f"sma{sma_period}"
        df_valid = nifty_df.dropna(subset=[col])

        bounces_support = 0
        breaks_support = 0
        bounces_resistance = 0
        breaks_resistance = 0
        support_pnl = []
        resistance_pnl = []

        for day_idx in range(1, len(df_valid)):
            row = df_valid.iloc[day_idx]
            prev_row = df_valid.iloc[day_idx - 1]
            sma_val = row[col]
            day_low, day_high, day_close = row["low"], row["high"], row["close"]
            prev_close = prev_row["close"]
            vix_val = row["vix"]

            if prev_close > sma_val:
                if abs(day_low - sma_val) / sma_val < SR_PROXIMITY_PCT:
                    if day_close > sma_val:
                        bounces_support += 1
                        support_pnl.append(price_trade(sma_val, day_close, vix_val, "BUY_CALL"))
                    else:
                        breaks_support += 1

            if prev_close < sma_val:
                if abs(day_high - sma_val) / sma_val < SR_PROXIMITY_PCT:
                    if day_close < sma_val:
                        bounces_resistance += 1
                        resistance_pnl.append(price_trade(sma_val, day_close, vix_val, "BUY_PUT"))
                    else:
                        breaks_resistance += 1

        total_sup = bounces_support + breaks_support
        total_res = bounces_resistance + breaks_resistance
        sup_rate = (bounces_support / total_sup * 100) if total_sup > 0 else 0
        res_rate = (bounces_resistance / total_res * 100) if total_res > 0 else 0

        avg_sup = np.mean(support_pnl) if support_pnl else 0
        avg_res = np.mean(resistance_pnl) if resistance_pnl else 0
        wr_sup = (sum(1 for p in support_pnl if p > 0) / len(support_pnl) * 100) if support_pnl else 0
        wr_res = (sum(1 for p in resistance_pnl if p > 0) / len(resistance_pnl) * 100) if resistance_pnl else 0

        print(f"\n  SMA{sma_period}:")
        print(f"    Support bounce rate:    {sup_rate:.1f}% ({bounces_support}/{total_sup})")
        print(f"    Resistance bounce rate: {res_rate:.1f}% ({bounces_resistance}/{total_res})")
        print(f"    BUY_CALL at SMA support: avg PnL={avg_sup:.0f}, WR={wr_sup:.1f}%, trades={len(support_pnl)}")
        print(f"    BUY_PUT at SMA resist:   avg PnL={avg_res:.0f}, WR={wr_res:.1f}%, trades={len(resistance_pnl)}")

        results[f"sma{sma_period}"] = {
            "support_bounce_rate": round(sup_rate, 1),
            "support_interactions": total_sup,
            "resistance_bounce_rate": round(res_rate, 1),
            "resistance_interactions": total_res,
            "buy_call_support": {
                "avg_pnl": round(float(avg_sup), 0),
                "win_rate": round(wr_sup, 1),
                "trades": len(support_pnl),
                "total_pnl": round(sum(support_pnl), 0),
            },
            "buy_put_resistance": {
                "avg_pnl": round(float(avg_res), 0),
                "win_rate": round(wr_res, 1),
                "trades": len(resistance_pnl),
                "total_pnl": round(sum(resistance_pnl), 0),
            },
        }

    return results


# ═══════════════════════════════════════════════════════════════════════════
# METHOD E: PREVIOUS DAY HIGH/LOW/CLOSE (PDH/PDL/PDC)
# ═══════════════════════════════════════════════════════════════════════════

def study_pdhl(nifty_df):
    """Test previous day high/low/close as S/R."""
    print("\n" + "=" * 70)
    print("METHOD E: PREVIOUS DAY HIGH/LOW/CLOSE")
    print("=" * 70)

    df = nifty_df.dropna(subset=["prev_high", "prev_low"])
    results = {}

    for level_name, level_col, sr_type in [
        ("PDH", "prev_high", "resistance"),
        ("PDL", "prev_low", "support"),
        ("PDC", "prev_close", "both"),
    ]:
        bounces = 0
        breaks = 0
        pnl_list = []

        for day_idx in range(len(df)):
            row = df.iloc[day_idx]
            level = row[level_col]
            day_low, day_high, day_close = row["low"], row["high"], row["close"]
            vix_val = row["vix"]

            if sr_type == "support" or sr_type == "both":
                if abs(day_low - level) / level < SR_PROXIMITY_PCT:
                    if day_close > level:
                        bounces += 1
                        pnl_list.append(price_trade(level, day_close, vix_val, "BUY_CALL"))
                    else:
                        breaks += 1

            if sr_type == "resistance" or sr_type == "both":
                if abs(day_high - level) / level < SR_PROXIMITY_PCT:
                    if day_close < level:
                        bounces += 1
                        pnl_list.append(price_trade(level, day_close, vix_val, "BUY_PUT"))
                    else:
                        breaks += 1

        total = bounces + breaks
        bounce_rate = (bounces / total * 100) if total > 0 else 0
        avg_pnl = np.mean(pnl_list) if pnl_list else 0
        wr = (sum(1 for p in pnl_list if p > 0) / len(pnl_list) * 100) if pnl_list else 0

        print(f"\n  {level_name} ({sr_type}):")
        print(f"    Bounce rate: {bounce_rate:.1f}% ({bounces}/{total})")
        print(f"    Avg PnL: {avg_pnl:.0f}, WR: {wr:.1f}%, trades: {len(pnl_list)}")
        print(f"    Total PnL: {sum(pnl_list):.0f}")

        results[level_name] = {
            "sr_type": sr_type,
            "bounce_rate": round(bounce_rate, 1),
            "total_interactions": total,
            "avg_pnl": round(float(avg_pnl), 0),
            "win_rate": round(wr, 1),
            "trades": len(pnl_list),
            "total_pnl": round(sum(pnl_list), 0),
        }

    return results


# ═══════════════════════════════════════════════════════════════════════════
# S/R-BASED EXIT STUDY
# ═══════════════════════════════════════════════════════════════════════════

def study_sr_exits(nifty_df):
    """
    Test S/R-based exits: exit at next resistance for calls, next support for puts.
    Compare with fixed trailing and EOD exit.
    """
    print("\n" + "=" * 70)
    print("S/R-BASED EXIT STRATEGIES")
    print("=" * 70)

    swing_highs, swing_lows = find_swing_points(nifty_df, lookback=10)

    exit_strategies = {
        "sr_target": {"BUY_CALL": [], "BUY_PUT": []},
        "trail_0.3pct": {"BUY_CALL": [], "BUY_PUT": []},
        "eod_exit": {"BUY_CALL": [], "BUY_PUT": []},
        "sr_trail_combo": {"BUY_CALL": [], "BUY_PUT": []},
    }

    for day_idx in range(20, len(nifty_df)):
        row = nifty_df.iloc[day_idx]
        vix_val = row["vix"]
        bars = simulate_intraday_bars(row)
        entry_bar = 3
        entry_spot = bars[entry_bar]

        active_supports = sorted([s["price"] for s in swing_lows if s["idx"] < day_idx], reverse=True)
        active_resistances = sorted([s["price"] for s in swing_highs if s["idx"] < day_idx])

        nearest_support = None
        nearest_resistance = None
        for s in active_supports:
            if s < entry_spot:
                nearest_support = s
                break
        for r in active_resistances:
            if r > entry_spot:
                nearest_resistance = r
                break

        if nearest_support is None:
            nearest_support = row["prev_low"]
        if nearest_resistance is None:
            nearest_resistance = row["prev_high"]

        for action in ["BUY_CALL", "BUY_PUT"]:
            if action == "BUY_CALL":
                target = nearest_resistance
                stop = nearest_support
            else:
                target = nearest_support
                stop = nearest_resistance

            # Strategy 1: S/R Target Exit
            sr_exit_spot = entry_spot
            for b in range(entry_bar + 1, TOTAL_BARS):
                if action == "BUY_CALL":
                    if bars[b] >= target:
                        sr_exit_spot = target
                        break
                    if bars[b] <= stop:
                        sr_exit_spot = stop
                        break
                else:
                    if bars[b] <= target:
                        sr_exit_spot = target
                        break
                    if bars[b] >= stop:
                        sr_exit_spot = stop
                        break
                sr_exit_spot = bars[b]

            pnl_sr = price_trade(entry_spot, sr_exit_spot, vix_val, action)
            exit_strategies["sr_target"][action].append(pnl_sr)

            # Strategy 2: 0.3% trailing stop
            trail_pct = 0.003
            best_spot = entry_spot
            trail_exit = entry_spot
            for b in range(entry_bar + 1, TOTAL_BARS):
                if action == "BUY_CALL":
                    if bars[b] > best_spot:
                        best_spot = bars[b]
                    trail_stop = best_spot * (1 - trail_pct)
                    if bars[b] <= trail_stop:
                        trail_exit = trail_stop
                        break
                else:
                    if bars[b] < best_spot:
                        best_spot = bars[b]
                    trail_stop = best_spot * (1 + trail_pct)
                    if bars[b] >= trail_stop:
                        trail_exit = trail_stop
                        break
                trail_exit = bars[b]

            pnl_trail = price_trade(entry_spot, trail_exit, vix_val, action)
            exit_strategies["trail_0.3pct"][action].append(pnl_trail)

            # Strategy 3: EOD exit
            pnl_eod = price_trade(entry_spot, bars[-1], vix_val, action)
            exit_strategies["eod_exit"][action].append(pnl_eod)

            # Strategy 4: S/R + Trail combo
            combo_exit = entry_spot
            target_hit = False
            combo_best = entry_spot
            for b in range(entry_bar + 1, TOTAL_BARS):
                if not target_hit:
                    if action == "BUY_CALL" and bars[b] >= target:
                        target_hit = True
                        combo_best = bars[b]
                    elif action == "BUY_PUT" and bars[b] <= target:
                        target_hit = True
                        combo_best = bars[b]
                    if action == "BUY_CALL" and bars[b] <= stop:
                        combo_exit = stop
                        break
                    elif action == "BUY_PUT" and bars[b] >= stop:
                        combo_exit = stop
                        break
                else:
                    if action == "BUY_CALL":
                        if bars[b] > combo_best:
                            combo_best = bars[b]
                        if bars[b] <= combo_best * (1 - trail_pct):
                            combo_exit = combo_best * (1 - trail_pct)
                            break
                    else:
                        if bars[b] < combo_best:
                            combo_best = bars[b]
                        if bars[b] >= combo_best * (1 + trail_pct):
                            combo_exit = combo_best * (1 + trail_pct)
                            break
                combo_exit = bars[b]

            pnl_combo = price_trade(entry_spot, combo_exit, vix_val, action)
            exit_strategies["sr_trail_combo"][action].append(pnl_combo)

    exit_summary = {}
    for strat_name, strat_data in exit_strategies.items():
        exit_summary[strat_name] = {}
        for action in ["BUY_CALL", "BUY_PUT"]:
            pnls = strat_data[action]
            total_pnl = sum(pnls)
            avg_pnl = np.mean(pnls)
            wr = sum(1 for p in pnls if p > 0) / len(pnls) * 100 if pnls else 0
            sharpe = (np.mean(pnls) / np.std(pnls)) * np.sqrt(252) if np.std(pnls) > 0 else 0

            exit_summary[strat_name][action] = {
                "total_pnl": round(total_pnl, 0),
                "avg_pnl": round(float(avg_pnl), 0),
                "win_rate": round(wr, 1),
                "sharpe": round(float(sharpe), 2),
                "trades": len(pnls),
            }
            print(f"\n  {strat_name} | {action}: total={total_pnl:.0f}, avg={avg_pnl:.0f}, WR={wr:.1f}%, Sharpe={sharpe:.2f}")

    best_exit = {}
    for action in ["BUY_CALL", "BUY_PUT"]:
        best_strat = max(exit_summary, key=lambda s: exit_summary[s][action]["sharpe"])
        best_exit[action] = {
            "strategy": best_strat,
            "sharpe": exit_summary[best_strat][action]["sharpe"],
            "win_rate": exit_summary[best_strat][action]["win_rate"],
            "avg_pnl": exit_summary[best_strat][action]["avg_pnl"],
        }
        print(f"\n  >>> BEST EXIT for {action}: {best_strat} (Sharpe={exit_summary[best_strat][action]['sharpe']:.2f})")

    return exit_summary, best_exit


# ═══════════════════════════════════════════════════════════════════════════
# S/R STRENGTH SCORING
# ═══════════════════════════════════════════════════════════════════════════

def compute_sr_strength(nifty_df):
    """
    Build S/R strength scoring model.
    Stronger S/R = level tested multiple times + never broken + recent + round number.
    """
    print("\n" + "=" * 70)
    print("S/R STRENGTH SCORING MODEL")
    print("=" * 70)

    swing_highs, swing_lows = find_swing_points(nifty_df, lookback=10)

    sr_levels = []
    for s in swing_lows:
        sr_levels.append({"price": s["price"], "type": "support", "idx": s["idx"], "date": s["date"]})
    for s in swing_highs:
        sr_levels.append({"price": s["price"], "type": "resistance", "idx": s["idx"], "date": s["date"]})

    clusters = []
    used = set()
    for i, level in enumerate(sr_levels):
        if i in used:
            continue
        cluster = [level]
        used.add(i)
        for j, other in enumerate(sr_levels):
            if j in used:
                continue
            if abs(level["price"] - other["price"]) <= 50:
                cluster.append(other)
                used.add(j)
        avg_price = np.mean([c["price"] for c in cluster])
        clusters.append({
            "price": round(float(avg_price), 0),
            "touches": len(cluster),
            "types": [c["type"] for c in cluster],
            "last_idx": max(c["idx"] for c in cluster),
            "is_round_500": (round(avg_price / 500) * 500 == round(avg_price / 50) * 50),
            "is_round_100": (round(avg_price / 100) * 100 == round(avg_price / 50) * 50),
        })

    total_days = len(nifty_df)
    for c in clusters:
        score = 0
        score += c["touches"] * 2.0
        recency = 1 - (total_days - c["last_idx"]) / total_days
        score += recency * 3.0
        if c["is_round_500"]:
            score += 2.0
        elif c["is_round_100"]:
            score += 1.0
        if "support" in c["types"] and "resistance" in c["types"]:
            score += 2.0
        c["strength_score"] = round(score, 1)

    clusters.sort(key=lambda x: x["strength_score"], reverse=True)

    print("\n  Top S/R Levels by Strength:")
    for c in clusters[:15]:
        type_str = "/".join(set(c["types"]))
        round_str = " [ROUND 500]" if c["is_round_500"] else (" [ROUND 100]" if c["is_round_100"] else "")
        print(f"    {c['price']:.0f}: score={c['strength_score']:.1f}, touches={c['touches']}, type={type_str}{round_str}")

    current_close = nifty_df["close"].iloc[-1]
    current_supports = sorted([c for c in clusters if c["price"] < current_close],
                             key=lambda x: x["price"], reverse=True)
    current_resistances = sorted([c for c in clusters if c["price"] > current_close],
                                key=lambda x: x["price"])

    print(f"\n  Current NIFTY: {current_close:.0f}")
    print(f"\n  Nearest Supports:")
    for s in current_supports[:5]:
        print(f"    {s['price']:.0f} (strength={s['strength_score']:.1f}, distance={current_close-s['price']:.0f} pts)")
    print(f"\n  Nearest Resistances:")
    for r in current_resistances[:5]:
        print(f"    {r['price']:.0f} (strength={r['strength_score']:.1f}, distance={r['price']-current_close:.0f} pts)")

    return {
        "all_levels": [{"price": c["price"], "strength": c["strength_score"],
                       "touches": c["touches"], "type": "/".join(set(c["types"]))}
                      for c in clusters[:20]],
        "current_supports": [{"price": s["price"], "strength": s["strength_score"],
                             "distance": round(current_close - s["price"], 0)}
                            for s in current_supports[:5]],
        "current_resistances": [{"price": r["price"], "strength": r["strength_score"],
                                "distance": round(r["price"] - current_close, 0)}
                               for r in current_resistances[:5]],
        "current_close": round(float(current_close), 0),
    }


# ═══════════════════════════════════════════════════════════════════════════
# S/R ENTRY SIGNAL QUALITY
# ═══════════════════════════════════════════════════════════════════════════

def study_sr_entry_quality(nifty_df):
    """Test: does entering ONLY when near S/R give better results than entering always?"""
    print("\n" + "=" * 70)
    print("S/R ENTRY SIGNAL QUALITY -- Near S/R vs. Random Entry")
    print("=" * 70)

    swing_highs, swing_lows = find_swing_points(nifty_df, lookback=10)

    always_pnl_call = []
    always_pnl_put = []
    near_support_pnl = []
    near_resistance_pnl = []

    for day_idx in range(20, len(nifty_df)):
        row = nifty_df.iloc[day_idx]
        vix_val = row["vix"]
        entry_spot = row["open"]
        exit_spot = row["close"]

        pnl_call = price_trade(entry_spot, exit_spot, vix_val, "BUY_CALL")
        pnl_put = price_trade(entry_spot, exit_spot, vix_val, "BUY_PUT")
        always_pnl_call.append(pnl_call)
        always_pnl_put.append(pnl_put)

        supports = sorted([s["price"] for s in swing_lows if s["idx"] < day_idx], reverse=True)
        resistances = sorted([s["price"] for s in swing_highs if s["idx"] < day_idx])

        near_sup = any(abs(entry_spot - s) / s < SR_PROXIMITY_PCT for s in supports[:5])
        if near_sup:
            near_support_pnl.append(pnl_call)

        near_res = any(abs(entry_spot - r) / r < SR_PROXIMITY_PCT for r in resistances[:5])
        if near_res:
            near_resistance_pnl.append(pnl_put)

    def _stats(pnls, label):
        if not pnls:
            print(f"  {label}: no trades")
            return {}
        avg = np.mean(pnls)
        wr = sum(1 for p in pnls if p > 0) / len(pnls) * 100
        sharpe = (np.mean(pnls) / np.std(pnls)) * np.sqrt(252) if np.std(pnls) > 0 else 0
        print(f"  {label}: avg={avg:.0f}, WR={wr:.1f}%, Sharpe={sharpe:.2f}, trades={len(pnls)}, total={sum(pnls):.0f}")
        return {
            "avg_pnl": round(float(avg), 0),
            "win_rate": round(wr, 1),
            "sharpe": round(float(sharpe), 2),
            "trades": len(pnls),
            "total_pnl": round(sum(pnls), 0),
        }

    print()
    r_always_call = _stats(always_pnl_call, "Always BUY_CALL")
    r_always_put = _stats(always_pnl_put, "Always BUY_PUT")
    r_near_sup = _stats(near_support_pnl, "BUY_CALL near support")
    r_near_res = _stats(near_resistance_pnl, "BUY_PUT near resistance")

    improvement_call = "YES" if (r_near_sup.get("sharpe", 0) > r_always_call.get("sharpe", 0)) else "NO"
    improvement_put = "YES" if (r_near_res.get("sharpe", 0) > r_always_put.get("sharpe", 0)) else "NO"

    print(f"\n  S/R filter improves BUY_CALL? {improvement_call}")
    print(f"  S/R filter improves BUY_PUT? {improvement_put}")

    return {
        "always_buy_call": r_always_call,
        "always_buy_put": r_always_put,
        "near_support_buy_call": r_near_sup,
        "near_resistance_buy_put": r_near_res,
        "sr_improves_call": improvement_call == "YES",
        "sr_improves_put": improvement_put == "YES",
    }


# ═══════════════════════════════════════════════════════════════════════════
# CURRENT NIFTY S/R MAP
# ═══════════════════════════════════════════════════════════════════════════

def build_current_sr_map(nifty_df):
    """Build current S/R map for Monday trading."""
    print("\n" + "=" * 70)
    print("CURRENT NIFTY S/R MAP (for Monday Apr 7 trading)")
    print("=" * 70)

    current_close = nifty_df["close"].iloc[-1]
    current_high = nifty_df["high"].iloc[-1]
    current_low = nifty_df["low"].iloc[-1]

    swing_highs, swing_lows = find_swing_points(nifty_df, lookback=10)
    swing_supports = sorted(set(round(s["price"] / 50) * 50 for s in swing_lows), reverse=True)
    swing_resistances = sorted(set(round(s["price"] / 50) * 50 for s in swing_highs))

    round_levels = list(range(21000, 27000, 500))

    sma20 = nifty_df["sma20"].iloc[-1] if not pd.isna(nifty_df["sma20"].iloc[-1]) else None
    sma50 = nifty_df["sma50"].iloc[-1] if not pd.isna(nifty_df["sma50"].iloc[-1]) else None

    pdh = current_high
    pdl = current_low
    pdc = current_close

    all_supports = set()
    all_resistances = set()

    for s in swing_supports:
        if s < current_close:
            all_supports.add(s)
    for r in swing_resistances:
        if r > current_close:
            all_resistances.add(r)
    for rl in round_levels:
        if rl < current_close - 100:
            all_supports.add(rl)
        elif rl > current_close + 100:
            all_resistances.add(rl)
    if sma20 and sma20 < current_close:
        all_supports.add(round(sma20 / 50) * 50)
    if sma50 and sma50 < current_close:
        all_supports.add(round(sma50 / 50) * 50)
    if sma20 and sma20 > current_close:
        all_resistances.add(round(sma20 / 50) * 50)
    if sma50 and sma50 > current_close:
        all_resistances.add(round(sma50 / 50) * 50)

    all_supports.add(round(pdl / 50) * 50)
    all_resistances.add(round(pdh / 50) * 50)

    sorted_supports = sorted(all_supports, reverse=True)[:7]
    sorted_resistances = sorted(all_resistances)[:7]

    print(f"\n  Current Close: {current_close:.0f}")
    print(f"  SMA20: {sma20:.0f}" if sma20 else "  SMA20: N/A")
    print(f"  SMA50: {sma50:.0f}" if sma50 else "  SMA50: N/A")
    print(f"  PDH: {pdh:.0f}, PDL: {pdl:.0f}, PDC: {pdc:.0f}")

    print(f"\n  KEY SUPPORTS (below {current_close:.0f}):")
    for s in sorted_supports:
        dist = current_close - s
        print(f"    {s:.0f}  ({dist:.0f} pts below, {dist/current_close*100:.2f}%)")

    print(f"\n  KEY RESISTANCES (above {current_close:.0f}):")
    for r in sorted_resistances:
        dist = r - current_close
        print(f"    {r:.0f}  ({dist:.0f} pts above, {dist/current_close*100:.2f}%)")

    immediate_support = sorted_supports[0] if sorted_supports else current_close - 200
    immediate_resistance = sorted_resistances[0] if sorted_resistances else current_close + 200
    strong_support = sorted_supports[1] if len(sorted_supports) > 1 else immediate_support - 200
    strong_resistance = sorted_resistances[1] if len(sorted_resistances) > 1 else immediate_resistance + 200

    monday_plan = {
        "current_close": round(float(current_close), 0),
        "pdh": round(float(pdh), 0),
        "pdl": round(float(pdl), 0),
        "pdc": round(float(pdc), 0),
        "sma20": round(float(sma20), 0) if sma20 else None,
        "sma50": round(float(sma50), 0) if sma50 else None,
        "immediate_support": round(float(immediate_support), 0),
        "immediate_resistance": round(float(immediate_resistance), 0),
        "strong_support": round(float(strong_support), 0),
        "strong_resistance": round(float(strong_resistance), 0),
        "all_supports": [round(float(s), 0) for s in sorted_supports],
        "all_resistances": [round(float(r), 0) for r in sorted_resistances],
        "buy_put_stop_loss": round(float(immediate_resistance), 0),
        "buy_put_target": round(float(immediate_support), 0),
        "buy_call_stop_loss": round(float(immediate_support), 0),
        "buy_call_target": round(float(immediate_resistance), 0),
    }

    print(f"\n  MONDAY TRADE LEVELS:")
    print(f"    BUY_PUT stop loss (resistance breach): {monday_plan['buy_put_stop_loss']}")
    print(f"    BUY_PUT target (support): {monday_plan['buy_put_target']}")
    print(f"    BUY_CALL stop loss (support breach): {monday_plan['buy_call_stop_loss']}")
    print(f"    BUY_CALL target (resistance): {monday_plan['buy_call_target']}")

    return monday_plan


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 70)
    print("DEEP SUPPORT/RESISTANCE STUDY FOR NIFTY")
    print("=" * 70)

    nifty_df = download_data()

    print("\n\nRunning 8 studies...\n")

    # Method A: Swing Points
    swing_results, best_swing_lb = study_swing_sr(nifty_df)

    # Method B: Price Clustering
    cluster_results = study_price_clustering(nifty_df)

    # Method C: Round Numbers
    round_results = study_round_numbers(nifty_df)

    # Method D: SMA S/R
    sma_results = study_sma_sr(nifty_df)

    # Method E: PDH/PDL
    pdhl_results = study_pdhl(nifty_df)

    # Exit study
    exit_results, best_exit = study_sr_exits(nifty_df)

    # S/R Strength
    strength_results = compute_sr_strength(nifty_df)

    # Entry Quality
    entry_quality = study_sr_entry_quality(nifty_df)

    # Current S/R Map
    current_map = build_current_sr_map(nifty_df)

    # ── Compile final ranking ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("METHOD RANKING -- Which S/R method is most profitable?")
    print("=" * 70)

    method_scores = {}

    # Score swing points
    best_sw = swing_results[best_swing_lb]
    sw_total = best_sw["buy_call_at_support"]["total_pnl"] + best_sw["buy_put_at_resistance"]["total_pnl"]
    sw_wr = (best_sw["buy_call_at_support"]["win_rate"] + best_sw["buy_put_at_resistance"]["win_rate"]) / 2
    method_scores["swing_points"] = {"total_pnl": sw_total, "avg_wr": round(sw_wr, 1), "best_lookback": best_swing_lb}

    # Score PDH/PDL
    pdhl_total = sum(pdhl_results[k]["total_pnl"] for k in pdhl_results)
    pdhl_wr = np.mean([pdhl_results[k]["win_rate"] for k in pdhl_results])
    method_scores["pdh_pdl_pdc"] = {"total_pnl": pdhl_total, "avg_wr": round(float(pdhl_wr), 1)}

    # Score SMA
    sma_total = sum(sma_results[k]["buy_call_support"]["total_pnl"] + sma_results[k]["buy_put_resistance"]["total_pnl"]
                   for k in sma_results)
    sma_wr = np.mean([
        sma_results[k]["buy_call_support"]["win_rate"]
        for k in sma_results
    ] + [
        sma_results[k]["buy_put_resistance"]["win_rate"]
        for k in sma_results
    ])
    method_scores["sma_sr"] = {"total_pnl": sma_total, "avg_wr": round(float(sma_wr), 1)}

    # Score round numbers
    rn_total = sum(round_results[k].get("support_total_pnl", 0) + round_results[k].get("resistance_total_pnl", 0)
                   for k in round_results)
    rn_wr = np.mean([round_results[k]["bounce_rate"] for k in round_results])
    method_scores["round_numbers"] = {"total_pnl": rn_total, "avg_wr": round(float(rn_wr), 1)}

    # Score clustering
    cl_total = cluster_results.get("high_freq_total_pnl", 0) + cluster_results.get("low_freq_total_pnl", 0)
    cl_wr = (cluster_results["high_freq_bounce_rate"] + cluster_results["low_freq_bounce_rate"]) / 2
    method_scores["price_clustering"] = {"total_pnl": cl_total, "avg_wr": round(cl_wr, 1)}

    ranked = sorted(method_scores.items(), key=lambda x: x[1]["total_pnl"], reverse=True)
    print()
    for rank, (method, scores) in enumerate(ranked, 1):
        print(f"  #{rank} {method}: total PnL={scores['total_pnl']:.0f}, avg WR={scores['avg_wr']:.1f}%")

    best_method = ranked[0][0]
    print(f"\n  >>> BEST S/R METHOD: {best_method}")

    # ── Save to JSON ──────────────────────────────────────────────────────

    sr_rules = {
        "_description": "S/R rules learned from 6 months of NIFTY data (Oct 2025 - Apr 2026)",
        "_best_method": best_method,
        "_backtest_period": "Oct 2025 - Apr 2026",

        "method_ranking": {m: s for m, s in ranked},

        "swing_points": swing_results,
        "price_clustering": cluster_results,
        "round_numbers": round_results,
        "sma_sr": sma_results,
        "pdh_pdl": pdhl_results,

        "exit_strategies": exit_results,
        "best_exit_per_action": best_exit,

        "sr_strength": strength_results,
        "entry_quality": entry_quality,

        "current_sr_map": current_map,

        "rules_for_agent": {
            "_description": "Actionable rules for the ensemble agent",
            "sr_detection": {
                "primary_method": "swing_points_lookback_10",
                "secondary_method": "pdh_pdl",
                "tertiary_method": "round_numbers_500",
                "lookback_bars": 10,
                "proximity_pct": SR_PROXIMITY_PCT,
            },
            "entry_rules": {
                "near_support": {
                    "action": "BUY_CALL",
                    "weight": 1.0,
                    "condition": "spot within 0.3% of nearest support",
                },
                "near_resistance": {
                    "action": "BUY_PUT",
                    "weight": 1.0,
                    "condition": "spot within 0.3% of nearest resistance",
                },
            },
            "exit_rules": {
                "BUY_CALL": {
                    "target": "nearest resistance above entry",
                    "stop_loss": "support breach (close below nearest support)",
                    "trailing": "switch to 0.3% trail after target hit",
                },
                "BUY_PUT": {
                    "target": "nearest support below entry",
                    "stop_loss": "resistance breach (close above nearest resistance)",
                    "trailing": "switch to 0.3% trail after target hit",
                },
            },
            "strength_scoring": {
                "touches_weight": 2.0,
                "recency_weight": 3.0,
                "round_500_bonus": 2.0,
                "round_100_bonus": 1.0,
                "dual_type_bonus": 2.0,
                "description": "Higher score = more reliable S/R level",
            },
            "monday_trade_levels": current_map,
        },
    }

    output_path = PROJECT_ROOT / "data" / "sr_rules.json"
    with open(output_path, "w") as f:
        json.dump(sr_rules, f, indent=2, default=str)

    print(f"\n\nResults saved to: {output_path}")
    print("=" * 70)
    print("S/R STUDY COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
