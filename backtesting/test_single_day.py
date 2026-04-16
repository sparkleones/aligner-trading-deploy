"""
SINGLE DAY TEST — April 6, 2026 (Monday)
Tests V14 model on yesterday's real data to compare with live trading results.
"""
import sys
import datetime as dt
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backtesting.daywise_analysis import add_all_indicators
from backtesting.multi_month_oos_test import download_range
from backtesting.oos_june2024_test import (
    V9_HYBRID_CONFIG, CAPITAL, simulate_day, detect_market_regime,
    get_dynamic_lots,
)
from backtesting.paper_trading_real_data import bs_premium, get_strike_and_type

TARGET_DATE = dt.date(2026, 4, 6)
TEST_CAPITAL = 30_000  # User's actual capital


def run_day_test(cfg, label, nifty_ind, daily, vix_lookup, day_groups,
                 close_prices, all_dates, equity):
    """Run model on a single day and print detailed trades."""

    target_idx = None
    for i, d in enumerate(all_dates):
        if d == TARGET_DATE:
            target_idx = i
            break

    if target_idx is None:
        print(f"  {label}: Date {TARGET_DATE} not found in data!")
        return

    day_bars = day_groups[TARGET_DATE]
    vix = vix_lookup.get(TARGET_DATE, 14.0)

    row = daily.iloc[target_idx]
    prev_ohlc = None
    if target_idx > 0:
        prev_d = all_dates[target_idx - 1]
        if prev_d in day_groups:
            pb = day_groups[prev_d]
            prev_ohlc = {
                "open": pb["open"].iloc[0], "high": pb["high"].max(),
                "low": pb["low"].min(), "close": pb["close"].iloc[-1],
            }

    above_sma50 = bool(row.get("AboveSMA50", True))
    above_sma20 = bool(row.get("AboveSMA20", True))
    rsi = float(row.get("RSI", 50))
    prev_change = float(row.get("PrevChange%", 0))
    vix_spike = bool(row.get("VIXSpike", False))
    sma20 = float(row.get("SMA20", row["Close"]))
    sma50 = float(row.get("SMA50", row["Close"]))
    ema9 = float(row.get("EMA9", row["Close"]))
    ema21 = float(row.get("EMA21", row["Close"]))
    weekly_sma = float(row.get("WeeklySMA", row["Close"]))
    gap_pct = float(row.get("GapPct", 0))

    # Regime detection
    regime_info = detect_market_regime(daily, target_idx)

    # DTE (assume Thursday expiry)
    days_to_thu = (3 - TARGET_DATE.weekday()) % 7
    if days_to_thu == 0:
        days_to_thu = 7
    dte = days_to_thu
    is_expiry = (TARGET_DATE.weekday() == 3)

    # Determine daily trend
    if ema9 > ema21 and row["Close"] > sma20:
        daily_trend = "bullish"
    elif ema9 < ema21 and row["Close"] < sma20:
        daily_trend = "bearish"
    else:
        daily_trend = "neutral"

    trades = simulate_day(
        cfg, day_bars, TARGET_DATE, prev_ohlc, vix, daily_trend,
        dte, is_expiry, daily, target_idx, close_prices,
        above_sma50, above_sma20, rsi, prev_change, vix_spike,
        sma20, sma50, ema9, ema21, weekly_sma, gap_pct,
        equity=equity, recent_wr=0.5, recent_trades=0,
        regime_info=regime_info,
    )

    print(f"\n  {'='*110}")
    print(f"  {label}")
    print(f"  {'='*110}")
    print(f"  Date: {TARGET_DATE} ({TARGET_DATE.strftime('%A')})")
    print(f"  NIFTY: Open={day_bars['open'].iloc[0]:.0f} High={day_bars['high'].max():.0f} "
          f"Low={day_bars['low'].min():.0f} Close={day_bars['close'].iloc[-1]:.0f}")
    print(f"  VIX: {vix:.2f} | Regime: {regime_info['regime']} | Daily trend: {daily_trend}")
    print(f"  Capital: Rs {equity:,} | DTE: {dte} | Expiry: {is_expiry}")
    print(f"  SMA20: {sma20:.0f} | SMA50: {sma50:.0f} | RSI: {rsi:.1f} | Gap: {gap_pct:+.2f}%")

    # Check VWAP and Squeeze at key minutes
    highs = day_bars["high"].values
    lows = day_bars["low"].values
    closes = day_bars["close"].values
    tp = (highs + lows + closes) / 3.0
    vwap_arr = np.cumsum(tp) / np.arange(1, len(tp) + 1, dtype=float)

    # Squeeze check
    has_squeeze_cols = all(c in day_bars.columns for c in ["bb_upper", "bb_lower", "ema21", "atr"])
    squeeze_pcts = []
    if has_squeeze_cols:
        bb_up = day_bars["bb_upper"].values
        bb_lo = day_bars["bb_lower"].values
        ema21_v = day_bars["ema21"].values
        atr_v = day_bars["atr"].values
        kc_upper = ema21_v + 1.5 * atr_v
        kc_lower = ema21_v - 1.5 * atr_v
        squeeze = (bb_lo > kc_lower) & (bb_up < kc_upper)
        squeeze_pcts = [f"min{m}: {'SQZ' if squeeze[m] else 'open'}" for m in [30, 60, 120, 180, 240, 300]]

    print(f"  VWAP samples: min30={vwap_arr[30]:.0f} min60={vwap_arr[60]:.0f} "
          f"min120={vwap_arr[120]:.0f} min180={vwap_arr[180]:.0f}")
    if squeeze_pcts:
        print(f"  Squeeze: {' | '.join(squeeze_pcts)}")

    if not trades:
        avoid = cfg.get("avoid_days", [])
        if TARGET_DATE.strftime("%A") in avoid:
            print(f"\n  >>> ZERO TRADES — Monday is in avoid_days filter <<<")
            print(f"  >>> Model says: DON'T TRADE on {TARGET_DATE.strftime('%A')} <<<")
        else:
            print(f"\n  >>> ZERO TRADES — No valid signals passed filters <<<")
        return trades

    total_pnl = 0
    print(f"\n  {'#':>3} {'Action':>9} {'Entry Time':>11} {'Exit Time':>10} {'Type':<20} "
          f"{'Spot In':>8} {'Spot Out':>8} {'Strike':>7} {'Prem In':>8} {'Prem Out':>8} "
          f"{'Lots':>4} {'Held':>5} {'Exit Reason':<12} {'P&L':>10}")
    print(f"  {'-'*135}")

    for i, t in enumerate(trades):
        entry_h = 9 + (15 + t["entry_minute"]) // 60
        entry_m = (15 + t["entry_minute"]) % 60
        exit_h = 9 + (15 + t["exit_minute"]) // 60
        exit_m = (15 + t["exit_minute"]) % 60

        print(f"  {i+1:>3} {t['action']:>9} {entry_h:02d}:{entry_m:02d}      "
              f"{exit_h:02d}:{exit_m:02d}     {t['entry_type']:<20} "
              f"{t['entry_spot']:>8.0f} {t['exit_spot']:>8.0f} {t['strike']:>7} "
              f"{t['entry_prem']:>8.1f} {t['exit_prem']:>8.1f} "
              f"{t['lots']:>4} {t['minutes_held']:>4}m {t['exit_reason']:<12} "
              f"Rs{t['pnl']:>+9,.0f}")
        total_pnl += t["pnl"]

    wins = sum(1 for t in trades if t["pnl"] > 0)
    losses = len(trades) - wins
    print(f"  {'-'*135}")
    print(f"  TOTAL: {len(trades)} trades ({wins}W/{losses}L) | Day P&L: Rs{total_pnl:>+,}")
    print(f"  Capital after: Rs {equity + total_pnl:,} ({total_pnl/equity*100:+.1f}%)")

    return trades


def main():
    print("=" * 110)
    print(f"  SINGLE DAY BACKTEST — {TARGET_DATE} ({TARGET_DATE.strftime('%A')})")
    print(f"  Testing all model variants on yesterday's REAL data")
    print(f"  Capital: Rs {TEST_CAPITAL:,} (user's actual)")
    print("=" * 110)

    # Load data
    nifty, vix = download_range("2026-03-01", "2026-04-07")
    if nifty is None:
        print("No data!")
        return

    vix_lookup = {}
    if vix is not None:
        for idx, row in vix.iterrows():
            vix_lookup[idx.date()] = row["close"]

    nifty_ind = add_all_indicators(nifty.copy())
    day_groups = {d: g for d, g in nifty_ind.groupby(nifty_ind.index.date)}
    all_dates = sorted(day_groups.keys())

    daily_rows = []
    for d in all_dates:
        bars = day_groups[d]
        daily_rows.append({
            "Date": d, "Open": bars["open"].iloc[0], "High": bars["high"].max(),
            "Low": bars["low"].min(), "Close": bars["close"].iloc[-1],
            "VIX": vix_lookup.get(d, 14.0),
        })
    daily = pd.DataFrame(daily_rows).set_index("Date")
    daily.index = pd.to_datetime(daily.index)

    for idx_date in daily.index:
        daily.loc[idx_date, "VIX"] = vix_lookup.get(idx_date.date(), 14.0)
    daily["VIX"] = daily["VIX"].ffill().bfill().fillna(14.0)
    daily["PrevVIX"] = daily["VIX"].shift(1).fillna(daily["VIX"].iloc[0])
    daily["Change%"] = daily["Close"].pct_change() * 100
    daily["PrevChange%"] = daily["Change%"].shift(1).fillna(0)
    daily["SMA50"] = daily["Close"].rolling(50, min_periods=1).mean()
    daily["SMA20"] = daily["Close"].rolling(20, min_periods=1).mean()
    daily["AboveSMA50"] = daily["Close"] > daily["SMA50"]
    daily["AboveSMA20"] = daily["Close"] > daily["SMA20"]
    daily["EMA9"] = daily["Close"].ewm(span=9).mean()
    daily["EMA21"] = daily["Close"].ewm(span=21).mean()
    daily["WeeklySMA"] = daily["Close"].rolling(5).mean().rolling(4, min_periods=1).mean()
    delta = daily["Close"].diff()
    gain = delta.clip(lower=0).rolling(14, min_periods=1).mean()
    loss_s = (-delta.clip(upper=0)).rolling(14, min_periods=1).mean()
    daily["RSI"] = 100 - (100 / (1 + gain / loss_s.replace(0, 0.001)))
    daily["VIXSpike"] = daily["VIX"] > daily["PrevVIX"] * 1.15
    daily["GapPct"] = (daily["Open"] - daily["Close"].shift(1)) / daily["Close"].shift(1) * 100

    close_prices = daily["Close"].values.tolist()

    # =====================================================================
    # TEST 1: V14 model AS-IS (Monday blocked)
    # =====================================================================
    cfg1 = V9_HYBRID_CONFIG.copy()
    run_day_test(cfg1, "TEST 1: V14 Model (PRODUCTION CONFIG — Monday in avoid_days)",
                 nifty_ind, daily, vix_lookup, day_groups, close_prices, all_dates,
                 equity=TEST_CAPITAL)

    # =====================================================================
    # TEST 2: V14 model WITHOUT Monday filter
    # =====================================================================
    cfg2 = V9_HYBRID_CONFIG.copy()
    cfg2["avoid_days"] = ["Wednesday"]  # Remove Monday from avoid list
    run_day_test(cfg2, "TEST 2: V14 Model (Monday ALLOWED)",
                 nifty_ind, daily, vix_lookup, day_groups, close_prices, all_dates,
                 equity=TEST_CAPITAL)

    # =====================================================================
    # TEST 3: V14 WITHOUT confluence filters (V13b equivalent)
    # =====================================================================
    cfg3 = V9_HYBRID_CONFIG.copy()
    cfg3["avoid_days"] = ["Wednesday"]
    cfg3["use_vwap_filter"] = False
    cfg3["use_squeeze_filter"] = False
    cfg3["use_rsi_hard_gate"] = False
    run_day_test(cfg3, "TEST 3: V13b Model (NO confluence filters, Monday allowed)",
                 nifty_ind, daily, vix_lookup, day_groups, close_prices, all_dates,
                 equity=TEST_CAPITAL)

    # =====================================================================
    # TEST 4: V14 with Monday allowed + Rs 2L capital (model's intended scale)
    # =====================================================================
    cfg4 = V9_HYBRID_CONFIG.copy()
    cfg4["avoid_days"] = ["Wednesday"]
    run_day_test(cfg4, "TEST 4: V14 Model (Monday allowed, Rs 2L capital — model's scale)",
                 nifty_ind, daily, vix_lookup, day_groups, close_prices, all_dates,
                 equity=200_000)

    # =====================================================================
    # TEST 5: ALL filters OFF, no day restrictions (raw signals)
    # =====================================================================
    cfg5 = V9_HYBRID_CONFIG.copy()
    cfg5["avoid_days"] = []
    cfg5["use_vwap_filter"] = False
    cfg5["use_squeeze_filter"] = False
    cfg5["use_rsi_hard_gate"] = False
    cfg5["use_regime_detection"] = False
    cfg5["min_confidence_filter"] = 0
    cfg5["block_late_entries"] = 999
    cfg5["block_call_4th_hour"] = False
    cfg5["avoid_windows"] = []
    run_day_test(cfg5, "TEST 5: RAW SIGNALS (all filters OFF — what the market offered)",
                 nifty_ind, daily, vix_lookup, day_groups, close_prices, all_dates,
                 equity=TEST_CAPITAL)

    print("\n" + "=" * 110)
    print("  CONCLUSION")
    print("=" * 110)
    print(f"  Your live loss: Rs -2,500 on Rs 30,000 capital (-8.3%)")
    print(f"  V14 production config: Monday is BLOCKED (avoid_days filter)")
    print(f"  Historical data shows Monday is the worst day (23.4% WR in 2024)")
    print(f"  The model's recommendation: DO NOT TRADE on Mondays")
    print("=" * 110)


if __name__ == "__main__":
    main()
