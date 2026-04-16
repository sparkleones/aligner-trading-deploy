"""
REAL OPTION PRICE BACKTEST — V9 Hybrid with Kaggle Data

Uses REAL 1-min option prices from Kaggle dataset for 2024 months.
Falls back to BS estimates + slippage for 2025 months (no Kaggle data).

DATA SOURCES:
  ✅ REAL: NIFTY spot prices (1-min from Kite Connect)
  ✅ REAL: VIX values (daily from Kite Connect)
  ✅ REAL: Option premiums for 2024 (1-min from Kaggle — actual traded prices)
  🔶 BS estimate: Option premiums for 2025 (with 0.5% slippage + Rs 2 spread)
  ✅ REAL: Lot size 75 (exchange standard since Jul 2024)
  ✅ REAL: Brokerage Rs 40/trade (Rs 20/leg, Zerodha flat fee)

CONSTRAINTS:
  - Max 20 lots per trade (realistic liquidity cap)
  - Buy at candle close+high avg (simulates buying at ask)
  - Sell at candle close+low avg (simulates selling at bid)
"""

import sys
import datetime as dt
import csv
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backtesting.daywise_analysis import add_all_indicators
from backtesting.oos_june2024_test import (
    V9_HYBRID_CONFIG, CAPITAL, run_model, get_dynamic_lots,
)
from backtesting.multi_month_oos_test import download_range

DATA_DIR = project_root / "data" / "historical"
KAGGLE_DIR = project_root / "data" / "kaggle" / "2024"
KAGGLE_V2_DIR = project_root / "data" / "kaggle" / "nifty_data" / "nifty_options" / "2024"

# Lot size: 25 for Jan-Jun 2024 (old SEBI rule), 75 from Jul 2024 onwards
def get_lot_size(year, month):
    if year == 2024 and month < 7:
        return 25  # Pre-July 2024
    return 75      # Jul 2024 onwards


TEST_MONTHS = [
    # === Jan-Jun 2024 — NEW real data from Kaggle v2 dataset (lot size = 25) ===
    ("2023-12-01", "2024-01-31", 2024,  1, "Jan-2024"),
    ("2024-01-01", "2024-02-29", 2024,  2, "Feb-2024"),
    ("2024-02-01", "2024-03-31", 2024,  3, "Mar-2024"),
    ("2024-03-01", "2024-04-30", 2024,  4, "Apr-2024"),
    ("2024-04-01", "2024-05-31", 2024,  5, "May-2024"),
    ("2024-05-01", "2024-06-30", 2024,  6, "Jun-2024"),
    # === Jul-Dec 2024 — existing real data (lot size = 75) ===
    ("2024-06-01", "2024-07-31", 2024,  7, "Jul-2024"),
    ("2024-07-01", "2024-08-31", 2024,  8, "Aug-2024"),
    ("2024-08-01", "2024-09-30", 2024,  9, "Sep-2024"),
    ("2024-09-01", "2024-10-31", 2024, 10, "Oct-2024"),
    ("2024-10-01", "2024-11-30", 2024, 11, "Nov-2024"),
    ("2024-11-01", "2024-12-31", 2024, 12, "Dec-2024"),
]

# Months with Kaggle real option data
# V1 format (data/kaggle/2024/{MONTH}/NIFTY-{EXP}-{DATE}.csv)
KAGGLE_V1_MONTHS = {7: "2024JUL", 8: "2024AUG", 9: "2024SEP", 10: "2024OCT",
                    11: "2024NOV", 12: "2024DEC"}
# V2 format (data/kaggle/nifty_data/nifty_options/2024/{month_num}/nifty_options_{DD}_{MM}_{YYYY}.csv)
KAGGLE_V2_MONTHS = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
# Combined: any month we have real data for
KAGGLE_MONTHS = KAGGLE_V1_MONTHS  # Kept for backward compat


def get_weekly_expiry(trade_date):
    """Get Thursday expiry for a trade date."""
    d = trade_date
    days_ahead = 3 - d.weekday()  # Thursday = 3
    if days_ahead < 0:
        days_ahead += 7
    return d + dt.timedelta(days=days_ahead)


def format_date_for_kaggle(d):
    """Convert date to Kaggle filename format: 01JUL24"""
    return d.strftime("%d%b%y").upper()


def load_kaggle_option_file(trade_date, expiry_date):
    """Load 1-min option data from Kaggle for a specific trade date and expiry.

    File naming: NIFTY-{EXPIRY}-{TRADEDATE}.csv
    Example: NIFTY-04JUL24-01JUL24.csv

    Returns: DataFrame indexed by (datetime, strike_price, right) or None
    """
    # Determine which month folder to look in
    month_key = f"2024{trade_date.strftime('%b').upper()}"
    month_dir = KAGGLE_DIR / month_key

    if not month_dir.exists():
        return None

    expiry_str = format_date_for_kaggle(expiry_date)
    trade_str = format_date_for_kaggle(trade_date)
    filename = f"NIFTY-{expiry_str}-{trade_str}.csv"
    filepath = month_dir / filename

    if not filepath.exists():
        # Try other expiry files in the folder (maybe monthly expiry?)
        # Search for any file with this trade date
        alt_files = list(month_dir.glob(f"NIFTY-*-{trade_str}.csv"))
        if alt_files:
            # Pick the one with the nearest expiry
            best = None
            best_diff = 999
            for af in alt_files:
                parts = af.stem.split("-")
                try:
                    exp_str = parts[1]
                    exp_dt = dt.datetime.strptime(exp_str, "%d%b%y").date()
                    diff = abs((exp_dt - trade_date).days)
                    if diff < best_diff and exp_dt >= trade_date:
                        best_diff = diff
                        best = af
                except:
                    continue
            if best:
                filepath = best
            else:
                return None
        else:
            return None

    try:
        df = pd.read_csv(filepath)
        return df
    except Exception as e:
        print(f"    Error reading {filepath.name}: {e}")
        return None


import re

def load_kaggle_v2_option_file(trade_date):
    """Load 1-min option data from Kaggle V2 dataset (nifty_data/nifty_options).

    File naming: nifty_options_{DD}_{MM}_{YYYY}.csv
    Columns: date, time, symbol, open, high, low, close, oi, volume
    Symbol: NIFTY{DDMON}{YY}{STRIKE}{CE/PE} e.g. NIFTY04JUL2424250CE

    Returns: DataFrame with columns [time_str, strike_price, right, open, high, low, close, volume]
             or None if file not found
    """
    month_num = trade_date.month
    month_dir = KAGGLE_V2_DIR / str(month_num)
    if not month_dir.exists():
        return None

    filename = f"nifty_options_{trade_date.day:02d}_{trade_date.month:02d}_{trade_date.year}.csv"
    filepath = month_dir / filename
    if not filepath.exists():
        return None

    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        print(f"    Error reading {filepath.name}: {e}")
        return None

    if df.empty:
        return None

    # Parse symbol column: NIFTY{DDMON}{YY}{STRIKE}{CE/PE}
    # Examples: NIFTY04JUL2424250CE, NIFTY04JUL2420150PE
    # Pattern: NIFTY + expiry(DDMONYY) + strike(digits) + type(CE/PE)
    symbol_pattern = re.compile(r'^NIFTY\d{2}[A-Z]{3}\d{2}(\d+)(CE|PE)$')

    strikes = []
    rights = []
    for sym in df["symbol"]:
        m = symbol_pattern.match(sym)
        if m:
            strikes.append(int(m.group(1)))
            rights.append(m.group(2))
        else:
            strikes.append(0)
            rights.append("")

    df["strike_price"] = strikes
    df["right"] = rights

    # Convert time to HH:MM format for matching
    df["datetime"] = df["time"].str[:5]  # "09:15:00" -> "09:15"

    # Filter out unparseable rows
    df = df[df["strike_price"] > 0].copy()

    return df


def get_real_option_premium(kaggle_df, strike, opt_type, minute_of_day, is_buy=True):
    """Get real option premium from Kaggle 1-min data.

    Args:
        kaggle_df: DataFrame with columns [datetime, strike_price, right, open, high, low, close, volume]
        strike: Strike price (e.g., 24100)
        opt_type: 'CE' or 'PE'
        minute_of_day: Minutes from 9:15 (0 = 9:15 AM)
        is_buy: True=entry (pay ask), False=exit (receive bid)

    Returns: Premium or None
    """
    if kaggle_df is None:
        return None

    # Filter for this strike and type
    mask = (kaggle_df["strike_price"] == strike) & (kaggle_df["right"] == opt_type)
    opt_bars = kaggle_df[mask]

    if opt_bars.empty:
        # Try nearest strike (within 50 points)
        nearby = kaggle_df[
            (abs(kaggle_df["strike_price"] - strike) <= 50) &
            (kaggle_df["right"] == opt_type)
        ]
        if nearby.empty:
            return None
        # Use the closest strike
        closest_strike = nearby.iloc[(nearby["strike_price"] - strike).abs().argsort().iloc[0]]["strike_price"]
        opt_bars = kaggle_df[(kaggle_df["strike_price"] == closest_strike) & (kaggle_df["right"] == opt_type)]

    # Convert minute_of_day to time string (HH:MM)
    target_hour = 9 + (15 + minute_of_day) // 60
    target_min = (15 + minute_of_day) % 60
    target_time = f"{target_hour:02d}:{target_min:02d}"

    # Find the bar
    bar = opt_bars[opt_bars["datetime"] == target_time]

    if bar.empty:
        # Try +/- 1 minute
        for offset in [1, -1, 2, -2]:
            adj_min = minute_of_day + offset
            adj_hour = 9 + (15 + adj_min) // 60
            adj_m = (15 + adj_min) % 60
            adj_time = f"{adj_hour:02d}:{adj_m:02d}"
            bar = opt_bars[opt_bars["datetime"] == adj_time]
            if not bar.empty:
                break

    if bar.empty:
        return None

    bar = bar.iloc[0]

    # Use candle data to model bid-ask
    if is_buy:
        # Buyer pays closer to ask: average of close and high
        premium = (bar["close"] + bar["high"]) / 2
    else:
        # Seller receives closer to bid: average of close and low
        premium = (bar["close"] + bar["low"]) / 2

    return max(0.05, float(premium))


def run_backtest():
    """Run complete backtest with real option prices where available."""

    print("=" * 130)
    print("  REAL OPTION PRICE BACKTEST — V11 Hybrid with Compounding")
    print(f"  {len(TEST_MONTHS)} months | Rs 2,00,000 starting capital | Dynamic lot sizing")
    print()
    print("  DATA SOURCES:")
    print("    Jan-Jun 2024: REAL 1-min option prices from Kaggle V2 (lot size=25)")
    print("    Jul-Dec 2024: REAL 1-min option prices from Kaggle V1 (lot size=75)")
    print("    NIFTY spot + VIX: Real data from Kite Connect (all months)")
    print("    Brokerage: Rs 40/trade (Rs 20/leg Zerodha flat fee)")
    print("=" * 130)

    cfg = V9_HYBRID_CONFIG.copy()
    equity = CAPITAL
    all_trades_csv = []
    monthly_summaries = []
    trade_num = 0

    for warmup_start, data_end, test_yr, test_mo, label in TEST_MONTHS:
        print(f"\n{'='*130}")
        print(f"  {label} | Capital entering: Rs {equity:>,}")

        # Check if we have Kaggle data for this month (V1 or V2 format)
        has_kaggle_v1 = (test_yr == 2024 and test_mo in KAGGLE_V1_MONTHS)
        has_kaggle_v2 = (test_yr == 2024 and test_mo in KAGGLE_V2_MONTHS)
        has_kaggle = has_kaggle_v1 or has_kaggle_v2
        if has_kaggle:
            src_label = "V1+V2" if has_kaggle_v1 else "V2"
            print(f"  >>> REAL OPTION PRICES from Kaggle dataset ({src_label}) <<<")
        else:
            print(f"  >>> BS-estimated premiums (no Kaggle data for {test_yr}) <<<")
        print(f"{'='*130}")

        nifty, vix = download_range(warmup_start, data_end)
        if nifty is None:
            print(f"  SKIPPED")
            monthly_summaries.append({
                "month": label, "start_eq": equity, "end_eq": equity,
                "pnl": 0, "trades": 0, "wins": 0, "losses": 0,
                "wr": 0, "dd": 0, "return_pct": 0,
            })
            continue

        # Run phase 1: get trade signals from the model
        vix_lookup = {}
        if vix is not None and not vix.empty:
            for idx, row in vix.iterrows():
                vix_lookup[idx.date()] = row["close"]

        nifty_ind = add_all_indicators(nifty.copy())
        day_groups = {date: group for date, group in nifty_ind.groupby(nifty_ind.index.date)}
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
        daily_trend_df = daily[["Close", "SMA20", "EMA9", "EMA21"]].rename(
            columns={"Close": "close", "SMA20": "sma20", "EMA9": "ema9", "EMA21": "ema21"})
        daily_trend_df.index = daily_trend_df.index.date

        test_start = dt.date(test_yr, test_mo, 1)

        # Run model to get trade signals (uses BS premiums internally)
        r = run_model(cfg, daily, close_prices, day_groups, all_dates,
                      vix_lookup, daily_trend_df, test_start,
                      starting_equity=equity)

        trades = r.get("all_trades", [])
        if not trades:
            print(f"  No trades in {label}")
            monthly_summaries.append({
                "month": label, "start_eq": equity, "end_eq": equity,
                "pnl": 0, "trades": 0, "wins": 0, "losses": 0,
                "wr": 0, "dd": 0, "return_pct": 0,
            })
            continue

        # Phase 2: Replace BS premiums with real Kaggle prices (for 2024)
        trades_by_date = defaultdict(list)
        for t in trades:
            trades_by_date[t["date"]].append(t)

        day_equity = equity
        peak_eq = equity
        max_dd_pct = 0
        month_pnl = 0
        real_count = 0
        bs_count = 0

        print(f"\n  {'Date':<12} {'#':>3} {'Dir':>9} {'Entry':>7} {'Type':<18} "
              f"{'Spot In':>8} {'Spot Out':>8} {'Prem In':>8} {'Prem Out':>8} "
              f"{'Lots':>4} {'Qty':>5} {'Held':>5} {'Exit':<12} "
              f"{'P&L':>10} {'Src':>4} {'Day P&L':>10} {'Equity':>12}")
        print(f"  {'-'*165}")

        sorted_dates = sorted(trades_by_date.keys())
        for date_str in sorted_dates:
            day_trades = trades_by_date[date_str]
            trade_date = dt.date.fromisoformat(date_str)
            expiry = get_weekly_expiry(trade_date)

            # Load Kaggle data for this day (try V1 first, then V2)
            kaggle_df = None
            if has_kaggle:
                if has_kaggle_v1:
                    kaggle_df = load_kaggle_option_file(trade_date, expiry)
                if kaggle_df is None and has_kaggle_v2:
                    kaggle_df = load_kaggle_v2_option_file(trade_date)

            day_pnl = 0
            for i, t in enumerate(day_trades):
                trade_num += 1
                strike = t["strike"]
                opt_type = t["opt_type"]

                # Recalculate lots with CURRENT equity (compounding)
                lot_size = get_lot_size(test_yr, test_mo)
                lots = get_dynamic_lots(
                    t["vix"], day_equity,
                    confidence=t.get("confidence", 0.5),
                    zero_hero=t.get("is_zero_hero", False),
                    recent_wr=0.5, recent_trades=trade_num,
                )
                qty = lots * lot_size

                # Try real option prices from Kaggle
                entry_real = get_real_option_premium(
                    kaggle_df, strike, opt_type, t["entry_minute"], is_buy=True)
                exit_real = get_real_option_premium(
                    kaggle_df, strike, opt_type, t.get("exit_minute", 0), is_buy=False)

                if entry_real is not None and exit_real is not None:
                    entry_prem = entry_real
                    exit_prem = exit_real
                    src = "REAL"
                    real_count += 1
                else:
                    # Fallback: BS with slippage (already in the trade from simulate_day)
                    entry_prem = t["entry_prem"]
                    exit_prem = t["exit_prem"]
                    src = "BS"
                    bs_count += 1

                # P&L
                pnl = (exit_prem - entry_prem) * qty - 40  # Rs 20/leg x 2 legs
                pnl = round(pnl, 0)

                # BTST
                btst_pnl = 0
                orig_btst = t.get("btst_pnl", 0)
                if orig_btst != 0 and t["qty"] > 0:
                    btst_pnl = round(orig_btst * (qty / t["qty"]), 0)
                    pnl += btst_pnl

                day_pnl += pnl

                entry_time = f"{t['entry_minute']//60:02d}:{t['entry_minute']%60:02d}"
                exit_min = t.get("exit_minute", 0)
                exit_time = f"{exit_min//60:02d}:{exit_min%60:02d}"
                btst_str = f" BTST:{btst_pnl:+,.0f}" if btst_pnl != 0 else ""

                if i == len(day_trades) - 1:
                    day_equity += day_pnl
                    month_pnl += day_pnl
                    if day_equity > peak_eq:
                        peak_eq = day_equity
                    dd = (peak_eq - day_equity) / peak_eq * 100 if peak_eq > 0 else 0
                    if dd > max_dd_pct:
                        max_dd_pct = dd
                    eq_str = f"Rs{day_equity:>10,}"
                    day_pnl_str = f"Rs{day_pnl:>+8,}"
                else:
                    eq_str = ""
                    day_pnl_str = ""

                print(f"  {date_str:<12} {trade_num:>3} {t['action']:>9} "
                      f"{entry_time:>4}-{exit_time:<4} {t.get('entry_type','?'):<18} "
                      f"{t['entry_spot']:>8.0f} {t['exit_spot']:>8.0f} "
                      f"{entry_prem:>8.1f} {exit_prem:>8.1f} "
                      f"{lots:>4} {qty:>5} {t['minutes_held']:>4}m "
                      f"{t['exit_reason']:<12} Rs{pnl:>+8,}{btst_str}"
                      f" {src:>4} {day_pnl_str:>10} {eq_str:>12}")

                all_trades_csv.append({
                    "trade_num": trade_num, "month": label, "date": date_str,
                    "action": t["action"], "entry_type": t.get("entry_type", "?"),
                    "entry_minute": t["entry_minute"], "entry_time": entry_time,
                    "exit_minute": exit_min, "exit_time": exit_time,
                    "entry_spot": t["entry_spot"], "exit_spot": t["exit_spot"],
                    "entry_prem": round(entry_prem, 2), "exit_prem": round(exit_prem, 2),
                    "strike": strike, "opt_type": opt_type,
                    "lots": lots, "qty": qty,
                    "minutes_held": t["minutes_held"], "exit_reason": t["exit_reason"],
                    "pnl": pnl, "btst_pnl": btst_pnl,
                    "price_source": src,
                    "confidence": t.get("confidence", 0), "vix": t.get("vix", 0),
                    "equity_after": day_equity if i == len(day_trades) - 1 else "",
                })

        # Month summary
        month_trades_list = [t for t in all_trades_csv if t["month"] == label]
        wins = len([t for t in month_trades_list if t["pnl"] > 0])
        losses = len([t for t in month_trades_list if t["pnl"] <= 0])
        total = wins + losses
        wr = wins / total * 100 if total else 0

        real_pct = real_count / (real_count + bs_count) * 100 if (real_count + bs_count) > 0 else 0

        print(f"\n  {label} SUMMARY:")
        print(f"    Start equity:  Rs {equity:>,}")
        print(f"    Month P&L:     Rs {month_pnl:>+,}")
        print(f"    End equity:    Rs {day_equity:>,}")
        print(f"    Return:        {month_pnl/equity*100:>+.1f}%")
        print(f"    Trades:        {total} ({wins}W/{losses}L)")
        print(f"    Win Rate:      {wr:.1f}%")
        print(f"    Max Drawdown:  {max_dd_pct:.1f}%")
        print(f"    Price source:  {real_count} REAL / {bs_count} BS ({real_pct:.0f}% real)")

        monthly_summaries.append({
            "month": label, "start_eq": equity, "end_eq": day_equity,
            "pnl": month_pnl, "trades": total, "wins": wins,
            "losses": losses, "wr": wr, "dd": max_dd_pct,
            "return_pct": month_pnl / equity * 100 if equity > 0 else 0,
            "real_pct": real_pct,
        })

        equity = day_equity

    # ==============================================================
    # FINAL SUMMARY
    # ==============================================================
    total_real = sum(1 for t in all_trades_csv if t["price_source"] == "REAL")
    total_bs = sum(1 for t in all_trades_csv if t["price_source"] == "BS")

    print(f"\n\n{'='*130}")
    print(f"  REAL OPTION PRICE BACKTEST — V11 Hybrid")
    print(f"  {len(all_trades_csv)} total trades across {len(TEST_MONTHS)} months")
    if (total_real + total_bs) > 0:
        print(f"  Real Kaggle prices: {total_real} trades | BS fallback: {total_bs} trades "
              f"({total_real/(total_real+total_bs)*100:.0f}% real)")
    print(f"  Dynamic lots | Lot size 25 (pre-Jul24) / 75 (Jul24+) | Rs 40 brokerage")
    print(f"{'='*130}")
    print(f"\n  {'Month':<12} {'Start Equity':>14} {'P&L':>12} {'End Equity':>14} {'Return':>8} "
          f"{'Trades':>7} {'WR':>6} {'DD':>6} {'Real%':>6}")
    print(f"  {'-'*100}")

    for s in monthly_summaries:
        print(f"  {s['month']:<12} Rs{s['start_eq']:>11,} Rs{s['pnl']:>+10,} Rs{s['end_eq']:>11,} "
              f"{s['return_pct']:>+7.1f}% {s['trades']:>5}t {s['wr']:>5.1f}% {s['dd']:>5.1f}% "
              f"{s.get('real_pct', 0):>5.0f}%")

    print(f"  {'-'*100}")
    total_pnl = equity - CAPITAL
    print(f"  {'TOTAL':<12} Rs{CAPITAL:>11,} Rs{total_pnl:>+10,} Rs{equity:>11,} "
          f"{total_pnl/CAPITAL*100:>+7.1f}% {trade_num:>5}t")
    print(f"\n  Rs {CAPITAL:>,} -> Rs {equity:>,} = {equity/CAPITAL:.1f}x in {len(TEST_MONTHS)} months")
    win_months = sum(1 for s in monthly_summaries if s["pnl"] > 0)
    print(f"  {win_months}/{len(monthly_summaries)} profitable months")

    # Save CSV
    csv_path = DATA_DIR / "v9_hybrid_real_option_trades.csv"
    if all_trades_csv:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_trades_csv[0].keys())
            writer.writeheader()
            writer.writerows(all_trades_csv)
        print(f"\n  Full trade log saved: {csv_path}")
        print(f"  ({len(all_trades_csv)} trades for audit)")
    print("=" * 130)


if __name__ == "__main__":
    run_backtest()
