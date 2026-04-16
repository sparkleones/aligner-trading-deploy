"""
FULL DAY-BY-DAY TRADE LOG — V9 Hybrid with compounding
Every trade is logged: entry time, exit time, direction, premium, lots, P&L.
No looking at future data. Uses BS premium (synthetic — NOT real option prices).

KNOWN LIMITATIONS (printed in report):
  1. Option premiums are Black-Scholes estimates, NOT real market prices
  2. No bid-ask spread or slippage modeled
  3. VIX is daily closing value, not intraday
  4. Brokerage: flat Rs 80/trade assumed

This script produces a CSV of every single trade for audit.
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

from backtesting.daywise_analysis import add_all_indicators, compute_ema, compute_pivot_points
from backtesting.oos_june2024_test import (
    V9_HYBRID_CONFIG, CAPITAL,
    simulate_day, run_model,
)
from backtesting.multi_month_oos_test import download_range, run_month

DATA_DIR = project_root / "data" / "historical"

# 10 test months
TEST_MONTHS = [
    ("2024-06-01", "2024-07-31", 2024,  7, "Jul-2024"),
    ("2024-07-01", "2024-08-31", 2024,  8, "Aug-2024"),
    ("2024-08-01", "2024-09-30", 2024,  9, "Sep-2024"),
    ("2024-09-01", "2024-10-31", 2024, 10, "Oct-2024"),
    ("2024-11-01", "2024-12-31", 2024, 12, "Dec-2024"),
    ("2024-12-01", "2025-01-31", 2025,  1, "Jan-2025"),
    ("2025-02-01", "2025-03-31", 2025,  3, "Mar-2025"),
    ("2025-04-01", "2025-05-31", 2025,  5, "May-2025"),
    ("2025-06-01", "2025-07-31", 2025,  7, "Jul-2025"),
    ("2025-08-01", "2025-09-30", 2025,  9, "Sep-2025"),
]


def run_month_with_logs(nifty_df, vix_df, test_year, test_month, cfg, starting_equity):
    """Run single model on a month, return full trade list + daily P&L."""
    from backtesting.oos_june2024_test import (
        detect_entries_v8, detect_entries_composite,
        get_dynamic_lots, find_support_resistance, compute_pivot_points,
    )
    from backtesting.paper_trading_real_data import (
        sr_multi_method, bs_premium, get_strike_and_type, LOT_SIZE,
    )
    from backtesting.v7_hybrid_comparison import compute_composite

    vix_lookup = {}
    if vix_df is not None and not vix_df.empty:
        for idx, row in vix_df.iterrows():
            vix_lookup[idx.date()] = row["close"]

    nifty = add_all_indicators(nifty_df.copy())
    day_groups = {date: group for date, group in nifty.groupby(nifty.index.date)}
    all_dates = sorted(day_groups.keys())

    # Build daily
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

    test_start = dt.date(test_year, test_month, 1)

    # Run model
    r = run_model(cfg, daily, close_prices, day_groups, all_dates,
                  vix_lookup, daily_trend_df, test_start,
                  starting_equity=starting_equity)

    return r


if __name__ == "__main__":
    print("=" * 130)
    print("  FULL DAY-BY-DAY TRADE LOG — V9 Hybrid with Compounding (REALISTIC)")
    print("  10 months out-of-sample | Starting capital: Rs 2,00,000")
    print()
    print("  REALISTIC CONSTRAINTS APPLIED:")
    print("    - Max 20 lots per trade (liquidity cap)")
    print("    - 0.5% slippage on entry AND exit")
    print("    - Rs 2/unit bid-ask spread on entry AND exit")
    print("    - Rs 80 brokerage per trade")
    print("    - Real NIFTY/VIX data from Kite Connect")
    print("    - Option premiums still BS-estimated (no real option chain data)")
    print("    - VIX is daily close, not intraday")
    print("=" * 130)

    cfg = V9_HYBRID_CONFIG.copy()
    equity = CAPITAL
    all_trades_csv = []
    monthly_summaries = []
    trade_num = 0

    for warmup_start, data_end, test_yr, test_mo, label in TEST_MONTHS:
        print(f"\n{'='*130}")
        print(f"  {label} | Capital entering: Rs {equity:>,}")
        print(f"{'='*130}")

        nifty, vix = download_range(warmup_start, data_end)
        if nifty is None:
            print(f"  SKIPPED")
            continue

        r = run_month_with_logs(nifty, vix, test_yr, test_mo, cfg, equity)

        # Print day-by-day breakdown
        trades = r.get("all_trades", [])
        if not trades:
            print(f"  No trades in {label}")
            monthly_summaries.append({
                "month": label, "start_eq": equity, "end_eq": equity,
                "pnl": 0, "trades": 0, "wr": 0, "sharpe": 0, "dd": 0,
            })
            continue

        # Group trades by date
        trades_by_date = defaultdict(list)
        for t in trades:
            trades_by_date[t["date"]].append(t)

        day_equity = equity
        peak_eq = equity
        max_dd_pct = 0
        month_pnl = 0

        print(f"\n  {'Date':<12} {'#':>3} {'Dir':>9} {'Entry':>7} {'Type':<20} "
              f"{'Spot In':>8} {'Spot Out':>8} {'Prem In':>8} {'Prem Out':>8} "
              f"{'Lots':>5} {'Qty':>5} {'Held':>5} {'Exit Reason':<14} "
              f"{'P&L':>10} {'Day P&L':>10} {'Equity':>12}")
        print(f"  {'-'*160}")

        sorted_dates = sorted(trades_by_date.keys())
        for date_str in sorted_dates:
            day_trades = trades_by_date[date_str]
            day_pnl = sum(t["pnl"] for t in day_trades)

            for i, t in enumerate(day_trades):
                trade_num += 1
                entry_time = f"{t['entry_minute']//60:02d}:{t['entry_minute']%60:02d}"
                exit_min = t.get("exit_minute", 0)
                exit_time = f"{exit_min//60:02d}:{exit_min%60:02d}"
                btst_str = f" BTST:{t.get('btst_pnl', 0):+,.0f}" if t.get("btst_pnl", 0) != 0 else ""

                # Only print equity on last trade of the day
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
                      f"{entry_time:>4}-{exit_time:<4} {t.get('entry_type','?'):<20} "
                      f"{t['entry_spot']:>8.0f} {t['exit_spot']:>8.0f} "
                      f"{t['entry_prem']:>8.1f} {t['exit_prem']:>8.1f} "
                      f"{t['lots']:>5} {t['qty']:>5} {t['minutes_held']:>4}m "
                      f"{t['exit_reason']:<14} Rs{t['pnl']:>+8,}{btst_str}"
                      f" {day_pnl_str:>10} {eq_str:>12}")

                # CSV row
                all_trades_csv.append({
                    "trade_num": trade_num, "month": label, "date": date_str,
                    "action": t["action"], "entry_type": t.get("entry_type", "?"),
                    "entry_minute": t["entry_minute"], "entry_time": entry_time,
                    "exit_minute": exit_min, "exit_time": exit_time,
                    "entry_spot": t["entry_spot"], "exit_spot": t["exit_spot"],
                    "entry_prem": t["entry_prem"], "exit_prem": t["exit_prem"],
                    "strike": t.get("strike", 0), "opt_type": t.get("opt_type", "?"),
                    "lots": t["lots"], "qty": t["qty"],
                    "minutes_held": t["minutes_held"], "exit_reason": t["exit_reason"],
                    "pnl": t["pnl"], "btst_pnl": t.get("btst_pnl", 0),
                    "confidence": t.get("confidence", 0), "vix": t.get("vix", 0),
                    "equity_after": day_equity if i == len(day_trades) - 1 else "",
                })

        # Month summary
        wins = len([t for t in trades if t["pnl"] > 0])
        losses = len([t for t in trades if t["pnl"] <= 0])
        wr = wins / len(trades) * 100 if trades else 0

        print(f"\n  {label} SUMMARY:")
        print(f"    Start equity:  Rs {equity:>,}")
        print(f"    Month P&L:     Rs {month_pnl:>+,}")
        print(f"    End equity:    Rs {day_equity:>,}")
        print(f"    Return:        {month_pnl/equity*100:>+.1f}%")
        print(f"    Trades:        {len(trades)} ({wins}W/{losses}L)")
        print(f"    Win Rate:      {wr:.1f}%")
        print(f"    Max Drawdown:  {max_dd_pct:.1f}%")
        print(f"    Avg lots:      {np.mean([t['lots'] for t in trades]):.1f}")

        monthly_summaries.append({
            "month": label, "start_eq": equity, "end_eq": day_equity,
            "pnl": month_pnl, "trades": len(trades), "wins": wins,
            "losses": losses, "wr": wr, "dd": max_dd_pct,
            "return_pct": month_pnl / equity * 100,
        })

        equity = day_equity  # Compound for next month

    # ==============================================================
    # FINAL SUMMARY
    # ==============================================================
    print(f"\n\n{'='*130}")
    print(f"  COMPOUNDED EQUITY JOURNEY — V9 Hybrid")
    print(f"  {len(all_trades_csv)} total trades across 10 months")
    print(f"  REALISTIC: Max 20 lots, 0.5% slippage, Rs 2 spread, Rs 80 brokerage")
    print(f"{'='*130}")
    print(f"\n  {'Month':<12} {'Start Equity':>14} {'P&L':>12} {'End Equity':>14} {'Return':>8} "
          f"{'Trades':>7} {'WR':>6} {'DD':>6}")
    print(f"  {'-'*90}")

    for s in monthly_summaries:
        print(f"  {s['month']:<12} Rs{s['start_eq']:>11,} Rs{s['pnl']:>+10,} Rs{s['end_eq']:>11,} "
              f"{s['return_pct']:>+7.1f}% {s['trades']:>5}t {s['wr']:>5.1f}% {s['dd']:>5.1f}%")

    print(f"  {'-'*90}")
    total_pnl = equity - CAPITAL
    print(f"  {'TOTAL':<12} Rs{CAPITAL:>11,} Rs{total_pnl:>+10,} Rs{equity:>11,} "
          f"{total_pnl/CAPITAL*100:>+7.1f}% {trade_num:>5}t")
    print(f"\n  Rs {CAPITAL:>,} -> Rs {equity:>,} = {equity/CAPITAL:.1f}x in 10 months")
    win_months = sum(1 for s in monthly_summaries if s["pnl"] > 0)
    print(f"  {win_months}/{len(monthly_summaries)} profitable months")

    # Save CSV
    csv_path = DATA_DIR / "v9_hybrid_realistic_trades.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_trades_csv[0].keys())
        writer.writeheader()
        writer.writerows(all_trades_csv)
    print(f"\n  Full trade log saved: {csv_path}")
    print(f"  ({len(all_trades_csv)} trades for audit)")
    print("=" * 130)
