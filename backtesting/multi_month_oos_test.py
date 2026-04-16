"""
MULTI-MONTH OUT-OF-SAMPLE TEST — V6_3T vs V8 vs V9_Hybrid
Tests on 10 different months from different market regimes (all unseen).

Training data:  Oct 2025 - Apr 2026
Test months:    Jul 2024 - Sep 2025 (10 months, all out-of-sample)

Each month uses the previous month as warmup for indicators (SMA50, etc).
"""

import sys
import time
import datetime as dt
import copy
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backtesting.paper_trading_real_data import (
    sr_multi_method, bs_premium, get_strike_and_type, LOT_SIZE,
)
from backtesting.v7_hybrid_comparison import compute_composite
from backtesting.daywise_analysis import (
    add_all_indicators, compute_ema, compute_pivot_points, find_support_resistance,
)
from backtesting.oos_june2024_test import (
    V6_3T_CONFIG, V8_CONFIG, V9_HYBRID_CONFIG, CAPITAL,
    get_dynamic_lots, detect_entries_composite, detect_entries_v8,
    simulate_day, run_model,
)

DATA_DIR = project_root / "data" / "historical"


# ==============================================================
# DOWNLOAD
# ==============================================================

def download_range(start_date, end_date):
    """Download 1-min NIFTY + daily VIX from Kite Connect."""
    nifty_path = DATA_DIR / f"nifty_min_{start_date}_{end_date}.csv"
    vix_path = DATA_DIR / f"vix_min_{start_date}_{end_date}.csv"

    if nifty_path.exists() and vix_path.exists():
        nifty = pd.read_csv(nifty_path, parse_dates=["timestamp"], index_col="timestamp")
        vix = pd.read_csv(vix_path, parse_dates=["timestamp"], index_col="timestamp")
        print(f"  Loaded cached: {nifty_path.name} ({len(nifty)} bars, {len(set(nifty.index.date))} days)")
        return nifty, vix

    print(f"  Downloading {start_date} to {end_date} from Kite Connect...")
    from config.settings import load_settings
    from broker.kite_connect import KiteConnectBroker

    settings = load_settings()
    broker = KiteConnectBroker(
        api_key=settings.broker.api_key,
        api_secret=settings.broker.api_secret,
        user_id=settings.broker.user_id,
        password=settings.broker.password,
        totp_secret=settings.broker.totp_secret,
    )
    if not broker.authenticate():
        print("  ERROR: Kite authentication failed!")
        return None, None
    print("  Authenticated.")

    start = dt.datetime.strptime(start_date, "%Y-%m-%d")
    end = dt.datetime.strptime(end_date, "%Y-%m-%d")

    all_nifty = []
    all_vix = []
    current = start
    chunk_days = 5
    total_days = (end - start).days
    chunks_needed = max(1, total_days // chunk_days + 1)
    chunk_num = 0

    while current < end:
        chunk_end = min(current + dt.timedelta(days=chunk_days), end)
        chunk_num += 1
        pct = chunk_num / chunks_needed * 100

        # NIFTY
        try:
            bars = broker.get_historical_data(
                symbol="NIFTY 50", from_dt=current, to_dt=chunk_end, interval="minute")
            all_nifty.extend(bars)
            print(f"    [{chunk_num}/{chunks_needed}] {current.date()} -> {chunk_end.date()}: {len(bars)} bars ({pct:.0f}%)", flush=True)
        except Exception as e:
            print(f"    [{chunk_num}/{chunks_needed}] NIFTY FAILED: {e}")
        time.sleep(0.4)

        # VIX
        try:
            vbars = broker.get_historical_data(
                symbol="INDIA VIX", from_dt=current, to_dt=chunk_end, interval="day")
            all_vix.extend(vbars)
        except Exception:
            pass
        time.sleep(0.4)

        current = chunk_end + dt.timedelta(days=1)

    if not all_nifty:
        print("  ERROR: No data downloaded!")
        return None, None

    nifty_df = pd.DataFrame(all_nifty)
    nifty_df["timestamp"] = pd.to_datetime(nifty_df["time"])
    nifty_df = nifty_df.set_index("timestamp").sort_index()
    nifty_df = nifty_df[~nifty_df.index.duplicated(keep="first")]

    vix_df = pd.DataFrame(all_vix) if all_vix else pd.DataFrame()
    if not vix_df.empty:
        vix_df["timestamp"] = pd.to_datetime(vix_df["time"])
        vix_df = vix_df.set_index("timestamp").sort_index()
        vix_df = vix_df[~vix_df.index.duplicated(keep="first")]

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    nifty_df.to_csv(nifty_path)
    if not vix_df.empty:
        vix_df.to_csv(vix_path)

    print(f"  Saved: {len(nifty_df)} NIFTY bars, {len(vix_df)} VIX bars")
    return nifty_df, vix_df


# ==============================================================
# BUILD DAILY + RUN MODEL FOR A SINGLE MONTH
# ==============================================================

def run_month(nifty_df, vix_df, test_year, test_month, configs,
              starting_equities=None):
    """Run all model configs on a single test month.

    nifty_df should include warmup data from previous month.
    starting_equities: dict {model_name: equity} for compounding.
    Returns dict: {model_name: results_dict}
    """
    if starting_equities is None:
        starting_equities = {}
    # VIX lookup
    vix_lookup = {}
    if vix_df is not None and not vix_df.empty:
        for idx, row in vix_df.iterrows():
            vix_lookup[idx.date()] = row["close"]

    # Add indicators
    nifty = add_all_indicators(nifty_df.copy())

    # Group by date
    day_groups = {date: group for date, group in nifty.groupby(nifty.index.date)}
    all_dates = sorted(day_groups.keys())

    # Build daily OHLC
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

    # Enrich daily
    for idx_date in daily.index:
        vix_val = vix_lookup.get(idx_date.date(), 14.0)
        daily.loc[idx_date, "VIX"] = vix_val
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
    daily["PrevHigh"] = daily["High"].shift(1)
    daily["PrevLow"] = daily["Low"].shift(1)
    daily["VIXSpike"] = daily["VIX"] > daily["PrevVIX"] * 1.15
    daily["GapPct"] = (daily["Open"] - daily["Close"].shift(1)) / daily["Close"].shift(1) * 100

    close_prices = daily["Close"].values.tolist()
    daily_trend_df = daily[["Close", "SMA20", "EMA9", "EMA21"]].rename(
        columns={"Close": "close", "SMA20": "sma20", "EMA9": "ema9", "EMA21": "ema21"})
    daily_trend_df.index = daily_trend_df.index.date

    # Test start = first day of test month
    test_start = dt.date(test_year, test_month, 1)
    test_dates = [d for d in all_dates if d >= test_start]
    if not test_dates:
        return {}

    # Detect expiry day (Thursday for NIFTY)
    # In 2024 it was Thursday, from Nov 2024 onwards it shifted — handle both
    results = {}
    for cfg in configs:
        eq = starting_equities.get(cfg["name"], CAPITAL)
        r = run_model(cfg, daily, close_prices, day_groups, test_dates,
                      vix_lookup, daily_trend_df, test_start,
                      starting_equity=eq)
        results[cfg["name"]] = r

    return results, len(test_dates)


# ==============================================================
# MAIN
# ==============================================================

if __name__ == "__main__":
    print("=" * 130)
    print("  MULTI-MONTH OUT-OF-SAMPLE TEST")
    print("  V6_3T vs V8 vs V9_Hybrid across 10 different months")
    print("  Training: Oct 2025 - Apr 2026 | All test months are UNSEEN")
    print("  Capital: Rs 2,00,000 | Dynamic lot sizing | ATR-based SR stop")
    print("=" * 130)

    # 10 test months — diverse market conditions
    # Each entry: (warmup_start, data_end, test_year, test_month, label)
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

    configs = [V6_3T_CONFIG, V8_CONFIG, V9_HYBRID_CONFIG]
    model_names = [c["name"] for c in configs]

    # Aggregate results
    all_month_results = {}
    month_summaries = []

    # COMPOUNDING: Track accumulated equity per model
    accumulated_equity = {name: CAPITAL for name in model_names}
    print(f"\n  Starting capital: Rs {CAPITAL:>,} per model (compounding across months)")

    for warmup_start, data_end, test_yr, test_mo, label in TEST_MONTHS:
        print(f"\n{'='*80}")
        print(f"  {label} (warmup from {warmup_start})")
        print(f"  Capital entering month: " + " | ".join(
            f"{n}: Rs {accumulated_equity[n]:>,}" for n in model_names))
        print(f"{'='*80}")

        # Download data (cached after first download)
        nifty, vix = download_range(warmup_start, data_end)
        if nifty is None:
            print(f"  SKIPPED: No data for {label}")
            continue

        # Run models with accumulated equity
        try:
            result, n_days = run_month(nifty, vix, test_yr, test_mo, configs,
                                       starting_equities=accumulated_equity)
        except Exception as e:
            print(f"  ERROR running {label}: {e}")
            import traceback
            traceback.print_exc()
            continue

        if not result:
            print(f"  SKIPPED: No test days for {label}")
            continue

        # Update accumulated equity for next month
        for name in model_names:
            if name in result:
                accumulated_equity[name] = result[name]["final_equity"]

        # Print month summary
        print(f"\n  {'Model':<16} {'Start Cap':>12} {'P&L':>12} {'Return':>8} {'End Cap':>12} "
              f"{'Trades':>7} {'WR':>6} {'Sharpe':>7} {'DD':>7} {'Lots':>5}")
        print(f"  {'-'*100}")
        for name in model_names:
            if name in result:
                r = result[name]
                print(f"  {name:<16} Rs{r['start_equity']:>9,} Rs{r['net_pnl']:>+9,} {r['return_pct']:>+7.1f}% "
                      f"Rs{r['final_equity']:>9,} {r['total_trades']:>5}t "
                      f"{r['win_rate']:>5.1f}% {r['sharpe']:>6.2f} {r['max_drawdown']:>6.1f}% {r['avg_lots']:>4.1f}")

        all_month_results[label] = result
        summary = {"label": label, "days": n_days}
        for name in model_names:
            if name in result:
                r = result[name]
                summary[f"{name}_pnl"] = r["net_pnl"]
                summary[f"{name}_start_eq"] = r["start_equity"]
                summary[f"{name}_end_eq"] = r["final_equity"]
                summary[f"{name}_sharpe"] = r["sharpe"]
                summary[f"{name}_dd"] = r["max_drawdown"]
                summary[f"{name}_wr"] = r["win_rate"]
                summary[f"{name}_trades"] = r["total_trades"]
                summary[f"{name}_pf"] = r["profit_factor"]
        month_summaries.append(summary)

    # ==============================================================
    # GRAND SUMMARY
    # ==============================================================
    print("\n\n" + "=" * 160)
    print("  GRAND SUMMARY — ALL 10 MONTHS OUT-OF-SAMPLE (COMPOUNDING)")
    print(f"  Training: Oct 2025 - Apr 2026 | Starting Capital: Rs {CAPITAL:>,} | Profits reinvested each month")
    print("=" * 160)

    # EQUITY JOURNEY — the compounding story
    print(f"\n  EQUITY JOURNEY (Rs {CAPITAL:>,} starting capital):")
    print(f"  {'Month':<12}", end="")
    for name in model_names:
        print(f" | {name+' Start':>14} {'P&L':>12} {'End':>14} {'Return':>8}", end="")
    print()
    print("  " + "-" * 155)

    for s in month_summaries:
        row = f"  {s['label']:<12}"
        for name in model_names:
            start_eq = s.get(f"{name}_start_eq", CAPITAL)
            end_eq = s.get(f"{name}_end_eq", CAPITAL)
            pnl = s.get(f"{name}_pnl", 0)
            ret = pnl / max(start_eq, 1) * 100
            row += f" | Rs{start_eq:>11,} Rs{pnl:>+10,} Rs{end_eq:>11,} {ret:>+7.1f}%"
        print(row)

    # Final equity
    print("  " + "-" * 155)
    final_row = f"  {'FINAL':>12}"
    for name in model_names:
        final_eq = accumulated_equity[name]
        total_pnl = final_eq - CAPITAL
        total_ret = total_pnl / CAPITAL * 100
        total_x = final_eq / CAPITAL
        final_row += f" |{'':>14} Rs{total_pnl:>+10,} Rs{final_eq:>11,} {total_x:>6.1f}x"
    print(final_row)

    # Monthly performance breakdown
    print(f"\n  MONTHLY PERFORMANCE:")
    header = f"  {'Month':<12} {'Days':>5}"
    for name in model_names:
        header += f" | {name+' P&L':>14} {'Sharpe':>7} {'DD':>6} {'WR':>6} {'PF':>5}"
    print(header)
    print("  " + "-" * 155)

    for s in month_summaries:
        row = f"  {s['label']:<12} {s['days']:>5}"
        for name in model_names:
            pnl = s.get(f"{name}_pnl", 0)
            sharpe = s.get(f"{name}_sharpe", 0)
            dd = s.get(f"{name}_dd", 0)
            wr = s.get(f"{name}_wr", 0)
            pf = s.get(f"{name}_pf", 0)
            row += f" | Rs{pnl:>+10,} {sharpe:>6.2f} {dd:>5.1f}% {wr:>5.1f}% {pf:>4.2f}"
        print(row)

    # Totals
    print("  " + "-" * 155)
    total_row = f"  {'TOTAL':<12} {'':>5}"
    for name in model_names:
        total_pnl = accumulated_equity[name] - CAPITAL
        avg_sharpe = np.mean([s.get(f"{name}_sharpe", 0) for s in month_summaries if s.get(f"{name}_sharpe", 0) != 0])
        max_dd = max(s.get(f"{name}_dd", 0) for s in month_summaries)
        avg_wr = np.mean([s.get(f"{name}_wr", 0) for s in month_summaries if s.get(f"{name}_wr", 0) != 0])
        avg_pf = np.mean([s.get(f"{name}_pf", 0) for s in month_summaries if s.get(f"{name}_pf", 0) != 0])
        total_row += f" | Rs{total_pnl:>+10,} {avg_sharpe:>6.2f} {max_dd:>5.1f}% {avg_wr:>5.1f}% {avg_pf:>4.2f}"
    print(total_row)

    # Win/Loss months
    print(f"\n  COMPOUNDED RETURNS:")
    for name in model_names:
        win_months = sum(1 for s in month_summaries if s.get(f"{name}_pnl", 0) > 0)
        loss_months = len(month_summaries) - win_months
        final_eq = accumulated_equity[name]
        total_pnl = final_eq - CAPITAL
        multiplier = final_eq / CAPITAL
        print(f"    {name:<16}: Rs {CAPITAL:>,} -> Rs {final_eq:>,} ({multiplier:.1f}x) | "
              f"Total P&L Rs {total_pnl:>+,} | {win_months}W/{loss_months}L months")

    # Best model per month
    print(f"\n  BEST MODEL PER MONTH:")
    model_wins = defaultdict(int)
    for s in month_summaries:
        best_name = ""
        best_pnl = -float("inf")
        for name in model_names:
            pnl = s.get(f"{name}_pnl", 0)
            if pnl > best_pnl:
                best_pnl = pnl
                best_name = name
        model_wins[best_name] += 1
        print(f"    {s['label']:<12}: {best_name:<16} Rs {best_pnl:>+,}")

    print(f"\n  MODEL WIN COUNT:")
    for name in model_names:
        print(f"    {name:<16}: {model_wins[name]} / {len(month_summaries)} months")

    # Consistency score
    print(f"\n  CONSISTENCY (months with Sharpe > 1.0 AND profitable):")
    for name in model_names:
        consistent = sum(1 for s in month_summaries
                        if s.get(f"{name}_pnl", 0) > 0 and s.get(f"{name}_sharpe", 0) > 1.0)
        print(f"    {name:<16}: {consistent} / {len(month_summaries)} months")

    # Final verdict
    print(f"\n{'='*160}")
    print(f"  FINAL VERDICT — COMPOUNDED EQUITY GROWTH (Rs {CAPITAL:>,} starting):")
    print()
    for name in sorted(model_names, key=lambda n: accumulated_equity[n], reverse=True):
        final_eq = accumulated_equity[name]
        total_pnl = final_eq - CAPITAL
        multiplier = final_eq / CAPITAL
        win_months = sum(1 for s in month_summaries if s.get(f"{name}_pnl", 0) > 0)
        avg_sharpe = np.mean([s.get(f"{name}_sharpe", 0) for s in month_summaries])
        marker = " <-- WINNER" if name == sorted(model_names, key=lambda n: accumulated_equity[n], reverse=True)[0] else ""
        print(f"    {name:<16}: Rs {CAPITAL:>,} -> Rs {final_eq:>,} = {multiplier:.1f}x | "
              f"P&L Rs {total_pnl:>+,} | Avg Sharpe {avg_sharpe:.2f} | "
              f"{win_months}/{len(month_summaries)} profitable{marker}")
    print("=" * 160)

    # Save results
    import json
    winner = sorted(model_names, key=lambda n: accumulated_equity[n], reverse=True)[0]
    save_data = {
        "test_months": [s["label"] for s in month_summaries],
        "monthly": month_summaries,
        "final_equity": {n: accumulated_equity[n] for n in model_names},
        "winner": winner,
        "compounding": True,
        "starting_capital": CAPITAL,
    }
    save_path = DATA_DIR / "multi_month_oos_compounded.json"
    with open(save_path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\nResults saved to {save_path}")
