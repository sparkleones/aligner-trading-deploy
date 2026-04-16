"""
V16 REVERSAL DETECTION TEST
============================
Tests V16 (adaptive reversal + position flip) vs V14 (day-based filter).

V16 Philosophy: Don't skip days. Read the market, detect reversals, flip positions.
V14 Philosophy: Skip Monday/Wednesday entirely (23.4% and worse WR).

Test 1: April 6, 2026 (Monday) — The day user lost Rs 2,500
Test 2: Full 12-month OOS backtest — Does V16 outperform V14?
Test 3: V16 vs V14 vs V14-no-filter — Which approach wins?
"""

import sys
import copy
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
    get_dynamic_lots, run_model,
)
from backtesting.paper_trading_real_data import bs_premium, get_strike_and_type

TARGET_DATE = dt.date(2026, 4, 6)


def run_single_day(cfg, label, nifty_ind, daily, vix_lookup, day_groups,
                   close_prices, all_dates, equity, target_date=TARGET_DATE):
    """Run model on a single day and return trades + print results."""

    target_idx = None
    for i, d in enumerate(all_dates):
        if d == target_date:
            target_idx = i
            break

    if target_idx is None:
        print(f"  {label}: Date {target_date} not found!")
        return []

    day_bars = day_groups[target_date]
    vix = vix_lookup.get(target_date, 14.0)
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
    ema9_val = float(row.get("EMA9", row["Close"]))
    ema21_val = float(row.get("EMA21", row["Close"]))
    weekly_sma = float(row.get("WeeklySMA", row["Close"]))
    gap_pct = float(row.get("GapPct", 0))

    regime_info = detect_market_regime(daily, target_idx)

    days_to_thu = (3 - target_date.weekday()) % 7
    if days_to_thu == 0:
        days_to_thu = 7
    dte = days_to_thu
    is_expiry = (target_date.weekday() == 3)

    if ema9_val > ema21_val and row["Close"] > sma20:
        daily_trend = "bullish"
    elif ema9_val < ema21_val and row["Close"] < sma20:
        daily_trend = "bearish"
    else:
        daily_trend = "neutral"

    trades = simulate_day(
        cfg, day_bars, target_date, prev_ohlc, vix, daily_trend,
        dte, is_expiry, daily, target_idx, close_prices,
        above_sma50, above_sma20, rsi, prev_change, vix_spike,
        sma20, sma50, ema9_val, ema21_val, weekly_sma, gap_pct,
        equity=equity, recent_wr=0.5, recent_trades=0,
        regime_info=regime_info,
    )

    # Print results
    print(f"\n  {'='*115}")
    print(f"  {label}")
    print(f"  {'='*115}")
    print(f"  Date: {target_date} ({target_date.strftime('%A')}) | VIX: {vix:.2f} | "
          f"Regime: {regime_info['regime']} | Trend: {daily_trend}")
    print(f"  NIFTY: O={day_bars['open'].iloc[0]:.0f} H={day_bars['high'].max():.0f} "
          f"L={day_bars['low'].min():.0f} C={day_bars['close'].iloc[-1]:.0f}")

    if not trades:
        avoid = cfg.get("avoid_days", [])
        if target_date.strftime("%A") in avoid:
            print(f"  >>> ZERO TRADES -- {target_date.strftime('%A')} is in avoid_days filter")
        else:
            print(f"  >>> ZERO TRADES -- No valid signals")
        return trades

    total_pnl = 0
    print(f"\n  {'#':>3} {'Action':>9} {'Entry':>6} {'Exit':>6} {'Type':<18} "
          f"{'SpotIn':>8} {'SpotOut':>8} {'Strike':>7} {'PremIn':>7} {'PremOut':>7} "
          f"{'Lots':>4} {'Held':>5} {'Exit Reason':<18} {'P&L':>10}")
    print(f"  {'-'*135}")

    for i, t in enumerate(trades):
        entry_h = 9 + (15 + t["entry_minute"]) // 60
        entry_m = (15 + t["entry_minute"]) % 60
        exit_h = 9 + (15 + t["exit_minute"]) // 60
        exit_m = (15 + t["exit_minute"]) % 60

        rev_tag = ""
        if t.get("reversal_signals"):
            rev_tag = f" [{'+'.join(t['reversal_signals'][:2])}]"

        print(f"  {i+1:>3} {t['action']:>9} {entry_h:02d}:{entry_m:02d}  "
              f"{exit_h:02d}:{exit_m:02d}  {t['entry_type']+rev_tag:<18} "
              f"{t['entry_spot']:>8.0f} {t['exit_spot']:>8.0f} {t['strike']:>7} "
              f"{t['entry_prem']:>7.1f} {t['exit_prem']:>7.1f} "
              f"{t['lots']:>4} {t['minutes_held']:>4}m "
              f"{t['exit_reason']:<18} Rs{t['pnl']:>+9,.0f}")
        total_pnl += t["pnl"]

    wins = sum(1 for t in trades if t["pnl"] > 0)
    losses = len(trades) - wins
    print(f"  {'-'*135}")
    print(f"  TOTAL: {len(trades)} trades ({wins}W/{losses}L) | Day P&L: Rs{total_pnl:>+,}")
    print(f"  Capital after: Rs {equity + total_pnl:,.0f} ({total_pnl/equity*100:+.1f}%)")

    return trades


def run_multi_month_test(cfg, label, verbose=False):
    """Run full 12-month OOS test and return summary."""
    print(f"\n  Running {label}...")

    # Download all test months
    test_months = [
        ("2024-06-01", "2024-06-30"),
        ("2024-07-01", "2024-08-31"),
        ("2024-08-01", "2024-09-30"),
        ("2024-09-01", "2024-10-31"),
        ("2024-10-01", "2024-11-30"),
        ("2024-11-01", "2024-12-31"),
        ("2024-12-01", "2025-01-31"),
        ("2025-02-01", "2025-03-31"),
        ("2025-04-01", "2025-05-31"),
        ("2025-06-01", "2025-07-31"),
        ("2025-08-01", "2025-09-30"),
    ]

    equity = CAPITAL  # 200,000
    total_trades = 0
    total_wins = 0
    total_pnl = 0
    monthly_results = []
    all_reversal_trades = []

    for start, end in test_months:
        nifty, vix_data = download_range(start, end)
        if nifty is None:
            continue

        nifty_ind = add_all_indicators(nifty.copy())
        day_groups = {d: g for d, g in nifty_ind.groupby(nifty_ind.index.date)}

        vix_lookup = {}
        if vix_data is not None and not vix_data.empty:
            for idx_dt, row in vix_data.iterrows():
                vix_lookup[idx_dt.date()] = row["close"]

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

        # Build daily trend DF
        daily_trend_df = daily.rename(columns={"Close": "close", "SMA20": "sma20",
                                                "EMA9": "ema9", "EMA21": "ema21"})
        daily_trend_df.index = [d.date() for d in daily_trend_df.index]

        test_start = dt.datetime.strptime(start, "%Y-%m-%d").date()
        result = run_model(cfg, daily, close_prices, day_groups, all_dates,
                          vix_lookup, daily_trend_df, test_start,
                          starting_equity=equity)

        month_pnl = result["net_pnl"]
        equity = result["final_equity"]
        total_trades += result["total_trades"]
        total_wins += result["wins"]
        total_pnl += month_pnl

        # Track reversal trades
        for t in result.get("all_trades", []):
            if t.get("entry_type") == "reversal_flip" or "reversal" in t.get("exit_reason", ""):
                all_reversal_trades.append(t)

        month_label = dt.datetime.strptime(start, "%Y-%m-%d").strftime("%b %Y")
        win_pct = result["win_rate"]
        status = "+" if month_pnl > 0 else "-"
        monthly_results.append({
            "month": month_label, "pnl": month_pnl, "trades": result["total_trades"],
            "wr": win_pct, "equity": equity,
        })

        if verbose:
            print(f"    {month_label}: Rs{month_pnl:>+10,} | {result['total_trades']:>3} trades | "
                  f"WR {win_pct:.0f}% | Equity: Rs{equity:,.0f} [{status}]")

    return_x = equity / CAPITAL
    overall_wr = (total_wins / total_trades * 100) if total_trades > 0 else 0
    months_positive = sum(1 for m in monthly_results if m["pnl"] > 0)

    return {
        "label": label,
        "equity": equity,
        "return_x": return_x,
        "total_pnl": total_pnl,
        "total_trades": total_trades,
        "overall_wr": overall_wr,
        "months_positive": months_positive,
        "total_months": len(monthly_results),
        "monthly": monthly_results,
        "reversal_trades": all_reversal_trades,
    }


def main():
    print("=" * 120)
    print("  V16 REVERSAL DETECTION vs V14 DAY FILTER")
    print("  Philosophy: Read the market, don't skip entire days")
    print("=" * 120)

    # ─── LOAD DATA FOR APRIL 6 ───
    nifty, vix_data = download_range("2026-03-01", "2026-04-07")
    if nifty is None:
        print("No data!")
        return

    vix_lookup = {}
    if vix_data is not None:
        for idx, row in vix_data.iterrows():
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

    # ==================================================================
    # TEST 1: APRIL 6 — V14 (Monday blocked) vs V16 (reversal detection)
    # ==================================================================
    print("\n" + "=" * 120)
    print("  TEST 1: APRIL 6, 2026 (Monday) — Single Day Comparison")
    print("=" * 120)

    # V14: Monday blocked
    cfg_v14 = copy.deepcopy(V9_HYBRID_CONFIG)
    cfg_v14["use_reversal_detection"] = False
    cfg_v14["reversal_flip_enabled"] = False
    cfg_v14["avoid_days"] = ["Monday", "Wednesday"]
    trades_v14 = run_single_day(cfg_v14, "V14 (Monday BLOCKED -- avoid_days filter)",
                                nifty_ind, daily, vix_lookup, day_groups,
                                close_prices, all_dates, equity=30_000)

    # V14 without day filter (to see what would happen)
    cfg_v14_nofilter = copy.deepcopy(V9_HYBRID_CONFIG)
    cfg_v14_nofilter["use_reversal_detection"] = False
    cfg_v14_nofilter["reversal_flip_enabled"] = False
    cfg_v14_nofilter["avoid_days"] = []
    trades_v14_nf = run_single_day(cfg_v14_nofilter, "V14 (Monday ALLOWED, no reversal)",
                                    nifty_ind, daily, vix_lookup, day_groups,
                                    close_prices, all_dates, equity=30_000)

    # V16: Reversal detection + flip (no day filter)
    cfg_v16 = copy.deepcopy(V9_HYBRID_CONFIG)
    # V16 config is already set in V9_HYBRID_CONFIG, just make sure it's enabled
    cfg_v16["use_reversal_detection"] = True
    cfg_v16["reversal_flip_enabled"] = True
    cfg_v16["avoid_days"] = []  # No day filter — let reversal handle it
    trades_v16 = run_single_day(cfg_v16, "V16 (REVERSAL DETECTION + FLIP -- no day filter)",
                                nifty_ind, daily, vix_lookup, day_groups,
                                close_prices, all_dates, equity=30_000)

    # Summary comparison
    v14_pnl = sum(t["pnl"] for t in trades_v14)
    v14nf_pnl = sum(t["pnl"] for t in trades_v14_nf)
    v16_pnl = sum(t["pnl"] for t in trades_v16)

    print(f"\n  {'='*115}")
    print(f"  APRIL 6 COMPARISON SUMMARY")
    print(f"  {'='*115}")
    print(f"  Your live result (V3):    Rs -2,500  (-8.3%)")
    print(f"  V14 (Monday blocked):     Rs{v14_pnl:>+7,}  ({v14_pnl/30000*100:+.1f}%)  [{len(trades_v14)} trades]")
    print(f"  V14 (Monday allowed):     Rs{v14nf_pnl:>+7,}  ({v14nf_pnl/30000*100:+.1f}%)  [{len(trades_v14_nf)} trades]")
    print(f"  V16 (Reversal + Flip):    Rs{v16_pnl:>+7,}  ({v16_pnl/30000*100:+.1f}%)  [{len(trades_v16)} trades]")

    # Show reversal details
    rev_trades = [t for t in trades_v16 if "reversal" in t.get("exit_reason", "") or
                  t.get("entry_type") == "reversal_flip"]
    if rev_trades:
        print(f"\n  REVERSAL TRADES:")
        for t in rev_trades:
            print(f"    {t['action']:>9} via {t['entry_type']:<18} | "
                  f"entry={t['entry_minute']}min exit={t['exit_minute']}min | "
                  f"exit_reason={t['exit_reason']} | P&L=Rs{t['pnl']:>+,}")
            if t.get("reversal_signals"):
                print(f"      Reversal signals: {', '.join(t['reversal_signals'])}")

    # ==================================================================
    # TEST 2: FULL 12-MONTH OOS BACKTEST
    # ==================================================================
    print("\n" + "=" * 120)
    print("  TEST 2: FULL 12-MONTH OOS BACKTEST")
    print("  Compounding equity, same test months as previous runs")
    print("=" * 120)

    # V14 production (with day filter)
    cfg_v14_full = copy.deepcopy(V9_HYBRID_CONFIG)
    cfg_v14_full["use_reversal_detection"] = False
    cfg_v14_full["reversal_flip_enabled"] = False
    cfg_v14_full["avoid_days"] = ["Monday", "Wednesday"]
    r_v14 = run_multi_month_test(cfg_v14_full, "V14 (Mon/Wed blocked)", verbose=True)

    # V14 without day filter
    cfg_v14nf_full = copy.deepcopy(V9_HYBRID_CONFIG)
    cfg_v14nf_full["use_reversal_detection"] = False
    cfg_v14nf_full["reversal_flip_enabled"] = False
    cfg_v14nf_full["avoid_days"] = []
    r_v14nf = run_multi_month_test(cfg_v14nf_full, "V14 (all days, no reversal)", verbose=True)

    # V16 (reversal detection, no day filter)
    cfg_v16_full = copy.deepcopy(V9_HYBRID_CONFIG)
    cfg_v16_full["use_reversal_detection"] = True
    cfg_v16_full["reversal_flip_enabled"] = True
    cfg_v16_full["avoid_days"] = []
    r_v16 = run_multi_month_test(cfg_v16_full, "V16 (Reversal + Flip, all days)", verbose=True)

    # ==================================================================
    # RESULTS COMPARISON
    # ==================================================================
    print(f"\n{'='*120}")
    print(f"  FULL BACKTEST COMPARISON")
    print(f"{'='*120}")
    print(f"\n  {'Model':<35} {'Return':>8} {'Trades':>7} {'WR':>6} {'Months+':>8} {'Final Equity':>14}")
    print(f"  {'-'*85}")
    for r in [r_v14, r_v14nf, r_v16]:
        print(f"  {r['label']:<35} {r['return_x']:>7.1f}x {r['total_trades']:>7} "
              f"{r['overall_wr']:>5.1f}% {r['months_positive']:>3}/{r['total_months']:<4} "
              f"Rs{r['equity']:>12,.0f}")

    # V16 reversal trade analysis
    if r_v16["reversal_trades"]:
        rev = r_v16["reversal_trades"]
        rev_wins = sum(1 for t in rev if t["pnl"] > 0)
        rev_pnl = sum(t["pnl"] for t in rev)
        print(f"\n  V16 REVERSAL TRADE ANALYSIS:")
        print(f"  Total reversal-related trades: {len(rev)}")
        print(f"  Wins: {rev_wins}, Losses: {len(rev) - rev_wins}")
        print(f"  Win Rate: {rev_wins/len(rev)*100:.1f}%")
        print(f"  Total P&L: Rs{rev_pnl:>+,}")
        print(f"  Avg P&L: Rs{rev_pnl/len(rev):>+,.0f}")

        # Show top reversal trades
        rev_sorted = sorted(rev, key=lambda x: x["pnl"], reverse=True)
        print(f"\n  Top 5 Reversal Trades:")
        for t in rev_sorted[:5]:
            print(f"    {t['date']} | {t['action']:>9} | {t['entry_type']:<18} | "
                  f"P&L=Rs{t['pnl']:>+,} | exit={t.get('exit_reason','')}")

        if len(rev_sorted) > 5:
            print(f"\n  Worst 5 Reversal Trades:")
            for t in rev_sorted[-5:]:
                print(f"    {t['date']} | {t['action']:>9} | {t['entry_type']:<18} | "
                      f"P&L=Rs{t['pnl']:>+,} | exit={t.get('exit_reason','')}")

    print(f"\n{'='*120}")
    print(f"  CONCLUSION")
    print(f"{'='*120}")
    v14_ret = r_v14["return_x"]
    v16_ret = r_v16["return_x"]
    if v16_ret > v14_ret:
        print(f"  V16 WINS: {v16_ret:.1f}x vs V14's {v14_ret:.1f}x")
        print(f"  Reversal detection + flip OUTPERFORMS day-based filtering!")
        print(f"  The model can now trade every day and adapt to market behavior.")
    elif v16_ret > v14_ret * 0.95:
        print(f"  V16 COMPARABLE: {v16_ret:.1f}x vs V14's {v14_ret:.1f}x")
        print(f"  Reversal detection trades more days with similar returns.")
        print(f"  Consider V16 for more opportunities without skipping days.")
    else:
        print(f"  V14 STILL BETTER: {v14_ret:.1f}x vs V16's {v16_ret:.1f}x")
        print(f"  Day filtering remains the safer approach for now.")
        print(f"  Reversal detection may need tuning (thresholds, signal quality).")
    print(f"{'='*120}")


if __name__ == "__main__":
    main()
