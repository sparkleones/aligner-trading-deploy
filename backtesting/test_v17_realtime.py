"""
V17 REAL-TIME ADAPTIVE ENGINE TEST
====================================
Tests V17 (market state machine, every-minute analysis) vs V14 (best production config).

V17 Philosophy: Read the market EVERY MINUTE. No time blocks, no avoided windows.
  Entries based on market STATE TRANSITIONS, not periodic indicator checks.
  Exits based on market state turning AGAINST position, not fixed trail percentage.

V14 Philosophy: Confluence filters (VWAP+RSI+Squeeze) handle signal quality.
  5-min V8 indicator checks + 15-min composite entries. Time-based avoid windows.

Comparison:
  A) V14 Production (avoid_days=[], VWAP+RSI+Squeeze, 5/15-min cadence)  -- 587.5x baseline
  B) V17 Realtime (state machine entries + state-based exits)
  C) V17 Hybrid (state machine + V8 at 5-min cadence -- best of both)
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
    get_dynamic_lots, run_model, compute_realtime_state,
)
from backtesting.paper_trading_real_data import bs_premium, get_strike_and_type

TARGET_DATE = dt.date(2026, 4, 6)


def prepare_daily(day_groups, all_dates, vix_lookup):
    """Build daily DataFrame from day_groups."""
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
    return daily, close_prices


def run_single_day(cfg, label, nifty_ind, daily, vix_lookup, day_groups,
                   close_prices, all_dates, equity, target_date=TARGET_DATE,
                   show_state_trace=False):
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
    rsi_val = float(row.get("RSI", 50))
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
        above_sma50, above_sma20, rsi_val, prev_change, vix_spike,
        sma20, sma50, ema9_val, ema21_val, weekly_sma, gap_pct,
        equity=equity, recent_wr=0.5, recent_trades=0,
        regime_info=regime_info,
    )

    # Print results
    print(f"\n  {'='*120}")
    print(f"  {label}")
    print(f"  {'='*120}")
    print(f"  Date: {target_date} ({target_date.strftime('%A')}) | VIX: {vix:.2f} | "
          f"Regime: {regime_info['regime']} | Trend: {daily_trend}")
    print(f"  NIFTY: O={day_bars['open'].iloc[0]:.0f} H={day_bars['high'].max():.0f} "
          f"L={day_bars['low'].min():.0f} C={day_bars['close'].iloc[-1]:.0f}")

    if not trades:
        avoid = cfg.get("avoid_days", [])
        if target_date.strftime("%A") in avoid:
            print(f"  >>> ZERO TRADES -- {target_date.strftime('%A')} is in avoid_days filter")
        else:
            print(f"  >>> ZERO TRADES -- No valid signals passed filters")
        return trades

    total_pnl = 0
    print(f"\n  {'#':>3} {'Action':>9} {'Entry':>6} {'Exit':>6} {'Type':<22} "
          f"{'SpotIn':>8} {'SpotOut':>8} {'Strike':>7} {'PremIn':>7} {'PremOut':>7} "
          f"{'Lots':>4} {'Held':>5} {'Exit Reason':<20} {'P&L':>10}")
    print(f"  {'-'*145}")

    for i, t in enumerate(trades):
        entry_h = 9 + (15 + t["entry_minute"]) // 60
        entry_m = (15 + t["entry_minute"]) % 60
        exit_h = 9 + (15 + t["exit_minute"]) // 60
        exit_m = (15 + t["exit_minute"]) % 60

        print(f"  {i+1:>3} {t['action']:>9} {entry_h:02d}:{entry_m:02d}  "
              f"{exit_h:02d}:{exit_m:02d}  {t['entry_type']:<22} "
              f"{t['entry_spot']:>8.0f} {t['exit_spot']:>8.0f} {t['strike']:>7} "
              f"{t['entry_prem']:>7.1f} {t['exit_prem']:>7.1f} "
              f"{t['lots']:>4} {t['minutes_held']:>4}m "
              f"{t['exit_reason']:<20} Rs{t['pnl']:>+9,.0f}")
        total_pnl += t["pnl"]

    wins = sum(1 for t in trades if t["pnl"] > 0)
    losses = len(trades) - wins
    print(f"  {'-'*145}")
    print(f"  TOTAL: {len(trades)} trades ({wins}W/{losses}L) | Day P&L: Rs{total_pnl:>+,}")
    print(f"  Capital after: Rs {equity + total_pnl:,.0f} ({total_pnl/equity*100:+.1f}%)")

    # Show state trace for V17
    if show_state_trace and cfg.get("use_realtime_engine", False):
        print(f"\n  V17 STATE TRACE (every 15 minutes):")
        closes_arr = day_bars["close"].values
        highs_arr = day_bars["high"].values
        lows_arr = day_bars["low"].values
        # Build simple VWAP
        tp = (highs_arr + lows_arr + closes_arr) / 3.0
        vwap_arr = np.cumsum(tp) / np.arange(1, len(tp) + 1, dtype=float)
        prev_state = "RANGING"
        for m in range(0, len(closes_arr), 15):
            st = compute_realtime_state(closes_arr, highs_arr, lows_arr, vwap_arr, m, prev_state)
            t_h = 9 + (15 + m) // 60
            t_m = (15 + m) % 60
            print(f"    {t_h:02d}:{t_m:02d} | State: {st['state']:<16} | "
                  f"Str: {st['strength']:.2f} | RSI7: {st['rsi']:.0f} | "
                  f"Mom: {st['momentum']:+.3f}% | VWAP: {st['vwap_pos']:<6} | "
                  f"EMA: {st['ema_trend']:<8} | Spot: {closes_arr[m]:.0f}")
            prev_state = st["state"]

    return trades


def run_multi_month_test(cfg, label, verbose=True):
    """Run full 12-month OOS test and return summary."""
    print(f"\n  Running {label}...")

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
    rt_entry_types = defaultdict(int)
    rt_exit_types = defaultdict(int)

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
        daily, close_prices = prepare_daily(day_groups, all_dates, vix_lookup)

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

        # Track RT entry/exit types
        for t in result.get("all_trades", []):
            et = t.get("entry_type", "unknown")
            rt_entry_types[et] += 1
            er = t.get("exit_reason", "unknown")
            rt_exit_types[er] += 1

        month_label = dt.datetime.strptime(start, "%Y-%m-%d").strftime("%b %Y")
        win_pct = result["win_rate"]
        status = "+" if month_pnl > 0 else "-"
        monthly_results.append({
            "month": month_label, "pnl": month_pnl, "trades": result["total_trades"],
            "wr": win_pct, "equity": equity,
        })

        if verbose:
            print(f"    {month_label}: Rs{month_pnl:>+12,} | {result['total_trades']:>3} trades | "
                  f"WR {win_pct:.0f}% | Equity: Rs{equity:>14,.0f} [{status}]")

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
        "entry_types": dict(rt_entry_types),
        "exit_types": dict(rt_exit_types),
    }


def main():
    print("=" * 120)
    print("  V17 REAL-TIME ADAPTIVE ENGINE TEST")
    print("  Market state machine: reads price action EVERY MINUTE")
    print("  States: TRENDING_UP/DOWN | EXHAUSTION_UP/DOWN | REVERSAL_UP/DOWN | RANGING")
    print("=" * 120)

    # ==========================================
    # PART 1: APRIL 6, 2026 (Monday) — Single Day
    # ==========================================
    print("\n" + "=" * 120)
    print("  PART 1: APRIL 6, 2026 (Monday) -- The V-Recovery Day")
    print("  Open 22780 -> Low 22543 (-237 pts) -> Close 22998 (+455 pt recovery)")
    print("=" * 120)

    nifty, vix_data = download_range("2026-03-01", "2026-04-07")
    if nifty is None:
        print("No data for April 6!")
        return

    vix_lookup = {}
    if vix_data is not None:
        for idx, row in vix_data.iterrows():
            vix_lookup[idx.date()] = row["close"]

    nifty_ind = add_all_indicators(nifty.copy())
    day_groups = {d: g for d, g in nifty_ind.groupby(nifty_ind.index.date)}
    all_dates = sorted(day_groups.keys())
    daily, close_prices = prepare_daily(day_groups, all_dates, vix_lookup)

    # --- Config A: V14 Production (baseline -- 587.5x) ---
    cfg_v14 = copy.deepcopy(V9_HYBRID_CONFIG)
    cfg_v14["use_realtime_engine"] = False
    cfg_v14["use_reversal_detection"] = False
    cfg_v14["reversal_flip_enabled"] = False
    cfg_v14["avoid_days"] = []  # Best config (all days, VWAP+RSI+Squeeze filter)
    trades_v14 = run_single_day(cfg_v14, "A: V14 Production (all days, VWAP+RSI+Squeeze, 5/15-min cadence)",
                                nifty_ind, daily, vix_lookup, day_groups,
                                close_prices, all_dates, equity=30_000)

    # --- Config B: V17 Realtime Only ---
    cfg_v17 = copy.deepcopy(V9_HYBRID_CONFIG)
    cfg_v17["use_realtime_engine"] = True
    cfg_v17["rt_check_every_minute"] = True
    cfg_v17["rt_remove_avoid_windows"] = True
    cfg_v17["rt_state_exit"] = True
    cfg_v17["use_reversal_detection"] = False
    cfg_v17["reversal_flip_enabled"] = False
    cfg_v17["avoid_days"] = []
    trades_v17 = run_single_day(cfg_v17, "B: V17 Realtime (state machine, every-minute, state-based exits)",
                                nifty_ind, daily, vix_lookup, day_groups,
                                close_prices, all_dates, equity=30_000,
                                show_state_trace=True)

    # --- Config C: V17 Hybrid (RT entries + V8 at 5-min + legacy exits as fallback) ---
    cfg_v17h = copy.deepcopy(V9_HYBRID_CONFIG)
    cfg_v17h["use_realtime_engine"] = True
    cfg_v17h["rt_check_every_minute"] = True
    cfg_v17h["rt_remove_avoid_windows"] = True
    cfg_v17h["rt_state_exit"] = True  # State exits checked FIRST, then legacy
    cfg_v17h["use_reversal_detection"] = False
    cfg_v17h["reversal_flip_enabled"] = False
    cfg_v17h["avoid_days"] = []
    # Keep all V14 filters active too (VWAP, squeeze, RSI gate)
    trades_v17h = run_single_day(cfg_v17h, "C: V17 Hybrid (RT state machine + V8 5-min + legacy exits fallback)",
                                 nifty_ind, daily, vix_lookup, day_groups,
                                 close_prices, all_dates, equity=30_000)

    # April 6 summary
    v14_pnl = sum(t["pnl"] for t in trades_v14)
    v17_pnl = sum(t["pnl"] for t in trades_v17)
    v17h_pnl = sum(t["pnl"] for t in trades_v17h)

    print(f"\n  {'='*120}")
    print(f"  APRIL 6 COMPARISON")
    print(f"  {'='*120}")
    print(f"  Your live result (V3):       Rs -2,500  (-8.3%)  [1 trade, PUT at 10:00]")
    print(f"  A) V14 Production:           Rs{v14_pnl:>+7,}  ({v14_pnl/30000*100:+.1f}%)  [{len(trades_v14)} trades]")
    print(f"  B) V17 Realtime:             Rs{v17_pnl:>+7,}  ({v17_pnl/30000*100:+.1f}%)  [{len(trades_v17)} trades]")
    print(f"  C) V17 Hybrid:               Rs{v17h_pnl:>+7,}  ({v17h_pnl/30000*100:+.1f}%)  [{len(trades_v17h)} trades]")

    # Show entry type breakdown
    for label, trades in [("V17 Realtime", trades_v17), ("V17 Hybrid", trades_v17h)]:
        if trades:
            et_counts = defaultdict(int)
            ex_counts = defaultdict(int)
            for t in trades:
                et_counts[t["entry_type"]] += 1
                ex_counts[t["exit_reason"]] += 1
            print(f"\n  {label} Entry types: {dict(et_counts)}")
            print(f"  {label} Exit reasons: {dict(ex_counts)}")

    # ==========================================
    # PART 2: FULL 12-MONTH OOS BACKTEST
    # ==========================================
    print("\n" + "=" * 120)
    print("  PART 2: FULL 12-MONTH OOS BACKTEST (Jun 2024 - Sep 2025)")
    print("  Starting capital: Rs 2,00,000 | Compounding equity")
    print("  V17b: Conservative state exits (loss protection only) + selective entries")
    print("=" * 120)

    # A: V14 Production (baseline)
    cfg_a = copy.deepcopy(V9_HYBRID_CONFIG)
    cfg_a["use_realtime_engine"] = False
    cfg_a["use_reversal_detection"] = False
    cfg_a["reversal_flip_enabled"] = False
    cfg_a["avoid_days"] = []
    result_a = run_multi_month_test(cfg_a, "A: V14 Production (587.5x baseline)")

    # B: V17b Full (RT entries + conservative state exits + no avoid windows)
    cfg_b = copy.deepcopy(V9_HYBRID_CONFIG)
    cfg_b["use_realtime_engine"] = True
    cfg_b["rt_check_every_minute"] = True
    cfg_b["rt_remove_avoid_windows"] = True
    cfg_b["rt_state_exit"] = True
    cfg_b["use_reversal_detection"] = False
    cfg_b["reversal_flip_enabled"] = False
    cfg_b["avoid_days"] = []
    result_b = run_multi_month_test(cfg_b, "B: V17b Full (RT entries + smart stops)")

    # C: V17b Entries Only (RT entries + V8 hybrid, legacy exits, no avoid windows)
    cfg_c = copy.deepcopy(V9_HYBRID_CONFIG)
    cfg_c["use_realtime_engine"] = True
    cfg_c["rt_check_every_minute"] = True
    cfg_c["rt_remove_avoid_windows"] = True
    cfg_c["rt_state_exit"] = False  # Legacy exits only -- isolate entry effect
    cfg_c["use_reversal_detection"] = False
    cfg_c["reversal_flip_enabled"] = False
    cfg_c["avoid_days"] = []
    result_c = run_multi_month_test(cfg_c, "C: V17b Entries + Legacy Exits")

    # D: V17b Smart Stops Only (legacy entries, RT state exits, keep windows)
    cfg_d = copy.deepcopy(V9_HYBRID_CONFIG)
    cfg_d["use_realtime_engine"] = True
    cfg_d["rt_check_every_minute"] = True
    cfg_d["rt_remove_avoid_windows"] = False  # Keep avoid windows
    cfg_d["rt_state_exit"] = True  # V17b conservative exits
    cfg_d["use_reversal_detection"] = False
    cfg_d["reversal_flip_enabled"] = False
    cfg_d["avoid_days"] = []
    result_d = run_multi_month_test(cfg_d, "D: V17b Smart Stops + Legacy Windows")

    # E: BEST COMBO — V14 entries + V17b smart stops (no RT entries)
    # Hypothesis: V14's entry system is optimal, but its exits are too slow
    # on losing trades. Use V17b's state-based stop-loss to cut losers faster
    # while keeping V14's avoid_windows and entry cadence.
    cfg_e = copy.deepcopy(V9_HYBRID_CONFIG)
    cfg_e["use_realtime_engine"] = True  # Needed for state computation
    cfg_e["rt_check_every_minute"] = True
    cfg_e["rt_remove_avoid_windows"] = False  # KEEP V14's avoid windows
    cfg_e["rt_state_exit"] = True  # V17b conservative smart stops
    cfg_e["use_reversal_detection"] = False
    cfg_e["reversal_flip_enabled"] = False
    cfg_e["avoid_days"] = []
    # Disable RT entry generation — only use V8+composite entries from legacy path
    cfg_e["rt_entries_disabled"] = True
    result_e = run_multi_month_test(cfg_e, "E: V14 Entries + V17b Smart Stops (best combo)")

    # ==========================================
    # FINAL COMPARISON TABLE
    # ==========================================
    print("\n" + "=" * 120)
    print("  FINAL COMPARISON — V14 vs V17 VARIANTS")
    print("=" * 120)

    results = [result_a, result_b, result_c, result_d, result_e]

    print(f"\n  {'Config':<55} {'Return':>10} {'Trades':>7} {'WR':>6} {'Months+':>8} {'Final Equity':>16}")
    print(f"  {'-'*110}")

    for r in results:
        print(f"  {r['label']:<55} {r['return_x']:>9.1f}x {r['total_trades']:>6} "
              f"{r['overall_wr']:>5.1f}% {r['months_positive']:>3}/{r['total_months']:<3} "
              f"Rs{r['equity']:>14,.0f}")

    # Show best result
    best = max(results, key=lambda x: x["return_x"])
    worst = min(results, key=lambda x: x["return_x"])

    print(f"\n  BEST:  {best['label']} -> {best['return_x']:.1f}x return")
    print(f"  WORST: {worst['label']} -> {worst['return_x']:.1f}x return")

    # Show entry/exit type analysis for V17
    for r in results:
        if r.get("entry_types"):
            print(f"\n  {r['label']}:")
            print(f"    Entry types: ", end="")
            sorted_et = sorted(r["entry_types"].items(), key=lambda x: -x[1])
            for et, count in sorted_et[:8]:
                print(f"{et}={count} ", end="")
            print()
            if r.get("exit_types"):
                print(f"    Exit types:  ", end="")
                sorted_ex = sorted(r["exit_types"].items(), key=lambda x: -x[1])
                for ex, count in sorted_ex[:8]:
                    print(f"{ex}={count} ", end="")
                print()

    print("\n" + "=" * 120)
    print("  CONCLUSION")
    print("=" * 120)
    if best["return_x"] > result_a["return_x"]:
        improvement = (best["return_x"] / result_a["return_x"] - 1) * 100
        print(f"  V17 OUTPERFORMS V14 by {improvement:.1f}%")
        print(f"  {best['label']} is the new best config")
        print(f"  Market reading works better than fixed cadence!")
    elif best == result_a:
        print(f"  V14 remains the best config at {result_a['return_x']:.1f}x")
        print(f"  V17 state machine needs tuning before deployment")
        # Show where V17 underperforms
        for r in results[1:]:
            if r["return_x"] < result_a["return_x"]:
                ratio = r["return_x"] / result_a["return_x"]
                print(f"    {r['label']}: {ratio:.1%} of V14 ({r['return_x']:.1f}x vs {result_a['return_x']:.1f}x)")
    print("=" * 120)


if __name__ == "__main__":
    main()
