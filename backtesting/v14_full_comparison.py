"""
V14 Enh-Tuned vs Enh-Tuned-Full -- Add Dual-Entry Engine + Zero-to-Hero + BTST
================================================================================
Tests whether enabling ALL three trade types on top of Enh-Tuned improves returns:

  1. Enh-Tuned      -- Current production config (21 scoring signals, fixed % trails)
  2. Enh-Tuned-Full  -- Enh-Tuned + dual-entry engine (V8 + composite) + zero-to-hero + BTST

Both use dynamic lot sizing with equity compounding (already built into run_model_enhanced).
Uses the same 6 months of cached historical data (Jul 2024 - Jan 2025).
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

from backtesting.oos_june2024_test import (
    V9_HYBRID_CONFIG, CAPITAL,
    simulate_day, run_model,
)
from backtesting.multi_month_oos_test import download_range, run_month
from backtesting.daywise_analysis import add_all_indicators
from backtesting.v14_enhanced_comparison import (
    make_v14_enh_tuned_config,
    run_model_enhanced,
    simulate_day_enhanced,
)

DATA_DIR = project_root / "data" / "historical"


def make_v14_enh_tuned_full_config():
    """Build Enh-Tuned-Full config -- Enh-Tuned + dual-entry + zero-to-hero + BTST.

    This re-enables the V9_HYBRID features that were stripped for the pure V14 test:
      - use_hybrid: True (dual entry: V8 indicators + V6_3T composite windows)
      - zero_hero: True (deep OTM entries on gap days)
      - btst_enabled: True (overnight holds for profitable positions)
      - composite_windows_put/call: V6_3T optimal windows
      - entry_type_lot_mult: scale lots by entry quality
      - max_concurrent=3, max_trades=7 (handle more concurrent positions)

    All Enh-Tuned scoring signals, fixed % trails, theta exit, etc. remain active.
    """
    cfg = make_v14_enh_tuned_config()
    cfg["name"] = "Enh-Tuned-Full"

    # ── ENABLE dual-entry engine (V8 + V6_3T composite) ──
    cfg["use_hybrid"] = True

    # ── ENABLE zero-to-hero entries ──
    cfg["zero_hero"] = True

    # ── ENABLE BTST ──
    cfg["btst_enabled"] = True
    cfg["btst_vix_cap"] = 25

    # ── Composite windows (from V9_HYBRID — proven V6_3T windows) ──
    cfg["composite_windows_put"] = [(30, 105), (120, 210)]   # bars 2-7, 8-14 -> min
    cfg["composite_windows_call"] = [(45, 180)]               # bars 3-12 -> min

    # ── Entry type lot multipliers (scale by entry quality) ──
    cfg["entry_type_lot_mult"] = {
        "v8_indicator": 1.0,
        "sr_bounce_resistance": 1.5,  # 61.5% WR, best entry type
        "orb_breakout_down": 1.5,
        "orb_breakout_up": 0.5,
        "composite": 0.8,
        "gap_entry": 1.0,
        "gap_fade": 0.7,
        "gap_zero_hero": 1.0,
    }

    # ── Position limits (2 entry sources = more positions) ──
    cfg["max_concurrent"] = 3
    cfg["max_trades"] = 7

    # ── Expiry close for Z2H ──
    cfg["expiry_close_min"] = 300

    # ── Block sr_bounce CALLs (proven loser: 18.2% WR) ──
    cfg["block_sr_bounce_call"] = True

    return cfg


def run_comparison():
    """Run Enh-Tuned vs Enh-Tuned-Full on 6 months of data."""
    print("=" * 130)
    print("  ENH-TUNED vs ENH-TUNED-FULL -- 6-Month Comparison")
    print("  Capital: Rs 2,00,000 | Dynamic lot sizing | Compounding across months")
    print("  Enh-Tuned:      21 scoring signals + fixed % trails + theta exit")
    print("  Enh-Tuned-Full: Enh-Tuned + dual-entry engine + zero-to-hero + BTST")
    print("=" * 130)

    # 6 test months (all have cached data)
    TEST_MONTHS = [
        ("2024-06-01", "2024-07-31", 2024,  7, "Jul-2024"),
        ("2024-07-01", "2024-08-31", 2024,  8, "Aug-2024"),
        ("2024-08-01", "2024-09-30", 2024,  9, "Sep-2024"),
        ("2024-09-01", "2024-10-31", 2024, 10, "Oct-2024"),
        ("2024-11-01", "2024-12-31", 2024, 12, "Dec-2024"),
        ("2024-12-01", "2025-01-31", 2025,  1, "Jan-2025"),
    ]

    # Build configs
    enh_tuned = make_v14_enh_tuned_config()
    enh_full = make_v14_enh_tuned_full_config()

    # Also test V14_Original as reference
    original_cfg = copy.deepcopy(V9_HYBRID_CONFIG)
    original_cfg["name"] = "V14_Original"

    model_configs = [
        ("V14_Original", original_cfg, "original"),
        ("Enh-Tuned",    enh_tuned,    "enh_tuned"),
        ("Enh-Full",     enh_full,     "enh_full"),
    ]

    model_names = [m[0] for m in model_configs]
    accumulated_equity = {name: CAPITAL for name in model_names}
    month_summaries = []

    for warmup_start, data_end, test_yr, test_mo, label in TEST_MONTHS:
        print(f"\n{'='*110}")
        print(f"  {label} (warmup from {warmup_start})")
        print(f"  Capital: " + " | ".join(
            f"{n}: Rs {accumulated_equity[n]:>,}" for n in model_names))
        print(f"{'='*100}")

        nifty, vix = download_range(warmup_start, data_end)
        if nifty is None:
            print(f"  SKIPPED: No data for {label}")
            continue

        # Add indicators
        nifty_ind = add_all_indicators(nifty.copy())

        # Group by date
        day_groups = {date: group for date, group in nifty_ind.groupby(nifty_ind.index.date)}
        all_dates = sorted(day_groups.keys())

        # Build daily OHLC
        vix_lookup = {}
        if vix is not None and not vix.empty:
            for idx, row in vix.iterrows():
                vix_lookup[idx.date()] = row["close"]

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

        test_start = dt.date(test_yr, test_mo, 1)
        test_dates = [d for d in all_dates if d >= test_start]
        if not test_dates:
            continue

        results = {}

        # Run V14 Original (uses base simulate_day)
        eq = accumulated_equity["V14_Original"]
        r_orig = run_model(original_cfg, daily, close_prices, day_groups, test_dates,
                           vix_lookup, daily_trend_df, test_start, starting_equity=eq)
        results["V14_Original"] = r_orig

        # Run Enhanced variants
        for name, cfg, _key in model_configs:
            if name == "V14_Original":
                continue  # Already ran
            eq = accumulated_equity[name]
            r = run_model_enhanced(cfg, daily, close_prices, day_groups, test_dates,
                                   vix_lookup, daily_trend_df, test_start, starting_equity=eq)
            results[name] = r

        # Update accumulated equity
        for name in model_names:
            if name in results:
                accumulated_equity[name] = results[name]["final_equity"]

        # Print month summary
        print(f"\n  {'Model':<16} {'Start Cap':>12} {'P&L':>12} {'Return':>8} {'End Cap':>12} "
              f"{'Trades':>7} {'WR':>6} {'Sharpe':>7} {'MaxDD':>7} {'PF':>5} {'Lots':>5}")
        print(f"  {'-'*110}")
        for name in model_names:
            if name in results:
                r = results[name]
                pf_str = f"{r['profit_factor']:.1f}" if r['profit_factor'] < 100 else "inf"
                print(f"  {name:<16} Rs{r['start_equity']:>9,} Rs{r['net_pnl']:>+9,} {r['return_pct']:>+7.1f}% "
                      f"Rs{r['final_equity']:>9,} {r['total_trades']:>5}t "
                      f"{r['win_rate']:>5.1f}% {r['sharpe']:>6.2f} {r['max_drawdown']:>6.1f}% "
                      f"{pf_str:>5} {r['avg_lots']:>4.1f}")

        month_summaries.append({
            "month": label,
            "results": {name: results.get(name, {}) for name in model_names},
        })

    # ================================================================
    # GRAND SUMMARY
    # ================================================================
    print(f"\n\n{'='*130}")
    print(f"  GRAND SUMMARY -- 6-Month Comparison (3 Variants)")
    print(f"{'='*130}")

    for name in model_names:
        all_trades = []
        for ms in month_summaries:
            r = ms["results"].get(name, {})
            all_trades.extend(r.get("all_trades", []))

        total = len(all_trades)
        wins = [t for t in all_trades if t["pnl"] > 0]
        losses = [t for t in all_trades if t["pnl"] <= 0]
        total_pnl = sum(t["pnl"] for t in all_trades)
        wr = len(wins) / total * 100 if total else 0
        gw = sum(t["pnl"] for t in wins)
        gl = abs(sum(t["pnl"] for t in losses))
        pf = gw / gl if gl > 0 else float("inf")
        avg_win = np.mean([t["pnl"] for t in wins]) if wins else 0
        avg_loss = np.mean([t["pnl"] for t in losses]) if losses else 0
        final_eq = accumulated_equity[name]
        ret_x = final_eq / CAPITAL

        print(f"\n  {name}:")
        print(f"    Final Equity:    Rs {final_eq:>12,}  ({ret_x:.1f}x return)")
        print(f"    Total P&L:       Rs {total_pnl:>+12,.0f}")
        print(f"    Trades:          {total:>8}")
        print(f"    Win Rate:        {wr:>7.1f}%")
        print(f"    Profit Factor:   {pf:>7.2f}")
        print(f"    Avg Win:         Rs {avg_win:>+10,.0f}")
        print(f"    Avg Loss:        Rs {avg_loss:>+10,.0f}")

        # BTST stats
        btst_trades = [t for t in all_trades if t.get("btst_pnl", 0) != 0]
        btst_pnl = sum(t.get("btst_pnl", 0) for t in all_trades)
        z2h_trades = [t for t in all_trades if t.get("is_zero_hero", False)]
        z2h_pnl = sum(t["pnl"] for t in z2h_trades)

        print(f"    BTST Trades:     {len(btst_trades):>5}  P&L: Rs {btst_pnl:>+10,.0f}")
        print(f"    Zero-Hero:       {len(z2h_trades):>5}  P&L: Rs {z2h_pnl:>+10,.0f}")

        # Entry type breakdown
        entry_counts = defaultdict(lambda: {"count": 0, "pnl": 0, "wins": 0})
        for t in all_trades:
            et = t.get("entry_type", "?")
            entry_counts[et]["count"] += 1
            entry_counts[et]["pnl"] += t["pnl"]
            if t["pnl"] > 0: entry_counts[et]["wins"] += 1

        print(f"\n    Entry Type Breakdown:")
        print(f"    {'Type':<25} {'Trades':>7} {'WR':>6} {'P&L':>14}")
        print(f"    {'-'*55}")
        for et, stats in sorted(entry_counts.items(), key=lambda x: x[1]["pnl"], reverse=True):
            et_wr = stats["wins"] / stats["count"] * 100 if stats["count"] else 0
            print(f"    {et:<25} {stats['count']:>5}t {et_wr:>5.1f}% Rs{stats['pnl']:>+12,.0f}")

        # Exit reason breakdown
        exit_counts = defaultdict(lambda: {"count": 0, "pnl": 0, "wins": 0})
        for t in all_trades:
            er = t.get("exit_reason", "?")
            exit_counts[er]["count"] += 1
            exit_counts[er]["pnl"] += t["pnl"]
            if t["pnl"] > 0: exit_counts[er]["wins"] += 1

        print(f"\n    Exit Reason Breakdown:")
        print(f"    {'Reason':<25} {'Trades':>7} {'WR':>6} {'P&L':>14}")
        print(f"    {'-'*55}")
        for er, stats in sorted(exit_counts.items(), key=lambda x: x[1]["pnl"], reverse=True):
            er_wr = stats["wins"] / stats["count"] * 100 if stats["count"] else 0
            print(f"    {er:<25} {stats['count']:>5}t {er_wr:>5.1f}% Rs{stats['pnl']:>+12,.0f}")

    # Month-by-month comparison table
    print(f"\n\n  {'='*120}")
    print(f"  MONTH-BY-MONTH COMPARISON")
    print(f"  {'='*120}")
    print(f"  {'Month':<10} {'Orig P&L':>11} {'Tuned P&L':>11} {'Full P&L':>11} "
          f"{'O-WR':>6} {'T-WR':>6} {'F-WR':>6} "
          f"{'O-Trd':>6} {'T-Trd':>6} {'F-Trd':>6}  Best")
    print(f"  {'-'*100}")
    totals = {n: 0 for n in model_names}
    wins_count = {n: 0 for n in model_names}
    for ms in month_summaries:
        pnls = {}
        for name in model_names:
            r = ms["results"].get(name, {})
            pnls[name] = r.get("net_pnl", 0)
            totals[name] += pnls[name]
        best_name = max(pnls, key=pnls.get)
        wins_count[best_name] += 1
        short_label = {"V14_Original": "Orig", "Enh-Tuned": "Tuned", "Enh-Full": "Full"}
        best_label = short_label.get(best_name, best_name)

        orig_r = ms["results"].get("V14_Original", {})
        tuned_r = ms["results"].get("Enh-Tuned", {})
        full_r = ms["results"].get("Enh-Full", {})

        print(f"  {ms['month']:<10} Rs{pnls.get('V14_Original', 0):>+9,} "
              f"Rs{pnls.get('Enh-Tuned', 0):>+9,} Rs{pnls.get('Enh-Full', 0):>+9,} "
              f"{orig_r.get('win_rate', 0):>5.1f}% {tuned_r.get('win_rate', 0):>5.1f}% {full_r.get('win_rate', 0):>5.1f}% "
              f"{orig_r.get('total_trades', 0):>5}t {tuned_r.get('total_trades', 0):>5}t {full_r.get('total_trades', 0):>5}t  "
              f"{best_label}")

    print(f"  {'-'*100}")
    print(f"  {'TOTAL':<10} Rs{totals.get('V14_Original', 0):>+9,} "
          f"Rs{totals.get('Enh-Tuned', 0):>+9,} Rs{totals.get('Enh-Full', 0):>+9,}")
    print(f"\n  Month wins: " + " | ".join(f"{n}: {wins_count[n]}" for n in model_names))

    print(f"\n  Final equity:")
    best_model = max(model_names, key=lambda n: accumulated_equity[n])
    for name in model_names:
        eq = accumulated_equity[name]
        ret_x = eq / CAPITAL
        marker = " *** BEST ***" if name == best_model else ""
        print(f"    {name:<16} Rs {eq:>12,}  ({ret_x:.1f}x){marker}")

    # Save all trades from best model for ML training
    best_trades = []
    for ms in month_summaries:
        r = ms["results"].get(best_model, {})
        best_trades.extend(r.get("all_trades", []))

    if best_trades:
        trades_path = DATA_DIR / "v14_full_backtest_trades.csv"
        trades_df = pd.DataFrame(best_trades)
        trades_df.to_csv(trades_path, index=False)
        print(f"\n  Saved {len(best_trades)} trades from {best_model} to {trades_path}")

    # Also save the combined trades from all models for ML training diversity
    all_model_trades = []
    for ms in month_summaries:
        for name in model_names:
            r = ms["results"].get(name, {})
            for t in r.get("all_trades", []):
                t_copy = t.copy()
                t_copy["model"] = name
                all_model_trades.append(t_copy)
    if all_model_trades:
        all_path = DATA_DIR / "v14_all_models_trades.csv"
        all_df = pd.DataFrame(all_model_trades)
        all_df.to_csv(all_path, index=False)
        print(f"  Saved {len(all_model_trades)} trades (all models) to {all_path}")

    print(f"\n{'='*130}")


if __name__ == "__main__":
    run_comparison()
