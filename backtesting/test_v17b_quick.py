"""
V17b QUICK TEST — V14 Entries + Smart Stop-Loss Only
=====================================================
Tests the hypothesis: V14's entries are optimal, but its exits are too slow
on losing trades. V17b's state-based smart stop cuts losers faster.

On April 6: V14 held losing PUT for 300 min -> Rs -6,207 loss.
V17b cut it at 205 min via rt_smart_stop -> Rs -3,827 loss (saved Rs 2,380).
Combined with the CALL winner, turned -Rs 1,470 day into +Rs 910.

This test runs ONLY the promising combo against V14 baseline.
"""

import sys
import copy
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backtesting.daywise_analysis import add_all_indicators
from backtesting.multi_month_oos_test import download_range
from backtesting.oos_june2024_test import (
    V9_HYBRID_CONFIG, CAPITAL, simulate_day, detect_market_regime,
    run_model,
)

from collections import defaultdict


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


def run_oos_test(cfg, label, verbose=True):
    """Run full OOS test."""
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

    equity = CAPITAL
    total_trades = 0
    total_wins = 0
    monthly_results = []
    exit_types = defaultdict(int)
    entry_types = defaultdict(int)

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

        for t in result.get("all_trades", []):
            entry_types[t.get("entry_type", "?")] += 1
            exit_types[t.get("exit_reason", "?")] += 1

        month_label = dt.datetime.strptime(start, "%Y-%m-%d").strftime("%b %Y")
        status = "+" if month_pnl > 0 else "-"
        monthly_results.append({
            "month": month_label, "pnl": month_pnl, "trades": result["total_trades"],
            "wr": result["win_rate"], "equity": equity,
        })

        if verbose:
            print(f"    {month_label}: Rs{month_pnl:>+12,} | {result['total_trades']:>3} trades | "
                  f"WR {result['win_rate']:.0f}% | Equity: Rs{equity:>14,.0f} [{status}]")

    return_x = equity / CAPITAL
    overall_wr = (total_wins / total_trades * 100) if total_trades > 0 else 0
    months_pos = sum(1 for m in monthly_results if m["pnl"] > 0)

    return {
        "label": label, "equity": equity, "return_x": return_x,
        "total_trades": total_trades, "overall_wr": overall_wr,
        "months_positive": months_pos, "total_months": len(monthly_results),
        "monthly": monthly_results,
        "entry_types": dict(entry_types), "exit_types": dict(exit_types),
    }


def main():
    print("=" * 120)
    print("  V17b QUICK TEST: V14 Entries + Smart Stop-Loss")
    print("  Hypothesis: Keep V14's entries, add state-based loss protection only")
    print("=" * 120)

    # A: V14 Production baseline (587.5x)
    cfg_a = copy.deepcopy(V9_HYBRID_CONFIG)
    cfg_a["use_realtime_engine"] = False
    cfg_a["use_reversal_detection"] = False
    cfg_a["reversal_flip_enabled"] = False
    cfg_a["avoid_days"] = []
    result_a = run_oos_test(cfg_a, "A: V14 Production (baseline)")

    # E: V14 entries + V17b smart stops
    cfg_e = copy.deepcopy(V9_HYBRID_CONFIG)
    cfg_e["use_realtime_engine"] = True  # Needed for state computation
    cfg_e["rt_check_every_minute"] = True
    cfg_e["rt_remove_avoid_windows"] = False  # KEEP V14's avoid windows
    cfg_e["rt_state_exit"] = True  # V17b conservative smart stops
    cfg_e["rt_entries_disabled"] = True  # NO RT entries — V14 entries only
    cfg_e["use_reversal_detection"] = False
    cfg_e["reversal_flip_enabled"] = False
    cfg_e["avoid_days"] = []
    result_e = run_oos_test(cfg_e, "E: V14 Entries + V17b Smart Stops")

    # Comparison
    print("\n" + "=" * 120)
    print("  COMPARISON: V14 vs V14 + Smart Stops")
    print("=" * 120)

    for r in [result_a, result_e]:
        print(f"\n  {r['label']}:")
        print(f"    Return: {r['return_x']:.1f}x | Trades: {r['total_trades']} | "
              f"WR: {r['overall_wr']:.1f}% | Months+: {r['months_positive']}/{r['total_months']}")
        print(f"    Final equity: Rs {r['equity']:,.0f}")

        # Exit type breakdown
        print(f"    Exit types:", end="")
        sorted_ex = sorted(r["exit_types"].items(), key=lambda x: -x[1])
        for ex, count in sorted_ex[:8]:
            print(f" {ex}={count}", end="")
        print()

    # Delta analysis
    delta_return = result_e["return_x"] - result_a["return_x"]
    delta_pct = (result_e["return_x"] / result_a["return_x"] - 1) * 100

    print(f"\n  {'='*80}")
    if result_e["return_x"] > result_a["return_x"]:
        print(f"  V17b SMART STOPS IMPROVE V14 by {delta_pct:+.1f}% ({delta_return:+.1f}x)")
        print(f"  {result_a['return_x']:.1f}x -> {result_e['return_x']:.1f}x")
        print(f"  Smart stops cut losers faster without affecting winners!")
    else:
        print(f"  V14 still better ({result_a['return_x']:.1f}x vs {result_e['return_x']:.1f}x)")
        print(f"  Smart stops hurt by {delta_pct:.1f}% -- some 'losers' were recovering trades")

    # Month-by-month delta
    print(f"\n  Month-by-month delta (E minus A):")
    for ma, me in zip(result_a["monthly"], result_e["monthly"]):
        delta = me["pnl"] - ma["pnl"]
        tag = "BETTER" if delta > 0 else "WORSE"
        print(f"    {ma['month']}: Rs{delta:>+12,} [{tag}] "
              f"(A: Rs{ma['pnl']:>+12,} | E: Rs{me['pnl']:>+12,})")

    # Which months does smart stop help vs hurt?
    better_months = sum(1 for ma, me in zip(result_a["monthly"], result_e["monthly"]) if me["pnl"] > ma["pnl"])
    worse_months = sum(1 for ma, me in zip(result_a["monthly"], result_e["monthly"]) if me["pnl"] < ma["pnl"])
    print(f"\n  Smart stops BETTER in {better_months} months, WORSE in {worse_months} months")
    print("=" * 120)


if __name__ == "__main__":
    main()
