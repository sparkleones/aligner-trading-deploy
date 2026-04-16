"""
MEGA Out-of-Sample (OOS) Comparison: V3 vs V4 across 10 DIFFERENT PERIODS (2015-2024).

Tests BOTH V3 and V4 logic on 10 diverse months spanning different years and market
conditions that were NEVER used during training/optimization.

Test Periods:
  1. Jan 2015 -- Normal market
  2. Aug 2015 -- China crash spillover, high VIX
  3. Nov 2016 -- Demonetization shock
  4. Feb 2018 -- Global VIX spike (Volmageddon)
  5. Sep 2019 -- Corporate tax cut rally
  6. Mar 2020 -- COVID crash (extreme VIX)
  7. Oct 2020 -- Post-COVID recovery
  8. Jun 2021 -- Bull market
  9. Jun 2022 -- Bear market (rate hikes)
 10. Dec 2024 -- Recent (already tested)

Capital: Rs 200,000 | Data: Yahoo Finance (^NSEI + ^INDIAVIX)
"""

import json
import math
import sys
import traceback
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import shared functions from the V4 engine
from backtesting.paper_trading_real_data import (
    generate_intraday_path,
    sr_multi_method,
    bs_premium,
    get_strike_and_type,
    get_lot_count,
    check_trade_exit,
    compute_composite_scores,
    LOT_SIZE,
    BROKERAGE,
    STRIKE_INTERVAL,
    CAPITAL,
    TOTAL_BARS,
    MAX_TRADES_PER_DAY,
    MAX_CONCURRENT,
    MIN_CONFIDENCE,
    TRAIL_PCT,
    PUT_MAX_HOLD,
    CALL_MAX_HOLD,
)

# Import V3 composite scoring and entry detection from oos_comparison
from backtesting.oos_comparison import (
    compute_composite_scores_v3,
    detect_entries_v3,
    detect_entries_v4,
    simulate_day_oos,
    run_backtest,
)


# ===========================================================================
# TEST PERIOD DEFINITIONS
# ===========================================================================

TEST_PERIODS = [
    {
        "label": "Jan 2015",
        "start": "2015-01-01",
        "end": "2015-02-01",
        "description": "Normal market",
        "default_vix": 14.0,
        "high_vix": False,
    },
    {
        "label": "Aug 2015",
        "start": "2015-08-01",
        "end": "2015-09-01",
        "description": "China crash spillover",
        "default_vix": 22.0,
        "high_vix": True,
    },
    {
        "label": "Nov 2016",
        "start": "2016-11-01",
        "end": "2016-12-01",
        "description": "Demonetization shock",
        "default_vix": 18.0,
        "high_vix": True,
    },
    {
        "label": "Feb 2018",
        "start": "2018-02-01",
        "end": "2018-03-01",
        "description": "Volmageddon VIX spike",
        "default_vix": 22.0,
        "high_vix": True,
    },
    {
        "label": "Sep 2019",
        "start": "2019-09-01",
        "end": "2019-10-01",
        "description": "Corporate tax cut rally",
        "default_vix": 15.0,
        "high_vix": False,
    },
    {
        "label": "Mar 2020",
        "start": "2020-03-01",
        "end": "2020-04-01",
        "description": "COVID crash",
        "default_vix": 45.0,
        "high_vix": True,
    },
    {
        "label": "Oct 2020",
        "start": "2020-10-01",
        "end": "2020-11-01",
        "description": "Post-COVID recovery",
        "default_vix": 20.0,
        "high_vix": False,
    },
    {
        "label": "Jun 2021",
        "start": "2021-06-01",
        "end": "2021-07-01",
        "description": "Bull market",
        "default_vix": 14.0,
        "high_vix": False,
    },
    {
        "label": "Jun 2022",
        "start": "2022-06-01",
        "end": "2022-07-01",
        "description": "Bear market (rate hikes)",
        "default_vix": 20.0,
        "high_vix": True,
    },
    {
        "label": "Dec 2024",
        "start": "2024-12-01",
        "end": "2025-01-01",
        "description": "Recent (already tested)",
        "default_vix": 14.0,
        "high_vix": False,
    },
]


# ===========================================================================
# DATA DOWNLOAD (adapted for older periods with VIX fallback)
# ===========================================================================

def download_period_data(start, end, default_vix=14.0, label=""):
    """Download real NIFTY + VIX data from Yahoo Finance for a given period.

    Handles:
    - MultiIndex columns from yfinance
    - Missing VIX data (uses default_vix)
    - Older date ranges where data may be sparse
    """
    import yfinance as yf

    # Need 90-day warmup for SMA50
    warmup_start = (pd.Timestamp(start) - pd.Timedelta(days=120)).strftime("%Y-%m-%d")

    print(f"  Downloading NIFTY data for {label} ({start} to {end})...")
    nifty = yf.download("^NSEI", start=warmup_start, end=end, interval="1d", progress=False)

    if nifty.empty:
        raise ValueError(f"No NIFTY data available for {label} ({start} to {end})")

    # Handle MultiIndex columns (yfinance quirk)
    if isinstance(nifty.columns, pd.MultiIndex):
        nifty.columns = nifty.columns.get_level_values(0)

    # Try to get VIX data - may not exist for older periods
    try:
        vix_data = yf.download("^INDIAVIX", start=warmup_start, end=end,
                               interval="1d", progress=False)
        if isinstance(vix_data.columns, pd.MultiIndex):
            vix_data.columns = vix_data.columns.get_level_values(0)
        if not vix_data.empty and "Close" in vix_data.columns:
            nifty["VIX"] = vix_data["Close"]
            nifty["VIX"] = nifty["VIX"].ffill().bfill().fillna(default_vix)
        else:
            nifty["VIX"] = default_vix
    except Exception:
        nifty["VIX"] = default_vix

    # Ensure VIX has no NaN
    nifty["VIX"] = nifty["VIX"].fillna(default_vix)

    # Compute all derived columns
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

    # Expiry day: Thursday for historical periods, Tuesday after Nov 2025
    nifty["IsExpiry"] = nifty.index.map(
        lambda d: d.strftime("%A") == ("Tuesday" if d >= pd.Timestamp("2025-11-01") else "Thursday")
    )

    dte_values = []
    for idx in nifty.index:
        current_dow = idx.weekday()
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

    # Trim warmup rows (need SMA50 to be valid)
    valid_start = nifty["SMA50"].first_valid_index()
    if valid_start is not None:
        nifty = nifty.loc[valid_start:]

    # Trim to requested date range
    nifty = nifty.loc[start:]

    if nifty.empty:
        raise ValueError(f"No trading days in {label} after trimming warmup period")

    nifty_low = nifty["Low"].min()
    nifty_high = nifty["High"].max()
    vix_low = nifty["VIX"].min()
    vix_high = nifty["VIX"].max()

    print(f"    Loaded: {len(nifty)} trading days | "
          f"NIFTY {nifty_low:.0f}-{nifty_high:.0f} | "
          f"VIX {vix_low:.1f}-{vix_high:.1f}")

    return nifty


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    print("=" * 100)
    print("  MEGA OOS TEST: V3 vs V4 across 10 DIFFERENT PERIODS (2015-2024)")
    print("  Data: Yahoo Finance (^NSEI + ^INDIAVIX)")
    print(f"  Capital: Rs {CAPITAL:,} per period | Strategy: Full Ensemble")
    print("=" * 100)

    results = []
    failed_periods = []
    succeeded_periods = []

    for pi, period in enumerate(TEST_PERIODS):
        label = period["label"]
        desc = period["description"]
        print(f"\n{'='*100}")
        print(f"  [{pi+1}/10] {label} -- {desc}")
        print(f"{'='*100}")

        try:
            # Download data for this period
            nifty = download_period_data(
                start=period["start"],
                end=period["end"],
                default_vix=period["default_vix"],
                label=label,
            )

            num_days = len(nifty)
            nifty_low = nifty["Low"].min()
            nifty_high = nifty["High"].max()
            vix_low = nifty["VIX"].min()
            vix_high = nifty["VIX"].max()

            # Run V3 backtest
            print(f"    Running V3 (Original)...")
            v3 = run_backtest(nifty, version="V3")
            print(f"      V3: {v3['total_trades']} trades, "
                  f"P&L Rs {v3['total_pnl']:+,.0f} ({v3['return_pct']:+.1f}%), "
                  f"Sharpe {v3['sharpe']:.2f}, DD {v3['max_dd_pct']:.1f}%")

            # Run V4 backtest
            print(f"    Running V4 (Optimized)...")
            v4 = run_backtest(nifty, version="V4")
            print(f"      V4: {v4['total_trades']} trades, "
                  f"P&L Rs {v4['total_pnl']:+,.0f} ({v4['return_pct']:+.1f}%), "
                  f"Sharpe {v4['sharpe']:.2f}, DD {v4['max_dd_pct']:.1f}%")

            # Determine winner for this period (based on return)
            if v4["return_pct"] > v3["return_pct"]:
                winner = "V4"
            elif v3["return_pct"] > v4["return_pct"]:
                winner = "V3"
            else:
                winner = "TIE"

            result = {
                "label": label,
                "description": desc,
                "start": period["start"],
                "end": period["end"],
                "trading_days": num_days,
                "nifty_low": round(float(nifty_low), 0),
                "nifty_high": round(float(nifty_high), 0),
                "vix_low": round(float(vix_low), 1),
                "vix_high": round(float(vix_high), 1),
                "v3_return": v3["return_pct"],
                "v4_return": v4["return_pct"],
                "v3_sharpe": v3["sharpe"],
                "v4_sharpe": v4["sharpe"],
                "v3_dd": v3["max_dd_pct"],
                "v4_dd": v4["max_dd_pct"],
                "v3_pnl": v3["total_pnl"],
                "v4_pnl": v4["total_pnl"],
                "v3_trades": v3["total_trades"],
                "v4_trades": v4["total_trades"],
                "v3_win_rate": v3["win_rate"],
                "v4_win_rate": v4["win_rate"],
                "v3_pf": v3["profit_factor"],
                "v4_pf": v4["profit_factor"],
                "winner": winner,
                "status": "SUCCESS",
            }
            results.append(result)
            succeeded_periods.append(label)
            print(f"    >>> Winner: {winner}")

        except Exception as e:
            print(f"    FAILED: {e}")
            traceback.print_exc()
            results.append({
                "label": label,
                "description": desc,
                "start": period["start"],
                "end": period["end"],
                "status": "FAILED",
                "error": str(e),
            })
            failed_periods.append(label)

    # =========================================================================
    # FINAL MEGA COMPARISON TABLE
    # =========================================================================
    print()
    print("=" * 130)
    print("  MEGA OOS TEST: V3 vs V4 across 10 DIFFERENT PERIODS (2015-2024)")
    print("=" * 130)

    # Header
    header = (f"  {'Period':<14s} {'NIFTY Range':<16s} {'VIX Range':<12s} "
              f"{'V3 Return':>10s} {'V4 Return':>10s} "
              f"{'V3 Sharpe':>10s} {'V4 Sharpe':>10s} "
              f"{'V3 DD':>7s} {'V4 DD':>7s} "
              f"{'V3 WR':>6s} {'V4 WR':>6s} "
              f"{'Winner':>8s}")
    sep_line = "-" * 130
    print(f"  {sep_line}")
    print(header)
    print(f"  {sep_line}")

    successful_results = [r for r in results if r["status"] == "SUCCESS"]

    for r in results:
        if r["status"] == "FAILED":
            print(f"  {r['label']:<14s} {'FAILED -- ' + r.get('error', 'unknown')[:60]}")
            continue

        nifty_range = f"{r['nifty_low']:.0f}-{r['nifty_high']:.0f}"
        vix_range = f"{r['vix_low']:.1f}-{r['vix_high']:.1f}"
        v3_ret = f"{r['v3_return']:+.1f}%"
        v4_ret = f"{r['v4_return']:+.1f}%"
        v3_sh = f"{r['v3_sharpe']:.2f}"
        v4_sh = f"{r['v4_sharpe']:.2f}"
        v3_dd = f"{r['v3_dd']:.1f}%"
        v4_dd = f"{r['v4_dd']:.1f}%"
        v3_wr = f"{r['v3_win_rate']:.0f}%"
        v4_wr = f"{r['v4_win_rate']:.0f}%"
        w = r["winner"]

        print(f"  {r['label']:<14s} {nifty_range:<16s} {vix_range:<12s} "
              f"{v3_ret:>10s} {v4_ret:>10s} "
              f"{v3_sh:>10s} {v4_sh:>10s} "
              f"{v3_dd:>7s} {v4_dd:>7s} "
              f"{v3_wr:>6s} {v4_wr:>6s} "
              f"{w:>8s}")

    print(f"  {sep_line}")

    # =========================================================================
    # AGGREGATE SCOREBOARD
    # =========================================================================
    if successful_results:
        v3_wins = sum(1 for r in successful_results if r["winner"] == "V3")
        v4_wins = sum(1 for r in successful_results if r["winner"] == "V4")
        ties = sum(1 for r in successful_results if r["winner"] == "TIE")
        n = len(successful_results)

        v3_avg_ret = np.mean([r["v3_return"] for r in successful_results])
        v4_avg_ret = np.mean([r["v4_return"] for r in successful_results])
        v3_avg_sharpe = np.mean([r["v3_sharpe"] for r in successful_results])
        v4_avg_sharpe = np.mean([r["v4_sharpe"] for r in successful_results])
        v3_avg_dd = np.mean([r["v3_dd"] for r in successful_results])
        v4_avg_dd = np.mean([r["v4_dd"] for r in successful_results])
        v3_total_pnl = sum(r["v3_pnl"] for r in successful_results)
        v4_total_pnl = sum(r["v4_pnl"] for r in successful_results)
        v3_avg_wr = np.mean([r["v3_win_rate"] for r in successful_results])
        v4_avg_wr = np.mean([r["v4_win_rate"] for r in successful_results])
        v3_avg_pf = np.mean([r["v3_pf"] for r in successful_results])
        v4_avg_pf = np.mean([r["v4_pf"] for r in successful_results])

        print()
        print(f"  AGGREGATE SCOREBOARD ({n} periods tested):")
        print(f"  {sep_line}")
        print(f"    V3 wins: {v3_wins}/{n}")
        print(f"    V4 wins: {v4_wins}/{n}")
        print(f"    Ties:    {ties}/{n}")
        print()
        print(f"    V3 avg return: {v3_avg_ret:+.1f}%       V4 avg return: {v4_avg_ret:+.1f}%")
        print(f"    V3 avg Sharpe: {v3_avg_sharpe:.2f}         V4 avg Sharpe: {v4_avg_sharpe:.2f}")
        print(f"    V3 avg DD:     {v3_avg_dd:.1f}%          V4 avg DD:     {v4_avg_dd:.1f}%")
        print(f"    V3 avg WR:     {v3_avg_wr:.0f}%            V4 avg WR:     {v4_avg_wr:.0f}%")
        print(f"    V3 avg PF:     {v3_avg_pf:.2f}          V4 avg PF:     {v4_avg_pf:.2f}")
        print(f"    V3 total P&L:  Rs {v3_total_pnl:>+12,.0f}    V4 total P&L:  Rs {v4_total_pnl:>+12,.0f}")

        # Detailed metric-by-metric comparison
        print()
        print(f"  METRIC-BY-METRIC VERDICT:")
        print(f"  {sep_line}")
        metric_verdicts = []
        metrics = [
            ("Avg Return", v3_avg_ret, v4_avg_ret, True),
            ("Avg Sharpe", v3_avg_sharpe, v4_avg_sharpe, True),
            ("Avg Max DD", v3_avg_dd, v4_avg_dd, False),
            ("Avg Win Rate", v3_avg_wr, v4_avg_wr, True),
            ("Avg Profit Factor", v3_avg_pf, v4_avg_pf, True),
            ("Total P&L", v3_total_pnl, v4_total_pnl, True),
            ("Period Wins", v3_wins, v4_wins, True),
        ]
        v3_metric_wins = 0
        v4_metric_wins = 0
        for name, v3v, v4v, higher_better in metrics:
            if higher_better:
                w = "V3" if v3v > v4v else ("V4" if v4v > v3v else "TIE")
            else:
                w = "V3" if v3v < v4v else ("V4" if v4v < v3v else "TIE")
            if w == "V3": v3_metric_wins += 1
            elif w == "V4": v4_metric_wins += 1
            print(f"    {name:<20s}: V3={v3v:>+10.1f}  V4={v4v:>+10.1f}  --> {w}")
            metric_verdicts.append({"metric": name, "v3": round(v3v, 2),
                                    "v4": round(v4v, 2), "winner": w})

        # Overall verdict
        if v4_metric_wins > v3_metric_wins:
            overall = "V4"
        elif v3_metric_wins > v4_metric_wins:
            overall = "V3"
        else:
            overall = "TIE"

        print()
        print(f"  {sep_line}")
        print(f"  VERDICT: {overall} is the superior model across diverse market conditions")
        print(f"           ({v3_metric_wins} metrics favor V3, {v4_metric_wins} metrics favor V4)")
        if failed_periods:
            print(f"  NOTE: {len(failed_periods)} period(s) failed to download: {', '.join(failed_periods)}")
        if succeeded_periods:
            print(f"  Succeeded: {', '.join(succeeded_periods)}")
        print("=" * 130)

    else:
        print("\n  ERROR: No periods completed successfully. Cannot compute aggregate.")
        overall = "INCONCLUSIVE"

    # =========================================================================
    # SAVE RESULTS TO JSON
    # =========================================================================
    output = {
        "test_date": datetime.now().isoformat(),
        "test_type": "mega_oos_comparison",
        "num_periods": len(TEST_PERIODS),
        "num_succeeded": len(succeeded_periods),
        "num_failed": len(failed_periods),
        "capital_per_period": CAPITAL,
        "overall_verdict": overall,
        "periods": results,
        "aggregate": {
            "v3_wins": int(v3_wins) if successful_results else 0,
            "v4_wins": int(v4_wins) if successful_results else 0,
            "ties": int(ties) if successful_results else 0,
            "v3_avg_return": round(v3_avg_ret, 2) if successful_results else 0,
            "v4_avg_return": round(v4_avg_ret, 2) if successful_results else 0,
            "v3_avg_sharpe": round(v3_avg_sharpe, 2) if successful_results else 0,
            "v4_avg_sharpe": round(v4_avg_sharpe, 2) if successful_results else 0,
            "v3_avg_dd": round(v3_avg_dd, 2) if successful_results else 0,
            "v4_avg_dd": round(v4_avg_dd, 2) if successful_results else 0,
            "v3_total_pnl": round(v3_total_pnl, 0) if successful_results else 0,
            "v4_total_pnl": round(v4_total_pnl, 0) if successful_results else 0,
        } if successful_results else {},
        "failed_periods": failed_periods,
        "succeeded_periods": succeeded_periods,
    }

    output_path = project_root / "data" / "oos_mega_comparison.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to: {output_path}")


if __name__ == "__main__":
    main()
