"""
Optimal Intraday Entry Timing Study for NIFTY Options.

Downloads 6 months of NIFTY and India VIX data, then simulates option trades
entered at different times of day to find the best entry windows for
BUY_CALL and BUY_PUT strategies.

Studies:
  1. Entry window analysis (first hour, late morning, midday, early afternoon, late afternoon)
  2. Gap-up vs gap-down effects on calls vs puts
  3. Opening Range Breakout (ORB) effectiveness
"""

import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

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
DTE = 2  # days to expiry for option pricing
RISK_FREE_RATE = 0.07

# Entry windows mapped to 15-min bars within trading day (9:15 - 3:30)
ENTRY_WINDOWS = {
    "First Hour (9:15-10:00)":      {"bars": (1, 3),   "label": "first_hour"},
    "Late Morning (10:00-11:15)":   {"bars": (4, 8),   "label": "late_morning"},
    "Midday (11:15-12:30)":         {"bars": (9, 13),  "label": "midday"},
    "Early Afternoon (12:30-1:45)": {"bars": (14, 18), "label": "early_afternoon"},
    "Late Afternoon (1:45-3:30)":   {"bars": (19, 25), "label": "late_afternoon"},
}

# Number of 15-min bars in trading day (9:15 to 3:30 = 6h15m = 25 bars)
TOTAL_BARS = 25


def compute_transaction_costs(premium: float, lots: int = 1) -> float:
    """
    Compute round-trip Zerodha transaction costs for options.
    Zerodha charges flat Rs 20 per executed order for options.
    """
    qty = lots * LOT_SIZE
    turnover = premium * qty

    # Zerodha brokerage: Rs 20 per order, 2 legs (buy + sell)
    brokerage = 20.0 * 2

    # STT on sell side only
    stt = turnover * STT_RATES["options_sell"]

    # Exchange charges
    exchange_charge = turnover * NSE_TRANSACTION_CHARGE

    # SEBI turnover fee
    sebi_fee = turnover * SEBI_TURNOVER_FEE

    # Stamp duty on buy side
    stamp = turnover * STAMP_DUTY_BUY

    # GST on brokerage + exchange charges
    gst = (brokerage + exchange_charge) * GST_RATE

    total = brokerage + stt + exchange_charge + sebi_fee + stamp + gst
    return round(total, 2)


def download_data():
    """Download NIFTY and India VIX data using yfinance."""
    import yfinance as yf

    print("=" * 70)
    print("DOWNLOADING MARKET DATA")
    print("=" * 70)

    start_date = "2025-10-01"
    end_date = "2026-04-05"

    # Download NIFTY daily data
    print(f"\nDownloading NIFTY (^NSEI) daily data: {start_date} to {end_date}")
    nifty = yf.download("^NSEI", start=start_date, end=end_date, interval="1d", progress=False)

    # Download India VIX daily data
    print(f"Downloading India VIX (^INDIAVIX): {start_date} to {end_date}")
    vix = yf.download("^INDIAVIX", start=start_date, end=end_date, interval="1d", progress=False)

    # Handle multi-level columns from yfinance
    if isinstance(nifty.columns, pd.MultiIndex):
        nifty.columns = nifty.columns.get_level_values(0)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)

    print(f"\nNIFTY data: {len(nifty)} trading days")
    print(f"India VIX data: {len(vix)} trading days")

    if len(nifty) == 0:
        print("WARNING: No NIFTY data downloaded. Check yfinance and network.")
        sys.exit(1)

    # Merge VIX with NIFTY on date
    nifty = nifty[["Open", "High", "Low", "Close", "Volume"]].copy()
    nifty.columns = ["open", "high", "low", "close", "volume"]

    if len(vix) > 0:
        vix_close = vix[["Close"]].copy()
        vix_close.columns = ["vix"]
        nifty = nifty.join(vix_close, how="left")
        nifty["vix"] = nifty["vix"].ffill().fillna(14.0)
    else:
        print("WARNING: No VIX data available. Using default VIX=14.")
        nifty["vix"] = 14.0

    # Compute previous close for gap analysis
    nifty["prev_close"] = nifty["close"].shift(1)
    nifty["gap_pct"] = (nifty["open"] - nifty["prev_close"]) / nifty["prev_close"] * 100

    # Drop first row (no previous close)
    nifty = nifty.dropna(subset=["prev_close"])

    print(f"Final dataset: {len(nifty)} trading days with VIX data")
    print(f"NIFTY range: {nifty['low'].min():.0f} - {nifty['high'].max():.0f}")
    print(f"VIX range: {nifty['vix'].min():.1f} - {nifty['vix'].max():.1f}")

    return nifty


def simulate_intraday_bars(row):
    """
    Simulate 25 intraday 15-min bars from daily OHLC using a realistic
    intraday volume/price profile.

    Indian market intraday pattern:
    - High volume/volatility in first 30 min (gap fills, institutional orders)
    - Quieter midday
    - Volume picks up after 2 PM (FII flow, futures rollovers)
    - Final 30 min: theta decay, sharp moves

    Returns array of 25 simulated bar close prices.
    """
    o, h, l, c = row["open"], row["high"], row["low"], row["close"]
    daily_range = h - l
    if daily_range == 0:
        return np.full(TOTAL_BARS, c)

    # Intraday price profile weights (how price evolves through the day)
    # Based on typical Indian market intraday patterns
    np.random.seed(int(abs(o * 100)) % (2**31))

    # Generate a random walk biased toward close
    raw = np.random.randn(TOTAL_BARS)
    raw = np.cumsum(raw)

    # Normalize to go from open to close
    raw = raw - raw[0]  # start at 0
    if raw[-1] != 0:
        # Scale so the last bar ends at close
        target_move = c - o
        raw = raw / raw[-1] * target_move
    raw = raw + o  # shift to start at open

    # Add intraday volatility envelope (touch high and low)
    # Find where high and low should occur
    # High tends to occur in first half for bearish days, second half for bullish
    if c > o:  # bullish day
        high_bar = np.random.randint(TOTAL_BARS // 2, TOTAL_BARS)
        low_bar = np.random.randint(0, TOTAL_BARS // 2)
    else:  # bearish day
        high_bar = np.random.randint(0, TOTAL_BARS // 2)
        low_bar = np.random.randint(TOTAL_BARS // 2, TOTAL_BARS)

    # Ensure we touch high and low
    raw[high_bar] = h
    raw[low_bar] = l
    raw[-1] = c
    raw[0] = o

    # Smooth with interpolation
    from scipy.interpolate import interp1d
    key_points = sorted(set([0, low_bar, high_bar, TOTAL_BARS - 1]))
    key_values = [raw[i] for i in key_points]
    f = interp1d(key_points, key_values, kind="linear")
    smooth = f(np.arange(TOTAL_BARS))

    # Add small noise
    noise = np.random.randn(TOTAL_BARS) * daily_range * 0.01
    smooth = smooth + noise

    # Enforce constraints: all within [L, H], first=O, last=C
    smooth = np.clip(smooth, l, h)
    smooth[0] = o
    smooth[-1] = c

    return smooth


def price_at_bar(spot_entry, spot_exit, vix_val, action):
    """
    Price an option trade: enter at spot_entry, exit at spot_exit.
    ATM strike based on entry spot.

    Returns PnL per lot in rupees.
    """
    strike = round(spot_entry / STRIKE_INTERVAL) * STRIKE_INTERVAL
    opt_type = "CE" if action == "BUY_CALL" else "PE"

    # Entry price
    entry_result = price_option(
        spot=spot_entry,
        strike=strike,
        dte_days=DTE,
        vix=vix_val,
        option_type=opt_type,
        r=RISK_FREE_RATE,
    )
    entry_premium = entry_result["premium"]

    # Exit price (at close, DTE reduced by fraction of day elapsed)
    exit_result = price_option(
        spot=spot_exit,
        strike=strike,
        dte_days=max(DTE - 0.8, 0.1),  # most of the day consumed
        vix=vix_val,
        option_type=opt_type,
        r=RISK_FREE_RATE,
    )
    exit_premium = exit_result["premium"]

    # PnL
    pnl_per_unit = exit_premium - entry_premium
    pnl_per_lot = pnl_per_unit * LOT_SIZE

    # Transaction costs
    avg_premium = (entry_premium + exit_premium) / 2
    costs = compute_transaction_costs(avg_premium, lots=1)

    net_pnl = pnl_per_lot - costs
    return net_pnl, entry_premium, exit_premium, costs


def analyze_entry_windows(nifty_df):
    """Analyze PnL for each entry window and action type."""
    print("\n" + "=" * 70)
    print("ENTRY WINDOW ANALYSIS")
    print("=" * 70)

    results = {}

    for window_name, window_info in ENTRY_WINDOWS.items():
        bar_start, bar_end = window_info["bars"]
        label = window_info["label"]
        results[label] = {"BUY_CALL": [], "BUY_PUT": []}

        for idx, row in nifty_df.iterrows():
            bars = simulate_intraday_bars(row)
            spot_close = row["close"]
            vix_val = row["vix"]

            # Entry at the midpoint of the window
            entry_bar = (bar_start + bar_end) // 2
            entry_bar = min(entry_bar, TOTAL_BARS - 1)
            spot_entry = bars[entry_bar]

            for action in ["BUY_CALL", "BUY_PUT"]:
                pnl, entry_p, exit_p, costs = price_at_bar(
                    spot_entry, spot_close, vix_val, action
                )
                results[label][action].append({
                    "date": str(idx.date()) if hasattr(idx, 'date') else str(idx),
                    "pnl": pnl,
                    "entry_premium": entry_p,
                    "exit_premium": exit_p,
                    "costs": costs,
                    "spot_entry": spot_entry,
                    "spot_exit": spot_close,
                })

    # Print summary table
    print(f"\n{'Window':<35} {'Action':<10} {'Trades':>6} {'Win%':>7} {'Avg PnL':>10} {'Total PnL':>12} {'Sharpe':>8}")
    print("-" * 95)

    summary = {}
    for window_name, window_info in ENTRY_WINDOWS.items():
        label = window_info["label"]
        for action in ["BUY_CALL", "BUY_PUT"]:
            trades = results[label][action]
            pnls = [t["pnl"] for t in trades]
            n = len(pnls)
            if n == 0:
                continue

            wins = sum(1 for p in pnls if p > 0)
            win_rate = wins / n * 100
            avg_pnl = np.mean(pnls)
            total_pnl = np.sum(pnls)
            sharpe = np.mean(pnls) / np.std(pnls) * np.sqrt(252) if np.std(pnls) > 0 else 0

            print(f"{window_name:<35} {action:<10} {n:>6} {win_rate:>6.1f}% {avg_pnl:>10.0f} {total_pnl:>12.0f} {sharpe:>8.2f}")

            summary[(label, action)] = {
                "window": window_name,
                "trades": n,
                "win_rate": round(win_rate, 1),
                "avg_pnl": round(avg_pnl, 2),
                "total_pnl": round(total_pnl, 2),
                "sharpe": round(sharpe, 2),
                "max_win": round(max(pnls), 2),
                "max_loss": round(min(pnls), 2),
                "profit_factor": round(
                    abs(sum(p for p in pnls if p > 0)) / abs(sum(p for p in pnls if p < 0))
                    if sum(p for p in pnls if p < 0) != 0 else 0, 2
                ),
            }

    # Find best window for each action
    best = {}
    for action in ["BUY_CALL", "BUY_PUT"]:
        action_results = {k: v for k, v in summary.items() if k[1] == action}
        if action_results:
            # Best by Sharpe ratio
            best_key = max(action_results.keys(), key=lambda k: action_results[k]["sharpe"])
            best[action] = {
                "window": action_results[best_key]["window"],
                "label": best_key[0],
                "sharpe": action_results[best_key]["sharpe"],
                "win_rate": action_results[best_key]["win_rate"],
                "avg_pnl": action_results[best_key]["avg_pnl"],
            }

    print(f"\n{'BEST ENTRY WINDOWS':>35}")
    print("-" * 50)
    for action, info in best.items():
        print(f"  {action}: {info['window']}")
        print(f"    Sharpe={info['sharpe']:.2f}, WinRate={info['win_rate']:.1f}%, AvgPnL={info['avg_pnl']:.0f}")

    return results, summary, best


def analyze_first_30min_wait(nifty_df):
    """
    Does waiting for the first 30 min (2 bars) to see direction improve win rate?
    Compare: entering at bar 1 (9:15) vs entering at bar 3 (9:45) after seeing direction.
    """
    print("\n" + "=" * 70)
    print("FIRST 30-MIN WAIT ANALYSIS")
    print("Does waiting for the first 30 min to see direction improve win rate?")
    print("=" * 70)

    strategies = {
        "Immediate Entry (Bar 1)": 0,
        "After 30 Min (Bar 3)": 2,
        "Direction-Based (Bar 3)": "direction",
    }

    strat_results = {}

    for strat_name, entry_spec in strategies.items():
        pnls_call = []
        pnls_put = []

        for idx, row in nifty_df.iterrows():
            bars = simulate_intraday_bars(row)
            vix_val = row["vix"]
            spot_close = row["close"]

            if entry_spec == "direction":
                # After 30 min, if price is UP from open, BUY_CALL; if DOWN, BUY_PUT
                spot_after_30 = bars[2]
                direction = spot_after_30 - bars[0]

                if direction > 0:
                    # Trend is up, buy call
                    pnl, _, _, _ = price_at_bar(spot_after_30, spot_close, vix_val, "BUY_CALL")
                    pnls_call.append(pnl)
                else:
                    # Trend is down, buy put
                    pnl, _, _, _ = price_at_bar(spot_after_30, spot_close, vix_val, "BUY_PUT")
                    pnls_put.append(pnl)
            else:
                spot_entry = bars[entry_spec]
                for action, pnl_list in [("BUY_CALL", pnls_call), ("BUY_PUT", pnls_put)]:
                    pnl, _, _, _ = price_at_bar(spot_entry, spot_close, vix_val, action)
                    pnl_list.append(pnl)

        # Combined results
        all_pnls = pnls_call + pnls_put
        wins = sum(1 for p in all_pnls if p > 0)
        wr = wins / len(all_pnls) * 100 if all_pnls else 0
        avg = np.mean(all_pnls) if all_pnls else 0
        total = np.sum(all_pnls) if all_pnls else 0

        strat_results[strat_name] = {
            "trades": len(all_pnls),
            "win_rate": round(wr, 1),
            "avg_pnl": round(avg, 2),
            "total_pnl": round(total, 2),
            "call_trades": len(pnls_call),
            "put_trades": len(pnls_put),
            "call_avg": round(np.mean(pnls_call), 2) if pnls_call else 0,
            "put_avg": round(np.mean(pnls_put), 2) if pnls_put else 0,
        }

    print(f"\n{'Strategy':<30} {'Trades':>6} {'Win%':>7} {'Avg PnL':>10} {'Total PnL':>12}")
    print("-" * 70)
    for name, s in strat_results.items():
        print(f"{name:<30} {s['trades']:>6} {s['win_rate']:>6.1f}% {s['avg_pnl']:>10.0f} {s['total_pnl']:>12.0f}")

    # Determine if waiting helps
    imm = strat_results.get("Immediate Entry (Bar 1)", {})
    wait = strat_results.get("After 30 Min (Bar 3)", {})
    dirn = strat_results.get("Direction-Based (Bar 3)", {})

    best_strat = max(strat_results.keys(), key=lambda k: strat_results[k]["avg_pnl"])
    print(f"\n  Best strategy: {best_strat}")
    print(f"  Waiting helps: {'Yes' if wait.get('avg_pnl', 0) > imm.get('avg_pnl', 0) else 'No'}")
    print(f"  Direction-based helps: {'Yes' if dirn.get('avg_pnl', 0) > imm.get('avg_pnl', 0) else 'No'}")

    return strat_results


def analyze_gap_effect(nifty_df):
    """
    Analyze: Is entering after a gap-up vs gap-down better for calls vs puts?
    """
    print("\n" + "=" * 70)
    print("GAP ANALYSIS")
    print("Gap-up vs Gap-down effect on BUY_CALL and BUY_PUT")
    print("=" * 70)

    gap_threshold = 0.3  # 0.3% gap threshold

    categories = {
        "Gap Up (>0.3%)": lambda g: g > gap_threshold,
        "Gap Down (<-0.3%)": lambda g: g < -gap_threshold,
        "Flat Open (|gap| <= 0.3%)": lambda g: abs(g) <= gap_threshold,
    }

    gap_results = {}

    for cat_name, condition in categories.items():
        gap_days = nifty_df[nifty_df["gap_pct"].apply(condition)]
        gap_results[cat_name] = {}

        for action in ["BUY_CALL", "BUY_PUT"]:
            pnls = []
            for idx, row in gap_days.iterrows():
                bars = simulate_intraday_bars(row)
                # Enter after first 30 min (bar 3)
                spot_entry = bars[2]
                spot_close = row["close"]
                vix_val = row["vix"]

                pnl, _, _, _ = price_at_bar(spot_entry, spot_close, vix_val, action)
                pnls.append(pnl)

            wins = sum(1 for p in pnls if p > 0) if pnls else 0
            wr = wins / len(pnls) * 100 if pnls else 0

            gap_results[cat_name][action] = {
                "trades": len(pnls),
                "win_rate": round(wr, 1),
                "avg_pnl": round(np.mean(pnls), 2) if pnls else 0,
                "total_pnl": round(np.sum(pnls), 2) if pnls else 0,
            }

    print(f"\n{'Gap Type':<25} {'Action':<10} {'Days':>5} {'Win%':>7} {'Avg PnL':>10} {'Total PnL':>12}")
    print("-" * 75)
    for cat_name, actions in gap_results.items():
        for action, stats in actions.items():
            print(f"{cat_name:<25} {action:<10} {stats['trades']:>5} {stats['win_rate']:>6.1f}% {stats['avg_pnl']:>10.0f} {stats['total_pnl']:>12.0f}")

    # Derive gap rules
    gap_rules = {}

    # Best action after gap up
    gu = gap_results.get("Gap Up (>0.3%)", {})
    if gu:
        gu_best = max(gu.keys(), key=lambda a: gu[a]["avg_pnl"])
        gap_rules["gap_up_best_action"] = gu_best
        gap_rules["gap_up_stats"] = gu[gu_best]

    # Best action after gap down
    gd = gap_results.get("Gap Down (<-0.3%)", {})
    if gd:
        gd_best = max(gd.keys(), key=lambda a: gd[a]["avg_pnl"])
        gap_rules["gap_down_best_action"] = gd_best
        gap_rules["gap_down_stats"] = gd[gd_best]

    # Flat open
    fl = gap_results.get("Flat Open (|gap| <= 0.3%)", {})
    if fl:
        fl_best = max(fl.keys(), key=lambda a: fl[a]["avg_pnl"])
        gap_rules["flat_open_best_action"] = fl_best
        gap_rules["flat_open_stats"] = fl[fl_best]

    print(f"\n  Gap-Up -> Best: {gap_rules.get('gap_up_best_action', 'N/A')}")
    print(f"  Gap-Down -> Best: {gap_rules.get('gap_down_best_action', 'N/A')}")
    print(f"  Flat Open -> Best: {gap_rules.get('flat_open_best_action', 'N/A')}")

    return gap_results, gap_rules


def analyze_orb(nifty_df):
    """
    Opening Range Breakout (ORB) analysis.
    - If the first-hour high is broken later, does BUY_CALL work?
    - If the first-hour low is broken later, does BUY_PUT work?
    """
    print("\n" + "=" * 70)
    print("OPENING RANGE BREAKOUT (ORB) ANALYSIS")
    print("=" * 70)

    orb_call_pnls = []
    orb_put_pnls = []
    orb_none = 0
    both_breakouts = 0

    for idx, row in nifty_df.iterrows():
        bars = simulate_intraday_bars(row)
        vix_val = row["vix"]
        spot_close = row["close"]

        # First hour = bars 0-2 (9:15-10:00)
        first_hour_high = np.max(bars[0:3])
        first_hour_low = np.min(bars[0:3])

        # Check rest of day for breakout
        rest_of_day = bars[3:]
        high_broken = np.any(rest_of_day > first_hour_high)
        low_broken = np.any(rest_of_day < first_hour_low)

        if high_broken and low_broken:
            both_breakouts += 1
            # Use the first breakout direction
            high_break_bar = None
            low_break_bar = None
            for i, b in enumerate(rest_of_day):
                if b > first_hour_high and high_break_bar is None:
                    high_break_bar = i
                if b < first_hour_low and low_break_bar is None:
                    low_break_bar = i

            if high_break_bar is not None and (low_break_bar is None or high_break_bar < low_break_bar):
                entry_spot = first_hour_high + 5  # enter slightly above
                pnl, _, _, _ = price_at_bar(entry_spot, spot_close, vix_val, "BUY_CALL")
                orb_call_pnls.append(pnl)
            else:
                entry_spot = first_hour_low - 5  # enter slightly below
                pnl, _, _, _ = price_at_bar(entry_spot, spot_close, vix_val, "BUY_PUT")
                orb_put_pnls.append(pnl)

        elif high_broken:
            entry_spot = first_hour_high + 5
            pnl, _, _, _ = price_at_bar(entry_spot, spot_close, vix_val, "BUY_CALL")
            orb_call_pnls.append(pnl)

        elif low_broken:
            entry_spot = first_hour_low - 5
            pnl, _, _, _ = price_at_bar(entry_spot, spot_close, vix_val, "BUY_PUT")
            orb_put_pnls.append(pnl)

        else:
            orb_none += 1

    total_days = len(nifty_df)
    print(f"\n  Total trading days: {total_days}")
    print(f"  Days with high breakout only: {len(orb_call_pnls) - (both_breakouts - len(orb_put_pnls) + orb_none - orb_none)}")
    print(f"  Days with ORB CALL signal: {len(orb_call_pnls)}")
    print(f"  Days with ORB PUT signal: {len(orb_put_pnls)}")
    print(f"  Days with both breakouts: {both_breakouts}")
    print(f"  Days with no breakout: {orb_none}")

    orb_results = {}

    for label, pnls in [("ORB_CALL", orb_call_pnls), ("ORB_PUT", orb_put_pnls)]:
        if not pnls:
            orb_results[label] = {"trades": 0, "win_rate": 0, "avg_pnl": 0, "total_pnl": 0}
            continue

        wins = sum(1 for p in pnls if p > 0)
        wr = wins / len(pnls) * 100
        avg = np.mean(pnls)
        total = np.sum(pnls)
        sharpe = np.mean(pnls) / np.std(pnls) * np.sqrt(252) if np.std(pnls) > 0 else 0

        orb_results[label] = {
            "trades": len(pnls),
            "win_rate": round(wr, 1),
            "avg_pnl": round(avg, 2),
            "total_pnl": round(total, 2),
            "sharpe": round(sharpe, 2),
            "max_win": round(max(pnls), 2),
            "max_loss": round(min(pnls), 2),
        }

    print(f"\n{'ORB Signal':<15} {'Trades':>6} {'Win%':>7} {'Avg PnL':>10} {'Total PnL':>12} {'Sharpe':>8}")
    print("-" * 65)
    for label, s in orb_results.items():
        if s["trades"] > 0:
            print(f"{label:<15} {s['trades']:>6} {s['win_rate']:>6.1f}% {s['avg_pnl']:>10.0f} {s['total_pnl']:>12.0f} {s.get('sharpe', 0):>8.2f}")

    # Derive ORB rules
    orb_rules = {
        "high_breakout_buy_call": {
            "enabled": orb_results["ORB_CALL"]["win_rate"] > 50 if orb_results["ORB_CALL"]["trades"] > 0 else False,
            "win_rate": orb_results["ORB_CALL"]["win_rate"],
            "avg_pnl": orb_results["ORB_CALL"]["avg_pnl"],
            "trades": orb_results["ORB_CALL"]["trades"],
        },
        "low_breakout_buy_put": {
            "enabled": orb_results["ORB_PUT"]["win_rate"] > 50 if orb_results["ORB_PUT"]["trades"] > 0 else False,
            "win_rate": orb_results["ORB_PUT"]["win_rate"],
            "avg_pnl": orb_results["ORB_PUT"]["avg_pnl"],
            "trades": orb_results["ORB_PUT"]["trades"],
        },
        "no_breakout_days_pct": round(orb_none / total_days * 100, 1),
        "both_breakout_days_pct": round(both_breakouts / total_days * 100, 1),
    }

    print(f"\n  ORB CALL signal reliable: {'YES' if orb_rules['high_breakout_buy_call']['enabled'] else 'NO'} (Win Rate: {orb_rules['high_breakout_buy_call']['win_rate']:.1f}%)")
    print(f"  ORB PUT signal reliable: {'YES' if orb_rules['low_breakout_buy_put']['enabled'] else 'NO'} (Win Rate: {orb_rules['low_breakout_buy_put']['win_rate']:.1f}%)")

    return orb_results, orb_rules


def save_results(best_windows, gap_rules, orb_rules, window_summary, wait_results):
    """Save all timing rules to JSON."""
    data_dir = PROJECT_ROOT / "data"
    data_dir.mkdir(exist_ok=True)
    output_path = data_dir / "timing_rules.json"

    # Build timing stats
    timing_stats = {}
    for (label, action), stats in window_summary.items():
        key = f"{label}_{action}"
        timing_stats[key] = stats

    # Build best entry window
    best_entry = {}
    for action, info in best_windows.items():
        best_entry[action] = info["label"]

    output = {
        "best_entry_window": best_entry,
        "best_entry_details": {
            action: {
                "window_name": info["window"],
                "sharpe": info["sharpe"],
                "win_rate": info["win_rate"],
                "avg_pnl": info["avg_pnl"],
            }
            for action, info in best_windows.items()
        },
        "gap_rules": {
            "gap_up_best_action": gap_rules.get("gap_up_best_action", "N/A"),
            "gap_down_best_action": gap_rules.get("gap_down_best_action", "N/A"),
            "flat_open_best_action": gap_rules.get("flat_open_best_action", "N/A"),
            "gap_up_win_rate": gap_rules.get("gap_up_stats", {}).get("win_rate", 0),
            "gap_down_win_rate": gap_rules.get("gap_down_stats", {}).get("win_rate", 0),
            "flat_open_win_rate": gap_rules.get("flat_open_stats", {}).get("win_rate", 0),
        },
        "orb_rules": orb_rules,
        "wait_30min_analysis": wait_results,
        "timing_stats": timing_stats,
        "metadata": {
            "study_period": "2025-10-01 to 2026-04-05",
            "lot_size": LOT_SIZE,
            "capital": CAPITAL,
            "dte": DTE,
            "strike_interval": STRIKE_INTERVAL,
            "generated_at": datetime.now().isoformat(),
        },
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to: {output_path}")
    return output


def main():
    print("=" * 70)
    print("  NIFTY OPTIMAL INTRADAY ENTRY TIMING STUDY")
    print(f"  Period: Oct 2025 - Apr 2026 | Lot Size: {LOT_SIZE} | Capital: {CAPITAL:,}")
    print("=" * 70)

    # Step 1: Download data
    nifty_df = download_data()

    # Step 2: Entry window analysis
    window_results, window_summary, best_windows = analyze_entry_windows(nifty_df)

    # Step 3: First 30-min wait analysis
    wait_results = analyze_first_30min_wait(nifty_df)

    # Step 4: Gap analysis
    gap_full_results, gap_rules = analyze_gap_effect(nifty_df)

    # Step 5: Opening Range Breakout
    orb_results, orb_rules = analyze_orb(nifty_df)

    # Step 6: Save results
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)

    output = save_results(best_windows, gap_rules, orb_rules, window_summary, wait_results)

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY - OPTIMAL ENTRY TIMING RULES")
    print("=" * 70)

    print(f"\n  1. BEST ENTRY WINDOWS:")
    for action, info in best_windows.items():
        print(f"     {action}: {info['window']} (Sharpe={info['sharpe']:.2f}, WR={info['win_rate']:.1f}%)")

    print(f"\n  2. GAP RULES:")
    print(f"     After Gap-Up:   Best = {gap_rules.get('gap_up_best_action', 'N/A')}")
    print(f"     After Gap-Down: Best = {gap_rules.get('gap_down_best_action', 'N/A')}")
    print(f"     Flat Open:      Best = {gap_rules.get('flat_open_best_action', 'N/A')}")

    print(f"\n  3. ORB RULES:")
    print(f"     First-hour high breakout -> BUY_CALL: {'ENABLED' if orb_rules['high_breakout_buy_call']['enabled'] else 'DISABLED'} (WR={orb_rules['high_breakout_buy_call']['win_rate']:.1f}%)")
    print(f"     First-hour low breakout  -> BUY_PUT:  {'ENABLED' if orb_rules['low_breakout_buy_put']['enabled'] else 'DISABLED'} (WR={orb_rules['low_breakout_buy_put']['win_rate']:.1f}%)")

    print(f"\n  4. 30-MIN WAIT ANALYSIS:")
    for name, s in wait_results.items():
        print(f"     {name}: WR={s['win_rate']:.1f}%, AvgPnL={s['avg_pnl']:.0f}")

    print("\n" + "=" * 70)
    print("Study complete. Results saved to data/timing_rules.json")
    print("=" * 70)


if __name__ == "__main__":
    main()
