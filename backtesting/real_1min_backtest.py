"""
REAL 1-MINUTE BAR BACKTESTER — Matches Live Engine Exactly.

This is the DEFINITIVE backtest. It uses actual 1-minute candle data from
Kite Connect and evaluates entries/exits on EVERY 1-min bar, exactly how
the live V3MultiTradeLiveAgent operates.

Key differences from previous backtests:
  - Previous: 25 bars/day (15-min synthetic), exits checked 25 times
  - This:    375 bars/day (1-min real), exits checked 375 times
  - This means 15x more chances for trail stops to trigger
  - This IS what happens in live trading

Data source: data/historical/nifty_min_2025-10-01_2026-04-06.csv
  - 38,685 real 1-minute bars from Kite Connect
  - 104 trading days (Oct 2025 - Apr 2026)

Usage:
  python backtesting/real_1min_backtest.py
"""

import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backtesting.paper_trading_real_data import (
    sr_multi_method,
    bs_premium,
    get_strike_and_type,
    LOT_SIZE,
    BROKERAGE,
    STRIKE_INTERVAL,
    MAX_TRADES_PER_DAY,
    MAX_CONCURRENT,
)

from backtesting.v7_hybrid_comparison import (
    VERSION_CONFIG,
    compute_composite,
    detect_entries,
    check_exit,
    get_lot_count_legacy,
    get_lot_count_span,
)

CAPITAL = 200_000
BARS_PER_DAY = 375  # 9:15 to 15:30 = 375 minutes


# ===========================================================================
# LOAD REAL DATA
# ===========================================================================

def load_1min_data():
    """Load real 1-minute NIFTY candle data from Kite Connect CSV."""
    data_dir = project_root / "data" / "historical"

    # 1-min NIFTY bars
    nifty_path = data_dir / "nifty_min_2025-10-01_2026-04-06.csv"
    if not nifty_path.exists():
        print(f"ERROR: {nifty_path} not found!")
        print("Run: python backtesting/download_real_intraday.py --source kite --interval minute --start 2025-10-01 --end 2026-04-06")
        sys.exit(1)

    nifty_1min = pd.read_csv(nifty_path, parse_dates=["timestamp"], index_col="timestamp")
    print(f"Loaded {len(nifty_1min)} 1-min bars from {nifty_1min.index[0].date()} to {nifty_1min.index[-1].date()}")

    # VIX (daily)
    vix_path = data_dir / "vix_min_2025-10-01_2026-04-06.csv"
    vix_daily = None
    if vix_path.exists():
        vix_daily = pd.read_csv(vix_path, parse_dates=["timestamp"], index_col="timestamp")
        print(f"Loaded {len(vix_daily)} VIX daily bars")

    return nifty_1min, vix_daily


def build_daily_data(nifty_1min, vix_daily):
    """Build daily OHLCV + indicators from 1-min bars.

    This creates the same DataFrame structure that download_real_data() returns,
    but computed from real 1-min data instead of Yahoo daily data.
    """
    # Aggregate 1-min bars to daily OHLCV
    daily = nifty_1min.resample("D").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna()

    daily.columns = ["Open", "High", "Low", "Close", "Volume"]

    # Add VIX
    if vix_daily is not None and not vix_daily.empty:
        # Match VIX to trading days
        vix_close = vix_daily["close"].reindex(daily.index, method="ffill")
        daily["VIX"] = vix_close.fillna(14.0)
    else:
        daily["VIX"] = 14.0
    daily["VIX"] = daily["VIX"].ffill().bfill().fillna(14.0)

    # Indicators
    daily["PrevVIX"] = daily["VIX"].shift(1).fillna(daily["VIX"].iloc[0])
    daily["Change%"] = daily["Close"].pct_change() * 100
    daily["PrevChange%"] = daily["Change%"].shift(1).fillna(0)
    daily["DOW"] = daily.index.day_name()
    daily["SMA50"] = daily["Close"].rolling(50, min_periods=1).mean()
    daily["SMA20"] = daily["Close"].rolling(20, min_periods=1).mean()
    daily["AboveSMA50"] = daily["Close"] > daily["SMA50"]
    daily["AboveSMA20"] = daily["Close"] > daily["SMA20"]
    daily["EMA9"] = daily["Close"].ewm(span=9).mean()
    daily["EMA21"] = daily["Close"].ewm(span=21).mean()
    daily["WeeklySMA"] = daily["Close"].rolling(5).mean().rolling(4, min_periods=1).mean()

    delta = daily["Close"].diff()
    gain = delta.clip(lower=0).rolling(14, min_periods=1).mean()
    loss = (-delta.clip(upper=0)).rolling(14, min_periods=1).mean()
    daily["RSI"] = 100 - (100 / (1 + gain / loss.replace(0, 0.001)))

    daily["PrevHigh"] = daily["High"].shift(1)
    daily["PrevLow"] = daily["Low"].shift(1)
    daily["VIXSpike"] = daily["VIX"] > daily["PrevVIX"] * 1.15

    # Expiry day: Tuesday after Nov 2025 (SEBI rule)
    daily["IsExpiry"] = daily.index.map(
        lambda d: d.strftime("%A") == ("Tuesday" if d >= pd.Timestamp("2025-11-01") else "Thursday")
    )

    # DTE
    dte_values = []
    for idx in daily.index:
        current_dow = idx.weekday()
        target = 1 if idx >= pd.Timestamp("2025-11-01") else 3
        if current_dow <= target:
            dte = target - current_dow
        else:
            dte = 7 - current_dow + target
        dte_values.append(max(dte, 0.5))
    daily["DTE"] = dte_values

    # Gap %
    daily["PrevClose"] = daily["Close"].shift(1)
    daily["GapPct"] = (daily["Open"] - daily["PrevClose"]) / daily["PrevClose"] * 100
    daily["GapPct"] = daily["GapPct"].fillna(0)

    return daily


# ===========================================================================
# 1-MIN BAR SIMULATION (matches live engine)
# ===========================================================================

def simulate_day_1min(daily_row, row_idx, daily_df, close_prices,
                      day_1min_bars, version="V7"):
    """Simulate a single trading day using REAL 1-minute bars.

    Key difference from simulate_day():
      - Instead of 25 synthetic bars, we use 375 real 1-min bars
      - Entries happen at equivalent 15-min bar index (bar_idx = minute // 15)
      - Exits are checked on EVERY 1-min bar (375 checks vs 25)
      - This matches exactly how the live engine operates
    """
    cfg = VERSION_CONFIG[version]

    entry_spot = float(daily_row["Open"])
    vix = float(daily_row["VIX"]) if pd.notna(daily_row["VIX"]) else 14.0
    dow = str(daily_row["DOW"])
    above_sma50 = bool(daily_row["AboveSMA50"]) if pd.notna(daily_row.get("AboveSMA50")) else True
    above_sma20 = bool(daily_row["AboveSMA20"]) if pd.notna(daily_row.get("AboveSMA20")) else True
    rsi = float(daily_row["RSI"]) if pd.notna(daily_row.get("RSI")) else 50
    prev_change = float(daily_row["PrevChange%"]) if pd.notna(daily_row.get("PrevChange%")) else 0
    vix_spike = bool(daily_row["VIXSpike"]) if pd.notna(daily_row.get("VIXSpike")) else False
    sma20 = float(daily_row["SMA20"]) if pd.notna(daily_row.get("SMA20")) else None
    sma50 = float(daily_row["SMA50"]) if pd.notna(daily_row.get("SMA50")) else None
    prev_high = float(daily_row["PrevHigh"]) if pd.notna(daily_row.get("PrevHigh")) else entry_spot * 1.01
    prev_low = float(daily_row["PrevLow"]) if pd.notna(daily_row.get("PrevLow")) else entry_spot * 0.99
    is_expiry = bool(daily_row.get("IsExpiry", False))
    dte_market = float(daily_row.get("DTE", 2.0))
    ema9 = float(daily_row["EMA9"]) if pd.notna(daily_row.get("EMA9")) else None
    ema21 = float(daily_row["EMA21"]) if pd.notna(daily_row.get("EMA21")) else None
    weekly_sma = float(daily_row["WeeklySMA"]) if pd.notna(daily_row.get("WeeklySMA")) else None
    gap_pct = float(daily_row["GapPct"]) if pd.notna(daily_row.get("GapPct")) else 0
    date_str = str(daily_df.index[row_idx].date())
    day_close = float(daily_row["Close"])

    # VIX skip
    if vix < cfg["vix_floor"] or vix > cfg["vix_ceil"]:
        return 0, [{"action": "SKIP", "reason": f"VIX {vix:.1f} out of range",
                     "date": date_str}]

    # S/R levels
    support, resistance = sr_multi_method(
        entry_spot, prev_high, prev_low, sma20, sma50,
        close_history=close_prices, idx=row_idx)

    # Composite scoring
    scores = compute_composite(
        version, vix, above_sma50, above_sma20, rsi, dow, prev_change,
        vix_spike, entry_spot, support, resistance,
        ema9=ema9, ema21=ema21, weekly_sma=weekly_sma)
    best_composite = max(scores, key=scores.get)
    total_score = sum(scores.values())
    composite_conf = scores[best_composite] / total_score if total_score > 0 else 0

    # Bias
    bias_val = "neutral"
    if above_sma50 and above_sma20:
        bias_val = "strong_bullish" if ema9 and ema21 and ema9 > ema21 else "bullish"
    elif not above_sma50 and not above_sma20:
        bias_val = "strong_bearish" if ema9 and ema21 and ema9 < ema21 else "bearish"

    # Build the 1-min close price array for the day
    # day_1min_bars is a DataFrame with 'close' column
    minute_closes = day_1min_bars["close"].values

    # Also build a 15-min equivalent path for entry detection
    # (entries are checked at 15-min intervals like in the live engine)
    n_15min_bars = min(25, len(minute_closes) // 15 + 1)
    path_15min = []
    for i in range(n_15min_bars):
        idx = min(i * 15, len(minute_closes) - 1)
        path_15min.append(minute_closes[idx])

    open_trades = []
    closed_trades = []
    total_day_trades = 0
    last_exit_bar = -99  # In 15-min bar units for cooldown
    last_exit_minute = -99  # In minute units

    n_minutes = len(minute_closes)

    for minute_idx in range(n_minutes):
        bar_spot = minute_closes[minute_idx]
        # Which 15-min bar does this minute fall in?
        bar_15min = minute_idx // 15
        # DTE decreases as the day progresses
        bar_dte = max(0.05, dte_market - minute_idx / 1440)

        # ====== 1. CHECK EXITS ON EVERY 1-MIN BAR ======
        # This is the critical difference — live engine checks exits every minute
        trades_to_close = []
        for ti, trade in enumerate(open_trades):
            # Convert bars_held to 15-min equivalent for the exit logic
            entry_minute = trade["entry_minute"]
            minutes_held = minute_idx - entry_minute
            bars_held_15min = minutes_held // 15

            # Create a modified trade dict for exit check
            # The check_exit function uses bar_idx and entry_bar to compute bars_held
            # We pass virtual bar indices that represent 15-min equivalent
            virtual_bar_idx = trade["entry_bar_15min"] + bars_held_15min

            exit_signal = check_exit(
                version, trade, virtual_bar_idx, bar_spot, bar_dte,
                vix, support, resistance, is_expiry, path_15min)

            if exit_signal:
                exit_prem = bs_premium(bar_spot, trade["strike"], bar_dte,
                                       vix, trade["opt_type"])
                pnl = (exit_prem - trade["entry_prem"]) * trade["qty"] - 80
                trade["exit_bar"] = bar_15min
                trade["exit_minute"] = minute_idx
                trade["exit_spot"] = round(bar_spot, 0)
                trade["exit_prem"] = round(exit_prem, 2)
                trade["exit_reason"] = exit_signal
                trade["intraday_pnl"] = round(pnl, 0)
                trade["total_pnl"] = round(pnl, 0)
                trade["minutes_held"] = minutes_held
                trades_to_close.append(ti)
                last_exit_bar = bar_15min
                last_exit_minute = minute_idx

        for ti in reversed(trades_to_close):
            closed_trades.append(open_trades.pop(ti))

        # ====== 2. CHECK ENTRIES AT 15-MIN BOUNDARIES ONLY ======
        # Entries are evaluated every 15 minutes (like composite scoring interval)
        is_15min_boundary = (minute_idx % 15 == 0)

        if (is_15min_boundary
                and len(open_trades) < MAX_CONCURRENT
                and total_day_trades < MAX_TRADES_PER_DAY
                and bar_15min - last_exit_bar >= cfg["cooldown"]
                and bar_15min < cfg["max_entry_bar"]):

            entries = detect_entries(
                version, bar_15min, path_15min, support, resistance, vix, gap_pct,
                best_composite, composite_conf, is_expiry,
                prev_high, prev_low, above_sma50, above_sma20,
                bias_val=bias_val)

            if entries:
                entries.sort(key=lambda x: x[2], reverse=True)
                action, entry_type, conf, is_zero_hero = entries[0]

                same_dir = [t for t in open_trades if t["action"] == action]
                if not same_dir:
                    strike, opt_type = get_strike_and_type(action, bar_spot, vix, is_zero_hero)

                    if version in ("V6", "V7"):
                        num_lots = get_lot_count_span(vix, is_zero_hero)
                    else:
                        num_lots = get_lot_count_legacy(vix, is_zero_hero)

                    qty = min(num_lots * LOT_SIZE, 1800)
                    entry_prem = bs_premium(bar_spot, strike, bar_dte, vix, opt_type)

                    trade = {
                        "day": row_idx + 1,
                        "date": date_str,
                        "dow": dow,
                        "action": action,
                        "entry_type": entry_type,
                        "is_zero_hero": is_zero_hero,
                        "confidence": round(conf, 2),
                        "entry_bar": bar_15min,        # 15-min bar index
                        "entry_bar_15min": bar_15min,  # Store for exit calc
                        "entry_minute": minute_idx,    # Actual minute
                        "entry_spot": round(bar_spot, 0),
                        "entry_prem": round(entry_prem, 2),
                        "strike": int(strike),
                        "opt_type": opt_type,
                        "lots": num_lots,
                        "qty": qty,
                        "vix": round(vix, 1),
                        "is_expiry": is_expiry,
                        "dte": round(bar_dte, 1),
                        "support": support,
                        "resistance": resistance,
                        "best_fav": bar_spot,
                        "sr_target_hit": False,
                        "exit_bar": -1, "exit_minute": -1,
                        "exit_spot": 0, "exit_prem": 0,
                        "exit_reason": "", "intraday_pnl": 0,
                        "overnight_pnl": 0, "total_pnl": 0,
                        "minutes_held": 0,
                    }
                    open_trades.append(trade)
                    total_day_trades += 1

        # ====== 3. UPDATE BEST FAVORABLE ON EVERY 1-MIN BAR ======
        for trade in open_trades:
            if trade["action"] == "BUY_CALL" and bar_spot > trade["best_fav"]:
                trade["best_fav"] = bar_spot
            elif trade["action"] == "BUY_PUT" and bar_spot < trade["best_fav"]:
                trade["best_fav"] = bar_spot

    # ====== 4. FORCE CLOSE OPEN TRADES ======
    for trade in open_trades:
        exit_prem = bs_premium(day_close, trade["strike"],
                               max(0.05, dte_market - n_minutes / 1440),
                               vix, trade["opt_type"])
        pnl = (exit_prem - trade["entry_prem"]) * trade["qty"] - 80
        trade["exit_bar"] = n_minutes // 15
        trade["exit_minute"] = n_minutes - 1
        trade["exit_spot"] = round(day_close, 0)
        trade["exit_prem"] = round(exit_prem, 2)
        trade["exit_reason"] = "eod_close"
        trade["intraday_pnl"] = round(pnl, 0)
        trade["total_pnl"] = round(pnl, 0)
        trade["minutes_held"] = n_minutes - 1 - trade["entry_minute"]
        closed_trades.append(trade)

    # ====== 5. BTST ======
    for trade in closed_trades:
        if version in ("V3",):
            btst_ok = (trade["action"] == "BUY_PUT" and trade["intraday_pnl"] > 0
                       and not is_expiry
                       and trade["exit_reason"] in ("eod_close", "trail_pct")
                       and row_idx + 1 < len(daily_df))
        elif version == "V4":
            btst_ok = (trade["action"] == "BUY_PUT" and trade["intraday_pnl"] >= 0
                       and not is_expiry
                       and trade["exit_reason"] in ("eod_close", "trail_pct", "time_exit")
                       and row_idx + 1 < len(daily_df))
        else:  # V6, V7
            btst_ok = (trade["action"] == "BUY_PUT" and trade["intraday_pnl"] > 0
                       and not is_expiry and vix < 20
                       and trade["exit_reason"] in ("eod_close", "time_exit")
                       and row_idx + 1 < len(daily_df))

        if btst_ok:
            next_row = daily_df.iloc[row_idx + 1]
            next_open = float(next_row["Open"])
            gap = (next_open - day_close) / day_close * 100
            if gap < 0:
                on_pnl = (day_close - next_open) * trade["qty"] * 0.5 - 50
                on_pnl = max(on_pnl, -trade["intraday_pnl"] * 0.5)
            else:
                on_pnl = -abs(gap) * trade["qty"] * 0.3
                on_pnl = max(on_pnl, -trade["intraday_pnl"] * 0.5)
            trade["overnight_pnl"] = round(on_pnl, 0)
            trade["total_pnl"] = round(trade["intraday_pnl"] + on_pnl, 0)

    if not closed_trades:
        return 0, [{"action": "SKIP", "reason": "No signals", "date": date_str}]

    return sum(t["total_pnl"] for t in closed_trades), closed_trades


# ===========================================================================
# RUN VERSION ON REAL 1-MIN DATA
# ===========================================================================

def run_version_1min(daily_df, close_prices, day_bars_dict, version="V7"):
    """Run a version backtest on real 1-minute bars."""
    equity = CAPITAL
    equity_curve = [CAPITAL]
    all_trades = []
    peak_equity = CAPITAL
    max_dd = 0
    daily_pnl_list = []
    entry_stats = defaultdict(lambda: {"count": 0, "pnl": 0, "wins": 0})
    exit_stats = defaultdict(lambda: {"count": 0, "pnl": 0, "wins": 0})

    trading_days = sorted(day_bars_dict.keys())

    for i in range(len(daily_df)):
        date = daily_df.index[i].date()

        if date not in day_bars_dict:
            equity_curve.append(equity)
            daily_pnl_list.append(0)
            continue

        day_1min = day_bars_dict[date]
        if len(day_1min) < 30:  # Skip partial days
            equity_curve.append(equity)
            daily_pnl_list.append(0)
            continue

        row = daily_df.iloc[i]
        day_pnl, day_trades = simulate_day_1min(
            row, i, daily_df, close_prices, day_1min, version)

        if len(day_trades) == 1 and day_trades[0].get("action") == "SKIP":
            equity_curve.append(equity)
            daily_pnl_list.append(0)
            continue

        equity += day_pnl
        daily_pnl_list.append(day_pnl)
        if equity > peak_equity:
            peak_equity = equity
        dd = (peak_equity - equity) / peak_equity * 100
        if dd > max_dd:
            max_dd = dd

        for t in day_trades:
            all_trades.append(t)
            et = t.get("entry_type", "?")
            entry_stats[et]["count"] += 1
            entry_stats[et]["pnl"] += t["total_pnl"]
            if t["total_pnl"] > 0:
                entry_stats[et]["wins"] += 1
            er = t.get("exit_reason", "?")
            exit_stats[er]["count"] += 1
            exit_stats[er]["pnl"] += t["total_pnl"]
            if t["total_pnl"] > 0:
                exit_stats[er]["wins"] += 1

        equity_curve.append(equity)

    net_pnl = equity - CAPITAL
    total = len(all_trades)
    wins = [t for t in all_trades if t["total_pnl"] > 0]
    losses = [t for t in all_trades if t["total_pnl"] <= 0]
    wr = len(wins) / total * 100 if total else 0

    daily_arr = np.array([d for d in daily_pnl_list if d != 0])
    sharpe = 0
    if len(daily_arr) > 1 and daily_arr.std() > 0:
        sharpe = (daily_arr.mean() / daily_arr.std()) * np.sqrt(252)

    gw = sum(t["total_pnl"] for t in wins)
    gl = abs(sum(t["total_pnl"] for t in losses))
    pf = gw / gl if gl > 0 else float("inf")

    # Trail stop analysis
    trail_killed = [t for t in all_trades if t.get("exit_reason") == "trail_pct"]
    time_exits = [t for t in all_trades if t.get("exit_reason") == "time_exit"]
    eod_exits = [t for t in all_trades if t.get("exit_reason") == "eod_close"]

    # Average minutes held for trail kills
    avg_trail_minutes = np.mean([t.get("minutes_held", 0) for t in trail_killed]) if trail_killed else 0

    return {
        "version": version,
        "net_pnl": round(net_pnl),
        "return_pct": round(net_pnl / CAPITAL * 100, 1),
        "total_trades": total,
        "winning_trades": len(wins),
        "losing_trades": len(losses),
        "win_rate": round(wr, 1),
        "sharpe": round(sharpe, 2),
        "profit_factor": round(pf, 2),
        "max_drawdown": round(max_dd, 2),
        "avg_win": round(np.mean([t["total_pnl"] for t in wins]) if wins else 0),
        "avg_loss": round(np.mean([t["total_pnl"] for t in losses]) if losses else 0),
        "final_equity": round(equity),
        "trail_kills": len(trail_killed),
        "trail_kill_avg_minutes": round(avg_trail_minutes),
        "time_exits": len(time_exits),
        "eod_exits": len(eod_exits),
        "entry_stats": dict(entry_stats),
        "exit_stats": dict(exit_stats),
        "equity_curve": equity_curve,
        "all_trades": all_trades,
    }


# ===========================================================================
# DISPLAY
# ===========================================================================

def print_table(results_dict, title):
    """Print comparison table."""
    versions = list(results_dict.keys())
    print(f"\n{'=' * 110}")
    print(f"  {title}")
    print(f"{'=' * 110}")

    header = f"{'Metric':<30}"
    for v in versions:
        header += f" {v:>18}"
    print(header)
    print("-" * 110)

    metrics = [
        ("Net P&L", lambda r: f"Rs {r['net_pnl']:>+,}"),
        ("Return %", lambda r: f"{r['return_pct']:>+.1f}%"),
        ("Total Trades", lambda r: f"{r['total_trades']}"),
        ("Win / Lose", lambda r: f"{r['winning_trades']}W / {r['losing_trades']}L"),
        ("Win Rate", lambda r: f"{r['win_rate']:.1f}%"),
        ("Sharpe Ratio", lambda r: f"{r['sharpe']:.2f}"),
        ("Profit Factor", lambda r: f"{r['profit_factor']:.2f}"),
        ("Max Drawdown", lambda r: f"{r['max_drawdown']:.2f}%"),
        ("Avg Win", lambda r: f"Rs {r['avg_win']:>+,}"),
        ("Avg Loss", lambda r: f"Rs {r['avg_loss']:>+,}"),
        ("", lambda r: ""),
        ("Trail Stop Kills", lambda r: f"{r['trail_kills']}"),
        ("Trail Kill Avg Hold", lambda r: f"{r['trail_kill_avg_minutes']} min"),
        ("Time Exits (winners)", lambda r: f"{r['time_exits']}"),
        ("EOD Closes", lambda r: f"{r['eod_exits']}"),
    ]

    for name, fmt in metrics:
        if name == "":
            print(f"  {'--- EXIT ANALYSIS ---':<28}")
            continue
        row = f"  {name:<28}"
        for v in versions:
            row += f" {fmt(results_dict[v]):>18}"
        print(row)

    # Exit reason breakdown
    print(f"\n  EXIT REASONS (detail):")
    all_exits = set()
    for r in results_dict.values():
        all_exits.update(r["exit_stats"].keys())
    for er in sorted(all_exits):
        row = f"    {er:<25}"
        for v in versions:
            s = results_dict[v]["exit_stats"].get(er, {"count": 0, "pnl": 0, "wins": 0})
            wr = s["wins"] / s["count"] * 100 if s["count"] > 0 else 0
            row += f" {s['count']:>3}t Rs{s['pnl']:>+9,.0f} {wr:>3.0f}%"
        print(row)

    # Entry type breakdown
    print(f"\n  ENTRY TYPES:")
    all_entries = set()
    for r in results_dict.values():
        all_entries.update(r["entry_stats"].keys())
    for et in sorted(all_entries):
        row = f"    {et:<25}"
        for v in versions:
            s = results_dict[v]["entry_stats"].get(et, {"count": 0, "pnl": 0, "wins": 0})
            wr = s["wins"] / s["count"] * 100 if s["count"] > 0 else 0
            row += f" {s['count']:>3}t Rs{s['pnl']:>+9,.0f} {wr:>3.0f}%"
        print(row)

    print("=" * 110)


# ===========================================================================
# MAIN
# ===========================================================================

if __name__ == "__main__":
    print("=" * 110)
    print("  REAL 1-MINUTE BAR BACKTEST — MATCHES LIVE ENGINE EXACTLY")
    print("  Data: 38,685 real 1-min bars from Kite Connect")
    print("  Period: Oct 2025 - Apr 2026 (~104 trading days)")
    print("  Capital: Rs 200,000")
    print("  Exit checks: EVERY 1 minute (375/day) — same as live engine")
    print("=" * 110)

    # Load data
    print("\n--- Loading data ---")
    nifty_1min, vix_daily = load_1min_data()

    # Group 1-min bars by date
    day_bars_dict = {}
    for date, group in nifty_1min.groupby(nifty_1min.index.date):
        day_bars_dict[date] = group

    print(f"Trading days with 1-min data: {len(day_bars_dict)}")
    sample_date = sorted(day_bars_dict.keys())[0]
    print(f"Sample day ({sample_date}): {len(day_bars_dict[sample_date])} bars")

    # Build daily data with indicators
    print("\n--- Building daily indicators ---")
    daily_df = build_daily_data(nifty_1min, vix_daily)
    close_prices = daily_df["Close"].values.tolist()
    print(f"Daily data: {len(daily_df)} days")
    print(f"NIFTY range: {daily_df['Close'].min():.0f} - {daily_df['Close'].max():.0f}")
    print(f"VIX range: {daily_df['VIX'].min():.1f} - {daily_df['VIX'].max():.1f}")

    # Run all 4 versions
    print("\n--- Running backtests ---")
    results = {}
    for ver in ["V3", "V4", "V6", "V7"]:
        print(f"\n  Running {ver}...", flush=True)
        results[ver] = run_version_1min(daily_df, close_prices, day_bars_dict, ver)
        r = results[ver]
        print(f"    {ver}: Rs {r['net_pnl']:>+,} | {r['total_trades']}t | "
              f"WR {r['win_rate']:.1f}% | Trail kills: {r['trail_kills']} | "
              f"Sharpe: {r['sharpe']:.2f}")

    # Print comparison
    print_table(results, "REAL 1-MIN BARS: V3 vs V4 vs V6 vs V7 (Oct 2025 - Apr 2026)")

    # ── DEGRADATION COMPARISON WITH PREVIOUS RESULTS ──
    print("\n\n" + "=" * 110)
    print("  DEGRADATION CHAIN: Smooth -> Real 15-min -> Real 1-min")
    print("  (Shows exactly why V3 fails in live trading)")
    print("=" * 110)
    print(f"\n  {'Version':<8} {'Smooth (fake)':>15} {'Real 15-min':>15} {'Real 1-min':>15} {'Live Estimate':>15}")
    print("-" * 75)

    # Load previous results if available
    prev_smooth = {"V3": 766265, "V4": 602781, "V6": 193587, "V7": None}
    prev_15min = {"V3": None, "V4": None, "V6": None, "V7": None}

    # Try loading real bars comparison
    real_bars_path = project_root / "data" / "real_bars_comparison.json"
    if real_bars_path.exists():
        with open(real_bars_path) as f:
            rb_data = json.load(f)
            for v in rb_data:
                prev_15min[v] = rb_data[v].get("net_pnl", None)

    for ver in ["V3", "V4", "V6", "V7"]:
        smooth = prev_smooth.get(ver)
        r15 = prev_15min.get(ver)
        r1 = results[ver]["net_pnl"]

        smooth_str = f"Rs {smooth:>+,}" if smooth is not None else "N/A"
        r15_str = f"Rs {r15:>+,}" if r15 is not None else "N/A"
        r1_str = f"Rs {r1:>+,}"

        print(f"  {ver:<8} {smooth_str:>15} {r15_str:>15} {r1_str:>15}")

    print("\n  KEY INSIGHT:")
    print(f"  V3 on smooth paths: Rs +766K (the backtest illusion)")
    print(f"  V3 on real 1-min:   Rs {results['V3']['net_pnl']:>+,} (the live reality)")
    v3_deg = (766265 - results['V3']['net_pnl']) / 766265 * 100 if results['V3']['net_pnl'] != 766265 else 0
    print(f"  V3 degradation:     {v3_deg:.1f}% of profits LOST due to real market noise")
    print()

    best = max(results.values(), key=lambda r: r["sharpe"])
    print(f"  BEST VERSION ON REAL 1-MIN DATA: {best['version']}")
    print(f"    P&L: Rs {best['net_pnl']:>+,} | Sharpe: {best['sharpe']:.2f} | "
          f"WR: {best['win_rate']:.1f}% | Max DD: {best['max_drawdown']:.1f}%")
    print("=" * 110)

    # Save results
    save_data = {}
    for v, r in results.items():
        save_data[v] = {k: val for k, val in r.items()
                       if k not in ("equity_curve", "all_trades", "entry_stats", "exit_stats")}
        save_data[v]["entry_type_stats"] = {
            k: {"count": s["count"], "pnl": float(s["pnl"]), "wins": s["wins"]}
            for k, s in r["entry_stats"].items()
        }
        save_data[v]["exit_reason_stats"] = {
            k: {"count": s["count"], "pnl": float(s["pnl"]), "wins": s["wins"]}
            for k, s in r["exit_stats"].items()
        }

    out_path = project_root / "data" / "real_1min_backtest_results.json"
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")
