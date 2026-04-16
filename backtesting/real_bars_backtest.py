"""
REAL 15-MIN BAR BACKTESTER — No more fake synthetic paths!

Uses actual 15-minute candle data from Yahoo Finance / Kite Connect
instead of the fake generate_intraday_path() that creates smooth paths.

This gives us the TRUTH about how each version performs in real markets.
"""

import json
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
LOT_SIZE = 65
MAX_TRADES_PER_DAY = 5
MAX_CONCURRENT = 2
TOTAL_BARS = 25


def load_real_data():
    """Load real 15-min and daily data."""
    data_dir = project_root / "data" / "historical"

    # Load 15-min intraday bars
    intraday_path = data_dir / "nifty_15min_real_60d.csv"
    if not intraday_path.exists():
        print("ERROR: Real 15-min data not found! Run download_real_intraday.py first.")
        sys.exit(1)

    intraday = pd.read_csv(intraday_path, index_col=0, parse_dates=True)
    intraday.columns = [c.lower() if c[0].isupper() else c for c in intraday.columns]
    # Ensure columns are lowercase
    col_map = {}
    for c in intraday.columns:
        col_map[c] = c.lower()
    intraday = intraday.rename(columns=col_map)

    # Load VIX 15-min
    vix_path = data_dir / "vix_15min_real_60d.csv"
    vix_df = None
    if vix_path.exists():
        vix_df = pd.read_csv(vix_path, index_col=0, parse_dates=True)
        vix_df.columns = [c.lower() for c in vix_df.columns]

    # Load daily data (for SMA50, RSI etc.)
    daily_path = data_dir / "nifty_daily_with_warmup.csv"
    daily = pd.read_csv(daily_path, index_col=0, parse_dates=True)
    if isinstance(daily.columns, pd.MultiIndex):
        daily.columns = daily.columns.get_level_values(0)
    daily.columns = [c.lower() for c in daily.columns]

    # Enrich daily data with indicators
    daily["sma50"] = daily["close"].rolling(50).mean()
    daily["sma20"] = daily["close"].rolling(20).mean()
    daily["ema9"] = daily["close"].ewm(span=9).mean()
    daily["ema21"] = daily["close"].ewm(span=21).mean()
    daily["weekly_sma"] = daily["close"].rolling(5).mean().rolling(4).mean()

    delta = daily["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    daily["rsi"] = 100 - (100 / (1 + gain / loss.replace(0, 0.001)))

    daily["prev_high"] = daily["high"].shift(1)
    daily["prev_low"] = daily["low"].shift(1)
    daily["prev_close"] = daily["close"].shift(1)
    daily["change_pct"] = daily["close"].pct_change() * 100
    daily["prev_change"] = daily["change_pct"].shift(1)
    daily["above_sma50"] = daily["close"] > daily["sma50"]
    daily["above_sma20"] = daily["close"] > daily["sma20"]
    daily["dow"] = daily.index.day_name()
    daily["gap_pct"] = (daily["open"] - daily["prev_close"]) / daily["prev_close"] * 100
    daily["gap_pct"] = daily["gap_pct"].fillna(0)

    # DTE calculation (Tuesday expiry after Nov 2025)
    dte_values = []
    for idx in daily.index:
        target = 1  # Tuesday
        current_dow = idx.weekday()
        if current_dow <= target:
            dte = target - current_dow
        else:
            dte = 7 - current_dow + target
        dte_values.append(max(dte, 0.5))
    daily["dte"] = dte_values
    daily["is_expiry"] = daily.index.map(lambda d: d.strftime("%A") == "Tuesday")

    return intraday, vix_df, daily


def simulate_day_real(day_bars, daily_row, daily_idx, daily_df,
                       close_history, version="V7"):
    """Simulate one day using REAL 15-min bars."""
    cfg = VERSION_CONFIG[version]

    # Extract daily indicators
    vix_val = 15.0  # Default
    if "vix" in daily_row.index and pd.notna(daily_row.get("vix")):
        vix_val = float(daily_row["vix"])

    # Try to get VIX from daily data columns
    for vix_col in ["vix", "india vix", "indiavix"]:
        if vix_col in daily_row.index and pd.notna(daily_row.get(vix_col)):
            vix_val = float(daily_row[vix_col])
            break

    entry_spot = float(day_bars.iloc[0]["open"])
    day_close_price = float(day_bars.iloc[-1]["close"])
    dow = str(daily_row.get("dow", "Monday"))
    above_sma50 = bool(daily_row.get("above_sma50", True))
    above_sma20 = bool(daily_row.get("above_sma20", True))
    rsi = float(daily_row.get("rsi", 50)) if pd.notna(daily_row.get("rsi")) else 50
    prev_change = float(daily_row.get("prev_change", 0)) if pd.notna(daily_row.get("prev_change")) else 0
    sma20 = float(daily_row.get("sma20")) if pd.notna(daily_row.get("sma20")) else None
    sma50 = float(daily_row.get("sma50")) if pd.notna(daily_row.get("sma50")) else None
    prev_high = float(daily_row.get("prev_high")) if pd.notna(daily_row.get("prev_high")) else None
    prev_low = float(daily_row.get("prev_low")) if pd.notna(daily_row.get("prev_low")) else None
    is_expiry = bool(daily_row.get("is_expiry", False))
    dte_market = float(daily_row.get("dte", 2.0))
    ema9 = float(daily_row.get("ema9")) if pd.notna(daily_row.get("ema9")) else None
    ema21 = float(daily_row.get("ema21")) if pd.notna(daily_row.get("ema21")) else None
    weekly_sma = float(daily_row.get("weekly_sma")) if pd.notna(daily_row.get("weekly_sma")) else None
    gap_pct = float(daily_row.get("gap_pct", 0)) if pd.notna(daily_row.get("gap_pct")) else 0
    date_str = str(daily_row.name.date())

    vix_spike = False  # Simplification

    # VIX check
    if vix_val < cfg["vix_floor"] or vix_val > cfg["vix_ceil"]:
        return 0, [{"action": "SKIP", "reason": f"VIX {vix_val:.1f}", "date": date_str}]

    # S/R levels
    support, resistance = sr_multi_method(
        entry_spot, prev_high, prev_low, sma20, sma50,
        close_history=close_history, idx=daily_idx)

    # Composite scoring
    scores = compute_composite(
        version, vix_val, above_sma50, above_sma20, rsi, dow, prev_change,
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

    # Build REAL price path from actual 15-min close prices
    path = day_bars["close"].values.tolist()
    # Pad to 25 bars if needed
    while len(path) < TOTAL_BARS:
        path.append(path[-1])

    open_trades = []
    closed_trades = []
    total_day_trades = 0
    last_exit_bar = -99
    n_bars = min(len(path), TOTAL_BARS)

    for bar_idx in range(n_bars):
        bar_spot = path[bar_idx]
        bar_dte = max(0.05, dte_market - bar_idx * 15 / 1440)

        # 1. CHECK EXITS
        trades_to_close = []
        for ti, trade in enumerate(open_trades):
            exit_signal = check_exit(version, trade, bar_idx, bar_spot, bar_dte,
                                     vix_val, support, resistance, is_expiry, path)
            if exit_signal:
                exit_prem = bs_premium(bar_spot, trade["strike"], bar_dte,
                                       vix_val, trade["opt_type"])
                pnl = (exit_prem - trade["entry_prem"]) * trade["qty"] - 80
                trade["exit_bar"] = bar_idx
                trade["exit_spot"] = round(bar_spot, 0)
                trade["exit_prem"] = round(exit_prem, 2)
                trade["exit_reason"] = exit_signal
                trade["intraday_pnl"] = round(pnl, 0)
                trade["total_pnl"] = round(pnl, 0)
                trades_to_close.append(ti)
                last_exit_bar = bar_idx

        for ti in reversed(trades_to_close):
            closed_trades.append(open_trades.pop(ti))

        # 2. CHECK ENTRIES
        if (len(open_trades) < MAX_CONCURRENT
                and total_day_trades < MAX_TRADES_PER_DAY
                and bar_idx - last_exit_bar >= cfg["cooldown"]
                and bar_idx < cfg["max_entry_bar"]):

            entries = detect_entries(
                version, bar_idx, path, support, resistance, vix_val, gap_pct,
                best_composite, composite_conf, is_expiry,
                prev_high or path[0], prev_low or path[0],
                above_sma50, above_sma20, bias_val=bias_val)

            if entries:
                entries.sort(key=lambda x: x[2], reverse=True)
                action, entry_type, conf, is_zero_hero = entries[0]

                same_dir = [t for t in open_trades if t["action"] == action]
                if not same_dir:
                    strike, opt_type = get_strike_and_type(action, bar_spot, vix_val, is_zero_hero)
                    if version in ("V6", "V7"):
                        num_lots = get_lot_count_span(vix_val, is_zero_hero)
                    else:
                        num_lots = get_lot_count_legacy(vix_val, is_zero_hero)

                    qty = min(num_lots * LOT_SIZE, 1800)
                    entry_prem = bs_premium(bar_spot, strike, bar_dte, vix_val, opt_type)

                    trade = {
                        "day": daily_idx + 1, "date": date_str, "dow": dow,
                        "action": action, "entry_type": entry_type,
                        "is_zero_hero": is_zero_hero, "confidence": round(conf, 2),
                        "entry_bar": bar_idx, "entry_spot": round(bar_spot, 0),
                        "entry_prem": round(entry_prem, 2), "strike": int(strike),
                        "opt_type": opt_type, "lots": num_lots, "qty": qty,
                        "vix": round(vix_val, 1), "is_expiry": is_expiry,
                        "dte": round(bar_dte, 1), "support": support,
                        "resistance": resistance, "best_fav": bar_spot,
                        "sr_target_hit": False,
                        "exit_bar": -1, "exit_spot": 0, "exit_prem": 0,
                        "exit_reason": "", "intraday_pnl": 0,
                        "overnight_pnl": 0, "total_pnl": 0,
                    }
                    open_trades.append(trade)
                    total_day_trades += 1

        # 3. UPDATE tracking
        for trade in open_trades:
            if trade["action"] == "BUY_CALL" and bar_spot > trade["best_fav"]:
                trade["best_fav"] = bar_spot
            elif trade["action"] == "BUY_PUT" and bar_spot < trade["best_fav"]:
                trade["best_fav"] = bar_spot

    # 4. FORCE CLOSE remaining
    for trade in open_trades:
        exit_prem = bs_premium(day_close_price, trade["strike"],
                               max(0.05, dte_market - 24 * 15 / 1440),
                               vix_val, trade["opt_type"])
        pnl = (exit_prem - trade["entry_prem"]) * trade["qty"] - 80
        trade["exit_bar"] = n_bars - 1
        trade["exit_spot"] = round(day_close_price, 0)
        trade["exit_prem"] = round(exit_prem, 2)
        trade["exit_reason"] = "eod_close"
        trade["intraday_pnl"] = round(pnl, 0)
        trade["total_pnl"] = round(pnl, 0)
        closed_trades.append(trade)

    if not closed_trades:
        return 0, [{"action": "SKIP", "reason": "No signals", "date": date_str}]

    return sum(t["total_pnl"] for t in closed_trades), closed_trades


def run_version_real(intraday, daily, close_history, version):
    """Run a version on real intraday data."""
    equity = CAPITAL
    all_trades = []
    peak_equity = CAPITAL
    max_dd = 0
    daily_pnl_list = []
    exit_stats = defaultdict(lambda: {"count": 0, "pnl": 0, "wins": 0})
    entry_stats = defaultdict(lambda: {"count": 0, "pnl": 0, "wins": 0})

    # Get trading days from intraday data
    trading_days = sorted(set(intraday.index.date))

    for d_idx, trade_date in enumerate(trading_days):
        day_bars = intraday[intraday.index.date == trade_date].copy()
        if len(day_bars) < 5:
            continue

        # Find matching daily row
        daily_match = daily[daily.index.date == trade_date]
        if daily_match.empty:
            continue
        daily_row = daily_match.iloc[0]
        daily_idx = daily.index.get_loc(daily_match.index[0])

        day_pnl, day_trades = simulate_day_real(
            day_bars, daily_row, daily_idx, daily, close_history, version)

        if len(day_trades) == 1 and day_trades[0].get("action") == "SKIP":
            daily_pnl_list.append(0)
            continue

        equity += day_pnl
        daily_pnl_list.append(day_pnl)
        if equity > peak_equity: peak_equity = equity
        dd = (peak_equity - equity) / peak_equity * 100
        if dd > max_dd: max_dd = dd

        for t in day_trades:
            all_trades.append(t)
            et = t.get("entry_type", "?")
            entry_stats[et]["count"] += 1
            entry_stats[et]["pnl"] += t["total_pnl"]
            if t["total_pnl"] > 0: entry_stats[et]["wins"] += 1
            er = t.get("exit_reason", "?")
            exit_stats[er]["count"] += 1
            exit_stats[er]["pnl"] += t["total_pnl"]
            if t["total_pnl"] > 0: exit_stats[er]["wins"] += 1

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
        "exit_stats": dict(exit_stats),
        "entry_stats": dict(entry_stats),
    }


if __name__ == "__main__":
    print("=" * 100)
    print("  REAL 15-MIN BAR BACKTEST — NO FAKE PATHS")
    print("  Using actual Yahoo Finance 15-minute candle data")
    print("  Capital: Rs 200,000 per version")
    print("=" * 100)

    intraday, vix_df, daily = load_real_data()

    # Add VIX to daily data
    if vix_df is not None and not vix_df.empty:
        # Get daily VIX from 15-min VIX data (close of day)
        vix_daily = vix_df.groupby(vix_df.index.date)["close"].last()
        for d in daily.index:
            if d.date() in vix_daily.index:
                daily.loc[d, "vix"] = vix_daily[d.date()]
        daily["vix"] = daily.get("vix", pd.Series(dtype=float)).fillna(15.0)
    else:
        daily["vix"] = 15.0

    trading_days = sorted(set(intraday.index.date))
    close_history = daily["close"].values.tolist()

    print(f"\nReal 15-min data: {len(trading_days)} trading days")
    print(f"Date range: {trading_days[0]} to {trading_days[-1]}")
    print(f"NIFTY range: {intraday['close'].min():.0f} - {intraday['close'].max():.0f}")
    print(f"VIX range: {daily['vix'].min():.1f} - {daily['vix'].max():.1f}")

    # Run all versions
    results = {}
    for ver in ["V3", "V4", "V6", "V7"]:
        print(f"\n  Running {ver} on REAL bars...", end=" ")
        results[ver] = run_version_real(intraday, daily, close_history, ver)
        print(f"Rs {results[ver]['net_pnl']:>+,} | "
              f"{results[ver]['total_trades']}t | "
              f"WR {results[ver]['win_rate']:.1f}% | "
              f"Sharpe {results[ver]['sharpe']:.2f}")

    # Print comparison
    print("\n" + "=" * 100)
    print("  REAL 15-MIN BARS — V3 vs V4 vs V6 vs V7")
    print("  This is what ACTUALLY happens in live trading!")
    print("=" * 100)
    versions = ["V3", "V4", "V6", "V7"]
    header = f"{'Metric':<25}"
    for v in versions:
        header += f" {v:>18}"
    print(header)
    print("-" * 100)

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
    ]
    for name, fmt in metrics:
        row = f"  {name:<23}"
        for v in versions:
            row += f" {fmt(results[v]):>18}"
        print(row)

    print(f"\n  EXIT REASONS:")
    all_exits = set()
    for r in results.values():
        all_exits.update(r["exit_stats"].keys())
    for er in sorted(all_exits):
        row = f"    {er:<21}"
        for v in versions:
            s = results[v]["exit_stats"].get(er, {"count": 0, "pnl": 0, "wins": 0})
            wr = s["wins"] / s["count"] * 100 if s["count"] > 0 else 0
            row += f" {s['count']:>3}t Rs{s['pnl']:>+8,.0f} {wr:>3.0f}%"
        print(row)

    print("\n" + "=" * 100)
    best = max(results.values(), key=lambda r: r["net_pnl"])
    best_risk = max(results.values(), key=lambda r: r["sharpe"])
    print(f"  WINNER (Absolute P&L):     {best['version']} with Rs {best['net_pnl']:>+,}")
    print(f"  WINNER (Risk-Adjusted):    {best_risk['version']} with Sharpe {best_risk['sharpe']:.2f}")
    print("=" * 100)

    # Save
    save_data = {v: {k: val for k, val in r.items()
                      if k not in ("exit_stats", "entry_stats")}
                 for v, r in results.items()}
    out_path = project_root / "data" / "real_bars_comparison.json"
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")
