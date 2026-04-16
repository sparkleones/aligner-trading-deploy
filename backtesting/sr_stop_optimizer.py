"""
SR Stop Optimizer — Test different sr_stop configurations on V8.

Problem: SR stop is 0% WR across ALL models, losing Rs -47K to -57K.
Current: sr_stop_buffer=0.4%, fires after 45min hold, CALL-only.

Tests:
  A) Remove SR stop entirely
  B) Widen buffer to 0.8%
  C) Widen buffer to 1.0% + min hold 90min
  D) VIX-dynamic buffer (0.6% low VIX, 1.2% high VIX) + min hold 90min
  E) Replace with ATR-based stop (2x ATR)
"""

import sys
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
    V8_CONFIG, CAPITAL, get_dynamic_lots,
    detect_entries_v8, run_model,
)

# =====================================================================
# MODIFIED simulate_day with configurable SR stop
# =====================================================================

def simulate_day_sr_test(cfg, day_bars_df, date, prev_day_ohlc, vix, daily_trend,
                         dte, is_expiry, daily_df, row_idx, close_prices,
                         above_sma50, above_sma20, rsi, prev_change, vix_spike,
                         sma20, sma50, ema9, ema21, weekly_sma, gap_pct,
                         equity=CAPITAL, recent_wr=0.5, recent_trades=0):
    """Same as simulate_day but with configurable SR stop modes."""
    n_bars = len(day_bars_df)
    if n_bars < 60:
        return []

    use_v8 = cfg.get("use_v8_scoring", False)
    sr_mode = cfg.get("sr_stop_mode", "original")  # NEW: sr stop mode

    # SR levels
    spot = day_bars_df["close"].iloc[0]
    if prev_day_ohlc:
        pivots = compute_pivot_points(prev_day_ohlc["high"], prev_day_ohlc["low"], prev_day_ohlc["close"])
    else:
        pivots = compute_pivot_points(spot * 1.005, spot * 0.995, spot)
    sr_levels = find_support_resistance(day_bars_df, prev_day_ohlc, pivots)

    # Daily trend info
    support = min([l["price"] for l in sr_levels if l["type"] == "support"], default=0)
    resistance = max([l["price"] for l in sr_levels if l["type"] == "resistance"], default=99999)

    # V8 only — no composite needed
    best_composite, composite_conf = None, 0

    prev_high = prev_day_ohlc["high"] if prev_day_ohlc else day_bars_df["high"].iloc[0]
    prev_low = prev_day_ohlc["low"] if prev_day_ohlc else day_bars_df["low"].iloc[0]

    bias_val = "neutral"
    if above_sma50 and above_sma20:
        bias_val = "strong_bullish" if ema9 and ema21 and ema9 > ema21 else "bullish"
    elif not above_sma50 and not above_sma20:
        bias_val = "strong_bearish" if ema9 and ema21 and ema9 < ema21 else "bearish"

    minute_closes = day_bars_df["close"].values
    n_15min = min(25, n_bars // 15 + 1)
    path_15min = [minute_closes[min(i * 15, n_bars - 1)] for i in range(n_15min)]

    # Compute ATR for ATR-based stop
    day_atr = None
    if sr_mode == "atr_based" and len(day_bars_df) > 14:
        highs = day_bars_df["high"].values[:min(60, n_bars)]
        lows = day_bars_df["low"].values[:min(60, n_bars)]
        closes = day_bars_df["close"].values[:min(60, n_bars)]
        tr_vals = []
        for j in range(1, len(highs)):
            tr = max(highs[j] - lows[j], abs(highs[j] - closes[j-1]), abs(lows[j] - closes[j-1]))
            tr_vals.append(tr)
        day_atr = np.mean(tr_vals[-14:]) if len(tr_vals) >= 14 else np.mean(tr_vals) if tr_vals else 100

    open_trades = []
    closed_trades = []
    total_day_trades = 0
    last_exit_minute = -cfg["cooldown_min"]
    day_close = day_bars_df["close"].iloc[-1]

    for minute_idx in range(n_bars):
        bar_spot = minute_closes[minute_idx]
        bar_15min = minute_idx // 15
        bar_dte = max(0.05, dte - minute_idx / 1440)

        # ====== EXITS ======
        trades_to_close = []
        for ti, trade in enumerate(open_trades):
            minutes_held = minute_idx - trade["entry_minute"]
            if minutes_held < 1:
                continue

            exit_reason = None
            action = trade["action"]

            if is_expiry and minute_idx >= cfg["expiry_close_min"]:
                exit_reason = "expiry_close"
            elif action == "BUY_PUT" and minutes_held >= cfg["max_hold_put"]:
                exit_reason = "time_exit"
            elif action == "BUY_CALL" and minutes_held >= cfg["max_hold_call"]:
                exit_reason = "time_exit"
            elif action == "BUY_PUT" and minutes_held >= cfg["min_hold_trail_put"]:
                trail_d = trade["entry_spot"] * cfg["trail_pct_put"]
                if bar_spot > trade["best_fav"] + trail_d:
                    exit_reason = "trail_stop"
            elif action == "BUY_CALL" and minutes_held >= cfg["min_hold_trail_call"]:
                trail_d = trade["entry_spot"] * cfg["trail_pct_call"]
                if bar_spot < trade["best_fav"] - trail_d:
                    exit_reason = "trail_stop"

            # ====== SR STOP (configurable) ======
            if not exit_reason and action == "BUY_CALL":
                if sr_mode == "disabled":
                    pass  # No SR stop

                elif sr_mode == "original":
                    # Original: 0.4% buffer, 45min hold
                    if cfg["sr_stop_buffer"] and minutes_held >= 45:
                        call_stop = trade["entry_spot"] * (1 - cfg["sr_stop_buffer"])
                        if bar_spot < call_stop:
                            exit_reason = "sr_stop"

                elif sr_mode == "wide_08":
                    # Wider: 0.8% buffer, 45min hold
                    if minutes_held >= 45:
                        call_stop = trade["entry_spot"] * (1 - 0.008)
                        if bar_spot < call_stop:
                            exit_reason = "sr_stop"

                elif sr_mode == "wide_10_90min":
                    # Very wide: 1.0% buffer, 90min min hold
                    if minutes_held >= 90:
                        call_stop = trade["entry_spot"] * (1 - 0.010)
                        if bar_spot < call_stop:
                            exit_reason = "sr_stop"

                elif sr_mode == "vix_dynamic":
                    # VIX-based: wider in high VIX, tighter in low VIX
                    if vix < 14:
                        buf = 0.006
                        min_hold = 75
                    elif vix < 20:
                        buf = 0.008
                        min_hold = 90
                    else:
                        buf = 0.012
                        min_hold = 60  # In high VIX, cut faster but with wider buffer
                    if minutes_held >= min_hold:
                        call_stop = trade["entry_spot"] * (1 - buf)
                        if bar_spot < call_stop:
                            exit_reason = "sr_stop"

                elif sr_mode == "atr_based":
                    # ATR-based: 2x ATR from entry
                    if day_atr and minutes_held >= 60:
                        call_stop = trade["entry_spot"] - 2.0 * day_atr
                        if bar_spot < call_stop:
                            exit_reason = "sr_stop"

            # Also add PUT sr_stop for extreme moves
            if not exit_reason and action == "BUY_PUT" and sr_mode == "vix_dynamic":
                if vix < 14:
                    buf = 0.006
                    min_hold = 75
                elif vix < 20:
                    buf = 0.008
                    min_hold = 90
                else:
                    buf = 0.012
                    min_hold = 60
                if minutes_held >= min_hold:
                    put_stop = trade["entry_spot"] * (1 + buf)
                    if bar_spot > put_stop:
                        exit_reason = "sr_stop"

            if exit_reason:
                exit_prem = bs_premium(bar_spot, trade["strike"], bar_dte, vix, trade["opt_type"])
                pnl = (exit_prem - trade["entry_prem"]) * trade["qty"] - 80
                trade.update({
                    "exit_minute": minute_idx, "exit_spot": round(bar_spot, 2),
                    "exit_prem": round(exit_prem, 2), "exit_reason": exit_reason,
                    "pnl": round(pnl, 0), "minutes_held": minutes_held,
                })
                trades_to_close.append(ti)
                last_exit_minute = minute_idx

        for ti in reversed(trades_to_close):
            closed_trades.append(open_trades.pop(ti))

        # Update tracking
        for trade in open_trades:
            if trade["action"] == "BUY_CALL" and bar_spot > trade["best_fav"]:
                trade["best_fav"] = bar_spot
            elif trade["action"] == "BUY_PUT" and bar_spot < trade["best_fav"]:
                trade["best_fav"] = bar_spot

        # ====== ENTRIES ======
        if minute_idx % cfg["entry_check_interval"] != 0:
            continue
        if minute_idx < 5 or minute_idx > cfg["max_entry_min"]:
            continue
        if len(open_trades) >= cfg["max_concurrent"] or total_day_trades >= cfg["max_trades"]:
            continue
        if minute_idx - last_exit_minute < cfg["cooldown_min"]:
            continue

        if vix < cfg["vix_floor"] or vix > cfg["vix_ceil"]:
            continue

        if any(s <= minute_idx < e for s, e in cfg.get("avoid_windows", [])):
            continue

        # V8 entry
        if use_v8:
            bar_data = day_bars_df.iloc[minute_idx]
            direction, conf, reasons = detect_entries_v8(
                bar_data, sr_levels, vix, daily_trend, minute_idx)
            entries = [(direction, "v8_indicator", conf, False)] if direction else []
        else:
            entries = []

        if not entries:
            continue

        entries.sort(key=lambda x: x[2], reverse=True)
        for action, entry_type, conf, is_zh in entries:
            if action is None:
                continue
            if len(open_trades) >= cfg["max_concurrent"]:
                break
            same_dir = [t for t in open_trades if t["action"] == action]
            if same_dir:
                continue

            strike, opt_type = get_strike_and_type(action, bar_spot, vix, is_zh)
            num_lots = get_dynamic_lots(vix, equity, confidence=conf,
                                        zero_hero=is_zh,
                                        recent_wr=recent_wr,
                                        recent_trades=recent_trades)
            qty = num_lots * LOT_SIZE
            entry_prem = bs_premium(bar_spot, strike, bar_dte, vix, opt_type)

            trade = {
                "date": str(date), "action": action, "entry_type": entry_type,
                "is_zero_hero": is_zh, "confidence": round(conf, 3),
                "entry_minute": minute_idx, "entry_spot": round(bar_spot, 2),
                "entry_prem": round(entry_prem, 2), "strike": int(strike),
                "opt_type": opt_type, "lots": num_lots, "qty": qty,
                "vix": round(vix, 1), "is_expiry": is_expiry,
                "dte": round(bar_dte, 2), "best_fav": bar_spot,
                "exit_minute": -1, "exit_spot": 0, "exit_prem": 0,
                "exit_reason": "", "pnl": 0, "minutes_held": 0,
            }
            open_trades.append(trade)
            total_day_trades += 1

    # Force close
    for trade in open_trades:
        exit_prem = bs_premium(day_close, trade["strike"],
                               max(0.05, dte - n_bars / 1440), vix, trade["opt_type"])
        pnl = (exit_prem - trade["entry_prem"]) * trade["qty"] - 80
        trade.update({
            "exit_minute": n_bars - 1, "exit_spot": round(day_close, 2),
            "exit_prem": round(exit_prem, 2), "exit_reason": "eod_close",
            "pnl": round(pnl, 0), "minutes_held": n_bars - 1 - trade["entry_minute"],
        })
        closed_trades.append(trade)

    return closed_trades


# =====================================================================
# MODIFIED run_model that uses our custom simulate_day
# =====================================================================

def run_model_sr(cfg, daily_df, close_prices, day_groups, trading_dates,
                 vix_lookup, daily_trend_df, test_start):
    """Run model with custom SR stop."""
    equity = CAPITAL
    peak = CAPITAL
    max_dd = 0
    all_trades = []
    daily_pnls = []
    entry_stats = defaultdict(lambda: {"count": 0, "pnl": 0, "wins": 0})
    exit_stats = defaultdict(lambda: {"count": 0, "pnl": 0, "wins": 0})
    recent_results = []

    for i in range(len(daily_df)):
        date = daily_df.index[i].date()
        if date < test_start:
            continue
        if date not in day_groups:
            continue
        day_bars = day_groups[date]
        if len(day_bars) < 60:
            continue

        prev_ohlc = None
        if i > 0:
            prev_idx = daily_df.index[i - 1].date()
            if prev_idx in day_groups:
                pd_bars = day_groups[prev_idx]
                prev_ohlc = {
                    "high": pd_bars["high"].max(), "low": pd_bars["low"].min(),
                    "close": pd_bars["close"].iloc[-1], "open": pd_bars["open"].iloc[0],
                }

        row = daily_df.iloc[i]
        vix = vix_lookup.get(date, 14.0)

        is_expiry = date.strftime("%A") == "Thursday"
        dow = date.weekday()
        target = 3
        if dow <= target:
            dte = max(target - dow, 0.5)
        else:
            dte = max(7 - dow + target, 0.5)

        if date in daily_trend_df.index:
            d = daily_trend_df.loc[date]
            if d["close"] > d["sma20"] and d["ema9"] > d["ema21"]:
                daily_trend = "bullish"
            elif d["close"] < d["sma20"] and d["ema9"] < d["ema21"]:
                daily_trend = "bearish"
            else:
                daily_trend = "neutral"
        else:
            daily_trend = "neutral"

        recent_wr = sum(recent_results[-10:]) / len(recent_results[-10:]) if recent_results else 0.5
        recent_n = len(recent_results)

        day_trades = simulate_day_sr_test(
            cfg, day_bars, date, prev_ohlc, vix, daily_trend, dte, is_expiry,
            daily_df, i, close_prices,
            bool(row.get("AboveSMA50", True)), bool(row.get("AboveSMA20", True)),
            float(row.get("RSI", 50)), float(row.get("PrevChange%", 0)),
            bool(row.get("VIXSpike", False)),
            float(row.get("SMA20")) if pd.notna(row.get("SMA20")) else None,
            float(row.get("SMA50")) if pd.notna(row.get("SMA50")) else None,
            float(row.get("EMA9")) if pd.notna(row.get("EMA9")) else None,
            float(row.get("EMA21")) if pd.notna(row.get("EMA21")) else None,
            float(row.get("WeeklySMA")) if pd.notna(row.get("WeeklySMA")) else None,
            float(row.get("GapPct", 0)),
            equity=equity, recent_wr=recent_wr, recent_trades=recent_n,
        )

        # BTST
        if cfg["btst_enabled"] and i + 1 < len(daily_df):
            next_row = daily_df.iloc[i + 1]
            next_open = float(next_row["Open"])
            day_close = float(row["Close"])
            for t in day_trades:
                if (t["action"] == "BUY_PUT" and t["pnl"] > 0 and not is_expiry
                        and vix < cfg["btst_vix_cap"]
                        and t["exit_reason"] in ("eod_close", "time_exit")):
                    gap = (next_open - day_close) / day_close * 100
                    if gap < 0:
                        on_pnl = (day_close - next_open) * t["qty"] * 0.5 - 50
                        on_pnl = max(on_pnl, -t["pnl"] * 0.5)
                    else:
                        on_pnl = -abs(gap) * t["qty"] * 0.3
                        on_pnl = max(on_pnl, -t["pnl"] * 0.5)
                    t["btst_pnl"] = round(on_pnl, 0)
                    t["pnl"] = round(t["pnl"] + on_pnl, 0)

        day_pnl = sum(t["pnl"] for t in day_trades)
        equity += day_pnl
        if equity > peak: peak = equity
        dd = (peak - equity) / peak * 100
        if dd > max_dd: max_dd = dd
        daily_pnls.append(day_pnl)

        for t in day_trades:
            all_trades.append(t)
            recent_results.append(1 if t["pnl"] > 0 else 0)
            et = t.get("entry_type", "?")
            entry_stats[et]["count"] += 1
            entry_stats[et]["pnl"] += t["pnl"]
            if t["pnl"] > 0: entry_stats[et]["wins"] += 1
            er = t.get("exit_reason", "?")
            exit_stats[er]["count"] += 1
            exit_stats[er]["pnl"] += t["pnl"]
            if t["pnl"] > 0: exit_stats[er]["wins"] += 1

    net = equity - CAPITAL
    total = len(all_trades)
    wins = [t for t in all_trades if t["pnl"] > 0]
    losses = [t for t in all_trades if t["pnl"] <= 0]
    wr = len(wins) / total * 100 if total else 0
    gw = sum(t["pnl"] for t in wins)
    gl = abs(sum(t["pnl"] for t in losses))
    pf = gw / gl if gl > 0 else float("inf")
    arr = np.array([d for d in daily_pnls if d != 0])
    sharpe = (arr.mean() / arr.std()) * np.sqrt(252) if len(arr) > 1 and arr.std() > 0 else 0

    lots_list = [t["lots"] for t in all_trades]

    return {
        "name": cfg["name"], "net_pnl": round(net), "return_pct": round(net / CAPITAL * 100, 1),
        "total_trades": total, "trades_per_day": round(total / max(len(daily_pnls), 1), 1),
        "win_rate": round(wr, 1), "wins": len(wins), "losses": len(losses),
        "sharpe": round(sharpe, 2), "profit_factor": round(pf, 2),
        "max_drawdown": round(max_dd, 1),
        "avg_win": round(np.mean([t["pnl"] for t in wins])) if wins else 0,
        "avg_loss": round(np.mean([t["pnl"] for t in losses])) if losses else 0,
        "pnl_per_trade": round(net / max(total, 1)),
        "btst_count": len([t for t in all_trades if t.get("btst_pnl", 0) != 0]),
        "btst_pnl": round(sum(t.get("btst_pnl", 0) for t in all_trades)),
        "avg_lots": round(np.mean(lots_list), 1) if lots_list else 0,
        "max_lots": max(lots_list) if lots_list else 0,
        "final_equity": round(equity),
        "entry_stats": dict(entry_stats), "exit_stats": dict(exit_stats),
        "all_trades": all_trades,
    }


# =====================================================================
# MAIN
# =====================================================================

if __name__ == "__main__":
    print("=" * 120)
    print("  SR STOP OPTIMIZER — Testing 6 configurations on V8 with dynamic lots")
    print("  Data: June 2024 (unseen) | Capital: Rs 2,00,000 | Dynamic lot sizing (no caps)")
    print("=" * 120)

    # Load data
    data_dir = project_root / "data" / "historical"
    nifty = pd.read_csv(
        data_dir / "nifty_min_2024-05-01_2024-06-30.csv",
        parse_dates=["timestamp"], index_col="timestamp")
    vix_df = pd.read_csv(
        data_dir / "vix_min_2024-05-01_2024-06-30.csv",
        parse_dates=["timestamp"], index_col="timestamp")
    vix_lookup = {idx.date(): row["close"] for idx, row in vix_df.iterrows()}

    print("Computing indicators...", flush=True)
    nifty = add_all_indicators(nifty)

    day_groups = {date: group for date, group in nifty.groupby(nifty.index.date)}
    all_dates = sorted(day_groups.keys())

    # Build daily
    daily_rows = []
    for d in all_dates:
        bars = day_groups[d]
        daily_rows.append({"Date": d, "Open": bars["open"].iloc[0], "High": bars["high"].max(),
                           "Low": bars["low"].min(), "Close": bars["close"].iloc[-1],
                           "VIX": vix_lookup.get(d, 14.0)})
    daily = pd.DataFrame(daily_rows).set_index("Date")
    daily.index = pd.to_datetime(daily.index)
    daily["SMA20"] = daily["Close"].rolling(20).mean()
    daily["SMA50"] = daily["Close"].rolling(50).mean()
    daily["EMA9"] = compute_ema(daily["Close"], 9)
    daily["EMA21"] = compute_ema(daily["Close"], 21)
    daily["AboveSMA50"] = daily["Close"] > daily["SMA50"]
    daily["AboveSMA20"] = daily["Close"] > daily["SMA20"]
    daily["RSI"] = 50
    daily["PrevChange%"] = daily["Close"].pct_change() * 100
    daily["VIXSpike"] = daily["VIX"] > daily["VIX"].rolling(5).mean() * 1.2
    daily["WeeklySMA"] = daily["Close"].rolling(5).mean()
    daily["GapPct"] = (daily["Open"] - daily["Close"].shift(1)) / daily["Close"].shift(1) * 100

    close_prices = daily["Close"].values.tolist()
    daily_trend_df = daily[["Close", "SMA20", "EMA9", "EMA21"]].rename(
        columns={"Close": "close", "SMA20": "sma20", "EMA9": "ema9", "EMA21": "ema21"})
    daily_trend_df.index = daily_trend_df.index.date

    test_start = dt.date(2024, 6, 1)

    # ====== Test Configurations ======
    configs = [
        ("V8-Original (0.4%/45m)", "original"),
        ("V8-NoSRstop", "disabled"),
        ("V8-Wide0.8%", "wide_08"),
        ("V8-Wide1.0%/90m", "wide_10_90min"),
        ("V8-VIX-Dynamic", "vix_dynamic"),
        ("V8-ATR-Based", "atr_based"),
    ]

    results = []
    for name, mode in configs:
        cfg = copy.deepcopy(V8_CONFIG)
        cfg["name"] = name
        cfg["sr_stop_mode"] = mode
        print(f"\n  Running {name}...", flush=True)
        r = run_model_sr(cfg, daily, close_prices, day_groups, all_dates,
                         vix_lookup, daily_trend_df, test_start)
        results.append(r)
        sr_count = r["exit_stats"].get("sr_stop", {}).get("count", 0)
        sr_pnl = r["exit_stats"].get("sr_stop", {}).get("pnl", 0)
        sr_wins = r["exit_stats"].get("sr_stop", {}).get("wins", 0)
        print(f"    {name}: Rs {r['net_pnl']:>+,} | {r['total_trades']}t | "
              f"WR {r['win_rate']:.1f}% | Sharpe {r['sharpe']:.2f} | DD {r['max_drawdown']:.1f}% | "
              f"SR: {sr_count}t Rs{sr_pnl:+,.0f} ({sr_wins}W) | "
              f"Final Rs {r['final_equity']:>,}")

    # ====== Comparison Table ======
    print("\n" + "=" * 140)
    print("  SR STOP OPTIMIZATION RESULTS — V8 on June 2024 (dynamic lots, Rs 2L capital)")
    print("=" * 140)

    header = f"{'Config':<26}"
    header += f"{'Net P&L':>14} {'Return':>8} {'Trades':>7} {'WR':>6} {'Sharpe':>7} {'PF':>6} {'DD':>7} {'SR Trades':>10} {'SR P&L':>12} {'Equity':>12}"
    print(header)
    print("-" * 140)

    for r in results:
        sr = r["exit_stats"].get("sr_stop", {"count": 0, "pnl": 0, "wins": 0})
        sr_str = f"{sr['count']}t ({sr['wins']}W)"
        print(f"  {r['name']:<24} Rs{r['net_pnl']:>+9,} {r['return_pct']:>+7.1f}% "
              f"{r['total_trades']:>5}t {r['win_rate']:>5.1f}% {r['sharpe']:>6.2f} "
              f"{r['profit_factor']:>5.2f} {r['max_drawdown']:>6.1f}% "
              f"{sr_str:>10} Rs{sr['pnl']:>+9,.0f} Rs{r['final_equity']:>+10,}")

    # Show exit breakdown for each
    print("\n" + "-" * 140)
    print("  EXIT REASON BREAKDOWN:")
    print("-" * 140)
    all_exits = set()
    for r in results:
        all_exits.update(r["exit_stats"].keys())
    for er in sorted(all_exits):
        row = f"  {er:<18}"
        for r in results:
            s = r["exit_stats"].get(er, {"count": 0, "pnl": 0, "wins": 0})
            wr_val = s["wins"] / s["count"] * 100 if s["count"] > 0 else 0
            row += f" | {s['count']:>2}t Rs{s['pnl']:>+8,.0f} {wr_val:>3.0f}%"
        print(row)

    # Best model
    print("\n" + "=" * 140)
    best_pnl = max(results, key=lambda r: r["net_pnl"])
    best_sharpe = max(results, key=lambda r: r["sharpe"])
    best_dd = min(results, key=lambda r: r["max_drawdown"])
    print(f"  BEST P&L:     {best_pnl['name']}  Rs {best_pnl['net_pnl']:>+,}")
    print(f"  BEST Sharpe:  {best_sharpe['name']}  {best_sharpe['sharpe']:.2f}")
    print(f"  BEST DD:      {best_dd['name']}  {best_dd['max_drawdown']:.1f}%")

    # Composite score
    print("\n  COMPOSITE RANKING (0.4*P&L_rank + 0.3*Sharpe_rank + 0.3*DD_rank):")
    n = len(results)
    for r in results:
        pnl_rank = sorted(results, key=lambda x: x["net_pnl"], reverse=True).index(r) + 1
        sharpe_rank = sorted(results, key=lambda x: x["sharpe"], reverse=True).index(r) + 1
        dd_rank = sorted(results, key=lambda x: x["max_drawdown"]).index(r) + 1
        score = 0.4 * (n + 1 - pnl_rank) + 0.3 * (n + 1 - sharpe_rank) + 0.3 * (n + 1 - dd_rank)
        r["composite_score"] = score

    for r in sorted(results, key=lambda x: x["composite_score"], reverse=True):
        print(f"    {r['composite_score']:.1f}  {r['name']:<26} Rs{r['net_pnl']:>+9,} Sharpe {r['sharpe']:.2f} DD {r['max_drawdown']:.1f}%")

    print("=" * 140)
