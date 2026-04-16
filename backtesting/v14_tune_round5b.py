"""
V14 Round 5B — Focused Combo Testing of R5 Winners
====================================================================
R5 identified these winning methodologies:
  #1  ATR-normalized sizing (11.24x, PF 2.29, 65% MaxDD)
  #2  PSAR confluence filter (10.33x, PF 2.46, 48% MaxDD) ← BEST risk-adjusted
  #3  Donchian Channel scoring (9.97x, PF 2.43)
  #4  KAMA confluence filter (9.90x, PF 2.42, 52% MaxDD)

This round tests:
  - PSAR filter + ATR sizing (combine #1 + #2)
  - PSAR filter + Donchian (combine #2 + #3)
  - All top 4 together
  - PSAR filter variants (different safety levels)
  - ATR sizing with tighter drawdown caps
"""

import sys
import copy
import datetime as dt
from pathlib import Path
from collections import defaultdict

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scoring.config import V14_CONFIG
from scoring.indicators import compute_indicators
from scoring.engine import (
    score_entry, passes_confluence, evaluate_exit,
    compute_lots, detect_composite_entries,
)
from backtesting.option_pricer import price_option

DATA_DIR = project_root / "data" / "historical"
CAPITAL = 200_000
LOT_SIZE = 75
STRIKE_INTERVAL = 50
SLIPPAGE_PCT = 0.005
SPREAD_RS = 2.0
BROKERAGE_RT = 80.0

import pandas as pd
import numpy as np


def load_data(start_date, end_date):
    warmup_start = start_date - dt.timedelta(days=10)
    all_dfs = []
    for f in sorted(DATA_DIR.glob("nifty_min_*.csv")):
        try:
            df = pd.read_csv(f, parse_dates=["timestamp"], index_col="timestamp")
            mask = (df.index.date >= warmup_start) & (df.index.date <= end_date)
            chunk = df[mask]
            if len(chunk) > 0:
                all_dfs.append(chunk)
        except Exception:
            continue
    if not all_dfs:
        raise FileNotFoundError("No data")
    nifty = pd.concat(all_dfs)
    nifty = nifty[~nifty.index.duplicated(keep="first")].sort_index()
    n5 = nifty.resample("5min").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum"
    }).dropna(subset=["open"])
    n5 = n5[(n5.index.time >= dt.time(9, 15)) & (n5.index.time <= dt.time(15, 30))]

    vix_lookup = {}
    for f in sorted(DATA_DIR.glob("vix_min_*.csv")):
        try:
            df = pd.read_csv(f, parse_dates=["timestamp"], index_col="timestamp")
            mask = (df.index.date >= warmup_start) & (df.index.date <= end_date)
            chunk = df[mask]
            for d, g in chunk.groupby(chunk.index.date):
                vix_lookup[d] = float(g["close"].iloc[-1])
        except Exception:
            continue

    day_groups = {}
    for ts, row in n5.iterrows():
        d = ts.date()
        if d not in day_groups:
            day_groups[d] = []
        day_groups[d].append({
            "open": float(row["open"]), "high": float(row["high"]),
            "low": float(row["low"]), "close": float(row["close"]),
            "volume": float(row.get("volume", 0)),
            "date": str(d), "time": str(ts),
        })

    all_dates = sorted(d for d in day_groups if d >= start_date)
    warmup_dates = sorted(d for d in day_groups if d < start_date)
    warmup_bars = []
    for d in warmup_dates[-3:]:
        warmup_bars.extend(day_groups[d])

    return day_groups, all_dates, warmup_bars, vix_lookup


def calc_premium(spot, strike, dte, vix, opt_type, slippage_sign=1):
    try:
        result = price_option(spot=spot, strike=strike, dte_days=max(0.1, dte),
                              vix=vix, option_type=opt_type)
        prem = result["premium"]
    except Exception:
        prem = 30.0
    prem = prem * (1 + slippage_sign * SLIPPAGE_PCT) + slippage_sign * SPREAD_RS
    return max(0.05, prem)


def simulate_day_r5b(bars, date, vix, cfg, prev_close, equity, warmup_bars,
                     is_expiry, consecutive_down_days):
    date_str = str(date)
    bar_history = list(warmup_bars or [])
    open_trades = []
    closed_trades = []
    trades_today = 0
    last_exit_bar = -10
    orb_high = orb_low = 0.0
    gap_detected = False
    day_open = bars[0]["close"] if bars else 0
    day_of_week = date.weekday()

    span_per_lot = 40000 if vix < 20 else (50000 if vix < 25 else 60000)
    base_lots = max(1, int(equity * 0.70 / span_per_lot))
    max_lots_cap = cfg.get("max_lots_cap", 999)

    avoid_days = cfg.get("avoid_days", [])
    if day_of_week in avoid_days:
        eod_spot = bars[-1]["close"] if bars else 0
        return [], 0.0, eod_spot

    days_to_expiry = (3 - day_of_week) % 7
    dte = max(0.2, days_to_expiry) if days_to_expiry > 0 else 0.2

    entry_windows = cfg.get("entry_windows_bars", None)
    day_pnl = 0.0

    vix_lot_mult = 1.0
    if cfg.get("vix_lot_scaling"):
        if vix < 13:
            vix_lot_mult = cfg.get("vix_below13_mult", 0.3)
        elif 14 <= vix < 15:
            vix_lot_mult = cfg.get("vix_14_15_mult", 0.5)
        elif 15 <= vix < 17:
            vix_lot_mult = cfg.get("vix_15_17_mult", 1.5)
        elif vix >= 17:
            vix_lot_mult = cfg.get("vix_17plus_mult", 2.0)

    orb_lot_mult = cfg.get("orb_lot_mult", 1.0)

    for bar_idx, bar in enumerate(bars):
        bar_history.append(bar)
        if len(bar_history) > 500:
            bar_history = bar_history[-500:]
        spot = bar["close"]

        if bar_idx == 0:
            orb_high = bar["high"]; orb_low = bar["low"]
        elif bar_idx == 1:
            orb_high = max(orb_high, bar["high"])
            orb_low = min(orb_low, bar["low"])

        for trade in list(open_trades):
            if trade["action"] == "BUY_CALL" and spot > trade["best_fav"]:
                trade["best_fav"] = spot
            elif trade["action"] == "BUY_PUT" and spot < trade["best_fav"]:
                trade["best_fav"] = spot

            indicators = compute_indicators(bar_history, date_str)
            exit_reason = evaluate_exit(trade, bar_idx, spot, indicators or {}, cfg,
                                         day_of_week=day_of_week)

            if exit_reason:
                exit_prem = calc_premium(spot, trade["strike"],
                                          max(0.05, dte - bar_idx * 5 / (6.25 * 60)),
                                          vix, trade["opt_type"], -1)
                pnl = (exit_prem - trade["entry_premium"]) * trade["qty"] - BROKERAGE_RT
                trade.update({"exit_bar": bar_idx, "exit_spot": spot, "exit_premium": exit_prem,
                               "exit_reason": exit_reason, "pnl": pnl,
                               "bars_held": bar_idx - trade["entry_bar"]})
                closed_trades.append(trade)
                open_trades.remove(trade)
                day_pnl += pnl
                last_exit_bar = bar_idx

        indicators = compute_indicators(bar_history, date_str)
        if indicators is None:
            continue
        if len(open_trades) >= cfg["max_concurrent"]:
            continue
        if trades_today >= cfg["max_trades_per_day"]:
            continue

        action = None; conf = 0; entry_type = "v8_indicator"; is_zero_hero = False

        if bar_idx < 3:
            prev_spot = bars[bar_idx - 1]["close"] if bar_idx > 0 else spot
            composites = detect_composite_entries(
                bar, bar_idx, spot, vix, cfg,
                prev_close=prev_close, gap_detected=gap_detected,
                orb_high=orb_high, orb_low=orb_low, prev_spot=prev_spot,
            )
            if bar_idx == 0:
                gap_detected = True
            if cfg.get("disable_zero_hero", False):
                composites = [c for c in composites if not c[3]]
            if composites:
                composites.sort(key=lambda x: x[2], reverse=True)
                action, entry_type, conf, is_zero_hero = composites[0]
        else:
            if entry_windows:
                in_window = any(s <= bar_idx <= e for s, e in entry_windows)
                if not in_window:
                    continue
            if bar_idx - last_exit_bar < cfg["cooldown_bars"]:
                continue

            action, conf = score_entry(indicators, vix, cfg, bar_idx=bar_idx,
                                        consecutive_down_days=consecutive_down_days)
            if action is None:
                prev_spot = bars[bar_idx - 1]["close"] if bar_idx > 0 else spot
                composites = detect_composite_entries(
                    bar, bar_idx, spot, vix, cfg,
                    prev_close=prev_close, gap_detected=gap_detected,
                    orb_high=orb_high, orb_low=orb_low, prev_spot=prev_spot,
                )
                if cfg.get("disable_zero_hero", False):
                    composites = [c for c in composites if not c[3]]
                if composites:
                    composites.sort(key=lambda x: x[2], reverse=True)
                    action, entry_type, conf, is_zero_hero = composites[0]

        if action is None:
            continue

        min_conf = cfg.get("min_confidence", 0.35)
        if conf < min_conf:
            continue

        if vix < cfg["vix_floor"] or vix > cfg["vix_ceil"]:
            continue
        if not passes_confluence(action, conf, indicators, bar_idx, cfg,
                                  current_spot=spot, prev_close=prev_close, day_open=day_open):
            continue
        if any(t["action"] == action for t in open_trades):
            continue

        cfg_with_atr = cfg.copy()
        if cfg.get("use_atr_sizing"):
            cfg_with_atr["_current_atr"] = indicators.get("atr", 0)

        lots = compute_lots(action, conf, vix, indicators.get("rsi", 50),
                            is_expiry, base_lots, cfg_with_atr)
        if is_zero_hero:
            lots = min(cfg.get("zero_hero_max_lots", 3), max(1, lots))

        lots = max(1, int(lots * vix_lot_mult))
        if "orb" in entry_type:
            lots = max(1, int(lots * orb_lot_mult))
        lots = min(lots, max_lots_cap)

        opt_type = "CE" if action == "BUY_CALL" else "PE"
        atm = round(spot / STRIKE_INTERVAL) * STRIKE_INTERVAL
        strike = (atm + 200 if action == "BUY_CALL" else atm - 200) if is_zero_hero else atm
        qty = lots * LOT_SIZE
        entry_prem = calc_premium(spot, strike, dte, vix, opt_type, 1)

        trade = {
            "date": date_str, "action": action, "entry_bar": bar_idx,
            "entry_spot": spot, "best_fav": spot, "strike": strike,
            "opt_type": opt_type, "lots": lots, "qty": qty,
            "entry_premium": entry_prem, "entry_type": entry_type,
            "is_zero_hero": is_zero_hero, "confidence": conf, "vix": vix,
        }
        open_trades.append(trade)
        trades_today += 1

    eod_spot = bars[-1]["close"] if bars else 0
    for trade in open_trades:
        exit_prem = calc_premium(eod_spot, trade["strike"], max(0.05, dte * 0.1),
                                  vix, trade["opt_type"], -1)
        pnl = (exit_prem - trade["entry_premium"]) * trade["qty"] - BROKERAGE_RT
        trade.update({"exit_bar": len(bars) - 1, "exit_spot": eod_spot,
                       "exit_premium": exit_prem, "exit_reason": "eod_close",
                       "pnl": pnl, "bars_held": len(bars) - 1 - trade["entry_bar"]})
        closed_trades.append(trade)
        day_pnl += pnl

    return closed_trades, day_pnl, eod_spot


def run_variant(name, cfg, day_groups, all_dates, warmup_bars_init, vix_lookup):
    equity = CAPITAL
    all_trades = []
    prev_close = 0.0
    consecutive_down_days = 0
    warmup_bars = list(warmup_bars_init)
    monthly_pnl = defaultdict(float)
    max_equity = equity
    max_dd = 0.0

    for date in all_dates:
        bars = day_groups[date]
        if len(bars) < 5:
            continue
        vix = vix_lookup.get(date, 14.0)
        is_expiry = (date.weekday() == 3)

        trades, day_pnl, eod_close = simulate_day_r5b(
            bars, date, vix, cfg, prev_close, equity,
            warmup_bars, is_expiry, consecutive_down_days,
        )

        equity += day_pnl
        all_trades.extend(trades)
        monthly_pnl[f"{date.year}-{date.month:02d}"] += day_pnl

        if equity > max_equity:
            max_equity = equity
        dd = (max_equity - equity) / max_equity if max_equity > 0 else 0
        if dd > max_dd:
            max_dd = dd

        consecutive_down_days = (consecutive_down_days + 1
                                 if len(bars) >= 2 and bars[-1]["close"] < bars[0]["open"]
                                 else 0)
        warmup_bars = warmup_bars[-(75 * 2):] + bars
        prev_close = eod_close

    total_pnl = sum(t.get("pnl", 0) for t in all_trades)
    wins = [t for t in all_trades if t.get("pnl", 0) > 0]
    wr = len(wins) / len(all_trades) * 100 if all_trades else 0
    pf_num = sum(t["pnl"] for t in wins) if wins else 0
    pf_den = abs(sum(t["pnl"] for t in all_trades if t["pnl"] <= 0)) + 1

    return {
        "name": name, "trades": len(all_trades), "wins": len(wins),
        "win_rate": wr, "total_pnl": total_pnl, "equity": equity,
        "return_x": equity / CAPITAL, "profit_factor": pf_num / pf_den,
        "max_dd_pct": max_dd * 100, "monthly_pnl": dict(monthly_pnl),
    }


def make_variants():
    v = {}
    base = copy.deepcopy(V14_CONFIG)

    v["B0_Baseline"] = copy.deepcopy(base)

    # ── B1: PSAR filter only (R5 best risk-adjusted: 10.33x, PF 2.46, 48% DD) ──
    c = copy.deepcopy(base)
    c["use_psar_confluence"] = True
    v["B1_PSAR_Filter"] = c

    # ── B2: ATR sizing only (R5 highest return: 11.24x, 65% DD) ──
    c = copy.deepcopy(base)
    c["use_atr_sizing"] = True
    c["atr_reference"] = 80
    v["B2_ATR_Sizing"] = c

    # ── B3: PSAR filter + ATR sizing (combine #1 + #2) ──
    c = copy.deepcopy(base)
    c["use_psar_confluence"] = True
    c["use_atr_sizing"] = True
    c["atr_reference"] = 80
    v["B3_PSAR+ATR"] = c

    # ── B4: PSAR filter + ATR (ref=70) ──
    c = copy.deepcopy(base)
    c["use_psar_confluence"] = True
    c["use_atr_sizing"] = True
    c["atr_reference"] = 70
    v["B4_PSAR+ATR70"] = c

    # ── B5: PSAR filter + ATR (ref=90, more conservative) ──
    c = copy.deepcopy(base)
    c["use_psar_confluence"] = True
    c["use_atr_sizing"] = True
    c["atr_reference"] = 90
    v["B5_PSAR+ATR90"] = c

    # ── B6: PSAR + Donchian scoring ──
    c = copy.deepcopy(base)
    c["use_psar_confluence"] = True
    c["use_donchian"] = True
    c["donchian_score"] = 1.0
    v["B6_PSAR+Donchian"] = c

    # ── B7: PSAR + KAMA confluence ──
    c = copy.deepcopy(base)
    c["use_psar_confluence"] = True
    c["use_kama_confluence"] = True
    v["B7_PSAR+KAMA"] = c

    # ── B8: PSAR + Donchian + ATR ──
    c = copy.deepcopy(base)
    c["use_psar_confluence"] = True
    c["use_donchian"] = True
    c["donchian_score"] = 1.0
    c["use_atr_sizing"] = True
    c["atr_reference"] = 80
    v["B8_PSAR+Donch+ATR"] = c

    # ── B9: PSAR + KAMA + ATR ──
    c = copy.deepcopy(base)
    c["use_psar_confluence"] = True
    c["use_kama_confluence"] = True
    c["use_atr_sizing"] = True
    c["atr_reference"] = 80
    v["B9_PSAR+KAMA+ATR"] = c

    # ── B10: Top 4 all together ──
    c = copy.deepcopy(base)
    c["use_psar_confluence"] = True
    c["use_kama_confluence"] = True
    c["use_donchian"] = True
    c["donchian_score"] = 1.0
    c["use_atr_sizing"] = True
    c["atr_reference"] = 80
    v["B10_Top4_All"] = c

    # ── B11: PSAR + ATR + tighter lot cap 25 ──
    c = copy.deepcopy(base)
    c["use_psar_confluence"] = True
    c["use_atr_sizing"] = True
    c["atr_reference"] = 80
    c["max_lots_cap"] = 25
    v["B11_PSAR+ATR+Cap25"] = c

    # ── B12: PSAR + ATR + lot cap 27 (NSE freeze limit) ──
    c = copy.deepcopy(base)
    c["use_psar_confluence"] = True
    c["use_atr_sizing"] = True
    c["atr_reference"] = 80
    c["max_lots_cap"] = 27
    v["B12_PSAR+ATR+Cap27"] = c

    # ── B13: ATR sizing + lot cap 25 (reduce DD) ──
    c = copy.deepcopy(base)
    c["use_atr_sizing"] = True
    c["atr_reference"] = 80
    c["max_lots_cap"] = 25
    v["B13_ATR+Cap25"] = c

    # ── B14: Donchian + PSAR + ConnorsRSI (R5-27 winner combo) ──
    c = copy.deepcopy(base)
    c["use_connors_rsi"] = True
    c["connors_rsi_oversold"] = 15
    c["connors_rsi_overbought"] = 85
    c["connors_rsi_score"] = 1.5
    c["use_donchian"] = True
    c["donchian_score"] = 1.5
    c["use_psar"] = True
    c["psar_score"] = 0.8
    c["use_psar_confluence"] = True
    v["B14_Donch+PSAR+CRSI"] = c

    # ── B15: B14 + ATR sizing ──
    c = copy.deepcopy(v["B14_Donch+PSAR+CRSI"])
    c["use_atr_sizing"] = True
    c["atr_reference"] = 80
    v["B15_B14+ATR"] = c

    return v


def main():
    start_date = dt.date(2024, 7, 1)
    end_date = dt.date(2025, 1, 1)

    print("V14 Round 5B — Focused Combo Testing")
    print("=" * 110)
    print(f"Period: {start_date} to {end_date} | Capital: Rs {CAPITAL:,.0f} | Equity COMPOUNDED")
    print()

    print("Loading data...", flush=True)
    day_groups, all_dates, warmup_bars, vix_lookup = load_data(start_date, end_date)
    print(f"Trading days: {len(all_dates)}")
    print()

    variants = make_variants()
    results = []

    for name, cfg in variants.items():
        print(f"  {name}...", end="", flush=True)
        r = run_variant(name, cfg, day_groups, all_dates, warmup_bars, vix_lookup)
        results.append(r)
        print(f" {r['trades']} trades | {r['win_rate']:.1f}% WR | "
              f"Rs {r['total_pnl']:+,.0f} | {r['return_x']:.2f}x | "
              f"PF {r['profit_factor']:.2f} | MaxDD {r['max_dd_pct']:.1f}%")

    results.sort(key=lambda x: x["total_pnl"], reverse=True)
    baseline_pnl = next(r["total_pnl"] for r in results if r["name"] == "B0_Baseline")

    print()
    print("=" * 130)
    print(f"{'Rank':<4} {'Config':<25} {'Trades':>6} {'WR':>6} {'PnL':>14} {'Return':>8} {'PF':>6} {'MaxDD':>7} {'vs Base':>14}")
    print("-" * 130)
    for i, r in enumerate(results, 1):
        delta = r["total_pnl"] - baseline_pnl
        marker = " *** WINNER" if i == 1 else ""
        print(f"{i:<4} {r['name']:<25} {r['trades']:>6} {r['win_rate']:>5.1f}% "
              f"{r['total_pnl']:>+14,.0f} {r['return_x']:>7.2f}x {r['profit_factor']:>6.2f} "
              f"{r['max_dd_pct']:>6.1f}% {delta:>+13,.0f}{marker}")

    print()
    print("MONTHLY P&L — TOP 5")
    print("=" * 90)
    for r in results[:5]:
        print(f"\n{r['name']} ({r['return_x']:.2f}x, PF {r['profit_factor']:.2f}, DD {r['max_dd_pct']:.1f}%):")
        cum = CAPITAL
        for m in sorted(r["monthly_pnl"]):
            cum += r["monthly_pnl"][m]
            print(f"  {m}: Rs {r['monthly_pnl'][m]:>+12,.0f}  (Equity: Rs {cum:>12,.0f})")

    best = results[0]
    print(f"\nWINNER: {best['name']}")
    print(f"  {best['return_x']:.2f}x | Rs {best['total_pnl']:+,.0f} | "
          f"{best['trades']} trades | {best['win_rate']:.1f}% WR | "
          f"PF {best['profit_factor']:.2f} | MaxDD {best['max_dd_pct']:.1f}%")

    # Risk-adjusted ranking (Calmar-like: Return / MaxDD)
    print()
    print("RISK-ADJUSTED RANKING (Return / MaxDD):")
    print("-" * 90)
    risk_sorted = sorted(results, key=lambda x: x["return_x"] / (x["max_dd_pct"] / 100 + 0.01), reverse=True)
    for i, r in enumerate(risk_sorted[:10], 1):
        calmar = r["return_x"] / (r["max_dd_pct"] / 100 + 0.01)
        print(f"  {i}. {r['name']:<25} Calmar: {calmar:.2f}  ({r['return_x']:.2f}x / {r['max_dd_pct']:.1f}% DD)")


if __name__ == "__main__":
    main()
