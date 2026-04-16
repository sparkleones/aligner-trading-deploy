"""
V14 Round 4 — Data-driven improvements from trade-by-trade analysis.
=====================================================================
Base: R4_2Windows (3.61x, 81 trades)

Key findings from trade analysis:
  1. 33+ lots = catastrophic (5 trades lost Rs -4.96L)
  2. Wednesday = toxic (23.5% WR, -Rs 2.69L)
  3. trail_stop = 0% WR, loses Rs -1.19L
  4. VIX <13 = 10% WR
  5. VIX 14-15 = danger zone (PF 0.72)
  6. VIX 15+ = sweet spot (PF 5.41)
  7. bar 57-58 = 10% WR, -Rs 1.07L
  8. ORB = 75% WR, PF 12.16 — emphasize
  9. gap_zero_hero = net negative
  10. confidence <0.35 = PF 0.43
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


def simulate_day_v2(bars, date, vix, cfg, prev_close, equity, warmup_bars,
                    is_expiry, consecutive_down_days):
    """Improved day simulation with all data-driven improvements."""
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

    # Lot sizing
    span_per_lot = 40000 if vix < 20 else (50000 if vix < 25 else 60000)
    base_lots = max(1, int(equity * 0.70 / span_per_lot))

    # ── NEW: Max lot cap ──
    max_lots_cap = cfg.get("max_lots_cap", 999)

    # ── NEW: Day-of-week blocking ──
    avoid_days = cfg.get("avoid_days", [])
    if day_of_week in avoid_days:
        # Still process bars for warmup, but no entries
        eod_spot = bars[-1]["close"] if bars else 0
        return [], 0.0, eod_spot

    days_to_expiry = (3 - day_of_week) % 7
    dte = max(0.2, days_to_expiry) if days_to_expiry > 0 else 0.2

    entry_windows = cfg.get("entry_windows_bars", None)
    day_pnl = 0.0

    # ── NEW: VIX lot scaling ──
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

    # ── NEW: ORB lot boost ──
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

        # ── EXITS ──
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

        # ── ENTRIES ──
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

            # ── NEW: Filter out gap_zero_hero if disabled ──
            if cfg.get("disable_zero_hero", False):
                composites = [c for c in composites if not c[3]]  # c[3] = is_zero_hero

            if composites:
                composites.sort(key=lambda x: x[2], reverse=True)
                action, entry_type, conf, is_zero_hero = composites[0]
        else:
            # Entry window check
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

        # ── NEW: Confidence floor ──
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

        lots = compute_lots(action, conf, vix, indicators.get("rsi", 50),
                            is_expiry, base_lots, cfg)
        if is_zero_hero:
            lots = min(cfg.get("zero_hero_max_lots", 3), max(1, lots))

        # ── NEW: Apply VIX lot scaling ──
        lots = max(1, int(lots * vix_lot_mult))

        # ── NEW: ORB lot boost ──
        if "orb" in entry_type:
            lots = max(1, int(lots * orb_lot_mult))

        # ── NEW: Enforce lot cap ──
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

    # Force close
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

        trades, day_pnl, eod_close = simulate_day_v2(
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

        consecutive_down_days = consecutive_down_days + 1 if len(bars) >= 2 and bars[-1]["close"] < bars[0]["open"] else 0
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

    # ── T0: Current R4 baseline ──
    v["T0_Baseline"] = copy.deepcopy(base)

    # ── T1: Cap lots at 32 ──
    c = copy.deepcopy(base); c["max_lots_cap"] = 32
    v["T1_LotCap32"] = c

    # ── T2: Cap lots at 25 ──
    c = copy.deepcopy(base); c["max_lots_cap"] = 25
    v["T2_LotCap25"] = c

    # ── T3: Cap lots at 20 ──
    c = copy.deepcopy(base); c["max_lots_cap"] = 20
    v["T3_LotCap20"] = c

    # ── T4: Block Wednesday ──
    c = copy.deepcopy(base); c["avoid_days"] = [2]  # Wed=2
    v["T4_NoWednesday"] = c

    # ── T5: Disable trail_stop (set min_hold to 999) ──
    c = copy.deepcopy(base)
    c["min_hold_trail_put"] = 999; c["min_hold_trail_call"] = 999
    v["T5_NoTrailStop"] = c

    # ── T6: Raise VIX floor to 13 ──
    c = copy.deepcopy(base); c["vix_floor"] = 13
    v["T6_VIXFloor13"] = c

    # ── T7: VIX-adaptive lot scaling ──
    c = copy.deepcopy(base)
    c["vix_lot_scaling"] = True
    c["vix_below13_mult"] = 0.3   # Near-zero in low VIX
    c["vix_14_15_mult"] = 0.5     # Half lots in danger zone
    c["vix_15_17_mult"] = 1.5     # Boost in sweet spot
    c["vix_17plus_mult"] = 2.0    # Double in high VIX
    v["T7_VIXLotScale"] = c

    # ── T8: Narrow afternoon window (59-69, skip bar 57-58) ──
    c = copy.deepcopy(base)
    c["entry_windows_bars"] = [(3, 15), (59, 69)]
    v["T8_SkipBar5758"] = c

    # ── T9: Disable gap_zero_hero ──
    c = copy.deepcopy(base); c["disable_zero_hero"] = True
    v["T9_NoZeroHero"] = c

    # ── T10: Raise min_confidence to 0.35 (already default, but enforce) ──
    c = copy.deepcopy(base); c["min_confidence"] = 0.38
    v["T10_HigherConf"] = c

    # ── T11: ORB lot boost (2x lots on ORB entries) ──
    c = copy.deepcopy(base); c["orb_lot_mult"] = 2.0
    v["T11_ORBBoost2x"] = c

    # ── T12: ORB lot boost 3x ──
    c = copy.deepcopy(base); c["orb_lot_mult"] = 3.0
    v["T12_ORBBoost3x"] = c

    # ── COMBOS ──

    # ── T13: LotCap25 + NoWednesday ──
    c = copy.deepcopy(base)
    c["max_lots_cap"] = 25; c["avoid_days"] = [2]
    v["T13_Cap25+NoWed"] = c

    # ── T14: Cap25 + NoWed + NoTrail ──
    c = copy.deepcopy(base)
    c["max_lots_cap"] = 25; c["avoid_days"] = [2]
    c["min_hold_trail_put"] = 999; c["min_hold_trail_call"] = 999
    v["T14_Cap25+NoWed+NoTrail"] = c

    # ── T15: T14 + VIX lot scaling ──
    c = copy.deepcopy(base)
    c["max_lots_cap"] = 25; c["avoid_days"] = [2]
    c["min_hold_trail_put"] = 999; c["min_hold_trail_call"] = 999
    c["vix_lot_scaling"] = True
    c["vix_below13_mult"] = 0.3; c["vix_14_15_mult"] = 0.5
    c["vix_15_17_mult"] = 1.5; c["vix_17plus_mult"] = 2.0
    v["T15_T14+VIXScale"] = c

    # ── T16: T15 + skip bar 57-58 ──
    c = copy.deepcopy(base)
    c["max_lots_cap"] = 25; c["avoid_days"] = [2]
    c["min_hold_trail_put"] = 999; c["min_hold_trail_call"] = 999
    c["vix_lot_scaling"] = True
    c["vix_below13_mult"] = 0.3; c["vix_14_15_mult"] = 0.5
    c["vix_15_17_mult"] = 1.5; c["vix_17plus_mult"] = 2.0
    c["entry_windows_bars"] = [(3, 15), (59, 69)]
    v["T16_T15+SkipBar5758"] = c

    # ── T17: T16 + VIX floor 13 + no zero hero ──
    c = copy.deepcopy(base)
    c["max_lots_cap"] = 25; c["avoid_days"] = [2]
    c["min_hold_trail_put"] = 999; c["min_hold_trail_call"] = 999
    c["vix_lot_scaling"] = True
    c["vix_below13_mult"] = 0.3; c["vix_14_15_mult"] = 0.5
    c["vix_15_17_mult"] = 1.5; c["vix_17plus_mult"] = 2.0
    c["entry_windows_bars"] = [(3, 15), (59, 69)]
    c["vix_floor"] = 13; c["disable_zero_hero"] = True
    v["T17_AllImprovements"] = c

    # ── T18: T17 + ORB boost 2x ──
    c = copy.deepcopy(v["T17_AllImprovements"])
    c["orb_lot_mult"] = 2.0
    v["T18_All+ORBBoost"] = c

    # ── T19: T18 + lot cap 20 (tighter) ──
    c = copy.deepcopy(v["T18_All+ORBBoost"])
    c["max_lots_cap"] = 20
    v["T19_All+Cap20"] = c

    # ── T20: T18 + lot cap 30 (looser) ──
    c = copy.deepcopy(v["T18_All+ORBBoost"])
    c["max_lots_cap"] = 30
    v["T20_All+Cap30"] = c

    return v


def main():
    start_date = dt.date(2024, 7, 1)
    end_date = dt.date(2025, 1, 1)

    print("V14 Round 4 — Data-Driven Improvements")
    print("=" * 90)
    print(f"Period: {start_date} to {end_date} | Capital: Rs {CAPITAL:,.0f}")
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

    print()
    print("=" * 115)
    print(f"{'Rank':<4} {'Config':<30} {'Trades':>6} {'WR':>6} {'PnL':>14} {'Return':>8} {'PF':>6} {'MaxDD':>7}")
    print("-" * 115)
    for i, r in enumerate(results, 1):
        marker = " ***" if i == 1 else ""
        print(f"{i:<4} {r['name']:<30} {r['trades']:>6} {r['win_rate']:>5.1f}% "
              f"{r['total_pnl']:>+14,.0f} {r['return_x']:>7.2f}x {r['profit_factor']:>6.2f} "
              f"{r['max_dd_pct']:>6.1f}%{marker}")

    # Monthly for top 3
    print()
    print("MONTHLY P&L — TOP 3")
    print("=" * 80)
    for r in results[:3]:
        print(f"\n{r['name']} ({r['return_x']:.2f}x):")
        cum = CAPITAL
        for m in sorted(r["monthly_pnl"]):
            cum += r["monthly_pnl"][m]
            print(f"  {m}: Rs {r['monthly_pnl'][m]:>+12,.0f}  (Equity: Rs {cum:>12,.0f})")

    best = results[0]
    print(f"\nWINNER: {best['name']}")
    print(f"  {best['return_x']:.2f}x | Rs {best['total_pnl']:+,.0f} | "
          f"{best['trades']} trades | {best['win_rate']:.1f}% WR | "
          f"PF {best['profit_factor']:.2f} | MaxDD {best['max_dd_pct']:.1f}%")

    # Improvement over baseline
    baseline = next(r for r in results if r["name"] == "T0_Baseline")
    print(f"\n  vs Baseline: {best['return_x']/baseline['return_x']:.1f}x improvement "
          f"({best['total_pnl'] - baseline['total_pnl']:+,.0f} more profit)")


if __name__ == "__main__":
    main()
