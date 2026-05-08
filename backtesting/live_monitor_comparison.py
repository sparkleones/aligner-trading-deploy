"""LiveTradeMonitor counterfactual comparison backtest.

Runs ONE V14 backtest (unchanged entries and exits) and, for every
trade, also computes what the LiveTradeMonitor *would* have done on
each bar from entry to actual V14 exit.

Per trade we record:
    pnl_v14   : what actually happened under V14 rules
    pnl_mon   : what would have happened if Monitor had been armed and
                fired before V14's exit. If Monitor never fires, equal
                to pnl_v14.

This isolates the *exit policy* difference — entries are identical,
capital is identical, portfolio state is identical. Any P&L delta is
100% attributable to Monitor changing the exit bar.

Usage:
    python -m backtesting.live_monitor_comparison               # default 6 months
    python -m backtesting.live_monitor_comparison --months 21
    python -m backtesting.live_monitor_comparison --start 2024-07-01 --months 21

CLI knobs expose the key Monitor thresholds so you can sweep them
without editing code.
"""

import argparse
import datetime as dt
import sys
from collections import defaultdict
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backtesting.v14_unified_backtest import (    # noqa: E402
    load_nifty_data,
    load_vix_data,
    resample_to_5min,
    bars_to_dicts,
    calc_premium,
    load_period_data,
    CAPITAL,
    LOT_SIZE,
    BROKERAGE_RT,
    get_strike_and_type,
    _v17_regime_blocked,
    _v17_monwed_gate_blocked,
    _v17_pick_product,
)
from scoring.config import V15_CONFIG as V14_CONFIG  # noqa: E402
from scoring.indicators import compute_indicators  # noqa: E402
from scoring.engine import (    # noqa: E402
    score_entry,
    passes_confluence,
    evaluate_exit,
    compute_lots,
    detect_composite_entries,
)
from orchestrator.live_trade_monitor import monitor_check, MONITOR_DEFAULTS  # noqa: E402


# ═══════════════════════════════════════════════════════════════════
# SINGLE-PORTFOLIO SIMULATION WITH COUNTERFACTUAL MONITOR SHADOW
# ═══════════════════════════════════════════════════════════════════
# Entries, capacity, cooldown, BTST carries — all V14 baseline.
# For each open trade we ALSO track a "monitor shadow" state. On every
# bar we call monitor_check; the FIRST bar Monitor would have fired is
# recorded as the counterfactual exit. The trade continues to exist in
# the portfolio until V14 actually exits it, so future entries are
# unaffected.

def _close_trade(trade, bar_idx, spot, exit_reason, dte, vix, premium=None):
    carry_dte_corr = max(0.05, dte - bar_idx * 5 / (6.25 * 60))
    if premium is None:
        premium = calc_premium(
            spot, trade["strike"], carry_dte_corr, vix,
            trade["opt_type"], slippage_sign=-1,
        )
    pnl = (premium - trade["entry_premium"]) * trade["qty"] - BROKERAGE_RT
    trade["exit_bar"] = bar_idx
    trade["exit_spot"] = spot
    trade["exit_premium"] = premium
    trade["exit_reason"] = exit_reason
    trade["pnl"] = pnl
    trade["bars_held"] = bar_idx - trade["entry_bar"]
    return pnl


def simulate_day_counterfactual(
    date,
    bars_5min,
    vix,
    cfg,
    prev_close=None,
    consecutive_down_days=0,
    btst_carry=None,
    is_expiry=False,
    monitor_cfg=None,
    warmup_bars=None,
):
    """Single V14 portfolio. Each trade gets a counterfactual Monitor shadow."""
    date_str = str(date)
    open_trades: list = []
    closed_trades: list = []
    btst_open = list(btst_carry or [])
    # Seed bar_history with prior-day warmup so indicators are armed on bar 0.
    # Real V14 driver carries ~3 days of bars before start_date (see
    # v14_unified_backtest.load_period_data) — same pattern here.
    bar_history: list = list(warmup_bars or [])
    last_exit_bar = -999
    trades_today = 0
    day_pnl = 0.0
    day_open = bars_5min[0]["open"] if bars_5min else 0
    orb_high = orb_low = 0
    gap_detected = False

    day_of_week = date.weekday()
    days_to_expiry = (3 - day_of_week) % 7
    dte = 0.2 if days_to_expiry == 0 else float(days_to_expiry)

    base_lots = max(1, int(CAPITAL / (CAPITAL / 3)))
    btst_exit_bar = 6

    def _update_monitor_shadow(trade, bar_idx, spot, ind):
        """Update the Monitor shadow state for a still-open trade.

        If Monitor would have fired this bar AND hasn't fired yet on a
        prior bar, record the counterfactual exit.
        """
        if trade.get("mon_fired"):
            return
        cur_prem = calc_premium(
            spot, trade["strike"],
            max(0.05, dte - bar_idx * 5 / (6.25 * 60)),
            vix, trade["opt_type"], slippage_sign=-1,
        )
        mon_reason = monitor_check(trade, ind, cur_prem, bar_idx, monitor_cfg)
        if mon_reason:
            mon_pnl = (cur_prem - trade["entry_premium"]) * trade["qty"] - BROKERAGE_RT
            trade["mon_fired"] = True
            trade["mon_exit_bar"] = bar_idx
            trade["mon_exit_spot"] = spot
            trade["mon_exit_premium"] = cur_prem
            trade["mon_exit_reason"] = mon_reason
            trade["mon_pnl"] = mon_pnl

    for bar_idx, bar in enumerate(bars_5min):
        bar_history.append(bar)
        if len(bar_history) > 500:
            bar_history = bar_history[-500:]
        spot = bar["close"]

        if bar_idx == 0:
            orb_high = bar["high"]
            orb_low = bar["low"]
        elif bar_idx == 1:
            orb_high = max(orb_high, bar["high"])
            orb_low = min(orb_low, bar["low"])

        indicators = compute_indicators(bar_history, date_str)
        ind = indicators or {}

        # ── BTST carries (V14 baseline exits only; Monitor shadow runs) ──
        for trade in list(btst_open):
            # Shadow first so peak_premium gets updated on every bar
            _update_monitor_shadow(trade, bar_idx, spot, ind)
            exit_reason = None
            if indicators:
                exit_reason = evaluate_exit(
                    trade, bar_idx, spot, indicators, cfg,
                    day_of_week=day_of_week,
                )
            if not exit_reason and bar_idx >= btst_exit_bar:
                exit_reason = "btst_deadline"
            if exit_reason:
                pnl = _close_trade(trade, bar_idx, spot, exit_reason, dte, vix)
                closed_trades.append(trade)
                btst_open.remove(trade)
                day_pnl += pnl

        # ── EXITS for today's positions ──
        for trade in list(open_trades):
            if trade["action"] == "BUY_CALL" and spot > trade["best_fav"]:
                trade["best_fav"] = spot
            elif trade["action"] == "BUY_PUT" and spot < trade["best_fav"]:
                trade["best_fav"] = spot

            # Monitor shadow BEFORE V14 exit so peak_premium tracks every bar
            _update_monitor_shadow(trade, bar_idx, spot, ind)

            exit_reason = evaluate_exit(
                trade, bar_idx, spot, ind, cfg, day_of_week=day_of_week,
            )
            if exit_reason == "eod_close" and trade.get("product") == "NRML":
                exit_reason = None
            if exit_reason:
                pnl = _close_trade(trade, bar_idx, spot, exit_reason, dte, vix)
                closed_trades.append(trade)
                open_trades.remove(trade)
                day_pnl += pnl
                last_exit_bar = bar_idx

        # ── ENTRIES (V14 unchanged) ──
        if indicators is None:
            continue

        entry_windows = cfg.get("entry_windows_bars")
        action = None
        conf = 0
        entry_type = "v8_indicator"
        is_zero_hero = False

        if bar_idx < 3:
            prev_spot = bars_5min[bar_idx - 1]["close"] if bar_idx > 0 else spot
            composites = detect_composite_entries(
                bar, bar_idx, spot, vix, cfg,
                prev_close=prev_close, gap_detected=gap_detected,
                orb_high=orb_high, orb_low=orb_low,
                prev_spot=prev_spot,
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
            if bar_idx - last_exit_bar < cfg.get("cooldown_bars", 0):
                continue
            action, conf = score_entry(
                indicators, vix, cfg,
                bar_idx=bar_idx,
                consecutive_down_days=consecutive_down_days,
            )
            if action is None:
                prev_spot = bars_5min[bar_idx - 1]["close"] if bar_idx > 0 else spot
                composites = detect_composite_entries(
                    bar, bar_idx, spot, vix, cfg,
                    prev_close=prev_close, gap_detected=gap_detected,
                    orb_high=orb_high, orb_low=orb_low,
                    prev_spot=prev_spot,
                )
                if cfg.get("disable_zero_hero", False):
                    composites = [c for c in composites if not c[3]]
                if composites:
                    composites.sort(key=lambda x: x[2], reverse=True)
                    action, entry_type, conf, is_zero_hero = composites[0]

        if action is None:
            continue
        if vix < cfg["vix_floor"] or vix > cfg["vix_ceil"]:
            continue
        if not passes_confluence(
            action, conf, indicators, bar_idx, cfg,
            current_spot=spot, prev_close=prev_close, day_open=day_open,
        ):
            continue
        if _v17_regime_blocked(cfg, indicators, vix, spot, conf=conf):
            continue
        if _v17_monwed_gate_blocked(
            cfg, indicators, vix, spot, bar_idx, day_of_week, action,
        ):
            continue
        if len(open_trades) >= cfg["max_concurrent"]:
            continue
        if trades_today >= cfg["max_trades_per_day"]:
            continue
        if any(t["action"] == action for t in open_trades):
            continue
        if any(t["action"] == action for t in btst_open):
            continue

        cfg_with_atr = cfg
        if cfg.get("use_atr_sizing"):
            cfg_with_atr = cfg.copy()
            cfg_with_atr["_current_atr"] = indicators.get("atr", 0)
        lots = compute_lots(
            action, conf, vix, indicators.get("rsi", 50),
            is_expiry, base_lots, cfg_with_atr,
        )
        if is_zero_hero:
            lots = min(cfg.get("zero_hero_max_lots", 3), max(1, lots))
        if cfg.get("vix_lot_scaling"):
            if vix < 13:
                lots = max(1, int(lots * cfg.get("vix_below13_mult", 0.3)))
            elif 14 <= vix < 15:
                lots = max(1, int(lots * cfg.get("vix_14_15_mult", 0.5)))
            elif 15 <= vix < 17:
                lots = max(1, int(lots * cfg.get("vix_15_17_mult", 1.5)))
            elif vix >= 17:
                lots = max(1, int(lots * cfg.get("vix_17plus_mult", 2.0)))
        if entry_type and "orb" in entry_type:
            lots = max(1, int(lots * cfg.get("orb_lot_mult", 1.0)))
        lots = min(lots, cfg.get("max_lots_cap", 999))

        strike, opt_type = get_strike_and_type(action, spot, vix, zero_hero=is_zero_hero)
        qty = lots * LOT_SIZE
        entry_prem = calc_premium(spot, strike, dte, vix, opt_type, slippage_sign=1)

        day_bars_so_far = bars_5min[: bar_idx + 1]
        day_high_sf = max(b["high"] for b in day_bars_so_far)
        day_low_sf = min(b["low"] for b in day_bars_so_far)
        day_open_sf = day_bars_so_far[0]["open"]
        product = _v17_pick_product(
            cfg, indicators, action, conf, bar_idx, dte, vix,
            spot=spot, day_high=day_high_sf,
            day_low=day_low_sf, day_open=day_open_sf,
        )
        trade = {
            "date": date_str,
            "action": action,
            "entry_bar": bar_idx,
            "entry_spot": spot,
            "best_fav": spot,
            "strike": strike,
            "opt_type": opt_type,
            "lots": lots,
            "qty": qty,
            "entry_premium": entry_prem,
            "entry_type": entry_type,
            "is_zero_hero": is_zero_hero,
            "confidence": conf,
            "vix": vix,
            "product": product,
            "dte_at_entry": dte,
            # Monitor shadow state
            "peak_premium": entry_prem,
            "underwater_bars": 0,
            "in_profit_phase": False,
            "mon_prev_macd_hist": float(indicators.get("macd_hist", 0.0) or 0.0),
            "mon_fired": False,
            "mon_exit_bar": None,
            "mon_exit_reason": None,
            "mon_pnl": None,
        }
        open_trades.append(trade)
        trades_today += 1

    # ── EOD handling ──
    eod_spot = bars_5min[-1]["close"] if bars_5min else 0
    btst_carry_next: list = []
    for trade in open_trades:
        if trade.get("product") == "NRML":
            btst_carry_next.append(trade)
            continue
        bar_idx = len(bars_5min) - 1
        # Shadow update on final bar
        _update_monitor_shadow(trade, bar_idx, eod_spot, ind)
        _close_trade(trade, bar_idx, eod_spot, "eod_close", dte, vix)
        closed_trades.append(trade)
        day_pnl += trade["pnl"]

    # Finalize mon_pnl for trades Monitor never fired on
    for t in closed_trades:
        if t.get("mon_pnl") is None:
            t["mon_pnl"] = t["pnl"]
            t["mon_exit_reason"] = t["exit_reason"]
            t["mon_exit_bar"] = t["exit_bar"]

    return {
        "closed": closed_trades,
        "day_pnl_v14": day_pnl,
        "day_pnl_mon": sum(t["mon_pnl"] for t in closed_trades),
        "btst_carry": btst_carry_next,
    }


# ═══════════════════════════════════════════════════════════════════
# DRIVER
# ═══════════════════════════════════════════════════════════════════

def run_comparison(start_date, months, monitor_cfg=None, quiet=False,
                   cfg_override=None, base_cfg=None):
    end_date = start_date + dt.timedelta(days=months * 31)
    # Use the shared loader so we get warmup_bars (3 days pre-start)
    # exactly like v14_unified_backtest.run_backtest does.
    day_groups, vix_lookup, all_dates, warmup_bars = load_period_data(
        start_date=start_date, end_date=end_date, months=months, quiet=quiet,
    )
    if not quiet:
        print(f"Loaded {len(all_dates)} trading days, warmup={len(warmup_bars)} bars",
              flush=True)

    cfg = dict(base_cfg if base_cfg is not None else V14_CONFIG)
    if cfg_override:
        cfg.update(cfg_override)

    all_trades: list = []
    daily_v14: dict = defaultdict(float)
    daily_mon: dict = defaultdict(float)
    btst_carry: list = []
    prev_close = 0.0
    consecutive_down = 0

    day_count = 0
    avoid_days = cfg.get("avoid_days", [])
    for date in all_dates:
        bars_5min = day_groups[date]
        if len(bars_5min) < 5:
            continue
        # ── Day-of-week block (matches v14_unified_backtest.simulate_day) ──
        # V15 sets [0,2] (block Mon+Wed); current live has [2] only. Without
        # this check the harness takes trades V14 would never touch.
        if date.weekday() in avoid_days:
            # Still need to exit any BTST carry on a blocked day, but no
            # new entries allowed. Simpler: drop the carry (matches behavior
            # closely enough for P&L comparison; real harness does orderly exit).
            btst_carry = []
            continue
        date_str = str(date)
        vix = vix_lookup.get(date, 14.0)
        is_expiry = (date.weekday() == 3)

        result = simulate_day_counterfactual(
            date, bars_5min, vix, cfg,
            prev_close=prev_close,
            consecutive_down_days=consecutive_down,
            btst_carry=btst_carry,
            is_expiry=is_expiry,
            monitor_cfg=monitor_cfg,
            warmup_bars=warmup_bars,
        )
        all_trades.extend(result["closed"])
        daily_v14[date_str] = result["day_pnl_v14"]
        daily_mon[date_str] = result["day_pnl_mon"]
        btst_carry = result["btst_carry"]

        prev_close_today = bars_5min[-1]["close"]
        if prev_close is not None:
            if prev_close_today < prev_close:
                consecutive_down += 1
            else:
                consecutive_down = 0
        prev_close = prev_close_today
        day_count += 1

    return {
        "trades": all_trades,
        "daily_v14": dict(daily_v14),
        "daily_mon": dict(daily_mon),
        "day_count": day_count,
    }


# ═══════════════════════════════════════════════════════════════════
# REPORTING
# ═══════════════════════════════════════════════════════════════════

def _metrics(trades: list, pnl_key: str, daily_pnl: dict, capital: float):
    if not trades:
        return {"trades": 0, "pnl": 0, "wr": 0, "pf": 0,
                "maxdd": 0, "avg_win": 0, "avg_loss": 0,
                "worst": 0, "best": 0, "ret_mult": 1.0}
    pnls = [t[pnl_key] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    total = sum(pnls)
    wr = len(wins) / len(pnls) * 100
    gross_win = sum(wins) if wins else 0
    gross_loss = -sum(losses) if losses else 1e-9
    pf = gross_win / gross_loss if gross_loss else 0

    running = capital
    peak = capital
    max_dd = 0.0
    for d in sorted(daily_pnl.keys()):
        running += daily_pnl[d]
        peak = max(peak, running)
        dd = (peak - running) / peak * 100
        max_dd = max(max_dd, dd)

    return {
        "trades": len(pnls),
        "pnl": total,
        "wr": wr,
        "pf": pf,
        "maxdd": max_dd,
        "avg_win": sum(wins) / len(wins) if wins else 0,
        "avg_loss": sum(losses) / len(losses) if losses else 0,
        "worst": min(pnls),
        "best": max(pnls),
        "ret_mult": (capital + total) / capital,
    }


def _print_report(res, monitor_cfg):
    capital = CAPITAL
    m_v = _metrics(res["trades"], "pnl", res["daily_v14"], capital)
    m_m = _metrics(res["trades"], "mon_pnl", res["daily_mon"], capital)

    # Monitor intervention analysis
    mon_fired_trades = [t for t in res["trades"] if t.get("mon_fired")]
    fire_count = len(mon_fired_trades)
    improve = sum(1 for t in mon_fired_trades if t["mon_pnl"] > t["pnl"])
    worsen = sum(1 for t in mon_fired_trades if t["mon_pnl"] < t["pnl"])
    same = fire_count - improve - worsen
    total_delta = sum(t["mon_pnl"] - t["pnl"] for t in mon_fired_trades)

    print("\n" + "=" * 88)
    print("  LIVE TRADE MONITOR  —  Counterfactual comparison over shared V14 portfolio")
    print("=" * 88)
    print(f"  Days simulated : {res['day_count']}")
    print(f"  Trades         : {len(res['trades'])}")
    print(f"  Capital        : Rs {capital:,}")
    print(f"  Monitor knobs  : trail_activate={monitor_cfg.get('mon_prem_trail_activate_mult')} "
          f"giveback={monitor_cfg.get('mon_prem_trail_giveback_frac')} "
          f"emergency=[peak>={monitor_cfg.get('mon_emergency_peak_mult')},"
          f"floor<={monitor_cfg.get('mon_emergency_floor_frac')}] "
          f"rev_min={monitor_cfg.get('mon_reversal_score_min')}")
    print("-" * 88)

    print(f"  {'Metric':<22}{'V14 baseline':>22}{'V14 + Monitor':>22}{'Delta':>22}")
    print("-" * 88)

    def row(label, key, fmt="{:>+20,.0f}"):
        a = m_v[key]
        b = m_m[key]
        diff = b - a
        print(f"  {label:<22}"
              f"{fmt.format(a).strip():>22}"
              f"{fmt.format(b).strip():>22}"
              f"{fmt.format(diff).strip():>22}")

    row("Trades", "trades", "{:>10}")
    row("Total P&L (Rs)", "pnl")
    row("Win rate (%)", "wr", "{:>10.1f}")
    row("Profit factor", "pf", "{:>10.2f}")
    row("Max drawdown (%)", "maxdd", "{:>10.1f}")
    row("Avg win (Rs)", "avg_win")
    row("Avg loss (Rs)", "avg_loss")
    row("WORST trade (Rs)", "worst")
    row("BEST trade (Rs)", "best")
    row("Return multiple", "ret_mult", "{:>10.2f}")

    print("-" * 88)
    print("  MONITOR INTERVENTIONS")
    print("-" * 88)
    print(f"  Trades where Monitor fired before V14  : {fire_count} / {len(res['trades'])}")
    print(f"    Improved outcome (mon_pnl > v14_pnl)  : {improve}")
    print(f"    Worsened outcome (mon_pnl < v14_pnl)  : {worsen}")
    print(f"    Same (Monitor tie-broke)              : {same}")
    print(f"    Net P&L delta from Monitor            : Rs {total_delta:+,.0f}")

    # Per-monitor-reason breakdown
    print("-" * 88)
    print("  MONITOR REASON BREAKDOWN  (count | sum_delta | improved | worsened)")
    print("-" * 88)
    by_reason: dict = defaultdict(lambda: {
        "count": 0, "delta_sum": 0.0, "improved": 0, "worsened": 0,
    })
    for t in mon_fired_trades:
        r = t["mon_exit_reason"]
        by_reason[r]["count"] += 1
        delta = t["mon_pnl"] - t["pnl"]
        by_reason[r]["delta_sum"] += delta
        if delta > 0:
            by_reason[r]["improved"] += 1
        elif delta < 0:
            by_reason[r]["worsened"] += 1
    for r, d in sorted(by_reason.items()):
        print(f"  {r:<24}  {d['count']:>4}  |  Rs {d['delta_sum']:>+13,.0f}  |  "
              f"imp {d['improved']:>3}  |  wor {d['worsened']:>3}")

    # Runaway-loss analysis — the "+4000 to -11000" case
    print("-" * 88)
    print("  RUNAWAY-LOSS RESCUE (V14 trades: peak >= 1.15 * entry, then pnl < -3000)")
    print("-" * 88)
    runaways = [
        t for t in res["trades"]
        if t.get("peak_premium", 0) >= t["entry_premium"] * 1.15
        and t["pnl"] < -3000
    ]
    if runaways:
        rescued = [t for t in runaways if t["mon_pnl"] > t["pnl"]]
        total_v14 = sum(t["pnl"] for t in runaways)
        total_mon = sum(t["mon_pnl"] for t in runaways)
        print(f"  Runaway trades       : {len(runaways)}")
        print(f"  Rescued by Monitor   : {len(rescued)} "
              f"({len(rescued) / len(runaways) * 100:.0f}%)")
        print(f"  V14 loss on runaways : Rs {total_v14:+,.0f}")
        print(f"  Monitor on runaways  : Rs {total_mon:+,.0f}  (rescue Rs {total_mon - total_v14:+,.0f})")
    else:
        print("  No runaway trades matched criteria.")

    print("=" * 88)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start", type=str, default="2024-07-01")
    p.add_argument("--months", type=int, default=6)
    p.add_argument("--trail-activate", type=float,
                   default=MONITOR_DEFAULTS["mon_prem_trail_activate_mult"])
    p.add_argument("--trail-giveback", type=float,
                   default=MONITOR_DEFAULTS["mon_prem_trail_giveback_frac"])
    p.add_argument("--emergency-peak", type=float,
                   default=MONITOR_DEFAULTS["mon_emergency_peak_mult"])
    p.add_argument("--emergency-floor", type=float,
                   default=MONITOR_DEFAULTS["mon_emergency_floor_frac"])
    p.add_argument("--rev-score-min", type=int,
                   default=MONITOR_DEFAULTS["mon_reversal_score_min"])
    p.add_argument("--struct-uw-bars", type=int,
                   default=MONITOR_DEFAULTS["mon_structural_underwater_bars"])
    p.add_argument("--struct-score-min", type=int,
                   default=MONITOR_DEFAULTS["mon_structural_score_min"])
    p.add_argument("--config", type=str, default="V15",
                   choices=["V15", "V17_PROD_ONLY"],
                   help="Which baseline config to run Monitor counterfactual against")
    args = p.parse_args()

    start_date = dt.datetime.strptime(args.start, "%Y-%m-%d").date()
    monitor_cfg = {
        **MONITOR_DEFAULTS,
        "mon_prem_trail_activate_mult": args.trail_activate,
        "mon_prem_trail_giveback_frac": args.trail_giveback,
        "mon_emergency_peak_mult": args.emergency_peak,
        "mon_emergency_floor_frac": args.emergency_floor,
        "mon_reversal_score_min": args.rev_score_min,
        "mon_structural_underwater_bars": args.struct_uw_bars,
        "mon_structural_score_min": args.struct_score_min,
    }

    # Resolve baseline config
    base_cfg = None
    cfg_override = None
    if args.config == "V17_PROD_ONLY":
        from scoring.config import V17_CONFIG
        base_cfg = dict(V17_CONFIG)
        cfg_override = {
            "avoid_days": [0, 2],
            "use_v17_regime_gate": False,
            "use_v17_monwed_gate": False,
        }
        print(f"[config] V17_PROD_ONLY  avoid_days={cfg_override['avoid_days']}  "
              f"BTST={base_cfg.get('use_v17_dynamic_product', False)}",
              flush=True)

    res = run_comparison(start_date, args.months, monitor_cfg=monitor_cfg,
                         base_cfg=base_cfg, cfg_override=cfg_override)
    _print_report(res, monitor_cfg)


if __name__ == "__main__":
    main()
