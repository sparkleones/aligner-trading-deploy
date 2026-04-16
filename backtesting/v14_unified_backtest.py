"""
V14 Unified 5-Min Backtest — Uses the SAME scoring engine as live trading.
==========================================================================
This backtester imports scoring/engine.py and scoring/indicators.py — the
exact same code that runs in the live V14 agent. This eliminates backtest-vs-live
divergence by having ONE source of truth.

Data: Resamples 1-min historical NIFTY data to 5-min bars.
P&L: Uses Black-Scholes option pricing for realistic P&L estimation.
Period: 6 months (configurable via --months or --start/--end).

Usage:
    python -m backtesting.v14_unified_backtest
    python -m backtesting.v14_unified_backtest --months 6
    python -m backtesting.v14_unified_backtest --start 2024-07-01 --end 2025-01-01
"""

import sys
import argparse
import datetime as dt
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# ── THE SHARED ENGINE (same code as live) ──
from scoring.config import V15_CONFIG as V14_CONFIG  # V15 deployed (LEGACY-restored config + V15 quality filters)
from scoring.indicators import compute_indicators
from scoring.engine import (
    score_entry,
    passes_confluence,
    evaluate_exit,
    compute_lots,
    detect_composite_entries,
    v17_btst_favorable as _shared_v17_btst_favorable,
)

# ── Backtest-only imports (option pricing, strike selection) ──
from backtesting.option_pricer import price_option

DATA_DIR = project_root / "data" / "historical"
CAPITAL = 200_000
LOT_SIZE = 75  # NIFTY lot size for 2024 backtest period
STRIKE_INTERVAL = 50
SLIPPAGE_PCT = 0.005   # 0.5% entry/exit slippage
SPREAD_RS = 2.0        # Rs 2 bid-ask spread
BROKERAGE_RT = 80.0    # Rs 80 round-trip brokerage


# ─────────────────────────────────────────────────────────────
# DATA LOADING AND RESAMPLING
# ─────────────────────────────────────────────────────────────

def load_nifty_data(start_date, end_date):
    """Load and combine NIFTY 1-min data files covering the date range."""
    all_dfs = []
    for f in sorted(DATA_DIR.glob("nifty_min_*.csv")):
        try:
            df = pd.read_csv(f, parse_dates=["timestamp"], index_col="timestamp")
            # Filter to date range
            mask = (df.index.date >= start_date) & (df.index.date <= end_date)
            chunk = df[mask]
            if len(chunk) > 0:
                all_dfs.append(chunk)
        except Exception:
            continue
    if not all_dfs:
        raise FileNotFoundError(f"No NIFTY data found for {start_date} to {end_date}")
    combined = pd.concat(all_dfs)
    combined = combined[~combined.index.duplicated(keep="first")]
    combined.sort_index(inplace=True)
    return combined


def load_vix_data(start_date, end_date):
    """Load VIX daily close lookup from 1-min VIX files."""
    vix_lookup = {}
    for f in sorted(DATA_DIR.glob("vix_min_*.csv")):
        try:
            df = pd.read_csv(f, parse_dates=["timestamp"], index_col="timestamp")
            mask = (df.index.date >= start_date) & (df.index.date <= end_date)
            chunk = df[mask]
            if len(chunk) > 0:
                for d, group in chunk.groupby(chunk.index.date):
                    vix_lookup[d] = float(group["close"].iloc[-1])
        except Exception:
            continue
    return vix_lookup


def resample_to_5min(df_1min):
    """Resample 1-min OHLCV to 5-min bars."""
    df_5 = df_1min.resample("5min").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna(subset=["open"])
    # Only keep market hours (09:15 to 15:30)
    df_5 = df_5[(df_5.index.time >= dt.time(9, 15)) & (df_5.index.time <= dt.time(15, 30))]
    return df_5


def bars_to_dicts(df_5min, date_str):
    """Convert a DataFrame of 5-min bars to list of dicts (for compute_indicators)."""
    bars = []
    for ts, row in df_5min.iterrows():
        bars.append({
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row.get("volume", 0)),
            "date": date_str,
            "time": str(ts),
        })
    return bars


# ─────────────────────────────────────────────────────────────
# OPTION PREMIUM HELPER
# ─────────────────────────────────────────────────────────────

def get_strike_and_type(action, spot, vix, zero_hero=False):
    """Select strike and option type — ATM for normal, deep OTM for Z2H."""
    opt_type = "CE" if action == "BUY_CALL" else "PE"
    atm = round(spot / STRIKE_INTERVAL) * STRIKE_INTERVAL
    if zero_hero:
        strike = atm + 200 if action == "BUY_CALL" else atm - 200
    else:
        strike = atm  # ATM — matching live agent
    return strike, opt_type


def calc_premium(spot, strike, dte, vix, opt_type, slippage_sign=1):
    """Calculate option premium with slippage."""
    try:
        result = price_option(spot=spot, strike=strike, dte_days=max(0.1, dte),
                              vix=vix, option_type=opt_type)
        prem = result["premium"]
    except Exception:
        prem = 30.0
    # Apply slippage: +slippage on buy, -slippage on sell
    prem = prem * (1 + slippage_sign * SLIPPAGE_PCT) + slippage_sign * SPREAD_RS
    return max(0.05, prem)


# ─────────────────────────────────────────────────────────────
# DAY SIMULATION
# ─────────────────────────────────────────────────────────────

def _v17_regime_blocked(
    cfg: dict, ind: dict, vix: float, spot: float, conf: float = 1.0,
) -> bool:
    """V17 regime gate: return True if the current bar is in a regime
    where entries should be blocked.

    Indicator-driven replacement for the avoid_days calendar filter.
    Only active when cfg['use_v17_regime_gate'] is True.
    """
    if not cfg.get("use_v17_regime_gate"):
        return False
    if not ind:
        return False
    adx = float(ind.get("adx", 25.0) or 25.0)
    atr = float(ind.get("atr", 0.0) or 0.0)
    squeeze_on = bool(ind.get("squeeze_on", False))
    dc_width_pct = float(ind.get("donchian_width_pct", 0.0) or 0.0)

    # 1. Hard ADX floor: no trend at all = block
    if adx < cfg.get("v17_gate_adx_hard", 20):
        return True
    # 2. Chop: moderate ADX but squeeze still on = block
    if adx < cfg.get("v17_gate_adx_chop", 22) and squeeze_on:
        return True
    # 3. Dead vol: ATR/spot below floor
    atr_pct = atr / spot if spot > 0 else 0.0
    if atr_pct < cfg.get("v17_gate_atr_pct_min", 0.0015):
        return True
    # 4. VIX extremes
    if vix < cfg.get("v17_gate_vix_low", 13):
        return True
    if vix > cfg.get("v17_gate_vix_high", 30):
        return True
    # 5. Donchian width too narrow
    if dc_width_pct > 0 and dc_width_pct < cfg.get("v17_gate_donchian_min", 0.003):
        return True
    # 6. Minimum confidence floor on the signal itself
    if conf < cfg.get("v17_gate_min_conf", 0.55):
        return True
    return False


def _v17_monwed_gate_blocked(
    cfg: dict, ind: dict, vix: float, spot: float,
    bar_idx: int, day_of_week: int, action: str,
) -> bool:
    """Conditional Mon/Wed entry gate: only allow entries when indicators
    confirm a genuine directional move.

    Unlike the old avoid_days=[0,2] hard block, this lets Mon/Wed trades
    through IF the tape is showing a clear trend aligned with the trade.
    On Tue/Thu/Fri this function is a no-op — V15 rules apply unchanged.
    """
    # Only gate Mon (0) and Wed (2)
    if day_of_week not in (0, 2):
        return False
    if not cfg.get("use_v17_monwed_gate"):
        return False
    if not ind:
        return True  # no indicators = no entry on Mon/Wed

    # ── 1. Delay: give indicators time to form a view ──
    if bar_idx < cfg.get("v17_monwed_bar_min", 12):
        return True

    adx = float(ind.get("adx", 0) or 0)
    atr = float(ind.get("atr", 0) or 0)
    trend_regime = int(ind.get("trend_regime", 0) or 0)
    squeeze_on = bool(ind.get("squeeze_on", False))

    # ── 2. Strong ADX (trend is present) ──
    if adx < cfg.get("v17_monwed_adx_min", 22):
        return True

    # ── 3. Trend direction aligned with trade ──
    want_trend = -1 if action == "BUY_PUT" else +1
    if trend_regime != want_trend:
        return True

    # ── 4. Reject squeeze (market is compressed, no clean move) ──
    if squeeze_on:
        return True

    # ── 5. Real intraday volatility ──
    atr_pct = atr / spot if spot > 0 else 0.0
    if atr_pct < cfg.get("v17_monwed_atr_pct_min", 0.0015):
        return True

    # ── 6. VIX sanity bounds ──
    if vix < cfg.get("v17_monwed_vix_min", 12):
        return True
    if vix > cfg.get("v17_monwed_vix_max", 30):
        return True

    return False


def _v17_btst_favorable(
    cfg: dict, ind: dict, action: str,
    bar_idx: int, dte: float, vix: float, spot: float,
    day_high: float, day_low: float, day_open: float,
) -> bool:
    """Thin backtest wrapper — delegates to the shared scoring engine
    so backtest and live use the exact same BTST favorability logic.
    """
    return _shared_v17_btst_favorable(
        cfg=cfg,
        ind=ind,
        action=action,
        bar_idx=bar_idx,
        dte=dte,
        vix=vix,
        spot=spot,
        day_high=day_high,
        day_low=day_low,
        day_open=day_open,
    )


def _v17_pick_product(
    cfg: dict, ind: dict, action: str, conf: float,
    bar_idx: int, dte: float, vix: float,
    spot: float = 0.0, day_high: float = 0.0, day_low: float = 0.0,
    day_open: float = 0.0,
) -> str:
    """V17 dynamic product selector: return 'MIS' or 'NRML'.

    Delegates to _v17_btst_favorable() for the indicator-based decision.
    """
    if _v17_btst_favorable(
        cfg, ind, action, bar_idx, dte, vix, spot,
        day_high, day_low, day_open,
    ):
        return "NRML"
    return "MIS"


def simulate_day(
    bars_5min: list,
    date,
    vix: float,
    cfg: dict,
    prev_close: float,
    equity: float,
    warmup_bars: list = None,
    is_expiry: bool = False,
    consecutive_down_days: int = 0,
    btst_carry: list = None,
) -> tuple:
    """Simulate one trading day using shared V14 scoring engine on 5-min bars.

    Parameters
    ----------
    bars_5min : list[dict]
        Today's 5-min bars as dicts.
    warmup_bars : list[dict]
        Previous day(s) bars for indicator warm-up.
    btst_carry : list[dict], optional
        NRML trades carried over from the previous day. They are
        closed during the first 6 bars of this day (by 9:45 AM IST,
        matching the live BTST_EXIT_DEADLINE). The overnight gap is
        captured naturally by the first bar's spot.

    Returns
    -------
    (closed_trades, day_pnl, end_of_day_close, btst_to_carry)
        btst_to_carry is the list of NRML trades still open at EOD
        that need to be carried to the NEXT trading day.
    """
    date_str = str(date)

    # ── Day-of-week blocking (only active if cfg['avoid_days'] set;
    #    V15 sets [0,2], V17 sets [], so V17 falls through) ──
    avoid_days = cfg.get("avoid_days", [])
    if date.weekday() in avoid_days:
        eod_spot = bars_5min[-1]["close"] if bars_5min else 0
        # Still need to handle any incoming BTST carry even on blocked days.
        # Match the live BTST_EXIT_DEADLINE of 09:45 IST: close at bar 6
        # (9:45-9:50 5-min bar close), not bar 0. This matters because on
        # gap-open days the bar-0 price can be meaningfully different from
        # bar-6 after the first 30 min of price discovery.
        closed_carry = []
        if btst_carry:
            # Index into the day's 5-min bars: bar 6 = 09:45-09:50, close at 09:50
            exit_bar_idx = min(6, len(bars_5min) - 1) if bars_5min else 0
            exit_spot = bars_5min[exit_bar_idx]["close"] if bars_5min else 0
            for t in btst_carry:
                carry_dte = max(0.05, float(t.get("dte_at_entry", 1.0)) - 1.0)
                exit_prem = calc_premium(
                    exit_spot, t["strike"], carry_dte,
                    vix, t["opt_type"], slippage_sign=-1,
                )
                pnl = (exit_prem - t["entry_premium"]) * t["qty"] - BROKERAGE_RT
                t["exit_bar"] = exit_bar_idx
                t["exit_spot"] = exit_spot
                t["exit_premium"] = exit_prem
                t["exit_reason"] = "btst_blocked_day_exit"
                t["pnl"] = pnl
                t["bars_held"] = -1  # overnight hold
                closed_carry.append(t)
        return closed_carry, sum(t["pnl"] for t in closed_carry), eod_spot, []

    # Build bar history with warmup
    bar_history = list(warmup_bars or [])

    open_trades = []
    closed_trades = []
    trades_today = 0
    last_exit_bar = -10

    # ORB tracking
    orb_high = 0.0
    orb_low = 0.0
    gap_detected = False
    day_open = bars_5min[0]["close"] if bars_5min else 0

    # Lot sizing: SPAN margin model (matching backtest standard)
    span_per_lot = 40000
    if vix >= 20:
        span_per_lot = 50000
    elif vix >= 25:
        span_per_lot = 60000
    base_lots = max(1, int(equity * 0.70 / span_per_lot))

    # DTE: approximate as days to next Thursday (2024 expiry day)
    day_of_week = date.weekday()  # 0=Mon ... 6=Sun
    # For 2024 data: Thursday expiry. For 2026: Tuesday expiry.
    # Use Thursday for backtest period (Jul 2024 - Jan 2025)
    days_to_expiry = (3 - day_of_week) % 7  # Days to Thursday
    if days_to_expiry == 0:
        dte = 0.2  # Expiry day: a few hours left
    else:
        dte = days_to_expiry

    day_pnl = 0.0

    # ── Process BTST carry from previous day ──
    # Mirror the live BTST_EXIT_DEADLINE of 09:45 IST: exit at bar 6
    # unless a standard exit rule fires earlier in bars 0-5.
    btst_open = list(btst_carry or [])
    btst_exit_bar = 6  # 09:45 IST (6 bars * 5 min after 09:15 open)

    for bar_idx, bar in enumerate(bars_5min):
        bar_history.append(bar)
        if len(bar_history) > 500:
            bar_history = bar_history[-500:]

        spot = bar["close"]

        # ── ORB tracking ──
        if bar_idx == 0:
            orb_high = bar["high"]
            orb_low = bar["low"]
        elif bar_idx == 1:
            orb_high = max(orb_high, bar["high"])
            orb_low = min(orb_low, bar["low"])

        # ── BTST (NRML carry) exits for trades opened yesterday ──
        # Close on standard exit rule or force at bar_idx == btst_exit_bar
        for trade in list(btst_open):
            indicators_btst = compute_indicators(bar_history, date_str)
            # Standard exit rule check (trend break, trail stop, etc.)
            exit_reason = None
            if indicators_btst:
                exit_reason = evaluate_exit(
                    trade, bar_idx, spot, indicators_btst, cfg,
                    day_of_week=day_of_week,
                )
            if not exit_reason and bar_idx >= btst_exit_bar:
                exit_reason = "btst_deadline"
            if exit_reason:
                carry_dte = max(0.05, float(trade.get("dte_at_entry", 1.0)) - 1.0)
                exit_prem = calc_premium(
                    spot, trade["strike"], carry_dte,
                    vix, trade["opt_type"], slippage_sign=-1,
                )
                pnl = (exit_prem - trade["entry_premium"]) * trade["qty"] - BROKERAGE_RT
                trade["exit_bar"] = bar_idx
                trade["exit_spot"] = spot
                trade["exit_premium"] = exit_prem
                trade["exit_reason"] = exit_reason
                trade["pnl"] = pnl
                trade["bars_held"] = -1  # overnight hold; bars_held is meaningless
                closed_trades.append(trade)
                btst_open.remove(trade)
                day_pnl += pnl

        # ── EXITS ──
        for trade in list(open_trades):
            # Update best favorable
            if trade["action"] == "BUY_CALL" and spot > trade["best_fav"]:
                trade["best_fav"] = spot
            elif trade["action"] == "BUY_PUT" and spot < trade["best_fav"]:
                trade["best_fav"] = spot

            indicators = compute_indicators(bar_history, date_str)
            # ── Mon/Wed-specific exit overrides (experiment) ──
            # On Mon/Wed, optionally apply tighter max_hold / trail_pct /
            # chandelier_mult to escape the reversal pattern.
            exit_cfg = cfg
            if day_of_week in (0, 2) and cfg.get("use_monwed_tight_exits"):
                exit_cfg = dict(cfg)
                if cfg.get("monwed_max_hold_put") is not None:
                    exit_cfg["max_hold_put"] = cfg["monwed_max_hold_put"]
                if cfg.get("monwed_max_hold_call") is not None:
                    exit_cfg["max_hold_call"] = cfg["monwed_max_hold_call"]
                if cfg.get("monwed_trail_pct_put") is not None:
                    exit_cfg["trail_pct_put"] = cfg["monwed_trail_pct_put"]
                if cfg.get("monwed_trail_pct_call") is not None:
                    exit_cfg["trail_pct_call"] = cfg["monwed_trail_pct_call"]
                if cfg.get("monwed_chandelier_mult") is not None:
                    exit_cfg["chandelier_atr_mult"] = cfg["monwed_chandelier_mult"]
                if cfg.get("monwed_min_hold_trail_put") is not None:
                    exit_cfg["min_hold_trail_put"] = cfg["monwed_min_hold_trail_put"]
                if cfg.get("monwed_min_hold_trail_call") is not None:
                    exit_cfg["min_hold_trail_call"] = cfg["monwed_min_hold_trail_call"]
            exit_reason = evaluate_exit(
                trade, bar_idx, spot, indicators or {}, exit_cfg,
                day_of_week=day_of_week,
            )
            # NRML trades are allowed to survive the shared-engine
            # eod_close bar — they'll be carried overnight by the
            # btst_to_carry block at end of day. Any OTHER exit reason
            # (time_exit, trail_stop, theta, momentum) still closes them.
            if exit_reason == "eod_close" and trade.get("product") == "NRML":
                exit_reason = None
            if exit_reason:
                # Compute exit P&L via Black-Scholes
                exit_prem = calc_premium(spot, trade["strike"], max(0.05, dte - bar_idx * 5 / (6.25 * 60)),
                                         vix, trade["opt_type"], slippage_sign=-1)
                pnl = (exit_prem - trade["entry_premium"]) * trade["qty"] - BROKERAGE_RT
                trade["exit_bar"] = bar_idx
                trade["exit_spot"] = spot
                trade["exit_premium"] = exit_prem
                trade["exit_reason"] = exit_reason
                trade["pnl"] = pnl
                trade["bars_held"] = bar_idx - trade["entry_bar"]
                closed_trades.append(trade)
                open_trades.remove(trade)
                day_pnl += pnl
                last_exit_bar = bar_idx

        # ── ENTRIES ──
        indicators = compute_indicators(bar_history, date_str)
        if indicators is None:
            continue

        # Guards
        if len(open_trades) >= cfg["max_concurrent"]:
            continue
        if trades_today >= cfg["max_trades_per_day"]:
            continue

        # ── Entry window gating (R4_2Windows: bars 3-15 + 54-69) ──
        # Composite entries at bars 0-2 (gap, ORB) bypass this gate
        entry_windows = cfg.get("entry_windows_bars")

        # ── Composite entries (bars 0-2) ──
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
            # Filter out zero-hero if disabled
            if cfg.get("disable_zero_hero", False):
                composites = [c for c in composites if not c[3]]
            if composites:
                composites.sort(key=lambda x: x[2], reverse=True)
                action, entry_type, conf, is_zero_hero = composites[0]
        else:
            # Entry window check (bar >= 3 only; bars 0-2 handled above)
            if entry_windows:
                in_window = any(s <= bar_idx <= e for s, e in entry_windows)
                if not in_window:
                    continue

            # Cooldown check
            if bar_idx - last_exit_bar < cfg["cooldown_bars"]:
                continue

            # V8 scoring
            action, conf = score_entry(
                indicators, vix, cfg,
                bar_idx=bar_idx,
                consecutive_down_days=consecutive_down_days,
            )

            if action is None:
                # Try composite entries
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

        # VIX bounds
        if vix < cfg["vix_floor"] or vix > cfg["vix_ceil"]:
            continue

        # Confluence check
        if not passes_confluence(
            action, conf, indicators, bar_idx, cfg,
            current_spot=spot,
            prev_close=prev_close,
            day_open=day_open,
        ):
            continue

        # ── V17 regime gate (indicator-based replacement for avoid_days) ──
        if _v17_regime_blocked(cfg, indicators, vix, spot, conf=conf):
            continue

        # ── V17 Mon/Wed conditional entry gate ──
        # Only active when avoid_days does NOT include 0/2 and the gate is on.
        # Tue/Thu/Fri pass through unchanged (V15 rules).
        if _v17_monwed_gate_blocked(
            cfg, indicators, vix, spot, bar_idx, day_of_week, action,
        ):
            continue

        # Check no duplicate direction (consider BTST carries too — don't
        # double-dip on the same direction we're still holding overnight)
        if any(t["action"] == action for t in open_trades):
            continue
        if any(t["action"] == action for t in btst_open):
            continue

        # ── Lot sizing (R5: pass ATR for ATR-normalized sizing) ──
        cfg_with_atr = cfg
        if cfg.get("use_atr_sizing"):
            cfg_with_atr = cfg.copy()
            cfg_with_atr["_current_atr"] = indicators.get("atr", 0)
        lots = compute_lots(action, conf, vix, indicators.get("rsi", 50),
                            is_expiry, base_lots, cfg_with_atr)
        if is_zero_hero:
            lots = min(cfg.get("zero_hero_max_lots", 3), max(1, lots))

        # ── T20: VIX lot scaling + ORB boost + lot cap ──
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

        # ── Strike selection ──
        strike, opt_type = get_strike_and_type(action, spot, vix, zero_hero=is_zero_hero)
        qty = lots * LOT_SIZE

        # ── Entry premium (Black-Scholes) ──
        entry_prem = calc_premium(spot, strike, dte, vix, opt_type, slippage_sign=1)

        # ── V17 dynamic product selection (indicator-based BTST favorability) ──
        # Compute day-level stats up to and including this bar
        day_bars_so_far = bars_5min[: bar_idx + 1]
        day_high_sf = max(b["high"] for b in day_bars_so_far)
        day_low_sf = min(b["low"] for b in day_bars_so_far)
        day_open_sf = day_bars_so_far[0]["open"] if day_bars_so_far else spot
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
        }
        open_trades.append(trade)
        trades_today += 1

    # ── EOD handling: MIS force-close, NRML carried to next day ──
    eod_spot = bars_5min[-1]["close"] if bars_5min else 0
    btst_to_carry: list = []
    for trade in open_trades:
        if trade.get("product") == "NRML":
            # Carry forward. P&L will be realised tomorrow by bar 6.
            btst_to_carry.append(trade)
            continue
        exit_prem = calc_premium(eod_spot, trade["strike"],
                                 max(0.05, dte * 0.1), vix, trade["opt_type"],
                                 slippage_sign=-1)
        pnl = (exit_prem - trade["entry_premium"]) * trade["qty"] - BROKERAGE_RT
        trade["exit_bar"] = len(bars_5min) - 1
        trade["exit_spot"] = eod_spot
        trade["exit_premium"] = exit_prem
        trade["exit_reason"] = "eod_close"
        trade["pnl"] = pnl
        trade["bars_held"] = trade["exit_bar"] - trade["entry_bar"]
        closed_trades.append(trade)
        day_pnl += pnl

    # Any BTST carries still open at EOD (shouldn't happen, fail-safe)
    for trade in list(btst_open):
        exit_prem = calc_premium(eod_spot, trade["strike"],
                                 max(0.05, float(trade.get("dte_at_entry", 1.0)) - 1.0),
                                 vix, trade["opt_type"], slippage_sign=-1)
        pnl = (exit_prem - trade["entry_premium"]) * trade["qty"] - BROKERAGE_RT
        trade["exit_bar"] = len(bars_5min) - 1
        trade["exit_spot"] = eod_spot
        trade["exit_premium"] = exit_prem
        trade["exit_reason"] = "btst_eod_safety"
        trade["pnl"] = pnl
        trade["bars_held"] = -1
        closed_trades.append(trade)
        day_pnl += pnl

    return closed_trades, day_pnl, eod_spot, btst_to_carry


# ─────────────────────────────────────────────────────────────
# MAIN: 6-MONTH BACKTEST
# ─────────────────────────────────────────────────────────────

def load_period_data(start_date=None, end_date=None, months=6, quiet=False):
    """One-shot loader: returns (day_groups, vix_lookup, all_dates, warmup_bars).

    Used by the sweep harness so multiple variants share the same loaded data.
    """
    if start_date is None:
        start_date = dt.date(2024, 7, 1)
    if end_date is None:
        end_date = start_date + dt.timedelta(days=months * 31)

    if not quiet:
        print("Loading NIFTY 1-min data...", flush=True)
    warmup_start = start_date - dt.timedelta(days=10)
    nifty_1min = load_nifty_data(warmup_start, end_date)
    if not quiet:
        print(f"  Loaded {len(nifty_1min):,} bars")
    vix_lookup = load_vix_data(warmup_start, end_date)
    nifty_5min = resample_to_5min(nifty_1min)
    if not quiet:
        print(f"  {len(nifty_5min):,} 5-min bars  /  VIX days: {len(vix_lookup)}")

    day_groups = {}
    for ts, row in nifty_5min.iterrows():
        d = ts.date()
        if d not in day_groups:
            day_groups[d] = []
        day_groups[d].append({
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row.get("volume", 0)),
            "date": str(d),
            "time": str(ts),
        })

    all_dates = sorted(d for d in day_groups.keys() if d >= start_date)
    warmup_dates = sorted(d for d in day_groups.keys() if d < start_date)
    warmup_bars = []
    for d in warmup_dates[-3:]:
        warmup_bars.extend(day_groups[d])
    return day_groups, vix_lookup, all_dates, warmup_bars


def run_backtest(start_date=None, end_date=None, months=6, cfg_override=None, quiet=False,
                 preloaded=None):
    """Run full backtest over the specified period.

    cfg_override : dict, optional
        Keys here override V14_CONFIG (V15) for this run only. Used by sweep harness.
    quiet : bool
        Suppress per-day output.
    preloaded : tuple, optional
        (day_groups, vix_lookup, all_dates, warmup_bars) from load_period_data().
        Skips redundant CSV loads in sweeps.
    """

    # Default: Jul 2024 - Jan 2025 (6 months)
    if start_date is None:
        start_date = dt.date(2024, 7, 1)
    if end_date is None:
        end_date = start_date + dt.timedelta(days=months * 31)

    cfg_local = dict(V14_CONFIG)
    if cfg_override:
        cfg_local.update(cfg_override)

    if not quiet:
        print(f"V14 Unified 5-Min Backtest")
        print(f"=" * 60)
        print(f"Period: {start_date} to {end_date}")
        print(f"Config: {cfg_local['name']}")
        print(f"Capital: Rs {CAPITAL:,.0f}")
        print(f"Lot size: {LOT_SIZE}")
        print()

    # Load data (or use preloaded for sweep harness)
    if preloaded is not None:
        day_groups, vix_lookup, all_dates, warmup_bars = preloaded
    else:
        if not quiet:
            print("Loading NIFTY 1-min data...", flush=True)
        warmup_start = start_date - dt.timedelta(days=10)
        nifty_1min = load_nifty_data(warmup_start, end_date)
        if not quiet:
            print(f"  Loaded {len(nifty_1min):,} bars ({nifty_1min.index[0].date()} to {nifty_1min.index[-1].date()})")
            print("Loading VIX data...", flush=True)
        vix_lookup = load_vix_data(warmup_start, end_date)
        if not quiet:
            print(f"  VIX data for {len(vix_lookup)} days")
            print("Resampling to 5-min bars...", flush=True)
        nifty_5min = resample_to_5min(nifty_1min)
        if not quiet:
            print(f"  {len(nifty_5min):,} five-minute bars")

        day_groups = {}
        for ts, row in nifty_5min.iterrows():
            d = ts.date()
            if d not in day_groups:
                day_groups[d] = []
            day_groups[d].append({
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row.get("volume", 0)),
                "date": str(d),
                "time": str(ts),
            })

        all_dates = sorted(d for d in day_groups.keys() if d >= start_date)
        warmup_dates = sorted(d for d in day_groups.keys() if d < start_date)
        if not quiet:
            print(f"  Trading days: {len(all_dates)}, Warmup days: {len(warmup_dates)}")
            print()

        warmup_bars = []
        for d in warmup_dates[-3:]:
            warmup_bars.extend(day_groups[d])

    # Run simulation
    cfg = cfg_local
    equity = CAPITAL
    all_trades = []
    monthly_pnl = defaultdict(float)
    prev_close = 0.0
    consecutive_down_days = 0
    btst_carry: list = []  # NRML trades carried from previous day

    if not quiet:
        print(f"{'Date':<12} {'Trades':>6} {'Day PnL':>10} {'Equity':>12} {'VIX':>5}")
        print("-" * 52)

    for i, date in enumerate(all_dates):
        bars = day_groups[date]
        if len(bars) < 5:
            continue

        vix = vix_lookup.get(date, 14.0)
        is_expiry = (date.weekday() == 3)  # Thursday for 2024

        trades, day_pnl, eod_close, btst_carry = simulate_day(
            bars, date, vix, cfg,
            prev_close=prev_close,
            equity=equity,
            warmup_bars=warmup_bars,
            is_expiry=is_expiry,
            consecutive_down_days=consecutive_down_days,
            btst_carry=btst_carry,
        )

        equity += day_pnl
        all_trades.extend(trades)
        month_key = f"{date.year}-{date.month:02d}"
        monthly_pnl[month_key] += day_pnl

        if trades and not quiet:
            print(f"{date}  {len(trades):>6}  {day_pnl:>+10,.0f}  {equity:>12,.0f}  {vix:>5.1f}")

        # Track consecutive down days
        if len(bars) >= 2:
            day_open_price = bars[0]["open"]
            day_close_price = bars[-1]["close"]
            if day_close_price < day_open_price:
                consecutive_down_days += 1
            else:
                consecutive_down_days = 0

        # Update warmup bars (keep last 3 days worth)
        warmup_bars = warmup_bars[-(75 * 2):] + bars  # ~2 days of 5-min bars
        prev_close = eod_close

    # ── Final safety-close: any NRML carry still open at end of backtest ──
    if btst_carry:
        final_spot = prev_close
        for trade in btst_carry:
            exit_prem = calc_premium(
                final_spot, trade["strike"],
                max(0.05, float(trade.get("dte_at_entry", 1.0)) - 1.0),
                trade.get("vix", 14.0), trade["opt_type"], slippage_sign=-1,
            )
            pnl = (exit_prem - trade["entry_premium"]) * trade["qty"] - BROKERAGE_RT
            trade["exit_bar"] = -1
            trade["exit_spot"] = final_spot
            trade["exit_premium"] = exit_prem
            trade["exit_reason"] = "backtest_end_close"
            trade["pnl"] = pnl
            trade["bars_held"] = -1
            all_trades.append(trade)
            equity += pnl
        btst_carry = []

    # ─────────────────────────────────────────────────────────
    # RESULTS
    # ─────────────────────────────────────────────────────────
    if not quiet:
        print()
        print("=" * 60)
        print("BACKTEST RESULTS")
        print("=" * 60)

    total_pnl = sum(t.get("pnl", 0) for t in all_trades)
    wins = [t for t in all_trades if t.get("pnl", 0) > 0]
    losses = [t for t in all_trades if t.get("pnl", 0) <= 0]
    win_rate = len(wins) / len(all_trades) * 100 if all_trades else 0
    avg_win = np.mean([t["pnl"] for t in wins]) if wins else 0
    avg_loss = np.mean([t["pnl"] for t in losses]) if losses else 0
    profit_factor = abs(sum(t["pnl"] for t in wins)) / (abs(sum(t["pnl"] for t in losses)) + 1) if losses else 999
    ret_x = equity / CAPITAL

    if not quiet:
        print(f"Total trades:     {len(all_trades)}")
        print(f"Wins:             {len(wins)} ({win_rate:.1f}%)")
        print(f"Losses:           {len(losses)}")
        print(f"Avg win:          Rs {avg_win:+,.0f}")
        print(f"Avg loss:         Rs {avg_loss:+,.0f}")
        print(f"Profit factor:    {profit_factor:.2f}")
        print(f"Total P&L:        Rs {total_pnl:+,.0f}")
        print(f"Final equity:     Rs {equity:,.0f}")
        print(f"Return:           {ret_x:.1f}x ({(ret_x - 1) * 100:.0f}%)")
        print()

    if not quiet:
        # Monthly breakdown
        print(f"{'Month':<10} {'P&L':>12}")
        print("-" * 24)
        for month in sorted(monthly_pnl):
            print(f"{month:<10} {monthly_pnl[month]:>+12,.0f}")
        print()

        # By exit reason
        reason_pnl = defaultdict(lambda: {"count": 0, "pnl": 0, "wins": 0})
        for t in all_trades:
            r = t.get("exit_reason", "unknown")
            reason_pnl[r]["count"] += 1
            reason_pnl[r]["pnl"] += t.get("pnl", 0)
            if t.get("pnl", 0) > 0:
                reason_pnl[r]["wins"] += 1

        print(f"{'Exit Reason':<22} {'Count':>6} {'P&L':>12} {'WR':>6}")
        print("-" * 50)
        for r in sorted(reason_pnl, key=lambda x: reason_pnl[x]["pnl"], reverse=True):
            d = reason_pnl[r]
            wr = d["wins"] / d["count"] * 100 if d["count"] else 0
            print(f"{r:<22} {d['count']:>6} {d['pnl']:>+12,.0f} {wr:>5.1f}%")
        print()

        # By entry type
        type_pnl = defaultdict(lambda: {"count": 0, "pnl": 0, "wins": 0})
        for t in all_trades:
            et = t.get("entry_type", "v8_indicator")
            type_pnl[et]["count"] += 1
            type_pnl[et]["pnl"] += t.get("pnl", 0)
            if t.get("pnl", 0) > 0:
                type_pnl[et]["wins"] += 1

        print(f"{'Entry Type':<22} {'Count':>6} {'P&L':>12} {'WR':>6}")
        print("-" * 50)
        for et in sorted(type_pnl, key=lambda x: type_pnl[x]["pnl"], reverse=True):
            d = type_pnl[et]
            wr = d["wins"] / d["count"] * 100 if d["count"] else 0
            print(f"{et:<22} {d['count']:>6} {d['pnl']:>+12,.0f} {wr:>5.1f}%")

        # By product (MIS vs NRML) — only meaningful when dynamic product is on
        prod_pnl = defaultdict(lambda: {"count": 0, "pnl": 0, "wins": 0})
        for t in all_trades:
            p = t.get("product", "MIS")
            prod_pnl[p]["count"] += 1
            prod_pnl[p]["pnl"] += t.get("pnl", 0)
            if t.get("pnl", 0) > 0:
                prod_pnl[p]["wins"] += 1
        if len(prod_pnl) > 1 or "NRML" in prod_pnl:
            print()
            print(f"{'Product':<22} {'Count':>6} {'P&L':>12} {'WR':>6}")
            print("-" * 50)
            for p in sorted(prod_pnl, key=lambda x: prod_pnl[x]["pnl"], reverse=True):
                d = prod_pnl[p]
                wr = d["wins"] / d["count"] * 100 if d["count"] else 0
                print(f"{p:<22} {d['count']:>6} {d['pnl']:>+12,.0f} {wr:>5.1f}%")

        # Save trades
        trades_df = pd.DataFrame(all_trades)
        out_path = DATA_DIR / "v14_unified_5min_trades.csv"
        trades_df.to_csv(out_path, index=False)
        print(f"\nTrades saved to {out_path}")

    return all_trades, equity


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="V14 Unified 5-Min Backtest")
    parser.add_argument("--months", type=int, default=6, help="Number of months to backtest")
    parser.add_argument("--start", type=str, default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument("--config", type=str, default="V15",
                        choices=["V15", "V17", "V17_GATE_ONLY", "V17_PROD_ONLY",
                                 "V17_MONWED", "V17_MONWED_ONLY",
                                 "V17_MW_HOLD", "V17_MW_TRAIL", "V17_MW_BOTH"],
                        help="Which config variant to run")
    args = parser.parse_args()

    start = dt.date.fromisoformat(args.start) if args.start else None
    end = dt.date.fromisoformat(args.end) if args.end else None

    # Config selection
    cfg_override = None
    if args.config == "V15":
        cfg_override = None  # V14_CONFIG alias is already V15
    elif args.config == "V17":
        from scoring.config import V17_CONFIG
        cfg_override = dict(V17_CONFIG)
    elif args.config == "V17_GATE_ONLY":
        # Regime gate but MIS-only (no NRML) — isolates gate contribution
        from scoring.config import V17_CONFIG
        cfg_override = dict(V17_CONFIG)
        cfg_override["use_v17_dynamic_product"] = False
    elif args.config == "V17_PROD_ONLY":
        # Dynamic product but keep V15 avoid_days calendar filter
        from scoring.config import V17_CONFIG
        cfg_override = dict(V17_CONFIG)
        cfg_override["avoid_days"] = [0, 2]
        cfg_override["use_v17_regime_gate"] = False
        cfg_override["use_v17_monwed_gate"] = False
    elif args.config == "V17_MONWED":
        # Full: Mon/Wed conditional gate + loose BTST
        from scoring.config import V17_CONFIG
        cfg_override = dict(V17_CONFIG)
        cfg_override["avoid_days"] = []
        cfg_override["use_v17_regime_gate"] = False
    elif args.config == "V17_MONWED_ONLY":
        # Mon/Wed gate only, no BTST carry (isolates gate contribution)
        from scoring.config import V17_CONFIG
        cfg_override = dict(V17_CONFIG)
        cfg_override["avoid_days"] = []
        cfg_override["use_v17_regime_gate"] = False
        cfg_override["use_v17_dynamic_product"] = False
    elif args.config == "V17_MW_HOLD":
        # Experiment 1: Mon/Wed 2hr max hold + V15 entries + loose BTST
        # Mon/Wed entries use V15 rules (no indicator gate) but are
        # force-exited after 24 bars (2 hrs) to escape reversals.
        from scoring.config import V17_CONFIG
        cfg_override = dict(V17_CONFIG)
        cfg_override["avoid_days"] = []
        cfg_override["use_v17_regime_gate"] = False
        cfg_override["use_v17_monwed_gate"] = False  # let all Mon/Wed entries through
        cfg_override["use_monwed_tight_exits"] = True
        # only tighten hold, keep trails as V15 defaults
        cfg_override["monwed_trail_pct_put"] = None
        cfg_override["monwed_trail_pct_call"] = None
        cfg_override["monwed_chandelier_mult"] = None
        cfg_override["monwed_min_hold_trail_put"] = None
        cfg_override["monwed_min_hold_trail_call"] = None
    elif args.config == "V17_MW_TRAIL":
        # Experiment 2: Mon/Wed tight trails + V15 entries + loose BTST
        from scoring.config import V17_CONFIG
        cfg_override = dict(V17_CONFIG)
        cfg_override["avoid_days"] = []
        cfg_override["use_v17_regime_gate"] = False
        cfg_override["use_v17_monwed_gate"] = False
        cfg_override["use_monwed_tight_exits"] = True
        # only tighten trails, keep max_hold as V15 defaults
        cfg_override["monwed_max_hold_put"] = None
        cfg_override["monwed_max_hold_call"] = None
    elif args.config == "V17_MW_BOTH":
        # Experiment 1+2: Mon/Wed 2hr hold AND tight trails combined
        from scoring.config import V17_CONFIG
        cfg_override = dict(V17_CONFIG)
        cfg_override["avoid_days"] = []
        cfg_override["use_v17_regime_gate"] = False
        cfg_override["use_v17_monwed_gate"] = False
        cfg_override["use_monwed_tight_exits"] = True

    run_backtest(start_date=start, end_date=end, months=args.months,
                 cfg_override=cfg_override)
