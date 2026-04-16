"""
V14 ORIGINAL vs ENH-LEAN vs ENH-TUNED vs ENH-TUNED2 -- 6-Month Side-by-Side Comparison
========================================================================================
Compares four variants:
  1. V14_Original -- baseline V9_HYBRID_CONFIG (run_model)
  2. Enh-Lean     -- entry filters + original exit logic (previous best at 8.9x)
  3. Enh-Tuned    -- ALL 21 scoring signals + Lean's fixed % trails + theta exit
                     (no ATR trail, no throttle, no drawdown scaling, no StdDev)
  4. Enh-Tuned2   -- Same as Enh-Tuned but with DAILY-scale ATR trails
                     (atr_trail_mult=30 so 1-min ATR * 30 ~ 1.5x daily ATR)

Runs through the same simulate_day_enhanced engine for fair comparison.
Uses 6 months of cached historical data (Jul 2024 - Jan 2025).
"""

import sys
import copy
import datetime as dt
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backtesting.oos_june2024_test import (
    V9_HYBRID_CONFIG, CAPITAL,
    simulate_day, run_model,
)
from backtesting.multi_month_oos_test import download_range, run_month
from backtesting.daywise_analysis import add_all_indicators

DATA_DIR = project_root / "data" / "historical"


# ==================================================================
# V14 ENHANCED CONFIG — all research improvements integrated
# ==================================================================

def make_v14_enhanced_config():
    """Build V14 Enhanced config with all research improvements."""
    cfg = copy.deepcopy(V9_HYBRID_CONFIG)
    cfg["name"] = "V14_Enhanced"

    # ── RSI gates relaxed to match live V14 (wider entry window, kill zone handles bad zone) ──
    cfg["rsi_call_min"] = 50        # Lowered from 60 (kill zone at 60 handles the bad zone)
    cfg["rsi_put_max"] = 50         # Relaxed from 40

    # ── RSI CALL kill zone (NEW: 23.3% WR, -722K in 198-trade analysis) ──
    cfg["rsi_call_kill_ceiling"] = 60   # Block CALL when RSI > 60

    # ── Trail widening (PUT trail was losing 250K at 1.5%) ──
    cfg["trail_pct_put"] = 0.020    # Widened from 0.015
    cfg["trail_pct_call"] = 0.012   # Widened from 0.008
    cfg["min_hold_trail_put"] = 150  # 150 min (was 120)
    cfg["min_hold_trail_call"] = 90  # 90 min (was 60)

    # ── Hard stop loss — DISABLED (backtest showed 0% WR, -Rs912K on 71 trades) ──
    # Options premium naturally fluctuates 30%+ before recovering. Hard SL exits too early.
    cfg["hard_sl_pct"] = 0.0  # Disabled

    # ── Consecutive loss throttle ──
    cfg["max_consecutive_losses"] = 2
    cfg["loss_cooldown_min"] = 60   # 60 min pause after 2 consecutive losses

    # ── Gap reversal filter ──
    cfg["gap_reversal_filter"] = True
    cfg["gap_threshold_pct"] = 0.004  # 0.4% gap

    # ── Drawdown lot scaling — lighter touch (heavy scaling shrank lot sizes too much) ──
    cfg["drawdown_lot_scale"] = True
    cfg["drawdown_50pct_mult"] = 0.7   # Was 0.5 — too aggressive
    cfg["drawdown_75pct_mult"] = 0.3   # Was 0.0 (stopped trading) — keep small positions instead

    # ── Standard deviation entry filter ──
    cfg["use_stddev_filter"] = True
    cfg["stddev_entry_threshold"] = 1.0

    # ── Regime detection (enhanced — controls trail/hold/reversion) ──
    cfg["use_regime_detection"] = True
    cfg["use_hurst_regime"] = True  # Use Hurst-based regime instead of simple SMA regime

    # ── ADX scoring (ADX < 18 = 0.6x, 25-35 = 0.8x, > 35 = directional +1.0) ──
    cfg["use_adx_scoring"] = True
    cfg["adx_weak_threshold"] = 18
    cfg["adx_weak_mult"] = 0.6
    cfg["adx_choppy_min"] = 25
    cfg["adx_choppy_max"] = 35
    cfg["adx_choppy_mult"] = 0.8
    cfg["adx_strong_threshold"] = 35

    # ── No day filter (VWAP+RSI+Squeeze filters handle day quality) ──
    cfg["avoid_days"] = []

    # ── Expiry on Tuesday (SEBI Nov 2025 change) ──
    cfg["expiry_day"] = "Tuesday"    # Was Thursday in 2024 data, but keep for forward compat

    return cfg


def make_v14_enh_full21_config():
    """Build Enh-Full21 config — ALL 21 new features integrated.

    On top of the base enhanced config, adds:
      - CCI, Williams %R, RSI divergence scoring (computed in simulate_day)
      - ATR-based trailing stops with ADX-adaptive multipliers
      - OI S/R gating, max pain reversion, PCR extreme (1.6), VIX contrarian
      - VWAP mean reversion (>1.5% deviation, ADX<25)
      - Closing hour bias (+0.3 call score after 2:30 PM)
      - Consecutive down day pattern (3+ days -> +0.5 call)
      - Theta exit on Monday 3 PM
      - NO consecutive loss throttle (disabled)
      - NO drawdown lot scaling (disabled)
      - StdDev filter OFF
      - ATR trails with ADX-adaptive multipliers
    """
    cfg = copy.deepcopy(V9_HYBRID_CONFIG)
    cfg["name"] = "Enh-Full21"

    # ── RSI gates (same relaxation as Enhanced) ──
    cfg["rsi_call_min"] = 50
    cfg["rsi_put_max"] = 50
    cfg["rsi_call_kill_ceiling"] = 60

    # ── Trail widening (PUT trail was losing 250K at 1.5%) ──
    cfg["trail_pct_put"] = 0.020
    cfg["trail_pct_call"] = 0.012
    cfg["min_hold_trail_put"] = 150
    cfg["min_hold_trail_call"] = 90

    # ── Hard stop loss DISABLED ──
    cfg["hard_sl_pct"] = 0.0

    # ── Consecutive loss throttle DISABLED ──
    cfg["max_consecutive_losses"] = 99
    cfg["loss_cooldown_min"] = 0

    # ── Gap reversal filter ──
    cfg["gap_reversal_filter"] = True
    cfg["gap_threshold_pct"] = 0.004

    # ── Drawdown lot scaling DISABLED ──
    cfg["drawdown_lot_scale"] = False

    # ── StdDev filter OFF ──
    cfg["use_stddev_filter"] = False

    # ── Regime detection ──
    cfg["use_regime_detection"] = True
    cfg["use_hurst_regime"] = True
    cfg["direction_aware_regime"] = True

    # ── ADX scoring ──
    cfg["use_adx_scoring"] = True
    cfg["adx_weak_threshold"] = 18
    cfg["adx_weak_mult"] = 0.6
    cfg["adx_choppy_min"] = 25
    cfg["adx_choppy_max"] = 35
    cfg["adx_choppy_mult"] = 0.8
    cfg["adx_strong_threshold"] = 35

    # ── No day filter ──
    cfg["avoid_days"] = []

    # ── ATR-based trailing stops (replaces fixed % trails) ──
    cfg["use_atr_trail"] = True
    cfg["atr_trail_mult"] = 1.5
    cfg["atr_trail_adx_adaptive"] = True
    cfg["atr_trail_adx_low"] = 1.0     # ADX < 20: tight (range-bound)
    cfg["atr_trail_adx_mid"] = 1.5     # ADX 20-35: standard
    cfg["atr_trail_adx_high"] = 2.0    # ADX > 35: wide (strong trend)

    # ── PCR extreme (contrarian at 1.6) ──
    cfg["use_pcr_filter"] = True
    cfg["pcr_bearish_min"] = 1.6
    cfg["pcr_bullish_max"] = 0.7

    # ── VIX contrarian scoring ──
    cfg["use_vix_contrarian"] = True

    # ── VWAP mean reversion (>1.5% deviation, ADX<25) ──
    cfg["use_vwap_mean_reversion"] = True
    cfg["vwap_reversion_pct"] = 0.015
    cfg["vwap_reversion_adx_max"] = 25

    # ── Closing hour bias (+0.3 call score after 2:30 PM / minute 315) ──
    cfg["use_closing_hour_bias"] = True
    cfg["closing_hour_min_idx"] = 315

    # ── Consecutive down day pattern (3+ days -> +0.5 call) ──
    cfg["use_consec_down_day"] = True
    cfg["consec_down_day_min"] = 3
    cfg["consec_down_day_call_boost"] = 0.5

    # ── Theta exit on Monday 3 PM ──
    cfg["theta_exit_enabled"] = True
    cfg["theta_exit_minute"] = 345    # 345 min from 9:15 = 3:00 PM

    # ── CCI scoring (computed in simulate_day) ──
    cfg["use_cci_scoring"] = True
    cfg["cci_threshold"] = 100

    # ── Williams %R scoring (computed in simulate_day) ──
    cfg["use_williams_scoring"] = True
    cfg["williams_oversold"] = -80
    cfg["williams_overbought"] = -20

    # ── OI S/R gating ──
    cfg["use_oi_levels"] = True
    cfg["oi_proximity_pct"] = 0.003

    # ── Expiry day ──
    cfg["expiry_day"] = "Tuesday"

    return cfg


def make_v14_enh_tuned_config():
    """Build Enh-Tuned config -- ALL 21 scoring signals + Lean's fixed % trails.

    Best of everything: all new scoring signals for better entries, but
    Lean's proven exit logic (fixed % trails, original min_hold values).
    No ATR trail, no throttle, no drawdown scaling, no StdDev filter.
    Essentially: Lean + all 21 scoring signals + theta exit.
    """
    cfg = make_v14_enh_full21_config()
    cfg["name"] = "Enh-Tuned"

    # ── DISABLE ATR trail -- use fixed % trails instead ──
    cfg["use_atr_trail"] = False

    # ── Fixed % trails (Lean values -- proven to work) ──
    cfg["trail_pct_put"] = 0.015       # Lean value
    cfg["trail_pct_call"] = 0.008      # Lean value
    cfg["min_hold_trail_put"] = 120    # Lean/Original value
    cfg["min_hold_trail_call"] = 60    # Lean/Original value

    # ── DISABLE StdDev filter ──
    cfg["use_stddev_filter"] = False

    # ── DISABLE consecutive loss throttle ──
    cfg["max_consecutive_losses"] = 99
    cfg["loss_cooldown_min"] = 0

    # ── DISABLE drawdown lot scaling ──
    cfg["drawdown_lot_scale"] = False

    # ── ALL scoring flags stay ON (inherited from Enh-Full21):
    #    use_cci_scoring, use_williams_scoring, use_vix_contrarian,
    #    use_vwap_mean_reversion, use_closing_hour_bias, use_consec_down_day,
    #    use_oi_levels, use_pcr_filter, theta_exit_enabled, etc. ──

    return cfg


def make_v14_enh_tuned2_config():
    """Build Enh-Tuned2 config -- same as Enh-Tuned but with DAILY-scale ATR trails.

    The research document's 194pt ATR was DAILY ATR, not 1-minute.
    On 1-min bars, ATR(14) ~ 5-15 points, so raw 1.5x ATR = 7-22 points (noise).
    Daily-equivalent ATR ~ bar_atr * sqrt(375) ~ bar_atr * 19.4.
    Using atr_trail_mult=30.0 gives ~1.5x daily ATR equivalent.
    ADX-adaptive: low=20, mid=30, high=40 (scaled proportionally).
    """
    cfg = make_v14_enh_tuned_config()
    cfg["name"] = "Enh-Tuned2"

    # ── ENABLE ATR trail with DAILY-scale multipliers ──
    cfg["use_atr_trail"] = True
    cfg["atr_trail_mult"] = 30.0           # ~1.5x daily ATR from 1-min bars
    cfg["atr_trail_adx_adaptive"] = True
    cfg["atr_trail_adx_low"] = 20.0        # ADX < 20: tighter (range-bound)
    cfg["atr_trail_adx_mid"] = 30.0        # ADX 20-35: standard (~1.5x daily)
    cfg["atr_trail_adx_high"] = 40.0       # ADX > 35: wide (~2x daily, let trends run)

    return cfg


# ==================================================================
# ENHANCED simulate_day — adds new features on top of base engine
# ==================================================================

def simulate_day_enhanced(cfg, day_bars_df, date, prev_day_ohlc, vix, daily_trend,
                          dte, is_expiry, daily_df, row_idx, close_prices,
                          above_sma50, above_sma20, rsi, prev_change, vix_spike,
                          sma20, sma50, ema9, ema21, weekly_sma, gap_pct,
                          equity=CAPITAL, recent_wr=0.5, recent_trades=0,
                          regime_info=None, consecutive_down_days=0):
    """Enhanced simulation with V14 research improvements.

    Wraps the base simulate_day but adds:
    - RSI CALL kill zone (block CALL when RSI > 60)
    - Hard stop loss (exit if premium drops 30%)
    - StdDev entry filter (>1 sigma from daily mean)
    - ADX-gated scoring adjustments
    - Gap reversal filter
    - Consecutive loss throttle
    - Regime-adaptive trail/hold adjustments
    - CCI scoring (14-period, computed from bar data)
    - Williams %R scoring (14-period, computed from bar data)
    - VWAP mean reversion (>1.5% from VWAP, ADX<25)
    - VIX contrarian (VIX>25 -> call+0.5, VIX>35 -> call+1.0)
    - Closing hour bias (+0.3 call score after 2:30 PM)
    - Consecutive down day pattern (3+ days -> +0.5 call)
    - ATR-based trailing stops with ADX-adaptive multipliers
    - Theta exit (Monday 3 PM, exit profitable positions)
    """
    from backtesting.paper_trading_real_data import (
        sr_multi_method, bs_premium, get_strike_and_type, LOT_SIZE,
    )
    from backtesting.v7_hybrid_comparison import compute_composite
    from backtesting.daywise_analysis import (
        compute_pivot_points, find_support_resistance,
    )
    from backtesting.oos_june2024_test import (
        get_dynamic_lots, detect_entries_v8, detect_entries_composite,
    )

    n_bars = len(day_bars_df)
    if n_bars < 60:
        return []

    use_v8 = cfg.get("use_v8_scoring", False)

    # Pivot points
    if prev_day_ohlc:
        pivots = compute_pivot_points(prev_day_ohlc["high"], prev_day_ohlc["low"], prev_day_ohlc["close"])
    else:
        spot = day_bars_df["close"].iloc[0]
        pivots = compute_pivot_points(spot * 1.005, spot * 0.995, spot)

    sr_levels = find_support_resistance(day_bars_df, prev_day_ohlc, pivots)

    entry_spot = day_bars_df["open"].iloc[0]
    prev_high = prev_day_ohlc["high"] if prev_day_ohlc else entry_spot * 1.01
    prev_low = prev_day_ohlc["low"] if prev_day_ohlc else entry_spot * 0.99
    support, resistance = sr_multi_method(
        entry_spot, prev_high, prev_low, sma20, sma50,
        close_history=close_prices, idx=row_idx)

    if not use_v8:
        scores = compute_composite(
            "V6", vix, above_sma50, above_sma20, rsi,
            date.strftime("%A"), prev_change, vix_spike,
            entry_spot, support, resistance,
            ema9=ema9, ema21=ema21, weekly_sma=weekly_sma)
        best_composite = max(scores, key=scores.get)
        total_score = sum(scores.values())
        composite_conf = scores[best_composite] / total_score if total_score > 0 else 0
    else:
        best_composite = "BUY_PUT"
        composite_conf = 0

    bias_val = "neutral"
    if above_sma50 and above_sma20:
        bias_val = "strong_bullish" if ema9 and ema21 and ema9 > ema21 else "bullish"
    elif not above_sma50 and not above_sma20:
        bias_val = "strong_bearish" if ema9 and ema21 and ema9 < ema21 else "bearish"

    minute_closes = day_bars_df["close"].values
    n_15min = min(25, n_bars // 15 + 1)
    path_15min = [minute_closes[min(i * 15, n_bars - 1)] for i in range(n_15min)]

    highs = day_bars_df["high"].values
    lows = day_bars_df["low"].values
    closes_arr = day_bars_df["close"].values

    # ATR (global day ATR for initial sizing + running ATR array for per-bar trail)
    day_atr = 100
    tr_vals = []
    for j in range(1, min(60, n_bars)):
        tr = max(highs[j] - lows[j],
                 abs(highs[j] - closes_arr[j-1]),
                 abs(lows[j] - closes_arr[j-1]))
        tr_vals.append(tr)
    if len(tr_vals) >= 14:
        day_atr = np.mean(tr_vals[-14:])
    elif tr_vals:
        day_atr = np.mean(tr_vals)

    # Running ATR array (14-period) for ATR-based trailing at each bar
    atr_arr = np.full(n_bars, day_atr)
    if n_bars > 14:
        all_tr = np.zeros(n_bars)
        all_tr[0] = highs[0] - lows[0]
        for j in range(1, n_bars):
            all_tr[j] = max(highs[j] - lows[j],
                            abs(highs[j] - closes_arr[j-1]),
                            abs(lows[j] - closes_arr[j-1]))
        for j in range(14, n_bars):
            atr_arr[j] = np.mean(all_tr[j-13:j+1])
        # Fill early bars with expanding window
        for j in range(1, min(14, n_bars)):
            atr_arr[j] = np.mean(all_tr[:j+1])

    # ADX array (use column if available, else default)
    has_adx_col = "adx" in day_bars_df.columns
    adx_arr = day_bars_df["adx"].values if has_adx_col else np.full(n_bars, 25.0)

    # VWAP (always computed — needed for VWAP mean reversion even when use_vwap_filter is off)
    use_vwap = cfg.get("use_vwap_filter", False)
    use_squeeze = cfg.get("use_squeeze_filter", False)
    use_rsi_gate = cfg.get("use_rsi_hard_gate", False)

    tp = (highs + lows + closes_arr) / 3.0
    if "volume" in day_bars_df.columns:
        vol = day_bars_df["volume"].values.astype(float)
        vol = np.where(vol <= 0, 1.0, vol)
        vwap_arr = np.cumsum(tp * vol) / np.cumsum(vol)
    else:
        vwap_arr = np.cumsum(tp) / np.arange(1, n_bars + 1, dtype=float)

    squeeze_arr = np.zeros(n_bars, dtype=bool)
    if use_squeeze:
        has_cols = all(c in day_bars_df.columns for c in ["bb_upper", "bb_lower", "ema21", "atr"])
        if has_cols:
            bb_up = day_bars_df["bb_upper"].values
            bb_lo = day_bars_df["bb_lower"].values
            ema21_v = day_bars_df["ema21"].values
            atr_v = day_bars_df["atr"].values
            kc_upper = ema21_v + 1.5 * atr_v
            kc_lower = ema21_v - 1.5 * atr_v
            squeeze_arr = (bb_lo > kc_lower) & (bb_up < kc_upper)

    # ── NEW: Regime detector (Hurst-based, direction-aware) ──
    regime_trail_mult = 1.0
    regime_hold_mult = 1.0
    regime_score_mult = 1.0
    regime_lot_mult = 1.0
    regime_block_reversion = False
    # Direction-aware overrides: PUTs get separate multipliers in VOLATILE regime
    regime_put_trail_mult = 1.0
    regime_put_hold_mult = 1.0
    regime_put_lot_mult = 1.0
    if cfg.get("use_hurst_regime") and n_bars > 50:
        from orchestrator.regime_detector import RegimeDetector
        rd = RegimeDetector(lookback=50)
        for k in range(min(n_bars, 100)):
            rd.update(closes_arr[k])
        adj = rd.get_strategy_adjustments().get("adjustments", {})
        regime_trail_mult = adj.get("trail_mult", 1.0)
        regime_hold_mult = adj.get("max_hold_mult", 1.0)
        regime_score_mult = adj.get("score_mult", 1.0)
        regime_lot_mult = adj.get("lot_mult", 1.0)
        regime_block_reversion = adj.get("block_reversion", False)
        # PUT-specific overrides (only when direction_aware_regime is enabled)
        if cfg.get("direction_aware_regime", False):
            regime_put_trail_mult = adj.get("put_trail_mult", regime_trail_mult)
            regime_put_hold_mult = adj.get("put_max_hold_mult", regime_hold_mult)
            regime_put_lot_mult = adj.get("put_lot_mult", regime_lot_mult)
        else:
            # Generic: PUTs penalized same as CALLs
            regime_put_trail_mult = regime_trail_mult
            regime_put_hold_mult = regime_hold_mult
            regime_put_lot_mult = regime_lot_mult

    # ── NEW: StdDev filter — compute daily running mean/std ──
    use_stddev = cfg.get("use_stddev_filter", False)
    stddev_threshold = cfg.get("stddev_entry_threshold", 1.0)

    # ── NEW: Gap detection for gap reversal filter ──
    day_open = day_bars_df["open"].iloc[0]
    prev_close_price = prev_day_ohlc["close"] if prev_day_ohlc else day_open
    gap_pct_raw = (day_open - prev_close_price) / prev_close_price if prev_close_price > 0 else 0

    # ── NEW: Consecutive loss tracking ──
    consecutive_losses = 0
    loss_pause_until = -1
    max_consec = cfg.get("max_consecutive_losses", 2)
    loss_cooldown = cfg.get("loss_cooldown_min", 60)

    # ── NEW: Daily P&L tracking for drawdown scaling ──
    daily_realized_pnl = 0.0

    open_trades = []
    closed_trades = []
    total_day_trades = 0
    last_exit_minute = -cfg["cooldown_min"]
    day_close = day_bars_df["close"].iloc[-1]

    for minute_idx in range(n_bars):
        bar_spot = minute_closes[minute_idx]
        bar_dte = max(0.05, dte - minute_idx / 1440)

        # ====== EXITS (every minute) ======
        trades_to_close = []
        for ti, trade in enumerate(open_trades):
            minutes_held = minute_idx - trade["entry_minute"]
            if minutes_held < 1:
                continue

            exit_reason = None
            action = trade["action"]

            # ── NEW: Hard stop loss (30% premium drop) ──
            if cfg.get("hard_sl_pct", 0) > 0:
                cur_prem = bs_premium(bar_spot, trade["strike"], bar_dte, vix, trade["opt_type"])
                prem_loss_pct = (trade["entry_prem"] - cur_prem) / trade["entry_prem"]
                if prem_loss_pct >= cfg["hard_sl_pct"]:
                    exit_reason = "hard_sl"

            if not exit_reason and is_expiry and minute_idx >= cfg["expiry_close_min"]:
                exit_reason = "expiry_close"

            # ── Regime-adjusted max hold (direction-aware) ──
            # PUTs use put_max_hold_mult if available (volatile: 1.0 instead of 0.7)
            if action == "BUY_PUT":
                eff_hold_mult = regime_put_hold_mult
            else:
                eff_hold_mult = regime_hold_mult
            max_hold_put = int(cfg["max_hold_put"] * eff_hold_mult)
            max_hold_call = int(cfg["max_hold_call"] * eff_hold_mult)
            if not exit_reason and action == "BUY_PUT" and minutes_held >= max_hold_put:
                exit_reason = "time_exit"
            elif not exit_reason and action == "BUY_CALL" and minutes_held >= max_hold_call:
                exit_reason = "time_exit"

            # ── Theta exit (Monday 3 PM: exit profitable positions before Tuesday theta decay) ──
            if not exit_reason and cfg.get("theta_exit_enabled", False):
                theta_min = cfg.get("theta_exit_minute", 345)
                # Monday = 0
                if hasattr(date, "weekday"):
                    day_of_week = date.weekday()
                else:
                    day_of_week = -1
                if day_of_week == 0 and minute_idx >= theta_min:
                    is_profitable = False
                    if action == "BUY_PUT" and bar_spot < trade["entry_spot"]:
                        is_profitable = True
                    elif action == "BUY_CALL" and bar_spot > trade["entry_spot"]:
                        is_profitable = True
                    if is_profitable:
                        exit_reason = "theta_exit"

            # ── Trail stop (ATR-based or fixed %, regime-adjusted, direction-aware) ──
            if not exit_reason and action == "BUY_PUT" and minutes_held >= cfg["min_hold_trail_put"]:
                if cfg.get("use_atr_trail", False):
                    # ATR-based trail with ADX-adaptive multiplier
                    cur_atr = atr_arr[minute_idx]
                    cur_adx = adx_arr[minute_idx]
                    if cfg.get("atr_trail_adx_adaptive", False):
                        if cur_adx < 20:
                            atr_mult = cfg.get("atr_trail_adx_low", 1.0)
                        elif cur_adx <= 35:
                            atr_mult = cfg.get("atr_trail_adx_mid", 1.5)
                        else:
                            atr_mult = cfg.get("atr_trail_adx_high", 2.0)
                    else:
                        atr_mult = cfg.get("atr_trail_mult", 1.5)
                    trail_d = cur_atr * atr_mult
                    if bar_spot > trade["best_fav"] + trail_d:
                        exit_reason = "trail_stop"
                else:
                    # Fixed percentage trail (original)
                    eff_trail_mult = regime_put_trail_mult
                    trail_d = trade["entry_spot"] * cfg["trail_pct_put"] * eff_trail_mult
                    if bar_spot > trade["best_fav"] + trail_d:
                        cur_prem_check = bs_premium(bar_spot, trade["strike"], bar_dte, vix, trade["opt_type"])
                        if cur_prem_check > trade["entry_prem"]:
                            exit_reason = "trail_stop"
            elif not exit_reason and action == "BUY_CALL" and minutes_held >= cfg["min_hold_trail_call"]:
                if cfg.get("use_atr_trail", False):
                    # ATR-based trail with ADX-adaptive multiplier
                    cur_atr = atr_arr[minute_idx]
                    cur_adx = adx_arr[minute_idx]
                    if cfg.get("atr_trail_adx_adaptive", False):
                        if cur_adx < 20:
                            atr_mult = cfg.get("atr_trail_adx_low", 1.0)
                        elif cur_adx <= 35:
                            atr_mult = cfg.get("atr_trail_adx_mid", 1.5)
                        else:
                            atr_mult = cfg.get("atr_trail_adx_high", 2.0)
                    else:
                        atr_mult = cfg.get("atr_trail_mult", 1.5)
                    trail_d = cur_atr * atr_mult
                    if bar_spot < trade["best_fav"] - trail_d:
                        exit_reason = "trail_stop"
                else:
                    # Fixed percentage trail (original)
                    trail_d = trade["entry_spot"] * cfg["trail_pct_call"] * regime_trail_mult
                    if bar_spot < trade["best_fav"] - trail_d:
                        exit_reason = "trail_stop"

            # SR stop for CALLs (disabled by config)
            if not exit_reason and action == "BUY_CALL" and minutes_held >= 90:
                if not cfg.get("disable_sr_stop_call", False):
                    call_stop = trade["entry_spot"] - 2.5 * day_atr
                    if bar_spot < call_stop:
                        exit_reason = "sr_stop"

            if exit_reason:
                raw_exit_prem = bs_premium(bar_spot, trade["strike"], bar_dte, vix, trade["opt_type"])
                exit_prem = max(0.05, raw_exit_prem * 0.995 - 2.0)
                pnl = (exit_prem - trade["entry_prem"]) * trade["qty"] - 80
                trade.update({
                    "exit_minute": minute_idx, "exit_spot": round(bar_spot, 2),
                    "exit_prem": round(exit_prem, 2), "exit_reason": exit_reason,
                    "pnl": round(pnl, 0), "minutes_held": minutes_held,
                })
                trades_to_close.append(ti)
                last_exit_minute = minute_idx

                # ── NEW: Track consecutive losses ──
                daily_realized_pnl += pnl
                if pnl < 0:
                    consecutive_losses += 1
                    if consecutive_losses >= max_consec:
                        loss_pause_until = minute_idx + loss_cooldown
                else:
                    consecutive_losses = 0

        for ti in reversed(trades_to_close):
            closed_trades.append(open_trades.pop(ti))

        # Update best favorable tracking
        for trade in open_trades:
            if trade["action"] == "BUY_CALL" and bar_spot > trade["best_fav"]:
                trade["best_fav"] = bar_spot
            elif trade["action"] == "BUY_PUT" and bar_spot < trade["best_fav"]:
                trade["best_fav"] = bar_spot

        # ====== ENTRIES ======
        if minute_idx < 5 or minute_idx > cfg["max_entry_min"]:
            continue
        block_late = cfg.get("block_late_entries", 999)
        if minute_idx > block_late:
            continue
        if len(open_trades) >= cfg["max_concurrent"] or total_day_trades >= cfg["max_trades"]:
            continue
        if minute_idx - last_exit_minute < cfg["cooldown_min"]:
            continue

        # VIX filter
        if vix < cfg["vix_floor"] or vix > cfg["vix_ceil"]:
            continue

        # ── NEW: Consecutive loss throttle ──
        if minute_idx < loss_pause_until:
            continue

        # Avoid windows
        if any(s <= minute_idx < e for s, e in cfg.get("avoid_windows", [])):
            continue

        # Squeeze filter
        if use_squeeze and minute_idx < len(squeeze_arr) and squeeze_arr[minute_idx]:
            continue

        # ── NEW: StdDev filter — only trade when price deviates > 1 sigma ──
        if use_stddev and minute_idx > 20:
            recent_mean = np.mean(closes_arr[max(0, minute_idx-75):minute_idx+1])
            recent_std = np.std(closes_arr[max(0, minute_idx-75):minute_idx+1], ddof=1)
            if recent_std > 0:
                zscore = abs(bar_spot - recent_mean) / recent_std
                if zscore < stddev_threshold:
                    continue  # Price within normal range — no edge

        # ── NEW: Drawdown lot scaling — reduce risk after daily losses ──
        drawdown_mult = 1.0
        if cfg.get("drawdown_lot_scale") and equity > 0:
            daily_loss_pct = daily_realized_pnl / equity
            if daily_loss_pct < -0.015:  # Lost > 1.5% today
                drawdown_mult = cfg.get("drawdown_50pct_mult", 0.5)
            if daily_loss_pct < -0.025:  # Lost > 2.5% today
                drawdown_mult = cfg.get("drawdown_75pct_mult", 0.0)
                if drawdown_mult <= 0:
                    continue  # Stop trading for the day

        # Get entries
        is_hybrid = cfg.get("use_hybrid", False)
        is_v8_tick = (minute_idx % 5 == 0)
        is_comp_tick = (minute_idx % 15 == 0)

        if is_hybrid:
            if not is_v8_tick and not is_comp_tick:
                continue

        entries = []
        if cfg.get("use_hybrid", False):
            if is_v8_tick:
                bar_data = day_bars_df.iloc[minute_idx]
                direction, conf, reasons = detect_entries_v8(
                    bar_data, sr_levels, vix, daily_trend, minute_idx)
                if direction:
                    entries.append((direction, "v8_indicator", conf, False))
            if is_comp_tick:
                comp_entries = detect_entries_composite(
                    cfg, minute_idx // 15, path_15min, support, resistance,
                    vix, gap_pct, best_composite, composite_conf,
                    is_expiry, prev_high, prev_low, above_sma50, above_sma20,
                    bias_val, minute_idx)
                for action, etype, conf, is_zh in comp_entries:
                    entries.append((action, etype, min(1.0, conf * 1.1), is_zh))
        elif use_v8:
            bar_data = day_bars_df.iloc[minute_idx]
            direction, conf, reasons = detect_entries_v8(
                bar_data, sr_levels, vix, daily_trend, minute_idx)
            entries = [(direction, "v8_indicator", conf, False)] if direction else []

        if not entries:
            continue

        entries.sort(key=lambda x: x[2], reverse=True)
        ri = regime_info if regime_info else {"regime": "neutral", "call_mult": 1.0, "put_mult": 1.0}

        for action, entry_type, conf, is_zh in entries:
            if action is None:
                continue

            min_conf = cfg.get("min_confidence_filter", 0)
            if conf < min_conf and not is_zh:
                continue

            # ── Regime score multiplier ──
            conf = min(1.0, conf * regime_score_mult)

            # VWAP filter
            if use_vwap and not is_zh and minute_idx < len(vwap_arr):
                cur_vwap = vwap_arr[minute_idx]
                if not np.isnan(cur_vwap):
                    if action == "BUY_CALL" and bar_spot <= cur_vwap:
                        continue
                    if action == "BUY_PUT" and bar_spot >= cur_vwap:
                        continue

            # RSI gates
            if use_rsi_gate and not is_zh and "rsi" in day_bars_df.columns:
                bar_rsi = float(day_bars_df["rsi"].iloc[minute_idx])
                rsi_call_min = cfg.get("rsi_call_min", 55)
                rsi_put_max = cfg.get("rsi_put_max", 45)
                if action == "BUY_CALL" and bar_rsi < rsi_call_min:
                    continue
                if action == "BUY_PUT" and bar_rsi > rsi_put_max:
                    continue

                # ── NEW: RSI CALL kill zone (block CALL when RSI > 60) ──
                rsi_kill = cfg.get("rsi_call_kill_ceiling", 999)
                if action == "BUY_CALL" and bar_rsi > rsi_kill:
                    continue

            # ── NEW: Gap reversal filter ──
            if cfg.get("gap_reversal_filter") and abs(gap_pct_raw) >= cfg.get("gap_threshold_pct", 0.004):
                if gap_pct_raw < 0 and bar_spot > prev_close_price and action == "BUY_PUT":
                    continue
                if gap_pct_raw > 0 and bar_spot < prev_close_price and action == "BUY_CALL":
                    continue

            # ── NEW: ADX-based confidence adjustment ──
            if cfg.get("use_adx_scoring") and "adx" in day_bars_df.columns:
                bar_adx = float(day_bars_df["adx"].iloc[minute_idx])
                if bar_adx < cfg.get("adx_weak_threshold", 18):
                    conf *= cfg.get("adx_weak_mult", 0.6)
                elif cfg.get("adx_choppy_min", 25) <= bar_adx < cfg.get("adx_choppy_max", 35):
                    conf *= cfg.get("adx_choppy_mult", 0.8)
                # Re-check minimum confidence after ADX dampening
                if conf < min_conf:
                    continue

            # ── NEW: CCI scoring (14-period, computed from bar data) ──
            if cfg.get("use_cci_scoring", False) and minute_idx >= 14:
                tp_arr = (highs[minute_idx-13:minute_idx+1]
                          + lows[minute_idx-13:minute_idx+1]
                          + closes_arr[minute_idx-13:minute_idx+1]) / 3.0
                tp_mean = np.mean(tp_arr)
                mean_dev = np.mean(np.abs(tp_arr - tp_mean))
                cci = (tp_arr[-1] - tp_mean) / (0.015 * mean_dev) if mean_dev > 0 else 0
                cci_thresh = cfg.get("cci_threshold", 100)
                if cci < -cci_thresh and action == "BUY_CALL":
                    conf = min(1.0, conf + 0.05)    # CCI oversold → call boost
                elif cci > cci_thresh and action == "BUY_PUT":
                    conf = min(1.0, conf + 0.05)    # CCI overbought → put boost

            # ── NEW: Williams %R scoring (14-period, computed from bar data) ──
            if cfg.get("use_williams_scoring", False) and minute_idx >= 14:
                hh = np.max(highs[minute_idx-13:minute_idx+1])
                ll = np.min(lows[minute_idx-13:minute_idx+1])
                wr = ((hh - closes_arr[minute_idx]) / (hh - ll) * -100) if hh != ll else -50
                wr_oversold = cfg.get("williams_oversold", -80)
                wr_overbought = cfg.get("williams_overbought", -20)
                if wr < wr_oversold and action == "BUY_CALL":
                    conf = min(1.0, conf + 0.05)    # W%R oversold → call boost
                elif wr > wr_overbought and action == "BUY_PUT":
                    conf = min(1.0, conf + 0.05)    # W%R overbought → put boost

            # ── NEW: VIX contrarian scoring ──
            if cfg.get("use_vix_contrarian", False):
                if vix > 35 and action == "BUY_CALL":
                    conf = min(1.0, conf + 0.10)    # Panic VIX → extreme contrarian buy
                elif vix > 25 and action == "BUY_CALL":
                    conf = min(1.0, conf + 0.05)    # Elevated VIX → contrarian buy

            # ── NEW: VWAP mean reversion (>1.5% from VWAP, ADX<25) ──
            if cfg.get("use_vwap_mean_reversion", False) and minute_idx < len(vwap_arr):
                cur_vwap = vwap_arr[minute_idx]
                cur_adx_val = adx_arr[minute_idx]
                vwap_rev_pct = cfg.get("vwap_reversion_pct", 0.015)
                vwap_adx_max = cfg.get("vwap_reversion_adx_max", 25)
                if not np.isnan(cur_vwap) and cur_vwap > 0 and cur_adx_val < vwap_adx_max:
                    vwap_dist = (bar_spot - cur_vwap) / cur_vwap
                    if vwap_dist < -vwap_rev_pct and action == "BUY_CALL":
                        conf = min(1.0, conf + 0.08)    # Below VWAP → expect reversion up
                    elif vwap_dist > vwap_rev_pct and action == "BUY_PUT":
                        conf = min(1.0, conf + 0.08)    # Above VWAP → expect reversion down

            # ── NEW: Closing hour bias (+0.3 call score after 2:30 PM) ──
            if cfg.get("use_closing_hour_bias", False):
                close_min_idx = cfg.get("closing_hour_min_idx", 315)
                if minute_idx >= close_min_idx and action == "BUY_CALL":
                    conf = min(1.0, conf + 0.03)    # Late-day bullish bias

            # ── NEW: Consecutive down day pattern (3+ days → +0.5 call) ──
            if cfg.get("use_consec_down_day", False):
                min_down_days = cfg.get("consec_down_day_min", 3)
                if consecutive_down_days >= min_down_days and action == "BUY_CALL":
                    conf = min(1.0, conf + 0.05)    # Mean reversion after multi-day decline

            if len(open_trades) >= cfg["max_concurrent"]:
                break
            same_dir = [t for t in open_trades if t["action"] == action]
            if same_dir:
                continue

            if cfg.get("block_call_4th_hour", False) and action == "BUY_CALL":
                if 225 <= minute_idx < 300:
                    continue

            # Regime filtering
            if cfg.get("use_regime_detection", False):
                if ri["regime"] == "bullish" and action == "BUY_PUT":
                    if conf < 0.45:
                        continue
                elif ri["regime"] == "sideways":
                    if conf < 0.35:
                        continue

            strike, opt_type = get_strike_and_type(action, bar_spot, vix, is_zh)
            num_lots = get_dynamic_lots(vix, equity, confidence=conf,
                                         zero_hero=is_zh,
                                         recent_wr=recent_wr,
                                         recent_trades=recent_trades)

            # ── Combined multiplier (apply once to avoid cascading int truncation) ──
            combined_mult = 1.0

            # Entry type quality
            combined_mult *= cfg.get("entry_type_lot_mult", {}).get(entry_type, 1.0)

            # Direction bias
            if cfg.get("use_regime_detection", False):
                if action == "BUY_PUT":
                    combined_mult *= ri.get("put_mult", 1.0)
                elif action == "BUY_CALL":
                    combined_mult *= ri.get("call_mult", 1.0)
            else:
                if action == "BUY_PUT":
                    combined_mult *= cfg.get("put_bias_lot_mult", 1.0)
                elif action == "BUY_CALL":
                    combined_mult *= cfg.get("call_bias_lot_mult", 1.0)

            # Expiry
            if is_expiry:
                combined_mult *= cfg.get("expiry_day_lot_mult", 1.0)

            # VIX lot scaling
            if cfg.get("vix_sweet_min", 0) <= vix <= cfg.get("vix_sweet_max", 999):
                combined_mult *= cfg.get("vix_sweet_lot_mult", 1.0)
            if cfg.get("vix_danger_min", 999) <= vix <= cfg.get("vix_danger_max", 999):
                combined_mult *= cfg.get("vix_danger_lot_mult", 1.0)

            # RSI lot scaling
            if "rsi" in day_bars_df.columns:
                bar_rsi = float(day_bars_df["rsi"].iloc[minute_idx])
                if cfg.get("rsi_sweet_low", 0) <= bar_rsi <= cfg.get("rsi_sweet_high", 0):
                    combined_mult *= cfg.get("rsi_sweet_lot_mult", 1.0)
                if cfg.get("rsi_danger_low", 999) <= bar_rsi <= cfg.get("rsi_danger_high", 999):
                    combined_mult *= cfg.get("rsi_danger_lot_mult", 1.0)

            # Confidence-based boost (reward higher-quality signals)
            if conf >= 0.50:
                combined_mult *= 1.3    # High conviction
            elif conf >= 0.40:
                combined_mult *= 1.15   # Above average

            # Regime lot multiplier (direction-aware: PUTs not penalized in volatile)
            if action == "BUY_PUT":
                combined_mult *= regime_put_lot_mult
            else:
                combined_mult *= regime_lot_mult

            # Drawdown lot scaling
            combined_mult *= drawdown_mult

            # Floor: never reduce below 50% of base
            combined_mult = max(0.5, combined_mult)
            num_lots = max(1, int(num_lots * combined_mult))

            qty = num_lots * LOT_SIZE
            raw_entry_prem = bs_premium(bar_spot, strike, bar_dte, vix, opt_type)
            entry_prem = raw_entry_prem * 1.005 + 2.0

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

    # Force close remaining
    for trade in open_trades:
        raw_exit_prem = bs_premium(day_close, trade["strike"],
                                    max(0.05, dte - n_bars / 1440), vix, trade["opt_type"])
        exit_prem = max(0.05, raw_exit_prem * 0.995 - 2.0)
        pnl = (exit_prem - trade["entry_prem"]) * trade["qty"] - 80
        trade.update({
            "exit_minute": n_bars - 1, "exit_spot": round(day_close, 2),
            "exit_prem": round(exit_prem, 2), "exit_reason": "eod_close",
            "pnl": round(pnl, 0), "minutes_held": n_bars - 1 - trade["entry_minute"],
        })
        closed_trades.append(trade)

    return closed_trades


# ==================================================================
# ENHANCED run_model — same as original but uses enhanced simulate_day
# ==================================================================

def run_model_enhanced(cfg, daily_df, close_prices, day_groups, trading_dates,
                       vix_lookup, daily_trend_df, test_start, starting_equity=None):
    """Run the enhanced V14 model."""
    from backtesting.oos_june2024_test import detect_market_regime

    equity = starting_equity if starting_equity is not None else CAPITAL
    start_equity = equity
    peak = equity
    max_dd = 0
    all_trades = []
    daily_pnls = []
    entry_stats = defaultdict(lambda: {"count": 0, "pnl": 0, "wins": 0})
    exit_stats = defaultdict(lambda: {"count": 0, "pnl": 0, "wins": 0})
    recent_results = []
    consecutive_down_days = 0  # Track consecutive down days for scoring

    for i in range(len(daily_df)):
        date = daily_df.index[i].date()
        if date < test_start:
            continue
        if date not in day_groups:
            continue
        avoid_days = cfg.get("avoid_days", [])
        if date.strftime("%A") in avoid_days:
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
        vix_val = vix_lookup.get(date, 14.0)

        is_expiry = date.strftime("%A") == "Thursday"  # 2024 data = Thursday expiry
        dow = date.weekday()
        target = 3
        if dow <= target:
            dte_val = max(target - dow, 0.5)
        else:
            dte_val = max(7 - dow + target, 0.5)

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

        regime_info = {"regime": "neutral", "strength": 0.0, "call_mult": 1.0, "put_mult": 1.0}
        if cfg.get("use_regime_detection", False):
            regime_info = detect_market_regime(daily_df, i)

        day_trades = simulate_day_enhanced(
            cfg, day_bars, date, prev_ohlc, vix_val, daily_trend, dte_val, is_expiry,
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
            regime_info=regime_info,
            consecutive_down_days=consecutive_down_days,
        )

        # Update consecutive down days: day was "down" if close < open
        day_open_price = day_bars["open"].iloc[0]
        day_close_price = day_bars["close"].iloc[-1]
        if day_close_price < day_open_price:
            consecutive_down_days += 1
        else:
            consecutive_down_days = 0

        # BTST
        if cfg.get("btst_enabled") and i + 1 < len(daily_df):
            next_row = daily_df.iloc[i + 1]
            next_open = float(next_row["Open"])
            day_close = float(row["Close"])
            for t in day_trades:
                if (t["action"] == "BUY_PUT" and t["pnl"] > 0 and not is_expiry
                        and vix_val < cfg.get("btst_vix_cap", 25)
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

    net = equity - start_equity
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
    avg_lots = np.mean(lots_list) if lots_list else 0
    max_lots = max(lots_list) if lots_list else 0

    return {
        "name": cfg["name"], "net_pnl": round(net),
        "return_pct": round(net / max(start_equity, 1) * 100, 1),
        "start_equity": round(start_equity), "final_equity": round(equity),
        "total_trades": total, "trades_per_day": round(total / max(len(daily_pnls), 1), 1),
        "win_rate": round(wr, 1), "wins": len(wins), "losses": len(losses),
        "sharpe": round(sharpe, 2), "profit_factor": round(pf, 2),
        "max_drawdown": round(max_dd, 1),
        "avg_win": round(np.mean([t["pnl"] for t in wins])) if wins else 0,
        "avg_loss": round(np.mean([t["pnl"] for t in losses])) if losses else 0,
        "pnl_per_trade": round(net / max(total, 1)),
        "btst_count": len([t for t in all_trades if t.get("btst_pnl", 0) != 0]),
        "btst_pnl": round(sum(t.get("btst_pnl", 0) for t in all_trades)),
        "avg_lots": round(avg_lots, 1), "max_lots": max_lots,
        "entry_stats": dict(entry_stats), "exit_stats": dict(exit_stats),
        "all_trades": all_trades,
    }


# ==================================================================
# COMPARISON RUNNER
# ==================================================================

def run_comparison():
    """Run V14 Original vs Enh-Lean vs Enh-Tuned vs Enh-Tuned2 on 6 months of data.

    Tests four variants:
      1. V14_Original  -- baseline V9_HYBRID_CONFIG (uses run_model)
      2. Enh-Lean      -- entry filters + original exit logic (prev best at 8.9x)
      3. Enh-Tuned     -- ALL 21 scoring signals + Lean's fixed % trails + theta exit
      4. Enh-Tuned2    -- Same as Enh-Tuned but with daily-scale ATR trails
    """
    print("=" * 130)
    print("  V14 ORIGINAL vs ENH-LEAN vs ENH-TUNED vs ENH-TUNED2 -- 6-Month Comparison")
    print("  Capital: Rs 2,00,000 | Dynamic lot sizing | Compounding across months")
    print("  V14_Original: Baseline V9_HYBRID_CONFIG")
    print("  Enh-Lean:     Entry filters + original exit logic (prev best, no throttle)")
    print("  Enh-Tuned:    ALL 21 scoring + Lean's fixed % trails + theta exit")
    print("  Enh-Tuned2:   ALL 21 scoring + daily-scale ATR trails (mult=30)")
    print("=" * 130)

    # 6 test months (all have cached data)
    TEST_MONTHS = [
        ("2024-06-01", "2024-07-31", 2024,  7, "Jul-2024"),
        ("2024-07-01", "2024-08-31", 2024,  8, "Aug-2024"),
        ("2024-08-01", "2024-09-30", 2024,  9, "Sep-2024"),
        ("2024-09-01", "2024-10-31", 2024, 10, "Oct-2024"),
        ("2024-11-01", "2024-12-31", 2024, 12, "Dec-2024"),
        ("2024-12-01", "2025-01-31", 2025,  1, "Jan-2025"),
    ]

    original_cfg = copy.deepcopy(V9_HYBRID_CONFIG)
    original_cfg["name"] = "V14_Original"

    # Variant 1: LEAN Enhanced -- entry filters + ORIGINAL exit logic (the previous best at 8.9x)
    # Hypothesis: wider trails kill slot turnover, throttle/drawdown block crash entries
    # Fix: keep Original trail values, no throttle, no drawdown scaling, no regime exit adj
    enh_lean = make_v14_enhanced_config()
    enh_lean["name"] = "Enh-Lean"
    enh_lean["use_stddev_filter"] = False
    enh_lean["direction_aware_regime"] = True
    # REVERT exit params to Original values (slot turnover matters)
    enh_lean["trail_pct_put"] = 0.015      # Original value (was 0.020)
    enh_lean["trail_pct_call"] = 0.008     # Original value (was 0.012)
    enh_lean["min_hold_trail_put"] = 120   # Original value (was 150)
    enh_lean["min_hold_trail_call"] = 60   # Original value (was 60)
    # DISABLE risk management that blocks crash entries
    enh_lean["max_consecutive_losses"] = 99   # Effectively disabled
    enh_lean["drawdown_lot_scale"] = False     # Don't scale lots on drawdown
    # DISABLE regime exit adjustments (Original has none)
    enh_lean["use_hurst_regime"] = False        # No regime trail/hold changes
    # KEEP entry quality improvements:
    # - RSI kill zone (rsi_call_kill_ceiling = 60)
    # - ADX scoring
    # - Gap reversal filter
    # - Combined lot multiplier with confidence boost
    # - No drawdown scaling so crash entries aren't blocked

    # Variant 2: Enh-Tuned -- ALL 21 scoring + Lean's fixed % trails
    enh_tuned = make_v14_enh_tuned_config()

    # Variant 3: Enh-Tuned2 -- same but with daily-scale ATR trails
    enh_tuned2 = make_v14_enh_tuned2_config()

    # All configs
    model_configs = [
        ("V14_Original", original_cfg, "original"),
        ("Enh-Lean",     enh_lean,     "enh_lean"),
        ("Enh-Tuned",    enh_tuned,    "enh_tuned"),
        ("Enh-Tuned2",   enh_tuned2,   "enh_tuned2"),
    ]

    model_names = [m[0] for m in model_configs]
    accumulated_equity = {name: CAPITAL for name in model_names}
    month_summaries = []

    for warmup_start, data_end, test_yr, test_mo, label in TEST_MONTHS:
        print(f"\n{'='*110}")
        print(f"  {label} (warmup from {warmup_start})")
        print(f"  Capital: " + " | ".join(
            f"{n}: Rs {accumulated_equity[n]:>,}" for n in model_names))
        print(f"{'='*100}")

        nifty, vix = download_range(warmup_start, data_end)
        if nifty is None:
            print(f"  SKIPPED: No data for {label}")
            continue

        # Add indicators
        nifty_ind = add_all_indicators(nifty.copy())

        # Group by date
        day_groups = {date: group for date, group in nifty_ind.groupby(nifty_ind.index.date)}
        all_dates = sorted(day_groups.keys())

        # Build daily OHLC
        vix_lookup = {}
        if vix is not None and not vix.empty:
            for idx, row in vix.iterrows():
                vix_lookup[idx.date()] = row["close"]

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
            vix_val = vix_lookup.get(idx_date.date(), 14.0)
            daily.loc[idx_date, "VIX"] = vix_val
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
        daily["PrevHigh"] = daily["High"].shift(1)
        daily["PrevLow"] = daily["Low"].shift(1)
        daily["VIXSpike"] = daily["VIX"] > daily["PrevVIX"] * 1.15
        daily["GapPct"] = (daily["Open"] - daily["Close"].shift(1)) / daily["Close"].shift(1) * 100

        close_prices = daily["Close"].values.tolist()
        daily_trend_df = daily[["Close", "SMA20", "EMA9", "EMA21"]].rename(
            columns={"Close": "close", "SMA20": "sma20", "EMA9": "ema9", "EMA21": "ema21"})
        daily_trend_df.index = daily_trend_df.index.date

        test_start = dt.date(test_yr, test_mo, 1)
        test_dates = [d for d in all_dates if d >= test_start]
        if not test_dates:
            continue

        results = {}

        # Run V14 Original
        eq = accumulated_equity["V14_Original"]
        r_orig = run_model(original_cfg, daily, close_prices, day_groups, test_dates,
                           vix_lookup, daily_trend_df, test_start, starting_equity=eq)
        results["V14_Original"] = r_orig

        # Run Enhanced variants
        for name, cfg, _key in model_configs:
            if name == "V14_Original":
                continue  # Already ran
            eq = accumulated_equity[name]
            r = run_model_enhanced(cfg, daily, close_prices, day_groups, test_dates,
                                   vix_lookup, daily_trend_df, test_start, starting_equity=eq)
            results[name] = r

        # Update accumulated equity
        for name in model_names:
            if name in results:
                accumulated_equity[name] = results[name]["final_equity"]

        # Print month summary
        print(f"\n  {'Model':<16} {'Start Cap':>12} {'P&L':>12} {'Return':>8} {'End Cap':>12} "
              f"{'Trades':>7} {'WR':>6} {'Sharpe':>7} {'MaxDD':>7} {'PF':>5} {'Lots':>5}")
        print(f"  {'-'*110}")
        for name in model_names:
            if name in results:
                r = results[name]
                pf_str = f"{r['profit_factor']:.1f}" if r['profit_factor'] < 100 else "inf"
                print(f"  {name:<16} Rs{r['start_equity']:>9,} Rs{r['net_pnl']:>+9,} {r['return_pct']:>+7.1f}% "
                      f"Rs{r['final_equity']:>9,} {r['total_trades']:>5}t "
                      f"{r['win_rate']:>5.1f}% {r['sharpe']:>6.2f} {r['max_drawdown']:>6.1f}% "
                      f"{pf_str:>5} {r['avg_lots']:>4.1f}")

        month_summaries.append({
            "month": label,
            "original": results.get("V14_Original", {}),
            "enh_lean": results.get("Enh-Lean", {}),
            "enh_tuned": results.get("Enh-Tuned", {}),
            "enh_tuned2": results.get("Enh-Tuned2", {}),
        })

    # ================================================================
    # GRAND SUMMARY
    # ================================================================
    print(f"\n\n{'='*130}")
    print(f"  GRAND SUMMARY -- 6-Month Comparison (4 Variants)")
    print(f"{'='*130}")

    # Map model names to summary keys
    summary_key_map = {
        "V14_Original": "original",
        "Enh-Lean": "enh_lean",
        "Enh-Tuned": "enh_tuned",
        "Enh-Tuned2": "enh_tuned2",
    }

    # Aggregate stats per model
    for name in model_names:
        all_trades = []
        skey = summary_key_map[name]
        for ms in month_summaries:
            r = ms.get(skey, {})
            all_trades.extend(r.get("all_trades", []))

        total = len(all_trades)
        wins = [t for t in all_trades if t["pnl"] > 0]
        losses = [t for t in all_trades if t["pnl"] <= 0]
        total_pnl = sum(t["pnl"] for t in all_trades)
        wr = len(wins) / total * 100 if total else 0
        gw = sum(t["pnl"] for t in wins)
        gl = abs(sum(t["pnl"] for t in losses))
        pf = gw / gl if gl > 0 else float("inf")
        avg_win = np.mean([t["pnl"] for t in wins]) if wins else 0
        avg_loss = np.mean([t["pnl"] for t in losses]) if losses else 0
        final_eq = accumulated_equity[name]
        ret_x = final_eq / CAPITAL

        print(f"\n  {name}:")
        print(f"    Final Equity:    Rs {final_eq:>12,}  ({ret_x:.1f}x return)")
        print(f"    Total P&L:       Rs {total_pnl:>+12,.0f}")
        print(f"    Trades:          {total:>8}")
        print(f"    Win Rate:        {wr:>7.1f}%")
        print(f"    Profit Factor:   {pf:>7.2f}")
        print(f"    Avg Win:         Rs {avg_win:>+10,.0f}")
        print(f"    Avg Loss:        Rs {avg_loss:>+10,.0f}")

        # Entry type breakdown
        entry_counts = defaultdict(lambda: {"count": 0, "pnl": 0, "wins": 0})
        for t in all_trades:
            et = t.get("entry_type", "?")
            entry_counts[et]["count"] += 1
            entry_counts[et]["pnl"] += t["pnl"]
            if t["pnl"] > 0: entry_counts[et]["wins"] += 1

        print(f"\n    Entry Type Breakdown:")
        print(f"    {'Type':<25} {'Trades':>7} {'WR':>6} {'P&L':>14}")
        print(f"    {'-'*55}")
        for et, stats in sorted(entry_counts.items(), key=lambda x: x[1]["pnl"], reverse=True):
            et_wr = stats["wins"] / stats["count"] * 100 if stats["count"] else 0
            print(f"    {et:<25} {stats['count']:>5}t {et_wr:>5.1f}% Rs{stats['pnl']:>+12,.0f}")

        # Exit reason breakdown
        exit_counts = defaultdict(lambda: {"count": 0, "pnl": 0, "wins": 0})
        for t in all_trades:
            er = t.get("exit_reason", "?")
            exit_counts[er]["count"] += 1
            exit_counts[er]["pnl"] += t["pnl"]
            if t["pnl"] > 0: exit_counts[er]["wins"] += 1

        print(f"\n    Exit Reason Breakdown:")
        print(f"    {'Reason':<25} {'Trades':>7} {'WR':>6} {'P&L':>14}")
        print(f"    {'-'*55}")
        for er, stats in sorted(exit_counts.items(), key=lambda x: x[1]["pnl"], reverse=True):
            er_wr = stats["wins"] / stats["count"] * 100 if stats["count"] else 0
            print(f"    {er:<25} {stats['count']:>5}t {er_wr:>5.1f}% Rs{stats['pnl']:>+12,.0f}")

    # Month-by-month comparison table (4-way)
    print(f"\n\n  {'='*150}")
    print(f"  MONTH-BY-MONTH COMPARISON")
    print(f"  {'='*150}")
    print(f"  {'Month':<10} {'Orig P&L':>11} {'Lean P&L':>11} {'Tuned P&L':>11} {'Tuned2 P&L':>12} "
          f"{'O-WR':>6} {'L-WR':>6} {'T-WR':>6} {'T2-WR':>6} "
          f"{'O-Trd':>6} {'L-Trd':>6} {'T-Trd':>6} {'T2-Trd':>7}  Best")
    print(f"  {'-'*140}")
    totals = {n: 0 for n in model_names}
    wins_count = {n: 0 for n in model_names}
    for ms in month_summaries:
        orig = ms.get("original", {})
        lean = ms.get("enh_lean", {})
        tuned = ms.get("enh_tuned", {})
        tuned2 = ms.get("enh_tuned2", {})
        pnls = {
            "V14_Original": orig.get("net_pnl", 0),
            "Enh-Lean": lean.get("net_pnl", 0),
            "Enh-Tuned": tuned.get("net_pnl", 0),
            "Enh-Tuned2": tuned2.get("net_pnl", 0),
        }
        for n in model_names:
            totals[n] += pnls[n]
        best_name = max(pnls, key=pnls.get)
        wins_count[best_name] += 1
        best_label = {"V14_Original": "Orig", "Enh-Lean": "Lean", "Enh-Tuned": "Tuned", "Enh-Tuned2": "Tuned2"}[best_name]
        print(f"  {ms['month']:<10} Rs{pnls['V14_Original']:>+9,} Rs{pnls['Enh-Lean']:>+9,} Rs{pnls['Enh-Tuned']:>+9,} Rs{pnls['Enh-Tuned2']:>+10,} "
              f"{orig.get('win_rate', 0):>5.1f}% {lean.get('win_rate', 0):>5.1f}% {tuned.get('win_rate', 0):>5.1f}% {tuned2.get('win_rate', 0):>5.1f}% "
              f"{orig.get('total_trades', 0):>5}t {lean.get('total_trades', 0):>5}t {tuned.get('total_trades', 0):>5}t {tuned2.get('total_trades', 0):>5}t  "
              f"{best_label}")

    print(f"  {'-'*140}")
    print(f"  {'TOTAL':<10} Rs{totals['V14_Original']:>+9,} Rs{totals['Enh-Lean']:>+9,} Rs{totals['Enh-Tuned']:>+9,} Rs{totals['Enh-Tuned2']:>+10,}")
    print(f"\n  Month wins: " + " | ".join(f"{n}: {wins_count[n]}" for n in model_names))

    print(f"\n  Final equity:")
    best_model = max(model_names, key=lambda n: accumulated_equity[n])
    for name in model_names:
        eq = accumulated_equity[name]
        ret_x = eq / CAPITAL
        marker = " *** BEST ***" if name == best_model else ""
        print(f"    {name:<16} Rs {eq:>12,}  ({ret_x:.1f}x){marker}")

    print(f"\n{'='*130}")


if __name__ == "__main__":
    run_comparison()
