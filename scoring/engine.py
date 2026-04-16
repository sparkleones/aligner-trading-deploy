"""V14 Shared Scoring Engine — ONE source of truth for backtest + live.

All entry scoring, confluence filtering, exit evaluation, and lot sizing
live here. Both the backtester and live agent import these functions.

Ported from v14_live_agent.py (the production code) to ensure live = backtest.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# ENTRY SCORING
# ─────────────────────────────────────────────────────────────

def score_entry(
    ind: dict,
    vix: float,
    cfg: dict,
    *,
    pcr: float = 0.0,
    bar_idx: int = 0,
    oi_support: float = 0.0,
    oi_resistance: float = 0.0,
    max_pain: float = 0.0,
    consecutive_down_days: int = 0,
    fii_net: float = 0.0,
    dii_net: float = 0.0,
    regime_block_reversion: bool = False,
    oi_ce_change_pct: float = 0.0,
    oi_pe_change_pct: float = 0.0,
    is_expiry: bool = False,
) -> tuple:
    """V14 indicator scoring. Returns (action, confidence) or (None, 0).

    Parameters
    ----------
    ind : dict
        Indicator snapshot from compute_indicators().
    vix : float
        Current India VIX.
    cfg : dict
        V14 config dict.
    pcr : float
        Put-Call Ratio (0 = unavailable).
    bar_idx : int
        Current bar index (0 = market open).
    oi_support/oi_resistance : float
        OI-derived S/R levels (0 = unavailable).
    max_pain : float
        Max pain strike (0 = unavailable).
    consecutive_down_days : int
        Number of consecutive down days.
    fii_net/dii_net : float
        Institutional flow data (0 = unavailable).
    regime_block_reversion : bool
        If True, suppress mean-reversion signals (from regime detector).
    """
    spot = ind["close"]
    call_score = 0.0
    put_score = 0.0

    # ── Supertrend ──
    if ind["st_direction"] == 1:
        call_score += 2.5
    elif ind["st_direction"] == -1:
        put_score += 3.0

    # ── EMA alignment ──
    if ind["ema9_above_ema21"]:
        call_score += 2.0
    else:
        put_score += 3.5

    # ── RSI — ADX-gated and regime-gated mean-reversion ──
    rsi = ind["rsi"]
    adx_val = ind.get("adx", 25)
    if 30 <= rsi < 50:
        put_score += 1.5
    elif rsi < 30:
        if regime_block_reversion:
            call_score += 0.0
        elif adx_val < 25:
            call_score += 2.0
        else:
            call_score += 0.5
    elif rsi > 70:
        if regime_block_reversion:
            put_score += 0.0
        elif adx_val < 25:
            put_score += 2.0
        else:
            put_score += 0.5

    # ── MACD ──
    if ind["macd_hist"] > 0:
        call_score += 0.5
    elif ind["macd_hist"] < 0:
        put_score += 0.5

    # ── Bollinger Bands ──
    if spot <= ind["bb_lower"]:
        call_score += 1.5
    if spot >= ind["bb_upper"]:
        put_score += 1.5

    # ── VIX regime ──
    if 13 <= vix < 16:
        put_score += 1.5
    elif vix >= 16:
        put_score += 1.0

    # ── ADX trend strength dampening/boosting ──
    adx = ind.get("adx", 25)
    if adx < cfg.get("adx_weak_threshold", 18):
        call_score *= cfg.get("adx_weak_mult", 0.6)
        put_score *= cfg.get("adx_weak_mult", 0.6)
    elif cfg.get("adx_choppy_min", 25) <= adx < cfg.get("adx_choppy_max", 35):
        call_score *= cfg.get("adx_choppy_mult", 0.8)
        put_score *= cfg.get("adx_choppy_mult", 0.8)
    elif adx >= cfg.get("adx_strong_threshold", 35):
        if ind.get("plus_di", 0) > ind.get("minus_di", 0):
            call_score += 1.0
        else:
            put_score += 1.0

    # ── Stochastic RSI reversal confirmation ──
    if ind.get("stoch_oversold") and ind.get("stoch_cross_up"):
        call_score += 1.0
    if ind.get("stoch_overbought") and ind.get("stoch_cross_down"):
        put_score += 1.0

    # ── CCI mean-reversion ──
    cci = ind.get("cci", 0)
    if cci < -100:
        call_score += 0.5
    elif cci > 100:
        put_score += 0.5

    # ── Williams %R extremes ──
    williams_r = ind.get("williams_r", -50)
    if williams_r < -80:
        call_score += 0.5
    elif williams_r > -20:
        put_score += 0.5

    # ── RSI Bullish Divergence ──
    if ind.get("rsi_bullish_divergence"):
        call_score += 1.5

    # ── PCR extreme contrarian ──
    if cfg.get("use_pcr_filter") and pcr > 0:
        if pcr > cfg.get("pcr_bearish_min", 1.6):
            call_score += 1.0
        elif pcr < cfg.get("pcr_bullish_max", 0.7):
            if pcr < 0.6:
                put_score += 1.0
            else:
                call_score += 0.5

    # ── OI Buildup S/R gating ──
    if cfg.get("use_oi_levels") and spot > 0:
        if oi_resistance > 0:
            dist_to_res = (oi_resistance - spot) / spot
            if 0 < dist_to_res < 0.005:
                call_score -= 1.0
        if oi_support > 0:
            dist_to_sup = (spot - oi_support) / spot
            if 0 < dist_to_sup < 0.005:
                put_score -= 1.0

    # ── Max pain mean reversion ──
    if max_pain > 0 and spot > 0 and vix < 25 and adx < 35:
        max_pain_dist_pct = (spot - max_pain) / max_pain
        if max_pain_dist_pct > 0.01:
            put_score += 0.5
        elif max_pain_dist_pct < -0.01:
            call_score += 0.5

    # ── VIX > 25 contrarian ──
    if vix > 35:
        call_score += 1.0
    elif vix > 25:
        call_score += 0.5

    # ── Closing hour bias (bar >= 63 = 2:30 PM) ──
    if bar_idx >= 63:
        call_score += 0.3

    # ── Consecutive DOWN day pattern ──
    if consecutive_down_days >= 3:
        call_score += 0.5

    # ── FII/DII flow ──
    if fii_net < -5000:
        put_score += 0.3
    if fii_net > 0 and dii_net > 0:
        call_score += 0.3

    # ── VWAP mean reversion (only in range-bound) ──
    vwap = ind.get("vwap", spot)
    if vwap > 0 and adx_val < 25:
        vwap_dist_pct = (spot - vwap) / vwap
        if vwap_dist_pct < -0.015:
            call_score += 0.8
        elif vwap_dist_pct > 0.015:
            put_score += 0.8

    # ══════════════════════════════════════════════════════════
    # ROUND 5: NEW ALGO TRADING SCORING FACTORS (config-gated)
    # ══════════════════════════════════════════════════════════

    # ── Connors RSI mean-reversion signal ──
    if cfg.get("use_connors_rsi"):
        crsi = ind.get("connors_rsi", 50)
        crsi_oversold = cfg.get("connors_rsi_oversold", 15)
        crsi_overbought = cfg.get("connors_rsi_overbought", 85)
        if crsi < crsi_oversold:
            call_score += cfg.get("connors_rsi_score", 1.5)
        elif crsi > crsi_overbought:
            put_score += cfg.get("connors_rsi_score", 1.5)

    # ── KAMA trend confirmation ──
    if cfg.get("use_kama_filter"):
        if ind.get("kama_slope_up") and not ind.get("kama_above_price"):
            call_score += cfg.get("kama_score", 1.0)
        elif not ind.get("kama_slope_up") and ind.get("kama_above_price"):
            put_score += cfg.get("kama_score", 1.0)

    # ── Heikin Ashi trend confirmation ──
    if cfg.get("use_heikin_ashi"):
        ha_score = cfg.get("ha_trend_score", 1.0)
        if ind.get("ha_bullish") and ind.get("ha_body_ratio", 0) > 0.6:
            call_score += ha_score
        elif ind.get("ha_bearish") and ind.get("ha_body_ratio", 0) > 0.6:
            put_score += ha_score

    # ── Donchian Channel breakout ──
    if cfg.get("use_donchian"):
        dc_score = cfg.get("donchian_score", 1.5)
        if ind.get("donchian_breakout_up"):
            call_score += dc_score
        elif ind.get("donchian_breakout_down"):
            put_score += dc_score

    # ── Parabolic SAR trend confirmation ──
    if cfg.get("use_psar"):
        psar_score = cfg.get("psar_score", 0.8)
        if ind.get("psar_bullish"):
            call_score += psar_score
        else:
            put_score += psar_score

    # ── RSI Bearish Divergence ──
    if cfg.get("use_rsi_bear_div") and ind.get("rsi_bearish_divergence"):
        put_score += cfg.get("rsi_bear_div_score", 1.5)

    # ── MACD Signal Line crossover ──
    if cfg.get("use_macd_crossover"):
        macd_x_score = cfg.get("macd_cross_score", 1.0)
        if ind.get("macd_cross_up"):
            call_score += macd_x_score
        elif ind.get("macd_cross_down"):
            put_score += macd_x_score

    # ── VWAP band extremes (2-sigma reversion) ──
    if cfg.get("use_vwap_bands"):
        vwap_bd = ind.get("vwap_band_dist", 0)
        if vwap_bd < -2.0:
            call_score += cfg.get("vwap_band_score", 1.0)
        elif vwap_bd > 2.0:
            put_score += cfg.get("vwap_band_score", 1.0)

    # ══════════════════════════════════════════════════════════
    # V15: ENHANCED SCORING FACTORS (config-gated)
    # ══════════════════════════════════════════════════════════

    # ── V15: Volume spike confirmation ──
    # High volume = institutional participation = better follow-through
    if cfg.get("use_volume_confirmation"):
        vol_ratio = ind.get("volume_ratio", 1.0)
        if vol_ratio > 2.0:
            # Strong volume spike: boost the winning direction
            call_score *= 1.15
            put_score *= 1.15
        elif vol_ratio < 0.5:
            # Low volume: weak signal, dampen scores
            call_score *= cfg.get("low_volume_mult", 0.8)
            put_score *= cfg.get("low_volume_mult", 0.8)

    # ── V15: OBV trend confirmation ──
    # OBV rising = accumulation → supports calls; OBV falling = distribution → supports puts
    if cfg.get("use_obv_confirmation"):
        obv_score = cfg.get("obv_score", 0.8)
        if ind.get("obv_rising"):
            call_score += obv_score
        elif ind.get("obv_falling"):
            put_score += obv_score

    # ── V15: Momentum acceleration ──
    # Accelerating momentum in trade direction = higher conviction
    if cfg.get("use_momentum_accel"):
        accel_score = cfg.get("momentum_accel_score", 0.7)
        if ind.get("momentum_accel_up"):
            call_score += accel_score
        elif ind.get("momentum_accel_down"):
            put_score += accel_score

    # ── V15: Multi-timeframe EMA alignment ──
    # All 3 EMAs (5, 9, 21) stacked = strong trend confirmation
    if cfg.get("use_ema_stack"):
        ema_stack_score = cfg.get("ema_stack_score", 1.2)
        if ind.get("ema_all_bullish"):
            call_score += ema_stack_score
        elif ind.get("ema_all_bearish"):
            put_score += ema_stack_score

    # ── V15: Session-specific weight adjustment ──
    # Morning = trend-following (boost momentum signals)
    # Afternoon = mean-reversion (boost reversal signals)
    if cfg.get("use_session_weights"):
        if 3 <= bar_idx <= 15:  # Morning power hour
            # Trend-following boost: Supertrend + EMA alignment matter more
            morning_boost = cfg.get("morning_trend_boost", 1.1)
            call_score *= morning_boost
            put_score *= morning_boost
        elif 59 <= bar_idx <= 69:  # Afternoon window
            # Afternoon requires higher conviction + ADX filter
            afternoon_adx_min = cfg.get("afternoon_adx_min", 28)
            if adx < afternoon_adx_min:
                call_score *= 0.7
                put_score *= 0.7

    # ── V15: RV/IV entry quality ──
    # When realized vol > implied vol, option buyers have edge
    if cfg.get("use_rv_iv_filter"):
        rv_iv = ind.get("rv_iv_ratio", 1.0)
        if rv_iv < cfg.get("rv_iv_min", 0.8):
            # IV overstates movement → options are expensive → dampen
            call_score *= 0.85
            put_score *= 0.85
        elif rv_iv > cfg.get("rv_iv_boost_threshold", 1.3):
            # Market moving more than options price → good for buyers
            call_score *= 1.1
            put_score *= 1.1

    # ══════════════════════════════════════════════════════════
    # V16: REGIME-AWARE SCORING (config-gated)
    # ══════════════════════════════════════════════════════════

    # ── V16: ADX + BB Width regime filter ──
    # Block entries when market is ranging (low ADX + compressed BB)
    if cfg.get("use_regime_filter"):
        regime_adx_min = cfg.get("regime_adx_min", 20)
        regime_bb_pctile_min = cfg.get("regime_bb_pctile_min", 30)
        bb_pctile = ind.get("bb_width_pctile", 50)
        if adx < regime_adx_min and bb_pctile < regime_bb_pctile_min:
            # Ranging market: dampen all scores
            regime_dampen = cfg.get("regime_ranging_mult", 0.5)
            call_score *= regime_dampen
            put_score *= regime_dampen

    # ── V16: OBV divergence boost ──
    # Price diverging from volume = reversal signal
    if cfg.get("use_obv_divergence"):
        obv_div_score = cfg.get("obv_div_score", 1.5)
        if ind.get("obv_bearish_div"):
            put_score += obv_div_score
        if ind.get("obv_bullish_div"):
            call_score += obv_div_score

    # ── V16: Volume climax reversal ──
    if cfg.get("use_volume_climax"):
        if ind.get("volume_climax"):
            climax_score = cfg.get("volume_climax_score", 1.0)
            # Volume climax often signals exhaustion → boost reversal
            if ind.get("obv_bearish_div"):
                put_score += climax_score
            if ind.get("obv_bullish_div"):
                call_score += climax_score

    # ══════════════════════════════════════════════════════════
    # RESEARCH-DRIVEN: OI Change scoring (delta OI buildup/unwinding)
    # ══════════════════════════════════════════════════════════
    # When Put OI builds rapidly = support forming (bullish)
    # When Call OI builds rapidly = resistance forming (bearish)
    # When OI unwinds at a level = support/resistance weakening
    if cfg.get("use_oi_change_scoring"):
        buildup_score = cfg.get("oi_change_buildup_score", 0.5)
        unwind_score = cfg.get("oi_change_unwinding_score", -0.3)

        # Put OI building = writers selling puts = expecting support = bullish
        if oi_pe_change_pct > 10:
            call_score += buildup_score
        elif oi_pe_change_pct < -10:
            call_score += unwind_score  # Put support weakening = bearish

        # Call OI building = writers selling calls = expecting resistance = bearish
        if oi_ce_change_pct > 10:
            put_score += buildup_score
        elif oi_ce_change_pct < -10:
            put_score += unwind_score  # Call resistance weakening = bullish

    # ══════════════════════════════════════════════════════════
    # RESEARCH-DRIVEN: Enhanced expiry day max pain convergence
    # ══════════════════════════════════════════════════════════
    # On expiry day after 2 PM, NIFTY gravitates toward max pain
    # Stronger pull effect in final hours — trade toward max pain
    if cfg.get("use_expiry_max_pain_boost") and is_expiry and max_pain > 0 and spot > 0:
        expiry_bar = cfg.get("expiry_max_pain_after_bar", 57)
        if bar_idx >= expiry_bar:
            mp_dist_pct = (spot - max_pain) / max_pain
            expiry_mp_score = cfg.get("expiry_max_pain_score", 1.0)
            if mp_dist_pct > 0.003:  # Spot above max pain = pull down
                put_score += expiry_mp_score
            elif mp_dist_pct < -0.003:  # Spot below max pain = pull up
                call_score += expiry_mp_score

    # ── Decision ──
    if put_score >= cfg["put_score_min"] and put_score > call_score:
        conf = min(1.0, put_score / 18.0)
        return "BUY_PUT", conf
    elif call_score >= cfg["call_score_min"] and call_score > put_score:
        conf = min(1.0, call_score / 18.0)
        return "BUY_CALL", conf
    return None, 0


# ─────────────────────────────────────────────────────────────
# CONFLUENCE FILTERS
# ─────────────────────────────────────────────────────────────

def passes_confluence(
    action: str,
    conf: float,
    ind: dict,
    bar_idx: int,
    cfg: dict,
    *,
    current_spot: float = 0.0,
    oi_support: float = 0.0,
    oi_resistance: float = 0.0,
    prev_close: float = 0.0,
    day_open: float = 0.0,
    iv_percentile: float = 50.0,
) -> bool:
    """V14 confluence filter — identical logic for backtest and live.

    Parameters
    ----------
    current_spot : float
        Real-time spot price (live). 0 = use ind["close"].
    prev_close / day_open : float
        For gap reversal filter.
    """
    if conf < cfg["min_confidence_filter"]:
        return False

    # ── Trend regime gate (EMA50-based) ──
    # Block counter-trend entries: CALLs in confirmed downtrend (close < ema50
    # AND ema50 sloping down) or PUTs in confirmed uptrend.
    # trend_regime: 1 = uptrend, -1 = downtrend, 0 = neutral (no block)
    if cfg.get("use_trend_regime_gate"):
        regime = ind.get("trend_regime", 0)
        if action == "BUY_CALL" and regime == -1:
            return False
        if action == "BUY_PUT" and regime == 1:
            return False

    # ── Day-of-week entry gate (per-entry block; complements full-day avoid_days) ──
    # block_entry_dows: [int] — Python weekday ints (0=Mon..6=Sun)
    block_dows = cfg.get("block_entry_dows")
    if block_dows:
        # Use ind's date if available, else fall back to skipping
        # The backtester passes day_of_week via the call site; the engine
        # itself doesn't know the date — so we rely on cfg-driven full-day
        # skipping in the simulator. Keep this stub for live-side use.
        pass

    # ── IV percentile entry gate (block when options too expensive) ──
    if cfg.get("use_iv_pctile_gate") and iv_percentile > 0:
        gate = cfg.get("iv_pctile_gate_threshold", 80)
        if iv_percentile > gate:
            logger.debug("CONFLUENCE BLOCKED: IV percentile %.0f > %d (too expensive)", iv_percentile, gate)
            return False

    price = current_spot if current_spot > 0 else ind["close"]

    # VWAP filter
    if cfg["use_vwap_filter"]:
        vwap = ind.get("vwap", ind["close"])
        if action == "BUY_CALL" and price <= vwap:
            return False
        if action == "BUY_PUT" and price >= vwap:
            return False

    # StdDev filter
    if cfg.get("use_stddev_filter"):
        zscore = ind.get("price_zscore", 0)
        threshold = cfg.get("stddev_entry_threshold", 1.0)
        if abs(zscore) < threshold:
            return False

    # RSI hard gate
    if cfg["use_rsi_hard_gate"]:
        rsi = ind["rsi"]
        if action == "BUY_CALL" and rsi < cfg["rsi_call_min"]:
            return False
        if action == "BUY_PUT" and rsi > cfg["rsi_put_max"]:
            return False

    # RSI CALL kill zone
    if action == "BUY_CALL":
        rsi = ind["rsi"]
        if rsi > cfg.get("rsi_call_kill_ceiling", 60):
            return False

    # Squeeze filter
    if cfg["use_squeeze_filter"] and ind.get("squeeze_on", False):
        return False

    # Gap reversal filter
    if cfg.get("gap_reversal_filter") and prev_close > 0 and day_open > 0:
        gap_pct = (day_open - prev_close) / prev_close
        gap_threshold = cfg.get("gap_threshold_pct", 0.004)
        if abs(gap_pct) >= gap_threshold:
            if gap_pct < 0 and price > prev_close and action == "BUY_PUT":
                return False
            if gap_pct > 0 and price < prev_close and action == "BUY_CALL":
                return False

    # OI proximity filter
    if cfg.get("use_oi_levels") and oi_support > 0 and oi_resistance > 0:
        proximity = cfg.get("oi_proximity_pct", 0.003)
        if action == "BUY_CALL" and oi_resistance > 0:
            dist_to_resistance = (oi_resistance - price) / price
            if 0 < dist_to_resistance < proximity:
                return False
        if action == "BUY_PUT" and oi_support > 0:
            dist_to_support = (price - oi_support) / price
            if 0 < dist_to_support < proximity:
                return False

    # ── ROUND 5: Heikin Ashi trend agreement filter ──
    if cfg.get("use_ha_confluence"):
        if action == "BUY_CALL" and ind.get("ha_red_streak", 0) >= 3:
            return False  # Don't buy calls in confirmed HA downtrend
        if action == "BUY_PUT" and ind.get("ha_green_streak", 0) >= 3:
            return False  # Don't buy puts in confirmed HA uptrend

    # ── ROUND 5: KAMA slope agreement filter ──
    if cfg.get("use_kama_confluence"):
        if action == "BUY_CALL" and not ind.get("kama_slope_up", True):
            return False
        if action == "BUY_PUT" and ind.get("kama_slope_up", False):
            return False

    # ── ROUND 5: Parabolic SAR agreement filter ──
    if cfg.get("use_psar_confluence"):
        if action == "BUY_CALL" and not ind.get("psar_bullish", True):
            return False
        if action == "BUY_PUT" and ind.get("psar_bullish", True):
            return False

    # ── V15: Volume confirmation filter ──
    if cfg.get("use_volume_entry_filter"):
        vol_ratio = ind.get("volume_ratio", 1.0)
        if vol_ratio < cfg.get("min_volume_ratio", 0.7):
            return False  # Reject entries on weak volume

    # ── V15: Momentum agreement filter ──
    # Price velocity must agree with trade direction
    if cfg.get("use_velocity_filter"):
        velocity = ind.get("price_velocity", 0.0)
        min_vel = cfg.get("min_velocity_pct", 0.05)
        if action == "BUY_CALL" and velocity < -min_vel:
            return False  # Don't buy calls when price declining fast
        if action == "BUY_PUT" and velocity > min_vel:
            return False  # Don't buy puts when price rising fast

    # Block CALL in 4th hour
    if cfg.get("block_call_4th_hour") and action == "BUY_CALL":
        if 45 <= bar_idx < 60:
            return False

    # Late entry block
    if bar_idx > cfg.get("block_late_entries", 61):
        return False

    # Avoid windows
    for s, e in cfg.get("avoid_windows_bars", []):
        if s <= bar_idx < e:
            return False

    return True


# ─────────────────────────────────────────────────────────────
# EXIT EVALUATION
# ─────────────────────────────────────────────────────────────

def evaluate_exit(
    pos: dict,
    bar_idx: int,
    spot: float,
    ind: dict,
    cfg: dict,
    *,
    day_of_week: int = -1,
) -> Optional[str]:
    """Evaluate whether a position should be exited.

    Parameters
    ----------
    pos : dict
        Position dict with keys: action, entry_bar, entry_spot, best_fav,
        is_zero_hero.
    bar_idx : int
        Current bar index.
    spot : float
        Current spot price.
    ind : dict
        Current indicator snapshot (for ATR if needed).
    cfg : dict
        Config dict.
    day_of_week : int
        0=Monday, 1=Tuesday, etc. -1 = unknown.

    Returns
    -------
    str or None
        Exit reason string, or None to keep holding.
    """
    bars_held = bar_idx - pos["entry_bar"]
    if bars_held < 1:
        return None

    action = pos["action"]

    # ── ZERO-TO-HERO EXIT ──
    if pos.get("is_zero_hero", False):
        zh_trail = pos["entry_spot"] * cfg.get("zero_hero_trail_pct", 0.008)
        if action == "BUY_PUT":
            move = (pos["entry_spot"] - spot) / pos["entry_spot"]
        else:
            move = (spot - pos["entry_spot"]) / pos["entry_spot"]

        if move >= cfg.get("zero_hero_target_pct", 0.02):
            return "zero_hero_target"
        elif move <= -cfg.get("zero_hero_stop_pct", 0.008):
            return "zero_hero_stop"
        elif move >= cfg.get("zero_hero_trail_activation", 0.01):
            if action == "BUY_PUT" and spot > pos["best_fav"] + zh_trail:
                return "zero_hero_trail"
            elif action == "BUY_CALL" and spot < pos["best_fav"] - zh_trail:
                return "zero_hero_trail"
        elif bars_held >= cfg.get("zero_hero_time_bars", 30):
            return "zero_hero_time"
        # Z2H: no further exit checks — return None to keep holding
        return None

    # ── THETA-AWARE EXIT (Monday 3 PM for Tuesday-expiry protection) ──
    if cfg.get("theta_exit_enabled") and day_of_week == 0:
        theta_bar = cfg.get("theta_exit_monday_bar", 69)
        if bar_idx >= theta_bar:
            is_profitable = False
            if action == "BUY_PUT" and spot < pos["entry_spot"]:
                is_profitable = True
            elif action == "BUY_CALL" and spot > pos["entry_spot"]:
                is_profitable = True
            if is_profitable:
                return "theta_exit"

    # ── V15: STALE TRADE EXIT ──
    # If trade hasn't moved 0.3% in favor within N bars, exit
    if cfg.get("use_stale_exit"):
        stale_bars = cfg.get("stale_exit_bars", 9)  # 45 min
        stale_threshold = cfg.get("stale_exit_pct", 0.003)  # 0.3%
        if bars_held >= stale_bars:
            if action == "BUY_PUT":
                fav_move = (pos["entry_spot"] - spot) / pos["entry_spot"]
            else:
                fav_move = (spot - pos["entry_spot"]) / pos["entry_spot"]
            if fav_move < stale_threshold:
                return "stale_exit"

    # ── V15: CHANDELIER EXIT (ATR-based adaptive trail) ──
    if cfg.get("use_chandelier_exit_v15"):
        chandelier_mult = cfg.get("chandelier_atr_mult", 2.5)
        chandelier_min_bars = cfg.get("chandelier_min_bars", 6)  # 30 min min hold
        atr_val = ind.get("atr", 50)
        chandelier_dist = atr_val * chandelier_mult

        if bars_held >= chandelier_min_bars:
            if action == "BUY_PUT":
                # For puts: exit if price rises too far from best low
                chandelier_stop = pos["best_fav"] + chandelier_dist
                if spot > chandelier_stop:
                    return "chandelier_exit"
            else:
                # For calls: exit if price drops too far from best high
                chandelier_stop = pos["best_fav"] - chandelier_dist
                if spot < chandelier_stop:
                    return "chandelier_exit"

    # ── V15: MOMENTUM EXHAUSTION EXIT ──
    # Exit when RSI reaches extreme opposite to trade direction
    if cfg.get("use_momentum_exhaustion_exit"):
        rsi_val = ind.get("rsi", 50)
        if action == "BUY_CALL" and rsi_val > cfg.get("exhaustion_rsi_call", 75):
            # Call trade + RSI overbought = take profit
            if bars_held >= 3:
                return "momentum_exhaustion"
        elif action == "BUY_PUT" and rsi_val < cfg.get("exhaustion_rsi_put", 25):
            # Put trade + RSI oversold = take profit
            if bars_held >= 3:
                return "momentum_exhaustion"

    # ── TIME EXIT ──
    max_hold = cfg["max_hold_put"] if action == "BUY_PUT" else cfg["max_hold_call"]
    if bars_held >= max_hold:
        return "time_exit"

    # ── TRAILING STOP ──
    min_hold_trail = (cfg["min_hold_trail_put"] if action == "BUY_PUT"
                      else cfg["min_hold_trail_call"])
    if bars_held >= min_hold_trail:
        min_profit_move = pos["entry_spot"] * cfg.get("trail_min_profit_pct", 0.003)
        if action == "BUY_PUT":
            has_profit = pos["entry_spot"] - pos["best_fav"] >= min_profit_move
        else:
            has_profit = pos["best_fav"] - pos["entry_spot"] >= min_profit_move

        if has_profit:
            trail_pct = (cfg["trail_pct_put"] if action == "BUY_PUT"
                         else cfg["trail_pct_call"])
            trail_d = pos["entry_spot"] * trail_pct
            if action == "BUY_PUT" and spot > pos["best_fav"] + trail_d:
                return "trail_stop"
            elif action == "BUY_CALL" and spot < pos["best_fav"] - trail_d:
                return "trail_stop"

    # ── EOD CLOSE ──
    if bar_idx >= cfg.get("eod_close_bar", 72):
        return "eod_close"

    return None


# ─────────────────────────────────────────────────────────────
# LOT SIZING
# ─────────────────────────────────────────────────────────────

def compute_lots(
    action: str,
    conf: float,
    vix: float,
    rsi: float,
    is_expiry: bool,
    base_lots: int,
    cfg: dict,
    *,
    regime: str = "neutral",
    regime_call_mult: float = 1.0,
    regime_put_mult: float = 1.0,
    iv_percentile: float = 50.0,
    daily_loss_pct: float = 0.0,
) -> int:
    """Compute position size in lots.

    Parameters
    ----------
    base_lots : int
        Base lot count from capital allocation (capital * 0.70 / SPAN_per_lot).
    """
    combined_mult = 1.0

    # Direction bias
    if action == "BUY_PUT":
        combined_mult *= cfg["put_bias_lot_mult"]
    else:
        combined_mult *= cfg["call_bias_lot_mult"]

    # VIX regime
    if cfg["vix_danger_min"] <= vix <= cfg["vix_danger_max"]:
        combined_mult *= cfg["vix_danger_lot_mult"]
    elif cfg["vix_sweet_min"] <= vix <= cfg["vix_sweet_max"]:
        combined_mult *= cfg["vix_sweet_lot_mult"]

    # RSI zone
    if cfg["rsi_sweet_low"] <= rsi <= cfg["rsi_sweet_high"]:
        combined_mult *= cfg["rsi_sweet_lot_mult"]
    if cfg["rsi_danger_low"] <= rsi <= cfg["rsi_danger_high"]:
        combined_mult *= cfg["rsi_danger_lot_mult"]

    # Expiry day
    if is_expiry:
        combined_mult *= cfg["expiry_day_lot_mult"]

    # Regime
    if regime == "bullish" and action == "BUY_PUT":
        combined_mult *= regime_put_mult
    elif regime == "bearish" and action == "BUY_CALL":
        combined_mult *= regime_call_mult

    # Confidence boost
    if conf >= 0.50:
        combined_mult *= 1.3
    elif conf >= 0.40:
        combined_mult *= 1.15

    # IV percentile
    if cfg.get("use_iv_pctile_scaling") and iv_percentile > 0:
        if iv_percentile > cfg.get("iv_pctile_high", 75):
            combined_mult *= cfg.get("iv_pctile_high_mult", 0.7)
        elif iv_percentile < cfg.get("iv_pctile_low", 25):
            combined_mult *= cfg.get("iv_pctile_low_mult", 1.3)

    # Drawdown scaling
    if cfg.get("drawdown_lot_scale") and daily_loss_pct < 0:
        loss_used = abs(daily_loss_pct) / 0.03
        if loss_used >= 0.75:
            combined_mult *= 0.3
        elif loss_used >= 0.50:
            combined_mult *= 0.7

    # ── ROUND 5: Half-Kelly position sizing ──
    # f* = (bp - q) / b   where b = avg_win/avg_loss, p = win_rate, q = 1-p
    # Half-Kelly = 0.5 * f*  → ~75% of full Kelly growth, ~50% variance
    if cfg.get("use_half_kelly"):
        kelly_wr = cfg.get("kelly_win_rate", 0.45)
        kelly_rr = cfg.get("kelly_reward_risk", 1.5)
        kelly_f = (kelly_rr * kelly_wr - (1 - kelly_wr)) / kelly_rr
        half_kelly_f = max(0.05, kelly_f * 0.5)
        # Scale combined_mult by half-Kelly fraction
        combined_mult *= (half_kelly_f / 0.25)  # Normalize: 0.25 = "normal" fraction

    # ── ROUND 5: ATR-normalized sizing ──
    # Size inversely to ATR: high ATR → fewer lots, low ATR → more lots
    # Normalizes risk per trade regardless of volatility
    if cfg.get("use_atr_sizing"):
        atr_val = cfg.get("_current_atr", 0)
        atr_ref = cfg.get("atr_reference", 80)  # "Normal" ATR for NIFTY 5-min
        if atr_val > 0:
            atr_ratio = atr_ref / atr_val
            atr_ratio = max(0.5, min(2.0, atr_ratio))  # Clamp 0.5x-2.0x
            combined_mult *= atr_ratio

    # ── V15: Losing streak sizing ──
    # After consecutive losses, reduce lot size to limit drawdown
    if cfg.get("use_streak_sizing"):
        recent_losses = cfg.get("_recent_losses", 0)
        if recent_losses >= 3:
            streak_mult = max(0.4, 1.0 - recent_losses * 0.15)
            combined_mult *= streak_mult

    # ── V15: Equity curve sizing ──
    # When in significant drawdown, reduce exposure
    if cfg.get("use_equity_curve_sizing"):
        dd_pct = cfg.get("_current_dd_pct", 0.0)
        if dd_pct > 0.30:
            combined_mult *= 0.5
        elif dd_pct > 0.15:
            combined_mult *= 0.7

    # Floor
    combined_mult = max(0.5, combined_mult)

    lots = max(1, int(base_lots * combined_mult))
    return lots


# ─────────────────────────────────────────────────────────────
# COMPOSITE ENTRY DETECTION
# ─────────────────────────────────────────────────────────────

def detect_composite_entries(
    bar: dict,
    bar_idx: int,
    spot: float,
    vix: float,
    cfg: dict,
    *,
    prev_close: float = 0.0,
    gap_detected: bool = False,
    orb_high: float = 0.0,
    orb_low: float = 0.0,
    support: float = 0.0,
    resistance: float = 0.0,
    prev_spot: float = 0.0,
    market_bias: str = "neutral",
) -> list:
    """Detect composite entries (gap, ORB, S/R bounce, zero-to-hero).

    Returns list of (action, entry_type, confidence, is_zero_hero) tuples.
    """
    signals = []

    # ── 1. GAP ENTRY (bar 0) ──
    disable_gap = cfg.get("disable_gap_entry", False)
    if bar_idx == 0 and prev_close > 0 and not gap_detected:
        gap_pct = (spot - prev_close) / prev_close * 100

        if gap_pct < -0.3:
            is_large_gap = abs(gap_pct) > 1.2
            if not disable_gap:
                if is_large_gap:
                    conf = min(0.85, 0.65 + abs(gap_pct) * 0.05)
                    signals.append(("BUY_CALL", "gap_fade", conf, False))
                else:
                    conf = min(0.90, 0.60 + abs(gap_pct) * 0.10)
                    signals.append(("BUY_PUT", "gap_entry", conf, False))
            # Z2H gap entries always allowed
            if -1.2 <= gap_pct < -0.5 and vix >= 13:
                signals.append(("BUY_PUT", "gap_zero_hero", 0.70, True))
        elif gap_pct > 0.3:
            is_large_gap = gap_pct > 1.2
            if is_large_gap and not disable_gap:
                conf = min(0.85, 0.65 + gap_pct * 0.05)
                signals.append(("BUY_PUT", "gap_fade", conf, False))
            if gap_pct > 1.0 and vix >= 13:
                signals.append(("BUY_PUT", "gap_zero_hero", 0.65, True))

    # ── 2. ORB ENTRY (bars 1-2) ──
    if bar_idx in (1, 2) and orb_high > 0:
        orb_range = orb_high - orb_low
        if orb_range > spot * 0.0015:
            if spot > orb_high:
                conf = min(0.80, 0.55 + (spot - orb_high) / orb_high * 10)
                signals.append(("BUY_CALL", "orb_breakout_up", conf, False))
            elif spot < orb_low:
                conf = min(0.80, 0.55 + (orb_low - spot) / orb_low * 10)
                signals.append(("BUY_PUT", "orb_breakout_down", conf, False))

    # ── 3. S/R BOUNCE (bar 2+) ──
    if bar_idx >= 2 and support > 0 and resistance > 0:
        sr_dist = resistance - support
        if sr_dist >= 150 and prev_spot > 0:
            bias_lower = market_bias.lower() if market_bias else "neutral"
            if abs(spot - support) / spot < 0.003:
                if spot > prev_spot and bias_lower in ("bullish", "strong_bullish", "neutral",
                                                        "strongly_bullish"):
                    sr_conf = 0.65 if "bullish" in bias_lower else 0.55
                    signals.append(("BUY_CALL", "sr_bounce_support", sr_conf, False))
            if abs(spot - resistance) / spot < 0.003:
                if spot < prev_spot and "strong_bullish" not in bias_lower:
                    sr_conf = 0.75 if "bearish" in bias_lower else 0.70
                    signals.append(("BUY_PUT", "sr_bounce_resistance", sr_conf, False))

    return signals


# ─────────────────────────────────────────────────────────────
# V17 BTST FAVORABILITY — shared by live + backtest
# ─────────────────────────────────────────────────────────────

def v17_btst_favorable(
    cfg: dict,
    ind: dict,
    action: str,
    bar_idx: int,
    dte: float,
    vix: float,
    spot: float,
    day_high: float,
    day_low: float,
    day_open: float,
) -> bool:
    """Return True if market conditions favor carrying this trade overnight
    as BTST (NRML product) instead of MIS.

    Indicator-driven replacement for calendar-based (Friday-only) BTST.
    Works on ANY day of the week — decision is based on trend alignment,
    trend strength, RSI exhaustion, intraday closing strength, intraday
    bias, and VIX regime. This is the single source of truth used by
    BOTH the live agent and the backtester.

    Parameters
    ----------
    cfg : dict
        Config dict containing v17_btst_* thresholds and
        use_v17_dynamic_product flag.
    ind : dict
        Indicator snapshot (adx, trend_regime, rsi).
    action : str
        "BUY_PUT" or "BUY_CALL".
    bar_idx : int
        Current bar index within the day (5-min bars, 0 = 9:15 open).
    dte : float
        Days to expiry for the option.
    vix : float
        Current India VIX.
    spot : float
        Current NIFTY spot price.
    day_high / day_low / day_open : float
        Intraday stats for "close near HOD/LOD" and "day move" checks.

    Returns
    -------
    bool
        True if BTST carry is favorable (→ NRML), False otherwise (→ MIS).
    """
    if not cfg.get("use_v17_dynamic_product"):
        return False
    if not ind:
        return False

    # ── Hard gates ──
    if dte < cfg.get("v17_btst_dte_min", 2):
        return False  # need runway for theta to be worth gap risk
    if bar_idx < cfg.get("v17_btst_bar_min", 30):
        return False  # too early — need late-session confirmation

    adx = float(ind.get("adx", 0) or 0)
    trend_regime = int(ind.get("trend_regime", 0) or 0)
    rsi = float(ind.get("rsi", 50) or 50)
    want_trend = -1 if action == "BUY_PUT" else +1

    # ── 1. Trend regime must align with our direction ──
    if trend_regime != want_trend:
        return False

    # ── 2. Trend strength (ADX) ──
    if adx < cfg.get("v17_btst_adx_min", 18):
        return False

    # ── 3. Not already exhausted (mean-reversion risk) ──
    if action == "BUY_PUT":
        if rsi < cfg.get("v17_btst_rsi_put_min", 25):
            return False
    else:  # BUY_CALL
        if rsi > cfg.get("v17_btst_rsi_call_max", 75):
            return False

    # ── 4. Closing strength: price closing in the direction of the bet ──
    day_range = max(day_high - day_low, 0.01)
    close_pos = (spot - day_low) / day_range  # 0 = LOD, 1 = HOD
    if action == "BUY_PUT":
        if close_pos > cfg.get("v17_btst_close_put_max", 0.50):
            return False  # not near LOD → weak continuation
    else:
        if close_pos < cfg.get("v17_btst_close_call_min", 0.50):
            return False

    # ── 5. Intraday bias: the day itself must be moving our way ──
    day_chg = (spot - day_open) / day_open if day_open > 0 else 0.0
    if action == "BUY_PUT":
        if day_chg > cfg.get("v17_btst_day_chg_put_max", 0.005):
            return False  # day is green — don't carry PUT overnight
    else:
        if day_chg < cfg.get("v17_btst_day_chg_call_min", -0.005):
            return False  # day is red — don't carry CALL

    # ── 6. VIX absolute regime ──
    if vix < cfg.get("v17_btst_vix_min", 11):
        return False  # too quiet — gap-open unlikely
    if vix > cfg.get("v17_btst_vix_max", 30):
        return False  # too chaotic — gap direction unreliable

    return True
