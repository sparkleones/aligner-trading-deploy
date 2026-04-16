"""Shared indicator computation — ONE source of truth for backtest + live.

Ported from v14_live_agent._compute_indicators() (the production code).
All indicators computed incrementally from a list of bar dicts.
Both the backtester and live agent call this same function.

Bar dict format:
    {"open": float, "high": float, "low": float, "close": float,
     "volume": float, "date": str, "time": str (optional)}

V14 Round 5 additions:
    - Connors RSI (3-component RSI for better mean-reversion)
    - KAMA (Kaufman Adaptive MA — adapts to noise vs trend)
    - Heikin Ashi trend detection (noise-reduced candle analysis)
    - Donchian Channel (20-period high/low breakout)
    - Parabolic SAR (trend direction confirmation)
    - RSI Bearish Divergence (complement to existing bullish divergence)
    - MACD Signal Line crossover (upgrade from histogram-only)
    - VWAP standard deviation bands (±1σ, ±2σ)
"""

from typing import Optional
import numpy as np


def _ema(data, period):
    """Compute EMA over an array."""
    if len(data) < period:
        return float(data[-1])
    mult = 2.0 / (period + 1)
    e = float(np.mean(data[:period]))
    for v in data[period:]:
        e = (float(v) - e) * mult + e
    return e


def _kama(data, period=10, fast_sc=2, slow_sc=30):
    """Kaufman Adaptive Moving Average.
    Adapts smoothing constant based on efficiency ratio (signal/noise).
    High ER (trending) → fast response. Low ER (choppy) → slow response.
    """
    if len(data) < period + 1:
        return float(data[-1])
    fast_alpha = 2.0 / (fast_sc + 1)
    slow_alpha = 2.0 / (slow_sc + 1)
    kama_val = float(data[period - 1])
    for i in range(period, len(data)):
        direction = abs(float(data[i]) - float(data[i - period]))
        volatility = sum(abs(float(data[j]) - float(data[j - 1]))
                         for j in range(i - period + 1, i + 1))
        if volatility == 0:
            er = 0.0
        else:
            er = direction / volatility
        sc = (er * (fast_alpha - slow_alpha) + slow_alpha) ** 2
        kama_val = kama_val + sc * (float(data[i]) - kama_val)
    return kama_val


def _rsi_series(closes_arr, period=14):
    """Compute RSI value for the last element of a close array."""
    if len(closes_arr) < period + 1:
        return 50.0
    deltas = np.diff(closes_arr[-(period + 1):])
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains) + 1e-10
    avg_loss = np.mean(losses) + 1e-10
    return float(100 - (100 / (1 + avg_gain / avg_loss)))


def _streak(closes_arr):
    """Count consecutive up/down streak. Positive = up, negative = down."""
    if len(closes_arr) < 2:
        return 0
    streak_val = 0
    for i in range(len(closes_arr) - 1, 0, -1):
        if closes_arr[i] > closes_arr[i - 1]:
            if streak_val >= 0:
                streak_val += 1
            else:
                break
        elif closes_arr[i] < closes_arr[i - 1]:
            if streak_val <= 0:
                streak_val -= 1
            else:
                break
        else:
            break
    return streak_val


def _pct_rank(value, lookback_arr):
    """Percentile rank of value within lookback array."""
    if len(lookback_arr) == 0:
        return 50.0
    count_below = np.sum(lookback_arr < value)
    return float(count_below / len(lookback_arr) * 100)


def compute_indicators(bars: list, today_date: str = "") -> Optional[dict]:
    """Compute all V14 indicators from bar history.

    Parameters
    ----------
    bars : list[dict]
        List of OHLCV bar dicts. Must have at least 15 bars.
    today_date : str
        Date string (YYYY-MM-DD) for VWAP anchoring. If empty, uses all bars.

    Returns
    -------
    dict or None
        Indicator snapshot dict, or None if insufficient bars.
    """
    n = len(bars)
    if n < 15:
        return None

    closes = np.array([b["close"] for b in bars], dtype=np.float64)
    highs = np.array([b["high"] for b in bars], dtype=np.float64)
    lows = np.array([b["low"] for b in bars], dtype=np.float64)

    ind = {"close": float(closes[-1])}

    # ── RSI (14-period) — Wilder's smoothed ──
    deltas = np.diff(closes[-15:])
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains) + 1e-10
    avg_loss = np.mean(losses) + 1e-10
    ind["rsi"] = float(100 - (100 / (1 + avg_gain / avg_loss)))

    # ── EMAs ──
    ind["ema9"] = _ema(closes, 9)
    ind["ema21"] = _ema(closes, 21)
    ind["ema9_above_ema21"] = ind["ema9"] > ind["ema21"]

    # ── Long-term trend regime (EMA50) ──
    # Used by trend-regime gate in passes_confluence to suppress
    # counter-trend entries (e.g. CALLs in confirmed downtrend).
    ind["ema50"] = _ema(closes, 50) if n >= 50 else float(closes[-1])
    if n >= 60:
        ema50_prev = _ema(closes[:-10], 50)
        ind["ema50_slope_up"] = ind["ema50"] > ema50_prev
    else:
        ind["ema50_slope_up"] = ind["close"] > ind["ema50"]
    ind["trend_regime"] = (
        1 if ind["close"] > ind["ema50"] and ind["ema50_slope_up"]
        else -1 if ind["close"] < ind["ema50"] and not ind["ema50_slope_up"]
        else 0
    )

    # ── Bollinger Bands (20, 2) ──
    if n >= 20:
        bb_slice = closes[-20:]
        bb_mid = float(np.mean(bb_slice))
        bb_std = float(np.std(bb_slice, ddof=1))
        ind["bb_upper"] = bb_mid + 2 * bb_std
        ind["bb_lower"] = bb_mid - 2 * bb_std
    else:
        ind["bb_upper"] = ind["bb_lower"] = float(closes[-1])

    # ── ATR (14-period) — TRUE RANGE with EWM smoothing ──
    period_atr = 14
    lookback = min(n, period_atr + 1)
    tr_vals = []
    for i in range(1, lookback):
        idx = n - lookback + i
        tr = max(
            highs[idx] - lows[idx],
            abs(highs[idx] - closes[idx - 1]),
            abs(lows[idx] - closes[idx - 1]),
        )
        tr_vals.append(float(tr))
    if tr_vals:
        atr_val = tr_vals[0]
        alpha = 1.0 / period_atr
        for tr in tr_vals[1:]:
            atr_val = atr_val * (1 - alpha) + tr * alpha
        ind["atr"] = atr_val
    else:
        ind["atr"] = 50.0

    # ── Keltner / Squeeze ──
    kc_upper = ind["ema21"] + 1.5 * ind["atr"]
    kc_lower = ind["ema21"] - 1.5 * ind["atr"]
    ind["squeeze_on"] = (ind["bb_lower"] > kc_lower) and (ind["bb_upper"] < kc_upper)

    # ── VWAP — today-only anchored ──
    today_tp = []
    today_vol = []
    for b in bars:
        bar_time = str(b.get("time", b.get("timestamp", b.get("date", ""))))
        if today_date and today_date in bar_time:
            tp_val = (b["high"] + b["low"] + b["close"]) / 3.0
            vol = b.get("volume", 0)
            if vol <= 0:
                vol = b["high"] - b["low"] + 1.0  # Range as volume proxy
            today_tp.append(tp_val)
            today_vol.append(vol)
    if today_tp:
        tp_arr = np.array(today_tp)
        vol_arr = np.array(today_vol)
        ind["vwap"] = float(np.sum(tp_arr * vol_arr) / (np.sum(vol_arr) + 1e-10))
    else:
        tp = (highs + lows + closes) / 3.0
        ind["vwap"] = float(np.mean(tp))

    # ── SUPERTREND — stateful with band ratcheting ──
    st_mult = 3.0
    st_period = 10
    if n >= st_period + 1:
        st_atr = ind["atr"]
        st_dir = 1
        upper_band = 0.0
        lower_band = 0.0
        for i in range(max(0, n - 50), n):
            hl2 = (highs[i] + lows[i]) / 2.0
            basic_upper = hl2 + st_mult * st_atr
            basic_lower = hl2 - st_mult * st_atr
            if i > 0:
                if basic_lower > lower_band or closes[i - 1] < lower_band:
                    lower_band = basic_lower
                if basic_upper < upper_band or closes[i - 1] > upper_band:
                    upper_band = basic_upper
            else:
                upper_band = basic_upper
                lower_band = basic_lower
            if st_dir == 1:
                if closes[i] < lower_band:
                    st_dir = -1
            else:
                if closes[i] > upper_band:
                    st_dir = 1
        ind["st_direction"] = st_dir
    else:
        ind["st_direction"] = 1 if closes[-1] > closes[-2] else -1

    # ── ADX — trend strength ──
    adx_period = 14
    if n >= adx_period + 2:
        plus_dm = []
        minus_dm = []
        for i in range(n - adx_period - 1, n):
            up_move = highs[i] - highs[i - 1]
            down_move = lows[i - 1] - lows[i]
            plus_dm.append(max(float(up_move), 0) if up_move > down_move else 0)
            minus_dm.append(max(float(down_move), 0) if down_move > up_move else 0)
        atr_sum = ind["atr"] * adx_period
        if atr_sum > 0:
            plus_di = 100 * np.mean(plus_dm) / (atr_sum / adx_period + 1e-10)
            minus_di = 100 * np.mean(minus_dm) / (atr_sum / adx_period + 1e-10)
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
            ind["adx"] = float(dx)
            ind["plus_di"] = float(plus_di)
            ind["minus_di"] = float(minus_di)
        else:
            ind["adx"] = 25.0
            ind["plus_di"] = 0.0
            ind["minus_di"] = 0.0
    else:
        ind["adx"] = 25.0
        ind["plus_di"] = 0.0
        ind["minus_di"] = 0.0

    # ── STOCHASTIC RSI (14, 14, 3, 3) ──
    stoch_need = 14 + 14 + 3
    if n >= stoch_need:
        compute_len = min(n, stoch_need + 5)
        rsi_series = []
        for j in range(n - compute_len, n):
            if j < 14:
                rsi_series.append(50.0)
                continue
            d = np.diff(closes[j - 14:j + 1])
            g = np.where(d > 0, d, 0)
            l_vals = np.where(d < 0, -d, 0)
            ag = np.mean(g) + 1e-10
            al = np.mean(l_vals) + 1e-10
            rsi_series.append(100 - 100 / (1 + ag / al))

        rsi_arr = np.array(rsi_series)
        stoch_raw = []
        for j in range(14, len(rsi_arr)):
            window = rsi_arr[j - 13:j + 1]
            rmin, rmax = np.min(window), np.max(window)
            stoch_raw.append((rsi_arr[j] - rmin) / (rmax - rmin + 1e-10) * 100)

        if len(stoch_raw) >= 6:
            k_vals = [np.mean(stoch_raw[max(0, j - 2):j + 1]) for j in range(len(stoch_raw))]
            d_vals = [np.mean(k_vals[max(0, j - 2):j + 1]) for j in range(len(k_vals))]
            ind["stoch_k"] = float(k_vals[-1])
            ind["stoch_d"] = float(d_vals[-1])
            ind["stoch_oversold"] = k_vals[-1] < 20
            ind["stoch_overbought"] = k_vals[-1] > 80
            ind["stoch_cross_up"] = (
                len(k_vals) >= 2 and len(d_vals) >= 2
                and k_vals[-1] > d_vals[-1] and k_vals[-2] <= d_vals[-2]
            )
            ind["stoch_cross_down"] = (
                len(k_vals) >= 2 and len(d_vals) >= 2
                and k_vals[-1] < d_vals[-1] and k_vals[-2] >= d_vals[-2]
            )
        else:
            ind["stoch_k"] = 50.0
            ind["stoch_d"] = 50.0
            ind["stoch_oversold"] = False
            ind["stoch_overbought"] = False
            ind["stoch_cross_up"] = False
            ind["stoch_cross_down"] = False
    else:
        ind["stoch_k"] = 50.0
        ind["stoch_d"] = 50.0
        ind["stoch_oversold"] = False
        ind["stoch_overbought"] = False
        ind["stoch_cross_up"] = False
        ind["stoch_cross_down"] = False

    # ── MACD (12, 26) ──
    if n >= 26:
        ema12 = _ema(closes, 12)
        ema26 = _ema(closes, 26)
        ind["macd_hist"] = ema12 - ema26
    else:
        ind["macd_hist"] = 0.0

    # ── CCI (14-period) ──
    cci_period = 14
    if n >= cci_period:
        tp_series = (highs[-cci_period:] + lows[-cci_period:] + closes[-cci_period:]) / 3.0
        tp_sma = np.mean(tp_series)
        mean_dev = np.mean(np.abs(tp_series - tp_sma))
        ind["cci"] = float((tp_series[-1] - tp_sma) / (0.015 * mean_dev + 1e-10))
    else:
        ind["cci"] = 0.0

    # ── Williams %R (14-period) ──
    wr_period = 14
    if n >= wr_period:
        highest_high = float(np.max(highs[-wr_period:]))
        lowest_low = float(np.min(lows[-wr_period:]))
        ind["williams_r"] = (highest_high - closes[-1]) / (highest_high - lowest_low + 1e-10) * -100
    else:
        ind["williams_r"] = -50.0

    # ── RSI Bullish Divergence ──
    div_lookback = 20
    if n >= div_lookback:
        lookback_closes = closes[-div_lookback:]
        rsi_lookback = []
        for j in range(n - div_lookback, n):
            if j < 14:
                rsi_lookback.append(50.0)
                continue
            d = np.diff(closes[j - 14:j + 1])
            g = np.where(d > 0, d, 0)
            l_vals = np.where(d < 0, -d, 0)
            ag = np.mean(g) + 1e-10
            al = np.mean(l_vals) + 1e-10
            rsi_lookback.append(100 - 100 / (1 + ag / al))
        rsi_lookback = np.array(rsi_lookback)
        price_lower_low = closes[-1] < np.min(lookback_closes[:-1])
        rsi_higher_low = ind["rsi"] > np.min(rsi_lookback[:-1])
        ind["rsi_bullish_divergence"] = bool(price_lower_low and rsi_higher_low)
    else:
        ind["rsi_bullish_divergence"] = False

    # ── Price Z-score (today only) ──
    today_closes = []
    for b in bars:
        bar_time = str(b.get("time", b.get("timestamp", b.get("date", ""))))
        if today_date and today_date in bar_time:
            today_closes.append(b["close"])
    if len(today_closes) >= 5:
        tc = np.array(today_closes)
        daily_mean = float(np.mean(tc))
        daily_std = float(np.std(tc, ddof=1))
        if daily_std > 0:
            ind["price_zscore"] = (closes[-1] - daily_mean) / daily_std
        else:
            ind["price_zscore"] = 0.0
    else:
        ind["price_zscore"] = 0.0

    # ══════════════════════════════════════════════════════════
    # ROUND 5: NEW ALGO TRADING INDICATORS
    # ══════════════════════════════════════════════════════════

    # ── CONNORS RSI (RSI + Up/Down Streak RSI + Rate of Change Rank) ──
    # 3-component composite RSI: standard RSI(3) + RSI(streak, 2) + %Rank(ROC, 100)
    # Widely used for mean-reversion timing; values <10 = oversold, >90 = overbought
    if n >= 20:
        crsi_rsi3 = _rsi_series(closes, 3)
        streak_val = _streak(closes[-20:])
        # Map streak to a small series then compute RSI of it
        streak_series = []
        for j in range(max(0, n - 10), n):
            streak_series.append(float(_streak(closes[max(0, j - 15):j + 1])))
        if len(streak_series) >= 3:
            streak_arr = np.array(streak_series, dtype=np.float64)
            crsi_streak_rsi = _rsi_series(streak_arr, 2)
        else:
            crsi_streak_rsi = 50.0
        # ROC percentile rank (100-bar lookback)
        roc_lookback = min(n, 100)
        if roc_lookback >= 2:
            roc_vals = np.diff(closes[-roc_lookback:]) / (closes[-roc_lookback:-1] + 1e-10) * 100
            current_roc = float(roc_vals[-1]) if len(roc_vals) > 0 else 0.0
            crsi_pct_rank = _pct_rank(current_roc, roc_vals[:-1]) if len(roc_vals) > 1 else 50.0
        else:
            crsi_pct_rank = 50.0
        ind["connors_rsi"] = (crsi_rsi3 + crsi_streak_rsi + crsi_pct_rank) / 3.0
    else:
        ind["connors_rsi"] = 50.0

    # ── KAMA (Kaufman Adaptive Moving Average, 10-period) ──
    ind["kama"] = _kama(closes, 10)
    ind["kama_above_price"] = ind["kama"] > closes[-1]
    ind["kama_slope_up"] = False
    if n >= 12:
        kama_prev = _kama(closes[:-1], 10)
        ind["kama_slope_up"] = ind["kama"] > kama_prev

    # ── HEIKIN ASHI trend detection ──
    # Compute last 5 HA candles, check if 3+ consecutive green/red
    opens = np.array([b["open"] for b in bars], dtype=np.float64)
    ha_lookback = min(n, 10)
    if ha_lookback >= 3:
        ha_close = (opens[-ha_lookback:] + highs[-ha_lookback:] +
                    lows[-ha_lookback:] + closes[-ha_lookback:]) / 4.0
        ha_open = np.zeros(ha_lookback)
        ha_open[0] = (opens[-ha_lookback] + closes[-ha_lookback]) / 2.0
        for j in range(1, ha_lookback):
            ha_open[j] = (ha_open[j - 1] + ha_close[j - 1]) / 2.0
        # Count consecutive green/red HA candles from the end
        ha_green_streak = 0
        ha_red_streak = 0
        for j in range(ha_lookback - 1, -1, -1):
            if ha_close[j] > ha_open[j]:
                if ha_red_streak == 0:
                    ha_green_streak += 1
                else:
                    break
            elif ha_close[j] < ha_open[j]:
                if ha_green_streak == 0:
                    ha_red_streak += 1
                else:
                    break
            else:
                break
        ind["ha_green_streak"] = ha_green_streak
        ind["ha_red_streak"] = ha_red_streak
        ind["ha_bullish"] = ha_green_streak >= 3
        ind["ha_bearish"] = ha_red_streak >= 3
        # HA candle body ratio (no wick = strong trend)
        last_ha_body = abs(ha_close[-1] - ha_open[-1])
        last_ha_range = max(highs[-1], ha_close[-1], ha_open[-1]) - min(lows[-1], ha_close[-1], ha_open[-1])
        ind["ha_body_ratio"] = last_ha_body / (last_ha_range + 1e-10)
    else:
        ind["ha_green_streak"] = 0
        ind["ha_red_streak"] = 0
        ind["ha_bullish"] = False
        ind["ha_bearish"] = False
        ind["ha_body_ratio"] = 0.5

    # ── DONCHIAN CHANNEL (20-period) ──
    dc_period = 20
    if n >= dc_period:
        ind["donchian_high"] = float(np.max(highs[-dc_period:]))
        ind["donchian_low"] = float(np.min(lows[-dc_period:]))
        ind["donchian_mid"] = (ind["donchian_high"] + ind["donchian_low"]) / 2.0
        ind["donchian_breakout_up"] = closes[-1] >= ind["donchian_high"]
        ind["donchian_breakout_down"] = closes[-1] <= ind["donchian_low"]
        # Width as % of price (for squeeze detection)
        ind["donchian_width_pct"] = (ind["donchian_high"] - ind["donchian_low"]) / (closes[-1] + 1e-10) * 100
    else:
        ind["donchian_high"] = float(closes[-1])
        ind["donchian_low"] = float(closes[-1])
        ind["donchian_mid"] = float(closes[-1])
        ind["donchian_breakout_up"] = False
        ind["donchian_breakout_down"] = False
        ind["donchian_width_pct"] = 0.0

    # ── PARABOLIC SAR ──
    # Standard SAR: AF starts 0.02, step 0.02, max 0.20
    sar_af_start = 0.02
    sar_af_step = 0.02
    sar_af_max = 0.20
    if n >= 5:
        sar_val = float(lows[-5])  # Start below price (assume uptrend)
        sar_bullish = True
        ep = float(highs[-5])  # Extreme point
        af = sar_af_start
        for j in range(n - 4, n):
            if sar_bullish:
                sar_val = sar_val + af * (ep - sar_val)
                sar_val = min(sar_val, float(lows[j - 1]), float(lows[j - 2]) if j >= 2 else float(lows[j - 1]))
                if float(highs[j]) > ep:
                    ep = float(highs[j])
                    af = min(af + sar_af_step, sar_af_max)
                if float(lows[j]) < sar_val:
                    sar_bullish = False
                    sar_val = ep
                    ep = float(lows[j])
                    af = sar_af_start
            else:
                sar_val = sar_val + af * (ep - sar_val)
                sar_val = max(sar_val, float(highs[j - 1]), float(highs[j - 2]) if j >= 2 else float(highs[j - 1]))
                if float(lows[j]) < ep:
                    ep = float(lows[j])
                    af = min(af + sar_af_step, sar_af_max)
                if float(highs[j]) > sar_val:
                    sar_bullish = True
                    sar_val = ep
                    ep = float(highs[j])
                    af = sar_af_start
        ind["psar"] = sar_val
        ind["psar_bullish"] = sar_bullish
    else:
        ind["psar"] = float(closes[-1])
        ind["psar_bullish"] = True

    # ── RSI BEARISH DIVERGENCE (complement to existing bullish) ──
    if n >= div_lookback:
        lookback_closes_b = closes[-div_lookback:]
        rsi_lookback_b = []
        for j in range(n - div_lookback, n):
            if j < 14:
                rsi_lookback_b.append(50.0)
                continue
            d = np.diff(closes[j - 14:j + 1])
            g = np.where(d > 0, d, 0)
            l_vals = np.where(d < 0, -d, 0)
            ag = np.mean(g) + 1e-10
            al = np.mean(l_vals) + 1e-10
            rsi_lookback_b.append(100 - 100 / (1 + ag / al))
        rsi_lookback_b = np.array(rsi_lookback_b)
        price_higher_high = closes[-1] > np.max(lookback_closes_b[:-1])
        rsi_lower_high = ind["rsi"] < np.max(rsi_lookback_b[:-1])
        ind["rsi_bearish_divergence"] = bool(price_higher_high and rsi_lower_high)
    else:
        ind["rsi_bearish_divergence"] = False

    # ── MACD SIGNAL LINE CROSSOVER ──
    if n >= 35:
        # Need enough bars for MACD(12,26) + Signal(9)
        macd_series = []
        for j in range(max(0, n - 15), n):
            e12 = _ema(closes[:j + 1], 12)
            e26 = _ema(closes[:j + 1], 26)
            macd_series.append(e12 - e26)
        if len(macd_series) >= 9:
            signal_line = _ema(np.array(macd_series), 9)
            ind["macd_signal"] = signal_line
            ind["macd_above_signal"] = macd_series[-1] > signal_line
            ind["macd_cross_up"] = (len(macd_series) >= 2 and
                                     macd_series[-1] > signal_line and
                                     macd_series[-2] <= signal_line)
            ind["macd_cross_down"] = (len(macd_series) >= 2 and
                                       macd_series[-1] < signal_line and
                                       macd_series[-2] >= signal_line)
        else:
            ind["macd_signal"] = 0.0
            ind["macd_above_signal"] = False
            ind["macd_cross_up"] = False
            ind["macd_cross_down"] = False
    else:
        ind["macd_signal"] = 0.0
        ind["macd_above_signal"] = False
        ind["macd_cross_up"] = False
        ind["macd_cross_down"] = False

    # ── VWAP STANDARD DEVIATION BANDS ──
    if today_tp and len(today_tp) >= 3:
        tp_arr_v = np.array(today_tp)
        vol_arr_v = np.array(today_vol)
        cum_vol = np.cumsum(vol_arr_v)
        cum_tpv = np.cumsum(tp_arr_v * vol_arr_v)
        vwap_val = cum_tpv[-1] / (cum_vol[-1] + 1e-10)
        # Variance around VWAP
        vwap_var = np.sum(vol_arr_v * (tp_arr_v - vwap_val) ** 2) / (cum_vol[-1] + 1e-10)
        vwap_std = float(np.sqrt(vwap_var))
        ind["vwap_upper1"] = vwap_val + vwap_std
        ind["vwap_lower1"] = vwap_val - vwap_std
        ind["vwap_upper2"] = vwap_val + 2 * vwap_std
        ind["vwap_lower2"] = vwap_val - 2 * vwap_std
        ind["vwap_band_dist"] = (closes[-1] - vwap_val) / (vwap_std + 1e-10)
    else:
        ind["vwap_upper1"] = ind.get("vwap", closes[-1])
        ind["vwap_lower1"] = ind.get("vwap", closes[-1])
        ind["vwap_upper2"] = ind.get("vwap", closes[-1])
        ind["vwap_lower2"] = ind.get("vwap", closes[-1])
        ind["vwap_band_dist"] = 0.0

    # ══════════════════════════════════════════════════════════
    # V15: ENHANCED INDICATORS
    # ══════════════════════════════════════════════════════════

    # ── Volume Ratio (current bar vol / 20-bar avg vol) ──
    volumes = np.array([b.get("volume", 0) for b in bars], dtype=np.float64)
    vol_period = min(n, 20)
    if vol_period >= 3 and volumes[-1] > 0:
        avg_vol = float(np.mean(volumes[-vol_period:-1])) if vol_period > 1 else float(volumes[-1])
        ind["volume_ratio"] = float(volumes[-1]) / (avg_vol + 1e-10)
    else:
        ind["volume_ratio"] = 1.0

    # ── Volume spike (> 1.5x average) ──
    ind["volume_spike"] = ind["volume_ratio"] > 1.5

    # ── OBV (On Balance Volume) trend ──
    # OBV direction: rising = accumulation (bullish), falling = distribution
    if n >= 10:
        obv = 0.0
        obv_series = []
        for j in range(max(0, n - 20), n):
            if j > 0:
                if closes[j] > closes[j - 1]:
                    obv += volumes[j]
                elif closes[j] < closes[j - 1]:
                    obv -= volumes[j]
            obv_series.append(obv)
        if len(obv_series) >= 5:
            obv_ema5 = _ema(np.array(obv_series), 5)
            ind["obv_rising"] = obv_series[-1] > obv_ema5
            ind["obv_falling"] = obv_series[-1] < obv_ema5
        else:
            ind["obv_rising"] = False
            ind["obv_falling"] = False
    else:
        ind["obv_rising"] = False
        ind["obv_falling"] = False

    # ── Momentum Acceleration (2nd derivative of price) ──
    # Positive = accelerating momentum, Negative = decelerating
    if n >= 6:
        roc1 = closes[-1] - closes[-3]   # 2-bar momentum (recent)
        roc2 = closes[-3] - closes[-5]   # 2-bar momentum (prior)
        ind["momentum_accel"] = float(roc1 - roc2)   # 2nd derivative
        ind["momentum_accel_up"] = roc1 > roc2 and roc1 > 0
        ind["momentum_accel_down"] = roc1 < roc2 and roc1 < 0
    else:
        ind["momentum_accel"] = 0.0
        ind["momentum_accel_up"] = False
        ind["momentum_accel_down"] = False

    # ── Price Velocity (5-bar ROC in %) ──
    if n >= 6:
        ind["price_velocity"] = float((closes[-1] - closes[-6]) / (closes[-6] + 1e-10) * 100)
    else:
        ind["price_velocity"] = 0.0

    # ── Realized Volatility vs ATR (proxy for RV/IV) ──
    # High ratio = market moving more than expected → option buyers have edge
    if n >= 20:
        returns_20 = np.diff(closes[-21:]) / (closes[-21:-1] + 1e-10)
        rv_20 = float(np.std(returns_20) * np.sqrt(75))  # Annualized intraday
        ind["realized_vol"] = rv_20
        # Compare to ATR-implied vol
        atr_implied = ind["atr"] / (closes[-1] + 1e-10) * np.sqrt(75)
        ind["rv_iv_ratio"] = rv_20 / (atr_implied + 1e-10)
    else:
        ind["realized_vol"] = 0.0
        ind["rv_iv_ratio"] = 1.0

    # ── Multi-timeframe EMA agreement (5, 9, 21 all aligned) ──
    if n >= 21:
        ema5 = _ema(closes, 5)
        ind["ema5"] = ema5
        ind["ema_all_bullish"] = ema5 > ind["ema9"] > ind["ema21"]
        ind["ema_all_bearish"] = ema5 < ind["ema9"] < ind["ema21"]
    else:
        ind["ema5"] = float(closes[-1])
        ind["ema_all_bullish"] = False
        ind["ema_all_bearish"] = False

    # ══════════════════════════════════════════════════════════
    # V16: REGIME DETECTION INDICATORS
    # ══════════════════════════════════════════════════════════

    # ── Bollinger Band Width (volatility regime proxy) ──
    bb_mid = (ind["bb_upper"] + ind["bb_lower"]) / 2.0
    if bb_mid > 0:
        ind["bb_width"] = (ind["bb_upper"] - ind["bb_lower"]) / bb_mid
    else:
        ind["bb_width"] = 0.0

    # ── BB Width percentile (rolling 60-bar = ~5 hours) ──
    if n >= 60:
        bb_widths = []
        for i in range(max(0, n - 60), n):
            _slice = closes[max(0, i-19):i+1]
            if len(_slice) >= 5:
                _mid = float(np.mean(_slice))
                _std = float(np.std(_slice, ddof=1)) if len(_slice) > 1 else 0
                _w = (4 * _std) / (_mid + 1e-10)
                bb_widths.append(_w)
        if bb_widths:
            current_bb_w = ind["bb_width"]
            ind["bb_width_pctile"] = float(
                sum(1 for w in bb_widths if w <= current_bb_w) / len(bb_widths) * 100
            )
        else:
            ind["bb_width_pctile"] = 50.0
    else:
        ind["bb_width_pctile"] = 50.0

    # ── OBV divergence detection (3-bar) ──
    if n >= 5:
        ph_hh = highs[-1] > max(highs[-4:-1])   # price higher high
        obv_arr_recent = []
        obv_val = 0.0
        for i in range(max(0, n - 5), n):
            if i > 0:
                if closes[i] > closes[i-1]:
                    obv_val += volumes[i]
                elif closes[i] < closes[i-1]:
                    obv_val -= volumes[i]
            obv_arr_recent.append(obv_val)
        if len(obv_arr_recent) >= 4:
            obv_hh = obv_arr_recent[-1] > max(obv_arr_recent[-4:-1])
            obv_lh = obv_arr_recent[-1] < max(obv_arr_recent[-4:-1])
            pl_ll = lows[-1] < min(lows[-4:-1])   # price lower low
            obv_hl = obv_arr_recent[-1] > min(obv_arr_recent[-4:-1])
            ind["obv_bearish_div"] = bool(ph_hh and obv_lh)   # price HH, OBV LH
            ind["obv_bullish_div"] = bool(pl_ll and obv_hl)   # price LL, OBV HL
        else:
            ind["obv_bearish_div"] = False
            ind["obv_bullish_div"] = False
    else:
        ind["obv_bearish_div"] = False
        ind["obv_bullish_div"] = False

    # ── Volume climax (2.5x average) ──
    vol_ratio = ind.get("volume_ratio", 1.0)
    ind["volume_climax"] = vol_ratio > 2.5

    return ind
