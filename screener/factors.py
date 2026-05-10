"""
Factor library for equity screening.

Each factor function takes a pandas DataFrame with OHLCV columns
(Open, High, Low, Close, Volume) indexed by date, and returns a single
scalar score for the *latest* bar. Higher = more bullish.

These factors are well-documented in Indian equities literature:
    - Momentum (12-1): Jegadeesh & Titman; persistent in NSE since 2003
    - Short-term reversal: 1M ret negatively predictive
    - Low volatility: Defensive premium documented globally
    - Liquidity filter: Avoid microcap manipulation
    - Trend confirmation: Above 200DMA + 50DMA rising
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_ret(df: pd.DataFrame, lookback: int, skip: int = 0) -> float:
    """Return total return over lookback bars, skipping the most recent `skip` bars."""
    if len(df) < lookback + skip + 1:
        return np.nan
    end = df["Close"].iloc[-1 - skip]
    start = df["Close"].iloc[-1 - skip - lookback]
    if start <= 0 or np.isnan(start):
        return np.nan
    return (end / start) - 1.0


def momentum_12_1(df: pd.DataFrame) -> float:
    """
    12-month minus 1-month momentum: return from t-252d to t-21d.
    Skipping last month avoids short-term reversal contamination.
    """
    if len(df) < 253:
        return np.nan
    end = df["Close"].iloc[-21]
    start = df["Close"].iloc[-252]
    if start <= 0:
        return np.nan
    return (end / start) - 1.0


def short_term_reversal(df: pd.DataFrame) -> float:
    """
    1-month return. NEGATIVE values are bullish for the next month
    (short-term reversal). We return the raw value; ranker inverts.
    """
    return _safe_ret(df, lookback=21, skip=0)


def low_volatility(df: pd.DataFrame) -> float:
    """
    Negative annualized daily-return stdev over 252 days.
    Higher (less negative) = lower vol = better.
    """
    if len(df) < 252:
        return np.nan
    rets = df["Close"].pct_change().tail(252).dropna()
    if len(rets) < 200:
        return np.nan
    vol = rets.std() * np.sqrt(252)
    return -float(vol)


def trend_strength(df: pd.DataFrame) -> float:
    """
    Composite trend: distance above 200DMA + slope of 50DMA.
    Both positive = strong uptrend.
    """
    if len(df) < 220:
        return np.nan
    close = df["Close"]
    ma200 = close.rolling(200).mean().iloc[-1]
    ma50 = close.rolling(50).mean()
    if pd.isna(ma200) or ma200 <= 0:
        return np.nan
    dist_above_200 = (close.iloc[-1] / ma200) - 1.0
    # 50DMA slope = (ma50_today - ma50_20d_ago) / ma50_20d_ago
    if len(ma50.dropna()) < 21:
        slope = 0.0
    else:
        ma50_now = ma50.iloc[-1]
        ma50_prev = ma50.iloc[-21]
        if pd.isna(ma50_prev) or ma50_prev <= 0:
            slope = 0.0
        else:
            slope = (ma50_now / ma50_prev) - 1.0
    return float(dist_above_200 + slope)


def liquidity_score(df: pd.DataFrame) -> float:
    """
    Log of 20-day median ₹-turnover. Used as a filter, not a ranking factor.
    """
    if len(df) < 20:
        return np.nan
    turnover = (df["Close"] * df["Volume"]).tail(20).median()
    if turnover <= 0:
        return np.nan
    return float(np.log10(turnover))


def atr_pct(df: pd.DataFrame, period: int = 20) -> float:
    """
    Average True Range as % of price — used for stop-loss sizing.
    """
    if len(df) < period + 1:
        return np.nan
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(period).mean().iloc[-1]
    px = close.iloc[-1]
    if px <= 0 or pd.isna(atr):
        return np.nan
    return float(atr / px)


def gap_risk(df: pd.DataFrame, lookback: int = 60) -> float:
    """
    Frequency of |gap| > 3% over last `lookback` days.
    Lower = safer overnight risk profile. Returned as NEGATIVE so higher
    z-score = better.
    """
    if len(df) < lookback + 1:
        return np.nan
    open_ = df["Open"].tail(lookback)
    prev_close = df["Close"].shift(1).tail(lookback)
    gaps = ((open_ / prev_close) - 1.0).abs()
    return -float((gaps > 0.03).mean())


def compute_all_factors(df: pd.DataFrame) -> dict:
    """Compute every factor for a single stock's price history."""
    return {
        "momentum_12_1": momentum_12_1(df),
        "reversal_1m": short_term_reversal(df),
        "low_vol": low_volatility(df),
        "trend": trend_strength(df),
        "liquidity": liquidity_score(df),
        "atr_pct": atr_pct(df),
        "gap_risk": gap_risk(df),
        "last_close": float(df["Close"].iloc[-1]) if len(df) else np.nan,
        "bars": len(df),
    }
