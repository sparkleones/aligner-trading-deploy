"""
Strategy 7: Connors RSI Mean Reversion (Larry Connors 2009)

Buy oversold stocks in established uptrends. Filters:
  - 200-DMA: must be ABOVE (uptrend confirmed)
  - RSI(2): must be BELOW 5 (extreme short-term oversold)
  - Close: must be 2+ % below 5-day MA (further confirmation)

Documented in "Short Term Trading Strategies That Work" (Connors 2009).
Win rate ~70% historically with 1-5 day holds in US equities.

Score = oversold severity
      = (5MA - close) / 5MA   (positive when below 5MA)
      Higher score = more deeply oversold = stronger reversion edge.
      NaN if not in uptrend or RSI(2) not extreme.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import BaseStrategy


def _rsi(closes: pd.Series, period: int) -> float:
    """Simple RSI on the last `period+1` bars."""
    if len(closes) < period + 1:
        return np.nan
    delta = closes.diff().dropna().tail(period)
    if len(delta) < period:
        return np.nan
    gain = delta.clip(lower=0).mean()
    loss = -delta.clip(upper=0).mean()
    if loss == 0:
        return 100.0
    rs = gain / loss
    return float(100.0 - (100.0 / (1.0 + rs)))


class MeanReversionStrategy(BaseStrategy):
    def __init__(self):
        super().__init__(name="mean_reversion", needs_fundamentals=False)

    def score(self, symbol, history, fundamentals=None, asof=None):
        df = self._slice(history, asof)
        if len(df) < 210:
            return np.nan
        close = df["Close"]
        px = float(close.iloc[-1])
        if px <= 0:
            return np.nan
        # Uptrend filter
        ma200 = close.rolling(200).mean().iloc[-1]
        if pd.isna(ma200) or px < ma200:
            return np.nan
        # RSI(2) extreme oversold
        rsi2 = _rsi(close.tail(20), 2)
        if pd.isna(rsi2) or rsi2 >= 5:
            return np.nan
        # Deeply below 5-DMA
        ma5 = close.rolling(5).mean().iloc[-1]
        if pd.isna(ma5) or ma5 <= 0:
            return np.nan
        below_pct = (ma5 - px) / ma5
        if below_pct < 0.02:
            return np.nan
        # Score = how oversold (higher = more deeply below 5MA)
        return float(below_pct)
