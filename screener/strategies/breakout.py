"""
Strategy 5: 52-Week Breakout (Donchian / Darvas Box)

When a stock makes a new 52-week high on rising volume, it tends to
continue higher (George/Hwang 2004: "The 52-Week High and Momentum
Investing"). The post-breakout drift is ~5-7% over the next 6 months
in their US sample, with similar effects documented in NSE.

Score = (current_close / 52week_high) - 1.0
        ranges from -1.0 (way below high) to 0.0 (at high)
        Boosted by volume confirmation:
        score = score + 0.05 if volume_today > 1.5 * avg_volume_20

Stocks scored near 0 are at the high (best). We invert sign so
higher score = better (i.e., closer to or at 52w high).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import BaseStrategy


class BreakoutStrategy(BaseStrategy):
    def __init__(self):
        super().__init__(name="breakout_52w", needs_fundamentals=False)

    def score(self, symbol, history, fundamentals=None, asof=None):
        df = self._slice(history, asof)
        if len(df) < 252:
            return np.nan
        close = df["Close"].iloc[-1]
        if close <= 0:
            return np.nan
        high_252 = df["High"].tail(252).max()
        if high_252 <= 0:
            return np.nan
        # Distance from 52w high: 0 = at high, negative = below
        dist_from_high = (close / high_252) - 1.0  # range [-1, 0]
        # Volume confirmation
        vol = df["Volume"]
        if len(vol) >= 21:
            today_vol = float(vol.iloc[-1])
            avg_vol = float(vol.tail(21).iloc[:-1].mean())
            vol_boost = 0.05 if (avg_vol > 0 and today_vol > 1.5 * avg_vol) else 0.0
        else:
            vol_boost = 0.0
        # Higher score = closer to 52w high (less negative). Stocks
        # at the high get score ~0, stocks 30% below get score -0.30.
        return float(dist_from_high + vol_boost)
