"""
Strategy 4: Weinstein Stage 2 (Stan Weinstein 1988)

Classic stage analysis: stocks oscillate between 4 stages.
  - Stage 1: Base (sideways after downtrend) — accumulate
  - Stage 2: Advance (uptrend, above rising 30-week MA) — BUY
  - Stage 3: Top (sideways after uptrend) — sell
  - Stage 4: Decline (downtrend, below falling 30-week MA) — avoid

Stage 2 criteria (we score with a continuous metric):
  1. Price above 30-week (150 trading day) MA
  2. 30-week MA itself is rising
  3. Price above 10-week (50 trading day) MA
  4. Higher highs and higher lows

Score = (px / ma_150) * (ma_150 / ma_150_prev) - 1
        when both are positive, else NaN.

Documented in "Secrets for Profiting in Bull and Bear Markets" (1988).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import BaseStrategy


class Stage2TrendStrategy(BaseStrategy):
    def __init__(self):
        super().__init__(name="stage2_trend", needs_fundamentals=False)

    def score(self, symbol, history, fundamentals=None, asof=None):
        df = self._slice(history, asof)
        if len(df) < 180:
            return np.nan
        close = df["Close"]
        ma_150 = close.rolling(150).mean().iloc[-1]
        ma_50 = close.rolling(50).mean().iloc[-1]
        ma_150_prev = close.rolling(150).mean().iloc[-22]  # ~1 month ago
        if pd.isna(ma_150) or pd.isna(ma_50) or pd.isna(ma_150_prev):
            return np.nan
        px = float(close.iloc[-1])
        if px <= 0 or ma_150 <= 0 or ma_150_prev <= 0:
            return np.nan
        # Stage 2 requires:
        if not (px > ma_150 and px > ma_50 and ma_150 > ma_150_prev):
            return np.nan
        dist_above_ma = (px / ma_150) - 1.0
        ma_slope = (ma_150 / ma_150_prev) - 1.0
        return float(dist_above_ma + ma_slope * 5)  # weight slope higher
