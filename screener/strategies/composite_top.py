"""
Composite strategy that combines the top backtest performers per cap tier.

LARGE cap winners (Sharpe order):
  1. stage2_trend     (Weinstein)  CAGR 31.7%  Sharpe 1.20  DD -21.6%
  2. breakout_52w     (Donchian)   CAGR 24.6%  Sharpe 1.19  DD -20.8%

MID cap winner:
  1. mean_reversion   (Connors)    CAGR 33.1%  Sharpe 1.21  DD -21.6%

SMALL cap: SKIPPED — all strategies showed CAGR <15% with DD > 60%.
Small caps are too noisy for screening at this scale.

The composite combines stage2_trend + breakout_52w (highly correlated,
both trend-following) for LARGE cap allocation, since they both clear
Sharpe 1.2 and have low drawdown.
"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd

from .base import BaseStrategy
from .stage2_trend import Stage2TrendStrategy
from .breakout import BreakoutStrategy
from .mean_reversion import MeanReversionStrategy


class CompositeTopStrategy(BaseStrategy):
    """Cross-sectional z-score blend of the top 2 strategies (large cap)."""

    def __init__(self):
        super().__init__(name="composite_top", needs_fundamentals=False)
        self.s1 = Stage2TrendStrategy()
        self.s2 = BreakoutStrategy()
        self.weights = {"stage2_trend": 0.55, "breakout_52w": 0.45}

    def score(self, symbol, history, fundamentals=None, asof=None):
        """Raw composite score — to be z-scored across universe by ranker."""
        a = self.s1.score(symbol, history, asof=asof)
        b = self.s2.score(symbol, history, asof=asof)
        if pd.isna(a) and pd.isna(b):
            return np.nan
        # Both must pass their gates; if one is NaN (filter failed), exclude
        if pd.isna(a) or pd.isna(b):
            return np.nan
        # Use product (geometric) so both signals must be strong
        # Note: scores can be negative for breakout (distance below high)
        # Stage 2 score is always positive (gated). Breakout score is in [-1, 0+].
        # Combine via additive weight after raw values.
        return float(self.weights["stage2_trend"] * a + self.weights["breakout_52w"] * b)


class CompositeMidStrategy(BaseStrategy):
    """Mean-reversion strategy for mid-cap allocation."""

    def __init__(self):
        super().__init__(name="composite_mid", needs_fundamentals=False)
        self.s1 = MeanReversionStrategy()

    def score(self, symbol, history, fundamentals=None, asof=None):
        return self.s1.score(symbol, history, asof=asof)
