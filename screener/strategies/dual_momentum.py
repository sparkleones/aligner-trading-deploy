"""
Strategy 3: Dual Momentum (Gary Antonacci 2014)

Combines RELATIVE momentum (stock vs peers) with ABSOLUTE momentum
(stock vs cash / risk-off filter). Position is taken only if BOTH
are positive.

Score = 12M return if (12M return > 0) else NaN

The 'absolute' filter (return > 0 over 12M) is what makes this defensive:
when broad market drops, the score gates to NaN and the stock is excluded.

Published edge in "Dual Momentum Investing" book — Antonacci showed
~+7% alpha vs simple momentum in 1974-2013 sample.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import BaseStrategy


class DualMomentumStrategy(BaseStrategy):
    def __init__(self):
        super().__init__(name="dual_momentum", needs_fundamentals=False)
        self.lookback = 252  # 12 months

    def score(self, symbol, history, fundamentals=None, asof=None):
        df = self._slice(history, asof)
        if len(df) < self.lookback + 1:
            return np.nan
        end = df["Close"].iloc[-1]
        start = df["Close"].iloc[-self.lookback]
        if start <= 0:
            return np.nan
        ret_12m = (end / start) - 1.0
        # Absolute momentum filter: only score if positive
        if ret_12m <= 0:
            return np.nan
        return float(ret_12m)
