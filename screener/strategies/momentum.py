"""
Strategy 1: 12-1 Momentum (Jegadeesh & Titman 1993)

Edge: Stocks that outperformed over the prior 12 months (excluding the
most recent month to avoid short-term reversal) tend to continue
outperforming over the next 1-12 months.

Score = return from t-252 to t-21.

Validated in NSE since 2003 — Mukherji et al. 2008 found ~+0.8%/month
in long-only Indian momentum portfolios.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import BaseStrategy


class MomentumStrategy(BaseStrategy):
    def __init__(self):
        super().__init__(name="momentum_12_1", needs_fundamentals=False)

    def score(self, symbol, history, fundamentals=None, asof=None):
        df = self._slice(history, asof)
        if len(df) < 253:
            return np.nan
        end = df["Close"].iloc[-21]
        start = df["Close"].iloc[-252]
        if start <= 0:
            return np.nan
        return float((end / start) - 1.0)
