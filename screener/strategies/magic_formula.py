"""
Strategy 8: Magic Formula (Joel Greenblatt 2005)

From "The Little Book That Beats the Market" (2005). Ranks stocks by
the SUM of their ranks on two metrics:

  1. Earnings Yield  = EBIT / Enterprise Value   (proxy: 1/PE)
  2. Return on Capital = EBIT / (Net Working Capital + Net Fixed Assets)
     (proxy: ROCE or ROE)

Greenblatt's original study: 30.8% CAGR vs 12.4% S&P 500 (1988-2004).
Indian replication by Singh & Yadav (2015): outperformed Nifty by
~8%/year in 2003-2014 backtest.

Snapshot caveat: we use yfinance .info which is point-in-time TODAY,
not historical. So historical backtest results are illustrative, not
publishable. Live signals are fine.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import BaseStrategy


class MagicFormulaStrategy(BaseStrategy):
    def __init__(self):
        super().__init__(name="magic_formula", needs_fundamentals=True)

    def score(self, symbol, history, fundamentals=None, asof=None):
        if not fundamentals:
            return np.nan
        # Earnings yield proxy
        pe = fundamentals.get("trailingPE") or fundamentals.get("forwardPE")
        if pe is None or pe <= 0 or not np.isfinite(pe):
            return np.nan
        earnings_yield = 1.0 / pe
        # Return on capital proxy
        roe = fundamentals.get("returnOnEquity")
        if roe is None or not np.isfinite(roe):
            return np.nan
        # Greenblatt rank combo — we return a z-style score, ranker will
        # cross-sectionally normalize. Simple geometric mean of the two:
        if earnings_yield <= 0 or roe <= 0:
            return np.nan
        return float(np.sqrt(earnings_yield * roe))
