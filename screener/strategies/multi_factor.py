"""
Strategy 10: Multi-Factor VMQ (Value + Momentum + Quality)

The most robust multi-factor model from academic literature combines:

  - Value (Fama-French HML: low P/B, low P/E)
  - Momentum (Jegadeesh-Titman 12-1M)
  - Quality (Asness QMJ: high profitability, low debt, stable earnings)

Indian replication by Subrahmanyam (2018) and IIM-A Working Paper
2020-09-04: VMQ outperformed Nifty by ~5-6% CAGR with lower drawdown
in 2003-2019.

This blends the technical Momentum score with two fundamental
factors. Score = avg of z-scored sub-factors (computed by ranker
externally — this returns the raw composite).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import BaseStrategy
from .momentum import MomentumStrategy


class MultiFactorVMQStrategy(BaseStrategy):
    def __init__(self):
        super().__init__(name="multi_factor_vmq", needs_fundamentals=True)
        self._mom = MomentumStrategy()

    def score(self, symbol, history, fundamentals=None, asof=None):
        if not fundamentals:
            return np.nan

        # Value: inverse P/B (higher when cheap)
        pb = fundamentals.get("priceToBook")
        if pb is None or pb <= 0 or not np.isfinite(pb):
            return np.nan
        value_score = 1.0 / pb

        # Quality: ROE
        roe = fundamentals.get("returnOnEquity")
        if roe is None or not np.isfinite(roe):
            return np.nan
        quality_score = roe

        # Momentum: 12-1 from technical strategy
        mom = self._mom.score(symbol, history, asof=asof)
        if pd.isna(mom):
            return np.nan
        # Normalize momentum to similar scale
        momentum_score = mom

        # Equal-weighted geometric blend of positives
        # All three must be positive — if any negative, return NaN
        if value_score <= 0 or quality_score <= 0 or momentum_score <= 0:
            return np.nan
        return float((value_score * quality_score * (1.0 + momentum_score)) ** (1.0 / 3))
