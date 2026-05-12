"""
Strategy 6: Low Volatility (Asness, Frazzini, Pedersen 2014)

The Low Volatility anomaly: low-vol stocks deliver higher risk-adjusted
returns than high-vol stocks. Documented globally — see "Betting Against
Beta" (Frazzini & Pedersen, JFE 2014) and Indian replications by
Agrawal et al. 2016.

This is a DEFENSIVE strategy: it's expected to underperform in raging
bull markets but cushion drawdowns and win on Sharpe.

Score = -annualized_volatility   (negative because higher = better)

We also require the stock to be in an uptrend (above its 200-DMA),
otherwise we'd just be buying flat low-vol garbage.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import BaseStrategy


class LowVolatilityStrategy(BaseStrategy):
    def __init__(self):
        super().__init__(name="low_volatility", needs_fundamentals=False)

    def score(self, symbol, history, fundamentals=None, asof=None):
        df = self._slice(history, asof)
        if len(df) < 252:
            return np.nan
        rets = df["Close"].pct_change().tail(252).dropna()
        if len(rets) < 200:
            return np.nan
        vol = rets.std() * np.sqrt(252)
        if vol <= 0 or not np.isfinite(vol):
            return np.nan
        # Require 200-DMA trend filter
        ma200 = df["Close"].rolling(200).mean().iloc[-1]
        if pd.isna(ma200) or df["Close"].iloc[-1] < ma200:
            return np.nan
        # Negative vol — lower vol = higher (less negative) score
        return float(-vol)
