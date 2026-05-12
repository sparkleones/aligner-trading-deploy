"""
Small-Cap Quality-Growth (SCQG) Strategy.

Modeled after how top Indian small-cap funds (Nippon India SC, HDFC SC,
SBI SC) actually operate. Combines 4 documented edges in small-caps:

  1. PRICE MOMENTUM (12-month) — small-caps trend in 6-12mo runs
  2. VOLATILITY-ADJUSTED MOMENTUM (Clenow Stocks-on-the-Move) — rewards
     smooth trends, penalizes whipsaw garbage that often dies in small-cap
  3. DRAWDOWN REVERSION — quality stocks 30-50% off highs that hold their
     200DMA = the "beaten-down quality" trade
  4. LOW-NOISE FILTER — exclude stocks with > 8% ATR (too jumpy for
     longer-term holding)

This is a PRICE-only strategy because fundamental backtesting requires
point-in-time data that yfinance doesn't provide.

For LIVE picks, we layer fundamentals on top via the FundamentalAnalyst
agent (ROE/PE/debt/growth). Backtest stays pure-price.

Hold period: held until next annual rebalance OR until stock breaks its
30-week MA. No fixed % stop-loss (matches MF behavior).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .base import BaseStrategy


class SCQGStrategy(BaseStrategy):
    """Small-Cap Quality-Growth composite."""

    def __init__(self):
        super().__init__(name="scqg", needs_fundamentals=False)
        # Sub-scorer weights
        self.weights = {
            "momentum_12m": 0.30,    # raw 12m return
            "trend_quality": 0.30,    # Clenow exp regression slope * R^2
            "drawdown_recovery": 0.20,  # bonus for above-200DMA quality at moderate DD
            "low_volatility": 0.20,   # penalize high-vol garbage
        }

    def _momentum_12m(self, c: pd.Series) -> float:
        if len(c) < 253:
            return np.nan
        start = c.iloc[-252]
        end = c.iloc[-1]
        if start <= 0:
            return np.nan
        return float((end / start) - 1.0)

    def _trend_quality(self, c: pd.Series) -> float:
        """Clenow exponential regression slope * R squared."""
        if len(c) < 91:
            return np.nan
        closes = c.tail(90).values
        if np.any(closes <= 0):
            return np.nan
        log_p = np.log(closes)
        x = np.arange(len(log_p))
        try:
            slope, _, r, _, _ = stats.linregress(x, log_p)
        except Exception:
            return np.nan
        annualized = np.exp(slope * 252) - 1.0
        return float(annualized * (r * r))

    def _drawdown_recovery(self, c: pd.Series, h: pd.Series) -> float:
        """+1 if 20-40% below 52w high but holding 200DMA (deep value of quality)."""
        if len(c) < 220:
            return np.nan
        high_252 = float(h.tail(252).max())
        if high_252 <= 0:
            return np.nan
        close = float(c.iloc[-1])
        dd = (close / high_252) - 1.0  # negative number, e.g. -0.30 = 30% below high
        ma_200 = float(c.rolling(200).mean().iloc[-1])
        above_200 = close > ma_200
        # Reward only if (a) holding 200DMA AND (b) drawdown 15-40%
        # i.e. "quality on sale"
        if above_200 and -0.40 <= dd <= -0.15:
            return float(abs(dd))   # bigger drawdown = bigger bonus, up to 0.40
        return 0.0

    def _low_volatility(self, c: pd.Series) -> float:
        """Negative annualized vol (low vol = better)."""
        if len(c) < 100:
            return np.nan
        rets = c.pct_change().tail(252).dropna()
        if len(rets) < 100:
            return np.nan
        vol = rets.std() * np.sqrt(252)
        if vol <= 0 or not np.isfinite(vol):
            return np.nan
        return float(-vol)

    def score(self, symbol, history, fundamentals=None, asof=None):
        df = self._slice(history, asof)
        if len(df) < 252:
            return np.nan
        c = df["Close"]
        h = df["High"]

        # Liquidity / quality filter — skip insanely volatile stocks
        atr_period = 20
        if len(df) >= atr_period + 1:
            high = df["High"]
            low = df["Low"]
            prev_close = c.shift(1)
            tr = pd.concat([
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ], axis=1).max(axis=1)
            atr = tr.rolling(atr_period).mean().iloc[-1]
            px = c.iloc[-1]
            if px <= 0 or pd.isna(atr) or (atr / px) > 0.08:
                return np.nan

        # Must be in uptrend (above 200DMA)
        ma_200 = c.rolling(200).mean().iloc[-1]
        if pd.isna(ma_200) or c.iloc[-1] < ma_200:
            return np.nan

        # Compute sub-scores
        s_mom = self._momentum_12m(c)
        s_trend = self._trend_quality(c)
        s_dd = self._drawdown_recovery(c, h)
        s_vol = self._low_volatility(c)

        # Require all primary scores
        if any(pd.isna(v) or not np.isfinite(v) for v in [s_mom, s_trend, s_vol]):
            return np.nan
        if s_dd is None or pd.isna(s_dd):
            s_dd = 0.0

        # Weighted composite — uses raw factors (the backtest engine
        # ranks cross-sectionally, so absolute scale doesn't matter)
        score = (
            self.weights["momentum_12m"] * s_mom
            + self.weights["trend_quality"] * s_trend
            + self.weights["drawdown_recovery"] * s_dd
            + self.weights["low_volatility"] * s_vol
        )
        return float(score) if np.isfinite(score) else np.nan
