"""
Regime-Adaptive Ensemble Strategy.

The strategy that DOESN'T blindly trend-follow. Instead, it classifies
the market regime at each rebalance date and allocates between:
  - Momentum (composite_top: stage2 + breakout)   — winners in TRENDING
  - Low-Volatility                                 — winners in MIXED / DEFENSIVE
  - Mean-Reversion (Connors RSI-2)                — winners in CHOPPY
  - Cash                                          — when nothing works

Regime detection requires NIFTY context (price, 50/200 DMA) + VIX
percentile + cross-sectional breadth (% of universe above 200DMA).
These are computed OUTSIDE the per-stock score() call, then passed
into the strategy via a per-date regime cache.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

from .base import BaseStrategy
from .composite_top import CompositeTopStrategy
from .low_volatility import LowVolatilityStrategy
from .mean_reversion import MeanReversionStrategy


class Regime(str, Enum):
    TRENDING_UP = "TRENDING_UP"
    MIXED       = "MIXED"
    CHOPPY      = "CHOPPY"
    DEFENSIVE   = "DEFENSIVE"
    CRISIS      = "CRISIS"


# Allocation table V3 — adds NIFTY ETF (nifty_etf) as a fallback for
# regimes where individual-stock momentum is broken. In 2024-2026,
# the strategy lost money picking individual stocks even when NIFTY
# itself was positive — so the right move is to match NIFTY rather
# than try to beat it with broken signals.
#
# Fractions per regime (must sum to 1.0):
#   momentum  — composite_top (stage2 + breakout) on LARGE caps
#   nifty_etf — simulated NIFTYBEES (just tracks NIFTY total return)
#   cash      — held in cash, earning 0
ALLOCATION = {
    Regime.TRENDING_UP: {"momentum": 1.00, "nifty_etf": 0.00, "cash": 0.00},
    Regime.MIXED:       {"momentum": 0.60, "nifty_etf": 0.40, "cash": 0.00},
    Regime.CHOPPY:      {"momentum": 0.20, "nifty_etf": 0.70, "cash": 0.10},
    Regime.DEFENSIVE:   {"momentum": 0.00, "nifty_etf": 0.50, "cash": 0.50},
    Regime.CRISIS:      {"momentum": 0.00, "nifty_etf": 0.00, "cash": 1.00},
}


@dataclass
class RegimeSnapshot:
    regime: Regime
    nifty_close: float
    nifty_ma_50: float
    nifty_ma_200: float
    above_200dma: bool
    golden_cross: bool
    rsi_14: float
    vix_percentile: float
    breadth_pct: float
    reason: str


def classify_regime(
    nifty_history: pd.DataFrame,
    vix_history: pd.DataFrame | None,
    breadth_pct: float,
    asof: pd.Timestamp,
) -> RegimeSnapshot:
    """Classify the market regime using NIFTY + VIX + breadth as of `asof`."""
    nifty = nifty_history.loc[:asof]
    if len(nifty) < 220:
        return RegimeSnapshot(
            regime=Regime.MIXED, nifty_close=0, nifty_ma_50=0, nifty_ma_200=0,
            above_200dma=False, golden_cross=False, rsi_14=50, vix_percentile=50,
            breadth_pct=breadth_pct,
            reason="insufficient history -> default MIXED",
        )

    close = float(nifty["Close"].iloc[-1])
    ma_50 = float(nifty["Close"].rolling(50).mean().iloc[-1])
    ma_200 = float(nifty["Close"].rolling(200).mean().iloc[-1])
    above_200 = close > ma_200
    golden = ma_50 > ma_200

    # RSI(14) daily
    delta = nifty["Close"].diff().dropna()
    gain = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
    loss = -delta.clip(upper=0).ewm(com=13, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi_14 = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50

    # NIFTY drawdown from 52w high
    high_252 = float(nifty["High"].tail(252).max())
    dd_pct = (close / high_252 - 1.0)

    # VIX percentile
    vix_pct = 50.0
    vix_now = None
    if vix_history is not None and not vix_history.empty:
        vix_slice = vix_history.loc[:asof]
        if len(vix_slice) >= 60:
            v = vix_slice["Close"]
            vix_now = float(v.iloc[-1])
            last_yr = v.tail(252)
            vix_pct = float((last_yr <= vix_now).mean() * 100)

    # ── Classification V2 — less restrictive on TRENDING_UP ──
    # CRISIS: very high VIX OR deep drawdown
    if (vix_now is not None and vix_now >= 25) or dd_pct <= -0.15:
        regime = Regime.CRISIS
        reason = f"VIX {vix_now} OR DD {dd_pct*100:.1f}%"
    # DEFENSIVE: NIFTY below 200DMA AND (high VIX OR breadth very weak)
    elif (not above_200) and ((vix_now is not None and vix_now >= 22) or breadth_pct < 35):
        regime = Regime.DEFENSIVE
        reason = f"<200DMA + VIX {vix_now} OR breadth {breadth_pct:.0f}<35"
    # CHOPPY: NIFTY below 200DMA OR breadth very weak (without crisis vol)
    elif (not above_200) or breadth_pct < 40:
        regime = Regime.CHOPPY
        reason = f"<200DMA OR breadth {breadth_pct:.0f}<40"
    # TRENDING_UP: bull alignment — above 200DMA AND golden cross
    elif above_200 and golden:
        regime = Regime.TRENDING_UP
        reason = f">200DMA + golden cross + breadth {breadth_pct:.0f}%"
    # MIXED: above 200DMA but no golden cross (early-stage recovery)
    else:
        regime = Regime.MIXED
        reason = "above 200DMA but no golden cross yet"

    return RegimeSnapshot(
        regime=regime, nifty_close=close, nifty_ma_50=ma_50, nifty_ma_200=ma_200,
        above_200dma=above_200, golden_cross=golden, rsi_14=rsi_14,
        vix_percentile=vix_pct, breadth_pct=breadth_pct, reason=reason,
    )


def compute_breadth(history: dict[str, pd.DataFrame], asof: pd.Timestamp,
                     sample_size: int = 60) -> float:
    """% of sampled stocks above their 200-DMA as of `asof`."""
    above = 0
    total = 0
    for sym, df in list(history.items())[:sample_size]:
        sl = df.loc[:asof]
        if len(sl) < 200:
            continue
        total += 1
        close = float(sl["Close"].iloc[-1])
        ma_200 = float(sl["Close"].rolling(200).mean().iloc[-1])
        if close > ma_200:
            above += 1
    return (above / total * 100) if total > 0 else 50.0


class MomentumWrapper(BaseStrategy):
    """Wrapper for the composite_top momentum strategy."""
    def __init__(self):
        super().__init__(name="ensemble_momentum")
        self._inner = CompositeTopStrategy()
    def score(self, symbol, history, fundamentals=None, asof=None):
        return self._inner.score(symbol, history, fundamentals, asof)


class LowVolWrapper(BaseStrategy):
    def __init__(self):
        super().__init__(name="ensemble_low_vol")
        self._inner = LowVolatilityStrategy()
    def score(self, symbol, history, fundamentals=None, asof=None):
        return self._inner.score(symbol, history, fundamentals, asof)


class MeanRevWrapper(BaseStrategy):
    def __init__(self):
        super().__init__(name="ensemble_mean_rev")
        self._inner = MeanReversionStrategy()
    def score(self, symbol, history, fundamentals=None, asof=None):
        return self._inner.score(symbol, history, fundamentals, asof)
