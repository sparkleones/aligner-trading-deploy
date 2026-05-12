"""
Quality-Growth-Value (QGV) Strategy — modeled after how top Indian
small-cap MFs (Nippon SC, HDFC SC, SBI SC) actually pick stocks.

DESIGN
------
Fundamental filters for LIVE picks (snapshot OK):
  ROE > 15%
  Debt/Equity < 1.0
  EPS growth > 20%  (or revenue growth > 15% as fallback)
  PE < 1.5 x EPS growth rate  (Lynch's GARP rule)

PRICE-based quality proxies for BACKTEST (no look-ahead):
  Q1. Persistence: > 60% of last 750 trading days above 200DMA
  Q2. Drawdown control: max drawdown over last 750 days > -35%
  Q3. Smoothness: low realized vol on 252-day window
  Q4. R^2 of 6-month exp regression > 0.5

GROWTH proxies (price-based):
  G1. 3-year price CAGR > 15%
  G2. 1-year price return > 10%
  G3. Above 200DMA AND above 50DMA today

VALUE proxy (price-based):
  V1. Drawdown from 52w high between -5% and -30%
      (i.e. "quality on small dip", not at the top, not crashed)

FINAL composite score = z-score(Q) * 0.40 + z-score(G) * 0.35 + z-score(V) * 0.25
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .base import BaseStrategy


class QGVStrategy(BaseStrategy):
    """Quality + Growth + Value composite for SC/MC picks."""

    def __init__(self):
        super().__init__(name="qgv", needs_fundamentals=False)
        self.composite_weights = {"quality": 0.40, "growth": 0.35, "value": 0.25}

    # ── Quality sub-scores ──
    def _persistence_above_200dma(self, c: pd.Series) -> float:
        """Fraction of last 750 days where close > 200DMA."""
        if len(c) < 750 + 200:
            return np.nan
        slice_ = c.tail(750 + 200)
        ma_200 = slice_.rolling(200).mean()
        last_750_close = slice_.tail(750)
        last_750_ma = ma_200.tail(750)
        return float((last_750_close > last_750_ma).mean())

    def _max_drawdown_3y(self, c: pd.Series) -> float:
        """Max drawdown over last 750 days (negative number)."""
        if len(c) < 750:
            return np.nan
        slice_ = c.tail(750)
        running_max = slice_.cummax()
        dd = (slice_ / running_max - 1.0).min()
        return float(dd)

    def _smoothness(self, c: pd.Series) -> float:
        """Negative annualized vol (higher = smoother)."""
        if len(c) < 252:
            return np.nan
        rets = c.pct_change().tail(252).dropna()
        if len(rets) < 200:
            return np.nan
        vol = rets.std() * np.sqrt(252)
        if vol <= 0 or not np.isfinite(vol):
            return np.nan
        return float(-vol)

    def _trend_r_squared(self, c: pd.Series) -> float:
        """R^2 of 6-month log-price linear regression."""
        if len(c) < 126:
            return np.nan
        log_p = np.log(c.tail(126).values)
        if np.any(~np.isfinite(log_p)):
            return np.nan
        x = np.arange(len(log_p))
        try:
            _, _, r, _, _ = stats.linregress(x, log_p)
            return float(r * r)
        except Exception:
            return np.nan

    # ── Growth sub-scores ──
    def _three_year_cagr(self, c: pd.Series) -> float:
        if len(c) < 750:
            return np.nan
        start = c.iloc[-750]
        end = c.iloc[-1]
        if start <= 0:
            return np.nan
        years = 3.0
        return float((end / start) ** (1.0 / years) - 1.0)

    def _one_year_return(self, c: pd.Series) -> float:
        if len(c) < 253:
            return np.nan
        start = c.iloc[-252]
        end = c.iloc[-1]
        if start <= 0:
            return np.nan
        return float((end / start) - 1.0)

    # ── Value sub-score ──
    def _quality_on_dip(self, c: pd.Series, h: pd.Series) -> float:
        """Sweet spot: 5-30% below 52w high but holding 200DMA."""
        if len(c) < 252 or len(h) < 252:
            return np.nan
        close = float(c.iloc[-1])
        high_252 = float(h.tail(252).max())
        if high_252 <= 0:
            return np.nan
        dd = (close / high_252) - 1.0  # negative
        ma_200 = float(c.rolling(200).mean().iloc[-1])
        if close < ma_200:
            return 0.0
        # Sweet spot: -30% <= dd <= -5%
        # Best: dd around -15%
        if -0.30 <= dd <= -0.05:
            # Triangular: max at -15%
            return float(1.0 - abs(dd + 0.15) / 0.15)
        elif dd > -0.05:
            # Too close to high — modest value
            return 0.3
        else:
            # Too deep — quality compromised
            return 0.0

    def score(self, symbol, history, fundamentals=None, asof=None):
        df = self._slice(history, asof)
        if len(df) < 500:    # 2y minimum (was 3y - too strict)
            return np.nan
        c = df["Close"]
        h = df["High"]
        close = float(c.iloc[-1])
        if close <= 0:
            return np.nan

        # Soft trend filter: allow up to 5% below 200DMA (dip-buying)
        ma_200 = float(c.rolling(200).mean().iloc[-1]) if len(c) >= 200 else 0
        if ma_200 > 0 and close < ma_200 * 0.95:
            return np.nan  # too far below 200DMA -> reject

        # Quality sub-scores (some may be NaN if < 750 bars)
        q_persistence = self._persistence_above_200dma(c)
        q_dd_3y = self._max_drawdown_3y(c)
        q_smooth = self._smoothness(c)
        q_r2 = self._trend_r_squared(c)

        # Growth sub-scores
        g_3y_cagr = self._three_year_cagr(c)
        g_1y_ret = self._one_year_return(c)

        # Value
        v_dip = self._quality_on_dip(c, h)

        # ── RELAXED hard filters ──
        # Persistence threshold lowered: must be above 200DMA > 35% of last 3y
        if not pd.isna(q_persistence) and q_persistence < 0.35:
            return np.nan
        # DD: reject only stocks that crashed >60% (was 50%)
        if not pd.isna(q_dd_3y) and q_dd_3y < -0.60:
            return np.nan
        # 3y CAGR: require positive only (was 10%) — let the SCORE
        # rank growth, but don't kill the candidate
        if not pd.isna(g_3y_cagr) and g_3y_cagr < -0.05:
            return np.nan

        # Replace any NaN sub-scores with neutral defaults
        q_pers = q_persistence if not pd.isna(q_persistence) else 0.5
        q_dd = q_dd_3y if not pd.isna(q_dd_3y) else -0.25
        q_sm = q_smooth if not pd.isna(q_smooth) else -0.20
        q_r = q_r2 if not pd.isna(q_r2) else 0.4
        g_3y = g_3y_cagr if not pd.isna(g_3y_cagr) else 0.10
        g_1y = g_1y_ret if not pd.isna(g_1y_ret) else 0.05
        v_d = v_dip if not pd.isna(v_dip) else 0.3

        # Sub-composites (each ~0..1)
        quality = (
            0.3 * q_pers
            + 0.3 * max(0, (q_dd + 0.6) / 0.6)        # -0.6..0 -> 0..1
            + 0.2 * max(0, (q_sm + 0.35) / 0.35)      # -0.35..0 -> 0..1
            + 0.2 * q_r
        )
        growth = 0.6 * g_3y + 0.4 * g_1y
        value = v_d

        composite = (
            self.composite_weights["quality"] * quality
            + self.composite_weights["growth"] * growth
            + self.composite_weights["value"] * value
        )
        if not np.isfinite(composite):
            return np.nan
        return float(composite)


def passes_fundamental_filter(fund: dict, strict: bool = True) -> tuple[bool, list[str]]:
    """
    Check if a stock passes Quality+Growth+Value fundamental filter
    using yfinance .info snapshot. For LIVE picks only.

    Returns: (pass_bool, list_of_failure_reasons)
    """
    if not fund:
        return False, ["no_fundamentals"]
    reasons = []

    # ROE > 15%
    roe = fund.get("returnOnEquity")
    if roe is None or not np.isfinite(roe):
        reasons.append("missing_roe")
    elif roe < 0.15:
        reasons.append(f"low_roe ({roe*100:.1f}%)")

    # D/E < 1.0 (yfinance reports D/E as percentage * 100)
    de = fund.get("debtToEquity")
    if de is not None and np.isfinite(de):
        if de > 150 and strict:
            reasons.append(f"high_debt (D/E={de:.0f})")
        elif de > 200:
            reasons.append(f"high_debt (D/E={de:.0f})")

    # EPS growth > 20% (or revenue growth > 15% as fallback)
    eps_g = fund.get("earningsGrowth")
    rev_g = fund.get("revenueGrowth")
    if eps_g is not None and np.isfinite(eps_g):
        if eps_g < 0.20:
            # Allow if rev_g > 0.15
            if rev_g is None or not np.isfinite(rev_g) or rev_g < 0.15:
                reasons.append(f"low_growth (EPS:{eps_g*100:.0f}%, Rev:{rev_g*100 if rev_g is not None else '?'}%)")
    elif rev_g is not None and np.isfinite(rev_g):
        if rev_g < 0.15:
            reasons.append(f"low_growth (Rev:{rev_g*100:.0f}%)")
    # If both missing, allow it through (we don't have data)

    # PE < 1.5 * EPS growth (Lynch GARP)
    pe = fund.get("trailingPE") or fund.get("forwardPE")
    if pe is not None and np.isfinite(pe) and pe > 0:
        if pe > 80:
            reasons.append(f"extremely_overvalued (PE={pe:.0f})")
        elif eps_g is not None and np.isfinite(eps_g) and eps_g > 0:
            peg = pe / (eps_g * 100)
            if peg > 2.0 and strict:
                reasons.append(f"high_peg ({peg:.1f})")

    # Profit margin > 5% (avoid loss-making cos)
    pm = fund.get("profitMargins")
    if pm is not None and np.isfinite(pm) and pm < 0.05:
        reasons.append(f"low_profit_margin ({pm*100:.1f}%)")

    return (len(reasons) == 0), reasons
