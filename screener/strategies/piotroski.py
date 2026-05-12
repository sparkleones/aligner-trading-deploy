"""
Strategy 9: Piotroski F-Score (Piotroski 2000)

Joseph Piotroski's seminal paper "Value Investing: The Use of Historical
Financial Statement Information to Separate Winners from Losers"
(Journal of Accounting Research, 2000) showed that within high-book-to-
market value stocks, a 9-point binary score predicted returns:

  Profitability (4 points):
    1. Net income > 0
    2. Operating cash flow > 0
    3. ROA > prior-year ROA
    4. CFO > Net Income (quality of earnings)
  Leverage / Liquidity (3 points):
    5. Long-term debt decreased YoY
    6. Current ratio increased YoY
    7. No share issuance in past year
  Operating Efficiency (2 points):
    8. Gross margin increased YoY
    9. Asset turnover increased YoY

Piotroski showed F-Score >= 8 returned +13.4% vs +5.5% for the full
value subset (1976-1996, US).

Snapshot caveat: yfinance .info gives current ratios but not all 9
binary comparisons. We approximate with available fields.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import BaseStrategy


class PiotroskiStrategy(BaseStrategy):
    def __init__(self):
        super().__init__(name="piotroski", needs_fundamentals=True)

    def score(self, symbol, history, fundamentals=None, asof=None):
        if not fundamentals:
            return np.nan

        score = 0
        signals = 0  # how many signals we could actually evaluate

        # 1. Profitability — net income > 0
        ni = fundamentals.get("netIncomeToCommon") or fundamentals.get("netIncome")
        if ni is not None and np.isfinite(ni):
            signals += 1
            if ni > 0:
                score += 1

        # 2. Operating cash flow > 0
        cfo = fundamentals.get("operatingCashflow")
        if cfo is not None and np.isfinite(cfo):
            signals += 1
            if cfo > 0:
                score += 1

        # 3. ROA (proxy: returnOnAssets)
        roa = fundamentals.get("returnOnAssets")
        if roa is not None and np.isfinite(roa):
            signals += 1
            if roa > 0:
                score += 1

        # 4. CFO > Net Income (earnings quality)
        if cfo is not None and ni is not None and np.isfinite(cfo) and np.isfinite(ni):
            signals += 1
            if cfo > ni:
                score += 1

        # 5. Low debt-to-equity
        de = fundamentals.get("debtToEquity")
        if de is not None and np.isfinite(de):
            signals += 1
            if de < 100:  # D/E < 1.0
                score += 1

        # 6. Current ratio strong
        cr = fundamentals.get("currentRatio")
        if cr is not None and np.isfinite(cr):
            signals += 1
            if cr > 1.5:
                score += 1

        # 7. Gross margin healthy
        gm = fundamentals.get("grossMargins")
        if gm is not None and np.isfinite(gm):
            signals += 1
            if gm > 0.20:
                score += 1

        # 8. Operating margin healthy
        om = fundamentals.get("operatingMargins")
        if om is not None and np.isfinite(om):
            signals += 1
            if om > 0.10:
                score += 1

        # 9. ROE healthy (proxy for asset turnover efficiency)
        roe = fundamentals.get("returnOnEquity")
        if roe is not None and np.isfinite(roe):
            signals += 1
            if roe > 0.12:
                score += 1

        if signals < 5:
            return np.nan
        # Normalize to 0..1 then return raw
        return float(score / signals)
