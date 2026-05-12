"""Base contract for all screening strategies."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class BaseStrategy:
    """All strategies inherit this and implement `score()`."""

    name: str = "base"
    needs_fundamentals: bool = False
    direction: str = "long"   # 'long' (high score = buy) or 'short'

    def score(
        self,
        symbol: str,
        history: pd.DataFrame,
        fundamentals: Optional[dict] = None,
        asof: Optional[pd.Timestamp] = None,
    ) -> float:
        """Return a single score for the stock. Higher = more bullish.

        history: OHLCV DataFrame indexed by date.
        fundamentals: optional dict from yfinance .info or similar.
        asof: snapshot date — if given, use only data up to that date.

        Returns np.nan if the stock cannot be scored (insufficient data).
        """
        raise NotImplementedError

    def _slice(self, history: pd.DataFrame, asof: Optional[pd.Timestamp]) -> pd.DataFrame:
        """Helper: slice history up to and including `asof`."""
        if asof is None:
            return history
        return history.loc[:asof]
