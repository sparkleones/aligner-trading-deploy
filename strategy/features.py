"""
Feature engineering for the DDQN trading agent.

Computes a high-dimensional state vector from raw market data including
technical indicators, options Greeks, volatility metrics, and inter-index
co-integration signals.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config.constants import DDQN_STATE_DIM

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper: Supertrend
# ---------------------------------------------------------------------------

def _supertrend(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int,
    multiplier: float,
) -> pd.Series:
    """Compute Supertrend indicator and return the trend direction series.

    +1 = bullish, -1 = bearish.
    """
    hl2 = (high + low) / 2.0
    atr = (
        pd.concat(
            [
                high - low,
                (high - close.shift(1)).abs(),
                (low - close.shift(1)).abs(),
            ],
            axis=1,
        )
        .max(axis=1)
        .rolling(window=period)
        .mean()
    )

    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr

    direction = pd.Series(np.ones(len(close)), index=close.index)  # 1 = bullish

    for i in range(1, len(close)):
        if close.iloc[i] > upper_band.iloc[i - 1]:
            direction.iloc[i] = 1.0
        elif close.iloc[i] < lower_band.iloc[i - 1]:
            direction.iloc[i] = -1.0
        else:
            direction.iloc[i] = direction.iloc[i - 1]
            if direction.iloc[i] == 1.0 and lower_band.iloc[i] < lower_band.iloc[i - 1]:
                lower_band.iloc[i] = lower_band.iloc[i - 1]
            if direction.iloc[i] == -1.0 and upper_band.iloc[i] > upper_band.iloc[i - 1]:
                upper_band.iloc[i] = upper_band.iloc[i - 1]

    return direction


# ---------------------------------------------------------------------------
# Feature Engine
# ---------------------------------------------------------------------------


class FeatureEngine:
    """Transforms raw market data into a normalised state vector for the agent.

    Parameters
    ----------
    state_dim : int
        Desired dimensionality of the output vector.  If the raw feature count
        is smaller the vector is zero-padded; if larger it is truncated.
    """

    def __init__(self, state_dim: int = DDQN_STATE_DIM) -> None:
        self.state_dim: int = state_dim
        self._feature_names: List[str] = []
        logger.info(
            "%s | FeatureEngine initialised with state_dim=%d",
            datetime.utcnow().isoformat(),
            self.state_dim,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(
        self,
        ohlcv_df: pd.DataFrame,
        greeks: Optional[Dict[str, float]] = None,
        vix: Optional[float] = None,
        nifty_price: Optional[float] = None,
        banknifty_price: Optional[float] = None,
    ) -> np.ndarray:
        """Build the full state vector.

        Parameters
        ----------
        ohlcv_df : pd.DataFrame
            Must contain columns ``open, high, low, close, volume`` with a
            DatetimeIndex at 15-min frequency.  At least 50 rows are needed
            for the longest look-back indicator (EMA-50).
        greeks : dict, optional
            Keys ``delta, gamma, theta, vega``.
        vix : float, optional
            India VIX value.
        nifty_price : float, optional
            Latest Nifty-50 spot for co-integration spread.
        banknifty_price : float, optional
            Latest BankNifty spot for co-integration spread.

        Returns
        -------
        np.ndarray
            Normalised feature vector of shape ``(state_dim,)``.
        """
        ts = datetime.utcnow().isoformat()

        if len(ohlcv_df) < 2:
            logger.warning("%s | Not enough OHLCV rows (%d); returning zeros", ts, len(ohlcv_df))
            return np.zeros(self.state_dim, dtype=np.float32)

        features: List[float] = []
        names: List[str] = []

        close = ohlcv_df["close"]
        high = ohlcv_df["high"]
        low = ohlcv_df["low"]
        volume = ohlcv_df["volume"]

        # --- OHLCV normalised returns ---
        ret = close.pct_change().iloc[-1] if len(close) > 1 else 0.0
        features.append(float(ret))
        names.append("close_return")

        vol_change = volume.pct_change().iloc[-1] if len(volume) > 1 else 0.0
        features.append(float(np.clip(vol_change, -5.0, 5.0)))
        names.append("volume_change")

        # --- High-low range relative to close ---
        hl_range = (high.iloc[-1] - low.iloc[-1]) / (close.iloc[-1] + 1e-9)
        features.append(float(hl_range))
        names.append("hl_range_pct")

        # --- EMAs (9, 21, 50) expressed as pct distance from close ---
        for span in (9, 21, 50):
            ema = close.ewm(span=span, adjust=False).mean()
            dist = (close.iloc[-1] - ema.iloc[-1]) / (close.iloc[-1] + 1e-9)
            features.append(float(dist))
            names.append(f"ema_{span}_dist")

        # --- Supertrend (multiple configs) ---
        for period, mult in [(10, 3.0), (11, 2.0), (12, 1.0)]:
            st_dir = _supertrend(high, low, close, period, mult)
            features.append(float(st_dir.iloc[-1]))
            names.append(f"supertrend_{period}_{int(mult)}")

        # --- RSI (14) ---
        rsi_val = self._rsi(close, 14)
        features.append(float(rsi_val))
        names.append("rsi_14")

        # --- MACD ---
        macd_line, signal_line, histogram = self._macd(close)
        features.extend([float(macd_line), float(signal_line), float(histogram)])
        names.extend(["macd_line", "macd_signal", "macd_hist"])

        # --- Pivot points ---
        pp, r1, s1, r2, s2 = self._pivot_points(high, low, close)
        close_now = close.iloc[-1]
        for label, level in [("pp", pp), ("r1", r1), ("s1", s1), ("r2", r2), ("s2", s2)]:
            dist = (close_now - level) / (close_now + 1e-9)
            features.append(float(dist))
            names.append(f"pivot_{label}_dist")

        # --- Nifty-BankNifty spread ---
        if nifty_price is not None and banknifty_price is not None:
            spread = (banknifty_price / (nifty_price + 1e-9)) - 1.0
            features.append(float(spread))
        else:
            features.append(0.0)
        names.append("nifty_bn_spread")

        # --- India VIX ---
        if vix is not None:
            normalised_vix = (vix - 15.0) / 15.0  # centre around 15, scale by 15
            features.append(float(np.clip(normalised_vix, -2.0, 2.0)))
        else:
            features.append(0.0)
        names.append("india_vix")

        # --- Options Greeks ---
        greek_defaults = {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0}
        if greeks is not None:
            greek_defaults.update(greeks)
        for g in ("delta", "gamma", "theta", "vega"):
            features.append(float(greek_defaults[g]))
            names.append(f"greek_{g}")

        self._feature_names = names
        raw = np.array(features, dtype=np.float32)

        # --- Normalise to [-1, 1] via tanh squashing ---
        normalised = np.tanh(raw)

        # --- Pad or truncate to state_dim ---
        state = np.zeros(self.state_dim, dtype=np.float32)
        n = min(len(normalised), self.state_dim)
        state[:n] = normalised[:n]

        logger.debug(
            "%s | Feature vector built: raw_dim=%d, final_dim=%d",
            ts,
            len(raw),
            self.state_dim,
        )
        return state

    @property
    def feature_names(self) -> List[str]:
        """Names of the most recently computed raw features (pre-padding)."""
        return list(self._feature_names)

    # ------------------------------------------------------------------
    # Tick-to-bar aggregation
    # ------------------------------------------------------------------

    @staticmethod
    def ticks_to_bars(
        ticks: pd.DataFrame,
        interval: str = "15min",
    ) -> pd.DataFrame:
        """Resample tick-level data into OHLCV bars.

        Parameters
        ----------
        ticks : pd.DataFrame
            Must contain ``price`` and ``volume`` columns with a
            DatetimeIndex.
        interval : str
            Pandas-compatible frequency string (default ``"15min"``).

        Returns
        -------
        pd.DataFrame
            Columns ``open, high, low, close, volume``.
        """
        bars = ticks["price"].resample(interval).ohlc()
        bars.columns = ["open", "high", "low", "close"]
        bars["volume"] = ticks["volume"].resample(interval).sum()
        bars.dropna(subset=["open"], inplace=True)
        return bars

    # ------------------------------------------------------------------
    # Internal indicator helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _rsi(series: pd.Series, period: int = 14) -> float:
        """Compute RSI and return the latest value normalised to [-1, 1]."""
        delta = series.diff()
        gain = delta.clip(lower=0).rolling(window=period).mean()
        loss = (-delta.clip(upper=0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-9)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        latest = rsi.iloc[-1] if not rsi.empty else 50.0
        return (latest - 50.0) / 50.0  # map [0,100] -> [-1,1]

    @staticmethod
    def _macd(
        series: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> Tuple[float, float, float]:
        """Return (macd_line, signal_line, histogram) as normalised values."""
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        hist = macd_line - signal_line
        # Normalise relative to current price
        price = series.iloc[-1] + 1e-9
        return (
            float(macd_line.iloc[-1] / price),
            float(signal_line.iloc[-1] / price),
            float(hist.iloc[-1] / price),
        )

    @staticmethod
    def _pivot_points(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
    ) -> Tuple[float, float, float, float, float]:
        """Standard pivot points from the previous bar's HLC.

        Returns (PP, R1, S1, R2, S2).
        """
        h = float(high.iloc[-2]) if len(high) > 1 else float(high.iloc[-1])
        l = float(low.iloc[-2]) if len(low) > 1 else float(low.iloc[-1])
        c = float(close.iloc[-2]) if len(close) > 1 else float(close.iloc[-1])
        pp = (h + l + c) / 3.0
        r1 = 2.0 * pp - l
        s1 = 2.0 * pp - h
        r2 = pp + (h - l)
        s2 = pp - (h - l)
        return pp, r1, s1, r2, s2
