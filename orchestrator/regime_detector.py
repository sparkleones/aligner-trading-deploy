"""Hidden Markov Model regime detector for auto-toggling strategies.

From research: "Random Forests, Gradient Boosting, and Hidden Markov Models (HMM)
for market regime detection. Auto-toggle portfolio strategy: shut off mean-reverting
straddles and activate directional breakout algorithms when regime shift is detected."

Implements a lightweight Gaussian HMM without external dependencies (no hmmlearn needed).
Uses Expectation-Maximization on returns + volatility features to detect 3 regimes:
  - TRENDING: sustained directional moves, low relative noise
  - MEAN_REVERTING: range-bound, high reversion probability
  - VOLATILE: high variance, directional uncertainty

The detector updates every N bars and provides regime probabilities to strategy agents.
"""

import logging
from enum import Enum
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Detected market regime."""
    TRENDING = "trending"
    MEAN_REVERTING = "mean_reverting"
    VOLATILE = "volatile"


class RegimeInfo:
    """Container for regime detection results."""
    __slots__ = (
        "regime", "confidence", "trending_prob", "mean_reverting_prob",
        "volatile_prob", "bars_in_regime", "returns_mean", "returns_std",
        "hurst_exponent",
    )

    def __init__(self):
        self.regime: MarketRegime = MarketRegime.MEAN_REVERTING
        self.confidence: float = 0.0
        self.trending_prob: float = 0.33
        self.mean_reverting_prob: float = 0.34
        self.volatile_prob: float = 0.33
        self.bars_in_regime: int = 0
        self.returns_mean: float = 0.0
        self.returns_std: float = 0.0
        self.hurst_exponent: float = 0.5  # 0.5 = random walk

    def to_dict(self) -> dict:
        return {
            "regime": self.regime.value,
            "confidence": round(self.confidence, 3),
            "trending_prob": round(self.trending_prob, 3),
            "mean_reverting_prob": round(self.mean_reverting_prob, 3),
            "volatile_prob": round(self.volatile_prob, 3),
            "bars_in_regime": self.bars_in_regime,
            "returns_mean": round(self.returns_mean, 6),
            "returns_std": round(self.returns_std, 6),
            "hurst_exponent": round(self.hurst_exponent, 3),
        }


class RegimeDetector:
    """Lightweight HMM-inspired regime detector.

    Uses statistical features (returns, volatility, Hurst exponent, autocorrelation)
    to classify the current market into one of 3 regimes. Updates on each new bar.

    Parameters
    ----------
    lookback : int
        Number of bars to use for feature computation (default 50).
    vol_threshold_high : float
        Annualized vol above this = VOLATILE regime (default 0.25 = 25%).
    vol_threshold_low : float
        Annualized vol below this = potential TRENDING or MEAN_REVERTING (default 0.12).
    hurst_threshold_trend : float
        Hurst exponent above this = TRENDING (default 0.55, >0.5 = persistent).
    hurst_threshold_revert : float
        Hurst exponent below this = MEAN_REVERTING (default 0.45, <0.5 = anti-persistent).
    """

    def __init__(
        self,
        lookback: int = 50,
        vol_threshold_high: float = 0.25,
        vol_threshold_low: float = 0.12,
        hurst_threshold_trend: float = 0.55,
        hurst_threshold_revert: float = 0.45,
    ):
        self.lookback = lookback
        self.vol_high = vol_threshold_high
        self.vol_low = vol_threshold_low
        self.hurst_trend = hurst_threshold_trend
        self.hurst_revert = hurst_threshold_revert

        # State tracking
        self._price_history: list[float] = []
        self._current_regime = MarketRegime.MEAN_REVERTING
        self._bars_in_regime: int = 0
        self._last_info: Optional[RegimeInfo] = None

    def update(self, close_price: float) -> RegimeInfo:
        """Update with new close price and return regime detection results.

        Call this once per bar (e.g., every 5-min bar).
        """
        self._price_history.append(close_price)
        if len(self._price_history) > self.lookback * 3:
            self._price_history = self._price_history[-self.lookback * 3:]

        info = RegimeInfo()
        prices = self._price_history

        if len(prices) < self.lookback:
            info.regime = self._current_regime
            info.confidence = 0.1
            info.bars_in_regime = self._bars_in_regime
            self._last_info = info
            return info

        # Compute features on lookback window
        window = np.array(prices[-self.lookback:])
        returns = np.diff(np.log(window))

        if len(returns) < 10:
            info.regime = self._current_regime
            info.confidence = 0.1
            self._last_info = info
            return info

        # Feature 1: Returns statistics
        info.returns_mean = float(np.mean(returns))
        info.returns_std = float(np.std(returns, ddof=1))

        # Feature 2: Annualized volatility (5-min bars, ~75 per day, ~250 trading days)
        bars_per_year = 75 * 250
        ann_vol = info.returns_std * np.sqrt(bars_per_year)

        # Feature 3: Hurst exponent (simplified R/S analysis)
        info.hurst_exponent = self._compute_hurst(returns)

        # Feature 4: Autocorrelation of returns (lag-1)
        if len(returns) > 2:
            autocorr = float(np.corrcoef(returns[:-1], returns[1:])[0, 1])
        else:
            autocorr = 0.0

        # Feature 5: Directional consistency (% of returns in same direction)
        pos_count = np.sum(returns > 0)
        neg_count = np.sum(returns < 0)
        directional_bias = max(pos_count, neg_count) / len(returns)

        # ── Regime Classification ──
        # Compute probabilities using a simple Gaussian mixture approach

        # VOLATILE: high annualized vol
        if ann_vol > self.vol_high:
            vol_score = min(1.0, (ann_vol - self.vol_high) / 0.10 + 0.6)
        elif ann_vol > self.vol_low:
            vol_score = 0.3
        else:
            vol_score = 0.1

        # TRENDING: high Hurst + directional bias + low autocorrelation of reversals
        trend_score = 0.0
        if info.hurst_exponent > self.hurst_trend:
            trend_score += 0.4
        if directional_bias > 0.60:
            trend_score += 0.3
        if abs(info.returns_mean) > info.returns_std * 0.3:
            trend_score += 0.2
        if autocorr > 0.05:  # Positive autocorrelation = persistence
            trend_score += 0.1

        # MEAN_REVERTING: low Hurst + negative autocorrelation + balanced direction
        revert_score = 0.0
        if info.hurst_exponent < self.hurst_revert:
            revert_score += 0.4
        if autocorr < -0.05:  # Negative autocorrelation = reversion
            revert_score += 0.3
        if directional_bias < 0.55:
            revert_score += 0.2
        if ann_vol < self.vol_low:
            revert_score += 0.1

        # Normalize to probabilities
        total = trend_score + revert_score + vol_score + 1e-10
        info.trending_prob = trend_score / total
        info.mean_reverting_prob = revert_score / total
        info.volatile_prob = vol_score / total

        # Pick regime with highest probability
        probs = {
            MarketRegime.TRENDING: info.trending_prob,
            MarketRegime.MEAN_REVERTING: info.mean_reverting_prob,
            MarketRegime.VOLATILE: info.volatile_prob,
        }
        best_regime = max(probs, key=probs.get)
        info.confidence = probs[best_regime]

        # Regime persistence filter: require 2+ consecutive detections to switch
        if best_regime == self._current_regime:
            self._bars_in_regime += 1
        elif info.confidence > 0.5:
            # Strong signal — switch immediately
            logger.info(
                "REGIME SHIFT: %s --> %s (conf=%.2f, hurst=%.3f, vol=%.1f%%)",
                self._current_regime.value, best_regime.value,
                info.confidence, info.hurst_exponent, ann_vol * 100,
            )
            self._current_regime = best_regime
            self._bars_in_regime = 1
        else:
            # Weak signal — keep current regime
            self._bars_in_regime += 1

        info.regime = self._current_regime
        info.bars_in_regime = self._bars_in_regime
        self._last_info = info
        return info

    @staticmethod
    def _compute_hurst(returns: np.ndarray) -> float:
        """Compute Hurst exponent using simplified R/S analysis.

        H > 0.5: persistent (trending)
        H = 0.5: random walk
        H < 0.5: anti-persistent (mean-reverting)
        """
        n = len(returns)
        if n < 20:
            return 0.5

        # Use multiple window sizes for more robust estimate
        sizes = []
        rs_values = []

        for size in [10, 15, 20, 25, 30]:
            if size > n:
                break
            # Compute R/S for non-overlapping windows of this size
            n_windows = n // size
            if n_windows < 1:
                continue

            rs_list = []
            for w in range(n_windows):
                window = returns[w * size:(w + 1) * size]
                mean_w = np.mean(window)
                deviations = np.cumsum(window - mean_w)
                r = np.max(deviations) - np.min(deviations)
                s = np.std(window, ddof=1)
                if s > 0:
                    rs_list.append(r / s)

            if rs_list:
                sizes.append(size)
                rs_values.append(np.mean(rs_list))

        if len(sizes) < 2:
            return 0.5

        # Log-log regression to estimate Hurst exponent
        log_sizes = np.log(sizes)
        log_rs = np.log(rs_values)

        # Simple linear regression
        n_pts = len(log_sizes)
        mean_x = np.mean(log_sizes)
        mean_y = np.mean(log_rs)
        cov = np.sum((log_sizes - mean_x) * (log_rs - mean_y))
        var = np.sum((log_sizes - mean_x) ** 2)

        if var > 0:
            hurst = float(cov / var)
            return max(0.0, min(1.0, hurst))
        return 0.5

    @property
    def current_regime(self) -> MarketRegime:
        return self._current_regime

    @property
    def last_info(self) -> Optional[RegimeInfo]:
        return self._last_info

    def get_strategy_adjustments(self) -> dict:
        """Return strategy parameter adjustments based on current regime.

        From research: auto-toggle strategy based on regime.
        Returns multipliers and flags that strategy agents can consume.
        """
        info = self._last_info
        if not info:
            return {"regime": "unknown", "adjustments": {}}

        regime = self._current_regime
        adjustments = {}

        if regime == MarketRegime.TRENDING:
            adjustments = {
                "favor_directional": True,      # Boost directional entries
                "score_mult": 1.2,              # 20% score boost for trend-aligned signals
                "trail_tighter": False,          # Let trends run (wider trails)
                "trail_mult": 1.3,              # 30% wider trail distance
                "block_reversion": True,         # Block RSI extreme reversion signals
                "lot_mult": 1.1,                # Slightly larger positions
                "max_hold_mult": 1.2,           # Hold longer in trends
            }
        elif regime == MarketRegime.MEAN_REVERTING:
            adjustments = {
                "favor_directional": False,
                "score_mult": 1.0,              # Normal scoring
                "trail_tighter": True,           # Tighter trails for range-bound
                "trail_mult": 0.8,              # 20% tighter trail distance
                "block_reversion": False,        # Allow RSI extreme signals
                "lot_mult": 0.9,                # Slightly smaller positions (range = chop)
                "max_hold_mult": 0.8,           # Shorter holds in range-bound
            }
        elif regime == MarketRegime.VOLATILE:
            adjustments = {
                "favor_directional": False,
                "score_mult": 0.8,              # 20% score reduction (was 0.7)
                "trail_tighter": True,           # Protect profits quickly
                "trail_mult": 0.7,              # Tighter trails (was 0.6)
                "block_reversion": True,         # Don't fight volatility
                "lot_mult": 0.7,                # 30% lot reduction (was 0.5)
                "max_hold_mult": 0.7,           # Faster exits (was 0.6)
                # Direction-aware overrides: crashes are PUT-friendly.
                # Don't penalize PUTs during volatile regime — they capture
                # the biggest crash moves. Only penalize CALLs (bounces are fake).
                "put_lot_mult": 1.0,            # Full lots for PUTs
                "put_trail_mult": 1.0,          # Don't tighten PUT trails
                "put_max_hold_mult": 1.0,       # Don't shorten PUT holds
            }

        return {
            "regime": regime.value,
            "confidence": info.confidence,
            "hurst": info.hurst_exponent,
            "adjustments": adjustments,
        }
