"""
Slippage Model by Moneyness.

Estimates expected slippage, selects order type (MARKET vs LIMIT), and
calculates limit prices with moneyness-aware buffers for Indian NSE
options.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from config.constants import ATM_DELTA_THRESHOLD, OTM_DELTA_THRESHOLD

logger = logging.getLogger(__name__)


# ── Enums & Data Structures ─────────────────────────────────────────────────


class Moneyness(str, Enum):
    DEEP_ITM = "DEEP_ITM"
    ITM = "ITM"
    ATM = "ATM"
    OTM = "OTM"
    DEEP_OTM = "DEEP_OTM"


@dataclass(frozen=True)
class SlippageEstimate:
    """Result of a slippage estimation."""

    moneyness: Moneyness
    slippage_pct: float           # estimated one-way slippage as a fraction
    recommended_order_type: str   # "MARKET" or "LIMIT"
    spread_pct: float             # bid-ask spread as a fraction of mid
    delta: float


# ── Slippage Model ───────────────────────────────────────────────────────────


class SlippageModel:
    """Estimates slippage and selects order types based on option delta
    (a proxy for moneyness and liquidity)."""

    # Slippage bands: (min_pct, max_pct) as fractions
    _SLIPPAGE_BANDS = {
        Moneyness.ATM:      (0.0001, 0.0005),   # 0.01-0.05%
        Moneyness.ITM:      (0.0003, 0.0010),   # 0.03-0.10%
        Moneyness.DEEP_ITM: (0.0005, 0.0015),   # 0.05-0.15%
        Moneyness.OTM:      (0.0005, 0.0015),   # 0.05-0.15%
        Moneyness.DEEP_OTM: (0.0015, 0.0050),   # 0.15-0.50%
    }

    # Limit-price buffer multipliers (fraction of spread to add/subtract)
    _BUFFER_MULTIPLIERS = {
        Moneyness.ATM:      0.10,
        Moneyness.ITM:      0.20,
        Moneyness.DEEP_ITM: 0.30,
        Moneyness.OTM:      0.30,
        Moneyness.DEEP_OTM: 0.50,
    }

    # ── Public API ───────────────────────────────────────────────────────

    def classify_moneyness(self, delta: float) -> Moneyness:
        """Map absolute *delta* to a :class:`Moneyness` category."""
        abs_delta = abs(delta)
        if abs_delta >= 0.90:
            return Moneyness.DEEP_ITM
        if abs_delta >= ATM_DELTA_THRESHOLD:
            return Moneyness.ATM
        if abs_delta >= OTM_DELTA_THRESHOLD:
            return Moneyness.OTM
        return Moneyness.DEEP_OTM

    def estimate_slippage(
        self,
        strike: float,
        spot: float,
        option_type: str,
        bid: float,
        ask: float,
        delta: Optional[float] = None,
    ) -> SlippageEstimate:
        """Estimate expected slippage for an option order.

        Parameters
        ----------
        strike, spot:
            Strike and underlying spot price.
        option_type:
            ``"CE"`` or ``"PE"``.
        bid, ask:
            Current best bid and ask prices.
        delta:
            Option delta.  If *None* a rough proxy is derived from
            strike/spot distance (less accurate).

        Returns
        -------
        SlippageEstimate
        """
        if delta is None:
            delta = self._approximate_delta(strike, spot, option_type)

        moneyness = self.classify_moneyness(delta)
        slippage_lo, slippage_hi = self._SLIPPAGE_BANDS[moneyness]

        # Interpolate within the band using spread width as a signal
        mid = (bid + ask) / 2.0 if (bid + ask) > 0 else 1.0
        spread_pct = (ask - bid) / mid if mid > 0 else 0.0

        # Wider spread -> closer to high end of slippage band
        spread_ratio = min(spread_pct / 0.02, 1.0)  # normalise against 2% spread
        slippage_pct = slippage_lo + (slippage_hi - slippage_lo) * spread_ratio

        order_type = self.select_order_type(delta)

        estimate = SlippageEstimate(
            moneyness=moneyness,
            slippage_pct=round(slippage_pct, 6),
            recommended_order_type=order_type,
            spread_pct=round(spread_pct, 6),
            delta=delta,
        )

        logger.debug(
            "Slippage estimate | strike=%.0f spot=%.0f type=%s delta=%.3f "
            "moneyness=%s slippage=%.4f%% order_type=%s spread=%.4f%%",
            strike,
            spot,
            option_type,
            delta,
            moneyness.value,
            slippage_pct * 100,
            order_type,
            spread_pct * 100,
        )
        return estimate

    def select_order_type(self, delta: float) -> str:
        """Choose ``"MARKET"`` or ``"LIMIT"`` based on *delta*.

        * delta >= 0.40  -> MARKET  (liquid ATM strikes)
        * delta < 0.40   -> LIMIT   (OTM, less liquid)
        * delta < 0.15   -> LIMIT   (mandatory for deep OTM)
        """
        abs_delta = abs(delta)
        if abs_delta >= ATM_DELTA_THRESHOLD:
            return "MARKET"
        return "LIMIT"

    def calculate_limit_price(
        self,
        side: str,
        bid: float,
        ask: float,
        delta: float,
    ) -> float:
        """Calculate an aggressive-yet-safe limit price.

        Parameters
        ----------
        side:
            ``"BUY"`` or ``"SELL"``.
        bid, ask:
            Current best bid / ask.
        delta:
            Option delta used to determine the buffer size.

        Returns
        -------
        float
            Rounded limit price (to 0.05 tick — NSE options tick size).
        """
        moneyness = self.classify_moneyness(delta)
        buffer_mult = self._BUFFER_MULTIPLIERS[moneyness]
        spread = ask - bid

        if side.upper() == "BUY":
            # Willing to pay slightly above ask for fills on illiquid strikes
            raw = ask + spread * buffer_mult
        else:
            # Willing to sell slightly below bid
            raw = bid - spread * buffer_mult

        # Round to nearest NSE tick (0.05)
        limit_price = round(round(raw / 0.05) * 0.05, 2)
        limit_price = max(limit_price, 0.05)  # floor at minimum tick

        logger.debug(
            "Limit price | side=%s bid=%.2f ask=%.2f delta=%.3f "
            "moneyness=%s buffer_mult=%.2f -> %.2f",
            side,
            bid,
            ask,
            delta,
            moneyness.value,
            buffer_mult,
            limit_price,
        )
        return limit_price

    # ── Internals ────────────────────────────────────────────────────────

    @staticmethod
    def _approximate_delta(
        strike: float,
        spot: float,
        option_type: str,
    ) -> float:
        """Rough delta proxy when Greeks are unavailable.

        Uses a simple logistic-style mapping of moneyness ratio — good
        enough for order-type selection, NOT for pricing.
        """
        if option_type.upper() == "CE":
            moneyness_ratio = spot / strike
        else:
            moneyness_ratio = strike / spot

        # Simple sigmoid-ish mapping: ratio ~1 -> delta ~0.50
        import math

        x = (moneyness_ratio - 1.0) * 20.0  # scale
        approx = 1.0 / (1.0 + math.exp(-x))
        return round(approx, 4)
