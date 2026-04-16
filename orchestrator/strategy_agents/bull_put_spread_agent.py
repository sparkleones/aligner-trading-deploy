"""Bull Put Spread live strategy agent — realistic Indian market design.

Indian Market Reality for Bull Put Spreads:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Put selling is the bread-and-butter of professional Indian options desks
2. OTM puts have inflated IV (fear premium) → sellers get paid MORE than fair value
3. NIFTY has institutional put support at round numbers (23000, 23500, etc.)
4. PCR > 1.2 means heavy put writing → strong support below
5. FII selling creates oversold bounces → ideal for bull put entry
6. Best entries: RSI oversold + market analysis bullish + near OI support
7. Exit at 50% profit (don't wait for expiry — theta decays fastest early)
8. Stop at 2x credit or strong bearish breakout

Strike Selection:
  Sell put at ~0.8-1.0 SD below spot (aligns with OI support when available)
  Buy put 1 strike interval below (defined risk)

Global Market Factors:
  - US futures green overnight → bullish gap → sell puts
  - DXY falling → FII inflows → supportive for NIFTY
  - Crude oil falling → positive for India (net importer)
"""

import logging
import math
from datetime import datetime
from typing import Optional, TYPE_CHECKING

from orchestrator.trade_signal import OrderLeg, TradeSignal
from orchestrator.strategy_agents.base_agent import BaseLiveAgent

if TYPE_CHECKING:
    from orchestrator.market_analyzer import MarketAnalysis

logger = logging.getLogger(__name__)

STRIKE_INTERVAL = 50


class BullPutSpreadLiveAgent(BaseLiveAgent):
    name = "bull_put_spread"

    def __init__(
        self,
        capital: float = 25000.0,
        lot_size: int = 65,  # NIFTY lot size as of SEBI Feb 2026
        rsi_entry: float = 45.0,    # enter when RSI drops below this
        rsi_exit: float = 58.0,     # exit when RSI recovers above this
    ):
        super().__init__(capital, lot_size)
        self.rsi_entry = rsi_entry
        self.rsi_exit = rsi_exit

        # Track entry metrics for profit-taking / stop-loss
        self._entry_credit_per_unit: float = 0.0
        self._entry_bar_idx: int = 0

        # Capital-aware: max affordable spread width
        max_risk = capital * 0.20
        self._max_spread_points = max_risk / lot_size if lot_size > 0 else 200
        logger.info(
            "BullPutSpread: max affordable spread = %.0f pts (capital=%g, lot=%d)",
            self._max_spread_points, capital, lot_size,
        )

    def _compute_rsi(self, period: int = 14) -> float:
        """Compute RSI from bars buffer."""
        if len(self._bars) < period + 1:
            return 50.0
        closes = [b["close"] for b in self._bars[-(period + 1):]]
        deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
        gains = [d for d in deltas if d > 0]
        losses = [-d for d in deltas if d < 0]
        avg_gain = sum(gains) / period if gains else 0
        avg_loss = sum(losses) / period if losses else 0.0001
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    def generate_signal(
        self,
        bar: dict,
        bar_idx: int,
        option_chain: Optional[dict] = None,
        market_analysis: Optional["MarketAnalysis"] = None,
    ) -> Optional[TradeSignal]:
        self.add_bar(bar)

        if len(self._bars) < 15:
            return None

        spot = bar["close"]
        rsi = self._compute_rsi()

        if market_analysis is not None:
            return self._signal_with_analysis(spot, rsi, bar_idx, option_chain, market_analysis)

        # Fallback
        if not self._position_open and rsi < self.rsi_entry:
            return self._build_entry_signal(spot, rsi, bar_idx, option_chain)
        if self._position_open and rsi > self.rsi_exit:
            return TradeSignal(
                strategy=self.name, action="CLOSE", confidence=0.7,
                underlying_price=spot, timestamp=datetime.now(),
                reasoning=f"RSI recovered to {rsi:.1f}",
            )
        return None

    def _signal_with_analysis(
        self,
        spot: float,
        rsi: float,
        bar_idx: int,
        option_chain: Optional[dict],
        ma: "MarketAnalysis",
    ) -> Optional[TradeSignal]:
        """MarketAnalysis-driven bull put spread with professional exits."""
        from orchestrator.market_analyzer import VIXRegime, MarketBias, TradeAction

        # ── EXIT CONDITIONS ───────────────────────────────────────────
        if self._position_open:
            # 1. RSI recovery exit
            if rsi > self.rsi_exit:
                return TradeSignal(
                    strategy=self.name, action="CLOSE", confidence=0.75,
                    underlying_price=spot, timestamp=datetime.now(),
                    reasoning=f"RSI recovered {rsi:.1f} + bias={ma.market_bias.value}",
                )

            # 2. Strong bearish breakout — cut losses immediately
            if ma.market_bias == MarketBias.STRONG_BEARISH:
                return TradeSignal(
                    strategy=self.name, action="CLOSE", confidence=0.9,
                    underlying_price=spot, timestamp=datetime.now(),
                    reasoning=f"STRONG BEARISH breakout — cut loss on bull spread",
                )

            # 3. VIX spike to EXTREME
            if ma.vix_regime == VIXRegime.EXTREME:
                return TradeSignal(
                    strategy=self.name, action="CLOSE", confidence=0.9,
                    underlying_price=spot, timestamp=datetime.now(),
                    reasoning=f"VIX extreme ({ma.vix:.1f}) — exit all credit sells",
                )

            # 4. Time-based exit after ~200 bars (~3.5 hours) if still open
            bars_held = bar_idx - self._entry_bar_idx
            if bars_held > 200:
                return TradeSignal(
                    strategy=self.name, action="CLOSE", confidence=0.7,
                    underlying_price=spot, timestamp=datetime.now(),
                    reasoning=f"Time exit: held {bars_held} bars",
                )

            return None

        # ── ENTRY CONDITIONS ──────────────────────────────────────────
        # BACKTEST LEARNING (Oct25-Mar26): Bull put spreads lost -53% in
        # declining market. Only enter when market bias is genuinely bullish
        # or neutral AND VIX is not elevated. Selling puts into fear = death.
        if not ma.timing_ok:
            return None

        # Don't enter late in the day
        if bar_idx > 300:
            return None

        # VIX must not be HIGH or EXTREME — selling puts into fear is suicide
        # Backtest showed: HIGH VIX = -Rs 188/trade avg, even LOW VIX = -Rs 101
        # Only enter when VIX < 20 (calm markets favor put sellers)
        if ma.vix_regime in (VIXRegime.HIGH, VIXRegime.EXTREME):
            return None

        # Market must be BULLISH or NEUTRAL — NEVER sell puts in bearish market
        # Backtest: selling puts into a declining NIFTY = guaranteed loss
        if ma.market_bias in (MarketBias.STRONG_BEARISH, MarketBias.BEARISH):
            return None

        # Trend filter: price must be above recent bars (short-term uptrend)
        # This was the #1 missing filter — selling puts requires bullish trend
        if len(self._bars) >= 10:
            recent_closes = [b["close"] for b in self._bars[-10:]]
            ema10 = sum(recent_closes) / len(recent_closes)
            if bar["close"] < ema10:
                return None  # Don't sell puts below EMA10

        # Entry triggers (need at least one)
        rsi_trigger = rsi < self.rsi_entry
        analysis_trigger = ma.recommended_action == TradeAction.SELL_PUT_SPREAD

        if not rsi_trigger and not analysis_trigger:
            return None

        # Confidence calculation
        confidence = 0.5

        # RSI oversold boost
        if rsi_trigger:
            confidence += min(0.15, (self.rsi_entry - rsi) / 40)

        # Bullish bias boost
        if ma.market_bias in (MarketBias.BULLISH, MarketBias.STRONG_BULLISH):
            confidence += 0.1

        # PCR bullish confirmation (PCR < 0.9 = more calls = bullish positioning)
        if ma.pcr < 0.9:
            confidence += 0.05

        # Near OI support (put wall below = safety net for put sellers)
        if ma.oi_support > 0:
            dist_to_support = (spot - ma.oi_support) / spot
            if dist_to_support < 0.02:
                confidence += 0.1

        # IV percentile boost (rich premiums = better credit)
        if ma.iv_percentile > 60:
            confidence += 0.05

        # Analyzer agreement
        if analysis_trigger:
            confidence += 0.1

        # VIX HIGH = richer premiums for sellers (good, not bad)
        if ma.vix_regime == VIXRegime.HIGH:
            confidence += 0.05

        confidence = min(0.95, confidence)
        if confidence < 0.55:
            return None

        # Smart strike selection: align with OI support when possible
        # VIX-adaptive: at higher VIX, sell further OTM for safety
        # VIX 10-15: sell at 0.8 SD (tight, more credit)
        # VIX 15-22: sell at 1.0 SD (balanced)
        # VIX 22+:   sell at 1.2 SD (wider, safer but less credit)
        if ma.vix < 15:
            sell_sd_mult = 0.8
        elif ma.vix < 22:
            sell_sd_mult = 1.0
        else:
            sell_sd_mult = 1.2

        if ma.oi_support > 0:
            support_otm = (spot - ma.oi_support) / spot
            if 0.005 < support_otm < 0.03:
                # Place sell strike near OI support (put wall protects us)
                sell_sd_mult = support_otm / (self._expected_move_pct(ma) or 0.01)

        extra = f"VIX={ma.vix:.1f}({ma.vix_regime.value}) bias={ma.market_bias.value} PCR={ma.pcr:.2f}"
        self._entry_bar_idx = bar_idx

        return self._build_entry_signal(
            spot, rsi, bar_idx, option_chain,
            confidence_override=confidence,
            sell_sd_mult=sell_sd_mult,
            ma=ma,
            extra_reasoning=extra,
        )

    def _expected_move_pct(self, ma: Optional["MarketAnalysis"]) -> float:
        """Expected move as percentage of spot."""
        vix = 22.0  # Default for Apr 2026 elevated VIX regime
        if ma and hasattr(ma, 'vix'):
            vix = ma.vix
        atm_iv = vix / 100.0 * 0.88
        dte_days = 2.0
        return atm_iv * math.sqrt(dte_days / 365.0)

    def _build_entry_signal(
        self,
        spot: float,
        rsi: float,
        bar_idx: int,
        option_chain: Optional[dict],
        confidence_override: Optional[float] = None,
        sell_sd_mult: float = 0.8,
        ma: Optional["MarketAnalysis"] = None,
        extra_reasoning: str = "",
    ) -> TradeSignal:
        """Build a 2-leg bull put spread with SD-based strike selection."""
        # SD-based strike distance
        vix = 22.0  # Default for Apr 2026 elevated VIX regime
        if ma and hasattr(ma, 'vix'):
            vix = ma.vix
        atm_iv = vix / 100.0 * 0.88
        dte_days = 2.0
        expected_move = spot * atm_iv * math.sqrt(dte_days / 365.0)
        sell_distance = expected_move * sell_sd_mult

        sell_strike = round((spot - sell_distance) / STRIKE_INTERVAL) * STRIKE_INTERVAL
        buy_strike = sell_strike - STRIKE_INTERVAL  # 1 strike below for protection

        # Capital constraint
        spread_width = sell_strike - buy_strike
        if spread_width > self._max_spread_points:
            buy_strike = sell_strike - STRIKE_INTERVAL

        spread_width = sell_strike - buy_strike
        # Put premiums are inflated by fear premium (~30-40% of width for 0.8 SD OTM)
        estimated_credit = spread_width * 0.30
        max_loss_per_unit = spread_width - estimated_credit
        max_loss = max_loss_per_unit * self._order_qty
        self._entry_credit_per_unit = estimated_credit

        def _sym(strike: float, opt_type: str) -> str:
            return self.resolve_symbol(strike, opt_type, option_chain)

        legs = [
            OrderLeg(symbol=_sym(sell_strike, "PE"), side="SELL", qty=self._order_qty,
                     option_type="PE", strike=sell_strike),
            OrderLeg(symbol=_sym(buy_strike, "PE"), side="BUY", qty=self._order_qty,
                     option_type="PE", strike=buy_strike),
        ]

        confidence = confidence_override or min(0.85, 0.5 + (self.rsi_entry - rsi) / 40)

        reasoning = (
            f"RSI={rsi:.1f} | SD-based spread: {buy_strike}/{sell_strike}PE | "
            f"exp_move={expected_move:.0f}pts"
        )
        if extra_reasoning:
            reasoning += f" | {extra_reasoning}"

        return TradeSignal(
            strategy=self.name,
            action="BULL_PUT_SPREAD",
            confidence=confidence,
            underlying_price=spot,
            timestamp=datetime.now(),
            reasoning=reasoning,
            legs=legs,
            estimated_credit=round(estimated_credit * self._order_qty, 2),
            max_loss=round(max_loss, 2),
        )
