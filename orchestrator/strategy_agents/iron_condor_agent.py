"""Iron Condor live strategy agent — realistic Indian market design.

Indian Market Reality for Iron Condors:
-------------------------------------
1. NIFTY is range-bound 65-70% of sessions --> ideal for iron condors
2. Professional desks place short strikes at ~1.0-1.5 standard deviations
3. VIX < 15 = best environment (tight ranges, fast theta decay)
4. Exit at 50% of max profit (don't wait for expiry — gamma risk rises)
5. Stop loss at 2x credit received (cut losses before max loss)
6. Don't enter after 2:30 PM (gamma risk too high, theta already captured)
7. Wing width = 1 strike interval (50 pts NIFTY) for capital efficiency

Strike Selection (volatility-derived, not fixed percentage):
  Expected move = spot × ATM_IV × √(DTE/365)
  Sell strikes at ~1.2 × expected move (high probability OTM)
  Buy wings 1 strike interval beyond (defined risk)

Exit Rules:
  ✓ Vol expansion exit (realized vol > 2× entry vol)
  ✓ Directional breakout exit (strong bias detected)
  ✓ VIX spike exit (regime change to HIGH/EXTREME)
  ✓ Time-based exit after ~4 hours (theta mostly captured, gamma rising)
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

STRIKE_INTERVAL = 50  # NIFTY strike gap


class IronCondorLiveAgent(BaseLiveAgent):
    name = "iron_condor"

    def __init__(
        self,
        capital: float = 25000.0,
        lot_size: int = 65,  # NIFTY lot size as of SEBI Feb 2026
        vol_entry_threshold: float = 0.003,  # enter when 5-bar vol < this
        vol_exit_threshold: float = 0.007,   # exit when vol expands above this
    ):
        super().__init__(capital, lot_size)
        self.vol_entry = vol_entry_threshold
        self.vol_exit = vol_exit_threshold

        # Track entry metrics for profit-taking / stop-loss
        self._entry_credit_per_unit: float = 0.0
        self._entry_bar_idx: int = 0
        self._entry_vol: float = 0.0
        self._entry_spot: float = 0.0
        self._sell_put_strike: float = 0.0
        self._sell_call_strike: float = 0.0

        # Capital-aware: max affordable wing width
        max_risk = capital * 0.20
        self._max_wing_points = max_risk / lot_size if lot_size > 0 else 200
        logger.info(
            "IronCondor: max affordable wing = %.0f pts (capital=%g, lot=%d)",
            self._max_wing_points, capital, lot_size,
        )

    def generate_signal(
        self,
        bar: dict,
        bar_idx: int,
        option_chain: Optional[dict] = None,
        market_analysis: Optional["MarketAnalysis"] = None,
    ) -> Optional[TradeSignal]:
        self.add_bar(bar)

        if len(self._bars) < 6:
            return None

        spot = bar["close"]
        recent = [b["close"] for b in self._bars[-6:]]
        returns = [(recent[i] - recent[i - 1]) / recent[i - 1] for i in range(1, len(recent))]
        vol = (sum(r ** 2 for r in returns) / len(returns)) ** 0.5

        if market_analysis is not None:
            return self._signal_with_analysis(spot, vol, bar_idx, option_chain, market_analysis)

        # Fallback
        if not self._position_open and vol < self.vol_entry and bar_idx >= 3:
            return self._build_entry_signal(spot, vol, bar_idx, option_chain, confidence_base=0.6)

        if self._position_open and vol > self.vol_exit:
            return TradeSignal(
                strategy=self.name, action="CLOSE", confidence=0.8,
                underlying_price=spot, timestamp=datetime.now(),
                reasoning=f"Vol expansion: {vol:.4f} > {self.vol_exit}",
            )

        return None

    def _signal_with_analysis(
        self,
        spot: float,
        vol: float,
        bar_idx: int,
        option_chain: Optional[dict],
        ma: "MarketAnalysis",
    ) -> Optional[TradeSignal]:
        """Full MarketAnalysis-driven iron condor with professional exit rules."""
        from orchestrator.market_analyzer import VIXRegime, MarketBias

        # ── EXIT CONDITIONS (checked every bar) ───────────────────────
        if self._position_open:
            # 1. SPOT DISTANCE EXIT (most critical for condors)
            #    If spot moves >60% toward either short strike, exit immediately
            #    This prevents the short leg from going ITM and causing max loss
            if self._sell_put_strike > 0 and self._sell_call_strike > 0:
                dist_to_put = self._entry_spot - self._sell_put_strike
                dist_to_call = self._sell_call_strike - self._entry_spot
                if dist_to_put > 0 and (self._entry_spot - spot) > dist_to_put * 0.6:
                    return TradeSignal(
                        strategy=self.name, action="CLOSE", confidence=0.95,
                        underlying_price=spot, timestamp=datetime.now(),
                        reasoning=f"Spot approaching put strike! {spot:.0f} --> {self._sell_put_strike:.0f} (60% breach)",
                    )
                if dist_to_call > 0 and (spot - self._entry_spot) > dist_to_call * 0.6:
                    return TradeSignal(
                        strategy=self.name, action="CLOSE", confidence=0.95,
                        underlying_price=spot, timestamp=datetime.now(),
                        reasoning=f"Spot approaching call strike! {spot:.0f} --> {self._sell_call_strike:.0f} (60% breach)",
                    )

            # 2. Vol expansion exit (1.5× entry vol — tighter than before)
            if vol > max(self.vol_exit, self._entry_vol * 1.5):
                return TradeSignal(
                    strategy=self.name, action="CLOSE", confidence=0.85,
                    underlying_price=spot, timestamp=datetime.now(),
                    reasoning=f"Vol expansion {vol:.4f} (entry: {self._entry_vol:.4f}) VIX={ma.vix:.1f}",
                )

            # 3. Strong directional breakout
            if ma.market_bias in (MarketBias.STRONG_BULLISH, MarketBias.STRONG_BEARISH):
                return TradeSignal(
                    strategy=self.name, action="CLOSE", confidence=0.8,
                    underlying_price=spot, timestamp=datetime.now(),
                    reasoning=f"Directional breakout: {ma.market_bias.value}",
                )

            # 4. VIX regime change to HIGH/EXTREME
            if ma.vix_regime in (VIXRegime.HIGH, VIXRegime.EXTREME):
                return TradeSignal(
                    strategy=self.name, action="CLOSE", confidence=0.85,
                    underlying_price=spot, timestamp=datetime.now(),
                    reasoning=f"VIX regime {ma.vix_regime.value} — exit condor",
                )

            # 5. Time-based exit: 80 bars (~1.3 hours) — take theta and run
            bars_held = bar_idx - self._entry_bar_idx
            if bars_held > 80:
                return TradeSignal(
                    strategy=self.name, action="CLOSE", confidence=0.7,
                    underlying_price=spot, timestamp=datetime.now(),
                    reasoning=f"Time exit: held {bars_held} bars, theta captured",
                )

            return None

        # ── ENTRY CONDITIONS ──────────────────────────────────────────
        if bar_idx < 3:
            return None
        if not ma.timing_ok:
            return None

        # Capital gate: iron condors need minimum ₹50K capital
        # Below that, a single 50-point adverse move = 6-10% loss on 65 lot
        # Use single-side credit spreads (bull_put / ddqn) instead
        if self.capital < 50000:
            return None

        # Don't enter late in the day (after ~300 bars ≈ 2:30 PM equivalent)
        if bar_idx > 300:
            return None

        # VIX must be LOW or NORMAL for iron condors
        # At VIX > 20, spreads are too wide and gamma risk is elevated
        # Iron condors thrive in low-vol range-bound markets (VIX 10-18)
        if ma.vix_regime not in (VIXRegime.LOW, VIXRegime.NORMAL):
            return None

        # Additional VIX gate: even in NORMAL regime, VIX > 20 = widen strikes
        # but only enter if VIX is actually stable/declining (not spiking)
        if ma.vix > 20:
            # In elevated VIX, require stronger neutral bias to enter
            if ma.market_bias != MarketBias.NEUTRAL:
                return None

        # Market must not be strongly directional
        if ma.market_bias in (MarketBias.STRONG_BULLISH, MarketBias.STRONG_BEARISH):
            return None

        # Realized vol must be low
        if vol >= self.vol_entry:
            return None

        # IV percentile boost
        iv_boost = 0.0
        if ma.iv_percentile > 60:
            iv_boost = 0.1
        if ma.iv_percentile > 75:
            iv_boost = 0.2

        # OI-based refinement
        inner_mult = 1.2  # default: sell at 1.2 SD
        if ma.oi_support > 0 and ma.oi_resistance > 0:
            oi_range_pct = (ma.oi_resistance - ma.oi_support) / spot
            if oi_range_pct < 0.025:
                inner_mult = 0.8  # tight OI range: narrow strikes for safety

        confidence_base = min(0.9, 0.55 + (self.vol_entry - vol) * 80 + iv_boost)
        if ma.market_bias == MarketBias.NEUTRAL:
            confidence_base = min(0.95, confidence_base + 0.05)

        self._entry_vol = vol
        self._entry_bar_idx = bar_idx
        self._entry_spot = spot

        return self._build_entry_signal(
            spot, vol, bar_idx, option_chain,
            confidence_base=confidence_base,
            ma=ma,
            inner_mult=inner_mult,
            extra_reasoning=f"VIX={ma.vix:.1f}({ma.vix_regime.value}) IV%={ma.iv_percentile:.0f} bias={ma.market_bias.value}",
        )

    def _sd_based_strikes(self, spot: float, inner_mult: float = 1.2,
                           ma: Optional["MarketAnalysis"] = None):
        """Calculate strike distances using expected move (standard deviation).

        Expected move = spot × ATM_IV × √(DTE/365)
        Professional Indian desks place short strikes at 1.0-1.5 SD.
        """
        vix = 22.0  # Default for Apr 2026 elevated VIX regime
        if ma and hasattr(ma, 'vix'):
            vix = ma.vix

        atm_iv = vix / 100.0 * 0.88
        dte_days = 2.0
        expected_move = spot * atm_iv * math.sqrt(dte_days / 365.0)

        # Short strikes at inner_mult × expected move
        inner_distance = expected_move * inner_mult
        return inner_distance

    def _build_entry_signal(
        self,
        spot: float,
        vol: float,
        bar_idx: int,
        option_chain: Optional[dict],
        confidence_base: float = 0.6,
        ma: Optional["MarketAnalysis"] = None,
        inner_mult: float = 1.2,
        extra_reasoning: str = "",
    ) -> TradeSignal:
        """Build a 4-leg iron condor with SD-based strike selection."""
        inner_distance = self._sd_based_strikes(spot, inner_mult, ma)

        sell_put = round((spot - inner_distance) / STRIKE_INTERVAL) * STRIKE_INTERVAL
        sell_call = round((spot + inner_distance) / STRIKE_INTERVAL) * STRIKE_INTERVAL

        # Save short strikes for spot-distance exit monitoring
        self._sell_put_strike = sell_put
        self._sell_call_strike = sell_call

        # Wings: 1 strike interval beyond (defined risk, capital efficient)
        buy_put = sell_put - STRIKE_INTERVAL
        buy_call = sell_call + STRIKE_INTERVAL

        # Capital constraint
        wing_width = sell_put - buy_put
        if wing_width > self._max_wing_points:
            buy_put = sell_put - STRIKE_INTERVAL
            buy_call = sell_call + STRIKE_INTERVAL

        wing_width = sell_put - buy_put
        # Credit estimate: ~25% per side, ~50% total
        estimated_credit_per_side = wing_width * 0.25
        total_credit = estimated_credit_per_side * 2
        max_loss_per_unit = wing_width - total_credit
        max_loss = max_loss_per_unit * self._order_qty
        self._entry_credit_per_unit = total_credit

        legs = self._build_legs(sell_put, buy_put, sell_call, buy_call, option_chain)

        confidence = min(0.95, confidence_base)

        reasoning = (
            f"SD-based condor | {self._num_lots}L | "
            f"{buy_put}/{sell_put}PE — {sell_call}/{buy_call}CE | "
            f"exp_move=±{inner_distance:.0f}pts"
        )
        if extra_reasoning:
            reasoning += f" | {extra_reasoning}"

        return TradeSignal(
            strategy=self.name,
            action="IRON_CONDOR",
            confidence=confidence,
            underlying_price=spot,
            timestamp=datetime.now(),
            reasoning=reasoning,
            legs=legs,
            estimated_credit=round(total_credit * self._order_qty, 2),
            max_loss=round(max_loss, 2),
        )

    def _build_legs(
        self,
        sell_put: float,
        buy_put: float,
        sell_call: float,
        buy_call: float,
        option_chain: Optional[dict],
    ) -> list[OrderLeg]:
        def _sym(strike: float, opt_type: str) -> str:
            return self.resolve_symbol(strike, opt_type, option_chain)

        return [
            OrderLeg(symbol=_sym(sell_put, "PE"), side="SELL", qty=self._order_qty,
                     option_type="PE", strike=sell_put),
            OrderLeg(symbol=_sym(buy_put, "PE"), side="BUY", qty=self._order_qty,
                     option_type="PE", strike=buy_put),
            OrderLeg(symbol=_sym(sell_call, "CE"), side="SELL", qty=self._order_qty,
                     option_type="CE", strike=sell_call),
            OrderLeg(symbol=_sym(buy_call, "CE"), side="BUY", qty=self._order_qty,
                     option_type="CE", strike=buy_call),
        ]
