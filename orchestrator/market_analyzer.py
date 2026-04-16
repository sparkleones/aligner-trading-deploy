"""
Comprehensive Indian Options Market Analyzer.

Combines 12+ indicators specific to Indian NSE options trading:
1. India VIX regime detection
2. Put-Call Ratio (PCR) sentiment
3. Max Pain calculation & proximity
4. Open Interest (OI) support/resistance levels
5. OI change analysis (buildup vs unwinding)
6. IV Percentile/Rank for premium pricing
7. IV Skew analysis (put vs call skew)
8. FII/DII flow impact
9. Multi-timeframe trend (EMA stack)
10. RSI + divergence detection
11. VWAP proximity for intraday bias
12. Supertrend confirmation
13. Intraday timing rules (avoid first 15 min, theta window)
14. Expiry day logic (Max Pain convergence)

Each indicator produces a score from -1.0 (strong bearish) to +1.0 (strong bullish).
The final decision aggregates all scores with configurable weights.
"""

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, time as dt_time, date
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

from config.constants import (
    VIX_LOW, VIX_NORMAL_LOW, VIX_NORMAL_HIGH, VIX_HIGH, VIX_EXTREME,
    PCR_OVERSOLD, PCR_BULLISH, PCR_NEUTRAL_LOW, PCR_NEUTRAL_HIGH,
    PCR_BEARISH, PCR_OVERBOUGHT,
    MAX_PAIN_RELIABILITY_RANGE, MAX_PAIN_VIX_THRESHOLD,
    MAX_PAIN_ENTRY_AFTER_HOUR, MAX_PAIN_ENTRY_AFTER_MINUTE,
    OI_SUPPORT_THRESHOLD, OI_RESISTANCE_THRESHOLD,
    IV_PERCENTILE_HIGH, IV_PERCENTILE_LOW, IV_SKEW_THRESHOLD,
    AVOID_ENTRY_UNTIL_HOUR, AVOID_ENTRY_UNTIL_MINUTE,
    THETA_ACCELERATION_HOUR, THETA_ACCELERATION_MINUTE,
    INDEX_CONFIG,
)

logger = logging.getLogger(__name__)


# ── Enums ────────────────────────────────────────────────────────────────────

class VIXRegime(Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    EXTREME = "extreme"


class MarketBias(Enum):
    STRONG_BULLISH = "strong_bullish"
    BULLISH = "bullish"
    NEUTRAL = "neutral"
    BEARISH = "bearish"
    STRONG_BEARISH = "strong_bearish"


class TradeAction(Enum):
    SELL_STRADDLE = "sell_straddle"          # Low VIX, sideways
    SELL_IRON_CONDOR = "sell_iron_condor"    # Moderate VIX, range
    SELL_PUT_SPREAD = "sell_put_spread"      # Bullish bias
    SELL_CALL_SPREAD = "sell_call_spread"    # Bearish bias
    BUY_CALL = "buy_call"                   # Strong bullish
    BUY_PUT = "buy_put"                     # Strong bearish
    BUY_STRADDLE = "buy_straddle"           # High VIX, big move expected
    HOLD = "hold"                            # No clear edge


@dataclass
class IndicatorScore:
    """A single indicator's contribution to the trading decision."""
    name: str
    score: float           # -1.0 (bearish) to +1.0 (bullish)
    confidence: float      # 0.0 to 1.0
    weight: float          # importance weight
    reasoning: str


@dataclass
class MarketAnalysis:
    """Complete market analysis result."""
    timestamp: datetime
    spot_price: float
    vix: float
    vix_regime: VIXRegime
    pcr: float
    max_pain: float
    max_pain_distance: float    # spot - max_pain (positive = above)
    oi_support: float           # nearest strong put OI level
    oi_resistance: float        # nearest strong call OI level
    iv_percentile: float
    iv_skew: float              # put_iv - call_iv (positive = put skew)
    market_bias: MarketBias
    recommended_action: TradeAction
    overall_score: float        # -1.0 to +1.0
    confidence: float           # 0.0 to 1.0
    indicators: list[IndicatorScore] = field(default_factory=list)
    reasoning: str = ""
    is_expiry_day: bool = False
    timing_ok: bool = True      # False during avoid-entry windows
    # ── FII/DII institutional flows ──
    fii_net: float = 0.0        # FII net buy/sell in Rs crore (negative = selling)
    dii_net: float = 0.0        # DII net buy/sell in Rs crore
    # ── Volume Profile (stub for future data hookup) ──
    poc: float = 0.0            # Point of Control (price with highest volume)
    vah: float = 0.0            # Value Area High (upper 70% volume boundary)
    val: float = 0.0            # Value Area Low (lower 70% volume boundary)
    # ── Crude oil correlation ──
    crude_oil_price: float = 0.0  # Brent/WTI price in USD/bbl
    # ── OI Change tracking (delta from previous snapshot) ──
    oi_ce_change_pct: float = 0.0   # % change in total Call OI from previous snapshot
    oi_pe_change_pct: float = 0.0   # % change in total Put OI from previous snapshot


# ── Market Analyzer ──────────────────────────────────────────────────────────

class MarketAnalyzer:
    """Comprehensive Indian options market analyzer.

    Combines all indicators into a single actionable decision.

    Parameters
    ----------
    symbol : str
        Underlying symbol (NIFTY, BANKNIFTY, FINNIFTY).
    capital : float
        Available capital (affects position sizing recommendations).
    """

    # Indicator weights — tuned for Indian options intraday
    WEIGHTS = {
        "vix_regime": 0.15,        # VIX is king for options
        "pcr": 0.12,               # Sentiment confirmation
        "max_pain": 0.10,          # Expiry day convergence
        "oi_levels": 0.12,         # Support/resistance from OI
        "oi_change": 0.08,         # Fresh buildup signals
        "iv_percentile": 0.10,     # Premium richness
        "iv_skew": 0.05,           # Directional skew
        "trend_ema": 0.10,         # Multi-TF trend
        "rsi": 0.06,               # Overbought/oversold
        "vwap": 0.05,              # Intraday bias
        "supertrend": 0.05,        # Trend confirmation
        "timing": 0.02,            # Intraday timing
    }

    def __init__(self, symbol: str = "NIFTY", capital: float = 25000.0, is_paper: bool = False):
        self.symbol = symbol
        self.capital = capital
        self.is_paper = is_paper  # bypass timing checks in paper/backtest mode
        self.lot_size = INDEX_CONFIG.get(symbol, {}).get("lot_size", 75)
        self._bars: list[dict] = []
        self._vix_history: list[float] = []
        self._oi_history: list[dict] = []
        # OI change tracking (delta from previous snapshot)
        self._prev_total_ce_oi: float = 0.0
        self._prev_total_pe_oi: float = 0.0

    def add_bar(self, bar: dict) -> None:
        """Add an OHLCV bar to the internal buffer."""
        self._bars.append(bar)
        if len(self._bars) > 500:
            self._bars = self._bars[-500:]

    def add_vix(self, vix: float) -> None:
        """Add a VIX reading."""
        self._vix_history.append(vix)
        if len(self._vix_history) > 500:
            self._vix_history = self._vix_history[-500:]

    def analyze(
        self,
        spot_price: float,
        vix: float = 14.0,
        pcr: float = 1.0,
        option_chain: Optional[dict] = None,
        fii_net: float = 0.0,
        dii_net: float = 0.0,
        is_expiry_day: bool = False,
    ) -> MarketAnalysis:
        """Run full market analysis and return recommendation.

        Parameters
        ----------
        spot_price : float
            Current underlying price.
        vix : float
            India VIX value.
        pcr : float
            Put-Call Ratio (OI based).
        option_chain : dict, optional
            Strike -> {"CE": {"oi": ..., "iv": ..., "ltp": ...},
                        "PE": {"oi": ..., "iv": ..., "ltp": ...}}
        fii_net : float
            FII net buy/sell in Rs crore (negative = selling).
        dii_net : float
            DII net buy/sell in Rs crore.
        is_expiry_day : bool
            Whether today is the weekly expiry.

        Returns
        -------
        MarketAnalysis
            Complete analysis with actionable recommendation.
        """
        indicators: list[IndicatorScore] = []

        # 1. VIX Regime
        indicators.append(self._analyze_vix(vix))

        # 2. Put-Call Ratio
        indicators.append(self._analyze_pcr(pcr))

        # 3. Max Pain
        max_pain = self._calculate_max_pain(option_chain, spot_price)
        indicators.append(self._analyze_max_pain(spot_price, max_pain, vix, is_expiry_day))

        # 4. OI Support/Resistance
        oi_support, oi_resistance = self._find_oi_levels(option_chain, spot_price)
        indicators.append(self._analyze_oi_levels(spot_price, oi_support, oi_resistance))

        # 5. OI Change (buildup/unwinding) — now returns (score, ce_change%, pe_change%)
        oi_change_result = self._analyze_oi_change(option_chain)
        oi_change_indicator, oi_ce_change_pct, oi_pe_change_pct = oi_change_result
        indicators.append(oi_change_indicator)

        # 6. IV Percentile
        iv_percentile, iv_skew = self._analyze_iv(option_chain, spot_price)
        indicators.append(self._score_iv_percentile(iv_percentile))

        # 7. IV Skew
        indicators.append(self._score_iv_skew(iv_skew))

        # 8. Multi-TF Trend (EMA stack)
        indicators.append(self._analyze_trend())

        # 9. RSI
        indicators.append(self._analyze_rsi())

        # 10. VWAP
        indicators.append(self._analyze_vwap(spot_price))

        # 11. Supertrend
        indicators.append(self._analyze_supertrend())

        # 12. Timing
        timing_ok, timing_score = self._check_timing(is_expiry_day)
        indicators.append(timing_score)

        # 13. FII/DII institutional flows
        indicators.append(self._score_institutional_flows(fii_net, dii_net))

        # 14. Volume Profile (compute from bar history)
        poc, vah, val = self._compute_volume_profile()

        # ── Aggregate ────────────────────────────────────────────────
        # Only normalize by indicators that have meaningful signal (abs(score) > 0.01)
        # This prevents zero-score indicators (missing option chain data, insufficient bars)
        # from diluting directional signals from trend, RSI, VWAP, supertrend.
        active_indicators = [i for i in indicators if abs(i.score) > 0.01]
        if active_indicators:
            overall_score = sum(i.score * i.confidence * i.weight for i in active_indicators)
            total_weight = sum(i.weight * i.confidence for i in active_indicators)
            if total_weight > 0:
                overall_score /= total_weight
        else:
            overall_score = 0.0

        confidence = min(1.0, sum(i.confidence * i.weight for i in indicators))

        # Determine bias
        market_bias = self._score_to_bias(overall_score)

        # Determine action
        vix_regime = self._get_vix_regime(vix)
        action = self._decide_action(
            overall_score, vix_regime, iv_percentile, is_expiry_day,
            spot_price, max_pain, timing_ok,
        )

        reasoning = self._build_reasoning(indicators, action, vix_regime, market_bias)

        return MarketAnalysis(
            timestamp=datetime.now(),
            spot_price=spot_price,
            vix=vix,
            vix_regime=vix_regime,
            pcr=pcr,
            max_pain=max_pain,
            max_pain_distance=spot_price - max_pain,
            oi_support=oi_support,
            oi_resistance=oi_resistance,
            iv_percentile=iv_percentile,
            iv_skew=iv_skew,
            market_bias=market_bias,
            recommended_action=action,
            overall_score=overall_score,
            confidence=confidence,
            indicators=indicators,
            reasoning=reasoning,
            is_expiry_day=is_expiry_day,
            timing_ok=timing_ok,
            fii_net=fii_net,
            dii_net=dii_net,
            poc=poc,
            vah=vah,
            val=val,
            oi_ce_change_pct=oi_ce_change_pct,
            oi_pe_change_pct=oi_pe_change_pct,
        )

    # ── Individual Indicator Analyzers ───────────────────────────────────────

    def _analyze_vix(self, vix: float) -> IndicatorScore:
        """VIX regime: low VIX = sell premium, high VIX = buy premium/hedge."""
        if vix < VIX_LOW:
            score = 0.3  # mild bullish bias (sell premium = expect range)
            reason = f"VIX {vix:.1f} < {VIX_LOW}: LOW regime — sell premium"
        elif vix < VIX_NORMAL_HIGH:
            score = 0.1
            reason = f"VIX {vix:.1f}: NORMAL — balanced strategies"
        elif vix < VIX_HIGH:
            score = -0.2
            reason = f"VIX {vix:.1f}: HIGH — reduce exposure, tighten SL"
        else:
            score = -0.5
            reason = f"VIX {vix:.1f}: EXTREME — hedges only, expect big move"

        conf = min(1.0, 0.6 + abs(vix - 15) / 20)
        return IndicatorScore("vix_regime", score, conf, self.WEIGHTS["vix_regime"], reason)

    def _analyze_pcr(self, pcr: float) -> IndicatorScore:
        """PCR analysis: contrarian signals at extremes."""
        if pcr < PCR_OVERSOLD:
            score = -0.3  # too many calls = contrarian bearish
            reason = f"PCR {pcr:.2f}: extremely low — contrarian bearish"
        elif pcr < PCR_BULLISH:
            score = 0.3
            reason = f"PCR {pcr:.2f}: bullish positioning"
        elif pcr < PCR_NEUTRAL_HIGH:
            score = 0.0
            reason = f"PCR {pcr:.2f}: neutral"
        elif pcr < PCR_BEARISH:
            score = -0.3
            reason = f"PCR {pcr:.2f}: bearish positioning"
        elif pcr > PCR_OVERBOUGHT:
            score = 0.3  # too many puts = contrarian bullish
            reason = f"PCR {pcr:.2f}: extremely high — contrarian bullish"
        else:
            score = -0.2
            reason = f"PCR {pcr:.2f}: moderately bearish"

        conf = min(1.0, 0.5 + abs(pcr - 1.0) * 0.8)
        return IndicatorScore("pcr", score, conf, self.WEIGHTS["pcr"], reason)

    def _calculate_max_pain(self, option_chain: Optional[dict], spot: float) -> float:
        """Calculate Max Pain — strike where total option pain is minimized."""
        if not option_chain:
            # Estimate: round to nearest 100
            return round(spot / 100) * 100

        strikes = sorted(option_chain.keys())
        if not strikes:
            return round(spot / 100) * 100

        min_pain = float("inf")
        max_pain_strike = strikes[len(strikes) // 2]

        for settlement in strikes:
            total_pain = 0.0
            for strike in strikes:
                ce = option_chain[strike].get("CE", {})
                pe = option_chain[strike].get("PE", {})
                ce_oi = ce.get("oi", 0)
                pe_oi = pe.get("oi", 0)

                # Call writers' pain
                if settlement > strike:
                    total_pain += (settlement - strike) * ce_oi
                # Put writers' pain
                if settlement < strike:
                    total_pain += (strike - settlement) * pe_oi

            if total_pain < min_pain:
                min_pain = total_pain
                max_pain_strike = settlement

        return max_pain_strike

    def _analyze_max_pain(
        self, spot: float, max_pain: float, vix: float, is_expiry: bool
    ) -> IndicatorScore:
        """Max Pain: price tends to converge to max pain near expiry."""
        distance = spot - max_pain
        distance_pct = distance / spot * 100

        now = datetime.now().time()
        in_window = (
            self.is_paper  # always in window for paper mode
            or (now >= dt_time(MAX_PAIN_ENTRY_AFTER_HOUR, MAX_PAIN_ENTRY_AFTER_MINUTE)
                and vix < MAX_PAIN_VIX_THRESHOLD)
        )

        if not in_window and not is_expiry:
            return IndicatorScore(
                "max_pain", 0.0, 0.2, self.WEIGHTS["max_pain"],
                f"Max Pain {max_pain:.0f} — too early/VIX too high for signal"
            )

        # Price above max pain --> bearish pull, below --> bullish pull
        if abs(distance) < MAX_PAIN_RELIABILITY_RANGE:
            score = -distance_pct / 2  # mean revert toward max pain
            conf = 0.7 if is_expiry else 0.4
            reason = f"Max Pain {max_pain:.0f}, spot {distance_pct:+.1f}% away — pull {'down' if distance > 0 else 'up'}"
        else:
            score = 0.0
            conf = 0.2
            reason = f"Max Pain {max_pain:.0f}, spot too far ({distance_pct:+.1f}%)"

        return IndicatorScore("max_pain", max(min(score, 1), -1), conf, self.WEIGHTS["max_pain"], reason)

    def _find_oi_levels(
        self, option_chain: Optional[dict], spot: float
    ) -> tuple[float, float]:
        """Find nearest strong support (put OI) and resistance (call OI)."""
        if not option_chain:
            interval = INDEX_CONFIG.get(self.symbol, {}).get("strike_interval", 50)
            return spot - 3 * interval, spot + 3 * interval

        total_oi = sum(
            option_chain[s].get("CE", {}).get("oi", 0) +
            option_chain[s].get("PE", {}).get("oi", 0)
            for s in option_chain
        )
        if total_oi == 0:
            interval = INDEX_CONFIG.get(self.symbol, {}).get("strike_interval", 50)
            return spot - 3 * interval, spot + 3 * interval

        # Find highest put OI below spot = support
        support = spot - 500
        max_put_oi = 0
        for strike in sorted(option_chain.keys()):
            if strike >= spot:
                continue
            put_oi = option_chain[strike].get("PE", {}).get("oi", 0)
            if put_oi > max_put_oi:
                max_put_oi = put_oi
                support = strike

        # Find highest call OI above spot = resistance
        resistance = spot + 500
        max_call_oi = 0
        for strike in sorted(option_chain.keys()):
            if strike <= spot:
                continue
            call_oi = option_chain[strike].get("CE", {}).get("oi", 0)
            if call_oi > max_call_oi:
                max_call_oi = call_oi
                resistance = strike

        return support, resistance

    def _analyze_oi_levels(
        self, spot: float, support: float, resistance: float
    ) -> IndicatorScore:
        """Score based on proximity to OI support/resistance."""
        range_size = resistance - support
        if range_size <= 0:
            return IndicatorScore("oi_levels", 0.0, 0.2, self.WEIGHTS["oi_levels"], "No OI data")

        # Position within range: 0=at support, 1=at resistance
        position = (spot - support) / range_size

        if position < 0.3:
            score = 0.4  # near support = bullish
            reason = f"Spot near OI support {support:.0f} (pos={position:.2f})"
        elif position > 0.7:
            score = -0.4  # near resistance = bearish
            reason = f"Spot near OI resistance {resistance:.0f} (pos={position:.2f})"
        else:
            score = 0.0  # middle of range
            reason = f"Spot in mid-range (S={support:.0f} R={resistance:.0f})"

        conf = 0.6
        return IndicatorScore("oi_levels", score, conf, self.WEIGHTS["oi_levels"], reason)

    def _analyze_oi_change(self, option_chain: Optional[dict]) -> tuple:
        """Analyze OI changes for fresh buildup vs unwinding signals.

        Returns (IndicatorScore, ce_change_pct, pe_change_pct) tuple.
        The change percentages are used by the scoring engine for direction-aware OI change scoring.
        """
        if not option_chain:
            return IndicatorScore("oi_change", 0.0, 0.1, self.WEIGHTS["oi_change"], "No OI data"), 0.0, 0.0

        total_ce_oi = sum(option_chain[s].get("CE", {}).get("oi", 0) for s in option_chain)
        total_pe_oi = sum(option_chain[s].get("PE", {}).get("oi", 0) for s in option_chain)

        if total_ce_oi + total_pe_oi == 0:
            return IndicatorScore("oi_change", 0.0, 0.1, self.WEIGHTS["oi_change"], "No OI"), 0.0, 0.0

        # ── Track delta OI from previous snapshot ──
        ce_change_pct = 0.0
        pe_change_pct = 0.0
        if self._prev_total_ce_oi > 0:
            ce_change_pct = (total_ce_oi - self._prev_total_ce_oi) / self._prev_total_ce_oi * 100
        if self._prev_total_pe_oi > 0:
            pe_change_pct = (total_pe_oi - self._prev_total_pe_oi) / self._prev_total_pe_oi * 100

        # Update previous snapshot
        self._prev_total_ce_oi = total_ce_oi
        self._prev_total_pe_oi = total_pe_oi

        ratio = total_pe_oi / max(total_ce_oi, 1)

        # Original ratio-based scoring
        if ratio > 1.3:
            score = 0.3
            reason = f"Put OI buildup (PE/CE={ratio:.2f})"
        elif ratio < 0.7:
            score = -0.3
            reason = f"Call OI buildup (PE/CE={ratio:.2f})"
        else:
            score = 0.0
            reason = f"Balanced OI (PE/CE={ratio:.2f})"

        # Enhance with delta OI info
        if abs(ce_change_pct) > 5 or abs(pe_change_pct) > 5:
            reason += f" | CE chg={ce_change_pct:+.1f}% PE chg={pe_change_pct:+.1f}%"

        return IndicatorScore("oi_change", score, 0.5, self.WEIGHTS["oi_change"], reason), ce_change_pct, pe_change_pct

    def _analyze_iv(
        self, option_chain: Optional[dict], spot: float
    ) -> tuple[float, float]:
        """Calculate IV percentile and put-call IV skew."""
        if not option_chain:
            return 50.0, 0.0  # neutral defaults

        # Collect ATM IVs
        interval = INDEX_CONFIG.get(self.symbol, {}).get("strike_interval", 50)
        atm = round(spot / interval) * interval

        ce_iv = option_chain.get(atm, {}).get("CE", {}).get("iv", 0)
        pe_iv = option_chain.get(atm, {}).get("PE", {}).get("iv", 0)

        if ce_iv == 0 and pe_iv == 0:
            return 50.0, 0.0

        atm_iv = (ce_iv + pe_iv) / 2 if (ce_iv > 0 and pe_iv > 0) else max(ce_iv, pe_iv)
        skew = (pe_iv - ce_iv) / atm_iv if atm_iv > 0 else 0.0

        # IV Percentile from VIX history
        if len(self._vix_history) >= 20:
            rank = sum(1 for v in self._vix_history if v < atm_iv) / len(self._vix_history) * 100
        else:
            rank = 50.0  # default

        return rank, skew

    def _score_iv_percentile(self, iv_pct: float) -> IndicatorScore:
        """High IV = sell premium, low IV = buy premium."""
        if iv_pct > IV_PERCENTILE_HIGH:
            score = 0.3  # high IV = sell premium (direction neutral --> mild bullish)
            reason = f"IV Pctl {iv_pct:.0f}%: HIGH — premiums rich, sell"
        elif iv_pct < IV_PERCENTILE_LOW:
            score = -0.1  # low IV = cheap, but direction unclear
            reason = f"IV Pctl {iv_pct:.0f}%: LOW — premiums cheap, buy"
        else:
            score = 0.0
            reason = f"IV Pctl {iv_pct:.0f}%: NORMAL"

        return IndicatorScore("iv_percentile", score, 0.6, self.WEIGHTS["iv_percentile"], reason)

    def _score_iv_skew(self, skew: float) -> IndicatorScore:
        """Positive skew (put IV > call IV) = market expects downside."""
        if skew > IV_SKEW_THRESHOLD:
            score = -0.3
            reason = f"Put skew {skew:.3f}: market pricing downside risk"
        elif skew < -IV_SKEW_THRESHOLD:
            score = 0.3
            reason = f"Call skew {skew:.3f}: market pricing upside"
        else:
            score = 0.0
            reason = f"Balanced IV skew {skew:.3f}"

        return IndicatorScore("iv_skew", score, 0.4, self.WEIGHTS["iv_skew"], reason)

    def _analyze_trend(self) -> IndicatorScore:
        """EMA stack: 9 > 21 > 50 = bullish, reverse = bearish."""
        if len(self._bars) < 50:
            return IndicatorScore("trend_ema", 0.0, 0.2, self.WEIGHTS["trend_ema"], "Insufficient data")

        closes = pd.Series([b["close"] for b in self._bars])
        ema9 = closes.ewm(span=9, adjust=False).mean().iloc[-1]
        ema21 = closes.ewm(span=21, adjust=False).mean().iloc[-1]
        ema50 = closes.ewm(span=50, adjust=False).mean().iloc[-1]
        price = closes.iloc[-1]

        if ema9 > ema21 > ema50 and price > ema9:
            score = 0.6
            reason = f"Strong uptrend: price > EMA9 > EMA21 > EMA50"
        elif ema9 > ema21 and price > ema21:
            score = 0.3
            reason = f"Uptrend: EMA9 > EMA21, price above EMA21"
        elif ema9 < ema21 < ema50 and price < ema9:
            score = -0.6
            reason = f"Strong downtrend: price < EMA9 < EMA21 < EMA50"
        elif ema9 < ema21 and price < ema21:
            score = -0.3
            reason = f"Downtrend: EMA9 < EMA21, price below EMA21"
        else:
            score = 0.0
            reason = f"Sideways: EMAs mixed"

        return IndicatorScore("trend_ema", score, 0.7, self.WEIGHTS["trend_ema"], reason)

    def _analyze_rsi(self, period: int = 14) -> IndicatorScore:
        """RSI with divergence detection."""
        if len(self._bars) < period + 5:
            return IndicatorScore("rsi", 0.0, 0.2, self.WEIGHTS["rsi"], "Insufficient data")

        closes = [b["close"] for b in self._bars[-(period + 5):]]
        deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
        gains = [max(d, 0) for d in deltas]
        losses = [-min(d, 0) for d in deltas]

        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period
        if avg_loss == 0:
            rsi = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))

        if rsi > 75:
            score = -0.5
            reason = f"RSI {rsi:.1f}: OVERBOUGHT — expect pullback"
        elif rsi > 60:
            score = 0.2
            reason = f"RSI {rsi:.1f}: bullish momentum"
        elif rsi < 25:
            score = 0.5
            reason = f"RSI {rsi:.1f}: OVERSOLD — expect bounce"
        elif rsi < 40:
            score = -0.2
            reason = f"RSI {rsi:.1f}: bearish momentum"
        else:
            score = 0.0
            reason = f"RSI {rsi:.1f}: neutral"

        return IndicatorScore("rsi", score, 0.6, self.WEIGHTS["rsi"], reason)

    def _analyze_vwap(self, spot: float) -> IndicatorScore:
        """VWAP: price above = bullish intraday, below = bearish."""
        if len(self._bars) < 5:
            return IndicatorScore("vwap", 0.0, 0.2, self.WEIGHTS["vwap"], "Insufficient data")

        # Calculate VWAP
        tp_vol = sum(
            (b["high"] + b["low"] + b["close"]) / 3 * b.get("volume", 1)
            for b in self._bars
        )
        total_vol = sum(b.get("volume", 1) for b in self._bars)
        vwap = tp_vol / total_vol if total_vol > 0 else spot

        dist_pct = (spot - vwap) / vwap * 100

        if dist_pct > 0.3:
            score = 0.3
            reason = f"Price {dist_pct:+.2f}% above VWAP — bullish intraday"
        elif dist_pct < -0.3:
            score = -0.3
            reason = f"Price {dist_pct:+.2f}% below VWAP — bearish intraday"
        else:
            score = 0.0
            reason = f"Price at VWAP ({dist_pct:+.2f}%)"

        return IndicatorScore("vwap", score, 0.5, self.WEIGHTS["vwap"], reason)

    def _analyze_supertrend(self, period: int = 10, multiplier: float = 3.0) -> IndicatorScore:
        """Supertrend direction confirmation."""
        if len(self._bars) < period + 5:
            return IndicatorScore("supertrend", 0.0, 0.2, self.WEIGHTS["supertrend"], "Insufficient data")

        df = pd.DataFrame(self._bars[-max(50, period + 10):])
        hl2 = (df["high"] + df["low"]) / 2
        atr = (df["high"] - df["low"]).rolling(period).mean()

        upper = hl2 + multiplier * atr
        lower = hl2 - multiplier * atr

        # Simplified: compare close to midband
        close = df["close"].iloc[-1]
        mid = hl2.iloc[-1]
        up = upper.iloc[-1]
        lo = lower.iloc[-1]

        if close > up:
            score = 0.4
            reason = "Supertrend: BULLISH (price > upper band)"
        elif close < lo:
            score = -0.4
            reason = "Supertrend: BEARISH (price < lower band)"
        elif close > mid:
            score = 0.15
            reason = "Supertrend: mild bullish"
        else:
            score = -0.15
            reason = "Supertrend: mild bearish"

        return IndicatorScore("supertrend", score, 0.5, self.WEIGHTS["supertrend"], reason)

    def _check_timing(self, is_expiry: bool) -> tuple[bool, IndicatorScore]:
        """Check intraday timing rules. Bypassed in paper mode."""
        # Paper mode: always allow entries
        if self.is_paper:
            return True, IndicatorScore(
                "timing", 0.0, 0.3, self.WEIGHTS["timing"],
                "Paper mode — timing bypass"
            )

        now = datetime.now().time()

        # Avoid first 15 minutes (gap & trap)
        if now < dt_time(AVOID_ENTRY_UNTIL_HOUR, AVOID_ENTRY_UNTIL_MINUTE):
            return False, IndicatorScore(
                "timing", 0.0, 0.8, self.WEIGHTS["timing"],
                f"Avoid entries before {AVOID_ENTRY_UNTIL_HOUR}:{AVOID_ENTRY_UNTIL_MINUTE:02d}"
            )

        # Theta acceleration window (last 30 min) — good for sellers
        if now >= dt_time(THETA_ACCELERATION_HOUR, THETA_ACCELERATION_MINUTE):
            return True, IndicatorScore(
                "timing", 0.2, 0.7, self.WEIGHTS["timing"],
                "Theta acceleration window — premium sellers advantage"
            )

        # Normal trading hours
        return True, IndicatorScore(
            "timing", 0.0, 0.3, self.WEIGHTS["timing"],
            "Normal trading hours"
        )

    # ── Institutional Flows & Volume Profile ───────────────────────────────

    def _score_institutional_flows(self, fii_net: float, dii_net: float) -> IndicatorScore:
        """Score based on FII/DII net flows.

        Research:
        - FII Net Flows correlation with NIFTY = +0.7 to +0.8
        - FII monthly sale > Rs 50,000 Cr = extreme stress
        - FII monthly sale > Rs 1 Lakh Cr = capitulation
        """
        score = 0.0
        reasons = []

        if fii_net > 0:
            score += 0.2
            reasons.append(f"FII net buying Rs{fii_net:.0f}Cr")
        elif fii_net < -5000:
            score -= 0.3
            reasons.append(f"FII heavy selling Rs{fii_net:.0f}Cr (< -5000)")

        # DII absorbing FII selling = stabilising force
        if dii_net > 0 and fii_net < 0 and dii_net > abs(fii_net):
            score += 0.1
            reasons.append(f"DII absorbing: Rs{dii_net:.0f}Cr > FII Rs{abs(fii_net):.0f}Cr")

        reason = " | ".join(reasons) if reasons else f"FII={fii_net:.0f} DII={dii_net:.0f}: neutral"
        conf = 0.5 if (fii_net != 0 or dii_net != 0) else 0.1
        # Weight 0.08 — meaningful but not dominant
        return IndicatorScore("fii_dii_flows", score, conf, 0.08, reason)

    def _compute_volume_profile(self) -> tuple[float, float, float]:
        """Compute Volume Profile from internal bar history.

        Divides the price range into buckets (10-point for NIFTY) and counts
        volume in each bucket to find POC, VAH, and VAL.

        Returns (poc, vah, val). All zero if insufficient data.
        """
        if len(self._bars) < 20:
            return 0.0, 0.0, 0.0

        closes = [b["close"] for b in self._bars]
        volumes = [b.get("volume", 1) for b in self._bars]
        highs = [b["high"] for b in self._bars]
        lows = [b["low"] for b in self._bars]

        price_low = min(lows)
        price_high = max(highs)

        bucket_size = INDEX_CONFIG.get(self.symbol, {}).get("strike_interval", 50)
        # Use 10-point buckets for finer resolution within strike intervals
        bucket_size = max(10, bucket_size // 5)

        if price_high - price_low < bucket_size:
            return 0.0, 0.0, 0.0

        # Create buckets
        n_buckets = int((price_high - price_low) / bucket_size) + 1
        bucket_vol = [0.0] * n_buckets

        for i, close in enumerate(closes):
            idx = int((close - price_low) / bucket_size)
            idx = min(idx, n_buckets - 1)
            bucket_vol[idx] += volumes[i]

        total_vol = sum(bucket_vol)
        if total_vol <= 0:
            return 0.0, 0.0, 0.0

        # POC = bucket with highest volume
        poc_idx = max(range(n_buckets), key=lambda x: bucket_vol[x])
        poc = price_low + poc_idx * bucket_size + bucket_size / 2

        # VAH/VAL = boundaries containing 70% of volume around POC
        target_vol = total_vol * 0.70
        accum = bucket_vol[poc_idx]
        low_idx = poc_idx
        high_idx = poc_idx

        while accum < target_vol and (low_idx > 0 or high_idx < n_buckets - 1):
            expand_low = bucket_vol[low_idx - 1] if low_idx > 0 else -1
            expand_high = bucket_vol[high_idx + 1] if high_idx < n_buckets - 1 else -1

            if expand_low >= expand_high:
                low_idx -= 1
                accum += bucket_vol[low_idx]
            else:
                high_idx += 1
                accum += bucket_vol[high_idx]

        val = price_low + low_idx * bucket_size
        vah = price_low + (high_idx + 1) * bucket_size

        return poc, vah, val

    # ── Decision Logic ───────────────────────────────────────────────────────

    def _get_vix_regime(self, vix: float) -> VIXRegime:
        if vix < VIX_LOW:
            return VIXRegime.LOW
        elif vix < VIX_NORMAL_HIGH:
            return VIXRegime.NORMAL
        elif vix < VIX_HIGH:
            return VIXRegime.HIGH
        else:
            return VIXRegime.EXTREME

    def _score_to_bias(self, score: float) -> MarketBias:
        if score > 0.3:
            return MarketBias.STRONG_BULLISH
        elif score > 0.05:
            return MarketBias.BULLISH
        elif score < -0.3:
            return MarketBias.STRONG_BEARISH
        elif score < -0.05:
            return MarketBias.BEARISH
        else:
            return MarketBias.NEUTRAL

    def _decide_action(
        self,
        score: float,
        vix_regime: VIXRegime,
        iv_percentile: float,
        is_expiry: bool,
        spot: float,
        max_pain: float,
        timing_ok: bool,
    ) -> TradeAction:
        """Core decision: what to trade based on all indicators."""
        if not timing_ok:
            return TradeAction.HOLD

        # VIX extreme --> only buy protection
        if vix_regime == VIXRegime.EXTREME:
            if score < -0.3:
                return TradeAction.BUY_PUT
            elif score > 0.3:
                return TradeAction.BUY_CALL
            return TradeAction.HOLD

        # VIX high --> directional plays only
        if vix_regime == VIXRegime.HIGH:
            if score > 0.3:
                return TradeAction.BUY_CALL
            elif score < -0.3:
                return TradeAction.BUY_PUT
            return TradeAction.HOLD

        # VIX low/normal --> premium selling strategies
        if vix_regime == VIXRegime.LOW:
            if abs(score) < 0.05:
                return TradeAction.SELL_STRADDLE    # sideways + low VIX
            elif score > 0.05:
                return TradeAction.SELL_PUT_SPREAD   # bullish + low VIX
            else:
                return TradeAction.SELL_CALL_SPREAD  # bearish + low VIX

        # VIX normal — graduated zones for strategy diversity
        if abs(score) < 0.05:
            return TradeAction.SELL_IRON_CONDOR     # truly range-bound only
        elif score >= 0.05:
            return TradeAction.SELL_PUT_SPREAD      # any bullish lean --> credit put spread
        elif score <= -0.05:
            return TradeAction.SELL_CALL_SPREAD     # any bearish lean --> credit call spread

        return TradeAction.SELL_IRON_CONDOR

    def _build_reasoning(
        self,
        indicators: list[IndicatorScore],
        action: TradeAction,
        vix_regime: VIXRegime,
        bias: MarketBias,
    ) -> str:
        """Build human-readable reasoning from indicators."""
        # Top 3 strongest signals
        sorted_ind = sorted(indicators, key=lambda x: abs(x.score * x.confidence), reverse=True)
        top = sorted_ind[:3]
        parts = [f"{i.name}: {i.reasoning}" for i in top]
        return (
            f"Action: {action.value} | Bias: {bias.value} | VIX: {vix_regime.value}\n"
            + "\n".join(f"  - {p}" for p in parts)
        )
