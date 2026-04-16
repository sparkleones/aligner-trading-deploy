"""Smart Strike Selector — Optimal option strike selection using Greeks, OI, IV & liquidity.

Instead of always buying ATM options, this module scores every strike in the
option chain across multiple dimensions to find the strike that maximises
risk-adjusted profit potential.

Scoring dimensions (configurable weights):
  1. Delta targeting    — favour strikes with delta in the sweet spot (0.40-0.55)
  2. IV efficiency      — prefer lower IV strikes (cheaper premium, better R:R)
  3. OI & volume        — high OI/volume = tight spreads, easy fills
  4. Bid-ask spread     — tighter spread = less entry slippage
  5. Premium efficiency — premium per delta point (bang for buck)
  6. Max pain avoidance — avoid strikes pinned near max pain on expiry days
  7. Gamma edge         — higher gamma on expiry for explosive moves

Usage:
    selector = SmartStrikeSelector(config)
    result = selector.select(
        option_chain=chain,
        spot=24050.0,
        opt_type="CE",
        vix=15.5,
        dte_days=2.0,
        max_pain=24000,
        is_expiry_day=False,
    )
    strike = result["strike"]
    reasoning = result["reasoning"]

Indian market specifics:
  - NIFTY strike interval = 50 pts, BANKNIFTY = 100 pts
  - NIFTY lot size = 65 (SEBI revised Feb 2026)
  - Weekly expiries: theta decay brutal, gamma spikes on expiry
  - VIX > 17 = wide spreads, need deeper OTM for risk control
  - RBI repo rate ~6.5% → risk-free rate ~7%
"""

import logging
import math
from typing import Optional

# Use the existing Black-Scholes pricer for Greeks
from backtesting.option_pricer import (
    black_scholes_price,
    bs_delta,
    bs_gamma,
    bs_theta,
    bs_vega,
    skewed_iv,
)

logger = logging.getLogger("smart_strike")

# ── Default configuration ─────────────────────────────────────────────

DEFAULT_STRIKE_CONFIG = {
    # Strike search range (strikes above/below ATM to consider)
    "search_range_strikes": 8,      # ±8 strikes → ±400 pts for NIFTY (50 interval)

    # Delta targeting — BACKTEST PROVEN on 52 trades (Jul 2024 - Jan 2025):
    #   ITM_1 (d=0.60): 15.87x return, 48.1% WR, PF 2.44, MaxDD 61.6%
    #   ATM   (d=0.48): 14.56x return, 46.2% WR, PF 2.59, MaxDD 54.4%
    #   OTM_1 (d=0.36): 11.85x return, 44.2% WR, PF 2.52, MaxDD 49.2%
    # Conclusion: For directional buying, 1 strike ITM maximises profit.
    # Higher delta = more Rs captured per correct directional move.
    "target_delta_min": 0.50,       # Minimum acceptable |delta|
    "target_delta_max": 0.75,       # Maximum acceptable |delta|
    "target_delta_sweet": 0.60,     # Ideal |delta| — 1 strike ITM

    # Scoring weights (sum to 1.0)
    # Backtest: delta and liquidity are the biggest profit drivers
    "w_delta": 0.35,                # Delta proximity to sweet spot (dominant factor)
    "w_iv": 0.05,                   # Lower IV — less important for ITM
    "w_liquidity": 0.25,            # OI + Volume — crucial for fills
    "w_spread": 0.15,               # Bid-ask tightness
    "w_premium_eff": 0.05,          # Premium per delta point
    "w_gamma": 0.15,                # Gamma edge (expiry day)

    # Liquidity filters (minimum thresholds to consider a strike)
    "min_oi": 50_000,               # Minimum OI to consider (skip illiquid)
    "min_volume": 5_000,            # Minimum volume
    "max_spread_pct": 0.05,         # Max bid-ask spread as % of mid price

    # Max pain avoidance
    "max_pain_avoid_range": 100,    # Avoid strikes within ±100 pts of max pain on expiry
    "max_pain_penalty": 0.3,        # Score multiplier for max-pain-adjacent strikes

    # Expiry day adjustments
    "expiry_gamma_boost": 2.0,      # Weight multiplier for gamma on expiry day
    "expiry_prefer_otm_offset": 50, # On expiry, prefer 1 strike OTM (cheaper theta)

    # VIX regime adjustments — backtest: ITM_1 beats all in every VIX regime
    # No need to fall back to ATM — just use ITM always
    "high_vix_threshold": 17.0,     # VIX above this
    "high_vix_delta_max": 0.70,     # Still allow ITM in high VIX
    "low_vix_threshold": 13.0,      # VIX below this

    # Risk-free rate for BS calculations
    "risk_free_rate": 0.07,         # RBI repo + spread
}


class SmartStrikeSelector:
    """Scores and selects optimal option strike from the chain."""

    def __init__(self, config: dict | None = None):
        self.cfg = {**DEFAULT_STRIKE_CONFIG, **(config or {})}
        self._last_selection: dict | None = None

    def select(
        self,
        option_chain: dict,
        spot: float,
        opt_type: str,
        vix: float = 15.0,
        dte_days: float = 5.0,
        max_pain: float = 0.0,
        is_expiry_day: bool = False,
        market_analysis=None,
        strike_interval: float = 50.0,
    ) -> dict:
        """Select the optimal strike from the option chain.

        Args:
            option_chain: {strike_float: {"CE": {...}, "PE": {...}}} from broker
            spot: Current underlying price
            opt_type: "CE" or "PE"
            vix: India VIX value
            dte_days: Days to expiry (fractional OK)
            max_pain: Max pain strike from MarketAnalyzer
            is_expiry_day: Whether today is expiry
            market_analysis: MarketAnalysis object (optional, for OI levels)
            strike_interval: Strike gap (50 for NIFTY, 100 for BANKNIFTY)

        Returns:
            dict with keys:
              - strike: float — selected strike price
              - score: float — composite score (0-1)
              - delta: float — BS delta at selected strike
              - iv: float — IV at selected strike
              - premium: float — estimated premium
              - reasoning: str — human-readable explanation
              - all_scores: list[dict] — scores for all evaluated strikes
              - fallback: bool — True if fell back to ATM
        """
        cfg = self.cfg
        atm = round(spot / strike_interval) * strike_interval

        # ── Collect candidate strikes ──
        candidates = self._get_candidates(option_chain, atm, opt_type,
                                          strike_interval, spot)

        if not candidates:
            # Fallback: ATM if no candidates in chain
            logger.warning("No candidate strikes found — falling back to ATM %.0f", atm)
            return self._atm_fallback(atm, opt_type, spot, vix, dte_days)

        # ── Compute Greeks for each candidate ──
        r = cfg["risk_free_rate"]
        T = max(dte_days / 365.0, 1.0 / 365.0 / 6.5)  # Min 1 hour
        atm_iv = vix / 100.0 * 0.88  # VIX → ATM IV conversion

        scored = []
        for c in candidates:
            strike = c["strike"]

            # IV: use broker-provided IV if available, else compute from BS skew
            broker_iv = c.get("iv", 0.0)
            if broker_iv > 0:
                iv = broker_iv / 100.0  # Broker gives %, convert to decimal
            else:
                iv = skewed_iv(atm_iv, spot, strike, opt_type)

            # Greeks
            delta = bs_delta(spot, strike, T, r, iv, opt_type)
            gamma = bs_gamma(spot, strike, T, r, iv)
            theta = bs_theta(spot, strike, T, r, iv, opt_type)
            vega = bs_vega(spot, strike, T, r, iv)

            # Premium: use broker LTP if available, else compute
            broker_ltp = c.get("ltp", 0.0)
            if broker_ltp > 0:
                premium = broker_ltp
            else:
                premium = black_scholes_price(spot, strike, T, r, iv, opt_type)
                premium = max(0.05, premium)

            c.update({
                "delta": delta,
                "abs_delta": abs(delta),
                "gamma": gamma,
                "theta": theta,
                "vega": vega,
                "iv_decimal": iv,
                "iv_pct": iv * 100,
                "premium": premium,
            })
            scored.append(c)

        # ── Score each candidate ──
        self._apply_vix_adjustments(vix)
        best = self._score_candidates(scored, spot, atm, opt_type, vix,
                                       max_pain, is_expiry_day, dte_days)

        if best is None:
            logger.warning("All candidates scored 0 — falling back to ATM %.0f", atm)
            return self._atm_fallback(atm, opt_type, spot, vix, dte_days)

        # ── Build result ──
        reasoning = self._build_reasoning(best, atm, opt_type, scored)

        result = {
            "strike": best["strike"],
            "score": best["total_score"],
            "delta": best["delta"],
            "gamma": best["gamma"],
            "theta": best["theta"],
            "vega": best["vega"],
            "iv": best["iv_pct"],
            "premium": best["premium"],
            "reasoning": reasoning,
            "all_scores": [
                {
                    "strike": s["strike"],
                    "score": s.get("total_score", 0),
                    "delta": s["delta"],
                    "iv": s["iv_pct"],
                    "premium": s["premium"],
                    "oi": s.get("oi", 0),
                    "volume": s.get("volume", 0),
                }
                for s in sorted(scored, key=lambda x: x.get("total_score", 0), reverse=True)
            ],
            "fallback": False,
        }

        self._last_selection = result
        logger.info(
            "SMART STRIKE: %s %.0f (ATM=%.0f, offset=%+.0f) | "
            "score=%.3f delta=%.3f IV=%.1f%% prem=%.1f | %s",
            opt_type, best["strike"], atm,
            best["strike"] - atm, best["total_score"],
            best["delta"], best["iv_pct"], best["premium"],
            reasoning,
        )

        return result

    # ── Internal methods ────────────────────────────────────────────

    def _get_candidates(
        self,
        option_chain: dict,
        atm: float,
        opt_type: str,
        strike_interval: float,
        spot: float,
    ) -> list[dict]:
        """Extract candidate strikes from the option chain within search range."""
        cfg = self.cfg
        search_range = cfg["search_range_strikes"]
        candidates = []

        for strike, data in option_chain.items():
            # Check if within search range
            distance_strikes = abs(strike - atm) / strike_interval
            if distance_strikes > search_range:
                continue

            opt_data = data.get(opt_type, {})
            if not opt_data:
                continue

            # Must have a tradingsymbol or symbol
            symbol = opt_data.get("tradingsymbol", opt_data.get("symbol", ""))
            if not symbol:
                continue

            candidate = {
                "strike": float(strike),
                "tradingsymbol": symbol,
                "instrument_token": opt_data.get("instrument_token"),
                "lot_size": opt_data.get("lot_size", 0),
                "ltp": float(opt_data.get("ltp", 0)),
                "oi": int(opt_data.get("oi", 0)),
                "volume": int(opt_data.get("volume", 0)),
                "iv": float(opt_data.get("iv", 0)),
                "bid": float(opt_data.get("bid", 0)),
                "ask": float(opt_data.get("ask", 0)),
                "distance_from_atm": strike - atm,
                "distance_strikes": distance_strikes,
            }
            candidates.append(candidate)

        return sorted(candidates, key=lambda x: abs(x["distance_from_atm"]))

    def _apply_vix_adjustments(self, vix: float):
        """Adjust delta targets based on VIX regime.

        Backtest finding (52 trades, Jul 2024 - Jan 2025):
        ITM_1 (always 1 strike ITM) returned 15.87x — best across ALL VIX regimes.
        No benefit to falling back to ATM in any VIX condition.
        Keep ITM bias always, only cap delta slightly in extreme VIX for safety.
        """
        cfg = self.cfg
        if vix >= cfg["high_vix_threshold"]:
            # High VIX: still ITM-biased, slight cap for safety
            cfg["_adj_delta_max"] = cfg["high_vix_delta_max"]
            cfg["_adj_delta_sweet"] = 0.58  # Still ITM
        elif vix <= cfg["low_vix_threshold"]:
            # Low VIX: ITM still best — higher delta captures the moves
            cfg["_adj_delta_max"] = 0.70
            cfg["_adj_delta_sweet"] = 0.58  # Still ITM
        else:
            # Normal VIX: full ITM for max profit capture
            cfg["_adj_delta_max"] = cfg["target_delta_max"]
            cfg["_adj_delta_sweet"] = cfg["target_delta_sweet"]

    def _score_candidates(
        self,
        candidates: list[dict],
        spot: float,
        atm: float,
        opt_type: str,
        vix: float,
        max_pain: float,
        is_expiry_day: bool,
        dte_days: float,
    ) -> dict | None:
        """Score all candidates and return the best one."""
        cfg = self.cfg
        delta_sweet = cfg.get("_adj_delta_sweet", cfg["target_delta_sweet"])
        delta_max = cfg.get("_adj_delta_max", cfg["target_delta_max"])
        delta_min = cfg["target_delta_min"]

        # Normalisation: find ranges for relative scoring
        all_oi = [c["oi"] for c in candidates if c["oi"] > 0]
        all_vol = [c["volume"] for c in candidates if c["volume"] > 0]
        max_oi = max(all_oi) if all_oi else 1
        max_vol = max(all_vol) if all_vol else 1

        best = None
        best_score = -1.0

        for c in candidates:
            abs_delta = c["abs_delta"]
            strike = c["strike"]

            # ── 1. Delta score (0-1) ──
            # Peak at sweet spot, falls off outside range
            if abs_delta < delta_min * 0.5:
                # Way too far OTM (delta < 0.175) — nearly worthless
                s_delta = 0.0
            elif abs_delta < delta_min:
                # Below min but not zero — partial credit
                s_delta = 0.3 * (abs_delta / delta_min)
            elif delta_min <= abs_delta <= delta_max:
                # In the sweet zone — score based on proximity to sweet spot
                deviation = abs(abs_delta - delta_sweet)
                max_deviation = max(delta_sweet - delta_min, delta_max - delta_sweet)
                s_delta = 1.0 - (deviation / max_deviation) * 0.4  # 0.6-1.0 range
            else:
                # Too deep ITM (high delta) — expensive, capped upside
                excess = abs_delta - delta_max
                s_delta = max(0.0, 0.6 - excess * 2.0)

            # ── 2. IV score (0-1) ──
            # Lower IV relative to ATM = better value
            # Use IV percentile within candidate set
            all_ivs = [x["iv_pct"] for x in candidates if x["iv_pct"] > 0]
            if all_ivs and c["iv_pct"] > 0:
                iv_rank = sum(1 for iv in all_ivs if iv > c["iv_pct"]) / len(all_ivs)
                s_iv = iv_rank  # Higher rank (lower IV) = higher score
            else:
                s_iv = 0.5  # No IV data — neutral

            # ── 3. Liquidity score (0-1) ──
            # Combine OI and volume (both normalised to 0-1)
            oi_norm = c["oi"] / max_oi if max_oi > 0 else 0
            vol_norm = c["volume"] / max_vol if max_vol > 0 else 0
            s_liquidity = 0.6 * oi_norm + 0.4 * vol_norm

            # Apply minimum thresholds
            has_live_data = any(x["oi"] > 0 for x in candidates)
            if has_live_data:
                if c["oi"] < cfg["min_oi"] and c["oi"] > 0:
                    s_liquidity *= 0.3  # Penalty for low OI
                if c["volume"] < cfg["min_volume"] and c["volume"] > 0:
                    s_liquidity *= 0.5  # Penalty for low volume

            # ── 4. Bid-ask spread score (0-1) ──
            bid = c["bid"]
            ask = c["ask"]
            if bid > 0 and ask > 0:
                mid = (bid + ask) / 2.0
                spread_pct = (ask - bid) / mid if mid > 0 else 1.0
                if spread_pct <= 0.01:
                    s_spread = 1.0  # Penny spread — excellent
                elif spread_pct <= cfg["max_spread_pct"]:
                    s_spread = 1.0 - (spread_pct / cfg["max_spread_pct"]) * 0.6
                else:
                    s_spread = 0.2  # Wide spread — poor
            else:
                # No bid-ask data — use delta-based proxy
                # ATM options have tightest spreads, widens with OTM distance
                s_spread = max(0.3, 1.0 - c["distance_strikes"] * 0.1)

            # ── 5. Premium efficiency score (0-1) ──
            # Premium per delta point — lower = more efficient
            if c["premium"] > 0 and abs_delta > 0.01:
                premium_per_delta = c["premium"] / abs_delta
                # Normalise: ATM premium/delta is reference
                atm_premium_approx = spot * c["iv_decimal"] * math.sqrt(
                    max(dte_days, 0.5) / 365.0
                ) * 0.4 if c["iv_decimal"] > 0 else 100.0
                ref_eff = atm_premium_approx / 0.5  # ATM delta ~0.5
                if ref_eff > 0:
                    eff_ratio = premium_per_delta / ref_eff
                    s_premium_eff = max(0.0, min(1.0, 1.5 - eff_ratio))
                else:
                    s_premium_eff = 0.5
            else:
                s_premium_eff = 0.3

            # ── 6. Gamma score (0-1) ──
            # Higher gamma = more explosive moves (especially good on expiry)
            all_gamma = [x["gamma"] for x in candidates if x["gamma"] > 0]
            if all_gamma and c["gamma"] > 0:
                gamma_rank = sum(1 for g in all_gamma if g <= c["gamma"]) / len(all_gamma)
                s_gamma = gamma_rank
            else:
                s_gamma = 0.5

            # ── Compute weighted total ──
            w = cfg
            gamma_weight = w["w_gamma"]
            if is_expiry_day:
                gamma_weight *= cfg["expiry_gamma_boost"]

            total = (
                w["w_delta"] * s_delta
                + w["w_iv"] * s_iv
                + w["w_liquidity"] * s_liquidity
                + w["w_spread"] * s_spread
                + w["w_premium_eff"] * s_premium_eff
                + gamma_weight * s_gamma
            )

            # Renormalise (gamma weight may have been boosted)
            weight_sum = (w["w_delta"] + w["w_iv"] + w["w_liquidity"]
                          + w["w_spread"] + w["w_premium_eff"] + gamma_weight)
            total = total / weight_sum if weight_sum > 0 else 0

            # ── Penalties ──

            # Max pain penalty on expiry day
            if is_expiry_day and max_pain > 0:
                mp_dist = abs(strike - max_pain)
                if mp_dist < cfg["max_pain_avoid_range"]:
                    total *= cfg["max_pain_penalty"]

            # Expiry day OTM preference
            if is_expiry_day:
                if opt_type == "CE" and strike > atm:
                    total *= 1.1  # Slight OTM CE boost on expiry
                elif opt_type == "PE" and strike < atm:
                    total *= 1.1  # Slight OTM PE boost on expiry
                # Clamp
                total = min(1.0, total)

            # Deep OTM penalty (>4 strikes away from ATM)
            if c["distance_strikes"] > 4:
                total *= max(0.5, 1.0 - (c["distance_strikes"] - 4) * 0.1)

            # Store component scores for debugging
            c["s_delta"] = s_delta
            c["s_iv"] = s_iv
            c["s_liquidity"] = s_liquidity
            c["s_spread"] = s_spread
            c["s_premium_eff"] = s_premium_eff
            c["s_gamma"] = s_gamma
            c["total_score"] = total

            if total > best_score:
                best_score = total
                best = c

        return best

    def _build_reasoning(
        self, best: dict, atm: float, opt_type: str, all_candidates: list[dict]
    ) -> str:
        """Build human-readable reasoning for the selection."""
        parts = []
        strike = best["strike"]

        # Position relative to ATM
        offset = strike - atm
        if abs(offset) < 1:
            parts.append("ATM")
        elif (opt_type == "CE" and offset > 0) or (opt_type == "PE" and offset < 0):
            parts.append(f"OTM by {abs(offset):.0f}pts")
        else:
            parts.append(f"ITM by {abs(offset):.0f}pts")

        # Delta
        parts.append(f"delta={best['delta']:.2f}")

        # Key score drivers
        scores = {
            "delta": best.get("s_delta", 0),
            "IV": best.get("s_iv", 0),
            "liquidity": best.get("s_liquidity", 0),
            "spread": best.get("s_spread", 0),
            "prem_eff": best.get("s_premium_eff", 0),
            "gamma": best.get("s_gamma", 0),
        }
        top_2 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:2]
        drivers = [f"{k}={v:.2f}" for k, v in top_2]
        parts.append("top:" + ",".join(drivers))

        # OI/volume info if available
        if best.get("oi", 0) > 0:
            oi_lakh = best["oi"] / 100_000
            parts.append(f"OI={oi_lakh:.1f}L")

        return " | ".join(parts)

    def _atm_fallback(
        self, atm: float, opt_type: str, spot: float, vix: float, dte_days: float
    ) -> dict:
        """Return ATM strike as fallback when smart selection can't proceed."""
        r = self.cfg["risk_free_rate"]
        T = max(dte_days / 365.0, 1.0 / 365.0 / 6.5)
        atm_iv = vix / 100.0 * 0.88
        iv = skewed_iv(atm_iv, spot, atm, opt_type)
        delta = bs_delta(spot, atm, T, r, iv, opt_type)
        premium = black_scholes_price(spot, atm, T, r, iv, opt_type)

        return {
            "strike": atm,
            "score": 0.5,
            "delta": delta,
            "gamma": 0.0,
            "theta": 0.0,
            "vega": 0.0,
            "iv": iv * 100,
            "premium": max(0.05, premium),
            "reasoning": "ATM fallback (no enriched chain data)",
            "all_scores": [],
            "fallback": True,
        }

    @property
    def last_selection(self) -> dict | None:
        """Get details of the last strike selection for dashboard display."""
        return self._last_selection


# ── Option Chain Enrichment ─────────────────────────────────────────

def enrich_option_chain_with_ltp(
    broker,
    option_chain: dict,
    spot: float,
    strike_interval: float = 50.0,
    range_strikes: int = 10,
) -> dict:
    """Enrich a Kite-style option chain (static metadata) with live LTP data.

    Kite's get_option_chain() only returns instrument_token and tradingsymbol.
    This function calls broker.get_ltp() for nearby strikes to add live pricing.

    For Fyers chains that already have ltp/oi/iv, this is a no-op (returns as-is).

    Args:
        broker: Broker instance with get_ltp() method
        option_chain: The raw option chain from broker.get_option_chain()
        spot: Current underlying price
        strike_interval: Strike gap (50 for NIFTY)
        range_strikes: How many strikes above/below ATM to enrich

    Returns:
        The same option_chain dict, enriched with 'ltp' field where available.
    """
    if not option_chain:
        return option_chain

    # Check if chain already has live data (Fyers)
    sample_strike = next(iter(option_chain.values()), {})
    sample_ce = sample_strike.get("CE", {})
    if sample_ce.get("ltp", 0) > 0 or sample_ce.get("oi", 0) > 0:
        logger.debug("Option chain already enriched (has ltp/oi) — skipping")
        return option_chain

    # Find ATM
    atm = round(spot / strike_interval) * strike_interval

    # Collect symbols to fetch LTP for (±range_strikes around ATM)
    symbols_to_fetch = []
    strike_symbol_map = {}  # symbol → (strike, opt_type)

    for strike, data in option_chain.items():
        if abs(strike - atm) / strike_interval > range_strikes:
            continue
        for ot in ("CE", "PE"):
            ts = data.get(ot, {}).get("tradingsymbol", "")
            if ts:
                symbols_to_fetch.append(ts)
                strike_symbol_map[ts] = (strike, ot)

    if not symbols_to_fetch:
        logger.warning("No symbols to fetch LTP for — chain may be empty")
        return option_chain

    # Batch LTP fetch (Kite supports up to ~200 symbols per call)
    try:
        # Split into batches of 50 to avoid API limits
        batch_size = 50
        all_ltps = {}
        for i in range(0, len(symbols_to_fetch), batch_size):
            batch = symbols_to_fetch[i:i + batch_size]
            ltps = broker.get_ltp(batch)
            all_ltps.update(ltps)

        # Enrich the chain
        enriched_count = 0
        for sym, price in all_ltps.items():
            if sym in strike_symbol_map and price > 0:
                strike, ot = strike_symbol_map[sym]
                option_chain[strike][ot]["ltp"] = price
                enriched_count += 1

        logger.info(
            "Option chain enriched with LTP | %d/%d symbols priced",
            enriched_count, len(symbols_to_fetch),
        )

    except Exception as e:
        logger.warning("Failed to enrich option chain with LTP: %s", e)

    return option_chain


def enrich_option_chain_with_quotes(
    broker,
    option_chain: dict,
    spot: float,
    strike_interval: float = 50.0,
    range_strikes: int = 8,
) -> dict:
    """Enrich option chain with full quote data (LTP, OI, volume, bid/ask).

    Uses Kite's quote() API which returns richer data than ltp().
    Falls back to ltp() enrichment if quote() fails.

    Args:
        broker: Broker instance (KiteConnectBroker)
        option_chain: Raw option chain
        spot: Current underlying price
        strike_interval: Strike gap
        range_strikes: Strikes above/below ATM to enrich

    Returns:
        Enriched option chain dict.
    """
    if not option_chain:
        return option_chain

    # Check if already enriched
    sample = next(iter(option_chain.values()), {})
    if sample.get("CE", {}).get("oi", 0) > 0:
        return option_chain

    atm = round(spot / strike_interval) * strike_interval

    # Collect NFO-qualified symbols
    symbols_qualified = []
    strike_map = {}

    for strike, data in option_chain.items():
        if abs(strike - atm) / strike_interval > range_strikes:
            continue
        for ot in ("CE", "PE"):
            ts = data.get(ot, {}).get("tradingsymbol", "")
            if ts:
                nfo_key = f"NFO:{ts}"
                symbols_qualified.append(nfo_key)
                strike_map[nfo_key] = (strike, ot, ts)

    if not symbols_qualified:
        return option_chain

    # Try Kite's quote() for full data
    try:
        kite = getattr(broker, "_kite", None)
        if kite is None:
            # Not Kite broker, fall back to LTP
            return enrich_option_chain_with_ltp(broker, option_chain, spot,
                                                strike_interval, range_strikes)

        # Batch quotes (Kite limit ~200 per call)
        batch_size = 50
        enriched = 0

        for i in range(0, len(symbols_qualified), batch_size):
            batch = symbols_qualified[i:i + batch_size]
            try:
                quotes = kite.quote(batch)
            except Exception:
                # quote() may fail with permission issues, try ohlc()
                try:
                    quotes = kite.ohlc(batch)
                except Exception:
                    continue

            for key, qdata in quotes.items():
                if key in strike_map:
                    strike, ot, ts = strike_map[key]
                    chain_entry = option_chain[strike][ot]

                    # Extract what's available
                    chain_entry["ltp"] = float(qdata.get("last_price", 0))
                    chain_entry["oi"] = int(qdata.get("oi", 0))
                    chain_entry["volume"] = int(qdata.get("volume", 0))

                    # Bid/ask from depth
                    depth = qdata.get("depth", {})
                    buy_depth = depth.get("buy", [])
                    sell_depth = depth.get("sell", [])
                    if buy_depth:
                        chain_entry["bid"] = float(buy_depth[0].get("price", 0))
                    if sell_depth:
                        chain_entry["ask"] = float(sell_depth[0].get("price", 0))

                    if chain_entry["ltp"] > 0:
                        enriched += 1

        logger.info(
            "Option chain enriched with quotes | %d/%d strikes with live data",
            enriched, len(symbols_qualified),
        )

    except Exception as e:
        logger.warning("Quote enrichment failed, falling back to LTP: %s", e)
        return enrich_option_chain_with_ltp(broker, option_chain, spot,
                                            strike_interval, range_strikes)

    return option_chain
