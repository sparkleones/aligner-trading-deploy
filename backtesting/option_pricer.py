"""
Black-Scholes Option Pricer for Indian Index Options (NIFTY / BANKNIFTY / FINNIFTY).

Implements:
  - Black-Scholes pricing with proper normal CDF
  - Realistic IV skew for Indian markets (OTM puts carry fear premium)
  - Full Greeks: delta, gamma, theta, vega
  - Intraday theta acceleration (theta decays faster near expiry)
  - VIX-to-ATM-IV conversion (VIX includes all strikes; ATM IV is lower)

Indian market specifics:
  - RBI repo rate ~6.5% → risk-free rate ~7%
  - Negative put skew: OTM puts trade 2-5 vol points above ATM
  - Weekly expiries: theta decay is brutal on expiry day
  - Lot sizes: NIFTY=75, BANKNIFTY=30, FINNIFTY=65
"""

import math
from typing import Optional


# ── Normal distribution functions (no scipy dependency) ─────────────────

def _norm_cdf(x: float) -> float:
    """Standard normal CDF — Abramowitz & Stegun approximation (|error| < 7.5e-8)."""
    if x >= 0:
        k = 1.0 / (1.0 + 0.2316419 * x)
        poly = k * (0.319381530 + k * (-0.356563782 + k * (
            1.781477937 + k * (-1.821255978 + k * 1.330274429))))
        return 1.0 - _norm_pdf(x) * poly
    return 1.0 - _norm_cdf(-x)


def _norm_pdf(x: float) -> float:
    """Standard normal PDF."""
    return (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * x * x)


# ── Black-Scholes core ─────────────────────────────────────────────────

def black_scholes_price(
    spot: float,
    strike: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "CE",
) -> float:
    """
    Black-Scholes European option price.

    Parameters
    ----------
    spot : float   — current underlying price
    strike : float — option strike
    T : float      — time to expiry in years (e.g., 2/365 for 2 days)
    r : float      — annualised risk-free rate (e.g., 0.07 for 7%)
    sigma : float  — annualised implied volatility (e.g., 0.14 for 14%)
    option_type : str — 'CE' (call) or 'PE' (put)

    Returns
    -------
    float — option premium
    """
    if T <= 0:
        # At expiry — intrinsic value only
        if option_type == "CE":
            return max(0.0, spot - strike)
        return max(0.0, strike - spot)

    if sigma <= 0:
        sigma = 0.001

    sqrt_T = math.sqrt(T)
    d1 = (math.log(spot / strike) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    if option_type == "CE":
        return spot * _norm_cdf(d1) - strike * math.exp(-r * T) * _norm_cdf(d2)
    else:
        return strike * math.exp(-r * T) * _norm_cdf(-d2) - spot * _norm_cdf(-d1)


def bs_delta(spot: float, strike: float, T: float, r: float, sigma: float,
             option_type: str = "CE") -> float:
    """Black-Scholes delta."""
    if T <= 0:
        if option_type == "CE":
            return 1.0 if spot > strike else (0.5 if spot == strike else 0.0)
        return -1.0 if spot < strike else (-0.5 if spot == strike else 0.0)

    sqrt_T = math.sqrt(T)
    d1 = (math.log(spot / strike) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)

    if option_type == "CE":
        return _norm_cdf(d1)
    return _norm_cdf(d1) - 1.0


def bs_gamma(spot: float, strike: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes gamma (same for CE and PE)."""
    if T <= 0 or sigma <= 0:
        return 0.0
    sqrt_T = math.sqrt(T)
    d1 = (math.log(spot / strike) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    return _norm_pdf(d1) / (spot * sigma * sqrt_T)


def bs_theta(spot: float, strike: float, T: float, r: float, sigma: float,
             option_type: str = "CE") -> float:
    """Black-Scholes theta — per calendar day (₹/day)."""
    if T <= 0:
        return 0.0
    sqrt_T = math.sqrt(T)
    d1 = (math.log(spot / strike) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    time_decay = -(spot * _norm_pdf(d1) * sigma) / (2.0 * sqrt_T)

    if option_type == "CE":
        rho_part = -r * strike * math.exp(-r * T) * _norm_cdf(d2)
    else:
        rho_part = r * strike * math.exp(-r * T) * _norm_cdf(-d2)

    return (time_decay + rho_part) / 365.0


def bs_vega(spot: float, strike: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes vega — per 1 percentage point IV change."""
    if T <= 0:
        return 0.0
    sqrt_T = math.sqrt(T)
    d1 = (math.log(spot / strike) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    return spot * _norm_pdf(d1) * sqrt_T * 0.01


# ── IV Skew Model (Indian Index Options) ───────────────────────────────

def skewed_iv(
    atm_iv: float,
    spot: float,
    strike: float,
    option_type: str = "CE",
) -> float:
    """
    Compute implied volatility with realistic Indian index option skew.

    Indian market characteristics:
      - Strong negative skew: OTM puts trade 2-8 vol points above ATM
      - Mild call smile: deep OTM calls slightly higher IV
      - Skew steepens on expiry day and in high-VIX environments
      - Put skew is fear-driven (crash protection demand)

    Parameters
    ----------
    atm_iv : float  — at-the-money implied volatility (annualised, e.g., 0.14)
    spot : float    — current underlying price
    strike : float  — option strike
    option_type : str — 'CE' or 'PE'

    Returns
    -------
    float — skew-adjusted IV
    """
    moneyness_pct = abs(strike - spot) / spot * 100  # e.g., 0.625 for 150 pts on 24000

    if option_type == "PE" and strike < spot:
        # ── OTM Put: strong negative skew (fear premium) ──
        # ~2 vol points per 1% OTM, accelerating for deep OTM
        linear = moneyness_pct * 0.020                          # 2% per 1% OTM
        quadratic = (moneyness_pct ** 1.5) * 0.008              # acceleration
        iv_add = (linear + quadratic) * atm_iv / 0.14           # scale with IV level
    elif option_type == "CE" and strike > spot:
        # ── OTM Call: mild smile ──
        linear = moneyness_pct * 0.005
        quadratic = (moneyness_pct ** 1.5) * 0.003
        iv_add = (linear + quadratic) * atm_iv / 0.14
    elif option_type == "PE" and strike >= spot:
        # ── ITM Put: follows call IV by put-call parity ──
        iv_add = moneyness_pct * 0.003 * atm_iv / 0.14
    else:
        # ── ITM Call: follows put IV ──
        iv_add = moneyness_pct * 0.005 * atm_iv / 0.14

    # Floor: IV can't go below 60% of ATM
    return max(atm_iv * 0.60, atm_iv + iv_add)


# ── High-Level Pricing Function ────────────────────────────────────────

def price_option(
    spot: float,
    strike: float,
    dte_days: float,
    vix: float,
    option_type: str = "CE",
    r: float = 0.07,
    apply_skew: bool = True,
) -> dict:
    """
    Price a NIFTY/BANKNIFTY option with realistic Indian market parameters.

    Parameters
    ----------
    spot : float        — current index price (e.g., 24000)
    strike : float      — option strike price (e.g., 23850)
    dte_days : float    — days to expiry (can be fractional, e.g., 1.5)
    vix : float         — India VIX value (e.g., 14.0)
    option_type : str   — 'CE' or 'PE'
    r : float           — risk-free rate (default 7% — RBI repo + spread)
    apply_skew : bool   — whether to apply IV skew

    Returns
    -------
    dict with keys: premium, delta, gamma, theta, vega, iv
    """
    # VIX-to-ATM-IV conversion
    # VIX includes all strikes (variance swap); ATM IV is typically 85-90% of VIX
    atm_iv = vix / 100.0 * 0.88

    # Minimum 1 hour to avoid numerical blow-up at exact expiry
    T = max(dte_days / 365.0, 1.0 / 365.0 / 6.5)

    # Apply IV skew
    if apply_skew:
        iv = skewed_iv(atm_iv, spot, strike, option_type)
    else:
        iv = atm_iv

    premium = black_scholes_price(spot, strike, T, r, iv, option_type)
    delta = bs_delta(spot, strike, T, r, iv, option_type)
    gamma = bs_gamma(spot, strike, T, r, iv)
    theta = bs_theta(spot, strike, T, r, iv, option_type)
    vega = bs_vega(spot, strike, T, r, iv)

    return {
        "premium": max(0.05, round(premium, 2)),
        "delta": round(delta, 4),
        "gamma": round(gamma, 6),
        "theta": round(theta, 2),
        "vega": round(vega, 2),
        "iv": round(iv * 100, 2),
    }


def price_spread(
    spot: float,
    sell_strike: float,
    buy_strike: float,
    dte_days: float,
    vix: float,
    option_type: str = "PE",
    lot_size: int = 75,
    r: float = 0.07,
) -> dict:
    """
    Price a vertical spread (credit or debit) with proper BS pricing.

    Returns
    -------
    dict with: net_credit, max_loss, breakeven, sell_premium, buy_premium,
               net_delta, net_gamma, net_theta, net_vega
    """
    sell = price_option(spot, sell_strike, dte_days, vix, option_type, r)
    buy = price_option(spot, buy_strike, dte_days, vix, option_type, r)

    net_credit = sell["premium"] - buy["premium"]
    spread_width = abs(sell_strike - buy_strike)
    max_loss_per_unit = spread_width - net_credit

    if option_type == "PE":
        breakeven = sell_strike - net_credit
    else:
        breakeven = sell_strike + net_credit

    return {
        "net_credit": round(net_credit, 2),
        "net_credit_total": round(net_credit * lot_size, 2),
        "max_loss_per_unit": round(max(0, max_loss_per_unit), 2),
        "max_loss_total": round(max(0, max_loss_per_unit) * lot_size, 2),
        "spread_width": spread_width,
        "breakeven": round(breakeven, 2),
        "sell_premium": sell["premium"],
        "buy_premium": buy["premium"],
        "sell_iv": sell["iv"],
        "buy_iv": buy["iv"],
        "net_delta": round((sell["delta"] - buy["delta"]) * lot_size, 2),
        "net_gamma": round((sell["gamma"] - buy["gamma"]) * lot_size, 4),
        "net_theta": round((sell["theta"] - buy["theta"]) * lot_size, 2),
        "net_vega": round((sell["vega"] - buy["vega"]) * lot_size, 2),
    }


# ── Quick self-test ────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("Black-Scholes Pricing — NIFTY @ 24000, VIX=14, DTE=2")
    print("=" * 70)

    spot = 24000
    vix = 14.0
    dte = 2.0

    strikes_ce = [23900, 23950, 24000, 24050, 24100, 24150, 24200, 24300]
    strikes_pe = [24100, 24050, 24000, 23950, 23900, 23850, 23800, 23700]

    print(f"\n{'Strike':>8} {'Type':>4} {'Premium':>8} {'Delta':>7} {'Theta':>7} {'IV%':>6}")
    print("-" * 50)

    for s in strikes_ce:
        p = price_option(spot, s, dte, vix, "CE")
        tag = "ATM" if s == spot else f"{s-spot:+d}"
        print(f"{s:>8} {'CE':>4} {p['premium']:>8.2f} {p['delta']:>7.3f} {p['theta']:>7.2f} {p['iv']:>6.1f}")

    print()
    for s in strikes_pe:
        p = price_option(spot, s, dte, vix, "PE")
        print(f"{s:>8} {'PE':>4} {p['premium']:>8.2f} {p['delta']:>7.3f} {p['theta']:>7.2f} {p['iv']:>6.1f}")

    print("\n" + "=" * 70)
    print("Iron Condor: SELL 23850PE/24100CE, BUY 23800PE/24150CE")
    print("=" * 70)

    put_spread = price_spread(spot, 23850, 23800, dte, vix, "PE", 75)
    call_spread = price_spread(spot, 24100, 24150, dte, vix, "CE", 75)

    print(f"\nPut spread:  SELL 23850PE @ {put_spread['sell_premium']:.2f}  BUY 23800PE @ {put_spread['buy_premium']:.2f}")
    print(f"  Credit: {put_spread['net_credit']:.2f}/unit = {put_spread['net_credit_total']:.0f} total")
    print(f"  Max loss: {put_spread['max_loss_per_unit']:.2f}/unit = {put_spread['max_loss_total']:.0f} total")

    print(f"\nCall spread: SELL 24100CE @ {call_spread['sell_premium']:.2f}  BUY 24150CE @ {call_spread['buy_premium']:.2f}")
    print(f"  Credit: {call_spread['net_credit']:.2f}/unit = {call_spread['net_credit_total']:.0f} total")
    print(f"  Max loss: {call_spread['max_loss_per_unit']:.2f}/unit = {call_spread['max_loss_total']:.0f} total")

    total_credit = put_spread['net_credit_total'] + call_spread['net_credit_total']
    total_max_loss = max(put_spread['max_loss_total'], call_spread['max_loss_total'])
    print(f"\nTotal credit: {total_credit:.0f}")
    print(f"Max risk (one wing): {total_max_loss:.0f}")
