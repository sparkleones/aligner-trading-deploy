"""
Build the live context for the OptionsResearchTeam.

Pulls everything from Python — never asks LLM to compute Greeks or
fetch market data.

Sources:
  - market_timing_analyzer (for NIFTY price, VIX, technicals)
  - engine_state (for current positions, Greeks)
  - option chain (for ATM IV, term structure)
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd


def _days_to_next_weekly_expiry(today: Optional[datetime] = None) -> int:
    """NSE NIFTY weekly expiry is now Tuesday (post Nov 2025 SEBI consolidation)."""
    today = today or datetime.now()
    # 0=Mon, 1=Tue, ...
    days = (1 - today.weekday()) % 7
    if days == 0:
        # Tuesday today
        return 0
    return days


def _days_to_next_monthly_expiry(today: Optional[datetime] = None) -> int:
    """Monthly expiry: last Thursday of the month."""
    today = today or datetime.now()
    # Find last Thursday of current month
    last_day = (today.replace(day=1) + timedelta(days=32)).replace(day=1) - timedelta(days=1)
    while last_day.weekday() != 3:  # 3 = Thursday
        last_day -= timedelta(days=1)
    if last_day.date() >= today.date():
        return (last_day.date() - today.date()).days
    # If passed, find next month's
    next_month = last_day.replace(day=1) + timedelta(days=32)
    next_last = next_month.replace(day=1) + timedelta(days=32)
    next_last = next_last.replace(day=1) - timedelta(days=1)
    while next_last.weekday() != 3:
        next_last -= timedelta(days=1)
    return (next_last.date() - today.date()).days


def build_options_context(
    nifty_history: pd.DataFrame,
    vix_history: Optional[pd.DataFrame] = None,
    option_chain: Optional[dict] = None,
    portfolio_greeks: Optional[dict] = None,
    capital_deployed: float = 0.0,
    capital_available: float = 0.0,
) -> dict:
    """Build a full options context dict for the team."""
    today = datetime.now()
    # Macro (reuse equity logic)
    if not nifty_history.empty:
        c = nifty_history["Close"]
        close = float(c.iloc[-1])
        high_252 = float(nifty_history["High"].tail(252).max())
        ma_200 = float(c.rolling(200).mean().iloc[-1])
        ma_50 = float(c.rolling(50).mean().iloc[-1])
        rsi_14 = _rsi(c, 14)
        rsi_14_w = _rsi(c.resample("W").last().dropna(), 14)
        macro = {
            "nifty_close": close,
            "nifty_dist_high_pct": (close / high_252 - 1.0) if high_252 > 0 else 0,
            "nifty_dist_200dma_pct": (close / ma_200 - 1.0) if ma_200 > 0 else 0,
            "above_200dma": close > ma_200,
            "golden_cross": ma_50 > ma_200,
            "rsi_14": rsi_14,
            "rsi_14_weekly": rsi_14_w,
            "regime": "RISK_ON" if (close > ma_200 and ma_50 > ma_200) else
                       ("RISK_OFF" if (close < ma_200) else "CHOPPY"),
            "bias": "bullish" if close > ma_200 and rsi_14 > 50 else
                    ("bearish" if close < ma_200 and rsi_14 < 50 else "neutral"),
        }
        rv_20 = float(c.pct_change().tail(20).std() * np.sqrt(252)) if len(c) >= 20 else 0
    else:
        macro = {"regime": "UNKNOWN", "bias": "neutral"}
        rv_20 = 0
        close = 0

    # Technical context for index
    technical = {
        "trend": "up" if macro.get("regime") == "RISK_ON" else
                  ("down" if macro.get("regime") == "RISK_OFF" else "sideways"),
        "range_or_trend": "trend" if abs(macro.get("rsi_14", 50) - 50) > 15 else "range",
        "rsi_14": macro.get("rsi_14", 50),
    }

    # Volatility context
    vol = {}
    if vix_history is not None and not vix_history.empty:
        v_close = vix_history["Close"]
        cur_vix = float(v_close.iloc[-1])
        avg_30 = float(v_close.tail(30).mean())
        avg_252 = float(v_close.tail(252).mean())
        # Percentile of current VIX in 252-day range
        last_yr = v_close.tail(252)
        pct = float((last_yr <= cur_vix).mean() * 100)
        vix_high = float(last_yr.max())
        vix_low = float(last_yr.min())
        vol = {
            "vix_current": cur_vix,
            "vix_avg_30d": avg_30,
            "vix_avg_252d": avg_252,
            "vix_percentile": pct,
        }
    else:
        cur_vix = 0
        pct = 50
        vol = {"vix_current": 0, "vix_percentile": 50}

    # ATM IV from option chain (if supplied)
    atm_iv = 0
    if option_chain:
        # Expect dict {strike: {ce_iv, pe_iv, ...}}
        strikes = sorted(option_chain.keys()) if option_chain else []
        if strikes and close > 0:
            atm = min(strikes, key=lambda s: abs(s - close))
            row = option_chain.get(atm, {})
            ce_iv = row.get("ce_iv") or row.get("call_iv") or 0
            pe_iv = row.get("pe_iv") or row.get("put_iv") or 0
            atm_iv = ((ce_iv + pe_iv) / 2) if (ce_iv and pe_iv) else (ce_iv or pe_iv or 0)
    vol["atm_iv"] = atm_iv / 100 if atm_iv > 5 else atm_iv  # normalize 18% -> 0.18
    vol["rv_20d"] = rv_20
    vol["iv_rv_spread"] = vol["atm_iv"] - rv_20
    vol["iv_rank"] = pct  # use VIX percentile as proxy for IV rank
    vol["term_ratio"] = 1.0  # default flat; needs second-month IV to populate
    # Regime tag
    if pct >= 80:
        vol["regime"] = "EXTREME"
    elif pct >= 60:
        vol["regime"] = "HIGH"
    elif pct >= 30:
        vol["regime"] = "FAIR"
    else:
        vol["regime"] = "LOW"

    # Greeks (from portfolio_greeks dict if provided)
    greeks = portfolio_greeks or {}
    greeks.setdefault("net_delta", 0)
    greeks.setdefault("net_theta_inr", 0)
    greeks.setdefault("net_vega_inr", 0)
    greeks.setdefault("net_gamma", 0)
    greeks["capital_deployed"] = capital_deployed
    greeks["capital_available"] = capital_available
    greeks["is_expiry"] = (_days_to_next_weekly_expiry() == 0)
    greeks["day_type"] = "expiry" if greeks["is_expiry"] else "normal"

    # Events
    events = {
        "days_to_weekly_expiry": _days_to_next_weekly_expiry(),
        "days_to_monthly_expiry": _days_to_next_monthly_expiry(),
        "is_expiry_today": (_days_to_next_weekly_expiry() == 0),
        "rbi_policy_within_7d": False,    # placeholder — wire to calendar if available
        "earnings_season": False,
        "days_to_budget": "unknown",
    }

    return {
        "symbol": "NIFTY",
        "macro": macro,
        "technical": technical,
        "vol": vol,
        "greeks": greeks,
        "events": events,
        "days_to_expiry": events["days_to_weekly_expiry"],
    }


def _rsi(closes: pd.Series, period: int = 14) -> float:
    if len(closes) < period + 1:
        return 50.0
    delta = closes.diff().dropna()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    val = rsi.iloc[-1]
    return float(val) if pd.notna(val) else 50.0
