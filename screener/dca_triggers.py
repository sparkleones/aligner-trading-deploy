"""
Trigger evaluator for the DCA plan.

Reads the live NIFTY/VIX/breadth state and decides:
  - which ACCELERATE triggers fired (deploy faster)
  - which STOP/PAUSE triggers fired (halt deployment)
  - the recommended tranche multiplier for this week

The triggers were specified in the market timing analysis:

ACCELERATE (any 2+ = double tranche, any 1 = 1.5x):
  A1: NIFTY closes above 50-DMA for 3 consecutive sessions
  A2: Daily RSI(14) crosses above 50 for 3 sessions
  A3: VIX drops below 16
  A4: Breadth >55% above 200-DMA
  A5: 50-DMA crosses back above 200-DMA (golden cross)

STOP/PAUSE (any 1 = halt deployment):
  S1: NIFTY closes below 23000 (deeper drawdown)
  S2: VIX spikes above 25
  S3: Breadth drops below 30%
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .market_timing_analyzer import (
    _fetch_nifty, _fetch_vix, _rsi,
    analyze_price_position, analyze_technical,
    analyze_volatility, analyze_breadth,
)
from .universe_extended import LARGE_CAP


@dataclass
class TriggerSnapshot:
    nifty_close: float = 0.0
    nifty_50dma: float = 0.0
    nifty_200dma: float = 0.0
    vix: float = 0.0
    rsi_daily: float = 0.0
    pct_above_200dma_breadth: float = 0.0

    accelerate_fired: list[str] = field(default_factory=list)
    stop_fired: list[str] = field(default_factory=list)

    tranche_multiplier: float = 1.0
    recommended_action: str = "DEPLOY"   # DEPLOY | PAUSE | STOP
    notes: list[str] = field(default_factory=list)


def _consecutive_above(series: pd.Series, threshold: float, n: int = 3) -> bool:
    if len(series) < n:
        return False
    return bool((series.tail(n) > threshold).all())


def _consecutive_above_rsi(closes: pd.Series, threshold: float = 50.0, n: int = 3) -> bool:
    """True if RSI(14) has been > threshold for n consecutive sessions."""
    if len(closes) < 30 + n:
        return False
    rsis = []
    for i in range(n):
        end_idx = len(closes) - i
        s = closes.iloc[:end_idx]
        rsis.append(_rsi(s, 14))
    return all((r is not None and not np.isnan(r) and r > threshold) for r in rsis)


def evaluate() -> TriggerSnapshot:
    """Evaluate all triggers on current live data."""
    snap = TriggerSnapshot()

    nifty = _fetch_nifty()
    vix_df = _fetch_vix()
    if nifty.empty:
        snap.notes.append("Could not fetch NIFTY data — defaulting to neutral")
        return snap

    price = analyze_price_position(nifty)
    tech = analyze_technical(nifty)
    vol = analyze_volatility(vix_df)
    breadth = analyze_breadth(LARGE_CAP)

    snap.nifty_close = price["close"]
    snap.nifty_50dma = price["ma_50"]
    snap.nifty_200dma = price["ma_200"]
    snap.vix = vol.get("vix_current", 0.0)
    snap.rsi_daily = tech["rsi_14_daily"]
    snap.pct_above_200dma_breadth = breadth.get("pct_above_200dma", 0.0)

    # ── ACCELERATE checks ──
    # A1: NIFTY above 50-DMA for 3 consecutive sessions
    if len(nifty) >= 53:
        ma_50_series = nifty["Close"].rolling(50).mean()
        close_above_50 = nifty["Close"] > ma_50_series
        if close_above_50.tail(3).all():
            snap.accelerate_fired.append(f"A1: NIFTY > 50-DMA for 3 sessions (close {snap.nifty_close:.0f} > 50DMA {snap.nifty_50dma:.0f})")

    # A2: RSI(14) > 50 for 3 sessions
    if _consecutive_above_rsi(nifty["Close"], 50.0, 3):
        snap.accelerate_fired.append(f"A2: RSI(14) > 50 for 3 sessions (current {snap.rsi_daily:.1f})")

    # A3: VIX below 16
    if 0 < snap.vix < 16:
        snap.accelerate_fired.append(f"A3: VIX < 16 (current {snap.vix:.2f})")

    # A4: Breadth > 55%
    if snap.pct_above_200dma_breadth > 55:
        snap.accelerate_fired.append(f"A4: Breadth {snap.pct_above_200dma_breadth:.0f}% > 55%")

    # A5: Golden cross
    if price["golden_cross"]:
        snap.accelerate_fired.append(f"A5: Golden cross active (50-DMA {snap.nifty_50dma:.0f} > 200-DMA {snap.nifty_200dma:.0f})")

    # ── STOP checks ──
    if snap.nifty_close < 23000:
        snap.stop_fired.append(f"S1: NIFTY below 23000 (closed {snap.nifty_close:.0f})")
    if snap.vix > 25:
        snap.stop_fired.append(f"S2: VIX above 25 (current {snap.vix:.2f})")
    if 0 < snap.pct_above_200dma_breadth < 30:
        snap.stop_fired.append(f"S3: Breadth below 30% ({snap.pct_above_200dma_breadth:.0f}%)")

    # ── Compute multiplier ──
    n_acc = len(snap.accelerate_fired)
    n_stop = len(snap.stop_fired)

    if n_stop >= 1:
        snap.tranche_multiplier = 0.0
        snap.recommended_action = "PAUSE"
        snap.notes.append(f"PAUSE — {n_stop} stop trigger(s) fired")
    elif n_acc >= 2:
        snap.tranche_multiplier = 2.0
        snap.recommended_action = "DEPLOY"
        snap.notes.append(f"ACCELERATE x2 — {n_acc} accelerate triggers fired")
    elif n_acc == 1:
        snap.tranche_multiplier = 1.5
        snap.recommended_action = "DEPLOY"
        snap.notes.append(f"ACCELERATE x1.5 — 1 accelerate trigger fired")
    else:
        # No accelerate triggers, no stop. Use base multiplier 1.0,
        # but slow to 0.5x if market still very weak.
        if not price["above_200dma"] and tech["rsi_14_daily"] < 40:
            snap.tranche_multiplier = 0.5
            snap.recommended_action = "DEPLOY"
            snap.notes.append("Reduced to 0.5x — still below 200-DMA + RSI weak")
        else:
            snap.tranche_multiplier = 1.0
            snap.recommended_action = "DEPLOY"
            snap.notes.append("Standard tranche")

    return snap
