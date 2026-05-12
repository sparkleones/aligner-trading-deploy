"""
Per-pick entry/SL/target/quantity calculator using ATR-based sizing
and 1% account risk parity.

Inputs: stock price history + composite score.
Outputs: concrete trade plan.

Holding period: 42 calendar days (~30 trading bars), as empirically
determined by the train-window hold-time sweep (Sharpe 1.92 optimum).

Stop-loss: 2 x ATR(20), clamped to [4%, 8%] of entry price.
Target:    +3R (3x stop distance above entry).
Position:  Risk-parity at 1% of equity capital per trade.
           Hard cap: 35% of capital in one name.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import pandas as pd


HOLD_DAYS_DEFAULT = 42        # calendar days
TRAILING_AFTER_PCT = 0.10     # activate trailing stop after +10%
TRAILING_PCT = 0.06           # 6% trailing once activated


@dataclass
class TradeLevels:
    symbol: str
    sector: str
    entry: float
    stop_loss: float
    target: float
    stop_distance_pct: float
    target_pct: float
    qty: int
    capital_deployed: float
    risk_inr: float
    hold_days: int = HOLD_DAYS_DEFAULT
    trailing_after_pct: float = TRAILING_AFTER_PCT
    trailing_pct: float = TRAILING_PCT
    composite: float = 0.0
    ai_verdict: str = ""
    ai_conviction: str = ""
    ai_flags: list[str] = None
    notes: list[str] = None

    def __post_init__(self):
        if self.ai_flags is None:
            self.ai_flags = []
        if self.notes is None:
            self.notes = []

    def to_dict(self) -> dict:
        return asdict(self)


def _atr_pct(history: pd.DataFrame, period: int = 20) -> float:
    if len(history) < period + 1:
        return 0.04
    h = history["High"]
    l = history["Low"]
    c = history["Close"]
    prev_c = c.shift(1)
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    atr = tr.rolling(period).mean().iloc[-1]
    px = c.iloc[-1]
    if px <= 0 or pd.isna(atr):
        return 0.04
    return float(atr / px)


def compute_levels(
    symbol: str,
    sector: str,
    history: pd.DataFrame,
    composite_score: float,
    account_capital: float = 100_000.0,
    risk_per_trade_pct: float = 0.01,
    max_position_pct: float = 0.35,
    min_stop_pct: float = 0.04,
    max_stop_pct: float = 0.08,
    reward_risk_ratio: float = 3.0,
    hold_days: int = HOLD_DAYS_DEFAULT,
    ai_verdict: str = "",
    ai_conviction: str = "",
    ai_flags: Optional[list[str]] = None,
) -> Optional[TradeLevels]:
    """Build a concrete trade plan from a ranked pick."""
    if history is None or history.empty:
        return None
    close = float(history["Close"].iloc[-1])
    if close <= 0 or not np.isfinite(close):
        return None

    notes: list[str] = []
    atr_p = _atr_pct(history, 20)
    raw_stop = max(2.0 * atr_p, min_stop_pct)
    stop_pct = min(raw_stop, max_stop_pct)
    if raw_stop > max_stop_pct:
        notes.append(f"ATR stop {raw_stop:.1%} clamped to {max_stop_pct:.1%}")

    entry = round(close, 2)
    stop_loss = round(entry * (1.0 - stop_pct), 2)
    stop_distance = entry - stop_loss
    target = round(entry + reward_risk_ratio * stop_distance, 2)
    target_pct = (target / entry) - 1.0

    # Sizing: 1% risk per trade, capped at 35% of capital per name
    risk_budget = account_capital * risk_per_trade_pct
    qty_by_risk = int(risk_budget / max(stop_distance, 1e-6))
    qty_by_cap = int((account_capital * max_position_pct) / entry)
    qty = max(0, min(qty_by_risk, qty_by_cap))
    if qty_by_cap < qty_by_risk:
        notes.append("Position sized down to 35% cap")
    if qty == 0:
        notes.append("Capital too small for 1% risk position")

    # AI override: low conviction trims position 50%
    if ai_conviction.upper() == "LOW":
        qty = qty // 2
        notes.append("Qty halved (AI conviction LOW)")
    if ai_verdict.upper() == "CAUTION":
        qty = max(1, qty // 2)
        notes.append("Qty halved (AI verdict CAUTION)")
    if ai_verdict.upper() == "SKIP":
        qty = 0
        notes.append("AI rejected pick")

    return TradeLevels(
        symbol=symbol, sector=sector, entry=entry, stop_loss=stop_loss,
        target=target, stop_distance_pct=stop_pct, target_pct=target_pct,
        qty=qty, capital_deployed=round(qty * entry, 2),
        risk_inr=round(qty * stop_distance, 2),
        hold_days=hold_days, composite=float(composite_score),
        ai_verdict=ai_verdict, ai_conviction=ai_conviction,
        ai_flags=list(ai_flags or []), notes=notes,
    )
