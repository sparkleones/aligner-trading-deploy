"""
Trade plan generator: turn a ranked pick into a concrete entry/SL/target
with position size.

Rules (conservative, ₹1-2L account):
    Entry:       Next-session open (assume = current close for live use)
    Stop-Loss:   max(2 × ATR(20),  4% of price)   — tighter wins
                 Wait — we want the SAFER of the two, which is the WIDER
                 stop, so we don't get stopped on noise. Use min in % terms:
                 i.e., 2*ATR if vol is low, 4% if vol is very low.
                 Actually we want max() so we give the trade room.
                 Decision: max(2*ATR, 4%) gives at least 4% room.
                 Capped at 8% so a single loss can't blow up sizing.
    Target:      3:1 reward:risk (target = entry + 3 × stop_distance)
    Position:    Risk-parity at 1% account per trade.
                 qty = floor( (account * 0.01) / stop_distance_per_share )
                 Hard cap: max 35% of account in one name.

Holding rule (managed at exit):
    - Hard stop hit → exit
    - Target hit → exit (or trail; trail enabled if held > 10 bars)
    - Time stop: exit after 20 bars regardless (avoid dead capital)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import math


@dataclass
class TradePlanConfig:
    account_capital: float = 100_000     # ₹1L default — override per call
    risk_per_trade_pct: float = 0.01     # 1% account risk per position
    max_position_pct: float = 0.35       # cap any single name at 35% account
    min_stop_pct: float = 0.04           # SL never tighter than 4% from entry
    max_stop_pct: float = 0.08           # SL never wider than 8% from entry
    reward_risk_ratio: float = 3.0       # target = entry + 3R
    time_stop_bars: int = 20             # exit after 20 sessions regardless
    activate_trail_after_bars: int = 10
    trail_pct: float = 0.05              # 5% trailing once activated


@dataclass
class TradePlan:
    symbol: str
    sector: str
    entry: float
    stop_loss: float
    target: float
    stop_distance_pct: float
    reward_pct: float
    qty: int
    capital_deployed: float
    risk_inr: float
    notes: list[str]
    composite: float = 0.0


def build_trade_plan(
    pick_row: dict,
    cfg: TradePlanConfig | None = None,
) -> Optional[TradePlan]:
    """
    Build a concrete trade plan from a ranked pick dict.
    Expects keys: symbol, sector, last_close, atr_pct, composite.
    """
    cfg = cfg or TradePlanConfig()

    symbol = str(pick_row["symbol"])
    sector = str(pick_row.get("sector", "OTHER"))
    close = float(pick_row["last_close"])
    atr_pct_v = float(pick_row.get("atr_pct", 0.03))
    composite = float(pick_row.get("composite", 0.0))

    if close <= 0 or math.isnan(close):
        return None

    notes: list[str] = []

    # Stop distance as % of price
    raw_stop_pct = max(2.0 * atr_pct_v, cfg.min_stop_pct)
    stop_pct = min(raw_stop_pct, cfg.max_stop_pct)
    if raw_stop_pct > cfg.max_stop_pct:
        notes.append(f"ATR-derived stop {raw_stop_pct:.1%} clamped to {cfg.max_stop_pct:.1%}")

    entry = round(close, 2)
    stop_loss = round(entry * (1.0 - stop_pct), 2)
    stop_distance = entry - stop_loss
    target = round(entry + cfg.reward_risk_ratio * stop_distance, 2)
    reward_pct = (target / entry) - 1.0

    # Position sizing — risk parity
    risk_budget_inr = cfg.account_capital * cfg.risk_per_trade_pct
    qty_by_risk = int(risk_budget_inr / max(stop_distance, 1e-6))
    qty_by_size_cap = int((cfg.account_capital * cfg.max_position_pct) / entry)
    qty = max(0, min(qty_by_risk, qty_by_size_cap))
    if qty_by_size_cap < qty_by_risk:
        notes.append("Sized down to 35% position cap")

    if qty == 0:
        notes.append("Account too small to size a position at 1% risk")

    return TradePlan(
        symbol=symbol,
        sector=sector,
        entry=entry,
        stop_loss=stop_loss,
        target=target,
        stop_distance_pct=stop_pct,
        reward_pct=reward_pct,
        qty=qty,
        capital_deployed=round(qty * entry, 2),
        risk_inr=round(qty * stop_distance, 2),
        notes=notes,
        composite=composite,
    )


def build_plans_for_picks(
    picks_df,
    account_capital: float,
    cfg: TradePlanConfig | None = None,
) -> list[TradePlan]:
    """Build trade plans for an entire picks DataFrame."""
    cfg = cfg or TradePlanConfig()
    cfg.account_capital = account_capital
    plans = []
    for _, row in picks_df.iterrows():
        plan = build_trade_plan(row.to_dict(), cfg=cfg)
        if plan is not None:
            plans.append(plan)
    return plans
