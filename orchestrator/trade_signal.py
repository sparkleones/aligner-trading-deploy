"""Shared data structures for the live trading orchestrator."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class OrderLeg:
    """A single leg of a multi-leg options trade."""
    symbol: str             # e.g. "NIFTY2640424500PE"
    side: str               # "BUY" or "SELL"
    qty: int
    order_type: str = "MARKET"
    price: float = 0.0      # limit price if order_type == "LIMIT"
    option_type: str = ""   # "CE" or "PE"
    strike: float = 0.0


@dataclass
class TradeSignal:
    """A signal emitted by a strategy agent."""
    strategy: str
    action: str             # IRON_CONDOR, BULL_PUT_SPREAD, BUY_CALL, BUY_PUT, HOLD
    confidence: float       # 0.0 - 1.0
    underlying_price: float
    timestamp: datetime = field(default_factory=datetime.now)
    reasoning: str = ""
    legs: list[OrderLeg] = field(default_factory=list)
    estimated_credit: float = 0.0   # net premium collected (positive = credit)
    max_loss: float = 0.0           # max risk on the trade
    expiry: str = ""
    metadata: dict = field(default_factory=dict)  # entry_type, vix, support, resistance, etc.


@dataclass
class TradeExecution:
    """Result of executing a trade signal."""
    signal: TradeSignal
    order_ids: list[str] = field(default_factory=list)
    status: str = "PENDING"     # EXECUTED, PARTIAL, FAILED, REJECTED
    fill_prices: list[float] = field(default_factory=list)
    total_premium: float = 0.0
    margin_used: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    error: str = ""
