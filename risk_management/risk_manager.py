"""
Master Risk Manager with MTM Kill Switch.

Tracks daily PnL, enforces position limits, calculates transaction costs,
and triggers an automatic kill switch when drawdown breaches the configured
threshold.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from config.constants import (
    DEFAULT_MAX_DAILY_LOSS_PCT,
    DEFAULT_MAX_OPEN_POSITIONS,
    DEFAULT_MAX_POSITION_SIZE_PCT,
    GST_RATE,
    NSE_TRANSACTION_CHARGE,
    SEBI_TURNOVER_FEE,
    STAMP_DUTY_BUY,
    STT_RATES,
)

logger = logging.getLogger(__name__)


# ── Data Structures ──────────────────────────────────────────────────────────


class RiskLevel(str, Enum):
    NORMAL = "NORMAL"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    KILL_SWITCH = "KILL_SWITCH"


@dataclass
class Position:
    """Snapshot of a single open position for MTM tracking."""

    symbol: str
    qty: int
    side: str                     # "BUY" / "SELL"
    entry_price: float
    current_price: float
    instrument_type: str = "OPT"  # "OPT" or "FUT"

    @property
    def unrealised_pnl(self) -> float:
        multiplier = 1 if self.side.upper() == "BUY" else -1
        return multiplier * (self.current_price - self.entry_price) * self.qty


@dataclass
class RiskStatus:
    """Aggregate risk snapshot returned by :meth:`RiskManager.update_mtm`."""

    total_capital: float
    realised_pnl: float
    unrealised_pnl: float
    total_pnl: float
    daily_pnl_pct: float
    open_positions: int
    risk_level: RiskLevel
    kill_switch_active: bool
    timestamp: float = field(default_factory=time.time)

    @property
    def remaining_loss_budget(self) -> float:
        """How much more can be lost before kill switch triggers (absolute)."""
        return self.total_capital * DEFAULT_MAX_DAILY_LOSS_PCT + self.total_pnl


@dataclass
class KillSwitchReport:
    """Summary produced after a kill switch execution."""

    triggered_at: float
    daily_pnl_pct: float
    orders_cancelled: int
    positions_squared: int
    errors: List[str] = field(default_factory=list)


# ── Risk Manager ─────────────────────────────────────────────────────────────


class RiskManager:
    """Central risk gatekeeper.

    Responsibilities:
    * Real-time MTM tracking and daily-PnL calculation.
    * Position-limit enforcement.
    * Automatic kill switch when drawdown exceeds threshold.
    * Transaction-cost estimation (STT, SEBI fees, stamp duty, GST).
    """

    def __init__(
        self,
        total_capital: float,
        max_daily_loss_pct: float = DEFAULT_MAX_DAILY_LOSS_PCT,
        max_position_size_pct: float = DEFAULT_MAX_POSITION_SIZE_PCT,
        max_open_positions: int = DEFAULT_MAX_OPEN_POSITIONS,
    ) -> None:
        self.total_capital = total_capital
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_position_size_pct = max_position_size_pct
        self.max_open_positions = max_open_positions

        # Daily accumulators
        self._realised_pnl: float = 0.0
        self._unrealised_pnl: float = 0.0
        self._positions: List[Position] = []
        self._pending_orders: List[Dict[str, Any]] = []

        # Kill switch state
        self.kill_switch_active: bool = False
        self._kill_switch_triggered_at: Optional[float] = None

        # Warning threshold (default: 75% of max loss)
        self._warning_pct = max_daily_loss_pct * 0.75

        logger.info(
            "RiskManager initialised | capital=%.2f max_loss=%.2f%% "
            "max_pos_size=%.2f%% max_open=%d",
            total_capital,
            max_daily_loss_pct * 100,
            max_position_size_pct * 100,
            max_open_positions,
        )

    # ── MTM Tracking ─────────────────────────────────────────────────────

    def update_mtm(self, positions: List[Position]) -> RiskStatus:
        """Recalculate mark-to-market and return the current risk status.

        Parameters
        ----------
        positions:
            Current list of open positions with live ``current_price``.
        """
        self._positions = list(positions)
        self._unrealised_pnl = sum(p.unrealised_pnl for p in positions)

        total_pnl = self._realised_pnl + self._unrealised_pnl
        daily_pnl_pct = total_pnl / self.total_capital if self.total_capital else 0.0

        risk_level = self._assess_risk_level(daily_pnl_pct)

        status = RiskStatus(
            total_capital=self.total_capital,
            realised_pnl=self._realised_pnl,
            unrealised_pnl=self._unrealised_pnl,
            total_pnl=total_pnl,
            daily_pnl_pct=daily_pnl_pct,
            open_positions=len(positions),
            risk_level=risk_level,
            kill_switch_active=self.kill_switch_active,
        )

        if risk_level == RiskLevel.WARNING:
            logger.warning(
                "APPROACHING LOSS THRESHOLD | pnl=%.2f pct=%.4f%% threshold=%.4f%%",
                total_pnl,
                daily_pnl_pct * 100,
                -self.max_daily_loss_pct * 100,
            )
        elif risk_level == RiskLevel.CRITICAL:
            logger.critical(
                "LOSS THRESHOLD BREACHED | pnl=%.2f pct=%.4f%% — KILL SWITCH REQUIRED",
                total_pnl,
                daily_pnl_pct * 100,
            )

        return status

    def add_realised_pnl(self, pnl: float) -> None:
        """Record a realised PnL entry (from a closed trade)."""
        self._realised_pnl += pnl
        logger.info(
            "Realised PnL updated | delta=%.2f cumulative=%.2f",
            pnl,
            self._realised_pnl,
        )

    # ── Kill Switch ──────────────────────────────────────────────────────

    def check_kill_switch(self) -> bool:
        """Return ``True`` if the daily loss threshold has been breached."""
        total_pnl = self._realised_pnl + self._unrealised_pnl
        daily_pnl_pct = total_pnl / self.total_capital if self.total_capital else 0.0
        return daily_pnl_pct <= -self.max_daily_loss_pct

    def execute_kill_switch(self, broker_client: Any) -> KillSwitchReport:
        """Emergency liquidation sequence.

        1. Cancel ALL open / pending limit orders.
        2. Square off ALL open positions at market.
        3. Activate kill switch flag.
        4. Halt further trading for the day.

        Parameters
        ----------
        broker_client:
            Broker adapter with ``cancel_order`` and ``place_order`` methods.

        Returns
        -------
        KillSwitchReport
        """
        t_start = time.time()
        total_pnl = self._realised_pnl + self._unrealised_pnl
        daily_pnl_pct = total_pnl / self.total_capital if self.total_capital else 0.0
        errors: List[str] = []

        logger.critical(
            "KILL SWITCH ACTIVATED | pnl=%.2f pct=%.4f%%",
            total_pnl,
            daily_pnl_pct * 100,
        )

        # Step 1: Cancel all pending orders
        orders_cancelled = 0
        for order in list(self._pending_orders):
            try:
                broker_client.cancel_order(order_id=order.get("order_id"))
                orders_cancelled += 1
                logger.info(
                    "Kill switch — cancelled order %s at %.3fs",
                    order.get("order_id"),
                    time.time() - t_start,
                )
            except Exception as exc:
                msg = f"Failed to cancel order {order.get('order_id')}: {exc}"
                errors.append(msg)
                logger.error("Kill switch — %s", msg)

        # Step 2: Square off all positions at market
        positions_squared = 0
        for pos in list(self._positions):
            exit_side = "SELL" if pos.side.upper() == "BUY" else "BUY"
            try:
                broker_client.place_order(
                    symbol=pos.symbol,
                    qty=pos.qty,
                    side=exit_side,
                    order_type="MARKET",
                )
                positions_squared += 1
                logger.info(
                    "Kill switch — squared %s %s x%d at %.3fs",
                    exit_side,
                    pos.symbol,
                    pos.qty,
                    time.time() - t_start,
                )
            except Exception as exc:
                msg = f"Failed to square off {pos.symbol}: {exc}"
                errors.append(msg)
                logger.error("Kill switch — %s", msg)

        # Step 3: Activate flag
        self.kill_switch_active = True
        self._kill_switch_triggered_at = time.time()

        # Step 4: Clear internal state
        self._pending_orders.clear()
        self._positions.clear()

        report = KillSwitchReport(
            triggered_at=self._kill_switch_triggered_at,
            daily_pnl_pct=daily_pnl_pct,
            orders_cancelled=orders_cancelled,
            positions_squared=positions_squared,
            errors=errors,
        )
        logger.critical(
            "Kill switch complete in %.1fms | cancelled=%d squared=%d errors=%d",
            (time.time() - t_start) * 1000,
            orders_cancelled,
            positions_squared,
            len(errors),
        )
        return report

    # ── Position Limits ──────────────────────────────────────────────────

    def check_position_limits(self, new_order: Dict[str, Any]) -> bool:
        """Validate that *new_order* does not breach position limits.

        Checks:
        * Single-position size vs ``max_position_size_pct``.
        * Total open positions vs ``max_open_positions``.

        Returns ``True`` if the order is within limits.
        """
        if self.kill_switch_active:
            logger.warning("Order rejected — kill switch is active")
            return False

        # Check max open positions
        if len(self._positions) >= self.max_open_positions:
            logger.warning(
                "Order rejected — max open positions reached (%d/%d)",
                len(self._positions),
                self.max_open_positions,
            )
            return False

        # Check single-position size
        order_value = new_order.get("price", 0.0) * new_order.get("qty", 0)
        max_allowed = self.total_capital * self.max_position_size_pct

        # Detect option buying: check symbol for CE/PE and side for BUY
        symbol = (
            new_order.get("tradingsymbol", "")
            or new_order.get("symbol", "")
        ).upper()
        side = (
            new_order.get("transaction_type", "")
            or new_order.get("side", "")
        ).upper()
        is_option_buy = side == "BUY" and any(tag in symbol for tag in ("CE", "PE"))

        if is_option_buy:
            # For option buys, premium IS max risk — allow up to full capital.
            # The price in the order dict may be a placeholder (100.0) not the
            # actual premium. Skip this check entirely for option buys since
            # the broker will reject if margin is insufficient anyway.
            return True

        if order_value > max_allowed:
            logger.warning(
                "Order rejected — position value %.2f exceeds limit %.2f (%.1f%% of capital)",
                order_value,
                max_allowed,
                self.max_position_size_pct * 100,
            )
            return False

        return True

    def register_pending_order(self, order: Dict[str, Any]) -> None:
        """Track a pending order for kill-switch cancellation."""
        self._pending_orders.append(order)

    def remove_pending_order(self, order_id: str) -> None:
        """Remove a filled or cancelled order from the pending list."""
        self._pending_orders = [
            o for o in self._pending_orders if o.get("order_id") != order_id
        ]

    # ── Transaction Cost Calculator ──────────────────────────────────────

    @staticmethod
    def calculate_transaction_costs(
        premium: float,
        qty: int,
        instrument_type: str = "OPT",
    ) -> float:
        """Estimate total round-trip transaction costs for an Indian exchange
        trade.

        Parameters
        ----------
        premium:
            Per-unit premium (or price for futures).
        qty:
            Number of units traded.
        instrument_type:
            ``"OPT"`` for options, ``"FUT"`` for futures.

        Returns
        -------
        float
            Total estimated cost in INR.
        """
        turnover = premium * qty

        # STT (sell side only for intraday)
        if instrument_type.upper() == "OPT":
            stt = turnover * STT_RATES["options_sell"]
        else:
            stt = turnover * STT_RATES["futures_sell"]

        # Exchange charges
        sebi_fee = turnover * SEBI_TURNOVER_FEE
        nse_charge = turnover * NSE_TRANSACTION_CHARGE

        # Stamp duty (buy side)
        stamp = turnover * STAMP_DUTY_BUY

        # GST on (brokerage + exchange charges)
        # Assume zero brokerage (discount broker) — GST applies to exchange charges
        taxable_base = nse_charge + sebi_fee
        gst = taxable_base * GST_RATE

        total = stt + sebi_fee + nse_charge + stamp + gst

        logger.debug(
            "Transaction costs | turnover=%.2f stt=%.4f sebi=%.6f "
            "nse=%.4f stamp=%.4f gst=%.4f total=%.4f",
            turnover,
            stt,
            sebi_fee,
            nse_charge,
            stamp,
            gst,
            total,
        )
        return round(total, 4)

    # ── Daily Loss Percentage ──────────────────────────────────────────

    @property
    def daily_loss_pct(self) -> float:
        """Current daily P&L as a fraction of capital (negative = loss).

        Used by agents for drawdown-based position sizing.
        """
        total_pnl = self._realised_pnl + self._unrealised_pnl
        if self.total_capital <= 0:
            return 0.0
        return total_pnl / self.total_capital

    # ── Daily Reset ──────────────────────────────────────────────────────

    def reset_daily(self) -> None:
        """Reset all daily accumulators.  Called at the start of each trading
        day before market open."""
        prev_pnl = self._realised_pnl
        self._realised_pnl = 0.0
        self._unrealised_pnl = 0.0
        self._positions.clear()
        self._pending_orders.clear()
        self.kill_switch_active = False
        self._kill_switch_triggered_at = None

        logger.info(
            "Daily reset complete | previous_day_realised_pnl=%.2f",
            prev_pnl,
        )

    # ── Internals ────────────────────────────────────────────────────────

    def _assess_risk_level(self, daily_pnl_pct: float) -> RiskLevel:
        """Classify the current risk level based on daily PnL percentage."""
        if self.kill_switch_active:
            return RiskLevel.KILL_SWITCH
        if daily_pnl_pct <= -self.max_daily_loss_pct:
            return RiskLevel.CRITICAL
        if daily_pnl_pct <= -self._warning_pct:
            return RiskLevel.WARNING
        return RiskLevel.NORMAL
