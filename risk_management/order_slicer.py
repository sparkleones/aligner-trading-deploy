"""
Iceberg Order Execution Module.

Splits large orders into sub-orders respecting NSE quantity freeze limits,
executes them sequentially with randomised delays to avoid signaling intent
to HFT participants.
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from config.constants import (
    FREEZE_LIMITS,
    SLICE_DELAY_MAX_MS,
    SLICE_DELAY_MIN_MS,
)

logger = logging.getLogger(__name__)


# ── Data Structures ──────────────────────────────────────────────────────────


@dataclass
class SubOrder:
    """A single tranche of a sliced order."""

    parent_id: str
    tranche_number: int
    total_tranches: int
    symbol: str
    quantity: int
    order_params: Dict[str, Any]
    created_at: float = field(default_factory=time.time)


@dataclass
class SubOrderResult:
    """Result of executing a single sub-order."""

    sub_order: SubOrder
    success: bool
    response: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    latency_ms: float = 0.0
    executed_at: float = 0.0


# ── Order Slicer ─────────────────────────────────────────────────────────────


class OrderSlicer:
    """Slices large orders into freeze-limit-compliant tranches and executes
    them sequentially with jittered delays."""

    def __init__(self, broker_client: Any = None) -> None:
        self._broker_client = broker_client
        self._freeze_limits: Dict[str, int] = dict(FREEZE_LIMITS)
        self._order_counter: int = 0

    # ── Public API ───────────────────────────────────────────────────────

    def slice_order(
        self,
        symbol: str,
        total_qty: int,
        order_params: Dict[str, Any],
    ) -> List[SubOrder]:
        """Split *total_qty* into sub-orders that respect the NSE freeze limit
        for *symbol*.

        Parameters
        ----------
        symbol:
            Index symbol (e.g. ``"NIFTY"``, ``"BANKNIFTY"``).
        total_qty:
            Total quantity to fill.
        order_params:
            Common order parameters forwarded to every sub-order (side,
            order_type, price, etc.).

        Returns
        -------
        list[SubOrder]
            One or more sub-orders whose quantities sum to *total_qty*.
        """
        freeze_limit = self._get_freeze_limit(symbol)
        self._order_counter += 1
        parent_id = f"SLICE-{self._order_counter}-{int(time.time() * 1000)}"

        if total_qty <= freeze_limit:
            sub = SubOrder(
                parent_id=parent_id,
                tranche_number=1,
                total_tranches=1,
                symbol=symbol,
                quantity=total_qty,
                order_params=dict(order_params),
            )
            logger.info(
                "Order %s: qty %d within freeze limit %d — single order",
                parent_id,
                total_qty,
                freeze_limit,
            )
            return [sub]

        # Build tranches
        sub_orders: List[SubOrder] = []
        remaining = total_qty
        tranche_num = 0
        total_tranches = (total_qty + freeze_limit - 1) // freeze_limit

        while remaining > 0:
            tranche_num += 1
            tranche_qty = min(remaining, freeze_limit)
            sub_orders.append(
                SubOrder(
                    parent_id=parent_id,
                    tranche_number=tranche_num,
                    total_tranches=total_tranches,
                    symbol=symbol,
                    quantity=tranche_qty,
                    order_params=dict(order_params),
                )
            )
            remaining -= tranche_qty

        logger.info(
            "Order %s: qty %d exceeds freeze limit %d — sliced into %d tranches",
            parent_id,
            total_qty,
            freeze_limit,
            total_tranches,
        )
        return sub_orders

    def execute_sliced_orders(
        self,
        sub_orders: List[SubOrder],
    ) -> List[SubOrderResult]:
        """Execute *sub_orders* sequentially with random inter-order delays.

        If a sub-order fails the error is logged but execution continues for
        the remaining tranches (partial fill semantics).

        Returns
        -------
        list[SubOrderResult]
            One result per sub-order preserving order.
        """
        results: List[SubOrderResult] = []

        for idx, sub in enumerate(sub_orders):
            # Random delay between tranches (skip before the first tranche)
            if idx > 0:
                delay_ms = random.randint(SLICE_DELAY_MIN_MS, SLICE_DELAY_MAX_MS)
                time.sleep(delay_ms / 1000.0)

            t_start = time.time()
            try:
                response = self._place_order(sub)
                latency_ms = (time.time() - t_start) * 1000.0
                result = SubOrderResult(
                    sub_order=sub,
                    success=True,
                    response=response,
                    latency_ms=latency_ms,
                    executed_at=time.time(),
                )
                logger.info(
                    "Tranche %d/%d executed | parent=%s qty=%d latency=%.1fms",
                    sub.tranche_number,
                    sub.total_tranches,
                    sub.parent_id,
                    sub.quantity,
                    latency_ms,
                )
            except Exception as exc:
                latency_ms = (time.time() - t_start) * 1000.0
                result = SubOrderResult(
                    sub_order=sub,
                    success=False,
                    error=str(exc),
                    latency_ms=latency_ms,
                    executed_at=time.time(),
                )
                logger.error(
                    "Tranche %d/%d FAILED | parent=%s qty=%d latency=%.1fms error=%s",
                    sub.tranche_number,
                    sub.total_tranches,
                    sub.parent_id,
                    sub.quantity,
                    latency_ms,
                    exc,
                )

            results.append(result)

        succeeded = sum(1 for r in results if r.success)
        logger.info(
            "Sliced execution complete | parent=%s tranches=%d succeeded=%d failed=%d",
            sub_orders[0].parent_id if sub_orders else "N/A",
            len(results),
            succeeded,
            len(results) - succeeded,
        )
        return results

    # ── Internals ────────────────────────────────────────────────────────

    def _get_freeze_limit(self, symbol: str) -> int:
        """Return the NSE freeze limit for *symbol*, raising on unknown."""
        base = symbol.upper().split()[0]
        if base not in self._freeze_limits:
            raise ValueError(
                f"Unknown symbol '{symbol}' — no freeze limit configured. "
                f"Known symbols: {list(self._freeze_limits.keys())}"
            )
        return self._freeze_limits[base]

    def _place_order(self, sub: SubOrder) -> Dict[str, Any]:
        """Forward a sub-order to the broker client.  If no broker is
        configured, return a dry-run acknowledgement."""
        if self._broker_client is None:
            return {
                "status": "DRY_RUN",
                "parent_id": sub.parent_id,
                "tranche": sub.tranche_number,
                "qty": sub.quantity,
            }
        return self._broker_client.place_order(
            symbol=sub.symbol,
            qty=sub.quantity,
            **sub.order_params,
        )
