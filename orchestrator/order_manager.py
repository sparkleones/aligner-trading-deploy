"""Order lifecycle manager — place, poll, confirm, retry.

Handles the full order lifecycle for autonomous trading:
  1. Place order with broker
  2. Poll for fill confirmation (exponential backoff)
  3. Retry on transient failures (network, timeout)
  4. Log all orders to persistent JSON for audit trail
  5. Handle partial fills gracefully

Every order goes through this manager — never call broker.place_order() directly.
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Order polling configuration
POLL_INITIAL_DELAY = 0.3    # 300ms first poll
POLL_MAX_DELAY = 5.0        # 5s max between polls
POLL_BACKOFF_FACTOR = 1.5   # Exponential backoff multiplier
POLL_TIMEOUT = 30.0         # Give up after 30s
MAX_RETRIES = 3             # Retry failed orders up to 3 times
RETRY_DELAY = 2.0           # Wait 2s between retries

# TWAP order slicing (from research: avoid HFT front-running on large orders)
TWAP_SLICE_THRESHOLD = 500  # Slice orders above this qty
TWAP_SLICE_SIZE = 195       # Max qty per slice (3 lots × 65 = 195, must be lot-aligned)
TWAP_SLICE_DELAY = 1.5      # Seconds between slices (randomized ±0.5s)
NIFTY_LOT_SIZE = 65         # NSE NIFTY lot size for qty alignment

# Terminal order statuses
FILLED_STATUSES = {"COMPLETE", "FILLED", "EXECUTED", "TRADED"}
FAILED_STATUSES = {"REJECTED", "CANCELLED", "CANCELED", "EXPIRED", "AMO_REJECTED"}
PENDING_STATUSES = {"OPEN", "PENDING", "PLACED", "TRIGGER_PENDING", "VALIDATION_PENDING"}

# Persistent log path
ORDER_LOG_PATH = Path(__file__).parent.parent / "data" / "order_log.json"


class OrderResult:
    """Result of an order placement + confirmation cycle."""

    __slots__ = (
        "order_id", "symbol", "side", "qty", "product",
        "status", "fill_price", "placed_at", "filled_at",
        "attempts", "error", "raw_response",
    )

    def __init__(self):
        self.order_id: str = ""
        self.symbol: str = ""
        self.side: str = ""
        self.qty: int = 0
        self.product: str = "MIS"
        self.status: str = "PENDING"
        self.fill_price: float = 0.0
        self.placed_at: datetime = datetime.now()
        self.filled_at: Optional[datetime] = None
        self.attempts: int = 0
        self.error: str = ""
        self.raw_response: dict = {}

    @property
    def is_filled(self) -> bool:
        return self.status in FILLED_STATUSES

    @property
    def is_failed(self) -> bool:
        return self.status in FAILED_STATUSES

    def to_dict(self) -> dict:
        return {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side,
            "qty": self.qty,
            "product": self.product,
            "status": self.status,
            "fill_price": self.fill_price,
            "placed_at": self.placed_at.isoformat(),
            "filled_at": self.filled_at.isoformat() if self.filled_at else None,
            "attempts": self.attempts,
            "error": self.error,
        }


class OrderManager:
    """Manages order lifecycle: place --> poll --> confirm --> retry.

    Usage:
        om = OrderManager(broker, rate_limiter)
        result = await om.place_and_confirm(
            symbol="NIFTY24500PE", side="BUY", qty=65,
            order_type="MARKET", product="MIS", tag="v3_entry"
        )
        if result.is_filled:
            print(f"Filled at {result.fill_price}")
    """

    def __init__(self, broker: Any, rate_limiter: Any = None):
        self.broker = broker
        self.rate_limiter = rate_limiter
        self._order_log: list[dict] = []
        self._load_log()

    def _load_log(self) -> None:
        """Load existing order log for the day."""
        try:
            if ORDER_LOG_PATH.exists():
                with open(ORDER_LOG_PATH) as f:
                    data = json.load(f)
                # Only keep today's orders
                today = datetime.now().strftime("%Y-%m-%d")
                self._order_log = [
                    o for o in data if o.get("placed_at", "").startswith(today)
                ]
        except Exception:
            self._order_log = []

    def _save_log(self) -> None:
        """Persist order log to disk."""
        try:
            ORDER_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(ORDER_LOG_PATH, "w") as f:
                json.dump(self._order_log, f, indent=2, default=str)
        except Exception as e:
            logger.error("Failed to save order log: %s", e)

    async def place_and_confirm(
        self,
        symbol: str,
        side: str,
        qty: int,
        order_type: str = "MARKET",
        price: float = 0.0,
        product: str = "MIS",
        tag: str = "",
    ) -> OrderResult:
        """Place an order and poll until filled or failed.

        Returns an OrderResult with the final status.
        Retries up to MAX_RETRIES on transient failures.
        """
        result = OrderResult()
        result.symbol = symbol
        result.side = side
        result.qty = qty
        result.product = product

        for attempt in range(1, MAX_RETRIES + 1):
            result.attempts = attempt

            try:
                # Rate limit
                if self.rate_limiter:
                    await self.rate_limiter.acquire()

                # Place the order
                logger.info(
                    "ORDER PLACE [attempt %d/%d] | %s %s %s qty=%d product=%s tag=%s",
                    attempt, MAX_RETRIES, side, symbol, order_type, qty, product, tag,
                )

                response = self.broker.place_order(
                    symbol=symbol,
                    side=side,
                    qty=qty,
                    order_type=order_type,
                    price=price,
                    product=product,
                    tag=tag[:20],
                )

                result.order_id = response.get("order_id", "")
                result.raw_response = response
                result.placed_at = datetime.now()

                # Paper broker fills immediately
                if response.get("status") in FILLED_STATUSES:
                    result.status = "COMPLETE"
                    result.fill_price = response.get("fill_price", response.get("price", price))
                    result.filled_at = datetime.now()
                    logger.info(
                        "ORDER FILLED (instant) | %s %s @ %.2f | id=%s",
                        side, symbol, result.fill_price, result.order_id,
                    )
                    self._log_order(result)
                    return result

                # Live broker — poll for fill
                if result.order_id:
                    result = await self._poll_until_terminal(result)
                    if result.is_filled:
                        self._log_order(result)
                        return result
                    if result.is_failed:
                        logger.warning(
                            "ORDER REJECTED | %s %s | id=%s | status=%s | reason=%s",
                            side, symbol, result.order_id, result.status,
                            result.error or "unknown",
                        )
                        # Rejected orders shouldn't be retried (margin, etc.)
                        if result.status == "REJECTED":
                            result.error = f"Order rejected by exchange: {response}"
                            self._log_order(result)
                            return result
                else:
                    result.status = "FAILED"
                    result.error = "No order_id returned"

            except Exception as e:
                result.error = str(e)
                logger.error(
                    "ORDER ERROR [attempt %d/%d] | %s %s | %s",
                    attempt, MAX_RETRIES, side, symbol, e,
                )

            # Wait before retry (not on last attempt)
            if attempt < MAX_RETRIES:
                logger.info("Retrying in %.1fs...", RETRY_DELAY)
                await asyncio.sleep(RETRY_DELAY)

        # All retries exhausted
        result.status = "FAILED"
        if not result.error:
            result.error = f"Failed after {MAX_RETRIES} attempts"
        logger.error(
            "ORDER FAILED (all retries) | %s %s qty=%d | %s",
            side, symbol, qty, result.error,
        )
        self._log_order(result)
        return result

    async def _poll_until_terminal(self, result: OrderResult) -> OrderResult:
        """Poll order status with exponential backoff until terminal state."""
        delay = POLL_INITIAL_DELAY
        start = time.monotonic()

        while time.monotonic() - start < POLL_TIMEOUT:
            await asyncio.sleep(delay)

            try:
                orders = self.broker.get_orders()
                for order in orders:
                    if order.get("order_id") == result.order_id:
                        status = order.get("status", "").upper()

                        if status in FILLED_STATUSES:
                            result.status = status
                            result.fill_price = order.get(
                                "fill_price",
                                order.get("average_price",
                                    order.get("price", 0)),
                            )
                            result.filled_at = datetime.now()
                            logger.info(
                                "ORDER FILLED | %s %s @ %.2f | id=%s | poll=%.1fs",
                                result.side, result.symbol, result.fill_price,
                                result.order_id, time.monotonic() - start,
                            )
                            return result

                        if status in FAILED_STATUSES:
                            result.status = status
                            result.error = order.get(
                                "status_message",
                                order.get("rejection_reason", "Unknown"),
                            )
                            return result

                        # Still pending — continue polling
                        break

            except Exception as e:
                logger.debug("Poll error (will retry): %s", e)

            # Exponential backoff
            delay = min(delay * POLL_BACKOFF_FACTOR, POLL_MAX_DELAY)

        # Timed out
        result.status = "TIMEOUT"
        result.error = f"Order not filled within {POLL_TIMEOUT}s"
        logger.warning(
            "ORDER TIMEOUT | %s %s | id=%s",
            result.side, result.symbol, result.order_id,
        )
        return result

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order. Returns True if successful."""
        try:
            if self.rate_limiter:
                await self.rate_limiter.acquire()
            self.broker.cancel_order(order_id)
            logger.info("ORDER CANCELLED | id=%s", order_id)
            return True
        except Exception as e:
            logger.error("Cancel failed | id=%s | %s", order_id, e)
            return False

    def _log_order(self, result: OrderResult) -> None:
        """Append order to persistent log."""
        entry = result.to_dict()
        self._order_log.append(entry)
        self._save_log()

    def get_today_orders(self) -> list[dict]:
        """Return all orders placed today."""
        return list(self._order_log)

    def get_fill_rate(self) -> float:
        """Return today's fill rate as a percentage."""
        if not self._order_log:
            return 0.0
        filled = sum(1 for o in self._order_log if o["status"] in FILLED_STATUSES)
        return (filled / len(self._order_log)) * 100

    async def place_twap(
        self,
        symbol: str,
        side: str,
        qty: int,
        order_type: str = "MARKET",
        price: float = 0.0,
        product: str = "MIS",
        tag: str = "",
    ) -> OrderResult:
        """TWAP order slicing — breaks large orders into smaller randomized chunks.

        From research: "Use TWAP or VWAP order slicing — break large orders into
        smaller randomized chunks. Paced execution avoids alerting HFT front-runners
        and minimizes impact cost."

        If qty <= TWAP_SLICE_THRESHOLD, delegates to place_and_confirm() directly.
        Otherwise, splits into TWAP_SLICE_SIZE chunks with randomized delays.

        Returns an aggregated OrderResult with volume-weighted average fill price.
        """
        import random

        if qty <= TWAP_SLICE_THRESHOLD:
            return await self.place_and_confirm(
                symbol=symbol, side=side, qty=qty,
                order_type=order_type, price=price,
                product=product, tag=tag,
            )

        # Split into lot-aligned slices (qty must be multiple of lot_size)
        lot = NIFTY_LOT_SIZE
        remaining = qty
        slices = []
        while remaining > 0:
            slice_qty = min(TWAP_SLICE_SIZE, remaining)
            # Round down to nearest lot multiple
            if lot > 0 and slice_qty >= lot:
                slice_qty = (slice_qty // lot) * lot
            elif lot > 0 and remaining < lot:
                # Last slice too small for a full lot — skip (shouldn't happen if qty is lot-aligned)
                break
            slices.append(slice_qty)
            remaining -= slice_qty

        logger.info(
            "TWAP START | %s %s qty=%d --> %d slices of %s",
            side, symbol, qty, len(slices), slices,
        )

        # Execute slices with randomized delays
        total_filled_qty = 0
        total_cost = 0.0  # qty * price for VWAP calculation
        all_order_ids = []
        last_result = OrderResult()

        for i, slice_qty in enumerate(slices):
            result = await self.place_and_confirm(
                symbol=symbol, side=side, qty=slice_qty,
                order_type=order_type, price=price,
                product=product, tag=f"{tag}_s{i+1}",
            )

            if result.is_filled:
                total_filled_qty += slice_qty
                total_cost += result.fill_price * slice_qty
                all_order_ids.append(result.order_id)
                logger.info(
                    "TWAP SLICE %d/%d filled | %s %d @ %.2f",
                    i + 1, len(slices), symbol, slice_qty, result.fill_price,
                )
            else:
                logger.warning(
                    "TWAP SLICE %d/%d FAILED | %s %d | %s",
                    i + 1, len(slices), symbol, slice_qty, result.error,
                )
                # Continue with remaining slices (partial fill is OK)

            last_result = result

            # Randomized delay between slices (not after last slice)
            if i < len(slices) - 1:
                delay = TWAP_SLICE_DELAY + random.uniform(-0.5, 0.5)
                await asyncio.sleep(max(0.5, delay))

        # Build aggregated result
        agg = OrderResult()
        agg.symbol = symbol
        agg.side = side
        agg.qty = total_filled_qty
        agg.product = product
        agg.order_id = ",".join(all_order_ids) if all_order_ids else ""

        if total_filled_qty > 0:
            agg.status = "COMPLETE"
            agg.fill_price = total_cost / total_filled_qty  # VWAP
            agg.filled_at = datetime.now()
            logger.info(
                "TWAP COMPLETE | %s %s %d/%d filled @ VWAP=%.2f",
                side, symbol, total_filled_qty, qty, agg.fill_price,
            )
        else:
            agg.status = "FAILED"
            agg.error = f"All {len(slices)} TWAP slices failed"

        agg.attempts = len(slices)
        self._log_order(agg)
        return agg
