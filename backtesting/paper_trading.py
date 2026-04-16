"""
Paper trading broker for forward-testing strategies without risking capital.

Simulates order execution against live or delayed market data feeds,
applying realistic transaction costs and slippage.
"""

import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from threading import Lock
from typing import Any, Callable, Optional

from config.constants import (
    FREEZE_LIMITS,
    STT_RATES,
    SEBI_TURNOVER_FEE,
    NSE_TRANSACTION_CHARGE,
    STAMP_DUTY_BUY,
    GST_RATE,
    MAX_ORDERS_PER_SECOND,
)

logger = logging.getLogger(__name__)


@dataclass
class PaperOrder:
    """Simulated order in paper trading."""
    order_id: str
    symbol: str
    side: str
    quantity: int
    order_type: str
    price: float
    trigger_price: float
    product: str
    status: str  # "OPEN", "COMPLETE", "CANCELLED", "REJECTED"
    filled_quantity: int = 0
    average_price: float = 0.0
    placed_at: datetime = field(default_factory=datetime.now)
    filled_at: Optional[datetime] = None
    tag: str = ""
    rejection_reason: str = ""


@dataclass
class PaperPosition:
    """Simulated position in paper trading."""
    symbol: str
    quantity: int  # Positive = long, negative = short
    average_price: float
    last_price: float = 0.0
    pnl: float = 0.0
    unrealized_pnl: float = 0.0
    product: str = "NRML"


class PaperTradingBroker:
    """
    Simulated broker for paper trading.

    Implements the same interface as real broker connectors, allowing
    strategies to be tested with live data feeds without risking capital.
    Applies 2026 STT rates and realistic slippage.
    """

    def __init__(
        self,
        initial_capital: float = 1_000_000.0,
        brokerage_per_order: float = 20.0,
        latency_ms: float = 5.0,
    ):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.available_margin = initial_capital
        self.brokerage_per_order = brokerage_per_order
        self.simulated_latency_ms = latency_ms

        self.orders: dict[str, PaperOrder] = {}
        self.positions: dict[str, PaperPosition] = {}
        self.trade_log: list[dict] = []
        self._lock = Lock()
        self._order_count_window: list[float] = []
        self._tick_callbacks: list[Callable] = []
        self._current_prices: dict[str, float] = {}

        logger.info(
            "PaperTradingBroker initialized | capital=%.2f brokerage=%.2f",
            initial_capital, brokerage_per_order,
        )

    def authenticate(self) -> bool:
        """Paper broker always authenticates successfully."""
        logger.info("Paper trading session authenticated")
        return True

    def _check_rate_limit(self) -> bool:
        """Enforce SEBI 9 OPS rate limit even in paper trading."""
        now = time.monotonic()
        self._order_count_window = [
            t for t in self._order_count_window if now - t < 1.0
        ]
        if len(self._order_count_window) >= MAX_ORDERS_PER_SECOND:
            logger.warning("Rate limit reached — %d orders in last 1s", len(self._order_count_window))
            return False
        self._order_count_window.append(now)
        return True

    def _calculate_costs(
        self, price: float, quantity: int, side: str, instrument_type: str = "options"
    ) -> float:
        """Calculate transaction costs using 2026 rates."""
        turnover = price * quantity
        brokerage = self.brokerage_per_order

        stt = 0.0
        if side == "SELL":
            rate = STT_RATES.get(f"{instrument_type}_sell", 0)
            stt = turnover * rate

        exchange = turnover * NSE_TRANSACTION_CHARGE
        sebi = turnover * SEBI_TURNOVER_FEE
        stamp = turnover * STAMP_DUTY_BUY if side == "BUY" else 0.0
        gst = (brokerage + exchange + sebi) * GST_RATE

        return brokerage + stt + exchange + sebi + stamp + gst

    def _simulate_fill_price(
        self, price: float, side: str, order_type: str
    ) -> float:
        """Simulate realistic fill price with slippage."""
        import random

        if order_type == "MARKET":
            slippage_pct = random.uniform(0.0001, 0.0005)
            if side == "BUY":
                return price * (1 + slippage_pct)
            else:
                return price * (1 - slippage_pct)
        else:
            # Limit orders fill at limit price or better
            return price

    def place_order(
        self,
        symbol: str,
        side: str,
        quantity: int | None = None,
        order_type: str = "MARKET",
        price: float = 0.0,
        trigger_price: float = 0.0,
        product: str = "NRML",
        tag: str = "",
        qty: int | None = None,
    ) -> dict:
        """Place a simulated order.

        Accepts both ``quantity=`` (legacy backtest scripts) and ``qty=``
        (OrderManager / KiteConnectBroker convention). Exactly one must
        be provided.
        """
        if quantity is None and qty is None:
            raise TypeError(
                "PaperTradingBroker.place_order: must provide 'quantity' or 'qty'"
            )
        if quantity is None:
            quantity = qty
        t_start = time.monotonic()

        if not self._check_rate_limit():
            return {
                "success": False,
                "message": "Rate limit exceeded (9 OPS)",
                "order_id": "",
            }

        # Check freeze limits
        for index_name, freeze_qty in FREEZE_LIMITS.items():
            if index_name in symbol.upper() and quantity > freeze_qty:
                logger.error(
                    "Order rejected — exceeds freeze limit | symbol=%s qty=%d limit=%d",
                    symbol, quantity, freeze_qty,
                )
                return {
                    "success": False,
                    "message": f"Quantity {quantity} exceeds freeze limit {freeze_qty}",
                    "order_id": "",
                }

        # Simulate network latency
        time.sleep(self.simulated_latency_ms / 1000.0)

        order_id = str(uuid.uuid4())[:12]

        # Get current price for market orders
        if order_type == "MARKET" and price <= 0:
            price = self._current_prices.get(symbol, 0)
            if price <= 0:
                return {
                    "success": False,
                    "message": f"No price available for {symbol}",
                    "order_id": "",
                }

        fill_price = self._simulate_fill_price(price, side, order_type)

        with self._lock:
            # Calculate costs
            costs = self._calculate_costs(fill_price, quantity, side)

            # Check margin
            required_margin = fill_price * quantity * 0.15  # ~15% margin
            if side == "BUY" and self.available_margin < required_margin + costs:
                return {
                    "success": False,
                    "message": "Insufficient margin",
                    "order_id": "",
                }

            order = PaperOrder(
                order_id=order_id,
                symbol=symbol,
                side=side,
                quantity=quantity,
                order_type=order_type,
                price=price,
                trigger_price=trigger_price,
                product=product,
                status="COMPLETE",
                filled_quantity=quantity,
                average_price=fill_price,
                filled_at=datetime.now(),
                tag=tag,
            )
            self.orders[order_id] = order

            # Update positions
            self._update_position(symbol, side, quantity, fill_price, product)

            # Deduct costs
            self.capital -= costs
            self.available_margin -= costs

        latency = (time.monotonic() - t_start) * 1000
        logger.info(
            "Paper order filled | id=%s symbol=%s side=%s qty=%d "
            "price=%.2f fill=%.2f costs=%.2f latency=%.1fms tag=%s",
            order_id, symbol, side, quantity,
            price, fill_price, costs, latency, tag,
        )

        return {
            "success": True,
            "order_id": order_id,
            "fill_price": fill_price,
            "costs": costs,
            "latency_ms": latency,
        }

    def _update_position(
        self, symbol: str, side: str, quantity: int, price: float, product: str
    ):
        """Update position tracking after a fill."""
        signed_qty = quantity if side == "BUY" else -quantity

        if symbol in self.positions:
            pos = self.positions[symbol]
            old_qty = pos.quantity
            new_qty = old_qty + signed_qty

            if new_qty == 0:
                # Position fully closed
                if old_qty > 0:
                    realized = (price - pos.average_price) * abs(old_qty)
                else:
                    realized = (pos.average_price - price) * abs(old_qty)
                self.capital += realized
                self.available_margin += realized
                del self.positions[symbol]

                self.trade_log.append({
                    "timestamp": datetime.now().isoformat(),
                    "symbol": symbol,
                    "action": "CLOSE",
                    "quantity": abs(old_qty),
                    "entry_price": pos.average_price,
                    "exit_price": price,
                    "realized_pnl": realized,
                })
            elif (old_qty > 0 and new_qty > 0) or (old_qty < 0 and new_qty < 0):
                # Adding to position
                total_cost = pos.average_price * abs(old_qty) + price * quantity
                pos.quantity = new_qty
                pos.average_price = total_cost / abs(new_qty)
            else:
                # Partial close + reversal
                if old_qty > 0:
                    realized = (price - pos.average_price) * abs(old_qty)
                else:
                    realized = (pos.average_price - price) * abs(old_qty)
                self.capital += realized
                pos.quantity = new_qty
                pos.average_price = price
        else:
            self.positions[symbol] = PaperPosition(
                symbol=symbol,
                quantity=signed_qty,
                average_price=price,
                last_price=price,
                product=product,
            )

    def cancel_order(self, order_id: str) -> dict:
        """Cancel a pending order."""
        with self._lock:
            if order_id not in self.orders:
                return {"success": False, "message": "Order not found"}
            order = self.orders[order_id]
            if order.status != "OPEN":
                return {"success": False, "message": f"Order already {order.status}"}
            order.status = "CANCELLED"
            logger.info("Paper order cancelled | id=%s", order_id)
            return {"success": True, "order_id": order_id}

    def get_positions(self) -> list[dict]:
        """Get all current positions."""
        with self._lock:
            result = []
            for sym, pos in self.positions.items():
                current = self._current_prices.get(sym, pos.average_price)
                if pos.quantity > 0:
                    unrealized = (current - pos.average_price) * pos.quantity
                else:
                    unrealized = (pos.average_price - current) * abs(pos.quantity)

                result.append({
                    "symbol": sym,
                    "quantity": pos.quantity,
                    "average_price": pos.average_price,
                    "last_price": current,
                    "pnl": pos.pnl,
                    "unrealized_pnl": unrealized,
                    "product": pos.product,
                })
            return result

    def get_orders(self) -> list[dict]:
        """Get all orders for today."""
        with self._lock:
            return [
                {
                    "order_id": o.order_id,
                    "symbol": o.symbol,
                    "side": o.side,
                    "quantity": o.quantity,
                    "order_type": o.order_type,
                    "price": o.price,
                    "status": o.status,
                    "filled_quantity": o.filled_quantity,
                    "average_price": o.average_price,
                    "tag": o.tag,
                }
                for o in self.orders.values()
            ]

    def get_portfolio(self) -> dict:
        """Get portfolio summary."""
        positions = self.get_positions()
        total_unrealized = sum(p["unrealized_pnl"] for p in positions)
        return {
            "capital": self.capital,
            "available_margin": self.available_margin,
            "used_margin": self.initial_capital - self.available_margin,
            "total_unrealized_pnl": total_unrealized,
            "net_value": self.capital + total_unrealized,
            "positions_count": len(positions),
        }

    def get_ltp(self, symbols: list) -> dict[str, float]:
        """Get last traded prices."""
        return {s: self._current_prices.get(s, 0.0) for s in symbols}

    def update_price(self, symbol: str, price: float):
        """Update the current market price for a symbol (from data feed)."""
        self._current_prices[symbol] = price

    def subscribe_ticks(self, symbols: list, callback: Callable):
        """Register a tick callback (for live data feed integration)."""
        self._tick_callbacks.append(callback)
        logger.info("Tick subscription registered for %d symbols", len(symbols))

    def on_tick(self, ticks: list[dict]):
        """Process incoming ticks and update prices."""
        for tick in ticks:
            symbol = tick.get("symbol", "")
            price = tick.get("last_price", 0.0)
            if symbol and price > 0:
                self._current_prices[symbol] = price

        for cb in self._tick_callbacks:
            try:
                cb(ticks)
            except Exception as e:
                logger.error("Tick callback error: %s", e)

    def close(self):
        """Close paper trading session."""
        portfolio = self.get_portfolio()
        logger.info(
            "Paper trading session closed | net_value=%.2f total_orders=%d",
            portfolio["net_value"], len(self.orders),
        )

    def get_session_report(self) -> dict:
        """Generate end-of-session report."""
        portfolio = self.get_portfolio()
        total_trades = len(self.trade_log)
        winning = sum(1 for t in self.trade_log if t.get("realized_pnl", 0) > 0)
        total_pnl = sum(t.get("realized_pnl", 0) for t in self.trade_log)

        return {
            "initial_capital": self.initial_capital,
            "final_capital": self.capital,
            "net_value": portfolio["net_value"],
            "total_pnl": total_pnl,
            "return_pct": (total_pnl / self.initial_capital) * 100,
            "total_trades": total_trades,
            "winning_trades": winning,
            "losing_trades": total_trades - winning,
            "win_rate": (winning / total_trades * 100) if total_trades > 0 else 0,
            "open_positions": len(self.positions),
            "total_orders": len(self.orders),
            "trade_log": self.trade_log,
        }
