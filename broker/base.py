"""
Abstract broker interface defining the contract for all broker implementations.
Every broker adapter (Zerodha, Fyers, etc.) must implement this interface.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Optional


class BaseBroker(ABC):
    """Abstract base class for broker integrations.

    All broker implementations must subclass this and provide concrete
    implementations for every abstract method. This ensures a uniform API
    across Zerodha Kite, Fyers, and any future broker adapters.
    """

    @abstractmethod
    def authenticate(self) -> bool:
        """Authenticate with the broker and establish a trading session.

        Returns:
            True if authentication succeeded, False otherwise.
        """

    @abstractmethod
    def place_order(
        self,
        symbol: str,
        side: str,
        qty: int,
        order_type: str = "MARKET",
        price: Optional[float] = None,
        trigger_price: Optional[float] = None,
        product: str = "MIS",
        tag: str = "",
    ) -> dict:
        """Place an order with the broker.

        Args:
            symbol: Trading symbol (e.g. 'NIFTY2430618000CE').
            side: 'BUY' or 'SELL'.
            qty: Order quantity (number of shares/units, not lots).
            order_type: 'MARKET', 'LIMIT', 'SL', or 'SL-M'.
            price: Limit price. Required for LIMIT and SL orders.
            trigger_price: Trigger price for SL / SL-M orders.
            product: 'MIS' (intraday), 'NRML' (carry-forward), or 'CNC' (delivery).
            tag: Free-text order tag for tracking (max 20 chars on most brokers).

        Returns:
            Dict containing at least 'order_id' and 'status'.
        """

    @abstractmethod
    def cancel_order(self, order_id: str) -> dict:
        """Cancel a pending order.

        Args:
            order_id: Broker-assigned order identifier.

        Returns:
            Dict containing 'order_id' and cancellation 'status'.
        """

    @abstractmethod
    def get_positions(self) -> list[dict]:
        """Fetch current net positions.

        Returns:
            List of position dicts with keys such as 'symbol', 'qty',
            'average_price', 'pnl', 'product'.
        """

    @abstractmethod
    def get_orders(self) -> list[dict]:
        """Fetch all orders placed today.

        Returns:
            List of order dicts with keys such as 'order_id', 'symbol',
            'side', 'qty', 'price', 'status', 'timestamp'.
        """

    @abstractmethod
    def get_portfolio(self) -> dict:
        """Fetch consolidated portfolio: holdings, positions, and margins.

        Returns:
            Dict with keys 'holdings', 'positions', 'margins'.
        """

    @abstractmethod
    def get_ltp(self, symbols: list[str]) -> dict[str, float]:
        """Get last traded prices for a list of symbols.

        Args:
            symbols: List of trading symbols.

        Returns:
            Mapping of symbol -> last traded price.
        """

    @abstractmethod
    def get_option_chain(self, symbol: str, expiry: str) -> dict:
        """Fetch the option chain for an underlying on a given expiry.

        Args:
            symbol: Underlying symbol (e.g. 'NIFTY').
            expiry: Expiry date string in 'YYYY-MM-DD' format.

        Returns:
            Dict keyed by strike price, each containing 'CE' and 'PE'
            sub-dicts with fields like 'ltp', 'oi', 'volume', 'iv',
            'instrument_token'.
        """

    @abstractmethod
    def subscribe_ticks(
        self,
        symbols: list[str],
        callback: Callable[[list[dict]], Any],
    ) -> None:
        """Subscribe to real-time tick data via WebSocket.

        Args:
            symbols: Symbols to subscribe to.
            callback: Function called with a list of tick dicts on each update.
        """

    @abstractmethod
    def unsubscribe_ticks(self, symbols: list[str]) -> None:
        """Unsubscribe from real-time tick data.

        Args:
            symbols: Symbols to unsubscribe from.
        """

    @abstractmethod
    def close(self) -> None:
        """Tear down connections and release resources.

        Should close WebSocket connections, HTTP sessions, etc.
        """
