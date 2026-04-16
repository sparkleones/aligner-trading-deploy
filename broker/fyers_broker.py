"""
Fyers API v3 broker implementation.

Wraps the fyers-apiv3 SDK for order management, positions, market data,
and WebSocket streaming with automatic reconnection.
"""

import hashlib
import logging
import threading
import time
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from config.settings import load_settings
from config.constants import (
    INDEX_CONFIG,
    WEBSOCKET_RECONNECT_BASE_DELAY_S,
    WEBSOCKET_RECONNECT_MAX_DELAY_S,
    WEBSOCKET_RECONNECT_MULTIPLIER,
)

from broker.base import BaseBroker
from broker.auth import TOTPAuthenticator, _mask

logger = logging.getLogger(__name__)


class FyersBroker(BaseBroker):
    """Broker adapter for Fyers API v3.

    Uses ``fyers_apiv3.FyersModel`` for REST calls and
    ``fyers_apiv3.FyersDataSocket`` for real-time tick streaming.

    Args:
        client_id: Fyers app client ID (format: ``XXXX-100``).
        secret_key: Fyers app secret key.
        user_id: Fyers user/client ID.
        password: Fyers account password.
        totp_secret: Base32 TOTP secret for 2FA.
        redirect_uri: OAuth2 redirect URI registered with Fyers.
    """

    _SIDE_MAP = {"BUY": 1, "SELL": -1}
    _ORDER_TYPE_MAP = {"MARKET": 2, "LIMIT": 1, "SL": 3, "SL-M": 4}
    _PRODUCT_MAP = {"MIS": "INTRADAY", "NRML": "MARGIN", "CNC": "CNC"}

    def __init__(
        self,
        client_id: str = "",
        secret_key: str = "",
        user_id: str = "",
        password: str = "",
        totp_secret: str = "",
        redirect_uri: str = "https://trade.fyers.in/api-login/redirect-uri/get",
    ) -> None:
        settings = load_settings()
        self._client_id = client_id or settings.broker.api_key
        self._secret_key = secret_key or settings.broker.api_secret
        self._user_id = user_id or settings.broker.user_id
        self._password = password or settings.broker.password
        self._totp_secret = totp_secret or settings.broker.totp_secret
        self._redirect_uri = redirect_uri

        self._fyers: Any = None  # fyers_apiv3.FyersModel instance
        self._data_socket: Any = None  # fyers_apiv3.FyersDataSocket instance
        self._session: dict = {}
        self._auth = TOTPAuthenticator(self._totp_secret)

        # tick subscription state
        self._tick_callback: Optional[Callable] = None
        self._subscribed_symbols: list[str] = []
        self._ws_lock = threading.Lock()
        self._reconnect_attempts: int = 0

        logger.info(
            "FyersBroker initialised",
            extra={"user_id": _mask(self._user_id), "client_id": _mask(self._client_id)},
        )

    # ── Authentication ──────────────────────────────────────────────────

    def authenticate(self) -> bool:
        """Perform Fyers OAuth2 login with TOTP-based 2FA.

        The flow:
        1. Generate auth code via Fyers session model with TOTP.
        2. Exchange auth code for access token.
        3. Initialise FyersModel with the token.

        Returns:
            True on success, False on failure.
        """
        try:
            from fyers_apiv3 import fyersModel

            # Step 1 -- create session model to get auth code
            session_model = fyersModel.SessionModel(
                client_id=self._client_id,
                secret_key=self._secret_key,
                redirect_uri=self._redirect_uri,
                response_type="code",
                grant_type="authorization_code",
            )

            # Generate TOTP for 2FA
            totp_value = self._auth.generate_totp()
            logger.debug("TOTP generated for Fyers login")

            # Authenticate with credentials + TOTP
            import requests as _requests

            # Step 1a -- send login credentials
            login_payload = {
                "fy_id": self._user_id,
                "password": hashlib.sha256(self._password.encode()).hexdigest(),
                "app_id": self._client_id.split("-")[0],
                "type": 2,
                "recaptcha_token": "",
            }
            login_url = "https://api-t2.fyers.in/vagator/v2/send_login_otp_v2"
            login_resp = _requests.post(login_url, json=login_payload, timeout=30)
            login_data = login_resp.json()
            request_key = login_data.get("request_key", "")

            # Step 1b -- verify TOTP
            verify_payload = {
                "request_key": request_key,
                "otp": totp_value,
            }
            verify_url = "https://api-t2.fyers.in/vagator/v2/verify_otp"
            verify_resp = _requests.post(verify_url, json=verify_payload, timeout=30)
            verify_data = verify_resp.json()
            verify_key = verify_data.get("request_key", "")

            # Step 1c -- verify PIN and get auth code
            pin_payload = {
                "request_key": verify_key,
                "identity_type": "pin",
                "identifier": hashlib.sha256(self._password.encode()).hexdigest(),
                "recaptcha_token": "",
            }
            pin_url = "https://api-t2.fyers.in/vagator/v2/verify_pin_v2"
            pin_resp = _requests.post(pin_url, json=pin_payload, timeout=30)
            pin_data = pin_resp.json()
            access_token_intermediate = pin_data.get("data", {}).get("access_token", "")

            # Step 2 -- exchange for final access token
            token_payload = {
                "fyers_id": self._user_id,
                "app_id": self._client_id.split("-")[0],
                "redirect_uri": self._redirect_uri,
                "appType": self._client_id.split("-")[1] if "-" in self._client_id else "100",
                "code_challenge": "",
                "state": "None",
                "scope": "",
                "nonce": "",
                "response_type": "code",
                "create_cookie": True,
            }
            token_headers = {"Authorization": f"Bearer {access_token_intermediate}"}
            token_url = "https://api-t1.fyers.in/api/v3/token"
            token_resp = _requests.post(
                token_url, json=token_payload, headers=token_headers, timeout=30
            )
            token_data = token_resp.json()
            auth_code = token_data.get("Url", "").split("auth_code=")[-1] if "Url" in token_data else ""

            if not auth_code:
                logger.error("Failed to extract auth_code from Fyers token response")
                return False

            # Step 2b -- generate final access token via session model
            session_model.set_token(auth_code)
            token_response = session_model.generate_token()
            access_token = token_response.get("access_token", "")

            if not access_token:
                logger.error("Fyers token generation returned no access_token")
                return False

            # Step 3 -- initialise FyersModel
            self._fyers = fyersModel.FyersModel(
                client_id=self._client_id,
                token=access_token,
                is_async=False,
                log_path="",
            )

            now = time.time()
            self._session = {
                "access_token": access_token,
                "expires_at": now + (6 * 60 * 60),
                "user_id": self._user_id,
                "authenticated_at": datetime.now(timezone.utc).isoformat(),
            }

            logger.info(
                "Fyers authenticated",
                extra={
                    "user_id": _mask(self._user_id),
                    "token_prefix": _mask(access_token, 6),
                },
            )
            return True

        except Exception:
            logger.exception(
                "Fyers authentication failed",
                extra={"user_id": _mask(self._user_id)},
            )
            return False

    def _ensure_session(self) -> None:
        """Re-authenticate if current session is expired."""
        if not self._auth.is_session_valid(self._session):
            logger.info("Fyers session expired; re-authenticating")
            self.authenticate()

    # ── Order Management ────────────────────────────────────────────────

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
        """Place an order via Fyers API.

        Args:
            symbol: Fyers symbol format (e.g. 'NSE:NIFTY2430618000CE').
            side: 'BUY' or 'SELL'.
            qty: Order quantity.
            order_type: 'MARKET', 'LIMIT', 'SL', or 'SL-M'.
            price: Limit price for LIMIT/SL orders.
            trigger_price: Stop trigger price for SL/SL-M orders.
            product: 'MIS', 'NRML', or 'CNC'.
            tag: Order tag for identification.

        Returns:
            Dict with 'order_id' and 'status'.
        """
        self._ensure_session()

        # Ensure symbol has exchange prefix
        if ":" not in symbol:
            symbol = f"NSE:{symbol}"

        order_data: dict[str, Any] = {
            "symbol": symbol,
            "qty": qty,
            "type": self._ORDER_TYPE_MAP.get(order_type.upper(), 2),
            "side": self._SIDE_MAP.get(side.upper(), 1),
            "productType": self._PRODUCT_MAP.get(product.upper(), "INTRADAY"),
            "validity": "DAY",
            "disclosedQty": 0,
            "offlineOrder": False,
        }
        if price is not None:
            order_data["limitPrice"] = price
        else:
            order_data["limitPrice"] = 0
        if trigger_price is not None:
            order_data["stopPrice"] = trigger_price
        else:
            order_data["stopPrice"] = 0

        logger.info(
            "Placing Fyers order",
            extra={
                "symbol": symbol,
                "side": side,
                "qty": qty,
                "order_type": order_type,
            },
        )

        try:
            response = self._fyers.place_order(data=order_data)
            order_id = response.get("id", "")
            status = "PLACED" if response.get("s") == "ok" else "FAILED"
            if status == "FAILED":
                logger.error(
                    "Fyers order placement failed",
                    extra={"response": response, "symbol": symbol},
                )
            else:
                logger.info("Fyers order placed", extra={"order_id": order_id})
            return {
                "order_id": str(order_id),
                "status": status,
                "message": response.get("message", ""),
            }
        except Exception as exc:
            logger.exception("Fyers order placement error", extra={"symbol": symbol})
            return {"order_id": "", "status": "FAILED", "error": str(exc)}

    def cancel_order(self, order_id: str) -> dict:
        """Cancel a pending order on Fyers.

        Returns:
            Dict with 'order_id' and 'status'.
        """
        self._ensure_session()
        logger.info("Cancelling Fyers order", extra={"order_id": order_id})
        try:
            response = self._fyers.cancel_order(data={"id": order_id})
            status = "CANCELLED" if response.get("s") == "ok" else "FAILED"
            logger.info("Fyers order cancel result", extra={"order_id": order_id, "status": status})
            return {"order_id": order_id, "status": status}
        except Exception as exc:
            logger.exception("Fyers order cancellation error", extra={"order_id": order_id})
            return {"order_id": order_id, "status": "FAILED", "error": str(exc)}

    # ── Position / Order / Portfolio ────────────────────────────────────

    def get_positions(self) -> list[dict]:
        """Fetch net positions from Fyers."""
        self._ensure_session()
        try:
            response = self._fyers.positions()
            positions_list = response.get("netPositions", [])
            logger.info("Fyers positions fetched", extra={"count": len(positions_list)})
            return [
                {
                    "symbol": p.get("symbol", ""),
                    "qty": p.get("netQty", 0),
                    "average_price": p.get("avgPrice", 0.0),
                    "ltp": p.get("ltp", 0.0),
                    "pnl": p.get("pl", 0.0),
                    "product": p.get("productType", ""),
                    "side": "BUY" if p.get("side", 0) == 1 else "SELL",
                }
                for p in positions_list
            ]
        except Exception:
            logger.exception("Failed to fetch Fyers positions")
            return []

    def get_orders(self) -> list[dict]:
        """Fetch today's orders from Fyers."""
        self._ensure_session()
        try:
            response = self._fyers.orderbook()
            orders_list = response.get("orderBook", [])
            logger.info("Fyers orders fetched", extra={"count": len(orders_list)})
            return [
                {
                    "order_id": o.get("id", ""),
                    "symbol": o.get("symbol", ""),
                    "side": "BUY" if o.get("side") == 1 else "SELL",
                    "qty": o.get("qty", 0),
                    "price": o.get("limitPrice", 0.0),
                    "status": str(o.get("status", "")),
                    "timestamp": o.get("orderDateTime", ""),
                    "order_type": o.get("type", ""),
                }
                for o in orders_list
            ]
        except Exception:
            logger.exception("Failed to fetch Fyers orders")
            return []

    def get_portfolio(self) -> dict:
        """Fetch holdings, positions, and fund details."""
        self._ensure_session()
        portfolio: dict[str, Any] = {"holdings": [], "positions": [], "margins": {}}
        try:
            holdings_resp = self._fyers.holdings()
            portfolio["holdings"] = holdings_resp.get("holdings", [])
        except Exception:
            logger.exception("Failed to fetch Fyers holdings")
        try:
            positions_resp = self._fyers.positions()
            portfolio["positions"] = positions_resp.get("netPositions", [])
        except Exception:
            logger.exception("Failed to fetch Fyers positions for portfolio")
        try:
            funds_resp = self._fyers.funds()
            portfolio["margins"] = funds_resp.get("fund_limit", {})
        except Exception:
            logger.exception("Failed to fetch Fyers funds")
        logger.info(
            "Fyers portfolio fetched",
            extra={
                "holdings": len(portfolio["holdings"]),
                "positions": len(portfolio["positions"]),
            },
        )
        return portfolio

    # ── Market Data ─────────────────────────────────────────────────────

    def get_ltp(self, symbols: list[str]) -> dict[str, float]:
        """Get last traded prices for symbols.

        Args:
            symbols: List of Fyers symbols (e.g. ['NSE:NIFTY2430618000CE']).

        Returns:
            Mapping of symbol -> LTP.
        """
        self._ensure_session()
        try:
            # Ensure exchange prefix
            qualified = [s if ":" in s else f"NSE:{s}" for s in symbols]
            data = {"symbols": ",".join(qualified)}
            response = self._fyers.quotes(data=data)
            result: dict[str, float] = {}
            for quote in response.get("d", []):
                sym = quote.get("n", "")
                ltp = quote.get("v", {}).get("lp", 0.0)
                # Strip exchange prefix for consistency
                clean_sym = sym.split(":")[-1] if ":" in sym else sym
                result[clean_sym] = float(ltp)
            logger.debug("Fyers LTP fetched", extra={"count": len(result)})
            return result
        except Exception:
            logger.exception("Failed to fetch Fyers LTP")
            return {}

    def get_option_chain(self, symbol: str, expiry: str) -> dict:
        """Fetch option chain for an underlying on a given expiry via Fyers API.

        Args:
            symbol: Underlying symbol (e.g. 'NIFTY').
            expiry: Expiry date in 'YYYY-MM-DD' format.

        Returns:
            Dict keyed by strike, each with 'CE' and 'PE' sub-dicts.
        """
        self._ensure_session()
        chain: dict[float, dict] = {}
        try:
            data = {
                "symbol": f"NSE:{symbol}-INDEX",
                "strikecount": 30,
                "timestamp": expiry,
            }
            response = self._fyers.optionchain(data=data)
            option_data = response.get("data", {}).get("optionsChain", [])

            for opt in option_data:
                strike = float(opt.get("strikePrice", 0))
                opt_type = opt.get("option_type", "")  # CE or PE
                if opt_type not in ("CE", "PE"):
                    continue
                chain.setdefault(strike, {"CE": {}, "PE": {}})
                chain[strike][opt_type] = {
                    "symbol": opt.get("symbol", ""),
                    "ltp": opt.get("ltp", 0.0),
                    "oi": opt.get("openInterest", 0),
                    "volume": opt.get("volume", 0),
                    "iv": opt.get("iv", 0.0),
                    "bid": opt.get("bid", 0.0),
                    "ask": opt.get("ask", 0.0),
                }

            logger.info(
                "Fyers option chain built",
                extra={"symbol": symbol, "expiry": expiry, "strikes": len(chain)},
            )
        except Exception:
            logger.exception(
                "Failed to build Fyers option chain",
                extra={"symbol": symbol, "expiry": expiry},
            )
        return chain

    # ── WebSocket Tick Streaming ────────────────────────────────────────

    def subscribe_ticks(
        self,
        symbols: list[str],
        callback: Callable[[list[dict]], Any],
    ) -> None:
        """Subscribe to live ticks via Fyers data socket.

        Args:
            symbols: List of Fyers symbols.
            callback: Invoked with list of tick dicts on each update.
        """
        self._ensure_session()
        self._tick_callback = callback
        # Ensure exchange prefix
        self._subscribed_symbols = [s if ":" in s else f"NSE:{s}" for s in symbols]

        try:
            from fyers_apiv3.FyersWebsocket import data_ws

            self._data_socket = data_ws.FyersDataSocket(
                access_token=f"{self._client_id}:{self._session['access_token']}",
                log_path="",
                litemode=False,
                write_to_file=False,
                reconnect=True,
                on_connect=self._on_ws_connect,
                on_close=self._on_ws_close,
                on_error=self._on_ws_error,
                on_message=self._on_ws_message,
            )

            logger.info(
                "Starting Fyers data socket",
                extra={"symbols": len(self._subscribed_symbols)},
            )
            # Connect in a background thread
            ws_thread = threading.Thread(
                target=self._data_socket.connect,
                daemon=True,
                name="fyers-ws",
            )
            ws_thread.start()

        except Exception:
            logger.exception("Failed to start Fyers data socket")

    def _on_ws_connect(self) -> None:
        """Called when Fyers WebSocket connects."""
        logger.info("Fyers data socket connected")
        self._reconnect_attempts = 0
        if self._data_socket and self._subscribed_symbols:
            self._data_socket.subscribe(symbols=self._subscribed_symbols, data_type="SymbolUpdate")
            logger.info(
                "Fyers symbols subscribed",
                extra={"count": len(self._subscribed_symbols)},
            )

    def _on_ws_message(self, message: Any) -> None:
        """Called on each message from Fyers data socket."""
        if self._tick_callback is None:
            return
        # Normalise to list of dicts
        ticks: list[dict] = []
        if isinstance(message, dict):
            ticks = [message]
        elif isinstance(message, list):
            ticks = message
        try:
            self._tick_callback(ticks)
        except Exception:
            logger.exception("Error in Fyers tick callback")

    def _on_ws_close(self) -> None:
        """Called when Fyers WebSocket closes -- attempt reconnect with backoff."""
        self._reconnect_attempts += 1
        delay = min(
            WEBSOCKET_RECONNECT_BASE_DELAY_S * (WEBSOCKET_RECONNECT_MULTIPLIER ** self._reconnect_attempts),
            WEBSOCKET_RECONNECT_MAX_DELAY_S,
        )
        logger.warning(
            "Fyers data socket closed, will reconnect",
            extra={"attempt": self._reconnect_attempts, "delay_seconds": delay},
        )

    def _on_ws_error(self, error: Any) -> None:
        """Called on Fyers WebSocket error."""
        logger.error("Fyers data socket error", extra={"error": str(error)})

    def unsubscribe_ticks(self, symbols: list[str]) -> None:
        """Unsubscribe from live ticks."""
        qualified = [s if ":" in s else f"NSE:{s}" for s in symbols]
        if self._data_socket:
            try:
                self._data_socket.unsubscribe(symbols=qualified)
                self._subscribed_symbols = [
                    s for s in self._subscribed_symbols if s not in qualified
                ]
                logger.info("Fyers symbols unsubscribed", extra={"count": len(qualified)})
            except Exception:
                logger.exception("Failed to unsubscribe Fyers symbols")

    # ── Teardown ────────────────────────────────────────────────────────

    def close(self) -> None:
        """Close data socket and release resources."""
        if self._data_socket:
            try:
                self._data_socket.close_connection()
                logger.info("Fyers data socket closed")
            except Exception:
                logger.exception("Error closing Fyers data socket")
            self._data_socket = None
        self._fyers = None
        self._session = {}
        logger.info("FyersBroker closed")
