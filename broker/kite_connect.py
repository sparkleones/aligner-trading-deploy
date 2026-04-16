"""
Zerodha Kite Connect broker implementation.

Wraps the kiteconnect Python SDK to provide order management, position
tracking, option chain construction, and WebSocket tick streaming with
automatic reconnection using exponential backoff.
"""

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


class KiteConnectBroker(BaseBroker):
    """Broker adapter for Zerodha Kite Connect.

    Uses the ``kiteconnect`` SDK for REST calls and ``KiteTicker`` for
    real-time WebSocket streaming.

    Args:
        api_key: Kite Connect API key. Loaded from settings if not provided.
        api_secret: Kite Connect API secret.
        user_id: Zerodha client ID.
        password: Zerodha password.
        totp_secret: Base32 TOTP secret for 2FA.
    """

    # Maps from our canonical order types to Kite constants
    _ORDER_TYPE_MAP = {
        "MARKET": "MARKET",
        "LIMIT": "LIMIT",
        "SL": "SL",
        "SL-M": "SL-M",
    }
    _SIDE_MAP = {"BUY": "BUY", "SELL": "SELL"}
    _PRODUCT_MAP = {"MIS": "MIS", "NRML": "NRML", "CNC": "CNC"}

    def __init__(
        self,
        api_key: str = "",
        api_secret: str = "",
        user_id: str = "",
        password: str = "",
        totp_secret: str = "",
    ) -> None:
        settings = load_settings()
        self._api_key = api_key or settings.broker.api_key
        self._api_secret = api_secret or settings.broker.api_secret
        self._user_id = user_id or settings.broker.user_id
        self._password = password or settings.broker.password
        self._totp_secret = totp_secret or settings.broker.totp_secret

        self._kite: Any = None  # kiteconnect.KiteConnect instance
        self._ticker: Any = None  # kiteconnect.KiteTicker instance
        self._session: dict = {}
        self._auth = TOTPAuthenticator(self._totp_secret)

        # instrument_token cache: symbol -> token
        self._token_map: dict[str, int] = {}
        self._reverse_token_map: dict[int, str] = {}
        self._instruments_loaded = False

        # tick subscription state
        self._tick_callback: Optional[Callable] = None
        self._subscribed_tokens: list[int] = []
        self._ws_lock = threading.Lock()

        logger.info(
            "KiteConnectBroker initialised",
            extra={"user_id": _mask(self._user_id), "api_key": _mask(self._api_key)},
        )

    # ── Authentication ──────────────────────────────────────────────────

    def authenticate(self) -> bool:
        """Login to Kite Connect using API key + TOTP and obtain access token.

        Returns:
            True if authentication succeeds.
        """
        try:
            from kiteconnect import KiteConnect

            self._kite = KiteConnect(api_key=self._api_key)

            # Generate TOTP and get request token
            totp_value = self._auth.generate_totp()

            # Authenticate session
            self._session = self._auth.authenticate_session(
                api_key=self._api_key,
                api_secret=self._api_secret,
                user_id=self._user_id,
                password=self._password,
                totp_secret=self._totp_secret,
            )

            self._kite.set_access_token(self._session["access_token"])
            logger.info(
                "Kite Connect authenticated",
                extra={"user_id": _mask(self._user_id)},
            )

            # Pre-load instrument token mapping
            self._load_instruments()
            return True

        except Exception:
            logger.exception(
                "Kite Connect authentication failed",
                extra={"user_id": _mask(self._user_id)},
            )
            return False

    def _ensure_session(self) -> None:
        """Re-authenticate if the current session has expired."""
        self._session = self._auth.refresh_if_needed(
            self._session,
            api_key=self._api_key,
            api_secret=self._api_secret,
            user_id=self._user_id,
            password=self._password,
            totp_secret=self._totp_secret,
        )
        if self._kite is not None:
            self._kite.set_access_token(self._session["access_token"])

    # ── Instrument Token Mapping ────────────────────────────────────────

    def _load_instruments(self, exchange: str = "NFO") -> None:
        """Download instrument list and build symbol <-> token maps.

        Loads both the requested exchange (default NFO for options) and NSE
        (for index underlyings like 'NIFTY 50', 'NIFTY BANK').
        """
        try:
            instruments = self._kite.instruments(exchange)
            for inst in instruments:
                symbol = inst.get("tradingsymbol", "")
                token = inst.get("instrument_token")
                if symbol and token is not None:
                    self._token_map[symbol] = int(token)
                    self._reverse_token_map[int(token)] = symbol
            logger.info(
                "Instruments loaded",
                extra={"exchange": exchange, "count": len(self._token_map)},
            )
        except Exception:
            logger.exception("Failed to load instruments", extra={"exchange": exchange})

        # Also load NSE instruments for index underlyings (NIFTY 50, NIFTY BANK, etc.)
        if exchange != "NSE":
            try:
                nse_instruments = self._kite.instruments("NSE")
                nse_count = 0
                for inst in nse_instruments:
                    symbol = inst.get("tradingsymbol", "")
                    token = inst.get("instrument_token")
                    if symbol and token is not None:
                        self._token_map[symbol] = int(token)
                        self._reverse_token_map[int(token)] = symbol
                        nse_count += 1
                logger.info(
                    "Instruments loaded",
                    extra={"exchange": "NSE", "count": nse_count},
                )
            except Exception:
                logger.exception("Failed to load NSE instruments")

        self._instruments_loaded = True

    def _resolve_tokens(self, symbols: list[str]) -> list[int]:
        """Resolve a list of trading symbols to instrument tokens."""
        if not self._instruments_loaded:
            self._load_instruments()
        tokens: list[int] = []
        for sym in symbols:
            token = self._token_map.get(sym)
            if token is not None:
                tokens.append(token)
            else:
                logger.warning("Symbol not found in instrument map", extra={"symbol": sym})
        return tokens

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
        """Place an order via Kite Connect.

        Returns:
            Dict with 'order_id' and 'status'.
        """
        self._ensure_session()

        # Kite API no longer allows naked MARKET orders — convert to LIMIT
        # with a slippage buffer so execution is near-instant.
        # NFO tick size is 0.05 — all prices must be rounded to nearest tick.
        def _tick_round(p: float, buy: bool) -> float:
            """Round to nearest 0.05 tick. BUY rounds up, SELL rounds down."""
            import math
            if buy:
                return max(0.05, math.ceil(p / 0.05) * 0.05)
            return max(0.05, math.floor(p / 0.05) * 0.05)

        effective_order_type = order_type.upper()
        effective_price = price
        if effective_order_type == "MARKET":
            is_buy = side.upper() == "BUY"
            if effective_price and effective_price > 0:
                # Price already provided (e.g. from BS pricing) — use as LIMIT
                slippage = 0.02
                raw = effective_price * (1 + slippage) if is_buy else effective_price * (1 - slippage)
                effective_price = _tick_round(raw, is_buy)
                effective_order_type = "LIMIT"
                logger.info("MARKET->LIMIT (price provided) | %s limit=%.2f", symbol, effective_price)
            else:
                # No price — fetch LTP
                try:
                    ltp_data = self.get_ltp([symbol])
                    ltp = ltp_data.get(symbol, 0)
                    if ltp > 0:
                        slippage = 0.02
                        raw = ltp * (1 + slippage) if is_buy else ltp * (1 - slippage)
                        effective_price = _tick_round(raw, is_buy)
                        effective_order_type = "LIMIT"
                        logger.info("MARKET->LIMIT (LTP) | %s LTP=%.2f limit=%.2f",
                                    symbol, ltp, effective_price)
                except Exception as e:
                    logger.warning("LTP fetch for MARKET->LIMIT failed: %s", e)

        kite_params: dict[str, Any] = {
            "tradingsymbol": symbol,
            "exchange": "NFO",
            "transaction_type": self._SIDE_MAP.get(side.upper(), side.upper()),
            "quantity": qty,
            "order_type": self._ORDER_TYPE_MAP.get(effective_order_type, effective_order_type),
            "product": self._PRODUCT_MAP.get(product.upper(), product.upper()),
            "variety": "regular",
        }
        if effective_price is not None:
            kite_params["price"] = effective_price
        if trigger_price is not None:
            kite_params["trigger_price"] = trigger_price
        if tag:
            kite_params["tag"] = tag[:20]

        logger.info(
            "Placing order",
            extra={
                "symbol": symbol,
                "side": side,
                "qty": qty,
                "order_type": order_type,
                "product": product,
            },
        )

        try:
            order_id = self._kite.place_order(**kite_params)
            logger.info("Order placed", extra={"order_id": order_id, "symbol": symbol})
            return {"order_id": str(order_id), "status": "PLACED"}
        except Exception as exc:
            logger.exception("Order placement failed", extra={"symbol": symbol, "error": str(exc)})
            return {"order_id": "", "status": "FAILED", "error": str(exc)}

    def cancel_order(self, order_id: str) -> dict:
        """Cancel a pending order.

        Returns:
            Dict with 'order_id' and 'status'.
        """
        self._ensure_session()
        logger.info("Cancelling order", extra={"order_id": order_id})
        try:
            self._kite.cancel_order(variety="regular", order_id=order_id)
            logger.info("Order cancelled", extra={"order_id": order_id})
            return {"order_id": order_id, "status": "CANCELLED"}
        except Exception as exc:
            logger.exception("Order cancellation failed", extra={"order_id": order_id})
            return {"order_id": order_id, "status": "FAILED", "error": str(exc)}

    # ── Position / Order / Portfolio ────────────────────────────────────

    def get_positions(self) -> list[dict]:
        """Fetch day positions with proper intraday P&L fields.

        Uses Kite's ``day`` positions for accurate intraday P&L breakdown
        (realised, unrealised, m2m).  Falls back to ``net`` if ``day`` is
        empty so that BTST/NRML carry-forward positions are still visible.
        """
        self._ensure_session()
        try:
            positions = self._kite.positions()
            day = positions.get("day", [])
            net = positions.get("net", [])

            # Build a lookup from day positions (accurate intraday P&L)
            day_lookup: dict[str, dict] = {}
            for p in day:
                key = f"{p.get('tradingsymbol', '')}_{p.get('product', '')}"
                day_lookup[key] = p

            # Use net as the base (has all positions incl. carry-forward)
            result = []
            for p in net:
                sym = p.get("tradingsymbol", "")
                product = p.get("product", "")
                key = f"{sym}_{product}"

                # Prefer day-position P&L fields for intraday accuracy
                dp = day_lookup.get(key)

                result.append({
                    "symbol": sym,
                    "qty": p.get("quantity", 0),
                    "average_price": p.get("average_price", 0.0),
                    "ltp": p.get("last_price", 0.0),
                    # Day P&L fields (from day positions if available)
                    "pnl": dp.get("pnl", p.get("pnl", 0.0)) if dp else p.get("pnl", 0.0),
                    "realised": dp.get("realised", 0.0) if dp else 0.0,
                    "unrealised": dp.get("unrealised", 0.0) if dp else p.get("unrealised", 0.0),
                    "m2m": dp.get("m2m", 0.0) if dp else p.get("m2m", 0.0),
                    "day_buy_qty": dp.get("day_buy_quantity", 0) if dp else p.get("day_buy_quantity", 0),
                    "day_sell_qty": dp.get("day_sell_quantity", 0) if dp else p.get("day_sell_quantity", 0),
                    "day_buy_value": dp.get("day_buy_value", 0.0) if dp else p.get("day_buy_value", 0.0),
                    "day_sell_value": dp.get("day_sell_value", 0.0) if dp else p.get("day_sell_value", 0.0),
                    "product": product,
                    "exchange": p.get("exchange", ""),
                    # Turnover fields for charge estimation
                    "buy_value": p.get("buy_value", 0.0),
                    "sell_value": p.get("sell_value", 0.0),
                    "buy_quantity": p.get("buy_quantity", 0),
                    "sell_quantity": p.get("sell_quantity", 0),
                })

            logger.debug("Positions fetched | net=%d day=%d merged=%d",
                         len(net), len(day), len(result))
            return result
        except Exception:
            logger.exception("Failed to fetch positions")
            return []

    def get_orders(self) -> list[dict]:
        """Fetch today's orders."""
        self._ensure_session()
        try:
            orders = self._kite.orders()
            logger.info("Orders fetched", extra={"count": len(orders)})
            return [
                {
                    "order_id": o.get("order_id", ""),
                    "symbol": o.get("tradingsymbol", ""),
                    "side": o.get("transaction_type", ""),
                    "qty": o.get("quantity", 0),
                    "price": o.get("price", 0.0),
                    "fill_price": o.get("average_price", 0.0),
                    "status": o.get("status", ""),
                    "status_message": o.get("status_message", ""),
                    "timestamp": str(o.get("order_timestamp", "")),
                    "tag": o.get("tag", ""),
                }
                for o in orders
            ]
        except Exception:
            logger.exception("Failed to fetch orders")
            return []

    def get_portfolio(self) -> dict:
        """Fetch holdings, positions, and available margins."""
        self._ensure_session()
        portfolio: dict[str, Any] = {"holdings": [], "positions": [], "margins": {}}
        try:
            portfolio["holdings"] = self._kite.holdings()
        except Exception:
            logger.exception("Failed to fetch holdings")
        try:
            positions = self._kite.positions()
            portfolio["positions"] = positions.get("net", [])
        except Exception:
            logger.exception("Failed to fetch positions for portfolio")
        try:
            margins = self._kite.margins()
            portfolio["margins"] = margins
        except Exception:
            logger.exception("Failed to fetch margins")
        logger.info(
            "Portfolio fetched",
            extra={
                "holdings": len(portfolio["holdings"]),
                "positions": len(portfolio["positions"]),
            },
        )
        return portfolio

    # ── Market Data ─────────────────────────────────────────────────────

    # Known NSE index underlyings (tradingsymbol as they appear on NSE exchange)
    _NSE_INDEX_SYMBOLS = {"NIFTY 50", "NIFTY BANK", "NIFTY FIN SERVICE", "INDIA VIX"}

    def get_ltp(self, symbols: list[str]) -> dict[str, float]:
        """Get last traded price for a list of symbols.

        Args:
            symbols: Trading symbols. Index underlyings (e.g. 'NIFTY 50')
                     are looked up on NSE; option instruments on NFO.

        Returns:
            Dict mapping symbol -> LTP.
        """
        self._ensure_session()

        qualified = []
        for s in symbols:
            if s in self._NSE_INDEX_SYMBOLS:
                qualified.append(f"NSE:{s}")
            else:
                qualified.append(f"NFO:{s}")

        # Try ltp() first, fall back to ohlc() or quote() if permissions deny it
        for method_name in ("ltp", "ohlc", "quote"):
            try:
                method = getattr(self._kite, method_name)
                data = method(qualified)
                result: dict[str, float] = {}
                for sym in symbols:
                    for prefix in ("NSE:", "NFO:"):
                        key = f"{prefix}{sym}"
                        if key in data:
                            entry = data[key]
                            price = entry.get("last_price", 0.0)
                            if price == 0.0 and "ohlc" in entry:
                                price = entry["ohlc"].get("close", 0.0)
                            if price > 0:
                                result[sym] = float(price)
                                break
                if result:
                    logger.debug("LTP fetched via %s", method_name, extra={"count": len(result)})
                    return result
            except Exception as e:
                if "PermissionException" in type(e).__name__ or "Insufficient permission" in str(e):
                    logger.debug("LTP method %s denied, trying fallback", method_name)
                    continue
                logger.exception("Failed to fetch LTP via %s", method_name)
                return {}

        logger.error("All LTP methods failed (ltp/ohlc/quote)")
        return {}

    def get_historical_data(
        self,
        symbol: str,
        from_dt: datetime,
        to_dt: datetime,
        interval: str = "minute",
    ) -> list[dict]:
        """Fetch historical OHLCV candles from Kite Connect.

        Args:
            symbol: Trading symbol (e.g. 'NIFTY 50').
            from_dt: Start datetime.
            to_dt: End datetime.
            interval: Candle interval — 'minute', '3minute', '5minute',
                      '15minute', '30minute', '60minute', 'day'.

        Returns:
            List of bar dicts with keys: time, open, high, low, close, volume.
        """
        self._ensure_session()
        if not self._instruments_loaded:
            self._load_instruments()

        token = self._token_map.get(symbol)
        if token is None:
            logger.error("No instrument token for %s", symbol)
            return []

        try:
            records = self._kite.historical_data(
                instrument_token=token,
                from_date=from_dt,
                to_date=to_dt,
                interval=interval,
            )
            bars = []
            for r in records:
                ts = r.get("date")
                if hasattr(ts, "strftime"):
                    ts = ts.strftime("%Y-%m-%d %H:%M:%S")
                bars.append({
                    "time": str(ts),
                    "timestamp": str(ts),
                    "open": float(r.get("open", 0)),
                    "high": float(r.get("high", 0)),
                    "low": float(r.get("low", 0)),
                    "close": float(r.get("close", 0)),
                    "volume": int(r.get("volume", 0)),
                })
            logger.info("Historical data fetched | %s %s | %d bars",
                        symbol, interval, len(bars))
            return bars
        except Exception:
            logger.exception("Failed to fetch historical data for %s", symbol)
            return []

    def get_option_chain(self, symbol: str, expiry: str) -> dict:
        """Build an option chain for *symbol* on *expiry* from cached instruments.

        Args:
            symbol: Underlying (e.g. 'NIFTY').
            expiry: Expiry date 'YYYY-MM-DD'.

        Returns:
            Dict keyed by strike price (float), each containing 'CE' and 'PE'
            sub-dicts with 'instrument_token', 'tradingsymbol', 'lot_size'.
        """
        self._ensure_session()
        if not self._instruments_loaded:
            self._load_instruments()

        chain: dict[float, dict] = {}
        try:
            instruments = self._kite.instruments("NFO")
            expiry_dt = datetime.strptime(expiry, "%Y-%m-%d").date()

            for inst in instruments:
                if inst.get("name") != symbol:
                    continue
                inst_expiry = inst.get("expiry")
                if inst_expiry is None:
                    continue
                if hasattr(inst_expiry, "date"):
                    inst_expiry = inst_expiry  # already a date
                elif isinstance(inst_expiry, str):
                    inst_expiry = datetime.strptime(inst_expiry, "%Y-%m-%d").date()
                if inst_expiry != expiry_dt:
                    continue

                strike = float(inst.get("strike", 0))
                opt_type = inst.get("instrument_type", "")  # CE or PE
                if opt_type not in ("CE", "PE"):
                    continue

                chain.setdefault(strike, {"CE": {}, "PE": {}})
                chain[strike][opt_type] = {
                    "instrument_token": inst.get("instrument_token"),
                    "tradingsymbol": inst.get("tradingsymbol", ""),
                    "lot_size": inst.get("lot_size", 0),
                    "exchange": inst.get("exchange", "NFO"),
                }

            # Log the actual lot size from exchange instruments
            sample_lot = 0
            for s_data in chain.values():
                for ot_data in s_data.values():
                    if isinstance(ot_data, dict) and ot_data.get("lot_size"):
                        sample_lot = ot_data["lot_size"]
                        break
                if sample_lot:
                    break
            logger.info(
                "Option chain built | %s expiry=%s strikes=%d lot_size=%d",
                symbol, expiry, len(chain), sample_lot,
            )
        except Exception:
            logger.exception(
                "Failed to build option chain",
                extra={"symbol": symbol, "expiry": expiry},
            )
        return chain

    # ── WebSocket Tick Streaming ────────────────────────────────────────

    def subscribe_ticks(
        self,
        symbols: list[str],
        callback: Callable[[list[dict]], Any],
    ) -> None:
        """Subscribe to real-time ticks via KiteTicker WebSocket.

        Args:
            symbols: Symbols to subscribe (resolved to instrument tokens).
            callback: Invoked with a list of tick dicts on each update.
        """
        tokens = self._resolve_tokens(symbols)
        if not tokens:
            # Retry: reload instruments (NSE may have failed on first load)
            logger.warning("No tokens resolved — reloading instruments and retrying")
            self._instruments_loaded = False
            self._load_instruments()
            tokens = self._resolve_tokens(symbols)
        if not tokens:
            logger.error("No valid tokens resolved after retry; tick subscription skipped")
            return

        self._tick_callback = callback
        self._subscribed_tokens = tokens
        self._ensure_session()

        # Close existing ticker before creating a new one
        if self._ticker is not None:
            try:
                self._ticker.close()
            except Exception:
                pass
            self._ticker = None

        from kiteconnect import KiteTicker

        self._ticker = KiteTicker(
            api_key=self._api_key,
            access_token=self._session["access_token"],
        )

        self._ticker.on_ticks = self._on_ticks
        self._ticker.on_connect = self._on_connect
        self._ticker.on_close = self._on_close
        self._ticker.on_error = self._on_error

        # Enable auto-reconnect with exponential backoff
        if hasattr(self._ticker, "enable_reconnect"):
            self._ticker.enable_reconnect(
                reconnect_interval=int(WEBSOCKET_RECONNECT_BASE_DELAY_S),
                reconnect_tries=50,
            )
        else:
            # Newer kiteconnect versions use class attributes
            self._ticker.RECONNECT_MAX_DELAY = int(WEBSOCKET_RECONNECT_MAX_DELAY_S)
            self._ticker.RECONNECT_MAX_TRIES = 50

        logger.info(
            "Starting KiteTicker WebSocket",
            extra={"tokens": len(tokens)},
        )
        # connect(threaded=True) runs the WS loop in its own daemon thread
        self._ticker.connect(threaded=True)

    def _on_connect(self, ws: Any, response: Any) -> None:
        """Called when WebSocket connection is established."""
        logger.info("KiteTicker connected, subscribing tokens=%s", self._subscribed_tokens)
        ws.subscribe(self._subscribed_tokens)
        ws.set_mode(ws.MODE_FULL, self._subscribed_tokens)
        logger.info("KiteTicker subscribed and mode set to FULL")

    def _on_ticks(self, ws: Any, ticks: list[dict]) -> None:
        """Called on each tick batch from KiteTicker."""
        if self._tick_callback is None:
            return
        # Enrich ticks with human-readable symbols
        enriched: list[dict] = []
        for t in ticks:
            token = t.get("instrument_token")
            t["symbol"] = self._reverse_token_map.get(token, str(token))
            enriched.append(t)
        try:
            self._tick_callback(enriched)
        except Exception:
            logger.exception("Error in tick callback")

    def _on_close(self, ws: Any, code: int, reason: str) -> None:
        """Called when WebSocket connection closes."""
        logger.warning(
            "KiteTicker disconnected",
            extra={"code": code, "reason": reason},
        )

    def _on_error(self, ws: Any, code: int, reason: str) -> None:
        """Called on WebSocket error."""
        logger.error(
            "KiteTicker error",
            extra={"code": code, "reason": reason},
        )

    def add_tick_subscription(self, symbols: list[str]) -> None:
        """Add symbols to the existing KiteTicker WebSocket subscription.

        Unlike subscribe_ticks(), this does NOT restart the WebSocket.
        It adds new tokens to the already-running connection.
        """
        tokens = self._resolve_tokens(symbols)
        if not tokens:
            return
        # Filter out already-subscribed tokens
        new_tokens = [t for t in tokens if t not in self._subscribed_tokens]
        if not new_tokens:
            return
        if self._ticker is not None:
            try:
                self._ticker.subscribe(new_tokens)
                self._ticker.set_mode(self._ticker.MODE_FULL, new_tokens)
                self._subscribed_tokens.extend(new_tokens)
                syms = [self._reverse_token_map.get(t, str(t)) for t in new_tokens]
                logger.info("Added tick subscription: %s (%d tokens)", syms, len(new_tokens))
            except Exception:
                logger.exception("Failed to add tick subscription")
        else:
            logger.warning("No active ticker — cannot add subscription for %s", symbols)

    def unsubscribe_ticks(self, symbols: list[str]) -> None:
        """Unsubscribe from symbols on the live WebSocket."""
        tokens = self._resolve_tokens(symbols)
        if self._ticker and tokens:
            try:
                self._ticker.unsubscribe(tokens)
                self._subscribed_tokens = [
                    t for t in self._subscribed_tokens if t not in tokens
                ]
                logger.info("Unsubscribed tokens", extra={"count": len(tokens)})
            except Exception:
                logger.exception("Failed to unsubscribe tokens")

    # ── Teardown ────────────────────────────────────────────────────────

    def close(self) -> None:
        """Close WebSocket and release resources."""
        if self._ticker:
            try:
                self._ticker.close()
                logger.info("KiteTicker WebSocket closed")
            except Exception:
                logger.exception("Error closing KiteTicker")
            self._ticker = None
        self._kite = None
        self._session = {}
        logger.info("KiteConnectBroker closed")
