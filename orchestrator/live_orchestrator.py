"""
Live Trading Orchestrator — runs multiple strategy agents on real market data.

Connects to Zerodha Kite Connect, subscribes to real-time ticks,
fans out market data to parallel strategy agents, aggregates signals
via a meta-agent, and executes trades with risk management.

Now integrates MarketAnalyzer for comprehensive Indian options analysis:
VIX regime, PCR, Max Pain, OI levels, IV percentile/skew, EMA trend,
RSI, VWAP, Supertrend, and intraday timing.

Architecture:
  KiteTicker (thread) --> tick_queue --> tick_dispatcher --> agent_queues
  MarketAnalyzer runs on each completed bar
  agent(bar, analysis) --> signal_queue --> meta_agent --> risk_check --> broker
"""

import asyncio
import logging
import queue
import time
from datetime import datetime, timedelta, date, time as dt_time
from typing import Any, Callable, Coroutine, Optional, Union

from config.constants import (
    GST_RATE,
    INDEX_CONFIG,
    MARKET_CLOSE_HOUR,
    MARKET_CLOSE_MINUTE,
    MARKET_OPEN_HOUR,
    MARKET_OPEN_MINUTE,
    NSE_TRANSACTION_CHARGE,
    SEBI_TURNOVER_FEE,
    STAMP_DUTY_BUY,
    STT_RATES,
)
from orchestrator.market_analyzer import MarketAnalyzer, MarketAnalysis
from orchestrator.meta_agent import MetaAgent
from orchestrator.order_manager import OrderManager
from orchestrator.position_tracker import PositionTracker
from orchestrator.rate_limiter import AsyncRateLimiter
from orchestrator.smart_strike_selector import enrich_option_chain_with_quotes
from orchestrator.trade_signal import OrderLeg, TradeExecution, TradeSignal
from orchestrator.strategy_agents import AGENT_REGISTRY
from orchestrator.strategy_agents.base_agent import BaseLiveAgent
from risk_management.risk_manager import Position, RiskManager

logger = logging.getLogger(__name__)

WIND_DOWN_TIME = dt_time(15, 25)  # no new positions after 15:25
SQUARE_OFF_TIME = dt_time(15, 28)  # force close at 15:28


class LiveTradingOrchestrator:
    """Orchestrates multiple strategy agents for live trading.

    Parameters
    ----------
    broker : object
        Authenticated broker instance (KiteConnectBroker or PaperTradingBroker).
    capital : float
        Starting capital.
    strategies : list[str]
        Strategy agent names to activate.
    callback : callable
        Async callback to emit events to the dashboard.
    """

    def __init__(
        self,
        broker: Any,
        capital: float = 200000.0,
        strategies: Optional[list[str]] = None,
        callback: Optional[Callable] = None,
        symbol: str = "NIFTY",
    ):
        self.broker = broker
        self.capital = capital
        self.symbol = symbol
        self.lot_size = INDEX_CONFIG.get(symbol, {}).get("lot_size", 65)

        # Detect paper mode early (needed for risk limits + analyzer)
        from backtesting.paper_trading import PaperTradingBroker
        self._is_paper = isinstance(broker, PaperTradingBroker)

        # Subsystems
        # Paper mode: 5% daily loss limit (options have wider MTM swings)
        # Live mode: 3% (tighter risk management with real money)
        # Paper mode: 8% kill switch (3 concurrent strategies, each up to 20% risk)
        # Live mode: 3% (tighter risk for real money)
        kill_pct = 0.08 if self._is_paper else 0.03
        self.risk_manager = RiskManager(total_capital=capital, max_daily_loss_pct=kill_pct)
        self.meta_agent = MetaAgent(capital=capital)
        self.rate_limiter = AsyncRateLimiter(max_ops=9)

        # Capital adequacy warning
        min_capital = self.lot_size * 50 * 3  # ~3 wings at 50-pt width
        if capital < min_capital:
            logger.warning(
                "Capital %.0f is LOW for %s (lot=%d). Recommended minimum: %.0f. "
                "Trades will be capital-constrained with narrow spreads.",
                capital, symbol, self.lot_size, min_capital,
            )

        # Market Analyzer — comprehensive Indian options analysis
        self.market_analyzer = MarketAnalyzer(symbol=symbol, capital=capital, is_paper=self._is_paper)
        self._latest_analysis: Optional[MarketAnalysis] = None

        # Live market data cache
        # IMPORTANT: default to 0.0 — a non-zero default silently masks
        # VIX fetch failures and lets entries slip through the V15
        # vix_floor=13 gate with a stale/fake number. A real VIX value
        # must arrive from the WebSocket (INDIA VIX subscription) or
        # from the 60s REST fallback before any trade can pass
        # scoring/engine.passes_confluence().
        self._live_vix: float = 0.0
        self._last_vix_tick_time: Optional[datetime] = None
        self._live_pcr: float = 1.0
        self._fii_net: float = 0.0
        self._dii_net: float = 0.0
        self._is_expiry_day: bool = False

        # Check if today is expiry day
        self._check_expiry_day()

        # Strategy agents
        strategy_names = strategies or ["learned_rules"]
        self.agents: dict[str, BaseLiveAgent] = {}
        for name in strategy_names:
            cls = AGENT_REGISTRY.get(name)
            if cls:
                self.agents[name] = cls(capital=capital, lot_size=self.lot_size)
                logger.info("Agent activated: %s (lot_size=%d)", name, self.lot_size)

        # Queues
        self._tick_queue: queue.Queue = queue.Queue(maxsize=10000)
        self._signal_queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
        self._agent_queues: dict[str, asyncio.Queue] = {
            name: asyncio.Queue(maxsize=5000) for name in self.agents
        }

        # State
        self._running = False
        self._shutdown = False
        self._bar_buffer: list[dict] = []
        self._current_bar: Optional[dict] = None
        self._bar_count = 0
        self._raw_tick_count = 0
        # Paper-mode termination: set True by _generate_synthetic_ticks
        # after it emits all 375 ticks. Main loop exits once this flag
        # is set AND the tick queue has drained. Necessary because the
        # aggregator buckets 375 one-minute ticks into ~75 five-minute
        # bars, so a bar-count threshold is unreachable.
        self._synthetic_ticks_exhausted = False
        # Real-time tick LTP cache (symbol --> last_price) for position PnL
        self._tick_ltp: dict[str, float] = {}
        self._option_chain: dict = {}
        self._open_trades: list[TradeExecution] = []
        self._closed_trades: list[TradeExecution] = []
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        # Dashboard callback
        self._callback = callback

        # Order manager and position tracker (injected by run_autonomous.py)
        self._order_manager: Optional[OrderManager] = None
        self._position_tracker: Optional[PositionTracker] = None

        # Hybrid "paper execution + real data" mode.
        # When --paper-live-data is set, run_autonomous.py creates a real
        # KiteConnectBroker for market data and injects it here. Order
        # execution still flows through self.broker (PaperTradingBroker),
        # so no real money is at risk — but the agent sees real NIFTY
        # ticks, real INDIA VIX, real historical bars, and a real
        # option chain. Used for Monday pre-deployment validation.
        self._data_broker: Optional[Any] = None

        # GTT backup stop-loss tracking: symbol -> gtt_id
        self._gtt_orders: dict[str, int] = {}

        # Config flags for safety features
        self._auto_close_orphans: bool = True
        self._use_gtt_backup_sl: bool = True
        self._gtt_sl_pct: float = 0.50  # 50% premium loss triggers GTT SL

        logger.info(
            "Orchestrator created | capital=%.0f lot=%d symbol=%s agents=%s",
            capital, self.lot_size, symbol, list(self.agents.keys()),
        )

    # ── Market-data broker routing ──
    def _md_broker(self) -> Any:
        """Return the broker to use for MARKET DATA calls.

        In live mode:              self.broker (KiteConnectBroker)
        In normal paper mode:      self.broker (PaperTradingBroker)
        In --paper-live-data mode: self._data_broker (real Kite for data)
                                   while self.broker stays paper for orders.

        This is the ONLY path for get_ltp/get_quote/get_vix/get_option_chain/
        get_historical_data/subscribe_ticks/add_tick_subscription/
        unsubscribe_ticks. Never call those on self.broker directly.
        """
        return self._data_broker if self._data_broker is not None else self.broker

    @property
    def _is_hybrid(self) -> bool:
        """True when running paper execution with real Kite market data."""
        return self._is_paper and self._data_broker is not None

    def _check_expiry_day(self) -> None:
        """Check if today is a weekly expiry day for the configured symbol."""
        config = INDEX_CONFIG.get(self.symbol, {})
        expiry_day = config.get("weekly_expiry_day", "Thursday")
        day_map = {
            "Monday": 0, "Tuesday": 1, "Wednesday": 2,
            "Thursday": 3, "Friday": 4,
        }
        self._is_expiry_day = date.today().weekday() == day_map.get(expiry_day, 3)
        if self._is_expiry_day:
            logger.info("Today is %s expiry day for %s", expiry_day, self.symbol)

    def _compute_dte(self) -> float:
        """Compute days to expiry for the nearest weekly expiry.

        Returns fractional days (e.g., 2.3 for 2 days + some hours).
        For paper mode, simulates a realistic mid-week day (2 DTE).
        """
        config = INDEX_CONFIG.get(self.symbol, {})
        expiry_day_name = config.get("weekly_expiry_day", "Thursday")
        day_map = {
            "Monday": 0, "Tuesday": 1, "Wednesday": 2,
            "Thursday": 3, "Friday": 4,
        }
        target_weekday = day_map.get(expiry_day_name, 3)
        today_weekday = date.today().weekday()

        if self._is_paper:
            # In paper mode, simulate Monday for NIFTY (1 day before Tuesday expiry)
            # This gives 1-2 DTE — realistic for weekly option strategies
            return 2.0

        # Calendar days to next expiry
        days_ahead = (target_weekday - today_weekday) % 7
        if days_ahead == 0:
            # Expiry day: compute hours remaining
            now = datetime.now()
            market_close = now.replace(hour=15, minute=30, second=0)
            if now < market_close:
                remaining_hours = (market_close - now).total_seconds() / 3600
                return max(0.01, remaining_hours / 24.0)
            return 0.01  # post-close on expiry
        return float(days_ahead)

    # ── Startup Safety Checks ──────────────────────────────────────────

    def _verify_ip_whitelist(self) -> bool:
        """Verify broker API connectivity (catches IP whitelist issues).

        Makes a simple profile() call to confirm the API key is valid and
        the current IP is whitelisted. If IP is not whitelisted, Kite
        raises InputException or PermissionError — we catch it and prevent
        the engine from starting with a stale/blocked session.

        Returns True if OK, False if blocked.
        """
        # Normal paper mode: no real broker to verify.
        if self._is_paper and not self._is_hybrid:
            return True

        # Live OR hybrid: ping the real Kite broker. In hybrid mode that's
        # the _data_broker (real Kite) — execution broker is paper and has
        # nothing to verify. Use _md_broker() which routes correctly.
        real_broker = self._md_broker()
        try:
            real_broker.get_positions()
            logger.info("IP whitelist check PASSED — broker API accessible")
            return True
        except PermissionError as e:
            logger.critical(
                "IP WHITELIST BLOCKED: %s — engine will NOT start. "
                "Add this server's IP to Kite Connect app whitelist.",
                e,
            )
            return False
        except Exception as e:
            err_msg = str(e).lower()
            if "ip" in err_msg or "whitelist" in err_msg or "permission" in err_msg:
                logger.critical(
                    "IP WHITELIST BLOCKED: %s — engine will NOT start. "
                    "Add this server's IP to Kite Connect app whitelist.",
                    e,
                )
                return False
            # Other errors (network glitch, etc.) — warn but allow startup
            logger.warning("IP whitelist check inconclusive: %s — proceeding", e)
            return True

    def _reconcile_orphan_positions(self) -> None:
        """Detect and close orphan positions from previous sessions.

        On engine startup, fetches all current Zerodha positions. Any position
        with non-zero quantity that isn't tracked by the orchestrator is an
        orphan from a crash/restart. These are closed with market orders to
        prevent unmanaged risk.

        Research post-mortem: "67% of today's loss came from orphan positions."
        """
        if self._is_paper:
            return
        if not self._auto_close_orphans:
            logger.info("Orphan position reconciliation DISABLED (auto_close_orphans=False)")
            return

        try:
            positions = self.broker.get_positions()
        except Exception as e:
            logger.error("Failed to fetch positions for orphan reconciliation: %s", e)
            return

        orphans = [p for p in positions if p.get("quantity", p.get("qty", 0)) != 0]
        if not orphans:
            logger.info("Orphan reconciliation: no open positions found — clean start")
            return

        logger.warning(
            "ORPHAN RECONCILIATION: Found %d open position(s) from previous session",
            len(orphans),
        )

        for pos in orphans:
            symbol = pos.get("tradingsymbol", pos.get("symbol", ""))
            qty = pos.get("quantity", pos.get("qty", 0))
            pnl = pos.get("pnl", 0.0)
            product = pos.get("product", "MIS")
            exchange = pos.get("exchange", "NFO")

            logger.warning(
                "ORPHAN POSITION: %s qty=%d pnl=%.2f product=%s exchange=%s",
                symbol, qty, pnl, product, exchange,
            )

            # Close orphan: sell if long, buy if short
            close_side = "SELL" if qty > 0 else "BUY"
            close_qty = abs(qty)

            try:
                result = self.broker.place_order(
                    symbol=symbol,
                    side=close_side,
                    qty=close_qty,
                    order_type="MARKET",
                    product=product,
                    tag="orphan_close",
                )
                order_id = result.get("order_id", "N/A")
                logger.info(
                    "ORPHAN CLOSED: %s %s x%d | order_id=%s | unrealized_pnl=%.2f",
                    close_side, symbol, close_qty, order_id, pnl,
                )
            except Exception as e:
                logger.error(
                    "ORPHAN CLOSE FAILED: %s %s x%d — %s (MANUAL INTERVENTION REQUIRED)",
                    close_side, symbol, close_qty, e,
                )

    # ── GTT Backup Stop-Loss ───────────────────────────────────────────

    def _place_gtt_backup_sl(
        self, symbol: str, exchange: str, entry_price: float,
        qty: int, side: str, product: str = "NRML",
    ) -> Optional[int]:
        """Place a GTT (Good Till Triggered) order as backup stop-loss.

        Research: "Consider broker-side SL orders (GTT) as backup for
        infrastructure failures." If the orchestrator crashes, the GTT
        order on Zerodha's servers will trigger and close the position.

        Note: Zerodha's GTT API does NOT support the MIS product on F&O —
        only NRML/CNC. For MIS entries we skip GTT entirely; Zerodha's own
        15:15-15:25 MIS auto-squareoff timer provides infrastructure-failure
        protection for intraday positions.

        Returns the GTT ID if successful, None otherwise.
        """
        if self._is_paper or not self._use_gtt_backup_sl:
            return None

        # GTT on F&O only supports NRML — skip for MIS (Zerodha auto-squares it)
        if product == "MIS":
            logger.debug(
                "GTT BACKUP SL skipped for %s: MIS product not supported by "
                "Zerodha GTT on F&O (broker auto-squareoff is the fallback)",
                symbol,
            )
            return None

        # Calculate SL trigger price based on premium loss percentage
        sl_price = round(entry_price * (1 - self._gtt_sl_pct), 2)
        if sl_price <= 0:
            sl_price = 0.05  # Minimum tick

        # GTT exit side is opposite of entry
        exit_side = "SELL" if side == "BUY" else "BUY"

        try:
            gtt_id = self.broker._kite.place_gtt(
                trigger_type=self.broker._kite.GTT_TYPE_SINGLE,
                tradingsymbol=symbol,
                exchange=exchange or "NFO",
                trigger_values=[sl_price],
                last_price=entry_price,
                orders=[{
                    "transaction_type": exit_side,
                    "quantity": qty,
                    "price": sl_price,
                    "order_type": "LIMIT",
                    "product": product,
                }],
            )
            gtt_trigger_id = gtt_id.get("trigger_id", gtt_id) if isinstance(gtt_id, dict) else gtt_id
            self._gtt_orders[symbol] = gtt_trigger_id
            logger.info(
                "GTT BACKUP SL placed | %s | entry=%.2f sl=%.2f (%.0f%% loss) | "
                "gtt_id=%s | qty=%d %s %s",
                symbol, entry_price, sl_price, self._gtt_sl_pct * 100,
                gtt_trigger_id, qty, exit_side, product,
            )
            return gtt_trigger_id
        except Exception as e:
            logger.warning(
                "GTT BACKUP SL failed for %s: %s — proceeding without broker-side SL",
                symbol, e,
            )
            return None

    def _cancel_gtt_backup_sl(self, symbol: str) -> None:
        """Cancel the GTT backup stop-loss when exiting a position normally.

        When the orchestrator handles the exit itself, the broker-side GTT
        is no longer needed and must be cancelled to avoid double-closing.
        """
        gtt_id = self._gtt_orders.pop(symbol, None)
        if gtt_id is None or self._is_paper:
            return

        try:
            self.broker._kite.delete_gtt(gtt_id)
            logger.info("GTT BACKUP SL cancelled | %s | gtt_id=%s", symbol, gtt_id)
        except Exception as e:
            logger.warning(
                "GTT cancel failed for %s (gtt_id=%s): %s — may trigger independently",
                symbol, gtt_id, e,
            )

    # ── Main Lifecycle ──────────────────────────────────────────────────

    async def run(self) -> dict:
        """Run the live trading session until market close or shutdown."""
        self._running = True
        self._shutdown = False
        self._loop = asyncio.get_running_loop()
        self.risk_manager.reset_daily()
        self.meta_agent.reset_daily()

        await self._emit({
            "type": "orchestrator_started",
            "timestamp": datetime.now().isoformat(),
            "agents": list(self.agents.keys()),
            "capital": self.capital,
            "lot_size": self.lot_size,
            "symbol": self.symbol,
            "is_expiry_day": self._is_expiry_day,
        })

        # ── Startup safety checks (before any trading activity) ──

        # 1. Verify IP whitelist / broker connectivity
        if not self._verify_ip_whitelist():
            self._running = False
            return {"error": "IP whitelist check failed — engine not started"}

        # 2. Close any orphan positions from previous sessions
        self._reconcile_orphan_positions()

        # Subscribe to market data
        self._subscribe_market_data()

        # Fetch option chain for strike selection
        await self._refresh_option_chain()

        # Fetch VIX if available
        await self._refresh_vix()

        # Pre-load today's historical bars so the agent has full context.
        # Live mode: always. Normal paper: skip (synthetic ticks generate
        # their own bars). Hybrid (--paper-live-data): load, because real
        # ticks will arrive at real-time cadence and we need warmup data.
        if not self._is_paper or self._is_hybrid:
            await self._load_historical_bars()

        # Spawn concurrent tasks
        tasks = [
            asyncio.create_task(self._tick_dispatcher(), name="tick_dispatcher"),
            asyncio.create_task(self._signal_processor(), name="signal_processor"),
            asyncio.create_task(self._risk_monitor(), name="risk_monitor"),
            asyncio.create_task(self._periodic_data_refresh(), name="data_refresh"),
        ]
        for name in self.agents:
            task = asyncio.create_task(
                self._run_agent(name), name=f"agent_{name}"
            )
            tasks.append(task)

        # Wait for market close or shutdown
        try:
            while not self._shutdown:
                if self._is_paper and not self._is_hybrid:
                    # Normal paper mode: end when the synthetic tick
                    # generator has emitted all 375 ticks AND the queue
                    # has drained. A bar-count threshold would be wrong
                    # because 375 one-minute ticks aggregate into only
                    # ~75 five-minute bars.
                    if self._synthetic_ticks_exhausted and self._tick_queue.empty():
                        logger.info(
                            "Paper session complete — %d bars processed "
                            "(synthetic ticks exhausted, queue drained)",
                            self._bar_count,
                        )
                        break
                else:
                    # Live OR hybrid (--paper-live-data): real wall-clock
                    # cadence, terminate on market close / square-off.
                    now = datetime.now().time()
                    if now >= dt_time(MARKET_CLOSE_HOUR, MARKET_CLOSE_MINUTE):
                        logger.info("Market closed — ending session")
                        break
                    if now >= SQUARE_OFF_TIME:
                        logger.info("Square-off time — closing all positions")
                        await self._square_off_all("eod_square_off")
                        break
                await asyncio.sleep(1)
        finally:
            self._shutdown = True
            # Cancel all tasks
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

            # Final square-off
            await self._square_off_all("session_end")
            self._running = False

        summary = self._build_summary()
        await self._emit({"type": "orchestrator_ended", "summary": summary})
        return summary

    async def shutdown(self) -> None:
        """Signal graceful shutdown."""
        logger.info("Shutdown requested")
        self._shutdown = True

    # ── Market Data ─────────────────────────────────────────────────────

    def _subscribe_market_data(self) -> None:
        """Subscribe to real-time ticks from the broker.

        Includes automatic reconnection with exponential backoff if
        the WebSocket connection drops during market hours.
        """
        from backtesting.paper_trading import PaperTradingBroker

        # Hybrid mode (--paper-live-data): execution broker is paper but
        # _data_broker is a real KiteConnectBroker. Subscribe to real ticks
        # via the data broker, NOT the synthetic generator.
        if self._is_hybrid:
            underlying = INDEX_CONFIG.get(self.symbol, {}).get(
                "underlying_symbol", f"NSE:{self.symbol} 50"
            )
            self._tick_symbols = [underlying, "INDIA VIX"]
            logger.info(
                "HYBRID MODE: paper execution + real Kite data | "
                "subscribing to live ticks: %s", self._tick_symbols,
            )
            try:
                self._md_broker().subscribe_ticks(
                    self._tick_symbols, self._on_tick_threadsafe
                )
                asyncio.ensure_future(self._tick_watchdog())
            except Exception as e:
                logger.error(
                    "HYBRID MODE: failed to subscribe real ticks: %s — "
                    "aborting (will NOT fall back to synthetic ticks in "
                    "hybrid mode)", e,
                )
                self._shutdown = True
            return

        # Paper broker doesn't generate ticks — use synthetic data
        if isinstance(self.broker, PaperTradingBroker):
            # Seed a paper-mode VIX so the synthetic tick generator has a
            # non-zero vol (live default is 0.0 fail-loud; paper needs a
            # realistic seed). 14.0 matches a calm-market regime.
            if self._live_vix <= 0.0:
                self._live_vix = 14.0
                self._last_vix_tick_time = datetime.now()
                logger.info("Paper mode: seeded VIX=%.1f", self._live_vix)
            logger.info("Paper trading mode — using synthetic tick generator")
            asyncio.ensure_future(self._generate_synthetic_ticks())
            return

        underlying = INDEX_CONFIG.get(self.symbol, {}).get(
            "underlying_symbol", f"NSE:{self.symbol} 50"
        )
        # Subscribe to the underlying (for 5-min bar aggregation) AND
        # INDIA VIX so self._live_vix is updated tick-by-tick instead
        # of via the 60s REST poll. Stale VIX during a volatility
        # spike is how 9:15 entries slip past the vix_floor=13 gate.
        self._tick_symbols = [underlying, "INDIA VIX"]
        logger.info("Subscribing to live ticks: %s", self._tick_symbols)

        try:
            self._md_broker().subscribe_ticks(self._tick_symbols, self._on_tick_threadsafe)
            # Start watchdog for reconnection
            asyncio.ensure_future(self._tick_watchdog())
        except Exception as e:
            logger.error("Failed to subscribe ticks: %s", e)
            asyncio.ensure_future(self._generate_synthetic_ticks())

    async def _load_historical_bars(self) -> None:
        """Fetch historical 5-min candles and seed the bar buffer.

        Loads PREVIOUS DAY's last 20 bars + today's bars so the agent
        has enough data for indicators (RSI, EMA, BB need 15-21 bars)
        immediately at startup — no 75-minute warmup delay.
        """
        underlying = INDEX_CONFIG.get(self.symbol, {}).get(
            "underlying_symbol", f"NSE:{self.symbol} 50"
        )
        now = datetime.now()
        market_open = now.replace(hour=MARKET_OPEN_HOUR, minute=MARKET_OPEN_MINUTE, second=0, microsecond=0)

        if now <= market_open:
            logger.info("Market not yet open — skipping historical bar fetch")
            return

        all_bars = []

        # ── Step 1: Load 30 calendar days of 5-min bars for full indicator warmup ──
        # Backtest had thousands of prior bars at every decision; live previously had
        # only 20.  Loading ~22 trading days (~1,650 bars) brings live in line with
        # backtest indicator stability (EMA/MACD/ADX all need 50-150+ bars to converge).
        # Single Kite API call (5-min interval supports up to 60 days per request).
        # Falls back to a 7-day window then to nothing on failure.
        from datetime import timedelta
        WARMUP_DAYS_PRIMARY = 30
        WARMUP_DAYS_FALLBACK = 7
        for warmup_days in (WARMUP_DAYS_PRIMARY, WARMUP_DAYS_FALLBACK):
            try:
                warmup_from = (now - timedelta(days=warmup_days)).replace(
                    hour=9, minute=15, second=0, microsecond=0,
                )
                warmup_to = market_open - timedelta(seconds=1)  # up to (not incl.) today's open
                prev_bars = self._md_broker().get_historical_data(
                    symbol=underlying,
                    from_dt=warmup_from,
                    to_dt=warmup_to,
                    interval="5minute",
                )
                if prev_bars:
                    all_bars.extend(prev_bars)
                    logger.info(
                        "Historical warmup | %d bars from %s to %s (%d-day window)",
                        len(prev_bars),
                        warmup_from.strftime("%Y-%m-%d"),
                        warmup_to.strftime("%Y-%m-%d"),
                        warmup_days,
                    )
                    break  # success — don't try the fallback window
                logger.warning("Historical warmup empty for %d-day window — trying fallback", warmup_days)
            except Exception as e:
                logger.warning("Historical warmup fetch failed for %d-day window: %s", warmup_days, e)

        # ── Step 2: Load today's bars ──
        try:
            bars = self._md_broker().get_historical_data(
                symbol=underlying,
                from_dt=market_open,
                to_dt=now,
                interval="5minute",
            )
            if bars:
                all_bars.extend(bars)
        except Exception as e:
            logger.warning("Historical data fetch failed: %s — starting fresh", e)
            return

        if not all_bars:
            logger.info("No historical bars returned — starting fresh")
            return

        # Seed bar buffer and market analyzer
        for bar in all_bars:
            self._bar_buffer.append(bar)
            self._bar_count += 1
            self.market_analyzer.add_bar(bar)

        logger.info(
            "Historical bars loaded | %d bars (%d warmup + %d today) | latest close=%.2f",
            len(all_bars),
            len(all_bars) - len(bars) if bars else len(all_bars),
            len(bars) if bars else 0,
            all_bars[-1].get("close", 0),
        )

    async def _tick_watchdog(self) -> None:
        """Monitor tick flow and reconnect if ticks stop arriving.

        If no ticks arrive for 30 seconds during market hours, attempt
        to reconnect with exponential backoff (1s --> 2s --> 4s --> 8s --> 16s --> 30s).
        """
        TICK_TIMEOUT = 30  # seconds without ticks before reconnect
        MAX_RECONNECT_DELAY = 30
        reconnect_delay = 1.0

        last_raw_tick_count = 0

        while not self._shutdown:
            await asyncio.sleep(TICK_TIMEOUT)

            # Only reconnect during market hours
            now = datetime.now().time()
            if now < dt_time(MARKET_OPEN_HOUR, MARKET_OPEN_MINUTE) or \
               now > dt_time(MARKET_CLOSE_HOUR, MARKET_CLOSE_MINUTE):
                continue

            current_raw = self._raw_tick_count
            if current_raw == last_raw_tick_count:
                # No new raw ticks — WebSocket may be dead
                logger.warning(
                    "TICK WATCHDOG: No new ticks in %ds — attempting reconnect",
                    TICK_TIMEOUT,
                )
                try:
                    # Unsubscribe and resubscribe
                    if hasattr(self, '_tick_symbols'):
                        try:
                            self._md_broker().unsubscribe_ticks(self._tick_symbols)
                        except Exception:
                            pass
                        await asyncio.sleep(reconnect_delay)
                        self._md_broker().subscribe_ticks(
                            self._tick_symbols, self._on_tick_threadsafe
                        )
                        logger.info("TICK WATCHDOG: Reconnected successfully")
                        reconnect_delay = 1.0  # Reset on success
                except Exception as e:
                    logger.error("TICK WATCHDOG: Reconnect failed: %s", e)
                    reconnect_delay = min(reconnect_delay * 2, MAX_RECONNECT_DELAY)
            else:
                reconnect_delay = 1.0  # Reset if ticks are flowing

            last_raw_tick_count = current_raw

    def _on_tick_threadsafe(self, ticks: list[dict]) -> None:
        """Called from the WebSocket thread — put ticks into thread-safe queue."""
        for tick in ticks:
            try:
                self._tick_queue.put_nowait(tick)
            except queue.Full:
                pass  # Drop tick if queue is full

    async def _generate_synthetic_ticks(self) -> None:
        """Generate realistic synthetic 1-minute ticks for a full trading day.

        375 bars (9:15 AM to 3:30 PM IST), paced at ~1 sec per bar.

        Realistic intraday patterns modelled on actual NSE behavior:
          1. Opening burst (9:15-9:30): high vol gap, wide range
          2. Initial range (9:30-10:00): trend establishment
          3. Mid-morning move (10:00-11:30): main trend of the day
          4. Lunch lull (11:30-13:00): low vol, mean reversion, tight range
          5. Afternoon push (13:00-14:30): secondary trend, often reversal
          6. Closing rush (14:30-15:30): increasing vol, trend extension/reversal

        Also:
          - VIX evolves dynamically with price moves
          - Volume profile follows U-shape (high at open/close, low at lunch)
        """
        import numpy as np

        # Apr 2026 levels — NIFTY still in crisis-bearish regime after Iran war
        base_prices = {"NIFTY": 22713.0, "BANKNIFTY": 48000.0, "FINNIFTY": 22000.0}
        base_price = base_prices.get(self.symbol, 22713.0)

        # Annualized vol from VIX
        ann_vol = self._live_vix / 100.0
        daily_vol = ann_vol / np.sqrt(252)         # ~0.88% for VIX=14
        minute_vol = daily_vol / np.sqrt(375)      # ~0.045% per minute

        logger.info(
            "Generating 375 realistic 1-min bars | %s @ %.0f | vol=%.2f%% daily",
            self.symbol, base_price, daily_vol * 100,
        )

        np.random.seed(None)
        base_ts = datetime.now().replace(hour=9, minute=15, second=0, microsecond=0)

        # ── Random day scenario ──
        # Decide the day's character: trending up, trending down, or range-bound
        day_type = np.random.choice(["trend_up", "trend_down", "range"], p=[0.35, 0.30, 0.35])
        day_drift = {"trend_up": 0.0003, "trend_down": -0.0003, "range": 0.0}[day_type]

        # Previous close gap (common in Indian markets after SGX/global cues)
        gap_pct = np.random.uniform(-0.005, 0.005)
        price = base_price * (1 + gap_pct)

        logger.info("Day type: %s | gap: %+.2f%% | drift: %+.4f", day_type, gap_pct * 100, day_drift)

        prices = [price]
        vix_path = [self._live_vix]

        for i in range(375):
            minutes_since_open = i
            hour = 9 + (15 + minutes_since_open) // 60
            minute = (15 + minutes_since_open) % 60

            # ── Intraday volatility multiplier (U-shape) ──
            if minutes_since_open < 15:
                # Opening 15 min: 2-3x normal vol
                vol_mult = 2.5 - minutes_since_open * 0.1
            elif minutes_since_open < 45:
                # 9:30-10:00: elevated
                vol_mult = 1.5
            elif minutes_since_open < 135:
                # 10:00-11:30: normal
                vol_mult = 1.0
            elif minutes_since_open < 225:
                # 11:30-13:00: lunch lull
                vol_mult = 0.6
            elif minutes_since_open < 315:
                # 13:00-14:30: picking up
                vol_mult = 1.0
            else:
                # 14:30-15:30: closing rush
                vol_mult = 1.3 + (minutes_since_open - 315) * 0.01

            # ── Drift varies by session ──
            if minutes_since_open < 30:
                drift = day_drift * 2  # opening momentum
            elif minutes_since_open < 135:
                drift = day_drift  # main trend
            elif minutes_since_open < 225:
                drift = -day_drift * 0.3  # mild reversal during lunch
            elif minutes_since_open < 315:
                drift = day_drift * 0.5  # resume but weaker
            else:
                # Last hour: sometimes reversal, sometimes extension
                drift = day_drift * np.random.choice([0.8, -0.5], p=[0.6, 0.4])

            # Mean-revert if too far from base (markets rarely move >1.5% intraday)
            deviation = (prices[-1] - base_price) / base_price
            mean_rev = 0.0
            if abs(deviation) > 0.01:
                mean_rev = -deviation * 0.03

            ret = drift + mean_rev + vol_mult * minute_vol * np.random.randn()
            new_price = prices[-1] * (1 + ret)
            prices.append(new_price)

            # ── Dynamic VIX: VIX spikes when market drops ──
            price_change_pct = ret * 100
            vix_change = -price_change_pct * 0.4 + np.random.randn() * 0.05
            new_vix = max(9.0, min(35.0, vix_path[-1] + vix_change))
            vix_path.append(new_vix)

        # ── Volume profile (U-shape: high at open/close, low at lunch) ──
        def _volume(bar_idx: int) -> int:
            if bar_idx < 15:
                return int(np.random.uniform(200000, 500000))
            elif bar_idx < 135:
                return int(np.random.uniform(80000, 200000))
            elif bar_idx < 225:
                return int(np.random.uniform(40000, 100000))
            elif bar_idx < 315:
                return int(np.random.uniform(80000, 180000))
            else:
                return int(np.random.uniform(150000, 400000))

        # ── Emit bars ──
        for i in range(375):
            if self._shutdown:
                break

            sim_ts = base_ts + timedelta(minutes=i)
            o = prices[i]
            c = prices[i + 1]
            h = max(o, c) * (1 + abs(np.random.randn()) * minute_vol * 0.3)
            l = min(o, c) * (1 - abs(np.random.randn()) * minute_vol * 0.3)

            # Update dynamic VIX
            self._live_vix = round(vix_path[i + 1], 2)

            tick = {
                "symbol": self.symbol,
                "last_price": round(c, 2),
                "open": round(o, 2),
                "high": round(h, 2),
                "low": round(l, 2),
                "volume": _volume(i),
                "timestamp": sim_ts,
            }
            self._tick_queue.put(tick)
            # Paper mode: fast-forward at ~50x speed (375 bars in ~8 seconds)
            # Live mode: real-time 1 tick/sec
            await asyncio.sleep(0.02 if self._is_paper else 1.0)

        # Final flush tick
        self._tick_queue.put({
            "symbol": self.symbol,
            "last_price": round(prices[-1], 2),
            "timestamp": base_ts + timedelta(minutes=376),
            "volume": 0,
        })
        logger.info(
            "Synthetic ticks done | %d bars | close=%.0f (%+.2f%%) | VIX=%.1f",
            375, prices[-1], (prices[-1] / base_price - 1) * 100, self._live_vix,
        )
        # Tell the main loop we're done — it will drain the queue and exit.
        self._synthetic_ticks_exhausted = True

    # ── Periodic Data Refresh ──────────────────────────────────────────

    async def _periodic_data_refresh(self) -> None:
        """Periodically refresh VIX, option chain, and other market data."""
        while not self._shutdown:
            try:
                await self._refresh_vix()
                await self._refresh_option_chain()
                await self._compute_pcr()
            except Exception as e:
                logger.error("Data refresh error: %s", e)
            await asyncio.sleep(60)  # Refresh every 60 seconds

    async def _refresh_vix(self) -> None:
        """Fetch current India VIX (REST fallback only).

        Primary VIX updates arrive on the WebSocket tick stream via
        _tick_dispatcher's INDIA VIX branch. This REST path is a
        defensive fallback for when the tick feed is silent (late
        subscribe, market pre-open, WebSocket drop that hasn't
        reconnected yet). We only overwrite self._live_vix here if
        the tick stream is stale (>90s since last VIX tick) or has
        never delivered one.
        """
        # If the WebSocket tick stream is delivering VIX ticks, leave it alone.
        now = datetime.now()
        if self._last_vix_tick_time is not None:
            age = (now - self._last_vix_tick_time).total_seconds()
            if age < 90:
                return

        try:
            fetched: Optional[float] = None
            md = self._md_broker()
            if hasattr(md, "get_vix"):
                fetched = md.get_vix()
            elif hasattr(md, "get_quote"):
                quote = md.get_quote("NSE:INDIA VIX")
                if quote and "last_price" in quote:
                    fetched = quote["last_price"]

            if fetched and fetched > 0:
                self._live_vix = float(fetched)
                self.market_analyzer.add_vix(self._live_vix)
                logger.info("VIX refresh (REST fallback): %.2f", self._live_vix)
            else:
                logger.warning(
                    "VIX REST fallback returned no data; live_vix=%.2f "
                    "(tick stream last=%s)",
                    self._live_vix, self._last_vix_tick_time,
                )
        except Exception as e:
            logger.warning(
                "VIX REST fallback failed (live_vix=%.2f): %s",
                self._live_vix, e,
            )

    async def _compute_pcr(self) -> None:
        """Compute Put-Call Ratio from the option chain."""
        if not self._option_chain:
            return
        total_put_oi = sum(
            self._option_chain[s].get("PE", {}).get("oi", 0)
            for s in self._option_chain
        )
        total_call_oi = sum(
            self._option_chain[s].get("CE", {}).get("oi", 0)
            for s in self._option_chain
        )
        if total_call_oi > 0:
            self._live_pcr = total_put_oi / total_call_oi
            logger.debug("PCR updated: %.3f", self._live_pcr)

    # ── Tick Processing ─────────────────────────────────────────────────

    async def _tick_dispatcher(self) -> None:
        """Read ticks from queue, aggregate into bars, fan out to agents."""
        bar_interval = timedelta(minutes=5)  # V14 uses 5-min bars
        current_bar_start: Optional[datetime] = None
        _last_midbar_eval: float = 0.0  # monotonic time of last mid-bar eval
        MIDBAR_INTERVAL = 15.0  # evaluate every 15s within the bar

        _logged_first_tick = False
        while not self._shutdown:
            try:
                tick = self._tick_queue.get(timeout=0.1)
            except queue.Empty:
                await asyncio.sleep(0.05)
                continue

            self._raw_tick_count += 1
            price = tick.get("last_price", 0)
            tick_sym = tick.get("symbol", "")

            # Cache tick LTP for ALL instruments (for real-time position PnL)
            if tick_sym and price > 0:
                self._tick_ltp[tick_sym] = price

            # ── INDIA VIX tick: update the live VIX cache and continue ──
            # VIX is published on NSE as "INDIA VIX". We subscribe to it
            # in _subscribe_market_data() specifically so self._live_vix
            # is updated tick-by-tick (not on a 60s REST cadence). This
            # is what the V15 vix_floor gate and V17 btst-favorable band
            # both read.
            if tick_sym == "INDIA VIX" and price > 0:
                prev_vix = self._live_vix
                self._live_vix = float(price)
                self._last_vix_tick_time = datetime.now()
                try:
                    self.market_analyzer.add_vix(self._live_vix)
                except Exception:
                    pass
                if abs(self._live_vix - prev_vix) >= 0.5 or prev_vix == 0.0:
                    logger.info(
                        "VIX tick | %.2f → %.2f", prev_vix, self._live_vix,
                    )
                continue  # VIX ticks don't feed the 5-min bar aggregator

            # Only build bars from the main underlying (NIFTY 50), skip option ticks
            tick_token = tick.get("instrument_token", 0)
            is_underlying = (tick_sym == "NIFTY 50" or tick_sym == self.symbol
                             or tick_token == 256265)  # NIFTY 50 token
            if not is_underlying:
                continue  # Skip option ticks for bar aggregation

            if not _logged_first_tick:
                logger.info("Tick dispatcher: first tick received (price=%.2f)",
                            tick.get("last_price", 0))
                _logged_first_tick = True
            ts = tick.get("exchange_timestamp") or tick.get("timestamp") or datetime.now()
            if isinstance(ts, (int, float)):
                ts = datetime.fromtimestamp(ts / 1000)

            # Aggregate into 5-minute bars (V14 config uses 5-min bar indices)
            bar_start = ts.replace(minute=(ts.minute // 5) * 5, second=0, microsecond=0)
            if current_bar_start is None:
                current_bar_start = bar_start
                logger.info("Tick flow started: bar_start=%s price=%.2f", bar_start, price)
                self._current_bar = {
                    "timestamp": bar_start,
                    "open": tick.get("open", price),
                    "high": tick.get("high", price),
                    "low": tick.get("low", price),
                    "close": price,
                    "volume": tick.get("volume", 0),
                }
            elif bar_start != current_bar_start and self._current_bar:
                # Bar completed — run analysis and dispatch to agents
                logger.debug("Bar completed: %s close=%.2f", current_bar_start, self._current_bar["close"])
                completed_bar = self._current_bar.copy()
                self._bar_buffer.append(completed_bar)
                self._bar_count += 1

                # Feed bar to market analyzer
                self.market_analyzer.add_bar(completed_bar)

                # Run comprehensive market analysis
                analysis = self._run_market_analysis(completed_bar["close"])

                for name, q in self._agent_queues.items():
                    try:
                        q.put_nowait((completed_bar, analysis))
                    except asyncio.QueueFull:
                        pass

                await self._emit({
                    "type": "bar",
                    "bar": completed_bar,
                    "bar_count": self._bar_count,
                })

                # Emit analysis to dashboard
                if analysis:
                    await self._emit_analysis(analysis)

                current_bar_start = bar_start
                self._current_bar = {
                    "timestamp": bar_start,
                    "open": price, "high": price,
                    "low": price, "close": price,
                    "volume": tick.get("volume", 0),
                }
            else:
                # Update current bar
                self._current_bar["high"] = max(self._current_bar["high"], price)
                self._current_bar["low"] = min(self._current_bar["low"], price)
                self._current_bar["close"] = price
                self._current_bar["volume"] += tick.get("volume", 0)

                # Mid-bar evaluation: dispatch snapshot every 15s for faster entries
                now_mono = time.monotonic()
                if not self._is_paper and (now_mono - _last_midbar_eval) >= MIDBAR_INTERVAL:
                    _last_midbar_eval = now_mono
                    snapshot = self._current_bar.copy()
                    snapshot["_is_midbar"] = True
                    analysis = self._run_market_analysis(price)
                    for name, q in self._agent_queues.items():
                        try:
                            q.put_nowait((snapshot, analysis))
                        except asyncio.QueueFull:
                            pass

            # Tick-level exit check — stops trigger instantly, not on bar close
            if self._open_trades and not self._is_paper:
                await self._tick_exit_check(price)

            # Update option premiums for paper trading P&L
            if self._is_paper:
                self._update_paper_option_prices(price)

            # Emit tick to dashboard
            await self._emit({
                "type": "live_tick",
                "price": price,
                "timestamp": str(ts),
                "symbol": tick.get("symbol", self.symbol),
            })

    async def _tick_exit_check(self, spot: float) -> None:
        """Check exit conditions on every tick for faster stop-loss execution.

        Delegates to each agent's exit logic using the current spot price
        without waiting for bar completion. This reduces stop-loss lag
        from up to 60s (bar close) to near-instant (tick-level).
        """
        for name, agent in self.agents.items():
            if not hasattr(agent, '_check_exits') or not hasattr(agent, '_open_positions'):
                continue
            if not agent._open_positions:
                continue

            # Use latest analysis if available
            analysis = self._latest_analysis

            try:
                exit_signal = agent._check_exits(
                    spot, agent._bar_idx if hasattr(agent, '_bar_idx') else self._bar_count,
                    self._live_vix, analysis,
                )
                if exit_signal is not None:
                    logger.info(
                        "TICK-LEVEL EXIT | %s | spot=%.2f | reason=%s",
                        name, spot, exit_signal.reasoning,
                    )
                    await self._signal_queue.put(exit_signal)
            except Exception as e:
                logger.debug("Tick exit check error for %s: %s", name, e)

    def _update_paper_option_prices(self, spot: float) -> None:
        """Update option premiums using Black-Scholes as the underlying moves.

        This gives realistic P&L for paper trading:
        - OTM options decay rapidly (theta)
        - Delta exposure creates directional P&L
        - IV changes affect all options via vega
        """
        if not hasattr(self.broker, 'positions') or not hasattr(self.broker, 'update_price'):
            return

        from backtesting.option_pricer import price_option

        dte = self._compute_dte()

        # Reduce DTE slightly as the day progresses (intraday theta)
        bars_done = self._bar_count
        intraday_decay = bars_done / 375.0  # fraction of day elapsed
        dte_adjusted = max(0.05, dte - intraday_decay)

        for sym, pos in list(self.broker.positions.items()):
            is_ce = sym.endswith("CE")
            is_pe = sym.endswith("PE")
            if not (is_ce or is_pe):
                continue

            try:
                strike_str = sym
                for prefix in ("NIFTY", "BANKNIFTY", "FINNIFTY"):
                    strike_str = strike_str.replace(prefix, "")
                strike_str = strike_str.replace("CE", "").replace("PE", "")
                strike = float(strike_str)
            except ValueError:
                continue

            opt_type = "CE" if is_ce else "PE"
            bs = price_option(
                spot=spot,
                strike=strike,
                dte_days=dte_adjusted,
                vix=self._live_vix,
                option_type=opt_type,
            )
            self.broker.update_price(sym, bs["premium"])

    def _run_market_analysis(self, spot_price: float) -> Optional[MarketAnalysis]:
        """Run the comprehensive market analyzer on current data."""
        try:
            analysis = self.market_analyzer.analyze(
                spot_price=spot_price,
                vix=self._live_vix,
                pcr=self._live_pcr,
                option_chain=self._option_chain or None,
                fii_net=self._fii_net,
                dii_net=self._dii_net,
                is_expiry_day=self._is_expiry_day,
            )
            self._latest_analysis = analysis
            logger.info(
                "Market Analysis | bias=%s action=%s score=%.3f conf=%.2f VIX=%.1f PCR=%.2f",
                analysis.market_bias.value, analysis.recommended_action.value,
                analysis.overall_score, analysis.confidence,
                analysis.vix, analysis.pcr,
            )
            return analysis
        except Exception as e:
            logger.error("Market analysis failed: %s", e)
            return None

    async def _emit_analysis(self, analysis: MarketAnalysis) -> None:
        """Emit market analysis to dashboard."""
        top_indicators = sorted(
            analysis.indicators,
            key=lambda x: abs(x.score * x.confidence),
            reverse=True
        )[:5]

        await self._emit({
            "type": "market_analysis",
            "timestamp": analysis.timestamp.isoformat(),
            "spot_price": analysis.spot_price,
            "vix": analysis.vix,
            "vix_regime": analysis.vix_regime.value,
            "pcr": analysis.pcr,
            "max_pain": analysis.max_pain,
            "max_pain_distance": analysis.max_pain_distance,
            "oi_support": analysis.oi_support,
            "oi_resistance": analysis.oi_resistance,
            "iv_percentile": analysis.iv_percentile,
            "iv_skew": analysis.iv_skew,
            "market_bias": analysis.market_bias.value,
            "recommended_action": analysis.recommended_action.value,
            "overall_score": round(analysis.overall_score, 4),
            "confidence": round(analysis.confidence, 3),
            "is_expiry_day": analysis.is_expiry_day,
            "timing_ok": analysis.timing_ok,
            "reasoning": analysis.reasoning,
            "top_indicators": [
                {"name": i.name, "score": round(i.score, 3), "reasoning": i.reasoning}
                for i in top_indicators
            ],
        })

    # ── Strategy Agents ─────────────────────────────────────────────────

    async def _run_agent(self, name: str) -> None:
        """Run a single strategy agent — consume bars + analysis, emit signals."""
        agent = self.agents[name]
        queue = self._agent_queues[name]

        # Replay historical bars so agent builds internal state (ORB, S/R, EMA)
        # bar_idx counts only TODAY's bars (not warmup from previous day)
        # so avoid_window and late_entry checks align with market time
        bar_idx = 0
        today_str = datetime.now().strftime("%Y-%m-%d")
        if self._bar_buffer:
            n_bars = len(self._bar_buffer)
            logger.info("Agent %s: replaying %d historical bars", name, n_bars)
            # Temporarily suppress verbose logging during replay
            agent_logger = logging.getLogger(agent.__class__.__module__)
            prev_agent_level = agent_logger.level
            prev_orch_level = logger.level
            agent_logger.setLevel(logging.WARNING)
            logger.setLevel(logging.WARNING)

            # Count warmup bars (yesterday) vs today's bars
            warmup_count = 0
            for hist_bar in self._bar_buffer:
                bar_ts = str(hist_bar.get("time", hist_bar.get("timestamp", "")))
                is_today = today_str in bar_ts
                if not is_today:
                    warmup_count += 1
                spot = hist_bar.get("close", 0)
                analysis = self._run_market_analysis(spot)
                try:
                    agent.generate_signal(
                        hist_bar, bar_idx,
                        option_chain=self._option_chain,
                        market_analysis=analysis,
                    )
                except Exception:
                    pass
                # Only increment bar_idx for today's bars
                if is_today:
                    bar_idx += 1

            logger.setLevel(prev_orch_level)
            agent_logger.setLevel(prev_agent_level)

            # CRITICAL: Clear phantom positions created during replay.
            # Replay builds indicator state (bar_history, EMA, RSI) but the
            # signals generated are discarded (never sent to broker).  However,
            # generate_signal() has the side-effect of appending to
            # agent._open_positions — these "phantom" entries block real trades
            # via the same-direction check.  We must clear them so the agent
            # starts live with no phantom positions.
            if hasattr(agent, "_open_positions"):
                n_phantom = len(agent._open_positions)
                if n_phantom:
                    logger.warning("Agent %s: clearing %d phantom positions from replay", name, n_phantom)
                agent._open_positions.clear()
            # Also reset trades_today so replay trades don't count
            if hasattr(agent, "_trades_today"):
                agent._trades_today = 0
            if hasattr(agent, "_last_exit_bar"):
                agent._last_exit_bar = -999

            logger.info("Agent %s: replay done | %d warmup + %d today bars | starting live at bar_idx=%d",
                        name, warmup_count, bar_idx, bar_idx)

        logger.info("Agent %s started", name)

        while not self._shutdown:
            try:
                item = await asyncio.wait_for(queue.get(), timeout=5.0)
            except asyncio.TimeoutError:
                continue

            # Unpack bar and analysis (backward compatible)
            if isinstance(item, tuple):
                bar, analysis = item
            else:
                bar, analysis = item, None

            # Check wind-down mode (skip in paper mode — uses simulated time)
            if not self._is_paper:
                now = datetime.now().time()
                if now >= WIND_DOWN_TIME:
                    continue  # No new signals in wind-down

            try:
                signal = agent.generate_signal(
                    bar, bar_idx,
                    option_chain=self._option_chain,
                    market_analysis=analysis,
                )
                if signal is not None:
                    await self._signal_queue.put(signal)
                    await self._emit({
                        "type": "agent_signal",
                        "agent": name,
                        "action": signal.action,
                        "confidence": signal.confidence,
                        "reasoning": signal.reasoning,
                        "underlying_price": signal.underlying_price,
                        "max_loss": signal.max_loss,
                        "timestamp": datetime.now().isoformat(),
                    })
            except Exception as e:
                logger.error("Agent %s error: %s", name, e)

            if not bar.get("_is_midbar"):
                bar_idx += 1

                # ── Equity compounding: update agent's lot sizing every 15 bars (~75 min) ──
                if bar_idx % 15 == 0 and hasattr(agent, "update_equity"):
                    try:
                        positions = self.broker.get_positions()
                        realized = sum(
                            float(p.get("pnl", p.get("realised", 0)) or 0)
                            for p in positions
                        )
                        unrealized = sum(
                            float(p.get("unrealised", p.get("unrealized", 0)) or 0)
                            for p in positions
                        )
                        current_equity = self._capital + realized + unrealized
                        agent.update_equity(current_equity)
                    except Exception as e:
                        logger.debug("Equity update for %s failed: %s", name, e)

    # ── Signal Processing ───────────────────────────────────────────────

    async def _signal_processor(self) -> None:
        """Process signals from all agents through the meta-agent."""
        signal_buffer: list[TradeSignal] = []
        buffer_window = 1.0  # 1s buffer (was 5s — too slow)

        while not self._shutdown:
            try:
                signal = await asyncio.wait_for(
                    self._signal_queue.get(), timeout=buffer_window
                )
                signal_buffer.append(signal)
            except asyncio.TimeoutError:
                pass

            if not signal_buffer:
                continue

            # Evaluate buffered signals
            if self.risk_manager.kill_switch_active:
                signal_buffer.clear()
                continue

            # Feed live VIX to meta-agent for loss-reduction filters
            self.meta_agent.set_vix(self._live_vix)

            approved = self.meta_agent.evaluate(signal_buffer)

            # Clear pending entries for signals that were NOT approved
            # This prevents agent/orchestrator state mismatch
            for rejected_signal in signal_buffer:
                if rejected_signal is not approved and rejected_signal.action != "CLOSE":
                    for agent in self.agents.values():
                        if agent.name == rejected_signal.strategy:
                            if hasattr(agent, "_pending_entry"):
                                agent._pending_entry = None

            signal_buffer.clear()

            if approved is not None:
                await self._execute_trade(approved)

    async def _execute_trade(self, signal: TradeSignal) -> None:
        """Execute an approved trade signal via the broker.

        Uses OrderManager when available (autonomous mode) for:
          - Order status polling and confirmation
          - Automatic retry on transient failures
          - Persistent order logging
        Falls back to direct broker calls otherwise.
        """
        if signal.action == "CLOSE":
            await self._close_open_trades(signal)
            return

        if not signal.legs:
            logger.warning("Signal has no legs — skipping")
            return

        # ── Pre-trade risk gate (NEW: wire check_position_limits) ──
        for leg in signal.legs:
            order_check = {
                "symbol": leg.symbol,
                "qty": leg.qty,
                "price": leg.price if leg.price > 0 else 100.0,  # Estimate if unknown
                "side": leg.side,
            }
            if not self.risk_manager.check_position_limits(order_check):
                logger.warning(
                    "RISK GATE: Order rejected by position limits | %s %s x%d",
                    leg.side, leg.symbol, leg.qty,
                )
                await self._emit({
                    "type": "risk_rejection",
                    "strategy": signal.strategy,
                    "reason": "position_limits_exceeded",
                    "symbol": leg.symbol,
                    "timestamp": datetime.now().isoformat(),
                })
                return

        logger.info(
            "Executing trade | strategy=%s action=%s legs=%d max_loss=%.0f",
            signal.strategy, signal.action, len(signal.legs), signal.max_loss,
        )

        execution = TradeExecution(signal=signal, timestamp=datetime.now())

        # Use Black-Scholes for realistic premium pricing
        from backtesting.option_pricer import price_option

        # Calculate DTE from expiry day
        dte_days = self._compute_dte()

        # Determine product type (MIS for intraday, NRML for BTST)
        product = signal.metadata.get("product", "MIS") if hasattr(signal, 'metadata') else "MIS"

        for i, leg in enumerate(signal.legs):
            # Get actual market price (LTP) — prefer over BS model
            order_price = leg.price
            if order_price <= 0:
                # Try real LTP first
                try:
                    ltp_data = self._md_broker().get_ltp([leg.symbol])
                    ltp = ltp_data.get(leg.symbol, 0)
                    if ltp > 0:
                        order_price = ltp
                        logger.info("LTP price | %s --> Rs%.2f", leg.symbol, order_price)
                except Exception:
                    pass

                # Fallback to Black-Scholes if LTP unavailable
                if order_price <= 0:
                    opt_type = leg.option_type or ("CE" if "CE" in leg.symbol else "PE")
                    bs = price_option(
                        spot=signal.underlying_price,
                        strike=leg.strike,
                        dte_days=dte_days,
                        vix=self._live_vix,
                        option_type=opt_type,
                    )
                    order_price = bs["premium"]
                    logger.info(
                        "BS price (fallback) | %s strike=%.0f dte=%.1f VIX=%.1f --> Rs%.2f",
                        opt_type, leg.strike, dte_days, self._live_vix, order_price,
                    )

            try:
                # Update paper broker's price map so it can fill the order
                if hasattr(self.broker, 'update_price'):
                    self.broker.update_price(leg.symbol, order_price)

                # Use OrderManager (with polling + retry) when available
                # Auto-slice large orders via TWAP to avoid HFT front-running
                if self._order_manager:
                    order_result = await self._order_manager.place_twap(
                        symbol=leg.symbol,
                        side=leg.side,
                        qty=leg.qty,
                        order_type=leg.order_type,
                        price=order_price,
                        product=product,
                        tag=signal.strategy[:20],
                    )
                    order_id = order_result.order_id
                    fill_price = order_result.fill_price or order_price
                    if not order_result.is_filled:
                        raise RuntimeError(
                            f"Order not filled: {order_result.status} — {order_result.error}"
                        )
                else:
                    # Direct broker call (legacy fallback)
                    await self.rate_limiter.acquire()
                    result = self.broker.place_order(
                        symbol=leg.symbol,
                        side=leg.side,
                        qty=leg.qty,
                        order_type=leg.order_type,
                        price=order_price,
                        product=product,
                        tag=signal.strategy[:20],
                    )
                    order_id = result.get("order_id", "")
                    fill_price = result.get("fill_price", order_price)

                execution.order_ids.append(order_id)
                execution.fill_prices.append(fill_price)

                # Track in position tracker
                if self._position_tracker:
                    self._position_tracker.add_position(
                        symbol=leg.symbol,
                        side=leg.side,
                        qty=leg.qty,
                        entry_price=fill_price,
                        order_id=order_id,
                        strategy=signal.strategy,
                        entry_type=signal.metadata.get("entry_type", "") if hasattr(signal, 'metadata') else "",
                        product=product,
                        metadata=signal.metadata if hasattr(signal, 'metadata') else {},
                    )

                # Place GTT backup stop-loss for BUY legs (entry orders)
                # Persist the returned GTT id onto the tracked position so
                # run_autonomous.exit_btst_positions() can cancel it next
                # day before closing the leg (otherwise the GTT stays live
                # and can trigger a duplicate exit order when the agent's
                # own exit fills, or persist across sessions as an orphan).
                if leg.side == "BUY" and signal.action != "CLOSE":
                    gtt_trigger_id = self._place_gtt_backup_sl(
                        symbol=leg.symbol,
                        exchange="NFO",
                        entry_price=fill_price,
                        qty=leg.qty,
                        side=leg.side,
                        product=product,
                    )
                    if gtt_trigger_id is not None and self._position_tracker:
                        for _tracked in self._position_tracker.open_positions:
                            if _tracked.get("symbol") == leg.symbol:
                                _meta = _tracked.setdefault("metadata", {})
                                _meta["gtt_id"] = gtt_trigger_id
                                break
                        # Re-persist state so the gtt_id is on disk.
                        self._position_tracker.save_state()

                await self._emit({
                    "type": "live_order",
                    "order_id": order_id,
                    "symbol": leg.symbol,
                    "side": leg.side,
                    "qty": leg.qty,
                    "fill_price": fill_price,
                    "strategy": signal.strategy,
                    "status": "FILLED",
                    "timestamp": datetime.now().isoformat(),
                })

            except Exception as e:
                logger.error("Order failed for %s: %s", leg.symbol, e)
                execution.status = "PARTIAL" if execution.order_ids else "FAILED"
                execution.error = str(e)

        if not execution.error:
            execution.status = "EXECUTED"
            for agent in self.agents.values():
                if agent.name == signal.strategy:
                    agent.set_position(True)
                    # Confirm execution so agent tracks position internally
                    # (only for entry signals, not CLOSE)
                    if signal.action != "CLOSE" and hasattr(agent, "confirm_execution"):
                        agent.confirm_execution(signal)

        self.meta_agent.record_trade(signal)
        if execution.status != "FAILED":
            self._open_trades.append(execution)
        else:
            logger.warning("Order FAILED for %s — not tracking as open position", signal.strategy)
            # Roll back agent's internal position so it doesn't keep trying to
            # exit a phantom position every tick.
            agent = self.agents.get(signal.strategy)
            if agent and hasattr(agent, "rollback_position"):
                agent.rollback_position(signal)

        await self._emit({
            "type": "trade_executed",
            "strategy": signal.strategy,
            "action": signal.action,
            "status": execution.status,
            "order_count": len(execution.order_ids),
            "max_loss": signal.max_loss,
            "estimated_credit": signal.estimated_credit,
            "timestamp": datetime.now().isoformat(),
        })

        logger.info(
            "Trade executed | strategy=%s status=%s orders=%d",
            signal.strategy, execution.status, len(execution.order_ids),
        )

    async def _close_open_trades(self, signal: TradeSignal) -> None:
        """Close open positions for a strategy that emitted a CLOSE signal.

        Uses the original trade legs to reverse exact quantities (not the net
        broker position, which may be shared across strategies). Computes
        realized P&L via pre/post close price snapshots and emits trade_closed.
        """
        try:
            # Find this strategy's open trades and collect their legs
            strategy_trades: list[TradeExecution] = []
            legs_to_close: list[OrderLeg] = []
            for trade in self._open_trades:
                if trade.signal.strategy == signal.strategy and trade.status == "EXECUTED":
                    strategy_trades.append(trade)
                    legs_to_close.extend(trade.signal.legs)

            if not legs_to_close:
                logger.warning("No open legs found for strategy %s", signal.strategy)
                for agent in self.agents.values():
                    if agent.name == signal.strategy:
                        agent.set_position(False)
                return

            # Snapshot unrealized P&L for this strategy's legs before closing
            pre_close_pnl = 0.0
            positions = self.broker.get_positions()
            pos_map = {p["symbol"]: p for p in positions}
            for leg in legs_to_close:
                pos = pos_map.get(leg.symbol)
                if not pos:
                    continue
                current = pos.get("last_price", pos.get("average_price", 0))
                entry = pos.get("average_price", 0)
                if leg.side == "BUY":
                    pre_close_pnl += (current - entry) * leg.qty
                else:
                    pre_close_pnl += (entry - current) * leg.qty

            # Close by reversing each original leg (exact qty, opposite side)
            closed_count = 0
            for leg in legs_to_close:
                close_side = "SELL" if leg.side == "BUY" else "BUY"
                try:
                    if self._order_manager:
                        result = await self._order_manager.place_and_confirm(
                            symbol=leg.symbol,
                            side=close_side,
                            qty=leg.qty,
                            order_type="MARKET",
                            product="MIS",
                            tag=f"close_{signal.strategy[:14]}",
                        )
                        if result.is_filled:
                            closed_count += 1
                    else:
                        await self.rate_limiter.acquire()
                        self.broker.place_order(
                            symbol=leg.symbol,
                            side=close_side,
                            qty=leg.qty,
                            order_type="MARKET",
                            product="MIS",
                            tag=f"close_{signal.strategy[:14]}",
                        )
                        closed_count += 1
                except Exception as e:
                    logger.error("Failed to close leg %s: %s", leg.symbol, e)

            # Cancel GTT backup stop-losses for closed positions
            for leg in legs_to_close:
                self._cancel_gtt_backup_sl(leg.symbol)

            # Calculate realized P&L from broker trade_log (recent CLOSE entries)
            realized_pnl = 0.0
            remaining_symbols = {leg.symbol for leg in legs_to_close}
            if hasattr(self.broker, 'trade_log'):
                for entry in reversed(self.broker.trade_log):
                    if entry.get("action") == "CLOSE" and entry.get("symbol") in remaining_symbols:
                        realized_pnl += entry.get("realized_pnl", 0)
                        remaining_symbols.discard(entry["symbol"])
                    if not remaining_symbols:
                        break

            # If trade_log didn't capture it (partial close), use the pre-close snapshot
            if realized_pnl == 0.0 and pre_close_pnl != 0.0:
                realized_pnl = pre_close_pnl

            # Transaction costs — proper Zerodha 2026 model per leg
            costs = self._calculate_trade_costs(legs_to_close, signal.underlying_price)

            # Deduct costs from realized P&L (costs reduce profit)
            realized_pnl -= costs

            # Update risk manager with net realized P&L (after costs)
            self.risk_manager.add_realised_pnl(realized_pnl)

            # Track in position tracker
            if self._position_tracker:
                exit_reason = ""
                if hasattr(signal, 'metadata') and signal.metadata:
                    exit_reason = signal.metadata.get("exit_reason", "")
                for leg in legs_to_close:
                    self._position_tracker.close_position(
                        symbol=leg.symbol,
                        exit_price=signal.underlying_price,
                        realized_pnl=realized_pnl / max(1, len(legs_to_close)),
                        exit_reason=exit_reason,
                    )

            # Move trades from open to closed
            for trade in strategy_trades:
                trade.status = "CLOSED"
                self._closed_trades.append(trade)
            self._open_trades = [
                t for t in self._open_trades if t not in strategy_trades
            ]

            # Mark agent as no position
            for agent in self.agents.values():
                if agent.name == signal.strategy:
                    agent.set_position(False)

            # Emit trade_closed event to dashboard
            await self._emit({
                "type": "trade_closed",
                "strategy": signal.strategy,
                "pnl": round(realized_pnl, 2),
                "costs": round(costs, 2),
                "legs_closed": closed_count,
                "timestamp": datetime.now().isoformat(),
            })

            logger.info(
                "Trade closed | strategy=%s pnl=%.2f costs=%.2f legs=%d",
                signal.strategy, realized_pnl, costs, closed_count,
            )

        except Exception as e:
            logger.error("Close positions failed: %s", e)

    def _calculate_trade_costs(self, legs: list[OrderLeg], spot: float) -> float:
        """Calculate total Zerodha transaction costs for closing legs.

        Zerodha 2026 cost structure per leg:
          - Brokerage: Rs20 per executed order (flat, both sides)
          - STT: 0.15% on sell-side premium (options_sell rate, increased Apr 2026)
          - NSE transaction charge: 0.03553% on turnover
          - SEBI turnover fee: Rs10 per crore
          - Stamp duty: 0.003% on buy-side turnover
          - GST: 18% on (brokerage + exchange charges + SEBI fee)

        Each close leg has an entry order + exit order = 2 brokerage charges.
        """
        brokerage_per_order = 20.0
        total_costs = 0.0

        for leg in legs:
            # Estimate premium from broker position or use ~2% of spot as fallback
            premium_estimate = 0.0
            if hasattr(self.broker, 'positions'):
                pos = self.broker.positions.get(leg.symbol)
                if pos:
                    premium_estimate = abs(pos.get("last_price", 0) or pos.get("average_price", 0))

            if premium_estimate <= 0:
                # Fallback: OTM option premium ≈ 0.5-2% of spot for 1-2 SD OTM
                premium_estimate = spot * 0.01

            turnover = premium_estimate * leg.qty

            # Entry + exit = 2 brokerage charges per leg
            brokerage = brokerage_per_order * 2

            # STT on sell side only (close of BUY = SELL, close of SELL = BUY)
            # Original entry side determines the close side
            close_side = "SELL" if leg.side == "BUY" else "BUY"
            stt = 0.0
            # STT on sell side of the close order
            if close_side == "SELL":
                stt = turnover * STT_RATES.get("options_sell", 0.0015)
            # Also STT on sell side of the original entry
            if leg.side == "SELL":
                stt += turnover * STT_RATES.get("options_sell", 0.0015)

            exchange_charge = turnover * NSE_TRANSACTION_CHARGE
            sebi_fee = turnover * SEBI_TURNOVER_FEE
            stamp = turnover * STAMP_DUTY_BUY if close_side == "BUY" else 0.0
            # Also stamp on entry buy side
            if leg.side == "BUY":
                stamp += turnover * STAMP_DUTY_BUY

            gst = (brokerage + exchange_charge + sebi_fee) * GST_RATE

            leg_cost = brokerage + stt + exchange_charge + sebi_fee + stamp + gst
            total_costs += leg_cost

        return total_costs

    async def _square_off_all(self, reason: str) -> None:
        """Square off all open MIS positions (BTST/NRML positions are kept).

        BTST positions are tracked by the position_tracker and will be
        exited the next trading morning by run_autonomous.py.

        V17 DYNAMIC PRODUCT: positions entered directly on NRML (because
        the V17 btst-favorable layer said so) sit in _open_positions with
        product="NRML". Before the skip, we MUST move them into the
        tracker's _btst_positions list via mark_btst(), which persists
        btst_positions.json so run_autonomous.exit_btst_positions()
        finds and closes them tomorrow morning. Without this step the
        positions get orphaned on the broker indefinitely.
        """
        logger.warning("Square-off all | reason=%s", reason)
        try:
            # ── Promote V17 NRML positions into the BTST carry pipeline ──
            # Any open position whose product is already NRML is a V17
            # overnight carry. Call mark_btst() to move it into
            # _btst_positions and persist btst_positions.json. This is
            # idempotent: if the position is already in _btst_positions
            # (legacy 15:20 path) this does nothing new.
            if self._position_tracker:
                nrml_carries = [
                    p for p in self._position_tracker.open_positions
                    if p.get("product") == "NRML"
                ]
                if nrml_carries:
                    logger.info(
                        "V17 BTST CARRY | promoting %d NRML position(s) "
                        "into BTST pipeline before square-off",
                        len(nrml_carries),
                    )
                    for p in nrml_carries:
                        logger.info(
                            "  → %s %s qty=%d entry=%.2f entry_type=%s",
                            p.get("side", "?"), p.get("symbol", "?"),
                            p.get("qty", 0), p.get("entry_price", 0.0),
                            p.get("entry_type", "?"),
                        )
                    self._position_tracker.mark_btst(nrml_carries)

            positions = self.broker.get_positions()

            # Identify BTST symbols to skip
            btst_symbols = set()
            if self._position_tracker:
                btst_symbols = {p["symbol"] for p in self._position_tracker.btst_positions}

            closed_count = 0
            skipped_btst = 0

            for pos in positions:
                qty = pos.get("qty", pos.get("quantity", 0))
                if qty == 0:
                    continue

                symbol = pos.get("symbol", "")
                product = pos.get("product", "MIS")

                # Skip BTST/NRML positions — they're held overnight
                if symbol in btst_symbols or product == "NRML":
                    skipped_btst += 1
                    logger.info(
                        "Square-off SKIP (BTST) | %s qty=%d product=%s",
                        symbol, qty, product,
                    )
                    continue

                close_side = "SELL" if qty > 0 else "BUY"

                if self._order_manager:
                    result = await self._order_manager.place_and_confirm(
                        symbol=symbol,
                        side=close_side,
                        qty=abs(qty),
                        order_type="MARKET",
                        product="MIS",
                        tag="square_off",
                    )
                    if result.is_filled:
                        closed_count += 1
                else:
                    await self.rate_limiter.acquire()
                    self.broker.place_order(
                        symbol=symbol,
                        side=close_side,
                        qty=abs(qty),
                        order_type="MARKET",
                        product="MIS",
                        tag="square_off",
                    )
                    closed_count += 1

            for agent in self.agents.values():
                agent.set_position(False)

            await self._emit({
                "type": "square_off",
                "reason": reason,
                "positions_closed": closed_count,
                "btst_kept": skipped_btst,
                "timestamp": datetime.now().isoformat(),
            })

            logger.info(
                "Square-off complete | closed=%d btst_kept=%d",
                closed_count, skipped_btst,
            )
        except Exception as e:
            logger.error("Square-off failed: %s", e)

    # ── Risk Monitor ────────────────────────────────────────────────────

    async def _risk_monitor(self) -> None:
        """Continuously monitor positions and risk."""
        while not self._shutdown:
            try:
                positions = self.broker.get_positions()
                risk_positions = [
                    Position(
                        symbol=p["symbol"],
                        qty=abs(p.get("qty", p.get("quantity", 0))),
                        side="BUY" if p.get("qty", p.get("quantity", 0)) > 0 else "SELL",
                        entry_price=p.get("average_price", 0),
                        current_price=p.get("ltp", p.get("last_price", 0)),
                    )
                    for p in positions if p.get("qty", p.get("quantity", 0)) != 0
                ]

                risk_status = self.risk_manager.update_mtm(risk_positions)

                await self._emit({
                    "type": "risk_status",
                    "risk_level": risk_status.risk_level.value,
                    "daily_pnl_pct": round(risk_status.daily_pnl_pct * 100, 2),
                    "unrealised_pnl": round(risk_status.unrealised_pnl, 2),
                    "realised_pnl": round(risk_status.realised_pnl, 2),
                    "total_pnl": round(risk_status.total_pnl, 2),
                    "kill_switch": risk_status.kill_switch_active,
                    "open_positions": len(risk_positions),
                    "timestamp": datetime.now().isoformat(),
                })

                if self.risk_manager.check_kill_switch() and not self.risk_manager.kill_switch_active:
                    logger.warning("KILL SWITCH TRIGGERED")
                    report = self.risk_manager.execute_kill_switch(self.broker)
                    await self._emit({
                        "type": "kill_switch",
                        "positions_closed": report.positions_squared,
                        "pnl_at_close": round(report.daily_pnl_pct * self.capital, 2),
                        "timestamp": datetime.now().isoformat(),
                    })
                    # Keep running for data/dashboard updates, just block new trades
                    logger.warning("Kill switch active — blocking new trades but continuing data stream")

            except Exception as e:
                logger.error("Risk monitor error: %s", e)

            await asyncio.sleep(5)

    # ── Option Chain ────────────────────────────────────────────────────

    async def _refresh_option_chain(self) -> None:
        """Fetch the option chain for strike selection.

        Uses the actual nearest expiry from exchange instruments to handle
        holidays (e.g., expiry preponed from Tuesday to Monday).
        """
        try:
            if hasattr(self._md_broker(), "get_option_chain"):
                today = date.today()
                config = INDEX_CONFIG.get(self.symbol, {})
                expiry_day_name = config.get("weekly_expiry_day", "Thursday")
                day_map = {
                    "Monday": 0, "Tuesday": 1, "Wednesday": 2,
                    "Thursday": 3, "Friday": 4,
                }
                target_weekday = day_map.get(expiry_day_name, 3)
                days_to_expiry = (target_weekday - today.weekday()) % 7
                if days_to_expiry == 0 and datetime.now().hour >= 15:
                    days_to_expiry = 7
                nearest_expiry = today + timedelta(days=days_to_expiry)

                # Try the calculated expiry first
                self._option_chain = self._md_broker().get_option_chain(
                    self.symbol, nearest_expiry.strftime("%Y-%m-%d")
                )

                # If 0 strikes, scan nearby dates (holiday preponment)
                if not self._option_chain:
                    logger.warning(
                        "0 strikes for %s — scanning nearby expiries (possible holiday)",
                        nearest_expiry,
                    )
                    # Try 1 day before, 2 days before, 1 day after
                    for delta in [-1, -2, 1, -3, 2]:
                        alt_expiry = nearest_expiry + timedelta(days=delta)
                        if alt_expiry < today:
                            continue
                        chain = self._md_broker().get_option_chain(
                            self.symbol, alt_expiry.strftime("%Y-%m-%d")
                        )
                        if chain:
                            logger.info(
                                "Found expiry at %s (%d strikes) — holiday shift from %s",
                                alt_expiry, len(chain), nearest_expiry,
                            )
                            self._option_chain = chain
                            nearest_expiry = alt_expiry
                            # Update expiry day check
                            self._is_expiry_day = (alt_expiry == today)
                            break

                logger.info(
                    "Option chain loaded | strikes=%d expiry=%s",
                    len(self._option_chain), nearest_expiry,
                )

                # ── Enrich with live quotes (LTP, OI, volume, bid/ask) ──
                # Kite chains only have static instrument metadata. This adds
                # live market data needed for SmartStrikeSelector scoring.
                if self._option_chain:
                    spot_price = 0.0
                    if self._bar_buffer:
                        spot_price = self._bar_buffer[-1].get("close", 0)
                    if spot_price <= 0:
                        # Try to get spot from broker
                        try:
                            config = INDEX_CONFIG.get(self.symbol, {})
                            spot_sym = config.get("underlying", "NIFTY 50")
                            ltps = self._md_broker().get_ltp([spot_sym])
                            spot_price = ltps.get(spot_sym, 0.0)
                        except Exception:
                            pass
                    if spot_price > 0:
                        strike_interval = INDEX_CONFIG.get(
                            self.symbol, {}
                        ).get("strike_interval", 50)
                        self._option_chain = enrich_option_chain_with_quotes(
                            self._md_broker(),
                            self._option_chain,
                            spot_price,
                            strike_interval=strike_interval,
                            range_strikes=10,
                        )
            else:
                logger.info("Broker does not support option chain — using synthetic symbols")
        except Exception as e:
            logger.warning("Failed to load option chain: %s", e)

    # ── Dashboard Events ────────────────────────────────────────────────

    async def _emit(self, event: dict) -> None:
        """Emit event to dashboard via callback."""
        if self._callback is None:
            return
        try:
            result = self._callback(event)
            if asyncio.iscoroutine(result) or asyncio.isfuture(result):
                await result
        except Exception:
            pass

    # ── Summary ─────────────────────────────────────────────────────────

    def _build_summary(self) -> dict:
        portfolio = {}
        try:
            portfolio = self.broker.get_portfolio()
        except Exception:
            pass

        analysis_summary = {}
        if self._latest_analysis:
            a = self._latest_analysis
            analysis_summary = {
                "vix": a.vix,
                "vix_regime": a.vix_regime.value,
                "pcr": a.pcr,
                "market_bias": a.market_bias.value,
                "recommended_action": a.recommended_action.value,
                "overall_score": round(a.overall_score, 4),
                "max_pain": a.max_pain,
            }

        return {
            "session_date": datetime.now().strftime("%Y-%m-%d"),
            "capital": self.capital,
            "symbol": self.symbol,
            "lot_size": self.lot_size,
            "agents": list(self.agents.keys()),
            "total_trades": len(self._open_trades) + len(self._closed_trades),
            "open_trades": len(self._open_trades),
            "closed_trades": len(self._closed_trades),
            "portfolio": portfolio,
            "bars_processed": self._bar_count,
            "kill_switch_triggered": self.risk_manager.kill_switch_active,
            "is_expiry_day": self._is_expiry_day,
            "last_analysis": analysis_summary,
        }

    # ── Status ──────────────────────────────────────────────────────────

    def get_status(self) -> dict:
        status = {
            "running": self._running,
            "symbol": self.symbol,
            "lot_size": self.lot_size,
            "agents": {
                name: {"position_open": a._position_open, "bars": len(a._bars)}
                for name, a in self.agents.items()
            },
            "bars_processed": self._bar_count,
            "open_trades": len(self._open_trades),
            "kill_switch": self.risk_manager.kill_switch_active,
            "rate_limiter_usage": self.rate_limiter.current_usage,
            "is_expiry_day": self._is_expiry_day,
            "vix": self._live_vix,
            "pcr": self._live_pcr,
        }
        if self._latest_analysis:
            a = self._latest_analysis
            status["market_analysis"] = {
                "bias": a.market_bias.value,
                "action": a.recommended_action.value,
                "score": round(a.overall_score, 4),
                "confidence": round(a.confidence, 3),
                "vix_regime": a.vix_regime.value,
            }
        return status
