"""Fully Autonomous Trading System — Zero Human Intervention.

Single entry point that handles EVERYTHING:
  1. Waits for market hours (skips weekends/holidays)
  2. Authenticates with broker (TOTP 2FA)
  3. Fetches previous close for gap calculation
  4. Exits BTST positions from previous day
  5. Starts the orchestrator with V3 agent
  6. Monitors throughout the day (entry, exit, multiple trades)
  7. Evaluates BTST candidates at 15:20
  8. Converts qualifying positions to NRML for overnight hold
  9. Squares off remaining MIS positions at 15:28
  10. Generates daily report
  11. Sleeps until next trading day
  12. Repeats forever

Usage:
    python run_autonomous.py                      # Live trading
    python run_autonomous.py --paper              # Paper trading (safe testing)
    python run_autonomous.py --paper --fast       # Fast paper (50x speed)
    python run_autonomous.py --paper-live-data    # Hybrid: paper orders + real Kite data

The system runs 24/7 — deploy on AWS EC2 / local machine and forget.
"""

import argparse
import asyncio
import atexit
import logging
import os
import signal
import sys
import time
from datetime import datetime, date, timedelta, time as dt_time
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import load_settings
from config.constants import (
    MARKET_OPEN_HOUR, MARKET_OPEN_MINUTE,
    MARKET_CLOSE_HOUR, MARKET_CLOSE_MINUTE,
    INDEX_CONFIG,
    STT_RATES, NSE_TRANSACTION_CHARGE, SEBI_TURNOVER_FEE,
    STAMP_DUTY_BUY, GST_RATE,
)
from orchestrator.live_orchestrator import LiveTradingOrchestrator
from orchestrator.order_manager import OrderManager
from orchestrator.position_tracker import PositionTracker
from backtesting.paper_trading import PaperTradingBroker
from dashboard.data_bridge import DashboardStateWriter
from orchestrator.claude_market_brain import ClaudeMarketBrain
import notifications as tg  # Telegram notifier


# ── Single Instance Lock ──────────────────────────────────────────
# Prevents multiple trading engines from running simultaneously.
# CRITICAL: Two engines = duplicate orders = double risk exposure.

_ENGINE_LOCK_FILE = Path(__file__).parent / "data" / ".trading_engine.lock"
_engine_lock_fd = None


def _acquire_engine_lock() -> bool:
    """Acquire exclusive engine lock. Returns True if we are the only instance."""
    global _engine_lock_fd
    _ENGINE_LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        _engine_lock_fd = open(_ENGINE_LOCK_FILE, "w")
        if sys.platform == "win32":
            import msvcrt
            msvcrt.locking(_engine_lock_fd.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(_engine_lock_fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        _engine_lock_fd.write(str(os.getpid()))
        _engine_lock_fd.flush()
        atexit.register(_release_engine_lock)
        return True
    except (OSError, IOError):
        _engine_lock_fd = None
        return False


def _release_engine_lock():
    """Release engine lock on exit."""
    global _engine_lock_fd
    if _engine_lock_fd is not None:
        try:
            if sys.platform == "win32":
                import msvcrt
                try:
                    _engine_lock_fd.seek(0)
                    msvcrt.locking(_engine_lock_fd.fileno(), msvcrt.LK_UNLCK, 1)
                except Exception:
                    pass
            _engine_lock_fd.close()
        except Exception:
            pass
        _engine_lock_fd = None
    try:
        if _ENGINE_LOCK_FILE.exists():
            _ENGINE_LOCK_FILE.unlink()
    except Exception:
        pass

# ── Logging ────────────────────────────────────────────────────────

LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)


def setup_logging(level: str = "INFO"):
    fmt = "%(asctime)s.%(msecs)03d | %(levelname)-8s | %(name)-25s | %(message)s"
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            LOG_DIR / f"autonomous_{date.today().isoformat()}.log", mode="a"
        ),
    ]
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=fmt,
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("websockets").setLevel(logging.WARNING)


logger = logging.getLogger("autonomous")


# ── Charge Estimation (match Kite app P&L) ────────────────────────

def estimate_charges_from_positions(
    broker_positions: list[dict],
    num_orders: int = 0,
) -> float:
    """Estimate total trading charges from broker positions' buy/sell values.

    Kite API position.pnl = sell_value - buy_value (raw, no charges).
    Kite App shows P&L AFTER deducting charges (STT, brokerage, etc.).

    This function estimates those charges so our dashboard matches Kite.

    Charges breakdown (NSE Options):
        - STT: 0.15% on sell-side premium (options)
        - Brokerage: Rs 20 per executed order (Zerodha flat fee)
        - NSE Transaction: 0.0003553 x turnover
        - SEBI Fee: 0.000001 x turnover
        - Stamp Duty: 0.003% on buy-side turnover
        - GST: 18% on (brokerage + transaction + SEBI fees)
    """
    total_buy_value = 0.0
    total_sell_value = 0.0

    for bp in broker_positions:
        bv = abs(bp.get("buy_value", 0.0))
        sv = abs(bp.get("sell_value", 0.0))
        total_buy_value += bv
        total_sell_value += sv

    total_turnover = total_buy_value + total_sell_value

    if total_turnover == 0:
        return 0.0

    # If we don't know number of orders, estimate from positions
    # Each position = ~2 orders (buy + sell), but some may still be open (1 order)
    if num_orders == 0:
        num_orders = len(broker_positions) * 2

    # STT: 0.15% on sell-side premium (options)
    stt = total_sell_value * STT_RATES.get("options_sell", 0.0015)

    # Brokerage: Rs 20 per executed order (capped)
    brokerage = num_orders * 20.0

    # NSE Transaction charges
    txn_charges = total_turnover * NSE_TRANSACTION_CHARGE

    # SEBI turnover fee
    sebi_fee = total_turnover * SEBI_TURNOVER_FEE

    # Stamp duty: on buy side only
    stamp = total_buy_value * STAMP_DUTY_BUY

    # GST: 18% on brokerage + transaction + SEBI
    gst = (brokerage + txn_charges + sebi_fee) * GST_RATE

    total_charges = stt + brokerage + txn_charges + sebi_fee + stamp + gst
    return round(total_charges, 2)


# ── Market Calendar ────────────────────────────────────────────────

# NSE holidays 2026 (add more as needed)
NSE_HOLIDAYS_2026 = {
    date(2026, 1, 26),   # Republic Day
    date(2026, 3, 10),   # Holi
    date(2026, 3, 30),   # Id-ul-Fitr
    date(2026, 4, 2),    # Ram Navami
    date(2026, 4, 3),    # Mahavir Jayanti
    date(2026, 4, 14),   # Dr. Ambedkar Jayanti
    date(2026, 5, 1),    # May Day
    date(2026, 6, 5),    # Id-ul-Zuha
    date(2026, 7, 6),    # Muharram
    date(2026, 8, 15),   # Independence Day
    date(2026, 8, 21),   # Janmashtami
    date(2026, 9, 4),    # Milad-un-Nabi
    date(2026, 10, 2),   # Gandhi Jayanti
    date(2026, 10, 20),  # Dussehra
    date(2026, 11, 9),   # Diwali (Lakshmi Puja)
    date(2026, 11, 10),  # Diwali (Balipratipada)
    date(2026, 11, 30),  # Guru Nanak Jayanti
    date(2026, 12, 25),  # Christmas
}

MARKET_OPEN = dt_time(MARKET_OPEN_HOUR, MARKET_OPEN_MINUTE)   # 9:15
MARKET_CLOSE = dt_time(MARKET_CLOSE_HOUR, MARKET_CLOSE_MINUTE)  # 15:30
PRE_MARKET_START = dt_time(9, 10)  # Start systems 5 min before market
BTST_EVALUATION_TIME = dt_time(15, 20)  # Evaluate BTST at 15:20
BTST_EXIT_DEADLINE = dt_time(9, 45)  # Exit BTST by 9:45 AM


def is_market_day(d: date = None) -> bool:
    """Check if given date is a trading day (not weekend/holiday)."""
    d = d or date.today()
    if d.weekday() >= 5:  # Saturday=5, Sunday=6
        return False
    if d in NSE_HOLIDAYS_2026:
        return False
    return True


def next_market_day(d: date = None) -> date:
    """Find the next trading day."""
    d = d or date.today()
    d += timedelta(days=1)
    while not is_market_day(d):
        d += timedelta(days=1)
    return d


def seconds_until(target: dt_time) -> float:
    """Seconds from now until target time today."""
    now = datetime.now()
    target_dt = now.replace(
        hour=target.hour, minute=target.minute, second=0, microsecond=0
    )
    diff = (target_dt - now).total_seconds()
    return max(0, diff)


# ── Broker Factory ─────────────────────────────────────────────────

def create_broker(settings, paper: bool = False):
    """Create broker instance."""
    if paper or settings.trading.paper_trading:
        logger.info("📄 Paper trading mode — using simulated broker")
        return PaperTradingBroker(initial_capital=settings.trading.capital)

    broker_type = settings.broker.broker_type.lower()
    if broker_type == "zerodha":
        from broker.kite_connect import KiteConnectBroker
        return KiteConnectBroker(
            api_key=settings.broker.api_key,
            api_secret=settings.broker.api_secret,
            user_id=settings.broker.user_id,
            password=settings.broker.password,
            totp_secret=settings.broker.totp_secret,
        )
    elif broker_type == "fyers":
        from broker.fyers_broker import FyersBroker
        return FyersBroker(
            client_id=settings.broker.api_key,
            secret_key=settings.broker.api_secret,
            totp_secret=settings.broker.totp_secret,
        )
    else:
        raise ValueError(f"Unsupported broker: {broker_type}")


def fetch_previous_close(broker, symbol: str = "NIFTY") -> float:
    """Fetch previous day's closing price for gap calculation."""
    from datetime import timedelta

    underlying = INDEX_CONFIG.get(symbol, {}).get(
        "underlying_symbol", f"NSE:{symbol} 50"
    )

    # Method 1: Kite OHLC API returns previous close directly
    try:
        qualified = [f"NSE:{underlying}"]
        data = broker._kite.ohlc(qualified)
        for key, entry in data.items():
            ohlc = entry.get("ohlc", {})
            prev_close = ohlc.get("close", 0)
            if prev_close > 0:
                logger.info("Previous close (via OHLC API): %.2f", prev_close)
                return float(prev_close)
    except Exception as e:
        logger.debug("OHLC prev close failed: %s", e)

    # Method 2: Historical data — yesterday's last bar
    try:
        now = datetime.now()
        # Go back up to 5 days to find last trading day
        for days_back in range(1, 6):
            prev_day = now - timedelta(days=days_back)
            bars = broker.get_historical_data(
                symbol=underlying,
                from_dt=prev_day.replace(hour=9, minute=15, second=0),
                to_dt=prev_day.replace(hour=15, minute=30, second=0),
                interval="day",
            )
            if bars:
                close = bars[-1].get("close", 0)
                if close > 0:
                    logger.info("Previous close (via historical): %.2f (date=%s)", close, prev_day.strftime("%Y-%m-%d"))
                    return float(close)
    except Exception as e:
        logger.debug("Historical prev close failed: %s", e)

    # Method 3: Fallback to LTP (worst case — returns current price, not prev close)
    try:
        ltp = broker.get_ltp([underlying])
        if ltp:
            price = list(ltp.values())[0]
            logger.warning("Previous close FALLBACK (using LTP, may be inaccurate): %.2f", price)
            return price
    except Exception as e:
        logger.debug("LTP fallback failed: %s", e)

    # Fallback: use approximate NIFTY level (updated manually)
    fallback = 22700.0  # Apr 2026 approximate
    logger.warning("Using fallback previous close: %.0f", fallback)
    return fallback


# ── BTST Exit Logic ────────────────────────────────────────────────

async def exit_btst_positions(
    broker,
    order_manager: OrderManager,
    position_tracker: PositionTracker,
    rate_limiter,
) -> float:
    """Exit BTST positions from previous day in the first 30 minutes.

    Strategy: Sell in the first 15-30 minutes when liquidity is high.
    If profitable, sell immediately. If at loss, wait up to 30 min for recovery.

    Returns total realized PnL from BTST exits.
    """
    btst_positions = position_tracker.load_btst()
    if not btst_positions:
        logger.info("No BTST positions to exit")
        return 0.0

    total_pnl = 0.0
    logger.info("=" * 60)
    logger.info("BTST EXIT: %d positions to close", len(btst_positions))
    logger.info("=" * 60)

    for pos in btst_positions:
        symbol = pos["symbol"]
        side = pos["side"]
        qty = pos["qty"]
        entry_price = pos["entry_price"]

        # Get current price
        try:
            ltp_map = broker.get_ltp([symbol])
            current = ltp_map.get(symbol, 0)
        except Exception:
            current = 0

        # ── Cancel GTT backup SL from yesterday BEFORE closing ──
        # The V17 NRML entry placed a broker-side GTT stop-loss whose
        # id was persisted into pos["metadata"]["gtt_id"] by
        # live_orchestrator._execute_trade. If we close the position
        # without cancelling the GTT first, the GTT can trigger
        # independently on the morning gap and create a duplicate
        # order (or worse, flip us to the other side).
        gtt_id = (pos.get("metadata") or {}).get("gtt_id")
        if gtt_id is not None and hasattr(broker, "_kite"):
            try:
                broker._kite.delete_gtt(gtt_id)
                logger.info(
                    "BTST GTT cancelled | %s | gtt_id=%s", symbol, gtt_id,
                )
            except Exception as e:
                logger.warning(
                    "BTST GTT cancel failed | %s | gtt_id=%s | %s — "
                    "GTT may trigger independently; proceeding with exit",
                    symbol, gtt_id, e,
                )

        close_side = "SELL" if side == "BUY" else "BUY"

        # Exit via order manager (with polling and retry)
        result = await order_manager.place_and_confirm(
            symbol=symbol,
            side=close_side,
            qty=qty,
            order_type="MARKET",
            product="NRML",  # BTST positions use NRML
            tag="btst_exit",
        )

        if result.is_filled:
            if side == "BUY":
                pnl = (result.fill_price - entry_price) * qty
            else:
                pnl = (entry_price - result.fill_price) * qty
            total_pnl += pnl
            position_tracker.close_position(symbol, result.fill_price, pnl, "btst_exit")
            logger.info(
                "BTST EXIT | %s | entry=%.2f exit=%.2f | pnl=%.2f",
                symbol, entry_price, result.fill_price, pnl,
            )
        else:
            logger.error("BTST EXIT FAILED | %s | status=%s | %s", symbol, result.status, result.error)

    position_tracker.clear_btst()
    logger.info("BTST EXIT COMPLETE | total_pnl=%.2f", total_pnl)
    return total_pnl


# ── BTST Evaluation (EOD) ──────────────────────────────────────────

async def evaluate_and_convert_btst(
    orchestrator: LiveTradingOrchestrator,
    broker,
    order_manager: OrderManager,
    position_tracker: PositionTracker,
) -> list[dict]:
    """At 15:20, evaluate open positions for BTST and convert to NRML.

    Steps:
      1. Get current VIX and market trend
      2. Evaluate which positions qualify for BTST
      3. Convert qualifying positions from MIS --> NRML
      4. Save BTST state for next-day exit

    Returns list of BTST positions.
    """
    vix = orchestrator._live_vix
    is_expiry = orchestrator._is_expiry_day
    trend = ""
    if orchestrator._latest_analysis:
        trend = orchestrator._latest_analysis.market_bias.value

    # Evaluate candidates
    btst_candidates = position_tracker.evaluate_btst(
        vix=vix,
        is_expiry_day=is_expiry,
        market_trend=trend,
    )

    if not btst_candidates:
        logger.info("No positions qualify for BTST")
        return []

    logger.info("=" * 60)
    logger.info("BTST CONVERSION: %d positions qualifying", len(btst_candidates))
    logger.info("=" * 60)

    # Convert MIS --> NRML by closing MIS and reopening as NRML
    # (Most Indian brokers require this — can't change product type mid-trade)
    converted = []
    for pos in btst_candidates:
        symbol = pos["symbol"]
        side = pos["side"]
        qty = pos["qty"]

        # Step 1: Close MIS position
        close_side = "SELL" if side == "BUY" else "BUY"
        close_result = await order_manager.place_and_confirm(
            symbol=symbol, side=close_side, qty=qty,
            order_type="MARKET", product="MIS", tag="btst_mis_close",
        )

        if not close_result.is_filled:
            logger.error("BTST: Failed to close MIS for %s — skipping", symbol)
            continue

        # Step 2: Reopen as NRML
        open_result = await order_manager.place_and_confirm(
            symbol=symbol, side=side, qty=qty,
            order_type="MARKET", product="NRML", tag="btst_nrml_open",
        )

        if open_result.is_filled:
            pos["product"] = "NRML"
            pos["btst_entry_price"] = open_result.fill_price
            converted.append(pos)
            logger.info(
                "BTST CONVERTED | %s %s qty=%d | MIS-->NRML @ %.2f",
                side, symbol, qty, open_result.fill_price,
            )
        else:
            logger.error(
                "BTST: MIS closed but NRML open FAILED for %s — POSITION LOST",
                symbol,
            )
            # This is critical — log prominently
            logger.critical(
                "CRITICAL: %s position closed (MIS) but NRML reopen failed. "
                "Manual intervention needed!",
                symbol,
            )

    # Save for next-day exit
    if converted:
        position_tracker.mark_btst(converted)
    return converted


# ── Main Autonomous Loop ───────────────────────────────────────────

class AutonomousTrader:
    """The fully autonomous trading daemon.

    Lifecycle per day:
      1. 09:10 — Wake up, authenticate broker
      2. 09:12 — Fetch previous close, load BTST
      3. 09:15 — Market opens, start orchestrator
      4. 09:15-09:45 — Exit BTST positions from yesterday
      5. 09:15-15:20 — Autonomous trading (entries, exits, multiple trades)
      6. 15:20 — Evaluate BTST candidates, convert MIS-->NRML
      7. 15:28 — Square off remaining MIS positions
      8. 15:30 — Market closes, generate daily report
      9. Sleep until next market day --> Repeat
    """

    def __init__(
        self,
        paper: bool = False,
        fast: bool = False,
        symbol: str = "NIFTY",
        paper_live_data: bool = False,
    ):
        # Hybrid "paper execution + real Kite data" mode:
        # Order flow still goes through PaperTradingBroker (zero real money
        # risk), but a real authenticated KiteConnectBroker is injected as
        # the orchestrator's _data_broker for ticks, historical bars, VIX,
        # and option chain. This is the Monday pre-deployment validation
        # mode — it exercises every live-data code path without placing
        # any real orders.
        self.paper_live_data = paper_live_data
        # paper_live_data implies paper: execution must be simulated.
        if paper_live_data:
            paper = True
        self.paper = paper
        self.fast = fast
        self.symbol = symbol
        self.settings = load_settings()
        self._shutdown = False

        # Initialise Telegram notifier (reads TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID from env)
        tg.configure()
        self._orchestrator: LiveTradingOrchestrator | None = None
        # Real Kite broker kept alongside self.broker in hybrid mode so
        # we can share authentication + instrument cache with the
        # orchestrator. None in normal paper and normal live.
        self._kite_data_broker = None

        # Override paper mode from CLI
        if paper:
            self.settings.trading.paper_trading = True

        # Subsystems
        self.position_tracker = PositionTracker()
        self.broker = None
        self.order_manager = None
        self.dashboard_writer = DashboardStateWriter()
        self._prev_close: float = 0.0
        self._session_started_at: str = ""
        self._signals_generated: int = 0
        self._signals_accepted: int = 0
        self._signals_filtered: int = 0
        self._last_signal: str = ""

        # Broker position cache (used by dashboard state and charge estimation)
        self._cached_broker_positions: list[dict] = []
        self._last_positions_fetch: float = 0.0
        self._last_capital_fetch: float = 0.0
        self._zerodha_capital: float = self.settings.trading.capital

        # Real-time tick LTP cache for position PnL (symbol --> ltp)
        # Updated from KiteTicker ticks, gives sub-second PnL updates
        self._tick_ltp_cache: dict[str, float] = {}
        self._position_tokens_subscribed: set[str] = set()

        # Claude AI Market Brain
        self.claude_brain = ClaudeMarketBrain(
            analysis_interval=300,  # Every 5 minutes
        )
        self._claude_last_status: str = ""

        logger.info(
            "AutonomousTrader initialized | paper=%s fast=%s symbol=%s capital=%.0f",
            paper, fast, symbol, self.settings.trading.capital,
        )

    async def run_forever(self) -> None:
        """Main loop — runs trading sessions day after day."""
        logger.info("=" * 70)
        logger.info("  AUTONOMOUS TRADING SYSTEM STARTED")
        logger.info("  Mode: %s | Symbol: %s | Capital: %.0f",
                     "PAPER" if self.paper else "LIVE", self.symbol,
                     self.settings.trading.capital)
        logger.info("=" * 70)

        while not self._shutdown:
            try:
                today = date.today()

                if not is_market_day(today):
                    next_day = next_market_day(today)
                    wait_hours = (
                        datetime.combine(next_day, PRE_MARKET_START) - datetime.now()
                    ).total_seconds() / 3600
                    logger.info(
                        "Not a market day. Next trading: %s (%.1f hours away)",
                        next_day, wait_hours,
                    )
                    # Sleep in chunks (check for shutdown)
                    await self._sleep_until_datetime(
                        datetime.combine(next_day, PRE_MARKET_START)
                    )
                    continue

                # It's a market day
                now = datetime.now().time()

                if now < PRE_MARKET_START:
                    # Wait until 9:10
                    wait = seconds_until(PRE_MARKET_START)
                    logger.info("Waiting %.0f seconds until pre-market (09:10)...", wait)
                    await self._sleep_for(wait)

                if now > MARKET_CLOSE:
                    # Market already closed today
                    next_day = next_market_day(today)
                    logger.info("Market closed for today. Next: %s", next_day)
                    await self._sleep_until_datetime(
                        datetime.combine(next_day, PRE_MARKET_START)
                    )
                    continue

                # Run today's session
                await self._run_daily_session()

                # After market close, sleep until next day
                next_day = next_market_day(today)
                logger.info(
                    "Session complete. Next trading day: %s", next_day,
                )
                await self._sleep_until_datetime(
                    datetime.combine(next_day, PRE_MARKET_START)
                )

            except KeyboardInterrupt:
                logger.info("Keyboard interrupt — shutting down gracefully")
                self._shutdown = True
            except Exception as e:
                logger.error("Unexpected error in main loop: %s", e, exc_info=True)
                # Don't crash — wait 60s and retry
                await asyncio.sleep(60)

        logger.info("Autonomous trader stopped.")

    async def _run_daily_session(self) -> None:
        """Execute a complete trading day session."""
        logger.info("=" * 70)
        logger.info("  DAILY SESSION: %s (%s)",
                     date.today().isoformat(),
                     datetime.now().strftime("%A"))
        logger.info("=" * 70)

        # ── Step 1: Authenticate ──
        logger.info("[1/8] Authenticating with broker...")
        self.broker = create_broker(self.settings, paper=self.paper)

        if not isinstance(self.broker, PaperTradingBroker):
            # Fully automated TOTP login — no manual input needed
            logger.info("Authenticating with Kite Connect (auto-TOTP)...")
            if not self.broker.authenticate():
                # If auto-TOTP fails, fall back to manual code entry
                logger.warning("Auto-TOTP login failed — trying manual code entry")
                print("\n" + "=" * 50)
                print("  KITE LOGIN — Auto-TOTP failed")
                print("  Enter the 6-digit code from your authenticator app:")
                print("=" * 50)
                app_code = input("  App Code > ").strip()
                print("=" * 50 + "\n")

                if app_code and len(app_code) == 6:
                    self.broker._auth.set_manual_totp(app_code)
                else:
                    logger.error("Invalid app code — skipping today")
                    return

                if not self.broker.authenticate():
                    logger.error("Broker authentication FAILED — skipping today")
                    return
        logger.info("[1/8] Broker authenticated")

        # ── Step 1b: Hybrid mode — create + authenticate real Kite for DATA only ──
        if self.paper_live_data:
            logger.info("[1b/8] HYBRID MODE: authenticating real Kite for market data...")
            from broker.kite_connect import KiteConnectBroker
            self._kite_data_broker = KiteConnectBroker(
                api_key=self.settings.broker.api_key,
                api_secret=self.settings.broker.api_secret,
                user_id=self.settings.broker.user_id,
                password=self.settings.broker.password,
                totp_secret=self.settings.broker.totp_secret,
            )
            if not self._kite_data_broker.authenticate():
                logger.warning("HYBRID: auto-TOTP failed for data broker — try manual code")
                print("\n" + "=" * 50)
                print("  KITE LOGIN (HYBRID DATA BROKER) — Auto-TOTP failed")
                print("  Enter the 6-digit code from your authenticator app:")
                print("=" * 50)
                app_code = input("  App Code > ").strip()
                print("=" * 50 + "\n")
                if app_code and len(app_code) == 6:
                    self._kite_data_broker._auth.set_manual_totp(app_code)
                else:
                    logger.error("HYBRID: invalid app code — aborting day")
                    return
                if not self._kite_data_broker.authenticate():
                    logger.error("HYBRID: data broker auth FAILED — aborting day")
                    return
            logger.info(
                "[1b/8] HYBRID: real Kite data broker authenticated — "
                "execution stays paper, ticks/bars/VIX/chain use live Kite"
            )

        # ── Step 2: Fetch previous close ──
        # In hybrid mode, use the real Kite data broker (paper broker
        # doesn't have historical OHLC access).
        logger.info("[2/8] Fetching previous close...")
        _prev_close_broker = (
            self._kite_data_broker if self.paper_live_data
            else self.broker
        )
        prev_close = fetch_previous_close(_prev_close_broker, self.symbol)
        self._prev_close = prev_close
        self._session_started_at = datetime.now().isoformat()
        logger.info("[2/8] Previous close: %.2f", prev_close)

        # ── Step 2.5: Fetch REAL capital from Zerodha ──
        if not self.paper:
            try:
                portfolio = self.broker.get_portfolio()
                margins = portfolio.get("margins", {})
                equity = margins.get("equity", {})
                available = equity.get("available", {})
                # Use equity.net (total equity) for dashboard display + PnL% calculation
                # Use available.live_balance for position sizing (actual buyable amount)
                net_equity = equity.get("net", 0)
                live_balance = available.get("live_balance", 0)
                # For dashboard capital display: use net equity (total account value)
                display_capital = net_equity if net_equity > 0 else live_balance
                # For position sizing: use live_balance (what's actually available to trade)
                trading_capital = live_balance if live_balance > 0 else display_capital
                if display_capital > 0:
                    self._zerodha_capital = display_capital
                    self._zerodha_trading_capital = trading_capital
                    self.settings.trading.capital = trading_capital  # Position sizing uses available
                    logger.info("[2.5/8] Zerodha capital: net_equity=Rs %.2f, available=Rs %.2f",
                                display_capital, trading_capital)
                else:
                    logger.warning("[2.5/8] Could not fetch balance — using config capital: Rs %.0f",
                                   self.settings.trading.capital)
            except Exception as e:
                logger.warning("[2.5/8] Capital fetch failed (%s) — using config: Rs %.0f",
                               e, self.settings.trading.capital)

        # ── Step 3: Initialize subsystems ──
        logger.info("[3/8] Initializing subsystems...")
        from orchestrator.rate_limiter import AsyncRateLimiter
        rate_limiter = AsyncRateLimiter(max_ops=9)
        self.order_manager = OrderManager(self.broker, rate_limiter)
        self.position_tracker.reset_daily()
        self.position_tracker.load_state()  # Recover from crash if any

        # ── Step 4: Create orchestrator ──
        logger.info("[4/8] Creating orchestrator...")

        # Dashboard callback — log events and feed dashboard state writer
        async def on_event(event):
            etype = event.get("type", "")
            if etype in ("trade_executed", "trade_closed", "kill_switch", "square_off"):
                logger.info("EVENT | %s | %s", etype, {
                    k: v for k, v in event.items() if k != "type"
                })
            # Track positions in our tracker
            if etype == "trade_executed" and event.get("status") == "EXECUTED":
                sym = event.get("symbol", "")
                action = event.get("action", "")
                price = event.get("entry_price", 0)
                qty = event.get("qty", 0)
                lots = event.get("lots", qty // 65)
                self.dashboard_writer.add_log(
                    f"TRADE: {action} {sym} x{qty} @ Rs{price:.2f}"
                )
                self._signals_accepted += 1
                self._last_signal = f"{action} @ {datetime.now().strftime('%H:%M')}"
                tg.on_trade_entry(
                    symbol=sym, side=action, qty=qty, price=price,
                    lots=lots, underlying=event.get("underlying_price", 0),
                )
            if etype == "trade_closed":
                trade_pnl = event.get("pnl", 0)
                self.position_tracker._daily_pnl += trade_pnl
                sym = event.get("symbol", "")
                reason = event.get("exit_reason", "")
                self.dashboard_writer.add_log(
                    f"EXIT: {sym} | PnL: Rs{trade_pnl:,.2f} | {reason}"
                )
                tg.on_trade_exit(
                    symbol=sym,
                    entry=event.get("entry_price", 0),
                    exit_price=event.get("exit_price", 0),
                    qty=event.get("qty", 0),
                    pnl=trade_pnl,
                    reason=reason,
                )
            if etype == "kill_switch":
                self.dashboard_writer.add_log("KILL SWITCH TRIGGERED!")
                tg.on_kill_switch(
                    day_pnl=event.get("day_pnl", 0),
                    threshold_pct=event.get("threshold_pct", 3.0),
                )
            if etype == "market_analysis":
                bias = event.get("market_bias", "")
                self.dashboard_writer.add_log(f"Analysis: {bias} (conf={event.get('confidence', 0):.0%})")
            if etype == "signal_filtered":
                self._signals_filtered += 1
            if etype in ("trade_executed", "signal_generated"):
                self._signals_generated += 1

        self._orchestrator = LiveTradingOrchestrator(
            broker=self.broker,
            capital=self.settings.trading.capital,
            strategies=["v14_production"],  # V14 = production best (587.5x, 11/11 months+, VWAP+RSI+Squeeze)
            callback=on_event,
            symbol=self.symbol,
        )

        # Set previous close on the V3 agent
        for agent in self._orchestrator.agents.values():
            if hasattr(agent, "set_previous_close"):
                agent.set_previous_close(prev_close)

        # Inject order manager and position tracker
        self._orchestrator._order_manager = self.order_manager
        self._orchestrator._position_tracker = self.position_tracker

        # Hybrid mode: inject the real Kite broker as the data source.
        # The orchestrator will route get_ltp/get_historical_data/
        # get_option_chain/subscribe_ticks/get_vix through this instead
        # of the paper broker, while order execution stays on self.broker.
        if self.paper_live_data and self._kite_data_broker is not None:
            self._orchestrator._data_broker = self._kite_data_broker
            logger.info(
                "[4/8] HYBRID: orchestrator._data_broker set to real Kite "
                "(is_hybrid=%s)", self._orchestrator._is_hybrid,
            )

        logger.info("[4/8] Orchestrator created with agents: %s",
                     list(self._orchestrator.agents.keys()))

        # ── Step 5: Wait for market open ──
        # Live AND hybrid both wait for the real opening bell (hybrid
        # subscribes to real KiteTicker). Only pure paper skips.
        if not self.paper or self.paper_live_data:
            wait = seconds_until(MARKET_OPEN)
            if wait > 0:
                logger.info("[5/8] Waiting %.0f seconds for market open (09:15)...", wait)
                await self._sleep_for(wait)
        logger.info("[5/8] Market open — starting session")

        # ── Step 6: Exit BTST positions from yesterday ──
        logger.info("[6/8] Checking BTST positions from yesterday...")
        btst_pnl = await exit_btst_positions(
            self.broker, self.order_manager, self.position_tracker, rate_limiter,
        )
        if btst_pnl != 0:
            logger.info("[6/8] BTST exits complete | PnL: %.2f", btst_pnl)

        # Cancel any stale open orders to free up margin
        try:
            orders = self.broker.get_orders()
            open_orders = [o for o in orders if o.get("status") in ("OPEN", "PENDING", "TRIGGER PENDING")]
            if open_orders:
                logger.info("Cancelling %d stale open orders to free margin", len(open_orders))
                for o in open_orders:
                    self.broker.cancel_order(o["order_id"])
        except Exception as e:
            logger.debug("Open order cleanup: %s", e)

        # Close orphaned positions from previous engine crashes
        try:
            broker_positions = self.broker.get_positions()
            orphans = [p for p in broker_positions
                       if p.get("qty", p.get("quantity", 0)) != 0
                       and p.get("product") == "MIS"]
            if orphans:
                logger.warning("ORPHAN CLEANUP: %d positions from previous engine", len(orphans))
                for p in orphans:
                    sym = p["symbol"]
                    qty = abs(p.get("qty", p.get("quantity", 0)))
                    side = "SELL" if p.get("qty", p.get("quantity", 0)) > 0 else "BUY"
                    logger.info("Closing orphan: %s %s qty=%d", side, sym, qty)
                    try:
                        result = await self.order_manager.place_and_confirm(
                            symbol=sym, side=side, qty=qty,
                            order_type="MARKET", product="MIS",
                            tag="orphan_cleanup",
                        )
                        if result.is_filled:
                            logger.info("Orphan closed: %s @ %.2f", sym, result.fill_price)
                        else:
                            logger.error("Orphan close FAILED: %s | %s", sym, result.error)
                    except Exception as e:
                        logger.error("Orphan close error: %s | %s", sym, e)
        except Exception as e:
            logger.debug("Orphan cleanup: %s", e)

        # ── Seed risk manager with broker's actual P&L ──
        # CRITICAL: After restart, the risk manager has no knowledge of earlier
        # losses (from previous engine session or manual trades). Fetch the
        # broker's total day P&L and seed it so the kill switch works correctly.
        try:
            broker_positions = self.broker.get_positions()
            broker_realized = sum(
                bp.get("pnl", 0) for bp in broker_positions
                if bp.get("qty", 0) == 0 and bp.get("pnl", 0) != 0
            )
            broker_unrealized = sum(
                bp.get("pnl", 0) for bp in broker_positions
                if bp.get("qty", 0) != 0
            )
            broker_total = broker_realized + broker_unrealized
            if broker_total < 0:
                self._orchestrator.risk_manager.add_realised_pnl(broker_realized)
                logger.warning(
                    "RISK SEED: Broker day P&L = %.2f (realized=%.2f, unrealized=%.2f) — "
                    "kill switch will account for pre-restart losses",
                    broker_total, broker_realized, broker_unrealized,
                )
                # Also seed the position tracker so dashboard stats are correct
                self.position_tracker._daily_pnl = broker_realized

                # Check if kill switch should already be triggered
                # NOTE: Disabled on startup — unrealized losses from adopted
                # positions skew the % against margin-only capital. The
                # continuous kill-switch check during the session still works.
                if self._orchestrator.risk_manager.check_kill_switch():
                    logger.warning(
                        "KILL SWITCH would trigger on startup (day loss %.2f / %.1f%%) "
                        "— SUPPRESSED to allow trading with adopted positions",
                        broker_total,
                        self._orchestrator.risk_manager.max_daily_loss_pct * 100,
                    )
        except Exception as e:
            logger.error("Risk seed from broker failed: %s", e)

        # ── Step 7: Run orchestrator (main trading loop) ──
        logger.info("[7/8] Starting autonomous trading...")

        # Initial dashboard state
        self.dashboard_writer.add_log("Trading session started")
        self._write_dashboard_state("TRADING")

        # Telegram: system started
        tg.on_system_start(
            symbol=self.symbol,
            capital=self._orchestrator.risk_manager.total_capital,
            mode="PAPER" if self.paper else "LIVE",
        )

        # Run orchestrator with BTST evaluation and dashboard update tasks
        orch_task = asyncio.create_task(
            self._orchestrator.run(), name="orchestrator"
        )
        btst_task = asyncio.create_task(
            self._btst_evaluation_loop(), name="btst_eval"
        )
        reconcile_task = asyncio.create_task(
            self._reconciliation_loop(), name="reconciliation"
        )
        dashboard_task = asyncio.create_task(
            self._dashboard_state_loop(), name="dashboard_state"
        )
        manual_order_task = asyncio.create_task(
            self._manual_order_loop(), name="manual_orders"
        )

        try:
            # Wait for orchestrator to finish (market close)
            summary = await orch_task
        except Exception as e:
            logger.error("Orchestrator error: %s", e, exc_info=True)
            summary = {}
        finally:
            btst_task.cancel()
            reconcile_task.cancel()
            dashboard_task.cancel()
            manual_order_task.cancel()
            try:
                await asyncio.gather(
                    btst_task, reconcile_task, dashboard_task, manual_order_task,
                    return_exceptions=True,
                )
            except Exception:
                pass

        # Final dashboard write
        self._write_dashboard_state("MARKET_CLOSED")

        # ── Final reconciliation safety net ──
        # After the reconciliation loop has been cancelled, do one more pass to
        # catch any broker auto-squareoff that fired between the last loop tick
        # and now. Without this, MIS positions squared by Zerodha between 15:15
        # and 15:28 can slip through and produce a zeroed daily report.
        try:
            if self.broker and self.position_tracker.open_positions:
                broker_positions = self.broker.get_positions() or []
                final_report = self.position_tracker.reconcile_with_broker(
                    broker_positions
                )
                if final_report.get("missing_in_broker"):
                    closed = self._close_broker_squared_positions(
                        final_report["missing_in_broker"]
                    )
                    if closed:
                        logger.warning(
                            "EOD safety-net recorded %d broker-squared exits: %s",
                            len(closed), closed,
                        )
        except Exception as e:
            logger.error("EOD final reconciliation failed: %s", e, exc_info=True)

        # ── Step 8: Daily report ──
        logger.info("[8/8] Generating daily report...")
        report = self.position_tracker.generate_daily_report()

        logger.info("=" * 70)
        logger.info("  DAILY SUMMARY: %s", date.today().isoformat())
        logger.info("  Trades: %d | Winners: %d | Losers: %d | Win Rate: %.1f%%",
                     report["total_trades"], report["winners"],
                     report["losers"], report["win_rate"])
        logger.info("  Daily PnL: Rs %.2f | BTST held: %d",
                     report["daily_pnl"], report["btst_positions"])
        logger.info("  Bars processed: %d", summary.get("bars_processed", 0))
        logger.info("=" * 70)

        # Telegram: daily summary
        tg.on_daily_summary(
            realized=report.get("daily_pnl", 0),
            unrealized=0,
            trades=report.get("total_trades", 0),
            wins=report.get("winners", 0),
            losses=report.get("losers", 0),
            capital=getattr(self._orchestrator.risk_manager, "total_capital", 0),
        )
        tg.on_system_stop(reason="Market closed (15:30)")

        # Cleanup
        if self.broker and hasattr(self.broker, "close"):
            try:
                self.broker.close()
            except Exception:
                pass

    async def _btst_evaluation_loop(self) -> None:
        """At 15:20, evaluate positions for BTST conversion."""
        while not self._shutdown:
            now = datetime.now().time()

            if self.paper:
                # Paper mode: evaluate after all bars are processed
                await asyncio.sleep(5)
                # Check if orchestrator is near end
                if (self._orchestrator and
                    self._orchestrator._bar_count >= 350 and
                    not hasattr(self, '_btst_evaluated')):
                    self._btst_evaluated = True
                    await self._do_btst_evaluation()
                continue

            # Live mode: wait for 15:20
            if now >= BTST_EVALUATION_TIME and not hasattr(self, '_btst_evaluated'):
                self._btst_evaluated = True
                await self._do_btst_evaluation()
                return

            await asyncio.sleep(10)

    async def _do_btst_evaluation(self) -> None:
        """Perform BTST evaluation and conversion."""
        if not self._orchestrator:
            return

        logger.info("BTST evaluation triggered at %s", datetime.now().strftime("%H:%M:%S"))

        try:
            btst_positions = await evaluate_and_convert_btst(
                self._orchestrator,
                self.broker,
                self.order_manager,
                self.position_tracker,
            )
            if btst_positions:
                logger.info("BTST: %d positions held overnight", len(btst_positions))
        except Exception as e:
            logger.error("BTST evaluation error: %s", e, exc_info=True)

    async def _reconciliation_loop(self) -> None:
        """Periodically reconcile internal state with broker positions.

        On first run (30s after start) and every 2 minutes thereafter:
        - Compare internal positions with broker
        - Auto-adopt orphaned broker positions (e.g. from crashed engine)
        - Auto-close positions the broker squared (MIS 3:20 timer, margin call)
          so the internal tracker and daily report stay accurate.
        """
        # First reconciliation after 30s (catch orphaned positions early)
        await asyncio.sleep(30)
        while not self._shutdown:
            try:
                if self.broker and self._orchestrator and self._orchestrator._running:
                    broker_positions = self.broker.get_positions()
                    report = self.position_tracker.reconcile_with_broker(broker_positions)
                    if not report["is_synced"]:
                        logger.warning("Reconciliation mismatch detected!")
                        # Auto-adopt positions the broker has but we don't
                        if report.get("missing_in_internal"):
                            adopted = self.position_tracker.adopt_broker_positions(broker_positions)
                            if adopted:
                                logger.warning("Auto-adopted %d positions: %s", len(adopted), adopted)
                                self.dashboard_writer.add_log(
                                    f"AUTO-ADOPTED {len(adopted)} broker positions: {', '.join(adopted)}"
                                )
                        # Auto-close positions we still think are open but broker
                        # has already squared (Zerodha MIS auto-squareoff 15:15-15:25,
                        # margin calls, manual squareoff via Kite app, etc.)
                        if report.get("missing_in_broker"):
                            closed = self._close_broker_squared_positions(
                                report["missing_in_broker"]
                            )
                            if closed:
                                logger.warning(
                                    "Auto-recorded %d broker-squared exits: %s",
                                    len(closed), closed,
                                )
            except Exception as e:
                logger.debug("Reconciliation error: %s", e)
            await asyncio.sleep(120)  # Every 2 minutes

    def _close_broker_squared_positions(self, symbols: list[str]) -> list[str]:
        """Record exits for positions the broker squared without engine action.

        When Zerodha's MIS auto-squareoff timer (15:15-15:25), a margin call,
        or a manual Kite app squareoff fires, our internal tracker never sees
        the SELL order. This causes the daily report to under-report both
        trade count and daily P&L. This method walks the broker's order
        history, finds the matching COMPLETE SELL/BUY order, and records a
        close with the real fill price. LTP is used as a fallback when no
        matching order is found.

        Returns list of symbols successfully closed.
        """
        if not self.broker or not self.position_tracker:
            return []

        # Fetch today's orders once, tolerate failure
        try:
            orders = self.broker.get_orders() or []
        except Exception as e:
            logger.error(
                "Failed to fetch orders for auto-squareoff reconciliation: %s", e
            )
            orders = []

        closed: list[str] = []
        for sym in symbols:
            internal_pos = next(
                (
                    p for p in self.position_tracker.open_positions
                    if p.get("symbol") == sym
                ),
                None,
            )
            if not internal_pos:
                continue

            try:
                entry_price = float(internal_pos.get("entry_price", 0) or 0)
                qty = int(internal_pos.get("qty", 0) or 0)
            except (TypeError, ValueError):
                logger.error(
                    "Bad internal position data for %s — skipping auto-close",
                    sym,
                )
                continue

            side = internal_pos.get("side", "BUY")
            exit_side = "SELL" if side == "BUY" else "BUY"

            # Find the most recent COMPLETE exit order matching this symbol
            exit_price = 0.0
            exit_source = "unknown"
            try:
                matching = [
                    o for o in orders
                    if o.get("symbol") == sym
                    and o.get("side") == exit_side
                    and o.get("status") == "COMPLETE"
                    and float(o.get("fill_price", 0) or 0) > 0
                ]
            except Exception:
                matching = []

            if matching:
                last_order = matching[-1]
                try:
                    exit_price = float(last_order.get("fill_price", 0) or 0)
                except (TypeError, ValueError):
                    exit_price = 0.0
                exit_source = f"order_id={last_order.get('order_id', '?')}"

            # Fall back to LTP if no order history matched
            if exit_price <= 0:
                try:
                    ltp_map = self.broker.get_ltp([sym]) or {}
                    exit_price = float(ltp_map.get(sym, 0) or 0)
                    exit_source = "ltp_fallback"
                except Exception as e:
                    logger.error("LTP fallback failed for %s: %s", sym, e)
                    exit_price = 0.0

            if exit_price <= 0:
                logger.error(
                    "Cannot determine exit price for %s — leaving as orphan "
                    "in internal tracker (manual reconciliation required)",
                    sym,
                )
                continue

            # Compute realized P&L (BUY profits when exit > entry)
            if side == "BUY":
                realized_pnl = (exit_price - entry_price) * qty
            else:
                realized_pnl = (entry_price - exit_price) * qty

            closed_pos = self.position_tracker.close_position(
                symbol=sym,
                exit_price=exit_price,
                realized_pnl=realized_pnl,
                exit_reason="broker_auto_squareoff",
            )
            if closed_pos:
                logger.warning(
                    "BROKER AUTO-SQUAREOFF RECORDED | %s | entry=%.2f exit=%.2f "
                    "pnl=%+.2f qty=%d | source=%s",
                    sym, entry_price, exit_price, realized_pnl, qty, exit_source,
                )
                closed.append(sym)
                try:
                    self.dashboard_writer.add_log(
                        f"AUTO-SQUAREOFF RECORDED: {sym} @ {exit_price:.2f} "
                        f"pnl={realized_pnl:+.0f}"
                    )
                except Exception:
                    pass

        return closed

    async def _manual_order_loop(self) -> None:
        """Poll for manual order requests from the dashboard UI."""
        from dashboard.data_bridge import MANUAL_ORDER_FILE

        while not self._shutdown:
            try:
                if MANUAL_ORDER_FILE.exists() and self.broker:
                    with open(MANUAL_ORDER_FILE, "r") as f:
                        request = json.load(f)

                    if request.get("status") == "PENDING":
                        symbol = request["symbol"]
                        side = request["side"]
                        qty = request["qty"]

                        # Mark as executing
                        request["status"] = "EXECUTING"
                        self._write_json_atomic(MANUAL_ORDER_FILE, request)

                        logger.info("MANUAL ORDER | %s %s qty=%d", side, symbol, qty)
                        self.dashboard_writer.add_log(f"MANUAL: {side} {symbol} x{qty}")

                        try:
                            result = self.broker.place_order(
                                symbol=symbol,
                                side=side,
                                qty=qty,
                                order_type=request.get("order_type", "MARKET"),
                                product=request.get("product", "MIS"),
                                tag="manual",
                            )

                            request["executed_at"] = datetime.now().isoformat()
                            order_id = result.get("order_id", "")
                            request["order_id"] = order_id

                            if order_id:
                                request["status"] = "PLACED"
                                logger.info("MANUAL ORDER PLACED | %s | order_id=%s", symbol, order_id)
                                self.dashboard_writer.add_log(
                                    f"MANUAL PLACED: {side} {symbol} x{qty} | order_id={order_id}"
                                )
                            else:
                                request["status"] = "FAILED"
                                request["error"] = result.get("error", "No order_id returned")
                                logger.error("MANUAL ORDER FAILED | %s | %s", symbol, request["error"])
                        except Exception as e:
                            request["status"] = "FAILED"
                            request["error"] = str(e)
                            request["executed_at"] = datetime.now().isoformat()
                            logger.error("MANUAL ORDER ERROR | %s | %s", symbol, e)

                        self._write_json_atomic(MANUAL_ORDER_FILE, request)

            except Exception as e:
                logger.debug("Manual order poll error: %s", e)

            await asyncio.sleep(1)

    def _write_json_atomic(self, path: Path, data: dict) -> None:
        """Atomic JSON write (write tmp, then replace)."""
        import json as _json
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            _json.dump(data, f, default=str)
        tmp.replace(path)

    async def _dashboard_state_loop(self) -> None:
        """Periodically write orchestrator state for the Streamlit dashboard."""
        while not self._shutdown:
            try:
                # Fetch real capital from Zerodha every 30s (avoid API rate limits)
                now = time.monotonic()
                if not self.paper and now - self._last_capital_fetch > 30:
                    try:
                        portfolio = self.broker.get_portfolio()
                        margins = portfolio.get("margins", {})
                        equity = margins.get("equity", {})
                        available = equity.get("available", {})
                        net_equity = equity.get("net", 0)
                        live_balance = available.get("live_balance", 0)
                        display_capital = net_equity if net_equity > 0 else live_balance
                        trading_capital = live_balance if live_balance > 0 else display_capital
                        if display_capital > 0:
                            self._zerodha_capital = display_capital  # For dashboard display
                            # Propagate TRADING capital (available) to agents for lot sizing
                            orch = self._orchestrator
                            if orch:
                                orch.capital = trading_capital
                                orch.risk_manager.total_capital = display_capital  # Risk uses total equity
                                for agent in orch.agents.values():
                                    if hasattr(agent, 'capital'):
                                        agent.capital = trading_capital
                                    if hasattr(agent, '_running_equity'):
                                        agent._running_equity = max(agent._running_equity, trading_capital)
                            logger.debug("Zerodha balance updated: Rs %.2f", live_balance)
                    except Exception as e:
                        logger.debug("Capital fetch error: %s", e)
                    self._last_capital_fetch = now

                # Subscribe open position symbols to KiteTicker for real-time LTP
                if not self.paper and self._cached_broker_positions:
                    try:
                        pos_symbols = [
                            bp["symbol"] for bp in self._cached_broker_positions
                            if bp.get("qty", 0) != 0 and bp["symbol"] not in self._position_tokens_subscribed
                        ]
                        if pos_symbols and hasattr(self.broker, 'add_tick_subscription'):
                            self.broker.add_tick_subscription(pos_symbols)
                            self._position_tokens_subscribed.update(pos_symbols)
                    except Exception as e:
                        logger.debug("Position tick subscription error: %s", e)

                self._write_dashboard_state("TRADING")
            except Exception as e:
                logger.debug("Dashboard state write error: %s", e)
            await asyncio.sleep(0.2)  # Update every 200ms for responsive dashboard

    def _write_dashboard_state(self, status: str = "TRADING") -> None:
        """Snapshot current state and write to dashboard JSON."""
        orch = self._orchestrator
        if not orch:
            return

        # ── Market data ──
        # Use current bar's close (real-time tick price) if available,
        # otherwise fall back to last completed bar
        spot = 0.0
        bars_list = []
        if orch._current_bar:
            spot = orch._current_bar.get("close", 0)
        elif orch._bar_buffer:
            spot = orch._bar_buffer[-1].get("close", 0)

        # Always send bars for chart display
        for b in orch._bar_buffer[-375:]:
            bars_list.append({
                "time": b.get("time", b.get("timestamp", "")),
                "open": round(b.get("open", 0), 2),
                "high": round(b.get("high", 0), 2),
                "low": round(b.get("low", 0), 2),
                "close": round(b.get("close", 0), 2),
                "volume": b.get("volume", 0),
            })

        # ── Analysis ──
        analysis = orch._latest_analysis
        bias = ""
        conf = 0.0
        support = 0.0
        resistance = 0.0
        if analysis:
            bias = analysis.market_bias.value
            conf = analysis.confidence
            support = analysis.oi_support
            resistance = analysis.oi_resistance

        # ── Positions & P&L (from Kite as source of truth) ──
        # Fetch ALL broker positions every 5s (API call), but use real-time
        # tick LTP from KiteTicker WebSocket for sub-second PnL updates.
        open_pos_list = []
        closed_pos_list = []
        realized = 0.0
        unrealized = 0.0

        # Real-time tick LTPs from KiteTicker (updated every tick, sub-second)
        tick_ltps = getattr(orch, '_tick_ltp', {})

        if not self.paper:
            try:
                now_mono = time.monotonic()
                if now_mono - self._last_positions_fetch >= 5.0:
                    self._cached_broker_positions = self.broker.get_positions()
                    self._last_positions_fetch = now_mono
                broker_positions = self._cached_broker_positions
                for bp in broker_positions:
                    sym = bp.get("symbol", "")
                    bp_qty = bp.get("qty", 0)
                    bp_avg = bp.get("average_price", 0)
                    bp_product = bp.get("product", "MIS")

                    # Use real-time tick LTP if available, else fall back to API LTP
                    bp_ltp = tick_ltps.get(sym, bp.get("ltp", 0))

                    # Use Kite's day-position P&L fields (accurate intraday)
                    bp_realised = bp.get("realised", 0.0)
                    bp_unrealised = bp.get("unrealised", 0.0)
                    bp_m2m = bp.get("m2m", 0.0)

                    if bp_qty != 0:
                        # Open position — use Kite's unrealised P&L + any
                        # partial realised from same symbol (e.g. scaled out)
                        side = "BUY" if bp_qty > 0 else "SELL"
                        abs_qty = abs(bp_qty)
                        # Recalculate with real-time LTP for sub-second updates
                        bp_pnl = (bp_ltp - bp_avg) * bp_qty
                        pnl_pct = (bp_pnl / (bp_avg * abs_qty) * 100) if (bp_avg * abs_qty) > 0 else 0

                        # Try to find entry time from our engine's tracked trades
                        entry_time = ""
                        strategy = "manual"
                        entry_type = "manual"
                        underlying_at = 0.0
                        for t in orch._open_trades:
                            tsig = t.signal
                            if tsig.legs and tsig.legs[0].symbol == sym:
                                entry_time = t.timestamp.strftime("%H:%M:%S") if t.timestamp else ""
                                strategy = tsig.strategy
                                entry_type = tsig.metadata.get("entry_type", tsig.action)
                                underlying_at = tsig.underlying_price
                                break

                        open_pos_list.append({
                            "symbol": sym,
                            "side": side,
                            "qty": abs_qty,
                            "entry_price": round(bp_avg, 2),
                            "current_price": round(bp_ltp, 2),
                            "pnl": round(bp_pnl, 2),
                            "pnl_pct": round(pnl_pct, 2),
                            "entry_time": entry_time,
                            "hold_minutes": 0,
                            "product": bp_product,
                            "strategy": strategy,
                            "entry_type": entry_type,
                            "underlying_at_entry": round(underlying_at, 2),
                        })
                        unrealized += bp_pnl
                        # Add any partial realised for this symbol
                        realized += bp_realised
                    else:
                        # Closed position (qty=0) — use Kite's realised P&L
                        # Prefer m2m (mark-to-market, most accurate for day),
                        # then realised, then fall back to net pnl
                        bp_pnl = bp_m2m if bp_m2m != 0 else (
                            bp_realised if bp_realised != 0 else bp.get("pnl", 0)
                        )
                        if bp_pnl != 0 or bp.get("day_buy_qty", 0) > 0 or bp.get("day_sell_qty", 0) > 0:
                            closed_pos_list.append({
                                "symbol": sym,
                                "side": "",
                                "qty": 0,
                                "entry_price": 0,
                                "exit_price": 0,
                                "pnl": round(bp_pnl, 2),
                                "entry_time": "",
                                "exit_time": "",
                                "hold_minutes": 0,
                                "exit_reason": "closed",
                                "entry_type": "",
                            })
                            realized += bp_pnl
            except Exception as e:
                logger.debug("Broker position fetch error: %s", e)
                # Fallback to internal tracking
                realized = self.position_tracker._daily_pnl
                unrealized = 0.0

        else:
            # Paper mode — use internal tracking
            for t in orch._open_trades:
                sig = t.signal
                entry_price = t.fill_prices[0] if t.fill_prices else 0
                leg_sym = sig.legs[0].symbol if sig.legs else ""
                qty = sig.legs[0].qty if sig.legs else 0
                side = sig.legs[0].side if sig.legs else "BUY"
                open_pos_list.append({
                    "symbol": leg_sym, "side": side, "qty": qty,
                    "entry_price": round(entry_price, 2), "current_price": round(entry_price, 2),
                    "pnl": 0, "pnl_pct": 0,
                    "entry_time": t.timestamp.strftime("%H:%M:%S") if t.timestamp else "",
                    "hold_minutes": 0, "product": "MIS",
                    "strategy": sig.strategy, "entry_type": sig.action,
                    "underlying_at_entry": round(sig.underlying_price, 2),
                })
            realized = self.position_tracker._daily_pnl

        # ── Estimate charges (to match Kite app P&L) ──
        estimated_charges = 0.0
        if not self.paper and self._cached_broker_positions:
            # Count filled orders for brokerage calculation
            num_filled_orders = 0
            if self.order_manager:
                num_filled_orders = sum(
                    1 for o in self.order_manager.get_today_orders()
                    if o.get("status") in ("COMPLETE", "FILLED", "EXECUTED", "TRADED")
                )
            estimated_charges = estimate_charges_from_positions(
                self._cached_broker_positions, num_filled_orders
            )

        total_pnl_gross = realized + unrealized  # Raw P&L (what API gives)
        total_pnl = total_pnl_gross - estimated_charges  # Net P&L (what Kite app shows)
        self.dashboard_writer.update_pnl_curve(total_pnl)

        # ── Hourly Telegram heartbeat ──
        tg.maybe_heartbeat(
            spot=spot,
            vix=getattr(orch, '_live_vix', 0) or 0,
            bias=getattr(orch, '_last_market_bias', 'neutral'),
            realized=realized,
            open_positions=len(open_pos_list),
        )

        # ── Sync risk manager with broker P&L (catches manual trades) ──
        if not self.paper and orch.risk_manager:
            # Override risk manager's realized P&L with broker's actual
            broker_day_pnl = realized + unrealized
            if broker_day_pnl != orch.risk_manager._realised_pnl:
                orch.risk_manager._realised_pnl = realized
                orch.risk_manager._unrealised_pnl = unrealized
            # Continuous kill switch check using broker P&L
            if not orch.risk_manager.kill_switch_active and orch.risk_manager.check_kill_switch():
                orch.risk_manager.kill_switch_active = True
                logger.critical(
                    "KILL SWITCH TRIGGERED — broker day loss %.2f (%.1f%%) exceeds limit",
                    broker_day_pnl,
                    broker_day_pnl / orch.risk_manager.total_capital * 100,
                )
                self.dashboard_writer.add_log(
                    f"KILL SWITCH: Day loss Rs {broker_day_pnl:,.0f} exceeds "
                    f"{orch.risk_manager.max_daily_loss_pct*100:.0f}% limit — NO NEW TRADES"
                )

        # ── Risk ──
        kill_pct = 0.08 if self.paper else 0.03
        kill_triggered = orch.risk_manager.kill_switch_active if orch.risk_manager else False
        # Use broker-derived counts for winners/losers
        winners = sum(1 for p in closed_pos_list if p.get("pnl", 0) > 0)
        losers = sum(1 for p in closed_pos_list if p.get("pnl", 0) <= 0)

        # ── Orders ──
        order_list = []
        if self.order_manager:
            for o in self.order_manager.get_today_orders():
                order_list.append({
                    "time": o.get("placed_at", ""),
                    "order_id": o.get("order_id", ""),
                    "symbol": o.get("symbol", ""),
                    "side": o.get("side", ""),
                    "qty": o.get("qty", 0),
                    "status": o.get("status", ""),
                    "fill_price": round(o.get("fill_price", 0), 2),
                    "tag": o.get("tag", ""),
                })

        # ── BTST ──
        btst_list = self.position_tracker._btst_positions or []

        # ── Agent info ──
        agent_name = ""
        for name in orch.agents:
            agent_name = name
            break

        # Use Zerodha account balance (fetched every 30s) or fallback to config
        live_capital = getattr(self, '_zerodha_capital', self.settings.trading.capital)

        # ── Live Indices (fetch every 10s) ──
        indices_list = getattr(self, '_cached_indices', [])
        now_mono = time.monotonic()
        last_idx_fetch = getattr(self, '_last_indices_fetch', 0.0)
        if not self.paper and (now_mono - last_idx_fetch) >= 10.0:
            try:
                idx_symbols = [
                    "NSE:NIFTY 50", "NSE:NIFTY BANK", "NSE:NIFTY FIN SERVICE",
                    "BSE:SENSEX", "NSE:NIFTY IT", "NSE:NIFTY MIDCAP 50",
                    "NSE:INDIA VIX",
                ]
                idx_data = self.broker._kite.ohlc(idx_symbols)
                indices_list = []
                for sym, data in idx_data.items():
                    ltp = data.get("last_price", 0)
                    ohlc = data.get("ohlc", {})
                    prev = ohlc.get("close", 0)
                    change = ltp - prev if prev > 0 else 0
                    change_pct = (change / prev * 100) if prev > 0 else 0
                    # Short name
                    short = sym.split(":")[-1]
                    if short == "NIFTY 50":
                        short = "NIFTY"
                    elif short == "NIFTY BANK":
                        short = "BANKNIFTY"
                    elif short == "NIFTY FIN SERVICE":
                        short = "FINNIFTY"
                    elif short == "NIFTY MIDCAP 50":
                        short = "MIDCAP"
                    elif short == "INDIA VIX":
                        short = "VIX"
                    elif short == "NIFTY IT":
                        short = "NIFTYIT"
                    indices_list.append({
                        "name": short,
                        "ltp": round(ltp, 2),
                        "change": round(change, 2),
                        "change_pct": round(change_pct, 2),
                    })
                self._cached_indices = indices_list
                self._last_indices_fetch = now_mono
            except Exception as e:
                logger.debug("Indices fetch failed: %s", e)

        # ── V14 Decision State (for dashboard) ──
        decision_state = {}
        for a_name, agent in orch.agents.items():
            if hasattr(agent, "get_decision_state"):
                try:
                    decision_state = agent.get_decision_state(
                        current_spot=spot, vix=orch._live_vix
                    )
                except Exception:
                    pass
            break

        # ── Option Chain for dashboard (enriched with live data) ──
        oc_display = []
        if orch._option_chain and spot > 0:
            try:
                from backtesting.option_pricer import bs_delta, skewed_iv
                atm_strike = round(spot / 50) * 50
                atm_iv_dec = orch._live_vix / 100.0 * 0.88 if orch._live_vix > 0 else 0.14
                # Estimate DTE
                from datetime import date as _date, timedelta as _td
                today = _date.today()
                config_idx = INDEX_CONFIG.get(self.symbol, {})
                exp_day_name = config_idx.get("weekly_expiry_day", "Thursday")
                day_map = {"Monday":0,"Tuesday":1,"Wednesday":2,"Thursday":3,"Friday":4}
                target_wd = day_map.get(exp_day_name, 3)
                dte_days = (target_wd - today.weekday()) % 7
                if dte_days == 0:
                    from datetime import datetime as _dt
                    hrs_left = max(0.5, (15.5 - _dt.now().hour - _dt.now().minute / 60.0))
                    dte_days = hrs_left / 6.25
                dte_days = max(0.1, float(dte_days))
                T = dte_days / 365.0
                r = 0.07

                sorted_strikes = sorted(orch._option_chain.keys())
                for strike in sorted_strikes:
                    if abs(strike - atm_strike) > 600:
                        continue
                    data = orch._option_chain[strike]
                    ce = data.get("CE", {})
                    pe = data.get("PE", {})
                    # Compute delta if not from broker
                    ce_iv = skewed_iv(atm_iv_dec, spot, strike, "CE")
                    pe_iv = skewed_iv(atm_iv_dec, spot, strike, "PE")
                    ce_delta = bs_delta(spot, strike, T, r, ce.get("iv", ce_iv * 100) / 100 if ce.get("iv", 0) > 0 else ce_iv, "CE")
                    pe_delta = bs_delta(spot, strike, T, r, pe.get("iv", pe_iv * 100) / 100 if pe.get("iv", 0) > 0 else pe_iv, "PE")

                    oc_display.append({
                        "strike": int(strike),
                        "is_atm": abs(strike - atm_strike) < 1,
                        "ce_ltp": round(float(ce.get("ltp", 0)), 1),
                        "ce_oi": int(ce.get("oi", 0)),
                        "ce_volume": int(ce.get("volume", 0)),
                        "ce_iv": round(float(ce.get("iv", ce_iv * 100)), 1),
                        "ce_delta": round(ce_delta, 2),
                        "ce_change": 0.0,
                        "pe_ltp": round(float(pe.get("ltp", 0)), 1),
                        "pe_oi": int(pe.get("oi", 0)),
                        "pe_volume": int(pe.get("volume", 0)),
                        "pe_iv": round(float(pe.get("iv", pe_iv * 100)), 1),
                        "pe_delta": round(pe_delta, 2),
                        "pe_change": 0.0,
                    })
            except Exception as e:
                logger.debug("Option chain display build error: %s", e)

        # ── Claude AI Market Brain (runs every 5 min) ──
        if self.claude_brain.should_analyze():
            try:
                # Get V14 agent indicators if available
                v14_indicators = {}
                v14_confluence = ""
                for a_name, agent in orch.agents.items():
                    if hasattr(agent, "_compute_indicators"):
                        v14_indicators = agent._compute_indicators() or {}
                    if hasattr(agent, "_bar_history"):
                        v14_confluence = self._last_signal or ""
                    break

                claude_result = self.claude_brain.analyze_market(
                    spot_price=spot,
                    prev_close=self._prev_close,
                    bars=bars_list,
                    vix=orch._live_vix,
                    pcr=orch._live_pcr,
                    support=support,
                    resistance=resistance,
                    is_expiry_day=orch._is_expiry_day,
                    open_positions=open_pos_list,
                    closed_positions=closed_pos_list,
                    realized_pnl=realized,
                    unrealized_pnl=unrealized,
                    capital=live_capital,
                    indicators=v14_indicators,
                    v14_last_signal=self._last_signal,
                    v14_confluence_status=v14_confluence,
                )
                one_liner = claude_result.get("one_liner", "")
                if one_liner:
                    self.dashboard_writer.add_log(f"AI [{self.claude_brain.provider.upper()}]: {one_liner}")
                    self._claude_last_status = one_liner

                # ── Pass AI analysis to V14 agent for decision influence ──
                for a_name, agent in orch.agents.items():
                    if hasattr(agent, "set_ai_brain_state"):
                        agent.set_ai_brain_state(claude_result)
                        logger.debug(
                            "AI brain state pushed to agent %s | action=%s conviction=%s risk=%s",
                            a_name,
                            claude_result.get("recommended_action", "?"),
                            claude_result.get("conviction", "?"),
                            claude_result.get("risk_assessment", "?"),
                        )
            except Exception as e:
                logger.debug("AI brain error: %s", e)

        # ── AI Brain state for dashboard ──
        ai_brain_state = {}
        try:
            brain_file = Path("data") / "claude_brain.json"
            if brain_file.exists():
                ai_brain_state = json.loads(brain_file.read_text(encoding="utf-8"))
        except Exception:
            pass

        self.dashboard_writer.write_state(
            system_status=status,
            mode="PAPER" if self.paper else "LIVE",
            symbol=self.symbol,
            capital=live_capital,
            bars_processed=orch._bar_count,
            total_bars=375,
            spot_price=spot,
            prev_close=self._prev_close,
            vix=orch._live_vix,
            pcr=orch._live_pcr,
            market_bias=bias,
            confidence=conf,
            support=support,
            resistance=resistance,
            is_expiry_day=orch._is_expiry_day,
            bars=bars_list,
            open_positions=open_pos_list,
            closed_positions=closed_pos_list,
            btst_positions=btst_list,
            realized_pnl=realized,
            unrealized_pnl=unrealized,
            estimated_charges=estimated_charges,
            kill_switch_pct=kill_pct,
            kill_switch_triggered=kill_triggered,
            max_positions=self.settings.trading.max_open_positions,
            trades_today=len(orch._open_trades) + len(orch._closed_trades),
            max_trades_per_day=5,
            winners=winners,
            losers=losers,
            orders=order_list,
            agent_name=agent_name,
            signals_generated=self._signals_generated,
            signals_accepted=self._signals_accepted,
            signals_filtered=self._signals_filtered,
            last_signal=self._last_signal,
            started_at=self._session_started_at,
            indices=indices_list,
            decision_state=decision_state,
            ai_brain=ai_brain_state,
            option_chain_display=oc_display,
        )

    async def _sleep_for(self, seconds: float) -> None:
        """Sleep in small chunks so we can respond to shutdown."""
        end = time.monotonic() + seconds
        while time.monotonic() < end and not self._shutdown:
            remaining = end - time.monotonic()
            await asyncio.sleep(min(remaining, 5.0))

    async def _sleep_until_datetime(self, target: datetime) -> None:
        """Sleep until a target datetime."""
        while datetime.now() < target and not self._shutdown:
            remaining = (target - datetime.now()).total_seconds()
            await asyncio.sleep(min(remaining, 30.0))

    def shutdown(self) -> None:
        """Signal graceful shutdown."""
        logger.info("Shutdown signal received")
        self._shutdown = True
        if self._orchestrator:
            asyncio.ensure_future(self._orchestrator.shutdown())


# ── Entry Point ────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Fully Autonomous Trading System"
    )
    parser.add_argument(
        "--paper", action="store_true",
        help="Run in paper trading mode (no real money)",
    )
    parser.add_argument(
        "--paper-live-data", action="store_true",
        help=(
            "Hybrid mode: paper execution + REAL Kite market data. "
            "Authenticates Kite Connect for ticks/bars/VIX/option chain "
            "but all orders flow through the simulated paper broker. "
            "Use for Monday pre-deployment validation of the live data "
            "infrastructure without placing real orders."
        ),
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="Fast-forward paper mode (50x speed)",
    )
    parser.add_argument(
        "--symbol", default="NIFTY",
        choices=["NIFTY", "BANKNIFTY", "FINNIFTY"],
        help="Primary index to trade (default: NIFTY)",
    )
    parser.add_argument(
        "--multi-index", nargs="*", default=None,
        choices=["NIFTY", "BANKNIFTY", "FINNIFTY"],
        help="Trade multiple indices simultaneously (from research: FINNIFTY diversification for better Sharpe)",
    )
    parser.add_argument(
        "--capital", type=float, default=None,
        help="Override trading capital",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    setup_logging(args.log_level)

    # ── Single instance check ──
    if not _acquire_engine_lock():
        # Try to read who holds the lock
        try:
            pid_text = _ENGINE_LOCK_FILE.read_text().strip()
            logger.critical(
                "ANOTHER TRADING ENGINE IS ALREADY RUNNING (PID %s). "
                "Cannot start a second instance — duplicate orders would occur. "
                "Kill the existing process first or use deploy/stop_trading.bat",
                pid_text,
            )
        except Exception:
            logger.critical(
                "ANOTHER TRADING ENGINE IS ALREADY RUNNING. "
                "Cannot start a second instance. Use deploy/stop_trading.bat to stop it."
            )
        sys.exit(1)

    logger.info("Engine lock acquired (PID %d) — single instance guaranteed", os.getpid())

    # ── Multi-index support (from research: FINNIFTY diversification) ──
    symbols = args.multi_index if args.multi_index else [args.symbol]

    if len(symbols) > 1:
        logger.info("Multi-index mode: trading %s simultaneously", symbols)
        # Split capital equally across indices
        per_index_capital = (args.capital or 30000) / len(symbols)
        logger.info("Capital per index: %.0f", per_index_capital)

        traders = []
        for sym in symbols:
            t = AutonomousTrader(
                paper=args.paper,
                fast=args.fast,
                symbol=sym,
                paper_live_data=args.paper_live_data,
            )
            t.settings.trading.capital = per_index_capital
            traders.append(t)

        # Handle SIGINT/SIGTERM gracefully
        def signal_handler(sig, frame):
            logger.info("Signal %s received — initiating graceful shutdown for %d indices", sig, len(traders))
            for t in traders:
                t.shutdown()

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Run all traders concurrently
        async def run_multi():
            tasks = [asyncio.create_task(t.run_forever()) for t in traders]
            await asyncio.gather(*tasks, return_exceptions=True)

        try:
            asyncio.run(run_multi())
        except KeyboardInterrupt:
            logger.info("Interrupted — shutting down all indices")
            for t in traders:
                t.shutdown()
    else:
        trader = AutonomousTrader(
            paper=args.paper,
            fast=args.fast,
            symbol=symbols[0],
            paper_live_data=args.paper_live_data,
        )

        if args.capital:
            trader.settings.trading.capital = args.capital

        # Handle SIGINT/SIGTERM gracefully
        def signal_handler(sig, frame):
            logger.info("Signal %s received — initiating graceful shutdown", sig)
            trader.shutdown()

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Run the event loop
        try:
            asyncio.run(trader.run_forever())
        except KeyboardInterrupt:
            logger.info("Interrupted — shutting down")
            trader.shutdown()


if __name__ == "__main__":
    main()
