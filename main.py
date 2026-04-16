"""
Indian Options Algorithmic Trading System — Main Entry Point

Orchestrates:
1. Broker authentication with daily TOTP 2FA
2. gRPC connection to Rust execution engine
3. DDQN strategy agent initialization
4. Risk management and kill switch monitoring
5. Live trading loop with WebSocket data feeds

Deployment: AWS EC2 M8g (Graviton4) in ap-south-1 (Mumbai)
Compliance: SEBI 2026 algo trading framework
"""

import argparse
import asyncio
import logging
import os
import signal
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# Local imports
from config.settings import AppSettings, load_settings
from config.constants import (
    MARKET_OPEN_HOUR,
    MARKET_OPEN_MINUTE,
    MARKET_CLOSE_HOUR,
    MARKET_CLOSE_MINUTE,
    STRADDLE_ENTRY_HOUR,
    STRADDLE_ENTRY_MINUTE,
    INDEX_CONFIG,
    GRPC_PORT,
    DEFAULT_MAX_DAILY_LOSS_PCT,
)
from strategy.ddqn_agent import DDQNAgent
from strategy.features import FeatureEngine
from strategy.environment import TradingEnvironment
from strategy.volatility import VolatilityEnsemble
from risk_management.risk_manager import RiskManager
from risk_management.order_slicer import OrderSlicer
from risk_management.slippage import SlippageModel
from broker.auth import TOTPAuthenticator
from backtesting.paper_trading import PaperTradingBroker


# ─── Logging Setup ──────────────────────────────────────────────────────────

def setup_logging(log_level: str = "INFO", log_file: str = "logs/trading.log"):
    """Configure structured logging with timestamps and latency metrics."""
    log_dir = Path(log_file).parent
    log_dir.mkdir(parents=True, exist_ok=True)

    fmt = (
        "%(asctime)s.%(msecs)03d | %(levelname)-8s | %(name)-25s | %(message)s"
    )
    datefmt = "%Y-%m-%d %H:%M:%S"

    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, mode="a"),
    ]

    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format=fmt,
        datefmt=datefmt,
        handlers=handlers,
    )

    # Suppress noisy third-party loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("websockets").setLevel(logging.WARNING)
    logging.getLogger("grpc").setLevel(logging.WARNING)


logger = logging.getLogger("main")


# ─── Broker Factory ─────────────────────────────────────────────────────────

def create_broker(settings: AppSettings):
    """Create the appropriate broker instance based on settings."""
    if settings.trading.paper_trading:
        logger.info("Paper trading mode — using simulated broker")
        return PaperTradingBroker(
            initial_capital=settings.trading.capital,
        )

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
        raise ValueError(f"Unsupported broker type: {broker_type}")


# ─── Trading Session ────────────────────────────────────────────────────────

class TradingSession:
    """
    Manages a single trading day session.

    Lifecycle:
    1. Authenticate with broker (TOTP 2FA)
    2. Initialize strategy components
    3. Subscribe to market data
    4. Run trading loop (9:15 AM - 3:30 PM IST)
    5. Square off and generate report
    """

    def __init__(self, settings: AppSettings):
        self.settings = settings
        self._shutdown = False

        # Core components
        self.broker = create_broker(settings)
        self.risk_manager = RiskManager(
            total_capital=settings.trading.capital,
            max_daily_loss_pct=settings.trading.max_daily_loss_pct,
        )
        self.order_slicer = OrderSlicer()
        self.slippage_model = SlippageModel()
        self.feature_engine = FeatureEngine()
        self.volatility_ensemble = VolatilityEnsemble()

        # DDQN Agent
        self.agent = DDQNAgent(
            state_dim=self.feature_engine.state_dim,
            action_dim=5,  # hold, buy_call, buy_put, sell_call, sell_put
        )

        # Try to load trained model
        model_path = settings.ml.model_path
        if Path(model_path).exists():
            self.agent.load(model_path)
            logger.info("DDQN model loaded from %s", model_path)
        else:
            logger.warning("No trained model found at %s — agent will use random policy", model_path)

        # State tracking
        self._ohlcv_buffer: list[dict] = []
        self._current_bar: dict = {}
        self._last_bar_time: datetime = datetime.min
        self._bar_interval = timedelta(minutes=settings.ml.bar_interval_minutes)

        logger.info(
            "TradingSession initialized | capital=%.2f index=%s paper=%s",
            settings.trading.capital,
            settings.trading.default_index,
            settings.trading.paper_trading,
        )

    def authenticate(self) -> bool:
        """Authenticate with broker using TOTP 2FA."""
        t_start = time.monotonic()
        try:
            result = self.broker.authenticate()
            latency = (time.monotonic() - t_start) * 1000
            if result:
                logger.info("Broker authentication successful | latency=%.1fms", latency)
            else:
                logger.error("Broker authentication failed | latency=%.1fms", latency)
            return result
        except Exception as e:
            logger.error("Broker authentication error: %s", e)
            return False

    def _on_tick(self, ticks: list[dict]):
        """Process incoming tick data."""
        for tick in ticks:
            symbol = tick.get("symbol", "")
            price = tick.get("last_price", 0.0)
            timestamp = tick.get("timestamp", datetime.now())

            if not symbol or price <= 0:
                continue

            # Update current prices in risk manager
            self.risk_manager.update_price(symbol, price)

            # Aggregate into OHLCV bars
            self._aggregate_bar(tick)

    def _aggregate_bar(self, tick: dict):
        """Aggregate ticks into OHLCV bars."""
        timestamp = tick.get("timestamp", datetime.now())
        if isinstance(timestamp, (int, float)):
            timestamp = datetime.fromtimestamp(timestamp / 1000)

        price = tick["last_price"]
        volume = tick.get("volume", 0)

        # Check if we need to start a new bar
        bar_start = timestamp.replace(
            minute=(timestamp.minute // self.settings.ml.bar_interval_minutes)
            * self.settings.ml.bar_interval_minutes,
            second=0,
            microsecond=0,
        )

        if bar_start != self._last_bar_time and self._current_bar:
            # Complete the previous bar and process it
            self._ohlcv_buffer.append(self._current_bar.copy())
            self._process_completed_bar()
            self._current_bar = {}

        if not self._current_bar:
            self._current_bar = {
                "timestamp": bar_start,
                "open": price,
                "high": price,
                "low": price,
                "close": price,
                "volume": volume,
            }
            self._last_bar_time = bar_start
        else:
            self._current_bar["high"] = max(self._current_bar["high"], price)
            self._current_bar["low"] = min(self._current_bar["low"], price)
            self._current_bar["close"] = price
            self._current_bar["volume"] += volume

    def _process_completed_bar(self):
        """Process a completed OHLCV bar through the strategy pipeline."""
        if len(self._ohlcv_buffer) < self.settings.ml.lookback_bars:
            logger.debug(
                "Buffering bars: %d/%d",
                len(self._ohlcv_buffer), self.settings.ml.lookback_bars,
            )
            return

        # Check kill switch
        if self.risk_manager.check_kill_switch():
            logger.warning("Kill switch active — skipping bar processing")
            return

        t_start = time.monotonic()

        # Build feature vector from recent bars
        bars_df = pd.DataFrame(self._ohlcv_buffer[-self.settings.ml.lookback_bars:])
        bars_df.set_index("timestamp", inplace=True)

        state = self.feature_engine.compute_state(bars_df)

        # Get action from DDQN agent (no exploration in live mode)
        action = self.agent.select_action(state, explore=False)

        # Map action to order
        self._execute_action(action, bars_df)

        latency = (time.monotonic() - t_start) * 1000
        logger.info(
            "Bar processed | action=%d buffer_size=%d latency=%.1fms",
            action, len(self._ohlcv_buffer), latency,
        )

    def _execute_action(self, action: int, bars_df: pd.DataFrame):
        """Execute the DDQN agent's action."""
        if action == 0:
            return  # Hold — do nothing

        index = self.settings.trading.default_index
        config = INDEX_CONFIG.get(index, {})
        lot_size = config.get("lot_size", 25)
        last_price = bars_df["close"].iloc[-1]

        # Determine strike and option type
        strike_interval = config.get("strike_interval", 50)
        atm_strike = round(last_price / strike_interval) * strike_interval

        action_map = {
            1: ("BUY", "CE", atm_strike),
            2: ("BUY", "PE", atm_strike),
            3: ("SELL", "CE", atm_strike),
            4: ("SELL", "PE", atm_strike),
        }

        side, opt_type, strike = action_map[action]

        # Check position limits
        if not self.risk_manager.check_position_limits(lot_size, last_price):
            logger.info("Position limit reached — order rejected")
            return

        # Determine order type based on moneyness
        delta = self.slippage_model.estimate_delta(strike, last_price, opt_type)
        order_type = self.slippage_model.select_order_type(delta)

        # Build symbol name
        symbol = f"{index}{strike}{opt_type}"

        # Slice if needed
        sub_orders = self.order_slicer.slice_order(
            index_symbol=index,
            total_qty=lot_size,
            symbol=symbol,
            side=side,
            order_type=order_type,
            price=last_price,
        )

        # Execute
        for sub in sub_orders:
            try:
                result = self.broker.place_order(**sub)
                logger.info(
                    "Order executed | symbol=%s side=%s qty=%d type=%s result=%s",
                    symbol, side, sub["quantity"], order_type, result.get("success"),
                )
            except Exception as e:
                logger.error("Order execution error: %s", e)

    async def run(self):
        """Run the trading session for the current day."""
        logger.info("═" * 60)
        logger.info("  TRADING SESSION STARTING")
        logger.info("  Date: %s", datetime.now().strftime("%Y-%m-%d"))
        logger.info("  Mode: %s", "PAPER" if self.settings.trading.paper_trading else "LIVE")
        logger.info("═" * 60)

        # Authenticate
        if not self.authenticate():
            logger.error("Authentication failed — aborting session")
            return

        # Reset daily state
        self.risk_manager.reset_daily()

        # Subscribe to market data
        index = self.settings.trading.default_index
        symbols = [INDEX_CONFIG[index]["underlying_symbol"]]
        logger.info("Subscribing to market data for: %s", symbols)

        try:
            self.broker.subscribe_ticks(symbols, self._on_tick)
        except Exception as e:
            logger.error("Failed to subscribe to market data: %s", e)
            return

        # Trading loop
        logger.info("Entering trading loop...")
        while not self._shutdown:
            now = datetime.now()

            # Check if market is closed
            if (now.hour > MARKET_CLOSE_HOUR or
                (now.hour == MARKET_CLOSE_HOUR and now.minute >= MARKET_CLOSE_MINUTE)):
                logger.info("Market closed — ending session")
                break

            # Check if before market open
            if (now.hour < MARKET_OPEN_HOUR or
                (now.hour == MARKET_OPEN_HOUR and now.minute < MARKET_OPEN_MINUTE)):
                await asyncio.sleep(1)
                continue

            # Check kill switch
            if self.risk_manager.check_kill_switch():
                logger.warning("Kill switch triggered — squaring off all positions")
                self._square_off_all()
                break

            # Update MTM
            positions = self.broker.get_positions()
            self.risk_manager.update_mtm(positions)

            await asyncio.sleep(0.1)  # 100ms loop

        # End of day cleanup
        self._end_of_day()

    def _square_off_all(self):
        """Emergency square-off of all positions."""
        logger.warning("EXECUTING SQUARE-OFF ALL POSITIONS")
        t_start = time.monotonic()

        positions = self.broker.get_positions()
        for pos in positions:
            if pos.get("quantity", 0) != 0:
                side = "SELL" if pos["quantity"] > 0 else "BUY"
                qty = abs(pos["quantity"])
                try:
                    self.broker.place_order(
                        symbol=pos["symbol"],
                        side=side,
                        quantity=qty,
                        order_type="MARKET",
                        tag="KILL_SWITCH",
                    )
                    logger.info(
                        "Kill switch: closed %s | qty=%d",
                        pos["symbol"], qty,
                    )
                except Exception as e:
                    logger.error(
                        "Kill switch: failed to close %s: %s",
                        pos["symbol"], e,
                    )

        latency = (time.monotonic() - t_start) * 1000
        logger.warning("Square-off complete | latency=%.1fms", latency)

    def _end_of_day(self):
        """End of day cleanup and reporting."""
        logger.info("═" * 60)
        logger.info("  END OF DAY REPORT")
        logger.info("═" * 60)

        # Close any remaining positions
        positions = self.broker.get_positions()
        open_count = sum(1 for p in positions if p.get("quantity", 0) != 0)
        if open_count > 0:
            logger.warning("%d positions still open at EOD — squaring off", open_count)
            self._square_off_all()

        # Print portfolio summary
        portfolio = self.broker.get_portfolio()
        logger.info("Portfolio: %s", portfolio)

        # Risk report
        risk_status = self.risk_manager.get_status()
        logger.info("Risk Status: %s", risk_status)

        # Close broker connection
        try:
            self.broker.close()
        except Exception:
            pass

        logger.info("Trading session ended")

    def shutdown(self):
        """Signal graceful shutdown."""
        logger.info("Shutdown signal received")
        self._shutdown = True


# ─── CLI & Main ──────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Indian Options Algorithmic Trading System",
    )
    parser.add_argument(
        "--grpc-host", default="127.0.0.1",
        help="gRPC server host (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--grpc-port", type=int, default=GRPC_PORT,
        help=f"gRPC server port (default: {GRPC_PORT})",
    )
    parser.add_argument(
        "--mode", choices=["live", "paper", "backtest", "train"],
        default="paper",
        help="Operating mode (default: paper)",
    )
    parser.add_argument(
        "--train-episodes", type=int, default=1000,
        help="Number of training episodes for DDQN (default: 1000)",
    )
    parser.add_argument(
        "--backtest-days", type=int, default=252,
        help="Number of days to backtest (default: 252)",
    )
    return parser.parse_args()


async def run_training(settings: AppSettings, episodes: int):
    """Train the DDQN agent on historical data."""
    from backtesting.data_loader import generate_synthetic_ohlcv

    logger.info("═" * 60)
    logger.info("  DDQN TRAINING MODE")
    logger.info("  Episodes: %d", episodes)
    logger.info("═" * 60)

    # Generate or load training data
    index = settings.trading.default_index
    logger.info("Generating synthetic training data for %s...", index)
    data = generate_synthetic_ohlcv(
        symbol=index,
        days=504,  # 2 years
        interval_minutes=settings.ml.bar_interval_minutes,
    )

    # Create environment and agent
    feature_engine = FeatureEngine()
    env = TradingEnvironment(
        data=data,
        feature_engine=feature_engine,
        initial_capital=settings.trading.capital,
        max_daily_loss_pct=settings.trading.max_daily_loss_pct,
    )

    agent = DDQNAgent(
        state_dim=feature_engine.state_dim,
        action_dim=5,
    )

    # Training loop
    rewards_history = []
    best_reward = float("-inf")

    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0.0
        done = False

        while not done:
            action = agent.select_action(state, explore=True)
            next_state, reward, done, info = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)
            loss = agent.train_step()
            state = next_state
            episode_reward += reward

        agent.decay_epsilon()
        rewards_history.append(episode_reward)

        if episode % 10 == 0:
            agent.update_target()

        if episode_reward > best_reward:
            best_reward = episode_reward
            agent.save(settings.ml.model_path)

        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(rewards_history[-50:])
            logger.info(
                "Episode %d/%d | reward=%.4f avg_50=%.4f best=%.4f epsilon=%.4f",
                episode + 1, episodes, episode_reward,
                avg_reward, best_reward, agent.epsilon,
            )

    logger.info("Training complete. Best reward: %.4f", best_reward)
    logger.info("Model saved to %s", settings.ml.model_path)


async def run_backtest(settings: AppSettings, days: int):
    """Run backtesting on historical data."""
    from backtesting.backtest_engine import BacktestEngine
    from backtesting.data_loader import generate_synthetic_ohlcv

    logger.info("═" * 60)
    logger.info("  BACKTEST MODE")
    logger.info("  Days: %d", days)
    logger.info("═" * 60)

    index = settings.trading.default_index
    data = generate_synthetic_ohlcv(
        symbol=index,
        days=days,
        interval_minutes=settings.ml.bar_interval_minutes,
    )

    feature_engine = FeatureEngine()
    agent = DDQNAgent(state_dim=feature_engine.state_dim, action_dim=5)

    model_path = settings.ml.model_path
    if Path(model_path).exists():
        agent.load(model_path)
        logger.info("Loaded trained model for backtesting")
    else:
        logger.warning("No trained model — backtesting with random policy")

    engine = BacktestEngine(
        initial_capital=settings.trading.capital,
        max_daily_loss_pct=settings.trading.max_daily_loss_pct,
    )
    slippage_model = SlippageModel()

    lookback = settings.ml.lookback_bars

    def strategy_fn(eng, bar_idx, row, spot_row):
        if bar_idx < lookback:
            return

        window = data.iloc[bar_idx - lookback:bar_idx]
        state = feature_engine.compute_state(window)
        action = agent.select_action(state, explore=False)

        if action == 0:
            return

        price = row["close"]
        side = "BUY" if action in (1, 2) else "SELL"
        opt_type = "CE" if action in (1, 3) else "PE"
        config = INDEX_CONFIG.get(index, {})
        lot_size = config.get("lot_size", 25)

        eng.open_position(
            timestamp=data.index[bar_idx],
            symbol=f"{index}_ATM_{opt_type}",
            side=side,
            quantity=lot_size,
            price=price,
            spot_price=price,
            option_type=opt_type,
            tag=f"DDQN_action_{action}",
        )

        # Close after 4 bars (~1 hour)
        for pos in eng.positions:
            if pos.is_open and bar_idx - data.index.get_loc(pos.entry_time) >= 4:
                eng.close_position(pos, data.index[bar_idx], price, price)

    engine.run_strategy(data, strategy_fn)
    result = engine.get_results()
    engine.print_report(result)


async def main():
    args = parse_args()
    settings = load_settings()

    # Override from CLI
    settings.infra.grpc_host = args.grpc_host
    settings.infra.grpc_port = args.grpc_port

    if args.mode == "live":
        settings.trading.paper_trading = False
    elif args.mode == "paper":
        settings.trading.paper_trading = True

    setup_logging(settings.infra.log_level, settings.infra.log_file)

    logger.info("═" * 60)
    logger.info("  INDIAN OPTIONS ALGORITHMIC TRADING SYSTEM")
    logger.info("  Mode: %s", args.mode.upper())
    logger.info("  Index: %s", settings.trading.default_index)
    logger.info("  Capital: ₹%.2f", settings.trading.capital)
    logger.info("═" * 60)

    if args.mode == "train":
        await run_training(settings, args.train_episodes)
    elif args.mode == "backtest":
        await run_backtest(settings, args.backtest_days)
    else:
        session = TradingSession(settings)

        # Handle graceful shutdown
        def signal_handler(sig, frame):
            session.shutdown()

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        await session.run()


if __name__ == "__main__":
    asyncio.run(main())
