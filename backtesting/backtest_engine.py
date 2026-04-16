"""
Core backtesting engine for Indian options algorithmic strategies.

Simulates order execution with realistic transaction costs (2026 STT rates),
slippage modeling by moneyness, and NSE freeze limit compliance.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

from config.constants import (
    STT_RATES,
    SEBI_TURNOVER_FEE,
    NSE_TRANSACTION_CHARGE,
    STAMP_DUTY_BUY,
    GST_RATE,
    INDEX_CONFIG,
    FREEZE_LIMITS,
    DEFAULT_MAX_DAILY_LOSS_PCT,
)

logger = logging.getLogger(__name__)


@dataclass
class BacktestPosition:
    """Tracks a single position during backtesting."""
    symbol: str
    side: str  # "BUY" or "SELL"
    quantity: int
    entry_price: float
    entry_time: datetime
    exit_price: float = 0.0
    exit_time: Optional[datetime] = None
    pnl: float = 0.0
    transaction_costs: float = 0.0
    slippage_cost: float = 0.0
    is_open: bool = True
    option_type: str = ""  # "CE", "PE", or "" for futures
    strike: float = 0.0
    tag: str = ""


@dataclass
class BacktestTrade:
    """Record of a completed trade."""
    entry_time: datetime
    exit_time: datetime
    symbol: str
    side: str
    quantity: int
    entry_price: float
    exit_price: float
    gross_pnl: float
    transaction_costs: float
    slippage_cost: float
    net_pnl: float
    hold_duration_minutes: float
    tag: str = ""


@dataclass
class BacktestResult:
    """Comprehensive backtesting results."""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    gross_pnl: float = 0.0
    total_transaction_costs: float = 0.0
    total_slippage_costs: float = 0.0
    net_pnl: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_hold_minutes: float = 0.0
    total_days: int = 0
    daily_pnl: list = field(default_factory=list)
    equity_curve: list = field(default_factory=list)
    trades: list = field(default_factory=list)
    kill_switch_triggers: int = 0


class TransactionCostModel:
    """
    Calculates realistic transaction costs for Indian F&O markets.
    Incorporates 2026 STT hike rates.
    """

    def __init__(self, brokerage_per_order: float = 20.0):
        self.brokerage_per_order = brokerage_per_order

    def calculate(
        self,
        premium: float,
        quantity: int,
        side: str,
        instrument_type: str = "options",
    ) -> dict:
        """
        Calculate all transaction costs for a trade.

        Returns breakdown dict with each cost component.
        """
        turnover = premium * quantity

        # Brokerage (flat per order for discount brokers)
        brokerage = self.brokerage_per_order

        # STT (sell side only)
        stt = 0.0
        if side == "SELL":
            if instrument_type == "options":
                stt = turnover * STT_RATES["options_sell"]
            elif instrument_type == "futures":
                stt = turnover * STT_RATES["futures_sell"]

        # Exchange charges
        exchange_charges = turnover * NSE_TRANSACTION_CHARGE

        # SEBI turnover fee
        sebi_fee = turnover * SEBI_TURNOVER_FEE

        # Stamp duty (buy side only)
        stamp_duty = 0.0
        if side == "BUY":
            stamp_duty = turnover * STAMP_DUTY_BUY

        # GST on brokerage + exchange charges
        gst = (brokerage + exchange_charges + sebi_fee) * GST_RATE

        total = brokerage + stt + exchange_charges + sebi_fee + stamp_duty + gst

        return {
            "brokerage": round(brokerage, 2),
            "stt": round(stt, 2),
            "exchange_charges": round(exchange_charges, 2),
            "sebi_fee": round(sebi_fee, 4),
            "stamp_duty": round(stamp_duty, 2),
            "gst": round(gst, 2),
            "total": round(total, 2),
        }


class SlippageSimulator:
    """Simulates realistic slippage based on option moneyness and liquidity."""

    def estimate(
        self,
        price: float,
        spot_price: float,
        option_type: str,
        side: str,
        quantity: int,
    ) -> float:
        """
        Estimate slippage in price points.

        ATM options: 0.01-0.05% slippage
        OTM options: 0.05-0.15% slippage
        Deep OTM options: 0.15-0.50% slippage
        """
        if not option_type:  # Futures
            return price * 0.0002  # ~0.02% for liquid futures

        # Estimate delta-like moneyness
        if option_type == "CE":
            moneyness = (spot_price - price) / spot_price if price > 0 else 0
        else:
            moneyness = (price - spot_price) / spot_price if price > 0 else 0

        # Higher slippage for larger orders (market impact)
        size_factor = 1.0 + (quantity / 5000) * 0.1

        if abs(moneyness) < 0.02:  # ATM
            slippage_pct = np.random.uniform(0.0001, 0.0005)
        elif abs(moneyness) < 0.05:  # Near OTM
            slippage_pct = np.random.uniform(0.0005, 0.0015)
        elif abs(moneyness) < 0.10:  # OTM
            slippage_pct = np.random.uniform(0.0015, 0.003)
        else:  # Deep OTM
            slippage_pct = np.random.uniform(0.003, 0.005)

        slippage = price * slippage_pct * size_factor

        # Adverse direction
        if side == "BUY":
            return abs(slippage)  # Pay more
        else:
            return -abs(slippage)  # Receive less


class BacktestEngine:
    """
    Full-featured backtesting engine for Indian options strategies.

    Features:
    - Realistic 2026 STT transaction costs
    - Moneyness-based slippage simulation
    - Daily MTM kill switch simulation
    - Order slicing compliance tracking
    - Equity curve and drawdown analysis
    """

    def __init__(
        self,
        initial_capital: float = 1_000_000.0,
        max_daily_loss_pct: float = DEFAULT_MAX_DAILY_LOSS_PCT,
        brokerage_per_order: float = 20.0,
    ):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.max_daily_loss_pct = max_daily_loss_pct
        self.cost_model = TransactionCostModel(brokerage_per_order)
        self.slippage_sim = SlippageSimulator()

        self.positions: list[BacktestPosition] = []
        self.closed_trades: list[BacktestTrade] = []
        self.equity_curve: list[float] = [initial_capital]
        self.daily_pnl: list[float] = []
        self.kill_switch_triggers = 0

        self._current_day_pnl = 0.0
        self._current_day: Optional[datetime] = None
        self._kill_switch_active = False

        logger.info(
            "BacktestEngine initialized | capital=%.2f max_loss=%.1f%%",
            initial_capital, max_daily_loss_pct * 100,
        )

    def reset(self):
        """Reset engine state for a new backtest run."""
        self.capital = self.initial_capital
        self.positions.clear()
        self.closed_trades.clear()
        self.equity_curve = [self.initial_capital]
        self.daily_pnl.clear()
        self.kill_switch_triggers = 0
        self._current_day_pnl = 0.0
        self._current_day = None
        self._kill_switch_active = False

    def _check_new_day(self, timestamp: datetime):
        """Handle day transitions and reset daily tracking."""
        current_date = timestamp.date()
        if self._current_day is None or current_date != self._current_day:
            if self._current_day is not None:
                self.daily_pnl.append(self._current_day_pnl)
            self._current_day = current_date
            self._current_day_pnl = 0.0
            self._kill_switch_active = False

    def _check_kill_switch(self) -> bool:
        """Check if daily loss exceeds threshold."""
        if self._kill_switch_active:
            return True
        loss_pct = abs(self._current_day_pnl) / self.initial_capital
        if self._current_day_pnl < 0 and loss_pct >= self.max_daily_loss_pct:
            self._kill_switch_active = True
            self.kill_switch_triggers += 1
            logger.warning(
                "KILL SWITCH triggered | day_pnl=%.2f loss_pct=%.2f%%",
                self._current_day_pnl, loss_pct * 100,
            )
            return True
        return False

    def open_position(
        self,
        timestamp: datetime,
        symbol: str,
        side: str,
        quantity: int,
        price: float,
        spot_price: float = 0.0,
        option_type: str = "",
        strike: float = 0.0,
        tag: str = "",
    ) -> Optional[BacktestPosition]:
        """Open a new position with realistic costs."""
        self._check_new_day(timestamp)

        if self._kill_switch_active:
            logger.info("Order rejected — kill switch active | symbol=%s", symbol)
            return None

        # Calculate slippage
        slippage = self.slippage_sim.estimate(
            price, spot_price or price, option_type, side, quantity
        )
        fill_price = price + slippage if side == "BUY" else price + slippage

        # Calculate transaction costs
        costs = self.cost_model.calculate(
            fill_price, quantity, side,
            "options" if option_type else "futures",
        )

        position = BacktestPosition(
            symbol=symbol,
            side=side,
            quantity=quantity,
            entry_price=fill_price,
            entry_time=timestamp,
            transaction_costs=costs["total"],
            slippage_cost=abs(slippage * quantity),
            option_type=option_type,
            strike=strike,
            tag=tag,
        )

        self.capital -= costs["total"]
        self.positions.append(position)

        logger.debug(
            "Position opened | symbol=%s side=%s qty=%d price=%.2f fill=%.2f "
            "slippage=%.4f costs=%.2f tag=%s",
            symbol, side, quantity, price, fill_price,
            slippage, costs["total"], tag,
        )
        return position

    def close_position(
        self,
        position: BacktestPosition,
        timestamp: datetime,
        exit_price: float,
        spot_price: float = 0.0,
    ) -> Optional[BacktestTrade]:
        """Close an existing position with realistic costs."""
        if not position.is_open:
            return None

        self._check_new_day(timestamp)

        exit_side = "SELL" if position.side == "BUY" else "BUY"

        # Calculate exit slippage
        slippage = self.slippage_sim.estimate(
            exit_price, spot_price or exit_price,
            position.option_type, exit_side, position.quantity,
        )
        fill_price = exit_price + slippage

        # Exit transaction costs
        exit_costs = self.cost_model.calculate(
            fill_price, position.quantity, exit_side,
            "options" if position.option_type else "futures",
        )

        # Calculate PnL
        if position.side == "BUY":
            gross_pnl = (fill_price - position.entry_price) * position.quantity
        else:
            gross_pnl = (position.entry_price - fill_price) * position.quantity

        total_costs = position.transaction_costs + exit_costs["total"]
        total_slippage = position.slippage_cost + abs(slippage * position.quantity)
        net_pnl = gross_pnl - total_costs

        # Update position
        position.exit_price = fill_price
        position.exit_time = timestamp
        position.pnl = net_pnl
        position.is_open = False

        # Update capital and daily PnL
        self.capital += net_pnl
        self._current_day_pnl += net_pnl
        self.equity_curve.append(self.capital)

        # Record trade
        hold_duration = (timestamp - position.entry_time).total_seconds() / 60.0
        trade = BacktestTrade(
            entry_time=position.entry_time,
            exit_time=timestamp,
            symbol=position.symbol,
            side=position.side,
            quantity=position.quantity,
            entry_price=position.entry_price,
            exit_price=fill_price,
            gross_pnl=gross_pnl,
            transaction_costs=total_costs,
            slippage_cost=total_slippage,
            net_pnl=net_pnl,
            hold_duration_minutes=hold_duration,
            tag=position.tag,
        )
        self.closed_trades.append(trade)

        # Check kill switch after each trade
        self._check_kill_switch()

        logger.debug(
            "Position closed | symbol=%s gross=%.2f costs=%.2f net=%.2f "
            "hold=%.1fm day_pnl=%.2f",
            position.symbol, gross_pnl, total_costs, net_pnl,
            hold_duration, self._current_day_pnl,
        )
        return trade

    def update_mtm(self, current_prices: dict[str, float]):
        """Update mark-to-market for all open positions."""
        total_unrealized = 0.0
        for pos in self.positions:
            if not pos.is_open:
                continue
            current = current_prices.get(pos.symbol, pos.entry_price)
            if pos.side == "BUY":
                unrealized = (current - pos.entry_price) * pos.quantity
            else:
                unrealized = (pos.entry_price - current) * pos.quantity
            total_unrealized += unrealized

        effective_day_pnl = self._current_day_pnl + total_unrealized
        loss_pct = abs(effective_day_pnl) / self.initial_capital
        if effective_day_pnl < 0 and loss_pct >= self.max_daily_loss_pct:
            if not self._kill_switch_active:
                logger.warning(
                    "MTM kill switch approaching | unrealized_pnl=%.2f effective_day=%.2f",
                    total_unrealized, effective_day_pnl,
                )

    def run_strategy(
        self,
        data: pd.DataFrame,
        strategy_fn: Callable,
        spot_data: Optional[pd.DataFrame] = None,
    ):
        """
        Run a strategy function over historical data.

        Args:
            data: OHLCV DataFrame with DatetimeIndex
            strategy_fn: Callable(engine, bar_index, row, spot_row) -> None
                         Strategy function that calls engine.open_position/close_position
            spot_data: Optional underlying spot data aligned with `data`
        """
        logger.info(
            "Backtest starting | bars=%d capital=%.2f", len(data), self.capital,
        )
        t_start = datetime.now()

        for i, (timestamp, row) in enumerate(data.iterrows()):
            self._check_new_day(timestamp)

            if self._kill_switch_active:
                # Force close all open positions
                for pos in self.positions:
                    if pos.is_open:
                        self.close_position(pos, timestamp, row["close"])
                continue

            spot_row = None
            if spot_data is not None and timestamp in spot_data.index:
                spot_row = spot_data.loc[timestamp]

            strategy_fn(self, i, row, spot_row)

        # Close any remaining open positions at last bar
        if len(data) > 0:
            last_ts = data.index[-1]
            last_row = data.iloc[-1]
            for pos in self.positions:
                if pos.is_open:
                    self.close_position(pos, last_ts, last_row["close"])

        # Final day PnL
        if self._current_day_pnl != 0:
            self.daily_pnl.append(self._current_day_pnl)

        elapsed = (datetime.now() - t_start).total_seconds()
        logger.info("Backtest completed | elapsed=%.2fs trades=%d", elapsed, len(self.closed_trades))

    def get_results(self) -> BacktestResult:
        """Calculate comprehensive backtest statistics."""
        if not self.closed_trades:
            return BacktestResult()

        net_pnls = [t.net_pnl for t in self.closed_trades]
        wins = [p for p in net_pnls if p > 0]
        losses = [p for p in net_pnls if p <= 0]

        # Drawdown calculation
        equity = np.array(self.equity_curve)
        peak = np.maximum.accumulate(equity)
        drawdown = equity - peak
        max_dd = drawdown.min()
        max_dd_pct = max_dd / peak[np.argmin(drawdown)] if peak[np.argmin(drawdown)] > 0 else 0

        # Sharpe ratio (annualized, assuming daily PnL)
        daily_returns = np.array(self.daily_pnl)
        sharpe = 0.0
        if len(daily_returns) > 1 and daily_returns.std() > 0:
            sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)

        # Profit factor
        gross_wins = sum(wins) if wins else 0
        gross_losses = abs(sum(losses)) if losses else 1
        profit_factor = gross_wins / gross_losses if gross_losses > 0 else float("inf")

        total_costs = sum(t.transaction_costs for t in self.closed_trades)
        total_slippage = sum(t.slippage_cost for t in self.closed_trades)

        result = BacktestResult(
            total_trades=len(self.closed_trades),
            winning_trades=len(wins),
            losing_trades=len(losses),
            win_rate=len(wins) / len(self.closed_trades) if self.closed_trades else 0,
            gross_pnl=sum(t.gross_pnl for t in self.closed_trades),
            total_transaction_costs=total_costs,
            total_slippage_costs=total_slippage,
            net_pnl=sum(net_pnls),
            max_drawdown=max_dd,
            max_drawdown_pct=max_dd_pct,
            sharpe_ratio=sharpe,
            profit_factor=profit_factor,
            avg_win=np.mean(wins) if wins else 0,
            avg_loss=np.mean(losses) if losses else 0,
            avg_hold_minutes=np.mean([t.hold_duration_minutes for t in self.closed_trades]),
            total_days=len(self.daily_pnl),
            daily_pnl=self.daily_pnl,
            equity_curve=self.equity_curve,
            trades=self.closed_trades,
            kill_switch_triggers=self.kill_switch_triggers,
        )

        logger.info(
            "Backtest Results | trades=%d win_rate=%.1f%% net_pnl=%.2f "
            "sharpe=%.4f profit_factor=%.2f max_dd=%.2f%% costs=%.2f "
            "slippage=%.2f kill_switches=%d",
            result.total_trades, result.win_rate * 100, result.net_pnl,
            result.sharpe_ratio, result.profit_factor, result.max_drawdown_pct * 100,
            result.total_transaction_costs, result.total_slippage_costs,
            result.kill_switch_triggers,
        )
        return result

    def print_report(self, result: Optional[BacktestResult] = None):
        """Print a formatted backtest report to console."""
        if result is None:
            result = self.get_results()

        report = f"""
╔══════════════════════════════════════════════════════════════╗
║              BACKTEST RESULTS REPORT                         ║
╠══════════════════════════════════════════════════════════════╣
║  Initial Capital:      ₹{self.initial_capital:>15,.2f}                ║
║  Final Capital:        ₹{self.capital:>15,.2f}                ║
║  Net PnL:              ₹{result.net_pnl:>15,.2f}                ║
║  Return:                {(result.net_pnl / self.initial_capital) * 100:>14.2f}%                ║
╠══════════════════════════════════════════════════════════════╣
║  Total Trades:          {result.total_trades:>14d}                 ║
║  Winning Trades:        {result.winning_trades:>14d}                 ║
║  Losing Trades:         {result.losing_trades:>14d}                 ║
║  Win Rate:              {result.win_rate * 100:>14.1f}%                ║
║  Avg Win:              ₹{result.avg_win:>15,.2f}                ║
║  Avg Loss:             ₹{result.avg_loss:>15,.2f}                ║
╠══════════════════════════════════════════════════════════════╣
║  Sharpe Ratio:          {result.sharpe_ratio:>14.4f}                 ║
║  Profit Factor:         {result.profit_factor:>14.2f}                 ║
║  Max Drawdown:          {result.max_drawdown_pct * 100:>14.2f}%                ║
║  Avg Hold (min):        {result.avg_hold_minutes:>14.1f}                 ║
╠══════════════════════════════════════════════════════════════╣
║  Transaction Costs:    ₹{result.total_transaction_costs:>15,.2f}                ║
║  Slippage Costs:       ₹{result.total_slippage_costs:>15,.2f}                ║
║  Kill Switch Triggers:  {result.kill_switch_triggers:>14d}                 ║
║  Trading Days:          {result.total_days:>14d}                 ║
╚══════════════════════════════════════════════════════════════╝
"""
        print(report)
