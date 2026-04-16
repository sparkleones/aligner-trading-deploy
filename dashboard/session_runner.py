"""
Live session simulator for dashboard visualization.
Replays market data at accelerated speed, places paper orders using
OPTION PREMIUM prices (not underlying), and emits real-time events
for the dashboard.

Key design: strategies generate signals from UNDERLYING price data,
but orders are executed at OPTION PREMIUM prices. This correctly models
options trading where:
- STT is on premium turnover (not notional)
- Theta decay benefits short premium sellers
- Costs are realistic (~₹50-100 per round trip vs ~₹1800 at underlying)
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
import uuid
from datetime import datetime, timedelta
from typing import Any, Callable, Coroutine, Optional, Union

import numpy as np
import pandas as pd

from backtesting.paper_trading import PaperTradingBroker
from config.constants import (
    FREEZE_LIMITS,
    INDEX_CONFIG,
    MARKET_CLOSE_HOUR,
    MARKET_CLOSE_MINUTE,
    MARKET_OPEN_HOUR,
    MARKET_OPEN_MINUTE,
)
from risk_management.order_slicer import OrderSlicer
from risk_management.risk_manager import Position, RiskLevel, RiskManager

logger = logging.getLogger(__name__)

# ── Premium Computation ────────────────────────────────────────────────────


def compute_option_premium(
    spot: float,
    bar_idx: int,
    total_bars: int,
    vol_annual: float = 0.14,
    premium_type: str = "straddle",
) -> float:
    """Compute synthetic option premium from spot price and time position.

    Models ATM option premium with theta decay through the trading day.
    At market open, ATM straddle premium ≈ 1.8-2.2% of spot.
    By close, premiums decay to ~0.3-0.5% of spot (mostly intrinsic).

    Parameters
    ----------
    spot : float
        Current underlying price.
    bar_idx : int
        Current bar index (0 = market open).
    total_bars : int
        Total bars in the session (typically 25 for 15-min bars).
    vol_annual : float
        Annualised implied volatility.
    premium_type : str
        "call", "put", or "straddle" (CE+PE combined).

    Returns
    -------
    float
        Synthetic option premium per unit.
    """
    total = max(total_bars, 1)
    # Time remaining as fraction of day [1.0 at open → ~0.04 at close]
    time_frac = max(0.04, 1.0 - bar_idx / total)

    # Daily vol from annual: vol_daily = vol_annual / sqrt(252)
    vol_daily = vol_annual / math.sqrt(252)

    # ATM option premium ≈ spot × vol_daily × sqrt(time_remaining) × 0.4
    # (simplified Black-Scholes for ATM: C ≈ S × σ × √T × 0.4)
    single_leg = spot * vol_daily * math.sqrt(time_frac) * 0.4

    if premium_type == "straddle":
        # Straddle = CE + PE, at ATM roughly 2× single leg
        premium = single_leg * 2.0
    elif premium_type == "condor":
        # Iron condor collects ~60% of straddle premium (OTM wings)
        premium = single_leg * 1.2
    elif premium_type == "spread":
        # Bull put spread collects ~40% of single leg
        premium = single_leg * 0.8
    else:
        premium = single_leg

    return round(max(premium, 5.0), 2)


# ── Strategy Signal Definitions ─────────────────────────────────────────────

_STRATEGY_REGISTRY: dict[str, Callable] = {}
_STRATEGY_PREMIUM_TYPE: dict[str, str] = {}


def _register_strategy(name: str, premium_type: str = "call"):
    """Decorator to register a strategy function with its premium type."""
    def wrapper(fn):
        _STRATEGY_REGISTRY[name] = fn
        _STRATEGY_PREMIUM_TYPE[name] = premium_type
        return fn
    return wrapper


@_register_strategy("mean_reversion", premium_type="call")
def _strategy_mean_reversion(
    bars: pd.DataFrame, current_idx: int, position_open: bool
) -> Optional[dict]:
    """Buy when price dips below 10-bar SMA by 0.15%, sell when above by 0.15%.
    Uses tighter thresholds for more trades and a trailing-style exit.
    """
    if current_idx < 10:
        return None
    window = bars.iloc[max(0, current_idx - 10) : current_idx + 1]
    sma = window["close"].mean()
    price = bars.iloc[current_idx]["close"]
    deviation = (price - sma) / sma

    if deviation < -0.0015 and not position_open:
        return {"side": "BUY", "reason": "price_below_sma"}
    elif deviation > 0.001 and position_open:
        return {"side": "SELL", "reason": "price_above_sma"}
    elif current_idx >= 23 and position_open:
        return {"side": "SELL", "reason": "eod_exit"}
    return None


@_register_strategy("momentum", premium_type="call")
def _strategy_momentum(
    bars: pd.DataFrame, current_idx: int, position_open: bool
) -> Optional[dict]:
    """Buy on 2 consecutive up bars with increasing volume, sell on reversal."""
    if current_idx < 3:
        return None
    recent = bars.iloc[current_idx - 2 : current_idx + 1]
    closes = recent["close"].values

    if not position_open:
        # 2 consecutive up bars
        if closes[-1] > closes[-2] > closes[-3]:
            # Confirm with momentum strength
            pct_gain = (closes[-1] - closes[-3]) / closes[-3]
            if pct_gain > 0.001:  # At least 0.1% gain over 2 bars
                return {"side": "BUY", "reason": "momentum_up"}
    else:
        # Exit on any down bar or EOD
        if closes[-1] < closes[-2]:
            return {"side": "SELL", "reason": "momentum_reversal"}
        if current_idx >= 22:
            return {"side": "SELL", "reason": "eod_exit"}
    return None


@_register_strategy("breakout", premium_type="call")
def _strategy_breakout(
    bars: pd.DataFrame, current_idx: int, position_open: bool
) -> Optional[dict]:
    """Buy on break above 8-bar high, sell on break below 8-bar low or trailing stop."""
    if current_idx < 8:
        return None
    lookback = bars.iloc[max(0, current_idx - 8) : current_idx]
    price = bars.iloc[current_idx]["close"]
    high = lookback["high"].max()
    low = lookback["low"].min()

    if price > high and not position_open:
        return {"side": "BUY", "reason": "breakout_high"}
    elif position_open:
        if price < low:
            return {"side": "SELL", "reason": "breakout_low"}
        if current_idx >= 22:
            return {"side": "SELL", "reason": "eod_exit"}
    return None


@_register_strategy("short_straddle", premium_type="straddle")
def _strategy_short_straddle(
    bars: pd.DataFrame, current_idx: int, position_open: bool
) -> Optional[dict]:
    """Sell ATM straddle at bar 1 (~9:30 AM). Exit at EOD or on stop-loss.

    Profits from theta decay. Premium naturally decays through the day,
    so selling early and buying back later captures the time value.
    Stop-loss: exit if underlying moves > 1.2% from entry (big breakout).
    """
    if current_idx < 1:
        return None

    # Enter short at bar 1 (9:30 AM)
    if current_idx == 1 and not position_open:
        return {"side": "SELL", "reason": "straddle_entry_0930"}

    if position_open:
        # Stop-loss: if underlying moves too much, delta losses overwhelm theta
        entry_price = bars.iloc[1]["close"]
        current_price = bars.iloc[current_idx]["close"]
        pct_move = abs(current_price - entry_price) / entry_price

        if pct_move > 0.012:  # 1.2% move = ~40% premium loss
            return {"side": "BUY", "reason": "straddle_stop_loss"}
        # EOD exit: capture full day's theta decay
        if current_idx >= 23:
            return {"side": "BUY", "reason": "straddle_eod_exit"}
    return None


@_register_strategy("delta_neutral", premium_type="straddle")
def _strategy_delta_neutral(
    bars: pd.DataFrame, current_idx: int, position_open: bool
) -> Optional[dict]:
    """Delta-neutral straddle: sell at bar 1, exit EOD. More tolerant of
    moves because delta hedging is simulated (wider stop-loss than naked straddle).
    """
    if current_idx < 1:
        return None

    if current_idx == 1 and not position_open:
        return {"side": "SELL", "reason": "delta_neutral_entry"}

    if position_open:
        # Wider stop-loss due to hedging (1.8% vs 1.2% for naked)
        entry_price = bars.iloc[1]["close"]
        current_price = bars.iloc[current_idx]["close"]
        pct_move = abs(current_price - entry_price) / entry_price

        if pct_move > 0.018:
            return {"side": "BUY", "reason": "delta_neutral_stop"}
        if current_idx >= 23:
            return {"side": "BUY", "reason": "delta_neutral_eod"}
    return None


@_register_strategy("bull_put_spread", premium_type="spread")
def _strategy_bull_put_spread(
    bars: pd.DataFrame, current_idx: int, position_open: bool
) -> Optional[dict]:
    """Bull put spread: sell when RSI < 42 (oversold bounce expected).
    Profits from premium decay if market stays above the short put strike.
    """
    if current_idx < 14:
        return None

    window = bars.iloc[max(0, current_idx - 14) : current_idx + 1]
    deltas = window["close"].diff().dropna()
    gains = deltas.clip(lower=0).mean()
    losses = (-deltas.clip(upper=0)).mean()
    rs = gains / losses if losses > 0 else 100
    rsi = 100 - (100 / (1 + rs))

    if rsi < 42 and not position_open:
        return {"side": "SELL", "reason": "rsi_oversold_spread"}
    elif position_open:
        # Profit target: RSI recovers above 55
        if rsi > 55:
            return {"side": "BUY", "reason": "rsi_recovered_spread"}
        if current_idx >= 23:
            return {"side": "BUY", "reason": "spread_eod_exit"}
    return None


@_register_strategy("iron_condor", premium_type="condor")
def _strategy_iron_condor(
    bars: pd.DataFrame, current_idx: int, position_open: bool
) -> Optional[dict]:
    """Iron condor: enter when realized vol is low. Exit on vol expansion or EOD.
    Profits from premium decay when market stays range-bound.
    """
    if current_idx < 5:
        return None
    window = bars.iloc[max(0, current_idx - 5) : current_idx + 1]
    volatility = window["close"].pct_change().std()

    # Enter when vol is low (sideways market)
    if volatility < 0.003 and not position_open and current_idx >= 3:
        return {"side": "SELL", "reason": "condor_low_vol_entry"}

    if position_open:
        # Exit on vol expansion (breakout risk)
        if volatility > 0.007:
            return {"side": "BUY", "reason": "condor_vol_breakout"}
        if current_idx >= 23:
            return {"side": "BUY", "reason": "condor_eod_exit"}
    return None


@_register_strategy("pairs_trade", premium_type="call")
def _strategy_pairs_trade(
    bars: pd.DataFrame, current_idx: int, position_open: bool
) -> Optional[dict]:
    """Pairs trading using z-score mean reversion on rolling price spread."""
    if current_idx < 12:
        return None
    window = bars.iloc[max(0, current_idx - 12) : current_idx + 1]
    closes = window["close"].values
    mean = closes.mean()
    std = closes.std()
    if std < 1e-6:
        return None
    z_score = (closes[-1] - mean) / std

    if not position_open:
        if z_score < -1.5:
            return {"side": "BUY", "reason": "pairs_z_low"}
        elif z_score > 1.5:
            return {"side": "SELL", "reason": "pairs_z_high"}
    elif position_open:
        if abs(z_score) < 0.3:
            return {"side": "BUY", "reason": "pairs_mean_revert"}
        if current_idx >= 23:
            return {"side": "BUY", "reason": "pairs_eod_exit"}
    return None


@_register_strategy("ddqn_agent", premium_type="call")
def _strategy_ddqn_agent(
    bars: pd.DataFrame, current_idx: int, position_open: bool
) -> Optional[dict]:
    """DDQN agent: EMA crossover + momentum filter for trend following."""
    if current_idx < 12:
        return None
    closes = bars.iloc[max(0, current_idx - 12) : current_idx + 1]["close"]

    # Fast EMA (5) vs Slow EMA (12)
    ema_fast = closes.ewm(span=5, adjust=False).mean().iloc[-1]
    ema_slow = closes.ewm(span=12, adjust=False).mean().iloc[-1]
    current_price = closes.iloc[-1]

    # Momentum: price above both EMAs = bullish
    cross_pct = (ema_fast - ema_slow) / ema_slow

    if not position_open:
        if cross_pct > 0.0008 and current_price > ema_fast:
            return {"side": "BUY", "reason": "ddqn_bullish_cross"}
        elif cross_pct < -0.0008 and current_price < ema_fast:
            return {"side": "SELL", "reason": "ddqn_bearish_cross"}
    else:
        # Exit on cross reversal or EOD
        if abs(cross_pct) < 0.0002:
            return {"side": "BUY", "reason": "ddqn_cross_reversal"}
        if current_idx >= 22:
            return {"side": "BUY", "reason": "ddqn_eod_exit"}
    return None


# ── Session Runner ──────────────────────────────────────────────────────────


class SessionRunner:
    """Runs a simulated trading session with OPTION PREMIUM pricing.

    Strategies generate signals from underlying OHLCV data, but orders
    are executed at synthetic option premium prices. This correctly models:
    - Theta decay (premiums shrink through the day)
    - Realistic transaction costs (STT on premium, not notional)
    - Short premium profitability (sell high premium, buy back lower)

    Parameters
    ----------
    strategy_name : str
        Registered strategy name.
    capital : float
        Starting capital.
    speed_multiplier : float
        Replay speed (10 = day in ~2.5 min).
    symbol : str
        Underlying symbol.
    lot_size : int or None
        Contract lot size.
    """

    def __init__(
        self,
        strategy_name: str,
        capital: float,
        speed_multiplier: float = 10.0,
        symbol: str = "NIFTY",
        lot_size: Optional[int] = None,
    ) -> None:
        self.strategy_name = strategy_name
        self.capital = capital
        self.speed_multiplier = speed_multiplier
        self.symbol = symbol
        self.lot_size = lot_size or INDEX_CONFIG.get(symbol, {}).get("lot_size", 25)

        # Core components
        self.broker = PaperTradingBroker(initial_capital=capital, brokerage_per_order=20.0)
        self.risk_manager = RiskManager(total_capital=capital)
        self.slicer = OrderSlicer(broker_client=self.broker)

        # Session state
        self._running = False
        self._stopped = False
        self._current_bar: int = 0
        self._total_bars: int = 0
        self._position_open: bool = False
        self._entry_bar: Optional[int] = None
        self._entry_price: float = 0.0  # Premium price at entry
        self._entry_spot: float = 0.0   # Underlying price at entry
        self._entry_side: str = "BUY"
        self._session_id: str = str(uuid.uuid4())[:8]

        # Premium type for this strategy
        self._premium_type = _STRATEGY_PREMIUM_TYPE.get(strategy_name, "call")

        # Resolve strategy function
        if strategy_name not in _STRATEGY_REGISTRY:
            available = list(_STRATEGY_REGISTRY.keys())
            raise ValueError(
                f"Unknown strategy '{strategy_name}'. Available: {available}"
            )
        self._strategy_fn = _STRATEGY_REGISTRY[strategy_name]

        logger.info(
            "SessionRunner created | id=%s strategy=%s premium_type=%s capital=%.0f",
            self._session_id, strategy_name, self._premium_type, capital,
        )

    # ── Main Loop ────────────────────────────────────────────────────────

    async def run_session(
        self,
        data: pd.DataFrame,
        callback: Callable[[dict], Union[None, Coroutine]],
    ) -> dict:
        """Replay *data* bar-by-bar, executing strategy logic and emitting
        events via *callback*.
        """
        self._running = True
        self._stopped = False
        self._total_bars = len(data)

        if "timestamp" in data.columns:
            data = data.set_index("timestamp")

        real_bar_interval_s = 15 * 60
        simulated_delay = real_bar_interval_s / self.speed_multiplier

        logger.info(
            "Session %s starting | bars=%d delay=%.2fs/bar premium_type=%s",
            self._session_id, self._total_bars, simulated_delay, self._premium_type,
        )

        for i in range(len(data)):
            if self._stopped:
                break

            self._current_bar = i
            bar = data.iloc[i]
            ts = str(data.index[i])
            spot_price = float(bar["close"])

            # Compute option premium for this bar
            premium = compute_option_premium(
                spot_price, i, self._total_bars,
                premium_type=self._premium_type,
            )

            # Update broker with PREMIUM price (not underlying)
            self.broker.update_price(self.symbol, premium)

            # ── Emit tick event ──────────────────────────────────────
            await self._emit(callback, {
                "type": "tick",
                "timestamp": ts,
                "price": spot_price,
                "premium": premium,
                "open": float(bar["open"]),
                "high": float(bar["high"]),
                "low": float(bar["low"]),
                "volume": int(bar["volume"]),
                "bar_index": i,
                "session_id": self._session_id,
                "strategy": self.strategy_name,
            })

            # ── Run strategy (signals based on underlying data) ───────
            if not self.risk_manager.kill_switch_active:
                signal = self._strategy_fn(data, i, self._position_open)
                if signal is not None:
                    await self._execute_signal(
                        signal, bar, ts, data, i, premium, callback
                    )

            # ── Update MTM and risk ──────────────────────────────────
            risk_status = self._update_risk(premium)

            await self._emit(callback, {
                "type": "position_update",
                "timestamp": ts,
                "positions": self.broker.get_positions(),
                "total_mtm": float(risk_status.unrealised_pnl),
                "capital": float(self.broker.capital),
                "premium": premium,
                "strategy": self.strategy_name,
            })

            portfolio = self.broker.get_portfolio()
            realized = sum(
                t.get("realized_pnl", 0) for t in self.broker.trade_log
            )
            unrealized = float(portfolio["total_unrealized_pnl"])

            await self._emit(callback, {
                "type": "pnl_update",
                "timestamp": ts,
                "realized_pnl": round(realized, 2),
                "unrealized_pnl": round(unrealized, 2),
                "total_pnl": round(realized + unrealized, 2),
                "pnl_pct": round((realized + unrealized) / self.capital * 100, 4),
                "strategy": self.strategy_name,
            })

            await self._emit(callback, {
                "type": "risk_update",
                "timestamp": ts,
                "risk_level": risk_status.risk_level.value,
                "daily_pnl_pct": float(risk_status.daily_pnl_pct * 100),
                "kill_switch": risk_status.kill_switch_active,
                "strategy": self.strategy_name,
            })

            # ── Check kill switch ────────────────────────────────────
            if self.risk_manager.check_kill_switch():
                report = self.risk_manager.execute_kill_switch(self.broker)
                self._position_open = False
                await self._emit(callback, {
                    "type": "kill_switch",
                    "timestamp": ts,
                    "reason": f"Daily loss {risk_status.daily_pnl_pct * 100:.2f}% breached threshold",
                    "positions_closed": report.positions_squared,
                    "strategy": self.strategy_name,
                })

            # ── Pace replay ──────────────────────────────────────────
            await asyncio.sleep(simulated_delay)

        # ── End of session: square off open positions ────────────────
        if self._position_open and not self._stopped:
            last_bar = data.iloc[-1]
            last_ts = str(data.index[-1])
            last_premium = compute_option_premium(
                float(last_bar["close"]), len(data) - 1, self._total_bars,
                premium_type=self._premium_type,
            )
            await self._close_position(
                last_bar, last_ts, data, len(data) - 1,
                last_premium, callback, reason="session_end",
            )

        self._running = False

        # ── Session summary ──────────────────────────────────────────
        session_report = self.broker.get_session_report()
        summary = {
            "session_id": self._session_id,
            "strategy": self.strategy_name,
            "premium_type": self._premium_type,
            "initial_capital": self.capital,
            "final_capital": session_report["final_capital"],
            "net_value": session_report["net_value"],
            "total_pnl": session_report["total_pnl"],
            "return_pct": session_report["return_pct"],
            "total_trades": session_report["total_trades"],
            "winning_trades": session_report["winning_trades"],
            "losing_trades": session_report["losing_trades"],
            "win_rate": session_report["win_rate"],
            "bars_processed": self._current_bar + 1,
            "total_orders": session_report["total_orders"],
        }

        await self._emit(callback, {
            "type": "session_end",
            "timestamp": str(data.index[-1]),
            "summary": summary,
            "strategy": self.strategy_name,
        })

        logger.info(
            "Session %s complete | pnl=%.2f return=%.2f%% trades=%d",
            self._session_id,
            summary["total_pnl"],
            summary["return_pct"],
            summary["total_trades"],
        )

        return summary

    # ── Signal Execution ─────────────────────────────────────────────

    async def _execute_signal(
        self,
        signal: dict,
        bar: Any,
        ts: str,
        data: pd.DataFrame,
        bar_idx: int,
        premium: float,
        callback: Callable,
    ) -> None:
        """Translate a strategy signal into a paper order at PREMIUM price."""
        side = signal["side"]
        qty = self.lot_size

        # ── Opening a new position ────────────────────────────────────
        if not self._position_open:
            result = self.broker.place_order(
                symbol=self.symbol,
                side=side,
                qty=qty,
                order_type="MARKET",
                price=premium,
                tag=self.strategy_name,
            )
            if result.get("success"):
                await self._emit_order_event(
                    ts, side, qty, premium, result, callback
                )
                self._position_open = True
                self._entry_bar = bar_idx
                self._entry_price = premium
                self._entry_spot = float(bar["close"])
                self._entry_side = side

        # ── Closing an existing position ──────────────────────────────
        elif self._position_open:
            await self._close_position(
                bar, ts, data, bar_idx, premium, callback,
                reason=signal.get("reason", "signal"),
            )

    async def _close_position(
        self,
        bar: Any,
        ts: str,
        data: pd.DataFrame,
        bar_idx: int,
        premium: float,
        callback: Callable,
        reason: str = "signal",
    ) -> None:
        """Close the current position at PREMIUM price and emit trade_closed."""
        qty = self.lot_size

        entry_side = getattr(self, "_entry_side", "BUY")
        close_side = "SELL" if entry_side == "BUY" else "BUY"

        result = self.broker.place_order(
            symbol=self.symbol,
            side=close_side,
            quantity=qty,
            order_type="MARKET",
            price=premium,
            tag=self.strategy_name,
        )

        if result.get("success"):
            await self._emit_order_event(
                ts, close_side, qty, premium, result, callback
            )

            # Compute trade PnL (direction-aware, at premium level)
            if entry_side == "BUY":
                pnl = (premium - self._entry_price) * qty
            else:
                # Short: sold high premium, bought back lower = profit
                pnl = (self._entry_price - premium) * qty
            hold_bars = bar_idx - (self._entry_bar or 0)
            hold_mins = hold_bars * 15

            self.risk_manager.add_realised_pnl(pnl)

            await self._emit(callback, {
                "type": "trade_closed",
                "timestamp": ts,
                "symbol": self.symbol,
                "side": entry_side,
                "entry_premium": round(self._entry_price, 2),
                "exit_premium": round(premium, 2),
                "entry_spot": round(self._entry_spot, 2),
                "exit_spot": round(float(bar["close"]), 2),
                "pnl": round(pnl, 2),
                "hold_mins": hold_mins,
                "reason": reason,
                "strategy": self.strategy_name,
            })

            logger.info(
                "Trade closed | strategy=%s side=%s entry_prem=%.2f exit_prem=%.2f "
                "pnl=%.2f hold=%dmin reason=%s",
                self.strategy_name, entry_side,
                self._entry_price, premium, pnl, hold_mins, reason,
            )

        self._position_open = False
        self._entry_bar = None
        self._entry_price = 0.0
        self._entry_spot = 0.0

    async def _emit_order_event(
        self,
        ts: str,
        side: str,
        qty: int,
        premium: float,
        result: dict,
        callback: Callable,
    ) -> None:
        """Emit an order event."""
        await self._emit(callback, {
            "type": "order",
            "timestamp": ts,
            "symbol": self.symbol,
            "side": side,
            "qty": qty,
            "price": premium,
            "fill_price": result.get("fill_price", premium),
            "costs": round(result.get("costs", 0.0), 2),
            "order_id": result.get("order_id", ""),
            "strategy": self.strategy_name,
        })

    # ── Risk Update ──────────────────────────────────────────────────

    def _update_risk(self, current_premium: float):
        """Update RiskManager using current premium price."""
        positions = []
        for pos_dict in self.broker.get_positions():
            side = "BUY" if pos_dict["quantity"] > 0 else "SELL"
            positions.append(Position(
                symbol=pos_dict["symbol"],
                qty=abs(pos_dict["quantity"]),
                side=side,
                entry_price=pos_dict["average_price"],
                current_price=current_premium,
            ))
        return self.risk_manager.update_mtm(positions)

    # ── Status / Control ─────────────────────────────────────────────

    def get_status(self) -> dict:
        """Return the current session state snapshot."""
        portfolio = self.broker.get_portfolio()
        realized = sum(t.get("realized_pnl", 0) for t in self.broker.trade_log)
        unrealized = portfolio["total_unrealized_pnl"]

        return {
            "session_id": self._session_id,
            "strategy": self.strategy_name,
            "premium_type": self._premium_type,
            "running": self._running,
            "stopped": self._stopped,
            "current_bar": self._current_bar,
            "total_bars": self._total_bars,
            "progress_pct": (
                (self._current_bar + 1) / self._total_bars * 100
                if self._total_bars > 0 else 0
            ),
            "position_open": self._position_open,
            "positions": self.broker.get_positions(),
            "capital": portfolio["capital"],
            "realized_pnl": realized,
            "unrealized_pnl": unrealized,
            "total_pnl": realized + unrealized,
            "risk_level": (
                "KILL_SWITCH" if self.risk_manager.kill_switch_active
                else "NORMAL"
            ),
            "total_orders": len(self.broker.orders),
            "total_trades": len(self.broker.trade_log),
        }

    async def stop(self) -> None:
        """Gracefully stop the session and square off open positions."""
        logger.info("Session %s — stop requested", self._session_id)
        self._stopped = True

        if self._position_open:
            positions = self.broker.get_positions()
            for pos in positions:
                exit_side = "SELL" if pos["quantity"] > 0 else "BUY"
                self.broker.place_order(
                    symbol=pos["symbol"],
                    side=exit_side,
                    quantity=abs(pos["quantity"]),
                    order_type="MARKET",
                    price=pos["last_price"],
                    tag="session_stop",
                )
            self._position_open = False

        self._running = False
        logger.info("Session %s stopped", self._session_id)

    # ── Helpers ──────────────────────────────────────────────────────

    @staticmethod
    async def _emit(
        callback: Callable[[dict], Union[None, Coroutine]],
        event: dict,
    ) -> None:
        """Call *callback* with *event*, handling both sync and async."""
        result = callback(event)
        if asyncio.iscoroutine(result) or asyncio.isfuture(result):
            await result


# ── Multi-Strategy Session ──────────────────────────────────────────────────


async def run_multi_strategy_session(
    data: pd.DataFrame,
    strategies: list[str],
    capital: float,
    speed_multiplier: float,
    callback: Callable[[dict], Union[None, Coroutine]],
) -> dict:
    """Run multiple strategies in parallel on the same market data.

    Each strategy gets its own SessionRunner with independent broker
    and risk manager. Events are tagged with the strategy name.
    """
    runners = []
    for name in strategies:
        runner = SessionRunner(
            strategy_name=name,
            capital=capital,
            speed_multiplier=speed_multiplier,
        )
        runners.append((name, runner))

    tasks = [
        asyncio.create_task(runner.run_session(data.copy(), callback))
        for _, runner in runners
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    combined: dict[str, Any] = {}
    for (name, _), result in zip(runners, results):
        if isinstance(result, Exception):
            logger.error("Strategy %s failed: %s", name, result)
            combined[name] = {"error": str(result)}
        else:
            combined[name] = result

    return combined


# ── Synthetic Intraday Data Generator ───────────────────────────────────────


def generate_intraday_data(
    base_price: float = 24000.0,
    volatility: float = 0.14,
    trend: str = "sideways",
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Generate realistic synthetic 15-minute OHLCV bars for a single trading
    day (9:15 AM to 3:30 PM IST, 25 bars).

    Parameters
    ----------
    base_price : float
        Opening price of the day.
    volatility : float
        Annualised volatility (e.g. 0.14 for 14%).
    trend : str
        One of "bullish", "bearish", "sideways", "volatile",
        "mean_reverting", "trending_up", "trending_down".
    seed : int or None
        Random seed for reproducibility.
    """
    rng = np.random.default_rng(seed)
    n_bars = 25
    today = datetime.now().replace(
        hour=MARKET_OPEN_HOUR,
        minute=MARKET_OPEN_MINUTE,
        second=0,
        microsecond=0,
    )
    timestamps = [today + timedelta(minutes=15 * i) for i in range(n_bars)]

    bar_vol = volatility / np.sqrt(6300)

    trend_map = {
        "bullish": 0.0006,
        "bearish": -0.0006,
        "sideways": 0.0,
        "volatile": 0.0,
        "mean_reverting": 0.0,
        "trending_up": 0.0008,
        "trending_down": -0.0008,
    }
    drift = trend_map.get(trend, 0.0)

    if trend == "volatile":
        bar_vol *= 2.5
    elif trend == "mean_reverting":
        # Oscillating pattern: good for mean reversion and straddles
        bar_vol *= 1.2

    returns = rng.normal(loc=drift, scale=bar_vol, size=n_bars)
    returns[0] = 0.0

    # For mean_reverting, add oscillation
    if trend == "mean_reverting":
        for i in range(1, n_bars):
            # Pull returns toward zero (mean reversion)
            cumulative = sum(returns[:i])
            returns[i] -= cumulative * 0.15

    closes = base_price * np.cumprod(1 + returns)

    opens = np.empty(n_bars)
    highs = np.empty(n_bars)
    lows = np.empty(n_bars)

    opens[0] = base_price
    for i in range(1, n_bars):
        opens[i] = closes[i - 1] * (1 + rng.normal(0, bar_vol * 0.1))

    for i in range(n_bars):
        bar_range = abs(closes[i] - opens[i])
        upper_wick = bar_range * rng.uniform(0.1, 0.6) + abs(rng.normal(0, bar_vol * base_price * 0.3))
        lower_wick = bar_range * rng.uniform(0.1, 0.6) + abs(rng.normal(0, bar_vol * base_price * 0.3))
        highs[i] = max(opens[i], closes[i]) + upper_wick
        lows[i] = min(opens[i], closes[i]) - lower_wick

    base_volume = 500_000
    volume_profile = np.array([
        _u_shaped_volume(i, n_bars, base_volume, rng) for i in range(n_bars)
    ])

    df = pd.DataFrame({
        "timestamp": timestamps,
        "open": np.round(opens, 2),
        "high": np.round(highs, 2),
        "low": np.round(lows, 2),
        "close": np.round(closes, 2),
        "volume": volume_profile.astype(int),
    })

    return df


def _u_shaped_volume(
    bar_idx: int,
    total_bars: int,
    base_volume: int,
    rng: np.random.Generator,
) -> float:
    """Generate a U-shaped intraday volume profile value."""
    t = bar_idx / max(total_bars - 1, 1)
    u_factor = 4.0 * (t - 0.5) ** 2
    multiplier = 0.5 + 2.0 * u_factor
    noise = rng.uniform(0.7, 1.3)
    return base_volume * multiplier * noise
