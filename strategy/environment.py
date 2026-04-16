"""
Gym-like trading environment for the DDQN agent.

Models a single trading day of Indian NSE options (9:15 AM -- 3:30 PM IST)
in 15-minute bars (~25 steps per episode).  The reward function explicitly
accounts for 2026 STT rates, slippage, and a flat trade penalty to
discourage over-trading.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config.constants import (
    DDQN_ACTION_DIM,
    DDQN_STATE_DIM,
    DDQN_TRADE_PENALTY,
    DEFAULT_MAX_DAILY_LOSS_PCT,
    MARKET_CLOSE_HOUR,
    MARKET_CLOSE_MINUTE,
    MARKET_OPEN_HOUR,
    MARKET_OPEN_MINUTE,
    STT_RATES,
)
from strategy.features import FeatureEngine

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Action space
# ---------------------------------------------------------------------------

class Action(IntEnum):
    HOLD = 0
    BUY_CALL = 1
    BUY_PUT = 2
    SELL_CALL = 3
    SELL_PUT = 4


# ---------------------------------------------------------------------------
# Position tracking
# ---------------------------------------------------------------------------

@dataclass
class Position:
    """Represents a single open options/futures position."""
    action: Action
    entry_price: float
    quantity: int = 1
    entry_step: int = 0
    unrealised_pnl: float = 0.0


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class TradingEnvironment:
    """Single-day episodic trading environment.

    Parameters
    ----------
    ohlcv_df : pd.DataFrame
        15-min OHLCV data for one trading day (columns: open, high, low,
        close, volume).
    capital : float
        Starting capital for the day.
    state_dim : int
        Dimensionality of the state vector.
    greeks_series : list[dict], optional
        Per-bar Greeks dicts with keys ``delta, gamma, theta, vega``.
    vix_series : list[float], optional
        Per-bar India VIX values.
    nifty_prices : list[float], optional
        Per-bar Nifty spot prices (for co-integration spread).
    banknifty_prices : list[float], optional
        Per-bar BankNifty spot prices.
    max_daily_loss_pct : float
        MTM kill-switch threshold as fraction of capital.
    """

    N_ACTIONS: int = DDQN_ACTION_DIM

    def __init__(
        self,
        ohlcv_df: pd.DataFrame,
        capital: float = 1_000_000.0,
        state_dim: int = DDQN_STATE_DIM,
        greeks_series: Optional[List[Dict[str, float]]] = None,
        vix_series: Optional[List[float]] = None,
        nifty_prices: Optional[List[float]] = None,
        banknifty_prices: Optional[List[float]] = None,
        max_daily_loss_pct: float = DEFAULT_MAX_DAILY_LOSS_PCT,
    ) -> None:
        self.ohlcv_df = ohlcv_df.reset_index(drop=True)
        self.n_steps: int = len(self.ohlcv_df)
        self.capital: float = capital
        self.state_dim: int = state_dim
        self.max_daily_loss_pct: float = max_daily_loss_pct

        # Optional per-bar auxiliary data (padded to n_steps if short)
        self.greeks_series = self._pad_list(greeks_series, self.n_steps, {})
        self.vix_series = self._pad_list(vix_series, self.n_steps, None)
        self.nifty_prices = self._pad_list(nifty_prices, self.n_steps, None)
        self.banknifty_prices = self._pad_list(banknifty_prices, self.n_steps, None)

        self.feature_engine = FeatureEngine(state_dim=state_dim)

        # Episode state -- initialised in reset()
        self.current_step: int = 0
        self.positions: List[Position] = []
        self.realised_pnl: float = 0.0
        self.total_trades: int = 0
        self.done: bool = False
        self._trade_log: List[Dict] = []

        logger.info(
            "%s | TradingEnvironment created: bars=%d, capital=%.0f, kill_switch=%.2f%%",
            datetime.utcnow().isoformat(),
            self.n_steps,
            self.capital,
            self.max_daily_loss_pct * 100,
        )

    # ------------------------------------------------------------------
    # Gym-like interface
    # ------------------------------------------------------------------

    def reset(self) -> np.ndarray:
        """Reset environment for a new episode (trading day).

        Returns the initial state vector.
        """
        self.current_step = 0
        self.positions = []
        self.realised_pnl = 0.0
        self.total_trades = 0
        self.done = False
        self._trade_log = []
        state = self._get_state()
        logger.debug(
            "%s | Environment reset. First bar ready.",
            datetime.utcnow().isoformat(),
        )
        return state

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute one action and advance to the next bar.

        Parameters
        ----------
        action : int
            Integer in ``[0, N_ACTIONS)``.

        Returns
        -------
        next_state : np.ndarray
        reward : float
        done : bool
        info : dict
        """
        ts = datetime.utcnow().isoformat()
        action = Action(action)

        current_close = float(self.ohlcv_df["close"].iloc[self.current_step])
        reward = 0.0
        info: Dict = {"action": action.name, "step": self.current_step}

        # --- Execute action ---
        if action == Action.HOLD:
            reward = 0.0  # no cost for holding

        elif action in (Action.BUY_CALL, Action.BUY_PUT):
            # Open a new long position
            pos = Position(
                action=action,
                entry_price=current_close,
                entry_step=self.current_step,
            )
            self.positions.append(pos)
            self.total_trades += 1

            # Buy-side stamp duty (tiny, but tracked)
            stamp = current_close * 0.00003
            reward -= stamp / self.capital
            reward -= DDQN_TRADE_PENALTY
            info["trade"] = "open_long"

        elif action in (Action.SELL_CALL, Action.SELL_PUT):
            # If we have a matching long position, close it; otherwise open short
            matching = self._find_matching_position(action)
            if matching is not None:
                # Close the position
                gross_pnl = self._close_pnl(matching, current_close)
                stt = self._compute_stt(current_close, is_options=True)
                slippage = self._estimate_slippage(current_close)
                net_pnl = gross_pnl - stt - slippage

                self.realised_pnl += net_pnl
                self.positions.remove(matching)
                self.total_trades += 1

                reward = net_pnl / self.capital
                reward -= DDQN_TRADE_PENALTY
                info["trade"] = "close"
                info["gross_pnl"] = gross_pnl
                info["stt"] = stt
                info["slippage"] = slippage
                info["net_pnl"] = net_pnl
            else:
                # Open a short position (sell to open)
                pos = Position(
                    action=action,
                    entry_price=current_close,
                    entry_step=self.current_step,
                )
                self.positions.append(pos)
                self.total_trades += 1

                stt = self._compute_stt(current_close, is_options=True)
                reward -= stt / self.capital
                reward -= DDQN_TRADE_PENALTY
                info["trade"] = "open_short"

        # --- Update unrealised PnL for open positions ---
        self._update_unrealised(current_close)

        # --- Advance step ---
        self.current_step += 1

        # --- Done conditions ---
        if self.current_step >= self.n_steps:
            self.done = True
            info["done_reason"] = "end_of_day"

        # MTM kill switch
        total_mtm = self.realised_pnl + sum(p.unrealised_pnl for p in self.positions)
        if total_mtm < -(self.capital * self.max_daily_loss_pct):
            self.done = True
            reward -= 0.01  # additional penalty for hitting kill switch
            info["done_reason"] = "mtm_kill_switch"
            logger.warning(
                "%s | MTM kill switch triggered at step %d, MTM=%.2f",
                ts,
                self.current_step,
                total_mtm,
            )

        next_state = self._get_state() if not self.done else np.zeros(self.state_dim, dtype=np.float32)
        self._trade_log.append(info)

        logger.debug(
            "%s | step=%d action=%s reward=%.6f done=%s",
            ts,
            self.current_step,
            action.name,
            reward,
            self.done,
        )

        return next_state, float(reward), self.done, info

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def trade_log(self) -> List[Dict]:
        """Full trade log for the episode."""
        return list(self._trade_log)

    @property
    def portfolio_value(self) -> float:
        """Current portfolio value (capital + realised + unrealised)."""
        unrealised = sum(p.unrealised_pnl for p in self.positions)
        return self.capital + self.realised_pnl + unrealised

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_state(self) -> np.ndarray:
        """Compute the state vector for the current step."""
        end = self.current_step + 1
        start = max(0, end - 60)  # look-back window for indicators
        window = self.ohlcv_df.iloc[start:end].copy()

        greeks = self.greeks_series[self.current_step] if self.current_step < len(self.greeks_series) else {}
        vix = self.vix_series[self.current_step] if self.current_step < len(self.vix_series) else None
        nifty = self.nifty_prices[self.current_step] if self.current_step < len(self.nifty_prices) else None
        bn = self.banknifty_prices[self.current_step] if self.current_step < len(self.banknifty_prices) else None

        return self.feature_engine.compute(
            ohlcv_df=window,
            greeks=greeks if greeks else None,
            vix=vix,
            nifty_price=nifty,
            banknifty_price=bn,
        )

    def _find_matching_position(self, sell_action: Action) -> Optional[Position]:
        """Find a matching open long position to close.

        SELL_CALL closes a BUY_CALL; SELL_PUT closes a BUY_PUT.
        """
        target = Action.BUY_CALL if sell_action == Action.SELL_CALL else Action.BUY_PUT
        for pos in self.positions:
            if pos.action == target:
                return pos
        return None

    @staticmethod
    def _close_pnl(position: Position, current_price: float) -> float:
        """Gross PnL from closing a position."""
        if position.action in (Action.BUY_CALL, Action.BUY_PUT):
            return current_price - position.entry_price
        # Short position (sell to open, buy to close)
        return position.entry_price - current_price

    @staticmethod
    def _compute_stt(premium: float, is_options: bool = True) -> float:
        """Compute STT cost for a sell-side transaction (2026 rates)."""
        if is_options:
            return abs(premium) * STT_RATES["options_sell"]
        return abs(premium) * STT_RATES["futures_sell"]

    @staticmethod
    def _estimate_slippage(price: float) -> float:
        """Estimate slippage as a fraction of price.

        Deeper OTM options have wider spreads, but without moneyness info
        we use a conservative 0.05% estimate.
        """
        return abs(price) * 0.0005

    def _update_unrealised(self, current_price: float) -> None:
        """Update unrealised PnL for all open positions."""
        for pos in self.positions:
            if pos.action in (Action.BUY_CALL, Action.BUY_PUT):
                pos.unrealised_pnl = current_price - pos.entry_price
            else:
                pos.unrealised_pnl = pos.entry_price - current_price

    @staticmethod
    def _pad_list(lst: Optional[List], target_len: int, fill_value) -> List:
        """Pad a list to ``target_len`` with ``fill_value``."""
        if lst is None:
            return [fill_value] * target_len
        if len(lst) >= target_len:
            return lst[:target_len]
        return lst + [fill_value] * (target_len - len(lst))
