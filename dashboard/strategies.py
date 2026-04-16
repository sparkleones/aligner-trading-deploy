"""
Multi-strategy comparison engine.
Runs 6 different strategies against the same data and returns comparative results.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from backtesting.backtest_engine import BacktestEngine, BacktestResult
from config.constants import (
    INDEX_CONFIG,
    STRADDLE_ENTRY_HOUR,
    STRADDLE_ENTRY_MINUTE,
    DDQN_STATE_DIM,
    DDQN_ACTION_DIM,
)
from strategy.features import FeatureEngine

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_LOT_SIZE = INDEX_CONFIG["NIFTY"]["lot_size"]  # 25
_STRIKE_INTERVAL = INDEX_CONFIG["NIFTY"]["strike_interval"]  # 50

_ENTRY_TIME_0920 = time(STRADDLE_ENTRY_HOUR, STRADDLE_ENTRY_MINUTE)
_ENTRY_TIME_0930 = time(9, 30)
_EXIT_TIME_1500 = time(15, 0)
_EXIT_TIME_1515 = time(15, 15)


def _bar_time(timestamp) -> time:
    """Extract time-of-day from a pandas Timestamp or datetime."""
    if isinstance(timestamp, pd.Timestamp):
        return timestamp.time()
    return timestamp.time()


def _atm_strike(spot: float) -> float:
    """Round spot to nearest ATM strike."""
    return round(spot / _STRIKE_INTERVAL) * _STRIKE_INTERVAL


def _otm_strike(spot: float, pct: float, option_type: str) -> float:
    """Return an OTM strike approximately *pct* away from spot."""
    if option_type == "CE":
        raw = spot * (1.0 + pct)
    else:
        raw = spot * (1.0 - pct)
    return round(raw / _STRIKE_INTERVAL) * _STRIKE_INTERVAL


def _simple_rsi(close_series: pd.Series, period: int = 14) -> float:
    """Compute latest RSI value on a 0-100 scale."""
    if len(close_series) < period + 1:
        return 50.0
    delta = close_series.diff()
    gain = delta.clip(lower=0).rolling(window=period).mean()
    loss = (-delta.clip(upper=0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-9)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    val = rsi.iloc[-1]
    return 50.0 if np.isnan(val) else float(val)


def _estimate_option_premium(
    spot: float, strike: float, option_type: str, bar_time=None
) -> float:
    """Synthetic premium estimate with intraday theta decay.

    At 09:15 (market open), ATM time value ≈ 2% of spot.
    By 15:30 (close), time value decays to ~15% of its opening value.
    This models realistic intraday theta capture for short premium sellers.
    """
    intrinsic = max(0.0, spot - strike) if option_type == "CE" else max(0.0, strike - spot)
    distance_pct = abs(spot - strike) / (spot + 1e-9)

    # Base time value at market open: ATM ~2% of spot, decays with distance
    base_tv = spot * 0.02 * max(0.0, 1.0 - distance_pct * 20.0)

    # Intraday theta decay: sqrt(time_remaining) decay curve
    if bar_time is not None:
        # Convert time to fraction of trading day remaining [1.0 at 9:15 → ~0.0 at 15:30]
        market_open_mins = 9 * 60 + 15   # 9:15
        market_close_mins = 15 * 60 + 30  # 15:30
        total_mins = market_close_mins - market_open_mins  # 375
        if hasattr(bar_time, 'hour'):
            current_mins = bar_time.hour * 60 + bar_time.minute
        else:
            current_mins = market_open_mins
        elapsed = max(0, current_mins - market_open_mins)
        time_remaining = max(0.04, 1.0 - elapsed / total_mins)
        # Premium decays with sqrt(time) — faster decay toward close
        theta_factor = time_remaining ** 0.5
    else:
        theta_factor = 1.0

    time_value = base_tv * theta_factor
    return max(intrinsic + time_value, 1.0)


def _simple_delta(spot: float, strike: float, option_type: str) -> float:
    """Quick delta approximation based on moneyness."""
    moneyness = (spot - strike) / (spot * 0.01 + 1e-9)
    if option_type == "CE":
        delta = 0.5 + 0.05 * moneyness
        return float(np.clip(delta, 0.01, 0.99))
    else:
        delta = -0.5 + 0.05 * moneyness
        return float(np.clip(delta, -0.99, -0.01))


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class BaseStrategy(ABC):
    """All strategies expose the same interface for BacktestEngine.run_strategy."""

    name: str = "base"

    def __init__(self) -> None:
        self._state: Dict[str, Any] = {}

    @abstractmethod
    def run(
        self,
        engine: BacktestEngine,
        bar_idx: int,
        row: pd.Series,
        spot_row: Optional[pd.Series],
    ) -> None:
        ...

    def reset(self) -> None:
        self._state.clear()


# ---------------------------------------------------------------------------
# 1. Short Straddle (0920 Straddle)
# ---------------------------------------------------------------------------


class ShortStraddleStrategy(BaseStrategy):
    """Enter short ATM straddle at 09:20; SL 30 % per leg; exit 15:15."""

    name = "short_straddle_0920"

    def __init__(self, sl_pct: float = 0.30) -> None:
        super().__init__()
        self.sl_pct = sl_pct

    def run(self, engine, bar_idx, row, spot_row):
        ts: pd.Timestamp = row.name if hasattr(row, "name") else row.get("timestamp", datetime.now())
        bt = _bar_time(ts)
        spot = float(row["close"])

        # --- Initialise per-day state ---
        day_key = ts.date() if hasattr(ts, "date") else ts
        if self._state.get("_day") != day_key:
            self._state.clear()
            self._state["_day"] = day_key
            self._state["entered"] = False
            self._state["ce_pos"] = None
            self._state["pe_pos"] = None
            self._state["ce_premium"] = 0.0
            self._state["pe_premium"] = 0.0
            self._state["exited"] = False

        if self._state["exited"]:
            return

        # --- Entry at first bar >= 09:20 ---
        if not self._state["entered"] and bt >= _ENTRY_TIME_0920:
            strike = _atm_strike(spot)
            ce_prem = _estimate_option_premium(spot, strike, "CE", bar_time=bt)
            pe_prem = _estimate_option_premium(spot, strike, "PE", bar_time=bt)

            ce_pos = engine.open_position(
                timestamp=ts, symbol=f"NIFTY_{int(strike)}_CE",
                side="SELL", quantity=_LOT_SIZE, price=ce_prem,
                spot_price=spot, option_type="CE", strike=strike,
                tag="straddle_ce",
            )
            pe_pos = engine.open_position(
                timestamp=ts, symbol=f"NIFTY_{int(strike)}_PE",
                side="SELL", quantity=_LOT_SIZE, price=pe_prem,
                spot_price=spot, option_type="PE", strike=strike,
                tag="straddle_pe",
            )
            self._state["entered"] = True
            self._state["ce_pos"] = ce_pos
            self._state["pe_pos"] = pe_pos
            self._state["ce_premium"] = ce_prem
            self._state["pe_premium"] = pe_prem
            logger.debug("ShortStraddle entry | strike=%s ce=%.2f pe=%.2f", strike, ce_prem, pe_prem)
            return

        if not self._state["entered"]:
            return

        # --- Stop-loss check per leg ---
        ce_pos = self._state["ce_pos"]
        pe_pos = self._state["pe_pos"]
        strike = ce_pos.strike if ce_pos else _atm_strike(spot)

        if ce_pos and ce_pos.is_open:
            current_ce = _estimate_option_premium(spot, strike, "CE", bar_time=bt)
            if current_ce >= self._state["ce_premium"] * (1.0 + self.sl_pct):
                engine.close_position(ce_pos, ts, current_ce, spot_price=spot)
                logger.debug("ShortStraddle CE SL hit at %.2f", current_ce)

        if pe_pos and pe_pos.is_open:
            current_pe = _estimate_option_premium(spot, strike, "PE", bar_time=bt)
            if current_pe >= self._state["pe_premium"] * (1.0 + self.sl_pct):
                engine.close_position(pe_pos, ts, current_pe, spot_price=spot)
                logger.debug("ShortStraddle PE SL hit at %.2f", current_pe)

        # --- Time-based exit at 15:15 ---
        if bt >= _EXIT_TIME_1515:
            if ce_pos and ce_pos.is_open:
                engine.close_position(ce_pos, ts, _estimate_option_premium(spot, strike, "CE", bar_time=bt), spot_price=spot)
            if pe_pos and pe_pos.is_open:
                engine.close_position(pe_pos, ts, _estimate_option_premium(spot, strike, "PE", bar_time=bt), spot_price=spot)
            self._state["exited"] = True


# ---------------------------------------------------------------------------
# 2. Delta-Neutral Strategy
# ---------------------------------------------------------------------------


class DeltaNeutralStrategy(BaseStrategy):
    """Short straddle at 09:20 with delta hedging when |net_delta| > 0.15."""

    name = "delta_neutral"

    def __init__(self, delta_threshold: float = 0.15) -> None:
        super().__init__()
        self.delta_threshold = delta_threshold
        self.hedge_count = 0

    def run(self, engine, bar_idx, row, spot_row):
        ts = row.name if hasattr(row, "name") else row.get("timestamp", datetime.now())
        bt = _bar_time(ts)
        spot = float(row["close"])

        day_key = ts.date() if hasattr(ts, "date") else ts
        if self._state.get("_day") != day_key:
            self._state.clear()
            self._state["_day"] = day_key
            self._state["entered"] = False
            self._state["ce_pos"] = None
            self._state["pe_pos"] = None
            self._state["hedge_positions"] = []
            self._state["exited"] = False
            self.hedge_count = 0

        if self._state["exited"]:
            return

        # --- Entry ---
        if not self._state["entered"] and bt >= _ENTRY_TIME_0920:
            strike = _atm_strike(spot)
            ce_prem = _estimate_option_premium(spot, strike, "CE", bar_time=bt)
            pe_prem = _estimate_option_premium(spot, strike, "PE", bar_time=bt)

            ce_pos = engine.open_position(
                timestamp=ts, symbol=f"NIFTY_{int(strike)}_CE",
                side="SELL", quantity=_LOT_SIZE, price=ce_prem,
                spot_price=spot, option_type="CE", strike=strike,
                tag="dn_ce",
            )
            pe_pos = engine.open_position(
                timestamp=ts, symbol=f"NIFTY_{int(strike)}_PE",
                side="SELL", quantity=_LOT_SIZE, price=pe_prem,
                spot_price=spot, option_type="PE", strike=strike,
                tag="dn_pe",
            )
            self._state["entered"] = True
            self._state["ce_pos"] = ce_pos
            self._state["pe_pos"] = pe_pos
            self._state["strike"] = strike
            return

        if not self._state["entered"]:
            return

        strike = self._state["strike"]

        # --- Delta monitoring & hedging ---
        ce_pos = self._state["ce_pos"]
        pe_pos = self._state["pe_pos"]

        net_delta = 0.0
        if ce_pos and ce_pos.is_open:
            net_delta += -_simple_delta(spot, strike, "CE") * _LOT_SIZE
        if pe_pos and pe_pos.is_open:
            net_delta += -_simple_delta(spot, strike, "PE") * _LOT_SIZE

        # Include hedge positions
        for hp in self._state["hedge_positions"]:
            if hp.is_open:
                sign = 1.0 if hp.side == "BUY" else -1.0
                net_delta += sign * hp.quantity * 1.0  # futures delta ~1

        if abs(net_delta) > self.delta_threshold * _LOT_SIZE and bt < _EXIT_TIME_1515:
            # Hedge with a futures-like position
            hedge_qty = int(abs(net_delta))
            hedge_side = "SELL" if net_delta > 0 else "BUY"
            hedge_pos = engine.open_position(
                timestamp=ts, symbol="NIFTY_FUT",
                side=hedge_side, quantity=max(1, hedge_qty),
                price=spot, spot_price=spot, option_type="",
                tag="dn_hedge",
            )
            if hedge_pos:
                self._state["hedge_positions"].append(hedge_pos)
                self.hedge_count += 1
                logger.debug("DeltaNeutral hedge #%d | side=%s qty=%d", self.hedge_count, hedge_side, hedge_qty)

        # --- Exit at 15:15 ---
        if bt >= _EXIT_TIME_1515:
            for pos_key in ("ce_pos", "pe_pos"):
                pos = self._state[pos_key]
                if pos and pos.is_open:
                    prem = _estimate_option_premium(spot, strike, pos.option_type, bar_time=bt) if pos.option_type else spot
                    engine.close_position(pos, ts, prem, spot_price=spot)
            for hp in self._state["hedge_positions"]:
                if hp.is_open:
                    engine.close_position(hp, ts, spot, spot_price=spot)
            self._state["exited"] = True


# ---------------------------------------------------------------------------
# 3. Bull Put Spread
# ---------------------------------------------------------------------------


class BullPutSpreadStrategy(BaseStrategy):
    """Enter bull put spread when RSI < 40 (oversold bounce expected)."""

    name = "bull_put_spread"

    def __init__(self, rsi_threshold: float = 40.0, strikes_below: int = 2) -> None:
        super().__init__()
        self.rsi_threshold = rsi_threshold
        self.strikes_below = strikes_below

    def run(self, engine, bar_idx, row, spot_row):
        ts = row.name if hasattr(row, "name") else row.get("timestamp", datetime.now())
        bt = _bar_time(ts)
        spot = float(row["close"])

        day_key = ts.date() if hasattr(ts, "date") else ts
        if self._state.get("_day") != day_key:
            self._state.clear()
            self._state["_day"] = day_key
            self._state["entered"] = False
            self._state["sell_pos"] = None
            self._state["buy_pos"] = None
            self._state["exited"] = False
            self._state["close_history"] = pd.Series(dtype=float)

        if self._state["exited"]:
            return

        # Accumulate close prices for RSI
        self._state["close_history"] = pd.concat(
            [self._state["close_history"], pd.Series([spot])],
            ignore_index=True,
        )

        if not self._state["entered"] and bt >= _ENTRY_TIME_0920:
            rsi = _simple_rsi(self._state["close_history"], period=14)
            if rsi < self.rsi_threshold:
                atm_strike = _atm_strike(spot)
                otm_strike = atm_strike - self.strikes_below * _STRIKE_INTERVAL

                sell_prem = _estimate_option_premium(spot, atm_strike, "PE", bar_time=bt)
                buy_prem = _estimate_option_premium(spot, otm_strike, "PE", bar_time=bt)

                sell_pos = engine.open_position(
                    timestamp=ts, symbol=f"NIFTY_{int(atm_strike)}_PE",
                    side="SELL", quantity=_LOT_SIZE, price=sell_prem,
                    spot_price=spot, option_type="PE", strike=atm_strike,
                    tag="bps_sell",
                )
                buy_pos = engine.open_position(
                    timestamp=ts, symbol=f"NIFTY_{int(otm_strike)}_PE",
                    side="BUY", quantity=_LOT_SIZE, price=buy_prem,
                    spot_price=spot, option_type="PE", strike=otm_strike,
                    tag="bps_buy",
                )
                self._state["entered"] = True
                self._state["sell_pos"] = sell_pos
                self._state["buy_pos"] = buy_pos
                self._state["net_premium"] = sell_prem - buy_prem
                self._state["max_loss"] = (atm_strike - otm_strike) - (sell_prem - buy_prem)
                self._state["sell_strike"] = atm_strike
                self._state["buy_strike"] = otm_strike
                logger.debug(
                    "BullPutSpread entry | sell=%s buy=%s net_prem=%.2f rsi=%.1f",
                    atm_strike, otm_strike, self._state["net_premium"], rsi,
                )
                return

        if not self._state["entered"]:
            return

        # --- Exit at end of day ---
        if bt >= _EXIT_TIME_1515:
            sell_pos = self._state["sell_pos"]
            buy_pos = self._state["buy_pos"]
            if sell_pos and sell_pos.is_open:
                prem = _estimate_option_premium(spot, self._state["sell_strike"], "PE", bar_time=bt)
                engine.close_position(sell_pos, ts, prem, spot_price=spot)
            if buy_pos and buy_pos.is_open:
                prem = _estimate_option_premium(spot, self._state["buy_strike"], "PE", bar_time=bt)
                engine.close_position(buy_pos, ts, prem, spot_price=spot)
            self._state["exited"] = True


# ---------------------------------------------------------------------------
# 4. Iron Condor
# ---------------------------------------------------------------------------


class IronCondorStrategy(BaseStrategy):
    """Enter iron condor at 09:30; exit 15:00 or 2x premium SL per leg."""

    name = "iron_condor"

    def __init__(self, inner_pct: float = 0.015, outer_pct: float = 0.03, sl_mult: float = 2.0) -> None:
        super().__init__()
        self.inner_pct = inner_pct
        self.outer_pct = outer_pct
        self.sl_mult = sl_mult

    def run(self, engine, bar_idx, row, spot_row):
        ts = row.name if hasattr(row, "name") else row.get("timestamp", datetime.now())
        bt = _bar_time(ts)
        spot = float(row["close"])

        day_key = ts.date() if hasattr(ts, "date") else ts
        if self._state.get("_day") != day_key:
            self._state.clear()
            self._state["_day"] = day_key
            self._state["entered"] = False
            self._state["positions"] = {}
            self._state["premiums"] = {}
            self._state["strikes"] = {}
            self._state["exited"] = False

        if self._state["exited"]:
            return

        # --- Entry at 09:30 ---
        if not self._state["entered"] and bt >= _ENTRY_TIME_0930:
            sell_ce_strike = _otm_strike(spot, self.inner_pct, "CE")
            sell_pe_strike = _otm_strike(spot, self.inner_pct, "PE")
            buy_ce_strike = _otm_strike(spot, self.outer_pct, "CE")
            buy_pe_strike = _otm_strike(spot, self.outer_pct, "PE")

            legs = [
                ("sell_ce", sell_ce_strike, "CE", "SELL"),
                ("sell_pe", sell_pe_strike, "PE", "SELL"),
                ("buy_ce", buy_ce_strike, "CE", "BUY"),
                ("buy_pe", buy_pe_strike, "PE", "BUY"),
            ]

            for leg_name, strike, otype, side in legs:
                prem = _estimate_option_premium(spot, strike, otype, bar_time=bt)
                pos = engine.open_position(
                    timestamp=ts, symbol=f"NIFTY_{int(strike)}_{otype}",
                    side=side, quantity=_LOT_SIZE, price=prem,
                    spot_price=spot, option_type=otype, strike=strike,
                    tag=f"ic_{leg_name}",
                )
                self._state["positions"][leg_name] = pos
                self._state["premiums"][leg_name] = prem
                self._state["strikes"][leg_name] = strike

            self._state["entered"] = True
            logger.debug(
                "IronCondor entry | sell_ce=%s sell_pe=%s buy_ce=%s buy_pe=%s",
                sell_ce_strike, sell_pe_strike, buy_ce_strike, buy_pe_strike,
            )
            return

        if not self._state["entered"]:
            return

        # --- Stop-loss check on sold legs (2x premium) ---
        for leg_name in ("sell_ce", "sell_pe"):
            pos = self._state["positions"].get(leg_name)
            if pos and pos.is_open:
                strike = self._state["strikes"][leg_name]
                otype = "CE" if "ce" in leg_name else "PE"
                current_prem = _estimate_option_premium(spot, strike, otype, bar_time=bt)
                if current_prem >= self._state["premiums"][leg_name] * self.sl_mult:
                    engine.close_position(pos, ts, current_prem, spot_price=spot)
                    # Also close the corresponding hedge
                    hedge_key = leg_name.replace("sell", "buy")
                    hedge_pos = self._state["positions"].get(hedge_key)
                    if hedge_pos and hedge_pos.is_open:
                        h_strike = self._state["strikes"][hedge_key]
                        h_prem = _estimate_option_premium(spot, h_strike, otype, bar_time=bt)
                        engine.close_position(hedge_pos, ts, h_prem, spot_price=spot)
                    logger.debug("IronCondor %s SL hit at %.2f", leg_name, current_prem)

        # --- Time exit at 15:00 ---
        if bt >= _EXIT_TIME_1500:
            for leg_name, pos in self._state["positions"].items():
                if pos and pos.is_open:
                    strike = self._state["strikes"][leg_name]
                    otype = "CE" if "ce" in leg_name else "PE"
                    prem = _estimate_option_premium(spot, strike, otype, bar_time=bt)
                    engine.close_position(pos, ts, prem, spot_price=spot)
            self._state["exited"] = True


# ---------------------------------------------------------------------------
# 5. Pairs Trade (Nifty vs BankNifty stat arb)
# ---------------------------------------------------------------------------


class PairsTradeStrategy(BaseStrategy):
    """Stat-arb between Nifty and BankNifty using rolling z-score of spread."""

    name = "pairs_nifty_banknifty"

    def __init__(
        self,
        z_entry: float = 2.0,
        z_exit: float = 0.5,
        lookback: int = 60,
    ) -> None:
        super().__init__()
        self.z_entry = z_entry
        self.z_exit = z_exit
        self.lookback = lookback

    def run(self, engine, bar_idx, row, spot_row):
        ts = row.name if hasattr(row, "name") else row.get("timestamp", datetime.now())
        bt = _bar_time(ts)
        spot = float(row["close"])

        # BankNifty price from spot_row or fallback to a ratio
        bn_price = float(spot_row["close"]) if spot_row is not None else spot * 2.2

        day_key = ts.date() if hasattr(ts, "date") else ts
        if self._state.get("_day") != day_key:
            # Preserve spread history across days for rolling window
            if "spread_history" not in self._state:
                self._state["spread_history"] = []
            self._state["_day"] = day_key
            self._state["position_side"] = None
            self._state["nifty_pos"] = None
            self._state["bn_pos"] = None

        # Track spread = BankNifty / Nifty ratio
        ratio = bn_price / (spot + 1e-9)
        self._state.setdefault("spread_history", []).append(ratio)

        if len(self._state["spread_history"]) < self.lookback:
            return

        window = np.array(self._state["spread_history"][-self.lookback:])
        mean = window.mean()
        std = window.std()
        if std < 1e-9:
            return
        z_score = (ratio - mean) / std

        # --- Entry ---
        if self._state.get("position_side") is None and bt < _EXIT_TIME_1500:
            if z_score > self.z_entry:
                # Spread too high: short BN, long Nifty
                nifty_pos = engine.open_position(
                    timestamp=ts, symbol="NIFTY_FUT", side="BUY",
                    quantity=_LOT_SIZE, price=spot, spot_price=spot,
                    tag="pairs_nifty_long",
                )
                bn_pos = engine.open_position(
                    timestamp=ts, symbol="BANKNIFTY_FUT", side="SELL",
                    quantity=INDEX_CONFIG["BANKNIFTY"]["lot_size"],
                    price=bn_price, spot_price=bn_price,
                    tag="pairs_bn_short",
                )
                self._state["position_side"] = "short_spread"
                self._state["nifty_pos"] = nifty_pos
                self._state["bn_pos"] = bn_pos
                logger.debug("PairsTrade ENTRY short_spread | z=%.2f", z_score)

            elif z_score < -self.z_entry:
                # Spread too low: long BN, short Nifty
                nifty_pos = engine.open_position(
                    timestamp=ts, symbol="NIFTY_FUT", side="SELL",
                    quantity=_LOT_SIZE, price=spot, spot_price=spot,
                    tag="pairs_nifty_short",
                )
                bn_pos = engine.open_position(
                    timestamp=ts, symbol="BANKNIFTY_FUT", side="BUY",
                    quantity=INDEX_CONFIG["BANKNIFTY"]["lot_size"],
                    price=bn_price, spot_price=bn_price,
                    tag="pairs_bn_long",
                )
                self._state["position_side"] = "long_spread"
                self._state["nifty_pos"] = nifty_pos
                self._state["bn_pos"] = bn_pos
                logger.debug("PairsTrade ENTRY long_spread | z=%.2f", z_score)

        # --- Exit when z-score reverts or at EOD ---
        elif self._state.get("position_side") is not None:
            should_exit = bt >= _EXIT_TIME_1500 or abs(z_score) < self.z_exit
            if should_exit:
                nifty_pos = self._state["nifty_pos"]
                bn_pos = self._state["bn_pos"]
                if nifty_pos and nifty_pos.is_open:
                    engine.close_position(nifty_pos, ts, spot, spot_price=spot)
                if bn_pos and bn_pos.is_open:
                    engine.close_position(bn_pos, ts, bn_price, spot_price=bn_price)
                logger.debug("PairsTrade EXIT | z=%.2f side=%s", z_score, self._state["position_side"])
                self._state["position_side"] = None
                self._state["nifty_pos"] = None
                self._state["bn_pos"] = None


# ---------------------------------------------------------------------------
# 6. DDQN Strategy
# ---------------------------------------------------------------------------


class DDQNStrategy(BaseStrategy):
    """Uses the DDQN agent for decision-making; falls back to random policy."""

    name = "ddqn_agent"

    # Action mapping: 0=hold, 1=buy_call, 2=buy_put, 3=sell_call, 4=sell_put
    _ACTION_HOLD = 0
    _ACTION_BUY_CE = 1
    _ACTION_BUY_PE = 2
    _ACTION_SELL_CE = 3
    _ACTION_SELL_PE = 4

    def __init__(self, model_path: Optional[str] = None) -> None:
        super().__init__()
        self.feature_engine = FeatureEngine(state_dim=DDQN_STATE_DIM)
        self.agent = None
        self.model_loaded = False
        self.agent_confidence = 0.0
        self.epsilon = 1.0
        self._ohlcv_buffer: List[dict] = []

        self._try_load_agent(model_path)

    def _try_load_agent(self, model_path: Optional[str]) -> None:
        """Attempt to load a trained DDQN agent."""
        try:
            from strategy.ddqn_agent import DDQNAgent
            self.agent = DDQNAgent(
                state_dim=DDQN_STATE_DIM,
                action_dim=DDQN_ACTION_DIM,
            )
            if model_path and Path(model_path).exists():
                self.agent.load(model_path)
                self.model_loaded = True
                self.epsilon = self.agent.epsilon
                logger.info("DDQNStrategy loaded model from %s", model_path)
            else:
                # Check default location
                default_path = Path("models/ddqn_latest.pt")
                if default_path.exists():
                    self.agent.load(str(default_path))
                    self.model_loaded = True
                    self.epsilon = self.agent.epsilon
                    logger.info("DDQNStrategy loaded model from default path")
                else:
                    logger.warning("DDQNStrategy: no model found, using random policy")
        except Exception as exc:
            logger.warning("DDQNStrategy: could not initialise agent (%s), using random policy", exc)
            self.agent = None

    def run(self, engine, bar_idx, row, spot_row):
        ts = row.name if hasattr(row, "name") else row.get("timestamp", datetime.now())
        bt = _bar_time(ts)
        spot = float(row["close"])

        day_key = ts.date() if hasattr(ts, "date") else ts
        if self._state.get("_day") != day_key:
            self._state.clear()
            self._state["_day"] = day_key
            self._state["positions"] = []
            self._ohlcv_buffer.clear()

        # Accumulate OHLCV for feature computation
        bar_dict = {
            "open": float(row.get("open", spot)),
            "high": float(row.get("high", spot)),
            "low": float(row.get("low", spot)),
            "close": spot,
            "volume": float(row.get("volume", 0)),
        }
        self._ohlcv_buffer.append(bar_dict)

        # Need at least a few bars for features
        if len(self._ohlcv_buffer) < 5:
            return

        # Build feature vector
        ohlcv_df = pd.DataFrame(self._ohlcv_buffer)
        state = self.feature_engine.compute(ohlcv_df)

        # Select action
        if self.agent is not None:
            action = self.agent.select_action(state, explore=not self.model_loaded)
            # Compute confidence as max Q-value spread
            try:
                import torch
                with torch.no_grad():
                    state_t = torch.FloatTensor(state).unsqueeze(0).to(self.agent.device)
                    q_values = self.agent.online_net(state_t).cpu().numpy()[0]
                    q_range = q_values.max() - q_values.min()
                    self.agent_confidence = float(np.clip(q_range, 0, 1))
            except Exception:
                self.agent_confidence = 0.0
            self.epsilon = self.agent.epsilon
        else:
            action = np.random.randint(0, DDQN_ACTION_DIM)
            self.agent_confidence = 0.0

        # Don't trade near close
        if bt >= _EXIT_TIME_1515:
            for pos in self._state["positions"]:
                if pos and pos.is_open:
                    prem = _estimate_option_premium(spot, pos.strike, pos.option_type, bar_time=bt) if pos.option_type else spot
                    engine.close_position(pos, ts, prem, spot_price=spot)
            return

        # Execute action
        if action == self._ACTION_HOLD:
            return

        strike = _atm_strike(spot)

        if action == self._ACTION_BUY_CE:
            prem = _estimate_option_premium(spot, strike, "CE", bar_time=bt)
            pos = engine.open_position(
                timestamp=ts, symbol=f"NIFTY_{int(strike)}_CE",
                side="BUY", quantity=_LOT_SIZE, price=prem,
                spot_price=spot, option_type="CE", strike=strike,
                tag="ddqn_buy_ce",
            )
            if pos:
                self._state["positions"].append(pos)

        elif action == self._ACTION_BUY_PE:
            prem = _estimate_option_premium(spot, strike, "PE", bar_time=bt)
            pos = engine.open_position(
                timestamp=ts, symbol=f"NIFTY_{int(strike)}_PE",
                side="BUY", quantity=_LOT_SIZE, price=prem,
                spot_price=spot, option_type="PE", strike=strike,
                tag="ddqn_buy_pe",
            )
            if pos:
                self._state["positions"].append(pos)

        elif action == self._ACTION_SELL_CE:
            prem = _estimate_option_premium(spot, strike, "CE", bar_time=bt)
            pos = engine.open_position(
                timestamp=ts, symbol=f"NIFTY_{int(strike)}_CE",
                side="SELL", quantity=_LOT_SIZE, price=prem,
                spot_price=spot, option_type="CE", strike=strike,
                tag="ddqn_sell_ce",
            )
            if pos:
                self._state["positions"].append(pos)

        elif action == self._ACTION_SELL_PE:
            prem = _estimate_option_premium(spot, strike, "PE", bar_time=bt)
            pos = engine.open_position(
                timestamp=ts, symbol=f"NIFTY_{int(strike)}_PE",
                side="SELL", quantity=_LOT_SIZE, price=prem,
                spot_price=spot, option_type="PE", strike=strike,
                tag="ddqn_sell_pe",
            )
            if pos:
                self._state["positions"].append(pos)


# ---------------------------------------------------------------------------
# Strategy registry
# ---------------------------------------------------------------------------

ALL_STRATEGIES: List[BaseStrategy] = [
    ShortStraddleStrategy(),
    DeltaNeutralStrategy(),
    BullPutSpreadStrategy(),
    IronCondorStrategy(),
    PairsTradeStrategy(),
    DDQNStrategy(),
]


# ---------------------------------------------------------------------------
# Main comparison runner
# ---------------------------------------------------------------------------


def run_all_strategies(
    data: pd.DataFrame,
    initial_capital: float = 1_000_000.0,
    nifty_data: pd.DataFrame = None,
    banknifty_data: pd.DataFrame = None,
) -> dict:
    """Run ALL 6 strategies against the same data and return comparative results.

    Parameters
    ----------
    data : pd.DataFrame
        OHLCV DataFrame with DatetimeIndex (the primary instrument data, typically Nifty).
    initial_capital : float
        Starting capital for each strategy engine.
    nifty_data : pd.DataFrame, optional
        Nifty spot data.  If *None*, ``data`` is used as the Nifty source.
    banknifty_data : pd.DataFrame, optional
        BankNifty spot data.  Required for PairsTradeStrategy to work properly.

    Returns
    -------
    dict
        ``strategy_name -> { result, trades, equity_curve, daily_pnl, metadata }``
    """
    if nifty_data is None:
        nifty_data = data

    results: Dict[str, dict] = {}
    strategies = [
        ShortStraddleStrategy(),
        DeltaNeutralStrategy(),
        BullPutSpreadStrategy(),
        IronCondorStrategy(),
        PairsTradeStrategy(),
        DDQNStrategy(),
    ]

    for strategy in strategies:
        logger.info("Running strategy: %s", strategy.name)
        strategy.reset()

        engine = BacktestEngine(
            initial_capital=initial_capital,
            brokerage_per_order=20.0,
        )

        # PairsTradeStrategy needs BankNifty as spot_data
        spot_data = banknifty_data if isinstance(strategy, PairsTradeStrategy) else None

        engine.run_strategy(data, strategy.run, spot_data=spot_data)
        result = engine.get_results()

        # Build trade list with timestamps
        trade_dicts = []
        for t in result.trades:
            trade_dicts.append({
                "entry_time": t.entry_time.isoformat() if hasattr(t.entry_time, "isoformat") else str(t.entry_time),
                "exit_time": t.exit_time.isoformat() if hasattr(t.exit_time, "isoformat") else str(t.exit_time),
                "symbol": t.symbol,
                "side": t.side,
                "quantity": t.quantity,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "gross_pnl": t.gross_pnl,
                "net_pnl": t.net_pnl,
                "transaction_costs": t.transaction_costs,
                "hold_duration_minutes": t.hold_duration_minutes,
                "tag": t.tag,
            })

        # Strategy-specific metadata
        metadata: Dict[str, Any] = {"strategy_class": strategy.__class__.__name__}
        if isinstance(strategy, DeltaNeutralStrategy):
            metadata["hedge_adjustments"] = strategy.hedge_count
        if isinstance(strategy, DDQNStrategy):
            metadata["model_loaded"] = strategy.model_loaded
            metadata["agent_confidence"] = strategy.agent_confidence
            metadata["epsilon"] = strategy.epsilon
        if isinstance(strategy, ShortStraddleStrategy):
            metadata["sl_pct"] = strategy.sl_pct
        if isinstance(strategy, IronCondorStrategy):
            metadata["inner_pct"] = strategy.inner_pct
            metadata["outer_pct"] = strategy.outer_pct

        results[strategy.name] = {
            "result": result,
            "trades": trade_dicts,
            "equity_curve": list(result.equity_curve),
            "daily_pnl": list(result.daily_pnl),
            "metadata": metadata,
        }

        logger.info(
            "Strategy %s done | trades=%d net_pnl=%.2f sharpe=%.4f",
            strategy.name, result.total_trades, result.net_pnl, result.sharpe_ratio,
        )

    return results


# ---------------------------------------------------------------------------
# Comparison report generator
# ---------------------------------------------------------------------------


def generate_comparison_report(results: dict) -> dict:
    """Produce a summary table comparing all strategies.

    Parameters
    ----------
    results : dict
        Output of :func:`run_all_strategies`.

    Returns
    -------
    dict
        ``{ "strategies": [...], "best": { metric: strategy_name, ... } }``
        Each strategy entry has: net_pnl, win_rate, sharpe_ratio, profit_factor,
        max_drawdown, total_trades, avg_hold_time, transaction_costs,
        kill_switch_triggers.
    """
    rows = []

    for name, data in results.items():
        r: BacktestResult = data["result"]
        rows.append({
            "strategy": name,
            "net_pnl": round(r.net_pnl, 2),
            "win_rate": round(r.win_rate * 100, 2),
            "sharpe_ratio": round(r.sharpe_ratio, 4),
            "profit_factor": round(r.profit_factor, 2),
            "max_drawdown": round(r.max_drawdown, 2),
            "max_drawdown_pct": round(r.max_drawdown_pct * 100, 2),
            "total_trades": r.total_trades,
            "avg_hold_time_min": round(r.avg_hold_minutes, 1),
            "transaction_costs": round(r.total_transaction_costs, 2),
            "slippage_costs": round(r.total_slippage_costs, 2),
            "kill_switch_triggers": r.kill_switch_triggers,
            "winning_trades": r.winning_trades,
            "losing_trades": r.losing_trades,
            "gross_pnl": round(r.gross_pnl, 2),
        })

    # Determine best strategy per metric
    best: Dict[str, str] = {}
    if rows:
        metrics_higher_better = ["net_pnl", "win_rate", "sharpe_ratio", "profit_factor"]
        metrics_lower_better = ["max_drawdown_pct", "transaction_costs", "kill_switch_triggers"]

        for metric in metrics_higher_better:
            best[metric] = max(rows, key=lambda r: r[metric])["strategy"]
        for metric in metrics_lower_better:
            # For drawdown and costs, lower absolute value is better
            best[metric] = min(rows, key=lambda r: abs(r[metric]))["strategy"]

    report = {
        "strategies": rows,
        "best": best,
        "generated_at": datetime.now().isoformat(),
        "num_strategies": len(rows),
    }

    logger.info("Comparison report generated for %d strategies", len(rows))
    for row in rows:
        logger.info(
            "  %-25s | PnL=%10.2f | WR=%5.1f%% | Sharpe=%7.4f | PF=%5.2f | DD=%6.2f%% | Trades=%d",
            row["strategy"], row["net_pnl"], row["win_rate"],
            row["sharpe_ratio"], row["profit_factor"],
            row["max_drawdown_pct"], row["total_trades"],
        )

    return report
