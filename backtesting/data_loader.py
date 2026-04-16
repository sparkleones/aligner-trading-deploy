"""
Historical data loaders for backtesting.
Supports TrueData, TickData vendors, and local CSV files.
"""

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class TickRecord:
    """Single tick-level data point."""
    timestamp: datetime
    symbol: str
    last_price: float
    bid_price: float
    ask_price: float
    bid_qty: int
    ask_qty: int
    volume: int
    oi: float
    strike_price: float = 0.0
    option_type: str = ""  # "CE" or "PE"
    expiry: str = ""


@dataclass
class OHLCVRecord:
    """Single OHLCV bar."""
    timestamp: datetime
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    oi: float = 0.0


class DataLoader(ABC):
    """Abstract base class for historical data loaders."""

    @abstractmethod
    def load_ticks(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """Load tick-level data for a symbol and date range."""
        ...

    @abstractmethod
    def load_ohlcv(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval_minutes: int = 15,
    ) -> pd.DataFrame:
        """Load OHLCV bars for a symbol and date range."""
        ...

    @abstractmethod
    def load_option_chain(
        self,
        index_symbol: str,
        expiry_date: datetime,
        trade_date: datetime,
    ) -> pd.DataFrame:
        """Load full option chain snapshot for an expiry."""
        ...


class TrueDataLoader(DataLoader):
    """
    TrueData historical data loader.

    Connects to TrueData's API for research-quality tick and OHLCV data
    covering NSE indices and F&O instruments.
    """

    def __init__(
        self,
        username: str,
        password: str,
        cache_dir: str = "data/cache",
    ):
        self.username = username
        self.password = password
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._connection = None
        logger.info(
            "TrueDataLoader initialized | cache_dir=%s",
            self.cache_dir,
        )

    def _connect(self):
        """Establish connection to TrueData API."""
        if self._connection is not None:
            return
        try:
            from truedata_ws.TrueDataWebSocket import TrueDataWebSocket

            self._connection = TrueDataWebSocket(self.username, self.password)
            logger.info("TrueData connection established")
        except ImportError:
            logger.warning(
                "truedata_ws not installed — using cached/CSV data only"
            )
        except Exception as e:
            logger.error("TrueData connection failed: %s", e)

    def _cache_path(self, symbol: str, start: datetime, end: datetime, suffix: str) -> Path:
        """Generate a deterministic cache file path."""
        key = f"{symbol}_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}_{suffix}"
        return self.cache_dir / f"{key}.parquet"

    def load_ticks(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """Load tick data from TrueData or cache."""
        cache_file = self._cache_path(symbol, start_date, end_date, "ticks")

        if cache_file.exists():
            logger.info("Loading ticks from cache | file=%s", cache_file)
            return pd.read_parquet(cache_file)

        self._connect()
        if self._connection is None:
            logger.error("No TrueData connection — cannot load ticks for %s", symbol)
            return pd.DataFrame()

        try:
            t_start = datetime.now()
            bars = self._connection.get_historic_data(
                symbol,
                duration="tick",
                bar_size="tick",
                start_time=start_date,
                end_time=end_date,
            )
            df = pd.DataFrame(bars)
            df.to_parquet(cache_file)
            elapsed = (datetime.now() - t_start).total_seconds()
            logger.info(
                "Ticks loaded from TrueData | symbol=%s rows=%d latency=%.2fs",
                symbol, len(df), elapsed,
            )
            return df
        except Exception as e:
            logger.error("Failed loading ticks | symbol=%s error=%s", symbol, e)
            return pd.DataFrame()

    def load_ohlcv(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval_minutes: int = 15,
    ) -> pd.DataFrame:
        """Load OHLCV bars from TrueData or cache."""
        cache_file = self._cache_path(
            symbol, start_date, end_date, f"ohlcv_{interval_minutes}m"
        )

        if cache_file.exists():
            logger.info("Loading OHLCV from cache | file=%s", cache_file)
            return pd.read_parquet(cache_file)

        self._connect()
        if self._connection is None:
            logger.error("No TrueData connection — cannot load OHLCV for %s", symbol)
            return pd.DataFrame()

        try:
            t_start = datetime.now()
            bars = self._connection.get_historic_data(
                symbol,
                duration=f"{interval_minutes} min",
                bar_size=f"{interval_minutes} min",
                start_time=start_date,
                end_time=end_date,
            )
            df = pd.DataFrame(bars)
            if not df.empty:
                df.columns = [c.lower().strip() for c in df.columns]
                rename_map = {
                    "time": "timestamp",
                    "date": "timestamp",
                    "o": "open",
                    "h": "high",
                    "l": "low",
                    "c": "close",
                    "v": "volume",
                }
                df.rename(columns=rename_map, inplace=True, errors="ignore")
                if "timestamp" in df.columns:
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                    df.set_index("timestamp", inplace=True)
                    df.sort_index(inplace=True)

            df.to_parquet(cache_file)
            elapsed = (datetime.now() - t_start).total_seconds()
            logger.info(
                "OHLCV loaded from TrueData | symbol=%s interval=%dm rows=%d latency=%.2fs",
                symbol, interval_minutes, len(df), elapsed,
            )
            return df
        except Exception as e:
            logger.error("Failed loading OHLCV | symbol=%s error=%s", symbol, e)
            return pd.DataFrame()

    def load_option_chain(
        self,
        index_symbol: str,
        expiry_date: datetime,
        trade_date: datetime,
    ) -> pd.DataFrame:
        """Load option chain snapshot — constructs from individual strikes."""
        cache_file = self._cache_path(
            index_symbol, trade_date, expiry_date, "optchain"
        )

        if cache_file.exists():
            logger.info("Loading option chain from cache | file=%s", cache_file)
            return pd.read_parquet(cache_file)

        logger.warning(
            "Live option chain loading requires active TrueData subscription. "
            "Use CSVDataLoader for offline data."
        )
        return pd.DataFrame()


class CSVDataLoader(DataLoader):
    """
    Load historical data from local CSV files.

    Expected directory structure:
      data_dir/
        {symbol}/
          ticks_{YYYYMMDD}.csv
          ohlcv_{interval}m_{YYYYMMDD}_{YYYYMMDD}.csv
          optchain_{expiry}_{YYYYMMDD}.csv
    """

    def __init__(self, data_dir: str = "data/historical"):
        self.data_dir = Path(data_dir)
        logger.info("CSVDataLoader initialized | data_dir=%s", self.data_dir)

    def load_ticks(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """Load tick data from CSV files for the date range."""
        frames = []
        current = start_date.date()
        end = end_date.date()

        while current <= end:
            filename = self.data_dir / symbol / f"ticks_{current.strftime('%Y%m%d')}.csv"
            if filename.exists():
                df = pd.read_csv(filename, parse_dates=["timestamp"])
                frames.append(df)
            current += timedelta(days=1)

        if not frames:
            logger.warning("No tick CSVs found | symbol=%s range=%s to %s", symbol, start_date, end_date)
            return pd.DataFrame()

        result = pd.concat(frames, ignore_index=True)
        result.sort_values("timestamp", inplace=True)
        logger.info(
            "Ticks loaded from CSV | symbol=%s rows=%d",
            symbol, len(result),
        )
        return result

    def load_ohlcv(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval_minutes: int = 15,
    ) -> pd.DataFrame:
        """Load OHLCV bars from CSV."""
        pattern = f"ohlcv_{interval_minutes}m_*.csv"
        symbol_dir = self.data_dir / symbol

        if not symbol_dir.exists():
            logger.warning("No data directory for symbol %s", symbol)
            return pd.DataFrame()

        frames = []
        for f in sorted(symbol_dir.glob(pattern)):
            df = pd.read_csv(f, parse_dates=["timestamp"])
            mask = (df["timestamp"] >= start_date) & (df["timestamp"] <= end_date)
            frames.append(df.loc[mask])

        if not frames:
            # Try a single consolidated file
            single_file = symbol_dir / f"ohlcv_{interval_minutes}m.csv"
            if single_file.exists():
                df = pd.read_csv(single_file, parse_dates=["timestamp"])
                mask = (df["timestamp"] >= start_date) & (df["timestamp"] <= end_date)
                frames.append(df.loc[mask])

        if not frames:
            logger.warning(
                "No OHLCV CSVs found | symbol=%s interval=%dm", symbol, interval_minutes
            )
            return pd.DataFrame()

        result = pd.concat(frames, ignore_index=True)
        result.sort_values("timestamp", inplace=True)
        result.set_index("timestamp", inplace=True)
        logger.info(
            "OHLCV loaded from CSV | symbol=%s interval=%dm rows=%d",
            symbol, interval_minutes, len(result),
        )
        return result

    def load_option_chain(
        self,
        index_symbol: str,
        expiry_date: datetime,
        trade_date: datetime,
    ) -> pd.DataFrame:
        """Load option chain from CSV."""
        filename = (
            self.data_dir
            / index_symbol
            / f"optchain_{expiry_date.strftime('%Y%m%d')}_{trade_date.strftime('%Y%m%d')}.csv"
        )
        if not filename.exists():
            logger.warning("Option chain CSV not found: %s", filename)
            return pd.DataFrame()

        df = pd.read_csv(filename)
        logger.info(
            "Option chain loaded from CSV | index=%s expiry=%s rows=%d",
            index_symbol, expiry_date.date(), len(df),
        )
        return df


def generate_synthetic_ohlcv(
    symbol: str = "NIFTY",
    days: int = 252,
    interval_minutes: int = 15,
    base_price: float = 24000.0,
    volatility: float = 0.15,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic OHLCV data for backtesting when no real data is available.

    Uses geometric Brownian motion with realistic intraday patterns.
    """
    rng = np.random.default_rng(seed)
    bars_per_day = int((6 * 60 + 15) / interval_minutes)  # 9:15 to 15:30
    total_bars = days * bars_per_day

    dt = interval_minutes / (252 * 6.25 * 60)  # fraction of annual trading minutes
    daily_vol = volatility * np.sqrt(dt)

    prices = np.zeros(total_bars)
    prices[0] = base_price

    for i in range(1, total_bars):
        shock = rng.normal(0, daily_vol)
        prices[i] = prices[i - 1] * np.exp(shock)

    timestamps = []
    base_date = datetime(2025, 1, 1, 9, 15)
    current_day = 0
    bar_in_day = 0

    for i in range(total_bars):
        if bar_in_day >= bars_per_day:
            current_day += 1
            bar_in_day = 0
        day_offset = current_day
        # Skip weekends
        actual_date = base_date + timedelta(days=day_offset)
        while actual_date.weekday() >= 5:
            current_day += 1
            day_offset = current_day
            actual_date = base_date + timedelta(days=day_offset)

        ts = actual_date + timedelta(minutes=bar_in_day * interval_minutes)
        timestamps.append(ts)
        bar_in_day += 1

    # Generate OHLCV from prices
    intrabar_vol = daily_vol * 0.5
    opens = prices.copy()
    highs = prices * (1 + np.abs(rng.normal(0, intrabar_vol, total_bars)))
    lows = prices * (1 - np.abs(rng.normal(0, intrabar_vol, total_bars)))
    closes = prices + rng.normal(0, daily_vol * prices * 0.3, total_bars)
    closes = np.maximum(lows, np.minimum(highs, closes))
    volumes = rng.integers(50000, 500000, total_bars)

    df = pd.DataFrame({
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
        "oi": rng.integers(1000000, 5000000, total_bars),
    }, index=pd.DatetimeIndex(timestamps, name="timestamp"))

    df["symbol"] = symbol
    logger.info(
        "Synthetic OHLCV generated | symbol=%s days=%d bars=%d",
        symbol, days, len(df),
    )
    return df
