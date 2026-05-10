"""
OHLCV data loader for the screener.

Primary source: yfinance (free, daily granularity, ~2 yrs reliable history
for Indian equities via .NS suffix). Cached on disk to avoid repeat
downloads.

For LIVE trading we would switch to Kite Connect historical_data() since
yfinance has known delays and gaps. For BACKTESTING, yfinance is fine.
"""
from __future__ import annotations

import os
import pickle
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "screener_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Re-download if cache older than this
CACHE_MAX_AGE_HOURS = 6


def _cache_path(symbol: str, period: str) -> Path:
    safe = symbol.replace("&", "_").replace("/", "_")
    return CACHE_DIR / f"{safe}__{period}.pkl"


def _cache_valid(path: Path, max_age_hours: int = CACHE_MAX_AGE_HOURS) -> bool:
    if not path.exists():
        return False
    age_sec = time.time() - path.stat().st_mtime
    return age_sec < max_age_hours * 3600


def load_history(
    nse_symbol: str,
    period: str = "3y",
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Fetch OHLCV history for an NSE symbol. Returns DataFrame with
    columns Open/High/Low/Close/Volume indexed by date. Empty df on failure.
    """
    from .universe import to_yahoo_symbol

    yahoo_sym = to_yahoo_symbol(nse_symbol)
    path = _cache_path(nse_symbol, period)

    if use_cache and _cache_valid(path):
        try:
            with open(path, "rb") as f:
                df = pickle.load(f)
            if isinstance(df, pd.DataFrame) and not df.empty:
                return df
        except Exception:
            pass

    try:
        import yfinance as yf
        ticker = yf.Ticker(yahoo_sym)
        df = ticker.history(period=period, auto_adjust=True, raise_errors=False)
    except Exception as e:
        print(f"[data_loader] {nse_symbol}: fetch failed — {e}")
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    # Drop tz info (yfinance returns tz-aware index — messes up alignment)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    # Standardize column names
    df = df.rename(columns={
        "Adj Close": "Close",  # auto_adjust=True already gives adjusted close
    })
    keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[keep].dropna()

    if not df.empty:
        try:
            with open(path, "wb") as f:
                pickle.dump(df, f)
        except Exception:
            pass

    return df


def load_universe(
    symbols: list[str],
    period: str = "3y",
    use_cache: bool = True,
    progress: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Fetch OHLCV for an entire universe. Returns dict {symbol: df}.
    Skips symbols that fail to fetch.
    """
    out = {}
    n = len(symbols)
    for i, sym in enumerate(symbols, 1):
        df = load_history(sym, period=period, use_cache=use_cache)
        if not df.empty and len(df) >= 252:
            out[sym] = df
        if progress and (i % 10 == 0 or i == n):
            print(f"[data_loader] {i}/{n} loaded ({len(out)} valid)")
    return out
