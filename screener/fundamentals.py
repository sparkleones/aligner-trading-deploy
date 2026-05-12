"""
Fundamentals fetcher via yfinance .info — cached on disk.

Note: .info is SNAPSHOT data (current values, not historical). So this
is fine for LIVE signal generation but unsuitable for backtest (look-
ahead bias). We only use it at live-pick time.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "fundamentals_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_TTL_HOURS = 24  # refresh once a day max


KEYS_TO_KEEP = [
    "trailingPE", "forwardPE", "priceToBook",
    "returnOnEquity", "returnOnAssets", "debtToEquity",
    "profitMargins", "operatingMargins", "grossMargins",
    "currentRatio", "quickRatio",
    "earningsGrowth", "revenueGrowth",
    "netIncomeToCommon", "operatingCashflow",
    "marketCap", "sector", "industry",
    "trailingEps", "forwardEps",
    "fiftyTwoWeekHigh", "fiftyTwoWeekLow",
    "dividendYield", "payoutRatio",
]


def _cache_path(symbol: str) -> Path:
    safe = symbol.replace("&", "_").replace("/", "_")
    return CACHE_DIR / f"{safe}.json"


def _cache_valid(path: Path) -> bool:
    if not path.exists():
        return False
    age_h = (time.time() - path.stat().st_mtime) / 3600
    return age_h < CACHE_TTL_HOURS


def fetch_fundamentals(nse_symbol: str, use_cache: bool = True) -> dict:
    """Fetch fundamentals dict for an NSE symbol. Cached 24h."""
    from .universe import to_yahoo_symbol

    path = _cache_path(nse_symbol)
    if use_cache and _cache_valid(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            pass

    try:
        import yfinance as yf
        ticker = yf.Ticker(to_yahoo_symbol(nse_symbol))
        info = ticker.info or {}
    except Exception as e:
        return {"_fetch_error": str(e)}

    # Slim to keep file small
    slim = {k: info.get(k) for k in KEYS_TO_KEEP if info.get(k) is not None}
    try:
        with open(path, "w") as f:
            json.dump(slim, f, indent=2, default=str)
    except Exception:
        pass
    return slim


def fetch_batch(symbols: list[str], use_cache: bool = True, progress: bool = False) -> dict[str, dict]:
    """Fetch fundamentals for a list of symbols. Returns dict {symbol: fundamentals}."""
    out = {}
    n = len(symbols)
    for i, sym in enumerate(symbols, 1):
        out[sym] = fetch_fundamentals(sym, use_cache=use_cache)
        if progress and (i % 10 == 0 or i == n):
            print(f"[fundamentals] {i}/{n}")
    return out
