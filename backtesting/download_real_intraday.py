"""
Download REAL intraday candle data from Kite Connect or Yahoo Finance.

This replaces the fake generate_intraday_path() with actual market data.
Downloads NIFTY 50 and India VIX candles for any date range.

Kite Connect limits:
  - Max 2000 candles per request
  - 1-min bars:  ~375 bars/day -> ~5 trading days per chunk
  - 15-min bars: ~25 bars/day  -> ~80 trading days per chunk
  - Rate limit: 3 requests/second

Usage:
  python backtesting/download_real_intraday.py --source kite --interval minute --start 2025-10-01 --end 2026-04-06
  python backtesting/download_real_intraday.py --source yahoo  (last 60 days only)

Saves to: data/historical/nifty_{interval}_{start}_{end}.csv
"""

import argparse
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def download_from_kite(start_date: str, end_date: str, interval: str = "15minute"):
    """Download real intraday data from Kite Connect.

    Auto-loads credentials from .env file via config/settings.py.
    """
    from config.settings import load_settings

    # Load credentials from .env
    settings = load_settings()
    if not settings.broker.api_key:
        print("ERROR: BROKER_API_KEY not found in .env file!")
        print("Check that .env exists at project root with Kite credentials.")
        sys.exit(1)

    print(f"Kite credentials loaded: user={settings.broker.user_id}")

    from broker.kite_connect import KiteConnectBroker

    broker = KiteConnectBroker(
        api_key=settings.broker.api_key,
        api_secret=settings.broker.api_secret,
        user_id=settings.broker.user_id,
        password=settings.broker.password,
        totp_secret=settings.broker.totp_secret,
    )

    # CRITICAL: Must authenticate before making API calls
    # KiteConnectBroker constructor does NOT auto-authenticate
    print("Authenticating with Kite Connect...")
    if not broker.authenticate():
        print("ERROR: Kite authentication failed!")
        print("Make sure your TOTP secret and credentials in .env are correct.")
        print("Also check if market hours / Kite servers are accessible.")
        sys.exit(1)
    print("Authentication successful!")

    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    all_nifty_bars = []
    all_vix_bars = []

    # Kite allows max ~2000 candles per request
    # Chunk size depends on interval:
    #   1-min:   375 bars/day -> 5 days per chunk
    #   5-min:   75 bars/day  -> 26 days per chunk
    #   15-min:  25 bars/day  -> 80 days per chunk
    if interval == "minute":
        chunk_days = 5
    elif interval == "3minute":
        chunk_days = 15
    elif interval == "5minute":
        chunk_days = 25
    elif interval == "15minute":
        chunk_days = 60
    elif interval == "30minute":
        chunk_days = 100
    else:
        chunk_days = 60

    current = start
    total_days = (end - start).days
    chunks_needed = max(1, total_days // chunk_days + 1)

    print(f"Downloading {interval} data from Kite Connect...")
    print(f"Period: {start_date} to {end_date} ({total_days} days)")
    print(f"Chunk size: {chunk_days} days ({chunks_needed} chunks)")
    print("-" * 60)

    chunk_num = 0
    while current < end:
        chunk_end = min(current + timedelta(days=chunk_days), end)
        chunk_num += 1

        # Download NIFTY 50
        print(f"  [{chunk_num}/{chunks_needed}] NIFTY 50: {current.date()} to {chunk_end.date()}...", end=" ", flush=True)
        try:
            nifty_bars = broker.get_historical_data(
                symbol="NIFTY 50",
                from_dt=current,
                to_dt=chunk_end,
                interval=interval,
            )
            print(f"{len(nifty_bars)} bars")
            all_nifty_bars.extend(nifty_bars)
        except Exception as e:
            print(f"FAILED: {e}")
        time.sleep(0.4)  # Rate limit

        # Download India VIX (use day interval - VIX doesn't need intraday)
        vix_interval = "day" if interval == "minute" else interval
        print(f"  [{chunk_num}/{chunks_needed}] INDIA VIX: {current.date()} to {chunk_end.date()}...", end=" ", flush=True)
        try:
            vix_bars = broker.get_historical_data(
                symbol="INDIA VIX",
                from_dt=current,
                to_dt=chunk_end,
                interval=vix_interval,
            )
            print(f"{len(vix_bars)} bars")
            all_vix_bars.extend(vix_bars)
        except Exception as e:
            print(f"FAILED: {e}")
        time.sleep(0.4)

        current = chunk_end + timedelta(days=1)

    if not all_nifty_bars:
        print("ERROR: No NIFTY data downloaded!")
        return None, None

    # Convert to DataFrames
    nifty_df = pd.DataFrame(all_nifty_bars)
    nifty_df["timestamp"] = pd.to_datetime(nifty_df["time"])
    nifty_df = nifty_df.set_index("timestamp").sort_index()
    nifty_df = nifty_df[~nifty_df.index.duplicated(keep="first")]

    vix_df = pd.DataFrame(all_vix_bars) if all_vix_bars else pd.DataFrame()
    if not vix_df.empty:
        vix_df["timestamp"] = pd.to_datetime(vix_df["time"])
        vix_df = vix_df.set_index("timestamp").sort_index()
        vix_df = vix_df[~vix_df.index.duplicated(keep="first")]

    # Save to CSV
    data_dir = project_root / "data" / "historical"
    data_dir.mkdir(parents=True, exist_ok=True)

    interval_tag = interval.replace("minute", "min")
    nifty_path = data_dir / f"nifty_{interval_tag}_{start_date}_{end_date}.csv"
    nifty_df.to_csv(nifty_path)
    print(f"\nSaved NIFTY: {nifty_path} ({len(nifty_df)} bars)")

    if not vix_df.empty:
        vix_path = data_dir / f"vix_{interval_tag}_{start_date}_{end_date}.csv"
        vix_df.to_csv(vix_path)
        print(f"Saved VIX:   {vix_path} ({len(vix_df)} bars)")

    # Summary
    trading_days = nifty_df.index.date
    unique_days = len(set(trading_days))
    print(f"\nSummary:")
    print(f"  Trading days: {unique_days}")
    print(f"  NIFTY range: {nifty_df['close'].min():.0f} - {nifty_df['close'].max():.0f}")
    print(f"  Bars per day: ~{len(nifty_df) / max(unique_days, 1):.0f}")

    return nifty_df, vix_df


def download_from_yfinance(start_date: str, end_date: str):
    """Fallback: Download from Yahoo Finance (limited to 60 days of 15-min data)."""
    import yfinance as yf

    print(f"Downloading 15-min data from Yahoo Finance...")
    print(f"NOTE: Yahoo only provides 60 days of intraday data!")
    print(f"Period requested: {start_date} to {end_date}")
    print("-" * 60)

    nifty = yf.download("^NSEI", start=start_date, end=end_date,
                         interval="15m", progress=True)

    if nifty.empty:
        print("ERROR: No data from Yahoo Finance (likely > 60 day limit)")
        return None, None

    if isinstance(nifty.columns, pd.MultiIndex):
        nifty.columns = nifty.columns.get_level_values(0)

    nifty.columns = [c.lower() for c in nifty.columns]

    # VIX (only available daily from Yahoo)
    vix = yf.download("^INDIAVIX", start=start_date, end=end_date,
                       interval="1d", progress=False)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)

    # Save
    data_dir = project_root / "data" / "historical"
    data_dir.mkdir(parents=True, exist_ok=True)

    nifty_path = data_dir / f"nifty_15min_{start_date}_{end_date}_yahoo.csv"
    nifty.to_csv(nifty_path)

    trading_days = len(set(nifty.index.date))
    print(f"\nSaved: {nifty_path}")
    print(f"  Trading days: {trading_days}")
    print(f"  Total bars: {len(nifty)}")
    print(f"  NIFTY range: {nifty['close'].min():.0f} - {nifty['close'].max():.0f}")

    return nifty, vix


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download real intraday NIFTY data")
    parser.add_argument("--start", default="2025-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default="2025-04-06", help="End date YYYY-MM-DD")
    parser.add_argument("--source", default="kite", choices=["kite", "yahoo"],
                        help="Data source")
    parser.add_argument("--interval", default="15minute", help="Candle interval")
    args = parser.parse_args()

    if args.source == "kite":
        download_from_kite(args.start, args.end, args.interval)
    else:
        download_from_yfinance(args.start, args.end)
