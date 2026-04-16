"""Verify live indicator values by recomputing them independently.

Fetches the SAME 30-day window of 5-min NIFTY 50 bars the live engine
loaded at startup, then computes each indicator two ways:
  (1) using scoring.indicators.compute_indicators() — what live actually uses
  (2) using a hand-rolled, textbook formula — independent ground truth

Prints both, plus the live engine's last logged value, so we can spot
any indicator that's drifted from textbook behavior.

Run from project root:
    python verify_live_indicators.py
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scoring.indicators import compute_indicators  # noqa: E402
from broker.kite_connect import KiteConnectBroker as KiteBroker  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────
# Independent textbook implementations (no shared code with scoring/)
# ─────────────────────────────────────────────────────────────────────────

def textbook_rsi(closes: np.ndarray, period: int = 14) -> float:
    """Wilder's RSI on the LAST `period` deltas.

    Mirrors the same simple-mean variant scoring/indicators.py uses
    (NOT Wilder's full smoothing) so we compare apples to apples.
    """
    if len(closes) < period + 1:
        return 50.0
    deltas = np.diff(closes[-(period + 1):])
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = gains.mean() + 1e-10
    avg_loss = losses.mean() + 1e-10
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def textbook_ema(closes: np.ndarray, period: int) -> float:
    """Standard EMA seeded with SMA of first `period` closes."""
    if len(closes) < period:
        return float(closes[-1])
    k = 2.0 / (period + 1)
    e = float(closes[:period].mean())
    for v in closes[period:]:
        e = float(v) * k + e * (1.0 - k)
    return e


def textbook_atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
                 period: int = 14) -> float:
    """ATR using Wilder's smoothing on the most recent `period+1` bars."""
    n = len(closes)
    lookback = min(n, period + 1)
    if lookback < 2:
        return 0.0
    tr_vals = []
    for i in range(1, lookback):
        idx = n - lookback + i
        tr = max(
            highs[idx] - lows[idx],
            abs(highs[idx] - closes[idx - 1]),
            abs(lows[idx] - closes[idx - 1]),
        )
        tr_vals.append(float(tr))
    if not tr_vals:
        return 0.0
    atr_val = tr_vals[0]
    alpha = 1.0 / period
    for tr in tr_vals[1:]:
        atr_val = atr_val * (1 - alpha) + tr * alpha
    return atr_val


def textbook_bb(closes: np.ndarray, period: int = 20, mult: float = 2.0):
    """Bollinger Bands: SMA(period) ± mult × population stdev (ddof=1)."""
    if len(closes) < period:
        return float(closes[-1]), float(closes[-1]), float(closes[-1])
    s = closes[-period:]
    mid = float(s.mean())
    std = float(np.std(s, ddof=1))
    return mid - mult * std, mid, mid + mult * std


def textbook_vwap_today(bars: list, today_str: str) -> tuple[float, int]:
    """True intraday VWAP — only bars whose timestamp starts with `today_str`."""
    tps, vols = [], []
    for b in bars:
        ts = str(b.get("time") or b.get("timestamp") or b.get("date") or "")
        if not ts.startswith(today_str):
            continue
        tp = (float(b["high"]) + float(b["low"]) + float(b["close"])) / 3.0
        vol = float(b.get("volume", 0)) or (float(b["high"]) - float(b["low"]) + 1.0)
        tps.append(tp)
        vols.append(vol)
    if not tps:
        return float("nan"), 0
    tp_arr = np.array(tps)
    vol_arr = np.array(vols)
    return float((tp_arr * vol_arr).sum() / (vol_arr.sum() + 1e-10)), len(tps)


def textbook_vwap_cumulative_all(bars: list) -> float:
    """Cumulative VWAP across the entire bar buffer (the broken 'fallback')."""
    tps, vols = [], []
    for b in bars:
        tp = (float(b["high"]) + float(b["low"]) + float(b["close"])) / 3.0
        vol = float(b.get("volume", 0)) or (float(b["high"]) - float(b["low"]) + 1.0)
        tps.append(tp)
        vols.append(vol)
    tp_arr = np.array(tps)
    vol_arr = np.array(vols)
    return float((tp_arr * vol_arr).sum() / (vol_arr.sum() + 1e-10))


# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 78)
    print("LIVE INDICATOR VERIFICATION")
    print("=" * 78)

    # Fetch the same window the live engine uses
    broker = KiteBroker()
    if not broker.authenticate():
        print("ERROR: Kite authentication failed")
        return
    now = datetime.now()
    market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)

    warmup_from = (now - timedelta(days=30)).replace(
        hour=9, minute=15, second=0, microsecond=0,
    )
    warmup_to = market_open - timedelta(seconds=1)

    print(f"\nFetching warmup window: {warmup_from} -> {warmup_to}")
    warmup_bars = broker.get_historical_data(
        symbol="NIFTY 50",
        from_dt=warmup_from,
        to_dt=warmup_to,
        interval="5minute",
    )
    print(f"  -> {len(warmup_bars)} warmup bars")

    print(f"\nFetching today's bars: {market_open} -> {now}")
    today_bars = broker.get_historical_data(
        symbol="NIFTY 50",
        from_dt=market_open,
        to_dt=now,
        interval="5minute",
    )
    print(f"  -> {len(today_bars)} today bars")

    bars = list(warmup_bars) + list(today_bars)
    if not bars:
        print("ERROR: no bars returned")
        return

    today_str = now.strftime("%Y-%m-%d")

    # Inspect bar dict shape — does it have a 'date' field?
    sample = bars[-1]
    print(f"\nSample bar keys: {sorted(sample.keys())}")
    print(f"Sample bar:      {sample}")
    print(f"Today date str:  {today_str!r}")

    closes = np.array([b["close"] for b in bars], dtype=np.float64)
    highs = np.array([b["high"]  for b in bars], dtype=np.float64)
    lows  = np.array([b["low"]   for b in bars], dtype=np.float64)

    # ── (1) Live shared function — exactly what v14_live_agent calls ──
    live_ind_no_date = compute_indicators(bars, today_date="")
    live_ind_today  = compute_indicators(bars, today_date=today_str)

    # ── (2) Independent textbook ──
    tb_close  = float(closes[-1])
    tb_rsi    = textbook_rsi(closes, 14)
    tb_ema9   = textbook_ema(closes, 9)
    tb_ema21  = textbook_ema(closes, 21)
    tb_ema12  = textbook_ema(closes, 12)
    tb_ema26  = textbook_ema(closes, 26)
    tb_atr    = textbook_atr(highs, lows, closes, 14)
    tb_bb_lo, tb_bb_mid, tb_bb_hi = textbook_bb(closes, 20, 2.0)
    tb_vwap_today, today_count   = textbook_vwap_today(bars, today_str)
    tb_vwap_cum_all              = textbook_vwap_cumulative_all(bars)

    print("\n" + "=" * 78)
    print(f"INDICATOR COMPARISON  ({len(bars)} total bars, {today_count} today)")
    print("=" * 78)
    print(f"{'Indicator':<22}{'Textbook':>16}{'Shared(no date)':>20}{'Shared(today)':>18}")
    print("-" * 78)

    def row(name, tb, sh_no, sh_yes, fmt="{:>16.4f}"):
        sh_no_str  = fmt.format(sh_no)  if sh_no  is not None else " " * 16
        sh_yes_str = fmt.format(sh_yes) if sh_yes is not None else " " * 16
        print(f"{name:<22}{fmt.format(tb)}{sh_no_str:>20}{sh_yes_str:>18}")

    row("close", tb_close,
        live_ind_no_date.get("close"), live_ind_today.get("close"))
    row("RSI(14)", tb_rsi,
        live_ind_no_date.get("rsi"),   live_ind_today.get("rsi"))
    row("EMA9",   tb_ema9,
        live_ind_no_date.get("ema9"),  live_ind_today.get("ema9"))
    row("EMA21",  tb_ema21,
        live_ind_no_date.get("ema21"), live_ind_today.get("ema21"))
    row("ATR(14)", tb_atr,
        live_ind_no_date.get("atr"),   live_ind_today.get("atr"))
    row("BB lower", tb_bb_lo,
        live_ind_no_date.get("bb_lower"), live_ind_today.get("bb_lower"))
    row("BB upper", tb_bb_hi,
        live_ind_no_date.get("bb_upper"), live_ind_today.get("bb_upper"))
    row("MACD hist (e12-e26)", tb_ema12 - tb_ema26,
        live_ind_no_date.get("macd_hist"), live_ind_today.get("macd_hist"))

    print()
    print(f"{'VWAP variants':<22}{'value':>16}")
    print("-" * 78)
    print(f"{'  textbook today-only':<22}{tb_vwap_today:>16.4f}    "
          f"({today_count} bars from {today_str})")
    print(f"{'  textbook cum ALL':<22}{tb_vwap_cum_all:>16.4f}    "
          f"({len(bars)} bars across 30 days — the broken state)")
    print(f"{'  shared(no date)':<22}{live_ind_no_date.get('vwap', 0):>16.4f}    "
          f"<- what live currently shows when today_date=''")
    print(f"{'  shared(today)':<22}{live_ind_today.get('vwap', 0):>16.4f}    "
          f"<- what live SHOULD show if today_date were passed")

    # Sanity check: do bar timestamps actually contain the YYYY-MM-DD?
    matches_today = sum(
        1 for b in bars
        if str(b.get("time") or b.get("timestamp") or b.get("date") or "").startswith(today_str)
    )
    print(f"\nBars whose ts starts with {today_str!r}: {matches_today}")
    if matches_today == 0:
        print("  !! ZERO bars have a timestamp matching today -- date filter would fail")

    # Live engine snapshot from log (entered manually for sanity)
    print("\n" + "=" * 78)
    print("LIVE LOG SNAPSHOT (most recent V14 SCORED line)")
    print("=" * 78)
    print("close=23978.2  RSI=78.7  vwap=23069.7   <- last logged values")


if __name__ == "__main__":
    main()
