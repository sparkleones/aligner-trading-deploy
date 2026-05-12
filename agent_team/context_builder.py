"""
Build the structured context dict for the agent team. ALL data is
computed in Python — agents only INTERPRET, never generate prices.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd


def _rsi(closes: pd.Series, period: int = 14) -> float:
    if len(closes) < period + 1:
        return float("nan")
    delta = closes.diff().dropna()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return float((100 - (100 / (1 + rs))).iloc[-1])


def _atr_pct(df: pd.DataFrame, period: int = 20) -> float:
    if len(df) < period + 1:
        return 0.04
    h, l, c = df["High"], df["Low"], df["Close"]
    prev_c = c.shift(1)
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    atr = tr.rolling(period).mean().iloc[-1]
    px = c.iloc[-1]
    return float(atr / px) if px > 0 else 0.04


def build_technical_context(history: pd.DataFrame) -> dict:
    """Compute every technical signal the agents need."""
    if history is None or history.empty or len(history) < 260:
        return {}
    c = history["Close"]
    close = float(c.iloc[-1])
    high_252 = float(history["High"].tail(252).max())
    ma_200 = float(c.rolling(200).mean().iloc[-1])
    ma_50 = float(c.rolling(50).mean().iloc[-1])
    ma_50_prev = float(c.rolling(50).mean().iloc[-21]) if len(c) > 21 else ma_50
    ma_50_slope = (ma_50 / ma_50_prev - 1.0) if ma_50_prev > 0 else 0
    # 12-1M momentum
    if len(c) >= 253:
        mom_end = c.iloc[-21]
        mom_start = c.iloc[-252]
        mom_12_1 = (mom_end / mom_start - 1.0) if mom_start > 0 else 0
    else:
        mom_12_1 = 0
    # MACD
    ema_12 = c.ewm(span=12, adjust=False).mean()
    ema_26 = c.ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26
    sig = macd.ewm(span=9, adjust=False).mean()
    macd_h = float(macd.iloc[-1] - sig.iloc[-1])
    # Vol
    vol_20 = float(c.pct_change().tail(20).std() * np.sqrt(252))
    # Volume
    v = history.get("Volume", pd.Series(dtype=float))
    if len(v) >= 21:
        today_v = float(v.iloc[-1])
        avg_v = float(v.tail(21).iloc[:-1].mean())
        vol_ratio = today_v / max(1, avg_v)
    else:
        vol_ratio = 1.0
    return {
        "close": close,
        "dist_from_52w_high_pct": (close / high_252 - 1.0) if high_252 > 0 else 0,
        "above_200dma": close > ma_200,
        "above_50dma": close > ma_50,
        "ma_50_slope_pct": ma_50_slope,
        "momentum_12_1": mom_12_1,
        "rsi_14": _rsi(c, 14),
        "macd_hist": macd_h,
        "atr_pct": _atr_pct(history, 20),
        "vol_20d_pct": vol_20,
        "vol_ratio": vol_ratio,
    }


def build_risk_context(history: pd.DataFrame, nifty_history: Optional[pd.DataFrame] = None) -> dict:
    """Drawdown, vol, beta vs NIFTY."""
    if history is None or history.empty or len(history) < 252:
        return {}
    c = history["Close"]
    rets = c.pct_change().dropna()
    vol_252 = float(rets.tail(252).std() * np.sqrt(252))
    # Drawdown right now
    high_252 = float(history["High"].tail(252).max())
    cur_dd = (float(c.iloc[-1]) / high_252 - 1.0) if high_252 > 0 else 0
    # Max DD over last 12m
    last_yr = c.tail(252)
    running_max = last_yr.cummax()
    dd_series = (last_yr / running_max - 1.0)
    max_dd = float(dd_series.min())
    # Gap freq
    o = history["Open"].tail(252)
    prev_c = c.shift(1).tail(252)
    gaps = ((o / prev_c - 1.0).abs()).dropna()
    gap_freq = float((gaps > 0.03).mean()) if len(gaps) else 0
    # Beta vs NIFTY
    beta = 1.0
    corr = 0.0
    if nifty_history is not None and not nifty_history.empty:
        n_rets = nifty_history["Close"].pct_change().dropna().tail(252)
        common = rets.tail(252).index.intersection(n_rets.index)
        if len(common) >= 100:
            sr = rets.loc[common]
            nr = n_rets.loc[common]
            cov = float(np.cov(sr, nr)[0, 1])
            varn = float(nr.var())
            if varn > 0:
                beta = cov / varn
            corr = float(np.corrcoef(sr, nr)[0, 1])
    return {
        "vol_252d_pct": vol_252,
        "current_drawdown_pct": cur_dd,
        "max_dd_12m_pct": max_dd,
        "gap_3pct_freq": gap_freq,
        "beta_vs_nifty": beta,
        "corr_nifty": corr,
        "would_overweight": False,
    }


def build_sector_context(sector: str, sector_histories: dict[str, pd.DataFrame],
                          nifty_history: Optional[pd.DataFrame] = None) -> dict:
    """Aggregate sector-level momentum + breadth."""
    if not sector_histories:
        return {"sector": sector}
    rets_3m = []
    rets_6m = []
    above_200 = 0
    n = 0
    for sym, df in sector_histories.items():
        if df is None or len(df) < 252:
            continue
        n += 1
        c = df["Close"]
        if len(c) >= 64:
            rets_3m.append(float(c.iloc[-1] / c.iloc[-63] - 1.0))
        if len(c) >= 127:
            rets_6m.append(float(c.iloc[-1] / c.iloc[-126] - 1.0))
        ma_200 = c.rolling(200).mean().iloc[-1]
        if not pd.isna(ma_200) and c.iloc[-1] > ma_200:
            above_200 += 1
    out = {
        "sector": sector,
        "sector_3m_return": float(np.median(rets_3m)) if rets_3m else 0,
        "sector_6m_return": float(np.median(rets_6m)) if rets_6m else 0,
        "sector_breadth_pct": (above_200 / n * 100) if n > 0 else 0,
    }
    # Relative strength vs NIFTY
    if nifty_history is not None and not nifty_history.empty:
        if len(nifty_history) >= 127:
            nifty_6m = float(nifty_history["Close"].iloc[-1] /
                              nifty_history["Close"].iloc[-126] - 1.0)
            out["sector_relative_strength"] = out["sector_6m_return"] - nifty_6m
        else:
            out["sector_relative_strength"] = 0
    return out


def build_event_context(fundamentals: dict) -> dict:
    """Best-effort upcoming event window from fundamentals snapshot."""
    if not fundamentals:
        return {}
    # Heuristic: most Indian companies report quarterly ~45-50 days after quarter end
    today = datetime.now().date()
    days_to_next = None
    next_earn = None
    # Quarter ends are Mar/Jun/Sep/Dec 31
    for m in (3, 6, 9, 12):
        try:
            qe = today.replace(month=m, day=30 if m != 12 else 31)
            results_date = qe + timedelta(days=47)
            if results_date >= today:
                next_earn = results_date.isoformat()
                days_to_next = (results_date - today).days
                break
        except Exception:
            pass
    div_y = fundamentals.get("dividendYield", 0)
    return {
        "next_earnings_estimated": next_earn or "unknown",
        "days_to_next_earnings": days_to_next if days_to_next is not None else "unknown",
        "dividend_yield": div_y or 0,
        "ex_dividend_date": "unknown",
    }


def build_context(
    symbol: str,
    sector: str,
    history: pd.DataFrame,
    fundamentals: Optional[dict] = None,
    nifty_history: Optional[pd.DataFrame] = None,
    sector_peers: Optional[dict[str, pd.DataFrame]] = None,
    macro_extra: Optional[dict] = None,
) -> dict:
    """Build the complete context dict for the agent team."""
    return {
        "symbol": symbol,
        "sector": sector,
        "technical": build_technical_context(history),
        "fundamentals": fundamentals or {},
        "risk": build_risk_context(history, nifty_history),
        "sector_metrics": (build_sector_context(sector, sector_peers, nifty_history)
                            if sector_peers else {"sector": sector}),
        "events": build_event_context(fundamentals or {}),
        "macro": macro_extra or {},
        "current_holdings_count": 0,
    }
