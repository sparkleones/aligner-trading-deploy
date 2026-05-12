"""
Market Timing Analyzer — "Should I buy now or wait?"

Pulls live NIFTY 50 data and computes objective signals across 5 axes:

  1. PRICE: Distance from 52w high (the deeper, the better historically)
  2. TECHNICAL: Weekly RSI, distance from 200DMA, 50/200 cross
  3. VOLATILITY: India VIX level + recent change
  4. BREADTH: % of stocks above 200DMA, % at 52w high
  5. HISTORICAL ANALOGS: When NIFTY was at similar setup, what happened next?

Returns an honest verdict:
  BUY NOW       - all signals aligned, conditions like historical bottoms
  DCA           - mixed signals, dollar-cost average is the right approach
  WAIT          - red flags dominant, momentum still down, wait for stabilization
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from screener.data_loader import load_history
from screener.universe_extended import LARGE_CAP, MID_CAP, ALL_STOCKS


def _rsi(closes: pd.Series, period: int = 14) -> float:
    if len(closes) < period + 1:
        return np.nan
    delta = closes.diff().dropna()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else np.nan


def _fetch_nifty():
    """Fetch NIFTY 50 history via yfinance ^NSEI."""
    import yfinance as yf
    df = yf.Ticker("^NSEI").history(period="5y", auto_adjust=True)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df = df.dropna()
    return df


def _fetch_vix():
    """Fetch India VIX via yfinance ^INDIAVIX."""
    import yfinance as yf
    try:
        df = yf.Ticker("^INDIAVIX").history(period="1y", auto_adjust=True)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        return df.dropna()
    except Exception:
        return pd.DataFrame()


def analyze_price_position(nifty: pd.DataFrame) -> dict:
    """Position vs 52-week high / 200-DMA."""
    close = float(nifty["Close"].iloc[-1])
    high_252 = float(nifty["High"].tail(252).max())
    low_252 = float(nifty["Low"].tail(252).min())
    dist_from_high = (close / high_252) - 1.0
    dist_from_low = (close / low_252) - 1.0
    ma_200 = float(nifty["Close"].rolling(200).mean().iloc[-1])
    dist_from_ma200 = (close / ma_200) - 1.0
    ma_50 = float(nifty["Close"].rolling(50).mean().iloc[-1])
    dist_from_ma50 = (close / ma_50) - 1.0
    return {
        "close": close,
        "52w_high": high_252,
        "52w_low": low_252,
        "dist_from_high_pct": dist_from_high * 100,
        "dist_from_low_pct": dist_from_low * 100,
        "ma_200": ma_200,
        "ma_50": ma_50,
        "dist_from_ma200_pct": dist_from_ma200 * 100,
        "dist_from_ma50_pct": dist_from_ma50 * 100,
        "above_200dma": close > ma_200,
        "above_50dma": close > ma_50,
        "golden_cross": ma_50 > ma_200,  # bullish trend cross
    }


def analyze_technical(nifty: pd.DataFrame) -> dict:
    """RSI on daily and weekly. MACD."""
    daily_close = nifty["Close"]
    rsi_14 = _rsi(daily_close, 14)
    # Weekly RSI: resample to weekly
    weekly = nifty["Close"].resample("W").last().dropna()
    rsi_14_w = _rsi(weekly, 14)
    # MACD
    ema_12 = daily_close.ewm(span=12, adjust=False).mean()
    ema_26 = daily_close.ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26
    signal = macd.ewm(span=9, adjust=False).mean()
    macd_hist = float(macd.iloc[-1] - signal.iloc[-1])
    # 20-day return + volatility
    ret_20 = (daily_close.iloc[-1] / daily_close.iloc[-21] - 1.0) if len(daily_close) >= 21 else 0
    vol_20 = float(daily_close.pct_change().tail(20).std() * np.sqrt(252))
    return {
        "rsi_14_daily": rsi_14,
        "rsi_14_weekly": rsi_14_w,
        "macd_histogram": macd_hist,
        "macd_bullish": macd_hist > 0,
        "ret_20d_pct": ret_20 * 100,
        "vol_20d_annualized_pct": vol_20 * 100,
    }


def analyze_volatility(vix: pd.DataFrame) -> dict:
    if vix.empty:
        return {"vix_available": False}
    cur = float(vix["Close"].iloc[-1])
    avg_30 = float(vix["Close"].tail(30).mean())
    avg_252 = float(vix["Close"].tail(252).mean())
    high_252 = float(vix["High"].tail(252).max())
    pct_above_avg = ((cur / avg_252) - 1) * 100
    # Classify regime
    if cur < 12:
        regime = "COMPLACENT"
    elif cur < 16:
        regime = "CALM"
    elif cur < 20:
        regime = "NORMAL"
    elif cur < 25:
        regime = "ELEVATED"
    else:
        regime = "CRISIS"
    return {
        "vix_available": True,
        "vix_current": cur,
        "vix_avg_30d": avg_30,
        "vix_avg_252d": avg_252,
        "vix_high_252d": high_252,
        "vix_pct_vs_avg_pct": pct_above_avg,
        "vix_regime": regime,
    }


def analyze_breadth(stocks: list[str]) -> dict:
    """How many large-caps are above their 200DMA right now? At 52w high?"""
    above_200 = 0
    near_52w_high = 0   # within 5%
    total = 0
    for sym in stocks[:60]:  # sample to keep fast
        try:
            df = load_history(sym, period="2y", use_cache=True)
        except Exception:
            continue
        if df.empty or len(df) < 220:
            continue
        total += 1
        close = float(df["Close"].iloc[-1])
        ma_200 = float(df["Close"].rolling(200).mean().iloc[-1])
        h_252 = float(df["High"].tail(252).max())
        if close > ma_200:
            above_200 += 1
        if h_252 > 0 and (close / h_252) >= 0.95:
            near_52w_high += 1
    if total == 0:
        return {"breadth_available": False}
    return {
        "breadth_available": True,
        "n_sampled": total,
        "pct_above_200dma": (above_200 / total) * 100,
        "pct_near_52w_high": (near_52w_high / total) * 100,
    }


def historical_analog(nifty: pd.DataFrame, dist_from_high_pct: float) -> dict:
    """
    Look back: every time NIFTY was within +-1% of the same drawdown depth
    historically, what were the forward 1/3/6/12 month returns?
    """
    close = nifty["Close"]
    rolling_high = close.rolling(252).max()
    drawdown = (close / rolling_high - 1.0) * 100
    target_dd = dist_from_high_pct
    band = 1.0
    matches = (drawdown >= (target_dd - band)) & (drawdown <= (target_dd + band))
    match_idx = drawdown.index[matches]
    fwd_returns = {1: [], 3: [], 6: [], 12: []}
    horizons = {1: 21, 3: 63, 6: 126, 12: 252}
    for idx in match_idx:
        pos = close.index.get_loc(idx)
        for months, bars in horizons.items():
            future_pos = pos + bars
            if future_pos < len(close):
                fwd = (close.iloc[future_pos] / close.iloc[pos] - 1.0) * 100
                fwd_returns[months].append(fwd)
    summary = {"n_historical_matches": len(match_idx)}
    for months, vals in fwd_returns.items():
        if vals:
            arr = np.array(vals)
            summary[f"forward_{months}m_median_pct"] = float(np.median(arr))
            summary[f"forward_{months}m_mean_pct"] = float(np.mean(arr))
            summary[f"forward_{months}m_win_rate_pct"] = float((arr > 0).mean() * 100)
        else:
            summary[f"forward_{months}m_median_pct"] = None
    return summary


def make_verdict(price, tech, vol, breadth, analog) -> dict:
    """Combine all signals into a verdict score."""
    score = 0
    factors = []

    # Drawdown signal (deeper = more bullish for fresh capital)
    dd = price["dist_from_high_pct"]
    if dd <= -15:
        score += 3
        factors.append(f"+3 NIFTY {dd:.1f}% below 52w high (deep correction territory)")
    elif dd <= -10:
        score += 2
        factors.append(f"+2 NIFTY {dd:.1f}% below 52w high (correction)")
    elif dd <= -5:
        score += 1
        factors.append(f"+1 NIFTY {dd:.1f}% below 52w high (mild pullback)")
    elif dd >= -2:
        score -= 1
        factors.append(f"-1 NIFTY very near 52w high ({dd:.1f}%) — buying late")

    # RSI signal
    rsi_d = tech["rsi_14_daily"]
    rsi_w = tech["rsi_14_weekly"]
    if rsi_d < 30 and rsi_w < 40:
        score += 3
        factors.append(f"+3 RSI deeply oversold (daily {rsi_d:.0f}, weekly {rsi_w:.0f})")
    elif rsi_d < 40:
        score += 1
        factors.append(f"+1 RSI mildly oversold (daily {rsi_d:.0f})")
    elif rsi_d > 70:
        score -= 2
        factors.append(f"-2 RSI overbought (daily {rsi_d:.0f}) — bad time for fresh entry")

    # 200DMA signal
    if not price["above_200dma"]:
        score -= 2
        factors.append(f"-2 Below 200-DMA — broader trend is DOWN (dist {price['dist_from_ma200_pct']:.1f}%)")
    else:
        score += 1
        factors.append(f"+1 Above 200-DMA (dist +{price['dist_from_ma200_pct']:.1f}%)")

    # Golden / Death cross
    if price["golden_cross"]:
        score += 1
        factors.append("+1 50-DMA above 200-DMA (golden cross intact)")
    else:
        score -= 2
        factors.append("-2 Death cross active (50-DMA below 200-DMA)")

    # VIX
    if vol.get("vix_available"):
        vix = vol["vix_current"]
        if vix >= 25:
            score += 3
            factors.append(f"+3 VIX at {vix:.1f} — fear is high, contrarian buy zone")
        elif vix >= 20:
            score += 1
            factors.append(f"+1 VIX at {vix:.1f} — elevated fear")
        elif vix <= 12:
            score -= 1
            factors.append(f"-1 VIX at {vix:.1f} — complacent (often before pullbacks)")

    # Breadth
    if breadth.get("breadth_available"):
        pct_above = breadth["pct_above_200dma"]
        if pct_above < 30:
            score += 2
            factors.append(f"+2 Only {pct_above:.0f}% of stocks above 200-DMA (extreme weakness — bottom signal)")
        elif pct_above < 50:
            score += 1
            factors.append(f"+1 {pct_above:.0f}% above 200-DMA (weak breadth)")
        elif pct_above > 80:
            score -= 1
            factors.append(f"-1 {pct_above:.0f}% above 200-DMA (extended)")

    # Historical analog
    if analog.get("forward_6m_median_pct") is not None:
        fwd_6m = analog["forward_6m_median_pct"]
        wr_6m = analog.get("forward_6m_win_rate_pct", 0)
        if fwd_6m > 8 and wr_6m > 65:
            score += 2
            factors.append(f"+2 Historical analog: median +{fwd_6m:.1f}% over next 6m ({wr_6m:.0f}% win rate, n={analog['n_historical_matches']})")
        elif fwd_6m > 0:
            score += 1
            factors.append(f"+1 Historical analog: median +{fwd_6m:.1f}% over next 6m")
        else:
            factors.append(f"0  Historical analog: median {fwd_6m:.1f}% over next 6m (weak)")

    # Verdict
    if score >= 6:
        verdict = "STRONG BUY"
        action = "Deploy capital now in 2-3 tranches over 2 weeks (DCA)."
    elif score >= 3:
        verdict = "BUY (cautious)"
        action = "Start deploying — first tranche 50% now, second 50% on any further dip."
    elif score >= 0:
        verdict = "DCA"
        action = "Mixed signals. Dollar-cost average — invest 1/4 of capital per week over 4 weeks."
    elif score >= -3:
        verdict = "WAIT"
        action = "Trend is down. Wait for stabilization: NIFTY to close above 50-DMA + RSI(daily) > 50 for 3 sessions."
    else:
        verdict = "STAY OUT"
        action = "Multiple red flags. Hold cash. Re-check in 2 weeks."

    return {"score": score, "verdict": verdict, "action": action, "factors": factors}


def main():
    print("=" * 78)
    print(" NIFTY 50 MARKET TIMING ANALYSIS")
    print(f" Generated: {datetime.now().strftime('%Y-%m-%d %H:%M IST')}")
    print("=" * 78)

    print("\nFetching live data...")
    nifty = _fetch_nifty()
    vix_df = _fetch_vix()
    if nifty.empty:
        print("[error] Could not fetch NIFTY data.")
        return

    price = analyze_price_position(nifty)
    tech = analyze_technical(nifty)
    vol = analyze_volatility(vix_df)
    breadth = analyze_breadth(LARGE_CAP)
    analog = historical_analog(nifty, price["dist_from_high_pct"])

    print(f"\n=== PRICE POSITION ===")
    print(f"  NIFTY 50:                  {price['close']:,.0f}")
    print(f"  52-week high:              {price['52w_high']:,.0f}  ({price['dist_from_high_pct']:+.2f}%)")
    print(f"  52-week low:               {price['52w_low']:,.0f}  ({price['dist_from_low_pct']:+.2f}%)")
    print(f"  200-DMA:                   {price['ma_200']:,.0f}  ({price['dist_from_ma200_pct']:+.2f}%)")
    print(f"  50-DMA:                    {price['ma_50']:,.0f}  ({price['dist_from_ma50_pct']:+.2f}%)")
    print(f"  Above 200-DMA:             {price['above_200dma']}")
    print(f"  Golden cross active:       {price['golden_cross']}")

    print(f"\n=== TECHNICAL ===")
    print(f"  RSI(14) daily:             {tech['rsi_14_daily']:.1f}")
    print(f"  RSI(14) weekly:            {tech['rsi_14_weekly']:.1f}")
    print(f"  MACD histogram:            {tech['macd_histogram']:+.2f}  ({'bullish' if tech['macd_bullish'] else 'bearish'})")
    print(f"  20-day return:             {tech['ret_20d_pct']:+.2f}%")
    print(f"  20-day vol (annualized):   {tech['vol_20d_annualized_pct']:.1f}%")

    print(f"\n=== VOLATILITY (INDIA VIX) ===")
    if vol.get("vix_available"):
        print(f"  VIX current:               {vol['vix_current']:.2f}  ({vol['vix_regime']})")
        print(f"  VIX 30-day avg:            {vol['vix_avg_30d']:.2f}")
        print(f"  VIX 252-day avg:           {vol['vix_avg_252d']:.2f}")
        print(f"  VIX vs avg:                {vol['vix_pct_vs_avg_pct']:+.1f}%")
    else:
        print("  VIX data unavailable")

    print(f"\n=== BREADTH (sampled LARGE caps) ===")
    if breadth.get("breadth_available"):
        print(f"  Sample size:               {breadth['n_sampled']}")
        print(f"  % above 200-DMA:           {breadth['pct_above_200dma']:.1f}%")
        print(f"  % within 5% of 52w high:   {breadth['pct_near_52w_high']:.1f}%")

    print(f"\n=== HISTORICAL ANALOGS (NIFTY drawdown ±1%) ===")
    print(f"  Matches found:             {analog['n_historical_matches']}")
    for m in (1, 3, 6, 12):
        med = analog.get(f"forward_{m}m_median_pct")
        wr = analog.get(f"forward_{m}m_win_rate_pct")
        if med is not None:
            print(f"  Forward {m:>2d}m median:        {med:+.2f}%   (win rate {wr:.0f}%)")

    verdict = make_verdict(price, tech, vol, breadth, analog)
    print(f"\n=== VERDICT ===")
    print(f"  Composite score:           {verdict['score']:+d}")
    print(f"  Verdict:                   {verdict['verdict']}")
    print(f"  Action:                    {verdict['action']}")
    print(f"\n  Factor breakdown:")
    for f in verdict["factors"]:
        print(f"    {f}")
    print("=" * 78)


if __name__ == "__main__":
    main()
