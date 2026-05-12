"""Trace exactly where QGV rejects each stock."""
import pandas as pd
import numpy as np
from .data_loader import load_universe
from .universe_extended import SMALL_CAP, MID_CAP
from .strategies.qgv import QGVStrategy


def trace(sym, df, asof):
    s = df.loc[:asof]
    print(f"\n{sym}: {len(s)} bars up to {asof.date()}")
    if len(s) < 500:
        print("  REJECT: < 500 bars")
        return
    c = s["Close"]
    close = float(c.iloc[-1])
    print(f"  close: {close:.1f}")
    if len(c) >= 200:
        ma_200 = float(c.rolling(200).mean().iloc[-1])
        print(f"  ma_200: {ma_200:.1f}  close vs 200DMA: {(close/ma_200-1)*100:+.2f}%")
        if close < ma_200 * 0.95:
            print("  REJECT: close < 95% of 200DMA")
            return
    if len(c) >= 750 + 200:
        ma_200s = c.rolling(200).mean()
        last_750 = c.tail(750)
        last_750_ma = ma_200s.tail(750)
        pers = float((last_750 > last_750_ma).mean())
        print(f"  persistence above 200DMA: {pers*100:.1f}%")
        if pers < 0.35:
            print("  REJECT: persistence < 35%")
            return
    if len(c) >= 750:
        slice_3y = c.tail(750)
        dd = float((slice_3y / slice_3y.cummax() - 1.0).min())
        print(f"  max DD 3y: {dd*100:.1f}%")
        if dd < -0.60:
            print("  REJECT: max DD < -60%")
            return
        cagr_3y = float((slice_3y.iloc[-1] / slice_3y.iloc[0]) ** (1/3) - 1)
        print(f"  3y CAGR: {cagr_3y*100:+.2f}%")
        if cagr_3y < -0.05:
            print("  REJECT: CAGR < -5%")
            return
    print("  PASS")


def main():
    h = load_universe(SMALL_CAP[:5] + MID_CAP[:5], period="5y", use_cache=True, progress=False)
    asof = pd.Timestamp("2024-01-01")
    for sym, df in list(h.items())[:8]:
        trace(sym, df, asof)


if __name__ == "__main__":
    main()
