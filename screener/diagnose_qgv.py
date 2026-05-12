"""Diagnose why QGV is picking so few stocks."""
from .data_loader import load_universe
from .universe_extended import SMALL_CAP, MID_CAP, LARGE_CAP
from .strategies.qgv import QGVStrategy
import pandas as pd


def main():
    strat = QGVStrategy()
    for name, uni in [("SMALL_CAP", SMALL_CAP), ("MID_CAP", MID_CAP), ("LARGE_CAP", LARGE_CAP)]:
        h = load_universe(uni, period="5y", use_cache=True, progress=False)
        asof = pd.Timestamp("2024-01-01")
        scores = []
        for sym, df in h.items():
            slice_ = df.loc[:asof]
            if len(slice_) < 750:
                continue
            s = strat.score(sym, slice_, asof=asof)
            scores.append((sym, s))
        n_total = len(h)
        n_passed = sum(1 for s, sc in scores if sc is not None and pd.notna(sc))
        print(f"{name}: {n_passed}/{n_total} pass at 2024-01-01")
        # Show top 5
        passed = [(s, sc) for s, sc in scores if sc is not None and pd.notna(sc)]
        passed.sort(key=lambda x: x[1], reverse=True)
        for s, sc in passed[:5]:
            print(f"  {s}: {sc:.3f}")
    asof = pd.Timestamp("2022-06-01")
    for name, uni in [("SMALL_CAP", SMALL_CAP), ("MID_CAP", MID_CAP)]:
        h = load_universe(uni, period="5y", use_cache=True, progress=False)
        scores = []
        for sym, df in h.items():
            slice_ = df.loc[:asof]
            if len(slice_) < 750:
                continue
            s = strat.score(sym, slice_, asof=asof)
            scores.append((sym, s))
        n_passed = sum(1 for s, sc in scores if sc is not None and pd.notna(sc))
        print(f"{name} at {asof.date()}: {n_passed}/{len(h)} pass")


if __name__ == "__main__":
    main()
