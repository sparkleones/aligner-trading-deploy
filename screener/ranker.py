"""
Composite ranking: z-score each factor across the universe, weight, sort.

Default factor weights chosen from literature + Indian-market backtests:
    momentum_12_1:  +0.35  (primary driver of cross-sectional returns)
    trend:          +0.25  (confirms momentum is not from a single spike)
    reversal_1m:    -0.15  (negative weight — short-term reversal)
    low_vol:        +0.15  (defensive premium)
    gap_risk:       +0.10  (penalize overnight-gap-prone names)

Liquidity is a FILTER (must exceed threshold), not a ranking factor.

A composite score < 0 = below universe average, will be excluded from
picks regardless of rank.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .factors import compute_all_factors
from .universe import get_sector


@dataclass
class RankerConfig:
    factor_weights: dict[str, float] = field(default_factory=lambda: {
        "momentum_12_1": 0.35,
        "trend":         0.25,
        "reversal_1m":  -0.15,  # negative — short-term reversal
        "low_vol":       0.15,
        "gap_risk":      0.10,
    })
    min_liquidity_log10_inr: float = 9.0   # 10^9 = ₹100 Cr median turnover/day
    min_bars: int = 252
    max_atr_pct: float = 0.06              # skip if ATR > 6% of price (too jumpy)
    require_above_200dma: bool = True
    max_per_sector: int = 1                # diversification: max 1 stock per sector


def _zscore(s: pd.Series) -> pd.Series:
    s = s.replace([np.inf, -np.inf], np.nan)
    mu = s.mean(skipna=True)
    sd = s.std(skipna=True, ddof=0)
    if sd is None or sd == 0 or math.isnan(sd):
        return s * 0.0
    return (s - mu) / sd


def rank_universe(
    history: dict[str, pd.DataFrame],
    cfg: RankerConfig | None = None,
) -> pd.DataFrame:
    """
    Compute composite score for each stock and return a sorted DataFrame
    (best first). Columns: symbol, sector, composite, plus all factors
    and z-scores. Rows that fail filters are excluded.
    """
    cfg = cfg or RankerConfig()

    rows = []
    for sym, df in history.items():
        if df is None or len(df) < cfg.min_bars:
            continue
        f = compute_all_factors(df)

        # Hard filters
        if math.isnan(f["liquidity"]) or f["liquidity"] < cfg.min_liquidity_log10_inr:
            continue
        if math.isnan(f["atr_pct"]) or f["atr_pct"] > cfg.max_atr_pct:
            continue
        if cfg.require_above_200dma:
            if len(df) < 200:
                continue
            ma200 = df["Close"].rolling(200).mean().iloc[-1]
            if pd.isna(ma200) or df["Close"].iloc[-1] < ma200:
                continue

        rows.append({"symbol": sym, "sector": get_sector(sym), **f})

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).set_index("symbol")

    # Z-score the ranking factors
    z_cols = {}
    for fac in cfg.factor_weights:
        if fac not in df.columns:
            continue
        z = _zscore(df[fac])
        df[f"z_{fac}"] = z
        z_cols[fac] = z

    # Composite = weighted sum of z-scores
    composite = pd.Series(0.0, index=df.index)
    for fac, w in cfg.factor_weights.items():
        if fac in z_cols:
            composite = composite + w * z_cols[fac].fillna(0.0)
    df["composite"] = composite

    df = df.sort_values("composite", ascending=False).reset_index()
    return df


def pick_top_n(
    ranked: pd.DataFrame,
    n: int = 3,
    cfg: RankerConfig | None = None,
) -> pd.DataFrame:
    """
    Pick top N after sector-diversification rule.
    Skips a candidate if its sector already has `max_per_sector` picks.
    Also requires composite > 0 (above universe average).
    """
    cfg = cfg or RankerConfig()
    if ranked.empty:
        return ranked

    picks = []
    sector_count: dict[str, int] = {}
    for _, row in ranked.iterrows():
        if row["composite"] <= 0:
            break  # no more good candidates
        sec = row["sector"]
        if sector_count.get(sec, 0) >= cfg.max_per_sector:
            continue
        picks.append(row)
        sector_count[sec] = sector_count.get(sec, 0) + 1
        if len(picks) >= n:
            break

    return pd.DataFrame(picks).reset_index(drop=True)
