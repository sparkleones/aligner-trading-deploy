"""
Multi-Strategy Consensus View.

Runs ALL major strategies in parallel today + shows what each picks.
INFORMATIONAL — does not switch the production strategy.

Why this matters for LIVE markets:
  - Tells you what 5 different "experts" think today
  - Consensus stocks (picked by 3+ strategies) = highest conviction
  - Stocks picked by only 1 strategy = riskier bets
  - You can decide manually if you want to add a high-conviction
    consensus pick to your portfolio
  - Does NOT auto-switch your strategy
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .data_loader import load_universe
from .strategies.composite_top import CompositeTopStrategy
from .strategies.momentum import MomentumStrategy
from .strategies.stage2_trend import Stage2TrendStrategy
from .strategies.breakout import BreakoutStrategy
from .strategies.low_volatility import LowVolatilityStrategy
from .strategies.qgv import QGVStrategy
from .universe import get_sector
from .universe_extended import LARGE_CAP, MID_CAP


STRATEGIES_TO_RUN = {
    "Composite (Stage2+Breakout)": CompositeTopStrategy(),
    "Pure 12-1 Momentum": MomentumStrategy(),
    "Weinstein Stage 2": Stage2TrendStrategy(),
    "52W Breakout": BreakoutStrategy(),
    "Low Volatility": LowVolatilityStrategy(),
    "Quality+Growth+Value": QGVStrategy(),
}


def _rank_strategy(strategy, history: dict, top_n: int = 10) -> list[dict]:
    rows = []
    for sym, df in history.items():
        if df is None or len(df) < 252:
            continue
        try:
            score = strategy.score(sym, df)
        except Exception:
            continue
        if score is None or pd.isna(score) or not np.isfinite(score):
            continue
        rows.append({"symbol": sym, "score": float(score),
                     "sector": get_sector(sym)})
    rows.sort(key=lambda r: r["score"], reverse=True)
    return rows[:top_n]


def generate_consensus(universe: str = "LARGE", top_n_per_strategy: int = 10) -> dict:
    """Run each strategy + compute consensus."""
    if universe == "LARGE":
        symbols = LARGE_CAP
    elif universe == "MIXED":
        symbols = LARGE_CAP + MID_CAP
    else:
        return {"error": f"unknown universe '{universe}'"}

    history = load_universe(symbols, period="3y", use_cache=True, progress=False)
    if len(history) < 20:
        return {"error": f"only {len(history)} stocks loaded"}

    strategy_picks: dict[str, list] = {}
    all_pick_symbols: dict[str, list] = {}   # symbol -> list of strategies that picked it

    for name, strat in STRATEGIES_TO_RUN.items():
        picks = _rank_strategy(strat, history, top_n=top_n_per_strategy)
        strategy_picks[name] = picks
        for p in picks:
            all_pick_symbols.setdefault(p["symbol"], []).append(name)

    # Consensus: stocks picked by 3+ strategies
    consensus = []
    for sym, strategies in all_pick_symbols.items():
        if len(strategies) >= 3:
            # Get the highest score across strategies
            scores = [p["score"] for n, picks in strategy_picks.items() if n in strategies
                       for p in picks if p["symbol"] == sym]
            consensus.append({
                "symbol": sym,
                "sector": get_sector(sym),
                "picked_by_n_strategies": len(strategies),
                "strategies": strategies,
                "best_score": max(scores) if scores else 0,
            })
    consensus.sort(key=lambda x: (x["picked_by_n_strategies"], x["best_score"]), reverse=True)

    # Also: high-conviction singletons (only 1 strategy picked, but strong score)
    singletons = []
    for sym, strategies in all_pick_symbols.items():
        if len(strategies) == 1:
            scores = [p["score"] for n, picks in strategy_picks.items() if n == strategies[0]
                       for p in picks if p["symbol"] == sym]
            singletons.append({
                "symbol": sym,
                "sector": get_sector(sym),
                "picked_only_by": strategies[0],
                "score": scores[0] if scores else 0,
            })
    singletons.sort(key=lambda x: x["score"], reverse=True)

    return {
        "universe": universe,
        "n_stocks_evaluated": len(history),
        "strategies": list(STRATEGIES_TO_RUN.keys()),
        "picks_by_strategy": strategy_picks,
        "consensus_picks": consensus,
        "singleton_high_score": singletons[:10],
        "production_recommendation": (
            "Use the Composite (Stage2+Breakout) picks via the LivePicks v3 "
            "panel for production. This module is informational only — "
            "shows you what other 'experts' would pick today so you have "
            "context, NOT a switching signal. The senior-PM rule: never "
            "auto-switch strategies based on what looks good today."
        ),
    }
