"""
Live signal generator — produces today's top picks using the proven
HTR Monthly 2pk default config from walk-forward.

Cached to disk so the dashboard endpoint is fast.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Optional

import pandas as pd

from .universe import get_universe
from .data_loader import load_universe
from .ranker import RankerConfig, rank_universe, pick_top_n
from .trade_plan import TradePlanConfig, build_plans_for_picks

CACHE_FILE = Path(__file__).resolve().parent.parent / "data" / "screener_live.json"
CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
CACHE_TTL_SEC = 6 * 3600  # refresh every 6 hours


# Proven config from walk-forward backtest
LIVE_RANKER_CONFIG = RankerConfig(
    factor_weights={
        "momentum_12_1": 0.35,
        "trend":         0.25,
        "reversal_1m":  -0.15,
        "low_vol":       0.15,
        "gap_risk":      0.10,
    },
    min_liquidity_log10_inr=9.0,
    min_bars=252,
    max_atr_pct=0.06,
    require_above_200dma=True,
    max_per_sector=1,
)


def _cache_valid() -> bool:
    if not CACHE_FILE.exists():
        return False
    age = time.time() - CACHE_FILE.stat().st_mtime
    return age < CACHE_TTL_SEC


def _read_cache() -> Optional[dict]:
    try:
        with open(CACHE_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return None


def _write_cache(payload: dict) -> None:
    try:
        with open(CACHE_FILE, "w") as f:
            json.dump(payload, f, indent=2, default=str)
    except Exception:
        pass


def generate_signals(
    account_capital: float = 100_000.0,
    n_picks: int = 2,
    force_refresh: bool = False,
) -> dict:
    """
    Return today's top picks with full trade plans. Cached.
    """
    if not force_refresh and _cache_valid():
        cached = _read_cache()
        if cached and cached.get("account_capital") == account_capital and cached.get("n_picks") == n_picks:
            cached["from_cache"] = True
            return cached

    universe = get_universe()
    history = load_universe(universe, period="3y", use_cache=True, progress=False)

    if len(history) < 20:
        return {"error": f"only {len(history)} stocks loaded; need 20+"}

    ranked = rank_universe(history, cfg=LIVE_RANKER_CONFIG)
    if ranked.empty:
        return {"error": "no stocks pass filters today"}

    picks = pick_top_n(ranked, n=n_picks, cfg=LIVE_RANKER_CONFIG)
    if picks.empty:
        return {"error": "no picks with composite > 0; market may be in risk-off"}

    tp_cfg = TradePlanConfig(account_capital=account_capital)
    plans = build_plans_for_picks(picks, account_capital=account_capital, cfg=tp_cfg)

    # Top 5 runners-up for transparency
    runners = ranked.head(8).copy()
    runners = runners[~runners["symbol"].isin([p.symbol for p in plans])]
    runners_list = []
    for _, r in runners.head(5).iterrows():
        runners_list.append({
            "symbol": r["symbol"],
            "sector": r["sector"],
            "composite": float(r["composite"]),
            "momentum_12_1": float(r.get("momentum_12_1", 0.0)),
        })

    plan_dicts = []
    for p in plans:
        plan_dicts.append({
            "symbol": p.symbol,
            "sector": p.sector,
            "entry": p.entry,
            "stop_loss": p.stop_loss,
            "target": p.target,
            "stop_distance_pct": p.stop_distance_pct,
            "reward_pct": p.reward_pct,
            "qty": p.qty,
            "capital_deployed": p.capital_deployed,
            "risk_inr": p.risk_inr,
            "composite": p.composite,
            "notes": p.notes,
        })

    payload = {
        "generated_at": int(time.time()),
        "account_capital": account_capital,
        "n_picks": n_picks,
        "n_universe_loaded": len(history),
        "n_passing_filters": len(ranked),
        "picks": plan_dicts,
        "runners_up": runners_list,
        "config_label": "HTR Monthly 2pk default (backtest 18.7% CAGR, Sharpe 1.02, MaxDD -26%)",
        "mode": "PAPER",  # explicit — not auto-executing
        "warning": (
            "Backtest beats NIFTY in 4/5 walk-forward windows but lagged in "
            "the most recent flat-NIFTY window. Recommend paper-tracking for "
            "3 months before deploying real capital."
        ),
        "from_cache": False,
    }
    _write_cache(payload)
    return payload
