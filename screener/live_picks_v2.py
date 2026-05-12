"""
Live picks producer — combines:
  1. Composite quant screen (stage2_trend + breakout_52w on LARGE caps)
  2. Mean-reversion screen (Connors RSI-2 on MID caps)
  3. Fundamentals snapshot (Piotroski-style health flags)
  4. AI agent review (LLM final-pass sanity check)
  5. Concrete trade levels (entry / SL / target / qty / hold days)

This is what the dashboard's Stock Screener panel will use going forward.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import pandas as pd

from .data_loader import load_universe
from .strategies import STRATEGIES
from .strategies.composite_top import CompositeTopStrategy, CompositeMidStrategy
from .universe import get_sector
from .universe_extended import LARGE_CAP, MID_CAP, get_cap_tier
from .levels import compute_levels, HOLD_DAYS_DEFAULT
from .fundamentals import fetch_fundamentals
from .ai_research_agent import StockResearchAgent, AIVerdict

CACHE_FILE = Path(__file__).resolve().parent.parent / "data" / "screener_live_v2.json"
CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
CACHE_TTL_SEC = 6 * 3600


def _rank_one_universe(strategy, history: dict, top_n: int = 5) -> list[dict]:
    """Score every stock in history with `strategy`, return top N picks."""
    rows = []
    for sym, df in history.items():
        if df is None or len(df) < 252:
            continue
        score = strategy.score(sym, df)
        if score is None or pd.isna(score):
            continue
        rows.append({"symbol": sym, "score": float(score)})
    rows.sort(key=lambda r: r["score"], reverse=True)
    return rows[:top_n]


def _enforce_sector_cap(ranked: list[dict], n: int, max_per_sector: int = 1) -> list[dict]:
    picks = []
    sec_count = {}
    for r in ranked:
        sec = get_sector(r["symbol"])
        if sec_count.get(sec, 0) >= max_per_sector:
            continue
        r["sector"] = sec
        picks.append(r)
        sec_count[sec] = sec_count.get(sec, 0) + 1
        if len(picks) >= n:
            break
    return picks


def generate(
    capital: float = 100_000.0,
    n_large: int = 2,
    n_mid: int = 1,
    enable_ai: bool = True,
    force_refresh: bool = False,
) -> dict:
    """
    Produce the live pick list with full trade plans.

    Default split: 2 LARGE composite + 1 MID mean-rev = 3 picks total.
    Cache TTL: 6h.
    """
    if not force_refresh and CACHE_FILE.exists():
        age = time.time() - CACHE_FILE.stat().st_mtime
        if age < CACHE_TTL_SEC:
            try:
                with open(CACHE_FILE, "r") as f:
                    payload = json.load(f)
                if payload.get("capital") == capital:
                    payload["from_cache"] = True
                    return payload
            except Exception:
                pass

    # ── 1. Fetch histories ──
    lg_hist = load_universe(LARGE_CAP, period="3y", use_cache=True, progress=False)
    mid_hist = load_universe(MID_CAP, period="3y", use_cache=True, progress=False)
    if len(lg_hist) < 20:
        return {"error": f"Only {len(lg_hist)} LARGE stocks loaded"}

    # ── 2. Score & pick ──
    lg_ranked = _rank_one_universe(CompositeTopStrategy(), lg_hist, top_n=10)
    lg_picks = _enforce_sector_cap(lg_ranked, n_large)
    mid_ranked = _rank_one_universe(CompositeMidStrategy(), mid_hist, top_n=10)
    mid_picks = _enforce_sector_cap(mid_ranked, n_mid)

    all_picks = []
    for p in lg_picks:
        p["cap_tier"] = "LARGE"
        all_picks.append(p)
    for p in mid_picks:
        p["cap_tier"] = "MID"
        all_picks.append(p)

    # ── 3. Fundamentals ──
    fund_map = {p["symbol"]: fetch_fundamentals(p["symbol"]) for p in all_picks}

    # ── 4. AI agent review ──
    history_map = {**lg_hist, **mid_hist}
    verdicts: dict[str, AIVerdict] = {}
    if enable_ai:
        try:
            agent = StockResearchAgent()
            for p in all_picks:
                v = agent.review(p["symbol"], {}, fund_map.get(p["symbol"], {}), p["sector"])
                verdicts[p["symbol"]] = v
        except Exception as e:
            print(f"AI agent unavailable: {e}")

    # ── 5. Compute levels (cap allocation: split capital between picks) ──
    # 70% to LARGE picks (split across n_large), 30% to MID (split across n_mid).
    # Each pick gets a position-size CAP equal to its share of capital.
    # Risk budget for SL sizing is 1% of TOTAL capital per trade.
    lg_pct_each = (0.70 / max(1, len(lg_picks)))
    mid_pct_each = (0.30 / max(1, len(mid_picks)))

    plans = []
    for p in all_picks:
        sym = p["symbol"]
        df = history_map.get(sym)
        per_trade_pct = lg_pct_each if p["cap_tier"] == "LARGE" else mid_pct_each
        v = verdicts.get(sym)
        plan = compute_levels(
            symbol=sym, sector=p["sector"], history=df,
            composite_score=p["score"],
            account_capital=capital,  # TOTAL capital
            risk_per_trade_pct=0.01,  # 1% risk per trade
            max_position_pct=per_trade_pct,  # cap per-pick at its allocation share
            ai_verdict=v.verdict if v else "",
            ai_conviction=v.conviction if v else "",
            ai_flags=v.flags if v else None,
            hold_days=v.hold_days_suggested if v else HOLD_DAYS_DEFAULT,
        )
        if plan is None:
            continue
        plan_dict = plan.to_dict()
        plan_dict["cap_tier"] = p["cap_tier"]
        plan_dict["ai_reasoning"] = v.reasoning if v else "AI disabled"
        # Annotate fundamentals snapshot for transparency
        fund = fund_map.get(sym, {})
        plan_dict["fundamentals_summary"] = {
            "pe": fund.get("trailingPE"),
            "pb": fund.get("priceToBook"),
            "roe": fund.get("returnOnEquity"),
            "de": fund.get("debtToEquity"),
            "rev_growth": fund.get("revenueGrowth"),
            "earnings_growth": fund.get("earningsGrowth"),
        }
        plans.append(plan_dict)

    # ── 6. Build response ──
    payload = {
        "generated_at": int(time.time()),
        "capital": capital,
        "n_large": n_large,
        "n_mid": n_mid,
        "picks": plans,
        "config": {
            "hold_days_default": HOLD_DAYS_DEFAULT,
            "rebal_freq": "Monthly (first business day)",
            "large_strategy": "composite_top (stage2 + breakout)",
            "mid_strategy": "mean_reversion (Connors RSI-2)",
            "allocation_split": "70% LARGE / 30% MID",
            "ai_enabled": enable_ai,
        },
        "backtest_summary": {
            "train_window": "2021-11 to 2025-11 (4.5y)",
            "test_window": "2025-11 to 2026-05 (6mo OOS)",
            "train_cagr": 0.3972,
            "train_sharpe": 1.92,
            "test_alpha_vs_nifty": 0.0274,
        },
        "from_cache": False,
    }
    try:
        with open(CACHE_FILE, "w") as f:
            json.dump(payload, f, indent=2, default=str)
    except Exception:
        pass
    return payload
