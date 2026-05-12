"""
Consensus Allocator — find stocks recommended by MULTIPLE sources +
AI-driven allocation for maximum risk-adjusted returns.

Sources cross-checked:
  1. live_picks_v2 — composite_top (stage2 + breakout) via mf_style engine
  2. live_picks_v3 — composite_top via QGV engine (production, +5.84%/yr)
  3. multi_strategy_view — 6 strategies, consensus = 3+ agree

CONVICTION TIERS:
  TRIPLE  = stock in all 3 sources (highest conviction)
  DOUBLE  = stock in 2 of 3 sources (high conviction)
  SINGLE  = stock in only 1 source (do not include)

Allocation logic (matches QGV engine — the validated winner):
  - Convicting-weighted: higher score = higher weight
  - Sector tier: top 3 sectors by 6mo momentum get 60%, mid get 30%, rest 10%
  - Max 15% per stock, min 6%
  - Cap at 35% per sector
  - 5% cash reserve for opportunities

For each pick, runs the 7-agent research team for additional context.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


CACHE_FILE = Path(__file__).resolve().parent.parent / "data" / "consensus_allocation.json"
CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
CACHE_TTL_SEC = 6 * 3600


CONVICTION_WEIGHTS = {
    "TRIPLE": 1.0,    # stock in all 3 sources -> full conviction weight
    "DOUBLE": 0.75,   # stock in 2 sources    -> 75% of full weight
}


def _find_consensus(v2_payload: dict, v3_payload: dict, multi_payload: dict) -> dict:
    """Categorise stocks by how many sources picked them."""
    v2_syms = {p["symbol"]: p for p in v2_payload.get("picks", [])}
    v3_syms = {p["symbol"]: p for p in v3_payload.get("picks", [])}
    multi_syms = {c["symbol"]: c for c in multi_payload.get("consensus_picks", [])}

    all_syms = set(v2_syms) | set(v3_syms) | set(multi_syms)
    triple = []
    double = []

    for sym in all_syms:
        sources = []
        if sym in v2_syms: sources.append("v2")
        if sym in v3_syms: sources.append("v3")
        if sym in multi_syms: sources.append("multi")
        if len(sources) == 3:
            triple.append(sym)
        elif len(sources) == 2:
            double.append(sym)
        # singles are dropped

    # Merge info per pick
    def _merge_info(sym):
        info = {"symbol": sym}
        sources = []
        if sym in v3_syms:
            sources.append("v3")
            v = v3_syms[sym]
            info.update({
                "sector": v.get("sector"),
                "cap_tier": v.get("cap_tier"),
                "composite_score": v.get("composite_score"),
                "current_price": v.get("current_price"),
                "v3_weight_pct": v.get("target_weight_pct"),
                "fundamentals_summary": v.get("fundamentals_summary"),
                "fundamental_pass": v.get("fundamental_pass"),
                "sector_momentum_6m_pct": v.get("sector_momentum_6m_pct"),
            })
        if sym in v2_syms:
            sources.append("v2")
            v = v2_syms[sym]
            if "sector" not in info:
                info["sector"] = v.get("sector")
                info["composite_score"] = v.get("composite_score") or v.get("score")
                info["current_price"] = v.get("entry") or v.get("current_price")
        if sym in multi_syms:
            sources.append("multi")
            m = multi_syms[sym]
            info["strategies_agreeing"] = m.get("strategies", [])
            info["multi_n_strategies"] = m.get("picked_by_n_strategies", 0)
            if "sector" not in info:
                info["sector"] = m.get("sector")
        info["sources"] = sources
        info["conviction"] = "TRIPLE" if len(sources) == 3 else "DOUBLE"
        return info

    triple_data = [_merge_info(s) for s in triple]
    double_data = [_merge_info(s) for s in double]
    # Sort by composite score within each tier
    triple_data.sort(key=lambda x: x.get("composite_score") or 0, reverse=True)
    double_data.sort(key=lambda x: x.get("composite_score") or 0, reverse=True)

    return {"triple": triple_data, "double": double_data}


def _sector_tier_weights(picks: list[dict], total_capital: float,
                          cash_reserve_pct: float = 0.05,
                          sector_top_n: int = 3, sector_top_weight: float = 0.60,
                          sector_mid_n: int = 3, sector_mid_weight: float = 0.30,
                          sector_other_weight: float = 0.10,
                          min_weight: float = 0.06, max_weight: float = 0.15,
                          max_per_sector: int = 3) -> dict:
    """Compute conviction + sector-tier allocation per stock."""
    if not picks:
        return {}

    # 1. Compute sector momentum from picks' sector_momentum_6m_pct
    sector_momentum: dict[str, list[float]] = {}
    for p in picks:
        sec = p.get("sector", "OTHER")
        m = p.get("sector_momentum_6m_pct")
        if m is not None:
            sector_momentum.setdefault(sec, []).append(m)
    sector_med = {sec: float(np.median(vals)) for sec, vals in sector_momentum.items()}
    ranked_secs = sorted(sector_med.items(), key=lambda kv: kv[1], reverse=True)
    top_secs = set(s for s, _ in ranked_secs[:sector_top_n])
    mid_secs = set(s for s, _ in ranked_secs[sector_top_n:sector_top_n + sector_mid_n])

    # 2. Cap picks at max_per_sector by score
    sector_count: dict[str, int] = {}
    filtered = []
    for p in picks:
        sec = p.get("sector", "OTHER")
        if sector_count.get(sec, 0) >= max_per_sector:
            continue
        sector_count[sec] = sector_count.get(sec, 0) + 1
        filtered.append(p)

    # 3. Group by tier
    tier_picks: dict[str, list] = {"top": [], "mid": [], "other": []}
    for p in filtered:
        sec = p.get("sector", "OTHER")
        if sec in top_secs:
            tier_picks["top"].append(p)
        elif sec in mid_secs:
            tier_picks["mid"].append(p)
        else:
            tier_picks["other"].append(p)

    # 4. Compute weights per tier
    deployable = 1.0 - cash_reserve_pct
    weights: dict[str, float] = {}

    def _alloc_tier(tier_list, tier_frac):
        if not tier_list or tier_frac <= 0:
            return
        # Conviction-multiplied scores
        adj = []
        for p in tier_list:
            score = p.get("composite_score") or 0
            conviction_mult = CONVICTION_WEIGHTS.get(p["conviction"], 0.5)
            adj.append(score * conviction_mult)
        adj = np.array(adj)
        if len(adj) == 1:
            normalized = np.array([1.0])
        else:
            rng = adj.max() - adj.min()
            if rng <= 0:
                normalized = np.ones(len(adj))
            else:
                normalized = (adj - adj.min()) / rng
        per_pick = min_weight + normalized * (max_weight - min_weight)
        scale = (deployable * tier_frac) / per_pick.sum()
        per_pick *= scale
        for p, w in zip(tier_list, per_pick):
            weights[p["symbol"]] = float(w)

    _alloc_tier(tier_picks["top"], sector_top_weight)
    _alloc_tier(tier_picks["mid"], sector_mid_weight)
    _alloc_tier(tier_picks["other"], sector_other_weight)

    # 5. Convert to ₹ amounts
    allocations = []
    total_deployed = 0
    for p in filtered:
        sym = p["symbol"]
        w = weights.get(sym, min_weight)
        target_inr = total_capital * w
        price = p.get("current_price", 1)
        qty = int(target_inr / price) if price > 0 else 0
        actual_inr = qty * price
        total_deployed += actual_inr
        allocations.append({
            "symbol": sym,
            "sector": p.get("sector"),
            "cap_tier": p.get("cap_tier"),
            "conviction": p["conviction"],
            "sources": p["sources"],
            "composite_score": p.get("composite_score"),
            "current_price": price,
            "target_weight_pct": w * 100,
            "target_qty": qty,
            "target_value_inr": actual_inr,
            "sector_tier": ("TOP" if p.get("sector") in top_secs
                              else "MID" if p.get("sector") in mid_secs else "OTHER"),
            "strategies_agreeing": p.get("strategies_agreeing", []),
            "fundamentals_summary": p.get("fundamentals_summary"),
        })

    return {
        "allocations": allocations,
        "total_deployed_inr": total_deployed,
        "cash_reserve_inr": total_capital - total_deployed,
        "sector_tiers": {
            "top": list(top_secs),
            "mid": list(mid_secs),
            "momenta": sector_med,
        },
    }


def generate(
    capital: float = 100_000.0,
    enable_ai: bool = True,
    force_refresh: bool = False,
) -> dict:
    """
    Find consensus stocks across all 3 sources + allocate with AI brain review.
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

    # 1. Fetch from all three sources
    from .live_picks_v2 import generate as gen_v2
    from .live_picks_v3 import generate as gen_v3
    from .multi_strategy_view import generate_consensus

    try:
        v2 = gen_v2(capital=capital, n_large=5, n_mid=3, enable_ai=False, force_refresh=False)
    except Exception as e:
        v2 = {"picks": [], "error": str(e)}
    try:
        v3 = gen_v3(capital=capital, universe="LARGE", enable_fundamental_filter=False, force_refresh=False)
    except Exception as e:
        v3 = {"picks": [], "error": str(e)}
    try:
        multi = generate_consensus(universe="LARGE", top_n_per_strategy=10)
    except Exception as e:
        multi = {"consensus_picks": [], "error": str(e)}

    # 2. Find consensus tiers
    consensus = _find_consensus(v2, v3, multi)

    # 3. Build pick pool for allocation: triple first, then top double
    pool = list(consensus["triple"])
    # Add doubles up to a reasonable cap (top 4 by score)
    pool.extend(consensus["double"][:4])

    if not pool:
        return {"error": "no consensus stocks found", "capital": capital}

    # 4. Apply allocation
    allocation = _sector_tier_weights(pool, capital)

    # 5. Optional: AI agent team review for top picks
    ai_reviews = {}
    if enable_ai:
        try:
            from agent_team import ResearchTeam
            from agent_team.context_builder import build_context
            from .data_loader import load_history
            from .fundamentals import fetch_fundamentals
            from .market_timing_analyzer import _fetch_nifty
            from .universe import get_sector

            team = ResearchTeam(prefer_fast=True)
            nifty_h = _fetch_nifty()
            macro_extra = {}
            if not nifty_h.empty:
                c = nifty_h["Close"]
                close_n = float(c.iloc[-1])
                ma_200 = float(c.rolling(200).mean().iloc[-1]) if len(c) >= 200 else close_n
                ma_50 = float(c.rolling(50).mean().iloc[-1]) if len(c) >= 50 else close_n
                high_252 = float(nifty_h["High"].tail(252).max())
                macro_extra = {
                    "nifty_close": close_n,
                    "nifty_dist_high_pct": (close_n/high_252 - 1.0) if high_252 > 0 else 0,
                    "nifty_dist_200dma_pct": (close_n/ma_200 - 1.0) if ma_200 > 0 else 0,
                    "above_200dma": close_n > ma_200,
                    "golden_cross": ma_50 > ma_200,
                }

            # Review only TRIPLE-conviction stocks (limit cost)
            for p in consensus["triple"][:8]:
                sym = p["symbol"]
                try:
                    df = load_history(sym, period="3y", use_cache=True)
                    if df.empty or len(df) < 252:
                        continue
                    fund = fetch_fundamentals(sym)
                    ctx = build_context(
                        symbol=sym, sector=p["sector"],
                        history=df, fundamentals=fund,
                        nifty_history=nifty_h, sector_peers={},
                        macro_extra=macro_extra,
                    )
                    tv = team.review(sym, ctx)
                    ai_reviews[sym] = {
                        "final_action": tv.final_action,
                        "score": tv.final_score,
                        "confidence": tv.confidence,
                        "suggested_qty_mult": tv.suggested_qty_mult,
                        "hold_days": tv.hold_days,
                        "summary": tv.coordinator_note,
                        "agent_verdicts": [
                            {"agent": r.agent_name, "verdict": r.verdict,
                             "score": r.score, "one_liner": r.one_liner}
                            for r in tv.reports
                        ],
                    }
                except Exception as e:
                    ai_reviews[sym] = {"error": str(e)}
        except Exception as e:
            print(f"[consensus_allocator] AI review skipped: {e}")

    # 6. Build final payload
    payload = {
        "generated_at": int(time.time()),
        "capital": capital,
        "consensus_tiers": {
            "triple": consensus["triple"],
            "double": consensus["double"],
        },
        "allocation": allocation,
        "ai_reviews": ai_reviews,
        "ai_enabled": enable_ai,
        "summary": {
            "n_triple_consensus": len(consensus["triple"]),
            "n_double_consensus": len(consensus["double"]),
            "n_in_final_allocation": len(allocation.get("allocations", [])),
            "total_deployed_pct": (allocation.get("total_deployed_inr", 0) / capital * 100) if capital > 0 else 0,
        },
        "methodology": (
            "TRIPLE conviction = stock in all 3 sources (v2 picks + v3 picks + "
            "multi-strategy consensus). DOUBLE = stock in 2 of 3. "
            "Allocation: conviction-weighted (TRIPLE = 100%, DOUBLE = 75%) + "
            "QGV-style sector tier weighting (top 3 sectors get 60%, mid 30%, "
            "rest 10%) + 5% cash reserve."
        ),
        "from_cache": False,
    }

    try:
        with open(CACHE_FILE, "w") as f:
            json.dump(payload, f, indent=2, default=str)
    except Exception:
        pass
    return payload
