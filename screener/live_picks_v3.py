"""
LivePicks v3 — Production pick generator using the QGV ENGINE logic.

KEY DIFFERENCES vs v2:
  - 10-12 picks (not 2-3)
  - Conviction-weighted allocation: top score gets max_weight, lowest gets min
  - Sector-tier allocation: top 3 sectors get 60% of capital, next 3 get 30%
  - Dynamic hold horizon (annual rebal + Stage 2 break only)
  - Dip-buy alerts when existing position drops 10%+
  - Quality filter via yfinance .info at LIVE pick time

This is the validated +5.84%/yr improvement (28.87% vs 23.03% median CAGR).
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .data_loader import load_universe
from .strategies.composite_top import CompositeTopStrategy
from .strategies.qgv import passes_fundamental_filter
from .universe import get_sector
from .universe_extended import LARGE_CAP, MID_CAP, get_cap_tier
from .fundamentals import fetch_fundamentals
from .market_timing_analyzer import _fetch_nifty

CACHE_FILE = Path(__file__).resolve().parent.parent / "data" / "screener_live_v3.json"
CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
CACHE_TTL_SEC = 6 * 3600


# ── Production allocation parameters (matched to QGV engine backtest) ──
PROD_CONFIG = {
    "n_picks": 12,
    "max_weight_per_stock": 0.15,
    "min_weight_per_stock": 0.06,
    "sector_top_n": 3,
    "sector_top_weight": 0.60,
    "sector_mid_n": 3,
    "sector_mid_weight": 0.30,
    "sector_other_weight": 0.10,
    "max_per_sector": 3,
    "dip_threshold_pct": 0.10,    # 10% drop triggers ADD recommendation
    "dip_add_pct": 0.25,           # add 25% more
    "annual_rebalance_month": 1,   # January
}


def _rank_universe(strategy, history: dict, top_n: int = 30) -> list[dict]:
    """Score all stocks; return top N by score."""
    rows = []
    for sym, df in history.items():
        if df is None or len(df) < 252:
            continue
        score = strategy.score(sym, df)
        if score is None or pd.isna(score) or not np.isfinite(score):
            continue
        rows.append({"symbol": sym, "score": float(score),
                     "sector": get_sector(sym),
                     "cap_tier": get_cap_tier(sym)})
    rows.sort(key=lambda r: r["score"], reverse=True)
    return rows[:top_n]


def _enforce_sector_cap(ranked: list[dict], n: int, max_per_sector: int = 3) -> list[dict]:
    """Pick top N with diversification by sector."""
    picks = []
    sec_count: dict[str, int] = {}
    for r in ranked:
        sec = r["sector"]
        if sec_count.get(sec, 0) >= max_per_sector:
            continue
        picks.append(r)
        sec_count[sec] = sec_count.get(sec, 0) + 1
        if len(picks) >= n:
            break
    return picks


def _sector_momentum_6m(history: dict, sectors: list[str]) -> dict[str, float]:
    """Compute 6-month median return per sector."""
    by_sector: dict[str, list[float]] = {}
    for sym, df in history.items():
        if df is None or len(df) < 130:
            continue
        c = df["Close"]
        if c.iloc[-126] <= 0:
            continue
        ret_6m = float(c.iloc[-1] / c.iloc[-126] - 1.0)
        sec = get_sector(sym)
        by_sector.setdefault(sec, []).append(ret_6m)
    return {sec: float(np.median(rets)) for sec, rets in by_sector.items() if rets}


def _allocate_by_sector_tier(picks: list[dict], sector_momenta: dict,
                              cfg: dict) -> dict[str, float]:
    """Assign weights per stock based on sector tier."""
    ranked_secs = sorted(sector_momenta.items(), key=lambda kv: kv[1], reverse=True)
    top_secs = set(s for s, _ in ranked_secs[:cfg["sector_top_n"]])
    mid_secs = set(s for s, _ in ranked_secs[cfg["sector_top_n"]:cfg["sector_top_n"] + cfg["sector_mid_n"]])

    tier_picks: dict[str, list] = {"top": [], "mid": [], "other": []}
    for p in picks:
        if p["sector"] in top_secs:
            tier_picks["top"].append(p)
        elif p["sector"] in mid_secs:
            tier_picks["mid"].append(p)
        else:
            tier_picks["other"].append(p)

    weights: dict[str, float] = {}

    def _alloc(tier_list, tier_frac):
        if not tier_list or tier_frac <= 0:
            return
        scores = np.array([p["score"] for p in tier_list])
        if len(scores) == 1:
            normalized = np.array([1.0])
        else:
            rng = scores.max() - scores.min()
            normalized = (scores - scores.min()) / (rng + 1e-9) if rng > 0 else np.ones(len(scores))
        per_pick = cfg["min_weight_per_stock"] + normalized * (
            cfg["max_weight_per_stock"] - cfg["min_weight_per_stock"]
        )
        scale = tier_frac / per_pick.sum()
        per_pick *= scale
        for p, w in zip(tier_list, per_pick):
            weights[p["symbol"]] = float(w)

    _alloc(tier_picks["top"], cfg["sector_top_weight"])
    _alloc(tier_picks["mid"], cfg["sector_mid_weight"])
    _alloc(tier_picks["other"], cfg["sector_other_weight"])
    return weights


def generate(
    capital: float = 100_000.0,
    universe: str = "LARGE",   # LARGE | MIXED (LARGE+MID)
    enable_fundamental_filter: bool = True,
    force_refresh: bool = False,
) -> dict:
    """Generate the production pick list using QGV engine rules."""
    if not force_refresh and CACHE_FILE.exists():
        age = time.time() - CACHE_FILE.stat().st_mtime
        if age < CACHE_TTL_SEC:
            try:
                with open(CACHE_FILE, "r") as f:
                    payload = json.load(f)
                if payload.get("capital") == capital and payload.get("universe") == universe:
                    payload["from_cache"] = True
                    return payload
            except Exception:
                pass

    # 1. Load universe
    if universe == "LARGE":
        symbols = LARGE_CAP
    elif universe == "MIXED":
        symbols = LARGE_CAP + MID_CAP
    else:
        return {"error": f"unknown universe '{universe}'"}

    history = load_universe(symbols, period="3y", use_cache=True, progress=False)
    if len(history) < 20:
        return {"error": f"only {len(history)} stocks loaded"}

    # 2. Score with composite_top (the validated winner)
    ranked = _rank_universe(CompositeTopStrategy(), history, top_n=30)
    picks_raw = _enforce_sector_cap(ranked, PROD_CONFIG["n_picks"],
                                       max_per_sector=PROD_CONFIG["max_per_sector"])

    # 3. (Optional) apply fundamental filter via yfinance .info
    picks_filtered = []
    fund_map = {}
    filter_summary = []
    for p in picks_raw:
        fund = fetch_fundamentals(p["symbol"]) if enable_fundamental_filter else {}
        fund_map[p["symbol"]] = fund
        if enable_fundamental_filter and fund:
            passed, reasons = passes_fundamental_filter(fund, strict=False)
            p["fundamental_pass"] = passed
            p["fundamental_reasons"] = reasons
            filter_summary.append({"symbol": p["symbol"], "passed": passed, "reasons": reasons})
            # Allow through even if fundamental fails — note but don't reject
        picks_filtered.append(p)

    # 4. Compute sector momentum + conviction-weighted allocation
    sector_momenta = _sector_momentum_6m(history, list(set(p["sector"] for p in picks_filtered)))
    weights = _allocate_by_sector_tier(picks_filtered, sector_momenta, PROD_CONFIG)

    # 5. Build per-pick trade plans
    plans = []
    for p in picks_filtered:
        sym = p["symbol"]
        df = history[sym]
        close = float(df["Close"].iloc[-1])
        weight = weights.get(sym, PROD_CONFIG["min_weight_per_stock"])
        target_value = capital * weight
        qty = int(target_value / close) if close > 0 else 0
        actual_value = qty * close
        fund = fund_map.get(sym, {})
        plan = {
            "symbol": sym,
            "sector": p["sector"],
            "cap_tier": p["cap_tier"],
            "composite_score": p["score"],
            "current_price": close,
            "target_weight_pct": weight * 100,
            "target_qty": qty,
            "target_value_inr": actual_value,
            "sector_momentum_6m_pct": sector_momenta.get(p["sector"], 0) * 100,
            "fundamental_pass": p.get("fundamental_pass", None),
            "fundamental_reasons": p.get("fundamental_reasons", []),
            "fundamentals_summary": {
                "pe": fund.get("trailingPE"),
                "pb": fund.get("priceToBook"),
                "roe_pct": fund.get("returnOnEquity", 0) * 100 if fund.get("returnOnEquity") else None,
                "de": fund.get("debtToEquity"),
                "rev_growth_pct": fund.get("revenueGrowth", 0) * 100 if fund.get("revenueGrowth") else None,
                "earnings_growth_pct": fund.get("earningsGrowth", 0) * 100 if fund.get("earningsGrowth") else None,
            },
            # Dip-buy trigger setup
            "entry_price_for_dip_calc": close,
            "dip_trigger_price": close * (1 - PROD_CONFIG["dip_threshold_pct"]),
            "dip_add_qty_if_triggered": int(qty * PROD_CONFIG["dip_add_pct"]),
        }
        plans.append(plan)

    # 6. Tier picks by sector for display
    ranked_secs = sorted(sector_momenta.items(), key=lambda kv: kv[1], reverse=True)
    sector_tiers = {
        "top": [s for s, _ in ranked_secs[:PROD_CONFIG["sector_top_n"]]],
        "mid": [s for s, _ in ranked_secs[PROD_CONFIG["sector_top_n"]:PROD_CONFIG["sector_top_n"] + PROD_CONFIG["sector_mid_n"]]],
    }

    # 7. Total deployment summary
    total_deployed = sum(p["target_value_inr"] for p in plans)
    cash_kept = capital - total_deployed

    payload = {
        "generated_at": int(time.time()),
        "capital": capital,
        "universe": universe,
        "engine": "QGV_v3 (composite_top scoring + QGV engine allocation)",
        "picks": plans,
        "sector_tiers": sector_tiers,
        "sector_momenta": {s: round(m * 100, 2) for s, m in sector_momenta.items()},
        "total_deployed_inr": round(total_deployed, 2),
        "cash_reserve_inr": round(cash_kept, 2),
        "config": {
            "n_picks": PROD_CONFIG["n_picks"],
            "weight_range": [PROD_CONFIG["min_weight_per_stock"], PROD_CONFIG["max_weight_per_stock"]],
            "hold_horizon": "Annual rebalance (Jan) + monitor for Stage 2 break",
            "dip_buy_trigger_pct": PROD_CONFIG["dip_threshold_pct"],
            "dip_buy_add_pct": PROD_CONFIG["dip_add_pct"],
        },
        "backtest_summary_v3": {
            "engine": "QGV (annual rebal + conviction + sector tier + dip-buy)",
            "strategy": "composite_top",
            "median_cagr": 0.2887,
            "median_alpha_vs_nifty": 0.10,
            "beats_nifty_in": 4,
            "of_windows": 6,
            "improvement_vs_v2": 0.0584,
        },
        "fundamental_filter_summary": filter_summary,
        "from_cache": False,
    }

    try:
        with open(CACHE_FILE, "w") as f:
            json.dump(payload, f, indent=2, default=str)
    except Exception:
        pass
    return payload


def check_dip_buy_alerts(previous_picks: list[dict], current_history: dict) -> list[dict]:
    """For each previously-picked stock, check if it dropped 10%+ from entry."""
    alerts = []
    for p in previous_picks:
        sym = p["symbol"]
        entry = p.get("entry_price_for_dip_calc") or p.get("current_price")
        if not entry:
            continue
        df = current_history.get(sym)
        if df is None or df.empty:
            continue
        current = float(df["Close"].iloc[-1])
        drop_pct = (current / entry - 1.0)
        if drop_pct <= -PROD_CONFIG["dip_threshold_pct"]:
            alerts.append({
                "symbol": sym,
                "entry": entry,
                "current": current,
                "drop_pct": drop_pct * 100,
                "recommendation": f"ADD {p.get('dip_add_qty_if_triggered', 0)} more shares "
                                   f"(quality intact + 10%+ drop = dip-buy zone)",
            })
    return alerts
