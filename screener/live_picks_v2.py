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

    # ── 4. AI agent TEAM review (7-agent research team) ──
    history_map = {**lg_hist, **mid_hist}
    verdicts: dict[str, AIVerdict] = {}
    team_verdicts: dict[str, dict] = {}  # full team report per symbol
    if enable_ai:
        try:
            from agent_team import ResearchTeam
            from agent_team.context_builder import build_context
            from screener.market_timing_analyzer import _fetch_nifty
            # Fetch NIFTY once for the macro/risk context
            try:
                nifty_h = _fetch_nifty()
            except Exception:
                nifty_h = None
            # Build a sector-peer map for sector context
            sector_peers_by_sym: dict[str, dict] = {}
            for p in all_picks:
                sym = p["symbol"]
                peers = {s: history_map[s] for s in history_map
                          if get_sector(s) == p["sector"] and s in history_map}
                sector_peers_by_sym[sym] = peers
            # Build macro context (same for every pick on this day)
            macro_extra = {}
            if nifty_h is not None and not nifty_h.empty:
                c = nifty_h["Close"]
                close_n = float(c.iloc[-1])
                ma_200 = float(c.rolling(200).mean().iloc[-1]) if len(c) >= 200 else close_n
                ma_50  = float(c.rolling(50).mean().iloc[-1]) if len(c) >= 50 else close_n
                high_252 = float(nifty_h["High"].tail(252).max())
                macro_extra = {
                    "nifty_close": close_n,
                    "nifty_dist_high_pct": (close_n/high_252 - 1.0) if high_252 > 0 else 0,
                    "nifty_dist_200dma_pct": (close_n/ma_200 - 1.0) if ma_200 > 0 else 0,
                    "above_200dma": close_n > ma_200,
                    "golden_cross": ma_50 > ma_200,
                }
            team = ResearchTeam(prefer_fast=True, enable_llm_arbitration=False)
            for p in all_picks:
                sym = p["symbol"]
                ctx = build_context(
                    symbol=sym, sector=p["sector"],
                    history=history_map.get(sym),
                    fundamentals=fund_map.get(sym, {}),
                    nifty_history=nifty_h,
                    sector_peers=sector_peers_by_sym.get(sym, {}),
                    macro_extra=macro_extra,
                )
                tv = team.review(sym, ctx)
                # Translate TeamVerdict back to AIVerdict shape for the
                # downstream sizing logic in compute_levels()
                v = AIVerdict(
                    symbol=sym,
                    verdict="BUY" if tv.final_action == "BUY" else
                            ("SKIP" if tv.final_action == "SKIP" else "CAUTION"),
                    conviction="HIGH" if tv.confidence >= 0.7 else
                               ("LOW" if tv.confidence < 0.4 else "MEDIUM"),
                    hold_days_suggested=tv.hold_days,
                    reasoning=tv.coordinator_note,
                    flags=list({f for r in tv.reports for f in r.flags}),
                )
                # Apply suggested_qty_mult from team into a flag the
                # caller (compute_levels) can read.
                if tv.suggested_qty_mult <= 0.5 and v.verdict == "BUY":
                    v.verdict = "CAUTION"
                verdicts[sym] = v
                team_verdicts[sym] = {
                    "final_action": tv.final_action,
                    "final_score": tv.final_score,
                    "confidence": tv.confidence,
                    "suggested_qty_mult": tv.suggested_qty_mult,
                    "hold_days": tv.hold_days,
                    "coordinator_note": tv.coordinator_note,
                    "reports": [
                        {"agent": r.agent_name, "score": r.score,
                         "verdict": r.verdict, "confidence": r.confidence,
                         "flags": r.flags, "one_liner": r.one_liner,
                         "error": r.error}
                        for r in tv.reports
                    ],
                }
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
    # Updated 2026-05-13 to reflect the honest rolling walk-forward
    # (engine bugs fixed, anti-look-ahead, MF-style hold-until-deterioration).
    payload = {
        "generated_at": int(time.time()),
        "capital": capital,
        "n_large": n_large,
        "n_mid": n_mid,
        "picks": plans,
        "team_review": team_verdicts,   # full 7-agent reports per symbol
        "config": {
            "hold_days_default": HOLD_DAYS_DEFAULT,
            "rebal_freq": "Monthly (first business day)",
            "large_strategy": "composite_top (stage2 + breakout)",
            "mid_strategy": "mean_reversion (Connors RSI-2)",
            "allocation_split": "70% LARGE / 30% MID",
            "ai_enabled": enable_ai,
            "agent_team": "7-agent research team (sector, fundamental, "
                           "technical, risk, macro, event, PM)",
        },
        "backtest_summary": {
            "methodology": "rolling 18-month windows across 4.5y",
            "n_windows": 6,
            "n_beat_nifty": 4,
            "median_strategy_cagr": 0.2481,
            "median_nifty_cagr": 0.1304,
            "median_alpha": 0.1008,
            "median_sharpe": 1.17,
            "median_max_dd": -0.2159,
            "worst_window_cagr": -0.1359,
            "best_window_cagr": 0.5093,
        },
        "from_cache": False,
    }
    try:
        with open(CACHE_FILE, "w") as f:
            json.dump(payload, f, indent=2, default=str)
    except Exception:
        pass
    return payload
