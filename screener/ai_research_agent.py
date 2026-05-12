"""
AI Research Agent for stock screening.

Wraps the existing Claude Haiku 4.5 brain (orchestrator/claude_market_brain.py)
to provide a second-pass review of every pick the technical composite
generates. The agent considers:

  - Sector outlook / macro context (from LLM general knowledge)
  - Earnings calendar risk (if upcoming earnings: caution)
  - Fundamental snapshot (yfinance .info: PE, ROE, debt, margin)
  - Recent price action context (52w high distance, recent volatility)

The agent does NOT generate picks — it only reviews picks the quant
strategy already selected. This avoids the "LLM hallucinates prices"
trap while still benefiting from the LLM's broad knowledge.

Returns per pick:
  - verdict: BUY | CAUTION | SKIP
  - conviction: HIGH | MEDIUM | LOW
  - hold_days_suggested: int (override the default 30 if needed)
  - reasoning: short text
  - flags: list[str] (e.g. ['EARNINGS_NEAR', 'SECTOR_HEADWIND'])
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from typing import Optional

import pandas as pd

# Reuse the existing AI brain infrastructure
try:
    from orchestrator.claude_market_brain import ClaudeMarketBrain
    _BRAIN_AVAILABLE = True
except Exception:
    _BRAIN_AVAILABLE = False


@dataclass
class AIVerdict:
    symbol: str
    verdict: str = "SKIP"            # BUY | CAUTION | SKIP
    conviction: str = "LOW"          # HIGH | MEDIUM | LOW
    hold_days_suggested: int = 30
    reasoning: str = ""
    flags: list[str] = None
    raw_response: Optional[str] = None

    def __post_init__(self):
        if self.flags is None:
            self.flags = []


class StockResearchAgent:
    """LLM-powered reviewer for quant-screened picks."""

    def __init__(self, brain: Optional[ClaudeMarketBrain] = None):
        if brain is None and _BRAIN_AVAILABLE:
            try:
                brain = ClaudeMarketBrain()
            except Exception:
                brain = None
        self.brain = brain

    def _build_prompt(self, symbol: str, technicals: dict, fundamentals: dict, sector: str) -> str:
        """Build the review prompt for a single stock."""
        tech_lines = []
        for k in ("last_close", "momentum_12_1", "stage2_score", "breakout_dist",
                  "atr_pct", "vol_252", "above_200dma"):
            v = technicals.get(k)
            if v is not None and v == v:  # not NaN
                if isinstance(v, float):
                    tech_lines.append(f"  - {k}: {v:.4f}")
                else:
                    tech_lines.append(f"  - {k}: {v}")
        tech_str = "\n".join(tech_lines) if tech_lines else "  (no technicals)"

        fund_lines = []
        for k in ("trailingPE", "priceToBook", "returnOnEquity", "debtToEquity",
                  "profitMargins", "operatingMargins", "earningsGrowth",
                  "revenueGrowth", "currentRatio"):
            v = fundamentals.get(k)
            if v is not None:
                if isinstance(v, float):
                    fund_lines.append(f"  - {k}: {v:.3f}")
                else:
                    fund_lines.append(f"  - {k}: {v}")
        fund_str = "\n".join(fund_lines) if fund_lines else "  (no fundamentals available)"

        return f"""You are reviewing a NSE Indian equity that a quant momentum/breakout screener has picked.
Your job is to either CONFIRM, FLAG WITH CAUTION, or REJECT, based on whether
anything in the fundamental or sector context contradicts the quant signal.

STOCK: {symbol}  (sector: {sector})

TECHNICAL SIGNAL:
{tech_str}

FUNDAMENTAL SNAPSHOT (point-in-time, from yfinance):
{fund_str}

Output a strict JSON object with EXACTLY these keys:
{{
  "verdict": "BUY" | "CAUTION" | "SKIP",
  "conviction": "HIGH" | "MEDIUM" | "LOW",
  "hold_days_suggested": <integer between 5 and 90>,
  "reasoning": "<one sentence under 30 words>",
  "flags": [<list of strings from: EARNINGS_NEAR, SECTOR_HEADWIND, HIGH_DEBT, OVERVALUED, FRAUD_RISK, GROWTH_STAGNANT, MOMENTUM_FADING, BULLISH_SETUP, FUNDAMENTAL_STRONG, NONE>]
}}

Rules:
- BUY only if technical + fundamental BOTH support it.
- CAUTION if technical is strong but fundamental is weak (or vice versa).
- SKIP if there's a serious red flag.
- DO NOT output anything else. Only the JSON object."""

    def review(self, symbol: str, technicals: dict, fundamentals: dict, sector: str = "OTHER") -> AIVerdict:
        """Send a single stock to the LLM for review. Returns AIVerdict."""
        if self.brain is None:
            return AIVerdict(
                symbol=symbol, verdict="BUY", conviction="MEDIUM",
                reasoning="AI brain unavailable — passing through quant signal",
                flags=["AI_UNAVAILABLE"],
            )

        prompt = self._build_prompt(symbol, technicals, fundamentals, sector)
        try:
            # ClaudeMarketBrain._call_llm returns a dict (auto-parses JSON
            # from LLM response). Reuse that path so we benefit from the
            # provider rotation chain (groq -> gemini -> haiku) for free.
            payload = self.brain._call_llm(prompt, max_tokens=300, use_fast=True)
        except Exception as e:
            return AIVerdict(
                symbol=symbol, verdict="BUY", conviction="LOW",
                reasoning=f"AI review failed: {e}",
                flags=["AI_ERROR"], raw_response=str(e),
            )

        try:
            return AIVerdict(
                symbol=symbol,
                verdict=str(payload.get("verdict", "SKIP")).upper(),
                conviction=str(payload.get("conviction", "LOW")).upper(),
                hold_days_suggested=int(payload.get("hold_days_suggested", 30)),
                reasoning=str(payload.get("reasoning", "")),
                flags=list(payload.get("flags", [])),
                raw_response=json.dumps(payload, default=str),
            )
        except Exception as e:
            return AIVerdict(
                symbol=symbol, verdict="BUY", conviction="LOW",
                reasoning=f"Parse error: {e}",
                flags=["PARSE_ERROR"], raw_response=str(payload),
            )

    def review_batch(
        self,
        picks: list[dict],
        history_map: dict[str, pd.DataFrame],
        fundamentals_map: dict[str, dict],
    ) -> list[AIVerdict]:
        """Review a list of picks. Each pick dict has 'symbol', 'sector', 'composite' etc."""
        verdicts = []
        for pick in picks:
            sym = pick["symbol"]
            history = history_map.get(sym, pd.DataFrame())
            fund = fundamentals_map.get(sym, {})
            tech = {
                "last_close": float(history["Close"].iloc[-1]) if len(history) else None,
                "momentum_12_1": pick.get("momentum_12_1"),
                "stage2_score": pick.get("stage2_score"),
                "breakout_dist": pick.get("breakout_dist"),
                "atr_pct": pick.get("atr_pct"),
                "vol_252": pick.get("vol_252"),
                "above_200dma": pick.get("above_200dma"),
            }
            v = self.review(sym, tech, fund, pick.get("sector", "OTHER"))
            verdicts.append(v)
        return verdicts
