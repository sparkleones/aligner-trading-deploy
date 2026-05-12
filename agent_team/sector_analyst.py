"""
Sector Analyst — evaluates how the stock's sector is performing relative to
the broader market and to peer sectors. Does NOT predict — just classifies.

Inputs (computed in Python, never from LLM):
  - sector tag (BANK / IT / PHARMA / METAL / ...)
  - sector_3m_return (median return of sector peers over last 60 bars)
  - sector_6m_return
  - sector_relative_strength (vs NIFTY 50)
  - sector_breadth (% of sector stocks above their own 200DMA)

LLM job: interpret these numbers + return a score/verdict on sector momentum.
"""
from __future__ import annotations

from .base_agent import BaseAgent, AgentReport


class SectorAnalyst(BaseAgent):
    name = "sector_analyst"
    role_description = "Sector rotation specialist"

    def _build_prompt(self, symbol: str, ctx: dict) -> str:
        sector = ctx.get("sector", "OTHER")
        return f"""You are a SECTOR ROTATION analyst at an Indian equity fund.
You don't predict prices. You just classify how the SECTOR is doing.

STOCK: {symbol}
SECTOR: {sector}

SECTOR METRICS (computed, not your guess):
  3-month sector median return:    {ctx.get('sector_3m_return', 0)*100:+.2f}%
  6-month sector median return:    {ctx.get('sector_6m_return', 0)*100:+.2f}%
  Sector relative strength (vs NIFTY): {ctx.get('sector_relative_strength', 0)*100:+.2f}%
  Sector breadth (above 200DMA):   {ctx.get('sector_breadth_pct', 0):.0f}%

REFERENCE THRESHOLDS:
  - "Hot" sector: 6m return > +10% AND RS vs NIFTY > +5% AND breadth > 60%
  - "Cold" sector: 6m return < -5% OR RS vs NIFTY < -5% OR breadth < 30%
  - Neutral otherwise

Return STRICTLY this JSON:
{{
  "score": <float -1.0 to +1.0>,
  "confidence": <float 0 to 1>,
  "regime": "HOT" | "COOL" | "NEUTRAL" | "COLD",
  "one_liner": "<under 20 words>"
}}

DO NOT hallucinate news. Base your answer only on the numbers above."""

    def analyze(self, symbol: str, context: dict) -> AgentReport:
        prompt = self._build_prompt(symbol, context)
        payload, provider, tokens, cost = self._call(prompt, max_tokens=200)
        if payload.get("error"):
            return AgentReport(
                agent_name=self.name, symbol=symbol,
                error=payload["error"], provider_used=provider,
            )
        score, conf = self._parse_score(payload)
        return AgentReport(
            agent_name=self.name, symbol=symbol,
            score=score, confidence=conf,
            verdict=self._score_to_verdict(score),
            flags=[payload.get("regime", "NEUTRAL")],
            one_liner=str(payload.get("one_liner", "")),
            raw=payload, provider_used=provider,
            tokens_used=tokens, cost_usd=cost,
        )
