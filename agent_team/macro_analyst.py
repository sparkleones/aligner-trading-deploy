"""
Macro Analyst — overall market regime. NIFTY position, VIX, breadth.
Independent of individual stocks — same answer for every symbol on a given day.
"""
from __future__ import annotations

from .base_agent import BaseAgent, AgentReport


class MacroAnalyst(BaseAgent):
    name = "macro_analyst"
    role_description = "Top-down market regime specialist"

    def _build_prompt(self, ctx: dict) -> str:
        m = ctx.get("macro", {}) or {}
        return f"""You are a MACRO STRATEGIST. Classify the broad market
regime. This is independent of any individual stock.

NIFTY 50 SNAPSHOT:
  Last close:            {m.get('nifty_close', 0):,.0f}
  Dist from 52w high:    {m.get('nifty_dist_high_pct', 0)*100:+.2f}%
  Dist from 200-DMA:     {m.get('nifty_dist_200dma_pct', 0)*100:+.2f}%
  Above 200-DMA:         {m.get('above_200dma', False)}
  Golden cross active:   {m.get('golden_cross', False)}
  RSI(14) daily:         {m.get('rsi_14', 0):.1f}
  RSI(14) weekly:        {m.get('rsi_14_weekly', 0):.1f}

VOLATILITY:
  India VIX:             {m.get('vix', 0):.2f}
  VIX regime:            {m.get('vix_regime', '?')}
  VIX vs 252d avg:       {m.get('vix_vs_avg_pct', 0):+.1f}%

BREADTH:
  % stocks above 200DMA: {m.get('breadth_pct', 0):.0f}%

REGIME RUBRIC:
  RISK-ON:  above 200DMA + golden cross + breadth > 60% + VIX < 16
  CHOPPY:   sideways, breadth 40-60%, VIX 16-22
  RISK-OFF: below 200DMA + death cross + breadth < 40% + VIX > 22
  CRISIS:   VIX > 30 OR drawdown > 15%

Return STRICTLY this JSON:
{{
  "score": <-1.0 to +1.0 - represents how favorable for long stocks NOW>,
  "confidence": <0 to 1>,
  "regime": "RISK_ON" | "CHOPPY" | "RISK_OFF" | "CRISIS",
  "exposure_recommendation": "FULL" | "REDUCED" | "DEFENSIVE" | "CASH",
  "one_liner": "<under 25 words>"
}}"""

    def analyze(self, symbol: str, context: dict) -> AgentReport:
        prompt = self._build_prompt(context)
        payload, provider, tokens, cost = self._call(prompt, max_tokens=300)
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
            flags=[
                f"REG:{payload.get('regime', 'CHOPPY')}",
                f"EXP:{payload.get('exposure_recommendation', 'REDUCED')}",
            ],
            one_liner=str(payload.get("one_liner", "")),
            raw=payload, provider_used=provider,
            tokens_used=tokens, cost_usd=cost,
        )
