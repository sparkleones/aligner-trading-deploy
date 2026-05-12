"""
Volatility Analyst — classifies the volatility regime for NIFTY F&O.

Inputs (all computed in Python):
  - India VIX current + 30d avg + 252d avg
  - ATM IV (from option chain mid)
  - IV rank (current IV vs 252d range)
  - VIX term structure (front vs back month, if available)
  - Realized 20-day NIFTY volatility
  - IV - RV spread

The LLM just classifies regime and recommends premium-buy / premium-sell.
"""
from __future__ import annotations

from agent_team.base_agent import BaseAgent, AgentReport


class VolatilityAnalyst(BaseAgent):
    name = "volatility_analyst"
    role_description = "Vol regime + premium pricing specialist"

    def _build_prompt(self, ctx: dict) -> str:
        v = ctx.get("vol", {}) or {}
        return f"""You are a VOLATILITY ANALYST for NIFTY index options on NSE.

Classify the vol regime. Do NOT predict prices or quote option premiums.
Just rate whether option PREMIUMS are CHEAP or EXPENSIVE right now.

INPUTS (computed):
  India VIX current:          {v.get('vix_current', 0):.2f}
  India VIX 30-day average:   {v.get('vix_avg_30d', 0):.2f}
  India VIX 252-day average:  {v.get('vix_avg_252d', 0):.2f}
  VIX percentile (252d):      {v.get('vix_percentile', 0):.0f}
  ATM IV (front month):       {v.get('atm_iv', 0):.1%}
  IV rank (252d range):       {v.get('iv_rank', 0):.0f}
  Realized vol 20d (NIFTY):   {v.get('rv_20d', 0):.1%}
  IV - RV spread:             {v.get('iv_rv_spread', 0):.1%}
  Term structure (back/front): {v.get('term_ratio', 1.0):.2f}

THE RUBRIC:
  CHEAP premium (BUY options):
    - VIX percentile < 25 OR IV rank < 20
    - IV - RV spread negative (vols underprice realized moves)
    - Term structure in contango (back > front by > 3%)
  EXPENSIVE premium (SELL options):
    - VIX percentile > 75 OR IV rank > 80
    - IV - RV spread > +3%
    - Backwardation (front > back, > 5%)
  FAIR: in between

Return STRICTLY JSON:
{{
  "score": <-1.0 to +1.0 (- = SELL premium, + = BUY premium)>,
  "confidence": <0..1>,
  "regime": "CHEAP" | "FAIR" | "EXPENSIVE" | "EXTREME",
  "iv_rank_bucket": "LOW" | "MID" | "HIGH",
  "one_liner": "<under 25 words>"
}}"""

    def analyze(self, symbol: str, context: dict) -> AgentReport:
        prompt = self._build_prompt(context)
        payload, provider, tokens, cost = self._call(prompt, max_tokens=250)
        if payload.get("error"):
            return AgentReport(agent_name=self.name, symbol=symbol,
                                error=payload["error"], provider_used=provider)
        score, conf = self._parse_score(payload)
        return AgentReport(
            agent_name=self.name, symbol=symbol,
            score=score, confidence=conf,
            verdict=self._score_to_verdict(score),
            flags=[f"REG:{payload.get('regime', 'FAIR')}",
                    f"IV:{payload.get('iv_rank_bucket', 'MID')}"],
            one_liner=str(payload.get("one_liner", "")),
            raw=payload, provider_used=provider,
            tokens_used=tokens, cost_usd=cost,
        )
