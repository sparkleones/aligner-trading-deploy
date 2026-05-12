"""
Greeks Analyst — assesses portfolio-level Greek exposure and the
risk profile of proposed option positions.

Inputs:
  - net portfolio Delta, Theta, Vega, Gamma
  - capital deployed in F&O
  - capital available
  - max acceptable theta-decay-per-day (as % of capital)
  - max acceptable single-day vega P&L
"""
from __future__ import annotations

from agent_team.base_agent import BaseAgent, AgentReport


class GreeksAnalyst(BaseAgent):
    name = "greeks_analyst"
    role_description = "Greeks exposure + position-sizing specialist"

    def _build_prompt(self, ctx: dict) -> str:
        g = ctx.get("greeks", {}) or {}
        return f"""You are a F&O GREEKS RISK ANALYST. Classify whether
the current portfolio exposure is healthy and whether new positions
would breach prudent Greek limits.

PORTFOLIO STATE (computed):
  Net delta (lots):             {g.get('net_delta', 0):.2f}
  Net theta (Rs/day):           {g.get('net_theta_inr', 0):,.0f}
  Net vega (Rs per 1% IV):      {g.get('net_vega_inr', 0):,.0f}
  Net gamma:                    {g.get('net_gamma', 0):.4f}
  Capital deployed (F&O):       Rs {g.get('capital_deployed', 0):,.0f}
  Available capital:            Rs {g.get('capital_available', 0):,.0f}
  Today's expiry?               {g.get('is_expiry', False)}
  Day type:                     {g.get('day_type', 'normal')}

REFERENCE LIMITS (single account):
  Max safe net delta:    +- 4 lots
  Max safe theta drag:   1% of total capital per day
  Max safe vega P&L:     2% of total capital per 1% IV move
  No new shorts on expiry day morning

Return STRICTLY JSON:
{{
  "score": <-1.0 to +1.0  (+ = safe to add, - = at limits / unsafe)>,
  "confidence": <0..1>,
  "exposure_level": "LIGHT" | "BALANCED" | "STRETCHED" | "OVER",
  "delta_bias": "LONG" | "NEUTRAL" | "SHORT",
  "primary_risk": "DELTA" | "THETA" | "VEGA" | "GAMMA" | "NONE",
  "one_liner": "<under 25 words>"
}}"""

    def analyze(self, symbol: str, context: dict) -> AgentReport:
        prompt = self._build_prompt(context)
        payload, provider, tokens, cost = self._call(prompt, max_tokens=300)
        if payload.get("error"):
            return AgentReport(agent_name=self.name, symbol=symbol,
                                error=payload["error"], provider_used=provider)
        score, conf = self._parse_score(payload)
        return AgentReport(
            agent_name=self.name, symbol=symbol,
            score=score, confidence=conf,
            verdict=self._score_to_verdict(score),
            flags=[f"EXP:{payload.get('exposure_level', 'BALANCED')}",
                    f"DELTA:{payload.get('delta_bias', 'NEUTRAL')}",
                    f"RISK:{payload.get('primary_risk', 'NONE')}"],
            one_liner=str(payload.get("one_liner", "")),
            raw=payload, provider_used=provider,
            tokens_used=tokens, cost_usd=cost,
        )
