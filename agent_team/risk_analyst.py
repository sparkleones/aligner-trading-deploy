"""
Risk Analyst — looks at position sizing, drawdown history, correlation
to existing portfolio. Does NOT make math up — Python supplies all stats.
"""
from __future__ import annotations

from .base_agent import BaseAgent, AgentReport


class RiskAnalyst(BaseAgent):
    name = "risk_analyst"
    role_description = "Position sizing + drawdown specialist"

    def _build_prompt(self, symbol: str, ctx: dict) -> str:
        r = ctx.get("risk", {}) or {}
        return f"""You are a RISK ANALYST. Score whether this stock is
SAFE to add to a portfolio right now, given drawdown history and
correlation. Score is independent of "will it go up" - that's
someone else's job.

STOCK: {symbol}

POSITION RISK METRICS (computed):
  Max drawdown 12m:       {r.get('max_dd_12m_pct', 0)*100:+.2f}%
  Volatility 252d:        {r.get('vol_252d_pct', 0)*100:.1f}%
  Beta vs NIFTY:          {r.get('beta_vs_nifty', 0):.2f}
  Correlation to NIFTY:   {r.get('corr_nifty', 0):.2f}
  Drawdown right now:     {r.get('current_drawdown_pct', 0)*100:+.2f}%  (from 52w high)
  Frequency of 3% gaps:   {r.get('gap_3pct_freq', 0)*100:.1f}% of days

PORTFOLIO CONTEXT:
  Sector overweight after this add: {r.get('would_overweight', False)}
  Existing positions:               {ctx.get('current_holdings_count', 0)}

REFERENCE THRESHOLDS:
  LOW risk:    vol < 22%, beta 0.7-1.1, gap freq < 2%, no recent crash
  MED risk:    vol 22-35%, mid drawdown history
  HIGH risk:   vol > 35%, gap freq > 5%, recent 30%+ drawdown

Return STRICTLY this JSON:
{{
  "score": <-1.0 to +1.0   (+ = safe to add, - = too risky)>,
  "confidence": <0 to 1>,
  "risk_grade": "LOW" | "MED" | "HIGH",
  "size_recommendation": "FULL" | "HALF" | "QUARTER" | "SKIP",
  "concerns": [<list of: HIGH_VOL, GAP_RISK, SECTOR_CONCENTRATION, RECENT_CRASH, NONE>],
  "one_liner": "<under 20 words>"
}}"""

    def analyze(self, symbol: str, context: dict) -> AgentReport:
        prompt = self._build_prompt(symbol, context)
        payload, provider, tokens, cost = self._call(prompt, max_tokens=300)
        if payload.get("error"):
            return AgentReport(
                agent_name=self.name, symbol=symbol,
                error=payload["error"], provider_used=provider,
            )
        score, conf = self._parse_score(payload)
        concerns = [c for c in payload.get("concerns", []) if c and c != "NONE"]
        return AgentReport(
            agent_name=self.name, symbol=symbol,
            score=score, confidence=conf,
            verdict=self._score_to_verdict(score),
            flags=concerns + [
                f"RISK:{payload.get('risk_grade', 'MED')}",
                f"SIZE:{payload.get('size_recommendation', 'HALF')}",
            ],
            one_liner=str(payload.get("one_liner", "")),
            raw=payload, provider_used=provider,
            tokens_used=tokens, cost_usd=cost,
        )
