"""
Event Analyst — flags upcoming events: earnings, ex-dividend, splits.
Honest scope: we don't have a real-time earnings calendar API, so we
flag based on:
  - fundamentals 'mostRecentQuarter' field
  - typical quarterly cadence (90-day cycle inference)
  - dividend yield + ex-div date if yfinance reports it
"""
from __future__ import annotations

from datetime import datetime
from .base_agent import BaseAgent, AgentReport


class EventAnalyst(BaseAgent):
    name = "event_analyst"
    role_description = "Earnings + corporate events specialist"

    def _build_prompt(self, symbol: str, ctx: dict) -> str:
        ev = ctx.get("events", {}) or {}
        return f"""You are an EVENT ANALYST. Flag risk from upcoming
corporate events. NO speculation about news that isn't in the data.

STOCK: {symbol}

EVENT METRICS:
  Last quarter end:        {ev.get('last_quarter_end', 'unknown')}
  Days since last result:  {ev.get('days_since_last_result', 'unknown')}
  Next earnings (estimated): {ev.get('next_earnings_estimated', 'unknown')}
  Days to next earnings:   {ev.get('days_to_next_earnings', 'unknown')}
  Dividend yield:          {ev.get('dividend_yield', 0)*100:.2f}%
  Recent dividend ex-date: {ev.get('ex_dividend_date', 'unknown')}

REFERENCE THRESHOLDS:
  HIGH RISK: earnings within 7 days (volatility spike risk)
  MED RISK:  earnings within 30 days
  LOW RISK:  earnings 30+ days out
  Special: ex-dividend in next 7 days adds minor risk
  Special: small/midcap with no earnings data -> moderate caution

Return STRICTLY this JSON:
{{
  "score": <-1.0 to +1.0   (- = event risk, + = no events near>,
  "confidence": <0 to 1>,
  "event_window": "WIDE" | "NORMAL" | "TIGHT",
  "flags": [<list of: EARNINGS_NEAR, EX_DIV_NEAR, NO_EVENT_DATA, NONE>],
  "one_liner": "<under 20 words>"
}}"""

    def analyze(self, symbol: str, context: dict) -> AgentReport:
        prompt = self._build_prompt(symbol, context)
        payload, provider, tokens, cost = self._call(prompt, max_tokens=250)
        if payload.get("error"):
            return AgentReport(
                agent_name=self.name, symbol=symbol,
                error=payload["error"], provider_used=provider,
            )
        score, conf = self._parse_score(payload)
        flags = [f for f in payload.get("flags", []) if f and f != "NONE"]
        return AgentReport(
            agent_name=self.name, symbol=symbol,
            score=score, confidence=conf,
            verdict=self._score_to_verdict(score),
            flags=flags + [f"WIN:{payload.get('event_window', 'NORMAL')}"],
            one_liner=str(payload.get("one_liner", "")),
            raw=payload, provider_used=provider,
            tokens_used=tokens, cost_usd=cost,
        )
