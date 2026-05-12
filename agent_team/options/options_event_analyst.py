"""
Options Event Analyst — F&O-specific event flags.

NSE F&O calendar items that materially affect options pricing:
  - Weekly expiry day (now Tuesday post-Nov 2025 SEBI consolidation)
  - Monthly expiry (last Thursday of month)
  - RBI policy day (1st Friday of bi-monthly cycle)
  - Major earnings (auto-flagged by Q1/Q2/Q3/Q4 cadence)
  - Budget day
  - F&O ban list (security-wise position limit breached)
"""
from __future__ import annotations

from datetime import datetime, timedelta
from agent_team.base_agent import BaseAgent, AgentReport


class OptionsEventAnalyst(BaseAgent):
    name = "options_event_analyst"
    role_description = "F&O calendar event specialist"

    def _build_prompt(self, ctx: dict) -> str:
        e = ctx.get("events", {}) or {}
        return f"""You are an F&O EVENT ANALYST for NSE NIFTY options.

Flag known calendar risks. Do NOT speculate on news. Stick to the
events listed below.

CURRENT STATE:
  Days to next weekly expiry:   {e.get('days_to_weekly_expiry', '?')}
  Days to next monthly expiry:  {e.get('days_to_monthly_expiry', '?')}
  Is expiry day today:          {e.get('is_expiry_today', False)}
  RBI policy in next 7 days:    {e.get('rbi_policy_within_7d', False)}
  Earnings season active:       {e.get('earnings_season', False)}
  Budget day proximity:         {e.get('days_to_budget', '?')}

REFERENCE RUBRIC:
  EXPIRY DAY:     theta accelerates 5-10x; AVOID new long premium
                  positions after 1 PM, FAVOR short straddle/condor
                  if VIX > 18.
  EXPIRY -1:      gamma scalping window for premium sellers
  PRE-RBI (1-3 days):  IV elevated, premium sellers favored if VIX > 17
                       but RISK if rate decision surprises
  EARNINGS SEASON: SECTORAL options have wider IV; index relatively
                   stable
  BUDGET DAY:     volatility spike, AVOID short premium 2 days before

Return STRICTLY JSON:
{{
  "score": <-1.0 to +1.0  (- = high event risk, + = clean window)>,
  "confidence": <0..1>,
  "event_window": "CLEAN" | "MILD" | "ELEVATED" | "EXTREME",
  "flags": [<list of: EXPIRY_TODAY, EXPIRY_TOMORROW, PRE_RBI, BUDGET_NEAR, EARNINGS_HEAVY, NONE>],
  "preferred_action": "BUILD_POSITION" | "TRIM_POSITION" | "AVOID" | "EXPIRY_PLAY",
  "one_liner": "<under 25 words>"
}}"""

    def analyze(self, symbol: str, context: dict) -> AgentReport:
        prompt = self._build_prompt(context)
        payload, provider, tokens, cost = self._call(prompt, max_tokens=300)
        if payload.get("error"):
            return AgentReport(agent_name=self.name, symbol=symbol,
                                error=payload["error"], provider_used=provider)
        score, conf = self._parse_score(payload)
        flags = [f for f in payload.get("flags", []) if f and f != "NONE"]
        return AgentReport(
            agent_name=self.name, symbol=symbol,
            score=score, confidence=conf,
            verdict=self._score_to_verdict(score),
            flags=flags + [
                f"WIN:{payload.get('event_window', 'CLEAN')}",
                f"ACT:{payload.get('preferred_action', 'BUILD_POSITION')}",
            ],
            one_liner=str(payload.get("one_liner", "")),
            raw=payload, provider_used=provider,
            tokens_used=tokens, cost_usd=cost,
        )
