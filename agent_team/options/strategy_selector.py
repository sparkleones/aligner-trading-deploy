"""
Strategy Selector — given the macro/technical/vol regime, pick the
right option strategy. Designed around the canonical 8-strategy Indian
F&O playbook:

  BUY_CE              - directional long, low IV, trending up
  BUY_PE              - directional short, low IV, trending down
  BULL_CALL_SPREAD    - mild bullish, mid IV (cap cost & risk)
  BEAR_PUT_SPREAD     - mild bearish, mid IV
  SHORT_STRADDLE      - range-bound, very high IV (premium-sell)
  IRON_CONDOR         - range-bound, high IV (premium-sell w/ wings)
  LONG_STRADDLE       - expect big move, low IV
  LONG_STRANGLE       - expect big move, very low IV
  HOLD                - no clear edge
"""
from __future__ import annotations

from agent_team.base_agent import BaseAgent, AgentReport


CANONICAL_STRATEGIES = [
    "BUY_CE", "BUY_PE",
    "BULL_CALL_SPREAD", "BEAR_PUT_SPREAD",
    "SHORT_STRADDLE", "IRON_CONDOR",
    "LONG_STRADDLE", "LONG_STRANGLE",
    "HOLD",
]


class StrategySelector(BaseAgent):
    name = "strategy_selector"
    role_description = "Option strategy chooser"

    def _build_prompt(self, ctx: dict) -> str:
        m = ctx.get("macro", {}) or {}
        v = ctx.get("vol", {}) or {}
        t = ctx.get("technical", {}) or {}
        return f"""You are a SENIOR F&O STRATEGIST. Given the regime,
pick ONE option strategy from the canonical list. Justify in one line.
Do NOT recommend specific strikes — that's a downstream step.

MACRO:
  NIFTY regime:        {m.get('regime', '?')}
  Direction bias:      {m.get('bias', 'neutral')}
  Days to expiry:      {ctx.get('days_to_expiry', '?')}

TECHNICAL:
  NIFTY trend:         {t.get('trend', '?')}
  Range vs trending:   {t.get('range_or_trend', '?')}
  RSI:                 {t.get('rsi_14', 0):.1f}

VOLATILITY:
  VIX regime:          {v.get('regime', '?')}
  IV rank:             {v.get('iv_rank', 0):.0f}
  IV - RV spread:      {v.get('iv_rv_spread', 0):.2%}

CANONICAL STRATEGY MATRIX:
  Trending up + low IV    -> BUY_CE
  Trending down + low IV  -> BUY_PE
  Mildly up + mid IV      -> BULL_CALL_SPREAD
  Mildly down + mid IV    -> BEAR_PUT_SPREAD
  Range-bound + high IV   -> IRON_CONDOR (capped wings)
  Range-bound + VERY high IV -> SHORT_STRADDLE (risk-aware)
  Pre-event + low IV      -> LONG_STRADDLE or LONG_STRANGLE
  Unclear                 -> HOLD

Return STRICTLY JSON:
{{
  "score": <-1.0 to +1.0  (sign reflects directional bias, -1 = strong short, +1 = strong long, 0 = market-neutral)>,
  "confidence": <0..1>,
  "strategy": "<one of: {' | '.join(CANONICAL_STRATEGIES)}>",
  "directional_bias": "LONG" | "SHORT" | "NEUTRAL",
  "expiry_preference": "WEEKLY" | "MONTHLY",
  "one_liner": "<under 30 words: why this strategy>"
}}"""

    def analyze(self, symbol: str, context: dict) -> AgentReport:
        prompt = self._build_prompt(context)
        payload, provider, tokens, cost = self._call(prompt, max_tokens=400)
        if payload.get("error"):
            return AgentReport(agent_name=self.name, symbol=symbol,
                                error=payload["error"], provider_used=provider)
        score, conf = self._parse_score(payload)
        strat = str(payload.get("strategy", "HOLD")).upper()
        if strat not in CANONICAL_STRATEGIES:
            strat = "HOLD"
        return AgentReport(
            agent_name=self.name, symbol=symbol,
            score=score, confidence=conf,
            verdict=self._score_to_verdict(score),
            flags=[f"STRAT:{strat}",
                    f"BIAS:{payload.get('directional_bias', 'NEUTRAL')}",
                    f"EXP:{payload.get('expiry_preference', 'WEEKLY')}"],
            one_liner=str(payload.get("one_liner", "")),
            raw=payload, provider_used=provider,
            tokens_used=tokens, cost_usd=cost,
        )
