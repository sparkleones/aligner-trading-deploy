"""
Technical Analyst — interprets PRE-COMPUTED technical signals.
Never asks the LLM to do math — Python computes everything, LLM only
classifies and writes a one-liner.
"""
from __future__ import annotations

from .base_agent import BaseAgent, AgentReport


class TechnicalAnalyst(BaseAgent):
    name = "technical_analyst"
    role_description = "Chart pattern + indicator confirmation specialist"

    def _build_prompt(self, symbol: str, ctx: dict) -> str:
        t = ctx.get("technical", {}) or {}
        return f"""You are a TECHNICAL ANALYST. Interpret these indicators
that Python has already computed. Do NOT predict prices.

STOCK: {symbol}

PRICE POSITION:
  Last close:               Rs {t.get('close', 0):.1f}
  Distance from 52w high:   {t.get('dist_from_52w_high_pct', 0)*100:+.2f}%
  Above 200-DMA:            {t.get('above_200dma', False)}
  Above 50-DMA:             {t.get('above_50dma', False)}
  50-DMA slope (1m):        {t.get('ma_50_slope_pct', 0)*100:+.2f}%

MOMENTUM / OSCILLATORS:
  12-1 month return:        {t.get('momentum_12_1', 0)*100:+.2f}%
  RSI(14):                  {t.get('rsi_14', 0):.1f}
  MACD histogram:           {t.get('macd_hist', 0):+.4f}

VOLATILITY:
  ATR(20) as % of price:    {t.get('atr_pct', 0)*100:.2f}%
  20-day vol (annualized):  {t.get('vol_20d_pct', 0)*100:.1f}%

VOLUME:
  Volume today vs 20d avg:  {t.get('vol_ratio', 0):.2f}x

PATTERN RUBRIC:
  STAGE 2: above 200DMA, 50DMA rising, momentum > 0 -> bullish
  STAGE 4: below 200DMA, 50DMA falling -> bearish
  BREAKOUT: within 5% of 52w high + volume spike -> bullish
  EXHAUSTION: RSI > 75 + extended -> caution
  OVERSOLD: RSI < 25 in uptrend -> mean-rev opportunity

Return STRICTLY this JSON:
{{
  "score": <-1.0 to +1.0>,
  "confidence": <0 to 1>,
  "regime": "STAGE_2" | "STAGE_4" | "BASE" | "TOP",
  "entry_quality": "STRONG" | "OK" | "WEAK",
  "warnings": [<list of: OVERBOUGHT, OVERSOLD, EXHAUSTION, GAP_RISK, LOW_VOL_BREAKOUT, NONE>],
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
        flags = [w for w in payload.get("warnings", []) if w and w != "NONE"]
        return AgentReport(
            agent_name=self.name, symbol=symbol,
            score=score, confidence=conf,
            verdict=self._score_to_verdict(score),
            flags=flags + [
                f"REG:{payload.get('regime', 'BASE')}",
                f"ENT:{payload.get('entry_quality', 'OK')}",
            ],
            one_liner=str(payload.get("one_liner", "")),
            raw=payload, provider_used=provider,
            tokens_used=tokens, cost_usd=cost,
        )
