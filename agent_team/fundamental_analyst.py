"""
Fundamental Analyst — evaluates company financial health from yfinance .info
snapshot. NO price prediction; just classifies quality.

Inputs (from screener/fundamentals.py):
  trailingPE, priceToBook, returnOnEquity, returnOnAssets, debtToEquity,
  profitMargins, operatingMargins, grossMargins, currentRatio,
  earningsGrowth, revenueGrowth, marketCap.
"""
from __future__ import annotations

from .base_agent import BaseAgent, AgentReport


class FundamentalAnalyst(BaseAgent):
    name = "fundamental_analyst"
    role_description = "Quality + valuation specialist"

    def _build_prompt(self, symbol: str, ctx: dict) -> str:
        f = ctx.get("fundamentals", {}) or {}
        def _g(k, fmt="{:.3f}", default="n/a"):
            v = f.get(k)
            if v is None:
                return default
            try:
                return fmt.format(float(v))
            except Exception:
                return str(v)

        return f"""You are a FUNDAMENTAL ANALYST. Score the financial health
of this Indian listed company. Do NOT predict prices. Score quality.

STOCK: {symbol}
SECTOR: {ctx.get('sector', 'OTHER')}

SNAPSHOT (point-in-time yfinance .info; values may be stale by 1 quarter):
  P/E (trailing):        {_g('trailingPE', '{:.1f}')}
  P/B:                   {_g('priceToBook', '{:.2f}')}
  ROE:                   {_g('returnOnEquity', '{:.2%}')}
  ROA:                   {_g('returnOnAssets', '{:.2%}')}
  Debt/Equity:           {_g('debtToEquity', '{:.0f}')}
  Profit margin:         {_g('profitMargins', '{:.2%}')}
  Operating margin:      {_g('operatingMargins', '{:.2%}')}
  Gross margin:          {_g('grossMargins', '{:.2%}')}
  Current ratio:         {_g('currentRatio', '{:.2f}')}
  Earnings growth YoY:   {_g('earningsGrowth', '{:.2%}')}
  Revenue growth YoY:    {_g('revenueGrowth', '{:.2%}')}
  Market cap:            {_g('marketCap', '{:.0f}')}

QUALITY RUBRIC:
  STRONG: ROE > 18%, D/E < 100, margins +YoY, growth > 10%, P/E reasonable
  WEAK:   ROE < 8%, D/E > 200, declining margins, growth < 0
  MIXED:  in between

Return STRICTLY this JSON:
{{
  "score": <-1.0 to +1.0>,
  "confidence": <0 to 1>,
  "quality_grade": "STRONG" | "MIXED" | "WEAK",
  "valuation": "CHEAP" | "FAIR" | "EXPENSIVE",
  "red_flags": [<list of: HIGH_DEBT, LOW_MARGINS, NEGATIVE_GROWTH, OVERVALUED, NONE>],
  "one_liner": "<under 20 words>"
}}

If most metrics are 'n/a', return score=0 and confidence=0.2."""

    def analyze(self, symbol: str, context: dict) -> AgentReport:
        prompt = self._build_prompt(symbol, context)
        payload, provider, tokens, cost = self._call(prompt, max_tokens=300)
        if payload.get("error"):
            return AgentReport(
                agent_name=self.name, symbol=symbol,
                error=payload["error"], provider_used=provider,
            )
        score, conf = self._parse_score(payload)
        flags = []
        for f in payload.get("red_flags", []):
            if f and f != "NONE":
                flags.append(f)
        return AgentReport(
            agent_name=self.name, symbol=symbol,
            score=score, confidence=conf,
            verdict=self._score_to_verdict(score),
            flags=flags + [
                f"GRADE:{payload.get('quality_grade', 'MIXED')}",
                f"VAL:{payload.get('valuation', 'FAIR')}",
            ],
            one_liner=str(payload.get("one_liner", "")),
            raw=payload, provider_used=provider,
            tokens_used=tokens, cost_usd=cost,
        )
