"""
Base classes: BaseAgent, AgentReport, TeamVerdict.

Each agent inherits BaseAgent, implements `analyze()`, and returns
an AgentReport. The coordinator collects reports + emits TeamVerdict.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Optional

# Reuse the existing brain infrastructure for provider rotation
try:
    from orchestrator.claude_market_brain import ClaudeMarketBrain
    _BRAIN_OK = True
except Exception:
    _BRAIN_OK = False


@dataclass
class AgentReport:
    agent_name: str
    symbol: str
    score: float = 0.0          # -1.0 to +1.0  (-1 = strong sell, +1 = strong buy)
    confidence: float = 0.5     # 0..1, how sure
    verdict: str = "NEUTRAL"    # POSITIVE | NEUTRAL | NEGATIVE
    flags: list[str] = field(default_factory=list)
    one_liner: str = ""
    raw: dict = field(default_factory=dict)
    error: Optional[str] = None
    provider_used: str = ""
    tokens_used: int = 0
    cost_usd: float = 0.0


@dataclass
class TeamVerdict:
    symbol: str
    final_action: str = "HOLD"       # BUY | HOLD | SKIP
    final_score: float = 0.0          # weighted avg of agent scores
    confidence: float = 0.5
    suggested_qty_mult: float = 1.0   # 0.0 = SKIP, 0.5 = half, 1.0 = full, 1.5 = boost
    hold_days: int = 42
    reports: list[AgentReport] = field(default_factory=list)
    coordinator_note: str = ""
    total_cost_usd: float = 0.0

    def by_agent(self, name: str) -> Optional[AgentReport]:
        for r in self.reports:
            if r.agent_name == name:
                return r
        return None


class BaseAgent:
    """Common LLM-call plumbing + JSON parsing for all specialist agents."""

    name: str = "base"
    role_description: str = "Generic agent"

    def __init__(self, brain: Optional[ClaudeMarketBrain] = None,
                 prefer_fast: bool = True):
        if brain is None and _BRAIN_OK:
            try:
                brain = ClaudeMarketBrain()
            except Exception:
                brain = None
        self.brain = brain
        self.prefer_fast = prefer_fast

    def _call(self, prompt: str, max_tokens: int = 300) -> tuple[dict, str, int, float]:
        """Call the LLM. Returns (parsed_json, provider_name, tokens, cost_usd).

        ClaudeMarketBrain._call_llm parses JSON automatically and rotates
        through the provider chain (groq -> gemini -> deepseek -> haiku
        -> openai -> sonnet -> claude). We rely on that here for cost
        management — Groq/Gemini are free, only fall through to paid
        models if free quotas are exhausted.
        """
        if self.brain is None:
            return {}, "no_brain", 0, 0.0
        try:
            payload = self.brain._call_llm(prompt, max_tokens=max_tokens,
                                            use_fast=self.prefer_fast)
            provider = payload.get("provider", "unknown")
            return payload, provider, 0, 0.0
        except Exception as e:
            return {"error": str(e)}, "error", 0, 0.0

    def _parse_score(self, payload: dict, key: str = "score") -> tuple[float, float]:
        """Pull score [-1, 1] and confidence [0, 1] from LLM response."""
        try:
            s = float(payload.get(key, 0.0))
            s = max(-1.0, min(1.0, s))
        except Exception:
            s = 0.0
        try:
            c = float(payload.get("confidence", 0.5))
            c = max(0.0, min(1.0, c))
        except Exception:
            c = 0.5
        return s, c

    def _score_to_verdict(self, score: float) -> str:
        if score >= 0.3:
            return "POSITIVE"
        if score <= -0.3:
            return "NEGATIVE"
        return "NEUTRAL"

    # Each specialist agent implements this
    def analyze(self, symbol: str, context: dict) -> AgentReport:
        raise NotImplementedError
