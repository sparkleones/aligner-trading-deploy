"""
Portfolio Manager — synthesizes all specialist reports into a final
TeamVerdict. Two synthesis modes:

  RULE-BASED (default):  Weighted average of specialist scores +
                         hard-coded gating rules. NO LLM CALL.
                         Cost: Rs 0. Latency: 1 ms.

  LLM-ARBITRATED:        When specialists disagree significantly,
                         call Sonnet 4.5 to tie-break with reasoning.
                         Cost: ~Rs 1 per call. Used rarely.

The default weights are designed to NOT overweight any single signal:
  Macro:        0.20  (top-down)
  Sector:       0.15  (rotation)
  Technical:    0.20  (momentum)
  Fundamental:  0.20  (quality)
  Risk:         0.15  (safety)
  Event:        0.10  (calendar risk)
"""
from __future__ import annotations

import os
from typing import Optional

from .base_agent import BaseAgent, AgentReport, TeamVerdict


DEFAULT_WEIGHTS = {
    "macro_analyst":       0.20,
    "sector_analyst":      0.15,
    "technical_analyst":   0.20,
    "fundamental_analyst": 0.20,
    "risk_analyst":        0.15,
    "event_analyst":       0.10,
}


class PortfolioManager:
    name = "portfolio_manager"

    def __init__(self,
                 weights: Optional[dict] = None,
                 llm_arbitrate_threshold: float = 0.6,
                 brain: Optional[object] = None,
                 enable_llm_arbitration: bool = False):
        """
        weights: per-agent weight for score synthesis
        llm_arbitrate_threshold: max stdev of agent scores before
            tie-breaking with LLM
        enable_llm_arbitration: if False (default), never call LLM —
            saves cost. Set True only for the most important decisions.
        """
        self.weights = weights or DEFAULT_WEIGHTS
        self.llm_threshold = llm_arbitrate_threshold
        self.enable_llm = enable_llm_arbitration
        self.brain = brain

    def synthesize(self, symbol: str, reports: list[AgentReport]) -> TeamVerdict:
        # Compute weighted score
        wsum = 0.0
        score_sum = 0.0
        conf_sum = 0.0
        cost_sum = 0.0
        for r in reports:
            w = self.weights.get(r.agent_name, 0.0)
            if w <= 0:
                continue
            wsum += w * r.confidence
            score_sum += w * r.confidence * r.score
            conf_sum += w * r.confidence
            cost_sum += r.cost_usd

        final_score = (score_sum / wsum) if wsum > 0 else 0.0
        final_score = max(-1.0, min(1.0, final_score))
        final_conf = (conf_sum / sum(self.weights.values())) if self.weights else 0.0
        final_conf = max(0.0, min(1.0, final_conf))

        # Compute agreement: stdev of agent scores (low stdev = consensus)
        scores = [r.score for r in reports if self.weights.get(r.agent_name, 0) > 0]
        if scores:
            mean = sum(scores) / len(scores)
            var = sum((s - mean) ** 2 for s in scores) / len(scores)
            stdev = var ** 0.5
        else:
            stdev = 0.0

        # ── Hard gating rules (override the score) ──
        gating_note = []
        suggested_mult = 1.0
        final_action = "HOLD"

        # 1. If macro says CRISIS or RISK_OFF score < -0.5, only buy with caution
        macro_report = next((r for r in reports if r.agent_name == "macro_analyst"), None)
        if macro_report and macro_report.score < -0.5:
            suggested_mult *= 0.5
            gating_note.append("Macro RISK_OFF -> 0.5x size")

        # 2. If risk analyst flags HIGH risk, halve
        risk_report = next((r for r in reports if r.agent_name == "risk_analyst"), None)
        if risk_report and "RISK:HIGH" in risk_report.flags:
            suggested_mult *= 0.5
            gating_note.append("Risk HIGH -> 0.5x")

        # 3. If event analyst flags EARNINGS_NEAR (< 7 days), 0.25x
        event_report = next((r for r in reports if r.agent_name == "event_analyst"), None)
        if event_report and "EARNINGS_NEAR" in event_report.flags:
            suggested_mult *= 0.25
            gating_note.append("Earnings near -> 0.25x")

        # 4. If fundamental_analyst grade = WEAK, halve
        fund_report = next((r for r in reports if r.agent_name == "fundamental_analyst"), None)
        if fund_report and "GRADE:WEAK" in fund_report.flags:
            suggested_mult *= 0.5
            gating_note.append("Fundamentals WEAK -> 0.5x")

        # ── Translate final_score + gating into action ──
        if final_score <= -0.3:
            final_action = "SKIP"
            suggested_mult = 0.0
        elif final_score >= 0.3 and suggested_mult >= 0.5:
            final_action = "BUY"
        else:
            final_action = "HOLD"
            if suggested_mult > 0:
                final_action = "BUY"  # small position still allowed
            else:
                suggested_mult = 0.25  # fallback

        # Hold days: shorten if event near or risk high, lengthen if quality high
        hold_days = 42
        if event_report and "EARNINGS_NEAR" in event_report.flags:
            hold_days = 14
        if fund_report and "GRADE:STRONG" in fund_report.flags:
            hold_days = 63

        # ── LLM arbitration when specialists strongly disagree ──
        coordinator_note = "Rule-based synthesis"
        if self.enable_llm and stdev > self.llm_threshold and self.brain:
            try:
                summary = "\n".join(
                    f"- {r.agent_name}: score={r.score:+.2f} conf={r.confidence:.2f} "
                    f"{r.verdict} | {r.one_liner}"
                    for r in reports
                )
                prompt = f"""You are the Chief PM. Six specialists gave conflicting reports on {symbol}.
Stdev of scores is {stdev:.2f}. Make a final call.

{summary}

Output JSON: {{"final_action": "BUY|HOLD|SKIP", "qty_mult": 0-1.5,
"hold_days": 14-90, "reasoning": "<short>"}}
"""
                payload = self.brain._call_llm(prompt, max_tokens=300, use_fast=False)
                if payload and not payload.get("error"):
                    final_action = str(payload.get("final_action", final_action)).upper()
                    if "qty_mult" in payload:
                        try:
                            suggested_mult = float(payload["qty_mult"])
                        except Exception:
                            pass
                    if "hold_days" in payload:
                        try:
                            hold_days = int(payload["hold_days"])
                        except Exception:
                            pass
                    coordinator_note = (
                        "LLM arbitrated (stdev "
                        f"{stdev:.2f}): {payload.get('reasoning', '')}"
                    )
            except Exception as e:
                coordinator_note = f"LLM arbitration failed: {e}"

        verdict = TeamVerdict(
            symbol=symbol,
            final_action=final_action,
            final_score=final_score,
            confidence=final_conf,
            suggested_qty_mult=max(0.0, min(1.5, suggested_mult)),
            hold_days=hold_days,
            reports=reports,
            coordinator_note=coordinator_note + (
                f" | gating: {'; '.join(gating_note)}" if gating_note else ""
            ),
            total_cost_usd=cost_sum,
        )
        return verdict
