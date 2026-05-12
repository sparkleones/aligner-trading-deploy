"""
Options Portfolio Manager — synthesises the options team's reports
and outputs a concrete trade recommendation.
"""
from __future__ import annotations

from typing import Optional

from agent_team.base_agent import BaseAgent, AgentReport, TeamVerdict


OPTIONS_WEIGHTS = {
    "macro_analyst":         0.20,
    "technical_analyst":     0.15,
    "volatility_analyst":    0.25,
    "greeks_analyst":        0.15,
    "options_event_analyst": 0.10,
    "strategy_selector":     0.15,
}


class OptionsPortfolioManager:
    name = "options_portfolio_manager"

    def __init__(self, weights: Optional[dict] = None,
                 enable_llm_arbitration: bool = False,
                 brain: Optional[object] = None):
        self.weights = weights or OPTIONS_WEIGHTS
        self.enable_llm = enable_llm_arbitration
        self.brain = brain

    def synthesize(self, symbol: str, reports: list[AgentReport]) -> TeamVerdict:
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

        # Pull strategy_selector's strategy
        strat_report = next((r for r in reports if r.agent_name == "strategy_selector"), None)
        chosen_strategy = "HOLD"
        if strat_report:
            for f in strat_report.flags:
                if f.startswith("STRAT:"):
                    chosen_strategy = f.split(":", 1)[1]
                    break

        # ── Hard gating rules ──
        gating = []
        suggested_mult = 1.0
        # Event analyst: EXPIRY_TODAY + chosen long premium -> override to HOLD
        event_report = next((r for r in reports if r.agent_name == "options_event_analyst"), None)
        if event_report and any("EXPIRY_TODAY" in f for f in event_report.flags):
            if chosen_strategy in ("BUY_CE", "BUY_PE", "LONG_STRADDLE", "LONG_STRANGLE"):
                chosen_strategy = "HOLD"
                suggested_mult = 0.0
                gating.append("Expiry today -> no new long premium")
        # Greeks: OVER exposure -> halve
        greeks_report = next((r for r in reports if r.agent_name == "greeks_analyst"), None)
        if greeks_report and any(f == "EXP:OVER" for f in greeks_report.flags):
            suggested_mult *= 0.5
            gating.append("Greeks OVER -> 0.5x")
        if greeks_report and any(f == "EXP:STRETCHED" for f in greeks_report.flags):
            suggested_mult *= 0.7
            gating.append("Greeks STRETCHED -> 0.7x")
        # Vol regime override
        vol_report = next((r for r in reports if r.agent_name == "volatility_analyst"), None)
        if vol_report and any(f == "REG:EXTREME" for f in vol_report.flags):
            # Don't short straddle in extreme vol unless explicitly chosen
            if chosen_strategy == "SHORT_STRADDLE":
                chosen_strategy = "IRON_CONDOR"
                gating.append("Vol EXTREME -> SHORT_STRADDLE -> IRON_CONDOR")

        # Final action
        if chosen_strategy == "HOLD" or suggested_mult <= 0:
            action = "HOLD"
        else:
            action = "DEPLOY"

        # Hold days based on expiry preference + strategy
        hold_days = 7  # weekly default
        if strat_report:
            for f in strat_report.flags:
                if f == "EXP:MONTHLY":
                    hold_days = 28

        verdict = TeamVerdict(
            symbol=symbol,
            final_action=action,
            final_score=final_score,
            confidence=final_conf,
            suggested_qty_mult=max(0.0, min(1.5, suggested_mult)),
            hold_days=hold_days,
            reports=reports,
            coordinator_note=f"Strategy: {chosen_strategy}"
                             + (f" | gating: {'; '.join(gating)}" if gating else ""),
            total_cost_usd=cost_sum,
        )
        # Stash the chosen strategy in the verdict for the dashboard
        verdict.chosen_strategy = chosen_strategy
        return verdict
