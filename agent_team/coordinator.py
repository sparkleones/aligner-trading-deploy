"""
ResearchTeam — the public entry point. Build the team, give it a symbol,
get back a TeamVerdict.
"""
from __future__ import annotations

from typing import Optional

from .base_agent import TeamVerdict
from .sector_analyst import SectorAnalyst
from .fundamental_analyst import FundamentalAnalyst
from .technical_analyst import TechnicalAnalyst
from .risk_analyst import RiskAnalyst
from .macro_analyst import MacroAnalyst
from .event_analyst import EventAnalyst
from .portfolio_manager import PortfolioManager


class ResearchTeam:
    def __init__(self,
                 brain=None,
                 prefer_fast: bool = True,
                 enable_llm_arbitration: bool = False):
        """
        prefer_fast: each specialist uses the fast (free-tier) model
                     by default (groq -> gemini -> haiku fallback).
        enable_llm_arbitration: PM only calls Sonnet if specialists
                                strongly disagree. Defaults OFF for cost.
        """
        self.sector = SectorAnalyst(brain=brain, prefer_fast=prefer_fast)
        self.fundamental = FundamentalAnalyst(brain=brain, prefer_fast=prefer_fast)
        self.technical = TechnicalAnalyst(brain=brain, prefer_fast=prefer_fast)
        self.risk = RiskAnalyst(brain=brain, prefer_fast=prefer_fast)
        self.macro = MacroAnalyst(brain=brain, prefer_fast=prefer_fast)
        self.event = EventAnalyst(brain=brain, prefer_fast=prefer_fast)
        self.pm = PortfolioManager(
            brain=brain,
            enable_llm_arbitration=enable_llm_arbitration,
        )
        self.brain = brain

    def review(self, symbol: str, context: dict) -> TeamVerdict:
        """Run all 6 specialists in series; PM synthesizes."""
        reports = []
        reports.append(self.sector.analyze(symbol, context))
        reports.append(self.fundamental.analyze(symbol, context))
        reports.append(self.technical.analyze(symbol, context))
        reports.append(self.risk.analyze(symbol, context))
        reports.append(self.macro.analyze(symbol, context))
        reports.append(self.event.analyze(symbol, context))
        return self.pm.synthesize(symbol, reports)
