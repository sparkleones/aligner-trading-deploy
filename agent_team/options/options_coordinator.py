"""
OptionsResearchTeam — entry point for options analysis.

Composes:
  - shared MacroAnalyst + TechnicalAnalyst (from equity team)
  - VolatilityAnalyst (new)
  - GreeksAnalyst (new)
  - OptionsEventAnalyst (new)
  - StrategySelector (new)
  - OptionsPortfolioManager (new)
"""
from __future__ import annotations

from typing import Optional

from agent_team.base_agent import TeamVerdict
from agent_team.macro_analyst import MacroAnalyst
from agent_team.technical_analyst import TechnicalAnalyst

from .volatility_analyst import VolatilityAnalyst
from .greeks_analyst import GreeksAnalyst
from .strategy_selector import StrategySelector
from .options_event_analyst import OptionsEventAnalyst
from .options_pm import OptionsPortfolioManager


class OptionsResearchTeam:
    def __init__(self, brain=None, prefer_fast: bool = True,
                 enable_llm_arbitration: bool = False):
        self.macro = MacroAnalyst(brain=brain, prefer_fast=prefer_fast)
        self.technical = TechnicalAnalyst(brain=brain, prefer_fast=prefer_fast)
        self.volatility = VolatilityAnalyst(brain=brain, prefer_fast=prefer_fast)
        self.greeks = GreeksAnalyst(brain=brain, prefer_fast=prefer_fast)
        self.event = OptionsEventAnalyst(brain=brain, prefer_fast=prefer_fast)
        self.strategy = StrategySelector(brain=brain, prefer_fast=prefer_fast)
        self.pm = OptionsPortfolioManager(
            enable_llm_arbitration=enable_llm_arbitration, brain=brain
        )
        self.brain = brain

    def review(self, symbol: str, context: dict) -> TeamVerdict:
        reports = []
        reports.append(self.macro.analyze(symbol, context))
        reports.append(self.technical.analyze(symbol, context))
        reports.append(self.volatility.analyze(symbol, context))
        reports.append(self.greeks.analyze(symbol, context))
        reports.append(self.event.analyze(symbol, context))
        reports.append(self.strategy.analyze(symbol, context))
        return self.pm.synthesize(symbol, reports)
