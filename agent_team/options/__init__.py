"""
Options Research Team — parallel to the equity team but specialised
for NIFTY / BANKNIFTY F&O.

THE OPTIONS TEAM
----------------
  Volatility Analyst   - VIX level, IV percentile, term structure
  Greeks/Risk Analyst  - position-level delta/theta/vega exposure
  Strategy Selector    - chooses STRADDLE / STRANGLE / IRON CONDOR /
                         BULL CALL SPREAD / BEAR PUT SPREAD / BUY CE /
                         BUY PE / HOLD based on regime
  Event Analyst        - expiry, RBI policy, F&O ban list
  Portfolio Manager    - synthesises + sizes lots

Reuses Macro + Technical analysts from the equity team.
"""
from .volatility_analyst import VolatilityAnalyst
from .greeks_analyst import GreeksAnalyst
from .strategy_selector import StrategySelector
from .options_event_analyst import OptionsEventAnalyst
from .options_pm import OptionsPortfolioManager
from .options_coordinator import OptionsResearchTeam

__all__ = [
    "VolatilityAnalyst", "GreeksAnalyst", "StrategySelector",
    "OptionsEventAnalyst", "OptionsPortfolioManager", "OptionsResearchTeam",
]
