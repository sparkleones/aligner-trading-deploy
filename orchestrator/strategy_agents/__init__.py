from orchestrator.strategy_agents.base_agent import BaseLiveAgent
from orchestrator.strategy_agents.iron_condor_agent import IronCondorLiveAgent
from orchestrator.strategy_agents.bull_put_spread_agent import BullPutSpreadLiveAgent
from orchestrator.strategy_agents.ddqn_live_agent import DDQNLiveAgent
from orchestrator.strategy_agents.learned_rules_agent import LearnedRulesLiveAgent
from orchestrator.strategy_agents.v3_multi_trade_agent import V3MultiTradeLiveAgent
from orchestrator.strategy_agents.v14_live_agent import V14LiveAgent

AGENT_REGISTRY: dict[str, type[BaseLiveAgent]] = {
    "iron_condor": IronCondorLiveAgent,
    "bull_put_spread": BullPutSpreadLiveAgent,
    "ddqn_agent": DDQNLiveAgent,
    "learned_rules": LearnedRulesLiveAgent,
    "learned_rules_v3": V3MultiTradeLiveAgent,
    "v14_production": V14LiveAgent,
}
