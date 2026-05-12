"""
Multi-Agent Research Team for stock screening + option analysis.

Modeled after a small financial research firm. Each agent has a
narrow specialty and writes a structured report. A coordinator
synthesizes all reports into a final decision.

DESIGN PRINCIPLES
-----------------
1. LLMs are good at: synthesis, pattern matching, structured reasoning,
   reading fundamentals/text.
2. LLMs are bad at: predicting prices, fetching real-time news, doing
   math accurately, having a current world model.
3. So every agent CONSUMES structured data fetched from Python (yfinance,
   live prices, computed indicators) and PRODUCES structured JSON
   verdicts. No agent generates raw prices or makes math up.

THE TEAM
--------
  Sector Analyst       - sector rotation, relative strength rank
  Fundamental Analyst  - PE/PB/ROE/D-E/margins/growth review
  Technical Analyst    - chart pattern + indicator confirmation
  Risk Analyst         - position sizing, drawdown risk, correlation
  Macro Analyst        - NIFTY/VIX/breadth regime read
  Event Analyst        - upcoming earnings, ex-div, corporate actions
  Portfolio Manager    - synthesizes all 6 specialist reports + decides

COST MANAGEMENT
---------------
  Default provider chain: Groq (free 1000 RPD) -> Gemini Flash (free
  1500 RPD) -> Haiku 4.5 ($1/M, fallback only).
  The Portfolio Manager (coordinator) uses Sonnet 4.5 ($3/M) sparingly,
  only when 3+ specialists disagree and a tie-break is needed.

  Cost per stock (full team review) ~= 6 calls x ~500 tokens each
  = 3000 tokens. On Groq/Gemini free tiers: Rs 0. On Haiku fallback:
  ~Rs 0.25 per stock.
"""

from .base_agent import BaseAgent, AgentReport, TeamVerdict
from .sector_analyst import SectorAnalyst
from .fundamental_analyst import FundamentalAnalyst
from .technical_analyst import TechnicalAnalyst
from .risk_analyst import RiskAnalyst
from .macro_analyst import MacroAnalyst
from .event_analyst import EventAnalyst
from .portfolio_manager import PortfolioManager
from .coordinator import ResearchTeam

__all__ = [
    "BaseAgent", "AgentReport", "TeamVerdict",
    "SectorAnalyst", "FundamentalAnalyst", "TechnicalAnalyst",
    "RiskAnalyst", "MacroAnalyst", "EventAnalyst",
    "PortfolioManager", "ResearchTeam",
]
