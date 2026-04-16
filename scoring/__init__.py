"""Shared scoring engine — ONE source of truth for backtest + live.

V15: Scoring-only improvements over V14 R5 (16.04x vs 15.87x).
  - OBV trend scoring, momentum acceleration, EMA stack alignment
  - Session-specific weights, RV/IV entry quality filter
  - All exit changes DISABLED (exhaustion, stale, chandelier all cut winners)
"""

from scoring.config import V15_CONFIG
from scoring.indicators import compute_indicators
from scoring.engine import score_entry, passes_confluence, evaluate_exit, compute_lots

__all__ = [
    "V15_CONFIG",
    "compute_indicators",
    "score_entry",
    "passes_confluence",
    "evaluate_exit",
    "compute_lots",
]
