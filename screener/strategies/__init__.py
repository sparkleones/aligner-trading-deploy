"""
Strategy library — 10 distinct stock screening strategies, each
implemented as a class with a common `score(history)` interface.

Strategies covered (paired with their academic / practitioner sources):

  TECHNICAL (OHLCV only — clean backtest)
  ---------
  1. Jegadeesh-Titman 12-1M Momentum   (momentum.py)
  2. Clenow Stocks on the Move          (stocks_on_move.py)
  3. Antonacci Dual Momentum            (dual_momentum.py)
  4. Weinstein Stage 2 Trend            (stage2_trend.py)
  5. Donchian/Darvas 52-Week Breakout   (breakout.py)
  6. Asness Low Volatility              (low_volatility.py)
  7. Connors RSI Mean Reversion         (mean_reversion.py)

  FUNDAMENTAL (yfinance .info — snapshot data, mild look-ahead caveat)
  -----------
  8. Greenblatt Magic Formula           (magic_formula.py)
  9. Piotroski F-Score                  (piotroski.py)
  10. Multi-Factor VMQ Composite        (multi_factor.py)

All strategies share the BaseStrategy contract:
    .name              -> str
    .needs_fundamentals -> bool
    .score(symbol, history_df, fundamentals=None) -> float | nan
"""

from .base import BaseStrategy
from .momentum import MomentumStrategy
from .stocks_on_move import StocksOnMoveStrategy
from .dual_momentum import DualMomentumStrategy
from .stage2_trend import Stage2TrendStrategy
from .breakout import BreakoutStrategy
from .low_volatility import LowVolatilityStrategy
from .mean_reversion import MeanReversionStrategy
from .magic_formula import MagicFormulaStrategy
from .piotroski import PiotroskiStrategy
from .multi_factor import MultiFactorVMQStrategy

STRATEGIES = {
    "momentum_12_1":    MomentumStrategy(),
    "stocks_on_move":   StocksOnMoveStrategy(),
    "dual_momentum":    DualMomentumStrategy(),
    "stage2_trend":     Stage2TrendStrategy(),
    "breakout_52w":     BreakoutStrategy(),
    "low_volatility":   LowVolatilityStrategy(),
    "mean_reversion":   MeanReversionStrategy(),
    "magic_formula":    MagicFormulaStrategy(),
    "piotroski":        PiotroskiStrategy(),
    "multi_factor_vmq": MultiFactorVMQStrategy(),
}

__all__ = ["BaseStrategy", "STRATEGIES"]
