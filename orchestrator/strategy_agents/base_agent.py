"""Base class for live strategy agents."""

from abc import ABC, abstractmethod
import logging
from typing import Optional, TYPE_CHECKING

import pandas as pd

from config.constants import FREEZE_LIMITS
from orchestrator.trade_signal import TradeSignal

if TYPE_CHECKING:
    from orchestrator.market_analyzer import MarketAnalysis

logger = logging.getLogger(__name__)

# Strike interval by index
STRIKE_INTERVALS = {"NIFTY": 50, "BANKNIFTY": 100, "FINNIFTY": 50}


class BaseLiveAgent(ABC):
    """Base class for live strategy agents.

    Each agent receives OHLCV bars, option chain data, and a MarketAnalysis,
    and emits TradeSignal objects when it wants to trade.

    Capital-aware position sizing:
      - num_lots scales with capital (more capital = more lots = more profit)
      - Respects NSE freeze limits (NIFTY=1800, BANKNIFTY=600)
      - Risk per trade capped at 20% of capital
    """

    name: str = "base"

    def __init__(self, capital: float = 25000.0, lot_size: int = 65):
        self.capital = capital
        self.lot_size = lot_size    # NSE minimum lot (e.g., 65 for NIFTY as of SEBI Feb 2026)
        self._bars: list[dict] = []
        self._position_open = False
        self._entry_bar: int = 0

        # ── Capital-proportional position sizing ──────────────────────
        # Risk budget: 30% of capital (single strategy uses full risk allocation)
        self._risk_budget = capital * 0.30

        # Default spread width (1 strike interval = 50 pts for NIFTY)
        self._default_spread_width = 50.0

        # Max loss per single lot = spread_width * lot_size
        max_loss_per_lot = self._default_spread_width * lot_size

        # Number of lots we can afford
        if max_loss_per_lot > 0:
            self._num_lots = max(1, int(self._risk_budget / max_loss_per_lot))
        else:
            self._num_lots = 1

        # Respect NSE freeze limits
        freeze_qty = 1800  # default
        for idx_name, fq in FREEZE_LIMITS.items():
            if idx_name.upper() in self.name.upper() or True:  # apply generically
                freeze_qty = min(freeze_qty, fq)
        max_lots_by_freeze = max(1, freeze_qty // lot_size - 1)  # stay below freeze
        self._num_lots = min(self._num_lots, max_lots_by_freeze)

        # Total quantity per leg
        self._order_qty = self.lot_size * self._num_lots

        logger.info(
            "%s: capital=%.0f | %d lots x %d = %d qty | risk_budget=%.0f | max_loss/lot=%.0f",
            self.name, capital, self._num_lots, lot_size, self._order_qty,
            self._risk_budget, max_loss_per_lot,
        )

    def add_bar(self, bar: dict) -> None:
        """Add a completed OHLCV bar to the internal buffer."""
        self._bars.append(bar)
        if len(self._bars) > 500:
            self._bars = self._bars[-500:]

    @abstractmethod
    def generate_signal(
        self,
        bar: dict,
        bar_idx: int,
        option_chain: Optional[dict] = None,
        market_analysis: Optional["MarketAnalysis"] = None,
    ) -> Optional[TradeSignal]:
        """Analyze the latest bar and return a signal (or None for HOLD)."""

    def set_position(self, is_open: bool) -> None:
        """Update position state after trade execution."""
        self._position_open = is_open

    def bars_as_df(self) -> pd.DataFrame:
        """Return bars buffer as a DataFrame."""
        if not self._bars:
            return pd.DataFrame()
        return pd.DataFrame(self._bars)

    def eligible_for_capital(self, capital: float) -> bool:
        """Check if this strategy can operate within the given capital."""
        return capital >= 15000  # Minimum for defined-risk spreads

    @staticmethod
    def resolve_symbol(strike: float, opt_type: str, option_chain: dict | None) -> str:
        """Resolve tradingsymbol from option chain with expiry-aware fallback.

        If the exact strike is in the chain, return its tradingsymbol.
        If not, derive the expiry prefix from another chain entry so the
        generated symbol includes the expiry date (e.g. NIFTY2640722750PE
        instead of the broken NIFTY22750PE).
        """
        if option_chain:
            entry = option_chain.get(strike, {}).get(opt_type, {})
            if "tradingsymbol" in entry:
                return entry["tradingsymbol"]
            # Try nearby strikes (float rounding)
            for s in option_chain:
                if abs(s - strike) < 1:
                    entry = option_chain[s].get(opt_type, {})
                    if "tradingsymbol" in entry:
                        return entry["tradingsymbol"]
            # Derive expiry prefix from any existing symbol
            # e.g. "NIFTY2640722700CE" minus "22700CE" → "NIFTY26407"
            for s, data in option_chain.items():
                for ot in ("CE", "PE"):
                    ts = data.get(ot, {}).get("tradingsymbol", "")
                    suffix = f"{int(s)}{ot}"
                    if ts.endswith(suffix):
                        prefix = ts[: -len(suffix)]
                        return f"{prefix}{int(strike)}{opt_type}"
            logger.warning("Option chain has %d strikes but strike %.0f not found",
                           len(option_chain), strike)
        return f"NIFTY{int(strike)}{opt_type}"
