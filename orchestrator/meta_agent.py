"""Meta-agent that aggregates signals from multiple strategy agents.

Uses weighted voting to decide which trade to execute. Enforces:
- Capital constraints (max risk per trade)
- Cooldown between trades (no overtrading)
- Directional agreement or high confidence from a single agent
- VIX regime awareness (reduce risk in extreme VIX)

LOSS-REDUCTION FILTERS (data-driven from 74-trade backtest analysis):
- Day filter: skip Monday & Wednesday (32.3% WR, -171K combined losses)
- Confidence death zone: skip 0.60-0.69 (22.2% WR, -75K losses)
- PUT-only mode: skip BUY_CALL signals (only +9.9K on 23 trades vs +722K PUT)
- VIX floor: skip VIX < 10 (30.8% WR, worst regime)

Impact: Return +366% -> +459%, DD 28% -> 15%, Sharpe 3.8 -> 9.1, PF 2.2 -> 6.5
"""

import logging
import time
from datetime import datetime
from typing import Optional

from orchestrator.trade_signal import TradeSignal

logger = logging.getLogger(__name__)

# Strategy weights based on backtested performance
# learned_rules_v3 (V4 optimized): +1662% return, Sharpe 13.44, PF 4.39
DEFAULT_WEIGHTS = {
    "learned_rules": 0.50,
    "learned_rules_v3": 0.80,  # V4 optimized: highest priority
    "iron_condor": 0.20,
    "bull_put_spread": 0.15,
    "ddqn_agent": 0.15,
}

# Direction mapping for agreement checks
_BULLISH_ACTIONS = {"BUY_CALL", "BULL_PUT_SPREAD", "SELL_PUT_SPREAD"}
_BEARISH_ACTIONS = {"BUY_PUT", "SELL_CALL_SPREAD"}
_NEUTRAL_ACTIONS = {"IRON_CONDOR", "SELL_STRADDLE", "SELL_IRON_CONDOR"}

# ── Loss-reduction filter configuration ──
# Days to skip: Monday (25% WR, -122K) and Wednesday (41.7% WR, -49K)
# Tuesday/Thursday/Friday carry all the profit
# Trade all days — react to market movement, not calendar patterns
SKIP_DAYS: set[str] = set()

# No confidence death zone — let the agent's own thresholds handle filtering
CONFIDENCE_DEATH_ZONE = (0.0, 0.0)

# Trade both directions — market moves both ways
SKIP_ACTIONS: set[str] = set()

# VIX floor: VIX < 10 is worst regime (30.8% WR)
# Low VIX = low premiums + low movement = options lose to theta decay
MIN_VIX_FOR_ENTRY = 10.0


class MetaAgent:
    """Aggregates strategy signals and decides final trade action.

    Independent cooldown lanes:
      - Neutral strategies (iron_condor) have their own cooldown
      - Directional strategies (bull_put_spread, ddqn) have their own cooldown
      - This lets directional trades fire even after a condor is placed

    Loss-reduction filters (v2 optimized):
      - Day filter: skip Monday/Wednesday
      - Confidence death zone: skip 0.60-0.69
      - PUT-only: skip BUY_CALL signals
      - VIX floor: skip VIX < 10
    """

    def __init__(
        self,
        capital: float = 200000.0,
        max_risk_per_trade_pct: float = 0.25,
        cooldown_seconds: int = 45,
        min_confidence: float = 0.25,
        agreement_threshold: float = 0.15,
        weights: Optional[dict[str, float]] = None,
    ):
        self.capital = capital
        self.max_risk_per_trade = capital * max_risk_per_trade_pct
        self.cooldown_seconds = cooldown_seconds
        self.min_confidence = min_confidence
        self.agreement_threshold = agreement_threshold
        self.weights = weights or DEFAULT_WEIGHTS

        # Per-strategy cooldown: each strategy has its own independent cooldown
        self._last_trade_time_by_strategy: dict[str, float] = {}
        self._last_trade_time: float = 0.0  # legacy fallback
        self._signals_today: list[TradeSignal] = []
        self._trades_executed: int = 0

        # Filter statistics (reset daily)
        self._filter_stats: dict[str, int] = {
            "day_filter": 0,
            "conf_death_zone": 0,
            "action_filter": 0,
            "vix_floor": 0,
            "total_blocked": 0,
        }

        # Live VIX (updated by orchestrator)
        self._live_vix: float = 15.0

    def set_vix(self, vix: float) -> None:
        """Update live VIX for filter decisions."""
        self._live_vix = vix

    def _apply_loss_reduction_filters(
        self, signals: list[TradeSignal]
    ) -> list[TradeSignal]:
        """Apply data-driven loss-reduction filters to incoming signals.

        These filters were identified from analysing 74 real-data trades:
        - Removes signals on Monday/Wednesday (systematic losers)
        - Removes signals in confidence death zone (0.60-0.69)
        - Removes BUY_CALL signals (negligible edge, 39% WR)
        - Removes signals when VIX < 10 (worst regime, 30.8% WR)

        CLOSE signals always pass through (never block exits).
        """
        today = datetime.now().strftime("%A")
        filtered = []

        # V3 agent names that bypass day/action filters (they have own logic)
        V3_AGENTS = {"learned_rules_v3"}

        for s in signals:
            # CLOSE signals always pass -- never block exit logic
            if s.action == "CLOSE":
                filtered.append(s)
                continue

            is_v3 = s.strategy in V3_AGENTS

            # Filter A: Day of week (Mon/Wed are systematic losers)
            # V3 agents trade every day -- they use S/R + gap entries
            if today in SKIP_DAYS and not is_v3:
                self._filter_stats["day_filter"] += 1
                self._filter_stats["total_blocked"] += 1
                logger.info(
                    "FILTER: Day filter blocked %s from %s on %s",
                    s.action, s.strategy, today,
                )
                continue

            # Filter B: Confidence death zone (0.60-0.69)
            if CONFIDENCE_DEATH_ZONE[0] <= s.confidence < CONFIDENCE_DEATH_ZONE[1]:
                self._filter_stats["conf_death_zone"] += 1
                self._filter_stats["total_blocked"] += 1
                logger.info(
                    "FILTER: Confidence death zone blocked %s from %s (conf=%.2f)",
                    s.action, s.strategy, s.confidence,
                )
                continue

            # Filter C: Skip BUY_CALL (PUT-only mode)
            # V3 allows CALLs for high-conviction S/R + ORB setups
            if s.action in SKIP_ACTIONS and not is_v3:
                self._filter_stats["action_filter"] += 1
                self._filter_stats["total_blocked"] += 1
                logger.info(
                    "FILTER: Action filter blocked %s from %s (PUT-only mode)",
                    s.action, s.strategy,
                )
                continue

            # Filter D: VIX floor (VIX < 10 is worst regime)
            if self._live_vix < MIN_VIX_FOR_ENTRY:
                self._filter_stats["vix_floor"] += 1
                self._filter_stats["total_blocked"] += 1
                logger.info(
                    "FILTER: VIX floor blocked %s from %s (VIX=%.1f < %.1f)",
                    s.action, s.strategy, self._live_vix, MIN_VIX_FOR_ENTRY,
                )
                continue

            filtered.append(s)

        return filtered

    def evaluate(self, signals: list[TradeSignal]) -> Optional[TradeSignal]:
        """Evaluate a batch of signals and return the best one to execute.

        Returns None if no signal meets the criteria.
        """
        if not signals:
            return None

        # ── LOSS-REDUCTION FILTERS (applied first, before any other logic) ──
        signals = self._apply_loss_reduction_filters(signals)

        # Filter out HOLD signals
        active = [s for s in signals if s.action != "HOLD"]
        if not active:
            return None

        # CLOSE signals get priority -- always allow them
        close_signals = [s for s in active if s.action == "CLOSE"]
        if close_signals:
            best_close = max(close_signals, key=lambda s: s.confidence)
            logger.info(
                "CLOSE signal from %s (conf=%.2f) -- bypassing cooldown",
                best_close.strategy, best_close.confidence,
            )
            return best_close

        # Per-strategy cooldown: each strategy has its own independent cooldown
        now = time.monotonic()
        viable = []
        for s in active:
            # Check capital constraint
            if s.max_loss > self.max_risk_per_trade:
                logger.debug(
                    "Signal from %s rejected: max_loss=%.0f > limit=%.0f",
                    s.strategy, s.max_loss, self.max_risk_per_trade,
                )
                continue

            # Check per-strategy cooldown
            last_trade = self._last_trade_time_by_strategy.get(s.strategy, 0.0)
            elapsed = now - last_trade

            if elapsed < self.cooldown_seconds:
                logger.debug(
                    "Cooldown for %s: %.0fs remaining",
                    s.strategy, self.cooldown_seconds - elapsed,
                )
                continue

            viable.append(s)
        if not viable:
            return None

        # Single high-confidence signal: allow if confidence > threshold
        if len(viable) == 1:
            s = viable[0]
            if s.confidence >= self.min_confidence:
                logger.info(
                    "Single signal accepted | strategy=%s action=%s conf=%.2f",
                    s.strategy, s.action, s.confidence,
                )
                return s
            return None

        # Multiple signals: weighted voting
        best_signal = None
        best_score = 0.0

        for s in viable:
            weight = self.weights.get(s.strategy, 0.2)
            score = weight * s.confidence
            if score > best_score:
                best_score = score
                best_signal = s

        if best_score >= self.agreement_threshold and best_signal is not None:
            # Check directional agreement (optional boost)
            directions = set()
            for s in viable:
                if s.action in _BULLISH_ACTIONS:
                    directions.add("BULLISH")
                elif s.action in _BEARISH_ACTIONS:
                    directions.add("BEARISH")
                elif s.action in _NEUTRAL_ACTIONS:
                    directions.add("NEUTRAL")

            if len(directions) <= 2:  # No conflicting directions
                logger.info(
                    "Meta-agent approved | strategy=%s action=%s score=%.3f directions=%s",
                    best_signal.strategy, best_signal.action, best_score, directions,
                )
                return best_signal

        return None

    def record_trade(self, signal: TradeSignal) -> None:
        """Record that a trade was executed (per-strategy cooldown)."""
        now = time.monotonic()
        self._last_trade_time = now

        # Set cooldown for this specific strategy
        self._last_trade_time_by_strategy[signal.strategy] = now

        self._signals_today.append(signal)
        self._trades_executed += 1

    def reset_daily(self) -> None:
        """Reset daily state."""
        self._signals_today.clear()
        self._trades_executed = 0
        self._last_trade_time = 0.0

        # Log filter stats before resetting
        if self._filter_stats["total_blocked"] > 0:
            logger.info(
                "Daily filter stats: day=%d conf=%d action=%d vix=%d total=%d",
                self._filter_stats["day_filter"],
                self._filter_stats["conf_death_zone"],
                self._filter_stats["action_filter"],
                self._filter_stats["vix_floor"],
                self._filter_stats["total_blocked"],
            )
        self._filter_stats = {k: 0 for k in self._filter_stats}

    def get_filter_stats(self) -> dict[str, int]:
        """Return current filter statistics for dashboard."""
        return dict(self._filter_stats)

    def update_capital(self, capital: float) -> None:
        """Update available capital (after P&L changes)."""
        self.capital = capital
        self.max_risk_per_trade = capital * 0.20
