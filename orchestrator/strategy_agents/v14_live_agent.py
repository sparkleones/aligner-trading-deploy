"""V14 Production Live Agent -- Uses shared scoring engine (scoring/).

V14 PRODUCTION CONFIG (V14_T20_Optimized — 9.84x on 6-month 5-min backtest):
  - V8 indicator scoring every 5 min (Supertrend, EMA, RSI, MACD, BB, ADX)
  - 3 confluence filters: VWAP direction, RSI momentum gate, Squeeze suppression
  - RSI gates: CALL min=60, PUT max=40
  - Score thresholds: PUT >= 5.0, CALL >= 6.0
  - Entry windows: 9:30-10:30 (bars 3-15) + 14:00-14:45 (bars 59-69)
  - Wednesday blocked (23.5% WR), VIX floor 13, zero-hero disabled
  - VIX-adaptive lot scaling (0.3x in <13, 0.5x in 14-15, 1.5x in 15-17, 2x in 17+)
  - ORB entries get 2x lots (75% WR, PF 12.16)
  - Max lot cap: 30 (prevents catastrophic oversizing)
  - 200K capital, 50 trades, 48.0% WR, PF 2.41, MaxDD 54%
  - 4 rounds of tuning (16 + 20 + 19 + 21 = 76 variants tested)

Shares scoring engine with backtesting/v14_unified_backtest.py — ONE source of truth.
"""

import logging
import math
from typing import Optional, TYPE_CHECKING

import numpy as np

from orchestrator.trade_signal import OrderLeg, TradeSignal
from orchestrator.strategy_agents.base_agent import BaseLiveAgent

# ── SHARED SCORING ENGINE (same code as backtest) ──
from scoring.config import V15_CONFIG as _SHARED_CONFIG
from scoring.indicators import compute_indicators as _shared_compute_indicators
from scoring.engine import (
    score_entry as _shared_score_entry,
    passes_confluence as _shared_passes_confluence,
    evaluate_exit as _shared_evaluate_exit,
    compute_lots as _shared_compute_lots,
    detect_composite_entries as _shared_detect_composite,
    v17_btst_favorable as _shared_v17_btst_favorable,
)

from orchestrator.smart_strike_selector import SmartStrikeSelector

if TYPE_CHECKING:
    from orchestrator.market_analyzer import MarketAnalysis

logger = logging.getLogger(__name__)

STRIKE_INTERVAL = 50

# V14 Config — imported from shared module
V14_CONFIG = _SHARED_CONFIG

# Legacy config dict (kept for reference only — shared config is source of truth)
_V14_CONFIG_LEGACY = {
    "cooldown_bars": 2,
    "min_confidence": 0.35,
    "min_confidence_filter": 0.25,
    # Trail stops — REVERTED to Original values (Lean backtest: 8.9x vs 3.3x with wide trails)
    # Tighter trails lose money on the trail trade itself (-82K) but free position slots for
    # new high-quality entries. Net effect: time_exit P&L jumps from -4K to +13.9L.
    "trail_pct_put": 0.015,            # Reverted from 0.020 (slot turnover > trail profit)
    "trail_pct_call": 0.008,           # Reverted from 0.012
    "min_hold_trail_put": 24,          # 24 bars x 5 min = 120 min (reverted from 150)
    "min_hold_trail_call": 12,         # 12 bars x 5 min = 60 min (reverted from 90)
    "max_hold_put": 60,             # 60 bars x 5 min = 300 min
    "max_hold_call": 54,            # 54 bars x 5 min = 270 min
    "max_trades_per_day": 7,
    "max_concurrent": 3,
    "block_call_4th_hour": True,
    "block_late_entries": 61,       # Bar 61 x 5 min = 305 min
    "avoid_windows_bars": [(33, 57)],  # Lunch hour only (12:00-14:00). Removed 10:00-10:30 — good trend time
    # Confluence filters
    "use_vwap_filter": True,
    "use_squeeze_filter": True,
    "use_rsi_hard_gate": True,
    "rsi_call_min": 50,             # Lowered from 55 — widen entry window (ceiling at 60 blocks bad zone)
    "rsi_put_max": 50,              # Relaxed from 40 — RSI rarely hits <40 on 5-min bars
    # Scoring thresholds
    "put_score_min": 4.0,
    "call_score_min": 5.0,
    # Lot modifiers
    "put_bias_lot_mult": 1.3,
    "call_bias_lot_mult": 1.0,
    "vix_sweet_min": 14.0,
    "vix_sweet_max": 16.0,
    "vix_sweet_lot_mult": 1.4,
    "vix_danger_min": 16.0,
    "vix_danger_max": 18.0,
    "vix_danger_lot_mult": 0.5,
    "expiry_day_lot_mult": 1.0,
    "rsi_sweet_low": 20,
    "rsi_sweet_high": 35,
    "rsi_sweet_lot_mult": 1.5,
    "rsi_danger_low": 55,
    "rsi_danger_high": 65,
    "rsi_danger_lot_mult": 0.5,
    # VIX bounds
    "vix_floor": 11,
    "vix_ceil": 35,
    # ── Per-trade stop loss ──
    # DISABLED: backtest showed 0% WR, -Rs9.12L on 71 trades over 6 months.
    # Options premium naturally fluctuates 30%+ before recovering — hard SL exits too early.
    "hard_sl_pct": 0.0,             # Disabled (was 0.30)
    # ── Consecutive loss throttle — DISABLED (Lean backtest proved this blocks crash entries) ──
    # During crash months (Oct-2024), the first 2 losses trigger 60-min pause, missing the
    # crash move entirely. Enhanced went from -37K to +265K when this was disabled.
    "max_consecutive_losses": 99,   # Effectively disabled
    "loss_cooldown_bars": 0,        # No pause
    # ── Gap reversal filter (NEW) ──
    "gap_reversal_filter": True,    # Detect gap + reversal and suppress counter-trend entries
    "gap_threshold_pct": 0.004,     # 0.4% gap (NIFTY ~100 pts) triggers gap filter
    # ── Capital drawdown scaling — DISABLED (Lean backtest: blocks entries during crash moves) ──
    # After losing 1.5% intraday, lots shrink to 0.7x. After -2.5%, lots shrink to 0.3x.
    # In crash months, this prevents the strategy from capturing its biggest moves.
    "drawdown_lot_scale": False,    # Disabled (Lean: +265K Oct vs Full: -37K Oct)
    "drawdown_50pct_mult": 0.7,     # Unused when drawdown_lot_scale=False
    "drawdown_75pct_mult": 0.3,     # Unused when drawdown_lot_scale=False
    # ── RSI CALL kill zone (NEW: 23.3% WR, -Rs722K in 198-trade analysis) ──
    "rsi_call_kill_ceiling": 60,    # Block CALL when RSI > 60 (chasing extended moves)
    # ── ADX trend strength scoring (NEW: was in backtest V8, missing from live) ──
    "adx_choppy_min": 25,           # ADX 25-35 = choppy, dampen scores 0.8x
    "adx_choppy_max": 35,
    "adx_choppy_mult": 0.8,         # 20% score reduction in choppy zone
    "adx_weak_threshold": 18,       # ADX < 18 = no trend, heavy dampening
    "adx_weak_mult": 0.6,           # 40% score reduction when trendless
    "adx_strong_threshold": 35,     # ADX > 35 = strong trend, +1.0 aligned direction
    # ── Trail stop profit gate (NEW: prevents trail on flat/losing positions) ──
    "trail_min_profit_pct": 0.003,  # 0.3% favorable move before trail activates
    # ── Market microstructure filters (NEW: from research — uses market_analyzer data) ──
    "use_pcr_filter": True,         # Use Put-Call Ratio for directional bias
    "pcr_bullish_max": 0.7,         # PCR < 0.7 = bullish sentiment --> favor CALLs +0.5
    "pcr_bearish_min": 1.6,         # PCR > 1.6 = extreme fear --> contrarian CALL +1.0 (research)
    "use_iv_pctile_scaling": True,   # Scale lots based on IV percentile
    "iv_pctile_high": 75,           # IV > 75th pctile = options expensive --> reduce lots 0.7x
    "iv_pctile_high_mult": 0.7,
    "iv_pctile_low": 25,            # IV < 25th pctile = options cheap --> boost lots 1.3x
    "iv_pctile_low_mult": 1.3,
    "use_oi_levels": True,           # Use OI support/resistance for entry filtering
    "oi_proximity_pct": 0.003,       # 0.3% — block entries near strong OI levels
    # ── ATR-based trailing stops — DISABLED (backtest: ATR trails too aggressive on any bar freq) ──
    # 6-month backtest results:
    #   ATR trail ON (1.5x):  2.7x return, 124 trail exits, -73K from trails
    #   ATR trail ON (30x):   4.0x return, 26 trail exits, -15L from trails
    #   Fixed % trail (1.5%): 10.0x return, 7 trail exits, best model
    # ATR trails churn positions and cut winners before time_exit/eod_close.
    # Fixed % trail at 1.5% ≈ 1.85x daily ATR — already within recommended range.
    "use_atr_trail": False,          # DISABLED — fixed % wins decisively
    "atr_trail_mult": 1.5,           # Unused when use_atr_trail=False
    "atr_trail_adx_adaptive": True,  # Unused when use_atr_trail=False
    # ADX-adaptive multipliers (available if ATR trail re-enabled):
    "atr_trail_adx_low": 1.0,        # ADX < 20: tight (range-bound)
    "atr_trail_adx_mid": 1.5,        # ADX 20-35: standard
    "atr_trail_adx_high": 2.0,       # ADX > 35: wide (strong trend, let it run)
    # Chandelier exit (alternative trail method)
    "use_chandelier_exit": False,     # Highest High - N*ATR for longs
    "chandelier_atr_mult": 3.0,      # 3.0x ATR from highest high/lowest low
    "chandelier_period": 22,          # Lookback for highest high / lowest low
    # ── Standard deviation entry filter (from research: bell curve filter) ──
    "use_stddev_filter": False,  # Backtest proved no impact (NoStd = Std05)       # Only enter when price deviates > 1σ from daily mean
    "stddev_entry_threshold": 1.0,   # Minimum standard deviations from daily mean
    # ── Debit spread mode (from research: risk-defined trades) ──
    "use_debit_spreads": False,      # When True, create 2-leg debit spreads instead of naked buys
    "spread_width_strikes": 2,       # Spread width in strike intervals (2 x 50 = 100 pts for NIFTY)
    # ── Theta-aware exit (from research: exit long options Monday 3 PM before Tuesday theta decay) ──
    "theta_exit_enabled": True,      # Exit profitable positions on Monday afternoon
    "theta_exit_monday_bar": 69,     # Bar 69 = 3:00 PM (345 min / 5 min = 69 bars from 9:15)
    # ── IV crush event awareness (from research: "Don't buy options day before events") ──
    # Pre-RBI Policy: IV rises 2-3 days before, then crushes post-event.
    # Updated weekly by the user with upcoming event dates (RBI policy, Budget, etc.)
    "iv_crush_events": [],            # List of dates (YYYY-MM-DD strings) for known events
    "iv_crush_lookback_days": 1,      # Don't buy options this many days before event
}


class V14LiveAgent(BaseLiveAgent):
    """V14 Production agent: V8 indicators + VWAP/RSI/Squeeze confluence."""

    name = "v14_production"

    def __init__(self, capital: float = 30000.0, lot_size: int = 65):
        super().__init__(capital, lot_size)
        self._trades_today = 0
        self._open_positions: list[dict] = []
        self._last_exit_bar = -10
        self._today_date: Optional[str] = None
        self._prev_close: float = 0
        self._day_open: float = 0
        self._bar_history: list[dict] = []
        # ── Per-trade risk tracking (NEW) ──
        self._consecutive_losses: int = 0
        self._loss_pause_until_bar: int = -1  # bar index until which new entries are blocked
        self._daily_realised_pnl: float = 0.0  # track realised P&L for drawdown scaling
        self._trade_results: list[dict] = []   # {pnl, bar, action} for post-session analysis
        # ── Cached indicators for exit logic (ATR trail needs atr/adx from last compute) ──
        self._current_indicators: dict = {}
        # ── Market microstructure cache (set in generate_signal for _score_entry access) ──
        self._market_analysis: Optional["MarketAnalysis"] = None
        # ── Consecutive down day tracking (research: 3+ down days --> 57.9% UP probability) ──
        self._consecutive_down_days: int = 0
        self._prev_day_open: float = 0.0
        self._prev_day_close: float = 0.0
        # ── Composite entry state (NEW: gap, ORB, S/R bounce, zero-to-hero) ──
        self._orb_high: float = 0.0
        self._orb_low: float = 0.0
        self._support: float = 0.0
        self._resistance: float = 0.0
        self._gap_pct: float = 0.0      # Today's gap percentage
        self._gap_detected: bool = False # Already processed gap on bar 0
        # ── Equity compounding (NEW: dynamic lots based on current equity) ──
        self._running_equity: float = capital
        # ── HMM Regime detector (NEW: from research) ──
        try:
            from orchestrator.regime_detector import RegimeDetector
            self._regime_detector = RegimeDetector(lookback=50)
        except ImportError:
            self._regime_detector = None
        self._regime_adj: dict = {}  # Current regime adjustments (trail_mult, block_reversion, etc.)
        # ── ML win-probability filter (NEW: XGBoost gate from backtest data) ──
        self._ml_filter = None
        try:
            from orchestrator.ml_trade_filter import MLTradeFilter
            self._ml_filter = MLTradeFilter()
            if self._ml_filter.is_ready:
                logger.info("V14: ML trade filter loaded (model ready)")
            else:
                logger.info("V14: ML trade filter initialized (no model yet — run backtesting/ml_trade_classifier.py first)")
        except ImportError:
            logger.info("V14: ML trade filter not available")
        # ── AI Market Brain integration (NEW: advisory influence on decisions) ──
        self._ai_brain_state: dict = {}
        self._ai_influenced_count: int = 0  # Track how many decisions AI modified
        # ── Smart Strike Selector (NEW: replaces fixed ATM strike selection) ──
        cfg = V14_CONFIG
        if cfg.get("use_smart_strike", False):
            strike_cfg = cfg.get("smart_strike_config", {})
            self._strike_selector = SmartStrikeSelector(strike_cfg)
            logger.info("V14: SmartStrikeSelector enabled (delta=%.2f-%.2f, sweet=%.2f)",
                        strike_cfg.get("target_delta_min", 0.35),
                        strike_cfg.get("target_delta_max", 0.60),
                        strike_cfg.get("target_delta_sweet", 0.45))
        else:
            self._strike_selector = None
            logger.info("V14: SmartStrikeSelector disabled — using ATM strikes")

    # ── AI Brain interface ──

    def set_ai_brain_state(self, state: dict) -> None:
        """Update the latest AI brain analysis for decision influence.

        Called by run_autonomous.py after each AI analysis cycle (~every 5 min).
        The AI brain acts as an ADVISORY layer — it can:
          - Force exit all positions (EXIT_ALL + high conviction)
          - Block new entries (risk_assessment=extreme)
          - Scale lot size down (REDUCE_SIZE)
          - Boost/reduce confidence when sentiment aligns/conflicts
          - Override confluence filters (override_confluence=true + high conviction)

        The V14 R5 backtested logic remains the PRIMARY decision maker.
        AI influence is logged for full transparency.
        """
        if isinstance(state, dict):
            # Extract the nested 'analysis' if present (file format has wrapper)
            self._ai_brain_state = state.get("analysis", state)

    def _get_ai_brain(self) -> dict:
        """Get current AI brain analysis with safe defaults."""
        return self._ai_brain_state or {}

    def _estimate_dte(self, today) -> float:
        """Estimate days to expiry for Greeks calculation.

        Uses weekday heuristic: NIFTY weekly expiry is Thursday.
        Returns fractional days (minimum 0.2 = ~5 hours for expiry day).
        """
        from datetime import timedelta
        target_weekday = 3  # Thursday
        days_to_expiry = (target_weekday - today.weekday()) % 7
        if days_to_expiry == 0:
            # Expiry day: use remaining trading hours
            from datetime import datetime
            now = datetime.now()
            hours_left = max(0.5, (15.5 - now.hour - now.minute / 60.0))
            return hours_left / 6.25  # 6.25 trading hours/day
        return float(days_to_expiry)

    def _pick_product(
        self,
        action: str,
        indicators: dict,
        bar_idx: int,
        spot: float,
        vix: float,
    ) -> str:
        """V17 dynamic product selector: 'NRML' (BTST carry) or 'MIS' (intraday).

        Delegates to the shared v17_btst_favorable() function in
        scoring/engine.py so backtest and live use identical logic.

        Safe by default: if config flag use_v17_dynamic_product is False,
        indicators are missing, or any day-stat computation fails, returns
        'MIS' (the pre-V17 behavior).
        """
        try:
            if not V14_CONFIG.get("use_v17_dynamic_product"):
                return "MIS"
            # Day stats: high/low/open across today's bars only.
            # self._bar_history is pruned to the last 500 bars and may
            # span multiple days; filter to today using self._today_date.
            today = self._today_date or ""
            today_bars = [
                b for b in self._bar_history
                if (b.get("date") or str(b.get("time") or b.get("timestamp") or ""))[:10] == today
            ]
            if not today_bars:
                return "MIS"
            day_high = max(b["high"] for b in today_bars)
            day_low = min(b["low"] for b in today_bars)
            day_open = self._day_open if self._day_open > 0 else today_bars[0]["open"]

            # dte estimate — weekly Thursday expiry heuristic
            from datetime import date as _date
            try:
                if today:
                    y, m, d = today.split("-")
                    dte = self._estimate_dte(_date(int(y), int(m), int(d)))
                else:
                    dte = self._estimate_dte(_date.today())
            except Exception:
                dte = self._estimate_dte(_date.today())

            favorable = _shared_v17_btst_favorable(
                cfg=V14_CONFIG,
                ind=indicators or {},
                action=action,
                bar_idx=bar_idx,
                dte=dte,
                vix=vix,
                spot=spot,
                day_high=day_high,
                day_low=day_low,
                day_open=day_open,
            )
            return "NRML" if favorable else "MIS"
        except Exception:
            # Never let product selection break entry flow — fall back to MIS
            logger.exception("V14 product selection failed; defaulting to MIS")
            return "MIS"

    def set_previous_close(self, close: float) -> None:
        """Set previous day's close for gap calculation."""
        self._prev_close = close

    def sync_daily_pnl(self, realised_pnl: float) -> None:
        """Sync daily realised P&L from broker/risk manager for drawdown scaling.

        Called by the orchestrator to give the agent accurate P&L data
        instead of relying on internal estimates.
        """
        self._daily_realised_pnl = realised_pnl

    def update_equity(self, equity: float) -> None:
        """Update running equity for dynamic lot compounding.

        Called by the orchestrator with the current broker equity.
        Recalculates base lot count so position sizes grow with profits.
        """
        if equity <= 0:
            return
        self._running_equity = equity
        # Recalculate base lots using SPAN margin model
        # Use 70% of equity (same as backtest get_dynamic_lots)
        available = equity * 0.70
        span_per_lot = 40000  # Default SPAN margin per lot
        new_lots = max(1, int(available / span_per_lot))
        # Respect NSE freeze limit (NIFTY max qty = 1800, so max lots = 1800 / 65 = 27)
        new_lots = min(new_lots, 27)
        if new_lots != self._num_lots:
            logger.info("V14 COMPOUND: equity=%.0f -> base lots %d -> %d",
                        equity, self._num_lots, new_lots)
            self._num_lots = new_lots

    def _reset_day(self) -> None:
        """Reset state for new trading day.

        NOTE: We deliberately do NOT clear ``self._bar_history`` here.
        Indicators (RSI/EMA/MACD/ATR/BB/ADX/...) need their full lookback
        window to remain converged across the day-boundary; wiping the
        buffer would force a multi-bar warmup at 09:15 every morning,
        during which compute_indicators() returns None and the agent
        sits idle. The 500-bar cap in _handle_bar() keeps memory bounded.
        VWAP today-anchoring is handled inside compute_indicators() via
        the today_date filter, so cross-day bars in the buffer do not
        contaminate the intraday VWAP.
        """
        self._trades_today = 0
        self._open_positions = []
        self._last_exit_bar = -10
        self._consecutive_losses = 0
        self._loss_pause_until_bar = -1
        self._daily_realised_pnl = 0.0
        self._trade_results = []
        self._orb_high = 0.0
        self._orb_low = 0.0
        self._support = 0.0
        self._resistance = 0.0
        self._gap_pct = 0.0
        self._gap_detected = False

    def rollback_position(self, signal=None) -> None:
        """Remove the most recently added position (e.g. when order execution fails).

        Called by the orchestrator when a trade signal was generated but the
        broker rejected / failed the order.  Without this, the agent would
        keep trying to exit a phantom position every tick.
        """
        if self._open_positions:
            removed = self._open_positions.pop()
            self._trades_today = max(0, self._trades_today - 1)
            logger.warning(
                "Rolled back phantom position: %s %s%s (order failed)",
                removed.get("action"), removed.get("strike"), removed.get("opt_type"),
            )

    def get_decision_state(self, current_spot: float = 0, vix: float = 0) -> dict:
        """Export current scoring and trigger levels for dashboard display."""
        ind = self._compute_indicators()
        if not ind:
            return {
                "ready": False,
                "reason": f"Warming up ({len(self._bar_history)}/15 bars)",
                "smart_strike_enabled": self._strike_selector is not None,
                "smart_strike_last": self._strike_selector.last_selection if self._strike_selector else None,
            }

        cfg = V14_CONFIG
        _, _ = self._score_entry(ind, vix)

        # Recompute individual scores for display
        spot = ind["close"]
        call_score = 0.0
        put_score = 0.0
        call_breakdown = []
        put_breakdown = []

        # Supertrend
        if ind["st_direction"] == 1:
            call_score += 2.5
            call_breakdown.append("Supertrend +2.5")
        else:
            put_score += 3.0
            put_breakdown.append("Supertrend +3.0")

        # EMA
        if ind["ema9_above_ema21"]:
            call_score += 2.0
            call_breakdown.append("EMA9>21 +2.0")
        else:
            put_score += 3.5
            put_breakdown.append("EMA9<21 +3.5")

        # RSI
        rsi = ind["rsi"]
        if 30 <= rsi < 50:
            put_score += 1.5
            put_breakdown.append(f"RSI={rsi:.0f} +1.5")
        elif rsi < 30:
            call_score += 2.0
            call_breakdown.append(f"RSI={rsi:.0f} +2.0")
        elif rsi > 70:
            put_score += 2.0
            put_breakdown.append(f"RSI={rsi:.0f} +2.0")

        # MACD
        if ind.get("macd_hist", 0) > 0:
            call_score += 0.5
            call_breakdown.append("MACD +0.5")
        elif ind.get("macd_hist", 0) < 0:
            put_score += 0.5
            put_breakdown.append("MACD +0.5")

        # Bollinger
        if spot <= ind.get("bb_lower", spot):
            call_score += 1.5
            call_breakdown.append("BB_low +1.5")
        if spot >= ind.get("bb_upper", spot + 1):
            put_score += 1.5
            put_breakdown.append("BB_high +1.5")

        # VIX
        if 13 <= vix < 16:
            put_score += 1.5
            put_breakdown.append(f"VIX={vix:.0f} +1.5")
        elif vix >= 16:
            put_score += 1.0
            put_breakdown.append(f"VIX={vix:.0f} +1.0")

        # ADX dampening/boosting (matches _score_entry)
        adx = ind.get("adx", 25)
        if adx < cfg.get("adx_weak_threshold", 18):
            call_score *= cfg.get("adx_weak_mult", 0.6)
            put_score *= cfg.get("adx_weak_mult", 0.6)
            call_breakdown.append(f"ADX={adx:.0f} ×0.6")
            put_breakdown.append(f"ADX={adx:.0f} ×0.6")
        elif cfg.get("adx_choppy_min", 25) <= adx < cfg.get("adx_choppy_max", 35):
            call_score *= cfg.get("adx_choppy_mult", 0.8)
            put_score *= cfg.get("adx_choppy_mult", 0.8)
            call_breakdown.append(f"ADX={adx:.0f} ×0.8")
            put_breakdown.append(f"ADX={adx:.0f} ×0.8")
        elif adx >= cfg.get("adx_strong_threshold", 35):
            if ind.get("plus_di", 0) > ind.get("minus_di", 0):
                call_score += 1.0
                call_breakdown.append(f"ADX={adx:.0f} +DI +1.0")
            else:
                put_score += 1.0
                put_breakdown.append(f"ADX={adx:.0f} -DI +1.0")

        # Stochastic RSI
        if ind.get("stoch_oversold") and ind.get("stoch_cross_up"):
            call_score += 1.0
            call_breakdown.append("StochRSI oversold↑ +1.0")
        if ind.get("stoch_overbought") and ind.get("stoch_cross_down"):
            put_score += 1.0
            put_breakdown.append("StochRSI overbought↓ +1.0")

        # CCI
        cci = ind.get("cci", 0)
        if cci < -100:
            call_score += 0.5
            call_breakdown.append(f"CCI={cci:.0f} +0.5")
        elif cci > 100:
            put_score += 0.5
            put_breakdown.append(f"CCI={cci:.0f} +0.5")

        # Williams %R
        williams_r = ind.get("williams_r", -50)
        if williams_r < -80:
            call_score += 0.5
            call_breakdown.append(f"W%R={williams_r:.0f} +0.5")
        elif williams_r > -20:
            put_score += 0.5
            put_breakdown.append(f"W%R={williams_r:.0f} +0.5")

        # RSI Bullish Divergence
        if ind.get("rsi_bullish_divergence"):
            call_score += 1.5
            call_breakdown.append("RSI Divergence +1.5")

        vwap = ind.get("vwap", spot)
        live_spot = current_spot if current_spot > 0 else spot

        # Decision levels
        put_needs = max(0, cfg["put_score_min"] - put_score)
        call_needs = max(0, cfg["call_score_min"] - call_score)

        # What would trigger each direction
        put_triggers = []
        call_triggers = []
        if live_spot >= vwap:
            put_triggers.append(f"Price below {vwap:.0f} (VWAP)")
        if rsi > cfg["rsi_put_max"]:
            put_triggers.append(f"RSI below {cfg['rsi_put_max']} (now {rsi:.0f})")
        if put_needs > 0:
            put_triggers.append(f"Score needs +{put_needs:.1f} more")

        if live_spot <= vwap:
            call_triggers.append(f"Price above {vwap:.0f} (VWAP)")
        if rsi < cfg["rsi_call_min"]:
            call_triggers.append(f"RSI above {cfg['rsi_call_min']} (now {rsi:.0f})")
        if call_needs > 0:
            call_triggers.append(f"Score needs +{call_needs:.1f} more")

        # ── Risk guard indicators (NEW) ──
        risk_guards = []
        if self._consecutive_losses >= cfg.get("max_consecutive_losses", 2):
            risk_guards.append(f"⚠ {self._consecutive_losses} consecutive losses — paused")
        if self._loss_pause_until_bar > 0:
            risk_guards.append(f"⏸ Entries paused until bar {self._loss_pause_until_bar}")
        daily_loss = self._daily_realised_pnl / self.capital if self.capital > 0 else 0.0
        if daily_loss < -0.015:  # >1.5% loss
            risk_guards.append(f"📉 Daily loss {daily_loss:.1%} — lots reduced")
        if cfg.get("gap_reversal_filter") and self._prev_close > 0 and self._day_open > 0:
            gap_pct = (self._day_open - self._prev_close) / self._prev_close
            if abs(gap_pct) >= cfg.get("gap_threshold_pct", 0.004):
                direction = "DOWN" if gap_pct < 0 else "UP"
                risk_guards.append(f"📊 Gap {direction} {abs(gap_pct):.1%} detected")

        # Call readiness must also pass the kill zone ceiling
        call_rsi_ok = (rsi >= cfg["rsi_call_min"]
                       and rsi <= cfg.get("rsi_call_kill_ceiling", 60))

        return {
            "ready": True,
            "put_score": round(put_score, 1),
            "put_min": cfg["put_score_min"],
            "put_breakdown": put_breakdown,
            "call_score": round(call_score, 1),
            "call_min": cfg["call_score_min"],
            "call_breakdown": call_breakdown,
            "rsi": round(rsi, 1),
            "vwap": round(vwap, 1),
            "spot": round(live_spot, 1),
            "ema9_above_ema21": ind["ema9_above_ema21"],
            "squeeze": ind.get("squeeze_on", False),
            "bb_squeeze": ind.get("squeeze_on", False),
            "supertrend": "Bullish" if ind["st_direction"] == 1 else "Bearish",
            "st_direction": ind.get("st_direction", 0),
            "adx": round(adx, 1),
            "stoch_k": round(ind.get("stoch_k", 50), 1),
            "stoch_d": round(ind.get("stoch_d", 50), 1),
            "cci": round(ind.get("cci", 0), 1),
            "williams_r": round(ind.get("williams_r", -50), 1),
            "macd_hist": round(ind.get("macd_hist", 0), 2),
            "rsi_bullish_divergence": ind.get("rsi_bullish_divergence", False),
            # ── R5 Indicators for Market Pulse ──
            "connors_rsi": round(ind.get("connors_rsi", 50), 1),
            "kama_slope_up": ind.get("kama_slope_up", False),
            "psar_bullish": ind.get("psar_bullish", False),
            "donchian_breakout_up": ind.get("donchian_breakout_up", False),
            "donchian_breakout_down": ind.get("donchian_breakout_down", False),
            "ha_bullish": ind.get("ha_green_streak", 0) > 0,
            "ha_bearish": ind.get("ha_red_streak", 0) > 0,
            "ha_green_streak": ind.get("ha_green_streak", 0),
            "ha_red_streak": ind.get("ha_red_streak", 0),
            "atr": round(ind.get("atr", 0), 1),
            "put_triggers": put_triggers,
            "call_triggers": call_triggers,
            "put_ready": put_score >= cfg["put_score_min"] and rsi <= cfg["rsi_put_max"],
            "call_ready": call_score >= cfg["call_score_min"] and call_rsi_ok,
            "risk_guards": risk_guards,
            "consecutive_losses": self._consecutive_losses,
            "daily_est_pnl": round(self._daily_realised_pnl, 0),
            "regime": self._regime_detector.current_regime.value if self._regime_detector else "unknown",
            "hurst": round(self._regime_detector.last_info.hurst_exponent, 3) if self._regime_detector and self._regime_detector.last_info else 0.5,
            # ── AI Brain influence state ──
            "ai_brain_active": bool(self._ai_brain_state),
            "ai_brain_action": self._ai_brain_state.get("recommended_action", "N/A") if self._ai_brain_state else "N/A",
            "ai_brain_sentiment": self._ai_brain_state.get("sentiment", "N/A") if self._ai_brain_state else "N/A",
            "ai_brain_conviction": self._ai_brain_state.get("conviction", "N/A") if self._ai_brain_state else "N/A",
            "ai_brain_risk": self._ai_brain_state.get("risk_assessment", "N/A") if self._ai_brain_state else "N/A",
            "ai_influenced_count": self._ai_influenced_count,
            # ── Smart Strike Selector state ──
            "smart_strike_enabled": self._strike_selector is not None,
            "smart_strike_last": self._strike_selector.last_selection if self._strike_selector else None,
        }

    def _compute_indicators(self) -> Optional[dict]:
        """Compute V8 indicators from bar history — delegates to shared engine."""
        return _shared_compute_indicators(self._bar_history, self._today_date or "")

    def _compute_indicators_LEGACY(self) -> Optional[dict]:
        """LEGACY: Original indicator computation (kept for reference).

        FIXED (Apr 7 post-mortem):
        - Supertrend: proper stateful implementation with band ratcheting
        - ATR: true range with EWM smoothing (not simple high-low)
        - VWAP: today-only anchored with volume approximation
        - Added: ADX for trend strength filtering
        """
        bars = self._bar_history
        n = len(bars)
        if n < 15:
            return None

        closes = np.array([b["close"] for b in bars])
        highs = np.array([b["high"] for b in bars])
        lows = np.array([b["low"] for b in bars])

        ind = {"close": closes[-1]}

        # RSI (14-period) — Wilder's smoothed
        if n >= 15:
            deltas = np.diff(closes[-15:])
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            avg_gain = np.mean(gains) + 1e-10
            avg_loss = np.mean(losses) + 1e-10
            ind["rsi"] = 100 - (100 / (1 + avg_gain / avg_loss))
        else:
            ind["rsi"] = 50.0

        # EMA helper
        def ema(data, period):
            if len(data) < period:
                return data[-1]
            mult = 2 / (period + 1)
            e = np.mean(data[:period])
            for v in data[period:]:
                e = (v - e) * mult + e
            return e

        ind["ema9"] = ema(closes, 9)
        ind["ema21"] = ema(closes, 21)
        ind["ema9_above_ema21"] = ind["ema9"] > ind["ema21"]

        # Bollinger Bands (20, 2)
        if n >= 20:
            bb_slice = closes[-20:]
            bb_mid = np.mean(bb_slice)
            bb_std = np.std(bb_slice, ddof=1)
            ind["bb_upper"] = bb_mid + 2 * bb_std
            ind["bb_lower"] = bb_mid - 2 * bb_std
        else:
            ind["bb_upper"] = ind["bb_lower"] = closes[-1]

        # ── ATR (14-period) — TRUE RANGE with EWM smoothing (FIXED) ──
        # True Range = max(H-L, |H-prevC|, |L-prevC|)
        period_atr = 14
        lookback = min(n, period_atr + 1)
        tr_vals = []
        for i in range(1, lookback):
            idx = n - lookback + i
            tr = max(
                highs[idx] - lows[idx],
                abs(highs[idx] - closes[idx - 1]),
                abs(lows[idx] - closes[idx - 1]),
            )
            tr_vals.append(tr)
        if tr_vals:
            # EWM smoothing (Wilder's method: alpha = 1/period)
            atr_val = tr_vals[0]
            alpha = 1.0 / period_atr
            for tr in tr_vals[1:]:
                atr_val = atr_val * (1 - alpha) + tr * alpha
            ind["atr"] = atr_val
        else:
            ind["atr"] = 50.0

        # Keltner/Squeeze
        kc_upper = ind["ema21"] + 1.5 * ind["atr"]
        kc_lower = ind["ema21"] - 1.5 * ind["atr"]
        ind["squeeze_on"] = (ind["bb_lower"] > kc_lower) and (ind["bb_upper"] < kc_upper)

        # ── VWAP — today-only anchored (FIXED) ──
        # Use bars from today only, approximate volume with range * 1.0
        today_str = self._today_date or ""
        today_tp = []
        today_vol = []
        for b in bars:
            bar_time = str(b.get("time", b.get("timestamp", b.get("date", ""))))
            if today_str and today_str in bar_time:
                tp_val = (b["high"] + b["low"] + b["close"]) / 3.0
                vol = b.get("volume", 0)
                if vol <= 0:
                    vol = b["high"] - b["low"] + 1.0  # Range as volume proxy
                today_tp.append(tp_val)
                today_vol.append(vol)
        if today_tp:
            tp_arr = np.array(today_tp)
            vol_arr = np.array(today_vol)
            ind["vwap"] = np.sum(tp_arr * vol_arr) / (np.sum(vol_arr) + 1e-10)
        else:
            # Fallback to simple mean if no today bars
            tp = (highs + lows + closes) / 3.0
            ind["vwap"] = np.mean(tp)

        # ── SUPERTREND — proper stateful implementation (FIXED) ──
        # Uses ATR multiplier = 3.0, with band ratcheting and direction memory
        st_mult = 3.0
        st_period = 10
        if n >= st_period + 1:
            # Compute ATR for Supertrend (separate from main ATR)
            st_atr = ind["atr"]  # Reuse main ATR

            # Walk through bars to build stateful Supertrend
            st_dir = 1  # Start bullish
            upper_band = 0.0
            lower_band = 0.0

            for i in range(max(0, n - 50), n):  # Last 50 bars for state buildup
                hl2 = (highs[i] + lows[i]) / 2.0
                basic_upper = hl2 + st_mult * st_atr
                basic_lower = hl2 - st_mult * st_atr

                # Band ratcheting: lower band only moves UP, upper band only moves DOWN
                if i > 0:
                    if basic_lower > lower_band or closes[i - 1] < lower_band:
                        lower_band = basic_lower
                    # else keep previous lower_band (ratchet up)
                    if basic_upper < upper_band or closes[i - 1] > upper_band:
                        upper_band = basic_upper
                    # else keep previous upper_band (ratchet down)
                else:
                    upper_band = basic_upper
                    lower_band = basic_lower

                # Direction flip
                if st_dir == 1:  # Currently bullish
                    if closes[i] < lower_band:
                        st_dir = -1  # Flip to bearish
                else:  # Currently bearish
                    if closes[i] > upper_band:
                        st_dir = 1  # Flip to bullish

            ind["st_direction"] = st_dir
        else:
            ind["st_direction"] = 1 if closes[-1] > closes[-2] else -1

        # ── ADX — trend strength (NEW, was in backtest but missing from live) ──
        adx_period = 14
        if n >= adx_period + 2:
            plus_dm = []
            minus_dm = []
            for i in range(n - adx_period - 1, n):
                up_move = highs[i] - highs[i - 1]
                down_move = lows[i - 1] - lows[i]
                plus_dm.append(max(up_move, 0) if up_move > down_move else 0)
                minus_dm.append(max(down_move, 0) if down_move > up_move else 0)
            atr_sum = ind["atr"] * adx_period  # Approximate
            if atr_sum > 0:
                plus_di = 100 * np.mean(plus_dm) / (atr_sum / adx_period + 1e-10)
                minus_di = 100 * np.mean(minus_dm) / (atr_sum / adx_period + 1e-10)
                dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
                ind["adx"] = dx
                ind["plus_di"] = plus_di
                ind["minus_di"] = minus_di
            else:
                ind["adx"] = 25.0
                ind["plus_di"] = 0.0
                ind["minus_di"] = 0.0
        else:
            ind["adx"] = 25.0
            ind["plus_di"] = 0.0
            ind["minus_di"] = 0.0

        # ── STOCHASTIC RSI (14, 14, 3, 3) — reversal confirmation (NEW) ──
        # Matches backtest V8: +1.0 on oversold crossup (CALL), overbought crossdown (PUT)
        stoch_need = 14 + 14 + 3  # RSI period + stoch period + K smoothing
        if n >= stoch_need:
            # Compute RSI for each of the last stoch_need+3 bars (need extra for D smoothing)
            compute_len = min(n, stoch_need + 5)
            rsi_series = []
            for j in range(n - compute_len, n):
                if j < 14:
                    rsi_series.append(50.0)
                    continue
                d = np.diff(closes[j - 14:j + 1])
                g = np.where(d > 0, d, 0)
                l_vals = np.where(d < 0, -d, 0)
                ag = np.mean(g) + 1e-10
                al = np.mean(l_vals) + 1e-10
                rsi_series.append(100 - 100 / (1 + ag / al))

            rsi_arr = np.array(rsi_series)

            # Stochastic of RSI over 14-period window
            stoch_raw = []
            for j in range(14, len(rsi_arr)):
                window = rsi_arr[j - 13:j + 1]
                rmin, rmax = np.min(window), np.max(window)
                stoch_raw.append((rsi_arr[j] - rmin) / (rmax - rmin + 1e-10) * 100)

            # K = 3-SMA of stoch_raw, D = 3-SMA of K
            if len(stoch_raw) >= 6:
                k_vals = [np.mean(stoch_raw[max(0, j - 2):j + 1]) for j in range(len(stoch_raw))]
                d_vals = [np.mean(k_vals[max(0, j - 2):j + 1]) for j in range(len(k_vals))]
                ind["stoch_k"] = k_vals[-1]
                ind["stoch_d"] = d_vals[-1]
                ind["stoch_oversold"] = k_vals[-1] < 20
                ind["stoch_overbought"] = k_vals[-1] > 80
                ind["stoch_cross_up"] = (
                    len(k_vals) >= 2 and len(d_vals) >= 2
                    and k_vals[-1] > d_vals[-1] and k_vals[-2] <= d_vals[-2]
                )
                ind["stoch_cross_down"] = (
                    len(k_vals) >= 2 and len(d_vals) >= 2
                    and k_vals[-1] < d_vals[-1] and k_vals[-2] >= d_vals[-2]
                )

        # MACD (simplified)
        if n >= 26:
            ema12 = ema(closes, 12)
            ema26 = ema(closes, 26)
            ind["macd_hist"] = ema12 - ema26
        else:
            ind["macd_hist"] = 0

        # ── Standard Deviation Filter (from research: enter only beyond 1σ) ──
        # Compute today's return distribution and current deviation from mean
        today_str = self._today_date or ""
        today_closes = []
        for b in bars:
            bar_time = str(b.get("time", b.get("timestamp", b.get("date", ""))))
            if today_str and today_str in bar_time:
                today_closes.append(b["close"])
        if len(today_closes) >= 5:
            tc = np.array(today_closes)
            daily_mean = np.mean(tc)
            daily_std = np.std(tc, ddof=1)
            if daily_std > 0:
                ind["price_zscore"] = (closes[-1] - daily_mean) / daily_std
            else:
                ind["price_zscore"] = 0.0
            ind["daily_mean"] = daily_mean
            ind["daily_std"] = daily_std
        else:
            ind["price_zscore"] = 0.0
            ind["daily_mean"] = closes[-1]
            ind["daily_std"] = 0.0

        # ── CCI (14-period) — Commodity Channel Index (NEW) ──
        # CCI = (TP - SMA(TP, 14)) / (0.015 * Mean Deviation)
        cci_period = 14
        if n >= cci_period:
            tp_series = (highs[-cci_period:] + lows[-cci_period:] + closes[-cci_period:]) / 3.0
            tp_sma = np.mean(tp_series)
            mean_dev = np.mean(np.abs(tp_series - tp_sma))
            ind["cci"] = (tp_series[-1] - tp_sma) / (0.015 * mean_dev + 1e-10)
        else:
            ind["cci"] = 0.0

        # ── Williams %R (14-period) (NEW) ──
        # %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
        wr_period = 14
        if n >= wr_period:
            highest_high = np.max(highs[-wr_period:])
            lowest_low = np.min(lows[-wr_period:])
            ind["williams_r"] = (highest_high - closes[-1]) / (highest_high - lowest_low + 1e-10) * -100
        else:
            ind["williams_r"] = -50.0

        # ── RSI Bullish Divergence Detection (NEW) ──
        # Price made lower low BUT RSI made higher low over last 20 bars
        # Strong contrarian buy signal (research: caught 350pt bounce)
        div_lookback = 20
        if n >= div_lookback and "rsi" in ind:
            lookback_closes = closes[-div_lookback:]
            # Compute RSI for each bar in the lookback window
            rsi_lookback = []
            for j in range(n - div_lookback, n):
                if j < 14:
                    rsi_lookback.append(50.0)
                    continue
                d = np.diff(closes[j - 14:j + 1])
                g = np.where(d > 0, d, 0)
                l_vals = np.where(d < 0, -d, 0)
                ag = np.mean(g) + 1e-10
                al = np.mean(l_vals) + 1e-10
                rsi_lookback.append(100 - 100 / (1 + ag / al))
            rsi_lookback = np.array(rsi_lookback)
            # Price lower low: current close < min of lookback closes (excluding current)
            price_lower_low = closes[-1] < np.min(lookback_closes[:-1])
            # RSI higher low: current RSI > min of lookback RSIs (excluding current)
            rsi_higher_low = ind["rsi"] > np.min(rsi_lookback[:-1])
            ind["rsi_bullish_divergence"] = bool(price_lower_low and rsi_higher_low)
        else:
            ind["rsi_bullish_divergence"] = False

        return ind

    def _score_entry(self, ind: dict, vix: float, pcr: float = 0.0,
                     bar_idx: int = 0, is_expiry: bool = False) -> tuple:
        """V8 indicator scoring — delegates to shared engine."""
        ma = self._market_analysis
        return _shared_score_entry(
            ind, vix, V14_CONFIG,
            pcr=pcr,
            bar_idx=bar_idx,
            oi_support=getattr(ma, "oi_support", 0.0) or 0.0 if ma else 0.0,
            oi_resistance=getattr(ma, "oi_resistance", 0.0) or 0.0 if ma else 0.0,
            max_pain=getattr(ma, "max_pain", 0.0) or 0.0 if ma else 0.0,
            consecutive_down_days=self._consecutive_down_days,
            fii_net=getattr(ma, "fii_net", 0.0) or 0.0 if ma else 0.0,
            dii_net=getattr(ma, "dii_net", 0.0) or 0.0 if ma else 0.0,
            regime_block_reversion=self._regime_adj.get("block_reversion", False),
            oi_ce_change_pct=getattr(ma, "oi_ce_change_pct", 0.0) or 0.0 if ma else 0.0,
            oi_pe_change_pct=getattr(ma, "oi_pe_change_pct", 0.0) or 0.0 if ma else 0.0,
            is_expiry=is_expiry,
        )

    def _score_entry_LEGACY(self, ind: dict, vix: float, pcr: float = 0.0,
                     bar_idx: int = 0) -> tuple:
        """LEGACY: Original scoring (kept for reference).

        Parameters
        ----------
        pcr : float
            Put-Call Ratio from market analyzer (0 = unavailable).
        bar_idx : int
            Current bar index (0 = market open). Used for closing hour bias.
        """
        spot = ind["close"]
        call_score = 0.0
        put_score = 0.0

        # Supertrend
        if ind["st_direction"] == 1:
            call_score += 2.5
        elif ind["st_direction"] == -1:
            put_score += 3.0

        # EMA alignment
        if ind["ema9_above_ema21"]:
            call_score += 2.0
        else:
            put_score += 3.5

        # RSI — with ADX-gated AND regime-gated mean-reversion signals
        # Research: block reversion (RSI < 30 CALL, RSI > 70 PUT) when ADX > 25 (trending)
        # Regime block_reversion: True in TRENDING/VOLATILE regimes --> no reversion at all
        rsi = ind["rsi"]
        adx_val = ind.get("adx", 25)
        regime_block_reversion = self._regime_adj.get("block_reversion", False)
        if 30 <= rsi < 50:
            put_score += 1.5
        elif rsi < 30:
            if regime_block_reversion:
                call_score += 0.0  # Regime says no reversion signals
            elif adx_val < 25:  # Only give reversion score in non-trending markets
                call_score += 2.0
            else:
                call_score += 0.5  # Reduced score — reversion risky in trends
        elif rsi > 70:
            if regime_block_reversion:
                put_score += 0.0  # Regime says no reversion signals
            elif adx_val < 25:  # Only give reversion score in non-trending markets
                put_score += 2.0
            else:
                put_score += 0.5  # Reduced score — reversion risky in trends

        # MACD
        if ind["macd_hist"] > 0:
            call_score += 0.5
        elif ind["macd_hist"] < 0:
            put_score += 0.5

        # Bollinger
        if spot <= ind["bb_lower"]:
            call_score += 1.5
        if spot >= ind["bb_upper"]:
            put_score += 1.5

        # VIX regime
        if 13 <= vix < 16:
            put_score += 1.5
        elif vix >= 16:
            put_score += 1.0

        # ── ADX trend strength dampening/boosting (NEW) ──
        # From V8 backtest: ADX 25-35 is choppy (38% WR, -Rs2K), ADX > 35 is strong trend
        cfg = V14_CONFIG
        adx = ind.get("adx", 25)
        if adx < cfg.get("adx_weak_threshold", 18):
            # No trend — heavy dampening
            call_score *= cfg.get("adx_weak_mult", 0.6)
            put_score *= cfg.get("adx_weak_mult", 0.6)
        elif cfg.get("adx_choppy_min", 25) <= adx < cfg.get("adx_choppy_max", 35):
            # Choppy zone — moderate dampening
            call_score *= cfg.get("adx_choppy_mult", 0.8)
            put_score *= cfg.get("adx_choppy_mult", 0.8)
        elif adx >= cfg.get("adx_strong_threshold", 35):
            # Strong trend — boost the direction aligned with DI
            if ind.get("plus_di", 0) > ind.get("minus_di", 0):
                call_score += 1.0
            else:
                put_score += 1.0

        # ── Stochastic RSI reversal confirmation (NEW) ──
        # From V8 backtest: oversold + crossup --> CALL +1.0, overbought + crossdown --> PUT +1.0
        if ind.get("stoch_oversold") and ind.get("stoch_cross_up"):
            call_score += 1.0
        if ind.get("stoch_overbought") and ind.get("stoch_cross_down"):
            put_score += 1.0

        # ── CCI mean-reversion signals (NEW) ──
        # CCI < -100 = oversold --> bounce expected --> favor CALLs
        # CCI > +100 = overbought --> drop expected --> favor PUTs
        cci = ind.get("cci", 0)
        if cci < -100:
            call_score += 0.5
        elif cci > 100:
            put_score += 0.5

        # ── Williams %R extremes (NEW) ──
        # %R < -80 = oversold --> favor CALLs
        # %R > -20 = overbought --> favor PUTs
        williams_r = ind.get("williams_r", -50)
        if williams_r < -80:
            call_score += 0.5
        elif williams_r > -20:
            put_score += 0.5

        # ── RSI Bullish Divergence (NEW: strong contrarian buy from research — caught 350pt bounce) ──
        # Price lower low + RSI higher low = exhaustion sell-off --> expect reversal up
        if ind.get("rsi_bullish_divergence"):
            call_score += 1.5

        # ── PCR extreme contrarian signals (UPDATED: from research) ──
        # PCR > 1.6 = extreme fear = contrarian buy --> +1.0 call_score
        # PCR < 0.6 = extreme greed = correction ahead --> +1.0 put_score
        if cfg.get("use_pcr_filter") and pcr > 0:
            if pcr > cfg.get("pcr_bearish_min", 1.6):
                call_score += 1.0  # Contrarian: extreme fear --> expect bounce
            elif pcr < cfg.get("pcr_bullish_max", 0.7):
                if pcr < 0.6:
                    put_score += 1.0  # Extreme greed --> correction ahead
                else:
                    call_score += 0.5  # Mild bullish flow

        # ── OI Buildup S/R gating (from research: reduce score near OI walls) ──
        # If spot is within 0.5% of OI resistance --> reduce call_score (resistance overhead)
        # If spot is within 0.5% of OI support --> reduce put_score (support below)
        ma = self._market_analysis
        if cfg.get("use_oi_levels") and ma:
            oi_resistance = getattr(ma, "oi_resistance", 0.0) or 0.0
            oi_support = getattr(ma, "oi_support", 0.0) or 0.0
            if oi_resistance > 0 and spot > 0:
                dist_to_res = (oi_resistance - spot) / spot
                if 0 < dist_to_res < 0.005:  # Within 0.5% of resistance
                    call_score -= 1.0
            if oi_support > 0 and spot > 0:
                dist_to_sup = (spot - oi_support) / spot
                if 0 < dist_to_sup < 0.005:  # Within 0.5% of support
                    put_score -= 1.0

        # ── Max pain mean reversion (from research: works in low-vol, non-trending regimes) ──
        # Price > 1% above max pain --> expect reversion down --> +0.5 put_score
        # Price > 1% below max pain --> expect reversion up --> +0.5 call_score
        # Only when VIX < 25 AND ADX < 35 (max pain unreliable in strong trends / high vol)
        if ma:
            max_pain = getattr(ma, "max_pain", 0.0) or 0.0
            if max_pain > 0 and spot > 0 and vix < 25 and adx < 35:
                max_pain_dist_pct = (spot - max_pain) / max_pain
                if max_pain_dist_pct > 0.01:  # > 1% above max pain
                    put_score += 0.5
                elif max_pain_dist_pct < -0.01:  # > 1% below max pain
                    call_score += 0.5

        # ── VIX > 25 contrarian signal (from research: 75% bounce chance in 4-6 weeks) ──
        if vix > 35:
            call_score += 1.0  # Panic VIX --> extreme contrarian buy
        elif vix > 25:
            call_score += 0.5  # Elevated VIX --> contrarian buy

        # ── Closing hour bias (from research: 2:30-3:30 PM = +0.21% avg, 57% green) ──
        # bar_idx >= 63 corresponds to 2:30 PM (63 bars x 5 min = 315 min from 9:15 AM)
        if bar_idx >= 63:
            call_score += 0.3

        # ── Consecutive DOWN day pattern (from research: 3+ down --> 57.9% UP probability) ──
        if self._consecutive_down_days >= 3:
            call_score += 0.5

        # ── FII/DII institutional flow scoring (from research: correlation +0.7 to +0.8) ──
        if ma:
            fii = getattr(ma, "fii_net", 0.0) or 0.0
            dii = getattr(ma, "dii_net", 0.0) or 0.0
            if fii < -5000:
                put_score += 0.3  # Heavy FII selling pressure
            if fii > 0 and dii > 0:
                call_score += 0.3  # Both FII and DII buying

        # ── VWAP mean reversion signal (from research: 60% win rate, 1:1 R:R) ──
        # Only in range-bound markets (ADX < 25) — mean reversion fails in trends
        vwap = ind.get("vwap", spot)
        if vwap > 0 and adx_val < 25:
            vwap_dist_pct = (spot - vwap) / vwap
            if vwap_dist_pct < -0.015:
                call_score += 0.8  # Price > 1.5% below VWAP --> expect reversion up
            elif vwap_dist_pct > 0.015:
                put_score += 0.8   # Price > 1.5% above VWAP --> expect reversion down

        # ── Crude oil correlation (from research: India macro stress above $100/bbl) ──
        if ma:
            crude = getattr(ma, "crude_oil_price", 0.0) or 0.0
            if crude > 120:
                put_score += 1.0  # Severe macro stress
            elif crude > 100:
                put_score += 0.5  # Moderate macro stress

        if put_score >= cfg["put_score_min"] and put_score > call_score:
            conf = min(1.0, put_score / 18.0)
            return "BUY_PUT", conf
        elif call_score >= cfg["call_score_min"] and call_score > put_score:
            conf = min(1.0, call_score / 18.0)
            return "BUY_CALL", conf
        return None, 0

    def _passes_confluence(self, action: str, conf: float, ind: dict,
                           bar_idx: int, is_expiry: bool,
                           current_spot: float = 0.0,
                           oi_support: float = 0.0, oi_resistance: float = 0.0,
                           iv_percentile: float = 50.0) -> bool:
        """V14 confluence filter — delegates to shared engine."""
        return _shared_passes_confluence(
            action, conf, ind, bar_idx, V14_CONFIG,
            current_spot=current_spot,
            oi_support=oi_support,
            oi_resistance=oi_resistance,
            prev_close=self._prev_close,
            day_open=self._day_open,
            iv_percentile=iv_percentile,
        )

    def _passes_confluence_LEGACY(self, action: str, conf: float, ind: dict,
                           bar_idx: int, is_expiry: bool,
                           current_spot: float = 0.0,
                           oi_support: float = 0.0, oi_resistance: float = 0.0) -> bool:
        """LEGACY: Original confluence check (kept for reference)."""
        cfg = V14_CONFIG

        if conf < cfg["min_confidence_filter"]:
            return False

        # VWAP filter — use real-time spot price, not stale bar close
        if cfg["use_vwap_filter"]:
            vwap = ind.get("vwap", ind["close"])
            price = current_spot if current_spot > 0 else ind["close"]
            if action == "BUY_CALL" and price <= vwap:
                return False
            if action == "BUY_PUT" and price >= vwap:
                return False

        # ── Standard deviation entry filter (from research: bell curve) ──
        # Only enter when price has deviated > 1σ from today's mean
        # This ensures we're trading meaningful moves, not noise
        if cfg.get("use_stddev_filter"):
            zscore = ind.get("price_zscore", 0)
            threshold = cfg.get("stddev_entry_threshold", 1.0)
            if abs(zscore) < threshold:
                return False  # Price is within normal range — no edge

        # RSI hard gate
        if cfg["use_rsi_hard_gate"]:
            rsi = ind["rsi"]
            if action == "BUY_CALL" and rsi < cfg["rsi_call_min"]:
                return False
            if action == "BUY_PUT" and rsi > cfg["rsi_put_max"]:
                return False

        # ── RSI CALL kill zone (NEW: blocks CALLs in RSI 60-80 zone) ──
        # Analysis: RSI 60-80 CALLs had 23.3% WR and -Rs722K loss — chasing extended moves
        if action == "BUY_CALL":
            rsi = ind["rsi"]
            if rsi > cfg.get("rsi_call_kill_ceiling", 60):
                return False

        # Squeeze filter
        if cfg["use_squeeze_filter"] and ind.get("squeeze_on", False):
            return False

        # ── Gap reversal filter (NEW: prevents counter-trend entries after gap days) ──
        if cfg.get("gap_reversal_filter") and self._prev_close > 0 and self._day_open > 0:
            gap_pct = (self._day_open - self._prev_close) / self._prev_close
            gap_threshold = cfg.get("gap_threshold_pct", 0.004)
            price = current_spot if current_spot > 0 else ind["close"]

            if abs(gap_pct) >= gap_threshold:
                # Gap DOWN (negative gap) but price has reversed UP above prev close
                if gap_pct < 0 and price > self._prev_close and action == "BUY_PUT":
                    return False  # Don't buy puts into a gap-down reversal
                # Gap UP (positive gap) but price has reversed DOWN below prev close
                if gap_pct > 0 and price < self._prev_close and action == "BUY_CALL":
                    return False  # Don't buy calls into a gap-up reversal

        # ── OI Support/Resistance proximity filter (NEW: from research) ──
        # Block CALL entries near strong OI resistance (price likely to stall)
        # Block PUT entries near strong OI support (price likely to bounce)
        if cfg.get("use_oi_levels") and oi_support > 0 and oi_resistance > 0:
            price = current_spot if current_spot > 0 else ind["close"]
            proximity = cfg.get("oi_proximity_pct", 0.003)
            if action == "BUY_CALL" and oi_resistance > 0:
                dist_to_resistance = (oi_resistance - price) / price
                if 0 < dist_to_resistance < proximity:
                    return False  # Too close to OI resistance wall
            if action == "BUY_PUT" and oi_support > 0:
                dist_to_support = (price - oi_support) / price
                if 0 < dist_to_support < proximity:
                    return False  # Too close to OI support wall

        # Block CALL in 4th hour
        if cfg["block_call_4th_hour"] and action == "BUY_CALL":
            if 45 <= bar_idx < 60:  # 225-300 min in 5-min bars
                return False

        # Late entry block
        if bar_idx > cfg["block_late_entries"]:
            return False

        # Avoid windows
        for s, e in cfg["avoid_windows_bars"]:
            if s <= bar_idx < e:
                return False

        return True

    def _get_lots(self, action: str, conf: float, vix: float, rsi: float,
                  is_expiry: bool, regime: dict, daily_loss_pct: float = 0.0,
                  iv_percentile: float = 50.0, atr: float = 0.0) -> int:
        """V14 lot sizing — delegates to shared engine with broker constraints."""
        cfg = V14_CONFIG.copy()
        # Pass ATR through for ATR-normalized sizing (R5)
        if cfg.get("use_atr_sizing") and atr > 0:
            cfg["_current_atr"] = atr
        # Pass streak sizing data (V15)
        if cfg.get("use_streak_sizing"):
            cfg["_recent_losses"] = self._consecutive_losses
        lots = _shared_compute_lots(
            action, conf, vix, rsi, is_expiry,
            self._num_lots, cfg,
            regime=regime.get("regime", "neutral") if isinstance(regime, dict) else "neutral",
            regime_call_mult=regime.get("call_mult", 1.0) if isinstance(regime, dict) else 1.0,
            regime_put_mult=regime.get("put_mult", 1.0) if isinstance(regime, dict) else 1.0,
            iv_percentile=iv_percentile,
            daily_loss_pct=daily_loss_pct,
        )
        # Broker constraint: NSE freeze limit (NIFTY max 1800 qty / 65 = 27 lots)
        lots = min(lots, 27)
        return lots

    def _apply_lot_adjustments(self, lots: int, vix: float, entry_type: str, cfg: dict) -> int:
        """Apply VIX lot scaling, ORB boost, and lot cap (T20 improvements)."""
        # VIX-adaptive lot scaling (most impactful improvement: 3.13x --> 7.32x)
        if cfg.get("vix_lot_scaling"):
            if vix < 13:
                lots = max(1, int(lots * cfg.get("vix_below13_mult", 0.3)))
            elif 14 <= vix < 15:
                lots = max(1, int(lots * cfg.get("vix_14_15_mult", 0.5)))
            elif 15 <= vix < 17:
                lots = max(1, int(lots * cfg.get("vix_15_17_mult", 1.5)))
            elif vix >= 17:
                lots = max(1, int(lots * cfg.get("vix_17plus_mult", 2.0)))

        # ORB lot boost (ORB has 75% WR, PF 12.16)
        if entry_type and "orb" in entry_type:
            orb_mult = cfg.get("orb_lot_mult", 1.0)
            lots = max(1, int(lots * orb_mult))

        # Lot cap (33+ lots catastrophic: 5 trades lost Rs -4.96L)
        max_cap = cfg.get("max_lots_cap", 999)
        lots = min(lots, max_cap)

        # Broker constraint: NSE freeze limit
        lots = min(lots, 27)

        return lots

    def _get_lots_LEGACY(self, action: str, conf: float, vix: float, rsi: float,
                  is_expiry: bool, regime: dict, daily_loss_pct: float = 0.0,
                  iv_percentile: float = 50.0) -> int:
        """LEGACY: Original lot sizing (kept for reference)."""
        cfg = V14_CONFIG
        base_lots = self._num_lots
        combined_mult = 1.0

        # Direction bias
        if action == "BUY_PUT":
            combined_mult *= cfg["put_bias_lot_mult"]      # 1.3
        else:
            combined_mult *= cfg["call_bias_lot_mult"]      # 1.0

        # VIX regime (mutually exclusive — danger takes priority over sweet)
        if cfg["vix_danger_min"] <= vix <= cfg["vix_danger_max"]:
            combined_mult *= cfg["vix_danger_lot_mult"]     # 0.5
        elif cfg["vix_sweet_min"] <= vix <= cfg["vix_sweet_max"]:
            combined_mult *= cfg["vix_sweet_lot_mult"]      # 1.4

        # RSI zone
        if cfg["rsi_sweet_low"] <= rsi <= cfg["rsi_sweet_high"]:
            combined_mult *= cfg["rsi_sweet_lot_mult"]      # 1.5
        if cfg["rsi_danger_low"] <= rsi <= cfg["rsi_danger_high"]:
            combined_mult *= cfg["rsi_danger_lot_mult"]     # 0.5

        # Expiry day
        if is_expiry:
            combined_mult *= cfg["expiry_day_lot_mult"]     # 1.0

        # Regime (market_analysis regime — bullish/bearish/neutral)
        if regime.get("regime") == "bullish" and action == "BUY_PUT":
            combined_mult *= regime.get("put_mult", 1.0)
        elif regime.get("regime") == "bearish" and action == "BUY_CALL":
            combined_mult *= regime.get("call_mult", 1.0)

        # ── Confidence-based boost (NEW: reward high-quality signals) ──
        # Enhanced V14 filters produce higher WR (44.8% vs 36.6%)
        # Scale up when confidence is high, don't penalize normal confidence
        if conf >= 0.50:
            combined_mult *= 1.3    # High conviction — 30% lot boost
        elif conf >= 0.40:
            combined_mult *= 1.15   # Above average — 15% boost

        # ── IV Percentile lot scaling (from research) ──
        if cfg.get("use_iv_pctile_scaling") and iv_percentile > 0:
            if iv_percentile > cfg.get("iv_pctile_high", 75):
                combined_mult *= cfg.get("iv_pctile_high_mult", 0.7)
            elif iv_percentile < cfg.get("iv_pctile_low", 25):
                combined_mult *= cfg.get("iv_pctile_low_mult", 1.3)

        # ── Drawdown-based lot scaling ──
        if cfg.get("drawdown_lot_scale") and daily_loss_pct < 0:
            kill_switch_pct = 0.03
            loss_used_pct = abs(daily_loss_pct) / kill_switch_pct
            if loss_used_pct >= 0.75:
                combined_mult *= cfg.get("drawdown_75pct_mult", 0.3)
            elif loss_used_pct >= 0.50:
                combined_mult *= cfg.get("drawdown_50pct_mult", 0.7)

        # ── Floor: never let combined multiplier reduce below 0.5x base ──
        # Prevents cascading reductions from killing position size entirely
        combined_mult = max(0.5, combined_mult)

        lots = max(1, int(base_lots * combined_mult))

        if combined_mult < 0.8 or combined_mult > 1.3:
            logger.info("V14 LOT SIZING: base=%d × %.2f = %d lots | conf=%.2f vix=%.1f rsi=%.0f iv=%.0f dd=%.1f%%",
                        base_lots, combined_mult, lots, conf, vix, rsi, iv_percentile, daily_loss_pct * 100)

        return lots

    # ─────────────────────────────────────────────────────────────
    # COMPOSITE ENTRY DETECTION (gap, ORB, S/R bounce, zero-to-hero)
    # From V9_HYBRID backtest: +Rs 170K from 27 additional trades
    # ─────────────────────────────────────────────────────────────

    def _detect_composite_entries(
        self, bar: dict, bar_idx: int, spot: float, vix: float,
        market_analysis: Optional["MarketAnalysis"] = None,
    ) -> list[tuple]:
        """Detect composite entries — delegates to shared engine."""
        ma = market_analysis
        prev_spot = self._bar_history[-2]["close"] if len(self._bar_history) >= 2 else spot
        bias_val = "neutral"
        if ma:
            bias = getattr(ma, "market_bias", None) or getattr(ma, "bias", None)
            if bias:
                bias_val = bias.value if hasattr(bias, "value") else str(bias)

        # Update ORB tracking
        if bar_idx == 0:
            self._orb_high = bar.get("high", spot)
            self._orb_low = bar.get("low", spot)
        elif bar_idx == 1:
            self._orb_high = max(self._orb_high, bar.get("high", spot))
            self._orb_low = min(self._orb_low, bar.get("low", spot))

        # Update S/R from market analysis
        if ma:
            support = getattr(ma, "support", 0) or getattr(ma, "pivot_s1", 0) or 0
            resistance = getattr(ma, "resistance", 0) or getattr(ma, "pivot_r1", 0) or 0
            if support > 0:
                self._support = support
            if resistance > 0:
                self._resistance = resistance

        signals = _shared_detect_composite(
            bar, bar_idx, spot, vix, V14_CONFIG,
            prev_close=self._prev_close,
            gap_detected=self._gap_detected,
            orb_high=self._orb_high,
            orb_low=self._orb_low,
            support=self._support,
            resistance=self._resistance,
            prev_spot=prev_spot,
            market_bias=bias_val,
        )

        if bar_idx == 0:
            self._gap_detected = True

        return signals

    def _detect_composite_entries_LEGACY(
        self, bar: dict, bar_idx: int, spot: float, vix: float,
        market_analysis: Optional["MarketAnalysis"] = None,
    ) -> list[tuple]:
        """LEGACY: Original composite entry detection (kept for reference)."""
        signals = []
        cfg = V14_CONFIG

        # ── 1. GAP ENTRY (bar 0) ──
        if bar_idx == 0 and self._prev_close > 0 and not self._gap_detected:
            self._gap_detected = True
            gap_pct = (spot - self._prev_close) / self._prev_close * 100
            self._gap_pct = gap_pct
            is_large_gap = abs(gap_pct) > 1.2

            if gap_pct < -0.3:
                if is_large_gap:
                    # Large gap down reverses 70% -> fade with CALL
                    conf = min(0.85, 0.65 + abs(gap_pct) * 0.05)
                    signals.append(("BUY_CALL", "gap_fade", conf, False))
                else:
                    # Normal gap down: momentum PUT
                    conf = min(0.90, 0.60 + abs(gap_pct) * 0.10)
                    signals.append(("BUY_PUT", "gap_entry", conf, False))
                # Zero-hero on medium gap downs (continuation zone)
                if -1.2 <= gap_pct < -0.5 and vix >= 13:
                    signals.append(("BUY_PUT", "gap_zero_hero", 0.70, True))
            elif gap_pct > 0.3:
                if is_large_gap:
                    conf = min(0.85, 0.65 + gap_pct * 0.05)
                    signals.append(("BUY_PUT", "gap_fade", conf, False))
                # Zero-hero on large gap ups
                if gap_pct > 1.0 and vix >= 13:
                    signals.append(("BUY_PUT", "gap_zero_hero", 0.65, True))

            if signals:
                logger.info("V14 GAP DETECTED: gap=%.2f%% | %d signals generated",
                            gap_pct, len(signals))

        # ── 2. ORB ENTRY (bars 1-2) ──
        if bar_idx == 0:
            self._orb_high = bar.get("high", spot)
            self._orb_low = bar.get("low", spot)
        elif bar_idx == 1:
            self._orb_high = max(self._orb_high, bar.get("high", spot))
            self._orb_low = min(self._orb_low, bar.get("low", spot))

        if bar_idx in (1, 2) and self._orb_high > 0:
            orb_range = self._orb_high - self._orb_low
            if orb_range > spot * 0.0015:  # Minimum 0.15% range
                above_sma50 = False
                if market_analysis:
                    above_sma50 = getattr(market_analysis, "ema_trend", "") == "BULLISH"

                if spot > self._orb_high:
                    conf = min(0.80, 0.55 + (spot - self._orb_high) / self._orb_high * 10)
                    if above_sma50 or vix < 14:
                        signals.append(("BUY_CALL", "orb_breakout_up", conf, False))
                elif spot < self._orb_low:
                    conf = min(0.80, 0.55 + (self._orb_low - spot) / self._orb_low * 10)
                    signals.append(("BUY_PUT", "orb_breakout_down", conf, False))

        # ── 3. S/R BOUNCE (bar 2+) ──
        if bar_idx >= 2 and market_analysis:
            # Get S/R levels from market analysis
            support = getattr(market_analysis, "support", 0) or 0
            resistance = getattr(market_analysis, "resistance", 0) or 0
            # Fallback to pivot-based estimates
            if support <= 0:
                support = getattr(market_analysis, "pivot_s1", 0) or 0
            if resistance <= 0:
                resistance = getattr(market_analysis, "pivot_r1", 0) or 0

            # Update cached S/R
            if support > 0:
                self._support = support
            if resistance > 0:
                self._resistance = resistance

            sr_dist = (self._resistance - self._support) if (self._support and self._resistance) else 0

            if sr_dist >= 150 and len(self._bar_history) >= 2:
                prev_spot = self._bar_history[-2]["close"]
                bias_val = "neutral"
                bias = getattr(market_analysis, "market_bias", None) or getattr(market_analysis, "bias", None)
                if bias:
                    bias_val = bias.value if hasattr(bias, "value") else str(bias)

                # Bounce off support -> BUY_CALL
                if self._support and abs(spot - self._support) / spot < 0.003:
                    if spot > prev_spot and bias_val in ("bullish", "strong_bullish", "neutral",
                                                          "BULLISH", "STRONGLY_BULLISH", "NEUTRAL"):
                        sr_conf = 0.65 if "bullish" in bias_val.lower() else 0.55
                        signals.append(("BUY_CALL", "sr_bounce_support", sr_conf, False))

                # Rejection at resistance -> BUY_PUT (backtest: 44.4% WR, +Rs 158K)
                if self._resistance and abs(spot - self._resistance) / spot < 0.003:
                    if spot < prev_spot and "strong_bullish" not in bias_val.lower():
                        sr_conf = 0.75 if "bearish" in bias_val.lower() else 0.70
                        signals.append(("BUY_PUT", "sr_bounce_resistance", sr_conf, False))

        return signals

    def generate_signal(
        self,
        bar: dict,
        bar_idx: int,
        option_chain: Optional[dict] = None,
        market_analysis: Optional["MarketAnalysis"] = None,
    ) -> Optional[TradeSignal]:
        """V14 signal generation: V8 scoring + confluence filters."""

        # Day reset
        # Bars from broker (historical) and the live tick aggregator both
        # store the bar timestamp under "time"/"timestamp", not "date".
        # Fall back to the timestamp's YYYY-MM-DD prefix so today_date is
        # actually populated -- otherwise VWAP today-anchoring (and the
        # daily reset) silently degrade to "all bars in history".
        today_str = bar.get("date") or str(
            bar.get("time") or bar.get("timestamp") or ""
        )[:10]
        if today_str and today_str != self._today_date:
            # ── Track consecutive down days before resetting ──
            # Previous day was "down" if its close < open
            if self._prev_day_close > 0 and self._prev_day_open > 0:
                if self._prev_day_close < self._prev_day_open:
                    self._consecutive_down_days += 1
                else:
                    self._consecutive_down_days = 0
            # Save today's open/close as "previous day" for tomorrow's check
            if self._bar_history:
                self._prev_day_open = self._day_open if self._day_open > 0 else self._bar_history[0]["open"]
                self._prev_day_close = self._bar_history[-1]["close"]

            self._reset_day()
            self._today_date = today_str
            self._day_open = bar.get("open", bar["close"])

        # Add bar (only completed bars, not mid-bar snapshots)
        # Mid-bar snapshots would corrupt indicator calculations (RSI, EMA, etc.)
        if not bar.get("_is_midbar"):
            self._bar_history.append(bar)
            if len(self._bar_history) > 500:
                self._bar_history = self._bar_history[-500:]
            # Update regime detector with new close price
            if self._regime_detector:
                self._regime_detector.update(bar["close"])

        spot = bar["close"]

        # ── AI BRAIN: FORCED EXIT CHECK ──
        # If AI recommends EXIT_ALL with high conviction, close all positions immediately.
        # This catches macro events (flash crash, RBI surprise) that indicators react to slowly.
        ai = self._get_ai_brain()
        ai_action = ai.get("recommended_action", "").upper()
        ai_conviction = ai.get("conviction", "low").lower()
        ai_risk = ai.get("risk_assessment", "moderate").lower()

        if (ai_action == "EXIT_ALL" and ai_conviction == "high"
                and self._open_positions):
            logger.warning(
                "AI BRAIN EXIT_ALL: High conviction — closing %d open position(s) | "
                "risk=%s | reason=%s",
                len(self._open_positions), ai_risk,
                ai.get("one_liner", "AI recommends immediate exit"),
            )
            self._ai_influenced_count += 1
            # Exit the first open position (one per bar cycle, orchestrator will re-call)
            pos = self._open_positions[0]
            self._open_positions.remove(pos)
            self._last_exit_bar = bar_idx
            action = pos["action"]
            leg = OrderLeg(
                symbol=pos.get("symbol", ""),
                side="SELL",
                qty=pos["qty"],
                option_type=pos["opt_type"],
                strike=pos["strike"],
            )
            return TradeSignal(
                strategy=self.name,
                action="EXIT",
                confidence=0.9,
                underlying_price=spot,
                reasoning=f"ai_brain_exit_all ({ai.get('one_liner', 'forced exit')})",
                legs=[leg],
            )

        # ── CHECK EXITS FIRST (shared engine) ──
        # Determine day of week for theta exit
        _dow = -1
        if self._today_date:
            try:
                from datetime import datetime as _dt
                _dow = _dt.strptime(self._today_date, "%Y-%m-%d").weekday()
            except (ValueError, TypeError):
                pass

        indicators_for_exit = self._compute_indicators()

        for pos in list(self._open_positions):
            # Update best favorable price BEFORE exit check
            if pos["action"] == "BUY_CALL" and spot > pos["best_fav"]:
                pos["best_fav"] = spot
            elif pos["action"] == "BUY_PUT" and spot < pos["best_fav"]:
                pos["best_fav"] = spot

            # ── Use shared engine for exit decision ──
            exit_reason = _shared_evaluate_exit(
                pos, bar_idx, spot, indicators_for_exit or {}, V14_CONFIG,
                day_of_week=_dow,
            )

            if exit_reason and exit_reason.startswith("zero_hero"):
                logger.info("V14 Z2H EXIT: %s | reason=%s | bars=%d",
                            pos["action"], exit_reason, bar_idx - pos["entry_bar"])

            # Legacy exit_reason block replaced above. Keeping hard SL as live-only safety:
            action = pos["action"]
            cfg = V14_CONFIG
            bars_held = bar_idx - pos["entry_bar"]
            if bars_held < 1:
                exit_reason = None

            # ── HARD STOP LOSS (live-only safety — checks delta-estimated premium) ──
            # Estimate current option premium loss from spot movement.
            # For ATM options, delta ≈ 0.5; premium change ≈ |spot_change| * 0.5.
            # Zero-hero exits (stop/trail/time) are handled by the shared engine above.
            if not exit_reason:
                entry_premium = pos.get("entry_premium", 0)
                if entry_premium > 0 and cfg.get("hard_sl_pct", 0) > 0:
                    spot_change = spot - pos["entry_spot"]
                    if action == "BUY_PUT":
                        est_premium_change = -spot_change * 0.5  # PUT gains when spot drops
                    else:
                        est_premium_change = spot_change * 0.5   # CALL gains when spot rises
                    est_current_premium = entry_premium + est_premium_change
                    premium_loss_pct = (entry_premium - est_current_premium) / entry_premium
                    if premium_loss_pct >= cfg["hard_sl_pct"]:
                        exit_reason = f"hard_sl ({premium_loss_pct:.0%} loss)"
                        logger.warning(
                            "V14 HARD SL: %s %s%s | entry_prem=%.1f est_prem=%.1f loss=%.0f%%",
                            action, pos["strike"], pos["opt_type"],
                            entry_premium, est_current_premium, premium_loss_pct * 100,
                        )

            # ── THETA-AWARE EXIT (research: exit long options Monday 3 PM) ──
            # Tuesday is NIFTY weekly expiry — theta accelerates 40-60% on expiry day.
            # Exit profitable positions by Monday 3:00 PM to avoid overnight theta decay.
            if not exit_reason and cfg.get("theta_exit_enabled", True):
                theta_bar = cfg.get("theta_exit_monday_bar", 69)
                # Determine day of week: from bar timestamp, self._today_date, or datetime.now()
                day_of_week = None
                bar_ts = bar.get("timestamp") or bar.get("date")
                if bar_ts:
                    from datetime import datetime as _dt
                    try:
                        if isinstance(bar_ts, str):
                            # Try full datetime first, then date-only
                            for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
                                try:
                                    day_of_week = _dt.strptime(bar_ts, fmt).weekday()
                                    break
                                except ValueError:
                                    continue
                        elif hasattr(bar_ts, "weekday"):
                            day_of_week = bar_ts.weekday()
                    except Exception:
                        pass
                if day_of_week is None and self._today_date:
                    from datetime import datetime as _dt
                    try:
                        day_of_week = _dt.strptime(self._today_date, "%Y-%m-%d").weekday()
                    except (ValueError, TypeError):
                        pass
                if day_of_week is None:
                    from datetime import datetime as _dt
                    day_of_week = _dt.now().weekday()

                # Monday = 0; check if position is profitable (spot moved favorably)
                if day_of_week == 0 and bar_idx >= theta_bar:
                    is_profitable = False
                    if action == "BUY_PUT" and spot < pos["entry_spot"]:
                        is_profitable = True
                    elif action == "BUY_CALL" and spot > pos["entry_spot"]:
                        is_profitable = True
                    if is_profitable:
                        exit_reason = "theta_exit"
                        logger.info(
                            "V14 THETA EXIT: %s %s%s | Monday bar=%d >= %d | "
                            "entry_spot=%.1f spot=%.1f (profitable, avoiding Tuesday theta decay)",
                            action, pos["strike"], pos["opt_type"],
                            bar_idx, theta_bar, pos["entry_spot"], spot,
                        )

            # ── Exit parameters — NO regime adjustments (Lean backtest: 8.9x) ──
            # Lean backtest proved regime trail/hold adjustments hurt returns:
            # - Wider trails block slot turnover (time_exit P&L: -4K vs +13.9L)
            # - Extended holds push exits into EOD premium decay zone
            # - Tighter volatile trails cut winners short during crash moves
            # Regime adjustments stay active for ENTRY scoring (score_mult, block_reversion).
            max_hold_put = cfg["max_hold_put"]
            max_hold_call = cfg["max_hold_call"]
            if not exit_reason and action == "BUY_PUT" and bars_held >= max_hold_put:
                exit_reason = "time_exit"
            elif not exit_reason and action == "BUY_CALL" and bars_held >= max_hold_call:
                exit_reason = "time_exit"

            # ── Trailing stop — ATR-based / Chandelier / fixed % fallback ──
            # Research: "0.3% trail = 0.35x ATR — stopped by noise. Min 1.0x ATR intraday."
            # ATR trails adapt to volatility; Chandelier uses highest-high/lowest-low anchor.
            # Trail stops lose money directly but free position slots for new entries.
            # Lean backtest: trail P&L -82K but freed slots for +1.39M time_exit profit.
            elif not exit_reason and action == "BUY_PUT" and bars_held >= cfg["min_hold_trail_put"]:
                min_profit_move = pos["entry_spot"] * cfg.get("trail_min_profit_pct", 0.003)
                has_profit = pos["entry_spot"] - pos["best_fav"] >= min_profit_move
                if has_profit:
                    if cfg.get("use_chandelier_exit"):
                        # Chandelier exit for PUTs: lowest_low + N*ATR
                        # If spot rises above this level, the downtrend is broken --> exit
                        ch_period = cfg.get("chandelier_period", 22)
                        ch_mult = cfg.get("chandelier_atr_mult", 3.0)
                        atr = self._current_indicators.get("atr", 100.0)
                        bars = self._bar_history
                        if len(bars) >= ch_period:
                            lowest_low = min(b["low"] for b in bars[-ch_period:])
                        else:
                            lowest_low = min(b["low"] for b in bars) if bars else pos["best_fav"]
                        chandelier_stop = lowest_low + ch_mult * atr
                        if spot > chandelier_stop:
                            exit_reason = "trail_stop_chandelier"
                            logger.info("V14 CHANDELIER PUT: spot=%.1f > stop=%.1f (low=%.1f + %.1f*ATR=%.1f)",
                                        spot, chandelier_stop, lowest_low, ch_mult, atr)
                    elif cfg.get("use_atr_trail"):
                        # ATR-based trail: trail_d = ATR * multiplier (ADX-adaptive)
                        atr = self._current_indicators.get("atr", 100.0)
                        adx = self._current_indicators.get("adx", 25.0)
                        if cfg.get("atr_trail_adx_adaptive"):
                            if adx < 20:
                                atr_mult = cfg.get("atr_trail_adx_low", 1.0)
                            elif adx <= 35:
                                atr_mult = cfg.get("atr_trail_adx_mid", 1.5)
                            else:
                                atr_mult = cfg.get("atr_trail_adx_high", 2.0)
                        else:
                            atr_mult = cfg.get("atr_trail_mult", 1.5)
                        trail_d = atr * atr_mult
                        if spot > pos["best_fav"] + trail_d:
                            exit_reason = "trail_stop"
                            logger.info("V14 ATR TRAIL PUT: spot=%.1f > best_fav=%.1f + %.1fx ATR(%.1f)=%.1f | ADX=%.0f",
                                        spot, pos["best_fav"], atr_mult, atr, trail_d, adx)
                    else:
                        # Fallback: fixed percentage trail (original behavior)
                        trail_d = pos["entry_spot"] * cfg["trail_pct_put"]
                        if spot > pos["best_fav"] + trail_d:
                            exit_reason = "trail_stop"
            elif not exit_reason and action == "BUY_CALL" and bars_held >= cfg["min_hold_trail_call"]:
                min_profit_move = pos["entry_spot"] * cfg.get("trail_min_profit_pct", 0.003)
                has_profit = pos["best_fav"] - pos["entry_spot"] >= min_profit_move
                if has_profit:
                    if cfg.get("use_chandelier_exit"):
                        # Chandelier exit for CALLs: highest_high - N*ATR
                        # If spot drops below this level, the uptrend is broken --> exit
                        ch_period = cfg.get("chandelier_period", 22)
                        ch_mult = cfg.get("chandelier_atr_mult", 3.0)
                        atr = self._current_indicators.get("atr", 100.0)
                        bars = self._bar_history
                        if len(bars) >= ch_period:
                            highest_high = max(b["high"] for b in bars[-ch_period:])
                        else:
                            highest_high = max(b["high"] for b in bars) if bars else pos["best_fav"]
                        chandelier_stop = highest_high - ch_mult * atr
                        if spot < chandelier_stop:
                            exit_reason = "trail_stop_chandelier"
                            logger.info("V14 CHANDELIER CALL: spot=%.1f < stop=%.1f (high=%.1f - %.1f*ATR=%.1f)",
                                        spot, chandelier_stop, highest_high, ch_mult, atr)
                    elif cfg.get("use_atr_trail"):
                        # ATR-based trail: trail_d = ATR * multiplier (ADX-adaptive)
                        atr = self._current_indicators.get("atr", 100.0)
                        adx = self._current_indicators.get("adx", 25.0)
                        if cfg.get("atr_trail_adx_adaptive"):
                            if adx < 20:
                                atr_mult = cfg.get("atr_trail_adx_low", 1.0)
                            elif adx <= 35:
                                atr_mult = cfg.get("atr_trail_adx_mid", 1.5)
                            else:
                                atr_mult = cfg.get("atr_trail_adx_high", 2.0)
                        else:
                            atr_mult = cfg.get("atr_trail_mult", 1.5)
                        trail_d = atr * atr_mult
                        if spot < pos["best_fav"] - trail_d:
                            exit_reason = "trail_stop"
                            logger.info("V14 ATR TRAIL CALL: spot=%.1f < best_fav=%.1f - %.1fx ATR(%.1f)=%.1f | ADX=%.0f",
                                        spot, pos["best_fav"], atr_mult, atr, trail_d, adx)
                    else:
                        # Fallback: fixed percentage trail (original behavior)
                        trail_d = pos["entry_spot"] * cfg["trail_pct_call"]
                        if spot < pos["best_fav"] - trail_d:
                            exit_reason = "trail_stop"

            # EOD close
            if not exit_reason and bar_idx >= 72:  # 72 bars x 5 min = 360 min
                exit_reason = "eod_close"

            if exit_reason:
                self._open_positions.remove(pos)
                self._last_exit_bar = bar_idx

                # ── Track trade result for consecutive loss detection ──
                # Estimate P&L for this trade
                spot_change = spot - pos["entry_spot"]
                if action == "BUY_PUT":
                    est_pnl = -spot_change * 0.5 * pos["qty"]
                else:
                    est_pnl = spot_change * 0.5 * pos["qty"]
                self._daily_realised_pnl += est_pnl
                self._trade_results.append({
                    "pnl": est_pnl, "bar": bar_idx, "action": action,
                    "reason": exit_reason,
                })

                # Update consecutive loss counter
                if est_pnl < 0:
                    self._consecutive_losses += 1
                    if self._consecutive_losses >= cfg.get("max_consecutive_losses", 2):
                        pause_bars = cfg.get("loss_cooldown_bars", 12)
                        self._loss_pause_until_bar = bar_idx + pause_bars
                        logger.warning(
                            "V14 LOSS THROTTLE: %d consecutive losses — pausing entries until bar %d (%.0f min)",
                            self._consecutive_losses, self._loss_pause_until_bar, pause_bars * 5,
                        )
                else:
                    self._consecutive_losses = 0  # Reset on win

                logger.info("V14 EXIT: %s %s%s | reason=%s | bars_held=%d | est_pnl=%.0f | consec_losses=%d",
                            action, pos["strike"], pos["opt_type"],
                            exit_reason, bars_held, est_pnl, self._consecutive_losses)

                # Return exit signal
                leg = OrderLeg(
                    symbol=pos.get("symbol", ""),
                    side="SELL",
                    qty=pos["qty"],
                    option_type=pos["opt_type"],
                    strike=pos["strike"],
                )
                return TradeSignal(
                    strategy=self.name,
                    action="EXIT",
                    confidence=0.5,
                    underlying_price=spot,
                    reasoning=exit_reason,
                    legs=[leg],
                )
            else:
                # Update best favorable price
                if action == "BUY_CALL" and spot > pos["best_fav"]:
                    pos["best_fav"] = spot
                elif action == "BUY_PUT" and spot < pos["best_fav"]:
                    pos["best_fav"] = spot

        # ── ENTRY LOGIC ──
        cfg = V14_CONFIG

        # ── Extract market microstructure data EARLY (must be before any vix usage) ──
        # FIX: vix was previously defined only after the bar_idx<3 branch, causing
        # UnboundLocalError on composite-entry path (bars 0-2). Define here so all
        # code paths have vix/pcr/iv_percentile/oi_* available.
        self._market_analysis = market_analysis
        vix = 14.0
        pcr = 0.0
        iv_percentile = 50.0
        oi_support = 0.0
        oi_resistance = 0.0
        if market_analysis:
            vix = getattr(market_analysis, "vix", 14.0) or 14.0
            pcr = getattr(market_analysis, "pcr", 0.0) or 0.0
            iv_percentile = getattr(market_analysis, "iv_percentile", 50.0) or 50.0
            oi_support = getattr(market_analysis, "oi_support", 0.0) or 0.0
            oi_resistance = getattr(market_analysis, "oi_resistance", 0.0) or 0.0

        # ── Day-of-week blocking (Wednesday: 23.5% WR, PF 0.38) ──
        avoid_days = cfg.get("avoid_days", [])
        if avoid_days and _dow in avoid_days:
            logger.debug("V14 SKIP: day_of_week=%d in avoid_days %s", _dow, avoid_days)
            return None

        # Guards — allow composite entries on bars 0-2 (gap/ORB), V8 needs bar >= 3
        # Composite entries detected first, V8 scoring requires enough bar history
        if bar_idx < 3:
            # Still check composite entries (gap on bar 0, ORB on bars 1-2)
            composite_entries = self._detect_composite_entries(bar, bar_idx, spot, vix, market_analysis)
            # Filter out zero-hero if disabled
            if cfg.get("disable_zero_hero", False):
                composite_entries = [c for c in composite_entries if not c[3]]
            if not composite_entries:
                logger.debug("V14 SKIP: bar_idx=%d < 3 (no composite signal)", bar_idx)
                return None
            # Have a composite entry — proceed with it (skip V8 scoring below)
            composite_entries.sort(key=lambda x: x[2], reverse=True)
            action_early, entry_type_early, conf_early, is_zh_early = composite_entries[0]

            # ── R5 PARITY FIX: Apply same gates as backtest for composites ──
            # VIX bounds (matches backtest line 340)
            if vix < cfg["vix_floor"] or vix > cfg["vix_ceil"]:
                return None
            # Full confluence check including PSAR filter (matches backtest line 344)
            indicators = self._compute_indicators()
            if indicators and not self._passes_confluence(
                action_early, conf_early, indicators, bar_idx, False,
                current_spot=spot,
            ):
                logger.debug("V14 SKIP: composite entry failed confluence (PSAR/VWAP/etc)")
                return None
            # Check same direction (matches backtest line 353)
            same_dir = [p for p in self._open_positions if p["action"] == action_early]
            if same_dir:
                return None
            # Build the signal
            is_expiry = False
            if bar.get("date"):
                from datetime import datetime
                try:
                    d = datetime.strptime(bar["date"], "%Y-%m-%d").date()
                    is_expiry = (d.weekday() == 1)
                except (ValueError, TypeError):
                    pass
            regime = {"regime": "neutral", "call_mult": 1.0, "put_mult": 1.0}
            if market_analysis and hasattr(market_analysis, "regime"):
                regime = market_analysis.regime or regime
            rsi_val = indicators.get("rsi", 50.0) if indicators else 50.0
            # ── R5 PARITY FIX: Pass ATR for ATR-normalized sizing (matches backtest) ──
            atr_val = indicators.get("atr", 0) if indicators else 0
            lots = self._get_lots(action_early, conf_early, vix, rsi_val, is_expiry, regime,
                                  atr=atr_val)
            if is_zh_early:
                lots = min(3, max(1, lots))  # Zero-hero: 1-3 lots max
            # Apply VIX lot scaling + ORB boost + lot cap
            lots = self._apply_lot_adjustments(lots, vix, entry_type_early, cfg)
            # Capital-based lot cap (option buying: premium * lot_size must fit capital)
            cpl = 200.0 * self.lot_size  # ATM option ~Rs 200/unit × 65 = Rs 13,000/lot
            if cpl > 0:
                max_aff = max(1, int(self.capital / cpl))
                if lots > max_aff:
                    logger.info("V14 EARLY LOT CAP: %d lots --> %d (capital=%.0f)", lots, max_aff, self.capital)
                    lots = max_aff
            qty = lots * self.lot_size
            atm = round(spot / STRIKE_INTERVAL) * STRIKE_INTERVAL
            opt_type = "PE" if action_early == "BUY_PUT" else "CE"
            if is_zh_early:
                # Zero-hero: 2 strikes deeper OTM
                if action_early == "BUY_PUT":
                    strike = atm - 200  # Deep OTM put
                else:
                    strike = atm + 200  # Deep OTM call
            else:
                strike = atm
            symbol = self.resolve_symbol(strike, opt_type, option_chain)
            pos = {
                "action": action_early, "entry_bar": bar_idx,
                "entry_spot": spot, "strike": strike,
                "opt_type": opt_type, "lots": lots, "qty": qty,
                "best_fav": spot, "symbol": symbol,
                "confidence": conf_early,
                "entry_premium": 50.0,  # Estimate
                "entry_type": entry_type_early,
                "is_zero_hero": is_zh_early,
            }
            self._open_positions.append(pos)
            self._trades_today += 1
            logger.info("V14 COMPOSITE ENTRY: %s %s%s via %s | conf=%.2f | lots=%d | VIX=%.1f%s",
                        action_early, strike, opt_type, entry_type_early,
                        conf_early, lots, vix, " [ZERO-HERO]" if is_zh_early else "")
            leg = OrderLeg(symbol=symbol, side="BUY", qty=qty,
                          option_type=opt_type, strike=strike)
            # V17 product selection (composite/early entries are at bar_idx<3
            # so v17_btst_bar_min=30 will virtually always force 'MIS' here;
            # we still call the helper so config changes take effect uniformly).
            product = self._pick_product(
                action=action_early,
                indicators=indicators or {},
                bar_idx=bar_idx,
                spot=spot,
                vix=vix,
            )
            if product == "NRML":
                logger.info("V14 BTST CARRY (composite): %s %s%s | NRML", action_early, strike, opt_type)
            return TradeSignal(
                strategy=self.name, action=action_early,
                confidence=conf_early, underlying_price=spot,
                reasoning=f"v14_{entry_type_early}",
                legs=[leg],
                metadata={"product": product, "entry_type": entry_type_early},
            )
        if len(self._open_positions) >= cfg["max_concurrent"]:
            return None
        if self._trades_today >= cfg["max_trades_per_day"]:
            return None
        if bar_idx - self._last_exit_bar < cfg["cooldown_bars"]:
            return None

        # ── Entry window gating (R4_2Windows: 3.48x winner) ──
        # Only allow entries during proven profitable windows.
        # Composite entries (bar 0-2: gap, ORB) bypass this gate (handled above).
        entry_windows = cfg.get("entry_windows_bars")
        if entry_windows:
            in_window = any(s <= bar_idx <= e for s, e in entry_windows)
            if not in_window:
                logger.debug("V14 SKIP: bar_idx=%d outside entry windows %s", bar_idx, entry_windows)
                return None

        # ── Consecutive loss throttle (NEW) ──
        if bar_idx < self._loss_pause_until_bar:
            logger.info("V14 SKIP: loss throttle active until bar %d (now %d, %d consec losses)",
                        self._loss_pause_until_bar, bar_idx, self._consecutive_losses)
            return None

        # (vix/pcr/iv_percentile/oi_* already extracted above before bar_idx<3 branch)

        # VIX check
        if vix < cfg["vix_floor"] or vix > cfg["vix_ceil"]:
            logger.debug("V14 SKIP: VIX=%.1f outside [%d,%d]", vix, cfg["vix_floor"], cfg["vix_ceil"])
            return None

        # ── IV crush event check (research: "Don't buy options day before events") ──
        iv_crush_events = cfg.get("iv_crush_events", [])
        if iv_crush_events and self._today_date:
            from datetime import datetime as _dt, timedelta
            try:
                today = _dt.strptime(self._today_date, "%Y-%m-%d").date()
                lookback = cfg.get("iv_crush_lookback_days", 1)
                for ev_str in iv_crush_events:
                    ev_date = _dt.strptime(ev_str, "%Y-%m-%d").date()
                    days_until = (ev_date - today).days
                    if 0 <= days_until <= lookback:
                        logger.info(
                            "V14 IV_CRUSH_RISK: event %s approaching (%d days), skipping entry",
                            ev_str, days_until,
                        )
                        return None
            except (ValueError, TypeError):
                pass  # Malformed date — skip the check

        # Compute indicators
        indicators = self._compute_indicators()
        if indicators is None:
            logger.debug("V14 SKIP: Not enough bars for indicators (need 15, have %d)", len(self._bar_history))
            return None

        # Cache indicators so exit logic can access ATR/ADX for trailing stops
        self._current_indicators = indicators

        # Determine expiry (moved before scoring so max pain boost can use it)
        is_expiry = False
        if bar.get("date"):
            from datetime import datetime
            try:
                d = datetime.strptime(bar["date"], "%Y-%m-%d").date()
                is_expiry = (d.weekday() == 1)  # Tuesday (SEBI Nov 2025: weekly expiry moved from Thu-->Tue)
            except (ValueError, TypeError):
                pass

        # Score — now includes PCR sentiment + microstructure signals + OI change + expiry max pain
        action, conf = self._score_entry(indicators, vix, pcr=pcr, bar_idx=bar_idx, is_expiry=is_expiry)
        entry_type = "v8_indicator"
        is_zero_hero = False

        if action is None:
            # ── V8 scoring didn't fire — try composite entries (gap, ORB, S/R bounce, Z2H) ──
            composite_entries = self._detect_composite_entries(bar, bar_idx, spot, vix, market_analysis)
            if cfg.get("disable_zero_hero", False):
                composite_entries = [c for c in composite_entries if not c[3]]
            if composite_entries:
                # Pick the best entry (highest confidence)
                composite_entries.sort(key=lambda x: x[2], reverse=True)
                action, entry_type, conf, is_zero_hero = composite_entries[0]
                logger.info("V14 COMPOSITE_SIGNAL: %s via %s | conf=%.2f | spot=%.0f | VIX=%.1f%s",
                            action, entry_type, conf, spot, vix,
                            " [ZERO-HERO]" if is_zero_hero else "")
            else:
                logger.info("V14 NO_SIGNAL: bar=%d RSI=%.1f EMA9>21=%s squeeze=%s VWAP=%.1f close=%.1f | put_score/call_score below threshold",
                            bar_idx, indicators.get("rsi", 0), indicators.get("ema9_above_ema21"),
                            indicators.get("squeeze_on"), indicators.get("vwap", 0), indicators.get("close", 0))
                return None

        # ── Regime-aware adjustments (from research: auto-toggle strategy) ──
        # Store on self so exit logic can access trail_mult, max_hold_mult, block_reversion
        if self._regime_detector:
            regime_data = self._regime_detector.get_strategy_adjustments()
            self._regime_adj = regime_data.get("adjustments", {})
            # Apply regime score multiplier
            score_mult = self._regime_adj.get("score_mult", 1.0)
            if score_mult != 1.0:
                conf = min(1.0, conf * score_mult)
        regime_adj = self._regime_adj

        logger.info("V14 SCORED: %s conf=%.2f bar=%d RSI=%.1f squeeze=%s close=%.1f vwap=%.1f regime=%s",
                    action, conf, bar_idx, indicators.get("rsi", 0),
                    indicators.get("squeeze_on"), indicators.get("close", 0), indicators.get("vwap", 0),
                    self._regime_detector.current_regime.value if self._regime_detector else "N/A")

        # Confluence check — pass real-time spot, OI levels, IV percentile for VWAP + OI + IV gate filters
        if not self._passes_confluence(action, conf, indicators, bar_idx, is_expiry,
                                        current_spot=spot,
                                        oi_support=oi_support, oi_resistance=oi_resistance,
                                        iv_percentile=iv_percentile):
            # Log WHY confluence failed
            vwap = indicators.get("vwap", indicators["close"])
            rsi = indicators.get("rsi", 50)
            squeeze = indicators.get("squeeze_on", False)
            reasons = []
            if cfg.get("use_stddev_filter"):
                zscore = indicators.get("price_zscore", 0)
                if abs(zscore) < cfg.get("stddev_entry_threshold", 1.0):
                    reasons.append(f"stddev: |z|={abs(zscore):.2f}<{cfg.get('stddev_entry_threshold', 1.0)}")
            if conf < cfg["min_confidence_filter"]:
                reasons.append(f"conf={conf:.2f}<{cfg['min_confidence_filter']}")
            if cfg["use_vwap_filter"]:
                if action == "BUY_CALL" and spot <= vwap:
                    reasons.append(f"CALL but spot={spot:.1f}<=VWAP={vwap:.1f}")
                if action == "BUY_PUT" and spot >= vwap:
                    reasons.append(f"PUT but spot={spot:.1f}>=VWAP={vwap:.1f}")
            if cfg["use_rsi_hard_gate"]:
                if action == "BUY_CALL" and rsi < cfg["rsi_call_min"]:
                    reasons.append(f"CALL but RSI={rsi:.1f}<{cfg['rsi_call_min']}")
                if action == "BUY_PUT" and rsi > cfg["rsi_put_max"]:
                    reasons.append(f"PUT but RSI={rsi:.1f}>{cfg['rsi_put_max']}")
            if action == "BUY_CALL" and rsi > cfg.get("rsi_call_kill_ceiling", 60):
                reasons.append(f"CALL RSI kill zone: RSI={rsi:.1f}>{cfg.get('rsi_call_kill_ceiling', 60)}")
            if cfg["use_squeeze_filter"] and squeeze:
                reasons.append("squeeze_on")
            if cfg.get("gap_reversal_filter") and self._prev_close > 0 and self._day_open > 0:
                gap_pct = (self._day_open - self._prev_close) / self._prev_close
                if abs(gap_pct) >= cfg.get("gap_threshold_pct", 0.004):
                    price = spot
                    if gap_pct < 0 and price > self._prev_close and action == "BUY_PUT":
                        reasons.append(f"gap_reversal: gap_down={gap_pct:.2%} but price above prev_close")
                    if gap_pct > 0 and price < self._prev_close and action == "BUY_CALL":
                        reasons.append(f"gap_reversal: gap_up={gap_pct:.2%} but price below prev_close")
            if cfg.get("use_oi_levels") and oi_support > 0 and oi_resistance > 0:
                proximity = cfg.get("oi_proximity_pct", 0.003)
                if action == "BUY_CALL" and oi_resistance > 0:
                    dist = (oi_resistance - spot) / spot
                    if 0 < dist < proximity:
                        reasons.append(f"OI resistance wall: {oi_resistance:.0f} ({dist:.2%} away)")
                if action == "BUY_PUT" and oi_support > 0:
                    dist = (spot - oi_support) / spot
                    if 0 < dist < proximity:
                        reasons.append(f"OI support wall: {oi_support:.0f} ({dist:.2%} away)")
            for s, e in cfg["avoid_windows_bars"]:
                if s <= bar_idx < e:
                    reasons.append(f"avoid_window bars {s}-{e}")
            if bar_idx > cfg["block_late_entries"]:
                reasons.append(f"late_entry bar={bar_idx}>{cfg['block_late_entries']}")
            # ── AI BRAIN: CONFLUENCE OVERRIDE ──
            # If AI says override_confluence=true with high conviction AND the signal
            # direction aligns with AI's recommended action, allow the entry.
            # This catches strong trend moves that VWAP/RSI filters would block.
            ai_override = ai.get("override_confluence", False)
            ai_override_reason = ai.get("override_reason", "")
            ai_rec = ai_action  # Already extracted above (EXIT_ALL, BUY_CALL, etc.)

            # Only allow override if AI action matches signal direction
            ai_direction_match = (
                (action == "BUY_CALL" and ai_rec in ("BUY_CALL", "HOLD"))
                or (action == "BUY_PUT" and ai_rec in ("BUY_PUT", "HOLD"))
            )

            if ai_override and ai_conviction == "high" and ai_direction_match:
                logger.warning(
                    "AI BRAIN OVERRIDE: Allowing %s despite confluence fail | "
                    "AI conviction=%s reason=%s | blocked_by=%s",
                    action, ai_conviction, ai_override_reason,
                    " | ".join(reasons) if reasons else "unknown",
                )
                self._ai_influenced_count += 1
                # Fall through to entry logic (don't return None)
            else:
                logger.info("V14 CONFLUENCE_FAIL: %s | %s", action,
                            " | ".join(reasons) if reasons else "unknown")
                return None

        # Check same direction
        same_dir = [p for p in self._open_positions if p["action"] == action]
        if same_dir:
            return None

        # ── ML win-probability gate — DISABLED ──
        # The XGBoost model was trained on older data and blocks trades that the
        # V15 scoring engine (backtested at 16.04x) would take. The backtest did NOT
        # use this filter, so enabling it diverges live from backtest.
        # win_prob=0.09 on every signal = model is miscalibrated for current strategy.
        # TODO: retrain on V15 backtest data before re-enabling.
        if False and self._ml_filter and self._ml_filter.is_ready:
            rsi_val = indicators.get("rsi", 50)
            adx_val = indicators.get("adx", 20)
            bb_w = (indicators.get("bb_upper", 0) - indicators.get("bb_lower", 0)) / max(indicators.get("close", 1), 1) * 100
            atr_pct = indicators.get("atr", 50) / max(spot, 1) * 100
            stoch_k = indicators.get("stoch_k", 50)
            ema_bull = 1 if indicators.get("ema9_above_ema21") else 0
            st_bull = 1 if indicators.get("st_direction", 0) == 1 else 0
            macd_bull = 1 if indicators.get("macd_hist", 0) > 0 else 0
            ml_features = {
                # Original 26 features
                "is_put": 1 if action == "BUY_PUT" else 0,
                "confidence": conf,
                "vix": vix,
                "entry_minute": bar_idx,
                "lots": 2,
                "hour_of_entry": bar_idx // 12,
                "day_of_week": 0,
                "month_num": 1,
                "is_expiry": 1 if is_expiry else 0,
                "strike_distance_pct": 0.0,
                "premium_pct": 0.0,
                "rsi_at_entry": rsi_val,
                "bb_width_at_entry": bb_w,
                "atr_pct_at_entry": atr_pct,
                "ema9_above_21": ema_bull,
                "above_ema50": 0,
                "st_direction": indicators.get("st_direction", 0),
                "macd_hist_at_entry": indicators.get("macd_hist", 0),
                "adx_at_entry": adx_val,
                "vwap_proximity": (spot - indicators.get("vwap", spot)) / max(indicators.get("vwap", spot), 1) * 100,
                "squeeze_at_entry": 1 if indicators.get("squeeze_on") else 0,
                "stoch_k_at_entry": stoch_k,
                "above_sma20": 0,
                "above_sma50": 0,
                "price_vs_open_pct": (spot - self._day_open) / max(self._day_open, 1) * 100 if self._day_open > 0 else 0,
                "entry_type_encoded": 0,
                # V14 Enhanced features (14 new)
                "cci_at_entry": indicators.get("cci", 0),
                "williams_r_at_entry": indicators.get("williams_r", -50),
                "rsi_divergence": 1 if indicators.get("rsi_bullish_divergence") else 0,
                "pcr_extreme": 1 if vix > 18 else (-1 if vix < 12 else 0),
                "vix_zone": 3 if vix > 20 else (2 if vix > 16 else (1 if vix > 13 else 0)),
                "adx_regime": 3 if adx_val > 35 else (2 if adx_val > 25 else (1 if adx_val > 18 else 0)),
                "rsi_zone": 0 if rsi_val < 30 else (2 if rsi_val > 70 else 1),
                "minutes_in_day": bar_idx / 75.0,
                "momentum_alignment": ema_bull + st_bull + macd_bull,
                "volatility_ratio": atr_pct / max(bb_w, 0.01),
                "stoch_zone": 0 if stoch_k < 20 else (2 if stoch_k > 80 else 1),
                "time_to_close": max(0, 75 - bar_idx),
                "is_monday": 0,
                "is_expiry_eve": 0,
            }
            # Fill date fields if available
            if bar.get("date"):
                try:
                    from datetime import datetime as _dt
                    d = _dt.strptime(bar["date"], "%Y-%m-%d").date()
                    ml_features["day_of_week"] = d.weekday()
                    ml_features["month_num"] = d.month
                    ml_features["is_monday"] = 1 if d.weekday() == 0 else 0
                    ml_features["is_expiry_eve"] = 1 if d.weekday() == 0 else 0  # Mon before Tue expiry
                except (ValueError, TypeError):
                    pass
            elif self._today_date:
                try:
                    from datetime import date as _date
                    d = self._today_date if isinstance(self._today_date, _date) else _date.fromisoformat(str(self._today_date))
                    ml_features["day_of_week"] = d.weekday()
                    ml_features["month_num"] = d.month
                    ml_features["is_monday"] = 1 if d.weekday() == 0 else 0
                    ml_features["is_expiry_eve"] = 1 if d.weekday() == 0 else 0
                except (ValueError, TypeError):
                    pass

            win_prob = self._ml_filter.predict(ml_features)
            if win_prob < 0.30:
                logger.info(
                    "V14 ML_GATE: %s blocked | win_prob=%.2f < 0.30 | RSI=%.0f ADX=%.0f VIX=%.1f",
                    action, win_prob, indicators.get("rsi", 0), indicators.get("adx", 0), vix,
                )
                return None
            logger.info("V14 ML_PASS: %s | win_prob=%.2f", action, win_prob)

        # ── AI BRAIN: ENTRY INFLUENCE ──
        # After V8 scoring, confluence, and ML gate have all passed,
        # the AI brain provides advisory adjustments.
        # These are SOFT influences — they modify confidence/lots, not hard blocks
        # (except risk=extreme which is a safety gate).
        ai_lot_mult = 1.0  # Lot multiplier from AI
        ai_conf_adj = 0.0  # Confidence adjustment from AI

        if ai and ai_action:
            ai_sentiment = ai.get("sentiment", "neutral").lower()

            # 1. RISK GATE: Disabled — backtest (14.11x) does not include AI veto.
            #    Log only, do NOT block.
            if ai_risk == "extreme" and ai_conviction in ("high", "medium"):
                logger.warning(
                    "AI BRAIN RISK_WARN (no block): %s | risk=%s conviction=%s | %s",
                    action, ai_risk, ai_conviction,
                    ai.get("one_liner", "extreme risk detected"),
                )

            # 2. REDUCE SIZE: Scale down lots when AI recommends caution
            if ai_action == "REDUCE_SIZE":
                ai_lot_mult = 0.5
                logger.info(
                    "AI BRAIN REDUCE_SIZE: %s lots x0.5 | conviction=%s | %s",
                    action, ai_conviction, ai.get("one_liner", ""),
                )
                self._ai_influenced_count += 1

            # 3. SENTIMENT ALIGNMENT: Boost/penalize confidence
            #    When AI sentiment agrees with trade direction --> +10% confidence
            #    When AI sentiment conflicts --> -15% confidence (but don't block)
            signal_is_bullish = (action == "BUY_CALL")
            ai_is_bullish = (ai_sentiment == "bullish")
            ai_is_bearish = (ai_sentiment == "bearish")

            if signal_is_bullish and ai_is_bullish:
                ai_conf_adj = 0.10  # AI agrees with CALL --> boost
                logger.info("AI BRAIN ALIGN: CALL + bullish sentiment --> conf +10%%")
            elif signal_is_bullish and ai_is_bearish:
                ai_conf_adj = -0.15  # AI disagrees with CALL --> penalize
                logger.info("AI BRAIN CONFLICT: CALL vs bearish sentiment --> conf -15%%")
            elif not signal_is_bullish and ai_is_bearish:
                ai_conf_adj = 0.10  # AI agrees with PUT --> boost
                logger.info("AI BRAIN ALIGN: PUT + bearish sentiment --> conf +10%%")
            elif not signal_is_bullish and ai_is_bullish:
                ai_conf_adj = -0.15  # AI disagrees with PUT --> penalize
                logger.info("AI BRAIN CONFLICT: PUT vs bullish sentiment --> conf -15%%")

            # 4. HIGH CONVICTION BOOST: When AI strongly recommends same direction
            if ai_conviction == "high":
                if (signal_is_bullish and ai_action == "BUY_CALL") or \
                   (not signal_is_bullish and ai_action == "BUY_PUT"):
                    ai_lot_mult = max(ai_lot_mult, 1.2)  # 20% lot boost
                    logger.info(
                        "AI BRAIN HIGH_CONVICTION: %s lot boost x1.2 | AI=%s",
                        action, ai_action,
                    )
                    self._ai_influenced_count += 1

            # Apply AI confidence adjustment
            if ai_conf_adj != 0:
                old_conf = conf
                conf = max(0.1, min(1.0, conf + ai_conf_adj))
                if abs(ai_conf_adj) > 0:
                    self._ai_influenced_count += 1
                logger.info(
                    "AI BRAIN CONF_ADJ: %s conf %.2f --> %.2f (adj=%+.2f)",
                    action, old_conf, conf, ai_conf_adj,
                )

        # Regime
        regime = {"regime": "neutral", "call_mult": 1.0, "put_mult": 1.0}
        if market_analysis and hasattr(market_analysis, "regime"):
            regime = market_analysis.regime or regime

        # Lot sizing — pass daily loss pct for drawdown scaling + IV percentile
        daily_loss_pct = self._daily_realised_pnl / self.capital if self.capital > 0 else 0.0
        lots = self._get_lots(action, conf, vix, indicators["rsi"],
                              is_expiry, regime, daily_loss_pct=daily_loss_pct,
                              iv_percentile=iv_percentile,
                              atr=indicators.get("atr", 0))
        if lots <= 0:
            logger.info("V14 SKIP: drawdown scaling returned 0 lots — blocking entry")
            return None

        # ── AI BRAIN: LOT SCALING ──
        if ai_lot_mult != 1.0:
            old_lots = lots
            lots = max(1, int(lots * ai_lot_mult))
            logger.info("AI BRAIN LOT_SCALE: %d --> %d (x%.1f)", old_lots, lots, ai_lot_mult)

        # ── Apply VIX lot scaling + ORB boost + lot cap ──
        lots = self._apply_lot_adjustments(lots, vix, entry_type, cfg)

        # ── Capital-based lot cap for option buying ──
        # Premium × lot_size × lots must not exceed available capital.
        # ATM NIFTY options cost ~Rs 100-400/unit; use 200 as conservative default.
        # ATR measures spot movement, NOT option premium — don't use ATR here.
        est_premium = 200.0  # Conservative ATM option premium estimate
        cost_per_lot = est_premium * self.lot_size  # e.g., 200 * 65 = Rs 13,000/lot
        if cost_per_lot > 0:
            max_affordable_lots = max(1, int(self.capital / cost_per_lot))
            if lots > max_affordable_lots:
                logger.info("V14 LOT CAP: %d lots --> %d (capital=%.0f, cost/lot=%.0f)",
                            lots, max_affordable_lots, self.capital, cost_per_lot)
                lots = max_affordable_lots

        qty = lots * self.lot_size

        # ── STRIKE SELECTION ──
        atm = round(spot / STRIKE_INTERVAL) * STRIKE_INTERVAL
        opt_type = "PE" if action == "BUY_PUT" else "CE"
        strike_tag = ""  # For logging

        if is_zero_hero:
            # Zero-hero: deep OTM for lottery-style entries
            strike = atm - 200 if action == "BUY_PUT" else atm + 200
            lots = min(3, max(1, lots))  # Cap zero-hero at 3 lots
            strike_tag = " [ZH-OTM]"
        elif self._strike_selector and option_chain:
            # ── SMART STRIKE SELECTION ──
            # Score all strikes on delta, IV, OI, liquidity, bid-ask spread,
            # premium efficiency, and gamma to find the optimal entry.
            max_pain_val = getattr(market_analysis, "max_pain", 0) if market_analysis else 0
            is_exp = is_expiry

            # Calculate days to expiry for Greeks
            from datetime import date, timedelta
            today = date.today()
            # Estimate DTE from option chain expiry or use weekday heuristic
            dte = self._estimate_dte(today)

            selection = self._strike_selector.select(
                option_chain=option_chain,
                spot=spot,
                opt_type=opt_type,
                vix=vix,
                dte_days=dte,
                max_pain=max_pain_val,
                is_expiry_day=is_exp,
                market_analysis=market_analysis,
                strike_interval=STRIKE_INTERVAL,
            )
            strike = selection["strike"]
            est_entry_premium_smart = selection.get("premium", 0)
            strike_tag = f" [SMART: score={selection['score']:.2f} d={selection['delta']:.2f}]"

            if selection.get("fallback"):
                strike_tag = " [SMART->ATM fallback]"
        else:
            # Fallback: ATM (no smart selector or no option chain)
            strike = atm

        symbol = self.resolve_symbol(strike, opt_type, option_chain)

        # Estimate entry premium for hard stop loss tracking
        # Use smart selector premium if available, else ATR approximation
        if self._strike_selector and option_chain and not is_zero_hero:
            est_entry_premium = est_entry_premium_smart if est_entry_premium_smart > 0 else indicators.get("atr", 50.0) * 0.5
        else:
            est_entry_premium = indicators.get("atr", 50.0) * 0.5

        # Record position
        pos = {
            "action": action, "entry_bar": bar_idx,
            "entry_spot": spot, "strike": strike,
            "opt_type": opt_type, "lots": lots, "qty": qty,
            "best_fav": spot, "symbol": symbol,
            "confidence": conf,
            "entry_premium": est_entry_premium,  # For hard SL tracking
            "entry_type": entry_type,
            "is_zero_hero": is_zero_hero,
        }
        self._open_positions.append(pos)
        self._trades_today += 1

        max_pain_val = getattr(market_analysis, "max_pain", 0) if market_analysis else 0
        ai_tag = ""
        if ai_lot_mult != 1.0 or ai_conf_adj != 0:
            ai_tag = f" [AI: lots_x{ai_lot_mult:.1f} conf_adj={ai_conf_adj:+.2f}]"
        logger.info("V14 ENTRY: %s %s%s via %s | conf=%.2f | lots=%d | VIX=%.1f | RSI=%.0f | ADX=%.0f%s%s%s",
                     action, strike, opt_type, entry_type, conf, lots, vix, indicators["rsi"],
                     indicators.get("adx", 0),
                     " [ZERO-HERO]" if is_zero_hero else "",
                     ai_tag, strike_tag)

        # Build signal — single leg or debit spread
        legs = []

        if cfg.get("use_debit_spreads"):
            # ── Debit spread mode (from research: risk-defined trades) ──
            # BUY_CALL --> Bull Call Spread (buy ATM CE, sell OTM CE)
            # BUY_PUT --> Bear Put Spread (buy ATM PE, sell OTM PE)
            # Research: "execute BUY hedge leg FIRST, SELL core SECOND"
            # For debit spreads, BUY is the main leg (placed first)
            spread_width = cfg.get("spread_width_strikes", 2) * STRIKE_INTERVAL
            if action == "BUY_CALL":
                sell_strike = atm + spread_width   # Sell OTM call
            else:
                sell_strike = atm - spread_width   # Sell OTM put

            sell_symbol = self.resolve_symbol(sell_strike, opt_type, option_chain)

            # Leg 1: BUY ATM (main leg — placed first)
            legs.append(OrderLeg(
                symbol=symbol, side="BUY", qty=qty,
                option_type=opt_type, strike=strike,
            ))
            # Leg 2: SELL OTM (hedge leg — placed second, reduces cost)
            legs.append(OrderLeg(
                symbol=sell_symbol, side="SELL", qty=qty,
                option_type=opt_type, strike=sell_strike,
            ))

            reasoning = f"v14_{action.lower()}_spread_{int(strike)}_{int(sell_strike)}"
            logger.info("V14 SPREAD: %s buy=%s%s sell=%s%s | width=%d pts",
                        action, strike, opt_type, sell_strike, opt_type, spread_width)
        else:
            # Single-leg naked buy (original V14)
            legs.append(OrderLeg(
                symbol=symbol, side="BUY", qty=qty,
                option_type=opt_type, strike=strike,
            ))
            reasoning = f"v14_{action.lower()}"

        # V17 product selection: decide MIS (intraday) vs NRML (BTST carry)
        # based on indicator favorability for overnight gap continuation.
        product = self._pick_product(
            action=action,
            indicators=indicators,
            bar_idx=bar_idx,
            spot=spot,
            vix=vix,
        )
        if product == "NRML":
            logger.info(
                "V14 BTST CARRY: %s %s%s | bar=%d | ADX=%.0f | RSI=%.0f | VIX=%.1f → NRML",
                action, strike, opt_type, bar_idx,
                indicators.get("adx", 0), indicators.get("rsi", 0), vix,
            )

        return TradeSignal(
            strategy=self.name,
            action=action,
            confidence=conf,
            underlying_price=spot,
            reasoning=reasoning,
            legs=legs,
            metadata={"product": product, "entry_type": entry_type},
        )
