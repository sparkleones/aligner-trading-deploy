"""V4 Optimized Multi-Trade Live Agent — 2-5 trades/day.

V4 OPTIMIZATION (data-driven from 227-trade analysis):
  - REMOVED: sr_breakout entries (0% WR, -Rs 27K)
  - FIXED: trail_pct exit widened 0.3% → 0.5%, profit-gated, min 6 bars
  - FIXED: sr_combo_trail widened 0.3% → 0.6%, profit-gated
  - COOLDOWN_BARS: 2 → 0 (analysis confirmed optimal)
  - Composite windows: bars 3-5 + bars 8-10 (peak WR bars)
  - BTST: enhanced criteria for overnight holds (100% WR in V3)
  - Zero-hero: lowered gap threshold, VIX floor reduced
  - Entry types: gap, ORB, S/R bounce, composite
  - NO Monday/Wednesday skip — trades every market day

V3 baseline: +1638% return, Sharpe 15.53, PF 6.08, DD 4.22%
"""

import json
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Optional, TYPE_CHECKING

import numpy as np

from orchestrator.trade_signal import OrderLeg, TradeSignal
from orchestrator.strategy_agents.base_agent import BaseLiveAgent

if TYPE_CHECKING:
    from orchestrator.market_analyzer import MarketAnalysis

logger = logging.getLogger(__name__)

STRIKE_INTERVAL = 50
MAX_TRADES_PER_DAY = 5
MAX_CONCURRENT = 2
COOLDOWN_BARS = 5       # V5: wait 5 bars after exit (was 0 — re-entered same loser)
MIN_CONFIDENCE = 0.55   # V6: require >50% edge (was 0.25 — worse than coin flip)

# V6 Exit parameters — VIX-scaled, options-aware
# Backtest truth: trail_pct was -560K, time_exit was 99% WR +3.8M.
# Options need wider trails because premium moves are non-linear (gamma).
TRAIL_PCT = 0.010            # 1.0% trail (options need room to breathe)
PUT_MAX_HOLD_BARS = 330      # 5.5 hours — hold through theta acceleration (3:00 PM)
CALL_MAX_HOLD_BARS = 300     # 5 hours
SR_STOP_BUFFER = 0.004       # 0.4% buffer below entry before sr_stop fires

# Time-of-day guardrails
from datetime import time as dt_time_const
NO_ENTRY_BEFORE = dt_time_const(9, 30)   # Skip first 15 min (gap traps)
NO_ENTRY_AFTER = dt_time_const(14, 45)   # No new entries after 2:45 PM
EXPIRY_CLOSE_BY = dt_time_const(14, 45)  # Close all by 2:45 PM on expiry


class V3MultiTradeLiveAgent(BaseLiveAgent):
    """V4 Optimized Multi-Trade agent: 4 entry types, 2-5 trades/day, all days."""

    name = "learned_rules_v3"

    def __init__(self, capital: float = 200000.0, lot_size: int = 65):
        super().__init__(capital, lot_size)

        # V3 state
        self._trades_today: int = 0
        self._open_positions: list[dict] = []
        self._last_exit_bar: int = -10
        self._today_date: Optional[str] = None
        self._orb_high: float = 0
        self._orb_low: float = 0
        self._prev_spot: float = 0
        self._prev_close: float = 0  # Previous day close for gap calc

        # S/R levels (recomputed each day)
        self._support: float = 0
        self._resistance: float = 0

        # Pending entry (not yet confirmed by orchestrator)
        self._pending_entry: dict | None = None

        # Bar history for this session
        self._session_spots: list[float] = []

        # VIX smoothing (3-bar average)
        self._vix_history: list[float] = []

        # Load S/R rules
        data_dir = Path(__file__).parent.parent.parent / "data"
        self._sr_rules = {}
        sr_path = data_dir / "sr_rules.json"
        if sr_path.exists():
            with open(sr_path) as f:
                self._sr_rules = json.load(f)

        self._risk_budget_pct = 0.08

        logger.info(
            "V3 Multi-Trade Agent initialized | capital=%.0f lots=%d",
            capital, lot_size,
        )

    def _reset_day(self) -> None:
        """Reset daily state."""
        self._trades_today = 0
        self._open_positions.clear()
        self._pending_entry = None
        self._last_exit_bar = -10
        self._orb_high = 0
        self._orb_low = 0
        self._session_spots.clear()

    def generate_signal(
        self,
        bar: dict,
        bar_idx: int,
        option_chain: Optional[dict] = None,
        market_analysis: Optional["MarketAnalysis"] = None,
    ) -> Optional[TradeSignal]:
        """Generate V3 multi-trade signals from live bars."""
        if market_analysis is None:
            return None

        spot = bar.get("close", 0)
        if spot <= 0:
            return None

        # Day reset check
        today = datetime.now().strftime("%Y-%m-%d")
        if self._today_date != today:
            self._today_date = today
            self._reset_day()

        self._session_spots.append(spot)

        ma = market_analysis
        raw_vix = getattr(ma, "vix", 15.0)
        self._vix_history.append(raw_vix)
        recent_vix = self._vix_history[-3:]
        vix = sum(recent_vix) / len(recent_vix)  # 3-bar smoothed VIX
        self._bar_idx = bar_idx  # Store for tick-level exit checks

        # VIX guardrails: skip extreme regimes
        if vix < 12:
            return None  # Too much theta decay for option buyers
        if vix > 35:
            return None  # Too much gamma risk for small capital

        is_expiry = getattr(ma, "is_expiry_day", False)
        now_time = datetime.now().time()

        # ── EXPIRY DAY: force close all by 2:45 PM ──
        if is_expiry and self._open_positions and now_time >= EXPIRY_CLOSE_BY:
            pos = self._open_positions[0]
            self._open_positions.remove(pos)
            self._last_exit_bar = bar_idx
            if not self._open_positions:
                self._position_open = False
            logger.info("EXPIRY DAY: closing %s before settlement", pos["action"])
            return TradeSignal(
                strategy=self.name, action="CLOSE", confidence=0.99,
                underlying_price=spot, max_loss=0,
                metadata={
                    "exit_reason": "expiry_day_close",
                    "entry_type": pos.get("entry_type", ""),
                    "original_action": pos["action"],
                },
            )

        # ── CHECK EXITS FIRST ──
        exit_signal = self._check_exits(spot, bar_idx, vix, ma)
        if exit_signal is not None:
            return exit_signal

        # ── ENTRY GUARDS ──
        if self._trades_today >= MAX_TRADES_PER_DAY:
            return None
        if len(self._open_positions) >= MAX_CONCURRENT:
            return None
        if bar_idx - self._last_exit_bar < COOLDOWN_BARS:
            return None

        # Time-of-day filter: no entries in first 15 min or last 75 min
        if now_time < NO_ENTRY_BEFORE or now_time >= NO_ENTRY_AFTER:
            return None

        # On expiry day: no new entries after 1 PM, max 1 lot
        if is_expiry and now_time >= dt_time_const(13, 0):
            return None

        # Compute S/R levels
        self._update_sr(spot, ma)

        # Track ORB
        if bar_idx == 0:
            self._orb_high = bar.get("high", spot)
            self._orb_low = bar.get("low", spot)
        elif bar_idx == 1:
            self._orb_high = max(self._orb_high, bar.get("high", spot))
            self._orb_low = min(self._orb_low, bar.get("low", spot))

        # Detect entries
        entries = self._detect_entries(bar, bar_idx, spot, vix, ma)

        if not entries:
            self._prev_spot = spot
            return None

        # Pick the best entry (highest confidence)
        entries.sort(key=lambda x: x[2], reverse=True)
        action, entry_type, confidence, is_zero_hero = entries[0]

        # Don't open same direction as existing position
        for pos in self._open_positions:
            if pos["action"] == action:
                self._prev_spot = spot
                return None

        # Build and return the signal
        signal = self._build_trade_signal(
            spot, action, entry_type, confidence, is_zero_hero, vix, ma, option_chain
        )

        if signal is not None:
            # Store pending entry — will be confirmed via confirm_execution()
            # when the meta-agent approves and orchestrator executes.
            self._pending_entry = {
                "action": action,
                "entry_type": entry_type,
                "entry_bar": bar_idx,
                "entry_spot": spot,
                "best_fav": spot,
                "is_zero_hero": is_zero_hero,
                "sr_target_hit": False,
                "signal": signal,
            }
            logger.info(
                "V3 SIGNAL #%d | %s via %s | conf=%.2f | spot=%.0f | VIX=%.1f%s",
                self._trades_today + 1, action, entry_type, confidence, spot, vix,
                " [ZERO-HERO]" if is_zero_hero else "",
            )

        self._prev_spot = spot
        return signal

    def confirm_execution(self, signal: TradeSignal) -> None:
        """Called by orchestrator after trade is confirmed executed.

        Only now do we track the position internally — this prevents
        mismatch between agent and orchestrator state when meta-agent
        filters reject a signal.
        """
        if self._pending_entry:
            pending_action = self._pending_entry.get("action", "")
            signal_action = signal.action if signal else ""

            if self._pending_entry.get("signal") is signal or pending_action == signal_action:
                self._open_positions.append(self._pending_entry)
                self._trades_today += 1
                self._position_open = True
                logger.info(
                    "V3 ENTRY CONFIRMED #%d | %s via %s",
                    self._trades_today,
                    self._pending_entry["action"],
                    self._pending_entry["entry_type"],
                )
                self._pending_entry = None
                return

        # No pending entry or action doesn't match — reconstruct from signal
        # This handles timing races where pending was overwritten
        if signal and signal.action in ("BUY_PUT", "BUY_CALL"):
            entry_type = signal.metadata.get("entry_type", "unknown") if signal.metadata else "unknown"
            self._open_positions.append({
                "action": signal.action,
                "entry_type": entry_type,
                "entry_bar": len(self._session_spots) - 1,
                "entry_spot": signal.underlying_price,
                "best_fav": signal.underlying_price,
                "is_zero_hero": signal.metadata.get("is_zero_hero", False) if signal.metadata else False,
                "sr_target_hit": False,
                "signal": signal,
            })
            self._trades_today += 1
            self._position_open = True
            self._pending_entry = None
            logger.info(
                "V3 ENTRY CONFIRMED (reconstructed) #%d | %s via %s",
                self._trades_today, signal.action, entry_type,
            )

    # ─────────────────────────────────────────────────────────────
    # ENTRY DETECTION (5 types)
    # ─────────────────────────────────────────────────────────────

    def _detect_entries(
        self, bar: dict, bar_idx: int, spot: float, vix: float,
        ma: "MarketAnalysis",
    ) -> list[tuple]:
        """Detect all possible entries. Returns [(action, type, conf, is_zh), ...]"""
        signals = []
        support = self._support
        resistance = self._resistance

        above_sma50 = getattr(ma, "ema_trend", "") == "BULLISH"
        rsi = getattr(ma, "rsi", 50)

        # 1. GAP ENTRY (bar 0) — V4: gap size awareness
        # Large gaps (>1.2%) reverse 70%, medium (0.8-1.2%) continue 75%
        if bar_idx == 0 and self._prev_close > 0:
            gap_pct = (spot - self._prev_close) / self._prev_close * 100
            is_large_gap = abs(gap_pct) > 1.2

            if gap_pct < -0.3:
                if is_large_gap:
                    # Large gap down reverses 70% → fade with CALL
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
                    # Large gap up reverses 70% → fade with PUT
                    conf = min(0.85, 0.65 + gap_pct * 0.05)
                    signals.append(("BUY_PUT", "gap_fade", conf, False))
                elif gap_pct > 0.5 and above_sma50:
                    conf = min(0.85, 0.55 + gap_pct * 0.08)
                    signals.append(("BUY_CALL", "gap_entry", conf, False))

        # 2. ORB ENTRY (bar 1-2)
        if bar_idx in (1, 2) and self._orb_high > 0:
            orb_range = self._orb_high - self._orb_low
            if orb_range > spot * 0.0015:
                if spot > self._orb_high:
                    conf = min(0.80, 0.55 + (spot - self._orb_high) / self._orb_high * 10)
                    if above_sma50 or vix < 14:
                        signals.append(("BUY_CALL", "orb_breakout_up", conf, False))
                elif spot < self._orb_low:
                    conf = min(0.80, 0.55 + (self._orb_low - spot) / self._orb_low * 10)
                    signals.append(("BUY_PUT", "orb_breakout_down", conf, False))

        # 3. S/R BOUNCE (bar 2+) — bias-aware, requires confirmation
        bias = getattr(ma, "market_bias", None)
        bias_val = bias.value if bias else "neutral"

        # Require minimum S/R distance — 200+ points apart prevents noise triggers
        sr_dist = (resistance - support) if (support and resistance) else 0
        sr_valid = sr_dist >= 150  # At least 150 pts between S and R

        if bar_idx >= 2 and self._prev_spot > 0 and sr_valid:
            # Bounce off support (bullish) — skip in bearish (support will break)
            if support and abs(spot - support) / spot < 0.003:
                if spot > self._prev_spot and bias_val in ("bullish", "strong_bullish", "neutral"):
                    sr_call_conf = 0.65 if bias_val in ("bullish", "strong_bullish") else 0.55
                    signals.append(("BUY_CALL", "sr_bounce_support", sr_call_conf, False))

            # Rejection at resistance (bearish) — skip in strong bullish
            if resistance and abs(spot - resistance) / spot < 0.003:
                if spot < self._prev_spot and bias_val not in ("strong_bullish",):
                    # Boost confidence when bearish — rejection more reliable
                    sr_put_conf = 0.75 if bias_val in ("bearish", "strong_bearish") else 0.70
                    signals.append(("BUY_PUT", "sr_bounce_resistance", sr_put_conf, False))

        # 4. S/R BREAKOUT — REMOVED in V4 (0% WR, -Rs 27K in analysis)
        #    False breakdowns trap entries; bounces are far more reliable.

        # 5. V4 COMPOSITE — optimized windows (bars 3-5 + 8-10 in 15-min equiv)
        #    In 1-min bars: window 1 = bars 45-75, window 2 = bars 120-150
        #    For 15-min bars: window 1 = bars 3-5, window 2 = bars 8-10
        if bar_idx > 25:
            # 1-min bar mode
            put_window = (45 <= bar_idx <= 75) or (120 <= bar_idx <= 150)
            call_window = (60 <= bar_idx <= 120)
        else:
            # 15-min bar mode
            put_window = (3 <= bar_idx <= 5) or (8 <= bar_idx <= 10)
            call_window = (4 <= bar_idx <= 8)

        if put_window or call_window:
            scores = self._compute_composite(spot, vix, ma)
            best_action = max(scores, key=scores.get)
            total = sum(scores.values())
            conf = scores[best_action] / total if total > 0 else 0

            if conf >= MIN_CONFIDENCE:
                if best_action == "BUY_PUT" and put_window:
                    signals.append(("BUY_PUT", "composite", conf, False))
                elif best_action == "BUY_CALL" and call_window and vix < 12 and conf >= 0.75:
                    signals.append(("BUY_CALL", "composite", conf, False))

        return signals

    def _compute_composite(
        self, spot: float, vix: float, ma: "MarketAnalysis"
    ) -> dict[str, float]:
        """9-rule composite scoring (V3 adapted for live data)."""
        scores = {"BUY_CALL": 0.0, "BUY_PUT": 0.0}

        # Rule 1: VIX regime (weight 3-4)
        if vix < 12:
            scores["BUY_CALL"] += 3.0
        elif vix < 17:
            scores["BUY_PUT"] += 3.0
        elif vix < 25:
            scores["BUY_PUT"] += 3.5
        else:
            scores["BUY_PUT"] += 4.0

        # Rule 2: Trend (SMA50 proxy via EMA trend)
        ema_trend = getattr(ma, "ema_trend", "")
        if ema_trend == "BEARISH":
            scores["BUY_PUT"] += 2.0
        elif ema_trend == "BULLISH":
            scores["BUY_CALL"] += 2.0

        # Rule 3: SMA20 proxy
        bias = getattr(ma, "bias", None)
        if bias is not None:
            from orchestrator.market_analyzer import MarketBias
            if bias in (MarketBias.BEARISH, MarketBias.STRONGLY_BEARISH):
                scores["BUY_PUT"] += 1.0
            elif bias in (MarketBias.BULLISH, MarketBias.STRONGLY_BULLISH):
                scores["BUY_CALL"] += 1.0

        # Rule 4: RSI — oversold = bounce up (CALL), overbought = pullback (PUT)
        rsi = getattr(ma, "rsi", 50)
        if rsi < 30:
            scores["BUY_CALL"] += 1.5  # Oversold → expect bounce
        elif rsi > 70:
            scores["BUY_PUT"] += 1.5   # Overbought → expect pullback

        # Rule 5: Day of week (V4: corrected from 123-day deep analysis)
        # Mon=64% UP, Tue=71% DOWN, Wed=60% UP, Thu=61% DOWN, Fri=50/50→PUT edge
        dow = datetime.now().strftime("%A")
        dow_map = {"Monday": "BUY_CALL", "Tuesday": "BUY_PUT",
                   "Wednesday": "BUY_CALL", "Thursday": "BUY_PUT",
                   "Friday": "BUY_PUT"}
        d = dow_map.get(dow)
        if d:
            scores[d] += 0.5

        # Rule 6: VIX spike
        vix_spike = getattr(ma, "vix_spike", False)
        if vix_spike:
            scores["BUY_CALL"] += 2.0

        # Rule 7: S/R proximity
        if self._support and spot:
            dp = (spot - self._support) / spot * 100
            if 0 < dp < 1.0:
                scores["BUY_CALL"] += 1.0
            elif dp < 0:
                scores["BUY_PUT"] += 1.0
        if self._resistance and spot:
            dp = (self._resistance - spot) / spot * 100
            if 0 < dp < 1.0:
                scores["BUY_PUT"] += 1.0
            elif dp < 0:
                scores["BUY_CALL"] += 1.0

        # Rule 8: Momentum (last 5 bars)
        if len(self._session_spots) >= 6:
            mom = self._session_spots[-1] - self._session_spots[-6]
            if mom < -spot * 0.003:
                scores["BUY_PUT"] += 1.0
            elif mom > spot * 0.003:
                scores["BUY_CALL"] += 1.0

        # Rule 9: Multi-TF alignment (via supertrend)
        supertrend = getattr(ma, "supertrend_direction", "")
        if supertrend == "DOWN":
            best = max(scores, key=scores.get)
            if best == "BUY_CALL":
                scores["BUY_CALL"] *= 0.5
        elif supertrend == "UP":
            best = max(scores, key=scores.get)
            if best == "BUY_PUT":
                scores["BUY_PUT"] *= 0.5

        return scores

    # ─────────────────────────────────────────────────────────────
    # EXIT LOGIC
    # ─────────────────────────────────────────────────────────────

    def _check_exits(
        self, spot: float, bar_idx: int, vix: float,
        ma: "MarketAnalysis",
    ) -> Optional[TradeSignal]:
        """Check all open positions for exit conditions."""
        if not self._open_positions:
            return None

        to_close = []
        for pos in self._open_positions:
            action = pos["action"]
            entry_bar = pos["entry_bar"]
            bars_held = bar_idx - entry_bar
            best_fav = pos["best_fav"]
            entry_spot = pos["entry_spot"]
            is_zh = pos.get("is_zero_hero", False)

            if bars_held < 1:
                continue

            trail_dist = entry_spot * TRAIL_PCT  # 0.3% trail (V3 proven)
            exit_reason = None

            # Track best favorable move
            if action == "BUY_CALL" and spot > best_fav:
                pos["best_fav"] = spot
                best_fav = spot
            elif action == "BUY_PUT" and spot < best_fav:
                pos["best_fav"] = spot
                best_fav = spot

            # Zero-to-hero exits
            if is_zh:
                zh_trail = entry_spot * 0.008
                if action == "BUY_PUT":
                    move = (entry_spot - spot) / entry_spot
                else:
                    move = (spot - entry_spot) / entry_spot

                if move >= 0.02:  # ~3x on deep OTM
                    exit_reason = "zero_hero_target"
                elif move <= -0.008:  # stop loss
                    exit_reason = "zero_hero_stop"
                elif move >= 0.01:  # trail after good move
                    if action == "BUY_PUT" and spot > best_fav + zh_trail:
                        exit_reason = "zero_hero_trail"
                    elif action == "BUY_CALL" and spot < best_fav - zh_trail:
                        exit_reason = "zero_hero_trail"
                elif bars_held >= 150:  # ~2.5 hours in 1-min bars
                    exit_reason = "zero_hero_time"

            # Regular PUT exit: trail_pct (V3 proven loss filter)
            elif action == "BUY_PUT":
                if bars_held >= 45:  # Min 3 bars × 15 = 45 in 1-min
                    if spot > best_fav + trail_dist:
                        exit_reason = "trail_pct"
                if bars_held >= PUT_MAX_HOLD_BARS and not exit_reason:
                    exit_reason = "time_exit"

            # Regular CALL exit: sr_trail_combo (V3 proven)
            elif action == "BUY_CALL":
                # Use entry_spot-based stop instead of S/R (S/R is too close to noise)
                call_stop = entry_spot * (1 - SR_STOP_BUFFER)  # 0.2% below entry
                if not pos.get("sr_target_hit", False):
                    if self._resistance and spot >= self._resistance:
                        pos["sr_target_hit"] = True
                        pos["best_fav"] = spot
                    if spot < call_stop and bars_held >= 3:
                        exit_reason = "sr_stop"
                else:
                    if spot < best_fav - trail_dist:
                        exit_reason = "sr_combo_trail"
                if bars_held >= CALL_MAX_HOLD_BARS and not exit_reason:
                    exit_reason = "time_exit"

            if exit_reason:
                to_close.append((pos, exit_reason))

        if not to_close:
            return None

        # Close the first position that hit exit
        pos, reason = to_close[0]
        self._open_positions.remove(pos)
        self._last_exit_bar = bar_idx
        if not self._open_positions:
            self._position_open = False

        logger.info(
            "V3 EXIT | %s | reason=%s | entry=%.0f exit=%.0f | bars=%d%s",
            pos["action"], reason, pos["entry_spot"], spot,
            bar_idx - pos["entry_bar"],
            " [ZERO-HERO]" if pos.get("is_zero_hero") else "",
        )

        # Return CLOSE signal
        return TradeSignal(
            strategy=self.name,
            action="CLOSE",
            confidence=0.99,
            underlying_price=spot,
            max_loss=0,
            metadata={
                "exit_reason": reason,
                "entry_type": pos["entry_type"],
                "original_action": pos["action"],
            },
        )

    # ─────────────────────────────────────────────────────────────
    # S/R LEVELS
    # ─────────────────────────────────────────────────────────────

    def _update_sr(self, spot: float, ma: "MarketAnalysis") -> None:
        """Compute multi-method S/R levels."""
        support_cands = []
        resist_cands = []

        # Round numbers (weight 3.0)
        for level in range(int(spot // 500) * 500 - 1500,
                           int(spot // 500) * 500 + 2000, 500):
            if level < spot:
                support_cands.append((level, 3.0))
            elif level > spot:
                resist_cands.append((level, 3.0))
        for level in range(int(spot // 100) * 100 - 500,
                           int(spot // 100) * 100 + 600, 100):
            if level % 500 != 0:
                if level < spot:
                    support_cands.append((level, 1.5))
                elif level > spot:
                    resist_cands.append((level, 1.5))

        # VWAP / SMA as S/R
        vwap = getattr(ma, "vwap", None)
        if vwap and vwap > 0:
            if vwap < spot:
                support_cands.append((round(vwap / 50) * 50, 2.0))
            else:
                resist_cands.append((round(vwap / 50) * 50, 2.0))

        # Pre-loaded S/R rules
        for s in self._sr_rules.get("current_supports", []):
            if s < spot:
                support_cands.append((s, 2.0))
        for r in self._sr_rules.get("current_resistances", []):
            if r > spot:
                resist_cands.append((r, 2.0))

        # Swing points from full session — keep exact values, don't round
        if len(self._session_spots) >= 10:
            window = self._session_spots[-100:]
            for i in range(1, len(window) - 1):
                if window[i] > window[i-1] and window[i] > window[i+1]:
                    if window[i] > spot:
                        resist_cands.append((window[i], 1.5))  # Exact swing high
                if window[i] < window[i-1] and window[i] < window[i+1]:
                    if window[i] < spot:
                        support_cands.append((window[i], 1.5))  # Exact swing low

        # Pick best S/R
        if support_cands:
            support_cands.sort(key=lambda x: (spot - x[0], -x[1]))
            self._support = support_cands[0][0]
        else:
            self._support = round((spot * 0.99) / 50) * 50

        if resist_cands:
            resist_cands.sort(key=lambda x: (x[0] - spot, -x[1]))
            self._resistance = resist_cands[0][0]
        else:
            self._resistance = round((spot * 1.01) / 50) * 50

    # ─────────────────────────────────────────────────────────────
    # BUILD TRADE SIGNAL
    # ─────────────────────────────────────────────────────────────

    def _build_trade_signal(
        self,
        spot: float,
        action: str,
        entry_type: str,
        confidence: float,
        is_zero_hero: bool,
        vix: float,
        ma: "MarketAnalysis",
        option_chain: Optional[dict],
    ) -> TradeSignal:
        """Build option trade signal with strike, sizing, and legs."""
        opt_type = "CE" if action == "BUY_CALL" else "PE"
        atm = round(spot / STRIKE_INTERVAL) * STRIKE_INTERVAL

        # Strike selection
        if is_zero_hero:
            strike = atm + 200 if action == "BUY_CALL" else atm - 200
            # Zero-hero liquidity check: verify strike exists with LTP > 0
            if option_chain:
                zh_entry = option_chain.get(int(strike), {}).get(opt_type, {})
                zh_ltp = zh_entry.get("ltp", 0)
                if zh_ltp <= 0:
                    logger.info(
                        "ZERO-HERO SKIPPED: strike %d %s has no liquidity (LTP=%.2f)",
                        strike, opt_type, zh_ltp,
                    )
                    return None
            else:
                # No option chain available — skip zero-hero (can't verify liquidity)
                logger.info("ZERO-HERO SKIPPED: no option chain available for liquidity check")
                return None
        else:
            if action == "BUY_CALL":
                strike = atm + (-50 if vix < 12 else (100 if vix < 20 else 200))
            else:
                strike = atm - (0 if vix < 12 else 150)

        # Position sizing — VIX-aware SPAN margin estimation
        # For option BUYERS: high VIX = bigger moves = more opportunity
        # But also higher margin requirements per lot
        SPAN_MARGIN_EST = {12: 30000, 15: 35000, 20: 40000, 25: 50000, 30: 60000}
        span_per_lot = 60000  # default high
        for vix_threshold in sorted(SPAN_MARGIN_EST.keys()):
            if vix < vix_threshold:
                span_per_lot = SPAN_MARGIN_EST[vix_threshold]
                break

        if is_zero_hero:
            num_lots = 1
        else:
            available_margin = self.capital * 0.70  # Keep 30% buffer
            max_lots = max(1, int(available_margin / span_per_lot))
            num_lots = min(2, max_lots)  # Cap at 2 lots for safety

        qty = num_lots * self.lot_size
        symbol = self.resolve_symbol(float(strike), opt_type, option_chain)

        # Calculate max_loss from actual premium if available
        max_loss = qty * 50  # Fallback
        if option_chain:
            chain_entry = option_chain.get(int(strike), {}).get(opt_type, {})
            premium = chain_entry.get("ltp", 0)
            if premium > 0:
                max_loss = int(qty * premium * 1.1)  # 10% slippage buffer

        leg = OrderLeg(
            symbol=symbol,
            strike=float(strike),
            option_type=opt_type,
            side="BUY",
            qty=qty,
            price=0,  # Will be priced by orchestrator via Black-Scholes
        )

        return TradeSignal(
            strategy=self.name,
            action=action,
            confidence=confidence,
            underlying_price=spot,
            max_loss=max_loss,
            legs=[leg],
            metadata={
                "entry_type": entry_type,
                "is_zero_hero": is_zero_hero,
                "vix": round(vix, 1),
                "support": self._support,
                "resistance": self._resistance,
                "num_lots": num_lots,
                "strike": strike,
                "opt_type": opt_type,
            },
        )

    def set_previous_close(self, close: float) -> None:
        """Set previous day's close for gap calculation."""
        self._prev_close = close
        logger.info("V3: Previous close set to %.0f", close)
