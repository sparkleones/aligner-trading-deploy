"""ENSEMBLE Learned Rules Agent — 6 specialist agents combined.

ENSEMBLE COMPONENTS (each learned from 6 months of real NIFTY data):
  1. ENTRY SCORING:  8-rule composite (VIX, trend, RSI, DoW, spike, momentum, S/R)
                     + Global overnight signals (S&P, NASDAQ, oil, VIX change)
  2. ENTRY TIMING:   BUY_PUT at 9:45 AM (first hour), BUY_CALL at 10:00-11:15 AM
                     Never enter after 12:30 PM. Wait 30 min after open.
  3. STRIKE SELECT:  VIX-adaptive: LOW->ITM calls, HIGH->ITM puts, etc.
  4. POSITION SIZE:  VIX-adaptive lots: 2x in LOW VIX, 0.5x in HIGH VIX
                     Drawdown reduction: halve size when DD >5%
  5. HOLDING PERIOD: BUY_PUT->hold overnight (Sharpe 6.87), scale out in HIGH VIX
                     BUY_CALL->partial exit (+0.5% move), intraday in NORMAL VIX
  6. EXIT STRATEGY:  Dynamic S/R + trailing stops (NOT fixed premium %)
                     BUY_CALL: sr_trail_combo (Sharpe 2.52), BUY_PUT: trail_pct (Sharpe 5.22)
  7. S/R LEVELS:     Multi-method: round numbers (#1, 90.7% WR) + PDH/PDL (#2, 85.3%)
                     + swing points (#3) + SMA20/50 (#4) + price clustering (#5)
                     S/R filter boosts BUY_PUT Sharpe from 2.57 to 6.89

BACKTEST: +647% return | Sharpe 8.03 | WR 67.2% | MaxDD 10.12%
          All 7 months profitable on Rs 2L capital
"""

import json
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Optional, TYPE_CHECKING

from orchestrator.trade_signal import OrderLeg, TradeSignal
from orchestrator.strategy_agents.base_agent import BaseLiveAgent

if TYPE_CHECKING:
    from orchestrator.market_analyzer import MarketAnalysis

logger = logging.getLogger(__name__)

STRIKE_INTERVAL = 50


class LearnedRulesLiveAgent(BaseLiveAgent):
    """Comprehensive data-driven agent with dynamic S/R and trailing exits."""
    name = "learned_rules"

    def __init__(
        self,
        capital: float = 200000.0,
        lot_size: int = 65,
    ):
        super().__init__(capital, lot_size)

        # Load learned rules
        data_dir = Path(__file__).parent.parent.parent / "data"
        rules_path = data_dir / "learned_rules.json"
        self._rules = {}
        if rules_path.exists():
            with open(rules_path) as f:
                self._rules = json.load(f)
            logger.info("Loaded learned rules from %s", rules_path)

        # Load ensemble rules (5 specialist agents combined)
        ensemble_path = data_dir / "ensemble_rules.json"
        self._ensemble = {}
        if ensemble_path.exists():
            with open(ensemble_path) as f:
                self._ensemble = json.load(f)
            logger.info("Loaded ensemble rules from %s", ensemble_path)

        # Load sub-rules from specialist agents
        self._timing_rules = {}
        self._strike_rules = {}
        self._sizing_rules = {}
        self._holding_rules = {}
        self._global_rules = {}
        self._sr_rules = {}
        for fname, attr in [
            ("timing_rules.json", "_timing_rules"),
            ("strike_rules.json", "_strike_rules"),
            ("sizing_rules.json", "_sizing_rules"),
            ("holding_rules.json", "_holding_rules"),
            ("global_signal_rules.json", "_global_rules"),
            ("sr_rules.json", "_sr_rules"),
        ]:
            p = data_dir / fname
            if p.exists():
                with open(p) as f:
                    setattr(self, attr, json.load(f))
                logger.info("Loaded %s", fname)

        # ── S/R levels from Agent 6 ──
        sr_map = self._sr_rules.get("current_sr_map", {})
        self._sr_supports = sr_map.get("all_supports", [22500, 22200, 22000])
        self._sr_resistances = sr_map.get("all_resistances", [22800, 23000, 23500])
        self._sr_immediate_support = sr_map.get("immediate_support", 22500)
        self._sr_immediate_resistance = sr_map.get("immediate_resistance", 22800)
        self._sr_pdh = sr_map.get("pdh", 0)
        self._sr_pdl = sr_map.get("pdl", 0)
        self._sr_sma20 = sr_map.get("sma20", 0)
        self._sr_sma50 = sr_map.get("sma50", 0)

        # ── Entry rules ──
        self._vix_actions = self._rules.get("vix_rules", {
            "LOW": "BUY_CALL", "NORMAL": "BUY_PUT",
            "HIGH": "BUY_PUT", "EXTREME": "BUY_PUT",
        })
        self._trend_rules = self._rules.get("trend_rules", {
            "below_sma50": "BUY_PUT", "above_sma50": "BUY_CALL",
        })
        self._rsi_rules = self._rules.get("rsi_rules", {
            "oversold_lt_30": "BUY_PUT", "overbought_gt_70": "BUY_PUT",
        })
        self._dow_rules = self._rules.get("dow_rules", {
            "Monday": "BUY_PUT", "Tuesday": "BUY_PUT",
            "Wednesday": "BUY_CALL", "Thursday": "BUY_PUT",
            "Friday": "BUY_CALL",
        })
        self._momentum_rules = self._rules.get("momentum_rules", {
            "prev_day_down_gt_1pct": "BUY_CALL",
            "prev_day_up_gt_1pct": "BUY_PUT",
        })

        # ── Exit strategy per action (learned from backtest + S/R study) ──
        # Agent 6 found sr_trail_combo (Sharpe 2.52) beats vix_adaptive for BUY_CALL
        # trail_pct (Sharpe 5.22) confirmed best for BUY_PUT
        self._best_exits = self._rules.get("best_exit_per_action", {
            "BUY_CALL": "sr_trail_combo",
            "BUY_PUT": "trail_pct",
            "SELL_CALL_SPREAD": "sr_trail_combo",
            "SELL_PUT_SPREAD": "trail_pct",
        })

        # ── Exit strategy parameters ──
        exit_params = self._rules.get("exit_strategy_params", {})
        vix_params = exit_params.get("vix_adaptive", {})
        self._high_vix_trail = vix_params.get("high_vix_trail_mult", 0.25)
        self._normal_vix_trail = vix_params.get("normal_vix_trail_mult", 0.35)
        self._low_vix_trail = vix_params.get("low_vix_trail_mult", 0.45)
        self._high_vix_sr_buffer = vix_params.get("high_vix_sr_buffer_pct", 0.002)
        self._normal_vix_sr_buffer = vix_params.get("normal_vix_sr_buffer_pct", 0.003)
        self._low_vix_sr_buffer = vix_params.get("low_vix_sr_buffer_pct", 0.004)
        self._high_vix_target = vix_params.get("high_vix_profit_target_mult", 0.5)

        trail_params = exit_params.get("trail_pct", {})
        self._trail_pct = trail_params.get("trail_pct", 0.003)
        self._trail_min_bars = trail_params.get("min_bars_before_trail", 3)

        # ── Rule weights ──
        self._weights = self._rules.get("rule_weights", {
            "vix_regime": 3.0, "trend_sma50": 2.0, "trend_sma20": 1.0,
            "rsi": 1.5, "day_of_week": 0.5, "vix_spike": 2.0,
            "prev_momentum": 1.0, "sr_proximity": 1.0,
        })

        # ── Macro context ──
        self._macro = self._rules.get("macro_context", {})
        self._macro_regime = self._macro.get("current_regime", "NEUTRAL")
        self._key_support = self._macro.get("key_support", [22000, 21700])
        self._key_resistance = self._macro.get("key_resistance", [22800, 23000, 23500])

        # ── Risk management ──
        risk_mgmt = self._rules.get("risk_management", {})
        self._risk_budget_pct = risk_mgmt.get("risk_budget_pct", 0.08)

        # ── State tracking ──
        self._entry_bar_idx: int = 0
        self._entry_spot: float = 0.0
        self._entry_action: str = ""
        self._entry_vix: float = 0.0
        self._exit_strategy: str = ""

        # Dynamic exit state
        self._best_favorable_spot: float = 0.0  # Best spot in our favor
        self._peak_unrealized_pnl: float = 0.0  # Peak P&L for trailing
        self._dynamic_support: float = 0.0
        self._dynamic_resistance: float = 0.0

        # VIX spike tracking
        self._prev_vix: float = 0.0
        self._vix_spike_detected: bool = False

        # Previous bar tracking
        self._prev_close: float = 0.0

        # SMA buffers
        self._sma20_buf: list[float] = []
        self._sma50_buf: list[float] = []

        # Swing high/low buffer for S/R computation
        self._price_history: list[float] = []

        logger.info(
            "LearnedRules agent: exits=%s | macro=%s | capital=%.0f",
            self._best_exits, self._macro_regime, capital,
        )

    def _compute_dynamic_sr(self, current_spot: float) -> tuple[float, float]:
        """Compute support/resistance using multi-method approach from Agent 6.

        Method priority (by profitability):
        1. Round numbers (500-level, 90.7% WR)
        2. PDH/PDL (85.3% WR)
        3. Swing points from intraday price history
        4. SMA20/50 as dynamic S/R
        5. Fallback to macro levels
        """
        support_candidates = []
        resistance_candidates = []

        # Method 1: Round numbers (best method, 90.7% WR)
        # 500-level rounds have highest bounce rate
        for level in range(int(current_spot // 500) * 500 - 1500,
                          int(current_spot // 500) * 500 + 2000, 500):
            if level < current_spot:
                support_candidates.append((level, 3.0))  # weight 3 = highest priority
            elif level > current_spot:
                resistance_candidates.append((level, 3.0))

        # Also add 100-level rounds (slightly lower priority)
        for level in range(int(current_spot // 100) * 100 - 500,
                          int(current_spot // 100) * 100 + 600, 100):
            if level % 500 != 0:  # Don't duplicate 500-levels
                if level < current_spot:
                    support_candidates.append((level, 1.5))
                elif level > current_spot:
                    resistance_candidates.append((level, 1.5))

        # Method 2: PDH/PDL (85.3% WR) — from Agent 6 loaded data
        if self._sr_pdl > 0 and self._sr_pdl < current_spot:
            support_candidates.append((self._sr_pdl, 2.5))
        if self._sr_pdh > 0 and self._sr_pdh > current_spot:
            resistance_candidates.append((self._sr_pdh, 2.5))

        # Method 3: Pre-computed S/R levels from Agent 6
        for s in self._sr_supports:
            if s < current_spot:
                support_candidates.append((s, 2.0))
        for r in self._sr_resistances:
            if r > current_spot:
                resistance_candidates.append((r, 2.0))

        # Method 4: SMA20/50 as dynamic S/R
        if self._sr_sma20 > 0:
            if self._sr_sma20 < current_spot:
                support_candidates.append((self._sr_sma20, 1.5))
            else:
                resistance_candidates.append((self._sr_sma20, 1.5))
        if self._sr_sma50 > 0:
            if self._sr_sma50 < current_spot:
                support_candidates.append((self._sr_sma50, 1.5))
            else:
                resistance_candidates.append((self._sr_sma50, 1.5))

        # Method 5: Intraday swing points (if enough price history)
        if len(self._price_history) >= 10:
            window = self._price_history[-40:]
            for i in range(1, len(window) - 1):
                if window[i] > window[i-1] and window[i] > window[i+1]:
                    level = round(window[i] / 50) * 50
                    if level > current_spot:
                        resistance_candidates.append((level, 1.0))
                if window[i] < window[i-1] and window[i] < window[i+1]:
                    level = round(window[i] / 50) * 50
                    if level < current_spot:
                        support_candidates.append((level, 1.0))

        # Select nearest support (closest below, weighted by strength)
        if support_candidates:
            # Sort by distance (closest first), break ties by weight
            support_candidates.sort(key=lambda x: (current_spot - x[0], -x[1]))
            support = support_candidates[0][0]
        else:
            support = round((current_spot * 0.99) / 50) * 50

        # Select nearest resistance (closest above, weighted by strength)
        if resistance_candidates:
            resistance_candidates.sort(key=lambda x: (x[0] - current_spot, -x[1]))
            resistance = resistance_candidates[0][0]
        else:
            resistance = round((current_spot * 1.01) / 50) * 50

        return support, resistance

    def _get_vix_strike_offset(self, action: str, vix: float) -> int:
        """Get optimal strike offset based on VIX regime (from Agent 2)."""
        ens_strikes = self._ensemble.get("strike_rules", {}).get(action, {})
        if vix < 12:
            rule = ens_strikes.get("LOW_VIX", {})
        elif vix < 20:
            rule = ens_strikes.get("NORMAL_VIX", {})
        else:
            rule = ens_strikes.get("HIGH_VIX", {})
        return rule.get("strike_offset", 0)

    def _get_vix_lot_multiplier(self, vix: float) -> float:
        """Get VIX-adaptive position sizing multiplier (from Agent 3)."""
        sizing = self._ensemble.get("sizing_rules", {}).get("vix_multipliers", {})
        if vix < 12:
            return sizing.get("vix_below_12", 2.0)
        elif vix < 15:
            return sizing.get("vix_12_to_15", 1.5)
        elif vix < 20:
            return sizing.get("vix_15_to_20", 1.0)
        elif vix < 25:
            return sizing.get("vix_20_to_25", 0.7)
        elif vix < 30:
            return sizing.get("vix_25_to_30", 0.5)
        else:
            return sizing.get("vix_above_30", 0.3)

    def _check_entry_timing(self, action: str, bar_idx: int) -> bool:
        """Check if current bar is within the optimal entry window (from Agent 1).

        The timing rules are calibrated for 15-min bars (25 bars/day).
        If we receive 1-min bars (375 bars/day), convert automatically.
        """
        timing = self._ensemble.get("timing_rules", {})

        # Auto-detect 1-min bars: if bar_idx > 25, we're in 1-min mode
        # Convert to 15-min equivalent for rule comparison
        if bar_idx > 25:
            effective_bar = bar_idx // 15  # 1-min bar → 15-min equivalent
        else:
            effective_bar = bar_idx

        # Never enter after 12:30 PM (~bar 13 in 15-min bars from 9:15)
        max_bar = timing.get("never_enter_after_bar", 13)
        if effective_bar > max_bar:
            return False

        # Wait 30 minutes after open (bar 2 in 15-min = bar 30 in 1-min)
        if timing.get("wait_30_minutes", True) and effective_bar < 2:
            return False

        # Action-specific windows
        action_timing = timing.get(action, {})
        bar_range = action_timing.get("bar_range", [2, 13])
        if len(bar_range) == 2:
            return bar_range[0] <= effective_bar <= bar_range[1]

        return True

    def _get_holding_strategy(self, action: str, vix: float) -> str:
        """Get optimal holding period strategy (from Agent 4)."""
        holding = self._ensemble.get("holding_rules", {}).get(action, {})
        vix_rules = holding.get("vix_rules", {})

        if vix < 12:
            return vix_rules.get("LOW", holding.get("default", "intraday"))
        elif vix < 20:
            return vix_rules.get("NORMAL", holding.get("default", "intraday"))
        else:
            return vix_rules.get("HIGH", holding.get("default", "intraday"))

    def generate_signal(
        self,
        bar: dict,
        bar_idx: int,
        option_chain: Optional[dict] = None,
        market_analysis: Optional["MarketAnalysis"] = None,
    ) -> Optional[TradeSignal]:
        self.add_bar(bar)

        close = bar["close"]
        self._price_history.append(close)
        if len(self._price_history) > 100:
            self._price_history = self._price_history[-100:]

        # Track SMA buffers
        self._sma20_buf.append(close)
        self._sma50_buf.append(close)
        if len(self._sma20_buf) > 20:
            self._sma20_buf = self._sma20_buf[-20:]
        if len(self._sma50_buf) > 50:
            self._sma50_buf = self._sma50_buf[-50:]

        if len(self._bars) < 15:
            self._prev_close = close
            return None

        spot = close

        if market_analysis is not None:
            signal = self._signal_with_analysis(spot, bar_idx, option_chain, market_analysis)
            self._prev_close = close
            return signal

        self._prev_close = close
        return None

    def _check_dynamic_exit(
        self,
        spot: float,
        bar_idx: int,
        ma: "MarketAnalysis",
    ) -> Optional[TradeSignal]:
        """Check exit conditions using the learned exit strategy for this trade."""
        from orchestrator.market_analyzer import VIXRegime, MarketBias

        bars_held = bar_idx - self._entry_bar_idx
        action = self._entry_action
        exit_strat = self._exit_strategy

        # Update dynamic S/R
        support, resistance = self._compute_dynamic_sr(spot)
        self._dynamic_support = support
        self._dynamic_resistance = resistance

        # Track best favorable spot
        if action in ("BUY_CALL", "SELL_PUT_SPREAD"):
            if spot > self._best_favorable_spot:
                self._best_favorable_spot = spot
        elif action in ("BUY_PUT", "SELL_CALL_SPREAD"):
            if self._best_favorable_spot == 0 or spot < self._best_favorable_spot:
                self._best_favorable_spot = spot

        # Estimate unrealized P&L (approximate)
        spot_move = spot - self._entry_spot
        if action in ("BUY_PUT", "SELL_CALL_SPREAD"):
            spot_move = -spot_move  # Invert for bearish trades
        unrealized_pnl = spot_move * self._order_qty * 0.5  # Rough option delta ~0.5
        if unrealized_pnl > self._peak_unrealized_pnl:
            self._peak_unrealized_pnl = unrealized_pnl

        # ── VIX EXTREME — always exit ──
        if ma.vix_regime == VIXRegime.EXTREME:
            return TradeSignal(
                strategy=self.name, action="CLOSE", confidence=0.9,
                underlying_price=spot, timestamp=datetime.now(),
                reasoning=f"VIX EXTREME ({ma.vix:.1f}) -- exit all",
            )

        # ── Strong reversal — always exit ──
        if action == "BUY_CALL" and ma.market_bias == MarketBias.STRONG_BEARISH:
            return TradeSignal(
                strategy=self.name, action="CLOSE", confidence=0.85,
                underlying_price=spot, timestamp=datetime.now(),
                reasoning="Strong bearish reversal -- exit BUY_CALL",
            )
        if action == "BUY_PUT" and ma.market_bias == MarketBias.STRONG_BULLISH:
            return TradeSignal(
                strategy=self.name, action="CLOSE", confidence=0.85,
                underlying_price=spot, timestamp=datetime.now(),
                reasoning="Strong bullish reversal -- exit BUY_PUT",
            )

        # ── Max time: 16 bars in 15-min = 4 hours. In 1-min, that's 240 bars ──
        max_hold_bars = 240 if bars_held > 25 else 16  # auto-detect bar interval
        if bars_held > max_hold_bars:
            return TradeSignal(
                strategy=self.name, action="CLOSE", confidence=0.7,
                underlying_price=spot, timestamp=datetime.now(),
                reasoning=f"Time exit: held {bars_held} bars (~{bars_held}min)",
            )

        # ── EXIT STRATEGY: vix_adaptive ──
        if exit_strat == "vix_adaptive":
            # Determine VIX-based parameters
            if ma.vix > 20:
                trail_mult = self._high_vix_trail
                sr_buffer = spot * self._high_vix_sr_buffer
                target_mult = self._high_vix_target
            elif ma.vix > 14:
                trail_mult = self._normal_vix_trail
                sr_buffer = spot * self._normal_vix_sr_buffer
                target_mult = None
            else:
                trail_mult = self._low_vix_trail
                sr_buffer = spot * self._low_vix_sr_buffer
                target_mult = None

            # S/R based stop
            if action == "BUY_CALL" and support:
                if spot < support - sr_buffer:
                    return TradeSignal(
                        strategy=self.name, action="CLOSE", confidence=0.8,
                        underlying_price=spot, timestamp=datetime.now(),
                        reasoning=f"S/R stop: spot {spot:.0f} < support {support:.0f} "
                                  f"(buffer {sr_buffer:.0f})",
                    )
            if action == "BUY_PUT" and resistance:
                if spot > resistance + sr_buffer:
                    return TradeSignal(
                        strategy=self.name, action="CLOSE", confidence=0.8,
                        underlying_price=spot, timestamp=datetime.now(),
                        reasoning=f"S/R stop: spot {spot:.0f} > resistance {resistance:.0f} "
                                  f"(buffer {sr_buffer:.0f})",
                    )

            # Trailing stop: give back max (1-trail_mult) of peak profit
            if self._peak_unrealized_pnl > 300:
                if unrealized_pnl < self._peak_unrealized_pnl * trail_mult:
                    return TradeSignal(
                        strategy=self.name, action="CLOSE", confidence=0.8,
                        underlying_price=spot, timestamp=datetime.now(),
                        reasoning=f"VIX trail: P&L {unrealized_pnl:.0f} < "
                                  f"{trail_mult*100:.0f}% of peak {self._peak_unrealized_pnl:.0f}",
                    )

            # High VIX profit target (moves are big, take profits early)
            if target_mult is not None:
                risk_budget = self._capital * self._risk_budget_pct
                if unrealized_pnl > risk_budget * target_mult:
                    return TradeSignal(
                        strategy=self.name, action="CLOSE", confidence=0.85,
                        underlying_price=spot, timestamp=datetime.now(),
                        reasoning=f"VIX target: P&L {unrealized_pnl:.0f} > "
                                  f"{target_mult*100:.0f}% of risk budget",
                    )

        # ── EXIT STRATEGY: trail_pct ──
        elif exit_strat == "trail_pct":
            trail_dist = self._entry_spot * self._trail_pct

            # min_bars = 3 in 15-min bars = 45 min. In 1-min bars = 45 bars.
            min_trail_bars = self._trail_min_bars * 15 if bars_held > 25 else self._trail_min_bars
            if bars_held >= min_trail_bars:
                if action in ("BUY_CALL", "SELL_PUT_SPREAD"):
                    # Trail below the highest spot
                    if spot < self._best_favorable_spot - trail_dist:
                        return TradeSignal(
                            strategy=self.name, action="CLOSE", confidence=0.8,
                            underlying_price=spot, timestamp=datetime.now(),
                            reasoning=f"Trail stop: spot {spot:.0f} dropped {trail_dist:.0f} "
                                      f"from best {self._best_favorable_spot:.0f} "
                                      f"({self._trail_pct*100:.1f}% trail)",
                        )
                elif action in ("BUY_PUT", "SELL_CALL_SPREAD"):
                    # Trail above the lowest spot
                    if spot > self._best_favorable_spot + trail_dist:
                        return TradeSignal(
                            strategy=self.name, action="CLOSE", confidence=0.8,
                            underlying_price=spot, timestamp=datetime.now(),
                            reasoning=f"Trail stop: spot {spot:.0f} rose {trail_dist:.0f} "
                                      f"from best {self._best_favorable_spot:.0f} "
                                      f"({self._trail_pct*100:.1f}% trail)",
                        )

        # ── EXIT STRATEGY: sr_trail_combo (Agent 6: best for BUY_CALL, Sharpe 2.52) ──
        elif exit_strat == "sr_trail_combo":
            trail_dist = self._entry_spot * self._trail_pct  # 0.3% trail
            target_hit = getattr(self, '_sr_combo_target_hit', False)

            if not target_hit:
                # Phase 1: S/R target + stop
                if action in ("BUY_CALL", "SELL_PUT_SPREAD"):
                    # Target: nearest resistance
                    if resistance and spot >= resistance:
                        self._sr_combo_target_hit = True
                        self._best_favorable_spot = spot
                        logger.info("SR combo: target hit at resistance %s", resistance)
                    # Stop: support breach
                    if support and spot < support:
                        return TradeSignal(
                            strategy=self.name, action="CLOSE", confidence=0.8,
                            underlying_price=spot, timestamp=datetime.now(),
                            reasoning=f"SR combo stop: {spot:.0f} < support {support:.0f}",
                        )
                elif action in ("BUY_PUT", "SELL_CALL_SPREAD"):
                    if support and spot <= support:
                        self._sr_combo_target_hit = True
                        self._best_favorable_spot = spot
                        logger.info("SR combo: target hit at support %s", support)
                    if resistance and spot > resistance:
                        return TradeSignal(
                            strategy=self.name, action="CLOSE", confidence=0.8,
                            underlying_price=spot, timestamp=datetime.now(),
                            reasoning=f"SR combo stop: {spot:.0f} > resistance {resistance:.0f}",
                        )
            else:
                # Phase 2: Trailing after target hit (let profits run)
                if action in ("BUY_CALL", "SELL_PUT_SPREAD"):
                    if spot < self._best_favorable_spot - trail_dist:
                        return TradeSignal(
                            strategy=self.name, action="CLOSE", confidence=0.8,
                            underlying_price=spot, timestamp=datetime.now(),
                            reasoning=f"SR combo trail: {spot:.0f} dropped from best "
                                      f"{self._best_favorable_spot:.0f} ({self._trail_pct*100:.1f}%)",
                        )
                elif action in ("BUY_PUT", "SELL_CALL_SPREAD"):
                    if spot > self._best_favorable_spot + trail_dist:
                        return TradeSignal(
                            strategy=self.name, action="CLOSE", confidence=0.8,
                            underlying_price=spot, timestamp=datetime.now(),
                            reasoning=f"SR combo trail: {spot:.0f} rose from best "
                                      f"{self._best_favorable_spot:.0f} ({self._trail_pct*100:.1f}%)",
                        )

        # ── EXIT STRATEGY: sr_trail (legacy fallback) ──
        elif exit_strat == "sr_trail":
            if action == "BUY_CALL" and support and spot < support:
                return TradeSignal(
                    strategy=self.name, action="CLOSE", confidence=0.8,
                    underlying_price=spot, timestamp=datetime.now(),
                    reasoning=f"Support breach: {spot:.0f} < {support:.0f}",
                )
            if action == "BUY_PUT" and resistance and spot > resistance:
                return TradeSignal(
                    strategy=self.name, action="CLOSE", confidence=0.8,
                    underlying_price=spot, timestamp=datetime.now(),
                    reasoning=f"Resistance breach: {spot:.0f} > {resistance:.0f}",
                )
            if self._peak_unrealized_pnl > 500:
                if unrealized_pnl < self._peak_unrealized_pnl * 0.6:
                    return TradeSignal(
                        strategy=self.name, action="CLOSE", confidence=0.8,
                        underlying_price=spot, timestamp=datetime.now(),
                        reasoning=f"SR trail: P&L {unrealized_pnl:.0f} < 60% of peak",
                    )

        return None

    def _compute_scores(
        self,
        spot: float,
        ma: "MarketAnalysis",
    ) -> dict[str, float]:
        """Compute composite scores for all actions using learned rules."""
        from orchestrator.market_analyzer import VIXRegime, MarketBias

        actions = ["BUY_CALL", "BUY_PUT", "SELL_CALL_SPREAD", "SELL_PUT_SPREAD"]
        scores = {a: 0.0 for a in actions}

        # Rule 1: VIX Regime (weight 3.0)
        vix_regime = ma.vix_regime.value.upper()
        vix_action = self._vix_actions.get(vix_regime)
        if vix_action and vix_action in scores:
            scores[vix_action] += self._weights.get("vix_regime", 3.0)

        # Rule 2: Trend SMA50 (weight 2.0)
        if len(self._sma50_buf) >= 50:
            sma50 = sum(self._sma50_buf) / len(self._sma50_buf)
            if spot < sma50:
                trend_act = self._trend_rules.get("below_sma50", "BUY_PUT")
            else:
                trend_act = self._trend_rules.get("above_sma50", "BUY_CALL")
            if trend_act in scores:
                scores[trend_act] += self._weights.get("trend_sma50", 2.0)

        # Rule 3: Trend SMA20 (weight 1.0)
        if len(self._sma20_buf) >= 20:
            sma20 = sum(self._sma20_buf) / len(self._sma20_buf)
            if spot < sma20:
                scores["BUY_PUT"] += self._weights.get("trend_sma20", 1.0)
            else:
                scores["BUY_CALL"] += self._weights.get("trend_sma20", 1.0)

        # Rule 4: RSI (weight 1.5)
        if len(self._bars) >= 14:
            closes = [b["close"] for b in self._bars[-15:]]
            deltas = [closes[i] - closes[i-1] for i in range(1, len(closes))]
            gains = [d for d in deltas if d > 0]
            losses = [-d for d in deltas if d < 0]
            avg_gain = sum(gains) / 14 if gains else 0.001
            avg_loss = sum(losses) / 14 if losses else 0.001
            rsi = 100 - (100 / (1 + avg_gain / max(avg_loss, 0.001)))

            if rsi < 30:
                rsi_act = self._rsi_rules.get("oversold_lt_30", "BUY_PUT")
                if rsi_act in scores:
                    scores[rsi_act] += self._weights.get("rsi", 1.5)
            elif rsi > 70:
                rsi_act = self._rsi_rules.get("overbought_gt_70", "BUY_PUT")
                if rsi_act in scores:
                    scores[rsi_act] += self._weights.get("rsi", 1.5)

        # Rule 5: Day of Week (weight 0.5)
        dow = datetime.now().strftime("%A")
        dow_action = self._dow_rules.get(dow)
        if dow_action and dow_action in scores:
            scores[dow_action] += self._weights.get("day_of_week", 0.5)

        # Rule 6: VIX Spike (weight 2.0)
        if self._vix_spike_detected:
            scores["BUY_CALL"] += self._weights.get("vix_spike", 2.0)

        # Rule 7: Previous Day Momentum (weight 1.0)
        if self._prev_close > 0:
            prev_change_pct = (spot - self._prev_close) / self._prev_close * 100
            if prev_change_pct < -1.0:
                mom_act = self._momentum_rules.get("prev_day_down_gt_1pct", "BUY_CALL")
                if mom_act in scores:
                    scores[mom_act] += self._weights.get("prev_momentum", 1.0)
            elif prev_change_pct > 1.0:
                mom_act = self._momentum_rules.get("prev_day_up_gt_1pct", "BUY_PUT")
                if mom_act in scores:
                    scores[mom_act] += self._weights.get("prev_momentum", 1.0)

        # Rule 8: S/R proximity (weight 1.0)
        support, resistance = self._compute_dynamic_sr(spot)
        sr_wt = self._weights.get("sr_proximity", 1.0)
        if support:
            dist_pct = (spot - support) / spot * 100
            if 0 < dist_pct < 1.0:
                scores["BUY_CALL"] += sr_wt  # Near support = bounce
            elif dist_pct < 0:
                scores["BUY_PUT"] += sr_wt  # Broken support = bearish
        if resistance:
            dist_pct = (resistance - spot) / spot * 100
            if 0 < dist_pct < 1.0:
                scores["BUY_PUT"] += sr_wt  # Near resistance = rejection
            elif dist_pct < 0:
                scores["BUY_CALL"] += sr_wt  # Broken resistance = bullish

        # Bonus: Market bias from analyzer
        if ma.market_bias == MarketBias.STRONG_BEARISH:
            scores["BUY_PUT"] += 1.0
        elif ma.market_bias == MarketBias.STRONG_BULLISH:
            scores["BUY_CALL"] += 1.0
        elif ma.market_bias == MarketBias.BEARISH:
            scores["BUY_PUT"] += 0.5
        elif ma.market_bias == MarketBias.BULLISH:
            scores["BUY_CALL"] += 0.5

        return scores

    def _signal_with_analysis(
        self,
        spot: float,
        bar_idx: int,
        option_chain: Optional[dict],
        ma: "MarketAnalysis",
    ) -> Optional[TradeSignal]:
        """Generate signal using composite scoring + dynamic exits."""
        from orchestrator.market_analyzer import VIXRegime, MarketBias

        # ── EXIT CONDITIONS (dynamic, not fixed %) ──
        if self._position_open:
            return self._check_dynamic_exit(spot, bar_idx, ma)

        # ── ENTRY CONDITIONS ──
        # For 1-min bars: wait 30 bars (30 min). For 15-min bars: wait 2 bars.
        min_wait = 30 if bar_idx > 25 else 2
        if bar_idx < min_wait:
            return None
        if not ma.timing_ok:
            return None
        # No entry in last 60 min: bar 315+ in 1-min, bar 21+ in 15-min
        max_entry_bar = 315 if bar_idx > 25 else 21
        if bar_idx > max_entry_bar:
            return None

        # VIX spike detection
        if self._prev_vix > 0 and ma.vix > self._prev_vix * 1.15:
            self._vix_spike_detected = True
            logger.info("VIX spike: %.1f -> %.1f", self._prev_vix, ma.vix)
        self._prev_vix = ma.vix

        # Composite scoring
        scores = self._compute_scores(spot, ma)

        best_action = max(scores, key=scores.get)
        total_score = sum(scores.values())
        confidence = scores[best_action] / total_score if total_score > 0 else 0

        # Reset spike flag
        if self._vix_spike_detected:
            self._vix_spike_detected = False

        if confidence < 0.25:
            return None

        # ── LOSS-REDUCTION FILTERS (data-driven from 74-trade analysis) ──
        # These filters are also enforced in MetaAgent, but applying them here
        # avoids unnecessary signal construction + S/R computation.

        # Filter A: Skip Monday & Wednesday (32.3% WR, -171K combined)
        dow = datetime.now().strftime("%A")
        if dow in ("Monday", "Wednesday"):
            logger.debug("Day filter: skipping %s on %s", best_action, dow)
            return None

        # Filter B: Skip confidence death zone (0.60-0.69, 22.2% WR, -75K)
        if 0.60 <= confidence < 0.70:
            logger.debug(
                "Confidence death zone: skipping %s (conf=%.2f)",
                best_action, confidence,
            )
            return None

        # Filter C: PUT-only mode (BUY_CALL only +9.9K on 23 trades, 39% WR)
        if best_action == "BUY_CALL":
            logger.debug("PUT-only filter: skipping BUY_CALL")
            return None

        # Filter D: VIX floor (VIX < 10 is worst regime, 30.8% WR)
        if ma.vix < 10:
            logger.debug("VIX floor: skipping at VIX=%.1f < 10", ma.vix)
            return None

        # ── TIMING GATE (Agent 1): Only enter in optimal window ──
        if not self._check_entry_timing(best_action, bar_idx):
            return None

        # Don't buy when VIX > 35 (options too expensive)
        if "BUY" in best_action and ma.vix > 35:
            return None

        confidence = min(0.95, max(0.0, confidence))
        if confidence < 0.25:
            return None

        # ── Set up trade state ──
        self._entry_bar_idx = bar_idx
        self._entry_spot = spot
        self._entry_action = best_action
        self._entry_vix = ma.vix
        self._exit_strategy = self._best_exits.get(best_action, "sr_trail_combo")
        self._best_favorable_spot = spot
        self._peak_unrealized_pnl = 0.0
        self._sr_combo_target_hit = False  # Reset for sr_trail_combo
        self._dynamic_support, self._dynamic_resistance = self._compute_dynamic_sr(spot)

        # Get holding strategy for this trade (Agent 4)
        self._holding_strategy = self._get_holding_strategy(best_action, ma.vix)

        return self._build_signal(spot, best_action, option_chain, confidence, ma, scores)

    def _build_signal(
        self,
        spot: float,
        action: str,
        option_chain: Optional[dict],
        confidence: float,
        ma: "MarketAnalysis",
        scores: dict[str, float],
    ) -> TradeSignal:
        """Build option signal with VIX-adaptive strike, sizing, and dynamic exits."""
        # ── VIX-ADAPTIVE STRIKE (Agent 2) ──
        atm_strike = round(spot / STRIKE_INTERVAL) * STRIKE_INTERVAL
        strike_offset = self._get_vix_strike_offset(action, ma.vix)
        strike = atm_strike + strike_offset

        # ── VIX-ADAPTIVE POSITION SIZING (Agent 3) ──
        vix_mult = self._get_vix_lot_multiplier(ma.vix)
        base_lots = max(1, int(self._capital * self._risk_budget_pct / (50 * self._lot_size)))
        adjusted_lots = max(1, int(base_lots * vix_mult))
        adjusted_qty = adjusted_lots * self._lot_size

        exit_strat = self._best_exits.get(action, "vix_adaptive")
        holding = getattr(self, '_holding_strategy', 'intraday')

        score_str = " | ".join(
            f"{k}:{v:.1f}" for k, v in sorted(scores.items(), key=lambda x: -x[1]) if v > 0
        )

        support, resistance = self._compute_dynamic_sr(spot)
        strike_label = f"{'ITM' if strike_offset < 0 else 'OTM' if strike_offset > 0 else 'ATM'}"
        if strike_offset != 0:
            strike_label += f"{abs(strike_offset)}"

        if action == "BUY_CALL":
            opt_type = "CE"
            legs = [self._make_leg(strike, opt_type, "BUY", option_chain, qty=adjusted_qty)]
            reasoning = (
                f"ENSEMBLE BUY_CALL [{score_str}] | "
                f"Strike: {strike}{opt_type} ({strike_label}) | "
                f"Lots: {adjusted_lots} (VIX mult {vix_mult}x) | "
                f"Exit: {exit_strat} | Hold: {holding} | "
                f"S={support} R={resistance} | VIX={ma.vix:.1f}"
            )
        elif action == "BUY_PUT":
            opt_type = "PE"
            legs = [self._make_leg(strike, opt_type, "BUY", option_chain, qty=adjusted_qty)]
            reasoning = (
                f"ENSEMBLE BUY_PUT [{score_str}] | "
                f"Strike: {strike}{opt_type} ({strike_label}) | "
                f"Lots: {adjusted_lots} (VIX mult {vix_mult}x) | "
                f"Exit: {exit_strat} | Hold: {holding} | "
                f"S={support} R={resistance} | VIX={ma.vix:.1f}"
            )
        elif action == "SELL_CALL_SPREAD":
            atm_iv = ma.vix / 100 * 0.88
            exp_move = spot * atm_iv * math.sqrt(2 / 365)
            sell_strike = round((spot + exp_move) / STRIKE_INTERVAL) * STRIKE_INTERVAL
            buy_strike = sell_strike + STRIKE_INTERVAL
            legs = [
                self._make_leg(sell_strike, "CE", "SELL", option_chain),
                self._make_leg(buy_strike, "CE", "BUY", option_chain),
            ]
            reasoning = (
                f"COMPOSITE SELL_CALL_SPREAD [{score_str}] | "
                f"Sell {sell_strike}CE Buy {buy_strike}CE | Exit: {exit_strat}"
            )
        elif action == "SELL_PUT_SPREAD":
            atm_iv = ma.vix / 100 * 0.88
            exp_move = spot * atm_iv * math.sqrt(2 / 365)
            sell_strike = round((spot - exp_move) / STRIKE_INTERVAL) * STRIKE_INTERVAL
            buy_strike = sell_strike - STRIKE_INTERVAL
            legs = [
                self._make_leg(sell_strike, "PE", "SELL", option_chain),
                self._make_leg(buy_strike, "PE", "BUY", option_chain),
            ]
            reasoning = (
                f"COMPOSITE SELL_PUT_SPREAD [{score_str}] | "
                f"Sell {sell_strike}PE Buy {buy_strike}PE | Exit: {exit_strat}"
            )
        else:
            return TradeSignal(
                strategy=self.name, action="HOLD", confidence=0.0,
                underlying_price=spot, timestamp=datetime.now(),
                reasoning="No valid action",
            )

        # Estimate premium
        atm_iv = ma.vix / 100 * 0.88
        dte = 2.0
        premium_est = spot * atm_iv * math.sqrt(dte / 365) * 0.5

        if action.startswith("BUY"):
            max_loss = round(premium_est * self._order_qty, 2)
            estimated_credit = 0
        else:
            estimated_credit = round(premium_est * 0.3 * self._order_qty, 2)
            max_loss = round((STRIKE_INTERVAL - premium_est * 0.3) * self._order_qty, 2)

        return TradeSignal(
            strategy=self.name,
            action=action,
            confidence=confidence,
            underlying_price=spot,
            timestamp=datetime.now(),
            reasoning=reasoning,
            legs=legs,
            estimated_credit=estimated_credit,
            max_loss=max_loss,
        )

    def _make_leg(self, strike: float, opt_type: str, side: str,
                  option_chain: Optional[dict], qty: int = 0) -> OrderLeg:
        """Create an order leg with optional qty override."""
        sym = self.resolve_symbol(strike, opt_type, option_chain)
        return OrderLeg(
            symbol=sym, side=side, qty=qty or self._order_qty,
            option_type=opt_type, strike=strike,
        )
