"""Live Trade Monitor — active management layer for open positions.

Runs on every new bar *after* the existing scoring/engine.evaluate_exit.
If evaluate_exit does not fire, the Monitor gets a chance to exit on its
own signals:

  1. Premium trail   — actual option premium gave back N% from peak
  2. Reversal        — 2+ technical indicators flipped against the trade
  3. Structural      — trade is underwater AND reversal signals confirm
  4. Emergency floor — premium collapsed to ~40% of peak (runaway loss)

All thresholds are config-driven so they can be tuned via backtest
without touching code.

Usage pattern (both backtest and live):

    from orchestrator.live_trade_monitor import LiveTradeMonitor

    monitor = LiveTradeMonitor(cfg)                       # once per session
    # inside the per-bar exit loop, AFTER evaluate_exit():
    if not exit_reason:
        exit_reason = monitor.check(pos, ind, current_prem, bar_idx)

The Monitor mutates `pos` with these tracking keys:
    pos["peak_premium"]     — highest premium seen since entry
    pos["underwater_bars"]  — consecutive bars where spot is against entry
    pos["in_profit_phase"]  — latched True once pos has been profitable
    pos["mon_prev_macd_hist"] — for MACD cross detection
"""

from __future__ import annotations

from typing import Optional


# ═══════════════════════════════════════════════════════════════════
# DEFAULT CONFIG — tunable via backtest, can be overridden per call
# ═══════════════════════════════════════════════════════════════════
MONITOR_DEFAULTS = {
    "monitor_enabled": True,

    # ── Premium trail ──
    # Exit when premium has given back this much of its peak,
    # IF peak was at least this much above entry.
    "mon_prem_trail_activate_mult": 1.30,   # peak must be >= 1.30 × entry
    "mon_prem_trail_giveback_frac": 0.60,   # then if current < 60 % of peak

    # ── Reversal confirmation ──
    # Reversal score counts flipped indicators. Exit when score >= threshold
    # AND position has already been in profit.
    "mon_reversal_score_min": 3,            # sum of weighted signals
    "mon_reversal_min_bars_held": 3,        # don't fire immediately after entry
    "mon_reversal_require_profit": True,    # only arm after profit phase

    # ── Structural breakdown ──
    # If trade never reached profit AND we see reversal signals AND
    # several consecutive bars are against us, cut losses.
    "mon_structural_underwater_bars": 5,
    "mon_structural_score_min": 3,

    # ── Emergency floor ──
    # Catches the "+4000 → -11000" trap: premium was positive, now collapsed.
    "mon_emergency_peak_mult": 1.20,        # peak was >= 1.20 × entry
    "mon_emergency_floor_frac": 0.50,       # and current <= 50 % of peak

    # ── Signal weights (reversal score) ──
    "mon_w_supertrend": 2,
    "mon_w_macd_cross": 1,
    "mon_w_ema_cross": 1,
    "mon_w_rsi_exhaust": 1,
    "mon_w_psar_flip": 1,
    "mon_rsi_call_exhaust": 72,
    "mon_rsi_put_exhaust": 28,
}


class LiveTradeMonitor:
    """Per-session stateful monitor. Also callable as pure function via
    `monitor_check()` for use inside backtests where you'd rather not carry
    an instance.
    """

    def __init__(self, cfg: Optional[dict] = None):
        self.cfg = {**MONITOR_DEFAULTS, **(cfg or {})}

    # ───────────────────────── public API ─────────────────────────

    def check(
        self,
        pos: dict,
        ind: dict,
        current_premium: float,
        bar_idx: int,
    ) -> Optional[str]:
        """Return an exit reason (starts with 'monitor_') or None."""
        if not self.cfg.get("monitor_enabled", True):
            return None
        return monitor_check(pos, ind, current_premium, bar_idx, self.cfg)


# ═══════════════════════════════════════════════════════════════════
# PURE FUNCTION — used by the backtest harness
# ═══════════════════════════════════════════════════════════════════

def monitor_check(
    pos: dict,
    ind: dict,
    current_premium: float,
    bar_idx: int,
    cfg: Optional[dict] = None,
) -> Optional[str]:
    """Evaluate monitor exits for a single open position on the current bar.

    Parameters
    ----------
    pos : dict
        Open-position dict. Must contain: action, entry_bar, entry_spot,
        entry_premium. Will be mutated to track peak_premium etc.
    ind : dict
        Indicator snapshot from scoring.indicators.compute_indicators.
        Expected keys (all optional — Monitor degrades gracefully):
        close, rsi, ema9, ema21, st_direction, macd_hist, macd_signal,
        psar_bullish.
    current_premium : float
        Live option premium (LTP in live; Black-Scholes in backtest).
    bar_idx : int
        Current bar index for the trading day.
    cfg : dict, optional
        Overrides for MONITOR_DEFAULTS.

    Returns
    -------
    str or None
        Exit reason like 'monitor_prem_trail', 'monitor_reversal',
        'monitor_structural', 'monitor_emergency'. None to keep holding.
    """
    c = {**MONITOR_DEFAULTS, **(cfg or {})}
    if not c.get("monitor_enabled", True):
        return None

    action = pos.get("action")
    entry_premium = float(pos.get("entry_premium") or 0.0)
    entry_spot = float(pos.get("entry_spot") or 0.0)
    if action not in ("BUY_CALL", "BUY_PUT"):
        # Monitor currently only handles long options. Spreads use their
        # own exit logic in scoring/engine.evaluate_exit.
        return None
    if entry_premium <= 0 or current_premium <= 0:
        return None

    bars_held = bar_idx - int(pos.get("entry_bar", bar_idx))
    if bars_held < 1:
        # Seed state on entry bar and wait.
        pos.setdefault("peak_premium", entry_premium)
        pos.setdefault("underwater_bars", 0)
        pos.setdefault("in_profit_phase", False)
        pos.setdefault("mon_prev_macd_hist", float(ind.get("macd_hist", 0.0) or 0.0))
        return None

    # ── Update tracking state ──
    peak_prem = float(pos.get("peak_premium", entry_premium))
    if current_premium > peak_prem:
        peak_prem = current_premium
    pos["peak_premium"] = peak_prem

    # in_profit_phase latches True once unrealized premium >= 1.10 × entry.
    # A 10% premium gain is a meaningful "we were winning" marker.
    if not pos.get("in_profit_phase"):
        if current_premium >= entry_premium * 1.10:
            pos["in_profit_phase"] = True

    # underwater_bars counts consecutive bars where the trade is losing money
    # relative to entry premium.
    if current_premium < entry_premium:
        pos["underwater_bars"] = int(pos.get("underwater_bars", 0)) + 1
    else:
        pos["underwater_bars"] = 0

    # ── Exit 4: EMERGENCY FLOOR ──
    # Runaway loss from a profitable peak — the "+4000 → -11000" trap.
    # Checked FIRST because it's the most urgent.
    emergency_peak_mult = float(c["mon_emergency_peak_mult"])
    emergency_floor_frac = float(c["mon_emergency_floor_frac"])
    if peak_prem >= entry_premium * emergency_peak_mult:
        if current_premium <= peak_prem * emergency_floor_frac:
            return "monitor_emergency"

    # ── Exit 1: PREMIUM TRAIL ──
    # Classic trailing exit based on actual option premium, not spot.
    trail_activate_mult = float(c["mon_prem_trail_activate_mult"])
    trail_giveback_frac = float(c["mon_prem_trail_giveback_frac"])
    if peak_prem >= entry_premium * trail_activate_mult:
        if current_premium <= peak_prem * trail_giveback_frac:
            return "monitor_prem_trail"

    # ── Compute reversal score from indicators ──
    score = _reversal_score(action, pos, ind, c)

    # ── Exit 2: REVERSAL CONFIRMED (while in profit) ──
    rev_min = int(c["mon_reversal_score_min"])
    rev_min_bars = int(c["mon_reversal_min_bars_held"])
    require_profit = bool(c["mon_reversal_require_profit"])
    if bars_held >= rev_min_bars and score >= rev_min:
        if (not require_profit) or pos.get("in_profit_phase"):
            return "monitor_reversal"

    # ── Exit 3: STRUCTURAL BREAKDOWN ──
    # Trade never made it, now indicators confirm reversal.
    uw_bars_min = int(c["mon_structural_underwater_bars"])
    struct_score = int(c["mon_structural_score_min"])
    if (int(pos.get("underwater_bars", 0)) >= uw_bars_min
            and score >= struct_score):
        return "monitor_structural"

    return None


# ═══════════════════════════════════════════════════════════════════
# REVERSAL SCORE
# ═══════════════════════════════════════════════════════════════════

def _reversal_score(action: str, pos: dict, ind: dict, cfg: dict) -> int:
    """Count weighted indicator signals that have flipped against the trade."""
    score = 0

    # Signal 1: Supertrend direction against trade
    st_dir = int(ind.get("st_direction", 0) or 0)
    if action == "BUY_CALL" and st_dir == -1:
        score += int(cfg["mon_w_supertrend"])
    elif action == "BUY_PUT" and st_dir == 1:
        score += int(cfg["mon_w_supertrend"])

    # Signal 2: MACD histogram cross (zero-line cross against trade)
    curr_hist = float(ind.get("macd_hist", 0.0) or 0.0)
    prev_hist = float(pos.get("mon_prev_macd_hist", curr_hist))
    if action == "BUY_CALL" and prev_hist >= 0 and curr_hist < 0:
        score += int(cfg["mon_w_macd_cross"])
    elif action == "BUY_PUT" and prev_hist <= 0 and curr_hist > 0:
        score += int(cfg["mon_w_macd_cross"])
    pos["mon_prev_macd_hist"] = curr_hist

    # Signal 3: EMA 9/21 cross against trade
    ema9 = float(ind.get("ema9", 0.0) or 0.0)
    ema21 = float(ind.get("ema21", 0.0) or 0.0)
    if ema9 and ema21:
        if action == "BUY_CALL" and ema9 < ema21:
            score += int(cfg["mon_w_ema_cross"])
        elif action == "BUY_PUT" and ema9 > ema21:
            score += int(cfg["mon_w_ema_cross"])

    # Signal 4: RSI exhaustion — price has run too far in trade direction
    # and is due for a reversal.
    rsi = float(ind.get("rsi", 50.0) or 50.0)
    if action == "BUY_CALL" and rsi >= float(cfg["mon_rsi_call_exhaust"]):
        score += int(cfg["mon_w_rsi_exhaust"])
    elif action == "BUY_PUT" and rsi <= float(cfg["mon_rsi_put_exhaust"]):
        score += int(cfg["mon_w_rsi_exhaust"])

    # Signal 5: Parabolic SAR flipped against trade
    psar_bull = ind.get("psar_bullish")
    if psar_bull is not None:
        if action == "BUY_CALL" and not psar_bull:
            score += int(cfg["mon_w_psar_flip"])
        elif action == "BUY_PUT" and psar_bull:
            score += int(cfg["mon_w_psar_flip"])

    return score


__all__ = ["LiveTradeMonitor", "monitor_check", "MONITOR_DEFAULTS"]
