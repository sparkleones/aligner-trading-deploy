"""V14 Enh-Tuned canonical config — shared by backtest + live.

This is THE ONE config dict. Both the backtester and the live agent import it.
All time values are in 5-min bar counts (bar_idx units).

Tuning history (Rs 200K capital, Jul 2024 — Jan 2025, equity compounded):
  Round 1: V3_EntryWindows     2.30x (3 windows: morning, midday, afternoon)
  Round 2: R4_2Windows         3.48x (2 windows: morning + afternoon only)
  Round 3: Confirmed robust    3.48x (12/19 variations identical — no overfitting)
  Round 4: T20_All+Cap30       9.84x (data-driven: lot cap, Wed block, VIX scaling,
                                       ORB boost, no zero-hero, skip bar 57-58)
  Round 5: R5_PSAR+ATR+Cap27  12.26x (algo trading methods: Parabolic SAR confluence,
                                       ATR-normalized sizing, lot cap at NSE freeze limit;
                                       PF 2.58, MaxDD 52.9%, Calmar 22.76)
"""

V14_CONFIG = {
    "name": "V14_LEGACY_Restored",

    # ── Bar resolution ──
    "bar_interval_min": 5,

    # ── Entry windows: empty = no window restriction (LEGACY behavior) ──
    "entry_windows_bars": [],              # Empty = ALL bars allowed (gated by block_late_entries below)

    # ── Cooldown ──
    "cooldown_bars": 2,              # 10 min

    # ── Confidence ──
    "min_confidence": 0.35,
    "min_confidence_filter": 0.25,

    # ── Scoring thresholds (LEGACY) ──
    "put_score_min": 4.0,
    "call_score_min": 5.0,

    # ── Trail stops ENABLED (LEGACY: tight trails free position slots for new entries) ──
    # The trail trade itself loses money (~-Rs283K on 6 trades), but freeing
    # position slots lets higher-quality entries reach time_exit and earn
    # ~+Rs340K — net positive (3.7x with trails ON vs 3.4x with trails OFF).
    "trail_pct_put": 0.015,          # 1.5%
    "trail_pct_call": 0.008,         # 0.8%
    "min_hold_trail_put": 24,        # 24 bars × 5 min = 120 min
    "min_hold_trail_call": 12,       # 12 bars × 5 min = 60 min
    "max_hold_put": 60,              # 300 min / 5 = 60 bars
    "max_hold_call": 54,             # 270 min / 5 = 54 bars

    # ── ATR trail DISABLED (LEGACY: fixed % wins decisively) ──
    "use_atr_trail": False,
    "use_chandelier_exit": False,

    # ── Trade limits (LEGACY: tight caps prevent over-exposure) ──
    "max_trades_per_day": 7,
    "max_concurrent": 3,
    "block_call_4th_hour": True,     # LEGACY: 4th hour CALLs (bar 45-60) consistently lose
    "block_late_entries": 61,        # LEGACY: 305 min cutoff (~2:20 PM) — late entries get 0-1 bars to develop
    "avoid_windows_bars": [(33, 57)], # LEGACY: lunch hour block (12:00-14:00 — choppy)

    # ── Day-of-week filter: empty (VIX lot scaling handles risk) ──
    "avoid_days": [],

    # ── Lot cap (NSE freeze limit = 27 lots; tighter cap improves risk-adjusted) ──
    "max_lots_cap": 27,

    # ── VIX-adaptive lot scaling (most impactful improvement: 3.13x → 7.32x) ──
    "vix_lot_scaling": True,
    "vix_below13_mult": 0.3,        # VIX <13: 10% WR, 0.21 PF → near-zero lots
    "vix_14_15_mult": 0.5,          # VIX 14-15: PF 0.72 → half lots
    "vix_15_17_mult": 1.5,          # VIX 15-17: PF 5.41 → boost lots
    "vix_17plus_mult": 2.0,         # VIX 17+: strong signals → double lots

    # ── ORB lot boost (75% WR, PF 12.16 → give more capital) ──
    "orb_lot_mult": 2.0,

    # ── Zero-to-hero (R6: enabled — allows aggressive OTM entries on strong signals) ──
    "disable_zero_hero": False,

    # ── Confluence filters (LEGACY) ──
    "use_vwap_filter": True,
    "use_squeeze_filter": True,
    "use_rsi_hard_gate": True,
    "rsi_call_min": 50,              # LEGACY: lowered from 55 — widen entry window (kill ceiling at 60 blocks bad zone)
    "rsi_put_max": 50,               # LEGACY: relaxed from 40 — RSI rarely hits <40 on 5-min bars
    "rsi_call_kill_ceiling": 60,     # LEGACY: 23.3% WR, -Rs722K loss when CALLs entered with RSI 60-72

    # ── Lot modifiers (base) ──
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

    # ── VIX bounds (LEGACY: floor 11, V15 raises to 13) ──
    "vix_floor": 11,
    "vix_ceil": 35,

    # ── Hard stop DISABLED ──
    "hard_sl_pct": 0.0,

    # ── Consecutive loss throttle DISABLED ──
    "max_consecutive_losses": 99,
    "loss_cooldown_bars": 0,

    # ── Drawdown lot scale DISABLED ──
    "drawdown_lot_scale": False,

    # ── StdDev filter OFF ──
    "use_stddev_filter": False,

    # ── Gap reversal filter ──
    "gap_reversal_filter": True,
    "gap_threshold_pct": 0.004,

    # ── Gap entry control ──
    "disable_gap_entry": True,

    # ── ADX scoring ──
    "adx_choppy_min": 25,
    "adx_choppy_max": 35,
    "adx_choppy_mult": 0.8,
    "adx_weak_threshold": 18,
    "adx_weak_mult": 0.6,
    "adx_strong_threshold": 35,

    # ── Trail profit gate ──
    "trail_min_profit_pct": 0.003,

    # ── PCR filter ──
    "use_pcr_filter": True,
    "pcr_bullish_max": 0.7,
    "pcr_bearish_min": 1.6,

    # ── OI levels ──
    "use_oi_levels": True,
    "oi_proximity_pct": 0.003,

    # ── IV percentile scaling ──
    "use_iv_pctile_scaling": True,
    "iv_pctile_high": 75,
    "iv_pctile_high_mult": 0.7,
    "iv_pctile_low": 25,
    "iv_pctile_low_mult": 1.3,

    # ── IV percentile entry GATE (block entries when options too expensive) ──
    # Research: buying options at IV >80th pctile = IV crush risk, net negative
    # Conservative filter — only blocks, never adds entries
    "use_iv_pctile_gate": True,
    "iv_pctile_gate_threshold": 80,       # Block entries when IV > 80th percentile

    # ── OI Change tracking (delta OI buildup/unwinding) ──
    # Research: Quantsapp "Trap Indicator" concept — detect when writers get trapped
    # OI buildup at a level that then breaks causes aggressive covering
    "use_oi_change_scoring": True,
    "oi_change_buildup_score": 0.5,       # Score for fresh OI buildup confirming direction
    "oi_change_unwinding_score": -0.3,    # Penalty for OI unwinding (positions closing)

    # ── Enhanced expiry day max pain convergence ──
    # Research: Max pain convergence most effective after 1:30 PM on expiry day
    "use_expiry_max_pain_boost": True,
    "expiry_max_pain_score": 1.0,         # Extra score for max pain pull on expiry
    "expiry_max_pain_after_bar": 57,      # After 2:00 PM (bar 57 = 14:00)

    # ── Theta exit ──
    "theta_exit_enabled": True,
    "theta_exit_monday_bar": 69,

    # ── Zero-to-hero (kept for reference but disabled via disable_zero_hero) ──
    "zero_hero_target_pct": 0.02,
    "zero_hero_stop_pct": 0.008,
    "zero_hero_trail_pct": 0.008,
    "zero_hero_trail_activation": 0.01,
    "zero_hero_time_bars": 30,
    "zero_hero_max_lots": 3,
    "zero_hero_strike_offset": 200,

    # ── EOD close bar ──
    "eod_close_bar": 72,

    # ══════════════════════════════════════════════════════════
    # ROUND 5: Algo Trading Methodology Improvements
    # ══════════════════════════════════════════════════════════

    # ── Parabolic SAR confluence filter (R5 best risk-adjusted improvement) ──
    # Filters entries against SAR direction: don't buy calls in SAR downtrend
    # Impact: 48.9% WR (from 48.0%), PF 2.46, MaxDD 48.3% (from 54.0%)
    "use_psar_confluence": True,

    # ── ATR-normalized lot sizing (R5 biggest return improvement) ──
    # Sizes positions inversely to ATR: low ATR → more lots, high ATR → fewer
    # Normalizes risk per trade regardless of market volatility
    # Impact: 11.24x standalone, 12.26x combined with PSAR filter
    "use_atr_sizing": True,
    "atr_reference": 80,    # "Normal" ATR for NIFTY 5-min bars

    # ══════════════════════════════════════════════════════════
    # SMART STRIKE SELECTION (replaces fixed ATM strike picking)
    # ══════════════════════════════════════════════════════════
    # Scores each strike on delta, IV, OI, liquidity, bid-ask,
    # premium efficiency, and gamma to find optimal entry strike.

    "use_smart_strike": True,         # Enable smart strike selector
    "smart_strike_config": {
        "search_range_strikes": 8,    # ±8 strikes around ATM
        # BACKTEST PROVEN (52 trades, Jul 2024 - Jan 2025):
        #   ITM_1 (d=0.60): 15.87x | ATM (d=0.48): 14.56x | OTM_1 (d=0.36): 11.85x
        # ITM_1 wins by Rs +2.6L over ATM. Higher delta = more profit always.
        "target_delta_min": 0.50,     # Min acceptable |delta|
        "target_delta_max": 0.75,     # Max acceptable |delta|
        "target_delta_sweet": 0.60,   # Ideal |delta| — 1 strike ITM
        "w_delta": 0.35,              # Weight: delta proximity (dominant)
        "w_iv": 0.05,                 # Weight: less important for ITM
        "w_liquidity": 0.25,          # Weight: OI + volume (fills matter)
        "w_spread": 0.15,             # Weight: bid-ask tightness
        "w_premium_eff": 0.05,        # Weight: premium per delta
        "w_gamma": 0.15,              # Weight: gamma edge (expiry)
        "min_oi": 50_000,             # Min OI to consider strike
        "min_volume": 5_000,          # Min volume
        "max_spread_pct": 0.05,       # Max bid-ask spread %
        "max_pain_avoid_range": 100,  # Avoid ±100pts of max pain on expiry
        "max_pain_penalty": 0.3,      # Score penalty near max pain
        "expiry_gamma_boost": 2.0,    # Gamma weight boost on expiry
        "high_vix_threshold": 17.0,   # VIX > 17 → conservative delta
        "low_vix_threshold": 13.0,    # VIX < 13 → ATM only
    },
}


# ══════════════════════════════════════════════════════════════
# V14_BASELINE — V14_CONFIG with every "added" feature forced OFF
# ══════════════════════════════════════════════════════════════
# Diagnostic config: matches the LEGACY _score_entry / _passes_confluence
# logic in v14_live_agent.py by disabling all post-LEGACY additions:
#   - R5: psar/kama/ha/donchian/connors/macd-crossover/vwap-bands
#   - V15: volume/obv/momentum/ema_stack/session/rv_iv/bear_div/streak
#   - V16: regime filter / obv divergence / volume climax
#   - Research: oi_change_scoring / expiry_max_pain / iv_pctile_gate
#
# Used to A/B test whether the regression in v14_unified_backtest.py
# is caused by the added scoring features vs the V14 base scoring itself.

V14_BASELINE_CONFIG = {**V14_CONFIG, **{
    "name": "V14_BASELINE_LegacyOnly",

    # ── Disable R5 additions ──
    "use_psar_confluence": False,
    "use_atr_sizing": False,
    "use_connors_rsi": False,
    "use_kama_filter": False,
    "use_kama_confluence": False,
    "use_heikin_ashi": False,
    "use_ha_confluence": False,
    "use_donchian": False,
    "use_psar": False,
    "use_macd_crossover": False,
    "use_vwap_bands": False,

    # ── Disable IV percentile gate (added research feature) ──
    "use_iv_pctile_gate": False,

    # ── Disable OI change scoring (added research feature) ──
    "use_oi_change_scoring": False,

    # ── Disable expiry max pain boost (added research feature) ──
    "use_expiry_max_pain_boost": False,
}}


# ══════════════════════════════════════════════════════════════
# V15 CONFIG — Enhanced V14 R5 with research-driven improvements
# ══════════════════════════════════════════════════════════════
#
# Changes from V14 R5:
#   1. Volume-confirmed entries (OBV + volume spike)
#   2. Momentum acceleration scoring
#   3. Multi-TF EMA stack alignment
#   4. Session-specific weights (morning trend / afternoon reversion)
#   5. RV/IV entry quality filter
#   6. Stale trade exit (45 min no-progress kill)
#   7. Chandelier exit (ATR-adaptive trailing)
#   8. Momentum exhaustion exit (RSI extreme take-profit)
#   9. Volume + velocity confluence filters
#  10. Bearish RSI divergence scoring (was missing in main scoring)

V15_CONFIG = {**V14_CONFIG, **{
    "name": "V15_AvoidMonWed",

    # ══════════════════════════════════════════════════════════
    # V15 DAY-OF-WEEK FILTER (proven over 21 months Jul24-Apr26)
    # ══════════════════════════════════════════════════════════
    # Block Monday (0) and Wednesday (2). Without this filter the
    # baseline V15 actually LOSES money over the full 21-month
    # window (-0.00x). With it: 18.53x, +Rs 35L, 161 trades, 40.4% WR.
    # Mon = unpredictable weekend gap, Wed = mid-week chop. Validated
    # in 4 separate periods (1 IS + 3 OOS) — V3 is the only filter
    # that's POSITIVE in every OOS period:
    #   IS Jul24-Jan25:  10.83x | OOS1 Feb-Jul25: 1.81x
    #   OOS2 Aug25-Jan26: 2.96x | OOS3 Feb-Apr26: 2.25x
    "avoid_days": [0, 2],       # Restored: Mon+Wed blocked (validated V17_PROD_ONLY 28.9x config)

    # ══════════════════════════════════════════════════════════
    # V15 RISK MANAGEMENT (backtest-proven)
    # ══════════════════════════════════════════════════════════

    # ── VIX floor at 12 (lowered from 13 post-Sep-2025 regime sweep:
    #    21-mo sweep on avoid_days=[0,2] → floor=12 gives 29.0x full / +Rs 14.2L post-Sep
    #    vs floor=13's 28.1x / +Rs 9.9L. PF post-Sep 1.73 (was 1.60). See OVERNIGHT_REPORT_2026-04-22.) ──
    "vix_floor": 12,

    # ── Losing streak sizing (after 3+ consecutive losses, reduce lots;
    #    +0.03x return, -2% MaxDD, Calmar 23.3→24.1) ──
    "use_streak_sizing": True,

    # ══════════════════════════════════════════════════════════
    # V15 SCORING ENHANCEMENTS (all backtest-proven)
    # ══════════════════════════════════════════════════════════

    # ── Volume confirmation (OBV + spike detection) ──
    "use_volume_confirmation": True,
    "low_volume_mult": 0.8,           # Dampen scores on weak volume

    # ── OBV trend scoring ──
    "use_obv_confirmation": True,
    "obv_score": 0.8,                 # OBV aligned with direction

    # ── Momentum acceleration ──
    "use_momentum_accel": True,
    "momentum_accel_score": 0.7,      # Accelerating momentum boost

    # ── Multi-TF EMA stack (5 > 9 > 21 = strong trend) ──
    "use_ema_stack": True,
    "ema_stack_score": 1.2,           # All EMAs aligned boost

    # ── Session-specific weights ──
    "use_session_weights": True,
    "morning_trend_boost": 1.1,       # Morning power hour: boost trend signals
    "afternoon_adx_min": 28,          # Afternoon: require stronger ADX

    # ── RV/IV entry quality ──
    "use_rv_iv_filter": True,
    "rv_iv_min": 0.8,                 # Below this = IV overpriced, dampen
    "rv_iv_boost_threshold": 1.3,     # Above this = market moving, boost

    # ── Bearish RSI divergence (complement to bullish) ──
    "use_rsi_bear_div": True,
    "rsi_bear_div_score": 1.5,

    # ══════════════════════════════════════════════════════════
    # V15 EXIT IMPROVEMENTS
    # ══════════════════════════════════════════════════════════

    # ── Stale trade exit — DISABLED (25% WR, lost -2.3L in backtest) ──
    "use_stale_exit": False,
    "stale_exit_bars": 9,
    "stale_exit_pct": 0.003,

    # ── Chandelier exit — DISABLED (0% WR, lost -2.0L in backtest) ──
    "use_chandelier_exit_v15": False,
    "chandelier_atr_mult": 2.5,
    "chandelier_min_bars": 6,

    # ── Momentum exhaustion exit — DISABLED ──
    # 63.6% WR but avg +Rs 9K/trade vs time_exit avg +Rs 44K.
    # Cuts winners short: trades that would be +44-77K on time/EOD
    # are closed early for +9K. Net effect: -Rs 30L vs V14 baseline.
    "use_momentum_exhaustion_exit": False,
    "exhaustion_rsi_call": 75,        # Take profit on calls when RSI > 75
    "exhaustion_rsi_put": 25,         # Take profit on puts when RSI < 25

    # ══════════════════════════════════════════════════════════
    # V15 CONFLUENCE FILTERS — DISABLED (no impact in backtest)
    # ══════════════════════════════════════════════════════════

    # ── Volume entry filter ──
    "use_volume_entry_filter": False,
    "min_volume_ratio": 0.7,

    # ── Velocity agreement filter ──
    "use_velocity_filter": False,
    "min_velocity_pct": 0.05,

    # ══════════════════════════════════════════════════════════
    # V17 BTST FAVORABILITY (MIS vs NRML selector) — V17_PROD_ONLY
    # ══════════════════════════════════════════════════════════
    # Indicator-driven decision to carry a trade overnight (NRML)
    # instead of squaring off intraday (MIS). Works ANY day of the
    # week — the old Friday-only BTST hack is gone. V15 keeps its
    # avoid_days=[0,2] filter, so entries only happen Tue/Thu/Fri;
    # this layer decides whether those entries carry overnight.
    #
    # Backtest proof (21 months Jul24-Apr26):
    #   V15 baseline:   18.53x | +Rs 35L | 161 trades | 40.4% WR
    #   V17_PROD_ONLY:  28.89x | +Rs 56L | 161 trades | 40.4% WR
    #     → 8 BTST carries (5W/3L, 62.5% WR, +Rs 20L net)
    #     → Top 2 carries: Trump tariff (+Rs 14.7L) + 2026-03-20 (+Rs 7.2L)
    #     → Remaining 6 carries: ~flat (+Rs 0.7L)
    # The decision is gap-continuation focused: trend-aligned, strong
    # ADX, closing near HOD/LOD, day moving our way, sane VIX regime.
    "use_v17_dynamic_product": True,
    "v17_btst_bar_min": 30,              # Need late-session confirmation
    "v17_btst_dte_min": 2,               # Enough runway for theta vs gap risk
    "v17_btst_adx_min": 18,              # Trend strength floor
    "v17_btst_rsi_put_min": 25,          # Don't carry exhausted puts
    "v17_btst_rsi_call_max": 75,         # Don't carry exhausted calls
    "v17_btst_close_put_max": 0.50,      # PUT: close at or below mid of day range
    "v17_btst_close_call_min": 0.50,     # CALL: close at or above mid
    "v17_btst_day_chg_put_max": 0.005,   # PUT: day not strongly green
    "v17_btst_day_chg_call_min": -0.005, # CALL: day not strongly red
    "v17_btst_vix_min": 11,              # Too quiet → gap unlikely
    "v17_btst_vix_max": 30,              # Too chaotic → gap direction unreliable
}}


# ══════════════════════════════════════════════════════════════
# V16 CONFIG — Regime-aware improvements over V15
# ══════════════════════════════════════════════════════════════
#
# Changes from V15:
#   1. ADX + BB Width regime filter (dampen signals in ranging markets)
#   2. OBV divergence scoring (price-volume divergence → reversal)
#   3. Volume climax reversal boost
#   4. ORB directional confirmation (align entries with opening range)
#   5. Day-of-week parameter optimization (different hold/entry per day)
#   6. Tighter VIX regime bucketing

# ══════════════════════════════════════════════════════════════
# V17 CONFIG — Dynamic regime gate + MIS/NRML selector
# ══════════════════════════════════════════════════════════════
#
# Hypothesis: Replace the blunt avoid_days=[0,2] calendar filter
# with an indicator-driven regime gate. Allow Mon/Wed entries when
# the tape actually looks tradeable; block any day (Tue/Thu/Fri too)
# when it doesn't. Additionally, dynamically pick MIS vs NRML at
# entry time based on trend strength + conviction + time-of-day,
# so high-conviction late-day trend entries can carry overnight
# instead of being force-closed by MIS auto-squareoff.
#
# BACKTEST FLAGS ONLY — no live engine changes until validated.
#
V17_CONFIG = {**V15_CONFIG, **{
    "name": "V17_DynamicRegime",

    # ── Turn OFF the calendar filter ──
    "avoid_days": [],

    # ══════════════════════════════════════════════════════════
    # V17 REGIME GATE (replaces avoid_days)
    # ══════════════════════════════════════════════════════════
    # Bar-by-bar check at every entry attempt. Blocks when tape
    # is in chop/squeeze/dead-vol regime. Deliberately permissive
    # — goal is to reject true chop, not filter hard. Anything
    # tighter kills V15's big composite-entry winners.
    "use_v17_regime_gate": True,
    "v17_gate_adx_hard": 0,             # Disabled (kills composite entries)
    "v17_gate_adx_chop": 18,            # ADX below + squeeze_on = block
    "v17_gate_atr_pct_min": 0.0010,     # ATR / spot < 0.10% = block
    "v17_gate_vix_low": 11,             # VIX below = no movement expected
    "v17_gate_vix_high": 33,            # VIX above = whipsaw regime
    "v17_gate_donchian_min": 0.0,       # Disabled
    "v17_gate_min_conf": 0.0,           # Disabled — composite entries have low conf

    # ══════════════════════════════════════════════════════════
    # V17 MON/WED CONDITIONAL ENTRY GATE
    # ══════════════════════════════════════════════════════════
    # Don't hard-block Mon/Wed like V15. Instead, require indicator
    # confirmation: strong ADX + aligned trend + real volatility +
    # not in squeeze + sane VIX. Tue/Thu/Fri are untouched.
    "use_v17_monwed_gate": True,
    "v17_monwed_bar_min": 12,           # Wait 1 hour for trend to form
    "v17_monwed_adx_min": 22,           # Strong trend required
    "v17_monwed_atr_pct_min": 0.0015,   # Real volatility present
    "v17_monwed_vix_min": 12,
    "v17_monwed_vix_max": 30,

    # ══════════════════════════════════════════════════════════
    # V17 BTST FAVORABILITY (indicator-based dynamic product selector)
    # ══════════════════════════════════════════════════════════
    # Looser thresholds (vs prior tight version) to capture more
    # BTST opportunities — user feedback: current filter is too
    # restrictive, there are genuine BTST carries being missed.
    "use_v17_dynamic_product": True,

    "v17_btst_bar_min": 30,             # Entries from bar 30+ (12:45 PM+)
    "v17_btst_dte_min": 2,

    "v17_btst_adx_min": 18,             # Looser ADX floor

    "v17_btst_rsi_put_min": 25,         # Allow slightly more oversold PUTs
    "v17_btst_rsi_call_max": 75,

    "v17_btst_close_put_max": 0.50,     # PUT: bottom 50% of day range
    "v17_btst_close_call_min": 0.50,    # CALL: top 50% of day range

    "v17_btst_day_chg_put_max": 0.005,  # PUT: day change ≤ +0.5% (tolerate choppy red)
    "v17_btst_day_chg_call_min": -0.005,

    "v17_btst_vix_min": 11,
    "v17_btst_vix_max": 30,

    # ══════════════════════════════════════════════════════════
    # V17 MON/WED TIGHT EXITS (experiment)
    # ══════════════════════════════════════════════════════════
    # Hypothesis: Mon/Wed entries reverse more often within our normal
    # holding window. Tighter max_hold / trail / chandelier on those days
    # might escape the reversal pattern. Enabled per-variant via CLI.
    "use_monwed_tight_exits": False,
    "monwed_max_hold_put": 24,          # 2 hrs (vs 60 normally)
    "monwed_max_hold_call": 24,         # 2 hrs (vs 54 normally)
    "monwed_trail_pct_put": 0.007,      # 0.7% (vs 1.5%)
    "monwed_trail_pct_call": 0.004,     # 0.4% (vs 0.8%)
    "monwed_chandelier_mult": 1.5,      # 1.5x ATR (vs 2.5)
    "monwed_min_hold_trail_put": 6,     # 30 min (vs 24) — trail can fire earlier
    "monwed_min_hold_trail_call": 6,    # 30 min (vs 12)
}}


V16_CONFIG = {**V15_CONFIG, **{
    "name": "V16_RegimeAware",

    # ══════════════════════════════════════════════════════════
    # V16 REGIME DETECTION
    # ══════════════════════════════════════════════════════════

    # ── ADX + BB Width regime filter ──
    # Block/dampen entries when ADX is low AND BB width is compressed
    # = market is ranging, options will chop without directional move
    "use_regime_filter": True,
    "regime_adx_min": 20,              # ADX below this = no trend
    "regime_bb_pctile_min": 30,        # BB width below 30th percentile = compressed
    "regime_ranging_mult": 0.5,        # Halve scores in ranging regime

    # ── OBV divergence (price vs volume disagreement = reversal) ──
    "use_obv_divergence": True,
    "obv_div_score": 1.5,              # Boost when divergence detected

    # ── Volume climax reversal ──
    "use_volume_climax": True,
    "volume_climax_score": 1.0,        # Extra boost when climax + divergence

    # ══════════════════════════════════════════════════════════
    # V16 ORB DIRECTIONAL CONFIRMATION (implemented in backtester)
    # ══════════════════════════════════════════════════════════
    # Require entries to align with opening range breakout direction
    # Morning window (bars 3-15): use 9:15-9:30 ORB direction
    # Afternoon window (bars 59-69): use 12:00-14:00 range direction
    "use_orb_confirmation": True,

    # ══════════════════════════════════════════════════════════
    # V16 DAY-OF-WEEK OPTIMIZATION
    # ══════════════════════════════════════════════════════════
    # Thursday (expiry) needs different parameters
    "use_dow_optimization": True,
    "thursday_max_hold_put": 36,       # 3 hrs (vs 5 hrs normally) — faster theta decay
    "thursday_max_hold_call": 30,      # 2.5 hrs
    "thursday_score_boost": 1.5,       # Require higher conviction on expiry
    "thursday_block_afternoon": True,  # No afternoon entries on expiry
    "monday_hold_boost": 1.1,          # Monday: slightly longer holds (more time value)
}}

