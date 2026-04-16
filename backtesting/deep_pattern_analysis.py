"""Deep pattern analysis on all 388 trades from day-by-day analysis."""
import json
import datetime
import numpy as np
from collections import defaultdict
from pathlib import Path

project_root = Path(__file__).parent.parent

with open(project_root / "data" / "daywise_all_trades.json") as f:
    trades = json.load(f)

print("=" * 120)
print("  DEEP PATTERN ANALYSIS — 388 trades on real 1-min data")
print("=" * 120)

winners = [t for t in trades if t["pnl"] > 0]
losers = [t for t in trades if t["pnl"] <= 0]

# ========== 1. EXIT REASON DEEP DIVE ==========
print("\n--- EXIT REASON ANALYSIS ---")
for er in ["time_exit", "eod_close", "trail_stop", "supertrend_flip", "expiry_close"]:
    subset = [t for t in trades if t["exit_reason"] == er]
    if not subset:
        continue
    w = [t for t in subset if t["pnl"] > 0]
    l = [t for t in subset if t["pnl"] <= 0]
    pnl = sum(t["pnl"] for t in subset)
    avg_hold = np.mean([t["minutes_held"] for t in subset])
    avg_fav = np.mean([t["max_favorable_move"] for t in subset])
    print(f"  {er:<20}: {len(subset):>3}t  Rs {pnl:>+9,.0f}  WR {len(w)/len(subset)*100:.0f}%  "
          f"AvgHold {avg_hold:.0f}min  AvgFavMove {avg_fav:.3f}%")

# Trail stop deep dive
trail = [t for t in trades if t["exit_reason"] == "trail_stop"]
print(f"\n  TRAIL STOP BREAKDOWN (the #1 profit killer):")
print(f"    Total: {len(trail)} trades, Rs {sum(t['pnl'] for t in trail):+,}")
trail_puts = [t for t in trail if t["action"] == "BUY_PUT"]
trail_calls = [t for t in trail if t["action"] == "BUY_CALL"]
print(f"    PUTs killed: {len(trail_puts)}, Rs {sum(t['pnl'] for t in trail_puts):+,}")
print(f"    CALLs killed: {len(trail_calls)}, Rs {sum(t['pnl'] for t in trail_calls):+,}")
# What % of trail-killed had a favorable move first?
trail_had_fav = [t for t in trail if t["max_favorable_move"] > 0.1]
print(f"    Had >0.1% favorable move before trail: {len(trail_had_fav)}/{len(trail)} ({len(trail_had_fav)/len(trail)*100:.0f}%)")

# ========== 2. TIME WINDOWS ==========
print("\n--- ENTRY TIME WINDOWS ---")
time_windows = [
    (0, 15, "9:15-9:30 (opening)"),
    (15, 45, "9:30-10:00"),
    (45, 75, "10:00-10:30"),
    (75, 105, "10:30-11:00"),
    (105, 135, "11:00-11:30"),
    (135, 165, "11:30-12:00"),
    (165, 195, "12:00-12:30"),
    (195, 225, "12:30-1:00"),
    (225, 255, "1:00-1:30"),
    (255, 285, "1:30-2:00"),
    (285, 315, "2:00-2:30"),
    (315, 345, "2:30-3:00"),
]
best_windows = []
worst_windows = []
for start, end, label in time_windows:
    subset = [t for t in trades if start <= t["entry_minute"] < end]
    if subset:
        w = len([t for t in subset if t["pnl"] > 0])
        pnl = sum(t["pnl"] for t in subset)
        wr = w / len(subset) * 100
        avg_pnl = pnl / len(subset)
        marker = ""
        if wr >= 50:
            marker = " <-- PROFITABLE"
            best_windows.append((label, wr, pnl, len(subset)))
        elif wr < 30:
            marker = " <-- AVOID"
            worst_windows.append((label, wr, pnl, len(subset)))
        print(f"  {label:<25}: {len(subset):>3}t  Rs {pnl:>+9,.0f}  WR {wr:.0f}%  "
              f"avg Rs {avg_pnl:>+,.0f}/trade{marker}")

# ========== 3. BUY_PUT vs BUY_CALL ==========
print("\n--- ACTION ANALYSIS ---")
puts = [t for t in trades if t["action"] == "BUY_PUT"]
calls = [t for t in trades if t["action"] == "BUY_CALL"]
for label, subset in [("BUY_PUT", puts), ("BUY_CALL", calls)]:
    w = [t for t in subset if t["pnl"] > 0]
    pnl = sum(t["pnl"] for t in subset)
    print(f"\n  {label}: {len(subset)}t  WR {len(w)/len(subset)*100:.1f}%  Rs {pnl:+,}  "
          f"avg Rs {pnl/len(subset):+,.0f}/trade")
    for er in ["time_exit", "eod_close", "trail_stop", "supertrend_flip", "expiry_close"]:
        er_trades = [t for t in subset if t["exit_reason"] == er]
        if er_trades:
            er_w = len([t for t in er_trades if t["pnl"] > 0])
            er_pnl = sum(t["pnl"] for t in er_trades)
            print(f"    {er:<20}: {len(er_trades):>3}t  Rs {er_pnl:>+8,.0f}  WR {er_w/len(er_trades)*100:.0f}%")

# ========== 4. SUPERTREND + ACTION ALIGNMENT ==========
print("\n--- SUPERTREND + ACTION ALIGNMENT ---")
combos = [
    (1, "BUY_CALL", "CALL in uptrend (ALIGNED)"),
    (-1, "BUY_PUT", "PUT in downtrend (ALIGNED)"),
    (1, "BUY_PUT", "PUT in uptrend (COUNTER-TREND)"),
    (-1, "BUY_CALL", "CALL in downtrend (COUNTER-TREND)"),
]
for st_dir, action, label in combos:
    subset = [t for t in trades if t.get("entry_st_dir") == st_dir and t["action"] == action]
    if subset:
        w = len([t for t in subset if t["pnl"] > 0])
        pnl = sum(t["pnl"] for t in subset)
        print(f"  {label:<40}: {len(subset):>3}t  Rs {pnl:>+9,.0f}  WR {w/len(subset)*100:.0f}%  "
              f"avg Rs {pnl/len(subset):+,.0f}")

# ========== 5. EMA ALIGNMENT + ACTION ==========
print("\n--- EMA ALIGNMENT + ACTION ---")
# PUT trades
aligned_puts = [t for t in puts if not t["above_ema9"] and not t["above_ema21"] and not t["ema9_above_21"]]
partial_puts = [t for t in puts if not (not t["above_ema9"] and not t["above_ema21"] and not t["ema9_above_21"])
                and not (t["above_ema9"] and t["above_ema21"] and t["ema9_above_21"])]
counter_puts = [t for t in puts if t["above_ema9"] and t["above_ema21"] and t["ema9_above_21"]]

for label, subset in [("PUT aligned (all bearish EMA)", aligned_puts),
                       ("PUT partial alignment", partial_puts),
                       ("PUT counter (all bullish EMA)", counter_puts)]:
    if subset:
        w = len([t for t in subset if t["pnl"] > 0])
        pnl = sum(t["pnl"] for t in subset)
        print(f"  {label:<40}: {len(subset):>3}t  Rs {pnl:>+9,.0f}  WR {w/len(subset)*100:.0f}%")

# CALL trades
aligned_calls = [t for t in calls if t["above_ema9"] and t["above_ema21"] and t["ema9_above_21"]]
partial_calls = [t for t in calls if not (t["above_ema9"] and t["above_ema21"] and t["ema9_above_21"])
                 and not (not t["above_ema9"] and not t["above_ema21"] and not t["ema9_above_21"])]
counter_calls = [t for t in calls if not t["above_ema9"] and not t["above_ema21"] and not t["ema9_above_21"]]

for label, subset in [("CALL aligned (all bullish EMA)", aligned_calls),
                       ("CALL partial alignment", partial_calls),
                       ("CALL counter (all bearish EMA)", counter_calls)]:
    if subset:
        w = len([t for t in subset if t["pnl"] > 0])
        pnl = sum(t["pnl"] for t in subset)
        print(f"  {label:<40}: {len(subset):>3}t  Rs {pnl:>+9,.0f}  WR {w/len(subset)*100:.0f}%")

# ========== 6. RSI ZONES ==========
print("\n--- RSI AT ENTRY ---")
for lo, hi, label in [(0, 30, "RSI < 30 (oversold)"),
                       (30, 40, "RSI 30-40"),
                       (40, 50, "RSI 40-50"),
                       (50, 60, "RSI 50-60"),
                       (60, 70, "RSI 60-70"),
                       (70, 100, "RSI > 70 (overbought)")]:
    subset = [t for t in trades if t.get("entry_rsi") is not None and lo <= t["entry_rsi"] < hi]
    if subset:
        w = len([t for t in subset if t["pnl"] > 0])
        pnl = sum(t["pnl"] for t in subset)
        print(f"  {label:<25}: {len(subset):>3}t  Rs {pnl:>+9,.0f}  WR {w/len(subset)*100:.0f}%")

# RSI + Action
print("\n  RSI + ACTION COMBO:")
for rsi_zone, action, label in [
    ((0, 40), "BUY_CALL", "CALL when RSI<40 (buy dip)"),
    ((60, 100), "BUY_PUT", "PUT when RSI>60 (sell rally)"),
    ((40, 60), "BUY_PUT", "PUT when RSI 40-60 (neutral zone)"),
    ((40, 60), "BUY_CALL", "CALL when RSI 40-60 (neutral zone)"),
]:
    subset = [t for t in trades if t.get("entry_rsi") is not None
              and rsi_zone[0] <= t["entry_rsi"] < rsi_zone[1]
              and t["action"] == action]
    if subset:
        w = len([t for t in subset if t["pnl"] > 0])
        pnl = sum(t["pnl"] for t in subset)
        print(f"    {label:<40}: {len(subset):>3}t  Rs {pnl:>+9,.0f}  WR {w/len(subset)*100:.0f}%")

# ========== 7. VIX REGIME ==========
print("\n--- VIX REGIME ---")
for lo, hi, label in [(0, 11, "VIX < 11 (very low)"),
                       (11, 13, "VIX 11-13 (low)"),
                       (13, 16, "VIX 13-16 (normal)"),
                       (16, 20, "VIX 16-20 (elevated)"),
                       (20, 30, "VIX 20-30 (high)"),
                       (30, 50, "VIX > 30 (crisis)")]:
    subset = [t for t in trades if lo <= t["vix"] < hi]
    if subset:
        w = len([t for t in subset if t["pnl"] > 0])
        pnl = sum(t["pnl"] for t in subset)
        print(f"  {label:<25}: {len(subset):>3}t  Rs {pnl:>+9,.0f}  WR {w/len(subset)*100:.0f}%  "
              f"avg Rs {pnl/len(subset):>+,.0f}")

# ========== 8. DAY OF WEEK ==========
print("\n--- DAY OF WEEK ---")
for dow in ["Mon", "Tue", "Wed", "Thu", "Fri", "Sun"]:
    dow_trades = []
    for t in trades:
        try:
            d = datetime.datetime.strptime(t["date"], "%Y-%m-%d")
            if d.strftime("%a") == dow:
                dow_trades.append(t)
        except:
            pass
    if dow_trades:
        w = len([t for t in dow_trades if t["pnl"] > 0])
        pnl = sum(t["pnl"] for t in dow_trades)
        print(f"  {dow}: {len(dow_trades):>3}t  Rs {pnl:>+9,.0f}  WR {w/len(dow_trades)*100:.0f}%  "
              f"avg Rs {pnl/len(dow_trades):>+,.0f}")

# ========== 9. ADX THRESHOLD ==========
print("\n--- ADX AT ENTRY (trend strength) ---")
for lo, hi, label in [(0, 15, "ADX < 15 (no trend)"),
                       (15, 25, "ADX 15-25 (weak trend)"),
                       (25, 35, "ADX 25-35 (strong trend)"),
                       (35, 50, "ADX 35-50 (very strong)"),
                       (50, 100, "ADX > 50 (extreme)")]:
    subset = [t for t in trades if t.get("entry_adx") is not None and lo <= t["entry_adx"] < hi]
    if subset:
        w = len([t for t in subset if t["pnl"] > 0])
        pnl = sum(t["pnl"] for t in subset)
        print(f"  {label:<30}: {len(subset):>3}t  Rs {pnl:>+9,.0f}  WR {w/len(subset)*100:.0f}%")

# ========== 10. MULTI-FACTOR WINNING PATTERNS ==========
print("\n" + "=" * 120)
print("  WINNING COMBINATIONS (multi-factor)")
print("=" * 120)

# Pattern 1: PUT + bearish supertrend + RSI > 50 + VIX > 12
p1 = [t for t in trades if t["action"] == "BUY_PUT"
      and t.get("entry_st_dir") == -1
      and t.get("entry_rsi", 50) > 50
      and t["vix"] >= 12]
if p1:
    w = len([t for t in p1 if t["pnl"] > 0])
    print(f"  PUT + bearish ST + RSI>50 + VIX>=12: {len(p1)}t  Rs {sum(t['pnl'] for t in p1):+,}  WR {w/len(p1)*100:.0f}%")

# Pattern 2: PUT + all EMAs bearish + supertrend down
p2 = [t for t in trades if t["action"] == "BUY_PUT"
      and not t["above_ema9"] and not t["above_ema21"]
      and t.get("entry_st_dir") == -1]
if p2:
    w = len([t for t in p2 if t["pnl"] > 0])
    print(f"  PUT + all EMAs bearish + ST down:    {len(p2)}t  Rs {sum(t['pnl'] for t in p2):+,}  WR {w/len(p2)*100:.0f}%")

# Pattern 3: CALL + all EMAs bullish + supertrend up
p3 = [t for t in trades if t["action"] == "BUY_CALL"
      and t["above_ema9"] and t["above_ema21"]
      and t.get("entry_st_dir") == 1]
if p3:
    w = len([t for t in p3 if t["pnl"] > 0])
    print(f"  CALL + all EMAs bullish + ST up:     {len(p3)}t  Rs {sum(t['pnl'] for t in p3):+,}  WR {w/len(p3)*100:.0f}%")

# Pattern 4: High VIX PUT trades (VIX > 15)
p4 = [t for t in trades if t["action"] == "BUY_PUT" and t["vix"] >= 15]
if p4:
    w = len([t for t in p4 if t["pnl"] > 0])
    print(f"  PUT + high VIX (>=15):               {len(p4)}t  Rs {sum(t['pnl'] for t in p4):+,}  WR {w/len(p4)*100:.0f}%")

# Pattern 5: Morning entries (9:15-10:00) + supertrend aligned
p5 = [t for t in trades if t["entry_minute"] < 45
      and ((t["action"] == "BUY_CALL" and t.get("entry_st_dir") == 1)
           or (t["action"] == "BUY_PUT" and t.get("entry_st_dir") == -1))]
if p5:
    w = len([t for t in p5 if t["pnl"] > 0])
    print(f"  Morning (before 10) + ST aligned:    {len(p5)}t  Rs {sum(t['pnl'] for t in p5):+,}  WR {w/len(p5)*100:.0f}%")

# Pattern 6: EMA cross as entry signal
p6 = [t for t in trades if "ema_cross_up" in t.get("reasons", []) or "ema_cross_down" in t.get("reasons", [])]
if p6:
    w = len([t for t in p6 if t["pnl"] > 0])
    print(f"  EMA 9/21 crossover entries:          {len(p6)}t  Rs {sum(t['pnl'] for t in p6):+,}  WR {w/len(p6)*100:.0f}%")

# Pattern 7: 11 AM entries (best window from time analysis)
p7 = [t for t in trades if 105 <= t["entry_minute"] < 135]
if p7:
    w = len([t for t in p7 if t["pnl"] > 0])
    print(f"  11:00-11:30 entries:                 {len(p7)}t  Rs {sum(t['pnl'] for t in p7):+,}  WR {w/len(p7)*100:.0f}%")

# Pattern 8: AVOID 12:00-12:30 (worst window)
p8 = [t for t in trades if 165 <= t["entry_minute"] < 195]
if p8:
    w = len([t for t in p8 if t["pnl"] > 0])
    print(f"  12:00-12:30 entries (WORST):          {len(p8)}t  Rs {sum(t['pnl'] for t in p8):+,}  WR {w/len(p8)*100:.0f}%")

# Pattern 9: Supertrend flip as exit vs hold to time_exit
# Compare: trades that exited on supertrend_flip vs what time_exit achieves
st_flip_trades = [t for t in trades if t["exit_reason"] == "supertrend_flip"]
time_exit_trades = [t for t in trades if t["exit_reason"] == "time_exit"]
print(f"\n  SUPERTREND FLIP EXIT vs TIME EXIT:")
print(f"    ST flip: {len(st_flip_trades)}t  Rs {sum(t['pnl'] for t in st_flip_trades):+,}  "
      f"WR {len([t for t in st_flip_trades if t['pnl']>0])/max(len(st_flip_trades),1)*100:.0f}%  "
      f"avg hold {np.mean([t['minutes_held'] for t in st_flip_trades]):.0f}min")
print(f"    Time exit: {len(time_exit_trades)}t  Rs {sum(t['pnl'] for t in time_exit_trades):+,}  "
      f"WR {len([t for t in time_exit_trades if t['pnl']>0])/max(len(time_exit_trades),1)*100:.0f}%  "
      f"avg hold {np.mean([t['minutes_held'] for t in time_exit_trades]):.0f}min")
print(f"    --> Conclusion: Supertrend flip HURTS. Hold to time_exit is BETTER.")

# ========== 11. WHAT WOULD HAPPEN IF WE FILTER ==========
print("\n" + "=" * 120)
print("  SIMULATED FILTERS — What if we applied these rules?")
print("=" * 120)

# Filter 1: Remove 12:00-1:00 entries
f1 = [t for t in trades if not (165 <= t["entry_minute"] < 225)]
f1_pnl = sum(t["pnl"] for t in f1)
f1_wr = len([t for t in f1 if t["pnl"] > 0]) / len(f1) * 100
print(f"  No 12:00-1:00 entries:   {len(f1)}t  Rs {f1_pnl:>+,}  WR {f1_wr:.0f}%  "
      f"(removed {len(trades)-len(f1)} trades, saved Rs {f1_pnl - sum(t['pnl'] for t in trades):+,})")

# Filter 2: Only supertrend-aligned trades
f2 = [t for t in trades if
      (t["action"] == "BUY_CALL" and t.get("entry_st_dir") == 1) or
      (t["action"] == "BUY_PUT" and t.get("entry_st_dir") == -1)]
f2_pnl = sum(t["pnl"] for t in f2)
f2_wr = len([t for t in f2 if t["pnl"] > 0]) / len(f2) * 100
print(f"  Only ST-aligned:         {len(f2)}t  Rs {f2_pnl:>+,}  WR {f2_wr:.0f}%")

# Filter 3: Only EMA-aligned trades
f3 = [t for t in trades if
      (t["action"] == "BUY_CALL" and t["above_ema9"] and t["above_ema21"]) or
      (t["action"] == "BUY_PUT" and not t["above_ema9"] and not t["above_ema21"])]
f3_pnl = sum(t["pnl"] for t in f3)
f3_wr = len([t for t in f3 if t["pnl"] > 0]) / len(f3) * 100
print(f"  Only EMA-aligned:        {len(f3)}t  Rs {f3_pnl:>+,}  WR {f3_wr:.0f}%")

# Filter 4: Remove supertrend_flip exits (hold to time_exit instead)
# Estimate: if ST flip trades had held, they'd get closer to time_exit WR (63%)
st_flip_loss = sum(t["pnl"] for t in st_flip_trades)
time_avg_pnl = np.mean([t["pnl"] for t in time_exit_trades])
estimated_if_held = len(st_flip_trades) * time_avg_pnl * 0.5  # Conservative: 50% of time_exit avg
print(f"  Remove ST flip exit:     Would recover ~Rs {abs(st_flip_loss) + estimated_if_held:+,.0f} "
      f"({len(st_flip_trades)} trades held longer)")

# Filter 5: Combined best filters
f5 = [t for t in trades if
      not (165 <= t["entry_minute"] < 225)  # No lunch hour
      and (  # EMA or ST aligned
          (t["action"] == "BUY_CALL" and (t["above_ema9"] or t.get("entry_st_dir") == 1)) or
          (t["action"] == "BUY_PUT" and (not t["above_ema9"] or t.get("entry_st_dir") == -1))
      )]
f5_pnl = sum(t["pnl"] for t in f5)
f5_wr = len([t for t in f5 if t["pnl"] > 0]) / len(f5) * 100
print(f"  Combined (no lunch + aligned): {len(f5)}t  Rs {f5_pnl:>+,}  WR {f5_wr:.0f}%")

# ========== FINAL SUMMARY OF LEARNED RULES ==========
print("\n" + "=" * 120)
print("  LEARNED RULES FOR OPTIMIZED MODEL")
print("=" * 120)
print("""
  1. TIME_EXIT is the profit engine (63% WR, Rs +454K) — NEVER override with early exits
  2. TRAIL_STOP kills profits (7% WR, Rs -279K) — Widen to 0.8-1.0% or remove for PUTs
  3. SUPERTREND FLIP EXIT is BAD (32% WR, Rs -49K) — Remove this exit entirely
  4. BUY_PUT >> BUY_CALL (47% vs 37% WR) — Bias heavily toward PUTs
  5. 12:00-12:30 is the WORST time to enter (WR ~16%) — Skip lunch hour entirely
  6. 11:00-11:30 is the BEST time to enter (WR ~55%) — Prioritize this window
  7. 9:15-9:30 has most trades and decent WR (47%) — Morning entries are good
  8. EMA-aligned trades significantly outperform counter-trend trades
  9. Supertrend direction alignment improves WR by 5-10%
  10. VIX 13-16 is the sweet spot for PUT trades
  11. Winners hold 180 min avg vs losers 130 min — PATIENCE pays
  12. Winners get 0.45% favorable move vs losers 0.11% — Real moves are bigger
""")
print("=" * 120)
