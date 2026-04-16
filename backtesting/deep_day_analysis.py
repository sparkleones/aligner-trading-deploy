"""
Deep Day-by-Day Analysis of NIFTY — Last 6 Months.
Research-only script. Does NOT modify any production files.
"""

import os
import sys
import warnings
from collections import defaultdict
from pathlib import Path

# Force UTF-8 output on Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8')
os.environ["PYTHONIOENCODING"] = "utf-8"

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backtesting.paper_trading_real_data import download_real_data, generate_intraday_path, sr_multi_method

# ═══════════════════════════════════════════════════════════════════
# 1. DOWNLOAD DATA
# ═══════════════════════════════════════════════════════════════════
print("=" * 90)
print("NIFTY DEEP DAY-BY-DAY ANALYSIS — LAST 6 MONTHS")
print("=" * 90)

df = download_real_data(start="2025-10-01", end="2026-04-06")
print(f"\nTotal trading days in analysis window: {len(df)}")
print()

# ═══════════════════════════════════════════════════════════════════
# 2. DAY-BY-DAY DETAIL TABLE
# ═══════════════════════════════════════════════════════════════════

TOTAL_BARS = 25  # 15-min bars per session

records = []
close_history = df["Close"].values.tolist()

for i in range(len(df)):
    row = df.iloc[i]
    date = df.index[i]

    o, h, l, c = float(row["Open"]), float(row["High"]), float(row["Low"]), float(row["Close"])
    vix = float(row["VIX"])
    prev_close = float(row["PrevClose"]) if pd.notna(row["PrevClose"]) else o
    gap_pct = float(row["GapPct"])
    dow = row["DOW"]
    is_expiry = bool(row["IsExpiry"])

    day_range_pct = (h - l) / o * 100
    is_up = c >= o
    direction = "UP" if is_up else "DOWN"

    # Gap classification
    gap_type = "NONE"
    gap_outcome = "N/A"
    if abs(gap_pct) > 0.5:
        gap_dir = "UP" if gap_pct > 0 else "DOWN"
        gap_type = f"GAP_{gap_dir}"
        # Continuation = close extends beyond gap direction
        if gap_pct > 0:
            gap_outcome = "CONTINUATION" if c > o else "REVERSAL"
        else:
            gap_outcome = "CONTINUATION" if c < o else "REVERSAL"

    # S/R levels
    sma20 = float(row["SMA20"]) if pd.notna(row["SMA20"]) else None
    sma50 = float(row["SMA50"]) if pd.notna(row["SMA50"]) else None
    prev_h = float(row["PrevHigh"]) if pd.notna(row["PrevHigh"]) else None
    prev_l = float(row["PrevLow"]) if pd.notna(row["PrevLow"]) else None

    support, resistance = sr_multi_method(
        o, prev_h, prev_l, sma20, sma50,
        close_history=close_history, idx=i
    )

    # Generate intraday path to analyse time-of-day moves
    path = generate_intraday_path(o, h, l, c, n_bars=TOTAL_BARS)

    # --- Best PUT trade (profit from price going DOWN from open) ---
    # PUT buyer profits when price drops. Best entry = near high, best exit = near low.
    # If entered at open, best exit = lowest point after open
    put_best_exit_idx = 0
    put_best_price = path[0]
    for bi in range(1, len(path)):
        if path[bi] < put_best_price:
            put_best_price = path[bi]
            put_best_exit_idx = bi

    put_move_pct = (o - put_best_price) / o * 100  # positive = profit for PUT

    # --- Best CALL trade (profit from price going UP from open) ---
    call_best_exit_idx = 0
    call_best_price = path[0]
    for bi in range(1, len(path)):
        if path[bi] > call_best_price:
            call_best_price = path[bi]
            call_best_exit_idx = bi

    call_move_pct = (call_best_price - o) / o * 100  # positive = profit for CALL

    # --- Best BTST trade (enter afternoon bar 17+, hold overnight) ---
    # Afternoon = bars 17-24 (after ~1:30 PM)
    afternoon_start = 17
    btst_entry_price = None
    btst_pnl_pct = 0.0
    if len(path) > afternoon_start:
        # Best BTST: enter at lowest point in afternoon (for CALL overnight)
        # or highest point (for PUT overnight)
        afternoon_prices = path[afternoon_start:]

        # CALL BTST: buy at afternoon low, profit = close - entry
        afternoon_low = min(afternoon_prices)
        btst_call_pnl = (c - afternoon_low) / afternoon_low * 100

        # PUT BTST: sell at afternoon high, profit = entry - close
        afternoon_high = max(afternoon_prices)
        btst_put_pnl = (afternoon_high - c) / afternoon_high * 100

        btst_pnl_pct = max(btst_call_pnl, btst_put_pnl)
        btst_entry_price = afternoon_low if btst_call_pnl >= btst_put_pnl else afternoon_high

    # First 30 min move (bars 0-1 out of 25)
    first_30min_move = abs(path[min(2, len(path)-1)] - path[0])
    total_day_move = abs(c - o)
    first_30min_pct_of_total = (first_30min_move / total_day_move * 100) if total_day_move > 0 else 0

    # Round number tests
    round_levels = list(range(int(l // 500) * 500, int(h // 500) * 500 + 501, 500))
    round_100_levels = list(range(int(l // 100) * 100, int(h // 100) * 100 + 101, 100))
    round_500_touched = [lv for lv in round_levels if l <= lv <= h]
    round_100_touched = [lv for lv in round_100_levels if l <= lv <= h]

    # Bar timing for best PUT/CALL (convert bar index to time window)
    def bar_to_time(bar_idx):
        # Bar 0 = 9:15, each bar = 15 min
        mins = bar_idx * 15
        hour = 9 + (15 + mins) // 60
        minute = (15 + mins) % 60
        return f"{hour:02d}:{minute:02d}"

    def bar_to_window(bar_idx):
        if bar_idx <= 2:
            return "MORNING_OPEN"    # 9:15 - 9:45
        elif bar_idx <= 6:
            return "MORNING"         # 9:45 - 10:45
        elif bar_idx <= 12:
            return "MIDDAY"          # 10:45 - 12:15
        elif bar_idx <= 18:
            return "AFTERNOON_EARLY" # 12:15 - 1:45
        else:
            return "AFTERNOON_LATE"  # 1:45 - 3:30

    rec = {
        "Date": date.strftime("%Y-%m-%d"),
        "DOW": dow,
        "Open": round(o, 1),
        "High": round(h, 1),
        "Low": round(l, 1),
        "Close": round(c, 1),
        "VIX": round(vix, 1),
        "VIXRegime": row["VIXRegime"],
        "GapPct": round(gap_pct, 2),
        "GapType": gap_type,
        "GapOutcome": gap_outcome,
        "RangePct": round(day_range_pct, 2),
        "Direction": direction,
        "IsExpiry": is_expiry,
        "Support": support,
        "Resistance": resistance,
        "PutBestMovePct": round(put_move_pct, 3),
        "PutBestExitBar": put_best_exit_idx,
        "PutBestExitTime": bar_to_time(put_best_exit_idx),
        "PutBestWindow": bar_to_window(put_best_exit_idx),
        "CallBestMovePct": round(call_move_pct, 3),
        "CallBestExitBar": call_best_exit_idx,
        "CallBestExitTime": bar_to_time(call_best_exit_idx),
        "CallBestWindow": bar_to_window(call_best_exit_idx),
        "BTSTPnlPct": round(btst_pnl_pct, 3),
        "First30minPctOfTotal": round(first_30min_pct_of_total, 1),
        "Round500Touched": round_500_touched,
        "Round100Touched": round_100_touched,
        "RSI": round(float(row["RSI"]), 1) if pd.notna(row["RSI"]) else None,
        "AboveSMA50": bool(row["AboveSMA50"]),
        "AboveSMA20": bool(row["AboveSMA20"]),
    }
    records.append(rec)

rdf = pd.DataFrame(records)

# ═══════════════════════════════════════════════════════════════════
# PRINT DAY-BY-DAY TABLE
# ═══════════════════════════════════════════════════════════════════

print("=" * 90)
print("DAY-BY-DAY DETAIL")
print("=" * 90)

for _, r in rdf.iterrows():
    exp_tag = " [EXPIRY]" if r["IsExpiry"] else ""
    gap_tag = f" | Gap {r['GapPct']:+.2f}% ({r['GapOutcome']})" if r["GapType"] != "NONE" else ""
    print(f"\n{r['Date']} ({r['DOW']}{exp_tag}) | {r['Direction']} | "
          f"O:{r['Open']} H:{r['High']} L:{r['Low']} C:{r['Close']} | "
          f"VIX:{r['VIX']} ({r['VIXRegime']})")
    print(f"  Range: {r['RangePct']:.2f}% | S:{r['Support']} R:{r['Resistance']}{gap_tag}")
    print(f"  Best PUT: {r['PutBestMovePct']:+.3f}% (exit@{r['PutBestExitTime']} {r['PutBestWindow']})")
    print(f"  Best CALL: {r['CallBestMovePct']:+.3f}% (exit@{r['CallBestExitTime']} {r['CallBestWindow']})")
    print(f"  Best BTST: {r['BTSTPnlPct']:+.3f}% | First30min: {r['First30minPctOfTotal']:.0f}% of day move")
    if r["Round500Touched"]:
        print(f"  Round 500s touched: {r['Round500Touched']}")

# ═══════════════════════════════════════════════════════════════════
# 3. SUMMARY STATISTICS
# ═══════════════════════════════════════════════════════════════════

print("\n\n" + "=" * 90)
print("SUMMARY STATISTICS")
print("=" * 90)

# ──────────────────────────────────────────────────────────────
# 3a. Average daily range by day of week
# ──────────────────────────────────────────────────────────────
print("\n─── AVERAGE DAILY RANGE (%) BY DAY OF WEEK ───")
dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
dow_stats = rdf.groupby("DOW")["RangePct"].agg(["mean", "median", "std", "count"])
dow_stats = dow_stats.reindex(dow_order)
for dow in dow_order:
    if dow in dow_stats.index:
        s = dow_stats.loc[dow]
        print(f"  {dow:<12s}: Mean={s['mean']:.3f}%  Median={s['median']:.3f}%  "
              f"StdDev={s['std']:.3f}%  Count={int(s['count'])}")

# Up/Down ratio by day of week
print("\n─── UP vs DOWN DAY RATIO BY DAY OF WEEK ───")
for dow in dow_order:
    sub = rdf[rdf["DOW"] == dow]
    if len(sub) == 0:
        continue
    up_count = (sub["Direction"] == "UP").sum()
    dn_count = (sub["Direction"] == "DOWN").sum()
    pct = up_count / len(sub) * 100
    print(f"  {dow:<12s}: UP={up_count}  DOWN={dn_count}  "
          f"UP%={pct:.1f}%  (n={len(sub)})")

# ──────────────────────────────────────────────────────────────
# 3b. Best time windows for PUT entries
# ──────────────────────────────────────────────────────────────
print("\n─── PUT OPTIMAL EXIT WINDOW DISTRIBUTION ───")
put_windows = rdf["PutBestWindow"].value_counts()
for w in ["MORNING_OPEN", "MORNING", "MIDDAY", "AFTERNOON_EARLY", "AFTERNOON_LATE"]:
    cnt = put_windows.get(w, 0)
    pct = cnt / len(rdf) * 100
    avg_move = rdf[rdf["PutBestWindow"] == w]["PutBestMovePct"].mean() if cnt > 0 else 0
    print(f"  {w:<20s}: {cnt:3d} days ({pct:5.1f}%) | Avg move: {avg_move:+.3f}%")

print("\n─── CALL OPTIMAL EXIT WINDOW DISTRIBUTION ───")
call_windows = rdf["CallBestWindow"].value_counts()
for w in ["MORNING_OPEN", "MORNING", "MIDDAY", "AFTERNOON_EARLY", "AFTERNOON_LATE"]:
    cnt = call_windows.get(w, 0)
    pct = cnt / len(rdf) * 100
    avg_move = rdf[rdf["CallBestWindow"] == w]["CallBestMovePct"].mean() if cnt > 0 else 0
    print(f"  {w:<20s}: {cnt:3d} days ({pct:5.1f}%) | Avg move: {avg_move:+.3f}%")

# ──────────────────────────────────────────────────────────────
# 3c. Gap continuation vs reversal rate
# ──────────────────────────────────────────────────────────────
print("\n─── GAP CONTINUATION vs REVERSAL (|gap| > 0.5%) ───")
gaps = rdf[rdf["GapType"] != "NONE"]
if len(gaps) > 0:
    for gt in ["GAP_UP", "GAP_DOWN"]:
        sub = gaps[gaps["GapType"] == gt]
        if len(sub) == 0:
            continue
        cont = (sub["GapOutcome"] == "CONTINUATION").sum()
        rev = (sub["GapOutcome"] == "REVERSAL").sum()
        print(f"  {gt}: {len(sub)} days | Continuation={cont} ({cont/len(sub)*100:.1f}%) | "
              f"Reversal={rev} ({rev/len(sub)*100:.1f}%)")
        # Average gap size
        avg_gap = sub["GapPct"].abs().mean()
        print(f"    Avg |gap|: {avg_gap:.2f}%")
else:
    print("  No gaps > 0.5% found in period.")

# Also do a finer breakdown by gap size
print("\n─── GAP ANALYSIS BY SIZE BUCKET ───")
for lo, hi, label in [(0.3, 0.5, "0.3-0.5%"), (0.5, 0.8, "0.5-0.8%"), (0.8, 1.2, "0.8-1.2%"), (1.2, 5.0, ">1.2%")]:
    sub = rdf[(rdf["GapPct"].abs() >= lo) & (rdf["GapPct"].abs() < hi)]
    if len(sub) == 0:
        continue
    up_days = (sub["Direction"] == "UP").sum()
    dn_days = (sub["Direction"] == "DOWN").sum()
    # Continuation vs reversal (direction matches gap direction)
    cont = 0
    rev = 0
    for _, r in sub.iterrows():
        if r["GapPct"] > 0:  # Gap up
            if r["Direction"] == "UP":
                cont += 1
            else:
                rev += 1
        else:  # Gap down
            if r["Direction"] == "DOWN":
                cont += 1
            else:
                rev += 1
    print(f"  {label:>8s}: {len(sub):2d} days | Cont={cont} Rev={rev} | "
          f"Cont%={cont/len(sub)*100:.0f}%")

# ──────────────────────────────────────────────────────────────
# 3d. Round number tests
# ──────────────────────────────────────────────────────────────
print("\n─── ROUND NUMBER TESTS ───")
round_500_counter = defaultdict(int)
round_100_counter = defaultdict(int)
for _, r in rdf.iterrows():
    for lv in r["Round500Touched"]:
        round_500_counter[lv] += 1
    for lv in r["Round100Touched"]:
        round_100_counter[lv] += 1

total_days = len(rdf)
days_with_500 = sum(1 for _, r in rdf.iterrows() if len(r["Round500Touched"]) > 0)
days_with_100 = sum(1 for _, r in rdf.iterrows() if len(r["Round100Touched"]) > 0)
print(f"  Days touching a round 500 level: {days_with_500}/{total_days} ({days_with_500/total_days*100:.1f}%)")
print(f"  Days touching a round 100 level: {days_with_100}/{total_days} ({days_with_100/total_days*100:.1f}%)")
print(f"\n  Most-tested 500-levels:")
for lv, cnt in sorted(round_500_counter.items(), key=lambda x: -x[1])[:10]:
    print(f"    {lv}: {cnt} days ({cnt/total_days*100:.1f}%)")

# ──────────────────────────────────────────────────────────────
# 3e. VIX regime correlation with daily range
# ──────────────────────────────────────────────────────────────
print("\n─── VIX REGIME vs DAILY RANGE ───")
vix_order = ["VERY_LOW", "LOW", "NORMAL_LOW", "NORMAL_HIGH", "HIGH", "VERY_HIGH"]
vix_stats = rdf.groupby("VIXRegime").agg(
    range_mean=("RangePct", "mean"),
    range_median=("RangePct", "median"),
    put_mean=("PutBestMovePct", "mean"),
    call_mean=("CallBestMovePct", "mean"),
    up_pct=("Direction", lambda x: (x == "UP").mean() * 100),
    count=("RangePct", "count"),
)
for regime in vix_order:
    if regime in vix_stats.index:
        s = vix_stats.loc[regime]
        print(f"  {regime:<12s}: AvgRange={s['range_mean']:.3f}% | "
              f"MedRange={s['range_median']:.3f}% | "
              f"AvgPUT={s['put_mean']:+.3f}% | AvgCALL={s['call_mean']:+.3f}% | "
              f"UP%={s['up_pct']:.0f}% | n={int(s['count'])}")

# ──────────────────────────────────────────────────────────────
# 3f. Expiry day (Tuesday) vs non-expiry
# ──────────────────────────────────────────────────────────────
print("\n─── EXPIRY DAY (TUESDAY) vs NON-EXPIRY ───")
expiry = rdf[rdf["IsExpiry"] == True]
non_expiry = rdf[rdf["IsExpiry"] == False]
if len(expiry) > 0:
    print(f"  EXPIRY DAYS (n={len(expiry)}):")
    print(f"    Avg Range: {expiry['RangePct'].mean():.3f}%  "
          f"Median: {expiry['RangePct'].median():.3f}%")
    print(f"    Avg PUT move: {expiry['PutBestMovePct'].mean():+.3f}%  "
          f"Avg CALL move: {expiry['CallBestMovePct'].mean():+.3f}%")
    up_pct = (expiry["Direction"] == "UP").mean() * 100
    print(f"    UP days: {up_pct:.1f}%  DOWN days: {100-up_pct:.1f}%")
    print(f"    Avg VIX: {expiry['VIX'].mean():.1f}")

if len(non_expiry) > 0:
    print(f"  NON-EXPIRY DAYS (n={len(non_expiry)}):")
    print(f"    Avg Range: {non_expiry['RangePct'].mean():.3f}%  "
          f"Median: {non_expiry['RangePct'].median():.3f}%")
    print(f"    Avg PUT move: {non_expiry['PutBestMovePct'].mean():+.3f}%  "
          f"Avg CALL move: {non_expiry['CallBestMovePct'].mean():+.3f}%")
    up_pct_ne = (non_expiry["Direction"] == "UP").mean() * 100
    print(f"    UP days: {up_pct_ne:.1f}%  DOWN days: {100-up_pct_ne:.1f}%")
    print(f"    Avg VIX: {non_expiry['VIX'].mean():.1f}")

# ──────────────────────────────────────────────────────────────
# 3g. Consecutive up/down day patterns
# ──────────────────────────────────────────────────────────────
print("\n─── CONSECUTIVE UP/DOWN DAY PATTERNS ───")
directions = rdf["Direction"].tolist()
streaks = []
current_dir = directions[0]
current_len = 1
for d in directions[1:]:
    if d == current_dir:
        current_len += 1
    else:
        streaks.append((current_dir, current_len))
        current_dir = d
        current_len = 1
streaks.append((current_dir, current_len))

up_streaks = [s[1] for s in streaks if s[0] == "UP"]
dn_streaks = [s[1] for s in streaks if s[0] == "DOWN"]

print(f"  UP streaks:   count={len(up_streaks)}  "
      f"avg_len={np.mean(up_streaks):.1f}  max={max(up_streaks) if up_streaks else 0}  "
      f"distribution: {dict(pd.Series(up_streaks).value_counts().sort_index())}")
print(f"  DOWN streaks: count={len(dn_streaks)}  "
      f"avg_len={np.mean(dn_streaks):.1f}  max={max(dn_streaks) if dn_streaks else 0}  "
      f"distribution: {dict(pd.Series(dn_streaks).value_counts().sort_index())}")

# What happens after N consecutive days in same direction?
print("\n  After 2+ consecutive UP days:")
for i in range(2, len(rdf)):
    if rdf.iloc[i-1]["Direction"] == "UP" and rdf.iloc[i-2]["Direction"] == "UP":
        pass  # collect these
after_2up = []
after_2dn = []
for i in range(2, len(rdf)):
    if rdf.iloc[i-1]["Direction"] == "UP" and rdf.iloc[i-2]["Direction"] == "UP":
        after_2up.append(rdf.iloc[i]["Direction"])
    if rdf.iloc[i-1]["Direction"] == "DOWN" and rdf.iloc[i-2]["Direction"] == "DOWN":
        after_2dn.append(rdf.iloc[i]["Direction"])

if after_2up:
    up_after = sum(1 for d in after_2up if d == "UP")
    print(f"    Next day UP: {up_after}/{len(after_2up)} ({up_after/len(after_2up)*100:.1f}%)")
    print(f"    Next day DOWN: {len(after_2up)-up_after}/{len(after_2up)} ({(len(after_2up)-up_after)/len(after_2up)*100:.1f}%)")
else:
    print("    No instances found.")

print("\n  After 2+ consecutive DOWN days:")
if after_2dn:
    up_after = sum(1 for d in after_2dn if d == "UP")
    print(f"    Next day UP: {up_after}/{len(after_2dn)} ({up_after/len(after_2dn)*100:.1f}%)")
    print(f"    Next day DOWN: {len(after_2dn)-up_after}/{len(after_2dn)} ({(len(after_2dn)-up_after)/len(after_2dn)*100:.1f}%)")
else:
    print("    No instances found.")

# After 3+ consecutive
after_3up = []
after_3dn = []
for i in range(3, len(rdf)):
    if all(rdf.iloc[i-j]["Direction"] == "UP" for j in range(1, 4)):
        after_3up.append(rdf.iloc[i]["Direction"])
    if all(rdf.iloc[i-j]["Direction"] == "DOWN" for j in range(1, 4)):
        after_3dn.append(rdf.iloc[i]["Direction"])

print(f"\n  After 3+ consecutive UP days ({len(after_3up)} instances):")
if after_3up:
    up_ct = sum(1 for d in after_3up if d == "UP")
    print(f"    Continues UP: {up_ct} ({up_ct/len(after_3up)*100:.1f}%)  Reverses DOWN: {len(after_3up)-up_ct} ({(len(after_3up)-up_ct)/len(after_3up)*100:.1f}%)")

print(f"  After 3+ consecutive DOWN days ({len(after_3dn)} instances):")
if after_3dn:
    up_ct = sum(1 for d in after_3dn if d == "UP")
    print(f"    Reverses UP: {up_ct} ({up_ct/len(after_3dn)*100:.1f}%)  Continues DOWN: {len(after_3dn)-up_ct} ({(len(after_3dn)-up_ct)/len(after_3dn)*100:.1f}%)")

# ──────────────────────────────────────────────────────────────
# 3h. First 30 min vs rest of day
# ──────────────────────────────────────────────────────────────
print("\n─── FIRST 30 MIN vs REST OF DAY ───")
f30 = rdf["First30minPctOfTotal"]
# Filter out infinite/huge values (days where total move was ~0)
f30_clean = f30[f30 < 500]
print(f"  Avg % of day's move in first 30 min: {f30_clean.mean():.1f}%")
print(f"  Median: {f30_clean.median():.1f}%")
print(f"  Days where first 30 min > 50% of total move: "
      f"{(f30_clean > 50).sum()}/{len(f30_clean)} ({(f30_clean > 50).mean()*100:.1f}%)")
print(f"  Days where first 30 min > 80% of total move: "
      f"{(f30_clean > 80).sum()}/{len(f30_clean)} ({(f30_clean > 80).mean()*100:.1f}%)")

# Breakdown by direction
for direction in ["UP", "DOWN"]:
    sub = rdf[rdf["Direction"] == direction]
    f30_sub = sub["First30minPctOfTotal"]
    f30_sub = f30_sub[f30_sub < 500]
    if len(f30_sub) > 0:
        print(f"  {direction} days — first 30 min captures: {f30_sub.mean():.1f}% (median {f30_sub.median():.1f}%)")

# ──────────────────────────────────────────────────────────────
# EXTRA: Overall best PUT/CALL stats
# ──────────────────────────────────────────────────────────────
print("\n─── OVERALL BEST TRADE OPPORTUNITIES ───")
print(f"  Average best PUT move (if entered at open): {rdf['PutBestMovePct'].mean():+.3f}%")
print(f"  Average best CALL move (if entered at open): {rdf['CallBestMovePct'].mean():+.3f}%")
print(f"  Average best BTST move: {rdf['BTSTPnlPct'].mean():+.3f}%")
print(f"  Median best PUT: {rdf['PutBestMovePct'].median():+.3f}%")
print(f"  Median best CALL: {rdf['CallBestMovePct'].median():+.3f}%")
print(f"  Median best BTST: {rdf['BTSTPnlPct'].median():+.3f}%")

# Days where PUT > 1% move available
big_put_days = rdf[rdf["PutBestMovePct"] > 1.0]
big_call_days = rdf[rdf["CallBestMovePct"] > 1.0]
print(f"\n  Days with PUT move > 1%: {len(big_put_days)} ({len(big_put_days)/len(rdf)*100:.1f}%)")
print(f"  Days with CALL move > 1%: {len(big_call_days)} ({len(big_call_days)/len(rdf)*100:.1f}%)")

# ──────────────────────────────────────────────────────────────
# EXTRA: Monthly breakdown
# ──────────────────────────────────────────────────────────────
print("\n─── MONTHLY BREAKDOWN ───")
rdf["Month"] = pd.to_datetime(rdf["Date"]).dt.to_period("M")
monthly = rdf.groupby("Month").agg(
    days=("RangePct", "count"),
    avg_range=("RangePct", "mean"),
    avg_vix=("VIX", "mean"),
    up_pct=("Direction", lambda x: (x == "UP").mean() * 100),
    avg_put=("PutBestMovePct", "mean"),
    avg_call=("CallBestMovePct", "mean"),
    avg_btst=("BTSTPnlPct", "mean"),
)
for month, s in monthly.iterrows():
    print(f"  {month}: {int(s['days'])} days | "
          f"AvgRange={s['avg_range']:.3f}% | AvgVIX={s['avg_vix']:.1f} | "
          f"UP={s['up_pct']:.0f}% | "
          f"PUT={s['avg_put']:+.3f}% CALL={s['avg_call']:+.3f}% BTST={s['avg_btst']:+.3f}%")

# ──────────────────────────────────────────────────────────────
# EXTRA: Day-of-week best trade type
# ──────────────────────────────────────────────────────────────
print("\n─── BEST TRADE TYPE BY DAY OF WEEK ───")
for dow in dow_order:
    sub = rdf[rdf["DOW"] == dow]
    if len(sub) == 0:
        continue
    avg_put = sub["PutBestMovePct"].mean()
    avg_call = sub["CallBestMovePct"].mean()
    avg_btst = sub["BTSTPnlPct"].mean()
    best = "PUT" if avg_put > avg_call else "CALL"
    print(f"  {dow:<12s}: PUT={avg_put:+.3f}% CALL={avg_call:+.3f}% BTST={avg_btst:+.3f}% -> Best={best}")

# ──────────────────────────────────────────────────────────────
# EXTRA: RSI-based patterns
# ──────────────────────────────────────────────────────────────
print("\n─── RSI-BASED PATTERNS ───")
rsi_valid = rdf[rdf["RSI"].notna()]
for lo, hi, label in [(0, 30, "Oversold (<30)"), (30, 50, "Bearish (30-50)"),
                        (50, 70, "Bullish (50-70)"), (70, 100, "Overbought (>70)")]:
    sub = rsi_valid[(rsi_valid["RSI"] >= lo) & (rsi_valid["RSI"] < hi)]
    if len(sub) == 0:
        continue
    up_pct = (sub["Direction"] == "UP").mean() * 100
    avg_range = sub["RangePct"].mean()
    print(f"  {label:<20s}: {len(sub):3d} days | UP%={up_pct:.1f}% | AvgRange={avg_range:.3f}%")

# ──────────────────────────────────────────────────────────────
# EXTRA: Big move days analysis (range > 1.5%)
# ──────────────────────────────────────────────────────────────
print("\n─── BIG MOVE DAYS (Range > 1.5%) ───")
big_days = rdf[rdf["RangePct"] > 1.5].sort_values("RangePct", ascending=False)
print(f"  Count: {len(big_days)} out of {len(rdf)} days ({len(big_days)/len(rdf)*100:.1f}%)")
if len(big_days) > 0:
    print(f"  Average VIX on big days: {big_days['VIX'].mean():.1f}")
    up_pct_big = (big_days["Direction"] == "UP").mean() * 100
    print(f"  UP: {up_pct_big:.0f}% DOWN: {100-up_pct_big:.0f}%")
    print(f"  Top 10 biggest range days:")
    for _, r in big_days.head(10).iterrows():
        print(f"    {r['Date']} ({r['DOW']}) Range={r['RangePct']:.2f}% "
              f"{r['Direction']} VIX={r['VIX']} "
              f"PUT={r['PutBestMovePct']:+.3f}% CALL={r['CallBestMovePct']:+.3f}%")

# ──────────────────────────────────────────────────────────────
# EXTRA: SMA position correlation
# ──────────────────────────────────────────────────────────────
print("\n─── SMA POSITION CORRELATION ───")
above_both = rdf[(rdf["AboveSMA50"] == True) & (rdf["AboveSMA20"] == True)]
below_both = rdf[(rdf["AboveSMA50"] == False) & (rdf["AboveSMA20"] == False)]
mixed = rdf[rdf["AboveSMA50"] != rdf["AboveSMA20"]]

for label, sub in [("Above SMA20 & SMA50", above_both),
                    ("Below SMA20 & SMA50", below_both),
                    ("Mixed (between SMAs)", mixed)]:
    if len(sub) == 0:
        continue
    up_pct = (sub["Direction"] == "UP").mean() * 100
    avg_range = sub["RangePct"].mean()
    avg_put = sub["PutBestMovePct"].mean()
    avg_call = sub["CallBestMovePct"].mean()
    print(f"  {label:<28s}: {len(sub):3d} days | UP%={up_pct:.1f}% | "
          f"Range={avg_range:.3f}% | PUT={avg_put:+.3f}% CALL={avg_call:+.3f}%")


print("\n" + "=" * 90)
print("ANALYSIS COMPLETE")
print("=" * 90)
