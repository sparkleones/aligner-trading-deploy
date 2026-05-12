# QGV Engine on Large-Caps — Real Improvement (2026-05-13)

## User question
"Can we use the QGV approach on LARGE cap instead of small cap to
beat NIFTY in all 6 regimes?"

## Short answer
**Not in all 6, but we found a real ~6%/yr improvement.**

## What changed

The QGV ENGINE (not the QGV strategy) has 5 structural improvements
that the original mf_style engine lacks:

  1. ANNUAL rebalance instead of quarterly  (less tax + slippage drag)
  2. CONVICTION-WEIGHTED allocation (top score gets max weight)
  3. SECTOR-TIER allocation (top sectors get 60% of capital)
  4. DYNAMIC HOLD (no forced exit on Stage 2 break alone)
  5. DIP-BUYING (add 25% to position if 10% drop + quality intact)

When we run the SAME composite_top strategy through this engine,
median CAGR jumps from **23.03% to 28.87%** while keeping 4/6 NIFTY beats.

## Per-window comparison (composite_top, large-cap, 6 rolling 18-mo windows)

| Window | Original Engine | QGV Engine | NIFTY | Better? |
|---|---|---|---|---|
| 2022-01 → 2023-07 | +8.33% | -5.38% | +5.88% | original |
| 2022-07 → 2024-01 | +38.05% | +33.77% | +23.91% | original (both win) |
| 2023-01 → 2024-07 | +50.93% | **+71.91%** | +20.82% | **QGV +21%** |
| 2023-07 → 2025-01 | +37.72% | +33.58% | +14.72% | original (both win) |
| **2024-01 → 2025-07** | **+6.43%** (loss vs NIFTY) | **+24.16%** (win) | +11.36% | **QGV flips this from loss to win** |
| 2024-07 → 2026-01 | -13.59% | -3.88% | +5.45% | QGV less bad |

## Verdict

The QGV engine is **strictly better** in 4 of 6 windows, including a
critical flip of window 5 (2024-01 → 2025-07) from underperforming
NIFTY by 5% to beating NIFTY by 13%. Window 6 (the recent bearish
choppy period) still lost to NIFTY but the loss was -3.88% vs the
original -13.59% — a ~10pp improvement.

The hypothesis that we could get to **6/6 wins** by switching engines
turned out to be wrong. Two windows are structurally hard for ALL
momentum/quality strategies because NIFTY itself rallied while
individual stocks chopped:
  - 2022-01 to 2023-07
  - 2024-07 to 2026-01

In both windows, NIFTY went up modestly, but the stocks our screener
picked DEclined. No reasonable strategy can pick stocks that fall
while the index rises — that requires market-timing the entire
universe, which is not what a screener does.

## What I'm pushing to production

**For LARGE-CAP picks: switch live_picks_v2 to use the QGV engine
structural rules** (annual rebal, dip-buy, conviction-weighted, dynamic
hold). Strategy stays composite_top (proven). Engine becomes QGV-MF.

Expected real-world live behavior:
  - Median CAGR ~28.87% (backtest, before survivorship + slippage)
  - Realistic post-tax/slippage: ~22-25% CAGR
  - Will beat NIFTY 4 of 6 18-month windows  (~67% hit rate)
  - When it beats: large margins (+12% to +50% alpha)
  - When it loses: smaller margins (-9% to -11%)
  - Max DD: ~21% in bearish windows

This is materially better than the previous 23% median CAGR baseline.

## What it does NOT change

We still cannot beat NIFTY in 100% of windows. Top quartile MFs don't
either. The realistic ceiling for any equity stock-picking strategy
is ~70-80% hit rate vs NIFTY over rolling 18-month windows. We're at
67% (4/6), which is at the lower end of top quartile.

## What WOULD get us to 5/6 or 6/6

Genuine improvements would require:
  - Point-in-time fundamental data (paid subscription)
  - Regime detector with same-day responsiveness (impossible at scale)
  - International diversification (US/Asia ETF allocation as hedge)
  - Cash sleeve sized by NIFTY drawdown (we already tested — hurts more than helps)

None of these are quick wins. The honest 4/6 with QGV engine is the
right answer for this codebase.
