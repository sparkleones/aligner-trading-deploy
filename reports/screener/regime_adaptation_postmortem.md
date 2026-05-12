# Regime-Adaptive Strategy Postmortem (2026-05-13)

## Goal
Make the screener beat NIFTY in all 6 rolling 18-month walk-forward windows
(originally 4/6). Three approaches were tested.

## Results

| Iteration | Strategy | Beats NIFTY | Median Alpha | Median CAGR |
|---|---|---|---|---|
| baseline  | Pure momentum (composite_top) | **4/6** | **+10.08%/yr** | **+24.81%** |
| v1 | Regime + 4-sleeve ensemble (mom/vol/mean-rev/cash) | 2/6 | -3.42% | +11.47% |
| v2 | Regime + tighter allocation (momentum dominant) | 3/6 | -0.09% | +14.48% |
| v3 | Regime + NIFTY ETF fallback (match index in choppy) | 2/6 | -5.08% | +11.05% |

## Why every adaptation hurt

The diagnostic on the 2 losing windows of pure momentum (2024-01 → 2025-07
and 2024-07 → 2026-01) found:
- 15% win rate (10/13 trades hit stage2_break and exited at the bottom)
- All sectors except CONSUMER (TRENT) lost money
- Average trade lost 9.22%

The natural fix: detect "broken-momentum" regimes and step back. BUT the
regime classifier itself is the problem:

1. **The regime detector lags.** By the time breadth drops below 40% or
   NIFTY crosses 200-DMA, the damage is already done. We exit at the bottom
   of the move and miss the recovery.

2. **The regime detector misclassifies trending markets.** In window 1
   (2022-01 → 2023-07), pure momentum returned +11.9% but regime-adaptive
   returned -2.1%. The classifier called it "CHOPPY" / "DEFENSIVE" too
   early and rotated out of momentum stocks that were going to keep
   trending.

3. **Adding NIFTY ETF in choppy regimes didn't help either.** In window 6
   (2024-07 → 2026-01), adaptive lost -8.71% vs pure momentum -13.59% —
   so the ETF DID cushion. But NIFTY itself was +5.45%. To MATCH NIFTY,
   I'd need to be in the ETF for the FULL 18 months — but my classifier
   keeps flipping between regimes, so I'm in/out of the ETF and miss the
   recovery legs.

## Senior PM Takeaway

The right answer is **NOT** to engineer a more complex strategy.

The right answer is to **accept the trade-off**: pure momentum has alpha
in trending markets (4/6 windows, +10%/yr median) and loses in choppy
markets. That's the same trade-off every top-quartile equity MF accepts.
Top quartile MFs beat NIFTY in roughly 4-5 of 6 rolling 18-month windows
historically. We're already at the top of that distribution.

The instinct to "fix" the losing windows almost always introduces more
problems than it solves. This is a classic case.

## What we keep

- Pure momentum (composite_top) as the production strategy
- The regime DETECTOR as an INFORMATIONAL panel in the dashboard
  (so user can see when we're in a known-difficult regime and choose
  to reduce position size MANUALLY)
- 4/6 windows beat NIFTY at +10%/yr median alpha
- Honest expectation: we will LOSE to NIFTY in roughly 1/3 of 18-month
  windows. That's the cost of momentum alpha.

## What we drop

- Regime-adaptive ensemble as a backtest target (saved as a research
  module for future reference but NOT used in live picks)
